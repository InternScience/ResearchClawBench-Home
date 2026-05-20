"""
Main experiment runner: Compare ByteTrack baseline vs SparseTrack.
Uses multiple baselines for comprehensive comparison.
"""
import json
import sys
import os
import time
import numpy as np
from copy import deepcopy

sys.path.insert(0, '.')
from code.bytetrack import ByteTrackBaseline
from code.sparsetrack import SparseTrack
from code.kalman_filter import KalmanFilter, compute_iou_matrix, hungarian_match
from code.evaluation import compute_mot_metrics

# Load data
with open('data/simulated_sequence.json') as f:
    data = json.load(f)

print(f"Loaded {len(data)} frames with {len(data[0]['gt_bboxes'])} objects per frame")

# Build gt_data dict
gt_data = {}
for frame in data:
    frame_idx = frame['frame']
    gt_data[frame_idx] = {
        'gt_bboxes': frame['gt_bboxes'],
        'gt_ids': frame['gt_ids'],
        'detections': frame['detections']
    }


# ============================================================
# Baseline 1: Simple SORT-like tracker (all detections, single-stage)
# ============================================================
class SimpleTracker:
    """Simple SORT-like tracker using all detections with single-stage IoU association."""
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        
    def process_frame(self, detections, frame_idx):
        pred_boxes = []
        for track in self.tracks:
            track['kf'].predict()
            pred_boxes.append(track['kf'].get_state())
        
        det_boxes = [d['bbox'] for d in detections]
        
        if pred_boxes and det_boxes:
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            cost = 1.0 - iou_mat
            matched, unmatched_tracks, unmatched_dets = hungarian_match(
                cost, threshold=1.0 - self.iou_thresh
            )
            
            for t_idx, d_idx in matched:
                self.tracks[t_idx]['kf'].update(detections[d_idx]['bbox'])
                self.tracks[t_idx]['time_since_update'] = 0
                self.tracks[t_idx]['hits'] += 1
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        
        # Mark unmatched
        for t_idx in unmatched_tracks:
            self.tracks[t_idx]['time_since_update'] += 1
        
        # Remove lost
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        # Create new
        for d_idx in unmatched_dets:
            kf = KalmanFilter()
            kf.initiate(detections[d_idx]['bbox'])
            self.tracks.append({
                'id': self.next_id, 'kf': kf, 'time_since_update': 0, 'hits': 1
            })
            self.next_id += 1
        
        results = {}
        for track in self.tracks:
            if track['time_since_update'] == 0:
                results[track['id']] = track['kf'].get_state()
        return results


# ============================================================
# Baseline 2: ByteTrack (proper implementation)
# ============================================================
class ByteTrackFixed:
    """
    ByteTrack with proper track creation from low-score unmatched detections.
    Reference: Zhang et al.
    """
    def __init__(self, score_thresh=0.5, iou_thresh=0.3, max_age=30):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        
    def process_frame(self, detections, frame_idx):
        # Split
        d_high = [d for d in detections if d['score'] >= self.score_thresh]
        d_low = [d for d in detections if d['score'] < self.score_thresh]
        
        # Predict
        pred_boxes = []
        for track in self.tracks:
            track['kf'].predict()
            pred_boxes.append(track['kf'].get_state())
        
        matched_all = []
        
        # 1st association: all tracks vs high-score
        if pred_boxes and d_high:
            det_boxes_h = [d['bbox'] for d in d_high]
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes_h)
            cost_h = 1.0 - iou_mat
            matched_h, unmatched_tracks, unmatched_dets_h = hungarian_match(
                cost_h, threshold=1.0 - self.iou_thresh
            )
            for t_idx, d_idx in matched_h:
                self.tracks[t_idx]['kf'].update(d_high[d_idx]['bbox'])
                self.tracks[t_idx]['time_since_update'] = 0
                self.tracks[t_idx]['hits'] += 1
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets_h = list(range(len(d_high)))
        
        # 2nd association: remaining tracks vs low-score
        remaining_pred = [pred_boxes[i] for i in unmatched_tracks]
        remaining_track_indices = [i for i in unmatched_tracks]
        
        if remaining_pred and d_low:
            det_boxes_l = [d['bbox'] for d in d_low]
            iou_mat_l = compute_iou_matrix(remaining_pred, det_boxes_l)
            cost_l = 1.0 - iou_mat_l
            matched_l, ut_l, ud_l = hungarian_match(cost_l, threshold=1.0 - self.iou_thresh)
            
            for t_local, d_local in matched_l:
                t_global = remaining_track_indices[t_local]
                self.tracks[t_global]['kf'].update(d_low[d_local]['bbox'])
                self.tracks[t_global]['time_since_update'] = 0
                self.tracks[t_global]['hits'] += 1
            
            # Mark unmatched remaining tracks
            matched_remaining = set(t_local for t_local, _ in matched_l)
            for t_local_idx in range(len(remaining_track_indices)):
                if t_local_idx not in matched_remaining:
                    self.tracks[remaining_track_indices[t_local_idx]]['time_since_update'] += 1
        else:
            # All remaining tracks unmatched
            for t_idx in remaining_track_indices:
                self.tracks[t_idx]['time_since_update'] += 1
        
        # Remove lost
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        # Create new tracks from unmatched HIGH-score detections
        for d_idx in unmatched_dets_h:
            kf = KalmanFilter()
            kf.initiate(d_high[d_idx]['bbox'])
            self.tracks.append({
                'id': self.next_id, 'kf': kf, 'time_since_update': 0, 'hits': 1
            })
            self.next_id += 1
        
        results = {}
        for track in self.tracks:
            if track['time_since_update'] == 0:
                results[track['id']] = track['kf'].get_state()
        return results


# ============================================================
# Run all trackers
# ============================================================
trackers = {
    'SORT': SimpleTracker(iou_thresh=0.3, max_age=30),
    'ByteTrack': ByteTrackFixed(score_thresh=0.5, iou_thresh=0.3, max_age=30),
    'SparseTrack': SparseTrack(score_thresh=0.5, iou_thresh=0.3, max_age=30, n_depth_layers=5)
}

all_metrics = {}
all_timings = {}
all_n_tracks = {}

for name, tracker in trackers.items():
    print(f"\n=== Running {name} ===")
    results = {}
    start_time = time.time()
    
    for frame in data:
        frame_idx = frame['frame']
        result = tracker.process_frame(frame['detections'], frame_idx)
        results[frame_idx] = result
    
    elapsed = time.time() - start_time
    n_tracks = tracker.next_id - 1 if hasattr(tracker, 'next_id') else len(tracker.tracks)
    
    metrics = compute_mot_metrics(results, gt_data)
    
    all_metrics[name] = metrics
    all_timings[name] = elapsed
    all_n_tracks[name] = n_tracks
    
    print(f"  Time: {elapsed:.2f}s, Tracks: {n_tracks}")
    print(f"  MOTA: {metrics['MOTA']:.2f}%, MOTP: {metrics['MOTP']:.2f}%, IDF1: {metrics['IDF1']:.2f}%")
    print(f"  IDsw: {metrics['IDsw']}, MT: {metrics['MT']}, ML: {metrics['ML']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

# ============================================================
# Ablation: Number of depth layers for SparseTrack
# ============================================================
print("\n=== Ablation: Number of Depth Layers ===")
n_layers_list = [1, 2, 3, 5, 7, 10]
ablation_layers = []

for n_layers in n_layers_list:
    tracker_abl = SparseTrack(score_thresh=0.5, iou_thresh=0.3, max_age=30, n_depth_layers=n_layers)
    abl_results = {}
    for frame in data:
        result = tracker_abl.process_frame(frame['detections'], frame['frame'])
        abl_results[frame['frame']] = result
    
    metrics = compute_mot_metrics(abl_results, gt_data)
    ablation_layers.append({
        'n_layers': n_layers,
        'MOTA': metrics['MOTA'],
        'MOTP': metrics['MOTP'],
        'IDF1': metrics['IDF1'],
        'IDsw': metrics['IDsw'],
        'MT': metrics['MT'],
        'ML': metrics['ML']
    })
    print(f"  Layers={n_layers}: MOTA={metrics['MOTA']:.2f}%, IDF1={metrics['IDF1']:.2f}%, IDsw={metrics['IDsw']}")

# ============================================================
# Ablation: IoU threshold
# ============================================================
print("\n=== Ablation: IoU Threshold ===")
iou_thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ablation_iou = []

for iou_t in iou_thresh_list:
    bt_met = None
    st_met = None
    
    for name, TrackerClass in [('ByteTrack', lambda: ByteTrackFixed(score_thresh=0.5, iou_thresh=iou_t)),
                                 ('SparseTrack', lambda: SparseTrack(score_thresh=0.5, iou_thresh=iou_t, n_depth_layers=5))]:
        tracker_abl = TrackerClass()
        abl_results = {}
        for frame in data:
            result = tracker_abl.process_frame(frame['detections'], frame['frame'])
            abl_results[frame['frame']] = result
        met = compute_mot_metrics(abl_results, gt_data)
        if name == 'ByteTrack':
            bt_met = met
        else:
            st_met = met
    
    ablation_iou.append({
        'iou_thresh': iou_t,
        'bt_MOTA': bt_met['MOTA'], 'bt_IDF1': bt_met['IDF1'], 'bt_IDsw': bt_met['IDsw'],
        'st_MOTA': st_met['MOTA'], 'st_IDF1': st_met['IDF1'], 'st_IDsw': st_met['IDsw']
    })
    print(f"  IoU={iou_t}: BT MOTA={bt_met['MOTA']:.2f}% IDF1={bt_met['IDF1']:.2f}% | "
          f"ST MOTA={st_met['MOTA']:.2f}% IDF1={st_met['IDF1']:.2f}%")

# ============================================================
# Save results
# ============================================================
output = {
    'metrics': {k: v for k, v in all_metrics.items()},
    'timings': all_timings,
    'n_tracks': all_n_tracks,
    'ablation_layers': ablation_layers,
    'ablation_iou': ablation_iou
}

with open('outputs/experiment_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*60)
print("FINAL COMPARISON TABLE")
print("="*60)
print(f"{'Method':<15} {'MOTA%':>8} {'MOTP%':>8} {'IDF1%':>8} {'IDsw':>8} {'MT':>6} {'ML':>6} {'FP':>8} {'FN':>8}")
print("-"*80)
for name in ['SORT', 'ByteTrack', 'SparseTrack']:
    m = all_metrics[name]
    print(f"{name:<15} {m['MOTA']:>8.2f} {m['MOTP']:>8.2f} {m['IDF1']:>8.2f} {m['IDsw']:>8d} {m['MT']:>6d} {m['ML']:>6d} {m['FP']:>8d} {m['FN']:>8d}")
