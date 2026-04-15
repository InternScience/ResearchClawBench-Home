"""
Fast Multi-Object Tracking Analysis
====================================

Optimized version for large-scale tracking analysis.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# Setup paths
workspace = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260415_154537')
data_path = workspace / 'data' / 'simulated_sequence.json'
output_dir = workspace / 'outputs'
report_img_dir = workspace / 'report' / 'images'

output_dir.mkdir(exist_ok=True)
report_img_dir.mkdir(exist_ok=True, parents=True)


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


def compute_pseudo_depth(bbox, frame_shape=(640, 640)):
    """Compute pseudo-depth from bounding box."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    bottom_y = y2
    normalized_area = area / (frame_shape[0] * frame_shape[1])
    normalized_y = bottom_y / frame_shape[0]
    depth = 1.0 / (normalized_area + 0.01) + normalized_y * 0.5
    return depth


class SimpleTracker:
    """Base tracker with common functionality."""
    
    def __init__(self):
        self.tracks = {}  # track_id -> {bbox, velocity, age, hits}
        self.next_id = 0
        self.frame_count = 0
    
    def predict_tracks(self):
        """Simple constant velocity prediction."""
        for tid in self.tracks:
            track = self.tracks[tid]
            if 'velocity' in track:
                cx = (track['bbox'][0] + track['bbox'][2]) / 2
                cy = (track['bbox'][1] + track['bbox'][3]) / 2
                w = track['bbox'][2] - track['bbox'][0]
                h = track['bbox'][3] - track['bbox'][1]
                
                cx += track['velocity'][0]
                cy += track['velocity'][1]
                
                track['bbox'] = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            track['age'] += 1
            track['time_since_update'] += 1


class SORTTracker(SimpleTracker):
    """SORT: Simple Online and Realtime Tracking."""
    
    def __init__(self, max_age=1, iou_threshold=0.3):
        super().__init__()
        self.max_age = max_age
        self.iou_threshold = iou_threshold
    
    def update(self, detections):
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Predict existing tracks
        self.predict_tracks()
        
        # Get detection boxes
        det_boxes = [d['bbox'] for d in detections]
        det_scores = [d.get('score', 1.0) for d in detections]
        
        # Associate
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        matched = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        
        if len(track_boxes) > 0 and len(det_boxes) > 0:
            iou_matrix = np.zeros((len(track_boxes), len(det_boxes)))
            for i, tbox in enumerate(track_boxes):
                for j, dbox in enumerate(det_boxes):
                    iou_matrix[i, j] = compute_iou(tbox, dbox)
            
            cost_matrix = 1 - iou_matrix
            t_indices, d_indices = linear_sum_assignment(cost_matrix)
            
            for ti, di in zip(t_indices, d_indices):
                if iou_matrix[ti, di] >= self.iou_threshold:
                    tid = track_ids[ti]
                    matched.append((tid, di))
                    unmatched_tracks.discard(tid)
                    unmatched_dets.discard(di)
        
        # Update matched tracks
        for tid, di in matched:
            old_bbox = self.tracks[tid]['bbox']
            new_bbox = det_boxes[di]
            
            # Update velocity
            old_cx = (old_bbox[0] + old_bbox[2]) / 2
            old_cy = (old_bbox[1] + old_bbox[3]) / 2
            new_cx = (new_bbox[0] + new_bbox[2]) / 2
            new_cy = (new_bbox[1] + new_bbox[3]) / 2
            
            self.tracks[tid]['bbox'] = new_bbox
            self.tracks[tid]['velocity'] = (new_cx - old_cx, new_cy - old_cy)
            self.tracks[tid]['time_since_update'] = 0
            self.tracks[tid]['hits'] += 1
            self.tracks[tid]['score'] = det_scores[di]
        
        # Remove old tracks
        for tid in list(unmatched_tracks):
            if self.tracks[tid]['time_since_update'] > self.max_age:
                del self.tracks[tid]
                unmatched_tracks.discard(tid)
        
        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox': det_boxes[di],
                'velocity': (0, 0),
                'age': 1,
                'hits': 1,
                'time_since_update': 0,
                'score': det_scores[di]
            }
        
        # Return confirmed tracks
        results = []
        for tid, track in self.tracks.items():
            if track['hits'] >= 3 or track['time_since_update'] == 0:
                results.append((tid, track['bbox']))
        return results


class ByteTracker(SimpleTracker):
    """ByteTrack: Two-stage association."""
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, second_match_thresh=0.5):
        super().__init__()
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.lost_tracks = {}
        self.lost_id = 0
    
    def update(self, detections):
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Separate high and low score detections
        high_dets = []
        low_dets = []
        for i, d in enumerate(detections):
            if d['score'] >= self.track_thresh:
                high_dets.append((i, d))
            else:
                low_dets.append((i, d))
        
        # Predict all tracks
        self.predict_tracks()
        for tid in self.lost_tracks:
            self.lost_tracks[tid]['age'] += 1
            self.lost_tracks[tid]['time_since_update'] += 1
        
        # Combine tracks and lost tracks
        all_tracks = {**self.tracks, **self.lost_tracks}
        
        # First association: high score detections
        track_ids = list(all_tracks.keys())
        track_boxes = [all_tracks[tid]['bbox'] for tid in track_ids]
        high_boxes = [d['bbox'] for _, d in high_dets]
        
        matched_track_ids = set()
        matched_high_indices = set()
        
        if len(track_boxes) > 0 and len(high_boxes) > 0:
            iou_matrix = np.zeros((len(track_boxes), len(high_boxes)))
            for i, tbox in enumerate(track_boxes):
                for j, dbox in enumerate(high_boxes):
                    iou_matrix[i, j] = compute_iou(tbox, dbox)
            
            cost_matrix = 1 - iou_matrix
            t_indices, d_indices = linear_sum_assignment(cost_matrix)
            
            for ti, di in zip(t_indices, d_indices):
                if iou_matrix[ti, di] >= self.match_thresh:
                    tid = track_ids[ti]
                    orig_idx, det = high_dets[di]
                    
                    # Update track
                    if tid not in self.tracks:
                        if tid in self.lost_tracks:
                            self.tracks[tid] = self.lost_tracks.pop(tid)
                    
                    old_bbox = self.tracks[tid]['bbox']
                    new_bbox = det['bbox']
                    old_cx = (old_bbox[0] + old_bbox[2]) / 2
                    old_cy = (old_bbox[1] + old_bbox[3]) / 2
                    new_cx = (new_bbox[0] + new_bbox[2]) / 2
                    new_cy = (new_bbox[1] + new_bbox[3]) / 2
                    
                    self.tracks[tid]['bbox'] = new_bbox
                    self.tracks[tid]['velocity'] = (new_cx - old_cx, new_cy - old_cy)
                    self.tracks[tid]['time_since_update'] = 0
                    self.tracks[tid]['hits'] += 1
                    self.tracks[tid]['score'] = det['score']
                    
                    matched_track_ids.add(tid)
                    matched_high_indices.add(di)
        
        # Second association: unmatched tracks with low score
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_track_ids]
        low_boxes = [d['bbox'] for _, d in low_dets]
        
        if len(unmatched_track_ids) > 0 and len(low_boxes) > 0:
            unmatched_boxes = [all_tracks[tid]['bbox'] for tid in unmatched_track_ids]
            iou_matrix = np.zeros((len(unmatched_boxes), len(low_boxes)))
            for i, tbox in enumerate(unmatched_boxes):
                for j, dbox in enumerate(low_boxes):
                    iou_matrix[i, j] = compute_iou(tbox, dbox)
            
            cost_matrix = 1 - iou_matrix
            t_indices, d_indices = linear_sum_assignment(cost_matrix)
            
            for ti, di in zip(t_indices, d_indices):
                if iou_matrix[ti, di] >= self.second_match_thresh:
                    tid = unmatched_track_ids[ti]
                    orig_idx, det = low_dets[di]
                    
                    if tid not in self.tracks:
                        if tid in self.lost_tracks:
                            self.tracks[tid] = self.lost_tracks.pop(tid)
                    
                    old_bbox = self.tracks[tid]['bbox']
                    new_bbox = det['bbox']
                    old_cx = (old_bbox[0] + old_bbox[2]) / 2
                    old_cy = (old_bbox[1] + old_bbox[3]) / 2
                    new_cx = (new_bbox[0] + new_bbox[2]) / 2
                    new_cy = (new_bbox[1] + new_bbox[3]) / 2
                    
                    self.tracks[tid]['bbox'] = new_bbox
                    self.tracks[tid]['velocity'] = (new_cx - old_cx, new_cy - old_cy)
                    self.tracks[tid]['time_since_update'] = 0
                    self.tracks[tid]['hits'] += 1
                    self.tracks[tid]['score'] = det['score']
                    
                    matched_track_ids.add(tid)
        
        # Move unmatched tracks to lost
        for tid in track_ids:
            if tid not in matched_track_ids:
                if tid in self.tracks:
                    track = self.tracks.pop(tid)
                    self.lost_tracks[tid] = track
        
        # Create new tracks for unmatched high score detections
        for i, det in high_dets:
            if i not in matched_high_indices:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': det['bbox'],
                    'velocity': (0, 0),
                    'age': 1,
                    'hits': 1,
                    'time_since_update': 0,
                    'score': det['score']
                }
        
        # Remove old lost tracks
        self.lost_tracks = {tid: t for tid, t in self.lost_tracks.items() 
                           if t['time_since_update'] < 30}
        
        # Return results
        results = []
        for tid, track in self.tracks.items():
            if track['hits'] >= 3 or track['time_since_update'] == 0:
                results.append((tid, track['bbox']))
        for tid, track in self.lost_tracks.items():
            if track['time_since_update'] <= 1:
                results.append((tid, track['bbox']))
        return results


class SparseTracker(SimpleTracker):
    """SparseTrack: Depth-based hierarchical tracking."""
    
    def __init__(self, n_levels=3, match_thresh=0.7):
        super().__init__()
        self.n_levels = n_levels
        self.match_thresh = match_thresh
        self.tracks_by_level = [{} for _ in range(n_levels)]
        self.next_ids = [0] * n_levels
    
    def update(self, detections):
        """Update tracker with depth-based hierarchical association."""
        self.frame_count += 1
        
        if len(detections) == 0:
            return []
        
        # Compute depths
        det_boxes = [d['bbox'] for d in detections]
        det_scores = [d['score'] for d in detections]
        depths = [compute_pseudo_depth(box) for box in det_boxes]
        
        # Sort by depth and divide into levels
        sorted_indices = np.argsort(depths)
        level_size = len(sorted_indices) // self.n_levels
        
        results = []
        
        for level in range(self.n_levels):
            # Get detections for this level
            start = level * level_size
            end = len(sorted_indices) if level == self.n_levels - 1 else (level + 1) * level_size
            level_indices = sorted_indices[start:end]
            
            if len(level_indices) == 0:
                continue
            
            level_dets = [(i, detections[i]) for i in level_indices]
            
            # Predict existing tracks at this level
            for tid in self.tracks_by_level[level]:
                track = self.tracks_by_level[level][tid]
                if 'velocity' in track:
                    cx = (track['bbox'][0] + track['bbox'][2]) / 2
                    cy = (track['bbox'][1] + track['bbox'][3]) / 2
                    w = track['bbox'][2] - track['bbox'][0]
                    h = track['bbox'][3] - track['bbox'][1]
                    cx += track['velocity'][0]
                    cy += track['velocity'][1]
                    track['bbox'] = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                track['age'] += 1
                track['time_since_update'] += 1
            
            # Associate
            track_ids = list(self.tracks_by_level[level].keys())
            track_boxes = [self.tracks_by_level[level][tid]['bbox'] for tid in track_ids]
            det_boxes_level = [d['bbox'] for _, d in level_dets]
            
            matched = []
            unmatched_tracks = set(track_ids)
            unmatched_dets = set(range(len(level_dets)))
            
            if len(track_boxes) > 0 and len(det_boxes_level) > 0:
                iou_matrix = np.zeros((len(track_boxes), len(det_boxes_level)))
                for i, tbox in enumerate(track_boxes):
                    for j, dbox in enumerate(det_boxes_level):
                        iou_matrix[i, j] = compute_iou(tbox, dbox)
                
                cost_matrix = 1 - iou_matrix
                t_indices, d_indices = linear_sum_assignment(cost_matrix)
                
                for ti, di in zip(t_indices, d_indices):
                    if iou_matrix[ti, di] >= self.match_thresh:
                        tid = track_ids[ti]
                        orig_idx, det = level_dets[di]
                        
                        old_bbox = self.tracks_by_level[level][tid]['bbox']
                        new_bbox = det['bbox']
                        old_cx = (old_bbox[0] + old_bbox[2]) / 2
                        old_cy = (old_bbox[1] + old_bbox[3]) / 2
                        new_cx = (new_bbox[0] + new_bbox[2]) / 2
                        new_cy = (new_bbox[1] + new_bbox[3]) / 2
                        
                        self.tracks_by_level[level][tid]['bbox'] = new_bbox
                        self.tracks_by_level[level][tid]['velocity'] = (new_cx - old_cx, new_cy - old_cy)
                        self.tracks_by_level[level][tid]['time_since_update'] = 0
                        self.tracks_by_level[level][tid]['hits'] += 1
                        self.tracks_by_level[level][tid]['score'] = det['score']
                        
                        matched.append((tid, di))
                        unmatched_tracks.discard(tid)
                        unmatched_dets.discard(di)
            
            # Remove old tracks
            for tid in list(unmatched_tracks):
                if self.tracks_by_level[level][tid]['time_since_update'] > 1:
                    del self.tracks_by_level[level][tid]
                    unmatched_tracks.discard(tid)
            
            # Create new tracks for unmatched detections
            for di in unmatched_dets:
                orig_idx, det = level_dets[di]
                tid = self.next_ids[level]
                self.next_ids[level] += 1
                self.tracks_by_level[level][tid] = {
                    'bbox': det['bbox'],
                    'velocity': (0, 0),
                    'age': 1,
                    'hits': 1,
                    'time_since_update': 0,
                    'score': det['score'],
                    'level': level
                }
            
            # Collect results from this level
            for tid, track in self.tracks_by_level[level].items():
                if track['hits'] >= 3 or track['time_since_update'] == 0:
                    results.append((tid + level * 100000, track['bbox']))  # Offset IDs by level
        
        return results


def evaluate(ground_truth, predictions, iou_threshold=0.5):
    """Evaluate tracking performance."""
    total_gt = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    
    prev_mapping = {}
    
    all_frames = sorted(set(list(ground_truth.keys()) + list(predictions.keys())))
    
    for frame_id in all_frames:
        gt_items = ground_truth.get(frame_id, [])
        pred_items = predictions.get(frame_id, [])
        
        gt_ids = [g[0] for g in gt_items]
        gt_boxes = [g[1] for g in gt_items]
        pred_ids = [p[0] for p in pred_items]
        pred_boxes = [p[1] for p in pred_items]
        
        total_gt += len(gt_items)
        
        # Match
        matched_pairs = []
        matched_gt = set()
        matched_pred = set()
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou_matrix[i, j] = compute_iou(gt_box, pred_box)
            
            # Greedy matching
            while True:
                if iou_matrix.size == 0 or iou_matrix.max() < iou_threshold:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[i, j] >= iou_threshold:
                    matched_pairs.append((i, j))
                    matched_gt.add(i)
                    matched_pred.add(j)
                    iou_matrix[i, :] = -1
                    iou_matrix[:, j] = -1
                else:
                    break
        
        # Check ID switches
        current_mapping = {}
        for gi, pi in matched_pairs:
            gt_id = gt_ids[gi]
            track_id = pred_ids[pi]
            current_mapping[gt_id] = track_id
            
            if gt_id in prev_mapping and prev_mapping[gt_id] != track_id:
                id_switches += 1
            
            total_matches += 1
        
        total_fn += len(gt_items) - len(matched_gt)
        total_fp += len(pred_items) - len(matched_pred)
        
        prev_mapping = current_mapping
    
    # Compute metrics
    mota = (1 - (total_fn + total_fp + id_switches) / total_gt * 100) if total_gt > 0 else 0
    precision = total_matches / (total_matches + total_fp) * 100 if (total_matches + total_fp) > 0 else 0
    recall = total_matches / total_gt * 100 if total_gt > 0 else 0
    idf1 = total_matches / (total_matches + 0.5 * (total_fp + total_fn)) * 100 if total_matches > 0 else 0
    
    return {
        'MOTA': mota,
        'IDF1': idf1,
        'Precision': precision,
        'Recall': recall,
        'ID_Switches': id_switches,
        'FP': total_fp,
        'FN': total_fn,
        'GT_Total': total_gt,
        'Matches': total_matches
    }


def main():
    print("=" * 60)
    print("Multi-Object Tracking Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} frames")
    
    # Prepare ground truth
    ground_truth = {}
    detections_by_frame = {}
    
    for frame_data in data:
        frame_id = frame_data['frame']
        gt_bboxes = [np.array(b) for b in frame_data['gt_bboxes']]
        gt_ids = frame_data['gt_ids']
        detections = frame_data['detections']
        
        ground_truth[frame_id] = list(zip(gt_ids, gt_bboxes))
        detections_by_frame[frame_id] = [
            {'bbox': np.array(d['bbox']), 'score': d['score'], 'gt_id': d.get('gt_id', -1)}
            for d in detections
        ]
    
    # Data overview
    print("\nGenerating data overview...")
    frame_stats = []
    all_scores = []
    all_depths = []
    
    for frame_data in data:
        n_gt = len(frame_data['gt_bboxes'])
        n_det = len(frame_data['detections'])
        frame_stats.append({
            'frame': frame_data['frame'],
            'n_gt': n_gt,
            'n_det': n_det,
            'det_rate': n_det / n_gt if n_gt > 0 else 0
        })
        all_scores.extend([d['score'] for d in frame_data['detections']])
        all_depths.extend([compute_pseudo_depth(np.array(b)) for b in frame_data['gt_bboxes']])
    
    # Plot data overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    frames = [s['frame'] for s in frame_stats]
    n_gts = [s['n_gt'] for s in frame_stats]
    n_dets = [s['n_det'] for s in frame_stats]
    det_rates = [s['det_rate'] for s in frame_stats]
    
    axes[0, 0].plot(frames, n_gts, 'b-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot(frames, n_dets, 'r--', label='Detections', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Objects per Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(frames, det_rates, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Detection Rate')
    axes[0, 1].set_title('Detection Rate Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(det_rates), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(det_rates):.2%}')
    axes[0, 1].legend()
    
    axes[1, 0].hist(all_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Detection Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Detection Score Distribution')
    axes[1, 0].axvline(x=np.mean(all_scores), color='r', linestyle='--',
                       label=f'Mean: {np.mean(all_scores):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(all_depths, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Pseudo-Depth')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Pseudo-Depth Distribution')
    axes[1, 1].axvline(x=np.mean(all_depths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(all_depths):.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_img_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save stats
    stats = {
        'total_frames': len(data),
        'total_gt_objects': sum(len(d['gt_bboxes']) for d in data),
        'total_detections': sum(len(d['detections']) for d in data),
        'mean_detection_rate': float(np.mean(det_rates)),
        'mean_detection_score': float(np.mean(all_scores)),
        'mean_pseudo_depth': float(np.mean(all_depths))
    }
    with open(output_dir / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Data statistics: {stats}")
    
    # Run trackers
    print("\n" + "=" * 60)
    print("Running Trackers")
    print("=" * 60)
    
    trackers = {
        'SORT': SORTTracker(max_age=1, iou_threshold=0.3),
        'ByteTrack': ByteTracker(track_thresh=0.5, match_thresh=0.8, second_match_thresh=0.5),
        'SparseTrack': SparseTracker(n_levels=3, match_thresh=0.7)
    }
    
    predictions = {}
    
    for name, tracker in trackers.items():
        print(f"\nRunning {name}...")
        preds = {}
        for frame_id in sorted(detections_by_frame.keys()):
            dets = detections_by_frame[frame_id]
            results = tracker.update(dets)
            preds[frame_id] = results
        predictions[name] = preds
        print(f"  Completed {len(preds)} frames")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating Trackers")
    print("=" * 60)
    
    results = {}
    for name, preds in predictions.items():
        metrics = evaluate(ground_truth, preds, iou_threshold=0.5)
        results[name] = metrics
        
        print(f"\n{name}:")
        print(f"  MOTA: {metrics['MOTA']:.2f}%")
        print(f"  IDF1: {metrics['IDF1']:.2f}%")
        print(f"  Precision: {metrics['Precision']:.2f}%")
        print(f"  Recall: {metrics['Recall']:.2f}%")
        print(f"  ID Switches: {metrics['ID_Switches']}")
        print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    # Save results
    with open(output_dir / 'tracking_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    tracker_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(['MOTA', 'IDF1', 'Precision', 'Recall']):
        values = [results[t][metric] for t in tracker_names]
        bars = axes[idx].bar(tracker_names, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(report_img_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ID Switches comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    id_switches = [results[t]['ID_Switches'] for t in tracker_names]
    bars = ax.bar(tracker_names, id_switches, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of ID Switches')
    ax.set_title('ID Switches Comparison (Lower is Better)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, id_switches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(report_img_dir / 'id_switches_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metrics table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Tracker', 'MOTA', 'IDF1', 'Precision', 'Recall', 'ID Switches', 'FP', 'FN']
    table_data = []
    
    for name in tracker_names:
        r = results[name]
        table_data.append([
            name,
            f"{r['MOTA']:.2f}",
            f"{r['IDF1']:.2f}",
            f"{r['Precision']:.2f}",
            f"{r['Recall']:.2f}",
            str(r['ID_Switches']),
            str(r['FP']),
            str(r['FN'])
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15] + [0.12] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Multi-Object Tracking Performance Comparison', fontsize=14, pad=20)
    plt.savefig(report_img_dir / 'metrics_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Depth analysis
    print("Generating depth analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sample_frames = [0, 20, 40, 60, 80, 99]
    
    for idx, frame_idx in enumerate(sample_frames):
        ax = axes[idx // 3, idx % 3]
        frame_data = data[frame_idx]
        
        depths = [compute_pseudo_depth(np.array(bbox)) 
                  for bbox in frame_data['gt_bboxes']]
        
        for bbox, depth in zip(frame_data['gt_bboxes'], depths):
            x1, y1, x2, y2 = bbox
            color = plt.cm.viridis((depth - min(depths)) / (max(depths) - min(depths) + 1e-6))
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(rect)
        
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame_idx}')
        ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(all_depths), vmax=max(all_depths)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Pseudo-Depth (Smaller = Closer)')
    
    plt.suptitle('SparseTrack: Depth-Based Object Clustering', fontsize=14)
    plt.tight_layout()
    plt.savefig(report_img_dir / 'depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Trajectory samples
    for tracker_name, preds in predictions.items():
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, frame_idx in enumerate(range(0, min(len(data), 100), 5)):
            if idx >= 20:
                break
            
            ax = axes[idx]
            frame_data = data[frame_idx]
            frame_id = frame_data['frame']
            
            for bbox in frame_data['gt_bboxes']:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    fill=False, edgecolor='blue', linewidth=1.5, alpha=0.5)
                ax.add_patch(rect)
            
            if frame_id in preds:
                for track_id, bbox in preds[frame_id][:20]:  # Limit to 20 tracks for clarity
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='red', linewidth=1.5, linestyle='--')
                    ax.add_patch(rect)
            
            ax.set_xlim(0, 640)
            ax.set_ylim(640, 0)
            ax.set_aspect('equal')
            ax.set_title(f'Frame {frame_id}')
            ax.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='blue', label='Ground Truth'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Tracked')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle(f'{tracker_name} - Sample Tracking Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(report_img_dir / f'trajectories_{tracker_name.lower()}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {report_img_dir}")
    
    return results


if __name__ == '__main__':
    main()
