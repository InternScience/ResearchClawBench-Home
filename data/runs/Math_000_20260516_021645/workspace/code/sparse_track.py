#!/usr/bin/env python3
"""
SparseTrack-style Multi-Object Tracking on simulated_sequence.json
- Baseline: ByteTrack-like greedy association
- Proposed: Pseudo-depth + hierarchical association (SparseTrack)
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# -----------------------------
# Data Loading
# -----------------------------
def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data  # list of 100 frames

# -----------------------------
# IoU and Association
# -----------------------------
def iou(bb_test, bb_gt):
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    area_test = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area_gt = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    return wh / (area_test + area_gt - wh + 1e-6)

def linear_assignment(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

# -----------------------------
# Baseline ByteTrack-like Tracker
# -----------------------------
class ByteTracker:
    def __init__(self, iou_thresh=0.5):
        self.tracks = {}  # id -> {'bbox': [...], 'age': 0, 'hits': 0}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.results = []

    def update(self, detections, frame_id):
        """detections: list of {'bbox': [x1,y1,x2,y2], 'score': float}"""
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {'bbox': det['bbox'], 'age': 0, 'hits': 1}
                self.results.append({'frame': frame_id, 'id': self.next_id, 'bbox': det['bbox']})
                self.next_id += 1
            return

        # Association
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        det_bboxes = [d['bbox'] for d in detections]

        cost = np.zeros((len(track_ids), len(det_bboxes)))
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                cost[i, j] = 1 - iou(tb, db)

        matches = []
        if cost.size > 0:
            matches = linear_assignment(cost)
            matches = [m for m in matches if cost[m[0], m[1]] < (1 - self.iou_thresh)]

        matched_tracks = set()
        matched_dets = set()
        for ti, di in matches:
            tid = track_ids[ti]
            self.tracks[tid]['bbox'] = detections[di]['bbox']
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1
            self.results.append({'frame': frame_id, 'id': tid, 'bbox': detections[di]['bbox']})
            matched_tracks.add(ti)
            matched_dets.add(di)

        # New tracks
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks[self.next_id] = {'bbox': det['bbox'], 'age': 0, 'hits': 1}
                self.results.append({'frame': frame_id, 'id': self.next_id, 'bbox': det['bbox']})
                self.next_id += 1

        # Age unmatched tracks
        for ti in range(len(track_ids)):
            if ti not in matched_tracks:
                tid = track_ids[ti]
                self.tracks[tid]['age'] += 1

        # Remove old tracks
        to_remove = [tid for tid, t in self.tracks.items() if t['age'] > 5]
        for tid in to_remove:
            del self.tracks[tid]

# -----------------------------
# SparseTrack: Pseudo-depth + Hierarchical
# -----------------------------
class SparseTracker:
    def __init__(self, iou_thresh=0.5, depth_bins=3):
        self.tracks = {}
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.depth_bins = depth_bins
        self.results = []

    def estimate_depth(self, bbox):
        """Pseudo-depth: inverse of box height (closer = taller)"""
        h = bbox[3] - bbox[1]
        return 1.0 / (h + 1e-6)

    def update(self, detections, frame_id):
        if not self.tracks:
            for det in detections:
                depth = self.estimate_depth(det['bbox'])
                self.tracks[self.next_id] = {'bbox': det['bbox'], 'age': 0, 'hits': 1, 'depth': depth}
                self.results.append({'frame': frame_id, 'id': self.next_id, 'bbox': det['bbox']})
                self.next_id += 1
            return

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        det_bboxes = [d['bbox'] for d in detections]
        det_depths = [self.estimate_depth(d['bbox']) for d in detections]

        # Hierarchical: group by depth bin
        depth_min = min(det_depths + [t['depth'] for t in self.tracks.values()])
        depth_max = max(det_depths + [t['depth'] for t in self.tracks.values()])
        bin_edges = np.linspace(depth_min, depth_max, self.depth_bins + 1)

        def get_bin(d):
            return min(np.digitize(d, bin_edges) - 1, self.depth_bins - 1)

        # Process each depth bin separately
        for b in range(self.depth_bins):
            track_bin = [i for i, tid in enumerate(track_ids) if get_bin(self.tracks[tid]['depth']) == b]
            det_bin = [i for i, d in enumerate(det_depths) if get_bin(d) == b]

            if not track_bin or not det_bin:
                continue

            cost = np.zeros((len(track_bin), len(det_bin)))
            for ii, ti in enumerate(track_bin):
                for jj, di in enumerate(det_bin):
                    cost[ii, jj] = 1 - iou(track_bboxes[ti], det_bboxes[di])

            matches = []
            if cost.size > 0:
                matches = linear_assignment(cost)
                matches = [m for m in matches if cost[m[0], m[1]] < (1 - self.iou_thresh)]

            matched_tracks = set()
            matched_dets = set()
            for ii, jj in matches:
                ti = track_bin[ii]
                di = det_bin[jj]
                tid = track_ids[ti]
                self.tracks[tid]['bbox'] = detections[di]['bbox']
                self.tracks[tid]['age'] = 0
                self.tracks[tid]['hits'] += 1
                self.tracks[tid]['depth'] = det_depths[di]
                self.results.append({'frame': frame_id, 'id': tid, 'bbox': detections[di]['bbox']})
                matched_tracks.add(ti)
                matched_dets.add(di)

            # New tracks in bin
            for di in det_bin:
                if di not in matched_dets:
                    self.tracks[self.next_id] = {'bbox': detections[di]['bbox'], 'age': 0, 'hits': 1, 'depth': det_depths[di]}
                    self.results.append({'frame': frame_id, 'id': self.next_id, 'bbox': detections[di]['bbox']})
                    self.next_id += 1

            # Age
            for ti in track_bin:
                if ti not in matched_tracks:
                    tid = track_ids[ti]
                    self.tracks[tid]['age'] += 1

        # Global cleanup
        to_remove = [tid for tid, t in self.tracks.items() if t['age'] > 5]
        for tid in to_remove:
            del self.tracks[tid]

# -----------------------------
# Evaluation (MOTA, IDF1 simplified)
# -----------------------------
def evaluate(results, gt_data):
    """Simplified MOTA / IDF1 using gt_ids"""
    id_switches = 0
    gt_matched = 0
    gt_total = 0
    id_map = defaultdict(dict)  # frame -> gt_id -> track_id

    for res in results:
        f = res['frame']
        tid = res['id']
        # Find matching GT
        for gt in gt_data[f]['gt_bboxes']:
            # match by bbox IoU (simplified)
            if iou(res['bbox'], gt) > 0.5:
                gtid = gt_data[f]['gt_ids'][gt_data[f]['gt_bboxes'].index(gt)]
                if f in id_map and gtid in id_map[f]:
                    if id_map[f][gtid] != tid:
                        id_switches += 1
                id_map[f][gtid] = tid
                gt_matched += 1
                break
        gt_total += len(gt_data[f]['gt_bboxes'])

    mota = 1 - (id_switches / max(gt_total, 1))
    idf1 = gt_matched / max(gt_total, 1)
    return {'MOTA': mota, 'IDF1': idf1, 'ID_switches': id_switches}

# -----------------------------
# Main
# -----------------------------
def main():
    data_path = 'data/simulated_sequence.json'
    data = load_data(data_path)

    byte_tracker = ByteTracker()
    sparse_tracker = SparseTracker()

    for frame_data in data:
        frame_id = frame_data['frame']
        dets = frame_data['detections']
        byte_tracker.update(dets, frame_id)
        sparse_tracker.update(dets, frame_id)

    byte_metrics = evaluate(byte_tracker.results, data)
    sparse_metrics = evaluate(sparse_tracker.results, data)

    print("ByteTrack Metrics:", byte_metrics)
    print("SparseTrack Metrics:", sparse_metrics)

    # Save metrics
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/metrics.json', 'w') as f:
        json.dump({'ByteTrack': byte_metrics, 'SparseTrack': sparse_metrics}, f, indent=2)

    # Visualization: IDF1 comparison
    plt.figure(figsize=(6,4))
    methods = ['ByteTrack', 'SparseTrack']
    idf1s = [byte_metrics['IDF1'], sparse_metrics['IDF1']]
    sns.barplot(x=methods, y=idf1s, palette='viridis')
    plt.ylabel('IDF1')
    plt.title('Tracking Performance (IDF1)')
    plt.ylim(0, 1)
    for i, v in enumerate(idf1s):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig('report/images/idf1_comparison.png', dpi=150)
    plt.close()

    # Trajectory length distribution
    def get_lengths(results):
        lengths = defaultdict(int)
        for r in results:
            lengths[r['id']] += 1
        return list(lengths.values())

    byte_lengths = get_lengths(byte_tracker.results)
    sparse_lengths = get_lengths(sparse_tracker.results)

    plt.figure(figsize=(8,4))
    sns.histplot(byte_lengths, bins=20, label='ByteTrack', color='blue', alpha=0.6)
    sns.histplot(sparse_lengths, bins=20, label='SparseTrack', color='orange', alpha=0.6)
    plt.xlabel('Trajectory Length (frames)')
    plt.ylabel('Count')
    plt.title('Trajectory Length Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/trajectory_lengths.png', dpi=150)
    plt.close()

    print("Report figures saved to report/images/")

if __name__ == "__main__":
    os.makedirs('report/images', exist_ok=True)
    main()