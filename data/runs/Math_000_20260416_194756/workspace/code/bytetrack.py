#!/usr/bin/env python3
"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box

This implementation follows the ByteTrack paper approach:
1. First association with high-score detections
2. Second association between unmatched tracklets and low-score detections
3. This helps recover occluded objects that have low detection scores
"""

import numpy as np
from collections import defaultdict
import json
from pathlib import Path


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class KalmanFilter:
    """Kalman Filter for bounding box prediction (following SORT/DeepSORT formulation)."""
    
    def __init__(self, dt=1.0):
        self.dt = dt
        # State: [x, y, s, r, vx, vy, vs, 0] where x,y=center, s=scale(area), r=aspect ratio
        self.x = None
        self.P = None
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[0, 4] = dt  # x += vx * dt
        self.F[1, 5] = dt  # y += vy * dt
        self.F[2, 6] = dt  # s += vs * dt
        
        # Process noise covariance
        self.Q = np.diag([1, 1, 1, 1e-4, 10, 10, 10, 1e-4])
        
        # Measurement matrix (we observe x, y, s, r)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        
        # Measurement noise covariance
        self.R = np.diag([1, 1, 10, 1e-2])
        
    def initialize(self, bbox):
        """Initialize filter with a bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)  # scale (area)
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0  # aspect ratio
        
        self.x = np.array([x, y, s, r, 0, 0, 0, 0])
        self.P = np.diag([10, 10, 100, 1, 1000, 1000, 1000, 1])
        
    def predict(self):
        """Predict next state."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # Return [x, y, s, r]
    
    def update(self, bbox):
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
        
        z = np.array([x, y, s, r])
        
        # Innovation
        y_meas = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S + np.eye(4) * 1e-6)
        
        # Update state
        self.x = self.x + K @ y_meas
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
    def get_bbox(self):
        """Get current bounding box estimate as [x1, y1, x2, y2]."""
        if self.x is None:
            return None
        x, y, s, r = self.x[:4]
        
        # Convert back to bbox
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        
        return [x1, y1, x2, y2]


class Tracklet:
    """Represents a single object track."""
    
    def __init__(self, track_id, bbox, frame):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.initialize(bbox)
        self.bbox_history = [bbox]
        self.frames = [frame]
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        
    def predict(self):
        """Predict next position."""
        return self.kf.predict()
    
    def update(self, bbox, frame):
        """Update with new detection."""
        self.kf.update(bbox)
        self.bbox_history.append(bbox)
        self.frames.append(frame)
        self.age += 1
        self.time_since_update = 0
        self.hits += 1
        
    def get_predicted_bbox(self):
        """Get predicted bounding box."""
        pred = self.kf.get_bbox()
        if pred is None:
            return None
        # Ensure valid bbox
        x1, y1, x2, y2 = pred
        if x2 <= x1 or y2 <= y1:
            return None
        return pred


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracker that associates every detection box.
    
    Key features:
    1. Two-stage association (high-score then low-score detections)
    2. Recovers occluded objects through low-score detection matching
    3. Simple yet effective motion-based association
    """
    
    def __init__(self, score_threshold=0.5, age_threshold=30, 
                 iou_threshold=0.3, low_iou_threshold=0.5):
        self.score_threshold = score_threshold  # Threshold to separate high/low score
        self.age_threshold = age_threshold
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold  # More lenient for low-score matching
        
        self.tracklets = []
        self.next_track_id = 0
        self.frame_count = 0
        
        # Metrics
        self.id_switches = 0
        self.gt_id_mapping = {}
        
    def _hungarian_match(self, cost_matrix, threshold):
        """Simple greedy matching based on IoU cost matrix."""
        if cost_matrix.size == 0:
            return [], [], []
        
        n_rows, n_cols = cost_matrix.shape
        matched_rows = set()
        matched_cols = set()
        matches = []
        
        # Get all valid assignments above threshold
        valid_matches = []
        for i in range(n_rows):
            for j in range(n_cols):
                if cost_matrix[i, j] >= threshold:
                    valid_matches.append((cost_matrix[i, j], i, j))
        
        # Sort by cost (descending for IoU) and greedily match
        valid_matches.sort(reverse=True)
        
        for cost, i, j in valid_matches:
            if i not in matched_rows and j not in matched_cols:
                matched_rows.add(i)
                matched_cols.add(j)
                matches.append((i, j))
        
        unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
        unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
        
        return matches, unmatched_rows, unmatched_cols
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections using ByteTrack's two-stage association.
        
        Stage 1: Associate high-score detections with tracklets
        Stage 2: Associate remaining tracklets with low-score detections
        """
        if frame is None:
            frame = self.frame_count
        self.frame_count = frame + 1
        
        # Split detections into high and low score groups
        high_score_dets = [d for d in detections if d['score'] >= self.score_threshold]
        low_score_dets = [d for d in detections if d['score'] < self.score_threshold]
        
        # Predict all tracklets
        for track in self.tracklets:
            track.predict()
            track.time_since_update += 1
        
        # === STAGE 1: Associate high-score detections ===
        if self.tracklets and high_score_dets:
            # Build IoU cost matrix
            cost_matrix = np.zeros((len(self.tracklets), len(high_score_dets)))
            for i, track in enumerate(self.tracklets):
                pred_bbox = track.get_predicted_bbox()
                if pred_bbox is None:
                    cost_matrix[i, :] = 0
                    continue
                for j, det in enumerate(high_score_dets):
                    iou = compute_iou(pred_bbox, det['bbox'])
                    cost_matrix[i, j] = iou
            
            # Match
            matches, unmatched_tracks, unmatched_dets = self._hungarian_match(
                cost_matrix, self.iou_threshold
            )
            stage1_matches = matches  # Save for later use
            
            # Update matched tracklets
            for track_idx, det_idx in matches:
                self.tracklets[track_idx].update(high_score_dets[det_idx]['bbox'], frame)
            
            # Collect unmatched tracklets for stage 2
            unmatched_track_indices = unmatched_tracks
        else:
            unmatched_track_indices = list(range(len(self.tracklets)))
        
        # === STAGE 2: Associate remaining tracklets with low-score detections ===
        # This is the key innovation of ByteTrack - recovering occluded objects
        if unmatched_track_indices and low_score_dets:
            # Build IoU cost matrix for unmatched tracklets
            cost_matrix = np.zeros((len(unmatched_track_indices), len(low_score_dets)))
            for i, track_idx in enumerate(unmatched_track_indices):
                track = self.tracklets[track_idx]
                pred_bbox = track.get_predicted_bbox()
                if pred_bbox is None:
                    cost_matrix[i, :] = 0
                    continue
                for j, det in enumerate(low_score_dets):
                    iou = compute_iou(pred_bbox, det['bbox'])
                    cost_matrix[i, j] = iou
            
            # Use more lenient threshold for low-score matching
            matches, _, _ = self._hungarian_match(
                cost_matrix, self.low_iou_threshold
            )
            
            # Update matched tracklets
            for match_i, det_idx in matches:
                track_idx = unmatched_track_indices[match_i]
                self.tracklets[track_idx].update(low_score_dets[det_idx]['bbox'], frame)
                # Mark this detection as used
                low_score_dets[det_idx]['used'] = True
        
        # === Create new tracklets for unmatched high-score detections ===
        # We only create new tracks from high-score detections to avoid false positives
        if not self.tracklets or not high_score_dets:
            stage1_matches = []
        else:
            stage1_matches = stage1_matches  # Already defined earlier
            
        for det in high_score_dets:
            # Check if this detection was already matched (we need to track this)
            is_matched = False
            for _, det_idx in stage1_matches:
                if det_idx < len(high_score_dets) and high_score_dets[det_idx] is det:
                    is_matched = True
                    break
            
            if not is_matched:
                new_track = Tracklet(self.next_track_id, det['bbox'], frame)
                self.tracklets.append(new_track)
                self.next_track_id += 1
        
        # Remove dead tracklets
        self.tracklets = [t for t in self.tracklets if t.time_since_update <= self.age_threshold]
        
        # Update gt_id mapping for evaluation
        for det in detections:
            if 'gt_id' in det:
                for track in self.tracklets:
                    pred_bbox = track.get_predicted_bbox()
                    if pred_bbox and compute_iou(pred_bbox, det['bbox']) > 0.5:
                        if track.track_id in self.gt_id_mapping:
                            if self.gt_id_mapping[track.track_id] != det['gt_id']:
                                self.id_switches += 1
                        self.gt_id_mapping[track.track_id] = det['gt_id']
                        break
        
        return self.tracklets
    
    def get_trajectories(self):
        """Return all trajectories as dict mapping track_id to list of bboxes."""
        trajectories = {}
        for track in self.tracklets:
            trajectories[track.track_id] = {
                'bboxes': track.bbox_history,
                'frames': track.frames,
                'age': track.age,
                'hits': track.hits
            }
        return trajectories


def run_tracking(data, tracker):
    """Run tracking on the entire sequence."""
    all_trajectories = []
    
    for frame_data in data:
        frame = frame_data['frame']
        detections = frame_data['detections']
        
        # Update tracker
        tracklets = tracker.update(detections, frame)
        
        # Store current frame results
        frame_results = {
            'frame': frame,
            'tracklets': [
                {
                    'track_id': t.track_id,
                    'bbox': t.bbox_history[-1],
                    'age': t.age
                }
                for t in tracklets
            ]
        }
        all_trajectories.append(frame_results)
    
    return all_trajectories


def compute_mot_metrics(tracking_results, ground_truth):
    """Compute MOT metrics."""
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for frame_result, gt_frame in zip(tracking_results, ground_truth):
        gt_bboxes = gt_frame['gt_bboxes']
        tracked_bboxes = [t['bbox'] for t in frame_result['tracklets']]
        
        total_gt += len(gt_bboxes)
        
        # IoU-based matching
        matched_gt = set()
        for tb in tracked_bboxes:
            matched = False
            for gi, gb in enumerate(gt_bboxes):
                if gi in matched_gt:
                    continue
                if compute_iou(tb, gb) > 0.5:
                    matched_gt.add(gi)
                    matched = True
                    break
            if matched:
                total_tp += 1
            else:
                total_fp += 1
        
        total_fn += len(gt_bboxes) - len(matched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    mota = 1 - (total_fp + total_fn) / total_gt if total_gt > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'mota': mota,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_gt': total_gt
    }


if __name__ == '__main__':
    # Load data
    data_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756/data/simulated_sequence.json')
    output_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756/outputs/bytetrack_results.json')
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print("Running ByteTrack...")
    tracker = ByteTracker(
        score_threshold=0.5,
        age_threshold=30,
        iou_threshold=0.3,
        low_iou_threshold=0.5
    )
    
    results = run_tracking(data, tracker)
    
    # Compute metrics
    metrics = compute_mot_metrics(results, data)
    
    # Save results
    output_data = {
        'tracking_results': results,
        'metrics': metrics,
        'id_switches': tracker.id_switches,
        'num_trajectories': len(tracker.get_trajectories())
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n=== ByteTrack Results ===")
    print(f"MOTA: {metrics['mota']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"ID Switches: {tracker.id_switches}")
    print(f"Total Trajectories: {len(tracker.get_trajectories())}")
    print(f"\nResults saved to: {output_path}")
