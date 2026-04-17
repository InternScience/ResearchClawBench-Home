#!/usr/bin/env python3
"""
SparseTrack: Multi-Object Tracking with Pseudo-Depth Based Sparse Subset Decomposition

This implementation follows the scientific target of handling occlusions by:
1. Estimating pseudo-depth for each detection
2. Decomposing dense target sets into sparse subsets based on depth layers
3. Performing hierarchical association within and across depth layers
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


def estimate_pseudo_depth(bbox, frame_height=640):
    """
    Estimate pseudo-depth from bounding box.
    Uses vertical position and box area to estimate relative depth.
    Returns a value where lower = closer to camera, higher = farther.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    center_y = (y1 + y2) / 2
    
    # Objects lower in the image (higher y) are typically closer
    # Objects with larger area are typically closer
    depth_from_y = 1.0 - (center_y / frame_height)  # Higher y = lower depth
    depth_from_area = 1.0 / (1.0 + np.sqrt(area) / 100)  # Larger area = lower depth
    
    # Combine with weights
    pseudo_depth = 0.6 * depth_from_y + 0.4 * depth_from_area
    return pseudo_depth


def decompose_into_depth_layers(detections, num_layers=4):
    """
    Decompose detections into sparse subsets based on pseudo-depth.
    Returns a list of lists, where each inner list contains detections at that depth layer.
    Layer 0 = closest, Layer num_layers-1 = farthest.
    """
    # Compute pseudo-depth for each detection
    depths = []
    for i, det in enumerate(detections):
        depth = estimate_pseudo_depth(det['bbox'])
        depths.append((i, depth))
    
    # Sort by depth
    depths.sort(key=lambda x: x[1])
    
    # Split into layers
    layer_size = len(depths) // num_layers
    layers = []
    for i in range(num_layers):
        start_idx = i * layer_size
        if i == num_layers - 1:
            end_idx = len(depths)
        else:
            end_idx = (i + 1) * layer_size
        
        layer_indices = [depths[j][0] for j in range(start_idx, end_idx)]
        layer_dets = [detections[idx] for idx in layer_indices]
        layers.append(layer_dets)
    
    return layers


class KalmanFilter:
    """Simple Kalman Filter for bounding box prediction."""
    
    def __init__(self, dt=1.0):
        self.dt = dt
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.x = None
        self.P = None
        self.F = np.eye(8)
        self.F[0:4, 4:8] = np.eye(4) * dt  # Position update from velocity
        
        # Process noise
        self.Q = np.eye(8) * 0.01
        
        # Measurement noise
        self.R = np.eye(4) * 0.1
        
    def initialize(self, bbox):
        """Initialize filter with a bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        self.x = np.array([x, y, w, h, 0, 0, 0, 0])
        self.P = np.eye(8) * 100  # High initial uncertainty
        
    def predict(self):
        """Predict next state."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # Return [x, y, w, h]
    
    def update(self, bbox):
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        
        H = np.zeros((4, 8))
        H[:, :4] = np.eye(4)
        
        # Kalman gain
        y_meas = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S + np.eye(4) * 1e-6)
        
        # Update
        self.x = self.x + K @ y_meas
        self.P = (np.eye(8) - K @ H) @ self.P
        
    def get_bbox(self):
        """Get current bounding box estimate."""
        if self.x is None:
            return None
        x, y, w, h = self.x[:4]
        return [x - w/2, y - h/2, x + w/2, y + h/2]


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
        self.depth_layer = estimate_pseudo_depth(bbox)
        
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
        self.depth_layer = estimate_pseudo_depth(bbox)
        
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


class SparseTracker:
    """
    SparseTrack: Multi-Object Tracker with pseudo-depth based decomposition.
    
    Key features:
    1. Decomposes detections into depth layers
    2. Associates within layers first (hierarchical)
    3. Handles occlusions by considering low-confidence detections
    """
    
    def __init__(self, score_threshold=0.1, depth_layers=4, 
                 age_threshold=3, iou_threshold=0.3):
        self.score_threshold = score_threshold
        self.depth_layers = depth_layers
        self.age_threshold = age_threshold
        self.iou_threshold = iou_threshold
        
        self.tracklets = []
        self.next_track_id = 0
        self.frame_count = 0
        
        # Metrics
        self.id_switches = 0
        self.gt_id_mapping = {}  # Maps gt_id to track_id for evaluation
        
    def _associate_within_layer(self, tracklets_layer, detections_layer):
        """Associate tracklets and detections within a depth layer using Hungarian-like greedy matching."""
        if not tracklets_layer or not detections_layer:
            return [], list(range(len(detections_layer)))
        
        # Build cost matrix (IoU-based)
        n_tracks = len(tracklets_layer)
        n_dets = len(detections_layer)
        
        cost_matrix = np.zeros((n_tracks, n_dets))
        for i, track_idx in enumerate(tracklets_layer):
            track = self.tracklets[track_idx]
            pred_bbox = track.get_predicted_bbox()
            if pred_bbox is None:
                cost_matrix[i, :] = -1
                continue
            for j, det_idx in enumerate(detections_layer):
                det = det_idx if isinstance(det_idx, dict) else detections_layer[det_idx]
                iou = compute_iou(pred_bbox, det['bbox'])
                cost_matrix[i, j] = iou
        
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        associations = []
        
        # Sort by highest IoU first
        matches = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if cost_matrix[i, j] >= self.iou_threshold:
                    matches.append((cost_matrix[i, j], i, j))
        matches.sort(reverse=True)
        
        for iou, i, j in matches:
            if i not in matched_tracks and j not in matched_dets:
                matched_tracks.add(i)
                matched_dets.add(j)
                associations.append((tracklets_layer[i], 
                                     detections_layer[j] if isinstance(detections_layer[j], dict) 
                                     else detections_layer[detections_layer[j]]))
        
        unmatched_det_indices = [j for j in range(n_dets) if j not in matched_dets]
        return associations, unmatched_det_indices
    
    def _hungarian_match(self, cost_matrix, threshold=0.3):
        """Simple Hungarian-style matching."""
        if cost_matrix.size == 0:
            return [], [], list(range(cost_matrix.shape[1])) if cost_matrix.shape[0] == 0 else []
        
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
        
        # Sort by cost (descending for IoU)
        valid_matches.sort(reverse=True)
        
        for cost, i, j in valid_matches:
            if i not in matched_rows and j not in matched_cols:
                matched_rows.add(i)
                matched_cols.add(j)
                matches.append((i, j))
        
        unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
        return matches, list(matched_rows), unmatched_cols
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'score', and optionally 'gt_id'
            frame: Frame number
            
        Returns:
            List of active tracklets with their current bboxes
        """
        if frame is None:
            frame = self.frame_count
        self.frame_count = frame + 1
        
        # Filter detections by score
        high_score_dets = [d for d in detections if d['score'] >= 0.5]
        low_score_dets = [d for d in detections if d['score'] < 0.5]
        
        # Decompose high-score detections into depth layers
        layers = decompose_into_depth_layers(high_score_dets, self.depth_layers)
        
        # Predict all tracklets
        for track in self.tracklets:
            track.predict()
            track.time_since_update += 1
        
        # Hierarchical association: process from closest to farthest layer
        all_associated_dets = set()
        track_indices_by_layer = [[] for _ in range(self.depth_layers)]
        
        # Assign tracklets to depth layers based on their predicted position
        for ti, track in enumerate(self.tracklets):
            if track.time_since_update > self.age_threshold:
                continue
            pred_bbox = track.get_predicted_bbox()
            if pred_bbox is not None:
                depth = estimate_pseudo_depth(pred_bbox)
                layer_idx = int(depth * self.depth_layers)
                layer_idx = min(layer_idx, self.depth_layers - 1)
                track_indices_by_layer[layer_idx].append(ti)
        
        # Associate layer by layer
        remaining_unmatched_tracks = []
        
        for layer_idx in range(self.depth_layers):
            layer_dets = layers[layer_idx]
            layer_tracks = track_indices_by_layer[layer_idx]
            
            # Filter out already matched tracks
            layer_tracks = [t for t in layer_tracks if t not in [a[0] for a in []]]
            
            if not layer_tracks or not layer_dets:
                continue
            
            # Build IoU cost matrix
            cost_matrix = np.zeros((len(layer_tracks), len(layer_dets)))
            for i, ti in enumerate(layer_tracks):
                track = self.tracklets[ti]
                pred_bbox = track.get_predicted_bbox()
                if pred_bbox is None:
                    cost_matrix[i, :] = 0
                    continue
                for j, det in enumerate(layer_dets):
                    iou = compute_iou(pred_bbox, det['bbox'])
                    cost_matrix[i, j] = iou
            
            # Match
            matches, matched_rows, unmatched_cols = self._hungarian_match(cost_matrix, self.iou_threshold)
            
            for row_i, col_j in matches:
                ti = layer_tracks[row_i]
                det = layer_dets[col_j]
                self.tracklets[ti].update(det['bbox'], frame)
                all_associated_dets.add(id(det))
            
            # Collect unmatched tracks for next layer
            for i, ti in enumerate(layer_tracks):
                if i not in matched_rows:
                    remaining_unmatched_tracks.append(ti)
        
        # Second association: match remaining tracklets with low-score detections
        # This helps recover occluded objects
        if remaining_unmatched_tracks and low_score_dets:
            cost_matrix = np.zeros((len(remaining_unmatched_tracks), len(low_score_dets)))
            for i, ti in enumerate(remaining_unmatched_tracks):
                track = self.tracklets[ti]
                pred_bbox = track.get_predicted_bbox()
                if pred_bbox is None:
                    cost_matrix[i, :] = 0
                    continue
                for j, det in enumerate(low_score_dets):
                    iou = compute_iou(pred_bbox, det['bbox'])
                    cost_matrix[i, j] = iou
            
            matches, _, _ = self._hungarian_match(cost_matrix, self.iou_threshold)
            for row_i, col_j in matches:
                ti = remaining_unmatched_tracks[row_i]
                det = low_score_dets[col_j]
                self.tracklets[ti].update(det['bbox'], frame)
                all_associated_dets.add(id(det))
        
        # Create new tracklets for unmatched high-score detections
        for det in high_score_dets:
            if id(det) not in all_associated_dets:
                new_track = Tracklet(self.next_track_id, det['bbox'], frame)
                self.tracklets.append(new_track)
                self.next_track_id += 1
        
        # Remove dead tracklets
        self.tracklets = [t for t in self.tracklets if t.time_since_update <= self.age_threshold]
        
        # Update gt_id mapping for evaluation
        for det in detections:
            if 'gt_id' in det:
                # Find matching tracklet
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
                'age': track.age
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
    """Compute simple MOT metrics."""
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for frame_result, gt_frame in zip(tracking_results, ground_truth):
        gt_bboxes = gt_frame['gt_bboxes']
        tracked_bboxes = [t['bbox'] for t in frame_result['tracklets']]
        
        total_gt += len(gt_bboxes)
        
        # Simple IoU-based matching
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
    output_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756/outputs/sparsetrack_results.json')
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print("Running SparseTrack...")
    tracker = SparseTracker(
        score_threshold=0.1,
        depth_layers=4,
        age_threshold=5,
        iou_threshold=0.2
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
    
    print(f"\n=== SparseTrack Results ===")
    print(f"MOTA: {metrics['mota']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"ID Switches: {tracker.id_switches}")
    print(f"Total Trajectories: {len(tracker.get_trajectories())}")
    print(f"\nResults saved to: {output_path}")
