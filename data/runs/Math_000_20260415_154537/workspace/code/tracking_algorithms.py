"""
Multi-Object Tracking Algorithms Implementation
==============================================

This module implements three tracking algorithms:
1. SORT (Simple Online and Realtime Tracking) - Baseline
2. ByteTrack - Two-stage association with high/low score boxes
3. SparseTrack - Proposed method with pseudo-depth estimation and hierarchical association

Author: Research Analysis
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import json


class KalmanFilter:
    """Kalman filter for bounding box tracking.
    
    State vector: [x_center, y_center, width, height, vx, vy, vw, vh]
    """
    
    def __init__(self):
        self.dt = 1.0  # time step
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.1  # Higher noise for velocity
        
        # Measurement noise covariance
        self.R = np.eye(4) * 1.0
        
        # Initial state covariance
        self.P = np.eye(8) * 10.0
        
        # State vector
        self.x = None
        
    def init_state(self, bbox):
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        
    def predict(self):
        """Predict next state."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Update state with measurement."""
        if self.x is None:
            self.init_state(bbox)
            return self.get_bbox()
        
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        z = np.array([cx, cy, w, h])
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """Get bounding box from state."""
        if self.x is None:
            return None
        cx, cy, w, h = self.x[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])


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


def compute_iou_matrix(track_boxes, det_boxes):
    """Compute IoU matrix between track boxes and detection boxes."""
    iou_matrix = np.zeros((len(track_boxes), len(det_boxes)))
    for i, tbox in enumerate(track_boxes):
        for j, dbox in enumerate(det_boxes):
            iou_matrix[i, j] = compute_iou(tbox, dbox)
    return iou_matrix


def compute_pseudo_depth(bbox, frame_shape=(640, 640)):
    """
    Compute pseudo-depth from bounding box.
    
    In crowded scenes, objects that appear larger (closer to camera) 
    and lower in the image are typically closer.
    
    Returns: depth value (smaller = closer)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Bottom center y position (closer objects appear lower in image)
    bottom_y = y2
    
    # Pseudo-depth combines area and vertical position
    # Larger area + lower position = closer to camera (smaller depth value)
    normalized_area = area / (frame_shape[0] * frame_shape[1])
    normalized_y = bottom_y / frame_shape[0]
    
    # Depth estimation: inverse relationship with size, positive with y position
    depth = 1.0 / (normalized_area + 0.01) + normalized_y * 0.5
    
    return depth


def hierarchical_cluster_by_depth(detections, depths, n_levels=3):
    """
    Cluster detections into hierarchical levels based on pseudo-depth.
    
    Args:
        detections: List of detection dicts
        depths: Array of depth values
        n_levels: Number of depth levels
        
    Returns:
        List of detection indices for each depth level
    """
    if len(detections) == 0:
        return [[] for _ in range(n_levels)]
    
    # Sort by depth
    sorted_indices = np.argsort(depths)
    
    # Divide into levels (equal-sized bins)
    level_size = len(sorted_indices) // n_levels
    levels = []
    
    for i in range(n_levels):
        start_idx = i * level_size
        if i == n_levels - 1:
            end_idx = len(sorted_indices)
        else:
            end_idx = (i + 1) * level_size
        levels.append(sorted_indices[start_idx:end_idx].tolist())
    
    return levels


class Track:
    """Tracklet for a single object."""
    
    _id_counter = 0
    
    def __init__(self, bbox, score=None, frame_id=0):
        self.id = Track._id_counter
        Track._id_counter += 1
        
        self.kf = KalmanFilter()
        self.kf.init_state(bbox)
        
        self.bboxes = [bbox]
        self.scores = [score] if score is not None else []
        self.frame_ids = [frame_id]
        
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        
        self.state = 'confirmed'  # 'tentative', 'confirmed', 'deleted'
        
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.get_bbox()
    
    def update(self, bbox, score=None, frame_id=None):
        """Update with new detection."""
        self.kf.update(bbox)
        self.bboxes.append(bbox)
        self.scores.append(score)
        if frame_id is not None:
            self.frame_ids.append(frame_id)
        
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def get_predicted_bbox(self):
        """Get predicted bounding box."""
        return self.kf.get_bbox()


class SORT:
    """SORT: Simple Online and Realtime Tracking."""
    
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        
    def update(self, detections, frame_id=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with 'bbox' key
        """
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Get predicted boxes from tracks
        track_boxes = [t.get_predicted_bbox() for t in self.tracks]
        det_boxes = [d['bbox'] for d in detections]
        det_scores = [d.get('score', 1.0) for d in detections]
        
        # Associate detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
            
            # Convert to cost (1 - IoU)
            cost_matrix = 1 - iou_matrix
            
            # Hungarian assignment
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            matched_pairs = []
            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    matched_pairs.append((t_idx, d_idx))
        else:
            matched_pairs = []
        
        matched_track_indices = set([p[0] for p in matched_pairs])
        matched_det_indices = set([p[1] for p in matched_pairs])
        
        # Update matched tracks
        for t_idx, d_idx in matched_pairs:
            self.tracks[t_idx].update(
                det_boxes[d_idx], 
                det_scores[d_idx],
                frame_id
            )
        
        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_det_indices:
                track = Track(det_boxes[d_idx], det_scores[d_idx], frame_id)
                track.state = 'tentative'
                self.tracks.append(track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update <= self.max_age and t.state != 'deleted']
        
        # Return active tracks
        return [(t.id, t.get_predicted_bbox()) for t in self.tracks 
                if t.state == 'confirmed' or (t.state == 'tentative' and t.hits >= self.min_hits)]


class ByteTrack:
    """ByteTrack: Two-stage association with high/low score boxes."""
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, 
                 second_match_thresh=0.5, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.track_buffer = track_buffer
        self.tracks = []
        self.lost_tracks = []
        self.frame_count = 0
        
    def update(self, detections, frame_id=None):
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Separate high and low score detections
        det_boxes = [d['bbox'] for d in detections]
        det_scores = [d.get('score', 1.0) for d in detections]
        
        high_dets = []
        low_dets = []
        for i, (box, score) in enumerate(zip(det_boxes, det_scores)):
            if score >= self.track_thresh:
                high_dets.append({'bbox': box, 'score': score, 'idx': i})
            else:
                low_dets.append({'bbox': box, 'score': score, 'idx': i})
        
        # Predict all tracks
        all_tracks = self.tracks + self.lost_tracks
        for track in all_tracks:
            track.predict()
        
        # First association: high score detections with all tracks
        track_boxes = [t.get_predicted_bbox() for t in all_tracks]
        high_boxes = [d['bbox'] for d in high_dets]
        
        matched_pairs = []
        unmatched_tracks = list(range(len(all_tracks)))
        unmatched_high_dets = list(range(len(high_dets)))
        
        if len(all_tracks) > 0 and len(high_dets) > 0:
            iou_matrix = compute_iou_matrix(track_boxes, high_boxes)
            cost_matrix = 1 - iou_matrix
            
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.match_thresh:
                    matched_pairs.append((t_idx, d_idx))
                    if t_idx in unmatched_tracks:
                        unmatched_tracks.remove(t_idx)
                    if d_idx in unmatched_high_dets:
                        unmatched_high_dets.remove(d_idx)
        
        # Update matched tracks
        for t_idx, d_idx in matched_pairs:
            all_tracks[t_idx].update(high_dets[d_idx]['bbox'], 
                                     high_dets[d_idx]['score'], frame_id)
        
        # Second association: unmatched tracks with low score detections
        low_boxes = [d['bbox'] for d in low_dets]
        
        tracks_to_remove = []
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            unmatched_track_boxes = [all_tracks[i].get_predicted_bbox() 
                                      for i in unmatched_tracks]
            iou_matrix = compute_iou_matrix(unmatched_track_boxes, low_boxes)
            cost_matrix = 1 - iou_matrix
            
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            matched_low_track_indices = set()
            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.second_match_thresh:
                    all_tracks[unmatched_tracks[t_idx]].update(
                        low_dets[d_idx]['bbox'],
                        low_dets[d_idx]['score'],
                        frame_id
                    )
                    matched_low_track_indices.add(unmatched_tracks[t_idx])
            
            # Build final unmatched list
            unmatched_tracks = [t_idx for t_idx in unmatched_tracks if t_idx not in matched_low_track_indices]
        
        # Mark unmatched tracks as lost
        for t_idx in unmatched_tracks:
            track = all_tracks[t_idx]
            if track in self.tracks:
                self.tracks.remove(track)
                self.lost_tracks.append(track)
        
        # Create new tracks for unmatched high score detections
        for d_idx in unmatched_high_dets:
            track = Track(high_dets[d_idx]['bbox'], 
                         high_dets[d_idx]['score'], frame_id)
            track.state = 'tentative'
            self.tracks.append(track)
        
        # Clean up lost tracks
        self.lost_tracks = [t for t in self.lost_tracks 
                           if t.time_since_update <= self.track_buffer]
        
        # Return active tracks
        active_tracks = self.tracks + [t for t in self.lost_tracks 
                                      if t.time_since_update <= 1]
        return [(t.id, t.get_predicted_bbox()) for t in active_tracks 
                if t.state == 'confirmed' or (t.state == 'tentative' and t.hits >= 3)]


class SparseTrack:
    """
    SparseTrack: Hierarchical tracking with pseudo-depth estimation.
    
    Decomposes dense target sets into sparse subsets based on pseudo-depth
    and performs hierarchical association to handle occlusions better.
    """
    
    def __init__(self, track_thresh=0.3, match_thresh=0.7, 
                 n_depth_levels=3, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.n_depth_levels = n_depth_levels
        self.track_buffer = track_buffer
        
        # Separate tracks by depth level
        self.tracks_by_level = [[] for _ in range(n_depth_levels)]
        self.lost_tracks_by_level = [[] for _ in range(n_depth_levels)]
        self.frame_count = 0
        
    def update(self, detections, frame_id=None):
        """Update tracker with hierarchical depth-based association."""
        self.frame_count += 1
        
        if len(detections) == 0:
            return []
        
        # Compute pseudo-depth for all detections
        det_boxes = [d['bbox'] for d in detections]
        det_scores = [d.get('score', 1.0) for d in detections]
        depths = np.array([compute_pseudo_depth(box) for box in det_boxes])
        
        # Cluster detections into depth levels
        depth_levels = hierarchical_cluster_by_depth(detections, depths, 
                                                      self.n_depth_levels)
        
        # Process each depth level (from closest to farthest)
        all_results = []
        used_detections = set()
        
        for level_idx in range(self.n_depth_levels):
            level_det_indices = depth_levels[level_idx]
            
            # Filter out already used detections
            level_det_indices = [i for i in level_det_indices 
                                if i not in used_detections]
            
            if len(level_det_indices) == 0:
                continue
            
            # Get detections for this level
            level_dets = [{'bbox': det_boxes[i], 
                          'score': det_scores[i], 
                          'idx': i,
                          'depth': depths[i]} 
                         for i in level_det_indices]
            
            # Get tracks for this level
            level_tracks = self.tracks_by_level[level_idx] + \
                          self.lost_tracks_by_level[level_idx]
            
            # Predict tracks
            for track in level_tracks:
                track.predict()
            
            # Associate within this depth level
            if len(level_tracks) > 0 and len(level_dets) > 0:
                track_boxes = [t.get_predicted_bbox() for t in level_tracks]
                det_boxes_level = [d['bbox'] for d in level_dets]
                
                iou_matrix = compute_iou_matrix(track_boxes, det_boxes_level)
                
                # Weight IoU by depth similarity for better matching
                depth_matrix = np.zeros_like(iou_matrix)
                for i, track in enumerate(level_tracks):
                    track_depth = getattr(track, 'depth', depths[level_det_indices[0]])
                    for j, det in enumerate(level_dets):
                        depth_diff = abs(track_depth - det['depth'])
                        depth_matrix[i, j] = np.exp(-depth_diff)
                
                # Combined similarity
                combined_sim = iou_matrix * 0.7 + depth_matrix * 0.3
                cost_matrix = 1 - combined_sim
                
                track_indices, det_indices = linear_sum_assignment(cost_matrix)
                
                matched_track_indices = set()
                matched_det_indices = set()
                
                for t_idx, d_idx in zip(track_indices, det_indices):
                    if combined_sim[t_idx, d_idx] >= self.match_thresh:
                        level_tracks[t_idx].update(level_dets[d_idx]['bbox'],
                                                   level_dets[d_idx]['score'],
                                                   frame_id)
                        level_tracks[t_idx].depth = level_dets[d_idx]['depth']
                        matched_track_indices.add(t_idx)
                        matched_det_indices.add(d_idx)
                        used_detections.add(level_dets[d_idx]['idx'])
                        all_results.append((level_tracks[t_idx].id, 
                                          level_tracks[t_idx].get_predicted_bbox()))
                
                # Handle unmatched tracks
                for t_idx in range(len(level_tracks)):
                    if t_idx not in matched_track_indices:
                        track = level_tracks[t_idx]
                        if track in self.tracks_by_level[level_idx]:
                            self.tracks_by_level[level_idx].remove(track)
                            self.lost_tracks_by_level[level_idx].append(track)
                
                # Create new tracks for unmatched detections
                for d_idx in range(len(level_dets)):
                    if d_idx not in matched_det_indices:
                        track = Track(level_dets[d_idx]['bbox'],
                                     level_dets[d_idx]['score'],
                                     frame_id)
                        track.depth = level_dets[d_idx]['depth']
                        track.state = 'tentative'
                        self.tracks_by_level[level_idx].append(track)
                        all_results.append((track.id, track.get_predicted_bbox()))
            else:
                # No tracks, create new ones for all detections
                for det in level_dets:
                    track = Track(det['bbox'], det['score'], frame_id)
                    track.depth = det['depth']
                    track.state = 'tentative'
                    self.tracks_by_level[level_idx].append(track)
                    all_results.append((track.id, track.get_predicted_bbox()))
                    used_detections.add(det['idx'])
        
        # Clean up lost tracks
        for level_idx in range(self.n_depth_levels):
            self.lost_tracks_by_level[level_idx] = [
                t for t in self.lost_tracks_by_level[level_idx]
                if t.time_since_update <= self.track_buffer
            ]
        
        return all_results
