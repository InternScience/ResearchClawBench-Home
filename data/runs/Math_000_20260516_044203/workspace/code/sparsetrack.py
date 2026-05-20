"""
SparseTrack: Decomposing Dense Target Sets into Sparse Subsets via 
Pseudo-Depth Estimation and Hierarchical Association

Key ideas:
1. Pseudo-depth estimation from bounding box properties (area + vertical position)
2. Decompose dense detection sets into sparse subsets by depth layers
3. Hierarchical association: intra-layer then inter-layer matching
4. Handle occlusions better by processing depth-ordered subsets
"""

import numpy as np
from mot_utils import KalmanFilter, hungarian_match, iou_cost_matrix, bbox_iou


class SparseTrack:
    """
    SparseTrack: Decomposes dense target sets into sparse subsets via 
    pseudo-depth estimation and performs hierarchical association.
    
    Parameters:
        num_depth_layers: Number of depth layers to decompose into (default 3)
        track_high_thresh: Detection score threshold for high-confidence (default 0.2)
        track_low_thresh: Minimum detection score (default 0.1)
        match_thresh: IoU threshold for matching (default 0.2)
        max_age: Maximum frames to keep track without update (default 30)
        min_hits: Minimum detections to confirm track (default 3)
        depth_method: Method for pseudo-depth estimation ('scale', 'position', 'combined')
    """
    
    def __init__(self, num_depth_layers=3, track_high_thresh=0.2,
                 track_low_thresh=0.1, match_thresh=0.2,
                 max_age=30, min_hits=3, depth_method='combined'):
        self.num_depth_layers = num_depth_layers
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.depth_method = depth_method
        
        self.tracks = []
        self.frame_count = 0
        self.track_id_counter = 0
    
    def estimate_pseudo_depth(self, bbox, frame_height=500.0, frame_width=500.0):
        """
        Estimate pseudo-depth for a bounding box.
        
        Uses a combination of:
        - Scale: larger boxes are typically closer (lower depth)
        - Position: lower boxes are typically closer (lower depth in image)
        
        Returns a scalar depth value (higher = further away).
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        if self.depth_method == 'scale':
            depth = 1.0 - np.clip(area / (frame_width * frame_height * 0.5), 0, 1)
        elif self.depth_method == 'position':
            center_y = (y1 + y2) / 2.0
            depth = center_y / frame_height
        elif self.depth_method == 'combined':
            area_norm = np.clip(area / (frame_width * frame_height * 0.3), 0, 1)
            center_y = (y1 + y2) / 2.0
            pos_norm = center_y / frame_height
            depth = 0.6 * (1.0 - area_norm) + 0.4 * pos_norm
        else:
            depth = 0.5
        
        return depth
    
    def decompose_by_depth(self, detections, frame_height=500.0, frame_width=500.0):
        """
        Decompose detections into sparse subsets based on pseudo-depth.
        Returns list of lists, where each sublist has (orig_idx, detection) for one layer.
        """
        if len(detections) == 0:
            return [[] for _ in range(self.num_depth_layers)]
        
        depths = [self.estimate_pseudo_depth(d['bbox'], frame_height, frame_width) 
                  for d in detections]
        
        sorted_pairs = sorted(enumerate(depths), key=lambda x: x[1])
        
        layer_size = max(1, len(detections) // self.num_depth_layers)
        layers = []
        for i in range(self.num_depth_layers):
            start_idx = i * layer_size
            if i == self.num_depth_layers - 1:
                end_idx = len(detections)
            else:
                end_idx = (i + 1) * layer_size
            layer_pairs = sorted_pairs[start_idx:end_idx]
            layers.append([(orig_idx, detections[orig_idx]) for orig_idx, _ in layer_pairs])
        
        return layers
    
    def update(self, detections, frame_height=500.0, frame_width=500.0):
        """
        Update tracker with new detections using hierarchical association.
        """
        self.frame_count += 1
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track['kf'].predict()
        
        # Separate high and low score detections
        high_dets = [d for d in detections if d['score'] >= self.track_high_thresh]
        low_dets = [d for d in detections 
                    if self.track_low_thresh <= d['score'] < self.track_high_thresh]
        
        # === FIRST ASSOCIATION: High-score detections with depth-aware matching ===
        
        # Decompose high detections by pseudo-depth
        high_layers = self.decompose_by_depth(high_dets, frame_height, frame_width)
        
        # Assign each track a depth based on predicted bbox
        track_depths = []
        for track in self.tracks:
            pred_bbox = track['kf'].get_bbox()
            if pred_bbox is not None:
                depth = self.estimate_pseudo_depth(pred_bbox, frame_height, frame_width)
            else:
                depth = self.estimate_pseudo_depth(track['bbox'], frame_height, frame_width)
            track_depths.append(depth)
        
        matched_track_indices = set()
        matched_det_indices = set()
        track_to_det = {}
        
        # Intra-layer matching
        for layer_idx in range(self.num_depth_layers):
            layer_items = high_layers[layer_idx]
            if len(layer_items) == 0:
                continue
            
            layer_det_indices = [orig_idx for orig_idx, _ in layer_items]
            layer_detections = [det for _, det in layer_items]
            
            depth_min = layer_idx / self.num_depth_layers
            depth_max = (layer_idx + 1) / self.num_depth_layers
            
            layer_track_indices = []
            layer_tracks = []
            for ti in range(len(self.tracks)):
                if ti in matched_track_indices:
                    continue
                td = track_depths[ti]
                relaxed_min = max(0, depth_min - 0.2)
                relaxed_max = min(1, depth_max + 0.2)
                if relaxed_min <= td <= relaxed_max:
                    layer_track_indices.append(ti)
                    layer_tracks.append(self.tracks[ti])
            
            if len(layer_tracks) == 0:
                continue
            
            track_bboxes = [t['kf'].get_bbox() for t in layer_tracks]
            det_bboxes = [d['bbox'] for d in layer_detections]
            
            cost = iou_cost_matrix(track_bboxes, det_bboxes)
            row_ind, col_ind = hungarian_match(cost)
            
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.match_thresh):
                    ti = layer_track_indices[r]
                    di = layer_det_indices[c]
                    matched_track_indices.add(ti)
                    matched_det_indices.add(di)
                    track_to_det[ti] = di
        
        # Cross-layer matching for remaining unmatched
        unmatched_track_indices = [i for i in range(len(self.tracks)) 
                                     if i not in matched_track_indices]
        unmatched_det_indices = [i for i in range(len(high_dets)) 
                                   if i not in matched_det_indices]
        
        if len(unmatched_track_indices) > 0 and len(unmatched_det_indices) > 0:
            unmatched_tracks_subset = [self.tracks[i] for i in unmatched_track_indices]
            unmatched_dets_subset = [high_dets[i] for i in unmatched_det_indices]
            
            track_bboxes = [t['kf'].get_bbox() for t in unmatched_tracks_subset]
            det_bboxes = [d['bbox'] for d in unmatched_dets_subset]
            
            cost = iou_cost_matrix(track_bboxes, det_bboxes)
            row_ind, col_ind = hungarian_match(cost)
            
            cross_thresh = self.match_thresh * 0.85
            
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - cross_thresh):
                    ti = unmatched_track_indices[r]
                    di = unmatched_det_indices[c]
                    matched_track_indices.add(ti)
                    matched_det_indices.add(di)
                    track_to_det[ti] = di
        
        # Update matched tracks
        for ti, di in track_to_det.items():
            det = high_dets[di]
            self.tracks[ti]['kf'].update(det['bbox'])
            self.tracks[ti]['bbox'] = det['bbox']
            self.tracks[ti]['score'] = det['score']
            self.tracks[ti]['age'] = 0
            self.tracks[ti]['hits'] += 1
        
        # === SECOND ASSOCIATION: Remaining tracks with low-score detections ===
        remaining_track_indices = [i for i in range(len(self.tracks)) 
                                     if i not in matched_track_indices]
        
        if len(remaining_track_indices) > 0 and len(low_dets) > 0:
            remaining_tracks = [self.tracks[i] for i in remaining_track_indices]
            
            track_bboxes = [t['kf'].get_bbox() for t in remaining_tracks]
            det_bboxes = [d['bbox'] for d in low_dets]
            
            cost = iou_cost_matrix(track_bboxes, det_bboxes)
            row_ind, col_ind = hungarian_match(cost)
            
            low_thresh = self.match_thresh * 0.7
            
            matched_low_track_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - low_thresh):
                    ti = remaining_track_indices[r]
                    det = low_dets[c]
                    self.tracks[ti]['kf'].update(det['bbox'])
                    self.tracks[ti]['bbox'] = det['bbox']
                    self.tracks[ti]['score'] = det['score']
                    self.tracks[ti]['age'] = 0
                    self.tracks[ti]['hits'] += 1
                    matched_low_track_indices.add(ti)
            
            for ti in remaining_track_indices:
                if ti not in matched_low_track_indices:
                    self.tracks[ti]['age'] += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        # Initialize new tracks from unmatched high-score detections
        for di in range(len(high_dets)):
            if di not in matched_det_indices:
                det = high_dets[di]
                track = {
                    'kf': KalmanFilter(),
                    'bbox': det['bbox'],
                    'score': det['score'],
                    'id': self.track_id_counter,
                    'age': 0,
                    'hits': 1,
                    'start_frame': self.frame_count,
                }
                track['kf'].init(det['bbox'])
                self.tracks.append(track)
                self.track_id_counter += 1
        
        # Return active confirmed tracks
        active_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                active_tracks.append({
                    'bbox': track['bbox'],
                    'id': track['id'],
                    'score': track['score'],
                })
        
        return active_tracks
