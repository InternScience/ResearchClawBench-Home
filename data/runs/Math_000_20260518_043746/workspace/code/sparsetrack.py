"""
SparseTrack: Pseudo-Depth Hierarchical Multi-Object Tracking.

Core idea: Decompose dense target sets into sparse subsets via pseudo-depth estimation
and perform hierarchical association to handle occlusions in crowded scenes.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from code.kalman_filter import KalmanFilter, compute_iou_matrix, hungarian_match


class Track:
    def __init__(self, track_id, bbox, score=1.0):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.initiate(bbox)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = [(0, bbox)]
        self.score = score
        
    def predict(self):
        return self.kf.predict()
    
    def update(self, bbox, frame):
        self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0
        self.history.append((frame, bbox))
        
    def mark_missed(self):
        self.time_since_update += 1
        
    def is_confirmed(self):
        return self.hits >= 3
    
    def is_lost(self):
        return self.time_since_update > 30


class SparseTrack:
    def __init__(self, 
                 score_thresh=0.5,
                 iou_thresh=0.3,
                 max_age=30,
                 min_hits=3,
                 n_depth_layers=5,
                 layer_merge_ratio=0.8):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.n_depth_layers = n_depth_layers
        self.layer_merge_ratio = layer_merge_ratio
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        
    def _estimate_pseudo_depth(self, bboxes):
        """Estimate pseudo-depth from bounding box size."""
        depths = []
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = max(w * h, 1.0)
            depth = 1.0 / np.sqrt(area)
            depths.append(depth)
        return np.array(depths)
    
    def _decompose_into_layers(self, bboxes, scores):
        """Decompose detections into depth layers based on pseudo-depth."""
        if len(bboxes) == 0:
            return []
        
        depths = self._estimate_pseudo_depth(bboxes)
        min_depth_val = np.min(depths)
        max_depth_val = np.max(depths)
        depth_range = max_depth_val - min_depth_val + 1e-6
        
        layer_indices = np.floor(
            (depths - min_depth_val) / depth_range * self.n_depth_layers
        ).astype(int)
        layer_indices = np.clip(layer_indices, 0, self.n_depth_layers - 1)
        
        layers = {}
        for i, layer_id in enumerate(layer_indices):
            if layer_id not in layers:
                layers[layer_id] = []
            layers[layer_id].append(i)
        
        sorted_layers = sorted(layers.items(), key=lambda x: x[0])
        result = []
        for layer_id, indices in sorted_layers:
            mean_depth = np.mean(depths[indices]) if indices else 0
            result.append((indices, mean_depth))
        
        return result
    
    def _hierarchical_association(self, tracks, detections, frame_idx):
        """Perform hierarchical association across depth layers."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        pred_boxes = [t.predict() for t in tracks]
        det_bboxes = [d['bbox'] for d in detections]
        det_scores = [d['score'] for d in detections]
        
        det_depths = self._estimate_pseudo_depth(det_bboxes)
        track_depths = self._estimate_pseudo_depth(pred_boxes)
        
        iou_mat = compute_iou_matrix(pred_boxes, det_bboxes)
        
        all_matched = []
        matched_tracks = set()
        matched_dets = set()
        
        # Stage 1: High-score detections
        high_score_dets = [i for i, s in enumerate(det_scores) if s >= self.score_thresh]
        low_score_dets = [i for i, s in enumerate(det_scores) if s < self.score_thresh]
        
        if high_score_dets:
            cost_h = 1.0 - iou_mat[:, high_score_dets]
            matched_h, _, _ = hungarian_match(cost_h, threshold=1.0 - self.iou_thresh)
            for t_idx, d_local in matched_h:
                d_global = high_score_dets[d_local]
                all_matched.append((t_idx, d_global))
                matched_tracks.add(t_idx)
                matched_dets.add(d_global)
        
        # Stage 2: Hierarchical depth-based association for low-score detections
        remaining_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        remaining_low_dets = [i for i in low_score_dets if i not in matched_dets]
        
        if remaining_tracks and remaining_low_dets:
            rem_det_bboxes = [det_bboxes[i] for i in remaining_low_dets]
            layers = self._decompose_into_layers(rem_det_bboxes, 
                                                  [det_scores[i] for i in remaining_low_dets])
            
            used_tracks_layer = set()
            
            all_track_depths_vals = [track_depths[i] for i in remaining_tracks] if remaining_tracks else [0]
            depth_span = max(all_track_depths_vals) - min(all_track_depths_vals) if len(all_track_depths_vals) > 1 else 0.01
            
            for layer_indices, layer_depth in layers:
                if not layer_indices:
                    continue
                
                global_det_indices = [remaining_low_dets[i] for i in layer_indices]
                depth_window = max(depth_span / self.n_depth_layers * 2, 0.005)
                
                nearby_tracks = [i for i in remaining_tracks 
                               if abs(track_depths[i] - layer_depth) < depth_window
                               and i not in used_tracks_layer]
                
                if nearby_tracks and global_det_indices:
                    layer_cost = 1.0 - iou_mat[np.ix_(nearby_tracks, global_det_indices)]
                    matched_layer, _, _ = hungarian_match(
                        layer_cost, threshold=1.0 - self.iou_thresh
                    )
                    
                    for t_local, d_local in matched_layer:
                        t_global = nearby_tracks[t_local]
                        d_global = global_det_indices[d_local]
                        all_matched.append((t_global, d_global))
                        used_tracks_layer.add(t_global)
        
        all_matched_tracks = set(m[0] for m in all_matched)
        all_matched_dets = set(m[1] for m in all_matched)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in all_matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in all_matched_dets]
        
        return all_matched, unmatched_tracks, unmatched_dets
    
    def process_frame(self, detections, frame_idx):
        """Process one frame of detections."""
        self.frame_count = frame_idx + 1
        
        matched, unmatched_tracks, unmatched_dets = self._hierarchical_association(
            self.tracks, detections, frame_idx
        )
        
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx]['bbox'], frame_idx)
        
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()
        
        self.tracks = [t for t in self.tracks if not t.is_lost()]
        
        for d_idx in unmatched_dets:
            new_track = Track(self.next_id, detections[d_idx]['bbox'], detections[d_idx]['score'])
            self.tracks.append(new_track)
            self.next_id += 1
        
        results = {}
        for track in self.tracks:
            if track.time_since_update == 0:
                results[track.track_id] = track.kf.get_state()
        
        return results
