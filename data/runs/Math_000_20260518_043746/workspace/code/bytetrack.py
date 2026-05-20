"""
ByteTrack baseline: Two-stage association (high score + low score).
Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
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


class ByteTrackBaseline:
    def __init__(self, score_thresh=0.5, iou_thresh=0.3, max_age=30, min_hits=3):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        
    def process_frame(self, detections, frame_idx):
        """Process one frame of detections."""
        self.frame_count = frame_idx + 1
        
        # Split detections into high and low score
        d_high = [d for d in detections if d['score'] >= self.score_thresh]
        d_low = [d for d in detections if d['score'] < self.score_thresh]
        
        # Predict existing tracks
        pred_boxes = [track.predict() for track in self.tracks]
        
        # First association: all tracks vs high score detections
        if pred_boxes and d_high:
            det_boxes_h = [d['bbox'] for d in d_high]
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes_h)
            cost_h = 1.0 - iou_mat
            matched_h, unmatched_tracks, unmatched_dets_h = hungarian_match(
                cost_h, threshold=1.0 - self.iou_thresh
            )
            for t_idx, d_idx in matched_h:
                self.tracks[t_idx].update(d_high[d_idx]['bbox'], frame_idx)
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets_h = list(range(len(d_high)))
        
        # Second association: remaining tracks vs low score detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        remaining_pred = [pred_boxes[i] for i in unmatched_tracks]
        
        if remaining_pred and d_low:
            det_boxes_l = [d['bbox'] for d in d_low]
            iou_mat_l = compute_iou_matrix(remaining_pred, det_boxes_l)
            cost_l = 1.0 - iou_mat_l
            matched_l, _, _ = hungarian_match(cost_l, threshold=1.0 - self.iou_thresh)
            for t_idx, d_idx in matched_l:
                remaining_tracks[t_idx].update(d_low[d_idx]['bbox'], frame_idx)
        
        # Mark unmatched tracks as missed
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if not t.is_lost()]
        
        # Create new tracks from unmatched high score detections
        for d_idx in unmatched_dets_h:
            new_track = Track(self.next_id, d_high[d_idx]['bbox'], d_high[d_idx]['score'])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Return current frame results
        results = {}
        for track in self.tracks:
            if track.time_since_update == 0:
                results[track.track_id] = track.kf.get_state()
        
        return results
