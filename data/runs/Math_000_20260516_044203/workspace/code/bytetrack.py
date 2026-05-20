"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box

Implementation based on:
Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.

Key ideas:
- Two-stage association: high-score detections first, then low-score detections
- Kalman filter for motion prediction
- IoU-based matching using Hungarian algorithm
"""

import numpy as np
from mot_utils import KalmanFilter, hungarian_match, iou_cost_matrix, bbox_iou


class ByteTrack:
    """
    ByteTrack tracker.
    
    Parameters:
        track_high_thresh: Detection score threshold for high-confidence detections (default 0.5)
        track_low_thresh: Detection score threshold for low-confidence detections (default 0.1)
        match_thresh: IoU threshold for matching (default 0.2)
        max_age: Maximum number of frames to keep a track without updates (default 30)
        min_hits: Minimum number of detections to confirm a track (default 3)
    """
    
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1,
                 match_thresh=0.2, max_age=30, min_hits=3):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks = []  # Active tracks
        self.frame_count = 0
        self.next_id = 0
        self.track_id_counter = 0
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with keys 'bbox' [x1,y1,x2,y2] and 'score'
            
        Returns:
            List of active tracks, each with 'bbox', 'id', 'score'
        """
        self.frame_count += 1
        
        # Separate high and low score detections
        det_high = []
        det_low = []
        for det in detections:
            if det['score'] >= self.track_high_thresh:
                det_high.append(det)
            elif det['score'] >= self.track_low_thresh:
                det_low.append(det)
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track['kf'].predict()
        
        # First association: match high-score detections with tracks
        matched, unmatched_tracks, unmatched_det_high = self._match(
            self.tracks, det_high, self.match_thresh)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = det_high[det_idx]
            track['kf'].update(det['bbox'])
            track['bbox'] = det['bbox']
            track['score'] = det['score']
            track['age'] = 0
            track['hits'] += 1
        
        # Second association: match remaining tracks with low-score detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(remaining_tracks) > 0 and len(det_low) > 0:
            matched_low, unmatched_tracks_low, _ = self._match(
                remaining_tracks, det_low, self.match_thresh)
            
            for track_idx, det_idx in matched_low:
                track = remaining_tracks[track_idx]
                det = det_low[det_idx]
                track['kf'].update(det['bbox'])
                track['bbox'] = det['bbox']
                track['score'] = det['score']
                track['age'] = 0
                track['hits'] += 1
            
            # Mark unmatched low tracks as aged (unmatched_tracks_low are indices into remaining_tracks)
            for idx in unmatched_tracks_low:
                remaining_tracks[idx]['age'] += 1
        else:
            # No low detections: mark all unmatched tracks as aged
            for idx in unmatched_tracks:
                self.tracks[idx]['age'] += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                       if t['age'] <= self.max_age]
        
        # Initialize new tracks from unmatched high-score detections
        for det_idx in unmatched_det_high:
            det = det_high[det_idx]
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
    
    def _match(self, tracks, detections, match_thresh):
        """Match tracks with detections using Hungarian algorithm."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Get predicted bboxes for tracks
        track_bboxes = [t['kf'].get_bbox() for t in tracks]
        det_bboxes = [d['bbox'] for d in detections]
        
        # Compute cost matrix
        cost = iou_cost_matrix(track_bboxes, det_bboxes)
        
        # Apply Hungarian algorithm
        row_ind, col_ind = hungarian_match(cost)
        
        # Filter matches by IoU threshold
        matched = []
        matched_track_set = set()
        matched_det_set = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < (1.0 - match_thresh):  # IoU > match_thresh
                matched.append((r, c))
                matched_track_set.add(r)
                matched_det_set.add(c)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_set]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_set]
        
        return matched, unmatched_tracks, unmatched_dets
