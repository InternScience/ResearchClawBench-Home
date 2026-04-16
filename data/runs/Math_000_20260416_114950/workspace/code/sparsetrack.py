import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import Track, iou

class SparseTrack:
    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, match_thresh=0.8, max_time_lost=30, depth_levels=3):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.depth_levels = depth_levels
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_id = 1
        
    def estimate_pseudo_depth(self, bboxes):
        if len(bboxes) == 0:
            return []
            
        bottom_y = bboxes[:, 3]
        
        # In crowded scenes, occlusion happens when objects are at different depths.
        # Objects with larger bottom_y are closer to the camera.
        # We should associate closer objects first, because they occlude objects behind them.
        
        # Sort indices by bottom_y descending (closest first)
        sorted_indices = np.argsort(bottom_y)[::-1]
        
        depths = np.zeros(len(bboxes), dtype=int)
        
        # Determine min and max y
        min_y = np.min(bottom_y)
        max_y = np.max(bottom_y)
        
        if max_y == min_y:
            return depths
            
        # Bin into depth levels based on y value
        bins = np.linspace(min_y, max_y, self.depth_levels + 1)
        
        for i in range(len(bboxes)):
            y = bottom_y[i]
            # Find which bin it belongs to (0 to depth_levels-1)
            # largest y should be level 0, so we invert the bin index
            bin_idx = np.digitize(y, bins) - 1
            bin_idx = max(0, min(bin_idx, self.depth_levels - 1))
            depths[i] = (self.depth_levels - 1) - bin_idx
                
        return depths
        
    def update(self, output_results):
        self.frame_id += 1
        
        if len(output_results) == 0:
            for track in self.tracked_stracks:
                track.predict()
                track.state = 'lost'
            self.lost_stracks.extend(self.tracked_stracks)
            self.tracked_stracks = []
            return []
            
        scores = np.array([res['score'] for res in output_results])
        bboxes = np.array([res['bbox'] for res in output_results])
        
        inds_high = scores >= self.track_high_thresh
        inds_low = np.logical_and(scores >= self.track_low_thresh, scores < self.track_high_thresh)
        
        dets_high = bboxes[inds_high]
        scores_high = scores[inds_high]
        
        dets_low = bboxes[inds_low]
        scores_low = scores[inds_low]
        
        strack_pool = self.tracked_stracks + self.lost_stracks
        for strack in strack_pool:
            strack.predict()
            
        # SparseTrack hierarchical association
        depths_high = self.estimate_pseudo_depth(dets_high)
        
        unmatched_stracks = list(range(len(strack_pool)))
        unmatched_dets_high = list(range(len(dets_high)))
        
        for level in range(self.depth_levels):
            level_det_indices = [i for i in unmatched_dets_high if depths_high[i] == level]
            if len(level_det_indices) == 0 or len(unmatched_stracks) == 0:
                continue
                
            level_dets = dets_high[level_det_indices]
            current_stracks = [strack_pool[i] for i in unmatched_stracks]
            
            # Use stricter match_thresh for closer objects?
            # Or just use the same match_thresh
            matched, un_stracks, un_dets = self.linear_assignment(current_stracks, level_dets, self.match_thresh)
            
            for itrack_idx, idet_idx in matched:
                actual_track_idx = unmatched_stracks[itrack_idx]
                actual_det_idx = level_det_indices[idet_idx]
                
                strack_pool[actual_track_idx].update(dets_high[actual_det_idx])
                strack_pool[actual_track_idx].state = 'active'
                
                unmatched_dets_high.remove(actual_det_idx)
                
            unmatched_stracks = [unmatched_stracks[i] for i in un_stracks]
            
        r_tracked_stracks = [strack_pool[i] for i in unmatched_stracks if strack_pool[i].state == 'active']
        matched_low, unmatched_stracks_low, unmatched_dets_low = self.linear_assignment(r_tracked_stracks, dets_low, 0.5)
        
        for itrack, idet in matched_low:
            r_tracked_stracks[itrack].update(dets_low[idet])
            r_tracked_stracks[itrack].state = 'active'
            
        for itrack in unmatched_stracks_low:
            r_tracked_stracks[itrack].state = 'lost'
            
        for idet in unmatched_dets_high:
            if scores_high[idet] >= self.new_track_thresh:
                track = Track(self.next_id, dets_high[idet])
                self.next_id += 1
                self.tracked_stracks.append(track)
                
        active_stracks = []
        lost_stracks = []
        for track in strack_pool:
            if track.state == 'active':
                active_stracks.append(track)
            elif track.state == 'lost':
                if track.time_since_update > self.max_time_lost:
                    track.state = 'removed'
                    self.removed_stracks.append(track)
                else:
                    lost_stracks.append(track)
                    
        self.tracked_stracks = active_stracks + [t for t in self.tracked_stracks if t not in strack_pool]
        self.lost_stracks = lost_stracks
        
        return [t for t in self.tracked_stracks if t.state == 'active']

    def linear_assignment(self, tracks, dets, thresh):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
            
        cost_matrix = np.zeros((len(tracks), len(dets)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(dets):
                cost_matrix[i, j] = 1 - iou(track.get_bbox(), det)
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_tracks = []
        unmatched_dets = list(range(len(dets)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > thresh:
                unmatched_tracks.append(i)
            else:
                matched.append((i, j))
                if j in unmatched_dets:
                    unmatched_dets.remove(j)
                
        for i in range(len(tracks)):
            if i not in row_ind:
                unmatched_tracks.append(i)
                
        return matched, unmatched_tracks, unmatched_dets
