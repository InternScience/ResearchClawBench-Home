import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import Track, iou

class ByteTrack:
    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, match_thresh=0.8, max_time_lost=30):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_id = 1
        
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
            
        matched_high, unmatched_stracks, unmatched_dets_high = self.linear_assignment(strack_pool, dets_high, self.match_thresh)
        
        for itrack, idet in matched_high:
            strack_pool[itrack].update(dets_high[idet])
            strack_pool[itrack].state = 'active'
            
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

