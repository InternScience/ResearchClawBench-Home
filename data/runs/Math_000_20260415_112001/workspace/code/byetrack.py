import numpy as np
from scipy.optimize import linear_sum_assignment
from code.kalman import KalmanBoxTracker
from code.utils import iou
from collections import deque

class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracked_stracks = []  # active
        self.lost_stracks = []  # lost
        self.removed_stracks = []
        self.track_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanBoxTracker

    def update(self, output_results, img_info):
        self.height, self.width = img_info['height'], img_info['width']
        if len(output_results) == 0:
            return self.multi_predict()

        dets = np.array([r['bbox'] for r in output_results])
        scores = np.array([r['score'] for r in output_results])
        remain_inds = scores > self.track_thresh
        dets = dets[remain_inds]
        scores = scores[remain_inds]

        # predict
        strack_pool = []
        if self.tracked_stracks:
            strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
            tmp_stracks = [strack.predict() for strack in strack_pool if strack.is_activated()]
            strack_pool = [t for t in tmp_stracks if t is not None]

        # first assoc high score
        ious = iou_dist(strack_pool, dets)
        if len(ious) > 0:
            ious[ious > 1-match_thresh] = 0  # cost = 1-iou
            matched, unmatched_dets, unmatched_tracks = linear_sum_assignment(ious)
            # filter valid
            valid = [m for m in matched if ious[m] < match_thresh]
            for iou_idx in valid:
                track_idx = matched[iou_idx]
                det_idx = iou_idx
                strack_pool[track_idx].update(dets[det_idx])
                self.tracked_stracks.append(strack_pool[track_idx])
                self.tracked_stracks = [t for t in self.tracked_stracks if t.time_since_update < self.max_time_lost]

        # second assoc low score
        dets_low = np.array([r['bbox'] for r in output_results if r['score'] <= self.track_thresh])
        if len(dets_low) > 0 and len(self.lost_stracks):
            ious_low = iou_dist([t for t in self.lost_stracks if t.time_since_update < self.max_time_lost], dets_low)
            if len(ious_low) > 0:
                ious_low[ious_low > 1-match_thresh] = 0
                matched_low, unmatched_low_dets, unmatched_low_tracks = linear_sum_assignment(ious_low)
                # similar filter/update

        # init new
        for det in dets[unmatched_dets]:
            strack = self.kalman_filter(det)
            strack.activate()
            self.tracked_stracks.append(strack)

        # lost
        # ... simplify for now

        outputs = []
        for strack in self.tracked_stracks:
            if strack.is_activated():
                bbox_tmp = strack.tlbr
                outputs.append([strack.id, bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3]])
        return outputs

def iou_dist(tracks, dets):
    if len(tracks)==0 or len(dets)==0:
        return np.empty((0,0))
    cost_matrix = np.zeros((len(tracks), len(dets)))
    for i, track in enumerate(tracks):
        for j, det in enumerate(dets):
            cost_matrix[i,j] = 1 - iou(track.tlbr, det)
    return cost_matrix
