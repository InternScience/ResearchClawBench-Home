import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import namedtuple
import motmetrics as mm
from motmetrics import utils
from code.utils import iou  # assume utils has iou

class KalmanBoxTracker:
    """From ByteTrack repo style"""
    count = 0
    def __init__(self, bbox):
        self.mean = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # x,y,w,h
        self.covariance = np.diag([44, 44, 14400, 14400])  # rough
        self.kf = self.init_kf()
        self.kf.x = np.hstack((self.mean, [0]*4))  # vel 0
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.is_activated = False
        self.score = 0

    def init_kf(self):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([[1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.],
                         [0.,0.,0.,1.,0.,0.,0.,1.],
                         [0.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,1.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.]], dtype=np.float32)
        kf.H = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,0.,0.,0.]], dtype=np.float32)
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        kf.P[:4,:4] *= 1.0
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        return kf

    def predict(self):
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance)  # no, predict first?
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self

    def update(self, bbox, score):
        self.score = score
        ndist = self.normalise(0.0, 1.0, score)
        self.kf.update(self.tlwh_to_xyah(bbox))
        self.time_since_update = 0

    def tlwh_to_xyah(self, tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def normalise(self, mean=0.0, std=1.0, value=None):
        if value is None:
            value = self.score
        return (value - mean) / std

# Simplified ByteTrack
class ByteTracker:
    def __init__(self, args):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.track_thresh = args.get('track_thresh', 0.5)
        self.track_buffer = args.get('track_buffer', 30)
        self.match_thresh = args.get('match_thresh', 0.8)
        self.frame_id = 0

    def update(self, dets_all, img_h, img_w):
        self.frame_id += 1
        dets = [d['bbox'] for d in dets_all]
        scores = [d['score'] for d in dets_all]
        dets = np.array(dets)
        scores = np.array(scores)
        remain_inds = scores > self.track_thresh
        dets_high = dets[remain_inds]
        scores_high = scores[remain_inds]
        dets_low = dets[~remain_inds]

        # predict
        strack_pool = self.tracked_stracks + self.lost_stracks
        pstracks = []
        for strack in strack_pool:
            strack.predict()
            pstracks.append(strack)
        strack_pool = [pstracks[i] for i in range(len(pstracks))]

        # first matching high
        if len(dets_high) > 0:
            ious = np.zeros((len(strack_pool), len(dets_high)), dtype=np.float32)
            for i, strack in enumerate(strack_pool):
                for j, det in enumerate(dets_high):
                    ious[i,j] = iou(strack.tlbr, det)
            ious = 1 - ious
            if min(ious.shape) > 0:
                a = (ious < self.match_thresh).sum(1)
                cal_cost = (ious * 0.05 + (1 - ious) * 0.95) * np.repeat(a[:, np.newaxis], ious.shape[1], axis=1)
                row, col = linear_sum_assignment(cal_cost)
                for ri, ci in zip(row, col):
                    if cal_cost[ri,ci] < self.match_thresh:
                        matched_stracks = pstracks[ri]
                        matched_stracks.update(dets_high[ci], scores_high[ci])
                        self.tracked_stracks.append(matched_stracks)
        # todo low score matching, init new, lost management
        # for demo, simple assoc high only
        tracks = []
        for strack in self.tracked_stracks:
            tlbr = strack.tlbr
            tracks.append({'bbox': tlbr.tolist(), 'id': strack.id})
        return tracks

# Load data and run demo
def run_byte_track():
    with open('data/simulated_sequence.json') as f:
        data = json.load(f)
    bt = ByteTracker({})
    tracks_per_frame = []
    gt_per_frame = []
    for frame_data in data:
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        gt_frame = [[gid, *bbox] for gid, bbox in zip(gt_ids, gt_bboxes)]
        gt_per_frame.append(gt_frame)
        dets_all = frame_data['detections']
        img_info = {'height': 640, 'width': 640}  # assume
        tracks = bt.update(dets_all, img_info['height'], img_info['width'])
        tracks_frame = [[t['id'], *t['bbox']] for t in tracks] if tracks else []
        tracks_per_frame.append(tracks_frame)
    # metrics
    metrics = compute_metrics(gt_per_frame, tracks_per_frame)
    with open('outputs/byte_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print('ByteTrack metrics:', metrics)
    with open('outputs/byte_trajectories.json', 'w') as f:
        json.dump({'tracks_per_frame': tracks_per_frame}, f)
    return tracks_per_frame, gt_per_frame

if __name__ == '__main__':
    run_byte_track()
