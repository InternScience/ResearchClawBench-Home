import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import motmetrics as mm

def iou(bb1, bb2):
    xx1 = np.maximum(bb1[0], bb2[0])
    yy1 = np.maximum(bb1[1], bb2[1])
    xx2 = np.minimum(bb1[2], bb2[2])
    yy2 = np.minimum(bb1[3], bb2[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    union = (bb1[2]-bb1[0])*(bb1[3]-bb1[1]) + (bb2[2]-bb2[0])*(bb2[3]-bb2[1]) - inter
    return inter / union if union > 0 else 0

def convert_to_mot_format(tracks_per_frame):
    # tracks_per_frame: list of list [[id, x1,y1,x2,y2], ...]
    accs = []
    for frame_tracks in tracks_per_frame:
        if not frame_tracks:
            accs.append(np.empty((0,5)))
            continue
        ft = np.array(frame_tracks)
        accs.append(ft[:, [1,2,3,4,0]].astype(float))  # x,y,w,h,id -> x1,y1,x2,y2,id
    return accs

def compute_metrics(gt_per_frame, tracks_per_frame):
    gt_accs = convert_to_mot_format(gt_per_frame)
    track_accs = convert_to_mot_format(tracks_per_frame)
    mh = mm.MetricsHost()
    mh.compute(gt_accs, track_accs, 'acc', names=['x1','y1','x2','y2','id'])
    summary = mh.compute_summary(
        metrics=['num_frames', 'mota', 'motp', 'idp', 'idf1'], name='acc'
    )
    return dict(summary['acc'])
