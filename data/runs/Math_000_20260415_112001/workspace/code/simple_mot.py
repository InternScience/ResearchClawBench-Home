import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import motmetrics as mm
mm.lapjv.lapjv_norm = mm.lapjv.lapjv_norm_maxrow  # for stability

def bbox_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def simple_byte_track(data, track_high_thresh=0.5, match_thresh=0.5):
    tracks = defaultdict(list)
    next_id = 0
    kalman_states = {}  # id -> [cx,cy,s,r, vx,vy,vs,vr] approx
    lost_count = defaultdict(int)
next_id = 0
    for f, frame in enumerate(data):
        gt_bboxes = np.array(frame['gt_bboxes'])
        gt_ids = frame['gt_ids']
        dets_raw = frame['detections']
        det_bboxes = np.array([d['bbox'] for d in dets_raw])
        det_scores = np.array([d['score'] for d in dets_raw])

        # predict
        pred_bboxes = []
        active_ids = []
        for tid in list(kalman_states):
            state = kalman_states[tid]
            # simple predict: add vel
            pred_state = state.copy()
            pred_state[0] += state[4]
            pred_state[1] += state[5]
            pred_state[2] += state[6]
            pred_state[3] += state[7]
            cx,cy,w,h = pred_state[0:4]  # s=area=w*h, r=h/w approx
            pred_bbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            pred_bboxes.append(pred_bbox)
            active_ids.append(tid)
        if pred_bboxes:
            pred_bboxes = np.array(pred_bboxes)
        else:
            pred_bboxes = np.empty((0,4))

        frame_tracks = []
        if len(pred_bboxes) > 0 and len(det_bboxes) > 0:
            cost_matrix = 1 - bbox_iou(pred_bboxes, det_bboxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r,c] < match_thresh:
                    matches.append((r,c))
            for r,c in matches:
                tid = active_ids[r]
                bbox = det_bboxes[c]
                tracks[tid].append((f, bbox))
                # update kalman simple
                cx = (bbox[0] + bbox[2])/2
                cy = (bbox[1] + bbox[3])/2
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                s = w*h
                r = h/w if w > 0 else 0
                vx = cx - kalman_states[tid][0]
                vy = cy - kalman_states[tid][1]
                vs = s - kalman_states[tid][2]
                vr = r - kalman_states[tid][3]
                kalman_states[tid] = [cx,cy,s,r, vx,vy,vs,vr]
                lost_count[tid] = 0
                frame_tracks.append((tid, bbox))

            # new tracks from high score unmatched
            high_mask = det_scores > track_high_thresh
            unmatched_high_cols = set(range(len(det_bboxes))) - set(c for _,c in matches)
            for c in unmatched_high_cols:
                if high_mask[c]:
                    bbox = det_bboxes[c]
                    #global
                    tid = next_id
                    next_id += 1
                    tracks[tid].append((f, bbox))
                    cx = (bbox[0] + bbox[2])/2
                    cy = (bbox[1] + bbox[3])/2
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    s = w*h
                    r = h/w if w > 0 else 0
                    kalman_states[tid] = [cx,cy,s,r, 0,0,0,0]
                    frame_tracks.append((tid, bbox))
                    lost_count[tid] = 0

            # low score assoc to lost/pred
            low_mask = det_scores <= track_high_thresh
            low_cols = [c for c in unmatched_high_cols if low_mask[c]]
            if low_cols and len(pred_bboxes) > 0:
                low_dets = det_bboxes[low_cols]
                cost_low = 1 - bbox_iou(pred_bboxes, low_dets)
                row_ind_low, col_ind_low = linear_sum_assignment(cost_low)
                for r, cl in zip(row_ind_low, col_ind_low):
                    if cost_low[r,cl] < match_thresh:
                        tid = active_ids[r]
                        bbox = low_dets[cl]
                        tracks[tid].append((f, bbox))
                        # update
                        cx = (bbox[0] + bbox[2])/2
                        cy = (bbox[1] + bbox[3])/2
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        s = w*h
                        r = h/w if w > 0 else 0
                        vx = cx - kalman_states[tid][0]
                        vy = cy - kalman_states[tid][1]
                        vs = s - kalman_states[tid][2]
                        vr = r - kalman_states[tid][3]
                        kalman_states[tid] = [cx,cy,s,r, vx,vy,vs,vr]
                        lost_count[tid] = 0
                        frame_tracks.append((tid, bbox))

        # remove lost
        to_remove = [tid for tid in kalman_states if lost_count[tid] > 10]
        for tid in to_remove:
            del kalman_states[tid]
            del lost_count[tid]

        tracks_per_frame.append(frame_tracks)

    # to mot format
    gt_per_frame = []
next_id = 0
    for f, frame in enumerate(data):
        gt_bboxes = np.array(frame['gt_bboxes'])
        gt_ids = frame['gt_ids']
        gt_frame = np.column_stack((gt_bboxes, gt_ids[:,np.newaxis])).tolist()
        gt_per_frame.append(gt_frame)

    return gt_per_frame, tracks_per_frame

def compute_mot_metrics(gt_frames, track_frames):
    acc_gt = mm.io.loadtxt('', np.array(gt_frames), 'gt')
    acc_trk = mm.io.loadtxt('', np.array(track_frames), 'trk')
    mh = mm.MetricsHost()
    mh.compute(acc_gt, acc_trk, 'acc', names=('x1', 'y1', 'x2', 'y2', 'id'))
    summary = mh.compute_summary(
        metrics=['num_frames', 'mota', 'motp', 'idp', 'idf1'], name='acc'
    )
    return summary['acc'][0]

if __name__ == '__main__':
    with open('data/simulated_sequence.json') as f:
        data = json.load(f)
    gt_frames, track_frames = simple_byte_track(data)
    metrics = compute_mot_metrics(gt_frames, track_frames)
    print('Metrics:', dict(metrics))
    with open('outputs/bytetrack_results.json', 'w') as f:
        json.dump({'metrics': dict(metrics), 'track_frames': track_frames}, f, indent=2)
