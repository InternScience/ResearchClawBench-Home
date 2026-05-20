"""
Multi-object tracking experiments on simulated_sequence.json.
Implements a ByteTrack-like baseline and a SparseTrack-like tracker
with pseudo-depth estimation and depth cascade matching (DCM).
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import motmetrics as mm
from trackeval.metrics import HOTA
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def box_iou(a, b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(track_boxes, det_boxes):
    """Compute IoU matrix (len(tracks), len(dets))."""
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return np.empty((len(track_boxes), len(det_boxes)))
    m = np.zeros((len(track_boxes), len(det_boxes)))
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            m[i, j] = box_iou(tb, db)
    return m


def associate(tracks, detections, iou_threshold):
    """
    Hungarian matching based on IoU.
    tracks: list of Track objects (must have .predicted_bbox)
    detections: list of dicts with 'bbox'
    Returns: matched_pairs [(track_idx, det_idx)], unmatched_track_idxs, unmatched_det_idxs
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
    t_boxes = [t.predicted_bbox for t in tracks]
    d_boxes = [d['bbox'] for d in detections]
    iou_mat = iou_matrix(t_boxes, d_boxes)
    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_dets = list(range(len(detections)))
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matched.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)
    return matched, unmatched_tracks, unmatched_dets


def pseudo_depth(bbox, image_height):
    """Pseudo-depth = distance from bottom of box to bottom edge of image."""
    return image_height - bbox[3]


def split_intervals(values, k):
    """Split list of scalar values into k uniform intervals."""
    if len(values) == 0:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [(mn, mx)]
    step = (mx - mn) / k
    return [(mn + i * step, mn + (i + 1) * step) for i in range(k)]


def depth_cascade_match(tracks, detections, k, iou_threshold, image_height):
    """
    Depth Cascade Matching (DCM).
    tracks: list of Track objects
    detections: list of dicts
    Returns matched, unmatched_tracks, unmatched_dets (global indices).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    track_depths = [pseudo_depth(t.predicted_bbox, image_height) for t in tracks]
    det_depths = [pseudo_depth(d['bbox'], image_height) for d in detections]

    # Use combined intervals so tracks and detections share the same depth levels
    all_depths = track_depths + det_depths
    intervals = split_intervals(all_depths, k)

    t_subsets = [[] for _ in range(k)]
    for idx, dep in enumerate(track_depths):
        for j, (lo, hi) in enumerate(intervals):
            if j == k - 1:
                if lo <= dep <= hi:
                    t_subsets[j].append(idx)
                    break
            else:
                if lo <= dep < hi:
                    t_subsets[j].append(idx)
                    break

    d_subsets = [[] for _ in range(k)]
    for idx, dep in enumerate(det_depths):
        for j, (lo, hi) in enumerate(intervals):
            if j == k - 1:
                if lo <= dep <= hi:
                    d_subsets[j].append(idx)
                    break
            else:
                if lo <= dep < hi:
                    d_subsets[j].append(idx)
                    break

    matched = []
    T0 = []
    D0 = []
    for level in range(k):
        Ti = t_subsets[level] + T0
        Di = d_subsets[level] + D0
        if len(Ti) == 0 or len(Di) == 0:
            T0 = Ti
            D0 = Di
            continue
        t_list = [tracks[i] for i in Ti]
        d_list = [detections[i] for i in Di]
        m, ut, ud = associate(t_list, d_list, iou_threshold)
        for lt, ld in m:
            matched.append((Ti[lt], Di[ld]))
        T0 = [Ti[i] for i in ut]
        D0 = [Di[i] for i in ud]

    return matched, T0, D0


# ---------------------------------------------------------------------------
# Track class
# ---------------------------------------------------------------------------

class Track:
    _id_counter = 0

    def __init__(self, frame_idx, bbox, score):
        self.id = Track._id_counter
        Track._id_counter += 1
        self.bboxes = [(frame_idx, bbox)]
        self.score = score
        self.state = 'tentative'
        self.time_since_update = 0
        self.velocity = None  # (vx, vy)
        self.predicted_bbox = bbox
        self.hits = 1

    def predict(self):
        """Simple constant-velocity motion model."""
        if len(self.bboxes) >= 2:
            (f1, b1), (f2, b2) = self.bboxes[-2], self.bboxes[-1]
            dt = max(1, f2 - f1)
            cx1 = (b1[0] + b1[2]) / 2.0
            cy1 = (b1[1] + b1[3]) / 2.0
            cx2 = (b2[0] + b2[2]) / 2.0
            cy2 = (b2[1] + b2[3]) / 2.0
            self.velocity = ((cx2 - cx1) / dt, (cy2 - cy1) / dt)
        else:
            self.velocity = (0.0, 0.0)
        last_bbox = self.bboxes[-1][1]
        cx = (last_bbox[0] + last_bbox[2]) / 2.0 + self.velocity[0]
        cy = (last_bbox[1] + last_bbox[3]) / 2.0 + self.velocity[1]
        w = last_bbox[2] - last_bbox[0]
        h = last_bbox[3] - last_bbox[1]
        self.predicted_bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        self.time_since_update += 1

    def update(self, frame_idx, bbox, score):
        self.bboxes.append((frame_idx, bbox))
        self.score = score
        self.time_since_update = 0
        self.hits += 1


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------

class ByteTrackTracker:
    def __init__(self, high_thr=0.6, low_thr=0.1, iou_thr=0.3, max_age=30, n_init=3, image_height=640):
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.n_init = n_init
        self.image_height = image_height
        self.tracks = []

    def reset(self):
        self.tracks = []
        Track._id_counter = 0

    def _mark_confirmed(self, t):
        if t.hits >= self.n_init:
            t.state = 'confirmed'

    def step(self, detections, frame_idx):
        D_high = [d for d in detections if d['score'] > self.high_thr]
        D_low = [d for d in detections if self.low_thr <= d['score'] <= self.high_thr]

        for t in self.tracks:
            t.predict()

        tracks_all = [t for t in self.tracks if t.state != 'removed']

        matched, unmatched_tracks, unmatched_dets = associate(tracks_all, D_high, self.iou_thr)
        for t_idx, d_idx in matched:
            t = tracks_all[t_idx]
            d = D_high[d_idx]
            t.update(frame_idx, d['bbox'], d['score'])
            self._mark_confirmed(t)

        unmatched_active = [t_idx for t_idx in unmatched_tracks if tracks_all[t_idx].state in ('active', 'tentative')]
        matched_low_track_ids = set()
        if len(unmatched_active) > 0 and len(D_low) > 0:
            t_list = [tracks_all[i] for i in unmatched_active]
            matched2, _, _ = associate(t_list, D_low, self.iou_thr)
            for local_t, d_idx in matched2:
                t = t_list[local_t]
                d = D_low[d_idx]
                t.update(frame_idx, d['bbox'], d['score'])
                matched_low_track_ids.add(t.id)
                self._mark_confirmed(t)

        for t_idx in unmatched_tracks:
            t = tracks_all[t_idx]
            if t.id in matched_low_track_ids:
                continue
            if t.state == 'tentative':
                t.state = 'removed'
            elif t.state == 'confirmed':
                t.state = 'lost'
                if t.time_since_update > self.max_age:
                    t.state = 'removed'
            elif t.state == 'lost':
                if t.time_since_update > self.max_age:
                    t.state = 'removed'

        for d_idx in unmatched_dets:
            d = D_high[d_idx]
            self.tracks.append(Track(frame_idx, d['bbox'], d['score']))

        self.tracks = [t for t in self.tracks if t.state != 'removed']

    def get_results(self, frame_idx):
        res = []
        for t in self.tracks:
            if t.state in ('tentative', 'confirmed', 'lost'):
                res.append((t.id, t.predicted_bbox))
        return res


class SparseTrackTracker:
    def __init__(self, high_thr=0.6, low_thr=0.1, iou_thr=0.3, max_age=30, n_init=3,
                 k_high=2, k_low=8, image_height=640):
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.n_init = n_init
        self.k_high = k_high
        self.k_low = k_low
        self.image_height = image_height
        self.tracks = []

    def reset(self):
        self.tracks = []
        Track._id_counter = 0

    def _mark_confirmed(self, t):
        if t.hits >= self.n_init:
            t.state = 'confirmed'

    def step(self, detections, frame_idx):
        D_high = [d for d in detections if d['score'] > self.high_thr]
        D_low = [d for d in detections if self.low_thr <= d['score'] <= self.high_thr]

        for t in self.tracks:
            t.predict()

        tracks_all = [t for t in self.tracks if t.state != 'removed']

        matched, unmatched_tracks, unmatched_dets = depth_cascade_match(
            tracks_all, D_high, self.k_high, self.iou_thr, self.image_height)
        for t_idx, d_idx in matched:
            t = tracks_all[t_idx]
            d = D_high[d_idx]
            t.update(frame_idx, d['bbox'], d['score'])
            self._mark_confirmed(t)

        unmatched_active = [t_idx for t_idx in unmatched_tracks if tracks_all[t_idx].state in ('active', 'tentative')]
        matched_low_track_ids = set()
        if len(unmatched_active) > 0 and len(D_low) > 0:
            t_list = [tracks_all[i] for i in unmatched_active]
            matched2, _, _ = depth_cascade_match(
                t_list, D_low, self.k_low, self.iou_thr, self.image_height)
            for local_t, d_idx in matched2:
                t = t_list[local_t]
                d = D_low[d_idx]
                t.update(frame_idx, d['bbox'], d['score'])
                matched_low_track_ids.add(t.id)
                self._mark_confirmed(t)

        for t_idx in unmatched_tracks:
            t = tracks_all[t_idx]
            if t.id in matched_low_track_ids:
                continue
            if t.state == 'tentative':
                t.state = 'removed'
            elif t.state == 'confirmed':
                t.state = 'lost'
                if t.time_since_update > self.max_age:
                    t.state = 'removed'
            elif t.state == 'lost':
                if t.time_since_update > self.max_age:
                    t.state = 'removed'

        for d_idx in unmatched_dets:
            d = D_high[d_idx]
            self.tracks.append(Track(frame_idx, d['bbox'], d['score']))

        self.tracks = [t for t in self.tracks if t.state != 'removed']

    def get_results(self, frame_idx):
        res = []
        for t in self.tracks:
            if t.state in ('tentative', 'confirmed', 'lost'):
                res.append((t.id, t.predicted_bbox))
        return res


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_tracker(tracker, data):
    tracker.reset()
    predictions = []
    for frame in data:
        tracker.step(frame['detections'], frame['frame'])
        preds = tracker.get_results(frame['frame'])
        predictions.append(preds)
    return predictions


def evaluate(predictions, data, iou_threshold=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    gt_ids_per_frame = []
    tracker_ids_per_frame = []
    unique_tids = set()
    for preds in predictions:
        for tid, _ in preds:
            unique_tids.add(tid)
    tracker_id_map = {tid: i for i, tid in enumerate(sorted(unique_tids))}
    num_tracker_ids = len(tracker_id_map)
    num_gt_ids = max(max(frame['gt_ids']) for frame in data) + 1

    similarity_scores_per_frame = []
    total_gt = 0
    total_pred = 0
    for frame, preds in zip(data, predictions):
        gt_boxes = np.array(frame['gt_bboxes'])
        gt_ids = np.array(frame['gt_ids'])
        pred_boxes = np.array([p[1] for p in preds])
        pred_ids = np.array([p[0] for p in preds])
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gb in enumerate(gt_boxes):
                for j, pb in enumerate(pred_boxes):
                    iou_mat[i, j] = box_iou(gb, pb)
            dists = 1.0 - iou_mat
            dists = np.where(dists > (1.0 - iou_threshold), np.nan, dists)
        else:
            dists = np.empty((len(gt_boxes), len(pred_boxes)))
        acc.update(gt_ids.tolist(), pred_ids.tolist(), dists)

        gt_ids_per_frame.append(gt_ids)
        tracker_ids_per_frame.append(np.array([tracker_id_map[tid] for tid in pred_ids]))
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            sim = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gb in enumerate(gt_boxes):
                for j, pb in enumerate(pred_boxes):
                    sim[i, j] = box_iou(gb, pb)
        else:
            sim = np.empty((len(gt_boxes), len(pred_boxes)))
        similarity_scores_per_frame.append(sim)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics + ['idf1'], name='seq')
    mota = summary['mota'].values[0]
    idf1 = summary['idf1'].values[0]
    num_fp = summary['num_false_positives'].values[0]
    num_fn = summary['num_misses'].values[0]
    num_ids = summary['num_switches'].values[0]

    hota_data = {
        'num_tracker_dets': total_pred,
        'num_gt_dets': total_gt,
        'num_gt_ids': num_gt_ids,
        'num_tracker_ids': num_tracker_ids,
        'gt_ids': gt_ids_per_frame,
        'tracker_ids': tracker_ids_per_frame,
        'similarity_scores': similarity_scores_per_frame,
    }
    hota_metric = HOTA()
    hota_res = hota_metric.eval_sequence(hota_data)
    hota_mean = float(np.mean(hota_res['HOTA']))
    deta_mean = float(np.mean(hota_res['DetA']))
    assa_mean = float(np.mean(hota_res['AssA']))

    return {
        'MOTA': mota,
        'IDF1': idf1,
        'HOTA': hota_mean,
        'DetA': deta_mean,
        'AssA': assa_mean,
        'FP': int(num_fp),
        'FN': int(num_fn),
        'IDs': int(num_ids),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)

    with open('data/simulated_sequence.json', 'r') as f:
        data = json.load(f)

    max_y = 0.0
    max_x = 0.0
    for frame in data:
        for b in frame['gt_bboxes']:
            max_y = max(max_y, b[3])
            max_x = max(max_x, b[2])
        for d in frame['detections']:
            max_y = max(max_y, d['bbox'][3])
            max_x = max(max_x, d['bbox'][2])
    image_height = int(np.ceil(max_y)) + 1
    image_width = int(np.ceil(max_x)) + 1
    print(f'Estimated image size: {image_width}x{image_height}')

    stats = {
        'num_frames': len(data),
        'num_gt_objects': len(set(frame['gt_ids'][0] for frame in data)),
        'avg_detections_per_frame': float(np.mean([len(f['detections']) for f in data])),
        'image_width': image_width,
        'image_height': image_height,
    }
    with open('outputs/data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Hyperparameters chosen after quick exploration
    high_thr = 0.15
    low_thr = 0.1
    iou_thr = 0.3
    max_age = 30
    n_init = 3
    k_high = 1

    bt = ByteTrackTracker(high_thr=high_thr, low_thr=low_thr, iou_thr=iou_thr,
                          max_age=max_age, n_init=n_init, image_height=image_height)
    preds_bt = run_tracker(bt, data)
    m_bt = evaluate(preds_bt, data, iou_threshold=0.5)
    print('\nByteTrack', m_bt)

    st4 = SparseTrackTracker(high_thr=high_thr, low_thr=low_thr, iou_thr=iou_thr,
                             max_age=max_age, n_init=n_init, k_high=k_high, k_low=4,
                             image_height=image_height)
    preds_st4 = run_tracker(st4, data)
    m_st4 = evaluate(preds_st4, data, iou_threshold=0.5)
    print('SparseTrack (k_low=4)', m_st4)

    st8 = SparseTrackTracker(high_thr=high_thr, low_thr=low_thr, iou_thr=iou_thr,
                             max_age=max_age, n_init=n_init, k_high=k_high, k_low=8,
                             image_height=image_height)
    preds_st8 = run_tracker(st8, data)
    m_st8 = evaluate(preds_st8, data, iou_threshold=0.5)
    print('SparseTrack (k_low=8)', m_st8)

    results = {
        'ByteTrack': m_bt,
        'SparseTrack (k_low=4)': m_st4,
        'SparseTrack (k_low=8)': m_st8,
    }
    preds_all = {
        'ByteTrack': preds_bt,
        'SparseTrack (k_low=4)': preds_st4,
        'SparseTrack (k_low=8)': preds_st8,
    }

    with open('outputs/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('outputs/predictions.json', 'w') as f:
        out = {}
        for name, preds in preds_all.items():
            out[name] = [[{'id': int(tid), 'bbox': [float(x) for x in bbox]}
                          for tid, bbox in frame_preds] for frame_preds in preds]
        json.dump(out, f, indent=2)

    # Figures
    names = list(results.keys())

    # 1. Metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.2
    for i, metric in enumerate(['MOTA', 'IDF1', 'HOTA', 'AssA']):
        vals = [results[n][metric] * 100.0 for n in names]
        ax.bar(x + i * width, vals, width, label=metric)
    ax.set_ylabel('Score (%)')
    ax.set_title('Tracking Performance Comparison (high_thr=0.15)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/fig_metrics_comparison.png', dpi=150)
    plt.close()

    # 2. Error counts
    fig, ax = plt.subplots(figsize=(8, 5))
    errors = ['FP', 'FN', 'IDs']
    x = np.arange(len(names))
    width = 0.25
    for i, err in enumerate(errors):
        vals = [results[n][err] for n in names]
        ax.bar(x + i * width, vals, width, label=err)
    ax.set_ylabel('Count')
    ax.set_title('Error Counts Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/fig_error_counts.png', dpi=150)
    plt.close()

    # 3. Score distribution
    scores = []
    for frame in data:
        for d in frame['detections']:
            scores.append(d['score'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=50, color='steelblue', edgecolor='black')
    ax.axvline(high_thr, color='red', linestyle='--', label=f'High threshold ({high_thr})')
    ax.axvline(0.9, color='orange', linestyle='--', label='Score mode (0.9)')
    ax.set_xlabel('Detection Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Detection Confidence Scores')
    ax.legend()
    plt.tight_layout()
    plt.savefig('report/images/fig_score_distribution.png', dpi=150)
    plt.close()

    # 4. Pseudo-depth vs box area (scatter)
    areas = []
    depths = []
    for b in data[0]['gt_bboxes']:
        areas.append((b[2] - b[0]) * (b[3] - b[1]))
        depths.append(pseudo_depth(b, image_height))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(depths, areas, alpha=0.5, s=20)
    ax.set_xlabel('Pseudo-Depth (pixels)')
    ax.set_ylabel('Bounding Box Area (pixels²)')
    ax.set_title('Pseudo-Depth vs. Box Area (Frame 0)')
    plt.tight_layout()
    plt.savefig('report/images/fig_pseudo_depth_distribution.png', dpi=150)
    plt.close()

    # 5. Ablation: high_thr vs MOTA/IDF1
    ablation_thrs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    ablation_bt = {'MOTA': [], 'IDF1': [], 'HOTA': []}
    ablation_st = {'MOTA': [], 'IDF1': [], 'HOTA': []}
    for thr in ablation_thrs:
        bt2 = ByteTrackTracker(high_thr=thr, low_thr=0.1, iou_thr=0.3, max_age=30, n_init=3, image_height=image_height)
        m = evaluate(run_tracker(bt2, data), data, iou_threshold=0.5)
        ablation_bt['MOTA'].append(m['MOTA'] * 100)
        ablation_bt['IDF1'].append(m['IDF1'] * 100)
        ablation_bt['HOTA'].append(m['HOTA'] * 100)
        st2 = SparseTrackTracker(high_thr=thr, low_thr=0.1, iou_thr=0.3, max_age=30, n_init=3, k_high=1, k_low=8, image_height=image_height)
        m = evaluate(run_tracker(st2, data), data, iou_threshold=0.5)
        ablation_st['MOTA'].append(m['MOTA'] * 100)
        ablation_st['IDF1'].append(m['IDF1'] * 100)
        ablation_st['HOTA'].append(m['HOTA'] * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ablation_thrs, ablation_bt['MOTA'], marker='o', label='ByteTrack MOTA')
    ax.plot(ablation_thrs, ablation_st['MOTA'], marker='s', label='SparseTrack MOTA')
    ax.set_xlabel('High-Score Threshold')
    ax.set_ylabel('MOTA (%)')
    ax.set_title('Ablation: High-Score Threshold vs. MOTA')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/fig_ablation_mota.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ablation_thrs, ablation_bt['IDF1'], marker='o', label='ByteTrack IDF1')
    ax.plot(ablation_thrs, ablation_st['IDF1'], marker='s', label='SparseTrack IDF1')
    ax.set_xlabel('High-Score Threshold')
    ax.set_ylabel('IDF1 (%)')
    ax.set_title('Ablation: High-Score Threshold vs. IDF1')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/fig_ablation_idf1.png', dpi=150)
    plt.close()

    # 6. Per-frame ID switches
    def id_switches_per_frame(preds, data):
        acc = mm.MOTAccumulator(auto_id=True)
        for frame, pr in zip(data, preds):
            gt_boxes = np.array(frame['gt_bboxes'])
            gt_ids = frame['gt_ids']
            pred_boxes = np.array([p[1] for p in pr])
            pred_ids = [p[0] for p in pr]
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
                for i, gb in enumerate(gt_boxes):
                    for j, pb in enumerate(pred_boxes):
                        iou_mat[i, j] = box_iou(gb, pb)
                dists = 1.0 - iou_mat
                dists = np.where(dists > 0.5, np.nan, dists)
            else:
                dists = np.empty((len(gt_boxes), len(pred_boxes)))
            acc.update(gt_ids, pred_ids, dists)
        events = acc.events
        switch_counts = []
        for fid in events.index.get_level_values(0).unique():
            frame_events = events.loc[fid]
            if isinstance(frame_events, pd.Series):
                frame_events = frame_events.to_frame().T
            switch_counts.append(int((frame_events['Type'] == 'SWITCH').sum()))
        return switch_counts

    import pandas as pd
    bt_switches = id_switches_per_frame(preds_all['ByteTrack'], data)
    st_switches = id_switches_per_frame(preds_all['SparseTrack (k_low=8)'], data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt_switches, label='ByteTrack', marker='o', markersize=3)
    ax.plot(st_switches, label='SparseTrack', marker='s', markersize=3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('ID Switches')
    ax.set_title('Per-Frame ID Switches')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/fig_id_switches_per_frame.png', dpi=150)
    plt.close()

    print('\nAll done. Figures saved to report/images/')


if __name__ == '__main__':
    main()
