"""
Final Multi-Object Tracking Implementation with proper analysis.
Key insight: SparseTrack decomposes the dense matching problem into smaller 
sub-problems by depth, reducing ambiguity in crowded scenes.
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# ============================================================
# Utility functions
# ============================================================

def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def iou_matrix(bboxes1, bboxes2):
    n, m = len(bboxes1), len(bboxes2)
    mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            mat[i, j] = iou(bboxes1[i], bboxes2[j])
    return mat

def linear_assignment_solve(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    matched = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= thresh:
            matched.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)
    return matched, list(unmatched_rows), list(unmatched_cols)


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox, track_id=None):
        self.id = track_id if track_id is not None else KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 10
        self.R = np.eye(4) * 1.0
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100
        self.F = np.eye(8)
        self.F[0, 4] = 1; self.F[1, 5] = 1; self.F[2, 6] = 1; self.F[3, 7] = 1
        self.H = np.zeros((4, 8))
        np.fill_diagonal(self.H, 1)
    
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()
    
    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
    
    def get_bbox(self):
        cx, cy, w, h = self.state[:4]
        w = max(w, 1); h = max(h, 1)
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def estimate_pseudo_depth(bbox):
    """Bottom y-coordinate as depth proxy (larger = closer to camera)."""
    return bbox[3]

def assign_depth_layers(bboxes, n_layers=3):
    if len(bboxes) == 0:
        return {i: [] for i in range(n_layers)}
    depths = np.array([estimate_pseudo_depth(b) for b in bboxes])
    if len(depths) < n_layers:
        return {0: list(range(len(bboxes)))}
    percentiles = np.linspace(0, 100, n_layers + 1)
    boundaries = np.percentile(depths, percentiles)
    boundaries[-1] = depths.max() + 1
    layers = {i: [] for i in range(n_layers)}
    for idx in range(len(depths)):
        for layer in range(n_layers):
            if depths[idx] <= boundaries[layer + 1]:
                layers[layer].append(idx)
                break
    return layers


# ============================================================
# SORT Tracker
# ============================================================
class SORTTracker:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.2, det_threshold=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_threshold = det_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
    
    def update(self, detections):
        self.frame_count += 1
        dets = [d for d in detections if d['score'] >= self.det_threshold]
        det_bboxes = [d['bbox'] for d in dets]
        
        for t in self.trackers:
            t.predict()
        predicted = [t.get_bbox() for t in self.trackers]
        
        if len(predicted) > 0 and len(det_bboxes) > 0:
            iou_mat = iou_matrix(predicted, det_bboxes)
            matched, unmatched_t, unmatched_d = linear_assignment_solve(iou_mat, self.iou_threshold)
        else:
            matched = []; unmatched_t = list(range(len(self.trackers))); unmatched_d = list(range(len(det_bboxes)))
        
        for ti, di in matched:
            self.trackers[ti].update(det_bboxes[di])
        for di in unmatched_d:
            self.trackers.append(KalmanBoxTracker(det_bboxes[di]))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        results = []
        for t in self.trackers:
            if (t.hits >= self.min_hits or self.frame_count <= self.min_hits) and t.time_since_update == 0:
                results.append((t.id, t.get_bbox()))
        return results


# ============================================================
# ByteTrack Tracker
# ============================================================
class ByteTracker:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.2,
                 high_threshold=0.25, low_threshold=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
    
    def update(self, detections):
        self.frame_count += 1
        high_dets = [d for d in detections if d['score'] >= self.high_threshold]
        low_dets = [d for d in detections if self.low_threshold <= d['score'] < self.high_threshold]
        high_bb = [d['bbox'] for d in high_dets]
        low_bb = [d['bbox'] for d in low_dets]
        
        for t in self.trackers:
            t.predict()
        predicted = [t.get_bbox() for t in self.trackers]
        
        # First: high score
        if len(predicted) > 0 and len(high_bb) > 0:
            iou_mat = iou_matrix(predicted, high_bb)
            matched, unmatched_t, unmatched_d = linear_assignment_solve(iou_mat, self.iou_threshold)
        else:
            matched = []; unmatched_t = list(range(len(self.trackers))); unmatched_d = list(range(len(high_bb)))
        
        for ti, di in matched:
            self.trackers[ti].update(high_bb[di])
        
        # Second: low score for remaining tracks
        remaining = [self.trackers[i] for i in unmatched_t]
        rem_pred = [t.get_bbox() for t in remaining]
        
        if len(rem_pred) > 0 and len(low_bb) > 0:
            iou_mat2 = iou_matrix(rem_pred, low_bb)
            matched2, _, _ = linear_assignment_solve(iou_mat2, self.iou_threshold)
            for tl, dl in matched2:
                remaining[tl].update(low_bb[dl])
        
        for di in unmatched_d:
            self.trackers.append(KalmanBoxTracker(high_bb[di]))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        results = []
        for t in self.trackers:
            if (t.hits >= self.min_hits or self.frame_count <= self.min_hits) and t.time_since_update == 0:
                results.append((t.id, t.get_bbox()))
        return results


# ============================================================
# SparseTrack Tracker
# ============================================================
class SparseTracker:
    """
    SparseTrack: Decomposes dense target sets into sparse subsets via pseudo-depth
    estimation and performs hierarchical association within each depth layer.
    
    Key steps:
    1. Estimate pseudo-depth from bbox bottom y-coordinate
    2. Assign both tracks and detections to depth layers
    3. Perform Hungarian matching within each layer (smaller, sparser problems)
    4. Cross-layer fallback for unmatched entities
    5. Second-stage low-score association (ByteTrack-style)
    """
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.2,
                 high_threshold=0.25, low_threshold=0.1, n_depth_layers=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.n_depth_layers = n_depth_layers
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
        self.layer_stats = defaultdict(lambda: {'matched': 0, 'total_tracks': 0, 'total_dets': 0})
    
    def update(self, detections):
        self.frame_count += 1
        
        # Split detections by confidence
        high_dets = [d for d in detections if d['score'] >= self.high_threshold]
        low_dets = [d for d in detections if self.low_threshold <= d['score'] < self.high_threshold]
        high_bb = [d['bbox'] for d in high_dets]
        low_bb = [d['bbox'] for d in low_dets]
        
        # Predict
        for t in self.trackers:
            t.predict()
        predicted = [t.get_bbox() for t in self.trackers]
        
        # === DEPTH-BASED HIERARCHICAL ASSOCIATION ===
        # Assign tracks and high-score detections to depth layers
        track_layers = assign_depth_layers(predicted, self.n_depth_layers) if len(predicted) > 0 else {}
        det_layers = assign_depth_layers(high_bb, self.n_depth_layers) if len(high_bb) > 0 else {}
        
        matched_tracks = set()
        matched_dets = set()
        
        # Layer-wise matching
        for layer in range(self.n_depth_layers):
            t_idx = [i for i in track_layers.get(layer, []) if i not in matched_tracks]
            d_idx = [i for i in det_layers.get(layer, []) if i not in matched_dets]
            
            self.layer_stats[layer]['total_tracks'] += len(t_idx)
            self.layer_stats[layer]['total_dets'] += len(d_idx)
            
            if not t_idx or not d_idx:
                continue
            
            t_bb = [predicted[i] for i in t_idx]
            d_bb = [high_bb[i] for i in d_idx]
            
            iou_mat = iou_matrix(t_bb, d_bb)
            matches, _, _ = linear_assignment_solve(iou_mat, self.iou_threshold)
            
            self.layer_stats[layer]['matched'] += len(matches)
            
            for tl, dl in matches:
                tg = t_idx[tl]
                dg = d_idx[dl]
                self.trackers[tg].update(high_bb[dg])
                matched_tracks.add(tg)
                matched_dets.add(dg)
        
        # Cross-layer fallback for unmatched
        um_t = [i for i in range(len(self.trackers)) if i not in matched_tracks]
        um_d = [i for i in range(len(high_bb)) if i not in matched_dets]
        
        if um_t and um_d:
            t_bb = [predicted[i] for i in um_t]
            d_bb = [high_bb[i] for i in um_d]
            iou_mat = iou_matrix(t_bb, d_bb)
            matches, _, um_d_remaining = linear_assignment_solve(iou_mat, self.iou_threshold)
            for tl, dl in matches:
                self.trackers[um_t[tl]].update(high_bb[um_d[dl]])
                matched_tracks.add(um_t[tl])
                matched_dets.add(um_d[dl])
            um_d_final = [um_d[i] for i in um_d_remaining]
        else:
            um_d_final = um_d
        
        # Second stage: low-score association for remaining tracks
        um_t2 = [i for i in range(len(self.trackers)) if i not in matched_tracks]
        if um_t2 and low_bb:
            t_bb = [self.trackers[i].get_bbox() for i in um_t2]
            iou_mat = iou_matrix(t_bb, low_bb)
            matches2, _, _ = linear_assignment_solve(iou_mat, self.iou_threshold)
            for tl, dl in matches2:
                self.trackers[um_t2[tl]].update(low_bb[dl])
                matched_tracks.add(um_t2[tl])
        
        # Init new tracks from unmatched high-score detections
        for di in um_d_final:
            self.trackers.append(KalmanBoxTracker(high_bb[di]))
        
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        results = []
        for t in self.trackers:
            if (t.hits >= self.min_hits or self.frame_count <= self.min_hits) and t.time_since_update == 0:
                results.append((t.id, t.get_bbox()))
        return results


# ============================================================
# Evaluation
# ============================================================

def compute_mot_metrics(gt_data, tracking_results):
    total_gt = total_tp = total_fp = total_fn = total_idsw = 0
    id_mapping = {}
    gt_tracked_frames = defaultdict(list)
    gt_total_frames = defaultdict(int)
    per_frame = []
    
    for fd in gt_data:
        fid = fd['frame']
        gt_bb = fd['gt_bboxes']
        gt_ids = fd['gt_ids']
        tracks = tracking_results.get(fid, [])
        t_bb = [t[1] for t in tracks]
        t_ids = [t[0] for t in tracks]
        
        n_gt = len(gt_bb)
        total_gt += n_gt
        for gid in gt_ids:
            gt_total_frames[gid] += 1
        
        f_idsw = 0
        if n_gt == 0 and len(t_bb) == 0:
            per_frame.append({'frame': fid, 'tp': 0, 'fp': 0, 'fn': 0, 'idsw': 0}); continue
        if n_gt == 0:
            total_fp += len(t_bb); per_frame.append({'frame': fid, 'tp': 0, 'fp': len(t_bb), 'fn': 0, 'idsw': 0}); continue
        if len(t_bb) == 0:
            total_fn += n_gt; per_frame.append({'frame': fid, 'tp': 0, 'fp': 0, 'fn': n_gt, 'idsw': 0}); continue
        
        iou_mat = iou_matrix(gt_bb, t_bb)
        matched, um_gt, um_trk = linear_assignment_solve(iou_mat, 0.5)
        
        f_tp = len(matched); f_fp = len(um_trk); f_fn = len(um_gt)
        total_tp += f_tp; total_fp += f_fp; total_fn += f_fn
        
        for gl, tl in matched:
            gid = gt_ids[gl]; tid = t_ids[tl]
            gt_tracked_frames[gid].append(fid)
            if gid in id_mapping and id_mapping[gid] != tid:
                total_idsw += 1; f_idsw += 1
            id_mapping[gid] = tid
        
        per_frame.append({'frame': fid, 'tp': f_tp, 'fp': f_fp, 'fn': f_fn, 'idsw': f_idsw})
    
    mota = 1 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    idf1 = 2 * prec * rec / max(prec + rec, 1e-6)
    
    mt = ml = pt = 0
    for gid in gt_total_frames:
        r = len(gt_tracked_frames[gid]) / gt_total_frames[gid]
        if r >= 0.8: mt += 1
        elif r <= 0.2: ml += 1
        else: pt += 1
    
    det_a = total_tp / max(total_tp + total_fn + total_fp, 1)
    ass_a = total_tp / max(total_tp + total_idsw, 1)
    hota = np.sqrt(det_a * ass_a)
    
    return {
        'MOTA': round(mota, 4), 'IDF1': round(idf1, 4), 'HOTA': round(hota, 4),
        'Precision': round(prec, 4), 'Recall': round(rec, 4),
        'TP': int(total_tp), 'FP': int(total_fp), 'FN': int(total_fn),
        'ID_Switches': int(total_idsw), 'MT': int(mt), 'ML': int(ml), 'PT': int(pt),
        'Num_GT_Objects': len(gt_total_frames), 'per_frame': per_frame
    }


def run_tracker(tracker, data):
    results = {}
    for fd in data:
        results[fd['frame']] = tracker.update(fd['detections'])
    return results


def main():
    with open('data/simulated_sequence.json') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} frames, {len(data[0]['gt_bboxes'])} objects/frame")
    print(f"Avg detections/frame: {np.mean([len(d['detections']) for d in data]):.1f}")
    
    # Score analysis
    all_scores = [d['score'] for fd in data for d in fd['detections']]
    print(f"Score distribution: mean={np.mean(all_scores):.3f}, median={np.median(all_scores):.3f}")
    print(f"  25th={np.percentile(all_scores, 25):.3f}, 75th={np.percentile(all_scores, 75):.3f}")
    
    configs = {
        'SORT': lambda: SORTTracker(max_age=30, min_hits=1, iou_threshold=0.2, det_threshold=0.1),
        'ByteTrack': lambda: ByteTracker(max_age=30, min_hits=1, iou_threshold=0.2,
                                          high_threshold=0.25, low_threshold=0.1),
        'SparseTrack_3L': lambda: SparseTracker(max_age=30, min_hits=1, iou_threshold=0.2,
                                                  high_threshold=0.25, low_threshold=0.1, n_depth_layers=3),
        'SparseTrack_5L': lambda: SparseTracker(max_age=30, min_hits=1, iou_threshold=0.2,
                                                  high_threshold=0.25, low_threshold=0.1, n_depth_layers=5),
        'SparseTrack_7L': lambda: SparseTracker(max_age=30, min_hits=1, iou_threshold=0.2,
                                                  high_threshold=0.25, low_threshold=0.1, n_depth_layers=7),
        'SparseTrack_10L': lambda: SparseTracker(max_age=30, min_hits=1, iou_threshold=0.2,
                                                   high_threshold=0.25, low_threshold=0.1, n_depth_layers=10),
    }
    
    all_results = {}
    all_metrics = {}
    
    for name, make_tracker in configs.items():
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        tracker = make_tracker()
        results = run_tracker(tracker, data)
        all_results[name] = results
        metrics = compute_mot_metrics(data, results)
        all_metrics[name] = metrics
        
        print(f"  MOTA={metrics['MOTA']:.4f}  IDF1={metrics['IDF1']:.4f}  HOTA={metrics['HOTA']:.4f}")
        print(f"  Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}")
        print(f"  IDsw={metrics['ID_Switches']}  FP={metrics['FP']}  FN={metrics['FN']}")
        print(f"  MT={metrics['MT']}  ML={metrics['ML']}  PT={metrics['PT']}")
        
        if hasattr(tracker, 'layer_stats'):
            for layer in sorted(tracker.layer_stats.keys()):
                s = tracker.layer_stats[layer]
                print(f"    L{layer}: matched={s['matched']}, tracks={s['total_tracks']}, dets={s['total_dets']}")
    
    # Save
    save_m = {n: {k: v for k, v in m.items() if k != 'per_frame'} for n, m in all_metrics.items()}
    with open('outputs/tracking_metrics.json', 'w') as f:
        json.dump(save_m, f, indent=2)
    
    pf = {n: m['per_frame'] for n, m in all_metrics.items()}
    with open('outputs/per_frame_metrics.json', 'w') as f:
        json.dump(pf, f, indent=2)
    
    sr = {}
    for name, results in all_results.items():
        sr[name] = {}
        for fid, tracks in results.items():
            sr[name][str(fid)] = [{'id': int(t[0]), 'bbox': [round(x, 2) for x in t[1]]} for t in tracks]
    with open('outputs/tracking_results.json', 'w') as f:
        json.dump(sr, f, indent=2)
    
    print("\nAll results saved.")
    return all_metrics, all_results, data


if __name__ == '__main__':
    main()
