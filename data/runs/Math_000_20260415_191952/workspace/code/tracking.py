"""
Multi-Object Tracking: ByteTrack vs SparseTrack
Implements both trackers on simulated sequence data and evaluates performance.
"""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os

def load_data(path='data/simulated_sequence.json'):
    with open(path) as f:
        return json.load(f)

class SimpleKalmanFilter:
    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        self.state = np.array([
            (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
            0, 0, 0, 0
        ], dtype=float)
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4) * 1.0
        self.P = np.eye(8) * 3.0
        self.F = np.eye(8)
        self.F[0, 4] = 1; self.F[1, 5] = 1; self.F[2, 6] = 1; self.F[3, 7] = 1
        self.H = np.eye(4, 8)
    
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._bbox()
    
    def update(self, bbox):
        z = np.array(bbox)
        y_m = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_m
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self._bbox()
    
    def _bbox(self):
        xc, yc, w, h = self.state[:4]
        return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]
    
    def get_bbox(self):
        return self._bbox()

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    iw = max(0, xi2 - xi1); ih = max(0, yi2 - yi1)
    inter = iw * ih
    a1 = max(0, (box1[2]-box1[0])*(box1[3]-box1[1]))
    a2 = max(0, (box2[2]-box2[0])*(box2[3]-box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def hungarian_match(pred_bboxes, det_bboxes, iou_thresh):
    if not pred_bboxes or not det_bboxes:
        return [], list(range(len(pred_bboxes))), list(range(len(det_bboxes)))
    n_p = len(pred_bboxes); n_d = len(det_bboxes)
    cost = np.zeros((n_p, n_d))
    for i in range(n_p):
        for j in range(n_d):
            cost[i, j] = 1.0 - compute_iou(pred_bboxes[i], det_bboxes[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []; mr = set(); mc = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < (1.0 - iou_thresh):
            matches.append((r, c)); mr.add(r); mc.add(c)
    ur = [i for i in range(n_p) if i not in mr]
    uc = [i for i in range(n_d) if i not in mc]
    return matches, ur, uc

# ============================================================
# ByteTrack
# ============================================================
class ByteTrack:
    def __init__(self, high_thresh=0.6, iou_high=0.5, iou_low=0.3, max_age=10):
        self.high_thresh = high_thresh
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
    
    def _new_track(self, det, fn):
        tid = self.next_id; self.next_id += 1
        kf = SimpleKalmanFilter(det['bbox'])
        self.tracks[tid] = {
            'kf': kf, 'age': 0, 'hits': 1, 'last_seen': fn,
            'trajectory': [(fn, det['bbox'], det['score'])]
        }
        return tid
    
    def update(self, fn, detections):
        high = [d for d in detections if d['score'] >= self.high_thresh]
        low = [d for d in detections if d['score'] < self.high_thresh]
        
        active = {tid: t for tid, t in self.tracks.items() if t['age'] <= self.max_age}
        tids = list(active.keys()); tlist = [active[tid] for tid in tids]
        
        # Stage 1: match high-score detections
        pred = [t['kf'].get_bbox() for t in tlist]
        db = [d['bbox'] for d in high]
        m1, ut1, ud1 = hungarian_match(pred, db, self.iou_high)
        
        # Stage 2: match remaining tracks with low-score detections
        rt = [tlist[i] for i in ut1]; rtids = [tids[i] for i in ut1]
        rp = [t['kf'].get_bbox() for t in rt]
        rl = [d['bbox'] for d in low]
        m2, _, _ = hungarian_match(rp, rl, self.iou_low)
        
        matched_tids = set(); matched_did_h = set(); matched_did_l = set()
        
        for ti, di in m1:
            tid = tids[ti]; det = high[di]; tr = active[tid]
            tr['kf'].update(det['bbox'])
            tr['trajectory'].append((fn, det['bbox'], det['score']))
            tr['hits'] += 1; tr['last_seen'] = fn; tr['age'] = 0
            matched_tids.add(tid); matched_did_h.add(id(det))
        
        for ti, di in m2:
            tid = rtids[ti]; det = low[di]; tr = active[tid]
            tr['kf'].update(det['bbox'])
            tr['trajectory'].append((fn, det['bbox'], det['score']))
            tr['hits'] += 1; tr['last_seen'] = fn; tr['age'] = 0
            matched_tids.add(tid); matched_did_l.add(id(det))
        
        for tid, tr in active.items():
            if tid not in matched_tids:
                tr['kf'].predict(); tr['age'] += 1
        
        for d in high:
            if id(d) not in matched_did_h:
                self._new_track(d, fn)
        for d in low:
            if id(d) not in matched_did_l:
                self._new_track(d, fn)
        
        return self.get_trajectories()
    
    def get_trajectories(self):
        return {tid: {'trajectory': t['trajectory'], 'last_frame': t['last_seen'],
                      'hits': t['hits']}
                for tid, t in self.tracks.items() if t['age'] <= self.max_age}

# ============================================================
# SparseTrack (Pseudo-Depth Hierarchical)
# ============================================================
class SparseTrack:
    def __init__(self, n_layers=4, iou_base=0.5, iou_low=0.3, max_age=10):
        self.n_layers = n_layers
        self.iou_base = iou_base
        self.iou_low = iou_low
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
    
    def _depth(self, bbox):
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        return 1.0 - min(area / (640*640), 1.0)
    
    def _layer(self, depth):
        return min(int(depth * self.n_layers), self.n_layers - 1)
    
    def _new_track(self, det, fn):
        tid = self.next_id; self.next_id += 1
        kf = SimpleKalmanFilter(det['bbox'])
        d = self._depth(det['bbox']); l = self._layer(d)
        self.tracks[tid] = {
            'kf': kf, 'age': 0, 'hits': 1, 'last_seen': fn,
            'trajectory': [(fn, det['bbox'], det['score'])], 'id': tid,
            'depth': d, 'layer': l
        }
        return tid
    
    def update(self, fn, detections):
        for det in detections:
            det['_depth'] = self._depth(det['bbox'])
            det['_layer'] = self._layer(det['_depth'])
        
        layers = defaultdict(list)
        for det in detections:
            layers[det['_layer']].append(det)
        
        active = {tid: t for tid, t in self.tracks.items() if t['age'] <= self.max_age}
        tids = list(active.keys()); tlist = [active[tid] for tid in tids]
        
        all_mt = set(); all_md = set()
        
        for li in range(self.n_layers):
            ldets = layers[li]
            if not ldets: continue
            
            lt = []; ltidx = []
            for idx, (tid, t) in enumerate(zip(tids, tlist)):
                if tid not in all_mt and abs(t.get('layer', li) - li) <= 1:
                    lt.append(t); ltidx.append(idx)
            
            if not lt: continue
            
            pred = [t['kf'].get_bbox() for t in lt]
            db = [d['bbox'] for d in ldets]
            matches, _, _ = hungarian_match(pred, db, self.iou_base)
            
            for ti, di in matches:
                tr = lt[ti]; det = ldets[di]; tid = tr['id']
                tr['kf'].update(det['bbox'])
                tr['trajectory'].append((fn, det['bbox'], det['score']))
                tr['hits'] += 1; tr['last_seen'] = fn; tr['age'] = 0
                tr['depth'] = self._depth(det['bbox'])
                tr['layer'] = self._layer(tr['depth'])
                all_mt.add(tid); all_md.add(id(det))
        
        # Cross-layer matching
        rt = []; rtidx = []
        for idx, (tid, t) in enumerate(zip(tids, tlist)):
            if tid not in all_mt:
                rt.append(t); rtidx.append(idx)
        
        rdets = [d for d in detections if id(d) not in all_md]
        if rt and rdets:
            pred = [t['kf'].get_bbox() for t in rt]
            db = [d['bbox'] for d in rdets]
            matches, _, _ = hungarian_match(pred, db, self.iou_low)
            for ti, di in matches:
                tr = rt[ti]; det = rdets[di]; tid = tr['id']
                tr['kf'].update(det['bbox'])
                tr['trajectory'].append((fn, det['bbox'], det['score']))
                tr['hits'] += 1; tr['last_seen'] = fn; tr['age'] = 0
                tr['depth'] = self._depth(det['bbox'])
                tr['layer'] = self._layer(tr['depth'])
                all_mt.add(tid); all_md.add(id(det))
        
        for tid, tr in active.items():
            if tid not in all_mt:
                tr['kf'].predict(); tr['age'] += 1
        
        for det in detections:
            if id(det) not in all_md:
                self._new_track(det, fn)
        
        return self.get_trajectories()
    
    def get_trajectories(self):
        return {tid: {'trajectory': t['trajectory'], 'last_frame': t['last_seen'],
                      'hits': t['hits'], 'depth': t.get('depth', 0.5), 'layer': t.get('layer', 1)}
                for tid, t in self.tracks.items() if t['age'] <= self.max_age}

# ============================================================
# Oracle Tracker (uses gt_id for upper bound)
# ============================================================
class OracleTracker:
    """Uses known gt_id to create perfect tracks."""
    def __init__(self):
        self.tracks = {}  # gt_id -> trajectory
        self.next_id = 0
    
    def update(self, fn, detections):
        detected_gids = set()
        for det in detections:
            gid = det['gt_id']
            detected_gids.add(gid)
            if gid not in self.tracks:
                self.tracks[gid] = []
            self.tracks[gid].append((fn, det['bbox'], det['score']))
        return self.get_trajectories()
    
    def get_trajectories(self):
        return {gid: {'trajectory': traj, 'last_frame': traj[-1][0],
                      'hits': len(traj)}
                for gid, traj in self.tracks.items()}

# ============================================================
# Evaluation
# ============================================================
def evaluate(trajectories, data):
    n_frames = len(data)
    gt_pf = {}
    for frame in data:
        fn = frame['frame']
        gt_pf[fn] = {gid: frame['gt_bboxes'][i] for i, gid in enumerate(frame['gt_ids'])}
    
    # Map track -> GT via detection gt_id voting
    t2g = {}
    for tid, ti in trajectories.items():
        votes = defaultdict(float)
        for fn, bbox, score in ti['trajectory']:
            for det in data[fn]['detections']:
                iou = compute_iou(bbox, det['bbox'])
                if iou > 0.5:
                    votes[det['gt_id']] += iou * score
        if votes:
            t2g[tid] = max(votes, key=votes.get)
    
    # Build per-track lookup by frame number
    track_by_frame = {}
    for tid, ti in trajectories.items():
        track_by_frame[tid] = {}
        for fn, bbox, score in ti['trajectory']:
            track_by_frame[tid][fn] = bbox
    
    tp = fp = fn_c = ids = 0
    prev_gt = {}
    
    for fn in range(n_frames):
        gids = set(gt_pf[fn].keys())
        for gid in gids:
            gt_bbox = gt_pf[fn][gid]
            best_t = None; best_iou = 0
            for tid, ti in trajectories.items():
                if t2g.get(tid) == gid:
                    if fn in track_by_frame[tid]:
                        bbox = track_by_frame[tid][fn]
                        iou = compute_iou(bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou; best_t = tid
            if best_t is not None and best_iou > 0.5:
                tp += 1
                if gid in prev_gt and prev_gt[gid] != best_t:
                    ids += 1
                prev_gt[gid] = best_t
            else:
                fn_c += 1
        
        for tid, ti in trajectories.items():
            if fn in track_by_frame[tid]:
                ag = t2g.get(tid)
                if ag is not None and ag not in gids:
                    fp += 1
    
    gta = defaultdict(list)
    for tid, gid in t2g.items():
        gta[gid].append(tid)
    frags = sum(len(v)-1 for v in gta.values() if len(v) > 1)
    
    total_gt = tp + fn_c
    mota = max(1.0 - (fp + fn_c + ids) / max(total_gt, 1), 0)
    id_correct = tp - ids
    idf1 = max(2 * id_correct / max(2 * id_correct + ids + frags, 1), 0)
    if tp > 0:
        da = tp / max(tp + fp, 1); aa = id_correct / max(tp, 1)
        hota = np.sqrt(da * aa)
    else:
        hota = 0
    
    return {'MOTA': mota, 'IDF1': idf1, 'HOTA': hota, 'ID_Switches': ids,
            'Fragments': frags, 'TP': tp, 'FP': fp, 'FN': fn_c,
            'n_tracked': len(trajectories), 'n_gt': len(gt_pf[0]),
            'track_to_gt': {str(k): v for k, v in t2g.items()}}

# ============================================================
# Per-frame analysis
# ============================================================
def per_frame_analysis(trajectories, data):
    """Compute per-frame detection rate and tracking quality."""
    n_frames = len(data)
    results = []
    
    gt_pf = {}
    for frame in data:
        fn = frame['frame']
        gt_pf[fn] = {gid: frame['gt_bboxes'][i] for i, gid in enumerate(frame['gt_ids'])}
    
    # Map track -> GT
    t2g = {}
    for tid, ti in trajectories.items():
        votes = defaultdict(float)
        for fn, bbox, score in ti['trajectory']:
            for det in data[fn]['detections']:
                iou = compute_iou(bbox, det['bbox'])
                if iou > 0.5:
                    votes[det['gt_id']] += iou * score
        if votes:
            t2g[tid] = max(votes, key=votes.get)
    
    track_by_frame = {}
    for tid, ti in trajectories.items():
        track_by_frame[tid] = {}
        for fn, bbox, score in ti['trajectory']:
            track_by_frame[tid][fn] = bbox
    
    for fn in range(n_frames):
        gids = set(gt_pf[fn].keys())
        tracked = 0
        for gid in gids:
            gt_bbox = gt_pf[fn][gid]
            for tid in trajectories:
                if t2g.get(tid) == gid and fn in track_by_frame[tid]:
                    iou = compute_iou(track_by_frame[tid][fn], gt_bbox)
                    if iou > 0.5:
                        tracked += 1
                        break
        
        n_det = len(data[fn]['detections'])
        results.append({
            'frame': fn,
            'n_gt': len(gids),
            'n_det': n_det,
            'n_tracked': tracked,
            'detection_rate': n_det / len(gids),
            'tracking_rate': tracked / len(gids)
        })
    
    return results

# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    data = load_data()
    n_frames = len(data)
    print(f"Loaded {n_frames} frames")
    
    # Run Oracle Tracker (upper bound)
    print("\n=== Running Oracle Tracker ===")
    oracle = OracleTracker()
    for frame in data:
        oracle.update(frame['frame'], frame['detections'])
    oracle_traj = oracle.get_trajectories()
    print(f"Oracle: {len(oracle_traj)} tracks")
    
    # Run ByteTrack
    print("\n=== Running ByteTrack ===")
    bt = ByteTrack(high_thresh=0.6, iou_high=0.5, iou_low=0.3, max_age=10)
    for frame in data:
        bt.update(frame['frame'], frame['detections'])
    bt_traj = bt.get_trajectories()
    print(f"ByteTrack: {len(bt_traj)} tracks")
    
    # Run SparseTrack
    print("\n=== Running SparseTrack ===")
    st = SparseTrack(n_layers=4, iou_base=0.5, iou_low=0.3, max_age=10)
    for frame in data:
        st.update(frame['frame'], frame['detections'])
    st_traj = st.get_trajectories()
    print(f"SparseTrack: {len(st_traj)} tracks")
    
    # Evaluate
    print("\n=== Oracle Metrics ===")
    om = evaluate(oracle_traj, data)
    for k, v in om.items():
        if k != 'track_to_gt':
            print(f"  {k}: {v}")
    
    print("\n=== ByteTrack Metrics ===")
    bm = evaluate(bt_traj, data)
    for k, v in bm.items():
        if k != 'track_to_gt':
            print(f"  {k}: {v}")
    
    print("\n=== SparseTrack Metrics ===")
    sm = evaluate(st_traj, data)
    for k, v in sm.items():
        if k != 'track_to_gt':
            print(f"  {k}: {v}")
    
    # Per-frame analysis
    bt_per_frame = per_frame_analysis(bt_traj, data)
    st_per_frame = per_frame_analysis(st_traj, data)
    
    # Save
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/oracle_trajectories.json', 'w') as f:
        json.dump({str(k): v for k, v in oracle_traj.items()}, f)
    with open('outputs/bytetrack_trajectories.json', 'w') as f:
        json.dump({str(k): v for k, v in bt_traj.items()}, f)
    with open('outputs/sparsetrack_trajectories.json', 'w') as f:
        json.dump({str(k): v for k, v in st_traj.items()}, f)
    with open('outputs/metrics.json', 'w') as f:
        json.dump({
            'Oracle': {k:v for k,v in om.items() if k!='track_to_gt'},
            'ByteTrack': {k:v for k,v in bm.items() if k!='track_to_gt'},
            'SparseTrack': {k:v for k,v in sm.items() if k!='track_to_gt'}
        }, f, indent=2)
    with open('outputs/track_to_gt_mapping.json', 'w') as f:
        json.dump({
            'Oracle': om['track_to_gt'],
            'ByteTrack': bm['track_to_gt'],
            'SparseTrack': sm['track_to_gt']
        }, f)
    
    bl = [len(t['trajectory']) for t in bt_traj.values()]
    sl = [len(t['trajectory']) for t in st_traj.values()]
    ol = [len(t['trajectory']) for t in oracle_traj.values()]
    with open('outputs/trajectory_lengths.json', 'w') as f:
        json.dump({
            'Oracle': {'mean': float(np.mean(ol)), 'std': float(np.std(ol)),
                      'min': int(np.min(ol)), 'max': int(np.max(ol))},
            'ByteTrack': {'mean': float(np.mean(bl)), 'std': float(np.std(bl)),
                         'min': int(np.min(bl)), 'max': int(np.max(bl))},
            'SparseTrack': {'mean': float(np.mean(sl)), 'std': float(np.std(sl)),
                           'min': int(np.min(sl)), 'max': int(np.max(sl))}
        }, f, indent=2)
    
    with open('outputs/per_frame_analysis.json', 'w') as f:
        json.dump({'ByteTrack': bt_per_frame, 'SparseTrack': st_per_frame}, f)
    
    print("\nResults saved.")
    return om, bm, sm

if __name__ == '__main__':
    main()
