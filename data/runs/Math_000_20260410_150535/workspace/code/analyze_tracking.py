import json, math, os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

DATA_PATH = 'data/simulated_sequence.json'
OUT_DIR = 'outputs'
IMG_DIR = 'report/images'


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0.0, boxA[2]-boxA[0]) * max(0.0, boxA[3]-boxA[1])
    areaB = max(0.0, boxB[2]-boxB[0]) * max(0.0, boxB[3]-boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def center(box):
    return np.array([(box[0]+box[2])/2.0, (box[1]+box[3])/2.0], dtype=float)


def size(box):
    return np.array([max(1e-6, box[2]-box[0]), max(1e-6, box[3]-box[1])], dtype=float)


def pseudo_depth(box):
    w, h = size(box)
    area = w*h
    return 1.0 / math.sqrt(area)


def greedy_match(cost_matrix, threshold):
    matches = []
    if cost_matrix.size == 0:
        return matches, list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    used_r, used_c = set(), set()
    flat = []
    for r in range(cost_matrix.shape[0]):
        for c in range(cost_matrix.shape[1]):
            flat.append((cost_matrix[r, c], r, c))
    flat.sort(key=lambda x: x[0])
    for cost, r, c in flat:
        if cost > threshold:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matches.append((r, c, cost))
    unmatched_r = [r for r in range(cost_matrix.shape[0]) if r not in used_r]
    unmatched_c = [c for c in range(cost_matrix.shape[1]) if c not in used_c]
    return matches, unmatched_r, unmatched_c


class Track:
    def __init__(self, tid, det, frame_idx):
        self.id = tid
        self.box = np.array(det['bbox'], dtype=float)
        self.prev_box = np.array(det['bbox'], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.last_frame = frame_idx
        self.hits = 1
        self.age = 1
        self.misses = 0
        self.history = [(frame_idx, det['bbox'], det['score'])]
        self.scores = [det['score']]
        self.depth = pseudo_depth(det['bbox'])

    def predict_box(self):
        c = center(self.box) + self.velocity
        s = size(self.box)
        return np.array([c[0]-s[0]/2, c[1]-s[1]/2, c[0]+s[0]/2, c[1]+s[1]/2], dtype=float)

    def update(self, det, frame_idx):
        new_box = np.array(det['bbox'], dtype=float)
        self.velocity = center(new_box) - center(self.box)
        self.prev_box = self.box.copy()
        self.box = new_box
        self.last_frame = frame_idx
        self.hits += 1
        self.age += 1
        self.misses = 0
        self.history.append((frame_idx, det['bbox'], det['score']))
        self.scores.append(det['score'])
        self.depth = pseudo_depth(det['bbox'])

    def mark_missed(self):
        self.age += 1
        self.misses += 1


def associate(tracks, detections, frame_idx, mode='byte', high_thr=0.3, low_thr=0.1, max_cost=0.9):
    if not tracks:
        return [], [], list(range(len(detections))), {'num_layers': 0, 'layer_sizes': []}

    pred_boxes = [t.predict_box() for t in tracks]

    def build_cost(track_indices, det_indices):
        mat = np.ones((len(track_indices), len(det_indices)), dtype=float) * 1e6
        for i, ti in enumerate(track_indices):
            tb = pred_boxes[ti]
            tc = center(tb)
            ts = size(tb)
            for j, di in enumerate(det_indices):
                db = detections[di]['bbox']
                dc = center(db)
                ds = size(db)
                ov = iou(tb, db)
                dist = np.linalg.norm((dc - tc) / np.maximum((ts + ds) / 2.0, 1.0))
                score = detections[di]['score']
                cost = 0.65 * (1 - ov) + 0.35 * min(dist / 4.0, 1.5)
                cost -= 0.05 * min(score, 1.0)
                mat[i, j] = cost
        return mat

    if mode == 'byte':
        high = [i for i, d in enumerate(detections) if d['score'] >= high_thr]
        low = [i for i, d in enumerate(detections) if low_thr <= d['score'] < high_thr]
        cost1 = build_cost(list(range(len(tracks))), high)
        m1, un_tracks, _ = greedy_match(cost1, max_cost)
        matches = [(ti, high[dj]) for ti, dj, _ in m1]
        if un_tracks and low:
            cost2 = build_cost(un_tracks, low)
            m2, un_tracks2, un_low = greedy_match(cost2, max_cost + 0.05)
            matches += [(un_tracks[ti], low[dj]) for ti, dj, _ in m2]
            un_det = [low[i] for i in un_low]
            un_det += [hi for hi in high if hi not in [d for _, d in matches]]
            return matches, un_tracks2, un_det, {'num_layers': 2, 'layer_sizes': [len(high), len(low)]}
        un_det = [hi for hi in high if hi not in [d for _, d in matches]] + low
        return matches, un_tracks, un_det, {'num_layers': 2, 'layer_sizes': [len(high), len(low)]}

    # sparse hierarchical: split by pseudo-depth quantiles
    depths = np.array([pseudo_depth(d['bbox']) for d in detections])
    if len(depths) == 0:
        return [], list(range(len(tracks))), [], {'num_layers': 0, 'layer_sizes': []}
    qs = np.quantile(depths, [0.33, 0.66]) if len(depths) >= 3 else [np.median(depths), np.median(depths)]
    det_layers = [[], [], []]
    for i, dep in enumerate(depths):
        if dep <= qs[0]:
            det_layers[0].append(i)
        elif dep <= qs[1]:
            det_layers[1].append(i)
        else:
            det_layers[2].append(i)
    track_layers = [[], [], []]
    tdepths = np.array([t.depth for t in tracks])
    tqs = np.quantile(tdepths, [0.33, 0.66]) if len(tdepths) >= 3 else [np.median(tdepths), np.median(tdepths)]
    for i, dep in enumerate(tdepths):
        if dep <= tqs[0]:
            track_layers[0].append(i)
        elif dep <= tqs[1]:
            track_layers[1].append(i)
        else:
            track_layers[2].append(i)
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))
    for layer in range(3):
        tis = [i for i in track_layers[layer] if i in unmatched_tracks]
        dis = [i for i in det_layers[layer] if i in unmatched_dets]
        if not tis or not dis:
            continue
        cost = build_cost(tis, dis)
        m, u_t, u_d = greedy_match(cost, max_cost - 0.05)
        for ti, dj, _ in m:
            matches.append((tis[ti], dis[dj]))
        matched_t = {tis[ti] for ti, _, _ in m}
        matched_d = {dis[dj] for _, dj, _ in m}
        unmatched_tracks -= matched_t
        unmatched_dets -= matched_d
    # residual cross-layer rescue using higher confidence detections first
    rem_t = sorted(unmatched_tracks)
    rem_d = sorted(unmatched_dets, key=lambda i: detections[i]['score'], reverse=True)
    if rem_t and rem_d:
        cost = build_cost(rem_t, rem_d)
        m, u_t, u_d = greedy_match(cost, max_cost + 0.02)
        for ti, dj, _ in m:
            matches.append((rem_t[ti], rem_d[dj]))
        unmatched_tracks = {rem_t[i] for i in u_t}
        unmatched_dets = {rem_d[i] for i in u_d}
    return matches, sorted(unmatched_tracks), sorted(unmatched_dets), {'num_layers': 3, 'layer_sizes': [len(x) for x in det_layers]}


def run_tracker(frames, mode='byte'):
    tracks = []
    finished = []
    next_id = 1
    assignments = []
    layer_logs = []
    for fi, frame in enumerate(frames):
        detections = frame['detections']
        matches, unmatched_tracks, unmatched_dets, meta = associate(tracks, detections, fi, mode=mode)
        layer_logs.append({'frame': fi, **meta})
        matched_track_ids = set()
        for ti, di in matches:
            tracks[ti].update(detections[di], fi)
            matched_track_ids.add(ti)
            assignments.append({'frame': fi, 'track_id': tracks[ti].id, 'gt_id': detections[di]['gt_id'], 'score': detections[di]['score']})
        for ti, t in enumerate(tracks):
            if ti not in matched_track_ids:
                t.mark_missed()
        survivors = []
        for t in tracks:
            if t.misses <= 3:
                survivors.append(t)
            else:
                finished.append(t)
        tracks = survivors
        for di in unmatched_dets:
            d = detections[di]
            if d['score'] >= 0.12:
                nt = Track(next_id, d, fi)
                tracks.append(nt)
                assignments.append({'frame': fi, 'track_id': nt.id, 'gt_id': d['gt_id'], 'score': d['score']})
                next_id += 1
    finished.extend(tracks)
    return finished, pd.DataFrame(assignments), pd.DataFrame(layer_logs)


def evaluate(assignments_df, frames):
    gt_total = sum(len(fr['gt_ids']) for fr in frames)
    det_total = sum(len(fr['detections']) for fr in frames)
    matched = len(assignments_df)
    recall = matched / gt_total if gt_total else 0.0
    det_recall = matched / det_total if det_total else 0.0
    purity = 0.0
    switches = 0
    fragments = 0
    track_majority = {}
    if not assignments_df.empty:
        for tid, group in assignments_df.groupby('track_id'):
            c = group['gt_id'].value_counts()
            majority = c.index[0]
            track_majority[tid] = majority
            purity += c.iloc[0]
        purity /= len(assignments_df)
        gt_tracks = defaultdict(list)
        for _, row in assignments_df.sort_values(['gt_id', 'frame']).iterrows():
            gt_tracks[row['gt_id']].append((int(row['frame']), int(row['track_id'])))
        for gt, seq in gt_tracks.items():
            prev_tid = None
            prev_frame = None
            segs = 0
            for fr, tid in seq:
                if prev_frame is None or fr != prev_frame + 1 or tid != prev_tid:
                    segs += 1
                if prev_tid is not None and tid != prev_tid:
                    switches += 1
                prev_tid = tid
                prev_frame = fr
            fragments += max(segs - 1, 0)
    mota_like = 1.0 - (gt_total - matched + switches) / gt_total if gt_total else 0.0
    return {
        'gt_total': gt_total,
        'det_total': det_total,
        'matched_assignments': matched,
        'recall_vs_gt': recall,
        'assignment_ratio_vs_dets': det_recall,
        'id_purity': purity,
        'id_switches': switches,
        'fragments': fragments,
        'mota_like': mota_like,
    }


def summarize_dataset(frames):
    rows = []
    ious = []
    for fr in frames:
        dets = fr['detections']
        boxes = [d['bbox'] for d in dets]
        overlaps = 0
        total_pairs = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                ov = iou(boxes[i], boxes[j])
                ious.append(ov)
                total_pairs += 1
                if ov > 0.2:
                    overlaps += 1
        rows.append({'frame': fr['frame'], 'num_gt': len(fr['gt_ids']), 'num_det': len(dets), 'dense_overlap_pairs': overlaps, 'pair_count': total_pairs})
    return pd.DataFrame(rows), np.array(ious)


def make_figures(dataset_df, pair_ious, byte_metrics, sparse_metrics, layer_df, frames):
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(dataset_df['frame'], dataset_df['num_gt'], label='Ground truth objects', lw=2)
    plt.plot(dataset_df['frame'], dataset_df['num_det'], label='Detections', lw=2)
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('Dataset overview across frames')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/data_overview.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(pair_ious, bins=30)
    plt.axvline(0.2, color='red', linestyle='--', label='Occlusion overlap threshold')
    plt.xlabel('Pairwise IoU among detections in the same frame')
    plt.ylabel('Frequency')
    plt.title('Crowding / overlap distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/overlap_distribution.png', dpi=200)
    plt.close()

    metrics_df = pd.DataFrame([
        {'tracker': 'ByteTrack-like', 'metric': k, 'value': v} for k, v in byte_metrics.items() if isinstance(v, (int, float))
    ] + [
        {'tracker': 'SparseTrack-like', 'metric': k, 'value': v} for k, v in sparse_metrics.items() if isinstance(v, (int, float))
    ])
    subset = metrics_df[metrics_df['metric'].isin(['recall_vs_gt','id_purity','mota_like'])]
    plt.figure(figsize=(7,4))
    sns.barplot(data=subset, x='metric', y='value', hue='tracker')
    plt.ylim(0, max(1.0, subset['value'].max()*1.1))
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Primary tracking metrics')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/main_metrics.png', dpi=200)
    plt.close()

    subset2 = metrics_df[metrics_df['metric'].isin(['id_switches','fragments','matched_assignments'])]
    plt.figure(figsize=(8,4))
    sns.barplot(data=subset2, x='metric', y='value', hue='tracker')
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.title('Association stability and coverage')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/stability_metrics.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7,4))
    layer_sizes = pd.DataFrame(layer_df['layer_sizes'].tolist(), columns=['near','mid','far'])
    layer_sizes['frame'] = layer_df['frame']
    melted = layer_sizes.melt(id_vars='frame', var_name='layer', value_name='count')
    sns.lineplot(data=melted, x='frame', y='count', hue='layer')
    plt.title('SparseTrack-like pseudo-depth layer sizes')
    plt.xlabel('Frame')
    plt.ylabel('Detections per layer')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/hierarchical_layers.png', dpi=200)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(DATA_PATH) as f:
        frames = json.load(f)
    dataset_df, pair_ious = summarize_dataset(frames)
    byte_tracks, byte_assignments, byte_layers = run_tracker(frames, mode='byte')
    sparse_tracks, sparse_assignments, sparse_layers = run_tracker(frames, mode='sparse')
    byte_metrics = evaluate(byte_assignments, frames)
    sparse_metrics = evaluate(sparse_assignments, frames)
    make_figures(dataset_df, pair_ious, byte_metrics, sparse_metrics, sparse_layers, frames)
    dataset_df.to_csv(f'{OUT_DIR}/dataset_summary.csv', index=False)
    byte_assignments.to_csv(f'{OUT_DIR}/bytetrack_assignments.csv', index=False)
    sparse_assignments.to_csv(f'{OUT_DIR}/sparsetrack_assignments.csv', index=False)
    sparse_layers.to_csv(f'{OUT_DIR}/sparsetrack_layers.csv', index=False)
    with open(f'{OUT_DIR}/metrics.json', 'w') as f:
        json.dump({'byte': byte_metrics, 'sparse': sparse_metrics}, f, indent=2)
    with open(f'{OUT_DIR}/track_counts.json', 'w') as f:
        json.dump({'byte_num_tracks': len(byte_tracks), 'sparse_num_tracks': len(sparse_tracks)}, f, indent=2)
    print(json.dumps({'byte': byte_metrics, 'sparse': sparse_metrics, 'byte_tracks': len(byte_tracks), 'sparse_tracks': len(sparse_tracks)}, indent=2))

if __name__ == '__main__':
    main()
