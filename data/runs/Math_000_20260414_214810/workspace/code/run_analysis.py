import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'simulated_sequence.json'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid')


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=float)


def bbox_height(b):
    return max(1e-6, b[3] - b[1])


def pseudo_depth_score(b):
    # Larger score interpreted as nearer object.
    return bbox_height(b) + 0.15 * b[3]


def group_from_depth(depth, quantiles):
    if depth >= quantiles[1]:
        return 'near'
    if depth >= quantiles[0]:
        return 'mid'
    return 'far'


def hungarian_match(track_boxes, det_boxes, iou_threshold):
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    cost = np.ones((len(track_boxes), len(det_boxes)), dtype=float)
    for i, tb in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            cost[i, j] = 1.0 - iou_xyxy(tb, db)
    rows, cols = linear_sum_assignment(cost)
    matches = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(rows, cols):
        iou = 1.0 - cost[r, c]
        if iou >= iou_threshold:
            matches.append((r, c, iou))
            matched_tracks.add(r)
            matched_dets.add(c)
    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


@dataclass
class Track:
    tid: int
    bbox: List[float]
    last_frame: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    age: int = 0
    hits: int = 1
    history: List[Dict] = field(default_factory=list)

    def predict(self):
        c = bbox_center(self.bbox)
        new_c = c + self.velocity
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return [new_c[0] - w / 2, new_c[1] - h / 2, new_c[0] + w / 2, new_c[1] + h / 2]

    def update(self, bbox, frame_idx, score, gt_id):
        prev_c = bbox_center(self.bbox)
        new_c = bbox_center(bbox)
        self.velocity = 0.7 * self.velocity + 0.3 * (new_c - prev_c)
        self.bbox = list(map(float, bbox))
        self.last_frame = frame_idx
        self.age = 0
        self.hits += 1
        self.history.append({'frame': frame_idx, 'bbox': list(map(float, bbox)), 'score': float(score), 'gt_id': int(gt_id)})

    def mark_missed(self):
        self.age += 1


def run_tracker(frames, method='bytetrack', high_thres=0.5, low_thres=0.1, max_age=3, iou_main=0.3, iou_low=0.2):
    tracks: Dict[int, Track] = {}
    next_tid = 0
    assignments = []
    stage_rows = []

    for frame in frames:
        fidx = frame['frame']
        detections = [d for d in frame['detections'] if d['score'] >= low_thres]
        high = [d for d in detections if d['score'] >= high_thres]
        low = [d for d in detections if d['score'] < high_thres]

        active_ids = [tid for tid, tr in tracks.items() if tr.age <= max_age]
        pred_boxes = [tracks[tid].predict() for tid in active_ids]

        matched_track_ids = set()
        matched_det_ids_high = set()
        matched_det_ids_low = set()

        if method == 'bytetrack':
            matches, un_tracks_idx, un_high_idx = hungarian_match(pred_boxes, [d['bbox'] for d in high], iou_main)
            for ti, di, iou in matches:
                tid = active_ids[ti]
                det = high[di]
                tracks[tid].update(det['bbox'], fidx, det['score'], det['gt_id'])
                assignments.append({'frame': fidx, 'track_id': tid, 'gt_id': det['gt_id'], 'score': det['score'], 'stage': 'high', 'iou': iou})
                stage_rows.append({'frame': fidx, 'track_id': tid, 'stage': 'high', 'det_score': det['score'], 'gt_id': det['gt_id']})
                matched_track_ids.add(tid)
                matched_det_ids_high.add(di)
            rem_track_ids = [active_ids[i] for i in un_tracks_idx]
            rem_pred = [tracks[tid].predict() for tid in rem_track_ids]
            matches2, _, un_low_idx = hungarian_match(rem_pred, [d['bbox'] for d in low], iou_low)
            matched_rem_tracks = set()
            for ti, di, iou in matches2:
                tid = rem_track_ids[ti]
                det = low[di]
                tracks[tid].update(det['bbox'], fidx, det['score'], det['gt_id'])
                assignments.append({'frame': fidx, 'track_id': tid, 'gt_id': det['gt_id'], 'score': det['score'], 'stage': 'low', 'iou': iou})
                stage_rows.append({'frame': fidx, 'track_id': tid, 'stage': 'low', 'det_score': det['score'], 'gt_id': det['gt_id']})
                matched_track_ids.add(tid)
                matched_rem_tracks.add(tid)
                matched_det_ids_low.add(di)
        elif method == 'sparsetrack_like':
            all_depths = [pseudo_depth_score(d['bbox']) for d in detections] or [0, 1]
            q1, q2 = np.quantile(all_depths, [1/3, 2/3])
            # First stage with high detections per depth group.
            remaining_track_ids = set(active_ids)
            for group in ['near', 'mid', 'far']:
                group_high = [d for d in high if group_from_depth(pseudo_depth_score(d['bbox']), (q1, q2)) == group]
                group_tracks = [tid for tid in active_ids if tid in remaining_track_ids and group_from_depth(pseudo_depth_score(tracks[tid].predict()), (q1, q2)) == group]
                matches, _, _ = hungarian_match([tracks[tid].predict() for tid in group_tracks], [d['bbox'] for d in group_high], iou_main)
                used_group_tracks = set()
                for ti, di, iou in matches:
                    tid = group_tracks[ti]
                    det = group_high[di]
                    tracks[tid].update(det['bbox'], fidx, det['score'], det['gt_id'])
                    assignments.append({'frame': fidx, 'track_id': tid, 'gt_id': det['gt_id'], 'score': det['score'], 'stage': f'{group}_high', 'iou': iou})
                    stage_rows.append({'frame': fidx, 'track_id': tid, 'stage': f'{group}_high', 'det_score': det['score'], 'gt_id': det['gt_id']})
                    matched_track_ids.add(tid)
                    used_group_tracks.add(tid)
                remaining_track_ids -= used_group_tracks
            # Second hierarchical recovery over all leftover tracks against low-score detections by group.
            rem_track_ids = [tid for tid in active_ids if tid not in matched_track_ids]
            used_low_global = set()
            for group in ['near', 'mid', 'far']:
                group_tracks = [tid for tid in rem_track_ids if group_from_depth(pseudo_depth_score(tracks[tid].predict()), (q1, q2)) == group]
                group_low_all = [(idx, d) for idx, d in enumerate(low) if idx not in used_low_global and group_from_depth(pseudo_depth_score(d['bbox']), (q1, q2)) == group]
                group_low = [d for _, d in group_low_all]
                matches, _, _ = hungarian_match([tracks[tid].predict() for tid in group_tracks], [d['bbox'] for d in group_low], iou_low)
                used_tracks = set()
                for ti, di, iou in matches:
                    tid = group_tracks[ti]
                    orig_idx, det = group_low_all[di]
                    tracks[tid].update(det['bbox'], fidx, det['score'], det['gt_id'])
                    assignments.append({'frame': fidx, 'track_id': tid, 'gt_id': det['gt_id'], 'score': det['score'], 'stage': f'{group}_low', 'iou': iou})
                    stage_rows.append({'frame': fidx, 'track_id': tid, 'stage': f'{group}_low', 'det_score': det['score'], 'gt_id': det['gt_id']})
                    matched_track_ids.add(tid)
                    used_tracks.add(tid)
                    used_low_global.add(orig_idx)
                rem_track_ids = [tid for tid in rem_track_ids if tid not in used_tracks]
        else:
            raise ValueError(method)

        # Mark missed active tracks
        for tid in list(tracks.keys()):
            if tracks[tid].last_frame != fidx:
                tracks[tid].mark_missed()

        # Start new tracks from unmatched detections, prioritizing high then remaining low
        used_gt_stage = {(a['frame'], a['gt_id']) for a in assignments if a['frame'] == fidx}
        for det in high + low:
            if (fidx, det['gt_id']) in used_gt_stage:
                continue
            tid = next_tid
            next_tid += 1
            tr = Track(tid=tid, bbox=list(map(float, det['bbox'])), last_frame=fidx)
            tr.history.append({'frame': fidx, 'bbox': list(map(float, det['bbox'])), 'score': float(det['score']), 'gt_id': int(det['gt_id'])})
            tracks[tid] = tr
            assignments.append({'frame': fidx, 'track_id': tid, 'gt_id': det['gt_id'], 'score': det['score'], 'stage': 'init', 'iou': None})
            stage_rows.append({'frame': fidx, 'track_id': tid, 'stage': 'init', 'det_score': det['score'], 'gt_id': det['gt_id']})

        # prune dead tracks
        dead = [tid for tid, tr in tracks.items() if tr.age > max_age]
        for tid in dead:
            del tracks[tid]

    assign_df = pd.DataFrame(assignments)
    stage_df = pd.DataFrame(stage_rows)
    return assign_df, stage_df


def evaluate(assign_df, frames):
    n_frames = len(frames)
    gt_per_frame = {fr['frame']: len(fr['gt_ids']) for fr in frames}
    det_gt_per_frame = {fr['frame']: {d['gt_id'] for d in fr['detections']} for fr in frames}
    assigned = assign_df.groupby(['frame', 'gt_id'])['track_id'].first().reset_index()
    tp = len(assigned)
    total_gt = sum(gt_per_frame.values())
    fn = total_gt - tp
    fp = 0
    idsw = 0
    frag = 0
    prev_track_for_gt = {}
    presence = {}
    for gt_id, g in assigned.sort_values(['gt_id', 'frame']).groupby('gt_id'):
        prev_frame = None
        seq = []
        for _, row in g.iterrows():
            frame = int(row['frame'])
            track = int(row['track_id'])
            seq.append((frame, track))
            if gt_id in prev_track_for_gt and prev_track_for_gt[gt_id] != track:
                idsw += 1
            prev_track_for_gt[gt_id] = track
            if prev_frame is not None and frame - prev_frame > 1:
                frag += 1
            prev_frame = frame
        presence[gt_id] = seq
    mota = 1.0 - (fn + fp + idsw) / total_gt
    idtp = 0
    idfp = 0
    idfn = 0
    # approximate IDF1 via dominant track assignment per gt and per track purity
    for gt_id, seq in presence.items():
        track_counts = {}
        for _, tr in seq:
            track_counts[tr] = track_counts.get(tr, 0) + 1
        best = max(track_counts.values()) if track_counts else 0
        idtp += best
        idfn += len(seq) - best
    for track_id, g in assigned.groupby('track_id'):
        gt_counts = g['gt_id'].value_counts().to_dict()
        best = max(gt_counts.values()) if gt_counts else 0
        idfp += len(g) - best
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    covered_frames = set(assigned['frame'].tolist())
    det_recall = tp / sum(len(v) for v in det_gt_per_frame.values())
    return {
        'TP_assignments': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'ID_switches': int(idsw),
        'Fragmentations': int(frag),
        'MOTA_proxy': float(mota),
        'IDF1_proxy': float(idf1),
        'Detection_coverage_ratio': float(det_recall),
        'Frames': int(n_frames),
        'Total_GT_boxes': int(total_gt),
    }


def occlusion_table(frames, assign_df):
    rows = []
    assigned_set = {(int(r.frame), int(r.gt_id)) for r in assign_df[['frame', 'gt_id']].itertuples(index=False)}
    for fr in frames:
        boxes = fr['gt_bboxes']
        ids = fr['gt_ids']
        overlaps = []
        for i, box_i in enumerate(boxes):
            max_iou = 0.0
            for j, box_j in enumerate(boxes):
                if i == j:
                    continue
                max_iou = max(max_iou, iou_xyxy(box_i, box_j))
            overlaps.append(max_iou)
        q1, q2 = np.quantile(overlaps, [1/3, 2/3])
        for gt_id, ov in zip(ids, overlaps):
            grp = 'low' if ov < q1 else ('mid' if ov < q2 else 'high')
            rows.append({'frame': fr['frame'], 'gt_id': gt_id, 'max_overlap': ov, 'occlusion_group': grp, 'tracked': (fr['frame'], gt_id) in assigned_set})
    df = pd.DataFrame(rows)
    return df.groupby('occlusion_group')['tracked'].agg(['mean', 'count']).reset_index()


def trajectory_samples(assign_df, frames):
    gt_to_boxes = {}
    for fr in frames:
        for gid, box in zip(fr['gt_ids'], fr['gt_bboxes']):
            gt_to_boxes.setdefault(gid, []).append((fr['frame'], box))
    # choose top-overlap IDs by average overlap burden
    burdens = []
    for gid, seq in gt_to_boxes.items():
        ovs = []
        for frame, box in seq:
            fr = frames[frame]
            boxes = fr['gt_bboxes']
            ids = fr['gt_ids']
            idx = ids.index(gid)
            max_iou = max([iou_xyxy(box, other) for j, other in enumerate(boxes) if j != idx] + [0])
            ovs.append(max_iou)
        burdens.append((gid, float(np.mean(ovs))))
    chosen = [gid for gid, _ in sorted(burdens, key=lambda x: x[1], reverse=True)[:8]]
    rows = []
    predicted = assign_df.groupby(['gt_id', 'frame'])['track_id'].first().reset_index()
    for gid in chosen:
        for _, row in predicted[predicted['gt_id'] == gid].iterrows():
            frame = int(row['frame'])
            gt_box = gt_to_boxes[gid][frame][1]
            c = bbox_center(gt_box)
            rows.append({'gt_id': gid, 'frame': frame, 'x': c[0], 'y': c[1], 'track_id': int(row['track_id'])})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True, parents=True)
    with open(DATA_PATH) as f:
        frames = json.load(f)

    overview = {
        'num_frames': len(frames),
        'objects_per_frame_mean': float(np.mean([len(fr['gt_ids']) for fr in frames])),
        'objects_per_frame_min': int(np.min([len(fr['gt_ids']) for fr in frames])),
        'objects_per_frame_max': int(np.max([len(fr['gt_ids']) for fr in frames])),
        'detections_per_frame_mean': float(np.mean([len(fr['detections']) for fr in frames])),
        'unique_gt_ids': int(len(set(g for fr in frames for g in fr['gt_ids']))),
        'detection_rate_empirical': float(sum(len(fr['detections']) for fr in frames) / sum(len(fr['gt_ids']) for fr in frames)),
        'score_mean': float(np.mean([d['score'] for fr in frames for d in fr['detections']])),
        'score_std': float(np.std([d['score'] for fr in frames for d in fr['detections']])),
    }

    byte_assign, byte_stage = run_tracker(frames, method='bytetrack')
    sparse_assign, sparse_stage = run_tracker(frames, method='sparsetrack_like')

    byte_metrics = evaluate(byte_assign, frames)
    sparse_metrics = evaluate(sparse_assign, frames)

    metrics_df = pd.DataFrame([
        {'method': 'ByteTrack', **byte_metrics},
        {'method': 'SparseTrack_like', **sparse_metrics},
    ])
    metrics_df.to_csv(OUT_DIR / 'comparison_metrics.csv', index=False)

    with open(OUT_DIR / 'data_overview.json', 'w') as f:
        json.dump(overview, f, indent=2)
    with open(OUT_DIR / 'byte_metrics.json', 'w') as f:
        json.dump(byte_metrics, f, indent=2)
    with open(OUT_DIR / 'sparse_metrics.json', 'w') as f:
        json.dump(sparse_metrics, f, indent=2)

    byte_assign.to_csv(OUT_DIR / 'bytetrack_assignments.csv', index=False)
    sparse_assign.to_csv(OUT_DIR / 'sparsetrack_assignments.csv', index=False)
    byte_stage.to_csv(OUT_DIR / 'bytetrack_stage_breakdown.csv', index=False)
    sparse_stage.to_csv(OUT_DIR / 'sparsetrack_stage_breakdown.csv', index=False)

    occ_byte = occlusion_table(frames, byte_assign)
    occ_byte['method'] = 'ByteTrack'
    occ_sparse = occlusion_table(frames, sparse_assign)
    occ_sparse['method'] = 'SparseTrack_like'
    occ_df = pd.concat([occ_byte, occ_sparse], ignore_index=True)
    occ_df.to_csv(OUT_DIR / 'occlusion_conditioned_tracking.csv', index=False)

    # Figure 1: score distribution
    plt.figure(figsize=(7,4))
    scores = [d['score'] for fr in frames for d in fr['detections']]
    sns.histplot(scores, bins=30, kde=True)
    plt.axvline(0.5, color='r', linestyle='--', label='high threshold')
    plt.axvline(0.1, color='orange', linestyle='--', label='low threshold')
    plt.title('Detection score distribution')
    plt.xlabel('score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'score_distribution.png', dpi=200)
    plt.close()

    # Figure 2: metric comparison
    plot_df = metrics_df.melt(id_vars='method', value_vars=['MOTA_proxy', 'IDF1_proxy', 'ID_switches', 'Fragmentations'], var_name='metric', value_name='value')
    plt.figure(figsize=(8,4.5))
    sns.barplot(data=plot_df, x='metric', y='value', hue='method')
    plt.title('Overall tracking comparison')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'metric_comparison.png', dpi=200)
    plt.close()

    # Figure 3: occlusion-conditioned recall
    plt.figure(figsize=(7,4.5))
    sns.barplot(data=occ_df, x='occlusion_group', y='mean', hue='method', order=['low','mid','high'])
    plt.ylabel('tracked fraction')
    plt.title('Tracking retention by occlusion burden')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'occlusion_conditioned_tracking.png', dpi=200)
    plt.close()

    # Figure 4: stage usage breakdown
    stage_summary = pd.concat([
        byte_stage.assign(method='ByteTrack'),
        sparse_stage.assign(method='SparseTrack_like')
    ]).groupby(['method', 'stage']).size().reset_index(name='count')
    stage_summary.to_csv(OUT_DIR / 'stage_usage_summary.csv', index=False)
    plt.figure(figsize=(10,4.5))
    sns.barplot(data=stage_summary, x='stage', y='count', hue='method')
    plt.title('Association stage usage')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'association_stage_usage.png', dpi=200)
    plt.close()

    # Figure 5: trajectory sample comparison using sparse result assignments
    traj_df = trajectory_samples(sparse_assign, frames)
    traj_df.to_csv(OUT_DIR / 'trajectory_samples_sparse.csv', index=False)
    plt.figure(figsize=(6,6))
    for gid, g in traj_df.groupby('gt_id'):
        plt.plot(g['x'], g['y'], marker='o', linewidth=1.5, markersize=2, label=f'GT {gid}')
    plt.gca().invert_yaxis()
    plt.title('Sample crowded-scene trajectories (SparseTrack-like)')
    plt.xlabel('center x')
    plt.ylabel('center y')
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'trajectory_samples.png', dpi=200)
    plt.close()

    claim_recovery = pd.DataFrame([
        {'claim': 'Dataset is dense and detection-imperfect', 'artifact': 'outputs/data_overview.json; report/images/score_distribution.png'},
        {'claim': 'SparseTrack-like hierarchical association improves identity continuity', 'artifact': 'outputs/comparison_metrics.csv; report/images/metric_comparison.png'},
        {'claim': 'Sparse decomposition helps under high occlusion burden', 'artifact': 'outputs/occlusion_conditioned_tracking.csv; report/images/occlusion_conditioned_tracking.png'},
        {'claim': 'Hierarchical tracker uses depth-group stages rather than global two-stage matching', 'artifact': 'outputs/stage_usage_summary.csv; report/images/association_stage_usage.png'},
    ])
    claim_recovery.to_csv(OUT_DIR / 'claim_recovery_table.csv', index=False)


if __name__ == '__main__':
    main()
