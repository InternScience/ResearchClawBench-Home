import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "simulated_sequence.json"
OUTPUTS_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_frames():
    frames = json.loads(DATA_PATH.read_text())
    frames = sorted(frames, key=lambda x: x["frame"])
    return frames


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def estimate_depth(box) -> float:
    height = max(1.0, box[3] - box[1])
    return 1.0 / height


def greedy_match(track_indices, detection_indices, tracks, detections, iou_thresh, use_depth=False, depth_gate=0.35):
    candidates = []
    for ti in track_indices:
        for di in detection_indices:
            iou = bbox_iou(tracks[ti].pred_box, detections[di]["bbox"])
            if iou < iou_thresh:
                continue
            score = iou
            if use_depth:
                depth_t = tracks[ti].depth
                depth_d = detections[di]["depth"]
                rel_gap = abs(depth_t - depth_d) / max(depth_t, depth_d, 1e-6)
                if rel_gap > depth_gate:
                    continue
                score += 0.1 * (1.0 - rel_gap)
            candidates.append((score, ti, di))
    candidates.sort(reverse=True)
    assigned_t = set()
    assigned_d = set()
    matches = []
    for _, ti, di in candidates:
        if ti in assigned_t or di in assigned_d:
            continue
        assigned_t.add(ti)
        assigned_d.add(di)
        matches.append((ti, di))
    unmatched_t = [ti for ti in track_indices if ti not in assigned_t]
    unmatched_d = [di for di in detection_indices if di not in assigned_d]
    return matches, unmatched_t, unmatched_d


@dataclass
class Track:
    track_id: int
    pred_box: list
    last_box: list
    last_frame: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    depth: float = 0.0

    def update(self, box, frame_idx):
        self.pred_box = box
        self.last_box = box
        self.last_frame = frame_idx
        self.hits += 1
        self.misses = 0
        self.confirmed = self.confirmed or self.hits >= 2
        self.depth = estimate_depth(box)

    def miss(self):
        self.misses += 1


class OnlineTracker:
    def __init__(self, mode, high_thresh=0.3, low_thresh=0.1, match_thresh=0.3, max_age=4, new_track_thresh=0.22):
        self.mode = mode
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.new_track_thresh = new_track_thresh
        self.tracks = []
        self.next_track_id = 1
        self.results = []
        self.active_counts = []

    def new_track(self, det, frame_idx):
        tr = Track(
            track_id=self.next_track_id,
            pred_box=det["bbox"],
            last_box=det["bbox"],
            last_frame=frame_idx,
            depth=det["depth"],
        )
        self.next_track_id += 1
        self.tracks.append(tr)
        return tr

    def step(self, frame):
        frame_idx = frame["frame"]
        detections = []
        for det in frame["detections"]:
            copied = dict(det)
            copied["depth"] = estimate_depth(copied["bbox"])
            detections.append(copied)

        high_idx = [i for i, d in enumerate(detections) if d["score"] >= self.high_thresh]
        low_idx = [i for i, d in enumerate(detections) if self.low_thresh <= d["score"] < self.high_thresh]
        track_idx = list(range(len(self.tracks)))

        if self.mode == "sparse":
            depth_values = [d["depth"] for d in detections]
            median_depth = float(np.median(depth_values)) if depth_values else 0.0
            near_tracks = [ti for ti in track_idx if self.tracks[ti].depth <= median_depth]
            far_tracks = [ti for ti in track_idx if self.tracks[ti].depth > median_depth]
            near_high = [di for di in high_idx if detections[di]["depth"] <= median_depth]
            far_high = [di for di in high_idx if detections[di]["depth"] > median_depth]

            matches = []
            unmatched_tracks = []
            unmatched_dets = []
            for t_group, d_group in ((near_tracks, near_high), (far_tracks, far_high)):
                m, ut, ud = greedy_match(
                    t_group,
                    d_group,
                    self.tracks,
                    detections,
                    self.match_thresh,
                    use_depth=True,
                    depth_gate=0.25,
                )
                matches.extend(m)
                unmatched_tracks.extend(ut)
                unmatched_dets.extend(ud)

            unclaimed_tracks = [ti for ti in track_idx if ti not in {m[0] for m in matches}]
            unclaimed_high = [di for di in high_idx if di not in {m[1] for m in matches}]
            m2, unmatched_tracks, unmatched_dets = greedy_match(
                unclaimed_tracks,
                unclaimed_high,
                self.tracks,
                detections,
                0.2,
                use_depth=True,
                depth_gate=0.35,
            )
            matches.extend(m2)
        else:
            matches, unmatched_tracks, unmatched_dets = greedy_match(
                track_idx,
                high_idx,
                self.tracks,
                detections,
                self.match_thresh,
                use_depth=False,
            )

        second_matches, unmatched_tracks, low_unmatched = greedy_match(
            unmatched_tracks,
            low_idx,
            self.tracks,
            detections,
            0.2,
            use_depth=(self.mode == "sparse"),
            depth_gate=0.4,
        )
        matches.extend(second_matches)

        matched_tracks = set()
        matched_dets = set()
        for ti, di in matches:
            self.tracks[ti].update(detections[di]["bbox"], frame_idx)
            matched_tracks.add(ti)
            matched_dets.add(di)
            self.results.append(
                {
                    "frame": frame_idx,
                    "track_id": self.tracks[ti].track_id,
                    "bbox": detections[di]["bbox"],
                    "score": detections[di]["score"],
                    "gt_id": detections[di]["gt_id"],
                    "source": "matched",
                }
            )

        new_track_dets = [
            di
            for di in high_idx
            if di not in matched_dets and detections[di]["score"] >= self.new_track_thresh
        ]
        for di in new_track_dets:
            tr = self.new_track(detections[di], frame_idx)
            self.results.append(
                {
                    "frame": frame_idx,
                    "track_id": tr.track_id,
                    "bbox": detections[di]["bbox"],
                    "score": detections[di]["score"],
                    "gt_id": detections[di]["gt_id"],
                    "source": "new",
                }
            )

        surviving_tracks = []
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_tracks and tr.last_frame != frame_idx:
                tr.miss()
            if tr.misses <= self.max_age:
                surviving_tracks.append(tr)
        self.tracks = surviving_tracks
        self.active_counts.append(len(self.tracks))

    def run(self, frames):
        for frame in frames:
            self.step(frame)
        return pd.DataFrame(self.results)


def summarize_dataset(frames):
    num_frames = len(frames)
    gt_per_frame = [len(f["gt_ids"]) for f in frames]
    det_per_frame = [len(f["detections"]) for f in frames]
    det_scores = [d["score"] for f in frames for d in f["detections"]]
    matched_gt_ids = [d["gt_id"] for f in frames for d in f["detections"]]

    occluded_gt = 0
    gt_boxes = 0
    pair_density = []
    for f in frames:
        boxes = f["gt_bboxes"]
        gt_boxes += len(boxes)
        overlap_count = 0
        total_pairs = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total_pairs += 1
                if bbox_iou(boxes[i], boxes[j]) > 0.2:
                    overlap_count += 1
                    occluded_gt += 2
        pair_density.append(overlap_count / total_pairs if total_pairs else 0.0)

    stats = {
        "num_frames": num_frames,
        "mean_gt_per_frame": float(np.mean(gt_per_frame)),
        "mean_det_per_frame": float(np.mean(det_per_frame)),
        "detection_recall_proxy": len(matched_gt_ids) / max(1, gt_boxes),
        "score_mean": float(np.mean(det_scores)),
        "score_std": float(np.std(det_scores)),
        "score_min": float(np.min(det_scores)),
        "score_max": float(np.max(det_scores)),
        "occlusion_pair_density_mean": float(np.mean(pair_density)),
        "occlusion_gt_touch_count": int(occluded_gt),
        "unique_gt_ids_in_detections": int(len(set(matched_gt_ids))),
    }
    return stats


def evaluate_tracking(frames, pred_df):
    gt_by_frame = {}
    for frame in frames:
        gt_by_frame[frame["frame"]] = list(zip(frame["gt_ids"], frame["gt_bboxes"]))

    pred_df = pred_df.copy()
    pred_df["correct"] = pred_df["gt_id"] >= 0
    pred_df["gt_id"] = pred_df["gt_id"].astype(int)

    total_gt = sum(len(frame["gt_ids"]) for frame in frames)
    matched_gt = int(pred_df["correct"].sum())
    false_pos = int((pred_df["gt_id"] < 0).sum())
    unique_gt = set()
    id_switches = 0
    fragments = 0
    gt_to_history = defaultdict(list)
    gt_seen_frames = defaultdict(list)

    for _, row in pred_df.sort_values(["frame", "track_id"]).iterrows():
        gt = int(row["gt_id"])
        if gt < 0:
            continue
        unique_gt.add(gt)
        gt_to_history[gt].append((int(row["frame"]), int(row["track_id"])))
        gt_seen_frames[gt].append(int(row["frame"]))

    for gt, hist in gt_to_history.items():
        hist = sorted(hist)
        prev_track = None
        prev_frame = None
        for frame_idx, track_id in hist:
            if prev_track is not None and track_id != prev_track:
                id_switches += 1
            if prev_frame is not None and frame_idx - prev_frame > 1:
                fragments += 1
            prev_track = track_id
            prev_frame = frame_idx

    gt_track_counts = {gt: len(set(track for _, track in hist)) for gt, hist in gt_to_history.items()}
    mostly_tracked = 0
    mostly_lost = 0
    id_purity_values = []
    covered_gt_frames = 0
    for frame in frames:
        gt_ids = frame["gt_ids"]
        covered_gt_frames += len(gt_ids)

    gt_total_frames = Counter()
    for frame in frames:
        for gt in frame["gt_ids"]:
            gt_total_frames[int(gt)] += 1

    for gt, total in gt_total_frames.items():
        tracked = len(gt_seen_frames.get(gt, []))
        ratio = tracked / total if total else 0.0
        if ratio >= 0.8:
            mostly_tracked += 1
        if ratio <= 0.2:
            mostly_lost += 1
        if gt in gt_to_history:
            counts = Counter(track for _, track in gt_to_history[gt])
            id_purity_values.append(max(counts.values()) / sum(counts.values()))

    metrics = {
        "total_gt_boxes": total_gt,
        "predicted_boxes": int(len(pred_df)),
        "matched_gt_boxes": matched_gt,
        "recall": matched_gt / total_gt if total_gt else 0.0,
        "precision_proxy": matched_gt / len(pred_df) if len(pred_df) else 0.0,
        "false_positives": false_pos,
        "unique_gt_tracked": len(unique_gt),
        "id_switches": id_switches,
        "fragments": fragments,
        "mostly_tracked_ids": mostly_tracked,
        "mostly_lost_ids": mostly_lost,
        "mean_id_purity": float(np.mean(id_purity_values)) if id_purity_values else 0.0,
        "tracks_created": int(pred_df["track_id"].nunique()) if len(pred_df) else 0,
    }

    occluded_frames = []
    for frame in frames:
        occluded_ids = set()
        boxes = list(zip(frame["gt_ids"], frame["gt_bboxes"]))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if bbox_iou(boxes[i][1], boxes[j][1]) > 0.2:
                    occluded_ids.add(int(boxes[i][0]))
                    occluded_ids.add(int(boxes[j][0]))
        if occluded_ids:
            occluded_frames.append((frame["frame"], occluded_ids))

    occ_total = 0
    occ_matched = 0
    occ_switches = 0
    last_track_for_gt = {}
    for frame_idx, occ_ids in occluded_frames:
        occ_total += len(occ_ids)
        frame_preds = pred_df[pred_df["frame"] == frame_idx]
        for gt in occ_ids:
            rows = frame_preds[frame_preds["gt_id"] == gt]
            if not rows.empty:
                occ_matched += 1
                track_id = int(rows.iloc[0]["track_id"])
                if gt in last_track_for_gt and last_track_for_gt[gt] != track_id:
                    occ_switches += 1
                last_track_for_gt[gt] = track_id

    metrics["occlusion_recall"] = occ_matched / occ_total if occ_total else 0.0
    metrics["occlusion_id_switches"] = occ_switches
    return metrics


def build_frame_level_results(frames, tracker_results):
    rows = []
    grouped = tracker_results.groupby("frame")
    for frame in frames:
        idx = frame["frame"]
        pred = grouped.get_group(idx) if idx in grouped.groups else pd.DataFrame(columns=tracker_results.columns)
        rows.append(
            {
                "frame": idx,
                "gt_boxes": len(frame["gt_ids"]),
                "detections": len(frame["detections"]),
                "pred_tracks": int(len(pred)),
                "matched_gt": int((pred["gt_id"] >= 0).sum()) if len(pred) else 0,
                "mean_score": float(pred["score"].mean()) if len(pred) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_figures(dataset_stats, baseline_frame_df, sparse_frame_df, comparison_df):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(baseline_frame_df["frame"], baseline_frame_df["detections"], label="Detections", color="#4c78a8")
    axes[0].plot(baseline_frame_df["frame"], baseline_frame_df["gt_boxes"], label="GT boxes", color="#f58518")
    axes[0].set_title("Frame-wise Density")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].plot(baseline_frame_df["frame"], baseline_frame_df["matched_gt"], label="ByteTrack-like", color="#54a24b")
    axes[1].plot(sparse_frame_df["frame"], sparse_frame_df["matched_gt"], label="Sparse hierarchical", color="#e45756")
    axes[1].set_title("Recovered GT Boxes per Frame")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Matched GT")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "tracking_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = comparison_df[["method", "recall", "occlusion_recall", "mean_id_purity"]].set_index("method")
    bars.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Tracking Quality Comparison")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "quality_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    penalty = comparison_df[["method", "id_switches", "fragments", "tracks_created"]].set_index("method")
    penalty.plot(kind="bar", ax=ax, color=["#e45756", "#72b7b2", "#b279a2"])
    ax.set_title("Identity Stability and Track Fragmentation")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "identity_stability.png", dpi=200)
    plt.close(fig)


def write_outputs(dataset_stats, baseline_metrics, sparse_metrics, baseline_df, sparse_df, baseline_frame_df, sparse_frame_df):
    (OUTPUTS_DIR / "dataset_summary.json").write_text(json.dumps(dataset_stats, indent=2))
    comparison = pd.DataFrame(
        [
            {"method": "ByteTrack-like", **baseline_metrics},
            {"method": "Sparse hierarchical", **sparse_metrics},
        ]
    )
    comparison.to_csv(OUTPUTS_DIR / "method_comparison.csv", index=False)
    baseline_df.to_csv(OUTPUTS_DIR / "bytetrack_like_tracks.csv", index=False)
    sparse_df.to_csv(OUTPUTS_DIR / "sparse_hierarchical_tracks.csv", index=False)
    baseline_frame_df.to_csv(OUTPUTS_DIR / "bytetrack_like_frame_metrics.csv", index=False)
    sparse_frame_df.to_csv(OUTPUTS_DIR / "sparse_hierarchical_frame_metrics.csv", index=False)
    return comparison


def main():
    ensure_dirs()
    frames = load_frames()
    dataset_stats = summarize_dataset(frames)

    baseline_tracker = OnlineTracker(mode="byte", high_thresh=0.28, low_thresh=0.1, match_thresh=0.25, max_age=5, new_track_thresh=0.2)
    sparse_tracker = OnlineTracker(mode="sparse", high_thresh=0.28, low_thresh=0.1, match_thresh=0.25, max_age=5, new_track_thresh=0.2)
    baseline_df = baseline_tracker.run(frames)
    sparse_df = sparse_tracker.run(frames)

    baseline_metrics = evaluate_tracking(frames, baseline_df)
    sparse_metrics = evaluate_tracking(frames, sparse_df)
    baseline_frame_df = build_frame_level_results(frames, baseline_df)
    sparse_frame_df = build_frame_level_results(frames, sparse_df)
    comparison_df = write_outputs(
        dataset_stats,
        baseline_metrics,
        sparse_metrics,
        baseline_df,
        sparse_df,
        baseline_frame_df,
        sparse_frame_df,
    )
    save_figures(dataset_stats, baseline_frame_df, sparse_frame_df, comparison_df)

    summary = {
        "dataset": dataset_stats,
        "methods": comparison_df.to_dict(orient="records"),
    }
    (OUTPUTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
