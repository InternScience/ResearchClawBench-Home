#!/usr/bin/env python3
"""Run a compact MOT study on the simulated dense-occlusion sequence.

This script implements:
1. Dataset profiling and inferred occlusion labels from GT overlap.
2. Three online trackers with a shared motion model:
   - SORT-style IoU association with high-confidence detections only.
   - ByteTrack-style two-stage association with high and low score boxes.
   - SparseTrack-lite pseudo-depth hierarchical association.
3. Evaluation with MOTA, IDF1, precision/recall, occlusion recall, ID switches,
   and fragmentation.
4. Figure generation and report-ready outputs.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment


DATA_PATH = ROOT / "data" / "simulated_sequence.json"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

FRAME_W = 640.0
FRAME_H = 640.0
IOU_MATCH = 0.5
OCCLUSION_IOU = 0.2
SEED = 0


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def clamp_bbox(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), FRAME_W - 1)
    y1 = min(max(y1, 0.0), FRAME_H - 1)
    x2 = min(max(x2, x1 + 1e-3), FRAME_W)
    y2 = min(max(y2, y1 + 1e-3), FRAME_H)
    return np.array([x1, y1, x2, y2], dtype=float)


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def iou_single(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]) -> np.ndarray:
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    mat = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            mat[i, j] = iou_single(a, b)
    return mat


def pseudo_depth_score(bbox: Sequence[float]) -> float:
    bottom = bbox[3] / FRAME_H
    area_norm = min(bbox_area(bbox) / (FRAME_W * FRAME_H), 1.0)
    area_term = math.sqrt(max(area_norm, 1e-8))
    return 0.7 * bottom + 0.3 * area_term


def depth_band(score: float, bands: int) -> int:
    return min(bands - 1, max(0, int(score * bands)))


def format_float(x: float) -> float:
    return round(float(x), 6)


@dataclass
class Detection:
    frame: int
    bbox: np.ndarray
    score: float
    gt_id: int | None = None
    inferred_occluded: bool = False
    depth: float = 0.0


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    last_bbox: np.ndarray
    velocity: np.ndarray
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    history: Dict[int, List[float]] = field(default_factory=dict)

    def predict(self) -> np.ndarray:
        return clamp_bbox(self.bbox + self.velocity)

    def update(self, frame: int, new_bbox: np.ndarray) -> None:
        self.last_bbox = self.bbox.copy()
        self.velocity = new_bbox - self.bbox
        self.bbox = clamp_bbox(new_bbox)
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 2:
            self.confirmed = True
        self.history[frame] = self.bbox.tolist()

    def miss(self) -> None:
        self.bbox = self.predict()
        self.age += 1
        self.time_since_update += 1


class OnlineTracker:
    def __init__(
        self,
        name: str,
        high_thresh: float = 0.35,
        low_thresh: float = 0.1,
        match_thresh: float = 0.1,
        max_age: int = 8,
        min_hits: int = 2,
        depth_bands: int = 3,
    ) -> None:
        self.name = name
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.depth_bands = depth_bands
        self.reset()

    def reset(self) -> None:
        self.tracks: List[Track] = []
        self.next_id = 1
        self.outputs: Dict[int, Dict[int, List[float]]] = {}

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.time_since_update <= self.max_age]

    def visible_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _new_track(self, frame: int, det: Detection) -> None:
        track = Track(
            track_id=self.next_id,
            bbox=det.bbox.copy(),
            last_bbox=det.bbox.copy(),
            velocity=np.zeros(4, dtype=float),
            confirmed=self.min_hits <= 1,
            history={frame: det.bbox.tolist()},
        )
        self.next_id += 1
        self.tracks.append(track)

    def _match(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        *,
        depth_penalty: float = 0.0,
        max_band_delta: int | None = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        pred_boxes = [t.predict() for t in tracks]
        det_boxes = [d.bbox for d in detections]
        ious = iou_matrix(pred_boxes, det_boxes)
        cost = 1.0 - ious
        if depth_penalty > 0.0:
            for i, track in enumerate(tracks):
                t_depth = pseudo_depth_score(track.predict())
                for j, det in enumerate(detections):
                    d_depth = det.depth
                    cost[i, j] += depth_penalty * abs(t_depth - d_depth)
        if max_band_delta is not None:
            for i, track in enumerate(tracks):
                t_band = depth_band(pseudo_depth_score(track.predict()), self.depth_bands)
                for j, det in enumerate(detections):
                    d_band = depth_band(det.depth, self.depth_bands)
                    if abs(t_band - d_band) > max_band_delta:
                        cost[i, j] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e5:
                continue
            if ious[r, c] < self.match_thresh:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _update_bookkeeping(self, frame: int) -> None:
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        self.outputs[frame] = {
            t.track_id: t.bbox.tolist()
            for t in self.visible_tracks()
        }

    def step(self, frame: int, detections: Sequence[Detection]) -> None:
        raise NotImplementedError

    def run(self, frames: Sequence[Dict[str, object]]) -> Dict[int, Dict[int, List[float]]]:
        self.reset()
        for record in frames:
            frame = int(record["frame"])
            detections = record["detections_processed"]
            self.step(frame, detections)
        return self.outputs


class SortTracker(OnlineTracker):
    def step(self, frame: int, detections: Sequence[Detection]) -> None:
        active = self.active_tracks()
        high = [d for d in detections if d.score >= self.high_thresh]
        matches, unmatched_tracks, unmatched_dets = self._match(active, high)
        for ti, di in matches:
            active[ti].update(frame, high[di].bbox)
        for ti in unmatched_tracks:
            active[ti].miss()
        for di in unmatched_dets:
            self._new_track(frame, high[di])
        self._update_bookkeeping(frame)


class ByteTrackerLite(OnlineTracker):
    def step(self, frame: int, detections: Sequence[Detection]) -> None:
        active = self.active_tracks()
        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]
        matches, unmatched_tracks, unmatched_high = self._match(active, high)
        for ti, di in matches:
            active[ti].update(frame, high[di].bbox)
        remaining_tracks = [active[i] for i in unmatched_tracks]
        low_matches, unmatched_tracks_2, _ = self._match(remaining_tracks, low)
        for ti, di in low_matches:
            remaining_tracks[ti].update(frame, low[di].bbox)
        matched_track_ids = {active[ti].track_id for ti, _ in matches}
        matched_track_ids.update({remaining_tracks[ti].track_id for ti, _ in low_matches})
        for track in active:
            if track.track_id not in matched_track_ids:
                track.miss()
        for di in unmatched_high:
            self._new_track(frame, high[di])
        self._update_bookkeeping(frame)


class SparseDepthTracker(OnlineTracker):
    def step(self, frame: int, detections: Sequence[Detection]) -> None:
        active = self.active_tracks()
        high = [d for d in detections if d.score >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        matches, unmatched_tracks, unmatched_high = self._match(
            active,
            high,
            depth_penalty=0.8,
            max_band_delta=0,
        )
        for ti, di in matches:
            active[ti].update(frame, high[di].bbox)

        residual_tracks = [active[i] for i in unmatched_tracks]
        residual_high = [high[i] for i in unmatched_high]
        stage2_matches, unmatched_tracks_2, unmatched_high_2 = self._match(
            residual_tracks,
            residual_high,
            depth_penalty=0.35,
            max_band_delta=1,
        )
        for ti, di in stage2_matches:
            residual_tracks[ti].update(frame, residual_high[di].bbox)

        residual_tracks_2 = [residual_tracks[i] for i in unmatched_tracks_2]
        low_matches, unmatched_tracks_3, _ = self._match(
            residual_tracks_2,
            low,
            depth_penalty=0.25,
            max_band_delta=1,
        )
        for ti, di in low_matches:
            residual_tracks_2[ti].update(frame, low[di].bbox)

        matched_track_ids = {active[ti].track_id for ti, _ in matches}
        matched_track_ids.update({residual_tracks[ti].track_id for ti, _ in stage2_matches})
        matched_track_ids.update({residual_tracks_2[ti].track_id for ti, _ in low_matches})

        for track in active:
            if track.track_id not in matched_track_ids:
                track.miss()

        unmatched_high_dets = [residual_high[i] for i in unmatched_high_2]
        medium_births = [
            d
            for d in unmatched_high_dets
            if d.depth < 0.80 or not d.inferred_occluded
        ]
        for det in medium_births:
            self._new_track(frame, det)
        self._update_bookkeeping(frame)


def load_frames() -> List[Dict[str, object]]:
    records = json.loads(DATA_PATH.read_text())
    frames: List[Dict[str, object]] = []
    for record in records:
        gt_boxes = [clamp_bbox(np.array(box, dtype=float)).tolist() for box in record["gt_bboxes"]]
        gt_ids = list(record["gt_ids"])
        occluded_ids = infer_occlusion_ids(gt_boxes, gt_ids)
        detections = []
        for det in record["detections"]:
            bbox = clamp_bbox(np.array(det["bbox"], dtype=float))
            gt_id = det.get("gt_id")
            detections.append(
                Detection(
                    frame=int(record["frame"]),
                    bbox=bbox,
                    score=float(det["score"]),
                    gt_id=int(gt_id) if gt_id is not None else None,
                    inferred_occluded=gt_id in occluded_ids if gt_id is not None else False,
                    depth=pseudo_depth_score(bbox),
                )
            )
        frames.append(
            {
                "frame": int(record["frame"]),
                "gt_bboxes": gt_boxes,
                "gt_ids": gt_ids,
                "occluded_ids": sorted(occluded_ids),
                "detections_processed": detections,
            }
        )
    return frames


def infer_occlusion_ids(gt_boxes: Sequence[Sequence[float]], gt_ids: Sequence[int]) -> set[int]:
    occluded: set[int] = set()
    for i in range(len(gt_boxes)):
        for j in range(i + 1, len(gt_boxes)):
            if iou_single(gt_boxes[i], gt_boxes[j]) >= OCCLUSION_IOU:
                occluded.add(int(gt_ids[i]))
                occluded.add(int(gt_ids[j]))
    return occluded


def summarize_dataset(frames: Sequence[Dict[str, object]]) -> Dict[str, object]:
    det_counts = [len(frame["detections_processed"]) for frame in frames]
    score_rows = []
    occlusion_rates = []
    for frame in frames:
        occluded_ids = set(frame["occluded_ids"])
        occlusion_rates.append(len(occluded_ids) / len(frame["gt_ids"]))
        for det in frame["detections_processed"]:
            score_rows.append(
                {
                    "frame": frame["frame"],
                    "score": det.score,
                    "occluded": "occluded" if det.inferred_occluded else "clear",
                    "depth": det.depth,
                }
            )
    score_df = pd.DataFrame(score_rows)
    stats = {
        "num_frames": len(frames),
        "num_gt_ids": len(set(frames[0]["gt_ids"])),
        "avg_detections_per_frame": format_float(np.mean(det_counts)),
        "min_detections_per_frame": int(np.min(det_counts)),
        "max_detections_per_frame": int(np.max(det_counts)),
        "mean_detection_score": format_float(score_df["score"].mean()),
        "mean_score_clear": format_float(score_df.loc[score_df["occluded"] == "clear", "score"].mean()),
        "mean_score_occluded": format_float(score_df.loc[score_df["occluded"] == "occluded", "score"].mean()),
        "mean_occlusion_fraction": format_float(np.mean(occlusion_rates)),
    }
    return {"stats": stats, "score_df": score_df, "det_counts": det_counts, "occlusion_rates": occlusion_rates}


def evaluate_tracking(
    frames: Sequence[Dict[str, object]],
    tracker_outputs: Dict[int, Dict[int, List[float]]],
    method_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_matches = 0
    total_occluded = 0
    total_occluded_matches = 0
    total_pred = 0
    id_switches = 0
    gt_last_match: Dict[int, int | None] = {}
    gt_last_seen_matched: Dict[int, bool] = {}
    fragments = 0
    match_pairs: Dict[Tuple[int, int], int] = {}
    depth_rows = []
    for frame in frames:
        frame_idx = int(frame["frame"])
        gt_ids = list(map(int, frame["gt_ids"]))
        gt_boxes = list(frame["gt_bboxes"])
        pred_dict = tracker_outputs.get(frame_idx, {})
        pred_ids = list(pred_dict.keys())
        pred_boxes = list(pred_dict.values())
        total_gt += len(gt_ids)
        total_pred += len(pred_ids)

        ious = iou_matrix(gt_boxes, pred_boxes)
        cost = 1.0 - ious
        if len(gt_ids) and len(pred_ids):
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        matches: List[Tuple[int, int]] = []
        matched_gt = set()
        matched_pred = set()
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] < IOU_MATCH:
                continue
            matches.append((r, c))
            matched_gt.add(r)
            matched_pred.add(c)
        fp = len(pred_ids) - len(matched_pred)
        fn = len(gt_ids) - len(matched_gt)
        total_fp += fp
        total_fn += fn
        total_matches += len(matches)

        occluded_ids = set(frame["occluded_ids"])
        for gt_i, gt_id in enumerate(gt_ids):
            is_matched = gt_i in matched_gt
            if gt_id in occluded_ids:
                total_occluded += 1
                if is_matched:
                    total_occluded_matches += 1
            current_match = None
            if is_matched:
                pred_i = next(c for r, c in matches if r == gt_i)
                current_match = pred_ids[pred_i]
                match_pairs[(gt_id, current_match)] = match_pairs.get((gt_id, current_match), 0) + 1
                if gt_id in gt_last_match and gt_last_match[gt_id] is not None and gt_last_match[gt_id] != current_match:
                    id_switches += 1
            if gt_id in gt_last_seen_matched and gt_last_seen_matched[gt_id] is False and is_matched:
                fragments += 1
            gt_last_match[gt_id] = current_match
            gt_last_seen_matched[gt_id] = is_matched

            depth_rows.append(
                {
                    "method": method_name,
                    "frame": frame_idx,
                    "gt_id": gt_id,
                    "depth": pseudo_depth_score(gt_boxes[gt_i]),
                    "matched": int(is_matched),
                    "occluded": int(gt_id in occluded_ids),
                }
            )

    mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt
    recall = total_matches / total_gt
    precision = total_matches / total_pred if total_pred else 0.0
    occ_recall = total_occluded_matches / total_occluded if total_occluded else 0.0

    gt_unique = sorted({gt_id for frame in frames for gt_id in frame["gt_ids"]})
    pred_unique = sorted({pred_id for frame_preds in tracker_outputs.values() for pred_id in frame_preds.keys()})
    id_mat = np.zeros((len(gt_unique), len(pred_unique)), dtype=int)
    gt_index = {gt_id: i for i, gt_id in enumerate(gt_unique)}
    pred_index = {pred_id: i for i, pred_id in enumerate(pred_unique)}
    for (gt_id, pred_id), count in match_pairs.items():
        id_mat[gt_index[gt_id], pred_index[pred_id]] = count
    if id_mat.size:
        row_ind, col_ind = linear_sum_assignment(-id_mat)
        idtp = int(id_mat[row_ind, col_ind].sum())
    else:
        idtp = 0
    idfp = total_pred - idtp
    idfn = total_gt - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0

    metrics = {
        "MOTA": format_float(mota),
        "IDF1": format_float(idf1),
        "Recall": format_float(recall),
        "Precision": format_float(precision),
        "OcclusionRecall": format_float(occ_recall),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "IDSW": int(id_switches),
        "Fragments": int(fragments),
        "Predictions": int(total_pred),
    }
    depth_df = pd.DataFrame(depth_rows)
    return metrics, depth_df


def build_depth_summary(depth_frames: pd.DataFrame, bands: int = 3) -> pd.DataFrame:
    if depth_frames.empty:
        return pd.DataFrame(columns=["method", "band", "recall", "occluded_share"])
    data = depth_frames.dropna(subset=["depth"]).copy()
    data["band"] = pd.qcut(data["depth"], q=bands, labels=[f"band_{i+1}" for i in range(bands)])
    grouped = data.groupby(["method", "band"], observed=False).agg(
        recall=("matched", "mean"),
        occluded_share=("occluded", "mean"),
        count=("gt_id", "size"),
    )
    return grouped.reset_index()


def serialize_tracks(frames: Sequence[Dict[str, object]], outputs: Dict[int, Dict[int, List[float]]]) -> Dict[str, object]:
    tracks: Dict[int, List[Dict[str, object]]] = {}
    for frame in frames:
        frame_idx = int(frame["frame"])
        for track_id, bbox in outputs.get(frame_idx, {}).items():
            tracks.setdefault(track_id, []).append({"frame": frame_idx, "bbox": [format_float(x) for x in bbox]})
    return {
        "num_tracks": len(tracks),
        "tracks": {str(track_id): entries for track_id, entries in sorted(tracks.items())},
    }


def plot_data_overview(profile: Dict[str, object], frames: Sequence[Dict[str, object]]) -> None:
    score_df = profile["score_df"]
    det_counts = profile["det_counts"]
    occlusion_rates = profile["occlusion_rates"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(range(len(det_counts)), det_counts, color="#1f77b4", linewidth=2)
    axes[0, 0].set_title("Detections per frame")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Count")

    sns.histplot(data=score_df, x="score", hue="occluded", bins=25, stat="density", common_norm=False, ax=axes[0, 1])
    axes[0, 1].set_title("Score distribution by inferred occlusion")

    axes[1, 0].plot(range(len(occlusion_rates)), occlusion_rates, color="#d62728", linewidth=2)
    axes[1, 0].set_title("Occluded GT fraction per frame")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Fraction")

    crowded_frame = max(frames, key=lambda f: len(f["occluded_ids"]))
    dets = crowded_frame["detections_processed"]
    axes[1, 1].scatter(
        [d.depth for d in dets],
        [bbox_area(d.bbox) for d in dets],
        c=[d.score for d in dets],
        cmap="viridis",
        s=40,
        alpha=0.8,
    )
    axes[1, 1].set_title(f"Pseudo-depth vs area, frame {crowded_frame['frame']}")
    axes[1, 1].set_xlabel("Pseudo-depth")
    axes[1, 1].set_ylabel("BBox area")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=180)
    plt.close(fig)


def plot_method_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    left = metrics_df.melt(
        id_vars="method",
        value_vars=["MOTA", "IDF1", "Recall", "OcclusionRecall"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=left, x="metric", y="value", hue="method", ax=axes[0])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Tracking quality")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)

    right = metrics_df.melt(
        id_vars="method",
        value_vars=["FP", "FN", "IDSW", "Fragments"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=right, x="metric", y="value", hue="method", ax=axes[1])
    axes[1].set_title("Error decomposition")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "method_comparison.png", dpi=180)
    plt.close(fig)


def plot_depth_summary(depth_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(data=depth_summary, x="band", y="recall", hue="method", ax=axes[0])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Recall by pseudo-depth band")
    axes[0].set_ylabel("Recall")

    sns.barplot(data=depth_summary, x="band", y="occluded_share", hue="method", ax=axes[1])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Occlusion density by band")
    axes[1].set_ylabel("Occluded GT share")

    for ax in axes:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "depth_band_analysis.png", dpi=180)
    plt.close(fig)


def plot_frame_case_study(frames: Sequence[Dict[str, object]]) -> None:
    crowded_frame = max(frames, key=lambda f: len(f["occluded_ids"]))
    detections = crowded_frame["detections_processed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].set_xlim(0, FRAME_W)
    axes[0].set_ylim(FRAME_H, 0)
    axes[0].set_title(f"GT boxes, crowded frame {crowded_frame['frame']}")
    for bbox in crowded_frame["gt_bboxes"][:80]:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, linewidth=0.7, alpha=0.45)
        axes[0].add_patch(rect)

    palette = sns.color_palette("crest", n_colors=3)
    axes[1].set_xlim(0, FRAME_W)
    axes[1].set_ylim(FRAME_H, 0)
    axes[1].set_title("Detections colored by pseudo-depth band")
    for det in detections:
        band = depth_band(det.depth, 3)
        bbox = det.bbox
        rect = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            linewidth=1.0,
            alpha=0.7,
            color=palette[band],
        )
        axes[1].add_patch(rect)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "pseudo_depth_case_study.png", dpi=180)
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=ablation_df, x="depth_bands", y="value", hue="metric", marker="o", ax=ax)
    ax.set_title("Sparse-depth ablation")
    ax.set_ylabel("Score")
    ax.set_xlabel("Number of pseudo-depth bands")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "sparsedepth_ablation.png", dpi=180)
    plt.close(fig)


def run_ablation(frames: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for bands in [2, 3, 4, 5]:
        tracker = SparseDepthTracker(
            name=f"SparseDepth-{bands}",
            high_thresh=0.35,
            low_thresh=0.1,
            match_thresh=0.12,
            max_age=8,
            min_hits=2,
            depth_bands=bands,
        )
        outputs = tracker.run(frames)
        metrics, _ = evaluate_tracking(frames, outputs, tracker.name)
        rows.append({"depth_bands": bands, "metric": "MOTA", "value": metrics["MOTA"]})
        rows.append({"depth_bands": bands, "metric": "IDF1", "value": metrics["IDF1"]})
        rows.append({"depth_bands": bands, "metric": "OcclusionRecall", "value": metrics["OcclusionRecall"]})
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="talk")
    ensure_dirs()

    frames = load_frames()
    profile = summarize_dataset(frames)
    (OUTPUT_DIR / "dataset_profile.json").write_text(json.dumps(profile["stats"], indent=2))

    trackers: List[OnlineTracker] = [
        SortTracker("SORT", high_thresh=0.35, low_thresh=0.1, match_thresh=0.15, max_age=6, min_hits=2),
        ByteTrackerLite("ByteTrack-lite", high_thresh=0.35, low_thresh=0.1, match_thresh=0.12, max_age=8, min_hits=2),
        SparseDepthTracker("SparseDepth", high_thresh=0.35, low_thresh=0.1, match_thresh=0.12, max_age=8, min_hits=2, depth_bands=3),
    ]

    metric_rows = []
    depth_frames = []
    for tracker in trackers:
        outputs = tracker.run(frames)
        metrics, depth_df = evaluate_tracking(frames, outputs, tracker.name)
        metric_rows.append({"method": tracker.name, **metrics})
        depth_frames.append(depth_df)
        (OUTPUT_DIR / f"trajectories_{tracker.name.lower().replace('-', '_')}.json").write_text(
            json.dumps(serialize_tracks(frames, outputs), indent=2)
        )

    metrics_df = pd.DataFrame(metric_rows)
    depth_raw = pd.concat(depth_frames, ignore_index=True)
    depth_summary = build_depth_summary(depth_raw, bands=3)
    ablation_df = run_ablation(frames)

    metrics_df.to_csv(OUTPUT_DIR / "tracking_metrics.csv", index=False)
    depth_summary.to_csv(OUTPUT_DIR / "depth_summary.csv", index=False)
    ablation_df.to_csv(OUTPUT_DIR / "sparsedepth_ablation.csv", index=False)

    plot_data_overview(profile, frames)
    plot_method_comparison(metrics_df)
    plot_depth_summary(depth_summary)
    plot_frame_case_study(frames)
    plot_ablation(ablation_df)

    summary = {
        "dataset": profile["stats"],
        "methods": json.loads(metrics_df.to_json(orient="records")),
        "best_method_by_mota": metrics_df.sort_values("MOTA", ascending=False).iloc[0]["method"],
        "best_method_by_idf1": metrics_df.sort_values("IDF1", ascending=False).iloc[0]["method"],
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
