"""
run_experiments.py
====================
Loads data/simulated_sequence.json, runs ByteTrack & SparseTrack, evaluates
with motmetrics + a custom evaluator that exploits the per-detection gt_id,
and saves results + figures.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from bytetrack import ByteTrackTracker
from sparsetrack import SparseTrackTracker
from tracker_core import iou_matrix


# --------------------------------------------------------------------- I/O
def load_sequence(path):
    with open(path) as f:
        seq = json.load(f)
    seq = sorted(seq, key=lambda x: x["frame"])
    return seq


# --------------------------------------------------------------------- run
def run_tracker(name, tracker, seq):
    """Returns:
      track_log: list of dicts with frame, track_id, bbox, score
    """
    log = []
    for frame in seq:
        f = frame["frame"]
        dets = np.array([d["bbox"] for d in frame["detections"]], dtype=np.float64) if frame["detections"] else np.zeros((0, 4))
        scs = np.array([d["score"] for d in frame["detections"]], dtype=np.float64) if frame["detections"] else np.zeros(0)
        tracks = tracker.update(dets, scs)
        for t in tracks:
            log.append({"frame": f, "track_id": t.track_id, "bbox": t.bbox.tolist(), "score": t.score})
    return log


# ------------------------------------------------------------- evaluation
def eval_motmetrics(seq, log, iou_thresh=0.5, name="tracker"):
    """Standard MOTChallenge metrics via motmetrics, treating gt_ids/gt_bboxes
    in the JSON as ground truth and the tracker log as hypotheses."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=False)
    by_frame = defaultdict(list)
    for r in log:
        by_frame[r["frame"]].append(r)

    for frame in seq:
        f = frame["frame"]
        gt_ids = frame["gt_ids"]
        gt_boxes = np.array(frame["gt_bboxes"], dtype=np.float64)
        h_records = by_frame.get(f, [])
        h_ids = [r["track_id"] for r in h_records]
        h_boxes = np.array([r["bbox"] for r in h_records], dtype=np.float64) if h_records else np.zeros((0, 4))
        # motmetrics expects distance = 1 - IoU, with values > distance threshold treated as no match
        ious = iou_matrix(gt_boxes, h_boxes) if len(gt_boxes) and len(h_boxes) else np.zeros((len(gt_boxes), len(h_boxes)))
        dist = 1.0 - ious
        dist[dist > 1 - iou_thresh] = np.nan
        acc.update(gt_ids, h_ids, dist, frameid=f)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "idf1", "idp", "idr",
            "num_switches", "num_false_positives", "num_misses",
            "mostly_tracked", "mostly_lost", "num_unique_objects",
            "num_fragmentations", "precision", "recall",
        ],
        name=name,
    )
    return summary, acc


def eval_id_consistency(seq, log):
    """Custom evaluator using the per-detection gt_id field.
    For each gt_id we measure (a) how many distinct track_ids the tracker
    used for it (>1 implies fragmentation/ID-switch), (b) the longest
    gt-track length and matched track length, (c) per-frame ID switches."""
    # Build mapping (frame, det_idx) -> gt_id
    det_gt = {}  # (frame, x1,y1,x2,y2) -> gt_id
    for fr in seq:
        for d in fr["detections"]:
            key = (fr["frame"], tuple(np.round(d["bbox"], 3)))
            det_gt[key] = d["gt_id"]

    # For each tracker output, find the gt_id it corresponded to (closest IoU
    # to a det in the same frame). Build per-track gt_id histogram.
    track_to_gtids = defaultdict(list)  # tid -> list of (frame, gt_id)
    by_frame_log = defaultdict(list)
    for r in log:
        by_frame_log[r["frame"]].append(r)

    for fr in seq:
        f = fr["frame"]
        dboxes = np.array([d["bbox"] for d in fr["detections"]], dtype=np.float64) if fr["detections"] else np.zeros((0, 4))
        d_gtids = [d["gt_id"] for d in fr["detections"]]
        for r in by_frame_log.get(f, []):
            if len(dboxes) == 0:
                continue
            box = np.array(r["bbox"]).reshape(1, 4)
            ious = iou_matrix(box, dboxes)[0]
            best = int(np.argmax(ious))
            if ious[best] >= 0.5:
                track_to_gtids[r["track_id"]].append((f, d_gtids[best]))

    # ID switches per frame: count when the gt_id assigned to a track changes
    idsw_per_frame = defaultdict(int)
    last_gt_for_track = {}
    for tid in track_to_gtids:
        seq_pairs = sorted(track_to_gtids[tid])
        prev = None
        for f, gid in seq_pairs:
            if prev is not None and gid != prev:
                idsw_per_frame[f] += 1
            prev = gid

    # Fragmentation per gt_id: how many distinct track_ids covered it
    gt_to_tracks = defaultdict(set)
    for tid, pairs in track_to_gtids.items():
        for _, gid in pairs:
            gt_to_tracks[gid].add(tid)

    return {
        "track_to_gtids": {int(k): v for k, v in track_to_gtids.items()},
        "idsw_per_frame": dict(idsw_per_frame),
        "frag_per_gt": {int(k): len(v) for k, v in gt_to_tracks.items()},
    }


# -------------------------------------------------------------- run main
def main():
    out_dir = os.path.join(ROOT, "outputs")
    img_dir = os.path.join(ROOT, "report", "images")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    seq = load_sequence(os.path.join(ROOT, "data", "simulated_sequence.json"))
    print(f"Loaded {len(seq)} frames; #unique gt_ids={len(set(i for f in seq for i in f['gt_ids']))}")

    # ----- run ByteTrack -----
    # Thresholds adapted to this simulated dataset whose score distribution
    # is bimodal: ~2.3% of detections at score 0.9 ("clean") and the rest
    # in [0.1, 0.4) ("occluded-noisy"). Original ByteTrack defaults
    # (high=0.6, new=0.7) would only ever start ~39/200 tracks; we use a
    # gap-aware split that places the high-bin above the score gap and a
    # new-track threshold low enough to admit occluded targets. Both
    # trackers use the SAME hyper-parameters so the comparison is fair.
    # We use high=0.2, low=0.1 because the simulated detector returns a
    # bimodal score: ~2% clean dets at 0.9 and the rest in [0.1, 0.4]. A
    # high_thresh of 0.2 keeps roughly the upper 35% of detections in the
    # confident bin so that all 200 targets are addressable; the remaining
    # low-score detections are recovered via the second association stage.
    # Both trackers share the SAME hyper-parameters so the comparison is fair.
    common = dict(high_thresh=0.2, low_thresh=0.1, new_track_thresh=0.2,
                  match_thresh=0.8, match_thresh_low=0.5,
                  match_thresh_unconf=0.7, max_time_lost=30)
    bt = ByteTrackTracker(**common)
    bt_log = run_tracker("ByteTrack", bt, seq)
    with open(os.path.join(out_dir, "results_bytetrack.json"), "w") as f:
        json.dump(bt_log, f)

    # ----- run SparseTrack (default 4 levels) -----
    st = SparseTrackTracker(**common, n_levels=4)
    st_log = run_tracker("SparseTrack(K=4)", st, seq)
    with open(os.path.join(out_dir, "results_sparsetrack.json"), "w") as f:
        json.dump(st_log, f)

    # ----- evaluate -----
    bt_sum, bt_acc = eval_motmetrics(seq, bt_log, name="ByteTrack")
    st_sum, st_acc = eval_motmetrics(seq, st_log, name="SparseTrack(K=4)")

    cmp = pd.concat([bt_sum, st_sum])
    cmp.to_csv(os.path.join(out_dir, "comparison_metrics.csv"))
    print("\n=== Comparison ===")
    print(cmp.to_string())

    # ----- custom ID metrics -----
    bt_id = eval_id_consistency(seq, bt_log)
    st_id = eval_id_consistency(seq, st_log)
    with open(os.path.join(out_dir, "id_consistency_bytetrack.json"), "w") as f:
        json.dump({k: (v if k != "track_to_gtids" else {str(a): b for a, b in v.items()}) for k, v in bt_id.items()}, f, default=str)
    with open(os.path.join(out_dir, "id_consistency_sparsetrack.json"), "w") as f:
        json.dump({k: (v if k != "track_to_gtids" else {str(a): b for a, b in v.items()}) for k, v in st_id.items()}, f, default=str)

    # per-frame idsw csv
    frames = [fr["frame"] for fr in seq]
    rows = []
    for f in frames:
        rows.append({
            "frame": f,
            "bytetrack_idsw": bt_id["idsw_per_frame"].get(f, 0),
            "sparsetrack_idsw": st_id["idsw_per_frame"].get(f, 0),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_frame_idsw.csv"), index=False)

    # ----- ablation: SparseTrack levels sweep -----
    sweep_rows = []
    for K in [1, 2, 3, 4, 6, 8]:
        from tracker_core import STrack
        st_k = SparseTrackTracker(**common, n_levels=K)
        log_k = run_tracker(f"SparseTrack(K={K})", st_k, seq)
        sm, _ = eval_motmetrics(seq, log_k, name=f"SparseTrack(K={K})")
        sweep_rows.append({
            "K": K,
            "MOTA": float(sm["mota"].iloc[0]),
            "IDF1": float(sm["idf1"].iloc[0]),
            "IDsw": int(sm["num_switches"].iloc[0]),
            "FP": int(sm["num_false_positives"].iloc[0]),
            "FN": int(sm["num_misses"].iloc[0]),
            "MT": int(sm["mostly_tracked"].iloc[0]),
            "ML": int(sm["mostly_lost"].iloc[0]),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(out_dir, "sparsetrack_levels_sweep.csv"), index=False)
    print("\n=== K sweep ===")
    print(sweep_df.to_string(index=False))

    # ====================== FIGURES ======================
    # 1. data overview
    n_dets = [len(fr["detections"]) for fr in seq]
    all_scores = np.array([d["score"] for fr in seq for d in fr["detections"]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frames, n_dets, "-o", color="tab:blue", ms=3)
    axes[0].axhline(200, color="k", ls="--", alpha=0.4, label="GT objects=200")
    axes[0].set_xlabel("Frame"); axes[0].set_ylabel("# detections")
    axes[0].set_title("Detections per frame"); axes[0].legend()
    axes[1].hist(all_scores, bins=40, color="tab:orange", alpha=0.85)
    axes[1].axvline(0.2, color="red", ls="--", label="high_thresh=0.2")
    axes[1].axvline(0.1, color="green", ls="--", label="low_thresh=0.1")
    axes[1].set_xlabel("Detection score"); axes[1].set_ylabel("Count")
    axes[1].set_title("Detection-score distribution"); axes[1].legend()
    # Pseudo-depth histogram
    bot_y = np.array([d["bbox"][3] for fr in seq for d in fr["detections"]])
    axes[2].hist(bot_y, bins=40, color="tab:green", alpha=0.85)
    axes[2].set_xlabel("bbox bottom-y (pseudo-depth proxy)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Pseudo-depth distribution (all detections)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "data_overview.png"), dpi=130); plt.close()

    # 2. score vs pseudo-depth scatter (joint)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bot_y, all_scores, s=5, alpha=0.25, color="navy")
    ax.set_xlabel("pseudo-depth (bbox bottom-y)")
    ax.set_ylabel("detection score")
    ax.set_title("Joint: detection score vs pseudo-depth")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "score_pseudo_depth.png"), dpi=130); plt.close()

    # 3. main comparison bar chart
    metrics = ["MOTA", "IDF1", "IDsw", "FN", "FP", "MT", "ML"]
    bt_vals = [
        float(bt_sum["mota"].iloc[0]),
        float(bt_sum["idf1"].iloc[0]),
        int(bt_sum["num_switches"].iloc[0]),
        int(bt_sum["num_misses"].iloc[0]),
        int(bt_sum["num_false_positives"].iloc[0]),
        int(bt_sum["mostly_tracked"].iloc[0]),
        int(bt_sum["mostly_lost"].iloc[0]),
    ]
    st_vals = [
        float(st_sum["mota"].iloc[0]),
        float(st_sum["idf1"].iloc[0]),
        int(st_sum["num_switches"].iloc[0]),
        int(st_sum["num_misses"].iloc[0]),
        int(st_sum["num_false_positives"].iloc[0]),
        int(st_sum["mostly_tracked"].iloc[0]),
        int(st_sum["mostly_lost"].iloc[0]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # First subplot: MOTA + IDF1
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w/2, [bt_vals[0], bt_vals[1]], w, label="ByteTrack", color="tab:blue")
    axes[0].bar(x + w/2, [st_vals[0], st_vals[1]], w, label="SparseTrack", color="tab:red")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["MOTA", "IDF1"])
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Score")
    axes[0].set_title("Tracking quality (higher is better)"); axes[0].legend()
    for i, (a, b) in enumerate(zip([bt_vals[0], bt_vals[1]], [st_vals[0], st_vals[1]])):
        axes[0].text(i - w/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        axes[0].text(i + w/2, b + 0.02, f"{b:.3f}", ha="center", fontsize=9)
    # Second subplot: counts
    cmetrics = ["IDsw", "FN", "FP", "MT", "ML"]
    cb = bt_vals[2:]; cs = st_vals[2:]
    x = np.arange(len(cmetrics))
    axes[1].bar(x - w/2, cb, w, label="ByteTrack", color="tab:blue")
    axes[1].bar(x + w/2, cs, w, label="SparseTrack", color="tab:red")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cmetrics)
    axes[1].set_title("Counts")
    axes[1].legend()
    for i, (a, b) in enumerate(zip(cb, cs)):
        axes[1].text(i - w/2, a + max(cb+cs) * 0.01, f"{a}", ha="center", fontsize=9)
        axes[1].text(i + w/2, b + max(cb+cs) * 0.01, f"{b}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "metric_comparison.png"), dpi=130); plt.close()

    # 4. per-frame ID switches
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, [bt_id["idsw_per_frame"].get(f, 0) for f in frames], "-o",
            label=f"ByteTrack (total={sum(bt_id['idsw_per_frame'].values())})",
            color="tab:blue", ms=4, alpha=0.8)
    ax.plot(frames, [st_id["idsw_per_frame"].get(f, 0) for f in frames], "-s",
            label=f"SparseTrack (total={sum(st_id['idsw_per_frame'].values())})",
            color="tab:red", ms=4, alpha=0.8)
    ax.set_xlabel("Frame"); ax.set_ylabel("ID switches at frame")
    ax.set_title("Per-frame ID switches (custom evaluator using gt_id)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "idsw_per_frame.png"), dpi=130); plt.close()

    # 5. K sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    ax.plot(sweep_df["K"], sweep_df["MOTA"], "-o", color="tab:blue", label="MOTA")
    ax.plot(sweep_df["K"], sweep_df["IDF1"], "-s", color="tab:green", label="IDF1")
    ax2.plot(sweep_df["K"], sweep_df["IDsw"], "-^", color="tab:red", label="IDsw")
    ax.set_xlabel("Number of depth levels K")
    ax.set_ylabel("MOTA / IDF1")
    ax2.set_ylabel("ID switches")
    # legends
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, loc="best")
    ax.set_title("SparseTrack ablation: number of depth levels")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "sparsetrack_levels_sweep.png"), dpi=130); plt.close()

    # 6. qualitative tracks: pick a subset of gt_ids, show GT box + assigned track id over time
    # Choose 6 GT ids that are heavily occluded -> highest fragmentation under ByteTrack
    bt_frag = sorted(bt_id["frag_per_gt"].items(), key=lambda x: -x[1])
    chosen = [k for k, _ in bt_frag[:6]] if bt_frag else list(range(6))

    # Build per-gt timeline of track ids for each tracker
    def timeline_for(gt_ids, tracker_id_map):
        # tracker_id_map: track_to_gtids dict
        per_gt = {g: {} for g in gt_ids}  # gt_id -> {frame: track_id}
        for tid, pairs in tracker_id_map.items():
            for f, gid in pairs:
                if gid in per_gt:
                    per_gt[gid][f] = tid
        return per_gt

    bt_tl = timeline_for(chosen, bt_id["track_to_gtids"])
    st_tl = timeline_for(chosen, st_id["track_to_gtids"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    color_pool = plt.cm.tab20(np.linspace(0, 1, 20))
    def plot_tl(ax, tl, title):
        for i, gid in enumerate(chosen):
            for f, tid in tl[gid].items():
                ax.scatter(f, i, color=color_pool[tid % 20], s=22)
        ax.set_yticks(range(len(chosen)))
        ax.set_yticklabels([f"gt={g}" for g in chosen])
        ax.set_title(title); ax.grid(alpha=0.3)
    plot_tl(axes[0], bt_tl, "ByteTrack: track-id assigned to each chosen GT id over time")
    plot_tl(axes[1], st_tl, "SparseTrack: track-id assigned to each chosen GT id over time")
    axes[1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "qualitative_tracks.png"), dpi=130); plt.close()

    # save final summary json
    summary_out = {
        "ByteTrack": {k: float(bt_sum[k.lower() if k.islower() else k].iloc[0]) if k.lower() in bt_sum.columns else None for k in []},
    }
    summary_out = {
        "ByteTrack": bt_sum.iloc[0].to_dict(),
        "SparseTrack_K4": st_sum.iloc[0].to_dict(),
        "K_sweep": sweep_df.to_dict(orient="records"),
    }
    # sanitize
    def sanitize(o):
        if isinstance(o, dict):
            return {str(k): sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(sanitize(summary_out), f, indent=2)
    print("Saved figures and outputs.")


if __name__ == "__main__":
    main()
