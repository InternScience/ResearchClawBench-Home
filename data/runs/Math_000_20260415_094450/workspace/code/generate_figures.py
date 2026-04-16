"""
Generate figures for the research report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from collections import defaultdict

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = '../report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
with open('../data/simulated_sequence.json', 'r') as f:
    sequence_data = json.load(f)

with open('../outputs/metrics_comparison.json', 'r') as f:
    metrics = json.load(f)

with open('../outputs/ablation_depth_layers.json', 'r') as f:
    ablation_depth = json.load(f)

with open('../outputs/ablation_score_threshold.json', 'r') as f:
    ablation_score = json.load(f)

with open('../outputs/ablation_iou_threshold.json', 'r') as f:
    ablation_iou = json.load(f)


# ============================================================
# Figure 1: Dataset Overview - Detection Score Distribution
# ============================================================
def fig_score_distribution():
    all_scores = []
    for frame in sequence_data:
        for d in frame['detections']:
            all_scores.append(d['score'])
    all_scores = np.array(all_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(all_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=0.2, color='red', linestyle='--', label='Score=0.2')
    axes[0].axvline(x=0.3, color='orange', linestyle='--', label='Score=0.3')
    axes[0].set_xlabel('Detection Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Detection Score Distribution')
    axes[0].legend()
    
    # Score categories pie chart
    cats = [
        ('Low (<0.2)', (all_scores < 0.2).sum()),
        ('Medium (0.2-0.3)', ((all_scores >= 0.2) & (all_scores < 0.3)).sum()),
        ('Medium-High (0.3-0.4)', ((all_scores >= 0.3) & (all_scores < 0.4)).sum()),
        ('High (>=0.4)', (all_scores >= 0.4).sum()),
    ]
    labels = [c[0] for c in cats]
    sizes = [c[1] for c in cats]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Detection Score Categories')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_score_distribution.png'))
    plt.close()
    print("Saved fig1_score_distribution.png")


# ============================================================
# Figure 2: Occlusion Analysis
# ============================================================
def fig_occlusion_analysis():
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = max(0, (box1[2]-box1[0]) * (box1[3]-box1[1]))
        area2 = max(0, (box2[2]-box2[0]) * (box2[3]-box2[1]))
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # Per-frame occlusion statistics
    frames = []
    occluded_pairs_per_frame = []
    max_ious_per_frame = []
    mean_ious_per_frame = []
    
    for frame in sequence_data:
        bboxes = frame['gt_bboxes']
        n = len(bboxes)
        frames.append(frame['frame'])
        
        ious = []
        for i in range(n):
            for j in range(i+1, n):
                ious.append(compute_iou(bboxes[i], bboxes[j]))
        
        ious = np.array(ious)
        occluded_pairs_per_frame.append((ious > 0.2).sum())
        max_ious_per_frame.append(ious.max() if len(ious) > 0 else 0)
        mean_ious_per_frame.append(ious.mean() if len(ious) > 0 else 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(frames, occluded_pairs_per_frame, color='steelblue', linewidth=1.5)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Number of Occluded Pairs (IoU>0.2)')
    axes[0].set_title('Occluded Pairs Over Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(frames, max_ious_per_frame, color='red', linewidth=1.5)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Max IoU')
    axes[1].set_title('Maximum Pairwise IoU Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # Distribution of pairwise IoUs (sampled)
    all_ious = []
    for frame in sequence_data[:20]:  # Sample 20 frames
        bboxes = frame['gt_bboxes']
        n = len(bboxes)
        for i in range(n):
            for j in range(i+1, n):
                all_ious.append(compute_iou(bboxes[i], bboxes[j]))
    
    axes[2].hist(all_ious, bins=50, color='coral', edgecolor='white', alpha=0.8)
    axes[2].axvline(x=0.2, color='blue', linestyle='--', label='IoU=0.2')
    axes[2].set_xlabel('Pairwise IoU')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Pairwise IoU (Sampled)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_occlusion_analysis.png'))
    plt.close()
    print("Saved fig2_occlusion_analysis.png")


# ============================================================
# Figure 3: Main Results Comparison Bar Chart
# ============================================================
def fig_main_comparison():
    methods = ['SORT', 'ByteTrack', 'SparseTrack_L3', 'SparseTrack_L5']
    labels = ['SORT', 'ByteTrack', 'SparseTrack\n(L=3)', 'SparseTrack\n(L=5)']
    
    mota_vals = [metrics[m]['MOTA'] for m in methods]
    motp_vals = [metrics[m]['MOTP'] for m in methods]
    idf1_vals = [metrics[m]['IDF1'] for m in methods]
    idsw_vals = [metrics[m]['ID_Switches'] for m in methods]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    
    # MOTA
    bars = axes[0, 0].bar(labels, mota_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[0, 0].set_ylabel('MOTA')
    axes[0, 0].set_title('MOTA (Higher is Better)')
    axes[0, 0].set_ylim(0, max(mota_vals) * 1.2)
    for bar, val in zip(bars, mota_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MOTP
    bars = axes[0, 1].bar(labels, motp_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[0, 1].set_ylabel('MOTP')
    axes[0, 1].set_title('MOTP (Higher is Better)')
    axes[0, 1].set_ylim(min(motp_vals) * 0.95, max(motp_vals) * 1.02)
    for bar, val in zip(bars, motp_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # IDF1
    bars = axes[1, 0].bar(labels, idf1_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[1, 0].set_ylabel('IDF1')
    axes[1, 0].set_title('IDF1 (Higher is Better)')
    axes[1, 0].set_ylim(0, max(idf1_vals) * 1.2)
    for bar, val in zip(bars, idf1_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # ID Switches
    bars = axes[1, 1].bar(labels, idsw_vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[1, 1].set_ylabel('ID Switches')
    axes[1, 1].set_title('ID Switches (Lower is Better)')
    for bar, val in zip(bars, idsw_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                        f'{val}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Multi-Object Tracking Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_main_comparison.png'))
    plt.close()
    print("Saved fig3_main_comparison.png")


# ============================================================
# Figure 4: Ablation - Number of Depth Layers
# ============================================================
def fig_ablation_depth():
    layers = sorted([int(k) for k in ablation_depth.keys()])
    mota_vals = [ablation_depth[str(l)]['MOTA'] for l in layers]
    idf1_vals = [ablation_depth[str(l)]['IDF1'] for l in layers]
    idsw_vals = [ablation_depth[str(l)]['ID_Switches'] for l in layers]
    mt_vals = [ablation_depth[str(l)]['Mostly_Tracked'] for l in layers]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(layers, mota_vals, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Depth Layers')
    axes[0, 0].set_ylabel('MOTA')
    axes[0, 0].set_title('MOTA vs. Depth Layers')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(layers, idf1_vals, 's-', color='#55A868', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Depth Layers')
    axes[0, 1].set_ylabel('IDF1')
    axes[0, 1].set_title('IDF1 vs. Depth Layers')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(layers, idsw_vals, '^-', color='#C44E52', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Depth Layers')
    axes[1, 0].set_ylabel('ID Switches')
    axes[1, 0].set_title('ID Switches vs. Depth Layers')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(layers, mt_vals, 'D-', color='#8172B2', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Depth Layers')
    axes[1, 1].set_ylabel('Mostly Tracked')
    axes[1, 1].set_title('Mostly Tracked vs. Depth Layers')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('SparseTrack Ablation: Effect of Depth Layers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_ablation_depth.png'))
    plt.close()
    print("Saved fig4_ablation_depth.png")


# ============================================================
# Figure 5: Ablation - Score Threshold
# ============================================================
def fig_ablation_score():
    thresholds = sorted([float(k) for k in ablation_score.keys()])
    mota_vals = [ablation_score[str(t)]['MOTA'] for t in thresholds]
    idf1_vals = [ablation_score[str(t)]['IDF1'] for t in thresholds]
    idsw_vals = [ablation_score[str(t)]['ID_Switches'] for t in thresholds]
    mt_vals = [ablation_score[str(t)]['Mostly_Tracked'] for t in thresholds]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(thresholds, mota_vals, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Score Threshold')
    axes[0, 0].set_ylabel('MOTA')
    axes[0, 0].set_title('MOTA vs. Score Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(thresholds, idf1_vals, 's-', color='#55A868', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Score Threshold')
    axes[0, 1].set_ylabel('IDF1')
    axes[0, 1].set_title('IDF1 vs. Score Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(thresholds, idsw_vals, '^-', color='#C44E52', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Score Threshold')
    axes[1, 0].set_ylabel('ID Switches')
    axes[1, 0].set_title('ID Switches vs. Score Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(thresholds, mt_vals, 'D-', color='#8172B2', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Score Threshold')
    axes[1, 1].set_ylabel('Mostly Tracked')
    axes[1, 1].set_title('Mostly Tracked vs. Score Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('ByteTrack Ablation: Effect of Score Threshold', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_ablation_score.png'))
    plt.close()
    print("Saved fig5_ablation_score.png")


# ============================================================
# Figure 6: IoU Threshold Ablation (All Methods)
# ============================================================
def fig_ablation_iou():
    iou_thresholds = sorted([float(k) for k in ablation_iou.keys()])
    
    sort_mota = [ablation_iou[str(t)]['SORT']['MOTA'] for t in iou_thresholds]
    bt_mota = [ablation_iou[str(t)]['ByteTrack']['MOTA'] for t in iou_thresholds]
    st_mota = [ablation_iou[str(t)]['SparseTrack']['MOTA'] for t in iou_thresholds]
    
    sort_idf1 = [ablation_iou[str(t)]['SORT']['IDF1'] for t in iou_thresholds]
    bt_idf1 = [ablation_iou[str(t)]['ByteTrack']['IDF1'] for t in iou_thresholds]
    st_idf1 = [ablation_iou[str(t)]['SparseTrack']['IDF1'] for t in iou_thresholds]
    
    sort_ids = [ablation_iou[str(t)]['SORT']['ID_Switches'] for t in iou_thresholds]
    bt_ids = [ablation_iou[str(t)]['ByteTrack']['ID_Switches'] for t in iou_thresholds]
    st_ids = [ablation_iou[str(t)]['SparseTrack']['ID_Switches'] for t in iou_thresholds]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(iou_thresholds, sort_mota, 'o-', label='SORT', linewidth=2)
    axes[0].plot(iou_thresholds, bt_mota, 's-', label='ByteTrack', linewidth=2)
    axes[0].plot(iou_thresholds, st_mota, '^-', label='SparseTrack', linewidth=2)
    axes[0].set_xlabel('IoU Threshold')
    axes[0].set_ylabel('MOTA')
    axes[0].set_title('MOTA vs. IoU Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iou_thresholds, sort_idf1, 'o-', label='SORT', linewidth=2)
    axes[1].plot(iou_thresholds, bt_idf1, 's-', label='ByteTrack', linewidth=2)
    axes[1].plot(iou_thresholds, st_idf1, '^-', label='SparseTrack', linewidth=2)
    axes[1].set_xlabel('IoU Threshold')
    axes[1].set_ylabel('IDF1')
    axes[1].set_title('IDF1 vs. IoU Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(iou_thresholds, sort_ids, 'o-', label='SORT', linewidth=2)
    axes[2].plot(iou_thresholds, bt_ids, 's-', label='ByteTrack', linewidth=2)
    axes[2].plot(iou_thresholds, st_ids, '^-', label='SparseTrack', linewidth=2)
    axes[2].set_xlabel('IoU Threshold')
    axes[2].set_ylabel('ID Switches')
    axes[2].set_title('ID Switches vs. IoU Threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('IoU Threshold Sensitivity Across Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_ablation_iou.png'))
    plt.close()
    print("Saved fig6_ablation_iou.png")


# ============================================================
# Figure 7: Spatial Visualization with Pseudo-Depth
# ============================================================
def fig_spatial_depth():
    frame_data = sequence_data[0]
    gt_bboxes = frame_data['gt_bboxes']
    gt_ids = frame_data['gt_ids']
    detections = frame_data['detections']
    
    # Compute pseudo-depth for GT
    depths = []
    for bbox in gt_bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = w * h
        depth = -np.log(max(area, 1.0) + 1.0)
        depths.append(depth)
    depths = np.array(depths)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot GT bboxes colored by depth
    ax = axes[0]
    norm_depth = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
    cmap = plt.cm.RdYlGn_r  # Red=close, Green=far
    
    for i, bbox in enumerate(gt_bboxes):
        color = cmap(norm_depth[i])
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 650)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Ground Truth BBoxes (Colored by Pseudo-Depth)')
    ax.invert_yaxis()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(depths.min(), depths.max()))
    plt.colorbar(sm, ax=ax, label='Pseudo-Depth (lower=closer)')
    
    # Plot detections colored by score
    ax = axes[1]
    scores = np.array([d['score'] for d in detections])
    det_bboxes = [d['bbox'] for d in detections]
    cmap_score = plt.cm.viridis
    
    for i, (bbox, score) in enumerate(zip(det_bboxes, scores)):
        color = cmap_score(score / max(scores.max(), 0.01))
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                         linewidth=1.0, edgecolor=color, facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 650)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Detection BBoxes (Colored by Score)')
    ax.invert_yaxis()
    
    sm2 = plt.cm.ScalarMappable(cmap=cmap_score, norm=plt.Normalize(0, scores.max()))
    plt.colorbar(sm2, ax=ax, label='Detection Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_spatial_depth.png'))
    plt.close()
    print("Saved fig7_spatial_depth.png")


# ============================================================
# Figure 8: FP/FN/IDSW Breakdown
# ============================================================
def fig_error_breakdown():
    methods = ['SORT', 'ByteTrack', 'SparseTrack_L3', 'SparseTrack_L5']
    labels = ['SORT', 'ByteTrack', 'SparseTrack\n(L=3)', 'SparseTrack\n(L=5)']
    
    fp_vals = [metrics[m]['FP'] for m in methods]
    fn_vals = [metrics[m]['FN'] for m in methods]
    idsw_vals = [metrics[m]['ID_Switches'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, fp_vals, width, label='False Positives', color='#4C72B0')
    bars2 = ax.bar(x, fn_vals, width, label='False Negatives', color='#55A868')
    bars3 = ax.bar(x + width, idsw_vals, width, label='ID Switches', color='#C44E52')
    
    ax.set_ylabel('Count')
    ax.set_title('Error Breakdown by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_error_breakdown.png'))
    plt.close()
    print("Saved fig8_error_breakdown.png")


# ============================================================
# Figure 9: Tracking Trajectory Visualization
# ============================================================
def fig_trajectory_viz():
    # Load trajectories
    with open('../outputs/trajectories/SORT_trajectories.json', 'r') as f:
        sort_trajs = json.load(f)
    with open('../outputs/trajectories/SparseTrack_L3_trajectories.json', 'r') as f:
        sparsedet_trajs = json.load(f)
    
    # Pick a few GT IDs and visualize their trajectories
    # Get GT trajectories for specific IDs
    target_gt_ids = [0, 1, 2, 3, 4]  # First 5 objects
    
    # Build GT trajectories
    gt_trajs = defaultdict(list)
    for frame in sequence_data:
        for bbox, gid in zip(frame['gt_bboxes'], frame['gt_ids']):
            if gid in target_gt_ids:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                gt_trajs[gid].append((frame['frame'], cx, cy))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_gt_ids)))
    
    for idx, gid in enumerate(target_gt_ids):
        gt_data = gt_trajs[gid]
        frames = [d[0] for d in gt_data]
        xs = [d[1] for d in gt_data]
        ys = [d[2] for d in gt_data]
        axes[0].plot(xs, ys, '-', color=colors[idx], alpha=0.5, linewidth=2, label=f'GT ID {gid}')
    
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].set_title('Ground Truth Trajectories (Sample)')
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    
    # For SparseTrack, show some tracked trajectories
    # Pick the longest trajectories
    traj_lengths = {tid: len(frames) for tid, frames in sparsedet_trajs.items()}
    top_trajs = sorted(traj_lengths.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for idx, (tid, _) in enumerate(top_trajs):
        frames_data = sparsedet_trajs[tid]
        xs = [(f[1][0] + f[1][2]) / 2 for f in frames_data]
        ys = [(f[1][1] + f[1][3]) / 2 for f in frames_data]
        frames = [f[0] for f in frames_data]
        axes[1].plot(xs, ys, '-', color=colors[idx], alpha=0.7, linewidth=2, label=f'Track {tid}')
    
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    axes[1].set_title('SparseTrack Trajectories (Top 5 Longest)')
    axes[1].legend(fontsize=8)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_trajectory_viz.png'))
    plt.close()
    print("Saved fig9_trajectory_viz.png")


# ============================================================
# Figure 10: Radar Chart Comparison
# ============================================================
def fig_radar_chart():
    methods = ['SORT', 'ByteTrack', 'SparseTrack_L3', 'SparseTrack_L5']
    labels = ['SORT', 'ByteTrack', 'SparseTrack\n(L=3)', 'SparseTrack\n(L=5)']
    
    # Normalize metrics to [0, 1] for radar chart
    categories = ['MOTA', 'MOTP', 'IDF1', 'MT Ratio', 'Low IDSW']
    
    # Get values
    mota = [metrics[m]['MOTA'] for m in methods]
    motp = [metrics[m]['MOTP'] for m in methods]
    idf1 = [metrics[m]['IDF1'] for m in methods]
    mt_ratio = [metrics[m]['Mostly_Tracked'] / 200.0 for m in methods]
    max_ids = max(metrics[m]['ID_Switches'] for m in methods)
    idsw_inv = [1.0 - metrics[m]['ID_Switches'] / max_ids for m in methods]
    
    values = np.array([mota, motp, idf1, mt_ratio, idsw_inv])
    
    # Number of variables
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for i, (method, label) in enumerate(zip(methods, labels)):
        vals = [mota[i], motp[i], idf1[i], mt_ratio[i], idsw_inv[i]]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Method Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_radar_chart.png'))
    plt.close()
    print("Saved fig10_radar_chart.png")


# ============================================================
# Figure 11: Detection Rate and Tracking Rate Over Time
# ============================================================
def fig_tracking_over_time():
    # Compute per-frame tracking metrics for each method
    methods_data = {}
    for method_name in ['SORT', 'ByteTrack', 'SparseTrack_L3']:
        traj_file = f'../outputs/trajectories/{method_name}_trajectories.json'
        with open(traj_file, 'r') as f:
            trajs = json.load(f)
        
        # Build per-frame tracking count
        frame_track_count = defaultdict(int)
        for tid, frames_bboxes in trajs.items():
            for f, bbox in frames_bboxes:
                frame_track_count[f] += 1
        
        methods_data[method_name] = frame_track_count
    
    # GT count per frame
    gt_count = {frame['frame']: len(frame['gt_ids']) for frame in sequence_data}
    det_count = {frame['frame']: len(frame['detections']) for frame in sequence_data}
    
    frames = sorted(gt_count.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, [gt_count[f] for f in frames], 'k-', linewidth=2, label='Ground Truth (200)', alpha=0.5)
    ax.plot(frames, [det_count[f] for f in frames], 'gray', linewidth=1.5, label='Detections', alpha=0.5)
    
    colors = {'SORT': '#4C72B0', 'ByteTrack': '#55A868', 'SparseTrack_L3': '#C44E52'}
    for method, data in methods_data.items():
        counts = [data.get(f, 0) for f in frames]
        ax.plot(frames, counts, '-', linewidth=1.5, label=method, color=colors[method])
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of Tracked Objects')
    ax.set_title('Tracking Coverage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_tracking_over_time.png'))
    plt.close()
    print("Saved fig11_tracking_over_time.png")


# Run all figures
fig_score_distribution()
fig_occlusion_analysis()
fig_main_comparison()
fig_ablation_depth()
fig_ablation_score()
fig_ablation_iou()
fig_spatial_depth()
fig_error_breakdown()
fig_trajectory_viz()
fig_radar_chart()
fig_tracking_over_time()

print("\nAll figures generated successfully!")
