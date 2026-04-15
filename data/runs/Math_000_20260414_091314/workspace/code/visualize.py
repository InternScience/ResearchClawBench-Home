"""
Visualization script for SparseTrack vs ByteTrack comparison.
Generates all figures for the research report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
import os

# Paths
data_path = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/data/simulated_sequence.json'
output_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/outputs'
images_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_000_20260414_091314/report/images'
os.makedirs(images_dir, exist_ok=True)

# Load data
with open(data_path) as f:
    data = json.load(f)
with open(f'{output_dir}/tracking_results.json') as f:
    results = json.load(f)
with open(f'{output_dir}/per_object_recall.json') as f:
    recall_data = json.load(f)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ============================================================
# Figure 1: Data Overview
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1a: Detection count per frame
det_counts = [len(f['detections']) for f in data]
gt_counts = [len(f['gt_bboxes']) for f in data]
frames = [f['frame'] for f in data]
axes[0].plot(frames, gt_counts, 'b-', label='Ground Truth', linewidth=2)
axes[0].plot(frames, det_counts, 'r-', label='Detections', linewidth=2)
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Detection Count per Frame')
axes[0].legend()
axes[0].set_ylim(0, 250)

# 1b: Score distribution
all_scores = []
for f in data:
    for d in f['detections']:
        all_scores.append(d['score'])
axes[1].hist(all_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0.3, color='red', linestyle='--', label='High threshold (0.3)')
axes[1].axvline(x=0.05, color='orange', linestyle='--', label='Low threshold (0.05)')
axes[1].set_xlabel('Detection Confidence Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('(b) Detection Score Distribution')
axes[1].legend(fontsize=9)

# 1c: Bounding box area distribution
all_areas = []
for f in data:
    for bbox in f['gt_bboxes']:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        all_areas.append(w * h)
axes[2].hist(all_areas, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('Bounding Box Area (pixels^2)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('(c) Bounding Box Area Distribution')

plt.tight_layout()
plt.savefig(f'{images_dir}/data_overview.png', bbox_inches='tight')
plt.close()
print("Figure 1: data_overview.png saved")

# ============================================================
# Figure 2: MOT Metrics Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2a: Bar chart of main metrics
metrics_names = ['MOTA', 'MOTP', 'IDF1']
sparse_vals = [results['SparseTrack']['metrics'][m] for m in metrics_names]
byte_vals = [results['ByteTrack']['metrics'][m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35
bars1 = axes[0].bar(x - width/2, sparse_vals, width, label='SparseTrack', color='#2196F3', edgecolor='white')
bars2 = axes[0].bar(x + width/2, byte_vals, width, label='ByteTrack', color='#FF9800', edgecolor='white')
axes[0].set_ylabel('Score')
axes[0].set_title('(a) Tracking Quality Metrics')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].legend()
axes[0].set_ylim(0, 1.0)
for bar, val in zip(bars1, sparse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
for bar, val in zip(bars2, byte_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)

# 2b: Error decomposition
error_names = ['FP', 'FN', 'ID Switches']
sparse_errors = [results['SparseTrack']['metrics'][e.replace(' ', '_')] if ' ' not in e else results['SparseTrack']['metrics']['ID_switches'] for e in error_names]
byte_errors = [results['ByteTrack']['metrics'][e.replace(' ', '_')] if ' ' not in e else results['ByteTrack']['metrics']['ID_switches'] for e in error_names]

# Fix the mapping
sparse_errors = [results['SparseTrack']['metrics']['FP'], results['SparseTrack']['metrics']['FN'], results['SparseTrack']['metrics']['ID_switches']]
byte_errors = [results['ByteTrack']['metrics']['FP'], results['ByteTrack']['metrics']['FN'], results['ByteTrack']['metrics']['ID_switches']]

x = np.arange(len(error_names))
bars1 = axes[1].bar(x - width/2, sparse_errors, width, label='SparseTrack', color='#2196F3', edgecolor='white')
bars2 = axes[1].bar(x + width/2, byte_errors, width, label='ByteTrack', color='#FF9800', edgecolor='white')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Error Decomposition')
axes[1].set_xticks(x)
axes[1].set_xticklabels(error_names)
axes[1].legend()
for bar, val in zip(bars1, sparse_errors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val}', ha='center', fontsize=9)
for bar, val in zip(bars2, byte_errors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{images_dir}/metrics_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 2: metrics_comparison.png saved")

# ============================================================
# Figure 3: Occlusion-Level Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

sparse_occ = results['SparseTrack']['occlusion_metrics']
byte_occ = results['ByteTrack']['occlusion_metrics']

levels = sorted([int(k) for k in sparse_occ.keys()])
sparse_rates = [sparse_occ[str(l)]['tracking_rate'] for l in levels]
byte_rates = [byte_occ[str(l)]['tracking_rate'] for l in levels]
counts = [sparse_occ[str(l)]['count'] for l in levels]

ax.plot(levels, sparse_rates, 'o-', color='#2196F3', label='SparseTrack', linewidth=2, markersize=8)
ax.plot(levels, byte_rates, 's-', color='#FF9800', label='ByteTrack', linewidth=2, markersize=8)
ax.set_xlabel('Number of Overlapping Objects (Occlusion Level)')
ax.set_ylabel('Tracking Rate')
ax.set_title('Tracking Performance vs. Occlusion Level')
ax.legend()
ax.set_ylim(0.4, 0.9)

# Add count annotations
for l, c in zip(levels, counts):
    ax.annotate(f'n={c}', (l, min(sparse_rates[levels.index(l)], byte_rates[levels.index(l)]) - 0.03),
                ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(f'{images_dir}/occlusion_analysis.png', bbox_inches='tight')
plt.close()
print("Figure 3: occlusion_analysis.png saved")

# ============================================================
# Figure 4: Per-Object Recall Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sparse_recall_vals = list(recall_data['SparseTrack'].values())
byte_recall_vals = list(recall_data['ByteTrack'].values())

axes[0].hist(sparse_recall_vals, bins=20, color='#2196F3', edgecolor='white', alpha=0.8, label='SparseTrack')
axes[0].axvline(x=np.mean(sparse_recall_vals), color='darkblue', linestyle='--', linewidth=2, label=f'Mean={np.mean(sparse_recall_vals):.3f}')
axes[0].set_xlabel('Per-Object Recall')
axes[0].set_ylabel('Number of Objects')
axes[0].set_title('(a) SparseTrack Per-Object Recall')
axes[0].legend()

axes[1].hist(byte_recall_vals, bins=20, color='#FF9800', edgecolor='white', alpha=0.8, label='ByteTrack')
axes[1].axvline(x=np.mean(byte_recall_vals), color='darkorange', linestyle='--', linewidth=2, label=f'Mean={np.mean(byte_recall_vals):.3f}')
axes[1].set_xlabel('Per-Object Recall')
axes[1].set_ylabel('Number of Objects')
axes[1].set_title('(b) ByteTrack Per-Object Recall')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{images_dir}/per_object_recall.png', bbox_inches='tight')
plt.close()
print("Figure 4: per_object_recall.png saved")

# ============================================================
# Figure 5: Trajectory Visualization (sample frames)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sample_frames = [0, 25, 50, 75, 99]
colors = plt.cm.tab20(np.linspace(0, 1, 200))

for idx, frame_idx in enumerate(sample_frames):
    if idx >= 5:
        break
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    frame_data = data[frame_idx]
    gt_bboxes = frame_data['gt_bboxes']
    gt_ids = frame_data['gt_ids']
    dets = frame_data['detections']
    
    # Draw GT boxes
    for bbox, gid in zip(gt_bboxes, gt_ids):
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                             edgecolor=colors[gid % 200], linewidth=0.5, alpha=0.3)
        ax.add_patch(rect)
    
    # Draw detections
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                             edgecolor='red', linewidth=1.0, alpha=0.7)
        ax.add_patch(rect)
    
    ax.set_xlim(-20, 500)
    ax.set_ylim(-20, 600)
    ax.set_title(f'Frame {frame_idx} ({len(dets)} detections)')
    ax.invert_yaxis()
    ax.set_aspect('equal')

# Remove empty subplot
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, f'Dataset: 100 frames\n200 objects\n~79% detection rate\nHigh overlap density',
                ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.suptitle('Sample Frames: Ground Truth (colored) and Detections (red)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{images_dir}/trajectory_samples.png', bbox_inches='tight')
plt.close()
print("Figure 5: trajectory_samples.png saved")

# ============================================================
# Figure 6: Depth Decomposition Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def estimate_pseudo_depth(bbox, img_height=600):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    y_center = (y1 + y2) / 2.0
    area = w * h
    position_factor = y_center / img_height
    return (1.0 / (area + 1e-6)) * (1.0 - position_factor + 0.5)

frame_data = data[0]
dets = frame_data['detections']
depths = [estimate_pseudo_depth(d['bbox']) for d in dets]

# 6a: Depth distribution
axes[0].hist(depths, bins=30, color='purple', edgecolor='white', alpha=0.8)
percentiles = [np.percentile(depths, p) for p in [0, 25, 50, 75, 100]]
for p in percentiles:
    axes[0].axvline(x=p, color='red', linestyle='--', alpha=0.7)
axes[0].set_xlabel('Pseudo-Depth Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('(a) Pseudo-Depth Distribution (Frame 0)')

# 6b: Spatial view colored by depth
depth_array = np.array(depths)
sc = axes[1].scatter([(d['bbox'][0]+d['bbox'][2])/2 for d in dets],
                     [(d['bbox'][1]+d['bbox'][3])/2 for d in dets],
                     c=depth_array, cmap='viridis', s=20, alpha=0.8)
axes[1].set_xlabel('X Center')
axes[1].set_ylabel('Y Center')
axes[1].set_title('(b) Detection Centers Colored by Pseudo-Depth')
axes[1].invert_yaxis()
plt.colorbar(sc, ax=axes[1], label='Pseudo-Depth')

plt.tight_layout()
plt.savefig(f'{images_dir}/depth_decomposition.png', bbox_inches='tight')
plt.close()
print("Figure 6: depth_decomposition.png saved")

print("\nAll figures generated successfully!")
