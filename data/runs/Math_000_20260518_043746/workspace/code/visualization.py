"""
Comprehensive visualization of tracking results and comparisons.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, Counter
import sys
sys.path.insert(0, '.')

with open('data/simulated_sequence.json') as f:
    data = json.load(f)

with open('outputs/experiment_results.json') as f:
    results = json.load(f)

os.makedirs('report/images', exist_ok=True)

# ============================================================
# Figure 4: Main Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

methods = ['SORT', 'ByteTrack', 'SparseTrack']
colors = ['#2196F3', '#FF9800', '#4CAF50']

# Panel 1: MOTA, MOTP, IDF1
metrics_to_plot = ['MOTA', 'MOTP', 'IDF1']
x = np.arange(len(metrics_to_plot))
width = 0.25

for i, method in enumerate(methods):
    vals = [results['metrics'][method][m] for m in metrics_to_plot]
    bars = axes[0].bar(x + i * width, vals, width, label=method, color=colors[i], alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_to_plot, fontsize=12)
axes[0].set_ylabel('Score (%)', fontsize=12)
axes[0].set_title('Tracking Accuracy Metrics', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim(0, 100)
axes[0].grid(True, alpha=0.3, axis='y')

# Panel 2: ID Switches and Tracks
x2 = np.arange(2)
metric_names = ['IDsw', 'Tracks Created']
for i, method in enumerate(methods):
    vals = [results['metrics'][method]['IDsw'], results['n_tracks'][method]]
    axes[1].bar(x2 + i * width, vals, width, label=method, color=colors[i], alpha=0.85, edgecolor='black')

axes[1].set_xticks(x2 + width)
axes[1].set_xticklabels(metric_names, fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('ID Switches & Track Count', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

# Panel 3: MT, ML, FP, FN
x3 = np.arange(4)
metric_names_3 = ['MT', 'ML', 'FP', 'FN']
for i, method in enumerate(methods):
    vals = [results['metrics'][method][m] for m in metric_names_3]
    axes[2].bar(x3 + i * width, vals, width, label=method, color=colors[i], alpha=0.85, edgecolor='black')

axes[2].set_xticks(x3 + width)
axes[2].set_xticklabels(metric_names_3, fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Tracking Statistics', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/fig4_main_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4: Main comparison saved.")

# ============================================================
# Figure 5: Ablation - Depth Layers
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ablation_layers = results['ablation_layers']
n_layers = [a['n_layers'] for a in ablation_layers]
mota_layers = [a['MOTA'] for a in ablation_layers]
idf1_layers = [a['IDF1'] for a in ablation_layers]
idsw_layers = [a['IDsw'] for a in ablation_layers]

axes[0].plot(n_layers, mota_layers, 'o-', color='#4CAF50', linewidth=2, markersize=8, label='SparseTrack')
axes[0].axhline(y=results['metrics']['SORT']['MOTA'], color='#2196F3', linestyle='--', 
                linewidth=2, label=f'SORT ({results["metrics"]["SORT"]["MOTA"]:.1f}%)')
axes[0].set_xlabel('Number of Depth Layers', fontsize=12)
axes[0].set_ylabel('MOTA (%)', fontsize=12)
axes[0].set_title('MOTA vs Depth Layers', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_layers, idf1_layers, 's-', color='#4CAF50', linewidth=2, markersize=8, label='SparseTrack')
axes[1].axhline(y=results['metrics']['SORT']['IDF1'], color='#2196F3', linestyle='--',
                linewidth=2, label=f'SORT ({results["metrics"]["SORT"]["IDF1"]:.1f}%)')
axes[1].set_xlabel('Number of Depth Layers', fontsize=12)
axes[1].set_ylabel('IDF1 (%)', fontsize=12)
axes[1].set_title('IDF1 vs Depth Layers', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(n_layers, idsw_layers, '^-', color='#FF5722', linewidth=2, markersize=8, label='SparseTrack')
axes[2].axhline(y=results['metrics']['SORT']['IDsw'], color='#2196F3', linestyle='--',
                linewidth=2, label=f'SORT ({results["metrics"]["SORT"]["IDsw"]})')
axes[2].set_xlabel('Number of Depth Layers', fontsize=12)
axes[2].set_ylabel('ID Switches', fontsize=12)
axes[2].set_title('ID Switches vs Depth Layers', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_ablation_layers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5: Ablation layers saved.")

# ============================================================
# Figure 6: Ablation - IoU Threshold
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ablation_iou = results['ablation_iou']
iou_vals = [a['iou_thresh'] for a in ablation_iou]

axes[0].plot(iou_vals, [a['bt_MOTA'] for a in ablation_iou], 'o-', color='#FF9800', linewidth=2, markersize=8, label='ByteTrack')
axes[0].plot(iou_vals, [a['st_MOTA'] for a in ablation_iou], 's-', color='#4CAF50', linewidth=2, markersize=8, label='SparseTrack')
axes[0].set_xlabel('IoU Threshold', fontsize=12)
axes[0].set_ylabel('MOTA (%)', fontsize=12)
axes[0].set_title('MOTA vs IoU Threshold', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(iou_vals, [a['bt_IDF1'] for a in ablation_iou], 'o-', color='#FF9800', linewidth=2, markersize=8, label='ByteTrack')
axes[1].plot(iou_vals, [a['st_IDF1'] for a in ablation_iou], 's-', color='#4CAF50', linewidth=2, markersize=8, label='SparseTrack')
axes[1].set_xlabel('IoU Threshold', fontsize=12)
axes[1].set_ylabel('IDF1 (%)', fontsize=12)
axes[1].set_title('IDF1 vs IoU Threshold', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig6_ablation_iou.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6: Ablation IoU saved.")

# ============================================================
# Figure 7: Tracking Qualitative Visualization (selected frames)
# ============================================================
# Load tracking results
with open('outputs/tracking_results.json') as f:
    tracking_results = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Pick 6 representative frames
viz_frames = [0, 20, 40, 60, 80, 99]

for idx, frame_idx in enumerate(viz_frames):
    ax = axes[idx // 3, idx % 3]
    frame_data = data[frame_idx]
    
    # Draw GT boxes in light gray
    for bbox in frame_data['gt_bboxes']:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
    
    # Draw SparseTrack results in green
    st_tracks = tracking_results['sparsetrack'].get(str(frame_idx), {})
    for tid, bbox in st_tracks.items():
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  linewidth=1.5, edgecolor='#4CAF50', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1]-2, f'S{tid[:3]}', fontsize=5, color='#4CAF50', fontweight='bold')
    
    # Draw SORT results in blue (fewer, offset slightly for visibility)
    sort_tracks = tracking_results.get('sort', {}).get(str(frame_idx), {})
    for tid, bbox in list(sort_tracks.items())[:50]:
        rect = patches.Rectangle((bbox[0]+1, bbox[1]+1), bbox[2]-bbox[0]-2, bbox[3]-bbox[1]-2,
                                  linewidth=0.8, edgecolor='#2196F3', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 600)
    ax.invert_yaxis()
    ax.set_title(f'Frame {frame_idx}\n({len(frame_data["gt_bboxes"])} GT, {len(frame_data["detections"])} det, '
                 f'{len(st_tracks)} tracked)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

plt.suptitle('SparseTrack (green) vs SORT (blue) - Selected Frames\nGray: Ground Truth', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig7_tracking_qualitative.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7: Qualitative tracking saved.")

# ============================================================
# Figure 8: Per-frame tracking density analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Count per-frame tracking results
frame_counts = defaultdict(lambda: {'sort': 0, 'sparsetrack': 0, 'bytetrack': 0})
for frame_idx in range(100):
    for method_key in ['sort', 'sparsetrack', 'bytetrack']:
        if method_key in tracking_results:
            frame_counts[frame_idx][method_key] = len(tracking_results[method_key].get(str(frame_idx), {}))

frame_ids = list(range(100))
sort_counts = [frame_counts[f]['sort'] for f in frame_ids]
st_counts = [frame_counts[f]['sparsetrack'] for f in frame_ids]
bt_counts = [frame_counts[f]['bytetrack'] for f in frame_ids]
gt_counts = [len(data[f]['gt_bboxes']) for f in frame_ids]
det_counts = [len(data[f]['detections']) for f in frame_ids]

# Panel 1: Tracking count per frame
ax = axes[0, 0]
ax.plot(frame_ids, gt_counts, 'k-', linewidth=2, label='Ground Truth', alpha=0.5)
ax.plot(frame_ids, det_counts, 'r-', linewidth=1.5, label='Detections', alpha=0.7)
ax.plot(frame_ids, st_counts, '-', color='#4CAF50', linewidth=2, label='SparseTrack')
ax.plot(frame_ids, sort_counts, '-', color='#2196F3', linewidth=1.5, label='SORT', alpha=0.8)
ax.plot(frame_ids, bt_counts, '-', color='#FF9800', linewidth=1.5, label='ByteTrack', alpha=0.8)
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Objects Tracked per Frame', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Score distribution of detections
ax = axes[0, 1]
all_scores = []
for d in data:
    for det in d['detections']:
        all_scores.append(det['score'])
ax.hist(all_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue', density=True)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='ByteTrack threshold=0.5')
ax.set_xlabel('Detection Score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Detection Score Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: Bbox size distribution (pseudo-depth proxy)
ax = axes[1, 0]
all_areas = []
all_pseudo_depths = []
for d in data:
    for det in d['detections']:
        bbox = det['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        all_areas.append(area)
        all_pseudo_depths.append(1.0 / np.sqrt(area))

ax.hist(all_pseudo_depths, bins=50, edgecolor='black', alpha=0.7, color='teal')
ax.set_xlabel('Pseudo-Depth (1/√area)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Pseudo-Depth Distribution\n(Hierarchical Decomposition Axis)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add layer boundaries for 5 layers
min_pd = min(all_pseudo_depths)
max_pd = max(all_pseudo_depths)
for i in range(1, 5):
    boundary = min_pd + i * (max_pd - min_pd) / 5
    ax.axvline(x=boundary, color='red', linestyle=':', alpha=0.5)
ax.text(max_pd * 0.85, ax.get_ylim()[1] * 0.8, '5 Depth\nLayers', fontsize=10, 
        color='red', ha='center', fontweight='bold')

# Panel 4: Track lifetime distribution
ax = axes[1, 1]
track_lifetimes = defaultdict(int)
for method_key in ['sort', 'sparsetrack', 'bytetrack']:
    if method_key not in tracking_results:
        continue
    # Count frames each track ID appears
    track_frames = defaultdict(int)
    for frame_str, tracks in tracking_results[method_key].items():
        for tid in tracks:
            track_frames[tid] += 1
    lifetimes = list(track_frames.values())
    if lifetimes:
        label = {'sort': 'SORT', 'sparsetrack': 'SparseTrack', 'bytetrack': 'ByteTrack'}[method_key]
        color = {'sort': '#2196F3', 'sparsetrack': '#4CAF50', 'bytetrack': '#FF9800'}[method_key]
        ax.hist(lifetimes, bins=range(1, max(lifetimes)+2), alpha=0.5, label=label, 
                color=color, edgecolor='black')

ax.set_xlabel('Track Lifetime (frames)', fontsize=11)
ax.set_ylabel('Number of Tracks', fontsize=11)
ax.set_title('Track Lifetime Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig8_density_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8: Density analysis saved.")

# ============================================================
# Figure 9: Methodology Diagram (conceptual)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(7, 5.7, 'SparseTrack Pipeline: Pseudo-Depth Hierarchical Association', 
        fontsize=16, ha='center', fontweight='bold', color='#333')

# Boxes for pipeline steps
box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
box_green = dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', edgecolor='#4CAF50', linewidth=2)
box_orange = dict(boxstyle='round,pad=0.5', facecolor='#FFE0B2', edgecolor='#FF9800', linewidth=2)
box_purple = dict(boxstyle='round,pad=0.5', facecolor='#E1BEE7', edgecolor='#9C27B0', linewidth=2)

# Step 1: Input
ax.text(1, 4.5, 'Detections', fontsize=11, ha='center', fontweight='bold', bbox=box_props)
ax.annotate('', xy=(2.5, 4.5), xytext=(1.8, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Step 2: Score Split
ax.text(3.5, 4.5, 'Score Split\n(High/Low)', fontsize=10, ha='center', fontweight='bold', bbox=box_orange)
ax.annotate('', xy=(5, 4.5), xytext=(4.3, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Step 3: High-score association
ax.text(6, 4.5, 'High-Score\nAssociation', fontsize=10, ha='center', fontweight='bold', bbox=box_green)
ax.annotate('', xy=(7.5, 4.5), xytext=(6.8, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Step 4: Pseudo-depth estimation
ax.text(8.5, 3.5, 'Pseudo-Depth\nEstimation\n(1/√area)', fontsize=10, ha='center', fontweight='bold', 
        bbox=box_purple)

# Step 5: Hierarchical decomposition
ax.text(11, 3.5, 'Depth Layer\nDecomposition\n(5 layers)', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFCDD2', edgecolor='#F44336', linewidth=2))

# Step 6: Layer-wise association
ax.text(13, 3.5, 'Layer-wise\nAssociation\n(front→back)', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#B2EBF2', edgecolor='#00BCD4', linewidth=2))

# Connect steps
ax.annotate('', xy=(8.5, 4.2), xytext=(7.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.annotate('', xy=(11, 3.7), xytext=(9.3, 3.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.annotate('', xy=(12.2, 3.5), xytext=(11.8, 3.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Output
ax.text(13, 1.5, 'Complete\nTrajectories', fontsize=11, ha='center', fontweight='bold', bbox=box_green)
ax.annotate('', xy=(13, 2.2), xytext=(13, 3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Annotation
ax.text(7, 1.2, 'Key Innovation: Decompose dense detections into sparse depth layers\n'
        'to reduce association ambiguity in crowded scenes', 
        fontsize=10, ha='center', style='italic', color='#666',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('report/images/fig9_methodology_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9: Methodology diagram saved.")

# ============================================================
# Figure 10: Per-frame MOTA breakdown
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Compute per-frame matching
frame_mota_sort = []
frame_mota_st = []

for frame_idx in range(100):
    frame_data = data[frame_idx]
    gt_bboxes = frame_data['gt_bboxes']
    n_gt = len(gt_bboxes)
    
    for method_key, storage in [('sort', frame_mota_sort), ('sparsetrack', frame_mota_st)]:
        tracked = tracking_results[method_key].get(str(frame_idx), {})
        n_tracked = len(tracked)
        
        # Simple per-frame accuracy
        if n_gt > 0:
            tp = min(n_tracked, n_gt)
            fn = max(0, n_gt - n_tracked)
            fp = max(0, n_tracked - n_gt)
            per_frame_mota = (tp - fp - fn) / n_gt * 100
        else:
            per_frame_mota = 0
        storage.append(per_frame_mota)

ax.plot(frame_ids, frame_mota_sort, '-', color='#2196F3', linewidth=2, label='SORT', alpha=0.8)
ax.plot(frame_ids, frame_mota_st, '-', color='#4CAF50', linewidth=2, label='SparseTrack', alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Per-Frame Accuracy (%)', fontsize=12)
ax.set_title('Per-Frame Tracking Accuracy: SORT vs SparseTrack', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig10_per_frame_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10: Per-frame accuracy saved.")

print("\nAll visualizations complete!")
