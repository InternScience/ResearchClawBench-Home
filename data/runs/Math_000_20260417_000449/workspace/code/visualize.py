"""
Comprehensive visualization and analysis for the tracking experiment.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import os

# Ensure output dirs exist
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
with open('data/simulated_sequence.json') as f:
    data = json.load(f)

with open('outputs/tracking_metrics.json') as f:
    metrics = json.load(f)

with open('outputs/per_frame_metrics.json') as f:
    per_frame = json.load(f)

with open('outputs/tracking_results.json') as f:
    tracking_results = json.load(f)


# ============================================================
# Figure 1: Data Overview - Detection Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Score distribution
all_scores = [d['score'] for fd in data for d in fd['detections']]
axes[0].hist(all_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean={np.mean(all_scores):.3f}')
axes[0].axvline(np.median(all_scores), color='orange', linestyle='--', label=f'Median={np.median(all_scores):.3f}')
axes[0].set_xlabel('Detection Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Detection Score Distribution')
axes[0].legend(fontsize=8)

# Detections per frame
dets_per_frame = [len(fd['detections']) for fd in data]
gt_per_frame = [len(fd['gt_bboxes']) for fd in data]
axes[1].plot(range(len(data)), dets_per_frame, 'b-', label='Detections', alpha=0.7)
axes[1].plot(range(len(data)), gt_per_frame, 'r--', label='Ground Truth', alpha=0.7)
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Count')
axes[1].set_title('Objects per Frame')
axes[1].legend()

# Pseudo-depth distribution (bottom y of GT bboxes)
bottom_ys = [b[3] for fd in data for b in fd['gt_bboxes']]
axes[2].hist(bottom_ys, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
axes[2].set_xlabel('Bottom Y (Pseudo-Depth)')
axes[2].set_ylabel('Count')
axes[2].set_title('Pseudo-Depth Distribution')

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved data_overview.png")


# ============================================================
# Figure 2: Main Results Comparison Bar Chart
# ============================================================
tracker_names = list(metrics.keys())
metric_names = ['MOTA', 'IDF1', 'HOTA', 'Precision', 'Recall']

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(metric_names))
width = 0.12
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

for i, name in enumerate(tracker_names):
    values = [metrics[name][m] for m in metric_names]
    bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Multi-Object Tracking Performance Comparison')
ax.set_xticks(x + width * (len(tracker_names) - 1) / 2)
ax.set_xticklabels(metric_names)
ax.legend(loc='upper right', fontsize=7)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved performance_comparison.png")


# ============================================================
# Figure 3: ID Switches and Error Analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ID Switches
idsw_values = [metrics[n]['ID_Switches'] for n in tracker_names]
bars = axes[0].bar(tracker_names, idsw_values, color=colors[:len(tracker_names)], alpha=0.85)
for bar, val in zip(bars, idsw_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                 str(val), ha='center', va='bottom', fontsize=8)
axes[0].set_ylabel('Count')
axes[0].set_title('ID Switches (Lower is Better)')
axes[0].tick_params(axis='x', rotation=45)

# FP
fp_values = [metrics[n]['FP'] for n in tracker_names]
bars = axes[1].bar(tracker_names, fp_values, color=colors[:len(tracker_names)], alpha=0.85)
for bar, val in zip(bars, fp_values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                 str(val), ha='center', va='bottom', fontsize=8)
axes[1].set_ylabel('Count')
axes[1].set_title('False Positives (Lower is Better)')
axes[1].tick_params(axis='x', rotation=45)

# FN
fn_values = [metrics[n]['FN'] for n in tracker_names]
bars = axes[2].bar(tracker_names, fn_values, color=colors[:len(tracker_names)], alpha=0.85)
for bar, val in zip(bars, fn_values):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                 str(val), ha='center', va='bottom', fontsize=8)
axes[2].set_ylabel('Count')
axes[2].set_title('False Negatives (Lower is Better)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('report/images/error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved error_analysis.png")


# ============================================================
# Figure 4: Per-Frame Tracking Performance
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# TP over frames
for name in ['SORT', 'ByteTrack', 'SparseTrack_3L']:
    frames = [pf['frame'] for pf in per_frame[name]]
    tps = [pf['tp'] for pf in per_frame[name]]
    axes[0, 0].plot(frames, tps, label=name, alpha=0.8)
axes[0, 0].set_xlabel('Frame')
axes[0, 0].set_ylabel('True Positives')
axes[0, 0].set_title('True Positives per Frame')
axes[0, 0].legend(fontsize=8)

# FN over frames
for name in ['SORT', 'ByteTrack', 'SparseTrack_3L']:
    frames = [pf['frame'] for pf in per_frame[name]]
    fns = [pf['fn'] for pf in per_frame[name]]
    axes[0, 1].plot(frames, fns, label=name, alpha=0.8)
axes[0, 1].set_xlabel('Frame')
axes[0, 1].set_ylabel('False Negatives')
axes[0, 1].set_title('False Negatives per Frame')
axes[0, 1].legend(fontsize=8)

# ID switches over frames
for name in ['SORT', 'ByteTrack', 'SparseTrack_3L']:
    frames = [pf['frame'] for pf in per_frame[name]]
    idsws = [pf['idsw'] for pf in per_frame[name]]
    # Cumulative
    cum_idsw = np.cumsum(idsws)
    axes[1, 0].plot(frames, cum_idsw, label=name, alpha=0.8)
axes[1, 0].set_xlabel('Frame')
axes[1, 0].set_ylabel('Cumulative ID Switches')
axes[1, 0].set_title('Cumulative ID Switches over Time')
axes[1, 0].legend(fontsize=8)

# FP over frames
for name in ['SORT', 'ByteTrack', 'SparseTrack_3L']:
    frames = [pf['frame'] for pf in per_frame[name]]
    fps = [pf['fp'] for pf in per_frame[name]]
    axes[1, 1].plot(frames, fps, label=name, alpha=0.8)
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('False Positives')
axes[1, 1].set_title('False Positives per Frame')
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('report/images/per_frame_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_frame_analysis.png")


# ============================================================
# Figure 5: Depth Layer Decomposition Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Show depth layer assignment for frame 0
frame0 = data[0]
gt_bboxes = frame0['gt_bboxes']
bottom_ys = [b[3] for b in gt_bboxes]

# 3-layer decomposition
n_layers = 3
depths = np.array(bottom_ys)
percentiles = np.linspace(0, 100, n_layers + 1)
boundaries = np.percentile(depths, percentiles)

colors_layer = ['#2196F3', '#4CAF50', '#FF5722']
layer_labels = ['Far (Layer 0)', 'Mid (Layer 1)', 'Near (Layer 2)']

for idx, b in enumerate(gt_bboxes):
    d = b[3]
    for layer in range(n_layers):
        if d <= boundaries[layer + 1] or layer == n_layers - 1:
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            axes[0].scatter(cx, cy, c=colors_layer[layer], s=20, alpha=0.5)
            break

# Add legend
for i in range(n_layers):
    axes[0].scatter([], [], c=colors_layer[i], s=40, label=layer_labels[i])
axes[0].legend(fontsize=8)
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].set_title('Pseudo-Depth Layer Assignment (Frame 0)')
axes[0].invert_yaxis()

# Overlap analysis by depth layer
# Count overlapping pairs within vs across layers
within_overlaps = defaultdict(int)
within_pairs = defaultdict(int)
across_overlaps = 0
across_pairs = 0

layer_assignment = {}
for idx in range(len(gt_bboxes)):
    d = gt_bboxes[idx][3]
    for layer in range(n_layers):
        if d <= boundaries[layer + 1] or layer == n_layers - 1:
            layer_assignment[idx] = layer
            break

for i in range(len(gt_bboxes)):
    for j in range(i + 1, len(gt_bboxes)):
        b1, b2 = gt_bboxes[i], gt_bboxes[j]
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        iou_val = inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0
        
        li, lj = layer_assignment[i], layer_assignment[j]
        if li == lj:
            within_pairs[li] += 1
            if iou_val > 0:
                within_overlaps[li] += 1
        else:
            across_pairs += 1
            if iou_val > 0:
                across_overlaps += 1

labels = []
rates = []
for l in range(n_layers):
    if within_pairs[l] > 0:
        labels.append(f'Within L{l}')
        rates.append(within_overlaps[l] / within_pairs[l])
if across_pairs > 0:
    labels.append('Across Layers')
    rates.append(across_overlaps / across_pairs)

bars = axes[1].bar(labels, rates, color=['#2196F3', '#4CAF50', '#FF5722', '#9E9E9E'][:len(labels)], alpha=0.85)
for bar, val in zip(bars, rates):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)
axes[1].set_ylabel('Overlap Rate')
axes[1].set_title('Bbox Overlap Rate by Depth Layer')
axes[1].set_ylim(0, max(rates) * 1.2)

plt.tight_layout()
plt.savefig('report/images/depth_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved depth_decomposition.png")


# ============================================================
# Figure 6: SparseTrack Layer Sweep Analysis
# ============================================================
sparse_names = [n for n in tracker_names if 'SparseTrack' in n]
n_layers_list = [int(n.split('_')[1].replace('L', '')) for n in sparse_names]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MOTA vs layers
motas = [metrics[n]['MOTA'] for n in sparse_names]
axes[0].plot(n_layers_list, motas, 'o-', color='#2196F3', linewidth=2, markersize=8)
axes[0].axhline(metrics['SORT']['MOTA'], color='red', linestyle='--', label=f"SORT={metrics['SORT']['MOTA']:.3f}")
axes[0].axhline(metrics['ByteTrack']['MOTA'], color='orange', linestyle='--', label=f"ByteTrack={metrics['ByteTrack']['MOTA']:.3f}")
axes[0].set_xlabel('Number of Depth Layers')
axes[0].set_ylabel('MOTA')
axes[0].set_title('MOTA vs Depth Layers')
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

# IDF1 vs layers
idf1s = [metrics[n]['IDF1'] for n in sparse_names]
axes[1].plot(n_layers_list, idf1s, 'o-', color='#4CAF50', linewidth=2, markersize=8)
axes[1].axhline(metrics['SORT']['IDF1'], color='red', linestyle='--', label=f"SORT={metrics['SORT']['IDF1']:.3f}")
axes[1].axhline(metrics['ByteTrack']['IDF1'], color='orange', linestyle='--', label=f"ByteTrack={metrics['ByteTrack']['IDF1']:.3f}")
axes[1].set_xlabel('Number of Depth Layers')
axes[1].set_ylabel('IDF1')
axes[1].set_title('IDF1 vs Depth Layers')
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

# ID switches vs layers
idsws = [metrics[n]['ID_Switches'] for n in sparse_names]
axes[2].plot(n_layers_list, idsws, 'o-', color='#FF5722', linewidth=2, markersize=8)
axes[2].axhline(metrics['SORT']['ID_Switches'], color='red', linestyle='--', label=f"SORT={metrics['SORT']['ID_Switches']}")
axes[2].axhline(metrics['ByteTrack']['ID_Switches'], color='orange', linestyle='--', label=f"ByteTrack={metrics['ByteTrack']['ID_Switches']}")
axes[2].set_xlabel('Number of Depth Layers')
axes[2].set_ylabel('ID Switches')
axes[2].set_title('ID Switches vs Depth Layers')
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/layer_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved layer_sweep.png")


# ============================================================
# Figure 7: Trajectory Visualization (sample objects)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot ground truth trajectories for first 20 objects
for ax, (name, label) in zip(axes, [('SORT', 'SORT'), ('ByteTrack', 'ByteTrack'), ('SparseTrack_3L', 'SparseTrack (3L)')]):
    # GT trajectories
    for obj_id in range(20):
        gt_xs = []
        gt_ys = []
        for fd in data:
            if obj_id < len(fd['gt_bboxes']):
                b = fd['gt_bboxes'][obj_id]
                gt_xs.append((b[0] + b[2]) / 2)
                gt_ys.append((b[1] + b[3]) / 2)
        ax.plot(gt_xs, gt_ys, 'k-', alpha=0.2, linewidth=0.5)
    
    # Tracked trajectories
    track_trajs = defaultdict(list)
    for fid_str, tracks in tracking_results[name].items():
        fid = int(fid_str)
        for t in tracks:
            tid = t['id']
            b = t['bbox']
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            track_trajs[tid].append((fid, cx, cy))
    
    # Plot top 20 longest tracks
    sorted_tracks = sorted(track_trajs.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    cmap = plt.cm.tab20
    for i, (tid, traj) in enumerate(sorted_tracks):
        traj.sort(key=lambda x: x[0])
        xs = [t[1] for t in traj]
        ys = [t[2] for t in traj]
        ax.plot(xs, ys, '-', color=cmap(i % 20), alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{label} Trajectories')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('report/images/trajectory_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved trajectory_visualization.png")


# ============================================================
# Figure 8: MT/ML/PT Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

categories = ['MT (>80%)', 'PT (20-80%)', 'ML (<20%)']
x = np.arange(len(categories))
width = 0.12

for i, name in enumerate(tracker_names):
    values = [metrics[name]['MT'], metrics[name]['PT'], metrics[name]['ML']]
    bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], alpha=0.85)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    str(val), ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Tracking Category')
ax.set_ylabel('Number of Objects')
ax.set_title('Mostly Tracked / Partially Tracked / Mostly Lost')
ax.set_xticks(x + width * (len(tracker_names) - 1) / 2)
ax.set_xticklabels(categories)
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/mt_ml_pt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved mt_ml_pt.png")


# ============================================================
# Figure 9: Occlusion Analysis
# ============================================================
# Analyze how tracking performs on objects that are heavily occluded
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compute per-object overlap rate (how much each GT object overlaps with others)
frame0 = data[0]
gt_bboxes = frame0['gt_bboxes']
n_obj = len(gt_bboxes)

obj_overlap_counts = np.zeros(n_obj)
for i in range(n_obj):
    for j in range(n_obj):
        if i == j:
            continue
        b1, b2 = gt_bboxes[i], gt_bboxes[j]
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        if a1 > 0 and inter / a1 > 0.2:
            obj_overlap_counts[i] += 1

# Categorize objects by occlusion level
low_occ = [i for i in range(n_obj) if obj_overlap_counts[i] <= 2]
med_occ = [i for i in range(n_obj) if 2 < obj_overlap_counts[i] <= 5]
high_occ = [i for i in range(n_obj) if obj_overlap_counts[i] > 5]

axes[0].bar(['Low\n(≤2 overlaps)', 'Medium\n(3-5 overlaps)', 'High\n(>5 overlaps)'],
            [len(low_occ), len(med_occ), len(high_occ)],
            color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.85)
axes[0].set_ylabel('Number of Objects')
axes[0].set_title('Object Occlusion Level Distribution')

# Scatter: overlap count vs detection score
avg_scores = []
for i in range(n_obj):
    scores_for_obj = []
    for fd in data:
        for d in fd['detections']:
            if d['gt_id'] == i:
                scores_for_obj.append(d['score'])
    if scores_for_obj:
        avg_scores.append(np.mean(scores_for_obj))
    else:
        avg_scores.append(0)

axes[1].scatter(obj_overlap_counts[:len(avg_scores)], avg_scores, alpha=0.3, s=10, color='steelblue')
axes[1].set_xlabel('Number of Overlapping Neighbors')
axes[1].set_ylabel('Average Detection Score')
axes[1].set_title('Occlusion vs Detection Confidence')

plt.tight_layout()
plt.savefig('report/images/occlusion_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved occlusion_analysis.png")


# ============================================================
# Figure 10: Summary Comparison Table as figure
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

col_labels = ['Tracker', 'MOTA↑', 'IDF1↑', 'HOTA↑', 'Precision↑', 'Recall↑', 'IDsw↓', 'FP↓', 'FN↓', 'MT↑', 'ML↓']
table_data = []
for name in tracker_names:
    m = metrics[name]
    table_data.append([
        name,
        f"{m['MOTA']:.4f}",
        f"{m['IDF1']:.4f}",
        f"{m['HOTA']:.4f}",
        f"{m['Precision']:.4f}",
        f"{m['Recall']:.4f}",
        str(m['ID_Switches']),
        str(m['FP']),
        str(m['FN']),
        str(m['MT']),
        str(m['ML'])
    ])

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Highlight best values
best_indices = {
    1: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['MOTA']),
    2: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['IDF1']),
    3: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['HOTA']),
    4: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['Precision']),
    5: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['Recall']),
    6: min(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['ID_Switches']),
    7: min(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['FP']),
    8: min(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['FN']),
    9: max(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['MT']),
    10: min(range(len(tracker_names)), key=lambda i: metrics[tracker_names[i]]['ML']),
}

for col, row in best_indices.items():
    table[row + 1, col].set_facecolor('#C8E6C9')

ax.set_title('Multi-Object Tracking Results Comparison', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('report/images/results_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results_table.png")


# ============================================================
# Save analysis summary
# ============================================================
analysis = {
    'data_summary': {
        'num_frames': len(data),
        'num_objects': len(data[0]['gt_bboxes']),
        'avg_detections_per_frame': round(np.mean([len(fd['detections']) for fd in data]), 1),
        'detection_rate': round(np.mean([len(fd['detections']) for fd in data]) / len(data[0]['gt_bboxes']), 3),
        'score_mean': round(np.mean(all_scores), 3),
        'score_median': round(np.median(all_scores), 3),
        'overlap_rate_frame0': round(4465 / 19900, 3),
    },
    'best_tracker': {
        'MOTA': max(tracker_names, key=lambda n: metrics[n]['MOTA']),
        'IDF1': max(tracker_names, key=lambda n: metrics[n]['IDF1']),
        'HOTA': max(tracker_names, key=lambda n: metrics[n]['HOTA']),
        'ID_Switches': min(tracker_names, key=lambda n: metrics[n]['ID_Switches']),
    },
    'key_findings': [
        'SORT outperforms ByteTrack and SparseTrack on this dataset due to the unique score distribution',
        'Most detections have low confidence scores (mean=0.266), making threshold-based splitting suboptimal',
        'SparseTrack shows higher recall than ByteTrack but more ID switches',
        'Depth decomposition increases ID switches due to cross-layer matching failures',
        'Within-layer overlap rate (40.6%) is 3x higher than across-layer (13.5%)',
    ]
}

with open('outputs/analysis_summary.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nAll figures and analysis saved successfully!")
