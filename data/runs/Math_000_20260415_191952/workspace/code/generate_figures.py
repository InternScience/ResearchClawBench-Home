"""
Generate figures for the MOT research report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Load data
with open('data/simulated_sequence.json') as f:
    data = json.load(f)

with open('outputs/metrics.json') as f:
    metrics = json.load(f)

with open('outputs/per_frame_analysis.json') as f:
    per_frame = json.load(f)

with open('outputs/trajectory_lengths.json') as f:
    traj_lengths = json.load(f)

with open('outputs/bytetrack_trajectories.json') as f:
    bt_traj = json.load(f)

with open('outputs/sparsetrack_trajectories.json') as f:
    st_traj = json.load(f)

with open('outputs/oracle_trajectories.json') as f:
    oracle_traj = json.load(f)

os.makedirs('report/images', exist_ok=True)

# ============================================================
# Figure 1: Data Overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a: Detection scores distribution
all_scores = []
for frame in data:
    for det in frame['detections']:
        all_scores.append(det['score'])

axes[0, 0].hist(all_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 0].axvline(x=0.6, color='red', linestyle='--', label='High-score threshold (0.6)')
axes[0, 0].set_xlabel('Detection Score', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].set_title('(a) Detection Score Distribution', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 1b: Detection rate per frame
det_rates = []
for frame in data:
    n_gt = len(frame['gt_ids'])
    n_det = len(frame['detections'])
    det_rates.append(n_det / n_gt * 100)

axes[0, 1].plot(range(len(det_rates)), det_rates, 'o-', color='coral', markersize=3, linewidth=1.5)
axes[0, 1].axhline(y=np.mean(det_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(det_rates):.1f}%')
axes[0, 1].set_xlabel('Frame Number', fontsize=12)
axes[0, 1].set_ylabel('Detection Rate (%)', fontsize=12)
axes[0, 1].set_title('(b) Per-Frame Detection Rate', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(70, 90)

# 1c: Bounding box area distribution
areas = []
for frame in data:
    for det in frame['detections']:
        x1, y1, x2, y2 = det['bbox']
        areas.append((x2-x1)*(y2-y1))

axes[1, 0].hist(areas, bins=50, color='mediumseagreen', edgecolor='white', alpha=0.8)
axes[1, 0].set_xlabel('Bounding Box Area (pixels²)', fontsize=12)
axes[1, 0].set_ylabel('Count', fontsize=12)
axes[1, 0].set_title('(c) Bounding Box Area Distribution', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 1d: Score vs Area scatter (sample)
idx = np.random.choice(len(areas), min(2000, len(areas)), replace=False)
scatter_areas = [areas[i] for i in idx]
scatter_scores = [all_scores[i] for i in idx]

axes[1, 1].scatter(scatter_areas, scatter_scores, alpha=0.3, s=5, c='purple')
axes[1, 1].set_xlabel('Bounding Box Area', fontsize=12)
axes[1, 1].set_ylabel('Detection Score', fontsize=12)
axes[1, 1].set_title('(d) Score vs. BBox Area', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

corr = np.corrcoef(areas, all_scores)[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure1_data_overview.png")

# ============================================================
# Figure 2: Main Results Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

methods = ['Oracle\n(Upper Bound)', 'ByteTrack\n(Baseline)', 'SparseTrack\n(Ours)']
mota_vals = [metrics['Oracle']['MOTA'], metrics['ByteTrack']['MOTA'], metrics['SparseTrack']['MOTA']]
idf1_vals = [metrics['Oracle']['IDF1'], metrics['ByteTrack']['IDF1'], metrics['SparseTrack']['IDF1']]
hota_vals = [metrics['Oracle']['HOTA'], metrics['ByteTrack']['HOTA'], metrics['SparseTrack']['HOTA']]

colors = ['#2ecc71', '#3498db', '#e74c3c']

# MOTA
bars = axes[0].bar(methods, mota_vals, color=colors, edgecolor='white', width=0.6)
axes[0].set_ylabel('MOTA', fontsize=12)
axes[0].set_title('MOTA Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mota_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# IDF1
bars = axes[1].bar(methods, idf1_vals, color=colors, edgecolor='white', width=0.6)
axes[1].set_ylabel('IDF1', fontsize=12)
axes[1].set_title('IDF1 Comparison', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, idf1_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# HOTA
bars = axes[2].bar(methods, hota_vals, color=colors, edgecolor='white', width=0.6)
axes[2].set_ylabel('HOTA', fontsize=12)
axes[2].set_title('HOTA Comparison', fontsize=13, fontweight='bold')
axes[2].set_ylim(0, 1.1)
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, hota_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure2_main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure2_main_results.png")

# ============================================================
# Figure 3: Per-Frame Tracking Performance
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bt_pf = per_frame['ByteTrack']
st_pf = per_frame['SparseTrack']

frames = [p['frame'] for p in bt_pf]
bt_rates = [p['tracking_rate'] * 100 for p in bt_pf]
st_rates = [p['tracking_rate'] * 100 for p in st_pf]
det_rates_plot = [p['detection_rate'] * 100 for p in bt_pf]

axes[0].plot(frames, det_rates_plot, 'k--', linewidth=1.5, label='Detection Rate', alpha=0.7)
axes[0].plot(frames, bt_rates, 'b-o', markersize=3, linewidth=1.5, label='ByteTrack')
axes[0].plot(frames, st_rates, 'r-s', markersize=3, linewidth=1.5, label='SparseTrack')
axes[0].set_xlabel('Frame Number', fontsize=12)
axes[0].set_ylabel('Rate (%)', fontsize=12)
axes[0].set_title('(a) Per-Frame Tracking Rate', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Cumulative tracked objects over time
bt_cumsum = np.cumsum([p['n_tracked'] for p in bt_pf])
st_cumsum = np.cumsum([p['n_tracked'] for p in st_pf])

axes[1].plot(frames, bt_cumsum, 'b-o', markersize=3, linewidth=1.5, label='ByteTrack')
axes[1].plot(frames, st_cumsum, 'r-s', markersize=3, linewidth=1.5, label='SparseTrack')
axes[1].set_xlabel('Frame Number', fontsize=12)
axes[1].set_ylabel('Cumulative Tracked Detections', fontsize=12)
axes[1].set_title('(b) Cumulative Tracking Performance', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure3_per_frame.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3_per_frame.png")

# ============================================================
# Figure 4: Trajectory Length Distribution & ID Analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4a: Trajectory length histogram
bt_lens = [len(t['trajectory']) for t in bt_traj.values()]
st_lens = [len(t['trajectory']) for t in st_traj.values()]
ol_lens = [len(t['trajectory']) for t in oracle_traj.values()]

axes[0].hist(ol_lens, bins=30, alpha=0.5, label='Oracle', color='#2ecc71')
axes[0].hist(bt_lens, bins=30, alpha=0.5, label='ByteTrack', color='#3498db')
axes[0].hist(st_lens, bins=30, alpha=0.5, label='SparseTrack', color='#e74c3c')
axes[0].set_xlabel('Trajectory Length (frames)', fontsize=12)
axes[0].set_ylabel('Number of Tracks', fontsize=12)
axes[0].set_title('(a) Trajectory Length Distribution', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 4b: Metric comparison table as bar chart
metric_names = ['MOTA', 'IDF1', 'HOTA']
x = np.arange(len(metric_names))
width = 0.25

bt_m = [metrics['ByteTrack']['MOTA'], metrics['ByteTrack']['IDF1'], metrics['ByteTrack']['HOTA']]
st_m = [metrics['SparseTrack']['MOTA'], metrics['SparseTrack']['IDF1'], metrics['SparseTrack']['HOTA']]
or_m = [metrics['Oracle']['MOTA'], metrics['Oracle']['IDF1'], metrics['Oracle']['HOTA']]

axes[1].bar(x - width, or_m, width, label='Oracle', color='#2ecc71')
axes[1].bar(x, bt_m, width, label='ByteTrack', color='#3498db')
axes[1].bar(x + width, st_m, width, label='SparseTrack', color='#e74c3c')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_names, fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('(b) Overall Metrics', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# 4c: ID Switches and Fragments
categories = ['ID Switches', 'Fragments']
bt_err = [metrics['ByteTrack']['ID_Switches'], metrics['ByteTrack']['Fragments']]
st_err = [metrics['SparseTrack']['ID_Switches'], metrics['SparseTrack']['Fragments']]

x = np.arange(len(categories))
width = 0.35
axes[2].bar(x - width/2, bt_err, width, label='ByteTrack', color='#3498db')
axes[2].bar(x + width/2, st_err, width, label='SparseTrack', color='#e74c3c')
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories, fontsize=11)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('(c) Error Analysis', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/figure4_trajectory_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure4_trajectory_analysis.png")

# ============================================================
# Figure 5: Pseudo-Depth Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a: Depth layer distribution
st_depths = [t.get('depth', 0.5) for t in st_traj.values()]
st_layers = [t.get('layer', 1) for t in st_traj.values()]

layer_counts = [st_layers.count(i) for i in range(4)]
layer_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
layer_labels = ['Layer 0\n(Closest)', 'Layer 1', 'Layer 2', 'Layer 3\n(Farthest)']

axes[0].bar(range(4), layer_counts, color=layer_colors, edgecolor='white', width=0.6)
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(layer_labels, fontsize=10)
axes[0].set_ylabel('Number of Tracks', fontsize=12)
axes[0].set_title('(a) Pseudo-Depth Layer Distribution', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, count in enumerate(layer_counts):
    axes[0].text(i, count + max(layer_counts)*0.02, str(count), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5b: Depth vs trajectory length
axes[1].scatter(st_depths, [len(t['trajectory']) for t in st_traj.values()], 
               alpha=0.3, s=10, c=st_layers, cmap='viridis')
axes[1].set_xlabel('Pseudo-Depth (1.0 = farthest)', fontsize=12)
axes[1].set_ylabel('Trajectory Length (frames)', fontsize=12)
axes[1].set_title('(b) Depth vs. Track Longevity', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=3))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[1], ticks=[0, 1, 2, 3])
cbar.set_label('Depth Layer', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/figure5_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5_depth_analysis.png")

# ============================================================
# Figure 6: Method Architecture Diagram
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(7, 5.7, 'SparseTrack: Pseudo-Depth Hierarchical Association Pipeline', 
        fontsize=16, fontweight='bold', ha='center')

# Boxes
boxes = [
    (0.5, 3.5, 2.5, 1.5, 'Input\nDetections\n(bbox, score)'),
    (3.5, 3.5, 2.5, 1.5, 'Pseudo-Depth\nEstimation\n(area-based)'),
    (6.5, 4.2, 2.5, 1.5, 'Layer 0\n(Closest)\nAssociation'),
    (6.5, 2.5, 2.5, 1.5, 'Layer 1\nAssociation'),
    (6.5, 0.8, 2.5, 1.5, 'Layer 2-3\n(Farther)\nAssociation'),
    (9.5, 3.5, 2.5, 1.5, 'Cross-Layer\nMatching\n(residuals)'),
    (12, 3.5, 1.5, 1.5, 'Output\nTrajectories'),
]

box_colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71', '#1abc9c', '#2c3e50']

for i, (x, y, w, h, text) in enumerate(boxes):
    rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=box_colors[i], 
                         edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, 
            color='white', fontweight='bold')

# Arrows
arrow_props = dict(arrowstyle='->', color='gray', lw=2)
ax.annotate('', xy=(3.5, 4.25), xytext=(3.0, 4.25), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 4.95), xytext=(6.0, 4.95), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 3.25), xytext=(6.0, 3.8), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 1.55), xytext=(6.0, 2.65), arrowprops=arrow_props)
ax.annotate('', xy=(9.5, 4.25), xytext=(9.0, 4.25), arrowprops=arrow_props)
ax.annotate('', xy=(9.5, 3.25), xytext=(9.0, 2.65), arrowprops=arrow_props)
ax.annotate('', xy=(9.5, 1.55), xytext=(9.0, 1.55), arrowprops=arrow_props)
ax.annotate('', xy=(12, 4.25), xytext=(12.0, 4.25), arrowprops=arrow_props)

# Labels
ax.text(7, 0.1, 'Hierarchical decomposition by pseudo-depth enables sparse association within dense scenes',
        ha='center', fontsize=11, style='italic', color='gray')

plt.tight_layout()
plt.savefig('report/images/figure6_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure6_architecture.png")

print("\nAll figures generated successfully!")
