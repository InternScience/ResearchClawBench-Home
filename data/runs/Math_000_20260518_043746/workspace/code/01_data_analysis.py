"""
Data analysis and visualization for simulated multi-object tracking sequence.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

with open('data/simulated_sequence.json') as f:
    data = json.load(f)

print(f"Total frames: {len(data)}")
print(f"Objects per frame: {len(data[0]['gt_bboxes'])}")

# === Analysis 1: Detection statistics per frame ===
frames = [d['frame'] for d in data]
n_dets = [len(d['detections']) for d in data]
n_gt = [len(d['gt_bboxes']) for d in data]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Detections vs GT per frame
ax = axes[0, 0]
ax.plot(frames, n_gt, 'b-', linewidth=2, label='Ground Truth')
ax.plot(frames, n_dets, 'r-', linewidth=2, label='Detections')
ax.fill_between(frames, n_dets, n_gt, alpha=0.2, color='red', label='Missed detections')
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Detections vs Ground Truth per Frame', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Score distribution
ax = axes[0, 1]
all_scores = []
for d in data:
    for det in d['detections']:
        all_scores.append(det['score'])
ax.hist(all_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
ax.axvline(x=np.mean(all_scores), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_scores):.3f}')
ax.set_xlabel('Detection Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Detection Score Distribution', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: Detection rate per frame
ax = axes[1, 0]
det_rates = [len(d['detections']) / len(d['gt_bboxes']) * 100 for d in data]
ax.plot(frames, det_rates, 'g-', linewidth=2)
ax.axhline(y=np.mean(det_rates), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(det_rates):.1f}%')
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Detection Rate (%)', fontsize=12)
ax.set_title('Detection Rate per Frame', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 4: Missed objects per frame (ID sets)
ax = axes[1, 1]
missed_ids = []
detected_ids = []
for d in data:
    gt_set = set(d['gt_ids'])
    det_set = set(det['gt_id'] for det in d['detections'])
    missed_ids.append(len(gt_set - det_set))
    detected_ids.append(len(det_set))
ax.bar(frames, missed_ids, color='coral', alpha=0.7, width=1.0)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Missed Object Count', fontsize=12)
ax.set_title('Missed Objects per Frame', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1: Data overview saved.")

# === Analysis 2: Spatial distribution and pseudo-depth ===
# Compute bbox areas as pseudo-depth proxy
all_areas = []
all_center_y = []
for d in data:
    for det in d['detections']:
        bbox = det['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        center_y = (bbox[1] + bbox[3]) / 2
        all_areas.append(area)
        all_center_y.append(center_y)

# Pseudo-depth: inverse of bbox area (larger = closer = lower depth value)
all_pseudo_depth = 1.0 / (np.sqrt(np.array(all_areas)) + 1e-6)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Bbox area distribution
ax = axes[0]
ax.hist(all_areas, bins=60, edgecolor='black', alpha=0.7, color='purple')
ax.set_xlabel('Bounding Box Area (pixels²)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Detection Size Distribution\n(Pseudo-Depth Proxy)', fontsize=13)
ax.grid(True, alpha=0.3)

# Panel 2: Pseudo-depth distribution
ax = axes[1]
ax.hist(all_pseudo_depth, bins=60, edgecolor='black', alpha=0.7, color='teal')
ax.set_xlabel('Pseudo-Depth (1/√area)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Pseudo-Depth Distribution', fontsize=13)
ax.grid(True, alpha=0.3)

# Panel 3: Area vs Center-Y scatter
ax = axes[2]
sc = ax.scatter(all_center_y, all_areas, c=all_pseudo_depth, cmap='viridis', alpha=0.1, s=5)
plt.colorbar(sc, ax=ax, label='Pseudo-Depth')
ax.set_xlabel('Center Y (pixels)', fontsize=12)
ax.set_ylabel('BBox Area (pixels²)', fontsize=12)
ax.set_title('Size vs Vertical Position\n(Color=Pseudo-Depth)', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_pseudo_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2: Pseudo-depth analysis saved.")

# === Analysis 3: Occlusion / overlap analysis ===
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Compute max IoU per detection in each frame
max_ious_per_frame = []
avg_ious_per_frame = []
for d in data:
    gt_bboxes = d['gt_bboxes']
    ious = []
    for det in d['detections']:
        max_iou = 0
        det_bbox = det['bbox']
        for gt_bbox in gt_bboxes:
            iou = compute_iou(det_bbox, gt_bbox)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
    max_ious_per_frame.append(np.mean(ious) if ious else 0)
    
    # Count detections with IoU < 0.5 with any GT
    low_iou_count = sum(1 for i in ious if i < 0.5)
    avg_ious_per_frame.append(low_iou_count)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Average IoU per frame
ax = axes[0]
ax.plot(frames, max_ious_per_frame, 'b-', linewidth=2)
ax.axhline(y=np.mean(max_ious_per_frame), color='red', linestyle='--', label=f'Mean={np.mean(max_ious_per_frame):.3f}')
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Mean Max IoU with GT', fontsize=12)
ax.set_title('Detection Localization Quality per Frame', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Low-IoU detections per frame
ax = axes[1]
ax.plot(frames, avg_ious_per_frame, 'r-', linewidth=2)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Low-IoU Detections (< 0.5)', fontsize=12)
ax.set_title('Poorly Localized Detections per Frame', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_occlusion_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3: Occlusion analysis saved.")

# Save analysis results
analysis = {
    'total_frames': len(data),
    'objects_per_frame': len(data[0]['gt_bboxes']),
    'detections_per_frame': {'min': min(n_dets), 'max': max(n_dets), 'mean': float(np.mean(n_dets))},
    'detection_rate': {'min': min(det_rates), 'max': max(det_rates), 'mean': float(np.mean(det_rates))},
    'score_stats': {'min': float(np.min(all_scores)), 'max': float(np.max(all_scores)), 
                    'mean': float(np.mean(all_scores)), 'std': float(np.std(all_scores))},
    'area_stats': {'min': float(np.min(all_areas)), 'max': float(np.max(all_areas)),
                   'mean': float(np.mean(all_areas))},
    'pseudo_depth_stats': {'min': float(np.min(all_pseudo_depth)), 'max': float(np.max(all_pseudo_depth)),
                           'mean': float(np.mean(all_pseudo_depth))}
}

with open('outputs/data_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
print("Data analysis complete.")
print(json.dumps(analysis, indent=2))
