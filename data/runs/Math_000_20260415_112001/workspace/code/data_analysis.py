import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os

data_path = 'data/simulated_sequence.json'
with open(data_path, 'r') as f:
    data = json.load(f)

num_frames = len(data)
print(f'Number of frames: {num_frames}')

# Extract per frame
gt_bboxes_all = []
gt_ids_all = []
dets_all = []
unique_gt_ids = set()
frame_gt_counts = []
frame_det_counts = []
occs = []  # if present

for frame_data in data:
    gt_bboxes = np.array(frame_data['gt_bboxes'])
    gt_ids = frame_data['gt_ids']
    dets = frame_data['detections']  # assume list of [x1,y1,x2,y2,conf]
    
    gt_bboxes_all.append(gt_bboxes)
    gt_ids_all.extend(gt_ids)
    unique_gt_ids.update(gt_ids)
    dets_all.append(np.array(dets) if dets else np.empty((0,5)))
    
    frame_gt_counts.append(len(gt_bboxes))
    frame_det_counts.append(len(dets))
    
    # Assume detections [bbox4 + conf], check shape
    if len(dets) > 0:
        print(f'Frame 0 det shape: {np.array(dets).shape}')

num_objects = len(unique_gt_ids)
print(f'Number of unique GT IDs (objects): {num_objects}')

avg_gt_per_frame = np.mean(frame_gt_counts)
avg_det_per_frame = np.mean(frame_det_counts)
detection_rate = avg_det_per_frame / avg_gt_per_frame * 100
print(f'Avg GT per frame: {avg_gt_per_frame:.1f}')
print(f'Avg det per frame: {avg_det_per_frame:.1f}')
print(f'Detection rate: {detection_rate:.1f}%')

# Occlusion: assume in detections last is conf, or separate? From task, occlusion labels
# Check if 'occlusion_labels' or in dets
has_occ = 'occlusion_labels' in data[0]
print(f'Has occlusion_labels key: {has_occ}')

# GT trajectories length dist
traj_lengths = Counter()
for oid in unique_gt_ids:
    length = sum(1 for ids in gt_ids_all if ids == oid)  # rough
    traj_lengths[oid] = length  # fix later
print('Sample traj lengths:', dict(list(traj_lengths.items())[:5]))

# Save summary
summary = {
    'num_frames': num_frames,
    'num_objects': num_objects,
    'avg_gt_per_frame': float(avg_gt_per_frame),
    'avg_det_per_frame': float(avg_det_per_frame),
    'detection_rate_pct': float(detection_rate),
    'gt_ids_range': [min(unique_gt_ids), max(unique_gt_ids)]
}
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Plot data overview: GT and det counts over frames
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(frame_gt_counts, label='GT')
plt.plot(frame_det_counts, label='Dets')
plt.title('Counts per frame')
plt.legend()

plt.subplot(122)
plt.hist([l for l in traj_lengths.values()], bins=20)
plt.title('GT Trajectory lengths')

plt.savefig('report/images/data_overview.png')
print('Saved data_overview.png')
