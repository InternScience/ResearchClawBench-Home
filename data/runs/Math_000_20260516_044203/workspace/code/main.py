"""
Main analysis script for SparseTrack vs ByteTrack comparison
on simulated multi-object tracking data.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bytetrack import ByteTrack
from sparsetrack import SparseTrack
from evaluate import evaluate_tracking, compute_iou


def load_data(data_path):
    """Load the simulated MOT sequence data."""
    with open(data_path) as f:
        return json.load(f)


def run_tracker(tracker, data, tracker_name):
    """
    Run a tracker on the data sequence and collect outputs.
    """
    tracker_output = {}
    all_active_tracks = []
    
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        
        # Run tracker update
        active_tracks = tracker.update(detections)
        tracker_output[frame_idx] = active_tracks
        
        # Collect for later analysis
        for t in active_tracks:
            all_active_tracks.append({
                'frame': frame_idx,
                'id': t['id'],
                'bbox': t['bbox'],
            })
    
    return tracker_output, all_active_tracks


def run_sparsetrack(tracker, data, tracker_name):
    """
    Run SparseTrack (which needs frame dimensions).
    """
    tracker_output = {}
    all_active_tracks = []
    
    # Estimate frame dimensions from bounding boxes
    all_x2 = []
    all_y2 = []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
        for gt_bbox in frame_data['gt_bboxes']:
            all_x2.append(gt_bbox[2])
            all_y2.append(gt_bbox[3])
    
    frame_width = max(all_x2) * 1.1
    frame_height = max(all_y2) * 1.1
    
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        
        active_tracks = tracker.update(detections, frame_height, frame_width)
        tracker_output[frame_idx] = active_tracks
        
        for t in active_tracks:
            all_active_tracks.append({
                'frame': frame_idx,
                'id': t['id'],
                'bbox': t['bbox'],
            })
    
    return tracker_output, all_active_tracks


def analyze_detections(data):
    """Analyze detection quality in the dataset."""
    stats = {
        'total_frames': len(data),
        'total_detections': 0,
        'scores': [],
        'detections_per_frame': [],
        'unique_gt_ids': set(),
        'gt_per_frame': [],
        'detection_rate_per_frame': [],
    }
    
    for frame_data in data:
        dets = frame_data['detections']
        stats['total_detections'] += len(dets)
        stats['detections_per_frame'].append(len(dets))
        stats['gt_per_frame'].append(len(frame_data['gt_bboxes']))
        stats['detection_rate_per_frame'].append(
            len(dets) / max(1, len(frame_data['gt_bboxes'])))
        for det in dets:
            stats['scores'].append(det['score'])
        for gt_id in frame_data['gt_ids']:
            stats['unique_gt_ids'].add(gt_id)
    
    stats['n_unique_gt'] = len(stats['unique_gt_ids'])
    stats['scores'] = np.array(stats['scores'])
    
    return stats


def analyze_occlusions(data):
    """Analyze occlusion patterns in the dataset."""
    # Compute pairwise IoU between all GT bboxes per frame
    occlusion_stats = {
        'overlapping_pairs_per_frame': [],
        'max_overlap_per_frame': [],
        'avg_overlap_per_frame': [],
    }
    
    for frame_data in data:
        gt_bboxes = frame_data['gt_bboxes']
        n = len(gt_bboxes)
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou(gt_bboxes[i], gt_bboxes[j])
                if iou > 0.0:
                    overlaps.append(iou)
        
        occlusion_stats['overlapping_pairs_per_frame'].append(len(overlaps))
        occlusion_stats['max_overlap_per_frame'].append(max(overlaps) if overlaps else 0)
        occlusion_stats['avg_overlap_per_frame'].append(np.mean(overlaps) if overlaps else 0)
    
    return occlusion_stats


def plot_detection_analysis(data, stats, save_dir):
    """Plot detection quality analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Detections per frame
    ax = axes[0, 0]
    frames = [d['frame'] for d in data]
    dets_per_frame = stats['detections_per_frame']
    gt_per_frame = stats['gt_per_frame']
    ax.plot(frames, gt_per_frame, 'b-', label='GT Objects', alpha=0.7)
    ax.plot(frames, dets_per_frame, 'r--', label='Detections', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Objects & Detections per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Detection score distribution
    ax = axes[0, 1]
    ax.hist(stats['scores'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0.5, color='red', linestyle='--', label='High Thresh (0.5)')
    ax.axvline(x=0.2, color='orange', linestyle='--', label='Low Thresh (0.2)')
    ax.set_xlabel('Detection Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Detection Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Detection rate per frame
    ax = axes[1, 0]
    ax.plot(frames, stats['detection_rate_per_frame'], 'g-', alpha=0.7)
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate per Frame')
    ax.grid(True, alpha=0.3)
    
    # 4. Occlusion analysis
    ax = axes[1, 1]
    occ = analyze_occlusions(data)
    ax.plot(frames, occ['overlapping_pairs_per_frame'], 'purple', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Overlapping Pairs')
    ax.set_title('Occlusion Density per Frame')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_tracking_comparison(bt_metrics, st_metrics, save_dir):
    """Plot tracking performance comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    metrics_names = ['MOTA', 'IDF1', 'MOTP', 'ID_Switches', 'FP', 'FN']
    bt_values = [bt_metrics[n] for n in metrics_names]
    st_values = [st_metrics[n] for n in metrics_names]
    
    # Bar chart comparison
    ax = axes[0, 0]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, bt_values, width, label='ByteTrack', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, st_values, width, label='SparseTrack', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=30, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Tracking Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                ha='center', va='bottom', fontsize=7)
    
    # MOTA and IDF1 focus
    ax = axes[0, 1]
    categories = ['MOTA', 'IDF1']
    bt_vals = [bt_metrics['MOTA'], bt_metrics['IDF1']]
    st_vals = [st_metrics['MOTA'], st_metrics['IDF1']]
    x = np.arange(len(categories))
    ax.bar(x - 0.2, bt_vals, 0.35, label='ByteTrack', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, st_vals, 0.35, label='SparseTrack', color='#FF9800', alpha=0.8)
    for i, (bv, sv) in enumerate(zip(bt_vals, st_vals)):
        ax.text(i - 0.2, bv + 0.5, f'{bv:.1f}', ha='center', fontsize=9)
        ax.text(i + 0.2, sv + 0.5, f'{sv:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Score (%)')
    ax.set_title('MOTA & IDF1 Scores')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Detection Errors (FP and FN)
    ax = axes[0, 2]
    error_categories = ['False Positives', 'False Negatives', 'ID Switches']
    bt_errors = [bt_metrics['FP'], bt_metrics['FN'], bt_metrics['ID_Switches']]
    st_errors = [st_metrics['FP'], st_metrics['FN'], st_metrics['ID_Switches']]
    x = np.arange(len(error_categories))
    ax.bar(x - 0.2, bt_errors, 0.35, label='ByteTrack', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, st_errors, 0.35, label='SparseTrack', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(error_categories)
    ax.set_ylabel('Count')
    ax.set_title('Detection & ID Errors')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Track quality metrics
    ax = axes[1, 0]
    quality_metrics = ['MT', 'ML', 'Num_Tracks']
    bt_q = [bt_metrics['MT'], bt_metrics['ML'], bt_metrics['Num_Tracks']]
    st_q = [st_metrics['MT'], st_metrics['ML'], st_metrics['Num_Tracks']]
    x = np.arange(len(quality_metrics))
    ax.bar(x - 0.2, bt_q, 0.35, label='ByteTrack', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, st_q, 0.35, label='SparseTrack', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Mostly Tracked', 'Mostly Lost', 'Total Tracks'])
    ax.set_ylabel('Count')
    ax.set_title('Track Quality Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Radar / spider chart-like comparison (normalized)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'ByteTrack', 'SparseTrack'],
        ['MOTA (%)', f'{bt_metrics["MOTA"]:.2f}', f'{st_metrics["MOTA"]:.2f}'],
        ['IDF1 (%)', f'{bt_metrics["IDF1"]:.2f}', f'{st_metrics["IDF1"]:.2f}'],
        ['MOTP (%)', f'{bt_metrics["MOTP"]:.2f}', f'{st_metrics["MOTP"]:.2f}'],
        ['ID Switches', f'{bt_metrics["ID_Switches"]}', f'{st_metrics["ID_Switches"]}'],
        ['FP', f'{bt_metrics["FP"]}', f'{st_metrics["FP"]}'],
        ['FN', f'{bt_metrics["FN"]}', f'{st_metrics["FN"]}'],
        ['MT', f'{bt_metrics["MT"]}', f'{st_metrics["MT"]}'],
        ['ML', f'{bt_metrics["ML"]}', f'{st_metrics["ML"]}'],
    ]
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    ax.set_title('Performance Summary', y=0.75)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracking_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory_visualization(data, bt_output, st_output, save_dir, max_frames=20):
    """Visualize tracking trajectories for a subset of frames."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Select frames to display (evenly spaced)
    total_frames = len(data)
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames)).astype(int)
    
    # Trajectory continuity analysis
    ax = axes[0, 0]
    
    # For each GT ID, check how consistently it's tracked by each method
    gt_coverage_bt = defaultdict(list)
    gt_coverage_st = defaultdict(list)
    
    for frame_data in data:
        frame_idx = frame_data['frame']
        
        # ByteTrack coverage
        bt_tracks = bt_output.get(frame_idx, [])
        for gt_id, gt_bbox in zip(frame_data['gt_ids'], frame_data['gt_bboxes']):
            covered = any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in bt_tracks)
            gt_coverage_bt[gt_id].append(1 if covered else 0)
        
        # SparseTrack coverage
        st_tracks = st_output.get(frame_idx, [])
        for gt_id, gt_bbox in zip(frame_data['gt_ids'], frame_data['gt_bboxes']):
            covered = any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in st_tracks)
            gt_coverage_st[gt_id].append(1 if covered else 0)
    
    # Compute per-GT coverage rates
    bt_coverages = [np.mean(v) for v in gt_coverage_bt.values()]
    st_coverages = [np.mean(v) for v in gt_coverage_st.values()]
    
    ax.bar(np.arange(len(bt_coverages)) - 0.2, bt_coverages, 0.35, 
           alpha=0.6, color='#2196F3', label='ByteTrack')
    ax.bar(np.arange(len(st_coverages)) + 0.2, st_coverages, 0.35,
           alpha=0.6, color='#FF9800', label='SparseTrack')
    ax.set_xlabel('GT Object ID')
    ax.set_ylabel('Coverage Rate')
    ax.set_title('Per-Object Tracking Coverage')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Coverage rate comparison scatter
    ax = axes[0, 1]
    ax.scatter(bt_coverages, st_coverages, alpha=0.5, c='purple', s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('ByteTrack Coverage')
    ax.set_ylabel('SparseTrack Coverage')
    ax.set_title('Coverage Rate Comparison')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Track length distribution
    ax = axes[1, 0]
    bt_track_lengths = [len(v) for v in gt_coverage_bt.values()]
    st_track_lengths = [len(v) for v in gt_coverage_st.values()]
    
    # Count track lengths >= threshold
    thresholds = np.arange(0, 1.05, 0.05)
    bt_above = [sum(1 for c in bt_coverages if c >= t) for t in thresholds]
    st_above = [sum(1 for c in st_coverages if c >= t) for t in thresholds]
    
    ax.plot(thresholds, bt_above, 'b-', linewidth=2, label='ByteTrack')
    ax.plot(thresholds, st_above, 'orange', linewidth=2, label='SparseTrack')
    ax.fill_between(thresholds, bt_above, alpha=0.2, color='blue')
    ax.fill_between(thresholds, st_above, alpha=0.2, color='orange')
    ax.set_xlabel('Coverage Threshold')
    ax.set_ylabel('Number of Objects Above Threshold')
    ax.set_title('Coverage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Frame-level tracking accuracy
    ax = axes[1, 1]
    frame_mota_bt = []
    frame_mota_st = []
    
    for frame_data in data:
        frame_idx = frame_data['frame']
        gt_count = len(frame_data['gt_bboxes'])
        
        bt_tracks = bt_output.get(frame_idx, [])
        st_tracks = st_output.get(frame_idx, [])
        
        # Compute frame-level accuracy (# matched / # GT)
        bt_matched = 0
        for gt_bbox in frame_data['gt_bboxes']:
            if any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in bt_tracks):
                bt_matched += 1
        st_matched = 0
        for gt_bbox in frame_data['gt_bboxes']:
            if any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in st_tracks):
                st_matched += 1
        
        frame_mota_bt.append(bt_matched / max(1, gt_count))
        frame_mota_st.append(st_matched / max(1, gt_count))
    
    frames = [d['frame'] for d in data]
    ax.plot(frames, frame_mota_bt, 'b-', alpha=0.7, label='ByteTrack')
    ax.plot(frames, frame_mota_st, 'orange', alpha=0.7, label='SparseTrack')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame Accuracy')
    ax.set_title('Frame-Level Tracking Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_occlusion_performance(data, bt_output, st_output, save_dir):
    """Analyze tracking performance under different occlusion levels."""
    
    # Compute per-frame occlusion level and tracking accuracy
    frame_occlusion = []
    frame_bt_acc = []
    frame_st_acc = []
    
    for frame_data in data:
        frame_idx = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        
        # Compute occlusion level: average pairwise IoU
        overlaps = []
        n = len(gt_bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou(gt_bboxes[i], gt_bboxes[j])
                if iou > 0:
                    overlaps.append(iou)
        occ_level = np.mean(overlaps) if overlaps else 0
        
        # ByteTrack accuracy
        bt_tracks = bt_output.get(frame_idx, [])
        bt_matched = sum(1 for gt_bbox in gt_bboxes 
                        if any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in bt_tracks))
        bt_acc = bt_matched / max(1, len(gt_bboxes))
        
        # SparseTrack accuracy
        st_tracks = st_output.get(frame_idx, [])
        st_matched = sum(1 for gt_bbox in gt_bboxes
                        if any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in st_tracks))
        st_acc = st_matched / max(1, len(gt_bboxes))
        
        frame_occlusion.append(occ_level)
        frame_bt_acc.append(bt_acc)
        frame_st_acc.append(st_acc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: occlusion vs accuracy
    ax = axes[0]
    ax.scatter(frame_occlusion, frame_bt_acc, alpha=0.5, c='#2196F3', label='ByteTrack', s=40)
    ax.scatter(frame_occlusion, frame_st_acc, alpha=0.5, c='#FF9800', label='SparseTrack', s=40)
    
    # Add trend lines
    z_bt = np.polyfit(frame_occlusion, frame_bt_acc, 1)
    z_st = np.polyfit(frame_occlusion, frame_st_acc, 1)
    x_line = np.linspace(min(frame_occlusion), max(frame_occlusion), 100)
    ax.plot(x_line, np.polyval(z_bt, x_line), 'b--', alpha=0.7)
    ax.plot(x_line, np.polyval(z_st, x_line), 'orange', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Occlusion Level (Avg IoU)')
    ax.set_ylabel('Frame Accuracy')
    ax.set_title('Tracking Accuracy vs Occlusion Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance difference (SparseTrack - ByteTrack) vs occlusion
    ax = axes[1]
    diff = np.array(frame_st_acc) - np.array(frame_bt_acc)
    colors = ['green' if d > 0 else 'red' for d in diff]
    ax.bar(range(len(diff)), diff, color=colors, alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Accuracy Difference (ST - BT)')
    ax.set_title('SparseTrack vs ByteTrack per Frame')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'occlusion_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_depth_analysis(data, st_output, save_dir):
    """Analyze pseudo-depth estimation and its effect on tracking."""
    from sparsetrack import SparseTrack
    
    # Instantiate a tracker to use its depth estimation
    tracker = SparseTrack(num_depth_layers=3)
    
    all_bboxes = []
    for frame_data in data:
        for det in frame_data['detections']:
            all_bboxes.append(det['bbox'])
    
    all_x2 = [b[2] for b in all_bboxes]
    all_y2 = [b[3] for b in all_bboxes]
    frame_width = max(all_x2) * 1.1
    frame_height = max(all_y2) * 1.1
    
    # Estimate depths for all detections in first frame
    depths = []
    areas = []
    y_positions = []
    for det in data[0]['detections']:
        bbox = det['bbox']
        depth = tracker.estimate_pseudo_depth(bbox, frame_height, frame_width)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        y_pos = (bbox[1] + bbox[3]) / 2.0
        depths.append(depth)
        areas.append(area)
        y_positions.append(y_pos)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Area vs depth
    ax = axes[0]
    ax.scatter(areas, depths, alpha=0.5, c=depths, cmap='viridis', s=30)
    ax.set_xlabel('Bounding Box Area')
    ax.set_ylabel('Pseudo-Depth')
    ax.set_title('Area vs Pseudo-Depth')
    ax.grid(True, alpha=0.3)
    
    # Y position vs depth
    ax = axes[1]
    ax.scatter(y_positions, depths, alpha=0.5, c=depths, cmap='viridis', s=30)
    ax.set_xlabel('Vertical Position (center y)')
    ax.set_ylabel('Pseudo-Depth')
    ax.set_title('Y-Position vs Pseudo-Depth')
    ax.grid(True, alpha=0.3)
    
    # Depth distribution
    ax = axes[2]
    ax.hist(depths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    for i in range(3):
        ax.axvline(x=(i+1)/3, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pseudo-Depth')
    ax.set_ylabel('Frequency')
    ax.set_title('Pseudo-Depth Distribution (3 Layers)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'depth_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Paths
    data_path = 'data/simulated_sequence.json'
    output_dir = 'outputs'
    images_dir = 'report/images'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Object Tracking: SparseTrack vs ByteTrack")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    data = load_data(data_path)
    print(f"  Loaded {len(data)} frames")
    
    # Analyze data
    print("\n[2/6] Analyzing data...")
    stats = analyze_detections(data)
    print(f"  Unique GT objects: {stats['n_unique_gt']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Detection score range: [{stats['scores'].min():.4f}, {stats['scores'].max():.4f}]")
    print(f"  Mean detection score: {stats['scores'].mean():.4f}")
    
    # Plot detection analysis
    plot_detection_analysis(data, stats, images_dir)
    print("  Saved detection analysis plots")
    
    # Run ByteTrack
    print("\n[3/6] Running ByteTrack...")
    bt_tracker = ByteTrack(
        track_high_thresh=0.2,
        track_low_thresh=0.1,
        match_thresh=0.2,
        max_age=30,
        min_hits=3,
    )
    bt_output, bt_all_tracks = run_tracker(bt_tracker, data, 'ByteTrack')
    bt_metrics = evaluate_tracking(data, bt_output)
    print(f"  MOTA: {bt_metrics['MOTA']:.2f}%")
    print(f"  IDF1: {bt_metrics['IDF1']:.2f}%")
    print(f"  ID Switches: {bt_metrics['ID_Switches']}")
    print(f"  MT: {bt_metrics['MT']}, ML: {bt_metrics['ML']}")
    
    # Run SparseTrack
    print("\n[4/6] Running SparseTrack...")
    st_tracker = SparseTrack(
        num_depth_layers=3,
        track_high_thresh=0.2,
        track_low_thresh=0.1,
        match_thresh=0.2,
        max_age=30,
        min_hits=3,
        depth_method='combined',
    )
    st_output, st_all_tracks = run_sparsetrack(st_tracker, data, 'SparseTrack')
    st_metrics = evaluate_tracking(data, st_output)
    print(f"  MOTA: {st_metrics['MOTA']:.2f}%")
    print(f"  IDF1: {st_metrics['IDF1']:.2f}%")
    print(f"  ID Switches: {st_metrics['ID_Switches']}")
    print(f"  MT: {st_metrics['MT']}, ML: {st_metrics['ML']}")
    
    # Save metrics
    print("\n[5/6] Saving results and generating figures...")
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'ByteTrack': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in bt_metrics.items()},
            'SparseTrack': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in st_metrics.items()},
        }, f, indent=2)
    
    # Save detailed trajectory data
    with open(os.path.join(output_dir, 'bytetrack_tracks.json'), 'w') as f:
        json.dump(bt_all_tracks, f)
    with open(os.path.join(output_dir, 'sparsetrack_tracks.json'), 'w') as f:
        json.dump(st_all_tracks, f)
    
    # Generate plots
    plot_tracking_comparison(bt_metrics, st_metrics, images_dir)
    print("  Saved tracking comparison plots")
    
    plot_trajectory_visualization(data, bt_output, st_output, images_dir)
    print("  Saved trajectory analysis plots")
    
    plot_occlusion_performance(data, bt_output, st_output, images_dir)
    print("  Saved occlusion analysis plots")
    
    compute_depth_analysis(data, st_output, images_dir)
    print("  Saved depth analysis plots")
    
    # Generate report
    print("\n[6/6] Generating report...")
    
    # Compute additional stats for the report
    # Ablation study: test SparseTrack with different depth layers
    ablation_results = {}
    for n_layers in [1, 2, 3, 5]:
        st_abl = SparseTrack(
            num_depth_layers=n_layers,
            track_high_thresh=0.2,
            track_low_thresh=0.1,
            match_thresh=0.2,
            max_age=30,
            min_hits=3,
            depth_method='combined',
        )
        st_abl_output, _ = run_sparsetrack(st_abl, data, f'SparseTrack-L{n_layers}')
        st_abl_metrics = evaluate_tracking(data, st_abl_output)
        ablation_results[f'L{n_layers}'] = {
            'MOTA': round(st_abl_metrics['MOTA'], 2),
            'IDF1': round(st_abl_metrics['IDF1'], 2),
            'IDS': st_abl_metrics['ID_Switches'],
            'MT': st_abl_metrics['MT'],
        }
    
    with open(os.path.join(output_dir, 'ablation.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("  Ablation results:")
    for k, v in ablation_results.items():
        print(f"    {k}: MOTA={v['MOTA']:.2f}, IDF1={v['IDF1']:.2f}, IDS={v['IDS']}")
    
    # Save detection stats
    with open(os.path.join(output_dir, 'data_stats.json'), 'w') as f:
        json.dump({
            'total_frames': stats['total_frames'],
            'n_unique_gt': stats['n_unique_gt'],
            'total_detections': stats['total_detections'],
            'score_mean': float(stats['scores'].mean()),
            'score_std': float(stats['scores'].std()),
            'score_min': float(stats['scores'].min()),
            'score_max': float(stats['scores'].max()),
            'detection_rate_mean': float(np.mean(stats['detection_rate_per_frame'])),
            'detection_rate_std': float(np.std(stats['detection_rate_per_frame'])),
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Results saved to outputs/ and report/images/")
    print("=" * 60)
    
    return bt_metrics, st_metrics, ablation_results, stats


if __name__ == '__main__':
    main()
