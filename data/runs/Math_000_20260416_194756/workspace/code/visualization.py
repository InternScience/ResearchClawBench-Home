#!/usr/bin/env python3
"""
Visualization module for multi-object tracking evaluation.
Generates all required figures for the research report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_detection_statistics(data_analysis, output_path):
    """Plot detection statistics overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    frame_stats = data_analysis['frame_stats']
    frames = [s['frame'] for s in frame_stats]
    
    # Plot 1: Detections per frame
    ax = axes[0, 0]
    num_gt = [s['num_gt'] for s in frame_stats]
    num_dets = [s['num_detections'] for s in frame_stats]
    ax.plot(frames, num_gt, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(frames, num_dets, 'b-', label='Detections', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Objects and Detections per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Detection rate per frame
    ax = axes[0, 1]
    det_rates = [s['detection_rate'] for s in frame_stats]
    ax.plot(frames, det_rates, 'r-', linewidth=2)
    ax.axhline(y=np.mean(det_rates), color='orange', linestyle='--', 
               label=f'Avg: {np.mean(det_rates):.2%}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 3: Detection score distribution
    ax = axes[1, 0]
    avg_scores = [s['avg_score'] for s in frame_stats]
    ax.plot(frames, avg_scores, 'purple', linewidth=2)
    ax.axhline(y=data_analysis['avg_detection_score'], color='red', linestyle='--',
               label=f'Avg: {data_analysis["avg_detection_score"]:.3f}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Detection Score per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Occluded detections per frame
    ax = axes[1, 1]
    occluded = [s['occluded_count'] for s in frame_stats]
    ax.bar(frames, occluded, color='orange', alpha=0.7)
    ax.axhline(y=data_analysis['occlusion_stats']['avg_occluded_per_frame'], 
               color='red', linestyle='--', linewidth=2,
               label=f'Avg: {data_analysis["occlusion_stats"]["avg_occluded_per_frame"]:.1f}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Count')
    ax.set_title('Low-Score (Potentially Occluded) Detections per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trajectory_overview(data, output_path, max_objects=30):
    """Plot ground truth trajectory overview."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get all unique object IDs
    all_ids = set()
    for frame_data in data:
        all_ids.update(frame_data.get('gt_ids', []))
    
    # Select subset for clarity
    selected_ids = sorted(list(all_ids))[:max_objects]
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, max_objects))
    
    for idx, obj_id in enumerate(selected_ids):
        x_centers = []
        y_centers = []
        frames_present = []
        
        for frame_data in data:
            frame = frame_data['frame']
            gt_ids = frame_data.get('gt_ids', [])
            gt_bboxes = frame_data.get('gt_bboxes', [])
            
            if obj_id in gt_ids:
                local_idx = gt_ids.index(obj_id)
                bbox = gt_bboxes[local_idx]
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                x_centers.append(x_center)
                y_centers.append(y_center)
                frames_present.append(frame)
        
        if len(x_centers) > 1:
            ax.scatter(frames_present, y_centers, c=[colors[idx]] * len(frames_present),
                      s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Vertical Position (y)', fontsize=12)
    ax.set_title(f'Ground Truth Trajectories Overview (showing {max_objects} objects)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Image coordinates
    
    # Add legend handles
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'ID {selected_ids[i]}')
                       for i in range(min(10, len(selected_ids)))]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tracking_comparison(sparse_results, byte_results, output_path):
    """Plot comparison of SparseTrack vs ByteTrack metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sparse_metrics = sparse_results['metrics']
    byte_metrics = byte_results['metrics']
    
    methods = ['SparseTrack', 'ByteTrack']
    
    # Plot 1: MOTA comparison
    ax = axes[0]
    mota_vals = [sparse_metrics['mota'], byte_metrics['mota']]
    bars = ax.bar(methods, mota_vals, color=['steelblue', 'coral'], edgecolor='black')
    ax.set_ylabel('MOTA', fontsize=12)
    ax.set_title('Multi-Object Tracking Accuracy', fontsize=14)
    ax.set_ylim(0, max(mota_vals) * 1.2)
    for bar, val in zip(bars, mota_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Precision/Recall comparison
    ax = axes[1]
    x = np.arange(len(methods))
    width = 0.35
    prec_vals = [sparse_metrics['precision'], byte_metrics['precision']]
    rec_vals = [sparse_metrics['recall'], byte_metrics['recall']]
    
    bars1 = ax.bar(x - width/2, prec_vals, width, label='Precision', color='forestgreen')
    bars2 = ax.bar(x + width/2, rec_vals, width, label='Recall', color='goldenrod')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision and Recall', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: ID Switches and Trajectories
    ax = axes[2]
    id_switches = [sparse_results['id_switches'], byte_results['id_switches']]
    num_trajs = [sparse_results['num_trajectories'], byte_results['num_trajectories']]
    
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, id_switches, width, label='ID Switches', color='crimson')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, num_trajs, width, label='Trajectories', color='teal')
    
    ax.set_ylabel('ID Switches', fontsize=12, color='crimson')
    ax2.set_ylabel('Number of Trajectories', fontsize=12, color='teal')
    ax.set_title('ID Switches and Trajectories', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_distribution(data, output_path):
    """Plot detection score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect all scores
    all_scores = []
    high_scores = []
    low_scores = []
    
    for frame_data in data:
        for det in frame_data.get('detections', []):
            score = det['score']
            all_scores.append(score)
            if score >= 0.5:
                high_scores.append(score)
            else:
                low_scores.append(score)
    
    # Plot 1: Overall distribution
    ax = axes[0]
    ax.hist(all_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.set_xlabel('Detection Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Detection Score Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: High vs Low score comparison
    ax = axes[1]
    categories = ['High Score\n(≥0.5)', 'Low Score\n(<0.5)']
    counts = [len(high_scores), len(low_scores)]
    percentages = [c/sum(counts)*100 for c in counts]
    
    bars = ax.bar(categories, counts, color=['forestgreen', 'coral'], edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('High vs Low Score Detections', fontsize=14)
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_frame_visualization(data, frame_num, output_path):
    """Visualize a single frame with detections and ground truth."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Find the frame
    frame_data = None
    for fd in data:
        if fd['frame'] == frame_num:
            frame_data = fd
            break
    
    if frame_data is None:
        print(f"Frame {frame_num} not found")
        return
    
    gt_bboxes = frame_data.get('gt_bboxes', [])
    gt_ids = frame_data.get('gt_ids', [])
    detections = frame_data.get('detections', [])
    
    # Plot ground truth boxes (green)
    for i, (bbox, gid) in enumerate(zip(gt_bboxes[:20], gt_ids[:20])):  # Limit for clarity
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        rect = Rectangle((x, y), w, h, fill=False, color='green', linewidth=2, 
                        label='GT' if i == 0 else '')
        ax.add_patch(rect)
        ax.text(x, y-5, f'GT:{gid}', color='green', fontsize=8, fontweight='bold')
    
    # Plot detection boxes (red for low score, blue for high score)
    for i, det in enumerate(detections[:20]):  # Limit for clarity
        bbox = det['bbox']
        score = det['score']
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        color = 'blue' if score >= 0.5 else 'red'
        rect = Rectangle((x, y), w, h, fill=False, color=color, linewidth=1.5,
                        linestyle='--', label='Det (high)' if score >= 0.5 and i == 0 else
                               'Det (low)' if score < 0.5 and i == 0 else '')
        ax.add_patch(rect)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(640, 0)  # Invert y for image coordinates
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Frame {frame_num}: Ground Truth (green) and Detections (blue=high, red=low score)', 
                fontsize=14)
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', linewidth=2, label='Ground Truth'),
        Patch(facecolor='none', edgecolor='blue', linewidth=1.5, linestyle='--', label='High Score Det'),
        Patch(facecolor='none', edgecolor='red', linewidth=1.5, linestyle='--', label='Low Score Det')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pseudo_depth_distribution(data, output_path):
    """Plot pseudo-depth distribution for SparseTrack analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute pseudo-depth for all detections in a sample of frames
    def estimate_pseudo_depth(bbox, frame_height=640):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        center_y = (y1 + y2) / 2
        depth_from_y = 1.0 - (center_y / frame_height)
        depth_from_area = 1.0 / (1.0 + np.sqrt(area) / 100)
        pseudo_depth = 0.6 * depth_from_y + 0.4 * depth_from_area
        return pseudo_depth
    
    all_depths = []
    depths_by_frame = {}
    
    for frame_data in data[:20]:  # Sample first 20 frames
        frame = frame_data['frame']
        depths = []
        for det in frame_data.get('detections', []):
            depth = estimate_pseudo_depth(det['bbox'])
            all_depths.append(depth)
            depths.append(depth)
        depths_by_frame[frame] = depths
    
    # Plot 1: Overall pseudo-depth distribution
    ax = axes[0]
    ax.hist(all_depths, bins=30, color='purple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pseudo-Depth Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pseudo-Depth Distribution (All Detections)', fontsize=14)
    ax.axvline(x=np.mean(all_depths), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_depths):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pseudo-depth by frame (boxplot style visualization)
    ax = axes[1]
    frames = list(depths_by_frame.keys())
    frame_means = [np.mean(depths_by_frame[f]) for f in frames]
    ax.plot(frames, frame_means, 'o-', color='purple', markersize=6)
    ax.fill_between(frames, 
                    [np.mean(all_depths) - np.std(all_depths)] * len(frames),
                    [np.mean(all_depths) + np.std(all_depths)] * len(frames),
                    alpha=0.3, color='purple', label='±1 std')
    ax.axhline(y=np.mean(all_depths), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Mean Pseudo-Depth', fontsize=12)
    ax.set_title('Mean Pseudo-Depth per Frame', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_occlusion_analysis(data_analysis, output_path):
    """Plot occlusion-related analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    frame_stats = data_analysis['frame_stats']
    frames = [s['frame'] for s in frame_stats]
    
    # Plot 1: Occluded count vs detection rate
    ax = axes[0]
    occluded = [s['occluded_count'] for s in frame_stats]
    det_rates = [s['detection_rate'] for s in frame_stats]
    
    ax.scatter(occluded, det_rates, alpha=0.5, color='coral', s=50, edgecolors='black')
    z = np.polyfit(occluded, det_rates, 1)
    p = np.poly1d(z)
    ax.plot(sorted(occluded), p(sorted(occluded)), "r--", linewidth=2, label='Trend line')
    
    ax.set_xlabel('Number of Low-Score Detections', fontsize=12)
    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_title('Occlusion Impact on Detection Rate', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time series of occluded detections
    ax = axes[1]
    ax.fill_between(frames, occluded, alpha=0.5, color='coral')
    ax.plot(frames, occluded, color='darkred', linewidth=2)
    ax.axhline(y=data_analysis['occlusion_stats']['avg_occluded_per_frame'], 
               color='black', linestyle='--', linewidth=2,
               label=f'Average: {data_analysis["occlusion_stats"]["avg_occluded_per_frame"]:.1f}')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Low-Score Detections', fontsize=12)
    ax.set_title('Occlusion Pattern Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    # Paths
    base_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756')
    data_path = base_path / 'data' / 'simulated_sequence.json'
    output_dir = base_path / 'report' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    data = load_json(data_path)
    data_analysis = load_json(base_path / 'outputs' / 'data_analysis.json')
    sparse_results = load_json(base_path / 'outputs' / 'sparsetrack_results.json')
    byte_results = load_json(base_path / 'outputs' / 'bytetrack_results.json')
    
    print("\nGenerating figures...")
    
    # Figure 1: Detection statistics overview
    plot_detection_statistics(data_analysis, output_dir / 'data_overview_detections.png')
    
    # Figure 2: Trajectory overview
    plot_trajectory_overview(data, output_dir / 'data_overview_trajectories.png')
    
    # Figure 3: Score distribution
    plot_score_distribution(data, output_dir / 'data_score_distribution.png')
    
    # Figure 4: Pseudo-depth distribution
    plot_pseudo_depth_distribution(data, output_dir / 'pseudo_depth_distribution.png')
    
    # Figure 5: Occlusion analysis
    plot_occlusion_analysis(data_analysis, output_dir / 'occlusion_analysis.png')
    
    # Figure 6: Method comparison
    plot_tracking_comparison(sparse_results, byte_results, output_dir / 'comparison_metrics.png')
    
    # Figure 7: Frame visualization
    plot_frame_visualization(data, frame_num=10, output_path=output_dir / 'frame_visualization.png')
    
    print("\n=== All figures generated successfully ===")
