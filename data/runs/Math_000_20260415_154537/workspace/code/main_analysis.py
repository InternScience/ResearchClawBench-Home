"""
Main Analysis Script for Multi-Object Tracking
==============================================

This script:
1. Loads the simulated sequence data
2. Runs SORT, ByteTrack, and SparseTrack algorithms
3. Evaluates performance using MOT metrics
4. Generates visualizations and reports results
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import tracking algorithms
sys.path.append(str(Path(__file__).parent))
from tracking_algorithms import SORT, ByteTrack, SparseTrack, compute_pseudo_depth
from evaluation import compute_clear_mot_metrics


def load_data(filepath):
    """Load simulated sequence data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def prepare_ground_truth(data):
    """Convert data to ground truth format."""
    ground_truth = {}
    for frame_data in data:
        frame_id = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        gt_ids = frame_data['gt_ids']
        
        ground_truth[frame_id] = [
            (gt_id, np.array(bbox)) 
            for gt_id, bbox in zip(gt_ids, gt_bboxes)
        ]
    return ground_truth


def prepare_detections(data):
    """Convert data to detection format."""
    detections_by_frame = {}
    for frame_data in data:
        frame_id = frame_data['frame']
        detections = frame_data['detections']
        
        detections_by_frame[frame_id] = [
            {
                'bbox': np.array(d['bbox']),
                'score': d['score'],
                'gt_id': d.get('gt_id', -1)
            }
            for d in detections
        ]
    return detections_by_frame


def run_tracker(tracker_class, tracker_name, detections_by_frame, **kwargs):
    """Run a tracker on all frames."""
    print(f"\nRunning {tracker_name}...")
    
    # Reset track ID counter
    from tracking_algorithms import Track
    Track._id_counter = 0
    
    tracker = tracker_class(**kwargs)
    predictions = {}
    
    frame_ids = sorted(detections_by_frame.keys())
    
    for frame_id in frame_ids:
        dets = detections_by_frame[frame_id]
        results = tracker.update(dets, frame_id)
        predictions[frame_id] = [(tid, bbox) for tid, bbox in results]
    
    return predictions


def analyze_data(data, output_dir):
    """Generate data overview visualizations."""
    print("\nGenerating data overview...")
    
    # Count statistics per frame
    frame_stats = []
    for frame_data in data:
        frame_id = frame_data['frame']
        n_gt = len(frame_data['gt_bboxes'])
        n_det = len(frame_data['detections'])
        
        # Compute detection rate
        det_rate = n_det / n_gt if n_gt > 0 else 0
        
        # Average detection score
        avg_score = np.mean([d['score'] for d in frame_data['detections']]) if n_det > 0 else 0
        
        # Compute pseudo-depths
        depths = [compute_pseudo_depth(np.array(bbox)) for bbox in frame_data['gt_bboxes']]
        avg_depth = np.mean(depths)
        
        frame_stats.append({
            'frame': frame_id,
            'n_gt': n_gt,
            'n_det': n_det,
            'det_rate': det_rate,
            'avg_score': avg_score,
            'avg_depth': avg_depth
        })
    
    # Create overview plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    frames = [s['frame'] for s in frame_stats]
    n_gts = [s['n_gt'] for s in frame_stats]
    n_dets = [s['n_det'] for s in frame_stats]
    det_rates = [s['det_rate'] for s in frame_stats]
    
    # Plot 1: GT vs Detections per frame
    axes[0, 0].plot(frames, n_gts, 'b-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot(frames, n_dets, 'r--', label='Detections', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Objects per Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Detection rate over time
    axes[0, 1].plot(frames, det_rates, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Detection Rate')
    axes[0, 1].set_title('Detection Rate Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(det_rates), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(det_rates):.2%}')
    axes[0, 1].legend()
    
    # Plot 3: Detection score distribution
    all_scores = []
    for frame_data in data:
        all_scores.extend([d['score'] for d in frame_data['detections']])
    
    axes[1, 0].hist(all_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Detection Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Detection Score Distribution')
    axes[1, 0].axvline(x=np.mean(all_scores), color='r', linestyle='--',
                       label=f'Mean: {np.mean(all_scores):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Pseudo-depth distribution
    all_depths = []
    for frame_data in data:
        depths = [compute_pseudo_depth(np.array(bbox)) 
                  for bbox in frame_data['gt_bboxes']]
        all_depths.extend(depths)
    
    axes[1, 1].hist(all_depths, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Pseudo-Depth')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Pseudo-Depth Distribution (Ground Truth)')
    axes[1, 1].axvline(x=np.mean(all_depths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(all_depths):.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats = {
        'total_frames': len(data),
        'total_gt_objects': sum(len(d['gt_bboxes']) for d in data),
        'total_detections': sum(len(d['detections']) for d in data),
        'mean_detection_rate': float(np.mean(det_rates)),
        'mean_detection_score': float(np.mean(all_scores)),
        'mean_pseudo_depth': float(np.mean(all_depths))
    }
    
    with open(output_dir / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Data statistics: {stats}")
    return stats


def compare_trackers(results, ground_truth, output_dir):
    """Generate comparison visualizations."""
    print("\nGenerating comparison visualizations...")
    
    # Extract metrics
    trackers = list(results.keys())
    metrics_names = ['MOTA', 'IDF1', 'Precision', 'Recall']
    
    # Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics_names):
        values = [results[tracker][metric] for tracker in trackers]
        
        bars = axes[idx].bar(trackers, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.1f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Detailed metrics table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Tracker', 'MOTA', 'IDF1', 'Precision', 'Recall', 'ID Switches', 'FP', 'FN']
    
    for tracker in trackers:
        r = results[tracker]
        table_data.append([
            tracker,
            f"{r['MOTA']:.2f}",
            f"{r['IDF1']:.2f}",
            f"{r['Precision']:.2f}",
            f"{r['Recall']:.2f}",
            str(r['ID_Switches']),
            str(r['FP']),
            str(r['FN'])
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15] + [0.12] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Multi-Object Tracking Performance Comparison', fontsize=14, pad=20)
    plt.savefig(output_dir / 'metrics_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ID Switches comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    id_switches = [results[tracker]['ID_Switches'] for tracker in trackers]
    bars = ax.bar(trackers, id_switches, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of ID Switches')
    ax.set_title('ID Switches Comparison (Lower is Better)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, id_switches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'id_switches_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_trajectories(data, predictions, tracker_name, output_dir, num_frames=20):
    """Visualize sample trajectories."""
    print(f"\nVisualizing {tracker_name} trajectories...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(range(0, min(len(data), num_frames * 5, 100), 5)):
        if idx >= num_frames:
            break
            
        ax = axes[idx]
        frame_data = data[frame_idx]
        frame_id = frame_data['frame']
        
        # Plot GT boxes in blue
        for bbox in frame_data['gt_bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='blue', linewidth=1.5, alpha=0.6)
            ax.add_patch(rect)
        
        # Plot tracked boxes in red
        if frame_id in predictions:
            for track_id, bbox in predictions[frame_id]:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='red', linewidth=1.5, linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1-5, str(track_id), color='red', fontsize=8)
        
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)  # Invert y-axis for image coordinates
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame_id}')
        ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', label='Ground Truth'),
        Patch(facecolor='none', edgecolor='red', linestyle='--', label='Tracked')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle(f'{tracker_name} - Sample Tracking Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'trajectories_{tracker_name.lower()}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis function."""
    # Setup paths
    workspace = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260415_154537')
    data_path = workspace / 'data' / 'simulated_sequence.json'
    output_dir = workspace / 'outputs'
    report_img_dir = workspace / 'report' / 'images'
    
    output_dir.mkdir(exist_ok=True)
    report_img_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Multi-Object Tracking Analysis")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    data = load_data(data_path)
    print(f"Loaded {len(data)} frames")
    
    # Prepare data
    ground_truth = prepare_ground_truth(data)
    detections_by_frame = prepare_detections(data)
    
    # Analyze data
    stats = analyze_data(data, output_dir)
    
    # Copy data overview to report
    import shutil
    shutil.copy(output_dir / 'data_overview.png', report_img_dir / 'data_overview.png')
    
    # Run trackers
    print("\n" + "=" * 60)
    print("Running Trackers")
    print("=" * 60)
    
    # SORT
    sort_predictions = run_tracker(SORT, 'SORT', detections_by_frame, 
                                   max_age=1, min_hits=3, iou_threshold=0.3)
    
    # ByteTrack
    bytetrack_predictions = run_tracker(ByteTrack, 'ByteTrack', detections_by_frame,
                                        track_thresh=0.5, match_thresh=0.8,
                                        second_match_thresh=0.5, track_buffer=30)
    
    # SparseTrack
    sparsetrack_predictions = run_tracker(SparseTrack, 'SparseTrack', detections_by_frame,
                                          track_thresh=0.3, match_thresh=0.7,
                                          n_depth_levels=3, track_buffer=30)
    
    # Evaluate trackers
    print("\n" + "=" * 60)
    print("Evaluating Trackers")
    print("=" * 60)
    
    results = {}
    
    for name, preds in [
        ('SORT', sort_predictions),
        ('ByteTrack', bytetrack_predictions),
        ('SparseTrack', sparsetrack_predictions)
    ]:
        metrics = compute_clear_mot_metrics(ground_truth, preds, iou_threshold=0.5)
        results[name] = metrics
        
        print(f"\n{name}:")
        print(f"  MOTA: {metrics['MOTA']:.2f}%")
        print(f"  IDF1: {metrics['IDF1']:.2f}%")
        print(f"  Precision: {metrics['Precision']:.2f}%")
        print(f"  Recall: {metrics['Recall']:.2f}%")
        print(f"  ID Switches: {metrics['ID_Switches']}")
        print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    # Save results
    with open(output_dir / 'tracking_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    compare_trackers(results, ground_truth, report_img_dir)
    
    # Visualize trajectories for each tracker
    visualize_trajectories(data, sort_predictions, 'SORT', report_img_dir)
    visualize_trajectories(data, bytetrack_predictions, 'ByteTrack', report_img_dir)
    visualize_trajectories(data, sparsetrack_predictions, 'SparseTrack', report_img_dir)
    
    # Generate depth analysis for SparseTrack
    print("\nGenerating depth analysis...")
    visualize_depth_analysis(data, report_img_dir)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {report_img_dir}")
    
    return results


def visualize_depth_analysis(data, output_dir):
    """Visualize depth-based clustering for SparseTrack."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    sample_frames = [0, 25, 50, 75, 99]
    
    for idx, frame_idx in enumerate(sample_frames):
        if idx >= 5:
            break
            
        ax = axes[idx // 3, idx % 3]
        frame_data = data[frame_idx]
        
        # Compute depths for GT boxes
        depths = [compute_pseudo_depth(np.array(bbox)) 
                  for bbox in frame_data['gt_bboxes']]
        
        # Plot boxes colored by depth
        for bbox, depth in zip(frame_data['gt_bboxes'], depths):
            x1, y1, x2, y2 = bbox
            # Normalize depth for coloring
            color = plt.cm.viridis((depth - min(depths)) / (max(depths) - min(depths) + 1e-6))
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
        
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame_idx} - Depth Levels')
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(depths), vmax=max(depths)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Pseudo-Depth (Closer objects have smaller depth)')
    
    plt.suptitle('SparseTrack: Depth-Based Object Clustering', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
