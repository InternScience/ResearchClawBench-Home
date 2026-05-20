"""
Final comparison run with best parameters for both trackers.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bytetrack import ByteTrack
from sparsetrack import SparseTrack
from evaluate import evaluate_tracking, compute_iou


def load_data(data_path='data/simulated_sequence.json'):
    with open(data_path) as f:
        return json.load(f)


def run_bt(data, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2):
    tracker = ByteTrack(
        track_high_thresh=high_thresh,
        track_low_thresh=0.1,
        match_thresh=match_thresh,
        max_age=max_age,
        min_hits=min_hits,
    )
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections)
        output[frame_idx] = active_tracks
    return output


def run_st(data, n_layers=3, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2, depth_method='combined'):
    all_x2, all_y2 = [], []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
    for frame_data in data:
        for gt_bbox in frame_data['gt_bboxes']:
            all_x2.append(gt_bbox[2])
            all_y2.append(gt_bbox[3])
    fw, fh = max(all_x2) * 1.1, max(all_y2) * 1.1
    
    tracker = SparseTrack(
        num_depth_layers=n_layers,
        track_high_thresh=high_thresh,
        track_low_thresh=0.1,
        match_thresh=match_thresh,
        max_age=max_age,
        min_hits=min_hits,
        depth_method=depth_method,
    )
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections, fh, fw)
        output[frame_idx] = active_tracks
    return output


def main():
    data = load_data()
    
    print("Running final comparison...")
    
    # Best ByteTrack params
    bt_output = run_bt(data, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2)
    bt_metrics = evaluate_tracking(data, bt_output)
    
    # Best SparseTrack params
    st_output = run_st(data, n_layers=3, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2, depth_method='position')
    st_metrics = evaluate_tracking(data, st_output)
    
    print(f"ByteTrack:  MOTA={bt_metrics['MOTA']:.2f}, IDF1={bt_metrics['IDF1']:.2f}, "
          f"IDS={bt_metrics['ID_Switches']}, MT={bt_metrics['MT']}, ML={bt_metrics['ML']}")
    print(f"SparseTrack: MOTA={st_metrics['MOTA']:.2f}, IDF1={st_metrics['IDF1']:.2f}, "
          f"IDS={st_metrics['ID_Switches']}, MT={st_metrics['MT']}, ML={st_metrics['ML']}")
    
    # Run ablation: SparseTrack with different layer counts
    ablation = {}
    for nl in [1, 2, 3, 4, 5]:
        st_out = run_st(data, n_layers=nl, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2)
        st_met = evaluate_tracking(data, st_out)
        ablation[f'L{nl}'] = {
            'MOTA': round(st_met['MOTA'], 2),
            'IDF1': round(st_met['IDF1'], 2),
            'IDS': st_met['ID_Switches'],
            'MT': st_met['MT'],
            'ML': st_met['ML'],
            'FP': st_met['FP'],
            'FN': st_met['FN'],
        }
        print(f"  SparseTrack-L{nl}: MOTA={st_met['MOTA']:.2f}, IDF1={st_met['IDF1']:.2f}")
    
    # Run ablation: depth methods
    depth_ablation = {}
    for dm in ['scale', 'position', 'combined']:
        st_out = run_st(data, n_layers=3, high_thresh=0.15, match_thresh=0.2, max_age=20, min_hits=2, depth_method=dm)
        st_met = evaluate_tracking(data, st_out)
        depth_ablation[dm] = {
            'MOTA': round(st_met['MOTA'], 2),
            'IDF1': round(st_met['IDF1'], 2),
            'IDS': st_met['ID_Switches'],
        }
        print(f"  Depth method '{dm}': MOTA={st_met['MOTA']:.2f}, IDF1={st_met['IDF1']:.2f}")
    
    # Save all results
    results = {
        'ByteTrack': {
            'params': {'high_thresh': 0.15, 'low_thresh': 0.1, 'match_thresh': 0.2, 'max_age': 20, 'min_hits': 2},
            'MOTA': round(bt_metrics['MOTA'], 2),
            'IDF1': round(bt_metrics['IDF1'], 2),
            'MOTP': round(bt_metrics['MOTP'], 2),
            'ID_Switches': bt_metrics['ID_Switches'],
            'FP': bt_metrics['FP'],
            'FN': bt_metrics['FN'],
            'MT': bt_metrics['MT'],
            'ML': bt_metrics['ML'],
        },
        'SparseTrack': {
            'params': {'n_layers': 3, 'high_thresh': 0.15, 'low_thresh': 0.1, 'match_thresh': 0.2, 'max_age': 20, 'min_hits': 2, 'depth_method': 'position'},
            'MOTA': round(st_metrics['MOTA'], 2),
            'IDF1': round(st_metrics['IDF1'], 2),
            'MOTP': round(st_metrics['MOTP'], 2),
            'ID_Switches': st_metrics['ID_Switches'],
            'FP': st_metrics['FP'],
            'FN': st_metrics['FN'],
            'MT': st_metrics['MT'],
            'ML': st_metrics['ML'],
        },
        'ablation_layers': ablation,
        'ablation_depth_methods': depth_ablation,
    }
    
    with open('outputs/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFinal results saved to outputs/final_results.json")
    
    # Generate figures
    images_dir = 'report/images'
    
    # Figure 1: Main comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    ax = axes[0, 0]
    metrics_names = ['MOTA', 'IDF1', 'MOTP']
    bt_vals = [bt_metrics[n] for n in metrics_names]
    st_vals = [st_metrics[n] for n in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, bt_vals, width, label='ByteTrack', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, st_vals, width, label='SparseTrack (Ours)', color='#FF9800', alpha=0.85)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3, 
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Tracking Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    error_metrics = ['FP', 'FN', 'ID_Switches']
    bt_err = [bt_metrics[n] for n in error_metrics]
    st_err = [st_metrics[n] for n in error_metrics]
    x = np.arange(len(error_metrics))
    ax.bar(x - width/2, bt_err, width, label='ByteTrack', color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, st_err, width, label='SparseTrack (Ours)', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['False Positives', 'False Negatives', 'ID Switches'], fontsize=10)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    quality_metrics = ['MT', 'ML']
    bt_q = [bt_metrics[n] for n in quality_metrics]
    st_q = [st_metrics[n] for n in quality_metrics]
    x = np.arange(len(quality_metrics))
    ax.bar(x - width/2, bt_q, width, label='ByteTrack', color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, st_q, width, label='SparseTrack (Ours)', color='#FF9800', alpha=0.85)
    for i, (bv, sv) in enumerate(zip(bt_q, st_q)):
        ax.text(i - 0.2, bv + 0.5, str(bv), ha='center', fontsize=9, fontweight='bold')
        ax.text(i + 0.2, sv + 0.5, str(sv), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mostly Tracked', 'Mostly Lost'], fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Quality Metrics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'ByteTrack', 'SparseTrack'],
        ['MOTA (%)', f'{bt_metrics["MOTA"]:.2f}', f'{st_metrics["MOTA"]:.2f}'],
        ['IDF1 (%)', f'{bt_metrics["IDF1"]:.2f}', f'{st_metrics["IDF1"]:.2f}'],
        ['MOTP (%)', f'{bt_metrics["MOTP"]:.2f}', f'{st_metrics["MOTP"]:.2f}'],
        ['ID Switches', str(bt_metrics['ID_Switches']), str(st_metrics['ID_Switches'])],
        ['False Positives', str(bt_metrics['FP']), str(st_metrics['FP'])],
        ['False Negatives', str(bt_metrics['FN']), str(st_metrics['FN'])],
        ['Mostly Tracked', str(bt_metrics['MT']), str(st_metrics['MT'])],
        ['Mostly Lost', str(bt_metrics['ML']), str(st_metrics['ML'])],
    ]
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.32, 0.28, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for i in range(3):
        table[0, i].set_facecolor('#404040')
        table[0, i].set_text_props(color='white', fontweight='bold')
    ax.set_title('Performance Summary', fontsize=13, fontweight='bold', y=0.72)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'main_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved main_comparison.png")
    
    # Figure 2: Ablation study
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    layers = [1, 2, 3, 4, 5]
    mota_vals = [ablation[f'L{l}']['MOTA'] for l in layers]
    idf1_vals = [ablation[f'L{l}']['IDF1'] for l in layers]
    ids_vals = [ablation[f'L{l}']['IDS'] for l in layers]
    
    ax.plot(layers, mota_vals, 'o-', color='#2196F3', linewidth=2, markersize=8, label='MOTA')
    ax.plot(layers, idf1_vals, 's-', color='#FF9800', linewidth=2, markersize=8, label='IDF1')
    ax.set_xlabel('Number of Depth Layers', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Effect of Depth Layers on Tracking Performance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    ax2 = ax.twinx()
    ax2.plot(layers, ids_vals, '^--', color='#E91E63', linewidth=2, markersize=8, label='ID Switches')
    ax2.set_ylabel('ID Switches', fontsize=11, color='#E91E63')
    ax2.tick_params(axis='y', labelcolor='#E91E63')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')
    
    ax = axes[1]
    methods = list(depth_ablation.keys())
    dm_mota = [depth_ablation[m]['MOTA'] for m in methods]
    dm_idf1 = [depth_ablation[m]['IDF1'] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, dm_mota, width, label='MOTA', color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, dm_idf1, width, label='IDF1', color='#FF9800', alpha=0.85)
    for i, (mv, iv) in enumerate(zip(dm_mota, dm_idf1)):
        ax.text(i - width/2, mv + 0.3, f'{mv:.1f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(i + width/2, iv + 0.3, f'{iv:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Scale-only', 'Position-only', 'Combined'], fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Effect of Depth Estimation Method', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'ablation_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved ablation_study.png")
    
    # Figure 3: Per-frame analysis
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Per-frame accuracy
    frame_bt_acc = []
    frame_st_acc = []
    for frame_data in data:
        frame_idx = frame_data['frame']
        gt_bboxes = frame_data['gt_bboxes']
        bt_tracks = bt_output.get(frame_idx, [])
        st_tracks = st_output.get(frame_idx, [])
        
        bt_matched = sum(1 for gb in gt_bboxes 
                        if any(compute_iou(gb, t['bbox']) >= 0.5 for t in bt_tracks))
        st_matched = sum(1 for gb in gt_bboxes 
                        if any(compute_iou(gb, t['bbox']) >= 0.5 for t in st_tracks))
        n_gt = max(1, len(gt_bboxes))
        frame_bt_acc.append(bt_matched / n_gt)
        frame_st_acc.append(st_matched / n_gt)
    
    frames = [d['frame'] for d in data]
    ax = axes[0]
    ax.plot(frames, frame_bt_acc, 'b-', alpha=0.7, linewidth=1.5, label='ByteTrack')
    ax.plot(frames, frame_st_acc, 'orange', alpha=0.7, linewidth=1.5, label='SparseTrack')
    ax.fill_between(frames, frame_bt_acc, frame_st_acc, alpha=0.15, color='gray')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Frame Accuracy', fontsize=11)
    ax.set_title('Per-Frame Tracking Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Track count per frame
    bt_track_counts = [len(bt_output.get(fd['frame'], [])) for fd in data]
    st_track_counts = [len(st_output.get(fd['frame'], [])) for fd in data]
    gt_counts = [len(fd['gt_bboxes']) for fd in data]
    
    ax = axes[1]
    ax.plot(frames, gt_counts, 'k-', alpha=0.5, linewidth=1, label='GT Objects')
    ax.plot(frames, bt_track_counts, 'b-', alpha=0.7, linewidth=1.5, label='ByteTrack Tracks')
    ax.plot(frames, st_track_counts, 'orange', alpha=0.7, linewidth=1.5, label='SparseTrack Tracks')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Count per Frame', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'per_frame_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved per_frame_analysis.png")
    
    # Figure 4: Occlusion robustness analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute occlusion level per frame and accuracy
    occ_levels = []
    occ_bt_acc = []
    occ_st_acc = []
    for frame_data in data:
        gt_bboxes = frame_data['gt_bboxes']
        overlaps = []
        n = len(gt_bboxes)
        for i in range(n):
            for j in range(i+1, n):
                iou = compute_iou(gt_bboxes[i], gt_bboxes[j])
                if iou > 0:
                    overlaps.append(iou)
        occ = np.mean(overlaps) if overlaps else 0
        occ_levels.append(occ)
        
        bt_tracks = bt_output.get(frame_data['frame'], [])
        st_tracks = st_output.get(frame_data['frame'], [])
        bt_m = sum(1 for gb in gt_bboxes if any(compute_iou(gb, t['bbox']) >= 0.5 for t in bt_tracks))
        st_m = sum(1 for gb in gt_bboxes if any(compute_iou(gb, t['bbox']) >= 0.5 for t in st_tracks))
        occ_bt_acc.append(bt_m / max(1, len(gt_bboxes)))
        occ_st_acc.append(st_m / max(1, len(gt_bboxes)))
    
    ax = axes[0]
    ax.scatter(occ_levels, occ_bt_acc, alpha=0.5, c='#2196F3', label='ByteTrack', s=30)
    ax.scatter(occ_levels, occ_st_acc, alpha=0.5, c='#FF9800', label='SparseTrack', s=30)
    z_bt = np.polyfit(occ_levels, occ_bt_acc, 1)
    z_st = np.polyfit(occ_levels, occ_st_acc, 1)
    x_line = np.linspace(min(occ_levels), max(occ_levels), 100)
    ax.plot(x_line, np.polyval(z_bt, x_line), 'b--', alpha=0.7, linewidth=1.5)
    ax.plot(x_line, np.polyval(z_st, x_line), '--', color='orange', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Occlusion Level (Mean IoU)', fontsize=11)
    ax.set_ylabel('Frame Accuracy', fontsize=11)
    ax.set_title('Tracking Accuracy vs. Occlusion', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    diff = np.array(occ_st_acc) - np.array(occ_bt_acc)
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in diff]
    ax.bar(frames, diff, color=colors, alpha=0.6, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Accuracy Difference (ST - BT)', fontsize=11)
    ax.set_title('SparseTrack Advantage over ByteTrack', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add mean line
    mean_diff = np.mean(diff)
    ax.axhline(y=mean_diff, color='purple', linestyle=':', linewidth=1.5, 
              label=f'Mean diff: {mean_diff:+.4f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'occlusion_robustness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved occlusion_robustness.png")
    
    # Figure 5: Per-object coverage
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    gt_coverage_bt = defaultdict(list)
    gt_coverage_st = defaultdict(list)
    for frame_data in data:
        frame_idx = frame_data['frame']
        bt_tracks = bt_output.get(frame_idx, [])
        st_tracks = st_output.get(frame_idx, [])
        for gt_id, gt_bbox in zip(frame_data['gt_ids'], frame_data['gt_bboxes']):
            bt_cov = any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in bt_tracks)
            st_cov = any(compute_iou(gt_bbox, t['bbox']) >= 0.5 for t in st_tracks)
            gt_coverage_bt[gt_id].append(1 if bt_cov else 0)
            gt_coverage_st[gt_id].append(1 if st_cov else 0)
    
    bt_covs = [np.mean(v) for v in gt_coverage_bt.values()]
    st_covs = [np.mean(v) for v in gt_coverage_st.values()]
    
    ax = axes[0]
    ax.scatter(bt_covs, st_covs, alpha=0.5, c='purple', s=25)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('ByteTrack Object Coverage', fontsize=11)
    ax.set_ylabel('SparseTrack Object Coverage', fontsize=11)
    ax.set_title('Per-Object Coverage Comparison', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    above = sum(1 for b, s in zip(bt_covs, st_covs) if s > b)
    below = sum(1 for b, s in zip(bt_covs, st_covs) if s < b)
    ax.text(0.05, 0.95, f'SparseTrack better: {above}\nByteTrack better: {below}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    thresholds = np.arange(0, 1.05, 0.05)
    bt_above = [sum(1 for c in bt_covs if c >= t) for t in thresholds]
    st_above = [sum(1 for c in st_covs if c >= t) for t in thresholds]
    ax.plot(thresholds, bt_above, 'b-', linewidth=2, label='ByteTrack')
    ax.plot(thresholds, st_above, 'orange', linewidth=2, label='SparseTrack')
    ax.fill_between(thresholds, bt_above, st_above, alpha=0.15, color='gray')
    ax.set_xlabel('Coverage Threshold', fontsize=11)
    ax.set_ylabel('Number of Objects', fontsize=11)
    ax.set_title('Coverage Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'per_object_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved per_object_coverage.png")
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
