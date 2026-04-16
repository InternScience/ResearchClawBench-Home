"""
Run experiments comparing SORT, ByteTrack, and SparseTrack on the simulated sequence.
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tracker import SORTTracker, ByteTrackTracker, SparseTrackTracker, run_tracker, compute_mot_metrics

def main():
    # Load data
    with open('../data/simulated_sequence.json', 'r') as f:
        sequence_data = json.load(f)

    print(f"Loaded {len(sequence_data)} frames")

    # Define tracker configurations
    # Key insight: score threshold must be adapted to data
    # Most scores are < 0.4, so use 0.2 as high-score threshold
    trackers = {
        'SORT': {
            'class': SORTTracker,
            'kwargs': {
                'iou_threshold': 0.3,
                'max_age': 30,
                'min_hits': 1,
                'score_threshold': 0.1
            }
        },
        'ByteTrack': {
            'class': ByteTrackTracker,
            'kwargs': {
                'score_threshold': 0.2,
                'low_score_threshold': 0.1,
                'iou_threshold': 0.3,
                'max_age': 30,
                'min_hits': 1
            }
        },
        'SparseTrack_L3': {
            'class': SparseTrackTracker,
            'kwargs': {
                'n_depth_layers': 3,
                'score_threshold': 0.2,
                'low_score_threshold': 0.1,
                'iou_threshold': 0.3,
                'max_age': 30,
                'min_hits': 1,
                'overlap_threshold': 0.3
            }
        },
        'SparseTrack_L5': {
            'class': SparseTrackTracker,
            'kwargs': {
                'n_depth_layers': 5,
                'score_threshold': 0.2,
                'low_score_threshold': 0.1,
                'iou_threshold': 0.3,
                'max_age': 30,
                'min_hits': 1,
                'overlap_threshold': 0.3
            }
        }
    }

    results = {}
    all_trajectories = {}

    for name, config in trackers.items():
        print(f"\nRunning {name}...")
        trajectories = run_tracker(config['class'], config['kwargs'], sequence_data)
        metrics = compute_mot_metrics(trajectories, sequence_data, iou_threshold=0.5)
        
        results[name] = metrics
        all_trajectories[name] = {str(k): v for k, v in trajectories.items()}
        
        print(f"  MOTA: {metrics['MOTA']:.4f}")
        print(f"  MOTP: {metrics['MOTP']:.4f}")
        print(f"  IDF1: {metrics['IDF1']:.4f}")
        print(f"  ID Switches: {metrics['ID_Switches']}")
        print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}, TP: {metrics['TP']}")
        print(f"  Mostly Tracked: {metrics['Mostly_Tracked']}")
        print(f"  Mostly Lost: {metrics['Mostly_Lost']}")
        print(f"  Num trajectories: {len(trajectories)}")

    # Save results
    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/metrics_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics to outputs/metrics_comparison.json")

    # Save trajectories (compact format)
    os.makedirs('../outputs/trajectories', exist_ok=True)
    for name, trajs in all_trajectories.items():
        serializable = {}
        for tid, frames in trajs.items():
            serializable[tid] = [(f, [round(b, 2) for b in bbox]) for f, bbox in frames]
        with open(f'../outputs/trajectories/{name}_trajectories.json', 'w') as f:
            json.dump(serializable, f)
    print("Saved trajectories to outputs/trajectories/")

    # Run ablation on depth layers for SparseTrack
    print("\n\n=== Ablation: Number of Depth Layers ===")
    ablation_results = {}
    for n_layers in [1, 2, 3, 4, 5, 7, 10]:
        name = f'SparseTrack_L{n_layers}'
        kwargs = {
            'n_depth_layers': n_layers,
            'score_threshold': 0.2,
            'low_score_threshold': 0.1,
            'iou_threshold': 0.3,
            'max_age': 30,
            'min_hits': 1,
            'overlap_threshold': 0.3
        }
        trajectories = run_tracker(SparseTrackTracker, kwargs, sequence_data)
        metrics = compute_mot_metrics(trajectories, sequence_data, iou_threshold=0.5)
        ablation_results[n_layers] = metrics
        print(f"  L{n_layers}: MOTA={metrics['MOTA']:.4f}, IDF1={metrics['IDF1']:.4f}, IDs={metrics['ID_Switches']}, MT={metrics['Mostly_Tracked']}")

    with open('../outputs/ablation_depth_layers.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("Saved ablation results to outputs/ablation_depth_layers.json")

    # Run ablation on score thresholds for ByteTrack
    print("\n\n=== Ablation: Score Threshold (ByteTrack) ===")
    score_ablation = {}
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        kwargs = {
            'score_threshold': thresh,
            'low_score_threshold': 0.1,
            'iou_threshold': 0.3,
            'max_age': 30,
            'min_hits': 1
        }
        trajectories = run_tracker(ByteTrackTracker, kwargs, sequence_data)
        metrics = compute_mot_metrics(trajectories, sequence_data, iou_threshold=0.5)
        score_ablation[thresh] = metrics
        print(f"  t={thresh}: MOTA={metrics['MOTA']:.4f}, IDF1={metrics['IDF1']:.4f}, IDs={metrics['ID_Switches']}, MT={metrics['Mostly_Tracked']}")

    with open('../outputs/ablation_score_threshold.json', 'w') as f:
        json.dump(score_ablation, f, indent=2)
    print("Saved score ablation to outputs/ablation_score_threshold.json")

    # Run ablation on overlap threshold for SparseTrack
    print("\n\n=== Ablation: Overlap Threshold (SparseTrack L3) ===")
    overlap_ablation = {}
    for overlap_t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        kwargs = {
            'n_depth_layers': 3,
            'score_threshold': 0.2,
            'low_score_threshold': 0.1,
            'iou_threshold': 0.3,
            'max_age': 30,
            'min_hits': 1,
            'overlap_threshold': overlap_t
        }
        trajectories = run_tracker(SparseTrackTracker, kwargs, sequence_data)
        metrics = compute_mot_metrics(trajectories, sequence_data, iou_threshold=0.5)
        overlap_ablation[overlap_t] = metrics
        print(f"  overlap_t={overlap_t}: MOTA={metrics['MOTA']:.4f}, IDF1={metrics['IDF1']:.4f}, IDs={metrics['ID_Switches']}, MT={metrics['Mostly_Tracked']}")

    with open('../outputs/ablation_overlap_threshold.json', 'w') as f:
        json.dump(overlap_ablation, f, indent=2)
    print("Saved overlap ablation to outputs/ablation_overlap_threshold.json")

    # Run IoU threshold ablation
    print("\n\n=== Ablation: IoU Threshold ===")
    iou_ablation = {}
    for iou_t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        # SORT
        kwargs_sort = {'iou_threshold': iou_t, 'max_age': 30, 'min_hits': 1, 'score_threshold': 0.1}
        traj = run_tracker(SORTTracker, kwargs_sort, sequence_data)
        m_sort = compute_mot_metrics(traj, sequence_data, iou_threshold=0.5)
        
        # ByteTrack
        kwargs_bt = {'score_threshold': 0.2, 'low_score_threshold': 0.1, 'iou_threshold': iou_t, 'max_age': 30, 'min_hits': 1}
        traj = run_tracker(ByteTrackTracker, kwargs_bt, sequence_data)
        m_bt = compute_mot_metrics(traj, sequence_data, iou_threshold=0.5)
        
        # SparseTrack
        kwargs_st = {'n_depth_layers': 3, 'score_threshold': 0.2, 'low_score_threshold': 0.1, 'iou_threshold': iou_t, 'max_age': 30, 'min_hits': 1, 'overlap_threshold': 0.3}
        traj = run_tracker(SparseTrackTracker, kwargs_st, sequence_data)
        m_st = compute_mot_metrics(traj, sequence_data, iou_threshold=0.5)
        
        iou_ablation[str(iou_t)] = {'SORT': m_sort, 'ByteTrack': m_bt, 'SparseTrack': m_st}
        print(f"  iou_t={iou_t}: SORT MOTA={m_sort['MOTA']:.4f}, BT MOTA={m_bt['MOTA']:.4f}, ST MOTA={m_st['MOTA']:.4f}")

    with open('../outputs/ablation_iou_threshold.json', 'w') as f:
        json.dump(iou_ablation, f, indent=2)
    print("Saved IoU ablation to outputs/ablation_iou_threshold.json")


if __name__ == '__main__':
    main()
