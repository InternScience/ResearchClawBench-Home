"""
Hyperparameter sweep and additional experiments for SparseTrack vs ByteTrack.
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bytetrack import ByteTrack
from sparsetrack import SparseTrack
from evaluate import evaluate_tracking


def load_data(data_path='data/simulated_sequence.json'):
    with open(data_path) as f:
        return json.load(f)


def run_tracker_on_data(tracker, data):
    """Run tracker and return output dict."""
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections)
        output[frame_idx] = active_tracks
    return output


def run_sparsetrack_on_data(tracker, data):
    """Run SparseTrack with frame dimensions."""
    all_x2 = []
    all_y2 = []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
    
    frame_width = max(all_x2) * 1.1
    frame_height = max(all_y2) * 1.1
    
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections, frame_height, frame_width)
        output[frame_idx] = active_tracks
    return output


def sweep_byte_track(data):
    """Sweep ByteTrack parameters."""
    results = []
    
    for high_thresh in [0.15, 0.2, 0.25]:
        for low_thresh in [0.1, 0.12, 0.15]:
            if low_thresh >= high_thresh:
                continue
            for match_thresh in [0.15, 0.2, 0.25]:
                for max_age in [20, 30, 50]:
                    for min_hits in [2, 3, 5]:
                        tracker = ByteTrack(
                            track_high_thresh=high_thresh,
                            track_low_thresh=low_thresh,
                            match_thresh=match_thresh,
                            max_age=max_age,
                            min_hits=min_hits,
                        )
                        output = run_tracker_on_data(tracker, data)
                        metrics = evaluate_tracking(data, output)
                        results.append({
                            'params': {
                                'high_thresh': high_thresh,
                                'low_thresh': low_thresh,
                                'match_thresh': match_thresh,
                                'max_age': max_age,
                                'min_hits': min_hits,
                            },
                            'MOTA': round(metrics['MOTA'], 2),
                            'IDF1': round(metrics['IDF1'], 2),
                            'IDS': metrics['ID_Switches'],
                            'MT': metrics['MT'],
                        })
    
    return results


def sweep_sparse_track(data):
    """Sweep SparseTrack parameters."""
    all_x2 = []
    all_y2 = []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
    frame_width = max(all_x2) * 1.1
    frame_height = max(all_y2) * 1.1
    
    results = []
    
    for n_layers in [2, 3, 4]:
        for high_thresh in [0.15, 0.2]:
            for match_thresh in [0.15, 0.2, 0.25]:
                for max_age in [20, 30]:
                    for min_hits in [2, 3]:
                        for depth_method in ['combined', 'scale', 'position']:
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
                                active_tracks = tracker.update(detections, frame_height, frame_width)
                                output[frame_idx] = active_tracks
                            metrics = evaluate_tracking(data, output)
                            results.append({
                                'params': {
                                    'n_layers': n_layers,
                                    'high_thresh': high_thresh,
                                    'match_thresh': match_thresh,
                                    'max_age': max_age,
                                    'min_hits': min_hits,
                                    'depth_method': depth_method,
                                },
                                'MOTA': round(metrics['MOTA'], 2),
                                'IDF1': round(metrics['IDF1'], 2),
                                'IDS': metrics['ID_Switches'],
                                'MT': metrics['MT'],
                            })
    
    return results


def main():
    data = load_data()
    
    print("Sweeping ByteTrack parameters...")
    bt_results = sweep_byte_track(data)
    bt_sorted = sorted(bt_results, key=lambda x: x['MOTA'], reverse=True)
    
    print("\nTop 5 ByteTrack configurations:")
    for i, r in enumerate(bt_sorted[:5]):
        print(f"  {i+1}. MOTA={r['MOTA']:.2f}, IDF1={r['IDF1']:.2f}, IDS={r['IDS']}, "
              f"params={r['params']}")
    
    print("\nSweeping SparseTrack parameters...")
    st_results = sweep_sparse_track(data)
    st_sorted = sorted(st_results, key=lambda x: x['MOTA'], reverse=True)
    
    print("\nTop 5 SparseTrack configurations:")
    for i, r in enumerate(st_sorted[:5]):
        print(f"  {i+1}. MOTA={r['MOTA']:.2f}, IDF1={r['IDF1']:.2f}, IDS={r['IDS']}, "
              f"params={r['params']}")
    
    # Save results
    with open('outputs/sweep_bt.json', 'w') as f:
        json.dump(bt_sorted[:20], f, indent=2)
    with open('outputs/sweep_st.json', 'w') as f:
        json.dump(st_sorted[:20], f, indent=2)
    
    print("\nResults saved to outputs/sweep_bt.json and outputs/sweep_st.json")


if __name__ == '__main__':
    main()
