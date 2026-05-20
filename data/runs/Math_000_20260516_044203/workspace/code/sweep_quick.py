"""
Quick parameter sweep for ByteTrack and SparseTrack.
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
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections)
        output[frame_idx] = active_tracks
    return output


def run_sparsetrack_on_data(tracker, data):
    all_x2, all_y2 = [], []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
    fw, fh = max(all_x2) * 1.1, max(all_y2) * 1.1
    
    output = {}
    for frame_data in data:
        frame_idx = frame_data['frame']
        detections = frame_data['detections']
        active_tracks = tracker.update(detections, fh, fw)
        output[frame_idx] = active_tracks
    return output


def main():
    data = load_data()
    
    # Quick ByteTrack sweep
    print("ByteTrack sweep...")
    bt_best = None
    bt_best_mota = -999
    
    for high_thresh in [0.15, 0.2, 0.25]:
        for match_thresh in [0.15, 0.2]:
            for max_age in [20, 30, 50]:
                for min_hits in [2, 3]:
                    tracker = ByteTrack(
                        track_high_thresh=high_thresh,
                        track_low_thresh=0.1,
                        match_thresh=match_thresh,
                        max_age=max_age,
                        min_hits=min_hits,
                    )
                    output = run_tracker_on_data(tracker, data)
                    metrics = evaluate_tracking(data, output)
                    mota = metrics['MOTA']
                    if mota > bt_best_mota:
                        bt_best_mota = mota
                        bt_best = {
                            'params': {'high_thresh': high_thresh, 'match_thresh': match_thresh,
                                      'max_age': max_age, 'min_hits': min_hits},
                            'MOTA': round(metrics['MOTA'], 2),
                            'IDF1': round(metrics['IDF1'], 2),
                            'IDS': metrics['ID_Switches'],
                            'MT': metrics['MT'],
                            'FP': metrics['FP'],
                            'FN': metrics['FN'],
                        }
    
    print(f"Best ByteTrack: {bt_best}")
    
    # Quick SparseTrack sweep
    print("\nSparseTrack sweep...")
    st_best = None
    st_best_mota = -999
    
    for n_layers in [2, 3, 4]:
        for high_thresh in [0.15, 0.2]:
            for match_thresh in [0.15, 0.2]:
                for max_age in [20, 30]:
                    for min_hits in [2, 3]:
                        for depth_method in ['combined', 'scale']:
                            tracker = SparseTrack(
                                num_depth_layers=n_layers,
                                track_high_thresh=high_thresh,
                                track_low_thresh=0.1,
                                match_thresh=match_thresh,
                                max_age=max_age,
                                min_hits=min_hits,
                                depth_method=depth_method,
                            )
                            output = run_sparsetrack_on_data(tracker, data)
                            metrics = evaluate_tracking(data, output)
                            mota = metrics['MOTA']
                            if mota > st_best_mota:
                                st_best_mota = mota
                                st_best = {
                                    'params': {'n_layers': n_layers, 'high_thresh': high_thresh,
                                              'match_thresh': match_thresh, 'max_age': max_age,
                                              'min_hits': min_hits, 'depth_method': depth_method},
                                    'MOTA': round(metrics['MOTA'], 2),
                                    'IDF1': round(metrics['IDF1'], 2),
                                    'IDS': metrics['ID_Switches'],
                                    'MT': metrics['MT'],
                                    'FP': metrics['FP'],
                                    'FN': metrics['FN'],
                                }
    
    print(f"Best SparseTrack: {st_best}")
    
    # Run final evaluation with best params
    print("\n--- Final Evaluation ---")
    
    # Best ByteTrack
    bp = bt_best['params']
    bt_tracker = ByteTrack(
        track_high_thresh=bp['high_thresh'],
        track_low_thresh=0.1,
        match_thresh=bp['match_thresh'],
        max_age=bp['max_age'],
        min_hits=bp['min_hits'],
    )
    bt_output = run_tracker_on_data(bt_tracker, data)
    bt_metrics = evaluate_tracking(data, bt_output)
    print(f"ByteTrack: MOTA={bt_metrics['MOTA']:.2f}, IDF1={bt_metrics['IDF1']:.2f}, "
          f"IDS={bt_metrics['ID_Switches']}, MT={bt_metrics['MT']}")
    
    # Best SparseTrack
    sp = st_best['params']
    st_tracker = SparseTrack(
        num_depth_layers=sp['n_layers'],
        track_high_thresh=sp['high_thresh'],
        track_low_thresh=0.1,
        match_thresh=sp['match_thresh'],
        max_age=sp['max_age'],
        min_hits=sp['min_hits'],
        depth_method=sp['depth_method'],
    )
    st_output = run_sparsetrack_on_data(st_tracker, data)
    st_metrics = evaluate_tracking(data, st_output)
    print(f"SparseTrack: MOTA={st_metrics['MOTA']:.2f}, IDF1={st_metrics['IDF1']:.2f}, "
          f"IDS={st_metrics['ID_Switches']}, MT={st_metrics['MT']}")
    
    # Save final metrics
    final_results = {
        'ByteTrack': {
            'params': bp,
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
            'params': sp,
            'MOTA': round(st_metrics['MOTA'], 2),
            'IDF1': round(st_metrics['IDF1'], 2),
            'MOTP': round(st_metrics['MOTP'], 2),
            'ID_Switches': st_metrics['ID_Switches'],
            'FP': st_metrics['FP'],
            'FN': st_metrics['FN'],
            'MT': st_metrics['MT'],
            'ML': st_metrics['ML'],
        },
    }
    
    with open('outputs/final_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nFinal metrics saved to outputs/final_metrics.json")


if __name__ == '__main__':
    main()
