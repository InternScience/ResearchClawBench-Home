"""
Quick targeted sweep for SparseTrack only.
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparsetrack import SparseTrack
from evaluate import evaluate_tracking


def load_data(data_path='data/simulated_sequence.json'):
    with open(data_path) as f:
        return json.load(f)


def run_st(data, n_layers, high_thresh, match_thresh, max_age, min_hits, depth_method):
    all_x2, all_y2 = [], []
    for frame_data in data:
        for det in frame_data['detections']:
            all_x2.append(det['bbox'][2])
            all_y2.append(det['bbox'][3])
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
    
    metrics = evaluate_tracking(data, output)
    return metrics


def main():
    data = load_data()
    
    print("Quick SparseTrack sweep...")
    st_best = None
    st_best_mota = -999
    count = 0
    
    # Smaller sweep
    for n_layers in [2, 3, 4]:
        for high_thresh in [0.15, 0.2]:
            for match_thresh in [0.2]:
                for max_age in [20, 30]:
                    for min_hits in [2, 3]:
                        for depth_method in ['combined', 'scale']:
                            count += 1
                            metrics = run_st(data, n_layers, high_thresh, match_thresh, 
                                           max_age, min_hits, depth_method)
                            mota = metrics['MOTA']
                            if mota > st_best_mota:
                                st_best_mota = mota
                                st_best = {
                                    'params': {'n_layers': n_layers, 'high_thresh': high_thresh,
                                              'match_thresh': match_thresh, 'max_age': max_age,
                                              'min_hits': min_hits, 'depth_method': depth_method},
                                    'MOTA': round(mota, 2),
                                    'IDF1': round(metrics['IDF1'], 2),
                                    'IDS': metrics['ID_Switches'],
                                    'MT': metrics['MT'],
                                    'FP': metrics['FP'],
                                    'FN': metrics['FN'],
                                }
                            if count % 10 == 0:
                                print(f"  {count} configs tested, best MOTA={st_best['MOTA']:.2f}")
    
    print(f"\nBest SparseTrack: {st_best}")
    
    # Also test depth_method='position'
    metrics = run_st(data, st_best['params']['n_layers'], st_best['params']['high_thresh'],
                    st_best['params']['match_thresh'], st_best['params']['max_age'],
                    st_best['params']['min_hits'], 'position')
    print(f"Position method: MOTA={metrics['MOTA']:.2f}, IDF1={metrics['IDF1']:.2f}")
    
    # Save
    with open('outputs/best_sparsetrack.json', 'w') as f:
        json.dump(st_best, f, indent=2)
    print("Saved to outputs/best_sparsetrack.json")


if __name__ == '__main__':
    main()
