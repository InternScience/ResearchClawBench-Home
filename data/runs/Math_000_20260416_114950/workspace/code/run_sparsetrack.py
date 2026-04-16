import json
from sparsetrack import SparseTrack

def main():
    with open('data/simulated_sequence.json', 'r') as f:
        data = json.load(f)
        
    tracker = SparseTrack(track_high_thresh=0.05, track_low_thresh=0.01, new_track_thresh=0.05, match_thresh=0.8, max_time_lost=30, depth_levels=3)
    results = []
    
    for frame_data in data:
        frame_id = frame_data['frame']
        detections = frame_data['detections']
        
        active_tracks = tracker.update(detections)
        
        frame_res = []
        for track in active_tracks:
            frame_res.append({
                'track_id': track.track_id,
                'bbox': track.get_bbox().tolist()
            })
            
        results.append({
            'frame': frame_id,
            'tracks': frame_res
        })
        
    with open('outputs/sparsetrack_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
