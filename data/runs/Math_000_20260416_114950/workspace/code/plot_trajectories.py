import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    with open('data/simulated_sequence.json', 'r') as f:
        gt_data = json.load(f)
        
    with open('outputs/bytetrack_results.json', 'r') as f:
        byte_data = json.load(f)
        
    with open('outputs/sparsetrack_results.json', 'r') as f:
        sparse_data = json.load(f)
        
    # Plot trajectories for a few objects
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ByteTrack tracks
    byte_tracks = {}
    for frame in byte_data:
        frame_id = frame['frame']
        for track in frame['tracks']:
            track_id = track['track_id']
            bbox = track['bbox']
            if track_id not in byte_tracks:
                byte_tracks[track_id] = []
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            byte_tracks[track_id].append((frame_id, center_x, center_y))
            
    # SparseTrack tracks
    sparse_tracks = {}
    for frame in sparse_data:
        frame_id = frame['frame']
        for track in frame['tracks']:
            track_id = track['track_id']
            bbox = track['bbox']
            if track_id not in sparse_tracks:
                sparse_tracks[track_id] = []
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            sparse_tracks[track_id].append((frame_id, center_x, center_y))
            
    ax = axes[0]
    for track_id, points in byte_tracks.items():
        points = np.array(points)
        ax.plot(points[:, 1], points[:, 2], '-', alpha=0.6, label=f'ID {track_id}' if track_id < 5 else "")
    ax.set_title('ByteTrack Trajectories')
    ax.invert_yaxis()
    
    ax = axes[1]
    for track_id, points in sparse_tracks.items():
        points = np.array(points)
        ax.plot(points[:, 1], points[:, 2], '-', alpha=0.6, label=f'ID {track_id}' if track_id < 5 else "")
    ax.set_title('SparseTrack Trajectories')
    ax.invert_yaxis()
    
    fig.tight_layout()
    os.makedirs('report/images', exist_ok=True)
    plt.savefig('report/images/trajectories.png')

if __name__ == '__main__':
    main()
