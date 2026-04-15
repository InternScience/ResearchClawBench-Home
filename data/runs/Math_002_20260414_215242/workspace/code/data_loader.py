import numpy as np
import glob
import json
import os
from collections import defaultdict

def compute_dataset_stats():
    stats = defaultdict(lambda: {'num_maps': 0, 'grid_sizes': [], 'obs_densities': [], 'agent_counts': []})
    data_dir = 'data'
    for dataset_dir in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        npy_files = glob.glob(os.path.join(dataset_path, '**/*.npy'), recursive=True)
        stats[dataset_dir]['num_maps'] = len(npy_files)
        sample_maps = npy_files[:10]  # sample 10
        for f in sample_maps:
            arr = np.load(f)
            # Assume arr is obstacle map, 0 free, 1 wall
            h, w = arr.shape
            stats[dataset_dir]['grid_sizes'].append((h, w))
            obs_density = np.mean(arr == 1)
            stats[dataset_dir]['obs_densities'].append(obs_density)
        # Agent count from dir name pattern, e.g. empty_maps_453_25_25 -> 453 agents
        if 'maps_' in dataset_dir or 'empty_maps_' in dataset_dir or 'maze_maps_' in dataset_dir:
            parts = dataset_dir.split('_')
            try:
                agent_str = parts[1] if dataset_dir.startswith('empty') or dataset_dir.startswith('maze') else parts[1]
                agents = int(agent_str)
                stats[dataset_dir]['agent_counts'] = [agents] * stats[dataset_dir]['num_maps']
            except:
                stats[dataset_dir]['agent_counts'] = [50] * stats[dataset_dir]['num_maps']  # default
    return stats

if __name__ == '__main__':
    stats = compute_dataset_stats()
    with open('outputs/data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print('Stats saved')