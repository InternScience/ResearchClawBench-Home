import numpy as np
import glob
import json
import os
from collections import defaultdict

def parse_agent_num(subdir):
    if '_maps_' in subdir:
        parts = subdir.split('_maps_')[1].split('_')
        return int(parts[0])
    return 50

def compute_dataset_stats():
    stats = defaultdict(lambda: {'num_maps': 0, 'grid_sizes': set(), 'obs_densities': [], 'agent_counts': []})
    data_dir = 'data'
    for dataset_dir in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        npy_files = glob.glob(os.path.join(dataset_path, '**/*.npy'), recursive=True)
        stats[dataset_dir]['num_maps'] = len(npy_files)
        sample_maps = npy_files[:10]
        densities = []
        sizes = set()
        agents_per_map = []
        for f in sample_maps:
            arr = np.load(f)
            h, w = arr.shape
            sizes.add((h, w))
            density = np.mean(arr == -1)
            densities.append(density)
            subdir = os.path.dirname(f).split('/')[-1]
            agents = parse_agent_num(subdir)
            agents_per_map.append(agents)
        stats[dataset_dir]['grid_sizes'] = list(sizes)
        stats[dataset_dir]['obs_densities'] = densities
        stats[dataset_dir]['agent_counts'] = agents_per_map
    return stats

stats = compute_dataset_stats()
with open('outputs/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Updated stats saved')