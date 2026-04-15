import numpy as np
import glob
import time
from mapf_utils import load_map, generate_agents, prioritized_planning, evaluate_solution
import json

def run_pp(dataset_dir, num_trials=5):
    results = {'success': [], 'makespan': [], 'sum_ic': [], 'runtime': [], 'agents': [], 'map_path': []}
    npy_files = glob.glob(f'{dataset_dir}/**/*.npy', recursive=True)[:10]  # 10 maps
    for map_path in npy_files:
        obs, h, w = load_map(map_path)
        # Parse num_agents from dir
        subdir = map_path.split('/')[-2]
        parts = subdir.split('_')
        num_agents = int(parts[1])
        for trial in range(num_trials):
            starts, goals = generate_agents(obs, num_agents)
            if starts is None:
                continue
            t0 = time.time()
            paths = prioritized_planning(obs, starts, goals)
            dt = time.time() - t0
            success, makespan, sum_ic = evaluate_solution(paths)
            results['success'].append(success)
            results['makespan'].append(makespan)
            results['sum_ic'].append(sum_ic)
            results['runtime'].append(dt)
            results['agents'].append(num_agents)
            results['map_path'].append(map_path)
    return results

datasets = ['data/maps_60_10_10_0.175', 'data/random_small/maps_50_10_10_0.175', 'data/room/room_maps_250_25_25']
all_results = {}
for ds in datasets:
    all_results[ds] = run_pp(ds)

with open('outputs/pp_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print('PP results saved')