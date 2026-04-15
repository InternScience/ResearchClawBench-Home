"""
Run MAPF experiments comparing PP, LNS, and Hybrid MARL-LNS.
Small-scale for tractability.
"""
import sys
sys.path.insert(0, 'code')
import numpy as np
import json
import os
import time
from mapf_algorithms import (
    MAPFEnv, load_map, generate_agents, parse_agent_count,
    prioritized_planning, lns, hybrid_marl_lns, count_collisions
)

def compute_path_cost(paths):
    if paths is None:
        return float('inf')
    return sum(len(p) for p in paths)

def find_map_file(map_dir, inst):
    for f in os.listdir(map_dir):
        if f.endswith(f'_{inst}.npy') or f == f'eval_map_{inst}.npy':
            return os.path.join(map_dir, f)
    return None

def run_experiment(grid, starts, goals, time_limit=10.0):
    env = MAPFEnv(grid, starts, goals)
    results = {}
    
    # 1. Prioritized Planning
    pp_paths, pp_time = prioritized_planning(env, time_limit=time_limit)
    pp_collisions = count_collisions(pp_paths) if pp_paths else float('inf')
    pp_cost = compute_path_cost(pp_paths)
    pp_success = pp_collisions == 0 if pp_paths else False
    results['PP'] = {
        'success': pp_success, 'time': pp_time, 'cost': pp_cost,
        'collisions': pp_collisions
    }
    
    # 2. LNS
    lns_paths, lns_time, lns_collisions = lns(env, time_limit=time_limit, num_destroy=2)
    lns_cost = compute_path_cost(lns_paths)
    lns_success = lns_collisions == 0
    results['LNS'] = {
        'success': lns_success, 'time': lns_time, 'cost': lns_cost,
        'collisions': lns_collisions
    }
    
    # 3. Hybrid MARL-LNS
    hyb_paths, hyb_time, hyb_collisions, hyb_details = hybrid_marl_lns(
        env, time_limit=time_limit, marl_time_ratio=0.3, marl_episodes=10
    )
    hyb_cost = compute_path_cost(hyb_paths)
    hyb_success = hyb_collisions == 0
    results['Hybrid_MARL_LNS'] = {
        'success': hyb_success, 'time': hyb_time, 'cost': hyb_cost,
        'collisions': hyb_collisions, 'marl_collisions': hyb_details['marl_collisions']
    }
    
    return results

def main():
    os.makedirs('outputs', exist_ok=True)
    
    # Use small agent counts for tractability
    # (map_type, subdir, num_agents)
    experiments = [
        ('random_small', 'maps_50_10_10_0.175', 5),
        ('random_small', 'maps_50_10_10_0.175', 8),
        ('random_medium', 'maps_312_25_25_0.175', 8),
        ('random_medium', 'maps_312_25_25_0.175', 15),
        ('random_medium', 'maps_312_25_25_0.175', 25),
        ('maze', 'maze_maps_125_25_25', 8),
        ('maze', 'maze_maps_125_25_25', 15),
        ('maze', 'maze_maps_125_25_25', 25),
        ('room', 'room_maps_250_25_25', 8),
        ('room', 'room_maps_250_25_25', 15),
        ('room', 'room_maps_250_25_25', 25),
        ('empty', 'empty_maps_453_25_25', 8),
        ('empty', 'empty_maps_453_25_25', 15),
        ('empty', 'empty_maps_453_25_25', 25),
        ('warehouse', 'warehouse_maps_266_25_25', 8),
        ('warehouse', 'warehouse_maps_266_25_25', 15),
        ('warehouse', 'warehouse_maps_266_25_25', 25),
        ('random_large', 'maps_1250_50_50_0.175', 15),
        ('random_large', 'maps_1250_50_50_0.175', 25),
    ]
    
    all_results = []
    num_instances = 3
    time_limit = 10.0
    
    for map_type, subdir, num_agents in experiments:
        map_dir = f'data/{map_type}/{subdir}'
        if not os.path.isdir(map_dir):
            continue
        
        for inst in range(1, num_instances + 1):
            map_path = find_map_file(map_dir, inst)
            if map_path is None:
                continue
            
            grid = load_map(map_path)
            free_cells = int(np.sum(grid == 0))
            actual_agents = min(num_agents, free_cells // 2)
            if actual_agents < 2:
                continue
            
            starts, goals = generate_agents(grid, actual_agents, seed=inst * 100 + 42)
            if starts is None:
                continue
            
            print(f"{map_type} n={actual_agents} inst={inst}...", flush=True)
            
            try:
                res = run_experiment(grid, starts, goals, time_limit=time_limit)
                res['_meta'] = {
                    'map_type': map_type, 'subdir': subdir, 'instance': inst,
                    'num_agents': actual_agents, 'map_shape': list(grid.shape),
                    'obstacle_density': round(float(np.sum(grid == -1)) / grid.size, 4)
                }
                all_results.append(res)
                
                for algo in ['PP', 'LNS', 'Hybrid_MARL_LNS']:
                    r = res[algo]
                    print(f"  {algo}: s={r['success']} t={r['time']:.2f}s c={r['collisions']} cost={r['cost']}", flush=True)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
    
    with open('outputs/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nTotal: {len(all_results)}")

if __name__ == '__main__':
    main()
