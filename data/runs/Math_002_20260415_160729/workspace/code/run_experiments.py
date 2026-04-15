"""
Simplified experiments runner for MAPF
"""
import numpy as np
import os
import sys
import random
import json
import time
from collections import defaultdict

sys.path.insert(0, 'code')
from mapf_utils import MAPFMap, MAPFInstance, count_collisions, compute_solution_cost, makespan
from lns_marl import LNS_MARL, BaselinePP


def find_maps(data_dir, dataset, max_maps=5):
    """Find map files in a dataset."""
    maps = []
    dataset_path = os.path.join(data_dir, dataset)
    if not os.path.exists(dataset_path):
        return maps
    
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.endswith('.npy') and 'eval_map' in file:
                maps.append(os.path.join(root, file))
                if len(maps) >= max_maps:
                    break
        if len(maps) >= max_maps:
            break
    return maps


def evaluate_on_dataset(dataset_name, map_files, agent_counts, trials_per_config=3):
    """Evaluate algorithms on a dataset."""
    results = {}
    
    for num_agents in agent_counts:
        print(f"  Testing {num_agents} agents...")
        
        lns_success = []
        lns_times = []
        lns_collisions_initial = []
        lns_collisions_final = []
        
        pp_success = []
        pp_times = []
        pp_collisions = []
        
        for map_file in map_files[:trials_per_config]:
            grid = np.load(map_file, allow_pickle=True)
            map_obj = MAPFMap(grid)
            
            free_cells = len(map_obj.get_free_positions())
            if free_cells < 2 * num_agents:
                continue
            
            for trial in range(trials_per_config):
                try:
                    instance = MAPFInstance.generate_random(map_obj, num_agents, 
                                                           seed=trial * 1000 + num_agents)
                    
                    # Run LNS-MARL
                    lns_solver = LNS_MARL(instance, seed=trial)
                    lns_solution, lns_stats = lns_solver.solve(max_iterations=20, 
                                                               neighborhood_size=3,
                                                               time_limit=3.0)
                    
                    lns_success.append(1 if lns_stats['success'] else 0)
                    lns_times.append(lns_stats['time'])
                    lns_collisions_initial.append(lns_stats['initial_collisions'])
                    lns_collisions_final.append(lns_stats['final_collisions'])
                    
                    # Run Baseline PP
                    pp_solver = BaselinePP(instance)
                    pp_solution, pp_stats = pp_solver.solve()
                    
                    pp_success.append(1 if pp_stats['success'] else 0)
                    pp_times.append(pp_stats['time'])
                    pp_collisions.append(pp_stats['collisions'])
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        if lns_success:
            results[num_agents] = {
                'lns_marl': {
                    'success_rate': np.mean(lns_success),
                    'avg_time': np.mean(lns_times),
                    'avg_collisions_initial': np.mean(lns_collisions_initial),
                    'avg_collisions_final': np.mean(lns_collisions_final),
                },
                'baseline_pp': {
                    'success_rate': np.mean(pp_success),
                    'avg_time': np.mean(pp_times),
                    'avg_collisions': np.mean(pp_collisions),
                }
            }
    
    return results


def main():
    """Run experiments on selected datasets."""
    data_dir = 'data'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Select representative datasets
    datasets_config = {
        'empty': [4, 8, 12],
        'maze': [4, 6, 8],
        'random_small': [2, 4, 6],
        'random_medium': [4, 8, 12],
        'warehouse': [4, 6, 8],
    }
    
    all_results = {}
    
    for dataset_name, agent_counts in datasets_config.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        map_files = find_maps(data_dir, dataset_name, max_maps=5)
        print(f"Found {len(map_files)} maps")
        
        if not map_files:
            continue
        
        results = evaluate_on_dataset(dataset_name, map_files, agent_counts, 
                                     trials_per_config=2)
        all_results[dataset_name] = results
        
        # Print summary
        for num_agents, res in results.items():
            print(f"  {num_agents} agents:")
            print(f"    LNS-MARL: Success={res['lns_marl']['success_rate']:.2%}, "
                  f"Time={res['lns_marl']['avg_time']:.3f}s")
            print(f"    Baseline: Success={res['baseline_pp']['success_rate']:.2%}, "
                  f"Time={res['baseline_pp']['avg_time']:.3f}s")
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/results.json")
    
    return all_results


if __name__ == '__main__':
    results = main()
