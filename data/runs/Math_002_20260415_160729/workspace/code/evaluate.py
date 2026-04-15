"""
Evaluation script for MAPF algorithms
"""
import numpy as np
import json
import os
import sys
import time
from typing import List, Dict, Tuple
import pickle
from collections import defaultdict

from mapf_utils import MAPFMap, MAPFInstance, count_collisions, compute_solution_cost, makespan
from lns_marl import LNS_MARL, BaselinePP


def load_maps(data_dir: str) -> Dict[str, List[np.ndarray]]:
    """Load all maps from data directory."""
    datasets = {}
    
    for dataset_name in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.isdir(dataset_path):
            maps = []
            # Find all .npy files recursively
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.npy'):
                        map_path = os.path.join(root, file)
                        try:
                            grid = np.load(map_path, allow_pickle=True)
                            maps.append(grid)
                        except Exception as e:
                            print(f"Error loading {map_path}: {e}")
            
            if maps:
                datasets[dataset_name] = maps
                print(f"Loaded {len(maps)} maps from {dataset_name}")
    
    return datasets


def evaluate_algorithm(algorithm_name: str, algorithm_fn, 
                       instances: List[MAPFInstance], seed: int = 42) -> Dict:
    """Evaluate an algorithm on a set of instances."""
    results = []
    
    for i, instance in enumerate(instances):
        try:
            solution, stats = algorithm_fn(instance, seed + i)
            results.append(stats)
        except Exception as e:
            print(f"Error on instance {i}: {e}")
            results.append({
                'success': False,
                'time': 0,
                'collisions': float('inf'),
                'cost': float('inf')
            })
    
    # Aggregate statistics
    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    avg_time = np.mean([r['time'] for r in results])
    avg_collisions = np.mean([r.get('final_collisions', r.get('collisions', 0)) for r in results])
    avg_cost = np.mean([r.get('final_cost', r.get('cost', 0)) for r in results])
    
    return {
        'algorithm': algorithm_name,
        'success_rate': success_rate,
        'avg_time': avg_time,
        'avg_collisions': avg_collisions,
        'avg_cost': avg_cost,
        'results': results
    }


def run_lns_marl(instance: MAPFInstance, seed: int) -> Tuple:
    """Run LNS-MARL algorithm."""
    solver = LNS_MARL(instance, seed=seed)
    return solver.solve(max_iterations=50, neighborhood_size=5, time_limit=10.0)


def run_baseline_pp(instance: MAPFInstance, seed: int) -> Tuple:
    """Run baseline Prioritized Planning."""
    solver = BaselinePP(instance)
    return solver.solve()


def create_test_instances(datasets: Dict[str, List[np.ndarray]], 
                          agent_counts: List[int],
                          instances_per_config: int = 5) -> Dict:
    """Create test instances for different configurations."""
    instances = defaultdict(lambda: defaultdict(list))
    
    for dataset_name, maps in datasets.items():
        for num_agents in agent_counts:
            for i in range(min(instances_per_config, len(maps))):
                map_obj = MAPFMap(maps[i])
                try:
                    instance = MAPFInstance.generate_random(map_obj, num_agents, seed=i * 1000 + num_agents)
                    instances[dataset_name][num_agents].append(instance)
                except ValueError as e:
                    print(f"Could not create instance for {dataset_name} with {num_agents} agents: {e}")
    
    return instances


def run_experiments(data_dir: str, output_dir: str):
    """Run full experimental evaluation."""
    print("Loading maps...")
    datasets = load_maps(data_dir)
    
    # Determine agent counts based on map sizes
    agent_configs = {
        'empty': [4, 8, 12, 16],
        'maze': [4, 8, 12, 16],
        'random_small': [2, 4, 6],
        'random_medium': [4, 8, 12, 16],
        'random_large': [8, 16, 24, 32],
        'room': [4, 8, 12, 16],
        'warehouse': [4, 8, 12, 16],
        'maps_60_10_10_0.175': [4, 8, 12]
    }
    
    all_results = {}
    
    for dataset_name, maps in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        agent_counts = agent_configs.get(dataset_name, [4, 8, 12])
        dataset_results = {}
        
        for num_agents in agent_counts:
            print(f"\n--- Testing with {num_agents} agents ---")
            
            instances = []
            for i, map_grid in enumerate(maps[:10]):  # Use up to 10 maps
                map_obj = MAPFMap(map_grid)
                free_cells = len(map_obj.get_free_positions())
                
                if free_cells < 2 * num_agents:
                    continue
                
                try:
                    instance = MAPFInstance.generate_random(map_obj, num_agents, seed=i * 1000)
                    instances.append(instance)
                except ValueError:
                    continue
            
            if not instances:
                print(f"No valid instances for {num_agents} agents")
                continue
            
            print(f"Created {len(instances)} instances")
            
            # Evaluate LNS-MARL
            print("Running LNS-MARL...")
            lns_results = evaluate_algorithm('LNS-MARL', run_lns_marl, instances)
            
            # Evaluate Baseline PP
            print("Running Baseline PP...")
            pp_results = evaluate_algorithm('Baseline-PP', run_baseline_pp, instances)
            
            dataset_results[num_agents] = {
                'lns_marl': lns_results,
                'baseline_pp': pp_results
            }
            
            print(f"LNS-MARL: Success Rate = {lns_results['success_rate']:.2%}, "
                  f"Avg Time = {lns_results['avg_time']:.3f}s, "
                  f"Avg Collisions = {lns_results['avg_collisions']:.2f}")
            print(f"Baseline-PP: Success Rate = {pp_results['success_rate']:.2%}, "
                  f"Avg Time = {pp_results['avg_time']:.3f}s, "
                  f"Avg Collisions = {pp_results['avg_collisions']:.2f}")
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON (without the detailed results list for brevity)
    json_results = {}
    for dataset, agents_data in all_results.items():
        json_results[dataset] = {}
        for agent_count, algos in agents_data.items():
            json_results[dataset][agent_count] = {
                'lns_marl': {k: v for k, v in algos['lns_marl'].items() if k != 'results'},
                'baseline_pp': {k: v for k, v in algos['baseline_pp'].items() if k != 'results'}
            }
    
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save full results as pickle
    with open(os.path.join(output_dir, 'results_full.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to {output_dir}")
    
    return all_results


if __name__ == '__main__':
    data_dir = 'data'
    output_dir = 'outputs'
    
    results = run_experiments(data_dir, output_dir)
