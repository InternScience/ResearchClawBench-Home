#!/usr/bin/env python3
"""
Quick experiment runner for MAPF solvers.
Evaluates different algorithms across selected dataset types.
"""

import numpy as np
import os
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Import solvers
from mapf_solver import (
    MAPFInstance, Agent, CollisionDetector,
    PrioritizedPlanning, MARLGuidedPlanner, HybridMARLLNSPP
)

WORKSPACE_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559"
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
OUTPUTS_ROOT = os.path.join(WORKSPACE_ROOT, "outputs")

def generate_agents_for_map(grid: np.ndarray, num_agents: int, 
                            seed: int = None) -> List[Agent]:
    """Generate agent start/goal positions for a grid map."""
    if seed is not None:
        random.seed(seed)
    
    height, width = grid.shape
    free_cells = list(zip(*np.where(grid == 0)))
    
    if len(free_cells) < num_agents * 2:
        num_agents = len(free_cells) // 2
    
    if num_agents == 0:
        return []
    
    selected_indices = random.sample(range(len(free_cells)), num_agents)
    starts = [free_cells[i] for i in selected_indices]
    
    remaining = [i for i in range(len(free_cells)) if i not in selected_indices]
    if len(remaining) < num_agents:
        goals = [free_cells[i % len(free_cells)] for i in range(num_agents)]
    else:
        goal_indices = random.sample(remaining, num_agents)
        goals = [free_cells[i] for i in goal_indices]
    
    agents = [
        Agent(id=i, start=starts[i], goal=goals[i])
        for i in range(num_agents)
    ]
    
    return agents

def compute_path_metrics(paths, agents):
    """Compute solution quality metrics."""
    if not paths:
        return {'sum_of_costs': float('inf'), 'makespan': 0, 'avg_path_length': 0}
    
    path_lengths = [len(p) - 1 for p in paths]
    sum_of_costs = sum(path_lengths)
    makespan = max(path_lengths)
    avg_path_length = np.mean(path_lengths)
    
    return {
        'sum_of_costs': sum_of_costs,
        'makespan': makespan,
        'avg_path_length': avg_path_length
    }

def evaluate_solver(solver_class, instance, solver_name, time_limit=15.0, num_restarts=1):
    """Evaluate a solver on a single instance."""
    results = {
        'solver': solver_name,
        'success': False,
        'runtime': float('inf'),
        'collisions': float('inf'),
        'sum_of_costs': float('inf'),
        'makespan': 0
    }
    
    best_result = None
    
    for restart in range(num_restarts):
        try:
            if solver_class == 'pp':
                solver = PrioritizedPlanning(instance)
                priority_order = list(range(len(instance.agents)))
                if restart > 0:
                    random.shuffle(priority_order)
                paths, runtime = solver.solve(priority_order)
                stats = {'restarts': restart + 1}
            elif solver_class == 'marl':
                solver = MARLGuidedPlanner(instance)
                paths, runtime = solver.generate_initial_solution()
                stats = {}
            elif solver_class == 'hybrid':
                solver = HybridMARLLNSPP(instance)
                paths, runtime, stats = solver.solve(time_limit=time_limit)
            else:
                paths, runtime, stats = None, float('inf'), {}
            
            if paths is not None:
                collisions = CollisionDetector.detect_collisions(paths)
                num_collisions = len(collisions)
                metrics = compute_path_metrics(paths, instance.agents)
                
                if num_collisions == 0:
                    results['success'] = True
                    results['runtime'] = runtime
                    results['collisions'] = 0
                    results['sum_of_costs'] = metrics['sum_of_costs']
                    results['makespan'] = metrics['makespan']
                    results['stats'] = stats
                    best_result = results.copy()
                    break
                else:
                    if best_result is None or num_collisions < best_result.get('collisions', float('inf')):
                        best_result = {
                            'solver': solver_name,
                            'success': False,
                            'runtime': runtime,
                            'collisions': num_collisions,
                            'sum_of_costs': metrics['sum_of_costs'],
                            'makespan': metrics['makespan'],
                            'stats': stats
                        }
        except Exception as e:
            continue
    
    if best_result:
        return best_result
    return results

def run_dataset_experiments(dataset_name, dataset_path, num_agents_list=[5, 10],
                            instances_per_config=3, time_limit=15.0):
    """Run experiments on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    map_files = []
    
    def collect_maps(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path) and item.endswith('.npy'):
                map_files.append(item_path)
            elif os.path.isdir(item_path):
                collect_maps(item_path)
    
    collect_maps(dataset_path)
    
    if not map_files:
        print(f"  No maps found in {dataset_path}")
        return []
    
    print(f"  Found {len(map_files)} map files")
    print(f"  Testing with {num_agents_list} agents, {instances_per_config} instances each")
    
    all_results = []
    solvers = ['pp', 'marl', 'hybrid']
    
    for num_agents in num_agents_list:
        print(f"\n  --- {num_agents} agents ---")
        
        for instance_idx in range(instances_per_config):
            map_file = random.choice(map_files)
            grid = np.load(map_file, allow_pickle=True)
            
            agents = generate_agents_for_map(grid, num_agents, seed=instance_idx)
            
            if len(agents) < num_agents:
                continue
            
            instance = MAPFInstance(grid=grid, agents=agents, 
                                   height=grid.shape[0], width=grid.shape[1])
            
            for solver in solvers:
                result = evaluate_solver(solver, instance, solver, 
                                        time_limit=time_limit, num_restarts=1)
                result['dataset'] = dataset_name
                result['map_file'] = os.path.basename(map_file)
                result['num_agents'] = num_agents
                result['instance_idx'] = instance_idx
                result['map_size'] = f"{grid.shape[0]}x{grid.shape[1]}"
                
                all_results.append(result)
                
                if instance_idx == 0:
                    status = "Y" if result['success'] else "N"
                    print(f"    {solver:8s}: {status}  t={result['runtime']:.3f}s  "
                          f"c={result['collisions']}  soc={result['sum_of_costs']}")
    
    return all_results

def aggregate_results(all_results):
    """Aggregate experiment results into summary statistics."""
    summary = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        key = (result['dataset'], result['num_agents'], result['solver'])
        summary[key]['success'].append(1 if result['success'] else 0)
        summary[key]['runtime'].append(result['runtime'])
        summary[key]['collisions'].append(result['collisions'])
        summary[key]['sum_of_costs'].append(result['sum_of_costs'])
    
    aggregated = {}
    for key, metrics in summary.items():
        dataset, num_agents, solver = key
        agg_key = f"{dataset}_{num_agents}_{solver}"
        
        aggregated[agg_key] = {
            'dataset': dataset,
            'num_agents': num_agents,
            'solver': solver,
            'success_rate': float(np.mean(metrics['success'])),
            'avg_runtime': float(np.mean([r for r in metrics['runtime'] if r < float('inf')])) if any(r < float('inf') for r in metrics['runtime']) else float('inf'),
            'avg_collisions': float(np.mean(metrics['collisions'])),
            'avg_sum_of_costs': float(np.mean([c for c in metrics['sum_of_costs'] if c < float('inf')])),
            'num_instances': len(metrics['success'])
        }
    
    return aggregated

def main():
    print("=" * 70)
    print("MAPF Hybrid Algorithm Experiments")
    print("=" * 70)
    
    datasets = {
        'empty': {'path': 'data/empty', 'agents': [10, 20]},
        'random_small': {'path': 'data/random_small', 'agents': [5, 10]},
        'random_medium': {'path': 'data/random_medium', 'agents': [10, 20]},
        'room': {'path': 'data/room', 'agents': [10, 20]},
        'warehouse': {'path': 'data/warehouse', 'agents': [10, 20]},
    }
    
    all_results = []
    
    for dataset_name, config in datasets.items():
        dataset_path = os.path.join(WORKSPACE_ROOT, config['path'])
        if os.path.exists(dataset_path):
            results = run_dataset_experiments(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                num_agents_list=config['agents'],
                instances_per_config=3,
                time_limit=15.0
            )
            all_results.extend(results)
    
    print("\n" + "=" * 70)
    print("Aggregating Results...")
    print("=" * 70)
    
    aggregated = aggregate_results(all_results)
    
    raw_output_path = os.path.join(OUTPUTS_ROOT, "experiment_results_raw.json")
    with open(raw_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to: {raw_output_path}")
    
    agg_output_path = os.path.join(OUTPUTS_ROOT, "experiment_results.json")
    with open(agg_output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to: {agg_output_path}")
    
    print("\n" + "=" * 70)
    print("Summary Table (Success Rate)")
    print("=" * 70)
    
    datasets_sorted = sorted(set(r['dataset'] for r in all_results))
    solvers = ['pp', 'marl', 'hybrid']
    
    for dataset in datasets_sorted:
        print(f"\n{dataset}:")
        print(f"  {'Agents':<8} {'PP':<10} {'MARL':<10} {'Hybrid':<10}")
        print(f"  {'-'*38}")
        
        agents_in_dataset = sorted(set(r['num_agents'] for r in all_results 
                                       if r['dataset'] == dataset))
        
        for num_agents in agents_in_dataset:
            rates = []
            for solver in solvers:
                key = f"{dataset}_{num_agents}_{solver}"
                if key in aggregated:
                    rate = aggregated[key]['success_rate']
                    rates.append(f"{rate:.2f}")
                else:
                    rates.append("N/A")
            print(f"  {num_agents:<8} {rates[0]:<10} {rates[1]:<10} {rates[2]:<10}")
    
    return all_results, aggregated

if __name__ == "__main__":
    all_results, aggregated = main()
