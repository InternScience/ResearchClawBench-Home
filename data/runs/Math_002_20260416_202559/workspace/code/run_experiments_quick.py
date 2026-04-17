#!/usr/bin/env python3
"""
Quick experiment runner for MAPF solvers - minimal version.
"""

import numpy as np
import os
import json
import random
from mapf_solver import (
    MAPFInstance, Agent, CollisionDetector,
    PrioritizedPlanning, MARLGuidedPlanner, HybridMARLLNSPP
)

WORKSPACE_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559"
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")
OUTPUTS_ROOT = os.path.join(WORKSPACE_ROOT, "outputs")

def generate_agents(grid, num_agents, seed=42):
    random.seed(seed)
    free_cells = list(zip(*np.where(grid == 0)))
    if len(free_cells) < num_agents * 2:
        num_agents = len(free_cells) // 2
    if num_agents == 0:
        return []
    selected = random.sample(range(len(free_cells)), num_agents)
    starts = [free_cells[i] for i in selected]
    remaining = [i for i in range(len(free_cells)) if i not in selected]
    goal_indices = random.sample(remaining, min(num_agents, len(remaining)))
    goals = [free_cells[i] for i in goal_indices]
    return [Agent(id=i, start=starts[i], goal=goals[i]) for i in range(num_agents)]

def evaluate(solver_name, instance, time_limit=10.0):
    try:
        if solver_name == 'pp':
            solver = PrioritizedPlanning(instance)
            paths, runtime = solver.solve()
        elif solver_name == 'marl':
            solver = MARLGuidedPlanner(instance)
            paths, runtime = solver.generate_initial_solution()
        elif solver_name == 'hybrid':
            solver = HybridMARLLNSPP(instance)
            paths, runtime, _ = solver.solve(time_limit=time_limit)
        else:
            return {'success': False, 'runtime': float('inf'), 'collisions': float('inf')}
        
        if paths is None:
            return {'success': False, 'runtime': runtime, 'collisions': float('inf')}
        
        collisions = CollisionDetector.detect_collisions(paths)
        return {
            'success': len(collisions) == 0,
            'runtime': runtime,
            'collisions': len(collisions),
            'sum_of_costs': sum(len(p)-1 for p in paths)
        }
    except Exception as e:
        return {'success': False, 'runtime': float('inf'), 'collisions': float('inf'), 'error': str(e)}

def run_single_dataset(dataset_name, rel_path, num_agents_list, n_instances=2):
    print(f"\nDataset: {dataset_name}")
    dataset_path = os.path.join(WORKSPACE_ROOT, rel_path)
    
    map_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.npy'):
                map_files.append(os.path.join(root, f))
    
    if not map_files:
        return []
    
    results = []
    for num_agents in num_agents_list:
        for inst_idx in range(n_instances):
            map_file = random.choice(map_files)
            grid = np.load(map_file, allow_pickle=True)
            agents = generate_agents(grid, num_agents, seed=inst_idx*100+num_agents)
            if len(agents) < num_agents:
                continue
            
            instance = MAPFInstance(grid=grid, agents=agents, height=grid.shape[0], width=grid.shape[1])
            
            for solver in ['pp', 'marl', 'hybrid']:
                res = evaluate(solver, instance, time_limit=5.0)
                res['dataset'] = dataset_name
                res['num_agents'] = num_agents
                res['solver'] = solver
                res['map_size'] = f"{grid.shape[0]}x{grid.shape[1]}"
                results.append(res)
                print(f"  {num_agents}a {solver:6s}: {'OK' if res['success'] else 'FAIL'} t={res['runtime']:.2f}s c={res['collisions']}")
    
    return results

def main():
    print("MAPF Experiments (Quick)")
    
    datasets = [
        ('empty', 'data/empty', [10, 15]),
        ('random_small', 'data/random_small', [5, 8]),
        ('random_medium', 'data/random_medium', [8, 12]),
        ('room', 'data/room', [8, 12]),
        ('warehouse', 'data/warehouse', [8, 12]),
    ]
    
    all_results = []
    for name, path, agents in datasets:
        results = run_single_dataset(name, path, agents, n_instances=2)
        all_results.extend(results)
    
    # Aggregate
    from collections import defaultdict
    summary = defaultdict(list)
    for r in all_results:
        key = (r['dataset'], r['num_agents'], r['solver'])
        summary[key].append(r)
    
    aggregated = {}
    for key, runs in summary.items():
        dataset, n_agents, solver = key
        agg_key = f"{dataset}_{n_agents}_{solver}"
        successes = [1 if r['success'] else 0 for r in runs]
        runtimes = [r['runtime'] for r in runs if r['runtime'] < float('inf')]
        aggregated[agg_key] = {
            'success_rate': float(np.mean(successes)) if successes else 0,
            'avg_runtime': float(np.mean(runtimes)) if runtimes else float('inf'),
            'n_runs': len(runs)
        }
    
    # Save
    with open(os.path.join(OUTPUTS_ROOT, "experiment_results.json"), 'w') as f:
        json.dump(aggregated, f, indent=2)
    with open(os.path.join(OUTPUTS_ROOT, "experiment_results_raw.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n=== Summary (Success Rate) ===")
    for dataset in sorted(set(r['dataset'] for r in all_results)):
        print(f"\n{dataset}:")
        for n_agents in sorted(set(r['num_agents'] for r in all_results if r['dataset']==dataset)):
            row = []
            for solver in ['pp', 'marl', 'hybrid']:
                key = f"{dataset}_{n_agents}_{solver}"
                if key in aggregated:
                    row.append(f"{aggregated[key]['success_rate']:.2f}")
                else:
                    row.append("N/A")
            print(f"  {n_agents:2d}a: PP={row[0]} MARL={row[1]} Hybrid={row[2]}")
    
    print("\nResults saved to outputs/")
    return all_results, aggregated

if __name__ == "__main__":
    main()
