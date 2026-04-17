#!/usr/bin/env python3
"""
Minimal experiment runner for MAPF solvers.
"""

import numpy as np
import os
import json
import random
import sys
sys.setrecursionlimit(1000)

from mapf_solver import (
    MAPFInstance, Agent, CollisionDetector,
    PrioritizedPlanning, MARLGuidedPlanner, HybridMARLLNSPP
)

WORKSPACE_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559"
OUTPUTS_ROOT = os.path.join(WORKSPACE_ROOT, "outputs")

def generate_agents(grid, num_agents, seed=42):
    random.seed(seed)
    free_cells = list(zip(*np.where(grid == 0)))
    if len(free_cells) < num_agents * 2:
        num_agents = max(1, len(free_cells) // 2)
    if num_agents == 0:
        return []
    selected = random.sample(range(len(free_cells)), min(num_agents, len(free_cells)))
    starts = [free_cells[i] for i in selected]
    remaining = [i for i in range(len(free_cells)) if i not in selected]
    if remaining:
        goal_indices = random.sample(remaining, min(num_agents, len(remaining)))
        goals = [free_cells[i] for i in goal_indices]
    else:
        goals = starts[::-1]
    return [Agent(id=i, start=starts[i], goal=goals[i % len(goals)]) for i in range(len(starts))]

def evaluate(solver_name, instance, time_limit=3.0):
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
            return {'success': False, 'runtime': 999, 'collisions': 999}
        
        if paths is None:
            return {'success': False, 'runtime': runtime, 'collisions': 999}
        
        collisions = CollisionDetector.detect_collisions(paths)
        return {
            'success': len(collisions) == 0,
            'runtime': min(runtime, 999),
            'collisions': len(collisions),
            'sum_of_costs': sum(len(p)-1 for p in paths)
        }
    except Exception as e:
        return {'success': False, 'runtime': 999, 'collisions': 999, 'error': str(e)[:50]}

# Simple test maps
print("MAPF Minimal Experiments")
print("="*50)

results = []

# Test on a few representative maps
test_configs = [
    ('empty_25x25', np.zeros((25, 25)), [10, 15]),
    ('random_10x10', np.random.choice([0, -1], size=(10, 10), p=[0.8, 0.2]), [5, 8]),
    ('random_25x25', np.random.choice([0, -1], size=(25, 25), p=[0.85, 0.15]), [8, 12]),
]

for map_name, grid, agent_counts in test_configs:
    print(f"\nMap: {map_name} ({grid.shape[0]}x{grid.shape[1]})")
    
    for num_agents in agent_counts:
        agents = generate_agents(grid, num_agents, seed=42)
        if not agents:
            continue
        
        instance = MAPFInstance(grid=grid, agents=agents, height=grid.shape[0], width=grid.shape[1])
        
        for solver in ['pp', 'marl', 'hybrid']:
            res = evaluate(solver, instance, time_limit=2.0)
            res['dataset'] = map_name
            res['num_agents'] = num_agents
            res['solver'] = solver
            results.append(res)
            status = 'OK' if res['success'] else 'FAIL'
            print(f"  {num_agents:2d}a {solver:6s}: {status} t={res['runtime']:.2f}s c={res['collisions']}")

# Aggregate results
from collections import defaultdict
summary = defaultdict(list)
for r in results:
    key = (r['dataset'], r['num_agents'], r['solver'])
    summary[key].append(r)

aggregated = {}
for key, runs in summary.items():
    dataset, n_agents, solver = key
    agg_key = f"{dataset}_{n_agents}_{solver}"
    successes = [1 if r['success'] else 0 for r in runs]
    runtimes = [r['runtime'] for r in runs if r['runtime'] < 999]
    aggregated[agg_key] = {
        'success_rate': float(np.mean(successes)) if successes else 0,
        'avg_runtime': float(np.mean(runtimes)) if runtimes else 999,
        'n_runs': len(runs)
    }

# Save
with open(os.path.join(OUTPUTS_ROOT, "experiment_results.json"), 'w') as f:
    json.dump(aggregated, f, indent=2)
with open(os.path.join(OUTPUTS_ROOT, "experiment_results_raw.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n=== Summary (Success Rate) ===")
for dataset in sorted(set(r['dataset'] for r in results)):
    print(f"\n{dataset}:")
    for n_agents in sorted(set(r['num_agents'] for r in results if r['dataset']==dataset)):
        row = []
        for solver in ['pp', 'marl', 'hybrid']:
            key = f"{dataset}_{n_agents}_{solver}"
            if key in aggregated:
                row.append(f"{aggregated[key]['success_rate']:.2f}")
            else:
                row.append("N/A")
        print(f"  {n_agents:2d}a: PP={row[0]} MARL={row[1]} Hybrid={row[2]}")

print("\nResults saved to outputs/")
