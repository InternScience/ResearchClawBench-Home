"""
Main evaluation script: Compare PP, LNS, MARL, and Hybrid MARL-LNS on MAPF benchmarks.
"""

import numpy as np
import os
import sys
import json
import time
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapf_env import (
    MAPFInstance, MAPFInstanceGenerator, load_map_dataset,
    detect_collisions, count_colliding_pairs
)
from mapf_solvers import (
    PrioritizedPlanning, LargeNeighborhoodSearch, 
    MARLPathPlanner, HybridMARLLNS
)
from deep_marl import DeepMARLPlanner


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


def run_benchmark(dataset_name, agent_counts, n_instances=5, time_limit=15, seed=42):
    """Run benchmark comparing all solvers on a dataset.
    
    Returns:
        dict with results per solver
    """
    random.seed(seed)
    np.random.seed(seed)
    
    results = {
        'PP': [],
        'LNS': [],
        'MARL': [],
        'Hybrid': [],
    }
    
    instances = load_map_dataset(DATA_DIR, dataset_name, 
                                  max_instances_per_config=n_instances)
    
    if len(instances) == 0:
        print(f"No instances found for {dataset_name}")
        return results
    
    print(f"Running benchmark on {dataset_name}: {len(instances)} instances")
    
    for idx, inst in enumerate(instances[:n_instances * 4]):  # limit
        print(f"  Instance {idx+1}/{min(len(instances), n_instances * 4)}: "
              f"{inst.n_agents} agents, {inst.H}x{inst.W}, name={inst.name}")
        
        # Prioritized Planning
        pp = PrioritizedPlanning(time_limit=time_limit)
        _, pp_success, pp_stats = pp.solve(inst, seed=seed + idx)
        results['PP'].append(pp_stats)
        
        # LNS
        lns = LargeNeighborhoodSearch(time_limit=time_limit, max_iterations=500)
        _, lns_success, lns_stats = lns.solve(inst, seed=seed + idx)
        results['LNS'].append(lns_stats)
        
        # MARL (Q-learning based)
        marl = MARLPathPlanner(n_episodes=30, max_steps=200, time_limit=time_limit)
        _, marl_success, marl_stats = marl.solve(inst, seed=seed + idx)
        results['MARL'].append(marl_stats)
        
        # Hybrid MARL-LNS
        hybrid = HybridMARLLNS(time_limit=time_limit, marl_episodes=20, 
                                lns_iterations=300, marl_fraction=0.5)
        _, hybrid_success, hybrid_stats = hybrid.solve(inst, seed=seed + idx)
        results['Hybrid'].append(hybrid_stats)
    
    return results


def compute_summary(results, dataset_name):
    """Compute summary statistics from benchmark results."""
    summary = {}
    
    for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
        solver_results = results[solver]
        if not solver_results:
            continue
        
        n = len(solver_results)
        success_rate = sum(1 for r in solver_results if r['success']) / n
        avg_time = np.mean([r['time'] for r in solver_results])
        avg_collisions = np.mean([r.get('n_colliding_pairs', r.get('n_collisions', 0)) 
                                   for r in solver_results])
        avg_soc = np.mean([r.get('sum_of_costs', 0) for r in solver_results])
        avg_makespan = np.mean([r.get('makespan', 0) for r in solver_results])
        
        summary[solver] = {
            'success_rate': float(success_rate),
            'avg_time': float(avg_time),
            'avg_collisions': float(avg_collisions),
            'avg_sum_of_costs': float(avg_soc),
            'avg_makespan': float(avg_makespan),
            'n_instances': n,
        }
    
    return summary


def main():
    """Run the full benchmark."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Datasets to evaluate
    datasets = [
        'random_small',
        'random_medium', 
        'maze',
        'room',
        'warehouse',
        'empty',
    ]
    
    time_limit = 15  # seconds per instance
    
    all_summaries = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        results = run_benchmark(dataset, None, n_instances=5, time_limit=time_limit, seed=42)
        summary = compute_summary(results, dataset)
        all_summaries[dataset] = summary
        
        # Save raw results
        with open(os.path.join(OUTPUT_DIR, f'results_{dataset}.json'), 'w') as f:
            json.dump({'raw': results, 'summary': summary}, f, indent=2, default=str)
        
        # Print summary
        print(f"\nSummary for {dataset}:")
        print(f"{'Solver':<12} {'Success':>8} {'Time(s)':>8} {'Collisions':>10} {'SoC':>8} {'Makespan':>8}")
        print("-" * 60)
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            if solver in summary:
                s = summary[solver]
                print(f"{solver:<12} {s['success_rate']:>7.1%} {s['avg_time']:>8.2f} "
                      f"{s['avg_collisions']:>10.1f} {s['avg_sum_of_costs']:>8.1f} "
                      f"{s['avg_makespan']:>8.1f}")
    
    # Save overall summary
    with open(os.path.join(OUTPUT_DIR, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    
    return all_summaries


if __name__ == '__main__':
    main()
