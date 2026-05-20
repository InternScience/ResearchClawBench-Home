"""
Quick benchmark: Compare solvers on a small subset of instances.
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


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


def run_benchmark(dataset_name, n_instances=3, time_limit=10, seed=42):
    """Run quick benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    
    results = {'PP': [], 'LNS': [], 'MARL': [], 'Hybrid': []}
    
    instances = load_map_dataset(DATA_DIR, dataset_name, max_instances_per_config=1)
    
    # Limit to manageable number
    instances = instances[:n_instances]
    
    if len(instances) == 0:
        print(f"No instances for {dataset_name}")
        return results
    
    print(f"Dataset {dataset_name}: {len(instances)} instances")
    
    for idx, inst in enumerate(instances):
        print(f"  [{idx+1}/{len(instances)}] {inst.n_agents} agents, {inst.H}x{inst.W}")
        
        # PP
        pp = PrioritizedPlanning(time_limit=time_limit)
        _, pp_success, pp_stats = pp.solve(inst, seed=seed + idx)
        results['PP'].append(pp_stats)
        print(f"    PP: success={pp_success}, time={pp_stats['time']:.2f}s, cp={pp_stats.get('n_colliding_pairs',0)}")
        
        # LNS
        lns = LargeNeighborhoodSearch(time_limit=time_limit, max_iterations=200)
        _, lns_success, lns_stats = lns.solve(inst, seed=seed + idx)
        results['LNS'].append(lns_stats)
        print(f"    LNS: success={lns_success}, time={lns_stats['time']:.2f}s, cp={lns_stats.get('n_colliding_pairs',0)}")
        
        # MARL
        marl = MARLPathPlanner(n_episodes=15, max_steps=150, time_limit=time_limit)
        _, marl_success, marl_stats = marl.solve(inst, seed=seed + idx)
        results['MARL'].append(marl_stats)
        print(f"    MARL: success={marl_success}, time={marl_stats['time']:.2f}s, cp={marl_stats.get('n_colliding_pairs',0)}")
        
        # Hybrid
        hybrid = HybridMARLLNS(time_limit=time_limit, marl_episodes=10, 
                                lns_iterations=150, marl_fraction=0.5)
        _, hybrid_success, hybrid_stats = hybrid.solve(inst, seed=seed + idx)
        results['Hybrid'].append(hybrid_stats)
        print(f"    Hybrid: success={hybrid_success}, time={hybrid_stats['time']:.2f}s, cp={hybrid_stats.get('n_colliding_pairs',0)}")
    
    return results


def compute_summary(results, dataset_name):
    """Compute summary statistics."""
    summary = {}
    for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
        sr = results[solver]
        if not sr:
            continue
        n = len(sr)
        summary[solver] = {
            'success_rate': float(sum(1 for r in sr if r['success']) / n),
            'avg_time': float(np.mean([r['time'] for r in sr])),
            'avg_collisions': float(np.mean([r.get('n_colliding_pairs', r.get('n_collisions', 0)) for r in sr])),
            'avg_sum_of_costs': float(np.mean([r.get('sum_of_costs', 0) for r in sr])),
            'avg_makespan': float(np.mean([r.get('makespan', 0) for r in sr])),
            'n_instances': n,
        }
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = ['random_small', 'random_medium', 'maze', 'room', 'warehouse', 'empty']
    time_limit = 8  # seconds per instance
    
    all_summaries = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print(f"{'='*50}")
        
        results = run_benchmark(dataset, n_instances=3, time_limit=time_limit, seed=42)
        summary = compute_summary(results, dataset)
        all_summaries[dataset] = summary
        
        with open(os.path.join(OUTPUT_DIR, f'results_{dataset}.json'), 'w') as f:
            json.dump({'raw': results, 'summary': summary}, f, indent=2, default=str)
        
        print(f"\n  Summary:")
        print(f"  {'Solver':<10} {'Success':>8} {'Time':>8} {'Collisions':>10} {'SoC':>8}")
        print(f"  {'-'*50}")
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            if solver in summary:
                s = summary[solver]
                print(f"  {solver:<10} {s['success_rate']:>7.1%} {s['avg_time']:>8.2f} "
                      f"{s['avg_collisions']:>10.1f} {s['avg_sum_of_costs']:>8.1f}")
    
    with open(os.path.join(OUTPUT_DIR, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nDone! Results in {OUTPUT_DIR}")
    return all_summaries


if __name__ == '__main__':
    main()
