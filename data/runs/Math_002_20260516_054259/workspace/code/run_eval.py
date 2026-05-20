"""
Simplified benchmark with controlled agent counts.
Generates MAPF instances with specified agent counts for fair comparison.
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
    MAPFInstance, MAPFInstanceGenerator, 
    detect_collisions, count_colliding_pairs
)
from mapf_solvers import (
    PrioritizedPlanning, LargeNeighborhoodSearch, 
    MARLPathPlanner, HybridMARLLNS
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


def load_maps(dataset_name, max_maps=10):
    """Load grid maps from a dataset (without generating agents)."""
    import os
    base = os.path.join(DATA_DIR, dataset_name)
    maps = []
    
    if not os.path.isdir(base):
        return maps
    
    subdirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    
    if not subdirs:
        files = sorted([f for f in os.listdir(base) if f.endswith('.npy')])
        for f in files[:max_maps]:
            grid = np.load(os.path.join(base, f), allow_pickle=True)
            maps.append((f, grid))
    else:
        for sd in subdirs[:2]:  # Limit subdirs
            sd_path = os.path.join(base, sd)
            files = sorted([f for f in os.listdir(sd_path) if f.endswith('.npy')])
            for f in files[:max(1, max_maps // len(subdirs[:2]))]:
                grid = np.load(os.path.join(sd_path, f), allow_pickle=True)
                maps.append((f"{sd}/{f}", grid))
    
    return maps


def get_agent_counts(grid):
    """Determine reasonable agent counts based on grid size and free cells."""
    free_cells = np.sum(grid == 0)
    H, W = grid.shape
    
    # Agent count as fraction of free cells, capped
    fractions = [0.1, 0.2, 0.3]
    counts = []
    for frac in fractions:
        n = int(free_cells * frac / 2)  # divide by 2 because each agent needs start+goal
        n = max(2, min(n, free_cells // 3))
        if n >= 2:
            counts.append(n)
    
    # Deduplicate and ensure at least one count
    counts = sorted(set(counts))
    if not counts:
        counts = [2]
    
    return counts


def run_evaluation():
    """Run evaluation across datasets and agent densities."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    datasets = ['random_small', 'random_medium', 'maze', 'room', 'warehouse', 'empty']
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        maps = load_maps(dataset, max_maps=5)
        print(f"Loaded {len(maps)} maps")
        
        for map_name, grid in maps:
            agent_counts = get_agent_counts(grid)
            
            for n_agents in agent_counts:
                # Generate instance
                inst = MAPFInstanceGenerator.generate_from_grid(grid, n_agents, seed=42)
                inst.name = f"{dataset}/{map_name}"
                
                print(f"  {inst.H}x{inst.W}, {n_agents} agents, {np.sum(grid==0)} free cells")
                
                time_limit = 10
                
                # PP
                pp = PrioritizedPlanning(time_limit=time_limit)
                _, pp_ok, pp_s = pp.solve(inst, seed=0)
                
                # LNS
                lns = LargeNeighborhoodSearch(time_limit=time_limit, max_iterations=200)
                _, lns_ok, lns_s = lns.solve(inst, seed=0)
                
                # MARL
                marl = MARLPathPlanner(n_episodes=20, max_steps=150, time_limit=time_limit)
                _, marl_ok, marl_s = marl.solve(inst, seed=0)
                
                # Hybrid
                hybrid = HybridMARLLNS(time_limit=time_limit, marl_episodes=10, 
                                       lns_iterations=150, marl_fraction=0.5)
                _, hyb_ok, hyb_s = hybrid.solve(inst, seed=0)
                
                result = {
                    'dataset': dataset,
                    'map': map_name,
                    'grid_shape': [int(inst.H), int(inst.W)],
                    'n_agents': n_agents,
                    'free_cells': int(np.sum(grid == 0)),
                    'PP': {
                        'success': pp_ok,
                        'time': pp_s['time'],
                        'collisions': pp_s.get('n_colliding_pairs', 0),
                        'soc': pp_s.get('sum_of_costs', 0),
                        'makespan': pp_s.get('makespan', 0),
                    },
                    'LNS': {
                        'success': lns_ok,
                        'time': lns_s['time'],
                        'collisions': lns_s.get('n_colliding_pairs', 0),
                        'soc': lns_s.get('sum_of_costs', 0),
                        'makespan': lns_s.get('makespan', 0),
                        'iterations': lns_s.get('iterations', 0),
                    },
                    'MARL': {
                        'success': marl_ok,
                        'time': marl_s['time'],
                        'collisions': marl_s.get('n_colliding_pairs', 0),
                        'soc': marl_s.get('sum_of_costs', 0),
                        'makespan': marl_s.get('makespan', 0),
                    },
                    'Hybrid': {
                        'success': hyb_ok,
                        'time': hyb_s['time'],
                        'collisions': hyb_s.get('n_colliding_pairs', 0),
                        'soc': hyb_s.get('sum_of_costs', 0),
                        'makespan': hyb_s.get('makespan', 0),
                        'marl_initial_collisions': hyb_s.get('marl_collisions_initial', 0),
                    },
                }
                
                all_results.append(result)
                
                print(f"    PP:    ok={pp_ok}, t={pp_s['time']:.2f}s, cp={pp_s.get('n_colliding_pairs',0)}")
                print(f"    LNS:   ok={lns_ok}, t={lns_s['time']:.2f}s, cp={lns_s.get('n_colliding_pairs',0)}")
                print(f"    MARL:  ok={marl_ok}, t={marl_s['time']:.2f}s, cp={marl_s.get('n_colliding_pairs',0)}")
                print(f"    Hybrid: ok={hyb_ok}, t={hyb_s['time']:.2f}s, cp={hyb_s.get('n_colliding_pairs',0)}")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute per-dataset summaries
    summaries = {}
    for dataset in datasets:
        ds_results = [r for r in all_results if r['dataset'] == dataset]
        if not ds_results:
            continue
        
        summary = {}
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            n = len(ds_results)
            success_rate = sum(1 for r in ds_results if r[solver]['success']) / n
            avg_time = np.mean([r[solver]['time'] for r in ds_results])
            avg_cp = np.mean([r[solver]['collisions'] for r in ds_results])
            avg_soc = np.mean([r[solver]['soc'] for r in ds_results])
            
            summary[solver] = {
                'success_rate': float(success_rate),
                'avg_time': float(avg_time),
                'avg_colliding_pairs': float(avg_cp),
                'avg_sum_of_costs': float(avg_soc),
                'n': n,
            }
        
        summaries[dataset] = summary
        
        print(f"\n  Summary for {dataset}:")
        print(f"  {'Solver':<10} {'Success':>8} {'Time(s)':>8} {'Collisions':>10} {'SoC':>8}")
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            s = summary[solver]
            print(f"  {solver:<10} {s['success_rate']:>7.1%} {s['avg_time']:>8.2f} "
                  f"{s['avg_colliding_pairs']:>10.1f} {s['avg_sum_of_costs']:>8.1f}")
    
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    
    return all_results, summaries


if __name__ == '__main__':
    run_evaluation()
