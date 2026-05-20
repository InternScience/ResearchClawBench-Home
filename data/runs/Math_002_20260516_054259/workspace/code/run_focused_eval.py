"""
Focused evaluation with controlled parameters.
Tests key hypotheses about the Hybrid MARL-LNS approach.
"""

import numpy as np
import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapf_env import MAPFInstance, MAPFInstanceGenerator, detect_collisions, count_colliding_pairs, a_star
from mapf_solvers import (
    PrioritizedPlanning, LargeNeighborhoodSearch, 
    MARLPathPlanner, HybridMARLLNS
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


# One map per dataset
MAP_FILES = {
    'random_small': 'random_small/maps_50_10_10_0.175/eval_map_1.npy',
    'random_medium': 'random_medium/maps_312_25_25_0.175/eval_map_1.npy',
    'maze': 'maze/maze_maps_125_25_25/eval_map_maze_1.npy',
    'room': 'room/room_maps_250_25_25/eval_map_room_1.npy',
    'warehouse': 'warehouse/warehouse_maps_266_25_25/eval_map_warehouse_1.npy',
    'empty': 'empty/empty_maps_453_25_25/eval_map_empty_1.npy',
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use two agent densities and two time limits
    densities = [0.10, 0.20]  # fraction of free cells
    time_limit = 5  # seconds
    seeds = [42, 123]
    
    all_results = []
    
    for dataset, map_rel_path in MAP_FILES.items():
        map_path = os.path.join(DATA_DIR, map_rel_path)
        if not os.path.exists(map_path):
            continue
        
        grid = np.load(map_path, allow_pickle=True)
        free_cells = int(np.sum(grid == 0))
        
        print(f"\n{'='*50}")
        print(f"{dataset}: {grid.shape}, {free_cells} free cells")
        
        for density in densities:
            n_agents = max(2, int(free_cells * density / 2))
            
            for seed in seeds:
                inst = MAPFInstanceGenerator.generate_from_grid(grid, n_agents, seed=seed)
                
                r = {
                    'dataset': dataset,
                    'grid_shape': list(grid.shape),
                    'free_cells': free_cells,
                    'n_agents': n_agents,
                    'density': density,
                    'seed': seed,
                }
                
                # PP
                pp = PrioritizedPlanning(time_limit=time_limit)
                _, ok, s = pp.solve(inst, seed=seed)
                r['PP'] = {'success': ok, 'time': s['time'], 'cp': s.get('n_colliding_pairs',0),
                           'soc': s.get('sum_of_costs',0), 'makespan': s.get('makespan',0)}
                
                # LNS
                lns = LargeNeighborhoodSearch(time_limit=time_limit, max_iterations=200)
                _, ok, s = lns.solve(inst, seed=seed)
                r['LNS'] = {'success': ok, 'time': s['time'], 'cp': s.get('n_colliding_pairs',0),
                            'soc': s.get('sum_of_costs',0), 'makespan': s.get('makespan',0),
                            'iterations': s.get('iterations',0)}
                
                # MARL
                marl = MARLPathPlanner(n_episodes=15, max_steps=150, time_limit=time_limit)
                _, ok, s = marl.solve(inst, seed=seed)
                r['MARL'] = {'success': ok, 'time': s['time'], 'cp': s.get('n_colliding_pairs',0),
                             'soc': s.get('sum_of_costs',0), 'makespan': s.get('makespan',0)}
                
                # Hybrid
                hyb = HybridMARLLNS(time_limit=time_limit, marl_episodes=8, 
                                    lns_iterations=100, marl_fraction=0.4)
                _, ok, s = hyb.solve(inst, seed=seed)
                r['Hybrid'] = {'success': ok, 'time': s['time'], 'cp': s.get('n_colliding_pairs',0),
                               'soc': s.get('sum_of_costs',0), 'makespan': s.get('makespan',0),
                               'marl_initial_cp': s.get('marl_collisions_initial',0)}
                
                all_results.append(r)
                
                print(f"  d={density:.2f} ag={n_agents} s={seed}: "
                      f"PP={r['PP']['success']} LNS={r['LNS']['success']} "
                      f"MARL={r['MARL']['success']} Hyb={r['Hybrid']['success']} "
                      f"| CP: PP={r['PP']['cp']} LNS={r['LNS']['cp']} "
                      f"MARL={r['MARL']['cp']} Hyb={r['Hybrid']['cp']}")
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, 'focused_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compute summaries
    from collections import defaultdict
    
    # By dataset
    by_dataset = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        ds = r['dataset']
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            by_dataset[ds][solver].append(r[solver])
    
    summaries = {}
    for ds in by_dataset:
        summaries[ds] = {}
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            items = by_dataset[ds][solver]
            n = len(items)
            summaries[ds][solver] = {
                'success_rate': sum(1 for it in items if it['success']) / n,
                'avg_time': np.mean([it['time'] for it in items]),
                'avg_cp': np.mean([it['cp'] for it in items]),
                'avg_soc': np.mean([it['soc'] for it in items]),
                'avg_makespan': np.mean([it['makespan'] for it in items]),
            }
    
    with open(os.path.join(OUTPUT_DIR, 'focused_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    
    # Print table
    print("\n\nFINAL SUMMARY:")
    print(f"{'Dataset':<16} {'Solver':<8} {'Success':>8} {'Time':>8} {'CP':>8} {'SoC':>8}")
    print("-" * 56)
    for ds in sorted(summaries.keys()):
        for solver in ['PP', 'LNS', 'MARL', 'Hybrid']:
            s = summaries[ds][solver]
            print(f"{ds:<16} {solver:<8} {s['success_rate']:>7.1%} {s['avg_time']:>8.2f} "
                  f"{s['avg_cp']:>8.1f} {s['avg_soc']:>8.1f}")
    
    return all_results, summaries


if __name__ == '__main__':
    main()
