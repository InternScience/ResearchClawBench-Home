"""
Focused evaluation: Compare solvers across datasets with multiple agent densities.
Generates data for figures and analysis.
"""

import numpy as np
import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapf_env import MAPFInstance, MAPFInstanceGenerator, detect_collisions, count_colliding_pairs
from mapf_solvers import (
    PrioritizedPlanning, LargeNeighborhoodSearch, 
    MARLPathPlanner, HybridMARLLNS
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


# Map files to use (one representative per dataset)
MAP_FILES = {
    'random_small': 'random_small/maps_50_10_10_0.175/eval_map_1.npy',
    'random_medium': 'random_medium/maps_312_25_25_0.175/eval_map_1.npy',
    'maze': 'maze/maze_maps_125_25_25/eval_map_maze_1.npy',
    'room': 'room/room_maps_250_25_25/eval_map_room_1.npy',
    'warehouse': 'warehouse/warehouse_maps_266_25_25/eval_map_warehouse_1.npy',
    'empty': 'empty/empty_maps_453_25_25/eval_map_empty_1.npy',
}


def run_full_evaluation():
    """Run comprehensive evaluation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Agent densities to test
    agent_densities = [0.05, 0.10, 0.15, 0.20]  # as fraction of free cells
    
    # Time limits to test
    time_limits = [2, 5, 10]
    
    all_results = []
    
    for dataset, map_rel_path in MAP_FILES.items():
        map_path = os.path.join(DATA_DIR, map_rel_path)
        if not os.path.exists(map_path):
            print(f"SKIP: {map_path} not found")
            continue
        
        grid = np.load(map_path, allow_pickle=True)
        free_cells = np.sum(grid == 0)
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({grid.shape}, {free_cells} free)")
        print(f"{'='*60}")
        
        for density in agent_densities:
            n_agents = max(2, int(free_cells * density / 2))
            if n_agents < 2:
                continue
            
            for tlimit in time_limits:
                # Run 3 seeds
                for seed in [42, 123, 456]:
                    inst = MAPFInstanceGenerator.generate_from_grid(grid, n_agents, seed=seed)
                    inst.name = f"{dataset}_{n_agents}agents"
                    
                    # PP
                    pp = PrioritizedPlanning(time_limit=tlimit)
                    _, pp_ok, pp_s = pp.solve(inst, seed=seed)
                    
                    # LNS
                    lns = LargeNeighborhoodSearch(time_limit=tlimit, max_iterations=200, neighborhood_size=0.3)
                    _, lns_ok, lns_s = lns.solve(inst, seed=seed)
                    
                    # MARL
                    marl = MARLPathPlanner(n_episodes=20, max_steps=150, time_limit=tlimit)
                    _, marl_ok, marl_s = marl.solve(inst, seed=seed)
                    
                    # Hybrid
                    hybrid = HybridMARLLNS(time_limit=tlimit, marl_episodes=10, 
                                           lns_iterations=100, marl_fraction=0.4)
                    _, hyb_ok, hyb_s = hybrid.solve(inst, seed=seed)
                    
                    result = {
                        'dataset': dataset,
                        'grid_shape': [int(grid.shape[0]), int(grid.shape[1])],
                        'free_cells': int(free_cells),
                        'n_agents': n_agents,
                        'agent_density': float(density),
                        'time_limit': tlimit,
                        'seed': seed,
                        'PP': {
                            'success': pp_ok, 'time': pp_s['time'],
                            'colliding_pairs': pp_s.get('n_colliding_pairs', 0),
                            'soc': pp_s.get('sum_of_costs', 0),
                            'makespan': pp_s.get('makespan', 0),
                        },
                        'LNS': {
                            'success': lns_ok, 'time': lns_s['time'],
                            'colliding_pairs': lns_s.get('n_colliding_pairs', 0),
                            'soc': lns_s.get('sum_of_costs', 0),
                            'makespan': lns_s.get('makespan', 0),
                            'iterations': lns_s.get('iterations', 0),
                        },
                        'MARL': {
                            'success': marl_ok, 'time': marl_s['time'],
                            'colliding_pairs': marl_s.get('n_colliding_pairs', 0),
                            'soc': marl_s.get('sum_of_costs', 0),
                            'makespan': marl_s.get('makespan', 0),
                        },
                        'Hybrid': {
                            'success': hyb_ok, 'time': hyb_s['time'],
                            'colliding_pairs': hyb_s.get('n_colliding_pairs', 0),
                            'soc': hyb_s.get('sum_of_costs', 0),
                            'makespan': hyb_s.get('makespan', 0),
                            'marl_initial_cp': hyb_s.get('marl_collisions_initial', 0),
                        },
                    }
                    all_results.append(result)
                    
                    status = f"  d={density:.2f} t={tlimit}s seed={seed}: "
                    status += f"PP={'✓' if pp_ok else '✗'} "
                    status += f"LNS={'✓' if lns_ok else '✗'} "
                    status += f"MARL={'✓' if marl_ok else '✗'} "
                    status += f"Hyb={'✓' if hyb_ok else '✗'}"
                    print(status)
    
    # Save raw results
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


if __name__ == '__main__':
    run_full_evaluation()
