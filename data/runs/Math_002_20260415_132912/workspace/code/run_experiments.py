"""
Run experiments across all map types and generate results.
Optimized for faster execution.
"""

import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, '.')

from code.mapf_core import MapLoader, generate_agents, CollisionChecker, AStarSearch
from code.solvers import PrioritizedPlanning, LNSSolver, MARLInformedSolver


def run_single_experiment(map_grid, agents, solver_name, solver, max_time=10.0):
    """Run a single MAPF experiment."""
    result = solver.solve(map_grid, agents, max_time=max_time)
    
    return {
        'solver': solver_name,
        'success': result['success'],
        'collisions': result['collisions'],
        'sum_of_costs': result['sum_of_costs'],
        'runtime': result['runtime'],
        'iterations': result.get('iterations', 0),
        'n_agents': len(agents),
        'map_size': list(map_grid.shape),
        'obstacle_ratio': float(np.sum(map_grid == -1) / (map_grid.shape[0] * map_grid.shape[1])),
        'history': result.get('history', None)
    }


def load_maps_from_dataset(loader, dataset, n_maps=5):
    """Load a subset of maps from a dataset."""
    map_files = loader.list_maps(dataset)
    
    if len(map_files) > n_maps:
        step = len(map_files) // n_maps
        map_files = map_files[::step][:n_maps]
    
    maps = []
    for f in map_files:
        m = loader.load_map(f)
        maps.append((f, m))
    
    return maps


def main():
    print("=" * 60)
    print("MARL-LNS MAPF Experiments")
    print("=" * 60)
    
    loader = MapLoader()
    datasets = loader.list_datasets()
    print(f"Available datasets: {datasets}")
    
    # Reduced configuration for faster execution
    agent_counts = {
        'random_small': [10, 20],
        'random_medium': [20, 40],
        'random_large': [50],
        'empty': [20, 40],
        'maze': [10, 20],
        'room': [15, 30],
        'warehouse': [20, 40],
        'maps_60_10_10_0.175': [10, 20],
    }
    
    n_maps_per_config = 5
    max_time_per_solve = 10.0
    
    all_results = []
    convergence_histories = {'PP': [], 'LNS': [], 'MARL-LNS': []}
    
    total_configs = sum(len(counts) for counts in agent_counts.values())
    config_idx = 0
    
    for dataset in datasets:
        if dataset not in agent_counts:
            continue
        
        print(f"\n--- Dataset: {dataset} ---")
        maps = load_maps_from_dataset(loader, dataset, n_maps=n_maps_per_config)
        
        if not maps:
            print(f"  No maps found in {dataset}")
            continue
        
        for n_agents in agent_counts[dataset]:
            config_idx += 1
            print(f"\n  Config {config_idx}/{total_configs}: {dataset}, {n_agents} agents")
            
            for map_idx, (map_file, map_grid) in enumerate(maps):
                seed = map_idx * 100 + n_agents
                agents = generate_agents(map_grid, n_agents, seed=seed)
                
                solvers = [
                    ('PP', PrioritizedPlanning(seed=seed)),
                    ('LNS', LNSSolver(seed=seed)),
                    ('MARL-LNS', MARLInformedSolver(seed=seed)),
                ]
                
                for solver_name, solver in solvers:
                    try:
                        result = run_single_experiment(
                            map_grid, agents, solver_name, solver, 
                            max_time=max_time_per_solve
                        )
                        result['dataset'] = dataset
                        result['map_file'] = os.path.basename(map_file)
                        result['map_idx'] = map_idx
                        result['seed'] = seed
                        
                        all_results.append(result)
                        
                        # Store convergence history
                        if result.get('history') and result['history'].get('collisions'):
                            entry = {
                                'dataset': dataset,
                                'n_agents': n_agents,
                                'map_idx': map_idx,
                                'solver': solver_name,
                                'iterations': result['history']['iteration'],
                                'collisions': result['history']['collisions'],
                                'runtime': result['history']['runtime'],
                            }
                            if 'phase' in result['history']:
                                entry['phase'] = result['history']['phase']
                            if 'neighborhood_size' in result['history']:
                                entry['neighborhood_size'] = result['history']['neighborhood_size']
                            
                            convergence_histories[solver_name].append(entry)
                        
                        status = "SUCCESS" if result['success'] else f"FAIL ({result['collisions']} collisions)"
                        print(f"    Map {map_idx}: {solver_name} -> {status}, SOC={result['sum_of_costs']}, "
                              f"time={result['runtime']:.2f}s")
                    
                    except Exception as e:
                        print(f"    Map {map_idx}: {solver_name} -> ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            'solver': solver_name,
                            'dataset': dataset,
                            'n_agents': n_agents,
                            'map_idx': map_idx,
                            'success': False,
                            'collisions': -1,
                            'sum_of_costs': -1,
                            'runtime': -1,
                            'error': str(e)
                        })
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    
    with open('outputs/experiment_results.json', 'w') as f:
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if k == 'history':
                    continue
                if isinstance(v, (np.integer,)):
                    sr[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    sr[k] = float(v)
                elif isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                elif isinstance(v, tuple):
                    sr[k] = list(v)
                else:
                    sr[k] = v
            serializable.append(sr)
        json.dump(serializable, f, indent=2)
    
    for solver_name, histories in convergence_histories.items():
        with open(f'outputs/convergence_{solver_name}.json', 'w') as f:
            json.dump(histories, f, indent=2)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    solvers = ['PP', 'LNS', 'MARL-LNS']
    for solver_name in solvers:
        solver_results = [r for r in all_results if r.get('solver') == solver_name]
        successes = sum(1 for r in solver_results if r.get('success', False))
        total = len(solver_results)
        success_rate = successes / total * 100 if total > 0 else 0
        
        valid_runtimes = [r['runtime'] for r in solver_results if r.get('runtime', 0) > 0]
        avg_runtime = float(np.mean(valid_runtimes)) if valid_runtimes else 0
        
        valid_soc = [r['sum_of_costs'] for r in solver_results if r.get('sum_of_costs', 0) > 0]
        avg_soc = float(np.mean(valid_soc)) if valid_soc else 0
        
        print(f"\n{solver_name}:")
        print(f"  Success Rate: {success_rate:.1f}% ({successes}/{total})")
        print(f"  Avg Runtime: {avg_runtime:.2f}s")
        print(f"  Avg Sum-of-Costs: {avg_soc:.1f}")
    
    print(f"\nResults saved to outputs/experiment_results.json")
    print(f"Convergence histories saved to outputs/convergence_*.json")
    
    return all_results


if __name__ == '__main__':
    main()
