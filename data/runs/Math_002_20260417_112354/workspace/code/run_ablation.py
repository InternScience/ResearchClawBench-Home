"""
Ablation study: Effect of switching threshold in MARL-LNS hybrid.
Also: agent density scaling experiment.
"""
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapf_core import MAPFGrid, generate_agent_tasks
from mapf_solvers import MARL_LNS, MAPFLNS2, PrioritizedPlanning, PPWithRandomRestarts

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')


def ablation_switching_threshold():
    """Test different switching thresholds."""
    print("=" * 60)
    print("ABLATION: Switching Threshold")
    print("=" * 60)
    
    thresholds = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    # Test on representative maps
    test_maps = [
        ('empty', 'data/empty/empty_maps_453_25_25', 20),
        ('random_medium', 'data/random_medium/maps_312_25_25_0.175', 18),
        ('maze', 'data/maze/maze_maps_125_25_25', 10),
        ('warehouse', 'data/warehouse/warehouse_maps_297_25_25', 16),
        ('room', 'data/room/room_maps_250_25_25', 15),
    ]
    
    results = {}
    
    for map_type, map_path, n_agents in test_maps:
        full_path = os.path.join(os.path.dirname(DATA_DIR), map_path)
        import glob
        npy_files = sorted(glob.glob(os.path.join(full_path, '*.npy')))[:5]
        
        print(f"\n{map_type} ({n_agents} agents):")
        results[map_type] = {}
        
        for thresh in thresholds:
            successes = 0
            total_collisions = 0
            total_time = 0
            total_marl_iters = 0
            total_pp_iters = 0
            
            for map_file in npy_files:
                grid_data = np.load(map_file)
                grid = MAPFGrid(grid_data)
                
                actual_agents = min(n_agents, len(grid.free_cells) // 2)
                rng = np.random.RandomState(42)
                tasks = generate_agent_tasks(grid, actual_agents, rng)
                
                max_time = min(grid.rows + grid.cols + 20, 80)
                nb_size = min(5, max(3, actual_agents // 4))
                
                ml = MARL_LNS(grid, max_time, 40, nb_size, 5.0, switch_ratio=thresh)
                _, info = ml.solve(tasks, np.random.RandomState(42))
                
                if info['success']:
                    successes += 1
                total_collisions += info['num_collisions']
                total_time += info['runtime']
                total_marl_iters += info.get('marl_iterations', 0)
                total_pp_iters += info.get('pp_iterations', 0)
            
            n = len(npy_files)
            sr = 100 * successes / n
            avg_coll = total_collisions / n
            avg_time = total_time / n
            avg_marl = total_marl_iters / n
            avg_pp = total_pp_iters / n
            
            results[map_type][str(thresh)] = {
                'success_rate': sr,
                'avg_collisions': avg_coll,
                'avg_runtime': avg_time,
                'avg_marl_iters': avg_marl,
                'avg_pp_iters': avg_pp,
            }
            
            print(f"  thresh={thresh:.1f}: SR={sr:5.1f}%, Coll={avg_coll:.1f}, "
                  f"Time={avg_time:.3f}s, MARL={avg_marl:.0f}, PP={avg_pp:.0f}")
    
    return results


def agent_density_scaling():
    """Test how algorithms scale with increasing agent density."""
    print("\n" + "=" * 60)
    print("SCALING: Agent Density")
    print("=" * 60)
    
    # Test on random_medium maps with varying agent counts
    map_path = os.path.join(DATA_DIR, 'random_medium', 'maps_312_25_25_0.175')
    import glob
    npy_files = sorted(glob.glob(os.path.join(map_path, '*.npy')))[:3]
    
    agent_counts = [5, 8, 10, 12, 15, 18, 20, 25]
    
    results = {}
    
    for n_agents in agent_counts:
        print(f"\n{n_agents} agents:")
        results[n_agents] = {}
        
        for solver_name in ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS']:
            successes = 0
            total_collisions = 0
            total_time = 0
            
            for map_file in npy_files:
                grid_data = np.load(map_file)
                grid = MAPFGrid(grid_data)
                
                actual_agents = min(n_agents, len(grid.free_cells) // 2)
                rng = np.random.RandomState(42)
                tasks = generate_agent_tasks(grid, actual_agents, rng)
                
                max_time = 60
                nb_size = min(5, max(3, actual_agents // 4))
                
                if solver_name == 'PP':
                    solver = PrioritizedPlanning(grid, max_time)
                    _, info = solver.solve(tasks)
                elif solver_name == 'PP_Restarts':
                    solver = PPWithRandomRestarts(grid, max_time, 10, 5.0)
                    _, info = solver.solve(tasks, np.random.RandomState(42))
                elif solver_name == 'LNS2':
                    solver = MAPFLNS2(grid, max_time, 40, nb_size, 5.0)
                    _, info = solver.solve(tasks, np.random.RandomState(42))
                elif solver_name == 'MARL_LNS':
                    solver = MARL_LNS(grid, max_time, 40, nb_size, 5.0, switch_ratio=0.5)
                    _, info = solver.solve(tasks, np.random.RandomState(42))
                
                if info['success']:
                    successes += 1
                total_collisions += info['num_collisions']
                total_time += info['runtime']
            
            n = len(npy_files)
            sr = 100 * successes / n
            avg_coll = total_collisions / n
            avg_time = total_time / n
            
            results[n_agents][solver_name] = {
                'success_rate': sr,
                'avg_collisions': avg_coll,
                'avg_runtime': avg_time,
            }
            
            print(f"  {solver_name:15s}: SR={sr:5.1f}%, Coll={avg_coll:.1f}, Time={avg_time:.3f}s")
    
    return results


def collision_reduction_curves():
    """Get collision reduction curves for LNS2 vs MARL-LNS."""
    print("\n" + "=" * 60)
    print("COLLISION REDUCTION CURVES")
    print("=" * 60)
    
    test_maps = [
        ('empty', 'data/empty/empty_maps_453_25_25', 20),
        ('random_medium', 'data/random_medium/maps_312_25_25_0.175', 18),
        ('maze', 'data/maze/maze_maps_125_25_25', 10),
    ]
    
    results = {}
    
    for map_type, map_path, n_agents in test_maps:
        full_path = os.path.join(os.path.dirname(DATA_DIR), map_path)
        import glob
        npy_files = sorted(glob.glob(os.path.join(full_path, '*.npy')))[:1]
        
        grid_data = np.load(npy_files[0])
        grid = MAPFGrid(grid_data)
        
        actual_agents = min(n_agents, len(grid.free_cells) // 2)
        rng = np.random.RandomState(42)
        tasks = generate_agent_tasks(grid, actual_agents, rng)
        
        max_time = min(grid.rows + grid.cols + 20, 80)
        nb_size = min(5, max(3, actual_agents // 4))
        
        # LNS2
        lns2 = MAPFLNS2(grid, max_time, 60, nb_size, 10.0)
        _, info_lns2 = lns2.solve(tasks, np.random.RandomState(42))
        
        # MARL-LNS
        ml = MARL_LNS(grid, max_time, 60, nb_size, 10.0, switch_ratio=0.5)
        _, info_ml = ml.solve(tasks, np.random.RandomState(42))
        
        results[map_type] = {
            'LNS2_history': info_lns2.get('collision_history', []),
            'MARL_LNS_history': info_ml.get('collision_history', []),
        }
        
        print(f"{map_type}: LNS2 final={info_lns2['num_collisions']}, "
              f"MARL-LNS final={info_ml['num_collisions']}")
    
    return results


if __name__ == '__main__':
    # Run ablation
    ablation_results = ablation_switching_threshold()
    
    # Run scaling
    scaling_results = agent_density_scaling()
    
    # Get collision curves
    curve_results = collision_reduction_curves()
    
    # Save all
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(convert(ablation_results), f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'scaling_results.json'), 'w') as f:
        json.dump(convert(scaling_results), f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'curve_results.json'), 'w') as f:
        json.dump(convert(curve_results), f, indent=2)
    
    print("\nAll ablation results saved.")
