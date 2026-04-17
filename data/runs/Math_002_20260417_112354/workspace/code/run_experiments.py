"""
Efficient experiment runner for MAPF algorithms comparison.
"""
import numpy as np
import os
import sys
import json
import time
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapf_core import MAPFGrid, generate_agent_tasks
from mapf_solvers import (PrioritizedPlanning, PPWithRandomRestarts, 
                          MAPFLNS2, MARL_LNS, PureMARLSolver)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_directory_info(dirname):
    """Parse directory name to extract agent count and grid size."""
    parts = dirname.split('_')
    nums = []
    for p in parts:
        try:
            if '.' not in p:
                nums.append(int(p))
        except:
            pass
    
    if len(nums) >= 3:
        return {'agents': nums[0], 'rows': nums[1], 'cols': nums[2]}
    return None


def get_test_configs():
    """Get representative test configurations - one per map_type/agent_density."""
    configs = []
    
    map_types = {
        'empty': 'data/empty',
        'maze': 'data/maze',
        'random_small': 'data/random_small',
        'random_medium': 'data/random_medium',
        'random_large': 'data/random_large',
        'room': 'data/room',
        'warehouse': 'data/warehouse',
    }
    
    for map_type, rel_path in map_types.items():
        map_dir = os.path.join(os.path.dirname(DATA_DIR), rel_path)
        if not os.path.isdir(map_dir):
            continue
        
        for subdir in sorted(os.listdir(map_dir)):
            subdir_path = os.path.join(map_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            info = parse_directory_info(subdir)
            if info is None:
                continue
            
            npy_files = sorted(glob.glob(os.path.join(subdir_path, '*.npy')))
            if not npy_files:
                continue
            
            configs.append({
                'map_type': map_type,
                'subdir': subdir,
                'path': subdir_path,
                'files': npy_files,
                'num_agents': info['agents'],
                'rows': info['rows'],
                'cols': info['cols'],
            })
    
    return configs


def run_experiment_batch(configs, max_maps=5, agent_caps=None):
    """Run experiments efficiently."""
    all_results = []
    
    # Default agent caps per map type for computational feasibility
    if agent_caps is None:
        agent_caps = {
            'random_small': 15,
            'random_medium': 20,
            'random_large': 15,
            'empty': 20,
            'maze': 12,
            'room': 18,
            'warehouse': 18,
        }
    
    total_configs = len(configs)
    
    for cfg_idx, cfg in enumerate(configs):
        map_type = cfg['map_type']
        cap = agent_caps.get(map_type, 15)
        effective_agents = min(cfg['num_agents'], cap)
        rows, cols = cfg['rows'], cfg['cols']
        max_time = min(rows + cols + 20, 80)
        
        print(f"\n[{cfg_idx+1}/{total_configs}] {map_type}/{cfg['subdir']}: "
              f"{effective_agents} agents on {rows}x{cols}")
        
        map_files = cfg['files'][:max_maps]
        
        for map_idx, map_file in enumerate(map_files):
            grid_data = np.load(map_file)
            grid = MAPFGrid(grid_data)
            
            actual_agents = min(effective_agents, len(grid.free_cells) // 2)
            if actual_agents < 2:
                continue
            
            rng = np.random.RandomState(42 + map_idx)
            tasks = generate_agent_tasks(grid, actual_agents, rng)
            
            # Adjust solver params
            nb_size = min(5, max(3, actual_agents // 4))
            max_iters = 40
            tl = 5.0
            
            result = {
                'map_type': map_type,
                'subdir': cfg['subdir'],
                'map_file': os.path.basename(map_file),
                'num_agents': actual_agents,
                'grid_size': f"{rows}x{cols}",
                'rows': rows, 'cols': cols,
                'original_agents': cfg['num_agents'],
            }
            
            # PP
            try:
                pp = PrioritizedPlanning(grid, max_time)
                _, info = pp.solve(tasks)
                for k, v in info.items():
                    result[f'PP_{k}'] = v
            except Exception as e:
                result['PP_error'] = str(e)
            
            # PP with Restarts
            try:
                pprr = PPWithRandomRestarts(grid, max_time, max_restarts=10, time_limit=tl)
                _, info = pprr.solve(tasks, np.random.RandomState(42))
                for k, v in info.items():
                    result[f'PP_Restarts_{k}'] = v
            except Exception as e:
                result['PP_Restarts_error'] = str(e)
            
            # LNS2
            try:
                lns2 = MAPFLNS2(grid, max_time, max_iters, nb_size, tl)
                _, info = lns2.solve(tasks, np.random.RandomState(42))
                for k, v in info.items():
                    if k not in ('collision_history',):
                        result[f'LNS2_{k}'] = v
                    elif k == 'collision_history':
                        result['LNS2_collision_history'] = info['collision_history']
            except Exception as e:
                result['LNS2_error'] = str(e)
            
            # MARL-LNS
            try:
                ml = MARL_LNS(grid, max_time, max_iters, nb_size, tl, switch_ratio=0.5)
                _, info = ml.solve(tasks, np.random.RandomState(42))
                for k, v in info.items():
                    if k not in ('collision_history', 'phase_history'):
                        result[f'MARL_LNS_{k}'] = v
                    elif k == 'collision_history':
                        result['MARL_LNS_collision_history'] = info['collision_history']
                    elif k == 'phase_history':
                        result['MARL_LNS_marl_iters'] = info.get('marl_iterations', 0)
                        result['MARL_LNS_pp_iters'] = info.get('pp_iterations', 0)
            except Exception as e:
                result['MARL_LNS_error'] = str(e)
            
            # Pure MARL
            try:
                marl = PureMARLSolver(grid, max_time, max_restarts=5)
                _, info = marl.solve(tasks, np.random.RandomState(42))
                for k, v in info.items():
                    result[f'Pure_MARL_{k}'] = v
            except Exception as e:
                result['Pure_MARL_error'] = str(e)
            
            all_results.append(result)
            
            if map_idx == 0 or map_idx == len(map_files) - 1:
                for solver in ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']:
                    s = result.get(f'{solver}_success', 'N/A')
                    c = result.get(f'{solver}_num_collisions', 'N/A')
                    t = result.get(f'{solver}_runtime', 'N/A')
                    if isinstance(t, float):
                        t = f'{t:.3f}s'
                    print(f"  {solver}: success={s}, collisions={c}, time={t}")
    
    return all_results


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def save_results(results, filename='experiment_results.json'):
    filepath = os.path.join(OUTPUT_DIR, filename)
    cleaned = convert_numpy(results)
    with open(filepath, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print(f"\nResults saved to {filepath}")
    return filepath


def print_summary(results):
    """Print summary statistics."""
    by_type = defaultdict(list)
    for r in results:
        by_type[r['map_type']].append(r)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    
    # Overall summary
    summary = {}
    for map_type in sorted(by_type.keys()):
        type_results = by_type[map_type]
        summary[map_type] = {}
        print(f"\n{map_type} ({len(type_results)} instances):")
        
        for solver in solvers:
            successes = sum(1 for r in type_results if r.get(f'{solver}_success', False))
            valid = [r for r in type_results if f'{solver}_num_collisions' in r]
            if valid:
                avg_coll = np.mean([r[f'{solver}_num_collisions'] for r in valid])
                avg_time = np.mean([r[f'{solver}_runtime'] for r in valid])
                avg_makespan = np.mean([r[f'{solver}_makespan'] for r in valid if f'{solver}_makespan' in r])
                avg_soc = np.mean([r[f'{solver}_sum_of_costs'] for r in valid if f'{solver}_sum_of_costs' in r])
            else:
                avg_coll = avg_time = avg_makespan = avg_soc = 0
            
            sr = 100 * successes / max(len(type_results), 1)
            print(f"  {solver:15s}: SR={sr:5.1f}%, Coll={avg_coll:6.1f}, "
                  f"Time={avg_time:6.3f}s, Makespan={avg_makespan:6.1f}, SoC={avg_soc:8.1f}")
            
            summary[map_type][solver] = {
                'success_rate': sr,
                'avg_collisions': float(avg_coll),
                'avg_runtime': float(avg_time),
                'avg_makespan': float(avg_makespan),
                'avg_sum_of_costs': float(avg_soc),
                'num_instances': len(type_results),
            }
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'summary_results.json')
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    return summary


if __name__ == '__main__':
    print("=" * 60)
    print("MAPF Experiment: PP vs LNS2 vs MARL-LNS Hybrid")
    print("=" * 60)
    
    configs = get_test_configs()
    print(f"Found {len(configs)} dataset configurations")
    
    results = run_experiment_batch(configs, max_maps=5)
    save_results(results)
    summary = print_summary(results)
