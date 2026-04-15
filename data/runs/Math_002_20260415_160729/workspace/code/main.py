"""
Main script to run MAPF experiments
"""
import os
import sys
import numpy as np
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Import our modules
from mapf_utils import MAPFMap, MAPFInstance
from lns_marl import LNS_MARL, BaselinePP
from evaluate import load_maps, create_test_instances, run_experiments
from visualize import (
    visualize_map, visualize_instance, visualize_solution,
    create_comparison_plots, visualize_map_types
)


def find_map_file(data_dir, dataset, filename):
    """Find a map file in the dataset directory."""
    for root, dirs, files in os.walk(os.path.join(data_dir, dataset)):
        if filename in files:
            return os.path.join(root, filename)
    return None

def run_demo():
    """Run a demo on a single instance to verify implementation."""
    print("=" * 60)
    print("Running MAPF Demo")
    print("=" * 60)
    
    # Load a sample map - find the first available one
    map_path = find_map_file('data', 'random_medium', 'eval_map_1.npy')
    if map_path is None:
        map_path = find_map_file('data', 'random_small', 'eval_map_1.npy')
    grid = np.load(map_path, allow_pickle=True)
    map_obj = MAPFMap(grid)
    
    print(f"Map shape: {grid.shape}")
    print(f"Free cells: {len(map_obj.get_free_positions())}")
    
    # Create instance
    num_agents = 8
    instance = MAPFInstance.generate_random(map_obj, num_agents, seed=42)
    
    print(f"\nInstance with {num_agents} agents created")
    print(f"Starts: {instance.starts}")
    print(f"Goals: {instance.goals}")
    
    # Visualize instance
    os.makedirs('report/images', exist_ok=True)
    visualize_instance(instance, "MAPF Instance", 'report/images/demo_instance.png')
    print("Instance visualization saved")
    
    # Run Baseline PP
    print("\n--- Running Baseline Prioritized Planning ---")
    pp_solver = BaselinePP(instance)
    pp_solution, pp_stats = pp_solver.solve()
    
    print(f"Success: {pp_stats['success']}")
    print(f"Runtime: {pp_stats['time']:.4f}s")
    print(f"Collisions: {pp_stats['collisions']}")
    print(f"Cost: {pp_stats['cost']}")
    
    visualize_solution(instance, pp_solution, "Baseline-PP Solution", 
                      'report/images/demo_pp_solution.png')
    print("Baseline-PP solution visualization saved")
    
    # Run LNS-MARL
    print("\n--- Running LNS-MARL ---")
    lns_solver = LNS_MARL(instance, seed=42)
    lns_solution, lns_stats = lns_solver.solve(max_iterations=50, neighborhood_size=5)
    
    print(f"Success: {lns_stats['success']}")
    print(f"Runtime: {lns_stats['time']:.4f}s")
    print(f"Iterations: {lns_stats['iterations']}")
    print(f"Initial Collisions: {lns_stats['initial_collisions']}")
    print(f"Final Collisions: {lns_stats['final_collisions']}")
    print(f"Cost: {lns_stats['final_cost']}")
    
    visualize_solution(instance, lns_solution, "LNS-MARL Solution",
                      'report/images/demo_lns_solution.png')
    print("LNS-MARL solution visualization saved")
    
    return True


def run_full_evaluation():
    """Run full evaluation on all datasets."""
    print("=" * 60)
    print("Running Full MAPF Evaluation")
    print("=" * 60)
    
    # Create visualizations of map types
    print("\nCreating map type overview...")
    visualize_map_types('data', 'report/images')
    
    # Run experiments
    print("\nRunning experiments...")
    results = run_experiments('data', 'outputs')
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results, 'report/images')
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    # First run demo
    if run_demo():
        print("\nDemo completed successfully!\n")
    
    # Then run full evaluation
    results = run_full_evaluation()
