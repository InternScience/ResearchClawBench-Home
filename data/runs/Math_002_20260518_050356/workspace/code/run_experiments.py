"""
Main Experiment Runner
Runs experiments on all datasets and generates results.
"""

import numpy as np
import os
import json
import time
import random
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from mapf_core import MAPFInstance, Path, generate_random_instance, count_all_collisions
from marl_agent import MARLAgent, pretrain_marl
from lns_framework import LNS, MARLGuidedLNS
from hybrid_marl_lns import HybridMARLLNS, compare_algorithms


class ExperimentRunner:
    """Runs experiments on MAPF datasets."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.all_results = {}
    
    def load_maps(self, dataset_name: str, subdir: str = None) -> List[np.ndarray]:
        """Load maps from a dataset."""
        if subdir:
            dataset_path = os.path.join(self.data_dir, dataset_name, subdir)
        else:
            dataset_path = os.path.join(self.data_dir, dataset_name)
        
        maps = []
        if os.path.exists(dataset_path):
            for f in sorted(os.listdir(dataset_path)):
                if f.endswith('.npy'):
                    maps.append(np.load(os.path.join(dataset_path, f), allow_pickle=True))
        
        return maps
    
    def run_experiment(self, dataset_name: str, subdir: str, 
                      agent_counts: List[int], num_instances: int = 10,
                      max_iterations: int = 200, timeout: float = 10.0) -> Dict:
        """
        Run experiment on a dataset.
        
        Args:
            dataset_name: Name of dataset
            subdir: Subdirectory name
            agent_counts: List of agent counts to test
            num_instances: Number of instances per configuration
            max_iterations: Max LNS iterations
            timeout: Time limit per instance
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {dataset_name}/{subdir}")
        print(f"Agent counts: {agent_counts}")
        print(f"{'='*60}")
        
        maps = self.load_maps(dataset_name, subdir)
        if not maps:
            print(f"No maps found for {dataset_name}/{subdir}")
            return {}
        
        results = {
            "dataset": dataset_name,
            "subdir": subdir,
            "map_size": maps[0].shape if maps else None,
            "agent_counts": agent_counts,
            "num_instances": num_instances,
            "algorithms": {}
        }
        
        for agent_count in agent_counts:
            print(f"\nAgent count: {agent_count}")
            
            # Sample maps for this agent count
            sample_maps = random.sample(maps, min(num_instances, len(maps)))
            
            agent_results = {
                "PurePP": {"success": 0, "collisions": [], "soc": [], "runtime": []},
                "MAPF-LNS2": {"success": 0, "collisions": [], "soc": [], "runtime": []},
                "HybridMARL-LNS": {"success": 0, "collisions": [], "soc": [], "runtime": []},
                "MARL-LNS-Only": {"success": 0, "collisions": [], "soc": [], "runtime": []}
            }
            
            for idx, grid in enumerate(sample_maps):
                print(f"  Instance {idx+1}/{len(sample_maps)}", end="", flush=True)
                
                try:
                    # Generate instance
                    instance = generate_random_instance(grid, agent_count, seed=idx)
                    
                    # Run comparison
                    instance_results = compare_algorithms(
                        instance, max_iterations, timeout
                    )
                    
                    # Aggregate results
                    for algo_name, algo_result in instance_results.items():
                        if algo_result["success"]:
                            agent_results[algo_name]["success"] += 1
                        agent_results[algo_name]["collisions"].append(algo_result["collisions"])
                        agent_results[algo_name]["soc"].append(algo_result["soc"])
                        agent_results[algo_name]["runtime"].append(algo_result["runtime"])
                    
                    print(f" - OK")
                    
                except Exception as e:
                    print(f" - FAILED: {str(e)}")
                    continue
            
            # Calculate statistics
            for algo_name in agent_results:
                algo = agent_results[algo_name]
                n = len(algo["collisions"])
                if n > 0:
                    algo["success_rate"] = algo["success"] / n
                    algo["avg_collisions"] = np.mean(algo["collisions"])
                    algo["avg_soc"] = np.mean(algo["soc"])
                    algo["avg_runtime"] = np.mean(algo["runtime"])
                else:
                    algo["success_rate"] = 0
                    algo["avg_collisions"] = float('inf')
                    algo["avg_soc"] = float('inf')
                    algo["avg_runtime"] = float('inf')
            
            results["algorithms"][agent_count] = agent_results
        
        return results
    
    def run_all_experiments(self, max_iterations: int = 200, 
                           timeout: float = 10.0, num_instances: int = 5):
        """Run experiments on all datasets."""
        
        # Define datasets and agent counts
        experiments = [
            ("random_small", "maps_50_10_10_0.175", [5, 10, 15, 20]),
            ("random_small", "maps_55_10_10_0.175", [5, 10, 15, 20]),
            ("random_medium", "maps_312_25_25_0.175", [10, 20, 30, 40]),
            ("random_medium", "maps_344_25_25_0.175", [10, 20, 30, 40]),
            ("random_large", "maps_1500_50_50_0.175", [20, 40, 60, 80]),
            ("random_large", "maps_1625_50_50_0.175", [20, 40, 60, 80]),
            ("empty", "empty_maps_453_25_25", [10, 20, 30, 40]),
            ("empty", "empty_maps_500_25_25", [10, 20, 30, 40]),
            ("maze", "maze_maps_125_25_25", [10, 20, 30, 40]),
            ("maze", "maze_maps_141_25_25", [10, 20, 30, 40]),
            ("room", "room_maps_250_25_25", [10, 20, 30, 40]),
            ("room", "room_maps_312_25_25", [10, 20, 30, 40]),
            ("warehouse", "warehouse_maps_281_25_25", [10, 20, 30, 40]),
            ("warehouse", "warehouse_maps_297_25_25", [10, 20, 30, 40]),
        ]
        
        all_results = {}
        
        for dataset_name, subdir, agent_counts in experiments:
            try:
                results = self.run_experiment(
                    dataset_name, subdir, agent_counts,
                    num_instances, max_iterations, timeout
                )
                key = f"{dataset_name}/{subdir}"
                all_results[key] = results
                
                # Save intermediate results
                self.save_results(all_results)
                
            except Exception as e:
                print(f"Error in {dataset_name}/{subdir}: {str(e)}")
                continue
        
        self.all_results = all_results
        return all_results
    
    def save_results(self, results: Dict = None):
        """Save results to JSON file."""
        if results is None:
            results = self.all_results
        
        output_path = os.path.join(self.output_dir, "experimental_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
    
    def generate_figures(self):
        """Generate publication-quality figures."""
        if not self.all_results:
            print("No results to plot")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Figure 1: Success rate by dataset type
        self._plot_success_rate_by_dataset()
        
        # Figure 2: Success rate by agent count
        self._plot_success_rate_by_agents()
        
        # Figure 3: Sum of costs comparison
        self._plot_soc_comparison()
        
        # Figure 4: Runtime comparison
        self._plot_runtime_comparison()
        
        # Figure 5: Convergence curve
        self._plot_convergence_curve()
        
        # Figure 6: Collision reduction
        self._plot_collision_reduction()
        
        print("All figures generated")
    
    def _plot_success_rate_by_dataset(self):
        """Plot success rate grouped by dataset type."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Aggregate by dataset type
        dataset_types = {}
        for key, results in self.all_results.items():
            dataset_type = key.split('/')[0]
            if dataset_type not in dataset_types:
                dataset_types[dataset_type] = {}
            
            for algo_name in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS", "MARL-LNS-Only"]:
                if algo_name in str(results.get("algorithms", {})):
                    for agent_count, agent_results in results["algorithms"].items():
                        if algo_name in agent_results:
                            if algo_name not in dataset_types[dataset_type]:
                                dataset_types[dataset_type][algo_name] = []
                            dataset_types[dataset_type][algo_name].append(
                                agent_results[algo_name]["success_rate"]
                            )
        
        # Calculate means
        x_labels = []
        pure_pp_means = []
        lns2_means = []
        hybrid_means = []
        marl_only_means = []
        
        for dtype in sorted(dataset_types.keys()):
            x_labels.append(dtype)
            
            algos = dataset_types[dtype]
            pure_pp_means.append(np.mean(algos.get("PurePP", [0])))
            lns2_means.append(np.mean(algos.get("MAPF-LNS2", [0])))
            hybrid_means.append(np.mean(algos.get("HybridMARL-LNS", [0])))
            marl_only_means.append(np.mean(algos.get("MARL-LNS-Only", [0])))
        
        x = np.arange(len(x_labels))
        width = 0.2
        
        ax.bar(x - 1.5*width, pure_pp_means, width, label='Pure PP', color='#1f77b4')
        ax.bar(x - 0.5*width, lns2_means, width, label='MAPF-LNS2', color='#ff7f0e')
        ax.bar(x + 0.5*width, hybrid_means, width, label='Hybrid MARL-LNS', color='#2ca02c')
        ax.bar(x + 1.5*width, marl_only_means, width, label='MARL-LNS Only', color='#d62728')
        
        ax.set_xlabel('Dataset Type')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Dataset Type')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images', 
                                 'figure_1_success_rate_by_dataset.png'), dpi=150)
        plt.close()
    
    def _plot_success_rate_by_agents(self):
        """Plot success rate vs agent count for each dataset type."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        dataset_types = list(set([k.split('/')[0] for k in self.all_results.keys()]))
        dataset_types.sort()
        
        for idx, dtype in enumerate(dataset_types[:8]):
            ax = axes[idx]
            
            # Collect data for this dataset type
            agent_data = {}
            for key, results in self.all_results.items():
                if key.startswith(dtype + '/'):
                    for agent_count, agent_results in results.get("algorithms", {}).items():
                        if agent_count not in agent_data:
                            agent_data[agent_count] = {a: [] for a in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]}
                        
                        for algo in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]:
                            if algo in agent_results:
                                agent_data[agent_count][algo].append(agent_results[algo]["success_rate"])
            
            if not agent_data:
                ax.set_visible(False)
                continue
            
            x = sorted(agent_data.keys())
            pure_pp_y = [np.mean(agent_data[a]["PurePP"]) for a in x]
            lns2_y = [np.mean(agent_data[a]["MAPF-LNS2"]) for a in x]
            hybrid_y = [np.mean(agent_data[a]["HybridMARL-LNS"]) for a in x]
            
            ax.plot(x, pure_pp_y, 'o-', label='Pure PP', color='#1f77b4')
            ax.plot(x, lns2_y, 's-', label='MAPF-LNS2', color='#ff7f0e')
            ax.plot(x, hybrid_y, '^-', label='Hybrid MARL-LNS', color='#2ca02c')
            
            ax.set_xlabel('Number of Agents')
            ax.set_ylabel('Success Rate')
            ax.set_title(dtype)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images',
                                 'figure_2_success_rate_by_agents.png'), dpi=150)
        plt.close()
    
    def _plot_soc_comparison(self):
        """Plot sum of costs comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect SOC data
        algo_data = {"PurePP": [], "MAPF-LNS2": [], "HybridMARL-LNS": []}
        
        for key, results in self.all_results.items():
            for agent_count, agent_results in results.get("algorithms", {}).items():
                for algo in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]:
                    if algo in agent_results:
                        soc = agent_results[algo]["avg_soc"]
                        if soc != float('inf'):
                            algo_data[algo].append(soc)
        
        # Box plot
        data_to_plot = [algo_data[a] for a in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]]
        labels = ["Pure PP", "MAPF-LNS2", "Hybrid MARL-LNS"]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Sum of Costs')
        ax.set_title('Solution Quality Comparison (Lower is Better)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images',
                                 'figure_3_soc_comparison.png'), dpi=150)
        plt.close()
    
    def _plot_runtime_comparison(self):
        """Plot runtime comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect runtime data
        algo_data = {"PurePP": [], "MAPF-LNS2": [], "HybridMARL-LNS": []}
        
        for key, results in self.all_results.items():
            for agent_count, agent_results in results.get("algorithms", {}).items():
                for algo in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]:
                    if algo in agent_results:
                        runtime = agent_results[algo]["avg_runtime"]
                        if runtime != float('inf'):
                            algo_data[algo].append(runtime)
        
        # Box plot
        data_to_plot = [algo_data[a] for a in ["PurePP", "MAPF-LNS2", "HybridMARL-LNS"]]
        labels = ["Pure PP", "MAPF-LNS2", "Hybrid MARL-LNS"]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Computational Efficiency Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images',
                                 'figure_4_runtime_comparison.png'), dpi=150)
        plt.close()
    
    def _plot_convergence_curve(self):
        """Plot convergence curve showing collision reduction over iterations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find a representative instance with collision history
        for key, results in self.all_results.items():
            for agent_count, agent_results in results.get("algorithms", {}).items():
                if "HybridMARL-LNS" in agent_results:
                    # Check if we have collision history
                    if "collisions_history" in agent_results["HybridMARL-LNS"]:
                        history = agent_results["HybridMARL-LNS"]["collisions_history"]
                        if history and len(history) > 1:
                            ax.plot(history, label=f'{key} ({agent_count} agents)')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Collisions')
        ax.set_title('Convergence: Collision Reduction Over Iterations')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images',
                                 'figure_5_convergence_curve.png'), dpi=150)
        plt.close()
    
    def _plot_collision_reduction(self):
        """Plot collision reduction percentage by algorithm."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate collision reduction
        reductions = {"PurePP": [], "MAPF-LNS2": [], "HybridMARL-LNS": []}
        
        for key, results in self.all_results.items():
            for agent_count, agent_results in results.get("algorithms", {}).items():
                if "PurePP" in agent_results:
                    initial_collisions = agent_results["PurePP"]["avg_collisions"]
                    
                    for algo in ["MAPF-LNS2", "HybridMARL-LNS"]:
                        if algo in agent_results and initial_collisions > 0:
                            final_collisions = agent_results[algo]["avg_collisions"]
                            reduction = (initial_collisions - final_collisions) / initial_collisions * 100
                            reductions[algo].append(max(0, reduction))
        
        # Bar plot
        x_labels = ["MAPF-LNS2", "Hybrid MARL-LNS"]
        means = [np.mean(reductions[a]) if reductions[a] else 0 for a in ["MAPF-LNS2", "HybridMARL-LNS"]]
        stds = [np.std(reductions[a]) if reductions[a] else 0 for a in ["MAPF-LNS2", "HybridMARL-LNS"]]
        
        x = np.arange(len(x_labels))
        ax.bar(x, means, yerr=stds, capsize=5, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        
        ax.set_ylabel('Collision Reduction (%)')
        ax.set_title('Average Collision Reduction vs Pure PP')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '..', 'report', 'images',
                                 'figure_6_collision_reduction.png'), dpi=150)
        plt.close()


def main():
    """Main entry point."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize runner
    data_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_002_20260518_050356/data'
    output_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_002_20260518_050356/outputs'
    
    runner = ExperimentRunner(data_dir, output_dir)
    
    # Run experiments
    print("Starting experiments...")
    print("This may take several minutes...")
    
    # Run on a subset of datasets for initial evaluation
    # Full experiments would take too long, so we focus on key datasets
    experiments_to_run = [
        ("random_small", "maps_50_10_10_0.175", [5, 10, 15]),
        ("random_small", "maps_55_10_10_0.175", [5, 10, 15]),
        ("random_medium", "maps_312_25_25_0.175", [10, 20, 30]),
        ("empty", "empty_maps_453_25_25", [10, 20, 30]),
        ("maze", "maze_maps_125_25_25", [10, 20, 30]),
        ("room", "room_maps_250_25_25", [10, 20, 30]),
    ]
    
    all_results = {}
    
    for dataset_name, subdir, agent_counts in experiments_to_run:
        try:
            results = runner.run_experiment(
                dataset_name, subdir, agent_counts,
                num_instances=3,  # Reduced for speed
                max_iterations=100,
                timeout=10.0
            )
            key = f"{dataset_name}/{subdir}"
            all_results[key] = results
            
            # Save intermediate results
            runner.all_results = all_results
            runner.save_results()
            
        except Exception as e:
            print(f"Error in {dataset_name}/{subdir}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate figures
    print("\nGenerating figures...")
    runner.all_results = all_results
    runner.generate_figures()
    
    print("\nExperiments complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: report/images/")


if __name__ == "__main__":
    main()
