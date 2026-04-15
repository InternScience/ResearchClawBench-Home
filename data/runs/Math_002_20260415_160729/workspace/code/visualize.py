"""
Visualization utilities for MAPF
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import os
from typing import List, Dict, Tuple
import pickle

from mapf_utils import MAPFMap, MAPFInstance, Path


def visualize_map(grid: np.ndarray, title: str = "Map", save_path: str = None):
    """Visualize a single map."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot obstacles
    obstacle_mask = (grid == -1)
    ax.imshow(obstacle_mask, cmap='binary', interpolation='nearest')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_instance(instance: MAPFInstance, title: str = "MAPF Instance", 
                       save_path: str = None):
    """Visualize a MAPF instance with starts and goals."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    grid = instance.map.grid
    height, width = grid.shape
    
    # Plot obstacles
    for i in range(height):
        for j in range(width):
            if grid[i, j] == -1:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                         linewidth=1, edgecolor='black', 
                                         facecolor='gray')
                ax.add_patch(rect)
    
    # Plot starts and goals
    colors = plt.cm.tab20(np.linspace(0, 1, instance.num_agents))
    
    for i, (start, goal) in enumerate(zip(instance.starts, instance.goals)):
        color = colors[i % len(colors)]
        
        # Start position (circle)
        circle = Circle((start[1], start[0]), 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(start[1], start[0], f'S{i}', ha='center', va='center', 
                fontsize=8, fontweight='bold')
        
        # Goal position (star)
        ax.plot(goal[1], goal[0], '*', color=color, markersize=15, markeredgecolor='black')
        ax.text(goal[1], goal[0], f'G{i}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold', color=color)
    
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_solution(instance: MAPFInstance, solution: List[Path], 
                       title: str = "MAPF Solution", save_path: str = None,
                       show_paths: bool = True):
    """Visualize a solution with paths."""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    grid = instance.map.grid
    height, width = grid.shape
    
    # Plot obstacles
    for i in range(height):
        for j in range(width):
            if grid[i, j] == -1:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                         linewidth=1, edgecolor='black', 
                                         facecolor='gray')
                ax.add_patch(rect)
    
    colors = plt.cm.tab20(np.linspace(0, 1, instance.num_agents))
    
    # Plot paths
    if show_paths:
        for i, path in enumerate(solution):
            color = colors[i % len(colors)]
            positions = path.positions
            
            # Plot path line
            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]
            ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=2)
            
            # Plot direction arrows
            for j in range(len(positions) - 1):
                dx = positions[j + 1][1] - positions[j][1]
                dy = positions[j + 1][0] - positions[j][0]
                if dx != 0 or dy != 0:
                    ax.arrow(positions[j][1], positions[j][0], dx * 0.6, dy * 0.6,
                            head_width=0.2, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Plot starts and goals
    for i, (start, goal) in enumerate(zip(instance.starts, instance.goals)):
        color = colors[i % len(colors)]
        
        circle = Circle((start[1], start[0]), 0.25, color=color, alpha=0.8)
        ax.add_patch(circle)
        
        ax.plot(goal[1], goal[0], '*', color=color, markersize=12, markeredgecolor='black')
    
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_plots(results: Dict, output_dir: str):
    """Create comparison plots from results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Success Rate Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    datasets = list(results.keys())
    
    for idx, dataset in enumerate(datasets[:8]):  # Max 8 datasets
        ax = axes[idx]
        
        if dataset not in results:
            continue
        
        agent_counts = sorted([int(k) for k in results[dataset].keys()])
        
        lns_success = []
        pp_success = []
        
        for agent_count in agent_counts:
            agent_str = str(agent_count)
            if agent_str in results[dataset]:
                lns_success.append(results[dataset][agent_str]['lns_marl']['success_rate'] * 100)
                pp_success.append(results[dataset][agent_str]['baseline_pp']['success_rate'] * 100)
            else:
                lns_success.append(0)
                pp_success.append(0)
        
        x = np.arange(len(agent_counts))
        width = 0.35
        
        ax.bar(x - width/2, lns_success, width, label='LNS-MARL', color='steelblue')
        ax.bar(x + width/2, pp_success, width, label='Baseline-PP', color='coral')
        
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_counts)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Runtime Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets[:8]):
        ax = axes[idx]
        
        if dataset not in results:
            continue
        
        agent_counts = sorted([int(k) for k in results[dataset].keys()])
        
        lns_time = []
        pp_time = []
        
        for agent_count in agent_counts:
            agent_str = str(agent_count)
            if agent_str in results[dataset]:
                lns_time.append(results[dataset][agent_str]['lns_marl']['avg_time'])
                pp_time.append(results[dataset][agent_str]['baseline_pp']['avg_time'])
            else:
                lns_time.append(0)
                pp_time.append(0)
        
        ax.plot(agent_counts, lns_time, 'o-', label='LNS-MARL', color='steelblue', linewidth=2)
        ax.plot(agent_counts, pp_time, 's-', label='Baseline-PP', color='coral', linewidth=2)
        
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Average Runtime (s)')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Collision Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets[:8]):
        ax = axes[idx]
        
        if dataset not in results:
            continue
        
        agent_counts = sorted([int(k) for k in results[dataset].keys()])
        
        lns_collisions = []
        pp_collisions = []
        
        for agent_count in agent_counts:
            agent_str = str(agent_count)
            if agent_str in results[dataset]:
                lns_collisions.append(results[dataset][agent_str]['lns_marl']['avg_collisions'])
                pp_collisions.append(results[dataset][agent_str]['baseline_pp']['avg_collisions'])
            else:
                lns_collisions.append(0)
                pp_collisions.append(0)
        
        x = np.arange(len(agent_counts))
        width = 0.35
        
        ax.bar(x - width/2, lns_collisions, width, label='LNS-MARL', color='steelblue')
        ax.bar(x + width/2, pp_collisions, width, label='Baseline-PP', color='coral')
        
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Average Collisions')
        ax.set_title(f'{dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_counts)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Overall Performance Summary
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_lns_success = []
    all_pp_success = []
    labels = []
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        for agent_str in results[dataset]:
            all_lns_success.append(results[dataset][agent_str]['lns_marl']['success_rate'] * 100)
            all_pp_success.append(results[dataset][agent_str]['baseline_pp']['success_rate'] * 100)
            labels.append(f"{dataset}\n{agent_str} agents")
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.barh(x - width/2, all_lns_success, width, label='LNS-MARL', color='steelblue')
    ax.barh(x + width/2, all_pp_success, width, label='Baseline-PP', color='coral')
    
    ax.set_xlabel('Success Rate (%)')
    ax.set_title('Overall Success Rate Comparison')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")


def visualize_map_types(data_dir: str, output_dir: str):
    """Visualize sample maps from each dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['empty', 'maze', 'random_small', 'random_medium', 'random_large', 'room', 'warehouse']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        dataset_path = os.path.join(data_dir, dataset)
        
        if not os.path.exists(dataset_path):
            continue
        
        # Find a map file
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.npy'):
                    map_path = os.path.join(root, file)
                    grid = np.load(map_path, allow_pickle=True)
                    
                    # Plot
                    obstacle_mask = (grid == -1)
                    ax.imshow(obstacle_mask, cmap='binary', interpolation='nearest')
                    ax.set_title(f'{dataset}\n({grid.shape[0]}x{grid.shape[1]})')
                    ax.axis('off')
                    break
            break
    
    # Hide extra subplot
    if len(datasets) < len(axes):
        for idx in range(len(datasets), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_types_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Map overview saved to {output_dir}")


if __name__ == '__main__':
    # Load results if they exist
    results_path = 'outputs/results_summary.json'
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        create_comparison_plots(results, 'report/images')
    
    # Visualize map types
    visualize_map_types('data', 'report/images')
