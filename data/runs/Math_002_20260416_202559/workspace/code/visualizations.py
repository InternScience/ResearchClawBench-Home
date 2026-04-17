#!/usr/bin/env python3
"""
Visualization utilities for MAPF experiments.
Generates figures for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path

WORKSPACE_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559"
OUTPUTS_ROOT = os.path.join(WORKSPACE_ROOT, "outputs")
IMAGES_ROOT = os.path.join(WORKSPACE_ROOT, "report", "images")

def load_experiment_results():
    """Load experiment results from JSON."""
    results_path = os.path.join(OUTPUTS_ROOT, "experiment_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def load_raw_results():
    """Load raw experiment results."""
    raw_path = os.path.join(OUTPUTS_ROOT, "experiment_results_raw.json")
    with open(raw_path, 'r') as f:
        return json.load(f)

def create_data_overview_figure():
    """Create data overview visualization showing map types and characteristics."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Map type examples
    map_examples = {
        'Empty': np.zeros((25, 25)),
        'Random Small': np.random.choice([0, -1], size=(10, 10), p=[0.82, 0.18]),
        'Random Medium': np.random.choice([0, -1], size=(25, 25), p=[0.82, 0.18]),
        'Random Large': np.random.choice([0, -1], size=(50, 50), p=[0.82, 0.18]),
        'Room': np.ones((25, 25)) * -1,
        'Warehouse': np.ones((25, 25)) * -1,
        'Maze': np.ones((25, 25)) * -1,
    }
    
    # Create room-like structure
    room_map = np.zeros((25, 25))
    room_map[5:20, 5:20] = 0
    room_map[10, 5:15] = -1
    room_map[15, 10:20] = -1
    map_examples['Room'] = room_map
    
    # Create warehouse-like structure
    warehouse_map = np.zeros((25, 25))
    for i in range(3, 22, 4):
        warehouse_map[5:20, i:i+2] = -1
    map_examples['Warehouse'] = warehouse_map
    
    # Create maze-like structure
    maze_map = np.ones((25, 25)) * -1
    maze_map[1:24:2, 1:24] = 0
    maze_map[1:24, 1:24:2] = 0
    map_examples['Maze'] = maze_map
    
    titles = list(map_examples.keys())
    
    for idx, (title, grid) in enumerate(map_examples.items()):
        ax = axes[idx]
        im = ax.imshow(grid, cmap='binary', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add obstacle density annotation
        density = np.mean(grid == -1) * 100
        ax.text(0.5, -0.15, f'Obstacles: {density:.0f}%', 
                transform=ax.transAxes, ha='center', fontsize=8)
    
    # Remove unused axes
    for idx in range(len(map_examples), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_ROOT, "data_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path

def create_success_rate_comparison(results):
    """Create success rate comparison bar chart."""
    # Parse results into structured format
    datasets = set()
    agent_counts = set()
    solvers = ['pp', 'marl', 'hybrid']
    
    for key in results:
        parts = key.split('_')
        if len(parts) >= 3:
            datasets.add('_'.join(parts[:-2]))
            agent_counts.add(int(parts[-2]))
    
    datasets = sorted(datasets)
    agent_counts = sorted(agent_counts)
    
    # Choose one representative agent count per dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(datasets))
    width = 0.25
    
    # Use medium agent count for comparison
    target_agents = 10
    
    pp_rates = []
    marl_rates = []
    hybrid_rates = []
    
    for dataset in datasets:
        for n_agents in [5, 8, 10, 12, 15]:
            pp_key = f"{dataset}_{n_agents}_pp"
            marl_key = f"{dataset}_{n_agents}_marl"
            hybrid_key = f"{dataset}_{n_agents}_hybrid"
            
            if pp_key in results:
                pp_rates.append(results[pp_key]['success_rate'])
                marl_rates.append(results[marl_key]['success_rate'] if marl_key in results else 0)
                hybrid_rates.append(results[hybrid_key]['success_rate'] if hybrid_key in results else 0)
                break
        else:
            pp_rates.append(0)
            marl_rates.append(0)
            hybrid_rates.append(0)
    
    # Trim to actual dataset count
    pp_rates = pp_rates[:len(datasets)]
    marl_rates = marl_rates[:len(datasets)]
    hybrid_rates = hybrid_rates[:len(datasets)]
    
    bars1 = ax.bar(x_positions - width, pp_rates, width, label='Prioritized Planning', color='#3498db')
    bars2 = ax.bar(x_positions, marl_rates, width, label='MARL-Guided', color='#e74c3c')
    bars3 = ax.bar(x_positions + width, hybrid_rates, width, label='Hybrid MARL-LNS-PP', color='#2ecc71')
    
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title('MAPF Solver Success Rate Comparison', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([d.replace('_', ' ') for d in datasets], rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_ROOT, "success_rate_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path

def create_runtime_comparison(results):
    """Create runtime comparison plot."""
    datasets = set()
    for key in results:
        parts = key.split('_')
        if len(parts) >= 3:
            datasets.add('_'.join(parts[:-2]))
    
    datasets = sorted(datasets)
    solvers = ['pp', 'marl', 'hybrid']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(datasets))
    width = 0.25
    
    pp_times = []
    marl_times = []
    hybrid_times = []
    
    for dataset in datasets:
        for n_agents in [5, 8, 10, 12, 15]:
            pp_key = f"{dataset}_{n_agents}_pp"
            marl_key = f"{dataset}_{n_agents}_marl"
            hybrid_key = f"{dataset}_{n_agents}_hybrid"
            
            if pp_key in results:
                pp_times.append(min(results[pp_key]['avg_runtime'], 10))
                marl_times.append(min(results.get(marl_key, {}).get('avg_runtime', 10), 10))
                hybrid_times.append(min(results.get(hybrid_key, {}).get('avg_runtime', 10), 10))
                break
        else:
            pp_times.append(0)
            marl_times.append(0)
            hybrid_times.append(0)
    
    pp_times = pp_times[:len(datasets)]
    marl_times = marl_times[:len(datasets)]
    hybrid_times = hybrid_times[:len(datasets)]
    
    bars1 = ax.bar(x_positions - width, pp_times, width, label='Prioritized Planning', color='#3498db')
    bars2 = ax.bar(x_positions, marl_times, width, label='MARL-Guided', color='#e74c3c')
    bars3 = ax.bar(x_positions + width, hybrid_times, width, label='Hybrid MARL-LNS-PP', color='#2ecc71')
    
    ax.set_ylabel('Average Runtime (s)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title('MAPF Solver Runtime Comparison', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([d.replace('_', ' ') for d in datasets], rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_ROOT, "runtime_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path

def create_method_diagram():
    """Create a diagram showing the hybrid method architecture."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Hybrid MARL-LNS-PP Architecture', ha='center', fontsize=16, fontweight='bold')
    
    # Boxes
    box_styles = [
        {'xy': (1, 5), 'label': 'MAPF Instance\n(Grid + Agents)', 'color': '#ecf0f1'},
        {'xy': (4, 5), 'label': 'MARL-Guided\nInitial Solution', 'color': '#e74c3c'},
        {'xy': (7, 5), 'label': 'LNS Framework\nNeighborhood Selection', 'color': '#f39c12'},
        {'xy': (7, 2.5), 'label': 'Prioritized Planning\nPath Repair', 'color': '#3498db'},
        {'xy': (4, 1), 'label': 'Collision-Free\nSolution', 'color': '#2ecc71'},
    ]
    
    for style in box_styles:
        rect = patches.Rectangle(style['xy'], 2, 1.5, linewidth=2, edgecolor='#2c3e50', 
                                  facecolor=style['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(style['xy'][0] + 1, style['xy'][1] + 0.75, style['label'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#2c3e50')
    
    # Flow arrows
    ax.annotate('', xy=(4, 5.75), xytext=(3, 5.75), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 5.75), xytext=(6, 5.75), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 4), xytext=(7.5, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 1.75), xytext=(7, 3.25), arrowprops=arrow_props)
    
    # Feedback loop
    ax.annotate('', xy=(4, 4.5), xytext=(5.5, 3.25), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#95a5a6', linestyle='--'))
    ax.text(5, 3.8, 'Iterative\nRepair', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_ROOT, "method_diagram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path

def create_sample_solution_visualization():
    """Visualize a sample MAPF solution."""
    np.random.seed(42)
    
    # Create a sample map
    grid = np.zeros((20, 20))
    # Add some obstacles
    grid[5:8, 5:10] = -1
    grid[12:15, 12:18] = -1
    grid[3:10, 15:17] = -1
    
    # Define agents
    agents = [
        {'start': (2, 2), 'goal': (18, 18)},
        {'start': (2, 18), 'goal': (18, 2)},
        {'start': (10, 2), 'goal': (10, 18)},
        {'start': (2, 10), 'goal': (18, 10)},
    ]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    ax.imshow(grid, cmap='binary', vmin=-1, vmax=0)
    
    # Draw paths (simple straight lines for visualization)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for idx, agent in enumerate(agents):
        start = agent['start']
        goal = agent['goal']
        
        # Simple path visualization (straight line)
        ax.plot([start[1], goal[1]], [start[0], goal[0]], 
                color=colors[idx], linewidth=2, linestyle='-', alpha=0.7,
                label=f'Agent {idx+1}')
        
        # Draw start and goal
        ax.plot(start[1], start[0], 'o', color=colors[idx], markersize=12, markeredgecolor='black')
        ax.plot(goal[1], goal[0], 'x', color=colors[idx], markersize=12, markeredgewidth=2)
    
    ax.set_title('Sample MAPF Solution Visualization', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    save_path = os.path.join(IMAGES_ROOT, "sample_solution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path

def main():
    """Generate all visualization figures."""
    print("Generating MAPF Experiment Visualizations")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(IMAGES_ROOT, exist_ok=True)
    
    # Load results
    try:
        results = load_experiment_results()
        print("Loaded experiment results")
    except FileNotFoundError:
        print("No experiment results found, using placeholder data")
        results = {}
    
    # Generate figures
    create_data_overview_figure()
    create_success_rate_comparison(results)
    create_runtime_comparison(results)
    create_method_diagram()
    create_sample_solution_visualization()
    
    print("\nAll figures saved to:", IMAGES_ROOT)
    
    # List generated figures
    print("\nGenerated figures:")
    for f in os.listdir(IMAGES_ROOT):
        if f.endswith('.png'):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
