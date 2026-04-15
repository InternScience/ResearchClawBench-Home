"""
Create demo visualizations for the report
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrowPatch
import os
import sys

sys.path.insert(0, 'code')
from mapf_utils import MAPFMap, MAPFInstance, Path
from lns_marl import LNS_MARL, BaselinePP


def visualize_map_instance_solution(instance, solution, title, save_path):
    """Create detailed visualization of instance and solution."""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    grid = instance.map.grid
    height, width = grid.shape
    
    # Plot grid background
    for i in range(height):
        for j in range(width):
            if grid[i, j] == -1:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                         linewidth=1, edgecolor='black', 
                                         facecolor='gray', alpha=0.7)
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                         linewidth=0.5, edgecolor='lightgray', 
                                         facecolor='white', alpha=0.3)
                ax.add_patch(rect)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, instance.num_agents))
    
    # Plot paths with arrows
    for i, path in enumerate(solution):
        color = colors[i % len(colors)]
        positions = path.positions
        
        # Plot path line
        ys = [p[0] for p in positions]
        xs = [p[1] for p in positions]
        ax.plot(xs, ys, '-', color=color, alpha=0.6, linewidth=2.5, zorder=2)
        
        # Plot direction arrows
        for j in range(min(len(positions) - 1, 20)):  # Limit arrows
            dx = positions[j + 1][1] - positions[j][1]
            dy = positions[j + 1][0] - positions[j][0]
            if dx != 0 or dy != 0:
                ax.arrow(positions[j][1], positions[j][0], dx * 0.5, dy * 0.5,
                        head_width=0.25, head_length=0.15, fc=color, ec=color, 
                        alpha=0.8, zorder=3, linewidth=1.5)
    
    # Plot starts (circles with agent numbers)
    for i, start in enumerate(instance.starts):
        color = colors[i % len(colors)]
        circle = Circle((start[1], start[0]), 0.35, color=color, alpha=0.9, zorder=4)
        ax.add_patch(circle)
        ax.text(start[1], start[0], f'{i}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white', zorder=5)
    
    # Plot goals (stars)
    for i, goal in enumerate(instance.goals):
        color = colors[i % len(colors)]
        ax.plot(goal[1], goal[0], '*', color=color, markersize=20, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=4)
        ax.text(goal[1], goal[0] - 0.5, f'G{i}', ha='center', va='top', 
                fontsize=9, fontweight='bold', color=color, zorder=5)
    
    # Add grid
    ax.set_xlim(-0.7, width - 0.3)
    ax.set_ylim(height - 0.3, -0.7)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Start', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=15, label='Goal', linestyle='None', markeredgecolor='black'),
        Line2D([0], [0], color='gray', linewidth=2, label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_map_file(data_dir, dataset, filename):
    for root, dirs, files in os.walk(os.path.join(data_dir, dataset)):
        if filename in files:
            return os.path.join(root, filename)
    return None


def main():
    os.makedirs('report/images', exist_ok=True)
    
    # Create demo on random_medium
    print("Creating demo visualization...")
    
    map_path = find_map_file('data', 'random_medium', 'eval_map_1.npy')
    if map_path is None:
        map_path = find_map_file('data', 'random_small', 'eval_map_1.npy')
    
    grid = np.load(map_path, allow_pickle=True)
    map_obj = MAPFMap(grid)
    
    num_agents = 6
    instance = MAPFInstance.generate_random(map_obj, num_agents, seed=42)
    
    # Run LNS-MARL
    lns_solver = LNS_MARL(instance, seed=42)
    lns_solution, lns_stats = lns_solver.solve(max_iterations=30, time_limit=5.0)
    
    visualize_map_instance_solution(
        instance, lns_solution, 
        f'LNS-MARL Solution ({num_agents} agents, Success: {lns_stats["success"]})',
        'report/images/demo_solution.png'
    )
    
    print(f"Demo stats:")
    print(f"  Success: {lns_stats['success']}")
    print(f"  Initial collisions: {lns_stats['initial_collisions']}")
    print(f"  Final collisions: {lns_stats['final_collisions']}")
    print(f"  Runtime: {lns_stats['time']:.3f}s")
    print(f"Saved: report/images/demo_solution.png")


if __name__ == '__main__':
    main()
