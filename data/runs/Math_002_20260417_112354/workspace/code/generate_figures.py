"""
Visualization code for MAPF experiment results.
Generates all figures for the research report.
"""
import numpy as np
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'report', 'images')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
sns.set_style("whitegrid")


def load_results():
    """Load all experiment results."""
    with open(os.path.join(OUTPUT_DIR, 'experiment_results.json')) as f:
        results = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'summary_results.json')) as f:
        summary = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json')) as f:
        ablation = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'scaling_results.json')) as f:
        scaling = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'curve_results.json')) as f:
        curves = json.load(f)
    return results, summary, ablation, scaling, curves


def plot_map_examples():
    """Plot example maps for each type."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    map_configs = [
        ('Empty (25×25)', 'data/empty/empty_maps_453_25_25', 'eval_map_empty_1.npy'),
        ('Maze (25×25)', 'data/maze/maze_maps_125_25_25', 'eval_map_maze_1.npy'),
        ('Random Small (10×10)', 'data/random_small/maps_50_10_10_0.175', 'eval_map_1.npy'),
        ('Random Medium (25×25)', 'data/random_medium/maps_312_25_25_0.175', 'eval_map_1.npy'),
        ('Random Large (50×50)', 'data/random_large/maps_1250_50_50_0.175', 'eval_map_1.npy'),
        ('Room (25×25)', 'data/room/room_maps_250_25_25', 'eval_map_room_1.npy'),
        ('Warehouse (25×25)', 'data/warehouse/warehouse_maps_297_25_25', 'eval_map_warehouse_1.npy'),
        ('Random Small 2 (10×10)', 'data/maps_60_10_10_0.175', 'eval_map_1.npy'),
    ]
    
    cmap = ListedColormap(['white', '#333333'])
    
    for idx, (title, path, filename) in enumerate(map_configs):
        ax = axes[idx // 4][idx % 4]
        full_path = os.path.join(os.path.dirname(DATA_DIR), path, filename)
        
        try:
            grid = np.load(full_path)
            # Convert: 0 -> 0 (free, white), -1 -> 1 (obstacle, dark)
            display = np.where(grid == -1, 1, 0)
            ax.imshow(display, cmap=cmap, interpolation='nearest')
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.suptitle('Map Types in MAPF Benchmark', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'map_examples.png'), bbox_inches='tight')
    plt.close()
    print("Saved map_examples.png")


def plot_success_rate_comparison(summary):
    """Plot success rate comparison across map types."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    map_types = list(summary.keys())
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS (Ours)', 'Pure MARL']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#C9B1FF']
    
    x = np.arange(len(map_types))
    width = 0.15
    
    for i, (solver, label, color) in enumerate(zip(solvers, solver_labels, colors)):
        rates = [summary[mt].get(solver, {}).get('success_rate', 0) for mt in map_types]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, rates, width, label=label, color=color, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Map Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison Across Map Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([mt.replace('_', '\n') for mt in map_types], fontsize=10)
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'success_rate_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved success_rate_comparison.png")


def plot_collision_comparison(summary):
    """Plot average collision count comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    map_types = list(summary.keys())
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS (Ours)', 'Pure MARL']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#C9B1FF']
    
    x = np.arange(len(map_types))
    width = 0.15
    
    for i, (solver, label, color) in enumerate(zip(solvers, solver_labels, colors)):
        colls = [summary[mt].get(solver, {}).get('avg_collisions', 0) for mt in map_types]
        offset = (i - 2) * width
        ax.bar(x + offset, colls, width, label=label, color=color, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Map Type')
    ax.set_ylabel('Average Collision Count')
    ax.set_title('Average Collision Count Across Map Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([mt.replace('_', '\n') for mt in map_types], fontsize=10)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'collision_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved collision_comparison.png")


def plot_runtime_comparison(summary):
    """Plot runtime comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    map_types = list(summary.keys())
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS (Ours)', 'Pure MARL']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#C9B1FF']
    
    x = np.arange(len(map_types))
    width = 0.15
    
    for i, (solver, label, color) in enumerate(zip(solvers, solver_labels, colors)):
        times = [summary[mt].get(solver, {}).get('avg_runtime', 0) for mt in map_types]
        offset = (i - 2) * width
        ax.bar(x + offset, times, width, label=label, color=color, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Map Type')
    ax.set_ylabel('Average Runtime (seconds)')
    ax.set_title('Average Runtime Across Map Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([mt.replace('_', '\n') for mt in map_types], fontsize=10)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'runtime_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved runtime_comparison.png")


def plot_ablation_threshold(ablation):
    """Plot ablation study for switching threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    map_types = list(ablation.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(map_types)))
    
    for idx, mt in enumerate(map_types):
        thresholds = sorted(ablation[mt].keys(), key=float)
        srs = [ablation[mt][t]['success_rate'] for t in thresholds]
        colls = [ablation[mt][t]['avg_collisions'] for t in thresholds]
        thresh_vals = [float(t) for t in thresholds]
        
        axes[0].plot(thresh_vals, srs, 'o-', label=mt, color=colors[idx], linewidth=2, markersize=6)
        axes[1].plot(thresh_vals, colls, 's-', label=mt, color=colors[idx], linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Switching Threshold')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate vs. Switching Threshold', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Switching Threshold')
    axes[1].set_ylabel('Average Collisions')
    axes[1].set_title('Collision Count vs. Switching Threshold', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Ablation Study: MARL-LNS Switching Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'ablation_threshold.png'), bbox_inches='tight')
    plt.close()
    print("Saved ablation_threshold.png")


def plot_agent_scaling(scaling):
    """Plot agent density scaling results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    agent_counts = sorted([int(k) for k in scaling.keys()])
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS (Ours)']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
    markers = ['o', 's', '^', 'D']
    
    for i, (solver, label, color, marker) in enumerate(zip(solvers, solver_labels, colors, markers)):
        srs = [scaling[str(n)].get(solver, {}).get('success_rate', 0) for n in agent_counts]
        colls = [scaling[str(n)].get(solver, {}).get('avg_collisions', 0) for n in agent_counts]
        
        axes[0].plot(agent_counts, srs, f'{marker}-', label=label, color=color, linewidth=2, markersize=7)
        axes[1].plot(agent_counts, colls, f'{marker}-', label=label, color=color, linewidth=2, markersize=7)
    
    axes[0].set_xlabel('Number of Agents')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate vs. Agent Count\n(Random Medium 25×25)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Number of Agents')
    axes[1].set_ylabel('Average Collisions')
    axes[1].set_title('Collision Count vs. Agent Count\n(Random Medium 25×25)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Scalability Analysis: Agent Density', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'agent_scaling.png'), bbox_inches='tight')
    plt.close()
    print("Saved agent_scaling.png")


def plot_collision_curves(curves):
    """Plot collision reduction curves."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5*len(curves), 5))
    if len(curves) == 1:
        axes = [axes]
    
    colors = {'LNS2': '#96CEB4', 'MARL_LNS': '#FF6B6B'}
    labels = {'LNS2': 'LNS2', 'MARL_LNS': 'MARL-LNS (Ours)'}
    
    for idx, (mt, data) in enumerate(curves.items()):
        ax = axes[idx]
        
        for solver in ['LNS2_history', 'MARL_LNS_history']:
            history = data.get(solver, [])
            solver_key = solver.replace('_history', '')
            if history:
                ax.plot(range(len(history)), history, '-', 
                       label=labels.get(solver_key, solver_key),
                       color=colors.get(solver_key, 'gray'),
                       linewidth=2)
        
        ax.set_xlabel('LNS Iteration')
        ax.set_ylabel('Collision Count')
        ax.set_title(f'{mt.replace("_", " ").title()}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Collision Reduction Over LNS Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'collision_curves.png'), bbox_inches='tight')
    plt.close()
    print("Saved collision_curves.png")


def plot_heatmap_summary(summary):
    """Plot a heatmap of success rates."""
    map_types = list(summary.keys())
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS\n(Ours)', 'Pure MARL']
    
    data = np.zeros((len(map_types), len(solvers)))
    for i, mt in enumerate(map_types):
        for j, s in enumerate(solvers):
            data[i, j] = summary[mt].get(s, {}).get('success_rate', 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn', 
                xticklabels=solver_labels,
                yticklabels=[mt.replace('_', '\n') for mt in map_types],
                ax=ax, vmin=0, vmax=100, linewidths=0.5,
                cbar_kws={'label': 'Success Rate (%)'})
    
    ax.set_title('Success Rate Heatmap (%) Across Map Types and Solvers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'success_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Saved success_heatmap.png")


def plot_algorithm_overview():
    """Create an algorithm overview diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(7, 5.5, 'MARL-LNS Hybrid Algorithm Overview', fontsize=16, fontweight='bold',
            ha='center', va='center')
    
    # Initial paths box
    rect1 = mpatches.FancyBboxPatch((0.5, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='#E8F4FD', edgecolor='#2196F3', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 4.1, 'Initial Paths\n(Individual A*)', ha='center', va='center', fontsize=10)
    
    # MARL Phase box
    rect2 = mpatches.FancyBboxPatch((4, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='#FFE0E0', edgecolor='#F44336', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5.25, 4.1, 'MARL Phase\n(Collision Reduction)', ha='center', va='center', fontsize=10)
    
    # Switch decision
    diamond_x = [8.25, 9, 8.25, 7.5, 8.25]
    diamond_y = [4.7, 4.1, 3.5, 4.1, 4.7]
    ax.fill(diamond_x, diamond_y, facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
    ax.text(8.25, 4.1, 'Switch?', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # PP Phase box
    rect3 = mpatches.FancyBboxPatch((10, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect3)
    ax.text(11.25, 4.1, 'PP Phase\n(Final Refinement)', ha='center', va='center', fontsize=10)
    
    # Solution box
    rect4 = mpatches.FancyBboxPatch((5.5, 1.0), 3, 1.2, boxstyle="round,pad=0.1",
                                     facecolor='#F3E5F5', edgecolor='#9C27B0', linewidth=2)
    ax.add_patch(rect4)
    ax.text(7, 1.6, 'Collision-Free\nSolution', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(4, 4.1), xytext=(3, 4.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(7.5, 4.1), xytext=(6.5, 4.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(10, 4.1), xytext=(9, 4.1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.text(9.5, 4.4, 'Yes', fontsize=9, ha='center', color='#4CAF50')
    
    # Loop back arrow from MARL
    ax.annotate('', xy=(5.25, 3.5), xytext=(7.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(6.3, 2.8, 'No (continue MARL)', fontsize=8, ha='center', color='#F44336')
    
    # Arrow to solution
    ax.annotate('', xy=(7, 2.2), xytext=(11.25, 3.5),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=2))
    
    # LNS framework label
    rect_lns = mpatches.FancyBboxPatch((3.5, 2.5), 9.5, 3, boxstyle="round,pad=0.2",
                                        facecolor='none', edgecolor='#666', linewidth=1, linestyle='--')
    ax.add_patch(rect_lns)
    ax.text(8.25, 5.2, 'Large Neighborhood Search (LNS) Framework', fontsize=10,
            ha='center', va='center', style='italic', color='#666')
    
    # Key insight text
    ax.text(7, 0.3, 'Key: MARL reduces collisions rapidly in early iterations → PP provides efficient final refinement',
            ha='center', va='center', fontsize=10, style='italic', color='#555')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'algorithm_overview.png'), bbox_inches='tight')
    plt.close()
    print("Saved algorithm_overview.png")


def plot_makespan_comparison(summary):
    """Plot makespan and sum-of-costs comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    map_types = list(summary.keys())
    solvers = ['PP', 'PP_Restarts', 'LNS2', 'MARL_LNS', 'Pure_MARL']
    solver_labels = ['PP', 'PP+Restarts', 'LNS2', 'MARL-LNS\n(Ours)', 'Pure MARL']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B', '#C9B1FF']
    
    x = np.arange(len(map_types))
    width = 0.15
    
    for i, (solver, label, color) in enumerate(zip(solvers, solver_labels, colors)):
        makespan = [summary[mt].get(solver, {}).get('avg_makespan', 0) for mt in map_types]
        soc = [summary[mt].get(solver, {}).get('avg_sum_of_costs', 0) for mt in map_types]
        offset = (i - 2) * width
        axes[0].bar(x + offset, makespan, width, label=label.replace('\n', ' '), color=color, edgecolor='white')
        axes[1].bar(x + offset, soc, width, label=label.replace('\n', ' '), color=color, edgecolor='white')
    
    axes[0].set_xlabel('Map Type')
    axes[0].set_ylabel('Average Makespan')
    axes[0].set_title('Makespan Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([mt.replace('_', '\n') for mt in map_types], fontsize=8)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].set_xlabel('Map Type')
    axes[1].set_ylabel('Average Sum of Costs')
    axes[1].set_title('Sum of Costs Comparison', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([mt.replace('_', '\n') for mt in map_types], fontsize=8)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Solution Quality Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'solution_quality.png'), bbox_inches='tight')
    plt.close()
    print("Saved solution_quality.png")


def plot_phase_analysis(ablation):
    """Plot MARL vs PP iteration distribution."""
    fig, axes = plt.subplots(1, len(ablation), figsize=(4*len(ablation), 4))
    if len(ablation) == 1:
        axes = [axes]
    
    for idx, (mt, data) in enumerate(ablation.items()):
        ax = axes[idx]
        thresholds = sorted(data.keys(), key=float)
        thresh_vals = [float(t) for t in thresholds]
        marl_iters = [data[t]['avg_marl_iters'] for t in thresholds]
        pp_iters = [data[t]['avg_pp_iters'] for t in thresholds]
        
        ax.bar(thresh_vals, marl_iters, 0.08, label='MARL iters', color='#FF6B6B', alpha=0.8)
        ax.bar(thresh_vals, pp_iters, 0.08, bottom=marl_iters, label='PP iters', color='#96CEB4', alpha=0.8)
        
        ax.set_xlabel('Switch Threshold')
        ax.set_ylabel('Iterations')
        ax.set_title(mt.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('MARL vs PP Iteration Distribution by Threshold', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'phase_analysis.png'), bbox_inches='tight')
    plt.close()
    print("Saved phase_analysis.png")


if __name__ == '__main__':
    results, summary, ablation, scaling, curves = load_results()
    
    print("Generating figures...")
    plot_map_examples()
    plot_success_rate_comparison(summary)
    plot_collision_comparison(summary)
    plot_runtime_comparison(summary)
    plot_ablation_threshold(ablation)
    plot_agent_scaling(scaling)
    plot_collision_curves(curves)
    plot_heatmap_summary(summary)
    plot_algorithm_overview()
    plot_makespan_comparison(summary)
    plot_phase_analysis(ablation)
    
    print("\nAll figures generated successfully!")
