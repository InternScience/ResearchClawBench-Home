"""
Generate figures for the MAPF research report.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'report', 'images')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_results():
    """Load the evaluation results."""
    results_path = os.path.join(OUTPUT_DIR, 'focused_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    
    summary_path = os.path.join(OUTPUT_DIR, 'focused_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return None


def fig1_success_rate_comparison():
    """Figure 1: Success rate comparison across datasets."""
    summary_path = os.path.join(OUTPUT_DIR, 'focused_summary.json')
    with open(summary_path) as f:
        summary = json.load(f)
    
    datasets = sorted(summary.keys())
    solvers = ['PP', 'LNS', 'MARL', 'Hybrid']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    patterns = ['/', '\\', '.', '']
    
    for i, solver in enumerate(solvers):
        rates = [summary[ds][solver]['success_rate'] * 100 for ds in datasets]
        bars = ax.bar(x + i * width, rates, width, label=solver, color=colors[i], 
                      edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate Comparison Across MAPF Datasets', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig1_success_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_success_rate.png")


def fig2_collision_reduction():
    """Figure 2: Colliding pairs comparison (lower is better)."""
    summary_path = os.path.join(OUTPUT_DIR, 'focused_summary.json')
    with open(summary_path) as f:
        summary = json.load(f)
    
    datasets = sorted(summary.keys())
    solvers = ['PP', 'LNS', 'MARL', 'Hybrid']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, solver in enumerate(solvers):
        cps = [summary[ds][solver]['avg_cp'] for ds in datasets]
        bars = ax.bar(x + i * width, cps, width, label=solver, color=colors[i], 
                      edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Average Colliding Pairs', fontweight='bold')
    ax.set_title('Collision Reduction: Colliding Pairs by Solver', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig2_collision_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_collision_reduction.png")


def fig3_runtime_comparison():
    """Figure 3: Runtime comparison across solvers."""
    summary_path = os.path.join(OUTPUT_DIR, 'focused_summary.json')
    with open(summary_path) as f:
        summary = json.load(f)
    
    datasets = sorted(summary.keys())
    solvers = ['PP', 'LNS', 'MARL', 'Hybrid']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, solver in enumerate(solvers):
        times = [summary[ds][solver]['avg_time'] for ds in datasets]
        bars = ax.bar(x + i * width, times, width, label=solver, color=colors[i], 
                      edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontweight='bold')
    ax.set_title('Computational Efficiency: Runtime by Solver', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig3_runtime.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_runtime.png")


def fig4_hybrid_breakdown():
    """Figure 4: Hybrid approach breakdown - MARL vs LNS contributions."""
    results_path = os.path.join(OUTPUT_DIR, 'focused_results.json')
    with open(results_path) as f:
        results = json.load(f)
    
    # Group by dataset
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r['dataset']].append(r)
    
    datasets = sorted(by_dataset.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets):
        if idx >= 6:
            break
        ax = axes[idx]
        
        ds_results = by_dataset[ds]
        
        labels = []
        pp_cp = []
        marl_cp = []
        hyb_cp = []
        lns_cp = []
        
        for r in ds_results:
            labels.append(f"n={r['n_agents']}")
            pp_cp.append(r['PP']['cp'])
            marl_cp.append(r['MARL']['cp'])
            hyb_cp.append(r['Hybrid']['cp'])
            lns_cp.append(r['LNS']['cp'])
        
        x = np.arange(len(labels))
        width = 0.2
        
        ax.bar(x - 1.5*width, pp_cp, width, label='PP', color='#e74c3c', alpha=0.8)
        ax.bar(x - 0.5*width, marl_cp, width, label='MARL', color='#2ecc71', alpha=0.8)
        ax.bar(x + 0.5*width, lns_cp, width, label='LNS', color='#3498db', alpha=0.8)
        ax.bar(x + 1.5*width, hyb_cp, width, label='Hybrid', color='#9b59b6', alpha=0.8)
        
        ax.set_title(ds.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        ax.set_ylabel('Colliding Pairs')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8)
    
    fig.suptitle('Collision Reduction: Per-Instance Breakdown', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig4_per_instance_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_per_instance_breakdown.png")


def fig5_solution_quality():
    """Figure 5: Solution quality (sum of costs) comparison."""
    summary_path = os.path.join(OUTPUT_DIR, 'focused_summary.json')
    with open(summary_path) as f:
        summary = json.load(f)
    
    datasets = sorted(summary.keys())
    solvers = ['PP', 'LNS', 'MARL', 'Hybrid']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, solver in enumerate(solvers):
        socs = [summary[ds][solver]['avg_soc'] for ds in datasets]
        bars = ax.bar(x + i * width, socs, width, label=solver, color=colors[i], 
                      edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Average Sum of Costs', fontweight='bold')
    ax.set_title('Solution Quality: Sum of Costs by Solver', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig5_solution_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_solution_quality.png")


def fig6_map_visualizations():
    """Figure 6: Visualize example maps from each dataset."""
    map_files = {
        'random_small': 'random_small/maps_50_10_10_0.175/eval_map_1.npy',
        'random_medium': 'random_medium/maps_312_25_25_0.175/eval_map_1.npy',
        'maze': 'maze/maze_maps_125_25_25/eval_map_maze_1.npy',
        'room': 'room/room_maps_250_25_25/eval_map_room_1.npy',
        'warehouse': 'warehouse/warehouse_maps_266_25_25/eval_map_warehouse_1.npy',
        'empty': 'empty/empty_maps_453_25_25/eval_map_empty_1.npy',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, rel_path) in enumerate(sorted(map_files.items())):
        if idx >= 6:
            break
        
        map_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(map_path):
            continue
        
        grid = np.load(map_path, allow_pickle=True)
        
        ax = axes[idx]
        cmap = plt.cm.RdYlBu_r
        im = ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=-1, vmax=0)
        
        # Count free and obstacle cells
        free = np.sum(grid == 0)
        obs = np.sum(grid == -1)
        
        ax.set_title(f"{name.replace('_', ' ').title()}\n{grid.shape[0]}×{grid.shape[1]}, {obs} obs, {free} free", 
                     fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle('Benchmark Map Visualizations', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig6_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig6_maps.png")


def fig7_algorithm_flow():
    """Figure 7: Algorithm flow diagram of the Hybrid MARL-LNS approach."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw boxes
    boxes = [
        (1, 4.5, 'MAPF Instance\n(Grid + Agents)', '#e8f5e9'),
        (1, 2.5, 'Phase 1: MARL\nQ-Learning Coordination', '#fff3e0'),
        (1, 0.5, 'Phase 2: LNS\nNeighborhood Repair', '#e3f2fd'),
        (4.5, 4.5, 'Prioritized\nPlanning (PP)', '#fce4ec'),
        (4.5, 2.5, 'Collision-Aware\nPath Generation', '#f3e5f5'),
        (4.5, 0.5, 'Iterative Collision\nResolution', '#e0f2f1'),
        (8, 3, 'Collision-Free\nSolution', '#c8e6c9'),
    ]
    
    for x, y, text, color in boxes:
        rect = Rectangle((x, y), 2.5, 1.4, fill=True, facecolor=color, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 1.25, y + 0.7, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw arrows
    arrows = [
        (3.5, 4.5, 4.5, 4.5),  # Instance -> PP
        (4.5, 3.8, 3.5, 3.8),  # PP -> MARL
        (3.5, 2.5, 4.5, 2.5),  # MARL -> Collision-Aware
        (4.5, 1.8, 3.5, 1.8),  # Collision-Aware -> LNS
        (3.5, 0.5, 4.5, 0.5),  # LNS -> Iterative
        (7, 4.2, 8, 3.7),      # Iterative -> Solution
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_title('Hybrid MARL-LNS Algorithm Workflow', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fig7_algorithm_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig7_algorithm_flow.png")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    fig1_success_rate_comparison()
    fig2_collision_reduction()
    fig3_runtime_comparison()
    fig4_hybrid_breakdown()
    fig5_solution_quality()
    fig6_map_visualizations()
    fig7_algorithm_flow()
    
    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
