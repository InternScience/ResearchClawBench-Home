"""
Generate figures for the research report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys

sys.path.insert(0, '.')

def load_results():
    """Load experiment results."""
    with open('outputs/experiment_results.json', 'r') as f:
        return json.load(f)

def load_convergence():
    """Load convergence histories."""
    histories = {}
    for solver in ['PP', 'LNS', 'MARL-LNS']:
        path = f'outputs/convergence_{solver}.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                histories[solver] = json.load(f)
        else:
            histories[solver] = []
    return histories


def fig1_data_overview(results, save_dir='report/images'):
    """Figure 1: Data overview - map types and obstacle densities."""
    datasets = sorted(set(r['dataset'] for r in results))
    
    # Count maps and agents per dataset
    data_info = {}
    for d in datasets:
        dr = [r for r in results if r['dataset'] == d]
        n_agents_list = sorted(set(r['n_agents'] for r in dr))
        # Map sizes are known from dataset names; use reasonable defaults
        map_size_defaults = {
            'random_small': (10, 10), 'maps_60_10_10_0.175': (10, 10),
            'random_medium': (25, 25), 'empty': (25, 25), 'maze': (25, 25),
            'room': (25, 25), 'warehouse': (25, 25), 'random_large': (50, 50),
        }
        ms = map_size_defaults.get(d, (25, 25))
        
        data_info[d] = {
            'n_maps': len(set((r['map_idx'], r['n_agents']) for r in dr)),
            'agent_counts': n_agents_list,
            'map_size': list(ms),
            'avg_obstacle': 17.5 if '0.175' in d or d in ['random_small', 'random_medium', 'random_large', 'maps_60_10_10_0.175'] else 0.0,
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Map sizes and obstacle densities
    names = list(data_info.keys())
    sizes = [data_info[n]['map_size'] for n in names]
    obstacles = [data_info[n]['avg_obstacle'] * 100 for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, [s[0]*s[1] for s in sizes], width, label='Map Size (cells)', 
                        color='#2196F3', alpha=0.8)
    axes[0].set_ylabel('Map Size (cells)', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].set_title('Dataset Overview: Map Sizes', fontsize=12)
    
    ax2 = axes[0].twinx()
    bars2 = ax2.bar(x + width/2, obstacles, width, label='Obstacle Density (%)', 
                    color='#FF5722', alpha=0.8)
    ax2.set_ylabel('Obstacle Density (%)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Right: Agent counts per dataset
    max_agents = [max(data_info[n]['agent_counts']) for n in names]
    min_agents = [min(data_info[n]['agent_counts']) for n in names]
    
    axes[1].barh(names, [ma - mi for ma, mi in zip(max_agents, min_agents)], 
                 left=min_agents, color='#4CAF50', alpha=0.8, height=0.5)
    axes[1].set_xlabel('Number of Agents', fontsize=11)
    axes[1].set_title('Agent Count Range per Dataset', fontsize=12)
    
    for i, (mi, ma) in enumerate(zip(min_agents, max_agents)):
        axes[1].text(mi + (ma-mi)/2, i, f'{mi}-{ma}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_data_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig1_data_overview.png")


def fig2_success_rates(results, save_dir='report/images'):
    """Figure 2: Success rates by dataset and agent count."""
    datasets = sorted(set(r['dataset'] for r in results))
    solvers = ['PP', 'LNS', 'MARL-LNS']
    colors = {'PP': '#2196F3', 'LNS': '#FF9800', 'MARL-LNS': '#E91E63'}
    
    # Compute success rates per (dataset, solver)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Overall success rate by dataset
    for solver in solvers:
        rates = []
        for d in datasets:
            sr = [r for r in results if r['dataset'] == d and r['solver'] == solver]
            if sr:
                rate = sum(1 for r in sr if r['success']) / len(sr) * 100
            else:
                rate = 0
            rates.append(rate)
        
        axes[0].plot(range(len(datasets)), rates, 'o-', label=solver, 
                    color=colors[solver], linewidth=2, markersize=8)
    
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Success Rate (%)', fontsize=11)
    axes[0].set_title('Success Rate by Dataset', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Success rate by agent count (aggregated)
    agent_counts = sorted(set(r['n_agents'] for r in results))
    
    for solver in solvers:
        rates = []
        for na in agent_counts:
            sr = [r for r in results if r['n_agents'] == na and r['solver'] == solver]
            if sr:
                rate = sum(1 for r in sr if r['success']) / len(sr) * 100
            else:
                rate = 0
            rates.append(rate)
        
        axes[1].plot(agent_counts, rates, 'o-', label=solver, 
                    color=colors[solver], linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Number of Agents', fontsize=11)
    axes[1].set_ylabel('Success Rate (%)', fontsize=11)
    axes[1].set_title('Success Rate vs. Agent Count', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_success_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig2_success_rates.png")


def fig3_runtime_comparison(results, save_dir='report/images'):
    """Figure 3: Runtime comparison between solvers."""
    datasets = sorted(set(r['dataset'] for r in results))
    solvers = ['PP', 'LNS', 'MARL-LNS']
    colors = {'PP': '#2196F3', 'LNS': '#FF9800', 'MARL-LNS': '#E91E63'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Average runtime by dataset
    for solver in solvers:
        runtimes = []
        for d in datasets:
            sr = [r for r in results if r['dataset'] == d and r['solver'] == solver]
            if sr:
                avg_rt = np.mean([r['runtime'] for r in sr])
            else:
                avg_rt = 0
            runtimes.append(avg_rt)
        
        axes[0].plot(range(len(datasets)), runtimes, 'o-', label=solver, 
                    color=colors[solver], linewidth=2, markersize=8)
    
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Average Runtime (s)', fontsize=11)
    axes[0].set_title('Average Runtime by Dataset', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Runtime distribution (box plot)
    data_for_box = []
    labels_for_box = []
    for solver in solvers:
        sr = [r for r in results if r['solver'] == solver]
        if sr:
            data_for_box.append([r['runtime'] for r in sr])
            labels_for_box.append(solver)
    
    bp = axes[1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    
    for patch, solver in zip(bp['boxes'], solvers):
        patch.set_facecolor(colors[solver])
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Runtime (s)', fontsize=11)
    axes[1].set_title('Runtime Distribution Across All Tests', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_runtime_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig3_runtime_comparison.png")


def fig4_sum_of_costs(results, save_dir='report/images'):
    """Figure 4: Sum-of-costs comparison."""
    datasets = sorted(set(r['dataset'] for r in results))
    solvers = ['PP', 'LNS', 'MARL-LNS']
    colors = {'PP': '#2196F3', 'LNS': '#FF9800', 'MARL-LNS': '#E91E63'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Average SOC by dataset
    for solver in solvers:
        socs = []
        for d in datasets:
            sr = [r for r in results if r['dataset'] == d and r['solver'] == solver]
            if sr:
                avg_soc = np.mean([r['sum_of_costs'] for r in sr])
            else:
                avg_soc = 0
            socs.append(avg_soc)
        
        axes[0].plot(range(len(datasets)), socs, 'o-', label=solver, 
                    color=colors[solver], linewidth=2, markersize=8)
    
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Average Sum-of-Costs', fontsize=11)
    axes[0].set_title('Solution Quality (Sum-of-Costs) by Dataset', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: SOC vs agent count
    agent_counts = sorted(set(r['n_agents'] for r in results))
    
    for solver in solvers:
        socs = []
        for na in agent_counts:
            sr = [r for r in results if r['n_agents'] == na and r['solver'] == solver]
            if sr:
                avg_soc = np.mean([r['sum_of_costs'] for r in sr])
            else:
                avg_soc = 0
            socs.append(avg_soc)
        
        axes[1].plot(agent_counts, socs, 'o-', label=solver, 
                    color=colors[solver], linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Number of Agents', fontsize=11)
    axes[1].set_ylabel('Average Sum-of-Costs', fontsize=11)
    axes[1].set_title('Solution Quality vs. Agent Count', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_sum_of_costs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig4_sum_of_costs.png")


def fig5_convergence(histories, save_dir='report/images'):
    """Figure 5: Convergence curves showing collision reduction over iterations."""
    solvers = ['LNS', 'MARL-LNS']
    colors = {'LNS': '#FF9800', 'MARL-LNS': '#E91E63'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Aggregate convergence by solver
    for solver in solvers:
        hist_list = histories.get(solver, [])
        if not hist_list:
            continue
        
        # Normalize iteration counts and average collisions
        max_iter = 0
        all_iters = []
        all_colls = []
        
        for h in hist_list:
            iters = h.get('iterations', [])
            colls = h.get('collisions', [])
            if len(iters) > 1 and len(colls) > 1:
                all_iters.append(iters)
                all_colls.append(colls)
                max_iter = max(max_iter, max(iters))
        
        if not all_iters:
            continue
        
        # Interpolate to common iteration grid
        grid_size = min(max_iter, 100)
        grid = np.linspace(0, max_iter, grid_size)
        avg_collisions = np.zeros(grid_size)
        counts = np.zeros(grid_size)
        
        for iters, colls in zip(all_iters, all_colls):
            iters_arr = np.array(iters)
            colls_arr = np.array(colls)
            
            for gi, g in enumerate(grid):
                # Find closest iteration
                idx = np.searchsorted(iters_arr, g)
                if idx == 0:
                    avg_collisions[gi] += colls_arr[0]
                    counts[gi] += 1
                elif idx >= len(iters_arr):
                    avg_collisions[gi] += colls_arr[-1]
                    counts[gi] += 1
                else:
                    # Linear interpolation
                    t = (g - iters_arr[idx-1]) / (iters_arr[idx] - iters_arr[idx-1])
                    val = colls_arr[idx-1] + t * (colls_arr[idx] - colls_arr[idx-1])
                    avg_collisions[gi] += val
                    counts[gi] += 1
        
        avg_collisions = np.where(counts > 0, avg_collisions / counts, np.nan)
        
        axes[0].plot(grid, avg_collisions, '-', label=solver, 
                    color=colors[solver], linewidth=2)
    
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Average Collisions', fontsize=11)
    axes[0].set_title('Convergence: Collision Reduction Over Iterations', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Phase transition visualization for MARL-LNS
    marl_hist = histories.get('MARL-LNS', [])
    phase_data = {'marl_lns': [], 'pp_cleanup': []}
    
    for h in marl_hist[:20]:  # Sample first 20
        phases = h.get('phase', [])
        colls = h.get('collisions', [])
        if len(phases) > 1 and len(colls) > 1:
            for i, (p, c) in enumerate(zip(phases, colls)):
                if p in phase_data:
                    phase_data[p].append(c)
    
    for phase_name, values in phase_data.items():
        if values:
            axes[1].hist(values, bins=20, alpha=0.6, label=phase_name.replace('_', ' ').title(),
                        edgecolor='black')
    
    axes[1].set_xlabel('Collision Count', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Collision Distribution by Phase (MARL-LNS)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig5_convergence.png")


def fig6_collision_heatmap(results, save_dir='report/images'):
    """Figure 6: Collision count heatmap across datasets and agent counts."""
    datasets = sorted(set(r['dataset'] for r in results))
    agent_counts = sorted(set(r['n_agents'] for r in results))
    solvers = ['PP', 'LNS', 'MARL-LNS']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for si, solver in enumerate(solvers):
        matrix = np.zeros((len(datasets), len(agent_counts)))
        
        for di, d in enumerate(datasets):
            for ai, na in enumerate(agent_counts):
                sr = [r for r in results 
                      if r['dataset'] == d and r['n_agents'] == na and r['solver'] == solver]
                if sr:
                    matrix[di, ai] = np.mean([r['collisions'] for r in sr])
                else:
                    matrix[di, ai] = np.nan
        
        im = axes[si].imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        axes[si].set_xticks(range(len(agent_counts)))
        axes[si].set_xticklabels(agent_counts, fontsize=9)
        axes[si].set_yticks(range(len(datasets)))
        axes[si].set_yticklabels(datasets, fontsize=9)
        axes[si].set_xlabel('Number of Agents', fontsize=10)
        axes[si].set_title(f'{solver}: Avg Collisions', fontsize=11)
        
        # Add text annotations
        for di in range(len(datasets)):
            for ai in range(len(agent_counts)):
                val = matrix[di, ai]
                if not np.isnan(val):
                    text_color = 'white' if val > 5 else 'black'
                    axes[si].text(ai, di, f'{val:.1f}', ha='center', va='center', 
                                 fontsize=8, color=text_color, fontweight='bold')
        
        plt.colorbar(im, ax=axes[si], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_collision_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig6_collision_heatmap.png")


def main():
    os.makedirs('report/images', exist_ok=True)
    
    results = load_results()
    histories = load_convergence()
    
    print("Generating figures...")
    fig1_data_overview(results)
    fig2_success_rates(results)
    fig3_runtime_comparison(results)
    fig4_sum_of_costs(results)
    fig5_convergence(histories)
    fig6_collision_heatmap(results)
    
    print("All figures generated.")


if __name__ == '__main__':
    main()
