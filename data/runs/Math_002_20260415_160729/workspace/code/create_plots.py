"""
Create plots for the MAPF report
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def create_all_plots(results, output_dir):
    """Create all plots for the report."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = list(results.keys())
    
    # 1. Success Rate Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        data = results[dataset]
        
        agent_counts = sorted([int(k) for k in data.keys()])
        lns_rates = [data[str(ac)]['lns_marl']['success_rate'] * 100 for ac in agent_counts]
        pp_rates = [data[str(ac)]['baseline_pp']['success_rate'] * 100 for ac in agent_counts]
        
        x = np.arange(len(agent_counts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lns_rates, width, label='LNS-MARL (Ours)', color='#2E86AB')
        bars2 = ax.bar(x + width/2, pp_rates, width, label='Prioritized Planning', color='#F24236')
        
        ax.set_xlabel('Number of Agents', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title(f'{dataset.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_counts)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplot
    if len(datasets) < len(axes):
        for idx in range(len(datasets), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: success_rate_comparison.png")
    
    # 2. Runtime Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        data = results[dataset]
        
        agent_counts = sorted([int(k) for k in data.keys()])
        lns_times = [data[str(ac)]['lns_marl']['avg_time'] for ac in agent_counts]
        pp_times = [data[str(ac)]['baseline_pp']['avg_time'] for ac in agent_counts]
        
        ax.plot(agent_counts, lns_times, 'o-', label='LNS-MARL (Ours)', 
                color='#2E86AB', linewidth=2, markersize=8)
        ax.plot(agent_counts, pp_times, 's-', label='Prioritized Planning', 
                color='#F24236', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Agents', fontweight='bold')
        ax.set_ylabel('Average Runtime (s)', fontweight='bold')
        ax.set_title(f'{dataset.replace("_", " ").title()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if len(datasets) < len(axes):
        for idx in range(len(datasets), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: runtime_comparison.png")
    
    # 3. Collision Comparison (LNS before/after)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        data = results[dataset]
        
        agent_counts = sorted([int(k) for k in data.keys()])
        initial_coll = [data[str(ac)]['lns_marl']['avg_collisions_initial'] for ac in agent_counts]
        final_coll = [data[str(ac)]['lns_marl']['avg_collisions_final'] for ac in agent_counts]
        
        x = np.arange(len(agent_counts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, initial_coll, width, label='Initial (PP)', color='#F6AE2D')
        bars2 = ax.bar(x + width/2, final_coll, width, label='After LNS', color='#86CD82')
        
        ax.set_xlabel('Number of Agents', fontweight='bold')
        ax.set_ylabel('Average Collisions', fontweight='bold')
        ax.set_title(f'{dataset.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_counts)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    if len(datasets) < len(axes):
        for idx in range(len(datasets), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_reduction.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: collision_reduction.png")
    
    # 4. Overall Performance Summary
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_lns = []
    all_pp = []
    labels = []
    
    for dataset in datasets:
        for agent_str in results[dataset]:
            all_lns.append(results[dataset][agent_str]['lns_marl']['success_rate'] * 100)
            all_pp.append(results[dataset][agent_str]['baseline_pp']['success_rate'] * 100)
            labels.append(f"{dataset[:8]}\n{agent_str} agt")
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, all_lns, width, label='LNS-MARL', color='#2E86AB')
    bars2 = ax.barh(x + width/2, all_pp, width, label='Prioritized Planning', color='#F24236')
    
    ax.set_xlabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Overall Performance Comparison Across All Configurations', fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.legend()
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: overall_summary.png")


def create_map_overview(data_dir, output_dir):
    """Create overview of map types."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['empty', 'maze', 'random_small', 'random_medium', 'warehouse']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
                if file.endswith('.npy') and 'eval_map' in file:
                    map_path = os.path.join(root, file)
                    grid = np.load(map_path, allow_pickle=True)
                    
                    # Plot
                    obstacle_mask = (grid == -1)
                    ax.imshow(obstacle_mask, cmap='binary', interpolation='nearest')
                    ax.set_title(f'{dataset.replace("_", " ").title()}\n({grid.shape[0]}x{grid.shape[1]})', 
                                fontweight='bold')
                    ax.axis('off')
                    break
            break
    
    # Hide extra subplot
    if len(datasets) < len(axes):
        for idx in range(len(datasets), len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_types_overview.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: map_types_overview.png")


if __name__ == '__main__':
    # Load results
    with open('outputs/results.json', 'r') as f:
        results = json.load(f)
    
    # Create plots
    create_all_plots(results, 'report/images')
    create_map_overview('data', 'report/images')
