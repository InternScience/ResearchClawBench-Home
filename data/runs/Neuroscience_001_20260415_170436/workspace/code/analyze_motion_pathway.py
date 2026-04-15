"""
Motion Pathway Analysis - Analyzing the Drosophila motion detection circuit
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml

sns.set_style("whitegrid")

DATA_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/data/flow/0000'
OUTPUT_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/outputs'
IMAGE_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/report/images'


def analyze_cell_type_diversity():
    """Analyze the diversity of cell types in the network."""
    # Load metadata to understand cell type structure
    meta_path = os.path.join(DATA_PATH, '000', '_meta.yaml')
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    print("Network Configuration:")
    print(f"  Connectome file: {meta['config']['network']['connectome']['file']}")
    print(f"  Extent: {meta['config']['network']['connectome']['extent']}")
    print(f"  Neuron dynamics: {meta['config']['network']['dynamics']['type']}")
    print(f"  Activation: {meta['config']['network']['dynamics']['activation']['type']}")
    
    return meta


def plot_neural_response_simulation(save_path):
    """Simulate and visualize neural responses to visual stimuli."""
    # Load aggregated data
    data = np.load(os.path.join(OUTPUT_PATH, 'aggregated_model_data.npz'))
    nodes_bias = data['nodes_bias']
    nodes_time_const = data['nodes_time_const']
    
    # Simulate responses to moving edge stimulus
    dt = 0.02  # 20ms time step from config
    t = np.arange(0, 2, dt)  # 2 seconds
    
    # Create a moving edge stimulus (simplified)
    stimulus_velocity = 30  # degrees/second
    edge_position = stimulus_velocity * t
    
    # Simulate responses for a subset of cell types
    n_cell_types = 20
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i in range(n_cell_types):
        # Use mean parameters across models
        bias = np.mean(nodes_bias[:, i])
        tau = np.mean(nodes_time_const[:, i])
        
        # Simulate simple leaky integrate-and-fire-like response
        response = np.zeros_like(t)
        for j in range(1, len(t)):
            # Simple dynamics: tau * dv/dt = -v + bias + input
            input_current = np.sin(2 * np.pi * edge_position[j] / 100)  # Sinusoidal input
            response[j] = response[j-1] + dt * (-response[j-1] + bias + input_current) / tau
        
        axes[i].plot(t, response, linewidth=1.5)
        axes[i].fill_between(t, 0, response, alpha=0.3)
        axes[i].set_title(f'Cell Type {i}\nτ={tau:.3f}, bias={bias:.3f}', fontsize=8)
        axes[i].set_xlabel('Time (s)', fontsize=7)
        axes[i].set_ylabel('Response', fontsize=7)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Simulated Neural Responses to Moving Edge Stimulus', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved neural response simulation to {save_path}")


def plot_motion_detection_analysis(save_path):
    """Analyze motion detection mechanisms in the network."""
    data = np.load(os.path.join(OUTPUT_PATH, 'aggregated_model_data.npz'))
    edges_syn_strength = data['edges_syn_strength']
    edges_sign = data['edges_sign']
    
    # Calculate connectivity statistics
    mean_strength = np.mean(edges_syn_strength, axis=0)
    std_strength = np.std(edges_syn_strength, axis=0)
    
    # Analyze strength distribution
    strong_synapses = np.sum(mean_strength > np.percentile(mean_strength, 90))
    weak_synapses = np.sum(mean_strength < np.percentile(mean_strength, 10))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Synaptic strength distribution
    axes[0, 0].hist(mean_strength, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(mean_strength), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mean_strength):.4f}')
    axes[0, 0].set_xlabel('Mean Synaptic Strength')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Mean Synaptic Strengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Excitatory vs Inhibitory balance
    exc_mask = edges_sign[0] == 1
    inh_mask = edges_sign[0] == -1
    
    exc_total = np.sum(mean_strength[exc_mask])
    inh_total = np.sum(mean_strength[inh_mask])
    
    colors = ['blue', 'red']
    axes[0, 1].bar(['Excitatory', 'Inhibitory'], [exc_total, inh_total], 
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Total Synaptic Strength')
    axes[0, 1].set_title('Excitatory vs Inhibitory Balance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add text annotations
    ei_ratio = exc_total / (inh_total + 1e-8)
    axes[0, 1].text(0.5, max(exc_total, inh_total) * 0.8, 
                    f'E/I Ratio: {ei_ratio:.2f}', 
                    ha='center', fontsize=12, fontweight='bold')
    
    # 3. Synaptic strength vs variability
    axes[1, 0].scatter(mean_strength, std_strength, alpha=0.5, c='purple', edgecolors='black')
    axes[1, 0].set_xlabel('Mean Synaptic Strength')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_title('Strength vs Variability Across Models')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add correlation
    r, pval = stats.pearsonr(mean_strength, std_strength)
    axes[1, 0].text(0.05, 0.95, f'r={r:.3f}, p={pval:.2e}', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Cumulative distribution of synaptic strengths
    sorted_strengths = np.sort(mean_strength)[::-1]
    cumsum = np.cumsum(sorted_strengths)
    cumsum_norm = cumsum / cumsum[-1]
    
    axes[1, 1].plot(range(len(sorted_strengths)), cumsum_norm, linewidth=2, color='green')
    axes[1, 1].axhline(0.5, color='red', linestyle='--', label='50% of total strength')
    axes[1, 1].axhline(0.8, color='orange', linestyle='--', label='80% of total strength')
    axes[1, 1].set_xlabel('Synapse Rank')
    axes[1, 1].set_ylabel('Cumulative Fraction of Total Strength')
    axes[1, 1].set_title('Cumulative Distribution of Synaptic Strength')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Find synapses contributing to 50% and 80%
    n_50 = np.argmax(cumsum_norm >= 0.5)
    n_80 = np.argmax(cumsum_norm >= 0.8)
    axes[1, 1].text(0.5, 0.3, f'{n_50} synapses → 50% strength\n{n_80} synapses → 80% strength',
                    transform=axes[1, 1].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved motion detection analysis to {save_path}")


def plot_connectome_structure(save_path):
    """Visualize the structure of the connectome-constrained network."""
    data = np.load(os.path.join(OUTPUT_PATH, 'aggregated_model_data.npz'))
    edges_syn_strength = data['edges_syn_strength']
    edges_sign = data['edges_sign']
    
    n_models, n_edges = edges_syn_strength.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Network sparsity visualization
    # Create adjacency matrix representation (edge-based visualization)
    mean_strength = np.mean(edges_syn_strength, axis=0)
    sign = edges_sign[0]
    
    # Plot synaptic weights as a 1D heatmap (since we don't have exact connectivity matrix)
    ax1 = axes[0, 0]
    n_show = min(200, n_edges)  # Show first 200 synapses
    
    # Sort by strength for visualization
    sorted_idx = np.argsort(mean_strength)[::-1][:n_show]
    strengths_to_plot = mean_strength[sorted_idx]
    signs_to_plot = sign[sorted_idx]
    
    colors = ['red' if s < 0 else 'blue' for s in signs_to_plot]
    ax1.barh(range(n_show), strengths_to_plot, color=colors, alpha=0.7)
    ax1.set_xlabel('Synaptic Strength')
    ax1.set_ylabel('Synapse Rank')
    ax1.set_title('Top 200 Strongest Synaptic Connections')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # 2. Synaptic weight distribution by sign
    exc_strengths = mean_strength[sign == 1]
    inh_strengths = mean_strength[sign == -1]
    
    axes[0, 1].hist([exc_strengths, inh_strengths], bins=30, 
                    label=[f'Excitatory (n={len(exc_strengths)})', 
                           f'Inhibitory (n={len(inh_strengths)})'],
                    color=['blue', 'red'], alpha=0.6, edgecolor='black')
    axes[0, 1].set_xlabel('Synaptic Strength')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Synaptic Strength Distribution by Polarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model variability heatmap
    variability = np.std(edges_syn_strength, axis=0)
    ax3 = axes[1, 0]
    im = ax3.imshow(variability.reshape(1, -1), aspect='auto', cmap='hot', vmin=0)
    ax3.set_xlabel('Synapse Index')
    ax3.set_title('Synaptic Weight Variability Across Models')
    plt.colorbar(im, ax=ax3, label='Standard Deviation')
    
    # 4. Network statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Network Statistics Summary
    ==========================
    
    Total Models: {n_models}
    Cell Types: 65
    Synaptic Connections: {n_edges}
    
    Synaptic Weights:
      Mean: {np.mean(mean_strength):.4f}
      Std: {np.std(mean_strength):.4f}
      Min: {np.min(mean_strength):.4f}
      Max: {np.max(mean_strength):.4f}
      Median: {np.median(mean_strength):.4f}
    
    Synapse Counts:
      Excitatory: {np.sum(sign == 1)}
      Inhibitory: {np.sum(sign == -1)}
      E/I Ratio: {np.sum(sign == 1) / np.sum(sign == -1):.2f}
    
    Strongest Synapse: {np.max(mean_strength):.4f}
    Weakest Synapse: {np.min(mean_strength):.4f}
    
    Top 10% synapses carry {np.sum(sorted(mean_strength)[-n_edges//10:]) / np.sum(mean_strength) * 100:.1f}% of total weight
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved connectome structure plot to {save_path}")


def generate_comparative_analysis(save_path):
    """Generate comparison between best and worst performing models."""
    data = np.load(os.path.join(OUTPUT_PATH, 'aggregated_model_data.npz'))
    losses = data['validation_loss']
    nodes_bias = data['nodes_bias']
    nodes_time_const = data['nodes_time_const']
    edges_syn_strength = data['edges_syn_strength']
    
    # Find best and worst models
    best_idx = np.argmin(losses)
    worst_idx = np.argmax(losses)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Resting potential comparison
    x = np.arange(nodes_bias.shape[1])
    axes[0, 0].plot(x, nodes_bias[best_idx], 'b-', linewidth=1.5, label=f'Best Model (loss={losses[best_idx]:.4f})')
    axes[0, 0].plot(x, nodes_bias[worst_idx], 'r-', linewidth=1.5, label=f'Worst Model (loss={losses[worst_idx]:.4f})')
    axes[0, 0].fill_between(x, nodes_bias[best_idx], nodes_bias[worst_idx], alpha=0.2, color='gray')
    axes[0, 0].set_xlabel('Cell Type Index')
    axes[0, 0].set_ylabel('Resting Potential')
    axes[0, 0].set_title('Resting Potentials: Best vs Worst Model')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time constant comparison
    axes[0, 1].plot(x, nodes_time_const[best_idx], 'b-', linewidth=1.5, label=f'Best Model')
    axes[0, 1].plot(x, nodes_time_const[worst_idx], 'r-', linewidth=1.5, label=f'Worst Model')
    axes[0, 1].fill_between(x, nodes_time_const[best_idx], nodes_time_const[worst_idx], alpha=0.2, color='gray')
    axes[0, 1].set_xlabel('Cell Type Index')
    axes[0, 1].set_ylabel('Time Constant')
    axes[0, 1].set_title('Time Constants: Best vs Worst Model')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Synaptic strength comparison
    y = np.arange(edges_syn_strength.shape[1])
    axes[1, 0].scatter(y, edges_syn_strength[best_idx], c='blue', alpha=0.5, s=10, label='Best Model')
    axes[1, 0].scatter(y, edges_syn_strength[worst_idx], c='red', alpha=0.5, s=10, label='Worst Model')
    axes[1, 0].set_xlabel('Synapse Index')
    axes[1, 0].set_ylabel('Synaptic Strength')
    axes[1, 0].set_title('Synaptic Strengths: Best vs Worst Model')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Parameter differences
    bias_diff = np.abs(nodes_bias[best_idx] - nodes_bias[worst_idx])
    tc_diff = np.abs(nodes_time_const[best_idx] - nodes_time_const[worst_idx])
    syn_diff = np.abs(edges_syn_strength[best_idx] - edges_syn_strength[worst_idx])
    
    axes[1, 1].hist([bias_diff, tc_diff], bins=20, label=['Resting Potential Diff', 'Time Constant Diff'],
                    alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Difference')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Parameter Differences Between Best/Worst Models')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative analysis to {save_path}")


def main():
    print("=" * 60)
    print("Motion Pathway Analysis")
    print("=" * 60)
    
    # Analyze cell type diversity
    print("\nAnalyzing network configuration...")
    meta = analyze_cell_type_diversity()
    
    # Generate motion pathway figures
    print("\nGenerating motion pathway figures...")
    
    plot_neural_response_simulation(
        os.path.join(IMAGE_PATH, 'fig7_neural_responses.png')
    )
    
    plot_motion_detection_analysis(
        os.path.join(IMAGE_PATH, 'fig8_motion_detection.png')
    )
    
    plot_connectome_structure(
        os.path.join(IMAGE_PATH, 'fig9_connectome_structure.png')
    )
    
    generate_comparative_analysis(
        os.path.join(IMAGE_PATH, 'fig10_model_comparison.png')
    )
    
    print("\n" + "=" * 60)
    print("Motion pathway analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
