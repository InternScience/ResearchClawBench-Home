"""
Deep Mechanistic Network (DMN) Analysis for Drosophila Motion Pathway

This script analyzes 50 pre-trained DMN models constrained by the fly connectome
and optimized for optic flow estimation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import yaml

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/data/flow/0000'
OUTPUT_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/outputs'
IMAGE_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260415_170436/report/images'

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)


def load_all_models():
    """Load all 50 DMN models and their parameters."""
    model_dirs = sorted([d for d in os.listdir(DATA_PATH) 
                         if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    all_data = {
        'nodes_bias': [],
        'nodes_time_const': [],
        'edges_sign': [],
        'edges_syn_count': [],
        'edges_syn_strength': [],
        'validation_loss': []
    }
    
    for model_dir in model_dirs:
        # Load checkpoint
        chkpt_path = os.path.join(DATA_PATH, model_dir, 'chkpts', 'chkpt_00000')
        if os.path.exists(chkpt_path):
            chkpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)
            all_data['nodes_bias'].append(chkpt['network']['nodes_bias'].numpy())
            all_data['nodes_time_const'].append(chkpt['network']['nodes_time_const'].numpy())
            all_data['edges_sign'].append(chkpt['network']['edges_sign'].numpy())
            all_data['edges_syn_strength'].append(chkpt['network']['edges_syn_strength'].numpy())
            
            # Get synapse count if available
            if 'edges_syn_count' in chkpt['network']:
                all_data['edges_syn_count'].append(chkpt['network']['edges_syn_count'].numpy())
        
        # Load validation loss
        loss_path = os.path.join(DATA_PATH, model_dir, 'validation_loss.h5')
        if os.path.exists(loss_path):
            with h5py.File(loss_path, 'r') as f:
                all_data['validation_loss'].append(f['data'][()])
    
    # Stack arrays
    for key in all_data:
        if len(all_data[key]) > 0:
            all_data[key] = np.stack(all_data[key])
    
    return all_data


def plot_validation_loss_distribution(losses, save_path):
    """Plot distribution of validation losses across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(losses, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.4f}')
    axes[0].axvline(np.median(losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.4f}')
    axes[0].set_xlabel('Validation Loss')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Validation Losses Across 50 DMN Models')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(losses, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Statistics')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved validation loss plot to {save_path}")


def plot_node_parameters(nodes_bias, nodes_time_const, save_path):
    """Plot distribution of node parameters (bias and time constants)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bias distribution across models
    axes[0, 0].boxplot(nodes_bias.T, vert=True)
    axes[0, 0].set_xlabel('Cell Type Index')
    axes[0, 0].set_ylabel('Resting Potential (Bias)')
    axes[0, 0].set_title('Distribution of Resting Potentials Across Cell Types')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time constant distribution across models
    axes[0, 1].boxplot(nodes_time_const.T, vert=True)
    axes[0, 1].set_xlabel('Cell Type Index')
    axes[0, 1].set_ylabel('Time Constant')
    axes[0, 1].set_title('Distribution of Time Constants Across Cell Types')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean and std for bias
    mean_bias = np.mean(nodes_bias, axis=0)
    std_bias = np.std(nodes_bias, axis=0)
    x = np.arange(len(mean_bias))
    axes[1, 0].bar(x, mean_bias, yerr=std_bias, capsize=3, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Cell Type Index')
    axes[1, 0].set_ylabel('Resting Potential (Mean ± Std)')
    axes[1, 0].set_title('Mean Resting Potential by Cell Type')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean and std for time constant
    mean_tc = np.mean(nodes_time_const, axis=0)
    std_tc = np.std(nodes_time_const, axis=0)
    axes[1, 1].bar(x, mean_tc, yerr=std_tc, capsize=3, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Cell Type Index')
    axes[1, 1].set_ylabel('Time Constant (Mean ± Std)')
    axes[1, 1].set_title('Mean Time Constant by Cell Type')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved node parameters plot to {save_path}")


def plot_synapse_parameters(edges_sign, edges_syn_strength, save_path):
    """Plot distribution of synaptic parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Synapse sign distribution
    unique_signs = np.unique(edges_sign[0])
    sign_counts = []
    for sign in unique_signs:
        count = np.sum(edges_sign[0] == sign)
        sign_counts.append(count)
    
    colors = ['red' if s < 0 else 'blue' if s > 0 else 'gray' for s in unique_signs]
    axes[0, 0].bar(range(len(unique_signs)), sign_counts, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xticks(range(len(unique_signs)))
    axes[0, 0].set_xticklabels([f'{s:.0f}' for s in unique_signs])
    axes[0, 0].set_xlabel('Synaptic Sign')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Synaptic Signs')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Synaptic strength distribution across models
    axes[0, 1].boxplot(edges_syn_strength.T, vert=True)
    axes[0, 1].set_xlabel('Synapse Index')
    axes[0, 1].set_ylabel('Synaptic Strength')
    axes[0, 1].set_title('Distribution of Synaptic Strengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean synaptic strength with variability
    mean_strength = np.mean(edges_syn_strength, axis=0)
    std_strength = np.std(edges_syn_strength, axis=0)
    sorted_idx = np.argsort(mean_strength)[::-1]
    
    axes[1, 0].plot(range(len(mean_strength)), mean_strength[sorted_idx], 'b-', linewidth=1, alpha=0.7)
    axes[1, 0].fill_between(range(len(mean_strength)), 
                             mean_strength[sorted_idx] - std_strength[sorted_idx],
                             mean_strength[sorted_idx] + std_strength[sorted_idx],
                             alpha=0.3, color='blue')
    axes[1, 0].set_xlabel('Synapse Rank (by strength)')
    axes[1, 0].set_ylabel('Synaptic Strength (Mean ± Std)')
    axes[1, 0].set_title('Sorted Synaptic Strengths Across Models')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation between sign and strength
    # Get absolute values for inhibitory vs excitatory
    exc_mask = edges_sign[0] > 0
    inh_mask = edges_sign[0] < 0
    
    exc_strengths = edges_syn_strength[:, exc_mask].flatten()
    inh_strengths = edges_syn_strength[:, inh_mask].flatten()
    
    axes[1, 1].hist([exc_strengths, inh_strengths], bins=30, label=['Excitatory', 'Inhibitory'], 
                     color=['blue', 'red'], alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('Synaptic Strength')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Synaptic Strength by Polarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved synapse parameters plot to {save_path}")


def plot_parameter_correlation(nodes_bias, nodes_time_const, edges_syn_strength, losses, save_path):
    """Plot correlations between parameters and performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean bias vs validation loss
    mean_bias = np.mean(nodes_bias, axis=1)
    axes[0, 0].scatter(mean_bias, losses, c='steelblue', alpha=0.6, edgecolors='black')
    z = np.polyfit(mean_bias, losses, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(mean_bias, p(mean_bias), "r--", linewidth=2, label=f'Trend')
    r, pval = stats.pearsonr(mean_bias, losses)
    axes[0, 0].set_xlabel('Mean Resting Potential')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title(f'Mean Bias vs Performance (r={r:.3f}, p={pval:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean time constant vs validation loss
    mean_tc = np.mean(nodes_time_const, axis=1)
    axes[0, 1].scatter(mean_tc, losses, c='coral', alpha=0.6, edgecolors='black')
    z = np.polyfit(mean_tc, losses, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(mean_tc, p(mean_tc), "r--", linewidth=2, label=f'Trend')
    r, pval = stats.pearsonr(mean_tc, losses)
    axes[0, 1].set_xlabel('Mean Time Constant')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title(f'Mean Time Constant vs Performance (r={r:.3f}, p={pval:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean synaptic strength vs validation loss
    mean_syn = np.mean(edges_syn_strength, axis=1)
    axes[1, 0].scatter(mean_syn, losses, c='green', alpha=0.6, edgecolors='black')
    z = np.polyfit(mean_syn, losses, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(mean_syn, p(mean_syn), "r--", linewidth=2, label=f'Trend')
    r, pval = stats.pearsonr(mean_syn, losses)
    axes[1, 0].set_xlabel('Mean Synaptic Strength')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title(f'Mean Synaptic Strength vs Performance (r={r:.3f}, p={pval:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total synaptic strength vs validation loss
    total_syn = np.sum(edges_syn_strength, axis=1)
    axes[1, 1].scatter(total_syn, losses, c='purple', alpha=0.6, edgecolors='black')
    z = np.polyfit(total_syn, losses, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(total_syn, p(total_syn), "r--", linewidth=2, label=f'Trend')
    r, pval = stats.pearsonr(total_syn, losses)
    axes[1, 1].set_xlabel('Total Synaptic Strength')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title(f'Total Synaptic Strength vs Performance (r={r:.3f}, p={pval:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter correlation plot to {save_path}")


def plot_connectivity_matrix(edges_syn_strength, edges_sign, save_path):
    """Visualize the effective connectivity matrix."""
    # Compute effective connectivity (strength * sign)
    effective_weights = edges_syn_strength * np.abs(edges_sign)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean connectivity pattern
    mean_weights = np.mean(effective_weights, axis=0)
    
    # Create a connectivity visualization (edge-based)
    im1 = axes[0].imshow(mean_weights.reshape(1, -1), aspect='auto', cmap='RdBu_r', 
                          vmin=-np.max(np.abs(mean_weights)), vmax=np.max(np.abs(mean_weights)))
    axes[0].set_xlabel('Synapse Index')
    axes[0].set_title('Mean Effective Synaptic Weights')
    plt.colorbar(im1, ax=axes[0], label='Weight')
    
    # Variance across models
    var_weights = np.var(effective_weights, axis=0)
    im2 = axes[1].imshow(var_weights.reshape(1, -1), aspect='auto', cmap='hot')
    axes[1].set_xlabel('Synapse Index')
    axes[1].set_title('Variance of Synaptic Weights Across Models')
    plt.colorbar(im2, ax=axes[1], label='Variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved connectivity matrix plot to {save_path}")


def plot_model_consistency(nodes_bias, nodes_time_const, edges_syn_strength, save_path):
    """Plot consistency of parameters across models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coefficient of variation for bias
    cv_bias = np.std(nodes_bias, axis=0) / (np.mean(nodes_bias, axis=0) + 1e-8)
    axes[0, 0].bar(range(len(cv_bias)), cv_bias, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cell Type Index')
    axes[0, 0].set_ylabel('Coefficient of Variation')
    axes[0, 0].set_title('Consistency of Resting Potentials Across Models')
    axes[0, 0].axhline(np.mean(cv_bias), color='red', linestyle='--', label=f'Mean CV: {np.mean(cv_bias):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coefficient of variation for time constants
    cv_tc = np.std(nodes_time_const, axis=0) / (np.mean(nodes_time_const, axis=0) + 1e-8)
    axes[0, 1].bar(range(len(cv_tc)), cv_tc, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Cell Type Index')
    axes[0, 1].set_ylabel('Coefficient of Variation')
    axes[0, 1].set_title('Consistency of Time Constants Across Models')
    axes[0, 1].axhline(np.mean(cv_tc), color='red', linestyle='--', label=f'Mean CV: {np.mean(cv_tc):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameter stability heatmap for top cell types
    # Select cell types with highest variance
    bias_var = np.var(nodes_bias, axis=0)
    top_var_idx = np.argsort(bias_var)[-20:]  # Top 20 most variable
    
    im = axes[1, 0].imshow(nodes_bias[:, top_var_idx].T, aspect='auto', cmap='viridis')
    axes[1, 0].set_xlabel('Model Index')
    axes[1, 0].set_ylabel('Cell Type Index (Top Variable)')
    axes[1, 0].set_title('Resting Potentials Across Models (Most Variable)')
    plt.colorbar(im, ax=axes[1, 0], label='Bias')
    
    # Synaptic strength stability
    syn_var = np.var(edges_syn_strength, axis=0)
    top_syn_idx = np.argsort(syn_var)[-50:]  # Top 50 most variable
    
    im2 = axes[1, 1].imshow(edges_syn_strength[:, top_syn_idx].T, aspect='auto', cmap='plasma')
    axes[1, 1].set_xlabel('Model Index')
    axes[1, 1].set_ylabel('Synapse Index (Top Variable)')
    axes[1, 1].set_title('Synaptic Strengths Across Models (Most Variable)')
    plt.colorbar(im2, ax=axes[1, 1], label='Strength')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model consistency plot to {save_path}")


def compute_summary_statistics(data):
    """Compute summary statistics for all parameters."""
    stats_dict = {}
    
    # Validation loss stats
    losses = data['validation_loss']
    stats_dict['validation_loss'] = {
        'mean': float(np.mean(losses)),
        'std': float(np.std(losses)),
        'min': float(np.min(losses)),
        'max': float(np.max(losses)),
        'median': float(np.median(losses))
    }
    
    # Node bias stats
    bias = data['nodes_bias']
    stats_dict['nodes_bias'] = {
        'mean': float(np.mean(bias)),
        'std': float(np.std(bias)),
        'mean_per_cell': np.mean(bias, axis=0).tolist(),
        'std_per_cell': np.std(bias, axis=0).tolist()
    }
    
    # Time constant stats
    tc = data['nodes_time_const']
    stats_dict['nodes_time_const'] = {
        'mean': float(np.mean(tc)),
        'std': float(np.std(tc)),
        'mean_per_cell': np.mean(tc, axis=0).tolist(),
        'std_per_cell': np.std(tc, axis=0).tolist()
    }
    
    # Synaptic strength stats
    syn = data['edges_syn_strength']
    stats_dict['edges_syn_strength'] = {
        'mean': float(np.mean(syn)),
        'std': float(np.std(syn)),
        'mean_per_edge': np.mean(syn, axis=0).tolist(),
        'std_per_edge': np.std(syn, axis=0).tolist()
    }
    
    # Synaptic sign distribution
    sign = data['edges_sign']
    unique, counts = np.unique(sign[0], return_counts=True)
    stats_dict['synaptic_signs'] = {
        'values': unique.tolist(),
        'counts': counts.tolist()
    }
    
    return stats_dict


def main():
    print("=" * 60)
    print("Deep Mechanistic Network (DMN) Analysis")
    print("Drosophila Motion Pathway")
    print("=" * 60)
    
    # Load data
    print("\nLoading 50 DMN models...")
    data = load_all_models()
    print(f"Loaded {len(data['validation_loss'])} models")
    print(f"Number of cell types: {data['nodes_bias'].shape[1]}")
    print(f"Number of synaptic connections: {data['edges_syn_strength'].shape[1]}")
    
    # Generate all plots
    print("\nGenerating figures...")
    
    plot_validation_loss_distribution(
        data['validation_loss'], 
        os.path.join(IMAGE_PATH, 'fig1_validation_loss.png')
    )
    
    plot_node_parameters(
        data['nodes_bias'], 
        data['nodes_time_const'],
        os.path.join(IMAGE_PATH, 'fig2_node_parameters.png')
    )
    
    plot_synapse_parameters(
        data['edges_sign'],
        data['edges_syn_strength'],
        os.path.join(IMAGE_PATH, 'fig3_synapse_parameters.png')
    )
    
    plot_parameter_correlation(
        data['nodes_bias'],
        data['nodes_time_const'],
        data['edges_syn_strength'],
        data['validation_loss'],
        os.path.join(IMAGE_PATH, 'fig4_parameter_correlations.png')
    )
    
    plot_connectivity_matrix(
        data['edges_syn_strength'],
        data['edges_sign'],
        os.path.join(IMAGE_PATH, 'fig5_connectivity_matrix.png')
    )
    
    plot_model_consistency(
        data['nodes_bias'],
        data['nodes_time_const'],
        data['edges_syn_strength'],
        os.path.join(IMAGE_PATH, 'fig6_model_consistency.png')
    )
    
    # Compute and save statistics
    print("\nComputing summary statistics...")
    stats_dict = compute_summary_statistics(data)
    
    # Save statistics as JSON
    import json
    stats_path = os.path.join(OUTPUT_PATH, 'summary_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    # Save aggregated data
    np.savez(os.path.join(OUTPUT_PATH, 'aggregated_model_data.npz'), **data)
    print(f"Saved aggregated data to {OUTPUT_PATH}/aggregated_model_data.npz")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Validation Loss: {stats_dict['validation_loss']['mean']:.4f} ± {stats_dict['validation_loss']['std']:.4f}")
    print(f"  - Resting Potential: {stats_dict['nodes_bias']['mean']:.4f} ± {stats_dict['nodes_bias']['std']:.4f}")
    print(f"  - Time Constant: {stats_dict['nodes_time_const']['mean']:.4f} ± {stats_dict['nodes_time_const']['std']:.4f}")
    print(f"  - Synaptic Strength: {stats_dict['edges_syn_strength']['mean']:.4f} ± {stats_dict['edges_syn_strength']['std']:.4f}")
    
    exc_count = stats_dict['synaptic_signs']['counts'][stats_dict['synaptic_signs']['values'].index(1.0)]
    inh_count = stats_dict['synaptic_signs']['counts'][stats_dict['synaptic_signs']['values'].index(-1.0)]
    print(f"  - Excitatory synapses: {exc_count}")
    print(f"  - Inhibitory synapses: {inh_count}")


if __name__ == '__main__':
    main()
