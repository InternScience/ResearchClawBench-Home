#!/usr/bin/env python3
"""
Analysis of Deep Mechanistic Network (DMN) models for Drosophila optic lobe motion pathway.
Extracts network parameters, analyzes connectome structure, and generates visualizations.
"""

import os
import json
import pickle
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path('data/flow/0000')
OUTPUT_DIR = Path('outputs')
REPORT_IMG_DIR = Path('report/images')
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Load all 50 model checkpoints and extract parameters
# ============================================================================
print("=" * 60)
print("Loading 50 DMN model checkpoints...")
print("=" * 60)

model_ids = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit() and len(d) == 3])
print(f"Found {len(model_ids)} model directories")

all_params = {}
for mid in model_ids:
    ckpt_path = DATA_DIR / mid / 'best_chkpt'
    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        all_params[mid] = data
        # Also load validation loss
        val_loss_path = DATA_DIR / mid / 'validation_loss.h5'
        if val_loss_path.exists():
            with h5py.File(val_loss_path, 'r') as f:
                all_params[mid]['val_loss'] = float(f['data'][()])

print(f"Loaded {len(all_params)} models")

# ============================================================================
# 2. Extract and analyze network structure
# ============================================================================
print("\n" + "=" * 60)
print("Analyzing network structure...")
print("=" * 60)

# Get reference model parameters
ref_model = all_params[model_ids[0]]
net_params = ref_model['network']

print(f"Network parameter keys: {list(net_params.keys())}")
print(f"  nodes_bias: {net_params['nodes_bias'].shape}")
print(f"  nodes_time_const: {net_params['nodes_time_const'].shape}")
print(f"  edges_sign: {net_params['edges_sign'].shape}")
print(f"  edges_syn_count: {net_params['edges_syn_count'].shape}")
print(f"  edges_syn_strength: {net_params['edges_syn_strength'].shape}")

n_nodes = net_params['nodes_bias'].shape[0]
n_edge_types = net_params['edges_sign'].shape[0]
n_synapse_groups = net_params['edges_syn_count'].shape[0]

print(f"\nNetwork summary:")
print(f"  Number of neuron types (nodes): {n_nodes}")
print(f"  Number of edge types: {n_edge_types}")
print(f"  Number of synapse count groups: {n_synapse_groups}")

# ============================================================================
# 3. Load cell type annotations from meta.yaml
# ============================================================================
print("\n" + "=" * 60)
print("Loading cell type annotations...")
print("=" * 60)

import yaml
meta_path = DATA_DIR / '000' / '_meta.yaml'
with open(meta_path, 'r') as f:
    meta = yaml.safe_load(f)

print(f"Config type: {meta['config']['type']}")
print(f"Connectome type: {meta['config']['network']['connectome']['type']}")
print(f"Dynamics type: {meta['config']['network']['dynamics']['type']}")
print(f"Task type: {meta['config']['task']['type']}")
print(f"Dataset: {meta['config']['task']['dataset']['type']}")
print(f"Training iterations: {meta['config']['task']['n_iters']}")

# ============================================================================
# 4. Load UMAP and clustering data
# ============================================================================
print("\n" + "=" * 60)
print("Loading UMAP and clustering data...")
print("=" * 60)

umap_dir = DATA_DIR / 'umap_and_clustering'
umap_files = sorted([f for f in os.listdir(umap_dir) if f.endswith('.pickle')])
print(f"Found {len(umap_files)} cell type UMAP files")

# Try to load with different approaches
cell_type_names = [f.replace('.pickle', '') for f in umap_files]
print(f"Cell types: {cell_type_names}")

# ============================================================================
# 5. Analyze learned parameters across all 50 models
# ============================================================================
print("\n" + "=" * 60)
print("Analyzing learned parameters across 50 models...")
print("=" * 60)

# Collect all parameters
all_biases = []
all_time_consts = []
all_syn_strengths = []
all_val_losses = []

for mid in model_ids:
    if mid in all_params:
        net = all_params[mid]['network']
        all_biases.append(net['nodes_bias'].numpy())
        all_time_consts.append(net['nodes_time_const'].numpy())
        all_syn_strengths.append(net['edges_syn_strength'].numpy())
        if 'val_loss' in all_params[mid]:
            all_val_losses.append(all_params[mid]['val_loss'])

all_biases = np.array(all_biases)  # (50, 65)
all_time_consts = np.array(all_time_consts)  # (50, 65)
all_syn_strengths = np.array(all_syn_strengths)  # (50, 604)
all_val_losses = np.array(all_val_losses)

print(f"Biases shape: {all_biases.shape}")
print(f"Time constants shape: {all_time_consts.shape}")
print(f"Synapse strengths shape: {all_syn_strengths.shape}")
print(f"Validation losses: {len(all_val_losses)} values")
print(f"  Mean val loss: {all_val_losses.mean():.4f} ± {all_val_losses.std():.4f}")
print(f"  Best val loss: {all_val_losses.min():.4f} (model {model_ids[np.argmin(all_val_losses)]})")
print(f"  Worst val loss: {all_val_losses.max():.4f}")

# ============================================================================
# 6. Analyze synapse sign (excitatory/inhibitory)
# ============================================================================
print("\n" + "=" * 60)
print("Analyzing synaptic polarity...")
print("=" * 60)

edge_signs = net_params['edges_sign'].numpy()
n_excitatory = np.sum(edge_signs > 0)
n_inhibitory = np.sum(edge_signs < 0)
n_zero = np.sum(edge_signs == 0)
print(f"Excitatory edges: {n_excitatory} ({100*n_excitatory/len(edge_signs):.1f}%)")
print(f"Inhibitory edges: {n_inhibitory} ({100*n_inhibitory/len(edge_signs):.1f}%)")
print(f"Zero edges: {n_zero} ({100*n_zero/len(edge_signs):.1f}%)")

# ============================================================================
# 7. Analyze synapse counts
# ============================================================================
print("\n" + "=" * 60)
print("Analyzing synapse count distribution...")
print("=" * 60)

syn_counts = net_params['edges_syn_count'].numpy()
print(f"Synapse count statistics:")
print(f"  Min: {syn_counts.min():.4f}")
print(f"  Max: {syn_counts.max():.4f}")
print(f"  Mean: {syn_counts.mean():.4f}")
print(f"  Median: {np.median(syn_counts):.4f}")
print(f"  Non-zero: {np.sum(syn_counts > 0)} / {len(syn_counts)}")

# ============================================================================
# 8. Generate Figures
# ============================================================================
print("\n" + "=" * 60)
print("Generating figures...")
print("=" * 60)

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# --- Figure 1: Validation loss distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(all_val_losses, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(all_val_losses.mean(), color='red', linestyle='--', label=f'Mean: {all_val_losses.mean():.3f}')
axes[0].axvline(all_val_losses.min(), color='green', linestyle='--', label=f'Best: {all_val_losses.min():.3f}')
axes[0].set_xlabel('Validation Loss')
axes[0].set_ylabel('Count')
axes[0].set_title('A. Distribution of Validation Losses Across 50 Models')
axes[0].legend()

# Sorted validation losses
sorted_idx = np.argsort(all_val_losses)
axes[1].plot(range(len(all_val_losses)), all_val_losses[sorted_idx], 'o-', color='steelblue', markersize=5)
axes[1].set_xlabel('Model Rank')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('B. Sorted Validation Losses')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig1_validation_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_validation_loss.png")

# --- Figure 2: Resting potential (bias) distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mean bias across models
mean_biases = all_biases.mean(axis=0)
std_biases = all_biases.std(axis=0)
node_idx = np.arange(n_nodes)

axes[0].bar(node_idx, mean_biases, yerr=std_biases, color='coral', alpha=0.7, capsize=2)
axes[0].set_xlabel('Node Index (Cell Type)')
axes[0].set_ylabel('Resting Potential (Bias)')
axes[0].set_title('A. Mean Resting Potential Across 50 Models')

# Distribution of biases
sampled = all_biases[:, ::5]
axes[1].violinplot([sampled[:, i] for i in range(sampled.shape[1])], positions=np.arange(sampled.shape[1]), showmeans=True, showmedians=True)
axes[1].set_xlabel('Node Index (sampled)')
axes[1].set_ylabel('Resting Potential (Bias)')
axes[1].set_title('B. Bias Distribution (Every 5th Node)')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig2_resting_potentials.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_resting_potentials.png")

# --- Figure 3: Time constants ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

mean_tc = all_time_consts.mean(axis=0)
std_tc = all_time_consts.std(axis=0)

axes[0].bar(node_idx, mean_tc, yerr=std_tc, color='mediumpurple', alpha=0.7, capsize=2)
axes[0].set_xlabel('Node Index (Cell Type)')
axes[0].set_ylabel('Time Constant (s)')
axes[0].set_title('A. Mean Time Constants Across 50 Models')

# Heatmap of time constants across models
im = axes[1].imshow(all_time_consts, aspect='auto', cmap='viridis')
axes[1].set_xlabel('Node Index')
axes[1].set_ylabel('Model Index')
axes[1].set_title('B. Time Constants Heatmap (All Models)')
plt.colorbar(im, ax=axes[1], label='Time Constant (s)')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig3_time_constants.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_time_constants.png")

# --- Figure 4: Synaptic polarity and strength ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Edge sign distribution
sign_colors = ['blue' if s < 0 else 'red' if s > 0 else 'gray' for s in edge_signs]
axes[0].bar(range(len(edge_signs)), edge_signs, color=sign_colors, alpha=0.7, width=1.0)
axes[0].set_xlabel('Edge Index')
axes[0].set_ylabel('Synapse Sign')
axes[0].set_title('A. Synaptic Polarity (Red=Exc, Blue=Inh)')

# Synapse count distribution
axes[1].hist(syn_counts[syn_counts > 0], bins=50, color='seagreen', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Synapse Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('B. Synapse Count Distribution')
axes[1].set_yscale('log')

# Synapse strength across models
mean_ss = all_syn_strengths.mean(axis=0)
std_ss = all_syn_strengths.std(axis=0)
axes[2].errorbar(range(len(mean_ss)), mean_ss, yerr=std_ss, fmt='o', markersize=2, alpha=0.5, color='darkorange')
axes[2].set_xlabel('Edge Index')
axes[2].set_ylabel('Synapse Strength')
axes[2].set_title('C. Mean Synapse Strength Across Models')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig4_synaptic_properties.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_synaptic_properties.png")

# --- Figure 5: Parameter variability across models ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Bias variability
bias_cv = all_biases.std(axis=0) / (np.abs(all_biases.mean(axis=0)) + 1e-8)
axes[0, 0].bar(node_idx, bias_cv, color='coral', alpha=0.7)
axes[0, 0].set_xlabel('Node Index')
axes[0, 0].set_ylabel('Coefficient of Variation')
axes[0, 0].set_title('A. Bias Variability (CV) Across Models')

# Time constant variability
tc_cv = all_time_consts.std(axis=0) / (np.abs(all_time_consts.mean(axis=0)) + 1e-8)
axes[0, 1].bar(node_idx, tc_cv, color='mediumpurple', alpha=0.7)
axes[0, 1].set_xlabel('Node Index')
axes[0, 1].set_ylabel('Coefficient of Variation')
axes[0, 1].set_title('B. Time Constant Variability (CV) Across Models')

# Synapse strength variability
ss_cv = all_syn_strengths.std(axis=0) / (np.abs(all_syn_strengths.mean(axis=0)) + 1e-8)
axes[1, 0].hist(ss_cv[np.isfinite(ss_cv)], bins=50, color='darkorange', edgecolor='white', alpha=0.8)
axes[1, 0].set_xlabel('Coefficient of Variation')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('C. Synapse Strength CV Distribution')

# Correlation between bias and time constant
axes[1, 1].scatter(mean_biases, mean_tc, c=node_idx, cmap='tab20', s=50, alpha=0.8)
axes[1, 1].set_xlabel('Mean Resting Potential')
axes[1, 1].set_ylabel('Mean Time Constant')
axes[1, 1].set_title('D. Resting Potential vs Time Constant')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig5_parameter_variability.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_parameter_variability.png")

# --- Figure 6: Connectome structure analysis ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Effective synaptic weights (sign * strength * count)
ref_strengths = net_params['edges_syn_strength'].numpy()
ref_counts = net_params['edges_syn_count'].numpy()
# Edge signs and strengths are per edge type (604), syn counts are per group (2355)
# Compute effective weights per edge type using mean syn count per edge
# For visualization, use sign * strength directly
effective_weights = edge_signs * ref_strengths

axes[0].hist(effective_weights, bins=50, color='teal', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_xlabel('Effective Weight')
axes[0].set_ylabel('Frequency')
axes[0].set_title('A. Effective Synaptic Weight Distribution')

# Cumulative distribution of synapse counts
sorted_counts = np.sort(ref_counts[ref_counts > 0])[::-1]
cum_frac = np.cumsum(sorted_counts) / sorted_counts.sum()
axes[1].plot(range(len(cum_frac)), cum_frac, color='darkred', linewidth=2)
axes[1].set_xlabel('Edge Rank (by synapse count)')
axes[1].set_ylabel('Cumulative Fraction')
axes[1].set_title('B. Cumulative Synapse Distribution')
axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[1].axhline(0.8, color='gray', linestyle=':', alpha=0.5)

# Node degree analysis (from edge connectivity)
# Count edges per node
node_in_degree = np.zeros(n_nodes)
node_out_degree = np.zeros(n_nodes)
# We need to infer the adjacency from the edge parameters
# Since edges are grouped by (source_type, target_type, du, dv), we count unique pairs
print(f"\n  Edge sign shape: {edge_signs.shape}")
print(f"  Synapse count shape: {ref_counts.shape}")

# For visualization, show the synapse count as a function of edge index
axes[2].semilogy(range(len(ref_counts)), ref_counts + 1e-6, 'o', markersize=2, color='seagreen', alpha=0.5)
axes[2].set_xlabel('Edge Group Index')
axes[2].set_ylabel('Synapse Count (log scale)')
axes[2].set_title('C. Synapse Counts by Edge Group')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig6_connectome_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_connectome_structure.png")

# --- Figure 7: Model convergence and optimization landscape ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parameter space: bias vs syn_strength (mean)
axes[0].scatter(mean_biases, all_syn_strengths.mean(axis=0)[:n_nodes] if all_syn_strengths.shape[1] >= n_nodes else np.zeros(n_nodes),
                c=node_idx, cmap='tab20', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Mean Resting Potential')
axes[0].set_ylabel('Mean Synapse Strength (first N)')
axes[0].set_title('A. Parameter Space: Nodes')

# Validation loss vs parameter diversity
param_diversity = []
for i, mid in enumerate(model_ids):
    if mid in all_params:
        # Compute diversity as std of biases
        diversity = all_biases[i].std()
        param_diversity.append(diversity)

axes[1].scatter(param_diversity, all_val_losses, c=range(len(all_val_losses)), cmap='RdYlGn_r', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Parameter Diversity (std of biases)')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('B. Loss vs Parameter Diversity')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig7_optimization_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_optimization_landscape.png")

# ============================================================================
# 9. Save intermediate results
# ============================================================================
print("\n" + "=" * 60)
print("Saving intermediate results...")
print("=" * 60)

results = {
    'n_models': len(all_params),
    'n_nodes': n_nodes,
    'n_edge_types': n_edge_types,
    'n_synapse_groups': n_synapse_groups,
    'n_excitatory_edges': int(n_excitatory),
    'n_inhibitory_edges': int(n_inhibitory),
    'n_cell_types': len(cell_type_names),
    'cell_type_names': cell_type_names,
    'validation_loss': {
        'mean': float(all_val_losses.mean()),
        'std': float(all_val_losses.std()),
        'min': float(all_val_losses.min()),
        'max': float(all_val_losses.max()),
        'best_model': model_ids[np.argmin(all_val_losses)],
    },
    'resting_potentials': {
        'mean': float(mean_biases.mean()),
        'std': float(mean_biases.std()),
        'range': [float(mean_biases.min()), float(mean_biases.max())],
    },
    'time_constants': {
        'mean': float(mean_tc.mean()),
        'std': float(mean_tc.std()),
        'range': [float(mean_tc.min()), float(mean_tc.max())],
    },
    'synapse_counts': {
        'mean': float(ref_counts[ref_counts > 0].mean()) if np.any(ref_counts > 0) else 0,
        'median': float(np.median(ref_counts[ref_counts > 0])) if np.any(ref_counts > 0) else 0,
        'non_zero_fraction': float(np.sum(ref_counts > 0) / len(ref_counts)),
    },
    'synapse_polarity': {
        'excitatory_fraction': float(n_excitatory / len(edge_signs)),
        'inhibitory_fraction': float(n_inhibitory / len(edge_signs)),
    },
}

with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved analysis_results.json")

# Save raw parameter arrays
np.save(OUTPUT_DIR / 'all_biases.npy', all_biases)
np.save(OUTPUT_DIR / 'all_time_constants.npy', all_time_consts)
np.save(OUTPUT_DIR / 'all_syn_strengths.npy', all_syn_strengths)
np.save(OUTPUT_DIR / 'all_val_losses.npy', all_val_losses)
np.save(OUTPUT_DIR / 'edge_signs.npy', edge_signs)
np.save(OUTPUT_DIR / 'synapse_counts.npy', ref_counts)
print("  Saved numpy arrays")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
