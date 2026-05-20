#!/usr/bin/env python3
"""
Comprehensive analysis of connectome-constrained Deep Mechanistic Networks
for motion detection in the Drosophila visual system.

This script:
1. Loads parameters from all 50 pre-trained DMN models
2. Analyzes resting potentials, time constants, synapse strengths
3. Analyzes cell type clustering from UMAP embeddings
4. Generates figures for the research report
"""

import numpy as np
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster import hierarchy

# ============================================================
# Setup
# ============================================================
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Neuroscience_001_20260516_064127'
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMG_DIR = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print("Loading model parameters...")
params = np.load(os.path.join(OUTPUT_DIR, 'all_params.npz'))
nodes_bias = params['nodes_bias']          # (50, 65)
nodes_time_const = params['nodes_time_const']  # (50, 65)
edges_sign = params['edges_sign']          # (50, 604)
edges_syn_count = params['edges_syn_count']  # (50, 2355)
edges_syn_strength = params['edges_syn_strength']  # (50, 604)
losses = params['loss']                    # (50,)

with open(os.path.join(OUTPUT_DIR, 'cell_type_list.json'), 'r') as f:
    cell_types = json.load(f)

print(f"Loaded {len(cell_types)} cell types, 50 models")

# ============================================================
# Figure 1: Model Performance Overview
# ============================================================
print("Generating Figure 1: Model Performance Overview...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 1a: Loss distribution
ax = axes[0]
ax.hist(losses, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(losses.mean(), color='darkred', linestyle='--', linewidth=2, 
           label=f'Mean = {losses.mean():.3f}')
ax.axvline(losses.min(), color='darkgreen', linestyle=':', linewidth=2,
           label=f'Min = {losses.min():.3f}')
ax.set_xlabel('Validation Loss (L2)')
ax.set_ylabel('Number of Models')
ax.set_title('Validation Loss Distribution\nAcross 50 DMN Models')
ax.legend(fontsize=8)

# 1b: Loss rank plot
ax = axes[1]
sorted_losses = np.sort(losses)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_losses)))
ax.bar(range(len(sorted_losses)), sorted_losses, color=colors, edgecolor='white', linewidth=0.3)
ax.set_xlabel('Model Rank')
ax.set_ylabel('Validation Loss')
ax.set_title('Ranked Model Performance')
ax.axhline(losses.mean(), color='black', linestyle='--', linewidth=1, alpha=0.5)

# 1c: Convergence analysis (std across models)
ax = axes[2]
param_names = ['Resting\nPotential', 'Time\nConstant', 'Synapse\nSign', 'Synapse\nStrength']
param_stds = [
    np.mean(np.std(nodes_bias, axis=0)),
    np.mean(np.std(nodes_time_const, axis=0)),
    np.mean(np.std(edges_sign, axis=0)),
    np.mean(np.std(edges_syn_strength, axis=0))
]
param_means = [
    np.mean(np.abs(nodes_bias)),
    np.mean(np.abs(nodes_time_const)),
    np.mean(np.abs(edges_sign)),
    np.mean(np.abs(edges_syn_strength))
]
cv = [s/m if m > 0 else 0 for s, m in zip(param_stds, param_means)]
bars = ax.bar(param_names, cv, color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'], edgecolor='white')
ax.set_ylabel('Coefficient of Variation\n(std/mean across models)')
ax.set_title('Parameter Convergence\nAcross Ensemble')
for bar, val in zip(bars, cv):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure1_model_performance.png'))
plt.close()
print("  -> figure1_model_performance.png saved")

# ============================================================
# Figure 2: Parameter Distributions
# ============================================================
print("Generating Figure 2: Parameter Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 2a: Resting potentials
ax = axes[0, 0]
mean_bias = nodes_bias.mean(axis=0)
std_bias = nodes_bias.std(axis=0)
sorted_idx = np.argsort(mean_bias)
ax.errorbar(range(65), mean_bias[sorted_idx], yerr=std_bias[sorted_idx], 
            fmt='o', markersize=3, capsize=2, color='steelblue', alpha=0.7, elinewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Cell Type Index (sorted)')
ax.set_ylabel('Resting Potential')
ax.set_title('Resting Potentials Across Cell Types')
# Mark key cell types
key_types = ['L1', 'L2', 'Mi1', 'Mi4', 'Mi9', 'T4a', 'T4b', 'T5a', 'T5b', 'Tm1', 'Tm2', 'Tm3', 'Tm9']
for kt in key_types:
    if kt in cell_types:
        idx = cell_types.index(kt)
        sorted_pos = np.where(sorted_idx == idx)[0][0]
        ax.annotate(kt, (sorted_pos, mean_bias[idx]), fontsize=6, 
                    xytext=(0, 8), textcoords='offset points', ha='center',
                    color='darkred', fontweight='bold')

# 2b: Time constants
ax = axes[0, 1]
mean_tc = nodes_time_const.mean(axis=0)
std_tc = nodes_time_const.std(axis=0)
sorted_idx_tc = np.argsort(mean_tc)
ax.errorbar(range(65), mean_tc[sorted_idx_tc], yerr=std_tc[sorted_idx_tc],
            fmt='o', markersize=3, capsize=2, color='forestgreen', alpha=0.7, elinewidth=0.5)
ax.set_xlabel('Cell Type Index (sorted)')
ax.set_ylabel('Time Constant')
ax.set_title('Membrane Time Constants Across Cell Types')
for kt in key_types:
    if kt in cell_types:
        idx = cell_types.index(kt)
        sorted_pos = np.where(sorted_idx_tc == idx)[0][0]
        ax.annotate(kt, (sorted_pos, mean_tc[idx]), fontsize=6,
                    xytext=(0, 8), textcoords='offset points', ha='center',
                    color='darkred', fontweight='bold')

# 2c: Synapse sign distribution
ax = axes[1, 0]
mean_sign = edges_sign.mean(axis=0)
ax.hist(mean_sign, bins=40, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Mean Synapse Sign')
ax.set_ylabel('Number of Edges')
ax.set_title(f'Synapse Sign Distribution\n({len(mean_sign)} type-to-type edges)')
n_exc = np.sum(mean_sign > 0.1)
n_inh = np.sum(mean_sign < -0.1)
n_none = len(mean_sign) - n_exc - n_inh
ax.text(0.95, 0.95, f'Exc: {n_exc}\nInh: {n_inh}\nNeutral: {n_none}', 
        transform=ax.transAxes, va='top', ha='right', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2d: Synapse strength distribution
ax = axes[1, 1]
mean_str = edges_syn_strength.mean(axis=0)
ax.hist(mean_str[mean_str > 0], bins=50, color='crimson', edgecolor='white', alpha=0.8)
ax.set_xlabel('Mean Synapse Strength')
ax.set_ylabel('Number of Edges')
ax.set_title(f'Synapse Strength Distribution\n(Non-zero edges: {np.sum(mean_str > 1e-6)})')

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure2_parameter_distributions.png'))
plt.close()
print("  -> figure2_parameter_distributions.png saved")

# ============================================================
# Figure 3: Cell Type Functional Organization
# ============================================================
print("Generating Figure 3: Cell Type Functional Organization...")
# Group cell types by family
families = {
    'Photoreceptors': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'],
    'Lamina': ['L1', 'L2', 'L3', 'L4', 'L5'],
    'Medulla Mi': ['Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi9', 'Mi10', 'Mi11', 'Mi12', 'Mi13', 'Mi14', 'Mi15'],
    'Medulla Tm': ['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30'],
    'Medulla TmY': ['TmY3', 'TmY4', 'TmY5a', 'TmY9', 'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18', 'Tm5Y'],
    'T4 (ON)': ['T4a', 'T4b', 'T4c', 'T4d'],
    'T5 (OFF)': ['T5a', 'T5b', 'T5c', 'T5d'],
    'T-cells': ['T1', 'T2', 'T2a', 'T3'],
    'Other Medulla': ['Tm5a', 'Tm5b', 'Tm5c'],
    'Lobula/Lobula Plate': ['Lawf1', 'Lawf2', 'Am', 'C2', 'C3', 'CT1(Lo1)', 'CT1(M10)'],
}

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 3a: Heatmap of bias parameters by family
ax1 = fig.add_subplot(gs[0, :2])
# Reorder cell types by family
family_order = list(families.keys())
ordered_types = []
family_boundaries = []
for fam in family_order:
    fam_types = [ct for ct in families[fam] if ct in cell_types]
    ordered_types.extend(fam_types)
    if len(ordered_types) > 0:
        family_boundaries.append(len(ordered_types))

mean_bias_all = nodes_bias.mean(axis=0)
bias_matrix = np.zeros((len(ordered_types), 50))
for i, ct in enumerate(ordered_types):
    idx = cell_types.index(ct)
    bias_matrix[i] = nodes_bias[:, idx]

im = ax1.imshow(bias_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax1.set_yticks(range(len(ordered_types)))
ax1.set_yticklabels(ordered_types, fontsize=7)
ax1.set_xlabel('Model Index')
ax1.set_ylabel('Cell Type')
ax1.set_title('Resting Potentials Across 50 Models\n(Organized by Cell Family)')
# Add family boundaries
for fb in family_boundaries[:-1]:
    ax1.axhline(y=fb - 0.5, color='black', linewidth=1.5, linestyle='-')
plt.colorbar(im, ax=ax1, shrink=0.8, label='Resting Potential')

# 3b: Family-level bias comparison
ax2 = fig.add_subplot(gs[0, 2])
family_biases = {}
family_tc = {}
for fam, members in families.items():
    indices = [cell_types.index(m) for m in members if m in cell_types]
    if indices:
        family_biases[fam] = np.mean(nodes_bias[:, indices])
        family_tc[fam] = np.mean(nodes_time_const[:, indices])

fam_names = list(family_biases.keys())
x_pos = range(len(fam_names))
colors_fam = plt.cm.tab10(np.linspace(0, 1, len(fam_names)))
ax2.barh(x_pos, [family_biases[f] for f in fam_names], color=colors_fam, edgecolor='white')
ax2.set_yticks(x_pos)
ax2.set_yticklabels(fam_names, fontsize=8)
ax2.set_xlabel('Mean Resting Potential')
ax2.set_title('Resting Potential by Cell Family')
ax2.axvline(x=0, color='black', linewidth=0.5)

# 3c: UMAP embeddings for key cell types
ax3 = fig.add_subplot(gs[1, 0])

# Load pickle data for UMAP
class MockEmbedding:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)
class MockGMM:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('flyvis'):
            if 'Embedding' in name: return MockEmbedding
            if 'GaussianMixtureClustering' in name: return MockGMM
            if 'Clustering' in name: return MockGMM
        if module == 'numpy.core.multiarray' and name == '_reconstruct':
            return np._core.multiarray._reconstruct
        if module == 'numpy' and name == 'ndarray': return np.ndarray
        if module == 'numpy' and name == 'dtype': return np.dtype
        try:
            return super().find_class(module, name)
        except:
            return type('Mock', (), {})

pickle_dir = os.path.join(WORKSPACE, 'data/flow/0000/umap_and_clustering/')

# Plot UMAP for T4 and T5 subtypes
t4_types = ['T4a', 'T4b', 'T4c', 'T4d']
t5_types = ['T5a', 'T5b', 'T5c', 'T5d']
colors_t4 = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, ct in enumerate(t4_types + t5_types):
    path = os.path.join(pickle_dir, f'{ct}.pickle')
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = CustomUnpickler(f).load()
    emb = data.embedding._embedding
    color = colors_t4[i % 4]
    marker = 'o' if i < 4 else 's'
    ax3.scatter(emb[:, 0], emb[:, 1], c=[color], marker=marker, 
                label=ct, s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')
ax3.set_title('UMAP: T4 (ON) and T5 (OFF)\nDirection-Selective Neurons')
ax3.legend(fontsize=6, ncol=2, loc='best')

# 3d: Key input neuron UMAP
ax4 = fig.add_subplot(gs[1, 1])
input_types = ['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm9', 'CT1(M10)']
colors_input = plt.cm.Set2(np.linspace(0, 1, len(input_types)))
for i, ct in enumerate(input_types):
    path = os.path.join(pickle_dir, f'{ct}.pickle')
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = CustomUnpickler(f).load()
    emb = data.embedding._embedding
    ax4.scatter(emb[:, 0], emb[:, 1], c=[colors_input[i]], label=ct, 
                s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
ax4.set_xlabel('UMAP 1')
ax4.set_ylabel('UMAP 2')
ax4.set_title('UMAP: Key Input Neurons\nto Motion Detectors')
ax4.legend(fontsize=6, ncol=2, loc='best')

# 3e: Number of clusters per cell type
ax5 = fig.add_subplot(gs[1, 2])
n_clusters_per_type = {}
for ct in cell_types:
    path = os.path.join(pickle_dir, f'{ct}.pickle')
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = CustomUnpickler(f).load()
    n_clusters_per_type[ct] = len(np.unique(data.labels))

cluster_counts = list(n_clusters_per_type.values())
ax5.hist(cluster_counts, bins=range(1, max(cluster_counts)+2), 
         color='mediumpurple', edgecolor='white', align='left')
ax5.set_xlabel('Number of Clusters')
ax5.set_ylabel('Number of Cell Types')
ax5.set_title('Functional Clustering\nWithin Cell Types')
ax5.set_xticks(range(1, max(cluster_counts)+1))

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure3_cell_type_organization.png'))
plt.close()
print("  -> figure3_cell_type_organization.png saved")

# ============================================================
# Figure 4: Motion Detection Circuit Analysis
# ============================================================
print("Generating Figure 4: Motion Detection Circuit Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 4a: T4/T5 input connectivity pattern
ax = axes[0, 0]
# Extract the direction-selective neuron biases
t4t5_types = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
t4t5_indices = [cell_types.index(ct) for ct in t4t5_types if ct in cell_types]
t4t5_names = [ct for ct in t4t5_types if ct in cell_types]

x = np.arange(len(t4t5_names))
width = 0.35
mean_vals = nodes_bias.mean(axis=0)[t4t5_indices]
std_vals = nodes_bias.std(axis=0)[t4t5_indices]

colors_ds = ['#2196F3', '#2196F3', '#2196F3', '#2196F3', 
             '#FF5722', '#FF5722', '#FF5722', '#FF5722']
bars = ax.bar(x, mean_vals, width, yerr=std_vals, capsize=4, 
              color=colors_ds, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(t4t5_names, fontsize=8)
ax.set_ylabel('Resting Potential')
ax.set_title('T4 (ON-pathway) vs T5 (OFF-pathway)\nResting Potentials')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
legend_elements = [Patch(facecolor='#2196F3', label='T4 (ON)'),
                   Patch(facecolor='#FF5722', label='T5 (OFF)')]
ax.legend(handles=legend_elements, fontsize=8)

# 4b: ON vs OFF pathway parameter comparison
ax = axes[0, 1]
on_inputs = ['Mi1', 'Mi4', 'Mi9', 'Tm3']
off_inputs = ['Mi2', 'Mi3', 'Tm1', 'Tm2', 'Tm4', 'Tm9']

def get_family_mean_bias(family_types):
    indices = [cell_types.index(ct) for ct in family_types if ct in cell_types]
    if indices:
        return [nodes_bias.mean(axis=0)[i] for i in indices], family_types
    return [], []

on_bias, on_names = get_family_mean_bias(on_inputs)
off_bias, off_names = get_family_mean_bias(off_inputs)

all_names = on_names + off_names
all_bias = on_bias + off_bias
colors_onoff = ['#2196F3']*len(on_names) + ['#FF5722']*len(off_names)

y_pos = range(len(all_names))
ax.barh(y_pos, all_bias, color=colors_onoff, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(all_names, fontsize=8)
ax.set_xlabel('Resting Potential')
ax.set_title('ON vs OFF Pathway Input Neurons')
ax.axvline(x=0, color='gray', linewidth=0.5)
legend_elements = [Patch(facecolor='#2196F3', label='ON pathway'),
                   Patch(facecolor='#FF5722', label='OFF pathway')]
ax.legend(handles=legend_elements, fontsize=8)

# 4c: Time constant comparison
ax = axes[1, 0]
all_tc_indices = [cell_types.index(ct) for ct in all_names if ct in cell_types]
tc_vals = nodes_time_const.mean(axis=0)[all_tc_indices]
tc_std = nodes_time_const.std(axis=0)[all_tc_indices]
tc_names_ordered = [ct for ct in all_names if ct in cell_types]

y_pos = range(len(tc_names_ordered))
ax.barh(y_pos, tc_vals, xerr=tc_std, color=colors_onoff[:len(tc_names_ordered)], 
        edgecolor='white', capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(tc_names_ordered, fontsize=8)
ax.set_xlabel('Time Constant')
ax.set_title('Membrane Time Constants\nON vs OFF Pathway')
ax.legend(handles=legend_elements, fontsize=8)

# 4d: Parameter relationship plot
ax = axes[1, 1]
mean_bias_all = nodes_bias.mean(axis=0)
mean_tc_all = nodes_time_const.mean(axis=0)

# Color by family
family_colors = {}
for i, (fam, members) in enumerate(families.items()):
    for m in members:
        family_colors[m] = plt.cm.tab10(i)

for i, ct in enumerate(cell_types):
    color = family_colors.get(ct, 'gray')
    ax.scatter(mean_bias_all[i], mean_tc_all[i], c=[color], s=40, 
               alpha=0.7, edgecolors='black', linewidth=0.3)
    if ct in t4t5_types or ct in on_inputs + off_inputs:
        ax.annotate(ct, (mean_bias_all[i], mean_tc_all[i]), fontsize=5,
                    xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Resting Potential')
ax.set_ylabel('Time Constant')
ax.set_title('Parameter Space:\nResting Potential vs Time Constant')

# Add family legend
legend_elements = []
legend_labels = []
for i, (fam, _) in enumerate(families.items()):
    legend_elements.append(Patch(facecolor=plt.cm.tab10(i), alpha=0.7))
    legend_labels.append(fam)
ax.legend(legend_elements, legend_labels, fontsize=5, loc='upper left', ncol=2,
          bbox_to_anchor=(1.02, 1))

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure4_motion_detection_circuit.png'))
plt.close()
print("  -> figure4_motion_detection_circuit.png saved")

# ============================================================
# Figure 5: Synapse Count Analysis
# ============================================================
print("Generating Figure 5: Synapse Count Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: Distribution of synapse counts
ax = axes[0, 0]
mean_sc = edges_syn_count.mean(axis=0)
ax.hist(mean_sc[mean_sc > 0], bins=60, color='teal', edgecolor='white', alpha=0.8)
ax.set_xlabel('Mean Synapse Count')
ax.set_ylabel('Number of Edges')
ax.set_title(f'Synapse Count Distribution\n({np.sum(mean_sc > 0)} non-zero out of {len(mean_sc)} edges)')
ax.axvline(mean_sc[mean_sc > 0].mean(), color='darkred', linestyle='--', 
           label=f'Mean = {mean_sc[mean_sc > 0].mean():.2f}')
ax.legend(fontsize=8)

# 5b: Synapse count vs strength scatter
ax = axes[0, 1]
nonzero_mask = mean_sc > 0
# syn_count has 2355 entries (with spatial offsets), syn_strength has 604 (type-to-type)
# Use per-edge-type mean syn_count for comparison
mean_sc_per_edge = edges_syn_count.mean(axis=0)
ax.scatter(mean_sc_per_edge[:604] + 1e-6, np.abs(edges_syn_strength.mean(axis=0)) + 1e-10,
           alpha=0.3, s=15, c='steelblue', edgecolors='none')
ax.set_xlabel('Synapse Count')
ax.set_ylabel('|Synapse Strength|')
ax.set_xscale('log')
ax.set_title('Synapse Count vs Strength\n(non-zero edges)')
# Add correlation
valid_mask = (mean_sc_per_edge[:604] > 0) & (np.abs(edges_syn_strength.mean(axis=0)) > 1e-10)
valid_mask = (mean_sc_per_edge[:604] > 0) & (edges_syn_strength.mean(axis=0) != 0)
r, p = stats.pearsonr(np.log10(mean_sc_per_edge[:604][valid_mask]), np.log10(np.abs(edges_syn_strength.mean(axis=0)[valid_mask]) + 1e-10))
ax.text(0.95, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
        va='top', ha='right', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5c: Top connected cell type pairs
ax = axes[1, 0]
# The syn_count has 2355 entries (cell type pairs with (du,dv) offsets)
# edges_sign and syn_strength have 604 entries (cell type pairs only)
# Estimate which connections are strongest
top_n = 30
top_idx = np.argsort(mean_sc)[-top_n:]
ax.barh(range(top_n), mean_sc[top_idx], color='teal', edgecolor='white')
ax.set_xlabel('Mean Synapse Count')
ax.set_title(f'Top {top_n} Connections by Synapse Count')
ax.set_yticks([])

# 5d: Parameter correlation matrix
ax = axes[1, 1]
param_summary = np.column_stack([
    nodes_bias.mean(axis=0),
    nodes_time_const.mean(axis=0),
])
# Per-cell-type summary
corr_labels = ['Bias', 'TimeConst']
corr_mat = np.corrcoef(param_summary.T)
im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels)
ax.set_yticks(range(len(corr_labels)))
ax.set_yticklabels(corr_labels)
ax.set_title('Parameter Correlations\nAcross Cell Types')
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax.text(j, i, f'{corr_mat[i,j]:.3f}', ha='center', va='center',
                fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure5_synapse_analysis.png'))
plt.close()
print("  -> figure5_synapse_analysis.png saved")

# ============================================================
# Figure 6: Ensemble Analysis and Model Agreement
# ============================================================
print("Generating Figure 6: Ensemble Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: Bias consistency across models
ax = axes[0, 0]
# Select top varying and least varying parameters
bias_stds = nodes_bias.std(axis=0)
top10_var = np.argsort(bias_stds)[-10:]
bot10_var = np.argsort(bias_stds)[:10]

for i, idx in enumerate(top10_var):
    ax.plot(range(50), nodes_bias[:, idx], alpha=0.5, linewidth=0.8, 
            label=cell_types[idx])
ax.set_xlabel('Model Index')
ax.set_ylabel('Resting Potential')
ax.set_title('High-Variance Cell Types\n(most sensitive to initialization)')
ax.legend(fontsize=6, ncol=2)

# 6b: Low-variance parameters
ax = axes[0, 1]
for i, idx in enumerate(bot10_var):
    ax.plot(range(50), nodes_bias[:, idx], alpha=0.5, linewidth=0.8,
            label=cell_types[idx])
ax.set_xlabel('Model Index')
ax.set_ylabel('Resting Potential')
ax.set_title('Low-Variance Cell Types\n(most robust to initialization)')
ax.legend(fontsize=6, ncol=2)

# 6c: Loss vs parameter summary
ax = axes[1, 0]
# Average absolute bias per model
mean_abs_bias = np.mean(np.abs(nodes_bias), axis=1)
ax.scatter(mean_abs_bias, losses, c=range(50), cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel('Mean |Resting Potential|')
ax.set_ylabel('Validation Loss')
ax.set_title('Loss vs Mean |Bias|')
r, p = stats.pearsonr(mean_abs_bias, losses)
ax.text(0.95, 0.05, f'r = {r:.3f}\np = {p:.3f}', transform=ax.transAxes,
        va='bottom', ha='right', fontsize=9)

# 6d: Model ranking heatmap
ax = axes[1, 1]
# Show relative bias values across top models for key types
key_display_types = on_inputs + off_inputs + t4t5_types
key_indices = [cell_types.index(ct) for ct in key_display_types if ct in cell_types]
key_names = [ct for ct in key_display_types if ct in cell_types]

top_n_models = 20
best_model_indices = np.argsort(losses)[:top_n_models]

bias_subset = nodes_bias[best_model_indices][:, key_indices]
im = ax.imshow(bias_subset.T, aspect='auto', cmap='RdBu_r')
ax.set_yticks(range(len(key_names)))
ax.set_yticklabels(key_names, fontsize=7)
ax.set_xlabel('Model (sorted by loss, best first)')
ax.set_title(f'Resting Potentials: Top {top_n_models} Models\nKey Motion Circuit Neurons')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'figure6_ensemble_analysis.png'))
plt.close()
print("  -> figure6_ensemble_analysis.png saved")

# ============================================================
# Save summary statistics
# ============================================================
print("\nSaving summary statistics...")
summary = {
    'n_models': 50,
    'n_cell_types': 65,
    'n_edge_types': 604,
    'n_spatial_edges': 2355,
    'loss': {
        'mean': float(losses.mean()),
        'std': float(losses.std()),
        'min': float(losses.min()),
        'max': float(losses.max()),
        'best_model': int(np.argmin(losses)),
    },
    'parameters': {
        'nodes_bias_mean': float(np.mean(np.abs(nodes_bias))),
        'nodes_bias_std': float(np.mean(np.std(nodes_bias, axis=0))),
        'nodes_time_const_mean': float(np.mean(np.abs(nodes_time_const))),
        'nodes_time_const_std': float(np.mean(np.std(nodes_time_const, axis=0))),
        'edges_sign_mean': float(np.mean(np.abs(edges_sign))),
        'edges_syn_strength_mean': float(np.mean(np.abs(edges_syn_strength))),
    },
    'connectivity': {
        'n_excitatory': int(np.sum(edges_sign.mean(axis=0) > 0.1)),
        'n_inhibitory': int(np.sum(edges_sign.mean(axis=0) < -0.1)),
        'n_nonzero_synapses': int(np.sum(edges_syn_count.mean(axis=0) > 0)),
        'mean_syn_count': float(edges_syn_count.mean()),
        'max_syn_count': float(edges_syn_count.mean(axis=0).max()),
    },
    'cell_type_clusters': {ct: int(n) for ct, n in n_clusters_per_type.items()},
}

with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("Analysis complete! All figures saved to report/images/")
print(f"Summary statistics saved to {OUTPUT_DIR}/summary_statistics.json")
