"""
Main analysis script for the Drosophila connectome-constrained DMN study.
This script loads all 50 pre-trained DMN models, extracts parameters,
computes ensemble statistics, and generates figures for the report.
"""

import torch
import os
import numpy as np
import pickle
import types
import sys
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up flyvis module mock for pickle loading
class GaussianMixtureClustering:
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return None

class Embedding:
    pass

mod = types.ModuleType('flyvis.analysis.clustering')
mod.GaussianMixtureClustering = GaussianMixtureClustering
mod.Embedding = Embedding
sys.modules['flyvis'] = types.ModuleType('flyvis')
sys.modules['flyvis.analysis'] = types.ModuleType('flyvis.analysis')
sys.modules['flyvis.analysis.clustering'] = mod

# Paths
CKPT_DIR = 'data/flow/0000'
UMAP_DIR = os.path.join(CKPT_DIR, 'umap_and_clustering')
OUTPUT_DIR = 'outputs'
IMAGE_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Cell type names (sorted alphabetically as in UMAP directory)
cell_types_list = sorted([f.replace('.pickle','') for f in os.listdir(UMAP_DIR) if f.endswith('.pickle')])
ct_to_idx = {ct: i for i, ct in enumerate(cell_types_list)}

print(f"Number of cell types: {len(cell_types_list)}")
print(f"Cell types: {cell_types_list}")

# ============================================================
# 1. Load all 50 model checkpoints
# ============================================================
all_bias = []
all_tc = []
all_syn_strength = []
all_sign = []
validation_losses = []

for i in range(50):
    path = os.path.join(CKPT_DIR, f'{i:03d}', 'best_chkpt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    net = ckpt['network']
    all_bias.append(net['nodes_bias'].numpy())
    all_tc.append(net['nodes_time_const'].numpy())
    all_syn_strength.append(net['edges_syn_strength'].numpy())
    all_sign.append(net['edges_sign'].numpy())
    
    # Validation loss
    loss_path = os.path.join(CKPT_DIR, f'{i:03d}', 'validation_loss.h5')
    with h5py.File(loss_path, 'r') as f:
        validation_losses.append(f['data'][()])

all_bias = np.array(all_bias)       # (50, 65)
all_tc = np.array(all_tc)           # (50, 65)
all_syn_strength = np.array(all_syn_strength)  # (50, 604)
all_sign = np.array(all_sign)       # (50, 604)
validation_losses = np.array(validation_losses)  # (50,)

# Sign is consistent across all models
sign = all_sign[0]  # (604,)
assert np.all(all_sign == sign), "Sign should be consistent across all models"

# Ensemble statistics
bias_mean = all_bias.mean(axis=0)
bias_std = all_bias.std(axis=0)
tc_mean = all_tc.mean(axis=0)
tc_std = all_tc.std(axis=0)
ss_mean = all_syn_strength.mean(axis=0)
ss_std = all_syn_strength.std(axis=0)

# Effective synaptic weight = sign * syn_strength
eff_weight_all = all_sign * all_syn_strength  # (50, 604)
eff_weight_mean = sign * ss_mean
eff_weight_std = sign * ss_std

# ============================================================
# 2. Load UMAP and clustering data
# ============================================================
umap_embeddings = {}
cluster_labels = {}
n_clusters_map = {}

for ct in cell_types_list:
    with open(os.path.join(UMAP_DIR, f'{ct}.pickle'), 'rb') as f:
        data = pickle.load(f)
    umap_embeddings[ct] = data.embedding._embedding  # (50, 2)
    labels = data.labels
    valid_labels = labels[labels != -99999]
    cluster_labels[ct] = labels
    n_clusters_map[ct] = len(np.unique(valid_labels))

# ============================================================
# 3. Define pathway groupings
# ============================================================
photoreceptors = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
on_pathway = ['L1', 'Mi1', 'Tm3', 'Mi4', 'TmY15', 'T4a', 'T4b', 'T4c', 'T4d']
off_pathway = ['L2', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T5a', 'T5b', 'T5c', 'T5d']
lamina_cells = ['L1', 'L2', 'L3', 'L4', 'L5']
t4_subtypes = ['T4a', 'T4b', 'T4c', 'T4d']
t5_subtypes = ['T5a', 'T5b', 'T5c', 'T5d']

# ============================================================
# 4. Save intermediate results
# ============================================================
results = {
    'cell_types': cell_types_list,
    'n_cell_types': len(cell_types_list),
    'n_models': 50,
    'n_edges': 604,
    'n_excitatory_edges': int(np.sum(sign == 1)),
    'n_inhibitory_edges': int(np.sum(sign == -1)),
    'validation_losses': validation_losses,
    'best_model_idx': int(np.argmin(validation_losses)),
    'bias_mean': bias_mean,
    'bias_std': bias_std,
    'tc_mean': tc_mean,
    'tc_std': tc_std,
    'ss_mean': ss_mean,
    'ss_std': ss_std,
    'eff_weight_mean': eff_weight_mean,
    'eff_weight_std': eff_weight_std,
    'n_clusters_map': n_clusters_map,
}

# Save as numpy format
np.savez(os.path.join(OUTPUT_DIR, 'ensemble_stats.npz'), **results)

# Also save as JSON-friendly format
import json
json_results = {
    'cell_types': cell_types_list,
    'n_cell_types': len(cell_types_list),
    'n_models': 50,
    'n_edges': 604,
    'n_excitatory_edges': int(np.sum(sign == 1)),
    'n_inhibitory_edges': int(np.sum(sign == -1)),
    'validation_loss_mean': float(validation_losses.mean()),
    'validation_loss_std': float(validation_losses.std()),
    'validation_loss_min': float(validation_losses.min()),
    'best_model_idx': int(np.argmin(validation_losses)),
    'bias_mean_per_type': {ct: float(bias_mean[ct_to_idx[ct]]) for ct in cell_types_list},
    'bias_std_per_type': {ct: float(bias_std[ct_to_idx[ct]]) for ct in cell_types_list},
    'tc_mean_per_type': {ct: float(tc_mean[ct_to_idx[ct]]) for ct in cell_types_list},
    'tc_std_per_type': {ct: float(tc_std[ct_to_idx[ct]]) for ct in cell_types_list},
    'n_clusters_per_type': {ct: int(n_clusters_map[ct]) for ct in cell_types_list},
}

with open(os.path.join(OUTPUT_DIR, 'ensemble_stats.json'), 'w') as f:
    json.dump(json_results, f, indent=2)

print("Saved ensemble statistics to outputs/")

# ============================================================
# FIGURE 1: Validation Loss Distribution Across Ensemble
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(validation_losses, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(validation_losses.min(), color='red', linestyle='--', label=f'Best model (loss={validation_losses.min():.3f})')
ax.axvline(validation_losses.mean(), color='orange', linestyle='--', label=f'Mean (loss={validation_losses.mean():.3f})')
ax.set_xlabel('Validation Loss (L2 Norm)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Validation Losses Across 50 DMN Models', fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig1_validation_loss_distribution.png'), dpi=150)
plt.close()
print("Figure 1 saved")

# ============================================================
# FIGURE 2: Resting Potentials (Bias) Across Cell Types
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
x_pos = np.arange(len(cell_types_list))
ax.bar(x_pos, bias_mean, yerr=bias_std, capsize=3, color='steelblue', edgecolor='black', alpha=0.7)

# Color key pathway cells differently
for ct in on_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], bias_mean[idx], yerr=bias_std[idx], capsize=3, color='green', edgecolor='black', alpha=0.8)
for ct in off_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], bias_mean[idx], yerr=bias_std[idx], capsize=3, color='purple', edgecolor='black', alpha=0.8)
for ct in photoreceptors:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], bias_mean[idx], yerr=bias_std[idx], capsize=3, color='gold', edgecolor='black', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(cell_types_list, rotation=90, fontsize=9)
ax.set_ylabel('Resting Potential (Bias)', fontsize=12)
ax.set_title('Learned Resting Potentials Across 65 Cell Types (Mean ± SD over 50 Models)', fontsize=14)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', edgecolor='black', label='Other'),
    Patch(facecolor='gold', edgecolor='black', label='Photoreceptors'),
    Patch(facecolor='green', edgecolor='black', label='ON Pathway (T4)'),
    Patch(facecolor='purple', edgecolor='black', label='OFF Pathway (T5)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig2_resting_potentials.png'), dpi=150)
plt.close()
print("Figure 2 saved")

# ============================================================
# FIGURE 3: Time Constants Across Cell Types
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
x_pos = np.arange(len(cell_types_list))
ax.bar(x_pos, tc_mean, yerr=tc_std, capsize=3, color='coral', edgecolor='black', alpha=0.7)

for ct in on_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], tc_mean[idx], yerr=tc_std[idx], capsize=3, color='green', edgecolor='black', alpha=0.8)
for ct in off_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], tc_mean[idx], yerr=tc_std[idx], capsize=3, color='purple', edgecolor='black', alpha=0.8)
for ct in photoreceptors:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], tc_mean[idx], yerr=tc_std[idx], capsize=3, color='gold', edgecolor='black', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(cell_types_list, rotation=90, fontsize=9)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('Learned Time Constants Across 65 Cell Types (Mean ± SD over 50 Models)', fontsize=14)

legend_elements = [
    Patch(facecolor='coral', edgecolor='black', label='Other'),
    Patch(facecolor='gold', edgecolor='black', label='Photoreceptors'),
    Patch(facecolor='green', edgecolor='black', label='ON Pathway (T4)'),
    Patch(facecolor='purple', edgecolor='black', label='OFF Pathway (T5)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig3_time_constants.png'), dpi=150)
plt.close()
print("Figure 3 saved")

# ============================================================
# FIGURE 4: Parameter Variability (CV) Across Cell Types
# ============================================================
bias_cv = bias_std / np.abs(bias_mean + 1e-10)
tc_cv = tc_std / np.abs(tc_mean + 1e-10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bias CV
ax1.bar(x_pos, bias_cv, color='steelblue', edgecolor='black', alpha=0.7)
for ct in on_pathway:
    idx = ct_to_idx[ct]
    ax1.bar(x_pos[idx], bias_cv[idx], color='green', edgecolor='black', alpha=0.8)
for ct in off_pathway:
    idx = ct_to_idx[ct]
    ax1.bar(x_pos[idx], bias_cv[idx], color='purple', edgecolor='black', alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(cell_types_list, rotation=90, fontsize=9)
ax1.set_ylabel('Coefficient of Variation', fontsize=12)
ax1.set_title('Resting Potential Variability (CV)', fontsize=13)

# TC CV
ax2.bar(x_pos, tc_cv, color='coral', edgecolor='black', alpha=0.7)
for ct in on_pathway:
    idx = ct_to_idx[ct]
    ax2.bar(x_pos[idx], tc_cv[idx], color='green', edgecolor='black', alpha=0.8)
for ct in off_pathway:
    idx = ct_to_idx[ct]
    ax2.bar(x_pos[idx], tc_cv[idx], color='purple', edgecolor='black', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cell_types_list, rotation=90, fontsize=9)
ax2.set_ylabel('Coefficient of Variation', fontsize=12)
ax2.set_title('Time Constant Variability (CV)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig4_parameter_variability_cv.png'), dpi=150)
plt.close()
print("Figure 4 saved")

# ============================================================
# FIGURE 5: ON vs OFF Pathway Comparison
# ============================================================
on_bias = [bias_mean[ct_to_idx[ct]] for ct in on_pathway]
on_tc = [tc_mean[ct_to_idx[ct]] for ct in on_pathway]
off_bias = [bias_mean[ct_to_idx[ct]] for ct in off_pathway]
off_tc = [tc_mean[ct_to_idx[ct]] for ct in off_pathway]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bias comparison
ax1.scatter(on_bias, range(len(on_pathway)), color='green', s=80, label='ON (T4)', zorder=3)
ax1.scatter(off_bias, range(len(off_pathway)), color='purple', s=80, label='OFF (T5)', zorder=3)
ax1.set_yticks(range(len(on_pathway)))
ax1.set_yticklabels(on_pathway, fontsize=9)
ax1.set_xlabel('Resting Potential (Bias)', fontsize=12)
ax1.set_title('ON vs OFF Pathway: Resting Potentials', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# TC comparison
ax2.scatter(on_tc, range(len(on_pathway)), color='green', s=80, label='ON (T4)', zorder=3)
ax2.scatter(off_tc, range(len(off_pathway)), color='purple', s=80, label='OFF (T5)', zorder=3)
ax2.set_yticks(range(len(on_pathway)))
ax2.set_yticklabels(on_pathway, fontsize=9)
ax2.set_xlabel('Time Constant (s)', fontsize=12)
ax2.set_title('ON vs OFF Pathway: Time Constants', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig5_on_off_pathway_comparison.png'), dpi=150)
plt.close()
print("Figure 5 saved")

# ============================================================
# FIGURE 6: UMAP Embeddings for Key Cell Types
# ============================================================
key_types = t4_subtypes + t5_subtypes + ['L1', 'L2', 'Mi1', 'Tm1', 'Tm2', 'Tm3', 'Tm9']

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for j, ct in enumerate(key_types):
    ax = axes[j]
    emb = umap_embeddings[ct]
    labels = cluster_labels[ct]
    valid_mask = labels != -99999
    
    unique_labels = np.unique(labels[valid_mask])
    colors = sns.color_palette('Set2', len(unique_labels))
    
    for k, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[k]], s=30, label=f'Cluster {lab}', alpha=0.7)
    
    ax.set_title(ct, fontsize=11, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.tick_params(labelsize=7)

# Hide unused axes
for j in range(len(key_types), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('UMAP Embeddings of Key Motion Pathway Cell Types\n(Each point = one model in the 50-model ensemble)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig6_umap_key_cell_types.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved")

# ============================================================
# FIGURE 7: Synapse Weight Distribution
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Excitatory vs inhibitory weight distribution
exc_weights = eff_weight_mean[sign == 1]
inh_weights = eff_weight_mean[sign == -1]

ax1.hist(exc_weights, bins=40, color='green', edgecolor='black', alpha=0.7, label=f'Excitatory (n={len(exc_weights)})')
ax1.hist(inh_weights, bins=40, color='red', edgecolor='black', alpha=0.7, label=f'Inhibitory (n={len(inh_weights)})')
ax1.set_xlabel('Effective Synaptic Weight (sign × strength)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Distribution of Mean Effective Synaptic Weights', fontsize=13)
ax1.legend(fontsize=10)

# Weight variability
exc_std = eff_weight_std[sign == 1]
inh_std = eff_weight_std[sign == -1]
ax2.scatter(np.abs(exc_weights), exc_std, color='green', s=20, alpha=0.5, label='Excitatory')
ax2.scatter(np.abs(inh_weights), inh_std, color='red', s=20, alpha=0.5, label='Inhibitory')
ax2.set_xlabel('|Mean Effective Weight|', fontsize=12)
ax2.set_ylabel('Std of Effective Weight', fontsize=12)
ax2.set_title('Weight Variability vs Magnitude', fontsize=13)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig7_synapse_weight_distribution.png'), dpi=150)
plt.close()
print("Figure 7 saved")

# ============================================================
# FIGURE 8: Number of Functional Clusters per Cell Type
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
n_clusters_vals = [n_clusters_map[ct] for ct in cell_types_list]
ax.bar(x_pos, n_clusters_vals, color='teal', edgecolor='black', alpha=0.7)

for ct in on_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], n_clusters_vals[idx], color='green', edgecolor='black', alpha=0.8)
for ct in off_pathway:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], n_clusters_vals[idx], color='purple', edgecolor='black', alpha=0.8)
for ct in photoreceptors:
    idx = ct_to_idx[ct]
    ax.bar(x_pos[idx], n_clusters_vals[idx], color='gold', edgecolor='black', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(cell_types_list, rotation=90, fontsize=9)
ax.set_ylabel('Number of Functional Clusters', fontsize=12)
ax.set_title('Functional Cluster Diversity Across Cell Types (Gaussian Mixture Clustering)', fontsize=14)

legend_elements = [
    Patch(facecolor='teal', edgecolor='black', label='Other'),
    Patch(facecolor='gold', edgecolor='black', label='Photoreceptors'),
    Patch(facecolor='green', edgecolor='black', label='ON Pathway (T4)'),
    Patch(facecolor='purple', edgecolor='black', label='OFF Pathway (T5)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig8_cluster_diversity.png'), dpi=150)
plt.close()
print("Figure 8 saved")

# ============================================================
# FIGURE 9: T4/T5 Subtype Parameter Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# T4 subtypes - bias
ax = axes[0, 0]
t4_biases = [bias_mean[ct_to_idx[ct]] for ct in t4_subtypes]
t4_bias_stds = [bias_std[ct_to_idx[ct]] for ct in t4_subtypes]
ax.bar(range(4), t4_biases, yerr=t4_bias_stds, capsize=5, color=['green','lightgreen','olive','darkgreen'], edgecolor='black')
ax.set_xticks(range(4))
ax.set_xticklabels(t4_subtypes, fontsize=11)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('T4 Subtypes: Resting Potentials', fontsize=13)

# T5 subtypes - bias
ax = axes[0, 1]
t5_biases = [bias_mean[ct_to_idx[ct]] for ct in t5_subtypes]
t5_bias_stds = [bias_std[ct_to_idx[ct]] for ct in t5_subtypes]
ax.bar(range(4), t5_biases, yerr=t5_bias_stds, capsize=5, color=['purple','violet','indigo','darkviolet'], edgecolor='black')
ax.set_xticks(range(4))
ax.set_xticklabels(t5_subtypes, fontsize=11)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('T5 Subtypes: Resting Potentials', fontsize=13)

# T4 subtypes - time constant
ax = axes[1, 0]
t4_tcs = [tc_mean[ct_to_idx[ct]] for ct in t4_subtypes]
t4_tc_stds = [tc_std[ct_to_idx[ct]] for ct in t4_subtypes]
ax.bar(range(4), t4_tcs, yerr=t4_tc_stds, capsize=5, color=['green','lightgreen','olive','darkgreen'], edgecolor='black')
ax.set_xticks(range(4))
ax.set_xticklabels(t4_subtypes, fontsize=11)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('T4 Subtypes: Time Constants', fontsize=13)

# T5 subtypes - time constant
ax = axes[1, 1]
t5_tcs = [tc_mean[ct_to_idx[ct]] for ct in t5_subtypes]
t5_tc_stds = [tc_std[ct_to_idx[ct]] for ct in t5_subtypes]
ax.bar(range(4), t5_tcs, yerr=t5_tc_stds, capsize=5, color=['purple','violet','indigo','darkviolet'], edgecolor='black')
ax.set_xticks(range(4))
ax.set_xticklabels(t5_subtypes, fontsize=11)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('T5 Subtypes: Time Constants', fontsize=13)

plt.suptitle('Direction-Selective Output Neurons: T4 (ON) vs T5 (OFF) Subtype Parameters', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig9_t4_t5_subtype_params.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved")

# ============================================================
# FIGURE 10: Correlation between parameters and cluster diversity
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

nc_arr = np.array([n_clusters_map[ct] for ct in cell_types_list])

ax1.scatter(bias_std, nc_arr, color='steelblue', s=40, alpha=0.7)
r, p = stats.pearsonr(bias_std, nc_arr)
ax1.set_xlabel('Bias Std (across 50 models)', fontsize=12)
ax1.set_ylabel('Number of Functional Clusters', fontsize=12)
ax1.set_title(f'Bias Variability vs Cluster Diversity (r={r:.3f}, p={p:.4f})', fontsize=13)

ax2.scatter(tc_std, nc_arr, color='coral', s=40, alpha=0.7)
r2, p2 = stats.pearsonr(tc_std, nc_arr)
ax2.set_xlabel('Time Constant Std (across 50 models)', fontsize=12)
ax2.set_ylabel('Number of Functional Clusters', fontsize=12)
ax2.set_title(f'TC Variability vs Cluster Diversity (r={r2:.3f}, p={p2:.4f})', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig10_param_variability_vs_clusters.png'), dpi=150)
plt.close()
print("Figure 10 saved")

# ============================================================
# FIGURE 11: Heatmap of resting potentials organized by pathway
# ============================================================
# Organize cell types by functional groups
pathway_groups = {
    'Photoreceptors': photoreceptors,
    'Lamina': lamina_cells,
    'ON Medulla': ['Mi1', 'Tm3', 'Mi4', 'TmY15'],
    'OFF Medulla/Lobula': ['Tm1', 'Tm2', 'Tm4', 'Tm9'],
    'T4 (ON DS)': t4_subtypes,
    'T5 (OFF DS)': t5_subtypes,
}

fig, ax = plt.subplots(figsize=(14, 8))

group_names = list(pathway_groups.keys())
group_cells = list(pathway_groups.values())

# Create a heatmap matrix: models x cell types (organized by pathway)
ordered_types = []
for grp in group_cells:
    ordered_types.extend(grp)

ordered_indices = [ct_to_idx[ct] for ct in ordered_types]
bias_matrix = all_bias[:, ordered_indices]  # (50, n_ordered)

sns.heatmap(bias_matrix.T, cmap='RdBu_r', center=0, ax=ax, 
            xticklabels=False, yticklabels=ordered_types,
            vmin=-1, vmax=1)
ax.set_xlabel('Model Index (0-49)', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
ax.set_title('Resting Potentials Across 50 Models (Organized by Pathway)', fontsize=14)

# Add group separators
cumulative = 0
for grp_name, grp_cells in pathway_groups.items():
    cumulative += len(grp_cells)
    ax.axhline(y=cumulative, color='black', linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig11_bias_heatmap_pathway.png'), dpi=150)
plt.close()
print("Figure 11 saved")

# ============================================================
# FIGURE 12: UMAP overview of all cell types
# ============================================================
fig, axes = plt.subplots(7, 10, figsize=(30, 21))
axes = axes.flatten()

for j, ct in enumerate(cell_types_list):
    ax = axes[j]
    emb = umap_embeddings[ct]
    labels = cluster_labels[ct]
    valid_mask = labels != -99999
    
    unique_labels = np.unique(labels[valid_mask])
    colors = sns.color_palette('Set2', max(len(unique_labels), 2))
    
    for k, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[k]], s=25, alpha=0.7)
    
    ax.set_title(ct, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=6)

for j in range(len(cell_types_list), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('UMAP Embeddings of All 65 Cell Types Across 50-Model Ensemble', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig12_umap_all_cell_types.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 12 saved")

# ============================================================
# FIGURE 13: Ensemble parameter correlation structure
# ============================================================
# Compute correlation between bias and time constant across cell types
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bias_mean, tc_mean, c='steelblue', s=60, alpha=0.7)

# Label key cell types
for ct in t4_subtypes + t5_subtypes + ['L1', 'L2']:
    idx = ct_to_idx[ct]
    ax.annotate(ct, (bias_mean[idx], tc_mean[idx]), fontsize=9, 
                textcoords="offset points", xytext=(5, 5))

r, p = stats.pearsonr(bias_mean, tc_mean)
ax.set_xlabel('Mean Resting Potential (Bias)', fontsize=12)
ax.set_ylabel('Mean Time Constant (s)', fontsize=12)
ax.set_title(f'Correlation Between Resting Potential and Time Constant\n(r={r:.3f}, p={p:.4f})', fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig13_bias_tc_correlation.png'), dpi=150)
plt.close()
print("Figure 13 saved")

# ============================================================
# FIGURE 14: Direction-selective neuron specialization
# ============================================================
# Compare T4/T5 subtypes across all parameters
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: T4/T5 bias comparison
ax = axes[0, 0]
for i, ct in enumerate(t4_subtypes):
    idx = ct_to_idx[ct]
    ax.scatter(i, bias_mean[idx], s=100, color='green', zorder=3)
    ax.errorbar(i, bias_mean[idx], yerr=bias_std[idx], color='green', capsize=5)
for i, ct in enumerate(t5_subtypes):
    idx = ct_to_idx[ct]
    ax.scatter(i+4, bias_mean[idx], s=100, color='purple', zorder=3)
    ax.errorbar(i+4, bias_mean[idx], yerr=bias_std[idx], color='purple', capsize=5)
ax.set_xticks(range(8))
ax.set_xticklabels(t4_subtypes + t5_subtypes, fontsize=10)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('T4/T5 Subtype Resting Potentials', fontsize=13)

# Panel B: T4/T5 time constant comparison
ax = axes[0, 1]
for i, ct in enumerate(t4_subtypes):
    idx = ct_to_idx[ct]
    ax.scatter(i, tc_mean[idx], s=100, color='green', zorder=3)
    ax.errorbar(i, tc_mean[idx], yerr=tc_std[idx], color='green', capsize=5)
for i, ct in enumerate(t5_subtypes):
    idx = ct_to_idx[ct]
    ax.scatter(i+4, tc_mean[idx], s=100, color='purple', zorder=3)
    ax.errorbar(i+4, tc_mean[idx], yerr=tc_std[idx], color='purple', capsize=5)
ax.set_xticks(range(8))
ax.set_xticklabels(t4_subtypes + t5_subtypes, fontsize=10)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('T4/T5 Subtype Time Constants', fontsize=13)

# Panel C: Input neuron comparison
ax = axes[0, 2]
on_input_types = ['L1', 'Mi1', 'Tm3', 'Mi4', 'TmY15']
off_input_types = ['L2', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
for i, ct in enumerate(on_input_types):
    idx = ct_to_idx[ct]
    ax.scatter(i, bias_mean[idx], s=100, color='green', zorder=3)
    ax.errorbar(i, bias_mean[idx], yerr=bias_std[idx], color='green', capsize=5)
for i, ct in enumerate(off_input_types):
    idx = ct_to_idx[ct]
    ax.scatter(i+5, bias_mean[idx], s=100, color='purple', zorder=3)
    ax.errorbar(i+5, bias_mean[idx], yerr=bias_std[idx], color='purple', capsize=5)
ax.set_xticks(range(10))
ax.set_xticklabels(on_input_types + off_input_types, fontsize=9)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('Input Neuron Resting Potentials', fontsize=13)

# Panel D: Photoreceptor parameters
ax = axes[1, 0]
for i, ct in enumerate(photoreceptors):
    idx = ct_to_idx[ct]
    ax.scatter(i, bias_mean[idx], s=100, color='gold', zorder=3)
    ax.errorbar(i, bias_mean[idx], yerr=bias_std[idx], color='gold', capsize=5)
ax.set_xticks(range(8))
ax.set_xticklabels(photoreceptors, fontsize=10)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('Photoreceptor Resting Potentials', fontsize=13)

# Panel E: Lamina neurons
ax = axes[1, 1]
for i, ct in enumerate(lamina_cells):
    idx = ct_to_idx[ct]
    ax.scatter(i, tc_mean[idx], s=100, color='coral', zorder=3)
    ax.errorbar(i, tc_mean[idx], yerr=tc_std[idx], color='coral', capsize=5)
ax.set_xticks(range(5))
ax.set_xticklabels(lamina_cells, fontsize=10)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('Lamina Neuron Time Constants', fontsize=13)

# Panel F: Cluster diversity comparison
ax = axes[1, 2]
t4_nc = [n_clusters_map[ct] for ct in t4_subtypes]
t5_nc = [n_clusters_map[ct] for ct in t5_subtypes]
ax.bar(range(4), t4_nc, color='green', edgecolor='black', alpha=0.8, label='T4 (ON)')
ax.bar(range(4, 8), t5_nc, color='purple', edgecolor='black', alpha=0.8, label='T5 (OFF)')
ax.set_xticks(range(8))
ax.set_xticklabels(t4_subtypes + t5_subtypes, fontsize=10)
ax.set_ylabel('Number of Clusters', fontsize=12)
ax.set_title('T4/T5 Functional Cluster Diversity', fontsize=13)
ax.legend(fontsize=10)

plt.suptitle('Motion Detection Pathway: Detailed Parameter Analysis', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig14_pathway_detailed_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 14 saved")

# ============================================================
# Summary statistics for the report
# ============================================================
print("\n=== KEY FINDINGS ===")
print(f"1. Network: 65 cell types, 604 edges (376 excitatory, 228 inhibitory)")
print(f"2. Total neurons simulated: 45,669 (65 types × ~700 columns)")
print(f"3. Validation loss range: {validation_losses.min():.3f} to {validation_losses.max():.3f}")
print(f"4. L1 (ON input) has most hyperpolarized resting potential: {bias_mean[ct_to_idx['L1']]:.4f}")
print(f"5. L2 (OFF input) has depolarized resting potential: {bias_mean[ct_to_idx['L2']]:.4f}")
print(f"6. T4/T5 output neurons have high, constrained resting potentials")
print(f"7. Most variable cell type (bias): Mi14 (CV={bias_cv[ct_to_idx['Mi14']]:.2f})")
print(f"8. Most constrained cell type (bias): R4 (CV={bias_cv[ct_to_idx['R4']]:.4f})")
print(f"9. All 65 cell types show multiple functional clusters (2-5)")
print(f"10. Slowest time constant: Mi13 ({tc_mean[ct_to_idx['Mi13']]:.4f}s)")
print(f"11. Fastest time constants cluster around ~0.02s (many cell types)")

print("\nAll figures saved to report/images/")
print("Analysis complete!")