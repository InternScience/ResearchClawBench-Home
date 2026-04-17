"""
Analysis Script 2: Generate all figures for the report.
"""
import numpy as np
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Neuroscience_001_20260417_004607'
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report/images')

# Load data
data = np.load(os.path.join(OUT, 'model_parameters.npz'), allow_pickle=True)
all_bias = data['bias']
all_time_const = data['time_const']
all_sign = data['sign']
all_syn_strength = data['syn_strength']
all_val_loss = data['val_loss']
cell_types = list(data['cell_types'])

with open(os.path.join(OUT, 'edge_info.json')) as f:
    edge_info = json.load(f)
edge_pairs = edge_info['edge_pairs']
input_units = edge_info['input_units']
output_units = edge_info['output_units']

n_models = all_bias.shape[0]

# Load connectome
with open('/home/chenyixin/.local/lib/python3.10/site-packages/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    conn = json.load(f)

# Define cell type categories for coloring
categories = {
    'Photoreceptors': ['R1','R2','R3','R4','R5','R6','R7','R8'],
    'Lamina': ['L1','L2','L3','L4','L5','Lawf1','Lawf2','Am','C2','C3'],
    'Medulla intrinsic': ['Mi1','Mi2','Mi3','Mi4','Mi9','Mi10','Mi11','Mi12','Mi13','Mi14','Mi15'],
    'T cells (ON)': ['T4a','T4b','T4c','T4d'],
    'T cells (OFF)': ['T5a','T5b','T5c','T5d'],
    'T cells (other)': ['T1','T2','T2a','T3'],
    'Tm cells': ['Tm1','Tm2','Tm3','Tm4','Tm5Y','Tm5a','Tm5b','Tm5c','Tm9','Tm16','Tm20','Tm28','Tm30'],
    'TmY cells': ['TmY3','TmY4','TmY5a','TmY9','TmY10','TmY13','TmY14','TmY15','TmY18'],
    'CT1': ['CT1(Lo1)','CT1(M10)'],
}

cat_colors = {
    'Photoreceptors': '#e41a1c',
    'Lamina': '#377eb8',
    'Medulla intrinsic': '#4daf4a',
    'T cells (ON)': '#ff7f00',
    'T cells (OFF)': '#984ea3',
    'T cells (other)': '#a65628',
    'Tm cells': '#f781bf',
    'TmY cells': '#999999',
    'CT1': '#dede00',
}

def get_category(ct):
    for cat, members in categories.items():
        if ct in members:
            return cat
    return 'Other'

def get_color(ct):
    cat = get_category(ct)
    return cat_colors.get(cat, '#000000')

# ============================================================
# Figure 1: Validation Loss Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(all_val_loss, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(all_val_loss.mean(), color='red', linestyle='--', label=f'Mean = {all_val_loss.mean():.3f}')
ax.set_xlabel('Validation Loss (L2 norm)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Validation Loss Across 50 DMN Models', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'validation_loss_distribution.png'), dpi=150)
plt.close()
print("Saved: validation_loss_distribution.png")

# ============================================================
# Figure 2: Learned Resting Potentials (Bias) per Cell Type
# ============================================================
mean_bias = all_bias.mean(axis=0)
std_bias = all_bias.std(axis=0)
sort_idx = np.argsort(mean_bias)

fig, ax = plt.subplots(figsize=(14, 6))
colors = [get_color(cell_types[i]) for i in sort_idx]
bars = ax.barh(range(len(cell_types)), mean_bias[sort_idx], xerr=std_bias[sort_idx],
               color=colors, edgecolor='gray', linewidth=0.5, capsize=2, alpha=0.85)
ax.set_yticks(range(len(cell_types)))
ax.set_yticklabels([cell_types[i] for i in sort_idx], fontsize=7)
ax.set_xlabel('Resting Potential (learned bias)', fontsize=12)
ax.set_title('Learned Resting Potentials Across 65 Cell Types (mean ± std, n=50 models)', fontsize=13)
ax.axvline(0, color='black', linewidth=0.5, linestyle='-')

# Legend
handles = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in categories]
ax.legend(handles=handles, fontsize=7, loc='lower right', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'resting_potentials.png'), dpi=150)
plt.close()
print("Saved: resting_potentials.png")

# ============================================================
# Figure 3: Learned Time Constants per Cell Type
# ============================================================
mean_tc = all_time_const.mean(axis=0)
std_tc = all_time_const.std(axis=0)
sort_idx_tc = np.argsort(mean_tc)

fig, ax = plt.subplots(figsize=(14, 6))
colors_tc = [get_color(cell_types[i]) for i in sort_idx_tc]
bars = ax.barh(range(len(cell_types)), mean_tc[sort_idx_tc], xerr=std_tc[sort_idx_tc],
               color=colors_tc, edgecolor='gray', linewidth=0.5, capsize=2, alpha=0.85)
ax.set_yticks(range(len(cell_types)))
ax.set_yticklabels([cell_types[i] for i in sort_idx_tc], fontsize=7)
ax.set_xlabel('Time Constant (learned)', fontsize=12)
ax.set_title('Learned Time Constants Across 65 Cell Types (mean ± std, n=50 models)', fontsize=13)

handles = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in categories]
ax.legend(handles=handles, fontsize=7, loc='lower right', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'time_constants.png'), dpi=150)
plt.close()
print("Saved: time_constants.png")

# ============================================================
# Figure 4: Connectivity Matrix (synapse counts)
# ============================================================
# Build connectivity matrix from connectome
n_types = len(cell_types)
ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
conn_matrix = np.zeros((n_types, n_types))
for e in conn['edges']:
    src_idx = ct_to_idx[e['src']]
    tar_idx = ct_to_idx[e['tar']]
    # Sum synapse counts across offsets
    total_syn = sum(off[1] for off in e['offsets'])
    conn_matrix[src_idx, tar_idx] = total_syn

fig, ax = plt.subplots(figsize=(14, 12))
# Use log scale for better visualization
log_conn = np.log10(conn_matrix + 1)
im = ax.imshow(log_conn, cmap='YlOrRd', aspect='auto', interpolation='nearest')
ax.set_xticks(range(n_types))
ax.set_xticklabels(cell_types, rotation=90, fontsize=6)
ax.set_yticks(range(n_types))
ax.set_yticklabels(cell_types, fontsize=6)
ax.set_xlabel('Target Cell Type', fontsize=12)
ax.set_ylabel('Source Cell Type', fontsize=12)
ax.set_title('Connectome: Synapse Count Matrix (log10 scale)', fontsize=14)
cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('log10(synapse count + 1)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'connectivity_matrix.png'), dpi=150)
plt.close()
print("Saved: connectivity_matrix.png")

# ============================================================
# Figure 5: Effective Synaptic Weight Matrix (sign × strength)
# ============================================================
# Compute mean effective weight per edge across models
mean_sign = all_sign.mean(axis=0)  # (604,) - should be consistent
mean_strength = all_syn_strength.mean(axis=0)  # (604,)
effective_weight = mean_sign * mean_strength

# Build matrix
eff_matrix = np.zeros((n_types, n_types))
for idx, (src, tar) in enumerate(edge_pairs):
    if idx < len(effective_weight):
        src_i = ct_to_idx.get(src, -1)
        tar_i = ct_to_idx.get(tar, -1)
        if src_i >= 0 and tar_i >= 0:
            eff_matrix[src_i, tar_i] = effective_weight[idx]

fig, ax = plt.subplots(figsize=(14, 12))
vmax = np.abs(eff_matrix).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(eff_matrix, cmap='RdBu_r', norm=norm, aspect='auto', interpolation='nearest')
ax.set_xticks(range(n_types))
ax.set_xticklabels(cell_types, rotation=90, fontsize=6)
ax.set_yticks(range(n_types))
ax.set_yticklabels(cell_types, fontsize=6)
ax.set_xlabel('Target Cell Type', fontsize=12)
ax.set_ylabel('Source Cell Type', fontsize=12)
ax.set_title('Effective Synaptic Weight Matrix (sign × strength, mean across 50 models)', fontsize=13)
cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Effective Weight', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'effective_weight_matrix.png'), dpi=150)
plt.close()
print("Saved: effective_weight_matrix.png")

# ============================================================
# Figure 6: Synaptic Sign Distribution (Excitatory vs Inhibitory)
# ============================================================
# Signs are fixed per connectome, check consistency
sign_consensus = np.sign(all_sign.mean(axis=0))
n_exc = (sign_consensus > 0).sum()
n_inh = (sign_consensus < 0).sum()
n_zero = (sign_consensus == 0).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Pie chart
axes[0].pie([n_exc, n_inh, n_zero], labels=[f'Excitatory ({n_exc})', f'Inhibitory ({n_inh})', f'Zero ({n_zero})'],
            colors=['#e74c3c', '#3498db', '#95a5a6'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Synaptic Sign Distribution', fontsize=13)

# Panel B: Sign per source cell type
src_types_unique = sorted(set(ep[0] for ep in edge_pairs))
exc_per_src = {}
inh_per_src = {}
for idx, (src, tar) in enumerate(edge_pairs):
    if idx < len(sign_consensus):
        if src not in exc_per_src:
            exc_per_src[src] = 0
            inh_per_src[src] = 0
        if sign_consensus[idx] > 0:
            exc_per_src[src] += 1
        elif sign_consensus[idx] < 0:
            inh_per_src[src] += 1

src_list = [ct for ct in cell_types if ct in exc_per_src]
exc_counts = [exc_per_src.get(ct, 0) for ct in src_list]
inh_counts = [inh_per_src.get(ct, 0) for ct in src_list]

x = np.arange(len(src_list))
width = 0.4
axes[1].bar(x - width/2, exc_counts, width, label='Excitatory', color='#e74c3c', alpha=0.8)
axes[1].bar(x + width/2, inh_counts, width, label='Inhibitory', color='#3498db', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(src_list, rotation=90, fontsize=6)
axes[1].set_ylabel('Number of Targets', fontsize=11)
axes[1].set_title('Excitatory vs Inhibitory Outputs per Cell Type', fontsize=13)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'synaptic_signs.png'), dpi=150)
plt.close()
print("Saved: synaptic_signs.png")

# ============================================================
# Figure 7: Parameter Consistency Across Models (CV)
# ============================================================
cv_bias = all_bias.std(axis=0) / (np.abs(all_bias.mean(axis=0)) + 1e-8)
cv_tc = all_time_const.std(axis=0) / (np.abs(all_time_const.mean(axis=0)) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bias CV
sort_cv = np.argsort(cv_bias)
axes[0].barh(range(len(cell_types)), cv_bias[sort_cv], 
             color=[get_color(cell_types[i]) for i in sort_cv], alpha=0.85)
axes[0].set_yticks(range(len(cell_types)))
axes[0].set_yticklabels([cell_types[i] for i in sort_cv], fontsize=6)
axes[0].set_xlabel('Coefficient of Variation', fontsize=11)
axes[0].set_title('Resting Potential: Cross-Model Variability', fontsize=12)

# Time constant CV
sort_cv_tc = np.argsort(cv_tc)
axes[1].barh(range(len(cell_types)), cv_tc[sort_cv_tc],
             color=[get_color(cell_types[i]) for i in sort_cv_tc], alpha=0.85)
axes[1].set_yticks(range(len(cell_types)))
axes[1].set_yticklabels([cell_types[i] for i in sort_cv_tc], fontsize=6)
axes[1].set_xlabel('Coefficient of Variation', fontsize=11)
axes[1].set_title('Time Constant: Cross-Model Variability', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'parameter_consistency.png'), dpi=150)
plt.close()
print("Saved: parameter_consistency.png")

# ============================================================
# Figure 8: T4/T5 Direction Selectivity - UMAP Clustering
# ============================================================
ds_cells = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
ds_colors = {
    'T4a': '#e41a1c', 'T4b': '#377eb8', 'T4c': '#4daf4a', 'T4d': '#984ea3',
    'T5a': '#ff7f00', 'T5b': '#a65628', 'T5c': '#f781bf', 'T5d': '#999999',
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
umap_dir = os.path.join(BASE, 'data/flow/0000/umap_and_clustering')

for idx, ct in enumerate(ds_cells):
    ax = axes[idx // 4, idx % 4]
    pkl_path = os.path.join(umap_dir, f'{ct}.pickle')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            clust = pickle.load(f)
        emb = clust.embedding._embedding
        labels = clust.labels
        
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(emb[mask, 0], emb[mask, 1], s=30, alpha=0.7, label=f'Cluster {lab}')
        
        ax.set_title(f'{ct} (n_clusters={len(unique_labels)})', fontsize=12)
        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f'{ct}\nnot found', ha='center', va='center', transform=ax.transAxes)

fig.suptitle('UMAP Embeddings and Clustering of Direction-Selective T4/T5 Neurons\n(50 models embedded in 2D)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'T4_T5_umap_clustering.png'), dpi=150)
plt.close()
print("Saved: T4_T5_umap_clustering.png")

# ============================================================
# Figure 9: ON vs OFF Pathway Comparison
# ============================================================
on_cells = ['T4a', 'T4b', 'T4c', 'T4d']
off_cells = ['T5a', 'T5b', 'T5c', 'T5d']

# Compare resting potentials and time constants
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Resting potentials
on_bias = [all_bias[:, cell_types.index(ct)].mean() for ct in on_cells]
off_bias = [all_bias[:, cell_types.index(ct)].mean() for ct in off_cells]
on_bias_std = [all_bias[:, cell_types.index(ct)].std() for ct in on_cells]
off_bias_std = [all_bias[:, cell_types.index(ct)].std() for ct in off_cells]

x = np.arange(4)
axes[0].bar(x - 0.2, on_bias, 0.35, yerr=on_bias_std, label='T4 (ON)', color='#ff7f00', capsize=3)
axes[0].bar(x + 0.2, off_bias, 0.35, yerr=off_bias_std, label='T5 (OFF)', color='#984ea3', capsize=3)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['a (→)', 'b (←)', 'c (↑)', 'd (↓)'])
axes[0].set_ylabel('Resting Potential', fontsize=11)
axes[0].set_title('Resting Potentials: T4 vs T5', fontsize=13)
axes[0].legend()

# Panel B: Time constants
on_tc = [all_time_const[:, cell_types.index(ct)].mean() for ct in on_cells]
off_tc = [all_time_const[:, cell_types.index(ct)].mean() for ct in off_cells]
on_tc_std = [all_time_const[:, cell_types.index(ct)].std() for ct in on_cells]
off_tc_std = [all_time_const[:, cell_types.index(ct)].std() for ct in off_cells]

axes[1].bar(x - 0.2, on_tc, 0.35, yerr=on_tc_std, label='T4 (ON)', color='#ff7f00', capsize=3)
axes[1].bar(x + 0.2, off_tc, 0.35, yerr=off_tc_std, label='T5 (OFF)', color='#984ea3', capsize=3)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['a (→)', 'b (←)', 'c (↑)', 'd (↓)'])
axes[1].set_ylabel('Time Constant', fontsize=11)
axes[1].set_title('Time Constants: T4 vs T5', fontsize=13)
axes[1].legend()

# Panel C: Input synaptic strengths to T4/T5
# Find all inputs to T4a-d and T5a-d
on_inputs = {}
off_inputs = {}
for idx_e, (src, tar) in enumerate(edge_pairs):
    if idx_e < len(mean_strength):
        if tar in on_cells:
            if src not in on_inputs:
                on_inputs[src] = []
            on_inputs[src].append(effective_weight[idx_e])
        if tar in off_cells:
            if src not in off_inputs:
                off_inputs[src] = []
            off_inputs[src].append(effective_weight[idx_e])

# Get top inputs by absolute weight
all_inputs_to_ds = set(list(on_inputs.keys()) + list(off_inputs.keys()))
input_list = sorted(all_inputs_to_ds)

on_weights = [np.mean(on_inputs.get(src, [0])) for src in input_list]
off_weights = [np.mean(off_inputs.get(src, [0])) for src in input_list]

x2 = np.arange(len(input_list))
axes[2].barh(x2 - 0.2, on_weights, 0.35, label='→ T4 (ON)', color='#ff7f00', alpha=0.8)
axes[2].barh(x2 + 0.2, off_weights, 0.35, label='→ T5 (OFF)', color='#984ea3', alpha=0.8)
axes[2].set_yticks(x2)
axes[2].set_yticklabels(input_list, fontsize=7)
axes[2].set_xlabel('Mean Effective Weight', fontsize=11)
axes[2].set_title('Input Weights to T4 vs T5', fontsize=13)
axes[2].legend(fontsize=9)
axes[2].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'ON_OFF_pathway_comparison.png'), dpi=150)
plt.close()
print("Saved: ON_OFF_pathway_comparison.png")

# ============================================================
# Figure 10: Network Architecture Schematic (simplified)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))

# Position cell types in layers
layer_positions = {
    'Photoreceptors': (0, ['R1','R2','R3','R4','R5','R6','R7','R8']),
    'Lamina': (1, ['L1','L2','L3','L4','L5','Lawf1','Lawf2','Am','C2','C3']),
    'Medulla': (2, ['Mi1','Mi2','Mi3','Mi4','Mi9','Mi10','Mi11','Mi12','Mi13','Mi14','Mi15',
                     'Tm1','Tm2','Tm3','Tm4','Tm5Y','Tm5a','Tm5b','Tm5c','Tm9','Tm16','Tm20','Tm28','Tm30',
                     'CT1(Lo1)','CT1(M10)']),
    'Lobula Plate': (3, ['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']),
    'Other': (2.5, ['T1','T2','T2a','T3','TmY3','TmY4','TmY5a','TmY9','TmY10','TmY13','TmY14','TmY15','TmY18']),
}

pos = {}
for layer_name, (y, members) in layer_positions.items():
    n = len(members)
    for i, ct in enumerate(members):
        x = (i - n/2) * 0.8
        pos[ct] = (x, -y * 3)

# Draw edges (subset for clarity - only strong ones)
for idx_e, (src, tar) in enumerate(edge_pairs):
    if idx_e < len(effective_weight) and src in pos and tar in pos:
        w = effective_weight[idx_e]
        if abs(w) > 0.02:  # Only draw strong connections
            color = '#e74c3c' if w > 0 else '#3498db'
            alpha = min(abs(w) * 10, 0.5)
            ax.plot([pos[src][0], pos[tar][0]], [pos[src][1], pos[tar][1]], 
                    color=color, alpha=alpha, linewidth=abs(w)*20, zorder=1)

# Draw nodes
for ct in cell_types:
    if ct in pos:
        color = get_color(ct)
        ax.scatter(*pos[ct], s=200, c=color, edgecolors='black', linewidth=0.5, zorder=2)
        ax.annotate(ct, pos[ct], fontsize=5, ha='center', va='bottom', 
                   xytext=(0, 8), textcoords='offset points')

# Layer labels
for layer_name, (y, _) in layer_positions.items():
    ax.text(-12, -y * 3, layer_name, fontsize=12, fontweight='bold', ha='right', va='center')

ax.set_xlim(-15, 15)
ax.set_ylim(-12, 2)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('DMN Network Architecture\n(Strong connections shown: red=excitatory, blue=inhibitory)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'network_architecture.png'), dpi=150)
plt.close()
print("Saved: network_architecture.png")

# ============================================================
# Figure 11: Synapse Count Distribution
# ============================================================
syn_counts = []
for e in conn['edges']:
    total = sum(off[1] for off in e['offsets'])
    syn_counts.append(total)
syn_counts = np.array(syn_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(syn_counts, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Total Synapse Count', fontsize=12)
axes[0].set_ylabel('Number of Connections', fontsize=12)
axes[0].set_title('Distribution of Synapse Counts per Connection', fontsize=13)

axes[1].hist(np.log10(syn_counts + 1), bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('log10(Synapse Count + 1)', fontsize=12)
axes[1].set_ylabel('Number of Connections', fontsize=12)
axes[1].set_title('Log-Scale Distribution of Synapse Counts', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'synapse_count_distribution.png'), dpi=150)
plt.close()
print("Saved: synapse_count_distribution.png")

# ============================================================
# Figure 12: Heatmap of Learned Parameters (Bias & Time Const) across models
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Bias heatmap
im1 = axes[0].imshow(all_bias.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
axes[0].set_yticks(range(len(cell_types)))
axes[0].set_yticklabels(cell_types, fontsize=5)
axes[0].set_xlabel('Model Index', fontsize=11)
axes[0].set_title('Learned Resting Potentials Across 50 Models', fontsize=13)
plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Resting Potential')

# Time constant heatmap
im2 = axes[1].imshow(all_time_const.T, cmap='viridis', aspect='auto', interpolation='nearest')
axes[1].set_yticks(range(len(cell_types)))
axes[1].set_yticklabels(cell_types, fontsize=5)
axes[1].set_xlabel('Model Index', fontsize=11)
axes[1].set_title('Learned Time Constants Across 50 Models', fontsize=13)
plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Time Constant')

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'parameter_heatmaps.png'), dpi=150)
plt.close()
print("Saved: parameter_heatmaps.png")

# ============================================================
# Figure 13: Correlation between Bias and Time Constant
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
mean_b = all_bias.mean(axis=0)
mean_t = all_time_const.mean(axis=0)
for i, ct in enumerate(cell_types):
    ax.scatter(mean_b[i], mean_t[i], c=get_color(ct), s=60, edgecolors='black', linewidth=0.5, zorder=2)
    ax.annotate(ct, (mean_b[i], mean_t[i]), fontsize=5, ha='left', va='bottom',
               xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Mean Resting Potential', fontsize=12)
ax.set_ylabel('Mean Time Constant', fontsize=12)
ax.set_title('Relationship Between Learned Resting Potential and Time Constant', fontsize=13)
handles = [mpatches.Patch(color=cat_colors[cat], label=cat) for cat in categories]
ax.legend(handles=handles, fontsize=7, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'bias_vs_timeconstant.png'), dpi=150)
plt.close()
print("Saved: bias_vs_timeconstant.png")

print("\n=== All figures generated successfully ===")
