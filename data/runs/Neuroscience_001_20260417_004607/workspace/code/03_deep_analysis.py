"""
Analysis Script 3: Deeper analysis - Direction selectivity, pathway analysis, 
and additional UMAP visualizations.
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

# Load connectome
with open('/home/chenyixin/.local/lib/python3.10/site-packages/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    conn = json.load(f)

ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}

# ============================================================
# Figure 14: Direction Selectivity Analysis - T4/T5 subtypes
# ============================================================
# T4a-d and T5a-d represent 4 cardinal directions
# a: front-to-back (→), b: back-to-front (←), c: upward (↑), d: downward (↓)
direction_labels = {'a': 'Front→Back', 'b': 'Back→Front', 'c': 'Upward', 'd': 'Downward'}
t4_cells = ['T4a', 'T4b', 'T4c', 'T4d']
t5_cells = ['T5a', 'T5b', 'T5c', 'T5d']

# Analyze inputs to each T4/T5 subtype
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for row, (cell_group, group_name) in enumerate([(t4_cells, 'T4 (ON)'), (t5_cells, 'T5 (OFF)')]):
    for col, target_ct in enumerate(cell_group):
        ax = axes[row, col]
        target_idx = ct_to_idx[target_ct]
        
        # Find all inputs
        input_weights = {}
        for idx_e, (src, tar) in enumerate(edge_pairs):
            if tar == target_ct and idx_e < len(all_syn_strength[0]):
                # Mean effective weight across models
                mean_ew = (all_sign[:, idx_e] * all_syn_strength[:, idx_e]).mean()
                input_weights[src] = mean_ew
        
        # Sort by absolute weight
        sorted_inputs = sorted(input_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        if sorted_inputs:
            src_names = [s[0] for s in sorted_inputs[:15]]
            weights = [s[1] for s in sorted_inputs[:15]]
            colors = ['#e74c3c' if w > 0 else '#3498db' for w in weights]
            
            ax.barh(range(len(src_names)), weights, color=colors, alpha=0.8)
            ax.set_yticks(range(len(src_names)))
            ax.set_yticklabels(src_names, fontsize=8)
            ax.axvline(0, color='black', linewidth=0.5)
        
        direction = target_ct[-1]
        ax.set_title(f'{target_ct} ({direction_labels[direction]})', fontsize=12)
        ax.set_xlabel('Effective Weight', fontsize=9)

fig.suptitle('Input Weights to Direction-Selective T4 and T5 Neurons\n(Red=Excitatory, Blue=Inhibitory)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'direction_selective_inputs.png'), dpi=150)
plt.close()
print("Saved: direction_selective_inputs.png")

# ============================================================
# Figure 15: Pathway Flow Analysis - Lamina to T4/T5
# ============================================================
# Trace signal flow from photoreceptors through lamina/medulla to T4/T5
# Build adjacency for path analysis
n_types = len(cell_types)
adj = np.zeros((n_types, n_types))
mean_sign_all = all_sign.mean(axis=0)
mean_str_all = all_syn_strength.mean(axis=0)

for idx_e, (src, tar) in enumerate(edge_pairs):
    if idx_e < len(mean_sign_all):
        si = ct_to_idx.get(src, -1)
        ti = ct_to_idx.get(tar, -1)
        if si >= 0 and ti >= 0:
            adj[si, ti] = mean_sign_all[idx_e] * mean_str_all[idx_e]

# Compute in-degree and out-degree
in_degree = (adj != 0).sum(axis=0)
out_degree = (adj != 0).sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

sort_in = np.argsort(in_degree)[::-1]
axes[0].barh(range(n_types), in_degree[sort_in], color='steelblue', alpha=0.8)
axes[0].set_yticks(range(n_types))
axes[0].set_yticklabels([cell_types[i] for i in sort_in], fontsize=6)
axes[0].set_xlabel('In-Degree', fontsize=11)
axes[0].set_title('Number of Input Connection Types', fontsize=13)

sort_out = np.argsort(out_degree)[::-1]
axes[1].barh(range(n_types), out_degree[sort_out], color='coral', alpha=0.8)
axes[1].set_yticks(range(n_types))
axes[1].set_yticklabels([cell_types[i] for i in sort_out], fontsize=6)
axes[1].set_xlabel('Out-Degree', fontsize=11)
axes[1].set_title('Number of Output Connection Types', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'degree_distribution.png'), dpi=150)
plt.close()
print("Saved: degree_distribution.png")

# ============================================================
# Figure 16: Learned Synaptic Strength Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All strengths
all_eff = (all_sign * all_syn_strength).flatten()
axes[0].hist(all_eff, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Effective Synaptic Weight', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of All Effective Synaptic Weights\n(sign × strength, all models)', fontsize=12)
axes[0].axvline(0, color='red', linestyle='--')

# Mean strength per edge
mean_eff = (all_sign * all_syn_strength).mean(axis=0)
axes[1].hist(mean_eff, bins=60, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Mean Effective Weight', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of Mean Effective Weights\n(averaged across 50 models)', fontsize=12)
axes[1].axvline(0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'synaptic_weight_distribution.png'), dpi=150)
plt.close()
print("Saved: synaptic_weight_distribution.png")

# ============================================================
# Figure 17: Top Strongest Connections
# ============================================================
mean_eff_weights = (all_sign * all_syn_strength).mean(axis=0)
abs_eff = np.abs(mean_eff_weights)
top_k = 30
top_idx = np.argsort(abs_eff)[::-1][:top_k]

fig, ax = plt.subplots(figsize=(12, 8))
labels = [f"{edge_pairs[i][0]} → {edge_pairs[i][1]}" for i in top_idx]
values = [mean_eff_weights[i] for i in top_idx]
colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

ax.barh(range(top_k), values, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(top_k))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Mean Effective Weight', fontsize=12)
ax.set_title(f'Top {top_k} Strongest Synaptic Connections\n(Red=Excitatory, Blue=Inhibitory)', fontsize=14)
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'top_connections.png'), dpi=150)
plt.close()
print("Saved: top_connections.png")

# ============================================================
# Figure 18: UMAP for Lamina and Medulla cells
# ============================================================
lamina_cells = ['L1', 'L2', 'L3', 'L4', 'L5']
medulla_cells = ['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm9']
umap_dir = os.path.join(BASE, 'data/flow/0000/umap_and_clustering')

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
cells_to_plot = lamina_cells + medulla_cells[:3]

for idx, ct in enumerate(cells_to_plot):
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
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, f'{ct}\nnot found', ha='center', va='center', transform=ax.transAxes)

fig.suptitle('UMAP Embeddings and Clustering of Lamina and Medulla Neurons\n(50 models embedded in 2D)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'lamina_medulla_umap.png'), dpi=150)
plt.close()
print("Saved: lamina_medulla_umap.png")

# ============================================================
# Figure 19: Number of neurons per cell type (from connectome extent)
# ============================================================
# The extent is 15 (hex grid), so compute number of columns
# For a hex grid with extent r, number of positions = 3r(r-1)+1
extent = 15
n_columns = 3 * extent * (extent - 1) + 1
print(f"Number of columns (hex grid extent {extent}): {n_columns}")

# Each cell type has one neuron per column (for most types)
# Total neurons = 65 * n_columns (approximately)
total_neurons = 65 * n_columns  # Simplified
print(f"Approximate total neurons: {total_neurons}")
# The paper says 45,669 neurons

# Compute actual neuron counts from connectome patterns
neuron_counts = {}
for node in conn['nodes']:
    name = node['name']
    pattern = node['pattern']
    # pattern[0] is 'stride', pattern[1] is [dx, dy]
    # For stride [1,1], one neuron per column
    # For stride [2,2], one neuron per 4 columns, etc.
    stride = pattern[1]
    # Approximate count
    count = n_columns // (stride[0] * stride[1]) if stride[0] > 0 and stride[1] > 0 else n_columns
    neuron_counts[name] = count

total_actual = sum(neuron_counts.values())
print(f"Estimated total neurons: {total_actual}")

fig, ax = plt.subplots(figsize=(14, 8))
cts = list(neuron_counts.keys())
counts = [neuron_counts[ct] for ct in cts]
colors = []
cat_map = {}
for ct in cts:
    for cat, members in {
        'Photoreceptors': ['R1','R2','R3','R4','R5','R6','R7','R8'],
        'Lamina': ['L1','L2','L3','L4','L5','Lawf1','Lawf2','Am','C2','C3'],
        'Medulla intrinsic': ['Mi1','Mi2','Mi3','Mi4','Mi9','Mi10','Mi11','Mi12','Mi13','Mi14','Mi15'],
        'T cells': ['T1','T2','T2a','T3','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],
        'Tm cells': ['Tm1','Tm2','Tm3','Tm4','Tm5Y','Tm5a','Tm5b','Tm5c','Tm9','Tm16','Tm20','Tm28','Tm30'],
        'TmY cells': ['TmY3','TmY4','TmY5a','TmY9','TmY10','TmY13','TmY14','TmY15','TmY18'],
        'CT1': ['CT1(Lo1)','CT1(M10)'],
    }.items():
        if ct in members:
            cat_map[ct] = cat
            break
    else:
        cat_map[ct] = 'Other'

cat_colors2 = {
    'Photoreceptors': '#e41a1c', 'Lamina': '#377eb8', 'Medulla intrinsic': '#4daf4a',
    'T cells': '#ff7f00', 'Tm cells': '#f781bf', 'TmY cells': '#999999', 'CT1': '#dede00', 'Other': '#000000'
}
colors = [cat_colors2[cat_map[ct]] for ct in cts]

ax.bar(range(len(cts)), counts, color=colors, edgecolor='gray', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(cts)))
ax.set_xticklabels(cts, rotation=90, fontsize=7)
ax.set_ylabel('Number of Neurons', fontsize=12)
ax.set_title(f'Estimated Neuron Count per Cell Type (Total ≈ {total_actual})', fontsize=14)
handles = [mpatches.Patch(color=cat_colors2[cat], label=cat) for cat in cat_colors2 if cat != 'Other']
ax.legend(handles=handles, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'neuron_counts.png'), dpi=150)
plt.close()
print("Saved: neuron_counts.png")

# ============================================================
# Save summary statistics
# ============================================================
summary = {
    'n_models': 50,
    'n_cell_types': len(cell_types),
    'n_edges': len(edge_pairs),
    'n_columns': n_columns,
    'estimated_total_neurons': total_actual,
    'mean_validation_loss': float(all_val_loss.mean()),
    'std_validation_loss': float(all_val_loss.std()),
    'min_validation_loss': float(all_val_loss.min()),
    'max_validation_loss': float(all_val_loss.max()),
    'mean_bias_overall': float(all_bias.mean()),
    'mean_time_const_overall': float(all_time_const.mean()),
}
with open(os.path.join(OUT, 'summary_statistics.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: summary_statistics.json")

print("\n=== All additional figures generated successfully ===")
