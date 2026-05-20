#!/usr/bin/env python3
"""
Comprehensive analysis of connectome-constrained deep mechanistic networks (DMNs)
for Drosophila motion detection.

This script analyzes 50 pre-trained DMN models constrained by the Drosophila
optic lobe connectome, extracting and analyzing learned neural parameters
(resting potentials, time constants, synaptic strengths) across all models.

Usage:
    python code/analyze_dmn.py

Output:
    - report/images/fig1-fig12: Publication figures
    - outputs/: Intermediate data and summaries
"""

import numpy as np
import torch
import h5py
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'data/flow/0000'
N_MODELS = 50
FIG_DIR = 'report/images'
OUT_DIR = 'outputs'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Cell type names (alphabetically sorted, matching model indices)
cell_types = sorted([
    f.replace('.pickle', '')
    for f in os.listdir(f'{DATA_DIR}/umap_and_clustering/')
    if f.endswith('.pickle')
])

# Cell type family mapping
def get_family(ct):
    if ct.startswith('R'): return 'Photoreceptor'
    if ct.startswith('L') or ct.startswith('Lawf'): return 'Lamina'
    if ct.startswith('Mi'): return 'Medulla Intrinsic'
    if ct.startswith('Tm') or ct.startswith('TmY'): return 'Medulla Tangential'
    if ct.startswith('T4') or ct.startswith('T5'): return 'Motion Selective'
    if ct.startswith('T'): return 'Medulla Columnar'
    if ct.startswith('C'): return 'Centrifugal'
    if ct.startswith('CT'): return 'Columnar'
    if ct == 'Am': return 'Amacrine'
    return 'Other'

family_colors = {
    'Photoreceptor': '#e74c3c', 'Lamina': '#3498db',
    'Medulla Intrinsic': '#2ecc71', 'Medulla Tangential': '#f39c12',
    'Motion Selective': '#9b59b6', 'Medulla Columnar': '#1abc9c',
    'Centrifugal': '#e67e22', 'Columnar': '#34495e',
    'Amacrine': '#95a5a6', 'Other': '#bdc3c7'
}

# ============================================================
# Data Loading
# ============================================================
print("Loading model parameters...")

all_bias = []
all_tc = []
all_ss = []
all_es = []
all_sc = []
losses = []

for i in range(N_MODELS):
    path = f'{DATA_DIR}/{i:03d}/best_chkpt'
    model_data = torch.load(path, map_location='cpu', weights_only=False)
    net = model_data['network']
    all_bias.append(net['nodes_bias'].numpy())
    all_tc.append(net['nodes_time_const'].numpy())
    all_ss.append(net['edges_syn_strength'].numpy())
    all_es.append(net['edges_sign'].numpy())
    all_sc.append(net['edges_syn_count'].numpy())
    
    with h5py.File(f'{DATA_DIR}/{i:03d}/validation_loss.h5', 'r') as f:
        losses.append(f['data'][()])

bias_matrix = np.stack(all_bias)      # [50, 65]
tc_matrix = np.stack(all_tc)          # [50, 65]
ss_matrix = np.stack(all_ss)          # [50, 604]
es_matrix = np.stack(all_es)          # [50, 604]
sc_matrix = np.stack(all_sc)          # [50, 2355]
losses = np.array(losses)

best_idx = np.argmin(losses)
worst_idx = np.argmax(losses)

print(f"Loaded {N_MODELS} models")
print(f"Best model: {best_idx} (loss={losses[best_idx]:.4f})")
print(f"Worst model: {worst_idx} (loss={losses[worst_idx]:.4f})")

# ============================================================
# Figure 1: Validation Loss Analysis
# ============================================================
print("Generating Figure 1...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.hist(losses, bins=15, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(losses.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {losses.mean():.4f}')
ax.axvline(losses.min(), color='green', linestyle='--', linewidth=2, label=f'Min: {losses.min():.4f}')
ax.set_xlabel('Validation Loss (EPE)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('A. Cross-Model Validation Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1]
sorted_losses = np.sort(losses)
ax.plot(range(1, 51), sorted_losses, 'o-', color='#e74c3c', markersize=4)
ax.set_xlabel('Model Rank', fontsize=12)
ax.set_ylabel('Validation Loss (EPE)', fontsize=12)
ax.set_title('B. Model Performance Ranking', fontsize=13, fontweight='bold')

ax = axes[2]
x = np.arange(65)
width = 0.35
ax.bar(x - width/2, bias_matrix[best_idx], width,
       label=f'Best (loss={losses[best_idx]:.4f})', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, bias_matrix[worst_idx], width,
       label=f'Worst (loss={losses[worst_idx]:.4f})', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Cell Type Index', fontsize=12)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('C. Best vs Worst Model Parameters', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig1_validation_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: Resting Potentials Heatmap
# ============================================================
print("Generating Figure 2...")
fig, ax = plt.subplots(figsize=(16, 8))
sort_idx = np.argsort(bias_matrix.mean(axis=0))[::-1]
sorted_types = [cell_types[i] for i in sort_idx]
im = ax.imshow(bias_matrix[:, sort_idx].T, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=2.5)
ax.set_xlabel('Model Index', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
ax.set_title('Resting Potentials Across 50 Models (Sorted by Mean)', fontsize=14, fontweight='bold')
ax.set_yticks(range(65))
ax.set_yticklabels(sorted_types, fontsize=7)
plt.colorbar(im, ax=ax, label='Resting Potential', shrink=0.6)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_resting_potentials_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 3: Time Constants Heatmap
# ============================================================
print("Generating Figure 3...")
fig, ax = plt.subplots(figsize=(16, 8))
sort_idx_tc = np.argsort(tc_matrix.mean(axis=0))[::-1]
sorted_types_tc = [cell_types[i] for i in sort_idx_tc]
im = ax.imshow(tc_matrix[:, sort_idx_tc].T, aspect='auto', cmap='viridis', vmin=0, vmax=0.5)
ax.set_xlabel('Model Index', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
ax.set_title('Time Constants Across 50 Models (Sorted by Mean)', fontsize=14, fontweight='bold')
ax.set_yticks(range(65))
ax.set_yticklabels(sorted_types_tc, fontsize=7)
plt.colorbar(im, ax=ax, label='Time Constant (s)', shrink=0.6)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_time_constants_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Parameter Scatter and Variability
# ============================================================
print("Generating Figure 4...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bias_mean = bias_matrix.mean(axis=0)
bias_std = bias_matrix.std(axis=0)
tc_mean = tc_matrix.mean(axis=0)
tc_std = tc_matrix.std(axis=0)
families = [get_family(ct) for ct in cell_types]

ax = axes[0]
for fam, color in family_colors.items():
    mask = [f == fam for f in families]
    if any(mask):
        ax.errorbar(bias_mean[mask], tc_mean[mask], xerr=bias_std[mask], yerr=tc_std[mask],
                    fmt='o', color=color, label=fam, markersize=5, alpha=0.8, capsize=2)
ax.set_xlabel('Mean Resting Potential', fontsize=12)
ax.set_ylabel('Mean Time Constant (s)', fontsize=12)
ax.set_title('A. Resting Potential vs Time Constant', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[1]
cv_bias = bias_std / (np.abs(bias_mean) + 1e-6)
cv_tc = tc_std / (np.abs(tc_mean) + 1e-6)
for fam, color in family_colors.items():
    mask = [f == fam for f in families]
    if any(mask):
        ax.scatter(cv_bias[mask], cv_tc[mask], c=color, label=fam, s=50, alpha=0.8,
                   edgecolors='white', linewidth=0.5)
ax.set_xlabel('CV of Resting Potential', fontsize=12)
ax.set_ylabel('CV of Time Constant', fontsize=12)
ax.set_title('B. Cross-Model Parameter Variability', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_parameter_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 5: Synapse Strength Analysis
# ============================================================
print("Generating Figure 5...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ss_mean = ss_matrix.mean(axis=0)
ss_std = ss_matrix.std(axis=0)
es_mean = es_matrix.mean(axis=0)

ax = axes[0]
ax.hist(ss_mean, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(np.median(ss_mean), color='red', linestyle='--', linewidth=2,
           label=f'Median: {np.median(ss_mean):.4f}')
ax.set_xlabel('Mean Synapse Strength', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('A. Distribution of Synapse Strengths', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1]
exc_mask = es_mean > 0
inh_mask = es_mean < 0
ax.scatter(ss_mean[exc_mask], ss_std[exc_mask], c='#2ecc71',
           label=f'Excitatory (n={exc_mask.sum()})', alpha=0.6, s=20)
ax.scatter(ss_mean[inh_mask], ss_std[inh_mask], c='#e74c3c',
           label=f'Inhibitory (n={inh_mask.sum()})', alpha=0.6, s=20)
ax.set_xlabel('Mean Synapse Strength', fontsize=12)
ax.set_ylabel('Std of Synapse Strength', fontsize=12)
ax.set_title('B. Strength Variability by Sign', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig5_synapse_strength.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 6: Motion Pathway Analysis
# ============================================================
print("Generating Figure 6...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
positions = np.arange(8)
colors_t4 = ['#9b59b6', '#8e44ad', '#7d3c98', '#6c3483']
colors_t5 = ['#e74c3c', '#c0392b', '#a93226', '#922b21']

ax = axes[0]
t4_bias = bias_matrix[:, [cell_types.index(ct) for ct in ['T4a', 'T4b', 'T4c', 'T4d']]]
t5_bias = bias_matrix[:, [cell_types.index(ct) for ct in ['T5a', 'T5b', 'T5c', 'T5d']]]
data_plot = [t4_bias[:, i] for i in range(4)] + [t5_bias[:, i] for i in range(4)]
bp = ax.boxplot(data_plot, positions=positions, widths=0.6, patch_artist=True)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_t4[i] if i < 4 else colors_t5[i-4])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'], rotation=45)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('A. T4/T5 Resting Potentials', fontsize=13, fontweight='bold')
ax.axvline(3.5, color='gray', linestyle=':', linewidth=1)
ax.legend([mpatches.Patch(facecolor='#9b59b6', alpha=0.7), mpatches.Patch(facecolor='#e74c3c', alpha=0.7)],
          ['ON (T4)', 'OFF (T5)'], fontsize=10)

ax = axes[1]
t4_tc = tc_matrix[:, [cell_types.index(ct) for ct in ['T4a', 'T4b', 'T4c', 'T4d']]]
t5_tc = tc_matrix[:, [cell_types.index(ct) for ct in ['T5a', 'T5b', 'T5c', 'T5d']]]
data_tc = [t4_tc[:, i] for i in range(4)] + [t5_tc[:, i] for i in range(4)]
bp = ax.boxplot(data_tc, positions=positions, widths=0.6, patch_artist=True)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_t4[i] if i < 4 else colors_t5[i-4])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'], rotation=45)
ax.set_ylabel('Time Constant (s)', fontsize=12)
ax.set_title('B. T4/T5 Time Constants', fontsize=13, fontweight='bold')
ax.axvline(3.5, color='gray', linestyle=':', linewidth=1)

ax = axes[2]
families_order = ['Photoreceptor', 'Lamina', 'Medulla Intrinsic', 'Medulla Tangential',
                  'Medulla Columnar', 'Motion Selective', 'Centrifugal', 'Columnar', 'Amacrine']
fam_means = {}
fam_stds = {}
for fam in families_order:
    mask = [f == fam for f in families]
    if any(mask):
        fam_means[fam] = bias_mean[mask].mean()
        fam_stds[fam] = bias_mean[mask].std()
fam_names = list(fam_means.keys())
ax.barh(range(len(fam_names)), [fam_means[f] for f in fam_names],
        xerr=[fam_stds[f] for f in fam_names],
        color=[family_colors[f] for f in fam_names], alpha=0.8, capsize=3)
ax.set_yticks(range(len(fam_names)))
ax.set_yticklabels(fam_names, fontsize=10)
ax.set_xlabel('Mean Resting Potential', fontsize=12)
ax.set_title('C. Resting Potential by Cell Family', fontsize=13, fontweight='bold')
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig6_motion_pathway.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 7: Cross-Model Correlation
# ============================================================
print("Generating Figure 7...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
corr_matrix = np.corrcoef(bias_matrix)
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=0.8, vmax=1.0)
ax.set_xlabel('Model Index', fontsize=12)
ax.set_ylabel('Model Index', fontsize=12)
ax.set_title('A. Cross-Model Parameter Correlation', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson Correlation', shrink=0.8)

ax = axes[1]
corrs_bias = [np.corrcoef(bias_matrix[best_idx], bias_matrix[i])[0,1] for i in range(N_MODELS)]
corrs_tc = [np.corrcoef(tc_matrix[best_idx], tc_matrix[i])[0,1] for i in range(N_MODELS)]
ax.scatter(losses, corrs_bias, c='#3498db', label='Resting Potential', alpha=0.7, s=40)
ax.scatter(losses, corrs_tc, c='#e74c3c', label='Time Constant', alpha=0.7, s=40)
ax.set_xlabel('Validation Loss', fontsize=12)
ax.set_ylabel('Correlation with Best Model', fontsize=12)
ax.set_title('B. Parameter Similarity vs Performance', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig7_cross_model_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 8: E/I Balance
# ============================================================
print("Generating Figure 8...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
exc_strengths = ss_mean[exc_mask]
inh_strengths = ss_mean[inh_mask]
ax.hist(exc_strengths, bins=20, alpha=0.6, color='#2ecc71',
        label=f'Excitatory (mean={exc_strengths.mean():.4f})', edgecolor='white')
ax.hist(inh_strengths, bins=20, alpha=0.6, color='#e74c3c',
        label=f'Inhibitory (mean={inh_strengths.mean():.4f})', edgecolor='white')
ax.set_xlabel('Synapse Strength', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('A. Excitatory vs Inhibitory Strength', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

ax = axes[1]
param_corr_bias = [np.corrcoef(bias_matrix[:, j], losses)[0,1] for j in range(65)]
param_corr_tc = [np.corrcoef(tc_matrix[:, j], losses)[0,1] for j in range(65)]
x = np.arange(65)
ax.bar(x - 0.2, param_corr_bias, 0.4, label='Resting Potential', color='#3498db', alpha=0.8)
ax.bar(x + 0.2, param_corr_tc, 0.4, label='Time Constant', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Cell Type Index', fontsize=12)
ax.set_ylabel('Correlation with Loss', fontsize=12)
ax.set_title('B. Parameter-Performance Correlation by Cell Type', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig8_ei_balance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 9: Best Model Parameter Profiles
# ============================================================
print("Generating Figure 9...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
best_bias = bias_matrix[best_idx]
best_tc = tc_matrix[best_idx]
bar_colors = [family_colors[get_family(ct)] for ct in cell_types]

sort_idx = np.argsort(best_bias)[::-1]
ax = axes[0]
ax.barh(range(65), best_bias[sort_idx], color=[bar_colors[i] for i in sort_idx], alpha=0.85)
ax.set_yticks(range(65))
ax.set_yticklabels([cell_types[i] for i in sort_idx], fontsize=7)
ax.set_xlabel('Resting Potential', fontsize=12)
ax.set_title('A. Resting Potentials (Best Model)', fontsize=13, fontweight='bold')
ax.axvline(0, color='gray', linewidth=0.5)

sort_idx_tc = np.argsort(best_tc)[::-1]
ax = axes[1]
ax.barh(range(65), best_tc[sort_idx_tc], color=[bar_colors[i] for i in sort_idx_tc], alpha=0.85)
ax.set_yticks(range(65))
ax.set_yticklabels([cell_types[i] for i in sort_idx_tc], fontsize=7)
ax.set_xlabel('Time Constant (s)', fontsize=12)
ax.set_title('B. Time Constants (Best Model)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig9_best_model_params.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 10: Synapse Analysis
# ============================================================
print("Generating Figure 10...")
best_ss = ss_matrix[best_idx]
best_es = es_matrix[best_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
actual_counts = np.exp(sc_matrix[best_idx])
positive_mask = sc_matrix[best_idx] >= 0
ax.hist(actual_counts[positive_mask], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
ax.set_xlabel('Synapse Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('A. Synapse Count Distribution', fontsize=13, fontweight='bold')
ax.set_xscale('log')

ax = axes[1]
top_n = 30
top_indices = np.argsort(best_ss)[::-1][:top_n]
ax.barh(range(top_n), best_ss[top_indices][::-1],
        color=['#2ecc71' if best_es[i] > 0 else '#e74c3c' for i in top_indices][::-1],
        alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels([f'Edge {i}' for i in top_indices[::-1]], fontsize=8)
ax.set_xlabel('Synapse Strength', fontsize=12)
ax.set_title('B. Top 30 Edges by Strength', fontsize=13, fontweight='bold')

ax = axes[2]
exc_ss = best_ss[best_es > 0]
inh_ss = best_ss[best_es < 0]
bp = ax.boxplot([exc_ss, inh_ss], tick_labels=['Excitatory', 'Inhibitory'],
                patch_artist=True, medianprops=dict(color='black'))
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_ylabel('Synapse Strength', fontsize=12)
ax.set_title('C. Strength by Synaptic Sign', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig10_synapse_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 11: Pathway Analysis
# ============================================================
print("Generating Figure 11...")
pathway_depth = {}
for ct in cell_types:
    if ct.startswith('R'): pathway_depth[ct] = 0
    elif ct in ['L1', 'L2', 'L3', 'L4', 'L5'] or ct.startswith('Lawf') or ct == 'Am':
        pathway_depth[ct] = 1
    elif ct.startswith('C'): pathway_depth[ct] = 2
    elif ct.startswith('Mi'): pathway_depth[ct] = 2
    elif ct.startswith('Tm') or ct.startswith('TmY'): pathway_depth[ct] = 3
    elif ct.startswith('T4') or ct.startswith('T5'): pathway_depth[ct] = 4
    elif ct.startswith('CT'): pathway_depth[ct] = 3
    else: pathway_depth[ct] = 3

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
depth_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12', 4: '#9b59b6'}

ax = axes[0]
for depth in sorted(set(pathway_depth.values())):
    dcts = [ct for ct, d in pathway_depth.items() if d == depth]
    dindices = [cell_types.index(ct) for ct in dcts]
    ax.scatter([depth]*len(dindices), best_bias[dindices],
              c=depth_colors[depth], alpha=0.6, s=40,
              label=f'Depth {depth} ({len(dcts)} types)')
ax.set_xlabel('Pathway Depth', fontsize=12)
ax.set_ylabel('Resting Potential', fontsize=12)
ax.set_title('A. Pathway Depth vs Resting Potential', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
on_bias = best_bias[[cell_types.index(ct) for ct in ['T4a', 'T4b', 'T4c', 'T4d']]].mean()
off_bias = best_bias[[cell_types.index(ct) for ct in ['T5a', 'T5b', 'T5c', 'T5d']]].mean()
on_tc = best_tc[[cell_types.index(ct) for ct in ['T4a', 'T4b', 'T4c', 'T4d']]].mean()
off_tc = best_tc[[cell_types.index(ct) for ct in ['T5a', 'T5b', 'T5c', 'T5d']]].mean()
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, [on_bias, on_tc], w, label='ON (T4)', color='#9b59b6', alpha=0.8)
ax.bar(x + w/2, [off_bias, off_tc], w, label='OFF (T5)', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Resting Potential', 'Time Constant'])
ax.set_title('B. ON vs OFF Pathway Parameters', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig11_pathway_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 12: PCA Analysis
# ============================================================
print("Generating Figure 12...")
from numpy.linalg import svd
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bias_centered = bias_matrix - bias_matrix.mean(axis=0)
U, S, Vt = svd(bias_centered, full_matrices=False)
explained_var = S**2 / np.sum(S**2)
scores = bias_centered @ Vt.T

ax = axes[0]
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=losses, cmap='RdYlGn_r',
                     s=60, edgecolors='white', linewidth=0.5, alpha=0.8)
plt.colorbar(scatter, ax=ax, label='Validation Loss', shrink=0.8)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)', fontsize=12)
ax.set_title('A. PCA of Resting Potentials Across Models', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
pc1_loadings = Vt[0]
sort_idx = np.argsort(np.abs(pc1_loadings))[::-1][:20]
ax.barh(range(20), pc1_loadings[sort_idx],
        color=['#e74c3c' if v < 0 else '#3498db' for v in pc1_loadings[sort_idx]], alpha=0.8)
ax.set_yticks(range(20))
ax.set_yticklabels([cell_types[i] for i in sort_idx], fontsize=9)
ax.set_xlabel('PC1 Loading', fontsize=12)
ax.set_title('B. Top Cell Types Contributing to PC1', fontsize=13, fontweight='bold')
ax.axvline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig12_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Save Summary JSON
# ============================================================
print("Saving analysis summary...")
summary = {
    'n_models': N_MODELS,
    'n_cell_types': len(cell_types),
    'n_edges': 604,
    'n_synapse_entries': 2355,
    'val_loss_mean': float(losses.mean()),
    'val_loss_std': float(losses.std()),
    'val_loss_min': float(losses.min()),
    'val_loss_max': float(losses.max()),
    'best_model_idx': int(best_idx),
    'n_excitatory_edges': int((es_matrix[best_idx] > 0).sum()),
    'n_inhibitory_edges': int((es_matrix[best_idx] < 0).sum()),
    'cell_types': cell_types,
}
with open(f'{OUT_DIR}/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== Analysis Complete ===")
print(f"Generated {12} figures in {FIG_DIR}/")
print(f"Saved analysis summary to {OUT_DIR}/analysis_summary.json")
