#!/usr/bin/env python3
"""
Detailed analysis of Drosophila motion pathway DMN.
Maps cell types to node indices, analyzes ON/OFF pathways, and generates pathway-specific figures.
"""

import os
import json
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('data/flow/0000')
OUTPUT_DIR = Path('outputs')
REPORT_IMG_DIR = Path('report/images')

# Load results from first analysis
with open(OUTPUT_DIR / 'analysis_results.json', 'r') as f:
    results = json.load(f)

all_biases = np.load(OUTPUT_DIR / 'all_biases.npy')
all_time_consts = np.load(OUTPUT_DIR / 'all_time_constants.npy')
all_syn_strengths = np.load(OUTPUT_DIR / 'all_syn_strengths.npy')
all_val_losses = np.load(OUTPUT_DIR / 'all_val_losses.npy')
edge_signs = np.load(OUTPUT_DIR / 'edge_signs.npy')
synapse_counts = np.load(OUTPUT_DIR / 'synapse_counts.npy')

cell_types = results['cell_type_names']
n_nodes = len(cell_types)
n_models = all_biases.shape[0]

# ============================================================================
# Define cell type categories for the Drosophila motion pathway
# ============================================================================
# Photoreceptors
photoreceptors = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
# Lamina neurons
lamina_neurons = ['L1', 'L2', 'L3', 'L4', 'L5', 'Lawf1', 'Lawf2', 'Am']
# Medulla intrinsic neurons
medulla_intrinsic = ['Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi9', 'Mi10', 'Mi11', 'Mi12', 'Mi13', 'Mi14', 'Mi15']
# Transmedulla neurons
transmedulla = ['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30',
                'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c']
# TmY neurons
tmy_neurons = ['TmY3', 'TmY4', 'TmY5a', 'TmY9', 'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18']
# T-neurons (medulla)
t_neurons = ['T1', 'T2', 'T2a', 'T3']
# Centrifugal neurons
centrifugal = ['C2', 'C3', 'CT1(Lo1)', 'CT1(M10)']
# ON pathway (T4 subtypes)
on_pathway = ['T4a', 'T4b', 'T4c', 'T4d']
# OFF pathway (T5 subtypes)
off_pathway = ['T5a', 'T5b', 'T5c', 'T5d']

# Create category mapping
category_map = {}
for ct in photoreceptors:
    if ct in cell_types:
        category_map[ct] = 'Photoreceptors'
for ct in lamina_neurons:
    if ct in cell_types:
        category_map[ct] = 'Lamina'
for ct in medulla_intrinsic:
    if ct in cell_types:
        category_map[ct] = 'Medulla Intrinsic'
for ct in transmedulla:
    if ct in cell_types:
        category_map[ct] = 'Transmedulla'
for ct in tmy_neurons:
    if ct in cell_types:
        category_map[ct] = 'TmY'
for ct in t_neurons:
    if ct in cell_types:
        category_map[ct] = 'T-neurons'
for ct in centrifugal:
    if ct in cell_types:
        category_map[ct] = 'Centrifugal'
for ct in on_pathway:
    if ct in cell_types:
        category_map[ct] = 'ON Pathway (T4)'
for ct in off_pathway:
    if ct in cell_types:
        category_map[ct] = 'OFF Pathway (T5)'

# Get indices for each category
cat_indices = defaultdict(list)
for ct, cat in category_map.items():
    cat_indices[cat].append(cell_types.index(ct))

print("Cell type categories:")
for cat, idxs in cat_indices.items():
    print(f"  {cat}: {len(idxs)} types - indices {idxs}")

# ============================================================================
# Figure 8: Cell type category analysis
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# A: Mean resting potential by category
categories = list(cat_indices.keys())
cat_biases = [all_biases[:, idxs].mean(axis=1) for idxs in cat_indices.values()]
bp = axes[0, 0].boxplot(cat_biases, labels=[c.replace(' ', '\n') for c in categories], patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 0].set_ylabel('Resting Potential (Bias)')
axes[0, 0].set_title('A. Resting Potentials by Cell Type Category')
axes[0, 0].tick_params(axis='x', rotation=45)

# B: Mean time constant by category
cat_tc = [all_time_consts[:, idxs].mean(axis=1) for idxs in cat_indices.values()]
bp = axes[0, 1].boxplot(cat_tc, labels=[c.replace(' ', '\n') for c in categories], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 1].set_ylabel('Time Constant (s)')
axes[0, 1].set_title('B. Time Constants by Cell Type Category')
axes[0, 1].tick_params(axis='x', rotation=45)

# C: ON vs OFF pathway comparison
on_idx = cat_indices.get('ON Pathway (T4)', [])
off_idx = cat_indices.get('OFF Pathway (T5)', [])
if on_idx and off_idx:
    on_biases = all_biases[:, on_idx].mean(axis=1)
    off_biases = all_biases[:, off_idx].mean(axis=1)
    on_tc = all_time_consts[:, on_idx].mean(axis=1)
    off_tc = all_time_consts[:, off_idx].mean(axis=1)

    x = np.arange(n_models)
    width = 0.35
    axes[1, 0].bar(x - width/2, on_biases, width, label='ON (T4)', color='gold', alpha=0.8)
    axes[1, 0].bar(x + width/2, off_biases, width, label='OFF (T5)', color='darkblue', alpha=0.8)
    axes[1, 0].set_xlabel('Model Index')
    axes[1, 0].set_ylabel('Mean Resting Potential')
    axes[1, 0].set_title('C. ON vs OFF Pathway Resting Potentials')
    axes[1, 0].legend()

    axes[1, 1].bar(x - width/2, on_tc, width, label='ON (T4)', color='gold', alpha=0.8)
    axes[1, 1].bar(x + width/2, off_tc, width, label='OFF (T5)', color='darkblue', alpha=0.8)
    axes[1, 1].set_xlabel('Model Index')
    axes[1, 1].set_ylabel('Mean Time Constant (s)')
    axes[1, 1].set_title('D. ON vs OFF Pathway Time Constants')
    axes[1, 1].legend()

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig8_cell_type_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_cell_type_categories.png")

# ============================================================================
# Figure 9: Detailed cell type heatmap
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Sort cell types by category
sorted_types = []
sorted_indices = []
for cat in ['Photoreceptors', 'Lamina', 'Medulla Intrinsic', 'Transmedulla', 'TmY',
            'T-neurons', 'Centrifugal', 'ON Pathway (T4)', 'OFF Pathway (T5)']:
    if cat in cat_indices:
        for idx in cat_indices[cat]:
            sorted_types.append(cell_types[idx])
            sorted_indices.append(idx)

# Heatmap of resting potentials
bias_matrix = all_biases[:, sorted_indices]
im = axes[0].imshow(bias_matrix.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
axes[0].set_yticks(range(len(sorted_types)))
axes[0].set_yticklabels(sorted_types, fontsize=7)
axes[0].set_xlabel('Model Index')
axes[0].set_title('A. Resting Potentials Across Models (sorted by category)')
plt.colorbar(im, ax=axes[0], label='Resting Potential')

# Heatmap of time constants
tc_matrix = all_time_consts[:, sorted_indices]
im = axes[1].imshow(tc_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1].set_yticks(range(len(sorted_types)))
axes[1].set_yticklabels(sorted_types, fontsize=7)
axes[1].set_xlabel('Model Index')
axes[1].set_title('B. Time Constants Across Models (sorted by category)')
plt.colorbar(im, ax=axes[1], label='Time Constant (s)')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig9_cell_type_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_cell_type_heatmaps.png")

# ============================================================================
# Figure 10: Motion detection pathway schematic with learned parameters
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Compute mean parameters per cell type
mean_biases = all_biases.mean(axis=0)
mean_tc = all_time_consts.mean(axis=0)

# Create a scatter plot of all cell types colored by category
cat_colors = {
    'Photoreceptors': '#e41a1c',
    'Lamina': '#377eb8',
    'Medulla Intrinsic': '#4daf4a',
    'Transmedulla': '#984ea3',
    'TmY': '#ff7f00',
    'T-neurons': '#a65628',
    'Centrifugal': '#f781bf',
    'ON Pathway (T4)': '#ffd700',
    'OFF Pathway (T5)': '#000080',
}

for cat, idxs in cat_indices.items():
    color = cat_colors.get(cat, 'gray')
    ax.scatter(mean_biases[idxs], mean_tc[idxs], c=color, s=100, label=cat,
              edgecolors='black', linewidth=0.5, alpha=0.8, zorder=3)
    for idx in idxs:
        ax.annotate(cell_types[idx], (mean_biases[idx], mean_tc[idx]),
                   fontsize=6, ha='center', va='bottom', alpha=0.7)

ax.set_xlabel('Mean Resting Potential', fontsize=12)
ax.set_ylabel('Mean Time Constant (s)', fontsize=12)
ax.set_title('Learned Parameters of 65 Cell Types in the Drosophila Motion Pathway', fontsize=14)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig10_parameter_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_parameter_space.png")

# ============================================================================
# Figure 11: Synapse count analysis by pathway
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribution of synapse counts (positive only)
pos_counts = synapse_counts[synapse_counts > 0]
axes[0].hist(pos_counts, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
axes[0].axvline(np.median(pos_counts), color='red', linestyle='--',
               label=f'Median: {np.median(pos_counts):.2f}')
axes[0].set_xlabel('Synapse Count (log scale)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('A. Distribution of Synapse Counts')
axes[0].set_yscale('log')
axes[0].legend()

# Edge sign distribution
exc_count = np.sum(edge_signs > 0)
inh_count = np.sum(edge_signs < 0)
axes[1].pie([exc_count, inh_count],
           labels=[f'Excitatory\n({exc_count})', f'Inhibitory\n({inh_count})'],
           colors=['#e74c3c', '#3498db'],
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11})
axes[1].set_title('B. Synaptic Polarity Distribution')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig11_synapse_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_synapse_analysis.png")

# ============================================================================
# Figure 12: Model ensemble analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A: Validation loss vs model index
axes[0, 0].plot(range(n_models), all_val_losses, 'o-', color='steelblue', markersize=6)
axes[0, 0].axhline(all_val_losses.mean(), color='red', linestyle='--',
                   label=f'Mean: {all_val_losses.mean():.3f}')
axes[0, 0].set_xlabel('Model Index')
axes[0, 0].set_ylabel('Validation Loss')
axes[0, 0].set_title('A. Validation Loss Across 50 Models')
axes[0, 0].legend()

# B: Parameter convergence - std of biases across models per node
bias_std = all_biases.std(axis=0)
axes[0, 1].bar(range(n_nodes), bias_std, color='coral', alpha=0.7)
axes[0, 1].set_xlabel('Node Index (Cell Type)')
axes[0, 1].set_ylabel('Std of Resting Potential')
axes[0, 1].set_title('B. Resting Potential Convergence Across Models')

# C: Time constant convergence
tc_std = all_time_consts.std(axis=0)
axes[1, 0].bar(range(n_nodes), tc_std, color='mediumpurple', alpha=0.7)
axes[1, 0].set_xlabel('Node Index (Cell Type)')
axes[1, 0].set_ylabel('Std of Time Constant')
axes[1, 0].set_title('C. Time Constant Convergence Across Models')

# D: Synapse strength convergence
ss_std = all_syn_strengths.std(axis=0)
axes[1, 1].hist(ss_std, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
axes[1, 1].set_xlabel('Std of Synapse Strength')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('D. Synapse Strength Convergence Distribution')

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'fig12_model_ensemble.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_model_ensemble.png")

# ============================================================================
# Save detailed results
# ============================================================================
detailed_results = {
    'cell_type_categories': {cat: [cell_types[i] for i in idxs] for cat, idxs in cat_indices.items()},
    'category_statistics': {},
    'on_off_comparison': {},
}

for cat, idxs in cat_indices.items():
    detailed_results['category_statistics'][cat] = {
        'n_types': len(idxs),
        'mean_bias': float(all_biases[:, idxs].mean()),
        'std_bias': float(all_biases[:, idxs].std()),
        'mean_time_const': float(all_time_consts[:, idxs].mean()),
        'std_time_const': float(all_time_consts[:, idxs].std()),
    }

on_idx = cat_indices.get('ON Pathway (T4)', [])
off_idx = cat_indices.get('OFF Pathway (T5)', [])
if on_idx and off_idx:
    detailed_results['on_off_comparison'] = {
        'on_mean_bias': float(all_biases[:, on_idx].mean()),
        'off_mean_bias': float(all_biases[:, off_idx].mean()),
        'on_mean_tc': float(all_time_consts[:, on_idx].mean()),
        'off_mean_tc': float(all_time_consts[:, off_idx].mean()),
        'bias_difference': float(all_biases[:, on_idx].mean() - all_biases[:, off_idx].mean()),
        'tc_difference': float(all_time_consts[:, on_idx].mean() - all_time_consts[:, off_idx].mean()),
    }

with open(OUTPUT_DIR / 'detailed_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)
print("Saved detailed_results.json")

print("\nDetailed analysis complete!")
