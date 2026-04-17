#!/usr/bin/env python3
"""
Dynamic Feature Selection for Single-Cell Trajectory Analysis
Modular script - runs in stages to avoid timeout
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# Set paths
DATA_PATH = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_003_20260416_213123/data/adata_RPE.h5ad'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_003_20260416_213123/outputs'
IMAGE_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_003_20260416_213123/report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 60)
print("Dynamic Feature Selection Analysis")
print("=" * 60)

# Step 1: Load Data
print("\n[1/6] Loading data...")
adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} features")

# Step 2: Data Overview
print("\n[2/6] Data overview...")
data_overview = {
    'n_cells': int(adata.n_obs),
    'n_features': int(adata.n_vars),
    'feature_names': adata.var_names.tolist(),
    'obs_columns': adata.obs.columns.tolist(),
    'phase_distribution': adata.obs['phase'].value_counts().to_dict(),
    'state_distribution': adata.obs['state'].value_counts().to_dict(),
    'batch_distribution': adata.obs['batch'].value_counts().to_dict(),
    'age_stats': {
        'mean': float(adata.obs['annotated_age'].mean()),
        'std': float(adata.obs['annotated_age'].std()),
        'min': float(adata.obs['annotated_age'].min()),
        'max': float(adata.obs['annotated_age'].max())
    },
    'data_stats': {
        'min': float(adata.X.min()),
        'max': float(adata.X.max()),
        'mean': float(adata.X.mean()),
        'std': float(adata.X.std())
    }
}
with open(f'{OUTPUT_DIR}/data_overview.json', 'w') as f:
    json.dump(data_overview, f, indent=2)
print("Saved data overview")

# Step 3: Feature Statistics and Selection
print("\n[3/6] Feature selection...")
feature_stats = pd.DataFrame(index=adata.var_names)
feature_stats['mean'] = np.array(adata.X.mean(axis=0)).flatten()
feature_stats['std'] = np.array(adata.X.std(axis=0)).flatten()
feature_stats['variance'] = feature_stats['std'] ** 2
feature_stats['cv'] = feature_stats['std'] / (feature_stats['mean'].abs() + 1e-10)
feature_stats['norm_var'] = feature_stats['variance'] / (feature_stats['mean'] ** 2 + 1e-10)

top_var = feature_stats.nlargest(50, 'variance').index.tolist()
top_cv = feature_stats.nlargest(50, 'cv').index.tolist()
top_disp = feature_stats.nlargest(50, 'norm_var').index.tolist()

from collections import Counter
all_selected = top_var + top_cv + top_disp
feature_counts = Counter(all_selected)
consensus = [f for f, c in feature_counts.items() if c >= 2]

if len(consensus) < 20:
    consensus = top_var[:30]

feature_stats.to_csv(f'{OUTPUT_DIR}/feature_statistics.csv')
selected_features_info = {
    'top_variance': top_var,
    'top_cv': top_cv,
    'top_dispersion': top_disp,
    'consensus': consensus,
    'n_consensus': len(consensus)
}
with open(f'{OUTPUT_DIR}/selected_features.json', 'w') as f:
    json.dump(selected_features_info, f, indent=2)
print(f"Selected {len(consensus)} consensus features")

# Step 4: Trajectory Analysis
print("\n[4/6] Trajectory analysis...")

# Full features trajectory
adata_full = adata.copy()
sc.pp.scale(adata_full, max_value=10)
sc.pp.pca(adata_full, n_comps=min(30, adata.n_vars - 1))
sc.pp.neighbors(adata_full, n_neighbors=15, n_pcs=min(20, adata.n_vars - 1))
sc.tl.diffmap(adata_full, n_comps=10)

# Find root cell (G0 phase)
g0_cells = adata_full.obs[adata_full.obs['phase'] == 'G0'].index
root_idx = 0
if len(g0_cells) > 0:
    root_idx = np.where(adata_full.obs_names == g0_cells[0])[0][0]
adata_full.uns['iroot'] = root_idx
sc.tl.dpt(adata_full, n_dcs=10)

full_pseudotime = adata_full.obs['dpt_pseudotime'].values.copy()

with open(f'{OUTPUT_DIR}/full_trajectory.json', 'w') as f:
    json.dump({'dpt_pseudotime': full_pseudotime.tolist(), 'root_index': int(root_idx)}, f, indent=2)
print("Full trajectory computed")

# Subset features trajectory
adata_subset = adata[:, consensus].copy()
sc.pp.scale(adata_subset, max_value=10)
sc.pp.pca(adata_subset, n_comps=min(15, adata_subset.n_vars - 1))
sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=min(10, adata_subset.n_vars - 1))
sc.tl.diffmap(adata_subset, n_comps=8)
adata_subset.uns['iroot'] = root_idx
sc.tl.dpt(adata_subset, n_dcs=8)

subset_pseudotime = adata_subset.obs['dpt_pseudotime'].values.copy()

with open(f'{OUTPUT_DIR}/subset_trajectory.json', 'w') as f:
    json.dump({
        'dpt_pseudotime': subset_pseudotime.tolist(),
        'n_features_used': int(adata_subset.n_vars),
        'features_used': adata_subset.var_names.tolist()
    }, f, indent=2)
print(f"Subset trajectory computed ({adata_subset.n_vars} features)")

# Step 5: Validation
print("\n[5/6] Validation...")
spearman_corr, spearman_pval = stats.spearmanr(full_pseudotime, subset_pseudotime)
pearson_corr, pearson_pval = stats.pearsonr(full_pseudotime, subset_pseudotime)
kendall_tau, kendall_pval = stats.kendalltau(full_pseudotime, subset_pseudotime)

validation_metrics = {
    'spearman_correlation': float(spearman_corr),
    'spearman_pvalue': float(spearman_pval),
    'pearson_correlation': float(pearson_corr),
    'pearson_pvalue': float(pearson_pval),
    'kendall_tau': float(kendall_tau),
    'kendall_pvalue': float(kendall_pval),
    'n_features_full': int(adata.n_vars),
    'n_features_subset': int(adata_subset.n_vars),
    'feature_reduction_ratio': float(adata_subset.n_vars / adata.n_vars)
}
with open(f'{OUTPUT_DIR}/validation_metrics.json', 'w') as f:
    json.dump(validation_metrics, f, indent=2)
print(f"Spearman r={spearman_corr:.4f}, Pearson r={pearson_corr:.4f}")

# Step 6: Visualizations
print("\n[6/6] Generating figures...")

# Figure 1: Data Overview
print("  Fig 1: Data overview")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(20, adata.n_vars - 1))
sc.tl.umap(adata)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax in axes.flatten():
    ax.set_aspect('auto')

sc.pl.umap(adata, color='phase', ax=axes[0, 0], show=False, legend_loc='right margin', title='Cell Cycle Phase')
sc.pl.umap(adata, color='state', ax=axes[0, 1], show=False, legend_loc='right margin', title='Cell State')
sc.pl.umap(adata, color='annotated_age', ax=axes[1, 0], show=False, cmap='viridis', title='Annotated Age')
sc.pl.umap(adata, color='batch', ax=axes[1, 1], show=False, legend_loc='right margin', title='Batch')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig1_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Feature Statistics
print("  Fig 2: Feature statistics")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(feature_stats['mean'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Mean Expression')
axes[0, 0].set_ylabel('Number of Features')
axes[0, 0].set_title('Distribution of Feature Means')

axes[0, 1].hist(feature_stats['variance'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Variance')
axes[0, 1].set_ylabel('Number of Features')
axes[0, 1].set_title('Distribution of Feature Variances')

axes[1, 0].scatter(feature_stats['mean'] + 1e-10, feature_stats['variance'], alpha=0.5, s=10, c='darkgray')
axes[1, 0].set_xlabel('Mean Expression (log)')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].set_title('Mean-Variance Relationship')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')

top20 = feature_stats.nlargest(20, 'variance')
colors = ['green' if f in consensus else 'blue' for f in top20.index]
axes[1, 1].barh(range(20), top20['variance'].values, color=colors)
axes[1, 1].set_yticks(range(20))
short_names = [f.split('_')[2] if '_' in f else f[:15] for f in top20.index]
axes[1, 1].set_yticklabels(short_names, fontsize=7)
axes[1, 1].set_xlabel('Variance')
axes[1, 1].set_title('Top 20 Features by Variance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig2_feature_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Pseudotime Analysis
print("  Fig 3: Pseudotime analysis")
sc.pp.neighbors(adata_full, n_neighbors=15, n_pcs=min(20, adata.n_vars - 1))
sc.tl.umap(adata_full)

sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=min(10, adata_subset.n_vars - 1))
sc.tl.umap(adata_subset)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

scatter1 = axes[0, 0].scatter(adata_full.obsm['X_umap'][:, 0], adata_full.obsm['X_umap'][:, 1],
                              c=full_pseudotime, cmap='viridis', s=10, alpha=0.7)
axes[0, 0].set_xlabel('UMAP1')
axes[0, 0].set_ylabel('UMAP2')
axes[0, 0].set_title('Full Features - Pseudotime')
plt.colorbar(scatter1, ax=axes[0, 0], label='DPT Pseudotime')

scatter2 = axes[0, 1].scatter(adata_subset.obsm['X_umap'][:, 0], adata_subset.obsm['X_umap'][:, 1],
                              c=subset_pseudotime, cmap='viridis', s=10, alpha=0.7)
axes[0, 1].set_xlabel('UMAP1')
axes[0, 1].set_ylabel('UMAP2')
axes[0, 1].set_title(f'Selected Features ({adata_subset.n_vars}) - Pseudotime')
plt.colorbar(scatter2, ax=axes[0, 1], label='DPT Pseudotime')

mask = adata_full.obs['annotated_age'].notna()
axes[1, 0].scatter(adata_full.obs.loc[mask, 'annotated_age'], full_pseudotime[mask], alpha=0.5, s=10, c='steelblue')
z = np.polyfit(adata_full.obs.loc[mask, 'annotated_age'], full_pseudotime[mask], 1)
p = np.poly1d(z)
x_sorted = adata_full.obs.loc[mask, 'annotated_age'].sort_values()
axes[1, 0].plot(x_sorted, p(x_sorted), 'r-', linewidth=2)
axes[1, 0].set_xlabel('Annotated Age')
axes[1, 0].set_ylabel('DPT Pseudotime')
axes[1, 0].set_title('Full Features: Pseudotime vs Age')

mask2 = adata_subset.obs['annotated_age'].notna()
axes[1, 1].scatter(adata_subset.obs.loc[mask2, 'annotated_age'], subset_pseudotime[mask2], alpha=0.5, s=10, c='darkorange')
z2 = np.polyfit(adata_subset.obs.loc[mask2, 'annotated_age'], subset_pseudotime[mask2], 1)
p2 = np.poly1d(z2)
x_sorted2 = adata_subset.obs.loc[mask2, 'annotated_age'].sort_values()
axes[1, 1].plot(x_sorted2, p2(x_sorted2), 'r-', linewidth=2)
axes[1, 1].set_xlabel('Annotated Age')
axes[1, 1].set_ylabel('DPT Pseudotime')
axes[1, 1].set_title('Selected Features: Pseudotime vs Age')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig3_pseudotime_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Trajectory Validation
print("  Fig 4: Trajectory validation")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(full_pseudotime, subset_pseudotime, alpha=0.3, s=10, c='steelblue')
axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
axes[0, 0].set_xlabel('Full Features Pseudotime')
axes[0, 0].set_ylabel('Selected Features Pseudotime')
axes[0, 0].set_title(f'Pseudotime Correlation (Spearman r={spearman_corr:.3f})')
axes[0, 0].legend()

diff = np.abs(full_pseudotime - subset_pseudotime)
axes[0, 1].hist(diff, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].axvline(diff.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {diff.mean():.3f}')
axes[0, 1].set_xlabel('Absolute Pseudotime Difference')
axes[0, 1].set_ylabel('Number of Cells')
axes[0, 1].set_title('Distribution of Pseudotime Differences')
axes[0, 1].legend()

phase_order = ['G0', 'G1', 'S', 'G2']
full_by_phase = [full_pseudotime[adata_full.obs['phase'] == p] for p in phase_order]
subset_by_phase = [subset_pseudotime[adata_subset.obs['phase'] == p] for p in phase_order]

bp1 = axes[1, 0].boxplot(full_by_phase, positions=[0.8, 1.8, 2.8, 3.8], widths=0.6, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 0].set_xticks([0.8, 1.8, 2.8, 3.8])
axes[1, 0].set_xticklabels(phase_order)
axes[1, 0].set_xlabel('Cell Cycle Phase')
axes[1, 0].set_ylabel('DPT Pseudotime')
axes[1, 0].set_title('Full Features - Pseudotime by Phase')

bp2 = axes[1, 1].boxplot(subset_by_phase, positions=[0.8, 1.8, 2.8, 3.8], widths=0.6, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
axes[1, 1].set_xticks([0.8, 1.8, 2.8, 3.8])
axes[1, 1].set_xticklabels(phase_order)
axes[1, 1].set_xlabel('Cell Cycle Phase')
axes[1, 1].set_ylabel('DPT Pseudotime')
axes[1, 1].set_title('Selected Features - Pseudotime by Phase')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig4_trajectory_validation.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Top Features Expression
print("  Fig 5: Top features expression")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

top4 = feature_stats.loc[consensus].nlargest(4, 'variance').index.tolist() if len(consensus) >= 4 else consensus[:4]

for idx, feature in enumerate(top4):
    ax = axes[idx // 2, idx % 2]
    expr_by_phase = [adata[:, feature].X[adata.obs['phase'] == p].flatten() for p in phase_order]
    bp = ax.boxplot(expr_by_phase, positions=[0.8, 1.8, 2.8, 3.8], widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    short_name = feature.split('_')[2] if '_' in feature else feature[:15]
    ax.set_xticks([0.8, 1.8, 2.8, 3.8])
    ax.set_xticklabels(phase_order)
    ax.set_xlabel('Cell Cycle Phase')
    ax.set_ylabel('Expression')
    ax.set_title(f'{short_name}\n(Var: {feature_stats.loc[feature, "variance"]:.4f})')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig5_top_features_expression.png', dpi=300, bbox_inches='tight')
plt.close()

# Save method contract and inventory
print("\nSaving summary files...")

method_contract = {
    'task': 'Dynamic feature selection for single-cell trajectory analysis',
    'input': 'Single-cell protein imaging data (RPE cells)',
    'output': 'Selected subset of dynamically expressed features preserving trajectories',
    'methods': {
        'feature_selection': [
            'Variance-based selection (top 50)',
            'Coefficient of variation (CV) based selection (top 50)',
            'Normalized dispersion selection (top 50)',
            'Consensus: features selected by >=2 methods'
        ],
        'trajectory_inference': 'Diffusion Pseudotime (DPT)',
        'validation': ['Spearman correlation', 'Pearson correlation', 'Kendall tau']
    },
    'results': {
        'n_cells': int(adata.n_obs),
        'n_features_original': int(adata.n_vars),
        'n_features_selected': int(adata_subset.n_vars),
        'feature_reduction_ratio': float(adata_subset.n_vars / adata.n_vars),
        'trajectory_preservation': {
            'spearman_r': float(spearman_corr),
            'pearson_r': float(pearson_corr),
            'kendall_tau': float(kendall_tau)
        }
    }
}
with open(f'{OUTPUT_DIR}/method_contract.json', 'w') as f:
    json.dump(method_contract, f, indent=2)

target_artifacts = {
    'data_overview': {'status': 'completed', 'path': f'{OUTPUT_DIR}/data_overview.json'},
    'feature_statistics': {'status': 'completed', 'path': f'{OUTPUT_DIR}/feature_statistics.csv'},
    'selected_features': {'status': 'completed', 'path': f'{OUTPUT_DIR}/selected_features.json'},
    'full_trajectory': {'status': 'completed', 'path': f'{OUTPUT_DIR}/full_trajectory.json'},
    'subset_trajectory': {'status': 'completed', 'path': f'{OUTPUT_DIR}/subset_trajectory.json'},
    'validation_metrics': {'status': 'completed', 'path': f'{OUTPUT_DIR}/validation_metrics.json'},
    'figures': {
        'fig1_data_overview': {'status': 'completed', 'path': f'{IMAGE_DIR}/fig1_data_overview.png'},
        'fig2_feature_statistics': {'status': 'completed', 'path': f'{IMAGE_DIR}/fig2_feature_statistics.png'},
        'fig3_pseudotime_analysis': {'status': 'completed', 'path': f'{IMAGE_DIR}/fig3_pseudotime_analysis.png'},
        'fig4_trajectory_validation': {'status': 'completed', 'path': f'{IMAGE_DIR}/fig4_trajectory_validation.png'},
        'fig5_top_features_expression': {'status': 'completed', 'path': f'{IMAGE_DIR}/fig5_top_features_expression.png'}
    }
}
with open(f'{OUTPUT_DIR}/target_artifact_inventory.json', 'w') as f:
    json.dump(target_artifacts, f, indent=2)

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print(f"Original features: {adata.n_vars}")
print(f"Selected features: {adata_subset.n_vars}")
print(f"Feature reduction: {100 * (1 - adata_subset.n_vars / adata.n_vars):.1f}%")
print(f"Trajectory preservation (Spearman r): {spearman_corr:.4f}")
print(f"\nOutputs: {OUTPUT_DIR}")
print(f"Figures: {IMAGE_DIR}")
