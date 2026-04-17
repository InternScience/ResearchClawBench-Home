#!/usr/bin/env python3
"""
Phase 1: Data Exploration and Preprocessing
Single-cell protein imaging data from RPE cells (4i technology)
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# Load data
print("Loading data...")
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"Shape: {adata.shape}")
print(f"Obs columns: {list(adata.obs.columns)}")

# Store raw for later comparison
adata_raw = adata.copy()

# ============================================================
# Figure 1: Data Overview
# ============================================================
print("\n=== Phase 1: Data Overview ===")

# PCA on full data
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata, random_state=42)

# Compute diffusion pseudotime
sc.tl.diffmap(adata, n_comps=15)

# Find a root cell (youngest age in G1 phase)
g1_mask = adata.obs['phase'] == 'G1'
if g1_mask.sum() > 0:
    youngest_g1 = adata.obs.loc[g1_mask, 'annotated_age'].idxmin()
    root_idx = adata.obs.index.get_loc(youngest_g1)
else:
    root_idx = adata.obs['annotated_age'].idxmin()
    root_idx = adata.obs.index.get_loc(root_idx)

adata.uns['iroot'] = root_idx
sc.tl.dpt(adata)

print(f"DPT computed. Range: {adata.obs['dpt_pseudotime'].min():.3f} - {adata.obs['dpt_pseudotime'].max():.3f}")

# Figure 1: UMAP overview
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1a: UMAP colored by phase
sc.pl.umap(adata, color='phase', ax=axes[0, 0], show=False, title='Cell Cycle Phase')
# 1b: UMAP colored by state
sc.pl.umap(adata, color='state', ax=axes[0, 1], show=False, title='Cell State')
# 1c: UMAP colored by annotated age
sc.pl.umap(adata, color='annotated_age', ax=axes[0, 2], show=False, title='Annotated Age', color_map='viridis')
# 1d: UMAP colored by batch
sc.pl.umap(adata, color='batch', ax=axes[1, 0], show=False, title='Batch')
# 1e: UMAP colored by DPT pseudotime
sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[1, 1], show=False, title='Diffusion Pseudotime', color_map='magma')
# 1f: PCA variance explained
axes[1, 2].bar(range(1, 21), adata.uns['pca']['variance_ratio'][:20] * 100)
axes[1, 2].set_xlabel('PC')
axes[1, 2].set_ylabel('Variance Explained (%)')
axes[1, 2].set_title('PCA Variance Explained')

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# Figure 1b: Phase and state distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Phase distribution
adata.obs['phase'].value_counts().plot.bar(ax=axes[0], color=sns.color_palette('Set2', 4))
axes[0].set_title('Cell Cycle Phase Distribution')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

# State distribution
adata.obs['state'].value_counts().plot.bar(ax=axes[1], color=sns.color_palette('Set3', 3))
axes[1].set_title('Cell State Distribution')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)

# Age distribution
axes[2].hist(adata.obs['annotated_age'], bins=30, color='steelblue', edgecolor='white')
axes[2].set_xlabel('Annotated Age')
axes[2].set_ylabel('Count')
axes[2].set_title('Age Distribution')

plt.tight_layout()
plt.savefig('report/images/fig1b_distributions.png', bbox_inches='tight')
plt.close()
print("Figure 1b saved.")

# Save intermediate results
adata.write('outputs/adata_processed.h5ad')
print("Processed data saved.")

# Print correlation between DPT and annotated age
corr = np.corrcoef(adata.obs['dpt_pseudotime'].values, adata.obs['annotated_age'].values)[0, 1]
print(f"\nCorrelation between DPT pseudotime and annotated age: {corr:.4f}")

# Save summary statistics
summary = {
    'n_cells': adata.shape[0],
    'n_features': adata.shape[1],
    'phases': adata.obs['phase'].value_counts().to_dict(),
    'states': adata.obs['state'].value_counts().to_dict(),
    'age_range': [float(adata.obs['annotated_age'].min()), float(adata.obs['annotated_age'].max())],
    'dpt_age_corr': float(corr),
    'batches': adata.obs['batch'].value_counts().to_dict()
}
import json
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved.")
