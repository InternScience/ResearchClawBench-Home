#!/usr/bin/env python3
"""Data exploration and preprocessing for RPE single-cell protein imaging data."""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor='white')
SAVE_DIR = Path('report/images')
OUTPUT_DIR = Path('outputs')

# Load data
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"Data shape: {adata.shape}")
print(f"Obs columns: {adata.obs.columns.tolist()}")
print(f"Feature names (first 10): {adata.var_names[:10].tolist()}")

# Basic stats
print(f"\nCells: {adata.n_obs}, Features: {adata.n_vars}")
print(f"Phase distribution:\n{adata.obs['phase'].value_counts()}")
print(f"State distribution:\n{adata.obs['state'].value_counts(dropna=False)}")
print(f"Batch distribution:\n{adata.obs['batch'].value_counts()}")

# Save processed data as log1p normalized version of raw
# The .X seems already processed; .raw layer has original values
adata_raw = adata.copy()
adata_raw.X = adata_raw.layers['raw'].copy()

# Standardize the data for downstream analysis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(adata.X)
adata.obsm['X_scaled'] = X_scaled

# Compute PCA
sc.tl.pca(adata, n_comps=50, use_highly_variable=False)
print(f"PCA variance ratio (first 10): {adata.uns['pca']['variance_ratio'][:10]}")

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

# Save
adata.write('outputs/adata_processed.h5ad', compression='gzip')
print("Processed data saved to outputs/adata_processed.h5ad")

# Basic UMAP visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sc.pl.umap(adata, color='phase', ax=axes[0, 0], show=False, title='Cell Cycle Phase', legend_loc='right margin')
sc.pl.umap(adata, color='state', ax=axes[0, 1], show=False, title='Cell State', legend_loc='right margin')
sc.pl.umap(adata, color='annotated_age', ax=axes[1, 0], show=False, title='Annotated Age (pseudotime)', cmap='viridis')
sc.pl.umap(adata, color='batch', ax=axes[1, 1], show=False, title='Batch', legend_loc='right margin')

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_umap_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_umap_overview.png")

# Distribution of annotated_age by phase
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, phase in enumerate(['G0', 'G1', 'S', 'G2']):
    subset = adata[adata.obs['phase'] == phase].obs['annotated_age']
    axes[0].hist(subset, bins=30, alpha=0.6, label=phase)
axes[0].set_xlabel('Annotated Age')
axes[0].set_ylabel('Count')
axes[0].set_title('Annotated Age Distribution by Phase')
axes[0].legend()

for state in ['cycling', 'arrested']:
    subset = adata[adata.obs['state'] == state].obs['annotated_age']
    if len(subset) > 0:
        axes[1].hist(subset, bins=30, alpha=0.6, label=state)
axes[1].set_xlabel('Annotated Age')
axes[1].set_ylabel('Count')
axes[1].set_title('Annotated Age Distribution by State')
axes[1].legend()

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_age_distribution.png")

# Feature variance ranking
variances = adata.X.var(axis=0)
var_df = pd.DataFrame({'feature': adata.var_names, 'variance': variances})
var_df = var_df.sort_values('variance', ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(30), var_df['variance'].values[:30])
ax.set_xticks(range(30))
ax.set_xticklabels(var_df['feature'].values[:30], rotation=90, fontsize=7)
ax.set_ylabel('Variance')
ax.set_title('Top 30 Features by Variance')
plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_feature_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_feature_variance.png")

# Save feature variance data
var_df.to_csv(OUTPUT_DIR / 'feature_variances.csv', index=False)
print("Feature variances saved to outputs/feature_variances.csv")
