"""
Phase 1: Data Exploration and Preprocessing
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create output directories
Path('outputs').mkdir(exist_ok=True)
Path('report/images').mkdir(parents=True, exist_ok=True)
Path('code').mkdir(exist_ok=True)

# Load data
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"Loaded data: {adata.shape[0]} cells, {adata.shape[1]} features")

# Basic stats
print("\n=== Cell metadata ===")
print(adata.obs.describe())
print("\n=== Feature matrix stats ===")
print(f"Min: {adata.X.min():.4f}, Max: {adata.X.max():.4f}")
print(f"Mean: {adata.X.mean():.4f}, Std: {adata.X.std():.4f}")

# Standard preprocessing
adata.X = (adata.X - adata.X.mean(axis=0)) / (adata.X.std(axis=0) + 1e-8)

# PCA
sc.pp.pca(adata, n_comps=50, random_state=42)
print(f"\nPCA explained variance ratio (top 10): {adata.uns['pca']['variance_ratio'][:10].round(3)}")
print(f"Total variance explained (50 PCs): {adata.uns['pca']['variance_ratio'].sum():.3f}")

# UMAP
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=42)
sc.tl.umap(adata, random_state=42)

# Figure 1: Data Overview
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1a: PCA colored by age
sc.pl.pca(adata, color='annotated_age', ax=axes[0,0], show=False, 
          title='PCA - Annotated Age', frameon=True)
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')

# 1b: UMAP colored by age
sc.pl.umap(adata, color='annotated_age', ax=axes[0,1], show=False, 
           title='UMAP - Annotated Age', frameon=True)

# 1c: UMAP colored by phase
sc.pl.umap(adata, color='phase', ax=axes[0,2], show=False, 
           title='UMAP - Cell Cycle Phase', frameon=True)

# 1d: UMAP colored by state
sc.pl.umap(adata, color='state', ax=axes[1,0], show=False, 
           title='UMAP - Cell State', frameon=True)

# 1e: UMAP colored by batch
sc.pl.umap(adata, color='batch', ax=axes[1,1], show=False, 
           title='UMAP - Batch', frameon=True)

# 1f: Age distribution
axes[1,2].hist(adata.obs['annotated_age'], bins=50, edgecolor='black', alpha=0.7)
axes[1,2].set_xlabel('Annotated Age')
axes[1,2].set_ylabel('Cell Count')
axes[1,2].set_title('Age Distribution')

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure1_data_overview.png")

# Save preprocessing results
adata.write('outputs/adata_preprocessed.h5ad')
print("Saved preprocessed adata")
