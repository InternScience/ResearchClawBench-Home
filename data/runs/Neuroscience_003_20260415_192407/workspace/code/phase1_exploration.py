"""
Phase 1: Data Exploration & Reference Trajectory Computation
- Load RPE dataset
- Compute PCA, diffusion map, pseudotime on full 241 features
- Generate UMAP visualizations colored by phase, age, state, batch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load data
adata = anndata.read_h5ad('data/adata_RPE.h5ad')
print(f"Dataset shape: {adata.shape}")
print(f"Features: {adata.n_vars}, Cells: {adata.n_obs}")

# Store raw data
adata.raw = adata.copy()

# Preprocessing - data is already normalized but let's ensure proper scaling
sc.pp.pca(adata, n_comps=50, random_state=42)
print(f"PCA computed with {adata.obsm['X_pca'].shape[1]} components")

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30, random_state=42)

# Compute diffusion map
sc.tl.diffmap(adata, n_comps=10, random_state=42)
print(f"Diffusion map computed")

# Compute UMAP for visualization
sc.tl.umap(adata, random_state=42)
print(f"UMAP computed")

# Compute pseudotime using diffusion pseudotime
# Set root cell as the one with lowest annotated_age in G1 phase (early cycling)
g1_cells = adata.obs[adata.obs['phase'] == 'G1']
youngest_g1_idx = g1_cells['annotated_age'].idxmin()
root_cell_idx = adata.obs_names.get_loc(youngest_g1_idx)
print(f"Root cell: {youngest_g1_idx}, age: {adata.obs.loc[youngest_g1_idx, 'annotated_age']:.2f}")

adata.uns['iroot'] = root_cell_idx
sc.tl.dpt(adata, n_dcs=10)
print(f"Diffusion pseudotime computed")
print(f"DPT range: {adata.obs['dpt_pseudotime'].min():.4f} to {adata.obs['dpt_pseudotime'].max():.4f}")

# Save pseudotime results
adata.write('outputs/adata_with_pseudotime.h5ad')

# === Visualization ===

# Figure 1: UMAP colored by different metadata
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Phase
sc.pl.umap(adata, color='phase', ax=axes[0,0], show=False, title='Cell Cycle Phase')
# Age
sc.pl.umap(adata, color='annotated_age', ax=axes[0,1], show=False, title='Annotated Age (hours)', cmap='viridis')
# State
sc.pl.umap(adata, color='state', ax=axes[1,0], show=False, title='Cell State')
# Pseudotime
sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[1,1], show=False, title='Diffusion Pseudotime', cmap='viridis')

plt.tight_layout()
plt.savefig('report/images/fig01_umap_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig01_umap_overview.png")

# Figure 2: Pseudotime vs Age correlation
fig, ax = plt.subplots(figsize=(8, 6))
# Filter out NaN pseudotime cells
valid = adata.obs['dpt_pseudotime'] != np.nan
scatter = ax.scatter(adata.obs['annotated_age'][valid], 
                     adata.obs['dpt_pseudotime'][valid],
                     c=adata.obs['phase'][valid].cat.codes,
                     cmap='Set1', alpha=0.3, s=10)
from scipy.stats import spearmanr, pearsonr
rho, p_spearman = spearmanr(adata.obs['annotated_age'][valid], adata.obs['dpt_pseudotime'][valid])
r, p_pearson = pearsonr(adata.obs['annotated_age'][valid], adata.obs['dpt_pseudotime'][valid])
ax.set_xlabel('Annotated Age (hours)')
ax.set_ylabel('Diffusion Pseudotime')
ax.set_title(f'Pseudotime vs Age\nSpearman ρ={rho:.3f} (p={p_spearman:.2e}), Pearson r={r:.3f} (p={p_pearson:.2e})')
plt.tight_layout()
plt.savefig('report/images/fig02_pseudotime_vs_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig02_pseudotime_vs_age.png")

# Figure 3: Diffusion map visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.diffmap(adata, color='phase', ax=axes[0], show=False, components='1,2', title='Diffusion Map (DC1 vs DC2) - Phase')
sc.pl.diffmap(adata, color='dpt_pseudotime', ax=axes[1], show=False, components='1,2', title='Diffusion Map (DC1 vs DC2) - Pseudotime', cmap='viridis')
plt.tight_layout()
plt.savefig('report/images/fig03_diffmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig03_diffmap.png")

# Save reference trajectory info
ref_info = {
    'n_features_full': adata.n_vars,
    'n_cells': adata.n_obs,
    'spearman_rho_age_pseudotime': rho,
    'pearson_r_age_pseudotime': r,
    'pseudotime_range': (float(adata.obs['dpt_pseudotime'].min()), float(adata.obs['dpt_pseudotime'].max())),
}
pd.DataFrame([ref_info]).to_json('outputs/reference_trajectory_info.json')
print("Reference trajectory info saved")
print(f"Spearman correlation between pseudotime and age: {rho:.4f}")