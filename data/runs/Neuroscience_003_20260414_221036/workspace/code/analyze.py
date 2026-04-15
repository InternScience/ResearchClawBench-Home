import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.set_figure_params(figsize=(8,6))
print("Start analysis")

# Load
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print("Loaded", adata.shape)

# QC
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
print("QC computed")

# HVG
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, min_disp=0.5, batch_key='batch')
print("HVGs:", sum(adata.var.highly_variable))

# Scale
sc.pp.scale(adata, max_value=10)

# PCA, neighbors, UMAP, leiden
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
print("Dim red done")

# PAGA
sc.tl.paga(adata, groups='leiden')

# DPT root on arrested
root_cells = adata.obs['state'] == 'arrested'
root = np.flatnonzero(root_cells)[0]
sc.tl.dpt(adata, root=root)
print("DPT done")

# Plots - non-interactive
fig, axs = plt.subplots(2,2, figsize=(12,10))
sc.pl.umap(adata, color=['state'], ax=axs[0,0], show=False, frameon=False)
sc.pl.umap(adata, color=['phase'], ax=axs[0,1], show=False, frameon=False)
sc.pl.umap(adata, color='dpt_pseudotime', ax=axs[1,0], show=False, frameon=False)
sc.pl.paga(adata, ax=axs[1,1], show=False)
plt.tight_layout()
plt.savefig('report/images/adata_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature selection
hvgs = adata.var_names[adata.var.highly_variable]
X_hvg = adata[:, hvgs].X
pseudotime = adata.obs['dpt_pseudotime'].values
corrs = pd.Series(index=hvgs, dtype=float)
for v in hvgs:
    i = np.where(adata.var_names == v)[0][0]
    corr = np.corrcoef(X_hvg[:, np.where(hvgs==v)[0][0]], pseudotime)[0,1]
    corrs[v] = corr
corrs = corrs.abs().sort_values(ascending=False)
top_features = corrs.head(50).index.tolist()
print("Top features:", top_features[:5])

# Subset
adata_subset = adata[:, top_features].copy()
sc.tl.pca(adata_subset)
sc.pp.neighbors(adata_subset, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_subset)
sc.tl.dpt(adata_subset, root=root)

# Plot subset
fig, ax = plt.subplots(figsize=(6,5))
sc.pl.umap(adata_subset, color='dpt_pseudotime', show=False, frameon=False, ax=ax)
plt.savefig('report/images/umap_subset.png', dpi=150, bbox_inches='tight')
plt.close()

# Preservation
pt_full = adata.obs['dpt_pseudotime']
pt_sub = adata_subset.obs['dpt_pseudotime']
corr_pt = np.corrcoef(pt_full, pt_sub)[0,1]
print("PT corr:", corr_pt)

# Save
corrs.to_csv('outputs/feature_corrs.csv')
adata.write('outputs/adata_processed.h5ad')
adata_subset.write('outputs/adata_subset.h5ad')
summary = {
    "n_hvg": int(sum(adata.var.highly_variable)),
    "selected_n": len(top_features),
    "pt_preservation_corr": float(corr_pt),
    "top_features": top_features
}
with open('outputs/results_summary.json', 'w') as f:
    json.dump(summary, f)
print("Done")