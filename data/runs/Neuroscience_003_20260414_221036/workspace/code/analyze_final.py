import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.figdir = 'report/images/'
print("Start final analysis")

# Load
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print("Loaded", adata.shape)
print("States:", adata.obs['state'].value_counts().to_dict())

# QC
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

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

# PAGA
sc.tl.paga(adata, groups='leiden')

# DPT: set iroot
arrested_idx = np.flatnonzero(adata.obs['state'] == 'arrested')
adata.uns['iroot'] = arrested_idx[0]
sc.tl.diffmap(adata)
sc.tl.dpt(adata)
print("DPT done")

# Plots
sc.pl.umap(adata, color=['state', 'phase'], ncols=2, save='adata_overview.png', show=False, frameon=False)
sc.pl.umap(adata, color='dpt_pseudotime', save='pseudotime_full.png', show=False, frameon=False)
sc.pl.paga(adata, save='paga.png', show=False)

# Feature selection
hvgs = adata.var_names[adata.var.highly_variable]
adata_hvg = adata[:, hvgs]
pseudotime = adata.obs['dpt_pseudotime'].values
corrs = {}
for i, v in enumerate(hvgs):
    corr = np.corrcoef(adata_hvg.X[:, i], pseudotime)[0,1]
    corrs[v] = corr
corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['pseudotime_corr']).sort_values('pseudotime_corr', key=abs, ascending=False)
top_features = corr_df.index[:50].tolist()
print("Top 10:", top_features[:10])
corr_df.to_csv('outputs/feature_corrs.csv')

# Subset
adata_subset = adata[:, top_features].copy()
sc.pp.scale(adata_subset, max_value=10)
sc.tl.pca(adata_subset)
sc.pp.neighbors(adata_subset)
sc.tl.umap(adata_subset)
adata_subset.uns['iroot'] = adata.uns['iroot']
sc.tl.diffmap(adata_subset)
sc.tl.dpt(adata_subset)
sc.pl.umap(adata_subset, color='dpt_pseudotime', save='pseudotime_subset.png', show=False, frameon=False)

# Corr
pt_full = adata.obs['dpt_pseudotime']
pt_sub = adata_subset.obs['dpt_pseudotime']
corr_pt = np.corrcoef(pt_full, pt_sub)[0,1]
print("PT corr:", corr_pt)

# Save
adata.write('outputs/adata_processed.h5ad')
adata_subset.write('outputs/adata_subset.h5ad')
summary = {
    "n_hvg": int(sum(adata.var.highly_variable)),
    "selected_n": len(top_features),
    "pt_preservation_corr": float(corr_pt),
    "top_features": top_features[:20],
    "iroot": int(adata.uns['iroot'])
}
with open('outputs/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Complete")