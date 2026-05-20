"""
Trajectory inference and feature selection for preserving cellular trajectories.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import umap
import os
import json

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
adata = sc.read_h5ad('data/adata_RPE.h5ad')
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()

# Drop nan state cells for cleaner analysis
mask = adata.obs['state'].notna()
adata_clean = adata[mask].copy()
X_clean = X[mask]
print(f"Clean data shape: {adata_clean.shape}")

# Use annotated_age as pseudotime (it's a continuous variable representing cell cycle progression)
pseudotime = adata_clean.obs['annotated_age'].values
adata_clean.obs['pseudotime'] = pseudotime

# Also compute diffusion pseudotime as independent measure
sc.pp.neighbors(adata_clean, n_neighbors=15, n_pcs=50, use_rep='X')
sc.tl.diffmap(adata_clean, n_comps=10)

# Use the root cell as the one with minimum pseudotime
root_idx = np.argmin(pseudotime)
adata_clean.uns['iroot'] = root_idx
sc.tl.dpt(adata_clean, n_dcs=5)
adata_clean.obs['dpt_pseudotime'] = adata_clean.obs['dpt_pseudotime'].values

# Compute PCA and UMAP for visualization
sc.tl.pca(adata_clean, svd_solver='arbo')
sc.tl.umap(adata_clean, min_dist=0.3)

# Save
adata_clean.write('outputs/adata_trajectory.h5ad')
print("Saved trajectory data to outputs/adata_trajectory.h5ad")

# Print correlation between annotated_age and DPT
corr = np.corrcoef(adata_clean.obs['pseudotime'], adata_clean.obs['dpt_pseudotime'])[0, 1]
print(f"Correlation between annotated_age and DPT: {corr:.4f}")
