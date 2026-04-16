import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import os

adata = ad.read_h5ad("outputs/adata_full_umap.h5ad")

# 1. Feature Selection based on trajectory correlation (annotated_age)
# We want features that vary smoothly along the trajectory (annotated_age).
X = adata.X
y = adata.obs['annotated_age'].values

# Calculate Spearman correlation for each feature with annotated_age
corrs = []
pvals = []
for i in range(X.shape[1]):
    corr, pval = spearmanr(X[:, i], y)
    corrs.append(corr)
    pvals.append(pval)

# Calculate Mutual Information
mi = mutual_info_regression(X, y)

# Create a dataframe of feature statistics
feature_stats = pd.DataFrame({
    'feature': adata.var_names,
    'spearman_corr': corrs,
    'abs_spearman_corr': np.abs(corrs),
    'spearman_pval': pvals,
    'mutual_info': mi
})

# We want features that are highly dynamic along the trajectory.
# We will rank them by a combined metric or just mutual information.
feature_stats['rank_mi'] = feature_stats['mutual_info'].rank(ascending=False)
feature_stats['rank_corr'] = feature_stats['abs_spearman_corr'].rank(ascending=False)
feature_stats['combined_rank'] = feature_stats['rank_mi'] + feature_stats['rank_corr']

feature_stats = feature_stats.sort_values('combined_rank')
feature_stats.to_csv('outputs/feature_statistics.csv', index=False)

# Select top N features (e.g., top 30)
top_n = 30
selected_features = feature_stats.head(top_n)['feature'].tolist()

with open('outputs/selected_features.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

print(f"Selected top {top_n} features based on trajectory correlation.")

# 2. Evaluate the selected subset
adata_subset = adata[:, selected_features].copy()

# Recompute PCA, Neighbors, UMAP on the subset
sc.pp.pca(adata_subset)
sc.pp.neighbors(adata_subset)
sc.tl.umap(adata_subset)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_subset, color='phase', ax=axes[0], show=False)
sc.pl.umap(adata_subset, color='annotated_age', ax=axes[1], show=False)
sc.pl.umap(adata_subset, color='batch', ax=axes[2], show=False)
plt.tight_layout()
plt.savefig('report/images/umap_selected_features.png', dpi=300)
plt.close()

adata_subset.write_h5ad("outputs/adata_subset_umap.h5ad")

# 3. Plot trajectory expression for top 6 features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(selected_features[:6]):
    ax = axes[i]
    # Scatter plot of annotated_age vs feature expression
    ax.scatter(adata.obs['annotated_age'], adata[:, feat].X.flatten(), alpha=0.3, s=5, c=adata.obs['annotated_age'], cmap='viridis')
    ax.set_title(feat)
    ax.set_xlabel('Annotated Age')
    ax.set_ylabel('Expression')
plt.tight_layout()
plt.savefig('report/images/top_features_trajectory.png', dpi=300)
plt.close()

print("Feature selection and evaluation completed.")
