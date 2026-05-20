"""
Feature selection methods for preserving cellular trajectories.
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
from scipy.interpolate import UnivariateSpline

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
adata = sc.read_h5ad('outputs/adata_trajectory.h5ad')
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
pseudotime = adata.obs['pseudotime'].values
feature_names = adata.var_names.tolist()
n_features = X.shape[1]

print(f"Data shape: {X.shape}")
print(f"Pseudotime range: [{pseudotime.min():.2f}, {pseudotime.max():.2f}]")

# ============================================================
# Method 1: High Variance Features
# ============================================================
variances = np.var(X, axis=0)
variance_ranks = np.argsort(variances)[::-1]

# ============================================================
# Method 2: Pseudotime Correlation
# ============================================================
correlations = np.array([np.corrcoef(X[:, i], pseudotime)[0, 1] for i in range(n_features)])
# Use absolute correlation since direction doesn't matter for trajectory preservation
corr_scores = np.abs(correlations)
corr_ranks = np.argsort(corr_scores)[::-1]

# ============================================================
# Method 3: Trajectory Dynamics (F-statistic from polynomial fit)
# ============================================================
# Fit a 3rd-order polynomial along pseudotime and compute explained variance
def fit_polynomial_score(y, x, degree=3):
    """Return R^2 of polynomial fit."""
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    y_pred = p(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return r2

dynamic_scores = np.array([fit_polynomial_score(X[:, i], pseudotime, degree=3) for i in range(n_features)])
dynamic_ranks = np.argsort(dynamic_scores)[::-1]

# ============================================================
# Method 4: Neighborhood Preservation Score
# ============================================================
# Compute k-NN graph on full data
k = 15
nbrs_full = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
_, indices_full = nbrs_full.kneighbors(X)
# Exclude self
indices_full = indices_full[:, 1:]

# For each feature, compute how well it preserves the k-NN graph
def neighborhood_preservation_score(feature_idx, k=15):
    x_feat = X[:, feature_idx:feature_idx+1]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(x_feat)
    _, indices = nbrs.kneighbors(x_feat)
    indices = indices[:, 1:]
    
    # Jaccard similarity between neighbor sets
    overlaps = 0
    for i in range(X.shape[0]):
        set_full = set(indices_full[i])
        set_feat = set(indices[i])
        overlaps += len(set_full & set_feat) / len(set_full | set_feat)
    return overlaps / X.shape[0]

print("Computing neighborhood preservation scores...")
# Sample a subset for speed, or compute all
np.random.seed(42)
# Compute for all features - it's fast enough
nbr_scores = np.array([neighborhood_preservation_score(i, k=15) for i in range(n_features)])
nbr_ranks = np.argsort(nbr_scores)[::-1]

# ============================================================
# Method 5: PCA Loadings (aggregate importance across top PCs)
# ============================================================
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)
# Weight loadings by explained variance
explained_var = pca.explained_variance_ratio_
loadings = pca.components_  # shape (n_components, n_features)
pca_scores = np.sum(np.abs(loadings) * explained_var[:, None], axis=0)
pca_ranks = np.argsort(pca_scores)[::-1]

# ============================================================
# Method 6: Mutual Information with Pseudotime (discretized)
# ============================================================
def mutual_information_discrete(x, y, n_bins=20):
    """Compute mutual information using histograms."""
    c_xy = np.histogram2d(x, y, n_bins)[0]
    # Add smoothing
    c_xy = c_xy + 1e-10
    c_xy = c_xy / np.sum(c_xy)
    c_x = np.sum(c_xy, axis=1)
    c_y = np.sum(c_xy, axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if c_xy[i, j] > 0:
                mi += c_xy[i, j] * np.log(c_xy[i, j] / (c_x[i] * c_y[j] + 1e-10))
    return mi

print("Computing mutual information scores...")
mi_scores = np.array([mutual_information_discrete(X[:, i], pseudotime, n_bins=20) for i in range(n_features)])
mi_ranks = np.argsort(mi_scores)[::-1]

# ============================================================
# Save all rankings
# ============================================================
results = pd.DataFrame({
    'feature': feature_names,
    'variance': variances,
    'abs_correlation': corr_scores,
    'dynamic_r2': dynamic_scores,
    'neighborhood_preservation': nbr_scores,
    'pca_score': pca_scores,
    'mutual_info': mi_scores,
    'variance_rank': np.argsort(variance_ranks) + 1,
    'correlation_rank': np.argsort(corr_ranks) + 1,
    'dynamic_rank': np.argsort(dynamic_ranks) + 1,
    'neighborhood_rank': np.argsort(nbr_ranks) + 1,
    'pca_rank': np.argsort(pca_ranks) + 1,
    'mi_rank': np.argsort(mi_ranks) + 1,
})

# Composite score: average rank across methods (lower is better)
results['avg_rank'] = results[['variance_rank', 'correlation_rank', 'dynamic_rank', 'neighborhood_rank', 'pca_rank', 'mi_rank']].mean(axis=1)
results['composite_rank'] = results['avg_rank'].rank().astype(int)
results = results.sort_values('composite_rank')

results.to_csv('outputs/feature_rankings.csv', index=False)
print("Saved feature rankings to outputs/feature_rankings.csv")

# Print top features by each method
print("\n=== Top 10 by Variance ===")
print(results.nsmallest(10, 'variance_rank')[['feature', 'variance']].to_string(index=False))

print("\n=== Top 10 by Pseudotime Correlation ===")
print(results.nsmallest(10, 'correlation_rank')[['feature', 'abs_correlation']].to_string(index=False))

print("\n=== Top 10 by Dynamic R² ===")
print(results.nsmallest(10, 'dynamic_rank')[['feature', 'dynamic_r2']].to_string(index=False))

print("\n=== Top 10 by Neighborhood Preservation ===")
print(results.nsmallest(10, 'neighborhood_rank')[['feature', 'neighborhood_preservation']].to_string(index=False))

print("\n=== Top 10 by PCA Score ===")
print(results.nsmallest(10, 'pca_rank')[['feature', 'pca_score']].to_string(index=False))

print("\n=== Top 10 by Mutual Information ===")
print(results.nsmallest(10, 'mi_rank')[['feature', 'mutual_info']].to_string(index=False))

print("\n=== Top 10 by Composite Score ===")
print(results.nsmallest(10, 'composite_rank')[['feature', 'avg_rank']].to_string(index=False))
