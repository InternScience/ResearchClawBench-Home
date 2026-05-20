"""
Validation of feature selection methods for trajectory preservation.
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
adata = sc.read_h5ad('outputs/adata_trajectory.h5ad')
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
pseudotime = adata.obs['pseudotime'].values
feature_names = adata.var_names.tolist()
phase = adata.obs['phase'].values
state = adata.obs['state'].values
batch = adata.obs['batch'].values

# Load rankings
rankings = pd.read_csv('outputs/feature_rankings.csv')

# ============================================================
# Validation metrics
# ============================================================

def compute_pseudotime_correlation(X_sub, pseudotime, n_pcs=10):
    """Compute pseudotime using PCA + diffusion map on subset of features."""
    if X_sub.shape[1] == 0:
        return 0, 0
    pca = PCA(n_components=min(n_pcs, X_sub.shape[1], X_sub.shape[0]-1))
    X_pca = pca.fit_transform(X_sub)
    
    # Use first PC correlation with pseudotime as a simple measure
    # (since trajectory should be captured by first PCs)
    if X_pca.shape[1] >= 1:
        corr = np.abs(np.corrcoef(X_pca[:, 0], pseudotime)[0, 1])
    else:
        corr = 0
    
    # Also compute k-NN graph and DPT
    if X_sub.shape[1] >= 2:
        nbrs = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(X_pca)
        distances, indices = nbrs.kneighbors(X_pca)
        # Simple trajectory score: average distance to nearest neighbors along pseudotime
        traj_score = 0
        for i in range(len(pseudotime)):
            neighbors = indices[i, 1:]
            traj_score += np.std(pseudotime[neighbors]) / (np.std(pseudotime) + 1e-10)
        traj_score = 1 - traj_score / len(pseudotime)  # Higher is better (more coherent)
    else:
        traj_score = 0
    
    return corr, traj_score

def compute_neighborhood_preservation(X_full, X_sub, k=15):
    """Compute how well X_sub preserves the k-NN graph of X_full."""
    nbrs_full = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_full)
    _, indices_full = nbrs_full.kneighbors(X_full)
    indices_full = indices_full[:, 1:]
    
    nbrs_sub = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_sub)
    _, indices_sub = nbrs_sub.kneighbors(X_sub)
    indices_sub = indices_sub[:, 1:]
    
    overlaps = 0
    for i in range(X_full.shape[0]):
        set_full = set(indices_full[i])
        set_sub = set(indices_sub[i])
        overlaps += len(set_full & set_sub) / k
    return overlaps / X_full.shape[0]

def compute_trajectory_smoothness(X_sub, pseudotime, n_pcs=5):
    """Compute how smooth the trajectory is in the reduced space."""
    if X_sub.shape[1] < 2:
        return 0
    pca = PCA(n_components=min(n_pcs, X_sub.shape[1], X_sub.shape[0]-1))
    X_pca = pca.fit_transform(X_sub)
    
    # Sort by pseudotime and compute smoothness as 1 - mean squared derivative
    order = np.argsort(pseudotime)
    X_sorted = X_pca[order]
    
    # Compute velocity along trajectory
    velocities = np.diff(X_sorted, axis=0)
    speed = np.linalg.norm(velocities, axis=1)
    
    # Smoothness: low variation in speed is good
    smoothness = 1.0 / (1.0 + np.std(speed) / (np.mean(speed) + 1e-10))
    
    return smoothness

def compute_phase_separability(X_sub, phase, n_pcs=10):
    """Compute how well cell cycle phases are separated."""
    if X_sub.shape[1] < 2:
        return 0
    pca = PCA(n_components=min(n_pcs, X_sub.shape[1], X_sub.shape[0]-1))
    X_pca = pca.fit_transform(X_sub)
    
    # Compute pairwise centroid distances between phases
    phases = np.unique(phase)
    centroids = []
    for p in phases:
        mask = phase == p
        centroids.append(X_pca[mask].mean(axis=0))
    centroids = np.array(centroids)
    
    # Average pairwise distance between centroids
    total_dist = 0
    count = 0
    for i in range(len(phases)):
        for j in range(i+1, len(phases)):
            total_dist += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    
    return total_dist / (count + 1e-10)

# ============================================================
# Compare feature selection methods at different subset sizes
# ============================================================
methods = {
    'Variance': 'variance_rank',
    'Pseudotime_Correlation': 'correlation_rank',
    'Dynamic_R2': 'dynamic_rank',
    'Neighborhood': 'neighborhood_rank',
    'PCA_Loadings': 'pca_rank',
    'Mutual_Info': 'mi_rank',
    'Composite': 'composite_rank',
}

subset_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150]

results = []

for method_name, rank_col in methods.items():
    print(f"\nEvaluating {method_name}...")
    top_features = rankings.sort_values(rank_col)['feature'].tolist()
    
    for n in subset_sizes:
        selected = top_features[:n]
        idx = [feature_names.index(f) for f in selected]
        X_sub = X[:, idx]
        
        pt_corr, traj_score = compute_pseudotime_correlation(X_sub, pseudotime)
        nbr_pres = compute_neighborhood_preservation(X, X_sub, k=15)
        smoothness = compute_trajectory_smoothness(X_sub, pseudotime)
        sep = compute_phase_separability(X_sub, phase)
        
        results.append({
            'method': method_name,
            'n_features': n,
            'pseudotime_correlation': pt_corr,
            'trajectory_score': traj_score,
            'neighborhood_preservation': nbr_pres,
            'smoothness': smoothness,
            'phase_separability': sep,
        })
        
        print(f"  n={n}: pt_corr={pt_corr:.4f}, nbr_pres={nbr_pres:.4f}, smoothness={smoothness:.4f}, sep={sep:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/validation_results.csv', index=False)
print("\nSaved validation results to outputs/validation_results.csv")

# Also evaluate all features as baseline
pt_corr_all, traj_score_all = compute_pseudotime_correlation(X, pseudotime)
nbr_pres_all = 1.0  # Perfect by definition
smoothness_all = compute_trajectory_smoothness(X, pseudotime)
sep_all = compute_phase_separability(X, phase)

baseline = {
    'pseudotime_correlation': pt_corr_all,
    'trajectory_score': traj_score_all,
    'neighborhood_preservation': nbr_pres_all,
    'smoothness': smoothness_all,
    'phase_separability': sep_all,
}

with open('outputs/baseline_metrics.json', 'w') as f:
    json.dump(baseline, f, indent=2)

print(f"\nBaseline (all {X.shape[1]} features):")
print(f"  pt_corr={pt_corr_all:.4f}, smoothness={smoothness_all:.4f}, sep={sep_all:.4f}")
