#!/usr/bin/env python3
"""
Main analysis pipeline for dynamic feature selection preserving cellular trajectories.
Uses single-cell protein imaging data (RPE cells) to identify dynamically expressed
molecular features that best preserve continuous cellular trajectories.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import cdist
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Neuroscience_003_20260415_110948'
DATA_PATH = os.path.join(WORKSPACE, 'data', 'adata_RPE.h5ad')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
CODE_DIR = os.path.join(WORKSPACE, 'code')
REPORT_IMG_DIR = os.path.join(WORKSPACE, 'report', 'images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading and exploring data")
print("=" * 60)

adata = sc.read_h5ad(DATA_PATH)
print(f"Data shape: {adata.shape}")
print(f"Cells: {adata.n_obs}, Features: {adata.n_vars}")
print(f"Obs columns: {list(adata.obs.columns)}")
print(f"Phase distribution: {adata.obs['phase'].value_counts().to_dict()}")
print(f"State distribution: {adata.obs['state'].value_counts().to_dict()}")
print(f"Batch distribution: {adata.obs['batch'].value_counts().to_dict()}")

# Save basic data info
data_info = {
    "n_cells": int(adata.n_obs),
    "n_features": int(adata.n_vars),
    "feature_names": list(adata.var_names),
    "obs_columns": list(adata.obs.columns),
    "phase_counts": adata.obs['phase'].value_counts().to_dict(),
    "state_counts": adata.obs['state'].value_counts().to_dict(),
    "batch_counts": adata.obs['batch'].value_counts().to_dict(),
    "age_stats": {
        "mean": float(adata.obs['annotated_age'].mean()),
        "std": float(adata.obs['annotated_age'].std()),
        "min": float(adata.obs['annotated_age'].min()),
        "max": float(adata.obs['annotated_age'].max())
    }
}

with open(os.path.join(OUTPUT_DIR, 'data_info.json'), 'w') as f:
    json.dump(data_info, f, indent=2)

# ============================================================
# 2. FEATURE DYNAMICS SCORING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Scoring feature dynamics")
print("=" * 60)

X = adata.X.copy()
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].values
feature_names = list(adata.var_names)
n_features = X.shape[1]

# 2a. Correlation with age (Spearman)
age_corr = np.zeros(n_features)
age_pval = np.zeros(n_features)
for i in range(n_features):
    corr, p = stats.spearmanr(X[:, i], age)
    age_corr[i] = corr
    age_pval[i] = p

# 2b. Phase-dependent expression (Kruskal-Wallis H-test)
phases_unique = ['G0', 'G1', 'S', 'G2']
kw_stat = np.zeros(n_features)
kw_pval = np.zeros(n_features)
for i in range(n_features):
    groups = [X[phase == p, i] for p in phases_unique]
    stat, p = stats.kruskal(*groups)
    kw_stat[i] = stat
    kw_pval[i] = p

# 2c. Coefficient of variation (CV)
feature_mean = X.mean(axis=0)
feature_std = X.std(axis=0)
cv = feature_std / (feature_mean + 1e-10)

# 2d. Variance explained by age (linear R²)
from sklearn.linear_model import LinearRegression
age_r2 = np.zeros(n_features)
for i in range(n_features):
    lr = LinearRegression()
    lr.fit(age.reshape(-1, 1), X[:, i])
    pred = lr.predict(age.reshape(-1, 1))
    ss_res = np.sum((X[:, i] - pred) ** 2)
    ss_tot = np.sum((X[:, i] - X[:, i].mean()) ** 2)
    age_r2[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

# 2e. Composite dynamic score
# Normalize each metric to [0, 1]
def normalize_to_01(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

age_corr_norm = normalize_to_01(np.abs(age_corr))  # Absolute correlation
kw_stat_norm = normalize_to_01(kw_stat)
cv_norm = normalize_to_01(cv)
age_r2_norm = normalize_to_01(age_r2)

# Composite score: weighted combination
dynamic_score = (
    0.30 * age_corr_norm +
    0.30 * kw_stat_norm +
    0.20 * cv_norm +
    0.20 * age_r2_norm
)

# Create feature scores dataframe
feature_scores_df = pd.DataFrame({
    'feature': feature_names,
    'age_spearman_corr': age_corr,
    'age_spearman_pval': age_pval,
    'kw_statistic': kw_stat,
    'kw_pval': kw_pval,
    'coefficient_of_variation': cv,
    'age_r_squared': age_r2,
    'dynamic_score': dynamic_score
})

# Sort by dynamic score
feature_scores_df = feature_scores_df.sort_values('dynamic_score', ascending=False)
print(f"\nTop 20 most dynamic features:")
print(feature_scores_df.head(20).to_string(index=False))

feature_scores_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_scores.csv'), index=False)

# ============================================================
# 3. FEATURE SELECTION AT DIFFERENT THRESHOLDS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Feature selection at different thresholds")
print("=" * 60)

# Test different numbers of top features
k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]
selection_results = {}

for k in k_values:
    top_k_features = feature_scores_df.head(k)['feature'].values
    top_k_indices = [feature_names.index(f) for f in top_k_features]
    X_subset = X[:, top_k_indices]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # PCA
    pca = PCA(n_components=min(50, k))
    X_pca = pca.fit_transform(X_scaled)
    
    # UMAP-like embedding via t-SNE on PCA components
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca[:, :min(15, X_pca.shape[1])])
    
    # Trajectory preservation metrics
    # 1. Age correlation with first t-SNE component
    tsne_age_corr_1, _ = stats.spearmanr(X_tsne[:, 0], age)
    tsne_age_corr_2, _ = stats.spearmanr(X_tsne[:, 1], age)
    
    # 2. Silhouette score for phase separation
    sil_phase = silhouette_score(X_tsne, phase)
    
    # 3. Silhouette score for state separation (excluding nan)
    mask_state = adata.obs['state'] != 'nan'
    if mask_state.sum() > 1:
        sil_state = silhouette_score(X_tsne[mask_state.values], 
                                      adata.obs.loc[mask_state, 'state'])
    else:
        sil_state = np.nan
    
    # 4. Variance explained by top PCA components
    var_exp_pca1 = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0
    var_exp_pca2 = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
    
    selection_results[k] = {
        'k': k,
        'top_features': list(top_k_features),
        'tsne_age_corr_1': float(tsne_age_corr_1),
        'tsne_age_corr_2': float(tsne_age_corr_2),
        'silhouette_phase': float(sil_phase),
        'silhouette_state': float(sil_state),
        'var_exp_pca1': float(var_exp_pca1),
        'var_exp_pca2': float(var_exp_pca2),
        'var_exp_cumulative_5': float(sum(pca.explained_variance_ratio_[:5])) if len(pca.explained_variance_ratio_) >= 5 else float(sum(pca.explained_variance_ratio_)),
    }
    
    print(f"k={k:3d}: age_corr_1={tsne_age_corr_1:+.3f}, sil_phase={sil_phase:.3f}, "
          f"var_pca1={var_exp_pca1:.3f}, var_pca2={var_exp_pca2:.3f}")

# Save selection results
with open(os.path.join(OUTPUT_DIR, 'selection_results.json'), 'w') as f:
    json.dump(selection_results, f, indent=2)

# Find optimal k (maximize age correlation + silhouette)
best_k = max(k_values, key=lambda k: abs(selection_results[k]['tsne_age_corr_1']) + 
             selection_results[k]['silhouette_phase'])
print(f"\nOptimal k (by combined score): {best_k}")

# Also select k=30 as a balanced choice
optimal_k = 30
optimal_features = feature_scores_df.head(optimal_k)['feature'].values
optimal_indices = [feature_names.index(f) for f in optimal_features]

print(f"\nSelected {optimal_k} optimal features:")
for i, f in enumerate(optimal_features):
    print(f"  {i+1}. {f} (score={feature_scores_df.iloc[i]['dynamic_score']:.4f})")

# Save optimal features
with open(os.path.join(OUTPUT_DIR, 'optimal_features.json'), 'w') as f:
    json.dump({
        'k': optimal_k,
        'features': list(optimal_features),
        'scores': feature_scores_df.head(optimal_k)['dynamic_score'].tolist()
    }, f, indent=2)

print("\nStep 3 complete.")
