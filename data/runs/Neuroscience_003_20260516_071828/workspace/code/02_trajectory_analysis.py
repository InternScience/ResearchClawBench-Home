#!/usr/bin/env python3
"""Trajectory inference and dynamic feature identification for RPE single-cell data."""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = Path('report/images')
OUTPUT_DIR = Path('outputs')

# Load processed data
adata = sc.read_h5ad('outputs/adata_processed.h5ad')
print(f"Loaded data: {adata.shape}")

# Use annotated_age as ground truth pseudotime
pseudotime = adata.obs['annotated_age'].values

# === 1. Diffusion Pseudotime (DPT) as alternative trajectory ===
# Set root cells as those with minimum annotated_age
root_idx = np.argmin(pseudotime)
adata.uns['iroot'] = root_idx
sc.tl.diffmap(adata)
sc.tl.dpt(adata, n_dcs=10)
print(f"DPT computed. DPT range: [{adata.obs['dpt_pseudotime'].min():.3f}, {adata.obs['dpt_pseudotime'].max():.3f}]")

# === 2. Identify dynamically expressed features along pseudotime ===
# Fit GAM-like models using polynomial regression for each feature
n_features = adata.n_vars
feature_names = adata.var_names.tolist()
X_data = adata.X

# For each feature, compute:
# - Spearman correlation with pseudotime
# - R² from polynomial regression (degree 3)
# - Mutual information with pseudotime

results = []
for i in range(n_features):
    y = X_data[:, i]
    
    # Spearman correlation
    rho, pval = spearmanr(pseudotime, y)
    
    # Polynomial fit (degree 3)
    t = pseudotime.reshape(-1, 1)
    # Create polynomial features
    t_poly = np.column_stack([t, t**2, t**3])
    model = LinearRegression()
    model.fit(t_poly, y)
    y_pred = model.predict(t_poly)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Mutual information
    mi = mutual_info_regression(t, y, random_state=42)[0]
    
    results.append({
        'feature': feature_names[i],
        'spearman_r': rho,
        'spearman_pval': pval,
        'r2_poly3': max(0, r2),  # clamp to 0
        'mutual_info': mi,
        'variance': np.var(y)
    })

results_df = pd.DataFrame(results)
results_df['abs_spearman_r'] = np.abs(results_df['spearman_r'])

# FDR correction for p-values
from statsmodels.stats.multitest import multipletests
_, results_df['spearman_qval'], _, _ = multipletests(results_df['spearman_pval'], method='fdr_bh')

# Rank features by dynamic score (combined metric)
results_df['dynamic_score'] = (
    results_df['abs_spearman_r'] * 0.4 +
    results_df['r2_poly3'] * 0.3 +
    (results_df['mutual_info'] / results_df['mutual_info'].max()) * 0.3
)
results_df = results_df.sort_values('dynamic_score', ascending=False).reset_index(drop=True)

print(f"\nTop 20 dynamically expressed features:")
print(results_df[['feature', 'spearman_r', 'r2_poly3', 'mutual_info', 'dynamic_score']].head(20))

# Save results
results_df.to_csv(OUTPUT_DIR / 'feature_dynamism_scores.csv', index=False)
print("Feature dynamism scores saved.")

# === 3. Plot top dynamic features along pseudotime ===
top_n = 12
top_features = results_df.head(top_n)['feature'].values

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    feat_idx = feature_names.index(feat)
    y = X_data[:, feat_idx]
    ax = axes[i]
    
    # Scatter with low alpha
    ax.scatter(pseudotime, y, alpha=0.3, s=3, c='steelblue')
    
    # Fit and plot smooth curve
    t_sorted = np.sort(pseudotime)
    t_poly = np.column_stack([t_sorted, t_sorted**2, t_sorted**3])
    model = LinearRegression()
    model.fit(np.column_stack([pseudotime, pseudotime**2, pseudotime**3]), y)
    y_smooth = model.predict(np.column_stack([t_sorted, t_sorted**2, t_sorted**3]))
    ax.plot(t_sorted, y_smooth, 'r-', linewidth=2)
    
    r = results_df.loc[results_df['feature'] == feat, 'spearman_r'].values[0]
    ax.set_title(f'{feat[:40]}\nρ={r:.3f}', fontsize=9)
    ax.set_xlabel('Annotated Age')
    ax.set_ylabel('Expression')

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_top_dynamic_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_top_dynamic_features.png")

# === 4. Heatmap of top dynamic features by pseudotime bins ===
n_bins = 10
adata.obs['pseudotime_bin'] = pd.cut(pseudotime, bins=n_bins, labels=False)
bin_means = []
for feat in results_df.head(30)['feature']:
    feat_idx = feature_names.index(feat)
    means = [adata.X[adata.obs['pseudotime_bin'] == b, feat_idx].mean() for b in range(n_bins)]
    bin_means.append(means)

bin_matrix = np.array(bin_means)
# Z-score normalize rows
bin_matrix_z = (bin_matrix - bin_matrix.mean(axis=1, keepdims=True)) / (bin_matrix.std(axis=1, keepdims=True) + 1e-10)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(bin_matrix_z, aspect='auto', cmap='RdBu_r')
ax.set_xticks(range(n_bins))
ax.set_xticklabels([f'{i}' for i in range(n_bins)])
ax.set_yticks(range(len(results_df.head(30))))
ax.set_yticklabels([f[:35] for f in results_df.head(30)['feature']], fontsize=7)
ax.set_xlabel('Pseudotime Bin')
ax.set_ylabel('Feature')
ax.set_title('Top 30 Dynamic Features: Z-scored Expression by Pseudotime Bin')
plt.colorbar(im, ax=ax, label='Z-score')
plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_dynamic_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_dynamic_heatmap.png")

# === 5. Compare ground truth pseudotime with DPT ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Correlation
r_dpt, _ = spearmanr(pseudotime, adata.obs['dpt_pseudotime'].values)
axes[0].scatter(pseudotime, adata.obs['dpt_pseudotime'], alpha=0.3, s=3)
axes[0].set_xlabel('Annotated Age (ground truth)')
axes[0].set_ylabel('Diffusion Pseudotime (DPT)')
axes[0].set_title(f'DPT vs Ground Truth Pseudotime\nSpearman ρ = {r_dpt:.4f}')

# DPT by phase
phases = ['G0', 'G1', 'S', 'G2']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for phase, c in zip(phases, colors):
    mask = adata.obs['phase'] == phase
    axes[1].scatter(pseudotime[mask], adata.obs['dpt_pseudotime'][mask], 
                    alpha=0.3, s=3, c=c, label=phase)
axes[1].set_xlabel('Annotated Age')
axes[1].set_ylabel('Diffusion Pseudotime (DPT)')
axes[1].set_title('DPT vs Ground Truth by Cell Cycle Phase')
axes[1].legend()

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_dpt_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_dpt_comparison.png")

# Save DPT data
adata.obs[['dpt_pseudotime']].to_csv(OUTPUT_DIR / 'dpt_pseudotime.csv')
print("DPT saved to outputs/dpt_pseudotime.csv")

print("\nTrajectory analysis complete.")
