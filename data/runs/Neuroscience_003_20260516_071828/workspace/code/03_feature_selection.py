#!/usr/bin/env python3
"""Feature selection for trajectory preservation in RPE single-cell data.

Strategy:
1. Rank features by dynamic score (from trajectory analysis)
2. Select top-k feature subsets at various k
3. Use selected features to reconstruct pseudotime via PCA + regression
4. Evaluate trajectory preservation quality (correlation with ground truth, variance explained)
5. Identify the minimal feature set that achieves >90% trajectory preservation
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = Path('report/images')
OUTPUT_DIR = Path('outputs')

# Load data
adata = sc.read_h5ad('outputs/adata_processed.h5ad')
results_df = pd.read_csv(OUTPUT_DIR / 'feature_dynamism_scores.csv')

pseudotime_gt = adata.obs['annotated_age'].values
X_full = adata.X
feature_names = adata.var_names.tolist()

# === 1. Feature selection strategies ===
strategies = {
    'dynamic_score': results_df.sort_values('dynamic_score', ascending=False)['feature'].tolist(),
    'abs_correlation': results_df.sort_values('abs_spearman_r', ascending=False)['feature'].tolist(),
    'variance': results_df.sort_values('variance', ascending=False)['feature'].tolist(),
    'mutual_info': results_df.sort_values('mutual_info', ascending=False)['feature'].tolist(),
}

# Also add a random baseline
np.random.seed(42)
random_order = np.random.permutation(feature_names).tolist()
strategies['random'] = random_order

# Evaluate each strategy at different subset sizes
k_values = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 241]

evaluation_results = []

for strat_name, ordered_features in strategies.items():
    for k in k_values:
        if k > len(ordered_features):
            continue
        
        selected_features = ordered_features[:k]
        feat_indices = [feature_names.index(f) for f in selected_features]
        X_subset = X_full[:, feat_indices]
        
        # PCA on selected features
        pca = PCA(n_components=min(k, 10))
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X_subset))
        
        # Use first PC to predict pseudotime
        pc1 = X_pca[:, 0]
        rho_pc1, _ = spearmanr(pseudotime_gt, pc1)
        
        # Linear regression with all PCs
        model = LinearRegression()
        model.fit(X_pca, pseudotime_gt)
        y_pred = model.predict(X_pca)
        rho_reg, _ = spearmanr(pseudotime_gt, y_pred)
        
        # Cross-validated R²
        cv_scores = cross_val_score(LinearRegression(), X_pca, pseudotime_gt, cv=5, scoring='r2')
        
        # Variance explained by PCs
        var_explained = pca.explained_variance_ratio_.sum()
        
        evaluation_results.append({
            'strategy': strat_name,
            'k': k,
            'spearman_pc1': abs(rho_pc1),
            'spearman_regression': abs(rho_reg),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'pca_variance_explained': var_explained,
            'features': ','.join(selected_features)
        })

eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv(OUTPUT_DIR / 'feature_selection_evaluation.csv', index=False)
print("Feature selection evaluation saved.")

# === 2. Plot: Trajectory preservation vs number of features ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

strategy_colors = {
    'dynamic_score': '#2ca02c',
    'abs_correlation': '#1f77b4',
    'variance': '#ff7f0e',
    'mutual_info': '#d62728',
    'random': '#7f7f7f'
}

for strat_name in ['dynamic_score', 'abs_correlation', 'variance', 'mutual_info', 'random']:
    sub = eval_df[eval_df['strategy'] == strat_name]
    axes[0].plot(sub['k'], sub['spearman_regression'], 'o-', 
                 color=strategy_colors[strat_name], label=strat_name.replace('_', ' '), markersize=4)
axes[0].set_xlabel('Number of Selected Features (k)')
axes[0].set_ylabel('Spearman ρ (Regression)')
axes[0].set_title('Trajectory Preservation vs Feature Count')
axes[0].legend(fontsize=8)
axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xscale('log')

for strat_name in ['dynamic_score', 'abs_correlation', 'variance', 'mutual_info', 'random']:
    sub = eval_df[eval_df['strategy'] == strat_name]
    axes[1].plot(sub['k'], sub['cv_r2_mean'], 'o-', 
                 color=strategy_colors[strat_name], label=strat_name.replace('_', ' '), markersize=4)
axes[1].set_xlabel('Number of Selected Features (k)')
axes[1].set_ylabel('Cross-validated R²')
axes[1].set_title('Prediction Accuracy vs Feature Count')
axes[1].legend(fontsize=8)
axes[1].set_xscale('log')

for strat_name in ['dynamic_score', 'abs_correlation', 'variance', 'mutual_info', 'random']:
    sub = eval_df[eval_df['strategy'] == strat_name]
    axes[2].plot(sub['k'], sub['spearman_pc1'], 'o-', 
                 color=strategy_colors[strat_name], label=strat_name.replace('_', ' '), markersize=4)
axes[2].set_xlabel('Number of Selected Features (k)')
axes[2].set_ylabel('|Spearman ρ| (PC1)')
axes[2].set_title('PC1 Correlation vs Feature Count')
axes[2].legend(fontsize=8)
axes[2].set_xscale('log')

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_feature_selection_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_feature_selection_curves.png")

# === 3. Find minimal feature set achieving >90% trajectory preservation ===
best_strategy = eval_df[eval_df['strategy'] == 'dynamic_score']
threshold = 0.90
above_threshold = best_strategy[best_strategy['spearman_regression'] >= threshold]
if len(above_threshold) > 0:
    min_k_row = above_threshold.iloc[0]
    min_k = int(min_k_row['k'])
    print(f"\nMinimal feature set achieving >{threshold:.0%} trajectory preservation:")
    print(f"  k = {min_k} features")
    print(f"  Spearman ρ = {min_k_row['spearman_regression']:.4f}")
    print(f"  CV R² = {min_k_row['cv_r2_mean']:.4f} ± {min_k_row['cv_r2_std']:.4f}")
    
    selected_feats = min_k_row['features'].split(',')
    print(f"  Selected features: {selected_feats}")
else:
    min_k = 20
    print(f"No subset reached {threshold:.0%}. Using top 20 features.")
    selected_feats = results_df.head(20)['feature'].tolist()

# === 4. Compare trajectory reconstruction with full vs reduced features ===
# Full feature set reconstruction
X_full_scaled = StandardScaler().fit_transform(X_full)
pca_full = PCA(n_components=10)
X_pca_full = pca_full.fit_transform(X_full_scaled)
model_full = LinearRegression().fit(X_pca_full, pseudotime_gt)
pt_pred_full = model_full.predict(X_pca_full)
rho_full, _ = spearmanr(pseudotime_gt, pt_pred_full)

# Reduced feature set reconstruction
selected_indices = [feature_names.index(f) for f in selected_feats]
X_sel = X_full[:, selected_indices]
X_sel_scaled = StandardScaler().fit_transform(X_sel)
n_pca_sel = min(len(selected_feats), 10)
pca_sel = PCA(n_components=n_pca_sel)
X_pca_sel = pca_sel.fit_transform(X_sel_scaled)
model_sel = LinearRegression().fit(X_pca_sel, pseudotime_gt)
pt_pred_sel = model_sel.predict(X_pca_sel)
rho_sel, _ = spearmanr(pseudotime_gt, pt_pred_sel)

# UMAP using selected features
import anndata
adata_sel = anndata.AnnData(
    X=X_sel,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=selected_feats)
)
sc.pp.neighbors(adata_sel, n_neighbors=15, use_rep='X')
sc.tl.umap(adata_sel, min_dist=0.3, spread=1.0)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Full feature set
sc.pl.umap(adata, color='annotated_age', ax=axes[0, 0], show=False, 
           title=f'UMAP (All {X_full.shape[1]} features)', cmap='viridis')
sc.pl.umap(adata, color='phase', ax=axes[0, 1], show=False, 
           title='UMAP (All features) - Phase', legend_loc='right margin')
sc.pl.umap(adata, color='state', ax=axes[0, 2], show=False,
           title='UMAP (All features) - State', legend_loc='right margin')

# Selected features
sc.pl.umap(adata_sel, color='annotated_age', ax=axes[1, 0], show=False,
           title=f'UMAP ({len(selected_feats)} selected features)', cmap='viridis')
sc.pl.umap(adata_sel, color='phase', ax=axes[1, 1], show=False,
           title='UMAP (Selected features) - Phase', legend_loc='right margin')
sc.pl.umap(adata_sel, color='state', ax=axes[1, 2], show=False,
           title='UMAP (Selected features) - State', legend_loc='right margin')

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_umap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_umap_comparison.png")

# === 5. Scatter plot: reconstructed vs ground truth pseudotime ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(pseudotime_gt, pt_pred_full, alpha=0.3, s=3, c='steelblue')
axes[0].plot([0, 25], [0, 25], 'r--', linewidth=1)
axes[0].set_xlabel('Ground Truth Pseudotime (Annotated Age)')
axes[0].set_ylabel('Reconstructed Pseudotime')
axes[0].set_title(f'Full Feature Set (241 features)\nSpearman ρ = {rho_full:.4f}')

axes[1].scatter(pseudotime_gt, pt_pred_sel, alpha=0.3, s=3, c='darkgreen')
axes[1].plot([0, 25], [0, 25], 'r--', linewidth=1)
axes[1].set_xlabel('Ground Truth Pseudotime (Annotated Age)')
axes[1].set_ylabel('Reconstructed Pseudotime')
axes[1].set_title(f'Selected Feature Set ({len(selected_feats)} features)\nSpearman ρ = {rho_sel:.4f}')

plt.tight_layout()
fig.savefig(SAVE_DIR / 'figure_trajectory_reconstruction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_trajectory_reconstruction.png")

# === 6. Summary statistics ===
print(f"\n=== Trajectory Preservation Summary ===")
print(f"Full features (241): Spearman ρ = {rho_full:.4f}")
print(f"Selected features ({len(selected_feats)}): Spearman ρ = {rho_sel:.4f}")
print(f"Preservation ratio: {rho_sel/rho_full:.2%}")
print(f"Feature reduction: {(1 - len(selected_feats)/241):.1%}")

# Save selected features
with open(OUTPUT_DIR / 'selected_features.txt', 'w') as f:
    f.write('\n'.join(selected_feats))
print(f"\nSelected features saved to outputs/selected_features.txt")

# Save summary JSON
import json
summary = {
    'n_cells': int(adata.n_obs),
    'n_features_total': int(adata.n_vars),
    'n_features_selected': len(selected_feats),
    'trajectory_preservation_rho': float(rho_sel),
    'full_model_rho': float(rho_full),
    'preservation_ratio': float(rho_sel / rho_full),
    'selected_features': selected_feats
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved to outputs/summary.json")
