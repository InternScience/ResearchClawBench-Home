"""
Phase 4: Trajectory Preservation Validation
Compare trajectory quality with full features vs. selected subset.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Load data
adata = sc.read_h5ad('outputs/adata_preprocessed.h5ad')
X = adata.X
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].values
state = adata.obs['state'].values
feature_names = list(adata.var_names)

# Load selected features
selected_df = pd.read_csv('outputs/selected_features.csv')
selected_idx = [feature_names.index(f) for f in selected_df['feature']]
X_selected = X[:, selected_idx]

print(f"Full features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")

# === Validation Metric 1: Trajectory ordering accuracy ===
# Use Spearman correlation between feature-based pseudotime and annotated age
def compute_trajectory_metrics(X_sub, age, n_neighbors=15):
    """Compute trajectory preservation metrics."""
    metrics = {}
    
    # 1. PCA-based pseudotime correlation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, X_sub.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_sub)
    
    # Use PC1 as simple pseudotime
    pc1 = X_pca[:, 0]
    r_pc1, p_pc1 = stats.spearmanr(pc1, age)
    metrics['pc1_spearman'] = r_pc1
    metrics['pc1_pval'] = p_pc1
    
    # 2. Neighborhood graph coherence
    # For each cell, check if neighbors have similar ages
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_sub)
    distances, indices = nn.kneighbors()
    
    # Coherence: mean absolute age difference between neighbors
    age_diffs = []
    for i in range(len(age)):
        neighbor_ages = age[indices[i]]
        age_diffs.append(np.mean(np.abs(neighbor_ages - age[i])))
    metrics['neighborhood_coherence'] = np.mean(age_diffs)
    
    # 3. Global trajectory ordering (Kendall tau)
    # Sort cells by PC1 and check monotonicity with age
    sort_idx = np.argsort(pc1)
    sorted_age = age[sort_idx]
    tau, _ = stats.kendalltau(np.arange(len(sorted_age)), sorted_age)
    metrics['kendall_tau'] = tau
    
    # 4. Silhouette score (age-based)
    # Bin age for silhouette computation
    age_bins = pd.qcut(age, q=10, labels=False, duplicates='drop')
    sil = silhouette_samples(X_pca, age_bins)
    metrics['silhouette_score'] = np.mean(sil)
    
    return metrics, X_pca

# Compute metrics for full and selected features
metrics_full, pca_full = compute_trajectory_metrics(X, age)
metrics_selected, pca_selected = compute_trajectory_metrics(X_selected, age)

print("\n=== Trajectory Preservation Metrics ===")
print(f"{'Metric':<30} {'Full Features':<15} {'Selected Features':<15} {'Improvement':<15}")
print("-" * 75)
for key in metrics_full:
    full_val = metrics_full[key]
    sel_val = metrics_selected[key]
    # For coherence and silhouette, lower is better for coherence, higher for silhouette
    if key == 'neighborhood_coherence':
        improvement = (full_val - sel_val) / abs(full_val) * 100
        better = '✓' if sel_val < full_val else '✗'
    else:
        improvement = (sel_val - full_val) / max(abs(full_val), 1e-8) * 100
        better = '✓' if abs(sel_val) > abs(full_val) else '✗'
    print(f"{key:<30} {full_val:<15.4f} {sel_val:<15.4f} {improvement:<+10.1f}% {better}")

# === Validation Metric 2: Feature importance stability ===
# Compare top PC loadings
print("\n=== PCA Component Comparison ===")
print("Full features PC1 top loadings:")
pca_model_full = PCA(n_components=min(10, X.shape[1]), random_state=42)
pca_model_full.fit(X)
full_loadings = pd.Series(pca_model_full.components_[0], index=feature_names)
print(full_loadings.abs().sort_values(ascending=False).head(10))

print("\nSelected features PC1 top loadings:")
pca_model_sel = PCA(n_components=min(10, X_selected.shape[1]), random_state=42)
pca_model_sel.fit(X_selected)
selected_loadings = pd.Series(pca_model_sel.components_[0], index=selected_df['feature'].tolist())
print(selected_loadings.abs().sort_values(ascending=False).head(10))

# === Validation Metric 3: Phase preservation ===
# Check if cell cycle phases are well-separated in trajectory space
for label, name in [(phase, 'Phase'), (state, 'State')]:
    valid = ~pd.isna(label)
    if valid.sum() > 2 and len(np.unique(label[valid])) > 1:
        sil_phase = silhouette_score(pca_full[valid], label[valid])
        sil_phase_sel = silhouette_score(pca_selected[valid], label[valid])
        print(f"\n{name} silhouette (full): {sil_phase:.4f}")
        print(f"{name} silhouette (selected): {sil_phase_sel:.4f}")

# === Validation Metric 4: Age prediction accuracy ===
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Predict age from features
ridge = Ridge(alpha=1.0)
cv_full = cross_val_score(ridge, X, age, cv=5, scoring='r2')
cv_selected = cross_val_score(ridge, X_selected, age, cv=5, scoring='r2')

print(f"\n=== Age Prediction (Ridge Regression, 5-fold CV) ===")
print(f"Full features R²: {cv_full.mean():.4f} ± {cv_full.std():.4f}")
print(f"Selected features R²: {cv_selected.mean():.4f} ± {cv_selected.std():.4f}")
print(f"Feature reduction: {X.shape[1]} → {X_selected.shape[1]} ({X_selected.shape[1]/X.shape[1]*100:.1f}%)")

# === Validation Metric 5: Mutual information preservation ===
from sklearn.feature_selection import mutual_info_regression

mi_full = mutual_info_regression(X, age, random_state=42, n_neighbors=10)
mi_selected = mutual_info_regression(X_selected, age, random_state=42, n_neighbors=10)

print(f"\n=== Mutual Information ===")
print(f"Mean MI (all features): {mi_full.mean():.4f}")
print(f"Mean MI (selected features): {mi_selected.mean():.4f}")
print(f"Total MI (all features): {mi_full.sum():.4f}")
print(f"Total MI (selected features): {mi_selected.sum():.4f}")

# === Figure 4: Trajectory Preservation Comparison ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 4a: PC1 vs Age comparison
axes[0,0].scatter(age, pca_full[:, 0], alpha=0.2, s=10, c='blue', label='Full features')
axes[0,0].scatter(age, pca_selected[:, 0], alpha=0.2, s=10, c='red', label='Selected features')
axes[0,0].set_xlabel('Annotated Age')
axes[0,0].set_ylabel('PC1')
axes[0,0].set_title('PC1 vs Age')
axes[0,0].legend()

# 4b: PCA full features colored by age
scatter1 = axes[0,1].scatter(pca_full[:, 0], pca_full[:, 1], c=age, cmap='viridis', 
                             alpha=0.3, s=10)
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')
axes[0,1].set_title(f'Full Features PCA (Spearman r={metrics_full["pc1_spearman"]:.3f})')
plt.colorbar(scatter1, ax=axes[0,1], label='Age')

# 4c: PCA selected features colored by age
scatter2 = axes[0,2].scatter(pca_selected[:, 0], pca_selected[:, 1], c=age, cmap='viridis', 
                             alpha=0.3, s=10)
axes[0,2].set_xlabel('PC1')
axes[0,2].set_ylabel('PC2')
axes[0,2].set_title(f'Selected Features PCA (Spearman r={metrics_selected["pc1_spearman"]:.3f})')
plt.colorbar(scatter2, ax=axes[0,2], label='Age')

# 4d: Bar chart comparison of metrics
metric_names = ['pc1_spearman', 'kendall_tau', 'silhouette_score']
metric_labels = ['PC1 Spearman', 'Kendall Tau', 'Silhouette']
full_vals = [abs(metrics_full[m]) for m in metric_names]
sel_vals = [abs(metrics_selected[m]) for m in metric_names]
x = np.arange(len(metric_names))
width = 0.35
axes[1,0].bar(x - width/2, full_vals, width, label='Full features', alpha=0.7)
axes[1,0].bar(x + width/2, sel_vals, width, label='Selected features', alpha=0.7)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(metric_labels, rotation=15)
axes[1,0].set_ylabel('Absolute Value')
axes[1,0].set_title('Trajectory Quality Metrics Comparison')
axes[1,0].legend()

# 4e: Age prediction comparison
axes[1,1].bar(['Full Features', 'Selected Features'], 
              [cv_full.mean(), cv_selected.mean()],
              yerr=[cv_full.std(), cv_selected.std()],
              capsize=5, alpha=0.7)
axes[1,1].set_ylabel('R² Score')
axes[1,1].set_title('Age Prediction Accuracy (5-fold CV)')

# 4f: Feature reduction summary
summary_text = (
    f"Feature Reduction: {X.shape[1]} → {X_selected.shape[1]}\n"
    f"Reduction ratio: {X_selected.shape[1]/X.shape[1]*100:.1f}%\n\n"
    f"Trajectory Preservation:\n"
    f"  PC1 Spearman: {metrics_full['pc1_spearman']:.3f} → {metrics_selected['pc1_spearman']:.3f}\n"
    f"  Kendall Tau: {metrics_full['kendall_tau']:.3f} → {metrics_selected['kendall_tau']:.3f}\n"
    f"  Silhouette: {metrics_full['silhouette_score']:.4f} → {metrics_selected['silhouette_score']:.4f}\n\n"
    f"Age Prediction R²: {cv_full.mean():.3f} → {cv_selected.mean():.3f}"
)
axes[1,2].text(0.1, 0.5, summary_text, transform=axes[1,2].transAxes, fontsize=11,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1,2].axis('off')
axes[1,2].set_title('Summary')

plt.tight_layout()
plt.savefig('report/images/figure4_trajectory_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure4_trajectory_validation.png")

# Save validation results
validation_results = {
    'full_metrics': metrics_full,
    'selected_metrics': metrics_selected,
    'cv_full_r2': cv_full.mean(),
    'cv_full_r2_std': cv_full.std(),
    'cv_selected_r2': cv_selected.mean(),
    'cv_selected_r2_std': cv_selected.std(),
    'n_features_full': X.shape[1],
    'n_features_selected': X_selected.shape[1]
}

import json
with open('outputs/validation_results.json', 'w') as f:
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer, np.float32, np.int32, np.float64, np.int64)):
            return float(obj)
        return obj
    json.dump(make_serializable(validation_results), f, indent=2)
print("Saved validation_results.json")
