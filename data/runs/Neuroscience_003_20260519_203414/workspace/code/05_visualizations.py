"""
Generate all figures for the report.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('report/images', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
adata = sc.read_h5ad('outputs/adata_trajectory.h5ad')
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()

rankings = pd.read_csv('outputs/feature_rankings.csv')
validation = pd.read_csv('outputs/validation_results.csv')

# ============================================================
# Figure 1: Data Overview - UMAP with phase, state, pseudotime
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# UMAP colored by phase
sc.pl.umap(adata, color='phase', ax=axes[0], show=False, title='Cell Cycle Phase', legend_loc='on data')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

# UMAP colored by state
sc.pl.umap(adata, color='state', ax=axes[1], show=False, title='Cell State')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')

# UMAP colored by pseudotime
sc.pl.umap(adata, color='pseudotime', ax=axes[2], show=False, title='Pseudotime (annotated_age)', color_map='viridis')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_data_overview.png")

# ============================================================
# Figure 2: Feature ranking comparison (top features by method)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
methods = ['variance_rank', 'correlation_rank', 'dynamic_rank', 'neighborhood_rank', 'pca_rank', 'mi_rank']
method_names = ['Variance', 'Pseudotime Correlation', 'Dynamic R²', 'Neighborhood', 'PCA Loadings', 'Mutual Information']

for ax, method, name in zip(axes.flat, methods, method_names):
    top = rankings.nsmallest(15, method)
    y_pos = np.arange(len(top))
    col = method.replace('_rank', '')
    if col == 'correlation':
        col = 'abs_correlation'
    elif col == 'dynamic':
        col = 'dynamic_r2'
    elif col == 'neighborhood':
        col = 'neighborhood_preservation'
    elif col == 'pca':
        col = 'pca_score'
    elif col == 'mi':
        col = 'mutual_info'
    
    values = top[col].values
    colors = sns.color_palette("viridis", len(top))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('Int_Med_', '').replace('Int_MeanEdge_', '').replace('Int_Std_', '').replace('Int_Intg_', '') for f in top['feature'].values], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Score')
    ax.set_title(name)
    
plt.tight_layout()
plt.savefig('report/images/figure2_feature_rankings.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_feature_rankings.png")

# ============================================================
# Figure 3: Validation curves
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['pseudotime_correlation', 'neighborhood_preservation', 'smoothness', 'phase_separability']
metric_names = ['Pseudotime Correlation', 'Neighborhood Preservation', 'Trajectory Smoothness', 'Phase Separability']

for ax, metric, name in zip(axes.flat, metrics, metric_names):
    for method in validation['method'].unique():
        sub = validation[validation['method'] == method]
        ax.plot(sub['n_features'], sub[metric], marker='o', label=method, linewidth=2)
    ax.set_xlabel('Number of Selected Features')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=7, loc='best')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure3_validation_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_validation_curves.png")

# ============================================================
# Figure 4: Top composite features expression along pseudotime
# ============================================================
top_composite = rankings.nsmallest(20, 'composite_rank')['feature'].tolist()
idx = [list(adata.var_names).index(f) for f in top_composite]
X_top = X[:, idx]

# Sort by pseudotime
order = np.argsort(adata.obs['pseudotime'].values)
X_sorted = X_top[order]
pt_sorted = adata.obs['pseudotime'].values[order]
phase_sorted = adata.obs['phase'].values[order]

# Smooth with rolling mean
window = 50
X_smooth = pd.DataFrame(X_sorted).rolling(window=window, center=True, min_periods=1).mean().values

fig, ax = plt.subplots(figsize=(14, 8))
for i, feat in enumerate(top_composite):
    label = feat.replace('Int_Med_', '').replace('Int_MeanEdge_', '').replace('Int_Std_', '').replace('Int_Intg_', '')
    ax.plot(pt_sorted, X_smooth[:, i], label=label, linewidth=1.5, alpha=0.8)

ax.set_xlabel('Pseudotime (annotated_age)', fontsize=12)
ax.set_ylabel('Expression (smoothed)', fontsize=12)
ax.set_title('Top 20 Composite-Ranked Features Along Pseudotime', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure4_top_features_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_top_features_trajectory.png")

# ============================================================
# Figure 5: Heatmap of top features x cells ordered by pseudotime
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
# Standardize each feature
X_top_scaled = StandardScaler().fit_transform(X_top)
X_top_sorted = X_top_scaled[order]

# Subsample cells for visualization
step = max(1, len(order) // 500)
sub_idx = order[::step]
X_heat = X_top_scaled[sub_idx]

# Use phase as color bar
phase_colors = {'G0': '#1f77b4', 'G1': '#ff7f0e', 'S': '#2ca02c', 'G2': '#d62728'}
phase_col = [phase_colors[p] for p in adata.obs['phase'].values[sub_idx]]

im = ax.imshow(X_heat.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_yticks(range(len(top_composite)))
ax.set_yticklabels([f.replace('Int_Med_', '').replace('Int_MeanEdge_', '').replace('Int_Std_', '').replace('Int_Intg_', '') for f in top_composite], fontsize=8)
ax.set_xlabel('Cells (ordered by pseudotime)', fontsize=12)
ax.set_title('Top 20 Features Heatmap (cells ordered by pseudotime)', fontsize=14)
plt.colorbar(im, ax=ax, label='Z-score')

# Add phase colorbar at top
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=phase_colors[p], label=p) for p in phase_colors]
ax.legend(handles=legend_elements, loc='upper right', title='Phase', fontsize=8)

plt.tight_layout()
plt.savefig('report/images/figure5_feature_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure5_feature_heatmap.png")

# ============================================================
# Figure 6: UMAP with selected features (n=20 composite)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Full features UMAP
axes[0].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
                c=adata.obs['pseudotime'], cmap='viridis', s=5, alpha=0.6)
axes[0].set_title('Full Features (241)')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

# Selected features UMAP (n=20)
sc.tl.pca(adata, svd_solver='arpack')
X_sel = X[:, idx[:20]]
pca_sel = PCA(n_components=10)
X_pca_sel = pca_sel.fit_transform(X_sel)

import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_umap_sel = reducer.fit_transform(X_pca_sel)

axes[1].scatter(X_umap_sel[:, 0], X_umap_sel[:, 1], 
                c=adata.obs['pseudotime'], cmap='viridis', s=5, alpha=0.6)
axes[1].set_title('Selected Features (20)')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')

# Selected features UMAP (n=50)
X_sel50 = X[:, idx[:50]]
pca_sel50 = PCA(n_components=10)
X_pca_sel50 = pca_sel50.fit_transform(X_sel50)
reducer50 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_umap_sel50 = reducer50.fit_transform(X_pca_sel50)

axes[2].scatter(X_umap_sel50[:, 0], X_umap_sel50[:, 1], 
                c=adata.obs['pseudotime'], cmap='viridis', s=5, alpha=0.6)
axes[2].set_title('Selected Features (50)')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=adata.obs['pseudotime'].min(), vmax=adata.obs['pseudotime'].max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Pseudotime')

plt.tight_layout()
plt.savefig('report/images/figure6_umap_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure6_umap_comparison.png")

# ============================================================
# Figure 7: Method comparison at n=20 (bar chart)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

n = 20
sub = validation[validation['n_features'] == n]

metrics = ['pseudotime_correlation', 'neighborhood_preservation', 'smoothness', 'phase_separability']
metric_names = ['Pseudotime\nCorrelation', 'Neighborhood\nPreservation', 'Trajectory\nSmoothness', 'Phase\nSeparability']

for ax, metric, name in zip(axes, metrics, metric_names):
    values = sub[metric].values
    colors = sns.color_palette("husl", len(sub))
    ax.bar(range(len(sub)), values, color=colors)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub['method'].values, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(name)
    ax.set_title(f'{name} (n={n})')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/figure7_method_comparison_n20.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure7_method_comparison_n20.png")

# ============================================================
# Figure 8: Scatter matrix of ranking scores
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = ['abs_correlation', 'dynamic_r2', 'neighborhood_preservation', 'pca_score', 'mutual_info']
labels = ['Pseudotime Corr', 'Dynamic R²', 'Nbr Pres', 'PCA Score', 'Mutual Info']
corr_matrix = rankings[corr_cols].corr()
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Correlation Between Feature Ranking Methods')
plt.tight_layout()
plt.savefig('report/images/figure8_ranking_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure8_ranking_correlation.png")

print("\nAll figures generated successfully!")
