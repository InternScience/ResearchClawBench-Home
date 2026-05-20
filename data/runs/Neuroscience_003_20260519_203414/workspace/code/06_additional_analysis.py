"""
Additional analysis: trajectory-specific validation and biological interpretation.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
adata = sc.read_h5ad('outputs/adata_trajectory.h5ad')
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
pseudotime = adata.obs['pseudotime'].values
phase = adata.obs['phase'].values
feature_names = adata.var_names.tolist()
rankings = pd.read_csv('outputs/feature_rankings.csv')

# ============================================================
# Figure 9: Cell cycle phase distribution along pseudotime
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
phase_order = ['G0', 'G1', 'S', 'G2']
phase_colors = {'G0': '#1f77b4', 'G1': '#ff7f0e', 'S': '#2ca02c', 'G2': '#d62728'}

for p in phase_order:
    mask = phase == p
    pt = pseudotime[mask]
    ax.hist(pt, bins=30, alpha=0.6, label=p, color=phase_colors[p], density=True)

ax.set_xlabel('Pseudotime (annotated_age)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Cell Cycle Phase Distribution Along Pseudotime', fontsize=14)
ax.legend(title='Phase')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure9_phase_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure9_phase_distribution.png")

# ============================================================
# Figure 10: Feature category analysis
# ============================================================
# Categorize features by type and location
def categorize_feature(name):
    loc = 'other'
    if '_nuc' in name:
        loc = 'nucleus'
    elif '_cyto' in name:
        loc = 'cytoplasm'
    elif '_cell' in name:
        loc = 'cell'
    elif '_ring' in name:
        loc = 'ring'
    
    ftype = 'other'
    if 'MeanEdge' in name:
        ftype = 'mean_edge'
    elif 'Med_' in name:
        ftype = 'median_intensity'
    elif 'Std_' in name:
        ftype = 'std_intensity'
    elif 'Intg_' in name:
        ftype = 'integrated_intensity'
    elif 'AreaShape' in name:
        ftype = 'morphology'
    
    return ftype, loc

rankings['feature_type'], rankings['location'] = zip(*rankings['feature'].apply(categorize_feature))

# Plot top 50 composite features by category
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

top50 = rankings.nsmallest(50, 'composite_rank')

# By location
loc_counts = top50['location'].value_counts()
axes[0].pie(loc_counts, labels=loc_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Top 50 Features by Subcellular Location')

# By feature type
type_counts = top50['feature_type'].value_counts()
axes[1].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Top 50 Features by Measurement Type')

plt.tight_layout()
plt.savefig('report/images/figure10_feature_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure10_feature_categories.png")

# ============================================================
# Figure 11: Pseudotime correlation for top features with polynomial fit
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

top6 = rankings.nsmallest(6, 'composite_rank')
for ax, (_, row) in zip(axes, top6.iterrows()):
    feat = row['feature']
    idx = feature_names.index(feat)
    y = X[:, idx]
    
    # Scatter with phase colors
    for p in ['G0', 'G1', 'S', 'G2']:
        mask = phase == p
        ax.scatter(pseudotime[mask], y[mask], label=p, alpha=0.5, s=10, color=phase_colors[p])
    
    # Polynomial fit
    z = np.polyfit(pseudotime, y, 3)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(pseudotime.min(), pseudotime.max(), 200)
    ax.plot(x_fit, p_fit(x_fit), 'k-', linewidth=2, label='Cubic fit')
    
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Expression')
    short_name = feat.replace('Int_Med_', '').replace('Int_MeanEdge_', '').replace('Int_Std_', '').replace('Int_Intg_', '')
    ax.set_title(f'{short_name}\n(r={row["abs_correlation"]:.3f}, R²={row["dynamic_r2"]:.3f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure11_top6_trajectory_fits.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure11_top6_trajectory_fits.png")

# ============================================================
# Compute final trajectory preservation scores for selected features
# ============================================================
selected_20 = rankings.nsmallest(20, 'composite_rank')['feature'].tolist()
idx20 = [feature_names.index(f) for f in selected_20]
X20 = X[:, idx20]

selected_50 = rankings.nsmallest(50, 'composite_rank')['feature'].tolist()
idx50 = [feature_names.index(f) for f in selected_50]
X50 = X[:, idx50]

# Correlation between full pseudotime and pseudotime from selected features
def compute_pseudotime_from_features(X_sub):
    pca = PCA(n_components=min(5, X_sub.shape[1]))
    X_pca = pca.fit_transform(X_sub)
    # First PC correlation with actual pseudotime
    corr = np.corrcoef(X_pca[:, 0], pseudotime)[0, 1]
    return corr, X_pca

corr_full, pca_full = compute_pseudotime_from_features(X)
corr_20, pca_20 = compute_pseudotime_from_features(X20)
corr_50, pca_50 = compute_pseudotime_from_features(X50)

# Procrustes-like alignment: compare distance matrices
def distance_matrix_correlation(X1, X2):
    D1 = pairwise_distances(X1, metric='euclidean')
    D2 = pairwise_distances(X2, metric='euclidean')
    # Use upper triangle
    triu = np.triu_indices_from(D1, k=1)
    return np.corrcoef(D1[triu], D2[triu])[0, 1]

from sklearn.metrics import pairwise_distances

dist_corr_20 = distance_matrix_correlation(pca_full[:, :3], pca_20[:, :3])
dist_corr_50 = distance_matrix_correlation(pca_full[:, :3], pca_50[:, :3])

# k-NN preservation
nbrs_full = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(pca_full[:, :10])
_, indices_full = nbrs_full.kneighbors(pca_full[:, :10])

nbrs_20 = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(pca_20[:, :10])
_, indices_20 = nbrs_20.kneighbors(pca_20[:, :10])

nbrs_50 = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(pca_50[:, :10])
_, indices_50 = nbrs_50.kneighbors(pca_50[:, :10])

def knn_preservation(idx1, idx2, k=15):
    overlaps = []
    for i in range(len(idx1)):
        set1 = set(idx1[i, 1:])
        set2 = set(idx2[i, 1:])
        overlaps.append(len(set1 & set2) / k)
    return np.mean(overlaps)

knn_20 = knn_preservation(indices_full, indices_20)
knn_50 = knn_preservation(indices_full, indices_50)

summary = {
    'full_features': {
        'n': X.shape[1],
        'pseudotime_corr': float(corr_full),
    },
    'selected_20': {
        'n': 20,
        'pseudotime_corr': float(corr_20),
        'dist_corr_to_full': float(dist_corr_20),
        'knn_preservation': float(knn_20),
    },
    'selected_50': {
        'n': 50,
        'pseudotime_corr': float(corr_50),
        'dist_corr_to_full': float(dist_corr_50),
        'knn_preservation': float(knn_50),
    }
}

import json
with open('outputs/trajectory_preservation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nTrajectory Preservation Summary:")
print(json.dumps(summary, indent=2))

# ============================================================
# Figure 12: Embedding comparison (PCA 1 vs Pseudotime)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(pseudotime, pca_full[:, 0], c=phase, cmap='tab10', alpha=0.5, s=10)
axes[0].set_xlabel('Annotated Pseudotime')
axes[0].set_ylabel('PC1')
axes[0].set_title(f'Full Features (r={corr_full:.3f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(pseudotime, pca_20[:, 0], c=phase, cmap='tab10', alpha=0.5, s=10)
axes[1].set_xlabel('Annotated Pseudotime')
axes[1].set_ylabel('PC1')
axes[1].set_title(f'Top 20 Features (r={corr_20:.3f})')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(pseudotime, pca_50[:, 0], c=phase, cmap='tab10', alpha=0.5, s=10)
axes[2].set_xlabel('Annotated Pseudotime')
axes[2].set_ylabel('PC1')
axes[2].set_title(f'Top 50 Features (r={corr_50:.3f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure12_pca_pseudotime.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure12_pca_pseudotime.png")

print("\nAll additional analyses complete!")
