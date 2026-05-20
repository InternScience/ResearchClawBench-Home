#!/usr/bin/env python3
"""
Feature selection for preserving continuous cellular trajectories in RPE single-cell imaging data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

# Setup
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
adata = sc.read_h5ad('data/adata_RPE.h5ad')
X = adata.layers['raw']
y = adata.obs['annotated_age'].values
features = adata.var_names.tolist()

print(f"Data shape: {X.shape}")
print(f"Age range: {y.min():.2f} - {y.max():.2f}")

# 1. Feature selection: top features by mutual information with age
print("Computing mutual information...")
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': features, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)
mi_df.to_csv('outputs/feature_mi_scores.csv', index=False)
print("Top 10 features by MI:")
print(mi_df.head(10))

# Select top k features (e.g., 20, 50)
k_values = [10, 20, 50, 100]
selected_features = {}
for k in k_values:
    selected = mi_df.head(k)['feature'].tolist()
    selected_features[k] = selected
    print(f"Selected top {k} features")

# 2. Validate trajectory preservation
def evaluate_trajectory_preservation(X_full, X_sel, y, n_components=2):
    """Compare how well age correlates with PC1 in full vs selected features."""
    pca_full = PCA(n_components=n_components).fit(X_full)
    pc1_full = pca_full.transform(X_full)[:, 0]
    corr_full, _ = spearmanr(pc1_full, y)
    
    pca_sel = PCA(n_components=n_components).fit(X_sel)
    pc1_sel = pca_sel.transform(X_sel)[:, 0]
    corr_sel, _ = spearmanr(pc1_sel, y)
    
    r2_full = r2_score(y, pc1_full)
    r2_sel = r2_score(y, pc1_sel)
    
    return corr_full, corr_sel, r2_full, r2_sel

results = []
for k in k_values:
    sel_idx = [features.index(f) for f in selected_features[k]]
    X_sel = X[:, sel_idx]
    corr_full, corr_sel, r2_full, r2_sel = evaluate_trajectory_preservation(X, X_sel, y)
    results.append({
        'k': k, 
        'corr_full': corr_full, 
        'corr_sel': corr_sel,
        'r2_full': r2_full,
        'r2_sel': r2_sel
    })
    print(f"k={k}: Full corr={corr_full:.3f}, Sel corr={corr_sel:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/trajectory_preservation_results.csv', index=False)

# 3. Generate figures
# Figure 1: MI score distribution
plt.figure(figsize=(8, 5))
sns.histplot(mi_df['mi_score'], bins=30, kde=True)
plt.xlabel('Mutual Information Score')
plt.ylabel('Number of Features')
plt.title('Distribution of Feature MI Scores with Annotated Age')
plt.tight_layout()
plt.savefig('report/images/figure1_mi_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Top features barplot
plt.figure(figsize=(10, 6))
top20 = mi_df.head(20)
sns.barplot(data=top20, y='feature', x='mi_score')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title('Top 20 Features by MI with Annotated Age')
plt.tight_layout()
plt.savefig('report/images/figure2_top_features.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Trajectory preservation comparison
plt.figure(figsize=(8, 5))
x = np.arange(len(k_values))
width = 0.35
fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, results_df['corr_full'], width, label='Full features (241)')
bars2 = ax.bar(x + width/2, results_df['corr_sel'], width, label='Selected features')
ax.set_ylabel('Spearman Correlation with Age (PC1)')
ax.set_xlabel('Number of Selected Features (k)')
ax.set_title('Trajectory Preservation: PC1-Age Correlation')
ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in k_values])
ax.legend()
ax.axhline(y=results_df['corr_full'].iloc[0], color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('report/images/figure3_trajectory_preservation.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Age vs PC1 for full and best k=50
k_best = 50
sel_idx = [features.index(f) for f in selected_features[k_best]]
X_sel = X[:, sel_idx]

pca_full = PCA(n_components=1).fit(X)
pc1_full = pca_full.transform(X)[:, 0]

pca_sel = PCA(n_components=1).fit(X_sel)
pc1_sel = pca_sel.transform(X_sel)[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=pc1_full, y=y, ax=axes[0], alpha=0.5, s=10)
axes[0].set_xlabel('PC1 (Full features)')
axes[0].set_ylabel('Annotated Age')
axes[0].set_title(f'Full (241 feats): corr={spearmanr(pc1_full, y)[0]:.3f}')

sns.scatterplot(x=pc1_sel, y=y, ax=axes[1], alpha=0.5, s=10, color='orange')
axes[1].set_xlabel('PC1 (Top 50 feats)')
axes[1].set_ylabel('Annotated Age')
axes[1].set_title(f'Selected (50 feats): corr={spearmanr(pc1_sel, y)[0]:.3f}')
plt.tight_layout()
plt.savefig('report/images/figure4_pc1_age_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("Analysis complete. Figures saved.")
print("Selected top 20 features saved to outputs/selected_top20_features.txt")
with open('outputs/selected_top20_features.txt', 'w') as f:
    f.write('\n'.join(selected_features[20])) 
print("Done.")