"""
Phase 3: Feature Selection with Redundancy Reduction
Select a minimal set of dynamic, non-redundant features.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load data
adata = sc.read_h5ad('outputs/adata_preprocessed.h5ad')
X = adata.X
age = adata.obs['annotated_age'].values
feature_names = list(adata.var_names)

# Load dynamics scores
scores_df = pd.read_csv('outputs/feature_dynamics_scores.csv')
print(f"Loaded {len(scores_df)} features with dynamics scores")

# Phase 3a: Correlation-based redundancy reduction
# Use hierarchical clustering on feature correlations to select representative features
feature_idx = {f: i for i, f in enumerate(feature_names)}

# Compute correlation matrix for all features
corr_matrix = np.corrcoef(X.T)

# Greedy feature selection: iteratively pick the most dynamic feature
# and remove all features correlated above a threshold
def greedy_select(scores_df, corr_matrix, threshold=0.85, target_n=40):
    """
    Greedy selection: pick top dynamic features, remove correlated ones.
    """
    selected = []
    excluded = set()
    
    sorted_features = scores_df.sort_values('dynamics_score', ascending=False)
    
    for _, row in sorted_features.iterrows():
        feat_name = row['feature']
        if feat_name in excluded:
            continue
        
        feat_idx = feature_names.index(feat_name)
        selected.append({
            'feature': feat_name,
            'dynamics_score': row['dynamics_score'],
            'protein': row['protein'],
            'compartment': row['compartment'],
            'measurement': row['measurement'],
            'spearman_corr': row['spearman_corr'],
            'mi_score': row['mi_score']
        })
        
        # Find and exclude features correlated above threshold
        for j, other_name in enumerate(feature_names):
            if other_name == feat_name or other_name in excluded:
                continue
            if abs(corr_matrix[feat_idx, j]) > threshold:
                excluded.add(other_name)
        
        if len(selected) >= target_n:
            break
    
    return pd.DataFrame(selected)

# Test different redundancy thresholds
thresholds = [0.7, 0.8, 0.85, 0.9]
selection_results = {}

for thresh in thresholds:
    sel = greedy_select(scores_df, corr_matrix, threshold=thresh, target_n=50)
    selection_results[thresh] = sel
    print(f"\nThreshold={thresh}: selected {len(sel)} features")

# Use threshold=0.85 as primary selection
selected_df = selection_results[0.85]
print(f"\n=== Final selected features (threshold=0.85): {len(selected_df)} features ===")
print(selected_df[['feature', 'protein', 'compartment', 'dynamics_score', 'spearman_corr']].to_string())

# Create feature selection summary
selected_idx = [feature_names.index(f) for f in selected_df['feature']]
selected_names = selected_df['feature'].tolist()

# Save selection
selected_df.to_csv('outputs/selected_features.csv', index=False)
np.save('outputs/selected_feature_indices.npy', selected_idx)

# Figure 3: Feature Selection Results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 3a: Correlation heatmap of selected features
selected_corr = corr_matrix[np.ix_(selected_idx, selected_idx)]
sns.heatmap(selected_corr, ax=axes[0,0], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False, square=True)
axes[0,0].set_title('Correlation Matrix of Selected Features')

# 3b: Selection summary by protein
protein_counts = selected_df['protein'].value_counts().head(15)
axes[0,1].barh(protein_counts.index[::-1], protein_counts.values[::-1])
axes[0,1].set_xlabel('Number of Selected Features')
axes[0,1].set_title('Selected Features by Protein')

# 3c: Selection summary by compartment
comp_counts = selected_df['compartment'].value_counts()
axes[0,2].bar(comp_counts.index, comp_counts.values)
axes[0,2].set_ylabel('Number of Selected Features')
axes[0,2].set_title('Selected Features by Compartment')

# 3d: Dynamics score comparison (selected vs all)
axes[1,0].hist(scores_df['dynamics_score'], bins=30, alpha=0.5, label='All features', edgecolor='black')
axes[1,0].hist(selected_df['dynamics_score'], bins=30, alpha=0.5, label='Selected features', edgecolor='black')
axes[1,0].set_xlabel('Dynamics Score')
axes[1,0].set_ylabel('Feature Count')
axes[1,0].set_title('Dynamics Score Distribution')
axes[1,0].legend()

# 3e: Selection efficiency across thresholds
n_selected = [len(selection_results[t]) for t in thresholds]
mean_score = [selection_results[t]['dynamics_score'].mean() for t in thresholds]
ax2 = axes[1,1].twinx()
axes[1,1].bar(range(len(thresholds)), n_selected, alpha=0.7, label='N features')
ax2.plot(range(len(thresholds)), mean_score, 'ro-', label='Mean score')
axes[1,1].set_xticks(range(len(thresholds)))
axes[1,1].set_xticklabels([str(t) for t in thresholds])
axes[1,1].set_xlabel('Correlation Threshold')
axes[1,1].set_ylabel('Number of Features')
ax2.set_ylabel('Mean Dynamics Score')
axes[1,1].set_title('Selection Efficiency vs Threshold')

# 3f: Scatter of top selected features (Age vs Feature Value)
top_feats = selected_df.head(6)
n_cols = 3
n_rows = 2
for idx, (_, row) in enumerate(top_feats.iterrows()):
    r = idx // n_cols
    c = idx % n_cols
    feat_idx_i = feature_names.index(row['feature'])
    axes[1,0].scatter(age, X[:, feat_idx_i], alpha=0.1, s=5, c='gray')
axes[1,0].clear()

# Replace with a proper scatter plot
ax = axes[1,0]
for i, (_, row) in enumerate(selected_df.head(6).iterrows()):
    feat_idx_i = feature_names.index(row['feature'])
    ax.scatter(age, X[:, feat_idx_i], alpha=0.15, s=5, label=row['feature'])
ax.set_xlabel('Annotated Age')
ax.set_ylabel('Feature Value (z-score)')
ax.set_title('Top 6 Selected Features vs Age')
ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig('report/images/figure3_feature_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure3_feature_selection.png")

# Figure 3b: Heatmap of selected features across cells
# Sort cells by age
age_sorted_idx = np.argsort(age)
X_selected = X[np.ix_(age_sorted_idx, selected_idx)]

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(X_selected.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_xlabel('Cells (sorted by annotated_age)')
ax.set_ylabel('Selected Features')
ax.set_title('Selected Features Heatmap (cells sorted by age)')
plt.colorbar(im, ax=ax, label='Z-score')

# Add age colorbar
age_sorted = age[age_sorted_idx]
age_norm = (age_sorted - age_sorted.min()) / (age_sorted.max() - age_sorted.min())
cmap = plt.cm.viridis
age_colors = cmap(age_norm)
# Add a second axis for age
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=age.min(), vmax=age.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Annotated Age')

plt.tight_layout()
plt.savefig('report/images/figure3b_heatmap_selected.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3b_heatmap_selected.png")
