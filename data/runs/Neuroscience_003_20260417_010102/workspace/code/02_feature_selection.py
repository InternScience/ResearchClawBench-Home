#!/usr/bin/env python3
"""
Phase 2: Feature Selection Methods for Trajectory Preservation
Multiple approaches to select dynamically expressed molecular features
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import json

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# Load processed data
print("Loading processed data...")
adata = sc.read_h5ad('outputs/adata_processed.h5ad')
# Also load raw for feature selection on unscaled data
adata_raw = sc.read_h5ad('data/adata_RPE.h5ad')

n_cells, n_features = adata_raw.shape
print(f"Shape: {n_cells} cells × {n_features} features")

# Get the expression matrix (use raw/original values for feature selection)
X = adata_raw.X.copy()
feature_names = adata_raw.var_names.tolist()

# Get metadata
ages = adata.obs['annotated_age'].values
phases = adata.obs['phase'].values
states = adata.obs['state'].values
dpt = adata.obs['dpt_pseudotime'].values

# ============================================================
# Method 1: Variance-based feature selection (Highly Variable Features)
# ============================================================
print("\n=== Method 1: Variance-based ===")
feature_variance = np.var(X, axis=0)
feature_cv = np.std(X, axis=0) / (np.mean(X, axis=0) + 1e-10)  # coefficient of variation
variance_ranks = np.argsort(-feature_variance)  # descending
cv_ranks = np.argsort(-feature_cv)

print(f"Top 10 by variance: {[feature_names[i] for i in variance_ranks[:10]]}")
print(f"Top 10 by CV: {[feature_names[i] for i in cv_ranks[:10]]}")

# ============================================================
# Method 2: Trajectory-correlation (correlation with annotated age & DPT)
# ============================================================
print("\n=== Method 2: Trajectory Correlation ===")

# Correlation with annotated age
age_corrs = np.array([stats.spearmanr(X[:, j], ages)[0] for j in range(n_features)])
age_corr_abs = np.abs(age_corrs)
age_corr_ranks = np.argsort(-age_corr_abs)

# Correlation with DPT pseudotime
dpt_corrs = np.array([stats.spearmanr(X[:, j], dpt)[0] for j in range(n_features)])
dpt_corr_abs = np.abs(dpt_corrs)
dpt_corr_ranks = np.argsort(-dpt_corr_abs)

# Combined trajectory score (average of age and DPT absolute correlations)
traj_score = (age_corr_abs + dpt_corr_abs) / 2
traj_ranks = np.argsort(-traj_score)

print(f"Top 10 by age correlation: {[feature_names[i] for i in age_corr_ranks[:10]]}")
print(f"Top 10 by DPT correlation: {[feature_names[i] for i in dpt_corr_ranks[:10]]}")
print(f"Top 10 by combined trajectory: {[feature_names[i] for i in traj_ranks[:10]]}")

# ============================================================
# Method 3: Laplacian Score (graph-based, preserving local structure)
# ============================================================
print("\n=== Method 3: Laplacian Score ===")

def laplacian_score(X, W):
    """Compute Laplacian score for each feature.
    Lower score = more important for preserving graph structure.
    """
    n = X.shape[0]
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    scores = np.zeros(X.shape[1])
    ones = np.ones(n)
    D_sum = D.sum()
    
    for j in range(X.shape[1]):
        f = X[:, j]
        f_tilde = f - (f @ D @ ones) / (ones @ D @ ones) * ones
        numerator = f_tilde @ L @ f_tilde
        denominator = f_tilde @ D @ f_tilde
        if denominator > 1e-10:
            scores[j] = numerator / denominator
        else:
            scores[j] = 1e10  # uninformative
    return scores

# Use k-NN graph from scanpy (already computed)
# Build adjacency from the UMAP neighbors
from scipy.sparse import issparse
connectivity = adata.obsp['connectivities']
if issparse(connectivity):
    W = connectivity.toarray()
else:
    W = connectivity

# Standardize features for Laplacian score
X_std = StandardScaler().fit_transform(X)
lap_scores = laplacian_score(X_std, W)
lap_ranks = np.argsort(lap_scores)  # ascending (lower = better)

print(f"Top 10 by Laplacian score: {[feature_names[i] for i in lap_ranks[:10]]}")

# ============================================================
# Method 4: Mutual Information with trajectory
# ============================================================
print("\n=== Method 4: Mutual Information ===")

# MI with annotated age
mi_age = mutual_info_regression(X, ages, random_state=42, n_neighbors=5)
mi_age_ranks = np.argsort(-mi_age)

# MI with DPT
mi_dpt = mutual_info_regression(X, dpt, random_state=42, n_neighbors=5)
mi_dpt_ranks = np.argsort(-mi_dpt)

# Combined MI score
mi_combined = (mi_age / mi_age.max() + mi_dpt / mi_dpt.max()) / 2
mi_ranks = np.argsort(-mi_combined)

print(f"Top 10 by MI (age): {[feature_names[i] for i in mi_age_ranks[:10]]}")
print(f"Top 10 by MI (DPT): {[feature_names[i] for i in mi_dpt_ranks[:10]]}")
print(f"Top 10 by MI (combined): {[feature_names[i] for i in mi_ranks[:10]]}")

# ============================================================
# Method 5: Differential expression across phases
# ============================================================
print("\n=== Method 5: Differential Expression ===")

# Kruskal-Wallis test across cell cycle phases
kw_stats = np.zeros(n_features)
kw_pvals = np.zeros(n_features)
unique_phases = np.unique(phases)
for j in range(n_features):
    groups = [X[phases == p, j] for p in unique_phases]
    stat, pval = stats.kruskal(*groups)
    kw_stats[j] = stat
    kw_pvals[j] = pval

# Also test across states (cycling vs arrested)
state_mask = states != 'nan'
X_state = X[state_mask]
states_clean = states[state_mask]
mw_stats = np.zeros(n_features)
mw_pvals = np.zeros(n_features)
for j in range(n_features):
    cycling = X_state[states_clean == 'cycling', j]
    arrested = X_state[states_clean == 'arrested', j]
    stat, pval = stats.mannwhitneyu(cycling, arrested, alternative='two-sided')
    mw_stats[j] = stat
    mw_pvals[j] = pval

de_ranks_phase = np.argsort(-kw_stats)
de_ranks_state = np.argsort(-mw_stats)

print(f"Top 10 by phase DE: {[feature_names[i] for i in de_ranks_phase[:10]]}")
print(f"Top 10 by state DE: {[feature_names[i] for i in de_ranks_state[:10]]}")

# ============================================================
# Composite Score: Aggregate all methods
# ============================================================
print("\n=== Computing Composite Score ===")

# Normalize each score to [0, 1]
def rank_normalize(scores, ascending=False):
    """Convert scores to normalized ranks in [0, 1]. Higher = better."""
    if ascending:
        ranks = stats.rankdata(scores)  # lower original = lower rank
        ranks = 1 - (ranks / len(ranks))  # invert so lower original = higher normalized
    else:
        ranks = stats.rankdata(-scores)
        ranks = 1 - (ranks / len(ranks))
    return ranks

norm_variance = rank_normalize(feature_variance)
norm_cv = rank_normalize(feature_cv)
norm_age_corr = rank_normalize(age_corr_abs)
norm_dpt_corr = rank_normalize(dpt_corr_abs)
norm_lap = rank_normalize(lap_scores, ascending=True)  # lower is better
norm_mi = rank_normalize(mi_combined)
norm_kw = rank_normalize(kw_stats)
norm_mw = rank_normalize(mw_stats)

# Weighted composite: emphasize trajectory-related scores
composite = (
    0.10 * norm_variance +
    0.05 * norm_cv +
    0.20 * norm_age_corr +
    0.15 * norm_dpt_corr +
    0.15 * norm_lap +
    0.15 * norm_mi +
    0.10 * norm_kw +
    0.10 * norm_mw
)

composite_ranks = np.argsort(-composite)

print(f"Top 30 features by composite score:")
for i, idx in enumerate(composite_ranks[:30]):
    print(f"  {i+1}. {feature_names[idx]}: {composite[idx]:.4f}")

# ============================================================
# Save all scores to a DataFrame
# ============================================================
scores_df = pd.DataFrame({
    'feature': feature_names,
    'variance': feature_variance,
    'cv': feature_cv,
    'age_corr': age_corrs,
    'age_corr_abs': age_corr_abs,
    'dpt_corr': dpt_corrs,
    'dpt_corr_abs': dpt_corr_abs,
    'traj_score': traj_score,
    'laplacian_score': lap_scores,
    'mi_age': mi_age,
    'mi_dpt': mi_dpt,
    'mi_combined': mi_combined,
    'kw_stat_phase': kw_stats,
    'kw_pval_phase': kw_pvals,
    'mw_stat_state': mw_stats,
    'mw_pval_state': mw_pvals,
    'composite_score': composite,
    'norm_variance': norm_variance,
    'norm_cv': norm_cv,
    'norm_age_corr': norm_age_corr,
    'norm_dpt_corr': norm_dpt_corr,
    'norm_lap': norm_lap,
    'norm_mi': norm_mi,
    'norm_kw': norm_kw,
    'norm_mw': norm_mw,
})
scores_df = scores_df.sort_values('composite_score', ascending=False)
scores_df.to_csv('outputs/feature_scores.csv', index=False)
print("\nFeature scores saved to outputs/feature_scores.csv")

# Extract protein names and compartments
def parse_feature_name(name):
    parts = name.split('_')
    # Format: Int_MeasureType_Protein_Compartment
    if len(parts) >= 4:
        protein = parts[2]
        compartment = parts[3]
    elif len(parts) == 3:
        protein = parts[2]
        compartment = 'unknown'
    else:
        protein = name
        compartment = 'unknown'
    return protein, compartment

proteins = []
compartments = []
for name in feature_names:
    p, c = parse_feature_name(name)
    proteins.append(p)
    compartments.append(c)

scores_df['protein'] = [parse_feature_name(n)[0] for n in scores_df['feature'].values]
scores_df['compartment'] = [parse_feature_name(n)[1] for n in scores_df['feature'].values]
scores_df.to_csv('outputs/feature_scores_annotated.csv', index=False)

# ============================================================
# Figure 2: Feature Selection Comparison
# ============================================================
print("\n=== Generating Figure 2 ===")

# Top 20 features by each method
n_top = 20
methods = {
    'Variance': variance_ranks[:n_top],
    'Age Corr': age_corr_ranks[:n_top],
    'DPT Corr': dpt_corr_ranks[:n_top],
    'Laplacian': lap_ranks[:n_top],
    'MI Combined': mi_ranks[:n_top],
    'Phase DE': de_ranks_phase[:n_top],
    'State DE': de_ranks_state[:n_top],
    'Composite': composite_ranks[:n_top],
}

# Jaccard similarity between methods
method_names = list(methods.keys())
n_methods = len(method_names)
jaccard_matrix = np.zeros((n_methods, n_methods))
for i in range(n_methods):
    for j in range(n_methods):
        set_i = set(methods[method_names[i]])
        set_j = set(methods[method_names[j]])
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 2a: Jaccard similarity heatmap
im = axes[0].imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1)
axes[0].set_xticks(range(n_methods))
axes[0].set_yticks(range(n_methods))
axes[0].set_xticklabels(method_names, rotation=45, ha='right')
axes[0].set_yticklabels(method_names)
for i in range(n_methods):
    for j in range(n_methods):
        axes[0].text(j, i, f'{jaccard_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=axes[0], label='Jaccard Similarity')
axes[0].set_title('Method Agreement (Top 20 Features)')

# 2b: Composite score distribution
axes[1].hist(composite, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
axes[1].axvline(np.sort(composite)[-30], color='red', linestyle='--', label='Top 30 threshold')
axes[1].axvline(np.sort(composite)[-50], color='orange', linestyle='--', label='Top 50 threshold')
axes[1].set_xlabel('Composite Score')
axes[1].set_ylabel('Count')
axes[1].set_title('Composite Feature Score Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('report/images/fig2_feature_selection_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Top features bar chart
# ============================================================
print("\n=== Generating Figure 3 ===")

top_n = 30
top_features = scores_df.head(top_n)

fig, ax = plt.subplots(figsize=(14, 8))
colors = []
cmap = {'cell': '#2196F3', 'cyto': '#4CAF50', 'nuc': '#FF9800', 'ring': '#9C27B0', 'unknown': '#607D8B'}
for comp in top_features['compartment']:
    colors.append(cmap.get(comp, '#607D8B'))

bars = ax.barh(range(top_n), top_features['composite_score'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Composite Score')
ax.set_title(f'Top {top_n} Dynamically Expressed Features')

# Legend for compartments
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=k) for k, c in cmap.items() if k != 'unknown']
ax.legend(handles=legend_elements, title='Compartment', loc='lower right')

plt.tight_layout()
plt.savefig('report/images/fig3_top_features.png', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# Save selected feature lists
selected_30 = scores_df.head(30)['feature'].tolist()
selected_50 = scores_df.head(50)['feature'].tolist()

with open('outputs/selected_features.json', 'w') as f:
    json.dump({
        'top_30': selected_30,
        'top_50': selected_50,
    }, f, indent=2)

print(f"\nSelected 30 features: {selected_30}")
print(f"\nPhase 2 complete.")
