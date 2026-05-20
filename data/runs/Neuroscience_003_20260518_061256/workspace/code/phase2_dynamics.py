"""
Phase 2: Dynamic Feature Scoring
Compute per-feature scores for how dynamically each feature changes along the continuous trajectory.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
adata = sc.read_h5ad('outputs/adata_preprocessed.h5ad')
X = adata.X
age = adata.obs['annotated_age'].values
feature_names = list(adata.var_names)

print(f"Computing dynamics scores for {len(feature_names)} features...")

# Parse feature info
feature_info = pd.DataFrame({
    'name': feature_names,
    'protein': [f.split('_')[2] if len(f.split('_')) >= 3 else 'unknown' for f in feature_names],
    'compartment': [f.split('_')[3] if len(f.split('_')) >= 4 else 'unknown' for f in feature_names],
    'measurement': ['_'.join(f.split('_')[:2]) for f in feature_names]
})

# Score 1: Spearman correlation with annotated_age
spearman_corr = []
spearman_pval = []
for i in range(X.shape[1]):
    r, p = stats.spearmanr(X[:, i], age)
    spearman_corr.append(r)
    spearman_pval.append(p)

# Score 2: Pearson correlation with annotated_age
pearson_corr = []
pearson_pval = []
for i in range(X.shape[1]):
    r, p = stats.pearsonr(X[:, i], age)
    pearson_corr.append(r)
    pearson_pval.append(p)

# Score 3: Mutual information (captures nonlinear relationships)
print("Computing mutual information...")
mi_scores = mutual_info_regression(X, age, random_state=42, n_neighbors=10)

# Score 4: Variance of feature values across age bins (ANOVA-like)
age_bins = pd.cut(age, bins=10)
within_var = []
between_var = []
f_statistics = []
for i in range(X.shape[1]):
    groups = [X[age_bins == b, i] for b in age_bins.unique() if len(X[age_bins == b, i]) > 0]
    if len(groups) > 1:
        grand_mean = X[:, i].mean()
        # Between-group variance
        group_means = [g.mean() for g in groups]
        group_sizes = [len(g) for g in groups]
        between_v = sum(s * (m - grand_mean)**2 for s, m in zip(group_sizes, group_means)) / max(len(groups) - 1, 1)
        # Within-group variance
        within_v = np.mean([np.var(g) for g in groups])
        if within_v > 0:
            f_stat = between_v / within_v
        else:
            f_stat = 0
    else:
        f_stat = 0
    between_var.append(between_v)
    within_var.append(within_v)
    f_statistics.append(f_stat)

# Score 5: Range of mean values across age bins
mean_by_bin = []
for i in range(X.shape[1]):
    means = []
    for b in age_bins.unique():
        mask = age_bins == b
        if mask.sum() > 0:
            means.append(X[mask, i].mean())
    mean_by_bin.append(max(means) - min(means) if len(means) > 0 else 0)

# Combine scores into composite dynamics score
# Normalize each score to [0, 1]
def normalize(x):
    x = np.array(x, dtype=float)
    mi, ma = np.nanmin(x), np.nanmax(x)
    if ma > mi:
        return (x - mi) / (ma - mi)
    return np.zeros_like(x)

spearman_abs = normalize(np.abs(spearman_corr))
pearson_abs = normalize(np.abs(pearson_corr))
mi_norm = normalize(mi_scores)
f_stat_norm = normalize(f_statistics)
range_norm = normalize(mean_by_bin)

# Composite dynamics score (weighted combination)
dynamics_score = (
    0.25 * spearman_abs + 
    0.25 * mi_norm + 
    0.25 * f_stat_norm + 
    0.15 * pearson_abs + 
    0.10 * range_norm
)

# Store all scores
scores_df = pd.DataFrame({
    'feature': feature_names,
    'protein': feature_info['protein'].values,
    'compartment': feature_info['compartment'].values,
    'measurement': feature_info['measurement'].values,
    'spearman_corr': spearman_corr,
    'spearman_pval': spearman_pval,
    'pearson_corr': pearson_corr,
    'pearson_pval': pearson_pval,
    'mi_score': mi_scores,
    'f_statistic': f_statistics,
    'dynamic_range': mean_by_bin,
    'dynamics_score': dynamics_score
})

# Sort by dynamics score
scores_df = scores_df.sort_values('dynamics_score', ascending=False).reset_index(drop=True)

# Print top features
print("\n=== Top 30 Dynamic Features ===")
print(scores_df[['feature', 'protein', 'compartment', 'dynamics_score', 
                  'spearman_corr', 'mi_score', 'f_statistic']].head(30).to_string())

print(f"\n=== Score Distributions ===")
print(f"Dynamics score: mean={scores_df['dynamics_score'].mean():.3f}, std={scores_df['dynamics_score'].std():.3f}")
print(f"Features with dynamics_score > 0.5: {(scores_df['dynamics_score'] > 0.5).sum()}")
print(f"Features with dynamics_score > 0.7: {(scores_df['dynamics_score'] > 0.7).sum()}")
print(f"Features with dynamics_score > 0.9: {(scores_df['dynamics_score'] > 0.9).sum()}")

# Save scores
scores_df.to_csv('outputs/feature_dynamics_scores.csv', index=False)
print("\nSaved feature_dynamics_scores.csv")

# Figure 2: Dynamics Score Overview
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 2a: Distribution of dynamics scores
axes[0,0].hist(scores_df['dynamics_score'], bins=30, edgecolor='black', alpha=0.7)
axes[0,0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
axes[0,0].set_xlabel('Dynamics Score')
axes[0,0].set_ylabel('Feature Count')
axes[0,0].set_title('Distribution of Dynamics Scores')
axes[0,0].legend()

# 2b: Spearman correlation vs MI score
scatter = axes[0,1].scatter(scores_df['spearman_corr'], scores_df['mi_score'], 
                           c=scores_df['dynamics_score'], cmap='viridis', alpha=0.6, s=20)
axes[0,1].set_xlabel('Spearman Correlation with Age')
axes[0,1].set_ylabel('Mutual Information Score')
axes[0,1].set_title('Correlation vs Nonlinear Dynamics')
plt.colorbar(scatter, ax=axes[0,1], label='Dynamics Score')

# 2c: Top features bar plot
top_n = 20
top_features = scores_df.head(top_n)
axes[0,2].barh(range(top_n), top_features['dynamics_score'].values[::-1])
axes[0,2].set_yticks(range(top_n))
axes[0,2].set_yticklabels(top_features['feature'].values[::-1], fontsize=8)
axes[0,2].set_xlabel('Dynamics Score')
axes[0,2].set_title(f'Top {top_n} Dynamic Features')

# 2d: Dynamics score by compartment
comp_scores = scores_df.groupby('compartment')['dynamics_score'].mean().sort_values(ascending=True)
axes[1,0].barh(comp_scores.index, comp_scores.values)
axes[1,0].set_xlabel('Mean Dynamics Score')
axes[1,0].set_title('Mean Dynamics Score by Compartment')

# 2e: Dynamics score by measurement type
meas_scores = scores_df.groupby('measurement')['dynamics_score'].mean().sort_values(ascending=True)
axes[1,1].barh(meas_scores.index, meas_scores.values)
axes[1,1].set_xlabel('Mean Dynamics Score')
axes[1,1].set_title('Mean Dynamics Score by Measurement Type')

# 2f: Top protein by dynamics score (aggregate across compartments)
protein_scores = scores_df.groupby('protein')['dynamics_score'].max().sort_values(ascending=False).head(20)
axes[1,2].barh(range(20), protein_scores.values[::-1])
axes[1,2].set_yticks(range(20))
axes[1,2].set_yticklabels(protein_scores.index[::-1], fontsize=8)
axes[1,2].set_xlabel('Max Dynamics Score')
axes[1,2].set_title('Top 20 Proteins by Max Feature Score')

plt.tight_layout()
plt.savefig('report/images/figure2_dynamics_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure2_dynamics_scores.png")
