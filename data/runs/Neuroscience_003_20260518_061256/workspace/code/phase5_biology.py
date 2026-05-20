"""
Phase 5: Biological Interpretation
Identify key proteins, compartments, and mechanisms driving trajectory.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load data
adata = sc.read_h5ad('outputs/adata_preprocessed.h5ad')
X = adata.X
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].values
state = adata.obs['state'].values
batch = adata.obs['batch'].values
feature_names = list(adata.var_names)

# Load selected features and scores
selected_df = pd.read_csv('outputs/selected_features.csv')
scores_df = pd.read_csv('outputs/feature_dynamics_scores.csv')
selected_idx = [feature_names.index(f) for f in selected_df['feature']]
X_selected = X[:, selected_idx]

# === Biological Analysis 1: Protein-level aggregation ===
# Average selected features per protein
protein_groups = selected_df.groupby('protein')
protein_summary = []
for protein, group in protein_groups:
    feat_idx_p = [feature_names.index(f) for f in group['feature']]
    protein_mean = X[:, feat_idx_p].mean(axis=1)
    r, p = stats.spearmanr(protein_mean, age)
    protein_summary.append({
        'protein': protein,
        'n_features': len(group),
        'spearman_r': r,
        'pvalue': p,
        'mean_dynamics_score': group['dynamics_score'].mean(),
        'compartments': ', '.join(group['compartment'].unique())
    })
protein_summary_df = pd.DataFrame(protein_summary).sort_values('spearman_r', key=abs, ascending=False)
print("=== Top Proteins by Trajectory Correlation ===")
print(protein_summary_df.head(15).to_string())

# === Biological Analysis 2: Compartment-specific dynamics ===
print("\n=== Compartment Analysis ===")
comp_analysis = selected_df.groupby('compartment').agg({
    'dynamics_score': ['mean', 'max', 'count'],
    'spearman_corr': 'mean'
})
print(comp_analysis)

# === Biological Analysis 3: Cell cycle analysis ===
print("\n=== Cell Cycle Phase Analysis ===")
# Average selected features per phase
phase_avg = pd.DataFrame(X_selected, columns=selected_df['feature'])
phase_avg['phase'] = phase
phase_means = phase_avg.groupby('phase').mean()

# Heatmap of phase means for top features
top_features = selected_df.head(15)['feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Phase heatmap
sns.heatmap(phase_means[top_features].T, ax=axes[0], cmap='RdBu_r', center=0,
            xticklabels=True, yticklabels=True, annot=True, fmt='.2f')
axes[0].set_title('Selected Features by Cell Cycle Phase')

# State analysis
print("\n=== Cell State Analysis ===")
state_avg = pd.DataFrame(X_selected, columns=selected_df['feature'])
state_avg['state'] = state
state_means = state_avg.groupby('state').mean()
print(state_means[top_features[:10]].T)

# State heatmap
sns.heatmap(state_means[top_features].T, ax=axes[1], cmap='RdBu_r', center=0,
            xticklabels=True, yticklabels=True, annot=True, fmt='.2f')
axes[1].set_title('Selected Features by Cell State')

plt.tight_layout()
plt.savefig('report/images/figure5_biology_cellcycle.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure5_biology_cellcycle.png")

# === Biological Analysis 4: Age-binned feature profiles ===
print("\n=== Age-Binned Feature Profiles ===")
age_bins = pd.cut(age, bins=8)
feature_by_age = pd.DataFrame(X_selected, columns=selected_df['feature'])
feature_by_age['age_bin'] = age_bins
age_means = feature_by_age.groupby('age_bin').mean()

# Figure 5b: Feature dynamics over age
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top 6 features trajectory
top6 = selected_df.head(6)
for i, (_, row) in enumerate(top6.iterrows()):
    ax = axes[i//3, i%3]
    feat_idx_i = feature_names.index(row['feature'])
    ax.scatter(age, X[:, feat_idx_i], alpha=0.1, s=5, c='gray')
    # Add smoothed line
    age_sorted = np.sort(np.unique(age))
    bin_means = []
    for a in age_sorted:
        mask = np.abs(age - a) < 1.0
        if mask.sum() > 0:
            bin_means.append((a, X[mask, feat_idx_i].mean()))
    if bin_means:
        bin_means = np.array(bin_means)
        ax.plot(bin_means[:, 0], bin_means[:, 1], 'r-', linewidth=2)
    ax.set_xlabel('Age')
    ax.set_ylabel('Feature Value')
    ax.set_title(f"{row['feature']}\n(r={row['spearman_corr']:.3f})")

plt.tight_layout()
plt.savefig('report/images/figure5b_feature_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5b_feature_trajectories.png")

# === Biological Analysis 5: Protein network analysis ===
# Identify correlated protein groups
print("\n=== Protein Correlation Network ===")
# Aggregate by protein
unique_proteins = selected_df['protein'].unique()
protein_matrix = np.zeros((len(age), len(unique_proteins)))
for j, protein in enumerate(unique_proteins):
    feat_mask = selected_df['protein'] == protein
    feat_idx_p = [feature_names.index(f) for f in selected_df[feat_mask]['feature']]
    protein_matrix[:, j] = X[:, feat_idx_p].mean(axis=1)

protein_corr = np.corrcoef(protein_matrix.T)

# Plot protein correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(protein_corr, xticklabels=unique_proteins, yticklabels=unique_proteins,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[0], square=True)
axes[0].set_title('Protein Correlation Heatmap (Selected Features)')
axes[0].tick_params(axis='x', rotation=45)

# Top protein correlations with age
protein_age_corr = []
for j, protein in enumerate(unique_proteins):
    r, p = stats.spearmanr(protein_matrix[:, j], age)
    protein_age_corr.append({'protein': protein, 'spearman_r': r, 'pvalue': p})
protein_age_df = pd.DataFrame(protein_age_corr).sort_values('spearman_r', key=abs, ascending=False)

colors = ['red' if r > 0 else 'blue' for r in protein_age_df['spearman_r']]
axes[1].barh(protein_age_df['protein'], protein_age_df['spearman_r'], color=colors, alpha=0.7)
axes[1].axvline(x=0, color='black', linewidth=0.5)
axes[1].set_xlabel('Spearman Correlation with Age')
axes[1].set_title('Protein-Level Trajectory Correlations')

plt.tight_layout()
plt.savefig('report/images/figure5c_protein_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5c_protein_network.png")

# === Biological Analysis 6: Batch effect analysis ===
print("\n=== Batch Effect Analysis ===")
# Check if batch effects are reduced in selected features
from sklearn.metrics import silhouette_score

# Batch effect: silhouette score with batch labels
for X_sub, name in [(X, 'Full'), (X_selected, 'Selected')]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, X_sub.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_sub)
    sil_batch = silhouette_score(X_pca, batch)
    sil_age = silhouette_score(X_pca, pd.qcut(age, q=10, labels=False, duplicates='drop'))
    print(f"{name} features: Batch silhouette={sil_batch:.4f}, Age silhouette={sil_age:.4f}")

# Save biological analysis results
bio_results = {
    'protein_summary': protein_summary_df.to_dict('records'),
    'protein_age_correlations': protein_age_df.to_dict('records')
}

import json
with open('outputs/biological_analysis.json', 'w') as f:
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    json.dump(make_serializable(bio_results), f, indent=2)
print("\nSaved biological_analysis.json")
