#!/usr/bin/env python3
"""
Phase 3: Evaluation of Feature Selection - Trajectory Preservation Assessment
Compare trajectory quality with different feature subsets
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# Load data
print("Loading data...")
adata_raw = sc.read_h5ad('data/adata_RPE.h5ad')
adata_proc = sc.read_h5ad('outputs/adata_processed.h5ad')

# Load selected features
with open('outputs/selected_features.json', 'r') as f:
    selected = json.load(f)
selected_30 = selected['top_30']
selected_50 = selected['top_50']

# Load feature scores
scores_df = pd.read_csv('outputs/feature_scores_annotated.csv')

X_full = adata_raw.X.copy()
feature_names = adata_raw.var_names.tolist()
ages = adata_raw.obs['annotated_age'].values
phases = adata_raw.obs['phase'].values
states = adata_raw.obs['state'].values

# ============================================================
# Function to evaluate trajectory quality for a feature subset
# ============================================================
def evaluate_trajectory(X_subset, feature_subset_name, ages, phases, states, n_neighbors=15, n_pcs=None):
    """Evaluate trajectory preservation for a given feature subset."""
    from anndata import AnnData
    
    adata_sub = AnnData(X=X_subset.copy())
    adata_sub.obs['annotated_age'] = ages
    adata_sub.obs['phase'] = pd.Categorical(phases)
    adata_sub.obs['state'] = pd.Categorical(states)
    
    # Scale
    sc.pp.scale(adata_sub, max_value=10)
    
    # PCA
    n_comps = min(50, X_subset.shape[1] - 1)
    sc.tl.pca(adata_sub, n_comps=n_comps)
    
    # Neighbors and UMAP
    n_pcs_use = min(30, n_comps)
    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, n_pcs=n_pcs_use)
    sc.tl.umap(adata_sub, random_state=42)
    
    # Diffusion pseudotime
    sc.tl.diffmap(adata_sub, n_comps=min(15, n_comps))
    
    # Find root (youngest G1 cell)
    g1_mask = adata_sub.obs['phase'] == 'G1'
    if g1_mask.sum() > 0:
        youngest_g1 = adata_sub.obs.loc[g1_mask, 'annotated_age'].idxmin()
        root_idx = adata_sub.obs.index.get_loc(youngest_g1)
    else:
        root_idx = 0
    adata_sub.uns['iroot'] = root_idx
    sc.tl.dpt(adata_sub)
    
    # Metrics
    dpt = adata_sub.obs['dpt_pseudotime'].values
    
    # 1. Correlation between DPT and annotated age
    corr_dpt_age, pval_dpt_age = stats.spearmanr(dpt, ages)
    
    # 2. Silhouette score for phase separation
    umap_coords = adata_sub.obsm['X_umap']
    sil_phase = silhouette_score(umap_coords, phases, metric='euclidean')
    
    # 3. Silhouette score for state separation
    state_mask = states != 'nan'
    if state_mask.sum() > 10:
        sil_state = silhouette_score(umap_coords[state_mask], states[state_mask], metric='euclidean')
    else:
        sil_state = np.nan
    
    # 4. Trajectory continuity: average distance between age-adjacent cells in UMAP
    age_order = np.argsort(ages)
    umap_ordered = umap_coords[age_order]
    diffs = np.sqrt(np.sum(np.diff(umap_ordered, axis=0)**2, axis=1))
    traj_continuity = np.mean(diffs)
    
    # 5. Local age consistency: for each cell, how well do its k nearest neighbors share similar ages
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=15, metric='euclidean')
    nn.fit(umap_coords)
    distances, indices = nn.kneighbors(umap_coords)
    age_consistency = np.mean([
        np.corrcoef(ages[i] * np.ones(15), ages[indices[i]])[0, 1] 
        for i in range(len(ages)) if np.std(ages[indices[i]]) > 0
    ])
    
    # 6. Batch mixing: average silhouette for batch (lower = better mixing)
    batches = adata_sub.obs.get('batch', None)
    if batches is not None:
        # Use adata_raw batch info
        batch_vals = adata_raw.obs['batch'].values
        sil_batch = silhouette_score(umap_coords, batch_vals, metric='euclidean')
    else:
        sil_batch = np.nan
    
    # 7. PCA variance explained by top components
    var_explained = np.sum(adata_sub.uns['pca']['variance_ratio'][:min(10, n_comps)])
    
    metrics = {
        'name': feature_subset_name,
        'n_features': X_subset.shape[1],
        'corr_dpt_age': float(corr_dpt_age),
        'pval_dpt_age': float(pval_dpt_age),
        'sil_phase': float(sil_phase),
        'sil_state': float(sil_state),
        'traj_continuity': float(traj_continuity),
        'age_consistency': float(age_consistency) if not np.isnan(age_consistency) else 0.0,
        'sil_batch': float(sil_batch),
        'var_explained_10pc': float(var_explained),
    }
    
    return metrics, adata_sub

# ============================================================
# Evaluate different feature subsets
# ============================================================
print("\n=== Evaluating Feature Subsets ===")

# Get feature indices
def get_feature_indices(feature_list):
    return [feature_names.index(f) for f in feature_list if f in feature_names]

# Random feature subsets for comparison
np.random.seed(42)
random_30 = np.random.choice(len(feature_names), 30, replace=False)
random_50 = np.random.choice(len(feature_names), 50, replace=False)

# Variance-based top features
var_ranks = scores_df.sort_values('variance', ascending=False)
var_30 = var_ranks.head(30)['feature'].tolist()
var_50 = var_ranks.head(50)['feature'].tolist()

# Subsets to evaluate
subsets = {
    'All Features (241)': (X_full, None),
    'Composite Top 30': (X_full[:, get_feature_indices(selected_30)], selected_30),
    'Composite Top 50': (X_full[:, get_feature_indices(selected_50)], selected_50),
    'Variance Top 30': (X_full[:, get_feature_indices(var_30)], var_30),
    'Variance Top 50': (X_full[:, get_feature_indices(var_50)], var_50),
    'Random 30': (X_full[:, random_30], [feature_names[i] for i in random_30]),
    'Random 50': (X_full[:, random_50], [feature_names[i] for i in random_50]),
}

results = []
adata_dict = {}

for name, (X_sub, feat_list) in subsets.items():
    print(f"\nEvaluating: {name} ({X_sub.shape[1]} features)")
    metrics, adata_sub = evaluate_trajectory(X_sub, name, ages, phases, states)
    results.append(metrics)
    adata_dict[name] = adata_sub
    print(f"  DPT-Age corr: {metrics['corr_dpt_age']:.4f}")
    print(f"  Phase silhouette: {metrics['sil_phase']:.4f}")
    print(f"  State silhouette: {metrics['sil_state']:.4f}")
    print(f"  Batch silhouette: {metrics['sil_batch']:.4f}")
    print(f"  Trajectory continuity: {metrics['traj_continuity']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/evaluation_results.csv', index=False)
print("\nEvaluation results saved.")
print(results_df.to_string())

# ============================================================
# Figure 4: UMAP Comparison
# ============================================================
print("\n=== Generating Figure 4: UMAP Comparison ===")

fig, axes = plt.subplots(3, 4, figsize=(24, 18))

plot_subsets = ['All Features (241)', 'Composite Top 30', 'Variance Top 30', 'Random 30']

for col, name in enumerate(plot_subsets):
    adata_sub = adata_dict[name]
    
    # Row 1: colored by phase
    sc.pl.umap(adata_sub, color='phase', ax=axes[0, col], show=False, 
               title=f'{name}\n(Phase)')
    
    # Row 2: colored by age
    sc.pl.umap(adata_sub, color='annotated_age', ax=axes[1, col], show=False,
               title=f'{name}\n(Age)', color_map='viridis')
    
    # Row 3: colored by DPT
    sc.pl.umap(adata_sub, color='dpt_pseudotime', ax=axes[2, col], show=False,
               title=f'{name}\n(DPT)', color_map='magma')

plt.tight_layout()
plt.savefig('report/images/fig4_umap_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Metrics Comparison Bar Chart
# ============================================================
print("\n=== Generating Figure 5: Metrics Comparison ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics_to_plot = [
    ('corr_dpt_age', 'DPT-Age Correlation', True),
    ('sil_phase', 'Phase Silhouette Score', True),
    ('sil_state', 'State Silhouette Score', True),
    ('sil_batch', 'Batch Silhouette (lower=better mixing)', False),
    ('traj_continuity', 'Trajectory Continuity (lower=smoother)', False),
    ('var_explained_10pc', 'Variance Explained (Top 10 PCs)', True),
]

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548']
subset_names = results_df['name'].values

for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = results_df[metric].values
    bars = ax.bar(range(len(subset_names)), values, color=colors[:len(subset_names)])
    ax.set_xticks(range(len(subset_names)))
    ax.set_xticklabels(subset_names, rotation=45, ha='right', fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(metric)
    
    # Highlight best
    if higher_better:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('report/images/fig5_metrics_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Feature Heatmap for Top 30 Selected Features
# ============================================================
print("\n=== Generating Figure 6: Feature Heatmap ===")

# Sort cells by annotated age
age_order = np.argsort(ages)
X_selected = X_full[:, get_feature_indices(selected_30)]
X_selected_ordered = X_selected[age_order]

# Standardize for visualization
X_viz = StandardScaler().fit_transform(X_selected_ordered)
X_viz = np.clip(X_viz, -3, 3)

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(X_viz.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
               interpolation='nearest')
ax.set_yticks(range(len(selected_30)))
ax.set_yticklabels(selected_30, fontsize=7)
ax.set_xlabel('Cells (ordered by age)')
ax.set_title('Top 30 Selected Features (cells ordered by annotated age)')

# Add age colorbar on top
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)
plt.colorbar(im, cax=cax, label='Standardized Expression')

# Add age annotation
ax_top = divider.append_axes("top", size="5%", pad=0.1)
ages_ordered = ages[age_order]
ax_top.fill_between(range(len(ages_ordered)), 0, ages_ordered, color='steelblue', alpha=0.5)
ax_top.set_xlim(0, len(ages_ordered))
ax_top.set_ylabel('Age')
ax_top.set_xticks([])

plt.savefig('report/images/fig6_feature_heatmap.png', bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================
# Figure 7: Protein-level analysis
# ============================================================
print("\n=== Generating Figure 7: Protein-level Analysis ===")

# Aggregate scores by protein
protein_scores = scores_df.groupby('protein').agg({
    'composite_score': 'max',
    'age_corr_abs': 'max',
    'dpt_corr_abs': 'max',
    'mi_combined': 'max',
    'kw_stat_phase': 'max',
}).reset_index()
protein_scores = protein_scores.sort_values('composite_score', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Top 20 proteins
top_proteins = protein_scores.head(20)
axes[0].barh(range(20), top_proteins['composite_score'].values, color='steelblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top_proteins['protein'].values)
axes[0].invert_yaxis()
axes[0].set_xlabel('Max Composite Score')
axes[0].set_title('Top 20 Proteins by Composite Score')

# Compartment distribution of top features
top_50_df = scores_df.head(50)
comp_counts = top_50_df['compartment'].value_counts()
axes[1].pie(comp_counts.values, labels=comp_counts.index, autopct='%1.1f%%',
            colors=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B'])
axes[1].set_title('Compartment Distribution of Top 50 Features')

plt.tight_layout()
plt.savefig('report/images/fig7_protein_analysis.png', bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

# ============================================================
# Figure 8: Individual feature trajectories
# ============================================================
print("\n=== Generating Figure 8: Feature Trajectories ===")

# Plot top 12 features against annotated age
top_12 = selected_30[:12]
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for idx, feat in enumerate(top_12):
    ax = axes[idx // 4, idx % 4]
    feat_idx = feature_names.index(feat)
    
    # Scatter with LOESS-like smoothing
    ax.scatter(ages, X_full[:, feat_idx], c=ages, cmap='viridis', s=3, alpha=0.3)
    
    # Moving average
    sorted_idx = np.argsort(ages)
    window = 100
    ages_sorted = ages[sorted_idx]
    vals_sorted = X_full[sorted_idx, feat_idx]
    ages_smooth = np.convolve(ages_sorted, np.ones(window)/window, mode='valid')
    vals_smooth = np.convolve(vals_sorted, np.ones(window)/window, mode='valid')
    ax.plot(ages_smooth, vals_smooth, 'r-', linewidth=2, label='Moving avg')
    
    ax.set_xlabel('Annotated Age')
    ax.set_ylabel('Expression')
    ax.set_title(feat.replace('Int_Med_', '').replace('Int_MeanEdge_', '').replace('Int_Std_', '').replace('Int_Intg_', ''), fontsize=10)
    ax.legend(fontsize=7)

plt.suptitle('Top 12 Selected Features vs. Annotated Age', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig8_feature_trajectories.png', bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

# ============================================================
# Additional: Batch effect analysis
# ============================================================
print("\n=== Batch Effect Analysis ===")

batch_vals = adata_raw.obs['batch'].values
batch_effects = {}
for feat in selected_30:
    feat_idx = feature_names.index(feat)
    b1 = X_full[batch_vals == '1', feat_idx]
    b2 = X_full[batch_vals == '2', feat_idx]
    stat, pval = stats.mannwhitneyu(b1, b2, alternative='two-sided')
    batch_effects[feat] = {'stat': float(stat), 'pval': float(pval)}

batch_df = pd.DataFrame(batch_effects).T
batch_df.columns = ['mw_stat', 'pval']
batch_df['significant'] = batch_df['pval'] < 0.05
print(f"Features with significant batch effect: {batch_df['significant'].sum()}/{len(batch_df)}")
batch_df.to_csv('outputs/batch_effects.csv')

# ============================================================
# Save final results summary
# ============================================================
final_summary = {
    'task': 'Dynamic Feature Selection for Cellular Trajectory Preservation',
    'dataset': 'RPE protein imaging (4i)',
    'n_cells': int(adata_raw.shape[0]),
    'n_total_features': int(adata_raw.shape[1]),
    'n_selected_features': 30,
    'selected_features': selected_30,
    'evaluation_metrics': results,
    'key_findings': {
        'top_proteins': protein_scores.head(10)['protein'].tolist(),
        'dpt_age_correlation_full': float(results_df[results_df['name'] == 'All Features (241)']['corr_dpt_age'].values[0]),
        'dpt_age_correlation_top30': float(results_df[results_df['name'] == 'Composite Top 30']['corr_dpt_age'].values[0]),
        'phase_silhouette_full': float(results_df[results_df['name'] == 'All Features (241)']['sil_phase'].values[0]),
        'phase_silhouette_top30': float(results_df[results_df['name'] == 'Composite Top 30']['sil_phase'].values[0]),
    }
}

with open('outputs/final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print("\n=== Phase 3 Complete ===")
print("All evaluation results and figures saved.")
