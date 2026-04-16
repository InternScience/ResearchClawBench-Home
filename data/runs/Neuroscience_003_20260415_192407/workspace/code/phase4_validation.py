"""
Phase 4: Validation & Comparison - Generate comprehensive comparison figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load results
df_greedy = pd.read_csv('outputs/greedy_forward_selection_results.csv')
df_dynamism = pd.read_csv('outputs/dynamism_topk_results.csv')
df_random = pd.read_csv('outputs/random_baseline_results.csv')
df_variance = pd.read_csv('outputs/variance_topk_results.csv')
df_metrics = pd.read_csv('outputs/feature_dynamism_metrics.csv')
optimal_sets = pd.read_csv('outputs/optimal_feature_sets.csv')

# Load data with pseudotime for final validation
adata_full = anndata.read_h5ad('outputs/adata_with_pseudotime.h5ad')
X_full = adata_full.X.copy()
age = adata_full.obs['annotated_age'].values
phase = adata_full.obs['phase'].values
batch = adata_full.obs['batch'].values
feature_names = adata_full.var_names.tolist()
ref_pseudotime = adata_full.obs['dpt_pseudotime'].values

ranked_features = df_metrics.sort_values('dynamism_score', ascending=False)['feature'].tolist()
feature_to_idx = {f: i for i, f in enumerate(feature_names)}

# === Figure 7: Comparison of selection strategies ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Spearman(age) vs n_features
ax = axes[0]
ax.plot(df_dynamism['n_features'], df_dynamism['spearman_age'], 'o-', color='steelblue', label='Dynamism Top-K', linewidth=2)
ax.plot(df_variance['n_features'], df_variance['spearman_age'], 's-', color='orange', label='Variance Top-K', linewidth=2)
ax.errorbar(df_random['n_features'], df_random['spearman_age_mean'], yerr=df_random['spearman_age_std'],
             fmt='^-', color='gray', label='Random (mean±std)', linewidth=1.5, capsize=3)
ax.axhline(y=0.5344, color='red', linestyle='--', alpha=0.7, label='Full 241 features (0.534)')
ax.set_xlabel('Number of Features')
ax.set_ylabel('Spearman ρ(Pseudotime, Age)')
ax.set_title('Trajectory-Age Correlation')
ax.legend(fontsize=9)
ax.set_xlim(0, 250)

# Spearman(ref) vs n_features
ax = axes[1]
ax.plot(df_dynamism['n_features'], df_dynamism['spearman_ref'], 'o-', color='steelblue', label='Dynamism Top-K', linewidth=2)
ax.plot(df_variance['n_features'], df_variance['spearman_ref'], 's-', color='orange', label='Variance Top-K', linewidth=2)
ax.errorbar(df_random['n_features'], df_random['spearman_ref_mean'], yerr=df_random['spearman_ref_std'],
             fmt='^-', color='gray', label='Random (mean±std)', linewidth=1.5, capsize=3)
ax.set_xlabel('Number of Features')
ax.set_ylabel('Spearman ρ(Pseudotime_subset, Pseudotime_full)')
ax.set_title('Trajectory Preservation')
ax.legend(fontsize=9)
ax.set_xlim(0, 250)

# Phase separation vs n_features
ax = axes[2]
ax.plot(df_dynamism['n_features'], df_dynamism['phase_separation'], 'o-', color='steelblue', label='Dynamism Top-K', linewidth=2)
ax.plot(df_variance['n_features'], df_variance['phase_separation'], 's-', color='orange', label='Variance Top-K', linewidth=2)
ax.errorbar(df_random['n_features'], df_random['phase_separation_mean'], yerr=df_random['phase_separation_std'],
             fmt='^-', color='gray', label='Random (mean±std)', linewidth=1.5, capsize=3)
ax.set_xlabel('Number of Features')
ax.set_ylabel('Phase Silhouette Score')
ax.set_title('Phase Separation Quality')
ax.legend(fontsize=9)
ax.set_xlim(0, 250)

plt.tight_layout()
plt.savefig('report/images/fig07_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig07_strategy_comparison.png")

# === Figure 8: Greedy forward selection trajectory ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(df_greedy['n_features'], df_greedy['spearman_age'], 'o-', color='darkgreen', linewidth=2)
ax.axhline(y=0.5344, color='red', linestyle='--', alpha=0.7, label='Full 241 features')
ax.set_xlabel('Number of Selected Features')
ax.set_ylabel('Spearman ρ(Pseudotime, Age)')
ax.set_title('Greedy Forward Selection: Trajectory-Age Correlation')
ax.legend()

ax = axes[1]
ax.plot(df_greedy['n_features'], df_greedy['spearman_ref'], 'o-', color='darkgreen', linewidth=2)
ax.set_xlabel('Number of Selected Features')
ax.set_ylabel('Spearman ρ(Pseudotime_subset, Pseudotime_full)')
ax.set_title('Greedy Forward Selection: Trajectory Preservation')

plt.tight_layout()
plt.savefig('report/images/fig08_greedy_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig08_greedy_selection.png")

# === Compute and visualize optimal feature subset trajectories ===

def compute_trajectory_on_subset(feature_indices, X_full, age, phase, batch, n_neighbors=30):
    """Compute full trajectory analysis on a feature subset."""
    X_subset = X_full[:, feature_indices]
    n_feat = len(feature_indices)
    
    temp = anndata.AnnData(X=X_subset)
    temp.obs['annotated_age'] = age
    temp.obs['phase'] = phase
    temp.obs['batch'] = batch
    
    n_pcs_use = min(n_feat - 1, 30) if n_feat > 2 else 1
    
    sc.pp.pca(temp, n_comps=n_pcs_use, random_state=42)
    sc.pp.neighbors(temp, n_neighbors=n_neighbors, n_pcs=min(n_pcs_use, n_feat-1), random_state=42)
    sc.tl.diffmap(temp, n_comps=min(n_pcs_use, 10), random_state=42)
    sc.tl.umap(temp, random_state=42)
    
    g1_cells_mask = temp.obs['phase'] == 'G1'
    youngest_idx = temp.obs[g1_cells_mask]['annotated_age'].idxmin()
    root_idx = temp.obs_names.get_loc(youngest_idx)
    temp.uns['iroot'] = root_idx
    sc.tl.dpt(temp, n_dcs=min(n_pcs_use, 10))
    
    return temp

# Optimal dynamism set
opt_dyn_n = int(optimal_sets[optimal_sets['method']=='dynamism_top_k']['optimal_n_features'].iloc[0])
opt_dyn_features = ranked_features[:opt_dyn_n]
opt_dyn_indices = [feature_to_idx[f] for f in opt_dyn_features]

print(f"Computing trajectory on optimal dynamism set ({opt_dyn_n} features)...")
adata_dyn = compute_trajectory_on_subset(opt_dyn_indices, X_full, age, phase, batch)

# Also compute on a smaller optimal set - let's use k=15 which showed good performance
small_k = 15
small_dyn_indices = [feature_to_idx[f] for f in ranked_features[:small_k]]
print(f"Computing trajectory on small dynamism set ({small_k} features)...")
adata_small = compute_trajectory_on_subset(small_dyn_indices, X_full, age, phase, batch)

# === Figure 9: UMAP comparison: Full vs Optimal vs Small ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Full features
sc.pl.umap(adata_full, color='phase', ax=axes[0,0], show=False, title=f'Full 241 Features - Phase')
sc.pl.umap(adata_full, color='dpt_pseudotime', ax=axes[0,1], show=False, title=f'Full 241 Features - Pseudotime', cmap='viridis')
sc.pl.umap(adata_full, color='annotated_age', ax=axes[0,2], show=False, title=f'Full 241 Features - Age', cmap='viridis')

# Optimal dynamism features
sc.pl.umap(adata_dyn, color='phase', ax=axes[1,0], show=False, title=f'Dynamism Top-{opt_dyn_n} - Phase')
sc.pl.umap(adata_dyn, color='dpt_pseudotime', ax=axes[1,1], show=False, title=f'Dynamism Top-{opt_dyn_n} - Pseudotime', cmap='viridis')
sc.pl.umap(adata_dyn, color='annotated_age', ax=axes[1,2], show=False, title=f'Dynamism Top-{opt_dyn_n} - Age', cmap='viridis')

plt.tight_layout()
plt.savefig('report/images/fig09_umap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig09_umap_comparison.png")

# === Figure 10: Pseudotime vs Age for different feature sets ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, adata_sub, title in zip(axes, 
    [adata_full, adata_dyn, adata_small],
    ['Full 241 Features', f'Dynamism Top-{opt_dyn_n}', f'Dynamism Top-{small_k}']):
    
    pt = adata_sub.obs['dpt_pseudotime'].values
    valid = ~np.isnan(pt)
    rho, p = spearmanr(pt[valid], age[valid])
    
    scatter = ax.scatter(age[valid], pt[valid], c=phase[valid].cat.codes, cmap='Set1', s=8, alpha=0.4)
    ax.set_xlabel('Annotated Age (hours)')
    ax.set_ylabel('Diffusion Pseudotime')
    ax.set_title(f'{title}\nSpearman ρ={rho:.3f}')
    
plt.tight_layout()
plt.savefig('report/images/fig10_pseudotime_age_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_pseudotime_age_comparison.png")

# === Figure 11: Feature category analysis ===
# Categorize features by compartment and protein type
categories = {
    'cell_level': [f for f in feature_names if '_cell' in f and '_cyto' not in f and '_nuc' not in f and '_ring' not in f],
    'cytoplasmic': [f for f in feature_names if '_cyto' in f],
    'nuclear': [f for f in feature_names if '_nuc' in f],
    'ring': [f for f in feature_names if '_ring' in f],
    'edge': [f for f in feature_names if 'MeanEdge' in f],
    'other': [f for f in feature_names if f not in [f2 for cat in categories.values() for f2 in cat]],
}

# Fix overlap: prioritize specific compartments
all_categorized = set()
for cat_name in ['cytoplasmic', 'nuclear', 'ring']:
    for f in categories[cat_name]:
        all_categorized.add(f)

categories['cell_level'] = [f for f in categories['cell_level'] if f not in all_categorized]
categories['edge'] = [f for f in categories['edge'] if f not in all_categorized]

# Count how many from each category are in the top 30
top30 = ranked_features[:30]
cat_counts = {}
for cat_name, cat_features in categories.items():
    cat_counts[cat_name] = sum(1 for f in top30 if f in cat_features)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart of top 30 feature categories
ax = axes[0]
labels = list(cat_counts.keys())
sizes = list(cat_counts.values())
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#af7aa1']
ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
ax.set_title('Category Distribution in Top 30 Features')

# Bar chart of dynamism scores by category
ax = axes[1]
cat_scores = {}
for cat_name, cat_features in categories.items():
    cat_metrics = df_metrics[df_metrics['feature'].isin(cat_features)]
    cat_scores[cat_name] = cat_metrics['dynamism_score'].mean()

bars = ax.bar(cat_scores.keys(), cat_scores.values(), color=colors[:len(cat_scores)])
ax.set_xlabel('Feature Category')
ax.set_ylabel('Mean Dynamism Score')
ax.set_title('Mean Dynamism Score by Feature Category')
ax.set_ylim(0, max(cat_scores.values()) * 1.2)

plt.tight_layout()
plt.savefig('report/images/fig11_feature_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_feature_categories.png")

# === Figure 12: Protein-level dynamism ranking ===
# Extract unique protein names
proteins = []
for f in feature_names:
    # Remove prefix and suffix
    name = f.replace('Int_MeanEdge_', '').replace('Int_Med_', '').replace('Int_Std_', '').replace('Int_Intg_', '').replace('AreaShape_', '')
    name = name.replace('_cell', '').replace('_cyto', '').replace('_nuc', '').replace('_ring', '')
    proteins.append(name)

unique_proteins = sorted(set(proteins))
print(f"Unique proteins: {len(unique_proteins)}")

# For each protein, compute max dynamism score across all its measurements
protein_max_scores = {}
protein_all_scores = {}
for protein in unique_proteins:
    protein_features = [f for f in feature_names if protein in f]
    protein_metrics = df_metrics[df_metrics['feature'].isin(protein_features)]
    protein_max_scores[protein] = protein_metrics['dynamism_score'].max()
    protein_all_scores[protein] = protein_metrics['dynamism_score'].tolist()

# Sort by max dynamism score
sorted_proteins = sorted(protein_max_scores.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(12, 10))
protein_names_sorted = [p[0] for p in sorted_proteins]
protein_scores_sorted = [p[1] for p in sorted_proteins]

bars = ax.barh(range(len(protein_names_sorted)), protein_scores_sorted[::-1], 
               color=['steelblue' if s > 0.5 else 'lightgray' for s in protein_scores_sorted[::-1]],
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(protein_names_sorted)))
ax.set_yticklabels(protein_names_sorted[::-1])
ax.set_xlabel('Max Dynamism Score (across compartments)')
ax.set_title('Protein-Level Dynamism Ranking')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold=0.5')
ax.legend()
plt.tight_layout()
plt.savefig('report/images/fig12_protein_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_protein_ranking.png")

# === Save comprehensive summary statistics ===
summary = {
    'full_features_spearman_age': float(spearmanr(adata_full.obs['dpt_pseudotime'][~np.isnan(adata_full.obs['dpt_pseudotime'])], age[~np.isnan(adata_full.obs['dpt_pseudotime'])])[0]),
    'dynamism_top_k_optimal_n': opt_dyn_n,
    'dynamism_top_k_best_spearman_age': float(df_dynamism['spearman_age'].max()),
    'dynamism_top_k_best_k': int(df_dynamism.loc[df_dynamism['spearman_age'].idxmax(), 'n_features']),
    'variance_best_spearman_age': float(df_variance['spearman_age'].max()),
    'random_mean_at_241': float(df_random[df_random['n_features']==241]['spearman_age_mean'].iloc[0]),
    'greedy_best_spearman_age': float(df_greedy['spearman_age'].max()),
    'greedy_best_n': int(df_greedy.loc[df_greedy['spearman_age'].idxmax(), 'n_features']),
    'n_dynamic_features_threshold_0.5': int(sum(1 for s in df_metrics['dynamism_score'] if s > 0.5)),
    'n_unique_proteins': len(unique_proteins),
}

pd.DataFrame([summary]).to_json('outputs/summary_statistics.json')
print("\nSummary statistics:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Save the selected feature subset data
selected_features_final = ranked_features[:15]  # Use top 15 as practical optimal
selected_indices_final = [feature_to_idx[f] for f in selected_features_final]

adata_selected = anndata.AnnData(X=X_full[:, selected_indices_final])
adata_selected.obs = adata_full.obs.copy()
adata_selected.var_names = selected_features_final
adata_selected.write('outputs/adata_selected_features.h5ad')

# Compute trajectory on selected features
sc.pp.pca(adata_selected, n_comps=min(len(selected_features_final)-1, 30), random_state=42)
sc.pp.neighbors(adata_selected, n_neighbors=30, n_pcs=min(len(selected_features_final)-1, 30), random_state=42)
sc.tl.diffmap(adata_selected, random_state=42)
sc.tl.umap(adata_selected, random_state=42)

g1_mask = adata_selected.obs['phase'] == 'G1'
youngest_idx = adata_selected.obs[g1_mask]['annotated_age'].idxmin()
adata_selected.uns['iroot'] = adata_selected.obs_names.get_loc(youngest_idx)
sc.tl.dpt(adata_selected, n_dcs=10)

rho_selected, _ = spearmanr(adata_selected.obs['dpt_pseudotime'][~np.isnan(adata_selected.obs['dpt_pseudotime'])], 
                              age[~np.isnan(adata_selected.obs['dpt_pseudotime'])])
print(f"\nFinal selected {len(selected_features_final)} features: Spearman ρ(pseudotime, age) = {rho_selected:.4f}")
print(f"Selected features: {selected_features_final}")

# Save final selected features list
pd.DataFrame({
    'rank': range(1, len(selected_features_final)+1),
    'feature': selected_features_final,
    'dynamism_score': [df_metrics[df_metrics['feature']==f]['dynamism_score'].iloc[0] for f in selected_features_final],
}).to_csv('outputs/final_selected_features.csv', index=False)

print("\nPhase 4 complete!")