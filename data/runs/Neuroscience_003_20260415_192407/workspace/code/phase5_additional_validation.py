"""
Phase 5: Additional validation - SHAP/interpretability analysis and batch effect comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
from scipy.stats import spearmanr, pearsonr, f_oneway
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load data
adata_full = anndata.read_h5ad('outputs/adata_with_pseudotime.h5ad')
df_metrics = pd.read_csv('outputs/feature_dynamism_metrics.csv')

X_full = adata_full.X.copy()
age = adata_full.obs['annotated_age'].values
phase = adata_full.obs['phase'].values
batch = adata_full.obs['batch'].values
pseudotime = adata_full.obs['dpt_pseudotime'].values
feature_names = adata_full.var_names.tolist()

ranked_features = df_metrics.sort_values('dynamism_score', ascending=False)['feature'].tolist()
feature_to_idx = {f: i for i, f in enumerate(feature_names)}

# === Permutation Importance for predicting pseudotime ===
print("Computing permutation importance for pseudotime prediction...")

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_full, pseudotime)

perm_imp = permutation_importance(rf, X_full, pseudotime, n_repeats=10, random_state=42, n_jobs=-1)

perm_importances = perm_imp.importances_mean
perm_std = perm_imp.importances_std

# Sort by importance
perm_order = np.argsort(perm_importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 30 features by permutation importance
ax = axes[0]
top30_perm = perm_order[:30]
short_names = [feature_names[i].replace('Int_MeanEdge_', 'ME_').replace('Int_Med_', 'M_').replace('_cell', '').replace('_cyto', '.cy').replace('_nuc', '.nu').replace('_ring', '.ri') for i in top30_perm]
ax.barh(range(30), perm_importances[top30_perm][::-1], 
        xerr=perm_std[top30_perm][::-1],
        color='steelblue', edgecolor='black', capsize=3)
ax.set_yticks(range(30))
ax.set_yticklabels(short_names[::-1])
ax.set_xlabel('Permutation Importance (mean ± std)')
ax.set_title('Top 30 Features by Permutation Importance\n(Random Forest predicting pseudotime)')
ax.invert_yaxis()

# Compare dynamism ranking vs permutation importance ranking
ax = axes[1]
dyn_ranks = {f: i for i, f in enumerate(ranked_features)}
perm_ranks = {feature_names[i]: rank for rank, i in enumerate(perm_order)}

common_features = ranked_features[:50]
dyn_rank_vals = [dyn_ranks[f] for f in common_features]
perm_rank_vals = [perm_ranks[f] for f in common_features]

ax.scatter(dyn_rank_vals, perm_rank_vals, s=30, alpha=0.7, color='steelblue')
rho, _ = spearmanr(dyn_rank_vals, perm_rank_vals)
ax.set_xlabel('Dynamism Score Rank')
ax.set_ylabel('Permutation Importance Rank')
ax.set_title(f'Rank Correlation: Dynamism vs Permutation Importance\nSpearman ρ={rho:.3f}')
ax.plot([0, 50], [0, 50], 'r--', alpha=0.5, label='Perfect agreement')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/fig13_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13_permutation_importance.png")

# === Figure 14: Batch effect analysis for selected vs all features ===
print("Analyzing batch effects...")

selected_features = ranked_features[:15]
selected_indices = [feature_to_idx[f] for f in selected_features]

# Compute batch effect for each feature
batch_f_stats = {}
for i, fname in enumerate(feature_names):
    feat_vals = X_full[:, i]
    groups = [feat_vals[batch == b] for b in ['1', '2']]
    f_stat, p_val = f_oneway(*groups)
    batch_f_stats[fname] = {'f_stat': f_stat, 'p_val': p_val}

# Compare batch F-statistics between selected and non-selected features
selected_batch_f = [batch_f_stats[f]['f_stat'] for f in selected_features]
all_batch_f = [batch_f_stats[f]['f_stat'] for f in feature_names]
non_selected_batch_f = [batch_f_stats[f]['f_stat'] for f in feature_names if f not in selected_features]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.hist(all_batch_f, bins=50, alpha=0.5, color='gray', label='All 241 features', edgecolor='black')
ax.hist(selected_batch_f, bins=20, alpha=0.7, color='steelblue', label=f'Selected {len(selected_features)} features', edgecolor='black')
ax.set_xlabel('Batch Effect (ANOVA F-statistic)')
ax.set_ylabel('Number of Features')
ax.set_title('Batch Effect Distribution')
ax.legend()

ax = axes[1]
ax.bar(['Selected (15)', 'Non-selected (226)', 'All (241)'],
       [np.mean(selected_batch_f), np.mean(non_selected_batch_f), np.mean(all_batch_f)],
       color=['steelblue', 'lightgray', 'darkgray'], edgecolor='black')
ax.set_ylabel('Mean Batch F-statistic')
ax.set_title('Mean Batch Effect Comparison')
ax.set_ylim(0, max(np.mean(all_batch_f), np.mean(selected_batch_f)) * 1.5)

plt.tight_layout()
plt.savefig('report/images/fig14_batch_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig14_batch_effect.png")

# === Figure 15: Heatmap of selected features across cell cycle phases ===
print("Creating heatmap of selected features...")

# Compute mean expression per phase for selected features
phase_means = {}
for ph in ['G0', 'G1', 'S', 'G2']:
    mask = phase == ph
    phase_means[ph] = X_full[mask, :][:, selected_indices].mean(axis=0)

heatmap_data = pd.DataFrame(phase_means, index=selected_features)
# Normalize each feature to its range for visualization
heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

short_names_heat = [f.replace('Int_MeanEdge_', 'ME_').replace('Int_Med_', 'M_').replace('_cell', '.c').replace('_cyto', '.cy').replace('_nuc', '.nu').replace('_ring', '.ri') for f in selected_features]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlBu_r',
            yticklabels=short_names_heat, xticklabels=['G0', 'G1', 'S', 'G2'],
            ax=ax, linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Normalized Mean Expression of Selected Features Across Cell Cycle Phases')
ax.set_ylabel('Feature')
ax.set_xlabel('Cell Cycle Phase')

plt.tight_layout()
plt.savefig('report/images/fig15_phase_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig15_phase_heatmap.png")

# === Figure 16: Trajectory smoothness comparison ===
print("Computing trajectory smoothness metrics...")

from scipy.ndimage import uniform_filter1d

def compute_trajectory_smoothness(pseudotime, feature_values, window=50):
    """Compute smoothness of feature along pseudotime-ordered cells."""
    sort_idx = np.argsort(pseudotime)
    sorted_vals = feature_values[sort_idx]
    
    # Smooth the trajectory
    smoothed = uniform_filter1d(sorted_vals, size=window)
    
    # Residual variance (noise) vs smoothed variance (signal)
    residual_var = np.var(sorted_vals - smoothed)
    signal_var = np.var(smoothed)
    
    # Signal-to-noise ratio
    snr = signal_var / residual_var if residual_var > 0 else float('inf')
    return snr

# Compare SNR for selected vs all features
selected_snrs = []
all_snrs = []

valid = ~np.isnan(pseudotime)
pt_valid = pseudotime[valid]

for i, fname in enumerate(feature_names):
    snr = compute_trajectory_smoothness(pt_valid, X_full[valid, i])
    all_snrs.append(snr)
    if fname in selected_features:
        selected_snrs.append(snr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.hist(all_snrs, bins=50, alpha=0.5, color='gray', label='All features', edgecolor='black')
ax.hist(selected_snrs, bins=20, alpha=0.7, color='steelblue', label='Selected features', edgecolor='black')
ax.set_xlabel('Signal-to-Noise Ratio (along pseudotime)')
ax.set_ylabel('Number of Features')
ax.set_title('Trajectory Signal-to-Noise Ratio')
ax.legend()

ax = axes[1]
ax.bar(['Selected (15)', 'All (241)'],
       [np.mean(selected_snrs), np.mean(all_snrs)],
       color=['steelblue', 'gray'], edgecolor='black')
ax.set_ylabel('Mean SNR')
ax.set_title('Mean Trajectory SNR Comparison')

plt.tight_layout()
plt.savefig('report/images/fig16_trajectory_snr.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig16_trajectory_snr.png")

# Save final comprehensive results
final_results = {
    'selected_features': selected_features,
    'n_selected': len(selected_features),
    'n_total': len(feature_names),
    'reduction_ratio': len(selected_features) / len(feature_names),
    'selected_mean_batch_f': np.mean(selected_batch_f),
    'all_mean_batch_f': np.mean(all_batch_f),
    'selected_mean_snr': np.mean(selected_snrs),
    'all_mean_snr': np.mean(all_snrs),
    'dynamism_vs_permutation_rank_correlation': rho,
}

pd.DataFrame([final_results]).to_json('outputs/final_results_summary.json')
print("\nFinal results summary:")
for k, v in final_results.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v}")
    elif isinstance(v, list):
        print(f"  {k}: {len(v)} items")

print("\nPhase 5 complete!")