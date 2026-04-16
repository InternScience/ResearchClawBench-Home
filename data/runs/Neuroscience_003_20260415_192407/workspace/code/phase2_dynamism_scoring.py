"""
Phase 2: Feature Dynamism Scoring
- Compute per-feature metrics along pseudotime:
  1. Variance explained by pseudotime bins (trajectory variance)
  2. Correlation with pseudotime (linear and monotonic)
  3. Autocorrelation in trajectory order
  4. Batch effect magnitude (how much feature varies by batch)
  5. Overall variance
- Compute composite dynamism score
- Rank features and visualize
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata
from scipy.stats import spearmanr, pearsonr, f_oneway
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load data with pseudotime
adata = anndata.read_h5ad('outputs/adata_with_pseudotime.h5ad')
print(f"Dataset shape: {adata.shape}")

# Get feature matrix and pseudotime
X = adata.X.copy()
pseudotime = adata.obs['dpt_pseudotime'].values
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].values
batch = adata.obs['batch'].values
feature_names = adata.var_names.tolist()

# Filter cells with valid pseudotime (not NaN)
valid_mask = ~np.isnan(pseudotime)
X_valid = X[valid_mask]
pt_valid = pseudotime[valid_mask]
age_valid = age[valid_mask]
phase_valid = phase[valid_mask]
batch_valid = batch[valid_mask]

print(f"Valid cells: {valid_mask.sum()}, NaN cells: {np.isnan(pseudotime).sum()}")

# Sort cells by pseudotime for trajectory-ordered analysis
sort_idx = np.argsort(pt_valid)
X_sorted = X_valid[sort_idx]
pt_sorted = pt_valid[sort_idx]

n_features = X_valid.shape[1]
n_cells = X_valid.shape[0]

# === Compute feature-level metrics ===

results = []

for i, fname in enumerate(feature_names):
    feat_vals = X_valid[:, i]
    feat_sorted = X_sorted[:, i]
    
    # 1. Spearman correlation with pseudotime
    rho_pt, p_pt = spearmanr(pt_valid, feat_vals)
    
    # 2. Pearson correlation with pseudotime
    r_pt, p_r = pearsonr(pt_valid, feat_vals)
    
    # 3. Spearman correlation with annotated age
    rho_age, p_age = spearmanr(age_valid, feat_vals)
    
    # 4. Overall variance
    overall_var = np.var(feat_vals)
    
    # 5. Trajectory variance: variance of smoothed values along pseudotime
    # Smooth the feature along sorted pseudotime using moving average
    window = max(10, n_cells // 50)
    smoothed = uniform_filter1d(feat_sorted, size=window)
    trajectory_var = np.var(smoothed)
    
    # 6. Dynamic signal ratio: trajectory_var / overall_var
    # High ratio means most variance is along the trajectory (structured), not noise
    dynamic_ratio = trajectory_var / overall_var if overall_var > 0 else 0
    
    # 7. Autocorrelation in trajectory order (lag=1)
    autocorr = np.corrcoef(feat_sorted[:-1], feat_sorted[1:])[0, 1] if np.std(feat_sorted) > 0 else 0
    
    # 8. Phase discrimination: ANOVA F-statistic across cell cycle phases
    groups = [feat_vals[phase_valid == ph] for ph in ['G0', 'G1', 'S', 'G2']]
    groups = [g for g in groups if len(g) > 10]  # filter small groups
    if len(groups) >= 2:
        f_stat, p_anova = f_oneway(*groups)
    else:
        f_stat, p_anova = 0, 1
    
    # 9. Batch effect: ANOVA F-statistic across batches
    batch_groups = [feat_vals[batch_valid == b] for b in ['1', '2']]
    if all(len(g) > 10 for g in batch_groups):
        f_batch, p_batch = f_oneway(*batch_groups)
    else:
        f_batch, p_batch = 0, 1
    
    # 10. Batch-to-trajectory ratio (confounding measure)
    # Higher means more batch effect relative to trajectory signal
    batch_confound_ratio = f_batch / f_stat if f_stat > 0 else float('inf')
    
    results.append({
        'feature': fname,
        'spearman_pt': rho_pt,
        'pearson_pt': r_pt,
        'spearman_age': rho_age,
        'overall_var': overall_var,
        'trajectory_var': trajectory_var,
        'dynamic_ratio': dynamic_ratio,
        'autocorrelation': autocorr,
        'f_stat_phase': f_stat,
        'p_anova': p_anova,
        'f_stat_batch': f_batch,
        'p_batch': p_batch,
        'batch_confound_ratio': batch_confound_ratio,
    })

df_metrics = pd.DataFrame(results)

# === Composite Dynamism Score ===
# Weighted combination of trajectory-relevant metrics, penalized by batch confounding
# Normalize each metric to [0, 1] range first
from scipy.stats import rankdata

def normalize_to_01(arr):
    """Rank-based normalization to [0, 1]"""
    ranks = rankdata(arr)
    return (ranks - 1) / (len(ranks) - 1)

# Positive signals (higher = more dynamic)
norm_spearman_pt = normalize_to_01(np.abs(df_metrics['spearman_pt'].values))
norm_dynamic_ratio = normalize_to_01(df_metrics['dynamic_ratio'].values)
norm_autocorr = normalize_to_01(df_metrics['autocorrelation'].values)
norm_f_phase = normalize_to_01(df_metrics['f_stat_phase'].values)
norm_trajectory_var = normalize_to_01(df_metrics['trajectory_var'].values)

# Negative signal (higher = more confounding)
norm_batch_confound = normalize_to_01(df_metrics['batch_confound_ratio'].values)

# Composite score: weighted sum of positive signals minus batch confounding penalty
weights = {
    'spearman_pt': 0.25,
    'dynamic_ratio': 0.20,
    'autocorr': 0.15,
    'f_phase': 0.20,
    'trajectory_var': 0.10,
    'batch_confound_penalty': 0.10,
}

df_metrics['dynamism_score'] = (
    weights['spearman_pt'] * norm_spearman_pt +
    weights['dynamic_ratio'] * norm_dynamic_ratio +
    weights['autocorr'] * norm_autocorr +
    weights['f_phase'] * norm_f_phase +
    weights['trajectory_var'] * norm_trajectory_var -
    weights['batch_confound_penalty'] * norm_batch_confound
)

# Sort by dynamism score
df_metrics = df_metrics.sort_values('dynamism_score', ascending=False).reset_index(drop=True)

# Save metrics
df_metrics.to_csv('outputs/feature_dynamism_metrics.csv', index=False)
print(f"Feature metrics computed and saved")
print(f"\nTop 20 features by dynamism score:")
print(df_metrics.head(20)[['feature', 'dynamism_score', 'spearman_pt', 'dynamic_ratio', 'f_stat_phase', 'batch_confound_ratio']].to_string())

# === Visualization ===

# Figure 4: Dynamism score distribution and top features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Dynamism score histogram
axes[0,0].hist(df_metrics['dynamism_score'], bins=30, edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Dynamism Score')
axes[0,0].set_ylabel('Number of Features')
axes[0,0].set_title('Distribution of Dynamism Scores')
axes[0,0].axvline(x=df_metrics['dynamism_score'].median(), color='red', linestyle='--', label='Median')
axes[0,0].legend()

# Top 30 features bar plot
top30 = df_metrics.head(30)
# Shorten feature names for display
short_names = [n.replace('Int_MeanEdge_', 'ME_').replace('Int_Med_', 'M_').replace('_cell', '').replace('_cyto', '.cy').replace('_nuc', '.nu').replace('_ring', '.ri') for n in top30['feature']]
axes[0,1].barh(range(30), top30['dynamism_score'].values[::-1], color='steelblue', edgecolor='black')
axes[0,1].set_yticks(range(30))
axes[0,1].set_yticklabels(short_names[::-1])
axes[0,1].set_xlabel('Dynamism Score')
axes[0,1].set_title('Top 30 Features by Dynamism Score')
axes[0,1].invert_yaxis()

# Scatter: spearman_pt vs batch_confound_ratio
axes[1,0].scatter(df_metrics['spearman_pt'].abs(), df_metrics['batch_confound_ratio'], 
                  c=df_metrics['dynamism_score'], cmap='RdYlBu_r', s=20, alpha=0.7)
axes[1,0].set_xlabel('|Spearman ρ with Pseudotime|')
axes[1,0].set_ylabel('Batch Confound Ratio')
axes[1,0].set_title('Trajectory Signal vs Batch Confounding')
axes[1,0].set_ylim(0, min(5, df_metrics['batch_confound_ratio'].quantile(0.95)*1.5))

# Scatter: dynamic_ratio vs f_stat_phase
axes[1,1].scatter(df_metrics['dynamic_ratio'], df_metrics['f_stat_phase'],
                  c=df_metrics['dynamism_score'], cmap='RdYlBu_r', s=20, alpha=0.7)
axes[1,1].set_xlabel('Dynamic Signal Ratio')
axes[1,1].set_ylabel('Phase Discrimination (F-stat)')
axes[1,1].set_title('Dynamic Structure vs Phase Discrimination')

plt.tight_layout()
plt.savefig('report/images/fig04_dynamism_scoring.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig04_dynamism_scoring.png")

# Figure 5: Feature expression along pseudotime for top features
top_features = df_metrics.head(12)['feature'].tolist()
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for idx, fname in enumerate(top_features):
    ax = axes[idx // 4, idx % 4]
    feat_idx = feature_names.index(fname)
    feat_sorted_vals = X_sorted[:, feat_idx]
    ax.scatter(pt_sorted, feat_sorted_vals, s=2, alpha=0.3, c='steelblue')
    # Add smoothed line
    window = max(10, n_cells // 50)
    smoothed = uniform_filter1d(feat_sorted_vals, size=window)
    ax.plot(pt_sorted, smoothed, color='red', linewidth=2)
    short_name = fname.replace('Int_MeanEdge_', 'ME_').replace('Int_Med_', 'M_')
    ax.set_title(f'{short_name}\n(score={df_metrics.iloc[idx]["dynamism_score"]:.3f})')
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Expression')
plt.tight_layout()
plt.savefig('report/images/fig05_top_features_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig05_top_features_trajectory.png")

# Figure 6: Bottom features (least dynamic) along pseudotime
bottom_features = df_metrics.tail(12)['feature'].tolist()
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for idx, fname in enumerate(bottom_features):
    ax = axes[idx // 4, idx % 4]
    feat_idx = feature_names.index(fname)
    feat_sorted_vals = X_sorted[:, feat_idx]
    ax.scatter(pt_sorted, feat_sorted_vals, s=2, alpha=0.3, c='gray')
    window = max(10, n_cells // 50)
    smoothed = uniform_filter1d(feat_sorted_vals, size=window)
    ax.plot(pt_sorted, smoothed, color='darkred', linewidth=2)
    short_name = fname.replace('Int_MeanEdge_', 'ME_').replace('Int_Med_', 'M_')
    row_idx = len(df_metrics) - 12 + idx
    ax.set_title(f'{short_name}\n(score={df_metrics.iloc[row_idx]["dynamism_score"]:.3f})')
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Expression')
plt.tight_layout()
plt.savefig('report/images/fig06_bottom_features_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig06_bottom_features_trajectory.png")

print("\nPhase 2 complete!")