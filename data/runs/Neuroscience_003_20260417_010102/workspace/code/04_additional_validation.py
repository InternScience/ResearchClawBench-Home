#!/usr/bin/env python3
"""
Phase 3b: Additional validation and figures
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
sns.set_style('whitegrid')

# Load data
adata_raw = sc.read_h5ad('data/adata_RPE.h5ad')
X_full = adata_raw.X.copy()
feature_names = adata_raw.var_names.tolist()
ages = adata_raw.obs['annotated_age'].values
phases = adata_raw.obs['phase'].values
states = adata_raw.obs['state'].values
batches = adata_raw.obs['batch'].values

with open('outputs/selected_features.json', 'r') as f:
    selected = json.load(f)
selected_30 = selected['top_30']

scores_df = pd.read_csv('outputs/feature_scores_annotated.csv')
results_df = pd.read_csv('outputs/evaluation_results.csv')

# ============================================================
# Figure 9: Robustness analysis - varying number of selected features
# ============================================================
print("=== Figure 9: Feature count sweep ===")

from anndata import AnnData

def quick_dpt_age_corr(X_sub, ages, phases):
    """Quick evaluation: compute DPT-age correlation."""
    adata_sub = AnnData(X=X_sub.copy())
    adata_sub.obs['annotated_age'] = ages
    adata_sub.obs['phase'] = pd.Categorical(phases)
    
    sc.pp.scale(adata_sub, max_value=10)
    n_comps = min(50, X_sub.shape[1] - 1)
    sc.tl.pca(adata_sub, n_comps=n_comps)
    n_pcs_use = min(30, n_comps)
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=n_pcs_use)
    n_dcs = min(15, n_comps, X_sub.shape[1] - 1)
    if n_dcs < 2:
        return np.nan
    sc.tl.diffmap(adata_sub, n_comps=n_dcs)
    
    g1_mask = adata_sub.obs['phase'] == 'G1'
    if g1_mask.sum() > 0:
        youngest_g1 = adata_sub.obs.loc[g1_mask, 'annotated_age'].idxmin()
        root_idx = adata_sub.obs.index.get_loc(youngest_g1)
    else:
        root_idx = 0
    adata_sub.uns['iroot'] = root_idx
    try:
        sc.tl.dpt(adata_sub, n_dcs=min(10, n_dcs))
    except Exception:
        return np.nan
    
    dpt = adata_sub.obs['dpt_pseudotime'].values
    corr, _ = stats.spearmanr(dpt, ages)
    return corr

# Sweep over feature counts
feature_counts = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 241]
composite_corrs = []
variance_corrs = []
random_corrs = []

sorted_composite = scores_df.sort_values('composite_score', ascending=False)['feature'].tolist()
sorted_variance = scores_df.sort_values('variance', ascending=False)['feature'].tolist()

np.random.seed(42)

for n in feature_counts:
    print(f"  n={n}...")
    
    # Composite selection
    feats = sorted_composite[:min(n, len(sorted_composite))]
    idxs = [feature_names.index(f) for f in feats if f in feature_names]
    if len(idxs) >= 2:
        corr = quick_dpt_age_corr(X_full[:, idxs], ages, phases)
        composite_corrs.append(corr)
    else:
        composite_corrs.append(np.nan)
    
    # Variance selection
    feats_v = sorted_variance[:min(n, len(sorted_variance))]
    idxs_v = [feature_names.index(f) for f in feats_v if f in feature_names]
    if len(idxs_v) >= 2:
        corr_v = quick_dpt_age_corr(X_full[:, idxs_v], ages, phases)
        variance_corrs.append(corr_v)
    else:
        variance_corrs.append(np.nan)
    
    # Random selection (average of 3 runs)
    rand_corrs_runs = []
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        rand_idxs = np.random.choice(len(feature_names), min(n, len(feature_names)), replace=False)
        if len(rand_idxs) >= 2:
            corr_r = quick_dpt_age_corr(X_full[:, rand_idxs], ages, phases)
            rand_corrs_runs.append(corr_r)
    random_corrs.append(np.mean(rand_corrs_runs) if rand_corrs_runs else np.nan)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(feature_counts, composite_corrs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Composite Selection')
ax.plot(feature_counts, variance_corrs, 's-', color='#4CAF50', linewidth=2, markersize=8, label='Variance Selection')
ax.plot(feature_counts, random_corrs, '^-', color='#F44336', linewidth=2, markersize=8, label='Random Selection')
ax.set_xlabel('Number of Selected Features')
ax.set_ylabel('DPT-Age Spearman Correlation')
ax.set_title('Trajectory Preservation vs. Number of Features')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/fig9_feature_count_sweep.png', bbox_inches='tight')
plt.close()
print("Figure 9 saved.")

# Save sweep results
sweep_df = pd.DataFrame({
    'n_features': feature_counts,
    'composite_corr': composite_corrs,
    'variance_corr': variance_corrs,
    'random_corr': random_corrs,
})
sweep_df.to_csv('outputs/feature_count_sweep.csv', index=False)

# ============================================================
# Figure 10: Correlation matrix of top features
# ============================================================
print("\n=== Figure 10: Correlation matrix ===")

idxs_30 = [feature_names.index(f) for f in selected_30]
X_sel = X_full[:, idxs_30]
corr_matrix = np.corrcoef(X_sel.T)

# Shorten feature names
short_names = []
for f in selected_30:
    name = f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
    short_names.append(name)

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, xticklabels=short_names, yticklabels=short_names,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5, annot=False)
ax.set_title('Correlation Matrix of Top 30 Selected Features')
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7)

plt.tight_layout()
plt.savefig('report/images/fig10_correlation_matrix.png', bbox_inches='tight')
plt.close()
print("Figure 10 saved.")

# ============================================================
# Figure 11: Phase-specific expression of top features
# ============================================================
print("\n=== Figure 11: Phase-specific expression ===")

# Select top 8 features for detailed phase analysis
top_8 = selected_30[:8]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, feat in enumerate(top_8):
    ax = axes[idx // 4, idx % 4]
    feat_idx = feature_names.index(feat)
    
    data = pd.DataFrame({
        'expression': X_full[:, feat_idx],
        'phase': phases,
        'age': ages,
    })
    
    # Phase order
    phase_order = ['G1', 'S', 'G2', 'G0']
    sns.violinplot(data=data, x='phase', y='expression', order=phase_order,
                   ax=ax, palette='Set2', inner='box')
    
    short_name = feat.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
    ax.set_title(short_name, fontsize=11)
    ax.set_xlabel('')
    if idx % 4 == 0:
        ax.set_ylabel('Expression')
    else:
        ax.set_ylabel('')

plt.suptitle('Phase-Specific Expression of Top 8 Selected Features', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig11_phase_expression.png', bbox_inches='tight')
plt.close()
print("Figure 11 saved.")

# ============================================================
# Figure 12: State transition analysis
# ============================================================
print("\n=== Figure 12: State transition ===")

# Compare cycling vs arrested for top features
state_mask = states != 'nan'
X_state = X_full[state_mask]
states_clean = states[state_mask]
ages_state = ages[state_mask]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, feat in enumerate(top_8):
    ax = axes[idx // 4, idx % 4]
    feat_idx = feature_names.index(feat)
    
    cycling = X_state[states_clean == 'cycling', feat_idx]
    arrested = X_state[states_clean == 'arrested', feat_idx]
    cycling_ages = ages_state[states_clean == 'cycling']
    arrested_ages = ages_state[states_clean == 'arrested']
    
    ax.scatter(cycling_ages, cycling, s=3, alpha=0.3, c='#2196F3', label='Cycling')
    ax.scatter(arrested_ages, arrested, s=3, alpha=0.3, c='#F44336', label='Arrested')
    
    short_name = feat.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
    ax.set_title(short_name, fontsize=11)
    ax.set_xlabel('Age')
    if idx % 4 == 0:
        ax.set_ylabel('Expression')
    ax.legend(fontsize=7, markerscale=3)

plt.suptitle('Cycling vs. Arrested State: Top 8 Features', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig12_state_transition.png', bbox_inches='tight')
plt.close()
print("Figure 12 saved.")

print("\n=== All additional figures complete ===")
