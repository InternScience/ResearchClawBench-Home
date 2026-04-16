"""
Phase 3: Trajectory-Preserving Feature Selection
- Implement greedy forward selection based on trajectory preservation metric
- Compare with variance-based filtering and random baselines
- For each feature subset size k, compute trajectory quality metrics
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

# Load data with pseudotime
adata_full = anndata.read_h5ad('outputs/adata_with_pseudotime.h5ad')
df_metrics = pd.read_csv('outputs/feature_dynamism_metrics.csv')

X_full = adata_full.X.copy()
age = adata_full.obs['annotated_age'].values
phase = adata_full.obs['phase'].values
batch = adata_full.obs['batch'].values
feature_names = adata_full.var_names.tolist()

# Reference pseudotime from full features
ref_pseudotime = adata_full.obs['dpt_pseudotime'].values

def compute_trajectory_quality(feature_indices, X_full, age, phase, ref_pseudotime, n_neighbors=30):
    """
    Compute trajectory quality metrics for a given feature subset.
    """
    X_subset = X_full[:, feature_indices]
    n_feat = len(feature_indices)
    
    temp = anndata.AnnData(X=X_subset)
    temp.obs['annotated_age'] = age
    temp.obs['phase'] = phase
    temp.obs['batch'] = batch
    
    n_pcs_use = min(n_feat - 1, 30) if n_feat > 2 else 1
    if n_feat < 2:
        return {'spearman_age': 0, 'spearman_ref': 0, 'phase_separation': 0, 'graph_connectivity': 0}
    
    try:
        sc.pp.pca(temp, n_comps=n_pcs_use, random_state=42)
        n_pcs_nn = min(n_pcs_use, n_feat - 1)
        sc.pp.neighbors(temp, n_neighbors=n_neighbors, n_pcs=n_pcs_nn, random_state=42)
        n_dcs = min(n_pcs_use, 10)
        sc.tl.diffmap(temp, n_comps=n_dcs, random_state=42)
    except Exception as e:
        return {'spearman_age': 0, 'spearman_ref': 0, 'phase_separation': 0, 'graph_connectivity': 0}
    
    g1_cells_mask = temp.obs['phase'] == 'G1'
    if g1_cells_mask.sum() > 0:
        youngest_idx = temp.obs[g1_cells_mask]['annotated_age'].idxmin()
        root_idx = temp.obs_names.get_loc(youngest_idx)
    else:
        root_idx = 0
    
    temp.uns['iroot'] = root_idx
    
    try:
        sc.tl.dpt(temp, n_dcs=min(n_dcs, n_pcs_use))
    except:
        return {'spearman_age': 0, 'spearman_ref': 0, 'phase_separation': 0, 'graph_connectivity': 0}
    
    pt_subset = temp.obs['dpt_pseudotime'].values
    valid = ~np.isnan(pt_subset)
    if valid.sum() < 50:
        return {'spearman_age': 0, 'spearman_ref': 0, 'phase_separation': 0, 'graph_connectivity': 0}
    
    rho_age, _ = spearmanr(pt_subset[valid], age[valid])
    rho_ref, _ = spearmanr(pt_subset[valid], ref_pseudotime[valid])
    
    try:
        n_sil_pcs = min(5, n_pcs_use)
        sil = silhouette_score(temp.obsm['X_pca'][:, :n_sil_pcs], phase[valid],
                               sample_size=min(1000, valid.sum()))
    except:
        sil = 0
    
    connectivity = valid.sum() / len(pt_subset)
    
    return {
        'spearman_age': rho_age,
        'spearman_ref': rho_ref,
        'phase_separation': sil,
        'graph_connectivity': connectivity,
    }


# === Strategy 1: Greedy Forward Selection ===
print("Starting greedy forward selection...")

ranked_features = df_metrics.sort_values('dynamism_score', ascending=False)['feature'].tolist()
feature_to_idx = {f: i for i, f in enumerate(feature_names)}

initial_features = [feature_to_idx[f] for f in ranked_features[:5]]
initial_quality = compute_trajectory_quality(initial_features, X_full, age, phase, ref_pseudotime)
print(f"Initial 5 features: spearman_age={initial_quality['spearman_age']:.4f}")

selected_indices = list(initial_features)
selected_features_greedy = [ranked_features[i] for i in range(5)]
greedy_results = []

greedy_results.append({
    'n_features': 5,
    'spearman_age': initial_quality['spearman_age'],
    'spearman_ref': initial_quality['spearman_ref'],
    'phase_separation': initial_quality['phase_separation'],
    'features': ';'.join(selected_features_greedy),
})

candidate_pool = ranked_features[5:50]
remaining_indices = [feature_to_idx[f] for f in candidate_pool]

for step in range(min(45, len(candidate_pool))):
    best_quality = None
    best_feat_idx = None
    best_feat_name = None
    
    for feat_idx, feat_name in zip(remaining_indices, candidate_pool):
        test_indices = selected_indices + [feat_idx]
        quality = compute_trajectory_quality(test_indices, X_full, age, phase, ref_pseudotime)
        
        if best_quality is None or quality['spearman_age'] > best_quality['spearman_age']:
            best_quality = quality
            best_feat_idx = feat_idx
            best_feat_name = feat_name
    
    if best_feat_idx is not None:
        selected_indices.append(best_feat_idx)
        selected_features_greedy.append(best_feat_name)
        remaining_indices.remove(best_feat_idx)
        
        greedy_results.append({
            'n_features': 5 + step + 1,
            'spearman_age': best_quality['spearman_age'],
            'spearman_ref': best_quality['spearman_ref'],
            'phase_separation': best_quality['phase_separation'],
            'features': ';'.join(selected_features_greedy),
        })
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}: added {best_feat_name}, "
                  f"spearman_age={best_quality['spearman_age']:.4f}, "
                  f"spearman_ref={best_quality['spearman_ref']:.4f}, "
                  f"total features={len(selected_indices)}")

df_greedy = pd.DataFrame(greedy_results)
df_greedy.to_csv('outputs/greedy_forward_selection_results.csv', index=False)
print("Greedy forward selection complete!")

# === Strategy 2: Dynamism-score-based top-k selection ===
print("\nComputing dynamism-based top-k selection...")

dynamism_results = []
for k in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 241]:
    if k > len(ranked_features):
        continue
    top_k_indices = [feature_to_idx[f] for f in ranked_features[:k]]
    quality = compute_trajectory_quality(top_k_indices, X_full, age, phase, ref_pseudotime)
    dynamism_results.append({
        'n_features': k,
        'spearman_age': quality['spearman_age'],
        'spearman_ref': quality['spearman_ref'],
        'phase_separation': quality['phase_separation'],
        'selection_method': 'dynamism_top_k',
    })
    print(f"  k={k}: spearman_age={quality['spearman_age']:.4f}")

df_dynamism = pd.DataFrame(dynamism_results)
df_dynamism.to_csv('outputs/dynamism_topk_results.csv', index=False)

# === Strategy 3: Random baseline ===
print("\nComputing random baseline selection...")

random_results = []
for k in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 241]:
    if k > len(feature_names):
        continue
    n_trials = 5
    rho_ages = []
    rho_refs = []
    sils = []
    for trial in range(n_trials):
        rand_indices = np.random.choice(len(feature_names), k, replace=False)
        quality = compute_trajectory_quality(rand_indices, X_full, age, phase, ref_pseudotime)
        rho_ages.append(quality['spearman_age'])
        rho_refs.append(quality['spearman_ref'])
        sils.append(quality['phase_separation'])
    random_results.append({
        'n_features': k,
        'spearman_age_mean': np.mean(rho_ages),
        'spearman_age_std': np.std(rho_ages),
        'spearman_ref_mean': np.mean(rho_refs),
        'spearman_ref_std': np.std(rho_refs),
        'phase_separation_mean': np.mean(sils),
        'phase_separation_std': np.std(sils),
        'selection_method': 'random',
    })
    print(f"  k={k}: spearman_age={np.mean(rho_ages):.4f}±{np.std(rho_ages):.4f}")

df_random = pd.DataFrame(random_results)
df_random.to_csv('outputs/random_baseline_results.csv', index=False)

# === Strategy 4: Variance-based selection ===
print("\nComputing variance-based selection...")

var_order = df_metrics.sort_values('overall_var', ascending=False)['feature'].tolist()
variance_results = []
for k in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 241]:
    if k > len(var_order):
        continue
    top_k_indices = [feature_to_idx[f] for f in var_order[:k]]
    quality = compute_trajectory_quality(top_k_indices, X_full, age, phase, ref_pseudotime)
    variance_results.append({
        'n_features': k,
        'spearman_age': quality['spearman_age'],
        'spearman_ref': quality['spearman_ref'],
        'phase_separation': quality['phase_separation'],
        'selection_method': 'variance_top_k',
    })
    print(f"  k={k}: spearman_age={quality['spearman_age']:.4f}")

df_variance = pd.DataFrame(variance_results)
df_variance.to_csv('outputs/variance_topk_results.csv', index=False)

print("All selection strategies computed!")

# === Determine optimal feature set ===
improvements_dyn = df_dynamism['spearman_age'].diff().fillna(0).abs().values
optimal_n_dyn = df_dynamism.iloc[-1]['n_features']
for i in range(3, len(improvements_dyn)):
    if improvements_dyn[i] < 0.01:
        optimal_n_dyn = df_dynamism.iloc[i]['n_features']
        break

print(f"\nOptimal number of features (dynamism top-k): {int(optimal_n_dyn)}")

if len(df_greedy) > 5:
    improvements_g = df_greedy['spearman_age'].diff().fillna(0).abs().values
    optimal_n_greedy = df_greedy.iloc[-1]['n_features']
    for i in range(5, len(improvements_g)):
        if improvements_g[i] < 0.005:
            optimal_n_greedy = df_greedy.iloc[i]['n_features']
            break
else:
    optimal_n_greedy = df_greedy.iloc[-1]['n_features']

print(f"Optimal number of features (greedy): {int(optimal_n_greedy)}")

optimal_features_dynamism = ranked_features[:int(optimal_n_dyn)]
optimal_features_greedy = selected_features_greedy[:int(optimal_n_greedy)]

pd.DataFrame({
    'method': ['dynamism_top_k', 'greedy_forward'],
    'optimal_n_features': [int(optimal_n_dyn), int(optimal_n_greedy)],
    'features': [';'.join(optimal_features_dynamism), ';'.join(optimal_features_greedy)]
}).to_csv('outputs/optimal_feature_sets.csv', index=False)

print(f"\nOptimal dynamism features ({int(optimal_n_dyn)}): {optimal_features_dynamism}")
print(f"Optimal greedy features ({int(optimal_n_greedy)}): {optimal_features_greedy}")

print("\nPhase 3 complete!")