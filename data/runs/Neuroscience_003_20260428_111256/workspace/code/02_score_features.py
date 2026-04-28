"""
Step 2: Compute feature scores and rank features.
"""
import os, sys, json
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_selection import (variance_score, spearman_pseudotime, anova_f_phase,
                               build_knn_graph, laplacian_score, graph_smoothness,
                               composite_dyn_score)

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'outputs')

a = ad.read_h5ad(os.path.join(OUT, 'adata_parsed.h5ad'))
print('Loaded', a)

X_raw = np.asarray(a.X)  # already float32
# Standardize per feature (z-score). Batch can confound -> remove batch means per feature.
scaler = StandardScaler().fit(X_raw)
Xz = scaler.transform(X_raw)
# regress out batch by subtracting per-batch mean
batch = a.obs['batch'].astype(str).values
Xc = Xz.copy()
for b in np.unique(batch):
    m = batch == b
    Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
print('Xc shape', Xc.shape, 'mean', Xc.mean(), 'std', Xc.std())

t = a.obs['annotated_age'].astype(float).values
phases = a.obs['phase'].astype(str).values

# Build kNN graph from full feature space (used for Laplacian/graph-smoothness)
knn_idx, knn_d = build_knn_graph(Xc, k=15)
print('kNN built', knn_idx.shape)

print('Computing scores...')
v_var = variance_score(X_raw)  # raw variance for context
v_sp = spearman_pseudotime(Xc, t)
v_f = anova_f_phase(Xc, phases)
v_lap = laplacian_score(Xc, knn_idx)
v_gs = graph_smoothness(Xc, knn_idx)
v_dyn = composite_dyn_score(v_gs, v_sp)

scores = pd.DataFrame({
    'feature': a.var_names,
    'protein': a.var['protein'].values,
    'measurement': a.var['measurement'].values,
    'compartment': a.var['compartment'].values,
    'variance': v_var,
    'spearman_abs_pseudotime': v_sp,
    'anova_F_phase': v_f,
    'laplacian_score_neg': v_lap,
    'graph_smoothness': v_gs,
    'dyn_score': v_dyn,
})

# rank columns (1 = best)
for col in ['variance','spearman_abs_pseudotime','anova_F_phase','laplacian_score_neg','graph_smoothness','dyn_score']:
    scores[col + '_rank'] = scores[col].rank(ascending=False, method='min').astype(int)

scores.to_csv(os.path.join(OUT, 'feature_scores.csv'), index=False)
print('Saved feature_scores.csv')

# Selected feature lists for each method at k = 10, 25, 50
methods = {
    'HVF_variance': 'variance',
    'Spearman_pseudotime': 'spearman_abs_pseudotime',
    'ANOVA_phase': 'anova_F_phase',
    'LaplacianScore': 'laplacian_score_neg',
    'GraphSmoothness': 'graph_smoothness',
    'DynScore': 'dyn_score',
}
selected = {}
for k in [10, 25, 50]:
    selected[k] = {}
    for m, col in methods.items():
        top = scores.sort_values(col, ascending=False).head(k)['feature'].tolist()
        selected[k][m] = top
    # random baseline (fixed seed)
    rng = np.random.RandomState(0)
    selected[k]['Random'] = list(rng.choice(a.var_names, size=k, replace=False))

with open(os.path.join(OUT, 'selected_features.json'), 'w') as f:
    json.dump({str(k): v for k, v in selected.items()}, f, indent=2)
print('Saved selected_features.json')

# Also print a small summary
print('\nTop 10 by DynScore:')
print(scores.sort_values('dyn_score', ascending=False).head(10)[['feature','dyn_score','graph_smoothness','spearman_abs_pseudotime']].to_string(index=False))
print('\nTop 10 by Spearman:')
print(scores.sort_values('spearman_abs_pseudotime', ascending=False).head(10)[['feature','spearman_abs_pseudotime','dyn_score']].to_string(index=False))
