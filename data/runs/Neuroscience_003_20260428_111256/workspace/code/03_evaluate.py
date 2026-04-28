"""
Step 3: Evaluate trajectory preservation for each method × k.

Metrics:
  - kNN Jaccard preservation between full-feature and subset-feature kNN graphs
  - Spearman correlation between subset-derived diffusion pseudotime (DPT) and annotated_age
  - kNN classifier accuracy for cell-cycle phase using subset features
  - Silhouette of phase labels in subset feature PCA
"""
import os, sys, json
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_selection import build_knn_graph

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'outputs')

a = ad.read_h5ad(os.path.join(OUT, 'adata_parsed.h5ad'))
with open(os.path.join(OUT, 'selected_features.json')) as f:
    selected = json.load(f)

X_raw = np.asarray(a.X)
scaler = StandardScaler().fit(X_raw)
Xz = scaler.transform(X_raw)
batch = a.obs['batch'].astype(str).values
Xc = Xz.copy()
for b in np.unique(batch):
    m = batch == b
    Xc[m] -= Xc[m].mean(axis=0, keepdims=True)

t = a.obs['annotated_age'].astype(float).values
phase = a.obs['phase'].astype(str).values

# Reference kNN on full feature set
K_EVAL = 30
ref_idx, _ = build_knn_graph(Xc, k=K_EVAL)
ref_sets = [set(row) for row in ref_idx]

def jaccard_preservation(idx_b):
    js = []
    for i, row in enumerate(idx_b):
        a_set = ref_sets[i]; b_set = set(row)
        inter = len(a_set & b_set); union = len(a_set | b_set)
        js.append(inter / union if union else 0.0)
    return float(np.mean(js))

def dpt_pseudotime(Xs):
    """Compute diffusion pseudotime from subset feature matrix."""
    ad_sub = ad.AnnData(Xs)
    # use PCA if dim allows
    n_comps = min(15, Xs.shape[1] - 1, Xs.shape[0] - 1)
    if n_comps >= 2:
        sc.pp.pca(ad_sub, n_comps=n_comps)
        sc.pp.neighbors(ad_sub, n_neighbors=15, use_rep='X_pca')
    else:
        sc.pp.neighbors(ad_sub, n_neighbors=15, use_rep='X')
    sc.tl.diffmap(ad_sub, n_comps=min(10, n_comps))
    # Choose root cell as the one with smallest annotated_age (proxy)
    root = int(np.argmin(t))
    ad_sub.uns['iroot'] = root
    try:
        sc.tl.dpt(ad_sub)
        pt = ad_sub.obs['dpt_pseudotime'].values
    except Exception as e:
        pt = ad_sub.obsm['X_diffmap'][:, 1]
    return pt

def eval_subset(features):
    cols = [list(a.var_names).index(f) for f in features]
    Xs = Xc[:, cols]
    # 1. kNN Jaccard
    idx_b, _ = build_knn_graph(Xs, k=K_EVAL)
    jac = jaccard_preservation(idx_b)
    # 2. DPT correlation with annotated_age
    try:
        pt = dpt_pseudotime(Xs)
        # take absolute spearman (sign arbitrary)
        rho, _ = spearmanr(pt, t)
        rho = abs(rho) if not np.isnan(rho) else 0.0
    except Exception as e:
        print('DPT error:', e); rho = float('nan')
    # 3. Phase classifier accuracy (5-fold CV)
    clf = KNeighborsClassifier(n_neighbors=15)
    try:
        accs = cross_val_score(clf, Xs, phase, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        acc = float(accs.mean()); acc_sd = float(accs.std())
    except Exception:
        acc = float('nan'); acc_sd = float('nan')
    # 4. Silhouette of phase in subset
    try:
        sil = float(silhouette_score(Xs, phase, sample_size=1500, random_state=0))
    except Exception:
        sil = float('nan')
    return jac, rho, acc, acc_sd, sil

# Reference (full)
jac_full, rho_full, acc_full, acc_full_sd, sil_full = eval_subset(list(a.var_names))
print('FULL ->', jac_full, rho_full, acc_full, acc_full_sd, sil_full)

rows = []
rows.append(dict(method='Full', k=241, knn_jaccard=jac_full,
                 dpt_spearman=rho_full, phase_acc=acc_full,
                 phase_acc_sd=acc_full_sd, silhouette=sil_full))

for k_str in selected:
    k = int(k_str)
    for method, feats in selected[k_str].items():
        jac, rho, acc, acc_sd, sil = eval_subset(feats)
        rows.append(dict(method=method, k=k, knn_jaccard=jac, dpt_spearman=rho,
                         phase_acc=acc, phase_acc_sd=acc_sd, silhouette=sil))
        print(f'{method:20s} k={k:3d} jac={jac:.3f} dpt_rho={rho:.3f} acc={acc:.3f}±{acc_sd:.3f} sil={sil:.3f}')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'evaluation_metrics.csv'), index=False)
print('\nSaved evaluation_metrics.csv')
print(df.to_string(index=False))
