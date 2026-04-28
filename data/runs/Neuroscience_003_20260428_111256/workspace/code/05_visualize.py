"""
Step 5: UMAP visualisations of full vs subset feature spaces, pseudotime
        recovery, kNN preservation curve, and heatmap along pseudotime.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_selection import build_knn_graph

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'outputs')
IMG = os.path.join(WS, 'report', 'images')

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

phase_order = ['G0','G1','S','G2']
phase_palette = dict(zip(phase_order, sns.color_palette('Set2', 4)))

def run_umap(Xs, seed=0):
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=seed)
    return reducer.fit_transform(Xs)

# ===== Figure 03: UMAP grid full vs DynScore (k=10,25,50) vs HVF (k=25) vs Random (k=25) =====
configs = [('Full (241)', list(a.var_names)),
           ('DynScore k=10', selected['10']['DynScore']),
           ('DynScore k=25', selected['25']['DynScore']),
           ('DynScore k=50', selected['50']['DynScore']),
           ('HVF k=25', selected['25']['HVF_variance']),
           ('Random k=25', selected['25']['Random'])]

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
for col, (name, feats) in enumerate(configs):
    cols = [list(a.var_names).index(f) for f in feats]
    Xs = Xc[:, cols]
    emb = run_umap(Xs, seed=0)
    # row 0 colored by phase
    for ph in phase_order:
        m = phase == ph
        axes[0, col].scatter(emb[m, 0], emb[m, 1], s=4, label=ph,
                             c=[phase_palette[ph]], alpha=0.7)
    axes[0, col].set_title(name)
    if col == 0: axes[0, col].set_ylabel('UMAP - phase'); axes[0, col].legend(fontsize=7, markerscale=2)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    # row 1 colored by annotated_age
    sc_ = axes[1, col].scatter(emb[:, 0], emb[:, 1], s=4, c=t, cmap='viridis')
    if col == 0: axes[1, col].set_ylabel('UMAP - pseudotime')
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
plt.colorbar(sc_, ax=axes[1, :].tolist(), shrink=0.6, label='annotated_age')
plt.suptitle('UMAP comparison: full vs subset feature spaces', y=1.02)
plt.savefig(os.path.join(IMG, '03_umap_comparison.png'), dpi=140, bbox_inches='tight')
plt.close()
print('saved 03')

# ===== Figure 04: Pseudotime recovery (annotated_age vs DPT pseudotime) =====
def dpt(Xs):
    ad_sub = ad.AnnData(Xs)
    n = min(15, Xs.shape[1]-1, Xs.shape[0]-1)
    if n >= 2:
        sc.pp.pca(ad_sub, n_comps=n); sc.pp.neighbors(ad_sub, n_neighbors=15, use_rep='X_pca')
    else:
        sc.pp.neighbors(ad_sub, n_neighbors=15, use_rep='X')
    sc.tl.diffmap(ad_sub, n_comps=min(10,n))
    ad_sub.uns['iroot'] = int(np.argmin(t))
    sc.tl.dpt(ad_sub)
    return ad_sub.obs['dpt_pseudotime'].values

panels = [('Full (241)', list(a.var_names)),
          ('DynScore k=25', selected['25']['DynScore']),
          ('Spearman k=25', selected['25']['Spearman_pseudotime']),
          ('HVF k=25', selected['25']['HVF_variance']),
          ('LaplacianScore k=25', selected['25']['LaplacianScore']),
          ('Random k=25', selected['25']['Random'])]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, feats) in zip(axes.ravel(), panels):
    cols = [list(a.var_names).index(f) for f in feats]
    Xs = Xc[:, cols]
    try:
        pt = dpt(Xs)
        rho, _ = spearmanr(pt, t); rho = abs(rho)
    except Exception as e:
        pt = np.zeros_like(t); rho = float('nan')
    for ph in phase_order:
        m = phase == ph
        ax.scatter(t[m], pt[m], s=6, label=ph, c=[phase_palette[ph]], alpha=0.7)
    ax.set_xlabel('annotated_age'); ax.set_ylabel('DPT pseudotime')
    ax.set_title(f'{name}  |Spearman|={rho:.3f}')
    ax.legend(fontsize=7, markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(IMG, '04_pseudotime_recovery.png'), dpi=140)
plt.close()
print('saved 04')

# ===== Figure 05: kNN preservation curve as function of k_features =====
ref_idx, _ = build_knn_graph(Xc, k=30)
ref_sets = [set(row) for row in ref_idx]

ks = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 241]
methods = ['DynScore','Spearman_pseudotime','ANOVA_phase','HVF_variance',
           'GraphSmoothness','LaplacianScore','Random']
scores_df = pd.read_csv(os.path.join(OUT, 'feature_scores.csv'))
score_cols = {'DynScore':'dyn_score','Spearman_pseudotime':'spearman_abs_pseudotime',
              'ANOVA_phase':'anova_F_phase','HVF_variance':'variance',
              'GraphSmoothness':'graph_smoothness','LaplacianScore':'laplacian_score_neg'}
rng = np.random.RandomState(0)

curve = {m: [] for m in methods}
for k_ in ks:
    for m in methods:
        if m == 'Random':
            feats = list(rng.choice(a.var_names, size=k_, replace=False))
        else:
            feats = scores_df.sort_values(score_cols[m], ascending=False).head(k_)['feature'].tolist()
        cols = [list(a.var_names).index(f) for f in feats]
        Xs = Xc[:, cols]
        idx_b, _ = build_knn_graph(Xs, k=30)
        js = [len(ref_sets[i] & set(row)) / len(ref_sets[i] | set(row)) for i, row in enumerate(idx_b)]
        curve[m].append(np.mean(js))
    print('k=', k_, {m: round(curve[m][-1],3) for m in methods})

plt.figure(figsize=(8,6))
for m in methods:
    plt.plot(ks, curve[m], marker='o', label=m)
plt.xscale('log')
plt.xlabel('# selected features (k)'); plt.ylabel('mean kNN Jaccard vs full feature space')
plt.title('kNN-graph preservation curve')
plt.legend(fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, '05_knn_preservation.png'), dpi=140)
plt.close()
# save curve csv
pd.DataFrame({'k':ks, **curve}).to_csv(os.path.join(OUT, 'knn_preservation_curve.csv'), index=False)
print('saved 05 + curve csv')

# ===== Figure 06: Heatmap of top-25 DynScore features along pseudotime =====
top25 = selected['25']['DynScore']
order = np.argsort(t)
M = Xc[np.ix_(order, [list(a.var_names).index(f) for f in top25])].T
plt.figure(figsize=(13, 7))
sns.heatmap(M, cmap='RdBu_r', center=0, vmin=-2.5, vmax=2.5,
            yticklabels=[f.replace('Int_','').replace('AreaShape_','') for f in top25],
            xticklabels=False, cbar_kws={'label':'z-score (batch corrected)'})
plt.xlabel(f'cells ordered by annotated_age (n={len(order)})')
plt.title('Top-25 DynScore features along pseudotime')
# overlay phase color bar at top
ax = plt.gca()
phase_codes = pd.Categorical(phase[order], categories=phase_order).codes
from matplotlib.colors import ListedColormap
cmap = ListedColormap([phase_palette[p] for p in phase_order])
ax2 = ax.inset_axes([0, 1.01, 1, 0.02])
ax2.imshow(phase_codes.reshape(1,-1), aspect='auto', cmap=cmap)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title('phase', fontsize=8, loc='left')
plt.savefig(os.path.join(IMG, '06_heatmap_pseudotime.png'), dpi=140, bbox_inches='tight')
plt.close()
print('saved 06')
