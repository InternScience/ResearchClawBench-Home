import json
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
sc.settings.verbosity = 0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'adata_RPE.h5ad'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

adata = sc.read_h5ad(DATA)
X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
features = np.array(adata.var_names)
obs = adata.obs.copy()

# Standardize features
Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
Xz = np.nan_to_num(Xz)

# Full-data graph and pseudotime surrogate
full = ad.AnnData(Xz.copy(), obs=obs.copy(), var=pd.DataFrame(index=features))
sc.pp.pca(full, n_comps=min(30, Xz.shape[1]-1))
sc.pp.neighbors(full, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(full, min_dist=0.35)
sc.tl.diffmap(full)
root_idx = int(np.argmin(obs['annotated_age'].values))
full.uns['iroot'] = root_idx
sc.tl.dpt(full)
full_pt = np.asarray(full.obs['dpt_pseudotime'])
full_umap = np.asarray(full.obsm['X_umap'])

# Association statistics for each feature
age = obs['annotated_age'].to_numpy(dtype=float)
state = obs['state'].astype(str).to_numpy()
phase = obs['phase'].astype(str).to_numpy()
state_binary = (state == 'cycling').astype(int)
phase_codes, _ = pd.factorize(phase)

rows = []
for j, feat in enumerate(features):
    x = Xz[:, j]
    rho_age = stats.spearmanr(x, age).statistic
    rho_pt = stats.spearmanr(x, full_pt).statistic
    f_state = stats.f_oneway(x[state=='cycling'], x[state!='cycling']).statistic if len(np.unique(state))>1 else 0.0
    f_phase = stats.f_oneway(*[x[phase==p] for p in pd.unique(phase)]).statistic if len(pd.unique(phase))>1 else 0.0
    dyn_score = abs(rho_pt) + abs(rho_age)
    conf_score = np.log1p(max(f_state,0)) + np.log1p(max(f_phase,0))
    score = dyn_score - 0.35 * conf_score
    rows.append([feat, rho_age, rho_pt, f_state, f_phase, dyn_score, conf_score, score])

res = pd.DataFrame(rows, columns=['feature','rho_age','rho_pseudotime','F_state','F_phase','dynamic_score','confound_score','trajectory_score'])
res = res.sort_values('trajectory_score', ascending=False).reset_index(drop=True)
res['rank'] = np.arange(1, len(res)+1)
res.to_csv(OUT / 'feature_scores.csv', index=False)

# Compare subsets
candidate_ks = [10, 20, 30, 40, 60, 80]
all_metrics = []
full_dist = pairwise_distances(full_umap)
full_rank = np.argsort(full_dist, axis=1)[:, 1:16]

for k in candidate_ks:
    selected = res.head(k)['feature'].tolist()
    idx = [np.where(features == s)[0][0] for s in selected]
    subX = Xz[:, idx]
    sub = ad.AnnData(subX.copy(), obs=obs.copy(), var=pd.DataFrame(index=selected))
    n_comps = max(2, min(15, subX.shape[1]-1))
    sc.pp.pca(sub, n_comps=n_comps)
    sc.pp.neighbors(sub, n_neighbors=15, use_rep='X_pca')
    sc.tl.umap(sub, min_dist=0.35)
    sc.tl.diffmap(sub)
    sub.uns['iroot'] = root_idx
    sc.tl.dpt(sub)
    sub_pt = np.asarray(sub.obs['dpt_pseudotime'])
    sub_umap = np.asarray(sub.obsm['X_umap'])

    rho_pt = stats.spearmanr(full_pt, sub_pt).statistic
    dist = pairwise_distances(sub_umap)
    sub_rank = np.argsort(dist, axis=1)[:, 1:16]
    jaccs = [len(set(full_rank[i]).intersection(set(sub_rank[i]))) / len(set(full_rank[i]).union(set(sub_rank[i]))) for i in range(len(full_rank))]
    nn_jaccard = float(np.mean(jaccs))
    age_r2 = np.corrcoef(sub_pt, age)[0,1]**2
    state_r2 = np.corrcoef(sub_pt, state_binary)[0,1]**2
    all_metrics.append([k, rho_pt, nn_jaccard, age_r2, state_r2])

metrics = pd.DataFrame(all_metrics, columns=['k','pseudotime_corr','neighbor_jaccard','age_r2','state_r2'])
metrics['objective'] = metrics['pseudotime_corr'] + metrics['neighbor_jaccard'] + 0.25*metrics['age_r2'] - 0.15*metrics['state_r2']
metrics.to_csv(OUT / 'subset_benchmark.csv', index=False)

best_k = int(metrics.sort_values('objective', ascending=False).iloc[0]['k'])
selected = res.head(best_k)['feature'].tolist()
idx = [np.where(features == s)[0][0] for s in selected]
selX = Xz[:, idx]
sel = ad.AnnData(selX.copy(), obs=obs.copy(), var=pd.DataFrame(index=selected))
sc.pp.pca(sel, n_comps=max(2, min(15, selX.shape[1]-1)))
sc.pp.neighbors(sel, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(sel, min_dist=0.35)
sc.tl.diffmap(sel)
sel.uns['iroot'] = root_idx
sc.tl.dpt(sel)

# Save selected features and embeddings
pd.DataFrame({'selected_feature': selected}).to_csv(OUT / 'selected_features.csv', index=False)
pd.DataFrame(sel.obsm['X_umap'], columns=['UMAP1','UMAP2']).assign(annotated_age=age, state=state, phase=phase, pseudotime=sel.obs['dpt_pseudotime'].to_numpy()).to_csv(OUT / 'selected_embedding.csv', index=False)
pd.DataFrame(full.obsm['X_umap'], columns=['UMAP1','UMAP2']).assign(annotated_age=age, state=state, phase=phase, pseudotime=full_pt).to_csv(OUT / 'full_embedding.csv', index=False)

# Figures
fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(full_umap[:,0], full_umap[:,1], c=age, s=10, cmap='viridis')
axes[0].set_title('Full feature space')
axes[1].scatter(sel.obsm['X_umap'][:,0], sel.obsm['X_umap'][:,1], c=age, s=10, cmap='viridis')
axes[1].set_title(f'Selected feature space (k={best_k})')
for ax in axes:
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
fig.colorbar(axes[1].collections[0], ax=axes, label='Annotated age')
fig.tight_layout()
fig.savefig(IMG / 'trajectory_umap_age.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
pal = {'cycling':'#1f77b4','arrested':'#d62728'}
for ax, emb, title in [(axes[0], full_umap, 'Full feature space'), (axes[1], sel.obsm['X_umap'], f'Selected feature space (k={best_k})')]:
    for st, col in pal.items():
        m = (state == st)
        ax.scatter(emb[m,0], emb[m,1], s=10, c=col, label=st, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG / 'trajectory_umap_state.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
plot_df = res.head(20).sort_values('trajectory_score')
ax.barh(plot_df['feature'], plot_df['trajectory_score'], color='#4c72b0')
ax.set_xlabel('Trajectory preservation score')
ax.set_ylabel('Feature')
ax.set_title('Top-ranked dynamic features')
fig.tight_layout()
fig.savefig(IMG / 'top_features.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(metrics['k'], metrics['pseudotime_corr'], marker='o', label='Pseudotime correlation')
ax.plot(metrics['k'], metrics['neighbor_jaccard'], marker='s', label='kNN Jaccard')
ax.plot(metrics['k'], metrics['objective'], marker='^', label='Composite objective')
ax.axvline(best_k, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Number of selected features')
ax.set_ylabel('Score')
ax.set_title('Subset size benchmark')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG / 'subset_benchmark.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(full_pt, sel.obs['dpt_pseudotime'], s=10, alpha=0.5)
axes[0].set_xlabel('Full-feature pseudotime')
axes[0].set_ylabel('Selected-feature pseudotime')
axes[0].set_title(f'Pseudotime agreement (rho={metrics.loc[metrics.k==best_k, "pseudotime_corr"].iloc[0]:.2f})')

corr_df = res[['dynamic_score','confound_score','trajectory_score']]
sns.heatmap(corr_df.corr(), annot=True, cmap='vlag', ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Score relationship summary')
fig.tight_layout()
fig.savefig(IMG / 'validation_plots.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# feature trends
sel_scores = res.set_index('feature').loc[selected].copy()
top6 = sel_scores.sort_values('trajectory_score', ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(13,7), sharex=True)
order = np.argsort(sel.obs['dpt_pseudotime'].to_numpy())
pt = sel.obs['dpt_pseudotime'].to_numpy()[order]
for ax, feat in zip(axes.flat, top6):
    j = np.where(features == feat)[0][0]
    y = Xz[:, j][order]
    smooth = pd.Series(y).rolling(151, min_periods=20, center=True).mean()
    ax.scatter(pt, y, s=4, alpha=0.2)
    ax.plot(pt, smooth, color='crimson', linewidth=2)
    ax.set_title(feat)
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('z-scored intensity')
fig.tight_layout()
fig.savefig(IMG / 'feature_dynamics.png', dpi=200, bbox_inches='tight')
plt.close(fig)

summary = {
    'n_cells': int(adata.n_obs),
    'n_features': int(adata.n_vars),
    'best_k': int(best_k),
    'selected_features': selected,
    'metrics': metrics.to_dict(orient='records')
}
with open(OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
