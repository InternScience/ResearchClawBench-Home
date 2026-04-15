import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse, stats
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import umap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'adata_RPE.h5ad'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_context('talk')
sns.set_style('whitegrid')
np.random.seed(42)


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def compute_pseudotime(Xz, obs):
    pca = PCA(n_components=min(20, Xz.shape[1], Xz.shape[0]-1), random_state=42)
    pcs = pca.fit_transform(Xz)
    nn = NearestNeighbors(n_neighbors=min(15, Xz.shape[0]-1), metric='euclidean')
    nn.fit(pcs)
    dists, idx = nn.kneighbors(pcs)
    n = Xz.shape[0]
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    vals = dists.ravel()
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.minimum(graph.T)

    root_candidates = np.arange(n)
    if 'annotated_age' in obs.columns:
        ages = pd.to_numeric(obs['annotated_age'], errors='coerce').values
        finite = np.isfinite(ages)
        if finite.any():
            min_age = np.nanmin(ages[finite])
            root_candidates = np.where(np.isclose(ages, min_age, equal_nan=False))[0]
    if len(root_candidates) == 0:
        root_candidates = np.arange(n)
    root = int(root_candidates[np.argmin(pcs[root_candidates, 0])])
    dist = shortest_path(graph, directed=False, indices=root)
    finite = np.isfinite(dist)
    maxd = dist[finite].max()
    pseudotime = dist.copy()
    pseudotime[~finite] = maxd
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-12)
    return pcs, graph, pseudotime, root


def dynamic_scores(Xz, graph, pseudotime):
    corr = []
    smooth = []
    for j in range(Xz.shape[1]):
        x = Xz[:, j]
        r = stats.spearmanr(x, pseudotime).statistic
        if np.isnan(r):
            r = 0.0
        corr.append(abs(r))
    A = graph.copy().tocsr()
    A.data = np.ones_like(A.data)
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    smooth_mat = A @ Xz / deg[:, None]
    smooth = ((smooth_mat - Xz) ** 2).mean(axis=0)
    smooth_score = 1 / (1 + smooth)
    corr = np.asarray(corr)
    smooth_score = np.asarray(smooth_score)
    dyn = 0.7 * (corr / (corr.max() + 1e-12)) + 0.3 * (smooth_score / (smooth_score.max() + 1e-12))
    return corr, smooth_score, dyn


def evaluate_subset(Xz, Xref_pcs, ref_pt, feature_idx):
    Xsub = Xz[:, feature_idx]
    npc = min(10, Xsub.shape[1], Xsub.shape[0]-1)
    pcs = PCA(n_components=npc, random_state=42).fit_transform(Xsub)
    refdist = pairwise_distances(Xref_pcs[:, :npc])
    subdist = pairwise_distances(pcs)
    tri = np.triu_indices_from(refdist, k=1)
    dist_corr = stats.spearmanr(refdist[tri], subdist[tri]).statistic
    if np.isnan(dist_corr):
        dist_corr = 0.0
    pt_sub = (pcs[:, 0] - pcs[:, 0].min()) / (pcs[:, 0].max() - pcs[:, 0].min() + 1e-12)
    pt_corr = stats.spearmanr(ref_pt, pt_sub).statistic
    if np.isnan(pt_corr):
        pt_corr = 0.0
    nbrs_ref = NearestNeighbors(n_neighbors=15).fit(Xref_pcs[:, :npc])
    nbrs_sub = NearestNeighbors(n_neighbors=15).fit(pcs)
    ir = nbrs_ref.kneighbors(return_distance=False)
    isub = nbrs_sub.kneighbors(return_distance=False)
    overlaps = [len(set(ir[i][1:]).intersection(set(isub[i][1:]))) / 14 for i in range(len(ir))]
    return {
        'distance_spearman': float(dist_corr),
        'pseudotime_spearman': float(pt_corr),
        'neighbor_overlap': float(np.mean(overlaps))
    }


def make_heatmap(df_long, outpath):
    pivot = df_long.pivot(index='feature', columns='cell_rank_bin', values='value')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': 'scaled expression'})
    plt.xlabel('Pseudotime bin')
    plt.ylabel('Selected feature')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    adata = ad.read_h5ad(DATA)
    X = to_dense(adata.X).astype(float)
    Xz = standardize(X)
    obs = adata.obs.copy()
    var_names = np.array(adata.var_names.astype(str))

    pcs, graph, pseudotime, root = compute_pseudotime(Xz, obs)
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.35)
    emb = reducer.fit_transform(pcs[:, :20])

    corr, smooth_score, dyn = dynamic_scores(Xz, graph, pseudotime)
    var_score = Xz.var(axis=0)
    rank_dyn = np.argsort(-dyn)
    rank_var = np.argsort(-var_score)

    sizes = [5, 10, 20, 30, 50, 75, 100]
    rows = []
    rng = np.random.default_rng(42)
    for k in sizes:
        for method, ranking in [('dynamic', rank_dyn), ('variance', rank_var)]:
            metrics = evaluate_subset(Xz, pcs, pseudotime, ranking[:k])
            rows.append({'method': method, 'subset_size': k, **metrics})
        rand_metrics = []
        for rep in range(10):
            feat_idx = rng.choice(Xz.shape[1], size=k, replace=False)
            m = evaluate_subset(Xz, pcs, pseudotime, feat_idx)
            rand_metrics.append(m)
            rows.append({'method': 'random', 'subset_size': k, 'replicate': rep, **m})
    metrics_df = pd.DataFrame(rows)

    topn = 20
    feature_table = pd.DataFrame({
        'feature': var_names,
        'dynamic_score': dyn,
        'pseudotime_abs_spearman': corr,
        'smoothness_score': smooth_score,
        'variance_score': var_score
    }).sort_values('dynamic_score', ascending=False)
    feature_table.to_csv(OUT / 'feature_ranking.csv', index=False)
    metrics_df.to_csv(OUT / 'subset_metrics.csv', index=False)

    overview = {
        'n_cells': int(adata.n_obs),
        'n_features': int(adata.n_vars),
        'phase_counts': {str(k): int(v) for k, v in obs['phase'].astype(str).value_counts().to_dict().items()} if 'phase' in obs.columns else {},
        'state_counts': {str(k): int(v) for k, v in obs['state'].astype(str).value_counts(dropna=False).to_dict().items()} if 'state' in obs.columns else {},
        'batch_counts': {str(k): int(v) for k, v in obs['batch'].astype(str).value_counts().to_dict().items()} if 'batch' in obs.columns else {},
        'root_cell_index': int(root)
    }
    with open(OUT / 'dataset_overview.json', 'w') as f:
        json.dump(overview, f, indent=2)

    emb_df = pd.DataFrame({
        'UMAP1': emb[:,0], 'UMAP2': emb[:,1], 'pseudotime': pseudotime,
        'phase': obs['phase'].astype(str).values if 'phase' in obs.columns else 'NA',
        'state': obs['state'].astype(str).values if 'state' in obs.columns else 'NA',
        'annotated_age': pd.to_numeric(obs['annotated_age'], errors='coerce').values if 'annotated_age' in obs.columns else np.nan,
    })
    emb_df.to_csv(OUT / 'embedding_pseudotime.csv', index=False)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=pseudotime, s=10, cmap='plasma', alpha=0.85)
    plt.colorbar(sc, label='Pseudotime')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Full-data manifold colored by inferred pseudotime')
    plt.tight_layout()
    plt.savefig(IMG / 'embedding_pseudotime.png', dpi=200)
    plt.close()

    plot_df = metrics_df.groupby(['method','subset_size'], as_index=False).agg({
        'distance_spearman':'mean',
        'pseudotime_spearman':'mean',
        'neighbor_overlap':'mean'
    })
    plot_df.to_csv(OUT / 'subset_metric_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16,5), sharex=True)
    for ax, col, title in zip(axes, ['distance_spearman','pseudotime_spearman','neighbor_overlap'], ['Distance preservation','Pseudotime agreement','Neighbor overlap']):
        sns.lineplot(data=plot_df, x='subset_size', y=col, hue='method', marker='o', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Subset size')
    axes[0].set_ylabel('Metric value')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    plt.tight_layout()
    plt.savefig(IMG / 'preservation_curves.png', dpi=200)
    plt.close()

    best_k = 20
    top_feats = feature_table.head(best_k)['feature'].tolist()
    order = np.argsort(pseudotime)
    bins = pd.qcut(np.arange(len(order)), q=20, labels=False)
    heat_rows = []
    for feat in top_feats:
        vals = Xz[order, np.where(var_names == feat)[0][0]]
        for b in np.unique(bins):
            mask = bins == b
            heat_rows.append({'feature': feat, 'cell_rank_bin': int(b), 'value': float(vals[mask].mean())})
    heat_df = pd.DataFrame(heat_rows)
    heat_df.to_csv(OUT / 'dynamic_feature_heatmap_values.csv', index=False)
    make_heatmap(heat_df, IMG / 'dynamic_feature_heatmap.png')

    comp = plot_df[plot_df['subset_size'] == best_k].copy()
    comp_m = comp.melt(id_vars=['method','subset_size'], value_vars=['distance_spearman','pseudotime_spearman','neighbor_overlap'], var_name='metric', value_name='value')
    comp_m.to_csv(OUT / 'best_subset_comparison_long.csv', index=False)
    plt.figure(figsize=(9,5))
    sns.barplot(data=comp_m, x='metric', y='value', hue='method')
    plt.xticks(rotation=20)
    plt.ylim(0, 1.05)
    plt.title(f'Comparison at subset size {best_k}')
    plt.tight_layout()
    plt.savefig(IMG / 'baseline_comparison_bestk.png', dpi=200)
    plt.close()

    claim_rows = []
    best_dyn = plot_df[(plot_df.method=='dynamic') & (plot_df.subset_size==best_k)].iloc[0]
    best_var = plot_df[(plot_df.method=='variance') & (plot_df.subset_size==best_k)].iloc[0]
    for metric in ['distance_spearman','pseudotime_spearman','neighbor_overlap']:
        claim_rows.append({
            'claim': f'dynamic_{metric}_better_than_variance_at_{best_k}',
            'dynamic_value': float(best_dyn[metric]),
            'variance_value': float(best_var[metric]),
            'supported': bool(best_dyn[metric] > best_var[metric]),
            'artifact': 'outputs/subset_metric_summary.csv'
        })
    pd.DataFrame(claim_rows).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    summary = {
        'best_k_for_reporting': best_k,
        'top_features': top_feats,
        'top5_features': feature_table.head(5)['feature'].tolist()
    }
    with open(OUT / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
