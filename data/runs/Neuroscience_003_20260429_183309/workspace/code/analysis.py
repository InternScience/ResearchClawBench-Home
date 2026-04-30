#!/usr/bin/env python3
"""Trajectory-preserving dynamic feature selection for adata_RPE.h5ad.

This script is intentionally self-contained and writes all tables/figures used by
report/report.md. It derives a full-feature reference trajectory, ranks molecular
features by dynamic association with the trajectory, selects a compact panel, and
validates trajectory preservation against high-variance and random baselines.
"""
import json
import os
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, f_oneway
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style='whitegrid', context='paper')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'adata_RPE.h5ad'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(1729)


def clean_feature_name(name: str) -> str:
    x = str(name)
    for prefix in ['Int_MeanEdge_', 'Int_Mean_', 'Int_Intg_', 'Int_']:
        x = x.replace(prefix, '')
    x = x.replace('_cell', '').replace('_nuclei', '').replace('_cyto', '')
    return x


def load_matrix(adata):
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def standardize(X):
    X = np.nan_to_num(X, nan=np.nanmedian(X), posinf=np.nanmax(np.isfinite(X)), neginf=0.0)
    return StandardScaler().fit_transform(X)


def knn_graph_distances(Z, k=15):
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(Z)), metric='euclidean').fit(Z)
    dists, inds = nbrs.kneighbors(Z)
    rows, cols, vals = [], [], []
    for i in range(len(Z)):
        for j, d in zip(inds[i, 1:], dists[i, 1:]):
            rows.append(i); cols.append(j); vals.append(float(d))
    G = sparse.csr_matrix((vals, (rows, cols)), shape=(len(Z), len(Z)))
    G = G.minimum(G.T) + G.maximum(G.T)  # symmetric, keep available edges
    return G, inds[:, 1:]


def graph_pseudotime(Z, root_idx, k=15):
    G, _ = knn_graph_distances(Z, k=k)
    dist = dijkstra(G, directed=False, indices=int(root_idx), unweighted=False)
    # replace disconnected infinities by Euclidean distance from root, then scale
    inf = ~np.isfinite(dist)
    if np.any(inf):
        dist[inf] = np.linalg.norm(Z[inf] - Z[root_idx], axis=1) + np.nanmax(dist[~inf])
    pt = (dist - dist.min()) / (dist.max() - dist.min() + 1e-12)
    return pt


def trajectory_embedding(Z, random_state=1729):
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.25, metric='euclidean', random_state=random_state)
        emb = reducer.fit_transform(Z)
        method = 'UMAP'
    else:
        emb = PCA(n_components=2, random_state=random_state).fit_transform(Z)
        method = 'PCA2'
    return emb, method


def feature_dynamic_scores(Xs, feature_names, pseudotime, n_bins=8):
    bins = pd.qcut(pseudotime, q=n_bins, labels=False, duplicates='drop')
    bins = np.asarray(bins, dtype=int)
    mi = mutual_info_regression(Xs, pseudotime, random_state=1729)
    rows = []
    for j, name in enumerate(feature_names):
        x = Xs[:, j]
        rho, p = spearmanr(x, pseudotime)
        if not np.isfinite(rho): rho, p = 0.0, 1.0
        groups = [x[bins == b] for b in sorted(np.unique(bins)) if np.sum(bins == b) > 1]
        if len(groups) > 1:
            try:
                F, ap = f_oneway(*groups)
            except Exception:
                F, ap = 0.0, 1.0
        else:
            F, ap = 0.0, 1.0
        # smoothness: fraction of variance explained by trajectory-bin means
        means = np.array([np.mean(x[bins == b]) for b in sorted(np.unique(bins))])
        counts = np.array([np.sum(bins == b) for b in sorted(np.unique(bins))])
        grand = np.mean(x)
        ss_between = float(np.sum(counts * (means - grand) ** 2))
        ss_total = float(np.sum((x - grand) ** 2) + 1e-12)
        eta2 = ss_between / ss_total
        rows.append({
            'feature': str(name), 'clean_feature': clean_feature_name(name),
            'spearman_rho': float(rho), 'spearman_abs': float(abs(rho)), 'spearman_p': float(p),
            'mutual_information': float(mi[j]), 'anova_F': float(F), 'anova_p': float(ap),
            'trajectory_eta2': float(eta2), 'variance': float(np.var(x))
        })
    df = pd.DataFrame(rows)
    for col in ['spearman_abs', 'mutual_information', 'trajectory_eta2']:
        c = df[col].to_numpy()
        df[col + '_z01'] = (c - c.min()) / (c.max() - c.min() + 1e-12)
    df['dynamic_score'] = 0.45*df['spearman_abs_z01'] + 0.35*df['mutual_information_z01'] + 0.20*df['trajectory_eta2_z01']
    return df.sort_values('dynamic_score', ascending=False).reset_index(drop=True), bins


def subset_representation(Xs, idx):
    n_comp = min(10, len(idx), Xs.shape[0]-1)
    return PCA(n_components=n_comp, random_state=1729).fit_transform(Xs[:, idx])


def neighbor_overlap(Z_ref, Z_sub, k=15):
    k = min(k, len(Z_ref)-1)
    ref = NearestNeighbors(n_neighbors=k+1).fit(Z_ref).kneighbors(return_distance=False)[:, 1:]
    sub = NearestNeighbors(n_neighbors=k+1).fit(Z_sub).kneighbors(return_distance=False)[:, 1:]
    vals = []
    for a, b in zip(ref, sub):
        vals.append(len(set(a).intersection(set(b))) / k)
    return float(np.mean(vals))


def validate_sets(Xs, Z_ref, ref_pt, feature_scores, feature_names, annotated_age, selected_idx, sizes=(5,10,15,20,30), n_random=50):
    var_rank = np.argsort(-np.var(Xs, axis=0))
    dyn_rank = feature_scores['feature'].map({f:i for i,f in enumerate(feature_names)}).to_numpy()
    root_by_age = int(np.argmin(annotated_age))
    rows=[]
    for size in sizes:
        if size > Xs.shape[1]:
            continue
        sets=[]
        sets.append(('dynamic', dyn_rank[:size]))
        sets.append(('high_variance', var_rank[:size]))
        rand_metrics=[]
        for r in range(n_random):
            sets.append((f'random_{r:02d}', RNG.choice(Xs.shape[1], size=size, replace=False)))
        for method, idx in sets:
            Z = subset_representation(Xs, idx)
            pt = graph_pseudotime(Z, root_by_age, k=15)
            rho_ref, _ = spearmanr(ref_pt, pt)
            rho_age, _ = spearmanr(annotated_age, pt)
            tw = trustworthiness(Z_ref, Z, n_neighbors=min(15, len(Z)-1))
            no = neighbor_overlap(Z_ref, Z, k=15)
            rows.append({'method': method, 'method_family': method.split('_')[0] if method.startswith('random') else method,
                         'n_features': int(size), 'spearman_with_reference_pseudotime': float(rho_ref),
                         'abs_spearman_with_reference_pseudotime': float(abs(rho_ref)),
                         'spearman_with_annotated_age': float(rho_age),
                         'trustworthiness_vs_full_PCA': float(tw), 'neighbor_overlap_vs_full_PCA': float(no),
                         'features': ';'.join([str(feature_names[i]) for i in idx])})
    df = pd.DataFrame(rows)
    return df


def main():
    adata = ad.read_h5ad(DATA)
    X = load_matrix(adata)
    feature_names = np.asarray(adata.var_names.astype(str))
    obs = adata.obs.copy()
    Xs = standardize(X)
    age = obs['annotated_age'].astype(float).to_numpy() if 'annotated_age' in obs else np.arange(adata.n_obs)

    # Full-feature reference: PCA -> KNN graph geodesic pseudotime rooted at earliest observed age.
    n_pc = min(30, Xs.shape[1], Xs.shape[0]-1)
    pca = PCA(n_components=n_pc, random_state=1729)
    Z_ref = pca.fit_transform(Xs)
    root_idx = int(np.argmin(age))
    ref_pt = graph_pseudotime(Z_ref[:, :min(15, n_pc)], root_idx, k=20)
    emb, emb_method = trajectory_embedding(Z_ref[:, :min(20, n_pc)])

    scores, bins = feature_dynamic_scores(Xs, feature_names, ref_pt, n_bins=8)
    top_n = min(20, Xs.shape[1])
    selected = scores.head(top_n).copy()
    selected_idx = np.array([np.where(feature_names == f)[0][0] for f in selected['feature']])

    validation = validate_sets(Xs, Z_ref[:, :min(20, n_pc)], ref_pt, scores, feature_names, age, selected_idx)
    # Summary table: dynamic/high_variance exact rows, random mean±sd per size.
    summary_rows=[]
    for (fam, n), g in validation.groupby(['method_family','n_features']):
        metrics=['abs_spearman_with_reference_pseudotime','spearman_with_annotated_age','trustworthiness_vs_full_PCA','neighbor_overlap_vs_full_PCA']
        row={'method_family':fam,'n_features':int(n),'n_replicates':int(len(g))}
        for m in metrics:
            row[m+'_mean']=float(g[m].mean())
            row[m+'_sd']=float(g[m].std(ddof=1) if len(g)>1 else 0.0)
        summary_rows.append(row)
    comp = pd.DataFrame(summary_rows).sort_values(['n_features','method_family'])

    # Profiles for top features by pseudotime bins.
    bin_labels = pd.qcut(ref_pt, q=8, labels=[f'Q{i+1}' for i in range(8)], duplicates='drop')
    prof = pd.DataFrame({'trajectory_bin': bin_labels, 'pseudotime': ref_pt, 'annotated_age': age})
    for f in selected['feature'].head(12):
        prof[f] = Xs[:, np.where(feature_names==f)[0][0]]
    bin_profile = prof.groupby('trajectory_bin', observed=True).mean(numeric_only=True).reset_index()

    # Data overview.
    overview={
        'n_cells': int(adata.n_obs), 'n_features': int(adata.n_vars),
        'obs_columns': list(map(str, adata.obs.columns)), 'var_columns': list(map(str, adata.var.columns)),
        'layers': list(map(str, adata.layers.keys())), 'embedding_method': emb_method,
        'reference_pca_explained_variance_ratio_first10': [float(x) for x in pca.explained_variance_ratio_[:10]],
        'reference_pca_cumulative_variance_30': float(np.sum(pca.explained_variance_ratio_)),
        'pseudotime_spearman_annotated_age': float(spearmanr(ref_pt, age).statistic),
        'phase_counts': obs['phase'].astype(str).value_counts().to_dict() if 'phase' in obs else {},
        'state_counts': obs['state'].astype(str).value_counts().to_dict() if 'state' in obs else {},
        'batch_counts': obs['batch'].astype(str).value_counts().to_dict() if 'batch' in obs else {}
    }

    # Save tables.
    pd.DataFrame({'cell': adata.obs_names.astype(str), 'reference_pseudotime': ref_pt, 'trajectory_bin': np.asarray(bin_labels).astype(str), 'embedding_1': emb[:,0], 'embedding_2': emb[:,1], 'annotated_age': age, **{c: obs[c].astype(str).to_numpy() for c in obs.columns if c != 'annotated_age'}}).to_csv(OUT/'cell_trajectory.csv', index=False)
    scores.to_csv(OUT/'feature_scores.csv', index=False)
    selected.to_csv(OUT/'selected_features.csv', index=False)
    validation.to_csv(OUT/'validation_metrics.csv', index=False)
    comp.to_csv(OUT/'method_comparison_table.csv', index=False)
    bin_profile.to_csv(OUT/'bin_feature_profiles.csv', index=False)
    with open(OUT/'data_overview.json','w') as f: json.dump(overview, f, indent=2)

    # Figures.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    if 'phase' in obs:
        obs['phase'].astype(str).value_counts().reindex(['G0','G1','S','G2']).dropna().plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'))
        axes[0].set_title('Cell-cycle phase')
    axes[0].set_ylabel('Cells')
    sns.histplot(age, bins=30, ax=axes[1], color='#4c78a8')
    axes[1].set_title('Annotated age distribution')
    axes[1].set_xlabel('Annotated age')
    axes[2].plot(np.arange(1, min(30, len(pca.explained_variance_ratio_))+1), np.cumsum(pca.explained_variance_ratio_[:30]), marker='o', ms=3)
    axes[2].set_title('PCA cumulative variance')
    axes[2].set_xlabel('PCs'); axes[2].set_ylabel('Variance explained')
    fig.tight_layout(); fig.savefig(IMG/'fig1_data_overview.png', dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    sc=axes[0].scatter(emb[:,0], emb[:,1], c=ref_pt, s=9, cmap='viridis', linewidths=0)
    axes[0].set_title(f'{emb_method} colored by reference pseudotime')
    plt.colorbar(sc, ax=axes[0], fraction=0.046, label='pseudotime')
    sc=axes[1].scatter(emb[:,0], emb[:,1], c=age, s=9, cmap='magma', linewidths=0)
    axes[1].set_title('Same embedding colored by annotated age')
    plt.colorbar(sc, ax=axes[1], fraction=0.046, label='age')
    if 'phase' in obs:
        palette=dict(zip(sorted(obs['phase'].astype(str).unique()), sns.color_palette('tab10', n_colors=obs['phase'].astype(str).nunique())))
        for ph, sub in pd.DataFrame({'x':emb[:,0],'y':emb[:,1],'phase':obs['phase'].astype(str)}).groupby('phase'):
            axes[2].scatter(sub.x, sub.y, s=8, label=ph, color=palette[ph], linewidths=0, alpha=.8)
        axes[2].legend(markerscale=2, fontsize=7, frameon=False)
    axes[2].set_title('Phase structure')
    for ax in axes: ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
    fig.tight_layout(); fig.savefig(IMG/'fig2_trajectory_embedding.png', dpi=220); plt.close(fig)

    top12 = selected['feature'].head(12).tolist()
    dyn = bin_profile[['trajectory_bin'] + top12].melt(id_vars='trajectory_bin', var_name='feature', value_name='z_expression')
    dyn['clean_feature'] = dyn['feature'].map(clean_feature_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=dyn, x='trajectory_bin', y='z_expression', hue='clean_feature', marker='o', ax=ax, linewidth=1.4)
    ax.axhline(0, color='black', lw=.6, alpha=.5)
    ax.set_title('Top dynamic features vary coherently along trajectory bins')
    ax.set_xlabel('Reference pseudotime bin'); ax.set_ylabel('Mean standardized expression')
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(IMG/'fig3_feature_dynamics.png', dpi=220); plt.close(fig)

    plot_comp = comp[comp['method_family'].isin(['dynamic','high_variance','random'])].copy()
    name_map={'dynamic':'Dynamic selected','high_variance':'High variance','random':'Random'}
    plot_comp['Method']=plot_comp['method_family'].map(name_map)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    metrics=[('abs_spearman_with_reference_pseudotime_mean','|Spearman| with full pseudotime'),('trustworthiness_vs_full_PCA_mean','Trustworthiness vs full PCA'),('neighbor_overlap_vs_full_PCA_mean','KNN overlap vs full PCA')]
    for ax,(m,label) in zip(axes, metrics):
        sns.lineplot(data=plot_comp, x='n_features', y=m, hue='Method', marker='o', ax=ax)
        # error bars for random
        for _, row in plot_comp.iterrows():
            sd_col=m.replace('_mean','_sd')
            if row['method_family']=='random':
                ax.errorbar(row['n_features'], row[m], yerr=row[sd_col], fmt='none', ecolor='gray', alpha=.5, capsize=2)
        ax.set_title(label); ax.set_xlabel('Number of features'); ax.set_ylabel(label)
    fig.tight_layout(); fig.savefig(IMG/'fig4_validation_comparison.png', dpi=220); plt.close(fig)

    # Additional heatmap of scores/top features for interpretability.
    heat = bin_profile.set_index('trajectory_bin')[top12].rename(columns={f: clean_feature_name(f) for f in top12})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(heat.T, cmap='vlag', center=0, ax=ax, cbar_kws={'label':'mean z-expression'})
    ax.set_title('Selected feature trajectory-bin heatmap')
    ax.set_xlabel('Pseudotime bin'); ax.set_ylabel('Selected feature')
    fig.tight_layout(); fig.savefig(IMG/'fig5_selected_feature_heatmap.png', dpi=220); plt.close(fig)

    # Claim recovery table.
    best20 = comp[(comp.n_features==20)].copy()
    dyn20 = best20[best20.method_family=='dynamic'].iloc[0]
    hv20 = best20[best20.method_family=='high_variance'].iloc[0]
    rnd20 = best20[best20.method_family=='random'].iloc[0]
    claims = [
        {'claim':'The dataset contains 2,759 cells and 241 protein imaging features with annotated age, phase, state, and batch metadata.', 'supporting_artifact':'outputs/data_overview.json; report/images/fig1_data_overview.png'},
        {'claim':f'The full-feature reference trajectory is biologically oriented: pseudotime Spearman correlation with annotated age = {overview["pseudotime_spearman_annotated_age"]:.3f}.', 'supporting_artifact':'outputs/cell_trajectory.csv; report/images/fig2_trajectory_embedding.png'},
        {'claim':f'The top 20 dynamic features preserve the reference pseudotime with mean |Spearman| {dyn20.abs_spearman_with_reference_pseudotime_mean:.3f}, compared with {hv20.abs_spearman_with_reference_pseudotime_mean:.3f} for high-variance features and {rnd20.abs_spearman_with_reference_pseudotime_mean:.3f}±{rnd20.abs_spearman_with_reference_pseudotime_sd:.3f} for random panels.', 'supporting_artifact':'outputs/method_comparison_table.csv; report/images/fig4_validation_comparison.png'},
        {'claim':'Selected molecules show interpretable, smooth expression programs across trajectory bins.', 'supporting_artifact':'outputs/bin_feature_profiles.csv; report/images/fig3_feature_dynamics.png; report/images/fig5_selected_feature_heatmap.png'}]
    pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv', index=False)

    # Update target inventory statuses.
    inventory={
      'primary_quantitative_answers':[
        {'artifact':'outputs/selected_features.csv','status':'satisfied','description':'ranked selected dynamic feature subset'},
        {'artifact':'outputs/validation_metrics.csv','status':'satisfied','description':'trajectory preservation metrics vs baselines'},
        {'artifact':'outputs/data_overview.json','status':'satisfied','description':'cells, features, obs/var fields, labels'}],
      'required_comparison_tables':[{'artifact':'outputs/method_comparison_table.csv','status':'satisfied'}],
      'expected_figures':[
        {'artifact':'report/images/fig1_data_overview.png','status':'satisfied'},
        {'artifact':'report/images/fig2_trajectory_embedding.png','status':'satisfied'},
        {'artifact':'report/images/fig3_feature_dynamics.png','status':'satisfied'},
        {'artifact':'report/images/fig4_validation_comparison.png','status':'satisfied'},
        {'artifact':'report/images/fig5_selected_feature_heatmap.png','status':'satisfied'}],
      'interpretability_artifacts':[{'artifact':'outputs/feature_scores.csv','status':'satisfied','description':'per-feature dynamic scores and components'}],
      'subgroup_outputs':[{'artifact':'outputs/bin_feature_profiles.csv','status':'satisfied','description':'mean selected feature profiles by trajectory bin'}]
    }
    with open(OUT/'target_artifact_inventory.json','w') as f: json.dump(inventory,f,indent=2)
    print(json.dumps({'status':'ok','n_cells':adata.n_obs,'n_features':adata.n_vars,'top_features':selected['clean_feature'].head(10).tolist(),'embedding_method':emb_method}, indent=2))

if __name__ == '__main__':
    main()
