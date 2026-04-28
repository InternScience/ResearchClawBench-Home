"""
Step 4: Visualization of feature scores and method comparison.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'outputs')
IMG = os.path.join(WS, 'report', 'images')

scores = pd.read_csv(os.path.join(OUT, 'feature_scores.csv'))
ev = pd.read_csv(os.path.join(OUT, 'evaluation_metrics.csv'))

# ===== Fig 02: feature score distributions =====
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
cols_show = [('variance', 'HVF (raw variance)'),
             ('spearman_abs_pseudotime', '|Spearman| with pseudotime'),
             ('anova_F_phase', 'ANOVA F (phase)'),
             ('laplacian_score_neg', '−Laplacian Score'),
             ('graph_smoothness', 'Graph smoothness (kNN)'),
             ('dyn_score', 'Composite DynScore')]
for ax, (col, title) in zip(axes.ravel(), cols_show):
    ax.hist(scores[col], bins=40, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(scores[col].quantile(0.9), color='red', ls='--', label='90th pct')
    ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel('# features')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, '02_feature_scores.png'), dpi=140)
plt.close()
print('saved 02')

# ===== Fig 02b: DynScore vs Spearman, with top-25 highlighted =====
fig, ax = plt.subplots(figsize=(7,6))
top25 = scores.sort_values('dyn_score', ascending=False).head(25)['feature'].tolist()
mask = scores['feature'].isin(top25)
ax.scatter(scores.loc[~mask, 'spearman_abs_pseudotime'], scores.loc[~mask, 'graph_smoothness'],
           c='lightgray', s=20, label='other')
ax.scatter(scores.loc[mask, 'spearman_abs_pseudotime'], scores.loc[mask, 'graph_smoothness'],
           c='crimson', s=40, label='top-25 DynScore')
for _, r in scores.loc[mask].head(12).iterrows():
    ax.annotate(r['feature'].replace('Int_','').replace('AreaShape_',''),
                (r['spearman_abs_pseudotime'], r['graph_smoothness']), fontsize=7)
ax.set_xlabel('|Spearman| with pseudotime'); ax.set_ylabel('Graph smoothness')
ax.set_title('Dynamic-feature plane: top-25 by DynScore')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, '02b_dynscore_plane.png'), dpi=140)
plt.close()
print('saved 02b')

# ===== Fig 07: bar charts of evaluation metrics across methods × k =====
methods_order = ['DynScore','Spearman_pseudotime','ANOVA_phase','HVF_variance',
                 'GraphSmoothness','LaplacianScore','Random']
ev_sub = ev[ev['method'].isin(methods_order)].copy()
metrics = [('knn_jaccard', 'kNN Jaccard preservation', False),
           ('dpt_spearman', '|Spearman| DPT vs annotated_age', False),
           ('phase_acc', 'kNN phase classification accuracy', False),
           ('silhouette', 'Phase silhouette in subset PCA', False)]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (col, title, _) in zip(axes.ravel(), metrics):
    pivot = ev_sub.pivot(index='method', columns='k', values=col).reindex(methods_order)
    pivot.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='k')
    full_val = ev[ev['method']=='Full'][col].values[0]
    ax.axhline(full_val, color='red', ls='--', label=f'Full ({full_val:.3f})')
    ax.set_title(title); ax.set_ylabel(col)
    ax.set_xticklabels(pivot.index, rotation=30, ha='right')
    ax.legend(title='k', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG, '07_method_comparison.png'), dpi=140)
plt.close()
print('saved 07')
