"""
Analysis Script 4: UMAP clustering summary and BIC scores analysis.
"""
import numpy as np
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Neuroscience_001_20260417_004607'
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report/images')
umap_dir = os.path.join(BASE, 'data/flow/0000/umap_and_clustering')

# Load all clustering results
clustering_summary = {}
for pkl_file in sorted(os.listdir(umap_dir)):
    if pkl_file.endswith('.pickle'):
        ct = pkl_file.replace('.pickle', '')
        with open(os.path.join(umap_dir, pkl_file), 'rb') as f:
            clust = pickle.load(f)
        
        n_clusters_range = clust.n_clusters
        scores = clust.scores
        labels = clust.labels
        n_opt = len(np.unique(labels))
        emb = clust.embedding._embedding
        
        clustering_summary[ct] = {
            'n_clusters_range': n_clusters_range.tolist() if hasattr(n_clusters_range, 'tolist') else list(n_clusters_range),
            'scores': [float(s) for s in scores],
            'optimal_n_clusters': n_opt,
            'n_models': len(labels),
            'embedding_shape': list(emb.shape),
        }

# Save
with open(os.path.join(OUT, 'clustering_summary.json'), 'w') as f:
    json.dump(clustering_summary, f, indent=2)

# Print summary
print("=== Clustering Summary ===")
print(f"{'Cell Type':15s} {'Optimal Clusters':>18s} {'Best BIC Score':>15s}")
print("-" * 50)
for ct in sorted(clustering_summary.keys()):
    info = clustering_summary[ct]
    best_score = info['scores'][0] if info['scores'] else 0
    print(f"{ct:15s} {info['optimal_n_clusters']:>18d} {best_score:>15.2f}")

# ============================================================
# Figure 20: Optimal number of clusters per cell type
# ============================================================
cts = sorted(clustering_summary.keys())
opt_clusters = [clustering_summary[ct]['optimal_n_clusters'] for ct in cts]

fig, ax = plt.subplots(figsize=(16, 6))
colors = []
for ct in cts:
    if ct.startswith('T4') or ct.startswith('T5'):
        colors.append('#ff7f00')
    elif ct.startswith('R'):
        colors.append('#e41a1c')
    elif ct.startswith('L') or ct in ['Am', 'C2', 'C3']:
        colors.append('#377eb8')
    elif ct.startswith('Mi'):
        colors.append('#4daf4a')
    elif ct.startswith('Tm') and not ct.startswith('TmY'):
        colors.append('#f781bf')
    elif ct.startswith('TmY'):
        colors.append('#999999')
    else:
        colors.append('#a65628')

ax.bar(range(len(cts)), opt_clusters, color=colors, edgecolor='gray', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(cts)))
ax.set_xticklabels(cts, rotation=90, fontsize=7)
ax.set_ylabel('Optimal Number of Clusters', fontsize=12)
ax.set_title('Optimal Number of Functional Clusters per Cell Type\n(from Gaussian Mixture Model on UMAP embeddings of 50 models)', fontsize=13)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'optimal_clusters.png'), dpi=150)
plt.close()
print("\nSaved: optimal_clusters.png")

# ============================================================
# Figure 21: BIC Score Curves for T4/T5 cells
# ============================================================
ds_cells = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

for idx, ct in enumerate(ds_cells):
    ax = axes[idx // 4, idx % 4]
    info = clustering_summary[ct]
    ax.plot(info['n_clusters_range'], info['scores'], 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Clusters', fontsize=10)
    ax.set_ylabel('BIC Score', fontsize=10)
    ax.set_title(f'{ct}', fontsize=12)
    # Mark optimal
    best_idx = np.argmax(info['scores'])
    ax.axvline(info['n_clusters_range'][best_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Optimal: {info["n_clusters_range"][best_idx]}')
    ax.legend(fontsize=9)

fig.suptitle('BIC Scores for Gaussian Mixture Clustering of T4/T5 Neurons', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'bic_scores_T4T5.png'), dpi=150)
plt.close()
print("Saved: bic_scores_T4T5.png")

# ============================================================
# Figure 22: Multi-cell-type functional diversity summary
# ============================================================
# Count cells with >1 cluster vs 1 cluster
multi_cluster = sum(1 for ct in clustering_summary if clustering_summary[ct]['optimal_n_clusters'] > 1)
single_cluster = sum(1 for ct in clustering_summary if clustering_summary[ct]['optimal_n_clusters'] == 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
axes[0].pie([single_cluster, multi_cluster], 
            labels=[f'Single cluster ({single_cluster})', f'Multiple clusters ({multi_cluster})'],
            colors=['#3498db', '#e74c3c'], autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Functional Diversity Across Cell Types', fontsize=13)

# Histogram of cluster counts
cluster_counts = [clustering_summary[ct]['optimal_n_clusters'] for ct in clustering_summary]
axes[1].hist(cluster_counts, bins=range(1, max(cluster_counts)+2), color='steelblue', 
             edgecolor='white', alpha=0.8, align='left')
axes[1].set_xlabel('Number of Functional Clusters', fontsize=12)
axes[1].set_ylabel('Number of Cell Types', fontsize=12)
axes[1].set_title('Distribution of Functional Cluster Counts', fontsize=13)
axes[1].set_xticks(range(1, max(cluster_counts)+1))

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'functional_diversity.png'), dpi=150)
plt.close()
print("Saved: functional_diversity.png")

print("\n=== Clustering analysis complete ===")
