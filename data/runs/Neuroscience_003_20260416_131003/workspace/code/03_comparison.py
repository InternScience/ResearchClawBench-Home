import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
import os

adata_full = ad.read_h5ad("outputs/adata_full_umap.h5ad")
adata_subset = ad.read_h5ad("outputs/adata_subset_umap.h5ad")

# Calculate Silhouette Score for Phase and Batch in Full vs Subset
def calc_silhouette(adata, label):
    # Only use cells where label is not nan
    mask = ~adata.obs[label].isna()
    if mask.sum() == 0:
        return np.nan
    X = adata.obsm['X_umap'][mask]
    labels = adata.obs[label][mask].astype(str)
    return silhouette_score(X, labels)

res = {
    'Full_Features': {
        'Phase_Silhouette': calc_silhouette(adata_full, 'phase'),
        'Batch_Silhouette': calc_silhouette(adata_full, 'batch')
    },
    'Selected_Features': {
        'Phase_Silhouette': calc_silhouette(adata_subset, 'phase'),
        'Batch_Silhouette': calc_silhouette(adata_subset, 'batch')
    }
}

df_res = pd.DataFrame(res).T
print("Silhouette Scores:")
print(df_res)
df_res.to_csv('outputs/silhouette_scores.csv')

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sc.pl.umap(adata_full, color='phase', ax=axes[0,0], show=False)
axes[0,0].set_title('Full Features: Phase')
sc.pl.umap(adata_full, color='annotated_age', ax=axes[0,1], show=False)
axes[0,1].set_title('Full Features: Annotated Age')

sc.pl.umap(adata_subset, color='phase', ax=axes[1,0], show=False)
axes[1,0].set_title('Selected Features: Phase')
sc.pl.umap(adata_subset, color='annotated_age', ax=axes[1,1], show=False)
axes[1,1].set_title('Selected Features: Annotated Age')

plt.tight_layout()
plt.savefig('report/images/umap_comparison.png', dpi=300)
plt.close()

# Also compare batch effects
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_full, color='batch', ax=axes[0], show=False)
axes[0].set_title('Full Features: Batch')
sc.pl.umap(adata_subset, color='batch', ax=axes[1], show=False)
axes[1].set_title('Selected Features: Batch')
plt.tight_layout()
plt.savefig('report/images/batch_comparison.png', dpi=300)
plt.close()

print("Comparison completed.")
