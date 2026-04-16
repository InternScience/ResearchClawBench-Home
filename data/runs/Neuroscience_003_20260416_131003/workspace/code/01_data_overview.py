import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
adata = ad.read_h5ad("data/adata_RPE.h5ad")

# Save summary stats
summary = {
    'n_cells': adata.n_obs,
    'n_features': adata.n_vars,
    'phases': adata.obs['phase'].value_counts().to_dict(),
    'states': adata.obs['state'].value_counts().to_dict(),
    'batches': adata.obs['batch'].value_counts().to_dict()
}
import json
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

# Plot 1: Age distribution by phase and state
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=adata.obs, x='phase', y='annotated_age', order=['G0', 'G1', 'S', 'G2'], ax=ax[0])
ax[0].set_title('Annotated Age by Cell Cycle Phase')
sns.boxplot(data=adata.obs, x='state', y='annotated_age', ax=ax[1])
ax[1].set_title('Annotated Age by Cell State')
plt.tight_layout()
plt.savefig('report/images/age_distribution.png', dpi=300)
plt.close()

# Plot 2: UMAP of full dataset
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata, color='phase', ax=axes[0], show=False)
axes[0].set_title('UMAP by Phase')
sc.pl.umap(adata, color='annotated_age', ax=axes[1], show=False)
axes[1].set_title('UMAP by Annotated Age')
sc.pl.umap(adata, color='batch', ax=axes[2], show=False)
axes[2].set_title('UMAP by Batch')
plt.tight_layout()
plt.savefig('report/images/umap_full_features.png', dpi=300)
plt.close()

# Save full adata with UMAP
adata.write_h5ad("outputs/adata_full_umap.h5ad")
print("Data overview completed.")
