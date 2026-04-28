"""
Trajectory-preserving feature selection for 4i RPE data.
Step 1: Load data, basic preprocessing, data overview figure.
"""
import os, json
import numpy as np, pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WS, 'data', 'adata_RPE.h5ad')
OUT = os.path.join(WS, 'outputs')
IMG = os.path.join(WS, 'report', 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

sc.settings.verbosity = 1
np.random.seed(0)

a = ad.read_h5ad(DATA)
# state has 'nan' string for some cells; keep but treat as 'unknown'
a.obs['state'] = a.obs['state'].astype(str).replace({'nan': 'unknown'})
a.obs['phase'] = a.obs['phase'].astype(str)
a.obs['batch'] = a.obs['batch'].astype(str)
print(a)

# Parse var_names into protein / measurement / compartment
def parse(v):
    parts = v.split('_')
    # measurement = first 2 tokens (Int_Intg / Int_MeanEdge / Int_Med / Int_Std / AreaShape_Area)
    meas = '_'.join(parts[:2])
    # compartment = last token
    comp = parts[-1]
    # protein = remaining middle (single token usually)
    prot = '_'.join(parts[2:-1]) if len(parts) > 3 else parts[-2]
    return pd.Series({'measurement': meas, 'protein': prot, 'compartment': comp})
meta = pd.DataFrame([parse(v) for v in a.var_names], index=a.var_names)
a.var = meta
a.var.to_csv(os.path.join(OUT, 'feature_metadata.csv'))
print(a.var.head())
print('proteins:', a.var.protein.nunique())
print('compartments:', a.var.compartment.unique())
print('measurements:', a.var.measurement.unique())

# Save a copy of preprocessed adata (already log/scaled style values in X based on stats; raw layer present)
a.write_h5ad(os.path.join(OUT, 'adata_parsed.h5ad'))

# === Data overview figure ===
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
# (a) phase counts
sns.countplot(x='phase', data=a.obs, order=['G0','G1','S','G2'], ax=axes[0,0],
              palette='Set2')
axes[0,0].set_title('Cell-cycle phase distribution'); axes[0,0].set_ylabel('# cells')
# (b) state counts
sns.countplot(x='state', data=a.obs, ax=axes[0,1], palette='Set1')
axes[0,1].set_title('Cellular state'); axes[0,1].set_ylabel('# cells')
# (c) batch counts
sns.countplot(x='batch', data=a.obs, ax=axes[0,2], palette='Set3')
axes[0,2].set_title('Batch'); axes[0,2].set_ylabel('# cells')
# (d) annotated_age histogram by phase
for ph, df in a.obs.groupby('phase'):
    axes[1,0].hist(df['annotated_age'], bins=40, alpha=0.5, label=ph)
axes[1,0].set_xlabel('annotated_age (pseudotime)'); axes[1,0].set_ylabel('# cells')
axes[1,0].set_title('Pseudotime by phase'); axes[1,0].legend()
# (e) feature category breakdown
cm = a.var.groupby(['measurement','compartment']).size().unstack(fill_value=0)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Features: measurement × compartment')
# (f) global feature value distribution
axes[1,2].hist(a.X.ravel(), bins=80, color='gray')
axes[1,2].set_title('Feature value distribution (X)'); axes[1,2].set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(IMG, '01_data_overview.png'), dpi=140)
plt.close()
print('saved 01_data_overview.png')
