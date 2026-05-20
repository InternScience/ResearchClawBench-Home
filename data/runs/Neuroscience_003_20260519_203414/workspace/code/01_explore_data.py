"""
Data exploration and preprocessing for single-cell protein imaging data.
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sc.settings.verbosity = 3

# Load data
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"Data shape: {adata.shape}")
print(f"Observations: {adata.obs.columns.tolist()}")
print(f"States: {adata.obs['state'].value_counts().to_dict()}")
print(f"Phases: {adata.obs['phase'].value_counts().to_dict()}")

# Basic preprocessing
adata.layers['scaled'] = ((adata.X - adata.X.mean(axis=0)) / (adata.X.std(axis=0) + 1e-8))

# Save processed data
os.makedirs('outputs', exist_ok=True)
adata.write('outputs/adata_processed.h5ad')
print("Saved processed data to outputs/adata_processed.h5ad")
