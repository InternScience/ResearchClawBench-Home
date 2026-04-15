import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Load initial verified data
path_initial = 'data/184_verified_Original Data_ML_20230926.xlsx'
df_initial = pd.read_excel(path_initial)

# Features
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

# Compute Glass_max
df_initial['Glass_10s'] = pd.to_numeric(df_initial['Glass (kPa)_10s'], errors='coerce')
df_initial['Glass_60s'] = pd.to_numeric(df_initial['Glass (kPa)_60s'], errors='coerce')
df_initial['Glass_max'] = df_initial[['Glass_10s', 'Glass_60s']].max(axis=1)

# Clean: drop rows with NaN in features or target
df_initial_clean = df_initial[features + ['Glass_max']].dropna()

# Check sum
df_initial_clean['sum_comps'] = df_initial_clean[features].sum(axis=1)
print('Comp sum mean:', df_initial_clean['sum_comps'].mean())
print('Comp sum std:', df_initial_clean['sum_comps'].std())
df_initial_clean = df_initial_clean[df_initial_clean['sum_comps'].between(0.99, 1.01)]
df_initial_clean = df_initial_clean[features + ['Glass_max']]

# Save processed
df_initial_clean.to_csv('outputs/initial_data_processed.csv', index=False)
print('Saved initial_data_processed.csv shape:', df_initial_clean.shape)

# Summary stats
summary = {
    'n_samples': len(df_initial_clean),
    'features': features,
    'target_mean': float(df_initial_clean['Glass_max'].mean()),
    'target_std': float(df_initial_clean['Glass_max'].std()),
    'target_max': float(df_initial_clean['Glass_max'].max()),
    'comp_sums': float(df_initial_clean[features].sum().sum())
}
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f)
print('Summary saved')

# Plots
Path('report/images').mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8')

# Hist target
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.hist(df_initial_clean['Glass_max'], bins=30)
plt.xlabel('Glass_max (kPa)')
plt.ylabel('Count')
plt.title('Target Distribution')

# Comp sums
plt.subplot(122)
plt.hist(df_initial_clean['sum_comps'], bins=30)
plt.xlabel('Sum Compositions')
plt.ylabel('Count')
plt.title('Comp Sum Check')
plt.savefig('report/images/data_overview_hist.png', dpi=300, bbox_inches='tight')
plt.close()

# Pairplot comps vs target
sns.pairplot(df_initial_clean[features + ['Glass_max']], diag_kind='hist')
plt.savefig('report/images/comp_target_pairs.png', dpi=300, bbox_inches='tight')
plt.close()

# Parallel coordinates (normalized)
from pandas.plotting import parallel_coordinates
df_norm = (df_initial_clean[features] - df_initial_clean[features].min()) / (df_initial_clean[features].max() - df_initial_clean[features].min())
df_norm['Glass_max'] = df_initial_clean['Glass_max']
df_norm['cluster'] = pd.qcut(df_initial_clean['Glass_max'], q=3, labels=['Low', 'Med', 'High'])
plt.figure(figsize=(12,6))
parallel_coordinates(df_norm, 'cluster', colormap='viridis')
plt.title('Parallel Coord: High vs Low Adhesion')
plt.savefig('report/images/parallel_coords.png', dpi=300, bbox_inches='tight')
plt.close()

print('Plots saved')