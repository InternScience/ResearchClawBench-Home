#!/usr/bin/env python3
"""
Data exploration and cleaning for hydrogel adhesive strength prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# Load datasets
df184 = pd.read_excel("data/184_verified_Original Data_ML_20230926.xlsx", sheet_name='Data_to_HU')
df_ei = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_pred = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')

monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target = 'Glass (kPa)_10s'

# Clean 184 data: drop rows with missing target
df184_clean = df184.dropna(subset=[target]).copy()
print(f"184 data: {df184.shape} -> {df184_clean.shape} after dropping missing {target}")

# Check monomer sum
sum_comp = df184_clean[monomers].sum(axis=1)
print(f"Monomer sum stats: min={sum_comp.min():.4f}, max={sum_comp.max():.4f}, mean={sum_comp.mean():.4f}")

# Descriptive stats
print("\n=== Monomer composition stats ===")
print(df184_clean[monomers].describe())
print("\n=== Target stats ===")
print(df184_clean[target].describe())

# Save cleaned data
df184_clean.to_csv("outputs/df184_clean.csv", index=False)
print("Saved outputs/df184_clean.csv")

# Figure 1: Data overview - distribution of target and monomers
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Target distribution
ax = axes[0]
ax.hist(df184_clean[target], bins=30, color='steelblue', edgecolor='white')
ax.axvline(1000, color='red', linestyle='--', label='1 MPa target')
ax.set_xlabel('Glass Adhesive Strength (kPa)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Adhesive Strength')
ax.legend()

for i, mon in enumerate(monomers):
    ax = axes[i+1]
    ax.hist(df184_clean[mon], bins=30, color='darkgreen', edgecolor='white', alpha=0.7)
    ax.set_xlabel(f'{mon} fraction')
    ax.set_ylabel('Count')
    ax.set_title(f'{mon}')

plt.tight_layout()
plt.savefig("report/images/fig1_data_overview.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig1_data_overview.png")

# Figure 2: Correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = monomers + [target, 'Steel (kPa)_10s', 'Q', 'Modulus (kPa)']
corr = df184_clean[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax,
            vmin=-1, vmax=1, square=True)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("report/images/fig2_correlation_matrix.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig2_correlation_matrix.png")

# Figure 3: Scatter plots of each monomer vs target
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, mon in enumerate(monomers):
    ax = axes[i]
    ax.scatter(df184_clean[mon], df184_clean[target], alpha=0.6, c='steelblue', edgecolors='white', s=50)
    ax.set_xlabel(f'{mon} fraction')
    ax.set_ylabel('Glass Adhesive Strength (kPa)')
    ax.set_title(f'{mon} vs Adhesive Strength')
    ax.axhline(1000, color='red', linestyle='--', alpha=0.5)
    # Add trend line
    z = np.polyfit(df184_clean[mon], df184_clean[target], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df184_clean[mon].min(), df184_clean[mon].max(), 100)
    ax.plot(x_line, p(x_line), color='orange', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig("report/images/fig3_monomer_vs_target.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig3_monomer_vs_target.png")

# Save summary stats
summary = {
    "n_samples": int(df184_clean.shape[0]),
    "n_features": len(monomers),
    "target_mean_kPa": float(df184_clean[target].mean()),
    "target_std_kPa": float(df184_clean[target].std()),
    "target_max_kPa": float(df184_clean[target].max()),
    "target_min_kPa": float(df184_clean[target].min()),
    "n_above_1MPa": int((df184_clean[target] >= 1000).sum()),
    "monomer_sum_min": float(sum_comp.min()),
    "monomer_sum_max": float(sum_comp.max()),
}
import json
with open("outputs/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved outputs/data_summary.json")
