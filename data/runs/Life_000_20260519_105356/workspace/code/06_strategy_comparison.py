#!/usr/bin/env python3
"""
Compare SMBO strategies across rounds.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")

# Load data
df_ei = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_pred = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')
df_ei['ML'] = df_ei['ML'].ffill()
df_pred['ML'] = df_pred['ML'].ffill()
df_ei['Glass (kPa)_max'] = pd.to_numeric(df_ei['Glass (kPa)_max'], errors='coerce')
df_pred['Glass (kPa)_max'] = pd.to_numeric(df_pred['Glass (kPa)_max'], errors='coerce')
monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
for m in monomers:
    df_pred[m] = pd.to_numeric(df_pred[m], errors='coerce')
    df_ei[m] = pd.to_numeric(df_ei[m], errors='coerce')
df_ei = df_ei.dropna(subset=['Glass (kPa)_max'])
df_pred = df_pred.dropna(subset=['Glass (kPa)_max'] + monomers)

# Normalize method names
def clean_method(ml):
    m = str(ml).strip()
    if 'RFR-GP' in m:
        return 'RFR-GP'
    if 'GP-GP' in m:
        return 'GP-GP'
    if 'old-SM-GP' in m:
        return 'old-SM-GP'
    if 'SM-ETR' in m:
        return 'SM-ETR'
    if 'SM-GBM' in m:
        return 'SM-GBM'
    return m

df_pred['Method'] = df_pred['ML'].apply(clean_method)

# Aggregate by method
method_stats = df_pred.groupby('Method')['Glass (kPa)_max'].agg(['max', 'mean', 'std', 'count']).reset_index()
method_stats = method_stats.sort_values('max', ascending=False)
print(method_stats)

# Figure 17: Method comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = method_stats['Method'].values
max_vals = method_stats['max'].values
mean_vals = method_stats['mean'].values
std_vals = method_stats['std'].values

x_pos = np.arange(len(methods))
ax.bar(x_pos, max_vals, yerr=std_vals, capsize=4, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel('Predicted Strength (kPa)')
ax.set_title('Max Predicted Strength by SMBO Method')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7)

ax = axes[1]
# Violin plot
sns.violinplot(data=df_pred, x='Method', y='Glass (kPa)_max', ax=ax, palette='Set2', inner='box')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Predicted Strength (kPa)')
ax.set_title('Distribution of Predicted Strength by Method')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("report/images/fig17_method_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig17_method_comparison.png")

# Save method stats
method_stats.to_csv("outputs/method_stats.csv", index=False)
print("Saved outputs/method_stats.csv")
