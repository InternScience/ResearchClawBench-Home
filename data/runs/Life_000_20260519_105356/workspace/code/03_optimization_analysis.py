#!/usr/bin/env python3
"""
Analyze optimization trajectory across rounds 1-3.
Compare SMBO strategies and track adhesive strength improvement.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")

# Load optimization data
df_ei = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_pred = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')

monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

# Forward-fill ML method names in EI sheet
df_ei['ML'] = df_ei['ML'].ffill()
# Forward-fill ML method names in PRED sheet
df_pred['ML'] = df_pred['ML'].ffill()

# Ensure target is numeric
df_ei['Glass (kPa)_max'] = pd.to_numeric(df_ei['Glass (kPa)_max'], errors='coerce')
df_pred['Glass (kPa)_max'] = pd.to_numeric(df_pred['Glass (kPa)_max'], errors='coerce')
df_ei = df_ei.dropna(subset=['Glass (kPa)_max'])
df_pred = df_pred.dropna(subset=['Glass (kPa)_max'])

print("EI methods:", df_ei['ML'].unique())
print("PRED methods:", df_pred['ML'].unique())

# Separate by round based on ML name suffix
def assign_round(ml_name):
    if '3rd' in str(ml_name):
        return 3
    elif '2rd' in str(ml_name) or '2nd' in str(ml_name):
        return 2
    else:
        return 1

df_ei['Round'] = df_ei['ML'].apply(assign_round)
df_pred['Round'] = df_pred['ML'].apply(assign_round)

print("\nEI round counts:")
print(df_ei['Round'].value_counts().sort_index())
print("\nPRED round counts:")
print(df_pred['Round'].value_counts().sort_index())

# Aggregate max predicted strength per method per round
pred_summary = df_pred.groupby(['Round', 'ML'])['Glass (kPa)_max'].agg(['max', 'mean', 'count']).reset_index()
print("\n=== PRED summary ===")
print(pred_summary)

# For EI, just show max values
ei_summary = df_ei.groupby(['Round', 'ML'])['Glass (kPa)_max'].agg(['max', 'mean', 'count']).reset_index()
print("\n=== EI summary ===")
print(ei_summary)

# Combine all predicted top formulations
df_pred_all = df_pred.copy()
df_pred_all = df_pred_all.sort_values('Glass (kPa)_max', ascending=False)
print("\n=== Top 10 predicted formulations ===")
print(df_pred_all[['ML', 'Round', 'NO.', 'Glass (kPa)_max'] + monomers].head(10))

# Save summaries
pred_summary.to_csv("outputs/pred_summary.csv", index=False)
ei_summary.to_csv("outputs/ei_summary.csv", index=False)
df_pred_all.to_csv("outputs/pred_all_sorted.csv", index=False)

# Figure 7: Optimization trajectory - predicted max strength by round and method
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PRED data
ax = axes[0]
for method in sorted(df_pred['ML'].unique()):
    sub = pred_summary[pred_summary['ML'] == method]
    if len(sub) > 0:
        ax.plot(sub['Round'], sub['max'], marker='o', label=method, linewidth=2, markersize=8)
ax.set_xlabel('Optimization Round')
ax.set_ylabel('Max Predicted Adhesive Strength (kPa)')
ax.set_title('PRED: Max Predicted Strength by Round')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7, label='1 MPa target')
ax.legend(fontsize=7, loc='upper left')
ax.set_xticks([1, 2, 3])

# EI data
ax = axes[1]
for method in sorted(df_ei['ML'].unique()):
    sub = ei_summary[ei_summary['ML'] == method]
    if len(sub) > 0:
        ax.plot(sub['Round'], sub['max'], marker='s', label=method, linewidth=2, markersize=8)
ax.set_xlabel('Optimization Round')
ax.set_ylabel('Max EI Adhesive Strength (kPa)')
ax.set_title('EI: Max Strength by Round')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7, label='1 MPa target')
ax.legend(fontsize=7, loc='upper left')
ax.set_xticks([1, 2, 3])

plt.tight_layout()
plt.savefig("report/images/fig7_optimization_trajectory.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig7_optimization_trajectory.png")

# Figure 8: Box plots of predicted strengths by round
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.boxplot(data=df_pred, x='Round', y='Glass (kPa)_max', ax=ax, palette='Set2')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7)
ax.set_title('PRED: Distribution by Round')
ax.set_ylabel('Predicted Strength (kPa)')

ax = axes[1]
sns.boxplot(data=df_ei, x='Round', y='Glass (kPa)_max', ax=ax, palette='Set3')
ax.axhline(1000, color='red', linestyle='--', alpha=0.7)
ax.set_title('EI: Distribution by Round')
ax.set_ylabel('EI Strength (kPa)')

plt.tight_layout()
plt.savefig("report/images/fig8_round_boxplots.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig8_round_boxplots.png")

# Figure 9: Top formulation compositions (radar/spider or stacked bar)
top_n = 10
top_pred = df_pred_all.head(top_n)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(top_n)
width = 0.12
colors = plt.cm.tab10(np.linspace(0, 1, len(monomers)))
for i, mon in enumerate(monomers):
    ax.bar(x + i*width, top_pred[mon].values, width, label=mon, color=colors[i], edgecolor='white')

ax.set_xticks(x + width * (len(monomers)-1)/2)
ax.set_xticklabels([f"{row['ML']}\nR{row['Round']}-#{int(row['NO.'])}" for _, row in top_pred.iterrows()], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Monomer Fraction')
ax.set_title(f'Top {top_n} Predicted Formulations - Composition')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("report/images/fig9_top_formulations.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig9_top_formulations.png")

# Track overall best per round
best_per_round = df_pred.groupby('Round')['Glass (kPa)_max'].max().reset_index()
best_per_round.columns = ['Round', 'Best_Predicted_kPa']
print("\nBest predicted per round:")
print(best_per_round)
best_per_round.to_csv("outputs/best_per_round.csv", index=False)
