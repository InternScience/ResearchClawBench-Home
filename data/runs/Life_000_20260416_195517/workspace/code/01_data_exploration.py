#!/usr/bin/env python3
"""
Phase 1: Data Exploration and Visualization
Bio-inspired Hydrogel Adhesive Strength Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_000_20260416_195517"

# ========== Load Data ==========
print("Loading datasets...")

# Primary training data (184 verified)
df_train = pd.read_excel(f"{BASE}/data/184_verified_Original Data_ML_20230926.xlsx", sheet_name='Data_to_HU')

# Optimization data
df_opt_ei = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_opt_pred = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')

# Earlier optimization data
df_opt2_ei = pd.read_excel(f"{BASE}/data/ML_ei&pred_20240213.xlsx", sheet_name='EI')
df_opt2_pred = pd.read_excel(f"{BASE}/data/ML_ei&pred_20240213.xlsx", sheet_name='PRED')

features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target = 'Glass (kPa)_10s'

print(f"Training data: {df_train.shape[0]} samples")
print(f"Optimization EI (3 rounds): {df_opt_ei.shape[0]} entries")
print(f"Optimization PRED (3 rounds): {df_opt_pred.shape[0]} entries")

# Save summary stats
stats = df_train[features + [target]].describe()
stats.to_csv(f"{BASE}/outputs/training_data_stats.csv")
print("\nTraining data statistics saved.")

# ========== Figure 1: Monomer Composition Distribution ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#795548']
monomer_names = ['Nucleophilic\n(HEA)', 'Hydrophobic\n(BA)', 'Acidic\n(CBEA)', 
                 'Cationic\n(ATAC)', 'Aromatic\n(PEA)', 'Amide\n(AAm)']

for i, (feat, name, color) in enumerate(zip(features, monomer_names, colors)):
    ax = axes[i//3, i%3]
    ax.hist(df_train[feat], bins=25, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(f'{name} Fraction')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}')
    ax.axvline(df_train[feat].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={df_train[feat].mean():.3f}')
    ax.legend(fontsize=9)

plt.suptitle('Distribution of Monomer Compositions in Training Data (n=184)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig1_monomer_distributions.png")
plt.close()
print("Figure 1 saved: monomer distributions")

# ========== Figure 2: Adhesive Strength Distribution ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(df_train[target], bins=30, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(df_train[target].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={df_train[target].mean():.1f} kPa')
ax.axvline(df_train[target].median(), color='green', linestyle='--', linewidth=2, label=f'Median={df_train[target].median():.1f} kPa')
ax.axvline(1000, color='orange', linestyle='-', linewidth=2, label='1 MPa Target')
ax.set_xlabel('Adhesive Strength (kPa)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Adhesive Strength')
ax.legend()

# Box plot by phase separation
ax = axes[1]
df_plot = df_train.copy()
df_plot['Phase'] = df_plot['Phase Seperation'].map({0: 'No Phase Sep.', 1: 'Phase Sep.'})
df_plot = df_plot.dropna(subset=['Phase'])
sns.boxplot(data=df_plot, x='Phase', y=target, ax=ax, palette=['#4CAF50', '#E91E63'])
ax.set_ylabel('Adhesive Strength (kPa)')
ax.set_title('Adhesive Strength by Phase Separation')

plt.suptitle('Adhesive Strength Analysis (Training Data)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig2_adhesive_strength_distribution.png")
plt.close()
print("Figure 2 saved: adhesive strength distribution")

# ========== Figure 3: Correlation Heatmap ==========
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = features + [target, 'Q', 'Phase Seperation']
corr_data = df_train[corr_cols].copy()
corr_data.columns = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm', 'Adhesion (kPa)', 'Q', 'Phase Sep.']
corr_matrix = corr_data.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, square=True, ax=ax, vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix: Monomer Compositions & Properties', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig3_correlation_heatmap.png")
plt.close()
print("Figure 3 saved: correlation heatmap")

# Save correlation matrix
corr_matrix.to_csv(f"{BASE}/outputs/correlation_matrix.csv")

# ========== Figure 4: Composition-Adhesion Scatter Plots ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, (feat, name, color) in enumerate(zip(features, monomer_names, colors)):
    ax = axes[i//3, i%3]
    scatter = ax.scatter(df_train[feat], df_train[target], c=df_train['Phase Seperation'],
                        cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.3, s=30)
    ax.set_xlabel(f'{name} Fraction')
    ax.set_ylabel('Adhesive Strength (kPa)')
    ax.set_title(f'Adhesion vs {name}')
    
    # Add trend line
    mask = ~(df_train[feat].isna() | df_train[target].isna())
    if mask.sum() > 2:
        z = np.polyfit(df_train.loc[mask, feat], df_train.loc[mask, target], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df_train[feat].min(), df_train[feat].max(), 100)
        ax.plot(x_range, p(x_range), 'r--', linewidth=1.5, alpha=0.7)

plt.suptitle('Monomer Composition vs Adhesive Strength\n(Color: Phase Separation, Green=No, Red=Yes)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig4_composition_vs_adhesion.png")
plt.close()
print("Figure 4 saved: composition vs adhesion scatter")

# ========== Figure 5: Ternary-like Composition Analysis ==========
# Top compositions analysis
df_sorted = df_train.sort_values(target, ascending=False)
top20 = df_sorted.head(20)
bottom20 = df_sorted.tail(20)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top 20 compositions
ax = axes[0]
top20_means = top20[features].mean()
ax.bar(range(len(features)), top20_means, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(features)))
ax.set_xticklabels(monomer_names, fontsize=9)
ax.set_ylabel('Mean Fraction')
ax.set_title(f'Top 20 Hydrogels\n(Mean Adhesion: {top20[target].mean():.0f} kPa)')
ax.set_ylim(0, 0.6)

# Bottom 20 compositions
ax = axes[1]
bot20_means = bottom20[features].mean()
ax.bar(range(len(features)), bot20_means, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(features)))
ax.set_xticklabels(monomer_names, fontsize=9)
ax.set_ylabel('Mean Fraction')
ax.set_title(f'Bottom 20 Hydrogels\n(Mean Adhesion: {bottom20[target].mean():.0f} kPa)')
ax.set_ylim(0, 0.6)

plt.suptitle('Monomer Composition: Top vs Bottom Performers', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig5_top_vs_bottom_compositions.png")
plt.close()
print("Figure 5 saved: top vs bottom compositions")

# Save top performers
top20[['No.'] + features + [target]].to_csv(f"{BASE}/outputs/top20_formulations.csv", index=False)

print("\n=== Phase 1 Complete ===")
print(f"Training data: {df_train.shape[0]} samples")
print(f"Features: {features}")
print(f"Target: {target}")
print(f"Adhesive strength range: {df_train[target].min():.1f} - {df_train[target].max():.1f} kPa")
print(f"Mean: {df_train[target].mean():.1f} kPa, Std: {df_train[target].std():.1f} kPa")
print(f"Phase separation: {(df_train['Phase Seperation']==1).sum()} yes, {(df_train['Phase Seperation']==0).sum()} no")
