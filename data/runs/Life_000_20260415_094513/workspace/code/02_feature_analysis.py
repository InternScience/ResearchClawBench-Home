"""
Phase 2: Feature Analysis & Visualization
Correlation analysis, feature distributions, and key insights.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load training data
df = pd.read_csv('outputs/training_data_184.csv')
monomer_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target_col = 'Glass (kPa)_10s'

# ============================================================
# Figure 1: Target distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df[target_col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Adhesive Strength on Glass (kPa)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Adhesive Strength', fontsize=13)
axes[0].axvline(x=1000, color='red', linestyle='--', label='1 MPa target')
axes[0].legend()

# Log scale
axes[1].hist(np.log10(df[target_col]+1), bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('log10(Adhesive Strength + 1) (kPa)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of log10(Adhesive Strength)', fontsize=13)

plt.tight_layout()
plt.savefig('report/images/fig1_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Monomer composition distributions
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
short_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']
for i, (col, sname) in enumerate(zip(monomer_cols, short_names)):
    ax = axes[i//3, i%3]
    ax.hist(df[col], bins=25, color=f'C{i}', edgecolor='white', alpha=0.8)
    ax.set_xlabel(f'{sname} fraction', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{col}', fontsize=11)
plt.suptitle('Monomer Composition Distributions (n=184)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig2_monomer_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Correlation heatmap
# ============================================================
corr_cols = monomer_cols + [target_col]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=short_names + ['Glass_kPa'],
            yticklabels=short_names + ['Glass_kPa'],
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Matrix: Monomer Compositions & Adhesive Strength', fontsize=13)
plt.tight_layout()
plt.savefig('report/images/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# Print correlations with target
print("\nCorrelations with Glass (kPa)_10s:")
for col in monomer_cols:
    print(f"  {col}: {corr_matrix.loc[col, target_col]:.3f}")

# ============================================================
# Figure 4: Scatter plots of each monomer vs adhesive strength
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (col, sname) in enumerate(zip(monomer_cols, short_names)):
    ax = axes[i//3, i%3]
    ax.scatter(df[col], df[target_col], alpha=0.5, s=30, c=f'C{i}')
    ax.set_xlabel(f'{sname} fraction', fontsize=11)
    ax.set_ylabel('Glass (kPa)', fontsize=11)
    ax.set_title(f'{col} vs Adhesive Strength', fontsize=11)
    # Add trend line
    z = np.polyfit(df[col], df[target_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7)
plt.suptitle('Monomer Fractions vs Adhesive Strength', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_monomer_vs_strength.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Pairplot of monomer compositions
# ============================================================
plot_df = df[monomer_cols].copy()
plot_df.columns = short_names
g = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 15})
g.fig.suptitle('Pairwise Monomer Composition Relationships', y=1.02, fontsize=14)
plt.savefig('report/images/fig5_monomer_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ============================================================
# Save correlation data
# ============================================================
corr_matrix.to_csv('outputs/correlation_matrix.csv')
print("\nPhase 2 complete.")
