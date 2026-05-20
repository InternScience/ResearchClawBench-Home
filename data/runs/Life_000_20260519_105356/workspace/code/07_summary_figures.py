#!/usr/bin/env python3
"""
Generate additional summary figures for the report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

np.random.seed(42)
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("outputs/df184_clean.csv")
monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

# Figure 18: Combined summary dashboard
fig = plt.figure(figsize=(18, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Target distribution with milestones
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['Glass (kPa)_10s'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
ax1.axvline(df['Glass (kPa)_10s'].mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean={df["Glass (kPa)_10s"].mean():.1f} kPa')
ax1.axvline(df['Glass (kPa)_10s'].max(), color='green', linestyle='-', linewidth=2, label=f'Max={df["Glass (kPa)_10s"].max():.1f} kPa')
ax1.set_xlabel('Glass Adhesive Strength (kPa)')
ax1.set_ylabel('Count')
ax1.set_title('(A) Adhesive Strength Distribution')
ax1.legend(fontsize=8)

# 2. Feature importance
ax2 = fig.add_subplot(gs[0, 1])
imp = pd.read_csv("outputs/feature_importance.csv").sort_values('Importance', ascending=True)
ax2.barh(imp['Feature'], imp['Importance'], color='darkgreen', edgecolor='white')
ax2.set_xlabel('Importance')
ax2.set_title('(B) Feature Importance (RFR)')

# 3. Model comparison
ax3 = fig.add_subplot(gs[0, 2])
results = {
    'RFR': {'r2': 0.700, 'rmse': 22.98},
    'GP': {'r2': 0.782, 'rmse': 19.54},
    'XGBoost': {'r2': 0.704, 'rmse': 23.31}
}
models = list(results.keys())
r2s = [results[m]['r2'] for m in models]
bars = ax3.bar(models, r2s, color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
ax3.set_ylabel('R² (5-fold CV)')
ax3.set_title('(C) Model Performance')
ax3.set_ylim(0, 1)
for bar, val in zip(bars, r2s):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=10)

# 4. High vs low composition
ax4 = fig.add_subplot(gs[1, 0])
high_perf = df[df['Glass (kPa)_10s'] > df['Glass (kPa)_10s'].quantile(0.9)]
low_perf = df[df['Glass (kPa)_10s'] < df['Glass (kPa)_10s'].quantile(0.1)]
x = np.arange(len(monomers))
width = 0.35
ax4.bar(x - width/2, high_perf[monomers].mean(), width, label='Top 10%', color='seagreen', edgecolor='white')
ax4.bar(x + width/2, low_perf[monomers].mean(), width, label='Bottom 10%', color='coral', edgecolor='white')
ax4.set_xticks(x)
ax4.set_xticklabels([m.split('-')[0] for m in monomers], rotation=30, ha='right', fontsize=9)
ax4.set_ylabel('Mean Fraction')
ax4.set_title('(D) Composition: High vs Low Adhesion')
ax4.legend(fontsize=8)

# 5. Optimization trajectory
ax5 = fig.add_subplot(gs[1, 1])
pred_summary = pd.read_csv("outputs/pred_summary.csv")
for method in ['RFR-GP', 'GP-GP', 'old-SM-GP']:
    sub = pred_summary[pred_summary['ML'].str.contains(method, na=False)]
    if len(sub) > 0:
        ax5.plot(sub['Round'], sub['max'], marker='o', label=method, linewidth=2, markersize=8)
ax5.set_xlabel('Round')
ax5.set_ylabel('Max Predicted (kPa)')
ax5.set_title('(E) Optimization Trajectory')
ax5.axhline(1000, color='red', linestyle='--', alpha=0.7)
ax5.legend(fontsize=8)
ax5.set_xticks([1, 2, 3])

# 6. PCA biplot simplified
ax6 = fig.add_subplot(gs[1, 2])
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[monomers].values)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Glass (kPa)_10s'], cmap='viridis', s=40, alpha=0.7, edgecolors='white')
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=monomers)
for i, mon in enumerate(monomers):
    ax6.arrow(0, 0, loadings.loc[mon, 'PC1']*2.5, loadings.loc[mon, 'PC2']*2.5,
              head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.7)
    ax6.text(loadings.loc[mon, 'PC1']*2.8, loadings.loc[mon, 'PC2']*2.8, mon.split('-')[0], color='red', fontsize=8)
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax6.set_title('(F) Composition Space PCA')
plt.colorbar(scatter, ax=ax6, shrink=0.6, label='Strength (kPa)')

# 7. Monomer vs target scatter (Hydrophobic-BA most important)
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(df['Hydrophobic-BA'], df['Glass (kPa)_10s'], alpha=0.6, c='steelblue', edgecolors='white', s=50)
z = np.polyfit(df['Hydrophobic-BA'], df['Glass (kPa)_10s'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Hydrophobic-BA'].min(), df['Hydrophobic-BA'].max(), 100)
ax7.plot(x_line, p(x_line), color='orange', linestyle='--', linewidth=2)
ax7.set_xlabel('Hydrophobic-BA fraction')
ax7.set_ylabel('Glass Adhesive Strength (kPa)')
ax7.set_title('(G) Hydrophobic-BA vs Strength')
ax7.axhline(1000, color='red', linestyle='--', alpha=0.5)

# 8. Aromatic-PEA vs target
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(df['Aromatic-PEA'], df['Glass (kPa)_10s'], alpha=0.6, c='steelblue', edgecolors='white', s=50)
z = np.polyfit(df['Aromatic-PEA'], df['Glass (kPa)_10s'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Aromatic-PEA'].min(), df['Aromatic-PEA'].max(), 100)
ax8.plot(x_line, p(x_line), color='orange', linestyle='--', linewidth=2)
ax8.set_xlabel('Aromatic-PEA fraction')
ax8.set_ylabel('Glass Adhesive Strength (kPa)')
ax8.set_title('(H) Aromatic-PEA vs Strength')
ax8.axhline(1000, color='red', linestyle='--', alpha=0.5)

# 9. Nucleophilic-HEA vs target
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(df['Nucleophilic-HEA'], df['Glass (kPa)_10s'], alpha=0.6, c='steelblue', edgecolors='white', s=50)
z = np.polyfit(df['Nucleophilic-HEA'], df['Glass (kPa)_10s'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Nucleophilic-HEA'].min(), df['Nucleophilic-HEA'].max(), 100)
ax9.plot(x_line, p(x_line), color='orange', linestyle='--', linewidth=2)
ax9.set_xlabel('Nucleophilic-HEA fraction')
ax9.set_ylabel('Glass Adhesive Strength (kPa)')
ax9.set_title('(I) Nucleophilic-HEA vs Strength')
ax9.axhline(1000, color='red', linestyle='--', alpha=0.5)

plt.savefig("report/images/fig18_summary_dashboard.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig18_summary_dashboard.png")
