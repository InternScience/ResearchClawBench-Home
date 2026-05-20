#!/usr/bin/env python3
"""
SHAP interpretability and composition space analysis.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap

np.random.seed(42)
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("outputs/df184_clean.csv")
monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
X = df[monomers].values
y = df['Glass (kPa)_10s'].values

# Train final RFR
rfr = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X, y)

# SHAP analysis
print("Computing SHAP values...")
explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(X)

# Figure 10: SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=monomers, show=False)
plt.tight_layout()
plt.savefig("report/images/fig10_shap_summary.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig10_shap_summary.png")

# Figure 11: SHAP bar plot
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, X, feature_names=monomers, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("report/images/fig11_shap_bar.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig11_shap_bar.png")

# Save SHAP values
shap_df = pd.DataFrame(shap_values, columns=monomers)
shap_df['prediction'] = rfr.predict(X)
shap_df['actual'] = y
shap_df.to_csv("outputs/shap_values.csv", index=False)
print("Saved outputs/shap_values.csv")

# Composition space analysis using PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Load optimization data for PCA overlay
df_pred = pd.read_excel("data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')
df_pred['ML'] = df_pred['ML'].ffill()
df_pred['Glass (kPa)_max'] = pd.to_numeric(df_pred['Glass (kPa)_max'], errors='coerce')
# Clean monomer columns
for m in monomers:
    df_pred[m] = pd.to_numeric(df_pred[m], errors='coerce')
df_pred = df_pred.dropna(subset=['Glass (kPa)_max'] + monomers)
X_opt = df_pred[monomers].astype(float).values
X_opt_scaled = scaler.transform(X_opt)
X_opt_pca = pca.transform(X_opt_scaled)

# Figure 12: PCA composition space
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=60, alpha=0.7, edgecolors='white', label='Initial 184')
opt_scatter = ax.scatter(X_opt_pca[:, 0], X_opt_pca[:, 1], c=df_pred['Glass (kPa)_max'], cmap='plasma', s=80, alpha=0.8, edgecolors='black', marker='^', label='Optimized')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Composition Space: Initial vs Optimized Formulations')
ax.legend()
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Initial Strength (kPa)')
cbar2 = plt.colorbar(opt_scatter, ax=ax, shrink=0.6, pad=0.08)
cbar2.set_label('Optimized Pred. Strength (kPa)')
plt.tight_layout()
plt.savefig("report/images/fig12_composition_space.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig12_composition_space.png")

# PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=monomers)
print("\nPCA Loadings:")
print(loadings)
loadings.to_csv("outputs/pca_loadings.csv")

# Figure 13: PCA loadings biplot
fig, ax = plt.subplots(figsize=(8, 8))
for i, mon in enumerate(monomers):
    ax.arrow(0, 0, loadings.loc[mon, 'PC1']*3, loadings.loc[mon, 'PC2']*3, 
             head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(loadings.loc[mon, 'PC1']*3.2, loadings.loc[mon, 'PC2']*3.2, mon, color='red', fontsize=10)

ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=40, alpha=0.5, edgecolors='white')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA Biplot: Monomer Loadings')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("report/images/fig13_pca_biplot.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig13_pca_biplot.png")

# Identify high-performing regions
print("\n=== Top 10 actual formulations ===")
top_actual = df.nlargest(10, 'Glass (kPa)_10s')[['No.', 'Glass (kPa)_10s'] + monomers]
print(top_actual)
top_actual.to_csv("outputs/top_actual_formulations.csv", index=False)

# Composition trends for high performers
high_perf = df[df['Glass (kPa)_10s'] > df['Glass (kPa)_10s'].quantile(0.9)]
low_perf = df[df['Glass (kPa)_10s'] < df['Glass (kPa)_10s'].quantile(0.1)]

print("\n=== Top 10% vs Bottom 10% composition ===")
print("High performers (top 10%):")
print(high_perf[monomers].mean())
print("\nLow performers (bottom 10%):")
print(low_perf[monomers].mean())

# Figure 14: Composition comparison (radar chart style via bar)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(monomers))
width = 0.35
bars1 = ax.bar(x - width/2, high_perf[monomers].mean(), width, label='Top 10% (High Adhesion)', color='seagreen', edgecolor='white')
bars2 = ax.bar(x + width/2, low_perf[monomers].mean(), width, label='Bottom 10% (Low Adhesion)', color='coral', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(monomers, rotation=30, ha='right')
ax.set_ylabel('Mean Monomer Fraction')
ax.set_title('Composition Comparison: High vs Low Adhesion Formulations')
ax.legend()
ax.set_ylim(0, 0.6)
plt.tight_layout()
plt.savefig("report/images/fig14_high_vs_low_comp.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig14_high_vs_low_comp.png")
