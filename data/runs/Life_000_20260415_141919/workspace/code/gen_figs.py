#!/usr/bin/env python3
"""Generate figures for hydrogel analysis"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')
for col in df.columns:
    if col not in ['No.', 'Tanδ', 'Log_Slope']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

MONOMERS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
            'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET = 'Glass (kPa)_10s'

print("Figure 1: Monomer Distribution")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, monomer in enumerate(MONOMERS):
    ax = axes[i]
    data = df[monomer].dropna()
    ax.hist(data, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(data.mean(), color='red', linestyle='--')
    ax.set_xlabel(f'{monomer} (mole fraction)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{monomer}', fontsize=11)
plt.suptitle('Distribution of Monomer Compositions (n=184)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig1_monomer_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2: Adhesive Distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
data = df[TARGET].dropna()
ax1.hist(data, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f} kPa')
ax1.axvline(1000, color='orange', linestyle='-', linewidth=2, label='Target: 1000 kPa')
ax1.set_xlabel('Glass Adhesion Strength (kPa)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Glass Adhesion Distribution (n=184)', fontsize=13)
ax1.legend()
ax2 = axes[1]
steel_data = df['Steel (kPa)_10s'].dropna()
ax2.hist(steel_data, bins=15, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(steel_data.mean(), color='red', linestyle='--', label=f'Mean: {steel_data.mean():.1f} kPa')
ax2.set_xlabel('Steel Adhesion Strength (kPa)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Steel Adhesion Distribution (n=28)', fontsize=13)
ax2.legend()
plt.tight_layout()
plt.savefig('report/images/fig2_adhesive_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3: Correlation Heatmap")
corr_cols = MONOMERS + [TARGET, 'Steel (kPa)_10s', 'Q', 'Modulus (kPa)', 'XlogP3']
corr_data = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix: Monomer Composition vs Properties', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4: Random Forest")
X = df[MONOMERS].copy()
y = df[TARGET].copy()
valid_idx = X.notna().all(axis=1) & y.notna()
X = X[valid_idx]
y = y[valid_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axes[0]
importance = pd.DataFrame({'Feature': MONOMERS, 'Importance': rf.feature_importances_}).sort_values('Importance')
ax1.barh(importance['Feature'], importance['Importance'], color='steelblue', edgecolor='black')
ax1.set_xlabel('Feature Importance', fontsize=12)
ax1.set_title('Random Forest Feature Importance', fontsize=13)
ax2 = axes[1]
ax2.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=80, color='steelblue')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Glass Adhesion (kPa)', fontsize=12)
ax2.set_ylabel('Predicted Glass Adhesion (kPa)', fontsize=12)
ax2.set_title(f'Parity Plot: R2 = {r2:.3f}', fontsize=13)
ax2.legend()
plt.tight_layout()
plt.savefig('report/images/fig4_rf_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5: Composition Space")
df_plot = df.copy()
df_plot['Hydrophilic'] = df['Nucleophilic-HEA'] + df['Amide-AAm']
df_plot['Hydrophobic'] = df['Hydrophobic-BA'] + df['Aromatic-PEA']
df_plot['Charged'] = df['Acidic-CBEA'] + df['Cationic-ATAC']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax1 = axes[0]
scatter = ax1.scatter(df_plot['Hydrophilic'], df_plot['Hydrophobic'], 
                     c=df_plot[TARGET], cmap='viridis', s=50, alpha=0.6, edgecolor='black')
plt.colorbar(scatter, ax=ax1, label='Glass Adhesion (kPa)')
ax1.set_xlabel('Hydrophilic Fraction (HEA + AAm)', fontsize=12)
ax1.set_ylabel('Hydrophobic Fraction (BA + PEA)', fontsize=12)
ax1.set_title('Composition Space: All Formulations', fontsize=13)
ax2 = axes[1]
top_plot = df_plot.nlargest(20, TARGET)
scatter2 = ax2.scatter(top_plot['Hydrophilic'], top_plot['Hydrophobic'], 
                      c=top_plot[TARGET], cmap='plasma', s=100, alpha=0.8, edgecolor='black')
plt.colorbar(scatter2, ax=ax2, label='Glass Adhesion (kPa)')
ax2.set_xlabel('Hydrophilic Fraction (HEA + AAm)', fontsize=12)
ax2.set_ylabel('Hydrophobic Fraction (BA + PEA)', fontsize=12)
ax2.set_title('Composition Space: Top 20 Performers', fontsize=13)
plt.tight_layout()
plt.savefig('report/images/fig5_composition_space.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 6: Optimization Results")
df_opt = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
df_opt['ML'] = df_opt['ML'].fillna(method='ffill')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axes[0]
ml_perf = df_opt.groupby('ML')['Glass (kPa)_max'].mean().sort_values()
y_pos = np.arange(len(ml_perf))
ax1.barh(y_pos, ml_perf.values, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(ml_perf.index)
ax1.set_xlabel('Mean Predicted Glass Adhesion (kPa)', fontsize=12)
ax1.set_title('ML Model Performance Comparison', fontsize=13)
ax1.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa Target')
ax1.legend()

ax2 = axes[1]
for ml_type in df_opt['ML'].unique():
    data = df_opt[df_opt['ML'] == ml_type]['Glass (kPa)_max']
    ax2.hist(data, bins=15, alpha=0.5, label=ml_type, edgecolor='black')
ax2.set_xlabel('Predicted Glass Adhesion (kPa)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Predicted Adhesion by ML Model', fontsize=13)
ax2.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa Target')
ax2.legend()
plt.tight_layout()
plt.savefig('report/images/fig6_optimization_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print(f"Random Forest R2: {r2:.4f}")
