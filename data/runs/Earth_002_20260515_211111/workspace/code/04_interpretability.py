#!/usr/bin/env python3
"""
SHAP-like interpretability analysis for the mangrove composite risk index.
Uses permutation importance and dependency analysis to understand risk drivers.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

print("=== Loading data ===")
gdf = gpd.read_parquet('outputs/mangrove_risk_final.parquet')

# ============================================================
# Feature importance for CRI prediction
# ============================================================

# Features
feature_cols = [
    'ssp5_8_5_slr_2080_2100', 
    'tc_total_freq', 'tc_major_freq', 'tc_intense_freq',
    'tc_shift_factor', 'tc_baseline_risk_score',
]

# Prepare data
X = gdf[feature_cols].copy()
y = gdf['ssp5_8_5_cri'].values

# Remove any NaN
valid = X.notna().all(axis=1) & ~np.isnan(y)
X = X[valid]
y = y[valid]

print(f"Training on {len(X)} samples")

# Train RF
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Permutation importance
perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)

# ============================================================
# Figure 7: Interpretability analysis
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel a: Permutation importance
ax = axes[0, 0]
importances = perm_result.importances_mean
importances_std = perm_result.importances_std
sorted_idx = np.argsort(importances)

feature_labels = [
    'RSLR Rate (SSP5-8.5)', 
    'Total TC Frequency', 
    'Major TC Frequency (Cat 3-5)', 
    'Intense TC Frequency (Cat 4-5)',
    'TC Shift Factor', 
    'Baseline TC Risk Score',
]

ax.barh(range(len(sorted_idx)), importances[sorted_idx], 
        xerr=importances_std[sorted_idx], 
        color=['#e74c3c' if 'SLR' in str(feature_labels[i]) else '#3498db' for i in sorted_idx],
        edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_labels[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel('Permutation Importance (MSE Increase)')
ax.set_title('(a) Relative Importance of Risk Drivers', fontweight='bold')

# Panel b: Partial dependence of CRI on SLR
ax = axes[0, 1]
slr_vals = np.linspace(X['ssp5_8_5_slr_2080_2100'].min(), X['ssp5_8_5_slr_2080_2100'].max(), 50)
# Simple scatter of actual data
sample = gdf.sample(min(3000, len(gdf)), random_state=42)
ax.scatter(sample['ssp5_8_5_slr_2080_2100'], sample['ssp5_8_5_cri'], 
          c=sample['tc_major_freq'], s=2, alpha=0.3, cmap='Purples')
ax.set_xlabel('RSLR Rate 2080-2100 (mm/yr)')
ax.set_ylabel('Composite Risk Index')
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Major TC Freq')
ax.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='7 mm/yr threshold')
ax.axvline(x=4, color='orange', linestyle='--', alpha=0.5, label='4 mm/yr threshold')
ax.legend(fontsize=7)
ax.set_title('(b) CRI vs. RSLR Rate', fontweight='bold')

# Panel c: Partial dependence of CRI on TC frequency
ax = axes[1, 0]
ax.scatter(np.log10(sample['tc_major_freq'] + 0.001), sample['ssp5_8_5_cri'],
          c=sample['ssp5_8_5_slr_2080_2100'], s=2, alpha=0.3, cmap='YlOrRd')
ax.set_xlabel('log10(Major TC Frequency + 0.001)')
ax.set_ylabel('Composite Risk Index')
cbar = plt.colorbar(ax.collections[0], ax=ax, label='RSLR Rate (mm/yr)')
ax.set_title('(c) CRI vs. Major TC Frequency', fontweight='bold')

# Panel d: Feature correlation matrix
ax = axes[1, 1]
corr_features = ['ssp5_8_5_slr_2080_2100', 'tc_total_freq', 'tc_major_freq', 
                 'tc_intense_freq', 'tc_shift_factor', 'ssp5_8_5_cri']
corr_labels = ['SLR Rate', 'TC Total Freq', 'TC Major Freq', 
               'TC Intense Freq', 'TC Shift Factor', 'CRI']
corr_matrix = gdf[corr_features].corr()
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(corr_labels)))
ax.set_yticklabels(corr_labels, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
# Add values
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=6)
ax.set_title('(d) Feature Correlation Matrix', fontweight='bold')

plt.suptitle('Interpretability Analysis: Drivers of Mangrove Risk', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure7_interpretability.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 7 saved.")

# ============================================================
# Additional: Subgroup analysis - risk by mangrove quality class
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel a: Risk by ref_cls
ax = axes[0]
classes = sorted(gdf['ref_cls'].unique())
class_data = []
for cls in classes:
    sub = gdf[gdf['ref_cls'] == cls]
    class_data.append({
        'Class': f'Class {int(cls)}',
        'N': len(sub),
        'Mean CRI': sub['ssp5_8_5_cri'].mean(),
        'Mean SLR': sub['ssp5_8_5_slr_2080_2100'].mean(),
    })
class_df = pd.DataFrame(class_data)
colors_cls = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_df)))
ax.bar(class_df['Class'], class_df['Mean CRI'], color=colors_cls, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Mean Composite Risk Index')
ax.set_title('(a) Risk by Mangrove Reference Class', fontweight='bold')
ax.tick_params(axis='x', rotation=30)

# Panel b: Ecosystem services overlay for top risk countries
ax = axes[1]
eco_gdf = gpd.read_file('data/ecosystem/UCSC_CWON_countrybounds.gpkg')
country_risk = pd.read_csv('outputs/country_risk_SSP5_8_5.csv')
top15 = country_risk[country_risk['n_points'] >= 100].nlargest(15, 'mean_cri')

# Merge with ecosystem services
es_data = []
for _, row in top15.iterrows():
    country_name = row['Country']
    eco_row = eco_gdf[eco_gdf['Country'] == country_name]
    if len(eco_row) > 0:
        es_data.append({
            'Country': country_name,
            'Mean CRI': row['mean_cri'],
            'Mangrove Area (ha)': eco_row['Mang_Ha_2020'].values[0],
            'Population at Risk': eco_row['Risk_Pop_2020'].values[0],
        })

es_df = pd.DataFrame(es_data).sort_values('Mean CRI', ascending=False)

# Plot - two y-axes
x = range(len(es_df))
bars = ax.bar(x, es_df['Mean CRI'], color='#e74c3c', alpha=0.7, label='Mean CRI')
ax2 = ax.twinx()
ax2.scatter(x, np.log10(es_df['Population at Risk'] + 1), color='#3498db', s=50, 
           label='log10(Population at Risk)', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(es_df['Country'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean CRI', color='#e74c3c')
ax2.set_ylabel('log10(Population at Risk + 1)', color='#3498db')
ax.set_title('(b) Risk vs. Population Exposure', fontweight='bold')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.suptitle('Subgroup Analysis: Mangrove Quality Classes and Human Dimensions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure8_subgroups.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 8 saved.")

print("\n=== Interpretability analysis complete! ===")
