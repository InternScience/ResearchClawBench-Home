#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Hierarchical Aggregation Verification and Correlation Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# 1. Load all data
# ============================================================
print("Loading data...")
buildings = {}
for i in range(1, 11):
    fname = f'BN{i:03d}_energy.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    buildings[f'BN{i:03d}'] = df

cn01 = pd.read_csv(os.path.join(DATA_DIR, 'CN01_energy.csv'))
cn01['datetime'] = pd.to_datetime(cn01[['year', 'month', 'day', 'hour']])

total = pd.read_csv(os.path.join(DATA_DIR, 'Total_energy.csv'))
total['datetime'] = pd.to_datetime(total[['year', 'month', 'day', 'hour']])

weather = pd.read_csv(os.path.join(DATA_DIR, 'Total_weather.csv'))
weather['datetime'] = pd.to_datetime(weather['datetime'])

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']

# ============================================================
# 2. Hierarchical Aggregation Verification
# ============================================================
print("\n" + "=" * 60)
print("HIERARCHICAL AGGREGATION VERIFICATION")
print("=" * 60)

# Compute sum of all 10 buildings
sum_buildings = buildings['BN001'].copy()
for col in energy_cols:
    sum_vals = np.zeros(len(sum_buildings))
    for b in buildings:
        sum_vals += buildings[b][col].values
    sum_buildings[col] = sum_vals

# Verify: Sum(BN001-BN010) == CN01 == Total
verification_results = []
for col in energy_cols:
    diff_cn01 = np.abs(sum_buildings[col] - cn01[col]).max()
    diff_total = np.abs(sum_buildings[col] - total[col]).max()
    diff_cn01_total = np.abs(cn01[col] - total[col]).max()
    verification_results.append({
        'Variable': col,
        'Sum vs CN01 max diff': f'{diff_cn01:.10f}',
        'Sum vs Total max diff': f'{diff_total:.10f}',
        'CN01 vs Total max diff': f'{diff_cn01_total:.10f}',
        'Consistent': 'YES' if diff_cn01 < 1e-6 and diff_total < 1e-6 else 'NO'
    })

verif_df = pd.DataFrame(verification_results)
print(verif_df.to_string(index=False))
verif_df.to_csv(os.path.join(OUTPUT_DIR, 'hierarchical_verification.csv'), index=False)

# ============================================================
# 3. Cross-Variable Correlation Analysis (Total Level)
# ============================================================
print("\n" + "=" * 60)
print("CROSS-VARIABLE CORRELATION ANALYSIS (Total Level)")
print("=" * 60)

# Merge energy and weather
merged = total[['datetime'] + energy_cols].merge(
    weather[['datetime'] + [c for c in weather.columns if c != 'datetime']],
    on='datetime', how='inner'
)

all_cols = energy_cols + ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                           'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]',
                           'Precipitation [in]']

corr_matrix = merged[all_cols].corr()
print("\nCorrelation Matrix (all variables):")
print(corr_matrix.round(3))

corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))

# ============================================================
# 4. Figure 3: Correlation Heatmap
# ============================================================
print("\nGenerating Figure 3: Correlation Heatmap")

# Simplify column names for display
short_names = {
    'Electricity [kW]': 'Electricity',
    'Heat [mmBTU]': 'Heat',
    'Cooling Energy [Ton]': 'Cooling',
    'PV Power Generation [kW]': 'PV Gen',
    'Greenhouse Gas Emission [Ton]': 'GHG',
    'Temperature [°F]': 'Temperature',
    'Dew Point [°F]': 'Dew Point',
    'Humidity [%]': 'Humidity',
    'Wind Speed [mph]': 'Wind Speed',
    'Wind Gust [mph]': 'Wind Gust',
    'Pressure [in]': 'Pressure',
    'Precipitation [in]': 'Precipitation'
}

corr_short = corr_matrix.rename(index=short_names, columns=short_names)

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_short, dtype=bool), k=1)
sns.heatmap(corr_short, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax)
ax.set_title('Figure 3: Correlation Matrix of Energy and Weather Variables (Total Level, 2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure3_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3_correlation_heatmap.png")

# ============================================================
# 5. Cross-Building Correlation
# ============================================================
print("\n" + "=" * 60)
print("CROSS-BUILDING CORRELATION ANALYSIS")
print("=" * 60)

building_names = [f'BN{i:03d}' for i in range(1, 11)]

for energy_var in energy_cols:
    # Build matrix: rows=time, cols=building
    bm = pd.DataFrame({b: buildings[b][energy_var].values for b in building_names})
    bcorr = bm.corr()
    print(f"\nCross-building correlation for {energy_var}:")
    print(f"  Mean pairwise correlation: {bcorr.values[np.triu_indices(10, k=1)].mean():.4f}")
    print(f"  Min pairwise correlation: {bcorr.values[np.triu_indices(10, k=1)].min():.4f}")
    print(f"  Max pairwise correlation: {bcorr.values[np.triu_indices(10, k=1)].max():.4f}")

# ============================================================
# 6. Figure 4: Cross-Building Correlation Heatmaps
# ============================================================
print("\nGenerating Figure 4: Cross-Building Correlation Heatmaps")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, energy_var in enumerate(energy_cols):
    ax = axes[idx]
    bm = pd.DataFrame({b: buildings[b][energy_var].values for b in building_names})
    bcorr = bm.corr()
    
    sns.heatmap(bcorr, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0.5, vmax=1.0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(short_names.get(energy_var, energy_var), fontsize=12, fontweight='bold')
    ax.set_xticklabels(building_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(building_names, rotation=0, fontsize=8)

axes[5].set_visible(False)
plt.suptitle('Figure 4: Cross-Building Pairwise Correlations by Energy Variable',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure4_cross_building_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure4_cross_building_correlation.png")

# ============================================================
# 7. Figure 5: Scatter Plots - Key Relationships
# ============================================================
print("\nGenerating Figure 5: Key Scatter Relationships")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Electricity vs Temperature
ax = axes[0, 0]
ax.scatter(merged['Temperature [°F]'], merged['Electricity [kW]'], alpha=0.3, s=5, c='steelblue')
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Electricity [kW]')
ax.set_title('Electricity vs Temperature', fontweight='bold')
r = merged['Temperature [°F]'].corr(merged['Electricity [kW]'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')

# Cooling vs Temperature
ax = axes[0, 1]
ax.scatter(merged['Temperature [°F]'], merged['Cooling Energy [Ton]'], alpha=0.3, s=5, c='coral')
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Cooling Energy [Ton]')
ax.set_title('Cooling Energy vs Temperature', fontweight='bold')
r = merged['Temperature [°F]'].corr(merged['Cooling Energy [Ton]'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')

# PV Generation vs Temperature
ax = axes[1, 0]
daytime = merged[merged['PV Power Generation [kW]'] > 0]
ax.scatter(daytime['Temperature [°F]'], daytime['PV Power Generation [kW]'], alpha=0.3, s=5, c='green')
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('PV Power Generation [kW]')
ax.set_title('PV Generation vs Temperature (Daytime)', fontweight='bold')
r = daytime['Temperature [°F]'].corr(daytime['PV Power Generation [kW]'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')

# GHG vs Electricity
ax = axes[1, 1]
ax.scatter(merged['Electricity [kW]'], merged['Greenhouse Gas Emission [Ton]'], alpha=0.3, s=5, c='purple')
ax.set_xlabel('Electricity [kW]')
ax.set_ylabel('GHG Emission [Ton]')
ax.set_title('GHG Emission vs Electricity', fontweight='bold')
r = merged['Electricity [kW]'].corr(merged['Greenhouse Gas Emission [Ton]'])
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, va='top')

plt.suptitle('Figure 5: Key Scatter Relationships Between Energy and Weather Variables',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure5_scatter_relationships.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5_scatter_relationships.png")

print("\nCorrelation analysis complete!")
