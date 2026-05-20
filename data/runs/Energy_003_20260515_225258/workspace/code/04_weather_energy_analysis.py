#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Weather-Energy Coupling Analysis and Anomaly Detection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
total = pd.read_csv(os.path.join(DATA_DIR, 'Total_energy.csv'))
total['datetime'] = pd.to_datetime(total[['year', 'month', 'day', 'hour']])

weather = pd.read_csv(os.path.join(DATA_DIR, 'Total_weather.csv'))
weather['datetime'] = pd.to_datetime(weather['datetime'])

# Merge
merged = total.merge(weather, on='datetime', how='inner')
merged['month'] = merged['datetime'].dt.month
merged['hour'] = merged['datetime'].dt.hour

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]',
                'Precipitation [in]']

# ============================================================
# 2. Figure 10: Temperature Binned Analysis
# ============================================================
print("Generating Figure 10: Temperature-Binned Energy Analysis")

# Bin temperature
temp_bins = pd.cut(merged['Temperature [°F]'], bins=15)
temp_binned = merged.groupby(temp_bins, observed=True)[['Electricity [kW]', 'Cooling Energy [Ton]',
                                                         'Heat [mmBTU]', 'PV Power Generation [kW]']].agg(['mean', 'std'])
temp_centers = [iv.mid for iv in temp_binned.index.categories]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Electricity vs Temperature bin
ax = axes[0, 0]
means = temp_binned['Electricity [kW]']['mean'].values
stds = temp_binned['Electricity [kW]']['std'].values
ax.errorbar(temp_centers, means, yerr=stds, fmt='o-', color='steelblue', capsize=4, linewidth=2)
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Electricity [kW]')
ax.set_title('Electricity vs Temperature (Binned)', fontweight='bold')
ax.grid(alpha=0.3)

# Cooling vs Temperature bin
ax = axes[0, 1]
means = temp_binned['Cooling Energy [Ton]']['mean'].values
stds = temp_binned['Cooling Energy [Ton]']['std'].values
ax.errorbar(temp_centers, means, yerr=stds, fmt='o-', color='coral', capsize=4, linewidth=2)
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Cooling Energy [Ton]')
ax.set_title('Cooling vs Temperature (Binned)', fontweight='bold')
ax.grid(alpha=0.3)

# Heat vs Temperature bin
ax = axes[1, 0]
means = temp_binned['Heat [mmBTU]']['mean'].values
stds = temp_binned['Heat [mmBTU]']['std'].values
ax.errorbar(temp_centers, means, yerr=stds, fmt='o-', color='red', capsize=4, linewidth=2)
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Heat [mmBTU]')
ax.set_title('Heat vs Temperature (Binned)', fontweight='bold')
ax.grid(alpha=0.3)

# PV Gen vs Temperature bin
ax = axes[1, 1]
means = temp_binned['PV Power Generation [kW]']['mean'].values
stds = temp_binned['PV Power Generation [kW]']['std'].values
ax.errorbar(temp_centers, means, yerr=stds, fmt='o-', color='green', capsize=4, linewidth=2)
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('PV Generation [kW]')
ax.set_title('PV Generation vs Temperature (Binned)', fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Figure 10: Temperature-Binned Energy Consumption Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure10_temperature_binned.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure10_temperature_binned.png")

# ============================================================
# 3. Figure 11: Anomaly Detection using IQR
# ============================================================
print("Generating Figure 11: Anomaly Detection")

# Detect anomalies in electricity using IQR method
elec = total['Electricity [kW]'].values
Q1 = np.percentile(elec, 25)
Q3 = np.percentile(elec, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
anomalies = (elec < lower) | (elec > upper)
n_anomalies = anomalies.sum()

print(f"Electricity IQR bounds: [{lower:.2f}, {upper:.2f}]")
print(f"Electricity anomalies: {n_anomalies} / {len(elec)} ({100*n_anomalies/len(elec):.2f}%)")

# Anomaly detection for all energy variables
anomaly_report = []
for col in energy_cols:
    vals = total[col].values
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_anom = ((vals < lo) | (vals > hi)).sum()
    anomaly_report.append({
        'Variable': col,
        'Q1': q1, 'Q3': q3, 'IQR': iqr,
        'Lower Bound': lo, 'Upper Bound': hi,
        'Anomaly Count': n_anom,
        'Anomaly %': f'{100*n_anom/len(vals):.2f}%'
    })

anomaly_df = pd.DataFrame(anomaly_report)
print(anomaly_df.to_string(index=False))
anomaly_df.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_detection.csv'), index=False)

# Plot anomaly detection
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    vals = total[col].values
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    
    ax.boxplot(vals, vert=True, patch_artist=True,
               boxprops=dict(facecolor=colors[idx] if 'colors' in dir() else 'lightblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel(col)
    ax.set_title(f'{col}\nAnomalies: {((vals < lo) | (vals > hi)).sum()}/8760', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)

axes[5].set_visible(False)
plt.suptitle('Figure 11: Anomaly Detection using IQR Method (Total Level, 2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure11_anomaly_detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure11_anomaly_detection.png")

# ============================================================
# 4. Figure 12: Hierarchical Consistency Heatmap
# ============================================================
print("Generating Figure 12: Hierarchical Consistency Verification")

# Load all buildings
buildings = {}
for i in range(1, 11):
    fname = f'BN{i:03d}_energy.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    buildings[f'BN{i:03d}'] = df

# Compute contribution of each building to total
building_names = [f'BN{i:03d}' for i in range(1, 11)]
contributions = {}
for col in energy_cols:
    total_sum = sum(buildings[b][col].sum() for b in building_names)
    contributions[col] = {b: 100 * buildings[b][col].sum() / total_sum for b in building_names}

contrib_df = pd.DataFrame(contributions, index=building_names)

# Plot building contribution to total
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(contrib_df.T, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Contribution (%)'},
            ax=ax)
ax.set_title('Figure 12: Building-Level Contribution to Total Energy (%)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Building')
ax.set_ylabel('Energy Variable')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure12_building_contribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure12_building_contribution.png")

# Save contributions
contrib_df.to_csv(os.path.join(OUTPUT_DIR, 'building_contributions.csv'))

# ============================================================
# 5. Save key results for report
# ============================================================
print("\nSaving key results...")

# Weather-energy correlations (key pairs)
key_pairs = [
    ('Temperature [°F]', 'Electricity [kW]'),
    ('Temperature [°F]', 'Cooling Energy [Ton]'),
    ('Temperature [°F]', 'Heat [mmBTU]'),
    ('Humidity [%]', 'Cooling Energy [Ton]'),
    ('Temperature [°F]', 'PV Power Generation [kW]'),
]

corr_results = []
for wcol, ecol in key_pairs:
    r, p = stats.pearsonr(merged[wcol], merged[ecol])
    corr_results.append({'Weather': wcol, 'Energy': ecol, 'Pearson r': r, 'p-value': p})

corr_results_df = pd.DataFrame(corr_results)
corr_results_df.to_csv(os.path.join(OUTPUT_DIR, 'key_correlations.csv'), index=False)
print("\nKey Weather-Energy Correlations:")
print(corr_results_df.to_string(index=False))

# PV statistics
pv_total = total['PV Power Generation [kW]'].sum()
pv_peak = total['PV Power Generation [kW]'].max()
pv_mean_daytime = total[total['PV Power Generation [kW]'] > 0]['PV Power Generation [kW]'].mean()
print(f"\nPV Generation - Annual total: {pv_total:.2f} kWh")
print(f"PV Generation - Peak: {pv_peak:.2f} kW")
print(f"PV Generation - Mean (daytime): {pv_mean_daytime:.2f} kW")

print("\nWeather-Energy analysis complete!")
