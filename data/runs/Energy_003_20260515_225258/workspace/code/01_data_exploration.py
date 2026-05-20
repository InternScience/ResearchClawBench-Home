#!/usr/bin/env python3
"""
HEEW Mini-Dataset Exploration and Statistical Summary
Analyzes the hierarchical structure, data quality, and basic statistics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

# Paths
DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

# ============================================================
# 1. Load all data
# ============================================================
print("=" * 60)
print("1. Loading all data files")
print("=" * 60)

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
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]',
                'Precipitation [in]']

print(f"Loaded {len(buildings)} building files, 1 community (CN01), 1 total, 1 weather")
print(f"Time range: {total['datetime'].min()} to {total['datetime'].max()}")
print(f"Records per file: {len(total)} (hourly, full year 2014)")

# ============================================================
# 2. Data Quality Assessment
# ============================================================
print("\n" + "=" * 60)
print("2. Data Quality Assessment")
print("=" * 60)

quality_report = []

for name, df in list(buildings.items()) + [('CN01', cn01), ('Total', total)]:
    missing = df[energy_cols].isnull().sum().sum()
    neg_count = {col: (df[col] < 0).sum() for col in energy_cols}
    quality_report.append({
        'Entity': name,
        'Records': len(df),
        'Missing Values': missing,
        'Negative Electricity': neg_count['Electricity [kW]'],
        'Negative Heat': neg_count['Heat [mmBTU]'],
        'Negative Cooling': neg_count['Cooling Energy [Ton]'],
        'Negative PV': neg_count['PV Power Generation [kW]'],
        'Negative GHG': neg_count['Greenhouse Gas Emission [Ton]'],
    })

quality_df = pd.DataFrame(quality_report)
print(quality_df.to_string(index=False))
quality_df.to_csv(os.path.join(OUTPUT_DIR, 'data_quality.csv'), index=False)

# Weather quality
print(f"\nWeather missing values: {weather[weather_cols].isnull().sum().sum()}")
print(f"Weather negative temp count: {(weather['Temperature [°F]'] < 0).sum()}")
print(f"Weather negative humidity count: {(weather['Humidity [%]'] < 0).sum()}")

# ============================================================
# 3. Statistical Summary
# ============================================================
print("\n" + "=" * 60)
print("3. Statistical Summary")
print("=" * 60)

# Per-building summary
summary_rows = []
for name, df in list(buildings.items()):
    row = {'Entity': name}
    for col in energy_cols:
        row[f'{col}_mean'] = df[col].mean()
        row[f'{col}_std'] = df[col].std()
        row[f'{col}_min'] = df[col].min()
        row[f'{col}_max'] = df[col].max()
    summary_rows.append(row)

# Add CN01 and Total
for name, df in [('CN01', cn01), ('Total', total)]:
    row = {'Entity': name}
    for col in energy_cols:
        row[f'{col}_mean'] = df[col].mean()
        row[f'{col}_std'] = df[col].std()
        row[f'{col}_min'] = df[col].min()
        row[f'{col}_max'] = df[col].max()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_summary.csv'), index=False)

# Print key stats for Total
print("\nTotal Energy Statistics:")
for col in energy_cols:
    vals = total[col]
    print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
          f"min={vals.min():.2f}, max={vals.max():.2f}")

print("\nWeather Statistics:")
for col in weather_cols:
    vals = weather[col]
    print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
          f"min={vals.min():.2f}, max={vals.max():.2f}")

# ============================================================
# 4. Figure 1: Data Overview - Distribution of Energy Variables
# ============================================================
print("\n" + "=" * 60)
print("4. Generating Figure 1: Energy Distribution Overview")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Plot distributions for Total
for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    ax.hist(total[col], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(total[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {total[col].mean():.1f}')
    ax.axvline(total[col].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {total[col].median():.1f}')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

# Remove extra subplot (6th)
axes[5].set_visible(False)

plt.suptitle('Figure 1: Distribution of Energy Variables at Total Level (2014)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure1_energy_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure1_energy_distribution.png")

# ============================================================
# 5. Figure 2: Per-Building Comparison
# ============================================================
print("5. Generating Figure 2: Per-Building Energy Profiles")

# Prepare building summary
building_names = [f'BN{i:03d}' for i in range(1, 11)]
b_means = {col: [buildings[b][col].mean() for b in building_names] for col in energy_cols}
b_stds = {col: [buildings[b][col].std() for b in building_names] for col in energy_cols}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    x = np.arange(len(building_names))
    means = b_means[col]
    stds = b_stds[col]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=sns.color_palette('Set2', 10), edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(building_names, rotation=45, ha='right', fontsize=9)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Value')
    ax.grid(axis='y', alpha=0.3)

axes[5].set_visible(False)
plt.suptitle('Figure 2: Per-Building Energy Consumption (Mean ± Std, 2014)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure2_building_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure2_building_comparison.png")

print("\nData exploration complete!")
