#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Correlation Analysis & All Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_003_20260416_185858'
DATA = os.path.join(BASE, 'data', 'HEEW_Mini-Dataset')
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
os.makedirs(IMG, exist_ok=True)

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})

# Load data
def load_energy(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].assign(minute=0, second=0))
    return df

buildings = {f'BN{i:03d}': load_energy(f'BN{i:03d}_energy.csv') for i in range(1, 11)}
cn01 = load_energy('CN01_energy.csv')
total_energy = load_energy('Total_energy.csv')
weather = pd.read_csv(os.path.join(DATA, 'Total_weather.csv'))
weather['datetime'] = pd.to_datetime(weather['datetime'])

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
energy_short = ['Electricity', 'Heat', 'Cooling', 'PV', 'GHG']
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
weather_short = ['Temp', 'DewPt', 'Humid', 'Wind', 'Gust', 'Press', 'Precip']

# Merge total energy + weather
merged = pd.merge(total_energy, weather, on='datetime', how='inner')

# =====================================================
# FIGURE 1: Time Series Overview of All Energy Variables
# =====================================================
print("Creating Figure 1: Time Series Overview...")
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    axes[i].plot(total_energy['datetime'], total_energy[col], color=colors[i], linewidth=0.3, alpha=0.8)
    axes[i].set_ylabel(short, fontsize=9)
    axes[i].set_title(f'{col} (Total)', fontsize=10)
    axes[i].grid(True, alpha=0.3)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].set_xlabel('Month (2014)')
fig.suptitle('HEEW Dataset: Hourly Energy Variables (Total Aggregate, 2014)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig1_timeseries_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig1_timeseries_overview.png")

# =====================================================
# FIGURE 2: Weather Time Series
# =====================================================
print("Creating Figure 2: Weather Time Series...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
w_groups = [
    (['Temperature [°F]', 'Dew Point [°F]'], ['Temp', 'Dew Point'], '°F'),
    (['Humidity [%]'], ['Humidity'], '%'),
    (['Wind Speed [mph]', 'Wind Gust [mph]'], ['Wind Speed', 'Wind Gust'], 'mph'),
    (['Pressure [in]'], ['Pressure'], 'in'),
]
for i, (cols, labels, unit) in enumerate(w_groups):
    for c, l in zip(cols, labels):
        axes[i].plot(weather['datetime'], weather[c], linewidth=0.3, alpha=0.8, label=l)
    axes[i].set_ylabel(unit)
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].set_xlabel('Month (2014)')
fig.suptitle('HEEW Dataset: Hourly Weather Variables (2014)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig2_weather_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig2_weather_timeseries.png")

# =====================================================
# FIGURE 3: Energy-Weather Correlation Heatmap
# =====================================================
print("Creating Figure 3: Correlation Heatmap...")
corr_data = merged[energy_cols + weather_cols]
corr_matrix = corr_data.corr()

# Save correlation matrix
corr_matrix.to_csv(os.path.join(OUT, 'correlation_matrix.csv'))

fig, ax = plt.subplots(figsize=(12, 10))
short_labels = energy_short + weather_short
mask = np.zeros_like(corr_matrix)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=short_labels, yticklabels=short_labels,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 7})
ax.set_title('Correlation Matrix: Energy and Weather Variables', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig3_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig3_correlation_heatmap.png")

# =====================================================
# FIGURE 4: Energy-Only Correlation (cross-variable)
# =====================================================
print("Creating Figure 4: Energy-only Correlation...")
energy_corr = merged[energy_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(energy_corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=energy_short, yticklabels=energy_short,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 10})
ax.set_title('Inter-Variable Correlation: Energy Variables', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig4_energy_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig4_energy_correlation.png")

# Save key correlations
key_corrs = {}
for i, ec in enumerate(energy_cols):
    for j, wc in enumerate(weather_cols):
        key_corrs[f'{energy_short[i]}_vs_{weather_short[j]}'] = round(float(corr_matrix.loc[ec, wc]), 4)
with open(os.path.join(OUT, 'key_correlations.json'), 'w') as f:
    json.dump(key_corrs, f, indent=2)

# =====================================================
# FIGURE 5: Hierarchical Aggregation Consistency
# =====================================================
print("Creating Figure 5: Hierarchical Consistency...")
building_sum = pd.DataFrame()
building_sum['datetime'] = buildings['BN001']['datetime']
for col in energy_cols:
    building_sum[col] = sum(buildings[k][col] for k in buildings)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    ax.scatter(cn01[col], building_sum[col], s=1, alpha=0.3, color='blue')
    lims = [min(cn01[col].min(), building_sum[col].min()),
            max(cn01[col].max(), building_sum[col].max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
    ax.set_xlabel(f'CN01 {short}')
    ax.set_ylabel(f'Sum(BN001..BN010) {short}')
    ax.set_title(f'{short}: Building Sum vs Community')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    corr = np.corrcoef(cn01[col], building_sum[col])[0, 1]
    ax.annotate(f'R² = {corr**2:.6f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9)

axes[5].axis('off')
fig.suptitle('Hierarchical Aggregation Consistency: Building Sum vs Community Aggregate', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig5_hierarchical_consistency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig5_hierarchical_consistency.png")

# =====================================================
# FIGURE 6: Monthly Patterns
# =====================================================
print("Creating Figure 6: Monthly Patterns...")
total_energy['month'] = total_energy['datetime'].dt.month
monthly = total_energy.groupby('month')[energy_cols].mean()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    ax.bar(range(1,13), monthly[col], color=colors[i], alpha=0.8)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_title(f'Monthly Average: {short}')
    ax.set_ylabel(col.split('[')[1].rstrip(']'))
    ax.grid(True, alpha=0.3, axis='y')
axes[5].axis('off')
fig.suptitle('Monthly Average Energy Profiles (Total Aggregate, 2014)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig6_monthly_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig6_monthly_patterns.png")

# Save monthly stats
monthly.to_csv(os.path.join(OUT, 'monthly_averages.csv'))

# =====================================================
# FIGURE 7: Hourly Diurnal Patterns
# =====================================================
print("Creating Figure 7: Diurnal Patterns...")
total_energy['hour_of_day'] = total_energy['datetime'].dt.hour
hourly = total_energy.groupby('hour_of_day')[energy_cols].mean()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    ax.plot(range(24), hourly[col], color=colors[i], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Hour of Day')
    ax.set_title(f'Diurnal Pattern: {short}')
    ax.set_ylabel(col.split('[')[1].rstrip(']'))
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, alpha=0.3)
axes[5].axis('off')
fig.suptitle('Average Diurnal Profiles (Total Aggregate, 2014)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig7_diurnal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig7_diurnal_patterns.png")

# =====================================================
# FIGURE 8: Building Comparison (Box plots)
# =====================================================
print("Creating Figure 8: Building Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    data_list = [buildings[k][col].values for k in sorted(buildings.keys())]
    bp = ax.boxplot(data_list, labels=[f'BN{j:03d}' for j in range(1,11)],
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)
    ax.set_title(f'{short}')
    ax.set_ylabel(col.split('[')[1].rstrip(']'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
axes[5].axis('off')
fig.suptitle('Building-Level Energy Distribution Comparison', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig8_building_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig8_building_comparison.png")

# =====================================================
# FIGURE 9: Scatter plots - Key Energy-Weather Relationships
# =====================================================
print("Creating Figure 9: Energy-Weather Scatter Plots...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
scatter_pairs = [
    ('Temperature [°F]', 'Electricity [kW]', 'Temp vs Electricity'),
    ('Temperature [°F]', 'Cooling Energy [Ton]', 'Temp vs Cooling'),
    ('Temperature [°F]', 'Heat [mmBTU]', 'Temp vs Heat'),
    ('Humidity [%]', 'Electricity [kW]', 'Humidity vs Electricity'),
    ('Temperature [°F]', 'PV Power Generation [kW]', 'Temp vs PV'),
]
for i, (wx, ey, title) in enumerate(scatter_pairs):
    ax = axes[i]
    ax.scatter(merged[wx], merged[ey], s=1, alpha=0.3, color=colors[i % len(colors)])
    ax.set_xlabel(wx)
    ax.set_ylabel(ey.split('[')[0].strip())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Add correlation
    r = merged[wx].corr(merged[ey])
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[5].axis('off')
fig.suptitle('Energy-Weather Relationships (Total Aggregate)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig9_energy_weather_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig9_energy_weather_scatter.png")

# =====================================================
# FIGURE 10: Cross-Building Correlation Heatmap
# =====================================================
print("Creating Figure 10: Cross-Building Correlation...")
# Electricity correlation across buildings
elec_df = pd.DataFrame()
for k in sorted(buildings.keys()):
    elec_df[k] = buildings[k]['Electricity [kW]'].values
cross_corr = elec_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cross_corr, annot=True, fmt='.3f', cmap='YlOrRd', 
            square=True, linewidths=0.5, ax=ax, vmin=0.8, vmax=1.0,
            annot_kws={'size': 8})
ax.set_title('Cross-Building Electricity Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig10_cross_building_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig10_cross_building_correlation.png")

# Save cross-building correlation
cross_corr.to_csv(os.path.join(OUT, 'cross_building_electricity_correlation.csv'))

# =====================================================
# FIGURE 11: Seasonal Decomposition (Electricity)
# =====================================================
print("Creating Figure 11: Seasonal Decomposition...")
from scipy import signal

# Weekly rolling average for trend
elec_series = total_energy.set_index('datetime')['Electricity [kW]']
trend = elec_series.rolling(window=24*7, center=True).mean()
detrended = elec_series - trend
seasonal = detrended.groupby(detrended.index.hour).transform('mean')
residual = detrended - seasonal

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(elec_series.index, elec_series.values, linewidth=0.3, color='blue')
axes[0].set_title('Original: Electricity [kW]')
axes[0].set_ylabel('kW')
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend.index, trend.values, linewidth=1, color='red')
axes[1].set_title('Trend (7-day rolling mean)')
axes[1].set_ylabel('kW')
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal.index, seasonal.values, linewidth=0.3, color='green')
axes[2].set_title('Seasonal Component (hourly)')
axes[2].set_ylabel('kW')
axes[2].grid(True, alpha=0.3)

axes[3].plot(residual.index, residual.values, linewidth=0.3, color='purple')
axes[3].set_title('Residual')
axes[3].set_ylabel('kW')
axes[3].grid(True, alpha=0.3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].set_xlabel('Month (2014)')
fig.suptitle('Seasonal Decomposition: Total Electricity', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig11_seasonal_decomposition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig11_seasonal_decomposition.png")

# =====================================================
# FIGURE 12: PV Generation Pattern
# =====================================================
print("Creating Figure 12: PV Generation Pattern...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PV by hour
pv_hourly = total_energy.groupby('hour_of_day')['PV Power Generation [kW]'].mean()
axes[0].bar(range(24), pv_hourly, color='gold', edgecolor='orange')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Average PV [kW]')
axes[0].set_title('Average PV Generation by Hour')
axes[0].set_xticks(range(0, 24, 2))
axes[0].grid(True, alpha=0.3, axis='y')

# PV by month
pv_monthly = total_energy.groupby('month')['PV Power Generation [kW]'].mean()
axes[1].bar(range(1, 13), pv_monthly, color='gold', edgecolor='orange')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_names, rotation=45)
axes[1].set_ylabel('Average PV [kW]')
axes[1].set_title('Average PV Generation by Month')
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Photovoltaic Power Generation Patterns', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig12_pv_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig12_pv_patterns.png")

# =====================================================
# FIGURE 13: Distribution Plots (Histograms)
# =====================================================
print("Creating Figure 13: Distribution Plots...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    ax.hist(total_energy[col], bins=50, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(total_energy[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={total_energy[col].mean():.1f}')
    ax.set_title(f'Distribution: {short}')
    ax.set_xlabel(col.split('[')[1].rstrip(']'))
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
axes[5].axis('off')
fig.suptitle('Energy Variable Distributions (Total Aggregate)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig13_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig13_distributions.png")

# =====================================================
# FIGURE 14: Weekday vs Weekend Patterns
# =====================================================
print("Creating Figure 14: Weekday vs Weekend...")
total_energy['dayofweek'] = total_energy['datetime'].dt.dayofweek
total_energy['is_weekend'] = total_energy['dayofweek'] >= 5

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (col, short) in enumerate(zip(energy_cols, energy_short)):
    ax = axes[i]
    weekday = total_energy[~total_energy['is_weekend']].groupby('hour_of_day')[col].mean()
    weekend = total_energy[total_energy['is_weekend']].groupby('hour_of_day')[col].mean()
    ax.plot(range(24), weekday, label='Weekday', linewidth=2, marker='o', markersize=2)
    ax.plot(range(24), weekend, label='Weekend', linewidth=2, marker='s', markersize=2)
    ax.set_title(f'{short}')
    ax.set_xlabel('Hour')
    ax.set_ylabel(col.split('[')[1].rstrip(']'))
    ax.legend()
    ax.set_xticks(range(0, 24, 4))
    ax.grid(True, alpha=0.3)
axes[5].axis('off')
fig.suptitle('Weekday vs Weekend Diurnal Profiles (Total Aggregate)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig14_weekday_weekend.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig14_weekday_weekend.png")

# =====================================================
# FIGURE 15: Hierarchical Structure Visualization
# =====================================================
print("Creating Figure 15: Hierarchical Structure...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Show one week of electricity for all buildings + CN01 + Total
week_mask = (total_energy['datetime'] >= '2014-07-01') & (total_energy['datetime'] < '2014-07-08')
for k in sorted(buildings.keys()):
    mask = (buildings[k]['datetime'] >= '2014-07-01') & (buildings[k]['datetime'] < '2014-07-08')
    axes[0].plot(buildings[k].loc[mask, 'datetime'], buildings[k].loc[mask, 'Electricity [kW]'],
                 linewidth=0.8, alpha=0.7, label=k)
axes[0].set_title('Individual Building Electricity (July 1-7, 2014)')
axes[0].set_ylabel('kW')
axes[0].legend(ncol=5, fontsize=7, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Community and Total
mask_cn = (cn01['datetime'] >= '2014-07-01') & (cn01['datetime'] < '2014-07-08')
mask_t = (total_energy['datetime'] >= '2014-07-01') & (total_energy['datetime'] < '2014-07-08')
axes[1].plot(cn01.loc[mask_cn, 'datetime'], cn01.loc[mask_cn, 'Electricity [kW]'],
             linewidth=2, label='CN01 (Community)', color='blue')
axes[1].plot(total_energy.loc[mask_t, 'datetime'], total_energy.loc[mask_t, 'Electricity [kW]'],
             linewidth=2, linestyle='--', label='Total', color='red')
axes[1].set_title('Community & Total Aggregate Electricity (July 1-7, 2014)')
axes[1].set_ylabel('kW')
axes[1].set_xlabel('Date')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Hierarchical Data Structure: Building → Community → Total', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig15_hierarchical_structure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig15_hierarchical_structure.png")

# =====================================================
# FIGURE 16: Building Contribution Stacked Area
# =====================================================
print("Creating Figure 16: Building Contribution...")
# Daily electricity by building
daily_data = {}
for k in sorted(buildings.keys()):
    daily = buildings[k].set_index('datetime')['Electricity [kW]'].resample('D').mean()
    daily_data[k] = daily

daily_df = pd.DataFrame(daily_data)

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(daily_df.index, [daily_df[c] for c in daily_df.columns],
             labels=daily_df.columns, alpha=0.8)
ax.set_title('Building Contribution to Total Electricity (Daily Average)', fontsize=14)
ax.set_xlabel('Date (2014)')
ax.set_ylabel('Electricity [kW]')
ax.legend(loc='upper left', ncol=5, fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig16_building_contribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig16_building_contribution.png")

# =====================================================
# Additional analysis: Per-building statistics
# =====================================================
print("\nComputing per-building statistics...")
building_stats = {}
for k in sorted(buildings.keys()):
    stats = {}
    for col in energy_cols:
        stats[col] = {
            'mean': float(buildings[k][col].mean()),
            'std': float(buildings[k][col].std()),
            'min': float(buildings[k][col].min()),
            'max': float(buildings[k][col].max()),
            'annual_sum': float(buildings[k][col].sum()),
        }
    building_stats[k] = stats

with open(os.path.join(OUT, 'building_statistics.json'), 'w') as f:
    json.dump(building_stats, f, indent=2)

# Print key correlation findings
print("\n--- Key Correlation Findings ---")
print(f"Temperature vs Electricity: r = {merged['Temperature [°F]'].corr(merged['Electricity [kW]']):.4f}")
print(f"Temperature vs Cooling: r = {merged['Temperature [°F]'].corr(merged['Cooling Energy [Ton]']):.4f}")
print(f"Temperature vs Heat: r = {merged['Temperature [°F]'].corr(merged['Heat [mmBTU]']):.4f}")
print(f"Temperature vs PV: r = {merged['Temperature [°F]'].corr(merged['PV Power Generation [kW]']):.4f}")
print(f"Temperature vs GHG: r = {merged['Temperature [°F]'].corr(merged['Greenhouse Gas Emission [Ton]']):.4f}")
print(f"Humidity vs Electricity: r = {merged['Humidity [%]'].corr(merged['Electricity [kW]']):.4f}")
print(f"Electricity vs GHG: r = {merged['Electricity [kW]'].corr(merged['Greenhouse Gas Emission [Ton]']):.4f}")
print(f"Electricity vs Cooling: r = {merged['Electricity [kW]'].corr(merged['Cooling Energy [Ton]']):.4f}")

print("\n✓ All figures and analyses complete!")
