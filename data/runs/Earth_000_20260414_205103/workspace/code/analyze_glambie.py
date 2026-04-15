#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Paths relative to workspace root
data_dir = Path('data/glambie/results/calendar_years')
output_dir = Path('outputs')
images_dir = Path('report/images')
images_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Region mapping
region_map = {
    '0_global': 'Global',
    '1_alaska': 'Alaska',
    '2_western_canada_us': 'Western Canada & US',
    '3_arctic_canada_north': 'Arctic Canada North',
    '4_arctic_canada_south': 'Arctic Canada South',
    '5_greenland_periphery': 'Greenland Periphery',
    '6_iceland': 'Iceland',
    '7_svalbard': 'Svalbard',
    '8_scandinavia': 'Scandinavia',
    '9_russian_arctic': 'Russian Arctic',
    '10_north_asia': 'Northern Asia',
    '11_central_europe': 'Central Europe',
    '12_caucasus_middle_east': 'Caucasus & Middle East',
    '13_central_asia': 'Central Asia',
    '14_south_asia_west': 'South Asia West',
    '15_south_asia_east': 'South Asia East',
    '16_low_latitudes': 'Low Latitudes',
    '17_southern_andes': 'Southern Andes',
    '18_new_zealand': 'New Zealand',
    '19_antarctic_and_subantarctic': 'Antarctic & Subantarctic'
}

# Load all data
dfs = {}
for f in data_dir.glob('*.csv'):
    region_key = f.stem
    if region_key in region_map:
        df = pd.read_csv(f)
        df['period_start'] = df['start_dates']
        df['period_end'] = df['end_dates']
        df['mid_year'] = (df['start_dates'] + df['end_dates']) / 2
        df['region_full'] = region_map[region_key]
        dfs[region_key] = df

# Global
global_df = dfs['0_global'].copy()
global_df.to_csv(output_dir / 'global_time_series.csv', index=False)

# Regional
regional_dfs = [df for key, df in dfs.items() if key != '0_global']
regional_data = pd.concat(regional_dfs, ignore_index=True)
regional_data.to_csv(output_dir / 'regional_time_series.csv', index=False)

# Compute 2000-2023 totals (sum annual changes 2000-01 to 2022-23)
mask_2000_2023 = (global_df['start_dates'] >= 2000.0) & (global_df['start_dates'] < 2023.0)
cumulative_gt_2000_2023 = global_df.loc[mask_2000_2023, 'combined_gt'].sum()
num_years = len(mask_2000_2023)
avg_rate_gt_yr = cumulative_gt_2000_2023 / num_years
avg_area = global_df.loc[mask_2000_2023, 'glacier_area'].mean()
avg_rate_mwe_yr = (cumulative_gt_2000_2023 / 1e9 / avg_area) * 1000  # m w.e./yr

summary = {
    'period': '2000-2023',
    'num_years': num_years,
    'cumulative_mass_loss_Gt': float(cumulative_gt_2000_2023),
    'average_rate_Gt_yr': float(avg_rate_gt_yr),
    'average_rate_mwe_yr': float(avg_rate_mwe_yr),
    'average_glacier_area_km2': float(avg_area)
}
with open(output_dir / 'summary_2000_2023.json', 'w') as fp:
    json.dump(summary, fp, indent=2)

print('Summary:', summary)

# Regional averages 2000-2023
regional_avg = regional_data[regional_data['period_start'] >= 2000.0].groupby('region_full').agg({
    'combined_gt': 'mean',
    'combined_mwe': 'mean',
    'glacier_area': 'mean'
}).round(3)
regional_avg.to_csv(output_dir / 'regional_averages_2000_2023.csv')

# Figures

plt.style.use('default')
sns.set_palette('husl')

# Fig 1: Global annual time series Gt/yr with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))
years = global_df['mid_year']
ax.errorbar(years, global_df['combined_gt'], yerr=global_df['combined_gt_errors'], fmt='o-', capsize=3, label='Mass change ±1σ')
ax.fill_between(years, global_df['combined_gt'] - global_df['combined_gt_errors'], global_df['combined_gt'] + global_df['combined_gt_errors'], alpha=0.3)
ax.set_xlabel('Year')
ax.set_ylabel('Mass change (Gt/yr)')
ax.set_title('Global Glacial Mass Change (Calendar Years)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(images_dir / 'global_annual_gt.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 2: Global cumulative mass loss
global_df['cumulative_gt'] = global_df['combined_gt'].cumsum()
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, global_df['cumulative_gt'], yerr=np.cumsum(global_df['combined_gt_errors']), fmt='o-', capsize=3)
ax.fill_between(years, global_df['cumulative_gt'] - np.cumsum(global_df['combined_gt_errors']), 
                global_df['cumulative_gt'] + np.cumsum(global_df['combined_gt_errors']), alpha=0.3)
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative mass change (Gt)')
ax.set_title('Global Cumulative Glacial Mass Loss')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(images_dir / 'global_cumulative_gt.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 3: Regional average rates bar plot
fig, ax = plt.subplots(figsize=(12, 8))
regional_avg['gt_per_yr'] = regional_avg['combined_gt']
colors = plt.cm.tab20(np.linspace(0, 1, len(regional_avg)))
regional_avg.sort_values('gt_per_yr').plot(kind='barh', x='region_full', y='gt_per_yr', ax=ax, color=colors)
ax.set_xlabel('Average mass change (Gt/yr, 2000+)')
ax.set_title('Average Annual Mass Change by Region (2000 onwards)')
plt.tight_layout()
plt.savefig(images_dir / 'regional_avg_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 4: Stacked regional contributions to global (approx proportional)
total_regional = regional_data.groupby(['period_start', 'region_full'])['combined_gt'].sum().reset_index()
pivot = total_regional.pivot(index='period_start', columns='region_full', values='combined_gt').fillna(0)
fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.8)
ax.set_xlabel('Period start year')
ax.set_ylabel('Mass change (Gt)')
ax.set_title('Stacked Regional Contributions to Global Mass Change')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(images_dir / 'regional_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

print('Figures saved to report/images/')
print('Data saved to outputs/')