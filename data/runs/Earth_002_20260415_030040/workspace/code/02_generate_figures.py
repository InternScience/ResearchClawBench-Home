"""
Generate all figures for the mangrove composite risk index research report.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.spatial import cKDTree
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("Loading processed mangrove risk data...")
mangroves = gpd.read_parquet('outputs/mangrove_full_risk.parquet')
print(f"Loaded {len(mangroves)} mangrove points")

with open('outputs/risk_summary_stats.json', 'r') as f:
    summary_stats = json.load(f)

countries = gpd.read_file('data/ecosystem/UCSC_CWON_countrybounds.gpkg')
print(f"Loaded {len(countries)} country boundaries")

print("Performing spatial join with country boundaries...")
mangroves_with_countries = gpd.sjoin(mangroves, countries[['ISO3', 'Country', 'Mang_Ha_2015', 
                                                             'Risk_Pop_2015', 'Risk_Stock_2015',
                                                             'Ben_Pop_2015', 'Ben_Stock_2015',
                                                             'geometry']], 
                                      how='left', predicate='within')
print(f"Mangroves matched to countries: {mangroves_with_countries['ISO3'].notna().sum()}")

mangroves_with_countries.to_parquet('outputs/mangrove_enriched.parquet')

lat_bins_plot = np.arange(-60, 61, 2)
lon_bins_plot = np.arange(-180, 181, 2)

def create_global_grid(values, lons, lats, lon_bins, lat_bins):
    grid, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins], weights=values)
    counts, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    counts = np.maximum(counts, 1)
    return grid / counts

print("\nGenerating Figure 1: Global SLR rates under different SSPs...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ssps = ['SSP245', 'SSP370', 'SSP585']
ssp_titles = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

for ax, ssp_col, ssp_title in zip(axes, ssps, ssp_titles):
    slr_col = f'slr_{ssp_col}'
    grid = create_global_grid(mangroves[slr_col].values, 
                               mangroves.geometry.x.values,
                               mangroves.geometry.y.values,
                               lon_bins_plot, lat_bins_plot)
    
    im = ax.pcolormesh(lon_bins_plot, lat_bins_plot, grid.T,
                       cmap='YlOrRd', vmin=0, vmax=25, shading='auto')
    ax.set_title(ssp_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax, label='SLR Rate (mm/yr)', shrink=0.8)

plt.suptitle('Projected Sea Level Rise Rates at Mangrove Locations by 2100', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure1_slr_rates.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_slr_rates.png")

print("\nGenerating Figure 2: TC frequency and wind exposure...")

import xarray as xr
tc_ds_path = 'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc'
tc_ds = xr.open_dataset(tc_ds_path)
tc_lats = tc_ds['lat'].values
tc_lons = tc_ds['lon'].values
tc_winds = tc_ds['wind'].values

lat_bins_tc = np.arange(-60, 61, 2)
lon_bins_tc = np.arange(-180, 181, 2)

tc_freq_grid, _, _ = np.histogram2d(tc_lons, tc_lats, bins=[lon_bins_tc, lat_bins_tc])
tc_wind_sum, _, _ = np.histogram2d(tc_lons, tc_lats, bins=[lon_bins_tc, lat_bins_tc], weights=tc_winds)
tc_count = np.maximum(tc_freq_grid, 1)
tc_mean_wind_grid = tc_wind_sum / tc_count
tc_annual_freq_grid = tc_freq_grid / 165

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].pcolormesh(lon_bins_tc, lat_bins_tc, tc_annual_freq_grid.T,
                          cmap='YlOrBr', vmin=0, vmax=0.5, shading='auto')
axes[0].set_title('Historical TC Annual Frequency', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
axes[0].axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)
axes[0].set_xlim(-180, 180)
axes[0].set_ylim(-60, 60)
plt.colorbar(im1, ax=axes[0], label='Events per year', shrink=0.8)

im2 = axes[1].pcolormesh(lon_bins_tc, lat_bins_tc, tc_mean_wind_grid.T,
                          cmap='Reds', vmin=0, vmax=80, shading='auto')
axes[1].set_title('Mean TC Wind Speed', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
axes[1].axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
axes[1].axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)
axes[1].set_xlim(-180, 180)
axes[1].set_ylim(-60, 60)
plt.colorbar(im2, ax=axes[1], label='Wind speed (m/s)', shrink=0.8)

plt.suptitle('Historical Tropical Cyclone Exposure (1850-2014)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure2_tc_exposure.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_tc_exposure.png")

print("\nGenerating Figure 3: Composite risk maps for each SSP...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, ssp_col, ssp_title in zip(axes, ssps, ssp_titles):
    risk_col = f'composite_risk_{ssp_col}'
    grid = create_global_grid(mangroves[risk_col].values,
                               mangroves.geometry.x.values,
                               mangroves.geometry.y.values,
                               lon_bins_plot, lat_bins_plot)
    
    im = ax.pcolormesh(lon_bins_plot, lat_bins_plot, grid.T,
                       cmap='RdYlGn_r', vmin=0, vmax=1, shading='auto')
    ax.set_title(ssp_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    cbar = plt.colorbar(im, ax=ax, label='Composite Risk Index', shrink=0.8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', 'Moderate-Low', 'Moderate', 'High', 'Very High'])

plt.suptitle('Composite Mangrove Risk Index by 2100', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure3_composite_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_composite_risk.png")

print("\nGenerating Figure 4: Risk distribution comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

colors_ssp = ['#2ca02c', '#ff7f0e', '#d62728']
for ax, ssp_col, ssp_title, color in zip(axes, ssps, ssp_titles, colors_ssp):
    risk_col = f'composite_risk_{ssp_col}'
    ax.hist(mangroves[risk_col].values, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax.set_title(ssp_title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Risk Index')
    ax.set_ylabel('Number of Mangrove Points')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

plt.suptitle('Distribution of Composite Risk Index Across SSP Scenarios', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('report/images/figure4_risk_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_risk_distributions.png")

print("\nGenerating Figure 5: Regional risk breakdown...")

country_risk = mangroves_with_countries.groupby('ISO3').agg({
    'composite_risk_SSP245': 'mean',
    'composite_risk_SSP370': 'mean',
    'composite_risk_SSP585': 'mean',
    'uid': 'count'
}).rename(columns={'uid': 'n_points'})

country_risk = country_risk.dropna(subset=['composite_risk_SSP245'])
country_risk = country_risk[country_risk['n_points'] >= 5]
country_risk = country_risk.sort_values('composite_risk_SSP585', ascending=False)

top15 = country_risk.head(15)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(top15))
width = 0.25

bars1 = ax.bar(x - width, top15['composite_risk_SSP245'], width, label='SSP2-4.5', color='#2ca02c', edgecolor='black', linewidth=0.3)
bars2 = ax.bar(x, top15['composite_risk_SSP370'], width, label='SSP3-7.0', color='#ff7f0e', edgecolor='black', linewidth=0.3)
bars3 = ax.bar(x + width, top15['composite_risk_SSP585'], width, label='SSP5-8.5', color='#d62728', edgecolor='black', linewidth=0.3)

ax.set_xlabel('Country (ISO3)')
ax.set_ylabel('Composite Risk Index')
ax.set_title('Top 15 Highest-Risk Countries by 2100', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top15.index, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High risk threshold')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/figure5_top_risk_countries.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure5_top_risk_countries.png")

print("\nGenerating Figure 6: Risk component comparison (SLR vs TC)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, ssp_col, ssp_title in zip(axes, ssps, ssp_titles):
    slr_risk_col = f'slr_risk_{ssp_col}'
    sample_size = min(5000, len(mangroves))
    idx = np.random.choice(len(mangroves), sample_size, replace=False)
    
    scatter = ax.scatter(mangroves.iloc[idx][slr_risk_col], 
                         mangroves.iloc[idx]['tc_risk'],
                         c=mangroves.iloc[idx][f'composite_risk_{ssp_col}'],
                         cmap='RdYlGn_r', alpha=0.5, s=10, vmin=0, vmax=1)
    ax.set_xlabel('SLR Risk Component')
    ax.set_ylabel('TC Risk Component')
    ax.set_title(ssp_title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(scatter, ax=ax, label='Composite Risk', shrink=0.8)

plt.suptitle('SLR vs TC Risk Components by Scenario', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('report/images/figure6_risk_components.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure6_risk_components.png")

print("\nGenerating Figure 7: Ecosystem services at risk...")

country_services = mangroves_with_countries.groupby('ISO3').agg({
    'Mang_Ha_2015': 'first',
    'Risk_Pop_2015': 'first',
    'Risk_Stock_2015': 'first',
    'Ben_Pop_2015': 'first',
    'Ben_Stock_2015': 'first',
    'composite_risk_SSP585': 'mean',
    'uid': 'count'
}).rename(columns={'uid': 'n_points'})

country_services = country_services.dropna(subset=['composite_risk_SSP585', 'Mang_Ha_2015'])

country_services['risk_category'] = pd.cut(country_services['composite_risk_SSP585'], 
                                            bins=[0, 0.4, 0.7, 1.0],
                                            labels=['Low', 'Moderate', 'High'])

risk_summary = country_services.groupby('risk_category', observed=True).agg({
    'Mang_Ha_2015': 'sum',
    'Risk_Pop_2015': 'sum',
    'Risk_Stock_2015': 'sum',
    'Ben_Pop_2015': 'sum',
    'Ben_Stock_2015': 'sum',
    'n_points': 'sum'
})

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

colors = ['#2ca02c', '#ff7f0e', '#d62728']
axes[0, 0].bar(range(len(risk_summary)), risk_summary['Mang_Ha_2015'] / 1000, 
               color=colors, edgecolor='black', linewidth=0.5)
axes[0, 0].set_xlabel('Risk Category')
axes[0, 0].set_ylabel('Mangrove Area (thousand ha)')
axes[0, 0].set_title('Mangrove Area at Risk', fontweight='bold')
axes[0, 0].set_xticks(range(len(risk_summary)))
axes[0, 0].set_xticklabels(risk_summary.index)

axes[0, 1].bar(range(len(risk_summary)), risk_summary['Risk_Pop_2015'] / 1e6, 
               color=colors, edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Risk Category')
axes[0, 1].set_ylabel('Population (millions)')
axes[0, 1].set_title('Population at Risk from Coastal Hazards', fontweight='bold')
axes[0, 1].set_xticks(range(len(risk_summary)))
axes[0, 1].set_xticklabels(risk_summary.index)

axes[1, 0].bar(range(len(risk_summary)), risk_summary['Risk_Stock_2015'] / 1e9, 
               color=colors, edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Risk Category')
axes[1, 0].set_ylabel('Property Value (billion USD)')
axes[1, 0].set_title('Coastal Property Stock at Risk', fontweight='bold')
axes[1, 0].set_xticks(range(len(risk_summary)))
axes[1, 0].set_xticklabels(risk_summary.index)

axes[1, 1].bar(range(len(risk_summary)), risk_summary['Ben_Pop_2015'] / 1e6, 
               color=colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Risk Category')
axes[1, 1].set_ylabel('Beneficiary Population (millions)')
axes[1, 1].set_title('Population Benefiting from Mangrove Services', fontweight='bold')
axes[1, 1].set_xticks(range(len(risk_summary)))
axes[1, 1].set_xticklabels(risk_summary.index)

plt.suptitle('Ecosystem Services at Risk by Composite Risk Category (SSP5-8.5)', 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/figure7_ecosystem_services.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure7_ecosystem_services.png")

print("\nGenerating Figure 8: Risk summary statistics...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metrics = ['Mean Composite\nRisk', 'Mean SLR\nRisk', 'Mean TC\nRisk']
ssp_colors = ['#2ca02c', '#ff7f0e', '#d62728']

x = np.arange(len(metrics))
width = 0.25

for i, (ssp_col, ssp_label, color) in enumerate(zip(ssps, ssp_titles, ssp_colors)):
    means = [
        summary_stats[ssp_label]['composite_risk_mean'],
        summary_stats[ssp_label]['slr_risk_mean'],
        summary_stats[ssp_label]['tc_risk_mean']
    ]
    axes[0].bar(x + i * width, means, width, label=ssp_label, color=color, edgecolor='black', linewidth=0.5)

axes[0].set_xlabel('Risk Metric')
axes[0].set_ylabel('Mean Risk Score')
axes[0].set_title('Risk Metrics by SSP Scenario', fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics, fontsize=9)
axes[0].legend()
axes[0].set_ylim(0, 1.1)

risk_fractions = []
for ssp_label in ssp_titles:
    high = summary_stats[ssp_label]['high_risk_fraction']
    mod = summary_stats[ssp_label]['moderate_risk_fraction']
    low = summary_stats[ssp_label]['low_risk_fraction']
    risk_fractions.append([low, mod, high])

for ax, fractions, ssp_label in zip(axes[1:], risk_fractions, ssp_titles):
    wedges, texts, autotexts = ax.pie(fractions, labels=['Low', 'Moderate', 'High'],
                                       autopct='%1.1f%%', colors=['#2ca02c', '#ff7f0e', '#d62728'],
                                       startangle=90, textprops={'fontsize': 8})
    ax.set_title(f'{ssp_label}', fontweight='bold')

plt.suptitle('Summary Statistics and Risk Category Distribution', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure8_risk_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure8_risk_summary.png")

print("\nGenerating Figure 9: Latitudinal risk pattern...")

lat_band_width = 5
mangroves['lat_band'] = (mangroves.geometry.y // lat_band_width * lat_band_width).astype(int)

fig, ax = plt.subplots(figsize=(10, 5))

for ssp_col, ssp_label, color in zip(ssps, ssp_titles, ssp_colors):
    risk_col = f'composite_risk_{ssp_col}'
    lat_risk = mangroves.groupby('lat_band')[risk_col].mean()
    ax.plot(lat_risk.index, lat_risk.values, marker='o', markersize=4, 
            label=ssp_label, color=color, linewidth=2)

ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High risk threshold')
ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('Mean Composite Risk Index')
ax.set_title('Latitudinal Pattern of Mangrove Risk', fontsize=12, fontweight='bold')
ax.legend()
ax.set_xlim(-35, 35)

plt.tight_layout()
plt.savefig('report/images/figure9_latitudinal_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure9_latitudinal_pattern.png")

print("\nSaving detailed results tables...")

country_risk_table = country_risk.reset_index()
country_risk_table.to_csv('outputs/country_risk_table.csv', index=False)

summary_df = pd.DataFrame(summary_stats).T
summary_df.to_csv('outputs/risk_summary_table.csv')

risk_summary.to_csv('outputs/ecosystem_services_at_risk.csv')

print("All figures and tables generated successfully!")
print(f"\nFigures saved to report/images/:")
for f in sorted(os.listdir('report/images')):
    if f.endswith('.png'):
        print(f"  - {f}")
