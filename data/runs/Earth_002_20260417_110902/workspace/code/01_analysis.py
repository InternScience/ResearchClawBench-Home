#!/usr/bin/env python3
"""
Composite Risk Index for Global Mangrove Ecosystems:
Combining Tropical Cyclone Regime Shifts and Sea Level Rise

This script implements the full analysis pipeline:
1. TC risk component (frequency, damage index, risk index)
2. SLR risk component (median rates, threshold classification)
3. Composite Risk Index (CRI)
4. Country-level aggregation with ecosystem services
5. Visualization and figure generation
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_002_20260417_110902'
DATA = f'{BASE}/data'
OUT = f'{BASE}/outputs'
IMG = f'{BASE}/report/images'

print("=" * 60)
print("PHASE 1: Loading and processing data")
print("=" * 60)

# ============================================================
# 1. Load mangrove data
# ============================================================
print("\n--- Loading mangrove data ---")
mangroves = gpd.read_file(f'{DATA}/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
mangroves['mang_lon'] = mangroves.geometry.x
mangroves['mang_lat'] = mangroves.geometry.y
print(f"Loaded {len(mangroves)} mangrove points")
print(f"Lat range: {mangroves['mang_lat'].min():.2f} to {mangroves['mang_lat'].max():.2f}")
print(f"Lon range: {mangroves['mang_lon'].min():.2f} to {mangroves['mang_lon'].max():.2f}")

# ============================================================
# 2. Load and process TC tracks
# ============================================================
print("\n--- Loading TC track data ---")
tc_ds = xr.open_dataset(f'{DATA}/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lat = tc_ds['lat'].values
tc_lon = tc_ds['lon'].values
tc_wind = tc_ds['wind'].values  # m/s
tc_ds.close()

print(f"Loaded {len(tc_wind)} TC track points")
print(f"Wind speed range: {tc_wind.min():.1f} - {tc_wind.max():.1f} m/s")

# Saffir-Simpson categories based on wind speed (m/s)
# Cat 1: 33-42.5 m/s, Cat 2: 42.5-49.5, Cat 3: 49.5-58, Cat 4: 58-70, Cat 5: >70
def classify_saffir_simpson(wind_ms):
    """Classify wind speed into Saffir-Simpson categories."""
    cats = np.zeros(len(wind_ms), dtype=int)
    cats[(wind_ms >= 33) & (wind_ms < 42.5)] = 1
    cats[(wind_ms >= 42.5) & (wind_ms < 49.5)] = 2
    cats[(wind_ms >= 49.5) & (wind_ms < 58)] = 3
    cats[(wind_ms >= 58) & (wind_ms < 70)] = 4
    cats[wind_ms >= 70] = 5
    return cats

tc_cats = classify_saffir_simpson(tc_wind)
for cat in range(1, 6):
    n = np.sum(tc_cats == cat)
    print(f"  Category {cat}: {n} track points ({100*n/len(tc_cats):.1f}%)")

# ============================================================
# 3. Grid TC frequency at 1° resolution
# ============================================================
print("\n--- Computing TC frequency on 1° grid ---")

# Create 1° grid
lon_bins = np.arange(-180, 181, 1)
lat_bins = np.arange(-90, 91, 1)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

# Historical period: 1850-2014 = 165 years
# But we use the data as-is (reduced/sampled track points)
# The description says historical 1850-2014
hist_years = 165  # years of historical simulation

# Count TC track points per grid cell per category
tc_freq = {}  # {cat: 2D array of annual frequency}
for cat in range(1, 6):
    mask = tc_cats == cat
    if mask.sum() == 0:
        tc_freq[cat] = np.zeros((len(lat_centers), len(lon_centers)))
        continue
    
    h, _, _ = np.histogram2d(
        tc_lat[mask], tc_lon[mask],
        bins=[lat_bins, lon_bins]
    )
    # Convert to annual frequency
    tc_freq[cat] = h / hist_years
    print(f"  Cat {cat}: max annual frequency = {tc_freq[cat].max():.3f}")

# Total TC frequency (all categories)
tc_freq_total = sum(tc_freq.values())
# Major TC frequency (Cat 3+)
tc_freq_major = tc_freq[3] + tc_freq[4] + tc_freq[5]

print(f"  Total: max annual frequency = {tc_freq_total.max():.3f}")
print(f"  Major (Cat 3+): max annual frequency = {tc_freq_major.max():.3f}")

# ============================================================
# 4. Calculate TC Damage Index (TCDI) per category
# ============================================================
print("\n--- Computing TC Damage Index (TCDI) ---")

# Following Mo et al. (2023): TCDI ratios for Cat 3:4:5 = 1:13:29
# Cat 1-2 cause negligible damage
# We normalize TCDI so that Cat 5 = 1.0
tcdi_ratios = {1: 0.0, 2: 0.0, 3: 1.0/29.0, 4: 13.0/29.0, 5: 29.0/29.0}
print(f"  TCDI ratios: {tcdi_ratios}")

# ============================================================
# 5. Calculate TC Risk Index (TCRI) per grid cell
# ============================================================
print("\n--- Computing TC Risk Index (TCRI) ---")

# TCRI = sum over categories of (TCDI_cat * Freq_cat)
tcri_grid = np.zeros((len(lat_centers), len(lon_centers)))
for cat in range(1, 6):
    tcri_grid += tcdi_ratios[cat] * tc_freq[cat]

print(f"  TCRI range: {tcri_grid.min():.6f} - {tcri_grid.max():.6f}")
print(f"  Non-zero cells: {np.sum(tcri_grid > 0)}")

# ============================================================
# 6. Assign TC risk to mangrove locations
# ============================================================
print("\n--- Assigning TC risk to mangrove points ---")

def get_grid_value(lats, lons, grid, lat_centers, lon_centers):
    """Get grid values at given lat/lon points."""
    lat_idx = np.searchsorted(lat_centers, lats) - 1
    lon_idx = np.searchsorted(lon_centers, lons) - 1
    lat_idx = np.clip(lat_idx, 0, len(lat_centers) - 1)
    lon_idx = np.clip(lon_idx, 0, len(lon_centers) - 1)
    return grid[lat_idx, lon_idx]

mangroves['tcri'] = get_grid_value(
    mangroves['mang_lat'].values,
    mangroves['mang_lon'].values,
    tcri_grid, lat_centers, lon_centers
)

# Also get per-category frequencies
for cat in range(1, 6):
    mangroves[f'tc_freq_cat{cat}'] = get_grid_value(
        mangroves['mang_lat'].values,
        mangroves['mang_lon'].values,
        tc_freq[cat], lat_centers, lon_centers
    )

mangroves['tc_freq_total'] = get_grid_value(
    mangroves['mang_lat'].values,
    mangroves['mang_lon'].values,
    tc_freq_total, lat_centers, lon_centers
)

mangroves['tc_freq_major'] = get_grid_value(
    mangroves['mang_lat'].values,
    mangroves['mang_lon'].values,
    tc_freq_major, lat_centers, lon_centers
)

print(f"  Mangroves with TC exposure (TCRI > 0): {(mangroves['tcri'] > 0).sum()} ({100*(mangroves['tcri'] > 0).mean():.1f}%)")
print(f"  TCRI stats for mangroves:")
print(f"    Mean: {mangroves['tcri'].mean():.6f}")
print(f"    Median: {mangroves['tcri'].median():.6f}")
print(f"    Max: {mangroves['tcri'].max():.6f}")

# ============================================================
# 7. Load and process SLR data
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Sea Level Rise Risk Component")
print("=" * 60)

ssp_scenarios = ['ssp245', 'ssp370', 'ssp585']
ssp_labels = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}

slr_data = {}
for ssp in ssp_scenarios:
    print(f"\n--- Loading SLR data for {ssp_labels[ssp]} ---")
    ds = xr.open_dataset(f'{DATA}/slr/total_{ssp}_medium_confidence_rates.nc')
    
    slr_lat = ds['lat'].values
    slr_lon = ds['lon'].values
    
    # Get median (quantile 0.5)
    q50_idx = np.argmin(np.abs(ds['quantiles'].values - 0.5))
    
    # Get rates for key years
    years = ds['years'].values
    yr_2020_idx = np.where(years == 2020)[0][0]
    yr_2050_idx = np.where(years == 2050)[0][0]
    yr_2100_idx = np.where(years == 2100)[0][0]
    
    # Extract median rates
    rate_2020 = ds['sea_level_change_rate'].isel(quantiles=q50_idx, years=yr_2020_idx).values
    rate_2050 = ds['sea_level_change_rate'].isel(quantiles=q50_idx, years=yr_2050_idx).values
    rate_2100 = ds['sea_level_change_rate'].isel(quantiles=q50_idx, years=yr_2100_idx).values
    
    # Also get 5th and 95th percentile for uncertainty
    q05_idx = np.argmin(np.abs(ds['quantiles'].values - 0.05))
    q95_idx = np.argmin(np.abs(ds['quantiles'].values - 0.95))
    rate_2100_q05 = ds['sea_level_change_rate'].isel(quantiles=q05_idx, years=yr_2100_idx).values
    rate_2100_q95 = ds['sea_level_change_rate'].isel(quantiles=q95_idx, years=yr_2100_idx).values
    
    slr_data[ssp] = {
        'lat': slr_lat,
        'lon': slr_lon,
        'rate_2020': rate_2020,
        'rate_2050': rate_2050,
        'rate_2100': rate_2100,
        'rate_2100_q05': rate_2100_q05,
        'rate_2100_q95': rate_2100_q95
    }
    
    print(f"  {len(slr_lat)} SLR locations")
    print(f"  Median rate 2100: min={np.nanmin(rate_2100):.1f}, mean={np.nanmean(rate_2100):.1f}, max={np.nanmax(rate_2100):.1f} mm/yr")
    
    ds.close()

# ============================================================
# 8. Match SLR to mangrove locations
# ============================================================
print("\n--- Matching SLR to mangrove locations ---")

# Build KD-tree from SLR locations
slr_lat_ref = slr_data['ssp245']['lat']
slr_lon_ref = slr_data['ssp245']['lon']

# Convert to radians for haversine-like distance
slr_coords = np.column_stack([slr_lat_ref, slr_lon_ref])
mang_coords = np.column_stack([mangroves['mang_lat'].values, mangroves['mang_lon'].values])

tree = cKDTree(slr_coords)
distances, indices = tree.query(mang_coords, k=1)

print(f"  Max distance to nearest SLR point: {distances.max():.2f}°")
print(f"  Mean distance: {distances.mean():.2f}°")

# Assign SLR rates to mangroves
for ssp in ssp_scenarios:
    mangroves[f'slr_rate_2020_{ssp}'] = slr_data[ssp]['rate_2020'][indices]
    mangroves[f'slr_rate_2050_{ssp}'] = slr_data[ssp]['rate_2050'][indices]
    mangroves[f'slr_rate_2100_{ssp}'] = slr_data[ssp]['rate_2100'][indices]
    mangroves[f'slr_rate_2100_q05_{ssp}'] = slr_data[ssp]['rate_2100_q05'][indices]
    mangroves[f'slr_rate_2100_q95_{ssp}'] = slr_data[ssp]['rate_2100_q95'][indices]
    
    r2100 = mangroves[f'slr_rate_2100_{ssp}']
    print(f"  {ssp_labels[ssp]} rate at 2100: mean={r2100.mean():.2f}, median={r2100.median():.2f} mm/yr")

# ============================================================
# 9. SLR risk classification
# ============================================================
print("\n--- SLR risk classification ---")

# Following Saintilan et al. (2023):
# < 4 mm/yr: Low risk (mangroves can likely keep pace)
# 4-7 mm/yr: Moderate risk (deficit likely)
# > 7 mm/yr: High risk (highly likely loss)
def classify_slr_risk(rate):
    """Classify SLR rate into risk categories."""
    risk = np.zeros(len(rate))
    risk[(rate >= 4) & (rate < 7)] = 1  # Moderate
    risk[rate >= 7] = 2  # High
    return risk

def slr_risk_score(rate):
    """Continuous SLR risk score normalized to [0, 1].
    Uses piecewise linear: 0 at 0 mm/yr, 0.5 at 4 mm/yr, 1.0 at 7+ mm/yr."""
    score = np.zeros(len(rate))
    # Below 4: linear from 0 to 0.5
    mask_low = (rate >= 0) & (rate < 4)
    score[mask_low] = rate[mask_low] / 8.0  # 0 at 0, 0.5 at 4
    # 4 to 7: linear from 0.5 to 1.0
    mask_mod = (rate >= 4) & (rate < 7)
    score[mask_mod] = 0.5 + (rate[mask_mod] - 4) / 6.0  # 0.5 at 4, 1.0 at 7
    # Above 7: capped at 1.0
    score[rate >= 7] = 1.0
    # Negative rates: 0
    score[rate < 0] = 0.0
    return score

for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    mangroves[f'slr_risk_cat_{ssp}'] = classify_slr_risk(rate)
    mangroves[f'slr_risk_score_{ssp}'] = slr_risk_score(rate)
    
    n_low = np.sum(rate < 4)
    n_mod = np.sum((rate >= 4) & (rate < 7))
    n_high = np.sum(rate >= 7)
    print(f"  {ssp_labels[ssp]}: Low={n_low} ({100*n_low/len(rate):.1f}%), "
          f"Moderate={n_mod} ({100*n_mod/len(rate):.1f}%), "
          f"High={n_high} ({100*n_high/len(rate):.1f}%)")

# ============================================================
# 10. Composite Risk Index (CRI)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Composite Risk Index")
print("=" * 60)

# Normalize TCRI to [0, 1]
tcri_vals = mangroves['tcri'].values
tcri_max = np.percentile(tcri_vals[tcri_vals > 0], 99) if np.sum(tcri_vals > 0) > 0 else 1.0
mangroves['tcri_norm'] = np.clip(tcri_vals / tcri_max, 0, 1)

print(f"  TCRI normalization: 99th percentile = {tcri_max:.6f}")
print(f"  TCRI_norm stats: mean={mangroves['tcri_norm'].mean():.3f}, "
      f"median={mangroves['tcri_norm'].median():.3f}")

# CRI = 0.5 * TC_norm + 0.5 * SLR_norm (equal weighting)
# Also compute with different weights for sensitivity
for ssp in ssp_scenarios:
    # Equal weighting
    mangroves[f'cri_{ssp}'] = 0.5 * mangroves['tcri_norm'] + 0.5 * mangroves[f'slr_risk_score_{ssp}']
    
    # TC-dominated
    mangroves[f'cri_tc_dom_{ssp}'] = 0.7 * mangroves['tcri_norm'] + 0.3 * mangroves[f'slr_risk_score_{ssp}']
    
    # SLR-dominated
    mangroves[f'cri_slr_dom_{ssp}'] = 0.3 * mangroves['tcri_norm'] + 0.7 * mangroves[f'slr_risk_score_{ssp}']
    
    cri = mangroves[f'cri_{ssp}']
    print(f"  {ssp_labels[ssp]} CRI: mean={cri.mean():.3f}, median={cri.median():.3f}, max={cri.max():.3f}")

# Risk categories for CRI
def classify_cri(cri):
    cats = pd.cut(cri, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                  include_lowest=True)
    return cats

for ssp in ssp_scenarios:
    mangroves[f'cri_cat_{ssp}'] = classify_cri(mangroves[f'cri_{ssp}'])

# ============================================================
# 11. Country-level aggregation
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Country-level aggregation")
print("=" * 60)

# Load country boundaries
countries = gpd.read_file(f'{DATA}/ecosystem/UCSC_CWON_countrybounds.gpkg')
print(f"Loaded {len(countries)} countries with ecosystem data")

# Spatial join: assign mangrove points to countries
mangroves_gdf = mangroves.copy()
mangroves_gdf = mangroves_gdf.set_geometry('geometry')
mangroves_country = gpd.sjoin(mangroves_gdf, countries[['Country', 'ISO3', 'geometry']], 
                               how='left', predicate='within')

# Some points may not fall within country polygons - use nearest
unmatched = mangroves_country['Country'].isna()
print(f"  Matched: {(~unmatched).sum()}, Unmatched: {unmatched.sum()}")

if unmatched.sum() > 0:
    # For unmatched, use nearest country
    unmatched_coords = np.column_stack([
        mangroves_country.loc[unmatched, 'mang_lat'].values,
        mangroves_country.loc[unmatched, 'mang_lon'].values
    ])
    
    # Get country centroids
    country_centroids = countries.geometry.centroid
    country_coords = np.column_stack([country_centroids.y, country_centroids.x])
    
    tree_country = cKDTree(country_coords)
    _, nearest_idx = tree_country.query(unmatched_coords, k=1)
    
    mangroves_country.loc[unmatched, 'Country'] = countries.iloc[nearest_idx]['Country'].values
    mangroves_country.loc[unmatched, 'ISO3'] = countries.iloc[nearest_idx]['ISO3'].values

# Country-level statistics
country_stats = []
for ssp in ssp_scenarios:
    grouped = mangroves_country.groupby('Country').agg({
        f'cri_{ssp}': ['mean', 'median', 'max', 'std'],
        'tcri_norm': 'mean',
        f'slr_risk_score_{ssp}': 'mean',
        f'slr_rate_2100_{ssp}': 'mean',
        'tc_freq_major': 'mean',
        'uid': 'count'
    }).reset_index()
    
    grouped.columns = ['Country', f'cri_mean_{ssp}', f'cri_median_{ssp}', 
                       f'cri_max_{ssp}', f'cri_std_{ssp}',
                       f'tcri_mean_{ssp}', f'slr_score_mean_{ssp}',
                       f'slr_rate_mean_{ssp}', f'tc_freq_major_mean_{ssp}',
                       f'n_points_{ssp}']
    
    if len(country_stats) == 0:
        country_stats = grouped
    else:
        country_stats = country_stats.merge(grouped, on='Country', how='outer')

# Merge with ecosystem service data
eco_cols = ['Country', 'ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020', 
            'Ben_Pop_2020', 'Ben_Stock_2020']
country_stats = country_stats.merge(countries[eco_cols], on='Country', how='left')

# Calculate ecosystem services at risk (weighted by CRI)
for ssp in ssp_scenarios:
    if 'Mang_Ha_2020' in country_stats.columns:
        country_stats[f'area_at_risk_{ssp}'] = country_stats['Mang_Ha_2020'] * country_stats[f'cri_mean_{ssp}']
        country_stats[f'pop_at_risk_{ssp}'] = country_stats['Ben_Pop_2020'] * country_stats[f'cri_mean_{ssp}']
        country_stats[f'stock_at_risk_{ssp}'] = country_stats['Ben_Stock_2020'] * country_stats[f'cri_mean_{ssp}']

print(f"\n  Country stats computed for {len(country_stats)} countries")

# Top 20 countries by CRI (SSP585)
top20 = country_stats.nlargest(20, 'cri_mean_ssp585')
print("\n  Top 20 countries by mean CRI (SSP5-8.5):")
for _, row in top20.iterrows():
    print(f"    {row['Country']}: CRI={row['cri_mean_ssp585']:.3f}")

# Save intermediate results
mangroves_country.drop(columns=['geometry']).to_csv(f'{OUT}/mangrove_risk_data.csv', index=False)
country_stats.to_csv(f'{OUT}/country_risk_stats.csv', index=False)
print(f"\n  Saved mangrove risk data and country stats")

# ============================================================
# 12. Summary statistics
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

summary = {}
for ssp in ssp_scenarios:
    cri = mangroves[f'cri_{ssp}']
    slr = mangroves[f'slr_rate_2100_{ssp}']
    
    summary[ssp_labels[ssp]] = {
        'CRI_mean': float(cri.mean()),
        'CRI_median': float(cri.median()),
        'CRI_std': float(cri.std()),
        'CRI_max': float(cri.max()),
        'pct_high_risk': float((cri > 0.6).mean() * 100),
        'pct_very_high_risk': float((cri > 0.8).mean() * 100),
        'SLR_rate_mean': float(slr.mean()),
        'SLR_above_4mm': float((slr >= 4).mean() * 100),
        'SLR_above_7mm': float((slr >= 7).mean() * 100),
        'TC_exposed_pct': float((mangroves['tcri'] > 0).mean() * 100),
    }
    
    print(f"\n{ssp_labels[ssp]}:")
    for k, v in summary[ssp_labels[ssp]].items():
        print(f"  {k}: {v:.2f}")

with open(f'{OUT}/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n\nPhase 1-4 complete. Proceeding to visualization...")
