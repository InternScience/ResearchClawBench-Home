#!/usr/bin/env python3
"""
Composite Risk Index for Global Mangrove Ecosystems:
Combining Tropical Cyclone Regime Shifts and Sea Level Rise

Revised analysis with improved SLR risk scoring and comprehensive visualization.
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
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_002_20260417_110902'
DATA = f'{BASE}/data'
OUT = f'{BASE}/outputs'
IMG = f'{BASE}/report/images'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# ============================================================
# 1. Load mangrove data
# ============================================================
print("\n--- Loading mangrove data ---")
mangroves = gpd.read_file(f'{DATA}/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
mangroves['mang_lon'] = mangroves.geometry.x
mangroves['mang_lat'] = mangroves.geometry.y
N = len(mangroves)
print(f"Loaded {N} mangrove points")

# ============================================================
# 2. TC RISK COMPONENT
# ============================================================
print("\n--- Loading TC track data ---")
tc_ds = xr.open_dataset(f'{DATA}/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lat = tc_ds['lat'].values
tc_lon = tc_ds['lon'].values
tc_wind = tc_ds['wind'].values
tc_ds.close()

# Saffir-Simpson categories
def classify_ss(wind_ms):
    cats = np.zeros(len(wind_ms), dtype=int)
    cats[(wind_ms >= 33) & (wind_ms < 42.5)] = 1
    cats[(wind_ms >= 42.5) & (wind_ms < 49.5)] = 2
    cats[(wind_ms >= 49.5) & (wind_ms < 58)] = 3
    cats[(wind_ms >= 58) & (wind_ms < 70)] = 4
    cats[wind_ms >= 70] = 5
    return cats

tc_cats = classify_ss(tc_wind)

# Grid at 1° resolution
lon_bins = np.arange(-180, 181, 1)
lat_bins = np.arange(-90, 91, 1)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
hist_years = 165

tc_freq = {}
for cat in range(1, 6):
    mask = tc_cats == cat
    h, _, _ = np.histogram2d(tc_lat[mask], tc_lon[mask], bins=[lat_bins, lon_bins])
    tc_freq[cat] = h / hist_years

# TCDI ratios from Mo et al. (2023): Cat 3:4:5 = 1:13:29
tcdi = {1: 0.0, 2: 0.0, 3: 1.0/29.0, 4: 13.0/29.0, 5: 1.0}

# TCRI grid
tcri_grid = np.zeros((len(lat_centers), len(lon_centers)))
for cat in range(1, 6):
    tcri_grid += tcdi[cat] * tc_freq[cat]

# Assign to mangroves
def grid_lookup(lats, lons, grid, lat_c, lon_c):
    li = np.clip(np.searchsorted(lat_c, lats) - 1, 0, len(lat_c)-1)
    lo = np.clip(np.searchsorted(lon_c, lons) - 1, 0, len(lon_c)-1)
    return grid[li, lo]

mangroves['tcri'] = grid_lookup(mangroves['mang_lat'].values, mangroves['mang_lon'].values,
                                 tcri_grid, lat_centers, lon_centers)
for cat in range(1, 6):
    mangroves[f'tc_freq_cat{cat}'] = grid_lookup(mangroves['mang_lat'].values, mangroves['mang_lon'].values,
                                                   tc_freq[cat], lat_centers, lon_centers)

tc_freq_total_grid = sum(tc_freq.values())
tc_freq_major_grid = tc_freq[3] + tc_freq[4] + tc_freq[5]
mangroves['tc_freq_total'] = grid_lookup(mangroves['mang_lat'].values, mangroves['mang_lon'].values,
                                          tc_freq_total_grid, lat_centers, lon_centers)
mangroves['tc_freq_major'] = grid_lookup(mangroves['mang_lat'].values, mangroves['mang_lon'].values,
                                          tc_freq_major_grid, lat_centers, lon_centers)

print(f"TC exposed mangroves (TCRI>0): {(mangroves['tcri']>0).sum()} ({100*(mangroves['tcri']>0).mean():.1f}%)")

# ============================================================
# 3. SLR RISK COMPONENT
# ============================================================
print("\n--- Processing SLR data ---")
ssp_scenarios = ['ssp245', 'ssp370', 'ssp585']
ssp_labels = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}

# Build KD-tree from SLR locations (same for all SSPs)
ds_ref = xr.open_dataset(f'{DATA}/slr/total_ssp245_medium_confidence_rates.nc')
slr_lat = ds_ref['lat'].values
slr_lon = ds_ref['lon'].values
ds_ref.close()

tree = cKDTree(np.column_stack([slr_lat, slr_lon]))
dists, indices = tree.query(np.column_stack([mangroves['mang_lat'].values, mangroves['mang_lon'].values]), k=1)

for ssp in ssp_scenarios:
    ds = xr.open_dataset(f'{DATA}/slr/total_{ssp}_medium_confidence_rates.nc')
    q50 = np.argmin(np.abs(ds['quantiles'].values - 0.5))
    q05 = np.argmin(np.abs(ds['quantiles'].values - 0.05))
    q95 = np.argmin(np.abs(ds['quantiles'].values - 0.95))
    years = ds['years'].values
    
    # Rate at 2100
    yr_2100 = np.where(years == 2100)[0][0]
    r2100 = ds['sea_level_change_rate'].isel(quantiles=q50, years=yr_2100).values[indices]
    r2100_q05 = ds['sea_level_change_rate'].isel(quantiles=q05, years=yr_2100).values[indices]
    r2100_q95 = ds['sea_level_change_rate'].isel(quantiles=q95, years=yr_2100).values[indices]
    
    # Average rate 2020-2100 (more representative of cumulative exposure)
    rates_period = []
    for yi in range(len(years)):
        if 2020 <= years[yi] <= 2100:
            rates_period.append(ds['sea_level_change_rate'].isel(quantiles=q50, years=yi).values[indices])
    avg_rate = np.mean(rates_period, axis=0)
    
    mangroves[f'slr_rate_2100_{ssp}'] = r2100
    mangroves[f'slr_rate_2100_q05_{ssp}'] = r2100_q05
    mangroves[f'slr_rate_2100_q95_{ssp}'] = r2100_q95
    mangroves[f'slr_avg_rate_{ssp}'] = avg_rate
    
    ds.close()
    print(f"  {ssp_labels[ssp]}: avg rate mean={avg_rate.mean():.2f}, 2100 rate mean={r2100.mean():.2f}")

# SLR risk score: continuous, based on rate at 2100
# Use quantile-based normalization within the mangrove dataset for better differentiation
# Also compute threshold-based categories
def slr_continuous_score(rate, ref_max=20.0):
    """Continuous SLR risk score using the rate relative to key thresholds.
    0 at 0, ~0.3 at 4 mm/yr, ~0.6 at 7 mm/yr, 1.0 at ref_max mm/yr."""
    score = np.clip(rate / ref_max, 0, 1)
    return score

for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    # Use the 2100 rate for risk scoring with a reference max
    ref_max = np.percentile(rate[rate > 0], 95)
    mangroves[f'slr_score_{ssp}'] = np.clip(rate / ref_max, 0, 1)
    
    # Threshold categories
    mangroves[f'slr_cat_{ssp}'] = 'Low (<4)'
    mangroves.loc[rate >= 4, f'slr_cat_{ssp}'] = 'Moderate (4-7)'
    mangroves.loc[rate >= 7, f'slr_cat_{ssp}'] = 'High (7-10)'
    mangroves.loc[rate >= 10, f'slr_cat_{ssp}'] = 'Very High (>10)'
    
    print(f"  {ssp_labels[ssp]} SLR score: mean={mangroves[f'slr_score_{ssp}'].mean():.3f}, "
          f"ref_max={ref_max:.1f}")

# ============================================================
# 4. COMPOSITE RISK INDEX
# ============================================================
print("\n--- Computing Composite Risk Index ---")

# Normalize TCRI
tcri_vals = mangroves['tcri'].values
tcri_p99 = np.percentile(tcri_vals[tcri_vals > 0], 99) if (tcri_vals > 0).sum() > 0 else 1
mangroves['tc_score'] = np.clip(tcri_vals / tcri_p99, 0, 1)

# CRI = 0.5 * TC + 0.5 * SLR (equal weighting)
for ssp in ssp_scenarios:
    mangroves[f'cri_{ssp}'] = 0.5 * mangroves['tc_score'] + 0.5 * mangroves[f'slr_score_{ssp}']
    
    # Also multiplicative version: captures compound risk
    mangroves[f'cri_mult_{ssp}'] = mangroves['tc_score'] * mangroves[f'slr_score_{ssp}']
    
    cri = mangroves[f'cri_{ssp}']
    print(f"  {ssp_labels[ssp]} CRI: mean={cri.mean():.3f}, median={cri.median():.3f}, "
          f"p95={cri.quantile(0.95):.3f}, max={cri.max():.3f}")

# Risk categories
def cri_category(cri):
    cats = pd.Series('Very Low', index=cri.index)
    cats[cri > 0.2] = 'Low'
    cats[cri > 0.4] = 'Moderate'
    cats[cri > 0.6] = 'High'
    cats[cri > 0.8] = 'Very High'
    return cats

for ssp in ssp_scenarios:
    mangroves[f'cri_cat_{ssp}'] = cri_category(mangroves[f'cri_{ssp}'])

# ============================================================
# 5. COUNTRY AGGREGATION
# ============================================================
print("\n--- Country-level aggregation ---")
countries = gpd.read_file(f'{DATA}/ecosystem/UCSC_CWON_countrybounds.gpkg')

# Spatial join
mang_gdf = mangroves.set_geometry('geometry')
mang_country = gpd.sjoin(mang_gdf, countries[['Country', 'ISO3', 'geometry']], 
                          how='left', predicate='within')

# Fill unmatched with nearest country
unmatched = mang_country['Country'].isna()
if unmatched.sum() > 0:
    um_coords = np.column_stack([mang_country.loc[unmatched, 'mang_lat'].values,
                                  mang_country.loc[unmatched, 'mang_lon'].values])
    cc = countries.geometry.centroid
    ctree = cKDTree(np.column_stack([cc.y, cc.x]))
    _, nidx = ctree.query(um_coords, k=1)
    mang_country.loc[unmatched, 'Country'] = countries.iloc[nidx]['Country'].values
    mang_country.loc[unmatched, 'ISO3'] = countries.iloc[nidx]['ISO3'].values

print(f"  Matched {(~unmatched).sum()}, nearest-assigned {unmatched.sum()}")

# Build country stats
agg_cols = {}
for ssp in ssp_scenarios:
    agg_cols[f'cri_{ssp}'] = ['mean', 'median', 'max', 'std']
    agg_cols[f'slr_score_{ssp}'] = 'mean'
    agg_cols[f'slr_rate_2100_{ssp}'] = ['mean', 'std']
agg_cols['tc_score'] = 'mean'
agg_cols['tcri'] = 'mean'
agg_cols['tc_freq_major'] = 'mean'
agg_cols['uid'] = 'count'

cstats = mang_country.groupby('Country').agg(agg_cols).reset_index()
cstats.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in cstats.columns]
# Flatten multi-level column names
flat_cols = []
for c in cstats.columns:
    if isinstance(c, tuple):
        flat_cols.append('_'.join([str(x) for x in c if x != '']))
    else:
        flat_cols.append(str(c))
cstats.columns = flat_cols

# Merge ecosystem services
eco_cols = ['Country', 'ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020',
            'Ben_Pop_2020', 'Ben_Stock_2020']
cstats = cstats.merge(countries[eco_cols], on='Country', how='left')

# Ecosystem services at risk
for ssp in ssp_scenarios:
    cri_col = [c for c in cstats.columns if f'cri_{ssp}' in c and 'mean' in c][0]
    cstats[f'area_risk_{ssp}'] = cstats['Mang_Ha_2020'].fillna(0) * cstats[cri_col].fillna(0)
    cstats[f'pop_risk_{ssp}'] = cstats['Ben_Pop_2020'].fillna(0) * cstats[cri_col].fillna(0)
    cstats[f'stock_risk_{ssp}'] = cstats['Ben_Stock_2020'].fillna(0) * cstats[cri_col].fillna(0)

# Save
mang_country_save = mang_country.drop(columns=['geometry'])
mang_country_save.to_csv(f'{OUT}/mangrove_risk_data.csv', index=False)
cstats.to_csv(f'{OUT}/country_risk_stats.csv', index=False)

print(f"  {len(cstats)} countries processed")

# ============================================================
# 6. REGIONAL ANALYSIS
# ============================================================
print("\n--- Regional analysis ---")

# Define ocean basin regions
def assign_region(lat, lon):
    """Assign mangrove point to ocean basin region."""
    if lon >= -100 and lon <= -60 and lat >= 7:
        return 'Gulf of Mexico / Caribbean'
    elif lon >= -100 and lon <= -30 and lat < 7 and lat >= -35:
        return 'South America (Atlantic)'
    elif lon >= -20 and lon <= 55 and lat >= -40 and lat <= 35:
        return 'West Africa / Indian Ocean (West)'
    elif lon >= 55 and lon <= 105 and lat >= -15 and lat <= 30:
        return 'South/Southeast Asia'
    elif lon >= 105 and lon <= 180 and lat >= 0:
        return 'Northwest Pacific'
    elif (lon >= 105 or lon <= -150) and lat < 0:
        return 'Oceania / Southwest Pacific'
    elif lon >= -180 and lon <= -100 and lat >= 0:
        return 'Eastern Pacific'
    else:
        return 'Other'

mang_country['region'] = [assign_region(lat, lon) for lat, lon in 
                           zip(mang_country['mang_lat'], mang_country['mang_lon'])]

region_stats = {}
for ssp in ssp_scenarios:
    rs = mang_country.groupby('region').agg({
        f'cri_{ssp}': ['mean', 'std', 'count'],
        'tc_score': 'mean',
        f'slr_score_{ssp}': 'mean',
        f'slr_rate_2100_{ssp}': 'mean',
        'tc_freq_major': 'mean'
    }).reset_index()
    rs.columns = ['region', 'cri_mean', 'cri_std', 'n_points', 'tc_mean', 'slr_mean', 'slr_rate', 'tc_freq']
    rs['ssp'] = ssp
    region_stats[ssp] = rs
    print(f"\n  {ssp_labels[ssp]}:")
    for _, row in rs.iterrows():
        print(f"    {row['region']}: CRI={row['cri_mean']:.3f} (TC={row['tc_mean']:.3f}, SLR={row['slr_mean']:.3f}), n={int(row['n_points'])}")

region_df = pd.concat(region_stats.values())
region_df.to_csv(f'{OUT}/regional_risk_stats.csv', index=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

summary = {}
for ssp in ssp_scenarios:
    cri = mangroves[f'cri_{ssp}']
    slr = mangroves[f'slr_rate_2100_{ssp}']
    
    summary[ssp_labels[ssp]] = {
        'CRI_mean': round(float(cri.mean()), 4),
        'CRI_median': round(float(cri.median()), 4),
        'CRI_std': round(float(cri.std()), 4),
        'CRI_p95': round(float(cri.quantile(0.95)), 4),
        'CRI_max': round(float(cri.max()), 4),
        'pct_moderate_or_higher': round(float((cri > 0.4).mean() * 100), 2),
        'pct_high_or_higher': round(float((cri > 0.6).mean() * 100), 2),
        'pct_very_high': round(float((cri > 0.8).mean() * 100), 2),
        'SLR_rate_2100_mean': round(float(slr.mean()), 2),
        'SLR_above_4mm': round(float((slr >= 4).mean() * 100), 2),
        'SLR_above_7mm': round(float((slr >= 7).mean() * 100), 2),
        'SLR_above_10mm': round(float((slr >= 10).mean() * 100), 2),
        'TC_exposed_pct': round(float((mangroves['tcri'] > 0).mean() * 100), 2),
        'TC_high_risk_pct': round(float((mangroves['tc_score'] > 0.5).mean() * 100), 2),
    }
    
    print(f"\n{ssp_labels[ssp]}:")
    for k, v in summary[ssp_labels[ssp]].items():
        print(f"  {k}: {v}")

with open(f'{OUT}/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save mangroves data for plotting
mangroves.to_pickle(f'{OUT}/mangroves_processed.pkl')
mang_country.drop(columns=['geometry']).to_pickle(f'{OUT}/mang_country_processed.pkl')

print("\n\nAnalysis complete. Data saved.")
