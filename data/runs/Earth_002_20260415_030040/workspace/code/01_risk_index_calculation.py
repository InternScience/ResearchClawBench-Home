"""
Main analysis script for Mangrove Composite Risk Index.
Combines tropical cyclone regime shifts and sea level rise to evaluate 
global mangrove ecosystem risk by 2100 under different SSP scenarios.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('code', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 60)
print("Step 1: Load and process Sea Level Rise data")
print("=" * 60)

# Load SLR data for all 3 SSPs
ssp_files = {
    'SSP2-4.5': 'data/slr/total_ssp245_medium_confidence_rates.nc',
    'SSP3-7.0': 'data/slr/total_ssp370_medium_confidence_rates.nc',
    'SSP5-8.5': 'data/slr/total_ssp585_medium_confidence_rates.nc'
}

slr_data = {}
for ssp_name, filepath in ssp_files.items():
    print(f"\nLoading {ssp_name}...")
    ds = xr.open_dataset(filepath)
    
    # Get median (0.5 quantile) rate at year 2100
    rates_2100 = ds['sea_level_change_rate'].sel(quantiles=0.5, years=2100).values
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    slr_data[ssp_name] = {
        'rates': rates_2100,  # mm/year
        'lats': lats,
        'lons': lons
    }
    
    print(f"  Locations: {len(lats)}")
    print(f"  Rate range: {rates_2100.min():.1f} to {rates_2100.max():.1f} mm/yr")
    print(f"  Mean rate: {rates_2100.mean():.1f} mm/yr")

# Create KD-tree for fast nearest neighbor lookup
slr_coords = np.column_stack([slr_data['SSP2-4.5']['lons'], slr_data['SSP2-4.5']['lats']])
slr_tree = cKDTree(slr_coords)

print("\nSLR KD-tree built successfully")

print("\n" + "=" * 60)
print("Step 2: Load and process Tropical Cyclone data")
print("=" * 60)

# Load historical TC tracks
tc_ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lats = tc_ds['lat'].values
tc_lons = tc_ds['lon'].values
tc_winds = tc_ds['wind'].values

print(f"Total TC track points: {len(tc_lats)}")
print(f"Wind speed range: {tc_winds.min():.1f} to {tc_winds.max():.1f} m/s")
print(f"Mean wind speed: {tc_winds.mean():.1f} m/s")

# Create spatial grid for TC frequency calculation (1-degree resolution)
lat_bins = np.arange(-60, 61, 1)
lon_bins = np.arange(-180, 181, 1)

# Calculate TC frequency per grid cell
tc_freq, _, _ = np.histogram2d(tc_lons, tc_lats, bins=[lon_bins, lat_bins])

# Calculate mean wind speed per grid cell
tc_wind_sum, _, _ = np.histogram2d(tc_lons, tc_lats, bins=[lon_bins, lat_bins], weights=tc_winds)
tc_count = np.maximum(tc_freq, 1)
tc_mean_wind = tc_wind_sum / tc_count

# Historical data covers 1850-2014 (165 years)
years_covered = 165
tc_annual_freq = tc_freq / years_covered

print(f"\nTC frequency grid shape: {tc_freq.shape}")
print(f"Max annual frequency in a cell: {tc_annual_freq.max():.2f}")
print(f"Non-zero cells: {np.sum(tc_annual_freq > 0)}")

# Save TC grid info
with open('outputs/tc_grid_info.json', 'w') as f:
    json.dump({
        'lat_bins': lat_bins.tolist(),
        'lon_bins': lon_bins.tolist(),
        'max_annual_freq': float(tc_annual_freq.max()),
        'non_zero_cells': int(np.sum(tc_annual_freq > 0)),
        'years_covered': years_covered
    }, f)

print("TC grid saved to outputs/tc_grid_info.json")

print("\n" + "=" * 60)
print("Step 3: Load mangrove data and assign SLR/TC values")
print("=" * 60)

# Load mangrove points
mangroves = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
print(f"Mangrove points loaded: {len(mangroves)}")
print(f"CRS: {mangroves.crs}")

# Extract coordinates
mangrove_lons = mangroves.geometry.x.values
mangrove_lats = mangroves.geometry.y.values

# Find nearest SLR location for each mangrove point
mangrove_coords = np.column_stack([mangrove_lons, mangrove_lats])
distances, indices = slr_tree.query(mangrove_coords, k=1)

print(f"Max distance to nearest SLR point: {distances.max():.4f} degrees")
print(f"Mean distance to nearest SLR point: {distances.mean():.4f} degrees")

# Assign SLR rates to each mangrove point using consistent column names
col_map = {'SSP2-4.5': 'slr_SSP245', 'SSP3-7.0': 'slr_SSP370', 'SSP5-8.5': 'slr_SSP585'}
for ssp_name, col_name in col_map.items():
    mangroves[col_name] = slr_data[ssp_name]['rates'][indices]

# Assign TC frequency and wind speed to each mangrove point
lon_indices = np.digitize(mangrove_lons, lon_bins) - 1
lat_indices = np.digitize(mangrove_lats, lat_bins) - 1

# Clip to valid range
lon_indices = np.clip(lon_indices, 0, tc_annual_freq.shape[0] - 1)
lat_indices = np.clip(lat_indices, 0, tc_annual_freq.shape[1] - 1)

mangroves['tc_annual_freq'] = tc_annual_freq[lon_indices, lat_indices]
mangroves['tc_mean_wind'] = tc_mean_wind[lon_indices, lat_indices]

print(f"\nMangrove SLR stats:")
for col in ['slr_SSP245', 'slr_SSP370', 'slr_SSP585']:
    print(f"  {col}: mean={mangroves[col].mean():.2f}, max={mangroves[col].max():.2f} mm/yr")

print(f"\nMangrove TC stats:")
print(f"  Annual frequency: mean={mangroves['tc_annual_freq'].mean():.4f}, max={mangroves['tc_annual_freq'].max():.2f}")
print(f"  Mean wind speed: mean={mangroves['tc_mean_wind'].mean():.2f}, max={mangroves['tc_mean_wind'].max():.2f} m/s")

# Save processed mangrove data
mangroves_out = mangroves[['uid', 'geometry', 'slr_SSP245', 'slr_SSP370', 'slr_SSP585', 
                            'tc_annual_freq', 'tc_mean_wind']].copy()
mangroves_out.to_parquet('outputs/mangrove_risk_data.parquet')
print("\nProcessed mangrove data saved to outputs/mangrove_risk_data.parquet")

print("\n" + "=" * 60)
print("Step 4: Calculate Composite Risk Index")
print("=" * 60)

def calc_slr_risk(slr_rate):
    """Calculate SLR risk score (0-1) based on mm/yr rate."""
    midpoint = 5.5  # mm/yr
    steepness = 0.8
    risk = 1 / (1 + np.exp(-steepness * (slr_rate - midpoint)))
    return np.clip(risk, 0, 1)

def calc_tc_risk(freq, mean_wind):
    """Calculate TC risk score (0-1) based on frequency and wind speed."""
    freq_norm = np.clip(freq / 0.3, 0, 1)
    wind_threshold = 17.5
    wind_norm = np.clip((mean_wind - wind_threshold) / (70 - wind_threshold), 0, 1)
    risk = freq_norm * (0.5 + 0.5 * wind_norm)
    return np.clip(risk, 0, 1)

# Calculate risk components for each SSP
ssps = ['SSP245', 'SSP370', 'SSP585']
ssp_labels = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

risk_results = {}

for ssp_col, ssp_label in zip(ssps, ssp_labels):
    slr_col = f'slr_{ssp_col}'
    
    slr_risk = calc_slr_risk(mangroves[slr_col].values)
    tc_risk = calc_tc_risk(mangroves['tc_annual_freq'].values, mangroves['tc_mean_wind'].values)
    
    w_slr = 0.5
    w_tc = 0.5
    composite_risk = w_slr * slr_risk + w_tc * tc_risk
    
    risk_results[ssp_label] = {
        'slr_risk': slr_risk,
        'tc_risk': tc_risk,
        'composite_risk': composite_risk,
        'slr_rate': mangroves[slr_col].values
    }
    
    print(f"\n{ssp_label}:")
    print(f"  SLR risk: mean={slr_risk.mean():.3f}, median={np.median(slr_risk):.3f}")
    print(f"  TC risk: mean={tc_risk.mean():.3f}, median={np.median(tc_risk):.3f}")
    print(f"  Composite risk: mean={composite_risk.mean():.3f}, median={np.median(composite_risk):.3f}")
    
    mangroves[f'slr_risk_{ssp_col}'] = slr_risk
    mangroves[f'tc_risk'] = tc_risk
    mangroves[f'composite_risk_{ssp_col}'] = composite_risk

# Save full results
mangroves.to_parquet('outputs/mangrove_full_risk.parquet')
print("\nFull risk data saved to outputs/mangrove_full_risk.parquet")

# Save summary statistics
summary_stats = {}
for ssp_label in ssp_labels:
    ssp_col = ssp_label.replace('-', '').replace('.', '')
    summary_stats[ssp_label] = {
        'slr_rate_mean': float(mangroves[f'slr_{ssp_col}'].mean()),
        'slr_rate_median': float(mangroves[f'slr_{ssp_col}'].median()),
        'slr_rate_max': float(mangroves[f'slr_{ssp_col}'].max()),
        'slr_risk_mean': float(mangroves[f'slr_risk_{ssp_col}'].mean()),
        'tc_risk_mean': float(mangroves['tc_risk'].mean()),
        'composite_risk_mean': float(mangroves[f'composite_risk_{ssp_col}'].mean()),
        'composite_risk_median': float(mangroves[f'composite_risk_{ssp_col}'].median()),
        'high_risk_fraction': float((mangroves[f'composite_risk_{ssp_col}'] > 0.7).sum() / len(mangroves)),
        'moderate_risk_fraction': float(((mangroves[f'composite_risk_{ssp_col}'] > 0.4) & (mangroves[f'composite_risk_{ssp_col}'] <= 0.7)).sum() / len(mangroves)),
        'low_risk_fraction': float((mangroves[f'composite_risk_{ssp_col}'] <= 0.4).sum() / len(mangroves))
    }

with open('outputs/risk_summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nSummary statistics saved to outputs/risk_summary_stats.json")
print(json.dumps(summary_stats, indent=2))

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
