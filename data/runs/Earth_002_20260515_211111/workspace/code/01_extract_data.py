#!/usr/bin/env python3
"""
Extract SLR rates and TC frequencies at global mangrove locations.
Saves intermediate results to outputs/.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

print("=== Step 1: Load mangrove data ===")
gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
print(f"Loaded {len(gdf)} mangrove points")
print(f"Columns: {list(gdf.columns)}")

# Extract coordinates
mangrove_lats = gdf.geometry.y.values
mangrove_lons = gdf.geometry.x.values

# Normalize longitudes to [-180, 180] for consistency with SLR data
mangrove_lons_norm = np.where(mangrove_lons > 180, mangrove_lons - 360, mangrove_lons)

print(f"\nLat range: {mangrove_lats.min():.2f} to {mangrove_lats.max():.2f}")
print(f"Lon range: {mangrove_lons_norm.min():.2f} to {mangrove_lons_norm.max():.2f}")

# === Step 2: Extract SLR data at mangrove locations ===
print("\n=== Step 2: Extract SLR data ===")

# Build KDTree for nearest-neighbor lookup from SLR grid
for ssp_name, ssp_file in [
    ('SSP2-4.5', 'data/slr/total_ssp245_medium_confidence_rates.nc'),
    ('SSP3-7.0', 'data/slr/total_ssp370_medium_confidence_rates.nc'),
    ('SSP5-8.5', 'data/slr/total_ssp585_medium_confidence_rates.nc'),
]:
    print(f"\nProcessing {ssp_name}...")
    ds = xr.open_dataset(ssp_file)
    
    slr_lats = ds.lat.values
    slr_lons = ds.lon.values
    
    # Build KD tree for SLR locations
    slr_coords = np.column_stack([slr_lats, slr_lons])
    tree = cKDTree(slr_coords)
    
    # Find nearest SLR grid point for each mangrove
    mangrove_coords = np.column_stack([mangrove_lats, mangrove_lons_norm])
    distances, indices = tree.query(mangrove_coords)
    
    print(f"  Mean distance to nearest SLR grid point: {distances.mean():.3f} degrees")
    
    # Extract median SLR rates (quantile 0.5, index 53)
    median_idx = 53
    slr_all = ds.sea_level_change_rate.isel(quantiles=median_idx).values  # (years, locations)
    years = ds.years.values
    
    # For each mangrove, get the SLR time series
    slr_at_mangroves = slr_all[:, indices]  # (years, n_mangroves)
    
    # Compute mean SLR for 2080-2100 period
    late_period_mask = (years >= 2080) & (years <= 2100)
    slr_2080_2100 = slr_at_mangroves[late_period_mask].mean(axis=0)
    
    # Also compute 2020-2100 trend
    early_period_mask = (years >= 2020) & (years <= 2100)
    slr_2020_2100_mean = slr_at_mangroves[early_period_mask].mean(axis=0)
    
    # Store results
    col_prefix = ssp_name.lower().replace('.', '_').replace('-', '_')
    gdf[f'{col_prefix}_slr_2080_2100'] = slr_2080_2100
    gdf[f'{col_prefix}_slr_2020_2100'] = slr_2020_2100_mean
    
    ds.close()

print("\nSLR extraction complete.")
print(gdf[['ssp2_4_5_slr_2080_2100', 'ssp3_7_0_slr_2080_2100', 'ssp5_8_5_slr_2080_2100']].describe())

# === Step 3: Extract TC frequency at mangrove locations ===
print("\n=== Step 3: Extract TC frequency data ===")

ds_tc = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lats = ds_tc.lat.values
tc_lons = ds_tc.lon.values
tc_winds = ds_tc.wind.values

print(f"TC records: {len(tc_lats)}")
print(f"Wind speed range: {tc_winds.min():.1f} - {tc_winds.max():.1f} m/s")

# Define TC categories based on Saffir-Simpson (m/s)
# Cat 1: 33-43 m/s (119-153 km/h)
# Cat 2: 43-50 m/s (154-177 km/h)  
# Cat 3: 50-58 m/s (178-208 km/h)
# Cat 4: 58-70 m/s (209-251 km/h)
# Cat 5: >= 70 m/s (>= 252 km/h)
cat_thresholds = {
    'cat1': (33.0, 43.0),
    'cat2': (43.0, 50.0),
    'cat3': (50.0, 58.0),
    'cat4': (58.0, 70.0),
    'cat5': (70.0, 200.0),
}

# Count TC passes within a radius of each mangrove
# Use ~100 km radius (roughly 1 degree)
radius_deg = 1.0
num_years = 165  # 1850-2014

# Build KD tree for TC points
tc_coords = np.column_stack([tc_lats, tc_lons])
tc_tree = cKDTree(tc_coords)
mangrove_coords = np.column_stack([mangrove_lats, mangrove_lons_norm])

# For each mangrove, count TC passes by category
print("Counting TC passes per mangrove location...")

# Batch query for efficiency
batch_size = 10000
n_mangroves = len(mangrove_coords)

# Initialize counts
for cat_name in cat_thresholds:
    gdf[f'tc_{cat_name}_count'] = 0

# Process in batches
for start in range(0, n_mangroves, batch_size):
    end = min(start + batch_size, n_mangroves)
    batch_coords = mangrove_coords[start:end]
    
    # Find all TC points within radius of each mangrove in batch
    indices_list = tc_tree.query_ball_point(batch_coords, r=radius_deg)
    
    for i, idx_list in enumerate(indices_list):
        if len(idx_list) > 0:
            winds = tc_winds[idx_list]
            for cat_name, (wmin, wmax) in cat_thresholds.items():
                gdf.loc[start + i, f'tc_{cat_name}_count'] = np.sum((winds >= wmin) & (winds < wmax))
    
    if (start // batch_size) % 10 == 0:
        print(f"  Processed {end}/{n_mangroves} mangroves...")

# Compute annual frequencies
for cat_name in cat_thresholds:
    gdf[f'tc_{cat_name}_freq'] = gdf[f'tc_{cat_name}_count'] / num_years

# Compute total annual TC frequency (all categories)
gdf['tc_total_freq'] = sum(gdf[f'tc_{cat_name}_freq'] for cat_name in cat_thresholds)
gdf['tc_major_freq'] = sum(gdf[f'tc_{cat_name}_freq'] for cat_name in ['cat3', 'cat4', 'cat5'])
gdf['tc_intense_freq'] = sum(gdf[f'tc_{cat_name}_freq'] for cat_name in ['cat4', 'cat5'])

ds_tc.close()

print("\nTC frequency extraction complete.")
print(gdf[['tc_total_freq', 'tc_major_freq', 'tc_intense_freq']].describe())

# === Step 4: Save intermediate data ===
print("\n=== Step 4: Save intermediate data ===")
gdf.to_file('outputs/mangrove_risk_data.gpkg', driver='GPKG')
print("Saved to outputs/mangrove_risk_data.gpkg")

# Also save as parquet for faster loading
gdf.to_parquet('outputs/mangrove_risk_data.parquet')
print("Saved to outputs/mangrove_risk_data.parquet")

# Save summary statistics
summary = gdf.describe()
summary.to_csv('outputs/data_summary.csv')
print("Saved summary to outputs/data_summary.csv")

print("\n=== Extraction complete! ===")
