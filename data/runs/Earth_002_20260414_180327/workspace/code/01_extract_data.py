#!/usr/bin/env python3
"""
Step 1: Extract mangrove points and SLR rates, compute TC frequency.
Save intermediate results for downstream analysis.
"""
import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import json
import os

os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. Load mangrove points
# ============================================================
print("Loading mangrove points...")
gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
gdf = gdf.to_crs('EPSG:4326')
lons = gdf.geometry.x.values
lats = gdf.geometry.y.values
n_pts = len(lats)
print(f"  {n_pts} mangrove points loaded")

# ============================================================
# 2. Extract SLR rates at mangrove locations
# ============================================================
ssp_files = {
    'ssp245': 'data/slr/total_ssp245_medium_confidence_rates.nc',
    'ssp370': 'data/slr/total_ssp370_medium_confidence_rates.nc',
    'ssp585': 'data/slr/total_ssp585_medium_confidence_rates.nc',
}

# We want median (quantile=0.5) rate for years 2080-2100
# The 'years' coord has 2020,2030,...,2150. 2080 is index 6, 2090 is 7, 2100 is 8.
# We'll take the mean over 2080-2100 (indices 6,7,8) at median quantile.
target_quantile = 0.5
target_year_indices = [6, 7, 8]  # 2080, 2090, 2100

slr_results = {}

for ssp_name, fpath in ssp_files.items():
    print(f"\nProcessing SLR: {ssp_name}...")
    ds = xr.open_dataset(fpath)
    slat = ds['lat'].values
    slon = ds['lon'].values
    
    # Build KDTree for nearest neighbor lookup
    # Handle longitude wrapping: use (lon+180)%360 - 180
    slon_wrapped = ((slon + 180) % 360) - 180
    tree = cKDTree(np.column_stack([slat, slon_wrapped]))
    
    # Query nearest SLR location for each mangrove point
    dists, idxs = tree.query(np.column_stack([lats, lons]), k=1)
    
    # Find quantile index closest to 0.5
    qvals = ds['quantiles'].values
    q_idx = np.argmin(np.abs(qvals - target_quantile))
    print(f"  Using quantile index {q_idx} (value={qvals[q_idx]})")
    
    # Extract rates: shape (quantiles, years, locations)
    # We want median rate averaged over 2080-2100
    rates_all = ds['sea_level_change_rate'].values  # (107, 14, 66190)
    
    # For each mangrove point, get the rate at its nearest SLR location
    rates_at_pts = rates_all[q_idx, :, :]  # (14, 66190)
    
    # Average over 2080-2100
    rates_2080_2100 = rates_at_pts[target_year_indices, :][:, idxs]  # (3, n_pts)
    median_rate = np.mean(rates_2080_2100, axis=0)  # (n_pts,)
    
    slr_results[ssp_name] = {
        'rate_mm_yr': median_rate,
        'nearest_dist_deg': dists,
    }
    print(f"  SLR rate stats: min={np.nanmin(median_rate):.2f}, median={np.nanmedian(median_rate):.2f}, max={np.nanmax(median_rate):.2f} mm/yr")
    
    ds.close()

# ============================================================
# 3. Compute TC frequency at mangrove locations
# ============================================================
print("\nProcessing TC tracks...")
tc_ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lat = tc_ds['lat'].values
tc_lon = tc_ds['lon'].values
tc_wind = tc_ds['wind'].values
tc_ds.close()

# Filter valid points (not NaN)
valid = np.isfinite(tc_lat) & np.isfinite(tc_lon) & np.isfinite(tc_wind)
tc_lat = tc_lat[valid]
tc_lon = tc_lon[valid]
tc_wind = tc_wind[valid]
print(f"  {len(tc_lat)} valid TC track points")

# For each mangrove point, count TC points within a radius (e.g., 2 degrees ~ 200km)
# and compute mean wind speed of nearby TC points
# This is a simplified proxy for TC exposure
from scipy.spatial import cKDTree

# Build tree for TC points
tc_tree = cKDTree(np.column_stack([tc_lat, tc_lon]))

# Query with radius of 2 degrees
radius_deg = 2.0
tc_freq = np.zeros(n_pts, dtype=np.float64)
tc_mean_wind = np.zeros(n_pts, dtype=np.float64)
tc_max_wind = np.zeros(n_pts, dtype=np.float64)

# Process in batches to avoid memory issues
batch_size = 5000
for i in range(0, n_pts, batch_size):
    end = min(i + batch_size, n_pts)
    batch_pts = np.column_stack([lats[i:end], lons[i:end]])
    neighbors_list = tc_tree.query_ball_point(batch_pts, r=radius_deg)
    
    for j, neighbors in enumerate(neighbors_list):
        if len(neighbors) > 0:
            tc_freq[i+j] = len(neighbors)
            tc_mean_wind[i+j] = np.mean(tc_wind[neighbors])
            tc_max_wind[i+j] = np.max(tc_wind[neighbors])
    
    if (i // batch_size) % 10 == 0:
        print(f"  TC batch {i//batch_size}/{(n_pts-1)//batch_size}")

# Historical period: 1850-2014 = 165 years
# Normalize to annual frequency
n_years = 165
tc_annual_freq = tc_freq / n_years

print(f"  TC annual frequency stats: min={np.min(tc_annual_freq):.4f}, median={np.median(tc_annual_freq):.4f}, max={np.max(tc_annual_freq):.4f}")

# ============================================================
# 4. Save results
# ============================================================
print("\nSaving results...")
results = {
    'lon': lons.tolist(),
    'lat': lats.tolist(),
    'uid': gdf['uid'].values.tolist(),
    'slr_ssp245_rate': slr_results['ssp245']['rate_mm_yr'].tolist(),
    'slr_ssp370_rate': slr_results['ssp370']['rate_mm_yr'].tolist(),
    'slr_ssp585_rate': slr_results['ssp585']['rate_mm_yr'].tolist(),
    'tc_annual_freq': tc_annual_freq.tolist(),
    'tc_mean_wind': tc_mean_wind.tolist(),
    'tc_max_wind': tc_max_wind.tolist(),
}

# Save as numpy arrays for efficiency
np.savez_compressed('outputs/mangrove_risk_data.npz',
    lon=lons, lat=lats,
    slr_ssp245=slr_results['ssp245']['rate_mm_yr'],
    slr_ssp370=slr_results['ssp370']['rate_mm_yr'],
    slr_ssp585=slr_results['ssp585']['rate_mm_yr'],
    tc_annual_freq=tc_annual_freq,
    tc_mean_wind=tc_mean_wind,
    tc_max_wind=tc_max_wind,
)

print("Done! Saved to outputs/mangrove_risk_data.npz")
print(f"\nSummary:")
print(f"  Total mangrove points: {n_pts}")
print(f"  Points with TC exposure: {np.sum(tc_annual_freq > 0)}")
print(f"  Points with SLR > 4 mm/yr (SSP585): {np.sum(slr_results['ssp585']['rate_mm_yr'] > 4)}")
print(f"  Points with SLR > 7 mm/yr (SSP585): {np.sum(slr_results['ssp585']['rate_mm_yr'] > 7)}")
