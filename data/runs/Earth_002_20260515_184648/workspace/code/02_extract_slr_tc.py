#!/usr/bin/env python3
"""
Extract median SLR rates (2020-2100) for each SSP scenario
and compute baseline TC frequency per mangrove centroid.
"""
import os
import geopandas as gpd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load mangrove points
print("Loading mangrove points...")
gdf = gpd.read_file("data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg")
gdf = gdf[gdf["ref_cls"] == 1].copy()  # keep only mangroves
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y
print(f"  {len(gdf)} mangrove points loaded")

# 2. Load SLR datasets and extract median rate
print("Processing SLR datasets...")
slr_files = {
    "ssp245": "data/slr/total_ssp245_medium_confidence_rates.nc",
    "ssp370": "data/slr/total_ssp370_medium_confidence_rates.nc",
    "ssp585": "data/slr/total_ssp585_medium_confidence_rates.nc",
}

for scenario, path in slr_files.items():
    ds = xr.open_dataset(path)
    # sea_level_change_rate has dims (locations, quantiles, years)
    # quantile index for median (50th percentile) is usually index 53 or 54
    # Use the middle quantile (index 53 for 107 quantiles)
    median_idx = 53
    # Average rate across years 2020-2100 (years 1 to 13, year 0 is 2020)
    rate = ds["sea_level_change_rate"].isel(quantiles=median_idx).mean(dim="years")
    # Attach to nearest mangrove point
    tree = cKDTree(np.vstack([ds.lon.values, ds.lat.values]).T)
    dist, idx = tree.query(np.vstack([gdf.lon, gdf.lat]).T, k=1)
    gdf[f"slr_rate_{scenario}"] = rate.values[idx] * 1000  # mm/yr
    print(f"  {scenario} median SLR rate extracted")
    ds.close()

# 3. Load TC tracks and compute frequency
print("Processing TC tracks...")
tc = xr.open_dataset("data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc")
tc_lon = tc.lon.values
tc_lat = tc.lat.values

# Build KDTree for mangrove points
mangrove_tree = cKDTree(np.vstack([gdf.lon, gdf.lat]).T)

# Count TC points within 100 km of each mangrove
print("  Computing TC frequency (100 km buffer)...")
counts = np.zeros(len(gdf))
for i in range(0, len(tc_lon), 50000):  # batch processing
    batch_lon = tc_lon[i : i + 50000]
    batch_lat = tc_lat[i : i + 50000]
    dist, idx = mangrove_tree.query(np.vstack([batch_lon, batch_lat]).T, k=1, distance_upper_bound=1.0)
    valid = dist < 1.0  # < 1 degree ~100 km
    np.add.at(counts, idx[valid], 1)

# Historical period ~165 years (1850-2014)
years_hist = 165
gdf["tc_freq"] = counts / years_hist  # annual frequency

# 4. Save intermediate results
gdf.to_file(os.path.join(OUTPUT_DIR, "mangrove_risk_points.gpkg"), driver="GPKG")
print(f"Saved {len(gdf)} points to {OUTPUT_DIR}/mangrove_risk_points.gpkg")

# Quick summary
print("\nSummary statistics:")
print(gdf[["slr_rate_ssp245", "slr_rate_ssp370", "slr_rate_ssp585", "tc_freq"]].describe())