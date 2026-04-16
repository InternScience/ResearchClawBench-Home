"""
Step 1: Process mangrove points and extract SLR rates for each point.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import json
import os

os.makedirs('outputs', exist_ok=True)

# Load mangrove points
print("Loading mangrove points...")
gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
mangrove_lats = gdf.geometry.y.values
mangrove_lons = gdf.geometry.x.values
print(f"  {len(mangrove_lats)} mangrove points loaded")

# Load SLR data for each scenario
scenarios = {
    'SSP2-4.5': 'data/slr/total_ssp245_medium_confidence_rates.nc',
    'SSP3-7.0': 'data/slr/total_ssp370_medium_confidence_rates.nc',
    'SSP5-8.5': 'data/slr/total_ssp585_medium_confidence_rates.nc',
}

# Get median quantile index
q50_idx = 53  # 0.5 quantile

# Year mask for 2020-2100
yr_indices = list(range(9))  # indices 0-8 for years 2020-2100

slr_results = {}

for scenario_name, filepath in scenarios.items():
    print(f"\nProcessing {scenario_name}...")
    ds = xr.open_dataset(filepath)
    
    slr_lats = ds['lat'].values
    slr_lons = ds['lon'].values
    
    # Build KD-tree for nearest neighbor lookup
    # Use cos(lat) weighting for geographic distance
    slr_coords = np.column_stack([
        np.cos(np.radians(slr_lats)) * np.cos(np.radians(slr_lons)),
        np.cos(np.radians(slr_lats)) * np.sin(np.radians(slr_lons)),
        np.sin(np.radians(slr_lats))
    ])
    
    mangrove_coords = np.column_stack([
        np.cos(np.radians(mangrove_lats)) * np.cos(np.radians(mangrove_lons)),
        np.cos(np.radians(mangrove_lats)) * np.sin(np.radians(mangrove_lons)),
        np.sin(np.radians(mangrove_lats))
    ])
    
    tree = cKDTree(slr_coords)
    _, indices = tree.query(mangrove_coords)
    
    # Extract median SLR rate for each mangrove point
    # Average over 2020-2100 period
    rates = ds['sea_level_change_rate'].values[q50_idx, yr_indices, :]  # (9, 66190)
    
    # Average rate over 2020-2100 for each SLR location
    avg_rates = np.mean(rates, axis=0)  # (66190,)
    
    # Map to mangrove points
    mangrove_slr_rates = avg_rates[indices]
    
    slr_results[scenario_name] = mangrove_slr_rates
    
    print(f"  SLR rates (mm/yr) - min: {mangrove_slr_rates.min():.2f}, "
          f"mean: {mangrove_slr_rates.mean():.2f}, "
          f"max: {mangrove_slr_rates.max():.2f}")
    print(f"  Points with rate >= 4 mm/yr: {(mangrove_slr_rates >= 4).sum()} ({(mangrove_slr_rates >= 4).mean()*100:.1f}%)")
    print(f"  Points with rate >= 7 mm/yr: {(mangrove_slr_rates >= 7).sum()} ({(mangrove_slr_rates >= 7).mean()*100:.1f}%)")
    
    ds.close()

# Save results
slr_df = pd.DataFrame({
    'lat': mangrove_lats,
    'lon': mangrove_lons,
})
for scenario_name, rates in slr_results.items():
    slr_df[f'slr_rate_{scenario_name}'] = rates

slr_df.to_csv('outputs/mangrove_slr_rates.csv', index=False)
print("\nSaved SLR rates to outputs/mangrove_slr_rates.csv")

# Summary statistics
summary = {}
for scenario_name, rates in slr_results.items():
    summary[scenario_name] = {
        'mean_rate_mm_yr': float(np.mean(rates)),
        'median_rate_mm_yr': float(np.median(rates)),
        'pct_above_4': float(np.mean(rates >= 4) * 100),
        'pct_above_7': float(np.mean(rates >= 7) * 100),
        'pct_above_10': float(np.mean(rates >= 10) * 100),
    }

with open('outputs/slr_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved SLR summary to outputs/slr_summary.json")
