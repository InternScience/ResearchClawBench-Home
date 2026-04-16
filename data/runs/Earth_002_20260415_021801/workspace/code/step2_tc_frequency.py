"""
Step 2: Calculate TC frequency and intensity from historical tracks,
then assign TC risk to each mangrove point.
"""
import numpy as np
import pandas as pd
import xarray as xr
import json
import os

os.makedirs('outputs', exist_ok=True)

# Load TC tracks
print("Loading TC tracks...")
ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lats = ds['lat'].values
tc_lons = ds['lon'].values
tc_winds = ds['wind'].values
ds.close()
print(f"  {len(tc_lats)} TC track points loaded")

# Historical period: 1850-2014 = 165 years
n_years = 165

# Define grid: 2°×2°
lat_bins = np.arange(-40, 52, 2)
lon_bins = np.arange(-180, 182, 2)

# Calculate TC frequency per grid cell for different intensity categories
# Categories based on Saffir-Simpson wind speeds (m/s):
# Cat 1: 33-42, Cat 2: 43-49, Cat 3: 50-58, Cat 4: 59-70, Cat 5: >=70
# "Major TC": wind >= 50 m/s (Cat 3+)
# "Intense TC": wind >= 59 m/s (Cat 4+)

categories = {
    'all_tc': (tc_winds >= 33),
    'major_tc': (tc_winds >= 50),
    'intense_tc': (tc_winds >= 59),
    'cat5_tc': (tc_winds >= 70),
}

tc_grid_data = {}

for cat_name, mask in categories.items():
    lats_f = tc_lats[mask]
    lons_f = tc_lons[mask]
    winds_f = tc_winds[mask]
    
    # 2D histogram
    freq_grid, _, _ = np.histogram2d(lats_f, lons_f, bins=[lat_bins, lon_bins])
    # Annual frequency
    freq_annual = freq_grid / n_years
    
    # Max wind per grid cell
    wind_grid = np.zeros_like(freq_grid)
    lat_idx = np.digitize(lats_f, lat_bins) - 1
    lon_idx = np.digitize(lons_f, lon_bins) - 1
    valid = (lat_idx >= 0) & (lat_idx < len(lat_bins)-1) & (lon_idx >= 0) & (lon_idx < len(lon_bins)-1)
    for i in range(len(lats_f)):
        if valid[i]:
            wind_grid[lat_idx[i], lon_idx[i]] = max(wind_grid[lat_idx[i], lon_idx[i]], winds_f[i])
    
    tc_grid_data[cat_name] = {
        'freq_annual': freq_annual,
        'max_wind': wind_grid,
    }
    
    print(f"  {cat_name}: {mask.sum()} points, "
          f"max freq: {freq_annual.max():.3f}/yr, "
          f"max wind: {wind_grid.max():.1f} m/s")

# Now assign TC metrics to each mangrove point
print("\nAssigning TC metrics to mangrove points...")
slr_df = pd.read_csv('outputs/mangrove_slr_rates.csv')
mangrove_lats = slr_df['lat'].values
mangrove_lons = slr_df['lon'].values

# Find grid cell for each mangrove point
lat_idx = np.digitize(mangrove_lats, lat_bins) - 1
lon_idx = np.digitize(mangrove_lons, lon_bins) - 1
lat_idx = np.clip(lat_idx, 0, len(lat_bins)-2)
lon_idx = np.clip(lon_idx, 0, len(lon_bins)-2)

# Extract TC metrics for each mangrove point
for cat_name in categories:
    freq = tc_grid_data[cat_name]['freq_annual']
    wind = tc_grid_data[cat_name]['max_wind']
    
    slr_df[f'tc_freq_{cat_name}'] = freq[lat_idx, lon_idx]
    slr_df[f'tc_maxwind_{cat_name}'] = wind[lat_idx, lon_idx]

# Save
slr_df.to_csv('outputs/mangrove_tc_metrics.csv', index=False)
print("Saved TC metrics to outputs/mangrove_tc_metrics.csv")

# Summary
tc_summary = {
    'pct_mangroves_exposed_major_tc': float(np.mean(slr_df['tc_freq_major_tc'].values > 0) * 100),
    'pct_mangroves_exposed_intense_tc': float(np.mean(slr_df['tc_freq_intense_tc'].values > 0) * 100),
    'mean_major_tc_freq': float(np.mean(slr_df['tc_freq_major_tc'].values)),
    'max_major_tc_freq': float(np.max(slr_df['tc_freq_major_tc'].values)),
    'mean_max_wind': float(np.mean(slr_df['tc_maxwind_all_tc'].values)),
    'max_max_wind': float(np.max(slr_df['tc_maxwind_all_tc'].values)),
}
with open('outputs/tc_summary.json', 'w') as f:
    json.dump(tc_summary, f, indent=2)
print("Saved TC summary to outputs/tc_summary.json")

# Print key stats
print(f"\nMangroves exposed to major TCs (Cat 3+): {tc_summary['pct_mangroves_exposed_major_tc']:.1f}%")
print(f"Mangroves exposed to intense TCs (Cat 4+): {tc_summary['pct_mangroves_exposed_intense_tc']:.1f}%")
print(f"Mean major TC frequency: {tc_summary['mean_major_tc_freq']:.4f}/yr")
print(f"Mean max wind at mangrove points: {tc_summary['mean_max_wind']:.1f} m/s")
