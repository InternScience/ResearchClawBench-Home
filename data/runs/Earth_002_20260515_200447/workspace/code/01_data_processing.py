"""
Data Processing Script: Calculate Mangrove Risk Index
Processes mangrove locations, sea level rise rates, and tropical cyclone data
"""

import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import json
import os

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 60)
print("Phase 1: Loading and Exploring Data")
print("=" * 60)

# Load mangrove data
print("\nLoading mangrove data...")
mangrove_gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
print(f"  Mangrove points: {len(mangrove_gdf)}")
print(f"  CRS: {mangrove_gdf.crs}")
print(f"  Columns: {mangrove_gdf.columns.tolist()}")

# Extract centroids (already points)
mangrove_lats = mangrove_gdf.geometry.y.values
mangrove_lons = mangrove_gdf.geometry.x.values

print(f"\n  Latitude range: {mangrove_lats.min():.2f} to {mangrove_lats.max():.2f}")
print(f"  Longitude range: {mangrove_lons.min():.2f} to {mangrove_lons.max():.2f}")

# Load sea level rise data for all scenarios
print("\nLoading sea level rise data...")
slr_scenarios = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    fname = f'data/slr/total_{scenario}_medium_confidence_rates.nc'
    ds = xr.open_dataset(fname)
    slr_scenarios[scenario] = ds
    print(f"  {scenario}: {ds.sea_level_change_rate.shape}")

# Load tropical cyclone data
print("\nLoading tropical cyclone data...")
tc_ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
print(f"  TC track points: {len(tc_ds.record)}")
print(f"  Wind range: {float(tc_ds.wind.min()):.2f} to {float(tc_ds.wind.max()):.2f} m/s")

print("\n" + "=" * 60)
print("Phase 2: Processing Sea Level Rise Data")
print("=" * 60)

from scipy.spatial import cKDTree

def extract_slr_at_locations(ds, lats, lons, quantile=0.5):
    """Extract sea level rise rates at specific locations using nearest-neighbor matching."""
    # Build a KD-tree from SLR data locations
    slr_lats = ds.lat.values
    slr_lons = ds.lon.values
    slr_coords = np.column_stack([slr_lats, slr_lons])
    tree = cKDTree(slr_coords)
    
    # Query for nearest neighbors
    mangrove_coords = np.column_stack([lats, lons])
    distances, indices = tree.query(mangrove_coords, k=1)
    
    # Find median quantile index
    quantile_idx = np.argmin(np.abs(ds.quantiles.values - quantile))
    
    # Extract rates for 2080-2100 period (end of century)
    year_mask = (ds.years.values >= 2080) & (ds.years.values <= 2100)
    year_indices = np.where(year_mask)[0]
    
    # Extract rates for each location
    # sea_level_change_rate shape: (quantiles, years, locations)
    all_rates = ds.sea_level_change_rate.values[quantile_idx, :, :]  # (years, locations)
    
    # Average over 2080-2100 period for each location
    loc_rates = np.mean(all_rates[year_indices, :], axis=0)  # (locations,)
    
    # Assign to mangrove points using nearest-neighbor indices
    rates = loc_rates[indices]
    
    # Report distances
    max_dist = distances.max()
    mean_dist = distances.mean()
    print(f"    Nearest-neighbor distances: mean={mean_dist:.2f}°, max={max_dist:.2f}°")
    
    return rates

# Extract SLR rates for all scenarios
print("Extracting sea level rise rates at mangrove locations...")
slr_rates = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    print(f"  Processing {scenario}...")
    slr_rates[scenario] = extract_slr_at_locations(
        slr_scenarios[scenario], mangrove_lats, mangrove_lons, quantile=0.5
    )
    print(f"    Mean SLR rate: {slr_rates[scenario].mean():.2f} mm/yr")
    print(f"    Max SLR rate: {slr_rates[scenario].max():.2f} mm/yr")

# Calculate SLR risk component (0-1 scale)
# Based on Saintilan et al. (2023): deficit likely at 4 mm/yr, highly likely at 7 mm/yr
def calculate_slr_risk(slr_rates):
    """Calculate SLR risk score (0-1) based on thresholds from literature.
    
    Risk levels:
    - < 2 mm/yr: Low risk (0-0.2)
    - 2-4 mm/yr: Moderate risk (0.2-0.5)
    - 4-7 mm/yr: High risk (0.5-0.8)
    - > 7 mm/yr: Very high risk (0.8-1.0)
    """
    risk = np.zeros_like(slr_rates)
    
    # Low risk: < 2 mm/yr
    mask_low = slr_rates < 2
    risk[mask_low] = slr_rates[mask_low] / 2 * 0.2
    
    # Moderate risk: 2-4 mm/yr
    mask_mod = (slr_rates >= 2) & (slr_rates < 4)
    risk[mask_mod] = 0.2 + (slr_rates[mask_mod] - 2) / 2 * 0.3
    
    # High risk: 4-7 mm/yr
    mask_high = (slr_rates >= 4) & (slr_rates < 7)
    risk[mask_high] = 0.5 + (slr_rates[mask_high] - 4) / 3 * 0.3
    
    # Very high risk: >= 7 mm/yr
    mask_vhigh = slr_rates >= 7
    risk[mask_vhigh] = 0.8 + np.minimum((slr_rates[mask_vhigh] - 7) / 3, 1) * 0.2
    
    return risk

slr_risk = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    slr_risk[scenario] = calculate_slr_risk(slr_rates[scenario])
    print(f"  {scenario} SLR risk: mean={slr_risk[scenario].mean():.3f}, max={slr_risk[scenario].max():.3f}")

print("\n" + "=" * 60)
print("Phase 3: Processing Tropical Cyclone Data")
print("=" * 60)

# Create gridded TC statistics
# Use 5-degree grid for global coverage
lat_bins = np.arange(-40, 55, 5)
lon_bins = np.arange(-180, 185, 5)

print("Creating gridded TC statistics...")

# Calculate TC frequency per grid cell (normalized by number of years)
n_years = 165  # 1850-2014
tc_freq = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
tc_mean_wind = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
tc_max_wind = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

for i in range(len(lat_bins)-1):
    for j in range(len(lon_bins)-1):
        mask = (tc_ds.lat.values >= lat_bins[i]) & (tc_ds.lat.values < lat_bins[i+1]) & \
               (tc_ds.lon.values >= lon_bins[j]) & (tc_ds.lon.values < lon_bins[j+1])
        if np.sum(mask) > 0:
            tc_freq[i, j] = np.sum(mask) / n_years
            tc_mean_wind[i, j] = np.mean(tc_ds.wind.values[mask])
            tc_max_wind[i, j] = np.max(tc_ds.wind.values[mask])

print(f"  TC frequency range: {tc_freq.min():.2f} to {tc_freq.max():.2f} events/year")
print(f"  TC mean wind range: {tc_mean_wind.min():.2f} to {tc_mean_wind.max():.2f} m/s")

# Calculate TC risk component (0-1 scale)
def assign_tc_risk(lats, lons, tc_freq, tc_max_wind, lat_bins, lon_bins):
    """Assign TC risk score to mangrove locations based on gridded TC statistics.
    
    Risk factors:
    - TC frequency (events/year)
    - TC intensity (max wind speed)
    
    Risk levels:
    - Low: < 5 events/year AND max wind < 50 m/s
    - Moderate: 5-20 events/year OR max wind 50-70 m/s
    - High: 20-50 events/year OR max wind 70-85 m/s
    - Very high: > 50 events/year OR max wind > 85 m/s
    """
    risks = np.zeros(len(lats))
    
    for i in range(len(lats)):
        # Find grid cell
        lat_idx = np.argmin(np.abs(lat_bins - lats[i])) - 1
        lon_idx = np.argmin(np.abs(lon_bins - lons[i])) - 1
        
        # Clamp to valid range
        lat_idx = max(0, min(lat_idx, len(lat_bins)-2))
        lon_idx = max(0, min(lon_idx, len(lon_bins)-2))
        
        freq = tc_freq[lat_idx, lon_idx]
        max_w = tc_max_wind[lat_idx, lon_idx]
        
        # Calculate risk based on frequency and intensity
        freq_risk = 0
        if freq < 5:
            freq_risk = freq / 5 * 0.2
        elif freq < 20:
            freq_risk = 0.2 + (freq - 5) / 15 * 0.3
        elif freq < 50:
            freq_risk = 0.5 + (freq - 20) / 30 * 0.3
        else:
            freq_risk = 0.8 + min((freq - 50) / 50, 1) * 0.2
        
        # Intensity risk
        int_risk = 0
        if max_w < 50:
            int_risk = max_w / 50 * 0.2
        elif max_w < 70:
            int_risk = 0.2 + (max_w - 50) / 20 * 0.3
        elif max_w < 85:
            int_risk = 0.5 + (max_w - 70) / 15 * 0.3
        else:
            int_risk = 0.8 + min((max_w - 85) / 30, 1) * 0.2
        
        # Combine frequency and intensity (weighted average)
        risks[i] = 0.5 * freq_risk + 0.5 * int_risk
    
    return risks

print("Assigning TC risk to mangrove locations...")
tc_risk = assign_tc_risk(mangrove_lats, mangrove_lons, tc_freq, tc_max_wind, lat_bins, lon_bins)
print(f"  TC risk: mean={tc_risk.mean():.3f}, max={tc_risk.max():.3f}")

print("\n" + "=" * 60)
print("Phase 4: Calculating Composite Risk Index")
print("=" * 60)

# Calculate composite risk index for each SSP scenario
# Weighting: 50% SLR risk + 50% TC risk
composite_risk = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    composite_risk[scenario] = 0.5 * slr_risk[scenario] + 0.5 * tc_risk
    print(f"  {scenario} composite risk: mean={composite_risk[scenario].mean():.3f}, max={composite_risk[scenario].max():.3f}")

# Categorize risk levels
def categorize_risk(risk_scores):
    """Categorize risk scores into levels."""
    categories = np.zeros_like(risk_scores, dtype=int)
    categories[risk_scores < 0.2] = 1  # Low
    categories[(risk_scores >= 0.2) & (risk_scores < 0.5)] = 2  # Moderate
    categories[(risk_scores >= 0.5) & (risk_scores < 0.8)] = 3  # High
    categories[risk_scores >= 0.8] = 4  # Very High
    return categories

risk_categories = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    risk_categories[scenario] = categorize_risk(composite_risk[scenario])

# Calculate risk statistics by category
risk_stats = {}
for scenario in ['ssp245', 'ssp370', 'ssp585']:
    cats = risk_categories[scenario]
    n_total = len(cats)
    stats = {
        'low': np.sum(cats == 1) / n_total * 100,
        'moderate': np.sum(cats == 2) / n_total * 100,
        'high': np.sum(cats == 3) / n_total * 100,
        'very_high': np.sum(cats == 4) / n_total * 100
    }
    risk_stats[scenario] = stats
    print(f"\n  {scenario} risk distribution:")
    for level, pct in stats.items():
        print(f"    {level}: {pct:.1f}%")

print("\n" + "=" * 60)
print("Phase 5: Saving Results")
print("=" * 60)

# Create results DataFrame
results_df = pd.DataFrame({
    'latitude': mangrove_lats,
    'longitude': mangrove_lons,
    'ref_cls': mangrove_gdf['ref_cls'].values,
    'tc_risk': tc_risk,
    'slr_risk_ssp245': slr_risk['ssp245'],
    'slr_risk_ssp370': slr_risk['ssp370'],
    'slr_risk_ssp585': slr_risk['ssp585'],
    'composite_risk_ssp245': composite_risk['ssp245'],
    'composite_risk_ssp370': composite_risk['ssp370'],
    'composite_risk_ssp585': composite_risk['ssp585']
})

# Save to CSV
results_df.to_csv('outputs/mangrove_risk_results.csv', index=False)
print("Saved results to outputs/mangrove_risk_results.csv")

# Save summary statistics
summary = {
    'total_mangrove_points': len(mangrove_gdf),
    'risk_statistics': risk_stats,
    'slr_rates_mean': {scenario: float(slr_rates[scenario].mean()) for scenario in ['ssp245', 'ssp370', 'ssp585']},
    'tc_risk_mean': float(tc_risk.mean()),
    'composite_risk_mean': {scenario: float(composite_risk[scenario].mean()) for scenario in ['ssp245', 'ssp370', 'ssp585']}
}

with open('outputs/risk_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved summary to outputs/risk_summary.json")

# Save gridded TC data for mapping
np.save('outputs/tc_freq_grid.npy', tc_freq)
np.save('outputs/tc_max_wind_grid.npy', tc_max_wind)
np.save('outputs/lat_bins.npy', lat_bins)
np.save('outputs/lon_bins.npy', lon_bins)

print("\nData processing complete!")
print(f"Results saved to outputs/")
print(f"Figures will be generated in report/images/")