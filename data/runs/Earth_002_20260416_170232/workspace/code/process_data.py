#!/usr/bin/env python3
"""
Process input data for mangrove composite risk index analysis.
Extracts SLR rates, processes TC tracks, and prepares mangrove locations.
"""

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_slr_data():
    """Extract median SLR rates from IPCC AR6 data for each SSP scenario."""
    ssp_scenarios = {
        "ssp245": "total_ssp245_medium_confidence_rates.nc",
        "ssp370": "total_ssp370_medium_confidence_rates.nc",
        "ssp585": "total_ssp585_medium_confidence_rates.nc"
    }
    
    slr_data = {}
    
    for scenario, filename in ssp_scenarios.items():
        filepath = DATA_DIR / "slr" / filename
        ds = xr.open_dataset(filepath)
        
        # Get median (0.5 quantile) SLR rates for 2020-2100
        # Find closest quantile to 0.5
        quantiles = ds.quantiles.values
        median_idx = np.argmin(np.abs(quantiles - 0.5))
        
        # Extract median rates across years and locations
        median_rates = ds.sea_level_change_rate.isel(quantiles=median_idx)
        
        # Calculate mean rate over 2020-2100 period for each location
        mean_rates = median_rates.mean(dim="years")
        
        slr_data[scenario] = {
            "locations": ds.locations.values,
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "mean_slr_rate": mean_rates.values,
            "years": ds.years.values,
            "full_rates": median_rates.values
        }
        
        print(f"{scenario}: {len(slr_data[scenario]['locations'])} locations, "
              f"SLR range: {slr_data[scenario]['mean_slr_rate'].min():.2f} - "
              f"{slr_data[scenario]['mean_slr_rate'].max():.2f} mm/yr")
    
    # Save processed SLR data
    for scenario, data in slr_data.items():
        df = pd.DataFrame({
            "location_id": data["locations"],
            "lat": data["lat"],
            "lon": data["lon"],
            "slr_rate_mm_yr": data["mean_slr_rate"]
        })
        df.to_csv(OUTPUT_DIR / f"slr_{scenario}.csv", index=False)
    
    return slr_data


def process_tc_tracks():
    """Process historical TC tracks to calculate baseline frequencies."""
    filepath = DATA_DIR / "tc" / "tracks_mit_mpi-esm1-2-hr_historical_reduced.nc"
    ds = xr.open_dataset(filepath)
    
    # Extract track data
    lats = ds.lat.values
    lons = ds.lon.values
    winds = ds.wind.values
    
    # Create DataFrame of TC track points
    tc_df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "wind_ms": winds
    })
    
    # Categorize by Saffir-Simpson scale
    # Category 1: 33-42 m/s, Cat 2: 43-49 m/s, Cat 3: 50-58 m/s
    # Cat 4: 59-69 m/s, Cat 5: >=70 m/s
    def get_category(wind):
        if wind < 33:
            return 0
        elif wind < 43:
            return 1
        elif wind < 50:
            return 2
        elif wind < 58:
            return 3
        elif wind < 69:
            return 4
        else:
            return 5
    
    tc_df["category"] = tc_df["wind_ms"].apply(get_category)
    
    # Create gridded frequency maps (1 degree resolution)
    tc_df["lat_grid"] = np.round(tc_df["lat"]).astype(int)
    tc_df["lon_grid"] = np.round(tc_df["lon"]).astype(int)
    
    # Calculate frequency by grid cell and category
    freq_all = tc_df.groupby(["lat_grid", "lon_grid"]).size().reset_index(name="freq_total")
    freq_by_cat = tc_df.groupby(["lat_grid", "lon_grid", "category"]).size().unstack(fill_value=0)
    freq_by_cat.columns = [f"cat_{c}" for c in freq_by_cat.columns]
    freq_by_cat = freq_by_cat.reset_index()
    
    # Merge
    freq_grid = freq_all.merge(freq_by_cat, on=["lat_grid", "lon_grid"], how="left")
    
    # Save TC frequency grid
    freq_grid.to_csv(OUTPUT_DIR / "tc_frequency_grid.csv", index=False)
    
    # Also save raw track points for reference
    tc_df.to_csv(OUTPUT_DIR / "tc_track_points.csv", index=False)
    
    print(f"TC tracks: {len(tc_df)} track points")
    print(f"Frequency grid: {len(freq_grid)} grid cells with TC activity")
    print(f"Category distribution:\n{tc_df['category'].value_counts().sort_index()}")
    
    return tc_df, freq_grid


def process_mangroves():
    """Process mangrove extent data to extract centroid points."""
    filepath = DATA_DIR / "mangroves" / "gmw_v4_ref_smpls_qad_v12.gpkg"
    gdf = gpd.read_file(filepath)
    
    print(f"Mangrove samples: {len(gdf)} points")
    print(f"Columns: {list(gdf.columns)}")
    
    # Extract coordinates from geometry
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    
    # Create gridded representation (1 degree)
    gdf["lat_grid"] = np.round(gdf["lat"]).astype(int)
    gdf["lon_grid"] = np.round(gdf["lon"]).astype(int)
    
    # Count mangrove presence per grid cell
    mangrove_grid = gdf.groupby(["lat_grid", "lon_grid"]).agg(
        mangrove_count=("uid", "count"),
        lat_mean=("lat", "mean"),
        lon_mean=("lon", "mean")
    ).reset_index()
    
    # Save mangrove data
    gdf.to_csv(OUTPUT_DIR / "mangrove_points.csv", index=False)
    mangrove_grid.to_csv(OUTPUT_DIR / "mangrove_grid.csv", index=False)
    
    print(f"Mangrove grid cells: {len(mangrove_grid)}")
    
    return gdf, mangrove_grid


if __name__ == "__main__":
    print("=" * 60)
    print("Processing SLR data...")
    print("=" * 60)
    slr_data = process_slr_data()
    
    print("\n" + "=" * 60)
    print("Processing TC tracks...")
    print("=" * 60)
    tc_df, freq_grid = process_tc_tracks()
    
    print("\n" + "=" * 60)
    print("Processing mangrove data...")
    print("=" * 60)
    mangrove_gdf, mangrove_grid = process_mangroves()
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print("=" * 60)
    print(f"Outputs saved to: {OUTPUT_DIR.absolute()}")
