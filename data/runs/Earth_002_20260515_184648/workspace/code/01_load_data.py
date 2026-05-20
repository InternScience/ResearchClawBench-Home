#!/usr/bin/env python3
"""
Load and inspect all input datasets for mangrove risk index.
"""
import geopandas as gpd
import xarray as xr
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def main():
    # Mangroves
    gdf = gpd.read_file(DATA_DIR / "mangroves/gmw_v4_ref_smpls_qad_v12.gpkg")
    print("Mangrove GPKG:")
    print(f"  CRS: {gdf.crs}")
    print(f"  Shape: {gdf.shape}")
    print(f"  Columns: {list(gdf.columns)}")
    print(f"  ref_cls value counts:\n{gdf['ref_cls'].value_counts().head()}")

    # SLR files
    for ssp in ["245", "370", "585"]:
        nc = DATA_DIR / f"slr/total_ssp{ssp}_medium_confidence_rates.nc"
        ds = xr.open_dataset(nc)
        print(f"\nSLR SSP{ssp}:")
        print(f"  Dims: {dict(ds.sizes)}")
        print(f"  Vars: {list(ds.data_vars.keys())}")
        print(f"  sea_level_change_rate shape: {ds['sea_level_change_rate'].shape}")

    # TC tracks
    tc = xr.open_dataset(DATA_DIR / "tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc")
    print("\nTC Tracks:")
    print(f"  Dims: {dict(tc.sizes)}")
    print(f"  Vars: {list(tc.data_vars.keys())}")
    print(f"  Sample lat range: {tc.lat.min().item():.1f} to {tc.lat.max().item():.1f}")

if __name__ == "__main__":
    main()
