#!/usr/bin/env python3
"""
GlaMBIE Data Loading and Processing Script

This script loads all CSV files from the GlaMBIE input dataset,
standardizes the data structure, and prepares for reconciliation.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
WORKSPACE_ROOT = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Earth_000_20260416_164538")
DATA_DIR = WORKSPACE_ROOT / "data" / "glambie" / "input"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

# Region mapping (folder name -> canonical name)
REGION_MAPPING = {
    "1_alaska": "Alaska",
    "2_western_canada_us": "Western Canada US",
    "3_arctic_canada_north": "Arctic Canada North",
    "4_arctic_canada_south": "Arctic Canada South",
    "5_greenland_periphery": "Greenland Periphery",
    "6_iceland": "Iceland",
    "7_svalbard": "Svalbard",
    "8_scandinavia": "Scandinavia",
    "9_russian_arctic": "Russian Arctic",
    "10_north_asia": "North Asia",
    "11_central_europe": "Central Europe",
    "12_caucasus_middle_east": "Caucasus Middle East",
    "13_central_asia": "Central Asia",
    "14_south_asia_west": "South Asia West",
    "15_south_asia_east": "South Asia East",
    "16_low_latitudes": "Low Latitudes",
    "17_southern_andes": "Southern Andes",
    "18_new_zealand": "New Zealand",
    "19_antarctic_and_subantarctic": "Antarctic and Subantarctic"
}

# Method type mapping (from filename patterns)
METHOD_MAPPING = {
    "glaciological": "Glaciological",
    "demdiff": "DEM Differencing",
    "altimetry": "Altimetry",
    "gravimetry": "Gravimetry",
    "combined": "Combined"
}


def extract_method_from_filename(filename):
    """Extract method type from filename."""
    fname_lower = filename.lower()
    for key, method in METHOD_MAPPING.items():
        if key in fname_lower:
            return method
    return "Unknown"


def extract_region_from_path(filepath):
    """Extract region name from file path."""
    parts = filepath.parts
    for i, part in enumerate(parts):
        if part.startswith(tuple(f"{j}_" for j in range(1, 20))):
            return REGION_MAPPING.get(part, part)
    return "Unknown"


def load_csv_file(filepath):
    """Load a single CSV file and add metadata."""
    try:
        df = pd.read_csv(filepath)
        df["source_file"] = filepath.name
        df["region"] = extract_region_from_path(filepath)
        df["method"] = extract_method_from_filename(filepath.name)
        df["author"] = df.get("author", "Unknown")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_all_data():
    """Load all CSV files from the input directory."""
    all_files = list(DATA_DIR.rglob("*.csv"))
    print(f"Found {len(all_files)} CSV files")
    
    dataframes = []
    for filepath in all_files:
        df = load_csv_file(filepath)
        if df is not None:
            dataframes.append(df)
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} rows, {len(all_files)} files")
        return combined_df
    else:
        return None


def standardize_units(df):
    """
    Standardize units to both m w.e. and Gt.
    
    Note: Conversion between m w.e. and Gt requires glacier area.
    For now, we keep the original units and flag them.
    1 Gt = 1 km³ of water = 1e9 m³
    1 m w.e. over 1 km² = 1e6 m³ = 0.001 Gt
    """
    # Check what units we have
    unique_units = df["unit"].unique()
    print(f"Unique units in dataset: {unique_units}")
    
    # Create standardized columns
    df["unit_original"] = df["unit"]
    
    # For datasets already in mwe or m, keep as is
    # For datasets in Gt, we need area to convert to mwe
    # For now, just normalize the unit naming
    df["unit_standardized"] = df["unit"].map({
        "mwe": "m w.e.",
        "m": "m w.e.",
        "Gt": "Gt",
        "m3": "m³"
    }).fillna(df["unit"])
    
    return df


def compute_annual_time_series(df, target_years=range(2000, 2024)):
    """
    Convert period-based measurements to annual time series.
    
    The data contains start_dates and end_dates in fractional years.
    We need to aggregate to calendar year resolution.
    """
    # Extract year from dates
    df["start_year"] = df["start_dates"].astype(int)
    df["end_year"] = df["end_dates"].astype(int)
    
    # For simplicity, assign measurement to midpoint year
    df["midpoint_year"] = ((df["start_dates"] + df["end_dates"]) / 2).round().astype(int)
    
    # Filter to target years
    annual_data = df[df["midpoint_year"].isin(target_years)].copy()
    
    return annual_data


def main():
    print("=" * 60)
    print("GlaMBIE Data Loading and Processing")
    print("=" * 60)
    
    # Load all data
    print("\n1. Loading all CSV files...")
    df = load_all_data()
    
    if df is None:
        print("Failed to load data!")
        return
    
    # Show basic statistics
    print(f"\n2. Dataset overview:")
    print(f"   Total records: {len(df)}")
    print(f"   Regions: {df['region'].nunique()}")
    print(f"   Methods: {df['method'].value_counts().to_dict()}")
    print(f"   Authors/sources: {df['author'].nunique()}")
    
    # Standardize units
    print(f"\n3. Standardizing units...")
    df = standardize_units(df)
    
    # Compute annual time series
    print(f"\n4. Computing annual time series (2000-2023)...")
    annual_df = compute_annual_time_series(df)
    print(f"   Records in target period: {len(annual_df)}")
    
    # Save processed data
    print(f"\n5. Saving processed data...")
    
    # Save full dataset
    full_output = OUTPUTS_DIR / "all_data.parquet"
    df.to_parquet(full_output, index=False)
    print(f"   Saved: {full_output}")
    
    # Save annual dataset
    annual_output = OUTPUTS_DIR / "annual_data.parquet"
    annual_df.to_parquet(annual_output, index=False)
    print(f"   Saved: {annual_output}")
    
    # Save summary statistics
    summary = {
        "total_records": len(df),
        "records_in_period": len(annual_df),
        "regions": df["region"].nunique(),
        "methods": df["method"].value_counts().to_dict(),
        "authors": df["author"].nunique(),
        "temporal_range": {
            "min_start": float(df["start_dates"].min()),
            "max_end": float(df["end_dates"].max())
        }
    }
    
    import json
    summary_output = OUTPUTS_DIR / "data_summary.json"
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {summary_output}")
    
    # Print regional breakdown
    print(f"\n6. Regional breakdown:")
    regional_counts = df.groupby("region")["method"].value_counts().unstack(fill_value=0)
    print(regional_counts.to_string())
    
    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)
    
    return df, annual_df


if __name__ == "__main__":
    main()
