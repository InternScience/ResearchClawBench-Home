#!/usr/bin/env python3
"""
GlaMBIE Data Loading and Exploration
Load all input and result data, categorize by method, produce initial statistics.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from pathlib import Path

# Paths
DATA_DIR = Path("data/glambie")
INPUT_DIR = DATA_DIR / "input"
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_DIR = Path("outputs")

# Region name mapping
REGION_NAMES = {
    "1_alaska": "Alaska",
    "2_western_canada_us": "Western Canada & US",
    "3_arctic_canada_north": "Arctic Canada North",
    "4_arctic_canada_south": "Arctic Canada South",
    "5_greenland_periphery": "Greenland Periphery",
    "6_iceland": "Iceland",
    "7_svalbard": "Svalbard",
    "8_scandinavia": "Scandinavia",
    "9_russian_arctic": "Russian Arctic",
    "10_north_asia": "North Asia",
    "11_central_europe": "Central Europe",
    "12_caucasus_middle_east": "Caucasus & Middle East",
    "13_central_asia": "Central Asia",
    "14_south_asia_west": "South Asia West",
    "15_south_asia_east": "South Asia East",
    "16_low_latitudes": "Low Latitudes",
    "17_southern_andes": "Southern Andes",
    "18_new_zealand": "New Zealand",
    "19_antarctic_and_subantarctic": "Antarctic & Subantarctic",
}

METHOD_MAP = {
    "glaciological": "Glaciological",
    "demdiff": "DEM Differencing",
    "altimetry": "Altimetry",
    "gravimetry": "Gravimetry",
    "combined": "Combined/Hybrid",
}

def extract_method(filename):
    """Extract method type from filename."""
    name = os.path.basename(filename).lower()
    for key in ["glaciological", "demdiff", "altimetry", "gravimetry", "combined"]:
        if key in name:
            return key
    return "unknown"

def extract_region(filename):
    """Extract region key from file path."""
    parts = filename.split("/")
    for p in parts:
        if any(p.startswith(str(i)) for i in range(1, 20)):
            return p
    return "unknown"

def load_input_data():
    """Load all input CSV files and compile metadata."""
    all_data = []
    csv_files = sorted(glob.glob(str(INPUT_DIR / "**/*.csv"), recursive=True))

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            region_key = extract_region(f)
            method = extract_method(f)
            basename = os.path.basename(f)

            df["region_key"] = region_key
            df["region_name"] = REGION_NAMES.get(region_key, region_key)
            df["method"] = method
            df["method_name"] = METHOD_MAP.get(method, method)
            df["source_file"] = basename
            df["author_short"] = df["author"].values[0] if "author" in df.columns else "unknown"

            all_data.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    return combined, csv_files

def load_results():
    """Load calendar year results for all regions and global."""
    results = {}
    cal_dir = RESULTS_DIR / "calendar_years"

    for f in sorted(cal_dir.glob("*.csv")):
        region_key = f.stem
        df = pd.read_csv(f)
        if region_key == "0_global":
            df["region_name"] = "Global"
            results["global"] = df
        else:
            df["region_name"] = REGION_NAMES.get(region_key, region_key)
            results[region_key] = df

    return results

def main():
    print("Loading GlaMBIE input data...")
    input_df, csv_files = load_input_data()

    print(f"\nTotal input files: {len(csv_files)}")
    print(f"Total data rows: {len(input_df)}")
    print(f"Unique regions: {input_df['region_key'].nunique()}")
    print(f"Unique authors: {input_df['author'].nunique()}")

    # Summary by method
    print("\n--- Method summary ---")
    for method in sorted(METHOD_MAP.keys()):
        subset = input_df[input_df["method"] == method]
        if len(subset) > 0:
            n_files = subset["source_file"].nunique()
            n_regions = subset["region_key"].nunique()
            print(f"  {METHOD_MAP[method]}: {n_files} files, {n_regions} regions, {len(subset)} rows")

    # Summary by region
    print("\n--- Region summary ---")
    for rk in sorted(REGION_NAMES.keys()):
        subset = input_df[input_df["region_key"] == rk]
        if len(subset) > 0:
            methods = subset["method"].unique()
            n_files = subset["source_file"].nunique()
            print(f"  {REGION_NAMES[rk]}: {n_files} files, methods: {', '.join(methods)}")

    # Save compiled data
    input_df.to_csv(OUTPUT_DIR / "compiled_input_data.csv", index=False)

    # Load results
    print("\nLoading GlaMBIE results...")
    results = load_results()
    for key, df in results.items():
        print(f"  {df['region_name'].iloc[0]}: {len(df)} years, {df['start_dates'].min():.0f}-{df['end_dates'].max():.0f}")

    # Save global and regional results
    for key, df in results.items():
        df.to_csv(OUTPUT_DIR / f"results_{key}.csv", index=False)

    # Compute summary statistics
    global_df = results.get("global")
    if global_df is not None:
        print("\n--- Global Summary (2000-2023) ---")
        total_mass = global_df["combined_gt"].sum()
        total_mass_err = np.sqrt((global_df["combined_gt_errors"]**2).sum())
        mean_rate = global_df["combined_gt"].mean()
        mean_rate_err = global_df["combined_gt_errors"].mean()
        print(f"  Total mass change: {total_mass:.1f} ± {total_mass_err:.1f} Gt")
        print(f"  Mean annual rate: {mean_rate:.1f} ± {mean_rate_err:.1f} Gt/yr")

        # SLE conversion (362.5 Gt ≈ 1 mm SLE)
        sle_total = total_mass / 362.5
        sle_err = total_mass_err / 362.5
        print(f"  Sea-level equivalent: {sle_total:.1f} ± {sle_err:.1f} mm SLE")

    # Save method and region statistics
    summary_stats = {
        "n_input_files": len(csv_files),
        "n_input_rows": len(input_df),
        "n_regions": input_df["region_key"].nunique(),
        "n_unique_authors": input_df["author"].nunique(),
        "method_counts": input_df.groupby("method")["source_file"].nunique().to_dict(),
        "region_file_counts": input_df.groupby("region_key")["source_file"].nunique().to_dict(),
    }
    with open(OUTPUT_DIR / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)

    print("\nDone. Data saved to outputs/")

if __name__ == "__main__":
    main()
