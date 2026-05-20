"""
Load and prepare GlaMBIE data for analysis.
"""
import pandas as pd
import numpy as np
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CAL = os.path.join(BASE, "data", "glambie", "results", "calendar_years")
DATA_HYD = os.path.join(BASE, "data", "glambie", "results", "hydrological_years")
DATA_INPUT = os.path.join(BASE, "data", "glambie", "input")
OUT = os.path.join(BASE, "outputs")

REGION_FILES = {
    0: "0_global.csv",
    1: "1_alaska.csv",
    2: "2_western_canada_us.csv",
    3: "3_arctic_canada_north.csv",
    4: "4_arctic_canada_south.csv",
    5: "5_greenland_periphery.csv",
    6: "6_iceland.csv",
    7: "7_svalbard.csv",
    8: "8_scandinavia.csv",
    9: "9_russian_arctic.csv",
    10: "10_north_asia.csv",
    11: "11_central_europe.csv",
    12: "12_caucasus_middle_east.csv",
    13: "13_central_asia.csv",
    14: "14_south_asia_west.csv",
    15: "15_south_asia_east.csv",
    16: "16_low_latitudes.csv",
    17: "17_southern_andes.csv",
    18: "18_new_zealand.csv",
    19: "19_antarctic_and_subantarctic.csv",
}

REGION_NAMES = {
    0: "Global",
    1: "Alaska",
    2: "Western Canada & US",
    3: "Arctic Canada North",
    4: "Arctic Canada South",
    5: "Greenland Periphery",
    6: "Iceland",
    7: "Svalbard",
    8: "Scandinavia",
    9: "Russian Arctic",
    10: "North Asia",
    11: "Central Europe",
    12: "Caucasus & Middle East",
    13: "Central Asia",
    14: "South Asia West",
    15: "South Asia East",
    16: "Low Latitudes",
    17: "Southern Andes",
    18: "New Zealand",
    19: "Antarctic & Subantarctic",
}

REGION_CODES = {
    0: "Global",
    1: "ALA", 2: "WNA", 3: "ACN", 4: "ACS", 5: "GRL",
    6: "ISL", 7: "SJM", 8: "SCA", 9: "RUA", 10: "ASN",
    11: "CEU", 12: "CAU", 13: "ASC", 14: "ASW", 15: "ASE",
    16: "TRP", 17: "SAN", 18: "NZL", 19: "ANT",
}


def load_calendar_years():
    """Load all calendar year data."""
    dfs = {}
    for idx, fname in REGION_FILES.items():
        if idx == 0:
            continue  # skip global in loop, we'll load separately
        path = os.path.join(DATA_CAL, fname)
        df = pd.read_csv(path)
        df["region_id"] = idx
        df["region_name"] = REGION_NAMES[idx]
        df["region_code"] = REGION_CODES[idx]
        dfs[idx] = df
    # Load global
    global_df = pd.read_csv(os.path.join(DATA_CAL, "0_global.csv"))
    global_df["region_id"] = 0
    global_df["region_name"] = "Global"
    global_df["region_code"] = "Global"
    return global_df, dfs


def load_hydrological_years():
    """Load all hydrological year data (with data group breakdowns)."""
    dfs = {}
    for idx in range(1, 20):
        fname = REGION_FILES[idx]
        path = os.path.join(DATA_HYD, fname)
        df = pd.read_csv(path)
        df["region_id"] = idx
        df["region_name"] = REGION_NAMES[idx]
        df["region_code"] = REGION_CODES[idx]
        dfs[idx] = df
    return dfs


def load_input_datasets():
    """Load all individual input datasets."""
    all_inputs = []
    for region_id in range(1, 20):
        region_dir = os.path.join(DATA_INPUT, REGION_NAMES[region_id].lower().replace(" ", "_").replace("&", "and").replace("'", ""))
        # Try matching the actual directory names
        region_dirs = [d for d in os.listdir(DATA_INPUT) if d.startswith(str(region_id) + "_")]
        if not region_dirs:
            continue
        region_dir = os.path.join(DATA_INPUT, region_dirs[0])
        if not os.path.isdir(region_dir):
            continue
        for f in sorted(os.listdir(region_dir)):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(region_dir, f))
                    df["input_file"] = f
                    df["region_id"] = region_id
                    df["region_name"] = REGION_NAMES[region_id]
                    # Parse data source from filename
                    parts = f.replace(".csv", "").split("_")
                    # Find data source
                    for src in ["altimetry", "gravimetry", "glaciological", "demdiff", "combined"]:
                        if src in f.lower():
                            df["data_source"] = src
                            break
                    else:
                        df["data_source"] = "unknown"
                    all_inputs.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return all_inputs


def save_outputs(global_df, regional_dfs, hydro_dfs):
    """Save prepared data to outputs."""
    os.makedirs(OUT, exist_ok=True)
    
    # Save global time series
    global_df.to_csv(os.path.join(OUT, "global_time_series.csv"), index=False)
    
    # Save regional combined
    all_regional = pd.concat(regional_dfs.values(), ignore_index=True)
    all_regional.to_csv(os.path.join(OUT, "regional_time_series.csv"), index=False)
    
    # Save summary statistics
    summary = []
    for idx in range(1, 20):
        df = regional_dfs[idx]
        # Compute total mass change over the period (2000-2023)
        total_gt = df["combined_gt"].sum()
        total_gt_err = np.sqrt((df["combined_gt_errors"]**2).sum())
        mean_mwe = df["combined_mwe"].mean()
        mean_mwe_err = np.sqrt((df["combined_mwe_errors"]**2).sum()) / len(df)
        area_mean = df["glacier_area"].mean()
        
        summary.append({
            "region_id": idx,
            "region_name": REGION_NAMES[idx],
            "region_code": REGION_CODES[idx],
            "total_mass_change_gt_2000_2023": round(total_gt, 1),
            "total_mass_change_error_gt": round(total_gt_err, 1),
            "mean_specific_mass_change_mwe_yr": round(mean_mwe, 4),
            "mean_specific_mass_change_error_mwe_yr": round(mean_mwe_err, 4),
            "mean_glacier_area_km2": round(area_mean, 1),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT, "regional_summary.csv"), index=False)
    
    # Save hydrological year data
    all_hydro = pd.concat(hydro_dfs.values(), ignore_index=True)
    all_hydro.to_csv(os.path.join(OUT, "hydrological_time_series.csv"), index=False)
    
    # Print key statistics
    g = global_df
    print(f"=== Global Summary (Calendar Years) ===")
    print(f"Period: {g['start_dates'].min():.0f} - {g['end_dates'].max():.0f}")
    print(f"Total mass change: {g['combined_gt'].sum():.1f} ± {np.sqrt((g['combined_gt_errors']**2).sum()):.1f} Gt")
    print(f"Mean annual mass change: {g['combined_gt'].mean():.1f} ± {g['combined_gt_errors'].mean():.1f} Gt/yr")
    print(f"Mean specific mass change: {g['combined_mwe'].mean():.4f} ± {g['combined_mwe_errors'].mean():.4f} m w.e./yr")
    print(f"Mean glacier area: {g['glacier_area'].mean():.0f} km²")
    print()
    
    print("=== Regional Summary ===")
    print(summary_df.to_string(index=False))
    
    return summary_df


if __name__ == "__main__":
    print("Loading calendar year data...")
    global_df, regional_dfs = load_calendar_years()
    
    print("Loading hydrological year data...")
    hydro_dfs = load_hydrological_years()
    
    print("Saving outputs...")
    summary_df = save_outputs(global_df, regional_dfs, hydro_dfs)
    
    print("\nDone!")
