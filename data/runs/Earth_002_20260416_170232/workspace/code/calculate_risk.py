#!/usr/bin/env python3
"""
Calculate composite risk index for mangroves.
Combines sea level rise and tropical cyclone risk components.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# Paths
OUTPUT_DIR = Path("outputs")

def load_processed_data():
    """Load all processed data files."""
    # Load SLR data for each scenario
    slr_scenarios = {}
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        slr_scenarios[scenario] = pd.read_csv(OUTPUT_DIR / f"slr_{scenario}.csv")
    
    # Load TC frequency grid
    tc_freq = pd.read_csv(OUTPUT_DIR / "tc_frequency_grid.csv")
    
    # Load mangrove grid
    mangrove_grid = pd.read_csv(OUTPUT_DIR / "mangrove_grid.csv")
    
    return slr_scenarios, tc_freq, mangrove_grid


def calculate_slr_risk(slr_rate_mm_yr):
    """
    Calculate SLR risk score based on rate thresholds.
    
    Based on Saintilan et al. (2023) and related work:
    - < 4 mm/yr: Low risk (mangroves can likely keep pace)
    - 4-7 mm/yr: Medium risk (adjustment deficit likely)
    - > 7 mm/yr: High risk (retreat highly likely)
    
    Returns normalized risk score 0-1.
    """
    # Normalize: 0 at 0 mm/yr, 1 at 10+ mm/yr
    risk = np.clip(slr_rate_mm_yr / 10.0, 0, 1)
    return risk


def calculate_tc_risk(freq_total, cat_weights=None):
    """
    Calculate TC risk score based on frequency and intensity.
    
    Based on Mo et al. (2023): Category 3-5 contribute most to damage.
    
    Returns normalized risk score 0-1.
    """
    if cat_weights is None:
        # Weight higher categories more heavily
        cat_weights = {
            "cat_1": 0.1,
            "cat_2": 0.2,
            "cat_3": 0.3,
            "cat_4": 0.4,
            "cat_5": 0.5
        }
    
    # Calculate weighted frequency
    weighted_freq = 0
    for cat, weight in cat_weights.items():
        if cat in freq_total.columns:
            weighted_freq += freq_total[cat].values * weight
    
    # Normalize: use log scale since frequency varies widely
    # Max observed frequency ~1000+, so log10(max) ~ 3
    max_freq = 100  # Reference maximum annual frequency
    risk = np.clip(np.log10(weighted_freq + 1) / np.log10(max_freq + 1), 0, 1)
    
    return risk


def spatial_join_slr_to_mangroves(slr_df, mangrove_df):
    """
    Join SLR data to mangrove locations using nearest neighbor.
    Uses KD-tree for efficient spatial matching.
    """
    # Build KD-tree from SLR locations
    slr_coords = np.column_stack([slr_df["lon"], slr_df["lat"]])
    tree = cKDTree(slr_coords)
    
    # Query mangrove locations
    mangrove_coords = np.column_stack([mangrove_df["lon_mean"], mangrove_df["lat_mean"]])
    distances, indices = tree.query(mangrove_coords, k=1)
    
    # Add SLR data to mangrove dataframe
    result = mangrove_df.copy()
    result["slr_location_idx"] = indices
    result["slr_distance_km"] = distances * 111  # Approximate km conversion
    
    return result, indices


def spatial_join_tc_to_mangroves(tc_freq_df, mangrove_df):
    """
    Join TC frequency data to mangrove locations using grid matching.
    """
    # Merge on grid coordinates
    result = mangrove_df.merge(
        tc_freq_df,
        on=["lat_grid", "lon_grid"],
        how="left"
    )
    
    # Fill NaN with 0 (no TC activity recorded)
    tc_cols = [c for c in result.columns if c.startswith("cat_") or c == "freq_total"]
    result[tc_cols] = result[tc_cols].fillna(0)
    
    return result


def calculate_composite_risk(slr_risk, tc_risk, weights=None):
    """
    Calculate composite risk index from SLR and TC components.
    
    Default: equal weighting (0.5 each)
    """
    if weights is None:
        weights = {"slr": 0.5, "tc": 0.5}
    
    cri = weights["slr"] * slr_risk + weights["tc"] * tc_risk
    return cri


def main():
    print("=" * 60)
    print("Loading processed data...")
    print("=" * 60)
    
    slr_scenarios, tc_freq, mangrove_grid = load_processed_data()
    
    print(f"SLR scenarios: {list(slr_scenarios.keys())}")
    print(f"TC frequency grid: {len(tc_freq)} cells")
    print(f"Mangrove grid cells: {len(mangrove_grid)}")
    
    # Join SLR data to mangrove locations
    print("\n" + "=" * 60)
    print("Spatially joining SLR data to mangrove locations...")
    print("=" * 60)
    
    mangrove_with_slr = {}
    for scenario, slr_df in slr_scenarios.items():
        joined, indices = spatial_join_slr_to_mangroves(slr_df, mangrove_grid)
        
        # Add SLR rates
        joined["slr_rate"] = slr_df["slr_rate_mm_yr"].values[indices]
        joined["slr_risk"] = calculate_slr_risk(joined["slr_rate"])
        
        mangrove_with_slr[scenario] = joined
        
        print(f"{scenario}: Mean SLR rate = {joined['slr_rate'].mean():.2f} mm/yr, "
              f"Mean SLR risk = {joined['slr_risk'].mean():.3f}")
    
    # Join TC data to mangrove locations
    print("\n" + "=" * 60)
    print("Spatially joining TC data to mangrove locations...")
    print("=" * 60)
    
    for scenario in mangrove_with_slr.keys():
        joined = spatial_join_tc_to_mangroves(tc_freq, mangrove_with_slr[scenario])
        
        # Calculate TC risk
        tc_cols = ["cat_1", "cat_2", "cat_3", "cat_4", "cat_5"]
        existing_cats = [c for c in tc_cols if c in joined.columns]
        
        # Use available categories
        cat_weights = {f"cat_{i}": i * 0.1 for i in range(1, 6)}
        cat_weights = {k: v for k, v in cat_weights.items() if k in existing_cats}
        
        joined["tc_risk"] = calculate_tc_risk(joined, cat_weights)
        
        # Calculate composite risk index
        joined["composite_risk"] = calculate_composite_risk(
            joined["slr_risk"], 
            joined["tc_risk"]
        )
        
        # Risk classification
        joined["risk_class"] = pd.cut(
            joined["composite_risk"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"]
        )
        
        mangrove_with_slr[scenario] = joined
        
        print(f"{scenario}: Mean TC risk = {joined['tc_risk'].mean():.3f}, "
              f"Mean composite risk = {joined['composite_risk'].mean():.3f}")
        print(f"  Risk distribution: {joined['risk_class'].value_counts().to_dict()}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving risk assessment results...")
    print("=" * 60)
    
    for scenario, df in mangrove_with_slr.items():
        output_file = OUTPUT_DIR / f"risk_assessment_{scenario}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    
    # Create summary statistics
    summary_data = []
    for scenario, df in mangrove_with_slr.items():
        summary_data.append({
            "scenario": scenario,
            "n_mangrove_cells": len(df),
            "mean_slr_rate": df["slr_rate"].mean(),
            "std_slr_rate": df["slr_rate"].std(),
            "mean_slr_risk": df["slr_risk"].mean(),
            "mean_tc_risk": df["tc_risk"].mean(),
            "mean_composite_risk": df["composite_risk"].mean(),
            "std_composite_risk": df["composite_risk"].std(),
            "pct_high_risk": (df["risk_class"] == "High").mean() * 100,
            "pct_medium_risk": (df["risk_class"] == "Medium").mean() * 100,
            "pct_low_risk": (df["risk_class"] == "Low").mean() * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "risk_summary.csv", index=False)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Risk calculation complete!")
    print("=" * 60)
    
    return mangrove_with_slr, summary_df


if __name__ == "__main__":
    mangrove_risk, summary = main()
