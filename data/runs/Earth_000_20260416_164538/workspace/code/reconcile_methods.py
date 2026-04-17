#!/usr/bin/env python3
"""
GlaMBIE Method Reconciliation Script

This script reconciles multiple observational methods to produce
consistent regional and global mass change time series.

Approach:
1. For each region and year, compute weighted ensemble mean from all available methods
2. Weight by inverse variance (uncertainty weighting)
3. Propagate uncertainties appropriately
4. Convert between m w.e. and Gt using regional glacier areas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
WORKSPACE_ROOT = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Earth_000_20260416_164538")
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

# Regional glacier areas (km²) - from GlaMBIE/RGI statistics
# These are approximate values for converting between m w.e. and Gt
# 1 m w.e. over 1 km² = 0.001 Gt (since density of water = 1000 kg/m³)
REGION_AREAS_KM2 = {
    "Alaska": 76900,
    "Western Canada US": 27700,
    "Arctic Canada North": 110500,
    "Arctic Canada South": 39200,
    "Greenland Periphery": 76400,
    "Iceland": 10900,
    "Svalbard": 33300,
    "Scandinavia": 2900,
    "Russian Arctic": 28200,
    "North Asia": 19000,
    "Central Europe": 2900,
    "Caucasus Middle East": 2100,
    "Central Asia": 96000,
    "South Asia West": 35700,
    "South Asia East": 36500,
    "Low Latitudes": 2400,
    "Southern Andes": 25900,
    "New Zealand": 1200,
    "Antarctic and Subantarctic": 3200
}

# Global total area
GLOBAL_AREA_KM2 = sum(REGION_AREAS_KM2.values())


def mwe_to_gt(mwe, area_km2):
    """Convert specific mass change (m w.e.) to total mass change (Gt)."""
    return mwe * area_km2 * 0.001


def gt_to_mwe(gt, area_km2):
    """Convert total mass change (Gt) to specific mass change (m w.e.)."""
    return gt / (area_km2 * 0.001)


def load_processed_data():
    """Load the processed annual data."""
    annual_file = OUTPUTS_DIR / "annual_data.parquet"
    df = pd.read_parquet(annual_file)
    return df


def normalize_units(df):
    """
    Normalize all measurements to m w.e. for consistency.
    
    For datasets in Gt, convert to m w.e. using regional areas.
    For datasets already in m or mwe, keep as is.
    """
    df_normalized = df.copy()
    
    # Identify Gt measurements
    gt_mask = df_normalized["unit"].str.lower() == "gt"
    
    # Convert Gt to m w.e. for those measurements
    for region in df_normalized["region"].unique():
        region_mask = df_normalized["region"] == region
        area = REGION_AREAS_KM2.get(region, 10000)  # default area if unknown
        
        gt_region_mask = gt_mask & region_mask
        if gt_region_mask.any():
            df_normalized.loc[gt_region_mask, "changes"] = gt_to_mwe(
                df_normalized.loc[gt_region_mask, "changes"], 
                area
            )
            df_normalized.loc[gt_region_mask, "errors"] = df_normalized.loc[gt_region_mask, "errors"] / (area * 0.001)
            df_normalized.loc[gt_region_mask, "unit_original"] = "Gt"
    
    df_normalized["unit_final"] = "m w.e."
    
    return df_normalized


def compute_weighted_ensemble(group):
    """
    Compute weighted ensemble mean from multiple estimates.
    
    Uses inverse variance weighting: weight = 1/σ²
    Ensemble mean = Σ(w_i * x_i) / Σ(w_i)
    Ensemble uncertainty = sqrt(1 / Σ(w_i))
    
    Also computes method spread as additional uncertainty measure.
    """
    if len(group) == 0:
        return pd.Series({
            "reconciled_value": np.nan,
            "reconciled_uncertainty": np.nan,
            "n_estimates": 0,
            "method_spread": np.nan,
            "methods_used": ""
        })
    
    changes = group["changes"].values
    errors = group["errors"].values
    
    # Filter out invalid values
    valid_mask = np.isfinite(changes) & np.isfinite(errors) & (errors > 0)
    if not valid_mask.any():
        return pd.Series({
            "reconciled_value": np.nan,
            "reconciled_uncertainty": np.nan,
            "n_estimates": 0,
            "method_spread": np.nan,
            "methods_used": ""
        })
    
    changes_valid = changes[valid_mask]
    errors_valid = errors[valid_mask]
    
    # Inverse variance weights
    weights = 1.0 / (errors_valid ** 2)
    
    # Weighted mean
    weighted_mean = np.sum(weights * changes_valid) / np.sum(weights)
    
    # Uncertainty from weighting
    uncertainty_weighted = np.sqrt(1.0 / np.sum(weights))
    
    # Method spread (standard deviation of estimates)
    method_spread = np.std(changes_valid) if len(changes_valid) > 1 else 0
    
    # Combined uncertainty: quadrature sum of weighted uncertainty and spread
    # This accounts for both measurement uncertainty and method disagreement
    combined_uncertainty = np.sqrt(uncertainty_weighted**2 + method_spread**2)
    
    # Record which methods contributed
    methods_used = ",".join(group["method"].unique())
    
    return pd.Series({
        "reconciled_value": weighted_mean,
        "reconciled_uncertainty": combined_uncertainty,
        "n_estimates": len(changes_valid),
        "method_spread": method_spread,
        "methods_used": methods_used
    })


def reconcile_by_region_and_year(df):
    """
    Reconcile all estimates for each region and year combination.
    """
    # Group by region and midpoint year
    grouped = df.groupby(["region", "midpoint_year"])
    
    results = grouped.apply(compute_weighted_ensemble).reset_index()
    
    return results


def compute_global_aggregate(regional_results):
    """
    Aggregate regional results to global totals.
    
    Global specific mass change (m w.e.) = area-weighted average of regional values
    Global total mass change (Gt) = sum of regional Gt values
    """
    regional_results = regional_results.copy()
    
    # Add area column
    regional_results["area_km2"] = regional_results["region"].map(REGION_AREAS_KM2)
    
    # Convert reconciled values to Gt
    regional_results["reconciled_Gt"] = mwe_to_gt(
        regional_results["reconciled_value"],
        regional_results["area_km2"]
    )
    regional_results["reconciled_Gt_uncertainty"] = (
        regional_results["reconciled_uncertainty"] * regional_results["area_km2"] * 0.001
    )
    
    # Aggregate to global by year
    global_agg = regional_results.groupby("midpoint_year").agg({
        "reconciled_value": lambda x: np.average(x, weights=regional_results.loc[x.index, "area_km2"]),
        "reconciled_uncertainty": lambda x: np.sqrt(np.sum(x**2)),  # RSS of uncertainties
        "reconciled_Gt": "sum",
        "reconciled_Gt_uncertainty": lambda x: np.sqrt(np.sum(x**2)),
        "n_estimates": "sum"
    }).reset_index()
    
    global_agg.columns = [
        "midpoint_year", 
        "global_mwe", 
        "global_mwe_uncertainty",
        "global_Gt", 
        "global_Gt_uncertainty",
        "n_estimates"
    ]
    
    return global_agg, regional_results


def filter_to_target_period(df, start_year=2000, end_year=2023):
    """Filter data to target period."""
    return df[(df["midpoint_year"] >= start_year) & (df["midpoint_year"] <= end_year)]


def main():
    print("=" * 60)
    print("GlaMBIE Method Reconciliation")
    print("=" * 60)
    
    # Load processed data
    print("\n1. Loading processed data...")
    df = load_processed_data()
    print(f"   Loaded {len(df)} records")
    
    # Normalize units to m w.e.
    print("\n2. Normalizing units to m w.e....")
    df_normalized = normalize_units(df)
    
    # Filter to target period
    print("\n3. Filtering to 2000-2023...")
    df_target = filter_to_target_period(df_normalized)
    print(f"   Records in target period: {len(df_target)}")
    
    # Reconcile by region and year
    print("\n4. Computing reconciled estimates by region and year...")
    regional_reconciled = reconcile_by_region_and_year(df_target)
    print(f"   Region-year combinations: {len(regional_reconciled)}")
    
    # Compute global aggregates
    print("\n5. Computing global aggregates...")
    global_results, regional_with_gt = compute_global_aggregate(regional_reconciled)
    print(f"   Years with global estimates: {len(global_results)}")
    
    # Save results
    print("\n6. Saving results...")
    
    # Regional time series
    regional_output = OUTPUTS_DIR / "regional_timeseries.csv"
    regional_with_gt.to_csv(regional_output, index=False)
    print(f"   Saved: {regional_output}")
    
    # Global time series
    global_output = OUTPUTS_DIR / "global_timeseries.csv"
    global_results.to_csv(global_output, index=False)
    print(f"   Saved: {global_output}")
    
    # Summary statistics
    summary = {
        "regional_summary": {},
        "global_summary": {
            "mean_annual_mwe": float(global_results["global_mwe"].mean()),
            "std_annual_mwe": float(global_results["global_mwe"].std()),
            "mean_annual_Gt": float(global_results["global_Gt"].mean()),
            "cumulative_Gt_2000_2023": float(global_results["global_Gt"].sum()),
            "total_years": len(global_results)
        }
    }
    
    for region in regional_with_gt["region"].unique():
        region_data = regional_with_gt[regional_with_gt["region"] == region]
        summary["regional_summary"][region] = {
            "mean_annual_mwe": float(region_data["reconciled_value"].mean()) if len(region_data) > 0 else None,
            "std_annual_mwe": float(region_data["reconciled_value"].std()) if len(region_data) > 0 else None,
            "mean_annual_Gt": float(region_data["reconciled_Gt"].mean()) if len(region_data) > 0 else None,
            "n_years": len(region_data)
        }
    
    summary_output = OUTPUTS_DIR / "reconciliation_summary.json"
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {summary_output}")
    
    # Print key results
    print("\n7. Key Results Summary:")
    print("-" * 60)
    print(f"Global mean annual mass change (2000-2023):")
    print(f"  {summary['global_summary']['mean_annual_mwe']:.2f} ± {global_results['global_mwe_uncertainty'].mean():.2f} m w.e./yr")
    print(f"  {summary['global_summary']['mean_annual_Gt']:.2f} ± {global_results['global_Gt_uncertainty'].mean():.2f} Gt/yr")
    print(f"\nCumulative global mass loss (2000-2023):")
    print(f"  {summary['global_summary']['cumulative_Gt_2000_2023']:.2f} Gt")
    
    print("\n" + "=" * 60)
    print("Reconciliation complete!")
    print("=" * 60)
    
    return regional_with_gt, global_results


if __name__ == "__main__":
    main()
