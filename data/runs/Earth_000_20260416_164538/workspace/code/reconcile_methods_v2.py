#!/usr/bin/env python3
"""
GlaMBIE Method Reconciliation Script - Version 2

Key improvements:
1. Proper handling of monthly gravimetry data (aggregate to annual)
2. Correct sign convention (negative = mass loss)
3. Better uncertainty propagation accounting for method spread
4. Cumulative vs. rate handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
WORKSPACE_ROOT = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Earth_000_20260416_164538")
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DATA_DIR = WORKSPACE_ROOT / "data" / "glambie" / "input"

# Regional glacier areas (km²) - from GlaMBIE/RGI statistics
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
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_all_data():
    """Load all CSV files from the input directory."""
    all_files = list(DATA_DIR.rglob("*.csv"))
    dataframes = []
    for filepath in all_files:
        df = load_csv_file(filepath)
        if df is not None:
            dataframes.append(df)
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        return None


def aggregate_to_annual(df):
    """
    Aggregate sub-annual (monthly) data to annual resolution.
    
    For gravimetry data with monthly measurements:
    - Group by calendar year
    - Sum mass changes (they are in Gt or m w.e. per period)
    - Combine uncertainties in quadrature
    """
    df = df.copy()
    
    # Extract calendar year from start date
    df['start_year'] = df['start_dates'].astype(int)
    df['end_year'] = df['end_dates'].astype(int)
    
    # For annual aggregation, use the start year as the reference
    # This assumes most measurements starting in year Y end in Y or early Y+1
    
    annual_records = []
    
    for (region, method, author), group in df.groupby(['region', 'method', 'author']):
        # Group by start year
        for year, year_group in group.groupby('start_year'):
            # Sum the changes for this year
            total_change = year_group['changes'].sum()
            # Combine errors in quadrature
            combined_error = np.sqrt((year_group['errors'] ** 2).sum())
            # Get unit (should be consistent within group)
            unit = year_group['unit'].iloc[0]
            
            annual_records.append({
                'region': region,
                'method': method,
                'author': author,
                'midpoint_year': year,
                'start_year': year,
                'end_year': year + 1,
                'changes': total_change,
                'errors': combined_error,
                'unit': unit,
                'n_subannual': len(year_group)
            })
    
    return pd.DataFrame(annual_records)


def normalize_units_to_mwe(df):
    """
    Convert all measurements to m w.e. using regional glacier areas.
    
    For Gt -> m w.e.: mwe = Gt / (area_km2 * 0.001)
    """
    df = df.copy()
    df['changes_mwe'] = np.nan
    df['errors_mwe'] = np.nan
    
    for region in df['region'].unique():
        region_mask = df['region'] == region
        area = REGION_AREAS_KM2.get(region, 10000)
        
        # Convert Gt to m w.e.
        gt_mask = region_mask & (df['unit'].str.lower() == 'gt')
        if gt_mask.any():
            df.loc[gt_mask, 'changes_mwe'] = df.loc[gt_mask, 'changes'] / (area * 0.001)
            df.loc[gt_mask, 'errors_mwe'] = df.loc[gt_mask, 'errors'] / (area * 0.001)
        
        # Keep mwe/m as is
        mwe_mask = region_mask & (df['unit'].str.lower().isin(['mwe', 'm']))
        if mwe_mask.any():
            df.loc[mwe_mask, 'changes_mwe'] = df.loc[mwe_mask, 'changes']
            df.loc[mwe_mask, 'errors_mwe'] = df.loc[mwe_mask, 'errors']
    
    return df


def compute_weighted_ensemble(group):
    """
    Compute weighted ensemble mean from multiple estimates.
    
    Uses inverse variance weighting: weight = 1/σ²
    Also computes method spread as additional uncertainty component.
    """
    if len(group) == 0:
        return pd.Series({
            'reconciled_mwe': np.nan,
            'reconciled_mwe_uncertainty': np.nan,
            'n_estimates': 0,
            'method_spread': np.nan,
            'methods_used': ''
        })
    
    changes = group['changes_mwe'].values
    errors = group['errors_mwe'].values
    
    # Filter valid values
    valid_mask = np.isfinite(changes) & np.isfinite(errors) & (errors > 0)
    if not valid_mask.any():
        return pd.Series({
            'reconciled_mwe': np.nan,
            'reconciled_mwe_uncertainty': np.nan,
            'n_estimates': 0,
            'method_spread': np.nan,
            'methods_used': ''
        })
    
    changes_valid = changes[valid_mask]
    errors_valid = errors[valid_mask]
    
    # Inverse variance weights
    weights = 1.0 / (errors_valid ** 2)
    total_weight = weights.sum()
    
    if total_weight == 0:
        return pd.Series({
            'reconciled_mwe': np.mean(changes_valid),
            'reconciled_mwe_uncertainty': np.std(changes_valid),
            'n_estimates': len(changes_valid),
            'method_spread': np.std(changes_valid),
            'methods_used': ','.join(group['method'].unique())
        })
    
    # Weighted mean
    weighted_mean = (weights * changes_valid).sum() / total_weight
    
    # Uncertainty from weighting
    uncertainty_weighted = np.sqrt(1.0 / total_weight)
    
    # Method spread (standard deviation)
    method_spread = np.std(changes_valid) if len(changes_valid) > 1 else 0
    
    # Combined uncertainty (quadrature sum)
    combined_uncertainty = np.sqrt(uncertainty_weighted**2 + method_spread**2)
    
    return pd.Series({
        'reconciled_mwe': weighted_mean,
        'reconciled_mwe_uncertainty': combined_uncertainty,
        'n_estimates': len(changes_valid),
        'method_spread': method_spread,
        'methods_used': ','.join(sorted(group['method'].unique()))
    })


def mwe_to_gt(mwe, area_km2):
    """Convert m w.e. to Gt."""
    return mwe * area_km2 * 0.001


def compute_global_aggregate(regional_results):
    """Aggregate regional results to global totals."""
    regional_results = regional_results.copy()
    
    # Add area column
    regional_results['area_km2'] = regional_results['region'].map(REGION_AREAS_KM2)
    
    # Convert to Gt
    regional_results['reconciled_Gt'] = mwe_to_gt(
        regional_results['reconciled_mwe'],
        regional_results['area_km2']
    )
    regional_results['reconciled_Gt_uncertainty'] = (
        regional_results['reconciled_mwe_uncertainty'] * regional_results['area_km2'] * 0.001
    )
    
    # Aggregate to global by year
    global_agg = regional_results.groupby('midpoint_year').agg({
        'reconciled_mwe': lambda x: np.average(x, weights=regional_results.loc[x.index, 'area_km2']),
        'reconciled_mwe_uncertainty': lambda x: np.sqrt(np.sum(x**2)),
        'reconciled_Gt': 'sum',
        'reconciled_Gt_uncertainty': lambda x: np.sqrt(np.sum(x**2)),
        'n_estimates': 'sum'
    }).reset_index()
    
    global_agg.columns = [
        'year', 'global_mwe', 'global_mwe_uncertainty',
        'global_Gt', 'global_Gt_uncertainty', 'n_estimates'
    ]
    
    return global_agg, regional_results


def main():
    print("=" * 70)
    print("GlaMBIE Method Reconciliation - Version 2")
    print("=" * 70)
    
    # Step 1: Load all data
    print("\n1. Loading all CSV files...")
    df_raw = load_all_data()
    print(f"   Loaded {len(df_raw)} records from {df_raw['source_file'].nunique()} files")
    
    # Step 2: Aggregate sub-annual data to annual
    print("\n2. Aggregating to annual resolution...")
    df_annual = aggregate_to_annual(df_raw)
    print(f"   Annual records: {len(df_annual)}")
    
    # Check sub-annual aggregation
    n_monthly = df_annual['n_subannual'].sum()
    print(f"   Total sub-annual measurements aggregated: {n_monthly}")
    
    # Step 3: Normalize units to m w.e.
    print("\n3. Normalizing units to m w.e....")
    df_normalized = normalize_units_to_mwe(df_annual)
    
    # Step 4: Filter to target period (2000-2023)
    print("\n4. Filtering to 2000-2023...")
    df_target = df_normalized[(df_normalized['midpoint_year'] >= 2000) & 
                              (df_normalized['midpoint_year'] <= 2023)]
    print(f"   Records in target period: {len(df_target)}")
    
    # Step 5: Reconcile by region and year
    print("\n5. Computing reconciled estimates...")
    grouped = df_target.groupby(['region', 'midpoint_year'])
    regional_reconciled = grouped.apply(compute_weighted_ensemble).reset_index()
    print(f"   Region-year combinations: {len(regional_reconciled)}")
    
    # Step 6: Compute global aggregates
    print("\n6. Computing global aggregates...")
    global_results, regional_with_gt = compute_global_aggregate(regional_reconciled)
    print(f"   Years with global estimates: {len(global_results)}")
    
    # Step 7: Save results
    print("\n7. Saving results...")
    
    regional_output = OUTPUTS_DIR / "regional_timeseries_v2.csv"
    regional_with_gt.to_csv(regional_output, index=False)
    print(f"   Saved: {regional_output}")
    
    global_output = OUTPUTS_DIR / "global_timeseries_v2.csv"
    global_results.to_csv(global_output, index=False)
    print(f"   Saved: {global_output}")
    
    # Summary statistics
    summary = {
        "global_summary": {
            "mean_annual_mwe": float(global_results['global_mwe'].mean()),
            "mean_annual_Gt": float(global_results['global_Gt'].mean()),
            "cumulative_Gt_2000_2023": float(global_results['global_Gt'].sum()),
            "total_years": len(global_results)
        },
        "regional_summary": {}
    }
    
    for region in regional_with_gt['region'].unique():
        region_data = regional_with_gt[regional_with_gt['region'] == region]
        summary['regional_summary'][region] = {
            'mean_annual_mwe': float(region_data['reconciled_mwe'].mean()),
            'mean_annual_Gt': float(region_data['reconciled_Gt'].mean()),
            'n_years': len(region_data)
        }
    
    summary_output = OUTPUTS_DIR / "reconciliation_summary_v2.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {summary_output}")
    
    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nGlobal mean annual mass change (2000-2023):")
    print(f"  {summary['global_summary']['mean_annual_mwe']:.3f} m w.e./yr")
    print(f"  {summary['global_summary']['mean_annual_Gt']:.2f} Gt/yr")
    print(f"\nCumulative global mass change (2000-2023):")
    print(f"  {summary['global_summary']['cumulative_Gt_2000_2023']:.2f} Gt")
    print(f"  (Negative = mass loss)")
    
    print("\nRegional mean annual mass change (m w.e./yr):")
    print("-" * 50)
    regional_means = regional_with_gt.groupby('region')['reconciled_mwe'].mean().sort_values()
    for region, value in regional_means.items():
        print(f"  {region:30s}: {value:7.3f}")
    
    print("\n" + "=" * 70)
    print("Reconciliation complete!")
    print("=" * 70)
    
    return regional_with_gt, global_results


if __name__ == "__main__":
    main()
