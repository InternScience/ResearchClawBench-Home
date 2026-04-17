#!/usr/bin/env python3
"""
GlaMBIE Data Analysis: Global Glacier Mass Change Assessment (2000-2023)
Comprehensive analysis of 233+ observational estimates across 19 glacial regions.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_000_20260417_110902'
DATA = os.path.join(BASE, 'data', 'glambie')
INPUT = os.path.join(DATA, 'input')
RESULTS_CAL = os.path.join(DATA, 'results', 'calendar_years')
RESULTS_HYD = os.path.join(DATA, 'results', 'hydrological_years')
OUTPUTS = os.path.join(BASE, 'outputs')
IMAGES = os.path.join(BASE, 'report', 'images')
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)

# Region mapping
REGION_MAP = {
    '1': 'Alaska',
    '2': 'Western Canada & US',
    '3': 'Arctic Canada North',
    '4': 'Arctic Canada South',
    '5': 'Greenland Periphery',
    '6': 'Iceland',
    '7': 'Svalbard',
    '8': 'Scandinavia',
    '9': 'Russian Arctic',
    '10': 'North Asia',
    '11': 'Central Europe',
    '12': 'Caucasus & Middle East',
    '13': 'Central Asia',
    '14': 'South Asia West',
    '15': 'South Asia East',
    '16': 'Low Latitudes',
    '17': 'Southern Andes',
    '18': 'New Zealand',
    '19': 'Antarctic & Subantarctic',
}

METHOD_COLORS = {
    'glaciological': '#2ca02c',
    'demdiff': '#d62728',
    'altimetry': '#1f77b4',
    'gravimetry': '#ff7f0e',
    'combined': '#9467bd',
}

METHOD_LABELS = {
    'glaciological': 'Glaciological',
    'demdiff': 'DEM Differencing',
    'altimetry': 'Altimetry',
    'gravimetry': 'Gravimetry',
    'combined': 'Combined/Hybrid',
}

# ============================================================
# 1. Load all input datasets
# ============================================================
print("Loading input datasets...")
all_inputs = []
for region_dir in sorted(glob.glob(os.path.join(INPUT, '*'))):
    if not os.path.isdir(region_dir):
        continue
    region_name = os.path.basename(region_dir)
    region_id = region_name.split('_')[0]
    for csv_file in sorted(glob.glob(os.path.join(region_dir, '*.csv'))):
        fname = os.path.basename(csv_file)
        # Extract method
        method = None
        for m in ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']:
            if f'_{m}_' in fname:
                method = m
                break
        if method is None:
            continue
        try:
            df = pd.read_csv(csv_file)
            df['region_id'] = region_id
            df['region_name'] = REGION_MAP.get(region_id, region_name)
            df['method'] = method
            df['filename'] = fname
            df['source'] = fname.replace('.csv', '')
            all_inputs.append(df)
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")

input_df = pd.concat(all_inputs, ignore_index=True)
print(f"  Loaded {len(all_inputs)} input datasets with {len(input_df)} records total")

# ============================================================
# 2. Load result datasets
# ============================================================
print("Loading result datasets...")

# Calendar year results
cal_results = {}
for csv_file in sorted(glob.glob(os.path.join(RESULTS_CAL, '*.csv'))):
    fname = os.path.basename(csv_file)
    key = fname.replace('.csv', '')
    df = pd.read_csv(csv_file)
    cal_results[key] = df

# Hydrological year results
hyd_results = {}
for csv_file in sorted(glob.glob(os.path.join(RESULTS_HYD, '*.csv'))):
    fname = os.path.basename(csv_file)
    key = fname.replace('.csv', '')
    df = pd.read_csv(csv_file)
    hyd_results[key] = df

print(f"  Calendar year results: {len(cal_results)} files")
print(f"  Hydrological year results: {len(hyd_results)} files")

# ============================================================
# 3. Summary statistics
# ============================================================
print("\nComputing summary statistics...")

# Input dataset summary
input_summary = input_df.groupby(['region_id', 'region_name', 'method']).agg(
    n_datasets=('filename', 'nunique'),
    n_records=('start_dates', 'count'),
    earliest_start=('start_dates', 'min'),
    latest_end=('end_dates', 'max'),
).reset_index()

# Save
input_summary.to_csv(os.path.join(OUTPUTS, 'input_summary_by_region_method.csv'), index=False)

# Count by method
method_counts = input_df.groupby('method')['filename'].nunique().to_dict()
print("  Datasets by method:", method_counts)

# Count by region
region_counts = input_df.groupby('region_name')['filename'].nunique().to_dict()
print("  Datasets by region:", region_counts)

# ============================================================
# 4. Global time series analysis
# ============================================================
print("\nAnalyzing global time series...")
global_cal = cal_results['0_global'].copy()
global_cal['year'] = global_cal['start_dates'].astype(int)

# Cumulative mass change
global_cal['cumulative_gt'] = global_cal['combined_gt'].cumsum()
global_cal['cumulative_mwe'] = global_cal['combined_mwe'].cumsum()

# Propagate errors for cumulative
global_cal['cumulative_gt_err'] = np.sqrt(np.cumsum(global_cal['combined_gt_errors']**2))
global_cal['cumulative_mwe_err'] = np.sqrt(np.cumsum(global_cal['combined_mwe_errors']**2))

# Sea level equivalent (1 Gt = ~0.00278 mm SLE, or 362.5 Gt = 1 mm)
global_cal['sle_mm'] = global_cal['combined_gt'] / -362.5  # positive SLE for negative mass change
global_cal['cumulative_sle_mm'] = global_cal['cumulative_gt'] / -362.5

# Save global results
global_cal.to_csv(os.path.join(OUTPUTS, 'global_timeseries_calendar.csv'), index=False)

# Summary stats
total_loss_gt = global_cal['combined_gt'].sum()
total_loss_err = np.sqrt((global_cal['combined_gt_errors']**2).sum())
avg_rate = global_cal['combined_gt'].mean()
avg_rate_err = total_loss_err / len(global_cal)

print(f"  Total mass change 2000-2023: {total_loss_gt:.1f} ± {total_loss_err:.1f} Gt")
print(f"  Average annual rate: {avg_rate:.1f} ± {avg_rate_err:.1f} Gt/yr")
print(f"  Total SLE contribution: {-total_loss_gt/362.5:.1f} mm")

# Acceleration analysis: compare first half vs second half
mid = len(global_cal) // 2
first_half_rate = global_cal['combined_gt'].iloc[:mid].mean()
second_half_rate = global_cal['combined_gt'].iloc[mid:].mean()
print(f"  First half avg rate (2000-{2000+mid}): {first_half_rate:.1f} Gt/yr")
print(f"  Second half avg rate ({2000+mid}-2023): {second_half_rate:.1f} Gt/yr")

# Linear trend for acceleration
from scipy import stats
years = global_cal['year'].values
rates = global_cal['combined_gt'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(years, rates)
print(f"  Linear trend: {slope:.2f} ± {std_err:.2f} Gt/yr² (p={p_value:.4f})")

# ============================================================
# 5. Regional analysis
# ============================================================
print("\nAnalyzing regional time series...")

regional_summary = []
for key, df in sorted(cal_results.items()):
    if key == '0_global':
        continue
    region_id = key.split('_')[0]
    region_name = REGION_MAP.get(region_id, key)
    
    total = df['combined_gt'].sum()
    total_err = np.sqrt((df['combined_gt_errors']**2).sum())
    avg_rate = df['combined_gt'].mean()
    avg_mwe = df['combined_mwe'].mean()
    avg_area = df['glacier_area'].mean()
    
    # Trend
    yrs = df['start_dates'].values
    sl, _, rv, pv, se = stats.linregress(yrs, df['combined_gt'].values)
    
    regional_summary.append({
        'region_id': region_id,
        'region_name': region_name,
        'total_gt': total,
        'total_gt_err': total_err,
        'avg_rate_gt_yr': avg_rate,
        'avg_rate_mwe_yr': avg_mwe,
        'avg_area_km2': avg_area,
        'trend_gt_yr2': sl,
        'trend_p_value': pv,
        'n_years': len(df),
    })

regional_df = pd.DataFrame(regional_summary)
regional_df = regional_df.sort_values('total_gt')
regional_df.to_csv(os.path.join(OUTPUTS, 'regional_summary.csv'), index=False)
print(regional_df[['region_name', 'total_gt', 'avg_rate_gt_yr', 'avg_rate_mwe_yr']].to_string())

# ============================================================
# 6. Method comparison per region (hydrological year data)
# ============================================================
print("\nAnalyzing method agreement...")

method_comparison = []
for key, df in sorted(hyd_results.items()):
    region_id = key.split('_')[0]
    region_name = REGION_MAP.get(region_id, key)
    
    # Check which method columns exist
    for method_prefix in ['altimetry', 'gravimetry', 'demdiff_and_glaciological']:
        gt_col = f'{method_prefix}_gt'
        mwe_col = f'{method_prefix}_mwe'
        if gt_col in df.columns:
            valid = df[gt_col].dropna()
            if len(valid) > 0:
                method_comparison.append({
                    'region_id': region_id,
                    'region_name': region_name,
                    'method': method_prefix,
                    'n_years': len(valid),
                    'avg_rate_gt': valid.mean(),
                    'total_gt': valid.sum(),
                })

method_comp_df = pd.DataFrame(method_comparison)
method_comp_df.to_csv(os.path.join(OUTPUTS, 'method_comparison.csv'), index=False)

# ============================================================
# Save key numbers as JSON
# ============================================================
key_results = {
    'total_mass_change_2000_2023_gt': round(total_loss_gt, 1),
    'total_mass_change_uncertainty_gt': round(total_loss_err, 1),
    'average_annual_rate_gt_yr': round(avg_rate, 1),
    'total_sle_mm': round(-total_loss_gt/362.5, 1),
    'acceleration_gt_yr2': round(slope, 2),
    'acceleration_p_value': round(p_value, 4),
    'first_half_rate_gt_yr': round(first_half_rate, 1),
    'second_half_rate_gt_yr': round(second_half_rate, 1),
    'n_input_datasets': len(all_inputs),
    'n_regions': 19,
    'period': '2000-2023',
}
with open(os.path.join(OUTPUTS, 'key_results.json'), 'w') as f:
    json.dump(key_results, f, indent=2)

print("\n=== KEY RESULTS ===")
for k, v in key_results.items():
    print(f"  {k}: {v}")

print("\nAnalysis complete. Now generating figures...")
