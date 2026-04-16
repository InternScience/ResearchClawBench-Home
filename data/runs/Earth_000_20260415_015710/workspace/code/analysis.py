"""
GlaMBIE Glacier Mass Change Reconciliation Analysis
====================================================
Main analysis script that loads, processes, and analyzes the GlaMBIE dataset
to produce reconciled 2000-2023 regional and global glacial mass change time series.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'glambie')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
IMAGE_DIR = os.path.join(BASE_DIR, 'report', 'images')
CODE_DIR = os.path.join(BASE_DIR, 'code')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Region mapping (RGI region numbers and names)
REGION_MAP = {
    0: 'Global',
    1: 'Alaska',
    2: 'Western Canada & US',
    3: 'Arctic Canada North',
    4: 'Arctic Canada South',
    5: 'Greenland Periphery',
    6: 'Iceland',
    7: 'Svalbard',
    8: 'Scandinavia',
    9: 'Russian Arctic',
    10: 'North Asia',
    11: 'Central Europe',
    12: 'Caucasus & Middle East',
    13: 'Central Asia',
    14: 'South Asia West',
    15: 'South Asia East',
    16: 'Low Latitudes',
    17: 'Southern Andes',
    18: 'New Zealand',
    19: 'Antarctic & Subantarctic'
}

REGION_CODES = {
    0: 'GLO', 1: 'ALA', 2: 'WNA', 3: 'ACN', 4: 'ACS',
    5: 'GRL', 6: 'ISL', 7: 'SJM', 8: 'SCA', 9: 'RUA',
    10: 'ASN', 11: 'CEU', 12: 'CAU', 13: 'ASC', 14: 'ASW',
    15: 'ASE', 16: 'TRP', 17: 'SAN', 18: 'NZL', 19: 'ANT'
}

# Method group names
METHOD_GROUPS = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']

# ============================================================
# Data Loading Functions
# ============================================================

def load_calendar_year_results():
    """Load all calendar year result files."""
    cal_dir = os.path.join(DATA_DIR, 'results', 'calendar_years')
    data = {}
    for f in sorted(glob.glob(os.path.join(cal_dir, '*.csv'))):
        fname = os.path.basename(f)
        region_id = int(fname.split('_')[0])
        df = pd.read_csv(f)
        df['region_id'] = region_id
        df['region_name'] = REGION_MAP.get(region_id, fname)
        data[region_id] = df
    return data

def load_hydrological_year_results():
    """Load all hydrological year result files."""
    hyd_dir = os.path.join(DATA_DIR, 'results', 'hydrological_years')
    data = {}
    for f in sorted(glob.glob(os.path.join(hyd_dir, '*.csv'))):
        fname = os.path.basename(f)
        region_id = int(fname.split('_')[0])
        df = pd.read_csv(f)
        df['region_id'] = region_id
        df['region_name'] = REGION_MAP.get(region_id, fname)
        data[region_id] = df
    return data

def catalog_input_data():
    """Catalog all input data files with metadata."""
    input_dir = os.path.join(DATA_DIR, 'input')
    records = []
    for region_dir in sorted(glob.glob(os.path.join(input_dir, '*'))):
        if not os.path.isdir(region_dir):
            continue
        region_name = os.path.basename(region_dir)
        region_id = int(region_name.split('_')[0])
        for f in sorted(glob.glob(os.path.join(region_dir, '*.csv'))):
            fname = os.path.basename(f)
            # Parse method type from filename
            parts = fname.replace('.csv', '').split('_')
            # Method is typically the second or third element after region name
            # Format: region_method_author_et_al.csv
            method = None
            for m in ['altimetry', 'gravimetry', 'demdiff', 'glaciological', 'combined']:
                if m in parts:
                    method = m
                    break
            
            # Read file to get temporal coverage
            try:
                df = pd.read_csv(f)
                n_rows = len(df)
                start_min = df['start_dates'].min() if 'start_dates' in df.columns else None
                end_max = df['end_dates'].max() if 'end_dates' in df.columns else None
                unit = df['unit'].iloc[0] if 'unit' in df.columns else None
                author = df['author'].iloc[0] if 'author' in df.columns else None
            except:
                n_rows = 0
                start_min = end_max = unit = author = None
            
            records.append({
                'region_id': region_id,
                'region_name': REGION_MAP.get(region_id, region_name),
                'filename': fname,
                'method': method,
                'author': author,
                'unit': unit,
                'n_rows': n_rows,
                'start_date': start_min,
                'end_date': end_max
            })
    return pd.DataFrame(records)

# ============================================================
# Analysis Functions
# ============================================================

def compute_annual_time_series(cal_data, target_years=range(2000, 2024)):
    """
    Compute annual time series for each region from calendar year data.
    Returns DataFrames for Gt and m w.e. with uncertainties.
    """
    gt_records = []
    mwe_records = []
    
    for region_id, df in cal_data.items():
        for _, row in df.iterrows():
            year = int(row['start_dates'])
            if year in target_years:
                gt_records.append({
                    'year': year,
                    'region_id': region_id,
                    'region_name': row.get('region_name', REGION_MAP.get(region_id, '')),
                    'glacier_area': row['glacier_area'],
                    'mass_change_gt': row['combined_gt'],
                    'mass_change_gt_err': row['combined_gt_errors'],
                    'mass_change_mwe': row['combined_mwe'],
                    'mass_change_mwe_err': row['combined_mwe_errors']
                })
    
    df_all = pd.DataFrame(gt_records)
    return df_all

def compute_cumulative_mass_change(ts_df):
    """Compute cumulative mass change from annual values."""
    cumul = []
    for region_id in ts_df['region_id'].unique():
        mask = ts_df['region_id'] == region_id
        sub = ts_df[mask].sort_values('year')
        sub = sub.copy()
        sub['cumulative_gt'] = sub['mass_change_gt'].cumsum()
        # Propagate uncertainty: sqrt(sum of squared errors)
        sub['cumulative_gt_err'] = sub['mass_change_gt_err'].cumsum().apply(np.sqrt)
        sub['cumulative_mwe'] = sub['mass_change_mwe'].cumsum()
        sub['cumulative_mwe_err'] = sub['mass_change_mwe_err'].cumsum().apply(np.sqrt)
        cumul.append(sub)
    return pd.concat(cumul, ignore_index=True)

def analyze_method_agreement(hyd_data):
    """
    Analyze agreement between observation methods in hydrological year data.
    For each region and year, compare estimates from different methods.
    """
    agreement_records = []
    
    for region_id, df in hyd_data.items():
        if region_id == 0:  # Skip global
            continue
        for _, row in df.iterrows():
            year = row['start_dates']
            methods_present = {}
            for method in METHOD_GROUPS:
                gt_col = f'{method}_gt'
                err_col = f'{method}_gt_errors'
                if gt_col in df.columns and pd.notna(row.get(gt_col)):
                    methods_present[method] = {
                        'gt': row[gt_col],
                        'gt_err': row.get(err_col, np.nan),
                        'mwe': row.get(f'{method}_mwe', np.nan),
                        'mwe_err': row.get(f'{method}_mwe_errors', np.nan),
                        'annual_var': row.get(f'{method}_annual_variability', np.nan)
                    }
            
            if len(methods_present) >= 2:
                gt_values = [v['gt'] for v in methods_present.values()]
                gt_spread = max(gt_values) - min(gt_values)
                gt_mean = np.mean(gt_values)
                
                agreement_records.append({
                    'region_id': region_id,
                    'region_name': REGION_MAP.get(region_id, ''),
                    'year': year,
                    'n_methods': len(methods_present),
                    'methods': list(methods_present.keys()),
                    'gt_spread': gt_spread,
                    'gt_mean': gt_mean,
                    'relative_spread': gt_spread / abs(gt_mean) if abs(gt_mean) > 0 else np.nan,
                    **{f'{k}_gt': v['gt'] for k, v in methods_present.items()},
                    **{f'{k}_gt_err': v['gt_err'] for k, v in methods_present.items()}
                })
    
    return pd.DataFrame(agreement_records)

def compute_method_coverage(hyd_data):
    """Compute temporal coverage of each method per region."""
    coverage = []
    for region_id, df in hyd_data.items():
        if region_id == 0:
            continue
        for method in METHOD_GROUPS:
            gt_col = f'{method}_gt'
            if gt_col in df.columns:
                valid = df[gt_col].notna()
                n_valid = valid.sum()
                if n_valid > 0:
                    years_present = df.loc[valid, 'start_dates'].values
                    coverage.append({
                        'region_id': region_id,
                        'region_name': REGION_MAP.get(region_id, ''),
                        'method': method,
                        'n_years': n_valid,
                        'first_year': years_present.min(),
                        'last_year': years_present.max(),
                        'annual_var_count': df.loc[valid, f'{method}_annual_variability'].sum() if f'{method}_annual_variability' in df.columns else 0
                    })
    return pd.DataFrame(coverage)

def compute_trends(ts_df, start_year=2000, end_year=2023):
    """Compute linear trends in mass change for each region."""
    trend_records = []
    for region_id in ts_df['region_id'].unique():
        mask = (ts_df['region_id'] == region_id) & (ts_df['year'] >= start_year) & (ts_df['year'] <= end_year)
        sub = ts_df[mask].sort_values('year')
        if len(sub) < 5:
            continue
        
        # Trend in annual mass change (Gt/yr)
        slope_gt, intercept_gt, r_gt, p_gt, se_gt = stats.linregress(sub['year'], sub['mass_change_gt'])
        # Trend in specific mass change (m w.e./yr)
        slope_mwe, intercept_mwe, r_mwe, p_mwe, se_mwe = stats.linregress(sub['year'], sub['mass_change_mwe'])
        
        # Mean mass change rate
        mean_gt = sub['mass_change_gt'].mean()
        mean_mwe = sub['mass_change_mwe'].mean()
        
        # Cumulative mass change
        cumulative_gt = sub['mass_change_gt'].sum()
        
        # Acceleration: fit quadratic
        try:
            coeffs = np.polyfit(sub['year'], sub['mass_change_gt'], 2)
            acceleration = 2 * coeffs[0]  # Gt/yr²
        except:
            acceleration = np.nan
        
        trend_records.append({
            'region_id': region_id,
            'region_name': REGION_MAP.get(region_id, ''),
            'mean_annual_gt': mean_gt,
            'mean_annual_mwe': mean_mwe,
            'cumulative_gt': cumulative_gt,
            'trend_gt_per_yr': slope_gt,
            'trend_gt_pvalue': p_gt,
            'trend_mwe_per_yr': slope_mwe,
            'trend_mwe_pvalue': p_mwe,
            'acceleration_gt_per_yr2': acceleration,
            'n_years': len(sub)
        })
    
    return pd.DataFrame(trend_records)

# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GlaMBIE Glacier Mass Change Reconciliation Analysis")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading calendar year results...")
    cal_data = load_calendar_year_results()
    print(f"  Loaded {len(cal_data)} regions (including global)")
    
    print("\n[2] Loading hydrological year results...")
    hyd_data = load_hydrological_year_results()
    print(f"  Loaded {len(hyd_data)} regions")
    
    print("\n[3] Cataloging input data...")
    input_catalog = catalog_input_data()
    print(f"  Cataloged {len(input_catalog)} input datasets")
    print(f"  Methods distribution:\n{input_catalog['method'].value_counts()}")
    
    # 2. Compute time series
    print("\n[4] Computing annual time series...")
    ts_df = compute_annual_time_series(cal_data)
    ts_df.to_csv(os.path.join(OUTPUT_DIR, 'annual_time_series.csv'), index=False)
    print(f"  Time series: {len(ts_df)} records, {ts_df['region_id'].nunique()} regions")
    
    # 3. Cumulative mass change
    print("\n[5] Computing cumulative mass change...")
    cumul_df = compute_cumulative_mass_change(ts_df)
    cumul_df.to_csv(os.path.join(OUTPUT_DIR, 'cumulative_mass_change.csv'), index=False)
    
    # 4. Method agreement
    print("\n[6] Analyzing method agreement...")
    agreement_df = analyze_method_agreement(hyd_data)
    agreement_df.to_csv(os.path.join(OUTPUT_DIR, 'method_agreement.csv'), index=False)
    print(f"  Agreement analysis: {len(agreement_df)} multi-method comparisons")
    
    # 5. Method coverage
    print("\n[7] Computing method coverage...")
    coverage_df = compute_method_coverage(hyd_data)
    coverage_df.to_csv(os.path.join(OUTPUT_DIR, 'method_coverage.csv'), index=False)
    print(f"  Coverage: {len(coverage_df)} method-region combinations")
    
    # 6. Trends
    print("\n[8] Computing trends...")
    trends_df = compute_trends(ts_df)
    trends_df.to_csv(os.path.join(OUTPUT_DIR, 'regional_trends.csv'), index=False)
    print(f"  Trends computed for {len(trends_df)} regions")
    
    # 7. Summary statistics
    print("\n[9] Computing summary statistics...")
    global_ts = ts_df[ts_df['region_id'] == 0].sort_values('year')
    regional_ts = ts_df[ts_df['region_id'] != 0]
    
    summary = {
        'period': '2000-2023',
        'global_mean_annual_gt': float(global_ts['mass_change_gt'].mean()),
        'global_mean_annual_mwe': float(global_ts['mass_change_mwe'].mean()),
        'global_cumulative_gt': float(global_ts['mass_change_gt'].sum()),
        'global_mean_uncertainty_gt': float(global_ts['mass_change_gt_err'].mean()),
        'n_regions': int(ts_df[ts_df['region_id'] != 0]['region_id'].nunique()),
        'n_input_datasets': int(len(input_catalog)),
        'n_methods_per_region': input_catalog.groupby('region_id')['method'].nunique().to_dict()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Global mean annual mass change: {summary['global_mean_annual_gt']:.1f} Gt/yr")
    print(f"  Global cumulative mass change: {summary['global_cumulative_gt']:.1f} Gt")
    print(f"  Global mean specific mass change: {summary['global_mean_annual_mwe']:.3f} m w.e./yr")
    
    # 8. Regional contribution table
    print("\n[10] Computing regional contributions...")
    regional_summary = regional_ts.groupby(['region_id', 'region_name']).agg(
        mean_annual_gt=('mass_change_gt', 'mean'),
        cumulative_gt=('mass_change_gt', 'sum'),
        mean_annual_mwe=('mass_change_mwe', 'mean'),
        mean_area=('glacier_area', 'mean'),
        mean_uncertainty_gt=('mass_change_gt_err', 'mean')
    ).reset_index()
    regional_summary = regional_summary.sort_values('cumulative_gt')
    regional_summary['pct_of_global'] = 100 * regional_summary['cumulative_gt'] / summary['global_cumulative_gt']
    regional_summary.to_csv(os.path.join(OUTPUT_DIR, 'regional_summary.csv'), index=False)
    
    print("\n  Top 5 contributors to global mass loss:")
    for _, row in regional_summary.head(5).iterrows():
        print(f"    {row['region_name']}: {row['cumulative_gt']:.0f} Gt ({row['pct_of_global']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Data processing complete. Proceeding to visualization...")
    print("=" * 60)
