#!/usr/bin/env python3
"""
GlaMBIE Analysis: Global Glacial Mass Change Time Series (2000-2023)

This script loads GlaMBIE result datasets, computes cumulative mass change
time series, performs regional/global aggregation, generates figures, and
exports intermediate results for the research report.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'glambie', 'results', 'calendar_years')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
IMAGES_DIR = os.path.join(BASE_DIR, 'report', 'images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Region name mapping
REGION_NAMES = {
    '1_alaska': 'Alaska',
    '2_western_canada_us': 'W. Canada & US',
    '3_arctic_canada_north': 'Arctic Canada (N)',
    '4_arctic_canada_south': 'Arctic Canada (S)',
    '5_greenland_periphery': 'Greenland Periph.',
    '6_iceland': 'Iceland',
    '7_svalbard': 'Svalbard',
    '8_scandinavia': 'Scandinavia',
    '9_russian_arctic': 'Russian Arctic',
    '10_north_asia': 'North Asia',
    '11_central_europe': 'Central Europe',
    '12_caucasus_middle_east': 'Caucasus & Middle East',
    '13_central_asia': 'Central Asia',
    '14_south_asia_west': 'South Asia (W)',
    '15_south_asia_east': 'South Asia (E)',
    '16_low_latitudes': 'Low Latitudes',
    '17_southern_andes': 'Southern Andes',
    '18_new_zealand': 'New Zealand',
    '19_antarctic_and_subantarctic': 'Antarctic & Subantarctic',
}

# ============================================================
# 1. Load Data
# ============================================================
print("Loading regional data...")

regional_data = {}
region_files = sorted(glob.glob(os.path.join(DATA_DIR, '[0-9]*.csv')))

for fpath in region_files:
    fname = os.path.basename(fpath).replace('.csv', '')
    if fname == '0_global':
        continue
    df = pd.read_csv(fpath)
    df['year'] = df['start_dates'].astype(int)
    df['region_key'] = fname
    regional_data[fname] = df
    print(f"  Loaded {REGION_NAMES.get(fname, fname)}: {len(df)} years, "
          f"area={df['glacier_area'].iloc[-1]:,.0f} km²")

# Load global data
global_file = os.path.join(DATA_DIR, '0_global.csv')
global_df = pd.read_csv(global_file)
global_df['year'] = global_df['start_dates'].astype(int)
print(f"\nLoaded global data: {len(global_df)} years")

# ============================================================
# 2. Compute Cumulative Time Series
# ============================================================
print("\nComputing cumulative mass change...")

# For each region, compute cumulative sum from first year
cumulative_results = {}
for rkey, df in regional_data.items():
    cum_mwe = df['combined_mwe'].cumsum()
    cum_mwe_err = np.sqrt((df['combined_mwe_errors']**2).cumsum())
    cum_gt = df['combined_gt'].cumsum()
    cum_gt_err = np.sqrt((df['combined_gt_errors']**2).cumsum())
    
    cumulative_results[rkey] = {
        'years': df['year'].values,
        'cum_mwe': cum_mwe.values,
        'cum_mwe_err': cum_mwe_err.values,
        'cum_gt': cum_gt.values,
        'cum_gt_err': cum_gt_err.values,
        'annual_mwe': df['combined_mwe'].values,
        'annual_mwe_err': df['combined_mwe_errors'].values,
        'annual_gt': df['combined_gt'].values,
        'annual_gt_err': df['combined_gt_errors'].values,
        'area_km2': df['glacier_area'].values,
    }

# Global cumulative
global_cum = {
    'years': global_df['year'].values,
    'cum_mwe': global_df['combined_mwe'].cumsum().values,
    'cum_mwe_err': np.sqrt((global_df['combined_mwe_errors']**2).cumsum()).values,
    'cum_gt': global_df['combined_gt'].cumsum().values,
    'cum_gt_err': np.sqrt((global_df['combined_gt_errors']**2).cumsum()).values,
    'annual_mwe': global_df['combined_mwe'].values,
    'annual_mwe_err': global_df['combined_mwe_errors'].values,
    'annual_gt': global_df['combined_gt'].values,
    'annual_gt_err': global_df['combined_gt_errors'].values,
}

# ============================================================
# 3. Export Intermediate Results
# ============================================================
print("Exporting intermediate results...")

# Global time series CSV
global_ts = pd.DataFrame({
    'year': global_cum['years'],
    'annual_mass_change_mwe': global_cum['annual_mwe'],
    'annual_mass_change_mwe_error': global_cum['annual_mwe_err'],
    'cumulative_mass_change_mwe': global_cum['cum_mwe'],
    'cumulative_mass_change_mwe_error': global_cum['cum_mwe_err'],
    'annual_mass_change_gt': global_cum['annual_gt'],
    'annual_mass_change_gt_error': global_cum['annual_gt_err'],
    'cumulative_mass_change_gt': global_cum['cum_gt'],
    'cumulative_mass_change_gt_error': global_cum['cum_gt_err'],
})
global_ts.to_csv(os.path.join(OUTPUTS_DIR, 'global_time_series.csv'), index=False)

# Regional summary
regional_summary = []
for rkey in sorted(cumulative_results.keys()):
    res = cumulative_results[rkey]
    regional_summary.append({
        'region': REGION_NAMES.get(rkey, rkey),
        'region_key': rkey,
        'total_cumulative_mwe_2000_2023': res['cum_mwe'][-1],
        'total_cumulative_mwe_error': res['cum_mwe_err'][-1],
        'total_cumulative_gt_2000_2023': res['cum_gt'][-1],
        'total_cumulative_gt_error': res['cum_gt_err'][-1],
        'mean_annual_mwe': np.mean(res['annual_mwe']),
        'std_annual_mwe': np.std(res['annual_mwe']),
        'latest_area_km2': res['area_km2'][-1],
    })

regional_summary_df = pd.DataFrame(regional_summary)
regional_summary_df.to_csv(os.path.join(OUTPUTS_DIR, 'regional_summary.csv'), index=False)

# Save cumulative data per region
for rkey, res in cumulative_results.items():
    rdf = pd.DataFrame({
        'year': res['years'],
        'annual_mwe': res['annual_mwe'],
        'annual_mwe_error': res['annual_mwe_err'],
        'cumulative_mwe': res['cum_mwe'],
        'cumulative_mwe_error': res['cum_mwe_err'],
        'annual_gt': res['annual_gt'],
        'annual_gt_error': res['annual_gt_err'],
        'cumulative_gt': res['cum_gt'],
        'cumulative_gt_error': res['cum_gt_err'],
    })
    rdf.to_csv(os.path.join(OUTPUTS_DIR, f'cumulative_{rkey}.csv'), index=False)

print(f"Exported {len(regional_summary)} regional summaries + global time series")

# ============================================================
# 4. Key Statistics
# ============================================================
print("\n=== Key Statistics ===")
print(f"Global cumulative mass change 2000-2023: {global_cum['cum_gt'][-1]:.1f} ± {global_cum['cum_gt_err'][-1]:.1f} Gt")
print(f"Global cumulative mass change 2000-2023: {global_cum['cum_mwe'][-1]:.3f} ± {global_cum['cum_mwe_err'][-1]:.3f} m w.e.")
print(f"Mean annual global mass change: {np.mean(global_cum['annual_gt']):.1f} ± {np.std(global_cum['annual_gt']):.1f} Gt/yr")
print(f"Mean annual global mass change: {np.mean(global_cum['annual_mwe']):.3f} ± {np.std(global_cum['annual_mwe']):.3f} m w.e./yr")

# Top contributors by absolute mass loss
sorted_regions = sorted(cumulative_results.items(), key=lambda x: x[1]['cum_gt'][-1])
print("\nRegional cumulative mass change (Gt, most negative first):")
for rkey, res in sorted_regions[:10]:
    print(f"  {REGION_NAMES.get(rkey, rkey):30s}: {res['cum_gt'][-1]:8.1f} ± {res['cum_gt_err'][-1]:.1f} Gt")

# ============================================================
# 5. Generate Figures
# ============================================================
print("\nGenerating figures...")

# --- Figure 1: Global cumulative mass change (dual axis) ---
fig, ax1 = plt.subplots(figsize=(10, 5))
years = global_cum['years']

color1 = '#1f77b4'
ax1.fill_between(years, 
                  global_cum['cum_mwe'] - global_cum['cum_mwe_err'],
                  global_cum['cum_mwe'] + global_cum['cum_mwe_err'],
                  alpha=0.2, color=color1, label='Uncertainty (±1σ)')
ax1.plot(years, global_cum['cum_mwe'], color=color1, linewidth=2.5, 
         marker='o', markersize=4, label='Cumulative specific mass change')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Cumulative Specific Mass Change (m w.e.)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = '#d62728'
ax2.fill_between(years,
                  global_cum['cum_gt'] - global_cum['cum_gt_err'],
                  global_cum['cum_gt'] + global_cum['cum_gt_err'],
                  alpha=0.15, color=color2)
ax2.plot(years, global_cum['cum_gt'], color=color2, linewidth=2,
         marker='s', markersize=4, label='Cumulative total mass change')
ax2.set_ylabel('Cumulative Total Mass Change (Gt)', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.title('Global Glacier Mass Change 2000–2023 (GlaMBIE Combined Estimate)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'global_cumulative_mass_change.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: global_cumulative_mass_change.png")

# --- Figure 2: Annual mass change bar chart ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
years = global_cum['years']

colors = ['#2ca02c' if v >= 0 else '#d62728' for v in global_cum['annual_gt']]
ax1.bar(years, global_cum['annual_gt'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.errorbar(years, global_cum['annual_gt'], yerr=global_cum['annual_gt_err'], 
             fmt='none', color='black', capsize=3, linewidth=0.8)
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_ylabel('Annual Mass Change (Gt)', fontsize=11)
ax1.set_title('Global Annual Glacier Mass Change 2000–2023', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

colors2 = ['#2ca02c' if v >= 0 else '#d62728' for v in global_cum['annual_mwe']]
ax2.bar(years, global_cum['annual_mwe'], color=colors2, edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.errorbar(years, global_cum['annual_mwe'], yerr=global_cum['annual_mwe_err'],
             fmt='none', color='black', capsize=3, linewidth=0.8)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Annual Specific Mass Change (m w.e.)', fontsize=11)
ax2.set_xlabel('Year', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'global_annual_mass_change.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: global_annual_mass_change.png")

# --- Figure 3: Regional cumulative mass change comparison ---
fig, ax = plt.subplots(figsize=(12, 8))

sorted_by_loss = sorted(cumulative_results.items(), key=lambda x: x[1]['cum_gt'][-1])
region_labels = [REGION_NAMES.get(k, k) for k, _ in sorted_by_loss]
cum_values = [v['cum_gt'][-1] for _, v in sorted_by_loss]
cum_errors = [v['cum_gt_err'][-1] for _, v in sorted_by_loss]

y_pos = np.arange(len(region_labels))
colors_bar = ['#d62728' if v < 0 else '#2ca02c' for v in cum_values]

bars = ax.barh(y_pos, cum_values, xerr=cum_errors, color=colors_bar, 
               edgecolor='black', linewidth=0.5, alpha=0.8, height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(region_labels, fontsize=9)
ax.set_xlabel('Cumulative Mass Change 2000–2023 (Gt)', fontsize=11)
ax.set_title('Regional Glacier Mass Change Comparison', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'regional_cumulative_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: regional_cumulative_comparison.png")

# --- Figure 4: Regional cumulative mass change time series (small multiples) ---
fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharex=True)
axes = axes.flatten()

sorted_keys = sorted(cumulative_results.keys())
for i, rkey in enumerate(sorted_keys):
    ax = axes[i]
    res = cumulative_results[rkey]
    ax.fill_between(res['years'],
                     res['cum_mwe'] - res['cum_mwe_err'],
                     res['cum_mwe'] + res['cum_mwe_err'],
                     alpha=0.2, color='#1f77b4')
    ax.plot(res['years'], res['cum_mwe'], color='#1f77b4', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.3)
    ax.set_title(REGION_NAMES.get(rkey, rkey), fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Regional Cumulative Specific Mass Change Time Series (m w.e.)', 
             fontsize=13, fontweight='bold', y=0.98)
fig.text(0.5, 0.02, 'Year', ha='center', fontsize=11)
fig.text(0.02, 0.5, 'Cumulative Mass Change (m w.e.)', va='center', rotation='vertical', fontsize=11)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
plt.savefig(os.path.join(IMAGES_DIR, 'regional_time_series.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: regional_time_series.png")

# --- Figure 5: Uncertainty analysis - annual variability ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: uncertainty over time (global)
ax1.plot(years, global_cum['annual_gt_err'], color='#ff7f0e', linewidth=2, marker='o', markersize=4)
ax1.fill_between(years, global_cum['annual_gt_err'], alpha=0.2, color='#ff7f0e')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Uncertainty (Gt)', fontsize=11)
ax1.set_title('Global Annual Mass Change Uncertainty Over Time', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Right: uncertainty by region (latest year)
region_unc = []
region_names_short = []
for rkey in sorted(cumulative_results.keys()):
    res = cumulative_results[rkey]
    region_unc.append(res['cum_gt_err'][-1])
    region_names_short.append(REGION_NAMES.get(rkey, rkey)[:15])

ax2.barh(range(len(region_unc)), region_unc, color='#9467bd', alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(region_unc)))
ax2.set_yticklabels(region_names_short, fontsize=8)
ax2.set_xlabel('Cumulative Uncertainty (Gt, ±1σ)', fontsize=11)
ax2.set_title('Regional Cumulative Uncertainty (2000–2023)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'uncertainty_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: uncertainty_analysis.png")

# --- Figure 6: Rate of mass change (acceleration analysis) ---
fig, ax = plt.subplots(figsize=(10, 5))

# Compute 5-year moving average of annual mass change
window = 5
annual_gt = global_cum['annual_gt']
years_arr = global_cum['years']
ma_gt = pd.Series(annual_gt).rolling(window=window, center=True).mean()
ma_err = pd.Series(global_cum['annual_gt_err']).rolling(window=window, center=True).mean()

ax.plot(years_arr, annual_gt, color='lightgray', linewidth=1, label='Annual')
ax.plot(years_arr, ma_gt, color='#d62728', linewidth=2.5, marker='o', markersize=5,
        label=f'{window}-year Moving Average')
ax.fill_between(years_arr, ma_gt - ma_err, ma_gt + ma_err, alpha=0.2, color='#d62728')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Mass Change (Gt/yr)', fontsize=12)
ax.set_title('Global Glacier Mass Loss Rate with Trend', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'mass_change_rate_trend.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: mass_change_rate_trend.png")

# --- Figure 7: Method comparison (using hydrological year data) ---
print("  Loading hydrological year data for method comparison...")

hydro_dir = os.path.join(os.path.dirname(DATA_DIR), 'hydrological_years')
method_comparison = {}

for fpath in sorted(glob.glob(os.path.join(hydro_dir, '[0-9]*.csv'))):
    fname = os.path.basename(fpath).replace('.csv', '')
    if fname == '0_global':
        continue
    df = pd.read_csv(fpath)
    
    # Extract available methods
    methods = {}
    for prefix in ['altimetry', 'gravimetry', 'demdiff_and_glaciological']:
        gt_col = f'{prefix}_gt'
        if gt_col in df.columns and df[gt_col].notna().any():
            methods[prefix] = {
                'years': df['start_dates'].values,
                'gt': df[gt_col].values,
                'gt_err': df[f'{prefix}_gt_errors'].values,
                'mwe': df[f'{prefix}_mwe'].values,
                'mwe_err': df[f'{prefix}_mwe_errors'].values,
            }
    
    if methods:
        method_comparison[fname] = {
            'combined_gt': df['combined_gt'].values,
            'combined_gt_err': df['combined_gt_errors'].values,
            'methods': methods,
        }

# Plot method comparison for selected regions with multiple methods
regions_with_methods = [k for k, v in method_comparison.items() if len(v['methods']) >= 2]
if regions_with_methods:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    method_colors = {
        'altimetry': '#2ca02c',
        'gravimetry': '#ff7f0e',
        'demdiff_and_glaciological': '#9467bd',
    }
    method_labels = {
        'altimetry': 'Satellite Altimetry',
        'gravimetry': 'Gravimetry (GRACE)',
        'demdiff_and_glaciological': 'DEM Diff. + Glaciological',
    }
    
    for idx, rkey in enumerate(regions_with_methods[:6]):
        ax = axes[idx]
        data = method_comparison[rkey]
        
        # Plot combined
        ax.plot(data['methods'][list(data['methods'].keys())[0]]['years'],
                data['combined_gt'], color='black', linewidth=2, 
                label='Combined', zorder=5)
        
        for mname, mdata in data['methods'].items():
            ax.plot(mdata['years'], mdata['gt'], 
                    color=method_colors.get(mname, 'gray'),
                    linewidth=1.5, alpha=0.8,
                    label=method_labels.get(mname, mname))
        
        ax.set_title(REGION_NAMES.get(rkey, rkey), fontsize=10, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Method Comparison: Regional Mass Change by Observation Type', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'method_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: method_comparison.png")

# --- Figure 8: Sea level contribution ---
fig, ax = plt.subplots(figsize=(10, 5))

# Convert Gt to mm SLE: 1 Gt ice ≈ 0.00273 mm SLE (density correction ~0.917)
# Standard conversion: 361.8 Gt = 1 mm SLE
sle_per_gt = 1.0 / 361.8
cum_sle = global_cum['cum_gt'] * sle_per_gt
cum_sle_err = global_cum['cum_gt_err'] * sle_per_gt

ax.fill_between(years, cum_sle - cum_sle_err, cum_sle + cum_sle_err,
                alpha=0.2, color='#17becf')
ax.plot(years, cum_sle, color='#17becf', linewidth=2.5, marker='o', markersize=4)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Sea Level Equivalent (mm)', fontsize=12)
ax.set_title('Global Glacier Contribution to Sea Level Rise (2000–2023)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'sea_level_contribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: sea_level_contribution.png")

# ============================================================
# 6. Save Metadata
# ============================================================
metadata = {
    'dataset': 'GlaMBIE (Glacier Mass Balance Intercomparison Exercise)',
    'doi': '10.5904/wgms-glambie-2024-07',
    'time_period': '2000-2023',
    'temporal_resolution': 'Annual (calendar years)',
    'regions': 19,
    'observation_methods': ['glaciological', 'DEM differencing', 'satellite altimetry', 'gravimetry'],
    'global_cumulative_gt_2000_2023': float(global_cum['cum_gt'][-1]),
    'global_cumulative_gt_error_2000_2023': float(global_cum['cum_gt_err'][-1]),
    'global_cumulative_mwe_2000_2023': float(global_cum['cum_mwe'][-1]),
    'global_cumulative_mwe_error_2000_2023': float(global_cum['cum_mwe_err'][-1]),
    'mean_annual_gt': float(np.mean(global_cum['annual_gt'])),
    'std_annual_gt': float(np.std(global_cum['annual_gt'])),
    'sea_level_equivalent_mm': float(cum_sle[-1]),
    'sea_level_equivalent_error_mm': float(cum_sle_err[-1]),
}

with open(os.path.join(OUTPUTS_DIR, 'analysis_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nAnalysis complete. Metadata saved to outputs/analysis_metadata.json")
print(f"Total figures generated: 8")
