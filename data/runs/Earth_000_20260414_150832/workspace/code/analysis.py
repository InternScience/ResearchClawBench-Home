"""
GlaMBIE Global Glacier Mass Change Analysis
=============================================
Reconcile diverse observational methods to deliver a consistent assessment
of global glacial mass change (2000-2023).
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'data/glambie'
OUTPUT_DIR = 'outputs'
REPORT_IMG_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# Region mapping
REGION_NAMES = {
    '0_global': 'Global',
    '1_alaska': 'Alaska',
    '2_western_canada_us': 'Western Canada & US',
    '3_arctic_canada_north': 'Arctic Canada North',
    '4_arctic_canada_south': 'Arctic Canada South',
    '5_greenland_periphery': 'Greenland Periphery',
    '6_iceland': 'Iceland',
    '7_svalbard': 'Svalbard',
    '8_scandinavia': 'Scandinavia',
    '9_russian_arctic': 'Russian Arctic',
    '10_north_asia': 'North Asia',
    '11_central_europe': 'Central Europe',
    '12_caucasus_middle_east': 'Caucasus & Middle East',
    '13_central_asia': 'Central Asia',
    '14_south_asia_west': 'South Asia West',
    '15_south_asia_east': 'South Asia East',
    '16_low_latitudes': 'Low Latitudes',
    '17_southern_andes': 'Southern Andes',
    '18_new_zealand': 'New Zealand',
    '19_antarctic_and_subantarctic': 'Antarctic & Subantarctic'
}

# ============================================================
# 1. DATA OVERVIEW - Count input datasets by region and method
# ============================================================
print("=" * 60)
print("STEP 1: Data Overview")
print("=" * 60)

input_dir = os.path.join(DATA_DIR, 'input')
dataset_inventory = []

for region_folder in sorted(os.listdir(input_dir)):
    region_path = os.path.join(input_dir, region_folder)
    if not os.path.isdir(region_path):
        continue
    for csv_file in glob.glob(os.path.join(region_path, '*.csv')):
        fname = os.path.basename(csv_file)
        parts = fname.replace('.csv', '').split('_')
        # Determine method from filename
        method_keywords = ['altimetry', 'demdiff', 'glaciological', 'gravimetry', 'combined']
        method = 'unknown'
        for kw in method_keywords:
            if kw in fname.lower():
                method = kw
                break
        region_name = REGION_NAMES.get(region_folder, region_folder)
        df = pd.read_csv(csv_file)
        dataset_inventory.append({
            'region_folder': region_folder,
            'region_name': region_name,
            'method': method,
            'filename': fname,
            'n_records': len(df),
            'time_start': df['start_dates'].min(),
            'time_end': df['end_dates'].max(),
            'unit': df['unit'].iloc[0] if 'unit' in df.columns else 'unknown'
        })

inventory_df = pd.DataFrame(dataset_inventory)
print(f"Total input datasets: {len(inventory_df)}")
print(f"\nDatasets by method:")
print(inventory_df['method'].value_counts().to_string())
print(f"\nDatasets by region:")
print(inventory_df['region_name'].value_counts().to_string())

# Save inventory
inventory_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_inventory.csv'), index=False)

# ============================================================
# 2. LOAD RESULT TIME SERIES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Load Result Time Series")
print("=" * 60)

cal_results = {}
hydro_results = {}

for csv_file in sorted(glob.glob(os.path.join(DATA_DIR, 'results', 'calendar_years', '*.csv'))):
    region_key = os.path.basename(csv_file).replace('.csv', '')
    df = pd.read_csv(csv_file)
    cal_results[region_key] = df
    print(f"  Calendar: {REGION_NAMES.get(region_key, region_key):30s} -> {len(df)} records, "
          f"years {df['start_dates'].min():.0f}-{df['end_dates'].max():.0f}")

for csv_file in sorted(glob.glob(os.path.join(DATA_DIR, 'results', 'hydrological_years', '*.csv'))):
    region_key = os.path.basename(csv_file).replace('.csv', '')
    df = pd.read_csv(csv_file)
    hydro_results[region_key] = df

# ============================================================
# 3. FIGURE 1: Data Overview
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Generate Figure 1 - Data Overview")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel A: Dataset counts by region and method
method_colors = {
    'altimetry': '#2196F3',
    'demdiff': '#4CAF50',
    'glaciological': '#FF9800',
    'gravimetry': '#9C27B0',
    'combined': '#F44336',
    'unknown': '#9E9E9E'
}

regions_sorted = inventory_df.groupby('region_name')['filename'].count().sort_values(ascending=True).index.tolist()
method_order = ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']

bar_bottom = np.zeros(len(regions_sorted))
for method in method_order:
    counts = []
    for r in regions_sorted:
        c = len(inventory_df[(inventory_df['region_name'] == r) & (inventory_df['method'] == method)])
        counts.append(c)
    axes[0].barh(regions_sorted, counts, left=bar_bottom, label=method.capitalize(),
                 color=method_colors.get(method, '#9E9E9E'), edgecolor='white', linewidth=0.5)
    bar_bottom += np.array(counts)

axes[0].set_xlabel('Number of Datasets', fontsize=12)
axes[0].set_title('A) Input Datasets by Region and Method', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_xlim(0, max(bar_bottom) + 2)

# Panel B: Time coverage heatmap
time_matrix = np.zeros((len(regions_sorted), 24))  # 2000-2023
for i, region in enumerate(regions_sorted):
    region_data = inventory_df[inventory_df['region_name'] == region]
    for _, row in region_data.iterrows():
        start = max(2000, int(np.floor(row['time_start'])))
        end = min(2023, int(np.floor(row['time_end'])))
        for y in range(start, end + 1):
            if 2000 <= y <= 2023:
                time_matrix[i, y - 2000] += 1

im = axes[1].imshow(time_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                     extent=[1999.5, 2023.5, -0.5, len(regions_sorted) - 0.5], origin='lower')
axes[1].set_yticks(range(len(regions_sorted)))
axes[1].set_yticklabels(regions_sorted, fontsize=8)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_title('B) Temporal Data Coverage (# datasets)', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(2000, 2024, 2))
plt.colorbar(im, ax=axes[1], label='Number of datasets', shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig1_data_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_data_overview.png")

# ============================================================
# 4. FIGURE 2: Global Mass Change Time Series
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Generate Figure 2 - Global Mass Change Time Series")
print("=" * 60)

global_df = cal_results['0_global'].copy()
global_df = global_df[(global_df['start_dates'] >= 2000) & (global_df['end_dates'] <= 2024)]

# Compute cumulative mass change
global_df['cumulative_gt'] = global_df['combined_gt'].cumsum()
global_df['cumulative_gt_upper'] = (global_df['combined_gt'] + global_df['combined_gt_errors']).cumsum()
global_df['cumulative_gt_lower'] = (global_df['combined_gt'] - global_df['combined_gt_errors']).cumsum()

# Also compute cumulative mwe
global_df['cumulative_mwe'] = global_df['combined_mwe'].cumsum()
global_df['cumulative_mwe_upper'] = (global_df['combined_mwe'] + global_df['combined_mwe_errors']).cumsum()
global_df['cumulative_mwe_lower'] = (global_df['combined_mwe'] - global_df['combined_mwe_errors']).cumsum()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel A: Annual mass change in Gt
years = global_df['start_dates'].values
axes[0].bar(years, global_df['combined_gt'], width=0.8, color='#2196F3', alpha=0.7, label='Annual change')
axes[0].errorbar(years, global_df['combined_gt'], yerr=global_df['combined_gt_errors'],
                 fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='-')
axes[0].set_ylabel('Mass Change (Gt yr⁻¹)', fontsize=12)
axes[0].set_title('A) Global Annual Glacier Mass Change', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower left', fontsize=10)

# Add running mean
window = 5
if len(global_df) >= window:
    running_mean = global_df['combined_gt'].rolling(window=window, center=True).mean()
    axes[0].plot(years, running_mean, color='red', linewidth=2, label=f'{window}-yr running mean')
    axes[0].legend(loc='lower left', fontsize=10)

# Panel B: Cumulative mass change in Gt
axes[1].fill_between(years, global_df['cumulative_gt_lower'], global_df['cumulative_gt_upper'],
                      alpha=0.3, color='#2196F3', label='±1σ uncertainty')
axes[1].plot(years, global_df['cumulative_gt'], color='#2196F3', linewidth=2, label='Cumulative change')
axes[1].axhline(y=0, color='black', linewidth=0.5, linestyle='-')
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Cumulative Mass Change (Gt)', fontsize=12)
axes[1].set_title('B) Cumulative Global Glacier Mass Change (2000–2023)', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig2_global_time_series.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_global_time_series.png")

# Print summary statistics
total_change_gt = global_df['combined_gt'].sum()
total_error_gt = np.sqrt((global_df['combined_gt_errors']**2).sum())
mean_annual_gt = global_df['combined_gt'].mean()
mean_annual_mwe = global_df['combined_mwe'].mean()
print(f"\n  Global Summary (2000-2023):")
print(f"    Total mass change: {total_change_gt:.1f} ± {total_error_gt:.1f} Gt")
print(f"    Mean annual change: {mean_annual_gt:.1f} Gt/yr")
print(f"    Mean specific change: {mean_annual_mwe:.3f} m w.e./yr")
print(f"    Final cumulative: {global_df['cumulative_gt'].iloc[-1]:.1f} Gt")

# ============================================================
# 5. FIGURE 3: Regional Mass Change Comparison
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Generate Figure 3 - Regional Mass Change")
print("=" * 60)

regional_summary = []
for region_key, region_df in cal_results.items():
    if region_key == '0_global':
        continue
    region_name = REGION_NAMES.get(region_key, region_key)
    rdf = region_df[(region_df['start_dates'] >= 2000) & (region_df['end_dates'] <= 2024)]
    if len(rdf) == 0:
        continue
    total_gt = rdf['combined_gt'].sum()
    total_err = np.sqrt((rdf['combined_gt_errors']**2).sum())
    mean_mwe = rdf['combined_mwe'].mean()
    mean_err_mwe = np.sqrt((rdf['combined_mwe_errors']**2).sum()) / len(rdf)
    regional_summary.append({
        'region_key': region_key,
        'region_name': region_name,
        'total_gt': total_gt,
        'total_gt_err': total_err,
        'mean_annual_mwe': mean_mwe,
        'mean_annual_mwe_err': mean_err_mwe,
        'n_years': len(rdf),
        'mean_area_km2': rdf['glacier_area'].mean()
    })

regional_df = pd.DataFrame(regional_summary)
regional_df = regional_df.sort_values('total_gt')

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Panel A: Total mass change by region (Gt)
colors = ['#d32f2f' if x < 0 else '#388e3c' for x in regional_df['total_gt']]
axes[0].barh(regional_df['region_name'], regional_df['total_gt'], xerr=regional_df['total_gt_err'],
             color=colors, alpha=0.8, capsize=3, error_kw={'linewidth': 1})
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].set_xlabel('Total Mass Change 2000-2023 (Gt)', fontsize=12)
axes[0].set_title('A) Regional Total Mass Change (Gt)', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()

# Panel B: Mean annual specific mass change (m w.e.)
regional_df_sorted_mwe = regional_df.sort_values('mean_annual_mwe')
colors_mwe = ['#d32f2f' if x < 0 else '#388e3c' for x in regional_df_sorted_mwe['mean_annual_mwe']]
axes[1].barh(regional_df_sorted_mwe['region_name'], regional_df_sorted_mwe['mean_annual_mwe'],
             xerr=regional_df_sorted_mwe['mean_annual_mwe_err'],
             color=colors_mwe, alpha=0.8, capsize=3, error_kw={'linewidth': 1})
axes[1].axvline(x=0, color='black', linewidth=0.5)
axes[1].set_xlabel('Mean Annual Specific Mass Change (m w.e. yr⁻¹)', fontsize=12)
axes[1].set_title('B) Regional Mean Annual Specific Mass Change', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig3_regional_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_regional_comparison.png")

# Save regional summary
regional_df.to_csv(os.path.join(OUTPUT_DIR, 'regional_summary.csv'), index=False)

# ============================================================
# 6. FIGURE 4: Method Comparison from Input Data
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Generate Figure 4 - Method Comparison")
print("=" * 60)

# Load input data for key regions and compare methods
def load_input_data(region_folder):
    """Load all input datasets for a region, standardize to Gt."""
    region_path = os.path.join(input_dir, region_folder)
    datasets = []
    for csv_file in glob.glob(os.path.join(region_path, '*.csv')):
        fname = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        
        # Determine method
        method = 'unknown'
        for kw in ['altimetry', 'demdiff', 'glaciological', 'gravimetry', 'combined']:
            if kw in fname.lower():
                method = kw
                break
        
        # Get region area from results for conversion
        region_key = region_folder
        if region_key in cal_results:
            area = cal_results[region_key]['glacier_area'].mean()
        else:
            area = 1.0
        
        # Convert to Gt if needed
        if 'unit' in df.columns:
            unit = df['unit'].iloc[0].lower()
            if unit == 'm' or unit == 'mwe':
                df['changes_gt'] = df['changes'] * area / 1000  # m w.e. * km2 / 1000 = Gt
                df['errors_gt'] = df['errors'] * area / 1000
            elif unit == 'gt':
                df['changes_gt'] = df['changes']
                df['errors_gt'] = df['errors']
            else:
                continue
        else:
            continue
        
        df['method'] = method
        df['source'] = fname
        datasets.append(df)
    
    return datasets

# Compare methods for major regions
comparison_regions = ['1_alaska', '5_greenland_periphery', '7_svalbard', 
                      '13_central_asia', '17_southern_andes', '19_antarctic_and_subantarctic']

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes_flat = axes.flatten()

for idx, region_key in enumerate(comparison_regions):
    if idx >= 6:
        break
    ax = axes_flat[idx]
    region_name = REGION_NAMES.get(region_key, region_key)
    
    datasets = load_input_data(region_key)
    
    for ds in datasets:
        # Filter to 2000-2023
        mask = (ds['start_dates'] >= 2000) & (ds['end_dates'] <= 2024)
        ds_filt = ds[mask]
        if len(ds_filt) == 0:
            continue
        
        method = ds_filt['method'].iloc[0]
        color = method_colors.get(method, '#9E9E9E')
        
        # Plot as scatter with connecting lines
        mid_years = (ds_filt['start_dates'] + ds_filt['end_dates']) / 2
        ax.plot(mid_years, ds_filt['changes_gt'], 'o-', color=color, alpha=0.6, 
                markersize=3, linewidth=0.8, label=method.capitalize())
    
    # Also plot the combined result
    if region_key in cal_results:
        rdf = cal_results[region_key]
        rdf_filt = rdf[(rdf['start_dates'] >= 2000) & (rdf['end_dates'] <= 2024)]
        mid_yr = (rdf_filt['start_dates'] + rdf_filt['end_dates']) / 2
        ax.plot(mid_yr, rdf_filt['combined_gt'], 'k-', linewidth=2, label='GlaMBIE combined', zorder=10)
        ax.fill_between(mid_yr, 
                        rdf_filt['combined_gt'] - rdf_filt['combined_gt_errors'],
                        rdf_filt['combined_gt'] + rdf_filt['combined_gt_errors'],
                        color='gray', alpha=0.2)
    
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title(region_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mass Change (Gt yr⁻¹)', fontsize=10)
    if idx >= 4:
        ax.set_xlabel('Year', fontsize=10)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc='lower left')

plt.suptitle('Method Comparison: Individual Estimates vs GlaMBIE Combined', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig4_method_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_method_comparison.png")

# ============================================================
# 7. FIGURE 5: Cumulative Regional Contributions
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Generate Figure 5 - Cumulative Regional Contributions")
print("=" * 60)

# Top contributors to global mass loss
top_regions = regional_df.nsmallest(8, 'total_gt')['region_key'].tolist()

fig, ax = plt.subplots(figsize=(14, 8))

# Stack plot of cumulative mass change for top regions
cumulative_data = {}
for region_key in top_regions:
    if region_key in cal_results:
        rdf = cal_results[region_key]
        rdf = rdf[(rdf['start_dates'] >= 2000) & (rdf['end_dates'] <= 2024)]
        cumulative_data[region_key] = rdf.set_index('start_dates')['combined_gt']

cum_df = pd.DataFrame(cumulative_data)
cum_df = cum_df.sort_index()
cum_df = cum_df.fillna(0)

# Cumulative sum
cum_cum = cum_df.cumsum()

cmap = plt.cm.Set3
colors = [cmap(i / len(top_regions)) for i in range(len(top_regions))]

ax.stackplot(cum_cum.index, [cum_cum[col].values for col in cum_cum.columns],
             labels=[REGION_NAMES.get(k, k) for k in cum_cum.columns],
             colors=colors, alpha=0.8)

# Add global total on top
if '0_global' in cal_results:
    gdf = cal_results['0_global']
    gdf = gdf[(gdf['start_dates'] >= 2000) & (gdf['end_dates'] <= 2024)]
    global_cum = gdf.set_index('start_dates')['combined_gt'].cumsum()
    ax.plot(global_cum.index, global_cum.values, 'k-', linewidth=2.5, label='Global total')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative Mass Change (Gt)', fontsize=12)
ax.set_title('Cumulative Regional Glacier Mass Change Contributions (2000–2023)', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig5_cumulative_regional.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_cumulative_regional.png")

# ============================================================
# 8. FIGURE 6: Uncertainty Analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Generate Figure 6 - Uncertainty Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Relative uncertainty over time
if '0_global' in cal_results:
    gdf = cal_results['0_global']
    gdf = gdf[(gdf['start_dates'] >= 2000) & (gdf['end_dates'] <= 2024)]
    
    rel_unc = np.abs(gdf['combined_gt_errors'] / gdf['combined_gt']) * 100
    axes[0].plot(gdf['start_dates'], rel_unc, 'o-', color='#F44336', markersize=4, linewidth=1.5)
    axes[0].set_ylabel('Relative Uncertainty (%)', fontsize=12)
    axes[0].set_title('A) Global Annual Mass Change Relative Uncertainty', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].axhline(y=100, color='gray', linestyle='--', linewidth=0.8, label='100% (signal = noise)')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, min(500, rel_unc.quantile(0.95) * 1.5))

# Panel B: Regional uncertainty comparison
regional_unc = []
for region_key in cal_results:
    if region_key == '0_global':
        continue
    rdf = cal_results[region_key]
    rdf = rdf[(rdf['start_dates'] >= 2000) & (rdf['end_dates'] <= 2024)]
    mean_signal = np.abs(rdf['combined_gt'].mean())
    mean_unc = rdf['combined_gt_errors'].mean()
    rel = (mean_unc / mean_signal * 100) if mean_signal > 0 else np.nan
    regional_unc.append({
        'region': REGION_NAMES.get(region_key, region_key),
        'mean_signal_gt': mean_signal,
        'mean_uncertainty_gt': mean_unc,
        'relative_uncertainty_pct': rel
    })

unc_df = pd.DataFrame(regional_unc).sort_values('relative_uncertainty_pct')

axes[1].barh(unc_df['region'], unc_df['relative_uncertainty_pct'], color='#FF9800', alpha=0.8)
axes[1].set_xlabel('Mean Relative Uncertainty (%)', fontsize=12)
axes[1].set_title('B) Regional Mean Relative Uncertainty', fontsize=13, fontweight='bold')
axes[1].axvline(x=100, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig6_uncertainty_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_uncertainty_analysis.png")

# ============================================================
# 9. SAVE COMPREHENSIVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Save Comprehensive Results")
print("=" * 60)

# Save global time series
global_df.to_csv(os.path.join(OUTPUT_DIR, 'global_time_series.csv'), index=False)

# Save all regional time series combined
all_regions = []
for region_key, rdf in cal_results.items():
    rdf_copy = rdf.copy()
    rdf_copy['region_key'] = region_key
    rdf_copy['region_name'] = REGION_NAMES.get(region_key, region_key)
    all_regions.append(rdf_copy)
all_regions_df = pd.concat(all_regions, ignore_index=True)
all_regions_df.to_csv(os.path.join(OUTPUT_DIR, 'all_regional_time_series.csv'), index=False)

# Summary statistics
summary = {
    'global': {
        'period': '2000-2023',
        'total_mass_change_gt': float(total_change_gt),
        'total_mass_change_gt_error': float(total_error_gt),
        'mean_annual_change_gt': float(mean_annual_gt),
        'mean_annual_specific_change_mwe': float(mean_annual_mwe),
        'cumulative_change_gt_2000_2023': float(global_df['cumulative_gt'].iloc[-1]),
    },
    'regional_ranking_by_loss': []
}

for _, row in regional_df.sort_values('total_gt').iterrows():
    summary['regional_ranking_by_loss'].append({
        'region': row['region_name'],
        'total_gt': float(row['total_gt']),
        'total_gt_error': float(row['total_gt_err']),
        'mean_annual_mwe': float(row['mean_annual_mwe']),
    })

with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("  Saved: summary_statistics.json")
print("  Saved: global_time_series.csv")
print("  Saved: all_regional_time_series.csv")
print("  Saved: regional_summary.csv")
print("  Saved: dataset_inventory.csv")

# ============================================================
# 10. PRINT FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nGlobal glacier mass change (2000-2023):")
print(f"  Total: {total_change_gt:.1f} ± {total_error_gt:.1f} Gt")
print(f"  Mean annual: {mean_annual_gt:.1f} Gt/yr ({mean_annual_mwe:.3f} m w.e./yr)")
print(f"  Cumulative: {global_df['cumulative_gt'].iloc[-1]:.1f} Gt")
print(f"\nTop 5 regions by total mass loss:")
for _, row in regional_df.sort_values('total_gt').head(5).iterrows():
    print(f"  {row['region_name']:30s}: {row['total_gt']:.1f} ± {row['total_gt_err']:.1f} Gt")
print(f"\nFigures saved to: {REPORT_IMG_DIR}/")
print(f"Results saved to: {OUTPUT_DIR}/")
