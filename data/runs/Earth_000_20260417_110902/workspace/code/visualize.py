#!/usr/bin/env python3
"""
GlaMBIE Visualization: Generate all figures for the report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_000_20260417_110902'
DATA = os.path.join(BASE, 'data', 'glambie')
RESULTS_CAL = os.path.join(DATA, 'results', 'calendar_years')
RESULTS_HYD = os.path.join(DATA, 'results', 'hydrological_years')
OUTPUTS = os.path.join(BASE, 'outputs')
IMAGES = os.path.join(BASE, 'report', 'images')
os.makedirs(IMAGES, exist_ok=True)

import glob

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

# Load data
cal_results = {}
for csv_file in sorted(glob.glob(os.path.join(RESULTS_CAL, '*.csv'))):
    fname = os.path.basename(csv_file)
    key = fname.replace('.csv', '')
    cal_results[key] = pd.read_csv(csv_file)

hyd_results = {}
for csv_file in sorted(glob.glob(os.path.join(RESULTS_HYD, '*.csv'))):
    fname = os.path.basename(csv_file)
    key = fname.replace('.csv', '')
    hyd_results[key] = pd.read_csv(csv_file)

global_cal = cal_results['0_global'].copy()
global_cal['year'] = global_cal['start_dates'].astype(int)
global_cal['cumulative_gt'] = global_cal['combined_gt'].cumsum()
global_cal['cumulative_mwe'] = global_cal['combined_mwe'].cumsum()
global_cal['cumulative_gt_err'] = np.sqrt(np.cumsum(global_cal['combined_gt_errors']**2))
global_cal['cumulative_mwe_err'] = np.sqrt(np.cumsum(global_cal['combined_mwe_errors']**2))

# ============================================================
# FIGURE 1: Global Annual and Cumulative Mass Change
# ============================================================
print("Figure 1: Global mass change time series...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Panel A: Annual rates
ax = axes[0]
years = global_cal['year'].values
rates_gt = global_cal['combined_gt'].values
errs_gt = global_cal['combined_gt_errors'].values

ax.bar(years, rates_gt, color='steelblue', alpha=0.7, label='Annual mass change')
ax.errorbar(years, rates_gt, yerr=errs_gt, fmt='none', ecolor='black', capsize=2, alpha=0.5)

# Add trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(years, rates_gt)
ax.plot(years, slope * years + intercept, 'r--', linewidth=2, 
        label=f'Trend: {slope:.1f} ± {std_err:.1f} Gt yr⁻² (p={p_value:.3f})')

ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Mass change (Gt yr⁻¹)', fontsize=12)
ax.set_title('(a) Global Annual Glacier Mass Change', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel B: Cumulative
ax = axes[1]
cum_gt = global_cal['cumulative_gt'].values
cum_err = global_cal['cumulative_gt_err'].values

ax.fill_between(years, cum_gt - cum_err, cum_gt + cum_err, alpha=0.3, color='steelblue')
ax.plot(years, cum_gt, 'o-', color='steelblue', linewidth=2, markersize=4, label='Cumulative mass change')

# Add SLE axis
ax2 = ax.twinx()
ax2.set_ylim(np.array(ax.get_ylim()) / -362.5)
ax2.set_ylabel('Sea level equivalent (mm)', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative mass change (Gt)', fontsize=12)
ax.set_title('(b) Cumulative Global Glacier Mass Change', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig1_global_mass_change.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_global_mass_change.png")

# ============================================================
# FIGURE 2: Regional Mass Change Rates (Bar Chart)
# ============================================================
print("Figure 2: Regional mass change rates...")
regional_df = pd.read_csv(os.path.join(OUTPUTS, 'regional_summary.csv'))
regional_df = regional_df.sort_values('total_gt')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel A: Total mass change (Gt)
ax = axes[0]
colors = ['#d62728' if v < 0 else '#2ca02c' for v in regional_df['total_gt']]
bars = ax.barh(range(len(regional_df)), regional_df['total_gt'], color=colors, alpha=0.8)
ax.set_yticks(range(len(regional_df)))
ax.set_yticklabels(regional_df['region_name'], fontsize=9)
ax.set_xlabel('Total mass change 2000–2023 (Gt)', fontsize=11)
ax.set_title('(a) Total Mass Change by Region', fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Add error bars
ax.errorbar(regional_df['total_gt'], range(len(regional_df)), 
            xerr=regional_df['total_gt_err'], fmt='none', ecolor='black', capsize=2, alpha=0.5)

# Panel B: Specific mass change rate (m w.e./yr)
ax = axes[1]
regional_df_sorted = regional_df.sort_values('avg_rate_mwe_yr')
colors2 = ['#d62728' if v < 0 else '#2ca02c' for v in regional_df_sorted['avg_rate_mwe_yr']]
ax.barh(range(len(regional_df_sorted)), regional_df_sorted['avg_rate_mwe_yr'], color=colors2, alpha=0.8)
ax.set_yticks(range(len(regional_df_sorted)))
ax.set_yticklabels(regional_df_sorted['region_name'], fontsize=9)
ax.set_xlabel('Average specific mass change (m w.e. yr⁻¹)', fontsize=11)
ax.set_title('(b) Specific Mass Change Rate by Region', fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig2_regional_mass_change.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_regional_mass_change.png")

# ============================================================
# FIGURE 3: Regional Time Series (Multi-panel)
# ============================================================
print("Figure 3: Regional time series panels...")

# Sort regions by total mass loss
region_order = regional_df.sort_values('total_gt')['region_id'].tolist()

fig, axes = plt.subplots(5, 4, figsize=(20, 22), sharex=True)
axes = axes.flatten()

for idx, rid in enumerate(region_order):
    ax = axes[idx]
    key = None
    for k in cal_results:
        if k.startswith(f'{rid}_'):
            key = k
            break
    if key is None:
        continue
    
    df = cal_results[key]
    years = df['start_dates'].astype(int).values
    
    # Cumulative
    cum = df['combined_gt'].cumsum().values
    cum_err = np.sqrt(np.cumsum(df['combined_gt_errors'].values**2))
    
    ax.fill_between(years, cum - cum_err, cum + cum_err, alpha=0.3, color='steelblue')
    ax.plot(years, cum, '-', color='steelblue', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(REGION_MAP.get(rid, rid), fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx >= 16:
        ax.set_xlabel('Year', fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel('Cumulative (Gt)', fontsize=9)

# Remove empty subplot
if len(region_order) < 20:
    for i in range(len(region_order), 20):
        axes[i].set_visible(False)

fig.suptitle('Cumulative Glacier Mass Change by Region (2000–2023)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig3_regional_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_regional_timeseries.png")

# ============================================================
# FIGURE 4: Method Comparison (Hydrological Year Data)
# ============================================================
print("Figure 4: Method comparison...")

fig, axes = plt.subplots(5, 4, figsize=(20, 22), sharex=True)
axes = axes.flatten()

method_colors_hyd = {
    'altimetry': '#1f77b4',
    'gravimetry': '#ff7f0e',
    'demdiff_and_glaciological': '#2ca02c',
    'combined': '#d62728',
}
method_labels_hyd = {
    'altimetry': 'Altimetry',
    'gravimetry': 'Gravimetry',
    'demdiff_and_glaciological': 'DEM diff. + Glaciol.',
    'combined': 'Combined',
}

for idx, rid in enumerate(region_order):
    ax = axes[idx]
    key = None
    for k in hyd_results:
        if k.startswith(f'{rid}_'):
            key = k
            break
    if key is None:
        continue
    
    df = hyd_results[key]
    years = (df['start_dates'] + df['end_dates']) / 2
    
    # Plot combined
    ax.fill_between(years, 
                     df['combined_mwe'] - df['combined_mwe_errors'],
                     df['combined_mwe'] + df['combined_mwe_errors'],
                     alpha=0.2, color='grey')
    ax.plot(years, df['combined_mwe'], '-', color='grey', linewidth=2, label='Combined', zorder=5)
    
    # Plot individual methods
    for method_prefix, color in method_colors_hyd.items():
        if method_prefix == 'combined':
            continue
        mwe_col = f'{method_prefix}_mwe'
        if mwe_col in df.columns:
            valid = df[mwe_col].notna()
            if valid.sum() > 0:
                ax.plot(years[valid], df[mwe_col][valid], 'o-', color=color, 
                       linewidth=1, markersize=2, alpha=0.7,
                       label=method_labels_hyd.get(method_prefix, method_prefix))
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(REGION_MAP.get(rid, rid), fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc='lower left')

for i in range(len(region_order), 20):
    axes[i].set_visible(False)

fig.suptitle('Method Comparison: Annual Specific Mass Change (m w.e.) by Region', 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig4_method_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_method_comparison.png")

# ============================================================
# FIGURE 5: Data Coverage Heatmap
# ============================================================
print("Figure 5: Data coverage heatmap...")

# Load input data
INPUT = os.path.join(DATA, 'input')
all_inputs = []
for region_dir in sorted(glob.glob(os.path.join(INPUT, '*'))):
    if not os.path.isdir(region_dir):
        continue
    region_name = os.path.basename(region_dir)
    region_id = region_name.split('_')[0]
    for csv_file in sorted(glob.glob(os.path.join(region_dir, '*.csv'))):
        fname = os.path.basename(csv_file)
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
            all_inputs.append(df)
        except:
            pass

input_df = pd.concat(all_inputs, ignore_index=True)

# Create coverage matrix: regions x methods (number of datasets)
coverage = input_df.groupby(['region_name', 'method'])['filename'].nunique().unstack(fill_value=0)
# Reorder
method_order = ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']
coverage = coverage.reindex(columns=[m for m in method_order if m in coverage.columns])

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(coverage.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(coverage.columns)))
ax.set_xticklabels(['Glaciological', 'DEM Diff.', 'Altimetry', 'Gravimetry', 'Combined'], 
                    fontsize=11, rotation=45, ha='right')
ax.set_yticks(range(len(coverage.index)))
ax.set_yticklabels(coverage.index, fontsize=10)

# Add text annotations
for i in range(len(coverage.index)):
    for j in range(len(coverage.columns)):
        val = coverage.values[i, j]
        color = 'white' if val > coverage.values.max() * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=10, color=color)

plt.colorbar(im, ax=ax, label='Number of datasets', shrink=0.7)
ax.set_title('Data Coverage: Number of Input Datasets by Region and Method', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig5_data_coverage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_data_coverage.png")

# ============================================================
# FIGURE 6: Uncertainty Analysis
# ============================================================
print("Figure 6: Uncertainty analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Global uncertainty over time
ax = axes[0]
years = global_cal['year'].values
ax.fill_between(years, 
                 global_cal['combined_gt'] - global_cal['combined_gt_errors'],
                 global_cal['combined_gt'] + global_cal['combined_gt_errors'],
                 alpha=0.3, color='steelblue', label='Uncertainty range')
ax.plot(years, global_cal['combined_gt'], 'o-', color='steelblue', linewidth=2, markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Mass change (Gt yr⁻¹)', fontsize=12)
ax.set_title('(a) Global Mass Change with Uncertainty', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel B: Relative uncertainty by region
ax = axes[1]
rel_unc = []
for key, df in sorted(cal_results.items()):
    if key == '0_global':
        continue
    rid = key.split('_')[0]
    rname = REGION_MAP.get(rid, key)
    avg_rate = abs(df['combined_gt'].mean())
    avg_err = df['combined_gt_errors'].mean()
    if avg_rate > 0:
        rel_unc.append({'region': rname, 'relative_uncertainty': avg_err / avg_rate * 100})

rel_unc_df = pd.DataFrame(rel_unc).sort_values('relative_uncertainty')
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rel_unc_df)))
ax.barh(range(len(rel_unc_df)), rel_unc_df['relative_uncertainty'], color=colors)
ax.set_yticks(range(len(rel_unc_df)))
ax.set_yticklabels(rel_unc_df['region'], fontsize=9)
ax.set_xlabel('Relative uncertainty (%)', fontsize=11)
ax.set_title('(b) Relative Uncertainty by Region', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig6_uncertainty_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_uncertainty_analysis.png")

# ============================================================
# FIGURE 7: Glacier Area Change
# ============================================================
print("Figure 7: Glacier area change...")

fig, ax = plt.subplots(figsize=(12, 6))

years = global_cal['year'].values
area = global_cal['glacier_area'].values / 1000  # Convert to thousands km²

ax.plot(years, area, 'o-', color='darkgreen', linewidth=2, markersize=5)
ax.fill_between(years, area * 0.98, area * 1.02, alpha=0.2, color='green')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Global glacier area (×10³ km²)', fontsize=12)
ax.set_title('Global Glacier Area Evolution (2000–2023)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add percentage change annotation
pct_change = (area[-1] - area[0]) / area[0] * 100
ax.annotate(f'Total change: {pct_change:.1f}%', xy=(years[-1], area[-1]),
            xytext=(years[-5], area[0] - 5), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig7_glacier_area.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_glacier_area.png")

# ============================================================
# FIGURE 8: Decadal Comparison
# ============================================================
print("Figure 8: Decadal comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# Define decades
decades = {
    '2000-2009': (2000, 2010),
    '2010-2019': (2010, 2020),
    '2020-2023': (2020, 2024),
}

decade_rates = {}
for label, (start, end) in decades.items():
    mask = (global_cal['year'] >= start) & (global_cal['year'] < end)
    decade_rates[label] = {
        'mean': global_cal.loc[mask, 'combined_gt'].mean(),
        'std': global_cal.loc[mask, 'combined_gt'].std(),
        'err': global_cal.loc[mask, 'combined_gt_errors'].mean(),
    }

x = range(len(decades))
means = [decade_rates[k]['mean'] for k in decades]
errs = [decade_rates[k]['err'] for k in decades]

bars = ax.bar(x, means, yerr=errs, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(decades.keys(), fontsize=12)
ax.set_ylabel('Average mass change rate (Gt yr⁻¹)', fontsize=12)
ax.set_title('Decadal Comparison of Global Glacier Mass Change Rates', fontsize=14, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (m, e) in enumerate(zip(means, errs)):
    ax.text(i, m - 15, f'{m:.0f} ± {e:.0f}\nGt yr⁻¹', ha='center', va='top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig8_decadal_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8_decadal_comparison.png")

# ============================================================
# FIGURE 9: Sea Level Contribution
# ============================================================
print("Figure 9: Sea level contribution...")

fig, ax = plt.subplots(figsize=(12, 6))

sle_cum = global_cal['cumulative_gt'].values / -362.5
sle_cum_err = global_cal['cumulative_gt_err'].values / 362.5

ax.fill_between(years, sle_cum - sle_cum_err, sle_cum + sle_cum_err, alpha=0.3, color='coral')
ax.plot(years, sle_cum, 'o-', color='coral', linewidth=2, markersize=4)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative sea level equivalent (mm)', fontsize=12)
ax.set_title('Glacier Contribution to Sea Level Rise (2000–2023)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add rate annotation
total_sle = sle_cum[-1]
rate_sle = total_sle / len(years)
ax.annotate(f'Total: {total_sle:.1f} mm\nRate: {rate_sle:.2f} mm yr⁻¹', 
            xy=(years[-1], sle_cum[-1]),
            xytext=(years[-8], sle_cum[-1] * 0.4),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig9_sea_level_contribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig9_sea_level_contribution.png")

# ============================================================
# FIGURE 10: Regional Contribution Stacked Area
# ============================================================
print("Figure 10: Regional contribution stacked area...")

# Get top contributors
regional_df = pd.read_csv(os.path.join(OUTPUTS, 'regional_summary.csv'))
top_regions = regional_df.nsmallest(8, 'total_gt')['region_id'].tolist()

fig, ax = plt.subplots(figsize=(14, 7))

# Collect cumulative data for top regions
region_data = {}
for rid in top_regions:
    key = None
    for k in cal_results:
        if k.startswith(f'{rid}_'):
            key = k
            break
    if key:
        df = cal_results[key]
        region_data[REGION_MAP.get(rid, rid)] = df['combined_gt'].cumsum().values

years_arr = cal_results[list(cal_results.keys())[0]]['start_dates'].astype(int).values

# Plot individual lines
colors_top = plt.cm.tab10(np.linspace(0, 1, len(region_data)))
for i, (rname, cum_vals) in enumerate(region_data.items()):
    if len(cum_vals) == len(years_arr):
        ax.plot(years_arr, cum_vals, '-', color=colors_top[i], linewidth=2, label=rname)

# Add global
ax.plot(years_arr, global_cal['cumulative_gt'].values, 'k-', linewidth=3, label='Global')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cumulative mass change (Gt)', fontsize=12)
ax.set_title('Top Contributing Regions to Global Glacier Mass Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left', ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig10_regional_contributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig10_regional_contributions.png")

# ============================================================
# FIGURE 11: Method Agreement Analysis
# ============================================================
print("Figure 11: Method agreement scatter...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

method_pairs = [
    ('altimetry', 'gravimetry', 'Altimetry vs Gravimetry'),
    ('altimetry', 'demdiff_and_glaciological', 'Altimetry vs DEM diff.+Glaciol.'),
    ('gravimetry', 'demdiff_and_glaciological', 'Gravimetry vs DEM diff.+Glaciol.'),
]

for ax_idx, (m1, m2, title) in enumerate(method_pairs):
    ax = axes[ax_idx]
    
    all_m1 = []
    all_m2 = []
    
    for key, df in hyd_results.items():
        col1 = f'{m1}_mwe'
        col2 = f'{m2}_mwe'
        if col1 in df.columns and col2 in df.columns:
            valid = df[[col1, col2]].dropna()
            if len(valid) > 0:
                all_m1.extend(valid[col1].tolist())
                all_m2.extend(valid[col2].tolist())
    
    if len(all_m1) > 0:
        all_m1 = np.array(all_m1)
        all_m2 = np.array(all_m2)
        
        ax.scatter(all_m1, all_m2, alpha=0.4, s=20, color='steelblue')
        
        # 1:1 line
        lims = [min(all_m1.min(), all_m2.min()), max(all_m1.max(), all_m2.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line')
        
        # Linear fit
        sl, ic, rv, pv, se = stats.linregress(all_m1, all_m2)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, sl * x_fit + ic, 'r-', linewidth=1.5, 
                label=f'Fit: y={sl:.2f}x+{ic:.2f} (R²={rv**2:.2f})')
        
        ax.set_xlabel(f'{m1.replace("_", " ").title()} (m w.e.)', fontsize=10)
        ax.set_ylabel(f'{m2.replace("_", " ").title()} (m w.e.)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

plt.suptitle('Inter-Method Agreement in Specific Mass Change Estimates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig11_method_agreement.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig11_method_agreement.png")

# ============================================================
# FIGURE 12: Input Data Temporal Coverage
# ============================================================
print("Figure 12: Input data temporal coverage...")

fig, ax = plt.subplots(figsize=(14, 8))

# For each region, show time span of input datasets
y_pos = 0
y_ticks = []
y_labels = []

method_colors = {
    'glaciological': '#2ca02c',
    'demdiff': '#d62728',
    'altimetry': '#1f77b4',
    'gravimetry': '#ff7f0e',
    'combined': '#9467bd',
}

for rid in sorted(REGION_MAP.keys(), key=lambda x: int(x)):
    rname = REGION_MAP[rid]
    region_inputs = input_df[input_df['region_id'] == rid]
    
    for fname in region_inputs['filename'].unique():
        subset = region_inputs[region_inputs['filename'] == fname]
        method = subset['method'].iloc[0]
        start = subset['start_dates'].min()
        end = subset['end_dates'].max()
        
        ax.barh(y_pos, end - start, left=start, height=0.6, 
                color=method_colors.get(method, 'grey'), alpha=0.6)
        y_pos += 1
    
    y_ticks.append(y_pos - region_inputs['filename'].nunique() / 2)
    y_labels.append(rname)
    y_pos += 1  # gap between regions

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel('Year', fontsize=12)
ax.set_title('Temporal Coverage of Input Datasets', fontsize=14, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, alpha=0.6, label=l.title()) 
                   for l, c in method_colors.items()]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig12_temporal_coverage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig12_temporal_coverage.png")

print("\nAll figures generated successfully!")
