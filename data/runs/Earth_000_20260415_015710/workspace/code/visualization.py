"""
GlaMBIE Glacier Mass Change Reconciliation - Visualization Module
=================================================================
Generates all figures for the research report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
IMAGE_DIR = os.path.join(BASE_DIR, 'report', 'images')

os.makedirs(IMAGE_DIR, exist_ok=True)

# Style settings
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

REGION_MAP = {
    0: 'Global', 1: 'Alaska', 2: 'Western Canada & US',
    3: 'Arctic Canada North', 4: 'Arctic Canada South',
    5: 'Greenland Periphery', 6: 'Iceland', 7: 'Svalbard',
    8: 'Scandinavia', 9: 'Russian Arctic', 10: 'North Asia',
    11: 'Central Europe', 12: 'Caucasus & Middle East',
    13: 'Central Asia', 14: 'South Asia West', 15: 'South Asia East',
    16: 'Low Latitudes', 17: 'Southern Andes', 18: 'New Zealand',
    19: 'Antarctic & Subantarctic'
}

REGION_CODES = {
    0: 'GLO', 1: 'ALA', 2: 'WNA', 3: 'ACN', 4: 'ACS',
    5: 'GRL', 6: 'ISL', 7: 'SJM', 8: 'SCA', 9: 'RUA',
    10: 'ASN', 11: 'CEU', 12: 'CAU', 13: 'ASC', 14: 'ASW',
    15: 'ASE', 16: 'TRP', 17: 'SAN', 18: 'NZL', 19: 'ANT'
}

# Color palette for regions
REGION_COLORS = {
    1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
    5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f',
    9: '#bcbd22', 10: '#17becf', 11: '#aec7e8', 12: '#ffbb78',
    13: '#98df8a', 14: '#ff9896', 15: '#c5b0d5', 16: '#c49c94',
    17: '#f7b6d2', 18: '#c7c7c7', 19: '#dbdb8d'
}

METHOD_COLORS = {
    'altimetry': '#2196F3',
    'gravimetry': '#FF5722',
    'demdiff_and_glaciological': '#4CAF50',
    'combined': '#9C27B0'
}

METHOD_LABELS = {
    'altimetry': 'Altimetry',
    'gravimetry': 'Gravimetry',
    'demdiff_and_glaciological': 'DEM diff. + Glaciological',
    'combined': 'Combined'
}

# ============================================================
# Load processed data
# ============================================================
ts_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'annual_time_series.csv'))
cumul_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cumulative_mass_change.csv'))
agreement_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'method_agreement.csv'))
coverage_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'method_coverage.csv'))
trends_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'regional_trends.csv'))
regional_summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'regional_summary.csv'))

with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'r') as f:
    summary_stats = json.load(f)

# ============================================================
# Figure 1: Global Mass Change Time Series
# ============================================================
def fig1_global_timeseries():
    """Global mass change time series with uncertainty bands."""
    global_ts = ts_df[ts_df['region_id'] == 0].sort_values('year')
    global_cumul = cumul_df[cumul_df['region_id'] == 0].sort_values('year')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Annual mass change in Gt
    ax = axes[0, 0]
    ax.fill_between(global_ts['year'],
                    global_ts['mass_change_gt'] - global_ts['mass_change_gt_err'],
                    global_ts['mass_change_gt'] + global_ts['mass_change_gt_err'],
                    alpha=0.3, color='#1f77b4', label='±1σ uncertainty')
    ax.plot(global_ts['year'], global_ts['mass_change_gt'], 'o-', color='#1f77b4',
            markersize=4, linewidth=1.5, label='Annual mass change')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mass change (Gt yr⁻¹)')
    ax.set_title('(a) Annual global glacier mass change')
    ax.legend(loc='lower left')
    
    # Panel B: Annual specific mass change in m w.e.
    ax = axes[0, 1]
    ax.fill_between(global_ts['year'],
                    global_ts['mass_change_mwe'] - global_ts['mass_change_mwe_err'],
                    global_ts['mass_change_mwe'] + global_ts['mass_change_mwe_err'],
                    alpha=0.3, color='#d62728', label='±1σ uncertainty')
    ax.plot(global_ts['year'], global_ts['mass_change_mwe'], 'o-', color='#d62728',
            markersize=4, linewidth=1.5, label='Annual specific mass change')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Specific mass change (m w.e. yr⁻¹)')
    ax.set_title('(b) Annual global specific mass change')
    ax.legend(loc='lower left')
    
    # Panel C: Cumulative mass change in Gt
    ax = axes[1, 0]
    ax.fill_between(global_cumul['year'],
                    global_cumul['cumulative_gt'] - global_cumul['cumulative_gt_err'],
                    global_cumul['cumulative_gt'] + global_cumul['cumulative_gt_err'],
                    alpha=0.3, color='#2ca02c', label='±1σ uncertainty')
    ax.plot(global_cumul['year'], global_cumul['cumulative_gt'], '-', color='#2ca02c',
            linewidth=2, label='Cumulative mass change')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative mass change (Gt)')
    ax.set_title('(c) Cumulative global glacier mass change')
    ax.legend(loc='upper right')
    
    # Panel D: Cumulative specific mass change in m w.e.
    ax = axes[1, 1]
    ax.fill_between(global_cumul['year'],
                    global_cumul['cumulative_mwe'] - global_cumul['cumulative_mwe_err'],
                    global_cumul['cumulative_mwe'] + global_cumul['cumulative_mwe_err'],
                    alpha=0.3, color='#9467bd', label='±1σ uncertainty')
    ax.plot(global_cumul['year'], global_cumul['cumulative_mwe'], '-', color='#9467bd',
            linewidth=2, label='Cumulative specific mass change')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative specific mass change (m w.e.)')
    ax.set_title('(d) Cumulative global specific mass change')
    ax.legend(loc='upper right')
    
    fig.suptitle('Global Glacier Mass Change (2000–2023)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig1_global_timeseries.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 1 saved: fig1_global_timeseries.png")

# ============================================================
# Figure 2: Regional Mass Change Bar Chart
# ============================================================
def fig2_regional_bar_chart():
    """Regional mean annual mass change and cumulative mass change."""
    regional = regional_summary.sort_values('cumulative_gt')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Panel A: Mean annual mass change (Gt/yr)
    ax = axes[0]
    colors = [REGION_COLORS.get(rid, '#333') for rid in regional['region_id']]
    bars = ax.barh(range(len(regional)), regional['mean_annual_gt'],
                   xerr=regional['mean_uncertainty_gt'],
                   color=colors, alpha=0.8, edgecolor='k', linewidth=0.5,
                   error_kw={'elinewidth': 1, 'capsize': 3})
    ax.set_yticks(range(len(regional)))
    ax.set_yticklabels([f"{REGION_CODES.get(rid, '')} - {rn}" 
                        for rid, rn in zip(regional['region_id'], regional['region_name'])],
                       fontsize=8)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Mean annual mass change (Gt yr⁻¹)')
    ax.set_title('(a) Mean annual mass change (2000–2023)')
    
    # Panel B: Cumulative mass change (Gt)
    ax = axes[1]
    ax.barh(range(len(regional)), regional['cumulative_gt'],
            color=colors, alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(regional)))
    ax.set_yticklabels([f"{REGION_CODES.get(rid, '')} - {rn}" 
                        for rid, rn in zip(regional['region_id'], regional['region_name'])],
                       fontsize=8)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Cumulative mass change (Gt)')
    ax.set_title('(b) Cumulative mass change (2000–2023)')
    
    fig.suptitle('Regional Glacier Mass Change (2000–2023)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig2_regional_bar_chart.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 2 saved: fig2_regional_bar_chart.png")

# ============================================================
# Figure 3: Method Comparison
# ============================================================
def fig3_method_comparison():
    """Compare estimates from different observation methods."""
    # Select regions with good multi-method coverage
    regions_with_methods = agreement_df.groupby('region_id')['n_methods'].max()
    good_regions = regions_with_methods[regions_with_methods >= 2].index.tolist()
    # Pick top 6 regions by number of multi-method comparisons
    region_counts = agreement_df[agreement_df['region_id'].isin(good_regions)].groupby('region_id').size()
    top_regions = region_counts.nlargest(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, region_id in enumerate(top_regions):
        ax = axes[idx]
        region_data = agreement_df[agreement_df['region_id'] == region_id].sort_values('year')
        
        for method in ['altimetry', 'gravimetry', 'demdiff_and_glaciological']:
            col = f'{method}_gt'
            err_col = f'{method}_gt_err'
            if col in region_data.columns:
                valid = region_data[col].notna()
                if valid.sum() > 0:
                    sub = region_data[valid]
                    ax.errorbar(sub['year'], sub[col], yerr=sub[err_col] if err_col in sub.columns else None,
                               fmt='o-', markersize=3, linewidth=1, alpha=0.7,
                               color=METHOD_COLORS.get(method, '#333'),
                               label=METHOD_LABELS.get(method, method))
        
        # Also plot combined
        combined_ts = ts_df[(ts_df['region_id'] == region_id)].sort_values('year')
        ax.plot(combined_ts['year'], combined_ts['mass_change_gt'], 'k-', linewidth=2,
                label='Combined', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(f'{REGION_MAP.get(region_id, f"Region {region_id}")}', fontsize=11)
        ax.set_xlabel('Year')
        ax.set_ylabel('Mass change (Gt yr⁻¹)')
        ax.legend(fontsize=7, loc='best')
    
    fig.suptitle('Method Comparison: Regional Mass Change Estimates', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig3_method_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 3 saved: fig3_method_comparison.png")

# ============================================================
# Figure 4: Stacked Area - Cumulative Regional Contributions
# ============================================================
def fig4_stacked_cumulative():
    """Stacked area chart of cumulative regional contributions to global mass loss."""
    regional_cumul = cumul_df[cumul_df['region_id'] != 0].copy()
    
    # Get top contributing regions
    top_regions = regional_summary.nsmallest(10, 'cumulative_gt')['region_id'].tolist()
    other_regions = [rid for rid in regional_cumul['region_id'].unique() if rid not in top_regions]
    
    # Pivot for stacking
    pivot_data = regional_cumul.pivot_table(index='year', columns='region_id', values='cumulative_gt')
    
    # Separate top and other
    top_data = pivot_data[top_regions].copy()
    other_data = pivot_data[other_regions].copy()
    
    # Sum others
    if len(other_regions) > 0:
        top_data['Other'] = other_data.sum(axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    labels = [f"{REGION_CODES.get(rid, '')} {REGION_MAP.get(rid, '')}" for rid in top_regions]
    if len(other_regions) > 0:
        labels.append('Other regions')
    
    colors = [REGION_COLORS.get(rid, '#888') for rid in top_regions]
    if len(other_regions) > 0:
        colors.append('#cccccc')
    
    ax.stackplot(top_data.index, [top_data[col] for col in top_data.columns],
                 labels=labels, colors=colors, alpha=0.8)
    
    # Overlay global cumulative
    global_cumul = cumul_df[cumul_df['region_id'] == 0].sort_values('year')
    ax.plot(global_cumul['year'], global_cumul['cumulative_gt'], 'k-', linewidth=2.5,
            label='Global total', zorder=10)
    
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative mass change (Gt)')
    ax.set_title('Cumulative Regional Contributions to Global Glacier Mass Loss (2000–2023)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig4_stacked_cumulative.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 4 saved: fig4_stacked_cumulative.png")

# ============================================================
# Figure 5: Regional Specific Mass Change Heatmap
# ============================================================
def fig5_specific_mass_change_heatmap():
    """Heatmap of annual specific mass change by region."""
    regional_ts = ts_df[ts_df['region_id'] != 0].copy()
    
    # Pivot
    pivot = regional_ts.pivot_table(index='region_id', columns='year', values='mass_change_mwe')
    
    # Sort by mean mass change
    mean_order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[mean_order]
    
    # Create labels
    ylabels = [f"{REGION_CODES.get(rid, '')} {REGION_MAP.get(rid, '')}" for rid in pivot.index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, cmap='RdBu', center=0, annot=False,
                xticklabels=True, yticklabels=ylabels,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Specific mass change (m w.e. yr⁻¹)'},
                ax=ax)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('')
    ax.set_title('Annual Specific Mass Change by Region (2000–2023)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig5_specific_mass_change_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 5 saved: fig5_specific_mass_change_heatmap.png")

# ============================================================
# Figure 6: Method Coverage and Uncertainty
# ============================================================
def fig6_method_coverage_uncertainty():
    """Method coverage and uncertainty contribution analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Panel A: Method coverage by region
    ax = axes[0]
    coverage_pivot = coverage_df.pivot_table(index='region_name', columns='method', values='n_years', fill_value=0)
    coverage_pivot = coverage_pivot.sort_values(by=coverage_pivot.columns.tolist(), ascending=False)
    
    coverage_pivot.plot(kind='barh', stacked=True, ax=ax,
                        color=[METHOD_COLORS.get(m.replace('demdiff_and_glaciological', 'demdiff_and_glaciological'), '#333')
                               for m in coverage_pivot.columns],
                        alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Number of years with data')
    ax.set_ylabel('')
    ax.set_title('(a) Temporal coverage by observation method')
    ax.legend([METHOD_LABELS.get(m, m) for m in coverage_pivot.columns],
              fontsize=7, loc='lower right')
    
    # Panel B: Relative uncertainty over time (global)
    ax = axes[1]
    global_ts = ts_df[ts_df['region_id'] == 0].sort_values('year')
    relative_uncertainty = global_ts['mass_change_gt_err'] / abs(global_ts['mass_change_gt']) * 100
    
    ax.bar(global_ts['year'], relative_uncertainty, color='#FF9800', alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative uncertainty (%)')
    ax.set_title('(b) Relative uncertainty of global annual mass change')
    ax.set_ylim(0, max(relative_uncertainty) * 1.2)
    
    fig.suptitle('Method Coverage and Uncertainty Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig6_method_coverage_uncertainty.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 6 saved: fig6_method_coverage_uncertainty.png")

# ============================================================
# Figure 7: Mass Loss Acceleration
# ============================================================
def fig7_acceleration_trends():
    """Mass loss acceleration and trends analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Global mass change with trend line
    ax = axes[0, 0]
    global_ts = ts_df[ts_df['region_id'] == 0].sort_values('year')
    ax.fill_between(global_ts['year'],
                    global_ts['mass_change_gt'] - global_ts['mass_change_gt_err'],
                    global_ts['mass_change_gt'] + global_ts['mass_change_gt_err'],
                    alpha=0.2, color='#1f77b4')
    ax.plot(global_ts['year'], global_ts['mass_change_gt'], 'o-', color='#1f77b4',
            markersize=4, linewidth=1.5, label='Annual')
    
    # Linear trend
    slope, intercept, r, p, se = stats.linregress(global_ts['year'], global_ts['mass_change_gt'])
    trend_line = slope * global_ts['year'] + intercept
    ax.plot(global_ts['year'], trend_line, 'r--', linewidth=2,
            label=f'Linear trend: {slope:.1f} Gt yr⁻²')
    
    # Quadratic fit
    coeffs = np.polyfit(global_ts['year'], global_ts['mass_change_gt'], 2)
    quad_line = np.polyval(coeffs, global_ts['year'])
    ax.plot(global_ts['year'], quad_line, 'g-', linewidth=2,
            label=f'Quadratic fit (accel: {2*coeffs[0]:.1f} Gt yr⁻²)')
    
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mass change (Gt yr⁻¹)')
    ax.set_title('(a) Global mass change with trend fits')
    ax.legend(fontsize=8)
    
    # Panel B: Running mean
    ax = axes[0, 1]
    window = 5
    rolling_mean = global_ts['mass_change_gt'].rolling(window=window, center=True).mean()
    ax.plot(global_ts['year'], global_ts['mass_change_gt'], 'o-', color='#1f77b4',
            markersize=3, linewidth=1, alpha=0.5, label='Annual')
    ax.plot(global_ts['year'], rolling_mean, 'r-', linewidth=2.5,
            label=f'{window}-year running mean')
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mass change (Gt yr⁻¹)')
    ax.set_title('(b) Global mass change with running mean')
    ax.legend()
    
    # Panel C: Regional acceleration
    ax = axes[1, 0]
    trends_sorted = trends_df[trends_df['region_id'] != 0].sort_values('acceleration_gt_per_yr2')
    colors = [REGION_COLORS.get(rid, '#333') for rid in trends_sorted['region_id']]
    ax.barh(range(len(trends_sorted)), trends_sorted['acceleration_gt_per_yr2'],
            color=colors, alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(trends_sorted)))
    ax.set_yticklabels([f"{REGION_CODES.get(rid, '')}" for rid in trends_sorted['region_id']], fontsize=8)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Mass loss acceleration (Gt yr⁻²)')
    ax.set_title('(c) Regional mass loss acceleration')
    
    # Panel D: Sea level equivalent contribution
    ax = axes[1, 1]
    global_cumul = cumul_df[cumul_df['region_id'] == 0].sort_values('year')
    # Convert Gt to mm SLE: 1 Gt = 1/(362.5e6) mm SLE over ocean area
    # 362.5e6 km² ocean area, 1 Gt = 1e12 kg, spread over 362.5e6 km² = 1e12/(362.5e6*1e6*1e3) m = 1/(362.5e3) m
    # = 1/362500 mm ≈ 0.00276 mm
    gt_to_mm_sle = 1 / 362.5  # 1 Gt ≈ 1/362.5 mm SLE
    sle_mm = global_cumul['cumulative_gt'] * gt_to_mm_sle
    sle_err = global_cumul['cumulative_gt_err'] * gt_to_mm_sle
    
    ax.fill_between(global_cumul['year'], sle_mm - sle_err, sle_mm + sle_err,
                    alpha=0.3, color='#2ca02c')
    ax.plot(global_cumul['year'], sle_mm, '-', color='#2ca02c', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea level equivalent (mm)')
    ax.set_title('(d) Cumulative sea level contribution')
    
    fig.suptitle('Mass Loss Trends and Acceleration (2000–2023)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig7_acceleration_trends.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 7 saved: fig7_acceleration_trends.png")

# ============================================================
# Figure 8: Regional Time Series Grid
# ============================================================
def fig8_regional_timeseries_grid():
    """Grid of regional mass change time series."""
    regional_ts = ts_df[ts_df['region_id'] != 0].sort_values(['region_id', 'year'])
    region_ids = sorted(regional_ts['region_id'].unique())
    
    n_regions = len(region_ids)
    ncols = 4
    nrows = (n_regions + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4*nrows), sharex=True)
    axes = axes.flatten()
    
    for idx, region_id in enumerate(region_ids):
        ax = axes[idx]
        sub = regional_ts[regional_ts['region_id'] == region_id].sort_values('year')
        
        ax.fill_between(sub['year'],
                        sub['mass_change_gt'] - sub['mass_change_gt_err'],
                        sub['mass_change_gt'] + sub['mass_change_gt_err'],
                        alpha=0.2, color=REGION_COLORS.get(region_id, '#333'))
        ax.plot(sub['year'], sub['mass_change_gt'], 'o-',
                color=REGION_COLORS.get(region_id, '#333'),
                markersize=3, linewidth=1)
        ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
        ax.set_title(f'{REGION_CODES.get(region_id, "")} - {REGION_MAP.get(region_id, "")}',
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('Gt yr⁻¹', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused axes
    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Regional Glacier Mass Change Time Series (2000–2023)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig8_regional_timeseries_grid.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 8 saved: fig8_regional_timeseries_grid.png")

# ============================================================
# Figure 9: Method Agreement Scatter
# ============================================================
def fig9_method_agreement_scatter():
    """Scatter plots comparing method estimates."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    method_pairs = [
        ('altimetry_gt', 'gravimetry_gt', 'Altimetry vs Gravimetry'),
        ('altimetry_gt', 'demdiff_and_glaciological_gt', 'Altimetry vs DEM+Glaciological'),
        ('gravimetry_gt', 'demdiff_and_glaciological_gt', 'Gravimetry vs DEM+Glaciological')
    ]
    
    for idx, (m1, m2, title) in enumerate(method_pairs):
        ax = axes[idx]
        valid = agreement_df[m1].notna() & agreement_df[m2].notna()
        if valid.sum() > 0:
            sub = agreement_df[valid]
            ax.scatter(sub[m1], sub[m2], alpha=0.4, s=20, c='#1f77b4', edgecolors='k', linewidths=0.3)
            
            # 1:1 line
            lims = [min(sub[m1].min(), sub[m2].min()), max(sub[m1].max(), sub[m2].max())]
            ax.plot(lims, lims, 'r--', linewidth=1, label='1:1 line')
            
            # Correlation
            r, p = stats.pearsonr(sub[m1], sub[m2])
            ax.set_title(f'{title}\nr = {r:.2f}, n = {valid.sum()}', fontsize=10)
        
        ax.set_xlabel(f'{m1.replace("_gt", "").replace("_", " ").title()} (Gt yr⁻¹)')
        ax.set_ylabel(f'{m2.replace("_gt", "").replace("_", " ").title()} (Gt yr⁻¹)')
        ax.legend(fontsize=8)
    
    fig.suptitle('Inter-Method Agreement in Mass Change Estimates', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig9_method_agreement_scatter.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 9 saved: fig9_method_agreement_scatter.png")

# ============================================================
# Figure 10: Regional Contribution Pie / Donut
# ============================================================
def fig10_regional_contribution_donut():
    """Donut chart of regional contributions to global mass loss."""
    regional = regional_summary.sort_values('cumulative_gt')
    
    # Group small contributors
    threshold = 3.0  # percent
    major = regional[regional['pct_of_global'].abs() >= threshold].copy()
    minor_pct = regional[regional['pct_of_global'].abs() < threshold]['pct_of_global'].sum()
    minor_cumul = regional[regional['pct_of_global'].abs() < threshold]['cumulative_gt'].sum()
    
    if abs(minor_pct) > 0.1:
        major = pd.concat([major, pd.DataFrame({
            'region_id': [99], 'region_name': ['Other regions'],
            'cumulative_gt': [minor_cumul], 'pct_of_global': [minor_pct]
        })], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    labels = [f"{REGION_CODES.get(rid, '')} {rn}\n({pct:.1f}%)"
              for rid, rn, pct in zip(major['region_id'], major['region_name'], major['pct_of_global'])]
    colors = [REGION_COLORS.get(rid, '#cccccc') for rid in major['region_id']]
    
    wedges, texts = ax.pie(abs(major['cumulative_gt']), labels=labels, colors=colors,
                           startangle=90, pctdistance=0.85,
                           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    
    for text in texts:
        text.set_fontsize(9)
    
    # Center text
    ax.text(0, 0, f'Total\n{summary_stats["global_cumulative_gt"]:.0f} Gt',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_title('Regional Contributions to Global Glacier Mass Loss (2000–2023)',
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'fig10_regional_contribution_donut.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Figure 10 saved: fig10_regional_contribution_donut.png")

# ============================================================
# Execute all figures
# ============================================================
if __name__ == '__main__':
    print("Generating figures...")
    fig1_global_timeseries()
    fig2_regional_bar_chart()
    fig3_method_comparison()
    fig4_stacked_cumulative()
    fig5_specific_mass_change_heatmap()
    fig6_method_coverage_uncertainty()
    fig7_acceleration_trends()
    fig8_regional_timeseries_grid()
    fig9_method_agreement_scatter()
    fig10_regional_contribution_donut()
    print("\nAll figures generated successfully!")
