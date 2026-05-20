"""
Comprehensive analysis and figure generation for GlaMBIE glacier mass change study.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
import os
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
IMG = os.path.join(BASE, "report", "images")
os.makedirs(IMG, exist_ok=True)

# Load data
global_df = pd.read_csv(os.path.join(OUT, "global_time_series.csv"))
regional_df = pd.read_csv(os.path.join(OUT, "regional_time_series.csv"))
summary_df = pd.read_csv(os.path.join(OUT, "regional_summary.csv"))
hydro_df = pd.read_csv(os.path.join(OUT, "hydrological_time_series.csv"))

REGION_NAMES = {
    1: "Alaska", 2: "Western Canada & US", 3: "Arctic Canada North",
    4: "Arctic Canada South", 5: "Greenland Periphery", 6: "Iceland",
    7: "Svalbard", 8: "Scandinavia", 9: "Russian Arctic", 10: "North Asia",
    11: "Central Europe", 12: "Caucasus & Middle East", 13: "Central Asia",
    14: "South Asia West", 15: "South Asia East", 16: "Low Latitudes",
    17: "Southern Andes", 18: "New Zealand", 19: "Antarctic & Subantarctic",
}

# Color scheme for regions
REGION_COLORS = plt.cm.tab20(np.linspace(0, 1, 19))

# Set style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def fig1_global_time_series():
    """Figure 1: Global mass change time series (Gt and m w.e.) with uncertainties."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    years = (global_df["start_dates"] + global_df["end_dates"]) / 2
    
    # Panel a: Total mass change in Gt
    ax1.bar(years, global_df["combined_gt"], width=0.8, color='steelblue', 
            alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax1.errorbar(years, global_df["combined_gt"], yerr=global_df["combined_gt_errors"],
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax1.set_ylabel('Mass change (Gt yr⁻¹)')
    ax1.set_title('a) Global glacier mass change rate')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Add trend line
    slope, intercept, r, p, se = stats.linregress(years, global_df["combined_gt"])
    trend_line = slope * years + intercept
    ax1.plot(years, trend_line, 'r--', linewidth=1.5, label=f'Linear trend: {slope:.1f} Gt/yr²')
    ax1.legend(loc='upper right')
    
    # Panel b: Specific mass change in m w.e.
    ax2.bar(years, global_df["combined_mwe"], width=0.8, color='coral', 
            alpha=0.7, edgecolor='darkred', linewidth=0.5)
    ax2.errorbar(years, global_df["combined_mwe"], yerr=global_df["combined_mwe_errors"],
                fmt='none', ecolor='black', capsize=2, capthick=0.8, linewidth=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax2.set_ylabel('Specific mass change (m w.e. yr⁻¹)')
    ax2.set_xlabel('Year')
    ax2.set_title('b) Global specific mass change rate')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Add trend line
    slope2, intercept2, _, _, _ = stats.linregress(years, global_df["combined_mwe"])
    trend_line2 = slope2 * years + intercept2
    ax2.plot(years, trend_line2, 'r--', linewidth=1.5, label=f'Linear trend: {slope2*1000:.2f} mm w.e./yr²')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_global_time_series.png"))
    plt.close()
    print("Figure 1 saved.")


def fig2_cumulative_mass_change():
    """Figure 2: Cumulative global mass change."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    years = (global_df["start_dates"] + global_df["end_dates"]) / 2
    cum_gt = global_df["combined_gt"].cumsum()
    cum_mwe = global_df["combined_mwe"].cumsum()
    
    # Error propagation for cumulative (sum of errors in quadrature)
    cum_err_gt = np.sqrt((global_df["combined_gt_errors"]**2).cumsum())
    
    # Panel a: Cumulative Gt
    ax1.fill_between(years, cum_gt - cum_err_gt, cum_gt + cum_err_gt, 
                     alpha=0.3, color='steelblue', label='Uncertainty (1σ)')
    ax1.plot(years, cum_gt, 'b-o', markersize=4, linewidth=2, label='GlaMBIE combined')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Cumulative mass change (Gt)')
    ax1.set_title('a) Cumulative global glacier mass change')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Add SLE equivalent text
    total_cum = cum_gt.iloc[-1]
    sle = total_cum / 362.5  # 1 mm SLE ≈ 362.5 Gt
    ax1.text(0.98, 0.05, f'Total: {total_cum:.0f} ± {cum_err_gt.iloc[-1]:.0f} Gt\n(~{sle:.1f} mm SLE)',
            transform=ax1.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel b: Specific mass change cumulative
    cum_err_mwe = np.sqrt((global_df["combined_mwe_errors"]**2).cumsum())
    ax2.fill_between(years, cum_mwe - cum_err_mwe, cum_mwe + cum_err_mwe,
                     alpha=0.3, color='coral', label='Uncertainty (1σ)')
    ax2.plot(years, cum_mwe, 'r-o', markersize=4, linewidth=2, label='GlaMBIE combined')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Cumulative specific mass change (m w.e.)')
    ax2.set_xlabel('Year')
    ax2.set_title('b) Cumulative global specific mass change')
    ax2.legend(loc='upper left')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_cumulative_mass_change.png"))
    plt.close()
    print("Figure 2 saved.")


def fig3_regional_specific_mass_change():
    """Figure 3: Regional specific mass change rates as bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Exclude global (id=0)
    reg_summary = summary_df[summary_df["region_id"] > 0].copy()
    reg_summary = reg_summary.sort_values("mean_specific_mass_change_mwe_yr")
    
    colors = ['#d73027' if v < -0.5 else '#fc8d59' if v < -0.3 else '#fee090' if v < -0.1 
              else '#91cf60' if v < 0 else '#1a9850' 
              for v in reg_summary["mean_specific_mass_change_mwe_yr"]]
    
    bars = ax.barh(range(len(reg_summary)), reg_summary["mean_specific_mass_change_mwe_yr"],
                   xerr=reg_summary["mean_specific_mass_change_error_mwe_yr"],
                   color=colors, edgecolor='black', linewidth=0.5, capsize=3)
    
    ax.set_yticks(range(len(reg_summary)))
    ax.set_yticklabels([f"{r['region_name']} ({r['region_code']})" 
                        for _, r in reg_summary.iterrows()], fontsize=9)
    ax.set_xlabel('Mean specific mass change (m w.e. yr⁻¹)')
    ax.set_title('Mean regional specific mass change rates (2000–2023)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Add value labels
    for i, (_, row) in enumerate(reg_summary.iterrows()):
        val = row["mean_specific_mass_change_mwe_yr"]
        ax.text(val - 0.03 if val < 0 else val + 0.01, i, f'{val:.3f}', 
               va='center', ha='right' if val < 0 else 'left', fontsize=7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_regional_specific_mass_change.png"))
    plt.close()
    print("Figure 3 saved.")


def fig4_regional_contribution_gt():
    """Figure 4: Regional contribution to total mass loss in Gt."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    reg_summary = summary_df[summary_df["region_id"] > 0].copy()
    reg_summary = reg_summary.sort_values("total_mass_change_gt_2000_2023")
    
    # Panel a: Absolute contributions
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(reg_summary)))
    ax1.barh(range(len(reg_summary)), reg_summary["total_mass_change_gt_2000_2023"],
             color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(reg_summary)))
    ax1.set_yticklabels([f"{r['region_name']}" for _, r in reg_summary.iterrows()], fontsize=8)
    ax1.set_xlabel('Total mass change 2000–2023 (Gt)')
    ax1.set_title('a) Regional contributions to total mass change')
    ax1.axvline(0, color='black', linewidth=0.8)
    
    # Panel b: Pie chart of contributions (top 7 + others)
    total_loss = abs(reg_summary["total_mass_change_gt_2000_2023"].sum())
    reg_summary_sorted = reg_summary.sort_values("total_mass_change_gt_2000_2023")
    
    top_n = 7
    top_regions = reg_summary_sorted.head(top_n)
    others_gt = reg_summary_sorted.iloc[top_n:]["total_mass_change_gt_2000_2023"].sum()
    
    labels = [r['region_name'] for _, r in top_regions.iterrows()] + ["Others"]
    sizes = [abs(v) for v in top_regions["total_mass_change_gt_2000_2023"]] + [abs(others_gt)]
    pcts = [s/total_loss*100 for s in sizes]
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax2.pie(sizes, labels=None, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    ax2.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax2.set_title('b) Share of global mass loss by region')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_regional_contribution_gt.png"))
    plt.close()
    print("Figure 4 saved.")


def fig5_method_comparison():
    """Figure 5: Comparison of observational methods from hydrological year data."""
    # Extract method-specific data
    methods = {
        'altimetry': ('altimetry_gt', 'altimetry_gt_errors'),
        'gravimetry': ('gravimetry_gt', 'gravimetry_gt_errors'),
        'demdiff_glaciological': ('demdiff_and_glaciological_gt', 'demdiff_and_glaciological_gt_errors'),
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Global analysis - aggregate across regions
    for i, (method_name, (gt_col, err_col)) in enumerate(methods.items()):
        ax = axes[i]
        
        # Get data for each region and plot
        for region_id in range(1, 20):
            region_data = hydro_df[hydro_df["region_id"] == region_id]
            if gt_col in region_data.columns and region_data[gt_col].notna().any():
                years = (region_data["start_dates"] + region_data["end_dates"]) / 2
                valid = region_data[gt_col].notna()
                ax.plot(years[valid], region_data.loc[valid, gt_col], 
                       '-', color=REGION_COLORS[region_id-1], alpha=0.6, linewidth=0.8)
        
        # Add global combined for reference
        # Aggregate all regions' method data
        method_by_year = {}
        for year in range(2000, 2024):
            mask = (hydro_df["start_dates"] >= year-0.5) & (hydro_df["end_dates"] <= year+1.5)
            region_data = hydro_df[mask]
            if gt_col in region_data.columns and region_data[gt_col].notna().any():
                total_gt = region_data[gt_col].sum()
                total_err = np.sqrt((region_data.loc[region_data[gt_col].notna(), err_col]**2).sum())
                method_by_year[year] = (total_gt, total_err)
        
        if method_by_year:
            yrs = sorted(method_by_year.keys())
            vals = [method_by_year[y][0] for y in yrs]
            errs = [method_by_year[y][1] for y in yrs]
            ax.errorbar(yrs, vals, yerr=errs, fmt='k-o', markersize=4, linewidth=2,
                       capsize=3, label='Regional sum', zorder=10)
        
        ax.set_ylabel('Mass change (Gt yr⁻¹)')
        ax.set_title(f'{method_name.replace("_", " + ").title()}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    axes[-1].set_xlabel('Year')
    plt.suptitle('Comparison of observational methods: regional mass change rates', y=1.01, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_method_comparison.png"))
    plt.close()
    print("Figure 5 saved.")


def fig6_regional_timeseries_heatmap():
    """Figure 6: Heatmap of regional specific mass change over time."""
    # Pivot data: rows=regions, columns=years
    years = (regional_df["start_dates"] + regional_df["end_dates"]).astype(int) / 2
    regional_df["mid_year"] = years
    
    # Create matrix
    region_ids = sorted(regional_df["region_id"].unique())
    year_vals = sorted(regional_df["mid_year"].unique())
    
    matrix = np.full((len(region_ids), len(year_vals)), np.nan)
    for i, rid in enumerate(region_ids):
        for j, yr in enumerate(year_vals):
            mask = regional_df["mid_year"] == yr
            reg_mask = mask & (regional_df["region_id"] == rid)
            if reg_mask.any():
                matrix[i, j] = regional_df.loc[reg_mask, "combined_mwe"].values[0]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Custom diverging colormap
    vmax = 1.5
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    
    # Labels
    region_labels = [f"{REGION_NAMES[rid]}" for rid in region_ids]
    ax.set_yticks(range(len(region_ids)))
    ax.set_yticklabels(region_labels, fontsize=8)
    
    # Show every other year
    tick_positions = list(range(0, len(year_vals), 2))
    tick_labels = [str(int(year_vals[p])) for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Specific mass change (m w.e. yr⁻¹)')
    
    ax.set_title('Regional specific mass change rates (2000–2023)\n(Calendar years, GlaMBIE combined estimates)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig6_regional_heatmap.png"))
    plt.close()
    print("Figure 6 saved.")


def fig7_acceleration_analysis():
    """Figure 7: Acceleration analysis - mass loss rate trends by period."""
    # Split into periods
    periods = {
        '2000-2005': (2000, 2005),
        '2006-2010': (2006, 2010),
        '2011-2015': (2011, 2015),
        '2016-2019': (2016, 2019),
        '2020-2023': (2020, 2023),
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel a: Global period comparison
    period_names = list(periods.keys())
    period_means = []
    period_errs = []
    
    for pname, (y0, y1) in periods.items():
        mask = (global_df["start_dates"] >= y0) & (global_df["end_dates"] <= y1 + 1)
        subset = global_df[mask]
        if len(subset) > 0:
            mean_gt = subset["combined_gt"].mean()
            err_gt = np.sqrt((subset["combined_gt_errors"]**2).sum()) / len(subset)
            period_means.append(mean_gt)
            period_errs.append(err_gt)
        else:
            period_means.append(0)
            period_errs.append(0)
    
    x = np.arange(len(period_names))
    bars = ax1.bar(x, period_means, yerr=period_errs, color='steelblue', 
                   edgecolor='navy', capsize=5, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_names, rotation=30)
    ax1.set_ylabel('Mean mass change rate (Gt yr⁻¹)')
    ax1.set_title('a) Global mean mass change by period')
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # Add values
    for bar, val, err in zip(bars, period_means, period_errs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() - err - 15,
                f'{val:.0f}', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Panel b: Top regions period comparison
    top_regions = [1, 5, 3, 4, 17, 9, 7]  # Alaska, GRL, ACN, ACS, SAN, RUA, SJM
    n_periods = len(period_names)
    n_regions = len(top_regions)
    bar_width = 0.12
    
    for i, rid in enumerate(top_regions):
        region_data = regional_df[regional_df["region_id"] == rid]
        region_means = []
        for pname, (y0, y1) in periods.items():
            mask = (region_data["start_dates"] >= y0) & (region_data["end_dates"] <= y1 + 1)
            subset = region_data[mask]
            region_means.append(subset["combined_gt"].mean() if len(subset) > 0 else 0)
        
        offset = (i - n_regions/2 + 0.5) * bar_width
        ax2.bar(x + offset, region_means, bar_width, 
               color=REGION_COLORS[rid-1], label=REGION_NAMES[rid], edgecolor='black', linewidth=0.3)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(period_names, rotation=30)
    ax2.set_ylabel('Mean mass change rate (Gt yr⁻¹)')
    ax2.set_title('b) Regional mass change by period (top contributing regions)')
    ax2.legend(fontsize=7, loc='lower left', ncol=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig7_acceleration_analysis.png"))
    plt.close()
    print("Figure 7 saved.")


def fig8_area_vs_mass_loss():
    """Figure 8: Relationship between glacier area and mass loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    reg_summary = summary_df[summary_df["region_id"] > 0].copy()
    
    # Panel a: Area vs total mass loss
    ax1.scatter(reg_summary["mean_glacier_area_km2"], 
               abs(reg_summary["total_mass_change_gt_2000_2023"]),
               s=60, c=reg_summary["mean_specific_mass_change_mwe_yr"],
               cmap='RdBu_r', edgecolors='black', linewidth=0.5, vmin=-1.2, vmax=0)
    
    for _, row in reg_summary.iterrows():
        ax1.annotate(row["region_code"], 
                    (row["mean_glacier_area_km2"], abs(row["total_mass_change_gt_2000_2023"])),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')
    
    ax1.set_xlabel('Mean glacier area (km²)')
    ax1.set_ylabel('Total mass loss 2000–2023 (Gt)')
    ax1.set_title('a) Glacier area vs total mass loss')
    cbar = plt.colorbar(ax1.collections[0], ax=ax1, shrink=0.8)
    cbar.set_label('Specific mass change (m w.e. yr⁻¹)')
    
    # Panel b: Area vs specific mass loss (rate)
    ax2.scatter(reg_summary["mean_glacier_area_km2"],
               abs(reg_summary["mean_specific_mass_change_mwe_yr"]),
               s=60, c='coral', edgecolors='black', linewidth=0.5)
    
    for _, row in reg_summary.iterrows():
        ax2.annotate(row["region_code"],
                    (row["mean_glacier_area_km2"], abs(row["mean_specific_mass_change_mwe_yr"])),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')
    
    ax2.set_xlabel('Mean glacier area (km²)')
    ax2.set_ylabel('Mean specific mass loss (m w.e. yr⁻¹)')
    ax2.set_title('b) Glacier area vs specific mass loss rate')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig8_area_vs_mass_loss.png"))
    plt.close()
    print("Figure 8 saved.")


def fig9_method_comparison_regional():
    """Figure 9: Method agreement for regions with multiple data sources."""
    # For each region, compute mean method rates and compare
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    regions_with_multi = []
    for rid in range(1, 20):
        region_hydro = hydro_df[hydro_df["region_id"] == rid]
        # Check which methods have data
        has_alt = region_hydro["altimetry_gt"].notna().any()
        has_grav = region_hydro["gravimetry_gt"].notna().any()
        has_dem = region_hydro["demdiff_and_glaciological_gt"].notna().any()
        if sum([has_alt, has_grav, has_dem]) >= 2:
            regions_with_multi.append(rid)
    
    # Select top 6 regions with most methods
    regions_to_plot = regions_with_multi[:6]
    
    for idx, rid in enumerate(regions_to_plot):
        ax = axes[idx]
        region_hydro = hydro_df[hydro_df["region_id"] == rid]
        years = (region_hydro["start_dates"] + region_hydro["end_dates"]) / 2
        
        methods_info = [
            ("altimetry_gt", "altimetry_gt_errors", "Altimetry", "blue"),
            ("gravimetry_gt", "gravimetry_gt_errors", "Gravimetry", "red"),
            ("demdiff_and_glaciological_gt", "demdiff_and_glaciological_gt_errors", 
             "DEM diff + Glaciological", "green"),
        ]
        
        for gt_col, err_col, label, color in methods_info:
            valid = region_hydro[gt_col].notna()
            if valid.any():
                ax.errorbar(years[valid], region_hydro.loc[valid, gt_col],
                           yerr=region_hydro.loc[valid, err_col],
                           fmt='o-', color=color, markersize=3, linewidth=1,
                           capsize=2, alpha=0.7, label=label)
        
        # Combined
        ax.plot(years, region_hydro["combined_gt"], 'k-', linewidth=2, 
               alpha=0.5, label='Combined', zorder=10)
        
        ax.set_title(REGION_NAMES[rid], fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel('Mass change (Gt yr⁻¹)')
        ax.legend(fontsize=7, loc='best')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    plt.suptitle('Method comparison for selected regions', y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig9_method_comparison_regional.png"))
    plt.close()
    print("Figure 9 saved.")


def fig10_validation_comparison():
    """Figure 10: Comparison with published estimates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Comparison with literature values
    # Hugonnet et al. 2021: 267 ± 16 Gt/yr for 2000-2019
    # Zemp et al. 2019: 335 ± 144 Gt/yr for 2006-2016
    # Rounce et al. 2023: projections calibrated with observations
    
    # Panel a: Global annual rate comparison
    studies = ['Hugonnet et al.\n(2000-2019)', 'Zemp et al.\n(2006-2016)', 
               'GlaMBIE\n(2000-2023)', 'GlaMBIE\n(2006-2016)']
    
    # Compute GlaMBIE rates for matching periods
    mask_2000_2019 = (global_df["start_dates"] >= 2000) & (global_df["end_dates"] <= 2020)
    rate_2000_2019 = global_df.loc[mask_2000_2019, "combined_gt"].mean()
    err_2000_2019 = np.sqrt((global_df.loc[mask_2000_2019, "combined_gt_errors"]**2).sum()) / mask_2000_2019.sum()
    
    mask_2006_2016 = (global_df["start_dates"] >= 2006) & (global_df["end_dates"] <= 2017)
    rate_2006_2016 = global_df.loc[mask_2006_2016, "combined_gt"].mean()
    err_2006_2016 = np.sqrt((global_df.loc[mask_2006_2016, "combined_gt_errors"]**2).sum()) / mask_2006_2016.sum()
    
    rates = [-267, -335, rate_2000_2019, rate_2006_2016]
    errors = [16, 144, err_2000_2019, err_2006_2016]
    colors = ['gray', 'gray', 'steelblue', 'coral']
    
    x = np.arange(len(studies))
    bars = ax1.bar(x, [abs(r) for r in rates], yerr=errors, color=colors,
                   edgecolor='black', capsize=5, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(studies, fontsize=8)
    ax1.set_ylabel('Mean mass loss rate (Gt yr⁻¹)')
    ax1.set_title('a) Comparison of global mass loss estimates')
    
    for bar, rate, err in zip(bars, rates, errors):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + err + 5,
                f'{abs(rate):.0f}±{err:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Panel b: Sea level equivalent
    # 1 mm SLE = 362.5 Gt
    sle_rates = [abs(r) / 362.5 for r in rates]
    sle_errs = [e / 362.5 for e in errors]
    
    bars2 = ax2.bar(x, sle_rates, yerr=sle_errs, color=colors,
                    edgecolor='black', capsize=5, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(studies, fontsize=8)
    ax2.set_ylabel('Sea level equivalent (mm yr⁻¹)')
    ax2.set_title('b) Sea level contribution rate')
    
    for bar, sle, err in zip(bars2, sle_rates, sle_errs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + err + 0.01,
                f'{sle:.2f}±{err:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMG, "fig10_validation_comparison.png"))
    plt.close()
    print("Figure 10 saved.")


def compute_statistics():
    """Compute and save key statistics for the report."""
    stats_out = {}
    
    # Global totals
    total_gt = global_df["combined_gt"].sum()
    total_err = np.sqrt((global_df["combined_gt_errors"]**2).sum())
    stats_out["global_total_gt"] = round(float(total_gt), 1)
    stats_out["global_total_err_gt"] = round(float(total_err), 1)
    stats_out["global_total_sle_mm"] = round(float(abs(total_gt) / 362.5), 1)
    stats_out["global_annual_mean_gt"] = round(float(global_df["combined_gt"].mean()), 1)
    stats_out["global_annual_mean_err_gt"] = round(float(global_df["combined_gt_errors"].mean()), 1)
    stats_out["global_annual_mean_mwe"] = round(float(global_df["combined_mwe"].mean()), 4)
    stats_out["global_annual_mean_err_mwe"] = round(float(global_df["combined_mwe_errors"].mean()), 4)
    stats_out["mean_glacier_area_km2"] = round(float(global_df["glacier_area"].mean()), 0)
    
    # Linear trend
    years = (global_df["start_dates"] + global_df["end_dates"]) / 2
    slope_gt, _, r_val, p_val, se = stats.linregress(years, global_df["combined_gt"])
    stats_out["linear_trend_gt_per_yr2"] = round(float(slope_gt), 2)
    stats_out["trend_r_squared"] = round(float(r_val**2), 3)
    stats_out["trend_p_value"] = float(p_val)
    
    slope_mwe, _, r_val_mwe, _, _ = stats.linregress(years, global_df["combined_mwe"])
    stats_out["linear_trend_mwe_per_yr2"] = round(float(slope_mwe), 4)
    
    # Period analysis
    periods = {
        "2000-2005": (2000, 2006),
        "2006-2010": (2006, 2011),
        "2011-2015": (2011, 2016),
        "2016-2019": (2016, 2020),
        "2020-2023": (2020, 2024),
    }
    
    period_stats = {}
    for pname, (y0, y1) in periods.items():
        mask = (global_df["start_dates"] >= y0) & (global_df["end_dates"] <= y1)
        subset = global_df[mask]
        if len(subset) > 0:
            period_stats[pname] = {
                "mean_gt": round(float(subset["combined_gt"].mean()), 1),
                "sum_gt": round(float(subset["combined_gt"].sum()), 1),
                "mean_mwe": round(float(subset["combined_mwe"].mean()), 4),
            }
    stats_out["period_analysis"] = period_stats
    
    # Regional ranking
    reg_summary = summary_df[summary_df["region_id"] > 0].copy()
    reg_summary = reg_summary.sort_values("total_mass_change_gt_2000_2023")
    
    stats_out["top_contributors_gt"] = [
        {"region": row["region_name"], "code": row["region_code"], 
         "total_gt": row["total_mass_change_gt_2000_2023"],
         "area_km2": row["mean_glacier_area_km2"]}
        for _, row in reg_summary.head(7).iterrows()
    ]
    
    # Most negative specific mass change
    reg_specific = reg_summary.sort_values("mean_specific_mass_change_mwe_yr")
    stats_out["most_negative_specific"] = [
        {"region": row["region_name"], "code": row["region_code"],
         "rate_mwe": row["mean_specific_mass_change_mwe_yr"]}
        for _, row in reg_specific.head(5).iterrows()
    ]
    
    # Cumulative mass change
    cum_gt = global_df["combined_gt"].sum()
    cum_err = np.sqrt((global_df["combined_gt_errors"]**2).sum())
    stats_out["cumulative_2000_2023_gt"] = round(float(cum_gt), 1)
    stats_out["cumulative_2000_2023_err_gt"] = round(float(cum_err), 1)
    stats_out["cumulative_sle_mm"] = round(float(abs(cum_gt) / 362.5), 1)
    
    # Method comparison from hydrological data
    for method in ["altimetry", "gravimetry", "demdiff_and_glaciological"]:
        gt_col = f"{method}_gt"
        err_col = f"{method}_gt_errors"
        if gt_col in hydro_df.columns:
            valid = hydro_df[gt_col].notna()
            if valid.any():
                # Aggregate by year
                yearly_totals = []
                for year in range(2000, 2024):
                    yr_mask = (hydro_df["start_dates"] >= year-0.5) & (hydro_df["end_dates"] <= year+1.5)
                    yr_data = hydro_df[yr_mask & valid]
                    if len(yr_data) > 0:
                        yearly_totals.append(yr_data[gt_col].sum())
                
                if yearly_totals:
                    stats_out[f"{method}_mean_annual_gt"] = round(float(np.mean(yearly_totals)), 1)
    
    # Save stats
    with open(os.path.join(OUT, "analysis_statistics.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    
    print("Statistics saved.")
    return stats_out


if __name__ == "__main__":
    print("Generating figures...")
    fig1_global_time_series()
    fig2_cumulative_mass_change()
    fig3_regional_specific_mass_change()
    fig4_regional_contribution_gt()
    fig5_method_comparison()
    fig6_regional_timeseries_heatmap()
    fig7_acceleration_analysis()
    fig8_area_vs_mass_loss()
    fig9_method_comparison_regional()
    fig10_validation_comparison()
    
    print("\nComputing statistics...")
    stats = compute_statistics()
    
    print("\nAll figures and statistics generated successfully!")
    print(json.dumps(stats, indent=2))
