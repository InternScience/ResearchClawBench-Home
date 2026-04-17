#!/usr/bin/env python3
"""
GlaMBIE Figure Generation Script

Generates all figures for the research report:
1. Data overview plots
2. Main result figures (time series)
3. Comparison/validation plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
WORKSPACE_ROOT = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Earth_000_20260416_164538")
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
FIGURES_DIR = WORKSPACE_ROOT / "report" / "images"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Regional order for plotting
REGION_ORDER = [
    "Alaska", "Western Canada US", "Arctic Canada North", "Arctic Canada South",
    "Greenland Periphery", "Iceland", "Svalbard", "Scandinavia", "Russian Arctic",
    "North Asia", "Central Europe", "Caucasus Middle East", "Central Asia",
    "South Asia West", "South Asia East", "Low Latitudes", "Southern Andes",
    "New Zealand", "Antarctic and Subantarctic"
]


def load_data():
    """Load processed data."""
    regional = pd.read_csv(OUTPUTS_DIR / "regional_timeseries_v2.csv")
    global_df = pd.read_csv(OUTPUTS_DIR / "global_timeseries_v2.csv")
    return regional, global_df


def plot_data_overview_methods(regional):
    """Figure 1: Methods per region bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Count methods per region from original data
    # Load annual data to get method counts
    annual = pd.read_parquet(OUTPUTS_DIR / "annual_data.parquet")
    method_counts = annual.groupby(['region', 'method']).size().unstack(fill_value=0)
    
    # Reorder regions
    method_counts = method_counts.reindex(REGION_ORDER)
    
    method_counts.plot(kind='barh', stacked=True, ax=ax, figsize=(12, 8))
    ax.set_xlabel('Number of measurements')
    ax.set_ylabel('Region')
    ax.set_title('GlaMBIE Dataset: Measurements by Method and Region (2000-2023)')
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig01_methods_per_region.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_global_timeseries(global_df):
    """Figure 2: Global mass change time series."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = global_df['year'].values
    mwe = global_df['global_mwe'].values
    mwe_err = global_df['global_mwe_uncertainty'].values
    gt = global_df['global_Gt'].values
    gt_err = global_df['global_Gt_uncertainty'].values
    
    # Plot with dual y-axis
    color1 = 'tab:blue'
    color2 = 'tab:red'
    
    ax.fill_between(years, mwe - mwe_err, mwe + mwe_err, alpha=0.3, color=color1)
    ax.plot(years, mwe, 'o-', color=color1, label='Specific mass change (m w.e.)', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Specific mass change (m w.e./yr)', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    ax2 = ax.twinx()
    ax2.fill_between(years, gt - gt_err, gt + gt_err, alpha=0.3, color=color2)
    ax2.plot(years, gt, 's-', color=color2, label='Total mass change (Gt/yr)', linewidth=2)
    ax2.set_ylabel('Total mass change (Gt/yr)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    plt.title('Global Glacier Mass Change Time Series (2000-2023)')
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig02_global_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_regional_timeseries(regional):
    """Figure 3: Regional mass change time series (multi-panel)."""
    fig, axes = plt.subplots(5, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    regions = regional['region'].unique()
    
    for i, region in enumerate(REGION_ORDER):
        if i >= len(axes):
            break
        
        ax = axes[i]
        region_data = regional[regional['region'] == region]
        
        if len(region_data) == 0:
            continue
        
        years = region_data['midpoint_year'].values
        mwe = region_data['reconciled_mwe'].values
        err = region_data['reconciled_mwe_uncertainty'].values
        
        ax.fill_between(years, mwe - err, mwe + err, alpha=0.3)
        ax.plot(years, mwe, 'o-', markersize=3, linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(region, fontsize=9)
        ax.set_xlim(1999.5, 2023.5)
        
        if i >= 16:  # Bottom row
            ax.set_xlabel('Year')
        if i % 4 == 0:  # Left column
            ax.set_ylabel('m w.e./yr')
        
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for i in range(len(REGION_ORDER), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Regional Glacier Mass Change Time Series (2000-2023)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig03_regional_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_cumulative_mass_change(regional, global_df):
    """Figure 4: Cumulative mass change."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Global cumulative
    years = global_df['year'].values
    gt = global_df['global_Gt'].values
    gt_cumsum = gt.cumsum()
    gt_err = global_df['global_Gt_uncertainty'].values
    gt_err_cumsum = np.sqrt(np.cumsum(gt_err**2))
    
    ax1.fill_between(years, gt_cumsum - gt_err_cumsum, gt_cumsum + gt_err_cumsum, 
                     alpha=0.3, color='tab:red')
    ax1.plot(years, gt_cumsum, 'o-', color='tab:red', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative mass change (Gt)')
    ax1.set_title('Global Cumulative Mass Change (2000-2023)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Regional cumulative (final values)
    regional_sorted = regional.groupby('region')['reconciled_Gt'].sum().sort_values()
    
    colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(regional_sorted)))
    bars = ax2.barh(regional_sorted.index, regional_sorted.values, color=colors)
    ax2.set_xlabel('Cumulative mass change (Gt, 2000-2023)')
    ax2.set_title('Regional Cumulative Mass Change')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig04_cumulative_mass_change.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_method_comparison(regional):
    """Figure 5: Method intercomparison for selected regions."""
    # Load annual data to get individual method estimates
    annual = pd.read_parquet(OUTPUTS_DIR / "annual_data.parquet")
    annual_norm = annual.copy()
    
    # Normalize units (simplified version)
    REGION_AREAS_KM2 = {
        "Alaska": 76900, "Western Canada US": 27700, "Arctic Canada North": 110500,
        "Arctic Canada South": 39200, "Greenland Periphery": 76400, "Iceland": 10900,
        "Svalbard": 33300, "Scandinavia": 2900, "Russian Arctic": 28200,
        "North Asia": 19000, "Central Europe": 2900, "Caucasus Middle East": 2100,
        "Central Asia": 96000, "South Asia West": 35700, "South Asia East": 36500,
        "Low Latitudes": 2400, "Southern Andes": 25900, "New Zealand": 1200,
        "Antarctic and Subantarctic": 3200
    }
    
    for region in annual_norm['region'].unique():
        area = REGION_AREAS_KM2.get(region, 10000)
        gt_mask = (annual_norm['region'] == region) & (annual_norm['unit'].str.lower() == 'gt')
        if gt_mask.any():
            annual_norm.loc[gt_mask, 'changes_mwe'] = annual_norm.loc[gt_mask, 'changes'] / (area * 0.001)
        mwe_mask = (annual_norm['region'] == region) & (annual_norm['unit'].str.lower().isin(['mwe', 'm']))
        if mwe_mask.any():
            annual_norm.loc[mwe_mask, 'changes_mwe'] = annual_norm.loc[mwe_mask, 'changes']
    
    # Select a few representative regions
    select_regions = ["Alaska", "Svalbard", "Central Asia", "Southern Andes"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, region in enumerate(select_regions):
        ax = axes[idx]
        region_data = annual_norm[annual_norm['region'] == region]
        
        # Filter to 2000-2023
        region_data = region_data[(region_data['start_year'] >= 2000) & 
                                  (region_data['start_year'] <= 2023)]
        
        # Group by year and method
        for method in region_data['method'].unique():
            method_data = region_data[region_data['method'] == method]
            yearly_avg = method_data.groupby('start_year')['changes_mwe'].mean()
            ax.plot(yearly_avg.index, yearly_avg.values, 'o-', 
                   label=method, markersize=3, alpha=0.7)
        
        ax.set_title(region)
        ax.set_xlabel('Year')
        ax.set_ylabel('Mass change (m w.e./yr)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(1999.5, 2023.5)
    
    plt.suptitle('Method Intercomparison for Selected Regions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig05_method_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def plot_regional_summary_bar(regional):
    """Figure 6: Regional mean mass change bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean for each region
    regional_means = regional.groupby('region')['reconciled_mwe'].agg(['mean', 'std'])
    regional_means = regional_means.reindex(REGION_ORDER)
    
    regions = regional_means.index
    means = regional_means['mean'].values
    stds = regional_means['std'].values
    
    # Create bar chart
    colors = ['tab:red' if m < 0 else 'tab:blue' for m in means]
    bars = ax.barh(regions, means, xerr=stds, color=colors, alpha=0.7)
    
    ax.set_xlabel('Mean annual mass change (m w.e./yr)')
    ax.set_title('Regional Mean Annual Mass Change (2000-2023)')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "fig06_regional_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("GlaMBIE Figure Generation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    regional, global_df = load_data()
    
    # Generate figures
    print("\nGenerating figures...")
    
    print("\n1. Data overview plots:")
    plot_data_overview_methods(regional)
    
    print("\n2. Main result figures:")
    plot_global_timeseries(global_df)
    plot_regional_timeseries(regional)
    plot_cumulative_mass_change(regional, global_df)
    
    print("\n3. Comparison/validation plots:")
    plot_method_comparison(regional)
    plot_regional_summary_bar(regional)
    
    print("\n" + "=" * 70)
    print("Figure generation complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
