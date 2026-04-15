#!/usr/bin/env python3
"""
GlaMBIE Data Analysis Script

This script analyzes the Glacier Mass Balance Intercomparison Exercise (GlaMBIE) dataset
to produce regional and global glacial mass change time series from 2000-2023.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Define paths
DATA_DIR = Path('data/glambie/results/calendar_years')
OUTPUT_DIR = Path('outputs')
FIGURE_DIR = Path('report/images')

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Define region names mapping
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

# Region groups for analysis
REGION_GROUPS = {
    'Arctic': ['3_arctic_canada_north', '4_arctic_canada_south', '7_svalbard', '9_russian_arctic'],
    'North America': ['1_alaska', '2_western_canada_us'],
    'Europe': ['6_iceland', '8_scandinavia', '11_central_europe', '12_caucasus_middle_east'],
    'Asia': ['10_north_asia', '13_central_asia', '14_south_asia_west', '15_south_asia_east'],
    'Polar': ['5_greenland_periphery', '19_antarctic_and_subantarctic'],
    'Southern Hemisphere': ['17_southern_andes', '18_new_zealand'],
    'Other': ['16_low_latitudes']
}

def load_all_regions():
    """Load all regional data files."""
    all_data = {}
    
    for csv_file in sorted(DATA_DIR.glob('*.csv')):
        region_key = csv_file.stem
        df = pd.read_csv(csv_file)
        
        # Convert to hydrological year (middle of period)
        df['year'] = (df['start_dates'] + df['end_dates']) / 2
        df['region'] = region_key
        df['region_name'] = REGION_NAMES.get(region_key, region_key)
        
        all_data[region_key] = df
    
    return all_data

def compute_cumulative_mass_change(all_data):
    """Compute cumulative mass change from annual rates."""
    for key, df in all_data.items():
        # Sort by year
        df = df.sort_values('year').reset_index(drop=True)
        
        # Calculate cumulative mass change
        df['cumulative_gt'] = df['combined_gt'].cumsum()
        df['cumulative_mwe'] = df['combined_mwe'].cumsum()
        
        # Calculate cumulative error (quadrature sum)
        df['cumulative_gt_error'] = np.sqrt((df['combined_gt_errors']**2).cumsum())
        df['cumulative_mwe_error'] = np.sqrt((df['combined_mwe_errors']**2).cumsum())
        
        all_data[key] = df
    
    return all_data

def create_global_timeseries_plot(all_data):
    """Create global mass change time series plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    global_df = all_data['0_global']
    years = global_df['year'].values
    
    # Annual mass change in Gt
    ax = axes[0, 0]
    ax.fill_between(years, 
                     global_df['combined_gt'] - global_df['combined_gt_errors'],
                     global_df['combined_gt'] + global_df['combined_gt_errors'],
                     alpha=0.3, color='steelblue', label='Uncertainty')
    ax.plot(years, global_df['combined_gt'], 'b-', linewidth=2, label='Annual mass change')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mass Change (Gt yr⁻¹)')
    ax.set_title('Global Annual Glacier Mass Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative mass change in Gt
    ax = axes[0, 1]
    ax.fill_between(years,
                     global_df['cumulative_gt'] - global_df['cumulative_gt_error'],
                     global_df['cumulative_gt'] + global_df['cumulative_gt_error'],
                     alpha=0.3, color='steelblue')
    ax.plot(years, global_df['cumulative_gt'], 'b-', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Mass Change (Gt)')
    ax.set_title('Global Cumulative Glacier Mass Change (2000-2023)')
    ax.grid(True, alpha=0.3)
    
    # Specific mass balance (m w.e.)
    ax = axes[1, 0]
    ax.fill_between(years,
                     global_df['combined_mwe'] - global_df['combined_mwe_errors'],
                     global_df['combined_mwe'] + global_df['combined_mwe_errors'],
                     alpha=0.3, color='darkgreen')
    ax.plot(years, global_df['combined_mwe'], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Specific Mass Balance (m w.e. yr⁻¹)')
    ax.set_title('Global Mean Specific Mass Balance')
    ax.grid(True, alpha=0.3)
    
    # Cumulative specific mass balance
    ax = axes[1, 1]
    ax.fill_between(years,
                     global_df['cumulative_mwe'] - global_df['cumulative_mwe_error'],
                     global_df['cumulative_mwe'] + global_df['cumulative_mwe_error'],
                     alpha=0.3, color='darkgreen')
    ax.plot(years, global_df['cumulative_mwe'], 'g-', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Specific Mass Balance (m w.e.)')
    ax.set_title('Global Cumulative Mean Specific Mass Balance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'global_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: global_timeseries.png")

def create_regional_comparison_plot(all_data):
    """Create regional comparison plot."""
    # Get all regional data (excluding global)
    regional_data = []
    for key, df in all_data.items():
        if key != '0_global':
            # Get total cumulative change from 2000 to 2023
            total_change = df['cumulative_gt'].iloc[-1]
            total_error = df['cumulative_gt_error'].iloc[-1]
            mean_annual = df['combined_gt'].mean()
            glacier_area = df['glacier_area'].iloc[0]
            
            regional_data.append({
                'region': REGION_NAMES[key],
                'region_key': key,
                'cumulative_gt': total_change,
                'cumulative_error': total_error,
                'mean_annual_gt': mean_annual,
                'glacier_area': glacier_area
            })
    
    reg_df = pd.DataFrame(regional_data)
    reg_df = reg_df.sort_values('cumulative_gt')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot 1: Cumulative mass change by region
    ax = axes[0]
    colors = ['darkred' if x < 0 else 'darkgreen' for x in reg_df['cumulative_gt']]
    bars = ax.barh(reg_df['region'], reg_df['cumulative_gt'], color=colors, alpha=0.7)
    ax.errorbar(reg_df['cumulative_gt'], reg_df['region'], 
                xerr=reg_df['cumulative_error'], fmt='none', color='black', capsize=3)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Cumulative Mass Change 2000-2023 (Gt)')
    ax.set_title('Regional Cumulative Glacier Mass Change\n(2000-2023)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Mean annual mass change rate
    ax = axes[1]
    colors = ['darkred' if x < 0 else 'darkgreen' for x in reg_df['mean_annual_gt']]
    ax.barh(reg_df['region'], reg_df['mean_annual_gt'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Mean Annual Mass Change (Gt yr⁻¹)')
    ax.set_title('Regional Mean Annual Mass Change Rate')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regional_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: regional_comparison.png")
    
    return reg_df

def create_regional_timeseries_plot(all_data):
    """Create time series plots for selected key regions."""
    # Select key regions for visualization
    key_regions = [
        '1_alaska', '5_greenland_periphery', '3_arctic_canada_north',
        '17_southern_andes', '13_central_asia', '11_central_europe'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, region_key in enumerate(key_regions):
        ax = axes[idx]
        df = all_data[region_key]
        years = df['year'].values
        
        ax.fill_between(years,
                        df['cumulative_gt'] - df['cumulative_gt_error'],
                        df['cumulative_gt'] + df['cumulative_gt_error'],
                        alpha=0.3, color='steelblue')
        ax.plot(years, df['cumulative_gt'], 'b-', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Mass Change (Gt)')
        ax.set_title(REGION_NAMES[region_key])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cumulative Glacier Mass Change by Region (2000-2023)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regional_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: regional_timeseries.png")

def create_contribution_pie_chart(reg_df):
    """Create pie chart showing regional contributions to global mass loss."""
    # Get negative contributions only (mass loss)
    mass_loss = reg_df[reg_df['cumulative_gt'] < 0].copy()
    mass_loss['abs_change'] = -mass_loss['cumulative_gt']
    mass_loss = mass_loss.sort_values('abs_change', ascending=False)
    
    # Group smaller regions
    top_regions = mass_loss.head(8)
    other_loss = mass_loss.iloc[8:]['abs_change'].sum()
    
    labels = list(top_regions['region']) + ['Other Regions']
    sizes = list(top_regions['abs_change']) + [other_loss]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Regional Contributions to Global Glacier Mass Loss (2000-2023)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regional_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: regional_contributions.png")

def create_decadal_analysis_plot(all_data):
    """Create decadal analysis plot."""
    global_df = all_data['0_global'].copy()
    
    # Define decades
    global_df['decade'] = pd.cut(global_df['year'], 
                                  bins=[1999, 2010, 2020, 2025],
                                  labels=['2000-2009', '2010-2019', '2020-2023'])
    
    decadal_stats = global_df.groupby('decade').agg({
        'combined_gt': ['mean', 'std', 'sum'],
        'combined_mwe': ['mean', 'std']
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Decadal mean mass change
    ax = axes[0]
    decades = ['2000-2009', '2010-2019', '2020-2023']
    means = [decadal_stats[decadal_stats['decade']==d]['combined_gt']['mean'].values[0] 
             for d in decades]
    stds = [decadal_stats[decadal_stats['decade']==d]['combined_gt']['std'].values[0] 
            for d in decades]
    
    bars = ax.bar(decades, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'darkred'])
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Mean Annual Mass Change (Gt yr⁻¹)')
    ax.set_title('Decadal Mean Glacier Mass Change')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cumulative by decade
    ax = axes[1]
    sums = [decadal_stats[decadal_stats['decade']==d]['combined_gt']['sum'].values[0] 
            for d in decades]
    
    ax.bar(decades, sums, color=['steelblue', 'coral', 'darkred'])
    ax.set_ylabel('Cumulative Mass Change (Gt)')
    ax.set_title('Decadal Cumulative Glacier Mass Change')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'decadal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: decadal_analysis.png")
    
    return decadal_stats

def create_specific_mass_balance_map(all_data):
    """Create visualization of specific mass balance by region."""
    regional_data = []
    for key, df in all_data.items():
        if key != '0_global':
            mean_smb = df['combined_mwe'].mean()
            std_smb = df['combined_mwe'].std()
            
            regional_data.append({
                'region': REGION_NAMES[key],
                'mean_smb': mean_smb,
                'std_smb': std_smb
            })
    
    smb_df = pd.DataFrame(regional_data)
    smb_df = smb_df.sort_values('mean_smb')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(smb_df)))
    colors = ['darkgreen' if x > -0.3 else 'orange' if x > -0.6 else 'red' 
              for x in smb_df['mean_smb']]
    
    bars = ax.barh(smb_df['region'], smb_df['mean_smb'], color=colors, alpha=0.7)
    ax.errorbar(smb_df['mean_smb'], smb_df['region'], 
                xerr=smb_df['std_smb'], fmt='none', color='black', capsize=3, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Mean Specific Mass Balance (m w.e. yr⁻¹)')
    ax.set_title('Regional Mean Specific Mass Balance (2000-2023)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations
    ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5, label='-0.5 m w.e. yr⁻¹')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'specific_mass_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: specific_mass_balance.png")

def compute_summary_statistics(all_data):
    """Compute and save summary statistics."""
    global_df = all_data['0_global']
    
    # Global statistics
    total_mass_loss_gt = global_df['cumulative_gt'].iloc[-1]
    total_mass_loss_error = global_df['cumulative_gt_error'].iloc[-1]
    total_smb = global_df['cumulative_mwe'].iloc[-1]
    total_smb_error = global_df['cumulative_mwe_error'].iloc[-1]
    
    # Annual statistics
    mean_annual_loss = global_df['combined_gt'].mean()
    std_annual_loss = global_df['combined_gt'].std()
    
    # Sea level equivalent (1 Gt = 0.00028 mm SLE)
    sle_conversion = 0.00028  # mm per Gt
    total_sle = total_mass_loss_gt * sle_conversion
    total_sle_error = total_mass_loss_error * sle_conversion
    annual_sle = mean_annual_loss * sle_conversion
    
    # Decadal trends
    early_period = global_df[global_df['year'] < 2010]['combined_gt'].mean()
    late_period = global_df[global_df['year'] >= 2015]['combined_gt'].mean()
    acceleration = late_period - early_period
    
    stats = {
        'total_mass_loss_gt': total_mass_loss_gt,
        'total_mass_loss_error_gt': total_mass_loss_error,
        'total_sle_mm': total_sle,
        'total_sle_error_mm': total_sle_error,
        'mean_annual_loss_gt': mean_annual_loss,
        'std_annual_loss_gt': std_annual_loss,
        'annual_sle_mm': annual_sle,
        'total_specific_mass_balance_mwe': total_smb,
        'acceleration_gt_per_decade': acceleration,
        'period': '2000-2023'
    }
    
    # Save to file
    with open(OUTPUT_DIR / 'summary_statistics.txt', 'w') as f:
        f.write("GlaMBIE Global Glacier Mass Change Summary Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Period: {stats['period']}\n\n")
        f.write(f"Total Cumulative Mass Loss: {total_mass_loss_gt:.1f} ± {total_mass_loss_error:.1f} Gt\n")
        f.write(f"Total Sea Level Contribution: {total_sle:.2f} ± {total_sle_error:.2f} mm\n")
        f.write(f"Mean Annual Mass Loss: {mean_annual_loss:.1f} ± {std_annual_loss:.1f} Gt yr⁻¹\n")
        f.write(f"Annual Sea Level Contribution: {annual_sle:.3f} mm yr⁻¹\n")
        f.write(f"Total Specific Mass Balance: {total_smb:.2f} ± {total_smb_error:.2f} m w.e.\n")
        f.write(f"Acceleration (2015-2023 vs 2000-2009): {acceleration:.1f} Gt yr⁻¹\n")
    
    print("\nSummary Statistics:")
    print(f"Total Mass Loss (2000-2023): {total_mass_loss_gt:.1f} ± {total_mass_loss_error:.1f} Gt")
    print(f"Sea Level Contribution: {total_sle:.2f} ± {total_sle_error:.2f} mm")
    print(f"Mean Annual Loss: {mean_annual_loss:.1f} ± {std_annual_loss:.1f} Gt yr⁻¹")
    
    return stats

def export_timeseries_data(all_data):
    """Export processed time series data."""
    # Global data
    global_df = all_data['0_global']
    global_df.to_csv(OUTPUT_DIR / 'global_timeseries.csv', index=False)
    
    # Regional summary
    regional_summary = []
    for key, df in all_data.items():
        if key != '0_global':
            regional_summary.append({
                'region': REGION_NAMES[key],
                'region_code': key,
                'cumulative_mass_change_gt': df['cumulative_gt'].iloc[-1],
                'cumulative_error_gt': df['cumulative_gt_error'].iloc[-1],
                'mean_annual_change_gt': df['combined_gt'].mean(),
                'total_specific_mass_balance_mwe': df['cumulative_mwe'].iloc[-1],
                'glacier_area_km2': df['glacier_area'].iloc[0]
            })
    
    reg_summary_df = pd.DataFrame(regional_summary)
    reg_summary_df = reg_summary_df.sort_values('cumulative_mass_change_gt')
    reg_summary_df.to_csv(OUTPUT_DIR / 'regional_summary.csv', index=False)
    
    print("Exported: global_timeseries.csv")
    print("Exported: regional_summary.csv")
    
    return reg_summary_df

def main():
    """Main analysis function."""
    print("GlaMBIE Glacier Mass Change Analysis")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading regional data...")
    all_data = load_all_regions()
    print(f"   Loaded {len(all_data)} regions")
    
    # Compute cumulative changes
    print("\n2. Computing cumulative mass changes...")
    all_data = compute_cumulative_mass_change(all_data)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    create_global_timeseries_plot(all_data)
    reg_df = create_regional_comparison_plot(all_data)
    create_regional_timeseries_plot(all_data)
    create_contribution_pie_chart(reg_df)
    decadal_stats = create_decadal_analysis_plot(all_data)
    create_specific_mass_balance_map(all_data)
    
    # Compute statistics
    print("\n4. Computing summary statistics...")
    stats = compute_summary_statistics(all_data)
    
    # Export data
    print("\n5. Exporting processed data...")
    reg_summary = export_timeseries_data(all_data)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Data saved to: {OUTPUT_DIR}")
    
    return all_data, stats, reg_summary

if __name__ == "__main__":
    all_data, stats, reg_summary = main()
