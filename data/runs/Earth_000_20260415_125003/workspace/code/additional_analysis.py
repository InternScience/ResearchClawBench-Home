#!/usr/bin/env python3
"""
Additional Analysis for GlaMBIE Data

This script creates additional visualizations and analyses for the GlaMBIE dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Define paths
DATA_DIR = Path('data/glambie/results/calendar_years')
OUTPUT_DIR = Path('outputs')
FIGURE_DIR = Path('report/images')

# Region names mapping
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

def create_uncertainty_analysis():
    """Create plot showing uncertainty evolution over time."""
    global_df = pd.read_csv(DATA_DIR / '0_global.csv')
    global_df['year'] = (global_df['start_dates'] + global_df['end_dates']) / 2
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Absolute and relative uncertainty
    ax = axes[0]
    years = global_df['year'].values
    errors = global_df['combined_gt_errors'].values
    
    ax.fill_between(years, errors, alpha=0.3, color='steelblue')
    ax.plot(years, errors, 'b-', linewidth=2, label='Annual uncertainty')
    ax.set_xlabel('Year')
    ax.set_ylabel('Uncertainty (Gt yr-1)')
    ax.set_title('Evolution of Mass Change Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Signal-to-noise ratio
    ax = axes[1]
    signal_to_noise = np.abs(global_df['combined_gt'].values) / global_df['combined_gt_errors'].values
    ax.plot(years, signal_to_noise, 'g-', linewidth=2)
    ax.axhline(y=2, color='r', linestyle='--', label='Significance threshold (2sigma)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('Mass Change Signal-to-Noise Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: uncertainty_analysis.png")

def create_sea_level_contribution_plot():
    """Create plot showing sea level contribution."""
    global_df = pd.read_csv(DATA_DIR / '0_global.csv')
    global_df['year'] = (global_df['start_dates'] + global_df['end_dates']) / 2
    
    # Convert to mm SLE (1 Gt = 0.00028 mm SLE)
    sle_conversion = 0.00028
    global_df['sle_mm'] = global_df['combined_gt'] * sle_conversion
    global_df['sle_cumulative'] = global_df['sle_mm'].cumsum()
    global_df['sle_error'] = global_df['combined_gt_errors'] * sle_conversion
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Annual contribution
    ax = axes[0]
    ax.fill_between(global_df['year'], 
                     global_df['sle_mm'] - global_df['sle_error'],
                     global_df['sle_mm'] + global_df['sle_error'],
                     alpha=0.3, color='steelblue')
    ax.plot(global_df['year'], global_df['sle_mm'], 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level Contribution (mm yr-1)')
    ax.set_title('Annual Glacier Contribution to Sea Level Rise')
    ax.grid(True, alpha=0.3)
    
    # Cumulative contribution
    ax = axes[1]
    ax.fill_between(global_df['year'],
                     global_df['sle_cumulative'] - global_df['sle_error'].cumsum(),
                     global_df['sle_cumulative'] + global_df['sle_error'].cumsum(),
                     alpha=0.3, color='darkgreen')
    ax.plot(global_df['year'], global_df['sle_cumulative'], 'g-', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Sea Level Contribution (mm)')
    ax.set_title('Cumulative Glacier Contribution to Sea Level Rise (2000-2023)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'sea_level_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: sea_level_contribution.png")

def create_interannual_variability_plot():
    """Create plot showing interannual variability."""
    global_df = pd.read_csv(DATA_DIR / '0_global.csv')
    global_df['year'] = (global_df['start_dates'] + global_df['end_dates']) / 2
    
    # Calculate 5-year moving average
    global_df['ma_5yr'] = global_df['combined_gt'].rolling(window=5, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(global_df['year'], global_df['combined_gt'], alpha=0.3, color='lightblue')
    ax.plot(global_df['year'], global_df['combined_gt'], 'b-', alpha=0.5, label='Annual')
    ax.plot(global_df['year'], global_df['ma_5yr'], 'r-', linewidth=2.5, label='5-year moving average')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Mass Change (Gt yr-1)')
    ax.set_title('Global Glacier Mass Change: Interannual Variability and Trends')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'interannual_variability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: interannual_variability.png")

def main():
    print("Additional GlaMBIE Analysis")
    print("=" * 40)
    
    create_uncertainty_analysis()
    create_sea_level_contribution_plot()
    create_interannual_variability_plot()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
