#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Region name mapping
region_map = {
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
    '10_north_asia': 'Northern Asia',
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

data_dir = Path('data/glambie/results/calendar_years')
files = list(data_dir.glob('*.csv'))

dfs = {}
for f in files:
    region = f.stem
    if region in region_map:
        df = pd.read_csv(f)
        df['year'] = df['start_dates'] + 0.5  # mid year
        df['region'] = region_map[region]
        dfs[region] = df

# Save regional data
regional_data = pd.concat([df for df in dfs.values() if df['region'].iloc[0] != 'Global'])
regional_data.to_csv('outputs/regional_time_series.csv', index=False)

global_df = dfs['0_global']
global_df.to_csv('outputs/global_time_series.csv', index=False)

# Compute totals 2000-2023
mask = (global_df['start_dates'] >= 2000) & (global_df['start_dates'] < 2023)
total_gt = global_df.loc[mask, 'combined_gt'].sum()
total_mwe = total_gt / global_df.loc[mask, 'glacier_area'].mean() * 1e-3  # rough avg

print(f'Total mass loss 2000-2023: {total_gt:.1f} Gt')
print(f'Average rate: {total_gt / 23:.1f} Gt/yr')