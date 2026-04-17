#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Comprehensive Analysis
Phase 1: Data Exploration & Quality Assessment
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_003_20260416_185858'
DATA = os.path.join(BASE, 'data', 'HEEW_Mini-Dataset')
OUT = os.path.join(BASE, 'outputs')

# Load all building energy data
buildings = {}
for i in range(1, 11):
    fname = f'BN{i:03d}_energy.csv'
    buildings[f'BN{i:03d}'] = pd.read_csv(os.path.join(DATA, fname))

# Load community and total
cn01 = pd.read_csv(os.path.join(DATA, 'CN01_energy.csv'))
total_energy = pd.read_csv(os.path.join(DATA, 'Total_energy.csv'))
weather = pd.read_csv(os.path.join(DATA, 'Total_weather.csv'))

# Create datetime for energy files
def add_datetime(df):
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].rename(
        columns={'year':'year','month':'month','day':'day','hour':'hour'}
    ).assign(minute=0, second=0))
    return df

for k in buildings:
    buildings[k] = add_datetime(buildings[k])
cn01 = add_datetime(cn01)
total_energy = add_datetime(total_energy)
weather['datetime'] = pd.to_datetime(weather['datetime'])

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']

# === Summary Statistics ===
print("=" * 60)
print("PHASE 1: DATA EXPLORATION & QUALITY ASSESSMENT")
print("=" * 60)

# Check shapes
print("\n--- Dataset Shapes ---")
for k, df in buildings.items():
    print(f"  {k}: {df.shape}")
print(f"  CN01: {cn01.shape}")
print(f"  Total: {total_energy.shape}")
print(f"  Weather: {weather.shape}")

# Check date range
print("\n--- Date Range ---")
for k, df in list(buildings.items())[:2]:
    print(f"  {k}: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Weather: {weather['datetime'].min()} to {weather['datetime'].max()}")

# Missing values
print("\n--- Missing Values ---")
missing_report = {}
for k, df in buildings.items():
    missing = df[energy_cols].isnull().sum().to_dict()
    missing_report[k] = missing
    if any(v > 0 for v in missing.values()):
        print(f"  {k}: {missing}")
        
cn01_missing = cn01[energy_cols].isnull().sum().to_dict()
total_missing = total_energy[energy_cols].isnull().sum().to_dict()
weather_missing = weather[weather_cols].isnull().sum().to_dict()
missing_report['CN01'] = cn01_missing
missing_report['Total'] = total_missing
missing_report['Weather'] = weather_missing

print(f"  CN01: {cn01_missing}")
print(f"  Total: {total_missing}")
print(f"  Weather: {weather_missing}")

# Negative values
print("\n--- Negative Values Check ---")
neg_report = {}
for k, df in buildings.items():
    neg_counts = (df[energy_cols] < 0).sum().to_dict()
    neg_report[k] = neg_counts
    if any(v > 0 for v in neg_counts.values()):
        print(f"  {k}: {neg_counts}")

cn01_neg = (cn01[energy_cols] < 0).sum().to_dict()
total_neg = (total_energy[energy_cols] < 0).sum().to_dict()
neg_report['CN01'] = cn01_neg
neg_report['Total'] = total_neg
if any(v > 0 for v in cn01_neg.values()):
    print(f"  CN01: {cn01_neg}")
if any(v > 0 for v in total_neg.values()):
    print(f"  Total: {total_neg}")
print("  (No negative values found)" if all(
    all(v == 0 for v in d.values()) for d in neg_report.values()
) else "")

# Summary statistics for each building
print("\n--- Summary Statistics (BN001 example) ---")
print(buildings['BN001'][energy_cols].describe().to_string())

print("\n--- Summary Statistics (Total) ---")
print(total_energy[energy_cols].describe().to_string())

print("\n--- Summary Statistics (Weather) ---")
print(weather[weather_cols].describe().to_string())

# Zero values analysis (especially PV)
print("\n--- Zero Values Analysis ---")
for k, df in buildings.items():
    pv_zeros = (df['PV Power Generation [kW]'] == 0).sum()
    print(f"  {k} PV zeros: {pv_zeros}/{len(df)} ({pv_zeros/len(df)*100:.1f}%)")

# Save summary
summary = {
    'n_buildings': 10,
    'n_communities': 1,
    'n_records_per_file': 8760,
    'total_records': 8760 * 13,
    'date_range': '2014-01-01 to 2014-12-31',
    'energy_variables': energy_cols,
    'weather_variables': weather_cols,
    'missing_values': {k: {kk: int(vv) for kk, vv in v.items()} for k, v in missing_report.items()},
}

with open(os.path.join(OUT, 'data_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n✓ Phase 1 complete. Summary saved to outputs/data_summary.json")
