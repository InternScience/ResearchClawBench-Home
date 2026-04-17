#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Data Cleaning, Outlier Detection, and Hierarchical Consistency
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

# Load all data
def load_energy(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].assign(minute=0, second=0))
    return df

buildings = {f'BN{i:03d}': load_energy(f'BN{i:03d}_energy.csv') for i in range(1, 11)}
cn01 = load_energy('CN01_energy.csv')
total_energy = load_energy('Total_energy.csv')
weather = pd.read_csv(os.path.join(DATA, 'Total_weather.csv'))
weather['datetime'] = pd.to_datetime(weather['datetime'])

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']

# =====================================================
# PHASE 2: DATA CLEANING ALGORITHMS
# =====================================================
print("=" * 60)
print("PHASE 2: DATA CLEANING & OUTLIER DETECTION")
print("=" * 60)

# 2a. IQR-based outlier detection
def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

# 2b. Z-score based outlier detection
def detect_outliers_zscore(series, threshold=3.0):
    z = np.abs((series - series.mean()) / series.std())
    return z > threshold

# Apply to all buildings
outlier_report = {}
for name, df in {**buildings, 'CN01': cn01, 'Total': total_energy}.items():
    outlier_report[name] = {}
    for col in energy_cols:
        iqr_outliers = detect_outliers_iqr(df[col]).sum()
        zscore_outliers = detect_outliers_zscore(df[col]).sum()
        outlier_report[name][col] = {
            'iqr_outliers': int(iqr_outliers),
            'zscore_outliers': int(zscore_outliers),
            'iqr_pct': round(iqr_outliers / len(df) * 100, 2),
            'zscore_pct': round(zscore_outliers / len(df) * 100, 2)
        }

print("\n--- Outlier Detection Summary (IQR method) ---")
for name in ['BN001', 'BN005', 'BN010', 'CN01', 'Total']:
    print(f"\n  {name}:")
    for col in energy_cols:
        info = outlier_report[name][col]
        print(f"    {col}: IQR={info['iqr_outliers']} ({info['iqr_pct']}%), Z-score={info['zscore_outliers']} ({info['zscore_pct']}%)")

# Weather outlier detection
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
weather_outliers = {}
for col in weather_cols:
    iqr_out = detect_outliers_iqr(weather[col]).sum()
    zs_out = detect_outliers_zscore(weather[col]).sum()
    weather_outliers[col] = {'iqr': int(iqr_out), 'zscore': int(zs_out)}
    
print("\n--- Weather Outlier Detection ---")
for col, info in weather_outliers.items():
    print(f"  {col}: IQR={info['iqr']}, Z-score={info['zscore']}")

# =====================================================
# PHASE 3: HIERARCHICAL AGGREGATION CONSISTENCY
# =====================================================
print("\n" + "=" * 60)
print("PHASE 3: HIERARCHICAL AGGREGATION CONSISTENCY")
print("=" * 60)

# Sum all buildings
building_sum = pd.DataFrame()
building_sum['datetime'] = buildings['BN001']['datetime']
for col in energy_cols:
    building_sum[col] = sum(buildings[k][col] for k in buildings)

# Compare building sum vs CN01
print("\n--- Building Sum vs CN01 ---")
consistency_results = {}
for col in energy_cols:
    diff = building_sum[col] - cn01[col]
    abs_diff = diff.abs()
    rel_diff = (abs_diff / cn01[col].abs().replace(0, np.nan)) * 100
    
    result = {
        'mean_abs_diff': float(abs_diff.mean()),
        'max_abs_diff': float(abs_diff.max()),
        'mean_rel_diff_pct': float(rel_diff.mean()),
        'max_rel_diff_pct': float(rel_diff.max()),
        'rmse': float(np.sqrt((diff**2).mean())),
        'correlation': float(building_sum[col].corr(cn01[col]))
    }
    consistency_results[col] = result
    print(f"\n  {col}:")
    print(f"    Mean Abs Diff: {result['mean_abs_diff']:.6f}")
    print(f"    Max Abs Diff: {result['max_abs_diff']:.6f}")
    print(f"    Mean Rel Diff: {result['mean_rel_diff_pct']:.6f}%")
    print(f"    RMSE: {result['rmse']:.6f}")
    print(f"    Correlation: {result['correlation']:.6f}")

# Compare CN01 vs Total
print("\n--- CN01 vs Total ---")
cn01_vs_total = {}
for col in energy_cols:
    diff = cn01[col] - total_energy[col]
    abs_diff = diff.abs()
    rel_diff = (abs_diff / total_energy[col].abs().replace(0, np.nan)) * 100
    
    result = {
        'mean_abs_diff': float(abs_diff.mean()),
        'max_abs_diff': float(abs_diff.max()),
        'mean_rel_diff_pct': float(rel_diff.mean()),
        'correlation': float(cn01[col].corr(total_energy[col]))
    }
    cn01_vs_total[col] = result
    print(f"\n  {col}:")
    print(f"    Mean Abs Diff: {result['mean_abs_diff']:.6f}")
    print(f"    Max Abs Diff: {result['max_abs_diff']:.6f}")
    print(f"    Mean Rel Diff: {result['mean_rel_diff_pct']:.6f}%")
    print(f"    Correlation: {result['correlation']:.6f}")

# Save results
all_results = {
    'outlier_detection': outlier_report,
    'weather_outliers': weather_outliers,
    'hierarchical_consistency': {
        'building_sum_vs_cn01': consistency_results,
        'cn01_vs_total': cn01_vs_total
    }
}

with open(os.path.join(OUT, 'cleaning_and_consistency.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Save building sum for later use
building_sum.to_csv(os.path.join(OUT, 'building_sum.csv'), index=False)

print("\n✓ Phase 2 & 3 complete.")
