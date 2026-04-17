#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Data Exploration and Quality Assessment
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260416_174903/data/HEEW_Mini-Dataset")
OUTPUT_DIR = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260416_174903/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_energy_data():
    """Load all energy data files"""
    energy_files = {
        'BN001': DATA_DIR / 'BN001_energy.csv',
        'BN002': DATA_DIR / 'BN002_energy.csv',
        'BN003': DATA_DIR / 'BN003_energy.csv',
        'BN004': DATA_DIR / 'BN004_energy.csv',
        'BN005': DATA_DIR / 'BN005_energy.csv',
        'BN006': DATA_DIR / 'BN006_energy.csv',
        'BN007': DATA_DIR / 'BN007_energy.csv',
        'BN008': DATA_DIR / 'BN008_energy.csv',
        'BN009': DATA_DIR / 'BN009_energy.csv',
        'BN010': DATA_DIR / 'BN010_energy.csv',
        'CN01': DATA_DIR / 'CN01_energy.csv',
        'Total': DATA_DIR / 'Total_energy.csv',
    }
    
    data = {}
    for name, path in energy_files.items():
        df = pd.read_csv(path)
        data[name] = df
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return data

def load_weather_data():
    """Load weather data"""
    weather_path = DATA_DIR / 'Total_weather.csv'
    df = pd.read_csv(weather_path)
    print(f"Weather: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def create_datetime(df, is_weather=False):
    """Create datetime column"""
    if is_weather:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']].assign(hour=df['hour']))
    return df

def assess_data_quality(data_dict, weather_df):
    """Assess data quality and missing values"""
    quality_report = {
        'energy_data': {},
        'weather_data': {},
        'summary': {}
    }
    
    # Energy data quality
    for name, df in data_dict.items():
        df_dt = create_datetime(df.copy())
        missing = df_dt.isnull().sum().to_dict()
        total_rows = len(df_dt)
        
        quality_report['energy_data'][name] = {
            'total_rows': int(total_rows),
            'expected_rows': 8760,  # 24 * 365
            'missing_values': {k: int(v) for k, v in missing.items()},
            'completeness_pct': float((1 - df_dt.isnull().sum().sum() / (total_rows * len(df_dt.columns))) * 100)
        }
    
    # Weather data quality
    weather_dt = create_datetime(weather_df.copy(), is_weather=True)
    weather_missing = weather_dt.isnull().sum().to_dict()
    quality_report['weather_data'] = {
        'total_rows': int(len(weather_dt)),
        'expected_rows': 8760,
        'missing_values': {k: int(v) for k, v in weather_missing.items()},
        'completeness_pct': float((1 - weather_dt.isnull().sum().sum() / (len(weather_dt) * len(weather_dt.columns))) * 100)
    }
    
    return quality_report

def check_hierarchical_consistency(data_dict):
    """Check if building sums match community and total"""
    consistency_report = {
        'building_sum_vs_CN01': {},
        'CN01_vs_Total': {},
        'all_buildings_vs_Total': {}
    }
    
    # Sum of BN001-BN010
    building_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                     'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    
    bn_sum = pd.DataFrame()
    bn_sum['year'] = data_dict['BN001']['year']
    bn_sum['month'] = data_dict['BN001']['month']
    bn_sum['day'] = data_dict['BN001']['day']
    bn_sum['hour'] = data_dict['BN001']['hour']
    
    for col in building_cols:
        bn_sum[col] = sum(data_dict[f'BN{i:03d}'][col] for i in range(1, 11))
    
    # Compare BN sum vs CN01
    cn01 = data_dict['CN01']
    for col in building_cols:
        bn_total = bn_sum[col].sum()
        cn01_total = cn01[col].sum()
        diff_pct = abs(bn_total - cn01_total) / cn01_total * 100 if cn01_total != 0 else 0
        consistency_report['building_sum_vs_CN01'][col] = {
            'bn_sum': float(bn_total),
            'cn01_sum': float(cn01_total),
            'difference_pct': float(diff_pct)
        }
    
    # Compare CN01 vs Total
    total = data_dict['Total']
    for col in building_cols:
        cn01_total = cn01[col].sum()
        total_sum = total[col].sum()
        diff_pct = abs(cn01_total - total_sum) / total_sum * 100 if total_sum != 0 else 0
        consistency_report['CN01_vs_Total'][col] = {
            'cn01_sum': float(cn01_total),
            'total_sum': float(total_sum),
            'difference_pct': float(diff_pct)
        }
    
    # Compare all buildings sum vs Total
    for col in building_cols:
        bn_total = bn_sum[col].sum()
        total_sum = total[col].sum()
        diff_pct = abs(bn_total - total_sum) / total_sum * 100 if total_sum != 0 else 0
        consistency_report['all_buildings_vs_Total'][col] = {
            'bn_sum': float(bn_total),
            'total_sum': float(total_sum),
            'difference_pct': float(diff_pct)
        }
    
    return consistency_report, bn_sum

def compute_basic_statistics(data_dict, weather_df):
    """Compute basic statistics for all variables"""
    stats = {'energy_stats': {}, 'weather_stats': {}}
    
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    
    # Energy stats for Total
    total_df = data_dict['Total']
    for col in energy_cols:
        stats['energy_stats'][col] = {
            'mean': float(total_df[col].mean()),
            'std': float(total_df[col].std()),
            'min': float(total_df[col].min()),
            'max': float(total_df[col].max()),
            'median': float(total_df[col].median())
        }
    
    # Weather stats
    weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                    'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
    
    for col in weather_cols:
        stats['weather_stats'][col] = {
            'mean': float(weather_df[col].mean()),
            'std': float(weather_df[col].std()),
            'min': float(weather_df[col].min()),
            'max': float(weather_df[col].max()),
            'median': float(weather_df[col].median())
        }
    
    return stats

if __name__ == '__main__':
    print("=" * 60)
    print("HEEW Mini-Dataset: Data Exploration")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading energy data...")
    energy_data = load_energy_data()
    
    print("\n2. Loading weather data...")
    weather_data = load_weather_data()
    
    # Data quality assessment
    print("\n3. Assessing data quality...")
    quality_report = assess_data_quality(energy_data, weather_data)
    
    # Hierarchical consistency
    print("\n4. Checking hierarchical consistency...")
    consistency_report, bn_sum = check_hierarchical_consistency(energy_data)
    
    # Basic statistics
    print("\n5. Computing basic statistics...")
    stats = compute_basic_statistics(energy_data, weather_data)
    
    # Save reports
    print("\n6. Saving reports...")
    
    with open(OUTPUT_DIR / 'data_summary.json', 'w') as f:
        json.dump({
            'quality_assessment': quality_report,
            'basic_statistics': stats
        }, f, indent=2)
    
    with open(OUTPUT_DIR / 'hierarchical_validation.json', 'w') as f:
        json.dump(consistency_report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data exploration complete!")
    print(f"Reports saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Print summary
    print("\n=== HIERARCHICAL CONSISTENCY SUMMARY ===")
    for level, metrics in consistency_report.items():
        print(f"\n{level}:")
        for var, vals in metrics.items():
            print(f"  {var}: {vals['difference_pct']:.2f}% difference")
