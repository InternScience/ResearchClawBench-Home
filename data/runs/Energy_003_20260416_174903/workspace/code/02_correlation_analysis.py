#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Correlation Analysis and Outlier Detection
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

def load_and_prepare_data():
    """Load and prepare all data for analysis"""
    # Load Total energy data
    total_energy = pd.read_csv(DATA_DIR / 'Total_energy.csv')
    total_energy['datetime'] = pd.to_datetime(total_energy[['year', 'month', 'day']].assign(hour=total_energy['hour']))
    
    # Load weather data
    weather = pd.read_csv(DATA_DIR / 'Total_weather.csv')
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    
    # Merge energy and weather
    merged = total_energy.merge(weather, on='datetime')
    
    return total_energy, weather, merged

def compute_correlation_matrix(merged_df):
    """Compute correlation matrix for all variables"""
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    
    weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                    'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
    
    all_cols = energy_cols + weather_cols
    
    corr_matrix = merged_df[all_cols].corr()
    
    return corr_matrix.to_dict()

def detect_outliers_iqr(df, columns, multiplier=1.5):
    """Detect outliers using IQR method"""
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        outliers[col] = {
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': int(outlier_count),
            'outlier_pct': float(outlier_count / len(df) * 100)
        }
    
    return outliers

def detect_outliers_zscore(df, columns, threshold=3):
    """Detect outliers using Z-score method"""
    outliers = {}
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = np.abs((df[col] - mean) / std)
        
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()
        
        outliers[col] = {
            'mean': float(mean),
            'std': float(std),
            'threshold': threshold,
            'outlier_count': int(outlier_count),
            'outlier_pct': float(outlier_count / len(df) * 100)
        }
    
    return outliers

def analyze_temporal_patterns(merged_df):
    """Analyze temporal patterns in the data"""
    merged_df['hour'] = merged_df['datetime'].dt.hour
    merged_df['dayofweek'] = merged_df['datetime'].dt.dayofweek
    merged_df['month'] = merged_df['datetime'].dt.month
    merged_df['dayofyear'] = merged_df['datetime'].dt.dayofyear
    
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    
    patterns = {
        'hourly': {},
        'monthly': {},
        'dayofweek': {}
    }
    
    # Hourly patterns
    for col in energy_cols:
        hourly_stats = merged_df.groupby('hour')[col].agg(['mean', 'std', 'min', 'max'])
        patterns['hourly'][col] = {
            'mean_by_hour': hourly_stats['mean'].to_dict(),
            'std_by_hour': hourly_stats['std'].to_dict()
        }
    
    # Monthly patterns
    for col in energy_cols:
        monthly_stats = merged_df.groupby('month')[col].agg(['mean', 'std', 'min', 'max'])
        patterns['monthly'][col] = {
            'mean_by_month': monthly_stats['mean'].to_dict(),
            'std_by_month': monthly_stats['std'].to_dict()
        }
    
    # Day of week patterns
    for col in energy_cols:
        dow_stats = merged_df.groupby('dayofweek')[col].agg(['mean', 'std'])
        patterns['dayofweek'][col] = {
            'mean_by_dow': dow_stats['mean'].to_dict(),
            'std_by_dow': dow_stats['std'].to_dict()
        }
    
    return patterns

if __name__ == '__main__':
    print("=" * 60)
    print("HEEW Mini-Dataset: Correlation Analysis")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and preparing data...")
    total_energy, weather, merged = load_and_prepare_data()
    print(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    
    # Correlation analysis
    print("\n2. Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(merged)
    
    # Outlier detection
    print("\n3. Detecting outliers...")
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                    'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
    
    iqr_outliers_energy = detect_outliers_iqr(total_energy, energy_cols)
    iqr_outliers_weather = detect_outliers_iqr(weather, weather_cols)
    
    zscore_outliers_energy = detect_outliers_zscore(total_energy, energy_cols)
    zscore_outliers_weather = detect_outliers_zscore(weather, weather_cols)
    
    # Temporal patterns
    print("\n4. Analyzing temporal patterns...")
    temporal_patterns = analyze_temporal_patterns(merged)
    
    # Save results
    print("\n5. Saving results...")
    
    with open(OUTPUT_DIR / 'correlation_matrix.json', 'w') as f:
        json.dump(corr_matrix, f, indent=2)
    
    with open(OUTPUT_DIR / 'outlier_detection.json', 'w') as f:
        json.dump({
            'iqr_method': {
                'energy': iqr_outliers_energy,
                'weather': iqr_outliers_weather
            },
            'zscore_method': {
                'energy': zscore_outliers_energy,
                'weather': zscore_outliers_weather
            }
        }, f, indent=2)
    
    with open(OUTPUT_DIR / 'temporal_patterns.json', 'w') as f:
        json.dump(temporal_patterns, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Correlation analysis complete!")
    print("=" * 60)
    
    # Print key correlations
    print("\n=== KEY CORRELATIONS (Energy vs Weather) ===")
    energy_vars = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 'PV Power Generation [kW]']
    weather_vars = ['Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]']
    
    for ev in energy_vars:
        for wv in weather_vars:
            corr_val = corr_matrix.get(ev, {}).get(wv, 'N/A')
            if isinstance(corr_val, float):
                print(f"{ev} vs {wv}: {corr_val:.3f}")
