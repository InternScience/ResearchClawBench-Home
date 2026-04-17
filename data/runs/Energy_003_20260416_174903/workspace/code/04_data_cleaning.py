#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Data Cleaning Algorithms
Implements data quality assessment and cleaning procedures
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260416_174903/data/HEEW_Mini-Dataset")
OUTPUT_DIR = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260416_174903/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_data():
    """Load all energy and weather data"""
    buildings = {}
    for i in range(1, 11):
        bn = f'BN{i:03d}'
        df = pd.read_csv(DATA_DIR / f'{bn}_energy.csv')
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']].assign(hour=df['hour']))
        buildings[bn] = df
    
    cn01 = pd.read_csv(DATA_DIR / 'CN01_energy.csv')
    cn01['datetime'] = pd.to_datetime(cn01[['year', 'month', 'day']].assign(hour=cn01['hour']))
    
    total = pd.read_csv(DATA_DIR / 'Total_energy.csv')
    total['datetime'] = pd.to_datetime(total[['year', 'month', 'day']].assign(hour=total['hour']))
    
    weather = pd.read_csv(DATA_DIR / 'Total_weather.csv')
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    
    return buildings, cn01, total, weather

def check_missing_values(df, name):
    """Check for missing values in dataframe"""
    missing = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = missing.sum()
    
    return {
        'dataset': name,
        'total_rows': int(df.shape[0]),
        'total_columns': int(df.shape[1]),
        'missing_by_column': missing.to_dict(),
        'total_missing_cells': int(missing_cells),
        'completeness_pct': float((1 - missing_cells / total_cells) * 100)
    }

def detect_outliers_iqr(df, columns, multiplier=1.5):
    """Detect outliers using IQR method"""
    outlier_report = {}
    outlier_indices = set()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        outlier_report[col] = {
            'method': 'IQR',
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': int(outlier_count),
            'outlier_pct': float(outlier_count / len(df) * 100)
        }
        
        outlier_indices.update(df[outlier_mask].index.tolist())
    
    return outlier_report, list(outlier_indices)

def detect_outliers_zscore(df, columns, threshold=3):
    """Detect outliers using Z-score method"""
    outlier_report = {}
    outlier_indices = set()
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = np.abs((df[col] - mean) / std)
        
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()
        
        outlier_report[col] = {
            'method': 'Z-score',
            'mean': float(mean),
            'std': float(std),
            'threshold': threshold,
            'outlier_count': int(outlier_count),
            'outlier_pct': float(outlier_count / len(df) * 100)
        }
        
        outlier_indices.update(df[outlier_mask].index.tolist())
    
    return outlier_report, list(outlier_indices)

def detect_outliers_mad(df, columns, threshold=3.5):
    """Detect outliers using Modified Z-score (MAD-based)"""
    outlier_report = {}
    outlier_indices = set()
    
    for col in columns:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        
        # Modified Z-score
        modified_z = 0.6745 * (df[col] - median) / mad if mad != 0 else 0
        outlier_mask = np.abs(modified_z) > threshold
        outlier_count = outlier_mask.sum()
        
        outlier_report[col] = {
            'method': 'Modified Z-score (MAD)',
            'median': float(median),
            'MAD': float(mad),
            'threshold': threshold,
            'outlier_count': int(outlier_count),
            'outlier_pct': float(outlier_count / len(df) * 100)
        }
        
        outlier_indices.update(df[outlier_mask].index.tolist())
    
    return outlier_report, list(outlier_indices)

def check_range_validity(df, columns, valid_ranges):
    """Check if values are within physically valid ranges"""
    range_report = {}
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col not in columns:
            continue
            
        below_min = (df[col] < min_val).sum()
        above_max = (df[col] > max_val).sum()
        
        range_report[col] = {
            'valid_range': [min_val, max_val],
            'below_min_count': int(below_min),
            'above_max_count': int(above_max),
            'invalid_count': int(below_min + above_max),
            'invalid_pct': float((below_min + above_max) / len(df) * 100)
        }
    
    return range_report

def verify_hierarchical_consistency(buildings, cn01, total):
    """Verify consistency across hierarchical levels"""
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    
    consistency_results = {}
    
    # Sum of all buildings
    bn_sum = {}
    for col in energy_cols:
        bn_sum[col] = sum(buildings[bn][col].sum() for bn in buildings.keys())
    
    cn01_sum = {col: cn01[col].sum() for col in energy_cols}
    total_sum = {col: total[col].sum() for col in energy_cols}
    
    for col in energy_cols:
        # BN sum vs CN01
        diff_bn_cn01 = abs(bn_sum[col] - cn01_sum[col])
        pct_bn_cn01 = diff_bn_cn01 / cn01_sum[col] * 100 if cn01_sum[col] != 0 else 0
        
        # CN01 vs Total
        diff_cn01_total = abs(cn01_sum[col] - total_sum[col])
        pct_cn01_total = diff_cn01_total / total_sum[col] * 100 if total_sum[col] != 0 else 0
        
        consistency_results[col] = {
            'bn_sum': float(bn_sum[col]),
            'cn01_sum': float(cn01_sum[col]),
            'total_sum': float(total_sum[col]),
            'bn_vs_cn01_diff': float(diff_bn_cn01),
            'bn_vs_cn01_pct': float(pct_bn_cn01),
            'cn01_vs_total_diff': float(diff_cn01_total),
            'cn01_vs_total_pct': float(pct_cn01_total),
            'consistent': pct_bn_cn01 < 0.01 and pct_cn01_total < 0.01
        }
    
    return consistency_results

def check_temporal_continuity(df, name):
    """Check for gaps in temporal sequence"""
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # Check for duplicate timestamps
    duplicates = df_sorted.duplicated(subset=['datetime']).sum()
    
    # Check for missing hours (should be continuous hourly data)
    time_diffs = df_sorted['datetime'].diff()
    expected_diff = pd.Timedelta(hours=1)
    
    # Count gaps (differences != 1 hour)
    gaps = (time_diffs != expected_diff).sum() - 1  # -1 for first NaN
    
    return {
        'dataset': name,
        'total_records': int(len(df)),
        'duplicate_timestamps': int(duplicates),
        'temporal_gaps': int(max(0, gaps)),
        'start_date': str(df_sorted['datetime'].min()),
        'end_date': str(df_sorted['datetime'].max()),
        'expected_hours': 8760,
        'actual_hours': int(len(df))
    }

def generate_cleaning_recommendations(quality_report):
    """Generate data cleaning recommendations based on quality assessment"""
    recommendations = []
    
    # Missing value recommendations
    for dataset, report in quality_report.get('missing_value_analysis', {}).items():
        if report.get('completeness_pct', 100) < 99:
            recommendations.append({
                'issue': f"Missing values in {dataset}",
                'severity': 'high' if report['completeness_pct'] < 95 else 'medium',
                'recommendation': 'Apply imputation or interpolation methods'
            })
        else:
            recommendations.append({
                'issue': f"Data completeness in {dataset}",
                'severity': 'info',
                'recommendation': 'No action needed - data is complete'
            })
    
    # Outlier recommendations
    outlier_summary = quality_report.get('outlier_summary', {})
    if outlier_summary.get('total_outliers_iqr', 0) > outlier_summary.get('total_records', 1) * 0.05:
        recommendations.append({
            'issue': 'High number of outliers detected (IQR method)',
            'severity': 'medium',
            'recommendation': 'Review outliers for data entry errors; consider robust statistical methods'
        })
    
    # Consistency recommendations
    consistency = quality_report.get('hierarchical_consistency', {})
    inconsistent_vars = [k for k, v in consistency.items() if not v.get('consistent', True)]
    if inconsistent_vars:
        recommendations.append({
            'issue': f'Hierarchical inconsistency in: {", ".join(inconsistent_vars)}',
            'severity': 'high',
            'recommendation': 'Investigate aggregation logic and source data'
        })
    else:
        recommendations.append({
            'issue': 'Hierarchical consistency',
            'severity': 'info',
            'recommendation': 'All hierarchical levels are consistent'
        })
    
    return recommendations

if __name__ == '__main__':
    print("=" * 60)
    print("HEEW Mini-Dataset: Data Cleaning Assessment")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    buildings, cn01, total, weather = load_all_data()
    
    # Define column groups
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                    'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
    
    # Define valid ranges for physical validation
    valid_ranges = {
        'Electricity [kW]': (0, 1000),
        'Heat [mmBTU]': (0, 500),
        'Cooling Energy [Ton]': (0, 500),
        'PV Power Generation [kW]': (0, 200),
        'Greenhouse Gas Emission [Ton]': (0, 1000),
        'Temperature [°F]': (-20, 130),
        'Humidity [%]': (0, 100),
        'Wind Speed [mph]': (0, 100),
        'Pressure [in]': (28, 31)
    }
    
    # Quality assessment
    quality_report = {
        'missing_value_analysis': {},
        'outlier_analysis': {},
        'range_validation': {},
        'temporal_continuity': {},
        'hierarchical_consistency': {},
        'recommendations': []
    }
    
    # Missing value analysis
    print("\n2. Checking missing values...")
    for bn in buildings.keys():
        quality_report['missing_value_analysis'][bn] = check_missing_values(buildings[bn], bn)
    quality_report['missing_value_analysis']['CN01'] = check_missing_values(cn01, 'CN01')
    quality_report['missing_value_analysis']['Total'] = check_missing_values(total, 'Total')
    quality_report['missing_value_analysis']['Weather'] = check_missing_values(weather, 'Weather')
    
    # Outlier detection (Total dataset)
    print("\n3. Detecting outliers...")
    iqr_outliers, iqr_indices = detect_outliers_iqr(total, energy_cols)
    zscore_outliers, zscore_indices = detect_outliers_zscore(total, energy_cols)
    mad_outliers, mad_indices = detect_outliers_mad(total, energy_cols)
    
    quality_report['outlier_analysis'] = {
        'iqr_method': iqr_outliers,
        'zscore_method': zscore_outliers,
        'mad_method': mad_outliers,
        'total_records': len(total),
        'total_outliers_iqr': len(set(iqr_indices)),
        'total_outliers_zscore': len(set(zscore_indices)),
        'total_outliers_mad': len(set(mad_indices))
    }
    
    # Range validation
    print("\n4. Validating physical ranges...")
    quality_report['range_validation']['energy'] = check_range_validity(total, energy_cols, valid_ranges)
    quality_report['range_validation']['weather'] = check_range_validity(weather, weather_cols, valid_ranges)
    
    # Temporal continuity
    print("\n5. Checking temporal continuity...")
    for bn in buildings.keys():
        quality_report['temporal_continuity'][bn] = check_temporal_continuity(buildings[bn], bn)
    
    # Hierarchical consistency
    print("\n6. Verifying hierarchical consistency...")
    quality_report['hierarchical_consistency'] = verify_hierarchical_consistency(buildings, cn01, total)
    
    # Generate recommendations
    print("\n7. Generating cleaning recommendations...")
    quality_report['recommendations'] = generate_cleaning_recommendations(quality_report)
    
    # Save report
    print("\n8. Saving cleaning report...")
    with open(OUTPUT_DIR / 'cleaning_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("Data Cleaning Assessment Complete!")
    print("=" * 60)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"\nMissing Values:")
    for dataset, report in quality_report['missing_value_analysis'].items():
        print(f"  {dataset}: {report['completeness_pct']:.1f}% complete")
    
    print(f"\nOutliers (Total dataset, IQR method):")
    for col, report in quality_report['outlier_analysis']['iqr_method'].items():
        print(f"  {col}: {report['outlier_count']} outliers ({report['outlier_pct']:.2f}%)")
    
    print(f"\nHierarchical Consistency:")
    for col, report in quality_report['hierarchical_consistency'].items():
        status = "✓" if report['consistent'] else "✗"
        print(f"  {status} {col}")
    
    print(f"\nRecommendations:")
    for rec in quality_report['recommendations']:
        print(f"  [{rec['severity'].upper()}] {rec['issue']}: {rec['recommendation']}")
