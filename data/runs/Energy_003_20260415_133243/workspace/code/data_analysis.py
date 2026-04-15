"""
HEEW Dataset Analysis Script
Analyzes the Hierarchical Energy and Weather (HEEW) Mini-Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directories
os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/outputs', exist_ok=True)
os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/report/images', exist_ok=True)

# Data paths
DATA_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/data/HEEW_Mini-Dataset'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/outputs'
REPORT_IMG_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/report/images'

def load_energy_data(building_id):
    """Load energy data for a specific building"""
    filepath = os.path.join(DATA_DIR, f'{building_id}_energy.csv')
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df

def load_weather_data():
    """Load weather data"""
    filepath = os.path.join(DATA_DIR, 'Total_weather.csv')
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def load_all_buildings():
    """Load all building energy data"""
    buildings = ['BN001', 'BN002', 'BN003', 'BN004', 'BN005', 
                 'BN006', 'BN007', 'BN008', 'BN009', 'BN010']
    data = {}
    for b in buildings:
        data[b] = load_energy_data(b)
    return data

def data_quality_analysis():
    """Analyze data quality - missing values, outliers, etc."""
    print("="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Load all data
    buildings = load_all_buildings()
    weather = load_weather_data()
    total = load_energy_data('Total')
    cn01 = load_energy_data('CN01')
    
    results = {}
    
    # Check for missing values
    print("\n1. Missing Values Analysis:")
    for b, df in buildings.items():
        missing = df.isnull().sum().sum()
        print(f"   {b}: {missing} missing values")
        results[b] = {'missing': missing}
    
    # Check data completeness (expected 8760 hours for a year)
    print("\n2. Data Completeness (records expected: 8760):")
    for b, df in buildings.items():
        n_records = len(df)
        completeness = (n_records / 8760) * 100
        print(f"   {b}: {n_records} records ({completeness:.1f}%)")
        results[b]['completeness'] = completeness
    
    # Check for negative values in energy consumption (anomaly detection)
    print("\n3. Anomaly Detection - Negative Values:")
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]']
    for b, df in buildings.items():
        anomalies = {}
        for col in energy_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                anomalies[col] = neg_count
        if anomalies:
            print(f"   {b}: {anomalies}")
        else:
            print(f"   {b}: No negative values found")
    
    # Check for extreme outliers using IQR method
    print("\n4. Outlier Detection (IQR method, >3*IQR):")
    for b, df in buildings.items():
        outliers = {}
        for col in energy_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 3*IQR)) | (df[col] > (Q3 + 3*IQR))).sum()
            outliers[col] = outlier_count
        print(f"   {b}: {outliers}")
    
    return results

def hierarchical_consistency_check():
    """Verify hierarchical aggregation: sum of buildings = CN01 = Total"""
    print("\n" + "="*60)
    print("HIERARCHICAL CONSISTENCY CHECK")
    print("="*60)
    
    buildings = load_all_buildings()
    total = load_energy_data('Total')
    cn01 = load_energy_data('CN01')
    
    # Sum of all buildings
    building_sum = sum([df.set_index('datetime')['Electricity [kW]'] for df in buildings.values()])
    building_sum = building_sum.sort_index()
    
    # Compare with Total
    total_elec = total.set_index('datetime')['Electricity [kW]'].sort_index()
    cn01_elec = cn01.set_index('datetime')['Electricity [kW]'].sort_index()
    
    # Calculate correlation and difference statistics
    corr_total = np.corrcoef(building_sum.values, total_elec.values)[0, 1]
    
    print(f"\nCorrelation between sum of buildings and Total: {corr_total:.6f}")
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((building_sum.values - total_elec.values) / (total_elec.values + 1e-10))) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    
    results = {
        'correlation': corr_total,
        'mape': mape
    }
    
    return results

def correlation_analysis():
    """Analyze correlations between energy variables and weather"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Load Total energy and weather
    total = load_energy_data('Total')
    weather = load_weather_data()
    
    # Merge datasets
    merged = pd.merge(total, weather, on='datetime', how='inner')
    
    # Select numeric columns for correlation
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    weather_cols = ['Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]', 
                    'Pressure [in]', 'Precipitation [in]']
    
    all_cols = energy_cols + weather_cols
    corr_matrix = merged[all_cols].corr()
    
    print("\nCorrelation Matrix (key correlations):")
    # Print energy-weather correlations
    for e_col in energy_cols:
        for w_col in weather_cols:
            corr = corr_matrix.loc[e_col, w_col]
            if abs(corr) > 0.3:  # Only show significant correlations
                print(f"   {e_col} vs {w_col}: {corr:.3f}")
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))
    
    return corr_matrix, merged
