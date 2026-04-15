#!/usr/bin/env python3
"""
HEEW Mini-Dataset Comprehensive Analysis
=========================================
Analysis of multi-source hierarchical time-series dataset from ASU Campus Metabolism Project.
Covers electricity, heat, cooling loads, PV generation, GHG emissions, and weather data
for 10 buildings (BN001-BN010), one community (CN01), and total area for 2014.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_003_20260415_133148'
DATA_DIR = os.path.join(WORKSPACE, 'data', 'HEEW_Mini-Dataset')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGES_DIR = os.path.join(WORKSPACE, 'report', 'images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 60)
print("PHASE 1: Data Loading & Preprocessing")
print("=" * 60)

def load_energy_data(filepath):
    """Load energy CSV and create datetime index."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].rename(
        columns={'hour': 'hour'}))
    df = df.set_index('datetime').sort_index()
    return df

def load_weather_data(filepath):
    """Load weather CSV and create datetime index."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df

# Load building energy data
building_ids = [f'BN{i:03d}' for i in range(1, 11)]
energy_data = {}
for bid in building_ids:
    filepath = os.path.join(DATA_DIR, f'{bid}_energy.csv')
    energy_data[bid] = load_energy_data(filepath)
    print(f"Loaded {bid}: {len(energy_data[bid])} records")

# Load community and total energy data
cn01_data = load_energy_data(os.path.join(DATA_DIR, 'CN01_energy.csv'))
total_energy = load_energy_data(os.path.join(DATA_DIR, 'Total_energy.csv'))
weather_data = load_weather_data(os.path.join(DATA_DIR, 'Total_weather.csv'))

print(f"\nCommunity CN01: {len(cn01_data)} records")
print(f"Total Energy: {len(total_energy)} records")
print(f"Weather Data: {len(weather_data)} records")

# Energy variable names
energy_vars = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
weather_vars = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']

# ============================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Data Quality Assessment")
print("=" * 60)

quality_report = {}
for bid in building_ids:
    df = energy_data[bid]
    report = {
        'records': len(df),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_timestamps': int(df.index.duplicated().sum()),
        'time_span_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600,
        'expected_hours': 8760,
        'completeness_pct': round((1 - df.isnull().sum().sum() / (len(df) * len(energy_vars))) * 100, 2)
    }
    quality_report[bid] = report
    
    # Check for outliers using IQR method
    for var in energy_vars:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = ((df[var] < lower_bound) | (df[var] > upper_bound)).sum()
        report[f'{var}_outliers'] = int(outliers)

# Save quality report
with open(os.path.join(OUTPUTS_DIR, 'data_quality_report.json'), 'w') as f:
    json.dump(quality_report, f, indent=2)
print("Data quality report saved.")

# Summary statistics
print("\nSummary Statistics (BN001 example):")
print(energy_data['BN001'][energy_vars].describe())

# ============================================================
# 3. HIERARCHICAL AGGREGATION VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Hierarchical Aggregation Verification")
print("=" * 60)

# Verify if CN01 ≈ sum(BN001:BN010)
hierarchy_results = {}
for var in energy_vars:
    # Sum of all buildings
    building_sum = sum(energy_data[bid][var] for bid in building_ids)
    
    # Compare with CN01
    cn01_vals = cn01_data[var]
    diff_cn01 = building_sum - cn01_vals
    rmse_cn01 = np.sqrt(mean_squared_error(cn01_vals, building_sum))
    mae_cn01 = np.mean(np.abs(diff_cn01))
    corr_cn01 = np.corrcoef(cn01_vals, building_sum)[0, 1]
    
    # Compare with Total
    total_vals = total_energy[var]
    diff_total = building_sum - total_vals
    rmse_total = np.sqrt(mean_squared_error(total_vals, building_sum))
    mae_total = np.mean(np.abs(diff_total))
    corr_total = np.corrcoef(total_vals, building_sum)[0, 1]
    
    hierarchy_results[var] = {
        'CN01_vs_Buildings_Sum': {
            'RMSE': float(rmse_cn01),
            'MAE': float(mae_cn01),
            'correlation': float(corr_cn01),
            'max_difference': float(diff_cn01.abs().max())
        },
        'Total_vs_Buildings_Sum': {
            'RMSE': float(rmse_total),
            'MAE': float(mae_total),
            'correlation': float(corr_total),
            'max_difference': float(diff_total.abs().max())
        }
    }
    
    print(f"\n{var}:")
    print(f"  CN01 vs Buildings: RMSE={rmse_cn01:.4f}, MAE={mae_cn01:.4f}, r={corr_cn01:.6f}")
    print(f"  Total vs Buildings: RMSE={rmse_total:.4f}, MAE={mae_total:.4f}, r={corr_total:.6f}")

with open(os.path.join(OUTPUTS_DIR, 'hierarchy_verification.json'), 'w') as f:
    json.dump(hierarchy_results, f, indent=2)
print("\nHierarchy verification saved.")

# ============================================================
# 4. TEMPORAL PATTERN ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Temporal Pattern Analysis")
print("=" * 60)

# Create datetime features
def add_temporal_features(df):
    df = df.copy()
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

# Apply to total energy and weather
total_combined = total_energy.copy()
total_combined = add_temporal_features(total_combined)
weather_combined = weather_data.copy()
weather_temporal = add_temporal_features(weather_combined)
# Only keep unique weather temporal columns to avoid duplicates
for col in ['hour_of_day', 'day_of_week', 'month', 'day_of_year', 'is_weekend']:
    if col in weather_temporal.columns:
        weather_temporal = weather_temporal.drop(columns=[col])

# Merge energy and weather
full_dataset = pd.concat([total_combined, weather_temporal], axis=1)

# Diurnal patterns (hourly averages)
diurnal_patterns = {}
for var in energy_vars:
    hourly_avg = full_dataset.groupby('hour_of_day')[var].agg(['mean', 'std', 'min', 'max'])
    diurnal_patterns[var] = hourly_avg.to_dict('index')

# Seasonal patterns (monthly averages)
seasonal_patterns = {}
for var in energy_vars:
    monthly_avg = full_dataset.groupby('month')[var].agg(['mean', 'std', 'min', 'max'])
    seasonal_patterns[var] = monthly_avg.to_dict('index')

# Save temporal patterns
with open(os.path.join(OUTPUTS_DIR, 'temporal_patterns.json'), 'w') as f:
    json.dump({
        'diurnal': diurnal_patterns,
        'seasonal': seasonal_patterns
    }, f, indent=2)
print("Temporal patterns saved.")

# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Correlation Analysis")
print("=" * 60)

# Correlation matrix for all variables
all_vars = energy_vars + weather_vars
corr_matrix = full_dataset[all_vars].corr()

# Save correlation matrix
corr_matrix.to_csv(os.path.join(OUTPUTS_DIR, 'correlation_matrix.csv'))
print("Correlation matrix saved.")

# Print key correlations
print("\nKey Correlations with Electricity:")
for var in weather_vars:
    corr_val = corr_matrix.loc['Electricity [kW]', var]
    print(f"  {var}: {corr_val:.4f}")

print("\nKey Correlations with Cooling Energy:")
for var in weather_vars:
    corr_val = corr_matrix.loc['Cooling Energy [Ton]', var]
    print(f"  {var}: {corr_val:.4f}")

# ============================================================
# 6. BUILDING CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Building Clustering")
print("=" * 60)

# Feature extraction for clustering: mean hourly profiles per building
building_profiles = []
for bid in building_ids:
    df = energy_data[bid].copy()
    df = add_temporal_features(df)
    profile = df.groupby('hour_of_day')[energy_vars].mean().values.flatten()
    building_profiles.append(profile)

building_profiles = np.array(building_profiles)
scaler = StandardScaler()
profiles_scaled = scaler.fit_transform(building_profiles)

# Optimal k using silhouette score
from sklearn.metrics import silhouette_score
silhouette_scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(profiles_scaled)
    score = silhouette_score(profiles_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette score={score:.4f}")

best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal number of clusters: {best_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(profiles_scaled)

cluster_results = {}
for i, bid in enumerate(building_ids):
    cluster_results[bid] = int(cluster_labels[i])
    print(f"  {bid} -> Cluster {cluster_labels[i]}")

with open(os.path.join(OUTPUTS_DIR, 'building_clusters.json'), 'w') as f:
    json.dump({'clusters': cluster_results, 'optimal_k': best_k, 
               'silhouette_scores': silhouette_scores}, f, indent=2)
print("Building clusters saved.")

# ============================================================
# 7. LOAD FORECASTING BASELINE
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: Load Forecasting Baseline")
print("=" * 60)

# Prepare features for forecasting
forecast_features = ['hour_of_day', 'day_of_week', 'month', 'is_weekend',
                     'Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]']
target = 'Electricity [kW]'

# Use building-level data for demonstration
X = full_dataset[forecast_features].values
y = full_dataset[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest baseline
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

rf_metrics = {
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    'MAE': float(np.mean(np.abs(y_test - y_pred))),
    'R2': float(r2_score(y_test, y_pred)),
    'MAPE': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100)
}

print(f"Random Forest Performance:")
for metric, value in rf_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Feature importance
feature_importance = dict(zip(forecast_features, rf_model.feature_importances_.tolist()))
print("\nFeature Importance:")
for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

with open(os.path.join(OUTPUTS_DIR, 'forecasting_results.json'), 'w') as f:
    json.dump({'metrics': rf_metrics, 'feature_importance': feature_importance}, f, indent=2)
print("Forecasting results saved.")

# ============================================================
# 8. ANOMALY DETECTION DEMONSTRATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 8: Anomaly Detection")
print("=" * 60)

# Z-score based anomaly detection for electricity
z_scores = np.abs(stats.zscore(full_dataset['Electricity [kW]']))
anomaly_threshold = 3.0
anomalies = full_dataset[z_scores > anomaly_threshold]
anomaly_indices = full_dataset.index[z_scores > anomaly_threshold]

print(f"Detected {len(anomalies)} anomalies (|z| > {anomaly_threshold})")
print(f"Anomaly rate: {len(anomalies)/len(full_dataset)*100:.2f}%")

# Save anomaly details
anomaly_report = {
    'count': int(len(anomalies)),
    'rate_pct': float(len(anomalies)/len(full_dataset)*100),
    'threshold': float(anomaly_threshold),
    'sample_anomalies': [
        {'timestamp': str(idx), 'electricity_kw': float(full_dataset.loc[idx, 'Electricity [kW]']),
         'z_score': float(z_scores[i])}
        for i, idx in enumerate(anomaly_indices[:20])
    ]
}

with open(os.path.join(OUTPUTS_DIR, 'anomaly_detection.json'), 'w') as f:
    json.dump(anomaly_report, f, indent=2)
print("Anomaly detection results saved.")

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Outputs saved to: {OUTPUTS_DIR}")
print(f"Figures will be saved to: {IMAGES_DIR}")
