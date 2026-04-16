#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Comprehensive Analysis Script
================================================
Replicates core experiments from the HEEW paper including:
1. Data loading, cleaning, and quality assessment
2. Descriptive statistics and data overview
3. Temporal pattern analysis (daily, weekly, seasonal)
4. Correlation analysis between energy and weather variables
5. Hierarchical aggregation consistency verification
6. Building-level clustering analysis
7. Anomaly detection demonstration
8. Data imputation benchmark
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = '../data/HEEW_Mini-Dataset'
OUTPUT_DIR = '../outputs'
IMAGE_DIR = '../report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set style
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
sns.set_style("whitegrid")

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

building_ids = [f'BN{i:03d}' for i in range(1, 11)]
energy_dfs = {}
for bid in building_ids:
    df = pd.read_csv(f'{DATA_DIR}/{bid}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df = df.set_index('datetime')
    energy_dfs[bid] = df

cn01_df = pd.read_csv(f'{DATA_DIR}/CN01_energy.csv')
cn01_df['datetime'] = pd.to_datetime(cn01_df[['year','month','day','hour']])
cn01_df = cn01_df.set_index('datetime')

total_df = pd.read_csv(f'{DATA_DIR}/Total_energy.csv')
total_df['datetime'] = pd.to_datetime(total_df[['year','month','day','hour']])
total_df = total_df.set_index('datetime')

weather_df = pd.read_csv(f'{DATA_DIR}/Total_weather.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df = weather_df.set_index('datetime')

print(f"Loaded {len(building_ids)} building datasets")
print(f"Each building: {energy_dfs['BN001'].shape[0]} hourly records")
print(f"Community (CN01): {cn01_df.shape[0]} records")
print(f"Total area: {total_df.shape[0]} records")
print(f"Weather: {weather_df.shape[0]} records")
print(f"Date range: {energy_dfs['BN001'].index.min()} to {energy_dfs['BN001'].index.max()}")

# =============================================================================
# 2. DATA CLEANING AND QUALITY ASSESSMENT
# =============================================================================
print("\n" + "=" * 60)
print("2. DATA CLEANING AND QUALITY ASSESSMENT")
print("=" * 60)

# Check for missing values
quality_report = {}
for bid in building_ids:
    df = energy_dfs[bid]
    missing = df.isnull().sum()
    negative_pv = (df['PV Power Generation [kW]'] < 0).sum()
    negative_elec = (df['Electricity [kW]'] < 0).sum()
    quality_report[bid] = {
        'missing_electricity': missing['Electricity [kW]'],
        'missing_heat': missing['Heat [mmBTU]'],
        'missing_cooling': missing['Cooling Energy [Ton]'],
        'missing_pv': missing['PV Power Generation [kW]'],
        'missing_ghg': missing['Greenhouse Gas Emission [Ton]'],
        'negative_pv_count': negative_pv,
        'negative_electricity_count': negative_elec,
        'total_records': len(df),
        'completeness_pct': (1 - df.isnull().any(axis=1).sum() / len(df)) * 100
    }

quality_df = pd.DataFrame(quality_report).T
print("Data Quality Report:")
print(quality_df.to_string())
quality_df.to_csv(f'{OUTPUT_DIR}/data_quality_report.csv')

# Weather data quality
weather_missing = weather_df.isnull().sum()
print(f"\nWeather data missing values:")
print(weather_missing.to_string())

# Check for duplicates
for bid in building_ids:
    dup = energy_dfs[bid].index.duplicated().sum()
    print(f"{bid}: {dup} duplicate timestamps")

# Outlier detection using IQR method
outlier_report = {}
energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']

for bid in building_ids:
    df = energy_dfs[bid]
    outlier_counts = {}
    for col in energy_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
        upper = Q3 + 3 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = outliers
    outlier_report[bid] = outlier_counts

outlier_df = pd.DataFrame(outlier_report).T
print("\nOutlier counts (3*IQR method):")
print(outlier_df.to_string())
outlier_df.to_csv(f'{OUTPUT_DIR}/outlier_report.csv')

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 60)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 60)

# Compute descriptive stats for all buildings
all_stats = []
for bid in building_ids:
    df = energy_dfs[bid]
    stats_dict = {'Building': bid}
    for col in energy_cols:
        stats_dict[f'{col}_mean'] = df[col].mean()
        stats_dict[f'{col}_std'] = df[col].std()
        stats_dict[f'{col}_min'] = df[col].min()
        stats_dict[f'{col}_max'] = df[col].max()
        stats_dict[f'{col}_median'] = df[col].median()
    all_stats.append(stats_dict)

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(f'{OUTPUT_DIR}/descriptive_statistics.csv', index=False)
print("Descriptive statistics saved")

# Weather statistics
weather_stats = weather_df.describe()
weather_stats.to_csv(f'{OUTPUT_DIR}/weather_statistics.csv')
print("Weather statistics saved")

# =============================================================================
# 4. FIGURE: DATA OVERVIEW - BUILDING LOAD PROFILES
# =============================================================================
print("\n" + "=" * 60)
print("4. GENERATING FIGURES")
print("=" * 60)

# Figure 1: Weekly load profiles for all buildings
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()
for i, bid in enumerate(building_ids):
    df = energy_dfs[bid]
    # Plot one week in January
    week_data = df.loc['2014-01-06':'2014-01-12']
    axes[i].plot(week_data.index, week_data['Electricity [kW]'], label='Electricity', color='#2196F3', linewidth=1)
    axes[i].plot(week_data.index, week_data['Heat [mmBTU]'], label='Heat', color='#FF5722', linewidth=1)
    axes[i].plot(week_data.index, week_data['Cooling Energy [Ton]'], label='Cooling', color='#4CAF50', linewidth=1)
    axes[i].set_title(f'{bid} - Weekly Load Profile (Jan 6-12, 2014)')
    axes[i].legend(fontsize=8)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
plt.suptitle('Building-Level Weekly Load Profiles', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig01_weekly_load_profiles.png')
plt.close()
print("Figure 1 saved: Weekly load profiles")

# Figure 2: Annual load duration curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
load_types = [('Electricity [kW]', 'Electricity (kW)', '#2196F3'),
              ('Heat [mmBTU]', 'Heat (mmBTU)', '#FF5722'),
              ('Cooling Energy [Ton]', 'Cooling (Ton)', '#4CAF50')]

for ax, (col, label, color) in zip(axes, load_types):
    for bid in building_ids:
        sorted_vals = energy_dfs[bid][col].sort_values(ascending=False).values
        ax.plot(range(len(sorted_vals)), sorted_vals, label=bid, linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Hours')
    ax.set_ylabel(label)
    ax.set_title(f'Load Duration Curve - {label}')
    ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig02_load_duration_curves.png')
plt.close()
print("Figure 2 saved: Load duration curves")

# Figure 3: Seasonal daily patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11]}
season_colors = {'Winter': '#2196F3', 'Spring': '#4CAF50', 'Summer': '#FF9800', 'Fall': '#FF5722'}

for idx, (season, months) in enumerate(seasons.items()):
    ax = axes[idx // 2][idx % 2]
    for bid in ['BN001', 'BN005', 'BN010']:
        df = energy_dfs[bid]
        seasonal = df[df['month'].isin(months)]
        hourly_avg = seasonal.groupby('hour')['Electricity [kW]'].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, label=bid, marker='o', markersize=3, linewidth=1.5)
    ax.set_title(f'{season} - Average Hourly Electricity')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity (kW)')
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
plt.suptitle('Seasonal Daily Electricity Patterns', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig03_seasonal_daily_patterns.png')
plt.close()
print("Figure 3 saved: Seasonal daily patterns")

# Figure 4: PV generation pattern
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Monthly PV generation
monthly_pv = pd.DataFrame()
for bid in building_ids:
    df = energy_dfs[bid]
    monthly = df.groupby('month')['PV Power Generation [kW]'].mean()
    monthly_pv[bid] = monthly

monthly_pv.plot(ax=axes[0], marker='o')
axes[0].set_title('Monthly Average PV Generation by Building')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('PV Generation (kW)')
axes[0].legend(fontsize=7, ncol=2)

# Hourly PV generation in summer
for bid in ['BN001', 'BN005', 'BN010']:
    df = energy_dfs[bid]
    summer = df[df['month'].isin([6, 7, 8])]
    hourly_pv = summer.groupby('hour')['PV Power Generation [kW]'].mean()
    axes[1].plot(hourly_pv.index, hourly_pv.values, label=bid, marker='o', markersize=3)
axes[1].set_title('Summer Hourly PV Generation Profile')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('PV Generation (kW)')
axes[1].legend()
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig04_pv_generation_patterns.png')
plt.close()
print("Figure 4 saved: PV generation patterns")

# Figure 5: Weather data overview
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
weather_cols = ['Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]', 
                'Pressure [in]', 'Precipitation [in]', 'Wind Gust [mph]']
colors = ['#FF5722', '#2196F3', '#4CAF50', '#9C27B0', '#00BCD4', '#FF9800']

for idx, (col, color) in enumerate(zip(weather_cols, colors)):
    ax = axes[idx // 2][idx % 2]
    ax.plot(weather_df.index, weather_df[col], color=color, linewidth=0.5, alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Weather Variables - 2014 Annual Time Series', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig05_weather_overview.png')
plt.close()
print("Figure 5 saved: Weather overview")

# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("5. CORRELATION ANALYSIS")
print("=" * 60)

# Merge energy and weather data for correlation analysis
merged_df = total_df.copy()
for col in weather_df.columns:
    merged_df[f'Weather_{col}'] = weather_df[col]

# Compute correlation matrix
corr_cols = energy_cols + [f'Weather_{c}' for c in weather_df.columns]
corr_matrix = merged_df[corr_cols].corr()

# Save correlation matrix
corr_matrix.to_csv(f'{OUTPUT_DIR}/correlation_matrix.csv')

# Figure 6: Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
short_labels = ['Electricity', 'Heat', 'Cooling', 'PV', 'GHG', 
                'Temp', 'DewPoint', 'Humidity', 'WindSpeed', 'WindGust', 'Pressure', 'Precip']
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax,
            vmin=-1, vmax=1, square=True, linewidths=0.5)
ax.set_title('Correlation Matrix: Energy & Weather Variables (Total Area)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig06_correlation_heatmap.png')
plt.close()
print("Figure 6 saved: Correlation heatmap")

# Building-level correlation with temperature
temp_corr = {}
for bid in building_ids:
    df = energy_dfs[bid].copy()
    df['Temperature'] = weather_df['Temperature [°F]'].values
    corr_vals = {}
    for col in energy_cols:
        corr_vals[col] = df[col].corr(df['Temperature'])
    temp_corr[bid] = corr_vals

temp_corr_df = pd.DataFrame(temp_corr).T
temp_corr_df.to_csv(f'{OUTPUT_DIR}/temperature_correlations.csv')

# Figure 7: Temperature correlation by building
fig, ax = plt.subplots(figsize=(12, 6))
short_energy = ['Electricity', 'Heat', 'Cooling', 'PV', 'GHG']
temp_corr_df.columns = short_energy
temp_corr_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Correlation with Temperature by Building')
ax.set_ylabel('Pearson Correlation')
ax.set_xlabel('Building')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend(title='Variable')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig07_temperature_correlation.png')
plt.close()
print("Figure 7 saved: Temperature correlation by building")

# Figure 8: Scatter plots - Temperature vs Energy variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
scatter_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
                'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
scatter_labels = ['Electricity', 'Heat', 'Cooling', 'PV', 'GHG']

for idx, (col, label) in enumerate(zip(scatter_cols, scatter_labels)):
    ax = axes[idx // 3][idx % 3]
    x = weather_df['Temperature [°F]'].values
    y = total_df[col].values
    ax.scatter(x, y, alpha=0.1, s=3, color='#2196F3')
    # Add trend line
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), 'r-', linewidth=2, label='Quadratic fit')
    ax.set_xlabel('Temperature (°F)')
    ax.set_ylabel(label)
    ax.set_title(f'Temperature vs {label}')
    ax.legend()

axes[1][2].axis('off')
plt.suptitle('Temperature vs Energy Variables (Total Area)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig08_temperature_scatter.png')
plt.close()
print("Figure 8 saved: Temperature scatter plots")

# =============================================================================
# 6. HIERARCHICAL AGGREGATION CONSISTENCY
# =============================================================================
print("\n" + "=" * 60)
print("6. HIERARCHICAL AGGREGATION CONSISTENCY")
print("=" * 60)

# Verify that sum of buildings ≈ CN01 ≈ Total
building_sum = pd.DataFrame()
for bid in building_ids:
    building_sum = building_sum.add(energy_dfs[bid][energy_cols], fill_value=0)

# Consistency metrics
consistency_results = {}
for col in energy_cols:
    # Building sum vs CN01
    mae_bn_cn = mean_absolute_error(building_sum[col].values, cn01_df[col].values)
    rmse_bn_cn = np.sqrt(mean_squared_error(building_sum[col].values, cn01_df[col].values))
    
    # CN01 vs Total
    mae_cn_total = mean_absolute_error(cn01_df[col].values, total_df[col].values)
    rmse_cn_total = np.sqrt(mean_squared_error(cn01_df[col].values, total_df[col].values))
    
    # Building sum vs Total
    mae_bn_total = mean_absolute_error(building_sum[col].values, total_df[col].values)
    rmse_bn_total = np.sqrt(mean_squared_error(building_sum[col].values, total_df[col].values))
    
    consistency_results[col] = {
        'MAE_BN_sum_vs_CN01': mae_bn_cn,
        'RMSE_BN_sum_vs_CN01': rmse_bn_cn,
        'MAE_CN01_vs_Total': mae_cn_total,
        'RMSE_CN01_vs_Total': rmse_cn_total,
        'MAE_BN_sum_vs_Total': mae_bn_total,
        'RMSE_BN_sum_vs_Total': rmse_bn_total,
        'MeanAbsPctError_BN_vs_Total': np.mean(np.abs(building_sum[col].values - total_df[col].values) / (total_df[col].values + 1e-10)) * 100
    }

consistency_df = pd.DataFrame(consistency_results).T
consistency_df.to_csv(f'{OUTPUT_DIR}/hierarchical_consistency.csv')
print("Hierarchical consistency results:")
print(consistency_df.to_string())

# Figure 9: Hierarchical aggregation consistency
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (col, label) in enumerate(zip(scatter_cols, scatter_labels)):
    ax = axes[idx // 3][idx % 3]
    # Plot one week
    week_idx = total_df.loc['2014-06-02':'2014-06-08'].index
    ax.plot(week_idx, total_df.loc[week_idx, col], label='Total', color='#2196F3', linewidth=2)
    ax.plot(week_idx, cn01_df.loc[week_idx, col], label='CN01', color='#FF5722', linewidth=1.5, linestyle='--')
    ax.plot(week_idx, building_sum.loc[week_idx, col], label='BN Sum', color='#4CAF50', linewidth=1, linestyle=':')
    ax.set_title(f'{label} - Hierarchical Comparison')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

axes[1][2].axis('off')
plt.suptitle('Hierarchical Aggregation Consistency (June 2-8, 2014)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig09_hierarchical_consistency.png')
plt.close()
print("Figure 9 saved: Hierarchical consistency")

# Figure 10: Parity plot for hierarchical aggregation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
parity_pairs = [
    (building_sum, total_df, 'Building Sum vs Total'),
    (cn01_df, total_df, 'CN01 vs Total'),
    (building_sum, cn01_df, 'Building Sum vs CN01')
]

for ax, (df1, df2, title) in zip(axes, parity_pairs):
    for col, color, label in zip(
        ['Electricity [kW]', 'Cooling Energy [Ton]', 'Heat [mmBTU]'],
        ['#2196F3', '#4CAF50', '#FF5722'],
        ['Electricity', 'Cooling', 'Heat']):
        ax.scatter(df1[col].values, df2[col].values, alpha=0.1, s=2, color=color, label=label)
    
    max_val = max(df1[energy_cols[:3]].values.max(), df2[energy_cols[:3]].values.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    ax.set_xlabel('Source 1')
    ax.set_ylabel('Source 2')
    ax.set_title(title)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig10_parity_plots.png')
plt.close()
print("Figure 10 saved: Parity plots")

# =============================================================================
# 7. BUILDING CLUSTERING ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("7. BUILDING CLUSTERING ANALYSIS")
print("=" * 60)

# Feature extraction for clustering
features = []
for bid in building_ids:
    df = energy_dfs[bid]
    feat = {'Building': bid}
    for col in energy_cols:
        feat[f'{col}_mean'] = df[col].mean()
        feat[f'{col}_std'] = df[col].std()
        feat[f'{col}_cv'] = df[col].std() / (df[col].mean() + 1e-10)  # coefficient of variation
        feat[f'{col}_max'] = df[col].max()
        feat[f'{col}_min'] = df[col].min()
        feat[f'{col}_range'] = df[col].max() - df[col].min()
    
    # Peak hour analysis
    hourly_elec = df.groupby('hour')['Electricity [kW]'].mean()
    feat['peak_hour_electricity'] = hourly_elec.idxmax()
    feat['peak_electricity'] = hourly_elec.max()
    feat['base_electricity'] = hourly_elec.min()
    feat['peak_base_ratio'] = hourly_elec.max() / (hourly_elec.min() + 1e-10)
    
    # Seasonal variation
    monthly_elec = df.groupby('month')['Electricity [kW]'].mean()
    feat['seasonal_range_electricity'] = monthly_elec.max() - monthly_elec.min()
    
    features.append(feat)

feature_df = pd.DataFrame(features).set_index('Building')
feature_df.to_csv(f'{OUTPUT_DIR}/clustering_features.csv')

# Standardize features for clustering
numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df[numeric_cols])

# Hierarchical clustering
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Figure 11: Dendrogram
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(Z, labels=building_ids, ax=ax, leaf_rotation=45, leaf_font_size=10)
ax.set_title('Hierarchical Clustering Dendrogram of Buildings')
ax.set_xlabel('Building')
ax.set_ylabel('Distance (Ward linkage)')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig11_clustering_dendrogram.png')
plt.close()
print("Figure 11 saved: Clustering dendrogram")

# K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
feature_df['Cluster'] = clusters

# Save cluster assignments
cluster_df = feature_df[['Cluster']].copy()
cluster_df.to_csv(f'{OUTPUT_DIR}/cluster_assignments.csv')

# Figure 12: Cluster visualization (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
colors_cluster = ['#2196F3', '#FF5722', '#4CAF50']
for c in range(3):
    mask = clusters == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_cluster[c], 
               s=150, label=f'Cluster {c+1}', edgecolors='black', linewidth=0.5)
    for i, bid in enumerate(building_ids):
        if clusters[i] == c:
            ax.annotate(bid, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, 
                        xytext=(5, 5), textcoords='offset points')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Building Clustering (K-Means, k=3) - PCA Projection')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig12_cluster_pca.png')
plt.close()
print("Figure 12 saved: Cluster PCA visualization")

# Figure 13: Cluster profiles
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for c in range(3):
    ax = axes[c]
    cluster_buildings = [building_ids[i] for i in range(len(building_ids)) if clusters[i] == c]
    for bid in cluster_buildings:
        df = energy_dfs[bid]
        hourly_avg = df.groupby('hour')['Electricity [kW]'].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, label=bid, marker='o', markersize=2)
    ax.set_title(f'Cluster {c+1} - Average Hourly Electricity')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity (kW)')
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, 24, 3))

plt.suptitle('Building Cluster Load Profiles', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig13_cluster_profiles.png')
plt.close()
print("Figure 13 saved: Cluster profiles")

# =============================================================================
# 8. ANOMALY DETECTION
# =============================================================================
print("\n" + "=" * 60)
print("8. ANOMALY DETECTION")
print("=" * 60)

# Isolation Forest on BN001
df_anomaly = energy_dfs['BN001'][energy_cols].copy()
iso_forest = IsolationForest(contamination=0.02, random_state=42)
anomaly_labels = iso_forest.fit_predict(df_anomaly)
df_anomaly['anomaly'] = anomaly_labels == -1

anomaly_count = df_anomaly['anomaly'].sum()
print(f"BN001: {anomaly_count} anomalies detected ({anomaly_count/len(df_anomaly)*100:.2f}%)")

# Figure 14: Anomaly detection visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
anomaly_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 'PV Power Generation [kW]']
anomaly_labels_short = ['Electricity', 'Heat', 'Cooling', 'PV']

for idx, (col, label) in enumerate(zip(anomaly_cols, anomaly_labels_short)):
    ax = axes[idx // 2][idx % 2]
    normal = df_anomaly[~df_anomaly['anomaly']]
    anomaly = df_anomaly[df_anomaly['anomaly']]
    ax.scatter(normal.index, normal[col], c='#2196F3', s=3, alpha=0.3, label='Normal')
    ax.scatter(anomaly.index, anomaly[col], c='#FF5722', s=10, alpha=0.7, label='Anomaly')
    ax.set_title(f'BN001 - Anomaly Detection: {label}')
    ax.set_xlabel('Date')
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Isolation Forest Anomaly Detection (BN001)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig14_anomaly_detection.png')
plt.close()
print("Figure 14 saved: Anomaly detection")

# =============================================================================
# 9. DATA IMPUTATION BENCHMARK
# =============================================================================
print("\n" + "=" * 60)
print("9. DATA IMPUTATION BENCHMARK")
print("=" * 60)

# Create artificial missing data and test imputation methods
np.random.seed(42)
df_impute = energy_dfs['BN001'][energy_cols].copy()

missing_rates = [0.05, 0.10, 0.20]
imputation_results = []

for rate in missing_rates:
    n_missing = int(len(df_impute) * rate)
    missing_idx = np.random.choice(len(df_impute), n_missing, replace=False)
    
    for col in energy_cols:
        true_values = df_impute[col].iloc[missing_idx].values
        
        # Linear interpolation
        df_test = df_impute[col].copy()
        df_test.iloc[missing_idx] = np.nan
        df_interp = df_test.interpolate(method='linear')
        predicted_interp = df_interp.iloc[missing_idx].values
        mae_interp = mean_absolute_error(true_values, predicted_interp)
        rmse_interp = np.sqrt(mean_squared_error(true_values, predicted_interp))
        
        # Forward fill
        df_ffill = df_test.fillna(method='ffill').fillna(method='bfill')
        predicted_ffill = df_ffill.iloc[missing_idx].values
        mae_ffill = mean_absolute_error(true_values, predicted_ffill)
        rmse_ffill = np.sqrt(mean_squared_error(true_values, predicted_ffill))
        
        # Mean imputation
        df_mean = df_test.fillna(df_test.mean())
        predicted_mean = df_mean.iloc[missing_idx].values
        mae_mean = mean_absolute_error(true_values, predicted_mean)
        rmse_mean = np.sqrt(mean_squared_error(true_values, predicted_mean))
        
        imputation_results.append({
            'Missing_Rate': rate,
            'Variable': col,
            'Method': 'Linear_Interpolation',
            'MAE': mae_interp,
            'RMSE': rmse_interp
        })
        imputation_results.append({
            'Missing_Rate': rate,
            'Variable': col,
            'Method': 'Forward_Fill',
            'MAE': mae_ffill,
            'RMSE': rmse_ffill
        })
        imputation_results.append({
            'Missing_Rate': rate,
            'Variable': col,
            'Method': 'Mean_Imputation',
            'MAE': mae_mean,
            'RMSE': rmse_mean
        })

impute_df = pd.DataFrame(imputation_results)
impute_df.to_csv(f'{OUTPUT_DIR}/imputation_benchmark.csv', index=False)
print("Imputation benchmark results saved")

# Figure 15: Imputation benchmark
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
methods = ['Linear_Interpolation', 'Forward_Fill', 'Mean_Imputation']
method_colors = {'Linear_Interpolation': '#2196F3', 'Forward_Fill': '#FF5722', 'Mean_Imputation': '#4CAF50'}

for method in methods:
    method_data = impute_df[impute_df['Method'] == method]
    # Average across variables for each missing rate
    avg_mae = method_data.groupby('Missing_Rate')['MAE'].mean()
    avg_rmse = method_data.groupby('Missing_Rate')['RMSE'].mean()
    axes[0].plot(avg_mae.index, avg_mae.values, marker='o', label=method, color=method_colors[method])
    axes[1].plot(avg_rmse.index, avg_rmse.values, marker='o', label=method, color=method_colors[method])

axes[0].set_title('Imputation MAE vs Missing Rate')
axes[0].set_xlabel('Missing Rate')
axes[0].set_ylabel('MAE (averaged across variables)')
axes[0].legend()

axes[1].set_title('Imputation RMSE vs Missing Rate')
axes[1].set_xlabel('Missing Rate')
axes[1].set_ylabel('RMSE (averaged across variables)')
axes[1].legend()

plt.suptitle('Data Imputation Benchmark (BN001)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig15_imputation_benchmark.png')
plt.close()
print("Figure 15 saved: Imputation benchmark")

# =============================================================================
# 10. GHG EMISSIONS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("10. GHG EMISSIONS ANALYSIS")
print("=" * 60)

# Figure 16: GHG emissions breakdown
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Monthly GHG by building
monthly_ghg = pd.DataFrame()
for bid in building_ids:
    df = energy_dfs[bid]
    monthly_ghg[bid] = df.groupby('month')['Greenhouse Gas Emission [Ton]'].mean()

monthly_ghg.plot(ax=axes[0], legend=False)
axes[0].set_title('Monthly Average GHG Emissions by Building')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('GHG Emission (Ton)')
axes[0].legend(fontsize=7, ncol=2)

# Total GHG by building (annual)
annual_ghg = {}
for bid in building_ids:
    annual_ghg[bid] = energy_dfs[bid]['Greenhouse Gas Emission [Ton]'].sum()
    
axes[1].bar(annual_ghg.keys(), annual_ghg.values(), color='#FF5722', alpha=0.7)
axes[1].set_title('Annual GHG Emissions by Building')
axes[1].set_xlabel('Building')
axes[1].set_ylabel('Total GHG Emission (Ton)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig16_ghg_emissions.png')
plt.close()
print("Figure 16 saved: GHG emissions")

# =============================================================================
# 11. ENERGY-WEATHER LAGGED CORRELATION
# =============================================================================
print("\n" + "=" * 60)
print("11. LAGGED CORRELATION ANALYSIS")
print("=" * 60)

# Cross-correlation between temperature and cooling load
max_lag = 48  # 48 hours
lags = range(-max_lag, max_lag + 1)
ccf_cooling = []
ccf_electricity = []
ccf_heat = []

temp = weather_df['Temperature [°F]'].values
cooling = total_df['Cooling Energy [Ton]'].values
electricity = total_df['Electricity [kW]'].values
heat = total_df['Heat [mmBTU]'].values

for lag in lags:
    if lag >= 0:
        ccf_cooling.append(np.corrcoef(temp[:len(temp)-lag] if lag > 0 else temp, 
                                        cooling[lag:] if lag > 0 else cooling)[0, 1])
        ccf_electricity.append(np.corrcoef(temp[:len(temp)-lag] if lag > 0 else temp, 
                                            electricity[lag:] if lag > 0 else electricity)[0, 1])
        ccf_heat.append(np.corrcoef(temp[:len(temp)-lag] if lag > 0 else temp, 
                                     heat[lag:] if lag > 0 else heat)[0, 1])
    else:
        ccf_cooling.append(np.corrcoef(temp[-lag:], cooling[:len(cooling)+lag])[0, 1])
        ccf_electricity.append(np.corrcoef(temp[-lag:], electricity[:len(electricity)+lag])[0, 1])
        ccf_heat.append(np.corrcoef(temp[-lag:], heat[:len(heat)+lag])[0, 1])

# Figure 17: Lagged correlation
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(list(lags), ccf_cooling, label='Cooling', color='#4CAF50', linewidth=2)
ax.plot(list(lags), ccf_electricity, label='Electricity', color='#2196F3', linewidth=2)
ax.plot(list(lags), ccf_heat, label='Heat', color='#FF5722', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Cross-Correlation')
ax.set_title('Cross-Correlation: Temperature vs Energy Variables')
ax.legend()
ax.set_xlim(-max_lag, max_lag)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/fig17_lagged_correlation.png')
plt.close()
print("Figure 17 saved: Lagged correlation")

# =============================================================================
# 12. COMPREHENSIVE SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 60)
print("12. GENERATING SUMMARY STATISTICS")
print("=" * 60)

# Dataset summary
summary = {
    'Total_buildings': len(building_ids),
    'Total_records_per_building': 8760,
    'Total_records_dataset': 8760 * (len(building_ids) + 2),  # +2 for CN01 and Total
    'Weather_records': len(weather_df),
    'Time_resolution': 'Hourly',
    'Year': 2014,
    'Energy_variables': len(energy_cols),
    'Weather_variables': len(weather_df.columns),
    'Total_variables': len(energy_cols) + len(weather_df.columns),
    'Date_range_start': str(energy_dfs['BN001'].index.min()),
    'Date_range_end': str(energy_dfs['BN001'].index.max()),
    'Data_completeness_pct': quality_df['completeness_pct'].mean(),
    'Hierarchical_levels': 3,  # Building, Community, Total
}

with open(f'{OUTPUT_DIR}/dataset_summary.json', 'w') as f:
    import json
    json.dump(summary, f, indent=2, default=str)

print("Dataset summary saved")
print(json.dumps(summary, indent=2, default=str))

# Save all building statistics
all_building_stats = pd.DataFrame()
for bid in building_ids:
    df = energy_dfs[bid]
    stats = df[energy_cols].describe()
    stats.columns = [c.split('[')[0].strip() for c in stats.columns]
    stats['Building'] = bid
    all_building_stats = pd.concat([all_building_stats, stats.reset_index()], axis=0)

all_building_stats.to_csv(f'{OUTPUT_DIR}/all_building_statistics.csv', index=False)

print("\n" + "=" * 60)
print("ALL ANALYSIS COMPLETE")
print("=" * 60)
