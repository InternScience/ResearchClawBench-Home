#!/usr/bin/env python3
"""
HEEW Mini-Dataset Comprehensive Analysis
Covers: data quality, cleaning, hierarchical aggregation, correlation, clustering, time-series patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import os
import json

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

ENERGY_VARS = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
WEATHER_VARS = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
BUILDING_IDS = [f'BN{i:03d}' for i in range(1, 11)]
AGG_LEVELS = BUILDING_IDS + ['CN01', 'Total']

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("=" * 60)

energy_dfs = {}
for bid in AGG_LEVELS:
    fpath = os.path.join(DATA_DIR, f'{bid}_energy.csv')
    df = pd.read_csv(fpath)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].rename(columns={'hour':'hour'}))
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(year=2014))
    # Fix: construct datetime properly
    df['datetime'] = pd.to_datetime(df[['year','month','day']].assign(hour=df['hour']))
    df = df.set_index('datetime').sort_index()
    energy_dfs[bid] = df
    print(f"  {bid}: {len(df)} rows, columns: {list(df.columns)}")

weather_df = pd.read_csv(os.path.join(DATA_DIR, 'Total_weather.csv'))
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df = weather_df.set_index('datetime').sort_index()
print(f"  Weather: {len(weather_df)} rows")

# ============================================================
# 2. DATA QUALITY ASSESSMENT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: DATA QUALITY ASSESSMENT")
print("=" * 60)

quality_report = {}
for bid in AGG_LEVELS:
    df = energy_dfs[bid]
    report = {
        'total_rows': len(df),
        'missing_values': int(df[ENERGY_VARS].isnull().sum().sum()),
        'negative_electricity': int((df['Electricity [kW]'] < 0).sum()),
        'negative_heat': int((df['Heat [mmBTU]'] < 0).sum()),
        'negative_cooling': int((df['Cooling Energy [Ton]'] < 0).sum()),
        'negative_pv': int((df['PV Power Generation [kW]'] < 0).sum()),
        'zero_pv_hours': int((df['PV Power Generation [kW]'] == 0).sum()),
        'duplicate_timestamps': int(df.index.duplicated().sum()),
    }
    quality_report[bid] = report
    print(f"  {bid}: missing={report['missing_values']}, neg_elec={report['negative_electricity']}, "
          f"neg_heat={report['negative_heat']}, neg_cool={report['negative_cooling']}, "
          f"zero_pv={report['zero_pv_hours']}")

# Weather quality
wq = {
    'total_rows': len(weather_df),
    'missing_values': int(weather_df[WEATHER_VARS].isnull().sum().sum()),
    'negative_temp': int((weather_df['Temperature [°F]'] < 0).sum()),
    'negative_wind': int((weather_df['Wind Speed [mph]'] < 0).sum()),
}
print(f"  Weather: missing={wq['missing_values']}, rows={wq['total_rows']}")

with open(os.path.join(OUTPUT_DIR, 'data_quality_report.json'), 'w') as f:
    json.dump({'energy': quality_report, 'weather': wq}, f, indent=2)

# ============================================================
# 3. DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: DATA CLEANING")
print("=" * 60)

def clean_energy_data(df, var_name):
    """IQR-based outlier detection and linear interpolation for gaps."""
    cleaned = df.copy()
    Q1 = cleaned[var_name].quantile(0.25)
    Q3 = cleaned[var_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((cleaned[var_name] < lower) | (cleaned[var_name] > upper))
    n_outliers = outliers.sum()
    cleaned.loc[outliers, var_name] = np.nan
    cleaned[var_name] = cleaned[var_name].interpolate(method='linear')
    return cleaned, int(n_outliers)

cleaning_stats = {}
for bid in AGG_LEVELS:
    df = energy_dfs[bid]
    stats_dict = {}
    for var in ENERGY_VARS:
        _, n_out = clean_energy_data(df, var)
        stats_dict[var] = n_out
    cleaning_stats[bid] = stats_dict
    total_outliers = sum(stats_dict.values())
    print(f"  {bid}: {int(total_outliers)} total outliers detected")

# Actually clean the data
cleaned_dfs = {}
for bid in AGG_LEVELS:
    df = energy_dfs[bid].copy()
    for var in ENERGY_VARS:
        df_clean, _ = clean_energy_data(df, var)
        df[var] = df_clean[var]
    cleaned_dfs[bid] = df

with open(os.path.join(OUTPUT_DIR, 'cleaning_statistics.json'), 'w') as f:
    json.dump(cleaning_stats, f, indent=2)

# ============================================================
# 4. HIERARCHICAL AGGREGATION VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: HIERARCHICAL AGGREGATION VERIFICATION")
print("=" * 60)

# Check: sum(BN001..BN010) == CN01
bn_sum = sum(cleaned_dfs[bid][ENERGY_VARS] for bid in BUILDING_IDS)
cn01 = cleaned_dfs['CN01'][ENERGY_VARS]
agg_error_cn01 = (bn_sum - cn01).abs()
print("  BN sum vs CN01:")
for var in ENERGY_VARS:
    mae = agg_error_cn01[var].mean()
    max_err = agg_error_cn01[var].max()
    rel_err = (agg_error_cn01[var] / (cn01[var].abs() + 1e-10)).mean() * 100
    print(f"    {var}: MAE={mae:.4f}, MaxErr={max_err:.4f}, RelErr={rel_err:.4f}%")

# Check: CN01 == Total
total = cleaned_dfs['Total'][ENERGY_VARS]
agg_error_total = (cn01 - total).abs()
print("  CN01 vs Total:")
for var in ENERGY_VARS:
    mae = agg_error_total[var].mean()
    print(f"    {var}: MAE={mae:.6f}")

agg_results = {
    'bn_sum_vs_cn01': {var: {'MAE': float(agg_error_cn01[var].mean()), 
                              'MaxErr': float(agg_error_cn01[var].max())} for var in ENERGY_VARS},
    'cn01_vs_total': {var: {'MAE': float(agg_error_total[var].mean())} for var in ENERGY_VARS}
}
with open(os.path.join(OUTPUT_DIR, 'aggregation_verification.json'), 'w') as f:
    json.dump(agg_results, f, indent=2)

# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: DESCRIPTIVE STATISTICS")
print("=" * 60)

desc_stats = {}
for bid in BUILDING_IDS:
    df = cleaned_dfs[bid]
    stats_dict = {}
    for var in ENERGY_VARS:
        stats_dict[var] = {
            'mean': float(df[var].mean()),
            'std': float(df[var].std()),
            'min': float(df[var].min()),
            'max': float(df[var].max()),
            'median': float(df[var].median()),
            'total_annual': float(df[var].sum()),
        }
    desc_stats[bid] = stats_dict
    print(f"  {bid}: Elec_mean={stats_dict['Electricity [kW]']['mean']:.2f} kW, "
          f"Total_elec={stats_dict['Electricity [kW]']['total_annual']:.0f} kWh")

with open(os.path.join(OUTPUT_DIR, 'descriptive_statistics.json'), 'w') as f:
    json.dump(desc_stats, f, indent=2)

# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: CORRELATION ANALYSIS")
print("=" * 60)

# Per-building correlation between energy variables
corr_results = {}
for bid in BUILDING_IDS:
    df = cleaned_dfs[bid]
    corr = df[ENERGY_VARS].corr()
    corr_results[bid] = corr.to_dict()

# Cross-correlation: energy vs weather (using Total)
total_df = cleaned_dfs['Total'].copy()
total_df = total_df.join(weather_df[WEATHER_VARS], how='inner')
cross_corr = total_df[ENERGY_VARS + WEATHER_VARS].corr()
cross_corr.to_csv(os.path.join(OUTPUT_DIR, 'cross_correlation_energy_weather.csv'))
print("  Cross-correlation (energy vs weather) saved")

# Save per-building energy correlations
for bid in BUILDING_IDS:
    corr_results[bid] = cleaned_dfs[bid][ENERGY_VARS].corr().to_dict()

with open(os.path.join(OUTPUT_DIR, 'energy_correlations.json'), 'w') as f:
    json.dump(corr_results, f, indent=2)

# ============================================================
# 7. TIME SERIES PATTERNS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: TIME SERIES PATTERN ANALYSIS")
print("=" * 60)

# Diurnal profiles
hourly_profiles = {}
for bid in BUILDING_IDS:
    df = cleaned_dfs[bid]
    hourly = df.groupby(df.index.hour)[ENERGY_VARS].mean()
    hourly_profiles[bid] = hourly.to_dict()

# Monthly profiles
monthly_profiles = {}
for bid in BUILDING_IDS:
    df = cleaned_dfs[bid]
    monthly = df.groupby(df.index.month)[ENERGY_VARS].mean()
    monthly_profiles[bid] = monthly.to_dict()

with open(os.path.join(OUTPUT_DIR, 'diurnal_profiles.json'), 'w') as f:
    json.dump(hourly_profiles, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'monthly_profiles.json'), 'w') as f:
    json.dump(monthly_profiles, f, indent=2)

# ============================================================
# 8. CLUSTERING ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 8: CLUSTERING ANALYSIS")
print("=" * 60)

# Feature engineering: hourly profiles for each building
feature_matrix = []
for bid in BUILDING_IDS:
    df = cleaned_dfs[bid]
    features = []
    for var in ENERGY_VARS:
        # 24-hour profile
        hourly = df.groupby(df.index.hour)[var].mean().values
        features.extend(hourly)
        # Monthly profile
        monthly = df.groupby(df.index.month)[var].mean().values
        features.extend(monthly)
        # Overall stats
        features.extend([df[var].mean(), df[var].std(), df[var].max()])
    feature_matrix.append(features)

feature_matrix = np.array(feature_matrix)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)

# Determine optimal k
silhouette_scores = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette={score:.4f}")

best_k = np.argmax(silhouette_scores) + 2
print(f"  Best k={best_k}")

# Final clustering
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(features_scaled)

# PCA for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)
explained_var = pca.explained_variance_ratio_

cluster_result = {bid: int(label) for bid, label in zip(BUILDING_IDS, cluster_labels)}
print(f"  Cluster assignments: {cluster_result}")

with open(os.path.join(OUTPUT_DIR, 'clustering_results.json'), 'w') as f:
    json.dump({
        'cluster_labels': cluster_result,
        'best_k': int(best_k),
        'silhouette_scores': {str(k+2): float(v) for k, v in enumerate(silhouette_scores)},
        'pca_explained_variance': explained_var.tolist()
    }, f, indent=2)

# ============================================================
# 9. FIGURES
# ============================================================
print("\n" + "=" * 60)
print("PHASE 9: GENERATING FIGURES")
print("=" * 60)

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# --- Figure 1: Data Overview - Stacked energy by building ---
fig, ax = plt.subplots(figsize=(14, 6))
monthly_elec = pd.DataFrame({bid: cleaned_dfs[bid]['Electricity [kW]'].resample('M').mean() 
                             for bid in BUILDING_IDS})
monthly_elec.plot.area(ax=ax, alpha=0.7, cmap='tab10')
ax.set_title('Monthly Average Electricity Load by Building (2014)')
ax.set_ylabel('Electricity [kW]')
ax.set_xlabel('Month')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_electricity_by_building.png'), bbox_inches='tight')
plt.close()
print("  fig1_electricity_by_building.png")

# --- Figure 2: Multi-energy time series for Total ---
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
total_clean = cleaned_dfs['Total']
for i, var in enumerate(ENERGY_VARS):
    axes[i].plot(total_clean.index, total_clean[var], linewidth=0.5, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'][i])
    axes[i].set_ylabel(var.split('[')[0].strip())
    axes[i].set_title(var)
plt.suptitle('Total Area Energy Variables - Full Year 2014', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_total_energy_timeseries.png'), bbox_inches='tight')
plt.close()
print("  fig2_total_energy_timeseries.png")

# --- Figure 3: Weather variables ---
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']
for i, var in enumerate(WEATHER_VARS):
    axes[i].plot(weather_df.index, weather_df[var], linewidth=0.5, color=colors[i])
    axes[i].set_ylabel(var.split('[')[0].strip())
plt.suptitle('Weather Variables - Full Year 2014', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_weather_timeseries.png'), bbox_inches='tight')
plt.close()
print("  fig3_weather_timeseries.png")

# --- Figure 4: Diurnal profiles ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, var in enumerate(ENERGY_VARS):
    for bid in BUILDING_IDS:
        hourly = cleaned_dfs[bid].groupby(cleaned_dfs[bid].index.hour)[var].mean()
        axes[i].plot(hourly.index, hourly.values, alpha=0.5, linewidth=1)
    axes[i].set_title(var.split('[')[0].strip())
    axes[i].set_xlabel('Hour of Day')
    axes[i].set_ylabel(var.split('[')[1].replace(']','').strip() if '[' in var else '')
if len(ENERGY_VARS) < 6:
    axes[-1].set_visible(False)
plt.suptitle('Average Diurnal Profiles by Building', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_diurnal_profiles.png'), bbox_inches='tight')
plt.close()
print("  fig4_diurnal_profiles.png")

# --- Figure 5: Monthly profiles ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, var in enumerate(ENERGY_VARS):
    for bid in BUILDING_IDS:
        monthly = cleaned_dfs[bid].groupby(cleaned_dfs[bid].index.month)[var].mean()
        axes[i].plot(monthly.index, monthly.values, alpha=0.5, linewidth=1, marker='o', markersize=3)
    axes[i].set_title(var.split('[')[0].strip())
    axes[i].set_xlabel('Month')
if len(ENERGY_VARS) < 6:
    axes[-1].set_visible(False)
plt.suptitle('Average Monthly Profiles by Building', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_monthly_profiles.png'), bbox_inches='tight')
plt.close()
print("  fig5_monthly_profiles.png")

# --- Figure 6: Energy correlation heatmap (Total) ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_total = total_df[ENERGY_VARS + WEATHER_VARS].corr()
mask = np.triu(np.ones_like(corr_total, dtype=bool), k=1)
sns.heatmap(corr_total, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            mask=mask, ax=ax, vmin=-1, vmax=1, square=True)
ax.set_title('Correlation Matrix: Energy & Weather Variables (Total Area)')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("  fig6_correlation_heatmap.png")

# --- Figure 7: Hierarchical aggregation verification ---
fig, axes = plt.subplots(1, len(ENERGY_VARS), figsize=(20, 4))
for i, var in enumerate(ENERGY_VARS):
    axes[i].scatter(bn_sum[var], cn01[var], alpha=0.1, s=1)
    lims = [min(bn_sum[var].min(), cn01[var].min()), max(bn_sum[var].max(), cn01[var].max())]
    axes[i].plot(lims, lims, 'r--', linewidth=1)
    axes[i].set_xlabel('Sum of Buildings')
    axes[i].set_ylabel('CN01')
    axes[i].set_title(var.split('[')[0].strip())
    axes[i].set_aspect('equal')
plt.suptitle('Hierarchical Aggregation: Sum(BN001-BN010) vs CN01', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_aggregation_verification.png'), bbox_inches='tight')
plt.close()
print("  fig7_aggregation_verification.png")

# --- Figure 8: Clustering results ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_cluster = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for i, bid in enumerate(BUILDING_IDS):
    c = colors_cluster[cluster_labels[i]]
    axes[0].scatter(features_2d[i, 0], features_2d[i, 1], c=c, s=100, zorder=5)
    axes[0].annotate(bid, (features_2d[i, 0], features_2d[i, 1]), fontsize=8, 
                     xytext=(5, 5), textcoords='offset points')
axes[0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
axes[0].set_title('Building Clusters (PCA Projection)')

ks = range(2, 6)
axes[1].plot(ks, silhouette_scores, 'bo-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Optimal Cluster Selection')
axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig8_clustering.png'), bbox_inches='tight')
plt.close()
print("  fig8_clustering.png")

# --- Figure 9: Box plots of energy variables per building ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, var in enumerate(ENERGY_VARS):
    data_to_plot = [cleaned_dfs[bid][var].values for bid in BUILDING_IDS]
    bp = axes[i].boxplot(data_to_plot, labels=BUILDING_IDS, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#87CEEB')
    axes[i].set_title(var.split('[')[0].strip())
    axes[i].tick_params(axis='x', rotation=45)
if len(ENERGY_VARS) < 6:
    axes[-1].set_visible(False)
plt.suptitle('Energy Variable Distributions by Building', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig9_boxplots.png'), bbox_inches='tight')
plt.close()
print("  fig9_boxplots.png")

# --- Figure 10: Scatter: Temperature vs Electricity (Total) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, var in enumerate(['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]']):
    axes[i].scatter(total_df['Temperature [°F]'], total_df[var], alpha=0.1, s=1)
    axes[i].set_xlabel('Temperature [°F]')
    axes[i].set_ylabel(var.split('[')[1].replace(']',''))
    axes[i].set_title(f'Temperature vs {var.split("[")[0].strip()}')
plt.suptitle('Temperature Dependence of Energy Loads (Total Area)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig10_temperature_dependence.png'), bbox_inches='tight')
plt.close()
print("  fig10_temperature_dependence.png")

# --- Figure 11: PV generation patterns ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# Monthly PV
monthly_pv = pd.DataFrame({bid: cleaned_dfs[bid]['PV Power Generation [kW]'].resample('M').sum() 
                           for bid in BUILDING_IDS})
monthly_pv.plot.bar(ax=axes[0,0], alpha=0.8)
axes[0,0].set_title('Monthly PV Generation (Total)')
axes[0,0].set_ylabel('PV [kW]')
axes[0,0].legend(fontsize=6)
# Diurnal PV
for bid in BUILDING_IDS:
    hourly_pv = cleaned_dfs[bid].groupby(cleaned_dfs[bid].index.hour)['PV Power Generation [kW]'].mean()
    axes[0,1].plot(hourly_pv.index, hourly_pv.values, alpha=0.6)
axes[0,1].set_title('Average Diurnal PV Profile')
axes[0,1].set_xlabel('Hour')
# PV vs Temperature
axes[1,0].scatter(total_df['Temperature [°F]'], total_df['PV Power Generation [kW]'], alpha=0.1, s=1)
axes[1,0].set_xlabel('Temperature [°F]')
axes[1,0].set_ylabel('PV [kW]')
axes[1,0].set_title('PV vs Temperature')
# PV vs Humidity
axes[1,1].scatter(total_df['Humidity [%]'], total_df['PV Power Generation [kW]'], alpha=0.1, s=1)
axes[1,1].set_xlabel('Humidity [%]')
axes[1,1].set_ylabel('PV [kW]')
axes[1,1].set_title('PV vs Humidity')
plt.suptitle('Photovoltaic Generation Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig11_pv_analysis.png'), bbox_inches='tight')
plt.close()
print("  fig11_pv_analysis.png")

# --- Figure 12: GHG emissions ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
monthly_ghg = pd.DataFrame({bid: cleaned_dfs[bid]['Greenhouse Gas Emission [Ton]'].resample('M').sum() 
                            for bid in BUILDING_IDS})
monthly_ghg.plot.area(ax=axes[0], alpha=0.7, cmap='tab10')
axes[0].set_title('Monthly GHG Emissions by Building')
axes[0].set_ylabel('GHG [Ton]')
# GHG vs Electricity
axes[1].scatter(total_df['Electricity [kW]'], total_df['Greenhouse Gas Emission [Ton]'], alpha=0.1, s=1)
axes[1].set_xlabel('Electricity [kW]')
axes[1].set_ylabel('GHG [Ton]')
axes[1].set_title('GHG vs Electricity')
# Diurnal GHG
for bid in BUILDING_IDS:
    hourly_ghg = cleaned_dfs[bid].groupby(cleaned_dfs[bid].index.hour)['Greenhouse Gas Emission [Ton]'].mean()
    axes[2].plot(hourly_ghg.index, hourly_ghg.values, alpha=0.6)
axes[2].set_title('Diurnal GHG Profile')
axes[2].set_xlabel('Hour')
plt.suptitle('Greenhouse Gas Emissions Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig12_ghg_analysis.png'), bbox_inches='tight')
plt.close()
print("  fig12_ghg_analysis.png")

print("\n" + "=" * 60)
print("ALL PHASES COMPLETE")
print("=" * 60)
