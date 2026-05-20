"""
HEEW Mini-Dataset Comprehensive Analysis Script
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMAGE_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Global style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

def load_building_energy(bid):
    df = pd.read_csv(os.path.join(DATA_DIR, f'{bid}_energy.csv'))
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def load_weather():
    df = pd.read_csv(os.path.join(DATA_DIR, 'Total_weather.csv'))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
building_ids = [f'BN{i:03d}' for i in range(1, 11)] + ['CN01', 'Total']
energy_dfs = {bid: load_building_energy(bid) for bid in building_ids}
weather_df = load_weather()

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
weather_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']

# ------------------------------------------------------------------
# 2. Data cleaning diagnostics and algorithm
# ------------------------------------------------------------------
cleaning_report = {}

def clean_dataframe(df, bid):
    """Detect missing, negative, and outliers; interpolate outliers."""
    report = {'building': bid, 'missing': {}, 'negative': {}, 'outliers': {}}
    df_clean = df.copy()
    for col in energy_cols + weather_cols:
        if col not in df_clean.columns:
            continue
        report['missing'][col] = int(df_clean[col].isna().sum())
        report['negative'][col] = int((df_clean[col] < 0).sum())
        # Outlier detection using IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        report['outliers'][col] = int(outlier_mask.sum())
        # Interpolate outliers
        df_clean.loc[outlier_mask, col] = np.nan
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
    return df_clean, report

cleaned_dfs = {}
for bid in building_ids:
    cleaned, rep = clean_dataframe(energy_dfs[bid], bid)
    cleaned_dfs[bid] = cleaned
    cleaning_report[bid] = rep

# Save cleaning report
with open(os.path.join(OUTPUT_DIR, 'cleaning_report.json'), 'w') as f:
    json.dump(cleaning_report, f, indent=2)

# Save cleaned CSVs for buildings
for bid in building_ids:
    cleaned_dfs[bid].to_csv(os.path.join(OUTPUT_DIR, f'{bid}_cleaned.csv'), index=False)

# ------------------------------------------------------------------
# 3. Hierarchical aggregation consistency
# ------------------------------------------------------------------
building_sum = cleaned_dfs['BN001'][['datetime'] + energy_cols].copy()
for col in energy_cols:
    building_sum[col] = sum(cleaned_dfs[bid][col] for bid in building_ids[:10])

cn01 = cleaned_dfs['CN01']
total = cleaned_dfs['Total']

hier_errors = {}
for col in energy_cols:
    err_cn = building_sum[col] - cn01[col]
    err_total = cn01[col] - total[col]
    hier_errors[col] = {
        'sum_vs_CN01_max_abs_error': float(err_cn.abs().max()),
        'sum_vs_CN01_rmse': float(np.sqrt((err_cn**2).mean())),
        'CN01_vs_Total_max_abs_error': float(err_total.abs().max()),
        'CN01_vs_Total_rmse': float(np.sqrt((err_total**2).mean()))
    }

with open(os.path.join(OUTPUT_DIR, 'hierarchical_errors.json'), 'w') as f:
    json.dump(hier_errors, f, indent=2)

# Plot hierarchical consistency for Electricity
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(building_sum['datetime'], building_sum['Electricity [kW]'], label='Sum(BN001-BN010)', alpha=0.7)
ax.plot(cn01['datetime'], cn01['Electricity [kW]'], label='CN01', alpha=0.7, linestyle='--')
ax.plot(total['datetime'], total['Electricity [kW]'], label='Total', alpha=0.7, linestyle=':')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity [kW]')
ax.set_title('Hierarchical Aggregation Consistency — Electricity')
ax.legend()
ax.set_xlim(building_sum['datetime'].iloc[0], building_sum['datetime'].iloc[0] + pd.Timedelta(days=14))
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_hierarchical_electricity.png'))
plt.close(fig)

# ------------------------------------------------------------------
# 4. Correlation analysis
# ------------------------------------------------------------------
# Merge Total energy with weather
total_weather = pd.merge(cleaned_dfs['Total'], weather_df, on='datetime', how='left')
corr_vars = energy_cols + weather_cols
corr_pearson = total_weather[corr_vars].corr(method='pearson')
corr_spearman = total_weather[corr_vars].corr(method='spearman')

corr_pearson.to_csv(os.path.join(OUTPUT_DIR, 'correlation_pearson.csv'))
corr_spearman.to_csv(os.path.join(OUTPUT_DIR, 'correlation_spearman.csv'))

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Pearson Correlation Matrix — Total Energy & Weather')
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_correlation_heatmap.png'))
plt.close(fig)

# Scatter of electricity vs temperature
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(total_weather['Temperature [°F]'], total_weather['Electricity [kW]'], alpha=0.3, s=8)
ax.set_xlabel('Temperature [°F]')
ax.set_ylabel('Electricity [kW]')
ax.set_title('Total Electricity vs Temperature')
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_scatter_elec_temp.png'))
plt.close(fig)

# ------------------------------------------------------------------
# 5. Seasonal and diurnal profiles
# ------------------------------------------------------------------
total_weather['month'] = total_weather['datetime'].dt.month
total_weather['hour'] = total_weather['datetime'].dt.hour

# Monthly averages
monthly = total_weather.groupby('month')[energy_cols].mean().reset_index()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    ax.plot(monthly['month'], monthly[col], marker='o')
    ax.set_title(col)
    ax.set_xlabel('Month')
    ax.set_xticks(range(1, 13))
fig.suptitle('Monthly Average Profiles (Total)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(IMAGE_DIR, 'fig_monthly_profiles.png'))
plt.close(fig)

# Diurnal profiles for summer (Jun-Aug) vs winter (Dec-Feb)
summer = total_weather[total_weather['month'].isin([6,7,8])]
winter = total_weather[total_weather['month'].isin([12,1,2])]
diurnal_summer = summer.groupby('hour')[energy_cols].mean()
diurnal_winter = winter.groupby('hour')[energy_cols].mean()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    ax.plot(diurnal_summer.index, diurnal_summer[col], label='Summer', marker='o')
    ax.plot(diurnal_winter.index, diurnal_winter[col], label='Winter', marker='s')
    ax.set_title(col)
    ax.set_xlabel('Hour of Day')
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
fig.suptitle('Diurnal Profiles — Summer vs Winter (Total)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(IMAGE_DIR, 'fig_diurnal_profiles.png'))
plt.close(fig)

# ------------------------------------------------------------------
# 6. Hierarchical clustering of buildings
# ------------------------------------------------------------------
# Use daily mean electricity profiles for each building
profiles = []
labels = []
for bid in building_ids[:10]:
    df = cleaned_dfs[bid].copy()
    df['hour'] = df['datetime'].dt.hour
    # Use Electricity, Heat, Cooling as features per hour
    piv = df.groupby('hour')[['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]']].mean()
    # flatten to 24*3 features
    vec = piv.values.flatten()
    profiles.append(vec)
    labels.append(bid)

profiles = np.array(profiles)
# Standardize
profiles_scaled = StandardScaler().fit_transform(profiles)
# Distance and linkage
Z = linkage(profiles_scaled, method='ward')

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, labels=labels, ax=ax)
ax.set_title('Hierarchical Clustering of Buildings (Daily Mean Profiles)')
ax.set_ylabel('Ward Distance')
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_clustering_dendrogram.png'))
plt.close(fig)

# Save cluster linkage
pd.DataFrame(Z, columns=['c1','c2','distance','sample_count']).to_csv(
    os.path.join(OUTPUT_DIR, 'cluster_linkage.csv'), index=False)

# ------------------------------------------------------------------
# 7. Baseline load forecasting
# ------------------------------------------------------------------
# Target: Total Electricity
forecast_df = total_weather.copy()
forecast_df['month'] = forecast_df['datetime'].dt.month
forecast_df['dayofweek'] = forecast_df['datetime'].dt.dayofweek
forecast_df['hour'] = forecast_df['datetime'].dt.hour

feature_cols = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                'Wind Speed [mph]', 'Pressure [in]', 'Precipitation [in]',
                'month', 'dayofweek', 'hour']
# Add lag-1 and lag-24 of target
forecast_df['elec_lag1'] = forecast_df['Electricity [kW]'].shift(1)
forecast_df['elec_lag24'] = forecast_df['Electricity [kW]'].shift(24)
feature_cols += ['elec_lag1', 'elec_lag24']

forecast_df = forecast_df.dropna(subset=feature_cols + ['Electricity [kW]'])

# Temporal split: first 80% train, last 20% test
split_idx = int(len(forecast_df) * 0.8)
train_df = forecast_df.iloc[:split_idx]
test_df = forecast_df.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df['Electricity [kW]']
X_test = test_df[feature_cols]
y_test = test_df['Electricity [kW]']

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

forecast_metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
with open(os.path.join(OUTPUT_DIR, 'forecast_metrics.json'), 'w') as f:
    json.dump(forecast_metrics, f, indent=2)

# Feature importance
fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
fi.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# Plot actual vs predicted (first 7 days of test)
fig, ax = plt.subplots(figsize=(12, 5))
test_df = test_df.copy()
test_df['pred'] = y_pred
sample = test_df.iloc[:168]  # 7 days
ax.plot(sample['datetime'], sample['Electricity [kW]'], label='Actual')
ax.plot(sample['datetime'], sample['pred'], label='Predicted', alpha=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('Electricity [kW]')
ax.set_title(f'Total Electricity Forecast (Test Set, 7 days) — RMSE={rmse:.2f}, MAPE={mape:.2f}%')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_forecast_total_electricity.png'))
plt.close(fig)

# Feature importance bar plot
fig, ax = plt.subplots(figsize=(8, 5))
fi_sorted = fi.sort_values('importance', ascending=True)
ax.barh(fi_sorted['feature'], fi_sorted['importance'])
ax.set_title('Feature Importance — Random Forest Forecast')
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_feature_importance.png'))
plt.close(fig)

# ------------------------------------------------------------------
# 8. Imputation experiment
# ------------------------------------------------------------------
# Use Total electricity series
impute_series = total_weather[['datetime', 'Electricity [kW]']].copy()
orig = impute_series['Electricity [kW]'].values.copy()
np.random.seed(42)
mask = np.random.rand(len(orig)) < 0.05  # 5% missing
impute_series.loc[mask, 'Electricity [kW]'] = np.nan

# Linear interpolation
linear = impute_series['Electricity [kW]'].interpolate(method='linear', limit_direction='both').values
# Cubic spline interpolation
spline = impute_series['Electricity [kW]'].interpolate(method='spline', order=3, limit_direction='both').values

rmse_linear = np.sqrt(mean_squared_error(orig[mask], linear[mask]))
rmse_spline = np.sqrt(mean_squared_error(orig[mask], spline[mask]))
mae_linear = mean_absolute_error(orig[mask], linear[mask])
mae_spline = mean_absolute_error(orig[mask], spline[mask])

impute_metrics = {
    'missing_rate': 0.05,
    'linear_interp': {'RMSE': rmse_linear, 'MAE': mae_linear},
    'spline_interp': {'RMSE': rmse_spline, 'MAE': mae_spline}
}
with open(os.path.join(OUTPUT_DIR, 'imputation_metrics.json'), 'w') as f:
    json.dump(impute_metrics, f, indent=2)

# Plot a zoomed segment with masked points and interpolants
segment = slice(2000, 2200)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(impute_series['datetime'].iloc[segment], orig[segment], label='Original', alpha=0.6)
ax.plot(impute_series['datetime'].iloc[segment], linear[segment], label='Linear interp', linestyle='--')
ax.plot(impute_series['datetime'].iloc[segment], spline[segment], label='Spline interp', linestyle=':')
mask_segment = mask[segment]
ax.scatter(impute_series['datetime'].iloc[segment][mask_segment], orig[segment][mask_segment],
           color='red', label='Masked (true)', zorder=5, s=20)
ax.set_title('Imputation Example — Total Electricity (zoomed)')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity [kW]')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'fig_imputation.png'))
plt.close(fig)

print('Analysis complete. Outputs saved to', OUTPUT_DIR, 'and images to', IMAGE_DIR)
