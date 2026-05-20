#!/usr/bin/env python3
"""
HEEW Mini-Dataset: Time Series Analysis
Daily, monthly, and seasonal patterns in energy consumption and weather.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

DATA_DIR = 'data/HEEW_Mini-Dataset'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
total = pd.read_csv(os.path.join(DATA_DIR, 'Total_energy.csv'))
total['datetime'] = pd.to_datetime(total[['year', 'month', 'day', 'hour']])

weather = pd.read_csv(os.path.join(DATA_DIR, 'Total_weather.csv'))
weather['datetime'] = pd.to_datetime(weather['datetime'])

buildings = {}
for i in range(1, 11):
    fname = f'BN{i:03d}_energy.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    buildings[f'BN{i:03d}'] = df

energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']

# Add temporal features
total['month'] = total['datetime'].dt.month
total['day_of_year'] = total['datetime'].dt.dayofyear
total['hour'] = total['datetime'].dt.hour
total['weekday'] = total['datetime'].dt.weekday

weather['month'] = weather['datetime'].dt.month
weather['hour'] = weather['datetime'].dt.hour

# ============================================================
# 2. Figure 6: Monthly Aggregation
# ============================================================
print("Generating Figure 6: Monthly Energy Patterns")

monthly_total = total.groupby('month')[energy_cols].mean()
monthly_weather = weather.groupby('month')[['Temperature [°F]', 'Humidity [%]']].mean()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors = ['steelblue', 'coral', 'green', 'gold', 'purple']
for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    ax.bar(range(1, 13), monthly_total[col], color=colors[idx], edgecolor='white')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean Value')
    ax.grid(axis='y', alpha=0.3)

# 6th subplot: Temperature
ax = axes[5]
ax.plot(range(1, 13), monthly_weather['Temperature [°F]'], 'o-', color='red', linewidth=2, markersize=8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, fontsize=9)
ax.set_title('Temperature [°F]', fontweight='bold', fontsize=11)
ax.set_ylabel('Mean Temperature [°F]')
ax.grid(alpha=0.3)

plt.suptitle('Figure 6: Monthly Average Energy Consumption and Temperature (2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure6_monthly_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure6_monthly_patterns.png")

# ============================================================
# 3. Figure 7: Hourly Patterns by Season
# ============================================================
print("Generating Figure 7: Hourly Patterns by Season")

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

total['season'] = total['month'].apply(get_season)
weather['season'] = weather['month'].apply(get_season)

hourly_season = total.groupby(['season', 'hour'])[energy_cols].mean().reset_index()
hourly_weather_season = weather.groupby(['season', 'hour'])['Temperature [°F]'].mean().reset_index()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
season_colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}

for idx, col in enumerate(energy_cols):
    ax = axes[idx]
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        data = hourly_season[hourly_season['season'] == season]
        ax.plot(data['hour'], data[col], '-', color=season_colors[season],
                linewidth=2, label=season, alpha=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(col)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.3)

# 6th subplot: Temperature
ax = axes[5]
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    data = hourly_weather_season[hourly_weather_season['season'] == season]
    ax.plot(data['hour'], data['Temperature [°F]'], '-', color=season_colors[season],
            linewidth=2, label=season, alpha=0.8)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Temperature [°F]')
ax.set_title('Temperature [°F]', fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.set_xticks(range(0, 24, 3))
ax.grid(alpha=0.3)

plt.suptitle('Figure 7: Average Hourly Energy Profiles by Season (2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure7_hourly_seasonal.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure7_hourly_seasonal.png")

# ============================================================
# 4. Figure 8: Time Series for a Sample Week
# ============================================================
print("Generating Figure 8: Sample Week Time Series")

# Select a summer week (July 7-13) and a winter week (Jan 6-12)
summer_week = total[(total['datetime'] >= '2014-07-07') & (total['datetime'] < '2014-07-14')]
winter_week = total[(total['datetime'] >= '2014-01-06') & (total['datetime'] < '2014-01-13')]

summer_w = weather[(weather['datetime'] >= '2014-07-07') & (weather['datetime'] < '2014-07-14')]
winter_w = weather[(weather['datetime'] >= '2014-01-06') & (weather['datetime'] < '2014-01-13')]

fig, axes = plt.subplots(3, 2, figsize=(18, 14))

plot_vars = ['Electricity [kW]', 'Cooling Energy [Ton]', 'PV Power Generation [kW]']
plot_var_labels = ['Electricity [kW]', 'Cooling Energy [Ton]', 'PV Generation [kW]']

for idx, (var, label) in enumerate(zip(plot_vars, plot_var_labels)):
    # Summer
    ax = axes[idx, 0]
    ax.plot(summer_week['datetime'], summer_week[var], color='red', linewidth=1.5)
    ax.set_title(f'{label} - Summer Week (Jul 7-13)', fontweight='bold', fontsize=11)
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.grid(alpha=0.3)
    
    # Winter
    ax = axes[idx, 1]
    ax.plot(winter_week['datetime'], winter_week[var], color='blue', linewidth=1.5)
    ax.set_title(f'{label} - Winter Week (Jan 6-12)', fontweight='bold', fontsize=11)
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.grid(alpha=0.3)

plt.suptitle('Figure 8: Hourly Time Series for Sample Summer and Winter Weeks (2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure8_sample_weeks.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure8_sample_weeks.png")

# ============================================================
# 5. Figure 9: Daily Load Duration Curves
# ============================================================
print("Generating Figure 9: Load Duration Curves")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Load duration curve for Electricity
sorted_elec = np.sort(total['Electricity [kW]'].values)[::-1]
ax = axes[0]
ax.plot(np.arange(1, len(sorted_elec)+1)/len(sorted_elec)*100, sorted_elec, color='steelblue', linewidth=2)
ax.set_xlabel('Percentage of Time (%)')
ax.set_ylabel('Electricity [kW]')
ax.set_title('Electricity Load Duration Curve', fontweight='bold')
ax.grid(alpha=0.3)
ax.axhline(total['Electricity [kW]'].mean(), color='red', linestyle='--', label=f'Mean: {total["Electricity [kW]"].mean():.1f}')
ax.legend()

# Load duration curves for Cooling
sorted_cool = np.sort(total['Cooling Energy [Ton]'].values)[::-1]
ax = axes[1]
ax.plot(np.arange(1, len(sorted_cool)+1)/len(sorted_cool)*100, sorted_cool, color='coral', linewidth=2)
ax.set_xlabel('Percentage of Time (%)')
ax.set_ylabel('Cooling Energy [Ton]')
ax.set_title('Cooling Load Duration Curve', fontweight='bold')
ax.grid(alpha=0.3)
ax.axhline(total['Cooling Energy [Ton]'].mean(), color='red', linestyle='--', label=f'Mean: {total["Cooling Energy [Ton]"].mean():.1f}')
ax.legend()

plt.suptitle('Figure 9: Load Duration Curves at Total Level (2014)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure9_load_duration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure9_load_duration.png")

# ============================================================
# 6. Save aggregated time series data
# ============================================================
print("\nSaving aggregated time series data...")
monthly_total.to_csv(os.path.join(OUTPUT_DIR, 'monthly_aggregation.csv'))
hourly_season.to_csv(os.path.join(OUTPUT_DIR, 'hourly_seasonal_aggregation.csv'))

# Annual totals
annual_totals = {}
for col in energy_cols:
    annual_totals[col] = total[col].sum()
    
print("\n=== Annual Totals (Total Level) ===")
for col, val in annual_totals.items():
    print(f"  {col}: {val:.2f}")

annual_df = pd.DataFrame([annual_totals])
annual_df.to_csv(os.path.join(OUTPUT_DIR, 'annual_totals.csv'), index=False)

print("\nTime series analysis complete!")
