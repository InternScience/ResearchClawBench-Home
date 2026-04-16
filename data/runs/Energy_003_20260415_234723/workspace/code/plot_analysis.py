import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_dir = 'data/HEEW_Mini-Dataset'
out_dir = 'report/images'

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Load data
bn_files = [f for f in os.listdir(data_dir) if f.startswith('BN') and f.endswith('.csv')]
bn_data = {f.split('_')[0]: pd.read_csv(os.path.join(data_dir, f)) for f in bn_files}
cn01 = pd.read_csv(os.path.join(data_dir, 'CN01_energy.csv'))
total = pd.read_csv(os.path.join(data_dir, 'Total_energy.csv'))
weather = pd.read_csv(os.path.join(data_dir, 'Total_weather.csv'))

# Preprocess datetime
for k, df in bn_data.items():
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

cn01['datetime'] = pd.to_datetime(cn01[['year', 'month', 'day', 'hour']])
cn01.set_index('datetime', inplace=True)
cn01.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

total['datetime'] = pd.to_datetime(total[['year', 'month', 'day', 'hour']])
total.set_index('datetime', inplace=True)
total.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

weather['datetime'] = pd.to_datetime(weather['datetime'])
weather.set_index('datetime', inplace=True)

# 1. Total energy consumption over the year (Monthly)
monthly_total = total.resample('M').sum()
plt.figure(figsize=(12, 6))
plt.plot(monthly_total.index, monthly_total['Electricity [kW]'], marker='o', label='Electricity [kW]')
plt.plot(monthly_total.index, monthly_total['Cooling Energy [Ton]'], marker='s', label='Cooling Energy [Ton]')
plt.plot(monthly_total.index, monthly_total['Heat [mmBTU]'], marker='^', label='Heat [mmBTU]')
plt.plot(monthly_total.index, monthly_total['PV Power Generation [kW]'], marker='x', label='PV Power Generation [kW]')
plt.title('Monthly Aggregated Energy Profile (Total)')
plt.xlabel('Month')
plt.ylabel('Energy Consumption / Generation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'monthly_energy_profile.png'))
plt.close()

# 2. Daily profile example (Average daily profile over the year)
hourly_avg = total.groupby(total.index.hour).mean()
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg.index, hourly_avg['Electricity [kW]'], label='Electricity [kW]')
plt.plot(hourly_avg.index, hourly_avg['Cooling Energy [Ton]'], label='Cooling Energy [Ton]')
plt.plot(hourly_avg.index, hourly_avg['Heat [mmBTU]'], label='Heat [mmBTU]')
plt.plot(hourly_avg.index, hourly_avg['PV Power Generation [kW]'], label='PV Power Generation [kW]')
plt.title('Average Daily Energy Profile (Total)')
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy Consumption / Generation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'average_daily_profile.png'))
plt.close()

# 3. Correlation between Total Energy and Weather
merged_data = pd.merge(total, weather, left_index=True, right_index=True)
corr_matrix = merged_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix: Energy Consumption vs Weather Attributes')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'correlation_matrix.png'))
plt.close()

# 4. Consistency verification plot
sum_bn = pd.DataFrame()
for col in cn01.columns:
    sum_bn[col] = sum(df[col] for df in bn_data.values())

plt.figure(figsize=(10, 6))
plt.scatter(cn01['Electricity [kW]'], sum_bn['Electricity [kW]'], alpha=0.5, s=1)
plt.plot([cn01['Electricity [kW]'].min(), cn01['Electricity [kW]'].max()], 
         [cn01['Electricity [kW]'].min(), cn01['Electricity [kW]'].max()], 'r--')
plt.title('Consistency Check: Community vs Sum of Buildings (Electricity)')
plt.xlabel('Community Level (CN01) [kW]')
plt.ylabel('Sum of Building Levels (BN001-BN010) [kW]')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'consistency_check.png'))
plt.close()

print("Plots generated and saved to report/images/")
