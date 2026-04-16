import pandas as pd
import numpy as np
import os
import json

data_dir = 'data/HEEW_Mini-Dataset'

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

# 1. Consistency Check
sum_bn = pd.DataFrame()
for col in cn01.columns:
    sum_bn[col] = sum(df[col] for df in bn_data.values())

diff_cn01_sum = (cn01 - sum_bn).abs().sum().to_dict()
diff_total_cn01 = (total - cn01).abs().sum().to_dict()

# 2. Correlation Analysis
merged_data = pd.merge(total, weather, left_index=True, right_index=True)
corr_matrix = merged_data.corr()

# Extract specific correlations
corr_cooling_temp = corr_matrix.loc['Cooling Energy [Ton]', 'Temperature [°F]']
corr_heating_temp = corr_matrix.loc['Heat [mmBTU]', 'Temperature [°F]']
corr_pv_temp = corr_matrix.loc['PV Power Generation [kW]', 'Temperature [°F]']
corr_pv_humidity = corr_matrix.loc['PV Power Generation [kW]', 'Humidity [%]']

# Save results
results = {
    "consistency": {
        "diff_cn01_sum": diff_cn01_sum,
        "diff_total_cn01": diff_total_cn01
    },
    "correlation": {
        "cooling_temp": corr_cooling_temp,
        "heating_temp": corr_heating_temp,
        "pv_temp": corr_pv_temp,
        "pv_humidity": corr_pv_humidity
    }
}

with open('outputs/quantitative_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Quantitative results saved to outputs/quantitative_results.json")
