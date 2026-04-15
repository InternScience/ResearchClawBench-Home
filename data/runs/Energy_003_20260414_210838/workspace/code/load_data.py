import pandas as pd
import numpy as np
import os
from pathlib import Path

data_dir = Path('data/HEEW_Mini-Dataset')
energy_files = list(data_dir.glob('BN*.csv')) + [data_dir / 'CN01_energy.csv', data_dir / 'Total_energy.csv']
weather_file = data_dir / 'Total_weather.csv'

# Load weather
weather = pd.read_csv(weather_file)
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather.set_index('datetime', inplace=True)
print('Weather shape:', weather.shape)
print(weather.head())

# Load energy files
energy_dfs = {}
for f in energy_files:
    name = f.stem
    df = pd.read_csv(f)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df.set_index('datetime', inplace=True)
    df = df.drop(['year','month','day','hour'], axis=1)
    energy_dfs[name] = df
    print(f'{name} shape: {df.shape}')

# Summary
summary = {
    'weather': {'shape': weather.shape, 'missing_pct': weather.isnull().mean().to_dict()},
    'energy_files': {k: {'shape': v.shape, 'missing_pct': v.isnull().mean().to_dict()} for k,v in energy_dfs.items()}
}
print(summary)

# Hierarchy check: sum buildings
buildings = [df for name, df in energy_dfs.items() if name.startswith('BN')]
sum_buildings = pd.DataFrame({col: sum(df[col] for df in buildings) for col in buildings[0].columns}, index=buildings[0].index)
print('Sum buildings vs CN01:')
diff_cn01 = sum_buildings - energy_dfs['CN01_energy']
print('Mean abs diff CN01:', diff_cn01.abs().mean())

print('Sum buildings vs Total:')
diff_total = sum_buildings - energy_dfs['Total_energy']
print('Mean abs diff Total:', diff_total.abs().mean())

# Save
summary['hierarchy_check'] = {
    'mae_cn01': diff_cn01.abs().mean().to_dict(),
    'mae_total': diff_total.abs().mean().to_dict()
}
pd.to_pickle({'weather': weather, 'energy': energy_dfs, 'sum_buildings': sum_buildings}, 'outputs/raw_data.pkl')
import json
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)