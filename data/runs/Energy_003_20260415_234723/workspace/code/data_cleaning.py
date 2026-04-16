import pandas as pd
import numpy as np
import os

data_dir = 'data/HEEW_Mini-Dataset'
out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

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

# 1. Missing value check
missing_report = {}
for k, df in bn_data.items():
    missing_report[k] = df.isnull().sum().sum()
missing_report['CN01'] = cn01.isnull().sum().sum()
missing_report['Total'] = total.isnull().sum().sum()
missing_report['Weather'] = weather.isnull().sum().sum()

print("Missing values report:")
print(missing_report)

# 2. Outlier detection (simple Z-score based approach for Total data)
from scipy import stats
z_scores = np.abs(stats.zscore(total))
outliers = (z_scores > 3).sum(axis=0)
print("\nOutliers detected (Z-score > 3) in Total dataset:")
print(outliers)

# 3. Save cleaned merged dataset
merged_data = pd.merge(total, weather, left_index=True, right_index=True)
merged_data.to_csv(os.path.join(out_dir, 'Cleaned_Total_Merged.csv'))

print("\nData cleaning and merging completed. Saved to outputs/Cleaned_Total_Merged.csv")
