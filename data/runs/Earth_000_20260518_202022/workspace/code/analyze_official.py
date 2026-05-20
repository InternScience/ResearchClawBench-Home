"""
Analyze official GlaMBIE combined results (calendar years).
Compute trends, acceleration, cumulative changes, and save outputs.
"""
import pandas as pd
import numpy as np
import os

RESULT_DIR = 'data/glambie/results/calendar_years'
OUTPUT_DIR = 'outputs'

# Load all regional result files
regions = []
for fname in sorted(os.listdir(RESULT_DIR)):
    if fname.endswith('.csv') and fname != '0_global.csv':
        df = pd.read_csv(os.path.join(RESULT_DIR, fname))
        df['year'] = df['start_dates'].astype(int)
        df = df[(df['year'] >= 2000) & (df['year'] <= 2023)]
        regions.append(df)

regional = pd.concat(regions, ignore_index=True)

# Global
global_df = pd.read_csv(os.path.join(RESULT_DIR, '0_global.csv'))
global_df['year'] = global_df['start_dates'].astype(int)
global_df = global_df[(global_df['year'] >= 2000) & (global_df['year'] <= 2023)]

# Cumulative mass change
global_df = global_df.sort_values('year')
global_df['cumulative_gt'] = global_df['combined_gt'].cumsum()
global_df['cumulative_gt_error'] = np.sqrt((global_df['combined_gt_errors']**2).cumsum())
global_df['cumulative_mwe'] = global_df['combined_mwe'].cumsum()
global_df['cumulative_mwe_error'] = np.sqrt((global_df['combined_mwe_errors']**2).cumsum())

# Regional cumulative
regional = regional.sort_values(['region','year'])
regional['cumulative_gt'] = regional.groupby('region')['combined_gt'].cumsum()
regional['cumulative_mwe'] = regional.groupby('region')['combined_mwe'].cumsum()

# Trend analysis (linear regression on annual mass change)
from scipy import stats
slope_gt, intercept_gt, r_value_gt, p_value_gt, std_err_gt = stats.linregress(global_df['year'], global_df['combined_gt'])
slope_mwe, intercept_mwe, r_value_mwe, p_value_mwe, std_err_mwe = stats.linregress(global_df['year'], global_df['combined_mwe'])

# Acceleration: compare first half vs second half
first = global_df[global_df['year'] <= 2011]
second = global_df[global_df['year'] >= 2012]
rate1 = first['combined_gt'].mean()
rate2 = second['combined_gt'].mean()
acc = (rate2 - rate1) / abs(rate1) * 100  # percent increase

# Save summaries
summary = {
    'global_mean_rate_gt_per_year_2000_2023': float(global_df['combined_gt'].mean()),
    'global_mean_error_gt_per_year_2000_2023': float(global_df['combined_gt_errors'].mean()),
    'global_mean_rate_mwe_per_year_2000_2023': float(global_df['combined_mwe'].mean()),
    'global_mean_error_mwe_per_year_2000_2023': float(global_df['combined_mwe_errors'].mean()),
    'cumulative_loss_gt_2000_2023': float(global_df['cumulative_gt'].iloc[-1]),
    'cumulative_loss_error_gt': float(global_df['cumulative_gt_error'].iloc[-1]),
    'trend_gt_per_year': float(slope_gt),
    'trend_gt_pvalue': float(p_value_gt),
    'trend_mwe_per_year': float(slope_mwe),
    'trend_mwe_pvalue': float(p_value_mwe),
    'rate_first_half_gt': float(rate1),
    'rate_second_half_gt': float(rate2),
    'acceleration_percent': float(acc),
}

pd.Series(summary).to_json(os.path.join(OUTPUT_DIR, 'summary.json'), indent=2)

# Save global and regional tables
global_df.to_csv(os.path.join(OUTPUT_DIR, 'official_global.csv'), index=False)
regional.to_csv(os.path.join(OUTPUT_DIR, 'official_regional.csv'), index=False)

print('Analysis complete.')
print('Global mean rate (2000-2023):', summary['global_mean_rate_gt_per_year_2000_2023'], 'Gt/yr')
print('Cumulative loss:', summary['cumulative_loss_gt_2000_2023'], 'Gt')
print('Acceleration:', summary['acceleration_percent'], '%')
