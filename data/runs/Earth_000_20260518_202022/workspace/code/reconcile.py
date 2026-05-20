"""
Reconcile annualized input estimates into regional and global time series (m w.e. then Gt).
"""
import pandas as pd
import numpy as np
import json
import os

INPUT_FILE = 'outputs/annualized_inputs_mwe.csv'
AREA_FILE = 'outputs/region_areas.json'
OFFICIAL_DIR = 'data/glambie/results/calendar_years'
OUTPUT_REGIONAL = 'outputs/reconciled_regional.csv'
OUTPUT_GLOBAL = 'outputs/reconciled_global.csv'

with open(AREA_FILE) as f:
    area_dict = json.load(f)

df = pd.read_csv(INPUT_FILE)
df = df[(df['year'] >= 2000) & (df['year'] <= 2023)]

# Aggregate per author-year in quadrature
df['error_sq'] = df['error_mwe'] ** 2
author_year = df.groupby(['region', 'method', 'author', 'year']).agg(
    change_mwe=('change_mwe', 'sum'),
    error_sq=('error_sq', 'sum')
).reset_index()
author_year['error_mwe'] = np.sqrt(author_year['error_sq'])

# Floor error to avoid infinite weights
MIN_ERROR = 0.001
author_year['error_mwe'] = author_year['error_mwe'].clip(lower=MIN_ERROR)

def weighted_mean(g):
    w = 1.0 / (g['error_mwe'] ** 2)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0)
    if w.sum() == 0:
        return pd.Series({'change_mwe': g['change_mwe'].mean(), 'error_mwe': g['error_mwe'].mean()})
    change = (w * g['change_mwe']).sum() / w.sum()
    error = np.sqrt(1.0 / w.sum())
    return pd.Series({'change_mwe': change, 'error_mwe': error})

method_year = author_year.groupby(['region', 'method', 'year']).apply(weighted_mean, include_groups=False).reset_index()

region_year = method_year.groupby(['region', 'year']).apply(weighted_mean, include_groups=False).reset_index()

# Load official area time series per region
area_ts = {}
for fname in os.listdir(OFFICIAL_DIR):
    if fname.endswith('.csv') and fname != '0_global.csv':
        region_key = fname.replace('.csv', '')
        dfa = pd.read_csv(os.path.join(OFFICIAL_DIR, fname))
        dfa = dfa[(dfa['start_dates'] >= 2000) & (dfa['start_dates'] <= 2023)]
        area_ts[region_key] = dict(zip(dfa['start_dates'].astype(int), dfa['glacier_area']))

region_year['area'] = region_year.apply(lambda r: area_ts.get(r['region'], {}).get(r['year'], np.nan), axis=1)
region_year['change_gt'] = region_year['change_mwe'] * region_year['area'] * 0.000997
region_year['error_gt'] = region_year['error_mwe'] * region_year['area'] * 0.000997

region_year.to_csv(OUTPUT_REGIONAL, index=False)

# Global aggregation
global_year = region_year.groupby('year').agg(
    global_gt=('change_gt', 'sum'),
    global_error_gt=('error_gt', lambda x: np.sqrt((x**2).sum())),
    global_area=('area', 'sum')
).reset_index()

global_year['global_mwe'] = global_year['global_gt'] / (global_year['global_area'] * 0.000997)
global_year['global_mwe_error'] = global_year['global_error_gt'] / (global_year['global_area'] * 0.000997)

global_year.to_csv(OUTPUT_GLOBAL, index=False)
print('Saved', OUTPUT_REGIONAL, 'and', OUTPUT_GLOBAL)
print('Global mean rate (2000-2023):', global_year['global_gt'].mean(), '+/-', global_year['global_error_gt'].mean())
