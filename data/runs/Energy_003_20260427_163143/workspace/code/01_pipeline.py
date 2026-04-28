"""
HEEW Mini-Dataset Analysis Pipeline
====================================
Stages:
  1. Load & schema validation
  2. Data cleaning algorithm (range + IQR + spike) and interpolation
  3. Hierarchical aggregation consistency
  4. Correlation analysis (energy & weather)
  5. Temporal patterns
  6. Building-level clustering
  7. Demonstrative ML use cases (forecasting, anomaly, imputation)
  8. Figures & summary tables
All figures saved as PNG under report/images/.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', context='notebook')

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_003_20260427_163143')
DATA = ROOT / 'data' / 'HEEW_Mini-Dataset'
OUT  = ROOT / 'outputs'
IMG  = ROOT / 'report' / 'images'
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

ENERGY_COLS = ['Electricity [kW]','Heat [mmBTU]','Cooling Energy [Ton]',
               'PV Power Generation [kW]','Greenhouse Gas Emission [Ton]']
WEATHER_COLS = ['Temperature [°F]','Dew Point [°F]','Humidity [%]',
                'Wind Speed [mph]','Wind Gust [mph]','Pressure [in]','Precipitation [in]']
BUILDINGS = [f'BN{str(i).zfill(3)}' for i in range(1,11)]

# ----------------------------------------------------------
# Stage 1: Load
# ----------------------------------------------------------
def load_energy(node):
    df = pd.read_csv(DATA / f'{node}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    return df.set_index('datetime')[ENERGY_COLS]

def load_weather():
    w = pd.read_csv(DATA / 'Total_weather.csv', parse_dates=['datetime'])
    return w.set_index('datetime')[WEATHER_COLS]

energy = {n: load_energy(n) for n in BUILDINGS + ['CN01','Total']}
weather = load_weather()
print('loaded files:', list(energy.keys()), 'weather rows', len(weather))

# Dataset summary
rows = []
for name, df in energy.items():
    rows.append({
        'Node': name, 'Type': 'Building' if name.startswith('BN') else
                 ('Community' if name=='CN01' else 'Total'),
        'Records': len(df),
        'Start': str(df.index.min()), 'End': str(df.index.max()),
        **{f'{c}_mean': df[c].mean() for c in ENERGY_COLS},
        **{f'{c}_min': df[c].min() for c in ENERGY_COLS},
        **{f'{c}_max': df[c].max() for c in ENERGY_COLS},
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'dataset_summary.csv', index=False)
print('dataset_summary saved')

# ----------------------------------------------------------
# Stage 2: Data Cleaning Algorithm
# Steps:
#   (a) Range check: clip to physical ranges (no negative for E/H/C/PV, GHG)
#   (b) IQR-based outlier flag (per variable per node, q1-3*IQR..q3+3*IQR)
#   (c) Z-score spike detection on first-difference (|dz|>5)
#   (d) Replace flagged values with NaN, then linear-interpolate (limit=6h)
#       and use seasonal/diurnal hourly mean fallback for longer gaps
# ----------------------------------------------------------
def clean_pipeline(df):
    cleaned = df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)

    for c in df.columns:
        x = cleaned[c].astype(float).copy()
        # (a) range
        if c == 'Greenhouse Gas Emission [Ton]':
            mask_neg = x < 0
        else:
            mask_neg = x < 0
        # (b) IQR
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        mask_iqr = (x < lo) | (x > hi)
        # (c) spike on diff z-score
        d = x.diff()
        if d.std() > 0:
            z = (d - d.mean()) / d.std()
            mask_spike = z.abs() > 6
        else:
            mask_spike = pd.Series(False, index=x.index)
        # PV power should be 0 at night (hour 0-5, 21-23) — flag values > small threshold
        mask_pv_night = pd.Series(False, index=x.index)
        if c == 'PV Power Generation [kW]':
            night = x.index.hour.isin(list(range(0,6))+list(range(21,24)))
            mask_pv_night = pd.Series(night, index=x.index) & (x > x.quantile(0.99))
        flag = mask_neg | mask_iqr | mask_spike | mask_pv_night
        flags[c] = flag
        x[flag] = np.nan
        # interpolate
        x = x.interpolate(method='time', limit=6, limit_direction='both')
        # for any remaining NaN: hourly seasonal mean
        if x.isna().any():
            hr_mean = x.groupby(x.index.hour).transform('mean')
            x = x.fillna(hr_mean)
        cleaned[c] = x
    return cleaned, flags

cleaned = {}
flag_summary = []
for name, df in energy.items():
    c, f = clean_pipeline(df)
    cleaned[name] = c
    flag_summary.append({
        'Node': name,
        **{f'{col}_flagged': int(f[col].sum()) for col in f.columns},
        'TotalFlagged': int(f.values.sum()),
        'FlagRate(%)': float(f.values.mean()*100),
    })
pd.DataFrame(flag_summary).to_csv(OUT / 'cleaning_report.csv', index=False)
print('cleaning_report saved')

# Save weather cleaning too
w_clean, w_flags = clean_pipeline(weather)
weather_clean_summary = pd.DataFrame({
    'Variable': WEATHER_COLS,
    'FlaggedCount': [int(w_flags[c].sum()) for c in WEATHER_COLS],
    'FlagRate(%)': [float(w_flags[c].mean()*100) for c in WEATHER_COLS],
})
weather_clean_summary.to_csv(OUT / 'weather_cleaning_report.csv', index=False)

# ----------------------------------------------------------
# Stage 3: Hierarchical aggregation consistency
# Sum of buildings should approximate community/total at hourly level.
# ----------------------------------------------------------
sum_bn = sum(cleaned[b] for b in BUILDINGS)
total = cleaned['Total']
cn01 = cleaned['CN01']

def agg_metrics(actual, expected, label):
    rows = []
    for c in ENERGY_COLS:
        a, e = actual[c].values, expected[c].values
        denom = np.maximum(np.abs(e).sum(), 1e-9)
        rel_err = np.abs(a - e).sum() / denom
        ratio = a.sum() / max(e.sum(), 1e-9)
        corr = np.corrcoef(a, e)[0,1] if np.std(a)>0 and np.std(e)>0 else np.nan
        rows.append({'Comparison': label, 'Variable': c,
                     'sum_actual': float(a.sum()), 'sum_expected': float(e.sum()),
                     'ratio_actual_over_expected': float(ratio),
                     'mean_abs_relative_err': float(rel_err),
                     'pearson_corr': float(corr)})
    return rows

rows = []
rows += agg_metrics(sum_bn, total, 'sum(BN001..BN010) vs Total')
rows += agg_metrics(sum_bn, cn01,  'sum(BN001..BN010) vs CN01')
rows += agg_metrics(cn01,   total, 'CN01 vs Total')
hier = pd.DataFrame(rows)
hier.to_csv(OUT / 'hierarchical_consistency.csv', index=False)
print('hierarchical_consistency saved')

# ----------------------------------------------------------
# Stage 4: Correlation analysis (Total + weather)
# ----------------------------------------------------------
joint = total.join(w_clean, how='inner')
corr = joint.corr(method='pearson')
corr.to_csv(OUT / 'correlation_energy_weather.csv')
print('correlation saved')

# ----------------------------------------------------------
# Stage 5: Temporal patterns
# ----------------------------------------------------------
diurnal = total.groupby(total.index.hour).mean()
diurnal.index.name = 'hour'
diurnal.to_csv(OUT / 'diurnal_total.csv')
monthly = total.groupby(total.index.month).mean()
monthly.index.name = 'month'
monthly.to_csv(OUT / 'monthly_total.csv')
weekly = total.groupby(total.index.dayofweek).mean()
weekly.index.name = 'dayofweek'
weekly.to_csv(OUT / 'weekly_total.csv')
print('temporal patterns saved')

# ----------------------------------------------------------
# Stage 6: Per-building clustering (hierarchical) on diurnal Electricity profile
# ----------------------------------------------------------
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
profiles = []
labels = []
for b in BUILDINGS:
    p = cleaned[b].groupby(cleaned[b].index.hour)['Electricity [kW]'].mean().values
    p = (p - p.mean()) / (p.std()+1e-9)
    profiles.append(p)
    labels.append(b)
profiles = np.array(profiles)
Z = linkage(profiles, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')
cluster_df = pd.DataFrame({'Building': labels, 'Cluster': clusters})
cluster_df.to_csv(OUT / 'building_clusters.csv', index=False)
print('clustering saved')

# ----------------------------------------------------------
# Stage 7: ML use cases
# ----------------------------------------------------------
# 7a. Day-ahead forecasting on Total electricity using simple ML
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def make_features(s, exog):
    df = pd.DataFrame({'y': s})
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f'lag_{lag}'] = s.shift(lag)
    df['hour'] = df.index.hour
    df['dow']  = df.index.dayofweek
    df['month']= df.index.month
    if exog is not None:
        df = df.join(exog)
    return df.dropna()

target = cleaned['Total']['Electricity [kW]']
exog = w_clean[['Temperature [°F]','Humidity [%]','Wind Speed [mph]']]
feat = make_features(target, exog)
split = int(len(feat)*0.8)
train, test = feat.iloc[:split], feat.iloc[split:]
Xtr, ytr = train.drop(columns='y'), train['y']
Xte, yte = test.drop(columns='y'),  test['y']

results = []
preds = {}
for name, mdl in [('Persistence(t-24)', None),
                  ('Ridge', Ridge(alpha=1.0)),
                  ('RandomForest', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0))]:
    if mdl is None:
        yhat = Xte['lag_24'].values
    else:
        mdl.fit(Xtr, ytr)
        yhat = mdl.predict(Xte)
    preds[name] = pd.Series(yhat, index=yte.index)
    mae = mean_absolute_error(yte, yhat)
    rmse = np.sqrt(mean_squared_error(yte, yhat))
    mape = np.mean(np.abs((yte.values - yhat) / np.maximum(np.abs(yte.values), 1e-6))) * 100
    r2 = r2_score(yte, yhat)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape, 'R2': r2})
fc_df = pd.DataFrame(results)
fc_df.to_csv(OUT / 'forecasting_results.csv', index=False)
print('forecasting_results saved\n', fc_df)

# 7b. Anomaly detection — synthetic injection on raw Total electricity, IsolationForest
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(42)
raw_total = energy['Total']['Electricity [kW]'].copy()
inject_idx = rng.choice(len(raw_total), size=50, replace=False)
inj_series = raw_total.copy()
inj_series.iloc[inject_idx] = inj_series.iloc[inject_idx] * rng.choice([0.2, 3.0], size=50)
truth = np.zeros(len(raw_total), dtype=bool); truth[inject_idx] = True

# Use detrended z + IsolationForest
hr_mean = inj_series.groupby(inj_series.index.hour).transform('mean')
hr_std  = inj_series.groupby(inj_series.index.hour).transform('std')
z = (inj_series - hr_mean) / (hr_std + 1e-9)
iso = IsolationForest(contamination=0.01, random_state=0)
iso_pred = iso.fit_predict(z.values.reshape(-1,1)) == -1

# evaluate
def metrics(truth, pred):
    tp = int(((truth) & (pred)).sum())
    fp = int(((~truth) & (pred)).sum())
    fn = int(((truth) & (~pred)).sum())
    tn = int(((~truth) & (~pred)).sum())
    prec = tp / max(tp+fp,1); rec = tp / max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'Precision':prec,'Recall':rec,'F1':f1}

ad_rows = []
ad_rows.append({'Method':'Z-score(|z|>3)', **metrics(truth, np.abs(z)>3)})
ad_rows.append({'Method':'Z-score(|z|>4)', **metrics(truth, np.abs(z)>4)})
ad_rows.append({'Method':'IsolationForest', **metrics(truth, iso_pred)})
pd.DataFrame(ad_rows).to_csv(OUT / 'anomaly_detection_summary.csv', index=False)
print('anomaly_detection saved')

# 7c. Imputation benchmark — randomly mask 5% of Total electricity, compare strategies
mask_rng = np.random.default_rng(0)
truth_e = cleaned['Total']['Electricity [kW]'].copy()
mask = mask_rng.random(len(truth_e)) < 0.05
masked = truth_e.copy(); masked[mask] = np.nan

results_imp = []
methods = {}
methods['Forward fill'] = masked.ffill().bfill()
methods['Linear interp'] = masked.interpolate(method='linear', limit_direction='both')
methods['Time interp']   = masked.interpolate(method='time', limit_direction='both')
hr_mean_imp = masked.groupby(masked.index.hour).transform('mean')
methods['Hourly mean']  = masked.fillna(hr_mean_imp)
# Ridge on lag/exog
feat_i = pd.DataFrame({'y': masked})
for lag in [1,24,168]:
    feat_i[f'lag_{lag}'] = truth_e.shift(lag)
feat_i = feat_i.join(w_clean[['Temperature [°F]','Humidity [%]']])
feat_i['hour']=feat_i.index.hour; feat_i['dow']=feat_i.index.dayofweek
train_i = feat_i.dropna()
mdl = Ridge(alpha=1.0).fit(train_i.drop(columns='y'), train_i['y'])
pred_full = pd.Series(mdl.predict(feat_i.drop(columns='y').fillna(method='ffill').fillna(method='bfill')), index=feat_i.index)
methods['Ridge regression'] = masked.copy()
methods['Ridge regression'][mask] = pred_full[mask]

for name, imp in methods.items():
    err = (imp - truth_e)[mask]
    mae = err.abs().mean(); rmse = np.sqrt((err**2).mean())
    mape = np.mean(np.abs(err.values / np.maximum(np.abs(truth_e[mask].values), 1e-6))) * 100
    results_imp.append({'Method': name, 'MAE': float(mae), 'RMSE': float(rmse), 'MAPE(%)': float(mape)})
pd.DataFrame(results_imp).to_csv(OUT / 'imputation_benchmark.csv', index=False)
print('imputation_benchmark saved')

# Save key numbers for report
key = {
    'n_buildings': len(BUILDINGS),
    'n_records_per_node': int(len(total)),
    'n_total_energy_records': int(sum(len(df) for df in energy.values())),
    'n_total_weather_records': int(len(weather)),
    'energy_variables': ENERGY_COLS,
    'weather_variables': WEATHER_COLS,
    'period': [str(total.index.min()), str(total.index.max())],
    'flag_rates': {r['Node']: r['FlagRate(%)'] for r in flag_summary},
    'forecasting': fc_df.to_dict(orient='records'),
    'anomaly_detection': ad_rows,
    'imputation': results_imp,
    'hierarchical_consistency_summary': hier.groupby('Comparison').agg(
        {'mean_abs_relative_err':'mean','ratio_actual_over_expected':'mean'}
    ).reset_index().to_dict(orient='records'),
}
with open(OUT / 'key_metrics.json','w') as fh:
    json.dump(key, fh, indent=2, default=str)
print('key_metrics saved')

# Save cleaned snapshots for reproducibility
cleaned['Total'].to_csv(OUT / 'Total_cleaned.csv')
w_clean.to_csv(OUT / 'Total_weather_cleaned.csv')
print('cleaned snapshots saved')

print('PIPELINE DONE')
