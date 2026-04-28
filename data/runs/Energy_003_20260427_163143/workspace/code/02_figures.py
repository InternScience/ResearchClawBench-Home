"""
Generate all PNG figures for HEEW Mini-Dataset report.
Reads cleaned outputs from outputs/, plus reloads raw data when needed.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_003_20260427_163143')
DATA = ROOT / 'data' / 'HEEW_Mini-Dataset'
OUT  = ROOT / 'outputs'
IMG  = ROOT / 'report' / 'images'
IMG.mkdir(parents=True, exist_ok=True)

ENERGY_COLS = ['Electricity [kW]','Heat [mmBTU]','Cooling Energy [Ton]',
               'PV Power Generation [kW]','Greenhouse Gas Emission [Ton]']
WEATHER_COLS = ['Temperature [°F]','Dew Point [°F]','Humidity [%]',
                'Wind Speed [mph]','Wind Gust [mph]','Pressure [in]','Precipitation [in]']
BUILDINGS = [f'BN{str(i).zfill(3)}' for i in range(1,11)]

def load_energy(node):
    df = pd.read_csv(DATA / f'{node}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    return df.set_index('datetime')[ENERGY_COLS]

def load_weather():
    return pd.read_csv(DATA / 'Total_weather.csv', parse_dates=['datetime']).set_index('datetime')[WEATHER_COLS]

energy_raw = {n: load_energy(n) for n in BUILDINGS+['CN01','Total']}
weather_raw = load_weather()

total_clean = pd.read_csv(OUT / 'Total_cleaned.csv', index_col=0, parse_dates=True)
weather_clean = pd.read_csv(OUT / 'Total_weather_cleaned.csv', index_col=0, parse_dates=True)

# ============================================================
# Fig 1: Dataset overview — bar of mean values per node
# ============================================================
summary = pd.read_csv(OUT / 'dataset_summary.csv')
summary['Order'] = summary['Node'].apply(lambda n: BUILDINGS.index(n) if n in BUILDINGS else (10 if n=='CN01' else 11))
summary = summary.sort_values('Order')
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, c in enumerate(ENERGY_COLS):
    ax = axes.flat[i]
    means = summary[f'{c}_mean'].values
    colors = ['#2b6cb0']*10 + ['#dd6b20','#c53030']
    bars = ax.bar(summary['Node'], means, color=colors[:len(summary)])
    ax.set_title(c, fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log' if means.max()/max(means.min()+1e-9,1e-9) > 100 else 'linear')
axes.flat[-1].axis('off')
fig.suptitle('Figure 1. Mean hourly energy variables per hierarchical node (HEEW Mini, 2014)', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig01_dataset_overview.png')
plt.close()

# ============================================================
# Fig 2: Total energy time series — daily aggregation
# ============================================================
daily = total_clean.resample('D').sum()
fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
colors = ['#1f77b4','#d62728','#17becf','#ff7f0e','#2ca02c']
for ax, c, col in zip(axes, ENERGY_COLS, colors):
    ax.plot(daily.index, daily[c], color=col, lw=1.0)
    ax.set_ylabel(c, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_title('Figure 2. Daily aggregated time series — Total node, 2014')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(IMG / 'fig02_time_series_total.png')
plt.close()

# ============================================================
# Fig 3: Weather overview
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(14, 11))
for ax, c in zip(axes.flat, WEATHER_COLS):
    ax.plot(weather_clean.index, weather_clean[c], lw=0.5, color='#444')
    ax.set_title(c, fontsize=10); ax.grid(alpha=0.3)
axes.flat[-1].axis('off')
fig.suptitle('Figure 3. Weather variables — Tempe, AZ, 2014', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig03_weather_overview.png')
plt.close()

# ============================================================
# Fig 4: Cleaning before/after — synthetic spike injection illustration
# Uses BN001 Electricity, injects spikes, runs clean to show recovery
# ============================================================
from scipy.stats import zscore
np.random.seed(7)
ex = energy_raw['BN001']['Electricity [kW]'].copy()
ex_inj = ex.copy()
spk = np.random.choice(len(ex_inj), 30, replace=False)
factors = np.random.choice([0.0, 5.0], size=30)
ex_inj.iloc[spk] = ex_inj.iloc[spk] * factors

# replicate cleaning per single var
def clean_var(x):
    x = x.copy().astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3*iqr, q3 + 3*iqr
    flag = (x < lo) | (x > hi) | (x < 0)
    d = x.diff()
    if d.std() > 0:
        z = (d - d.mean())/d.std(); flag = flag | (z.abs() > 6)
    x[flag] = np.nan
    x = x.interpolate('time', limit=6, limit_direction='both')
    return x, flag

ex_clean, flag = clean_var(ex_inj)
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
window = (ex.index >= '2014-06-15') & (ex.index <= '2014-06-22')
axes[0].plot(ex.index[window], ex_inj[window], color='#e63946', lw=0.8, label='With injected anomalies')
axes[0].plot(ex.index[window], ex[window], color='#1d3557', lw=0.8, alpha=0.7, label='Original')
axes[0].scatter(ex.index[window & flag.values], ex_inj[window & flag.values], color='black', s=20, zorder=5, label='Flagged points')
axes[0].set_title('Figure 4a. BN001 electricity — anomaly injection and detection (1 week)')
axes[0].legend(); axes[0].set_ylabel('kW')
axes[1].plot(ex.index[window], ex[window], color='#1d3557', lw=0.8, label='Original (truth)')
axes[1].plot(ex.index[window], ex_clean[window], color='#2a9d8f', lw=0.8, label='After cleaning')
axes[1].set_title('Figure 4b. After IQR + spike + interpolation cleaning')
axes[1].legend(); axes[1].set_ylabel('kW'); axes[1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(IMG / 'fig04_cleaning_before_after.png')
plt.close()

# ============================================================
# Fig 5: Hierarchical consistency — sum BNs vs CN01/Total
# ============================================================
hier = pd.read_csv(OUT / 'hierarchical_consistency.csv')
sum_bn = sum(pd.read_csv(OUT / 'Total_cleaned.csv', index_col=0, parse_dates=True) for _ in [1])  # placeholder
# Recompute sum_bn directly from cleaned per-building (re-clean quickly)
def reclean(df):
    cleaned = df.copy()
    for c in df.columns:
        x = cleaned[c].astype(float).copy()
        q1, q3 = x.quantile(0.25), x.quantile(0.75); iqr = q3-q1
        lo, hi = q1-3*iqr, q3+3*iqr
        flag = (x<lo)|(x>hi)|(x<0)
        d = x.diff()
        if d.std()>0: flag = flag | (((d - d.mean())/d.std()).abs() > 6)
        x[flag]=np.nan
        x = x.interpolate('time', limit=6, limit_direction='both')
        x = x.fillna(x.groupby(x.index.hour).transform('mean'))
        cleaned[c]=x
    return cleaned
cleaned_bn = {b: reclean(energy_raw[b]) for b in BUILDINGS}
sum_bn = sum(cleaned_bn[b] for b in BUILDINGS)
total_c = total_clean

# Panel A: scatter sum_BN vs Total for each variable (downsample daily)
fig = plt.figure(figsize=(15, 9))
for i, c in enumerate(ENERGY_COLS):
    ax = fig.add_subplot(2, 3, i+1)
    a = sum_bn[c].resample('D').sum()
    e = total_c[c].resample('D').sum()
    ax.scatter(e, a, s=6, alpha=0.4, color='#3a86ff')
    lim = max(e.max(), a.max())*1.05
    ax.plot([0,lim],[0,lim], 'k--', lw=1)
    ax.set_xlabel(f'Total (daily sum)'); ax.set_ylabel(f'Σ BN001..BN010 (daily sum)')
    ax.set_title(c, fontsize=10)
# Panel: bar of relative err
ax = fig.add_subplot(2, 3, 6)
sub = hier[hier['Comparison']=='sum(BN001..BN010) vs Total']
ax.bar(sub['Variable'], sub['mean_abs_relative_err']*100, color='#fb5607')
ax.set_title('Mean absolute relative error vs Total (%)')
ax.tick_params(axis='x', rotation=30)
fig.suptitle('Figure 5. Hierarchical aggregation consistency: Σ buildings vs Total', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig05_hierarchical_consistency.png')
plt.close()

# ============================================================
# Fig 6: Correlation heatmap (energy + weather) on Total
# ============================================================
corr = pd.read_csv(OUT / 'correlation_energy_weather.csv', index_col=0)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label':'Pearson r'})
ax.set_title('Figure 6. Pearson correlation across Total energy and weather variables')
plt.tight_layout()
plt.savefig(IMG / 'fig06_correlation_heatmap.png')
plt.close()

# ============================================================
# Fig 7: Diurnal & monthly patterns
# ============================================================
diurnal = total_c.groupby(total_c.index.hour).mean()
monthly = total_c.groupby(total_c.index.month).mean()
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for j, c in enumerate(ENERGY_COLS):
    axes[0, j].plot(diurnal.index, diurnal[c], marker='o', color='#264653')
    axes[0, j].set_title(f'Diurnal: {c}', fontsize=9)
    axes[0, j].set_xlabel('Hour'); axes[0, j].grid(alpha=0.3)
    axes[1, j].plot(monthly.index, monthly[c], marker='s', color='#e76f51')
    axes[1, j].set_title(f'Monthly: {c}', fontsize=9)
    axes[1, j].set_xlabel('Month'); axes[1, j].grid(alpha=0.3)
fig.suptitle('Figure 7. Diurnal (top) and monthly (bottom) average profiles — Total node, 2014', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig07_diurnal_seasonal.png')
plt.close()

# ============================================================
# Fig 8: Per-building hierarchical clustering on diurnal electricity profile
# ============================================================
profiles = []
for b in BUILDINGS:
    p = cleaned_bn[b].groupby(cleaned_bn[b].index.hour)['Electricity [kW]'].mean().values
    p = (p - p.mean()) / (p.std()+1e-9)
    profiles.append(p)
profiles = np.array(profiles)
Z = linkage(profiles, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 2, 1)
dendrogram(Z, labels=BUILDINGS, leaf_rotation=0, ax=ax1, color_threshold=Z[-2,2])
ax1.set_title('8a. Ward dendrogram on standardized diurnal electricity profile')
ax1.set_ylabel('Distance')

ax2 = fig.add_subplot(1, 2, 2)
hours = np.arange(24)
palette = sns.color_palette('Set1', n_colors=int(clusters.max()))
for prof, lbl, cl in zip(profiles, BUILDINGS, clusters):
    ax2.plot(hours, prof, color=palette[cl-1], alpha=0.8, lw=1.5, label=f'{lbl} (C{cl})')
ax2.set_title('8b. Standardized diurnal profiles (color=cluster)')
ax2.set_xlabel('Hour'); ax2.set_ylabel('z-score')
ax2.legend(ncol=2, fontsize=8, loc='upper right')
fig.suptitle('Figure 8. Hierarchical clustering of buildings by electricity diurnal profile', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig08_building_clustering.png')
plt.close()

# ============================================================
# Fig 9: Forecasting performance & visualization
# ============================================================
fc_df = pd.read_csv(OUT / 'forecasting_results.csv')

# Recompute one-week test predictions to plot
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
def make_features(s, exog):
    df = pd.DataFrame({'y': s})
    for lag in [1,2,3,24,48,168]: df[f'lag_{lag}'] = s.shift(lag)
    df['hour']=df.index.hour; df['dow']=df.index.dayofweek; df['month']=df.index.month
    df = df.join(exog)
    return df.dropna()
target = total_c['Electricity [kW]']
exog = weather_clean[['Temperature [°F]','Humidity [%]','Wind Speed [mph]']]
feat = make_features(target, exog)
split = int(len(feat)*0.8)
train, test = feat.iloc[:split], feat.iloc[split:]
Xtr, ytr = train.drop(columns='y'), train['y']
Xte, yte = test.drop(columns='y'),  test['y']
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(Xtr, ytr)
ri = Ridge(alpha=1.0).fit(Xtr, ytr)
yhat_rf = pd.Series(rf.predict(Xte), index=yte.index)
yhat_ri = pd.Series(ri.predict(Xte), index=yte.index)
yhat_pe = Xte['lag_24']

window = (yte.index >= yte.index[0]) & (yte.index <= yte.index[0] + pd.Timedelta(days=10))
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
ax = axes[0]
ax.plot(yte.index[window], yte[window], 'k', lw=1.6, label='Observed')
ax.plot(yhat_pe.index[window], yhat_pe[window], '--', lw=1, label='Persistence(t-24)')
ax.plot(yhat_ri.index[window], yhat_ri[window], lw=1, label='Ridge')
ax.plot(yhat_rf.index[window], yhat_rf[window], lw=1, label='Random Forest')
ax.set_ylabel('Electricity [kW]'); ax.legend(loc='upper right')
ax.set_title('9a. 10-day test-window forecast for Total electricity')
ax = axes[1]
metrics_to_plot = ['MAE','RMSE','MAPE(%)','R2']
x = np.arange(len(fc_df))
width = 0.2
for i, m in enumerate(metrics_to_plot):
    vals = fc_df[m].values
    ax.bar(x + i*width, vals, width, label=m)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(fc_df['Model'])
ax.set_yscale('symlog')
ax.legend(); ax.set_title('9b. Forecasting metrics (Persistence vs Ridge vs Random Forest)')
fig.suptitle('Figure 9. Day-ahead electricity load forecasting — Total node', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig09_forecasting.png')
plt.close()

# ============================================================
# Fig 10: Anomaly detection visualization
# ============================================================
ad_df = pd.read_csv(OUT / 'anomaly_detection_summary.csv')
rng = np.random.default_rng(42)
raw = energy_raw['Total']['Electricity [kW]'].copy()
inj = raw.copy()
idx = rng.choice(len(inj), 50, replace=False)
inj.iloc[idx] = inj.iloc[idx] * rng.choice([0.2,3.0], size=50)
truth = np.zeros(len(raw), dtype=bool); truth[idx] = True
hr_mean = inj.groupby(inj.index.hour).transform('mean')
hr_std  = inj.groupby(inj.index.hour).transform('std')
z = (inj - hr_mean) / (hr_std + 1e-9)
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=0)
iso_pred = iso.fit_predict(z.values.reshape(-1,1)) == -1

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
ax = axes[0]
two_weeks = (inj.index >= '2014-07-01') & (inj.index <= '2014-07-15')
ax.plot(inj.index[two_weeks], inj[two_weeks], lw=0.7, color='#444', label='Injected series')
inj_pts = (truth & two_weeks)
ax.scatter(inj.index[inj_pts], inj[inj_pts], color='red', s=30, zorder=5, label='Injected anomaly (truth)')
det_pts = (iso_pred & two_weeks)
ax.scatter(inj.index[det_pts], inj[det_pts], facecolors='none', edgecolors='blue', s=70, zorder=4, label='IsolationForest detection')
ax.legend(); ax.set_ylabel('Electricity [kW]')
ax.set_title('10a. Anomaly injection and detection (2-week window)')
ax = axes[1]
metr = ['Precision','Recall','F1']
x = np.arange(len(ad_df))
w = 0.25
for i, m in enumerate(metr):
    ax.bar(x + i*w, ad_df[m], w, label=m)
ax.set_xticks(x + w); ax.set_xticklabels(ad_df['Method'])
ax.set_ylim(0,1.05); ax.legend()
ax.set_title('10b. Anomaly detection performance against injected truth')
fig.suptitle('Figure 10. Anomaly detection demonstrative use case', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig10_anomaly_detection.png')
plt.close()

# ============================================================
# Fig 11: Imputation benchmark
# ============================================================
imp_df = pd.read_csv(OUT / 'imputation_benchmark.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
m_to_plot = ['MAE','RMSE','MAPE(%)']
xs = np.arange(len(imp_df))
w = 0.25
for i, m in enumerate(m_to_plot):
    ax[0].bar(xs+i*w, imp_df[m], w, label=m)
ax[0].set_xticks(xs+w); ax[0].set_xticklabels(imp_df['Method'], rotation=30)
ax[0].legend(); ax[0].set_title('11a. Imputation error by method (5% missing)')

# Demonstrative reconstruction snippet
mask_rng = np.random.default_rng(0)
truth_e = total_c['Electricity [kW]'].copy()
mask = mask_rng.random(len(truth_e)) < 0.05
masked = truth_e.copy(); masked[mask] = np.nan
linear = masked.interpolate(method='linear', limit_direction='both')
hr_mean_imp = masked.groupby(masked.index.hour).transform('mean')
hourly = masked.fillna(hr_mean_imp)

w2 = (truth_e.index >= '2014-09-01') & (truth_e.index <= '2014-09-08')
ax[1].plot(truth_e.index[w2], truth_e[w2], 'k', lw=1, label='Truth')
ax[1].plot(masked.index[w2], masked[w2], 'o', color='#264653', ms=2.5, label='Observed (5% gap)')
ax[1].plot(linear.index[w2], linear[w2], color='#e76f51', lw=1, label='Linear interp')
ax[1].plot(hourly.index[w2], hourly[w2], color='#2a9d8f', lw=1, alpha=0.8, label='Hourly mean')
ax[1].legend(fontsize=8); ax[1].set_title('11b. 1-week reconstruction example')
fig.suptitle('Figure 11. Imputation benchmark on Total electricity', fontsize=13)
plt.tight_layout()
plt.savefig(IMG / 'fig11_imputation_benchmark.png')
plt.close()

print('All figures saved to', IMG)
for f in sorted(IMG.iterdir()):
    print(' ', f.name, f.stat().st_size//1024, 'KB')
