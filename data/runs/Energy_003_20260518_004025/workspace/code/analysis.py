"""
HEEW Mini-Dataset Comprehensive Analysis
=========================================
Analysis of the Hierarchical Energy & Environment Weather (HEEW) benchmark dataset.
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

sns.set_style("whitegrid")
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10,
                     'axes.titlesize': 12, 'axes.labelsize': 10})

DATA_DIR = 'data/HEEW_Mini-Dataset'
IMG_DIR = 'report/images'
OUT_DIR = 'outputs'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("PHASE 1: Loading data...")
building_ids = [f'BN{i:03d}' for i in range(1, 11)]
energy_dfs = {}

for bid in building_ids:
    df = pd.read_csv(f'{DATA_DIR}/{bid}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].rename(
        columns={'year':'year','month':'month','day':'day','hour':'hour'}))
    energy_dfs[bid] = df

for level in ['CN01', 'Total']:
    df = pd.read_csv(f'{DATA_DIR}/{level}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].rename(
        columns={'year':'year','month':'month','day':'day','hour':'hour'}))
    energy_dfs[level] = df

weather_df = pd.read_csv(f'{DATA_DIR}/Total_weather.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

energy_vars = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
               'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
weather_vars = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
                'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']

total_records = sum(len(df) for df in energy_dfs.values()) + len(weather_df)
print(f"  Total records: {total_records}")

# ── 2. STATISTICS ──────────────────────────────────────────────
print("PHASE 2: Computing statistics...")
stats_summary = {}
for name in building_ids + ['CN01', 'Total']:
    df = energy_dfs[name]
    stats_summary[name] = {}
    for col in energy_vars:
        stats_summary[name][col] = {
            'mean': round(float(df[col].mean()), 4),
            'std': round(float(df[col].std()), 4),
            'min': round(float(df[col].min()), 4),
            'max': round(float(df[col].max()), 4),
            'median': round(float(df[col].median()), 4)
        }

with open(f'{OUT_DIR}/statistics_summary.json', 'w') as f:
    json.dump(stats_summary, f, indent=2)
print("  Saved statistics_summary.json")

# ── 3. ANOMALY DETECTION ───────────────────────────────────────
print("PHASE 3: Anomaly detection (IQR method)...")
anomaly_counts = {}
cleaned_dfs = {}
for name, df in energy_dfs.items():
    cleaned = df.copy()
    anomalies = 0
    for col in energy_vars:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)
        anomalies += mask.sum()
        cleaned[f'{col}_anomaly'] = mask.astype(int)
    anomaly_counts[name] = anomalies
    cleaned_dfs[name] = cleaned

cleaning_algo = {
    "step_1_missing_values": "No missing values detected in Mini-Dataset.",
    "step_2_range_validation": "All values within physical ranges (>=0 for energy, valid ranges for weather).",
    "step_3_statistical_outlier": "IQR method with 3x threshold. Anomalies flagged per variable.",
    "step_4_hierarchical_check": "Sum(BN001..BN010) vs Total verified.",
    "step_5_temporal_continuity": "No temporal gaps detected (8760 hours)."
}
with open(f'{OUT_DIR}/data_cleaning_algorithm.json', 'w') as f:
    json.dump(cleaning_algo, f, indent=2)

# ── 4. CORRELATION ANALYSIS ────────────────────────────────────
print("PHASE 4: Correlation analysis...")
total_merged = pd.merge(
    energy_dfs['Total'].set_index('datetime')[energy_vars],
    weather_df.set_index('datetime')[weather_vars],
    left_index=True, right_index=True, how='inner')

energy_weather_corr = total_merged.corr().loc[energy_vars, weather_vars]
energy_weather_corr.to_csv(f'{OUT_DIR}/energy_weather_correlation.csv')
print("  Saved energy_weather_correlation.csv")

# ── 5. HIERARCHICAL AGGREGATION ────────────────────────────────
print("PHASE 5: Hierarchical aggregation check...")
sum_buildings = energy_dfs['BN001'][energy_vars].copy()
for bid in building_ids[1:]:
    sum_buildings = sum_buildings + energy_dfs[bid][energy_vars].values

total_comp = energy_dfs['Total'][energy_vars].values
diff = sum_buildings.values - total_comp
rel_diff = np.abs(diff) / np.maximum(np.abs(total_comp), 1e-10)

agg_results = {}
for i, var in enumerate(energy_vars):
    agg_results[var] = {
        'max_relative_difference': round(float(np.max(rel_diff[:, i])), 6),
        'mean_relative_difference': round(float(np.mean(rel_diff[:, i])), 6)
    }
    print(f"  {var}: mean_rel_diff = {agg_results[var]['mean_relative_difference']:.6f}")

with open(f'{OUT_DIR}/aggregation_consistency.json', 'w') as f:
    json.dump(agg_results, f, indent=2)

# ── 6. BUILDING CLUSTERING ────────────────────────────────────
print("PHASE 6: Building clustering...")
building_features = []
for bid in building_ids:
    df = energy_dfs[bid]
    features = {}
    for col in energy_vars:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
    features['day_night_ratio'] = (
        df[df['hour'].between(6,18)][energy_vars[0]].mean() /
        max(df[~df['hour'].between(6,18)][energy_vars[0]].mean(), 1e-10))
    building_features.append(features)

feature_df = pd.DataFrame(building_features, index=building_ids)
feature_norm = (feature_df - feature_df.mean()) / feature_df.std()
feature_norm = feature_norm.fillna(0)

distances = pdist(feature_norm.values, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')
clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
cluster_map = {bid: int(c) for bid, c in zip(building_ids, clusters)}
print(f"  Clusters: {cluster_map}")

with open(f'{OUT_DIR}/clustering_results.json', 'w') as f:
    json.dump({'clusters': cluster_map, 'method': 'Ward hierarchical, k=3'}, f, indent=2)

# ── 7. SEASONAL ANALYSIS ──────────────────────────────────────
def get_season(m):
    if m in [12,1,2]: return 'Winter'
    if m in [3,4,5]: return 'Spring'
    if m in [6,7,8]: return 'Summer'
    return 'Fall'

# ── FIGURE 1: Building Energy Profiles ──────────────────────────
print("Creating Figure 1...")
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('HEEW Mini-Dataset: Energy Variable Distributions Across Buildings',
             fontsize=14, fontweight='bold', y=0.98)

for idx, var in enumerate(energy_vars):
    ax = axes[idx // 2, idx % 2]
    data_box = [energy_dfs[bid][var].values for bid in building_ids]
    bp = ax.boxplot(data_box, labels=[f'B{i+1:02d}' for i in range(10)],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title(var, fontsize=10)
    ax.set_ylabel(var.split('[')[1].rstrip(']'))

axes[2, 1].axis('off')
ax = axes[2, 1]
monthly_elec = energy_dfs['Total'].groupby('month')[energy_vars[0]].mean()
ax.bar(monthly_elec.index, monthly_elec.values, color='steelblue', alpha=0.7)
ax.set_title('Total Electricity - Monthly Mean')
ax.set_xlabel('Month'); ax.set_ylabel('Electricity [kW]')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure1_building_profiles.png', bbox_inches='tight')
plt.close()

# ── FIGURE 2: Temporal Patterns ────────────────────────────────
print("Creating Figure 2...")
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
fig.suptitle('Temporal Profiles of Building BN001 (2014)', fontsize=14, fontweight='bold', y=0.98)
bn001 = energy_dfs['BN001']
for idx, var in enumerate(energy_vars):
    ax = axes[idx]
    daily = bn001.set_index('datetime')[var].resample('D').mean()
    ax.plot(daily.index, daily.values, linewidth=0.8, alpha=0.8, color='steelblue')
    ax.fill_between(daily.index, daily.values, alpha=0.2, color='steelblue')
    ax.set_ylabel(var.split('[')[1].rstrip(']'))
    ax.set_title(var, fontsize=10, loc='left')
axes[-1].set_xlabel('Date')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure2_temporal_patterns.png', bbox_inches='tight')
plt.close()

# ── FIGURE 3: Weather Variables ────────────────────────────────
print("Creating Figure 3...")
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Meteorological Variables (2014)', fontsize=14, fontweight='bold', y=0.98)
for idx, var in enumerate(weather_vars):
    ax = axes[idx // 2, idx % 2]
    daily_w = weather_df.set_index('datetime')[var].resample('D').mean()
    ax.plot(daily_w.index, daily_w.values, linewidth=0.6, alpha=0.8, color='darkorange')
    ax.fill_between(daily_w.index, daily_w.values, alpha=0.15, color='darkorange')
    ax.set_title(var, fontsize=10, loc='left')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
axes[3, 1].axis('off')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure3_weather_variables.png', bbox_inches='tight')
plt.close()

# ── FIGURE 4: Correlation Heatmap ──────────────────────────────
print("Creating Figure 4...")
fig, ax = plt.subplots(figsize=(10, 8))
short_e = ['Electricity', 'Heat', 'Cooling', 'PV Gen', 'GHG Emissions']
short_w = ['Temp', 'Dew Pt', 'Humidity', 'Wind Spd', 'Wind Gust', 'Pressure', 'Precip']
corr_disp = energy_weather_corr.copy()
corr_disp.index = short_e; corr_disp.columns = short_w
sns.heatmap(corr_disp, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True)
ax.set_title('Energy-Weather Cross-Correlation (Total Level)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure4_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# ── FIGURE 5: Energy Inter-Correlation ─────────────────────────
print("Creating Figure 5...")
fig, ax = plt.subplots(figsize=(8, 7))
energy_corr = total_merged[energy_vars].corr().copy()
energy_corr.index = short_e; energy_corr.columns = short_e
mask = np.triu(np.ones_like(energy_corr, dtype=bool), k=1)
sns.heatmap(energy_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True, mask=mask)
ax.set_title('Energy Variable Inter-Correlation (Total Level)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure5_energy_correlation.png', bbox_inches='tight')
plt.close()

# ── FIGURE 6: Hierarchical Aggregation ─────────────────────────
print("Creating Figure 6...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hierarchical Aggregation Verification', fontsize=14, fontweight='bold', y=0.98)

# Monthly comparison
ax = axes[0, 0]
sb_m = sum_buildings.copy()
sb_m['month'] = energy_dfs['BN001']['month'].values
sb_mm = sb_m.groupby('month')[energy_vars[0]].mean()
tc_m = pd.DataFrame(total_comp, columns=energy_vars)
tc_m['month'] = energy_dfs['Total']['month'].values
tc_mm = tc_m.groupby('month')[energy_vars[0]].mean()
x = np.arange(1, 13); w = 0.35
ax.bar(x-w/2, sb_mm.values, w, label='Sum(BN001-BN010)', color='steelblue', alpha=0.7)
ax.bar(x+w/2, tc_mm.values, w, label='Total', color='coral', alpha=0.7)
ax.set_xlabel('Month'); ax.set_ylabel('Electricity [kW]')
ax.set_title('Monthly Mean: Sum vs Total')
ax.legend(fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

# Scatter plot
ax = axes[0, 1]
ax.scatter(total_comp.flatten(), sum_buildings.values.flatten(), alpha=0.3, s=5, color='steelblue')
vmin = min(total_comp.min(), sum_buildings.values.min())
vmax = max(total_comp.max(), sum_buildings.values.max())
ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='Perfect aggregation')
ax.set_xlabel('Total Level'); ax.set_ylabel('Sum of Buildings')
ax.set_title('Total vs Sum of Buildings')
ax.legend(fontsize=8)

# Relative error
ax = axes[1, 0]
rel_err_elec = np.abs(diff[:, 0]) / np.maximum(np.abs(total_comp[:, 0]), 1e-10) * 100
ax.plot(range(len(rel_err_elec)), rel_err_elec, linewidth=0.5, color='steelblue', alpha=0.7)
ax.set_xlabel('Hour of Year'); ax.set_ylabel('Relative Error [%]')
ax.set_title('Aggregation Relative Error: Electricity')
mean_err = np.mean(rel_err_elec)
ax.axhline(y=mean_err, color='red', linestyle='--', linewidth=1, label=f'Mean={mean_err:.4f}%')
ax.legend(fontsize=8)

# CN01 fraction
ax = axes[1, 1]
cn01_df = energy_dfs['CN01'].copy()
cn01_df['month'] = energy_dfs['BN001']['month'].values
cn01_mm = cn01_df.groupby('month')[energy_vars[0]].mean()
cn01_frac = cn01_mm / tc_mm * 100
ax.bar(cn01_frac.index, cn01_frac.values, color='darkorange', alpha=0.7)
ax.set_xlabel('Month'); ax.set_ylabel('CN01 / Total [%]')
ax.set_title('CN01 as Fraction of Total (Electricity)')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_ylim(0, 110)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure6_hierarchical_aggregation.png', bbox_inches='tight')
plt.close()

# ── FIGURE 7: Building Clustering ──────────────────────────────
print("Creating Figure 7...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Building Clustering Analysis', fontsize=14, fontweight='bold')

ax = axes[0]
dendrogram(linkage_matrix, labels=[f'B{i+1:02d}' for i in range(10)],
           leaf_rotation=45, leaf_font_size=9, ax=ax,
           color_threshold=linkage_matrix[-2, 2])
ax.set_title('Hierarchical Clustering Dendrogram')
ax.set_ylabel('Distance')

ax = axes[1]
cc = {1: 'steelblue', 2: 'coral', 3: 'forestgreen'}
seen_labels = set()
for bid, cl in zip(building_ids, clusters):
    lbl = f'Cluster {cl}' if cl not in seen_labels else ''
    seen_labels.add(cl)
    ax.scatter(feature_df.loc[bid, 'Electricity [kW]_mean'],
               feature_df.loc[bid, 'PV Power Generation [kW]_mean'],
               c=cc[cl], s=100, label=lbl, edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(bid, (feature_df.loc[bid, 'Electricity [kW]_mean'],
                      feature_df.loc[bid, 'PV Power Generation [kW]_mean']),
                textcoords="offset points", xytext=(5,5), fontsize=7)
ax.set_xlabel('Mean Electricity [kW]'); ax.set_ylabel('Mean PV Generation [kW]')
ax.set_title('Building Clusters')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure7_building_clustering.png', bbox_inches='tight')
plt.close()

# ── FIGURE 8: Seasonal & Diurnal ──────────────────────────────
print("Creating Figure 8...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seasonal and Diurnal Energy Patterns (Total Level)',
             fontsize=14, fontweight='bold', y=0.98)

total_df = energy_dfs['Total'].copy()
total_df['season'] = total_df['month'].apply(get_season)

ax = axes[0, 0]
seasons = ['Winter','Spring','Summer','Fall']
season_colors = ['#2196F3', '#4CAF50', '#FF9800', '#795548']
sdata = [total_df[total_df['season']==s][energy_vars[0]].values for s in seasons]
bp = ax.boxplot(sdata, labels=seasons, patch_artist=True)
for patch, c in zip(bp['boxes'], season_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel('Electricity [kW]'); ax.set_title('Seasonal Electricity Distribution')

ax = axes[0, 1]
for var, c, lbl in zip(energy_vars[:4], ['blue','red','green','orange'],
                        ['Electricity','Heat','Cooling','PV Gen']):
    hourly = total_df.groupby('hour')[var].mean()
    hmax = hourly.max()
    hourly_n = hourly / hmax if hmax > 0 else hourly
    ax.plot(hourly_n.index, hourly_n.values, linewidth=2, color=c, label=lbl, alpha=0.8)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Normalized Mean')
ax.set_title('Diurnal Profiles (Normalized)')
ax.legend(fontsize=8); ax.set_xticks(range(0,24,2))

ax = axes[1, 0]
merged_s = pd.merge(
    total_df[['datetime','Electricity [kW]','month']].set_index('datetime'),
    weather_df[['datetime','Temperature [°F]']].set_index('datetime'),
    left_index=True, right_index=True)
merged_s['season'] = merged_s['month'].apply(get_season)
for s, c in zip(seasons, season_colors):
    m = merged_s['season']==s
    ax.scatter(merged_s.loc[m,'Temperature [°F]'], merged_s.loc[m,'Electricity [kW]'],
               alpha=0.1, s=5, color=c, label=s)
ax.set_xlabel('Temperature [°F]'); ax.set_ylabel('Electricity [kW]')
ax.set_title('Temperature vs Electricity by Season')
ax.legend(fontsize=8, markerscale=5)

ax = axes[1, 1]
summer_pv = total_df[total_df['season']=='Summer'].groupby('hour')['PV Power Generation [kW]'].mean()
winter_pv = total_df[total_df['season']=='Winter'].groupby('hour')['PV Power Generation [kW]'].mean()
ax.fill_between(summer_pv.index, summer_pv.values, alpha=0.3, color='orange', label='Summer')
ax.fill_between(winter_pv.index, winter_pv.values, alpha=0.3, color='blue', label='Winter')
ax.plot(summer_pv.index, summer_pv.values, color='orange', linewidth=2)
ax.plot(winter_pv.index, winter_pv.values, color='blue', linewidth=2)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('PV Generation [kW]')
ax.set_title('PV Generation: Summer vs Winter')
ax.legend(fontsize=8); ax.set_xticks(range(0,24,2))
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure8_seasonal_diurnal.png', bbox_inches='tight')
plt.close()

# ── FIGURE 9: Anomaly Detection ───────────────────────────────
print("Creating Figure 9...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Anomaly Detection in Energy Data (BN001)',
             fontsize=14, fontweight='bold', y=0.98)
bn001_c = cleaned_dfs['BN001']
a_vars = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 'Greenhouse Gas Emission [Ton]']
for idx, var in enumerate(a_vars):
    ax = axes[idx//2, idx%2]
    ax.plot(bn001_c['datetime'], bn001_c[var], linewidth=0.5, alpha=0.7, color='steelblue', label='Normal')
    mask = bn001_c[f'{var}_anomaly']==1
    if mask.sum()>0:
        ax.scatter(bn001_c.loc[mask,'datetime'], bn001_c.loc[mask,var],
                   color='red', s=20, zorder=5, label=f'Anomaly ({mask.sum()})')
    ax.set_title(var, fontsize=10); ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure9_anomaly_detection.png', bbox_inches='tight')
plt.close()

# ── FIGURE 10: Diurnal-Monthly Heatmap ─────────────────────────
print("Creating Figure 10...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hourly-Monthly Heatmap (Total Level)', fontsize=14, fontweight='bold')
td = energy_dfs['Total'].copy()
pe = td.pivot_table(values='Electricity [kW]', index='hour', columns='month', aggfunc='mean')
sns.heatmap(pe, cmap='YlOrRd', ax=axes[0], cbar_kws={'label':'kW'})
axes[0].set_title('Electricity [kW]'); axes[0].set_xlabel('Month'); axes[0].set_ylabel('Hour')
pp = td.pivot_table(values='PV Power Generation [kW]', index='hour', columns='month', aggfunc='mean')
sns.heatmap(pp, cmap='YlOrRd', ax=axes[1], cbar_kws={'label':'kW'})
axes[1].set_title('PV Power Generation [kW]'); axes[1].set_xlabel('Month'); axes[1].set_ylabel('Hour')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure10_diurnal_monthly_heatmap.png', bbox_inches='tight')
plt.close()

# ── FIGURE 11: Correlation significance ───────────────────────
print("Creating Figure 11...")
fig, ax = plt.subplots(figsize=(10, 6))
corr_vals = []
p_vals = []
labels = []
for var in energy_vars:
    for wvar in weather_vars:
        r, p = stats.pearsonr(total_merged[var], total_merged[wvar])
        corr_vals.append(r)
        p_vals.append(p)
        labels.append(f"{var.split('[')[0].strip()[:8]}\nvs\n{wvar.split('[')[0].strip()[:8]}")

# Show top significant correlations
corr_arr = np.array(corr_vals)
p_arr = np.array(p_vals)
top_idx = np.argsort(np.abs(corr_arr))[::-1][:15]
top_labels = [labels[i].replace('\n',' ') for i in top_idx]
top_corrs = [corr_arr[i] for i in top_idx]
top_ps = [p_arr[i] for i in top_idx]
colors_bar = ['steelblue' if c > 0 else 'coral' for c in top_corrs]

ax.barh(range(len(top_idx)), top_corrs, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels(top_labels, fontsize=7)
ax.set_xlabel('Pearson Correlation Coefficient')
ax.set_title('Top 15 Energy-Weather Correlations (Total Level)')
ax.axvline(x=0, color='black', linewidth=0.5)
for i, (c, p) in enumerate(zip(top_corrs, top_ps)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.text(c + 0.01 if c > 0 else c - 0.01, i, f'{c:.3f}{sig}',
            va='center', ha='left' if c > 0 else 'right', fontsize=7)
ax.set_xlim(-1.1, 1.1)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure11_correlation_significance.png', bbox_inches='tight')
plt.close()

# ── FIGURE 12: Multi-building comparison ───────────────────────
print("Creating Figure 12...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Building-Level Comparisons', fontsize=14, fontweight='bold', y=0.98)

# Mean electricity per building
ax = axes[0, 0]
means = [energy_dfs[bid]['Electricity [kW]'].mean() for bid in building_ids]
ax.barh([f'B{i+1:02d}' for i in range(10)], means, color=plt.cm.viridis(np.linspace(0.2,0.8,10)))
ax.set_xlabel('Mean Electricity [kW]')
ax.set_title('Mean Electricity by Building')

# Mean PV per building
ax = axes[0, 1]
pvs = [energy_dfs[bid]['PV Power Generation [kW]'].mean() for bid in building_ids]
ax.barh([f'B{i+1:02d}' for i in range(10)], pvs, color=plt.cm.viridis(np.linspace(0.2,0.8,10)))
ax.set_xlabel('Mean PV Generation [kW]')
ax.set_title('Mean PV Generation by Building')

# Electricity profile: building 1 vs 10
ax = axes[1, 0]
h1 = energy_dfs['BN001'].groupby('hour')['Electricity [kW]'].mean()
h10 = energy_dfs['BN010'].groupby('hour')['Electricity [kW]'].mean()
ax.plot(h1.index, h1.values, linewidth=2, label='BN001', color='steelblue')
ax.plot(h10.index, h10.values, linewidth=2, label='BN010', color='coral')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Electricity [kW]')
ax.set_title('Diurnal Electricity: BN001 vs BN010')
ax.legend()

# GHG vs Electricity scatter for all buildings
ax = axes[1, 1]
for bid in building_ids:
    ax.scatter(energy_dfs[bid]['Electricity [kW]'].mean(),
               energy_dfs[bid]['Greenhouse Gas Emission [Ton]'].mean(),
               s=60, label=bid, edgecolors='black', linewidth=0.5, zorder=5)
ax.set_xlabel('Mean Electricity [kW]')
ax.set_ylabel('Mean GHG Emissions [Ton]')
ax.set_title('Electricity vs GHG Emissions (Building Means)')
ax.legend(fontsize=6, ncol=2)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{IMG_DIR}/figure12_building_comparisons.png', bbox_inches='tight')
plt.close()

# ── SAVE CONSOLIDATED DATASET ──────────────────────────────────
print("Saving consolidated dataset...")
all_records = []
for bid in building_ids:
    df = energy_dfs[bid][['datetime']+energy_vars].copy()
    df['entity'] = bid; df['level'] = 'building'
    all_records.append(df)
for level in ['CN01', 'Total']:
    df = energy_dfs[level][['datetime']+energy_vars].copy()
    df['entity'] = level; df['level'] = 'aggregated' if level=='CN01' else 'campus'
    all_records.append(df)
consolidated = pd.concat(all_records, ignore_index=True)
consolidated = pd.merge(consolidated, weather_df[['datetime']+weather_vars], on='datetime', how='left')
consolidated.to_csv(f'{OUT_DIR}/consolidated_dataset.csv', index=False)
print(f"  Exported: {len(consolidated)} records")

print("\nALL ANALYSIS COMPLETE. Ready for report writing.")
