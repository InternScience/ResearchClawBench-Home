"""
Step 6: Build all figures for the report.
- data_overview.png : map of mangrove sample points + TC tracks
- tc_baseline_map.png : TC exposure map at mangrove points
- slr_scenario_maps.png : 3-panel SLR rate maps
- composite_risk_map.png : 3-panel composite risk map
- country_topN_ranking.png : top countries by SSP5-8.5 risk
- ecosystem_services_exposure.png : exposed pop/stock/area per scenario
- threshold_decomposition.png : fraction of points above each Saintilan threshold
- tc_vs_slr_scatter.png : interpretability — TC vs SLR risk per point per scenario
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
IMG = os.path.join(ROOT, 'report/images')
os.makedirs(IMG, exist_ok=True)

mg = pd.read_csv(os.path.join(ROOT, 'outputs/mangrove_point_risk_with_country.csv'))
country = pd.read_csv(os.path.join(ROOT, 'outputs/country_risk_summary.csv'))
es = pd.read_csv(os.path.join(ROOT, 'outputs/ecosystem_service_exposure.csv'))
sg = pd.read_csv(os.path.join(ROOT, 'outputs/scenario_comparison_global.csv'))

# Load tc tracks lightly for backdrop
tc = xr.open_dataset(os.path.join(ROOT, 'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc'))
tc_lat = tc.lat.values
tc_lon = tc.lon.values
tc_w = tc.wind.values
tc.close()

plt.rcParams.update({'font.size':10, 'figure.dpi':110})

# --- 1. Data overview ---
fig, ax = plt.subplots(figsize=(12,5.5))
ax.scatter(tc_lon[::20], tc_lat[::20], s=0.5, c='#888', alpha=0.4, label='MIT TC tracks (≥33 m/s, sub-sampled)')
ax.scatter(mg.lon, mg.lat, s=2, c='#1b7837', alpha=0.7, label='GMW v4 mangrove sample points')
ax.set_xlim(-180,180); ax.set_ylim(-50,50)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Data overview: mangrove sample sites and historical TC tracks')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG,'data_overview.png'), dpi=140)
plt.close()
print('saved data_overview.png')

# --- 2. TC baseline exposure map at mangrove sites ---
fig, ax = plt.subplots(figsize=(12,5.5))
sc = ax.scatter(mg.lon, mg.lat, s=4, c=mg.tc_intense_per_decade, cmap='YlOrRd',
                vmin=0, vmax=np.percentile(mg.tc_intense_per_decade, 98))
plt.colorbar(sc, ax=ax, label='Intense TC (Cat 4-5) points per decade within 200 km')
ax.set_xlim(-180,180); ax.set_ylim(-50,50)
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Tropical-cyclone baseline exposure at mangrove sample points\n(MIT downscaled MPI-ESM1-2-HR historical, 1850-2014)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG,'tc_baseline_map.png'), dpi=140)
plt.close()
print('saved tc_baseline_map.png')

# --- 3. SLR scenario maps (3-panel) ---
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, sharey=True)
for ax, sc_name, label in zip(axes, ['ssp245','ssp370','ssp585'], ['SSP2-4.5','SSP3-7.0','SSP5-8.5']):
    rate = mg[f'slr_rate_mm_yr_{sc_name}_2100']
    s = ax.scatter(mg.lon, mg.lat, s=3, c=rate, cmap='viridis',
                   vmin=0, vmax=np.percentile(rate,99))
    plt.colorbar(s, ax=ax, label='SLR rate at 2100 (mm yr⁻¹)')
    ax.set_title(f'AR6 medium-confidence median SLR rate at 2100 — {label}')
    ax.set_xlim(-180,180); ax.set_ylim(-45,45)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Longitude')
for a in axes: a.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(IMG,'slr_scenario_maps.png'), dpi=140)
plt.close()
print('saved slr_scenario_maps.png')

# --- 4. Composite risk map (3-panel) ---
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, sharey=True)
for ax, sc_name, label in zip(axes, ['ssp245','ssp370','ssp585'], ['SSP2-4.5','SSP3-7.0','SSP5-8.5']):
    s = ax.scatter(mg.lon, mg.lat, s=3, c=mg[f'composite_risk_{sc_name}'], cmap='magma_r',
                   vmin=0, vmax=1)
    plt.colorbar(s, ax=ax, label='Composite TC+SLR risk (0–1)')
    ax.set_title(f'Composite mangrove risk index — {label} (2100)')
    ax.set_xlim(-180,180); ax.set_ylim(-45,45)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Longitude')
for a in axes: a.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(IMG,'composite_risk_map.png'), dpi=140)
plt.close()
print('saved composite_risk_map.png')

# --- 5. Country top-N ranking ---
top = country[country['n_points'] >= 20].sort_values('mean_composite_risk_ssp585', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(11,8))
ypos = np.arange(len(top))
w = 0.27
ax.barh(ypos+w, top['mean_composite_risk_ssp245'], w, label='SSP2-4.5', color='#4575b4')
ax.barh(ypos,   top['mean_composite_risk_ssp370'], w, label='SSP3-7.0', color='#fdae61')
ax.barh(ypos-w, top['mean_composite_risk_ssp585'], w, label='SSP5-8.5', color='#d73027')
ax.set_yticks(ypos); ax.set_yticklabels(top['Country'])
ax.invert_yaxis()
ax.set_xlabel('Country-mean composite risk (0–1)')
ax.set_title('Top 20 countries by SSP5-8.5 mean composite risk\n(restricted to ≥20 mangrove sample points)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG,'country_topN_ranking.png'), dpi=140)
plt.close()
print('saved country_topN_ranking.png')

# --- 6. Ecosystem service exposure ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
labels = ['SSP2-4.5','SSP3-7.0','SSP5-8.5']
colors = ['#4575b4','#fdae61','#d73027']
es2 = es.set_index('scenario')

ax = axes[0]
ax.bar(labels, es2.loc[['ssp245','ssp370','ssp585'],'global_high_risk_pop_2020']/1e6, color=colors)
ax.set_title('Population in high-risk countries (2020 baseline)')
ax.set_ylabel('Million people')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ax.bar(labels, es2.loc[['ssp245','ssp370','ssp585'],'global_high_risk_stock_2020']/1e9, color=colors)
ax.set_title('Mangrove natural-capital stock at risk')
ax.set_ylabel('Stock value (USD billion, 2020)')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.bar(labels, es2.loc[['ssp245','ssp370','ssp585'],'global_high_risk_mang_ha_2020']/1e6, color=colors)
ax.set_title('Mangrove area in high-risk countries')
ax.set_ylabel('Area (Mha)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG,'ecosystem_services_exposure.png'), dpi=140)
plt.close()
print('saved ecosystem_services_exposure.png')

# --- 7. Threshold decomposition / risk class composition ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
scs = ['ssp245','ssp370','ssp585']
labels_sc = ['SSP2-4.5','SSP3-7.0','SSP5-8.5']
classes = ['low','medium','high','very_high']
class_cols = ['#1a9850','#fee08b','#fdae61','#d73027']
data = np.array([[ (mg[f'risk_class_{s}'] == c).mean() for c in classes ] for s in scs])
bottom = np.zeros(len(scs))
for i,c in enumerate(classes):
    ax.bar(labels_sc, data[:,i], bottom=bottom, color=class_cols[i], label=c.replace('_',' '))
    bottom += data[:,i]
ax.set_ylim(0,1.0)
ax.set_ylabel('Fraction of mangrove sample points')
ax.set_title('Composite risk class composition by scenario')
ax.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
above4 = [ (mg[f'slr_rate_mm_yr_{s}_2100']>=4).mean() for s in scs ]
above7 = [ (mg[f'slr_rate_mm_yr_{s}_2100']>=7).mean() for s in scs ]
above10 = [ (mg[f'slr_rate_mm_yr_{s}_2100']>=10).mean() for s in scs ]
x = np.arange(len(scs)); w = 0.27
ax.bar(x-w, above4, w, label='≥4 mm/yr (likely deficit)', color='#fee08b')
ax.bar(x,   above7, w, label='≥7 mm/yr (highly likely deficit)', color='#fdae61')
ax.bar(x+w, above10, w, label='≥10 mm/yr (extreme)', color='#d73027')
ax.set_xticks(x); ax.set_xticklabels(labels_sc)
ax.set_ylabel('Fraction of mangrove sample points'); ax.set_ylim(0,1.05)
ax.set_title('Mangrove fraction above Saintilan et al. (2023) RSLR thresholds at 2100')
ax.legend(loc='lower right'); ax.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG,'threshold_decomposition.png'), dpi=140)
plt.close()
print('saved threshold_decomposition.png')

# --- 8. TC vs SLR scatter (interpretability) ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
for ax, sc_name, label in zip(axes, scs, labels_sc):
    sc = ax.scatter(mg['tc_risk'], mg[f'slr_risk_{sc_name}'], s=3,
                    c=mg[f'composite_risk_{sc_name}'], cmap='magma_r', vmin=0, vmax=1, alpha=0.6)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.set_xlabel('TC risk component')
    if ax is axes[0]: ax.set_ylabel('SLR risk component')
    ax.set_title(label)
    ax.grid(alpha=0.3)
fig.colorbar(sc, ax=axes, label='Composite risk', shrink=0.8)
plt.savefig(os.path.join(IMG,'tc_vs_slr_scatter.png'), dpi=140, bbox_inches='tight')
plt.close()
print('saved tc_vs_slr_scatter.png')

# --- 9. Country exposed pop/stock bar (top-15) ---
country['exposed_pop_585'] = country['Risk_Pop_2020'] * country['frac_at_or_above_high_ssp585']
country['exposed_stock_585'] = country['Risk_Stock_2020'] * country['frac_at_or_above_high_ssp585']
top15p = country.sort_values('exposed_pop_585', ascending=False).head(15)
top15s = country.sort_values('exposed_stock_585', ascending=False).head(15)

fig, axes = plt.subplots(1,2, figsize=(14,6))
ax = axes[0]
ax.barh(top15p['Country'][::-1], top15p['exposed_pop_585'][::-1]/1e6, color='#d73027')
ax.set_xlabel('Population in high-risk mangroves under SSP5-8.5 (millions)')
ax.set_title('Top 15 countries by exposed population (Risk_Pop_2020 × P(high risk))')
ax.grid(axis='x', alpha=0.3)
ax = axes[1]
ax.barh(top15s['Country'][::-1], top15s['exposed_stock_585'][::-1]/1e9, color='#7d3c98')
ax.set_xlabel('Mangrove natural-capital stock at risk (USD billion, 2020)')
ax.set_title('Top 15 countries by exposed stock value')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG,'country_exposed_es.png'), dpi=140)
plt.close()
print('saved country_exposed_es.png')

print('\nAll figures saved to', IMG)
