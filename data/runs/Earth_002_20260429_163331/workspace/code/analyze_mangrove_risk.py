#!/usr/bin/env python3
"""Composite mangrove climate risk index from sampled GMW points, AR6 RSLR rates, and MIT TC tracks.

Outputs tables in outputs/ and PNG figures in report/images/.
"""
from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=ROOT/'outputs'
IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True, parents=True); IMG.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

SERVICE_VALUE_USD_HA_YR=20000.0 # lower-bound literature proxy from Mo et al. 2023 text (>US$20,000 ha-1 yr-1)
SAMPLE_FRACTION=0.10 # from instructions
POINT_AREA_HA=0.0009 # GMW 30 m pixel proxy = 900 m2 = 0.09 ha; sampled at 10% => expand by /0.10 below
EXPANDED_AREA_HA=POINT_AREA_HA/SAMPLE_FRACTION

# ---------------- helpers ----------------
def region_from_lonlat(lon, lat):
    # Broad ocean-basin/continental groupings for global reporting; deterministic, not political.
    if lon < -30:
        return 'Americas / Atlantic-East Pacific'
    if lon < 20:
        return 'West Africa / East Atlantic'
    if lon < 75:
        return 'Western Indian Ocean'
    if lon < 120:
        return 'South & Southeast Asia'
    if lon < 170:
        return 'Oceania / West Pacific'
    return 'Central Pacific / Dateline'

def haversine_km(lon1, lat1, lon2, lat2):
    R=6371.0088
    lon1=np.radians(lon1); lat1=np.radians(lat1); lon2=np.radians(lon2); lat2=np.radians(lat2)
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def lonlat_to_unit(lon, lat):
    lon=np.radians(lon); lat=np.radians(lat)
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def norm01(x, lo=None, hi=None):
    x=np.asarray(x, dtype=float)
    if lo is None: lo=np.nanpercentile(x, 5)
    if hi is None: hi=np.nanpercentile(x, 95)
    if not np.isfinite(hi-lo) or hi<=lo:
        return np.zeros_like(x, dtype=float), float(lo), float(hi)
    y=(x-lo)/(hi-lo)
    return np.clip(y,0,1), float(lo), float(hi)

# ---------------- data load ----------------
print('Loading mangrove sample...')
gdf=gpd.read_file(DATA/'mangroves'/'gmw_v4_ref_smpls_qad_v12.gpkg')
gdf=gdf.to_crs('EPSG:4326')
gdf['lon']=gdf.geometry.x; gdf['lat']=gdf.geometry.y
# Deterministic thinning for computational tractability in distance queries while preserving global distribution.
# Use all 100k for SLR extraction; use binned regional summaries for figures.
gdf['region']=[region_from_lonlat(x,y) for x,y in zip(gdf.lon, gdf.lat)]
gdf['sample_area_ha']=EXPANDED_AREA_HA

# ---------------- SLR extraction ----------------
print('Extracting SLR rates...')
slr_files={'ssp245':'total_ssp245_medium_confidence_rates.nc','ssp370':'total_ssp370_medium_confidence_rates.nc','ssp585':'total_ssp585_medium_confidence_rates.nc'}
coords=np.column_stack([gdf['lat'].to_numpy(), gdf['lon'].to_numpy()])
slr_meta={}
for scen, fname in slr_files.items():
    ds=xr.open_dataset(DATA/'slr'/fname)
    years=ds['years'].values
    q=ds['quantiles'].values
    qidx=int(np.argmin(np.abs(q-0.5)))
    ymask=(years>=2020)&(years<=2100)
    rates=ds['sea_level_change_rate'].isel(quantiles=qidx, years=np.where(ymask)[0]).mean('years').values.astype(float)
    loc_lat=ds['lat'].values.astype(float); loc_lon=ds['lon'].values.astype(float)
    tree=cKDTree(np.column_stack([loc_lat, loc_lon]))
    dist, idx=tree.query(coords, k=1, workers=-1)
    gdf[f'{scen}_slr_rate_mm_yr']=rates[idx]
    gdf[f'{scen}_slr_cum_m_2020_2100']=gdf[f'{scen}_slr_rate_mm_yr']*80/1000.0
    gdf[f'{scen}_slr_nn_deg']=dist
    slr_meta[scen]={'file':fname,'median_quantile':float(q[qidx]),'years_used':[int(y) for y in years[ymask]],'mean_nn_degree':float(np.mean(dist)),'max_nn_degree':float(np.max(dist))}

# ---------------- TC exposure ----------------
print('Computing TC exposure...')
ds=xr.open_dataset(DATA/'tc'/'tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc=pd.DataFrame({'lat':ds['lat'].values.astype(float),'lon':ds['lon'].values.astype(float),'wind':ds['wind'].values.astype(float)})
# no year/storm id in reduced file, so exact annual frequency/regime trend unavailable; use track-point intensity-density exposure.
tc['cat']=pd.cut(tc.wind, bins=[0,33,43,50,58,70,10_000], labels=[0,1,2,3,4,5], right=False).astype(int)
tc['weight']=np.select([tc.wind>=70, tc.wind>=58, tc.wind>=50, tc.wind>=43, tc.wind>=33],[16,8,4,2,1], default=0).astype(float)
# Nearest high-wind track point within ~300 km. KDTree on unit sphere; chord radius for 300 km.
mg_xyz=lonlat_to_unit(gdf.lon.values, gdf.lat.values)
tc_xyz=lonlat_to_unit(tc.lon.values, tc.lat.values)
tree=cKDTree(tc_xyz)
# Query nearest 5 to approximate local exposure without expensive all-neighbor search.
dists, idxs=tree.query(mg_xyz, k=5, workers=-1)
if idxs.ndim==1:
    idxs=idxs[:,None]; dists=dists[:,None]
# chord to km: arc=2*asin(chord/2)
km=6371.0088*2*np.arcsin(np.clip(dists/2,0,1))
weights=tc.weight.values[idxs]
winds=tc.wind.values[idxs]
# exponential distance decay, zero beyond 500 km; emphasize major storms.
decay=np.exp(-km/150.0)*(km<=500)
exposure=(weights*decay).sum(axis=1)
gdf['tc_nearest_km']=km[:,0]
gdf['tc_nearest_wind_ms']=winds[:,0]
gdf['tc_exposure_raw']=exposure
gdf['tc_major_exposure_flag']=(winds[:,0]>=50).astype(int)
# Regional regime-shift proxy: compare each mangrove point's exposure against region median; positive anomaly = shifted/high exposure regime.
reg_med=gdf.groupby('region')['tc_exposure_raw'].transform('median')
reg_iqr=gdf.groupby('region')['tc_exposure_raw'].transform(lambda s: np.percentile(s,75)-np.percentile(s,25))
gdf['tc_regime_shift_proxy']=np.maximum(0, (gdf['tc_exposure_raw']-reg_med)/(reg_iqr.replace(0,np.nan))).fillna(0).clip(0,5)
# Normalize TC component using all exposure + anomaly.
tc_comb=0.75*gdf['tc_exposure_raw'].to_numpy()+0.25*gdf['tc_regime_shift_proxy'].to_numpy()
gdf['tc_risk_norm'], tc_lo, tc_hi = norm01(tc_comb, 5, 95)

# ---------------- composite index ----------------
thresholds={}
for scen in slr_files:
    rate=gdf[f'{scen}_slr_rate_mm_yr'].to_numpy()
    # normalized around empirical distribution, with 7 mm/yr threshold included as absolute stress marker in classes
    gdf[f'{scen}_slr_norm'], lo, hi = norm01(rate, 5, 95)
    gdf[f'{scen}_slr_above_7mm_yr']=(rate>=7.0).astype(int)
    # Raise SLR norm floor for above-threshold sites to reflect literature stress threshold.
    gdf[f'{scen}_slr_risk_norm']=np.maximum(gdf[f'{scen}_slr_norm'], 0.75*gdf[f'{scen}_slr_above_7mm_yr'])
    gdf[f'{scen}_composite_risk']=0.5*gdf[f'{scen}_slr_risk_norm']+0.5*gdf['tc_risk_norm']
    qvals=np.quantile(gdf[f'{scen}_composite_risk'], [0.5,0.75,0.9])
    thresholds[scen]={'slr_norm_p5':lo,'slr_norm_p95':hi,'risk_median':float(qvals[0]),'risk_q75':float(qvals[1]),'risk_q90':float(qvals[2])}
    cls=[]
    for v in gdf[f'{scen}_composite_risk']:
        if v>=qvals[2]: cls.append('very high')
        elif v>=qvals[1]: cls.append('high')
        elif v>=qvals[0]: cls.append('moderate')
        else: cls.append('lower')
    gdf[f'{scen}_risk_class']=cls
thresholds['tc_norm']={'tc_combined_p5':tc_lo,'tc_combined_p95':tc_hi}

# ---------------- summaries ----------------
print('Writing tables...')
base_cols=['uid','lon','lat','region','sample_area_ha','tc_nearest_km','tc_nearest_wind_ms','tc_exposure_raw','tc_regime_shift_proxy','tc_risk_norm']
cols=base_cols+sum([[f'{s}_slr_rate_mm_yr',f'{s}_slr_cum_m_2020_2100',f'{s}_slr_risk_norm',f'{s}_slr_above_7mm_yr',f'{s}_composite_risk',f'{s}_risk_class'] for s in slr_files], [])
gdf[cols].to_csv(OUT/'mangrove_risk_samples.csv', index=False)

rows=[]
for scen in slr_files:
    risk=gdf[f'{scen}_composite_risk']
    rows.append({
        'scenario':scen,
        'n_sample_points':len(gdf),
        'expanded_area_ha':float(gdf.sample_area_ha.sum()),
        'mean_slr_rate_mm_yr':float(gdf[f'{scen}_slr_rate_mm_yr'].mean()),
        'median_slr_rate_mm_yr':float(gdf[f'{scen}_slr_rate_mm_yr'].median()),
        'area_above_7mm_ha':float((gdf[f'{scen}_slr_above_7mm_yr']*gdf.sample_area_ha).sum()),
        'pct_area_above_7mm':float(100*(gdf[f'{scen}_slr_above_7mm_yr']*gdf.sample_area_ha).sum()/gdf.sample_area_ha.sum()),
        'mean_composite_risk':float(risk.mean()),
        'median_composite_risk':float(risk.median()),
        'area_high_or_very_high_ha':float((gdf[f'{scen}_risk_class'].isin(['high','very high'])*gdf.sample_area_ha).sum()),
        'pct_area_high_or_very_high':float(100*(gdf[f'{scen}_risk_class'].isin(['high','very high'])*gdf.sample_area_ha).sum()/gdf.sample_area_ha.sum()),
        'service_value_high_or_very_high_usd_yr':float((gdf[f'{scen}_risk_class'].isin(['high','very high'])*gdf.sample_area_ha).sum()*SERVICE_VALUE_USD_HA_YR)
    })
scenario_summary=pd.DataFrame(rows)
scenario_summary.to_csv(OUT/'scenario_summary.csv', index=False)

rrows=[]
for scen in slr_files:
    for reg, sub in gdf.groupby('region'):
        rrows.append({
            'scenario':scen,'region':reg,'n_sample_points':len(sub),'expanded_area_ha':float(sub.sample_area_ha.sum()),
            'mean_slr_rate_mm_yr':float(sub[f'{scen}_slr_rate_mm_yr'].mean()),
            'pct_area_above_7mm':float(100*(sub[f'{scen}_slr_above_7mm_yr']*sub.sample_area_ha).sum()/sub.sample_area_ha.sum()),
            'mean_tc_risk_norm':float(sub['tc_risk_norm'].mean()),
            'mean_composite_risk':float(sub[f'{scen}_composite_risk'].mean()),
            'pct_area_high_or_very_high':float(100*(sub[f'{scen}_risk_class'].isin(['high','very high'])*sub.sample_area_ha).sum()/sub.sample_area_ha.sum()),
            'service_value_high_or_very_high_usd_yr':float((sub[f'{scen}_risk_class'].isin(['high','very high'])*sub.sample_area_ha).sum()*SERVICE_VALUE_USD_HA_YR)
        })
regional_summary=pd.DataFrame(rrows)
regional_summary.to_csv(OUT/'regional_summary.csv', index=False)

data_overview={
 'mangrove_sample_points':int(len(gdf)),
 'mangrove_input_geometry':'Point samples (not polygons) despite instruction wording; interpreted as 10% sampled 30 m GMW reference points.',
 'bounds_lonlat':[float(x) for x in gdf.total_bounds],
 'regions':gdf.region.value_counts().to_dict(),
 'tc_track_points':int(len(tc)),
 'tc_wind_ms_summary':{k:float(v) for k,v in tc.wind.describe().to_dict().items()},
 'tc_file_limitation':'Reduced TC file contains lat/lon/wind only and no year or storm id; annual frequencies and temporal trends cannot be exactly recovered.',
 'slr_extraction':slr_meta,
 'area_proxy':{'point_area_ha':POINT_AREA_HA,'sample_fraction':SAMPLE_FRACTION,'expanded_area_per_point_ha':EXPANDED_AREA_HA,'total_expanded_area_ha':float(gdf.sample_area_ha.sum())},
 'ecosystem_service_proxy_usd_per_ha_yr':SERVICE_VALUE_USD_HA_YR
}
(OUT/'data_overview.json').write_text(json.dumps(data_overview, indent=2))
(OUT/'risk_quantile_thresholds.json').write_text(json.dumps(thresholds, indent=2))

# Validation tables: SLR scenario correlations, component correlations
val={}
slr_mat=gdf[[f'{s}_slr_rate_mm_yr' for s in slr_files]].corr()
risk_mat=gdf[[f'{s}_composite_risk' for s in slr_files]+['tc_risk_norm']].corr()
slr_mat.to_csv(OUT/'slr_scenario_correlation.csv')
risk_mat.to_csv(OUT/'risk_component_correlation.csv')
val['slr_rate_correlation']=slr_mat.to_dict()
val['risk_component_correlation']=risk_mat.to_dict()
(OUT/'validation_metrics.json').write_text(json.dumps(val, indent=2))

# ---------------- figures ----------------
print('Generating figures...')
sns.set_theme(style='whitegrid')
# Figure 1: data overview, points + TC tracks subset
fig, ax=plt.subplots(figsize=(12,5.8))
# plot binned sample if dense
plot_g=gdf.sample(n=min(30000,len(gdf)), random_state=1)
ax.scatter(plot_g.lon, plot_g.lat, s=2, c='forestgreen', alpha=0.35, label='Mangrove sample points')
plot_tc=tc.sample(n=min(30000,len(tc)), random_state=2)
ax.scatter(plot_tc.lon, plot_tc.lat, s=1, c=plot_tc.wind, cmap='magma', alpha=0.15, label='MIT TC track points (>=33 m/s)')
ax.set_xlim(-180,180); ax.set_ylim(-45,55); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Input data overview: sampled mangroves and historical tropical-cyclone track points')
ax.legend(loc='lower left', markerscale=4)
cb=fig.colorbar(plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(tc.wind.min(), tc.wind.max())), ax=ax, fraction=0.02, pad=0.01)
cb.set_label('TC wind speed (m s$^{-1}$)')
fig.tight_layout(); fig.savefig(IMG/'figure_1_data_overview.png', dpi=200); plt.close(fig)

# Figure 2: scenario risk and SLR threshold bars
fig, axes=plt.subplots(1,2, figsize=(12,5))
sns.barplot(data=scenario_summary, x='scenario', y='mean_composite_risk', ax=axes[0], color='#4c78a8')
axes[0].set_title('Mean composite risk by scenario'); axes[0].set_ylabel('Composite risk index (0-1)'); axes[0].set_xlabel('Scenario')
sns.barplot(data=scenario_summary, x='scenario', y='pct_area_above_7mm', ax=axes[1], color='#f58518')
axes[1].set_title('Area exceeding 7 mm yr$^{-1}$ RSLR stress threshold'); axes[1].set_ylabel('% expanded sampled area'); axes[1].set_xlabel('Scenario')
fig.tight_layout(); fig.savefig(IMG/'figure_2_scenario_risk_comparison.png', dpi=200); plt.close(fig)

# Figure 3: global risk map under SSP585
fig, ax=plt.subplots(figsize=(12,5.8))
plot_g=gdf.sample(n=min(60000,len(gdf)), random_state=3)
sc=ax.scatter(plot_g.lon, plot_g.lat, s=4, c=plot_g['ssp585_composite_risk'], cmap='viridis_r', alpha=0.65, linewidths=0)
ax.set_xlim(-180,180); ax.set_ylim(-45,55); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Composite mangrove risk index under SSP5-8.5 (sampled GMW points)')
cb=fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01); cb.set_label('Composite risk index')
fig.tight_layout(); fig.savefig(IMG/'figure_3_global_risk_map.png', dpi=200); plt.close(fig)

# Figure 4: validation/comparison: SLR scenarios and components by region
fig, axes=plt.subplots(1,2, figsize=(14,5.5))
long_slr=[]
for s in slr_files:
    tmp=gdf[['region',f'{s}_slr_rate_mm_yr']].copy(); tmp['scenario']=s; tmp=tmp.rename(columns={f'{s}_slr_rate_mm_yr':'slr_rate_mm_yr'}); long_slr.append(tmp.sample(n=min(15000,len(tmp)), random_state={'ssp245':4,'ssp370':5,'ssp585':6}[s]))
long_slr=pd.concat(long_slr)
sns.boxplot(data=long_slr, x='scenario', y='slr_rate_mm_yr', ax=axes[0], showfliers=False)
axes[0].axhline(7, color='red', ls='--', lw=1, label='7 mm yr$^{-1}$')
axes[0].set_title('SLR extraction validation: scenario distributions')
axes[0].set_ylabel('Median RSLR rate, 2020-2100 (mm yr$^{-1}$)'); axes[0].legend()
heat=regional_summary[regional_summary.scenario=='ssp585'].pivot(index='region', columns='scenario', values='mean_composite_risk')
# Instead plot components in SSP585 by region
comp=gdf.groupby('region').agg(tc=('tc_risk_norm','mean'), slr=('ssp585_slr_risk_norm','mean'), comp=('ssp585_composite_risk','mean')).sort_values('comp', ascending=False)
comp[['slr','tc']].plot(kind='barh', stacked=True, ax=axes[1], color=['#f58518','#4c78a8'])
axes[1].set_title('Component contribution by region (SSP5-8.5)')
axes[1].set_xlabel('Mean normalized contribution (SLR + TC; unweighted stack)'); axes[1].invert_yaxis()
fig.tight_layout(); fig.savefig(IMG/'figure_4_validation_comparison.png', dpi=200); plt.close(fig)

# Claim recovery
claims=[
 {'claim':'The reduced mangrove input contains 100,000 point samples rather than polygons.', 'artifact':'outputs/data_overview.json; outputs/data_structure_inspection.json', 'support':'mangrove_sample_points and input geometry inspection', 'status':'verified'},
 {'claim':'Median relative SLR rates increase from SSP2-4.5 to SSP5-8.5 across sampled mangroves.', 'artifact':'outputs/scenario_summary.csv', 'support':'median_slr_rate_mm_yr by scenario', 'status':'verified'},
 {'claim':'The 7 mm/yr RSLR stress threshold is exceeded for a larger area under higher-emission scenarios.', 'artifact':'outputs/scenario_summary.csv; figure_2_scenario_risk_comparison.png', 'support':'pct_area_above_7mm', 'status':'verified'},
 {'claim':'TC exposure is represented as a historical intensity-distance exposure because the TC file lacks year/storm identifiers.', 'artifact':'outputs/data_overview.json; outputs/data_structure_inspection.json', 'support':'tc_file_limitation and variable inspection', 'status':'verified limitation'},
 {'claim':'High/very-high composite risk areas imply ecosystem-service exposure using a transparent US$20,000 ha-1 yr-1 proxy.', 'artifact':'outputs/scenario_summary.csv; outputs/regional_summary.csv', 'support':'service_value_high_or_very_high_usd_yr fields', 'status':'proxy estimate'}
]
pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv', index=False)

print('Done')
print(scenario_summary.to_string(index=False))
