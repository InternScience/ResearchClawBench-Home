import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid')


def robust_minmax(x, qlow=0.05, qhigh=0.95):
    a = np.asarray(x, dtype=float)
    lo = np.nanquantile(a, qlow)
    hi = np.nanquantile(a, qhigh)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(a)
        hi = np.nanmax(a)
    return lo, hi


def scale01(x, lo, hi):
    a = np.asarray(x, dtype=float)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def load_slr_values(path):
    ds = xr.open_dataset(path)
    arr = ds['sea_level_change_rate'].sel(quantiles=0.5, years=2100)
    df = pd.DataFrame({
        'lat': ds['lat'].values.astype(float),
        'lon': ds['lon'].values.astype(float),
        'slr_rate_mm_yr': arr.values.astype(float),
    })
    ds.close()
    return df


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    IMG.mkdir(exist_ok=True, parents=True)

    # Load mangrove sample points and country polygons
    mang = gpd.read_file(DATA / 'mangroves' / 'gmw_v4_ref_smpls_qad_v12.gpkg')
    countries = gpd.read_file(DATA / 'ecosystem' / 'UCSC_CWON_countrybounds.gpkg')[['ISO3', 'Country', 'Mang_Ha_2020', 'geometry']]
    mang = mang[['uid', 'geometry']].copy()
    mang = gpd.sjoin(mang, countries, how='left', predicate='within').drop(columns=['index_right'])

    # Fallback nearest-country join for unmatched points
    unmatched = mang['ISO3'].isna()
    if unmatched.any():
        nearest = gpd.sjoin_nearest(mang.loc[unmatched, ['uid', 'geometry']], countries, how='left', distance_col='dist_deg')
        nearest = nearest[['uid', 'ISO3', 'Country', 'Mang_Ha_2020']].drop_duplicates('uid')
        mang = mang.merge(nearest, on='uid', how='left', suffixes=('', '_near'))
        for col in ['ISO3', 'Country', 'Mang_Ha_2020']:
            mang[col] = mang[col].fillna(mang[f'{col}_near'])
            mang = mang.drop(columns=[f'{col}_near'])

    # Country-calibrated area weights in hectares per sampled point
    counts = mang.groupby('ISO3').size().rename('sample_count')
    countries2 = countries[['ISO3', 'Country', 'Mang_Ha_2020']].drop_duplicates().merge(counts, on='ISO3', how='left')
    countries2['sample_count'] = countries2['sample_count'].fillna(0)
    countries2['ha_per_point'] = np.where(countries2['sample_count'] > 0, countries2['Mang_Ha_2020'] / countries2['sample_count'], np.nan)
    mang = mang.merge(countries2[['ISO3', 'ha_per_point']], on='ISO3', how='left')
    mang['ha_per_point'] = mang['ha_per_point'].fillna(mang['ha_per_point'].median())
    mang['area_ha_est'] = mang['ha_per_point']
    mang['lon'] = mang.geometry.x.astype(float)
    mang['lat'] = mang.geometry.y.astype(float)

    # Sea-level rise extraction by nearest AR6 projection point
    scenario_files = {
        'SSP245': DATA / 'slr' / 'total_ssp245_medium_confidence_rates.nc',
        'SSP370': DATA / 'slr' / 'total_ssp370_medium_confidence_rates.nc',
        'SSP585': DATA / 'slr' / 'total_ssp585_medium_confidence_rates.nc',
    }
    mang_df = pd.DataFrame(mang.drop(columns='geometry'))
    mang_xy = np.deg2rad(np.c_[mang_df['lat'].values, mang_df['lon'].values])

    slr_summary = []
    for scen, path in scenario_files.items():
        slr_df = load_slr_values(path)
        tree = cKDTree(np.deg2rad(np.c_[slr_df['lat'].values, slr_df['lon'].values]))
        dist, idx = tree.query(mang_xy, k=1)
        mang_df[f'slr_{scen.lower()}_mm_yr'] = slr_df['slr_rate_mm_yr'].values[idx]
        slr_summary.append({
            'scenario': scen,
            'min_mm_yr': float(np.nanmin(mang_df[f'slr_{scen.lower()}_mm_yr'])),
            'median_mm_yr': float(np.nanmedian(mang_df[f'slr_{scen.lower()}_mm_yr'])),
            'mean_mm_yr': float(np.nanmean(mang_df[f'slr_{scen.lower()}_mm_yr'])),
            'max_mm_yr': float(np.nanmax(mang_df[f'slr_{scen.lower()}_mm_yr'])),
            'share_area_above_4mm_yr': float(np.average(mang_df[f'slr_{scen.lower()}_mm_yr'] >= 4, weights=mang_df['area_ha_est'])),
            'share_area_above_7mm_yr': float(np.average(mang_df[f'slr_{scen.lower()}_mm_yr'] >= 7, weights=mang_df['area_ha_est'])),
        })

    # Tropical cyclone regime-pressure proxy from historical major/intense track density
    ds_tc = xr.open_dataset(DATA / 'tc' / 'tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
    tc_lat = ds_tc['lat'].values.astype(float)
    tc_lon = ((ds_tc['lon'].values.astype(float) + 180) % 360) - 180
    tc_wind = ds_tc['wind'].values.astype(float)
    ds_tc.close()

    lat_bins = np.arange(-90, 91, 1.0)
    lon_bins = np.arange(-180, 181, 1.0)
    cat3 = tc_wind >= 50.0
    cat4 = tc_wind >= 58.0
    count_cat3, _, _, _ = binned_statistic_2d(tc_lat[cat3], tc_lon[cat3], None, statistic='count', bins=[lat_bins, lon_bins])
    count_cat4, _, _, _ = binned_statistic_2d(tc_lat[cat4], tc_lon[cat4], None, statistic='count', bins=[lat_bins, lon_bins])
    centers_lat = (lat_bins[:-1] + lat_bins[1:]) / 2
    centers_lon = (lon_bins[:-1] + lon_bins[1:]) / 2
    ii = np.clip(np.digitize(mang_df['lat'], lat_bins) - 1, 0, len(centers_lat) - 1)
    jj = np.clip(np.digitize(mang_df['lon'], lon_bins) - 1, 0, len(centers_lon) - 1)
    mang_df['tc_cat3_track_count'] = np.nan_to_num(count_cat3[ii, jj], nan=0.0)
    mang_df['tc_cat4_track_count'] = np.nan_to_num(count_cat4[ii, jj], nan=0.0)
    mang_df['tc_cat4_share'] = mang_df['tc_cat4_track_count'] / np.maximum(mang_df['tc_cat3_track_count'], 1.0)
    # historical regime pressure proxy: density of cat3+ plus extra weight on intense storms
    mang_df['tc_regime_raw'] = np.log1p(mang_df['tc_cat3_track_count']) + 1.5 * np.log1p(mang_df['tc_cat4_track_count'])

    # Normalize components
    tc_lo, tc_hi = robust_minmax(mang_df['tc_regime_raw'])
    mang_df['tc_regime_norm'] = scale01(mang_df['tc_regime_raw'], tc_lo, tc_hi)
    for scen in scenario_files:
        col = f'slr_{scen.lower()}_mm_yr'
        lo, hi = robust_minmax(mang_df[col])
        mang_df[f'{col}_norm'] = scale01(mang_df[col], lo, hi)
        mang_df[f'composite_{scen.lower()}'] = 0.5 * mang_df['tc_regime_norm'] + 0.5 * mang_df[f'{col}_norm']
        mang_df[f'risk_class_{scen.lower()}'] = pd.qcut(mang_df[f'composite_{scen.lower()}'], q=5, labels=['Very low', 'Low', 'Moderate', 'High', 'Very high'])

    # Export core tables
    mang_df.to_csv(OUT / 'mangrove_point_risk_sample.csv', index=False)
    countries2.to_csv(OUT / 'country_area_weight_calibration.csv', index=False)
    pd.DataFrame(slr_summary).to_csv(OUT / 'slr_scenario_summary.csv', index=False)

    # Scenario summaries by risk class
    class_summaries = []
    for scen in scenario_files:
        tmp = mang_df.groupby(f'risk_class_{scen.lower()}', observed=False)['area_ha_est'].sum().reset_index()
        tmp.columns = ['risk_class', 'area_ha']
        tmp['share_global_area'] = tmp['area_ha'] / tmp['area_ha'].sum()
        tmp['scenario'] = scen
        class_summaries.append(tmp)
    risk_class_summary = pd.concat(class_summaries, ignore_index=True)
    risk_class_summary.to_csv(OUT / 'risk_class_area_summary.csv', index=False)

    # Country hotspot summary
    hotspot_rows = []
    for scen in scenario_files:
        grp = mang_df.groupby(['ISO3', 'Country'], dropna=False).agg(
            mangrove_area_ha=('area_ha_est', 'sum'),
            mean_composite=(f'composite_{scen.lower()}', 'mean'),
            mean_slr_mm_yr=(f'slr_{scen.lower()}_mm_yr', 'mean'),
            mean_tc_norm=('tc_regime_norm', 'mean'),
            very_high_area_ha=('area_ha_est', lambda s, scen=scen: s[mang_df.loc[s.index, f'risk_class_{scen.lower()}'] == 'Very high'].sum()),
            high_or_very_high_area_ha=('area_ha_est', lambda s, scen=scen: s[mang_df.loc[s.index, f'risk_class_{scen.lower()}'].isin(['High', 'Very high'])].sum())
        ).reset_index()
        grp['scenario'] = scen
        grp['share_high_or_very_high'] = grp['high_or_very_high_area_ha'] / grp['mangrove_area_ha']
        hotspot_rows.append(grp.sort_values(['high_or_very_high_area_ha', 'mean_composite'], ascending=False))
    hotspots = pd.concat(hotspot_rows, ignore_index=True)
    hotspots.to_csv(OUT / 'country_hotspot_summary.csv', index=False)
    hotspots.groupby('scenario').head(20).to_csv(OUT / 'country_hotspot_top20_by_scenario.csv', index=False)

    # Direct answer table for primary target quantity
    direct_rows = []
    for scen in scenario_files:
        area_total = mang_df['area_ha_est'].sum()
        hv = mang_df.loc[mang_df[f'risk_class_{scen.lower()}'].isin(['High', 'Very high']), 'area_ha_est'].sum()
        vh = mang_df.loc[mang_df[f'risk_class_{scen.lower()}'] == 'Very high', 'area_ha_est'].sum()
        direct_rows.append({
            'scenario': scen,
            'global_mangrove_area_ha_est': area_total,
            'high_or_very_high_risk_area_ha': hv,
            'share_high_or_very_high_risk': hv / area_total,
            'very_high_risk_area_ha': vh,
            'share_very_high_risk': vh / area_total,
            'area_above_4mm_yr_ha': mang_df.loc[mang_df[f'slr_{scen.lower()}_mm_yr'] >= 4, 'area_ha_est'].sum(),
            'share_area_above_4mm_yr': mang_df.loc[mang_df[f'slr_{scen.lower()}_mm_yr'] >= 4, 'area_ha_est'].sum() / area_total,
            'area_above_7mm_yr_ha': mang_df.loc[mang_df[f'slr_{scen.lower()}_mm_yr'] >= 7, 'area_ha_est'].sum(),
            'share_area_above_7mm_yr': mang_df.loc[mang_df[f'slr_{scen.lower()}_mm_yr'] >= 7, 'area_ha_est'].sum() / area_total,
        })
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(OUT / 'direct_constraint_results.csv', index=False)

    # Claim recovery table
    claims = [
        {
            'claim_id': 'C1',
            'claim': 'Higher sea-level-rise scenarios increase mangrove exposure to hazardous RSLR rates and composite risk.',
            'artifact': 'outputs/direct_constraint_results.csv; outputs/slr_scenario_summary.csv; report/images/scenario_risk_distribution.png'
        },
        {
            'claim_id': 'C2',
            'claim': 'Composite risk hotspots are geographically concentrated rather than globally uniform.',
            'artifact': 'report/images/global_composite_risk_map_ssp585.png; outputs/country_hotspot_top20_by_scenario.csv'
        },
        {
            'claim_id': 'C3',
            'claim': 'Historical intense tropical cyclone regime pressure strongly shapes the upper tail of the composite index.',
            'artifact': 'report/images/component_relationships.png; outputs/mangrove_point_risk_sample.csv'
        },
        {
            'claim_id': 'C4',
            'claim': 'A substantial fraction of sampled global mangrove area exceeds 4 mm/yr and 7 mm/yr RSLR under late-century higher-emissions scenarios.',
            'artifact': 'outputs/direct_constraint_results.csv; outputs/slr_scenario_summary.csv'
        }
    ]
    pd.DataFrame(claims).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    # Figures
    # 1. Global map for SSP585
    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw={'projection': None})
    ax.scatter(mang_df['lon'], mang_df['lat'], c=mang_df['composite_ssp585'], s=2, cmap='viridis', alpha=0.7, linewidths=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Global mangrove composite risk (SSP5-8.5, end-century)')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Composite risk index (0-1)')
    fig.tight_layout()
    fig.savefig(IMG / 'global_composite_risk_map_ssp585.png', dpi=220)
    plt.close(fig)

    # 2. Scenario risk distribution
    long = []
    for scen in scenario_files:
        tmp = mang_df[[f'composite_{scen.lower()}', 'area_ha_est']].copy()
        tmp.columns = ['composite', 'area_ha_est']
        tmp['scenario'] = scen
        long.append(tmp)
    long = pd.concat(long, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.violinplot(data=long, x='scenario', y='composite', inner='quartile', cut=0, ax=axes[0])
    axes[0].set_title('Distribution of composite risk by scenario')
    axes[0].set_ylabel('Composite risk index')
    rcs = risk_class_summary.copy()
    order = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
    sns.barplot(data=rcs, x='scenario', y='share_global_area', hue='risk_class', hue_order=order, ax=axes[1])
    axes[1].set_title('Global mangrove area share by risk class')
    axes[1].set_ylabel('Share of estimated mangrove area')
    axes[1].legend(title='Risk class', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(IMG / 'scenario_risk_distribution.png', dpi=220)
    plt.close(fig)

    # 3. Component relationships / validation figure
    sample = mang_df.sample(min(20000, len(mang_df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(sample['tc_regime_norm'], sample['slr_ssp585_mm_yr'], c=sample['composite_ssp585'], s=5, alpha=0.35, cmap='viridis', linewidths=0)
    axes[0].axhline(4, color='orange', linestyle='--', linewidth=1, label='4 mm/yr reference')
    axes[0].axhline(7, color='red', linestyle='--', linewidth=1, label='7 mm/yr reference')
    axes[0].set_xlabel('TC regime pressure (normalized)')
    axes[0].set_ylabel('SLR rate in 2100 (mm/yr, SSP5-8.5 median)')
    axes[0].set_title('Component relationship at mangrove sample points')
    axes[0].legend(loc='upper left')
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04, label='Composite risk (SSP5-8.5)')

    slr_long = pd.concat([
        pd.DataFrame({'scenario': scen, 'slr_mm_yr': mang_df[f'slr_{scen.lower()}_mm_yr'], 'area_ha_est': mang_df['area_ha_est']})
        for scen in scenario_files
    ], ignore_index=True)
    sns.ecdfplot(data=slr_long, x='slr_mm_yr', hue='scenario', ax=axes[1])
    axes[1].axvline(4, color='orange', linestyle='--', linewidth=1)
    axes[1].axvline(7, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('SLR distribution across mangrove sample points')
    axes[1].set_xlabel('SLR rate in 2100 (mm/yr)')
    axes[1].set_ylabel('Cumulative share of points')
    fig.tight_layout()
    fig.savefig(IMG / 'component_relationships.png', dpi=220)
    plt.close(fig)

    # 4. Top hotspot countries
    top585 = hotspots[hotspots['scenario'] == 'SSP585'].sort_values('high_or_very_high_area_ha', ascending=False).head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top585, y='Country', x='high_or_very_high_area_ha', color='teal', ax=ax)
    ax.set_title('Top countries by high/very high mangrove risk area (SSP5-8.5)')
    ax.set_xlabel('Estimated mangrove area in high or very high risk (ha)')
    ax.set_ylabel('Country')
    fig.tight_layout()
    fig.savefig(IMG / 'top_country_hotspots_ssp585.png', dpi=220)
    plt.close(fig)

    # Final metadata summary
    summary = {
        'n_mangrove_points': int(len(mang_df)),
        'n_countries': int(countries2['ISO3'].nunique()),
        'total_estimated_mangrove_area_ha': float(mang_df['area_ha_est'].sum()),
        'tc_proxy_description': 'log1p(cat3 track count) + 1.5*log1p(cat4 track count) within 1-degree grid cell',
        'figures': [
            'report/images/global_composite_risk_map_ssp585.png',
            'report/images/scenario_risk_distribution.png',
            'report/images/component_relationships.png',
            'report/images/top_country_hotspots_ssp585.png'
        ]
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
