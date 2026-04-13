import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

sns.set_theme(style='whitegrid')

BASE = Path('.')
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

MANGROVE_FP = BASE / 'data' / 'mangroves' / 'gmw_v4_ref_smpls_qad_v12.gpkg'
SLR_FILES = {
    'ssp245': BASE / 'data' / 'slr' / 'total_ssp245_medium_confidence_rates.nc',
    'ssp370': BASE / 'data' / 'slr' / 'total_ssp370_medium_confidence_rates.nc',
    'ssp585': BASE / 'data' / 'data' / 'slr' / 'total_ssp585_medium_confidence_rates.nc',
}
# correct mistaken path if needed
if not SLR_FILES['ssp585'].exists():
    SLR_FILES['ssp585'] = BASE / 'data' / 'slr' / 'total_ssp585_medium_confidence_rates.nc'
TC_FP = BASE / 'data' / 'tc' / 'tracks_mit_mpi-esm1-2-hr_historical_reduced.nc'
COUNTRIES_FP = BASE / 'data' / 'ecosystem' / 'UCSC_CWON_countrybounds.gpkg'


def normalize_lon_180(lon):
    return ((lon + 180) % 360) - 180


def tc_regime_score(event_count, mean_wind):
    # frequency and intensity normalized using robust observed ranges
    f = np.clip(event_count / 8.0, 0, 1)  # ~95th percentile close to 8 in sampled data
    w = np.clip((mean_wind - 33.0) / (70.0 - 33.0), 0, 1)
    return 0.6 * f + 0.4 * w


def composite_risk(tc_score, slr_rate):
    slr = np.clip((slr_rate - 4.0) / (10.0 - 4.0), 0, 1)
    return 0.5 * tc_score + 0.5 * slr


def quantile_label(x):
    if x < 0.25:
        return 'Low'
    if x < 0.5:
        return 'Moderate'
    if x < 0.75:
        return 'High'
    return 'Very High'


def main():
    print('Reading mangrove points...')
    gdf = gpd.read_file(MANGROVE_FP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y

    # area proxy by equal-area buffering around sampled point.
    # assume each sampled point represents a 10x multiple of equal area units because dataset is 10% sample.
    area_crs = 'EPSG:6933'
    gdf_area = gdf.to_crs(area_crs)
    cell_area_km2 = np.pi * (125.0 ** 2) / 1e6  # proxy 25 m pixel-equivalent buffered half-cell; conservative surrogate
    gdf['sample_area_km2'] = cell_area_km2 * 10.0

    print('Reading country polygons...')
    countries = gpd.read_file(COUNTRIES_FP).to_crs(4326)
    name_col = None
    for c in countries.columns:
        if c.lower() in {'name','country','country_na','admin','sovereignt','name_long'}:
            name_col = c
            break
    if name_col is None:
        name_col = countries.columns[0]
    countries = countries[[name_col, 'geometry']].rename(columns={name_col: 'country'})
    gdf = gpd.sjoin(gdf, countries, how='left', predicate='within').drop(columns=['index_right'])
    gdf['country'] = gdf['country'].fillna('Unassigned')

    print('Reading tropical cyclone data...')
    tcds = xr.open_dataset(TC_FP)
    tc = pd.DataFrame({
        'lon': normalize_lon_180(tcds['lon'].values.astype(float)),
        'lat': tcds['lat'].values.astype(float),
        'wind': tcds['wind'].values.astype(float),
    }).dropna()
    tc = tc[(tc['lat'].between(-40, 40))]
    tc['lat_bin'] = np.floor((tc['lat'] + 90) * 2) / 2 - 90
    tc['lon_bin'] = np.floor((tc['lon'] + 180) * 2) / 2 - 180
    tc_grid = tc.groupby(['lat_bin', 'lon_bin']).agg(
        tc_event_count=('wind', 'size'),
        tc_mean_wind=('wind', 'mean'),
        tc_p90_wind=('wind', lambda s: np.quantile(s, 0.9)),
    ).reset_index()

    gdf['lat_bin'] = np.floor((gdf['lat'] + 90) * 2) / 2 - 90
    gdf['lon_bin'] = np.floor((gdf['lon'] + 180) * 2) / 2 - 180
    gdf = gdf.merge(tc_grid, on=['lat_bin', 'lon_bin'], how='left')
    gdf[['tc_event_count', 'tc_mean_wind', 'tc_p90_wind']] = gdf[['tc_event_count', 'tc_mean_wind', 'tc_p90_wind']].fillna(0)
    gdf['tc_score'] = tc_regime_score(gdf['tc_event_count'], gdf['tc_mean_wind'])

    print('Reading sea-level rise data...')
    scenario_cols = []
    slr_location_tables = {}
    for scenario, fp in SLR_FILES.items():
        ds = xr.open_dataset(fp)
        q = ds['quantiles'].values
        q_idx = int(np.argmin(np.abs(q - 0.5)))
        rate = ds['sea_level_change_rate'].isel(quantiles=q_idx).sel(years=2100)
        df = pd.DataFrame({
            'slr_lat': ds['lat'].values.astype(float),
            'slr_lon': normalize_lon_180(ds['lon'].values.astype(float)),
            f'slr_rate_{scenario}': rate.values.astype(float),
        }).dropna()
        slr_location_tables[scenario] = df
        scenario_cols.append(f'slr_rate_{scenario}')

    # nearest-neighbor merge in rounded 0.5 degree bins for scalability
    for scenario, df in slr_location_tables.items():
        df['lat_bin'] = np.round(df['slr_lat'] * 2) / 2
        df['lon_bin'] = np.round(df['slr_lon'] * 2) / 2
        grid = df.groupby(['lat_bin', 'lon_bin'])[f'slr_rate_{scenario}'].mean().reset_index()
        gdf = gdf.merge(grid, on=['lat_bin', 'lon_bin'], how='left')
        if gdf[f'slr_rate_{scenario}'].isna().any():
            global_mean = grid[f'slr_rate_{scenario}'].mean()
            gdf[f'slr_rate_{scenario}'] = gdf[f'slr_rate_{scenario}'].fillna(global_mean)

    for scenario in ['ssp245', 'ssp370', 'ssp585']:
        gdf[f'composite_{scenario}'] = composite_risk(gdf['tc_score'], gdf[f'slr_rate_{scenario}'])
        q1, q2, q3 = gdf[f'composite_{scenario}'].quantile([0.25, 0.5, 0.75]).tolist()
        bins = [-1e9, q1, q2, q3, 1e9]
        labels = ['Low','Moderate','High','Very High']
        if len(set([round(q1,10), round(q2,10), round(q3,10)])) < 3:
            gdf[f'risk_class_{scenario}'] = pd.cut(gdf[f'composite_{scenario}'], bins=[-1e9, 0.25, 0.5, 0.75, 1e9], labels=labels)
        else:
            gdf[f'risk_class_{scenario}'] = pd.cut(gdf[f'composite_{scenario}'], bins=bins, labels=labels, include_lowest=True)

    gdf['dominant_driver_ssp585'] = np.where(
        gdf['tc_score'] >= np.clip((gdf['slr_rate_ssp585'] - 4.0) / 6.0, 0, 1),
        'Tropical cyclone regime',
        'Sea-level rise'
    )

    point_df = pd.DataFrame(gdf.drop(columns='geometry'))
    point_df.to_csv(OUT / 'mangrove_point_risk.csv', index=False)

    country = point_df.groupby('country').agg(
        mangrove_area_km2=('sample_area_km2', 'sum'),
        mean_tc_score=('tc_score', 'mean'),
        mean_tc_wind=('tc_mean_wind', 'mean'),
        tc_points=('tc_event_count', 'sum'),
        mean_slr_ssp245=('slr_rate_ssp245', 'mean'),
        mean_slr_ssp370=('slr_rate_ssp370', 'mean'),
        mean_slr_ssp585=('slr_rate_ssp585', 'mean'),
        risk_ssp245=('composite_ssp245', 'mean'),
        risk_ssp370=('composite_ssp370', 'mean'),
        risk_ssp585=('composite_ssp585', 'mean'),
    ).reset_index()
    country = country.sort_values('risk_ssp585', ascending=False)
    country.to_csv(OUT / 'country_risk_summary.csv', index=False)

    scenario_summary = []
    for scenario in ['ssp245', 'ssp370', 'ssp585']:
        for lbl in ['Low', 'Moderate', 'High', 'Very High']:
            area = point_df.loc[point_df[f'risk_class_{scenario}'] == lbl, 'sample_area_km2'].sum()
            scenario_summary.append({'scenario': scenario, 'risk_class': lbl, 'area_km2': area})
    scen_df = pd.DataFrame(scenario_summary)
    scen_df.to_csv(OUT / 'scenario_area_by_risk_class.csv', index=False)

    top10 = country.head(10)
    top10.to_csv(OUT / 'top10_countries_ssp585.csv', index=False)

    global_summary = {
        'n_mangrove_points': int(len(point_df)),
        'sampled_mangrove_area_km2_proxy': float(point_df['sample_area_km2'].sum()),
        'mean_tc_score': float(point_df['tc_score'].mean()),
        'mean_slr_ssp245_mm_yr_2100': float(point_df['slr_rate_ssp245'].mean()),
        'mean_slr_ssp370_mm_yr_2100': float(point_df['slr_rate_ssp370'].mean()),
        'mean_slr_ssp585_mm_yr_2100': float(point_df['slr_rate_ssp585'].mean()),
        'mean_composite_ssp245': float(point_df['composite_ssp245'].mean()),
        'mean_composite_ssp370': float(point_df['composite_ssp370'].mean()),
        'mean_composite_ssp585': float(point_df['composite_ssp585'].mean()),
        'share_high_or_very_high_ssp585': float((point_df['composite_ssp585'] >= point_df['composite_ssp585'].median()).mean()),
    }
    (OUT / 'global_summary.json').write_text(json.dumps(global_summary, indent=2))

    # Figures
    plt.figure(figsize=(9,5))
    sns.histplot(point_df['tc_mean_wind'], bins=40, color='#355C7D')
    plt.xlabel('Historical tropical cyclone wind at sampled track points (m/s)')
    plt.ylabel('Count of mangrove samples')
    plt.title('Mangrove exposure to historical tropical cyclone intensity')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_tc_intensity_histogram.png', dpi=200)
    plt.close()

    plt.figure(figsize=(9,5))
    for scenario, color in [('ssp245','#1b9e77'),('ssp370','#d95f02'),('ssp585','#7570b3')]:
        sns.kdeplot(point_df[f'slr_rate_{scenario}'], label=scenario.upper(), linewidth=2, color=color)
    plt.xlabel('Relative sea-level rise rate in 2100 (mm/yr, median)')
    plt.ylabel('Density')
    plt.title('Distribution of sea-level rise rates across mangrove samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / 'figure_slr_distribution.png', dpi=200)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for i, scenario in enumerate(['ssp245', 'ssp370', 'ssp585']):
        sns.scatterplot(
            data=point_df.sample(min(20000, len(point_df)), random_state=42),
            x='tc_score', y=f'slr_rate_{scenario}', hue=f'composite_{scenario}',
            palette='viridis', s=8, linewidth=0, ax=ax[i], legend=False
        )
        ax[i].set_title(scenario.upper())
        ax[i].set_xlabel('Cyclone regime score')
        ax[i].set_ylabel('SLR rate 2100 (mm/yr)')
    fig.suptitle('Composite risk is jointly shaped by cyclone regime and sea-level rise', y=1.02)
    plt.tight_layout()
    plt.savefig(IMG / 'figure_bivariate_risk_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()

    top_plot = country.head(15).sort_values('risk_ssp585')
    plt.figure(figsize=(10,7))
    plt.barh(top_plot['country'], top_plot['risk_ssp585'], color=sns.color_palette('Reds', n_colors=len(top_plot)))
    plt.xlabel('Mean composite risk (SSP5-8.5 analogue)')
    plt.ylabel('Country')
    plt.title('Countries with highest mean mangrove risk by 2100')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_top_countries_ssp585.png', dpi=200)
    plt.close()

    area_pivot = scen_df.pivot(index='risk_class', columns='scenario', values='area_km2').reindex(['Low','Moderate','High','Very High'])
    area_pivot.plot(kind='bar', figsize=(9,5), color=['#1b9e77','#d95f02','#7570b3'])
    plt.ylabel('Mangrove area proxy (km²)')
    plt.xlabel('Risk class')
    plt.title('Shift in mangrove area across composite risk classes')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_risk_class_shift.png', dpi=200)
    plt.close()

    # spatial figure
    plot_df = point_df.sample(min(30000, len(point_df)), random_state=1)
    fig, ax = plt.subplots(figsize=(12,6))
    world = countries.dissolve().reset_index(drop=True)
    world.boundary.plot(ax=ax, color='lightgray', linewidth=0.5)
    sc = ax.scatter(plot_df['lon'], plot_df['lat'], c=plot_df['composite_ssp585'], s=2, cmap='magma', alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Global pattern of mangrove composite risk under SSP5-8.5 analogue')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Composite risk')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_global_risk_map_ssp585.png', dpi=200)
    plt.close()

    driver = point_df.groupby('dominant_driver_ssp585')['sample_area_km2'].sum().reset_index()
    plt.figure(figsize=(6,5))
    plt.pie(driver['sample_area_km2'], labels=driver['dominant_driver_ssp585'], autopct='%1.1f%%', colors=['#4C78A8','#F58518'])
    plt.title('Dominant driver of composite risk under SSP5-8.5 analogue')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_driver_share_ssp585.png', dpi=200)
    plt.close()

    print('Done.')
    print(country.head(10).to_string(index=False))
    print(json.dumps(global_summary, indent=2))


if __name__ == '__main__':
    main()
