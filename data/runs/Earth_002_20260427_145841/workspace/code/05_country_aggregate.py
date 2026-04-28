"""
Step 5: Country attribution + ecosystem services linkage.
- Use the UCSC_CWON_countrybounds.gpkg country polygons to assign each mangrove
  point to a country (spatial join).
- Then aggregate composite risk per country and merge with UCSC mangrove ecosystem
  service variables (Mang_Ha_2020, Risk_Pop_2020, Risk_Stock_2020, Ben_Pop_2020,
  Ben_Stock_2020).
- Save outputs/country_risk_summary.csv.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
df = pd.read_csv(os.path.join(ROOT, 'outputs/mangrove_point_risk.csv'))

cb = gpd.read_file(os.path.join(ROOT, 'data/ecosystem/UCSC_CWON_countrybounds.gpkg'))
print('Country bounds CRS:', cb.crs, 'rows:', len(cb))
print(cb.columns.tolist())

# Build geo points
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')

if cb.crs is None:
    cb.set_crs('EPSG:4326', inplace=True)
elif cb.crs.to_string() != 'EPSG:4326':
    cb = cb.to_crs('EPSG:4326')

print('Spatial join...')
joined = gpd.sjoin(gdf, cb[['Country','ISO3','Mang_Ha_2020','Risk_Pop_2020','Risk_Stock_2020','Ben_Pop_2020','Ben_Stock_2020','geometry']], how='left', predicate='within')
print('Join produced', len(joined), 'rows; with country:', joined.Country.notna().sum())

# Some points fall on slivers / outside polygons -> nearest-country fallback
missing_mask = joined.Country.isna()
if missing_mask.any():
    print('Filling missing country via nearest centroid for', int(missing_mask.sum()), 'points')
    # build centroids in projected CRS for nearest
    cb_proj = cb.to_crs(3857)
    cb_proj['cx'] = cb_proj.geometry.centroid.x
    cb_proj['cy'] = cb_proj.geometry.centroid.y
    from sklearn.neighbors import BallTree
    # Use haversine on country centroids in lat/lon
    cb_ll = cb.copy()
    cb_ll['cx_ll'] = cb.geometry.centroid.x
    cb_ll['cy_ll'] = cb.geometry.centroid.y
    coords = np.column_stack([np.deg2rad(cb_ll.cy_ll.values), np.deg2rad(cb_ll.cx_ll.values)])
    tree = BallTree(coords, metric='haversine')
    miss = joined[missing_mask]
    pts = np.column_stack([np.deg2rad(miss.lat.values), np.deg2rad(miss.lon.values)])
    _, near = tree.query(pts, k=1)
    near = near.flatten()
    for col in ['Country','ISO3','Mang_Ha_2020','Risk_Pop_2020','Risk_Stock_2020','Ben_Pop_2020','Ben_Stock_2020']:
        joined.loc[missing_mask, col] = cb_ll.iloc[near][col].values

print('After fallback, with country:', joined.Country.notna().sum())

# Drop duplicates (sjoin can produce duplicates if a point falls in multi-polygons)
joined = joined.drop_duplicates(subset=['uid'])
print('After dedup:', len(joined))

# Save the per-point with country
joined_out = pd.DataFrame(joined.drop(columns='geometry'))
joined_out.to_csv(os.path.join(ROOT, 'outputs/mangrove_point_risk_with_country.csv'), index=False)

# ----- aggregate per country -----
agg_rows = []
for country, sub in joined_out.groupby('Country', dropna=True):
    row = {
        'Country': country,
        'ISO3': sub['ISO3'].iloc[0] if 'ISO3' in sub else '',
        'n_points': len(sub),
        'mean_tc_risk': sub['tc_risk'].mean(),
        'frac_intense_storm_exposed': float((sub['tc_intense_per_decade'] > 0).mean()),
    }
    for sc in ['ssp245','ssp370','ssp585']:
        row[f'mean_composite_risk_{sc}'] = sub[f'composite_risk_{sc}'].mean()
        row[f'frac_very_high_{sc}'] = float((sub[f'risk_class_{sc}'] == 'very_high').mean())
        row[f'frac_high_{sc}'] = float((sub[f'risk_class_{sc}'] == 'high').mean())
        row[f'frac_at_or_above_high_{sc}'] = float(sub[f'risk_class_{sc}'].isin(['high','very_high']).mean())
        row[f'mean_slr_rate_{sc}_2100'] = sub[f'slr_rate_mm_yr_{sc}_2100'].mean()
    # link ecosystem service values (per-country totals from UCSC)
    row['Mang_Ha_2020'] = sub['Mang_Ha_2020'].iloc[0]
    row['Risk_Pop_2020'] = sub['Risk_Pop_2020'].iloc[0]
    row['Risk_Stock_2020'] = sub['Risk_Stock_2020'].iloc[0]
    row['Ben_Pop_2020'] = sub['Ben_Pop_2020'].iloc[0]
    row['Ben_Stock_2020'] = sub['Ben_Stock_2020'].iloc[0]
    agg_rows.append(row)

country_df = pd.DataFrame(agg_rows).sort_values('mean_composite_risk_ssp585', ascending=False)
country_df.to_csv(os.path.join(ROOT, 'outputs/country_risk_summary.csv'), index=False)
print('Saved outputs/country_risk_summary.csv -- top 15 by SSP5-8.5 risk:')
print(country_df.head(15)[['Country','n_points','mean_composite_risk_ssp245','mean_composite_risk_ssp370','mean_composite_risk_ssp585','frac_at_or_above_high_ssp585','Mang_Ha_2020','Risk_Pop_2020']].to_string(index=False))

# ----- exposed ES summary: weight country ES values by frac_at_or_above_high -----
es_rows = []
for sc in ['ssp245','ssp370','ssp585']:
    col = f'frac_at_or_above_high_{sc}'
    sub = country_df[country_df['n_points'] >= 5].copy()  # require at least 5 pts for stable fraction
    sub['exposed_pop'] = sub['Risk_Pop_2020'] * sub[col]
    sub['exposed_stock'] = sub['Risk_Stock_2020'] * sub[col]
    sub['exposed_mang_ha'] = sub['Mang_Ha_2020'] * sub[col]
    es_rows.append({
        'scenario': sc,
        'global_high_risk_pop_2020': sub['exposed_pop'].sum(),
        'global_high_risk_stock_2020': sub['exposed_stock'].sum(),
        'global_high_risk_mang_ha_2020': sub['exposed_mang_ha'].sum(),
        'global_total_pop_at_risk_2020': sub['Risk_Pop_2020'].sum(),
        'global_total_stock_at_risk_2020': sub['Risk_Stock_2020'].sum(),
        'global_total_mang_ha_2020': sub['Mang_Ha_2020'].sum(),
        'fraction_pop_exposed': sub['exposed_pop'].sum() / max(sub['Risk_Pop_2020'].sum(),1),
        'fraction_stock_exposed': sub['exposed_stock'].sum() / max(sub['Risk_Stock_2020'].sum(),1),
        'fraction_mang_ha_exposed': sub['exposed_mang_ha'].sum() / max(sub['Mang_Ha_2020'].sum(),1),
    })
pd.DataFrame(es_rows).to_csv(os.path.join(ROOT, 'outputs/ecosystem_service_exposure.csv'), index=False)
print('\nEcosystem service exposure summary:')
print(pd.DataFrame(es_rows).to_string(index=False))
