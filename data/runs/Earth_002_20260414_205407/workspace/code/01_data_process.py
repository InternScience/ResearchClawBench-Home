#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print('Loading mangroves...')
gdf_mang = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
gdf_mang = gdf_mang.to_crs('EPSG:4326')
gdf_mang['lon'] = gdf_mang.geometry.centroid.x
gdf_mang['lat'] = gdf_mang.geometry.centroid.y
gdf_mang['weight'] = 1  # proxy for area since small polygons
mang_df = gdf_mang[['uid', 'lon', 'lat', 'weight']].copy()
mang_df.to_csv('outputs/mangrove_centroids.csv', index=False)
print('Mangroves: %d samples' % len(gdf_mang))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
gdf_mang.plot(ax=ax, markersize=1, color='green', alpha=0.5, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.set_global()
ax.set_title('Mangrove Centroids (10%% sample)')
plt.savefig('report/images/mangrove_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

scenarios = ['ssp245', 'ssp370', 'ssp585']
slr_dfs = {}
for scen in scenarios:
    ds = xr.open_dataset('data/slr/total_%s_medium_confidence_rates.nc' % scen)
    rate_med = ds.sea_level_change_rate.sel(quantiles=0.5, years=slice(2020,2100)).mean(dim='years')
    slr_df = pd.DataFrame({
        'location_id': ds.locations.values,
        'lon': ds.lon.values,
        'lat': ds.lat.values,
        'rate_mmyr': rate_med.values,
        'slr_2100_mm': rate_med.values * 80
    })
    slr_df['scenario'] = scen.upper()
    slr_dfs[scen] = slr_df
    slr_df.to_csv('outputs/slr_%s_medians.csv' % scen, index=False)

slr_all = pd.concat(slr_dfs.values())
print('SLR locations: %d' % len(slr_all.location_id.unique()))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
slr585 = slr_dfs['ssp585']
sc = ax.scatter(slr585.lon, slr585.lat, c=slr585.rate_mmyr, s=1, cmap='Reds', transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.set_global()
ax.set_title('SLR Rates SSP5-8.5 (mm/yr)')
plt.colorbar(sc, ax=ax, label='mm/yr')
plt.savefig('report/images/slr_rates_map.png', dpi=300, bbox_inches='tight')
plt.close()

print('Loading TC...')
ds_tc = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_df = ds_tc.to_dataframe().reset_index()
print('TC points: %d, mean wind %.1f m/s' % (len(tc_df), tc_df.wind.mean()))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
tc_sub = tc_df.sample(n=min(50000, len(tc_df)), random_state=42)
ax.scatter(tc_sub.lon, tc_sub.lat, s=1, alpha=0.6, c='purple', transform=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.set_global()
ax.set_title('Historical TC Tracks')
plt.savefig('report/images/tc_tracks.png', dpi=300, bbox_inches='tight')
plt.close()

print('Matching...')
slr_locs = np.deg2rad(np.column_stack([slr_all.lat, slr_all.lon]))
mang_locs_rad = np.deg2rad(np.column_stack([mang_df.lat, mang_df.lon]))
slr_tree = BallTree(slr_locs, metric='haversine')
dist, idx = slr_tree.query(mang_locs_rad)
slr_matched = slr_all.iloc[idx].copy()
slr_matched['mang_uid'] = mang_df.uid.values
rate_pivot = slr_matched.pivot(index='mang_uid', columns='scenario', values='rate_mmyr').add_prefix('rate_mmyr_')
slr_pivot = slr_matched.pivot(index='mang_uid', columns='scenario', values='slr_2100_mm').add_prefix('slr_2100_mm_')
pivot_df = pd.concat([rate_pivot, slr_pivot], axis=1)
mang_df = mang_df.merge(pivot_df.reset_index(), on='uid')

tc_locs_rad = np.deg2rad(np.column_stack([tc_df.lat, tc_df.lon]))
tc_tree = BallTree(tc_locs_rad, metric='haversine')
tc_counts = tc_tree.query_radius(mang_locs_rad, r=200/6371, count_only=True)
mang_df['tc_count'] = tc_counts
mang_df['tc_freq_yr'] = mang_df['tc_count'] / 165.0

mang_df.to_csv('outputs/mangroves_matched.csv', index=False)
print('Data processing complete!')