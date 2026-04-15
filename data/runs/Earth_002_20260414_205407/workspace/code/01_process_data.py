#!/usr/bin/env python3
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.neighbors import BallTree
from geopy.distance import geodesic
import os

# Save overview plots
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Mangroves
print('Loading mangroves...')
gdf = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
gdf['centroid'] = gdf.geometry.centroid
gdf['lon'] = gdf.centroid.x
gdf['lat'] = gdf.centroid.y
gdf['area_deg'] = gdf.geometry.area  # geographic
# Project to compute area properly
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf_utm = gdf.estimate_utm_crs()
gdf_proj = gdf.to_crs(gdf_utm)
gdf['area_ha'] = gdf_proj.geometry.area / 1e4  # ha
mang_centroids = gdf[['uid', 'lon', 'lat', 'area_ha']].copy()
mang_centroids.to_csv('outputs/mangrove_centroids.csv', index=False)
print(f'Mangroves: {len(gdf)} polygons, total area {gdf[\"area_ha\"].sum():.1f} ha')

# Plot mangrove distribution
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
gdf.plot(ax=ax, facecolor='green', alpha=0.6, edgecolor='none', linewidth=0.1)
ax.coastlines()
ax.set_global()
plt.savefig('report/images/mangrove_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. SLR data for all scenarios
scenarios = ['ssp245', 'ssp370', 'ssp585']
slr_data = {}
for scen in scenarios:
    ds = xr.open_dataset(f'data/slr/total_{scen}_medium_confidence_rates.nc')
    rate_median = ds.sea_level_change_rate.sel(years=slice(2020,2100), quantiles=0.5).mean('years')
    slr_df = pd.DataFrame({
        'location': ds.locations.values,
        'lon': ds.lon.isel(quantiles=50, years=slice(0,9)).mean('years').values,  # median lon
        'lat': ds.lat.isel(quantiles=50, years=slice(0,9)).mean('years').values,
        'rate_mmyr': rate_median.values,
        'slr_2100_mm': rate_median.values * 80  # 2020-2100
    })
    slr_df['scenario'] = scen.upper()
    slr_data[scen] = slr_df
    slr_df.to_csv(f'outputs/slr_{scen}_medians.csv', index=False)

slr_all = pd.concat(slr_data.values())
print(f'SLR locations: {len(slr_all)}')

# Plot SLR rates map (median across scenarios)
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
slr_mean = slr_all.groupby(['lon','lat'])['rate_mmyr'].mean().reset_index()
slr_gdf = gpd.GeoDataFrame(slr_mean, geometry=gpd.points_from_xy(slr_mean.lon, slr_mean.lat))
slr_gdf.boundary.plot(ax=ax, color='blue', markersize=1, alpha=0.5)
ax.coastlines()
ax.set_global()
plt.title('SLR Median Rates mm/yr (2020-2100)')
plt.savefig('report/images/slr_rates_map.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. TC tracks
ds_tc = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_df = ds_tc[['lat','lon','wind']].to_dataframe().reset_index()
print(f'TC points: {len(tc_df)}, wind mean {tc_df.wind.mean():.1f} m/s')

# Plot TC density
fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.scatter(tc_df.lon, tc_df.lat, s=1, alpha=0.5, c='red')
ax.coastlines()
ax.set_global()
plt.title('Historical TC Tracks (points >=33 m/s)')
plt.savefig('report/images/tc_tracks.png', dpi=300, bbox_inches='tight')
plt.close()

# Compute TC exposure: for efficiency, KDTree for counts within 200km
print('Computing TC exposure...')
tc_locs = np.deg2rad(np.column_stack((tc_df.lat, tc_df.lon)))
mang_locs = np.deg2rad(np.column_stack((mang_centroids.lat, mang_centroids.lon)))
tree = BallTree(tc_locs, metric='haversine', radius=200/6371)  # 200km in radians
counts = tree.query_radius(mang_locs, r=200/6371, count_only=True)
mang_centroids['tc_exposure'] = counts  # raw count of points within 200km
mang_centroids['tc_freq_per_year'] = counts / 165  # approx 1850-2014
mang_centroids.to_csv('outputs/mangrove_tc_exposure.csv', index=False)

print('Data processing complete. Check outputs/')

