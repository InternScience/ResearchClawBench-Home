"""
Step 3: TC exposure per mangrove point.
For each mangrove point, count MIT historical track points within 200 km, weighted
by Saffir-Simpson category. Also derive intense-storm count (Cat4-5).

Categories (m/s, 1-min sustained):
  TS   33-43
  Cat1 33-42  (we will use 33-42)
  Cat2 43-49
  Cat3 50-58 (major)
  Cat4 58-70
  Cat5 >=70
We follow Mo et al. 2023 distinction major (Cat3-5) and intense (Cat4-5).
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
mg = pd.read_csv(os.path.join(ROOT, 'outputs/mangrove_with_slr.csv'))

ds = xr.open_dataset(os.path.join(ROOT, 'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc'))
tc_lat = ds['lat'].values
tc_lon = ds['lon'].values
tc_w   = ds['wind'].values
ds.close()
print('TC points:', len(tc_lat))

# Use BallTree haversine in radians.
RAD = np.pi/180.
tc_pts = np.column_stack([tc_lat * RAD, tc_lon * RAD])
mg_pts = np.column_stack([mg.lat.values * RAD, mg.lon.values * RAD])
tree = BallTree(tc_pts, metric='haversine')

R_EARTH_KM = 6371.0
SEARCH_RADIUS_KM = 200.0
r = SEARCH_RADIUS_KM / R_EARTH_KM

print('querying radius...')
ind = tree.query_radius(mg_pts, r=r, return_distance=False)

n = len(mg)
n_total = np.zeros(n, dtype=np.int32)
n_major = np.zeros(n, dtype=np.int32)   # Cat3-5 (>=50 m/s)
n_intense = np.zeros(n, dtype=np.int32) # Cat4-5 (>=58 m/s)
mean_w = np.zeros(n, dtype=np.float32)
max_w  = np.zeros(n, dtype=np.float32)

for i in range(n):
    idxs = ind[i]
    if len(idxs) == 0:
        continue
    w = tc_w[idxs]
    n_total[i] = len(idxs)
    n_major[i] = int((w >= 50).sum())
    n_intense[i] = int((w >= 58).sum())
    mean_w[i] = float(w.mean())
    max_w[i]  = float(w.max())

# The MIT historical record covers 1850-2014 = 165 years. Convert to per-decade rates
# for interpretability. Note this is only a subsample (max_points=200000) but consistent
# across mangrove sites so still a relative measure.
N_YEARS = 165.0
mg['tc_pts_within_200km'] = n_total
mg['tc_major_within_200km'] = n_major
mg['tc_intense_within_200km'] = n_intense
mg['tc_mean_wind_within_200km'] = mean_w
mg['tc_max_wind_within_200km'] = max_w
mg['tc_total_per_decade'] = n_total * 10.0 / N_YEARS
mg['tc_intense_per_decade'] = n_intense * 10.0 / N_YEARS

print('TC summary:')
print(mg[['tc_pts_within_200km','tc_major_within_200km','tc_intense_within_200km',
          'tc_mean_wind_within_200km','tc_max_wind_within_200km',
          'tc_total_per_decade','tc_intense_per_decade']].describe())

mg.to_csv(os.path.join(ROOT, 'outputs/mangrove_with_tc_slr.csv'), index=False)
print('Saved outputs/mangrove_with_tc_slr.csv')
