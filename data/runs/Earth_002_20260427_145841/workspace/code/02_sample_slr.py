"""
Step 2: Sample SLR rates per scenario at each mangrove point.
- Use median quantile (0.5) at year 2090 (mid 2080-2100) and at 2100.
- Nearest neighbour from AR6 RSLR locations (with valid lat/lon).
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
mg = pd.read_csv(os.path.join(ROOT,'outputs/mangrove_points.csv'))

scenarios = {
    'ssp245':'data/slr/total_ssp245_medium_confidence_rates.nc',
    'ssp370':'data/slr/total_ssp370_medium_confidence_rates.nc',
    'ssp585':'data/slr/total_ssp585_medium_confidence_rates.nc',
}

YEAR_TARGETS = [2090, 2100]
TARGET_QUANTILE = 0.5  # median

results = mg.copy()
for sc, path in scenarios.items():
    print(f'--- {sc} ---')
    ds = xr.open_dataset(os.path.join(ROOT, path))
    lat = ds['lat'].values
    lon = ds['lon'].values
    # Filter out invalid coordinates
    valid = np.isfinite(lat) & np.isfinite(lon)
    print('valid AR6 locations:', valid.sum(), 'of', len(lat))
    qvals = ds['quantiles'].values
    qidx = int(np.argmin(np.abs(qvals - TARGET_QUANTILE)))
    print('quantile chosen:', qvals[qidx])
    yvals = ds['years'].values
    print('years available:', yvals)
    # Convert AR6 lat/lon (lon may be 0..360) to -180..180
    lon_n = np.where(lon > 180, lon - 360, lon)
    coords = np.column_stack([lat[valid], lon_n[valid]])
    valid_idx = np.where(valid)[0]
    tree = cKDTree(coords)
    pts = np.column_stack([mg.lat.values, mg.lon.values])
    # cKDTree uses Euclidean on lat/lon — fine for small distances; for nearest match it’s OK.
    dist, near = tree.query(pts, k=1)
    nearest_loc_idx = valid_idx[near]
    for yr in YEAR_TARGETS:
        yi = int(np.where(yvals == yr)[0][0])
        rates = ds['sea_level_change_rate'].isel(quantiles=qidx, years=yi).values  # shape (locations,)
        # rate units: mm/yr per AR6 documentation
        rate_at_pts = rates[nearest_loc_idx]
        col = f'slr_rate_mm_yr_{sc}_{yr}'
        results[col] = rate_at_pts
        print(f'  {col}: median {np.nanmedian(rate_at_pts):.3f}  mean {np.nanmean(rate_at_pts):.3f}')
    ds.close()

results.to_csv(os.path.join(ROOT, 'outputs/mangrove_with_slr.csv'), index=False)
print('Saved outputs/mangrove_with_slr.csv with cols:', [c for c in results.columns if c.startswith('slr_')])
