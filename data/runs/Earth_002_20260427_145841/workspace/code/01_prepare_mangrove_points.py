"""
Step 1: Load all data and produce the working mangrove sample dataset.
- Read 100k GMW sample points and downsample further to 30000 random for global coverage.
- Save outputs/mangrove_points.parquet (or csv).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)

print('Loading mangrove sample points...')
mg = gpd.read_file(os.path.join(ROOT, 'data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg'))
print(f'Total mangrove rows: {len(mg)}')
print('classes (ref_cls)', mg['ref_cls'].value_counts().to_dict())

# Per dataset description, ref_cls = 1 indicates mangrove (vs 0 non-mangrove).
mg = mg[mg['ref_cls'] == 1].copy()
print(f'After filtering ref_cls=1: {len(mg)}')

mg['lon'] = mg.geometry.x
mg['lat'] = mg.geometry.y

# Downsample for tractable nearest-neighbour matrices.
TARGET_N = 20000
if len(mg) > TARGET_N:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(mg), size=TARGET_N, replace=False)
    mg = mg.iloc[idx].reset_index(drop=True)
print(f'Downsampled mangrove points: {len(mg)}')

# Save lightweight csv
out_df = mg[['uid','lat','lon','ref_cls','gmw_v4_qa']].copy()
out_df.to_csv(os.path.join(ROOT, 'outputs/mangrove_points.csv'), index=False)
print('Saved outputs/mangrove_points.csv')

# Print bbox
print('Mangrove bbox lat:', out_df.lat.min(), out_df.lat.max())
print('Mangrove bbox lon:', out_df.lon.min(), out_df.lon.max())
