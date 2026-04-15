import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

ROOT = Path('/home/chenyixin/ResearchClawBench/workspaces/Earth_003_20260414_171511')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def parse_level_name(name: str):
    if name in {'T2M', 'U10', 'V10', 'MSL', 'TP'}:
        return {'channel': name, 'pressure_hpa': None, 'kind': 'surface'}
    prefix = name[0]
    pressure = int(name[1:])
    return {'channel': prefix, 'pressure_hpa': pressure, 'kind': 'upper_air'}


input_ds = xr.open_dataset(DATA_DIR / '20231012-06_input_netcdf.nc')
forecast_ds = xr.open_dataset(DATA_DIR / '006.nc')
input_data = input_ds['data']
forecast_data = forecast_ds['data'].isel(time=0, step=0)
latest_input = input_data.isel(time=1)
previous_input = input_data.isel(time=0)
forecast_minus_latest = forecast_data - latest_input
input_tendency = latest_input - previous_input

levels = [str(x) for x in input_ds['level'].values.tolist()]
lat = input_ds['lat'].values
lon = input_ds['lon'].values

# Dataset summary
summary = {
    'workspace_assessment': {
        'input_resolution_deg': float(lon[1] - lon[0]),
        'lat_points': int(len(lat)),
        'lon_points': int(len(lon)),
        'input_times': [str(pd.to_datetime(t)) for t in input_ds['time'].values],
        'forecast_init_time': str(pd.to_datetime(forecast_ds['time'].values[0])),
        'forecast_step_hours': int(forecast_ds['step'].values[0]),
        'input_shape': [int(x) for x in input_data.shape],
        'forecast_shape': [int(x) for x in forecast_ds['data'].shape],
        'available_levels': levels,
        'note': 'Workspace contains one two-step input sample and one 6-hour forecast sample at 1.0° resolution.'
    }
}
(OUTPUT_DIR / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))

# Per-channel statistics
rows = []
for idx, name in enumerate(levels):
    meta = parse_level_name(name)
    prev = previous_input.isel(level=idx).values
    curr = latest_input.isel(level=idx).values
    fc = forecast_data.isel(level=idx).values
    diff = fc - curr
    tend = curr - prev

    a = curr.reshape(-1)
    b = fc.reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() > 1:
        aa = a[mask] - a[mask].mean()
        bb = b[mask] - b[mask].mean()
        denom = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
        corr = float((aa * bb).sum() / denom) if denom else np.nan
    else:
        corr = np.nan

    rows.append({
        'index': idx,
        'name': name,
        'channel': meta['channel'],
        'kind': meta['kind'],
        'pressure_hpa': meta['pressure_hpa'],
        'input_prev_mean': float(np.nanmean(prev)),
        'input_curr_mean': float(np.nanmean(curr)),
        'forecast_mean': float(np.nanmean(fc)),
        'input_curr_std': float(np.nanstd(curr)),
        'forecast_std': float(np.nanstd(fc)),
        'forecast_minus_latest_mean': float(np.nanmean(diff)),
        'forecast_minus_latest_std': float(np.nanstd(diff)),
        'forecast_minus_latest_mae': float(np.nanmean(np.abs(diff))),
        'input_tendency_std': float(np.nanstd(tend)),
        'forecast_to_latest_corr': corr,
    })

channel_df = pd.DataFrame(rows)
channel_df.to_csv(OUTPUT_DIR / 'channel_statistics.csv', index=False)
channel_df.to_json(OUTPUT_DIR / 'channel_statistics.json', orient='records', indent=2)

agg_df = channel_df.groupby(['channel', 'kind'], dropna=False).agg(
    mean_mae=('forecast_minus_latest_mae', 'mean'),
    mean_diff_std=('forecast_minus_latest_std', 'mean'),
    mean_corr=('forecast_to_latest_corr', 'mean'),
    n_levels=('name', 'count')
).reset_index()
agg_df.to_csv(OUTPUT_DIR / 'channel_group_summary.csv', index=False)

# Representative variables
selected = ['Z500', 'T850', 'U10', 'TP']
selected_idx = [levels.index(x) for x in selected]

# Figure 1: latest input maps
fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
for ax, idx, name in zip(axes.flat, selected_idx, selected):
    field = latest_input.isel(level=idx).values
    im = ax.imshow(field, origin='upper', aspect='auto', extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    ax.set_title(f'Latest input field: {name}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Representative latest-input meteorological fields', fontsize=14)
fig.savefig(IMG_DIR / 'figure_input_maps.png', dpi=160)
plt.close(fig)

# Figure 2: forecast increment maps
fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
for ax, idx, name in zip(axes.flat, selected_idx, selected):
    field = forecast_minus_latest.isel(level=idx).values
    vmax = np.nanpercentile(np.abs(field), 99)
    im = ax.imshow(field, origin='upper', aspect='auto', extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(f'6h forecast minus latest input: {name}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Forecast perturbations relative to the latest input state', fontsize=14)
fig.savefig(IMG_DIR / 'figure_forecast_increment_maps.png', dpi=160)
plt.close(fig)

# Figure 3: zonal means for representative variables
fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
for ax, idx, name in zip(axes.flat, selected_idx, selected):
    curr_zonal = latest_input.isel(level=idx).mean('lon').values
    fc_zonal = forecast_data.isel(level=idx).mean('lon').values
    prev_zonal = previous_input.isel(level=idx).mean('lon').values
    ax.plot(lat, prev_zonal, label='previous input', linewidth=1.5)
    ax.plot(lat, curr_zonal, label='latest input', linewidth=1.5)
    ax.plot(lat, fc_zonal, label='forecast +6h', linewidth=1.5)
    ax.set_title(f'Zonal mean latitude profile: {name}')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
fig.suptitle('Zonal-mean evolution across the two input states and forecast sample', fontsize=14)
fig.savefig(IMG_DIR / 'figure_zonal_profiles.png', dpi=160)
plt.close(fig)

# Figure 4: per-channel diagnostics
plot_df = channel_df.copy()
plot_df['label'] = plot_df['name']
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
sns.barplot(data=agg_df, x='channel', y='mean_mae', hue='kind', ax=axes[0])
axes[0].set_title('Mean absolute forecast-minus-latest difference by variable family')
axes[0].set_xlabel('Variable family')
axes[0].set_ylabel('MAE')

sns.scatterplot(data=plot_df, x='forecast_minus_latest_std', y='forecast_to_latest_corr', hue='channel', style='kind', ax=axes[1], s=55)
axes[1].set_title('Per-channel spatial variability change vs. correlation')
axes[1].set_xlabel('Std of forecast minus latest input')
axes[1].set_ylabel('Spatial correlation with latest input')
axes[1].axhline(0, color='black', linewidth=0.8)
fig.savefig(IMG_DIR / 'figure_channel_diagnostics.png', dpi=160)
plt.close(fig)

# Claim recovery table
claim_rows = [
    {
        'claim': 'The workspace provides only a sample-based analysis setting, not a full 15-day evaluation corpus.',
        'status': 'verified',
        'evidence_artifact': 'outputs/dataset_summary.json',
        'notes': 'Forecast file contains one 6-hour step at 1.0° resolution.'
    },
    {
        'claim': 'Forecast perturbations relative to the latest input are large for most upper-air channels.',
        'status': 'verified',
        'evidence_artifact': 'outputs/channel_statistics.csv',
        'notes': 'Upper-air forecast-minus-latest std is ~14 for most normalized channels.'
    },
    {
        'claim': 'Direct comparison to ECMWF ensemble mean is not possible from workspace data alone.',
        'status': 'verified',
        'evidence_artifact': 'memory.md',
        'notes': 'No ECMWF reference fields exist in workspace.'
    },
    {
        'claim': 'The intended scientific framing is a cascade U-Transformer system to reduce error accumulation.',
        'status': 'supported_by_related_work_and_task',
        'evidence_artifact': 'outputs/method_contract.json; outputs/method_fidelity_checklist.json',
        'notes': 'The method is specified by task; full reproduction assets are absent.'
    }
]
pd.DataFrame(claim_rows).to_csv(OUTPUT_DIR / 'claim_recovery_table.csv', index=False)

# Artifact inventory status refresh
artifact_inventory = {
    'artifacts': [
        {'name': 'input_data_summary', 'type': 'json/csv', 'status': 'satisfied', 'path': 'outputs/dataset_summary.json'},
        {'name': 'forecast_data_summary', 'type': 'json/csv', 'status': 'satisfied', 'path': 'outputs/channel_statistics.csv'},
        {'name': 'leadtime_variability_table', 'type': 'csv/json', 'status': 'partial', 'path': 'outputs/channel_group_summary.csv', 'reason': 'Only a single +6h forecast step is available, so full lead-time sweep is impossible.'},
        {'name': 'forecast_global_maps', 'type': 'png', 'status': 'satisfied', 'path': 'report/images/figure_forecast_increment_maps.png'},
        {'name': 'forecast_latlon_or_zonal_diagnostics', 'type': 'png', 'status': 'satisfied', 'path': 'report/images/figure_zonal_profiles.png'},
        {'name': 'forecast_channel_evolution_plot', 'type': 'png', 'status': 'partial', 'path': 'report/images/figure_channel_diagnostics.png', 'reason': 'Evolution across long lead times cannot be shown with only one forecast step.'},
        {'name': 'report', 'type': 'markdown', 'status': 'planned'}
    ]
}
(OUTPUT_DIR / 'target_artifact_inventory.json').write_text(json.dumps(artifact_inventory, indent=2))

print('Wrote analysis outputs to', OUTPUT_DIR)
print('Wrote figures to', IMG_DIR)
