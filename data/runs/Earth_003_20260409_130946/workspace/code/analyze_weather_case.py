import json, os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('.')
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid')

input_fp = ROOT / 'data' / '20231012-06_input_netcdf.nc'
fcst_fp = ROOT / 'data' / '006.nc'

inp = xr.open_dataset(input_fp)
fcst = xr.open_dataset(fcst_fp)
levels = [str(x) for x in inp.level.values]
lats = inp.lat.values
lons = inp.lon.values
wlat = np.cos(np.deg2rad(lats))

X0 = inp['data'].isel(time=0)
X1 = inp['data'].isel(time=1)
Y6 = fcst['data'].isel(time=0, step=0)

# Persist key diagnostics
metrics = {}
metrics['input_shape'] = list(inp['data'].shape)
metrics['forecast_shape'] = list(fcst['data'].shape)
metrics['n_levels'] = len(levels)
metrics['forecast_step_hours'] = int(fcst['step'].values[0])

# Baseline comparisons at +6h
err_persist = (Y6 - X1)
err_lin = (Y6 - (2*X1 - X0))

rmse_persist = np.sqrt((err_persist**2).mean(dim=('lat','lon'))).values
rmse_lin = np.sqrt((err_lin**2).mean(dim=('lat','lon'))).values
mae_persist = np.abs(err_persist).mean(dim=('lat','lon')).values
mae_lin = np.abs(err_lin).mean(dim=('lat','lon')).values

# latitude weighted rmse
W = xr.DataArray(wlat / wlat.mean(), dims=['lat'], coords={'lat': inp.lat})
def lat_weighted_rmse(da):
    mse = ((da**2) * W).mean(dim=('lat','lon'))
    return np.sqrt(mse).values

metrics['weighted_rmse_persistence_mean'] = float(np.mean(lat_weighted_rmse(err_persist)))
metrics['weighted_rmse_linear_mean'] = float(np.mean(lat_weighted_rmse(err_lin)))
metrics['rmse_skill_vs_persistence_mean'] = float(1 - np.mean(rmse_lin / (rmse_persist + 1e-12)))

# Variable group diagnostics
var_groups = {
    'Z': [i for i,s in enumerate(levels) if s.startswith('Z')],
    'T': [i for i,s in enumerate(levels) if s.startswith('T') and s != 'T2M'],
    'U': [i for i,s in enumerate(levels) if s.startswith('U') and s != 'U10'],
    'V': [i for i,s in enumerate(levels) if s.startswith('V') and s != 'V10'],
    'R': [i for i,s in enumerate(levels) if s.startswith('R')],
    'Surface': [i for i,s in enumerate(levels) if s in ['T2M','U10','V10','MSL','TP']],
}
metrics['group_rmse'] = {}
for g, idx in var_groups.items():
    e = err_persist.isel(level=idx)
    metrics['group_rmse'][g] = float(np.sqrt((e**2).mean()).values)

# Channel summary table
rows = []
for i, lev in enumerate(levels):
    rows.append({
        'level': lev,
        'rmse_persistence': float(rmse_persist[i]),
        'rmse_linear': float(rmse_lin[i]),
        'mae_persistence': float(mae_persist[i]),
        'mae_linear': float(mae_lin[i]),
        'skill_linear_vs_persistence': float(1 - rmse_lin[i]/(rmse_persist[i] + 1e-12)),
        'x0_mean': float(X0.isel(level=i).mean().values),
        'x1_mean': float(X1.isel(level=i).mean().values),
        'y6_mean': float(Y6.isel(level=i).mean().values),
    })

import pandas as pd
channel_df = pd.DataFrame(rows)
channel_df.to_csv(OUT/'channel_metrics.csv', index=False)

# Figure 1: channel RMSE comparison
fig, ax = plt.subplots(figsize=(16,6))
ax.plot(channel_df['level'], channel_df['rmse_persistence'], label='FuXi vs persistence', marker='o', ms=3)
ax.plot(channel_df['level'], channel_df['rmse_linear'], label='FuXi vs linear extrapolation', marker='o', ms=3)
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('RMSE (normalized units)')
ax.set_title('6-hour forecast error by channel')
ax.legend()
fig.tight_layout()
fig.savefig(IMG/'channel_rmse.png', dpi=180)
plt.close(fig)

# Figure 2: skill distribution
fig, ax = plt.subplots(figsize=(16,6))
colors = ['tab:green' if s > 0 else 'tab:red' for s in channel_df['skill_linear_vs_persistence']]
ax.bar(channel_df['level'], channel_df['skill_linear_vs_persistence'], color=colors)
ax.axhline(0, color='k', lw=1)
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('Skill of linear baseline relative to persistence')
ax.set_title('Does 6-hour tendency help beyond persistence?')
fig.tight_layout()
fig.savefig(IMG/'skill_vs_persistence.png', dpi=180)
plt.close(fig)

# spatial maps for representative channels
for lev in ['Z500','T850','U500','V500','R500','T2M','MSL','TP']:
    if lev not in levels:
        continue
    i = levels.index(lev)
    fig, axes = plt.subplots(1,4, figsize=(18,4), constrained_layout=True)
    fields = [X1.isel(level=i), Y6.isel(level=i), err_persist.isel(level=i), (X1-X0).isel(level=i)]
    titles = [f'Analysis t+0 ({lev})', f'FuXi +6h ({lev})', 'FuXi - persistence error', 'Recent 6h tendency']
    cmaps = ['coolwarm','coolwarm','RdBu_r','RdBu_r']
    for ax, da, title, cmap in zip(axes, fields, titles, cmaps):
        im = ax.imshow(da.values, aspect='auto', cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude index')
        ax.set_ylabel('Latitude index')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(IMG/f'map_{lev}.png', dpi=180)
    plt.close(fig)

# Figure 3: group RMSE
fig, ax = plt.subplots(figsize=(8,5))
groups = list(metrics['group_rmse'].keys())
vals = [metrics['group_rmse'][g] for g in groups]
ax.bar(groups, vals, color='tab:blue')
ax.set_ylabel('RMSE (normalized units)')
ax.set_title('Grouped 6-hour forecast error by variable family')
fig.tight_layout()
fig.savefig(IMG/'group_rmse.png', dpi=180)
plt.close(fig)

# Figure 4: global mean state comparison
sel = ['Z500','T850','U500','V500','R500','T2M','MSL','TP']
sub = channel_df[channel_df['level'].isin(sel)].copy()
fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(sub))
w = 0.25
ax.bar(x-w, sub['x1_mean'], width=w, label='analysis t+0')
ax.bar(x, sub['y6_mean'], width=w, label='FuXi +6h')
ax.bar(x+w, sub['x0_mean'], width=w, label='analysis t-6h')
ax.set_xticks(x)
ax.set_xticklabels(sub['level'])
ax.set_ylabel('Global mean (normalized units)')
ax.set_title('Representative channel means')
ax.legend()
fig.tight_layout()
fig.savefig(IMG/'representative_means.png', dpi=180)
plt.close(fig)

# Proposal for cascade system based on literature + diagnostic constraints
proposal = {
    'system_name': 'Cascade U-Transformer (proposed)',
    'stages': [
        {
            'name': 'Short-range dynamics model',
            'lead_range_days': '0-5',
            'training_target': '1-step and 2-step 6h forecasts',
            'focus': 'Fast synoptic dynamics, full-resolution updates for all 70 channels'
        },
        {
            'name': 'Medium-range correction model',
            'lead_range_days': '5-10',
            'training_target': 'residual correction to stage-1 trajectories',
            'focus': 'Suppress autoregressive drift using multi-step rollout loss and spectral/latitude weighting'
        },
        {
            'name': 'Extended-range stabilization model',
            'lead_range_days': '10-15',
            'training_target': 'large-scale anomaly evolution and calibrated bias correction',
            'focus': 'Preserve planetary-scale patterns and surface-variable realism when deterministic skill decays'
        }
    ],
    'rationale': [
        'Related work shows autoregressive accumulation is the central barrier in medium-range AI forecasting.',
        'Current case data provide only a 6h forecast, but reveal which variable families are harder to extrapolate from two prior states.',
        'A cascade lets each model specialize by lead time rather than forcing one network to learn all temporal regimes.'
    ],
    'loss_recommendations': [
        'Latitude-weighted RMSE + anomaly correlation term for Z/T/U/V/R fields',
        'Log or transformed loss for precipitation',
        'Spectral loss to preserve synoptic and planetary scales',
        'Rollout consistency loss across handoff boundaries between the three models'
    ],
    'validation_recommendations': [
        'Compare against persistence, linear extrapolation, and ECMWF ensemble mean',
        'Report ACC for Z500, RMSE for T2M/U10/V10/MSL, and event metrics for TP',
        'Evaluate stage-handoff stability at days 5 and 10'
    ]
}
with open(OUT/'cascade_proposal.json','w') as f:
    json.dump(proposal, f, indent=2)
with open(OUT/'metrics_summary.json','w') as f:
    json.dump(metrics, f, indent=2)

print('Wrote analysis outputs and figures.')
print(json.dumps(metrics, indent=2))
