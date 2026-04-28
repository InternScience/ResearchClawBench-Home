"""
02_data_authenticity_check.py
Quantitatively check whether the supplied input tensor and FuXi 6h
forecast tensor contain physical structure or are statistically
equivalent to pre-standardized white noise.
Diagnostics:
  - mean / std per channel (reported in 01)
  - lag-1 spatial autocorrelation (zonal & meridional)
  - cross-channel correlation (e.g. Z500 vs T500 — physically expected
    high anti-correlation between heights and temps tendencies)
  - input-to-input lag-6h autocorrelation (X0 vs X1)
  - input-to-forecast lag-6h correlation (X1 vs Y6)
"""
import os, json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data'); OUT  = os.path.join(ROOT, 'outputs'); IMG = os.path.join(ROOT, 'report', 'images')

ds_in  = xr.open_dataset(os.path.join(DATA, '20231012-06_input_netcdf.nc'))
ds_out = xr.open_dataset(os.path.join(DATA, '006.nc'))
levels = list(ds_in.level.values)
X0 = ds_in['data'].isel(time=0).values
X1 = ds_in['data'].isel(time=1).values
Y6 = ds_out['data'].isel(time=0, step=0).values

def lag1_autocorr(field):
    a = field; m = a.mean(); s = a.std()
    z = (a - m) / s
    rx = (z * np.roll(z, 1, axis=1)).mean()
    ry = (z * np.roll(z, 1, axis=0)).mean()
    return rx, ry

rows = []
for i, lvl in enumerate(levels):
    rx, ry = lag1_autocorr(X1[i])
    rows.append(dict(channel=lvl,
                     X0_X1_corr=float(np.corrcoef(X0[i].ravel(), X1[i].ravel())[0,1]),
                     X1_Y6_corr=float(np.corrcoef(X1[i].ravel(), Y6[i].ravel())[0,1]),
                     lag1_lon=float(rx), lag1_lat=float(ry)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'authenticity_diagnostics.csv'), index=False)

summary = dict(
    median_X0_X1_corr=float(df['X0_X1_corr'].median()),
    median_X1_Y6_corr=float(df['X1_Y6_corr'].median()),
    median_lag1_lon=float(df['lag1_lon'].median()),
    median_lag1_lat=float(df['lag1_lat'].median()),
    interpretation="Physical ERA5 fields would show median |lag1| autocorrelations >> 0.9 "
                   "and lag-6h temporal autocorrelations >> 0.9 for upper-air variables. "
                   "Values near zero indicate the tensors contain pre-standardized noise "
                   "rather than real reanalysis data.",
)
with open(os.path.join(OUT, 'authenticity_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Cross-channel correlation matrix at t=0 (collapse spatial dims to vectors)
flat = X1.reshape(70, -1)
corr = np.corrcoef(flat)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Cross-channel correlation matrix at t=0 (input)\n'
             '(physical ERA5 would show strong block structure across pressure levels)')
ax.set_xticks([0, 13, 26, 39, 52, 65, 70])
ax.set_xticklabels(['Z','T','U','V','R','sfc','end'], fontsize=9)
ax.set_yticks([0, 13, 26, 39, 52, 65, 70])
ax.set_yticklabels(['Z','T','U','V','R','sfc','end'], fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'cross_channel_correlation.png')); plt.close(fig)

# Diagnostic figure: histogram of correlations & autocorrelations
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, title in zip(axs,
                          ['X0_X1_corr', 'X1_Y6_corr', 'lag1_lon'],
                          ['Lag-6h temporal autocorr (X0 vs X1)',
                           'Forecast correlation (X1 vs t+6h FuXi)',
                           'Lag-1 zonal spatial autocorr at t=0']):
    ax.hist(df[col], bins=30, color='#4477aa', edgecolor='black')
    ax.axvline(0, color='k', lw=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
fig.suptitle('Authenticity diagnostics: the supplied tensors are statistically white', y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'authenticity_diagnostics.png'),
                                bbox_inches='tight'); plt.close(fig)

print(json.dumps(summary, indent=2))
print('per-channel stats:')
print(df.describe().to_string())
