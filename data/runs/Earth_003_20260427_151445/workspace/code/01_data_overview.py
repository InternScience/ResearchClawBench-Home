"""
01_data_overview.py
Inventory the FuXi-style ERA5-derived input and the supplied 6h forecast,
compute per-channel statistics, and dump CSVs/figures for the report.
"""
import os, json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT  = os.path.join(ROOT, 'outputs')
IMG  = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

mpl.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 140, 'font.size': 10})

ds_in  = xr.open_dataset(os.path.join(DATA, '20231012-06_input_netcdf.nc'))
ds_out = xr.open_dataset(os.path.join(DATA, '006.nc'))

levels = list(ds_in.level.values)
lats = ds_in.lat.values  # 90..-90
lons = ds_in.lon.values  # 0..359
print('Number of channels:', len(levels))
print('Lat/Lon shape:', lats.shape, lons.shape)
print('Input time:', ds_in.time.values, 'Output time:', ds_out.time.values, 'step:', ds_out.step.values)

X0 = ds_in['data'].isel(time=0).values   # (70, 181, 360) — t-6h
X1 = ds_in['data'].isel(time=1).values   # (70, 181, 360) — t0
Y6 = ds_out['data'].isel(time=0, step=0).values  # (70, 181, 360) — t+6h forecast (FuXi)
print('X0', X0.shape, 'X1', X1.shape, 'Y6', Y6.shape)

# --- Per-channel statistics
def stats(arr, name):
    rows = []
    for i, lvl in enumerate(levels):
        a = arr[i]
        rows.append(dict(channel=lvl, idx=i, src=name,
                         min=float(a.min()), max=float(a.max()),
                         mean=float(a.mean()), std=float(a.std()),
                         abs_mean=float(np.abs(a).mean())))
    return pd.DataFrame(rows)

df = pd.concat([stats(X0,'t_-6h_input'), stats(X1,'t_0_input'), stats(Y6,'t_+6h_fuxi')], ignore_index=True)
df.to_csv(os.path.join(OUT, 'channel_statistics.csv'), index=False)
print('saved channel_statistics.csv with', len(df), 'rows')

# --- Per-channel RMSE/MAE/correlation between t0 and t+6h, and persistence baseline (t-6h vs t+6h)
metric_rows = []
# latitude weights (cos lat)
w = np.cos(np.deg2rad(lats)); w = w / w.mean()
W = np.broadcast_to(w[:, None], (181, 360))
for i, lvl in enumerate(levels):
    a, b, c = X1[i], Y6[i], X0[i]
    # weighted RMSE
    diff = (b - a)
    rmse_fc = np.sqrt(np.average(diff**2, weights=W))
    mae_fc  = np.average(np.abs(diff), weights=W)
    # persistence baseline: predicting t+6 = t0 has residual b - a (same), but we also test naive change
    rmse_persist = np.sqrt(np.average((a - a)**2, weights=W))  # zero by definition
    rmse_change_t0 = np.sqrt(np.average((a - c)**2, weights=W))  # 6h temporal diff in obs
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    metric_rows.append(dict(channel=lvl, idx=i,
                            rmse_t0_vs_t6_fc=rmse_fc, mae_t0_vs_t6_fc=mae_fc,
                            corr_t0_t6=corr,
                            rmse_t_neg6_vs_t0=rmse_change_t0))
mdf = pd.DataFrame(metric_rows)
mdf.to_csv(os.path.join(OUT, 'fuxi_6h_per_channel_metrics.csv'), index=False)
print('saved fuxi_6h_per_channel_metrics.csv')

# --- Figure: per-channel RMSE bar grouped by variable family
fam_color = {'Z':'#1f77b4','T':'#ff7f0e','U':'#2ca02c','V':'#d62728','R':'#9467bd',
             'T2M':'#8c564b','U10':'#e377c2','V10':'#7f7f7f','MSL':'#bcbd22','TP':'#17becf'}
def family(c):
    if c.startswith('Z'): return 'Z'
    if c.startswith('T') and c not in ('T2M',): return 'T'
    if c.startswith('U') and c != 'U10': return 'U'
    if c.startswith('V') and c != 'V10': return 'V'
    if c.startswith('R'): return 'R'
    return c

fig, ax = plt.subplots(figsize=(14, 4.5))
colors = [fam_color[family(l)] for l in levels]
ax.bar(range(len(levels)), mdf['rmse_t0_vs_t6_fc'].values, color=colors)
ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, rotation=90, fontsize=7)
ax.set_ylabel('Normalized 6 h RMSE (forecast vs analysis)')
ax.set_title('Per-channel 6 h tendency magnitude (FuXi forecast minus input analysis, normalized space)')
ax.grid(True, axis='y', alpha=0.3)
# legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=v,label=k) for k,v in fam_color.items()]
ax.legend(handles=handles, ncol=10, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(IMG,'per_channel_rmse_bar.png')); plt.close(fig)

# --- Figure: data overview - mean/std per channel
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
axs[0].bar(range(len(levels)), df[df.src=='t_0_input']['mean'].values, color=colors)
axs[0].set_xticks(range(len(levels))); axs[0].set_xticklabels(levels, rotation=90, fontsize=7)
axs[0].set_ylabel('Channel mean (normalized)'); axs[0].set_title('Per-channel mean at t=0 (input)')
axs[0].grid(True, axis='y', alpha=0.3)
axs[1].bar(range(len(levels)), df[df.src=='t_0_input']['std'].values, color=colors)
axs[1].set_xticks(range(len(levels))); axs[1].set_xticklabels(levels, rotation=90, fontsize=7)
axs[1].set_ylabel('Channel std (normalized)'); axs[1].set_title('Per-channel std at t=0 (input)')
axs[1].grid(True, axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'data_overview_channels.png')); plt.close(fig)

# --- Figure: sample maps of input fields
def plot_map(ax, field, title, cmap='RdBu_r'):
    vmax = np.percentile(np.abs(field), 99)
    im = ax.imshow(field, extent=[lons.min(),lons.max(),lats.min(),lats.max()],
                   origin='upper', cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

key_fields = [('Z500','Geopotential @ 500 hPa'),
              ('T850','Temperature @ 850 hPa'),
              ('T2M', '2 m temperature'),
              ('U10', '10 m zonal wind'),
              ('V10', '10 m meridional wind'),
              ('MSL', 'Mean sea-level pressure'),
              ('R500','Relative humidity @ 500 hPa'),
              ('TP',  'Total precipitation')]
fig, axs = plt.subplots(4, 2, figsize=(13, 12))
for ax,(name,title) in zip(axs.ravel(), key_fields):
    idx = levels.index(name)
    cmap = 'viridis' if name == 'TP' else 'RdBu_r'
    if name == 'TP':
        f = X1[idx]
        vmax = np.percentile(f, 99)
        im = ax.imshow(f, extent=[lons.min(),lons.max(),lats.min(),lats.max()],
                       origin='upper', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(f'{title} (t=0, normalized)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    else:
        plot_map(ax, X1[idx], f'{title} (t=0, normalized)')
fig.suptitle('Input atmospheric state at 2023-10-12 06UTC (normalized)', y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'input_state_maps.png')); plt.close(fig)

# --- Figure: 6h tendency (forecast - input) maps
fig, axs = plt.subplots(4, 2, figsize=(13, 12))
for ax,(name,title) in zip(axs.ravel(), key_fields):
    idx = levels.index(name)
    diff = Y6[idx] - X1[idx]
    plot_map(ax, diff, f'{title}: (t+6h FuXi) − (t=0) [normalized]')
fig.suptitle('6-hour tendency: FuXi forecast minus input analysis (normalized)', y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'forecast_tendency_maps.png')); plt.close(fig)

# --- Latitudinal RMSE profile for selected fields
sel = ['Z500','T850','T2M','U10','MSL','TP']
fig, ax = plt.subplots(figsize=(9, 5))
for name in sel:
    idx = levels.index(name)
    diff = Y6[idx] - X1[idx]
    rmse_lat = np.sqrt(np.mean(diff**2, axis=1))   # rmse along longitude per latitude
    ax.plot(lats, rmse_lat, label=name)
ax.set_xlabel('Latitude (deg)'); ax.set_ylabel('Zonal RMSE (normalized space)')
ax.set_title('Latitudinal profile of FuXi 6 h forecast tendency RMSE')
ax.invert_xaxis()
ax.legend(ncol=3); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'latitudinal_rmse.png')); plt.close(fig)

# --- Zonal wavenumber power spectrum for Z500 and T850
def zonal_spectrum(field):
    f = np.fft.rfft(field, axis=1)  # (lat, k)
    p = (np.abs(f)**2).mean(axis=0)
    return p
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, name in zip(axs, ['Z500','T850']):
    idx = levels.index(name)
    p_in = zonal_spectrum(X1[idx])
    p_fc = zonal_spectrum(Y6[idx])
    k = np.arange(len(p_in))
    ax.loglog(k[1:], p_in[1:], label=f'{name} t=0')
    ax.loglog(k[1:], p_fc[1:], label=f'{name} t+6h FuXi', linestyle='--')
    ax.set_xlabel('Zonal wavenumber k'); ax.set_ylabel('Power')
    ax.set_title(f'Zonal power spectrum — {name}')
    ax.grid(True, which='both', alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(IMG,'spectral_power.png')); plt.close(fig)

# --- Save summary JSON
summary = dict(
    n_channels=len(levels),
    grid=(int(lats.size), int(lons.size)),
    nominal_resolution_deg = round(float(lons[1] - lons[0]), 4),
    init_time=str(ds_in.time.values[1]),
    forecast_lead_hours=int(ds_out.step.values[0]),
    families_count={k: sum(1 for l in levels if family(l)==k) for k in fam_color.keys()},
    sample_metrics={
        'Z500_rmse_t0_t6': float(mdf.loc[mdf.channel=='Z500','rmse_t0_vs_t6_fc'].iloc[0]),
        'T850_rmse_t0_t6': float(mdf.loc[mdf.channel=='T850','rmse_t0_vs_t6_fc'].iloc[0]),
        'T2M_rmse_t0_t6':  float(mdf.loc[mdf.channel=='T2M','rmse_t0_vs_t6_fc'].iloc[0]),
        'MSL_rmse_t0_t6':  float(mdf.loc[mdf.channel=='MSL','rmse_t0_vs_t6_fc'].iloc[0]),
        'TP_rmse_t0_t6':   float(mdf.loc[mdf.channel=='TP','rmse_t0_vs_t6_fc'].iloc[0]),
    }
)
with open(os.path.join(OUT,'data_summary.json'),'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print('OK')
