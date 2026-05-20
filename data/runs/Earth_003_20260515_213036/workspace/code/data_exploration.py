"""
Data Exploration and Baseline Analysis for Cascade U-Transformer Weather Forecasting
"""
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import json
import os

# Paths
DATA_DIR = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_003_20260515_213036/data'
OUTPUT_DIR = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_003_20260515_213036/outputs'
REPORT_IMG_DIR = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Earth_003_20260515_213036/report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# Load data
ds_in = nc.Dataset(f'{DATA_DIR}/20231012-06_input_netcdf.nc')
ds_fx = nc.Dataset(f'{DATA_DIR}/006.nc')

d_in = ds_in['data'][:]  # (2, 70, 181, 360)
d_fx = ds_fx['data'][:]  # (1, 1, 70, 181, 360)
level_names = [str(l) for l in ds_in['level'][:]]
lats = ds_in['lat'][:]
lons = ds_in['lon'][:]

# Variable groups
z_idx = list(range(0, 13))     # Geopotential
t_idx = list(range(13, 26))    # Temperature
u_idx = list(range(26, 39))    # U-wind
v_idx = list(range(39, 52))    # V-wind
r_idx = list(range(52, 65))    # Relative humidity
sfc_idx = list(range(65, 70))  # Surface variables

var_groups = {
    'Geopotential (Z)': z_idx,
    'Temperature (T)': t_idx,
    'U-wind (U)': u_idx,
    'V-wind (V)': v_idx,
    'Relative Humidity (R)': r_idx,
    'Surface': sfc_idx,
}

# ============================================================
# Figure 1: Data Overview - Variable distributions
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = plt.cm.tab10(np.linspace(0, 1, 6))

for ax, (name, idxs), color in zip(axes.flat, var_groups.items(), colors):
    data_t0 = d_in[0, idxs].flatten()
    data_t1 = d_in[1, idxs].flatten()
    ax.hist(data_t0, bins=100, alpha=0.5, density=True, label='t=0h', color=color, edgecolor='none')
    ax.hist(data_t1, bins=100, alpha=0.5, density=True, label='t=6h', color=color, edgecolor='black', 
            histtype='step', linewidth=2, linestyle='--')
    ax.set_title(name)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Distribution of Atmospheric Variables at t=0h and t=6h', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT_IMG_DIR}/data_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: data_distribution.png")

# ============================================================
# Figure 2: Spatial maps of key variables
# ============================================================
key_vars = [(7, 'Z500 - Geopotential at 500 hPa'), 
            (20, 'T500 - Temperature at 500 hPa'),
            (33, 'U500 - U-wind at 500 hPa'),
            (46, 'V500 - V-wind at 500 hPa'),
            (59, 'R500 - Rel. Humidity at 500 hPa'),
            (65, 'T2M - 2m Temperature')]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (c, name) in zip(axes.flat, key_vars):
    im = ax.imshow(d_in[1, c], cmap='RdBu_r', aspect='auto')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Global Atmospheric State at t=6h (2023-10-12 12:00 UTC)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT_IMG_DIR}/spatial_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: spatial_maps.png")

# ============================================================
# Figure 3: FuXi 6h forecast error analysis
# ============================================================
err = d_in[1] - d_fx[0, 0]  # (70, 181, 360)
persist_err = d_in[1] - d_in[0]

rmse_per_channel = np.sqrt(np.mean(err**2, axis=(1, 2)))
rmse_persist_per_channel = np.sqrt(np.mean(persist_err**2, axis=(1, 2)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Per-channel RMSE comparison
x = np.arange(70)
width = 0.35
ax1.bar(x - width/2, rmse_persist_per_channel, width, label='Persistence', color='steelblue', alpha=0.7)
ax1.bar(x + width/2, rmse_per_channel, width, label='FuXi', color='coral', alpha=0.7)
ax1.set_xlabel('Channel Index')
ax1.set_ylabel('RMSE')
ax1.set_title('Per-Channel RMSE: Persistence vs FuXi (6h)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE by variable group
groups = ['Z', 'T', 'U', 'V', 'R', 'Sfc']
group_rmse_fx = [np.mean(rmse_per_channel[idx]) for idx in [z_idx, t_idx, u_idx, v_idx, r_idx, sfc_idx]]
group_rmse_ps = [np.mean(rmse_persist_per_channel[idx]) for idx in [z_idx, t_idx, u_idx, v_idx, r_idx, sfc_idx]]

x2 = np.arange(len(groups))
ax2.bar(x2 - width/2, group_rmse_ps, width, label='Persistence', color='steelblue', alpha=0.7)
ax2.bar(x2 + width/2, group_rmse_fx, width, label='FuXi', color='coral', alpha=0.7)
ax2.set_xticks(x2)
ax2.set_xticklabels(groups)
ax2.set_xlabel('Variable Group')
ax2.set_ylabel('Mean RMSE')
ax2.set_title('Mean RMSE by Variable Group')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle('FuXi 6-Hour Forecast Error Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT_IMG_DIR}/fuxi_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: fuxi_error_analysis.png")

# ============================================================
# Figure 4: Error spatial distribution
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (c, name) in zip(axes.flat, key_vars):
    err_map = err[c]
    vmax = np.percentile(np.abs(err_map), 99)
    im = ax.imshow(err_map, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{name} Error', fontsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('FuXi 6h Forecast Spatial Error Maps', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT_IMG_DIR}/error_spatial_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: error_spatial_maps.png")

# Save analysis results
results = {
    'global_rmse_fuxi': float(np.sqrt(np.mean(err**2))),
    'global_rmse_persistence': float(np.sqrt(np.mean(persist_err**2))),
    'global_mae_fuxi': float(np.mean(np.abs(err))),
    'global_mae_persistence': float(np.mean(np.abs(persist_err))),
    'rmse_per_channel_fuxi': rmse_per_channel.tolist(),
    'rmse_per_channel_persistence': rmse_persist_per_channel.tolist(),
    'group_mean_rmse_fuxi': {g: float(v) for g, v in zip(groups, group_rmse_fx)},
    'group_mean_rmse_persistence': {g: float(v) for g, v in zip(groups, group_rmse_ps)},
}

with open(f'{OUTPUT_DIR}/baseline_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis complete. Results saved to outputs/baseline_analysis.json")
