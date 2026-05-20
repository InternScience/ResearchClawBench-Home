#!/usr/bin/env python3
"""
Phase 1: Data Analysis and Visualization
- Input data structure analysis
- Variable group statistics
- Spatial and spectral characteristics
- FuXi reference output analysis
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import json
import os

# Setup
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white'
})

# ==========================================
# 1. Load and analyze data
# ==========================================
ds_in = xr.open_dataset('data/20231012-06_input_netcdf.nc')
ds_out = xr.open_dataset('data/006.nc')

levels = list(ds_in.level.values)
lats = ds_in.lat.values
lons = ds_in.lon.values
data_in = ds_in.data.values  # [2, 70, 181, 360]
data_out = ds_out.data.values  # [1, 1, 70, 181, 360]

# Variable groups
var_groups = {
    'Geopotential (Z)': {'vars': ['Z50','Z100','Z150','Z200','Z250','Z300','Z400','Z500','Z600','Z700','Z850','Z925','Z1000'], 'color': '#1f77b4'},
    'Temperature (T)': {'vars': ['T50','T100','T150','T200','T250','T300','T400','T500','T600','T700','T850','T925','T1000','T2M'], 'color': '#ff7f0e'},
    'U-Wind (U)': {'vars': ['U50','U100','U150','U200','U250','U300','U400','U500','U600','U700','U850','U925','U1000','U10'], 'color': '#2ca02c'},
    'V-Wind (V)': {'vars': ['V50','V100','V150','V200','V250','V300','V400','V500','V600','V700','V850','V925','V1000','V10'], 'color': '#d62728'},
    'Humidity (R)': {'vars': ['R50','R100','R150','R200','R250','R300','R400','R500','R600','R700','R850','R925','R1000'], 'color': '#9467bd'},
    'MSL Pressure': {'vars': ['MSL'], 'color': '#8c564b'}
}

# Compute indices for each group
group_indices = {}
all_indices = list(range(70))
for name, info in var_groups.items():
    group_indices[name] = [levels.index(v) for v in info['vars'] if v in levels]

# Additional surface: TP
tp_idx = levels.index('TP')
group_indices['Precipitation (TP)'] = [tp_idx]

# Save comprehensive stats
stats = {}
for name, idx in group_indices.items():
    for t in range(2):
        subset = data_in[t, idx, :, :]
        stats[f'{name}_t{t}'] = {
            'min': float(subset.min()),
            'max': float(subset.max()),
            'mean': float(subset.mean()),
            'std': float(subset.std()),
            'median': float(np.median(subset)),
            'q25': float(np.percentile(subset, 25)),
            'q75': float(np.percentile(subset, 75))
        }

with open('outputs/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

# ==========================================
# Figure 1: Variable group overview (multi-panel)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ERA5 Input Variable Groups: Spatial Mean at t=0 (2023-10-12 00:00 UTC)', fontsize=14, fontweight='bold')

group_list = list(group_indices.keys())[:6]
for ax, gname in zip(axes.flat, group_list):
    idx = group_indices[gname]
    # Zonal mean profile
    mean_data = np.mean(data_in[0, idx, :, :], axis=(1, 2))
    level_labels = [levels[i] for i in idx]
    y = range(len(idx))
    
    color = var_groups.get(gname, {}).get('color', '#333333')
    ax.plot(mean_data, y, 'o-', color=color, markersize=4, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(level_labels, fontsize=7)
    ax.set_xlabel('Global Mean Value')
    ax.set_title(gname, fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_variable_groups_overview.png')
plt.close()
print("Figure 1 saved.")

# ==========================================
# Figure 2: Global mean maps for key variables
# ==========================================
key_var_indices = {
    'Z500': levels.index('Z500'),
    'T850': levels.index('T850'),
    'U200': levels.index('U200'),
    'TP': levels.index('TP'),
}

fig, axes = plt.subplots(2, 2, figsize=(18, 8), subplot_kw={'projection': None})
fig.suptitle('Global Field Snapshots: Input State t=0 (2023-10-12 00:00 UTC)', fontsize=13, fontweight='bold')

for ax, (vname, vidx) in zip(axes.flat, key_var_indices.items()):
    field = data_in[0, vidx, :, :]
    vmin, vmax = float(field.min()), float(field.max())
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(field, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f'{vname}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Value (normalized)')

plt.tight_layout()
plt.savefig('report/images/fig2_global_field_snapshots.png')
plt.close()
print("Figure 2 saved.")

# ==========================================
# Figure 3: Latitude-zonal cross-sections
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Latitude-Altitude Cross-Sections (Mean over Longitudes): Input t=0', fontsize=13, fontweight='bold')

pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
cross_vars = [('Z', 'Geopotential'), ('T', 'Temperature'), ('U', 'Zonal Wind')]

for ax, (prefix, title) in zip(axes, cross_vars):
    prefix_vars = [f'{prefix}{p}' for p in pressure_levels if f'{prefix}{p}' in levels]
    idx = [levels.index(v) for v in prefix_vars]
    data_cs = data_in[0, idx, :, :]  # [nlevels, nlat, nlon]
    zonal_mean = data_cs.mean(axis=2)  # [nlevels, nlat]
    
    pcm = ax.pcolormesh(lats, range(len(idx)), zonal_mean, cmap='RdBu_r', shading='auto')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([f'{p} hPa' for p in pressure_levels[:len(idx)]], fontsize=8)
    ax.set_xlabel('Latitude')
    ax.set_title(f'{title}', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    plt.colorbar(pcm, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('report/images/fig3_latitude_cross_sections.png')
plt.close()
print("Figure 3 saved.")

# ==========================================
# Figure 4: Spectral analysis
# ==========================================
from scipy.fft import fft2, fftshift

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Power Spectral Density of Key Variables (t=0)', fontsize=13, fontweight='bold')

spectral_vars = {'Z500': levels.index('Z500'), 'T850': levels.index('T850'), 
                 'U200': levels.index('U200'), 'V500': levels.index('V500'),
                 'R850': levels.index('R850'), 'MSL': levels.index('MSL')}

for ax, (vname, vidx) in zip(axes.flat, spectral_vars.items()):
    field = data_in[0, vidx, :, :]
    # Remove mean
    field_anom = field - field.mean()
    # 2D FFT
    fft_field = fftshift(fft2(field_anom))
    power = np.abs(fft_field)**2
    # Radial average
    ny, nx = power.shape
    cy, cx = ny//2, nx//2
    Y, X = np.mgrid[0:ny, 0:nx]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    
    # Radial mean
    r_bins = np.arange(0, min(cy, cx), 2)
    radial_power = []
    for r in r_bins[:-1]:
        mask = (R >= r_bins[0]) & (R < r_bins[1]) if r == r_bins[0] else (R >= r) & (R < r+2)
        if mask.sum() > 0:
            radial_power.append(power[mask].mean())
        else:
            radial_power.append(np.nan)
    
    # Simpler approach: just show log power spectrum
    log_power = np.log10(power + 1)
    im = ax.imshow(log_power, cmap='inferno', aspect='auto')
    ax.set_title(f'{vname}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Wavenumber (kx)')
    ax.set_ylabel('Wavenumber (ky)')
    plt.colorbar(im, ax=ax, shrink=0.7, label='log₁₀(PSD)')

plt.tight_layout()
plt.savefig('report/images/fig4_spectral_analysis.png')
plt.close()
print("Figure 4 saved.")

# ==========================================
# Figure 5: FuXi output vs Input comparison
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Input (t=0) vs FuXi Forecast (t+6h): Variable Comparison', fontsize=13, fontweight='bold')

compare_vars = {'Z500': levels.index('Z500'), 'T850': levels.index('T850'), 
                'U200': levels.index('U200'), 'R850': levels.index('R850'),
                'V500': levels.index('V500'), 'MSL': levels.index('MSL')}

for ax, (vname, vidx) in zip(axes.flat, compare_vars.items()):
    input_field = data_in[1, vidx, :, :]  # t=1 (the latest input)
    output_field = data_out[0, 0, vidx, :, :]  # FuXi 6h forecast
    
    # Difference
    diff = output_field - input_field
    rmse = np.sqrt(np.mean(diff**2))
    corr = np.corrcoef(input_field.flatten(), output_field.flatten())[0, 1]
    
    dmin, dmax = float(diff.min()), float(diff.max())
    if dmin < 0 < dmax:
        norm = TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    else:
        norm = None
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f'{vname}\nΔRMSE={rmse:.4f}, r={corr:.4f}', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Forecast - Input')

plt.tight_layout()
plt.savefig('report/images/fig5_fuxi_comparison.png')
plt.close()
print("Figure 5 saved.")

# ==========================================
# Figure 6: Variable distribution histograms
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Value Distributions by Variable Group (Input t=0)', fontsize=13, fontweight='bold')

for ax, gname in zip(axes.flat, list(group_indices.keys())[:6]):
    idx = group_indices[gname]
    data_flat = data_in[0, idx, :, :].flatten()
    # Subsample for speed
    if len(data_flat) > 100000:
        data_flat = np.random.choice(data_flat, 100000, replace=False)
    
    color = var_groups.get(gname, {}).get('color', '#333333')
    ax.hist(data_flat, bins=80, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Value (normalized)')
    ax.set_ylabel('Count')
    ax.set_title(gname, fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/fig6_variable_distributions.png')
plt.close()
print("Figure 6 saved.")

# Close datasets
ds_in.close()
ds_out.close()

print("\nPhase 1 complete. All figures and statistics saved.")
