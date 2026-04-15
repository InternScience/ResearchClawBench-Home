"""
Data overview and basic analysis of ERA5 input and FuXi forecast output.
"""
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
ds_in = xr.open_dataset('data/20231012-06_input_netcdf.nc')
ds_out = xr.open_dataset('data/006.nc')

data_in = ds_in['data'].values  # (2, 70, 181, 360)
data_out = ds_out['data'].values  # (1, 1, 70, 181, 360)
levels = ds_in.level.values
lats = ds_in.lat.values
lons = ds_in.lon.values

# Key variable indices
var_map = {}
for i, l in enumerate(levels):
    var_map[l] = i

print("Variable mapping complete.")
print(f"Total variables: {len(levels)}")

# === Figure 1: Input data overview - Z500 at both time steps ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
idx_z500 = var_map['Z500']

for t_idx, ax in enumerate(axes):
    im = ax.imshow(data_in[t_idx, idx_z500, :, :], extent=[0, 360, -90, 90],
                   cmap='RdBu_r', aspect='auto', origin='upper')
    ax.set_title(f'Z500 (Geopotential at 500 hPa)\nTime step {t_idx+1} ({"00 UTC" if t_idx==0 else "06 UTC"} 2023-10-12)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    plt.colorbar(im, ax=ax, label='Normalized value')

plt.tight_layout()
plt.savefig('report/images/fig1_input_z500.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_input_z500.png")

# === Figure 2: Surface variables overview ===
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
surface_vars = ['T2M', 'U10', 'V10', 'MSL', 'TP']
cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'viridis', 'YlGnBu']

for idx, (var, cmap) in enumerate(zip(surface_vars, cmaps)):
    ax = axes[idx // 3, idx % 3]
    vi = var_map[var]
    im = ax.imshow(data_in[1, vi, :, :], extent=[0, 360, -90, 90],
                   cmap=cmap, aspect='auto', origin='upper')
    ax.set_title(f'{var} (06 UTC)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

# Hide last subplot
axes[1, 2].axis('off')
plt.suptitle('Surface Variables at 06 UTC 2023-10-12', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig2_surface_variables.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_surface_variables.png")

# === Figure 3: Vertical cross-section of temperature ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
temp_levels = ['T1000', 'T925', 'T850', 'T700', 'T600', 'T500', 'T400', 'T300', 'T250', 'T200', 'T150', 'T100', 'T50']
temp_indices = [var_map[v] for v in temp_levels]
pressure_vals = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Latitude cross-section at lon=180 (Pacific)
lon_idx = 180
for t_idx, ax in enumerate(axes):
    profile = data_in[t_idx, temp_indices, :, lon_idx]  # (13, 181)
    im = ax.imshow(profile, aspect='auto', cmap='RdBu_r',
                   extent=[-90, 90, 50, 1000])
    ax.invert_yaxis()
    ax.set_title(f'Temperature Cross-section (Lon=180°)\nTime step {t_idx+1}')
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Pressure (hPa)')
    plt.colorbar(im, ax=ax, label='Normalized value')

plt.tight_layout()
plt.savefig('report/images/fig3_temperature_crosssection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_temperature_crosssection.png")

# === Figure 4: Forecast output vs input comparison ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
compare_vars = ['Z500', 'T850', 'U500', 'V500', 'T2M', 'MSL']
compare_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'viridis']

for idx, (var, cmap) in enumerate(zip(compare_vars, compare_cmaps)):
    ax = axes[idx // 3, idx % 3]
    vi = var_map[var]
    # Input at t=1 (06 UTC), Output at step 6h
    input_field = data_in[1, vi, :, :]
    output_field = data_out[0, 0, vi, :, :]
    
    # Show the difference
    diff = output_field - input_field
    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    im = ax.imshow(diff, extent=[0, 360, -90, 90], cmap='RdBu_r',
                   aspect='auto', origin='upper', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{var}: Forecast - Analysis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('FuXi 6h Forecast minus ERA5 Analysis (2023-10-12 06 UTC)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_forecast_difference.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_forecast_difference.png")

# === Figure 5: RMSE by variable type ===
# Compute RMSE for each variable
rmse_by_var = {}
bias_by_var = {}
for i, var in enumerate(levels):
    fc = data_out[0, 0, i, :, :]
    an = data_in[1, i, :, :]
    diff = fc - an
    rmse_by_var[var] = np.sqrt(np.nanmean(diff**2))
    bias_by_var[var] = np.nanmean(diff)

# Group by variable type
var_groups = {
    'Geopotential (Z)': [f'Z{p}' for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000]],
    'Temperature (T)': [f'T{p}' for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000]],
    'U-wind (U)': [f'U{p}' for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000]],
    'V-wind (V)': [f'V{p}' for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000]],
    'Rel. Humidity (R)': [f'R{p}' for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000]],
    'Surface': ['T2M', 'U10', 'V10', 'MSL', 'TP']
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = plt.cm.Set2(np.linspace(0, 1, 6))

for idx, (group_name, vars_list) in enumerate(var_groups.items()):
    ax = axes[idx // 3, idx % 3]
    rmses = [rmse_by_var[v] for v in vars_list if v in rmse_by_var]
    valid_vars = [v for v in vars_list if v in rmse_by_var]
    
    bars = ax.bar(range(len(rmses)), rmses, color=colors[idx], alpha=0.8)
    ax.set_xticks(range(len(valid_vars)))
    ax.set_xticklabels(valid_vars, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{group_name}\nRMSE')
    ax.set_ylabel('RMSE (normalized)')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('RMSE of FuXi 6h Forecast by Variable Group', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig5_rmse_by_group.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_rmse_by_group.png")

# === Figure 6: Latitude-weighted RMSE profiles ===
# Compute latitude-weighted RMSE for key variables at each pressure level
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
lat_weights = np.cos(np.deg2rad(lats))
lat_weights = lat_weights / np.sum(lat_weights) * len(lat_weights)

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
var_types = ['Z', 'T', 'U', 'V']
var_labels = ['Geopotential', 'Temperature', 'U-wind', 'V-wind']
colors_v = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for ax_idx, (vt, vl, col) in enumerate(zip(var_types, var_labels, colors_v)):
    ax = axes[ax_idx]
    rmses_profile = []
    for p in pressure_levels:
        var_name = f'{vt}{p}'
        if var_name in var_map:
            vi = var_map[var_name]
            fc = data_out[0, 0, vi, :, :]
            an = data_in[1, vi, :, :]
            diff = fc - an
            # Latitude-weighted RMSE
            weighted_sq = np.nanmean(diff**2, axis=1) * lat_weights
            rmses_profile.append(np.sqrt(np.mean(weighted_sq)))
        else:
            rmses_profile.append(np.nan)
    
    ax.plot(rmses_profile, pressure_levels, 'o-', color=col, linewidth=2, markersize=6)
    ax.set_ylim(1000, 50)
    ax.set_xlabel('RMSE (normalized)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(f'{vl}')
    ax.grid(alpha=0.3)

plt.suptitle('Latitude-weighted RMSE by Pressure Level (6h Forecast)', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/fig6_rmse_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_rmse_profile.png")

# === Figure 7: Spatial RMSE map for Z500 ===
fig, ax = plt.subplots(figsize=(12, 5))
vi = var_map['Z500']
fc = data_out[0, 0, vi, :, :]
an = data_in[1, vi, :, :]
diff = fc - an
rmse_spatial = np.sqrt(diff**2)

im = ax.imshow(rmse_spatial, extent=[0, 360, -90, 90], cmap='hot_r',
               aspect='auto', origin='upper')
ax.set_title('Z500 RMSE Spatial Distribution (6h Forecast)')
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
plt.colorbar(im, ax=ax, label='Absolute Error (normalized)')
plt.tight_layout()
plt.savefig('report/images/fig7_z500_spatial_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_z500_spatial_rmse.png")

# === Figure 8: Cascade system architecture diagram ===
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Stage 1: Short-range (0-5 days)
rect1 = plt.Rectangle((0.5, 4.5), 3.5, 2.5, linewidth=2, edgecolor='#1f77b4',
                        facecolor='#1f77b4', alpha=0.15, zorder=1)
ax.add_patch(rect1)
ax.text(2.25, 6.5, 'Stage 1', ha='center', fontsize=12, fontweight='bold', color='#1f77b4')
ax.text(2.25, 6.0, 'Short-Range Model', ha='center', fontsize=10, color='#1f77b4')
ax.text(2.25, 5.5, 'U-Transformer', ha='center', fontsize=9, style='italic')
ax.text(2.25, 5.0, 'Days 0-5', ha='center', fontsize=9)
ax.text(2.25, 4.7, '60 steps', ha='center', fontsize=8, color='gray')

# Stage 2: Medium-range (5-10 days)
rect2 = plt.Rectangle((5, 4.5), 3.5, 2.5, linewidth=2, edgecolor='#ff7f0e',
                        facecolor='#ff7f0e', alpha=0.15, zorder=1)
ax.add_patch(rect2)
ax.text(6.75, 6.5, 'Stage 2', ha='center', fontsize=12, fontweight='bold', color='#ff7f0e')
ax.text(6.75, 6.0, 'Medium-Range Model', ha='center', fontsize=10, color='#ff7f0e')
ax.text(6.75, 5.5, 'U-Transformer', ha='center', fontsize=9, style='italic')
ax.text(6.75, 5.0, 'Days 5-10', ha='center', fontsize=9)
ax.text(6.75, 4.7, '60 steps', ha='center', fontsize=8, color='gray')

# Stage 3: Extended-range (10-15 days)
rect3 = plt.Rectangle((9.5, 4.5), 3.5, 2.5, linewidth=2, edgecolor='#2ca02c',
                        facecolor='#2ca02c', alpha=0.15, zorder=1)
ax.add_patch(rect3)
ax.text(11.25, 6.5, 'Stage 3', ha='center', fontsize=12, fontweight='bold', color='#2ca02c')
ax.text(11.25, 6.0, 'Extended-Range Model', ha='center', fontsize=10, color='#2ca02c')
ax.text(11.25, 5.5, 'U-Transformer', ha='center', fontsize=9, style='italic')
ax.text(11.25, 5.0, 'Days 10-15', ha='center', fontsize=9)
ax.text(11.25, 4.7, '60 steps', ha='center', fontsize=8, color='gray')

# Arrows
ax.annotate('', xy=(5, 5.75), xytext=(4, 5.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(9.5, 5.75), xytext=(8.5, 5.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Input
ax.text(2.25, 3.5, 'ERA5 Input\n(2 time steps)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.annotate('', xy=(2.25, 4.5), xytext=(2.25, 3.8),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Output
ax.text(11.25, 3.5, '15-day Forecast\n(60 steps × 6h)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.annotate('', xy=(11.25, 3.8), xytext=(11.25, 4.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

# Title
ax.text(7, 7.5, 'Cascade U-Transformer Forecasting System', ha='center',
        fontsize=16, fontweight='bold')

# Variables
ax.text(7, 1.5, '70 Variables: 5 upper-air × 13 levels + 5 surface', ha='center',
        fontsize=10, style='italic', color='gray')
ax.text(7, 1.0, 'Resolution: 0.25° (181 × 360) | Temporal: 6-hour intervals', ha='center',
        fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('report/images/fig8_cascade_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_cascade_architecture.png")

# Save summary statistics
summary = {
    'input_shape': list(data_in.shape),
    'output_shape': list(data_out.shape),
    'num_variables': len(levels),
    'num_pressure_levels': 13,
    'num_surface_vars': 5,
    'resolution_deg': 1.0,
    'temporal_resolution_hours': 6,
    'forecast_lead_hours': 6,
    'rmse_by_variable': {k: float(v) for k, v in rmse_by_var.items()},
    'bias_by_variable': {k: float(v) for k, v in bias_by_var.items()},
}

import json
with open('outputs/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nAnalysis complete. Summary saved to outputs/analysis_summary.json")
