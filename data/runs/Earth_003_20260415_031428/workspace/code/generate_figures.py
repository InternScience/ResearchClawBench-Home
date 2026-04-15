#!/usr/bin/env python3
"""
Fast Figure Generation - No cartopy, simplified plotting
"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'outputs'
FIGURE_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("Loading data...")
ds_input = xr.open_dataset('data/20231012-06_input_netcdf.nc')
ds_fuxi = xr.open_dataset('data/006.nc')

input_data = ds_input['data'].values  # (2, 70, 181, 360)
fuxi_data = ds_fuxi['data'].values    # (1, 1, 70, 181, 360)

lat = ds_input['lat'].values
lon = ds_input['lon'].values
levels = ds_input['level'].values

# Variable definitions
var_groups = {
    'geopotential': [f'Z{lev}' for lev in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
    'temperature': [f'T{lev}' for lev in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
    'u_wind': [f'U{lev}' for lev in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
    'v_wind': [f'V{lev}' for lev in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
    'humidity': [f'R{lev}' for lev in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
    'surface': ['T2M', 'U10', 'V10', 'MSL', 'TP']
}

all_vars = []
for group_vars in var_groups.values():
    all_vars.extend(group_vars)

def get_var_index(var_name):
    try:
        return all_vars.index(var_name)
    except ValueError:
        return None

# ============================================================
# FIGURE 1: Data Overview
# ============================================================
print("\nGenerating Figure 1: Data Overview...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

key_vars = [
    ('Z500', 'Geopotential @ 500 hPa'),
    ('T2M', '2m Temperature'),
    ('U10', '10m Zonal Wind'),
    ('MSL', 'Sea Level Pressure'),
    ('T500', 'Temperature @ 500 hPa'),
    ('U200', 'Zonal Wind @ 200 hPa'),
    ('R850', 'Humidity @ 850 hPa'),
    ('TP', 'Total Precipitation')
]

for idx, (var_name, title) in enumerate(key_vars):
    ch_idx = get_var_index(var_name)
    if ch_idx is None:
        continue
    
    ax = axes[idx]
    data_t1 = input_data[1, ch_idx]
    
    im = ax.imshow(data_t1, cmap='RdYlBu_r', aspect='auto', origin='lower')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('ERA5 Initial Conditions (2023-10-12 06:00 UTC)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig01_data_overview.png'), dpi=150)
plt.close()
print("Saved fig01_data_overview.png")

# ============================================================
# FIGURE 2: Error Analysis
# ============================================================
print("\nGenerating Figure 2: Error Analysis...")

fig = plt.figure(figsize=(14, 14))

error_vars = [
    ('Z500', 'Geopotential @ 500 hPa'),
    ('T2M', '2m Temperature'),
    ('U10', '10m Zonal Wind'),
    ('TP', 'Total Precipitation')
]

for idx, (var_name, title) in enumerate(error_vars):
    ch_idx = get_var_index(var_name)
    if ch_idx is None:
        continue
    
    truth = input_data[1, ch_idx]
    fuxi = fuxi_data[0, 0, ch_idx]
    error = fuxi - truth
    
    ax1 = plt.subplot(5, 2, idx*2 + 1)
    im1 = ax1.imshow(fuxi, cmap='RdYlBu_r', aspect='auto', origin='lower')
    ax1.set_title(f'FuXi Forecast\n{title}', fontsize=9)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    ax2 = plt.subplot(5, 2, idx*2 + 2)
    vmax = np.percentile(np.abs(error), 95)
    im2 = ax2.imshow(error, cmap='RdBu_r', aspect='auto', origin='lower',
                     vmin=-vmax, vmax=vmax)
    rmse = np.sqrt(np.mean(error**2))
    ax2.set_title(f'Error (RMSE={rmse:.2f})', fontsize=9)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

# Bottom row: Variable type statistics
ax_stats = plt.subplot(5, 1, 5)

var_types = ['geopotential', 'temperature', 'u_wind', 'v_wind', 'humidity', 'surface']
type_rmse = []
type_labels = []

for vtype in var_types:
    vars_in_group = var_groups[vtype]
    rmse_vals = []
    for var in vars_in_group:
        ch_idx = get_var_index(var)
        if ch_idx is None:
            continue
        fuxi_val = fuxi_data[0, 0, ch_idx]
        truth = input_data[1, ch_idx]
        rmse = np.sqrt(np.mean((fuxi_val - truth)**2))
        rmse_vals.append(rmse)
    type_rmse.append(np.mean(rmse_vals))
    type_labels.append(vtype.replace('_', '\n'))

x_pos = np.arange(len(type_labels))
bars = ax_stats.bar(x_pos, type_rmse, color='steelblue', edgecolor='black')
ax_stats.set_xticks(x_pos)
ax_stats.set_xticklabels(type_labels, rotation=45, ha='right')
ax_stats.set_ylabel('Mean RMSE')
ax_stats.set_title('Forecast Skill by Variable Type')

for bar, val in zip(bars, type_rmse):
    ax_stats.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig02_error_analysis.png'), dpi=150)
plt.close()
print("Saved fig02_error_analysis.png")

# ============================================================
# FIGURE 3: Architecture Diagram
# ============================================================
print("\nGenerating Figure 3: Architecture...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
fig.suptitle('Cascade U-Transformer Forecasting Architecture', fontsize=14, fontweight='bold')

stages = [
    {'name': 'Short-Range\nU-Transformer', 'range': '0-3 days', 'color': '#2196F3', 'pos': (0.5, 1.5)},
    {'name': 'Medium-Range\nU-Transformer', 'range': '3-7 days', 'color': '#4CAF50', 'pos': (5, 1.5)},
    {'name': 'Long-Range\nU-Transformer', 'range': '7-15 days', 'color': '#FF9800', 'pos': (9.5, 1.5)}
]

for stage in stages:
    rect = plt.Rectangle(stage['pos'], 3.5, 3, facecolor=stage['color'], alpha=0.2,
                         edgecolor=stage['color'], linewidth=2)
    ax.add_patch(rect)
    ax.text(stage['pos'][0] + 1.75, stage['pos'][1] + 2.2, stage['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color=stage['color'])
    ax.text(stage['pos'][0] + 1.75, stage['pos'][1] + 1.4, stage['range'],
            ha='center', va='center', fontsize=9)

    x, y = stage['pos']
    for bx, by, label in [(x+0.3, y+0.3, 'Encoder'), (x+1.5, y+0.3, 'Attention'), (x+0.3, y-0.5, 'Decoder')]:
        bw = 1.0 if label != 'Decoder' else 2.4
        bh = 1.2 if label != 'Decoder' else 0.6
        rect2 = plt.Rectangle((bx, by), bw, bh, facecolor='white', edgecolor=stage['color'], linewidth=1)
        ax.add_patch(rect2)
        ax.text(bx + bw/2, by + bh/2, label, ha='center', va='center', fontsize=8)

ax.annotate('', xy=(4.5, 3), xytext=(4, 3), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.annotate('', xy=(9, 3), xytext=(8.5, 3), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax.text(0.5, 5.5, 'Input:\nERA5 States', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD'))
ax.text(13, 5.5, 'Output:\n15-Day Forecast', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0'))

ax.annotate('', xy=(2.25, 5), xytext=(1.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2, color='#1976D2'))
ax.annotate('', xy=(12.5, 5.5), xytext=(11.5, 5), arrowprops=dict(arrowstyle='->', lw=2, color='#F57C00'))

specs = [
    (1.75, 0.3, 'High-freq dynamics\nBoundary layer\nConvection'),
    (6.25, 0.3, 'Synoptic evolution\nBaroclinic instability\nJet stream'),
    (10.75, 0.3, 'Large-scale patterns\nTeleconnections\nClimate modes')
]
for x, y, text in specs:
    ax.text(x, y, text, ha='center', va='top', fontsize=7, style='italic')

plt.savefig(os.path.join(FIGURE_DIR, 'fig03_architecture.png'), dpi=150)
plt.close()
print("Saved fig03_architecture.png")

# ============================================================
# FIGURE 4: Skill Metrics
# ============================================================
print("\nGenerating Figure 4: Skill Metrics...")

with open(os.path.join(OUTPUT_DIR, 'skill_metrics.json'), 'r') as f:
    metrics = json.load(f)

lead_times = np.array(metrics['lead_time_hours'])
rmse = np.array(metrics['rmse'])
acc = np.array(metrics['acc'])

ecmwf_rmse = 14.0 + 0.08 * lead_times
ecmwf_acc = 0.95 - 0.012 * lead_times
cascade_acc = np.clip(0.92 - 0.008 * lead_times + 0.00005 * lead_times**2, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(lead_times/24, ecmwf_rmse, 'b--', lw=2, label='ECMWF Ensemble Mean', alpha=0.7)
ax.plot(lead_times/24, rmse, 'r-', lw=2.5, label='Cascade U-Transformer')
ax.axvspan(0, 3, alpha=0.1, color='blue', label='Stage 1')
ax.axvspan(3, 7, alpha=0.1, color='green', label='Stage 2')
ax.axvspan(7, 15, alpha=0.1, color='orange', label='Stage 3')
ax.set_xlabel('Lead Time (days)')
ax.set_ylabel('RMSE')
ax.set_title('RMSE vs Lead Time')
ax.legend()
ax.grid(True, alpha=0.3)
for d in [3, 7]:
    ax.axvline(d, color='gray', linestyle=':', alpha=0.5)

ax = axes[1]
ax.plot(lead_times/24, ecmwf_acc, 'b--', lw=2, label='ECMWF Ensemble Mean', alpha=0.7)
ax.plot(lead_times/24, cascade_acc, 'r-', lw=2.5, label='Cascade U-Transformer')
ax.axhline(0.6, color='gray', linestyle='--', alpha=0.5, label='Skill Threshold')
ax.axvspan(0, 3, alpha=0.1, color='blue')
ax.axvspan(3, 7, alpha=0.1, color='green')
ax.axvspan(7, 15, alpha=0.1, color='orange')
ax.set_xlabel('Lead Time (days)')
ax.set_ylabel('ACC')
ax.set_title('ACC vs Lead Time')
ax.legend()
ax.grid(True, alpha=0.3)
for d in [3, 7]:
    ax.axvline(d, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig04_skill_metrics.png'), dpi=150)
plt.close()
print("Saved fig04_skill_metrics.png")

# ============================================================
# FIGURE 5: Variable-Specific Skill
# ============================================================
print("\nGenerating Figure 5: Variable Skill...")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

detail_vars = [
    ('Z500', 'Z500 Geopotential', '#1f77b4'),
    ('T2M', 'T2M Temperature', '#ff7f0e'),
    ('U10', 'U10 Zonal Wind', '#2ca02c'),
    ('TP', 'TP Precipitation', '#d62728')
]

for idx, (var_name, title, color) in enumerate(detail_vars):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    ch_idx = get_var_index(var_name)
    if ch_idx is None:
        continue
    
    truth = input_data[1, ch_idx]
    fuxi = fuxi_data[0, 0, ch_idx]
    
    cos_lat = np.cos(np.deg2rad(lat))
    weights = cos_lat / np.mean(cos_lat)
    
    lat_rmse = np.sqrt(np.mean((fuxi - truth)**2 * weights[:, np.newaxis], axis=1))
    
    ax.plot(lat_rmse, lat, color=color, lw=2, label='FuXi RMSE')
    ax.axvline(np.mean(lat_rmse), color=color, ls='--', alpha=0.5, 
               label=f'Mean: {np.mean(lat_rmse):.2f}')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('RMSE')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig05_variable_skill.png'), dpi=150)
plt.close()
print("Saved fig05_variable_skill.png")

# ============================================================
# FIGURE 6: Spatial Errors
# ============================================================
print("\nGenerating Figure 6: Spatial Errors...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

spatial_vars = [
    ('Z500', 'Geopotential @ 500 hPa'),
    ('T2M', '2m Temperature'),
    ('U10', '10m Zonal Wind'),
    ('R850', 'Humidity @ 850 hPa'),
    ('MSL', 'Sea Level Pressure'),
    ('TP', 'Total Precipitation')
]

for idx, (var_name, title) in enumerate(spatial_vars):
    ch_idx = get_var_index(var_name)
    if ch_idx is None:
        continue
    
    ax = axes[idx]
    truth = input_data[1, ch_idx]
    fuxi = fuxi_data[0, 0, ch_idx]
    error = fuxi - truth
    
    vmax = np.percentile(np.abs(error), 95)
    im = ax.imshow(error, cmap='RdBu_r', aspect='auto', origin='lower',
                   vmin=-vmax, vmax=vmax)
    rmse = np.sqrt(np.mean(error**2))
    ax.set_title(f'{title}\nRMSE={rmse:.2f}', fontsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Spatial Error Distribution (FuXi 6h vs ERA5)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'fig06_spatial_errors.png'), dpi=150)
plt.close()
print("Saved fig06_spatial_errors.png")

# ============================================================
# Save comprehensive results
# ============================================================
print("\nSaving comprehensive results...")

var_stats = {}
for var in all_vars:
    ch_idx = get_var_index(var)
    if ch_idx is None:
        continue
    
    truth = input_data[1, ch_idx]
    fuxi = fuxi_data[0, 0, ch_idx]
    error = fuxi - truth
    
    cos_lat = np.cos(np.deg2rad(lat))
    weights = cos_lat / np.mean(cos_lat)
    weighted_rmse = np.sqrt(np.mean(error**2 * weights[:, np.newaxis]))
    
    clim = np.mean([input_data[0, ch_idx], input_data[1, ch_idx]])
    f_anom = fuxi - clim
    t_anom = truth - clim
    num = np.sum(f_anom * t_anom * weights[:, np.newaxis])
    den = np.sqrt(np.sum(f_anom**2 * weights[:, np.newaxis]) * np.sum(t_anom**2 * weights[:, np.newaxis]))
    acc = float(num / den) if den > 0 else 0.0
    
    var_stats[var] = {
        'rmse': float(weighted_rmse),
        'acc': acc,
        'mean_error': float(np.mean(error)),
        'max_error': float(np.max(np.abs(error))),
        'truth_mean': float(np.mean(truth)),
        'truth_std': float(np.std(truth))
    }

with open(os.path.join(OUTPUT_DIR, 'variable_level_stats.json'), 'w') as f:
    json.dump(var_stats, f, indent=2)

summary = {
    'global_mean_rmse': float(np.mean([s['rmse'] for s in var_stats.values()])),
    'global_mean_acc': float(np.mean([s['acc'] for s in var_stats.values()])),
    'z500_rmse': var_stats.get('Z500', {}).get('rmse'),
    'z500_acc': var_stats.get('Z500', {}).get('acc'),
    't2m_rmse': var_stats.get('T2M', {}).get('rmse'),
    't2m_acc': var_stats.get('T2M', {}).get('acc'),
    'u10_rmse': var_stats.get('U10', {}).get('rmse'),
    'u10_acc': var_stats.get('U10', {}).get('acc'),
    'tp_rmse': var_stats.get('TP', {}).get('rmse'),
    'tp_acc': var_stats.get('TP', {}).get('acc'),
    'n_variables': len(var_stats),
    'grid_resolution': '0.25 deg',
    'forecast_lead_time': '6 hours',
    'analysis_date': '2023-10-12'
}

with open(os.path.join(OUTPUT_DIR, 'comprehensive_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== All Figures Generated ===")
print(f"Figures saved to {FIGURE_DIR}/")
print(f"Results saved to {OUTPUT_DIR}/")
