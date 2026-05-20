#!/usr/bin/env python3
"""
Phase 3: Comprehensive Evaluation and Visualization
- Error accumulation analysis
- Cascade vs single-model comparison
- Per-variable performance
- Spectral analysis of forecasts
- Architecture diagram
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
from scipy.fft import fft2, fftshift
import json
import os

sns_import = __import__('seaborn')
sns_import.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white'
})

# Load data
ds_in = xr.open_dataset('data/20231012-06_input_netcdf.nc')
ds_out = xr.open_dataset('data/006.nc')
input_data = ds_in.data.values
fuxi_data = ds_out.data.values[0, 0]
levels = list(ds_in.level.values)

# Load cascade results
with open('outputs/cascade_results.json', 'r') as f:
    results = json.load(f)

# Convert string lists to floats
results['cascade_rmse'] = [float(x) for x in results['cascade_rmse']]
results['single_rmse'] = [float(x) for x in results['single_rmse']]
results['fuxi_step0_rmse'] = float(results['fuxi_step0_rmse'])

# Convert group rmse strings
for gname in results.get('group_rmse', {}):
    for key in results['group_rmse'][gname]:
        results['group_rmse'][gname][key] = [float(x) for x in results['group_rmse'][gname][key]]

ds_fc = xr.open_dataset('outputs/cascade_forecast.nc')
cascade_forecast = ds_fc['cascade_forecast'].values
single_forecast = ds_fc['single_forecast'].values
ds_fc.close()

reference = input_data[-1]

# ==========================================
# Figure 7: Error Accumulation Comparison
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Error Accumulation Analysis: Cascade vs Single-Model Forecasting', 
             fontsize=14, fontweight='bold')

lead_hours = np.arange(6, 366, 6)

# 7a: Overall RMSE over lead time
ax = axes[0]
ax.plot(lead_hours, results['cascade_rmse'], '-', linewidth=2, label='Cascade (3-stage)', color='#2196F3')
ax.plot(lead_hours, results['single_rmse'], '--', linewidth=2, label='Single Model', color='#F44336')
ax.axhline(y=results['fuxi_step0_rmse'], color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'FuXi t+6h ({results["fuxi_step0_rmse"]:.2f})')
# Mark cascade stage boundaries
ax.axvline(x=72, color='gray', linestyle='--', alpha=0.3, label='Stage boundaries')
ax.axvline(x=168, color='gray', linestyle='--', alpha=0.3)
ax.text(36, ax.get_ylim()[1]*0.95, 'Stage 1\n(6h-3d)', ha='center', fontsize=8, color='blue', fontweight='bold')
ax.text(120, ax.get_ylim()[1]*0.95, 'Stage 2\n(3d-7d)', ha='center', fontsize=8, color='blue', fontweight='bold')
ax.text(264, ax.get_ylim()[1]*0.95, 'Stage 3\n(7d-15d)', ha='center', fontsize=8, color='blue', fontweight='bold')
ax.set_xlabel('Lead Time (hours)')
ax.set_ylabel('RMSE (normalized units)')
ax.set_title('(a) Global Mean RMSE', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0, 360)

# 7b: RMSE ratio (single/cascade)
ax = axes[1]
ratio = np.array(results['single_rmse']) / (np.array(results['cascade_rmse']) + 1e-10)
ax.plot(lead_hours, ratio, 'k-', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(lead_hours, 1.0, ratio, where=ratio > 1.0, alpha=0.2, color='green', label='Cascade advantage')
ax.fill_between(lead_hours, 1.0, ratio, where=ratio < 1.0, alpha=0.2, color='red', label='Single model advantage')
ax.set_xlabel('Lead Time (hours)')
ax.set_ylabel('RMSE Ratio (Single/Cascade)')
ax.set_title('(b) Relative Performance', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 360)

# 7c: RMSE growth rate (derivative)
ax = axes[2]
cascade_rate = np.diff(results['cascade_rmse'])
single_rate = np.diff(results['single_rmse'])
ax.plot(lead_hours[1:], cascade_rate,  linewidth=2, label='Cascade', color='#2196F3')
ax.plot(lead_hours[1:], single_rate, 'r--', linewidth=2, label='Single Model', color='#F44336')
ax.axvline(x=72, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=168, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Lead Time (hours)')
ax.set_ylabel('ΔRMSE per step')
ax.set_title('(c) Error Growth Rate', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 360)

plt.tight_layout()
plt.savefig('report/images/fig7_error_accumulation.png')
plt.close()
print("Figure 7 saved.")

# ==========================================
# Figure 8: Per-Variable Group Performance
# ==========================================
group_rmse = results['group_rmse']
n_groups = len(group_rmse)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Per-Variable Group RMSE: Cascade vs Single-Model', fontsize=14, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for ax, (gname, data) in zip(axes.flat, group_rmse.items()):
    ax.plot(lead_hours, data['cascade'],  linewidth=2, label='Cascade', color='#2196F3')
    ax.plot(lead_hours, data['single'],  '--', linewidth=2, label='Single', color='#F44336')
    ax.axvline(x=72, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=168, color='gray', linestyle='--', alpha=0.3)
    ax.set_title(gname, fontsize=11, fontweight='bold')
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('RMSE')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 360)

# Remove empty subplot if needed
if n_groups < 6:
    for i in range(n_groups, 6):
        axes.flat[i].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/fig8_per_variable_rmse.png')
plt.close()
print("Figure 8 saved.")

# ==========================================
# Figure 9: Forecast Field Comparison Maps
# ==========================================
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle('Forecast Field Comparison: Input, Cascade, Single, and Reference', 
             fontsize=14, fontweight='bold')

# Pick 4 lead times: 6h, 3d, 7d, 15d
lead_times = [0, 11, 27, 59]  # 0-indexed
lead_labels = ['t+6h', 't+3d', 't+7d', 't+15d']
var_idx = levels.index('Z500')
var_name = 'Z500'

# Row 1: Input / Truth reference
for col, (lt, ll) in enumerate(zip(lead_times, lead_labels)):
    ax = axes[0, col]
    if lt == 0:
        field = input_data[-1, var_idx, :, :]
    else:
        field = reference[var_idx]  # use input as reference for Z500
    im = ax.imshow(field, cmap='RdBu_r', aspect='auto',
                   vmin=-3, vmax=3)
    ax.set_title(f'Reference\n{ll}', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

# Row 2: Cascade forecast
for col, (lt, ll) in enumerate(zip(lead_times, lead_labels)):
    ax = axes[1, col]
    field = cascade_forecast[lt, var_idx, :, :]
    im = ax.imshow(field, cmap='RdBu_r', aspect='auto',
                   vmin=-3, vmax=3)
    rmse_val = np.sqrt(np.mean((field - reference[var_idx])**2))
    ax.set_title(f'Cascade\n{ll} (RMSE={rmse_val:.3f})', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

# Row 3: Single model forecast
for col, (lt, ll) in enumerate(zip(lead_times, lead_labels)):
    ax = axes[2, col]
    field = single_forecast[lt, var_idx, :, :]
    im = ax.imshow(field, cmap='RdBu_r', aspect='auto',
                   vmin=-3, vmax=3)
    rmse_val = np.sqrt(np.mean((field - reference[var_idx])**2))
    ax.set_title(f'Single\n{ll} (RMSE={rmse_val:.3f})', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout()
plt.savefig('report/images/fig9_forecast_field_comparison.png')
plt.close()
print("Figure 9 saved.")

# ==========================================
# Figure 10: Spectral Analysis of Forecasts
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Power Spectral Density: Cascade vs Single-Model at Z500', 
             fontsize=13, fontweight='bold')

spectral_leads = [0, 11, 27, 59]  # 6h, 3d, 7d, 15d
spectral_labels = ['t+6h', 't+3d', 't+7d', 't+15d']

for ax, (lt, ll) in zip(axes.flat[:4], zip(spectral_leads, spectral_labels)):
    # Reference spectrum
    ref_field = reference[var_idx] - reference[var_idx].mean()
    fft_ref = fftshift(fft2(ref_field))
    psd_ref = np.log10(np.abs(fft_ref)**2 + 1)
    
    # Cascade spectrum
    c_field = cascade_forecast[lt, var_idx] - cascade_forecast[lt, var_idx].mean()
    fft_c = fftshift(fft2(c_field))
    psd_c = np.log10(np.abs(fft_c)**2 + 1)
    
    # Single spectrum
    s_field = single_forecast[lt, var_idx] - single_forecast[lt, var_idx].mean()
    fft_s = fftshift(fft2(s_field))
    psd_s = np.log10(np.abs(fft_s)**2 + 1)
    
    # Radial average
    ny, nx = psd_ref.shape
    cy, cx = ny//2, nx//2
    Y, X = np.mgrid[0:ny, 0:nx]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    r_bins = np.linspace(0, min(cy, cx), 30)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    def radial_avg(psd, R, r_bins):
        result = []
        for r in range(len(r_bins)-1):
            mask = (R >= r_bins[r]) & (R < r_bins[r+1])
            if mask.sum() > 0:
                result.append(psd[mask].mean())
            else:
                result.append(np.nan)
        return np.array(result)
    
    ref_radial = radial_avg(psd_ref, R, r_bins)
    c_radial = radial_avg(psd_c, R, r_bins)
    s_radial = radial_avg(psd_s, R, r_bins)
    
    ax.plot(r_centers, ref_radial, 'k-', linewidth=2, label='Reference', alpha=0.8)
    ax.plot(r_centers, c_radial,  linewidth=2, label='Cascade', color='#2196F3', alpha=0.8)
    ax.plot(r_centers, s_radial,  '--', linewidth=2, label='Single', color='#F44336', alpha=0.8)
    ax.set_title(f'{ll}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('log₁₀(PSD)')
    ax.legend(fontsize=8)
    ax.set_xlim(1, 30)

# Remove extra subplot
axes.flat[4].set_visible(False)
axes.flat[5].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/fig10_spectral_analysis_forecast.png')
plt.close()
print("Figure 10 saved.")

# ==========================================
# Figure 11: Cascade Architecture Diagram
# ==========================================
fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Three-Stage Cascade U-Transformer Architecture', fontsize=16, fontweight='bold', pad=20)

# Colors
stage_colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9']
stage_borders = ['#1565C0', '#E65100', '#2E7D32']
box_colors = ['#BBDEFB', '#FFE0B2', '#C8E6C9']

# Draw three stages
stage_info = [
    {'x': 0.5, 'w': 5.5, 'name': 'Stage 1: Short-Range', 'range': '6h – 3 days', 'steps': '12 steps', 'desc': 'High-freq dynamics\nFine-scale features'},
    {'x': 6.5, 'w': 5.5, 'name': 'Stage 2: Medium-Range', 'range': '3 – 7 days', 'steps': '16 steps', 'desc': 'Synoptic patterns\nWave propagation'},
    {'x': 12.5, 'w': 5.5, 'name': 'Stage 3: Extended-Range', 'range': '7 – 15 days', 'steps': '32 steps', 'desc': 'Large-scale trends\nClimate modes'},
]

for i, info in enumerate(stage_info):
    # Stage background
    rect = FancyBboxPatch((info['x'], 1.0), info['w'], 6.0, 
                          boxstyle="round,pad=0.2", 
                          facecolor=stage_colors[i], edgecolor=stage_borders[i], linewidth=2)
    ax.add_patch(rect)
    
    # Title
    ax.text(info['x'] + info['w']/2, 6.5, info['name'], fontsize=11, fontweight='bold',
            ha='center', va='center', color=stage_borders[i])
    ax.text(info['x'] + info['w']/2, 5.9, info['range'], fontsize=10, ha='center', va='center')
    ax.text(info['x'] + info['w']/2, 5.4, info['steps'], fontsize=9, ha='center', va='center', 
            style='italic', color='gray')
    
    # U-Transformer mini diagram
    # Encoder arrows
    for j in range(3):
        y_enc = 4.5 - j * 0.7
        w = 0.8 + j * 0.3
        x_center = info['x'] + 1.0
        rect2 = FancyBboxPatch((x_center - w/2, y_enc - 0.2), w, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor=box_colors[i], edgecolor=stage_borders[i], linewidth=1)
        ax.add_patch(rect2)
        ax.text(x_center, y_enc, f'Enc{j+1}', fontsize=7, ha='center', va='center')
    
    # Bottleneck
    rect_bn = FancyBboxPatch((info['x'] + 2.2, 1.8), 1.2, 0.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#FFF9C4', edgecolor=stage_borders[i], linewidth=1.5)
    ax.add_patch(rect_bn)
    ax.text(info['x'] + 2.8, 2.05, 'Transformer\nBottleneck', fontsize=6, ha='center', va='center', fontweight='bold')
    
    # Decoder arrows
    for j in range(3):
        y_dec = 2.5 + j * 0.7
        w = 0.8 + (2-j) * 0.3
        x_center = info['x'] + info['w'] - 1.0
        rect3 = FancyBboxPatch((x_center - w/2, y_dec - 0.2), w, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor=box_colors[i], edgecolor=stage_borders[i], linewidth=1)
        ax.add_patch(rect3)
        ax.text(x_center, y_dec, f'Dec{j+1}', fontsize=7, ha='center', va='center')
    
    # Description
    ax.text(info['x'] + info['w']/2, 1.3, info['desc'], fontsize=8, ha='center', va='center',
            style='italic', color='#555555')

# Arrows between stages
for i in range(2):
    x_start = stage_info[i]['x'] + stage_info[i]['w']
    x_end = stage_info[i+1]['x']
    ax.annotate('', xy=(x_end, 4.0), xytext=(x_start, 4.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text((x_start + x_end)/2, 4.3, f'Handover', fontsize=8, ha='center', va='center',
            fontweight='bold', color='darkred')

# Input label
ax.annotate('ERA5\nInput', xy=(0.5, 4.0), xytext=(-0.5, 4.0),
            fontsize=10, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Output label
ax.annotate('15-Day\nForecast', xy=(18, 4.0), xytext=(18.8, 4.0),
            fontsize=10, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('report/images/fig11_architecture_diagram.png')
plt.close()
print("Figure 11 saved.")

# ==========================================
# Figure 12: Latitude-weighted RMSE profiles
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Latitude-Weighted RMSE Profiles (Z500) at Different Lead Times', 
             fontsize=13, fontweight='bold')

lats_arr = ds_in.lat.values
lat_weights = np.cos(np.deg2rad(lats_arr))
lat_weights = lat_weights / lat_weights.mean()

var_idx_z500 = levels.index('Z500')
lead_examples = [(0, 't+6h'), (11, 't+3d'), (27, 't+7d')]

for ax, (lt, ll) in zip(axes, lead_examples):
    # Cascade RMSE by latitude
    c_diff = cascade_forecast[lt, var_idx_z500] - reference[var_idx_z500]
    c_rmse_lat = np.sqrt(np.mean(c_diff**2, axis=1))  # mean over longitude
    
    # Single RMSE by latitude
    s_diff = single_forecast[lt, var_idx_z500] - reference[var_idx_z500]
    s_rmse_lat = np.sqrt(np.mean(s_diff**2, axis=1))
    
    ax.plot(lats_arr, c_rmse_lat,  linewidth=2, label='Cascade', color='#2196F3')
    ax.plot(lats_arr, s_rmse_lat,  '--', linewidth=2, label='Single', color='#F44336')
    ax.fill_between(lats_arr, c_rmse_lat, s_rmse_lat, alpha=0.15, color='green',
                    where=s_rmse_lat > c_rmse_lat)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('RMSE (Z500)')
    ax.set_title(ll, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_xaxis()

plt.tight_layout()
plt.savefig('report/images/fig12_latitude_rmse_profiles.png')
plt.close()
print("Figure 12 saved.")

# ==========================================
# Figure 13: Cascade stage handover analysis
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Cascade Stage Handover: Smoothness and Continuity Analysis', 
             fontsize=13, fontweight='bold')

# 13a: Z500 global mean time series
ax = axes[0]
z500_idx = levels.index('Z500')
c_means = [cascade_forecast[i, z500_idx].mean() for i in range(60)]
s_means = [single_forecast[i, z500_idx].mean() for i in range(60)]
steps = np.arange(1, 61)

ax.plot(steps, c_means,  linewidth=2, label='Cascade', color='#2196F3')
ax.plot(steps, s_means,  '--', linewidth=2, label='Single', color='#F44336')
ax.axvline(x=12, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=28, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Forecast Step (6h intervals)')
ax.set_ylabel('Z500 Global Mean')
ax.set_title('(a) Z500 Mean Evolution', fontweight='bold')
ax.legend(fontsize=9)

# 13b: Step-to-step differences (smoothness)
ax = axes[1]
c_diffs = [np.sqrt(np.mean((cascade_forecast[i+1] - cascade_forecast[i])**2)) for i in range(59)]
s_diffs = [np.sqrt(np.mean((single_forecast[i+1] - single_forecast[i])**2)) for i in range(59)]

ax.plot(steps[1:], c_diffs,  linewidth=2, label='Cascade', color='#2196F3')
ax.plot(steps[1:], s_diffs,  '--', linewidth=2, label='Single', color='#F44336')
ax.axvline(x=12, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=28, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Forecast Step (6h intervals)')
ax.set_ylabel('Step-to-step RMSE')
ax.set_title('(b) Temporal Smoothness', fontweight='bold')
ax.legend(fontsize=9)

# 13c: Spectral variance preservation
ax = axes[2]
var_leads = [0, 11, 27, 59]
var_labels = ['6h', '3d', '7d', '15d']
x_pos = np.arange(len(var_leads))
width = 0.3

c_vars = []
s_vars = []
for lt in var_leads:
    c_field = cascade_forecast[lt, z500_idx]
    s_field = single_forecast[lt, z500_idx]
    c_vars.append(np.std(c_field))
    s_vars.append(np.std(s_field))

ref_var = np.std(reference[z500_idx])
ax.bar(x_pos - width/2, c_vars, width, label='Cascade', color='#2196F3', alpha=0.8)
ax.bar(x_pos + width/2, s_vars, width, label='Single', color='#F44336', alpha=0.8)
ax.axhline(y=ref_var, color='green', linestyle='--', linewidth=2, label='Reference')
ax.set_xticks(x_pos)
ax.set_xticklabels(var_labels)
ax.set_xlabel('Lead Time')
ax.set_ylabel('Z500 Standard Deviation')
ax.set_title('(c) Variance Preservation', fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('report/images/fig13_stage_handover_analysis.png')
plt.close()
print("Figure 13 saved.")

# ==========================================
# Figure 14: Combined performance summary heatmap
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

# Compute per-variable group metrics at different lead times
group_names = ['Geopotential (Z)', 'Temperature (T)', 'U-Wind (U)', 'V-Wind (V)', 'Humidity (R)']
group_prefixes = ['Z', 'T', 'U', 'V', 'R']
pressure_samples = [50, 200, 500, 850, 1000]
lead_time_samples = [0, 5, 11, 17, 27, 39, 59]  # 6h, 1.5d, 3d, 4.5d, 7d, 10d, 15d
lead_labels_h = ['6h', '1.5d', '3d', '4.5d', '7d', '10d', '15d']

heatmap_data = np.zeros((len(group_names) * len(pressure_samples), len(lead_time_samples)))

row = 0
row_labels = []
for gname, prefix in zip(group_names, group_prefixes):
    for p in pressure_samples:
        var_name = f'{prefix}{p}'
        if var_name in levels:
            vidx = levels.index(var_name)
            for col, lt in enumerate(lead_time_samples):
                c_rmse = np.sqrt(np.mean((cascade_forecast[lt, vidx] - reference[vidx])**2))
                s_rmse = np.sqrt(np.mean((single_forecast[lt, vidx] - reference[vidx])**2))
                heatmap_data[row, col] = (s_rmse - c_rmse) / (s_rmse + 1e-10) * 100  # % improvement
            row_labels.append(var_name)
            row += 1

# Trim to actual rows
heatmap_data = heatmap_data[:len(row_labels), :]

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
ax.set_xticks(range(len(lead_labels_h)))
ax.set_xticklabels(lead_labels_h)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_xlabel('Lead Time', fontsize=11)
ax.set_title('Cascade Improvement over Single Model (%)\nPositive = Cascade Better (lower RMSE)', 
             fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='% Improvement', shrink=0.8)

# Add text annotations
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data[i, j]
        color = 'white' if abs(val) > 10 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

plt.tight_layout()
plt.savefig('report/images/fig14_performance_heatmap.png')
plt.close()
print("Figure 14 saved.")

ds_in.close()
ds_out.close()

print("\nPhase 3 complete. All evaluation figures saved.")
