#!/usr/bin/env python3
"""
Generate all publication-quality figures for the MMGA research report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_000_20260416_182016"
IMG_DIR = os.path.join(WORKSPACE, 'report/images')
os.makedirs(IMG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(WORKSPACE, 'code'))
from data_loading import load_nasa_data, load_cs2_36_data, load_oxford_data

# Load saved results
val_data = np.load(os.path.join(WORKSPACE, 'outputs/validation_data.npz'), allow_pickle=True)
with open(os.path.join(WORKSPACE, 'outputs/results.json')) as f:
    results = json.load(f)
with open(os.path.join(WORKSPACE, 'outputs/sensitivity_analysis.json')) as f:
    sens_data = json.load(f)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150
})

# ============================================================
# Figure 1: Data Overview - All Three Datasets
# ============================================================
print("Generating Figure 1: Data Overview...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# CS2_36
cs2 = load_cs2_36_data(0)
cycles = np.unique(cs2['cycle_index'])
colors_cs2 = plt.cm.viridis(np.linspace(0, 1, min(10, len(cycles))))
ax = axes[0, 0]
for ci, cyc in enumerate(cycles[:10]):
    mask = (cs2['cycle_index'] == cyc) & (cs2['current'] < -0.01)
    if mask.sum() > 5:
        t = cs2['time'][mask]
        v = cs2['voltage'][mask]
        ax.plot((t - t[0])/60, v, color=colors_cs2[ci], linewidth=0.8, label=f'Cycle {cyc}')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(a) CS2_36 Discharge Curves')
ax.legend(fontsize=7, ncol=2)
ax.set_ylim([2.5, 4.3])

# NASA B0005
nasa = load_nasa_data('B0005')
colors_nasa = plt.cm.plasma(np.linspace(0, 1, 8))
ax = axes[0, 1]
for ci, idx in enumerate([0, 20, 50, 80, 100, 120, 140, 160]):
    if idx < len(nasa):
        c = nasa[idx]
        ax.plot(c['time']/60, c['voltage'], color=colors_nasa[ci], linewidth=0.8, 
                label=f'Cycle {idx}')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(b) NASA B0005 Aging Discharge')
ax.legend(fontsize=7, ncol=2)
ax.set_ylim([2.3, 4.3])

# Oxford dynamic
ox = load_oxford_data()
ax = axes[0, 2]
ox_t = ox['discharge']['time'] - ox['discharge']['time'][0]
ax.plot(ox_t/60, ox['discharge']['voltage'], 'b-', linewidth=0.5)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(c) Oxford Dynamic Discharge')
ax.set_ylim([3.4, 4.3])

# Temperature profiles
ax = axes[1, 0]
for ci, cyc in enumerate(cycles[:5]):
    mask = (cs2['cycle_index'] == cyc) & (cs2['current'] < -0.01)
    if mask.sum() > 5:
        t = cs2['time'][mask]
        v = cs2['voltage'][mask]
        cap = abs(cs2['current'][mask]) * (t - t[0]) / 3600
        ax.plot(cap, v, color=colors_cs2[ci], linewidth=0.8, label=f'Cycle {cyc}')
ax.set_xlabel('Capacity (Ah)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(d) CS2_36 V vs Capacity')
ax.legend(fontsize=7)

# NASA temperature
ax = axes[1, 1]
for ci, idx in enumerate([0, 50, 100, 150]):
    if idx < len(nasa):
        c = nasa[idx]
        ax.plot(c['time']/60, c['temperature'], color=colors_nasa[ci*2], linewidth=0.8,
                label=f'Cycle {idx}')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('(e) NASA B0005 Temperature')
ax.legend(fontsize=7)

# NASA capacity fade
ax = axes[1, 2]
caps = [nasa[i]['capacity'] for i in range(len(nasa)) if nasa[i]['capacity'] is not None]
ax.plot(range(len(caps)), caps, 'ro-', markersize=2, linewidth=0.8)
ax.set_xlabel('Cycle Number')
ax.set_ylabel('Capacity (Ah)')
ax.set_title('(f) NASA B0005 Capacity Fade')
ax.axhline(y=1.4, color='k', linestyle='--', linewidth=0.5, label='EOL (1.4 Ah)')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_data_overview.png'))
plt.close()
print("  Saved fig1_data_overview.png")

# ============================================================
# Figure 2: LHS Parameter Space Sampling
# ============================================================
print("Generating Figure 2: LHS Sampling...")
train_data = np.load(os.path.join(WORKSPACE, 'outputs/training_data.npz'), allow_pickle=True)
X = train_data['X']
param_names = list(train_data['param_names'])

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()
for j in range(min(15, X.shape[1])):
    ax = axes[j]
    ax.hist(X[:, j], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_title(param_names[j], fontsize=9)
    ax.tick_params(labelsize=7)
    if param_names[j] in ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']:
        ax.set_xscale('log')

plt.suptitle('LHS Parameter Space Sampling Distribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_lhs_sampling.png'))
plt.close()
print("  Saved fig2_lhs_sampling.png")

# ============================================================
# Figure 3: ANN Training Curves
# ============================================================
print("Generating Figure 3: ANN Training...")
train_losses = val_data['train_losses']
val_losses_arr = val_data['val_losses']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(train_losses, 'b-', linewidth=1, label='Training Loss', alpha=0.8)
ax.plot(val_losses_arr, 'r-', linewidth=1, label='Validation Loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('(a) ANN Training and Validation Loss')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# ANN prediction accuracy (parity plot)
ax = axes[1]
Y = train_data['Y'] if 'Y' in train_data else None
if Y is not None:
    # Load ANN and predict
    try:
        from mmga_framework import ANNMetaModel
        ann = ANNMetaModel(X.shape[1], Y.shape[1], hidden_layers=[128, 256, 128])
        ann.load(os.path.join(WORKSPACE, 'outputs/ann_model.pt'))
        Y_pred = ann.predict(X)
        
        # Voltage parity
        v_true = Y[:, :50].flatten()
        v_pred = Y_pred[:, :50].flatten()
        
        # Subsample for plotting
        idx = np.random.choice(len(v_true), min(5000, len(v_true)), replace=False)
        ax.scatter(v_true[idx], v_pred[idx], s=1, alpha=0.3, c='steelblue')
        vmin, vmax = v_true.min(), v_true.max()
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
        ax.set_xlabel('ECAT Model Voltage (V)')
        ax.set_ylabel('ANN Predicted Voltage (V)')
        ax.set_title('(b) ANN Prediction Parity Plot')
        
        rmse = np.sqrt(np.mean((v_true - v_pred)**2))
        ax.text(0.05, 0.95, f'RMSE = {rmse*1000:.1f} mV', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        ax.text(0.5, 0.5, f'ANN loading error:\n{e}', transform=ax.transAxes,
                ha='center', va='center')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_ann_training.png'))
plt.close()
print("  Saved fig3_ann_training.png")

# ============================================================
# Figure 4: GA Convergence
# ============================================================
print("Generating Figure 4: GA Convergence...")
fitness_history = val_data['fitness_history']
avg_history = val_data['avg_history']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(fitness_history, 'b-', linewidth=1.5, label='Best Fitness')
ax.plot(avg_history, 'r-', linewidth=1, alpha=0.5, label='Average Fitness')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness (Lower is Better)')
ax.set_title('(a) MMGA Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Direct GA comparison
ax = axes[1]
direct_history = val_data['direct_history']
mmga_time = results['computation_time']['MMGA_optimization_s']
direct_time = results['computation_time']['DirectGA_s']

bars = ['MMGA\n(ANN+GA)', 'Direct GA']
times = [results['computation_time']['MMGA_total_s'], direct_time]
colors = ['steelblue', 'coral']
ax.bar(bars, times, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Computation Time (s)')
ax.set_title('(b) Computation Time Comparison')
for i, t in enumerate(times):
    ax.text(i, t + 0.5, f'{t:.1f}s', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_ga_convergence.png'))
plt.close()
print("  Saved fig4_ga_convergence.png")

# ============================================================
# Figure 5: CS2_36 Validation
# ============================================================
print("Generating Figure 5: CS2_36 Validation...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
t_exp = val_data['cs2_time_exp']
v_exp = val_data['cs2_voltage_exp']
t_sim = val_data['cs2_time_sim']
v_sim = val_data['cs2_voltage_sim']

ax.plot(t_exp/60, v_exp, 'ro', markersize=3, label='Experimental', alpha=0.7)
ax.plot(t_sim/60, v_sim, 'b-', linewidth=1.5, label='MMGA Identified Model')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(a) CS2_36 Voltage Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

rmse = results['validation_metrics']['CS2_36']['RMSE_mV']
ax.text(0.05, 0.05, f'RMSE = {rmse:.1f} mV', transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Voltage error
ax = axes[1]
common_t = np.linspace(0, min(t_exp[-1], t_sim[-1]), 200)
v_exp_i = np.interp(common_t, t_exp, v_exp)
v_sim_i = np.interp(common_t, t_sim, v_sim)
error = (v_sim_i - v_exp_i) * 1000  # mV

ax.plot(common_t/60, error, 'g-', linewidth=1)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.fill_between(common_t/60, error, alpha=0.3, color='green')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage Error (mV)')
ax.set_title('(b) CS2_36 Voltage Error')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_cs2_validation.png'))
plt.close()
print("  Saved fig5_cs2_validation.png")

# ============================================================
# Figure 6: NASA Validation
# ============================================================
print("Generating Figure 6: NASA Validation...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Voltage comparison
ax = axes[0]
t_exp_n = val_data['nasa_time_exp']
v_exp_n = val_data['nasa_voltage_exp']
t_sim_n = val_data['nasa_time_sim']
v_sim_n = val_data['nasa_voltage_sim']

ax.plot(t_exp_n/60, v_exp_n, 'ro', markersize=2, label='Experimental', alpha=0.7)
ax.plot(t_sim_n/60, v_sim_n, 'b-', linewidth=1.5, label='MMGA Model')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(a) NASA B0005 Voltage')
ax.legend()
ax.grid(True, alpha=0.3)
rmse_n = results['validation_metrics']['NASA_B0005']['RMSE_mV']
ax.text(0.05, 0.05, f'RMSE = {rmse_n:.1f} mV', transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Temperature comparison
ax = axes[1]
temp_exp_n = val_data['nasa_temp_exp']
temp_sim_n = val_data['nasa_temp_sim']
ax.plot(t_exp_n/60, temp_exp_n, 'ro', markersize=2, label='Experimental', alpha=0.7)
ax.plot(t_sim_n/60, temp_sim_n, 'b-', linewidth=1.5, label='MMGA Model')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('(b) NASA B0005 Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

# Capacity degradation
ax = axes[2]
caps = [nasa[i]['capacity'] for i in range(len(nasa)) if nasa[i]['capacity'] is not None]
ax.plot(range(len(caps)), caps, 'ro-', markersize=2, linewidth=0.8, label='Experimental')
if len(val_data['aging_cap_sim']) > 0:
    ax.plot(val_data['aging_cycles'], val_data['aging_cap_sim'], 'bs-', 
            markersize=3, linewidth=0.8, label='Model')
ax.set_xlabel('Discharge Cycle')
ax.set_ylabel('Capacity (Ah)')
ax.set_title('(c) Capacity Degradation')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_nasa_validation.png'))
plt.close()
print("  Saved fig6_nasa_validation.png")

# ============================================================
# Figure 7: Oxford Dynamic Validation
# ============================================================
print("Generating Figure 7: Oxford Validation...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ox_t = val_data['ox_time']
ox_v_exp = val_data['ox_voltage_exp']
ox_v_sim = val_data['ox_voltage_sim']
ox_temp_exp = val_data['ox_temp_exp']
ox_temp_sim = val_data['ox_temp_sim']

# Full voltage comparison
ax = axes[0, 0]
ax.plot(ox_t/60, ox_v_exp, 'r-', linewidth=0.5, label='Experimental', alpha=0.7)
ax.plot(ox_t/60, ox_v_sim, 'b-', linewidth=0.5, label='MMGA Model', alpha=0.7)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(a) Oxford Dynamic Voltage - Full Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Zoomed section
ax = axes[0, 1]
zoom_mask = (ox_t > 500) & (ox_t < 1500)
ax.plot(ox_t[zoom_mask]/60, ox_v_exp[zoom_mask], 'r-', linewidth=1, label='Experimental')
ax.plot(ox_t[zoom_mask]/60, ox_v_sim[zoom_mask], 'b--', linewidth=1, label='MMGA Model')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('(b) Oxford Voltage - Zoomed')
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature
ax = axes[1, 0]
ax.plot(ox_t/60, ox_temp_exp, 'r-', linewidth=0.5, label='Experimental', alpha=0.7)
ax.plot(ox_t/60, ox_temp_sim, 'b-', linewidth=0.5, label='MMGA Model', alpha=0.7)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('(c) Oxford Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

# Current profile
ox_full = load_oxford_data()
ax = axes[1, 1]
ox_current = ox_full['discharge']['current']
ax.plot(ox_t/60, ox_current/1000, 'k-', linewidth=0.3)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Current (A)')
ax.set_title('(d) Oxford Dynamic Current Profile')
ax.grid(True, alpha=0.3)

rmse_ox = results['validation_metrics']['Oxford_dynamic']['RMSE_mV']
fig.suptitle(f'Oxford Battery Degradation Dataset - Dynamic Validation (RMSE = {rmse_ox:.1f} mV)', 
             fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_oxford_validation.png'))
plt.close()
print("  Saved fig7_oxford_validation.png")

# ============================================================
# Figure 8: Parameter Sensitivity Analysis
# ============================================================
print("Generating Figure 8: Sensitivity Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
ax = axes[0]
param_list = sorted(sens_data.keys(), key=lambda x: -sens_data[x]['voltage'])
v_sens = [sens_data[p]['voltage'] for p in param_list]
t_sens = [sens_data[p]['temperature'] for p in param_list]

x_pos = np.arange(len(param_list))
width = 0.35
ax.barh(x_pos - width/2, v_sens, width, label='Voltage Sensitivity', color='steelblue')
ax.barh(x_pos + width/2, t_sens, width, label='Temperature Sensitivity', color='coral')
ax.set_yticks(x_pos)
ax.set_yticklabels(param_list, fontsize=8)
ax.set_xlabel('Sensitivity Index')
ax.set_title('(a) Parameter Sensitivity Ranking')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Heatmap
ax = axes[1]
categories = ['voltage', 'capacity', 'temperature']
sens_matrix = np.zeros((len(param_list), len(categories)))
for i, p in enumerate(param_list):
    for j, cat in enumerate(categories):
        sens_matrix[i, j] = sens_data[p][cat]

# Normalize each column
for j in range(len(categories)):
    col_max = sens_matrix[:, j].max()
    if col_max > 0:
        sens_matrix[:, j] /= col_max

im = ax.imshow(sens_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(['Voltage', 'Capacity', 'Temperature'])
ax.set_yticks(range(len(param_list)))
ax.set_yticklabels(param_list, fontsize=8)
ax.set_title('(b) Normalized Sensitivity Heatmap')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig8_sensitivity_analysis.png'))
plt.close()
print("  Saved fig8_sensitivity_analysis.png")

# ============================================================
# Figure 9: Identified Parameters Summary
# ============================================================
print("Generating Figure 9: Parameters Summary...")
from ecat_model import get_parameter_bounds

bounds = get_parameter_bounds()
id_params = results['identified_parameters']

fig, ax = plt.subplots(figsize=(12, 7))

param_list = list(id_params.keys())
n_params = len(param_list)

# Normalize all parameters to [0, 1] range within their bounds
norm_vals = []
norm_lbs = []
norm_ubs = []
labels = []

for pname in param_list:
    lb, ub = bounds[pname]
    val = id_params[pname]
    
    # Use log scale for parameters spanning orders of magnitude
    log_params = ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']
    if pname in log_params:
        norm_val = (np.log10(val) - np.log10(lb)) / (np.log10(ub) - np.log10(lb))
    else:
        norm_val = (val - lb) / (ub - lb)
    
    norm_vals.append(norm_val)
    labels.append(pname)

y_pos = np.arange(n_params)
colors = plt.cm.RdYlGn_r(np.array(norm_vals))

ax.barh(y_pos, norm_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Normalized Parameter Value (within bounds)')
ax.set_title('Identified Parameters (Normalized to Search Bounds)')
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')

# Add actual values as text
for i, pname in enumerate(labels):
    val = id_params[pname]
    if abs(val) < 0.01 or abs(val) > 1000:
        txt = f'{val:.2e}'
    else:
        txt = f'{val:.4f}'
    ax.text(max(norm_vals[i] + 0.02, 0.02), i, txt, va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig9_identified_params.png'))
plt.close()
print("  Saved fig9_identified_params.png")

# ============================================================
# Figure 10: MMGA Framework Schematic (conceptual)
# ============================================================
print("Generating Figure 10: MMGA Framework...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Boxes
boxes = [
    (1, 3, 2.5, 1.5, 'LHS\nSampling', 'lightblue'),
    (4.5, 3, 2.5, 1.5, 'ECAT Model\nSimulation', 'lightyellow'),
    (8, 3, 2.5, 1.5, 'ANN\nMeta-Model\nTraining', 'lightgreen'),
    (8, 0.5, 2.5, 1.5, 'Genetic\nAlgorithm\nOptimization', 'lightsalmon'),
    (4.5, 0.5, 2.5, 1.5, 'Identified\nParameters', 'plum'),
    (1, 0.5, 2.5, 1.5, 'Experimental\nData', 'lightcoral'),
    (11.5, 1.5, 2, 1.5, 'Validation', 'lightcyan'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5)
ax.annotate('', xy=(4.5, 3.75), xytext=(3.5, 3.75), arrowprops=arrow_props)
ax.annotate('', xy=(8, 3.75), xytext=(7, 3.75), arrowprops=arrow_props)
ax.annotate('', xy=(9.25, 3), xytext=(9.25, 2), arrowprops=arrow_props)
ax.annotate('', xy=(8, 1.25), xytext=(7, 1.25), arrowprops=arrow_props)
ax.annotate('', xy=(4.5, 1.25), xytext=(3.5, 1.25), arrowprops=arrow_props)
ax.annotate('', xy=(11.5, 2.25), xytext=(10.5, 2.25), arrowprops=arrow_props)

# Labels
ax.text(7, 5.5, 'MMGA: Meta-Model based Genetic Algorithm Framework', 
        ha='center', fontsize=14, fontweight='bold')
ax.text(3.8, 4.3, 'Parameter\nSamples', ha='center', fontsize=8, style='italic')
ax.text(7.3, 4.3, 'Simulation\nResults', ha='center', fontsize=8, style='italic')
ax.text(10, 2.5, 'Trained\nANN', ha='center', fontsize=8, style='italic')
ax.text(7.3, 0.9, 'Fitness\nEvaluation', ha='center', fontsize=8, style='italic')
ax.text(3.8, 0.9, 'Target\nMatching', ha='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig10_mmga_framework.png'))
plt.close()
print("  Saved fig10_mmga_framework.png")

# ============================================================
# Figure 11: Multi-dataset Comparison Summary
# ============================================================
print("Generating Figure 11: Comparison Summary...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = ['CS2_36', 'NASA_B0005', 'Oxford_dynamic']
rmse_vals = [results['validation_metrics'][d]['RMSE_mV'] for d in datasets]
mae_vals = [results['validation_metrics'][d]['MAE_mV'] for d in datasets]

# RMSE comparison
ax = axes[0]
x = np.arange(3)
width = 0.35
bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE', color='coral', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(['CS2_36\n(CC)', 'NASA\n(CC)', 'Oxford\n(Dynamic)'], fontsize=9)
ax.set_ylabel('Error (mV)')
ax.set_title('(a) Validation Error Metrics')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{bar.get_height():.0f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.0f}', ha='center', fontsize=8)

# Timing breakdown
ax = axes[1]
time_labels = ['LHS\nGeneration', 'ANN\nTraining', 'GA\nOptimization']
time_vals = [results['computation_time']['LHS_generation_s'],
             results['computation_time']['ANN_training_s'],
             results['computation_time']['MMGA_optimization_s']]
colors = ['steelblue', 'coral', 'seagreen']
bars = ax.bar(time_labels, time_vals, color=colors, edgecolor='black')
ax.set_ylabel('Time (s)')
ax.set_title('(b) MMGA Pipeline Time Breakdown')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}s', ha='center', fontsize=9)

# Speedup
ax = axes[2]
total_mmga = sum(time_vals)
direct = results['computation_time']['DirectGA_s']
# Scale direct GA to equivalent population/generations
direct_scaled = direct * (300/30) * (150/20)  # scale to same pop/gen
ax.bar(['MMGA\n(Total)', 'Direct GA\n(Scaled)'], [total_mmga, direct_scaled],
       color=['steelblue', 'coral'], edgecolor='black')
ax.set_ylabel('Estimated Time (s)')
ax.set_title('(c) Computation Efficiency')
ax.grid(True, alpha=0.3, axis='y')
speedup = direct_scaled / total_mmga
ax.text(0.5, 0.9, f'Speedup: ~{speedup:.0f}x', transform=ax.transAxes,
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig11_comparison_summary.png'))
plt.close()
print("  Saved fig11_comparison_summary.png")

# ============================================================
# Figure 12: NASA Multi-Battery Comparison
# ============================================================
print("Generating Figure 12: NASA Multi-Battery...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for bi, (bid, ax) in enumerate(zip(['B0005', 'B0006', 'B0007', 'B0018'], axes.flatten())):
    nasa_bat = load_nasa_data(bid)
    
    # Plot first discharge cycle
    if len(nasa_bat) > 0:
        c = nasa_bat[0]
        ax.plot(c['time']/60, c['voltage'], 'ro', markersize=2, label='Experimental', alpha=0.7)
        
        # Simulate
        from ecat_model import ECATModel
        model = ECATModel(results['identified_parameters'])
        model.params['Q_nom'] = c['capacity'] if c['capacity'] else 2.0
        model.params['T_amb'] = c['temperature'][0] + 273.15
        
        I_app = abs(c['current'].mean())
        sim = model.simulate_cc_discharge(I_app=I_app, t_end=5000, dt=1.0, V_cutoff=2.5)
        ax.plot(sim['time']/60, sim['voltage'], 'b-', linewidth=1.5, label='MMGA Model')
        
        # Compute RMSE
        ct = np.linspace(0, min(c['time'][-1], sim['time'][-1]), 100)
        ve = np.interp(ct, c['time'], c['voltage'])
        vs = np.interp(ct, sim['time'], sim['voltage'])
        rmse = np.sqrt(np.mean((ve - vs)**2))
        
        ax.set_title(f'{bid} (RMSE = {rmse*1000:.1f} mV)')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Voltage (V)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('NASA Battery Validation - Multiple Cells', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig12_nasa_multi_battery.png'))
plt.close()
print("  Saved fig12_nasa_multi_battery.png")

print("\n" + "=" * 60)
print("All figures generated successfully!")
print("=" * 60)
