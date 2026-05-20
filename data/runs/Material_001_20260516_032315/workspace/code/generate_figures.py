#!/usr/bin/env python3
"""
Generate all figures for the research report.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
img_dir = os.path.join(os.path.dirname(__file__), '..', 'report', 'images')
os.makedirs(img_dir, exist_ok=True)

# ============================================================
# Figure 1: Data Overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Parse raw data again for original shapes
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
with open(data_path, 'r') as f:
    content = f.read()

sections = content.split('# 文件')

# File 1 data
lines1 = [l.strip() for l in sections[1].split('\n') if l.strip().startswith('[')]
features_raw = json.loads(lines1[1])
targets_raw = json.loads(lines1[3])

# File 2 data
lines2 = [l.strip() for l in sections[2].split('\n') if l.strip().startswith('[')]
param_a = json.loads(lines2[0])
param_b = json.loads(lines2[1])

# File 3 data
lines3 = [l.strip() for l in sections[3].split('\n') if l.strip().startswith('[')]
temp_range = json.loads(lines3[0])
time_range = json.loads(lines3[1])

# Plot 1: Features vs Targets scatter
ax = axes[0, 0]
n_align = min(len(features_raw), len(targets_raw))
ax.scatter(features_raw[:n_align], targets_raw, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Structural Feature', fontweight='bold')
ax.set_ylabel('Target Property', fontweight='bold')
ax.set_title('(a) Property Prediction Data', fontweight='bold')
# Add trend line
z = np.polyfit(features_raw[:n_align], targets_raw, 1)
p = np.poly1d(z)
x_line = np.linspace(min(features_raw[:n_align]), max(features_raw[:n_align]), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
ax.legend(fontsize=9)

# Plot 2: Lattice parameter distribution
ax = axes[0, 1]
ax.hist(param_a, bins=20, alpha=0.6, label='Parameter A', color='steelblue', edgecolor='black')
ax.hist(param_b, bins=20, alpha=0.6, label='Parameter B', color='coral', edgecolor='black')
ax.set_xlabel('Lattice Parameter Value', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('(b) Structure Data Distribution', fontweight='bold')
ax.legend(fontsize=9)

# Plot 3: Lattice params 2D
ax = axes[1, 0]
ax.scatter(param_a, param_b, alpha=0.5, s=30, c='purple', edgecolors='black', linewidth=0.3)
ax.set_xlabel('Parameter A', fontweight='bold')
ax.set_ylabel('Parameter B', fontweight='bold')
ax.set_title('(c) Structure Parameter Space', fontweight='bold')
corr = np.corrcoef(param_a, param_b)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Optimization parameter space
ax = axes[1, 1]
ax.barh(['Temperature (K)', 'Time (h)'], 
        [temp_range[1] - temp_range[0], time_range[1] - time_range[0]],
        left=[temp_range[0], time_range[0]], color=['orangered', 'steelblue'], 
        alpha=0.7, edgecolor='black')
ax.axvline(x=350, color='darkred', linestyle='--', linewidth=2, label='Opt T=350K')
ax.axvline(x=20, color='darkblue', linestyle='--', linewidth=2, label='Opt t=20h')
ax.set_xlabel('Value', fontweight='bold')
ax.set_title('(d) Optimization Parameter Ranges', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')

plt.suptitle('Figure 1: M-AI-Synth Dataset Overview', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'figure1_data_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Property Prediction Results
# ============================================================
with open(os.path.join(output_dir, 'property_prediction_results.json'), 'r') as f:
    pp_results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Model comparison - MAE
models = list(pp_results['test_set_results'].keys())
mae_values = [pp_results['test_set_results'][m]['MAE'] for m in models]
r2_values = [pp_results['test_set_results'][m]['R2'] for m in models]
rmse_values = [pp_results['test_set_results'][m]['RMSE'] for m in models]

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

# MAE comparison
ax = axes[0]
bars = ax.barh(range(len(models)), mae_values, color=colors, edgecolor='black')
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('MAE', fontweight='bold')
ax.set_title('(a) Mean Absolute Error', fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=8, fontweight='bold')

# R² comparison
ax = axes[1]
bars = ax.barh(range(len(models)), r2_values, color=colors, edgecolor='black')
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('R² Score', fontweight='bold')
ax.set_title('(b) R² Score', fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    x_pos = val + 0.02 if val >= 0 else val - 0.02
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=8, fontweight='bold', 
            ha='left' if val >= 0 else 'right')

# Cross-validation results
ax = axes[2]
cv_models = list(pp_results['cross_validation'].keys())
cv_mae = [pp_results['cross_validation'][m]['CV_MAE_mean'] for m in cv_models]
cv_std = [pp_results['cross_validation'][m]['CV_MAE_std'] for m in cv_models]
bars = ax.bar(range(len(cv_models)), cv_mae, yerr=cv_std, color=colors[:len(cv_models)], 
              edgecolor='black', capsize=5)
ax.set_xticks(range(len(cv_models)))
ax.set_xticklabels(cv_models, fontsize=8, rotation=30, ha='right')
ax.set_ylabel('CV MAE (mean ± std)', fontweight='bold')
ax.set_title('(c) 5-Fold Cross-Validation', fontweight='bold')

plt.suptitle('Figure 2: Property Prediction Model Comparison', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'figure2_property_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Structure Generation Results
# ============================================================
with open(os.path.join(output_dir, 'structure_generation_results.json'), 'r') as f:
    sg_results = json.load(f)

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Real data distribution
ax = fig.add_subplot(gs[0, 0])
ax.scatter(param_a, param_b, alpha=0.5, s=25, c='steelblue', edgecolors='black', linewidth=0.3, label='Real Data')
ax.set_xlabel('Parameter A', fontweight='bold')
ax.set_ylabel('Parameter B', fontweight='bold')
ax.set_title('(a) Real Structure Data', fontweight='bold')
ax.legend(fontsize=9)

# KDE generated
ax = fig.add_subplot(gs[0, 1])
kde_a = sg_results['generated_samples']['KDE']['param_a']
kde_b = sg_results['generated_samples']['KDE']['param_b']
ax.scatter(kde_a, kde_b, alpha=0.5, s=25, c='coral', edgecolors='black', linewidth=0.3, label='KDE Generated')
ax.set_xlabel('Parameter A', fontweight='bold')
ax.set_ylabel('Parameter B', fontweight='bold')
ax.set_title('(b) KDE-Generated Structures', fontweight='bold')
ax.legend(fontsize=9)

# GMM generated
ax = fig.add_subplot(gs[1, 0])
gmm_a = sg_results['generated_samples']['GMM']['param_a']
gmm_b = sg_results['generated_samples']['GMM']['param_b']
ax.scatter(gmm_a, gmm_b, alpha=0.5, s=25, c='green', edgecolors='black', linewidth=0.3, label='GMM Generated')
ax.set_xlabel('Parameter A', fontweight='bold')
ax.set_ylabel('Parameter B', fontweight='bold')
ax.set_title('(c) GMM-Generated Structures', fontweight='bold')
ax.legend(fontsize=9)

# GMM BIC/AIC selection
ax = fig.add_subplot(gs[1, 1])
n_comp = list(range(1, len(sg_results['gmm_selection']['bic_scores']) + 1))
ax.plot(n_comp, sg_results['gmm_selection']['bic_scores'], 'o-', color='steelblue', linewidth=2, label='BIC')
ax.plot(n_comp, sg_results['gmm_selection']['aic_scores'], 's-', color='coral', linewidth=2, label='AIC')
ax.axvline(x=sg_results['gmm_selection']['best_n_bic'], color='green', linestyle='--', 
           label=f'Best n={sg_results["gmm_selection"]["best_n_bic"]}')
ax.set_xlabel('Number of Components', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('(d) GMM Model Selection', fontweight='bold')
ax.legend(fontsize=9)

# Distribution comparison - marginals
ax = fig.add_subplot(gs[2, :])
x_range = np.linspace(5.0, 6.1, 100)

# Real distribution
kde_real = sns.kdeplot(param_a, ax=ax, color='steelblue', linewidth=2.5, label='Real (Param A)')
kde_real = sns.kdeplot(param_b, ax=ax, color='steelblue', linewidth=2.5, linestyle='--', label='Real (Param B)')

# KDE generated
sns.kdeplot(kde_a, ax=ax, color='coral', linewidth=2, alpha=0.7, label='KDE Gen (Param A)')
sns.kdeplot(kde_b, ax=ax, color='coral', linewidth=2, alpha=0.7, linestyle='--', label='KDE Gen (Param B)')

# GMM generated
sns.kdeplot(gmm_a, ax=ax, color='green', linewidth=2, alpha=0.7, label='GMM Gen (Param A)')
sns.kdeplot(gmm_b, ax=ax, color='green', linewidth=2, alpha=0.7, linestyle='--', label='GMM Gen (Param B)')

ax.set_xlabel('Lattice Parameter Value', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('(e) Marginal Distribution Comparison', fontweight='bold')
ax.legend(fontsize=7, ncol=3, loc='upper right')

plt.suptitle('Figure 3: Structure Generation via Generative Models', fontweight='bold', fontsize=14, y=1.01)
plt.savefig(os.path.join(img_dir, 'figure3_structure_generation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: Experimental Optimization
# ============================================================
with open(os.path.join(output_dir, 'optimization_results.json'), 'r') as f:
    opt_results = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Response surface with optimization paths
ax = axes[0, 0]
surf_T = np.array(opt_results['response_surface']['temperature'])
surf_t = np.array(opt_results['response_surface']['time'])
surf_Y = np.array(opt_results['response_surface']['yields'])
TT, tt = np.meshgrid(surf_T, surf_t)

contour = ax.contourf(TT, tt, surf_Y, levels=20, cmap='YlOrRd', alpha=0.8)
cbar = plt.colorbar(contour, ax=ax, label='Synthesis Yield')
ax.scatter([350], [20], marker='*', s=300, color='blue', edgecolors='white', 
           linewidth=2, zorder=10, label='True Optimum (350K, 20h)')

# Grid search best
gs = opt_results['grid_search']
ax.scatter([gs['best_temperature']], [gs['best_time']], marker='s', s=120, 
           color='green', edgecolors='black', zorder=10, label='Grid Search')

# Random search best
rs = opt_results['random_search']
ax.scatter([rs['best_temperature']], [rs['best_time']], marker='^', s=120, 
           color='purple', edgecolors='black', zorder=10, label='Random Search')

# BO path
bo_hist = opt_results['bayesian_optimization']['history']
bo_temps = [h['temperature'] for h in bo_hist]
bo_times = [h['time'] for h in bo_hist]
ax.plot(bo_temps, bo_times, 'o-', color='darkorange', linewidth=2, markersize=8, 
        label='Bayesian Opt.', zorder=9)
ax.scatter([bo_temps[0]], [bo_times[0]], marker='o', s=100, color='darkorange', 
           edgecolors='black', zorder=10)
ax.scatter([bo_temps[-1]], [bo_times[-1]], marker='X', s=120, color='darkred', 
           edgecolors='black', zorder=10, label='BO Final')

ax.set_xlabel('Temperature (K)', fontweight='bold')
ax.set_ylabel('Time (h)', fontweight='bold')
ax.set_title('(a) Response Surface & Optimization Paths', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')

# Convergence plot
ax = axes[0, 1]
bo_yields = [h['yield'] for h in bo_hist]
grad_hist = opt_results['gradient_ascent']['history']
grad_yields = [h['yield'] for h in grad_hist]

ax.plot(range(1, len(bo_yields)+1), bo_yields, 'o-', color='darkorange', linewidth=2, 
        markersize=8, label='Bayesian Optimization')
ax.plot(range(1, len(grad_yields)+1), grad_yields, 's-', color='teal', linewidth=2, 
        markersize=8, label='Gradient Ascent')

# Add reference lines
ax.axhline(y=gs['best_yield'], color='green', linestyle='--', linewidth=1.5, label=f'Grid Search Best = {gs["best_yield"]:.3f}')
ax.axhline(y=rs['best_yield'], color='purple', linestyle='--', linewidth=1.5, label=f'Random Search Best = {rs["best_yield"]:.3f}')

ax.set_xlabel('Iteration', fontweight='bold')
ax.set_ylabel('Yield', fontweight='bold')
ax.set_title('(b) Optimization Convergence', fontweight='bold')
ax.legend(fontsize=8)

# Parameter convergence
ax = axes[1, 0]
ax.plot(range(1, len(bo_yields)+1), bo_temps, 'o-', color='darkorange', linewidth=2, 
        markersize=8, label='BO Temperature')
ax.plot(range(1, len(bo_yields)+1), bo_times, 's-', color='steelblue', linewidth=2, 
        markersize=8, label='BO Time')
ax.axhline(y=350, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=20, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Iteration', fontweight='bold')
ax.set_ylabel('Parameter Value', fontweight='bold')
ax.set_title('(c) Parameter Convergence (Bayesian Opt.)', fontweight='bold')
ax.legend(fontsize=8)

# Final comparison
ax = axes[1, 1]
methods = ['Grid Search', 'Random Search', 'Bayesian Opt.', 'Gradient Ascent']
temp_errors = [opt_results['grid_search']['temp_error'],
               opt_results['random_search']['temp_error'],
               opt_results['bayesian_optimization']['temp_error'],
               opt_results['gradient_ascent']['temp_error']]
time_errors = [opt_results['grid_search']['time_error'],
               opt_results['random_search']['time_error'],
               opt_results['bayesian_optimization']['time_error'],
               opt_results['gradient_ascent']['time_error']]
yields = [opt_results['grid_search']['best_yield'],
          opt_results['random_search']['best_yield'],
          opt_results['bayesian_optimization']['best_yield'],
          opt_results['gradient_ascent']['best_yield']]

x_pos = np.arange(len(methods))
width = 0.25
bars1 = ax.bar(x_pos - width, temp_errors, width, color='orangered', edgecolor='black', label='Temperature Error (K)')
bars2 = ax.bar(x_pos, time_errors, width, color='steelblue', edgecolor='black', label='Time Error (h)')
bars3 = ax.bar(x_pos + width, yields, width, color='green', edgecolor='black', label='Best Yield')

for bar, val in zip(bars3, yields):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
            ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9, rotation=15)
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('(d) Optimization Method Comparison', fontweight='bold')
ax.legend(fontsize=8)

plt.suptitle('Figure 4: Autonomous Synthesis Parameter Optimization', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'figure4_optimization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Multimodal Integration Framework (Schematic)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Create a flow diagram
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Input box
inputs = ['Atomic Structures', 'Chemical Compositions', 'Crystal Graphs', 
          'Microscopy Images', 'Spectral Data (XRD/FTIR)', 'Scientific Literature',
          'Property Databases', 'Synthesis Parameters']
input_box = FancyBboxPatch((0.3, 0.5), 3.5, 7, boxstyle="round,pad=0.1", 
                           facecolor='lightblue', edgecolor='navy', linewidth=2, alpha=0.7)
ax.add_patch(input_box)
ax.text(2.05, 7.3, 'MULTIMODAL INPUTS', ha='center', fontweight='bold', fontsize=12, color='navy')
for i, inp in enumerate(inputs):
    ax.text(2.05, 6.8 - i*0.85, f'• {inp}', ha='center', fontsize=9)

# AI/ML box
ml_box = FancyBboxPatch((5, 2.5), 4, 3.5, boxstyle="round,pad=0.1", 
                        facecolor='lightyellow', edgecolor='darkorange', linewidth=2, alpha=0.7)
ax.add_patch(ml_box)
ax.text(7, 5.7, 'AI/ML MODELS', ha='center', fontweight='bold', fontsize=12, color='darkorange')
ml_methods = ['CGCNN (Crystal Graphs)', 'Random Forest / GBM', 
              'Physics-Informed NNs', 'GMM / KDE (Generative)',
              'Bayesian Optimization']
for i, m in enumerate(ml_methods):
    ax.text(7, 5.2 - i*0.6, f'▸ {m}', ha='center', fontsize=9)

# Output box
outputs = ['Predicted Properties\n(Mechanical, Electronic,\nCatalytic)', 
           'Novel Structures\n& Microstructures',
           'Optimized Synthesis\nParameters',
           'Classification &\nSegmentation Results']
output_box = FancyBboxPatch((10.5, 0.5), 3.2, 7, boxstyle="round,pad=0.1", 
                            facecolor='lightgreen', edgecolor='darkgreen', linewidth=2, alpha=0.7)
ax.add_patch(output_box)
ax.text(12.1, 7.3, 'OUTPUTS', ha='center', fontweight='bold', fontsize=12, color='darkgreen')
for i, out in enumerate(outputs):
    ax.text(12.1, 6.5 - i*1.6, out, ha='center', fontsize=9)

# Arrows
arrow1 = FancyArrowPatch((3.9, 5.5), (4.9, 5.2), arrowstyle='->', lw=3, color='gray')
ax.add_patch(arrow1)
arrow1b = FancyArrowPatch((3.9, 3.5), (4.9, 3.5), arrowstyle='->', lw=3, color='gray')
ax.add_patch(arrow1b)

arrow2 = FancyArrowPatch((9.1, 5.2), (10.4, 5.8), arrowstyle='->', lw=3, color='gray')
ax.add_patch(arrow2)
arrow2b = FancyArrowPatch((9.1, 3.5), (10.4, 3.5), arrowstyle='->', lw=3, color='gray')
ax.add_patch(arrow2b)

# Feedback loop
ax.annotate('', xy=(2.0, 0.8), xytext=(12.0, 0.8),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2, connectionstyle='arc3,rad=-0.5'))
ax.text(7, 0.2, 'Data-Driven Inverse Design & Feedback', ha='center', fontsize=10, 
        fontstyle='italic', color='purple')

plt.suptitle('Figure 5: Multimodal AI-Driven Materials Discovery Framework', 
             fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'figure5_framework.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print("\nAll figures generated successfully!")
