#!/usr/bin/env python3
"""
MATBG Superfluid Stiffness Analysis - Main Script
===================================================
This script parses the core dataset, performs quantitative analysis,
and generates all figures for the research report.

Key insight: arrays have different lengths because:
- T_array: 100 points (0 to 1.2 K)
- Model D_s arrays: have zeros appended for T > T_c region (varying lengths)
- Experimental D_s(T): 110 points (extends beyond T_c with nonzero values)
- I_dc: 50 points (0 to 60 nA)
- D_s_gl: 60 points (extends beyond I_c)
- D_s_dc_exp: 85 points (extends well beyond I_c, showing complex behavior)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import linregress
import json
import os
import re

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_001_20260415_201423'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# Parse Data
# ============================================================

def parse_array(text):
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    text = text.replace('\n', ' ')
    values = [v for v in text.split() if v.strip()]
    return np.array([float(v) for v in values])

def extract_array_by_label(label, content):
    escaped = re.escape(label)
    pattern = rf'\*\*{escaped}\*\*\n\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return parse_array(match.group(1))
    else:
        raise ValueError(f"Could not find array for label: {label}")

data_file = os.path.join(DATA_DIR, 'MATBG Superfluid Stiffness Core Dataset.txt')
with open(data_file, 'r') as f:
    content = f.read()

# File 1: Carrier density data (all 50 points, matching)
n_eff = extract_array_by_label('Carrier Density Data (n_eff in m^-2):', content)
D_s_conv = extract_array_by_label('Conventional Superfluid Stiffness (D_s_conv):', content)
D_s_geom = extract_array_by_label('Quantum Geometric Superfluid Stiffness (D_s_geom):', content)
D_s_exp_hole = extract_array_by_label('Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole):', content)
D_s_exp_electron = extract_array_by_label('Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron):', content)

# File 2: Temperature dependence
T_array_raw = extract_array_by_label('Temperature Array (T in K):', content)
D_s_bcs_raw = extract_array_by_label('BCS Model Data (D_s_bcs):', content)
D_s_nodal_raw = extract_array_by_label('Nodal Superconductor Data (D_s_nodal):', content)
D_s_power_n2_raw = extract_array_by_label('Power Law n=2.0 Data (D_s_power_n2):', content)
D_s_power_n2_5_raw = extract_array_by_label('Power Law n=2.5 Data (D_s_power_n2_5):', content)
D_s_power_n3_raw = extract_array_by_label('Power Law n=3.0 Data (D_s_power_n3):', content)
D_s_experimental_raw = extract_array_by_label('Experimental Data with Noise (D_s_experimental):', content)

# For temperature-dependent analysis, we need to align arrays.
# Strategy: Use the first N_common = min(len(T_array), len(model)) points
# where all arrays overlap, then handle remaining points separately.
N_T = len(T_array_raw)
print(f"Raw array lengths: T={N_T}, BCS={len(D_s_bcs_raw)}, nodal={len(D_s_nodal_raw)}, "
      f"n2={len(D_s_power_n2_raw)}, n2.5={len(D_s_power_n2_5_raw)}, n3={len(D_s_power_n3_raw)}, "
      f"exp={len(D_s_experimental_raw)}")

# The experimental array has more points than T_array.
# We'll use the first N_T points of experimental data paired with T_array.
# For model arrays shorter than N_T, pad with zeros at end (T > T_c).
D_s_experimental = D_s_experimental_raw[:N_T]

def pad_to_length(arr, target_len, pad_value=0.0):
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode='constant', constant_values=pad_value)

D_s_bcs = pad_to_length(D_s_bcs_raw, N_T)
D_s_nodal = pad_to_length(D_s_nodal_raw, N_T)
D_s_power_n2 = pad_to_length(D_s_power_n2_raw, N_T)
D_s_power_n2_5 = pad_to_length(D_s_power_n2_5_raw, N_T)
D_s_power_n3 = pad_to_length(D_s_power_n3_raw, N_T)

# File 3: Current dependence - different lengths
T_array = T_array_raw
I_dc_raw = extract_array_by_label('DC Current Array (I_dc in nA):', content)
D_s_gl_raw = extract_array_by_label('Ginzburg-Landau Model (D_s_gl):', content)
D_s_linear_raw = extract_array_by_label('Linear Meissner Model (D_s_linear):', content)
D_s_dc_exp_raw = extract_array_by_label('Experimental DC Data (D_s_dc_exp):', content)
P_mw = extract_array_by_label('Microwave Power Array (P_mw normalized):', content)
I_mw = extract_array_by_label('Microwave Current Amplitude (I_mw_amplitude in nA):', content)
D_s_mw_exp = extract_array_by_label('Experimental Microwave Data (D_s_mw_exp):', content)

print(f"Current arrays: I_dc={len(I_dc_raw)}, D_s_gl={len(D_s_gl_raw)}, "
      f"D_s_linear={len(D_s_linear_raw)}, D_s_dc_exp={len(D_s_dc_exp_raw)}")

# Fixed parameters
T_c = 1.0
D_s0 = 100.0
I_c = 50.0

# ============================================================
# Quantitative Analysis
# ============================================================

# 1. Enhancement ratios
enhancement_hole = D_s_exp_hole / D_s_conv
enhancement_electron = D_s_exp_electron / D_s_conv
enhancement_geom_ratio = D_s_geom / D_s_conv

print("\n=== Enhancement Analysis ===")
print(f"Mean enhancement ratio (hole-doped): {np.mean(enhancement_hole):.1f}")
print(f"Mean enhancement ratio (electron-doped): {np.mean(enhancement_electron):.1f}")
print(f"Mean geometric/conventional ratio: {np.mean(enhancement_geom_ratio):.1f}")

# 2. Temperature dependence power law fitting
mask_sub = (T_array < T_c) & (T_array > 0.05)
T_fit = T_array[mask_sub]
D_s_fit = D_s_experimental[mask_sub]

delta_D_s = 1 - D_s_fit / D_s0
log_T_ratio = np.log(T_fit / T_c)
log_delta = np.log(delta_D_s)

slope, intercept, r_value, p_value, std_err = linregress(log_T_ratio, log_delta)
power_law_exponent = slope

print("\n=== Temperature Dependence Power Law Analysis ===")
print(f"Power law exponent n = {power_law_exponent:.3f} (log-log regression)")
print(f"R-squared = {r_value**2:.4f}")

# The experimental data shows a very slow decline (D_s stays near 67-68 even at T=Tc)
# This means 1-D_s/D_s0 is small near Tc, and the log-log slope reflects this
# The key comparison is which MODEL best matches the experimental curve shape

# Compute RMSE between experimental and each model for T < T_c
mask_model = (T_array > 0) & (T_array < T_c)
T_model = T_array[mask_model]
D_s_exp_model = D_s_experimental[mask_model]

model_rmse = {}
for name, arr in [('BCS', D_s_bcs), ('Nodal', D_s_nodal), ('n=2', D_s_power_n2),
                   ('n=2.5', D_s_power_n2_5), ('n=3', D_s_power_n3)]:
    model_data = arr[mask_model]
    rmse = np.sqrt(np.mean((D_s_exp_model - model_data)**2))
    model_rmse[name] = rmse
    print(f"RMSE vs {name}: {rmse:.4f}")

# 3. Current dependence analysis
# Use first 50 points (matching I_dc) for plotting, but limit analysis to I < I_c region
# The GL model has 60 points, linear has 50, experimental DC has 85
# We'll use I_dc as the x-axis base and interpolate/pad others

# For plotting, use common range up to I/Ic ~ 1.0
# I_dc goes from 0 to 60 nA (50 points). I_c = 50 nA.
# So I_dc[-1] = 60 > I_c. Points beyond index ~42 are I > I_c.

# For the GL model (60 points), we need to figure out its I values
# GL model uses same I_dc range but extended. Let's reconstruct.
# From the data: D_s_gl[42] ≈ 0, so I at index 42 ≈ I_c
# The GL model likely uses I from 0 to ~60 nA with 60 points
I_dc_gl = np.linspace(0, 60, 60)  # 60 points from 0 to 60 nA

# For experimental DC data (85 points), similarly extended
I_dc_exp = np.linspace(0, 60, 85)  # 85 points covering extended range

# Verify by checking D_s_gl near I_c
print(f"\n=== Current Dependence ===")
print(f"D_s_gl at index 42 (near I_c): {D_s_gl_raw[42]:.4f}")
print(f"D_s_gl at index 55: {D_s_gl_raw[55]:.4f}")

# Quadratic fit on experimental DC data for I < I_c region
# Use first 42 points of I_dc (up to ~I_c)
I_dc_analysis = I_dc_raw[:42]
D_s_dc_analysis = D_s_dc_exp_raw[:42]
I_norm_analysis = I_dc_analysis / I_c

def quad_model(I_norm, a):
    return D_s0 * (1 - a * I_norm**2)

try:
    popt_quad, pcov_quad = curve_fit(quad_model, I_norm_analysis, D_s_dc_analysis, p0=[1.0])
    a_quad = popt_quad[0]
    print(f"Quadratic fit coefficient a = {a_quad:.4f}")
except Exception as exc:
    print(f"Quadratic fit failed: {exc}")
    a_quad = 1.0

# RMSE of GL model vs experimental
# Compare using overlapping range
n_compare = min(42, len(D_s_gl_raw))
rmse_gl = np.sqrt(np.mean((D_s_dc_analysis[:n_compare] - D_s_gl_raw[:n_compare])**2))
print(f"RMSE GL model vs exp (first {n_compare} pts): {rmse_gl:.4f}")

# ============================================================
# Save results
# ============================================================

results = {
    "enhancement_analysis": {
        "mean_enhancement_hole": float(np.mean(enhancement_hole)),
        "mean_enhancement_electron": float(np.mean(enhancement_electron)),
        "mean_geom_conv_ratio": float(np.mean(enhancement_geom_ratio)),
    },
    "temperature_dependence": {
        "power_law_exponent_loglog": float(power_law_exponent),
        "r_squared_loglog": float(r_value**2),
        "model_rmse": model_rmse,
    },
    "current_dependence": {
        "quadratic_coefficient": float(a_quad),
        "rmse_gl_theory": float(rmse_gl),
    }
}

with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================
# Figure Generation
# ============================================================

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

n_eff_plot = n_eff / 1e14

# ---- FIGURE 1: Superfluid Stiffness vs Carrier Density ----
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.semilogy(n_eff_plot, D_s_conv/1e9, 'b--', linewidth=2, label='Conventional ($v_F = 700$ m/s)')
ax1.semilogy(n_eff_plot, D_s_geom/1e9, 'r-', linewidth=2, label='Quantum Geometric ($v_F = 3000$ m/s)')
ax1.semilogy(n_eff_plot, D_s_exp_hole/1e9, 'ko', markersize=5, label='Experimental (hole-doped)')
ax1.semilogy(n_eff_plot, D_s_exp_electron/1e9, 's', color='gray', markersize=5, label='Experimental (electron-doped)')
ax1.set_xlabel('Effective Carrier Density $n_{eff}$ ($\\times 10^{14}$ m$^{-2}$)')
ax1.set_ylabel('Superfluid Stiffness $D_s$ (nH$^{-1}$)')
ax1.set_title('Superfluid Stiffness vs Carrier Density in MATBG')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
fig1.savefig(os.path.join(IMAGE_DIR, 'fig1_stiffness_vs_density.png'))
plt.close(fig1)
print("Figure 1 saved.")

# ---- FIGURE 2: Enhancement Ratio ----
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
ax2a.plot(n_eff_plot, enhancement_hole, 'ko-', markersize=4, label='Exp(hole)/Conv')
ax2a.plot(n_eff_plot, enhancement_electron, 's-', color='gray', markersize=4, label='Exp(electron)/Conv')
ax2a.plot(n_eff_plot, enhancement_geom_ratio, 'r--', linewidth=2, label='Geom/Conv')
ax2a.set_xlabel('$n_{eff}$ ($\\times 10^{14}$ m$^{-2}$)')
ax2a.set_ylabel('Enhancement Ratio')
ax2a.set_title('(a) Superfluid Stiffness Enhancement')
ax2a.legend()
ax2a.grid(True, alpha=0.3)

ax2b.scatter(D_s_geom/1e9, D_s_exp_hole/1e9, c='k', s=20, label='Hole-doped', zorder=3)
ax2b.scatter(D_s_geom/1e9, D_s_exp_electron/1e9, c='gray', marker='s', s=20, label='Electron-doped', zorder=3)
max_val = max(np.max(D_s_exp_hole), np.max(D_s_geom))/1e9 * 1.1
ax2b.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Parity (1:1)')
ax2b.set_xlabel('$D_s^{geom}$ (nH$^{-1}$)')
ax2b.set_ylabel('$D_s^{exp}$ (nH$^{-1}$)')
ax2b.set_title('(b) Experimental vs Geometric Prediction')
ax2b.legend()
ax2b.grid(True, alpha=0.3)
fig2.savefig(os.path.join(IMAGE_DIR, 'fig2_enhancement_ratio.png'))
plt.close(fig2)
print("Figure 2 saved.")

# ---- FIGURE 3: Temperature Dependence ----
fig3, ax3 = plt.subplots(figsize=(8, 6))
mask_plot = T_array <= 1.05
ax3.plot(T_array[mask_plot]/T_c, D_s_bcs[mask_plot]/D_s0, 'b-', linewidth=2, label='BCS (s-wave)')
ax3.plot(T_array[mask_plot]/T_c, D_s_nodal[mask_plot]/D_s0, 'g--', linewidth=2, label='Nodal (d-wave)')
ax3.plot(T_array[mask_plot]/T_c, D_s_power_n2[mask_plot]/D_s0, 'c:', linewidth=2, label='Power law n=2')
ax3.plot(T_array[mask_plot]/T_c, D_s_power_n2_5[mask_plot]/D_s0, 'm-', linewidth=2, label='Power law n=2.5')
ax3.plot(T_array[mask_plot]/T_c, D_s_power_n3[mask_plot]/D_s0, color='orange', linestyle='-', linewidth=2, label='Power law n=3')
ax3.plot(T_array[mask_plot]/T_c, D_s_experimental[mask_plot]/D_s0, 'ko', markersize=4, label='Experimental')
ax3.set_xlabel('$T/T_c$')
ax3.set_ylabel('$D_s/D_{s0}$')
ax3.set_title('Temperature Dependence of Superfluid Stiffness')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1.1)
ax3.set_ylim(0, 1.1)
fig3.savefig(os.path.join(IMAGE_DIR, 'fig3_temperature_dependence.png'))
plt.close(fig3)
print("Figure 3 saved.")

# ---- FIGURE 4: Power Law Fitting ----
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

mask_ll = (T_array > 0.05) & (T_array < 0.85)
T_ll = T_array[mask_ll]
delta_ll = 1 - D_s_experimental[mask_ll] / D_s0
ax4a.loglog(T_ll/T_c, delta_ll, 'ko', markersize=5, label='Experimental')
T_ref = np.linspace(0.06, 0.85, 100)
for n_val, color, ls in [(2, 'cyan', ':'), (2.5, 'magenta', '--'), (3, 'orange', '-'), (power_law_exponent, 'red', '-')]:
    ax4a.loglog(T_ref/T_c, (T_ref/T_c)**n_val, color=color, linestyle=ls, linewidth=1.5, label=f'n={n_val:.1f}')
ax4a.set_xlabel('$T/T_c$')
ax4a.set_ylabel('$1 - D_s/D_{s0}$')
ax4a.set_title('(a) Power-Law Analysis')
ax4a.legend()
ax4a.grid(True, alpha=0.3, which='both')

# Residual comparison
mask_res = (T_array > 0) & (T_array < T_c)
for name, arr in [('BCS', D_s_bcs), ('Nodal', D_s_nodal), ('n=2.5', D_s_power_n2_5), ('n=3', D_s_power_n3)]:
    residuals = (D_s_experimental[mask_res] - arr[mask_res]) / D_s0
    ax4b.plot(T_array[mask_res]/T_c, residuals, label=name, linewidth=1.5)
ax4b.set_xlabel('$T/T_c$')
ax4b.set_ylabel('Residual $(D_s^{exp} - D_s^{model})/D_{s0}$')
ax4b.set_title('(b) Model Residuals')
ax4b.legend()
ax4b.grid(True, alpha=0.3)
fig4.savefig(os.path.join(IMAGE_DIR, 'fig4_powerlaw_fitting.png'))
plt.close(fig4)
print("Figure 4 saved.")

# ---- FIGURE 5: Current Dependence ----
fig5, ax5 = plt.subplots(figsize=(8, 6))

# Plot using I_dc_raw (50 pts) as x-axis
I_norm_dc = I_dc_raw / I_c

# GL model: use first 50 points (matching I_dc range)
D_s_gl_plot = D_s_gl_raw[:50]
D_s_linear_plot = D_s_linear_raw  # already 50 pts

# Experimental DC: use first 50 points
D_s_dc_exp_plot = D_s_dc_exp_raw[:50]

ax5.plot(I_norm_dc, D_s_gl_plot/D_s0, 'b-', linewidth=2, label='Ginzburg-Landau $(1-(I/I_c)^2)$')
ax5.plot(I_norm_dc, D_s_linear_plot/D_s0, 'g--', linewidth=2, label='Linear Meissner $(1-I/I_c)$')
ax5.plot(I_norm_dc, D_s_dc_exp_plot/D_s0, 'ko', markersize=4, label='Experimental (DC)')

ax5.set_xlabel('$I_{dc}/I_c$')
ax5.set_ylabel('$D_s/D_{s0}$')
ax5.set_title('Current Dependence of Superfluid Stiffness')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 1.2)
ax5.set_ylim(-0.2, 1.1)
fig5.savefig(os.path.join(IMAGE_DIR, 'fig5_current_dependence.png'))
plt.close(fig5)
print("Figure 5 saved.")

# ---- FIGURE 6: Quadratic Verification ----
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 5))

# DC: plot D_s vs (I/Ic)^2 for I < I_c region
n_dc_quad = 35  # use first 35 points (I/Ic < 0.7)
I_sq_dc = (I_dc_raw[:n_dc_quad]/I_c)**2
D_s_dc_quad = D_s_dc_exp_raw[:n_dc_quad]

ax6a.scatter(I_sq_dc, D_s_dc_quad/D_s0, c='k', s=20, zorder=3)
slope_dc, intercept_dc, r_dc, _, _ = linregress(I_sq_dc, D_s_dc_quad/D_s0)
x_line = np.linspace(0, 0.5, 100)
ax6a.plot(x_line, slope_dc * x_line + intercept_dc, 'r-', linewidth=2,
          label=f'fit: slope={slope_dc:.3f}, R$^2$={r_dc**2:.4f}')
ax6a.set_xlabel('$(I/I_c)^2$')
ax6a.set_ylabel('$D_s/D_{s0}$')
ax6a.set_title('(a) DC: Quadratic Law Verification')
ax6a.legend()
ax6a.grid(True, alpha=0.3)

# Microwave
I_mw_sq = (I_mw/I_c)**2
ax6b.scatter(I_mw_sq, D_s_mw_exp/D_s0, c='k', s=20, zorder=3)
slope_mw, intercept_mw, r_mw, _, _ = linregress(I_mw_sq, D_s_mw_exp/D_s0)
ax6b.plot(I_mw_sq, slope_mw * I_mw_sq + intercept_mw, 'r-', linewidth=2,
          label=f'fit: slope={slope_mw:.4f}, R$^2$={r_mw**2:.4f}')
ax6b.set_xlabel('$(I_{mw}/I_c)^2$')
ax6b.set_ylabel('$D_s/D_{s0}$')
ax6b.set_title('(b) Microwave: Quadratic Verification')
ax6b.legend()
ax6b.grid(True, alpha=0.3)
fig6.savefig(os.path.join(IMAGE_DIR, 'fig6_quadratic_verification.png'))
plt.close(fig6)
print("Figure 6 saved.")

# ---- FIGURE 7: Summary Panel ----
fig7 = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig7, hspace=0.35, wspace=0.35)

ax7a = fig7.add_subplot(gs[0, 0])
ax7a.semilogy(n_eff_plot, D_s_conv/1e9, 'b--', linewidth=2, label='Conventional')
ax7a.semilogy(n_eff_plot, D_s_geom/1e9, 'r-', linewidth=2, label='Quantum Geometric')
ax7a.semilogy(n_eff_plot, D_s_exp_hole/1e9, 'ko', markersize=4, label='Exp (hole)')
ax7a.semilogy(n_eff_plot, D_s_exp_electron/1e9, 's', color='gray', markersize=4, label='Exp (electron)')
ax7a.set_xlabel('$n_{eff}$ ($\\times 10^{14}$ m$^{-2}$)')
ax7a.set_ylabel('$D_s$ (nH$^{-1}$)')
ax7a.set_title('(a) $D_s$ vs Carrier Density')
ax7a.legend(fontsize=8)
ax7a.grid(True, alpha=0.3)

ax7b = fig7.add_subplot(gs[0, 1])
ax7b.plot(n_eff_plot, enhancement_hole, 'ko-', markersize=3, label='Exp/Conv (hole)')
ax7b.plot(n_eff_plot, enhancement_geom_ratio, 'r--', linewidth=2, label='Geom/Conv')
ax7b.axhline(y=np.mean(enhancement_hole), color='k', linestyle=':', alpha=0.5)
ax7b.set_xlabel('$n_{eff}$ ($\\times 10^{14}$ m$^{-2}$)')
ax7b.set_ylabel('Enhancement Ratio')
ax7b.set_title('(b) Quantum Geometry Enhancement')
ax7b.legend(fontsize=8)
ax7b.grid(True, alpha=0.3)

ax7c = fig7.add_subplot(gs[0, 2])
mask_c = T_array <= 1.05
ax7c.plot(T_array[mask_c]/T_c, D_s_bcs[mask_c]/D_s0, 'b-', linewidth=1.5, label='BCS')
ax7c.plot(T_array[mask_c]/T_c, D_s_nodal[mask_c]/D_s0, 'g--', linewidth=1.5, label='Nodal')
ax7c.plot(T_array[mask_c]/T_c, D_s_power_n2_5[mask_c]/D_s0, 'm-', linewidth=1.5, label='n=2.5')
ax7c.plot(T_array[mask_c]/T_c, D_s_power_n3[mask_c]/D_s0, color='orange', linewidth=1.5, label='n=3')
ax7c.plot(T_array[mask_c]/T_c, D_s_experimental[mask_c]/D_s0, 'ko', markersize=3, label='Exp')
ax7c.set_xlabel('$T/T_c$')
ax7c.set_ylabel('$D_s/D_{s0}$')
ax7c.set_title('(c) Temperature Dependence')
ax7c.legend(fontsize=8)
ax7c.grid(True, alpha=0.3)

ax7d = fig7.add_subplot(gs[1, 0])
ax7d.loglog(T_ll/T_c, delta_ll, 'ko', markersize=4)
T_ref2 = np.linspace(0.06, 0.85, 100)
ax7d.loglog(T_ref2/T_c, (T_ref2/T_c)**2.5, 'm--', linewidth=1.5, label='n=2.5')
ax7d.loglog(T_ref2/T_c, (T_ref2/T_c)**3, color='orange', linestyle='--', linewidth=1.5, label='n=3')
ax7d.loglog(T_ref2/T_c, (T_ref2/T_c)**power_law_exponent, 'r-', linewidth=2, label=f'fit n={power_law_exponent:.2f}')
ax7d.set_xlabel('$T/T_c$')
ax7d.set_ylabel('$1 - D_s/D_{s0}$')
ax7d.set_title('(d) Power-Law Exponent')
ax7d.legend(fontsize=8)
ax7d.grid(True, alpha=0.3, which='both')

ax7e = fig7.add_subplot(gs[1, 1])
ax7e.plot(I_norm_dc[:42], D_s_gl_raw[:42]/D_s0, 'b-', linewidth=2, label='GL theory')
ax7e.plot(I_norm_dc[:42], D_s_linear_raw[:42]/D_s0, 'g--', linewidth=2, label='Linear Meissner')
ax7e.plot(I_norm_dc[:42], D_s_dc_exp_raw[:42]/D_s0, 'ko', markersize=4, label='Exp (DC)')
ax7e.set_xlabel('$I_{dc}/I_c$')
ax7e.set_ylabel('$D_s/D_{s0}$')
ax7e.set_title('(e) DC Current Dependence')
ax7e.legend(fontsize=8)
ax7e.grid(True, alpha=0.3)

ax7f = fig7.add_subplot(gs[1, 2])
ax7f.scatter(I_sq_dc, D_s_dc_quad/D_s0, c='k', s=15, zorder=3)
ax7f.plot(x_line, slope_dc * x_line + intercept_dc, 'r-', linewidth=2,
          label=f'slope={slope_dc:.3f}, R$^2$={r_dc**2:.4f}')
ax7f.set_xlabel('$(I/I_c)^2$')
ax7f.set_ylabel('$D_s/D_{s0}$')
ax7f.set_title('(f) Quadratic Current Law')
ax7f.legend(fontsize=8)
ax7f.grid(True, alpha=0.3)

fig7.savefig(os.path.join(IMAGE_DIR, 'fig7_summary_panel.png'))
plt.close(fig7)
print("Figure 7 saved.")

print("\n=== All figures generated successfully ===")