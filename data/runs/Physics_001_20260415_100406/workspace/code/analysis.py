#!/usr/bin/env python3
"""
Superfluid Stiffness Analysis of Magic-Angle Twisted Bilayer Graphene
Parses the core dataset and generates all analysis figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os

# Ensure output directories exist
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# Parse the dataset
# ============================================================

def parse_array_from_text(text, marker):
    """Extract a numpy array from the text file between markers."""
    lines = text.split('\n')
    in_array = False
    array_lines = []
    for line in lines:
        if marker in line:
            in_array = True
            continue
        if in_array:
            if line.strip().startswith('['):
                # Collect lines until we find the closing bracket
                array_lines.append(line.strip().lstrip('['))
                if ']' in line:
                    array_lines[-1] = array_lines[-1].rstrip(']').strip()
                    break
            elif array_lines:
                cleaned = line.strip().rstrip(']').strip()
                array_lines.append(cleaned)
                if ']' in line:
                    break
    # Join all number tokens and parse as numpy array
    all_tokens = []
    for aline in array_lines:
        all_tokens.extend(aline.split())
    return np.array([float(x) for x in all_tokens])

# Read the dataset
with open('data/MATBG Superfluid Stiffness Core Dataset.txt', 'r') as f:
    data_text = f.read()

# --- File 1: Carrier Density Dependence ---
n_eff = parse_array_from_text(data_text, 'Carrier Density Data')
D_s_conv = parse_array_from_text(data_text, 'Conventional Superfluid Stiffness (D_s_conv)')
D_s_geom = parse_array_from_text(data_text, 'Quantum Geometric Superfluid Stiffness (D_s_geom)')
D_s_exp_hole = parse_array_from_text(data_text, 'Experimental Superfluid Stiffness Hole-doped')
D_s_exp_electron = parse_array_from_text(data_text, 'Experimental Superfluid Stiffness Electron-doped')

# --- File 2: Temperature Dependence ---
T_arr = parse_array_from_text(data_text, 'Temperature Array (T in K)')
D_s_bcs = parse_array_from_text(data_text, 'BCS Model Data (D_s_bcs)')
D_s_nodal = parse_array_from_text(data_text, 'Nodal Superconductor Data (D_s_nodal)')
D_s_power_n2 = parse_array_from_text(data_text, 'Power Law n=2.0 Data')
D_s_power_n2_5 = parse_array_from_text(data_text, 'Power Law n=2.5 Data')
D_s_power_n3 = parse_array_from_text(data_text, 'Power Law n=3.0 Data')
D_s_experimental = parse_array_from_text(data_text, 'Experimental Data with Noise')

# --- File 3: Current Dependence ---
I_dc = parse_array_from_text(data_text, 'DC Current Array (I_dc in nA)')
D_s_gl = parse_array_from_text(data_text, 'Ginzburg-Landau Model (D_s_gl)')
D_s_linear = parse_array_from_text(data_text, 'Linear Meissner Model (D_s_linear)')
D_s_dc_exp = parse_array_from_text(data_text, 'Experimental DC Data (D_s_dc_exp)')
P_mw = parse_array_from_text(data_text, 'Microwave Power Array')
I_mw = parse_array_from_text(data_text, 'Microwave Current Amplitude')
D_s_mw_exp = parse_array_from_text(data_text, 'Experimental Microwave Data')

# Fixed parameters
e_charge = 1.602e-19
hbar = 1.054e-34
v_f_conv = 700.0
v_f_geom = 3000.0
T_c = 1.0
D_s0 = 100.0
I_c = 50.0
n_values = [2.0, 2.5, 3.0]

print(f"Carrier density points: {len(n_eff)}")
print(f"Temperature points: {len(T_arr)}")
print(f"DC current points: {len(I_dc)}")
print(f"Microwave power points: {len(P_mw)}")

# ============================================================
# Figure 1: Carrier Density Dependence
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Theory comparison
ax = axes[0]
ax.plot(n_eff/1e15, D_s_conv/1e9, 'b-', lw=2, label='Conventional (Fermi liquid)')
ax.plot(n_eff/1e15, D_s_geom/1e9, 'r-', lw=2, label='Quantum geometric')
ax.set_xlabel(r'Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)', fontsize=12)
ax.set_ylabel(r'Superfluid stiffness $D_s$ (10$^9$ H$^{-1}$)', fontsize=12)
ax.set_title('(a) Theoretical predictions', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel (b): Experimental data
ax = axes[1]
ax.plot(n_eff/1e15, D_s_exp_hole/1e9, 'ro-', ms=3, lw=1.5, label='Hole-doped (exp.)')
ax.plot(n_eff/1e15, D_s_exp_electron/1e9, 'bs-', ms=3, lw=1.5, label='Electron-doped (exp.)')
ax.plot(n_eff/1e15, D_s_conv/1e9, 'k--', lw=1.5, label='Conventional (theory)')
ax.plot(n_eff/1e15, D_s_geom/1e9, 'g--', lw=1.5, label='Quantum geometric (theory)')
ax.set_xlabel(r'Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)', fontsize=12)
ax.set_ylabel(r'Superfluid stiffness $D_s$ (10$^9$ H$^{-1}$)', fontsize=12)
ax.set_title('(b) Experimental vs theoretical', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_carrier_density.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Temperature Dependence - Model Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pad shorter arrays to match T_arr length (100 points)
def pad_to_length(arr, target_len, pad_value=0.0):
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.full(target_len - len(arr), pad_value)])

D_s_bcs_padded = pad_to_length(D_s_bcs, len(T_arr))
D_s_nodal_padded = pad_to_length(D_s_nodal, len(T_arr))
D_s_power_n2_padded = pad_to_length(D_s_power_n2, len(T_arr))
D_s_power_n2_5_padded = pad_to_length(D_s_power_n2_5, len(T_arr))
D_s_power_n3_padded = pad_to_length(D_s_power_n3, len(T_arr))
D_s_exp_padded = pad_to_length(D_s_experimental, len(T_arr))

# Panel (a): All models
ax = axes[0]
ax.plot(T_arr/T_c, D_s_bcs_padded/D_s0, 'b-', lw=2, label='BCS (s-wave)')
ax.plot(T_arr/T_c, D_s_nodal_padded/D_s0, 'r-', lw=2, label='Nodal (d-wave)')
ax.plot(T_arr/T_c, D_s_power_n2_padded/D_s0, 'g--', lw=1.5, label=r'Power law $n=2.0$')
ax.plot(T_arr/T_c, D_s_power_n2_5_padded/D_s0, 'm--', lw=1.5, label=r'Power law $n=2.5$')
ax.plot(T_arr/T_c, D_s_power_n3_padded/D_s0, 'c--', lw=1.5, label=r'Power law $n=3.0$')
ax.plot(T_arr/T_c, D_s_exp_padded/D_s0, 'ko', ms=2, alpha=0.5, label='Experimental (sim.)')
ax.set_xlabel(r'Temperature $T/T_c$', fontsize=12)
ax.set_ylabel(r'$D_s(T)/D_s(0)$', fontsize=12)
ax.set_title('(a) Temperature dependence: model comparison', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel (b): Log-log to identify power law
ax = axes[1]
# Only use data below Tc for power-law analysis
mask = T_arr < T_c
mask &= D_s_exp_padded > 0
T_below = T_arr[mask]
D_exp_below = D_s_exp_padded[mask]
D_bcs_below = D_s_bcs_padded[mask]

# Plot 1 - D_s(0) - D_s(T) vs T on log-log
delta_D_exp = D_s0 - D_exp_below
delta_D_bcs = D_s0 - D_bcs_below
# Avoid log(0)
valid = delta_D_exp > 0
ax.loglog(T_below[valid]/T_c, delta_D_exp[valid]/D_s0, 'ko', ms=3, label='Experimental')
ax.loglog(T_below[valid]/T_c, delta_D_bcs[valid]/D_s0, 'b-', lw=2, label='BCS reference')

# Power law references
for n_val, color in zip([2.0, 2.5, 3.0], ['g', 'm', 'c']):
    D_pw = D_s0 * (1 - (T_below/T_c)**n_val)
    delta_pw = D_s0 - D_pw
    valid_pw = delta_pw > 0
    ax.loglog(T_below[valid_pw]/T_c, delta_pw[valid_pw]/D_s0, '--', color=color, lw=1.5, 
              label=rf'$\propto T^{{{n_val}}}$')

ax.set_xlabel(r'$T/T_c$', fontsize=12)
ax.set_ylabel(r'$[D_s(0) - D_s(T)]/D_s(0)$', fontsize=12)
ax.set_title('(b) Power-law analysis (log-log)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('report/images/fig2_temperature.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Current Dependence
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): DC current dependence
ax = axes[0]
# GL has 60 points, I_dc has 50. Create matching I_dc for GL
I_dc_gl = np.linspace(0, 60, 60)  # GL model goes to 60 nA
ax.plot(I_dc_gl/I_c, D_s_gl/D_s0, 'b-', lw=2, label='Ginzburg-Landau')
ax.plot(I_dc/I_c, D_s_linear/D_s0, 'r-', lw=2, label='Linear Meissner')
# DC experimental has 85 points - create matching current array
I_dc_exp = np.linspace(0, I_dc.max(), len(D_s_dc_exp))
ax.plot(I_dc_exp/I_c, D_s_dc_exp/D_s0, 'ko', ms=3, alpha=0.7, label='Experimental DC')
ax.set_xlabel(r'DC current $I_{dc}/I_c$', fontsize=12)
ax.set_ylabel(r'$D_s(I)/D_s(0)$', fontsize=12)
ax.set_title('(a) DC current dependence', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel (b): Microwave current dependence
ax = axes[1]
ax.plot(I_mw, D_s_mw_exp/D_s0, 'ko-', ms=3, lw=1.5, label='Experimental MW')
# Fit quadratic to low-power regime
low_power_mask = I_mw < 12
if np.sum(low_power_mask) > 3:
    coeffs = np.polyfit(I_mw[low_power_mask], D_s_mw_exp[low_power_mask]/D_s0, 2)
    I_fit = np.linspace(0, I_mw.max(), 100)
    D_fit = np.polyval(coeffs, I_fit)
    ax.plot(I_fit, D_fit, 'r--', lw=2, label=rf'Quadratic fit: $1 - ({abs(coeffs[0]):.4f})I^2$')
ax.set_xlabel(r'Microwave current amplitude $I_{mw}$ (nA)', fontsize=12)
ax.set_ylabel(r'$D_s(I_{mw})/D_s(0)$', fontsize=12)
ax.set_title('(b) Microwave current dependence', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_current.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: Enhancement Ratio (Quantum Geometry Evidence)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ratio_hole = D_s_exp_hole / D_s_conv
ratio_electron = D_s_exp_electron / D_s_conv
ratio_geom = D_s_geom / D_s_conv

ax.plot(n_eff/1e15, ratio_hole, 'ro-', ms=3, lw=1.5, label=r'$D_s^{\mathrm{exp,hole}} / D_s^{\mathrm{conv}}$')
ax.plot(n_eff/1e15, ratio_electron, 'bs-', ms=3, lw=1.5, label=r'$D_s^{\mathrm{exp,electron}} / D_s^{\mathrm{conv}}$')
ax.plot(n_eff/1e15, ratio_geom, 'g--', lw=2, label=r'$D_s^{\mathrm{geom}} / D_s^{\mathrm{conv}}$')
ax.axhline(y=1, color='k', ls=':', alpha=0.5)
ax.set_xlabel(r'Carrier density $n_{\mathrm{eff}}$ ($10^{15}$ m$^{-2}$)', fontsize=12)
ax.set_ylabel('Enhancement ratio', fontsize=12)
ax.set_title('Superfluid stiffness enhancement beyond conventional Fermi liquid theory', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('report/images/fig4_enhancement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Power-law exponent fitting
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Fit power law to experimental data in the range 0.1 < T/Tc < 0.8
fit_mask = (T_arr > 0.1 * T_c) & (T_arr < 0.8 * T_c) & (D_s_exp_padded > 0)
T_fit = T_arr[fit_mask]
D_fit = D_s_exp_padded[fit_mask]
delta_D_fit = D_s0 - D_fit

# Linear fit in log-log space
log_T = np.log(T_fit / T_c)
log_delta = np.log(delta_D_fit / D_s0)
slope, intercept = np.polyfit(log_T, log_delta, 1)
n_fitted = slope
prefactor = np.exp(intercept)

print(f"Fitted power-law exponent: n = {n_fitted:.2f}")

# Plot
ax.semilogy(T_arr/T_c, (D_s0 - D_s_exp_padded)/D_s0, 'ko', ms=3, alpha=0.6, label='Experimental data')
T_model = np.linspace(0.05, 0.95, 100) * T_c
ax.semilogy(T_model/T_c, prefactor * (T_model/T_c)**n_fitted, 'r-', lw=2, 
            label=rf'Fit: $\propto T^{{{n_fitted:.2f}}}$')
ax.semilogy(T_model/T_c, (D_s0 - np.interp(T_model, T_arr, D_s_bcs_padded))/D_s0, 'b--', lw=1.5, 
            label='BCS (exponential)')

ax.set_xlabel(r'$T/T_c$', fontsize=12)
ax.set_ylabel(r'$[D_s(0) - D_s(T)]/D_s(0)$', fontsize=12)
ax.set_title('Power-law temperature dependence of superfluid stiffness', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0, 1.0)

plt.tight_layout()
plt.savefig('report/images/fig5_powerlaw_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Quadratic current dependence verification
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): DC - verify quadratic
ax = axes[0]
# For GL model, D_s ~ 1 - (I/Ic)^2 at low current
low_I_mask = I_dc_exp < 0.7 * I_c
I_norm = I_dc_exp[low_I_mask] / I_c
D_norm = D_s_dc_exp[low_I_mask] / D_s0
delta_D = 1 - D_norm

# Fit delta_D = a * I^2
coeffs_dc = np.polyfit(I_norm**2, delta_D, 1)
ax.plot(I_norm**2, delta_D, 'ko', ms=4, label='Experimental DC')
I2_model = np.linspace(0, I_norm.max()**2, 50)
ax.plot(I2_model, coeffs_dc[0]*I2_model + coeffs_dc[1], 'r-', lw=2,
        label=rf'Linear fit: slope = {coeffs_dc[0]:.3f}')
ax.set_xlabel(r'$(I_{dc}/I_c)^2$', fontsize=12)
ax.set_ylabel(r'$1 - D_s(I)/D_s(0)$', fontsize=12)
ax.set_title(r'(a) DC: $\Delta D_s \propto I^2$ verification', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel (b): Microwave - verify quadratic
ax = axes[1]
low_mw_mask = I_mw < 15
I_mw_norm = I_mw[low_mw_mask]
D_mw_norm = D_s_mw_exp[low_mw_mask] / D_s0
delta_D_mw = 1 - D_mw_norm

coeffs_mw = np.polyfit(I_mw_norm**2, delta_D_mw, 1)
ax.plot(I_mw_norm**2, delta_D_mw, 'ko', ms=4, label='Experimental MW')
I2_mw_model = np.linspace(0, I_mw_norm.max()**2, 50)
ax.plot(I2_mw_model, coeffs_mw[0]*I2_mw_model + coeffs_mw[1], 'r-', lw=2,
        label=rf'Linear fit: slope = {coeffs_mw[0]:.4f}')
ax.set_xlabel(r'$I_{mw}^2$ (nA$^2$)', fontsize=12)
ax.set_ylabel(r'$1 - D_s(I_{mw})/D_s(0)$', fontsize=12)
ax.set_title(r'(b) Microwave: $\Delta D_s \propto I_{mw}^2$ verification', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig6_quadratic_verification.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================
# Save intermediate results
# ============================================================
results = {
    "carrier_density": {
        "n_eff_range": [float(n_eff.min()), float(n_eff.max())],
        "D_s_conv_range": [float(D_s_conv.min()), float(D_s_conv.max())],
        "D_s_geom_range": [float(D_s_geom.min()), float(D_s_geom.max())],
        "enhancement_ratio_hole_mean": float(np.mean(ratio_hole)),
        "enhancement_ratio_electron_mean": float(np.mean(ratio_electron)),
        "enhancement_ratio_geom_mean": float(np.mean(ratio_geom)),
    },
    "temperature": {
        "T_range": [float(T_arr.min()), float(T_arr.max())],
        "fitted_power_law_exponent": float(n_fitted),
        "T_c": T_c,
    },
    "current": {
        "I_dc_range": [float(I_dc.min()), float(I_dc.max())],
        "I_c": I_c,
        "dc_quadratic_slope": float(coeffs_dc[0]),
        "mw_quadratic_slope": float(coeffs_mw[0]),
    },
    "physical_parameters": {
        "e": e_charge,
        "hbar": hbar,
        "v_f_conventional": v_f_conv,
        "v_f_geometric": v_f_geom,
    }
}

with open('outputs/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAll figures and results saved successfully.")
print(f"Enhancement ratio (hole-doped / conventional): {np.mean(ratio_hole):.1f}x")
print(f"Enhancement ratio (electron-doped / conventional): {np.mean(ratio_electron):.1f}x")
print(f"Fitted power-law exponent: n = {n_fitted:.2f}")
print(f"DC quadratic coefficient: {coeffs_dc[0]:.4f}")
print(f"MW quadratic coefficient: {coeffs_mw[0]:.5f}")
