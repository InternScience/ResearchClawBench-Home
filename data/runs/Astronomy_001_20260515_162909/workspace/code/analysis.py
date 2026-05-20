#!/usr/bin/env python3
"""
Analysis of Early Dark Energy (EDE) model constraints from DESI DR2 BAO data.
Reproduces and visualizes key results from the DESI DR2 EDE paper.
"""

import numpy as np
import json
import os

# ============================================================
# 1. Parse the data file
# ============================================================

data = {}

# Manually define the data structures as in the file
data['lcdm_params'] = {
    'omega_m': (0.3037, 0.0037),
    'H0': (68.12, 0.28),
    'sigma8': (0.8101, 0.0055),
    'ns': (0.9672, 0.0034),
    'ombh2': (0.02229, 0.00012),
    'ln10As': (3.056, 0.014),
    'tau': (0.0621, 0.0075)
}

data['ede_params'] = {
    'omega_m': (0.2999, 0.0038),
    'H0': (70.9, 1.0),
    'sigma8': (0.8283, 0.0093),
    'f_EDE': (0.093, 0.031),
    'log10_ac': (-3.564, 0.075),
    'ns': (0.9817, 0.0063),
    'ombh2': (0.02241, 0.00018),
    'ln10As': (3.067, 0.017),
    'tau': (0.0582, 0.0074)
}

data['w0wa_params'] = {
    'omega_m': (0.353, 0.021),
    'H0': (63.5, 1.9),
    'sigma8': (0.780, 0.016),
    'w0': (-0.42, 0.21),
    'wa': (-1.75, 0.58),
    'ns': (0.9632, 0.0037),
    'ombh2': (0.02218, 0.00013),
    'ln10As': (3.037, 0.013),
    'tau': (0.0520, 0.0071)
}

data['desi_dvrd_points'] = [
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012)
]

data['desi_fap_points'] = [
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04)
]

data['sne_mu_points'] = [
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05)
]

# ============================================================
# 2. Compute derived quantities and chi-squared values
# ============================================================

# Reference SH0ES H0 value
H0_shoes = 73.52  # km/s/Mpc
H0_shoes_err = 1.62

# SH0ES H0 constraint from each model (compute tension sigma)
def tension_sigma(H0_model, H0_model_err, H0_ref, H0_ref_err):
    diff = abs(H0_model - H0_ref)
    combined_err = np.sqrt(H0_model_err**2 + H0_ref_err**2)
    return diff / combined_err

# Tension with SH0ES
tension_lcdm = tension_sigma(data['lcdm_params']['H0'][0], data['lcdm_params']['H0'][1], H0_shoes, H0_shoes_err)
tension_ede = tension_sigma(data['ede_params']['H0'][0], data['ede_params']['H0'][1], H0_shoes, H0_shoes_err)
tension_w0wa = tension_sigma(data['w0wa_params']['H0'][0], data['w0wa_params']['H0'][1], H0_shoes, H0_shoes_err)

print(f"SH0ES H0 tension:")
print(f"  ΛCDM:  {tension_lcdm:.1f}σ")
print(f"  EDE:   {tension_ede:.1f}σ")
print(f"  w₀wₐ:  {tension_w0wa:.1f}σ")

# Compute S8 = sigma8 * (omega_m / 0.3)^0.5 for each model
def compute_S8(omega_m, sigma8):
    return sigma8 * np.sqrt(omega_m / 0.3)

S8_lcdm = compute_S8(data['lcdm_params']['omega_m'][0], data['lcdm_params']['sigma8'][0])
S8_ede = compute_S8(data['ede_params']['omega_m'][0], data['ede_params']['sigma8'][0])
S8_w0wa = compute_S8(data['w0wa_params']['omega_m'][0], data['w0wa_params']['sigma8'][0])

# Propagate errors for S8
def S8_error(om, om_err, s8, s8_err):
    dS8_dom = s8 * 0.5 / np.sqrt(0.3 * om) 
    dS8_ds8 = np.sqrt(om / 0.3)
    return np.sqrt((dS8_dom * om_err)**2 + (dS8_ds8 * s8_err)**2)

S8_err_lcdm = S8_error(*data['lcdm_params']['omega_m'], *data['lcdm_params']['sigma8'])
S8_err_ede = S8_error(*data['ede_params']['omega_m'], *data['ede_params']['sigma8'])
S8_err_w0wa = S8_error(*data['w0wa_params']['omega_m'], *data['w0wa_params']['sigma8'])

print(f"\nS8 values:")
print(f"  ΛCDM:  {S8_lcdm:.4f} ± {S8_err_lcdm:.4f}")
print(f"  EDE:   {S8_ede:.4f} ± {S8_err_ede:.4f}")
print(f"  w₀wₐ:  {S8_w0wa:.4f} ± {S8_err_w0wa:.4f}")

# ============================================================
# 3. Compute chi-squared for BAO data (simplified model comparison)
# ============================================================

# DESI BAO D_V/r_d measurements (absolute values based on typical DESI DR2)
# These are approximate absolute D_V/r_d from DESI DR2 for the analysis
# Based on DESI DR2 published values for BAO measurements
desi_dvrd_absolute = {
    'z': [0.295, 0.510, 0.700, 0.934, 1.100, 1.320, 2.330],
    'value': [7.93, 10.08, 11.55, 13.62, 15.45, 17.89, 26.07],
    'error': [0.15, 0.12, 0.14, 0.16, 0.20, 0.27, 0.45]
}

# Compute chi-squared for each model with BAO data
# Using the deviation from fiducial as model differences
def chi2_bao(model_params, z_points, fiducial_devs, fid_errs):
    """Compute approximate chi2 for BAO data given model parameters."""
    omega_m = model_params['omega_m'][0]
    H0 = model_params['H0'][0]
    # Simple model: DV/rd ~ (D_V/rd)_fid + delta_model(z, Omega_m, H0)
    # For simplicity, use Omega_m and H0 to estimate DV shift relative to fiducial
    # This is approximate - in a full analysis you'd compute DV numerically
    
    # A simplified approximation for DV/r_d shift
    chi2 = 0
    for i, z in enumerate(z_points):
        # Model prediction relative to fiducial (simplified)
        model_shift = (omega_m - 0.3153) * 2.0 + (H0 - 67.36) * 0.003  # approximate sensitivity
        model_shift *= (1 + z) / (1 + 1.0)  # redshift scaling
        residual = fiducial_devs[i] - model_shift
        chi2 += (residual / fid_errs[i])**2
    return chi2

z_bao = [p[0] for p in data['desi_dvrd_points']]
dvrd_devs = [p[1] for p in data['desi_dvrd_points']]
dvrd_errs = [p[2] for p in data['desi_dvrd_points']]

chi2_lcdm_bao = chi2_bao(data['lcdm_params'], z_bao, dvrd_devs, dvrd_errs)
chi2_ede_bao = chi2_bao(data['ede_params'], z_bao, dvrd_devs, dvrd_errs)
chi2_w0wa_bao = chi2_bao(data['w0wa_params'], z_bao, dvrd_devs, dvrd_errs)

print(f"\nSimplified BAO chi2:")
print(f"  ΛCDM:  {chi2_lcdm_bao:.1f}")
print(f"  EDE:   {chi2_ede_bao:.1f}")
print(f"  w₀wₐ:  {chi2_w0wa_bao:.1f}")

# ============================================================
# 4. Save intermediate results
# ============================================================

results = {
    'models': {
        'LCDM': {k: {'mean': v[0], 'sigma': v[1]} for k, v in data['lcdm_params'].items()},
        'EDE': {k: {'mean': v[0], 'sigma': v[1]} for k, v in data['ede_params'].items()},
        'w0wa': {k: {'mean': v[0], 'sigma': v[1]} for k, v in data['w0wa_params'].items()}
    },
    'derived': {
        'LCDM': {'S8': S8_lcdm, 'S8_err': S8_err_lcdm},
        'EDE': {'S8': S8_ede, 'S8_err': S8_err_ede},
        'w0wa': {'S8': S8_w0wa, 'S8_err': S8_err_w0wa}
    },
    'tension_with_SH0ES': {
        'LCDM': tension_lcdm,
        'EDE': tension_ede,
        'w0wa': tension_w0wa
    },
    'chi2_BAO': {
        'LCDM': chi2_lcdm_bao,
        'EDE': chi2_ede_bao,
        'w0wa': chi2_w0wa_bao
    },
    'SH0ES_reference': {'H0': H0_shoes, 'H0_err': H0_shoes_err}
}

with open('outputs/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nResults saved to outputs/analysis_results.json")
print("Analysis complete!")
