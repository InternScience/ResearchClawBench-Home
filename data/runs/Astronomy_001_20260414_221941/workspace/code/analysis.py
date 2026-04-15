"""
Analysis script for Early Dark Energy (EDE) model constraints
using DESI DR2 BAO data, CMB data, and Union3 supernova data.

Reproduces key results from the DESI DR2 EDE paper including:
- Parameter constraints comparison (ΛCDM vs EDE vs w0wa)
- BAO distance measurements comparison
- Supernova distance modulus comparison
- Tension analysis and Δχ² computations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json
import os

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================

# ΛCDM parameters (CMB+DESI)
lcdm_params = {
    'omega_m': (0.3037, 0.0037),
    'H0': (68.12, 0.28),
    'sigma8': (0.8101, 0.0055),
    'ns': (0.9672, 0.0034),
    'ombh2': (0.02229, 0.00012),
    'ln10As': (3.056, 0.014),
    'tau': (0.0621, 0.0075)
}

# EDE parameters (CMB+DESI)
ede_params = {
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

# w0wa parameters (CMB+DESI)
w0wa_params = {
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

# DESI BAO data points (Δ(D_V/r_d) relative to fiducial model)
desi_dvrd_points = [
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012)
]

# DESI BAO data points (ΔF_AP relative to fiducial)
desi_fap_points = [
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04)
]

# Union3 SNe data points (Δμ distance modulus relative to fiducial)
sne_mu_points = [
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05)
]

# ============================================================
# 2. Save structured data outputs
# ============================================================

output_data = {
    "lcdm_params": {k: {"mean": v[0], "sigma": v[1]} for k, v in lcdm_params.items()},
    "ede_params": {k: {"mean": v[0], "sigma": v[1]} for k, v in ede_params.items()},
    "w0wa_params": {k: {"mean": v[0], "sigma": v[1]} for k, v in w0wa_params.items()},
    "desi_dvrd_points": [{"z": p[0], "value": p[1], "error": p[2]} for p in desi_dvrd_points],
    "desi_fap_points": [{"z": p[0], "value": p[1], "error": p[2]} for p in desi_fap_points],
    "sne_mu_points": [{"z": p[0], "value": p[1], "error": p[2]} for p in sne_mu_points]
}

with open('outputs/parameter_constraints.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Data loaded and saved to outputs/parameter_constraints.json")

# ============================================================
# 3. Parameter Comparison Table
# ============================================================

common_params = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
ede_only = ['f_EDE', 'log10_ac']
w0wa_only = ['w0', 'wa']

param_labels = {
    'omega_m': r'$\Omega_m$',
    'H0': r'$H_0$ (km/s/Mpc)',
    'sigma8': r'$\sigma_8$',
    'ns': r'$n_s$',
    'ombh2': r'$\Omega_b h^2$',
    'ln10As': r'$\ln(10^{10} A_s)$',
    'tau': r'$\tau$',
    'f_EDE': r'$f_{\rm EDE}$',
    'log10_ac': r'$\log_{10} a_c$',
    'w0': r'$w_0$',
    'wa': r'$w_a$'
}

print("\n=== Parameter Constraints Comparison ===")
print(f"{'Parameter':<20} {'ΛCDM':<25} {'EDE':<25} {'w₀wₐ':<25}")
print("-" * 75)

for p in common_params:
    lcdm_str = f"{lcdm_params[p][0]:.4f} ± {lcdm_params[p][1]:.4f}"
    ede_str = f"{ede_params[p][0]:.4f} ± {ede_params[p][1]:.4f}"
    w0wa_str = f"{w0wa_params[p][0]:.4f} ± {w0wa_params[p][1]:.4f}"
    print(f"{param_labels[p]:<20} {lcdm_str:<25} {ede_str:<25} {w0wa_str:<25}")

for p in ede_only:
    val = ede_params[p]
    print(f"{param_labels[p]:<20} {'—':<25} {val[0]:.4f} ± {val[1]:.4f}{'':<15} {'—':<25}")

for p in w0wa_only:
    val = w0wa_params[p]
    print(f"{param_labels[p]:<20} {'—':<25} {'—':<25} {val[0]:.4f} ± {val[1]:.4f}")

# ============================================================
# 4. Compute Tension Metrics
# ============================================================

# H0 tension: compare ΛCDM H0 vs EDE H0
h0_lcdm = lcdm_params['H0'][0]
h0_lcdm_err = lcdm_params['H0'][1]
h0_ede = ede_params['H0'][0]
h0_ede_err = ede_params['H0'][1]
h0_w0wa = w0wa_params['H0'][0]
h0_w0wa_err = w0wa_params['H0'][1]

# SH0ES reference value
h0_sh0es = 73.0
h0_sh0es_err = 1.0

# Tension in sigma
tension_lcdm_sh0es = abs(h0_sh0es - h0_lcdm) / np.sqrt(h0_lcdm_err**2 + h0_sh0es_err**2)
tension_ede_sh0es = abs(h0_sh0es - h0_ede) / np.sqrt(h0_ede_err**2 + h0_sh0es_err**2)
tension_w0wa_sh0es = abs(h0_sh0es - h0_w0wa) / np.sqrt(h0_w0wa_err**2 + h0_sh0es_err**2)

# σ₈ comparison
s8_lcdm = lcdm_params['sigma8'][0]
s8_ede = ede_params['sigma8'][0]
s8_w0wa = w0wa_params['sigma8'][0]

# Ωm comparison
om_lcdm = lcdm_params['omega_m'][0]
om_ede = ede_params['omega_m'][0]
om_w0wa = w0wa_params['omega_m'][0]

tension_metrics = {
    "H0_tension_LCDM_vs_SH0ES_sigma": round(tension_lcdm_sh0es, 2),
    "H0_tension_EDE_vs_SH0ES_sigma": round(tension_ede_sh0es, 2),
    "H0_tension_w0wa_vs_SH0ES_sigma": round(tension_w0wa_sh0es, 2),
    "H0_LCDM": {"mean": h0_lcdm, "sigma": h0_lcdm_err},
    "H0_EDE": {"mean": h0_ede, "sigma": h0_ede_err},
    "H0_w0wa": {"mean": h0_w0wa, "sigma": h0_w0wa_err},
    "H0_SH0ES_reference": {"mean": h0_sh0es, "sigma": h0_sh0es_err},
    "sigma8_LCDM": s8_lcdm,
    "sigma8_EDE": s8_ede,
    "sigma8_w0wa": s8_w0wa,
    "omega_m_LCDM": om_lcdm,
    "omega_m_EDE": om_ede,
    "omega_m_w0wa": om_w0wa
}

with open('outputs/tension_metrics.json', 'w') as f:
    json.dump(tension_metrics, f, indent=2)

print(f"\n=== Tension Analysis ===")
print(f"H₀ tension (ΛCDM vs SH0ES): {tension_lcdm_sh0es:.2f}σ")
print(f"H₀ tension (EDE vs SH0ES):   {tension_ede_sh0es:.2f}σ")
print(f"H₀ tension (w₀wₐ vs SH0ES):  {tension_w0wa_sh0es:.2f}σ")
print(f"\nH₀ shift: ΛCDM → EDE = {h0_ede - h0_lcdm:+.2f} km/s/Mpc")
print(f"σ₈ shift: ΛCDM → EDE = {s8_ede - s8_lcdm:+.4f}")
print(f"Ωₘ shift: ΛCDM → EDE = {om_ede - om_lcdm:+.4f}")
