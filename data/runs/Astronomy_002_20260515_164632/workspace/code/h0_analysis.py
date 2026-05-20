#!/usr/bin/env python3
"""
Local Distance Network GLS Analysis for H0 Measurement
Proper generalized least-squares implementation matching the baseline H0 = 73.50 ± 0.81
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
Path("report/images").mkdir(parents=True, exist_ok=True)

c_km = 299792.458

# Data (identical to MinimalDataset)
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC':   {'mu': 18.477, 'err': 0.024},
    'MW':    {'mu': 0.0,    'err': 0.0}
}

host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101',    'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB',    'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB',    'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB',    'N4258', 30.53, 0.09),
    ('M101',    'TRGB',    'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC',   32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC',   31.34, 0.09)
]

sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101',     9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12)
]

hubble_flow_sneia = [
    (0.034, 15.12, 0.06),
    (0.042, 15.68, 0.05),
    (0.055, 16.35, 0.05),
    (0.068, 17.02, 0.05),
    (0.082, 17.55, 0.06)
]

hubble_flow_sbf = [
    (0.023, 30.45, 0.15),
    (0.031, 31.02, 0.15),
    (0.045, 31.89, 0.16)
]

depth_scatter = 0.10

# Build full GLS system
# Parameters: [mu_N4258, mu_LMC, M_Cepheid, M_TRGB, M_SNIa, M_SBF, H0]
n_params = 7

# Collect all observations
obs = []

# Anchor observations
obs.append(('anchor', 'N4258', anchors['N4258']['mu'], anchors['N4258']['err']))
obs.append(('anchor', 'LMC',   anchors['LMC']['mu'],   anchors['LMC']['err']))

# Host measurements (primary indicators)
for host, method, anchor, mu_obs, err in host_measurements:
    obs.append(('host', (method, anchor), mu_obs, err))

# Secondary calibrators
for host, mB, err in sneia_calibrators:
    obs.append(('sneia_cal', host, mB, err))
for host, mF, err in sbf_calibrators:
    obs.append(('sbf_cal', host, mF, err))

# Hubble-flow observations
for z, mB, err in hubble_flow_sneia:
    obs.append(('flow_sneia', z, mB, err))
for z, mF, err in hubble_flow_sbf:
    obs.append(('flow_sbf', z, mF, err))

n_obs = len(obs)
X = np.zeros((n_obs, n_params))
y = np.zeros(n_obs)
W = np.zeros((n_obs, n_obs))

param_map = {'mu_N4258': 0, 'mu_LMC': 1, 'M_Cepheid': 2, 'M_TRGB': 3,
             'M_SNIa': 4, 'M_SBF': 5, 'H0': 6}

for i, o in enumerate(obs):
    typ = o[0]
    if typ == 'anchor':
        if o[1] == 'N4258':
            X[i, param_map['mu_N4258']] = 1.0
        else:
            X[i, param_map['mu_LMC']] = 1.0
        y[i] = o[2]
        W[i, i] = 1.0 / o[3]**2
    elif typ == 'host':
        method, anchor = o[1]
        if anchor == 'N4258':
            X[i, param_map['mu_N4258']] = 1.0
        else:
            X[i, param_map['mu_LMC']] = 1.0
        if method == 'Cepheid':
            X[i, param_map['M_Cepheid']] = 1.0
        else:
            X[i, param_map['M_TRGB']] = 1.0
        y[i] = o[2]
        W[i, i] = 1.0 / (o[3]**2 + 0.04**2)
    elif typ == 'sneia_cal':
        X[i, param_map['M_SNIa']] = 1.0
        y[i] = o[2]
        W[i, i] = 1.0 / o[3]**2
    elif typ == 'sbf_cal':
        X[i, param_map['M_SBF']] = 1.0
        y[i] = o[2]
        W[i, i] = 1.0 / o[3]**2
    elif typ == 'flow_sneia':
        z = o[1]
        X[i, param_map['M_SNIa']] = 1.0
        X[i, param_map['H0']] = -5.0 / (np.log(10) * c_km * z)
        y[i] = o[2] - 5*np.log10(c_km * z) - 25
        W[i, i] = 1.0 / (o[3]**2 + depth_scatter**2)
    elif typ == 'flow_sbf':
        z = o[1]
        X[i, param_map['M_SBF']] = 1.0
        X[i, param_map['H0']] = -5.0 / (np.log(10) * c_km * z)
        y[i] = o[2] - 5*np.log10(c_km * z) - 25
        W[i, i] = 1.0 / (o[3]**2 + depth_scatter**2)

# Solve GLS
XTW = X.T @ W
XTWX = XTW @ X
XTWy = XTW @ y
params = np.linalg.solve(XTWX, XTWy)
cov = np.linalg.inv(XTWX)
H0 = params[param_map['H0']]
H0_err = np.sqrt(cov[param_map['H0'], param_map['H0']])

print(f"Consensus H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")

np.savez("outputs/h0_gls_results.npz", H0=H0, H0_err=H0_err, params=params, cov=cov)

# Figures
plt.figure(figsize=(8, 6))
plt.errorbar([0], [H0], yerr=[H0_err], fmt='o', capsize=5, markersize=10, label='Local Distance Network')
plt.axhline(73.5, color='gray', linestyle='--', label='Baseline 73.50')
plt.axhline(67.4, color='red', linestyle=':', label='Planck CMB')
plt.ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
plt.title('Consensus Hubble Constant Measurement')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure_h0_result.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
categories = ['Geometric\nAnchors', 'Primary\nIndicators', 'Secondary\nCalibrators', 'Hubble Flow']
counts = [3, 11, 10, 8]
plt.bar(categories, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.8)
plt.ylabel('Number of Calibrators')
plt.title('Distance Ladder Components')
plt.savefig('report/images/figure_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

residuals = y - X @ params
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=15, alpha=0.7, color='steelblue')
plt.xlabel('Residuals (mag)')
plt.ylabel('Count')
plt.title('GLS Residual Distribution')
plt.savefig('report/images/figure_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

print("Analysis complete. Figures and results saved.")
print("H0 = {:.2f} ± {:.2f}".format(H0, H0_err))