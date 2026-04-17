#!/usr/bin/env python3
"""
H0 Distance Network: Complete GLS Analysis and Figure Generation
================================================================
"""

import numpy as np
import json
import os
import sys
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
from collections import defaultdict

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# DATA
# ============================================================
c_km = 299792.458

anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC': {'mu': 18.477, 'err': 0.024},
    'MW': {'mu': 0.0, 'err': 0.0}
}

host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101', 'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB', 'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB', 'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB', 'N4258', 30.53, 0.09),
    ('M101', 'TRGB', 'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC', 32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC', 31.34, 0.09)
]

sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101', 9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12)
]

hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]

hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250)
]

method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'): 0.03,
    ('Cepheid', 'MW'): 0.02,
    ('TRGB', 'N4258'): 0.05
}

host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo'
}

depth_scatter = 0.10

# ============================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================

def luminosity_distance(z, H0, Om=0.3):
    def integrand(zp):
        return 1.0 / np.sqrt(Om * (1+zp)**3 + (1 - Om))
    result, _ = quad(integrand, 0, z)
    return (c_km / H0) * (1 + z) * result

def distance_modulus_cosmo(z, H0, Om=0.3):
    return 5 * np.log10(luminosity_distance(z, H0, Om)) + 25

# ============================================================
# GLS FRAMEWORK
# ============================================================

def build_and_solve_gls(use_anchors=None, use_methods=None,
                        use_sneia_hf=True, use_sbf_hf=True):
    if use_anchors is None:
        use_anchors = ['N4258', 'LMC']
    if use_methods is None:
        use_methods = ['Cepheid', 'TRGB']
    
    sneia_hosts = {h for h, _, _ in sneia_calibrators}
    measured_hosts = set()
    for host, method, anchor, _, _ in host_measurements:
        if anchor in use_anchors and method in use_methods:
            measured_hosts.add(host)
    calibrated_hosts = sorted(sneia_hosts & measured_hosts)
    
    if not calibrated_hosts:
        return None
    
    param_names = [f'mu_{h}' for h in calibrated_hosts]
    sbf_groups = []
    if use_sbf_hf:
        sbf_groups = sorted(set(host_group[h] for h in host_group))
        param_names.extend([f'mu_{g}' for g in sbf_groups])
    
    param_names.extend(['M_B'])
    if use_sbf_hf:
        param_names.append('M_SBF')
    param_names.append('5logH0')
    
    n_params = len(param_names)
    param_idx = {name: i for i, name in enumerate(param_names)}
    
    y_list, L_list, sigma_list, labels = [], [], [], []
    
    # Primary indicators
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if anchor not in use_anchors or method not in use_methods:
            continue
        if host not in calibrated_hosts:
            continue
        err_a = anchors[anchor]['err']
        err_m = method_anchor_err.get((method, anchor), 0.0)
        err_total = np.sqrt(err_meas**2 + err_a**2 + err_m**2)
        row = np.zeros(n_params)
        row[param_idx[f'mu_{host}']] = 1.0
        y_list.append(mu_meas)
        L_list.append(row)
        sigma_list.append(err_total)
        labels.append(f'Primary:{host}/{method}/{anchor}')
    
    # SN Ia calibrators
    for host, mB, err_mB in sneia_calibrators:
        if host not in calibrated_hosts:
            continue
        row = np.zeros(n_params)
        row[param_idx[f'mu_{host}']] = 1.0
        row[param_idx['M_B']] = 1.0
        y_list.append(mB)
        L_list.append(row)
        sigma_list.append(err_mB)
        labels.append(f'SNcal:{host}')
    
    # SBF system
    if use_sbf_hf:
        host_to_grp = {'NGC1316': 'Fornax', 'NGC1365': 'Fornax'}
        for host, grp in host_to_grp.items():
            if host in calibrated_hosts and f'mu_{grp}' in param_idx:
                row = np.zeros(n_params)
                row[param_idx[f'mu_{host}']] = 1.0
                row[param_idx[f'mu_{grp}']] = -1.0
                y_list.append(0.0)
                L_list.append(row)
                sigma_list.append(depth_scatter)
                labels.append(f'Group:{host}->{grp}')
        
        for host, mF, err_mF in sbf_calibrators:
            grp = host_group[host]
            if f'mu_{grp}' not in param_idx:
                continue
            row = np.zeros(n_params)
            row[param_idx[f'mu_{grp}']] = 1.0
            row[param_idx['M_SBF']] = 1.0
            y_list.append(mF)
            L_list.append(row)
            sigma_list.append(np.sqrt(err_mF**2 + depth_scatter**2))
            labels.append(f'SBFcal:{host}')
    
    # Hubble flow SN Ia
    H0_ref = 70.0
    Om = 0.3
    if use_sneia_hf:
        for z, mB, err_mB, vpec in hubble_flow_sneia:
            cz = c_km * z
            mu_exact = distance_modulus_cosmo(z, H0_ref, Om)
            mu_linear = 5*np.log10(cz/H0_ref) + 25
            delta_z = mu_exact - mu_linear
            y_val = mB - 5*np.log10(cz) - 25 - delta_z
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mB**2 + err_vpec**2)
            row = np.zeros(n_params)
            row[param_idx['M_B']] = 1.0
            row[param_idx['5logH0']] = -1.0
            y_list.append(y_val)
            L_list.append(row)
            sigma_list.append(err_total)
            labels.append(f'HFSN:z={z:.3f}')
    
    # Hubble flow SBF
    if use_sbf_hf and 'M_SBF' in param_idx:
        for z, mF, err_mF, vpec in hubble_flow_sbf:
            cz = c_km * z
            mu_exact = distance_modulus_cosmo(z, H0_ref, Om)
            mu_linear = 5*np.log10(cz/H0_ref) + 25
            delta_z = mu_exact - mu_linear
            y_val = mF - 5*np.log10(cz) - 25 - delta_z
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mF**2 + err_vpec**2)
            row = np.zeros(n_params)
            row[param_idx['M_SBF']] = 1.0
            row[param_idx['5logH0']] = -1.0
            y_list.append(y_val)
            L_list.append(row)
            sigma_list.append(err_total)
            labels.append(f'HFSBF:z={z:.3f}')
    
    y = np.array(y_list)
    L = np.array(L_list)
    sigma = np.array(sigma_list)
    n_obs = len(y)
    C = np.diag(sigma**2)
    
    # Off-diagonal covariance for shared anchors
    for i in range(n_obs):
        for j in range(i+1, n_obs):
            if labels[i].startswith('Primary:') and labels[j].startswith('Primary:'):
                pi = labels[i].split('/')
                pj = labels[j].split('/')
                ai, aj = pi[-1], pj[-1]
                mi, mj = pi[1], pj[1]
                if ai == aj:
                    cov = anchors[ai]['err']**2
                    if mi == mj:
                        cov += method_anchor_err.get((mi, ai), 0.0)**2
                    C[i, j] = cov
                    C[j, i] = cov
    
    try:
        C_inv = np.linalg.inv(C)
        LtCiL = L.T @ C_inv @ L
        cq = np.linalg.inv(LtCiL)
        q_best = cq @ (L.T @ C_inv @ y)
        residuals = y - L @ q_best
        chi2 = float(residuals.T @ C_inv @ residuals)
        dof = n_obs - n_params
        
        idx = param_idx['5logH0']
        val = q_best[idx]
        err = np.sqrt(cq[idx, idx])
        H0 = 10**(val/5.0)
        H0_err = H0 * np.log(10) / 5.0 * err
        
        MB = q_best[param_idx['M_B']]
        MB_err = np.sqrt(cq[param_idx['M_B'], param_idx['M_B']])
        
        return {
            'H0': float(H0), 'H0_err': float(H0_err),
            'M_B': float(MB), 'M_B_err': float(MB_err),
            '5logH0': float(val), '5logH0_err': float(err),
            'chi2': chi2, 'dof': dof,
            'q_best': q_best, 'cq': cq,
            'param_names': param_names, 'param_idx': param_idx,
            'y': y, 'L': L, 'C': C, 'labels': labels,
            'calibrated_hosts': calibrated_hosts
        }
    except np.linalg.LinAlgError:
        return None

# ============================================================
# RUN ALL ANALYSES
# ============================================================

print("Running baseline analysis...")
baseline = build_and_solve_gls(
    use_anchors=['N4258', 'LMC'],
    use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=True, use_sbf_hf=True
)

print(f"Baseline H0 = {baseline['H0']:.2f} +/- {baseline['H0_err']:.2f} km/s/Mpc")
print(f"Baseline M_B = {baseline['M_B']:.4f} +/- {baseline['M_B_err']:.4f}")
print(f"chi2/dof = {baseline['chi2']:.1f}/{baseline['dof']}")

# Variants
variant_configs = {
    'Baseline (all anchors, all methods)': 
        dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
             use_sneia_hf=True, use_sbf_hf=True),
    'N4258 anchor only': 
        dict(use_anchors=['N4258'], use_methods=['Cepheid', 'TRGB'],
             use_sneia_hf=True, use_sbf_hf=True),
    'LMC anchor only (Cepheids)': 
        dict(use_anchors=['LMC'], use_methods=['Cepheid'],
             use_sneia_hf=True, use_sbf_hf=False),
    'Cepheids only (both anchors)': 
        dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid'],
             use_sneia_hf=True, use_sbf_hf=True),
    'TRGB only (N4258)': 
        dict(use_anchors=['N4258'], use_methods=['TRGB'],
             use_sneia_hf=True, use_sbf_hf=True),
    'SN Ia Hubble flow only': 
        dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
             use_sneia_hf=True, use_sbf_hf=False),
    'SBF Hubble flow only': 
        dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
             use_sneia_hf=False, use_sbf_hf=True),
    'N4258 + Cepheids only': 
        dict(use_anchors=['N4258'], use_methods=['Cepheid'],
             use_sneia_hf=True, use_sbf_hf=True),
    'N4258 + TRGB only': 
        dict(use_anchors=['N4258'], use_methods=['TRGB'],
             use_sneia_hf=True, use_sbf_hf=True),
}

variant_results = {}
print("\nRunning variants...")
for name, config in variant_configs.items():
    res = build_and_solve_gls(**config)
    if res is not None:
        variant_results[name] = res
        print(f"  {name}: H0 = {res['H0']:.2f} +/- {res['H0_err']:.2f}")
    else:
        print(f"  {name}: FAILED (singular matrix)")

# ============================================================
# SAVE RESULTS
# ============================================================

results_output = {
    'baseline': {
        'H0': round(baseline['H0'], 2),
        'H0_err': round(baseline['H0_err'], 2),
        'M_B': round(baseline['M_B'], 4),
        'M_B_err': round(baseline['M_B_err'], 4),
        'chi2': round(baseline['chi2'], 2),
        'dof': baseline['dof'],
        'chi2_per_dof': round(baseline['chi2']/max(1, baseline['dof']), 2),
        'parameters': {
            name: {'value': round(float(baseline['q_best'][i]), 4),
                   'error': round(float(np.sqrt(baseline['cq'][i,i])), 4)}
            for i, name in enumerate(baseline['param_names'])
        }
    },
    'variants': {
        name: {
            'H0': round(res['H0'], 2),
            'H0_err': round(res['H0_err'], 2),
            'M_B': round(res['M_B'], 4),
            'chi2': round(res['chi2'], 2),
            'dof': res['dof']
        }
        for name, res in variant_results.items()
    },
    'CMB_comparison': {
        'Planck_H0': 67.4,
        'Planck_H0_err': 0.5,
        'tension_sigma': round(
            (baseline['H0'] - 67.4) / np.sqrt(baseline['H0_err']**2 + 0.5**2), 1
        )
    },
    'host_distances': {
        name: {
            'value': round(float(baseline['q_best'][i]), 4),
            'error': round(float(np.sqrt(baseline['cq'][i,i])), 4)
        }
        for i, name in enumerate(baseline['param_names'])
        if name.startswith('mu_')
    }
}

with open('outputs/h0_results.json', 'w') as f:
    json.dump(results_output, f, indent=2)

# Save variant comparison table
variant_table = []
for name, res in variant_results.items():
    variant_table.append({
        'variant': name,
        'H0': round(res['H0'], 2),
        'H0_err': round(res['H0_err'], 2),
        'M_B': round(res['M_B'], 4),
        'chi2': round(res['chi2'], 2),
        'dof': res['dof']
    })

with open('outputs/variant_table.json', 'w') as f:
    json.dump(variant_table, f, indent=2)

# Save host distance table
host_dist_table = []
for host in baseline['calibrated_hosts']:
    key = f'mu_{host}'
    idx = baseline['param_idx'][key]
    mu = baseline['q_best'][idx]
    err = np.sqrt(baseline['cq'][idx, idx])
    D_Mpc = 10**((mu - 25)/5)
    host_dist_table.append({
        'host': host,
        'mu': round(float(mu), 4),
        'mu_err': round(float(err), 4),
        'D_Mpc': round(float(D_Mpc), 1)
    })

with open('outputs/host_distances.json', 'w') as f:
    json.dump(host_dist_table, f, indent=2)

print("\nResults saved to outputs/")
print(f"Baseline: H0 = {baseline['H0']:.2f} +/- {baseline['H0_err']:.2f} km/s/Mpc")
