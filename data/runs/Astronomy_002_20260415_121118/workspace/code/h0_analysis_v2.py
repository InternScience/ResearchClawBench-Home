#!/usr/bin/env python3
"""
H0 Distance Network: Full GLS Analysis
=======================================
Implements the covariance-weighted GLS framework for measuring H0
from the minimal dataset, following the SH0ES methodology.
"""

import numpy as np
import json
import os
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

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
# COSMOLOGICAL DISTANCE
# ============================================================

def luminosity_distance(z, H0, Om=0.3):
    """Luminosity distance in Mpc for flat LCDM."""
    def integrand(zp):
        return 1.0 / np.sqrt(Om * (1+zp)**3 + (1 - Om))
    result, _ = quad(integrand, 0, z)
    d_L = (c_km / H0) * (1 + z) * result
    return d_L

def distance_modulus_cosmo(z, H0, Om=0.3):
    """Distance modulus for flat LCDM."""
    d_L = luminosity_distance(z, H0, Om)
    return 5 * np.log10(d_L) + 25

# ============================================================
# METHOD 1: Direct chi-squared minimization
# ============================================================

def compute_chi2_direct(H0_trial, use_anchors=None, use_methods=None, 
                        use_sneia_hf=True, use_sbf_hf=True, Om=0.3):
    """
    Compute chi-squared for a given H0 by:
    1. Using primary indicators to get host distance moduli
    2. Using SN Ia calibrators to get M_B
    3. Comparing Hubble flow predictions with observations
    """
    if use_anchors is None:
        use_anchors = ['N4258', 'LMC']
    if use_methods is None:
        use_methods = ['Cepheid', 'TRGB']
    
    # Step 1: Get host distance moduli (weighted average of all measurements)
    from collections import defaultdict
    host_mu_vals = defaultdict(list)
    host_mu_errs = defaultdict(list)
    
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if anchor not in use_anchors or method not in use_methods:
            continue
        err_a = anchors[anchor]['err']
        err_m = method_anchor_err.get((method, anchor), 0.0)
        err_total = np.sqrt(err_meas**2 + err_a**2 + err_m**2)
        host_mu_vals[host].append(mu_meas)
        host_mu_errs[host].append(err_total)
    
    # Weighted average for each host
    host_mu = {}
    host_mu_err = {}
    for host in host_mu_vals:
        vals = np.array(host_mu_vals[host])
        errs = np.array(host_mu_errs[host])
        weights = 1.0 / errs**2
        mu_avg = np.sum(weights * vals) / np.sum(weights)
        err_avg = 1.0 / np.sqrt(np.sum(weights))
        host_mu[host] = mu_avg
        host_mu_err[host] = err_avg
    
    # Step 2: Get M_B from calibrators
    MB_vals = []
    MB_errs = []
    for host, mB, err_mB in sneia_calibrators:
        if host not in host_mu:
            continue
        MB = mB - host_mu[host]
        err_MB = np.sqrt(err_mB**2 + host_mu_err[host]**2)
        MB_vals.append(MB)
        MB_errs.append(err_MB)
    
    MB_vals = np.array(MB_vals)
    MB_errs = np.array(MB_errs)
    weights_MB = 1.0 / MB_errs**2
    MB_best = np.sum(weights_MB * MB_vals) / np.sum(weights_MB)
    MB_err = 1.0 / np.sqrt(np.sum(weights_MB))
    
    # Step 3: Chi-squared from Hubble flow
    chi2 = 0.0
    
    if use_sneia_hf:
        for z, mB, err_mB, vpec in hubble_flow_sneia:
            mu_pred = distance_modulus_cosmo(z, H0_trial, Om)
            mB_pred = mu_pred + MB_best
            cz = c_km * z
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mB**2 + err_vpec**2 + MB_err**2)
            chi2 += ((mB - mB_pred) / err_total)**2
    
    if use_sbf_hf:
        # For SBF, we need M_SBF calibration
        # Get group distances from hosts in those groups
        group_mu = {}
        group_mu_err = {}
        host_to_group = {'NGC1316': 'Fornax', 'NGC1365': 'Fornax'}
        for host, grp in host_to_group.items():
            if host in host_mu:
                if grp not in group_mu:
                    group_mu[grp] = host_mu[host]
                    group_mu_err[grp] = host_mu_err[host]
                else:
                    # Weighted average
                    w1 = 1.0/group_mu_err[grp]**2
                    w2 = 1.0/host_mu_err[host]**2
                    group_mu[grp] = (w1*group_mu[grp] + w2*host_mu[host])/(w1+w2)
                    group_mu_err[grp] = 1.0/np.sqrt(w1+w2)
        
        # Get M_SBF from calibrators
        MSBF_vals = []
        MSBF_errs = []
        for host, mF, err_mF in sbf_calibrators:
            grp = host_group[host]
            if grp in group_mu:
                MSBF = mF - group_mu[grp]
                err_MSBF = np.sqrt(err_mF**2 + group_mu_err[grp]**2 + depth_scatter**2)
                MSBF_vals.append(MSBF)
                MSBF_errs.append(err_MSBF)
        
        if MSBF_vals:
            MSBF_vals = np.array(MSBF_vals)
            MSBF_errs = np.array(MSBF_errs)
            weights_MSBF = 1.0 / MSBF_errs**2
            MSBF_best = np.sum(weights_MSBF * MSBF_vals) / np.sum(weights_MSBF)
            MSBF_err = 1.0 / np.sqrt(np.sum(weights_MSBF))
            
            for z, mF, err_mF, vpec in hubble_flow_sbf:
                mu_pred = distance_modulus_cosmo(z, H0_trial, Om)
                mF_pred = mu_pred + MSBF_best
                cz = c_km * z
                err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
                err_total = np.sqrt(err_mF**2 + err_vpec**2 + MSBF_err**2)
                chi2 += ((mF - mF_pred) / err_total)**2
    
    # Add chi2 from calibrator consistency
    for i, (MB_val, MB_err_i) in enumerate(zip(MB_vals, MB_errs)):
        chi2 += ((MB_val - MB_best) / MB_err_i)**2
    
    return chi2, MB_best, MB_err


def find_H0_direct(use_anchors=None, use_methods=None, 
                   use_sneia_hf=True, use_sbf_hf=True):
    """Find best-fit H0 by minimizing chi-squared."""
    
    def chi2_func(H0):
        c2, _, _ = compute_chi2_direct(H0, use_anchors, use_methods,
                                        use_sneia_hf, use_sbf_hf)
        return c2
    
    # Grid search first
    H0_grid = np.linspace(50, 120, 200)
    chi2_grid = [chi2_func(h) for h in H0_grid]
    H0_init = H0_grid[np.argmin(chi2_grid)]
    
    # Refine
    result = minimize_scalar(chi2_func, bounds=(50, 120), method='bounded')
    H0_best = result.x
    chi2_min = result.fun
    
    # Error from delta chi2 = 1
    def find_bound(direction):
        target = chi2_min + 1.0
        if direction > 0:
            h_range = np.linspace(H0_best, H0_best + 20, 1000)
        else:
            h_range = np.linspace(H0_best - 20, H0_best, 1000)[::-1]
        for h in h_range:
            if chi2_func(h) > target:
                return abs(h - H0_best)
        return 10.0
    
    err_plus = find_bound(1)
    err_minus = find_bound(-1)
    H0_err = (err_plus + err_minus) / 2
    
    _, MB_best, MB_err = compute_chi2_direct(H0_best, use_anchors, use_methods,
                                              use_sneia_hf, use_sbf_hf)
    
    return H0_best, H0_err, chi2_min, MB_best, MB_err


# ============================================================
# METHOD 2: Full GLS with proper design matrix
# ============================================================

def build_and_solve_gls(use_anchors=None, use_methods=None,
                        use_sneia_hf=True, use_sbf_hf=True):
    """
    Build and solve the full GLS system.
    
    The key equation from SH0ES: Y = Lq + noise
    where q includes host distance moduli, M_B, and 5*log10(H0).
    
    For the Hubble flow, we use the low-z approximation with
    second-order correction: mu(z) ≈ 5*log10(cz) + 25 - 5*log10(H0) + correction(z)
    where correction accounts for deceleration parameter q0.
    """
    if use_anchors is None:
        use_anchors = ['N4258', 'LMC']
    if use_methods is None:
        use_methods = ['Cepheid', 'TRGB']
    
    # Identify hosts with both primary measurements and SN Ia calibration
    sneia_hosts = {h for h, _, _ in sneia_calibrators}
    measured_hosts = set()
    for host, method, anchor, _, _ in host_measurements:
        if anchor in use_anchors and method in use_methods:
            measured_hosts.add(host)
    calibrated_hosts = sorted(sneia_hosts & measured_hosts)
    
    # Parameters: [mu_host_1, ..., mu_host_n, M_B, M_SBF, 5logH0]
    param_names = [f'mu_{h}' for h in calibrated_hosts]
    
    # Add SBF group distances if SBF HF is used
    sbf_groups = []
    if use_sbf_hf:
        sbf_groups = sorted(set(host_group[h] for h in host_group))
        param_names.extend([f'mu_{g}' for g in sbf_groups])
    
    param_names.extend(['M_B', 'M_SBF', '5logH0'])
    n_params = len(param_names)
    param_idx = {name: i for i, name in enumerate(param_names)}
    
    y_list, L_list, sigma_list, labels = [], [], [], []
    
    # (A) Primary indicator measurements
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
        labels.append(f'Primary: {host}/{method}/{anchor}')
    
    # (B) SN Ia calibrators
    for host, mB, err_mB in sneia_calibrators:
        if host not in calibrated_hosts:
            continue
        
        row = np.zeros(n_params)
        row[param_idx[f'mu_{host}']] = 1.0
        row[param_idx['M_B']] = 1.0
        
        y_list.append(mB)
        L_list.append(row)
        sigma_list.append(err_mB)
        labels.append(f'SN cal: {host}')
    
    # (C) Group links and SBF calibrators
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
                labels.append(f'Group: {host}->{grp}')
        
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
            labels.append(f'SBF cal: {host}')
    
    # (D) Hubble flow SN Ia
    # Use cosmological correction for each redshift
    # mu(z, H0) = 5*log10(cz) - 5*log10(H0) + 25 + delta(z)
    # where delta(z) accounts for higher-order terms
    # For flat LCDM with Om=0.3:
    # delta(z) = 5*log10(1 + z*(1 - q0)/2 + ...) 
    # q0 = Om/2 - OL = 0.15 - 0.7 = -0.55
    # delta(z) ≈ 5*log10(1 + z*(1+q0)/2) for small z
    # But this is small for z < 0.1
    
    # Actually, let me compute the exact correction for each z
    # mu_exact(z, H0) - [5*log10(cz) - 5*log10(H0) + 25]
    H0_ref = 70.0  # reference for computing corrections
    Om = 0.3
    
    if use_sneia_hf:
        for z, mB, err_mB, vpec in hubble_flow_sneia:
            cz = c_km * z
            
            # Exact distance modulus minus linear approximation
            mu_exact = distance_modulus_cosmo(z, H0_ref, Om)
            mu_linear = 5*np.log10(cz) - 5*np.log10(H0_ref) + 25
            delta_z = mu_exact - mu_linear
            
            # Observation equation:
            # mB = mu(z, H0) + M_B
            # mB = [5*log10(cz) - 5*log10(H0) + 25 + delta(z)] + M_B
            # mB - 5*log10(cz) - 25 - delta(z) = M_B - 5*log10(H0)
            
            y_val = mB - 5*np.log10(cz) - 25 - delta_z
            
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mB**2 + err_vpec**2)
            
            row = np.zeros(n_params)
            row[param_idx['M_B']] = 1.0
            row[param_idx['5logH0']] = -1.0
            
            y_list.append(y_val)
            L_list.append(row)
            sigma_list.append(err_total)
            labels.append(f'HF SN: z={z:.3f}')
    
    # (E) Hubble flow SBF
    if use_sbf_hf:
        for z, mF, err_mF, vpec in hubble_flow_sbf:
            cz = c_km * z
            
            mu_exact = distance_modulus_cosmo(z, H0_ref, Om)
            mu_linear = 5*np.log10(cz) - 5*np.log10(H0_ref) + 25
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
            labels.append(f'HF SBF: z={z:.3f}')
    
    # Convert to arrays
    y = np.array(y_list)
    L = np.array(L_list)
    sigma = np.array(sigma_list)
    n_obs = len(y)
    
    # Build covariance matrix
    C = np.diag(sigma**2)
    
    # Add off-diagonal covariance for shared anchor uncertainties
    for i in range(n_obs):
        for j in range(i+1, n_obs):
            if labels[i].startswith('Primary:') and labels[j].startswith('Primary:'):
                parts_i = labels[i].split('/')
                parts_j = labels[j].split('/')
                anchor_i = parts_i[-1]
                anchor_j = parts_j[-1]
                method_i = parts_i[1]
                method_j = parts_j[1]
                
                if anchor_i == anchor_j:
                    cov = anchors[anchor_i]['err']**2
                    if method_i == method_j:
                        err_m = method_anchor_err.get((method_i, anchor_i), 0.0)
                        cov += err_m**2
                    C[i, j] = cov
                    C[j, i] = cov
    
    # Solve GLS
    C_inv = np.linalg.inv(C)
    LtCiL = L.T @ C_inv @ L
    LtCiy = L.T @ C_inv @ y
    cq = np.linalg.inv(LtCiL)
    q_best = cq @ LtCiy
    
    residuals = y - L @ q_best
    chi2 = residuals.T @ C_inv @ residuals
    dof = n_obs - n_params
    
    # Extract H0
    idx_5logH0 = param_idx['5logH0']
    val_5logH0 = q_best[idx_5logH0]
    err_5logH0 = np.sqrt(cq[idx_5logH0, idx_5logH0])
    H0 = 10**(val_5logH0 / 5.0)
    H0_err = H0 * np.log(10) / 5.0 * err_5logH0
    
    return {
        'H0': H0, 'H0_err': H0_err,
        '5logH0': val_5logH0, '5logH0_err': err_5logH0,
        'chi2': chi2, 'dof': dof,
        'q_best': q_best, 'cq': cq,
        'param_names': param_names, 'param_idx': param_idx,
        'y': y, 'L': L, 'C': C, 'labels': labels
    }


# ============================================================
# RUN ANALYSES
# ============================================================

print("=" * 70)
print("BASELINE ANALYSIS (GLS)")
print("=" * 70)

result_baseline = build_and_solve_gls(
    use_anchors=['N4258', 'LMC'],
    use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=True, use_sbf_hf=True
)

print(f"\nH0 = {result_baseline['H0']:.2f} ± {result_baseline['H0_err']:.2f} km/s/Mpc")
print(f"5*log10(H0) = {result_baseline['5logH0']:.4f} ± {result_baseline['5logH0_err']:.4f}")
print(f"chi2 = {result_baseline['chi2']:.2f}, dof = {result_baseline['dof']}")
print(f"chi2/dof = {result_baseline['chi2']/result_baseline['dof']:.2f}")

print("\nAll parameters:")
for i, name in enumerate(result_baseline['param_names']):
    print(f"  {name:15s} = {result_baseline['q_best'][i]:8.4f} ± {np.sqrt(result_baseline['cq'][i,i]):.4f}")

# Direct minimization
print("\n" + "=" * 70)
print("DIRECT CHI-SQUARED MINIMIZATION")
print("=" * 70)

H0_dir, H0_err_dir, chi2_dir, MB_dir, MB_err_dir = find_H0_direct(
    use_anchors=['N4258', 'LMC'],
    use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=True, use_sbf_hf=True
)
print(f"H0 = {H0_dir:.2f} ± {H0_err_dir:.2f} km/s/Mpc")
print(f"M_B = {MB_dir:.3f} ± {MB_err_dir:.3f}")
print(f"chi2_min = {chi2_dir:.2f}")

# Variants
print("\n" + "=" * 70)
print("ANALYSIS VARIANTS")
print("=" * 70)

variant_configs = {
    'Baseline (all)': dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
                           use_sneia_hf=True, use_sbf_hf=True),
    'N4258 only': dict(use_anchors=['N4258'], use_methods=['Cepheid', 'TRGB'],
                       use_sneia_hf=True, use_sbf_hf=True),
    'LMC only': dict(use_anchors=['LMC'], use_methods=['Cepheid'],
                     use_sneia_hf=True, use_sbf_hf=False),
    'Cepheids only': dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid'],
                          use_sneia_hf=True, use_sbf_hf=True),
    'TRGB only': dict(use_anchors=['N4258'], use_methods=['TRGB'],
                      use_sneia_hf=True, use_sbf_hf=True),
    'SN Ia HF only': dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
                           use_sneia_hf=True, use_sbf_hf=False),
    'SBF HF only': dict(use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
                         use_sneia_hf=False, use_sbf_hf=True),
    'N4258+Cepheids': dict(use_anchors=['N4258'], use_methods=['Cepheid'],
                           use_sneia_hf=True, use_sbf_hf=True),
    'N4258+TRGB': dict(use_anchors=['N4258'], use_methods=['TRGB'],
                       use_sneia_hf=True, use_sbf_hf=True),
}

variant_results = {}
print(f"\n{'Variant':<25s} {'H0':>8s} {'±err':>7s} {'M_B':>8s} {'chi2':>8s} {'dof':>4s} {'chi2/dof':>8s}")
print("-" * 70)

for name, config in variant_configs.items():
    try:
        res = build_and_solve_gls(**config)
        MB_val = res['q_best'][res['param_idx']['M_B']]
        variant_results[name] = res
        print(f"{name:<25s} {res['H0']:8.2f} {res['H0_err']:7.2f} {MB_val:8.3f} "
              f"{res['chi2']:8.2f} {res['dof']:4d} {res['chi2']/max(1,res['dof']):8.2f}")
    except Exception as e:
        print(f"{name:<25s} FAILED: {e}")

# Also run direct minimization variants
print("\n\nDirect minimization variants:")
print(f"{'Variant':<25s} {'H0':>8s} {'±err':>7s} {'M_B':>8s}")
print("-" * 50)

for name, config in variant_configs.items():
    try:
        H0_v, H0_err_v, chi2_v, MB_v, MB_err_v = find_H0_direct(**config)
        print(f"{name:<25s} {H0_v:8.2f} {H0_err_v:7.2f} {MB_v:8.3f}")
    except Exception as e:
        print(f"{name:<25s} FAILED: {e}")

# CMB comparison
H0_baseline = result_baseline['H0']
H0_err_baseline = result_baseline['H0_err']
H0_planck = 67.4
H0_planck_err = 0.5
tension = (H0_baseline - H0_planck) / np.sqrt(H0_err_baseline**2 + H0_planck_err**2)

print(f"\n\n{'='*70}")
print(f"CMB COMPARISON")
print(f"{'='*70}")
print(f"This work: H0 = {H0_baseline:.2f} ± {H0_err_baseline:.2f} km/s/Mpc")
print(f"Planck CMB: H0 = {H0_planck:.1f} ± {H0_planck_err:.1f} km/s/Mpc")
print(f"Tension: {tension:.1f}σ")

# Save results
os.makedirs('outputs', exist_ok=True)
results_dict = {
    'baseline': {
        'H0': round(float(H0_baseline), 2),
        'H0_err': round(float(H0_err_baseline), 2),
        'M_B': round(float(result_baseline['q_best'][result_baseline['param_idx']['M_B']]), 4),
        'chi2': round(float(result_baseline['chi2']), 2),
        'dof': int(result_baseline['dof']),
    },
    'variants': {},
    'CMB_tension_sigma': round(float(tension), 1)
}

for name, res in variant_results.items():
    results_dict['variants'][name] = {
        'H0': round(float(res['H0']), 2),
        'H0_err': round(float(res['H0_err']), 2),
        'chi2': round(float(res['chi2']), 2),
        'dof': int(res['dof'])
    }

with open('outputs/h0_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nResults saved to outputs/h0_results.json")
