#!/usr/bin/env python3
"""
Local Distance Network Analysis: Consensus H0 Measurement
Using Generalized Least Squares (GLS) to combine multiple distance indicators.

Framework:
- Geometric anchors: N4258, LMC, MW
- Primary distance indicators: Cepheids, TRGB
- Secondary calibrators: SNe Ia, SBF
- Hubble flow: SNe Ia, SBF
"""

import numpy as np
from scipy.linalg import block_diag, sqrtm
from scipy.optimize import minimize
import json
import os

# ============================================================
# Load Data
# ============================================================

# Geometric anchors (distance moduli)
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC':   {'mu': 18.477, 'err': 0.024},
    'MW':    {'mu': 0.0,    'err': 0.0}
}

# Host distance measurements: (host, method, anchor, mu_meas, err_meas)
host_measurements_raw = [
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

# SNe Ia calibrators: (host, mB, err_mB)
sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101',    9.85,  0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

# SBF calibrators: (host, mF110W, err_mF110W)
sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12)
]

# Hubble flow SNe Ia: (z, mB, err_mB, peculiar velocity sigma in km/s)
hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]

# Hubble flow SBF: (z, mF110W, err_mF110W, peculiar velocity sigma)
hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250)
]

# Method-anchor calibration errors
method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'):   0.03,
    ('Cepheid', 'MW'):    0.02,
    ('TRGB', 'N4258'):    0.05
}

# Host group mapping
host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo'
}
depth_scatter = 0.10  # mag

c_km = 299792.458  # km/s

# ============================================================
# Approach 1: Standard distance-ladder step-by-step
# ============================================================

def compute_h0_stepwise():
    """Stepwise distance ladder: anchor → hosts → SNe Ia → H0."""
    results = {}
    
    # --- Step 1: Host distances from each method-anchor pair ---
    host_dists = {}
    for host, method, anchor, mu_meas, err_meas in host_measurements_raw:
        anchor_mu = anchors[anchor]['mu']
        anchor_err = anchors[anchor]['err']
        cal_err = method_anchor_err.get((method, anchor), 0.0)
        total_err = np.sqrt(err_meas**2 + anchor_err**2 + cal_err**2)
        
        key = (host, method, anchor)
        host_dists[key] = {
            'mu': mu_meas,
            'err': total_err,
            'mu_anchor': anchor_mu
        }
    
    # --- Step 2: Calibrate SNe Ia absolute magnitude ---
    # For each SN Ia host, M_B = m_B - mu_host
    sneia_MB = []
    for host, mB, err_mB in sneia_calibrators:
        # Collect all host distance measurements
        host_mus = []
        for (h, method, anchor), v in host_dists.items():
            if h == host:
                host_mus.append((v['mu'], v['err']))
        
        if len(host_mus) > 0:
            # Weighted average of host mu
            weights = [1.0/e**2 for _, e in host_mus]
            w_sum = sum(weights)
            mu_avg = sum(mu * w for (mu, _), w in zip(host_mus, weights)) / w_sum
            mu_err = np.sqrt(1.0 / w_sum)
        else:
            mu_avg, mu_err = None, None
            continue
        
        M_B = mB - mu_avg
        err_MB = np.sqrt(err_mB**2 + mu_err**2)
        sneia_MB.append({
            'host': host,
            'M_B': M_B,
            'err_MB': err_MB,
            'mu_host': mu_avg,
            'err_mu_host': mu_err
        })
    
    # Weighted mean M_B
    weights_MB = [1.0/s['err_MB']**2 for s in sneia_MB]
    w_sum_MB = sum(weights_MB)
    M_B_mean = sum(s['M_B'] * w for s, w in zip(sneia_MB, weights_MB)) / w_sum_MB
    M_B_err = np.sqrt(1.0 / w_sum_MB)
    
    results['M_B'] = M_B_mean
    results['M_B_err'] = M_B_err
    results['sneia_calibrators'] = sneia_MB
    
    # --- Step 3: H0 from Hubble flow SNe Ia ---
    h0_estimates = []
    for z, mB, err_mB, v_pec in hubble_flow_sneia:
        mu = mB - M_B_mean
        # Peculiar velocity error in magnitudes
        v_pec_frac = v_pec / (c_km * z)
        err_mu_pec = (5.0 / np.log(10)) * v_pec_frac  # d(mu)/d(ln v) = 5/ln(10)
        err_mu = np.sqrt(err_mB**2 + M_B_err**2 + err_mu_pec**2)
        
        # H0 in km/s/Mpc
        D_L = 10**((mu - 25.0) / 5.0)
        v_cmb = c_km * z
        H0 = v_cmb / D_L
        err_H0 = H0 * (err_mu * np.log(10) / 5.0)
        
        h0_estimates.append({
            'z': z,
            'mB': mB,
            'mu': mu,
            'err_mu': err_mu,
            'H0': H0,
            'err_H0': err_H0
        })
    
    # Weighted mean H0
    weights_H0 = [1.0/h['err_H0']**2 for h in h0_estimates]
    w_sum_H0 = sum(weights_H0)
    H0_mean = sum(h['H0'] * w for h, w in zip(h0_estimates, weights_H0)) / w_sum_H0
    H0_err = np.sqrt(1.0 / w_sum_H0)
    
    results['H0'] = H0_mean
    results['H0_err'] = H0_err
    results['h0_estimates'] = h0_estimates
    results['method'] = 'stepwise'
    
    return results


# ============================================================
# Approach 2: Generalized Least Squares — Distance Network
# ============================================================

def run_gls_distance_network():
    """
    Full GLS approach: simultaneously fit host distances, SNe Ia M_B,
    SBF calibration, and H0.

    State vector:
    - mu for each unique host
    - M_B (SNe Ia absolute magnitude)
    - M_SBF (SBF absolute magnitude)
    - H0

    Observations:
    - Host mu measurements (from indicators + anchors)
    - SNe Ia mB (→ M_B = mB - mu_host)
    - SBF mF110W (→ M_SBF = mF110W - mu_host)
    - Hubble flow SNe: mu(z, H0) = 5 log10(cz/H0) + 25
    - Hubble flow SBF: mu(z, H0) same
    """
    c_km = 299792.458

    # --- Identify unique hosts ---
    unique_hosts = set()
    for host, _, _, _, _ in host_measurements_raw:
        unique_hosts.add(host)
    for host, _, _ in sneia_calibrators:
        unique_hosts.add(host)
    for host, _, _ in sbf_calibrators:
        unique_hosts.add(host)
    unique_hosts = sorted(unique_hosts)

    n_hosts = len(unique_hosts)
    host_idx = {h: i for i, h in enumerate(unique_hosts)}

    # State vector: [mu_hosts (n_hosts), M_B, M_SBF, H0]
    n_state = n_hosts + 3
    idx_MB = n_hosts
    idx_MSBF = n_hosts + 1
    idx_H0 = n_hosts + 2

    # --- Build observation vector and design matrix ---
    obs = []
    sigma = []
    design_rows = []

    # 1. Anchor constraints (as prior measurements of specific hosts)
    # We don't directly observe anchor hosts, but anchor uncertainties propagate
    # through indicator measurements. However, for the pure GLS, we can add
    # "anchor mu" as a prior if we had anchor host distances.
    # Instead, the primary indicators already fold in the anchor.

    # 2. Host distance measurements from indicators
    for host, method, anchor, mu_meas, err_meas in host_measurements_raw:
        row = np.zeros(n_state)
        row[host_idx[host]] = 1.0
        design_rows.append(row)
        obs.append(mu_meas)
        # Include anchor uncertainty and calibration error
        anchor_err = anchors[anchor]['err']
        cal_err = method_anchor_err.get((method, anchor), 0.0)
        total_err = np.sqrt(err_meas**2 + anchor_err**2 + cal_err**2)
        sigma.append(total_err)

    # 3. SNe Ia calibrator constraints
    for host, mB, err_mB in sneia_calibrators:
        row = np.zeros(n_state)
        row[host_idx[host]] = 1.0
        row[idx_MB] = 1.0
        design_rows.append(row)
        obs.append(mB)
        sigma.append(err_mB)

    # 4. SBF calibrator constraints
    for host, mF110W, err_mF110W in sbf_calibrators:
        row = np.zeros(n_state)
        row[host_idx[host]] = 1.0
        row[idx_MSBF] = 1.0
        design_rows.append(row)
        obs.append(mF110W)
        # Add intra-group depth scatter for cluster galaxies
        group = host_group.get(host, None)
        extra_err = depth_scatter if group else 0.0
        total_err = np.sqrt(err_mF110W**2 + extra_err**2)
        sigma.append(total_err)

    # 5. Hubble flow SNe Ia
    for z, mB, err_mB, v_pec in hubble_flow_sneia:
        # mu_pred = 5*log10(cz/H0) + 25
        # mB_pred = M_B + mu_pred
        # Linearize around initial H0 guess? No — let's do full non-linear GLS via optimization.
        # For now, we'll use the GLS with a non-linear measurement function.
        pass  # Handled in non-linear approach

    # For a fully linear approach, we'd need to linearize H0-dependent terms.
    # Better: use non-linear least squares.

    # --- Non-linear GLS via optimization ---
    
    def compute_residuals(params):
        """Compute all residuals given parameter vector."""
        mu_hosts = params[:n_hosts]
        M_B = params[idx_MB]
        M_SBF = params[idx_MSBF]
        H0 = params[idx_H0]
        
        residuals = []
        weights_list = []
        
        # Host mu measurements
        for host, method, anchor, mu_meas, err_meas in host_measurements_raw:
            pred = mu_hosts[host_idx[host]]
            anchor_err_val = anchors[anchor]['err']
            cal_err = method_anchor_err.get((method, anchor), 0.0)
            total_err = np.sqrt(err_meas**2 + anchor_err**2 + cal_err**2)
            residuals.append((mu_meas - pred) / total_err)
        
        # SNe Ia calibrators
        for host, mB, err_mB in sneia_calibrators:
            pred = mu_hosts[host_idx[host]] + M_B
            residuals.append((mB - pred) / err_mB)
        
        # SBF calibrators
        for host, mF110W, err_mF110W in sbf_calibrators:
            pred = mu_hosts[host_idx[host]] + M_SBF
            group = host_group.get(host, None)
            extra_err = depth_scatter if group else 0.0
            total_err = np.sqrt(err_mF110W**2 + extra_err**2)
            residuals.append((mF110W - pred) / total_err)
        
        # Hubble flow SNe Ia
        for z, mB, err_mB, v_pec in hubble_flow_sneia:
            mu_pred = 5.0 * np.log10(c_km * z / H0) + 25.0
            pred = M_B + mu_pred
            v_pec_frac = v_pec / (c_km * z)
            err_mu_pec = (5.0 / np.log(10)) * v_pec_frac
            total_err = np.sqrt(err_mB**2 + err_mu_pec**2)
            residuals.append((mB - pred) / total_err)
        
        # Hubble flow SBF
        for z, mF110W, err_mF110W, v_pec in hubble_flow_sbf:
            mu_pred = 5.0 * np.log10(c_km * z / H0) + 25.0
            pred = M_SBF + mu_pred
            v_pec_frac = v_pec / (c_km * z)
            err_mu_pec = (5.0 / np.log(10)) * v_pec_frac
            total_err = np.sqrt(err_mF110W**2 + err_mu_pec**2)
            residuals.append((mF110W - pred) / total_err)
        
        return np.array(residuals)
    
    def chi2(params):
        r = compute_residuals(params)
        return np.sum(r**2)
    
    # --- Initial guess ---
    # Start with stepwise results
    step_results = compute_h0_stepwise()
    
    mu_init = np.zeros(n_hosts)
    for host in unique_hosts:
        host_mus = []
        for h, method, anchor, mu_meas, err_meas in host_measurements_raw:
            if h == host:
                host_mus.append(mu_meas)
        if host_mus:
            mu_init[host_idx[host]] = np.mean(host_mus)
        else:
            # For SBF hosts with no primary indicator
            mu_init[host_idx[host]] = 31.0  # reasonable guess
    
    params_init = np.zeros(n_state)
    params_init[:n_hosts] = mu_init
    params_init[idx_MB] = step_results['M_B']
    params_init[idx_MSBF] = -19.5  # approximate SBF absolute magnitude
    params_init[idx_H0] = step_results['H0']
    
    # --- Optimize ---
    result = minimize(chi2, params_init, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    params_best = result.x
    mu_hosts_best = params_best[:n_hosts]
    M_B_best = params_best[idx_MB]
    M_SBF_best = params_best[idx_MSBF]
    H0_best = params_best[idx_H0]
    
    # --- Compute covariance via Fisher matrix (Jacobian) ---
    def get_jacobian(params, eps=1e-6):
        """Numerical Jacobian of normalized residuals."""
        r0 = compute_residuals(params)
        J = np.zeros((len(r0), len(params)))
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            r_plus = compute_residuals(params_plus)
            J[:, i] = (r_plus - r0) / eps
        return J
    
    J = get_jacobian(params_best)
    # Fisher matrix: J^T J
    # Covariance: (J^T J)^{-1}
    try:
        fisher = J.T @ J
        cov = np.linalg.inv(fisher)
        param_errs = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        # Fallback: use Hessian from optimization if available
        param_errs = np.full(n_state, np.nan)
        cov = np.full((n_state, n_state), np.nan)
    
    H0_err = param_errs[idx_H0]
    M_B_err = param_errs[idx_MB]
    M_SBF_err = param_errs[idx_MSBF]
    
    # Chi2/dof
    n_obs = len(compute_residuals(params_best))
    n_params = n_state
    chi2_val = chi2(params_best)
    dof = n_obs - n_params
    chi2_red = chi2_val / dof if dof > 0 else np.nan
    
    # --- Package results ---
    results = {
        'method': 'GLS',
        'H0': H0_best,
        'H0_err': H0_err,
        'M_B': M_B_best,
        'M_B_err': M_B_err,
        'M_SBF': M_SBF_best,
        'M_SBF_err': M_SBF_err,
        'chi2': chi2_val,
        'dof': dof,
        'chi2_red': chi2_red,
        'n_obs': n_obs,
        'n_params': n_params,
        'hosts': {}
    }
    
    for i, host in enumerate(unique_hosts):
        results['hosts'][host] = {
            'mu': mu_hosts_best[i],
            'err': param_errs[i]
        }
    
    # Store individual H0 contributions
    h0_contrib = []
    for z, mB, err_mB, v_pec in hubble_flow_sneia:
        mu_pred = 5.0 * np.log10(c_km * z / H0_best) + 25.0
        v_pec_frac = v_pec / (c_km * z)
        err_mu_pec = (5.0 / np.log(10)) * v_pec_frac
        D = 10**((mu_pred - 25.0) / 5.0)
        h0_i = c_km * z / D
        h0_contrib.append({
            'z': z, 'mB': mB, 'mu': mu_pred,
            'H0_i': h0_i
        })
    results['h0_contrib_sneia'] = h0_contrib
    
    # Residuals
    residuals = compute_residuals(params_best)
    results['residuals'] = residuals.tolist()
    
    return results


# ============================================================
# Approach 3: Covariance-weighted combination (variant analysis)
# ============================================================

def run_variant_analysis():
    """Run analysis variants and compute H0 with different combinations."""
    variants = {}
    
    # Baseline: full GLS
    variants['baseline'] = run_gls_distance_network()
    
    # Variant 1: SNe Ia only (no SBF)
    # We'll just compute using stepwise method, only SNe Ia
    r_snia_only = compute_h0_stepwise()
    variants['SNeIa_only'] = {
        'H0': r_snia_only['H0'],
        'H0_err': r_snia_only['H0_err'],
        'M_B': r_snia_only['M_B'],
        'M_B_err': r_snia_only['M_B_err']
    }
    
    # Variant 2: Cepheids only
    variants['Cepheids_only'] = compute_indicator_only('Cepheid')
    
    # Variant 3: TRGB only
    variants['TRGB_only'] = compute_indicator_only('TRGB')
    
    # Variant 4: N4258 anchor only
    variants['N4258_only'] = compute_anchor_only('N4258')
    
    # Variant 5: LMC anchor only
    variants['LMC_only'] = compute_anchor_only('LMC')
    
    return variants


def compute_indicator_only(indicator):
    """Compute H0 using only a specific indicator type."""
    c_km = 299792.458
    
    # Filter to this indicator
    host_meas = [(h, m, a, mu, e) for h, m, a, mu, e in host_measurements_raw if m == indicator]
    
    if not host_meas:
        return {'H0': np.nan, 'H0_err': np.nan}
    
    # Average host distances
    host_mu = {}
    for host, _, _, mu, e in host_meas:
        if host not in host_mu:
            host_mu[host] = []
        host_mu[host].append((mu, e))
    
    # SNe Ia calibration
    sneia_MB = []
    for host, mB, err_mB in sneia_calibrators:
        if host in host_mu:
            mus = host_mu[host]
            weights = [1.0/e**2 for _, e in mus]
            w_sum = sum(weights)
            mu_avg = sum(mu_val * wt for (mu_val, _), wt in zip(mus, weights)) / w_sum
            mu_err = np.sqrt(1.0 / w_sum)
            M_B = mB - mu_avg
            err_MB_val = np.sqrt(err_mB**2 + mu_err**2)
            sneia_MB.append((M_B, err_MB_val))
    
    if not sneia_MB:
        return {'H0': np.nan, 'H0_err': np.nan}
    
    weights = [1.0/e**2 for _, e in sneia_MB]
    w_sum = sum(weights)
    M_B = sum(m * w for (m, _), w in zip(sneia_MB, weights)) / w_sum
    M_B_err = np.sqrt(1.0 / w_sum)
    
    # H0
    h0s = []
    for z, mB, err_mB, v_pec in hubble_flow_sneia:
        mu = mB - M_B
        v_pec_frac = v_pec / (c_km * z)
        err_mu_pec = (5.0 / np.log(10)) * v_pec_frac
        D = 10**((mu - 25.0) / 5.0)
        H0_i = c_km * z / D
        err_H0 = H0_i * (np.sqrt(err_mB**2 + M_B_err**2 + err_mu_pec**2) * np.log(10) / 5.0)
        h0s.append((H0_i, err_H0))
    
    wh = [1.0/e**2 for _, e in h0s]
    wsh = sum(wh)
    H0 = sum(h * w for (h, _), w in zip(h0s, wh)) / wsh
    H0_err = np.sqrt(1.0 / wsh)
    
    return {'H0': H0, 'H0_err': H0_err, 'M_B': M_B, 'M_B_err': M_B_err}


def compute_anchor_only(anchor):
    """Compute H0 using only one anchor."""
    c_km = 299792.458
    
    host_meas = [(h, m, a, mu, e) for h, m, a, mu, e in host_measurements_raw if a == anchor]
    
    if not host_meas:
        return {'H0': np.nan, 'H0_err': np.nan}
    
    host_mu = {}
    for host, method, _, mu, e in host_meas:
        if host not in host_mu:
            host_mu[host] = []
        # Add anchor + calibration errors
        anchor_err = anchors[anchor]['err']
        cal_err = method_anchor_err.get((method, anchor), 0.0)
        total_err = np.sqrt(e**2 + anchor_err**2 + cal_err**2)
        host_mu[host].append((mu, total_err))
    
    sneia_MB = []
    for host, mB, err_mB in sneia_calibrators:
        if host in host_mu:
            mus = host_mu[host]
            weights = [1.0/e**2 for _, e in mus]
            w_sum = sum(weights)
            mu_avg = sum(mu_val * wt for (mu_val, _), wt in zip(mus, weights)) / w_sum
            mu_err = np.sqrt(1.0 / w_sum)
            M_B = mB - mu_avg
            err_MB_val = np.sqrt(err_mB**2 + mu_err**2)
            sneia_MB.append((M_B, err_MB_val))
    
    if not sneia_MB:
        return {'H0': np.nan, 'H0_err': np.nan}
    
    weights = [1.0/e**2 for _, e in sneia_MB]
    w_sum = sum(weights)
    M_B = sum(m * w for (m, _), w in zip(sneia_MB, weights)) / w_sum
    M_B_err = np.sqrt(1.0 / w_sum)
    
    h0s = []
    for z, mB, err_mB, v_pec in hubble_flow_sneia:
        mu = mB - M_B
        v_pec_frac = v_pec / (c_km * z)
        err_mu_pec = (5.0 / np.log(10)) * v_pec_frac
        D = 10**((mu - 25.0) / 5.0)
        H0_i = c_km * z / D
        err_H0 = H0_i * (np.sqrt(err_mB**2 + M_B_err**2 + err_mu_pec**2) * np.log(10) / 5.0)
        h0s.append((H0_i, err_H0))
    
    wh = [1.0/e**2 for _, e in h0s]
    wsh = sum(wh)
    H0 = sum(h * w for (h, _), w in zip(h0s, wh)) / wsh
    H0_err = np.sqrt(1.0 / wsh)
    
    return {'H0': H0, 'H0_err': H0_err, 'M_B': M_B, 'M_B_err': M_B_err}


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 60)
    print("Local Distance Network — H0 Measurement")
    print("=" * 60)
    
    # Stepwise
    sw = compute_h0_stepwise()
    print(f"\nStepwise method:")
    print(f"  M_B = {sw['M_B']:.4f} ± {sw['M_B_err']:.4f} mag")
    print(f"  H0  = {sw['H0']:.2f} ± {sw['H0_err']:.2f} km/s/Mpc")
    
    # GLS
    gls = run_gls_distance_network()
    print(f"\nGLS Distance Network:")
    print(f"  H0     = {gls['H0']:.2f} ± {gls['H0_err']:.2f} km/s/Mpc")
    print(f"  M_B    = {gls['M_B']:.4f} ± {gls['M_B_err']:.4f} mag")
    print(f"  M_SBF  = {gls['M_SBF']:.4f} ± {gls['M_SBF_err']:.4f} mag")
    print(f"  χ²/dof = {gls['chi2']:.2f}/{gls['dof']} = {gls['chi2_red']:.3f}")
    
    print(f"\nHost distances from GLS:")
    for host, info in gls['hosts'].items():
        print(f"  {host}: μ = {info['mu']:.4f} ± {info['err']:.4f} mag")
    
    # Variants
    variants = run_variant_analysis()
    print(f"\nAnalysis Variants:")
    for name, v in variants.items():
        print(f"  {name:20s}: H0 = {v['H0']:.2f} ± {v['H0_err']:.2f} km/s/Mpc")
    
    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return obj
    
    # Save GLS results
    with open('outputs/gls_results.json', 'w') as f:
        json.dump(make_serializable(gls), f, indent=2)
    
    with open('outputs/stepwise_results.json', 'w') as f:
        json.dump(make_serializable(sw), f, indent=2)
    
    with open('outputs/variants.json', 'w') as f:
        json.dump(make_serializable(variants), f, indent=2)
    
    print("\nResults saved to outputs/")
