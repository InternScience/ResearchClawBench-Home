#!/usr/bin/env python3
"""
H0 Distance Network: Generalized Least Squares Framework
=========================================================
Reproduces the Hubble constant measurement using a covariance-weighted
approach combining geometric anchors, primary distance indicators,
secondary calibrations, and Hubble flow observations.
"""

import numpy as np
import json
import os

# ============================================================
# 1. DATA INPUT (from H0DN_MinimalDataset.txt)
# ============================================================

# Anchors (geometric distance moduli)
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC': {'mu': 18.477, 'err': 0.024},
    'MW': {'mu': 0.0, 'err': 0.0}
}

# Host distance measurements from primary indicators
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

# SNe Ia calibrators
sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101', 9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

# SBF calibrators
sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12)
]

# Hubble flow SNe Ia
hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]

# Hubble flow SBF galaxies
hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250)
]

# Additional calibration uncertainties
method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'): 0.03,
    ('Cepheid', 'MW'): 0.02,
    ('TRGB', 'N4258'): 0.05
}

# Host to group mapping
host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo'
}

depth_scatter = 0.10
c_km = 299792.458

# ============================================================
# 2. GLS FRAMEWORK
# ============================================================

def build_gls_system(use_anchors=None, use_methods=None, use_sneia_hf=True, use_sbf_hf=True):
    """
    Build the GLS system Y = Lq with covariance C.
    
    Parameters (free):
      - mu_host for each unique SN Ia / SBF host
      - M_B (fiducial SN Ia absolute magnitude)
      - M_SBF (fiducial SBF absolute magnitude)  
      - 5*log10(H0) (the Hubble constant parameter)
    
    Observation equations:
      1. Primary indicator constraints: mu_meas = mu_host - mu_anchor + err_method_anchor
         -> mu_host = mu_meas + mu_anchor  (with combined uncertainty)
         -> y = mu_meas + mu_anchor, equation: mu_host = y
      2. SN Ia calibrator: mB = mu_host + M_B
         -> y = mB, equation: mu_host + M_B = mB
      3. SBF calibrator: mF110W = mu_host_group + M_SBF  (need mu for SBF host groups)
      4. Hubble flow SN Ia: mB = 5*log10(cz) + 25 - 5*log10(H0) + M_B
         -> y = mB - 5*log10(cz) - 25, equation: -5*log10(H0) + M_B = y
      5. Hubble flow SBF: mF110W = 5*log10(cz) + 25 - 5*log10(H0) + M_SBF
    """
    
    if use_anchors is None:
        use_anchors = ['N4258', 'LMC']
    if use_methods is None:
        use_methods = ['Cepheid', 'TRGB']
    
    # Identify unique hosts that have SN Ia or SBF calibration
    sneia_hosts = set(h for h, _, _ in sneia_calibrators)
    sbf_hosts = set(h for h, _, _ in sbf_calibrators)
    
    # Identify hosts that have primary indicator measurements with allowed anchors/methods
    measured_hosts = set()
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if anchor in use_anchors and method in use_methods:
            measured_hosts.add(host)
    
    # Only include hosts that have BOTH primary measurements AND secondary calibration
    calibrated_sneia_hosts = sorted(sneia_hosts & measured_hosts)
    
    # For SBF, we need to link SBF hosts to hosts with primary measurements
    # SBF hosts are in clusters; we need cluster distances from primary indicators
    # For simplicity, SBF hosts get their own distance parameters from the group
    # We need at least one host in the group with a primary measurement
    # In this dataset, SBF hosts don't have direct primary measurements
    # We'll handle SBF through a separate path: SBF calibrators provide M_SBF
    # using distances from nearby hosts in the same group
    
    # Build parameter vector q:
    # [mu_host_1, mu_host_2, ..., M_B, M_SBF, 5*log10(H0)]
    
    param_names = []
    param_idx = {}
    
    # Host distance moduli (for SN Ia calibrated hosts)
    for host in calibrated_sneia_hosts:
        param_names.append(f'mu_{host}')
        param_idx[f'mu_{host}'] = len(param_names) - 1
    
    # SBF group distance moduli
    sbf_groups = sorted(set(host_group[h] for h in sbf_hosts))
    for grp in sbf_groups:
        param_names.append(f'mu_{grp}')
        param_idx[f'mu_{grp}'] = len(param_names) - 1
    
    # Absolute magnitudes
    param_names.append('M_B')
    idx_MB = len(param_names) - 1
    param_idx['M_B'] = idx_MB
    
    param_names.append('M_SBF')
    idx_MSBF = len(param_names) - 1
    param_idx['M_SBF'] = idx_MSBF
    
    # H0 parameter: we'll solve for 5*log10(H0)
    param_names.append('5logH0')
    idx_5logH0 = len(param_names) - 1
    param_idx['5logH0'] = idx_5logH0
    
    n_params = len(param_names)
    
    # Build observation vectors
    y_list = []
    L_list = []
    sigma_list = []
    eq_labels = []
    
    # --- (A) Primary indicator constraints ---
    # For each host measurement: mu_meas = mu_host - mu_anchor
    # => mu_host = mu_meas + mu_anchor
    # y = mu_meas + mu_anchor, L row: [... 1 at mu_host ...] = y
    
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if anchor not in use_anchors or method not in use_methods:
            continue
        if host not in calibrated_sneia_hosts:
            continue
        
        key = f'mu_{host}'
        if key not in param_idx:
            continue
        
        # y value: measured distance modulus relative to anchor + anchor distance
        y_val = mu_meas + anchors[anchor]['mu']  # This gives absolute mu_host
        
        # Wait - the measurement is: mu_meas is the distance modulus of the host
        # measured relative to the anchor. Actually, looking at the data more carefully:
        # The values like 32.50 for NGC1309 are absolute distance moduli already
        # measured using Cepheids calibrated through N4258.
        # The anchor distance is already folded in.
        # So: mu_host = mu_meas, with uncertainty from measurement + anchor + method
        
        # Actually let me reconsider. The data says:
        # host_measurements = (host, method, anchor, mu_meas, err_meas)
        # mu_meas is the measured distance modulus of the host
        # The anchor provides the zero-point calibration
        # So the observation equation is:
        # mu_meas = mu_host + (mu_anchor_true - mu_anchor_adopted) + method_offset
        # If we assume the anchor distance is known, then mu_meas = mu_host
        # But the anchor uncertainty propagates as additional error
        
        y_val = mu_meas
        
        # Combined uncertainty
        err_anchor = anchors[anchor]['err']
        err_method = method_anchor_err.get((method, anchor), 0.0)
        err_total = np.sqrt(err_meas**2 + err_anchor**2 + err_method**2)
        
        row = np.zeros(n_params)
        row[param_idx[key]] = 1.0
        
        y_list.append(y_val)
        L_list.append(row)
        sigma_list.append(err_total)
        eq_labels.append(f'Primary: {host} via {method}/{anchor}')
    
    # --- (B) SN Ia calibrator constraints ---
    # mB = mu_host + M_B
    # => mu_host + M_B = mB
    # y = mB, L row: [... 1 at mu_host ..., 1 at M_B ...]
    
    for host, mB, err_mB in sneia_calibrators:
        if host not in calibrated_sneia_hosts:
            continue
        
        key = f'mu_{host}'
        row = np.zeros(n_params)
        row[param_idx[key]] = 1.0
        row[idx_MB] = 1.0
        
        y_list.append(mB)
        L_list.append(row)
        sigma_list.append(err_mB)
        eq_labels.append(f'SN Ia cal: {host}')
    
    # --- (C) SBF calibrator constraints ---
    # Need to connect SBF hosts to the distance network
    # SBF hosts are in groups (Fornax, Virgo)
    # We need group distances. Some SN Ia hosts may be in these groups too.
    # NGC1316 is in Fornax and has TRGB measurement
    # NGC1365 is in Fornax and has Cepheid/TRGB measurements
    
    # First, add constraints linking SN Ia hosts to groups if applicable
    # NGC1316 -> Fornax, NGC1365 -> Fornax
    host_to_group = {
        'NGC1316': 'Fornax',
        'NGC1365': 'Fornax',
    }
    
    for host, grp in host_to_group.items():
        if host in calibrated_sneia_hosts and f'mu_{grp}' in param_idx:
            # mu_host = mu_group (with depth scatter)
            row = np.zeros(n_params)
            row[param_idx[f'mu_{host}']] = 1.0
            row[param_idx[f'mu_{grp}']] = -1.0
            y_list.append(0.0)
            L_list.append(row)
            sigma_list.append(depth_scatter)
            eq_labels.append(f'Group link: {host} -> {grp}')
    
    # SBF calibrator: mF110W = mu_group + M_SBF
    for host, mF, err_mF in sbf_calibrators:
        grp = host_group[host]
        if f'mu_{grp}' not in param_idx:
            continue
        
        row = np.zeros(n_params)
        row[param_idx[f'mu_{grp}']] = 1.0
        row[idx_MSBF] = 1.0
        
        y_list.append(mF)
        L_list.append(row)
        sigma_list.append(np.sqrt(err_mF**2 + depth_scatter**2))
        eq_labels.append(f'SBF cal: {host} ({grp})')
    
    # --- (D) Hubble flow SN Ia ---
    # mB = 5*log10(c*z) + 25 - 5*log10(H0) + M_B
    # => M_B - 5*log10(H0) = mB - 5*log10(c*z) - 25
    # y = mB - 5*log10(c*z) - 25
    
    if use_sneia_hf:
        for z, mB, err_mB, vpec in hubble_flow_sneia:
            cz = c_km * z
            y_val = mB - 5.0 * np.log10(cz) - 25.0
            
            # Peculiar velocity uncertainty in magnitudes
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mB**2 + err_vpec**2)
            
            row = np.zeros(n_params)
            row[idx_MB] = 1.0
            row[idx_5logH0] = -1.0
            
            y_list.append(y_val)
            L_list.append(row)
            sigma_list.append(err_total)
            eq_labels.append(f'HF SN Ia: z={z:.3f}')
    
    # --- (E) Hubble flow SBF ---
    # mF110W = 5*log10(c*z) + 25 - 5*log10(H0) + M_SBF
    # => M_SBF - 5*log10(H0) = mF110W - 5*log10(c*z) - 25
    
    if use_sbf_hf:
        for z, mF, err_mF, vpec in hubble_flow_sbf:
            cz = c_km * z
            y_val = mF - 5.0 * np.log10(cz) - 25.0
            
            err_vpec = (5.0 / np.log(10.0)) * (vpec / cz)
            err_total = np.sqrt(err_mF**2 + err_vpec**2)
            
            row = np.zeros(n_params)
            row[idx_MSBF] = 1.0
            row[idx_5logH0] = -1.0
            
            y_list.append(y_val)
            L_list.append(row)
            sigma_list.append(err_total)
            eq_labels.append(f'HF SBF: z={z:.3f}')
    
    # Convert to arrays
    y = np.array(y_list)
    L = np.array(L_list)
    sigma = np.array(sigma_list)
    
    # Build diagonal covariance matrix
    # Add off-diagonal terms for correlated anchor uncertainties
    n_obs = len(y)
    C = np.diag(sigma**2)
    
    # Add anchor covariance: measurements using the same anchor share anchor uncertainty
    # This creates off-diagonal terms
    for i in range(n_obs):
        for j in range(i+1, n_obs):
            if 'Primary:' in eq_labels[i] and 'Primary:' in eq_labels[j]:
                # Extract anchor from label
                anchor_i = eq_labels[i].split('/')[-1]
                anchor_j = eq_labels[j].split('/')[-1]
                if anchor_i == anchor_j:
                    anchor_name = anchor_i
                    err_a = anchors[anchor_name]['err']
                    # Method-anchor error is also shared
                    method_i = eq_labels[i].split(' via ')[1].split('/')[0]
                    method_j = eq_labels[j].split(' via ')[1].split('/')[0]
                    # Anchor error is shared regardless of method
                    cov_val = err_a**2
                    # If same method and anchor, also share method-anchor error
                    if method_i == method_j:
                        err_m = method_anchor_err.get((method_i, anchor_name), 0.0)
                        cov_val += err_m**2
                    C[i, j] = cov_val
                    C[j, i] = cov_val
    
    return y, L, C, param_names, param_idx, eq_labels


def solve_gls(y, L, C):
    """Solve the GLS system and return best-fit parameters and covariance."""
    C_inv = np.linalg.inv(C)
    LtCiL = L.T @ C_inv @ L
    LtCiy = L.T @ C_inv @ y
    cq = np.linalg.inv(LtCiL)
    q_best = cq @ LtCiy
    
    # Chi-squared
    residuals = y - L @ q_best
    chi2 = residuals.T @ C_inv @ residuals
    dof = len(y) - len(q_best)
    
    return q_best, cq, chi2, dof


def extract_H0(q_best, cq, param_idx):
    """Extract H0 and its uncertainty from the parameter vector."""
    idx = param_idx['5logH0']
    val_5logH0 = q_best[idx]
    err_5logH0 = np.sqrt(cq[idx, idx])
    
    H0 = 10**(val_5logH0 / 5.0)
    # Error propagation: dH0/d(5logH0) = H0 * ln(10)/5
    H0_err = H0 * np.log(10) / 5.0 * err_5logH0
    
    return H0, H0_err, val_5logH0, err_5logH0


# ============================================================
# 3. BASELINE ANALYSIS
# ============================================================

print("=" * 70)
print("BASELINE ANALYSIS: All anchors, all methods, all Hubble flow")
print("=" * 70)

y, L, C, param_names, param_idx, eq_labels = build_gls_system(
    use_anchors=['N4258', 'LMC'],
    use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=True,
    use_sbf_hf=True
)

print(f"\nNumber of observations: {len(y)}")
print(f"Number of parameters: {len(param_names)}")
print(f"Parameters: {param_names}")
print(f"\nObservation equations:")
for i, label in enumerate(eq_labels):
    print(f"  {i:2d}: {label}  (y={y[i]:.4f}, sigma={np.sqrt(C[i,i]):.4f})")

q_best, cq, chi2, dof = solve_gls(y, L, C)

print(f"\nBest-fit parameters:")
for i, name in enumerate(param_names):
    print(f"  {name:15s} = {q_best[i]:8.4f} ± {np.sqrt(cq[i,i]):.4f}")

H0, H0_err, val_5logH0, err_5logH0 = extract_H0(q_best, cq, param_idx)
print(f"\nH0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
print(f"5*log10(H0) = {val_5logH0:.4f} ± {err_5logH0:.4f}")
print(f"chi2 = {chi2:.2f}, dof = {dof}, chi2/dof = {chi2/dof:.2f}")

# ============================================================
# 4. ANALYSIS VARIANTS
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS VARIANTS")
print("=" * 70)

variants = {}

# Variant 1: N4258 only
y1, L1, C1, pn1, pi1, el1 = build_gls_system(
    use_anchors=['N4258'], use_methods=['Cepheid', 'TRGB'])
q1, cq1, chi2_1, dof1 = solve_gls(y1, L1, C1)
H0_1, H0_err_1, _, _ = extract_H0(q1, cq1, pi1)
variants['N4258 only'] = (H0_1, H0_err_1, chi2_1, dof1)

# Variant 2: LMC only
y2, L2, C2, pn2, pi2, el2 = build_gls_system(
    use_anchors=['LMC'], use_methods=['Cepheid'])
q2, cq2, chi2_2, dof2 = solve_gls(y2, L2, C2)
H0_2, H0_err_2, _, _ = extract_H0(q2, cq2, pi2)
variants['LMC only'] = (H0_2, H0_err_2, chi2_2, dof2)

# Variant 3: Cepheids only
y3, L3, C3, pn3, pi3, el3 = build_gls_system(
    use_anchors=['N4258', 'LMC'], use_methods=['Cepheid'])
q3, cq3, chi2_3, dof3 = solve_gls(y3, L3, C3)
H0_3, H0_err_3, _, _ = extract_H0(q3, cq3, pi3)
variants['Cepheids only'] = (H0_3, H0_err_3, chi2_3, dof3)

# Variant 4: TRGB only
y4, L4, C4, pn4, pi4, el4 = build_gls_system(
    use_anchors=['N4258'], use_methods=['TRGB'])
q4, cq4, chi2_4, dof4 = solve_gls(y4, L4, C4)
H0_4, H0_err_4, _, _ = extract_H0(q4, cq4, pi4)
variants['TRGB only'] = (H0_4, H0_err_4, chi2_4, dof4)

# Variant 5: SN Ia Hubble flow only (no SBF)
y5, L5, C5, pn5, pi5, el5 = build_gls_system(
    use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=True, use_sbf_hf=False)
q5, cq5, chi2_5, dof5 = solve_gls(y5, L5, C5)
H0_5, H0_err_5, _, _ = extract_H0(q5, cq5, pi5)
variants['SN Ia HF only'] = (H0_5, H0_err_5, chi2_5, dof5)

# Variant 6: SBF Hubble flow only (no SN Ia HF)
y6, L6, C6, pn6, pi6, el6 = build_gls_system(
    use_anchors=['N4258', 'LMC'], use_methods=['Cepheid', 'TRGB'],
    use_sneia_hf=False, use_sbf_hf=True)
q6, cq6, chi2_6, dof6 = solve_gls(y6, L6, C6)
H0_6, H0_err_6, _, _ = extract_H0(q6, cq6, pi6)
variants['SBF HF only'] = (H0_6, H0_err_6, chi2_6, dof6)

# Variant 7: N4258 + Cepheids only
y7, L7, C7, pn7, pi7, el7 = build_gls_system(
    use_anchors=['N4258'], use_methods=['Cepheid'])
q7, cq7, chi2_7, dof7 = solve_gls(y7, L7, C7)
H0_7, H0_err_7, _, _ = extract_H0(q7, cq7, pi7)
variants['N4258+Cepheids'] = (H0_7, H0_err_7, chi2_7, dof7)

# Variant 8: LMC + Cepheids only
# Same as variant 2
variants['LMC+Cepheids'] = variants['LMC only']

# Variant 9: N4258 + TRGB only
# Same as variant 4
variants['N4258+TRGB'] = variants['TRGB only']

# Print all variants
print(f"\n{'Variant':<25s} {'H0':>8s} {'err':>6s} {'chi2':>8s} {'dof':>4s} {'chi2/dof':>8s}")
print("-" * 60)
for name, (h0, err, c2, d) in variants.items():
    print(f"{name:<25s} {h0:8.2f} {err:6.2f} {c2:8.2f} {d:4d} {c2/d:8.2f}")

# Add baseline
variants['Baseline (all)'] = (H0, H0_err, chi2, dof)

# ============================================================
# 5. SAVE RESULTS
# ============================================================

results = {
    'baseline': {
        'H0': round(float(H0), 2),
        'H0_err': round(float(H0_err), 2),
        'chi2': round(float(chi2), 2),
        'dof': int(dof),
        'chi2_dof': round(float(chi2/dof), 2),
        'parameters': {name: {'value': round(float(q_best[i]), 4), 
                              'error': round(float(np.sqrt(cq[i,i])), 4)}
                       for i, name in enumerate(param_names)}
    },
    'variants': {name: {'H0': round(float(h0), 2), 'H0_err': round(float(err), 2),
                         'chi2': round(float(c2), 2), 'dof': int(d)}
                 for name, (h0, err, c2, d) in variants.items()},
    'CMB_comparison': {
        'Planck_H0': 67.4,
        'Planck_H0_err': 0.5,
        'tension_sigma': round(float((H0 - 67.4) / np.sqrt(H0_err**2 + 0.5**2)), 1)
    }
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/h0_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to outputs/h0_results.json")
print(f"\nBaseline: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
print(f"Planck CMB: H0 = 67.4 ± 0.5 km/s/Mpc")
tension = (H0 - 67.4) / np.sqrt(H0_err**2 + 0.5**2)
print(f"Tension: {tension:.1f}σ")
