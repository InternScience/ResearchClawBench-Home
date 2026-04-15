#!/usr/bin/env python3
"""
Hubble Constant Measurement via Local Distance Network
Generalized Least Squares Framework

Implements the full distance ladder from geometric anchors through primary
indicators, secondary calibrations, to Hubble flow observations.
Includes proper treatment of covariance, systematic uncertainties, and
intrinsic scatter of distance indicators.
"""

import numpy as np
import json
import os
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. DATA FROM MINIMAL DATASET
# ============================================================

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
c_km = 299792.458
h0_cmb = 67.4
h0_cmb_err = 0.5

# SNe Ia intrinsic scatter (standardized)
sigma_int_sn = 0.15

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def mu_to_distance(mu):
    return 10 ** ((mu - 25) / 5.0)

def distance_to_mu(d):
    return 5 * np.log10(d) + 25

def mu_from_H0_z(H0, z):
    d_L = c_km * z / H0
    return distance_to_mu(d_L)

def H0_from_mu_z(mu, z):
    d_L = mu_to_distance(mu)
    return c_km * z / d_L

# ============================================================
# 3. HOST DISTANCE MODULI
# ============================================================

print("=" * 60)
print("STEP 1: Host Distance Moduli")
print("=" * 60)

host_data = {}
for host, method, anchor, mu_meas, err_meas in host_measurements:
    if host not in host_data:
        host_data[host] = {'mus': [], 'errs': [], 'methods': [], 'anchors': []}
    host_data[host]['mus'].append(mu_meas)
    host_data[host]['errs'].append(err_meas)
    host_data[host]['methods'].append(method)
    host_data[host]['anchors'].append(anchor)

host_best_mu = {}
host_best_mu_err = {}
for host, data in host_data.items():
    mus = np.array(data['mus'])
    errs = np.array(data['errs'])
    weights = 1.0 / errs**2
    wavg = np.sum(weights * mus) / np.sum(weights)
    werr = 1.0 / np.sqrt(np.sum(weights))
    host_best_mu[host] = wavg
    host_best_mu_err[host] = werr
    print(f"  {host}: mu = {wavg:.3f} +/- {werr:.3f}")

# ============================================================
# 4. SNe Ia CALIBRATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: SNe Ia Absolute Magnitude Calibration")
print("=" * 60)

snia_cal_data = []
for host, mB, err_mB in sneia_calibrators:
    if host in host_best_mu:
        mu_host = host_best_mu[host]
        err_mu = host_best_mu_err[host]
        M_abs = mB - mu_host
        err_M = np.sqrt(err_mB**2 + err_mu**2)
        snia_cal_data.append({'host': host, 'M': M_abs, 'err': err_M, 'mB': mB})
        print(f"  {host}: M_B = {M_abs:.3f} +/- {err_M:.3f}")

M_vals = np.array([d['M'] for d in snia_cal_data])
M_errs = np.array([d['err'] for d in snia_cal_data])

# Check for outliers using median absolute deviation
M_median = np.median(M_vals)
M_mad = np.median(np.abs(M_vals - M_median))
print(f"\n  Median M_B = {M_median:.3f}, MAD = {M_mad:.3f}")

# Use robust weighted mean (exclude >3 sigma outliers)
outlier_mask = np.abs(M_vals - M_median) < 3 * M_mad
if np.sum(outlier_mask) < len(M_vals):
    print(f"  Excluding outliers: {[snia_cal_data[i]['host'] for i in range(len(M_vals)) if not outlier_mask[i]]}")

M_vals_clean = M_vals[outlier_mask]
M_errs_clean = M_errs[outlier_mask]

# Add intrinsic scatter in quadrature
M_errs_total = np.sqrt(M_errs_clean**2 + sigma_int_sn**2)

M_weights = 1.0 / M_errs_total**2
M_mean = np.sum(M_weights * M_vals_clean) / np.sum(M_weights)
M_mean_err_stat = 1.0 / np.sqrt(np.sum(M_weights))
# Systematic error from anchor uncertainty
M_mean_err_sys = 0.04  # typical anchor systematic
M_mean_err = np.sqrt(M_mean_err_stat**2 + M_mean_err_sys**2)

print(f"\n  Weighted mean M_B = {M_mean:.4f} +/- {M_mean_err:.4f}")
print(f"  (stat: {M_mean_err_stat:.4f}, sys: {M_mean_err_sys:.4f})")

# ============================================================
# 5. SNe Ia HUBBLE FLOW
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: SNe Ia Hubble Flow Analysis")
print("=" * 60)

z_hf = np.array([x[0] for x in hubble_flow_sneia])
mB_hf = np.array([x[1] for x in hubble_flow_sneia])
err_mB_hf = np.array([x[2] for x in hubble_flow_sneia])
sigma_v_hf = np.array([x[3] for x in hubble_flow_sneia])

# Model: mB = M_B + mu(z, H0)
# Total error per SN: measurement + intrinsic scatter + peculiar velocity
# Calibration error is shared (systematic)

def chi2_H0(H0):
    """Chi-squared for SNe Ia Hubble flow."""
    chi2 = 0.0
    for i in range(len(z_hf)):
        mu_pred = mu_from_H0_z(H0, z_hf[i])
        sigma_pv = 5.0 / np.log(10) * sigma_v_hf[i] / (c_km * z_hf[i])
        sigma_total = np.sqrt(err_mB_hf[i]**2 + sigma_int_sn**2 + sigma_pv**2)
        residual = mB_hf[i] - M_mean - mu_pred
        chi2 += (residual / sigma_total)**2
    return chi2

result = minimize_scalar(chi2_H0, bounds=(50, 100), method='bounded')
H0_sneia = result.x
chi2_min = result.fun

# Error from chi^2 = chi2_min + 1
def chi2_diff(H0):
    return (chi2_H0(H0) - chi2_min - 1.0)**2

res_lo = minimize_scalar(chi2_diff, bounds=(50, H0_sneia), method='bounded')
res_hi = minimize_scalar(chi2_diff, bounds=(H0_sneia, 100), method='bounded')
H0_sneia_err_stat = (res_hi.x - res_lo.x) / 2.0
# Add systematic from M_B
H0_sneia_err_sys = H0_sneia * M_mean_err_sys * np.log(10) / 5.0
H0_sneia_err = np.sqrt(H0_sneia_err_stat**2 + H0_sneia_err_sys**2)

print(f"  Best-fit H0 = {H0_sneia:.2f} +/- {H0_sneia_err:.2f} km/s/Mpc")
print(f"  (stat: {H0_sneia_err_stat:.2f}, sys: {H0_sneia_err_sys:.2f})")
print(f"  chi2_min = {chi2_min:.2f} (dof = {len(z_hf) - 1})")

# Individual H0 values
mu_obs_sneia = mB_hf - M_mean
H0_individual_sneia = np.array([H0_from_mu_z(mu, z) for z, mu in zip(z_hf, mu_obs_sneia)])
for z, H0_i in zip(z_hf, H0_individual_sneia):
    print(f"    z={z:.3f}: H0 = {H0_i:.1f}")

# ============================================================
# 6. SBF ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: SBF Distance Ladder")
print("=" * 60)

sbf_cluster_mu = {'Fornax': 31.50, 'Virgo': 31.10}
sbf_cluster_err = {'Fornax': 0.15, 'Virgo': 0.15}

sbf_cal_data = []
for host, mF, err_mF in sbf_calibrators:
    group = host_group.get(host, None)
    if group and group in sbf_cluster_mu:
        mu_cluster = sbf_cluster_mu[group]
        err_mu_cluster = sbf_cluster_err[group]
        M_F = mF - mu_cluster
        err_M_F = np.sqrt(err_mF**2 + err_mu_cluster**2)
        sbf_cal_data.append({'host': host, 'group': group, 'M': M_F, 'err': err_M_F, 'mF': mF})
        print(f"  {host} ({group}): M_F110W = {M_F:.3f} +/- {err_M_F:.3f}")

if len(sbf_cal_data) > 0:
    M_F_vals = np.array([d['M'] for d in sbf_cal_data])
    M_F_errs = np.array([d['err'] for d in sbf_cal_data])
    M_F_weights = 1.0 / M_F_errs**2
    M_F_mean = np.sum(M_F_weights * M_F_vals) / np.sum(M_F_weights)
    M_F_mean_err = 1.0 / np.sqrt(np.sum(M_F_weights))
    
    print(f"\n  Weighted mean M_F110W = {M_F_mean:.4f} +/- {M_F_mean_err:.4f}")
    
    z_hf_sbf = np.array([x[0] for x in hubble_flow_sbf])
    mF_hf = np.array([x[1] for x in hubble_flow_sbf])
    err_mF_hf = np.array([x[2] for x in hubble_flow_sbf])
    sigma_v_hf_sbf = np.array([x[3] for x in hubble_flow_sbf])
    
    def chi2_H0_sbf(H0):
        chi2 = 0.0
        for i in range(len(z_hf_sbf)):
            mu_pred = mu_from_H0_z(H0, z_hf_sbf[i])
            sigma_pv = 5.0 / np.log(10) * sigma_v_hf_sbf[i] / (c_km * z_hf_sbf[i])
            sigma_total = np.sqrt(err_mF_hf[i]**2 + sigma_pv**2)
            residual = mF_hf[i] - M_F_mean - mu_pred
            chi2 += (residual / sigma_total)**2
        return chi2
    
    result_sbf = minimize_scalar(chi2_H0_sbf, bounds=(40, 200), method='bounded')
    H0_sbf = result_sbf.x
    chi2_min_sbf = result_sbf.fun
    
    def chi2_diff_sbf(H0):
        return (chi2_H0_sbf(H0) - chi2_min_sbf - 1.0)**2
    
    rlo = minimize_scalar(chi2_diff_sbf, bounds=(40, H0_sbf), method='bounded')
    rhi = minimize_scalar(chi2_diff_sbf, bounds=(H0_sbf, 200), method='bounded')
    H0_sbf_err = (rhi.x - rlo.x) / 2.0
    
    print(f"  SBF H0 = {H0_sbf:.2f} +/- {H0_sbf_err:.2f} km/s/Mpc")
    
    mu_obs_sbf = mF_hf - M_F_mean
    H0_individual_sbf = np.array([H0_from_mu_z(mu, z) for z, mu in zip(z_hf_sbf, mu_obs_sbf)])
else:
    H0_sbf = H0_sbf_err = M_F_mean = M_F_mean_err = None
    H0_individual_sbf = None
    z_hf_sbf = None

# ============================================================
# 7. GLS COMBINATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: GLS Combination")
print("=" * 60)

all_H0 = list(H0_individual_sneia)
all_err_ind = list(H0_individual_sneia * np.sqrt(err_mB_hf**2 + sigma_int_sn**2) * np.log(10) / 5.0)
all_labels = ['SNe Ia'] * len(z_hf)

if H0_individual_sbf is not None:
    all_H0.extend(H0_individual_sbf.tolist())
    all_err_ind.extend((H0_individual_sbf * err_mF_hf * np.log(10) / 5.0).tolist())
    all_labels.extend(['SBF'] * len(z_hf_sbf))

all_H0 = np.array(all_H0)
all_err_ind = np.array(all_err_ind)
n = len(all_H0)
n_sneia = len(H0_individual_sneia)
n_sbf = len(H0_individual_sbf) if H0_individual_sbf is not None else 0

C = np.diag(all_err_ind**2)

# Shared calibration covariance
if n_sneia > 1:
    sigma_M_log = M_mean_err * np.log(10) / 5.0
    for i in range(n_sneia):
        for j in range(n_sneia):
            if i != j:
                C[i, j] += all_H0[i] * all_H0[j] * sigma_M_log**2

if n_sbf > 1 and M_F_mean_err is not None:
    sigma_M_F_log = M_F_mean_err * np.log(10) / 5.0
    for i in range(n_sneia, n):
        for j in range(n_sneia, n):
            if i != j:
                C[i, j] += all_H0[i] * all_H0[j] * sigma_M_F_log**2

C_inv = np.linalg.inv(C)
ones = np.ones(n)
H0_gls = float(np.dot(ones, np.dot(C_inv, all_H0)) / np.dot(ones, np.dot(C_inv, ones)))
H0_gls_err = float(1.0 / np.sqrt(np.dot(ones, np.dot(C_inv, ones))))

print(f"  GLS combined H0 = {H0_gls:.2f} +/- {H0_gls_err:.2f} km/s/Mpc")
print(f"  Precision: {100*H0_gls_err/H0_gls:.2f}%")

tension = abs(H0_gls - h0_cmb) / np.sqrt(H0_gls_err**2 + h0_cmb_err**2)
print(f"  Tension with CMB: {tension:.1f} sigma")

# ============================================================
# 8. VARIANTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Analysis Variants")
print("=" * 60)

variants = {}
variants['SNe Ia (baseline)'] = {'H0': float(H0_sneia), 'H0_err': float(H0_sneia_err)}
if H0_sbf is not None:
    variants['SBF only'] = {'H0': float(H0_sbf), 'H0_err': float(H0_sbf_err)}
variants['GLS Combined'] = {'H0': float(H0_gls), 'H0_err': float(H0_gls_err)}

# Cepheid-only
cep_hosts = set(h for h, m, a, _, _ in host_measurements if m == 'Cepheid')
cep_cal = [(h, mB, e) for h, mB, e in sneia_calibrators if h in cep_hosts]
if len(cep_cal) > 1:
    cep_M = np.array([mB - host_best_mu[h] for h, mB, e in cep_cal])
    cep_e = np.array([np.sqrt(e**2 + host_best_mu_err[h]**2) for h, mB, e in cep_cal])
    cep_e_tot = np.sqrt(cep_e**2 + sigma_int_sn**2)
    w = 1.0/cep_e_tot**2
    M_cep = np.sum(w*cep_M)/np.sum(w)
    M_cep_err = 1.0/np.sqrt(np.sum(w))
    
    def chi2_cep(H0):
        chi2 = 0
        for i in range(len(z_hf)):
            mu_p = mu_from_H0_z(H0, z_hf[i])
            s = np.sqrt(err_mB_hf[i]**2 + sigma_int_sn**2 + (5/np.log(10)*sigma_v_hf[i]/(c_km*z_hf[i]))**2)
            chi2 += ((mB_hf[i] - M_cep - mu_p) / s)**2
        return chi2
    
    res = minimize_scalar(chi2_cep, bounds=(50, 100), method='bounded')
    H0_cep = res.x
    cm = res.fun
    def cd(H0): return (chi2_cep(H0) - cm - 1)**2
    rlo = minimize_scalar(cd, bounds=(50, H0_cep), method='bounded')
    rhi = minimize_scalar(cd, bounds=(H0_cep, 100), method='bounded')
    H0_cep_err = (rhi.x - rlo.x) / 2
    variants['Cepheid-calibrated SNe'] = {'H0': float(H0_cep), 'H0_err': float(H0_cep_err)}

# TRGB-only
trgb_hosts = set(h for h, m, a, _, _ in host_measurements if m == 'TRGB')
trgb_cal = [(h, mB, e) for h, mB, e in sneia_calibrators if h in trgb_hosts]
if len(trgb_cal) > 1:
    trgb_M = np.array([mB - host_best_mu[h] for h, mB, e in trgb_cal])
    trgb_e = np.array([np.sqrt(e**2 + host_best_mu_err[h]**2) for h, mB, e in trgb_cal])
    trgb_e_tot = np.sqrt(trgb_e**2 + sigma_int_sn**2)
    w = 1.0/trgb_e_tot**2
    M_trgb = np.sum(w*trgb_M)/np.sum(w)
    M_trgb_err = 1.0/np.sqrt(np.sum(w))
    
    def chi2_trgb(H0):
        chi2 = 0
        for i in range(len(z_hf)):
            mu_p = mu_from_H0_z(H0, z_hf[i])
            s = np.sqrt(err_mB_hf[i]**2 + sigma_int_sn**2 + (5/np.log(10)*sigma_v_hf[i]/(c_km*z_hf[i]))**2)
            chi2 += ((mB_hf[i] - M_trgb - mu_p) / s)**2
        return chi2
    
    res = minimize_scalar(chi2_trgb, bounds=(50, 100), method='bounded')
    H0_trgb = res.x
    cm = res.fun
    def cd(H0): return (chi2_trgb(H0) - cm - 1)**2
    rlo = minimize_scalar(cd, bounds=(50, H0_trgb), method='bounded')
    rhi = minimize_scalar(cd, bounds=(H0_trgb, 100), method='bounded')
    H0_trgb_err = (rhi.x - rlo.x) / 2
    variants['TRGB-calibrated SNe'] = {'H0': float(H0_trgb), 'H0_err': float(H0_trgb_err)}

# Redshift cuts
for z_min in [0.03, 0.04, 0.05]:
    mask = z_hf >= z_min
    if np.sum(mask) >= 2:
        z_c = z_hf[mask]
        m_c = mB_hf[mask]
        e_c = err_mB_hf[mask]
        v_c = sigma_v_hf[mask]
        
        def chi2_cut(H0):
            chi2 = 0
            for i in range(len(z_c)):
                mu_p = mu_from_H0_z(H0, z_c[i])
                s = np.sqrt(e_c[i]**2 + sigma_int_sn**2 + (5/np.log(10)*v_c[i]/(c_km*z_c[i]))**2)
                chi2 += ((m_c[i] - M_mean - mu_p) / s)**2
            return chi2
        
        res = minimize_scalar(chi2_cut, bounds=(50, 100), method='bounded')
        H0_c = res.x
        cm = res.fun
        def cd(H0): return (chi2_cut(H0) - cm - 1)**2
        rlo = minimize_scalar(cd, bounds=(50, H0_c), method='bounded')
        rhi = minimize_scalar(cd, bounds=(H0_c, 100), method='bounded')
        H0_c_err = (rhi.x - rlo.x) / 2
        variants[f'SNe Ia z >= {z_min}'] = {'H0': float(H0_c), 'H0_err': float(H0_c_err)}

for name, v in variants.items():
    print(f"  {name:30s}: H0 = {v['H0']:.2f} +/- {v['H0_err']:.2f} km/s/Mpc")

# ============================================================
# 9. SAVE RESULTS
# ============================================================

os.makedirs('outputs', exist_ok=True)

results = {
    'baseline': {
        'H0': float(H0_gls),
        'H0_err': float(H0_gls_err),
        'precision_pct': float(100 * H0_gls_err / H0_gls)
    },
    'gls_combined': {
        'H0': float(H0_gls),
        'H0_err': float(H0_gls_err),
        'precision_pct': float(100 * H0_gls_err / H0_gls)
    },
    'sneia_result': {
        'H0': float(H0_sneia),
        'H0_err': float(H0_sneia_err),
        'M_B_mean': float(M_mean),
        'M_B_err': float(M_mean_err),
        'M_B_median': float(M_median)
    },
    'sbf_result': {
        'H0': float(H0_sbf) if H0_sbf is not None else None,
        'H0_err': float(H0_sbf_err) if H0_sbf_err is not None else None,
        'M_F_mean': float(M_F_mean) if M_F_mean is not None else None,
        'M_F_err': float(M_F_mean_err) if M_F_mean_err is not None else None
    },
    'variants': variants,
    'cmb': {'H0': h0_cmb, 'H0_err': h0_cmb_err},
    'tension_sigma': float(tension),
    'individual_sneia': [
        {'z': float(z), 'H0': float(h)}
        for z, h in zip(z_hf, H0_individual_sneia)
    ],
    'individual_sbf': [
        {'z': float(z), 'H0': float(h)}
        for z, h in zip(z_hf_sbf, H0_individual_sbf)
    ] if H0_individual_sbf is not None else [],
    'host_distances': {h: {'mu': float(v), 'err': float(host_best_mu_err[h])} for h, v in host_best_mu.items()},
    'snia_absolute_mags': {d['host']: {'M': float(d['M']), 'err': float(d['err'])} for d in snia_cal_data}
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save('outputs/H0_individual_sneia.npy', H0_individual_sneia)
np.save('outputs/z_hf.npy', z_hf)
if H0_individual_sbf is not None:
    np.save('outputs/H0_individual_sbf.npy', H0_individual_sbf)
    np.save('outputs/z_hf_sbf.npy', z_hf_sbf)
np.save('outputs/covariance_matrix.npy', C)

# Save chi^2 scan for plotting
H0_scan = np.linspace(50, 100, 200)
chi2_scan = np.array([chi2_H0(h) for h in H0_scan])
np.save('outputs/H0_scan.npy', H0_scan)
np.save('outputs/chi2_profile.npy', chi2_scan)

print(f"\nResults saved to outputs/")
print("Done.")
