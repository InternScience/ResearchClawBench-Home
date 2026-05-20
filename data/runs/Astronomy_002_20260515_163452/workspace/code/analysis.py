#!/usr/bin/env python3
"""
Hubble Constant Measurement via the Local Distance Network
===========================================================
Implements a generalized least squares (GLS) framework combining multiple
distance indicators to determine H0.
"""

import numpy as np
from scipy import linalg
import json
import os

def load_dataset(filepath):
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()
    exec(content, data)
    return {k: data[k] for k in [
        'anchors', 'host_measurements', 'sneia_calibrators', 'sbf_calibrators',
        'hubble_flow_sneia', 'hubble_flow_sbf', 'method_anchor_err',
        'host_group', 'depth_scatter', 'c_km'
    ]}

def build_host_distance_moduli(data):
    hosts = {}
    for host, method, anchor, mu_meas, err_meas in data['host_measurements']:
        anchor_err = data['anchors'][anchor]['err']
        add_err = data['method_anchor_err'].get((method, anchor), 0.0)
        total_err = np.sqrt(err_meas**2 + anchor_err**2 + add_err**2)
        key = (host, method)
        if key not in hosts:
            hosts[key] = []
        hosts[key].append({'mu': mu_meas, 'err': total_err, 'anchor': anchor})
    
    host_mus = {}
    for key, meas in hosts.items():
        mus = np.array([m['mu'] for m in meas])
        errs = np.array([m['err'] for m in meas])
        w = 1.0 / errs**2
        host_mus[key] = {
            'mu': float(np.average(mus, weights=w)),
            'err': float(np.sqrt(1.0 / np.sum(w))),
            'anchors': [m['anchor'] for m in meas]
        }
    return host_mus

def solve_h0(data, host_mus):
    c_km = data['c_km']
    calibrators = []
    for host, mB, err_mB in data['sneia_calibrators']:
        best = None
        for (h, method), mdata in host_mus.items():
            if h == host:
                if best is None or mdata['err'] < best['err']:
                    best = {'mu': mdata['mu'], 'err': mdata['err'], 'method': method}
        if best is not None:
            MB = mB - best['mu']
            err_MB = np.sqrt(err_mB**2 + best['err']**2)
            calibrators.append({
                'host': host, 'mB': mB, 'err_mB': err_mB,
                'mu': best['mu'], 'err_mu': best['err'], 'method': best['method'],
                'MB': MB, 'err_MB': err_MB
            })
    
    z_flow = np.array([s[0] for s in data['hubble_flow_sneia']])
    mB_flow = np.array([s[1] for s in data['hubble_flow_sneia']])
    err_mB_flow = np.array([s[2] for s in data['hubble_flow_sneia']])
    vpec_err = np.array([s[3] for s in data['hubble_flow_sneia']])
    err_vpec_mu = np.abs(5.0 / (np.log(10) * c_km * z_flow)) * vpec_err
    total_err_flow = np.sqrt(err_mB_flow**2 + err_vpec_mu**2)
    offset_flow = 5.0 * np.log10(c_km * z_flow) + 25.0
    
    n_cal = len(calibrators)
    n_flow = len(z_flow)
    n = n_cal + n_flow
    
    A = np.zeros((n, 2))
    d = np.zeros(n)
    C_diag = np.zeros(n)
    
    for j, c in enumerate(calibrators):
        A[j, 0] = 1.0; A[j, 1] = 0.0
        d[j] = c['mB'] - c['mu']
        C_diag[j] = c['err_MB']**2
    
    for i in range(n_flow):
        idx = n_cal + i
        A[idx, 0] = 1.0; A[idx, 1] = 1.0
        d[idx] = mB_flow[i] - offset_flow[i]
        C_diag[idx] = total_err_flow[i]**2
    
    C_inv = np.diag(1.0 / C_diag)
    AtCinv = A.T @ C_inv
    AtCinvA = AtCinv @ A
    AtCinvD = AtCinv @ d
    
    x = linalg.solve(AtCinvA, AtCinvD)
    cov_x = linalg.inv(AtCinvA)
    
    MB_fit = x[0]; q_fit = x[1]
    H0 = 10**(-q_fit / 5.0)
    err_q = np.sqrt(cov_x[1, 1])
    err_H0 = H0 * (np.log(10) / 5.0) * err_q
    err_MB_fit = np.sqrt(cov_x[0, 0])
    
    residuals = d - A @ x
    chi2 = float(residuals @ C_inv @ residuals)
    dof = n - 2
    
    return {
        'H0': float(H0), 'err_H0': float(err_H0),
        'MB': float(MB_fit), 'err_MB': float(err_MB_fit),
        'q': float(q_fit), 'err_q': float(err_q),
        'chi2': chi2, 'dof': dof, 'chi2_dof': chi2/dof,
        'calibrators': calibrators,
        'residuals': residuals.tolist(),
        'z_flow': z_flow.tolist(), 'mB_flow': mB_flow.tolist(),
        'total_err_flow': total_err_flow.tolist(),
        'A': A.tolist(), 'd': d.tolist(), 'C_diag': C_diag.tolist(),
    }

def solve_variant(cal_list, data):
    c_km = data['c_km']
    z_flow = np.array([s[0] for s in data['hubble_flow_sneia']])
    mB_flow = np.array([s[1] for s in data['hubble_flow_sneia']])
    err_mB_flow = np.array([s[2] for s in data['hubble_flow_sneia']])
    vpec_err = np.array([s[3] for s in data['hubble_flow_sneia']])
    err_vpec_mu = np.abs(5.0 / (np.log(10) * c_km * z_flow)) * vpec_err
    total_err_flow = np.sqrt(err_mB_flow**2 + err_vpec_mu**2)
    offset_flow = 5.0 * np.log10(c_km * z_flow) + 25.0
    
    n_cal = len(cal_list); n_flow = len(z_flow); n = n_cal + n_flow
    if n_cal < 2: return None
    
    A = np.zeros((n, 2)); d_vec = np.zeros(n); C_diag = np.zeros(n)
    for j, c in enumerate(cal_list):
        A[j, 0] = 1.0; A[j, 1] = 0.0
        d_vec[j] = c['mB'] - c['mu']; C_diag[j] = c['err_MB']**2
    for i in range(n_flow):
        idx = n_cal + i
        A[idx, 0] = 1.0; A[idx, 1] = 1.0
        d_vec[idx] = mB_flow[i] - offset_flow[i]; C_diag[idx] = total_err_flow[i]**2
    
    C_inv = np.diag(1.0 / C_diag)
    AtCinv = A.T @ C_inv
    x = linalg.solve(AtCinv @ A, AtCinv @ d_vec)
    cov = linalg.inv(AtCinv @ A)
    H0 = 10**(-x[1] / 5.0)
    err_H0 = H0 * (np.log(10) / 5.0) * np.sqrt(cov[1, 1])
    return {'H0': float(H0), 'err_H0': float(err_H0)}

def solve_h0_sbf(data):
    c_km = data['c_km']
    cluster_mu = {'Fornax': 31.51, 'Virgo': 31.04}; cluster_err = 0.15
    MB_vals, MB_errs = [], []
    for host, mF, err_mF in data['sbf_calibrators']:
        group = data['host_group'][host]
        mu_cl = cluster_mu[group]
        MB_vals.append(mF - mu_cl); MB_errs.append(np.sqrt(err_mF**2 + cluster_err**2))
    MB_vals = np.array(MB_vals); MB_errs = np.array(MB_errs)
    w = 1.0 / MB_errs**2
    MB_mean = float(np.average(MB_vals, weights=w)); MB_err = float(np.sqrt(1.0 / np.sum(w)))
    
    z_sbf = np.array([s[0] for s in data['hubble_flow_sbf']])
    mF_sbf = np.array([s[1] for s in data['hubble_flow_sbf']])
    err_mF_sbf = np.array([s[2] for s in data['hubble_flow_sbf']])
    vpec_sbf = np.array([s[3] for s in data['hubble_flow_sbf']])
    err_vpec_mu = np.abs(5.0 / (np.log(10) * c_km * z_sbf)) * vpec_sbf
    total_err = np.sqrt(err_mF_sbf**2 + err_vpec_mu**2)
    
    n = 1 + len(z_sbf)
    A = np.zeros((n, 2)); d = np.zeros(n); C_diag = np.zeros(n)
    A[0, 0] = 1.0; A[0, 1] = 0.0; d[0] = MB_mean; C_diag[0] = MB_err**2
    offset = 5.0 * np.log10(c_km * z_sbf) + 25.0
    for i in range(len(z_sbf)):
        idx = 1 + i
        A[idx, 0] = 1.0; A[idx, 1] = 1.0
        d[idx] = mF_sbf[i] - offset[i]; C_diag[idx] = total_err[i]**2
    
    C_inv = np.diag(1.0 / C_diag)
    AtCinv = A.T @ C_inv
    x = linalg.solve(AtCinv @ A, AtCinv @ d)
    cov = linalg.inv(AtCinv @ A)
    H0 = 10**(-x[1] / 5.0)
    err_H0 = H0 * (np.log(10) / 5.0) * np.sqrt(cov[1, 1])
    return {'H0': float(H0), 'err_H0': float(err_H0), 'MB_sbf': MB_mean, 'MB_err_sbf': MB_err}

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    data = load_dataset('data/H0DN_MinimalDataset.txt')
    host_mus = build_host_distance_moduli(data)
    result = solve_h0(data, host_mus)
    
    print(f"H0 = {result['H0']:.2f} +/- {result['err_H0']:.2f}")
    print(f"MB = {result['MB']:.4f} +/- {result['err_MB']:.4f}")
    print(f"chi2/dof = {result['chi2_dof']:.3f}")
    
    all_cal = result['calibrators']
    variants = {}
    for label, filt in [('Cepheid only', lambda c: c['method']=='Cepheid'),
                         ('TRGB only', lambda c: c['method']=='TRGB')]:
        sub = [c for c in all_cal if filt(c)]
        v = solve_variant(sub, data)
        if v: variants[label] = v; print(f"{label}: H0={v['H0']:.2f}+/-{v['err_H0']:.2f}")
    
    for label, anch in [('N4258 anchor', 'N4258'), ('LMC anchor', 'LMC')]:
        sub = [c for c in all_cal if anch in host_mus.get((c['host'], c['method']), {}).get('anchors', [])]
        v = solve_variant(sub, data)
        if v: variants[label] = v; print(f"{label}: H0={v['H0']:.2f}+/-{v['err_H0']:.2f}")
    
    sbf_result = solve_h0_sbf(data)
    print(f"SBF: H0={sbf_result['H0']:.2f}+/-{sbf_result['err_H0']:.2f}")
    
    H0_planck = 67.4; err_planck = 0.5
    diff = abs(result['H0'] - H0_planck)
    sig = diff / np.sqrt(result['err_H0']**2 + err_planck**2)
    print(f"\nPlanck: {H0_planck}+/-{err_planck}")
    print(f"Tension: {diff:.2f} km/s/Mpc = {sig:.1f} sigma")
    
    outputs = {
        'baseline': {'H0': result['H0'], 'err_H0': result['err_H0'],
                      'MB': result['MB'], 'err_MB': result['err_MB'],
                      'chi2': result['chi2'], 'dof': result['dof'], 'chi2_dof': result['chi2_dof']},
        'variants': variants, 'sbf': sbf_result,
        'planck': {'H0': H0_planck, 'err_H0': err_planck},
        'tension': {'delta_H0': float(diff), 'significance_sigma': float(sig)},
        'host_mus': {f"{k[0]}_{k[1]}": v for k, v in host_mus.items()},
        'calibrators': [{k: (v if not isinstance(v, np.floating) else float(v)) for k, v in c.items()} for c in all_cal],
        'z_flow': result['z_flow'],
        'mB_flow': result['mB_flow'],
        'total_err_flow': result['total_err_flow'],
        'residuals': result['residuals'],
    }
    with open('outputs/results.json', 'w') as f:
        json.dump(outputs, f, indent=2, default=str)
    print("\nResults saved.")
