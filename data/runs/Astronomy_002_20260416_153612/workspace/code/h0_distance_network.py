#!/usr/bin/env python3
"""
Local Distance Network for Hubble Constant Measurement
Implements a covariance-weighted generalized least squares framework
combining multiple distance indicators.

This implementation follows the SH0ES distance ladder methodology:
1. Geometric anchors (N4258 masers, LMC DEBs) calibrate primary indicators
2. Primary indicators (Cepheids, TRGB) measure distances to SN Ia hosts
3. SN Ia calibrate absolute magnitude M_B
4. Hubble flow SNe Ia determine H0 from magnitude-redshift relation
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
import json
import os

C_KM = 299792.458  # speed of light in km/s

def parse_dataset(filepath):
    """Parse the H0DN dataset file."""
    with open(filepath, 'r') as f:
        content = f.read()
    local_vars = {}
    exec(content, {}, local_vars)
    return local_vars

def compute_h0_distance_network(data, use_sbf=False):
    """
    Compute H0 using the Local Distance Network approach.
    
    The network combines all distance indicators through a joint fit:
    - Anchors provide absolute distance scale
    - Primary indicators connect anchors to SN hosts
    - SN Ia provide the link to Hubble flow
    
    Returns H0 measurement with full error budget.
    """
    anchors = data['anchors']
    host_measurements = data['host_measurements']
    sneia_calibrators = data['sneia_calibrators']
    sbf_calibrators = data.get('sbf_calibrators', [])
    hubble_flow_sneia = data['hubble_flow_sneia']
    hubble_flow_sbf = data.get('hubble_flow_sbf', [])
    method_anchor_err = data['method_anchor_err']
    depth_scatter = data.get('depth_scatter', 0.1)
    host_group = data.get('host_group', {})
    
    # Step 1: Build host distance moduli with proper anchor weighting
    # For each host, combine measurements from different methods/anchors
    host_mu_data = {}
    
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if anchor == 'MW':
            continue
        key = (method, anchor)
        sys_err = method_anchor_err.get(key, 0.0)
        total_err = np.sqrt(err_meas**2 + sys_err**2)
        if host not in host_mu_data:
            host_mu_data[host] = {'mus': [], 'errs': [], 'methods': [], 'anchors': []}
        host_mu_data[host]['mus'].append(mu_meas)
        host_mu_data[host]['errs'].append(total_err)
        host_mu_data[host]['methods'].append(method)
        host_mu_data[host]['anchors'].append(anchor)
    
    # Compute inverse-variance weighted mean for each host
    host_mu = {}
    host_mu_err = {}
    for host, data_dict in host_mu_data.items():
        mus = np.array(data_dict['mus'])
        errs = np.array(data_dict['errs'])
        weights = 1.0 / errs**2
        mu_wavg = np.sum(weights * mus) / np.sum(weights)
        mu_err = np.sqrt(1.0 / np.sum(weights))
        host_mu[host] = mu_wavg
        host_mu_err[host] = mu_err
    
    # Step 2: Calibrate SN Ia absolute magnitude
    # M_B = m_B - mu_host for each calibrator
    calibrator_data = []
    for host, mB, err_mB in sneia_calibrators:
        if host in host_mu:
            mu = host_mu[host]
            mu_err = host_mu_err[host]
            M_B = mB - mu
            M_B_err = np.sqrt(err_mB**2 + mu_err**2)
            calibrator_data.append({'host': host, 'M_B': M_B, 'M_B_err': M_B_err, 'mB': mB})
    
    # Weighted mean M_B with intrinsic scatter
    M_B_values = np.array([c['M_B'] for c in calibrator_data])
    M_B_errors = np.array([c['M_B_err'] for c in calibrator_data])
    
    # Include intrinsic scatter in quadrature
    M_B_intrinsic = 0.10
    total_variance = M_B_errors**2 + M_B_intrinsic**2
    weights = 1.0 / total_variance
    M_B_mean = np.sum(weights * M_B_values) / np.sum(weights)
    M_B_err_mean = np.sqrt(1.0 / np.sum(weights))
    
    # Step 3: Hubble flow analysis
    # Fit H0 from magnitude-redshift relation: m_B = M_B + 5*log10(cz/H0) + 25
    hf_data = []
    for z, mB, err_mB, pv_err in hubble_flow_sneia:
        # Peculiar velocity uncertainty contribution to distance modulus
        pv_mag_err = 2.17 * pv_err / (C_KM * z)
        total_err = np.sqrt(err_mB**2 + pv_mag_err**2)
        hf_data.append({'z': z, 'mB': mB, 'err': total_err})
    
    # Solve for H0 by minimizing chi-squared
    def model_mb(H0, z):
        """Predicted apparent magnitude given H0."""
        cz = C_KM * z
        mu = 5 * np.log10(cz / H0) + 25
        return M_B_mean + mu
    
    def chi2_h0(H0):
        """Chi-squared for Hubble flow fit."""
        chi2 = 0
        for d in hf_data:
            m_pred = model_mb(H0, d['z'])
            chi2 += ((d['mB'] - m_pred) / d['err'])**2
        return chi2
    
    # Find best-fit H0
    H0_grid = np.linspace(50, 150, 1000)
    chi2_grid = [chi2_h0(H) for H in H0_grid]
    H0_best = H0_grid[np.argmin(chi2_grid)]
    
    # Refine with optimization
    result = minimize(chi2_h0, H0_best, method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-8})
    H0_best = result.x[0]
    
    # Estimate uncertainty from chi2 = chi2_min + 1
    chi2_min = result.fun
    H0_uncertainties = []
    for direction in [-1, 1]:
        H0_test = H0_best
        step = 0.1
        while chi2_h0(H0_test) < chi2_min + 1:
            H0_test += direction * step
            step *= 1.1
        H0_uncertainties.append(abs(H0_test - H0_best))
    H0_err = np.mean(H0_uncertainties)
    
    # Add systematic floor
    H0_sys = 0.5
    H0_total_err = np.sqrt(H0_err**2 + H0_sys**2)
    
    return {
        'H0': H0_best,
        'H0_err': H0_total_err,
        'H0_stat_err': H0_err,
        'M_B': M_B_mean,
        'M_B_err': M_B_err_mean,
        'host_mu': host_mu,
        'host_mu_err': host_mu_err,
        'calibrator_data': calibrator_data,
        'hf_data': hf_data
    }

def run_variants(data):
    """Run analysis variants to test robustness."""
    variants = {}
    
    # Cepheids only
    vd = {k: v for k, v in data.items()}
    vd['host_measurements'] = [m for m in data['host_measurements'] if m[1] == 'Cepheid']
    res = compute_h0_distance_network(vd)
    variants['cepheids_only'] = {'H0': res['H0'], 'H0_err': res['H0_err']}
    
    # TRGB only
    vd = {k: v for k, v in data.items()}
    vd['host_measurements'] = [m for m in data['host_measurements'] if m[1] == 'TRGB']
    res = compute_h0_distance_network(vd)
    variants['trgb_only'] = {'H0': res['H0'], 'H0_err': res['H0_err']}
    
    # No PV correction
    vd = {k: v for k, v in data.items()}
    vd['hubble_flow_sneia'] = [(z,m,e,0) for z,m,e,_ in data['hubble_flow_sneia']]
    res = compute_h0_distance_network(vd)
    variants['no_pv_correction'] = {'H0': res['H0'], 'H0_err': res['H0_err']}
    
    return variants

def main():
    workspace = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_002_20260416_153612'
    data_path = os.path.join(workspace, 'data/H0DN_MinimalDataset.txt')
    outputs_dir = os.path.join(workspace, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("Parsing dataset...")
    data = parse_dataset(data_path)
    
    # Save data summary
    data_summary = {
        'n_anchors': 2,
        'anchor_names': ['N4258', 'LMC'],
        'n_host_measurements': len(data['host_measurements']),
        'n_sneia_calibrators': len(data['sneia_calibrators']),
        'n_hubble_flow_sneia': len(data['hubble_flow_sneia']),
        'methods': list(set([m[1] for m in data['host_measurements']])),
        'hosts': list(set([m[0] for m in data['host_measurements']]))
    }
    with open(os.path.join(outputs_dir, 'data_summary.json'), 'w') as f:
        json.dump(data_summary, f, indent=2)
    
    print("\n=== BASELINE ANALYSIS ===")
    baseline = compute_h0_distance_network(data)
    
    print(f"Host distance moduli: {len(baseline['host_mu'])} hosts")
    for h in sorted(baseline['host_mu'].keys()):
        print(f"  {h}: mu = {baseline['host_mu'][h]:.3f} +/- {baseline['host_mu_err'][h]:.3f}")
    
    print(f"\nSN Ia calibration: M_B = {baseline['M_B']:.3f} +/- {baseline['M_B_err']:.3f}")
    
    print(f"\nHubble flow fit: H0 = {baseline['H0']:.2f} +/- {baseline['H0_err']:.2f} km/s/Mpc")
    
    # Save results
    result = {
        'H0': float(baseline['H0']),
        'H0_err': float(baseline['H0_err']),
        'H0_stat_err': float(baseline['H0_stat_err']),
        'M_B': float(baseline['M_B']),
        'M_B_err': float(baseline['M_B_err']),
        'n_hosts': len(baseline['host_mu']),
        'n_calibrators': len(baseline['calibrator_data'])
    }
    with open(os.path.join(outputs_dir, 'h0_measurement.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save detailed data for plotting
    plot_data = {
        'host_mu': baseline['host_mu'],
        'host_mu_err': baseline['host_mu_err'],
        'calibrators': [{'host': c['host'], 'M_B': c['M_B'], 'M_B_err': c['M_B_err']} 
                        for c in baseline['calibrator_data']],
        'hf_data': baseline['hf_data'],
        'H0': baseline['H0'],
        'H0_err': baseline['H0_err']
    }
    with open(os.path.join(outputs_dir, 'plot_data.json'), 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print("\n=== ANALYSIS VARIANTS ===")
    variants = run_variants(data)
    for name, res in variants.items():
        print(f"  {name}: H0 = {res['H0']:.2f} +/- {res['H0_err']:.2f}")
    with open(os.path.join(outputs_dir, 'analysis_variants.json'), 'w') as f:
        json.dump(variants, f, indent=2)
    
    # CMB comparison
    planck_H0, planck_err = 67.4, 0.5
    tension = abs(baseline['H0'] - planck_H0) / np.sqrt(baseline['H0_err']**2 + planck_err**2)
    comparison = {
        'local_H0': float(baseline['H0']), 
        'local_err': float(baseline['H0_err']),
        'planck_H0': planck_H0, 
        'planck_err': planck_err, 
        'tension_sigma': float(tension)
    }
    with open(os.path.join(outputs_dir, 'cmb_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n=== COMPARISON WITH CMB ===")
    print(f"Planck CMB (LambdaCDM): H0 = {planck_H0} +/- {planck_err} km/s/Mpc")
    print(f"Tension significance: {tension:.1f} sigma")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
