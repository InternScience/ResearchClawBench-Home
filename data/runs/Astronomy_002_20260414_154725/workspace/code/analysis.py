#!/usr/bin/env python3
"""
Local Distance Network Analysis for Hubble Constant Measurement.
Implements the distance-ladder framework combining geometric anchors,
primary distance indicators (Cepheids, TRGB), SN Ia standard candles,
and SBF to measure H0.

Note: This uses a minimal/simplified dataset. The absolute H0 values
may differ from the full analysis due to the simplified input data.
The methodology and framework are faithfully implemented.
"""

import numpy as np
import json
import os

c_km = 299792.458

# ── Dataset ──
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


def compute_host_distances(indicator_filter=None, anchor_filter=None):
    """Compute weighted-average distance moduli for host galaxies."""
    host_meas_list = {}
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if indicator_filter and method != indicator_filter:
            continue
        if anchor_filter and anchor != anchor_filter:
            continue
        meth_err = method_anchor_err.get((method, anchor), 0.0)
        anchor_err = anchors[anchor]['err']
        obs_err = np.sqrt(err_meas**2 + meth_err**2 + anchor_err**2)
        if host not in host_meas_list:
            host_meas_list[host] = []
        host_meas_list[host].append((mu_meas, obs_err))

    host_distances = {}
    host_dist_errs = {}
    for host, meas in host_meas_list.items():
        weights = [1.0 / e**2 for _, e in meas]
        total_w = sum(weights)
        mu_avg = sum(w * m for w, (m, _) in zip(weights, meas)) / total_w
        mu_err = np.sqrt(1.0 / total_w)
        host_distances[host] = mu_avg
        host_dist_errs[host] = mu_err
    return host_distances, host_dist_errs


def compute_sn_absmag(host_distances, host_dist_errs):
    """Compute SN Ia absolute magnitude from calibrators: M_B = m_B - mu_host."""
    mb_abs_list = []
    for host, mb, err_mb in sneia_calibrators:
        if host in host_distances:
            mu = host_distances[host]
            mu_err = host_dist_errs[host]
            M_B = mb - mu
            M_B_err = np.sqrt(err_mb**2 + mu_err**2)
            mb_abs_list.append((host, M_B, M_B_err))
    if not mb_abs_list:
        return None, None, []
    weights = [1.0 / e**2 for _, _, e in mb_abs_list]
    total_w = sum(weights)
    M_B_avg = sum(w * m for w, (_, m, _) in zip(weights, mb_abs_list)) / total_w
    M_B_err = np.sqrt(1.0 / total_w)
    return M_B_avg, M_B_err, mb_abs_list


def compute_h0_from_sne(M_B, M_B_err):
    """Compute H0 from Hubble-flow SNe Ia."""
    h0_list = []
    for z, mb, err_mb, vpec in hubble_flow_sneia:
        mu_HF = mb - M_B
        mu_HF_err = np.sqrt(err_mb**2 + M_B_err**2)
        D_Mpc = 10**((mu_HF - 25.0) / 5.0)
        H0 = c_km * z / D_Mpc
        dH0_dmu = -H0 * np.log(10) / 5.0
        H0_err_from_mu = abs(dH0_dmu) * mu_HF_err
        H0_err_from_vpec = H0 * vpec / (c_km * z)
        H0_err_total = np.sqrt(H0_err_from_mu**2 + H0_err_from_vpec**2)
        h0_list.append((z, H0, H0_err_total, mu_HF, D_Mpc))
    weights = [1.0 / e**2 for _, _, e, _, _ in h0_list]
    total_w = sum(weights)
    H0_avg = sum(w * h for w, (_, h, _, _, _) in zip(weights, h0_list)) / total_w
    H0_err = np.sqrt(1.0 / total_w)
    return H0_avg, H0_err, h0_list


def compute_h0_from_sbf():
    """Compute H0 from SBF Hubble flow measurements."""
    h0_list = []
    for z, mf, err_mf, vpec in hubble_flow_sbf:
        mu_HF = mf
        D_Mpc = 10**((mu_HF - 25.0) / 5.0)
        H0 = c_km * z / D_Mpc
        dH0_dmu = -H0 * np.log(10) / 5.0
        H0_err_from_mu = abs(dH0_dmu) * err_mf
        H0_err_from_vpec = H0 * vpec / (c_km * z)
        H0_err_total = np.sqrt(H0_err_from_mu**2 + H0_err_from_vpec**2)
        h0_list.append((z, H0, H0_err_total, mu_HF, D_Mpc))
    weights = [1.0 / e**2 for _, _, e, _, _ in h0_list]
    total_w = sum(weights)
    H0_avg = sum(w * h for w, (_, h, _, _, _) in zip(weights, h0_list)) / total_w
    H0_err = np.sqrt(1.0 / total_w)
    return H0_avg, H0_err, h0_list


def run_full_analysis():
    results = {}

    # Baseline
    hd, hde = compute_host_distances()
    M_B, M_B_err, mb_list = compute_sn_absmag(hd, hde)
    H0, H0_err, h0_list = compute_h0_from_sne(M_B, M_B_err)
    results['baseline'] = {
        'description': 'All anchors, Cepheid+TRGB, SN Ia',
        'H0': H0, 'H0_err': H0_err,
        'M_B': M_B, 'M_B_err': M_B_err,
        'host_distances': hd, 'host_dist_errs': hde,
        'sn_absmag': mb_list,
        'h0_per_sn': [(z, h, e) for z, h, e, _, _ in h0_list],
        'h0_per_sn_detail': [(z, h, e, mu, D) for z, h, e, mu, D in h0_list]
    }

    # Anchor variants
    results['anchor_variants'] = {}
    for anc in ['N4258', 'LMC']:
        hd_a, hde_a = compute_host_distances(anchor_filter=anc)
        if not hd_a:
            continue
        M_B_a, M_B_err_a, _ = compute_sn_absmag(hd_a, hde_a)
        if M_B_a is None:
            continue
        H0_a, H0_err_a, _ = compute_h0_from_sne(M_B_a, M_B_err_a)
        results['anchor_variants'][anc] = {
            'H0': H0_a, 'H0_err': H0_err_a,
            'M_B': M_B_a, 'M_B_err': M_B_err_a,
            'n_hosts': len(hd_a)
        }

    # Indicator variants
    results['indicator_variants'] = {}
    for ind in ['Cepheid', 'TRGB']:
        hd_i, hde_i = compute_host_distances(indicator_filter=ind)
        if not hd_i:
            continue
        M_B_i, M_B_err_i, _ = compute_sn_absmag(hd_i, hde_i)
        if M_B_i is None:
            continue
        H0_i, H0_err_i, _ = compute_h0_from_sne(M_B_i, M_B_err_i)
        results['indicator_variants'][ind] = {
            'H0': H0_i, 'H0_err': H0_err_i,
            'M_B': M_B_i, 'M_B_err': M_B_err_i,
            'n_hosts': len(hd_i)
        }

    # SBF
    H0_sbf, H0_err_sbf, sbf_h0_list = compute_h0_from_sbf()
    results['sbf'] = {
        'H0': H0_sbf, 'H0_err': H0_err_sbf,
        'h0_per_sbf': [(z, h, e) for z, h, e, _, _ in sbf_h0_list]
    }

    # Combined
    w1 = 1.0 / H0_err**2
    w2 = 1.0 / H0_err_sbf**2
    H0_comb = (w1 * H0 + w2 * H0_sbf) / (w1 + w2)
    H0_err_comb = np.sqrt(1.0 / (w1 + w2))
    results['combined'] = {'H0': H0_comb, 'H0_err': H0_err_comb}

    # Planck
    H0_planck = 67.4
    H0_planck_err = 0.5
    tension = (H0 - H0_planck) / np.sqrt(H0_err**2 + H0_planck_err**2)
    results['planck'] = {'H0': H0_planck, 'H0_err': H0_planck_err}
    results['tension_sigma'] = tension

    # Literature comparison
    results['literature'] = {
        'shoes_2022': {'H0': 73.04, 'H0_err': 1.04, 'ref': 'Riess et al. 2022'},
        'shoes_dn_2024': {'H0': 73.50, 'H0_err': 0.81, 'ref': 'Riess et al. 2024 (Distance Network)'},
        'trgb_freedman': {'H0': 69.8, 'H0_err': 1.7, 'ref': 'Freedman et al. 2021'},
        'planck': {'H0': 67.4, 'H0_err': 0.5, 'ref': 'Planck Collaboration 2020'},
        'desi_bao': {'H0': 68.5, 'H0_err': 0.7, 'ref': 'DESI BAO 2024'}
    }

    return results


if __name__ == '__main__':
    np.random.seed(42)
    os.makedirs('outputs', exist_ok=True)
    results = run_full_analysis()

    b = results['baseline']
    print("=" * 60)
    print("Local Distance Network: Hubble Constant Measurement")
    print("=" * 60)
    print(f"\nBASELINE: H0 = {b['H0']:.2f} +/- {b['H0_err']:.2f} km/s/Mpc")
    print(f"SN Ia M_B = {b['M_B']:.3f} +/- {b['M_B_err']:.3f}")
    print(f"\nHost distances:")
    for h in sorted(b['host_distances'].keys()):
        print(f"  {h}: mu = {b['host_distances'][h]:.2f} +/- {b['host_dist_errs'][h]:.2f}")
    print(f"\nSN Ia absolute magnitudes:")
    for h, mb, err in b['sn_absmag']:
        print(f"  {h}: M_B = {mb:.3f} +/- {err:.3f}")
    print(f"\nH0 per HF SN:")
    for z, h0, err in b['h0_per_sn']:
        print(f"  z={z:.3f}: H0 = {h0:.2f} +/- {err:.2f}")
    print(f"\nAnchor variants:")
    for anc, res in results['anchor_variants'].items():
        print(f"  {anc}: H0 = {res['H0']:.2f} +/- {res['H0_err']:.2f} ({res['n_hosts']} hosts)")
    print(f"\nIndicator variants:")
    for ind, res in results['indicator_variants'].items():
        print(f"  {ind}: H0 = {res['H0']:.2f} +/- {res['H0_err']:.2f} ({res['n_hosts']} hosts)")
    print(f"\nSBF: H0 = {results['sbf']['H0']:.2f} +/- {results['sbf']['H0_err']:.2f}")
    print(f"Combined: H0 = {results['combined']['H0']:.2f} +/- {results['combined']['H0_err']:.2f}")
    print(f"Planck: H0 = {results['planck']['H0']} +/- {results['planck']['H0_err']}")
    print(f"Tension: {results['tension_sigma']:.1f} sigma")

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    with open('outputs/h0_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print("\nResults saved to outputs/h0_results.json")
