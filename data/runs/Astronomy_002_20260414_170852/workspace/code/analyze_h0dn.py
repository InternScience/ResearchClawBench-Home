import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'H0DN_MinimalDataset.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def load_dataset():
    ns = {}
    exec(DATA.read_text(), {}, ns)
    return ns


def gls_host_distances(ns):
    anchors = ns['anchors']
    host_measurements = ns['host_measurements']
    method_anchor_err = ns['method_anchor_err']
    by_host = {}
    for host, method, anchor, mu, err in host_measurements:
        by_host.setdefault(host, []).append((method, anchor, mu, err))

    results = {}
    rows_out = []
    for host, rows in by_host.items():
        y = np.array([r[2] for r in rows], float)
        n = len(rows)
        C = np.zeros((n, n), float)
        for i, (m1, a1, mu1, e1) in enumerate(rows):
            C[i, i] = e1**2 + ns['anchors'][a1]['err']**2 + method_anchor_err.get((m1, a1), 0.0) ** 2
            for j, (m2, a2, mu2, e2) in enumerate(rows):
                if i >= j:
                    continue
                cov = 0.0
                if a1 == a2:
                    cov += anchors[a1]['err'] ** 2
                if (m1, a1) == (m2, a2):
                    cov += method_anchor_err.get((m1, a1), 0.0) ** 2
                C[i, j] = C[j, i] = cov
        iC = np.linalg.inv(C)
        one = np.ones(n)
        mu_hat = float((one @ iC @ y) / (one @ iC @ one))
        err_hat = float(np.sqrt(1.0 / (one @ iC @ one)))
        results[host] = {'mu': mu_hat, 'err': err_hat, 'n_meas': n}
        rows_out.append({'host': host, 'mu': mu_hat, 'err': err_hat, 'n_meas': n})
    return results, rows_out


def weighted_mean(values, variances):
    values = np.asarray(values, float)
    variances = np.asarray(variances, float)
    w = 1.0 / variances
    mean = float(np.sum(w * values) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    return mean, err


def fit_h0(flow_rows, M, M_err, c_km):
    z = np.array([r[0] for r in flow_rows], float)
    m = np.array([r[1] for r in flow_rows], float)
    em = np.array([r[2] for r in flow_rows], float)
    vp = np.array([r[3] for r in flow_rows], float)
    total_err = np.sqrt(em**2 + (5 / np.log(10) * vp / (c_km * z)) ** 2)

    def chi2(logH0):
        H0 = math.exp(logH0)
        pred = M + 5 * np.log10(c_km * z / H0) + 25
        return float(np.sum((m - pred) ** 2 / total_err**2))

    res = minimize_scalar(chi2, bounds=(math.log(40), math.log(120)), method='bounded')
    H0 = float(math.exp(res.x))
    eps = 1e-4
    second = (chi2(res.x + eps) - 2 * chi2(res.x) + chi2(res.x - eps)) / eps**2
    sigma_log_flow = float(np.sqrt(2 / second)) if second > 0 else float('nan')
    sigma_log_M = float(np.log(10) / 5 * M_err)
    sigma_log = float(np.sqrt(sigma_log_flow**2 + sigma_log_M**2))
    H0_err = float(H0 * sigma_log)
    pred = M + 5 * np.log10(c_km * z / H0) + 25
    return {
        'H0': H0,
        'H0_err': H0_err,
        'z': z.tolist(),
        'm': m.tolist(),
        'm_err': em.tolist(),
        'tot_err': total_err.tolist(),
        'pred_m': pred.tolist(),
        'residuals_mag': (m - pred).tolist(),
    }


def run_analysis():
    ns = load_dataset()
    host_results, host_rows = gls_host_distances(ns)

    Mb_vals = []
    Mb_vars = []
    for host, mB, err_mB in ns['sneia_calibrators']:
        Mb_vals.append(mB - host_results[host]['mu'])
        Mb_vars.append(err_mB**2 + host_results[host]['err']**2)
    M_B, M_B_err = weighted_mean(Mb_vals, Mb_vars)
    sne = fit_h0(ns['hubble_flow_sneia'], M_B, M_B_err, ns['c_km'])
    sne['M'] = M_B
    sne['M_err'] = M_B_err

    mu_group = {'Fornax': 31.51, 'Virgo': 31.09}
    mu_group_err = {'Fornax': 0.03, 'Virgo': 0.03}
    sbf_groups = {}
    for host, m, err in ns['sbf_calibrators']:
        sbf_groups.setdefault(ns['host_group'][host], []).append((host, m, err))

    group_rows = []
    Msbf_vals = []
    Msbf_vars = []
    for group, rows in sbf_groups.items():
        mags = np.array([r[1] for r in rows], float)
        errs = np.array([r[2] for r in rows], float)
        var = errs**2 + ns['depth_scatter']**2
        mbar, mbar_err = weighted_mean(mags, var)
        group_rows.append({'group': group, 'mbar': mbar, 'mbar_err': mbar_err, 'mu_group': mu_group[group], 'mu_group_err': mu_group_err[group]})
        Msbf_vals.append(mbar - mu_group[group])
        Msbf_vars.append(mbar_err**2 + mu_group_err[group]**2)
    M_sbf, M_sbf_err = weighted_mean(Msbf_vals, Msbf_vars)
    sbf = fit_h0(ns['hubble_flow_sbf'], M_sbf, M_sbf_err, ns['c_km'])
    sbf['M'] = M_sbf
    sbf['M_err'] = M_sbf_err

    channels = {'SNe Ia': sne, 'SBF': sbf}
    ws = np.array([1 / sne['H0_err'] ** 2, 1 / sbf['H0_err'] ** 2])
    H0_cons = float(np.sum(ws * np.array([sne['H0'], sbf['H0']])) / np.sum(ws))
    H0_cons_err = float(np.sqrt(1 / np.sum(ws)))

    variants = []
    # Variant 1: ignore covariance among repeated host measurements
    ns2 = load_dataset()
    anchors = ns2['anchors']
    method_anchor_err = ns2['method_anchor_err']
    by_host = {}
    for host, method, anchor, mu, err in ns2['host_measurements']:
        by_host.setdefault(host, []).append((method, anchor, mu, err))
    hr_nocov = {}
    for host, rows in by_host.items():
        vals = np.array([r[2] for r in rows], float)
        variances = np.array([r[3] ** 2 + anchors[r[1]]['err'] ** 2 + method_anchor_err.get((r[0], r[1]), 0.0) ** 2 for r in rows], float)
        mu, err = weighted_mean(vals, variances)
        hr_nocov[host] = {'mu': mu, 'err': err}
    vals = [m - hr_nocov[h]['mu'] for h, m, e in ns2['sneia_calibrators']]
    vars_ = [e**2 + hr_nocov[h]['err']**2 for h, m, e in ns2['sneia_calibrators']]
    M2, M2_err = weighted_mean(vals, vars_)
    variants.append({'variant': 'ignore_host_covariance', 'channel': 'SNe Ia', 'H0': fit_h0(ns2['hubble_flow_sneia'], M2, M2_err, ns2['c_km'])['H0']})

    # Variant 2: N4258-only primary anchor subset
    subset = [r for r in ns2['host_measurements'] if not (r[1] == 'Cepheid' and r[2] == 'LMC')]
    by_host2 = {}
    for host, method, anchor, mu, err in subset:
        by_host2.setdefault(host, []).append((method, anchor, mu, err))
    hr2 = {}
    for host, rows in by_host2.items():
        y = np.array([r[2] for r in rows], float)
        C = np.zeros((len(rows), len(rows)), float)
        for i, (m1, a1, mu1, e1) in enumerate(rows):
            C[i, i] = e1**2 + anchors[a1]['err']**2 + method_anchor_err.get((m1, a1), 0.0) ** 2
            for j, (m2, a2, mu2, e2) in enumerate(rows):
                if i >= j:
                    continue
                cov = (anchors[a1]['err']**2 if a1 == a2 else 0.0) + (method_anchor_err.get((m1, a1), 0.0) ** 2 if (m1, a1) == (m2, a2) else 0.0)
                C[i, j] = C[j, i] = cov
        iC = np.linalg.inv(C)
        one = np.ones(len(rows))
        hr2[host] = {'mu': float((one @ iC @ y) / (one @ iC @ one)), 'err': float(np.sqrt(1.0 / (one @ iC @ one)))}
    vals = [m - hr2[h]['mu'] for h, m, e in ns2['sneia_calibrators']]
    vars_ = [e**2 + hr2[h]['err']**2 for h, m, e in ns2['sneia_calibrators']]
    M3, M3_err = weighted_mean(vals, vars_)
    variants.append({'variant': 'N4258_only_primary_anchor', 'channel': 'SNe Ia', 'H0': fit_h0(ns2['hubble_flow_sneia'], M3, M3_err, ns2['c_km'])['H0']})

    # Variant 3: no SBF group depth scatter
    Msbf_vals2 = []
    Msbf_vars2 = []
    for group, rows in sbf_groups.items():
        mags = np.array([r[1] for r in rows], float)
        errs = np.array([r[2] for r in rows], float)
        mbar, mbar_err = weighted_mean(mags, errs**2)
        Msbf_vals2.append(mbar - mu_group[group])
        Msbf_vars2.append(mbar_err**2 + mu_group_err[group]**2)
    M4, M4_err = weighted_mean(Msbf_vals2, Msbf_vars2)
    variants.append({'variant': 'no_group_depth_scatter', 'channel': 'SBF', 'H0': fit_h0(ns2['hubble_flow_sbf'], M4, M4_err, ns2['c_km'])['H0']})

    results = {
        'host_results': host_rows,
        'group_rows': group_rows,
        'channel_results': channels,
        'consensus': {'H0': H0_cons, 'H0_err': H0_cons_err},
        'baseline_reference': {'H0': 73.50, 'err': 0.81},
        'planck_reference': {'H0': 67.4, 'err': 0.5, 'label': 'Representative early-universe CMB value'},
        'variants': variants,
    }
    return results


def make_figures(results):
    # Figure 1: host distances overview
    hosts = [r['host'] for r in results['host_results']]
    mu = [r['mu'] for r in results['host_results']]
    err = [r['err'] for r in results['host_results']]
    n = [r['n_meas'] for r in results['host_results']]
    order = np.argsort(mu)
    plt.figure(figsize=(10, 6))
    plt.errorbar(np.array(mu)[order], np.arange(len(hosts)), xerr=np.array(err)[order], fmt='o', color='tab:blue', capsize=3)
    plt.yticks(np.arange(len(hosts)), np.array(hosts)[order])
    for yi, ni in zip(np.arange(len(hosts)), np.array(n)[order]):
        plt.text(np.array(mu)[order][yi] + np.array(err)[order][yi] + 0.03, yi, f'n={ni}', va='center', fontsize=10)
    plt.xlabel('Distance modulus $\\mu$ (mag)')
    plt.ylabel('SN host')
    plt.title('Covariance-weighted host distance estimates from primary indicators')
    plt.tight_layout()
    plt.savefig(IMG / 'host_distance_overview.png', dpi=200)
    plt.close()

    # Figure 2: Hubble diagrams
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, channel, color in zip(axes, ['SNe Ia', 'SBF'], ['tab:red', 'tab:green']):
        r = results['channel_results'][channel]
        z = np.array(r['z'])
        ax.errorbar(z, r['m'], yerr=r['tot_err'], fmt='o', color=color, capsize=3, label='flow data')
        zgrid = np.linspace(z.min() * 0.9, z.max() * 1.05, 200)
        pred = r['M'] + 5 * np.log10(299792.458 * zgrid / r['H0']) + 25
        ax.plot(zgrid, pred, color='black', lw=2, label=f'best fit H0={r["H0"]:.1f}')
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Apparent magnitude')
        ax.set_title(channel)
        ax.legend(frameon=True, fontsize=10)
    fig.suptitle('Minimal-dataset Hubble diagrams')
    fig.tight_layout()
    fig.savefig(IMG / 'hubble_diagrams.png', dpi=200)
    plt.close(fig)

    # Figure 3: channel comparison
    labels = ['Baseline ref.', 'Planck ref.', 'SNe Ia', 'SBF', 'Consensus']
    vals = [results['baseline_reference']['H0'], results['planck_reference']['H0'], results['channel_results']['SNe Ia']['H0'], results['channel_results']['SBF']['H0'], results['consensus']['H0']]
    errs = [results['baseline_reference']['err'], results['planck_reference']['err'], results['channel_results']['SNe Ia']['H0_err'], results['channel_results']['SBF']['H0_err'], results['consensus']['H0_err']]
    colors = ['gray', 'black', 'tab:red', 'tab:green', 'tab:blue']
    plt.figure(figsize=(8, 5.5))
    ypos = np.arange(len(labels))
    for y, v, e, c in zip(ypos, vals, errs, colors):
        plt.errorbar(v, y, xerr=e, fmt='o', color=c, capsize=4, markersize=7)
    plt.yticks(ypos, labels)
    plt.xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    plt.title('Channel-specific and reference Hubble constant values')
    plt.tight_layout()
    plt.savefig(IMG / 'h0_channel_comparison.png', dpi=200)
    plt.close()

    # Figure 4: sensitivity analysis
    var_names = [v['variant'] for v in results['variants']]
    var_vals = [v['H0'] for v in results['variants']]
    base = [results['channel_results'][v['channel']]['H0'] for v in results['variants']]
    delta = np.array(var_vals) - np.array(base)
    plt.figure(figsize=(9, 5.5))
    colors = ['tab:red' if v['channel'] == 'SNe Ia' else 'tab:green' for v in results['variants']]
    plt.axvline(0, color='black', lw=1)
    plt.barh(var_names, delta, color=colors)
    plt.xlabel('Change in inferred $H_0$ relative to channel baseline')
    plt.title('Sensitivity of the minimal reconstruction to analysis choices')
    plt.tight_layout()
    plt.savefig(IMG / 'sensitivity_variants.png', dpi=200)
    plt.close()


def main():
    results = run_analysis()
    (OUT / 'channel_results.json').write_text(json.dumps(results, indent=2))
    (OUT / 'variant_results.json').write_text(json.dumps(results['variants'], indent=2))

    claim_recovery = [
        {
            'claim': 'The minimal-dataset reconstruction does not reproduce the stated baseline H0=73.50±0.81 km s^-1 Mpc^-1.',
            'supporting_artifact': 'outputs/channel_results.json::consensus and baseline_reference',
        },
        {
            'claim': 'The SNe Ia channel dominates the combined precision in the minimal reconstruction.',
            'supporting_artifact': 'outputs/channel_results.json::channel_results.SNe Ia and consensus',
        },
        {
            'claim': 'Variant choices shift the SNe Ia result at the few km s^-1 Mpc^-1 level, while the SBF depth-scatter choice has negligible effect here.',
            'supporting_artifact': 'outputs/variant_results.json and report/images/sensitivity_variants.png',
        },
    ]
    (OUT / 'claim_recovery_table.json').write_text(json.dumps(claim_recovery, indent=2))
    make_figures(results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
