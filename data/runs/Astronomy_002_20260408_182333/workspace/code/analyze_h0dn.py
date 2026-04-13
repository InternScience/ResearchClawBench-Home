import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'H0DN_MinimalDataset.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

ns = {}
exec(DATA.read_text(), {}, ns)
anchors = ns['anchors']
host_measurements = ns['host_measurements']
sneia_calibrators = ns['sneia_calibrators']
sbf_calibrators = ns['sbf_calibrators']
hubble_flow_sneia = ns['hubble_flow_sneia']
hubble_flow_sbf = ns['hubble_flow_sbf']
method_anchor_err = ns['method_anchor_err']
depth_scatter = ns['depth_scatter']
c_km = ns['c_km']
ln10 = math.log(10.0)

host_rows = []
for host, method, anchor, mu_meas, err_meas in host_measurements:
    extra = method_anchor_err.get((method, anchor), 0.0)
    err_total = math.sqrt(err_meas**2 + anchors[anchor]['err']**2 + extra**2)
    host_rows.append(dict(host=host, method=method, anchor=anchor, mu=mu_meas, err=err_total))
host_df = pd.DataFrame(host_rows)

method_combo = []
for (host, method), g in host_df.groupby(['host', 'method']):
    w = 1.0 / g['err']**2
    mu = np.sum(w * g['mu']) / np.sum(w)
    err = (1.0 / np.sum(w)) ** 0.5
    method_combo.append(dict(host=host, method=method, mu=mu, err=err, n=len(g)))
method_df = pd.DataFrame(method_combo)

host_combo = []
for host, g in method_df.groupby('host'):
    w = 1.0 / g['err']**2
    mu = np.sum(w * g['mu']) / np.sum(w)
    err = (1.0 / np.sum(w)) ** 0.5
    scatter = np.sqrt(np.average((g['mu'] - mu) ** 2, weights=w)) if len(g) > 1 else 0.0
    host_combo.append(dict(host=host, mu=mu, err=err, n_methods=len(g), method_scatter=scatter))
host_dist = pd.DataFrame(host_combo).sort_values('mu')

sn_cal = []
for host, mB, err_mB in sneia_calibrators:
    row = host_dist.loc[host_dist.host == host].iloc[0]
    M = mB - row.mu
    err = math.sqrt(err_mB**2 + row.err**2)
    sn_cal.append(dict(host=host, mB=mB, mu=row.mu, M=M, err=err))
sn_cal_df = pd.DataFrame(sn_cal)
w = 1.0 / sn_cal_df['err']**2
M_sn = np.sum(w * sn_cal_df['M']) / np.sum(w)
M_sn_err = (1.0 / np.sum(w)) ** 0.5

# Infer minimal SBF zero point from approximate Fornax/Virgo group distances.
mu_ngc1316 = float(host_dist.loc[host_dist.host == 'NGC1316', 'mu'].iloc[0])
mu_ngc1365 = float(host_dist.loc[host_dist.host == 'NGC1365', 'mu'].iloc[0])
mu_fornax = np.average([mu_ngc1316, mu_ngc1365], weights=[1 / 0.12**2, 1 / 0.10**2])
mu_fornax_err = math.sqrt(1 / (1 / 0.12**2 + 1 / 0.10**2) + depth_scatter**2)
mu_virgo = mu_fornax + 0.92
mu_virgo_err = math.sqrt(mu_fornax_err**2 + depth_scatter**2)

sbf_rows = []
for host, mbar, err in sbf_calibrators:
    if host in ('NGC1399', 'NGC1404'):
        mu, muerr, group = mu_fornax, mu_fornax_err, 'Fornax'
    else:
        mu, muerr, group = mu_virgo, mu_virgo_err, 'Virgo'
    M = mbar - mu
    terr = math.sqrt(err**2 + muerr**2)
    sbf_rows.append(dict(host=host, group=group, mbar=mbar, mu=mu, M=M, err=terr))
sbf_df = pd.DataFrame(sbf_rows)
w = 1.0 / sbf_df['err']**2
M_sbf = np.sum(w * sbf_df['M']) / np.sum(w)
M_sbf_err = (1.0 / np.sum(w)) ** 0.5

def intercept_fit(flow_rows, mag_key='m'):
    vals, ws = [], []
    rows = []
    for z, m, em, pv in flow_rows:
        x = 5 * math.log10(c_km * z)
        sigma = math.sqrt(em**2 + ((5 / ln10) * (pv / (c_km * z)))**2)
        val = m - x
        vals.append(val)
        ws.append(1 / sigma**2)
        rows.append((z, m, sigma, val))
    a = np.sum(np.array(vals) * np.array(ws)) / np.sum(ws)
    ea = math.sqrt(1 / np.sum(ws))
    return a, ea, rows

def h0_from_intercept(M, eM, a, ea):
    logH = 0.2 * (M + 25 - a)
    H = 10 ** logH
    sigma_logH = 0.2 * math.sqrt(eM**2 + ea**2)
    eH = ln10 * H * sigma_logH
    return H, eH

aB_sn, aB_sn_err, sn_rows = intercept_fit(hubble_flow_sneia)
H0_sn, H0_sn_err = h0_from_intercept(M_sn, M_sn_err, aB_sn, aB_sn_err)

aB_sbf, aB_sbf_err, sbf_rows_int = intercept_fit(hubble_flow_sbf)
H0_sbf, H0_sbf_err = h0_from_intercept(M_sbf, M_sbf_err, aB_sbf, aB_sbf_err)

# Predicted distance moduli for plotting and residual diagnostics.
sn_flow_rows = []
for (z, m, em, pv), (_, _, sigma_tot, a_i) in zip(hubble_flow_sneia, sn_rows):
    mu = m - M_sn
    mu_model = 5 * math.log10(c_km * z / H0_sn) + 25
    sn_flow_rows.append(dict(z=z, m=m, mu=mu, sigma_mu=math.sqrt(em**2 + M_sn_err**2 + ((5/ln10)*(pv/(c_km*z)))**2), mu_model=mu_model, residual=mu-mu_model, H0=H0_sn, sigma_H0=H0_sn_err, indicator='SN Ia'))
sn_flow_df = pd.DataFrame(sn_flow_rows)

sbf_flow_rows = []
for (z, m, em, pv), (_, _, sigma_tot, a_i) in zip(hubble_flow_sbf, sbf_rows_int):
    mu = m - M_sbf
    mu_model = 5 * math.log10(c_km * z / H0_sbf) + 25
    sbf_flow_rows.append(dict(z=z, m=m, mu=mu, sigma_mu=math.sqrt(em**2 + M_sbf_err**2 + ((5/ln10)*(pv/(c_km*z)))**2), mu_model=mu_model, residual=mu-mu_model, H0=H0_sbf, sigma_H0=H0_sbf_err, indicator='SBF'))
sbf_flow_df = pd.DataFrame(sbf_flow_rows)

cov = np.array([
    [H0_sn_err**2, 0.2 * H0_sn_err * H0_sbf_err],
    [0.2 * H0_sn_err * H0_sbf_err, H0_sbf_err**2],
])
vals = np.array([H0_sn, H0_sbf])
inv = np.linalg.inv(cov)
one = np.ones(2)
H0_cons = (one @ inv @ vals) / (one @ inv @ one)
H0_cons_err = math.sqrt(1.0 / (one @ inv @ one))

variants = []
def add_variant(name, H, e):
    variants.append(dict(variant=name, H0=H, err=e))

add_variant('SN Ia only', H0_sn, H0_sn_err)
add_variant('SBF only', H0_sbf, H0_sbf_err)
add_variant('Consensus (rho=0.2)', H0_cons, H0_cons_err)

for anchor_keep in ['N4258', 'LMC']:
    rows = []
    for host, method, anchor, mu_meas, err_meas in host_measurements:
        if method == 'Cepheid' and anchor == anchor_keep:
            extra = method_anchor_err.get((method, anchor), 0.0)
            err_total = math.sqrt(err_meas**2 + anchors[anchor]['err']**2 + extra**2)
            rows.append((host, mu_meas, err_total))
    if rows:
        tmp = pd.DataFrame(rows, columns=['host', 'mu', 'err'])
        cal = []
        wcal = []
        for host, mB, em in sneia_calibrators:
            if host in set(tmp.host):
                r = tmp[tmp.host == host].iloc[0]
                M = mB - r.mu
                e = math.sqrt(em**2 + r.err**2)
                cal.append(M)
                wcal.append(1/e**2)
        if len(cal) >= 3:
            M = np.sum(np.array(cal) * np.array(wcal)) / np.sum(wcal)
            e = math.sqrt(1/np.sum(wcal))
            H, E = h0_from_intercept(M, e, aB_sn, aB_sn_err)
            add_variant(f'SN Ia with {anchor_keep} Cepheid anchor', H, E)

rows = []
for host, method, anchor, mu_meas, err_meas in host_measurements:
    if method == 'TRGB':
        extra = method_anchor_err.get((method, anchor), 0.0)
        err_total = math.sqrt(err_meas**2 + anchors[anchor]['err']**2 + extra**2)
        rows.append((host, mu_meas, err_total))
trgb = pd.DataFrame(rows, columns=['host', 'mu', 'err'])
cal = []
wcal = []
for host, mB, em in sneia_calibrators:
    if host in set(trgb.host):
        r = trgb[trgb.host == host].iloc[0]
        M = mB - r.mu
        e = math.sqrt(em**2 + r.err**2)
        cal.append(M)
        wcal.append(1/e**2)
if len(cal) >= 3:
    M = np.sum(np.array(cal) * np.array(wcal)) / np.sum(wcal)
    e = math.sqrt(1/np.sum(wcal))
    H, E = h0_from_intercept(M, e, aB_sn, aB_sn_err)
    add_variant('SN Ia with TRGB calibration', H, E)

variant_df = pd.DataFrame(variants)
planck = 67.4
planck_err = 0.5
sigma_tension = (H0_cons - planck) / math.sqrt(H0_cons_err**2 + planck_err**2)
reference_baseline = 73.50
reference_baseline_err = 0.81
baseline_pull = (H0_cons - reference_baseline) / math.sqrt(H0_cons_err**2 + reference_baseline_err**2)

summary = {
    'H0_sn': H0_sn, 'H0_sn_err': H0_sn_err,
    'H0_sbf': H0_sbf, 'H0_sbf_err': H0_sbf_err,
    'H0_consensus': H0_cons, 'H0_consensus_err': H0_cons_err,
    'M_sn': M_sn, 'M_sn_err': M_sn_err,
    'M_sbf': M_sbf, 'M_sbf_err': M_sbf_err,
    'aB_sn': aB_sn, 'aB_sn_err': aB_sn_err,
    'aB_sbf': aB_sbf, 'aB_sbf_err': aB_sbf_err,
    'sigma_tension_vs_planck': sigma_tension,
    'baseline_pull_sigma': baseline_pull,
    'planck_H0': planck, 'planck_err': planck_err,
    'reference_baseline': reference_baseline, 'reference_baseline_err': reference_baseline_err,
    'mu_fornax': float(mu_fornax), 'mu_fornax_err': float(mu_fornax_err),
    'mu_virgo': float(mu_virgo), 'mu_virgo_err': float(mu_virgo_err),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
host_df.to_csv(OUT / 'host_measurements_expanded.csv', index=False)
method_df.to_csv(OUT / 'method_level_distances.csv', index=False)
host_dist.to_csv(OUT / 'host_distances.csv', index=False)
sn_cal_df.to_csv(OUT / 'snia_calibrators.csv', index=False)
sn_flow_df.to_csv(OUT / 'snia_hubble_flow.csv', index=False)
sbf_df.to_csv(OUT / 'sbf_calibrators.csv', index=False)
sbf_flow_df.to_csv(OUT / 'sbf_hubble_flow.csv', index=False)
variant_df.to_csv(OUT / 'analysis_variants.csv', index=False)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {'Cepheid': '#1f77b4', 'TRGB': '#ff7f0e'}
for method, g in method_df.groupby('method'):
    ax.errorbar(g['host'], g['mu'], yerr=g['err'], fmt='o', label=method, color=colors.get(method, None), capsize=3)
ax.set_ylabel('Distance modulus $\\mu$ (mag)')
ax.set_xlabel('Host galaxy')
ax.set_title('Primary-indicator host distance moduli')
ax.tick_params(axis='x', rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(IMG / 'distance_ladder_overview.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(sn_flow_df['z'], sn_flow_df['mu'], yerr=sn_flow_df['sigma_mu'], fmt='o', label='SN Ia', capsize=3)
ax.errorbar(sbf_flow_df['z'], sbf_flow_df['mu'], yerr=sbf_flow_df['sigma_mu'], fmt='s', label='SBF', capsize=3)
zgrid = np.linspace(0.02, 0.085, 200)
mu_line = 5 * np.log10(c_km * zgrid / H0_cons) + 25
ax.plot(zgrid, mu_line, '-', color='k', label=f'Consensus H0={H0_cons:.2f}')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Distance modulus $\\mu$ (mag)')
ax.set_title('Hubble-flow diagram for the minimal local distance network')
ax.legend()
fig.tight_layout()
fig.savefig(IMG / 'hubble_flow_diagram.png', dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
y = np.arange(len(variant_df))
ax.errorbar(variant_df['H0'], y, xerr=variant_df['err'], fmt='o', color='#1f77b4', capsize=3)
ax.axvspan(reference_baseline - reference_baseline_err, reference_baseline + reference_baseline_err, color='tab:green', alpha=0.2, label='Paper baseline $73.50\\pm0.81$')
ax.axvspan(planck - planck_err, planck + planck_err, color='tab:red', alpha=0.18, label='Planck $67.4\\pm0.5$')
ax.set_yticks(y)
ax.set_yticklabels(variant_df['variant'])
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_title('Analysis variants and external comparison')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(IMG / 'variant_comparison.png', dpi=200)
plt.close(fig)

print(json.dumps(summary, indent=2))
