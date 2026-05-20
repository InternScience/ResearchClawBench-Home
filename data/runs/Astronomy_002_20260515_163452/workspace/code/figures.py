#!/usr/bin/env python3
"""Generate all figures for the H0 Distance Network report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import norm
import json
import os

# Load results and data
with open('outputs/results.json') as f:
    R = json.load(f)

data = {}
with open('data/H0DN_MinimalDataset.txt') as f:
    exec(f.read(), data)

os.makedirs('report/images', exist_ok=True)

C = {
    'cep': '#1976D2', 'trgb': '#F57C00', 'sbf': '#388E3C',
    'sn': '#C62828', 'anc': '#6A1B9A', 'plk': '#5D4037', 'flow': '#546E7A',
}
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'serif',
})

H0b = R['baseline']['H0']; eH0b = R['baseline']['err_H0']
MBb = R['baseline']['MB']; eMBb = R['baseline']['err_MB']
H0p = R['planck']['H0']; eH0p = R['planck']['err_H0']
c_km = data['c_km']

# ============================================================
# Figure 1: Distance Ladder Schematic
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.5, 7); ax.set_ylim(-1.5, 3.8); ax.axis('off')
ax.text(3.25, 3.5, 'The Local Distance Network', fontsize=16, fontweight='bold', ha='center')

rung_boxes = [
    (0.75, 'Geometric Anchors', ['NGC 4258 (Masers)', 'LMC (DEBs)', 'MW (Parallaxes)'], '#E8EAF6', '#3F51B5'),
    (2.75, 'Primary Indicators', ['Cepheids (P-L)', 'TRGB (Edge)', 'Miras (P-L)'], '#E8F5E9', '#2E7D32'),
    (4.75, 'Secondary Indicators', ['SNe Ia (Std Candle)', 'SBF (Fluctuations)', 'TF/FP (Empirical)'], '#FFF3E0', '#E65100'),
    (6.5, 'Hubble Flow', ['z > 0.01 SNe Ia', 'z > 0.02 SBF'], '#FCE4EC', '#AD1457'),
]
for x, title, items, bg, tc in rung_boxes:
    r = mpatches.FancyBboxPatch((x-0.9, 0.5), 1.8, 2.3, boxstyle='round,pad=0.1',
                                 facecolor=bg, edgecolor=tc, lw=2)
    ax.add_patch(r)
    ax.text(x, 2.6, title, fontsize=10, fontweight='bold', ha='center', color=tc)
    for j, item in enumerate(items):
        ax.text(x, 2.1-j*0.5, item, fontsize=8, ha='center', color='#333')

for x1, x2 in [(1.65, 1.85), (3.65, 3.85), (5.65, 5.85)]:
    ax.annotate('', xy=(x2, 1.65), xytext=(x1, 1.65),
               arrowprops=dict(arrowstyle='->', color='#555', lw=2))

rect_h0 = mpatches.FancyBboxPatch((1.25, -1.2), 4.0, 0.8, boxstyle='round,pad=0.1',
                                   facecolor='#E3F2FD', edgecolor='#1565C0', lw=2)
ax.add_patch(rect_h0)
ax.text(3.25, -0.8, f'$H_0 = {H0b:.2f} \\pm {eH0b:.2f}$ km s$^{{-1}}$ Mpc$^{{-1}}$',
        fontsize=13, fontweight='bold', ha='center', color='#1565C0')
ax.annotate('', xy=(3.25, -0.4), xytext=(5.5, 0.5),
           arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2, connectionstyle='arc3,rad=-0.15'))
ax.text(4.8, -0.05, 'GLS\nFit', fontsize=9, ha='center', color='#1565C0', fontweight='bold')
plt.tight_layout()
fig.savefig('report/images/figure1_distance_ladder.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 1 done")

# ============================================================
# Figure 2: H₀ Posterior and Variant Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: posterior
ax = axes[0]
x = np.linspace(H0b - 6*eH0b, max(H0b + 6*eH0b, H0p + 6*eH0p), 500)
ax.fill_between(x, norm.pdf(x, H0b, eH0b), alpha=0.3, color=C['cep'])
ax.plot(x, norm.pdf(x, H0b, eH0b), color=C['cep'], lw=2)
ax.axvline(H0b, color=C['cep'], ls='--', lw=1.5, label=f'Local: ${H0b:.1f} \\pm {eH0b:.1f}$')
ax.fill_between(x, norm.pdf(x, H0p, eH0p), alpha=0.2, color=C['plk'])
ax.plot(x, norm.pdf(x, H0p, eH0p), color=C['plk'], lw=2)
ax.axvline(H0p, color=C['plk'], ls='--', lw=1.5, label=f'Planck: ${H0p:.1f} \\pm {eH0p:.1f}$')
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_ylabel('Probability Density')
ax.set_title('(a) $H_0$ Posterior Distributions')
ax.legend(fontsize=9)

# Panel B: variant comparison
ax = axes[1]
vars_all = list(R['variants'].keys()) + ['SBF pathway']
H0s = [R['variants'][k]['H0'] for k in R['variants']] + [R['sbf']['H0']]
errs = [R['variants'][k]['err_H0'] for k in R['variants']] + [R['sbf']['err_H0']]
cols = [C['cep'], C['trgb'], C['anc'], '#FF5722', '#607D8B', C['sbf']]
y_pos = np.arange(len(vars_all))
ax.errorbar(H0s, y_pos, xerr=errs, fmt='o', color='black', ms=8, capsize=5, elinewidth=1.5, zorder=3)
for i, (h0, c) in enumerate(zip(H0s, cols[:len(vars_all)])):
    ax.plot(h0, i, 'o', color=c, ms=10, zorder=4)
ax.axvline(H0p, color=C['plk'], ls=':', lw=1.5, alpha=0.7, label='Planck')
ax.axvline(H0b, color=C['cep'], ls='--', lw=1.5, alpha=0.7, label='Baseline')
ax.set_yticks(y_pos); ax.set_yticklabels(vars_all, fontsize=10)
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_title('(b) Analysis Variants')
ax.legend(fontsize=8, loc='lower right'); ax.invert_yaxis()
plt.tight_layout()
fig.savefig('report/images/figure2_h0_posterior.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 2 done")

# ============================================================
# Figure 3: Hubble Diagram
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: mB vs z
ax = axes[0]
for c in R['calibrators']:
    d_cal = 10**((c['mu'] - 25) / 5.0)
    z_cal = d_cal * H0b / c_km
    col = C['cep'] if c['method'] == 'Cepheid' else C['trgb']
    ax.errorbar(z_cal, c['mB'], xerr=z_cal*0.15, yerr=c['err_mB'], fmt='s',
               color=col, ms=8, capsize=3, label=f"{c['host']} ({c['method']})", zorder=4)

zf = np.array(R['z_flow']); mBf = np.array(R['mB_flow']); ef = np.array(R['total_err_flow'])
ax.errorbar(zf, mBf, yerr=ef, fmt='D', color=C['sn'], ms=7, capsize=3, label='Flow SNe Ia', zorder=4)

zm = np.linspace(0.002, 0.15, 200)
mu_m = 5.0*np.log10(c_km*zm/H0b)+25.0
mB_m = MBb + mu_m
ax.plot(zm, mB_m, 'k-', lw=2, alpha=0.5, label=f'GLS model ($H_0={H0b:.1f}$)')
ax.set_xscale('log'); ax.set_xlabel('Redshift $z$'); ax.set_ylabel('Apparent Magnitude $m_B$')
ax.set_title('(a) Hubble Diagram')
ax.legend(fontsize=6.5, loc='lower right', ncol=2); ax.set_xlim(0.001, 0.15)

# Panel b: mu vs z
ax = axes[1]
for c in R['calibrators']:
    d_cal = 10**((c['mu']-25)/5.0); z_cal = d_cal*H0b/c_km
    col = C['cep'] if c['method']=='Cepheid' else C['trgb']
    ax.errorbar(z_cal, c['mu'], yerr=c['err_mu'], fmt='s', color=col, ms=8, capsize=3, zorder=4)

mu_flow = mBf - MBb
err_mu_flow = np.sqrt(ef**2 + eMBb**2)
ax.errorbar(zf, mu_flow, yerr=err_mu_flow, fmt='D', color=C['sn'], ms=7, capsize=3, zorder=4)
ax.plot(zm, mu_m, 'k-', lw=2, alpha=0.5)
ax.set_xscale('log'); ax.set_xlabel('Redshift $z$'); ax.set_ylabel('Distance Modulus $\\mu$ (mag)')
ax.set_title('(b) Distance-Redshift Relation'); ax.set_xlim(0.001, 0.15)
leg = [Line2D([0],[0], marker='s', color='w', mfc=C['cep'], ms=8, label='Cepheid'),
       Line2D([0],[0], marker='s', color='w', mfc=C['trgb'], ms=8, label='TRGB'),
       Line2D([0],[0], marker='D', color='w', mfc=C['sn'], ms=7, label='SNe Ia (flow)'),
       Line2D([0],[0], color='k', lw=2, alpha=0.5, label='GLS model')]
ax.legend(handles=leg, fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig('report/images/figure3_hubble_diagram.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 3 done")

# ============================================================
# Figure 4: Calibrator MB and Residuals
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: MB per calibrator
ax = axes[0]
hosts = [c['host'] for c in R['calibrators']]
MBs = [c['MB'] for c in R['calibrators']]
eMBs = [c['err_MB'] for c in R['calibrators']]
methods = [c['method'] for c in R['calibrators']]
cols_c = [C['cep'] if m=='Cepheid' else C['trgb'] for m in methods]
yp = np.arange(len(hosts))
ax.errorbar(MBs, yp, xerr=eMBs, fmt='o', color='black', ms=6, capsize=4, elinewidth=1.5, zorder=3)
for i, (mb, c) in enumerate(zip(MBs, cols_c)):
    ax.plot(mb, i, 'o', color=c, ms=9, zorder=4)
ax.axvline(MBb, color='gray', ls='--', lw=1.5, label=f'Mean: ${MBb:.3f}$')
ax.set_yticks(yp); ax.set_yticklabels([f"{h} ({m})" for h, m in zip(hosts, methods)], fontsize=9)
ax.set_xlabel('$M_B$ (mag)'); ax.set_title('(a) SNe Ia Absolute Magnitudes')
ax.legend(fontsize=9); ax.invert_yaxis()

# Panel b: Residuals
ax = axes[1]
res = R['residuals']
n_cal = len(R['calibrators'])
res_cal = res[:n_cal]; res_flow = res[n_cal:]
labs_cal = [c['host'] for c in R['calibrators']]
labs_flow = [f'z={z:.3f}' for z in R['z_flow']]
all_labs = labs_cal + labs_flow; all_res = res_cal + res_flow
cols_r = [C['cep'] if c['method']=='Cepheid' else C['trgb'] for c in R['calibrators']] + [C['sn']]*len(res_flow)
yp = np.arange(len(all_res))
ax.barh(yp, all_res, color=cols_r, alpha=0.7, height=0.6)
ax.axvline(0, color='black', lw=1)
ax.axhline(n_cal-0.5, color='gray', ls=':', lw=1)
ax.set_yticks(yp); ax.set_yticklabels(all_labs, fontsize=8)
ax.set_xlabel('Residual (mag)'); ax.set_title('(b) GLS Fit Residuals'); ax.invert_yaxis()
plt.tight_layout()
fig.savefig('report/images/figure4_calibrators.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 4 done")

# ============================================================
# Figure 5: Tension & Method Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel a: all H0 values
ax = axes[0]
all_h0 = [('Distance Network\n(Baseline)', H0b, eH0b, C['cep'])]
for vn, vd in R['variants'].items():
    if vn != 'Combined (baseline)':
        short = vn.replace(' only', '\nonly').replace(' anchor', '\nanchor')
        all_h0.append((short, vd['H0'], vd['err_H0'], '#888'))
all_h0.append(('SBF Pathway', R['sbf']['H0'], R['sbf']['err_H0'], C['sbf']))
all_h0.append(('Planck CMB\n(ΛCDM)', H0p, eH0p, C['plk']))
yp = np.arange(len(all_h0))
for i, (lab, h0, e, c) in enumerate(all_h0):
    ax.errorbar(h0, i, xerr=e, fmt='o', color=c, ms=10, capsize=5, elinewidth=2, zorder=4)
ax.set_yticks(yp); ax.set_yticklabels([h[0] for h in all_h0], fontsize=9)
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_title('(a) All $H_0$ Measurements'); ax.invert_yaxis()

sig = R['tension']['significance_sigma']
ax.annotate(f'Tension: ${sig:.1f}\\sigma$',
           xy=(H0b, 0), xytext=(H0p+10, 2),
           fontsize=11, ha='center', color='red', fontweight='bold',
           arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

# Panel b: bar chart
ax = axes[1]
labels = list(R['variants'].keys()) + ['SBF']
vals = [R['variants'][k]['H0'] for k in R['variants']] + [R['sbf']['H0']]
errs = [R['variants'][k]['err_H0'] for k in R['variants']] + [R['sbf']['err_H0']]
bcols = [C['cep'], C['trgb'], C['anc'], '#FF5722', '#607D8B', C['sbf']][:len(labels)]
ax.barh(range(len(labels)), vals, xerr=errs, color=bcols, alpha=0.7, capsize=4, edgecolor='white', lw=1.5)
ax.axvline(H0p, color=C['plk'], ls='--', lw=2, label=f'Planck: {H0p}')
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_title('(b) $H_0$ by Method / Anchor'); ax.legend(fontsize=9); ax.invert_yaxis()
plt.tight_layout()
fig.savefig('report/images/figure5_tension.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 5 done")

# ============================================================
# Figure 6: Anchor Distances & Host Moduli Heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: anchors
ax = axes[0]
anames = ['NGC 4258\n(Masers)', 'LMC\n(DEBs)']
amus = [data['anchors']['N4258']['mu'], data['anchors']['LMC']['mu']]
ad = [10**((mu-25)/5.0) for mu in amus]
ad_errs = [d * err * np.log(10)/5.0 for d, err in zip(ad, [data['anchors']['N4258']['err'], data['anchors']['LMC']['err']])]
bars = ax.bar(range(2), ad, color=[C['anc'], '#7B1FA2'], alpha=0.7, edgecolor='white', lw=2)
ax.errorbar(range(2), ad, yerr=ad_errs, fmt='none', color='black', capsize=5, elinewidth=1.5)
ax.set_xticks(range(2)); ax.set_xticklabels(anames, fontsize=10)
ax.set_ylabel('Distance (Mpc)'); ax.set_title('(a) Geometric Anchor Distances')
for i, (d, mu) in enumerate(zip(ad, amus)):
    ax.text(i, d + ad_errs[i] + 0.3, f'$\\mu = {mu:.3f}$\n$d = {d:.2f}$ Mpc',
           ha='center', va='bottom', fontsize=9)

# Panel b: heatmap of host mus
ax = axes[1]
hosts_set = set()
for k in R['host_mus']:
    parts = k.split('_')
    hosts_set.add('_'.join(parts[:-1]))
hosts_list = sorted(hosts_set)
meths = ['Cepheid', 'TRGB']
mat = np.full((len(hosts_list), len(meths)), np.nan)
for k, v in R['host_mus'].items():
    parts = k.split('_')
    # host may contain underscores, method is the last part
    method = parts[-1]
    host = '_'.join(parts[:-1])
    if method in meths:
        i = hosts_list.index(host)
        j = meths.index(method)
        mat[i, j] = v['mu']
im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
for i in range(len(hosts_list)):
    for j in range(len(meths)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='gray')
ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths, fontsize=10)
ax.set_yticks(range(len(hosts_list))); ax.set_yticklabels(hosts_list, fontsize=9)
ax.set_title('(b) Host Distance Moduli ($\\mu$)')
plt.colorbar(im, ax=ax, label='$\\mu$ (mag)', shrink=0.8)
plt.tight_layout()
fig.savefig('report/images/figure6_anchors.png', dpi=150, bbox_inches='tight')
plt.close(); print("Fig 6 done")

print("\nAll figures generated successfully!")
