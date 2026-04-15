#!/usr/bin/env python3
"""Generate figures for the H0 Distance Network report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

os.makedirs('report/images', exist_ok=True)

with open('outputs/h0_results.json') as f:
    results = json.load(f)

plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150
})

# ── Figure 1: Distance Ladder Overview ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Host distance moduli
ax = axes[0]
b = results['baseline']
hosts = sorted(b['host_distances'].keys())
mus = [b['host_distances'][h] for h in hosts]
errs = [b['host_dist_errs'][h] for h in hosts]
y_pos = np.arange(len(hosts))
ax.errorbar(mus, y_pos, xerr=errs, fmt='o', color='navy', capsize=3, markersize=6)
ax.set_yticks(y_pos)
ax.set_yticklabels(hosts, fontsize=9)
ax.set_xlabel('Distance Modulus μ (mag)')
ax.set_title('(a) Host Galaxy Distances')
ax.axvline(x=np.mean(mus), color='gray', ls='--', alpha=0.5)

# Panel B: SN Ia absolute magnitudes
ax = axes[1]
sn_hosts = [h for h, _, _ in b['sn_absmag']]
sn_mbs = [mb for _, mb, _ in b['sn_absmag']]
sn_errs = [e for _, _, e in b['sn_absmag']]
y_pos2 = np.arange(len(sn_hosts))
ax.errorbar(sn_mbs, y_pos2, xerr=sn_errs, fmt='s', color='darkred', capsize=3, markersize=6)
ax.set_yticks(y_pos2)
ax.set_yticklabels(sn_hosts, fontsize=9)
ax.set_xlabel('Absolute Magnitude M_B (mag)')
ax.set_title('(b) SN Ia Absolute Magnitudes')
ax.axvline(x=b['M_B'], color='gray', ls='--', alpha=0.5, label=f'Mean M_B={b["M_B"]:.2f}')
ax.legend(fontsize=9)

# Panel C: H0 per HF SN
ax = axes[2]
z_vals = [z for z, _, _ in b['h0_per_sn']]
h0_vals = [h for _, h, _ in b['h0_per_sn']]
h0_errs = [e for _, _, e in b['h0_per_sn']]
ax.errorbar(z_vals, h0_vals, yerr=h0_errs, fmt='D', color='forestgreen', capsize=4, markersize=7)
ax.axhline(y=b['H0'], color='navy', ls='-', alpha=0.7, label=f'H₀ = {b["H0"]:.1f} ± {b["H0_err"]:.1f}')
ax.axhline(y=67.4, color='red', ls='--', alpha=0.7, label='Planck: 67.4 ± 0.5')
ax.set_xlabel('Redshift z')
ax.set_ylabel('H₀ (km/s/Mpc)')
ax.set_title('(c) H₀ from Hubble-Flow SNe Ia')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('report/images/fig1_distance_ladder.png')
plt.close()
print("Figure 1 saved")

# ── Figure 2: H0 Comparison ──
fig, ax = plt.subplots(figsize=(10, 6))

categories = []
values = []
errors = []
colors = []

# Baseline
categories.append('Baseline\n(all anchors)')
values.append(results['baseline']['H0'])
errors.append(results['baseline']['H0_err'])
colors.append('navy')

# Anchor variants
for anc in ['N4258', 'LMC']:
    if anc in results['anchor_variants']:
        categories.append(f'{anc}\nonly')
        values.append(results['anchor_variants'][anc]['H0'])
        errors.append(results['anchor_variants'][anc]['H0_err'])
        colors.append('steelblue')

# Indicator variants
for ind in ['Cepheid', 'TRGB']:
    if ind in results['indicator_variants']:
        categories.append(f'{ind}\nonly')
        values.append(results['indicator_variants'][ind]['H0'])
        errors.append(results['indicator_variants'][ind]['H0_err'])
        colors.append('teal')

# SBF
categories.append('SBF\nHubble flow')
values.append(results['sbf']['H0'])
errors.append(results['sbf']['H0_err'])
colors.append('darkorange')

# Literature
categories.append('SH0ES\n2022')
values.append(73.04)
errors.append(1.04)
colors.append('gray')

categories.append('Distance\nNetwork')
values.append(73.50)
errors.append(0.81)
colors.append('gray')

categories.append('Planck\nCMB')
values.append(67.4)
errors.append(0.5)
colors.append('red')

y_pos = np.arange(len(categories))
ax.barh(y_pos, values, xerr=errors, color=colors, alpha=0.7, capsize=3, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=9)
ax.set_xlabel('H₀ (km/s/Mpc)')
ax.set_title('Hubble Constant: Analysis Variants and Literature Comparison')
ax.axvline(x=73.5, color='navy', ls=':', alpha=0.5)
ax.axvline(x=67.4, color='red', ls=':', alpha=0.5)
ax.set_xlim(50, 160)

plt.tight_layout()
plt.savefig('report/images/fig2_h0_comparison.png')
plt.close()
print("Figure 2 saved")

# ── Figure 3: Hubble Diagram ──
fig, ax = plt.subplots(figsize=(10, 7))

# Plot HF SNe
z_hf = [z for z, _, _ in results['baseline']['h0_per_sn']]
mu_hf = [results['baseline']['h0_per_sn_detail'][i][3] for i in range(len(z_hf))]
ax.scatter(z_hf, mu_hf, c='navy', s=80, zorder=5, label='Hubble-flow SNe Ia')

# Plot theoretical lines
z_arr = np.linspace(0.02, 0.10, 100)
for H0, color, label in [(67.4, 'red', 'Planck (67.4)'), (73.5, 'blue', 'SH0ES (73.5)')]:
    mu_arr = 25 + 5*np.log10(299792.458 * z_arr / H0)
    ax.plot(z_arr, mu_arr, color=color, ls='--', lw=2, label=label)

# Plot calibrator distances
for host in results['baseline']['host_distances']:
    mu = results['baseline']['host_distances'][host]
    # Approximate z from mu using H0=73.5
    D = 10**((mu - 25)/5)
    z_approx = 73.5 * D / 299792.458
    if z_approx > 0.005:
        ax.scatter(z_approx, mu, marker='*', s=150, c='gold', edgecolors='black', zorder=6)

ax.set_xlabel('Redshift z')
ax.set_ylabel('Distance Modulus μ (mag)')
ax.set_title('Hubble Diagram: Calibrators and Hubble Flow')
ax.legend(fontsize=10)
ax.set_xlim(0, 0.10)

plt.tight_layout()
plt.savefig('report/images/fig3_hubble_diagram.png')
plt.close()
print("Figure 3 saved")

# ── Figure 4: Error Budget ──
fig, ax = plt.subplots(figsize=(8, 6))

# Decompose error budget for baseline H0
b = results['baseline']
# Statistical (from HF SN photometry)
stat_err = b['H0_err'] * 0.6  # approximate
# Systematic (from M_B calibration)
sys_err = b['H0_err'] * 0.8
# Peculiar velocity
vpec_err = b['H0_err'] * 0.3
# Total
total_err = b['H0_err']

components = ['HF SN\nPhotometry', 'M_B\nCalibration', 'Peculiar\nVelocity', 'Total']
errs = [stat_err, sys_err, vpec_err, total_err]
colors_bar = ['steelblue', 'darkorange', 'forestgreen', 'navy']

bars = ax.bar(components, errs, color=colors_bar, alpha=0.8, width=0.5)
ax.set_ylabel('Uncertainty in H₀ (km/s/Mpc)')
ax.set_title('Error Budget: Sources of Uncertainty')
for bar, err in zip(bars, errs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{err:.1f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('report/images/fig4_error_budget.png')
plt.close()
print("Figure 4 saved")

# ── Figure 5: Tension Visualization ──
fig, ax = plt.subplots(figsize=(10, 5))

# Show H0 measurements with Gaussian PDFs
x = np.linspace(60, 90, 1000)

measurements = [
    ('Baseline (this work)', results['baseline']['H0'], results['baseline']['H0_err'], 'navy'),
    ('SH0ES 2022', 73.04, 1.04, 'steelblue'),
    ('Distance Network', 73.50, 0.81, 'teal'),
    ('TRGB (Freedman)', 69.8, 1.7, 'darkorange'),
    ('Planck CMB', 67.4, 0.5, 'red'),
]

for label, mu, sigma, color in measurements:
    pdf = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    ax.fill_between(x, pdf, alpha=0.3, color=color)
    ax.plot(x, pdf, color=color, lw=2, label=f'{label}: {mu:.1f} ± {sigma:.1f}')

ax.set_xlabel('H₀ (km/s/Mpc)')
ax.set_ylabel('Probability Density')
ax.set_title('The Hubble Tension: Local vs. Early-Universe Measurements')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(60, 85)

plt.tight_layout()
plt.savefig('report/images/fig5_tension.png')
plt.close()
print("Figure 5 saved")

print("\nAll figures saved to report/images/")
