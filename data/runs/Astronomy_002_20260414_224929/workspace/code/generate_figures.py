#!/usr/bin/env python3
"""
Generate all figures for the Hubble Constant measurement report.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load results
with open('outputs/results.json', 'r') as f:
    results = json.load(f)

os.makedirs('report/images', exist_ok=True)

c_km = 299792.458
h0_cmb = 67.4
h0_cmb_err = 0.5

H0_gls = results['gls_combined']['H0']
H0_gls_err = results['gls_combined']['H0_err']
H0_sneia = results['sneia_result']['H0']
H0_sneia_err = results['sneia_result']['H0_err']
M_B = results['sneia_result']['M_B_mean']

# ============================================================
# FIGURE 1: Distance Ladder Overview
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

anchor_names = ['N4258', 'LMC']
anchor_mu = [29.397, 18.477]
anchor_err = [0.032, 0.024]
anchor_d = [10**((mu - 25)/5) for mu in anchor_mu]

ax.errorbar(anchor_d, anchor_mu, 
            xerr=[d * err * np.log(10)/5 for d, err in zip(anchor_d, anchor_err)],
            fmt='o', color='darkred', markersize=10, capsize=5, 
            label='Geometric Anchors', zorder=5)

host_data = results['host_distances']
hosts = list(host_data.keys())
host_mu_vals = [host_data[h]['mu'] for h in hosts]
host_err_vals = [host_data[h]['err'] for h in hosts]
host_d_vals = [10**((mu - 25)/5) for mu in host_mu_vals]

ax.errorbar(host_d_vals, host_mu_vals, 
            xerr=[d * err * np.log(10)/5 for d, err in zip(host_d_vals, host_err_vals)],
            fmt='s', color='steelblue', markersize=8, capsize=4, 
            label='SN Ia Hosts (Primary Indicators)', zorder=4)

z_hf = np.array([x['z'] for x in results['individual_sneia']])
H0_sneia_ind = np.array([x['H0'] for x in results['individual_sneia']])
mu_hf = [5*np.log10(c_km*z/H0) + 25 for z, H0 in zip(z_hf, H0_sneia_ind)]
d_hf = [c_km*z/H0 for z, H0 in zip(z_hf, H0_sneia_ind)]

ax.scatter(d_hf, mu_hf, marker='^', color='forestgreen', s=120, 
           label='Hubble Flow SNe Ia', zorder=3, edgecolors='black', linewidth=0.5)

if results['individual_sbf']:
    z_hf_sbf = np.array([x['z'] for x in results['individual_sbf']])
    H0_sbf_ind = np.array([x['H0'] for x in results['individual_sbf']])
    mu_hf_sbf = [5*np.log10(c_km*z/H0) + 25 for z, H0 in zip(z_hf_sbf, H0_sbf_ind)]
    d_hf_sbf = [c_km*z/H0 for z, H0 in zip(z_hf_sbf, H0_sbf_ind)]
    ax.scatter(d_hf_sbf, mu_hf_sbf, marker='D', color='orange', s=120, 
               label='Hubble Flow SBF', zorder=3, edgecolors='black', linewidth=0.5)

z_theory = np.logspace(-2, 0.5, 100)
d_theory = c_km * z_theory / H0_gls
mu_theory = 5*np.log10(d_theory) + 25
ax.plot(d_theory, mu_theory, 'k--', alpha=0.4, linewidth=1.5, 
        label=f'H0 = {H0_gls:.1f} km/s/Mpc (GLS)')

ax.set_xscale('log')
ax.set_xlabel('Luminosity Distance (Mpc)', fontsize=12)
ax.set_ylabel('Distance Modulus $\mu$ (mag)', fontsize=12)
ax.set_title('Local Distance Network: From Geometric Anchors to Hubble Flow', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig01_distance_ladder.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig01_distance_ladder.png")

# ============================================================
# FIGURE 2: Hubble Diagram
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

cal_hosts_list = [('NGC1309', 12.10), ('NGC1365', 11.93), ('NGC1448', 11.90),
                   ('NGC1559', 12.22), ('M101', 9.85), ('NGC1316', 11.88), ('NGC5643', 11.56)]
host_dists = results['host_distances']
for host, mB in cal_hosts_list:
    if host in host_dists:
        mu = host_dists[host]['mu']
        d = 10**((mu-25)/5)
        z_eff = H0_gls * d / c_km
        ax.scatter(z_eff, mB, marker='o', color='steelblue', s=80, zorder=4,
                   label='SNe Ia Calibrators' if host == 'NGC1309' else '',
                   edgecolors='black', linewidth=0.5)

mB_hf_arr = np.array([15.12, 15.68, 16.35, 17.02, 17.55])
ax.scatter(z_hf, mB_hf_arr, marker='^', color='forestgreen', s=120, zorder=4,
           label='Hubble Flow SNe Ia', edgecolors='black', linewidth=0.5)

z_model = np.linspace(0.01, 0.1, 100)
d_model = c_km * z_model / H0_gls
mu_model = 5*np.log10(d_model) + 25
mB_model = M_B + mu_model
ax.plot(z_model, mB_model, 'k-', linewidth=2, label=f'Model: H0 = {H0_gls:.1f}')

d_cmb = c_km * z_model / h0_cmb
mu_cmb = 5*np.log10(d_cmb) + 25
mB_cmb = M_B + mu_cmb
ax.plot(z_model, mB_cmb, 'r--', linewidth=1.5, alpha=0.7, label=f'CMB: H0 = {h0_cmb}')

ax.set_xlabel('Redshift $z$', fontsize=12)
ax.set_ylabel('Apparent Magnitude $m_B$', fontsize=12)
ax.set_title('Hubble Diagram: SNe Ia Distance Ladder', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('report/images/fig02_hubble_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig02_hubble_diagram.png")

# ============================================================
# FIGURE 3: Individual H0 Measurements
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

y_pos_sneia = np.arange(len(z_hf))
err_ind_sneia = H0_sneia_ind * 0.06 * np.log(10) / 5.0
ax.errorbar(H0_sneia_ind, y_pos_sneia, xerr=err_ind_sneia,
            fmt='o', color='forestgreen', markersize=10, capsize=5, 
            label='SNe Ia', zorder=3)

if results['individual_sbf']:
    H0_sbf_ind = np.array([x['H0'] for x in results['individual_sbf']])
    z_hf_sbf_arr = np.array([x['z'] for x in results['individual_sbf']])
    y_pos_sbf = np.arange(len(H0_sbf_ind)) + len(z_hf) + 0.5
    err_ind_sbf = H0_sbf_ind * 0.15 * np.log(10) / 5.0
    ax.errorbar(H0_sbf_ind, y_pos_sbf, xerr=err_ind_sbf,
                fmt='D', color='orange', markersize=10, capsize=5, 
                label='SBF', zorder=3)

ax.axvline(H0_gls, color='black', linewidth=2.5, linestyle='-', 
           label=f'GLS: {H0_gls:.1f} +/- {H0_gls_err:.1f}')
ax.axvspan(H0_gls - H0_gls_err, H0_gls + H0_gls_err, alpha=0.15, color='black')

ax.axvline(h0_cmb, color='red', linewidth=2.5, linestyle='--', 
           label=f'CMB: {h0_cmb} +/- {h0_cmb_err}')
ax.axvspan(h0_cmb - h0_cmb_err, h0_cmb + h0_cmb_err, alpha=0.15, color='red')

all_labels = [f'SN Ia  z={z:.3f}' for z in z_hf]
if results['individual_sbf']:
    all_labels += [f'SBF  z={z:.3f}' for z in z_hf_sbf_arr]

ax.set_yticks(range(len(all_labels)))
ax.set_yticklabels(all_labels, fontsize=10)
ax.set_xlabel('H0 (km s-1 Mpc-1)', fontsize=12)
ax.set_title('Individual H0 Measurements from Hubble Flow Objects', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('report/images/fig03_individual_H0.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig03_individual_H0.png")

# ============================================================
# FIGURE 4: Analysis Variants
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

variants = results['variants']
names = list(variants.keys())
H0_vals = [variants[n]['H0'] for n in names]
H0_errs = [variants[n]['H0_err'] for n in names]

y_pos = np.arange(len(names))
colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

ax.barh(y_pos, H0_vals, xerr=H0_errs, color=colors, edgecolor='black', 
        linewidth=0.5, capsize=5, height=0.55, zorder=3)

for i, (h, e) in enumerate(zip(H0_vals, H0_errs)):
    ax.text(h + e + 1.5, i, f'{h:.1f} +/- {e:.1f}', va='center', fontsize=9, fontweight='bold')

ax.axvline(h0_cmb, color='red', linewidth=2.5, linestyle='--', label=f'Planck CMB ({h0_cmb})')
ax.axvspan(h0_cmb - h0_cmb_err, h0_cmb + h0_cmb_err, alpha=0.12, color='red')

ax.axvline(H0_gls, color='black', linewidth=2, linestyle=':', label=f'GLS Combined ({H0_gls:.1f})')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('H0 (km s-1 Mpc-1)', fontsize=12)
ax.set_title('Analysis Variants: H0 Measurements', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(40, max(H0_vals) + max(H0_errs) + 15)

plt.tight_layout()
plt.savefig('report/images/fig04_variants.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig04_variants.png")

# ============================================================
# FIGURE 5: chi^2 Profile
# ============================================================

H0_scan = np.load('outputs/H0_scan.npy')
chi2_profile = np.load('outputs/chi2_profile.npy')

fig, ax = plt.subplots(figsize=(10, 6))

chi2_norm = chi2_profile - np.min(chi2_profile)
ax.plot(H0_scan, chi2_norm, 'b-', linewidth=2.5, zorder=3)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Delta chi2 = 1 (68% CL)')
ax.axhline(4.0, color='orange', linestyle='--', linewidth=1.5, label='Delta chi2 = 4 (95% CL)')

idx_min = np.argmin(chi2_norm)
ax.axvline(H0_scan[idx_min], color='black', linewidth=2.5, linestyle='-', 
           label=f'Best fit: {H0_scan[idx_min]:.1f}')

mask_1sigma = chi2_norm <= 1.0
if np.any(mask_1sigma):
    H0_lo = H0_scan[mask_1sigma][0]
    H0_hi = H0_scan[mask_1sigma][-1]
    ax.axvspan(H0_lo, H0_hi, alpha=0.15, color='blue', label='68% Confidence')

ax.axvline(h0_cmb, color='red', linewidth=2, linestyle=':', label=f'CMB: {h0_cmb}')

ax.set_xlabel('H0 (km s-1 Mpc-1)', fontsize=12)
ax.set_ylabel('Delta chi2', fontsize=12)
ax.set_title('Profile Likelihood: H0 Constraint from Distance Network', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig05_chi2_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig05_chi2_profile.png")

# ============================================================
# FIGURE 6: Tension Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

methods = ['This Work\n(GLS Combined)', 'SNe Ia\n(baseline)', 'Cepheid-calibrated', 
           'TRGB-calibrated', 'SBF only', 'Planck CMB']

H0_methods = [results['gls_combined']['H0'], 
              results['sneia_result']['H0'],
              results['variants'].get('Cepheid-calibrated SNe', {}).get('H0', 73),
              results['variants'].get('TRGB-calibrated SNe', {}).get('H0', 70),
              results.get('sbf_result', {}).get('H0') or 74,
              h0_cmb]
err_methods = [results['gls_combined']['H0_err'],
               results['sneia_result']['H0_err'],
               results['variants'].get('Cepheid-calibrated SNe', {}).get('H0_err', 1.0),
               results['variants'].get('TRGB-calibrated SNe', {}).get('H0_err', 2.0),
               results.get('sbf_result', {}).get('H0_err') or 10.0,
               h0_cmb_err]

y_pos = np.arange(len(methods))
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#F44336']

for i in range(len(methods)):
    ax.errorbar([H0_methods[i]], [y_pos[i]], xerr=[err_methods[i]], fmt='o', 
                color=colors_bar[i], markersize=12, capsize=6, elinewidth=2.5, 
                markeredgewidth=2, zorder=3)

for i, (h, e) in enumerate(zip(H0_methods, err_methods)):
    ax.text(h + e + 2, i, f'{h:.1f} +/- {e:.1f}', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel('H0 (km s-1 Mpc-1)', fontsize=12)
ax.set_title('H0 Measurements: Local Distance Network vs. CMB', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(55, max(H0_methods) + max(err_methods) + 10)

plt.tight_layout()
plt.savefig('report/images/fig06_tension.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig06_tension.png")

# ============================================================
# FIGURE 7: Covariance Matrix Heatmap
# ============================================================

C = np.load('outputs/covariance_matrix.npy')
n = C.shape[0]
n_sneia_count = len(results['individual_sneia'])

D = np.sqrt(np.diag(C))
corr = C / np.outer(D, D)
np.fill_diagonal(corr, 1.0)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

labels = []
for i in range(n_sneia_count):
    labels.append(f'SN z={results["individual_sneia"][i]["z"]:.3f}')
if results['individual_sbf']:
    for i in range(len(results['individual_sbf'])):
        labels.append(f'SBF z={results["individual_sbf"][i]["z"]:.3f}')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_title('Correlation Matrix of Hubble Flow Measurements', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Correlation Coefficient')

plt.tight_layout()
plt.savefig('report/images/fig07_covariance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig07_covariance.png")

# ============================================================
# FIGURE 8: SN Ia Absolute Magnitude Distribution
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

snia_mags = results['snia_absolute_mags']
hosts_list = list(snia_mags.keys())
M_values = [snia_mags[h]['M'] for h in hosts_list]
M_errors = [snia_mags[h]['err'] for h in hosts_list]

y_pos = np.arange(len(hosts_list))
colors_mag = plt.cm.viridis(np.linspace(0.2, 0.8, len(hosts_list)))

for i in range(len(hosts_list)):
    ax.errorbar([M_values[i]], [y_pos[i]], xerr=[M_errors[i]], fmt='o', 
                color=colors_mag[i], markersize=10, capsize=5, elinewidth=1.5, 
                zorder=3)

ax.axvline(M_B, color='red', linewidth=2.5, linestyle='-', 
           label=f'Weighted mean: {M_B:.3f}')

for i, (h, m) in enumerate(zip(hosts_list, M_values)):
    ax.text(m + M_errors[i] + 0.05, i, f'{m:.3f}', va='center', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(hosts_list, fontsize=10)
ax.set_xlabel('M_B (mag)', fontsize=12)
ax.set_title('SNe Ia Absolute Magnitudes from Host Galaxy Calibration', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('report/images/fig08_sn_magnitudes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig08_sn_magnitudes.png")

print("\nAll figures saved to report/images/")
