"""
Figure generation script for EDE analysis.
Generates all publication-quality figures for the report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os

os.makedirs('report/images', exist_ok=True)

# ============================================================
# Data
# ============================================================

lcdm_params = {
    'omega_m': (0.3037, 0.0037),
    'H0': (68.12, 0.28),
    'sigma8': (0.8101, 0.0055),
    'ns': (0.9672, 0.0034),
    'ombh2': (0.02229, 0.00012),
    'ln10As': (3.056, 0.014),
    'tau': (0.0621, 0.0075)
}

ede_params = {
    'omega_m': (0.2999, 0.0038),
    'H0': (70.9, 1.0),
    'sigma8': (0.8283, 0.0093),
    'f_EDE': (0.093, 0.031),
    'log10_ac': (-3.564, 0.075),
    'ns': (0.9817, 0.0063),
    'ombh2': (0.02241, 0.00018),
    'ln10As': (3.067, 0.017),
    'tau': (0.0582, 0.0074)
}

w0wa_params = {
    'omega_m': (0.353, 0.021),
    'H0': (63.5, 1.9),
    'sigma8': (0.780, 0.016),
    'w0': (-0.42, 0.21),
    'wa': (-1.75, 0.58),
    'ns': (0.9632, 0.0037),
    'ombh2': (0.02218, 0.00013),
    'ln10As': (3.037, 0.013),
    'tau': (0.0520, 0.0071)
}

desi_dvrd_points = [
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012)
]

desi_fap_points = [
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04)
]

sne_mu_points = [
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05)
]

# Color scheme
COLOR_LCDM = '#1f77b4'
COLOR_EDE = '#d62728'
COLOR_W0WA = '#2ca02c'
COLOR_SH0ES = '#ff7f0e'

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


# ============================================================
# Figure 1: Posterior Distributions of Key Parameters
# ============================================================

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Posterior Distributions of Cosmological Parameters\n(CMB + DESI DR2)', fontsize=14, fontweight='bold')

key_params = [
    ('H0', r'$H_0$ (km/s/Mpc)', 60, 78),
    ('omega_m', r'$\Omega_m$', 0.27, 0.39),
    ('sigma8', r'$\sigma_8$', 0.74, 0.87),
    ('ns', r'$n_s$', 0.94, 1.01),
    ('ombh2', r'$\Omega_b h^2$', 0.0215, 0.0230),
    ('ln10As', r'$\ln(10^{10} A_s)$', 2.99, 3.12)
]

for idx, (param, label, x_min, x_max) in enumerate(key_params):
    ax = axes[idx // 3, idx % 3]
    x = np.linspace(x_min, x_max, 500)

    # ΛCDM
    lcdm_mu, lcdm_sig = lcdm_params[param]
    y_lcdm = gaussian_pdf(x, lcdm_mu, lcdm_sig)
    ax.plot(x, y_lcdm, color=COLOR_LCDM, linewidth=2, label=r'$\Lambda$CDM')
    ax.axvline(lcdm_mu, color=COLOR_LCDM, linestyle='--', alpha=0.5)

    # EDE
    ede_mu, ede_sig = ede_params[param]
    y_ede = gaussian_pdf(x, ede_mu, ede_sig)
    ax.plot(x, y_ede, color=COLOR_EDE, linewidth=2, label='EDE')
    ax.axvline(ede_mu, color=COLOR_EDE, linestyle='--', alpha=0.5)

    # w0wa
    w0wa_mu, w0wa_sig = w0wa_params[param]
    y_w0wa = gaussian_pdf(x, w0wa_mu, w0wa_sig)
    ax.plot(x, y_w0wa, color=COLOR_W0WA, linewidth=2, label=r'$w_0w_a$')
    ax.axvline(w0wa_mu, color=COLOR_W0WA, linestyle='--', alpha=0.5)

    ax.set_xlabel(label)
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, alpha=0.3)

axes[1, 2].axis('off')  # Hide unused subplot

plt.tight_layout()
plt.savefig('report/images/fig1_posterior_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig1_posterior_distributions.png")


# ============================================================
# Figure 2: H0 Tension Whisker Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

models = [r'$\Lambda$CDM', 'EDE', r'$w_0w_a$']
h0_means = [lcdm_params['H0'][0], ede_params['H0'][0], w0wa_params['H0'][0]]
h0_errs = [lcdm_params['H0'][1], ede_params['H0'][1], w0wa_params['H0'][1]]
colors = [COLOR_LCDM, COLOR_EDE, COLOR_W0WA]

y_pos = np.arange(len(models))

for i, (model, mean, err, color) in enumerate(zip(models, h0_means, h0_errs, colors)):
    ax.errorbar(mean, i, xerr=err, fmt='o', color=color, capsize=6, markersize=10,
                elinewidth=2, capthick=2, label=f'{model}: {mean:.1f} ± {err:.1f}')

# SH0ES reference
ax.axvline(73.0, color=COLOR_SH0ES, linestyle='--', linewidth=2, alpha=0.7, label='SH0ES: 73.0 ± 1.0')
ax.axvspan(72.0, 74.0, alpha=0.15, color=COLOR_SH0ES)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel(r'$H_0$ (km/s/Mpc)', fontsize=13)
ax.set_title('Hubble Constant Constraints from Different Models\n(CMB + DESI DR2)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(60, 78)

plt.tight_layout()
plt.savefig('report/images/fig2_h0_tension.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig2_h0_tension.png")


# ============================================================
# Figure 3: BAO Distance Measurements (D_V/r_d and F_AP)
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
fig.suptitle('DESI DR2 BAO Distance Measurements\nRelative to Fiducial Model', fontsize=14, fontweight='bold')

# D_V/r_d
z_dvrd = [p[0] for p in desi_dvrd_points]
val_dvrd = [p[1] for p in desi_dvrd_points]
err_dvrd = [p[2] for p in desi_dvrd_points]

ax1.errorbar(z_dvrd, val_dvrd, yerr=err_dvrd, fmt='o', color='#2196F3',
             capsize=5, markersize=8, elinewidth=2, capthick=1.5, label='DESI DR2')
ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax1.fill_between([0, 3], -0.01, 0.01, alpha=0.1, color='gray', label='±1% band')
ax1.set_ylabel(r'$\Delta(D_V/r_d)$', fontsize=12)
ax1.set_title(r'$D_V/r_d$ Deviation from Fiducial', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# F_AP
z_fap = [p[0] for p in desi_fap_points]
val_fap = [p[1] for p in desi_fap_points]
err_fap = [p[2] for p in desi_fap_points]

ax2.errorbar(z_fap, val_fap, yerr=err_fap, fmt='s', color='#FF9800',
             capsize=5, markersize=8, elinewidth=2, capthick=1.5, label='DESI DR2')
ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.fill_between([0, 3], -0.02, 0.02, alpha=0.1, color='gray', label='±2% band')
ax2.set_ylabel(r'$\Delta F_{\rm AP}$', fontsize=12)
ax2.set_xlabel('Redshift $z$', fontsize=12)
ax2.set_title(r'$F_{\rm AP}$ Deviation from Fiducial', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2.8)

plt.tight_layout()
plt.savefig('report/images/fig3_bao_distances.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig3_bao_distances.png")


# ============================================================
# Figure 4: Supernova Distance Modulus
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

z_sne = [p[0] for p in sne_mu_points]
val_sne = [p[1] for p in sne_mu_points]
err_sne = [p[2] for p in sne_mu_points]

ax.errorbar(z_sne, val_sne, yerr=err_sne, fmt='o', color='#9C27B0',
            capsize=5, markersize=8, elinewidth=2, capthick=1.5,
            label='Union3 SNe (extracted)')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Redshift $z$', fontsize=12)
ax.set_ylabel(r'$\Delta \mu$ (mag)', fontsize=12)
ax.set_title('Union3 Supernova Distance Modulus\nRelative to Fiducial Model', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.8)

plt.tight_layout()
plt.savefig('report/images/fig4_sn_distance.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig4_sn_distance.png")


# ============================================================
# Figure 5: EDE Parameter Posteriors
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Early Dark Energy Parameter Constraints\n(CMB + DESI DR2)', fontsize=14, fontweight='bold')

# f_EDE
f_mean, f_sig = ede_params['f_EDE']
x_f = np.linspace(0, 0.25, 500)
y_f = gaussian_pdf(x_f, f_mean, f_sig)
axes[0].plot(x_f, y_f, color=COLOR_EDE, linewidth=2.5)
axes[0].fill_between(x_f, y_f, alpha=0.2, color=COLOR_EDE)
axes[0].axvline(f_mean, color=COLOR_EDE, linestyle='--', alpha=0.7)
axes[0].axvline(0, color='black', linestyle=':', alpha=0.5, label=r'$f_{\rm EDE}=0$ ($\Lambda$CDM limit)')
axes[0].set_xlabel(r'$f_{\rm EDE}$', fontsize=12)
axes[0].set_ylabel('Probability Density', fontsize=12)
axes[0].set_title(r'Maximum EDE Fraction $f_{\rm EDE}$', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.95, f'{f_mean:.3f} ± {f_sig:.3f}', transform=axes[0].transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# log10_ac
ac_mean, ac_sig = ede_params['log10_ac']
x_ac = np.linspace(ac_mean - 4*ac_sig, ac_mean + 4*ac_sig, 500)
y_ac = gaussian_pdf(x_ac, ac_mean, ac_sig)
axes[1].plot(x_ac, y_ac, color=COLOR_EDE, linewidth=2.5)
axes[1].fill_between(x_ac, y_ac, alpha=0.2, color=COLOR_EDE)
axes[1].axvline(ac_mean, color=COLOR_EDE, linestyle='--', alpha=0.7)
axes[1].set_xlabel(r'$\log_{10} a_c$', fontsize=12)
axes[1].set_ylabel('Probability Density', fontsize=12)
axes[1].set_title(r'Critical Scale Factor $\log_{10} a_c$', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.95, f'{ac_mean:.3f} ± {ac_sig:.3f}', transform=axes[1].transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/fig5_ede_parameters.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig5_ede_parameters.png")


# ============================================================
# Figure 6: Model Comparison Summary (Parameter shifts)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Parameters to compare
params_to_compare = ['H0', 'omega_m', 'sigma8']
param_names = [r'$H_0$', r'$\Omega_m$', r'$\sigma_8$']

# Compute fractional shifts relative to ΛCDM
shifts_ede = []
shifts_w0wa = []
errs_ede = []
errs_w0wa = []

for p in params_to_compare:
    base = lcdm_params[p][0]
    shifts_ede.append((ede_params[p][0] - base) / base * 100)
    shifts_w0wa.append((w0wa_params[p][0] - base) / base * 100)
    errs_ede.append(ede_params[p][1] / base * 100)
    errs_w0wa.append(w0wa_params[p][1] / base * 100)

y_pos = np.arange(len(params_to_compare))
width = 0.3

bars1 = ax.barh(y_pos - width/2, shifts_ede, width, xerr=errs_ede,
                color=COLOR_EDE, alpha=0.8, label='EDE', capsize=4)
bars2 = ax.barh(y_pos + width/2, shifts_w0wa, width, xerr=errs_w0wa,
                color=COLOR_W0WA, alpha=0.8, label=r'$w_0w_a$', capsize=4)

ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(param_names, fontsize=12)
ax.set_xlabel('Fractional Shift Relative to ΛCDM (%)', fontsize=12)
ax.set_title('Parameter Shifts: EDE and $w_0w_a$ vs ΛCDM', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('report/images/fig6_parameter_shifts.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig6_parameter_shifts.png")


# ============================================================
# Figure 7: Combined BAO+SNe overview (multi-panel)
# ============================================================

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Panel 1: D_V/r_d
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(z_dvrd, val_dvrd, yerr=err_dvrd, fmt='o', color='#2196F3',
             capsize=4, markersize=7, elinewidth=1.5, label='DESI DR2')
ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
ax1.set_ylabel(r'$\Delta(D_V/r_d)$', fontsize=11)
ax1.set_title(r'$D_V/r_d$ vs Redshift', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: F_AP
ax2 = fig.add_subplot(gs[0, 1])
ax2.errorbar(z_fap, val_fap, yerr=err_fap, fmt='s', color='#FF9800',
             capsize=4, markersize=7, elinewidth=1.5, label='DESI DR2')
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
ax2.set_ylabel(r'$\Delta F_{\rm AP}$', fontsize=11)
ax2.set_title(r'$F_{\rm AP}$ vs Redshift', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: SNe
ax3 = fig.add_subplot(gs[1, 0])
ax3.errorbar(z_sne, val_sne, yerr=err_sne, fmt='^', color='#9C27B0',
             capsize=4, markersize=7, elinewidth=1.5, label='Union3 SNe')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
ax3.set_ylabel(r'$\Delta \mu$ (mag)', fontsize=11)
ax3.set_xlabel('Redshift $z$', fontsize=11)
ax3.set_title('Distance Modulus vs Redshift', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: H0 comparison
ax4 = fig.add_subplot(gs[1, 1])
h0_models = [r'$\Lambda$CDM', 'EDE', r'$w_0w_a$']
h0_vals = [lcdm_params['H0'][0], ede_params['H0'][0], w0wa_params['H0'][0]]
h0_errors = [lcdm_params['H0'][1], ede_params['H0'][1], w0wa_params['H0'][1]]
h0_colors = [COLOR_LCDM, COLOR_EDE, COLOR_W0WA]

for i, (m, v, e, c) in enumerate(zip(h0_models, h0_vals, h0_errors, h0_colors)):
    ax4.errorbar(v, i, xerr=e, fmt='o', color=c, capsize=5, markersize=9,
                 elinewidth=2, label=f'{m}: {v:.1f} ± {e:.1f}')

ax4.axvline(73.0, color=COLOR_SH0ES, linestyle='--', linewidth=1.5, alpha=0.6, label='SH0ES')
ax4.set_yticks(range(3))
ax4.set_yticklabels(h0_models, fontsize=10)
ax4.set_xlabel(r'$H_0$ (km/s/Mpc)', fontsize=11)
ax4.set_title(r'$H_0$ Model Comparison', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim(60, 78)

fig.suptitle('Multi-Probe Cosmological Constraints: BAO, SNe, and $H_0$', fontsize=15, fontweight='bold')
plt.savefig('report/images/fig7_combined_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig7_combined_overview.png")


# ============================================================
# Figure 8: σ₈ - Ωm plane comparison
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Generate contour ellipses (1σ and 2σ)
theta = np.linspace(0, 2*np.pi, 200)

for n_sigma, alpha_val, lw in [(1, 0.3, 1.5), (2, 0.15, 1.0)]:
    # ΛCDM
    om_c, om_e = lcdm_params['omega_m']
    s8_c, s8_e = lcdm_params['sigma8']
    x_lcdm = om_c + n_sigma * om_e * np.cos(theta)
    y_lcdm = s8_c + n_sigma * s8_e * np.sin(theta)
    ax.plot(x_lcdm, y_lcdm, color=COLOR_LCDM, alpha=alpha_val, linewidth=lw)

    # EDE
    om_e2, om_e2e = ede_params['omega_m']
    s8_e2, s8_e2e = ede_params['sigma8']
    x_ede = om_e2 + n_sigma * om_e2e * np.cos(theta)
    y_ede = s8_e2 + n_sigma * s8_e2e * np.sin(theta)
    ax.plot(x_ede, y_ede, color=COLOR_EDE, alpha=alpha_val, linewidth=lw)

    # w0wa
    om_w, om_we = w0wa_params['omega_m']
    s8_w, s8_we = w0wa_params['sigma8']
    x_w0wa = om_w + n_sigma * om_we * np.cos(theta)
    y_w0wa = s8_w + n_sigma * s8_we * np.sin(theta)
    ax.plot(x_w0wa, y_w0wa, color=COLOR_W0WA, alpha=alpha_val, linewidth=lw)

# Mark best-fit points
ax.plot(om_c, s8_c, 'o', color=COLOR_LCDM, markersize=10, label=r'$\Lambda$CDM')
ax.plot(om_e2, s8_e2, 's', color=COLOR_EDE, markersize=10, label='EDE')
ax.plot(om_w, s8_w, '^', color=COLOR_W0WA, markersize=10, label=r'$w_0w_a$')

ax.set_xlabel(r'$\Omega_m$', fontsize=13)
ax.set_ylabel(r'$\sigma_8$', fontsize=13)
ax.set_title(r'$\sigma_8$ – $\Omega_m$ Constraint Comparison\n(CMB + DESI DR2)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig8_s8_omega_contours.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig8_s8_omega_contours.png")

print("\nAll figures generated successfully!")
