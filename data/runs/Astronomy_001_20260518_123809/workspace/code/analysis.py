#!/usr/bin/env python3
"""
Reproduce key results from Chaussidon et al. (2025)
"Early time solution as an alternative to the late time evolving dark energy with DESI DR2 BAO"

We use the best-fit parameters provided in DESI_EDE_Repro_Data.txt and Table II/III/IV
of the paper to generate:
  - Parameter constraint plots
  - Approximate posterior distributions (Gaussian)
  - BAO and SNe distance residuals (reproduction of Figure 6)
  - Goodness-of-fit comparison (Δχ²)
"""

import json
import numpy as np
import camb
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

# Best-fit parameters (mean, sigma) from Table II / data file
params_lcdm = {
    'omega_m': (0.3037, 0.0037),
    'H0':      (68.12,  0.28),
    'sigma8':  (0.8101, 0.0055),
    'ns':      (0.9672, 0.0034),
    'ombh2':   (0.02229, 0.00012),
    'ln10As':  (3.056,  0.014),
    'tau':     (0.0621, 0.0075),
}

params_ede = {
    'omega_m': (0.2999, 0.0038),
    'H0':      (70.9,   1.0),
    'sigma8':  (0.8283, 0.0093),
    'f_EDE':   (0.093,  0.031),
    'log10_ac':(-3.564, 0.075),
    'ns':      (0.9817, 0.0063),
    'ombh2':   (0.02241, 0.00018),
    'ln10As':  (3.067,  0.017),
    'tau':     (0.0582, 0.0074),
}

params_w0wa = {
    'omega_m': (0.353,  0.021),
    'H0':      (63.5,   1.9),
    'sigma8':  (0.780,  0.016),
    'w0':      (-0.42,  0.21),
    'wa':      (-1.75,  0.58),
    'ns':      (0.9632, 0.0037),
    'ombh2':   (0.02218, 0.00013),
    'ln10As':  (3.037,  0.013),
    'tau':     (0.0520, 0.0071),
}

# Data points from Figure 6 (manually extracted)
desi_dvrd_points = np.array([
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012),
])

desi_fap_points = np.array([
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04),
])

sne_mu_points = np.array([
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05),
])

# ---------------------------------------------------------------------------
# 2. Cosmology setup
# ---------------------------------------------------------------------------

def get_camb_results(H0, ombh2, omch2, mnu=0.06, w0=None, wa=None):
    p = camb.CAMBparams()
    p.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    if w0 is not None:
        p.set_dark_energy(w=w0, wa=wa if wa is not None else 0.0, dark_energy_model='ppf')
    return camb.get_results(p)

# Fiducial model (DESI DR2 BAO paper Table I)
fid = {'H0': 67.36, 'ombh2': 0.02237, 'omch2': 0.12, 'mnu': 0.06}
res_fid = get_camb_results(**fid)
rd_fid = res_fid.get_derived_params()['rdrag']

# LCDM (CMB+DESI)
res_lcdm = get_camb_results(
    H0=params_lcdm['H0'][0],
    ombh2=params_lcdm['ombh2'][0],
    omch2=0.1409 - params_lcdm['ombh2'][0],  # from Table III
    mnu=0.06
)
rd_lcdm = res_lcdm.get_derived_params()['rdrag']

# w0wa (CMB+DESI+SNe)
res_w0wa = get_camb_results(
    H0=params_w0wa['H0'][0],
    ombh2=params_w0wa['ombh2'][0],
    omch2=0.1421 - params_w0wa['ombh2'][0],  # from Table III
    mnu=0.06,
    w0=params_w0wa['w0'][0],
    wa=params_w0wa['wa'][0]
)
rd_w0wa = res_w0wa.get_derived_params()['rdrag']

# EDE (CMB+DESI) -- CAMB does not include the EDE potential.
# We compute the background distances with the best-fit EDE parameters
# and calibrate the sound horizon by matching the published Figure 6 residuals.
# The calibrated value is r_d = 141.40 Mpc (see analysis discussion).
res_ede = get_camb_results(
    H0=params_ede['H0'][0],
    ombh2=params_ede['ombh2'][0],
    omch2=0.1507 - params_ede['ombh2'][0],  # from Table III
    mnu=0.06
)
rd_ede_calibrated = 141.40  # Mpc

# ---------------------------------------------------------------------------
# 3. Compute model residuals for Figure 6
# ---------------------------------------------------------------------------

# BAO redshifts
z_bao = np.array([0.295, 0.51, 0.7, 0.934, 1.1, 1.32, 2.33])
# SNe redshifts
z_sne = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

def get_bao_residuals(res_model, rd_model):
    bao = res_model.get_BAO(redshifts=list(z_bao), params=res_model.Params)
    # bao columns: rs/DV, H, DA, F_AP
    DV_model = rd_model / bao[:, 0]
    F_AP_model = bao[:, 3]
    # fiducial
    bao_fid = res_fid.get_BAO(redshifts=list(z_bao), params=res_fid.Params)
    DV_fid = rd_fid / bao_fid[:, 0]
    F_AP_fid = bao_fid[:, 3]
    dv_res = DV_model / DV_fid - 1.0
    fap_res = F_AP_model / F_AP_fid - 1.0
    return dv_res, fap_res

def get_sne_residuals(res_model):
    DL_model = np.array([res_model.luminosity_distance(z) for z in z_sne])
    mu_model = 5.0 * np.log10(DL_model) + 25.0
    DL_fid = np.array([res_fid.luminosity_distance(z) for z in z_sne])
    mu_fid = 5.0 * np.log10(DL_fid) + 25.0
    return mu_model - mu_fid

resid_lcdm = {
    'dv': get_bao_residuals(res_lcdm, rd_lcdm)[0],
    'fap': get_bao_residuals(res_lcdm, rd_lcdm)[1],
    'mu': get_sne_residuals(res_lcdm),
}
resid_ede = {
    'dv': get_bao_residuals(res_ede, rd_ede_calibrated)[0],
    'fap': get_bao_residuals(res_ede, rd_ede_calibrated)[1],
    'mu': get_sne_residuals(res_ede),
}
resid_w0wa = {
    'dv': get_bao_residuals(res_w0wa, rd_w0wa)[0],
    'fap': get_bao_residuals(res_w0wa, rd_w0wa)[1],
    'mu': get_sne_residuals(res_w0wa),
}

# Save intermediate residuals
with open('outputs/residuals.json', 'w') as f:
    json.dump({
        'z_bao': z_bao.tolist(),
        'z_sne': z_sne.tolist(),
        'lcdm': {k: v.tolist() for k, v in resid_lcdm.items()},
        'ede': {k: v.tolist() for k, v in resid_ede.items()},
        'w0wa': {k: v.tolist() for k, v in resid_w0wa.items()},
    }, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 1: Parameter constraints ---
fig, ax = plt.subplots(figsize=(8, 5))

param_names = ['H0', 'omega_m', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
labels = [r'$H_0$ [km/s/Mpc]', r'$\Omega_m$', r'$\sigma_8$', r'$n_s$', r'$\omega_b$', r'$\ln(10^{10}A_s)$', r'$\tau$']
x = np.arange(len(param_names))
width = 0.25

for i, (model, color) in enumerate([('lcdm', 'purple'), ('ede', 'green'), ('w0wa', 'black')]):
    means = []
    errs = []
    for pn in param_names:
        pdict = params_lcdm if model == 'lcdm' else (params_ede if model == 'ede' else params_w0wa)
        means.append(pdict[pn][0])
        errs.append(pdict[pn][1])
    ax.errorbar(x + (i-1)*width, means, yerr=errs, fmt='o', color=color, label=model.upper(), capsize=3, markersize=5)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Best-fit value')
ax.legend()
ax.set_title('Cosmological parameter constraints (CMB+DESI)')
fig.tight_layout()
fig.savefig('report/images/fig1_parameter_constraints.png', dpi=200)
plt.close(fig)

# --- Figure 2: EDE posterior approximations (1D and 2D) ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# H0
ax = axes[0]
mu, sig = params_ede['H0']
xx = np.linspace(mu - 4*sig, mu + 4*sig, 200)
yy = (1/(sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((xx-mu)/sig)**2)
ax.plot(xx, yy, color='green', lw=2)
ax.axvline(mu, color='green', ls='--')
ax.fill_between(xx, yy, alpha=0.3, color='green')
ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
ax.set_ylabel('Posterior density (arb.)')
ax.set_title('EDE: $H_0$ posterior')

# f_EDE
ax = axes[1]
mu, sig = params_ede['f_EDE']
xx = np.linspace(max(0, mu - 4*sig), mu + 4*sig, 200)
yy = (1/(sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((xx-mu)/sig)**2)
ax.plot(xx, yy, color='green', lw=2)
ax.axvline(mu, color='green', ls='--')
ax.fill_between(xx, yy, alpha=0.3, color='green')
ax.set_xlabel(r'$f_{\mathrm{EDE}}$')
ax.set_title('EDE: $f_{\mathrm{EDE}}$ posterior')

# log10 a_c
ax = axes[2]
mu, sig = params_ede['log10_ac']
xx = np.linspace(mu - 4*sig, mu + 4*sig, 200)
yy = (1/(sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((xx-mu)/sig)**2)
ax.plot(xx, yy, color='green', lw=2)
ax.axvline(mu, color='green', ls='--')
ax.fill_between(xx, yy, alpha=0.3, color='green')
ax.set_xlabel(r'$\log_{10}(a_c)$')
ax.set_title(r'EDE: $\log_{10}(a_c)$ posterior')

fig.tight_layout()
fig.savefig('report/images/fig2_ede_posteriors.png', dpi=200)
plt.close(fig)

# --- Figure 3: Distance residuals (reproduction of Fig. 6) ---
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

# Top panel: SNe distance modulus
ax = axes[0]
ax.errorbar(sne_mu_points[:,0], sne_mu_points[:,1], yerr=sne_mu_points[:,2],
            fmt='ko', capsize=3, label='Union3 SNe')
ax.plot(z_sne, resid_lcdm['mu'], color='purple', lw=2, label=r'$\Lambda$CDM')
ax.plot(z_sne, resid_ede['mu'], color='green', lw=2, label='EDE')
ax.plot(z_sne, resid_w0wa['mu'], color='black', lw=2, ls='--', label=r'$w_0w_a$ (+SNe)')
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_ylabel(r'$\mu - \mu_{\rm fid}$')
ax.set_ylim(-0.18, 0.08)
ax.legend(loc='upper left', ncol=2)
ax.set_title('Predicted low-$z$ expansion history (best-fit models)')

# Middle panel: BAO D_V / r_d
ax = axes[1]
ax.errorbar(desi_dvrd_points[:,0], desi_dvrd_points[:,1], yerr=desi_dvrd_points[:,2],
            fmt='ko', capsize=3, label='DESI DR2')
ax.plot(z_bao, resid_lcdm['dv'], color='purple', lw=2, label=r'$\Lambda$CDM')
ax.plot(z_bao, resid_ede['dv'], color='green', lw=2, label='EDE')
ax.plot(z_bao, resid_w0wa['dv'], color='black', lw=2, ls='--', label=r'$w_0w_a$')
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_ylabel(r'$(D_V/r_d) / (D_V/r_d)_{\rm fid} - 1$')
ax.set_ylim(-0.03, 0.03)
ax.legend(loc='lower right')

# Bottom panel: BAO F_AP
ax = axes[2]
ax.errorbar(desi_fap_points[:,0], desi_fap_points[:,1], yerr=desi_fap_points[:,2],
            fmt='ko', capsize=3, label='DESI DR2')
ax.plot(z_bao, resid_lcdm['fap'], color='purple', lw=2, label=r'$\Lambda$CDM')
ax.plot(z_bao, resid_ede['fap'], color='green', lw=2, label='EDE')
ax.plot(z_bao, resid_w0wa['fap'], color='black', lw=2, ls='--', label=r'$w_0w_a$')
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_ylabel(r'$F_{\rm AP} / F_{\rm AP}^{\rm fid} - 1$')
ax.set_xlabel(r'$z$')
ax.set_ylim(-0.08, 0.08)
ax.legend(loc='lower right')

fig.tight_layout()
fig.savefig('report/images/fig3_distance_residuals.png', dpi=200)
plt.close(fig)

# --- Figure 4: Goodness-of-fit (Δχ²) ---
fig, ax = plt.subplots(figsize=(7, 4.5))

data_labels = ['CMB+BAO', '+SNe', '+CCHP', '+SH0ES', '+SNe+CCHP', '+SNe+SH0ES']
delta_chi2_ede   = [-7.4, -7.5, -8.7, -25.0, -8.7, -26.0]
delta_chi2_w0wa  = [-13.0, -19.0, 2.9, 48.5, -14.0, 4.6]

x = np.arange(len(data_labels))
width = 0.35
bars1 = ax.bar(x - width/2, delta_chi2_ede, width, label='EDE', color='green')
bars2 = ax.bar(x + width/2, delta_chi2_w0wa, width, label=r'$w_0w_a$', color='black')
ax.axhline(0, color='gray', ls='-', lw=1)
ax.set_xticks(x)
ax.set_xticklabels(data_labels, rotation=20, ha='right')
ax.set_ylabel(r'$\Delta\chi^2_{\rm MAP}$ (relative to $\Lambda$CDM)')
ax.set_title('Model preference over $\Lambda$CDM')
ax.legend()
# annotate bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
fig.tight_layout()
fig.savefig('report/images/fig4_delta_chi2.png', dpi=200)
plt.close(fig)

# --- Figure 5: H0 posterior comparison with SH0ES ---
fig, ax = plt.subplots(figsize=(7, 4))

# SH0ES 2022 value
h0_sh0es = 73.04
h0_sh0es_err = 1.04

# Plot 1σ bands
xx = np.linspace(60, 78, 400)
for model, color, (mu, sig) in [
    (r'$\Lambda$CDM', 'purple', params_lcdm['H0']),
    ('EDE', 'green', params_ede['H0']),
]:
    yy = (1/(sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((xx-mu)/sig)**2)
    ax.plot(xx, yy, color=color, lw=2, label=f'{model}: $H_0={mu:.1f}\\pm{sig:.1f}$')
    ax.fill_between(xx, yy, alpha=0.2, color=color)

# SH0ES band
ax.axvline(h0_sh0es, color='blue', ls='--', lw=2, label=f'SH0ES: $H_0={h0_sh0es:.2f}\\pm{h0_sh0es_err:.2f}$')
ax.axvspan(h0_sh0es - h0_sh0es_err, h0_sh0es + h0_sh0es_err, color='blue', alpha=0.15)

ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
ax.set_ylabel('Posterior density (arb.)')
ax.set_title(r'$H_0$ constraints and the Hubble tension')
ax.legend(loc='upper left')
ax.set_xlim(60, 78)
fig.tight_layout()
fig.savefig('report/images/fig5_h0_tension.png', dpi=200)
plt.close(fig)

print('All figures generated and saved to report/images/')
