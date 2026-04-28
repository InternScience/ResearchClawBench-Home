"""Compute distance residuals (D_V/r_d, F_AP, mu) of LCDM, EDE, w0wa relative
to a fiducial Planck-2018 ΛCDM baseline. Save to outputs/ and produce the
Figure-6 reproduction.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_io import LCDM, EDE, W0WA, DESI_DV, DESI_FAP, SNE_MU
import cosmology as cz

OUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Fiducial: Planck 2018 ΛCDM
FID_OM = 0.3158
FID_H0 = 67.36
RD_FID = 147.05  # Mpc, Planck 2018

# Per-model rd is shifted in EDE (smaller). We use the ratio rd/rd_fid that
# follows from: H0 * rd ≈ const for CMB-anchored fits. In paper_003 the
# baseline LCDM has H0~68.12 and EDE H0~70.9 so rd_EDE/rd_LCDM ≈ 68.12/70.9.
# We adopt rd_LCDM = 147.05 Mpc and scale the others by H0_fid/H0_model.
def rd_for(model_name, H0):
    return RD_FID * (FID_H0 / H0)

def compute_obs_lcdm(Om, H0):
    rd = rd_for('LCDM', H0)
    return Om, H0, rd, lambda z: cz.D_V(z, H0, cz.E_LCDM, Om), \
           lambda z: cz.F_AP(z, H0, cz.E_LCDM, Om), \
           lambda z: cz.mu_dist(z, H0, cz.E_LCDM, Om)

def compute_obs_w0wa(Om, H0, w0, wa):
    rd = rd_for('w0wa', H0)
    return Om, H0, rd, lambda z: cz.D_V(z, H0, cz.E_w0wa, Om, w0, wa), \
           lambda z: cz.F_AP(z, H0, cz.E_w0wa, Om, w0, wa), \
           lambda z: cz.mu_dist(z, H0, cz.E_w0wa, Om, w0, wa)

def compute_obs_ede(Om, H0, f_EDE, log10_ac):
    rd = rd_for('EDE', H0)
    return Om, H0, rd, lambda z: cz.D_V(z, H0, cz.E_EDE, Om, f_EDE, log10_ac), \
           lambda z: cz.F_AP(z, H0, cz.E_EDE, Om, f_EDE, log10_ac), \
           lambda z: cz.mu_dist(z, H0, cz.E_EDE, Om, f_EDE, log10_ac)

# Fiducial functions
fid_DV = lambda z: cz.D_V(z, FID_H0, cz.E_LCDM, FID_OM)
fid_FAP = lambda z: cz.F_AP(z, FID_H0, cz.E_LCDM, FID_OM)
fid_mu  = lambda z: cz.mu_dist(z, FID_H0, cz.E_LCDM, FID_OM)
RD_LCDM_fid = RD_FID

models = {
    'LCDM': compute_obs_lcdm(LCDM['omega_m'][0], LCDM['H0'][0]),
    'EDE':  compute_obs_ede(EDE['omega_m'][0], EDE['H0'][0],
                            EDE['f_EDE'][0], EDE['log10_ac'][0]),
    'w0wa': compute_obs_w0wa(W0WA['omega_m'][0], W0WA['H0'][0],
                             W0WA['w0'][0], W0WA['wa'][0]),
}

z_grid = np.linspace(0.01, 2.5, 100)
fid_DV_vals = fid_DV(z_grid) / RD_LCDM_fid
fid_FAP_vals = fid_FAP(z_grid)
fid_mu_vals  = fid_mu(z_grid)

curves = {}
for name, (Om, H0, rd, fdv, ffap, fmu) in models.items():
    dvr = fdv(z_grid) / rd
    fap = ffap(z_grid)
    mu  = fmu(z_grid)
    curves[name] = {
        'z': z_grid,
        'dDV_rd_over_fid': dvr / fid_DV_vals - 1.0,
        'dFAP': fap - fid_FAP_vals,
        'dmu':  mu - fid_mu_vals,
        'rd_Mpc': rd, 'H0': H0, 'Om': Om,
    }

# Save numeric tables
import csv
with open('outputs/distance_residuals.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['z'] + [f'{m}_dDVrd' for m in models] +
               [f'{m}_dFAP' for m in models] +
               [f'{m}_dmu'  for m in models])
    for i, z in enumerate(z_grid):
        row = [z] + [curves[m]['dDV_rd_over_fid'][i] for m in models] + \
                    [curves[m]['dFAP'][i] for m in models] + \
                    [curves[m]['dmu'][i]  for m in models]
        w.writerow(row)

# Save chosen rd for reference
with open('outputs/model_rd.json', 'w') as f:
    json.dump({m: {'rd_Mpc': curves[m]['rd_Mpc'],
                   'H0': curves[m]['H0'], 'Om': curves[m]['Om']}
               for m in models}, f, indent=2)

print('rd values:', {m: round(curves[m]['rd_Mpc'], 2) for m in models})

# ---- Figure 6 reproduction ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
colors = {'LCDM': 'C0', 'EDE': 'C3', 'w0wa': 'C2'}
ls     = {'LCDM': '-',  'EDE': '--', 'w0wa': '-.'}

# Panel 1: Δ(D_V/r_d) / fid
ax = axes[0]
for m in ['LCDM', 'EDE', 'w0wa']:
    ax.plot(z_grid, curves[m]['dDV_rd_over_fid'], color=colors[m], ls=ls[m],
            lw=2, label=m)
ax.errorbar(DESI_DV[:, 0], DESI_DV[:, 1], yerr=DESI_DV[:, 2], fmt='ko',
            ms=5, capsize=3, label='DESI DR2')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\Delta(D_V/r_d)/(D_V/r_d)_{\rm fid}$')
ax.set_title(r'BAO isotropic distance')
ax.legend(loc='best', frameon=False)

# Panel 2: ΔF_AP
ax = axes[1]
for m in ['LCDM', 'EDE', 'w0wa']:
    ax.plot(z_grid, curves[m]['dFAP'], color=colors[m], ls=ls[m], lw=2, label=m)
ax.errorbar(DESI_FAP[:, 0], DESI_FAP[:, 1], yerr=DESI_FAP[:, 2], fmt='ko',
            ms=5, capsize=3, label='DESI DR2')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\Delta F_{AP}$')
ax.set_title(r'Alcock–Paczyński')
ax.legend(loc='best', frameon=False)

# Panel 3: Δμ (Union3)
ax = axes[2]
for m in ['LCDM', 'EDE', 'w0wa']:
    ax.plot(z_grid, curves[m]['dmu'], color=colors[m], ls=ls[m], lw=2, label=m)
ax.errorbar(SNE_MU[:, 0], SNE_MU[:, 1], yerr=SNE_MU[:, 2], fmt='ks',
            ms=5, capsize=3, label='Union3')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\Delta\mu$ [mag]')
ax.set_xlim(0, 1.0)
ax.set_title(r'SNe Ia distance modulus')
ax.legend(loc='best', frameon=False)

plt.suptitle('Reproduction of Fig. 6: residuals of best-fit models vs '
             'fiducial Planck-2018 ΛCDM (DESI DR2 BAO + Union3 SNe)', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig6_repro.png'), dpi=160,
            bbox_inches='tight')
print('saved', os.path.join(IMG_DIR, 'fig6_repro.png'))
