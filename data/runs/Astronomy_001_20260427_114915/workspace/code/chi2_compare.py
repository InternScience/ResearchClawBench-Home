"""Δχ² and AIC comparison + parameter-shift summary."""
import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_io import LCDM, EDE, W0WA, CHI2, CHI2_W0WA_ESTIMATE, NPAR

IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Build delta-chi2 + AIC table
rows = []
for ds, vals in CHI2.items():
    base = vals['LCDM']
    for m in ['LCDM', 'EDE']:
        c = vals[m]
        npar = NPAR[m]
        aic = c + 2 * npar
        dc = c - base
        daic = aic - (base + 2 * NPAR['LCDM'])
        rows.append({'dataset': ds, 'model': m, 'chi2_min': c,
                     'k_params': npar, 'AIC': aic,
                     'dchi2_vs_LCDM': dc, 'dAIC_vs_LCDM': daic})
    # w0wa estimate
    cw = CHI2_W0WA_ESTIMATE[ds]
    aicw = cw + 2 * NPAR['w0wa']
    rows.append({'dataset': ds, 'model': 'w0wa (lit. est.)',
                 'chi2_min': cw, 'k_params': NPAR['w0wa'],
                 'AIC': aicw, 'dchi2_vs_LCDM': cw - base,
                 'dAIC_vs_LCDM': aicw - (base + 2 * NPAR['LCDM'])})

with open('outputs/delta_chi2_AIC.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('saved outputs/delta_chi2_AIC.csv')

# ---- Bar chart of Δχ² and ΔAIC ----
datasets = list(CHI2.keys())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(datasets))
width = 0.32

dchi_lcdm = [0 for _ in datasets]
dchi_ede  = [CHI2[d]['EDE'] - CHI2[d]['LCDM'] for d in datasets]
dchi_w0wa = [CHI2_W0WA_ESTIMATE[d] - CHI2[d]['LCDM'] for d in datasets]

ax = axes[0]
ax.bar(x - width, dchi_lcdm, width, label='ΛCDM', color='C0')
ax.bar(x,         dchi_ede,  width, label='EDE',  color='C3')
ax.bar(x + width, dchi_w0wa, width, label='w0wa (lit.)', color='C2')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel(r'$\Delta\chi^2_{\rm min}$ vs ΛCDM')
ax.set_title(r'Goodness of fit improvement')
ax.legend(frameon=False)

daic_lcdm = [0 for _ in datasets]
daic_ede  = [(CHI2[d]['EDE'] + 2 * NPAR['EDE']) -
             (CHI2[d]['LCDM'] + 2 * NPAR['LCDM']) for d in datasets]
daic_w0wa = [(CHI2_W0WA_ESTIMATE[d] + 2 * NPAR['w0wa']) -
             (CHI2[d]['LCDM'] + 2 * NPAR['LCDM']) for d in datasets]
ax = axes[1]
ax.bar(x - width, daic_lcdm, width, label='ΛCDM', color='C0')
ax.bar(x,         daic_ede,  width, label='EDE',  color='C3')
ax.bar(x + width, daic_w0wa, width, label='w0wa (lit.)', color='C2')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel(r'$\Delta {\rm AIC}$ vs ΛCDM')
ax.set_title(r'Information-criterion penalty')
ax.legend(frameon=False)

plt.suptitle(r'$\chi^2$ comparison from Tables II & III of Poulin+ 2025', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'chi2_bar.png'), dpi=160,
            bbox_inches='tight')
print('saved chi2_bar.png')

# ---- Param shift summary ----
keys = ['omega_m', 'H0', 'sigma8', 'ns']
labels = [r'$\Omega_m$', r'$H_0$', r'$\sigma_8$', r'$n_s$']
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, m in enumerate(['LCDM', 'EDE', 'w0wa']):
    p = {'LCDM': LCDM, 'EDE': EDE, 'w0wa': W0WA}[m]
    means = np.array([p[k][0] for k in keys])
    sigs  = np.array([p[k][1] for k in keys])
    base  = np.array([LCDM[k][0] for k in keys])
    bsigs = np.array([LCDM[k][1] for k in keys])
    delta_in_sigma = (means - base) / np.sqrt(sigs ** 2 + bsigs ** 2 + 1e-12)
    xs = np.arange(len(keys)) + (i - 1) * 0.25
    ax.errorbar(xs, delta_in_sigma, fmt='o', color=['C0','C3','C2'][i],
                ms=8, label=m)
ax.axhline(0, color='gray', lw=0.6)
ax.axhline(2, color='red', ls='--', lw=0.8); ax.axhline(-2, color='red', ls='--', lw=0.8)
ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(labels)
ax.set_ylabel(r'shift relative to ΛCDM, in units of $\sqrt{\sigma_m^2+\sigma_{\Lambda}^2}$')
ax.set_title('Parameter shifts of EDE / w0wa relative to ΛCDM (CMB+DESI)')
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'param_shift.png'), dpi=160,
            bbox_inches='tight')
print('saved param_shift.png')

# ---- Save params_table.csv ----
with open('outputs/params_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    all_keys = sorted(set(LCDM.keys()) | set(EDE.keys()) | set(W0WA.keys()))
    w.writerow(['parameter', 'LCDM_mean', 'LCDM_sigma', 'EDE_mean', 'EDE_sigma',
                'w0wa_mean', 'w0wa_sigma'])
    for k in all_keys:
        row = [k]
        for d in (LCDM, EDE, W0WA):
            if k in d:
                row += [d[k][0], d[k][1]]
            else:
                row += ['', '']
        w.writerow(row)
print('saved outputs/params_table.csv')
