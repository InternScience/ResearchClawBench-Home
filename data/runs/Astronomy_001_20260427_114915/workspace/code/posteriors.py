"""1D Gaussian posteriors for {Omega_m, H0, sigma8} across LCDM/EDE/w0wa, and
2D joint Gaussian ellipses for {Omega_m,H0}, {H0,sigma8}, {f_EDE,log10 a_c}.
We treat each parameter as a marginal Gaussian with the published mean and
1σ error and assume zero correlation in absence of full chains (clearly noted
as a simplification)."""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_io import LCDM, EDE, W0WA

IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

models = {'LCDM': LCDM, 'EDE': EDE, 'w0wa': W0WA}
colors = {'LCDM': 'C0', 'EDE': 'C3', 'w0wa': 'C2'}

# ---- 1D Posteriors ----
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
params_to_plot = [('omega_m', r'$\Omega_m$', (0.27, 0.40)),
                  ('H0',      r'$H_0$ [km/s/Mpc]', (60, 76)),
                  ('sigma8',  r'$\sigma_8$', (0.74, 0.86))]
for ax, (key, lab, lim) in zip(axes, params_to_plot):
    x = np.linspace(*lim, 400)
    for m, p in models.items():
        mu, sg = p[key]
        y = np.exp(-0.5 * ((x - mu) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
        ax.plot(x, y, color=colors[m], lw=2, label=f'{m}: {mu:.3f}±{sg:.3f}')
        ax.axvline(mu, color=colors[m], ls=':', lw=1)
    ax.set_xlabel(lab); ax.set_ylabel('posterior')
    ax.legend(fontsize=8, frameon=False)
plt.suptitle('1D marginal Gaussian posteriors (CMB+DESI best fits)', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'posteriors_1d.png'), dpi=160,
            bbox_inches='tight')
print('saved posteriors_1d.png')

# ---- 2D Posteriors ----
def ellipse(ax, mx, my, sx, sy, color, label=None, n_sigma=(1, 2)):
    for ns in n_sigma:
        e = Ellipse((mx, my), 2 * ns * sx, 2 * ns * sy,
                    fill=False, edgecolor=color, lw=1.6,
                    alpha=0.9 if ns == 1 else 0.45,
                    label=label if ns == 1 else None)
        ax.add_patch(e)
    ax.plot(mx, my, marker='+', color=color, ms=12, mew=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
for m, p in models.items():
    ellipse(ax, p['omega_m'][0], p['H0'][0],
            p['omega_m'][1], p['H0'][1], colors[m], m)
ax.set_xlabel(r'$\Omega_m$'); ax.set_ylabel(r'$H_0$ [km/s/Mpc]')
ax.set_title(r'$\Omega_m$ vs $H_0$')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
# add SH0ES band for reference
ax.axhspan(73.04 - 1.04, 73.04 + 1.04, color='gold', alpha=0.25,
           label='SH0ES 73.04±1.04')
ax.legend(frameon=False)

ax = axes[1]
for m, p in models.items():
    ellipse(ax, p['H0'][0], p['sigma8'][0],
            p['H0'][1], p['sigma8'][1], colors[m], m)
ax.set_xlabel(r'$H_0$'); ax.set_ylabel(r'$\sigma_8$')
ax.set_title(r'$H_0$ vs $\sigma_8$')
ax.legend(frameon=False); ax.grid(alpha=0.3)

ax = axes[2]
ellipse(ax, EDE['f_EDE'][0], EDE['log10_ac'][0],
        EDE['f_EDE'][1], EDE['log10_ac'][1], 'C3', 'EDE')
ax.set_xlabel(r'$f_{\rm EDE}$'); ax.set_ylabel(r'$\log_{10} a_c$')
ax.set_title(r'EDE-specific parameters')
ax.set_xlim(0, 0.2); ax.set_ylim(-3.85, -3.30)
ax.grid(alpha=0.3); ax.legend(frameon=False)

plt.suptitle('2D Gaussian ellipses (1σ and 2σ contours from published '
             'best-fit means and marginal errors)', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'posteriors_2d.png'), dpi=160,
            bbox_inches='tight')
print('saved posteriors_2d.png')
