"""Make all report figures."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(__file__))
from superradiance import alpha_coupling, SEC_PER_YR

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG = os.path.join(ROOT, 'report', 'images')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(IMG, exist_ok=True)

DATA = np.load(os.path.join(OUT, 'exclusion_grids.npz'))
SUMMARY = json.load(open(os.path.join(OUT, 'summary_constraints.json')))

# ------------------------------------------------------------------
# Fig 1: Data overview - posterior corner plots / scatter for both
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
                          gridspec_kw={'height_ratios': [3, 1.2]})
for col, name in enumerate(['M33_X-7', 'IRAS_09149-6206']):
    M = DATA[f'{name}_M']
    a = DATA[f'{name}_a']
    ax = axes[0, col]
    ax.hexbin(np.log10(M), a, gridsize=50, cmap='viridis', mincnt=1)
    ax.set_xlabel(r'$\log_{10}(M_{\rm BH} / M_\odot)$')
    ax.set_ylabel(r'BH spin $a_*$')
    ax.set_title(f'{name}: posterior samples (N={len(M)})')
    ax.grid(alpha=0.3)

    # marginal histograms below
    axb = axes[1, col]
    axb.hist(a, bins=40, color='steelblue', alpha=0.7)
    axb.set_xlabel(r'$a_*$')
    axb.set_ylabel('N')
    axb.set_title(f'{name}: spin marginal'
                  f'\n median={np.median(a):.3f}, $\\sigma$={np.std(a):.3f}')
    axb.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'data_overview.png'), dpi=130)
plt.close()
print('saved data_overview.png')

# ------------------------------------------------------------------
# Fig 2: Regge plane — both BHs in superradiance phase space
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
M_grid = np.geomspace(1, 1e10, 400)

# overlay several mu Regge curves: a* curves where alpha = ~0.4 for n=2 (l=m=1)
# Equivalent to the 'optimal' SR mu for given M.  Plot mu lines from 1e-19 to 1e-10 eV.
mus_eV = np.geomspace(1e-21, 1e-9, 13)
cmap = plt.cm.plasma
for i, mu in enumerate(mus_eV):
    al = alpha_coupling(M_grid, mu)
    # SR threshold: a* = 4 alpha (1 + sqrt(1-a*^2))/(1) for omega=mu, m=1
    # Simpler: for each M, SR forbidden when alpha > 0.5 (use that as ceiling)
    # Plot a_* threshold curve for level n=2 m=1: a_threshold = 4 alpha / (1 + 4 alpha^2)
    a_thr = 4.0 * al / (1.0 + 4.0 * al**2)
    a_thr = np.where(al < 0.5, a_thr, np.nan)
    ax.plot(M_grid, a_thr, color=cmap(i / (len(mus_eV)-1)), alpha=0.6,
            label=f'$\\mu={mu:.0e}$ eV')

# Posterior samples
for name, color, marker in [('M33_X-7', 'C0', 'o'),
                             ('IRAS_09149-6206', 'C3', 's')]:
    M = DATA[f'{name}_M']
    a = DATA[f'{name}_a']
    sub = np.random.default_rng(0).choice(len(M), min(len(M), 1500), replace=False)
    ax.scatter(M[sub], a[sub], s=4, alpha=0.4, color=color, label=name)

ax.set_xscale('log')
ax.set_xlim(1, 1e10)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$M_{\rm BH} \, [M_\odot]$')
ax.set_ylabel(r'$a_*$')
ax.set_title(r'Regge plane: BH posterior samples vs. SR threshold curves $a_{\rm SR}(M;\mu)$')
ax.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.85)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'regge_plane.png'), dpi=130)
plt.close()
print('saved regge_plane.png')

# ------------------------------------------------------------------
# Fig 3: 1D P_excl(mu) per source with tau_BH systematic band
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ['M33_X-7', 'IRAS_09149-6206']):
    mu = DATA[f'{name}_mu']
    P = DATA[f'{name}_Pexcl']
    Plo = DATA[f'{name}_Pexcl_lo']
    Phi = DATA[f'{name}_Pexcl_hi']
    ax.fill_between(mu, Plo, Phi, alpha=0.3, color='C0',
                    label=r'$\tau_{\rm BH}$ systematic band')
    ax.plot(mu, P, 'C0-', lw=2, label='Fiducial $P_{\\rm excl}$')
    ax.axhline(0.95, color='red', ls='--', lw=1, label='95%')
    ax.axhline(0.68, color='orange', ls=':', lw=1, label='68%')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu$ [eV]')
    ax.set_ylabel(r'$P_{\rm excl}(\mu)$')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'Pexcl_mu_per_source.png'), dpi=130)
plt.close()
print('saved Pexcl_mu_per_source.png')

# ------------------------------------------------------------------
# Fig 4: Combined Pexcl(mu) on a single log-axis (mass union)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(DATA['M33_X-7_mu'], DATA['M33_X-7_Pexcl'], 'C0-', lw=2,
        label='M33 X-7 (stellar)')
ax.plot(DATA['IRAS_09149-6206_mu'], DATA['IRAS_09149-6206_Pexcl'], 'C3-', lw=2,
        label='IRAS 09149-6206 (SMBH)')
ax.axhline(0.95, color='red', ls='--', lw=1, label='95%')
ax.axhline(0.68, color='orange', ls=':', lw=1, label='68%')
ax.set_xscale('log')
ax.set_xlabel(r'$\mu$ [eV]')
ax.set_ylabel(r'$P_{\rm excl}(\mu)$')
ax.set_title('Bayesian exclusion probability for ULB mass — both sources')
ax.legend(loc='center right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'Pexcl_mu_combined.png'), dpi=130)
plt.close()
print('saved Pexcl_mu_combined.png')

# ------------------------------------------------------------------
# Fig 5: 2D (mu, f_a) constraint per source
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, ['M33_X-7', 'IRAS_09149-6206']):
    mu = DATA[f'{name}_mu']
    fa = DATA[f'{name}_fa_grid']
    Z = DATA[f'{name}_2D_Pexcl']
    Mg, Fg = np.meshgrid(mu, fa)
    pcm = ax.pcolormesh(Mg, Fg, Z, cmap='inferno', vmin=0, vmax=1, shading='auto')
    cs = ax.contour(Mg, Fg, Z, levels=[0.68, 0.95], colors=['orange', 'red'],
                    linestyles=['--', '-'])
    ax.clabel(cs, inline=True, fontsize=8, fmt={0.68: '68%', 0.95: '95%'})
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\mu$ [eV]')
    ax.set_ylabel(r'$f_a$ [GeV]')
    ax.set_title(f'{name}: $P_{{\\rm excl}}(\\mu, f_a)$')
    plt.colorbar(pcm, ax=ax, label=r'$P_{\rm excl}$')
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'mu_fa_constraint.png'), dpi=130)
plt.close()
print('saved mu_fa_constraint.png')

# ------------------------------------------------------------------
# Fig 6: Sensitivity to tau_BH (curves at three lifetimes)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ['M33_X-7', 'IRAS_09149-6206']):
    mu = DATA[f'{name}_mu']
    ax.plot(mu, DATA[f'{name}_Pexcl_lo'], 'C2--', lw=1.5,
            label=f'$\\tau_{{\\rm BH}}$ low')
    ax.plot(mu, DATA[f'{name}_Pexcl'],    'C0-',  lw=2,
            label=f'$\\tau_{{\\rm BH}}$ fiducial')
    ax.plot(mu, DATA[f'{name}_Pexcl_hi'], 'C3:',  lw=1.5,
            label=f'$\\tau_{{\\rm BH}}$ high')
    ax.axhline(0.95, color='gray', ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu$ [eV]')
    ax.set_ylabel(r'$P_{\rm excl}$')
    ax.set_title(f'{name}: BH-age sensitivity')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'tauBH_sensitivity.png'), dpi=130)
plt.close()
print('saved tauBH_sensitivity.png')

print('\nAll figures saved.')
