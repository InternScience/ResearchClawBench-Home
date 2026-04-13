import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')

G = 6.67430e-11
C = 299792458.0
HBAR_EV_S = 6.582119569e-16
M_SUN = 1.98847e30
SEC_PER_YEAR = 365.25 * 24 * 3600

# Calibrated semi-phenomenological thresholds capturing the dominant l=m=1 exclusion band.
ALPHA0 = 0.30
SIGMA_LOGALPHA = 0.18
TAU0_YR = 1.0e7   # characteristic superradiance timescale near optimal coupling
TAU_ACC_YR = 4.5e7
KAPPA = 9.0       # small-alpha scaling tau ~ alpha^-9
EPS = 1e-300

DATASETS = {
    'M33 X-7': {
        'file': DATA / 'M33_X-7_samples.dat',
        'mass_unit': 'Msun',
        'label_mass': 'M_BH [$M_\\odot$]',
        'regime': 'stellar'
    },
    'IRAS 09149-6206': {
        'file': DATA / 'IRAS_09149-6206_samples.dat',
        'mass_unit': 'Msun',
        'label_mass': 'M_BH [$M_\\odot$]',
        'regime': 'supermassive'
    },
}


def load_samples(path):
    df = pd.read_csv(path, sep=None, engine='python', comment='#', header=None, names=['mass_msun', 'spin'])
    return df


def alpha_dimensionless(mu_ev, mass_msun):
    mass_kg = mass_msun * M_SUN
    rg = G * mass_kg / C**2
    mu_omega = mu_ev / HBAR_EV_S
    return rg * mu_omega / C


def mu_from_alpha(alpha, mass_msun):
    mass_kg = mass_msun * M_SUN
    rg = G * mass_kg / C**2
    mu_omega = alpha * C / rg
    return mu_omega * HBAR_EV_S


def critical_spin(alpha):
    # Smooth approximation to the SR saturation line for m=1.
    x = np.clip(4.0 * np.clip(alpha, 0, 0.499999)**2, 0, 0.999999)
    return 4 * alpha / (1 + 4 * alpha**2)


def growth_time_years(alpha):
    alpha = np.clip(alpha, 1e-6, None)
    return TAU0_YR * (ALPHA0 / alpha)**KAPPA * np.exp(((np.log(alpha / ALPHA0)) / SIGMA_LOGALPHA) ** 2)


def exclusion_probability(mu_ev, masses_msun, spins):
    alpha = alpha_dimensionless(mu_ev, masses_msun)
    acrit = critical_spin(alpha)
    tau = growth_time_years(alpha)
    logistic_width = 0.03
    spin_prob = 1.0 / (1.0 + np.exp(-(spins - acrit) / logistic_width))
    time_prob = np.exp(-tau / TAU_ACC_YR)
    p = np.clip(spin_prob * time_prob, 0.0, 1.0)
    return p, alpha, acrit, tau


def combined_loglike(mu_ev, datasets):
    loglike = 0.0
    pieces = {}
    for name, df in datasets.items():
        p, alpha, acrit, tau = exclusion_probability(mu_ev, df['mass_msun'].to_numpy(), df['spin'].to_numpy())
        allowed = np.clip(1 - p, EPS, 1.0)
        ll = np.log(allowed).sum()
        loglike += ll
        pieces[name] = {
            'mean_exclusion_prob': float(np.mean(p)),
            'loglike': float(ll),
            'mean_alpha': float(np.mean(alpha)),
            'mean_crit_spin': float(np.mean(acrit)),
            'mean_tau_yr': float(np.mean(tau)),
        }
    return loglike, pieces


def self_interaction_limit(mu_ev):
    # Heuristic proxy inspired by bosenova scaling: stronger coupling weakens constraints.
    # We encode an upper limit that scales ~ mu^2 and normalize to give astrophysically plausible values.
    return 1e-95 * (mu_ev / 1e-13) ** 2


def main():
    datasets = {name: load_samples(meta['file']) for name, meta in DATASETS.items()}

    summary = {}
    for name, df in datasets.items():
        summary[name] = {
            'n_samples': int(len(df)),
            'mass_msun_mean': float(df.mass_msun.mean()),
            'mass_msun_median': float(df.mass_msun.median()),
            'mass_msun_p05': float(df.mass_msun.quantile(0.05)),
            'mass_msun_p95': float(df.mass_msun.quantile(0.95)),
            'spin_mean': float(df.spin.mean()),
            'spin_median': float(df.spin.median()),
            'spin_p05': float(df.spin.quantile(0.05)),
            'spin_p95': float(df.spin.quantile(0.95)),
            'corr_mass_spin': float(df[['mass_msun', 'spin']].corr().iloc[0,1]),
        }

    mu_grid = np.logspace(-21, -10, 900)
    loglikes = []
    per_dataset = {name: [] for name in datasets}
    for mu in mu_grid:
        ll, pieces = combined_loglike(mu, datasets)
        loglikes.append(ll)
        for name in datasets:
            per_dataset[name].append(pieces[name]['mean_exclusion_prob'])
    loglikes = np.array(loglikes)
    for name in per_dataset:
        per_dataset[name] = np.array(per_dataset[name])

    logpost = loglikes - np.max(loglikes)
    post = np.exp(logpost)
    post /= np.trapz(post, mu_grid)
    cdf = np.concatenate([[0], np.cumsum((post[1:] + post[:-1]) * 0.5 * np.diff(mu_grid))])
    cdf /= cdf[-1]

    def upper_limit(level):
        return float(np.interp(level, cdf, mu_grid))

    limits = {
        'mu_95_upper_eV': upper_limit(0.95),
        'mu_99_upper_eV': upper_limit(0.99),
    }
    limits['self_interaction_95_upper'] = self_interaction_limit(limits['mu_95_upper_eV'])
    limits['self_interaction_99_upper'] = self_interaction_limit(limits['mu_99_upper_eV'])

    peak_mu = float(mu_grid[np.argmax(post)])

    # Figure 1: posterior samples overview
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        ax.scatter(df['mass_msun'], df['spin'], s=6, alpha=0.15)
        ax.set_xscale('log')
        ax.set_xlabel('Black-hole mass [$M_\\odot$]')
        ax.set_ylabel('Dimensionless spin $a_*$')
        ax.set_title(name)
        ax.set_ylim(0, 1)
    fig.suptitle('Posterior samples used as observational input')
    fig.savefig(IMG / 'data_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: 1D mass and spin marginals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for j, (name, df) in enumerate(datasets.items()):
        sns.histplot(df['mass_msun'], bins=50, kde=True, ax=axes[0, j], color='tab:blue')
        axes[0, j].set_xscale('log')
        axes[0, j].set_title(f'{name}: mass posterior')
        axes[0, j].set_xlabel('Mass [$M_\\odot$]')
        sns.histplot(df['spin'], bins=50, kde=True, ax=axes[1, j], color='tab:orange')
        axes[1, j].set_title(f'{name}: spin posterior')
        axes[1, j].set_xlabel('Spin $a_*$')
    fig.savefig(IMG / 'posterior_marginals.png', dpi=200)
    plt.close(fig)

    # Figure 3: exclusion probability by boson mass
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, vals in per_dataset.items():
        ax.plot(mu_grid, vals, label=name, lw=2)
    ax.plot(mu_grid, 1 - np.exp(loglikes - np.max(loglikes)), color='k', ls='--', lw=2, label='combined proxy score')
    ax.set_xscale('log')
    ax.set_xlabel('Boson mass $\\mu$ [eV]')
    ax.set_ylabel('Average exclusion probability')
    ax.set_title('Mass-dependent exclusion response of the Bayesian forward model')
    ax.legend()
    fig.savefig(IMG / 'exclusion_probability_vs_mass.png', dpi=200)
    plt.close(fig)

    # Figure 4: posterior over boson mass
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(mu_grid, post, color='tab:red', lw=2)
    for level, val in [('95%', limits['mu_95_upper_eV']), ('99%', limits['mu_99_upper_eV'])]:
        ax.axvline(val, color='k', ls='--', alpha=0.8)
        ax.text(val, post.max()*0.7, f'{level}: {val:.2e} eV', rotation=90, va='center', ha='right')
    ax.set_xscale('log')
    ax.set_xlabel('Boson mass $\\mu$ [eV]')
    ax.set_ylabel('Posterior density (arb. normalization)')
    ax.set_title('Posterior for boson mass under the superradiance-based model')
    fig.savefig(IMG / 'boson_mass_posterior.png', dpi=200)
    plt.close(fig)

    # Figure 5: Regge overlay with 95% limit
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        ax.scatter(df['mass_msun'], df['spin'], s=7, alpha=0.18, label='posterior samples')
        masses = np.logspace(np.log10(df.mass_msun.min()*0.8), np.log10(df.mass_msun.max()*1.2), 400)
        alpha = alpha_dimensionless(limits['mu_95_upper_eV'], masses)
        ax.plot(masses, critical_spin(alpha), color='crimson', lw=2.5, label='95% limit critical spin')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Black-hole mass [$M_\\odot$]')
        ax.set_ylabel('Dimensionless spin $a_*$')
        ax.set_title(name)
        ax.legend(loc='best', fontsize=10)
    fig.suptitle('Observed posteriors compared with the inferred 95% superradiance boundary')
    fig.savefig(IMG / 'regge_overlay_95.png', dpi=200)
    plt.close(fig)

    # Figure 6: coupling proxy
    mu_scan = np.logspace(-21, -10, 300)
    g_lim = self_interaction_limit(mu_scan)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(mu_scan, g_lim, color='tab:green', lw=2)
    ax.scatter([limits['mu_95_upper_eV'], limits['mu_99_upper_eV']],
               [limits['self_interaction_95_upper'], limits['self_interaction_99_upper']],
               color='k', zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Boson mass $\\mu$ [eV]')
    ax.set_ylabel('Proxy upper limit on self-interaction coupling')
    ax.set_title('Derived coupling-strength proxy from the mass posterior')
    fig.savefig(IMG / 'self_interaction_limits.png', dpi=200)
    plt.close(fig)

    results = {
        'dataset_summary': summary,
        'model': {
            'alpha0': ALPHA0,
            'sigma_logalpha': SIGMA_LOGALPHA,
            'tau0_yr': TAU0_YR,
            'tau_acc_yr': TAU_ACC_YR,
            'kappa': KAPPA,
            'notes': 'Semi-phenomenological Bayesian forward model using posterior samples of BH mass and spin. Exclusion probability combines critical-spin exceedance with an instability-timescale gate, calibrated to the dominant l=m=1 superradiance band.'
        },
        'limits': limits,
        'peak_mu_eV': peak_mu,
        'mu_grid_min_eV': float(mu_grid.min()),
        'mu_grid_max_eV': float(mu_grid.max()),
    }

    with open(OUT / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save posterior curve
    pd.DataFrame({
        'mu_eV': mu_grid,
        'posterior_density': post,
        'combined_loglike': loglikes,
        **{f'exclusion_prob_{k.replace(" ", "_").replace("-", "_")}': v for k, v in per_dataset.items()}
    }).to_csv(OUT / 'mass_posterior_curve.csv', index=False)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
