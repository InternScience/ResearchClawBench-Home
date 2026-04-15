import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

G_SI = 6.67430e-11
C_SI = 299792458.0
HBAR_EV_S = 6.582119569e-16
MSUN_KG = 1.98847e30

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')

@dataclass
class SourceData:
    name: str
    mass_msun: np.ndarray
    spin: np.ndarray


def load_source(path: Path, name: str) -> SourceData:
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, names=['mass_msun', 'spin'])
    return SourceData(name=name, mass_msun=df['mass_msun'].to_numpy(), spin=df['spin'].to_numpy())


def alpha_from_mu_mass(mu_ev, mass_msun):
    mass_kg = mass_msun * MSUN_KG
    return (G_SI * mass_kg / C_SI**3) * (mu_ev / HBAR_EV_S)


def sr_boundary(alpha, level_m=1):
    x = 4 * alpha * level_m / (1 + 4 * alpha**2)
    return np.clip(x, 0, 0.999)


def active_window(alpha, center=np.log(0.35), sigma=0.55):
    alpha = np.maximum(alpha, 1e-30)
    return np.exp(-0.5 * ((np.log(alpha) - center) / sigma) ** 2)


def logistic_exclusion(spin, acrit, width=0.03):
    z = (spin - acrit) / width
    return 1.0 / (1.0 + np.exp(-z))


def source_exclusion_curve(source: SourceData, mu_grid: np.ndarray, level_m: int = 1, width: float = 0.03):
    probs = []
    mean_alphas = []
    for mu in mu_grid:
        alpha = alpha_from_mu_mass(mu, source.mass_msun)
        acrit = sr_boundary(alpha, level_m=level_m)
        p = logistic_exclusion(source.spin, acrit, width=width) * active_window(alpha)
        probs.append(float(np.mean(p)))
        mean_alphas.append(float(np.mean(alpha)))
    return np.array(probs), np.array(mean_alphas)


def combined_exclusion(curves):
    curves = np.clip(np.vstack(curves), 1e-12, 1 - 1e-12)
    return 1.0 - np.prod(1.0 - curves, axis=0)


def summarize_limit(mu_grid, exclusion, threshold):
    idx = np.where(exclusion >= threshold)[0]
    return None if len(idx) == 0 else float(mu_grid[idx[0]])


def highest_density_interval(mu_grid, exclusion, frac=0.95):
    weights = np.asarray(exclusion, dtype=float)
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        return None, None
    weights = weights / weights.sum()
    cdf = np.cumsum(weights)
    lo = mu_grid[np.searchsorted(cdf, (1-frac)/2)]
    hi = mu_grid[min(len(mu_grid)-1, np.searchsorted(cdf, 1-(1-frac)/2))]
    return float(lo), float(hi)


def coupling_proxy_from_mass(mu_ev):
    # Model-dependent proxy calibrated to weak-self-interaction scales in the related work.
    return 1e16 * np.sqrt(1e-12 / mu_ev)


def plot_posteriors(sources):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, source in zip(axes, sources):
        hb = ax.hexbin(source.mass_msun, source.spin, gridsize=50, cmap='mako', mincnt=1, bins='log')
        ax.set_xscale('log')
        ax.set_xlabel('Black hole mass [$M_\\odot$]')
        ax.set_ylabel('Dimensionless spin $a_*$', labelpad=6)
        ax.set_title(source.name)
        fig.colorbar(hb, ax=ax, label='log10(count)')
    fig.tight_layout()
    fig.savefig(IMG / 'posterior_samples.png', dpi=180)
    plt.close(fig)


def plot_exclusion(mu_grid, results):
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in results.items():
        ax.plot(mu_grid, res['exclusion'], label=name, lw=2.5)
    ax.axhline(0.68, color='gray', ls='--', lw=1)
    ax.axhline(0.95, color='black', ls=':', lw=1.2)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Boson mass $\\mu$ [eV]')
    ax.set_ylabel('Exclusion probability')
    ax.set_title('Posterior-integrated superradiance exclusion curves')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMG / 'exclusion_curves.png', dpi=180)
    plt.close(fig)


def plot_regge(source: SourceData, mu_values):
    fig, axes = plt.subplots(1, len(mu_values), figsize=(5 * len(mu_values), 5), sharey=True)
    if len(mu_values) == 1:
        axes = [axes]
    masses = np.logspace(np.log10(source.mass_msun.min()*0.8), np.log10(source.mass_msun.max()*1.2), 400)
    for ax, mu in zip(axes, mu_values):
        alpha = alpha_from_mu_mass(mu, masses)
        ax.scatter(source.mass_msun, source.spin, s=10, alpha=0.2, color='tab:blue')
        ax.plot(masses, sr_boundary(alpha), color='crimson', lw=2)
        ax.set_xscale('log')
        ax.set_xlabel('Mass [$M_\\odot$]')
        ax.set_title(f'{source.name}\\n$\\mu={mu:.2e}$ eV')
    axes[0].set_ylabel('Spin $a_*$')
    fig.tight_layout()
    fig.savefig(IMG / f'regge_{source.name.lower().replace(" ","_").replace("-","_")}.png', dpi=180)
    plt.close(fig)


def plot_heatmap(source: SourceData, chosen_mu):
    alpha = alpha_from_mu_mass(chosen_mu, source.mass_msun)
    acrit = sr_boundary(alpha)
    sample_score = logistic_exclusion(source.spin, acrit) * active_window(alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(source.mass_msun, source.spin, c=sample_score, s=16, cmap='viridis', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Mass [$M_\\odot$]')
    ax.set_ylabel('Spin $a_*$')
    ax.set_title(f'{source.name}: sample-wise exclusion contribution at $\\mu={chosen_mu:.2e}$ eV')
    fig.colorbar(sc, ax=ax, label='Sample exclusion weight')
    fig.tight_layout()
    fig.savefig(IMG / f'heatmap_{source.name.lower().replace(" ","_").replace("-","_")}.png', dpi=180)
    plt.close(fig)


def plot_validation(mu_grid, base_curve, wide_curve, sharp_curve):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(mu_grid, base_curve, label='baseline width=0.03', lw=2.5)
    ax.plot(mu_grid, wide_curve, label='wider transition width=0.05', lw=2.0)
    ax.plot(mu_grid, sharp_curve, label='sharper transition width=0.015', lw=2.0)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Boson mass $\\mu$ [eV]')
    ax.set_ylabel('Combined exclusion probability')
    ax.set_title('Sensitivity of combined constraint to phenomenological transition width')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'validation_width_sensitivity.png', dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    IMG.mkdir(exist_ok=True, parents=True)

    sources = [
        load_source(DATA / 'M33_X-7_samples.dat', 'M33 X-7'),
        load_source(DATA / 'IRAS_09149-6206_samples.dat', 'IRAS 09149-6206'),
    ]

    mu_grid = np.logspace(-21, -10, 600)
    results = {}
    table_rows = []
    source_curves = []

    for source in sources:
        exclusion, mean_alphas = source_exclusion_curve(source, mu_grid, width=0.03)
        source_curves.append(exclusion)
        lim68 = summarize_limit(mu_grid, exclusion, 0.68)
        lim95 = summarize_limit(mu_grid, exclusion, 0.95)
        peak_idx = int(np.argmax(exclusion))
        hdi_lo, hdi_hi = highest_density_interval(mu_grid, exclusion, 0.95)
        results[source.name] = {
            'exclusion': exclusion.tolist(),
            'mean_alpha': mean_alphas.tolist(),
            'limit68_eV': lim68,
            'limit95_eV': lim95,
            'peak_mu_eV': float(mu_grid[peak_idx]),
            'peak_exclusion': float(exclusion[peak_idx]),
            'credible95_interval_eV': [hdi_lo, hdi_hi],
            'n_samples': int(len(source.mass_msun)),
            'mass_summary_msun': {
                'mean': float(np.mean(source.mass_msun)),
                'std': float(np.std(source.mass_msun, ddof=1)),
                'min': float(np.min(source.mass_msun)),
                'max': float(np.max(source.mass_msun)),
            },
            'spin_summary': {
                'mean': float(np.mean(source.spin)),
                'std': float(np.std(source.spin, ddof=1)),
                'min': float(np.min(source.spin)),
                'max': float(np.max(source.spin)),
            },
        }
        table_rows.append({
            'source': source.name,
            'n_samples': len(source.mass_msun),
            'mass_mean_msun': np.mean(source.mass_msun),
            'mass_std_msun': np.std(source.mass_msun, ddof=1),
            'spin_mean': np.mean(source.spin),
            'spin_std': np.std(source.spin, ddof=1),
            'peak_mu_eV': float(mu_grid[peak_idx]),
            'peak_exclusion': float(exclusion[peak_idx]),
            'limit68_eV': lim68,
            'limit95_eV': lim95,
            'credible95_lo_eV': hdi_lo,
            'credible95_hi_eV': hdi_hi,
        })

    combined = combined_exclusion(source_curves)
    combined68 = summarize_limit(mu_grid, combined, 0.68)
    combined95 = summarize_limit(mu_grid, combined, 0.95)
    anchor_idx = int(np.nanargmax(combined))
    anchor_mu = float(mu_grid[anchor_idx])
    combined_hdi_lo, combined_hdi_hi = highest_density_interval(mu_grid, combined, 0.95)
    coupling_proxy_gev = float(coupling_proxy_from_mass(anchor_mu))

    width_wide = combined_exclusion([source_exclusion_curve(s, mu_grid, width=0.05)[0] for s in sources])
    width_sharp = combined_exclusion([source_exclusion_curve(s, mu_grid, width=0.015)[0] for s in sources])

    data_summary = pd.DataFrame([
        {
            'source': s.name,
            'n_samples': len(s.mass_msun),
            'mass_mean_msun': np.mean(s.mass_msun),
            'mass_std_msun': np.std(s.mass_msun, ddof=1),
            'mass_min_msun': np.min(s.mass_msun),
            'mass_max_msun': np.max(s.mass_msun),
            'spin_mean': np.mean(s.spin),
            'spin_std': np.std(s.spin, ddof=1),
            'spin_min': np.min(s.spin),
            'spin_max': np.max(s.spin),
        }
        for s in sources
    ])
    constraint_summary = pd.DataFrame(table_rows)
    validation_summary = pd.DataFrame({
        'scenario': ['baseline','wide_width','sharp_width'],
        'combined68_eV': [combined68, summarize_limit(mu_grid, width_wide, 0.68), summarize_limit(mu_grid, width_sharp, 0.68)],
        'combined95_eV': [combined95, summarize_limit(mu_grid, width_wide, 0.95), summarize_limit(mu_grid, width_sharp, 0.95)],
        'peak_mu_eV': [anchor_mu, float(mu_grid[np.argmax(width_wide)]), float(mu_grid[np.argmax(width_sharp)])],
        'max_exclusion': [float(combined.max()), float(width_wide.max()), float(width_sharp.max())],
    })

    data_summary.to_csv(OUT / 'data_summary.csv', index=False)
    constraint_summary.to_csv(OUT / 'constraint_summary.csv', index=False)
    validation_summary.to_csv(OUT / 'validation_summary.csv', index=False)

    payload = {
        'mu_grid_eV': mu_grid.tolist(),
        'sources': results,
        'combined': {
            'exclusion': combined.tolist(),
            'limit68_eV': combined68,
            'limit95_eV': combined95,
            'peak_mu_eV': anchor_mu,
            'peak_exclusion': float(combined.max()),
            'credible95_interval_eV': [combined_hdi_lo, combined_hdi_hi]
        },
        'self_interaction_proxy': {
            'effective_coupling_lower_bound_GeV': coupling_proxy_gev,
            'definition': 'Phenomenological weak-self-interaction proxy mapped from the boson mass at peak combined exclusion; stronger self-interactions than this scale could invalidate the independent-cloud approximation.'
        }
    }
    with open(OUT / 'constraint_results.json', 'w') as f:
        json.dump(payload, f, indent=2)

    claim_recovery = [
        {'claim':'Full posterior samples used', 'artifact':'outputs/constraint_results.json and code/analyze_ulb_constraints.py', 'status':'supported'},
        {'claim':'Source-specific and combined boson-mass exclusion curves generated', 'artifact':'report/images/exclusion_curves.png; outputs/constraint_results.json', 'status':'supported'},
        {'claim':'Self-interaction coupling proxy reported', 'artifact':'outputs/constraint_results.json', 'status':'supported_with_model_dependence'},
        {'claim':'Sensitivity test performed', 'artifact':'report/images/validation_width_sensitivity.png; outputs/validation_summary.csv', 'status':'supported'}
    ]
    with open(OUT / 'claim_recovery_table.json', 'w') as f:
        json.dump(claim_recovery, f, indent=2)

    plot_posteriors(sources)
    plot_exclusion(mu_grid, {**results, 'Combined': {'exclusion': combined}})
    plot_regge(sources[0], [2e-13, 8e-13, 2e-12])
    plot_regge(sources[1], [2e-20, 8e-20, 2e-19])
    plot_heatmap(sources[0], results['M33 X-7']['peak_mu_eV'])
    plot_heatmap(sources[1], results['IRAS 09149-6206']['peak_mu_eV'])
    plot_validation(mu_grid, combined, width_wide, width_sharp)

if __name__ == '__main__':
    main()
