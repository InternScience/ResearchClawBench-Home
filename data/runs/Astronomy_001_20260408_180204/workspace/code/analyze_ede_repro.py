import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'DESI_EDE_Repro_Data.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')


def load_structured_text(path: Path):
    ns = {}
    text = path.read_text()
    exec(text, {}, ns)
    return ns


def dict_to_df(name, d):
    rows = []
    for p, (mean, sigma) in d.items():
        rows.append({'model': name, 'parameter': p, 'mean': mean, 'sigma': sigma})
    return pd.DataFrame(rows)


def points_to_df(name, pts, yname):
    return pd.DataFrame(pts, columns=['z', 'value', 'error']).assign(dataset=name, observable=yname)


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()


def main():
    ns = load_structured_text(DATA)
    params = pd.concat([
        dict_to_df('LambdaCDM', ns['lcdm_params']),
        dict_to_df('EDE', ns['ede_params']),
        dict_to_df('w0wa', ns['w0wa_params']),
    ], ignore_index=True)
    params.to_csv(OUT / 'parameter_summary.csv', index=False)

    common = params.pivot(index='parameter', columns='model', values='mean')
    common_err = params.pivot(index='parameter', columns='model', values='sigma')
    common_params = sorted(set(ns['lcdm_params']).intersection(ns['ede_params']).intersection(ns['w0wa_params']))
    common_df = params[params['parameter'].isin(common_params)].copy()
    common_df.to_csv(OUT / 'common_parameter_summary.csv', index=False)

    # Derived comparison table
    lcdm = {k: v[0] for k, v in ns['lcdm_params'].items()}
    ede = {k: v[0] for k, v in ns['ede_params'].items()}
    w0wa = {k: v[0] for k, v in ns['w0wa_params'].items()}
    rows = []
    for p in common_params:
        rows.append({
            'parameter': p,
            'lcdm_mean': lcdm[p],
            'ede_mean': ede[p],
            'w0wa_mean': w0wa[p],
            'ede_minus_lcdm': ede[p] - lcdm[p],
            'w0wa_minus_lcdm': w0wa[p] - lcdm[p],
            'ede_shift_sigma_lcdm': (ede[p] - lcdm[p]) / ns['lcdm_params'][p][1],
            'w0wa_shift_sigma_lcdm': (w0wa[p] - lcdm[p]) / ns['lcdm_params'][p][1],
        })
    derived = pd.DataFrame(rows).sort_values('parameter')
    derived.to_csv(OUT / 'derived_model_shifts.csv', index=False)

    # Distance residual data
    dist = pd.concat([
        points_to_df('DESI_BAO_DVr_rd', ns['desi_dvrd_points'], 'Delta(D_V/r_d)'),
        points_to_df('DESI_BAO_FAP', ns['desi_fap_points'], 'Delta(F_AP)'),
        points_to_df('Union3_SNe', ns['sne_mu_points'], 'Delta(mu)'),
    ], ignore_index=True)
    dist.to_csv(OUT / 'distance_residual_points.csv', index=False)

    # Simple chi2 vs zero-residual fiducial for extracted points
    chi_rows = []
    for ds, g in dist.groupby('dataset'):
        chi2 = np.sum((g['value'] / g['error']) ** 2)
        chi_rows.append({'dataset': ds, 'n_points': len(g), 'chi2_vs_zero': chi2, 'chi2_per_point': chi2 / len(g)})
    chi = pd.DataFrame(chi_rows)
    chi.to_csv(OUT / 'distance_residual_chi2_vs_zero.csv', index=False)

    # Figure 1: common parameter comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    order = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
    display = {'omega_m': r'$\Omega_m$', 'H0': r'$H_0$', 'sigma8': r'$\sigma_8$', 'ns': r'$n_s$', 'ombh2': r'$\omega_b$', 'ln10As': r'$\ln(10^{10}A_s)$', 'tau': r'$\tau$'}
    colors = {'LambdaCDM': '#4C72B0', 'EDE': '#C44E52', 'w0wa': '#55A868'}
    for ax, p in zip(axes, order):
        sub = common_df[common_df['parameter'] == p]
        ypos = np.arange(len(sub))
        for i, (_, row) in enumerate(sub.reset_index(drop=True).iterrows()):
            ax.errorbar(row['mean'], i, xerr=row['sigma'], fmt='o', color=colors[row['model']], capsize=4)
        ax.set_yticks(np.arange(3), ['LambdaCDM', 'EDE', 'w0wa'])
        ax.set_title(display[p])
    axes[-1].axis('off')
    save_fig(IMG / 'parameter_constraints_comparison.png')

    # Figure 2: H0-Omega_m plane with 1 sigma ellipse proxies
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, color in colors.items():
        sub = params[params['model'] == model].set_index('parameter')
        x = sub.loc['omega_m', 'mean']
        y = sub.loc['H0', 'mean']
        xs = sub.loc['omega_m', 'sigma']
        ys = sub.loc['H0', 'sigma']
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt='o', color=color, capsize=4, label=model)
        ax.text(x + xs*1.2, y + ys*0.2, model, color=color, fontsize=11)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
    ax.set_title(r'Model-dependent shifts in $(\Omega_m, H_0)$')
    ax.legend(frameon=True)
    save_fig(IMG / 'omega_m_h0_model_shifts.png')

    # Figure 3: EDE parameter Gaussian proxies
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ede_sub = params[params['model'] == 'EDE'].set_index('parameter')
    for ax, p, title in zip(axes, ['f_EDE', 'log10_ac'], [r'$f_{\rm EDE}$', r'$\log_{10} a_c$']):
        mu = ede_sub.loc[p, 'mean']
        sig = ede_sub.loc[p, 'sigma']
        xs = np.linspace(mu - 4*sig, mu + 4*sig, 300)
        ys = np.exp(-0.5*((xs-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))
        ax.plot(xs, ys, color=colors['EDE'], lw=2)
        ax.axvline(mu, color='k', ls='--', lw=1)
        ax.fill_between(xs, 0, ys, where=(xs > mu-sig) & (xs < mu+sig), color=colors['EDE'], alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel('Gaussian proxy density')
    save_fig(IMG / 'ede_parameter_posteriors.png')

    # Figure 4: distance residuals
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    mapping = [
        ('DESI_BAO_DVr_rd', r'$\Delta(D_V/r_d)$'),
        ('DESI_BAO_FAP', r'$\Delta F_{AP}$'),
        ('Union3_SNe', r'$\Delta\mu$'),
    ]
    for ax, (ds, ylabel) in zip(axes, mapping):
        g = dist[dist['dataset'] == ds]
        ax.errorbar(g['z'], g['value'], yerr=g['error'], fmt='o', color='#4C72B0', capsize=3)
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Redshift z')
        ax.set_title(ds)
    save_fig(IMG / 'distance_residuals.png')

    # Figure 5: normalized residual significance
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, (ds, title) in zip(axes, mapping):
        g = dist[dist['dataset'] == ds].copy()
        g['snr'] = g['value'] / g['error']
        ax.bar(g['z'].astype(str), g['snr'], color='#8172B2')
        ax.axhline(0, color='k', lw=1)
        ax.axhline(1, color='gray', ls=':')
        ax.axhline(-1, color='gray', ls=':')
        ax.set_title(title)
        ax.set_xlabel('z')
        ax.set_ylabel('Residual / error')
        ax.tick_params(axis='x', rotation=45)
    save_fig(IMG / 'distance_residual_significance.png')

    # Summary stats markdown fragment
    h0_values = params[params['parameter'] == 'H0'][['model', 'mean', 'sigma']].sort_values('mean', ascending=False)
    omega_values = params[params['parameter'] == 'omega_m'][['model', 'mean', 'sigma']]
    summary = {
        'h0_ranking': h0_values.to_dict(orient='records'),
        'omega_m_ranking': omega_values.sort_values('mean').to_dict(orient='records'),
        'chi2_vs_zero': chi.to_dict(orient='records'),
    }
    (OUT / 'summary_stats.txt').write_text(str(summary))
    print('Analysis complete.')


if __name__ == '__main__':
    main()
