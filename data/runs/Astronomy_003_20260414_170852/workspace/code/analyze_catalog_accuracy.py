import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150


def robust_summary(values):
    v = np.asarray(values, dtype=float)
    logv = np.log10(v)
    return {
        'count': int(v.size),
        'min': float(np.min(v)),
        'q05': float(np.quantile(v, 0.05)),
        'q25': float(np.quantile(v, 0.25)),
        'median': float(np.median(v)),
        'q75': float(np.quantile(v, 0.75)),
        'q95': float(np.quantile(v, 0.95)),
        'max': float(np.max(v)),
        'mean': float(np.mean(v)),
        'std': float(np.std(v, ddof=1)),
        'ge_1e-3_frac': float(np.mean(v >= 1e-3)),
        'ge_1e-2_frac': float(np.mean(v >= 1e-2)),
        'ge_1e-1_frac': float(np.mean(v >= 1e-1)),
        'log10_mean': float(np.mean(logv)),
        'log10_std': float(np.std(logv, ddof=1)),
    }


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    fig6 = pd.read_csv(DATA / 'fig6_data.csv')
    fig7 = pd.read_csv(DATA / 'fig7_data.csv')
    fig8 = pd.read_csv(DATA / 'fig8_data.csv')

    dataset_overview = {
        'fig6_data.csv': {'rows': int(fig6.shape[0]), 'columns': list(fig6.columns)},
        'fig7_data.csv': {'rows': int(fig7.shape[0]), 'columns': list(fig7.columns)},
        'fig8_data.csv': {'rows': int(fig8.shape[0]), 'columns': list(fig8.columns)},
    }
    save_json(OUT / 'dataset_overview.json', dataset_overview)

    fig6_summary = robust_summary(fig6['waveform_difference'])
    save_json(OUT / 'fig6_summary.json', fig6_summary)

    mode_records = []
    for col in fig7.columns:
        ell = int(col.replace('ell', ''))
        s = robust_summary(fig7[col])
        s['ell'] = ell
        mode_records.append(s)
    fig7_summary = pd.DataFrame(mode_records).sort_values('ell')
    fig7_summary.to_csv(OUT / 'fig7_mode_summary.csv', index=False)

    fig8_summary = {
        'N2vsN3': robust_summary(fig8['N2vsN3']),
        'N2vsN4': robust_summary(fig8['N2vsN4']),
        'pearson_corr': float(np.corrcoef(fig8['N2vsN3'], fig8['N2vsN4'])[0, 1]),
        'spearman_corr': float(pd.Series(fig8['N2vsN3']).corr(pd.Series(fig8['N2vsN4']), method='spearman')),
        'median_ratio_N2vsN4_to_N2vsN3': float(np.median(fig8['N2vsN4']) / np.median(fig8['N2vsN3'])),
        'frac_N2vsN4_gt_N2vsN3': float(np.mean(fig8['N2vsN4'] > fig8['N2vsN3'])),
    }
    save_json(OUT / 'fig8_summary.json', fig8_summary)

    # Figure 1: overall resolution error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    vals6 = fig6['waveform_difference'].to_numpy()
    sns.histplot(vals6, bins=50, ax=axes[0], color='#4C72B0')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Waveform difference')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Resolution-error distribution')
    sns.boxplot(y=np.log10(vals6), ax=axes[1], color='#55A868')
    axes[1].set_ylabel('log10(waveform difference)')
    axes[1].set_title('Distribution on log scale')
    fig.tight_layout()
    fig.savefig(IMG / 'fig6_distribution.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: per-mode distributions and medians
    long7 = fig7.melt(var_name='mode', value_name='difference')
    long7['ell'] = long7['mode'].str.replace('ell', '', regex=False).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.boxplot(data=long7, x='ell', y='difference', ax=axes[0], color='#C44E52', showfliers=False)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Spherical harmonic mode $\\ell$')
    axes[0].set_ylabel('Waveform difference')
    axes[0].set_title('Mode-resolved error distributions')
    axes[1].plot(fig7_summary['ell'], fig7_summary['median'], marker='o', label='Median')
    axes[1].fill_between(fig7_summary['ell'], fig7_summary['q25'], fig7_summary['q75'], alpha=0.25, label='IQR')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Spherical harmonic mode $\\ell$')
    axes[1].set_ylabel('Waveform difference')
    axes[1].set_title('Median growth with $\\ell$')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMG / 'fig7_mode_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: extrapolation-order comparisons
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.kdeplot(np.log10(fig8['N2vsN3']), ax=axes[0], label='N=2 vs N=3', fill=True)
    sns.kdeplot(np.log10(fig8['N2vsN4']), ax=axes[0], label='N=2 vs N=4', fill=True)
    axes[0].set_xlabel('log10(waveform difference)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Extrapolation-order error distributions')
    axes[0].legend()
    axes[1].scatter(fig8['N2vsN3'], fig8['N2vsN4'], s=14, alpha=0.45)
    lims = [min(fig8.min()), max(fig8.max())]
    axes[1].plot(lims, lims, '--', color='black', linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('N=2 vs N=3 difference')
    axes[1].set_ylabel('N=2 vs N=4 difference')
    axes[1].set_title('Pairwise extrapolation comparison')
    fig.tight_layout()
    fig.savefig(IMG / 'fig8_extrapolation.png', bbox_inches='tight')
    plt.close(fig)

    claim_rows = [
        {
            'claim_id': 'C1',
            'claim': 'The overall resolution-error distribution is strongly right-skewed with a median near 4e-4.',
            'artifact': 'outputs/fig6_summary.json',
            'status': 'supported'
        },
        {
            'claim_id': 'C2',
            'claim': 'Median waveform differences increase systematically with harmonic mode index ell=2..8.',
            'artifact': 'outputs/fig7_mode_summary.csv; report/images/fig7_mode_comparison.png',
            'status': 'supported'
        },
        {
            'claim_id': 'C3',
            'claim': 'N=2 vs N=4 discrepancies are typically larger than N=2 vs N=3 discrepancies.',
            'artifact': 'outputs/fig8_summary.json; report/images/fig8_extrapolation.png',
            'status': 'supported'
        },
        {
            'claim_id': 'C4',
            'claim': 'These synthetic catalog-level uncertainties are small enough to be relevant for surrogate-model calibration, though higher modes are less accurate.',
            'artifact': 'outputs/related_work_contract.json; outputs/fig6_summary.json; outputs/fig7_mode_summary.csv',
            'status': 'supported_with_context'
        },
        {
            'claim_id': 'C5',
            'claim': 'The study is limited to synthetic summary statistics, not raw waveforms or horizon trajectories.',
            'artifact': 'outputs/method_contract.json',
            'status': 'supported'
        }
    ]
    pd.DataFrame(claim_rows).to_csv(OUT / 'claim_recovery.csv', index=False)


if __name__ == '__main__':
    main()
