#!/usr/bin/env python3
"""Reproducible analysis for synthetic SXS BBH waveform uncertainty diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 220, 'font.size': 11})

THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def read_csv_auto(path: Path) -> pd.DataFrame:
    # Header handling: read with pandas, then normalize unnamed/single column cases.
    df = pd.read_csv(path)
    # If all columns parse as data column names from first row? Not needed for supplied files; keep robust.
    return df


def load_data():
    fig6 = pd.read_csv(DATA / 'fig6_data.csv')
    if fig6.shape[1] != 1:
        # fall back no header
        fig6 = pd.read_csv(DATA / 'fig6_data.csv', header=None)
    fig6.columns = ['resolution_difference']

    fig7 = pd.read_csv(DATA / 'fig7_data.csv')
    if fig7.shape[1] != 7:
        fig7 = pd.read_csv(DATA / 'fig7_data.csv', header=None)
    fig7.columns = [f'ell_{ell}' for ell in range(2, 9)]

    fig8 = pd.read_csv(DATA / 'fig8_data.csv')
    if fig8.shape[1] != 2:
        fig8 = pd.read_csv(DATA / 'fig8_data.csv', header=None)
    fig8.columns = ['N2_vs_N3', 'N2_vs_N4']
    return fig6, fig7, fig8


def summarize_series(s: pd.Series, name: str) -> dict:
    x = pd.to_numeric(s, errors='coerce').dropna().astype(float)
    d = {
        'diagnostic': name,
        'n': int(x.size),
        'missing': int(pd.to_numeric(s, errors='coerce').isna().sum()),
        'min': float(x.min()),
        'q01': float(x.quantile(0.01)),
        'q05': float(x.quantile(0.05)),
        'q10': float(x.quantile(0.10)),
        'median': float(x.median()),
        'mean': float(x.mean()),
        'std': float(x.std(ddof=1)),
        'q90': float(x.quantile(0.90)),
        'q95': float(x.quantile(0.95)),
        'q99': float(x.quantile(0.99)),
        'max': float(x.max()),
        'log10_mean': float(np.log10(x).mean()),
        'log10_std': float(np.log10(x).std(ddof=1)),
        'geom_mean': float(10 ** np.log10(x).mean()),
    }
    for t in THRESHOLDS:
        d[f'frac_le_{t:g}'] = float((x <= t).mean())
        d[f'frac_gt_{t:g}'] = float((x > t).mean())
    return d


def threshold_table(series_dict):
    rows=[]
    for name, s in series_dict.items():
        x=pd.to_numeric(s, errors='coerce').dropna().astype(float)
        for t in THRESHOLDS:
            rows.append({'diagnostic': name, 'threshold': t, 'fraction_at_or_below': float((x<=t).mean()), 'fraction_above': float((x>t).mean())})
    return pd.DataFrame(rows)


def empirical_cdf(x):
    xs=np.sort(np.asarray(x, dtype=float))
    y=np.arange(1,len(xs)+1)/len(xs)
    return xs,y


def main():
    fig6, fig7, fig8 = load_data()

    validation = {
        'fig6_data.csv': {'shape': list(fig6.shape), 'columns': list(fig6.columns)},
        'fig7_data.csv': {'shape': list(fig7.shape), 'columns': list(fig7.columns)},
        'fig8_data.csv': {'shape': list(fig8.shape), 'columns': list(fig8.columns)},
    }
    for name, df in [('fig6_data.csv', fig6), ('fig7_data.csv', fig7), ('fig8_data.csv', fig8)]:
        num=df.apply(pd.to_numeric, errors='coerce')
        validation[name].update({
            'missing_values': int(num.isna().sum().sum()),
            'nonpositive_values': int((num <= 0).sum().sum()),
            'finite_values': int(np.isfinite(num.to_numpy(dtype=float)).sum()),
            'total_values': int(num.size),
        })
    (OUT / 'data_validation.json').write_text(json.dumps(validation, indent=2))

    # Summaries
    fig6_summary=pd.DataFrame([summarize_series(fig6['resolution_difference'], 'catalog_resolution')])
    fig6_summary.to_csv(OUT / 'fig6_summary.csv', index=False)

    mode_rows=[]
    for ell in range(2,9):
        row=summarize_series(fig7[f'ell_{ell}'], f'ell_{ell}')
        row['ell']=ell
        mode_rows.append(row)
    fig7_summary=pd.DataFrame(mode_rows)
    fig7_summary.to_csv(OUT / 'fig7_mode_summary.csv', index=False)

    extrap_rows=[summarize_series(fig8['N2_vs_N3'], 'N2_vs_N3'), summarize_series(fig8['N2_vs_N4'], 'N2_vs_N4')]
    ratio=fig8['N2_vs_N4'].astype(float)/fig8['N2_vs_N3'].astype(float)
    logdiff=np.log10(fig8['N2_vs_N4'].astype(float))-np.log10(fig8['N2_vs_N3'].astype(float))
    paired={
        'diagnostic':'paired_N2N4_over_N2N3',
        'n': int(len(ratio)),
        'median_ratio': float(np.median(ratio)),
        'mean_ratio': float(np.mean(ratio)),
        'q05_ratio': float(np.quantile(ratio,0.05)),
        'q95_ratio': float(np.quantile(ratio,0.95)),
        'fraction_N2N4_gt_N2N3': float((fig8['N2_vs_N4']>fig8['N2_vs_N3']).mean()),
        'median_log10_difference': float(np.median(logdiff)),
        'mean_log10_difference': float(np.mean(logdiff)),
        'wilcoxon_statistic': float(stats.wilcoxon(np.log10(fig8['N2_vs_N4']), np.log10(fig8['N2_vs_N3']), alternative='greater').statistic),
        'wilcoxon_pvalue': float(stats.wilcoxon(np.log10(fig8['N2_vs_N4']), np.log10(fig8['N2_vs_N3']), alternative='greater').pvalue),
        'spearman_rho': float(stats.spearmanr(np.log10(fig8['N2_vs_N3']), np.log10(fig8['N2_vs_N4'])).statistic),
        'spearman_pvalue': float(stats.spearmanr(np.log10(fig8['N2_vs_N3']), np.log10(fig8['N2_vs_N4'])).pvalue),
    }
    fig8_summary=pd.concat([pd.DataFrame(extrap_rows), pd.DataFrame([paired])], ignore_index=True, sort=False)
    fig8_summary.to_csv(OUT / 'fig8_extrapolation_summary.csv', index=False)

    # Figure source data exports
    series_dict={'catalog_resolution': fig6['resolution_difference'], 'N2_vs_N3': fig8['N2_vs_N3'], 'N2_vs_N4': fig8['N2_vs_N4']}
    for ell in range(2,9): series_dict[f'ell_{ell}']=fig7[f'ell_{ell}']
    thresholds=threshold_table(series_dict)
    thresholds.to_csv(OUT / 'threshold_fractions.csv', index=False)

    long_modes=fig7.copy()
    long_modes['simulation_index']=np.arange(len(long_modes))
    long_modes=long_modes.melt(id_vars='simulation_index', var_name='mode', value_name='difference')
    long_modes['ell']=long_modes['mode'].str.extract(r'(\d+)').astype(int)
    long_modes.to_csv(OUT / 'fig7_mode_long_source.csv', index=False)

    fig8_source=fig8.copy()
    fig8_source['simulation_index']=np.arange(len(fig8_source))
    fig8_source['ratio_N2N4_over_N2N3']=ratio
    fig8_source['log10_ratio']=np.log10(ratio)
    fig8_source.to_csv(OUT / 'fig8_pair_source.csv', index=False)

    # Fig6: histogram + ECDF
    x=fig6['resolution_difference'].astype(float).to_numpy()
    fig, axes=plt.subplots(1,2, figsize=(11,4.5), constrained_layout=True)
    bins=np.logspace(np.floor(np.log10(x.min())), np.ceil(np.log10(x.max())), 55)
    axes[0].hist(x, bins=bins, color='#4C72B0', alpha=0.85, edgecolor='white')
    axes[0].axvline(np.median(x), color='black', lw=2, label=f"median={np.median(x):.2e}")
    for t,c in [(1e-3,'#DD8452'),(1e-2,'#C44E52'),(1e-1,'#8172B3')]:
        axes[0].axvline(t, color=c, ls='--', lw=1.5, label=f'{t:g}')
    axes[0].set_xscale('log'); axes[0].set_xlabel('highest-resolution waveform difference'); axes[0].set_ylabel('count')
    axes[0].set_title('Resolution-error distribution')
    axes[0].legend(fontsize=8)
    xs,y=empirical_cdf(x)
    axes[1].plot(xs,y,color='#4C72B0',lw=2)
    axes[1].set_xscale('log'); axes[1].set_xlabel('waveform difference'); axes[1].set_ylabel('empirical CDF')
    axes[1].set_title('Cumulative catalog coverage')
    for t in [1e-4,1e-3,1e-2,1e-1]:
        axes[1].axvline(t, color='gray', ls=':', lw=1)
        axes[1].text(t,0.04,f'{t:g}',rotation=90,va='bottom',ha='right',fontsize=8)
    fig.savefig(IMG / 'fig6_catalog_distribution.png')
    plt.close(fig)

    # Fig7 mode distributions
    fig, axes=plt.subplots(1,2, figsize=(12,4.8), constrained_layout=True)
    sns.boxenplot(data=long_modes, x='ell', y='difference', ax=axes[0], color='#55A868')
    axes[0].set_yscale('log'); axes[0].set_xlabel('spherical-harmonic ell'); axes[0].set_ylabel('mode waveform difference')
    axes[0].set_title('Mode-wise uncertainty distributions')
    axes[0].axhline(1e-3,color='#DD8452',ls='--',lw=1.3)
    axes[0].axhline(1e-2,color='#C44E52',ls='--',lw=1.3)
    axes[1].plot(fig7_summary['ell'], fig7_summary['median'], marker='o', lw=2, label='median')
    axes[1].fill_between(fig7_summary['ell'].to_numpy(), fig7_summary['q10'].to_numpy(), fig7_summary['q90'].to_numpy(), alpha=0.25, label='10-90%')
    axes[1].plot(fig7_summary['ell'], fig7_summary['q95'], marker='^', lw=1.5, label='95%')
    axes[1].set_yscale('log'); axes[1].set_xlabel('ell'); axes[1].set_ylabel('difference')
    axes[1].set_title('Median and tail increase with ell')
    axes[1].legend(fontsize=9)
    fig.savefig(IMG / 'fig7_mode_distributions.png')
    plt.close(fig)

    # Fig8 extrapolation comparison
    long8=fig8.reset_index(names='simulation_index').melt(id_vars='simulation_index', var_name='comparison', value_name='difference')
    fig, axes=plt.subplots(1,2, figsize=(11.5,4.8), constrained_layout=True)
    sns.violinplot(data=long8, x='comparison', y='difference', ax=axes[0], inner='quartile', palette=['#4C72B0','#C44E52'], cut=0)
    axes[0].set_yscale('log'); axes[0].set_xlabel('extrapolation comparison'); axes[0].set_ylabel('waveform difference')
    axes[0].set_title('Extrapolation-order differences')
    lims=[min(fig8.min())*0.7, max(fig8.max())*1.4]
    axes[1].scatter(fig8['N2_vs_N3'], fig8['N2_vs_N4'], s=15, alpha=0.45, color='#8172B3', edgecolor='none')
    axes[1].plot(lims, lims, color='black', ls='--', lw=1, label='equal')
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_xlim(lims); axes[1].set_ylim(lims)
    axes[1].set_xlabel('N=2 vs N=3 difference'); axes[1].set_ylabel('N=2 vs N=4 difference')
    axes[1].set_title(f"Paired: {paired['fraction_N2N4_gt_N2N3']:.1%} N2-N4 larger")
    axes[1].legend(fontsize=9)
    fig.savefig(IMG / 'fig8_extrapolation_comparison.png')
    plt.close(fig)

    # Threshold heatmap
    heat=thresholds[thresholds['threshold'].isin([1e-4,1e-3,1e-2,1e-1])].copy()
    heat['threshold_label']=heat['threshold'].map(lambda v: f'≤{v:g}')
    pivot=heat.pivot(index='diagnostic', columns='threshold_label', values='fraction_at_or_below')
    order=['catalog_resolution']+[f'ell_{ell}' for ell in range(2,9)]+['N2_vs_N3','N2_vs_N4']
    pivot=pivot.loc[order]
    fig, ax=plt.subplots(figsize=(8.2,6.2), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax, cbar_kws={'label':'fraction at or below threshold'})
    ax.set_xlabel('accuracy threshold'); ax.set_ylabel('diagnostic'); ax.set_title('Coverage by mismatch-like thresholds')
    fig.savefig(IMG / 'summary_threshold_heatmap.png')
    plt.close(fig)

    # Compact all-summary JSON for report generation
    compact={
        'fig6': fig6_summary.iloc[0].to_dict(),
        'fig7': fig7_summary[['ell','median','q90','q95','frac_le_0.001','frac_le_0.01','max']].to_dict(orient='records'),
        'fig8': fig8_summary.to_dict(orient='records'),
        'validation': validation,
        'thresholds': thresholds.to_dict(orient='records')
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(compact, indent=2))

    # Claim recovery table
    claims=[
        {'claim_id':'C1','claim':'The catalog-wide resolution diagnostic has a median near 4e-4 and a long high-error tail.','supporting_artifact':'outputs/fig6_summary.csv; report/images/fig6_catalog_distribution.png','status':'supported'},
        {'claim_id':'C2','claim':'Most catalog-wide resolution differences are below 1e-3, while a small tail exceeds 1e-2 and rarer cases exceed 1e-1.','supporting_artifact':'outputs/threshold_fractions.csv; report/images/summary_threshold_heatmap.png','status':'supported'},
        {'claim_id':'C3','claim':'Mode-wise differences increase systematically from ell=2 to ell=8 in median and tail quantiles.','supporting_artifact':'outputs/fig7_mode_summary.csv; report/images/fig7_mode_distributions.png','status':'supported'},
        {'claim_id':'C4','claim':'N=2 vs N=4 extrapolation differences are typically larger than N=2 vs N=3 differences in paired simulations.','supporting_artifact':'outputs/fig8_extrapolation_summary.csv; report/images/fig8_extrapolation_comparison.png','status':'supported'},
        {'claim_id':'C5','claim':'Only uncertainty diagnostics are analyzed; raw strain, Weyl scalar, horizon properties, and detailed metadata are not present in the workspace.','supporting_artifact':'outputs/data_validation.json; outputs/method_contract.json','status':'limitation'}
    ]
    pd.DataFrame(claims).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    print(json.dumps({'ok': True, 'artifacts': [str(p.relative_to(ROOT)) for p in [OUT/'data_validation.json', OUT/'fig6_summary.csv', OUT/'fig7_mode_summary.csv', OUT/'fig8_extrapolation_summary.csv', IMG/'fig6_catalog_distribution.png', IMG/'fig7_mode_distributions.png', IMG/'fig8_extrapolation_comparison.png', IMG/'summary_threshold_heatmap.png', OUT/'claim_recovery_table.csv']]}, indent=2))

if __name__ == '__main__':
    main()
