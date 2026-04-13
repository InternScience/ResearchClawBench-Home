import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.dpi'] = 200

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

fig6 = pd.read_csv(DATA / 'fig6_data.csv')
fig7 = pd.read_csv(DATA / 'fig7_data.csv')
fig8 = pd.read_csv(DATA / 'fig8_data.csv')

summary = {}
summary['fig6'] = fig6['waveform_difference'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]).to_dict()
summary['fig6']['fraction_below_1e-3'] = float((fig6['waveform_difference'] < 1e-3).mean())
summary['fig6']['fraction_below_1e-2'] = float((fig6['waveform_difference'] < 1e-2).mean())
summary['fig6']['fraction_above_1e-1'] = float((fig6['waveform_difference'] > 1e-1).mean())

fig7_desc = fig7.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
fig7_desc['mode'] = fig7_desc.index
fig7_desc['ell'] = fig7_desc['mode'].str.replace('ell','').astype(int)
summary['fig7'] = fig7_desc.to_dict(orient='records')

fig8_desc = fig8.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
fig8_desc['comparison'] = fig8_desc.index
summary['fig8'] = fig8_desc.reset_index(drop=True).to_dict(orient='records')
summary['fig8_extra'] = {
    'median_ratio_N2vsN4_to_N2vsN3': float(fig8['N2vsN4'].median() / fig8['N2vsN3'].median()),
    'mean_ratio_N2vsN4_to_N2vsN3': float(fig8['N2vsN4'].mean() / fig8['N2vsN3'].mean()),
    'fraction_N2vsN4_gt_N2vsN3': float((fig8['N2vsN4'] > fig8['N2vsN3']).mean()),
    'spearman_like_rank_corr': float(pd.Series(fig8['N2vsN3']).rank().corr(pd.Series(fig8['N2vsN4']).rank()))
}

for name, series in {
    'fig6_log10': np.log10(fig6['waveform_difference']),
    'fig8_N2vsN3_log10': np.log10(fig8['N2vsN3']),
    'fig8_N2vsN4_log10': np.log10(fig8['N2vsN4'])
}.items():
    summary[name] = pd.Series(series).describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]).to_dict()

fig, ax = plt.subplots(figsize=(9,6))
vals = fig6['waveform_difference'].values
bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40)
ax.hist(vals, bins=bins, color='#4C78A8', alpha=0.85, edgecolor='white')
ax.axvline(np.median(vals), color='crimson', linestyle='--', linewidth=2, label=f"Median = {np.median(vals):.2e}")
ax.axvline(np.quantile(vals, 0.9), color='darkorange', linestyle=':', linewidth=2, label=f"90th pct = {np.quantile(vals,0.9):.2e}")
ax.set_xscale('log')
ax.set_xlabel('Waveform difference between highest resolutions')
ax.set_ylabel('Number of simulations')
ax.set_title('Distribution of synthetic SXS numerical-resolution differences')
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(IMG / 'fig6_resolution_hist.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,6))
xs = np.sort(vals)
ys = np.arange(1, len(xs)+1) / len(xs)
ax.plot(xs, ys, color='#59A14F', linewidth=2.5)
for thr, color in [(1e-4,'#9C755F'), (1e-3,'#F28E2B'), (1e-2,'#E15759')]:
    ax.axvline(thr, color=color, linestyle='--', alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('Waveform difference')
ax.set_ylabel('Empirical cumulative fraction')
ax.set_title('Cumulative accuracy profile of the catalog')
fig.tight_layout()
fig.savefig(IMG / 'fig6_ecdf.png')
plt.close(fig)

long7 = fig7.melt(var_name='mode', value_name='difference')
long7['ell'] = long7['mode'].str.replace('ell','').astype(int)
mode_stats = long7.groupby('ell')['difference'].agg([
    ('median','median'),
    ('q25', lambda s: s.quantile(0.25)),
    ('q75', lambda s: s.quantile(0.75)),
    ('q90', lambda s: s.quantile(0.90))
]).reset_index()
mode_stats.to_csv(OUT / 'fig7_mode_stats.csv', index=False)

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(mode_stats['ell'], mode_stats['median'], marker='o', linewidth=2.5, color='#4C78A8', label='Median')
ax.fill_between(mode_stats['ell'], mode_stats['q25'], mode_stats['q75'], color='#4C78A8', alpha=0.25, label='IQR')
ax.plot(mode_stats['ell'], mode_stats['q90'], marker='s', linewidth=2, color='#E15759', label='90th percentile')
ax.set_yscale('log')
ax.set_xticks(mode_stats['ell'])
ax.set_xlabel('Spherical-harmonic index $\\ell$')
ax.set_ylabel('Mode-wise waveform difference')
ax.set_title('Higher multipoles carry systematically larger numerical differences')
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(IMG / 'fig7_mode_trend.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(data=long7, x='ell', y='difference', ax=ax, color='#72B7B2', showfliers=False)
ax.set_yscale('log')
ax.set_xlabel('Mode $\\ell$')
ax.set_ylabel('Waveform difference')
ax.set_title('Distribution of modal differences across the catalog')
fig.tight_layout()
fig.savefig(IMG / 'fig7_mode_boxplot.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8,7))
ax.scatter(fig8['N2vsN3'], fig8['N2vsN4'], s=18, alpha=0.45, color='#B279A2', edgecolors='none')
lo = float(min(fig8.min().min(), 1e-6))
hi = float(max(fig8.max().max(), 1))
line = np.logspace(np.log10(lo), np.log10(hi), 200)
ax.plot(line, line, linestyle='--', color='black', linewidth=1.5, label='Equality')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Difference between extrapolation orders N=2 and N=3')
ax.set_ylabel('Difference between extrapolation orders N=2 and N=4')
ax.set_title('Higher-order extrapolation comparison is typically larger')
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(IMG / 'fig8_scatter.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,6))
for col, color, label in [('N2vsN3','#4C78A8','N=2 vs N=3'), ('N2vsN4','#E15759','N=2 vs N=4')]:
    v = fig8[col].values
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 35)
    ax.hist(v, bins=bins, histtype='step', linewidth=2.2, color=color, label=f"{label} (median {np.median(v):.2e})")
ax.set_xscale('log')
ax.set_xlabel('Extrapolation-order waveform difference')
ax.set_ylabel('Count')
ax.set_title('Convergence behavior of finite-radius extrapolation')
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(IMG / 'fig8_hist_compare.png')
plt.close(fig)

with open(OUT / 'summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

mode_stats.to_csv(OUT / 'fig7_mode_stats_rounded.csv', index=False)
print('Analysis complete')
