"""
Analysis of SXS Binary Black Hole Simulation Catalog Accuracy
==============================================================
This script analyzes waveform accuracy metrics from the SXS catalog:
- Fig 6: Overall resolution error distribution
- Fig 7: Modal error distributions by spherical harmonic mode ℓ
- Fig 8: Extrapolation order convergence comparison

Author: Research Agent
Date: 2026-05-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================================
# 1. Load Data
# ============================================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

fig6 = pd.read_csv('data/fig6_data.csv')
fig7 = pd.read_csv('data/fig7_data.csv')
fig8 = pd.read_csv('data/fig8_data.csv')

print(f"Fig 6: {fig6.shape[0]} simulations, columns: {list(fig6.columns)}")
print(f"Fig 7: {fig7.shape[0]} simulations, columns: {list(fig7.columns)}")
print(f"Fig 8: {fig8.shape[0]} simulations, columns: {list(fig8.columns)}")

# ============================================================================
# 2. Statistical Summaries
# ============================================================================
print("\n" + "=" * 60)
print("Computing statistical summaries...")
print("=" * 60)

# Fig 6 statistics
fig6_stats = {
    'n_simulations': len(fig6),
    'mean': float(fig6['waveform_difference'].mean()),
    'std': float(fig6['waveform_difference'].std()),
    'median': float(fig6['waveform_difference'].median()),
    'min': float(fig6['waveform_difference'].min()),
    'max': float(fig6['waveform_difference'].max()),
    'q25': float(fig6['waveform_difference'].quantile(0.25)),
    'q75': float(fig6['waveform_difference'].quantile(0.75)),
    'percentile_90': float(fig6['waveform_difference'].quantile(0.90)),
    'percentile_99': float(fig6['waveform_difference'].quantile(0.99)),
    'fraction_below_1e3': float((fig6['waveform_difference'] < 1e-3).mean()),
    'fraction_below_1e4': float((fig6['waveform_difference'] < 1e-4).mean()),
}
print("\nFig 6 (Resolution Error) Statistics:")
for k, v in fig6_stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4e}")
    else:
        print(f"  {k}: {v}")

# Fig 7 statistics by mode
fig7_stats = {}
for col in fig7.columns:
    ell = col.replace('ell', '')
    fig7_stats[f'ell={ell}'] = {
        'mean': float(fig7[col].mean()),
        'std': float(fig7[col].std()),
        'median': float(fig7[col].median()),
        'q25': float(fig7[col].quantile(0.25)),
        'q75': float(fig7[col].quantile(0.75)),
    }

print("\nFig 7 (Modal Error) Statistics:")
for mode, stats in fig7_stats.items():
    print(f"  {mode}: median={stats['median']:.3e}, mean={stats['mean']:.3e}, std={stats['std']:.3e}")

# Fig 8 statistics
fig8_stats = {
    'N2vsN3': {
        'mean': float(fig8['N2vsN3'].mean()),
        'std': float(fig8['N2vsN3'].std()),
        'median': float(fig8['N2vsN3'].median()),
        'q25': float(fig8['N2vsN3'].quantile(0.25)),
        'q75': float(fig8['N2vsN3'].quantile(0.75)),
    },
    'N2vsN4': {
        'mean': float(fig8['N2vsN4'].mean()),
        'std': float(fig8['N2vsN4'].std()),
        'median': float(fig8['N2vsN4'].median()),
        'q25': float(fig8['N2vsN4'].quantile(0.25)),
        'q75': float(fig8['N2vsN4'].quantile(0.75)),
    }
}
print("\nFig 8 (Extrapolation Order) Statistics:")
for comp, stats in fig8_stats.items():
    print(f"  {comp}: median={stats['median']:.3e}, mean={stats['mean']:.3e}, std={stats['std']:.3e}")

# Save statistics as JSON
all_stats = {
    'fig6_resolution_error': fig6_stats,
    'fig7_modal_error': fig7_stats,
    'fig8_extrapolation': fig8_stats
}
with open('outputs/statistical_summary.json', 'w') as f:
    json.dump(all_stats, f, indent=2)
print("\nSaved statistical summary to outputs/statistical_summary.json")

# ============================================================================
# 3. Figure 1: Resolution Error Distribution (Fig 6)
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 1: Resolution Error Distribution...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Histogram
ax = axes[0]
data = fig6['waveform_difference'].values
log_data = np.log10(data)
ax.hist(log_data, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(np.log10(fig6_stats['median']), color='red', linestyle='--', linewidth=2, 
           label=f'Median = {fig6_stats["median"]:.2e}')
ax.axvline(np.log10(fig6_stats['percentile_90']), color='orange', linestyle='--', linewidth=1.5,
           label=f'90th %ile = {fig6_stats["percentile_90"]:.2e}')
ax.axvline(np.log10(fig6_stats['percentile_99']), color='darkred', linestyle=':', linewidth=1.5,
           label=f'99th %ile = {fig6_stats["percentile_99"]:.2e}')
ax.set_xlabel(r'$\log_{10}$(Waveform Difference)')
ax.set_ylabel('Count')
ax.set_title('(a) Resolution Error Distribution')
ax.legend(fontsize=10)

# Panel (b): ECDF
ax = axes[1]
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax.plot(sorted_data, cdf, color='steelblue', linewidth=2)
ax.set_xscale('log')
ax.set_xlabel(r'Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(b) Empirical CDF')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5)
ax.axvline(fig6_stats['median'], color='red', linestyle='--', alpha=0.7, label=f'Median')
ax.axvline(fig6_stats['percentile_90'], color='orange', linestyle='--', alpha=0.7, label=f'90th %ile')
ax.axvline(fig6_stats['percentile_99'], color='darkred', linestyle=':', alpha=0.7, label=f'99th %ile')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig1_resolution_error.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig1_resolution_error.png")

# ============================================================================
# 4. Figure 2: Modal Error Distributions (Fig 7)
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 2: Modal Error Distributions...")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel (a): Box plot
ax = axes[0]
ell_values = [col.replace('ell', '') for col in fig7.columns]
box_data = [fig7[col].values for col in fig7.columns]
bp = ax.boxplot(box_data, labels=[f'ℓ={e}' for e in ell_values], 
                patch_artist=True, showfliers=True, flierprops=dict(markersize=2, alpha=0.3))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(box_data)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel(r'Waveform Difference')
ax.set_title('(a) Modal Error Distribution by Spherical Harmonic Mode')
ax.set_yscale('log')
ax.axhline(1e-3, color='red', linestyle='--', alpha=0.5, label=r'$10^{-3}$ threshold')
ax.legend()

# Panel (b): Median and IQR vs ℓ
ax = axes[1]
ell_nums = [int(e) for e in ell_values]
medians = [fig7[col].median() for col in fig7.columns]
q25s = [fig7[col].quantile(0.25) for col in fig7.columns]
q75s = [fig7[col].quantile(0.75) for col in fig7.columns]
means = [fig7[col].mean() for col in fig7.columns]

ax.errorbar(ell_nums, medians, 
            yerr=[[m - q for m, q in zip(medians, q25s)], [q - m for m, q in zip(medians, q75s)]],
            fmt='o-', color='steelblue', capsize=5, capthick=2, linewidth=2, markersize=8,
            label='Median ± IQR')
ax.plot(ell_nums, means, 's--', color='coral', markersize=7, alpha=0.8, label='Mean')
ax.set_xlabel(r'Spherical Harmonic Mode $\ell$')
ax.set_ylabel(r'Waveform Difference')
ax.set_title('(b) Median and Mean Error vs. Mode Number')
ax.set_yscale('log')
ax.set_xticks(ell_nums)
ax.set_xticklabels([f'ℓ={e}' for e in ell_nums])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_modal_error.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig2_modal_error.png")

# ============================================================================
# 5. Figure 3: Extrapolation Order Comparison (Fig 8)
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 3: Extrapolation Order Comparison...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Side-by-side histograms
ax = axes[0]
log_n23 = np.log10(fig8['N2vsN3'].values)
log_n24 = np.log10(fig8['N2vsN4'].values)
bins = np.linspace(min(min(log_n23), min(log_n24)), max(max(log_n23), max(log_n24)), 40)
ax.hist(log_n23, bins=bins, alpha=0.65, color='steelblue', edgecolor='white', label='N=2 vs N=3')
ax.hist(log_n24, bins=bins, alpha=0.65, color='coral', edgecolor='white', label='N=2 vs N=4')
ax.axvline(np.log10(fig8_stats['N2vsN3']['median']), color='darkblue', linestyle='--', linewidth=1.5)
ax.axvline(np.log10(fig8_stats['N2vsN4']['median']), color='darkred', linestyle='--', linewidth=1.5)
ax.set_xlabel(r'$\log_{10}$(Waveform Difference)')
ax.set_ylabel('Count')
ax.set_title('(a) Extrapolation Error Distributions')
ax.legend()

# Panel (b): ECDF comparison
ax = axes[1]
sorted_n23 = np.sort(fig8['N2vsN3'].values)
sorted_n24 = np.sort(fig8['N2vsN4'].values)
cdf_n23 = np.arange(1, len(sorted_n23) + 1) / len(sorted_n23)
cdf_n24 = np.arange(1, len(sorted_n24) + 1) / len(sorted_n24)
ax.plot(sorted_n23, cdf_n23, color='steelblue', linewidth=2, label='N=2 vs N=3')
ax.plot(sorted_n24, cdf_n24, color='coral', linewidth=2, label='N=2 vs N=4')
ax.set_xscale('log')
ax.set_xlabel(r'Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(b) Empirical CDF Comparison')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_ylim(0, 1.05)
ax.legend()

# Panel (c): Ratio N2vsN4 / N2vsN3
ax = axes[2]
# Match simulation indices
common_idx = fig8.index[:min(len(fig8), len(fig8))]
ratio = fig8['N2vsN4'].values / fig8['N2vsN3'].values
# Remove infinities and very large values for clean plot
valid = np.isfinite(ratio) & (ratio > 0) & (ratio < 100)
ratio_valid = ratio[valid]
ax.hist(np.log10(ratio_valid), bins=40, color='mediumpurple', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Ratio = 1')
ax.axvline(np.log10(np.median(ratio_valid)), color='darkblue', linestyle='--', linewidth=1.5,
           label=f'Median ratio = {np.median(ratio_valid):.2f}')
ax.set_xlabel(r'$\log_{10}$(Ratio: N=2 vs N=4 / N=2 vs N=3)')
ax.set_ylabel('Count')
ax.set_title('(c) Error Ratio Distribution')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig3_extrapolation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig3_extrapolation.png")

# ============================================================================
# 6. Figure 4: Comprehensive Comparison Summary
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 4: Comprehensive Comparison Summary...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel (a): All error distributions together (violin-like comparison)
ax = axes[0, 0]
# Create comparison data
comparison_data = {
    'Resolution\nError': np.log10(fig6['waveform_difference'].values),
    'ℓ=2 Mode\nError': np.log10(fig7['ell2'].values),
    'ℓ=4 Mode\nError': np.log10(fig7['ell4'].values),
    'ℓ=6 Mode\nError': np.log10(fig7['ell6'].values),
    'ℓ=8 Mode\nError': np.log10(fig7['ell8'].values),
    'Extrap.\nN=2v3': np.log10(fig8['N2vsN3'].values),
    'Extrap.\nN=2v4': np.log10(fig8['N2vsN4'].values),
}
positions = range(len(comparison_data))
bp = ax.boxplot(comparison_data.values(), positions=positions, patch_artist=True,
                showfliers=False, widths=0.6)
colors_list = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#E5AE38']
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(comparison_data.keys(), fontsize=9, rotation=0)
ax.set_ylabel(r'$\log_{10}$(Waveform Difference)')
ax.set_title('(a) Error Distribution Comparison Across All Metrics')
ax.axhline(np.log10(1e-3), color='red', linestyle='--', alpha=0.5, label=r'$10^{-3}$')
ax.legend(fontsize=9)

# Panel (b): Resolution error vs extrapolation error
ax = axes[0, 1]
# Resample to create paired comparison (since they have different simulation counts)
np.random.seed(42)
n_common = 500
idx_res = np.random.choice(len(fig6), n_common, replace=False)
idx_ext = np.random.choice(len(fig8), n_common, replace=False)
ax.scatter(fig6['waveform_difference'].values[idx_res], 
           fig8['N2vsN3'].values[idx_ext],
           alpha=0.3, s=15, color='steelblue', label='Resolution vs Extrap. N=2v3')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Resolution Error (Waveform Difference)')
ax.set_ylabel(r'Extrapolation Error N=2 vs N=3')
ax.set_title('(b) Resolution vs Extrapolation Error (Random Sample)')
ax.plot([1e-7, 1], [1e-7, 1], 'k--', alpha=0.3, label='1:1 line')
ax.legend(fontsize=9)

# Panel (c): Cumulative distribution of mode errors
ax = axes[1, 0]
for col, ell, color in zip(fig7.columns, ell_values, colors_list[:7]):
    sorted_vals = np.sort(fig7[col].values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, linewidth=1.5, label=f'ℓ={ell}', color=color)
ax.set_xscale('log')
ax.set_xlabel(r'Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(c) CDF of Modal Errors by ℓ')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax.legend(ncol=2, fontsize=9)

# Panel (d): Key quantitative summary
ax = axes[1, 1]
ax.axis('off')
summary_text = (
    f"KEY QUANTITATIVE SUMMARY\n"
    f"{'='*45}\n\n"
    f"Resolution Error (N_sim = {fig6_stats['n_simulations']}):\n"
    f"  Median:  {fig6_stats['median']:.2e}\n"
    f"  90th %ile: {fig6_stats['percentile_90']:.2e}\n"
    f"  99th %ile: {fig6_stats['percentile_99']:.2e}\n"
    f"  Fraction < 10⁻³: {fig6_stats['fraction_below_1e3']:.1%}\n\n"
    f"Modal Error (median by ℓ):\n"
    f"  ℓ=2: {fig7_stats['ell=2']['median']:.2e}\n"
    f"  ℓ=4: {fig7_stats['ell=4']['median']:.2e}\n"
    f"  ℓ=6: {fig7_stats['ell=6']['median']:.2e}\n"
    f"  ℓ=8: {fig7_stats['ell=8']['median']:.2e}\n\n"
    f"Extrapolation Error:\n"
    f"  N=2 vs N=3 median: {fig8_stats['N2vsN3']['median']:.2e}\n"
    f"  N=2 vs N=4 median: {fig8_stats['N2vsN4']['median']:.2e}\n"
    f"  Ratio (N4/N3): {np.median(ratio_valid):.2f}"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('(d) Summary Statistics')

plt.tight_layout()
plt.savefig('report/images/fig4_comprehensive_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig4_comprehensive_summary.png")

# ============================================================================
# 7. Figure 5: Log-normal fit analysis
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 5: Log-normal Distribution Analysis...")
print("=" * 60)

from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Fit log-normal to fig6
log_fig6 = np.log(fig6['waveform_difference'].values)
mu6, sigma6 = stats.norm.fit(log_fig6)
ax = axes[0]
ax.hist(np.log10(fig6['waveform_difference'].values), bins=50, density=True, 
        alpha=0.7, color='steelblue', edgecolor='white', label='Data')
x_fit = np.linspace(-7, 0, 200)
# Convert log-normal params to log10 scale
mu6_log10 = mu6 / np.log(10)
sigma6_log10 = sigma6 / np.log(10)
pdf_fit = stats.norm.pdf(x_fit, mu6_log10, sigma6_log10)
ax.plot(x_fit, pdf_fit, 'r-', linewidth=2, label=f'Log-normal fit\nμ={mu6_log10:.2f}, σ={sigma6_log10:.2f}')
ax.set_xlabel(r'$\log_{10}$(Waveform Difference)')
ax.set_ylabel('Density')
ax.set_title('(a) Resolution Error: Log-normal Fit')
ax.legend(fontsize=9)

# QQ plot for fig6
ax = axes[1]
stats.probplot(log_fig6, dist="norm", plot=ax)
ax.set_title('(b) Resolution Error: Q-Q Plot')
ax.get_lines()[0].set_markersize(3)
ax.get_lines()[0].set_color('steelblue')
ax.get_lines()[1].set_color('red')

# Shapiro-Wilk test on log data
stat_sw, p_sw = stats.shapiro(log_fig6[:min(5000, len(log_fig6))])
ks_stat, ks_p = stats.kstest(log_fig6, 'norm', args=(mu6, sigma6))

# Summary for fig7 modes
ax = axes[2]
mode_params = []
for col in fig7.columns:
    ell = col.replace('ell', '')
    log_data = np.log(fig7[col].values)
    mu, sigma = stats.norm.fit(log_data)
    mode_params.append((int(ell), mu/np.log(10), sigma/np.log(10)))
    
ell_nums_fit = [p[0] for p in mode_params]
mus = [p[1] for p in mode_params]
sigmas = [p[2] for p in mode_params]

ax.errorbar(ell_nums_fit, mus, yerr=sigmas, fmt='o-', color='steelblue', 
            capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$\mu_{\log}$ ± $\sigma_{\log}$')
ax.set_xlabel(r'Spherical Harmonic Mode $\ell$')
ax.set_ylabel(r'Log-normal Parameters (log₁₀ scale)')
ax.set_title('(c) Log-normal Fit Parameters vs. Mode ℓ')
ax.set_xticks(ell_nums_fit)
ax.set_xticklabels([f'ℓ={e}' for e in ell_nums_fit])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_lognormal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig5_lognormal_analysis.png")

# Save log-normal fit results
fit_results = {
    'resolution_error': {
        'mu_log10': float(mu6_log10),
        'sigma_log10': float(sigma6_log10),
        'shapiro_wilk_stat': float(stat_sw),
        'shapiro_wilk_p': float(p_sw),
        'ks_stat': float(ks_stat),
        'ks_p': float(ks_p),
    },
    'modal_error_fits': [
        {'ell': int(ell), 'mu_log10': float(mu), 'sigma_log10': float(sig)}
        for ell, mu, sig in mode_params
    ]
}
with open('outputs/lognormal_fit_results.json', 'w') as f:
    json.dump(fit_results, f, indent=2)
print("Saved outputs/lognormal_fit_results.json")

# ============================================================================
# 8. Figure 6: Physical Interpretation - Error Budget
# ============================================================================
print("\n" + "=" * 60)
print("Generating Figure 6: Error Budget Analysis...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Error sources comparison
ax = axes[0]
categories = ['Resolution\n(σ_trunc)', 'ℓ=2\nMode', 'ℓ=4\nMode', 'ℓ=6\nMode', 'ℓ=8\nMode', 
              'Extrap.\nN=2v3', 'Extrap.\nN=2v4']
medians_bar = [
    fig6_stats['median'],
    fig7_stats['ell=2']['median'],
    fig7_stats['ell=4']['median'],
    fig7_stats['ell=6']['median'],
    fig7_stats['ell=8']['median'],
    fig8_stats['N2vsN3']['median'],
    fig8_stats['N2vsN4']['median'],
]
colors_bar = ['#4C72B0', '#55A868', '#55A868', '#55A868', '#55A868', '#C44E52', '#C44E52']
bars = ax.bar(categories, medians_bar, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=1.5)
ax.set_ylabel(r'Median Waveform Difference')
ax.set_title('(a) Median Error by Source')
ax.set_yscale('log')
ax.axhline(1e-3, color='red', linestyle='--', alpha=0.5, label=r'$10^{-3}$ (LIGO threshold)')
for bar, val in zip(bars, medians_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
            f'{val:.1e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(axis='x', labelsize=9)

# Panel (b): Fraction of simulations meeting accuracy thresholds
ax = axes[1]
thresholds = np.logspace(-6, -1, 50)
fractions_resolution = [(fig6['waveform_difference'] < t).mean() for t in thresholds]
fractions_ell2 = [(fig7['ell2'] < t).mean() for t in thresholds]
fractions_ell6 = [(fig7['ell6'] < t).mean() for t in thresholds]
fractions_extrap_n23 = [(fig8['N2vsN3'] < t).mean() for t in thresholds]

ax.plot(thresholds, fractions_resolution, 'o-', color='#4C72B0', linewidth=2, markersize=3, label='Resolution Error')
ax.plot(thresholds, fractions_ell2, 's-', color='#55A868', linewidth=2, markersize=3, label='ℓ=2 Mode Error')
ax.plot(thresholds, fractions_ell6, '^-', color='#8172B2', linewidth=2, markersize=3, label='ℓ=6 Mode Error')
ax.plot(thresholds, fractions_extrap_n23, 'D-', color='#C44E52', linewidth=2, markersize=3, label='Extrap. N=2v3')
ax.set_xscale('log')
ax.set_xlabel(r'Accuracy Threshold')
ax.set_ylabel('Fraction of Simulations Below Threshold')
ax.set_title('(b) Fraction Meeting Accuracy Threshold')
ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% line')
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('report/images/fig6_error_budget.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved report/images/fig6_error_budget.png")

# ============================================================================
# 9. Save all quantitative results
# ============================================================================
print("\n" + "=" * 60)
print("Saving final results...")
print("=" * 60)

# Save a comprehensive results table
results_table = {
    'metric': [],
    'n_simulations': [],
    'median': [],
    'mean': [],
    'std': [],
    'q25': [],
    'q75': [],
}

# Add resolution error
results_table['metric'].append('Resolution Error')
results_table['n_simulations'].append(fig6_stats['n_simulations'])
results_table['median'].append(fig6_stats['median'])
results_table['mean'].append(fig6_stats['mean'])
results_table['std'].append(fig6_stats['std'])
results_table['q25'].append(fig6_stats['q25'])
results_table['q75'].append(fig6_stats['q75'])

# Add modal errors
for col in fig7.columns:
    ell = col.replace('ell', '')
    results_table['metric'].append(f'Mode ℓ={ell} Error')
    results_table['n_simulations'].append(len(fig7))
    results_table['median'].append(fig7_stats[f'ell={ell}']['median'])
    results_table['mean'].append(fig7_stats[f'ell={ell}']['mean'])
    results_table['std'].append(fig7_stats[f'ell={ell}']['std'])
    results_table['q25'].append(fig7_stats[f'ell={ell}']['q25'])
    results_table['q75'].append(fig7_stats[f'ell={ell}']['q75'])

# Add extrapolation errors
for comp in ['N2vsN3', 'N2vsN4']:
    results_table['metric'].append(f'Extrapolation {comp}')
    results_table['n_simulations'].append(len(fig8))
    results_table['median'].append(fig8_stats[comp]['median'])
    results_table['mean'].append(fig8_stats[comp]['mean'])
    results_table['std'].append(fig8_stats[comp]['std'])
    results_table['q25'].append(fig8_stats[comp]['q25'])
    results_table['q75'].append(fig8_stats[comp]['q75'])

results_df = pd.DataFrame(results_table)
results_df.to_csv('outputs/comprehensive_results_table.csv', index=False)
print("Saved outputs/comprehensive_results_table.csv")

print("\n" + "=" * 60)
print("ALL ANALYSIS COMPLETE")
print("=" * 60)
print(f"Generated figures:")
for f in sorted(os.listdir('report/images')):
    if f.endswith('.png'):
        print(f"  - report/images/{f}")
print(f"Generated outputs:")
for f in sorted(os.listdir('outputs')):
    print(f"  - outputs/{f}")
