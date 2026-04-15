#!/usr/bin/env python3
"""
Analysis of SXS Binary Black Hole Waveform Catalog Accuracy Metrics.

This script analyzes three datasets:
1. fig6_data.csv - Overall waveform differences between highest resolutions
2. fig7_data.csv - Mode-by-mode waveform differences (ell=2 through ell=8)
3. fig8_data.csv - Extrapolation order comparisons (N2vsN3, N2vsN4)

Generates comprehensive statistics and publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import json
import os

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

fig6 = pd.read_csv('data/fig6_data.csv')
fig7 = pd.read_csv('data/fig7_data.csv')
fig8 = pd.read_csv('data/fig8_data.csv')

fig6_vals = fig6['waveform_difference'].values
fig7_vals = {f'ell{ell}': fig7[f'ell{ell}'].values for ell in range(2, 9)}
fig8_n2n3 = fig8['N2vsN3'].values
fig8_n2n4 = fig8['N2vsN4'].values

print(f"Fig6: {len(fig6_vals)} simulations")
print(f"Fig7: {len(fig7_vals['ell2'])} simulations, modes ell=2..8")
print(f"Fig8: {len(fig8_n2n3)} simulations, 2 extrapolation comparisons")

# ============================================================
# 2. Descriptive Statistics
# ============================================================
print("\n" + "=" * 60)
print("Descriptive Statistics")
print("=" * 60)

def compute_stats(name, values):
    """Compute descriptive statistics for a dataset."""
    log_vals = np.log10(values[values > 0])
    return {
        'name': name,
        'count': len(values),
        'median': float(np.median(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'p5': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'log10_median': float(np.median(log_vals)),
        'log10_mean': float(np.mean(log_vals)),
        'log10_std': float(np.std(log_vals)),
    }

stats_fig6 = compute_stats('fig6_overall', fig6_vals)
stats_fig7 = {}
for ell in range(2, 9):
    col = f'ell{ell}'
    stats_fig7[col] = compute_stats(col, fig7_vals[col])

stats_fig8_n2n3 = compute_stats('fig8_N2vsN3', fig8_n2n3)
stats_fig8_n2n4 = compute_stats('fig8_N2vsN4', fig8_n2n4)

# Print summary
print("\n--- Fig6: Overall Waveform Differences ---")
for k, v in stats_fig6.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6e}")
    else:
        print(f"  {k}: {v}")

print("\n--- Fig7: Mode-by-Mode Waveform Differences ---")
for ell in range(2, 9):
    col = f'ell{ell}'
    print(f"  ell={ell}: median={stats_fig7[col]['median']:.4e}, "
          f"mean={stats_fig7[col]['mean']:.4e}, std={stats_fig7[col]['std']:.4e}")

print("\n--- Fig8: Extrapolation Order Comparisons ---")
print(f"  N2vsN3: median={stats_fig8_n2n3['median']:.4e}, mean={stats_fig8_n2n3['mean']:.4e}")
print(f"  N2vsN4: median={stats_fig8_n2n4['median']:.4e}, mean={stats_fig8_n2n4['mean']:.4e}")

# Save all statistics
all_stats = {
    'fig6': stats_fig6,
    'fig7': stats_fig7,
    'fig8_N2vsN3': stats_fig8_n2n3,
    'fig8_N2vsN4': stats_fig8_n2n4,
}
with open('outputs/descriptive_statistics.json', 'w') as f:
    json.dump(all_stats, f, indent=2)
print("\nSaved descriptive statistics to outputs/descriptive_statistics.json")

# ============================================================
# 3. Distribution Fitting and Validation
# ============================================================
print("\n" + "=" * 60)
print("Distribution Fitting (Log-Normal Validation)")
print("=" * 60)

def fit_lognormal(name, values):
    """Fit log-normal distribution and perform KS test."""
    # Fit log-normal: scipy uses shape parameter s, loc, scale
    # For log-normal, log(X) ~ Normal(mu, sigma)
    params = stats.lognorm.fit(values, floc=0)
    shape, loc, scale = params
    
    # KS test
    ks_stat, ks_pvalue = stats.kstest(values, 'lognorm', args=params)
    
    # Also fit normal to log-values
    log_vals = np.log(values)
    mu, sigma = stats.norm.fit(log_vals)
    
    # Anderson-Darling test on log-values
    ad_result = stats.anderson(log_vals, dist='norm')
    
    result = {
        'name': name,
        'lognorm_shape': float(shape),
        'lognorm_scale': float(scale),
        'log_mu': float(mu),
        'log_sigma': float(sigma),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'ad_statistic': float(ad_result.statistic),
        'ad_critical_5pct': float(ad_result.critical_values[2]),
        'ad_pass_5pct': bool(ad_result.statistic < ad_result.critical_values[2]),
    }
    print(f"\n  {name}:")
    print(f"    Log-normal params: shape={shape:.4f}, scale={scale:.4e}")
    print(f"    Log-space: mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"    KS test: stat={ks_stat:.4f}, p={ks_pvalue:.4f}")
    print(f"    AD test: stat={ad_result.statistic:.4f}, passes 5%: {result['ad_pass_5pct']}")
    return result

fit_fig6 = fit_lognormal('fig6_overall', fig6_vals)
fit_fig8_n2n3 = fit_lognormal('fig8_N2vsN3', fig8_n2n3)
fit_fig8_n2n4 = fit_lognormal('fig8_N2vsN4', fig8_n2n4)

dist_fits = {
    'fig6': fit_fig6,
    'fig8_N2vsN3': fit_fig8_n2n3,
    'fig8_N2vsN4': fit_fig8_n2n4,
}
with open('outputs/distribution_fits.json', 'w') as f:
    json.dump(dist_fits, f, indent=2)

# ============================================================
# 4. Generate Figures
# ============================================================
print("\n" + "=" * 60)
print("Generating Figures")
print("=" * 60)

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# ----------------------------------------------------------
# Figure 1: Overall Waveform Difference Distribution (Fig6)
# ----------------------------------------------------------
print("  Generating Figure 1: Overall waveform difference distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Histogram in linear space (zoomed)
ax = axes[0]
mask_zoom = fig6_vals < 0.01
ax.hist(fig6_vals[mask_zoom], bins=80, color='#2166ac', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Count')
ax.set_title('(a) Linear Scale (zoomed)')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, -4))
ax.axvline(stats_fig6['median'], color='#d6604d', linestyle='--', linewidth=1.5,
           label=f'Median = {stats_fig6["median"]:.2e}')
ax.legend(loc='upper right', fontsize=8)

# Panel B: Histogram in log space
ax = axes[1]
log_vals = np.log10(fig6_vals[fig6_vals > 0])
ax.hist(log_vals, bins=60, color='#2166ac', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.set_xlabel(r'$\log_{10}$ (Waveform Difference)')
ax.set_ylabel('Count')
ax.set_title('(b) Log$_{10}$ Scale')
ax.axvline(stats_fig6['log10_median'], color='#d6604d', linestyle='--', linewidth=1.5,
           label=f'Median = {stats_fig6["log10_median"]:.2f}')
ax.legend(loc='upper right', fontsize=8)

# Panel C: CDF
ax = axes[2]
sorted_vals = np.sort(fig6_vals)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
ax.semilogx(sorted_vals, cdf, color='#2166ac', linewidth=1.5)
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(c) Cumulative Distribution')
ax.set_xlim(1e-6, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
# Mark key thresholds
for thresh in [1e-4, 1e-3, 1e-2]:
    frac_below = np.sum(sorted_vals < thresh) / len(sorted_vals)
    ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
    ax.text(thresh * 1.5, frac_below + 0.03, f'{frac_below:.1%}', fontsize=8, rotation=0)

plt.suptitle('SXS Catalog Waveform Accuracy: Overall Resolution Error', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig1_overall_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig1_overall_distribution.png")

# ----------------------------------------------------------
# Figure 2: Modal Error Decomposition by ell (Fig7)
# ----------------------------------------------------------
print("  Generating Figure 2: Modal error decomposition...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Violin/Box plot by mode
ax = axes[0]
ell_values = []
ell_labels = []
for ell in range(2, 9):
    col = f'ell{ell}'
    ell_values.append(np.log10(fig7_vals[col][fig7_vals[col] > 0]))
    ell_labels.append(f'ℓ={ell}')

bp = ax.boxplot(ell_values, labels=ell_labels, patch_artist=True, 
                medianprops=dict(color='#d6604d', linewidth=2),
                boxprops=dict(facecolor='#2166ac', alpha=0.6),
                whiskerprops=dict(color='#2166ac'),
                capprops=dict(color='#2166ac'))
ax.set_xlabel('Spherical Harmonic Mode ℓ')
ax.set_ylabel(r'$\log_{10}$ (Waveform Difference)')
ax.set_title('(a) Distribution by Mode')
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Median error vs ell
ax = axes[1]
ell_nums = list(range(2, 9))
medians = [stats_fig7[f'ell{ell}']['median'] for ell in ell_nums]
means = [stats_fig7[f'ell{ell}']['mean'] for ell in ell_nums]
p25s = [stats_fig7[f'ell{ell}']['p25'] for ell in ell_nums]
p75s = [stats_fig7[f'ell{ell}']['p75'] for ell in ell_nums]

ax.errorbar(ell_nums, medians, fmt='o', color='#2166ac', markersize=8, 
            capsize=4, label='Median')
ax.fill_between(ell_nums, p25s, p75s, alpha=0.2, color='#2166ac', label='IQR')
ax.set_yscale('log')
ax.set_xlabel('Spherical Harmonic Mode ℓ')
ax.set_ylabel('Median Waveform Difference')
ax.set_title('(b) Median Error vs Mode')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# Panel C: Full distribution per mode (overlaid histograms)
ax = axes[2]
colors_mode = plt.cm.viridis(np.linspace(0.1, 0.9, 7))
for i, ell in enumerate(ell_nums):
    col = f'ell{ell}'
    vals = fig7_vals[col][fig7_vals[col] > 0]
    log_v = np.log10(vals)
    ax.hist(log_v, bins=40, alpha=0.4, color=colors_mode[i], 
            label=f'ℓ={ell}', density=True, histtype='stepfilled', linewidth=0.5)
ax.set_xlabel(r'$\log_{10}$ (Waveform Difference)')
ax.set_ylabel('Density')
ax.set_title('(c) Density by Mode')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.suptitle('SXS Catalog Waveform Accuracy: Modal Error Decomposition', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig2_modal_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig2_modal_decomposition.png")

# ----------------------------------------------------------
# Figure 3: Extrapolation Order Comparison (Fig8)
# ----------------------------------------------------------
print("  Generating Figure 3: Extrapolation order comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Overlaid histograms
ax = axes[0]
ax.hist(np.log10(fig8_n2n3[fig8_n2n3 > 0]), bins=50, alpha=0.5, color='#2166ac', 
        label='N=2 vs N=3', density=True)
ax.hist(np.log10(fig8_n2n4[fig8_n2n4 > 0]), bins=50, alpha=0.5, color='#d6604d', 
        label='N=2 vs N=4', density=True)
ax.set_xlabel(r'$\log_{10}$ (Waveform Difference)')
ax.set_ylabel('Density')
ax.set_title('(a) Distribution Comparison')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
# Mark medians
ax.axvline(stats_fig8_n2n3['log10_median'], color='#2166ac', linestyle='--', linewidth=1.5)
ax.axvline(stats_fig8_n2n4['log10_median'], color='#d6604d', linestyle='--', linewidth=1.5)

# Panel B: Scatter / paired comparison
ax = axes[1]
ax.scatter(fig8_n2n3, fig8_n2n4, alpha=0.3, s=8, color='#4d4d4d', edgecolors='none')
lim_min = min(fig8_n2n3.min(), fig8_n2n4.min()) * 0.5
lim_max = max(fig8_n2n3.max(), fig8_n2n4.max()) * 1.2
ax.loglog([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('N=2 vs N=3 Waveform Difference')
ax.set_ylabel('N=2 vs N=4 Waveform Difference')
ax.set_title('(b) Paired Comparison')
ax.grid(True, alpha=0.3, which='both')

# Panel C: CDF comparison
ax = axes[2]
sorted_n2n3 = np.sort(fig8_n2n3)
sorted_n2n4 = np.sort(fig8_n2n4)
cdf_n2n3 = np.arange(1, len(sorted_n2n3) + 1) / len(sorted_n2n3)
cdf_n2n4 = np.arange(1, len(sorted_n2n4) + 1) / len(sorted_n2n4)
ax.semilogx(sorted_n2n3, cdf_n2n3, color='#2166ac', linewidth=1.5, label='N=2 vs N=3')
ax.semilogx(sorted_n2n4, cdf_n2n4, color='#d6604d', linewidth=1.5, label='N=2 vs N=4')
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('(c) Cumulative Distribution')
ax.set_xlim(1e-7, 1e-1)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('SXS Catalog Waveform Accuracy: Extrapolation Order Convergence', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig3_extrapolation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig3_extrapolation_comparison.png")

# ----------------------------------------------------------
# Figure 4: Accuracy Summary & Threshold Analysis
# ----------------------------------------------------------
print("  Generating Figure 4: Accuracy summary and threshold analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Fraction below various thresholds
ax = axes[0]
thresholds = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
datasets = {
    'Overall (Fig6)': fig6_vals,
    'ℓ=2 mode': fig7_vals['ell2'],
    'ℓ=4 mode': fig7_vals['ell4'],
    'ℓ=8 mode': fig7_vals['ell8'],
    'N2vsN3': fig8_n2n3,
    'N2vsN4': fig8_n2n4,
}
colors_thresh = ['#2166ac', '#67a9cf', '#d1e5f0', '#fdcbc8', '#ef8a62', '#d6604d']

for idx, (name, vals) in enumerate(datasets.items()):
    fracs = [np.sum(vals < t) / len(vals) for t in thresholds]
    ax.semilogx(thresholds, fracs, 'o-', color=colors_thresh[idx % len(colors_thresh)],
                linewidth=1.5, markersize=6, label=name)

ax.set_xlabel('Waveform Difference Threshold')
ax.set_ylabel('Fraction Below Threshold')
ax.set_title('(a) Accuracy Compliance by Threshold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0, 1.02)

# Panel B: Heatmap of median error by mode
ax = axes[1]
ell_nums = list(range(2, 9))
median_matrix = np.array([stats_fig7[f'ell{ell}']['median'] for ell in ell_nums])
im = ax.bar(ell_nums, median_matrix, color=plt.cm.viridis(np.linspace(0.2, 0.8, 7)),
            edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
ax.set_xlabel('Spherical Harmonic Mode ℓ')
ax.set_ylabel('Median Waveform Difference')
ax.set_title('(b) Median Error Growth with Mode')
ax.grid(True, alpha=0.3, axis='y')
# Add value labels
for rect, val in zip(im, median_matrix):
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() * 1.3,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.suptitle('SXS Catalog Waveform Accuracy: Summary Analysis', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_accuracy_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig4_accuracy_summary.png")

# ----------------------------------------------------------
# Figure 5: Q-Q Plots and Validation
# ----------------------------------------------------------
print("  Generating Figure 5: Distribution validation (Q-Q plots)...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Q-Q plots for log-normal fit
qq_datasets = [
    ('Overall (Fig6)', fig6_vals, fit_fig6),
    ('N=2 vs N=3', fig8_n2n3, fit_fig8_n2n3),
    ('N=2 vs N=4', fig8_n2n4, fit_fig8_n2n4),
]

for idx, (name, vals, fit) in enumerate(qq_datasets):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    log_v = np.log(vals[vals > 0])
    stats.probplot(log_v, dist="norm", plot=ax)
    ax.set_title(f'(a-c) {name}\nKS p={fit["ks_pvalue"]:.3f}, AD pass={fit["ad_pass_5pct"]}')
    ax.set_ylabel('Sample Quantiles')
    ax.set_xlabel('Theoretical Quantiles')
    ax.grid(True, alpha=0.3)

# Residual distributions
for idx, (name, vals, fit) in enumerate(qq_datasets):
    row, col = 1, idx
    ax = axes[row, col]
    log_v = np.log(vals[vals > 0])
    residuals = log_v - fit['log_mu']
    ax.hist(residuals, bins=50, alpha=0.7, color='#2166ac', edgecolor='white', density=True)
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm, 0, fit['log_sigma']), 'r-', linewidth=2,
            label=f'σ={fit["log_sigma"]:.3f}')
    ax.set_xlabel('Residual from Log-Normal Fit')
    ax.set_ylabel('Density')
    ax.set_title(f'(d-f) {name} Residuals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('SXS Catalog Waveform Accuracy: Distribution Validation', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig5_distribution_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig5_distribution_validation.png")

# ----------------------------------------------------------
# Figure 6: Comprehensive comparison panel
# ----------------------------------------------------------
print("  Generating Figure 6: Comprehensive accuracy overview...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: All distributions overlaid
ax = axes[0, 0]
all_distributions = {
    'Overall Resolution': fig6_vals,
    'ℓ=2': fig7_vals['ell2'],
    'ℓ=5': fig7_vals['ell5'],
    'ℓ=8': fig7_vals['ell8'],
}
colors_comp = ['#2166ac', '#67a9cf', '#ef8a62', '#d6604d']
for idx, (name, vals) in enumerate(all_distributions.items()):
    log_v = np.log10(vals[vals > 0])
    ax.hist(log_v, bins=50, alpha=0.4, color=colors_comp[idx], 
            label=name, density=True, histtype='step', linewidth=2)
ax.set_xlabel(r'$\log_{10}$ (Waveform Difference)')
ax.set_ylabel('Density')
ax.set_title('Key Distributions')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top-right: Modal trend with error bars
ax = axes[0, 1]
ell_nums = list(range(2, 9))
medians = [stats_fig7[f'ell{ell}']['median'] for ell in ell_nums]
p10s = [stats_fig7[f'ell{ell}']['p10'] if 'p10' in stats_fig7[f'ell{ell}'] 
        else float(np.percentile(fig7_vals[f'ell{ell}'], 10)) for ell in ell_nums]
p90s = [stats_fig7[f'ell{ell}']['p90'] if 'p90' in stats_fig7[f'ell{ell}']
        else float(np.percentile(fig7_vals[f'ell{ell}'], 90)) for ell in ell_nums]
ax.errorbar(ell_nums, medians, yerr=[np.array(medians) - np.array(p10s), 
                                       np.array(p90s) - np.array(medians)],
            fmt='o-', color='#2166ac', capsize=4, markersize=8, linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Spherical Harmonic Mode ℓ')
ax.set_ylabel('Waveform Difference')
ax.set_title('Modal Error Trend (10th-90th percentile)')
ax.grid(True, alpha=0.3, which='both')

# Bottom-left: Extrapolation convergence ratio
ax = axes[1, 0]
ratio = fig8_n2n4 / fig8_n2n3
valid_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
ax.hist(np.log10(valid_ratio), bins=50, color='#2166ac', alpha=0.7, edgecolor='white')
ax.axvline(np.log10(np.median(valid_ratio)), color='#d6604d', linestyle='--', linewidth=2,
           label=f'Median ratio = {np.median(valid_ratio):.2f}')
ax.set_xlabel(r'$\log_{10}$ (N2vsN4 / N2vsN3)')
ax.set_ylabel('Count')
ax.set_title('Extrapolation Convergence Ratio')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom-right: Accuracy budget pie-like breakdown
ax = axes[1, 1]
# Fraction of simulations meeting different accuracy levels
accuracy_levels = ['< 10⁻⁴', '10⁻⁴–10⁻³', '10⁻³–10⁻²', '> 10⁻²']
fracs_level = [
    np.sum(fig6_vals < 1e-4) / len(fig6_vals),
    np.sum((fig6_vals >= 1e-4) & (fig6_vals < 1e-3)) / len(fig6_vals),
    np.sum((fig6_vals >= 1e-3) & (fig6_vals < 1e-2)) / len(fig6_vals),
    np.sum(fig6_vals >= 1e-2) / len(fig6_vals),
]
colors_pie = ['#2166ac', '#67a9cf', '#ef8a62', '#d6604d']
wedges, texts, autotexts = ax.pie(fracs_level, labels=accuracy_levels, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90,
                                    textprops={'fontsize': 10})
ax.set_title('Overall Accuracy Budget\n(Fraction of Simulations)')

plt.suptitle('SXS Catalog Waveform Accuracy: Comprehensive Overview', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig6_comprehensive_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved report/images/fig6_comprehensive_overview.png")

# ============================================================
# 5. Save Key Results Tables
# ============================================================
print("\n" + "=" * 60)
print("Saving Result Tables")
print("=" * 60)

# Table: Modal statistics
modal_table = []
for ell in range(2, 9):
    col = f'ell{ell}'
    s = stats_fig7[col]
    modal_table.append({
        'ell': ell,
        'median': s['median'],
        'mean': s['mean'],
        'std': s['std'],
        'p25': s['p25'],
        'p75': s['p75'],
        'log10_median': s['log10_median'],
    })
modal_df = pd.DataFrame(modal_table)
modal_df.to_csv('outputs/modal_statistics.csv', index=False)
print("Saved outputs/modal_statistics.csv")

# Table: Threshold compliance
threshold_table = []
for name, vals in datasets.items():
    row = {'dataset': name, 'count': len(vals)}
    for t in thresholds:
        row[f'below_{t:.0e}'] = float(np.sum(vals < t) / len(vals))
    threshold_table.append(row)
threshold_df = pd.DataFrame(threshold_table)
threshold_df.to_csv('outputs/threshold_compliance.csv', index=False)
print("Saved outputs/threshold_compliance.csv")

# Table: Extrapolation comparison
extrap_table = pd.DataFrame({
    'metric': ['median', 'mean', 'std', 'p25', 'p75', 'log10_median'],
    'N2vsN3': [stats_fig8_n2n3[k] for k in ['median', 'mean', 'std', 'p25', 'p75', 'log10_median']],
    'N2vsN4': [stats_fig8_n2n4[k] for k in ['median', 'mean', 'std', 'p25', 'p75', 'log10_median']],
})
extrap_table.to_csv('outputs/extrapolation_comparison.csv', index=False)
print("Saved outputs/extrapolation_comparison.csv")

# Summary JSON
summary = {
    'overall_median_waveform_diff': stats_fig6['median'],
    'overall_fraction_below_1e3': float(np.sum(fig6_vals < 1e-3) / len(fig6_vals)),
    'overall_fraction_below_1e4': float(np.sum(fig6_vals < 1e-4) / len(fig6_vals)),
    'modal_medians': {f'ell{ell}': stats_fig7[f'ell{ell}']['median'] for ell in range(2, 9)},
    'extrapolation_medians': {
        'N2vsN3': stats_fig8_n2n3['median'],
        'N2vsN4': stats_fig8_n2n4['median'],
    },
    'extrapolation_convergence_ratio': float(np.median(fig8_n2n4 / fig8_n2n3)),
    'log_normal_fit_quality': {
        'fig6_ks_pvalue': fit_fig6['ks_pvalue'],
        'fig6_ad_passes': fit_fig6['ad_pass_5pct'],
    },
}
with open('outputs/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved outputs/analysis_summary.json")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
