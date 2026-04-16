"""
SXS BBH Catalog Accuracy Assessment — Main Analysis Script
Analyzes waveform difference data to assess numerical accuracy of the SXS catalog.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})

# Paths
DATA_DIR = '../data'
OUTPUT_DIR = '../outputs'
IMAGE_DIR = '../report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
fig6 = pd.read_csv(os.path.join(DATA_DIR, 'fig6_data.csv'))
fig7 = pd.read_csv(os.path.join(DATA_DIR, 'fig7_data.csv'))
fig8 = pd.read_csv(os.path.join(DATA_DIR, 'fig8_data.csv'))

print(f"fig6 shape: {fig6.shape}")
print(f"fig7 shape: {fig7.shape}")
print(f"fig8 shape: {fig8.shape}")

# ============================================================
# 2. Summary Statistics
# ============================================================
print("\n=== Fig 6: Overall Waveform Differences (Resolution Error) ===")
fig6_vals = fig6['waveform_difference'].values
fig6_log = np.log10(fig6_vals)
stats_fig6 = {
    'n_simulations': int(len(fig6_vals)),
    'mean': float(np.mean(fig6_vals)),
    'median': float(np.median(fig6_vals)),
    'std': float(np.std(fig6_vals)),
    'min': float(np.min(fig6_vals)),
    'max': float(np.max(fig6_vals)),
    'p5': float(np.percentile(fig6_vals, 5)),
    'p25': float(np.percentile(fig6_vals, 25)),
    'p75': float(np.percentile(fig6_vals, 75)),
    'p95': float(np.percentile(fig6_vals, 95)),
    'log10_mean': float(np.mean(fig6_log)),
    'log10_median': float(np.median(fig6_log)),
    'log10_std': float(np.std(fig6_log)),
    'fraction_below_1e3': float(np.mean(fig6_vals < 1e-3)),
    'fraction_below_1e4': float(np.mean(fig6_vals < 1e-4)),
    'fraction_below_1e2': float(np.mean(fig6_vals < 1e-2)),
}
for k, v in stats_fig6.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6e}" if v < 0.01 else f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n=== Fig 7: Modal Errors by ℓ ===")
ell_labels = ['ell2', 'ell3', 'ell4', 'ell5', 'ell6', 'ell7', 'ell8']
ell_values = [2, 3, 4, 5, 6, 7, 8]
stats_fig7 = {}
for col, ell in zip(ell_labels, ell_values):
    vals = fig7[col].values
    log_vals = np.log10(vals)
    stats_fig7[ell] = {
        'median': float(np.median(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'log10_median': float(np.median(log_vals)),
        'log10_mean': float(np.mean(log_vals)),
        'log10_std': float(np.std(log_vals)),
        'p5': float(np.percentile(vals, 5)),
        'p95': float(np.percentile(vals, 95)),
    }
    print(f"  ℓ={ell}: median={np.median(vals):.3e}, log10_median={np.median(log_vals):.3f}")

print("\n=== Fig 8: Extrapolation Order Comparison ===")
n2vsn3 = fig8['N2vsN3'].values
n2vsn4 = fig8['N2vsN4'].values
stats_fig8 = {
    'n_simulations': int(len(n2vsn3)),
    'N2vsN3_median': float(np.median(n2vsn3)),
    'N2vsN3_mean': float(np.mean(n2vsn3)),
    'N2vsN3_std': float(np.std(n2vsn3)),
    'N2vsN3_log10_median': float(np.median(np.log10(n2vsn3))),
    'N2vsN4_median': float(np.median(n2vsn4)),
    'N2vsN4_mean': float(np.mean(n2vsn4)),
    'N2vsN4_std': float(np.std(n2vsn4)),
    'N2vsN4_log10_median': float(np.median(np.log10(n2vsn4))),
    'ratio_median': float(np.median(n2vsn4 / n2vsn3)),
    'fraction_N2vsN4_larger': float(np.mean(n2vsn4 > n2vsn3)),
}
for k, v in stats_fig8.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6e}" if v < 0.01 else f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Save summary stats
all_stats = {'fig6': stats_fig6, 'fig7': stats_fig7, 'fig8': stats_fig8}
with open(os.path.join(OUTPUT_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(all_stats, f, indent=2)
print("\nSummary statistics saved.")

# ============================================================
# 3. Figure 1: Overall Waveform Difference Distribution
# ============================================================
print("\nGenerating Figure 1: Overall waveform difference histogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: log-scale histogram
ax = axes[0]
ax.hist(fig6_log, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(np.log10(4e-4), color='red', linestyle='--', linewidth=1.5, label='Expected median\n(4×10⁻⁴)')
ax.axvline(np.median(fig6_log), color='orange', linestyle='-', linewidth=1.5, label=f'Actual median\n({10**np.median(fig6_log):.2e})')
ax.set_xlabel('log₁₀(Waveform Difference)')
ax.set_ylabel('Number of Simulations')
ax.set_title('Distribution of Resolution-Induced Waveform Differences')
ax.legend(fontsize=9)

# Right: CDF
ax = axes[1]
sorted_vals = np.sort(fig6_vals)
cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
ax.semilogx(sorted_vals, cdf, color='steelblue', linewidth=2)
for thresh, label in [(1e-2, '1e-2'), (1e-3, '1e-3'), (1e-4, '1e-4')]:
    frac = np.mean(fig6_vals < thresh)
    ax.axvline(thresh, color='gray', linestyle=':', alpha=0.6)
    ax.text(thresh*1.5, 0.05, f'{frac:.1%} < {label}', fontsize=9, color='gray')
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('Cumulative Distribution of Waveform Differences')
ax.set_xlim(1e-7, 1)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig1_waveform_difference_distribution.png'), bbox_inches='tight')
plt.close()
print("  Figure 1 saved.")

# ============================================================
# 4. Figure 2: Modal Error by ℓ (Box + Violin)
# ============================================================
print("Generating Figure 2: Modal errors by ℓ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Box plot on log scale
ax = axes[0]
log_fig7 = np.log10(fig7)
bp = ax.boxplot([log_fig7[col].values for col in ell_labels],
                labels=[f'ℓ={ell}' for ell in ell_values],
                patch_artist=True,
                medianprops=dict(color='red', linewidth=1.5))
colors = sns.color_palette("Blues_d", 7)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('log₁₀(Waveform Difference)')
ax.set_title('Modal Error Distribution by ℓ')
ax.grid(axis='y', alpha=0.3)

# Right: Violin plot
ax = axes[1]
log_melted = log_fig7.melt(var_name='Mode', value_name='log10_diff')
log_melted['ℓ'] = log_melted['Mode'].map(dict(zip(ell_labels, ell_values)))
sns.violinplot(data=log_melted, x='ℓ', y='log10_diff', palette='Blues_d', inner='quartile', ax=ax)
ax.set_ylabel('log₁₀(Waveform Difference)')
ax.set_xlabel('ℓ (Spherical Harmonic Mode)')
ax.set_title('Modal Error Violin Plot by ℓ')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig2_modal_errors_by_ell.png'), bbox_inches='tight')
plt.close()
print("  Figure 2 saved.")

# ============================================================
# 5. Figure 3: Extrapolation Order Comparison
# ============================================================
print("Generating Figure 3: Extrapolation order comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Scatter plot
ax = axes[0]
ax.scatter(np.log10(n2vsn3), np.log10(n2vsn4), alpha=0.3, s=12, color='steelblue', edgecolors='none')
lims = [min(np.log10(n2vsn3).min(), np.log10(n2vsn4).min())-0.2,
        max(np.log10(n2vsn3).max(), np.log10(n2vsn4).max())+0.2]
ax.plot(lims, lims, 'r--', alpha=0.5, label='1:1 line')
ax.set_xlabel('log₁₀(Diff N2 vs N3)')
ax.set_ylabel('log₁₀(Diff N2 vs N4)')
ax.set_title('Extrapolation Order Comparison')
ax.legend()
ax.set_aspect('equal')

# Right: Overlapping histograms
ax = axes[1]
ax.hist(np.log10(n2vsn3), bins=50, alpha=0.65, label=f'N2 vs N3 (median={np.median(n2vsn3):.1e})',
        color='steelblue', edgecolor='white')
ax.hist(np.log10(n2vsn4), bins=50, alpha=0.65, label=f'N2 vs N4 (median={np.median(n2vsn4):.1e})',
        color='coral', edgecolor='white')
ax.set_xlabel('log₁₀(Waveform Difference)')
ax.set_ylabel('Number of Simulations')
ax.set_title('Distribution of Extrapolation Differences')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig3_extrapolation_comparison.png'), bbox_inches='tight')
plt.close()
print("  Figure 3 saved.")

# ============================================================
# 6. Figure 4: Error Scaling with ℓ
# ============================================================
print("Generating Figure 4: Error scaling with ℓ...")
fig, ax = plt.subplots(figsize=(8, 5.5))

medians = [stats_fig7[ell]['log10_median'] for ell in ell_values]
means = [stats_fig7[ell]['log10_mean'] for ell in ell_values]
stds = [stats_fig7[ell]['log10_std'] for ell in ell_values]

ax.errorbar(ell_values, means, yerr=stds, fmt='o-', color='steelblue', capsize=5,
            label='Mean ± 1σ', markersize=8, linewidth=2)
ax.plot(ell_values, medians, 's--', color='red', markersize=8, linewidth=1.5,
        label='Median')

# Fit a linear trend to the median
slope, intercept, r_value, p_value, std_err = stats.linregress(ell_values, medians)
x_fit = np.linspace(1.5, 8.5, 100)
ax.plot(x_fit, intercept + slope*x_fit, 'k:', alpha=0.5,
        label=f'Linear fit: slope={slope:.3f}/ℓ\n(R²={r_value**2:.3f})')

ax.set_xlabel('ℓ (Spherical Harmonic Mode Number)')
ax.set_ylabel('log₁₀(Waveform Difference)')
ax.set_title('Waveform Error Scaling with Multipole Order ℓ')
ax.legend()
ax.set_xticks(ell_values)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig4_error_scaling_with_ell.png'), bbox_inches='tight')
plt.close()
print("  Figure 4 saved.")

# ============================================================
# 7. Figure 5: Resolution vs Extrapolation Error Comparison
# ============================================================
print("Generating Figure 5: Resolution vs extrapolation error scale comparison...")
fig, ax = plt.subplots(figsize=(8, 5.5))

# Compare the scale of resolution error (fig6) with extrapolation error (fig8)
# Use CDFs on the same plot
sorted_fig6 = np.sort(fig6_vals)
sorted_n2vsn3 = np.sort(n2vsn3)
sorted_n2vsn4 = np.sort(n2vsn4)

cdf_fig6 = np.arange(1, len(sorted_fig6)+1) / len(sorted_fig6)
cdf_n2vsn3 = np.arange(1, len(sorted_n2vsn3)+1) / len(sorted_n2vsn3)
cdf_n2vsn4 = np.arange(1, len(sorted_n2vsn4)+1) / len(sorted_n2vsn4)

ax.semilogx(sorted_fig6, cdf_fig6, linewidth=2, color='steelblue',
             label=f'Resolution error (median={np.median(fig6_vals):.1e})')
ax.semilogx(sorted_n2vsn3, cdf_n2vsn3, linewidth=2, color='forestgreen',
             label=f'Extrap N2 vs N3 (median={np.median(n2vsn3):.1e})')
ax.semilogx(sorted_n2vsn4, cdf_n2vsn4, linewidth=2, color='coral',
             label=f'Extrap N2 vs N4 (median={np.median(n2vsn4):.1e})')

ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('Comparison of Resolution and Extrapolation Error Scales')
ax.legend(fontsize=9)
ax.set_xlim(1e-8, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig5_resolution_vs_extrapolation.png'), bbox_inches='tight')
plt.close()
print("  Figure 5 saved.")

# ============================================================
# 8. Figure 6: Log-normal fits
# ============================================================
print("Generating Figure 6: Log-normal distribution fits...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Fit log-normal to fig6
ax = axes[0]
shape6, loc6, scale6 = stats.lognorm.fit(fig6_vals, floc=0)
x_range = np.logspace(np.log10(fig6_vals.min())-0.5, np.log10(fig6_vals.max())+0.5, 500)
pdf_fitted = stats.lognorm.pdf(x_range, shape6, loc6, scale6)
ax.hist(fig6_vals, bins=60, density=True, color='steelblue', edgecolor='white', alpha=0.7, log=True)
ax.plot(x_range, pdf_fitted, 'r-', linewidth=2, label=f'Log-normal fit\nσ={shape6:.2f}, μ={np.log(scale6):.2f}')
ax.set_xscale('log')
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Density (log scale)')
ax.set_title('Resolution Error Distribution\nwith Log-normal Fit')
ax.legend(fontsize=9)

# Fit log-normal to fig7 ell=2
ax = axes[1]
ell2_vals = fig7['ell2'].values
shape_2, loc_2, scale_2 = stats.lognorm.fit(ell2_vals, floc=0)
x_range2 = np.logspace(np.log10(ell2_vals.min())-0.5, np.log10(ell2_vals.max())+0.5, 500)
pdf_fitted2 = stats.lognorm.pdf(x_range2, shape_2, loc_2, scale_2)
ax.hist(ell2_vals, bins=60, density=True, color='steelblue', edgecolor='white', alpha=0.7, log=True)
ax.plot(x_range2, pdf_fitted2, 'r-', linewidth=2, label=f'Log-normal fit\nσ={shape_2:.2f}, μ={np.log(scale_2):.2f}')
ax.set_xscale('log')
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Density (log scale)')
ax.set_title('ℓ=2 Modal Error Distribution\nwith Log-normal Fit')
ax.legend(fontsize=9)

# Fit log-normal to fig8 N2vsN3
ax = axes[2]
shape_n3, loc_n3, scale_n3 = stats.lognorm.fit(n2vsn3, floc=0)
x_range3 = np.logspace(np.log10(n2vsn3.min())-0.5, np.log10(n2vsn3.max())+0.5, 500)
pdf_fitted3 = stats.lognorm.pdf(x_range3, shape_n3, loc_n3, scale_n3)
ax.hist(n2vsn3, bins=60, density=True, color='steelblue', edgecolor='white', alpha=0.7, log=True)
ax.plot(x_range3, pdf_fitted3, 'r-', linewidth=2, label=f'Log-normal fit\nσ={shape_n3:.2f}, μ={np.log(scale_n3):.2f}')
ax.set_xscale('log')
ax.set_xlabel('Waveform Difference')
ax.set_ylabel('Density (log scale)')
ax.set_title('Extrapolation N2 vs N3 Error\nwith Log-normal Fit')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig6_lognormal_fits.png'), bbox_inches='tight')
plt.close()
print("  Figure 6 saved.")

# ============================================================
# 9. Figure 7: Heatmap of modal error percentiles
# ============================================================
print("Generating Figure 7: Modal error percentile heatmap...")
fig, ax = plt.subplots(figsize=(10, 5))

percentiles = [5, 25, 50, 75, 95]
heatmap_data = np.zeros((len(percentiles), len(ell_values)))
for i, p in enumerate(percentiles):
    for j, col in enumerate(ell_labels):
        heatmap_data[i, j] = np.log10(np.percentile(fig7[col].values, p))

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(ell_values)))
ax.set_xticklabels([f'ℓ={ell}' for ell in ell_values])
ax.set_yticks(range(len(percentiles)))
ax.set_yticklabels([f'P{p}' for p in percentiles])
ax.set_title('log₁₀(Waveform Difference) by Percentile and ℓ Mode')

# Add text annotations
for i in range(len(percentiles)):
    for j in range(len(ell_values)):
        val = heatmap_data[i, j]
        text_color = 'white' if val > np.median(heatmap_data) else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=9)

cbar = plt.colorbar(im, ax=ax, label='log₁₀(Waveform Difference)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig7_modal_error_heatmap.png'), bbox_inches='tight')
plt.close()
print("  Figure 7 saved.")

# ============================================================
# 10. Save detailed analysis results
# ============================================================
print("\nSaving detailed analysis results...")

# Correlation analysis between modes
corr_matrix = log_fig7.corr()
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'modal_correlation_matrix.csv'))

# Percentile table for fig7
percentile_table = pd.DataFrame(index=ell_labels)
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    percentile_table[f'P{p}'] = [np.percentile(fig7[col].values, p) for col in ell_labels]
percentile_table.to_csv(os.path.join(OUTPUT_DIR, 'fig7_percentile_table.csv'))

# Ratio analysis for fig8
ratio_n4_n3 = n2vsn4 / n2vsn3
ratio_stats = {
    'median_ratio': float(np.median(ratio_n4_n3)),
    'mean_ratio': float(np.mean(ratio_n4_n3)),
    'std_ratio': float(np.std(ratio_n4_n3)),
    'geometric_mean_ratio': float(np.exp(np.mean(np.log(ratio_n4_n3)))),
}
with open(os.path.join(OUTPUT_DIR, 'extrapolation_ratio_stats.json'), 'w') as f:
    json.dump(ratio_stats, f, indent=2)

# Log-normal fit parameters
lognorm_params = {
    'fig6': {'shape': float(shape6), 'scale': float(scale6), 'log_mean': float(np.log(scale6))},
    'fig7_ell2': {'shape': float(shape_2), 'scale': float(scale_2), 'log_mean': float(np.log(scale_2))},
    'fig8_N2vsN3': {'shape': float(shape_n3), 'scale': float(scale_n3), 'log_mean': float(np.log(scale_n3))},
}
with open(os.path.join(OUTPUT_DIR, 'lognormal_fit_params.json'), 'w') as f:
    json.dump(lognorm_params, f, indent=2)

# Accuracy classification
accuracy_classes = {
    'high_accuracy_lt_1e4': int(np.sum(fig6_vals < 1e-4)),
    'good_accuracy_lt_1e3': int(np.sum(fig6_vals < 1e-3)),
    'moderate_accuracy_lt_1e2': int(np.sum((fig6_vals >= 1e-3) & (fig6_vals < 1e-2))),
    'low_accuracy_gt_1e2': int(np.sum(fig6_vals >= 1e-2)),
    'fractions': {
        'high_accuracy': float(np.mean(fig6_vals < 1e-4)),
        'good_accuracy': float(np.mean(fig6_vals < 1e-3)),
        'moderate_accuracy': float(np.mean((fig6_vals >= 1e-3) & (fig6_vals < 1e-2))),
        'low_accuracy': float(np.mean(fig6_vals >= 1e-2)),
    }
}
with open(os.path.join(OUTPUT_DIR, 'accuracy_classification.json'), 'w') as f:
    json.dump(accuracy_classes, f, indent=2)

print("\nAll analysis complete!")
print(f"Figures saved in: {IMAGE_DIR}")
print(f"Results saved in: {OUTPUT_DIR}")
