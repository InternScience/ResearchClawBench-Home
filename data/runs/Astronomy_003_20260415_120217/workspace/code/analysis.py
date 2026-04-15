#!/usr/bin/env python3
"""
Analysis code for Binary Black Hole Gravitational Waveform Catalog

This script analyzes waveform error data from numerical relativity simulations
to assess numerical uncertainty, modal decomposition, and extrapolation convergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directories
os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# Load data
print("Loading data...")
fig6_data = pd.read_csv('../data/fig6_data.csv')
fig7_data = pd.read_csv('../data/fig7_data.csv')
fig8_data = pd.read_csv('../data/fig8_data.csv')

print(f"Figure 6 data shape: {fig6_data.shape}")
print(f"Figure 7 data shape: {fig7_data.shape}")
print(f"Figure 8 data shape: {fig8_data.shape}")

# ============================================================================
# FIGURE 6: Overall Numerical Resolution Error Distribution
# ============================================================================
print("\n=== Analyzing Figure 6 Data: Resolution Error Distribution ===")

waveform_diff = fig6_data['waveform_difference'].values

# Basic statistics
median_val = np.median(waveform_diff)
mean_val = np.mean(waveform_diff)
std_val = np.std(waveform_diff)
min_val = np.min(waveform_diff)
max_val = np.max(waveform_diff)

print(f"Median: {median_val:.4e}")
print(f"Mean: {mean_val:.4e}")
print(f"Standard Deviation: {std_val:.4e}")
print(f"Min: {min_val:.4e}")
print(f"Max: {max_val:.4e}")

# Percentiles
p16 = np.percentile(waveform_diff, 16)
p84 = np.percentile(waveform_diff, 84)
p90 = np.percentile(waveform_diff, 90)
p95 = np.percentile(waveform_diff, 95)
p99 = np.percentile(waveform_diff, 99)

print(f"16th percentile: {p16:.4e}")
print(f"84th percentile: {p84:.4e}")
print(f"90th percentile: {p90:.4e}")
print(f"95th percentile: {p95:.4e}")
print(f"99th percentile: {p99:.4e}")

# Fit log-normal distribution
log_data = np.log(waveform_diff)
log_mean = np.mean(log_data)
log_std = np.std(log_data)
print(f"\nLog-normal fit parameters:")
print(f"  mu (log mean): {log_mean:.4f}")
print(f"  sigma (log std): {log_std:.4f}")

# Create Figure 6 visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): Histogram with log-normal fit
ax1 = axes[0, 0]
counts, bins, patches = ax1.hist(waveform_diff, bins=50, density=True, 
                                   alpha=0.7, color='steelblue', edgecolor='black')
# Overlay log-normal fit
x_fit = np.logspace(np.log10(min_val*0.5), np.log10(max_val*2), 500)
lognormal_pdf = stats.lognorm.pdf(x_fit, s=log_std, scale=np.exp(log_mean))
ax1.plot(x_fit, lognormal_pdf, 'r-', lw=2, label=f'Log-normal fit\n$\mu$={log_mean:.2f}, $\sigma$={log_std:.2f}')
ax1.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Median = {median_val:.2e}')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Waveform Difference', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('(a) Distribution of Resolution Errors', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel (b): Cumulative distribution
ax2 = axes[0, 1]
sorted_data = np.sort(waveform_diff)
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.semilogx(sorted_data, cumulative, 'b-', lw=2)
ax2.axvline(median_val, color='green', linestyle='--', lw=2, label=f'Median = {median_val:.2e}')
ax2.axhline(0.5, color='green', linestyle='--', lw=1, alpha=0.5)
ax2.axvline(p90, color='orange', linestyle='--', lw=2, label=f'90th %ile = {p90:.2e}')
ax2.axhline(0.9, color='orange', linestyle='--', lw=1, alpha=0.5)
ax2.set_xlabel('Waveform Difference', fontsize=11)
ax2.set_ylabel('Cumulative Fraction', fontsize=11)
ax2.set_title('(b) Cumulative Distribution', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min_val*0.5, max_val*2)
ax2.set_ylim(0, 1)

# Panel (c): Q-Q plot against log-normal
ax3 = axes[1, 0]
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(log_data)))
ax3.scatter(theoretical_quantiles, np.sort(log_data), alpha=0.5, s=20, c='steelblue')
ax3.plot([-4, 4], [-4, 4], 'r--', lw=2, label='Perfect fit')
ax3.set_xlabel('Theoretical Quantiles (Normal)', fontsize=11)
ax3.set_ylabel('Sample Quantiles (Log-transformed)', fontsize=11)
ax3.set_title('(c) Q-Q Plot: Log-normal Assessment', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel (d): Error categories
ax4 = axes[1, 1]
categories = ['High accuracy\n(< 10⁻³)', 'Good\n(10⁻³ - 10⁻²)', 'Moderate\n(10⁻² - 10⁻¹)', 'Poor\n(> 10⁻¹)']
counts_cat = [
    np.sum(waveform_diff < 1e-3),
    np.sum((waveform_diff >= 1e-3) & (waveform_diff < 1e-2)),
    np.sum((waveform_diff >= 1e-2) & (waveform_diff < 1e-1)),
    np.sum(waveform_diff >= 1e-1)
]
colors = ['darkgreen', 'green', 'orange', 'red']
bars = ax4.bar(categories, counts_cat, color=colors, edgecolor='black', alpha=0.8)
ax4.set_ylabel('Number of Simulations', fontsize=11)
ax4.set_title('(d) Error Categories', fontsize=12, fontweight='bold')
for bar, count in zip(bars, counts_cat):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}\n({100*count/len(waveform_diff):.1f}%)',
             ha='center', va='bottom', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../report/images/fig6_resolution_errors.png', dpi=300, bbox_inches='tight')
plt.savefig('../outputs/fig6_resolution_errors.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================================
# FIGURE 7: Modal Decomposition Analysis
# ============================================================================
print("\n=== Analyzing Figure 7 Data: Spherical Harmonic Mode Errors ===")

# Compute statistics for each mode
mode_stats = {}
modes = ['ell2', 'ell3', 'ell4', 'ell5', 'ell6', 'ell7', 'ell8']
ell_values = [2, 3, 4, 5, 6, 7, 8]

for mode in modes:
    data = fig7_data[mode].values
    mode_stats[mode] = {
        'median': np.median(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'p16': np.percentile(data, 16),
        'p84': np.percentile(data, 84)
    }
    print(f"{mode}: median={mode_stats[mode]['median']:.4e}, "
          f"mean={mode_stats[mode]['mean']:.4e}")

# Create Figure 7 visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): Box plot by mode
ax1 = axes[0, 0]
data_for_box = [fig7_data[m].values for m in modes]
bp = ax1.boxplot(data_for_box, labels=[f'$\\ell={i}$' for i in ell_values],
                  patch_artist=True, showfliers=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_yscale('log')
ax1.set_xlabel('Spherical Harmonic Mode', fontsize=11)
ax1.set_ylabel('Waveform Difference', fontsize=11)
ax1.set_title('(a) Error Distribution by Mode', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel (b): Median error vs ell
ax2 = axes[0, 1]
medians = [mode_stats[m]['median'] for m in modes]
means = [mode_stats[m]['mean'] for m in modes]
errors_low = [medians[i] - mode_stats[m]['p16'] for i, m in enumerate(modes)]
errors_high = [mode_stats[m]['p84'] - medians[i] for i, m in enumerate(modes)]

ax2.errorbar(ell_values, medians, yerr=[errors_low, errors_high], 
             fmt='o-', capsize=5, capthick=2, lw=2, markersize=8,
             color='steelblue', label='Median ± 1σ')
ax2.plot(ell_values, means, 's--', color='coral', lw=2, markersize=8, label='Mean')
ax2.set_xlabel('Spherical Harmonic Index ℓ', fontsize=11)
ax2.set_ylabel('Waveform Difference', fontsize=11)
ax2.set_yscale('log')
ax2.set_title('(b) Error vs Spherical Harmonic Mode', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel (c): Violin plot
ax3 = axes[1, 0]
parts = ax3.violinplot(data_for_box, positions=range(1, len(modes)+1), 
                        showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
ax3.set_xticks(range(1, len(modes)+1))
ax3.set_xticklabels([f'$\\ell={i}$' for i in ell_values])
ax3.set_yscale('log')
ax3.set_xlabel('Spherical Harmonic Mode', fontsize=11)
ax3.set_ylabel('Waveform Difference', fontsize=11)
ax3.set_title('(c) Violin Plot of Mode Errors', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Panel (d): Cumulative contribution
ax4 = axes[1, 1]
cumulative_median = np.cumsum(medians)
total = cumulative_median[-1]
colors_list = [plt.cm.viridis(i/len(modes)) for i in range(len(modes))]
ax4.bar(ell_values, 100*np.array(medians)/total, color=colors_list, edgecolor='black', alpha=0.8)
ax4.plot(ell_values, 100*cumulative_median/total, 'ro-', lw=2, markersize=8, 
         label='Cumulative contribution')
ax4.set_xlabel('Spherical Harmonic Index ℓ', fontsize=11)
ax4.set_ylabel('Contribution to Total Error (%)', fontsize=11)
ax4.set_title('(d) Relative Error Contribution by Mode', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../report/images/fig7_modal_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig('../outputs/fig7_modal_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

# ============================================================================
# FIGURE 8: Extrapolation Order Convergence
# ============================================================================
print("\n=== Analyzing Figure 8 Data: Extrapolation Convergence ===")

n2vsn3 = fig8_data['N2vsN3'].values
n2vsn4 = fig8_data['N2vsN4'].values

# Statistics
print(f"N=2 vs N=3: median={np.median(n2vsn3):.4e}, mean={np.mean(n2vsn3):.4e}")
print(f"N=2 vs N=4: median={np.median(n2vsn4):.4e}, mean={np.mean(n2vsn4):.4e}")

# Create Figure 8 visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): Scatter comparison
ax1 = axes[0, 0]
ax1.scatter(n2vsn3, n2vsn4, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot([1e-7, 1e-3], [1e-7, 1e-3], 'r--', lw=2, label='y=x (equal difference)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('N=2 vs N=3 Difference', fontsize=11)
ax1.set_ylabel('N=2 vs N=4 Difference', fontsize=11)
ax1.set_title('(a) Extrapolation Order Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel (b): Histograms comparison
ax2 = axes[0, 1]
bins = np.logspace(np.log10(min(np.min(n2vsn3), np.min(n2vsn4))*0.5), 
                   np.log10(max(np.max(n2vsn3), np.max(n2vsn4))*2), 40)
ax2.hist(n2vsn3, bins=bins, alpha=0.6, label='N=2 vs N=3', color='steelblue', edgecolor='black')
ax2.hist(n2vsn4, bins=bins, alpha=0.6, label='N=2 vs N=4', color='coral', edgecolor='black')
ax2.axvline(np.median(n2vsn3), color='steelblue', linestyle='--', lw=2)
ax2.axvline(np.median(n2vsn4), color='coral', linestyle='--', lw=2)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Waveform Difference', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('(b) Distribution Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel (c): Ratio analysis
ax3 = axes[1, 0]
ratio = n2vsn4 / n2vsn3
# Filter out extreme outliers for visualization
ratio_filtered = ratio[(ratio > 0.01) & (ratio < 100)]
ax3.hist(ratio_filtered, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(np.median(ratio_filtered), color='red', linestyle='--', lw=2, 
            label=f'Median ratio = {np.median(ratio_filtered):.2f}')
ax3.axvline(1, color='green', linestyle='-', lw=1, alpha=0.5, label='Ratio = 1')
ax3.set_xlabel('Ratio (N2vsN4 / N2vsN3)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('(c) Convergence Ratio Distribution', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel (d): Box plot comparison
ax4 = axes[1, 1]
bp = ax4.boxplot([n2vsn3, n2vsn4], labels=['N=2 vs N=3', 'N=2 vs N=4'],
                  patch_artist=True, showfliers=False)
colors_box = ['steelblue', 'coral']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Waveform Difference', fontsize=11)
ax4.set_yscale('log')
ax4.set_title('(d) Extrapolation Order Statistics', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add text with statistics
stats_text = f"N=2 vs N=3:\n  Median: {np.median(n2vsn3):.2e}\n  Mean: {np.mean(n2vsn3):.2e}\n\n"
stats_text += f"N=2 vs N=4:\n  Median: {np.median(n2vsn4):.2e}\n  Mean: {np.mean(n2vsn4):.2e}"
ax4.text(1.6, np.median(n2vsn3)*3, stats_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../report/images/fig8_extrapolation_convergence.png', dpi=300, bbox_inches='tight')
plt.savefig('../outputs/fig8_extrapolation_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================
print("\n=== Generating Summary Statistics ===")

summary_stats = {
    'Figure 6 (Resolution Errors)': {
        'Sample Size': len(waveform_diff),
        'Median': median_val,
        'Mean': mean_val,
        'Std Dev': std_val,
        'Min': min_val,
        'Max': max_val,
        '90th %ile': p90,
        '99th %ile': p99
    },
    'Figure 8 (N=2 vs N=3)': {
        'Sample Size': len(n2vsn3),
        'Median': np.median(n2vsn3),
        'Mean': np.mean(n2vsn3),
        'Std Dev': np.std(n2vsn3),
        'Min': np.min(n2vsn3),
        'Max': np.max(n2vsn3),
        '90th %ile': np.percentile(n2vsn3, 90),
        '99th %ile': np.percentile(n2vsn3, 99)
    },
    'Figure 8 (N=2 vs N=4)': {
        'Sample Size': len(n2vsn4),
        'Median': np.median(n2vsn4),
        'Mean': np.mean(n2vsn4),
        'Std Dev': np.std(n2vsn4),
        'Min': np.min(n2vsn4),
        'Max': np.max(n2vsn4),
        '90th %ile': np.percentile(n2vsn4, 90),
        '99th %ile': np.percentile(n2vsn4, 99)
    }
}

# Save summary to CSV
summary_df = pd.DataFrame(summary_stats).T
summary_df.to_csv('../outputs/summary_statistics.csv')
print("Summary statistics saved to outputs/summary_statistics.csv")

# Mode summary
mode_summary = pd.DataFrame({
    'Mode': modes,
    'l_value': ell_values,
    'Median': [mode_stats[m]['median'] for m in modes],
    'Mean': [mode_stats[m]['mean'] for m in modes],
    'Std': [mode_stats[m]['std'] for m in modes],
    'Min': [mode_stats[m]['min'] for m in modes],
    'Max': [mode_stats[m]['max'] for m in modes]
})
mode_summary.to_csv('../outputs/mode_statistics.csv', index=False)
print("Mode statistics saved to outputs/mode_statistics.csv")

print("\n=== Analysis Complete ===")
