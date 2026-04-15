#!/usr/bin/env python3
"""
Analysis of SXS Binary Black Hole Waveform Catalog Numerical Uncertainties
Reproducing Figures 6, 7, 8 from the SXS third catalog paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import json
import os

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# Load data
# ============================================================
fig6 = pd.read_csv('data/fig6_data.csv')
fig7 = pd.read_csv('data/fig7_data.csv')
fig8 = pd.read_csv('data/fig8_data.csv')

print("fig6 shape:", fig6.shape)
print("fig7 shape:", fig7.shape)
print("fig8 shape:", fig8.shape)

# ============================================================
# Figure 6: Overall waveform difference distribution
# ============================================================
fig6_vals = fig6['waveform_difference'].values

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.logspace(np.log10(fig6_vals.min()), np.log10(fig6_vals.max()), 50)
ax.hist(fig6_vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('Waveform difference (minimal alignment)', fontsize=13)
ax.set_ylabel('Number of simulations', fontsize=13)
ax.set_title('Figure 6: Distribution of Waveform Differences\nBetween Two Highest Resolutions', fontsize=14)
median_val = np.median(fig6_vals)
ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, label=f'Median = {median_val:.2e}')
ax.legend(fontsize=12)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('report/images/fig6_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 7: Modal decomposition of waveform differences
# ============================================================
ell_cols = ['ell2', 'ell3', 'ell4', 'ell5', 'ell6', 'ell7', 'ell8']
ell_labels = [r'$\ell=2$', r'$\ell=3$', r'$\ell=4$', r'$\ell=5$', r'$\ell=6$', r'$\ell=7$', r'$\ell=8$']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (col, label) in enumerate(zip(ell_cols, ell_labels)):
    vals = fig7[col].values
    bins = np.logspace(np.log10(vals[vals > 0].min()), np.log10(vals.max()), 40)
    axes[i].hist(vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_xscale('log')
    axes[i].set_title(label, fontsize=13)
    axes[i].set_xlabel('Difference', fontsize=10)
    axes[i].set_ylabel('Count', fontsize=10)
    med = np.median(vals)
    axes[i].axvline(med, color='red', linestyle='--', linewidth=1.2, label=f'Med={med:.1e}')
    axes[i].legend(fontsize=8)

# Remove the unused 8th subplot
axes[7].axis('off')
fig.suptitle('Figure 7: Waveform Differences by Spherical Harmonic Mode', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig7_modal_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary stats for fig7
fig7_medians = {col: np.median(fig7[col].values) for col in ell_cols}
print("\nFigure 7 medians by mode:")
for k, v in fig7_medians.items():
    print(f"  {k}: {v:.4e}")

# ============================================================
# Figure 8: Extrapolation order comparison
# ============================================================
n2n3 = fig8['N2vsN3'].values
n2n4 = fig8['N2vsN4'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: side-by-side histograms
bins3 = np.logspace(np.log10(n2n3[n2n3 > 0].min()), np.log10(n2n3.max()), 40)
bins4 = np.logspace(np.log10(n2n4[n2n4 > 0].min()), np.log10(n2n4.max()), 40)
axes[0].hist(n2n3, bins=bins3, alpha=0.7, color='steelblue', edgecolor='white', label='N=2 vs N=3')
axes[0].hist(n2n4, bins=bins4, alpha=0.7, color='coral', edgecolor='white', label='N=2 vs N=4')
axes[0].set_xscale('log')
axes[0].set_xlabel('Waveform difference', fontsize=12)
axes[0].set_ylabel('Number of simulations', fontsize=12)
axes[0].set_title('Extrapolation Order Comparison', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].axvline(np.median(n2n3), color='navy', linestyle='--', linewidth=1.2)
axes[0].axvline(np.median(n2n4), color='darkred', linestyle='--', linewidth=1.2)

# Right: ratio plot
ratio = n2n4 / n2n3
ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
bins_r = np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), 40)
axes[1].hist(ratio, bins=bins_r, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[1].set_xscale('log')
axes[1].set_xlabel('Ratio (N2vsN4 / N2vsN3)', fontsize=12)
axes[1].set_ylabel('Number of simulations', fontsize=12)
axes[1].set_title('Ratio of Extrapolation Differences', fontsize=13)
axes[1].axvline(np.median(ratio), color='red', linestyle='--', linewidth=1.2,
                label=f'Median ratio = {np.median(ratio):.2f}')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('report/images/fig8_extrapolation.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Additional analysis: Combined summary figure
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Overall catalog accuracy (fig6)
bins6 = np.logspace(np.log10(fig6_vals.min()), np.log10(fig6_vals.max()), 50)
axes[0].hist(fig6_vals, bins=bins6, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_xscale('log')
axes[0].set_xlabel('Waveform difference', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('(a) Overall Resolution Error', fontsize=13)
axes[0].axvline(median_val, color='red', linestyle='--', linewidth=1.5, label=f'Median = {median_val:.2e}')
axes[0].legend(fontsize=10)

# Panel B: Median difference vs ell
medians = [fig7_medians[c] for c in ell_cols]
ell_numbers = [2, 3, 4, 5, 6, 7, 8]
axes[1].bar(ell_numbers, medians, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].set_yscale('log')
axes[1].set_xlabel(r'Spherical harmonic mode $\ell$', fontsize=12)
axes[1].set_ylabel('Median waveform difference', fontsize=12)
axes[1].set_title('(b) Error Growth with Mode Order', fontsize=13)
axes[1].set_xticks(ell_numbers)

# Panel C: Extrapolation convergence
med_n2n3 = np.median(n2n3)
med_n2n4 = np.median(n2n4)
axes[2].bar(['N=2 vs N=3', 'N=2 vs N=4'], [med_n2n3, med_n2n4],
            color=['steelblue', 'coral'], edgecolor='white', alpha=0.85)
axes[2].set_yscale('log')
axes[2].set_ylabel('Median waveform difference', fontsize=12)
axes[2].set_title('(c) Extrapolation Convergence', fontsize=13)

plt.suptitle('SXS Binary Black Hole Waveform Catalog: Numerical Uncertainty Summary', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/summary_panels.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Save quantitative results
# ============================================================
results = {
    "fig6_overall": {
        "n_simulations": int(len(fig6_vals)),
        "median": float(np.median(fig6_vals)),
        "mean": float(np.mean(fig6_vals)),
        "std": float(np.std(fig6_vals)),
        "min": float(np.min(fig6_vals)),
        "max": float(np.max(fig6_vals)),
        "percentile_5": float(np.percentile(fig6_vals, 5)),
        "percentile_95": float(np.percentile(fig6_vals, 95)),
        "fraction_below_1e3": float(np.mean(fig6_vals < 1e-3)),
        "fraction_below_1e4": float(np.mean(fig6_vals < 1e-4)),
    },
    "fig7_by_mode": {},
    "fig8_extrapolation": {
        "n_simulations": int(len(n2n3)),
        "N2vsN3_median": float(med_n2n3),
        "N2vsN3_mean": float(np.mean(n2n3)),
        "N2vsN4_median": float(med_n2n4),
        "N2vsN4_mean": float(np.mean(n2n4)),
        "ratio_median": float(np.median(ratio)),
        "ratio_mean": float(np.mean(ratio)),
    }
}

for col in ell_cols:
    vals = fig7[col].values
    results["fig7_by_mode"][col] = {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

with open('outputs/quantitative_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Log-normal fit for fig6
log_fig6 = np.log10(fig6_vals)
mu, sigma = np.mean(log_fig6), np.std(log_fig6)
fit_results = {
    "fig6_lognormal_fit": {
        "log10_mu": float(mu),
        "log10_sigma": float(sigma),
        "median_from_fit": float(10**mu),
    }
}
with open('outputs/statistical_fits.json', 'w') as f:
    json.dump(fit_results, f, indent=2)

print("\nAll figures saved to report/images/")
print("Quantitative results saved to outputs/quantitative_results.json")
print("Statistical fits saved to outputs/statistical_fits.json")

# Print summary
print("\n=== SUMMARY ===")
print(f"Fig 6: {len(fig6_vals)} simulations, median difference = {median_val:.4e}")
print(f"  95% of simulations have difference < {np.percentile(fig6_vals, 95):.4e}")
print(f"  {100*np.mean(fig6_vals < 1e-3):.1f}% have difference < 1e-3")
print(f"\nFig 7: Modal decomposition")
for col, med in fig7_medians.items():
    print(f"  {col}: median = {med:.4e}")
print(f"\nFig 8: Extrapolation")
print(f"  N2vsN3 median = {med_n2n3:.4e}")
print(f"  N2vsN4 median = {med_n2n4:.4e}")
print(f"  Ratio median = {np.median(ratio):.2f}")
