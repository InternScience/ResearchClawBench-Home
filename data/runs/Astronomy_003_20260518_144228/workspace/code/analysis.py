"""
Analysis script for SXS BBH waveform catalog accuracy assessment.
Analyzes resolution, modal, and extrapolation errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import json
import os

# Ensure directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150

# =============================================================================
# Load Data
# =============================================================================
fig6 = pd.read_csv('data/fig6_data.csv')
fig7 = pd.read_csv('data/fig7_data.csv')
fig8 = pd.read_csv('data/fig8_data.csv')

res_err = fig6['waveform_difference'].values
modal_err = fig7.values  # shape (1500, 7)
extrap_err_n23 = fig8['N2vsN3'].values
extrap_err_n24 = fig8['N2vsN4'].values

# =============================================================================
# Statistical Characterization
# =============================================================================

def lognormal_stats(data, name):
    """Fit log-normal and compute key statistics."""
    log_data = np.log(data)
    mu, sigma = log_data.mean(), log_data.std()
    # Median and mode of log-normal
    median = np.exp(mu)
    mode = np.exp(mu - sigma**2)
    mean = np.exp(mu + sigma**2 / 2)
    # Percentiles
    p5, p25, p75, p95 = np.percentile(data, [5, 25, 75, 95])
    # KS test against fitted log-normal
    fitted_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    ks_stat, ks_p = stats.kstest(data, fitted_dist.cdf)
    return {
        'name': name,
        'count': int(len(data)),
        'mean': float(mean),
        'median': float(median),
        'mode': float(mode),
        'std': float(sigma),
        'mu_log': float(mu),
        'sigma_log': float(sigma),
        'min': float(data.min()),
        'max': float(data.max()),
        'p5': float(p5),
        'p25': float(p25),
        'p75': float(p75),
        'p95': float(p95),
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_p),
    }

stats_res = lognormal_stats(res_err, 'resolution')
stats_n23 = lognormal_stats(extrap_err_n23, 'extrap_N2vsN3')
stats_n24 = lognormal_stats(extrap_err_n24, 'extrap_N2vsN4')

# Modal stats
modal_stats = {}
for i, col in enumerate(fig7.columns):
    data = fig7[col].values
    modal_stats[col] = lognormal_stats(data, col)

all_stats = {
    'resolution': stats_res,
    'extrap_N2vsN3': stats_n23,
    'extrap_N2vsN4': stats_n24,
    'modal': modal_stats,
}

with open('outputs/statistics.json', 'w') as f:
    json.dump(all_stats, f, indent=2)

# =============================================================================
# Figure 1: Resolution Error Distribution (Histogram + Log-Normal Fit)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: histogram on log scale
ax = axes[0]
bins = np.logspace(np.log10(res_err.min()*0.8), np.log10(res_err.max()*1.2), 60)
ax.hist(res_err, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
# Overlay log-normal fit
log_data = np.log(res_err)
mu, sigma = log_data.mean(), log_data.std()
x_fit = np.logspace(np.log10(res_err.min()*0.5), np.log10(res_err.max()*2), 500)
pdf_fit = stats.lognorm.pdf(x_fit, s=sigma, scale=np.exp(mu))
ax.plot(x_fit, pdf_fit, 'r-', lw=2, label=f'Log-normal fit\n$\\mu_{{\\ln}}={mu:.2f}$, $\\sigma_{{\\ln}}={sigma:.2f}$')
ax.axvline(np.median(res_err), color='darkgreen', ls='--', lw=1.5, label=f'Median = {np.median(res_err):.2e}')
ax.axvline(np.percentile(res_err, 95), color='orange', ls='-.', lw=1.5, label=f'95th %ile = {np.percentile(res_err, 95):.2e}')
ax.set_xscale('log')
ax.set_xlabel('Waveform difference (resolution)')
ax.set_ylabel('Probability density')
ax.set_title('Resolution Error Distribution')
ax.legend(loc='upper left')

# Right: CDF
ax = axes[1]
sorted_err = np.sort(res_err)
cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
ax.semilogx(sorted_err, cdf, color='steelblue', lw=2)
ax.axvline(np.median(res_err), color='darkgreen', ls='--', lw=1.5)
ax.axhline(0.5, color='darkgreen', ls='--', lw=1.5, alpha=0.5)
ax.axvline(np.percentile(res_err, 95), color='orange', ls='-.', lw=1.5)
ax.axhline(0.95, color='orange', ls='-.', lw=1.5, alpha=0.5)
ax.set_xlabel('Waveform difference (resolution)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Resolution Error Cumulative Distribution')
ax.set_xlim(1e-6, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/fig1_resolution_error.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Modal Error Distributions by Spherical Harmonic ℓ
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: boxplot
ax = axes[0]
bp = ax.boxplot([fig7[col].values for col in fig7.columns],
                labels=[f'$\\ell={i+2}$' for i in range(7)],
                patch_artist=True, showfliers=False,
                medianprops=dict(color='darkred', lw=2))
for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, 7))):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_yscale('log')
ax.set_ylabel('Waveform difference (modal)')
ax.set_xlabel('Spherical harmonic mode $\\ell$')
ax.set_title('Modal Error Distributions by $\\ell$')

# Right: median + IQR as function of ell
ax = axes[1]
ell_vals = np.arange(2, 9)
medians = [np.median(fig7[f'ell{i}'].values) for i in ell_vals]
p25s = [np.percentile(fig7[f'ell{i}'].values, 25) for i in ell_vals]
p75s = [np.percentile(fig7[f'ell{i}'].values, 75) for i in ell_vals]
ax.errorbar(ell_vals, medians, yerr=[np.array(medians)-np.array(p25s), np.array(p75s)-np.array(medians)],
            fmt='o-', capsize=4, capthick=1.5, color='steelblue', ecolor='gray', lw=2, markersize=8)
ax.set_yscale('log')
ax.set_xlabel('Spherical harmonic mode $\\ell$')
ax.set_ylabel('Median waveform difference')
ax.set_title('Median Modal Error vs. $\\ell$')
ax.set_xticks(ell_vals)

plt.tight_layout()
plt.savefig('report/images/fig2_modal_errors.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: Extrapolation Error Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: histograms
ax = axes[0]
bins = np.logspace(np.log10(min(extrap_err_n23.min(), extrap_err_n24.min())*0.5),
                   np.log10(max(extrap_err_n23.max(), extrap_err_n24.max())*2), 60)
ax.hist(extrap_err_n23, bins=bins, density=True, alpha=0.6, color='steelblue', label='N=2 vs N=3')
ax.hist(extrap_err_n24, bins=bins, density=True, alpha=0.6, color='coral', label='N=2 vs N=4')
ax.set_xscale('log')
ax.set_xlabel('Waveform difference (extrapolation)')
ax.set_ylabel('Probability density')
ax.set_title('Extrapolation Error Distributions')
ax.legend()

# Right: scatter + CDF comparison
ax = axes[1]
# CDFs
for data, label, color in [(extrap_err_n23, 'N=2 vs N=3', 'steelblue'),
                           (extrap_err_n24, 'N=2 vs N=4', 'coral')]:
    sorted_d = np.sort(data)
    cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
    ax.semilogx(sorted_d, cdf, color=color, lw=2, label=label)
    ax.axvline(np.median(data), color=color, ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('Waveform difference (extrapolation)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Extrapolation Error CDFs')
ax.legend()
ax.set_xlim(1e-7, 1e-2)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/fig3_extrapolation_errors.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: Cumulative Comparison of All Error Sources
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for data, label, color in [(res_err, 'Resolution', 'steelblue'),
                           (extrap_err_n23, 'Extrapolation N2 vs N3', 'coral'),
                           (extrap_err_n24, 'Extrapolation N2 vs N4', 'seagreen')]:
    sorted_d = np.sort(data)
    cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
    ax.semilogx(sorted_d, cdf, color=color, lw=2.5, label=label)

ax.set_xlabel('Waveform difference')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Cumulative Distribution of All Error Sources')
ax.legend(loc='lower right')
ax.set_xlim(1e-7, 1)
ax.set_ylim(0, 1)
ax.grid(True, which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/fig4_all_errors_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: Modal Error Correlation Matrix
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
corr = fig7.corr()
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(7))
ax.set_yticks(range(7))
ax.set_xticklabels([f'$\\ell={i+2}$' for i in range(7)])
ax.set_yticklabels([f'$\\ell={i+2}$' for i in range(7)])
for i in range(7):
    for j in range(7):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)
plt.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Modal Error Correlation Matrix')
plt.tight_layout()
plt.savefig('report/images/fig5_modal_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 6: Resolution vs. Combined Modal Error (conceptual composite)
# =============================================================================
# Since we don't have 1:1 mapping between fig6 and fig7 indices, we compare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: violin plot comparing all error sources side-by-side
ax = axes[0]
all_data = [res_err, extrap_err_n23, extrap_err_n24]
labels = ['Resolution', 'Extrap. N2vsN3', 'Extrap. N2vsN4']
parts = ax.violinplot(all_data, positions=range(3), showmedians=True, widths=0.7)
for pc, color in zip(parts['bodies'], ['steelblue', 'coral', 'seagreen']):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.set_ylabel('Waveform difference')
ax.set_title('Error Source Comparison (Violin)')

# Right: fraction below thresholds
ax = axes[1]
thresholds = np.logspace(-6, -1, 50)
fracs_res = [np.mean(res_err < t) for t in thresholds]
fracs_n23 = [np.mean(extrap_err_n23 < t) for t in thresholds]
fracs_n24 = [np.mean(extrap_err_n24 < t) for t in thresholds]
ax.semilogx(thresholds, fracs_res, 'steelblue', lw=2, label='Resolution')
ax.semilogx(thresholds, fracs_n23, 'coral', lw=2, label='Extrap. N2vsN3')
ax.semilogx(thresholds, fracs_n24, 'seagreen', lw=2, label='Extrap. N2vsN4')
ax.axhline(0.9, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Waveform difference threshold')
ax.set_ylabel('Fraction below threshold')
ax.set_title('Error Fraction vs. Threshold')
ax.legend()
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig('report/images/fig6_error_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 7: Log-normal parameter fits summary
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: QQ plot for resolution errors
ax = axes[0]
log_res = np.log(res_err)
stats.probplot(log_res, dist=stats.norm, plot=ax)
ax.set_title('Q-Q Plot: Resolution Errors vs. Log-Normal')
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('none')
ax.get_lines()[0].set_alpha(0.5)
ax.get_lines()[1].set_color('darkred')

# Right: modal median trend with log-normal sigma
ax = axes[1]
ell_vals = np.arange(2, 9)
medians = [modal_stats[f'ell{i}']['median'] for i in ell_vals]
sigmas = [modal_stats[f'ell{i}']['sigma_log'] for i in ell_vals]
ax2 = ax.twinx()
ax.plot(ell_vals, medians, 'o-', color='steelblue', lw=2, markersize=8, label='Median')
ax2.plot(ell_vals, sigmas, 's--', color='coral', lw=2, markersize=8, label='$\\sigma_{\\ln}$')
ax.set_xlabel('Spherical harmonic mode $\\ell$')
ax.set_ylabel('Median error', color='steelblue')
ax2.set_ylabel('Log-normal $\\sigma_{\\ln}$', color='coral')
ax.set_yscale('log')
ax.set_title('Modal Error Median and Spread vs. $\\ell$')
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')
ax.set_xticks(ell_vals)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('report/images/fig7_qq_and_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Save summary table
# =============================================================================
summary_rows = []
for name, s in [('Resolution', stats_res), ('Extrap N2vsN3', stats_n23), ('Extrap N2vsN4', stats_n24)]:
    summary_rows.append({
        'Error source': name,
        'N': s['count'],
        'Median': s['median'],
        'Mean': s['mean'],
        '95th percentile': s['p95'],
        'Max': s['max'],
        'sigma_ln': s['sigma_log'],
        'KS p-value': s['ks_pvalue'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/error_summary.csv', index=False)

modal_summary = []
for i in range(2, 9):
    s = modal_stats[f'ell{i}']
    modal_summary.append({
        'ell': i,
        'Median': s['median'],
        'Mean': s['mean'],
        '95th percentile': s['p95'],
        'sigma_ln': s['sigma_log'],
    })
pd.DataFrame(modal_summary).to_csv('outputs/modal_summary.csv', index=False)

print("Analysis complete. Generated figures and statistics.")
print(f"Resolution median: {np.median(res_err):.2e}")
print(f"Extrap N2vsN3 median: {np.median(extrap_err_n23):.2e}")
print(f"Extrap N2vsN4 median: {np.median(extrap_err_n24):.2e}")
