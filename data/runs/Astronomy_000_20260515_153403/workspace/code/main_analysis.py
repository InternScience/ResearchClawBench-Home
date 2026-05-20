#!/usr/bin/env python3
"""
Main Analysis: Bayesian Constraints on Ultralight Bosons from Black Hole Superradiance.

Loads posterior samples, applies the Bayesian framework, and generates
publication-quality figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import json
import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superradiance_physics import (
    compute_alpha, horizon_angular_velocity, superradiance_timescale,
    superradiance_condition_satisfied, is_in_exclusion_region,
    bosenova_condition, compute_exclusion_fraction, compute_likelihood,
    compute_upper_limit, ALPHA_FACTOR
)

# Set up plotting style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 150
rcParams['savefig.bbox'] = 'tight'

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_samples(filename):
    """Load posterior samples from a data file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])
    data = np.array(data)
    return data[:, 0], data[:, 1]  # M, a_star


def main():
    print("=" * 70)
    print("Bayesian Constraints on Ultralight Bosons")
    print("=" * 70)
    
    # ============================================================
    # 1. Load data
    # ============================================================
    print("\n[1] Loading posterior samples...")
    
    M_m33, a_m33 = load_samples(os.path.join(DATA_DIR, 'M33_X-7_samples.dat'))
    M_iras, a_iras = load_samples(os.path.join(DATA_DIR, 'IRAS_09149-6206_samples.dat'))
    
    print(f"  M33 X-7: {len(M_m33)} samples")
    print(f"    M  = {np.mean(M_m33):.2f} ± {np.std(M_m33):.2f} Msun")
    print(f"    a* = {np.mean(a_m33):.4f} ± {np.std(a_m33):.4f}")
    print(f"  IRAS 09149-6206: {len(M_iras)} samples")
    print(f"    M  = {np.mean(M_iras):.2e} ± {np.std(M_iras):.2e} Msun")
    print(f"    a* = {np.mean(a_iras):.4f} ± {np.std(a_iras):.4f}")
    
    # Save summary statistics
    summary = {
        'M33_X-7': {
            'n_samples': int(len(M_m33)),
            'M_mean': float(np.mean(M_m33)),
            'M_std': float(np.std(M_m33)),
            'M_median': float(np.median(M_m33)),
            'M_16pc': float(np.percentile(M_m33, 16)),
            'M_84pc': float(np.percentile(M_m33, 84)),
            'a_mean': float(np.mean(a_m33)),
            'a_std': float(np.std(a_m33)),
            'a_median': float(np.median(a_m33)),
            'a_16pc': float(np.percentile(a_m33, 16)),
            'a_84pc': float(np.percentile(a_m33, 84)),
        },
        'IRAS_09149-6206': {
            'n_samples': int(len(M_iras)),
            'M_mean': float(np.mean(M_iras)),
            'M_std': float(np.std(M_iras)),
            'M_median': float(np.median(M_iras)),
            'M_16pc': float(np.percentile(M_iras, 16)),
            'M_84pc': float(np.percentile(M_iras, 84)),
            'a_mean': float(np.mean(a_iras)),
            'a_std': float(np.std(a_iras)),
            'a_median': float(np.median(a_iras)),
            'a_16pc': float(np.percentile(a_iras, 16)),
            'a_84pc': float(np.percentile(a_iras, 84)),
        }
    }
    with open(os.path.join(OUTPUTS_DIR, 'sample_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ============================================================
    # 2. Generate Data Overview Figure
    # ============================================================
    print("\n[2] Generating data overview figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # M33 X-7
    ax = axes[0]
    h = ax.hist2d(M_m33, a_m33, bins=40, cmap='Blues', density=True)
    ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]')
    ax.set_ylabel(r'Spin Parameter $a_*$')
    ax.set_title('M33 X-7 (Stellar-mass BH)')
    plt.colorbar(h[3], ax=ax, label='Density')
    
    # IRAS 09149-6206
    ax = axes[1]
    h = ax.hist2d(M_iras / 1e8, a_iras, bins=40, cmap='Reds', density=True)
    ax.set_xlabel(r'Black Hole Mass $M$ [$10^8 M_\odot$]')
    ax.set_ylabel(r'Spin Parameter $a_*$')
    ax.set_title('IRAS 09149-6206 (Supermassive BH)')
    plt.colorbar(h[3], ax=ax, label='Density')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'data_overview.png'))
    plt.close()
    print("  → Saved report/images/data_overview.png")
    
    # ============================================================
    # 3. Superradiance Physics Validation
    # ============================================================
    print("\n[3] Computing superradiance physics grid...")
    
    # Define ULB mass grids for each BH
    # M33 X-7 probes mu ~ 10^-13 to 10^-10 eV
    mu_grid_m33 = np.logspace(-13, -9.5, 200)  # eV
    
    # IRAS 09149 probes mu ~ 10^-19 to 10^-16 eV
    mu_grid_iras = np.logspace(-19, -15.5, 200)  # eV
    
    # ============================================================
    # 4. Bayesian Exclusion Analysis
    # ============================================================
    print("\n[4] Computing Bayesian exclusion fractions...")
    
    # M33 X-7: age ~ 2-3 Myr (young), but we'll use 10^7 yr
    tau_m33 = 1e7  # years
    
    # IRAS 09149-6206: age ~ few Gyr
    tau_iras = 5e9  # years
    
    # Compute exclusion fractions for M33 X-7 (no self-interactions first)
    excl_m33 = []
    loglik_m33 = []
    for mu in mu_grid_m33:
        frac = compute_exclusion_fraction(M_m33, a_m33, mu, tau_bh_years=tau_m33, f_a_GeV=None)
        ll = compute_likelihood(M_m33, a_m33, mu, tau_bh_years=tau_m33, f_a_GeV=None)
        excl_m33.append(frac)
        loglik_m33.append(ll)
    excl_m33 = np.array(excl_m33)
    loglik_m33 = np.array(loglik_m33)
    
    # Compute exclusion fractions for IRAS 09149-6206
    excl_iras = []
    loglik_iras = []
    for mu in mu_grid_iras:
        frac = compute_exclusion_fraction(M_iras, a_iras, mu, tau_bh_years=tau_iras, f_a_GeV=None)
        ll = compute_likelihood(M_iras, a_iras, mu, tau_bh_years=tau_iras, f_a_GeV=None)
        excl_iras.append(frac)
        loglik_iras.append(ll)
    excl_iras = np.array(excl_iras)
    loglik_iras = np.array(loglik_iras)
    
    # Compute 95% CL upper limits
    mu_upper_m33 = compute_upper_limit(mu_grid_m33, loglik_m33, mu_grid_m33[0])
    mu_upper_iras = compute_upper_limit(mu_grid_iras, loglik_iras, mu_grid_iras[0])
    
    print(f"  M33 X-7 95% CL upper limit on μ: {mu_upper_m33:.2e} eV")
    print(f"  IRAS 09149 95% CL upper limit on μ: {mu_upper_iras:.2e} eV")
    
    # Combined analysis: use joint likelihood
    mu_grid_combined = np.logspace(-19, -9.5, 400)
    loglik_combined = []
    excl_combined_m33 = []
    excl_combined_iras = []
    for mu in mu_grid_combined:
        ll_m33 = compute_likelihood(M_m33, a_m33, mu, tau_bh_years=tau_m33, f_a_GeV=None)
        ll_iras = compute_likelihood(M_iras, a_iras, mu, tau_bh_years=tau_iras, f_a_GeV=None)
        loglik_combined.append(ll_m33 + ll_iras)
        
        frac_m33 = compute_exclusion_fraction(M_m33, a_m33, mu, tau_bh_years=tau_m33, f_a_GeV=None)
        frac_iras = compute_exclusion_fraction(M_iras, a_iras, mu, tau_bh_years=tau_iras, f_a_GeV=None)
        excl_combined_m33.append(frac_m33)
        excl_combined_iras.append(frac_iras)
    
    loglik_combined = np.array(loglik_combined)
    excl_combined_m33 = np.array(excl_combined_m33)
    excl_combined_iras = np.array(excl_combined_iras)
    
    # Save results
    results = {
        'M33_X-7': {
            'mu_grid_eV': mu_grid_m33.tolist(),
            'exclusion_fraction': excl_m33.tolist(),
            'log_likelihood': loglik_m33.tolist(),
            'mu_upper_95CL_eV': float(mu_upper_m33),
            'tau_bh_years': tau_m33,
        },
        'IRAS_09149-6206': {
            'mu_grid_eV': mu_grid_iras.tolist(),
            'exclusion_fraction': excl_iras.tolist(),
            'log_likelihood': loglik_iras.tolist(),
            'mu_upper_95CL_eV': float(mu_upper_iras),
            'tau_bh_years': tau_iras,
        },
        'combined': {
            'mu_grid_eV': mu_grid_combined.tolist(),
            'log_likelihood': loglik_combined.tolist(),
        }
    }
    with open(os.path.join(OUTPUTS_DIR, 'exclusion_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # ============================================================
    # 5. Figure: Exclusion Likelihood Curves
    # ============================================================
    print("\n[5] Generating exclusion likelihood figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # M33 X-7
    ax = axes[0]
    delta_chi2_m33 = 2 * (np.max(loglik_m33) - loglik_m33)
    ax.plot(mu_grid_m33, excl_m33, 'b-', linewidth=2, label='Exclusion fraction')
    ax.axvline(mu_upper_m33, color='blue', linestyle='--', 
               label=f'95% CL: {mu_upper_m33:.1e} eV')
    ax.set_xscale('log')
    ax.set_xlabel(r'ULB Mass $\mu$ [eV]')
    ax.set_ylabel('Exclusion Fraction')
    ax.set_title('M33 X-7')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # IRAS 09149
    ax = axes[1]
    ax.plot(mu_grid_iras, excl_iras, 'r-', linewidth=2, label='Exclusion fraction')
    ax.axvline(mu_upper_iras, color='red', linestyle='--',
               label=f'95% CL: {mu_upper_iras:.1e} eV')
    ax.set_xscale('log')
    ax.set_xlabel(r'ULB Mass $\mu$ [eV]')
    ax.set_ylabel('Exclusion Fraction')
    ax.set_title('IRAS 09149-6206')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Combined Δχ²
    ax = axes[2]
    delta_chi2_combined = 2 * (np.max(loglik_combined) - loglik_combined)
    ax.plot(mu_grid_combined, delta_chi2_combined, 'k-', linewidth=2)
    ax.axhline(3.841, color='red', linestyle='--', label=r'95% CL ($\Delta\chi^2$ = 3.84)')
    ax.set_xscale('log')
    ax.set_xlabel(r'ULB Mass $\mu$ [eV]')
    ax.set_ylabel(r'$\Delta\chi^2$')
    ax.set_title('Combined Constraint')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'exclusion_likelihood.png'))
    plt.close()
    print("  → Saved report/images/exclusion_likelihood.png")
    
    # ============================================================
    # 6. Figure: Regge Plot with Exclusion Regions
    # ============================================================
    print("\n[6] Generating Regge plot figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # M33 X-7
    ax = axes[0]
    scatter = ax.scatter(M_m33, a_m33, c='blue', alpha=0.15, s=2, label='Posterior samples')
    ax.scatter([np.mean(M_m33)], [np.mean(a_m33)], c='darkblue', marker='*', s=200,
               zorder=5, label=f'Mean: M={np.mean(M_m33):.1f} Msun, a*={np.mean(a_m33):.3f}')
    
    # Add exclusion boundary for mu_upper_m33
    M_grid = np.logspace(np.log10(np.min(M_m33)*0.8), np.log10(np.max(M_m33)*1.2), 200)
    a_crit_m33 = []
    for M in M_grid:
        alpha_val = compute_alpha(M, mu_upper_m33)
        # Critical spin from superradiance condition: mu = m * Omega_H
        # For l=m=1: mu = Omega_H = a*/(2*r_g*(1+sqrt(1-a*^2)))
        # Solve for a* given mu and M
        from scipy.optimize import brentq
        def f(a):
            if a <= 0 or a >= 1:
                return 1e10
            return horizon_angular_velocity(a, M) - mu_upper_m33 * 1.602176634e-19 / 1.054571817e-34
        try:
            a_crit = brentq(f, 1e-6, 0.9999)
        except:
            a_crit = np.nan
        a_crit_m33.append(a_crit)
    a_crit_m33 = np.array(a_crit_m33)
    valid = ~np.isnan(a_crit_m33)
    ax.plot(M_grid[valid], a_crit_m33[valid], 'k--', linewidth=2, 
            label=f'Exclusion boundary (μ = {mu_upper_m33:.1e} eV)')
    
    ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]')
    ax.set_ylabel(r'Spin Parameter $a_*$')
    ax.set_title('M33 X-7: Regge Plot')
    ax.legend(fontsize=8, loc='lower right')
    
    # IRAS 09149-6206
    ax = axes[1]
    scatter = ax.scatter(M_iras / 1e8, a_iras, c='red', alpha=0.1, s=1, label='Posterior samples')
    ax.scatter([np.mean(M_iras) / 1e8], [np.mean(a_iras)], c='darkred', marker='*', s=200,
               zorder=5, label=f'Mean: M={np.mean(M_iras)/1e8:.2f}e8 Msun, a*={np.mean(a_iras):.3f}')
    
    # Add exclusion boundary for mu_upper_iras
    M_grid_iras = np.logspace(np.log10(np.min(M_iras)*0.8), np.log10(np.max(M_iras)*1.2), 200)
    a_crit_iras = []
    for M in M_grid_iras:
        alpha_val = compute_alpha(M, mu_upper_iras)
        def f(a):
            if a <= 0 or a >= 1:
                return 1e10
            return horizon_angular_velocity(a, M) - mu_upper_iras * 1.602176634e-19 / 1.054571817e-34
        try:
            a_crit = brentq(f, 1e-6, 0.9999)
        except:
            a_crit = np.nan
        a_crit_iras.append(a_crit)
    a_crit_iras = np.array(a_crit_iras)
    valid = ~np.isnan(a_crit_iras)
    ax.plot(M_grid_iras[valid] / 1e8, a_crit_iras[valid], 'k--', linewidth=2,
            label=f'Exclusion boundary (μ = {mu_upper_iras:.1e} eV)')
    
    ax.set_xlabel(r'Black Hole Mass $M$ [$10^8 M_\odot$]')
    ax.set_ylabel(r'Spin Parameter $a_*$')
    ax.set_title('IRAS 09149-6206: Regge Plot')
    ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'regge_plot.png'))
    plt.close()
    print("  → Saved report/images/regge_plot.png")
    
    # ============================================================
    # 7. Figure: Superradiance Timescale Map
    # ============================================================
    print("\n[7] Generating superradiance timescale figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for idx, (ax, M_samples, a_samples, mu_grid, tau_bh, name, cmap) in enumerate([
        (axes[0], M_m33, a_m33, mu_grid_m33, tau_m33, 'M33 X-7', 'Blues'),
        (axes[1], M_iras, a_iras, mu_grid_iras, tau_iras, 'IRAS 09149-6206', 'Reds'),
    ]):
        # For each sample, find the range of mu where superradiance is efficient
        M_vals = M_samples[::max(1, len(M_samples)//200)]
        a_vals = a_samples[::max(1, len(a_samples)//200)]
        
        # Build a 2D map: mu vs effective timescale
        n_mu = len(mu_grid)
        exclusion_map = np.zeros((n_mu,))
        for i, mu in enumerate(mu_grid):
            exclusion_map[i] = compute_exclusion_fraction(
                M_samples, a_samples, mu, tau_bh_years=tau_bh
            )
        
        # Create the plot
        ax.fill_between(mu_grid, 0, exclusion_map, alpha=0.5, color='steelblue' if idx==0 else 'coral')
        ax.plot(mu_grid, exclusion_map, 'k-', linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'ULB Mass $\mu$ [eV]')
        ax.set_ylabel('Exclusion Fraction')
        ax.set_title(f'{name}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Mark 95% exclusion threshold
        ax.axhline(0.95, color='red', linestyle=':', alpha=0.5, label='95% threshold')
        
        # Mark peak exclusion
        peak_idx = np.argmax(exclusion_map)
        ax.axvline(mu_grid[peak_idx], color='green', linestyle='--', alpha=0.5,
                   label=f'Peak: {mu_grid[peak_idx]:.1e} eV')
        
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'exclusion_fraction_map.png'))
    plt.close()
    print("  → Saved report/images/exclusion_fraction_map.png")
    
    # ============================================================
    # 8. Self-Interaction (Bosenova) Analysis
    # ============================================================
    print("\n[8] Computing self-interaction constraints...")
    
    # Grid of f_a values
    f_a_grid_GeV = np.logspace(14, 18, 80)  # GeV
    
    # For M33 X-7: use the peak sensitivity mass
    peak_mu_m33 = mu_grid_m33[np.argmax(excl_m33)]
    
    # Compute exclusion fraction as function of f_a at peak mu
    excl_vs_fa_m33 = []
    for f_a in f_a_grid_GeV:
        frac = compute_exclusion_fraction(M_m33, a_m33, peak_mu_m33, 
                                           tau_bh_years=tau_m33, f_a_GeV=f_a)
        excl_vs_fa_m33.append(frac)
    excl_vs_fa_m33 = np.array(excl_vs_fa_m33)
    
    # For IRAS: similar
    peak_mu_iras = mu_grid_iras[np.argmax(excl_iras)]
    excl_vs_fa_iras = []
    for f_a in f_a_grid_GeV:
        frac = compute_exclusion_fraction(M_iras, a_iras, peak_mu_iras,
                                           tau_bh_years=tau_iras, f_a_GeV=f_a)
        excl_vs_fa_iras.append(frac)
    excl_vs_fa_iras = np.array(excl_vs_fa_iras)
    
    # 2D analysis: exclusion fraction as function of both mu and f_a for M33 X-7
    n_mu_2d = 60
    mu_grid_2d = np.logspace(-13, -10, n_mu_2d)
    n_fa_2d = 50
    f_a_grid_2d = np.logspace(14, 18, n_fa_2d)
    
    excl_2d_m33 = np.zeros((n_mu_2d, n_fa_2d))
    for i, mu in enumerate(mu_grid_2d):
        for j, f_a in enumerate(f_a_grid_2d):
            excl_2d_m33[i, j] = compute_exclusion_fraction(
                M_m33, a_m33, mu, tau_bh_years=tau_m33, f_a_GeV=f_a
            )
    
    # Save self-interaction results
    si_results = {
        'M33_X-7': {
            'peak_mu_eV': float(peak_mu_m33),
            'f_a_grid_GeV': f_a_grid_GeV.tolist(),
            'exclusion_vs_fa': excl_vs_fa_m33.tolist(),
        },
        'IRAS_09149-6206': {
            'peak_mu_eV': float(peak_mu_iras),
            'f_a_grid_GeV': f_a_grid_GeV.tolist(),
            'exclusion_vs_fa': excl_vs_fa_iras.tolist(),
        },
        '2D_map_M33': {
            'mu_grid_eV': mu_grid_2d.tolist(),
            'f_a_grid_GeV': f_a_grid_2d.tolist(),
            'exclusion_map': excl_2d_m33.tolist(),
        }
    }
    with open(os.path.join(OUTPUTS_DIR, 'self_interaction_results.json'), 'w') as f:
        json.dump(si_results, f, indent=2)
    
    # ============================================================
    # 9. Figure: Self-Interaction Constraints
    # ============================================================
    print("\n[9] Generating self-interaction figures...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # M33 X-7: exclusion vs f_a
    ax = axes[0]
    ax.plot(f_a_grid_GeV, excl_vs_fa_m33, 'b-', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1e16, color='purple', linestyle='--', alpha=0.5, label=r'$f_a = 10^{16}$ GeV (GUT)')
    ax.set_xscale('log')
    ax.set_xlabel(r'Axion Decay Constant $f_a$ [GeV]')
    ax.set_ylabel('Exclusion Fraction')
    ax.set_title(f'M33 X-7: Self-Interaction Effects\n(at μ = {peak_mu_m33:.1e} eV)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 2D map: exclusion fraction in (mu, f_a) plane for M33 X-7
    ax = axes[1]
    X, Y = np.meshgrid(f_a_grid_2d, mu_grid_2d)
    c = ax.pcolormesh(X, Y, excl_2d_m33, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Axion Decay Constant $f_a$ [GeV]')
    ax.set_ylabel(r'ULB Mass $\mu$ [eV]')
    ax.set_title('M33 X-7: Exclusion Map\n(μ vs f_a)')
    plt.colorbar(c, ax=ax, label='Exclusion Fraction')
    
    # Add GUT scale line
    ax.axvline(1e16, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(1.2e16, 1.5e-10, 'GUT', color='purple', fontsize=9, ha='left')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'self_interaction_constraints.png'))
    plt.close()
    print("  → Saved report/images/self_interaction_constraints.png")
    
    # ============================================================
    # 10. Figure: Combined Superradiance Timescale Visualization
    # ============================================================
    print("\n[10] Generating timescale visualization figure...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create a grid of BH mass and mu for timescale computation
    M_grid_viz = np.logspace(0, 10, 300)  # 1 to 10^10 Msun
    mu_grid_viz = np.logspace(-20, -9, 300)  # eV
    
    # For efficiency, compute on a coarser grid and interpolate
    n_M, n_mu = 100, 100
    M_coarse = np.logspace(0, 10, n_M)
    mu_coarse = np.logspace(-20, -9, n_mu)
    
    tau_map = np.zeros((n_mu, n_M))
    for i, mu in enumerate(mu_coarse):
        for j, M in enumerate(M_coarse):
            if M < 3:  # Skip below 3 Msun
                tau_map[i, j] = np.inf
                continue
            alpha_val = compute_alpha(M, mu)
            a_val = 0.9  # Assume near-maximal spin
            tau = superradiance_timescale(alpha_val, a_val, M, mu)
            if alpha_val < 0.01 or alpha_val > 5:
                tau = np.inf
            tau_map[i, j] = tau
    
    # Plot the timescale map
    tau_map_masked = np.ma.masked_where(tau_map > 1e12, tau_map)
    X, Y = np.meshgrid(M_coarse, mu_coarse)
    c = ax.pcolormesh(X, Y, np.log10(tau_map_masked), cmap='viridis_r', 
                       shading='auto', vmin=-5, vmax=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]')
    ax.set_ylabel(r'ULB Mass $\mu$ [eV]')
    ax.set_title('Superradiance E-folding Timescale [log10(years)]')
    plt.colorbar(c, ax=ax, label=r'$\log_{10}(\tau_{\rm SR} / {\rm yr})$')
    
    # Mark the two BH systems
    ax.scatter([np.mean(M_m33)], [peak_mu_m33], marker='*', s=300, 
               color='white', edgecolors='black', linewidths=1.5, zorder=10,
               label=f'M33 X-7 (μ = {peak_mu_m33:.1e} eV)')
    ax.scatter([np.mean(M_iras)], [peak_mu_iras], marker='*', s=300,
               color='white', edgecolors='red', linewidths=1.5, zorder=10,
               label=f'IRAS 09149 (μ = {peak_mu_iras:.1e} eV)')
    
    # Add age of universe line
    tau_universe = 1.38e10  # years
    ax.contour(X, Y, tau_map, levels=[tau_universe], colors=['white'], 
               linewidths=2, linestyles='--')
    ax.text(5e6, 3e-14, r'$\tau_{\rm SR} = \tau_{\rm Universe}$', 
            color='white', fontsize=9, rotation=-25)
    
    ax.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'timescale_map.png'))
    plt.close()
    print("  → Saved report/images/timescale_map.png")
    
    # ============================================================
    # 11. Summary Table
    # ============================================================
    print("\n[11] Saving summary results...")
    
    final_results = {
        'upper_limits_95CL': {
            'M33_X-7_mu_eV': float(mu_upper_m33),
            'IRAS_09149-6206_mu_eV': float(mu_upper_iras),
        },
        'peak_sensitivity': {
            'M33_X-7_mu_eV': float(peak_mu_m33),
            'M33_X-7_exclusion_fraction': float(excl_m33[np.argmax(excl_m33)]),
            'IRAS_09149-6206_mu_eV': float(peak_mu_iras),
            'IRAS_09149-6206_exclusion_fraction': float(excl_iras[np.argmax(excl_iras)]),
        },
        'method': 'Bayesian exclusion from posterior samples',
        'superradiance_model': 'Scalar field, l=m=1 dominant mode',
        'alpha_factor': float(ALPHA_FACTOR),
    }
    with open(os.path.join(OUTPUTS_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  M33 X-7 95% CL upper limit on μ:    {mu_upper_m33:.2e} eV")
    print(f"  IRAS 09149 95% CL upper limit on μ:  {mu_upper_iras:.2e} eV")
    print(f"  M33 X-7 peak sensitivity: μ = {peak_mu_m33:.2e} eV")
    print(f"  IRAS 09149 peak sensitivity: μ = {peak_mu_iras:.2e} eV")


if __name__ == "__main__":
    main()
