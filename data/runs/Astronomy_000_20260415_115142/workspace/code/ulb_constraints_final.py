#!/usr/bin/env python3
"""
Ultralight Boson (ULB) Constraint Analysis using Black Hole Superradiance - Final Version

This version properly implements the constraint logic:
- High observed BH spin implies that superradiance has NOT occurred (or is slow)
- We compute the probability that a given ULB mass would have caused observable spin-down
- If this probability is high, the observation of high spin excludes that ULB mass
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import json
import os

# Set plotting style
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

# Physical constants
G = 6.674e-11
c = 2.998e8
hbar = 1.055e-34
Msun = 1.989e30
year_to_sec = 3.154e7
KM_PER_MSUN = 1.477
EV_TO_PER_M = 5.068e6


def load_posterior_samples(filepath):
    """Load posterior samples from data file."""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) >= 2:
                samples.append([float(parts[0]), float(parts[1])])
    return np.array(samples)


def compute_alpha(M_msun, mu_eV):
    """Compute dimensionless coupling alpha = G*M*mu/(hbar*c)."""
    r_g_m = KM_PER_MSUN * 1e3 * M_msun
    mu_per_m = mu_eV * EV_TO_PER_M
    return r_g_m * mu_per_m


def horizon_angular_velocity_hz(a_star, M_msun):
    """Calculate the horizon angular velocity Omega_H in Hz."""
    r_g_m = KM_PER_MSUN * 1e3 * M_msun
    Omega_H = (a_star / (2 * r_g_m)) / (1 + np.sqrt(1 - a_star**2))
    return Omega_H


def superradiance_timescale_years(alpha, a_star, M_msun, l=1, m=1):
    """Calculate the superradiance e-folding timescale in years."""
    M_geom = G * M_msun * Msun / c**3
    
    if l == 1 and m == 1:
        M_omega_I = (a_star / 48.0) * alpha**9
    else:
        M_omega_I = (a_star / 48.0) * alpha**(4*l + 4)
    
    M_omega_I = np.maximum(M_omega_I, 1e-200)
    tau_sec = M_geom / M_omega_I
    return tau_sec / year_to_sec


def critical_boson_mass(a_star, M_msun, m=1):
    """Calculate the critical boson mass for superradiance in eV."""
    Omega_H = horizon_angular_velocity_hz(a_star, M_msun)
    h_eVs = 4.136e-15
    return m * Omega_H * h_eVs


def compute_exclusion_probability(mu_eV, samples, age_years=1e10, l=1, m=1):
    """
    Compute the exclusion probability for a given boson mass.
    
    Key logic: If a ULB with mass mu exists and SR is fast (tau << age),
    the BH would be spun down. Since we observe HIGH spin, this excludes mu.
    
    We compute: P(SR would have occurred | samples, mu)
    """
    M_samples = samples[:, 0]
    a_samples = samples[:, 1]
    
    alpha = compute_alpha(M_samples, mu_eV)
    mu_crit = critical_boson_mass(a_samples, M_samples, m)
    
    # SR is possible when mu < mu_crit (frequency condition satisfied)
    sr_possible = mu_eV < mu_crit
    
    # SR growth timescale
    tau = superradiance_timescale_years(alpha, a_samples, M_samples, l, m)
    
    # SR is "dangerous" (would cause observable spin-down) when:
    # 1. SR is possible (mu < mu_crit)
    # 2. SR is fast (tau < age)
    # 3. alpha is in the relevant range (not too small)
    
    # For the l=1 mode, the strongest instability is around alpha ≈ 0.42
    # The growth rate scales as alpha^9, so we need alpha not too small
    
    sr_dangerous = sr_possible & (tau < age_years) & (alpha > 0.01)
    
    # Exclusion probability = fraction of samples where SR would have occurred
    return np.mean(sr_dangerous)


def find_upper_limit(samples, mu_min, mu_max, age_years=1e10,
                     confidence_levels=[0.5, 0.9, 0.95]):
    """Find upper limits at specified confidence levels."""
    mu_range = np.logspace(np.log10(mu_min), np.log10(mu_max), 500)
    probs = [compute_exclusion_probability(mu, samples, age_years) for mu in mu_range]
    probs = np.array(probs)
    
    limits = {}
    for cl in confidence_levels:
        if np.any(probs >= cl):
            idx = np.where(probs >= cl)[0]
            if len(idx) > 0:
                limits[f'{int(cl*100)}%'] = mu_range[idx[0]]
        else:
            limits[f'{int(cl*100)}%'] = None
    
    return mu_range, probs, limits


def plot_mass_spin_distribution(samples, title, output_path):
    """Create corner plot of mass-spin posterior distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    M = samples[:, 0]
    a = samples[:, 1]
    
    # Mass histogram
    ax = axes[0, 0]
    ax.hist(M, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('BH Mass [M$_\\odot$]')
    ax.set_ylabel('Count')
    ax.axvline(np.median(M), color='red', linestyle='--',
               label=f'Median: {np.median(M):.2e}')
    ax.legend()
    ax.set_title('Mass Distribution')
    
    # Spin histogram
    ax = axes[1, 1]
    ax.hist(a, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Dimensionless Spin $a_*$')
    ax.set_ylabel('Count')
    ax.axvline(np.median(a), color='red', linestyle='--',
               label=f'Median: {np.median(a):.3f}')
    ax.legend()
    ax.set_title('Spin Distribution')
    
    # 2D scatter
    ax = axes[1, 0]
    ax.scatter(M, a, alpha=0.3, s=1, c='navy')
    ax.set_xlabel('BH Mass [M$_\\odot$]')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_title('Mass-Spin Correlation')
    
    # 2D histogram
    ax = axes[0, 1]
    h = ax.hist2d(M, a, bins=50, cmap='Blues')
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('BH Mass [M$_\\odot$]')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_title('2D Histogram')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exclusion_curves(results, output_path):
    """Plot exclusion probability curves for multiple systems."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'M33_X-7': 'steelblue', 'IRAS_09149-6206': 'darkorange'}
    
    for name, res in results.items():
        mu_range = res['mu_range']
        probs = res['probs']
        
        if 'M33' in name:
            ax = axes[0]
            ax.set_xlim(1e-13, 1e-10)
        else:
            ax = axes[1]
            ax.set_xlim(1e-20, 1e-17)
        
        ax.semilogx(mu_range, probs, label=name, color=colors[name], linewidth=2.5)
        
        # Mark limits
        for cl_key, mu_val in res['limits'].items():
            if mu_val is not None:
                cl_num = int(cl_key.replace('%', ''))
                ax.axvline(mu_val, color=colors[name], linestyle='--', alpha=0.5)
                ax.annotate(f'{cl_key}: {mu_val:.1e} eV',
                           xy=(mu_val, cl_num/100), fontsize=9,
                           xytext=(mu_val*2, cl_num/100 + 0.1),
                           color=colors[name])
    
    for ax in axes:
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% CL')
        ax.axhline(0.90, color='orange', linestyle='--', alpha=0.7, label='90% CL')
        ax.axhline(0.50, color='gray', linestyle=':', alpha=0.5, label='50% CL')
        ax.set_ylabel('Exclusion Probability', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
    
    axes[0].set_title('M33 X-7: Stellar-mass BH Constraints', fontsize=13)
    axes[1].set_title('IRAS 09149-6206: Supermassive BH Constraints', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_regge_constraints(samples_dict, mu_test_dict, output_path):
    """Plot Regge plane (mass vs spin) with superradiance constraints."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, samples) in enumerate(samples_dict.items()):
        ax = axes[idx]
        
        M = samples[:, 0]
        a = samples[:, 1]
        
        ax.scatter(M, a, alpha=0.2, s=2, c='navy', label='Posterior samples')
        
        M_grid = np.linspace(M.min(), M.max(), 100)
        mu_test = mu_test_dict[name]
        
        for mu in mu_test:
            h_eVs = 4.136e-15
            Omega_H_needed = mu / h_eVs
            
            r_g = KM_PER_MSUN * 1e3 * M_grid
            
            a_crit = np.zeros_like(M_grid)
            for i, rg in enumerate(r_g):
                Omega_H_max = 1 / (4 * rg)
                if Omega_H_needed >= Omega_H_max:
                    a_crit[i] = 0.0
                else:
                    a_low, a_high = 0.0, 0.999
                    for _ in range(50):
                        a_mid = (a_low + a_high) / 2
                        Omega_mid = (a_mid / (2*rg)) / (1 + np.sqrt(1-a_mid**2))
                        if Omega_mid < Omega_H_needed:
                            a_low = a_mid
                        else:
                            a_high = a_mid
                    a_crit[i] = a_mid
            
            a_crit = np.clip(a_crit, 0, 0.99)
            label = f'$\\mu = {mu:.1e}$ eV'
            ax.plot(M_grid, a_crit, '--', alpha=0.8, linewidth=2, label=label)
        
        ax.set_xlabel('BH Mass [M$_\\odot$]', fontsize=12)
        ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
        ax.set_title(f'{name}: Regge Plane Constraints', fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_summary(samples_dict, results, output_path):
    """Create a comprehensive summary plot."""
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    colors = {'M33_X-7': 'steelblue', 'IRAS_09149-6206': 'darkorange'}
    
    # Row 1: Combined exclusion curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, res in results.items():
        ax1.semilogx(res['mu_range'], res['probs'],
                     label=name, color=colors[name], linewidth=2.5)
        for cl_key, mu_val in res['limits'].items():
            if mu_val is not None and '95%' in cl_key:
                ax1.axvline(mu_val, color=colors[name], linestyle='--', alpha=0.5)
                ax1.annotate(f'{name} 95%: {mu_val:.1e} eV',
                            xy=(mu_val, 0.95), xytext=(mu_val*2, 0.85),
                            color=colors[name], fontsize=9)
    ax1.axhline(0.95, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
    ax1.set_ylabel('Exclusion Probability', fontsize=12)
    ax1.set_title('ULB Mass Exclusion Probabilities', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Rows 2-4: Individual system details
    for idx, (name, samples) in enumerate(samples_dict.items()):
        res = results[name]
        
        # Mass distribution
        ax_m = fig.add_subplot(gs[1, idx])
        M = samples[:, 0]
        ax_m.hist(M, bins=50, color=colors[name], edgecolor='white', alpha=0.7)
        ax_m.axvline(np.median(M), color='red', linestyle='--', linewidth=2)
        ax_m.set_xlabel('BH Mass [M$_\\odot$]', fontsize=11)
        ax_m.set_ylabel('Count', fontsize=11)
        ax_m.set_title(f'{name}: M = {np.median(M):.2e} M$_\\odot$', fontsize=12)
        
        # Spin distribution
        ax_s = fig.add_subplot(gs[2, idx])
        a = samples[:, 1]
        ax_s.hist(a, bins=50, color=colors[name], edgecolor='white', alpha=0.7)
        ax_s.axvline(np.median(a), color='red', linestyle='--', linewidth=2)
        ax_s.set_xlabel('Dimensionless Spin $a_*$', fontsize=11)
        ax_s.set_ylabel('Count', fontsize=11)
        ax_s.set_title(f'{name}: a$_*$ = {np.median(a):.3f}', fontsize=12)
        
        # Individual exclusion curve
        ax_e = fig.add_subplot(gs[3, idx])
        ax_e.semilogx(res['mu_range'], res['probs'], color=colors[name], linewidth=2.5)
        for cl_key, mu_val in res['limits'].items():
            if mu_val is not None:
                cl_num = int(cl_key.replace('%', ''))
                ax_e.axvline(mu_val, color=colors[name], linestyle='--', alpha=0.5)
                ax_e.axhline(cl_num/100, color='gray', linestyle=':', alpha=0.3)
        ax_e.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=11)
        ax_e.set_ylabel('Exclusion Probability', fontsize=11)
        ax_e.set_title(f'{name}: Exclusion Probability', fontsize=12)
        ax_e.grid(True, alpha=0.3)
        ax_e.set_ylim(-0.05, 1.05)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("Ultralight Boson Constraint Analysis - Final")
    print("Using Black Hole Superradiance")
    print("=" * 70)
    
    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../report/images', exist_ok=True)
    
    # Load data
    print("\nLoading posterior samples...")
    m33_samples = load_posterior_samples('../data/M33_X-7_samples.dat')
    iras_samples = load_posterior_samples('../data/IRAS_09149-6206_samples.dat')
    
    print(f"M33 X-7: {len(m33_samples)} samples")
    print(f"IRAS 09149-6206: {len(iras_samples)} samples")
    
    samples_dict = {
        'M33_X-7': m33_samples,
        'IRAS_09149-6206': iras_samples
    }
    
    # Summary statistics
    print("\n" + "-" * 50)
    print("Summary Statistics:")
    print("-" * 50)
    for name, samples in samples_dict.items():
        M = samples[:, 0]
        a = samples[:, 1]
        mu_opt = 0.42 / (KM_PER_MSUN * 1e3 * EV_TO_PER_M * np.median(M))
        print(f"\n{name}:")
        print(f"  Mass: {np.median(M):.3e} ± {np.std(M):.3e} M_sun")
        print(f"  Spin: {np.median(a):.3f} ± {np.std(a):.3f}")
        print(f"  Optimal probe ULB mass: {mu_opt:.2e} eV")
    
    # Generate data overview plots
    print("\n" + "-" * 50)
    print("Generating data overview plots...")
    print("-" * 50)
    plot_mass_spin_distribution(m33_samples, 'M33 X-7: Stellar-mass Black Hole',
                                '../report/images/m33_corner.png')
    plot_mass_spin_distribution(iras_samples, 'IRAS 09149-6206: Supermassive Black Hole',
                                '../report/images/iras_corner.png')
    
    # Compute upper limits
    print("\n" + "-" * 50)
    print("Computing ULB mass upper limits...")
    print("-" * 50)
    
    results = {}
    
    # M33 X-7: stellar-mass BH
    print("\nAnalyzing M33 X-7...")
    M_median = np.median(m33_samples[:, 0])
    mu_opt_m33 = 0.42 / (KM_PER_MSUN * 1e3 * EV_TO_PER_M * M_median)
    # Wider range to capture the exclusion region
    mu_range_m33, probs_m33, limits_m33 = find_upper_limit(
        m33_samples, mu_opt_m33 * 0.001, mu_opt_m33 * 100, age_years=1e10
    )
    results['M33_X-7'] = {
        'mu_range': mu_range_m33,
        'probs': probs_m33,
        'limits': limits_m33,
        'M_median': M_median,
        'a_median': np.median(m33_samples[:, 1]),
        'M_std': np.std(m33_samples[:, 0]),
        'a_std': np.std(m33_samples[:, 1]),
        'mu_opt': mu_opt_m33
    }
    
    # IRAS: supermassive BH
    print("\nAnalyzing IRAS 09149-6206...")
    M_median = np.median(iras_samples[:, 0])
    mu_opt_iras = 0.42 / (KM_PER_MSUN * 1e3 * EV_TO_PER_M * M_median)
    mu_range_iras, probs_iras, limits_iras = find_upper_limit(
        iras_samples, mu_opt_iras * 0.001, mu_opt_iras * 100, age_years=1e10
    )
    results['IRAS_09149-6206'] = {
        'mu_range': mu_range_iras,
        'probs': probs_iras,
        'limits': limits_iras,
        'M_median': M_median,
        'a_median': np.median(iras_samples[:, 1]),
        'M_std': np.std(iras_samples[:, 0]),
        'a_std': np.std(iras_samples[:, 1]),
        'mu_opt': mu_opt_iras
    }
    
    # Print results
    print("\n" + "-" * 50)
    print("Results:")
    print("-" * 50)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  BH mass: {res['M_median']:.3e} ± {res['M_std']:.1e} M_sun")
        print(f"  BH spin: {res['a_median']:.3f} ± {res['a_std']:.3f}")
        print(f"  Optimal ULB mass: {res['mu_opt']:.2e} eV")
        for cl, mu in res['limits'].items():
            if mu:
                print(f"  {cl} ULB mass limit: {mu:.2e} eV")
    
    # Generate plots
    print("\n" + "-" * 50)
    print("Generating exclusion curves...")
    print("-" * 50)
    plot_exclusion_curves(results, '../report/images/exclusion_curves.png')
    
    print("\n" + "-" * 50)
    print("Generating Regge plane constraints...")
    print("-" * 50)
    mu_test = {
        'M33_X-7': [1e-12, 3.6e-12, 1e-11],
        'IRAS_09149-6206': [1e-19, 5e-19, 1e-18]
    }
    plot_regge_constraints(samples_dict, mu_test,
                          '../report/images/regge_constraints.png')
    
    print("\n" + "-" * 50)
    print("Generating combined summary plot...")
    print("-" * 50)
    plot_combined_summary(samples_dict, results,
                         '../report/images/combined_constraints.png')
    
    # Save results
    print("\n" + "-" * 50)
    print("Saving results...")
    print("-" * 50)
    
    with open('../outputs/ulb_limits_table.md', 'w') as f:
        f.write("# Ultralight Boson Constraint Results\n\n")
        f.write("| System | M_BH [Msun] | a_* | mu_opt [eV] | 90% Limit [eV] | 95% Limit [eV] |\n")
        f.write("|--------|-------------|-----|-------------|----------------|----------------|\n")
        for name, res in results.items():
            mu_90 = res['limits'].get('90%', None)
            mu_95 = res['limits'].get('95%', None)
            mu_90_str = f"{mu_90:.2e}" if mu_90 else "N/A"
            mu_95_str = f"{mu_95:.2e}" if mu_95 else "N/A"
            f.write(f"| {name} | {res['M_median']:.2e} ± {res['M_std']:.1e} | "
                   f"{res['a_median']:.3f} ± {res['a_std']:.3f} | "
                   f"{res['mu_opt']:.2e} | {mu_90_str} | {mu_95_str} |\n")
    print("Saved: ../outputs/ulb_limits_table.md")
    
    # Save JSON
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            'M_median_Msun': float(res['M_median']),
            'M_std_Msun': float(res['M_std']),
            'a_median': float(res['a_median']),
            'a_std': float(res['a_std']),
            'mu_opt_eV': float(res['mu_opt']),
            'limits': {k: float(v) if v else None for k, v in res['limits'].items()}
        }
    
    with open('../outputs/ulb_constraints.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print("Saved: ../outputs/ulb_constraints.json")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
