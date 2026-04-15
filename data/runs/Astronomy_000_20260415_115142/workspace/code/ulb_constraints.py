#!/usr/bin/env python3
"""
Ultralight Boson (ULB) Constraint Analysis using Black Hole Superradiance

This module implements a Bayesian statistical framework to constrain ULB properties
using black hole mass and spin measurements. The method leverages the superradiance
instability that occurs when a boson's Compton wavelength matches a black hole's size.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
import json
import os

# Set plotting style
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

# Physical constants
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
hbar = 1.055e-34  # J s
Msun = 1.989e30  # kg
Mpc = 3.086e22   # m

# Conversion: M (in solar masses) * mu (in eV) -> dimensionless alpha
ALPHA_CONV = 1.477  # km/Msun * (1 eV in 1/km)


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
                mass = float(parts[0])
                spin = float(parts[1])
                samples.append([mass, spin])
    return np.array(samples)


def horizon_angular_velocity(a_star, M):
    """
    Calculate the horizon angular velocity Omega_H in Hz.
    """
    r_g = 1.477e3 * M  # meters (gravitational radius)
    Omega_H = (a_star / (2 * r_g)) / (1 + np.sqrt(1 - a_star**2))
    return Omega_H


def compute_alpha(M, mu):
    """
    Compute dimensionless coupling alpha = G*M*mu/(hbar*c).
    
    Parameters:
    -----------
    M : float or array
        BH mass in solar masses
    mu : float or array
        Boson mass in eV
    
    Returns:
    --------
    alpha : float or array
        Dimensionless coupling
    """
    return ALPHA_CONV * M * mu


def superradiance_condition(mu, a_star, M, l=1, m=1):
    """
    Check if superradiance condition is satisfied for a given boson mass.
    
    Returns:
    --------
    is_superradiant : bool or array
    alpha : float or array
    mu_crit : float or array
        Critical boson mass for superradiance
    """
    alpha = compute_alpha(M, mu)
    
    # Horizon angular velocity
    Omega_H = horizon_angular_velocity(a_star, M)
    
    # Critical boson mass (approximate, for small alpha)
    h_eVs = 4.136e-15  # eV*s
    mu_crit = m * Omega_H * h_eVs  # eV
    
    is_sr = mu < mu_crit
    
    return is_sr, alpha, mu_crit


def superradiance_timescale(alpha, a_star, M, l=1, m=1):
    """
    Calculate the superradiance growth timescale.
    
    For l=m=1 mode and small alpha:
    tau = 48 * M / (a_star * alpha^9)
    
    Returns:
    --------
    tau : float or array
        Growth timescale in seconds
    """
    M_kg = M * Msun
    M_sec = G * M_kg / c**3  # Mass in seconds (geometric units)
    
    if l == 1 and m == 1:
        M_omega_I = (a_star / 48.0) * alpha**9
    else:
        M_omega_I = (a_star / 48.0) * alpha**(4*l + 4)
    
    # Avoid division by zero
    M_omega_I = np.maximum(M_omega_I, 1e-100)
    
    tau = M_sec / M_omega_I
    return tau


def exclusion_probability(mu, samples, age=1e10, l=1, m=1):
    """
    Compute the exclusion probability for a given boson mass.
    
    The exclusion probability is the fraction of posterior samples where
    superradiance would have had time to spin down the BH.
    
    Parameters:
    -----------
    mu : float
        Boson mass in eV
    samples : array (N, 2)
        Posterior samples of [mass, spin]
    age : float
        Age of the system in years (default: 10 Gyr)
    l, m : int
        Quantum numbers
    
    Returns:
    --------
    prob : float
        Exclusion probability (0 to 1)
    """
    M_samples = samples[:, 0]
    a_samples = samples[:, 1]
    
    # Check superradiance condition
    is_sr, alpha, mu_crit = superradiance_condition(mu, a_samples, M_samples, l, m)
    
    # Compute growth timescales
    tau = superradiance_timescale(alpha, a_samples, M_samples, l, m)
    
    # Convert age to seconds
    age_sec = age * 3.154e7
    
    # Exclusion: superradiance is possible AND timescale is shorter than age
    excluded = is_sr & (tau < age_sec)
    
    prob = np.mean(excluded)
    return prob


def compute_exclusion_curve(samples, mu_range, age=1e10, l=1, m=1):
    """
    Compute exclusion probability curve over a range of boson masses.
    """
    probs = []
    for mu in mu_range:
        prob = exclusion_probability(mu, samples, age, l, m)
        probs.append(prob)
    return np.array(probs)


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
    ax.axvline(np.median(M), color='red', linestyle='--', label=f'Median: {np.median(M):.2f}')
    ax.legend()
    ax.set_title('Mass Distribution')
    
    # Spin histogram
    ax = axes[1, 1]
    ax.hist(a, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Dimensionless Spin $a_*$')
    ax.set_ylabel('Count')
    ax.axvline(np.median(a), color='red', linestyle='--', label=f'Median: {np.median(a):.3f}')
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


def plot_exclusion_curves(samples_dict, mu_ranges, output_path, age=1e10):
    """Plot exclusion probability curves for multiple systems."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'M33_X-7': 'steelblue', 'IRAS_09149-6206': 'darkorange'}
    
    for name, samples in samples_dict.items():
        mu_range = mu_ranges[name]
        probs = compute_exclusion_curve(samples, mu_range, age)
        
        # Plot on linear scale
        axes[0].plot(mu_range * 1e12, probs, label=name, color=colors[name], linewidth=2)
        
        # Plot on log scale
        axes[1].semilogx(mu_range, probs, label=name, color=colors[name], linewidth=2)
        
        # 95% exclusion level
        if np.any(probs >= 0.95):
            mu_95_idx = np.where(probs >= 0.95)[0][0]
            mu_95 = mu_range[mu_95_idx]
            print(f"{name}: 95% exclusion at mu = {mu_95:.2e} eV")
    
    for ax in axes:
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% exclusion')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% exclusion')
        ax.set_ylabel('Exclusion Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_xlabel('ULB Mass $\\mu$ [$\\times 10^{-12}$ eV]')
    axes[0].set_title('Exclusion Probability (Linear Scale)')
    
    axes[1].set_xlabel('ULB Mass $\\mu$ [eV]')
    axes[1].set_title('Exclusion Probability (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_regge_constraints(samples_dict, mu_test, output_path):
    """
    Plot Regge plane (mass vs spin) with superradiance constraints.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, samples) in enumerate(samples_dict.items()):
        ax = axes[idx]
        
        M = samples[:, 0]
        a = samples[:, 1]
        
        # Scatter plot
        ax.scatter(M, a, alpha=0.3, s=1, c='navy')
        
        # Plot superradiance boundaries for different mu values
        M_grid = np.linspace(M.min(), M.max(), 100)
        
        for mu in mu_test:
            alpha_grid = compute_alpha(M_grid, mu)
            
            # Critical spin for which omega = m*Omega_H
            # Omega_H = (a/(2r_g))/(1+sqrt(1-a^2))
            # For superradiance: mu < m*Omega_H (in appropriate units)
            # This gives a minimum spin requirement
            
            # Simplified: approximate boundary curve
            # For small alpha, superradiance occurs when mu < mu_crit
            # where mu_crit = m*Omega_H * h_eVs
            
            h_eVs = 4.136e-15
            # Omega_H = a / (4*pi*r_g) for a->1... need to solve numerically
            
            # Approximate: for given mu and M, find minimum a for SR
            # mu = m * Omega_H * h_eVs
            # Omega_H = mu / (m * h_eVs)
            
            Omega_H_needed = mu / (1 * h_eVs)
            
            # Convert to dimensionless: Omega_H * r_g / c = (a/(2*(1+sqrt(1-a^2))))
            # This is messy to invert analytically; use approximation
            
            # For high spin: Omega_H ≈ 1/(4*r_g)
            # So a_crit ≈ 4*r_g * Omega_H_needed * c / (dimensionless factor)
            
            r_g_grid = 1.477e3 * M_grid  # meters
            
            # Approximate spin threshold
            # Using Omega_H = a/(4*r_g) for a≈1
            # a_crit = 4 * r_g * Omega_H_needed / c
            
            a_crit = 4 * r_g_grid * Omega_H_needed / c
            a_crit = np.clip(a_crit, 0, 0.99)
            
            label = f'$\\mu = {mu:.1e}$ eV'
            ax.plot(M_grid, a_crit, '--', alpha=0.7, label=label)
        
        ax.set_xlabel('BH Mass [M$_\\odot$]')
        ax.set_ylabel('Dimensionless Spin $a_*$')
        ax.set_title(f'{name}: Regge Plane with SR Constraints')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_upper_limits(samples_dict, age=1e10):
    """Compute 95% upper limits on ULB masses."""
    results = {}
    
    for name, samples in samples_dict.items():
        # Determine appropriate mass range for this BH
        M_median = np.median(samples[:, 0])
        
        # Optimal coupling: alpha ≈ 0.42
        # mu_opt = 0.42 / (ALPHA_CONV * M)
        mu_opt = 0.42 / (ALPHA_CONV * M_median)
        
        # Scan around optimal value
        mu_range = np.logspace(np.log10(mu_opt * 0.1), np.log10(mu_opt * 10), 200)
        
        probs = compute_exclusion_curve(samples, mu_range, age)
        
        # Find 95% limit
        if np.any(probs >= 0.95):
            mu_95_idx = np.where(probs >= 0.95)[0][0]
            mu_95 = mu_range[mu_95_idx]
        else:
            mu_95 = None
        
        # Find 50% limit
        if np.any(probs >= 0.5):
            mu_50_idx = np.where(probs >= 0.5)[0][0]
            mu_50 = mu_range[mu_50_idx]
        else:
            mu_50 = None
        
        results[name] = {
            'mu_95': mu_95,
            'mu_50': mu_50,
            'mu_opt': mu_opt,
            'M_median': M_median,
            'a_median': np.median(samples[:, 1]),
            'probs': probs,
            'mu_range': mu_range
        }
    
    return results


def plot_combined_constraints(samples_dict, results, output_path):
    """Create a comprehensive constraints plot."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {'M33_X-7': 'steelblue', 'IRAS_09149-6206': 'darkorange'}
    
    # Row 1: Exclusion probability curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, res in results.items():
        ax1.semilogx(res['mu_range'], res['probs'], 
                     label=name, color=colors[name], linewidth=2)
        if res['mu_95']:
            ax1.axvline(res['mu_95'], color=colors[name], linestyle='--', alpha=0.5)
    ax1.axhline(0.95, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('ULB Mass $\\mu$ [eV]')
    ax1.set_ylabel('Exclusion Probability')
    ax1.set_title('ULB Mass Exclusion Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Row 2-3: Individual system details
    for idx, (name, samples) in enumerate(samples_dict.items()):
        ax_m = fig.add_subplot(gs[1, idx])
        ax_s = fig.add_subplot(gs[2, idx])
        
        M = samples[:, 0]
        a = samples[:, 1]
        
        # Mass distribution
        ax_m.hist(M, bins=50, color=colors[name], edgecolor='white', alpha=0.7)
        ax_m.axvline(np.median(M), color='red', linestyle='--')
        ax_m.set_xlabel('BH Mass [M$_\\odot$]')
        ax_m.set_ylabel('Count')
        ax_m.set_title(f'{name}: Mass Distribution')
        
        # Spin distribution
        ax_s.hist(a, bins=50, color=colors[name], edgecolor='white', alpha=0.7)
        ax_s.axvline(np.median(a), color='red', linestyle='--')
        ax_s.set_xlabel('Dimensionless Spin $a_*$')
        ax_s.set_ylabel('Count')
        ax_s.set_title(f'{name}: Spin Distribution')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_results_table(results, output_path):
    """Save results to a formatted table."""
    with open(output_path, 'w') as f:
        f.write("# Ultralight Boson Constraint Results\n")
        f.write("#\n")
        f.write("| System | M_BH [Msun] | a_* | mu_opt [eV] | mu_95 [eV] |\n")
        f.write("|--------|-------------|-----|-------------|------------|\n")
        
        for name, res in results.items():
            mu_95_str = f"{res['mu_95']:.2e}" if res['mu_95'] else "N/A"
            f.write(f"| {name} | {res['M_median']:.2e} | {res['a_median']:.3f} | "
                   f"{res['mu_opt']:.2e} | {mu_95_str} |\n")
    print(f"Saved: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Ultralight Boson Constraint Analysis")
    print("Using Black Hole Superradiance")
    print("=" * 60)
    
    # Create output directories
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
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for name, samples in samples_dict.items():
        M = samples[:, 0]
        a = samples[:, 1]
        print(f"\n{name}:")
        print(f"  Mass: {np.median(M):.3e} +/- {np.std(M):.3e} Msun")
        print(f"  Spin: {np.median(a):.3f} +/- {np.std(a):.3f}")
    
    # Generate data overview plots
    print("\nGenerating data overview plots...")
    plot_mass_spin_distribution(m33_samples, 'M33 X-7: Stellar-mass Black Hole',
                                '../report/images/m33_corner.png')
    plot_mass_spin_distribution(iras_samples, 'IRAS 09149-6206: Supermassive Black Hole',
                                '../report/images/iras_corner.png')
    
    # Compute upper limits
    print("\nComputing ULB mass upper limits...")
    results = compute_upper_limits(samples_dict)
    
    # Print results
    print("\nResults:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Median BH mass: {res['M_median']:.3e} Msun")
        print(f"  Median BH spin: {res['a_median']:.3f}")
        print(f"  Optimal ULB mass (alpha=0.42): {res['mu_opt']:.2e} eV")
        if res['mu_95']:
            print(f"  95% upper limit: {res['mu_95']:.2e} eV")
        if res['mu_50']:
            print(f"  50% limit: {res['mu_50']:.2e} eV")
    
    # Generate exclusion curves
    print("\nGenerating exclusion curves...")
    mu_ranges = {
        'M33_X-7': np.linspace(1e-13, 1e-11, 100),
        'IRAS_09149-6206': np.linspace(1e-21, 1e-19, 100)
    }
    plot_exclusion_curves(samples_dict, mu_ranges, 
                         '../report/images/exclusion_curves.png')
    
    # Generate Regge plane plots
    print("\nGenerating Regge plane constraints...")
    mu_test = [1e-12, 5e-12, 1e-11]  # For M33 X-7
    plot_regge_constraints({'M33_X-7': m33_samples}, mu_test,
                          '../report/images/regge_m33.png')
    
    mu_test_smbh = [1e-20, 5e-20, 1e-19]  # For IRAS
    plot_regge_constraints({'IRAS_09149-6206': iras_samples}, mu_test_smbh,
                          '../report/images/regge_iras.png')
    
    # Generate combined constraints plot
    print("\nGenerating combined constraints plot...")
    plot_combined_constraints(samples_dict, results,
                             '../report/images/combined_constraints.png')
    
    # Save results table
    save_results_table(results, '../outputs/ulb_limits_table.md')
    
    # Save numerical results as JSON
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            'M_median': float(res['M_median']),
            'a_median': float(res['a_median']),
            'mu_opt_eV': float(res['mu_opt']),
            'mu_95_eV': float(res['mu_95']) if res['mu_95'] else None,
            'mu_50_eV': float(res['mu_50']) if res['mu_50'] else None
        }
    
    with open('../outputs/ulb_constraints.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: ../outputs/ulb_constraints.json")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = main()
