#!/usr/bin/env python3
"""
Ultralight Boson (ULB) Constraint Analysis using Black Hole Superradiance - Version 2

This module implements a corrected Bayesian statistical framework to constrain ULB properties.
Key corrections:
- Proper unit conversions for dimensionless coupling alpha
- Correct superradiance timescale calculations
- Proper exclusion probability computation
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
G = 6.674e-11       # m^3 kg^-1 s^-2
c = 2.998e8         # m/s
hbar = 1.055e-34    # J s
Msun = 1.989e30     # kg
eV_to_kg = 1.783e-36  # kg/eV
year_to_sec = 3.154e7  # seconds per year

# Important conversion factors
# Gravitational radius: r_g = G*M/c^2 = 1.477 km * (M/Msun)
KM_PER_MSUN = 1.477  # km per solar mass

# Dimensionless coupling: alpha = G*M*mu/(hbar*c) = (r_g/c) * (mu*c/hbar)
# In natural units (hbar=c=1): alpha = r_g * mu
# mu in eV -> mu_natural in 1/m: mu * eV_to_J / (hbar*c) = mu * 5.068e6 1/(eV*m)
EV_TO_PER_M = 5.068e6  # eV to 1/meters


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


def compute_alpha(M_msun, mu_eV):
    """
    Compute dimensionless coupling alpha = G*M*mu/(hbar*c).
    
    Parameters:
    -----------
    M_msun : float or array
        BH mass in solar masses
    mu_eV : float or array
        Boson mass in eV
    
    Returns:
    --------
    alpha : float or array
        Dimensionless coupling
    """
    # r_g in meters = 1.477 km * M_msun = 1477 * M_msun meters
    r_g_m = KM_PER_MSUN * 1e3 * M_msun
    
    # mu in 1/meters
    mu_per_m = mu_eV * EV_TO_PER_M
    
    # alpha = r_g * mu (in natural units)
    alpha = r_g_m * mu_per_m
    return alpha


def horizon_angular_velocity_hz(a_star, M_msun):
    """
    Calculate the horizon angular velocity Omega_H in Hz.
    
    Omega_H = (a / (2*r_g)) / (1 + sqrt(1 - a^2))
    
    Parameters:
    -----------
    a_star : float or array
        Dimensionless spin parameter (0 <= a_star < 1)
    M_msun : float or array
        Black hole mass in solar masses
    
    Returns:
    --------
    Omega_H : float or array
        Horizon angular velocity in Hz
    """
    r_g_m = KM_PER_MSUN * 1e3 * M_msun
    Omega_H = (a_star / (2 * r_g_m)) / (1 + np.sqrt(1 - a_star**2))
    return Omega_H


def superradiance_growth_rate(alpha, a_star, l=1, m=1):
    """
    Calculate the superradiance growth rate M*omega_I (dimensionless).
    
    For l=m=1 mode and alpha << 1:
        M*omega_I ≈ (a_star/48) * alpha^9
    
    Parameters:
    -----------
    alpha : float or array
        Dimensionless coupling
    a_star : float or array
        Dimensionless BH spin
    l, m : int
        Quantum numbers
    
    Returns:
    --------
    M_omega_I : float or array
        Growth rate in dimensionless units (M*omega_I)
    """
    # For l=m=1, the growth rate is proportional to alpha^9
    if l == 1 and m == 1:
        M_omega_I = (a_star / 48.0) * alpha**9
    else:
        # Generic scaling for higher l
        M_omega_I = (a_star / 48.0) * alpha**(4*l + 4)
    
    return M_omega_I


def superradiance_timescale_years(alpha, a_star, M_msun, l=1, m=1):
    """
    Calculate the superradiance e-folding timescale in years.
    
    tau = 1/omega_I = M / (M*omega_I) * (G*M_sun/c^3)
    
    Parameters:
    -----------
    alpha : float or array
        Dimensionless coupling
    a_star : float or array
        Dimensionless BH spin
    M_msun : float or array
        BH mass in solar masses
    l, m : int
        Quantum numbers
    
    Returns:
    --------
    tau : float or array
        Growth timescale in years
    """
    # Mass in seconds (geometric units): M_geom = G*M/c^3
    M_geom = G * M_msun * Msun / c**3
    
    # Growth rate
    M_omega_I = superradiance_growth_rate(alpha, a_star, l, m)
    
    # Avoid division by zero
    M_omega_I = np.maximum(M_omega_I, 1e-100)
    
    # Timescale in seconds
    tau_sec = M_geom / M_omega_I
    
    # Convert to years
    tau_years = tau_sec / year_to_sec
    
    return tau_years


def critical_boson_mass(a_star, M_msun, m=1):
    """
    Calculate the critical boson mass for superradiance.
    
    The superradiance condition is approximately:
        mu < m * Omega_H (in appropriate units)
    
    Parameters:
    -----------
    a_star : float or array
        Dimensionless BH spin
    M_msun : float or array
        BH mass in solar masses
    m : int
        Azimuthal quantum number
    
    Returns:
    --------
    mu_crit : float or array
        Critical boson mass in eV
    """
    Omega_H = horizon_angular_velocity_hz(a_star, M_msun)
    
    # Convert Omega_H (Hz) to energy (eV)
    # E = h * f, where h = 4.136e-15 eV*s
    h_eVs = 4.136e-15
    mu_crit = m * Omega_H * h_eVs
    
    return mu_crit


def exclusion_probability(mu_eV, samples, age_years=1e10, l=1, m=1):
    """
    Compute the exclusion probability for a given boson mass.
    
    The exclusion probability is the fraction of posterior samples where
    superradiance would have had time to significantly spin down the BH.
    
    Parameters:
    -----------
    mu_eV : float
        Boson mass in eV
    samples : array (N, 2)
        Posterior samples of [mass, spin]
    age_years : float
        Age of the system in years
    l, m : int
        Quantum numbers
    
    Returns:
    --------
    prob : float
        Exclusion probability (0 to 1)
    """
    M_samples = samples[:, 0]
    a_samples = samples[:, 1]
    
    # Dimensionless coupling
    alpha = compute_alpha(M_samples, mu_eV)
    
    # Check if superradiance condition is satisfied
    mu_crit = critical_boson_mass(a_samples, M_samples, m)
    is_superradiant = mu_eV < mu_crit
    
    # Compute growth timescales
    tau = superradiance_timescale_years(alpha, a_samples, M_samples, l, m)
    
    # Exclusion: superradiance is possible AND timescale is shorter than age
    # We also need alpha to be in the relevant range (not too small, not too large)
    # For l=1, optimal alpha is around 0.42
    excluded = is_superradiant & (tau < age_years) & (alpha > 0.01) & (alpha < 1.0)
    
    prob = np.mean(excluded)
    return prob


def compute_exclusion_curve(samples, mu_range, age_years=1e10, l=1, m=1):
    """
    Compute exclusion probability curve over a range of boson masses.
    """
    probs = []
    for mu in mu_range:
        prob = exclusion_probability(mu, samples, age_years, l, m)
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
        
        # Plot on appropriate scale
        if name == 'M33_X-7':
            axes[0].semilogx(mu_range, probs, label=name, 
                           color=colors[name], linewidth=2)
        else:
            axes[1].semilogx(mu_range, probs, label=name, 
                           color=colors[name], linewidth=2)
        
        # 95% exclusion level
        if np.any(probs >= 0.95):
            mu_95_idx = np.where(probs >= 0.95)[0][0]
            mu_95 = mu_range[mu_95_idx]
            if name == 'M33_X-7':
                axes[0].axvline(mu_95, color=colors[name], linestyle='--', alpha=0.5)
            else:
                axes[1].axvline(mu_95, color=colors[name], linestyle='--', alpha=0.5)
    
    for ax in axes:
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% exclusion')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% exclusion')
        ax.set_ylabel('Exclusion Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('ULB Mass $\\mu$ [eV]')
    
    axes[0].set_title('M33 X-7: Stellar-mass BH Constraints')
    axes[1].set_title('IRAS 09149-6206: Supermassive BH Constraints')
    
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
        
        # Scatter plot of posterior samples
        ax.scatter(M, a, alpha=0.2, s=1, c='navy')
        
        # Plot superradiance boundaries for different mu values
        M_grid = np.linspace(M.min(), M.max(), 100)
        
        for mu in mu_test[name]:
            # For each mu, compute the minimum spin required for superradiance
            # mu = m * Omega_H * h_eVs
            # Omega_H = mu / (m * h_eVs)
            h_eVs = 4.136e-15
            Omega_H_needed = mu / (1 * h_eVs)  # Hz
            
            # Solve for a_star: Omega_H = (a/(2*r_g)) / (1 + sqrt(1-a^2))
            # This is transcendental; use approximation for a ≈ 1
            # For high spin: Omega_H ≈ 1/(4*r_g)
            # For general a: use numerical solution
            
            r_g = KM_PER_MSUN * 1e3 * M_grid  # meters
            
            # Approximate: a_crit such that Omega_H(a_crit) = Omega_H_needed
            # Omega_H = (a/(2*r_g)) / (1 + sqrt(1-a^2))
            # Let x = sqrt(1-a^2), then a = sqrt(1-x^2)
            # Omega_H = sqrt(1-x^2) / (2*r_g*(1+x))
            
            # For high spin (x small): Omega_H ≈ 1/(4*r_g)
            # So if Omega_H_needed < 1/(4*r_g), SR is possible
            
            Omega_H_max = 1 / (4 * r_g)  # Hz (for a -> 1)
            
            # Compute minimum spin for SR
            # Numerically solve: (a/(2*r_g)) / (1 + sqrt(1-a^2)) = Omega_H_needed
            # Let f(a) = (a/(2*r_g)) / (1 + sqrt(1-a^2))
            # We need f(a) = Omega_H_needed
            
            a_crit = np.zeros_like(M_grid)
            for i, rg in enumerate(r_g):
                # Define function to solve
                def f(a):
                    if a >= 0.999:
                        return 1/(4*rg)
                    return (a/(2*rg)) / (1 + np.sqrt(1-a**2))
                
                # Binary search for a such that f(a) = Omega_H_needed
                if Omega_H_needed > 1/(4*rg):
                    a_crit[i] = 1.0  # Always SR for high mu
                elif Omega_H_needed < 0:
                    a_crit[i] = 0.0
                else:
                    # Find minimum a
                    a_low, a_high = 0.0, 0.999
                    for _ in range(50):
                        a_mid = (a_low + a_high) / 2
                        if f(a_mid) < Omega_H_needed:
                            a_low = a_mid
                        else:
                            a_high = a_mid
                    a_crit[i] = a_mid
            
            a_crit = np.clip(a_crit, 0, 0.99)
            
            label = f'$\\mu = {mu:.1e}$ eV'
            ax.plot(M_grid, a_crit, '--', alpha=0.7, linewidth=2, label=label)
        
        ax.set_xlabel('BH Mass [M$_\\odot$]', fontsize=12)
        ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
        ax.set_title(f'{name}: Regge Plane with SR Constraints', fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_upper_limits(samples_dict, age_years=1e10):
    """Compute 95% upper limits on ULB masses."""
    results = {}
    
    for name, samples in samples_dict.items():
        M_median = np.median(samples[:, 0])
        
        # Optimal coupling: alpha ≈ 0.42 for l=m=1 mode
        # alpha = G*M*mu/(hbar*c) = r_g * mu_natural
        # r_g = 1.477 km * M_msun
        # mu_natural [1/m] = mu_eV * 5.068e6
        # alpha = (1477 * M_msun) * (mu_eV * 5.068e6)
        # mu_opt [eV] = 0.42 / (1477 * M_msun * 5.068e6)
        # mu_opt [eV] = 0.42 / (7.49e9 * M_msun)
        
        mu_opt = 0.42 / (KM_PER_MSUN * 1e3 * EV_TO_PER_M * M_median)
        
        # Scan around optimal value (over 3 orders of magnitude)
        mu_range = np.logspace(np.log10(mu_opt * 0.1), np.log10(mu_opt * 10), 300)
        
        probs = compute_exclusion_curve(samples, mu_range, age_years)
        
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
        
        # Find 90% limit
        if np.any(probs >= 0.90):
            mu_90_idx = np.where(probs >= 0.90)[0][0]
            mu_90 = mu_range[mu_90_idx]
        else:
            mu_90 = None
        
        results[name] = {
            'mu_95': mu_95,
            'mu_90': mu_90,
            'mu_50': mu_50,
            'mu_opt': mu_opt,
            'M_median': M_median,
            'a_median': np.median(samples[:, 1]),
            'M_std': np.std(samples[:, 0]),
            'a_std': np.std(samples[:, 1]),
            'probs': probs,
            'mu_range': mu_range
        }
    
    return results


def plot_combined_constraints(samples_dict, results, output_path):
    """Create a comprehensive constraints plot."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    colors = {'M33_X-7': 'steelblue', 'IRAS_09149-6206': 'darkorange'}
    
    # Row 1: Exclusion probability curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, res in results.items():
        ax1.semilogx(res['mu_range'], res['probs'], 
                     label=name, color=colors[name], linewidth=2.5)
        if res['mu_95']:
            ax1.axvline(res['mu_95'], color=colors[name], linestyle='--', alpha=0.6)
            ax1.annotate(f'95%: {res["mu_95"]:.1e} eV', 
                        xy=(res['mu_95'], 0.95), 
                        xytext=(res['mu_95']*2, 0.85),
                        color=colors[name], fontsize=10,
                        arrowprops=dict(arrowstyle='->', color=colors[name], alpha=0.6))
    ax1.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% exclusion')
    ax1.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
    ax1.set_ylabel('Exclusion Probability', fontsize=12)
    ax1.set_title('ULB Mass Exclusion Probabilities', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
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
        ax_m.axvline(np.median(M), color='red', linestyle='--', linewidth=2)
        ax_m.set_xlabel('BH Mass [M$_\\odot$]', fontsize=11)
        ax_m.set_ylabel('Count', fontsize=11)
        ax_m.set_title(f'{name}: Mass Distribution', fontsize=12)
        
        # Spin distribution
        ax_s.hist(a, bins=50, color=colors[name], edgecolor='white', alpha=0.7)
        ax_s.axvline(np.median(a), color='red', linestyle='--', linewidth=2)
        ax_s.set_xlabel('Dimensionless Spin $a_*$', fontsize=11)
        ax_s.set_ylabel('Count', fontsize=11)
        ax_s.set_title(f'{name}: Spin Distribution', fontsize=12)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def save_results_table(results, output_path):
    """Save results to a formatted table."""
    with open(output_path, 'w') as f:
        f.write("# Ultralight Boson Constraint Results\n")
        f.write("#\n")
        f.write("| System | M_BH [Msun] | a_* | mu_opt [eV] | mu_90 [eV] | mu_95 [eV] |\n")
        f.write("|--------|-------------|-----|-------------|------------|------------|\n")
        
        for name, res in results.items():
            mu_95_str = f"{res['mu_95']:.2e}" if res['mu_95'] else "N/A"
            mu_90_str = f"{res['mu_90']:.2e}" if res['mu_90'] else "N/A"
            f.write(f"| {name} | {res['M_median']:.2e} ± {res['M_std']:.1e} | "
                   f"{res['a_median']:.3f} ± {res['a_std']:.3f} | "
                   f"{res['mu_opt']:.2e} | {mu_90_str} | {mu_95_str} |\n")
    print(f"Saved: {output_path}")


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("Ultralight Boson Constraint Analysis v2")
    print("Using Black Hole Superradiance")
    print("=" * 70)
    
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
    print("\n" + "-" * 50)
    print("Summary Statistics:")
    print("-" * 50)
    for name, samples in samples_dict.items():
        M = samples[:, 0]
        a = samples[:, 1]
        print(f"\n{name}:")
        print(f"  Mass: {np.median(M):.3e} ± {np.std(M):.3e} M_sun")
        print(f"  Spin: {np.median(a):.3f} ± {np.std(a):.3f}")
        
        # Expected ULB mass range
        mu_opt = 0.42 / (KM_PER_MSUN * 1e3 * EV_TO_PER_M * np.median(M))
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
    results = compute_upper_limits(samples_dict)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Median BH mass: {res['M_median']:.3e} ± {res['M_std']:.1e} M_sun")
        print(f"  Median BH spin: {res['a_median']:.3f} ± {res['a_std']:.3f}")
        print(f"  Optimal ULB mass (alpha=0.42): {res['mu_opt']:.2e} eV")
        if res['mu_90']:
            print(f"  90% upper limit: {res['mu_90']:.2e} eV")
        if res['mu_95']:
            print(f"  95% upper limit: {res['mu_95']:.2e} eV")
    
    # Generate exclusion curves
    print("\n" + "-" * 50)
    print("Generating exclusion curves...")
    print("-" * 50)
    plot_exclusion_curves(results, '../report/images/exclusion_curves.png')
    
    # Generate Regge plane plots
    print("\n" + "-" * 50)
    print("Generating Regge plane constraints...")
    print("-" * 50)
    mu_test = {
        'M33_X-7': [5e-12, 1e-11, 2e-11],
        'IRAS_09149-6206': [1e-20, 3e-20, 1e-19]
    }
    plot_regge_constraints(samples_dict, mu_test,
                          '../report/images/regge_constraints.png')
    
    # Generate combined constraints plot
    print("\n" + "-" * 50)
    print("Generating combined constraints plot...")
    print("-" * 50)
    plot_combined_constraints(samples_dict, results,
                             '../report/images/combined_constraints.png')
    
    # Save results table
    save_results_table(results, '../outputs/ulb_limits_table.md')
    
    # Save numerical results as JSON
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            'M_median_Msun': float(res['M_median']),
            'M_std_Msun': float(res['M_std']),
            'a_median': float(res['a_median']),
            'a_std': float(res['a_std']),
            'mu_opt_eV': float(res['mu_opt']),
            'mu_90_eV': float(res['mu_90']) if res['mu_90'] else None,
            'mu_95_eV': float(res['mu_95']) if res['mu_95'] else None
        }
    
    with open('../outputs/ulb_constraints.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: ../outputs/ulb_constraints.json")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
