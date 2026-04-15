"""
Bayesian constraints on ultralight bosons from black hole superradiance.

This module implements a statistical framework that ingests full posterior
distributions of black hole mass and spin measurements to derive upper limits
on ultralight boson (ULB) masses via the superradiance mechanism.

Physics references:
- Arvanitaki & Dubovsky (2011), Phys. Rev. D 83, 044026
- Arvanitaki et al. (2017), Phys. Rev. Lett. 116, 061102
- Stott & Marsh (2018), Phys. Rev. D 98, 083006
"""

import numpy as np
import os
import json

# =============================================================================
# Physical constants (in natural units where G = c = hbar = 1, but we track conversions)
# =============================================================================

# Conversion factor: alpha = G_N * M_BH * mu_a / (hbar * c)
# In practical units: alpha = (M_BH / M_sun) * (mu_a / eV) * kappa
# From literature: alpha ≈ 0.22 * (M_BH / 30 M_sun) * (mu_a / 1e-12 eV)
# Therefore: kappa = 0.22 / (30 * 1e-12) = 7.333...e9
KAPPA = 0.22 / (30.0 * 1e-12)  # = 7.333e9

# Gravitational radius conversion: r_g = G_N * M_BH / c^2
# For M_sun: r_g_sun = 1.477 km = 1.477e3 m
# In natural units (hbar=c=1): r_g = M (in length units)
# Time unit: t_g = r_g / c = G_N * M_BH / c^3
# For M_sun: t_g_sun = 4.926e-6 s
T_G_SUN = 4.925490947e-6  # seconds per solar mass in geometric time units

# Age of universe ~ 13.8 Gyr
AGE_UNIVERSE_YR = 13.8e9
SEC_PER_YEAR = 3.15576e7
AGE_UNIVERSE_SEC = AGE_UNIVERSE_YR * SEC_PER_YEAR


def compute_alpha(M_bh, mu_a):
    """
    Compute the gravitational fine-structure constant alpha.
    
    Parameters
    ----------
    M_bh : float or array
        Black hole mass in solar masses.
    mu_a : float or array
        Ultralight boson mass in eV.
    
    Returns
    -------
    alpha : float or array
        Dimensionless gravitational coupling.
    """
    return KAPPA * M_bh * mu_a


def horizon_angular_velocity(a_star):
    """
    Compute the dimensionless horizon angular velocity Omega_H * r_g.
    
    Omega_H = a_* / [2 * r_g * (1 + sqrt(1 - a_*^2))]
    
    We return Omega_H * r_g = a_* / [2 * (1 + sqrt(1 - a_*^2))]
    
    Parameters
    ----------
    a_star : float or array
        Dimensionless spin parameter (0 <= a_* < 1).
    
    Returns
    -------
    omega_H_rg : float or array
        Horizon angular velocity times gravitational radius (dimensionless).
    """
    a_star = np.clip(a_star, 0, 0.9999)  # avoid division issues at extremality
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))


def bound_state_energy(n, l, alpha):
    """
    Compute the bound state energy for a scalar field around a Kerr BH.
    
    For small alpha (hydrogenic approximation):
    E_nl ≈ mu_a * (1 - alpha^2 / (2 * n^2))
    
    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital quantum number.
    alpha : float or array
        Gravitational fine-structure constant.
    
    Returns
    -------
    E_over_mu : float or array
        Energy divided by mu_a (dimensionless).
    """
    return 1.0 - alpha**2 / (2.0 * n**2)


def superradiance_condition(alpha, a_star, n=2, l=1, m=1):
    """
    Check the superradiance condition: omega / m < Omega_H
    
    For the dominant l=m=1 mode with n=2:
    omega ≈ mu_a * (1 - alpha^2 / 8)
    
    The condition becomes:
    alpha * (1 - alpha^2 / 8) < a_* / [2 * (1 + sqrt(1 - a_*^2))]
    
    Parameters
    ----------
    alpha : float or array
        Gravitational fine-structure constant.
    a_star : float or array
        Dimensionless spin parameter.
    n, l, m : int
        Quantum numbers (default: n=2, l=1, m=1 for fastest growing mode).
    
    Returns
    -------
    satisfied : bool or array
        True if superradiance condition is met.
    """
    omega_over_mu = bound_state_energy(n, l, alpha)
    omega_H_rg = horizon_angular_velocity(a_star)
    return (alpha * omega_over_mu / m) < omega_H_rg


def instability_timescale(alpha, a_star, M_bh, l=1, m=1):
    """
    Estimate the superradiant instability timescale for the dominant mode.
    
    For small alpha and l=m=1:
    M * omega_I ≈ (a_* / 48) * alpha^9  (from Witek et al. 2013)
    
    tau_SR = 1 / omega_I = M / (M * omega_I) = M / [(a_*/48) * alpha^9]
    
    In physical units: tau_SR = t_g * 48 / (a_* * alpha^9)
    where t_g = G_N * M_BH / c^3
    
    Parameters
    ----------
    alpha : float or array
        Gravitational fine-structure constant.
    a_star : float or array
        Dimensionless spin parameter.
    M_bh : float or array
        Black hole mass in solar masses.
    l, m : int
        Mode quantum numbers.
    
    Returns
    -------
    tau_sec : float or array
        Instability timescale in seconds.
    """
    a_star = np.clip(a_star, 1e-6, 1.0)  # avoid division by zero
    alpha = np.clip(alpha, 1e-20, None)  # avoid division by zero
    
    # Geometric time unit for this BH
    t_g = T_G_SUN * M_bh  # seconds
    
    # Growth rate: M * omega_I ≈ (a_*/48) * alpha^(4l+5) for l=m=1
    # Using the formula from Arvanitaki & Dubovsky: M*omega_I ~ (a_*/48)*alpha^9
    M_omega_I = (a_star / 48.0) * alpha**(4*l + 5)  # general: 4l+5 for scalar
    
    # Timescale in geometric units
    tau_geom = 1.0 / M_omega_I
    
    # Convert to seconds
    tau_sec = t_g * tau_geom
    
    return tau_sec


def exclusion_probability_single_sample(alpha_i, a_star_i, M_bh_i, age_sec=None):
    """
    Determine if a single (M, a_*) sample point excludes a given ULB mass.
    
    A ULB mass is excluded if:
    1. The superradiance condition is satisfied
    2. The instability timescale is shorter than the BH age
    
    Parameters
    ----------
    alpha_i : float
        Gravitational coupling for this sample.
    a_star_i : float
        Spin parameter for this sample.
    M_bh_i : float
        BH mass for this sample (solar masses).
    age_sec : float, optional
        BH age in seconds. Default: age of universe.
    
    Returns
    -------
    excluded : bool
        True if this sample point excludes the ULB mass.
    """
    if age_sec is None:
        age_sec = AGE_UNIVERSE_SEC
    
    # Check superradiance condition
    if not superradiance_condition(alpha_i, a_star_i):
        return False
    
    # Check timescale
    tau = instability_timescale(alpha_i, a_star_i, M_bh_i)
    if tau > age_sec:
        return False
    
    return True


def compute_exclusion_probability(samples_M, samples_a, mu_a_values, age_sec=None):
    """
    Compute the exclusion probability P_excl(mu_a) for a grid of ULB masses.
    
    This marginalizes over the full posterior distribution of (M, a_*).
    
    Parameters
    ----------
    samples_M : array
        Posterior samples of BH mass (solar masses).
    samples_a : array
        Posterior samples of dimensionless spin.
    mu_a_values : array
        Grid of ULB masses to evaluate (eV).
    age_sec : float, optional
        BH age in seconds.
    
    Returns
    -------
    P_excl : array
        Exclusion probability for each mu_a value.
    """
    n_samples = len(samples_M)
    P_excl = np.zeros(len(mu_a_values))
    
    for i, mu_a in enumerate(mu_a_values):
        alpha_vals = compute_alpha(samples_M, mu_a)
        
        # Vectorized exclusion check
        sr_cond = superradiance_condition(alpha_vals, samples_a)
        tau_vals = instability_timescale(alpha_vals, samples_a, samples_M)
        
        if age_sec is None:
            age_sec = AGE_UNIVERSE_SEC
        
        excluded = sr_cond & (tau_vals < age_sec)
        P_excl[i] = np.mean(excluded)
    
    return P_excl


def compute_bayesian_upper_limit(mu_a_values, P_excl, confidence=0.95):
    """
    Compute the Bayesian upper limit on ULB mass.
    
    The upper limit is the mass above which the exclusion probability
    drops below (1 - confidence).
    
    Parameters
    ----------
    mu_a_values : array
        Grid of ULB masses (eV).
    P_excl : array
        Exclusion probability at each mass.
    confidence : float
        Confidence level (default: 0.95).
    
    Returns
    -------
    mu_limit : float
        Upper limit on ULB mass (eV).
    """
    # Find where P_excl drops below (1 - confidence)
    threshold = 1.0 - confidence
    
    # Find the transition point
    idx = np.where(P_excl < threshold)[0]
    if len(idx) == 0:
        # All masses are excluded up to the maximum
        return mu_a_values[-1]
    
    first_below = idx[0]
    if first_below == 0:
        return mu_a_values[0]
    
    # Interpolate
    mu_low = mu_a_values[first_below - 1]
    mu_high = mu_a_values[first_below]
    p_low = P_excl[first_below - 1]
    p_high = P_excl[first_below]
    
    if p_low == p_high:
        return mu_low
    
    frac = (threshold - p_low) / (p_high - p_low)
    mu_limit = mu_low + frac * (mu_high - mu_low)
    
    return mu_limit


def load_posterior(filepath):
    """
    Load posterior samples from a .dat file.
    
    Parameters
    ----------
    filepath : str
        Path to the data file.
    
    Returns
    -------
    samples_M : array
        BH mass samples (solar masses).
    samples_a : array
        Dimensionless spin samples.
    """
    data = np.loadtxt(filepath, comments='#')
    samples_M = data[:, 0]
    samples_a = data[:, 1]
    return samples_M, samples_a


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Bayesian ULB Constraints from BH Superradiance")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("report/images", exist_ok=True)
    
    # ---- Load data ----
    print("\nLoading posterior samples...")
    
    # M33 X-7 (stellar-mass BH)
    M_m33, a_m33 = load_posterior("data/M33_X-7_samples.dat")
    print(f"  M33 X-7: {len(M_m33)} samples")
    print(f"    Mass:  {np.mean(M_m33):.1f} ± {np.std(M_m33):.1f} M_sun")
    print(f"    Spin:  {np.mean(a_m33):.3f} ± {np.std(a_m33):.3f}")
    
    # IRAS 09149-6206 (SMBH)
    M_iras, a_iras = load_posterior("data/IRAS_09149-6206_samples.dat")
    print(f"  IRAS 09149-6206: {len(M_iras)} samples")
    print(f"    Mass:  {np.mean(M_iras):.2e} ± {np.std(M_iras):.2e} M_sun")
    print(f"    Spin:  {np.mean(a_iras):.4f} ± {np.std(a_iras):.4f}")
    
    # ---- Define ULB mass grids ----
    # Stellar-mass BH probes: ~10^-13 to ~10^-10 eV
    mu_stellar = np.logspace(-13.5, -10.0, 500)
    
    # SMBH probes: ~10^-20 to ~10^-17 eV
    mu_smbh = np.logspace(-20.5, -16.5, 500)
    
    # ---- Compute exclusion probabilities ----
    print("\nComputing exclusion probabilities...")
    
    # M33 X-7
    print("  M33 X-7 (stellar-mass)...")
    P_excl_m33 = compute_exclusion_probability(M_m33, a_m33, mu_stellar)
    
    # IRAS 09149-6206
    print("  IRAS 09149-6206 (SMBH)...")
    P_excl_iras = compute_exclusion_probability(M_iras, a_iras, mu_smbh)
    
    # ---- Compute upper limits ----
    print("\nComputing 95% credible upper limits...")
    
    limit_m33 = compute_bayesian_upper_limit(mu_stellar, P_excl_m33, 0.95)
    limit_iras = compute_bayesian_upper_limit(mu_smbh, P_excl_iras, 0.95)
    
    print(f"  M33 X-7:     mu_a < {limit_m33:.3e} eV  (95% CL)")
    print(f"  IRAS 09149-6206: mu_a < {limit_iras:.3e} eV  (95% CL)")
    
    # ---- Save results ----
    results = {
        "M33_X-7": {
            "mu_a_grid_eV": mu_stellar.tolist(),
            "P_exclusion": P_excl_m33.tolist(),
            "upper_limit_95_eV": float(limit_m33),
            "n_samples": len(M_m33),
            "mass_mean_Msun": float(np.mean(M_m33)),
            "mass_std_Msun": float(np.std(M_m33)),
            "spin_mean": float(np.mean(a_m33)),
            "spin_std": float(np.std(a_m33)),
        },
        "IRAS_09149-6206": {
            "mu_a_grid_eV": mu_smbh.tolist(),
            "P_exclusion": P_excl_iras.tolist(),
            "upper_limit_95_eV": float(limit_iras),
            "n_samples": len(M_iras),
            "mass_mean_Msun": float(np.mean(M_iras)),
            "mass_std_Msun": float(np.std(M_iras)),
            "spin_mean": float(np.mean(a_iras)),
            "spin_std": float(np.std(a_iras)),
        }
    }
    
    with open("outputs/bayesian_constraints.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/bayesian_constraints.json")
    
    # Save numpy arrays for plotting
    np.savez("outputs/exclusion_curves.npz",
             mu_stellar=mu_stellar, P_excl_m33=P_excl_m33,
             mu_smbh=mu_smbh, P_excl_iras=P_excl_iras,
             limit_m33=limit_m33, limit_iras=limit_iras)
    
    # ---- Generate figures ----
    print("\nGenerating figures...")
    generate_figures(M_m33, a_m33, M_iras, a_iras,
                     mu_stellar, P_excl_m33, limit_m33,
                     mu_smbh, P_excl_iras, limit_iras)
    
    print("\nAnalysis complete.")
    return results


def generate_figures(M_m33, a_m33, M_iras, a_iras,
                     mu_stellar, P_excl_m33, limit_m33,
                     mu_smbh, P_excl_iras, limit_iras):
    """Generate all publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 150,
    })
    
    # =====================================================================
    # Figure 1: Data overview - posterior distributions
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # M33 X-7 mass distribution
    ax = axes[0, 0]
    ax.hist(M_m33, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(M_m33), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(M_m33):.1f}')
    ax.set_xlabel('BH Mass [$M_\\odot$]')
    ax.set_ylabel('Probability Density')
    ax.set_title('M33 X-7: Mass Posterior')
    ax.legend(fontsize=9)
    
    # M33 X-7 spin distribution
    ax = axes[0, 1]
    ax.hist(a_m33, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(np.mean(a_m33), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(a_m33):.3f}')
    ax.set_xlabel('Dimensionless Spin $a_*$')
    ax.set_ylabel('Probability Density')
    ax.set_title('M33 X-7: Spin Posterior')
    ax.legend(fontsize=9)
    
    # IRAS 09149-6206 mass distribution
    ax = axes[1, 0]
    ax.hist(M_iras, bins=80, density=True, alpha=0.7, color='forestgreen', edgecolor='white')
    ax.axvline(np.mean(M_iras), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(M_iras)/1e8:.2f}$\\times 10^8$')
    ax.set_xlabel('BH Mass [$M_\\odot$]')
    ax.set_ylabel('Probability Density')
    ax.set_title('IRAS 09149-6206: Mass Posterior')
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    # IRAS 09149-6206 spin distribution
    ax = axes[1, 1]
    ax.hist(a_iras, bins=50, density=True, alpha=0.7, color='purple', edgecolor='white')
    ax.axvline(np.mean(a_iras), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(a_iras):.4f}')
    ax.set_xlabel('Dimensionless Spin $a_*$')
    ax.set_ylabel('Probability Density')
    ax.set_title('IRAS 09149-6206: Spin Posterior')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report/images/figure1_data_overview.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure1_data_overview.png")
    
    # =====================================================================
    # Figure 2: Mass-Spin Regge plane with posterior samples
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot posterior samples
    ax.scatter(M_m33, a_m33, s=2, alpha=0.3, color='steelblue', label='M33 X-7')
    ax.scatter(M_iras, a_iras, s=2, alpha=0.3, color='forestgreen', label='IRAS 09149-6206')
    
    # Mark means
    ax.scatter(np.mean(M_m33), np.mean(a_m33), s=100, marker='*', color='red',
               zorder=5, label='M33 X-7 mean')
    ax.scatter(np.mean(M_iras), np.mean(a_iras), s=100, marker='*', color='orange',
               zorder=5, label='IRAS mean')
    
    ax.set_xlabel('BH Mass [$M_\\odot$]')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_title('Black Hole Mass-Spin Distribution')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure2_regge_plane.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure2_regge_plane.png")
    
    # =====================================================================
    # Figure 3: Exclusion probability curves (main result)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # M33 X-7 exclusion curve
    ax = axes[0]
    ax.semilogx(mu_stellar, P_excl_m33, 'b-', linewidth=2.5, label='M33 X-7')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='95% CL threshold')
    ax.axvline(limit_m33, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.fill_between(mu_stellar, 0, P_excl_m33, alpha=0.15, color='blue')
    ax.set_xlabel(r'ULB Mass $\mu_a$ [eV]')
    ax.set_ylabel('Exclusion Probability $P_{\\rm excl}$')
    ax.set_title('M33 X-7: ULB Mass Constraint')
    ax.set_xlim(mu_stellar[0], mu_stellar[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'$\\mu_a < {limit_m33:.2e}$ eV', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # IRAS 09149-6206 exclusion curve
    ax = axes[1]
    ax.semilogx(mu_smbh, P_excl_iras, 'g-', linewidth=2.5, label='IRAS 09149-6206')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='95% CL threshold')
    ax.axvline(limit_iras, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.fill_between(mu_smbh, 0, P_excl_iras, alpha=0.15, color='green')
    ax.set_xlabel(r'ULB Mass $\mu_a$ [eV]')
    ax.set_ylabel('Exclusion Probability $P_{\\rm excl}$')
    ax.set_title('IRAS 09149-6206: ULB Mass Constraint')
    ax.set_xlim(mu_smbh[0], mu_smbh[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'$\\mu_a < {limit_iras:.2e}$ eV', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('report/images/figure3_exclusion_curves.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure3_exclusion_curves.png")
    
    # =====================================================================
    # Figure 4: Combined constraint summary
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both exclusion curves on same axis (normalized to their ranges)
    # Use two x-axes or plot as function of alpha
    mu_combined = np.logspace(-21, -9, 1000)
    
    # Compute combined exclusion
    P_excl_combined = np.zeros(len(mu_combined))
    
    # For stellar-mass range
    mask_stellar = (mu_combined >= 1e-14) & (mu_combined <= 1e-10)
    for i, mu in enumerate(mu_combined):
        if mask_stellar[i]:
            idx = np.argmin(np.abs(mu_stellar - mu))
            P_excl_combined[i] = P_excl_m33[idx]
        elif (mu >= 1e-21) & (mu <= 1e-16):
            idx = np.argmin(np.abs(mu_smbh - mu))
            P_excl_combined[i] = P_excl_iras[idx]
    
    # Plot exclusion regions
    ax.axhspan(0.95, 1.0, alpha=0.1, color='red', label='Excluded region')
    
    # M33 X-7 curve
    ax.semilogx(mu_stellar, P_excl_m33, 'b-', linewidth=2, label='M33 X-7 (stellar-mass)', alpha=0.8)
    
    # IRAS curve
    ax.semilogx(mu_smbh, P_excl_iras, 'g-', linewidth=2, label='IRAS 09149-6206 (SMBH)', alpha=0.8)
    
    # Threshold line
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Upper limit markers
    ax.axvline(limit_m33, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(limit_iras, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel(r'ULB Mass $\mu_a$ [eV]', fontsize=12)
    ax.set_ylabel('Exclusion Probability $P_{\\rm excl}$', fontsize=12)
    ax.set_title('Combined ULB Mass Constraints from Black Hole Superradiance', fontsize=13)
    ax.set_xlim(1e-21, 1e-9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(0.98, 0.05, 
            f'Stellar-mass limit: $\\mu_a < {limit_m33:.2e}$ eV\n'
            f'SMBH limit: $\\mu_a < {limit_iras:.2e}$ eV',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('report/images/figure4_combined_constraints.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure4_combined_constraints.png")
    
    # =====================================================================
    # Figure 5: Alpha vs Spin - Superradiance regime visualization
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # For M33 X-7
    ax = axes[0]
    # Compute alpha for each sample at the median ULB mass probed
    mu_median = np.median(mu_stellar)
    alpha_m33 = compute_alpha(M_m33, mu_median)
    
    # Plot exclusion boundary
    a_star_grid = np.linspace(0, 0.999, 200)
    alpha_boundary = np.array([horizon_angular_velocity(a) for a in a_star_grid])
    ax.plot(alpha_boundary, a_star_grid, 'r-', linewidth=2, label='SR boundary')
    ax.fill_betweenx(a_star_grid, alpha_boundary, 10, alpha=0.15, color='red',
                     label='Excluded (SR active)')
    
    # Plot samples
    ax.scatter(alpha_m33, a_m33, s=3, alpha=0.4, color='steelblue', label='Posterior samples')
    
    ax.set_xlabel(r'Gravitational Coupling $\alpha$')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_title(f'M33 X-7: SR Regime ($\\mu_a = {mu_median:.1e}$ eV)')
    ax.set_xlim(0, max(alpha_m33)*1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # For IRAS 09149-6206
    ax = axes[1]
    mu_median_iras = np.median(mu_smbh)
    alpha_iras = compute_alpha(M_iras, mu_median_iras)
    
    ax.plot(alpha_boundary, a_star_grid, 'r-', linewidth=2, label='SR boundary')
    ax.fill_betweenx(a_star_grid, alpha_boundary, 10, alpha=0.15, color='red',
                     label='Excluded (SR active)')
    
    ax.scatter(alpha_iras, a_iras, s=3, alpha=0.4, color='forestgreen', label='Posterior samples')
    
    ax.set_xlabel(r'Gravitational Coupling $\alpha$')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_title(f'IRAS 09149-6206: SR Regime ($\\mu_a = {mu_median_iras:.1e}$ eV)')
    ax.set_xlim(0, max(alpha_iras)*1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure5_sr_regime.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure5_sr_regime.png")
    
    # =====================================================================
    # Figure 6: Validation - convergence check
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subsample convergence for M33 X-7
    n_subsamples = [50, 100, 200, 500, 1000, min(len(M_m33), 1840)]
    colors_conv = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_subsamples)))
    
    ax = axes[0]
    for j, n_sub in enumerate(n_subsamples):
        idx = np.random.RandomState(42).choice(len(M_m33), size=min(n_sub, len(M_m33)), replace=False)
        P_sub = compute_exclusion_probability(M_m33[idx], a_m33[idx], mu_stellar)
        ax.semilogx(mu_stellar, P_sub, color=colors_conv[j], linewidth=1.5,
                    alpha=0.8, label=f'n={n_sub}')
    
    # Full sample
    ax.semilogx(mu_stellar, P_excl_m33, 'k-', linewidth=2.5, label=f'Full (n={len(M_m33)})')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'ULB Mass $\mu_a$ [eV]')
    ax.set_ylabel('Exclusion Probability')
    ax.set_title('Convergence Check: M33 X-7')
    ax.set_xlim(mu_stellar[0], mu_stellar[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Bootstrap uncertainty bands
    ax = axes[1]
    n_bootstrap = 50
    P_boot = np.zeros((n_bootstrap, len(mu_stellar)))
    rng = np.random.RandomState(123)
    for b in range(n_bootstrap):
        idx = rng.choice(len(M_m33), size=len(M_m33), replace=True)
        P_boot[b] = compute_exclusion_probability(M_m33[idx], a_m33[idx], mu_stellar)
    
    P_mean = np.mean(P_boot, axis=0)
    P_std = np.std(P_boot, axis=0)
    
    ax.semilogx(mu_stellar, P_mean, 'b-', linewidth=2, label='Bootstrap mean')
    ax.fill_between(mu_stellar, P_mean - P_std, P_mean + P_std,
                    alpha=0.3, color='blue', label=r'$\pm 1\sigma$')
    ax.semilogx(mu_stellar, P_excl_m33, 'k--', linewidth=1.5, label='Full posterior')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'ULB Mass $\mu_a$ [eV]')
    ax.set_ylabel('Exclusion Probability')
    ax.set_title('Bootstrap Uncertainty: M33 X-7')
    ax.set_xlim(mu_stellar[0], mu_stellar[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure6_validation.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure6_validation.png")
    
    # =====================================================================
    # Figure 7: Instability timescale map
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create 2D grid of (M, a_*) and compute tau_SR for a fixed mu_a
    mu_fixed = 1e-12  # eV, representative for stellar-mass BHs
    
    M_grid = np.logspace(1, 2.5, 100)  # 10 to ~316 M_sun
    a_grid = np.linspace(0.01, 0.999, 100)
    M_mesh, a_mesh = np.meshgrid(M_grid, a_grid)
    alpha_mesh = compute_alpha(M_mesh, mu_fixed)
    
    # Compute timescale
    tau_mesh = instability_timescale(alpha_mesh, a_mesh, M_mesh)
    
    # Convert to years for display
    tau_yr = tau_mesh / SEC_PER_YEAR
    
    # Log scale
    log_tau = np.log10(tau_yr)
    log_tau = np.clip(log_tau, -5, 12)  # clip for visualization
    
    # Mask where SR condition not satisfied
    sr_mask = superradiance_condition(alpha_mesh, a_mesh)
    log_tau[~sr_mask] = np.nan
    
    im = ax.pcolormesh(M_grid, a_grid, log_tau, cmap='viridis', shading='auto',
                       vmin=0, vmax=10)
    cbar = plt.colorbar(im, ax=ax, label=r'$\log_{10}(\tau_{\rm SR} / {\rm yr})$')
    
    # Overlay M33 X-7 posterior
    ax.scatter(M_m33, a_m33, s=5, alpha=0.5, color='red', label='M33 X-7 samples')
    ax.scatter(np.mean(M_m33), np.mean(a_m33), s=100, marker='*', color='white',
               edgecolors='red', linewidth=2, zorder=5, label='M33 X-7 mean')
    
    ax.axhline(0.95, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('BH Mass [$M_\\odot$]')
    ax.set_ylabel('Dimensionless Spin $a_*$')
    ax.set_xscale('log')
    ax.set_title(f'Superradiance Timescale Map ($\\mu_a = {mu_fixed:.0e}$ eV)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure7_timescale_map.png', bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved figure7_timescale_map.png")
    
    # =====================================================================
    # Save summary statistics
    # =====================================================================
    summary = {
        "M33_X-7": {
            "mass_mean_Msun": float(np.mean(M_m33)),
            "mass_std_Msun": float(np.std(M_m33)),
            "spin_mean": float(np.mean(a_m33)),
            "spin_std": float(np.std(a_m33)),
            "upper_limit_95_eV": float(limit_m33),
            "n_samples": len(M_m33),
        },
        "IRAS_09149-6206": {
            "mass_mean_Msun": float(np.mean(M_iras)),
            "mass_std_Msun": float(np.std(M_iras)),
            "spin_mean": float(np.mean(a_iras)),
            "spin_std": float(np.std(a_iras)),
            "upper_limit_95_eV": float(limit_iras),
            "n_samples": len(M_iras),
        }
    }
    
    with open("outputs/summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved summary_statistics.json")


if __name__ == "__main__":
    main()
