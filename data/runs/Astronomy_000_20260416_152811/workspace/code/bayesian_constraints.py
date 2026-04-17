"""
Bayesian framework for deriving ULB constraints from black hole superradiance.

This module implements the full Bayesian statistical framework that:
1. Ingests posterior samples of BH mass and spin
2. Computes exclusion probabilities across axion mass parameter space
3. Derives credible upper limits on ULB masses
4. Optionally includes self-interaction coupling constraints
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from superradiance_physics import (
    compute_alpha, 
    horizon_angular_velocity, 
    superradiance_condition,
    instability_growth_rate,
    exclusion_probability
)

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_000_20260416_152811"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")


def load_samples():
    """Load saved posterior samples."""
    iras_samples = np.load(os.path.join(OUTPUTS_DIR, "iras_samples.npy"))
    m33_samples = np.load(os.path.join(OUTPUTS_DIR, "m33_samples.npy"))
    return iras_samples, m33_samples


def scan_exclusion_probability(samples_mass, samples_spin, mu_range):
    """
    Scan exclusion probability over a range of axion masses.
    
    Parameters:
    -----------
    samples_mass : array
        BH mass posterior samples
    samples_spin : array
        BH spin posterior samples
    mu_range : array
        Array of axion masses to test [eV]
    
    Returns:
    --------
    p_excluded : array
        Exclusion probability for each mu
    all_details : list
        List of detail dictionaries for each mu
    """
    p_excluded = []
    all_details = []
    
    for mu in mu_range:
        p, details = exclusion_probability(samples_mass, samples_spin, mu)
        p_excluded.append(p)
        all_details.append(details)
    
    return np.array(p_excluded), all_details


def derive_upper_limit(mu_range, p_excluded, confidence=0.95):
    """
    Derive upper limit on axion mass from exclusion probability curve.
    
    The upper limit is the mass at which P(excluded) drops below (1 - confidence).
    For example, for 95% CL, find where P(excluded) < 0.05.
    
    Parameters:
    -----------
    mu_range : array
        Axion masses tested
    p_excluded : array
        Exclusion probabilities
    confidence : float
        Confidence level (default 0.95 for 95% CL)
    
    Returns:
    --------
    upper_limit : float
        Upper limit on axion mass [eV]
    """
    # Find where p_excluded drops below threshold
    threshold = 1 - confidence
    
    # Look for crossing point
    above_threshold = p_excluded >= threshold
    
    if not np.any(above_threshold):
        return None  # No constraint
    
    # Find the highest mass still excluded
    excluded_indices = np.where(above_threshold)[0]
    max_idx = excluded_indices[-1]
    
    # Interpolate for more precise limit
    if max_idx < len(mu_range) - 1:
        # Linear interpolation
        mu1, mu2 = mu_range[max_idx], mu_range[max_idx + 1]
        p1, p2 = p_excluded[max_idx], p_excluded[max_idx + 1]
        if p1 != p2:
            upper_limit = mu1 + (threshold - p1) * (mu2 - mu1) / (p2 - p1)
        else:
            upper_limit = mu1
    else:
        upper_limit = mu_range[max_idx]
    
    return upper_limit


def create_exclusion_plot(mu_range, p_excluded_iras, p_excluded_m33, 
                          upper_limit_iras, upper_limit_m33):
    """Create exclusion probability plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.semilogx(mu_range, p_excluded_iras, 'b-', linewidth=2, 
                label='IRAS 09149-6206 (SMBH)')
    ax.semilogx(mu_range, p_excluded_m33, 'r-', linewidth=2, 
                label='M33 X-7 (Stellar-mass)')
    
    # Mark upper limits
    if upper_limit_iras is not None:
        ax.axvline(x=upper_limit_iras, color='b', linestyle='--', alpha=0.7,
                   label=f'IRAS 95% CL: {upper_limit_iras:.2e} eV')
    if upper_limit_m33 is not None:
        ax.axvline(x=upper_limit_m33, color='r', linestyle='--', alpha=0.7,
                   label=f'M33 X-7 95% CL: {upper_limit_m33:.2e} eV')
    
    # Reference lines
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Axion Mass μ [eV]', fontsize=12)
    ax.set_ylabel('Exclusion Probability P(excluded)', fontsize=12)
    ax.set_title('ULB Exclusion Probability from Black Hole Superradiance', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "exclusion_probability.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved exclusion plot to {os.path.join(IMAGES_DIR, 'exclusion_probability.png')}")


def create_mass_spin_scatter(iras_samples, m33_samples):
    """Create mass-spin scatter plot with α contours."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot samples
    ax.scatter(iras_samples[:, 0], iras_samples[:, 1], 
               c='blue', alpha=0.1, s=10, label='IRAS 09149-6206', zorder=1)
    ax.scatter(m33_samples[:, 0], m33_samples[:, 1], 
               c='red', alpha=0.3, s=20, label='M33 X-7', zorder=2)
    
    # Add α = 0.4 contours for different axion masses
    mu_values = [1e-18, 1e-15, 1e-12, 1e-10]  # eV
    M_values = []
    for mu in mu_values:
        # α = 0.4 => M = 0.4 / (0.22 * (mu / 1e-12) / 30)
        M_for_alpha_04 = 0.4 * 30.0 / (0.22 * (mu / 1e-12))
        M_values.append(M_for_alpha_04)
        ax.axvline(x=M_for_alpha_04, color='green', linestyle='--', alpha=0.5)
    
    # Label the contours
    for mu, M in zip(mu_values, M_values):
        ax.text(M, 0.98, f'μ={mu:.0e} eV\n(α=0.4)', rotation=90, 
                va='top', ha='right', fontsize=8, color='green')
    
    ax.set_xscale('log')
    ax.set_xlabel('Black Hole Mass [Msol]', fontsize=12)
    ax.set_ylabel('Dimensionless Spin a*', fontsize=12)
    ax.set_title('BH Mass-Spin Distribution with Optimal Sensitivity Lines', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "mass_spin_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mass-spin plot to {os.path.join(IMAGES_DIR, 'mass_spin_distribution.png')}")


def create_alpha_distribution_plot(iras_samples, m33_samples):
    """Create distribution of α values for representative axion masses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Representative axion masses for each BH type
    mu_iras = 5e-19  # Optimal for SMBH
    mu_m33 = 3e-12   # Optimal for stellar-mass
    
    # IRAS
    ax = axes[0]
    alphas_iras = compute_alpha(iras_samples[:, 0], mu_iras)
    ax.hist(np.log10(alphas_iras), bins=50, color='blue', alpha=0.7, density=True)
    ax.axvline(x=np.log10(0.4), color='red', linestyle='--', linewidth=2, label='Optimal α=0.4')
    ax.axvline(x=np.log10(0.01), color='orange', linestyle=':', linewidth=2, label='α=0.01 threshold')
    ax.set_xlabel('log10(α)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'IRAS 09149-6206 α Distribution (μ = {mu_iras:.1e} eV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # M33 X-7
    ax = axes[1]
    alphas_m33 = compute_alpha(m33_samples[:, 0], mu_m33)
    ax.hist(np.log10(alphas_m33), bins=50, color='red', alpha=0.7, density=True)
    ax.axvline(x=np.log10(0.4), color='red', linestyle='--', linewidth=2, label='Optimal α=0.4')
    ax.axvline(x=np.log10(0.01), color='orange', linestyle=':', linewidth=2, label='α=0.01 threshold')
    ax.set_xlabel('log10(α)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'M33 X-7 α Distribution (μ = {mu_m33:.1e} eV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "alpha_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved α distribution plot to {os.path.join(IMAGES_DIR, 'alpha_distribution.png')}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Bayesian ULB Constraint Analysis")
    print("=" * 60)
    
    # Load samples
    print("\n1. Loading posterior samples...")
    iras_samples, m33_samples = load_samples()
    print(f"   IRAS 09149-6206: {len(iras_samples)} samples")
    print(f"   M33 X-7: {len(m33_samples)} samples")
    
    # Define axion mass scan range
    # SMBH sensitive to ~10^-20 to 10^-16 eV
    # Stellar-mass sensitive to ~10^-14 to 10^-10 eV
    print("\n2. Scanning exclusion probability...")
    mu_range = np.logspace(-20, -10, 200)  # 10^-20 to 10^-10 eV
    
    # Compute exclusion probabilities
    p_iras, _ = scan_exclusion_probability(
        iras_samples[:, 0], iras_samples[:, 1], mu_range
    )
    p_m33, _ = scan_exclusion_probability(
        m33_samples[:, 0], m33_samples[:, 1], mu_range
    )
    
    # Derive upper limits
    print("\n3. Deriving 95% confidence upper limits...")
    ul_iras = derive_upper_limit(mu_range, p_iras, confidence=0.95)
    ul_m33 = derive_upper_limit(mu_range, p_m33, confidence=0.95)
    
    if ul_iras is not None:
        print(f"   IRAS 09149-6206: μ < {ul_iras:.2e} eV (95% CL)")
    else:
        print("   IRAS 09149-6206: No constraint in scanned range")
    
    if ul_m33 is not None:
        print(f"   M33 X-7: μ < {ul_m33:.2e} eV (95% CL)")
    else:
        print("   M33 X-7: No constraint in scanned range")
    
    # Also find the most excluded mass (peak of exclusion curve)
    print("\n4. Finding most sensitive mass ranges...")
    max_idx_iras = np.argmax(p_iras)
    max_idx_m33 = np.argmax(p_m33)
    print(f"   IRAS 09149-6206: Max exclusion P={p_iras[max_idx_iras]:.3f} at μ={mu_range[max_idx_iras]:.2e} eV")
    print(f"   M33 X-7: Max exclusion P={p_m33[max_idx_m33]:.3f} at μ={mu_range[max_idx_m33]:.2e} eV")
    
    # Create plots
    print("\n5. Generating figures...")
    create_exclusion_plot(mu_range, p_iras, p_m33, ul_iras, ul_m33)
    create_mass_spin_scatter(iras_samples, m33_samples)
    create_alpha_distribution_plot(iras_samples, m33_samples)
    
    # Save results
    print("\n6. Saving results...")
    results = {
        'mu_range_eV': mu_range.tolist(),
        'p_excluded_IRAS': p_iras.tolist(),
        'p_excluded_M33_X7': p_m33.tolist(),
        'upper_limit_95CL_IRAS_eV': float(ul_iras) if ul_iras else None,
        'upper_limit_95CL_M33_X7_eV': float(ul_m33) if ul_m33 else None,
        'max_exclusion_IRAS': {
            'mu_eV': float(mu_range[max_idx_iras]),
            'p_excluded': float(p_iras[max_idx_iras])
        },
        'max_exclusion_M33_X7': {
            'mu_eV': float(mu_range[max_idx_m33]),
            'p_excluded': float(p_m33[max_idx_m33])
        }
    }
    
    with open(os.path.join(OUTPUTS_DIR, "constraint_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved results to {os.path.join(OUTPUTS_DIR, 'constraint_results.json')}")
    
    # Save exclusion curve data
    exclusion_data = np.column_stack([mu_range, p_iras, p_m33])
    np.savetxt(os.path.join(OUTPUTS_DIR, "exclusion_curve.dat"), 
               exclusion_data, 
               header="mu_eV  p_excluded_IRAS  p_excluded_M33_X7",
               fmt='%.6e')
    print(f"   Saved exclusion curve data to {os.path.join(OUTPUTS_DIR, 'exclusion_curve.dat')}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = main()
