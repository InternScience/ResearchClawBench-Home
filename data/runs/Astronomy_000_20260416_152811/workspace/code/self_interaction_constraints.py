"""
Constraints on ULB self-interaction coupling from black hole superradiance.

When axion self-interactions become significant, they can:
1. Trigger Bosenova collapse of the axion cloud
2. Shut down superradiance through level mixing
3. Modify the exclusion boundaries

The self-interaction strength is parameterized by the decay constant f_a:
- QCD axion: f_a ~ 10^9 - 10^17 GeV
- Strong self-interactions: f_a < 10^14 GeV
- Weak self-interactions (our baseline): f_a >= 10^14 GeV

This module explores how self-interactions modify the constraints.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from superradiance_physics import compute_alpha, horizon_angular_velocity, superradiance_condition

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_000_20260416_152811"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")


def bosenova_critical_alpha(f_a_GeV, M_BH_msol):
    """
    Estimate the critical α where Bosenova collapse occurs.
    
    From Arvanitaki et al., Bosenova happens when attractive self-interactions
    overcome gravitational binding. This occurs at a critical occupation number
    that depends on f_a.
    
    Parameters:
    -----------
    f_a_GeV : float or array
        Axion decay constant in GeV
    M_BH_msol : float or array
        Black hole mass in solar masses
    
    Returns:
    --------
    alpha_critical : float or array
        Critical α value for Bosenova onset
    """
    # The critical particle number for Bosenova scales as:
    # N_crit ~ (f_a / M_pl)^2 * (M_pl / μ)^2
    # where M_pl = 1.22e19 GeV is the Planck mass
    
    M_pl_GeV = 1.22e19  # Planck mass in GeV
    
    # For typical axion masses, estimate critical α
    # This is a simplified model - full treatment requires solving Gross-Pitaevskii equation
    
    # Rough scaling: stronger self-interactions (lower f_a) => Bosenova at lower α
    # We parameterize this as a modification to the exclusion boundary
    
    # Return a representative critical α (this is approximate)
    alpha_crit = 0.1 * (f_a_GeV / 1e14)**0.5 * (M_BH_msol / 30.0)**(-0.25)
    return alpha_crit


def modified_exclusion_with_self_interaction(samples_mass, samples_spin, mu_eV, f_a_GeV=1e16):
    """
    Compute exclusion probability including self-interaction effects.
    
    Self-interactions can:
    1. Reduce the maximum cloud size (Bosenova)
    2. Shorten the instability time scale
    3. Create gaps in the exclusion region
    
    Parameters:
    -----------
    samples_mass : array
        BH mass posterior samples [Msol]
    samples_spin : array
        BH spin posterior samples
    mu_eV : float
        Axion mass [eV]
    f_a_GeV : float
        Axion decay constant [GeV]
    
    Returns:
    --------
    p_excluded : float
        Modified exclusion probability
    details : dict
        Detailed calculation information
    """
    n_samples = len(samples_mass)
    
    # Compute α for each sample
    alphas = compute_alpha(samples_mass, mu_eV)
    
    # Check superradiance condition
    is_sr = superradiance_condition(alphas, samples_spin)
    
    # Compute critical α for Bosenova
    alpha_crit = bosenova_critical_alpha(f_a_GeV, samples_mass)
    
    # Self-interaction suppression factor
    # When α > alpha_crit, self-interactions suppress the instability
    si_suppression = np.ones_like(alphas)
    si_mask = alphas > alpha_crit
    si_suppression[si_mask] = np.exp(-(alphas[si_mask] - alpha_crit[si_mask])**2 / (0.1**2))
    
    # Sensitive range: 0.01 < α < 1
    in_range = (alphas > 0.01) & (alphas < 1.0)
    
    # Apply self-interaction suppression
    effective_sr = is_sr & (si_suppression > 0.1)
    
    # Count excluded samples
    excluded = effective_sr & in_range
    p_excluded = np.sum(excluded) / n_samples
    
    return p_excluded, {
        'alphas': alphas,
        'alpha_crit': alpha_crit,
        'si_suppression': si_suppression,
        'is_sr': is_sr,
        'excluded': excluded
    }


def scan_fa_constraints(samples_mass, samples_spin, mu_range, fa_values):
    """
    Scan exclusion probability over both μ and f_a parameter space.
    
    Parameters:
    -----------
    samples_mass : array
        BH mass posterior samples
    samples_spin : array
        BH spin posterior samples
    mu_range : array
        Axion masses to test [eV]
    fa_values : array
        Decay constants to test [GeV]
    
    Returns:
    --------
    exclusion_matrix : 2D array
        P(excluded) for each (mu, f_a) combination
    """
    n_mu = len(mu_range)
    n_fa = len(fa_values)
    
    exclusion_matrix = np.zeros((n_mu, n_fa))
    
    for i, mu in enumerate(mu_range):
        for j, fa in enumerate(fa_values):
            p, _ = modified_exclusion_with_self_interaction(
                samples_mass, samples_spin, mu, f_a_GeV=fa
            )
            exclusion_matrix[i, j] = p
    
    return exclusion_matrix


def create_fa_constraint_plot(mu_range, fa_values, exclusion_matrix, bh_name):
    """Create 2D exclusion plot in (μ, f_a) plane."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for plotting
    mu_mesh, fa_mesh = np.meshgrid(mu_range, fa_values)
    
    # Plot as heatmap
    im = ax.pcolormesh(mu_mesh, fa_mesh, exclusion_matrix.T, 
                       cmap='viridis', shading='auto', vmin=0, vmax=1)
    
    # Add contour lines
    cs = ax.contour(mu_mesh, fa_mesh, exclusion_matrix.T, 
                    levels=[0.5, 0.9, 0.95], colors='white', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=9, fmt='P=%.2f')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Axion Mass μ [eV]', fontsize=12)
    ax.set_ylabel('Decay Constant f_a [GeV]', fontsize=12)
    ax.set_title(f'{bh_name}: ULB Constraints in (μ, f_a) Plane', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Exclusion Probability', fontsize=11)
    
    # Mark QCD axion band
    ax.axhline(y=1e14, color='red', linestyle='--', alpha=0.5, label='Strong SI limit')
    ax.axhline(y=1e16, color='orange', linestyle=':', alpha=0.5, label='Reference f_a')
    ax.axhline(y=1e18, color='green', linestyle=':', alpha=0.5, label='Weak SI')
    
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"fa_constraint_{bh_name.replace(' ', '_').lower()}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved f_a constraint plot for {bh_name}")


def load_samples():
    """Load saved posterior samples."""
    iras_samples = np.load(os.path.join(OUTPUTS_DIR, "iras_samples.npy"))
    m33_samples = np.load(os.path.join(OUTPUTS_DIR, "m33_samples.npy"))
    return iras_samples, m33_samples


def main():
    """Main analysis for self-interaction constraints."""
    print("=" * 60)
    print("Self-Interaction Constraint Analysis")
    print("=" * 60)
    
    # Load samples
    print("\n1. Loading posterior samples...")
    iras_samples, m33_samples = load_samples()
    
    # Define parameter ranges
    mu_range = np.logspace(-20, -10, 50)  # eV
    fa_values = np.logspace(12, 18, 30)   # GeV
    
    # Scan IRAS 09149-6206 (use subset for speed)
    print("\n2. Scanning (μ, f_a) parameter space for IRAS 09149-6206...")
    excl_iras = scan_fa_constraints(
        iras_samples[:, 0], iras_samples[:, 1], mu_range, fa_values
    )
    
    # Scan M33 X-7
    print("3. Scanning (μ, f_a) parameter space for M33 X-7...")
    excl_m33 = scan_fa_constraints(
        m33_samples[:, 0], m33_samples[:, 1], mu_range, fa_values
    )
    
    # Create plots
    print("\n4. Generating constraint plots...")
    create_fa_constraint_plot(mu_range, fa_values, excl_iras, "IRAS_09149-6206")
    create_fa_constraint_plot(mu_range, fa_values, excl_m33, "M33_X-7")
    
    # Save results
    print("\n5. Saving results...")
    results = {
        'mu_range_eV': mu_range.tolist(),
        'fa_values_GeV': fa_values.tolist(),
        'exclusion_IRAS': excl_iras.tolist(),
        'exclusion_M33_X7': excl_m33.tolist()
    }
    
    with open(os.path.join(OUTPUTS_DIR, "self_interaction_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to {os.path.join(OUTPUTS_DIR, 'self_interaction_results.json')}")
    
    print("\n" + "=" * 60)
    print("Self-Interaction Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
