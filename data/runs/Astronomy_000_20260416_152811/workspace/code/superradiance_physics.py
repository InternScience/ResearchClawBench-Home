"""
Implement the physics of black hole superradiance for ULB constraints.

Key physics:
1. Superradiance occurs when axion Compton wavelength ~ BH gravitational radius
2. Fine-structure constant: α = G_N * M_BH * μ_a / (ℏ c)
3. In natural units (G = c = ℏ = 1): α = M_BH * μ_a
4. Superradiant condition: ω/m < Ω_H where Ω_H is horizon angular velocity
5. Fastest growth for l=m=1 mode when α ~ 0.4

The instability time scale depends on α and spin a*.
High-spin BHs exclude axion masses that would cause rapid spin-down.

Using the scaling from Arvanitaki et al. (paper_002):
α ≈ 0.22 * (M_BH / 30 Msol) * (μ_a / 10^-12 eV)
"""
import numpy as np


def compute_alpha(M_BH_msol, mu_eV):
    """
    Compute the gravitational fine-structure constant α.
    
    Using the scaling from Arvanitaki et al. (paper_002):
    α ≈ 0.22 * (M_BH / 30 Msol) * (μ_a / 10^-12 eV)
    
    Parameters:
    -----------
    M_BH_msol : float or array
        Black hole mass in solar masses
    mu_eV : float or array
        Axion/ULB mass in eV
    
    Returns:
    --------
    alpha : float or array
        Dimensionless fine-structure constant
    """
    # Use the literature scaling (Arvanitaki et al.)
    alpha = 0.22 * (M_BH_msol / 30.0) * (mu_eV / 1e-12)
    return alpha


def horizon_angular_velocity(a_star):
    """
    Compute the horizon angular velocity Ω_H for a Kerr black hole.
    
    Ω_H = (1 / (2 * r_g)) * (a_* / (1 + sqrt(1 - a_*^2)))
    
    where r_g = G * M / c^2 is the gravitational radius.
    
    Parameters:
    -----------
    a_star : float or array
        Dimensionless spin parameter (0 <= a_* < 1)
    
    Returns:
    --------
    Omega_H : float or array
        Horizon angular velocity in units of c^3/(G*M)
    """
    # Return in dimensionless units (c^3/(G*M))
    sqrt_term = np.sqrt(1 - a_star**2)
    Omega_H_dimless = 0.5 * a_star / (1 + sqrt_term)
    return Omega_H_dimless


def superradiance_condition(alpha, a_star, l=1, m=1):
    """
    Check if superradiance condition is satisfied.
    
    For hydrogenic levels, ω ≈ μ * (1 - α^2/(2*n^2))
    Superradiance occurs when ω/m < Ω_H
    
    The fastest growing mode is typically l=m=1, n=2.
    
    Parameters:
    -----------
    alpha : float or array
        Fine-structure constant
    a_star : float or array
        Dimensionless spin parameter
    l, m : int
        Angular momentum quantum numbers
    
    Returns:
    --------
    is_superradiant : bool or array
        True if superradiance condition is satisfied
    """
    # For the fastest growing mode (n=2, l=m=1)
    n = 2
    omega_mu = 1 - alpha**2 / (2 * n**2)  # ω/μ
    
    # Dimensionless horizon angular velocity
    Omega_H_dimless = horizon_angular_velocity(a_star)
    
    # Superradiance condition: ω/m < Ω_H * (G*M/c^3) * c^2/μ
    # In our units: ω/μ < m * Ω_H_dimless / α
    # So: (1 - α^2/(2n^2)) < m * Ω_H_dimless / α
    # For m=1: α * (1 - α^2/(2n^2)) < Ω_H_dimless
    
    lhs = alpha * omega_mu
    rhs = m * Omega_H_dimless
    
    return lhs < rhs


def instability_growth_rate(alpha, a_star, l=1, m=1):
    """
    Approximate the superradiance instability growth rate.
    
    Γ_sr ∝ α^(4l+4) * μ_a * (m * Ω_H - ω)
    
    This is a simplified approximation based on the literature.
    
    Parameters:
    -----------
    alpha : float or array
        Fine-structure constant
    a_star : float or array
        Dimensionless spin parameter
    l, m : int
        Angular momentum quantum numbers
    
    Returns:
    --------
    Gamma : float or array
        Growth rate in units of μ_a (dimensionless when multiplied by c^2/ℏ)
    """
    n = l + 1  # Principal quantum number for fastest mode
    
    # ω/μ for hydrogenic levels
    omega_mu = 1 - alpha**2 / (2 * n**2)
    
    # Dimensionless horizon angular velocity
    Omega_H_dimless = horizon_angular_velocity(a_star)
    
    # Superradiant factor (m * Ω_H - ω) / μ
    sr_factor = m * Omega_H_dimless / alpha - omega_mu
    
    # Only positive for superradiant modes
    sr_factor = np.maximum(sr_factor, 0)
    
    # Growth rate scaling (from Arvanitaki et al.)
    # Γ ∝ α^(4l+4) for small α
    Gamma_dimless = alpha**(4*l + 4) * sr_factor
    
    return Gamma_dimless


def exclusion_probability(samples_mass, samples_spin, mu_eV, threshold_alpha=0.01):
    """
    Compute the probability that a given axion mass is excluded by the BH observations.
    
    A mass is "excluded" if the observed BH has high enough spin that
    superradiance should have occurred (and spun it down) if that axion mass existed.
    
    We consider a sample as "in tension" if:
    1. α is in the sensitive range (roughly 0.01 to 1)
    2. The superradiance condition is satisfied
    3. The instability time scale is shorter than the BH age
    
    Parameters:
    -----------
    samples_mass : array
        BH mass posterior samples [Msol]
    samples_spin : array
        BH spin posterior samples
    mu_eV : float
        Axion mass to test [eV]
    threshold_alpha : float
        Minimum α for significant superradiance
    
    Returns:
    --------
    p_excluded : float
        Probability that this axion mass is excluded [0, 1]
    details : dict
        Detailed information about the calculation
    """
    n_samples = len(samples_mass)
    
    # Compute α for each sample
    alphas = compute_alpha(samples_mass, mu_eV)
    
    # Check superradiance condition
    is_sr = superradiance_condition(alphas, samples_spin)
    
    # Compute growth rates
    growth_rates = instability_growth_rate(alphas, samples_spin)
    
    # Consider excluded if:
    # 1. α is in sensitive range (not too small, not too large)
    # 2. Superradiance condition satisfied
    # 3. Growth rate is significant
    
    # Sensitive range: roughly 0.01 < α < 1
    in_range = (alphas > threshold_alpha) & (alphas < 1.0)
    
    # Significant growth (heuristic threshold)
    significant_growth = growth_rates > 1e-10
    
    # Count excluded samples
    excluded = is_sr & in_range & significant_growth
    p_excluded = np.sum(excluded) / n_samples
    
    return p_excluded, {
        'alphas': alphas,
        'is_sr': is_sr,
        'growth_rates': growth_rates,
        'in_range': in_range,
        'significant_growth': significant_growth,
        'excluded': excluded
    }


def main():
    """Test the physics functions."""
    print("Testing superradiance physics functions...")
    
    # IRAS 09149-6206: M ~ 10^8 Msol
    M_iras = 1e8
    # M33 X-7: M ~ 15 Msol
    M_m33 = 15.7
    
    # Test axion masses
    mu_test = 1e-18  # eV - typical for SMBH sensitivity
    
    print(f"\nFor μ = {mu_test} eV:")
    print(f"  IRAS 09149-6206 (M = {M_iras:.1e} Msol): α = {compute_alpha(M_iras, mu_test):.4f}")
    print(f"  M33 X-7 (M = {M_m33:.1f} Msol): α = {compute_alpha(M_m33, mu_test):.4f}")
    
    # Find optimal α range for each BH
    print("\nOptimal axion mass ranges (α ~ 0.4):")
    for M, name in [(M_iras, "IRAS 09149-6206"), (M_m33, "M33 X-7")]:
        # Using the scaling: μ = (α / 0.22) * (30 / M_Msol) * 1e-12 eV
        mu_opt = (0.4 / 0.22) * (30.0 / M) * 1e-12  # eV
        print(f"  {name}: μ ~ {mu_opt:.2e} eV")


if __name__ == "__main__":
    main()
