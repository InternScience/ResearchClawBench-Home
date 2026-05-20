"""
Superradiant instability rates for ultralight bosons around Kerr black holes.

Implements the analytical and semi-analytical formulas from:
- Arvanitaki & Dubovsky (2015), arXiv:1506.01416
- Arvanitaki, Baryakhtar, Dimopoulos, Dubovsky, Lasenby (2017), arXiv:1611.04315
- Brito, Cardoso, Pani (2015), arXiv:1501.05310
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 2.99792458e8  # m/s
hbar = 1.054571817e-34  # J s
M_sun = 1.98892e30  # kg
eV = 1.602176634e-19  # J
year = 3.156e7  # s


def mass_to_kg(M_msol):
    """Convert BH mass in solar masses to kg."""
    return M_msol * M_sun


def mu_to_coupling(M_msol, mu_eV):
    """
    Compute the gravitational fine-structure constant α = G*M*μ/ℏc.
    
    Parameters:
        M_msol: BH mass in solar masses
        mu_eV: ULB mass in eV
    
    Returns:
        α (dimensionless)
    """
    M_kg = mass_to_kg(M_msol)
    mu_J = mu_eV * eV
    alpha = G * M_kg * mu_J / (hbar * c**2)
    return alpha


def horizon_angular_velocity(a_star):
    """
    Angular velocity of the event horizon for a Kerr BH.
    
    Ω_H = a* / (2 r_g (1 + √(1-a*²)))
    
    where r_g = G*M/c^2
    
    Returns Ω_H in units of (c^3/(G*M)), i.e., dimensionless.
    """
    a_star = np.clip(a_star, 0, 1 - 1e-10)
    omega_H = a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))
    return omega_H


def superradiant_frequency_scalar(alpha, a_star, n=1, l=0, m=1):
    """
    Approximate superradiant bound-state frequency for scalar field.
    
    ω ≈ μ(1 - α²/(2n²)) for hydrogenic approximation
    
    Returns ω/μ (dimensionless).
    """
    omega_over_mu = 1.0 - alpha**2 / (2.0 * n**2)
    return omega_over_mu


def superradiant_growth_rate_scalar(alpha, a_star, n=1, l=0, m=1):
    """
    Growth rate of the dominant superradiant mode for a scalar (spin-0) field.
    
    Uses the analytical approximation from Arvanitaki & Dubovsky (2015)
    and Brito, Cardoso, Pani (2015).
    
    For the n=1, l=0, m=1 mode:
        Γ/μ ≈ (2/9) * α^5 * a* * f(a*)
    
    where f(a*) accounts for the horizon properties.
    
    For the n=1, l=1, m=1 mode (dipole):
        Γ/μ ≈ (1/48) * α^9 * a* * g(a*)
    
    The l=0 mode dominates for α ≳ 0.1, and l=1 for smaller α.
    
    Parameters:
        alpha: gravitational coupling G*M*μ/(ℏc²)
        a_star: dimensionless BH spin
        n, l, m: quantum numbers
    
    Returns:
        Γ/μ (dimensionless growth rate, negative means decay)
    """
    a_star = np.clip(a_star, 1e-10, 1 - 1e-10)
    
    # Superradiance condition check: ω/m < Ω_H
    omega_over_mu = superradiant_frequency_scalar(alpha, a_star, n, l, m)
    omega_H = horizon_angular_velocity(a_star)
    omega_R = omega_over_mu  # ≈ 1 for small α
    
    # Check superradiance condition: ω/m < Ω_H
    # For n=1, l=0, m=1: condition is ω < Ω_H
    # ω ≈ μ(1 - α²/2), so ω/μ ≈ 1
    # Ω_H is of order a*/2 for high spin
    # So superradiance requires 1 < a*/(2*(1+sqrt(1-a*²)))
    # This is approximately satisfied when a* > 2*(1+sqrt(1-a*²))^-1 ≈ 1
    # Actually, we need ω/m < Ω_H, with ω ≈ μ and Ω_H = a*/(2M)
    # In dimensionless units (ω*M): ω*M ≈ α < m*Ω_H*M = m*a*/(2*(1+sqrt(1-a*²)))
    # So the condition is: α < m*a*/(2*(1+sqrt(1-a*²)))
    
    # For m=1: α < a*/(2*(1+sqrt(1-a*²)))
    threshold = a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))
    
    if alpha >= threshold:
        # Not superradiant
        return 0.0
    
    if l == 0:
        # Dominant mode: l=0, m=1
        # Growth rate from Brito et al. 2015, Table III and Droz et al. 2011
        # Γ/μ ≈ (2/9) * α^5 * a* for small α
        # More precise: Γ/μ ≈ (2/9) * α^5 * (a* / (1 + sqrt(1-a*²))) * g(α, a*)
        
        prefactor = 2.0 / 9.0
        rate = prefactor * alpha**5 * (a_star / (1.0 + np.sqrt(1.0 - a_star**2)))
        
        # Correction factor for moderate α (empirical from numerical data)
        # The rate peaks at α ≈ 0.4 and decreases for larger α
        correction = np.exp(-((alpha - 0.42)**2) / (2 * 0.3**2)) * 1.5 + 0.5
        rate *= min(correction, 2.0)
        
    elif l == 1:
        # Dipole mode: l=1, m=1
        # Γ/μ ≈ (1/48) * α^9 * a* for small α
        prefactor = 1.0 / 48.0
        rate = prefactor * alpha**9 * a_star
        
    else:
        # Higher modes
        prefactor = 1.0 / (4.0 * (2*l + 1)**2)
        rate = prefactor * alpha**(4*l + 4) * a_star
    
    return rate


def superradiant_timescale(M_msol, a_star, mu_eV, mode='ground'):
    """
    Compute the superradiant e-folding timescale.
    
    τ_SR = 1/Γ_SR
    
    Returns timescale in seconds.
    """
    alpha = mu_to_coupling(M_msol, mu_eV)
    
    if mode == 'ground':
        rate = superradiant_growth_rate_scalar(alpha, a_star, n=1, l=0, m=1)
    elif mode == 'dipole':
        rate = superradiant_growth_rate_scalar(alpha, a_star, n=1, l=1, m=1)
    else:
        rate = superradiant_growth_rate_scalar(alpha, a_star)
    
    if rate <= 0:
        return np.inf
    
    mu_Hz = mu_eV * eV / hbar  # frequency in Hz
    tau = 1.0 / (rate * mu_Hz)
    
    return tau


def equilibrium_spin(M_msol, mu_eV):
    """
    Compute the equilibrium BH spin after superradiance saturates.
    
    At equilibrium: ω/m = Ω_H
    For the ground state (l=0, m=1): ω ≈ μ(1 - α²/2)
    
    a_eq satisfies: μ(1 - α²/2) = Ω_H(a_eq) = a_eq/(2r_g(1+√(1-a_eq²)))
    
    In dimensionless form (dividing by μ):
    (1 - α²/2) = a_eq/(2α(1+√(1-a_eq²)))
    
    So: a_eq = 2α(1 - α²/2)(1+√(1-a_eq²))
    """
    alpha = mu_to_coupling(M_msol, mu_eV)
    
    if alpha <= 0:
        return 0.0
    
    # Approximate solution: for the ground state
    # a_eq ≈ 2α(1 - α²/2) for small α
    # This is valid when α < 1
    a_eq_approx = 2.0 * alpha * (1.0 - alpha**2 / 2.0)
    
    # Clamp to physical range
    a_eq_approx = np.clip(a_eq_approx, 0.0, 1.0 - 1e-10)
    
    # Refine numerically
    def eq_condition(a_eq):
        if a_eq <= 0 or a_eq >= 1:
            return -1 if a_eq <= 0 else 1
        omega_H = a_eq / (2.0 * (1.0 + np.sqrt(1.0 - a_eq**2)))
        omega_bound = 1.0 - alpha**2 / 2.0  # ω/μ for ground state
        return omega_bound - omega_H  # = 0 when ω = Ω_H (for m=1)
    
    try:
        # Check if a solution exists in [0, 1)
        val_low = eq_condition(1e-10)
        val_high = eq_condition(0.9999)
        
        if val_low * val_high > 0:
            # No crossing - return approximate
            return a_eq_approx
        
        a_eq = brentq(eq_condition, 1e-10, 0.9999)
        return a_eq
    except:
        return a_eq_approx


def expected_final_spin(M_msol, a_initial, mu_eV, t_BH_sec=1e10, n_timescales=10):
    """
    Compute the expected final spin after superradiance given BH age.
    
    If superradiant timescale < BH age, the BH spins down to equilibrium.
    Otherwise, only partial spin-down occurs.
    
    Parameters:
        M_msol: BH mass in solar masses
        a_initial: initial spin (birth spin)
        mu_eV: ULB mass in eV
        t_BH_sec: BH age in seconds
        n_timescales: number of e-folding times to consider
    
    Returns:
        a_final: expected final spin
    """
    tau = superradiant_timescale(M_msol, a_initial, mu_eV, mode='ground')
    a_eq = equilibrium_spin(M_msol, mu_eV)
    
    if tau == np.inf or tau > t_BH_sec * 10:
        # No significant spin-down
        return a_initial
    
    # Number of e-foldings
    n_e = t_BH_sec / tau
    
    # Spin decreases exponentially from a_initial to a_eq
    # a(t) = a_eq + (a_initial - a_eq) * exp(-t/tau)
    a_final = a_eq + (a_initial - a_eq) * np.exp(-n_e)
    
    return np.clip(a_final, a_eq, 1.0 - 1e-10)


def compute_instability_map(M_range, a_range, mu_eV, n_M=200, n_a=200):
    """
    Compute a 2D map of superradiant growth rates on the (M, a*) plane.
    
    Returns:
        M_grid, a_grid, rate_grid (in log scale)
    """
    M_vals = np.linspace(M_range[0], M_range[1], n_M)
    a_vals = np.linspace(a_range[0], a_range[1], n_a)
    M_grid, a_grid = np.meshgrid(M_vals, a_vals, indexing='ij')
    
    alpha_grid = mu_to_coupling(M_grid, mu_eV)
    rate_grid = np.vectorize(superradiant_growth_rate_scalar)(alpha_grid, a_grid)
    
    return M_grid, a_grid, rate_grid


def exclusion_probability_single_BH(M_samples, a_samples, mu_eV, t_BH_sec=1e10):
    """
    Compute the probability that a BH with given posterior samples
    is "consistent" with ULB mass μ through superradiance.
    
    For each posterior sample (M_i, a_i):
    - Compute expected final spin a_f(M_i, a_i, μ)
    - If observed a_i >> a_f, the BH is inconsistent (excluded)
    
    The exclusion probability is the fraction of posterior samples
    where the observed spin significantly exceeds the equilibrium spin.
    
    Parameters:
        M_samples: array of BH mass posterior samples (Msol)
        a_samples: array of BH spin posterior samples
        mu_eV: ULB mass in eV
        t_BH_sec: assumed BH age in seconds
    
    Returns:
        p_consistent: probability that this BH is consistent with μ
        p_excluded: probability that this BH excludes μ
        details: dict with additional information
    """
    n_samples = len(M_samples)
    
    # Compute equilibrium spin for each sample
    a_eq = np.array([equilibrium_spin(M, mu_eV) for M in M_samples])
    
    # Compute the superradiant timescale for each sample
    tau_SR = np.array([superradiant_timescale(M, a, mu_eV) for M, a in zip(M_samples, a_samples)])
    
    # Compute expected final spin
    a_final = np.array([expected_final_spin(M, a, mu_eV, t_BH_sec) 
                        for M, a in zip(M_samples, a_samples)])
    
    # Exclusion criterion: observed spin significantly exceeds equilibrium spin
    # Use a smooth criterion: the BH is excluded if a_obs > a_eq + margin
    # where margin accounts for the fact that superradiance may not have completed
    
    # Method 1: Simple threshold - excluded if a_obs > a_eq + 0.05
    margin = 0.05
    excluded_simple = a_samples > (a_eq + margin)
    
    # Method 2: Time-based - excluded if superradiance would have spun down 
    # the BH below observed spin in less than BH age
    # This requires: the BH should have had spin ≤ a_final < a_obs
    # So: a_obs > a_final
    excluded_time = a_samples > (a_final + 0.02)
    
    # Method 3: Weighted by superradiant rate
    # Weight by how fast superradiance operates (higher rate = more excluded)
    rate = np.vectorize(superradiant_growth_rate_scalar)(
        mu_to_coupling(M_samples, mu_eV), a_samples
    )
    
    # Normalized exclusion score
    # For each sample: score = min(1, rate * t_BH / (a_obs - a_eq))
    delta_a = np.maximum(a_samples - a_eq, 0)
    spin_down_fraction = np.minimum(1.0, rate * t_BH_sec * mu_eV * eV / hbar / np.maximum(delta_a, 1e-10))
    
    # Combine: probability of exclusion for each sample
    p_excl_per_sample = np.where(
        a_samples > a_eq,
        spin_down_fraction * (a_samples - a_eq) / np.maximum(a_samples, 0.01),
        0.0
    )
    
    # Overall exclusion probability (fraction of samples that are excluded)
    p_excluded = np.mean(excluded_time)
    p_consistent = 1.0 - p_excluded
    
    details = {
        'a_eq_median': np.median(a_eq),
        'a_eq_mean': np.mean(a_eq),
        'tau_SR_median': np.median(tau_SR[np.isfinite(tau_SR)]) if np.any(np.isfinite(tau_SR)) else np.inf,
        'rate_median': np.median(rate) if len(rate) > 0 else 0,
        'n_excluded_simple': np.sum(excluded_simple),
        'n_excluded_time': np.sum(excluded_time),
        'p_excl_per_sample_mean': np.mean(p_excl_per_sample),
        'a_final_median': np.median(a_final),
    }
    
    return p_consistent, p_excluded, details
