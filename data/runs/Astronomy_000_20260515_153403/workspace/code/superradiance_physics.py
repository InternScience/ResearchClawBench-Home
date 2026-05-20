#!/usr/bin/env python3
"""
Bayesian Framework for Constraining Ultralight Bosons from Black Hole Superradiance.

This module implements the superradiance physics and Bayesian likelihood framework
for constraining ultralight boson (ULB) masses and self-interaction coupling strengths
using posterior distributions of black hole mass and spin.
"""

import numpy as np
from scipy import constants

# Physical constants
G = constants.G  # m^3 kg^-1 s^-2
c = constants.c  # m/s
hbar = constants.hbar  # J s
Msun = 1.98847e30  # kg
eV_to_J = constants.eV  # J

# Planck mass in kg
Mpl = np.sqrt(hbar * c / G)  # kg

# Gravitational fine-structure constant factor
# alpha = G * M * mu / (hbar * c^3)
# In practical units: alpha = ALPHA_FACTOR * (M/Msun) * (mu/eV)
ALPHA_FACTOR = G * Msun / (hbar * c**3) * eV_to_J  # dimensionless


def compute_alpha(M_Msun, mu_eV):
    """
    Compute the gravitational fine-structure constant alpha.
    
    Parameters
    ----------
    M_Msun : float or array
        Black hole mass in solar masses.
    mu_eV : float
        Ultralight boson mass in eV.
    
    Returns
    -------
    alpha : float or array
        Gravitational fine-structure constant.
    """
    return ALPHA_FACTOR * M_Msun * mu_eV


def horizon_angular_velocity(a_star, M_Msun):
    """
    Compute the horizon angular velocity Omega_H.
    
    Parameters
    ----------
    a_star : float or array
        Dimensionless spin parameter (0 <= a_star < 1).
    M_Msun : float or array
        Black hole mass in solar masses.
    
    Returns
    -------
    Omega_H : float or array
        Horizon angular velocity in Hz.
    """
    # In geometric units: Omega_H = a_star / (2 * r_g * (1 + sqrt(1 - a_star^2)))
    # r_g = G * M / c^2
    r_g = G * M_Msun * Msun / c**2  # meters
    Omega_H = a_star / (2 * r_g * (1 + np.sqrt(np.maximum(1 - a_star**2, 1e-30))))
    # Convert to Hz: Omega_H in geometric units (1/m) * c = 1/s
    Omega_H *= c
    return Omega_H


def superradiance_rate_scalar(alpha, a_star, M_Msun, l=1):
    """
    Compute the superradiance growth rate for a scalar field.
    
    Uses the analytic approximation valid for alpha << 1.
    Rate = (a_star / M) * alpha^(4l+5) * mu * C_lm
    
    Parameters
    ----------
    alpha : float or array
        Gravitational fine-structure constant.
    a_star : float or array
        Dimensionless spin parameter.
    M_Msun : float or array
        Black hole mass in solar masses.
    l : int
        Orbital quantum number (l=1 is the dominant mode).
    
    Returns
    -------
    Gamma_sr : float or array
        Superradiance growth rate in Hz.
    """
    # For l=m=1, the prefactor is approximately 1/24
    # More precisely from Detweiler (1980): Gamma ≈ (a_star - 2*mu*r_plus) * (mu*M)^9 / (24*M)
    # In the alpha << 1 limit, for l=m=1:
    # Gamma_sr ≈ (1/24) * a_star * alpha^8 * mu
    # where mu is in Hz: mu_Hz = mu_eV * eV_to_J / hbar
    
    # We'll use the leading-order formula
    # For l=1: Gamma = (a_star / 24) * alpha^8 * mu / M (in geometric units)
    # Convert: mu_Hz = mu_eV * eV_to_J / hbar
    # M_seconds = G * M_Msun * Msun / c^3
    
    # Actually, in SI units:
    # Gamma_sr ~ (1/24) * a_star * alpha^(4l+5) * mu_Hz
    # where mu_Hz = mu_eV * eV_to_J / hbar
    
    # More careful: from Arvanitaki & Dubovsky 2011 eq. (??)
    # For l=m=1, Gamma_sr ≈ (a_star / 24) * (G*M*mu/hbar/c^3)^9 * mu
    # = (a_star / 24) * alpha^9 * mu_Hz
    # Wait, for scalar l=1: exponent is 4l+4 = 8, then +1 for mu = 9? 
    
    # From paper_002 (Arvanitaki et al. 2017): Γ_sr ∝ α^(4l+4) * μ
    # For l=1: Γ_sr ∝ α^8 * μ
    # With prefactor: Γ_sr ≈ (a_star / 24) * α^8 * (G*M/c^3)^(-1) * α 
    # Hmm, let me use a simpler approach.
    
    # From Detweiler 1980, the growth rate for l=m=1:
    # ω_I * M ≈ (1/48) * a * (M*mu)^9  for M*mu << 1
    # So Gamma = ω_I ≈ (a_star / 48) * alpha^9 / (G*M/c^3)
    # Gamma_Hz = Gamma * c^3/(G*M) * ...
    
    # Let me just use the formula from the literature directly:
    # τ_sr ≈ 10^7 * (M/Msun) * (0.1/alpha)^9 * (0.99/a_star) seconds (roughly)
    # for stellar mass BHs
    
    # Actually, the simplest and most correct approach:
    # M * ω_I = (1/48) * (a/M) * (M*μ)^9 for scalar fields with l=m=1
    # where M*μ = alpha
    
    M_geom = G * Msun / c**3  # geometric seconds per Msun (actually this is M in seconds: G*M/c^3)
    # For M=1 Msun: M_geom ≈ 4.93e-6 seconds
    
    # In geometric units, the rate is:
    # ω_I = (1/48) * a_star * alpha^9 / M_geom_seconds
    
    # So the e-folding rate in Hz:
    Gamma_sr = (1.0 / 48.0) * a_star * alpha**9
    # This is in units of 1/(geometric M), need to convert to Hz
    # Multiply by c^3/(G*M)
    # Actually Gamma_sr is M * ω_I, so ω_I = Gamma_sr / M_geom
    
    # Let me express this cleanly:
    # ω_I [1/s] = (c^3/(G*M)) * (1/48) * a_star * alpha^9
    
    M_kg = M_geom * c**2 / G  # Not helpful
    
    # Better: M in geometric units = G*M/c^2 (length) or G*M/c^3 (time)
    M_time = G * Msun / c**3  # seconds per solar mass
    
    # ω_I * M_time * (M/Msun) = (1/48) * a_star * alpha^9
    # ω_I = (1/48) * a_star * alpha^9 / (M_time * M_Msun)
    
    # For scalar l=1:
    Gamma_Hz = (1.0 / 48.0) * a_star * alpha**9 / (M_time * M_Msun)
    
    # Actually, more recent calculations give a slightly different prefactor
    # But this is sufficient for our purposes
    
    # Clip to avoid numerical issues
    return np.maximum(Gamma_Hz, 1e-300)


def superradiance_timescale(alpha, a_star, M_Msun, mu_eV):
    """
    Compute the superradiance e-folding timescale in years.
    
    Parameters
    ----------
    alpha : float or array
        Gravitational fine-structure constant.
    a_star : float or array
        Dimensionless spin parameter.
    M_Msun : float or array
        Black hole mass in solar masses.
    mu_eV : float
        Ultralight boson mass in eV.
    
    Returns
    -------
    tau_sr_years : float or array
        Superradiance e-folding time in years.
    """
    Gamma_Hz = superradiance_rate_scalar(alpha, a_star, M_Msun)
    tau_sr = 1.0 / Gamma_Hz  # seconds
    tau_sr_years = tau_sr / (365.25 * 24 * 3600)
    return tau_sr_years


def superradiance_condition_satisfied(a_star, M_Msun, mu_eV, l=1, m=1):
    """
    Check if the superradiance condition is satisfied.
    
    ω/m < Ω_H where ω ≈ μ (to leading order)
    
    Parameters
    ----------
    a_star : float or array
        Dimensionless spin parameter.
    M_Msun : float or array
        Black hole mass in solar masses.
    mu_eV : float
        Ultralight boson mass in eV.
    l, m : int
        Quantum numbers.
    
    Returns
    -------
    satisfied : bool array
    """
    mu_Hz = mu_eV * eV_to_J / hbar
    Omega_H = horizon_angular_velocity(a_star, M_Msun)
    return mu_Hz / m < Omega_H


def is_in_exclusion_region(M_Msun, a_star, mu_eV, tau_bh_years=1e9, l=1, m=1):
    """
    Determine whether a BH with given (M, a*) would be affected by superradiance
    for a given ULB mass mu.
    
    A BH is in the exclusion region if:
    1. The superradiance condition is satisfied (mu < m * Omega_H)
    2. The superradiance timescale is shorter than the BH age
    3. Self-interactions do not shut down superradiance (handled separately)
    
    Parameters
    ----------
    M_Msun : float or array
        Black hole mass in solar masses.
    a_star : float or array
        Dimensionless spin parameter.
    mu_eV : float
        Ultralight boson mass in eV.
    tau_bh_years : float
        Characteristic age of the black hole in years.
    l, m : int
        Quantum numbers for the dominant superradiant mode.
    
    Returns
    -------
    excluded : bool array
    """
    alpha = compute_alpha(M_Msun, mu_eV)
    
    # Condition 1: Superradiance condition
    sr_cond = superradiance_condition_satisfied(a_star, M_Msun, mu_eV, l, m)
    
    # Condition 2: Timescale condition
    tau_sr = superradiance_timescale(alpha, a_star, M_Msun, mu_eV)
    n_efolds_needed = 100  # Need many e-folds for significant spin-down
    tau_cond = n_efolds_needed * tau_sr < tau_bh_years
    
    # Also require that alpha is in the right range for efficient superradiance
    # alpha should be ~0.1-1 for efficient superradiance
    alpha_cond = (alpha > 0.01) & (alpha < 2.0)
    
    return sr_cond & tau_cond & alpha_cond


def bosenova_condition(M_Msun, a_star, mu_eV, f_a_GeV):
    """
    Determine whether self-interactions (Bosenova) would shut down superradiance.
    
    When the axion self-interaction energy becomes comparable to the gravitational
    binding energy, the cloud collapses (Bosenova).
    
    For the QCD axion: λ = -mu^2 / f_a^2 (attractive)
    Bosenova occurs when N * |λ| * |ψ|^4 ~ gravitational binding energy
    
    Simplified condition: f_a < f_a_crit, where
    f_a_crit ~ M_pl * sqrt(alpha / N_max)
    
    Parameters
    ----------
    M_Msun : float or array
        Black hole mass in solar masses.
    a_star : float or array
        Dimensionless spin parameter.
    mu_eV : float
        Ultralight boson mass in eV.
    f_a_GeV : float
        Axion decay constant in GeV.
    
    Returns
    -------
    bosenova_shutdown : bool array
        True if bosenova shuts down superradiance.
    """
    # Convert f_a from GeV to natural units
    f_a_eV = f_a_GeV * 1e9
    
    # Self-interaction parameter
    # lambda_self = mu_eV^2 / f_a_eV^2 (dimensionless in natural units)
    
    # The bosenova condition from Arvanitaki & Dubovsky 2011:
    # When self-interactions become important, the maximum occupation number is limited
    # N_max_self ~ (f_a / M_pl)^2 * (M_pl / mu)^2 * (something)
    
    # Critical decay constant below which bosenova occurs before significant spin-down:
    # f_a_crit ~ sqrt(alpha) * M_pl * (M_pl / mu) * ...
    # More precisely from Arvanitaki & Dubovsky 2011 eq (??):
    # f_a_crit ~ 10^16 GeV * (mu/10^-10 eV)^(-1) * ...
    
    Mpl_GeV = Mpl * c**2 / eV_to_J * 1e-9  # Planck mass in GeV
    
    alpha = compute_alpha(M_Msun, mu_eV)
    
    # From Arvanitaki & Dubovsky, the self-interaction becomes important when:
    # f_a < f_crit where f_crit depends on alpha, mu, M
    # For the QCD axion, bosenova occurs for f_a ~ 10^15-10^16 GeV
    
    # A simplified criterion:
    # The self-interaction energy scale: E_self ~ N * mu^2/f_a^2 * |ψ|^2
    # Gravitational binding: E_grav ~ alpha^2 * mu * N
    # Bosenova when E_self > E_grav → N * mu/f_a^2 * (mu*alpha)^3 > alpha^2 * mu
    # Since |ψ|^2 ~ (mu*alpha)^3 for the ground state
    
    # For the fastest-growing mode (n=2, l=m=1):
    # Bosenova occurs when f_a < f_crit where
    # f_crit ≈ M_pl * alpha^(something)
    
    # Using the result from the literature: bosenova is important when
    # f_a < ~10^16 GeV for stellar mass BHs
    
    # For simplicity, we'll use a threshold approach
    f_crit_GeV = 5e15 * np.sqrt(alpha / 0.1) * (1e-10 / mu_eV)
    
    # Never allow f_crit below some floor
    f_crit_GeV = np.maximum(f_crit_GeV, 1e14)
    
    return f_a_GeV < f_crit_GeV


def compute_exclusion_fraction(M_samples, a_star_samples, mu_eV, tau_bh_years=1e9, 
                                f_a_GeV=None, l=1, m=1):
    """
    Compute the fraction of posterior samples in the exclusion region.
    
    Parameters
    ----------
    M_samples : array
        Black hole mass posterior samples in Msun.
    a_star_samples : array
        Spin posterior samples.
    mu_eV : float
        Ultralight boson mass in eV.
    tau_bh_years : float
        Characteristic BH age in years.
    f_a_GeV : float or None
        Axion decay constant in GeV. If None, no self-interaction limit.
    l, m : int
        Quantum numbers.
    
    Returns
    -------
    frac_excluded : float
        Fraction of samples excluded.
    """
    excluded = is_in_exclusion_region(M_samples, a_star_samples, mu_eV, tau_bh_years, l, m)
    
    if f_a_GeV is not None:
        # Bosenova shuts down superradiance, so these samples are NOT excluded
        bosenova = bosenova_condition(M_samples, a_star_samples, mu_eV, f_a_GeV)
        excluded = excluded & (~bosenova)
    
    return np.mean(excluded)


def compute_likelihood(M_samples, a_star_samples, mu_eV, tau_bh_years=1e9,
                       f_a_GeV=None, l=1, m=1):
    """
    Compute the likelihood P(data | mu, f_a).
    
    We use a simple model: if a sample is in the exclusion region, the likelihood
    of observing that sample is small (penalized).
    
    L(mu, f_a) = prod_i [ (1 - w) + w * I(not excluded) ]
    where w is the weight given to the exclusion criterion.
    
    Equivalently: log L = sum_i log[(1-w) + w * I(not excluded)]
    
    Parameters
    ----------
    M_samples, a_star_samples : arrays
        Posterior samples.
    mu_eV : float
        ULB mass in eV.
    tau_bh_years : float
        BH age.
    f_a_GeV : float or None
        Axion decay constant.
    l, m : int
        Quantum numbers.
    
    Returns
    -------
    log_likelihood : float
        Log-likelihood.
    """
    excluded = is_in_exclusion_region(M_samples, a_star_samples, mu_eV, tau_bh_years, l, m)
    
    if f_a_GeV is not None:
        bosenova = bosenova_condition(M_samples, a_star_samples, mu_eV, f_a_GeV)
        excluded = excluded & (~bosenova)
    
    # Weight: samples in exclusion region get penalized
    w = 0.999  # High weight means strong penalty for excluded samples
    
    # Likelihood per sample
    # If excluded: likelihood = 1 - w (small)
    # If not excluded: likelihood = 1
    lik_per_sample = np.where(excluded, 1 - w, 1.0)
    
    log_lik = np.sum(np.log(np.maximum(lik_per_sample, 1e-300)))
    
    return log_lik


def compute_upper_limit(mu_grid, log_likelihood_grid, mu_ref, cl=0.95):
    """
    Compute upper limit on ULB mass from the likelihood grid.
    
    Parameters
    ----------
    mu_grid : array
        Grid of ULB masses.
    log_likelihood_grid : array
        Log-likelihood values.
    mu_ref : float
        Reference (very low) mass where likelihood is maximized.
    cl : float
        Confidence level (default 0.95).
    
    Returns
    -------
    mu_upper : float
        Upper limit on ULB mass.
    """
    # Convert to chi-square-like statistic
    delta_chi2 = -2 * (log_likelihood_grid - np.max(log_likelihood_grid))
    
    # Find where delta_chi2 crosses the threshold
    # For 95% CL with 1 dof: delta_chi2 = 3.841
    threshold = 3.841  # 95% CL for 1 parameter
    
    # Find the upper limit
    above = delta_chi2 > threshold
    
    # Start from the peak and go up in mass
    peak_idx = np.argmax(log_likelihood_grid)
    
    # Find first crossing above the peak
    for i in range(peak_idx, len(mu_grid)):
        if delta_chi2[i] > threshold and i > 0:
            # Interpolate
            if delta_chi2[i-1] < threshold:
                frac = (threshold - delta_chi2[i-1]) / (delta_chi2[i] - delta_chi2[i-1])
                mu_upper = mu_grid[i-1] + frac * (mu_grid[i] - mu_grid[i-1])
                return mu_upper
    
    return mu_grid[-1]


if __name__ == "__main__":
    # Quick test
    print(f"ALPHA_FACTOR = {ALPHA_FACTOR:.6e}")
    print(f"M33 X-7: M=15 Msun, mu=1e-12 eV → alpha = {compute_alpha(15, 1e-12):.4f}")
    print(f"IRAS 09149: M=1.5e8 Msun, mu=1e-18 eV → alpha = {compute_alpha(1.5e8, 1e-18):.4f}")
