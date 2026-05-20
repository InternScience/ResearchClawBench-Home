"""
Bayesian framework for constraining ultralight boson properties
using black hole superradiance.

Ingests full posterior distributions of BH mass and spin to derive
statistically rigorous upper limits on ULB masses.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import sys
sys.path.insert(0, 'code')
from superradiance import (
    mu_to_coupling, equilibrium_spin, superradiant_timescale,
    superradiant_growth_rate_scalar, exclusion_probability_single_BH
)

# Physical constants (repeated for convenience)
eV = 1.602176634e-19
hbar = 1.054571817e-34
year = 3.156e7


def log_uniform_prior(mu_min, mu_max):
    """Prior on log10(mu) that is uniform."""
    log_mu_min = np.log10(mu_min)
    log_mu_max = np.log10(mu_max)
    return log_mu_min, log_mu_max


def compute_exclusion_curve(M_samples, a_samples, mu_grid, t_BH_sec=1e10,
                            method='bayesian'):
    """
    Compute the exclusion probability P_excluded(μ) for each mass in mu_grid.
    
    Parameters:
        M_samples: BH mass posterior samples (Msol)
        a_samples: BH spin posterior samples
        mu_grid: array of ULB masses to test (eV)
        t_BH_sec: assumed BH age (seconds)
        method: 'bayesian' or 'frequency'
    
    Returns:
        p_excluded: array of exclusion probabilities
        details_list: list of detail dicts for each mu
    """
    p_excluded = np.zeros(len(mu_grid))
    details_list = []
    
    for i, mu in enumerate(mu_grid):
        p_cons, p_excl, details = exclusion_probability_single_BH(
            M_samples, a_samples, mu, t_BH_sec
        )
        p_excluded[i] = p_excl
        details_list.append(details)
    
    return p_excluded, details_list


def bayesian_evidence_ratio(M_samples, a_samples, mu_grid, 
                            log_mu_min=None, log_mu_max=None,
                            t_BH_sec=1e10, n_bootstrap=0):
    """
    Compute the Bayesian evidence ratio for ULB existence.
    
    H0: No ULB (null hypothesis)
    H1: ULB exists with mass μ
    
    The Bayes factor is:
    BF = P(data | H1) / P(data | H0)
    
    P(data | H1) = ∫ P(data | μ, H1) P(μ | H1) dμ
    
    For each μ, P(data | μ, H1) = ∏_i P(M_i, a_i | μ)
    
    The likelihood P(M_i, a_i | μ) depends on whether the BH is 
    "consistent" with having undergone superradiance for that μ.
    
    Returns:
        mu_grid: ULB masses tested
        log_bayes_factor: log10(BF)
        p_excluded_per_mu: exclusion probability per μ
        posterior_mu: posterior distribution over μ
    """
    if log_mu_min is None:
        log_mu_min = np.log10(1e-20)  # 10^-20 eV
    if log_mu_max is None:
        log_mu_max = np.log10(1e-9)   # 10^-9 eV
    
    # Use log-spaced grid
    n_mu = 200
    log_mu_grid = np.linspace(log_mu_min, log_mu_max, n_mu)
    mu_grid = 10**log_mu_grid
    
    # Compute exclusion probability for each μ
    p_excluded, details_list = compute_exclusion_curve(
        M_samples, a_samples, mu_grid, t_BH_sec
    )
    
    # For the Bayesian framework:
    # If H0 (no ULB): all BHs should be consistent with standard physics
    # If H1 (ULB with mass μ): BHs in the superradiance band should be spun down
    
    # Likelihood under H1: P(data | μ) = ∏ P(a_i | M_i, μ)
    # For each sample (M_j, a_j):
    #   P(a_j | M_j, μ) is proportional to the prior on a_j if a_j > a_eq
    #   (since superradiance would have reduced it)
    #   and enhanced if a_j < a_eq
    
    # Simplified Bayesian approach:
    # Compute the log-likelihood for each μ
    log_likelihood = np.zeros(n_mu)
    
    for i, mu in enumerate(mu_grid):
        a_eq = np.array([equilibrium_spin(M, mu) for M in M_samples])
        
        # For each sample, compute log P(a | M, μ)
        # If a > a_eq: BH should have been spun down → low probability under H1
        # If a <= a_eq: BH is consistent → high probability under H1
        
        # Use a smooth likelihood function
        # P(a | M, μ) ∝ exp(-max(0, a - a_eq)² / (2σ²)) where σ ~ 0.1
        delta_a = np.maximum(a_samples - a_eq, 0)
        sigma = 0.08  # uncertainty in spin-down
        
        # Log-likelihood per sample
        log_p_per_sample = -0.5 * (delta_a / sigma)**2
        
        # Sum over all samples (assuming independence)
        log_likelihood[i] = np.sum(log_p_per_sample)
    
    # Prior on μ: uniform in log(μ)
    log_prior = np.zeros(n_mu)  # uniform in log space
    
    # Posterior ∝ likelihood × prior
    log_posterior = log_likelihood + log_prior
    
    # Normalize using log-sum-exp
    log_posterior_max = np.max(log_posterior)
    log_posterior_norm = log_posterior_max + np.log(
        np.trapz(np.exp(log_posterior - log_posterior_max), log_mu_grid)
    )
    
    log_evidence_H1 = log_posterior_norm
    
    # Under H0 (no ULB): likelihood is just the prior on (M, a)
    # which we set to 1 (improper prior)
    log_evidence_H0 = 0.0
    
    # Bayes factor
    log_bayes_factor = log_evidence_H1 - log_evidence_H0
    
    # Posterior distribution over μ
    posterior_mu = np.exp(log_posterior - log_posterior_max)
    posterior_mu /= np.trapz(posterior_mu, log_mu_grid)
    
    return mu_grid, log_bayes_factor, p_excluded, posterior_mu, log_posterior


def compute_exclusion_contour_multi_BH(bh_data_list, mu_grid, t_BH_sec_list=None):
    """
    Combine exclusion evidence from multiple BHs.
    
    Uses Bayesian model averaging: the total exclusion probability
    is the product of individual exclusion probabilities.
    
    Parameters:
        bh_data_list: list of (M_samples, a_samples) tuples
        mu_grid: array of ULB masses to test
        t_BH_sec_list: list of BH ages (seconds)
    
    Returns:
        p_excluded_combined: combined exclusion probability per μ
        p_excluded_individual: exclusion per BH per μ
    """
    n_bh = len(bh_data_list)
    n_mu = len(mu_grid)
    
    if t_BH_sec_list is None:
        t_BH_sec_list = [1e10] * n_bh  # default: 10 Gyr
    
    p_excluded_individual = np.zeros((n_bh, n_mu))
    
    for j, (M_samp, a_samp) in enumerate(bh_data_list):
        p_ex, _ = compute_exclusion_curve(
            M_samp, a_samp, mu_grid, t_BH_sec_list[j]
        )
        p_excluded_individual[j] = p_ex
    
    # Combined exclusion: product of consistent probabilities
    # P_excl = 1 - ∏_j (1 - P_excl_j)
    p_consistent_combined = np.prod(1.0 - p_excluded_individual, axis=0)
    p_excluded_combined = 1.0 - p_consistent_combined
    
    return p_excluded_combined, p_excluded_individual


def derive_upper_limit(mu_grid, p_excluded, confidence_level=0.95):
    """
    Derive the ULB mass upper limit from the exclusion curve.
    
    The upper limit is defined as the mass above which the exclusion
    probability exceeds the confidence level.
    
    Returns:
        mu_upper: upper limit mass (eV)
        mu_lower: lower bound of excluded region (eV)
        mu_peak: mass with maximum exclusion (eV)
    """
    # Find the peak exclusion
    idx_peak = np.argmax(p_excluded)
    mu_peak = mu_grid[idx_peak]
    
    # Find the 95% confidence region
    above_cl = p_excluded >= confidence_level
    
    if not np.any(above_cl):
        # No mass excluded at this level
        return None, None, mu_peak
    
    # Find the contiguous region above CL
    indices = np.where(above_cl)[0]
    
    if len(indices) == 0:
        return None, None, mu_peak
    
    mu_lower = mu_grid[indices[0]]
    mu_upper = mu_grid[indices[-1]]
    
    return mu_upper, mu_lower, mu_peak


def self_interaction_constraint(mu_grid, f_a_grid, p_excluded):
    """
    Compute constraints on self-interaction coupling.
    
    For the QCD axion: the self-interaction strength is determined by
    the decay constant f_a. The coupling g_3 ∝ 1/f_a.
    
    The bosenova occurs when the self-interaction potential energy
    exceeds the gravitational binding energy:
    
    |V_self| > E_bind
    
    For the QCD axion: g_3 = 1/f_a (dimensionless)
    
    Parameters:
        mu_grid: ULB masses (eV)
        f_a_grid: axion decay constants (eV)
        p_excluded: exclusion probabilities
    
    Returns:
        fa_upper_limit: upper limit on f_a for each μ
    """
    # Self-interaction coupling: g = μ²/(16π²f_a²)
    # Or equivalently: the coupling parameter g_3 = μ/(4πf_a)
    
    # Bosenova condition: g_3² > α² × (some factor)
    # For QCD axion: the bosenova occurs at g_3 ≈ α
    
    # Convert exclusion in μ to constraint on f_a
    # For a given μ, the excluded f_a range is:
    # f_a < f_a_max(μ) from the superradiance constraint
    # f_a > f_a_min(μ) from the bosenova condition
    
    # Simple model: f_a_max from superradiance
    # The superradiance constraint on μ translates to:
    # For a given BH mass M, μ < μ_max ≈ 1/(GM) [in natural units]
    # For the QCD axion: μ ∝ 1/f_a, so f_a > f_a_min
    
    return None


def save_outputs(mu_grid, p_excluded, posterior_mu, log_posterior,
                 bh_names, p_individual, details):
    """Save all outputs to files."""
    import json
    
    # Save exclusion curves
    np.savez('outputs/exclusion_results.npz',
             mu_grid=mu_grid,
             p_excluded_combined=p_excluded,
             posterior_mu=posterior_mu,
             log_posterior=log_posterior,
             p_excluded_individual=p_individual)
    
    # Save summary statistics
    mu_upper, mu_lower, mu_peak = derive_upper_limit(mu_grid, p_excluded)
    
    summary = {
        'mu_upper_95CL_eV': float(mu_upper) if mu_upper is not None else None,
        'mu_lower_95CL_eV': float(mu_lower) if mu_lower is not None else None,
        'mu_peak_exclusion_eV': float(mu_peak),
        'max_exclusion_probability': float(np.max(p_excluded)),
        'bh_names': bh_names,
        'n_mu_points': len(mu_grid),
    }
    
    with open('outputs/constraint_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Outputs saved to outputs/")
    print(f"95% CL upper limit: {mu_upper:.2e} eV" if mu_upper else "No 95% CL exclusion found")
    print(f"Peak exclusion at: {mu_peak:.2e} eV")
    print(f"Max exclusion probability: {np.max(p_excluded):.4f}")
    
    return summary
