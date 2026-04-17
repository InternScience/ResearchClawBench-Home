"""
Bayesian Framework for Constraining Ultralight Bosons via Black Hole Superradiance

This module implements:
1. Superradiance physics (critical spin curves, instability rates)
2. Bayesian statistical framework using posterior samples
3. Constraint derivation for boson masses and self-interaction couplings
"""

import numpy as np
from scipy.special import factorial, gamma as gamma_func
from scipy.integrate import quad
import json
import os

# ============================================================
# Physical Constants (natural units where ℏ = c = 1 where convenient)
# ============================================================
G_N = 6.674e-11        # m^3 kg^-1 s^-2
c_light = 2.998e8      # m/s
hbar = 1.055e-34       # J s
M_sun = 1.989e30       # kg
M_Pl = 1.221e19        # GeV (Planck mass)
eV_to_kg = 1.783e-36   # kg per eV/c^2
GeV_to_eV = 1e9
yr_to_s = 3.156e7      # seconds per year

# Gravitational radius: r_g = G M / c^2
def r_g(M_solar):
    """Gravitational radius in meters for BH mass in solar masses."""
    return G_N * M_solar * M_sun / c_light**2

def alpha_grav(mu_eV, M_solar):
    """
    Gravitational fine structure constant: α = G M μ / (ℏ c)
    where μ is the boson mass in eV/c^2 and M is the BH mass in solar masses.
    
    α = r_g * μ / ℏ = (G M / c^2) * (μ / (ℏ c))
    """
    rg = r_g(M_solar)  # meters
    mu_kg = mu_eV * eV_to_kg  # kg
    # α = r_g * μ * c / ℏ  (since μ in natural units is μc/ℏ in inverse meters)
    alpha = rg * mu_kg * c_light / hbar
    return alpha

def omega_plus(a_star):
    """
    Angular velocity of the BH horizon: ω_+ = a* / (2 r_g (1 + sqrt(1 - a*^2)))
    Returns ω_+ in units of 1/r_g.
    """
    r_plus_over_rg = 1.0 + np.sqrt(1.0 - a_star**2)
    return a_star / (2.0 * r_plus_over_rg)  # in units of r_g^{-1}

def critical_spin_lm(alpha, l, m):
    """
    Critical spin a*_crit for a given gravitational coupling alpha and mode (l,m).
    
    Superradiance condition: ω < m ω_+
    For hydrogen-like levels: ω ≈ μ (1 - α²/(2n̄²)) where n̄ = n + l + 1
    
    The fastest growing mode has n=0, so n̄ = l+1.
    At the boundary: μ = m ω_+
    → μ = m a* / (2 r_g (1 + sqrt(1-a*²)))
    → α = m a* / (2 (1 + sqrt(1-a*²)))  [since α = μ r_g]
    
    Solving for a*: 
    2α(1 + sqrt(1-a*²)) = m a*
    Let x = a*², then:
    2α(1 + sqrt(1-x)) = m sqrt(x)
    """
    if alpha <= 0:
        return 0.0
    
    # Solve: 2α(1 + sqrt(1-a*²)) = m * a*
    # Rearrange: 2α + 2α*sqrt(1-a*²) = m*a*
    # Square both sides after isolating sqrt:
    # 2α*sqrt(1-a*²) = m*a* - 2α
    # Need m*a* > 2α, i.e., a* > 2α/m
    # 4α²(1-a*²) = (m*a* - 2α)²
    # 4α² - 4α²a*² = m²a*² - 4mαa* + 4α²
    # -4α²a*² = m²a*² - 4mαa*
    # 0 = (m² + 4α²)a*² - 4mαa*
    # a*(m² + 4α²) = 4mα  (for a* ≠ 0)
    # a* = 4mα / (m² + 4α²)
    
    a_crit = 4.0 * m * alpha / (m**2 + 4.0 * alpha**2)
    return min(a_crit, 0.998)  # cap at near-extremal

def superradiance_rate_nlm(alpha, a_star, n, l, m, M_solar):
    """
    Superradiance rate Γ_{nlm} using the non-relativistic (Detweiler) formula.
    
    Γ_{lmn} = 2μ α^{4l+4} r_+(mω_+ - μ) C_{lmn}
    
    where C_{lmn} involves factorials and products.
    
    Returns rate in s^{-1}.
    """
    if alpha <= 0 or a_star <= 0:
        return 0.0
    
    n_bar = n + l + 1  # principal quantum number
    
    # BH parameters
    r_plus = 1.0 + np.sqrt(1.0 - a_star**2)  # in units of r_g
    w_plus = a_star / (2.0 * r_plus)  # in units of r_g^{-1}
    
    # Frequency ≈ μ in natural units, so ω ≈ α/r_g (in units of r_g^{-1})
    # Actually μ = α/r_g, so mω_+ - μ = (m*w_plus - alpha)/r_g ... no
    # Let's work in units of r_g^{-1}:
    # μ in units of r_g^{-1} is just α (since α = μ r_g)
    mu_rg = alpha  # μ in units of r_g^{-1}
    
    # mω_+ - μ in units of r_g^{-1}
    diff = m * w_plus - mu_rg
    
    if diff <= 0:
        return 0.0  # Not superradiant
    
    # C_{lmn} factor
    # C = 2^{4l+2} (2l+n+1)! / ((l+n+1)^{2l+4} n!) × (l! / ((2l)!(2l+1)!))^2 
    #     × prod_{j=1}^{l} [j² (1-a*²) + 4r_+²(mω_+ - μ)²]  ... wait, that's (mw+ - μ)
    
    # Let me use the simplified form for the dominant l=m mode with n=0
    # For l=m=1, n=0: n̄=2
    # Γ ∝ α^{4l+4} × (mω_+ - μ) × product terms
    
    # Full formula from Eq. 18:
    # Γ_{lmn} = 2μ α^{4l+4} r_+ (mw_+ - μ_a) C_{lmn}
    
    # Compute C_{lmn}
    try:
        factor1 = 2**(4*l + 2)
        factor2 = float(factorial(2*l + n + 1, exact=True))
        factor3 = float((l + n + 1)**(2*l + 4))
        factor4 = float(factorial(n, exact=True))
        factor5 = float(factorial(l, exact=True))
        factor6 = float(factorial(2*l, exact=True))
        factor7 = float(factorial(2*l + 1, exact=True))
        
        C = factor1 * factor2 / (factor3 * factor4) * (factor5 / (factor6 * factor7))**2
        
        # Product term
        prod = 1.0
        for j in range(1, l + 1):
            term = j**2 * (1.0 - a_star**2) + 4.0 * r_plus**2 * diff**2
            prod *= term
        
        C *= prod
    except (OverflowError, ValueError):
        return 0.0
    
    # Rate in units of r_g^{-1} (natural units)
    Gamma_natural = 2.0 * mu_rg * alpha**(4*l + 4) * r_plus * diff * C
    
    # Convert to s^{-1}: multiply by c/r_g
    rg_meters = r_g(M_solar)
    Gamma_si = Gamma_natural * c_light / rg_meters
    
    return Gamma_si

def instability_timescale(alpha, a_star, M_solar, l=1, m=1, n=0):
    """
    Compute the e-folding timescale for superradiance in seconds.
    Returns np.inf if not superradiant.
    """
    rate = superradiance_rate_nlm(alpha, a_star, n, l, m, M_solar)
    if rate <= 0:
        return np.inf
    return 1.0 / rate

def is_excluded_by_superradiance(mu_eV, M_solar, a_star, 
                                  max_age_yr=1e10, l_max=2):
    """
    Check if a BH with given mass and spin would be excluded by superradiance
    for a boson of mass mu_eV.
    
    A BH is "excluded" if:
    1. The superradiance instability timescale is shorter than the BH age
    2. The BH spin is above the critical spin (Regge trajectory)
    
    Returns True if the observed (M, a*) is inconsistent with the boson existing.
    """
    alpha = alpha_grav(mu_eV, M_solar)
    
    for l in range(1, l_max + 1):
        m = l  # fastest growing mode
        a_crit = critical_spin_lm(alpha, l, m)
        
        if a_star > a_crit and a_crit > 0:
            # Check if instability timescale is short enough
            tau = instability_timescale(alpha, a_star, M_solar, l=l, m=m, n=0)
            tau_yr = tau / yr_to_s
            
            if tau_yr < max_age_yr:
                return True
    
    return False

def bayesian_exclusion_probability(mu_eV, M_samples, a_samples, 
                                    max_age_yr=1e10, l_max=2):
    """
    Compute the Bayesian exclusion probability for a given boson mass.
    
    P(excluded | μ) = fraction of posterior samples that are inconsistent
    with the boson existing.
    
    This is the key Bayesian quantity: if P(excluded | μ) is high,
    the boson mass μ is disfavored by the data.
    """
    N = len(M_samples)
    n_excluded = 0
    
    for i in range(N):
        if is_excluded_by_superradiance(mu_eV, M_samples[i], a_samples[i],
                                         max_age_yr=max_age_yr, l_max=l_max):
            n_excluded += 1
    
    return n_excluded / N

def compute_exclusion_curve(mu_range_eV, M_samples, a_samples,
                            max_age_yr=1e10, l_max=2):
    """
    Compute exclusion probability as a function of boson mass.
    """
    probs = []
    for mu in mu_range_eV:
        p = bayesian_exclusion_probability(mu, M_samples, a_samples,
                                            max_age_yr=max_age_yr, l_max=l_max)
        probs.append(p)
    return np.array(probs)

def bosenova_fa_limit(mu_eV, M_solar, a_star, l=1):
    """
    Compute the upper limit on the decay constant f_a from the Bosenova criterion.
    
    From Arvanitaki et al. 2011, Eq. 48:
    M_cloud/M_BH > 2 l^4 / α² × (f_a / M_Pl)²
    
    The maximum cloud mass before Bosenova is:
    M_cloud_max / M_BH = 2 l^4 / α² × (f_a / M_Pl)²
    
    For superradiance to spin down the BH, the cloud must extract
    ΔJ ~ M_BH * r_g * Δa* worth of angular momentum.
    The cloud mass needed is roughly M_cloud ~ ΔJ / (m * r_cloud * ω)
    
    If Bosenova happens before enough spin is extracted, superradiance
    is interrupted. The constraint is:
    f_a < M_Pl * α / (l² √2) × √(M_cloud_needed / M_BH)
    
    For a rough estimate, the spin change needed is Δa* ~ a* - a*_crit,
    and M_cloud ~ Δa* * M_BH * α / m (in natural units).
    
    Returns f_a upper limit in GeV.
    """
    alpha = alpha_grav(mu_eV, M_solar)
    
    if alpha <= 0:
        return np.inf
    
    a_crit = critical_spin_lm(alpha, l, l)
    
    if a_star <= a_crit:
        return np.inf  # No constraint
    
    # Fraction of BH mass that needs to be in the cloud
    # ΔM/M ~ (a* - a_crit) * α / m for the l=m mode
    delta_a = a_star - a_crit
    frac_needed = delta_a * alpha / l  # rough estimate
    
    # Bosenova limit: frac_needed < 2 l^4 / α² × (f_a / M_Pl)²
    # → f_a < M_Pl × α / (l²) × √(frac_needed / 2)
    # → f_a < M_Pl × √(frac_needed * α² / (2 l^4))
    
    fa_over_mpl = np.sqrt(frac_needed * alpha**2 / (2.0 * l**4))
    fa_GeV = fa_over_mpl * M_Pl  # in GeV
    
    return fa_GeV

def load_samples(filepath):
    """Load posterior samples from data file."""
    data = np.loadtxt(filepath)
    M_samples = data[:, 0]  # BH mass in M_sun
    a_samples = data[:, 1]  # dimensionless spin
    return M_samples, a_samples

if __name__ == "__main__":
    # Test with M33 X-7
    M_test, a_test = load_samples("data/M33_X-7_samples.dat")
    print(f"M33 X-7: {len(M_test)} samples")
    print(f"  Mass range: {M_test.min():.1f} - {M_test.max():.1f} M_sun")
    print(f"  Spin range: {a_test.min():.3f} - {a_test.max():.3f}")
    
    # Test alpha calculation
    mu_test = 1e-12  # eV
    M_test_val = 15.0  # M_sun
    alpha_test = alpha_grav(mu_test, M_test_val)
    print(f"\nTest: α = {alpha_test:.4f} for μ = {mu_test:.1e} eV, M = {M_test_val} M_sun")
    
    # Test critical spin
    a_crit = critical_spin_lm(alpha_test, 1, 1)
    print(f"Critical spin (l=m=1): a*_crit = {a_crit:.4f}")
    
    # Test instability timescale
    tau = instability_timescale(alpha_test, 0.8, M_test_val, l=1, m=1, n=0)
    print(f"Instability timescale: {tau/yr_to_s:.2e} yr")
