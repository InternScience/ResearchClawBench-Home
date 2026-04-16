"""
Bayesian Constraints on Ultralight Bosons from Black Hole Superradiance
========================================================================

This script implements a Bayesian statistical framework that translates the physics
of black hole superradiance into a probabilistic model, using full posterior
distributions of BH mass and spin measurements to derive upper limits on
ultralight boson (ULB) masses and self-interaction coupling strengths.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json
import os

# ============================================================
# Physical Constants
# ============================================================
G_N = 6.67430e-11       # m^3 kg^-1 s^-2
c = 2.99792458e8         # m/s
hbar = 1.054571817e-34   # J s
M_sun_kg = 1.98892e30    # kg
eV_to_J = 1.602176634e-19  # J/eV
M_Pl_eV = 1.2209e28      # Planck mass in eV (= 1.22e19 GeV)
yr_to_s = 365.25 * 24 * 3600  # seconds per year

# ============================================================
# Data Loading
# ============================================================
def load_samples(filepath):
    masses = []
    spins = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            masses.append(float(parts[0]))
            spins.append(float(parts[1]))
    return np.array(masses), np.array(spins)

m33_mass, m33_spin = load_samples('data/M33_X-7_samples.dat')
iras_mass, iras_spin = load_samples('data/IRAS_09149-6206_samples.dat')

print(f"M33 X-7: {len(m33_mass)} samples, M = {np.mean(m33_mass):.2f} +/- {np.std(m33_mass):.2f} Msun, a* = {np.mean(m33_spin):.3f} +/- {np.std(m33_spin):.3f}")
print(f"IRAS 09149-6206: {len(iras_mass)} samples, M = {np.mean(iras_mass):.2e} +/- {np.std(iras_mass):.2e} Msun, a* = {np.mean(iras_spin):.3f} +/- {np.std(iras_spin):.3f}")

# ============================================================
# Superradiance Physics
# ============================================================

def alpha_param(M_bh_msun, mu_eV):
    """Gravitational fine-structure constant: alpha = M*mu/M_Pl^2."""
    M_eV = M_bh_msun * M_sun_kg * c**2 / eV_to_J
    return M_eV * mu_eV / M_Pl_eV**2

def regge_trajectory(alpha, l=1, m=1):
    """
    Critical spin a*_crit for the Regge trajectory boundary.
    a*_crit = 2R/(1+R^2) where R = 2*alpha*(1-alpha^2/(2n^2))/m, n=l+1.
    """
    n = l + 1
    alpha_eff = np.minimum(alpha, np.sqrt(2 * n**2) * 0.99)
    R = 2 * alpha_eff * (1 - alpha_eff**2 / (2 * n**2)) / m
    R = np.clip(R, 0, 1)
    a_crit = 2 * R / (1 + R**2)
    return a_crit

def superradiance_timescale(M_bh_msun, a_star, mu_eV, l=1, m=1):
    """
    Superradiance instability e-folding timescale in years.
    Gamma_SR ~ alpha^(4l+5) * mu * delta_a / (24*(l+1)^(4l+5))
    """
    alpha = alpha_param(M_bh_msun, mu_eV)
    a_crit = regge_trajectory(alpha, l, m)
    delta_a = np.maximum(a_star - a_crit, 0)
    mu_J = mu_eV * eV_to_J
    mu_inv_s = mu_J / hbar
    gamma_rel = alpha**(4*l + 5) * delta_a / (24 * (l+1)**(4*l+5))
    gamma_abs = gamma_rel * mu_inv_s
    tau_s = np.where(gamma_abs > 0, 1.0 / gamma_abs, np.inf)
    tau_yr = tau_s / yr_to_s
    return tau_yr

def compute_exclusion_probability(M_samples, a_samples, mu_eV, l=1, m=1,
                                   tau_max_years=5e7):
    """
    Bayesian exclusion probability: fraction of posterior samples where
    superradiance would have spun down the BH, contradicting the observation.
    """
    alphas = alpha_param(M_samples, mu_eV)
    a_crits = regge_trajectory(alphas, l, m)
    spin_exceeded = a_samples > a_crits
    alpha_valid = (alphas > 0.005) & (alphas < 1.0)
    tau_sr = superradiance_timescale(M_samples, a_samples, mu_eV, l, m)
    fast_enough = tau_sr < tau_max_years
    excluded = spin_exceeded & alpha_valid & fast_enough
    return np.mean(excluded)

# ============================================================
# Compute Exclusion Probabilities
# ============================================================
mu_range_stellar = np.logspace(-13, -9.5, 1000)
mu_range_smbh = np.logspace(-21, -17, 1000)

print("\nComputing exclusion probabilities for M33 X-7...")
P_excl_m33_l1 = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=1, m=1)
                           for mu in mu_range_stellar])
P_excl_m33_l2 = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=2, m=2)
                           for mu in mu_range_stellar])

print("Computing exclusion probabilities for IRAS 09149-6206...")
P_excl_iras_l1 = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=1, m=1)
                            for mu in mu_range_smbh])
P_excl_iras_l2 = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=2, m=2)
                            for mu in mu_range_smbh])

def find_exclusion_limits(mu_range, P_excl, confidence=0.95):
    excluded_mask = P_excl >= confidence
    if not np.any(excluded_mask):
        return None, None
    mu_excluded = mu_range[excluded_mask]
    return float(mu_excluded[0]), float(mu_excluded[-1])

m33_l1_low, m33_l1_high = find_exclusion_limits(mu_range_stellar, P_excl_m33_l1)
m33_l2_low, m33_l2_high = find_exclusion_limits(mu_range_stellar, P_excl_m33_l2)
iras_l1_low, iras_l1_high = find_exclusion_limits(mu_range_smbh, P_excl_iras_l1)
iras_l2_low, iras_l2_high = find_exclusion_limits(mu_range_smbh, P_excl_iras_l2)

print(f"\n95% Exclusion Limits:")
if m33_l1_low: print(f"  M33 X-7 (l=m=1): [{m33_l1_low:.2e}, {m33_l1_high:.2e}] eV")
else: print(f"  M33 X-7 (l=m=1): No 95% exclusion")
if m33_l2_low: print(f"  M33 X-7 (l=m=2): [{m33_l2_low:.2e}, {m33_l2_high:.2e}] eV")
else: print(f"  M33 X-7 (l=m=2): No 95% exclusion")
if iras_l1_low: print(f"  IRAS 09149-6206 (l=m=1): [{iras_l1_low:.2e}, {iras_l1_high:.2e}] eV")
else: print(f"  IRAS 09149-6206 (l=m=1): No 95% exclusion")
if iras_l2_low: print(f"  IRAS 09149-6206 (l=m=2): [{iras_l2_low:.2e}, {iras_l2_high:.2e}] eV")
else: print(f"  IRAS 09149-6206 (l=m=2): No 95% exclusion")

# ============================================================
# Self-Interaction Constraints
# ============================================================

def compute_exclusion_with_self_interaction(M_samples, a_samples, mu_eV, f_a_GeV,
                                             l=1, m=1, tau_max_years=5e7):
    """
    Exclusion probability with self-interaction (bosenova) effects.
    When N_max > N_crit ~ (f_a/mu)^2, bosenova collapse shuts down superradiance.
    """
    alphas = alpha_param(M_samples, mu_eV)
    a_crits = regge_trajectory(alphas, l, m)
    delta_a = np.maximum(a_samples - a_crits, 0)
    M_eV = M_samples * M_sun_kg * c**2 / eV_to_J
    N_max = delta_a * alphas * (M_eV / mu_eV) / m
    f_a_eV = f_a_GeV * 1e9
    N_crit = (f_a_eV / mu_eV)**2
    shutdown = N_max > N_crit
    spin_exceeded = a_samples > a_crits
    alpha_valid = (alphas > 0.005) & (alphas < 1.0)
    tau_sr = superradiance_timescale(M_samples, a_samples, mu_eV, l, m)
    fast_enough = tau_sr < tau_max_years
    excluded = spin_exceeded & alpha_valid & fast_enough & ~shutdown
    return np.mean(excluded)

print("\nComputing self-interaction constraints for M33 X-7...")
mu_scan_fa = np.logspace(-12.5, -10.0, 60)
fa_scan = np.logspace(12, 19, 60)
P_excl_fa = np.zeros((len(mu_scan_fa), len(fa_scan)))
for i, mu in enumerate(mu_scan_fa):
    for j, fa in enumerate(fa_scan):
        P_excl_fa[i, j] = compute_exclusion_with_self_interaction(
            m33_mass, m33_spin, mu, fa, l=1, m=1)
print("M33 X-7 self-interaction grid done.")

print("Computing self-interaction constraints for IRAS 09149-6206...")
mu_scan_fa_iras = np.logspace(-20.0, -17.5, 60)
fa_scan_iras = np.logspace(12, 19, 60)
P_excl_fa_iras = np.zeros((len(mu_scan_fa_iras), len(fa_scan_iras)))
for i, mu in enumerate(mu_scan_fa_iras):
    for j, fa in enumerate(fa_scan_iras):
        P_excl_fa_iras[i, j] = compute_exclusion_with_self_interaction(
            iras_mass, iras_spin, mu, fa, l=1, m=1)
print("IRAS self-interaction grid done.")

# ============================================================
# Save Results
# ============================================================
results = {
    "M33_X7": {
        "n_samples": int(len(m33_mass)),
        "mass_mean_msun": float(np.mean(m33_mass)),
        "mass_std_msun": float(np.std(m33_mass)),
        "spin_mean": float(np.mean(m33_spin)),
        "spin_std": float(np.std(m33_spin)),
        "l1_exclusion_low_eV": m33_l1_low,
        "l1_exclusion_high_eV": m33_l1_high,
        "l2_exclusion_low_eV": m33_l2_low,
        "l2_exclusion_high_eV": m33_l2_high,
    },
    "IRAS_09149": {
        "n_samples": int(len(iras_mass)),
        "mass_mean_msun": float(np.mean(iras_mass)),
        "mass_std_msun": float(np.std(iras_mass)),
        "spin_mean": float(np.mean(iras_spin)),
        "spin_std": float(np.std(iras_spin)),
        "l1_exclusion_low_eV": iras_l1_low,
        "l1_exclusion_high_eV": iras_l1_high,
        "l2_exclusion_low_eV": iras_l2_low,
        "l2_exclusion_high_eV": iras_l2_high,
    }
}

with open('outputs/constraint_results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.savez('outputs/exclusion_probabilities.npz',
         mu_stellar=mu_range_stellar,
         mu_smbh=mu_range_smbh,
         P_m33_l1=P_excl_m33_l1,
         P_m33_l2=P_excl_m33_l2,
         P_iras_l1=P_excl_iras_l1,
         P_iras_l2=P_excl_iras_l2,
         mu_scan_fa=mu_scan_fa,
         fa_scan=fa_scan,
         P_excl_fa=P_excl_fa,
         mu_scan_fa_iras=mu_scan_fa_iras,
         fa_scan_iras=fa_scan_iras,
         P_excl_fa_iras=P_excl_fa_iras)

print("\nAll results saved to outputs/")
