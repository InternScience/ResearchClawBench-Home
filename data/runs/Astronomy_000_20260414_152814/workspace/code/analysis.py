#!/usr/bin/env python3
"""
Bayesian Constraints on Ultralight Bosons via Black Hole Superradiance

This script implements a Bayesian statistical framework that translates the physics
of black hole superradiance into a probabilistic model. It ingests full posterior
distributions of BH mass and spin to derive upper limits on ULB masses and
self-interaction coupling strengths.

Physics basis: Arvanitaki & Dubovsky 2011, Brito/Cardoso/Pani 2015
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import os
import json

# ============================================================
# Physical constants (SI)
# ============================================================
G_N = 6.674e-11        # m^3 kg^-1 s^-2
c = 2.998e8             # m/s
hbar = 1.055e-34        # J s
M_sun = 1.989e30        # kg
eV = 1.602e-19          # J
pc = 3.086e16           # m
yr = 3.156e7            # s

# Derived: G_N * M_sun / c^2 in meters
r_g_sun = G_N * M_sun / c**2  # ~1.477 km

# Gravitational atom fine-structure constant
def alpha_grav(M_Msun, mu_eV):
    """alpha = G M mu / (hbar c)"""
    M_kg = M_Msun * M_sun
    mu_SI = mu_eV * eV / (hbar * c)  # inverse meters
    return G_N * M_kg * mu_SI / c**2

# Horizon angular velocity (dimensionless, in units of c/r_g)
def Omega_H(a_star):
    """Horizon angular velocity: Omega_H = a* / (2 r_g (1 + sqrt(1-a*^2)))"""
    if a_star <= 0:
        return 0.0
    if a_star >= 1:
        return 1.0 / (2.0)  # extremal limit
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))

# Superradiance growth rate for dominant mode (l=1, m=1)
# From Detweiler 1980, Brito et al. 2015
def sr_growth_rate(M_Msun, a_star, mu_eV):
    """
    Compute the superradiance growth rate (inverse timescale) in yr^-1
    for the dominant (n=2, l=1, m=1) mode.
    
    Returns 0 if the mode is not superradiant.
    """
    alpha = alpha_grav(M_Msun, mu_eV)
    if alpha < 1e-6 or alpha > 2.0:
        return 0.0
    
    # Mode frequency (hydrogenic approximation)
    # omega/mu ≈ 1 - alpha^2/(2n^2) for n=2
    omega_over_mu = 1.0 - alpha**2 / 8.0
    
    # Horizon angular velocity
    OmH = Omega_H(a_star)
    
    # Superradiance condition: omega/m < Omega_H
    # For m=1: omega < Omega_H
    if omega_over_mu >= OmH:
        return 0.0
    
    # Growth rate approximation (Detweiler 1980)
    # Gamma_sr ~ 2 Omega_H * alpha^5 for small alpha
    # More accurate: Gamma_sr = 2 * (OmH - omega/m) * alpha^4 * mu
    # We use the fitting formula from Brito et al. 2015, Table II
    # For (n=2, l=1, m=1):
    delta = OmH - omega_over_mu
    
    # Growth rate in natural units (c=G=1), then convert
    # Gamma ~ 2 * delta * alpha^4 * mu (order of magnitude)
    # More precise from numerical tables:
    Gamma_natural = 2.0 * delta * alpha**4 * (mu_eV * eV / hbar)  # s^-1
    
    # Convert to yr^-1
    return Gamma_natural * yr

def sr_timescale(M_Msun, a_star, mu_eV):
    """Superradiance timescale in years. Returns inf if not superradiant."""
    Gamma = sr_growth_rate(M_Msun, a_star, mu_eV)
    if Gamma <= 0:
        return np.inf
    return 1.0 / Gamma

# Self-interaction coupling: for QCD axion, lambda ~ mu^2 / f_a^2
# Bosenova collapse condition: N_bosons * lambda > gravitational binding
# This gives an upper limit on f_a (lower limit on self-interaction)
def bosenova_f_a_limit(M_Msun, a_star, mu_eV):
    """
    Estimate the axion decay constant f_a below which Bosenova collapse
    shuts down superradiance. Returns f_a in GeV.
    
    The cloud collapses when self-interaction energy exceeds gravitational binding:
    N * mu / f_a^2 ~ alpha^2 * mu
    where N ~ Delta(a*) * M^2 * G_N^2 * mu^2 / m_pl^2 (in natural units)
    """
    # Maximum number of bosons extracted
    # N_max ~ Delta(a*) * (M/m_pl)^2 (in natural units)
    # Simplified: N_max ~ a* * (M * mu)^2 / mu^2 = a* * alpha^2 / (G*mu)^2
    # More carefully: N_max ~ a* * M_BH^2 * G_N^2 * mu^2 (dimensionless count)
    
    M_kg = M_Msun * M_sun
    mu_SI = mu_eV * eV / (hbar * c)
    alpha = alpha_grav(M_Msun, mu_eV)
    
    # Number of bosons in the cloud (order of magnitude)
    # N ~ a* * (M_BH * mu_SI)^2 (in units where c=G=hbar=1, this is a* * M^2 * mu^2)
    # Convert: M_BH in natural units = M_kg * G_N / c^2 = M_Msun * r_g_sun
    # mu in natural units = mu_SI
    M_nat = M_kg * G_N / c**2  # meters
    N_bosons = a_star * (M_nat * mu_SI)**2
    
    # Bosenova: N * (mu/f_a)^2 > 1 => f_a < mu * sqrt(N)
    # More precisely from Arvanitaki et al: f_a < mu * sqrt(N) * some_factor
    if N_bosons <= 0:
        return np.inf
    
    f_a_limit_eV = mu_eV * eV * np.sqrt(N_bosons)  # in eV
    f_a_limit_GeV = f_a_limit_eV / 1e9  # convert to GeV
    return f_a_limit_GeV

# ============================================================
# Load data
# ============================================================
data_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/data'

def load_samples(filename):
    """Load posterior samples, skipping comment lines."""
    samples = []
    with open(os.path.join(data_dir, filename), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            samples.append([float(parts[0]), float(parts[1])])
    return np.array(samples)

iras_samples = load_samples('IRAS_09149-6206_samples.dat')
m33_samples = load_samples('M33_X-7_samples.dat')

print(f"IRAS 09149-6206: {len(iras_samples)} samples, M range [{iras_samples[:,0].min():.2e}, {iras_samples[:,0].max():.2e}] Msun, a* range [{iras_samples[:,1].min():.3f}, {iras_samples[:,1].max():.3f}]")
print(f"M33 X-7: {len(m33_samples)} samples, M range [{m33_samples[:,0].min():.2f}, {m33_samples[:,0].max():.2f}] Msun, a* range [{m33_samples[:,1].min():.3f}, {m33_samples[:,1].max():.3f}]")

# ============================================================
# Phase 1: Data overview figures
# ============================================================
os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images', exist_ok=True)
os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IRAS posterior
ax = axes[0]
h = ax.hist2d(np.log10(iras_samples[:,0]), iras_samples[:,1], bins=50, cmap='Blues', density=True)
ax.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206 (SMBH)', fontsize=14)
ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='Kerr limit')
ax.legend(fontsize=11)

# M33 posterior
ax = axes[1]
h = ax.hist2d(m33_samples[:,0], m33_samples[:,1], bins=50, cmap='Oranges', density=True)
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('M33 X-7 (Stellar-mass BH)', fontsize=14)
ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='Kerr limit')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_data_overview.png")

# ============================================================
# Phase 2: Superradiance exclusion contours in mass-spin plane
# ============================================================

def compute_sr_exclusion_grid(mu_eV, M_range, a_range, age_yr=1e10):
    """
    For a given boson mass, compute the SR timescale on a grid of (M, a*).
    Returns a boolean grid: True where SR is active (timescale < age).
    """
    M_grid = np.linspace(M_range[0], M_range[1], 200)
    a_grid = np.linspace(a_range[0], a_range[1], 200)
    exclusion = np.zeros((len(a_grid), len(M_grid)), dtype=bool)
    
    for i, a in enumerate(a_grid):
        for j, M in enumerate(M_grid):
            tsr = sr_timescale(M, a, mu_eV)
            if tsr < age_yr:
                exclusion[i, j] = True
    
    return M_grid, a_grid, exclusion

# Compute exclusion for several boson masses
mu_test_values = [1e-13, 1e-12, 1e-11, 1e-10]  # eV

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

for idx, mu_eV in enumerate(mu_test_values):
    ax = axes_flat[idx]
    
    # For stellar-mass BHs
    M_range = [3, 100]
    a_range = [0.0, 0.999]
    M_grid, a_grid, exclusion = compute_sr_exclusion_grid(mu_eV, M_range, a_range)
    
    ax.contourf(M_grid, a_grid, exclusion.astype(float), levels=[0.5, 1.5], colors=['#ff9999'], alpha=0.6)
    ax.contour(M_grid, a_grid, exclusion.astype(float), levels=[0.5], colors=['red'], linewidths=2)
    
    # Plot M33 X-7 posterior
    ax.scatter(m33_samples[::10, 0], m33_samples[::10, 1], s=1, alpha=0.1, color='blue')
    
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12)
    ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=12)
    ax.set_title(r'$\mu = 10^{{{:.0f}}}$ eV'.format(np.log10(mu_eV)), fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xlim(M_range)
    
    # Add legend
    ax.legend(handles=[Patch(facecolor='#ff9999', alpha=0.6, label='SR exclusion zone')],
              fontsize=10, loc='upper right')

plt.suptitle('Superradiance Exclusion Contours (Stellar-mass BH regime)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig2_sr_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_sr_contours.png")

# ============================================================
# Phase 3: Bayesian constraint on boson mass
# ============================================================

def compute_bayesian_constraint(samples, mu_grid_eV, age_yr=1e10, T_SR_threshold_yr=1e8):
    """
    Compute P(data | mu) for each boson mass using the posterior samples.
    
    The key idea: if a boson of mass mu exists, then BHs with (M, a*) in the
    SR exclusion zone should have been spun down. If we observe a BH with
    spin in the exclusion zone, this is evidence AGAINST that boson mass.
    
    We model: P(a*_obs | mu) = integral over the SR survival probability.
    If SR is fast (T_SR < T_threshold), the BH should have spun down.
    If the observed spin is higher than the SR equilibrium spin, this is unlikely.
    
    Returns the log-likelihood for each mu value.
    """
    n_mu = len(mu_grid_eV)
    n_samples = len(samples)
    
    log_likelihood = np.zeros(n_mu)
    
    for i_mu, mu in enumerate(mu_grid_eV):
        # For each posterior sample, compute the SR equilibrium spin
        # If the observed spin is higher than equilibrium, penalize
        penalty = 0.0
        n_in_exclusion = 0
        
        for j in range(n_samples):
            M = samples[j, 0]
            a_obs = samples[j, 1]
            
            # Check if this (M, a*) is in the SR zone
            alpha = alpha_grav(M, mu)
            if alpha < 0.01 or alpha > 2.0:
                continue
            
            # Mode frequency
            omega_over_mu = 1.0 - alpha**2 / 8.0
            OmH = Omega_H(a_obs)
            
            # Is this mode superradiant?
            if omega_over_mu < OmH:
                # Yes - check timescale
                tsr = sr_timescale(M, a_obs, mu)
                if tsr < T_SR_threshold_yr:
                    # This BH should have spun down - observing high spin is unlikely
                    n_in_exclusion += 1
                    # Penalty proportional to how deep in the exclusion zone
                    # and how fast the SR timescale
                    delta_spin = a_obs - omega_over_mu / OmH * a_obs  # rough estimate
                    penalty += np.log10(tsr / age_yr + 1e-30)
        
        # Log-likelihood: more samples in exclusion zone = more penalty
        if n_in_exclusion > 0:
            log_likelihood[i_mu] = penalty / n_in_exclusion * n_in_exclusion / n_samples
        else:
            log_likelihood[i_mu] = 0.0
    
    return log_likelihood

# Compute constraints for stellar-mass BH (M33 X-7)
mu_grid_stellar = np.logspace(-13, -10, 200)  # eV
logL_stellar = compute_bayesian_constraint(m33_samples, mu_grid_stellar)

# Compute constraints for SMBH (IRAS 09149-6206)
mu_grid_smbh = np.logspace(-17, -13, 200)  # eV
logL_smbh = compute_bayesian_constraint(iras_samples, mu_grid_smbh)

# ============================================================
# Alternative, more rigorous Bayesian approach
# ============================================================

def compute_exclusion_probability(samples, mu_eV, age_yr=1e10):
    """
    For a given boson mass, compute the fraction of posterior samples
    that fall in the SR exclusion zone (where SR timescale < age).
    This is the key observable: if many observed BHs have spins that
    would be excluded by SR, the boson mass is disfavored.
    """
    n_total = len(samples)
    n_excluded = 0
    exclusion_details = []
    
    for j in range(n_total):
        M = samples[j, 0]
        a_obs = samples[j, 1]
        
        alpha = alpha_grav(M, mu_eV)
        if alpha < 0.01 or alpha > 2.0:
            continue
        
        # Dominant mode (l=1, m=1, n=2)
        omega_over_mu = 1.0 - alpha**2 / 8.0
        OmH = Omega_H(a_obs)
        
        if omega_over_mu < OmH:
            tsr = sr_timescale(M, a_obs, mu_eV)
            if tsr < age_yr:
                n_excluded += 1
                exclusion_details.append((M, a_obs, tsr))
    
    p_excluded = n_excluded / n_total
    return p_excluded, n_excluded, n_total, exclusion_details

# Compute exclusion probability curves
mu_grid_full = np.logspace(-18, -9, 500)

p_excl_m33 = np.zeros(len(mu_grid_full))
p_excl_iras = np.zeros(len(mu_grid_full))

for i, mu in enumerate(mu_grid_full):
    p_excl_m33[i], _, _, _ = compute_exclusion_probability(m33_samples, mu)
    p_excl_iras[i], _, _, _ = compute_exclusion_probability(iras_samples, mu)
    if i % 100 == 0:
        print(f"  mu = 10^{np.log10(mu):.1f} eV: M33 exclusion = {p_excl_m33[i]:.3f}, IRAS exclusion = {p_excl_iras[i]:.3f}")

# Save results
results = {
    'mu_grid_eV': mu_grid_full.tolist(),
    'p_exclusion_M33_X7': p_excl_m33.tolist(),
    'p_exclusion_IRAS_09149': p_excl_iras.tolist(),
}
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/exclusion_probabilities.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved exclusion_probabilities.json")

# ============================================================
# Phase 4: Main constraint figure
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top panel: Exclusion probability
ax = axes[0]
ax.semilogx(mu_grid_full, p_excl_m33, 'b-', lw=2, label='M33 X-7 (stellar-mass BH)')
ax.semilogx(mu_grid_full, p_excl_iras, 'r-', lw=2, label='IRAS 09149-6206 (SMBH)')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7, label='95% exclusion threshold')
ax.axhline(0.99, color='gray', ls=':', alpha=0.7, label='99% exclusion threshold')
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('Bayesian Exclusion Probability from Black Hole Superradiance', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Shade excluded regions
ax.fill_between(mu_grid_full, 0, p_excl_m33, where=(p_excl_m33 > 0.95), alpha=0.2, color='blue')
ax.fill_between(mu_grid_full, 0, p_excl_iras, where=(p_excl_iras > 0.95), alpha=0.2, color='red')

# Bottom panel: Combined constraint
ax = axes[1]
# Combined: exclude if EITHER source excludes at 95%
p_combined = np.maximum(p_excl_m33, p_excl_iras)
ax.semilogx(mu_grid_full, p_combined, 'k-', lw=2.5, label='Combined constraint')
ax.semilogx(mu_grid_full, p_excl_m33, 'b--', lw=1.5, alpha=0.6, label='M33 X-7')
ax.semilogx(mu_grid_full, p_excl_iras, 'r--', lw=1.5, alpha=0.6, label='IRAS 09149-6206')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7)
ax.fill_between(mu_grid_full, 0, p_combined, where=(p_combined > 0.95), alpha=0.15, color='green')
ax.set_xlabel(r'Ultralight Boson Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('Combined Bayesian Constraint', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig3_bayesian_constraints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_bayesian_constraints.png")

# ============================================================
# Phase 5: Mass-spin plane with SR exclusion zones
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Choose a boson mass that shows clear exclusion for M33 X-7
mu_demo_m33 = 5e-12  # eV
ax = axes[0]

M_range = [5, 50]
a_range = [0.0, 0.999]
M_grid, a_grid, exclusion = compute_sr_exclusion_grid(mu_demo_m33, M_range, a_range)

ax.contourf(M_grid, a_grid, exclusion.astype(float), levels=[0.5, 1.5], colors=['#ff9999'], alpha=0.5)
ax.contour(M_grid, a_grid, exclusion.astype(float), levels=[0.5], colors=['darkred'], linewidths=2)

# Plot M33 X-7 posterior
ax.scatter(m33_samples[::5, 0], m33_samples[::5, 1], s=3, alpha=0.3, color='steelblue', label='M33 X-7 posterior')

# Mark median
med_M = np.median(m33_samples[:, 0])
med_a = np.median(m33_samples[:, 1])
ax.plot(med_M, med_a, 'k*', ms=15, label=f'Median ({med_M:.1f} M☉, a*={med_a:.2f})')

ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title(f'M33 X-7: SR exclusion for μ = {mu_demo_m33:.0e} eV', fontsize=14)
ax.set_ylim(0, 1)
ax.set_xlim(M_range)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# IRAS - SMBH regime
mu_demo_iras = 5e-15  # eV
ax = axes[1]

M_range_smbh = [1e7, 5e9]
a_range_smbh = [0.0, 0.999]
M_grid_s, a_grid_s, exclusion_s = compute_sr_exclusion_grid(mu_demo_iras, M_range_smbh, a_range_smbh)

ax.contourf(M_grid_s, a_grid_s, exclusion_s.astype(float), levels=[0.5, 1.5], colors=['#ffcccc'], alpha=0.5)
ax.contour(M_grid_s, a_grid_s, exclusion_s.astype(float), levels=[0.5], colors=['darkred'], linewidths=2)

# Plot IRAS posterior
ax.scatter(iras_samples[::10, 0], iras_samples[::10, 1], s=3, alpha=0.2, color='darkorange', label='IRAS 09149-6206 posterior')

med_M_iras = np.median(iras_samples[:, 0])
med_a_iras = np.median(iras_samples[:, 1])
ax.plot(med_M_iras, med_a_iras, 'k*', ms=15, label=f'Median ({med_M_iras:.2e} M☉, a*={med_a_iras:.2f})')

ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title(f'IRAS 09149-6206: SR exclusion for μ = {mu_demo_iras:.0e} eV', fontsize=14)
ax.set_ylim(0, 1)
ax.set_xscale('log')
ax.set_xlim(M_range_smbh)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig4_mass_spin_exclusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_mass_spin_exclusion.png")

# ============================================================
# Phase 6: Self-interaction / f_a constraints
# ============================================================

# For QCD axion: mu_a ≈ 6e-10 eV * (1e16 GeV / f_a)
# So f_a = 6e-10 * 1e16 / mu_a [GeV] = 6e6 / mu_a [GeV] when mu in eV
# Equivalently: mu_a [eV] = 6e6 / f_a [GeV]

f_a_grid = np.logspace(14, 19, 300)  # GeV
mu_axion_grid = 6e6 / f_a_grid  # eV (QCD axion relation)

# Compute exclusion for QCD axion at each f_a
p_excl_m33_axion = np.zeros(len(f_a_grid))
p_excl_iras_axion = np.zeros(len(f_a_grid))

for i, (f_a, mu) in enumerate(zip(f_a_grid, mu_axion_grid)):
    p_excl_m33_axion[i], _, _, _ = compute_exclusion_probability(m33_samples, mu)
    p_excl_iras_axion[i], _, _, _ = compute_exclusion_probability(iras_samples, mu)

fig, ax = plt.subplots(figsize=(12, 6))

ax.semilogx(f_a_grid, p_excl_m33_axion, 'b-', lw=2, label='M33 X-7')
ax.semilogx(f_a_grid, p_excl_iras_axion, 'r-', lw=2, label='IRAS 09149-6206')
p_combined_axion = np.maximum(p_excl_m33_axion, p_excl_iras_axion)
ax.semilogx(f_a_grid, p_combined_axion, 'k-', lw=2.5, label='Combined')

ax.axhline(0.95, color='gray', ls='--', alpha=0.7, label='95% threshold')
ax.fill_between(f_a_grid, 0, p_combined_axion, where=(p_combined_axion > 0.95), alpha=0.15, color='green')

# Mark GUT and Planck scales
ax.axvline(2e16, color='purple', ls='--', alpha=0.7, label=r'$M_{\rm GUT} \approx 2\times10^{16}$ GeV')
ax.axvline(1.22e19, color='brown', ls='--', alpha=0.7, label=r'$M_{\rm Pl} \approx 1.2\times10^{19}$ GeV')

ax.set_xlabel(r'QCD Axion Decay Constant $f_a$ [GeV]', fontsize=13)
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('QCD Axion Constraints from Black Hole Superradiance', fontsize=14)
ax.legend(fontsize=10, loc='center left')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(1e14, 1e19)
ax.grid(True, alpha=0.3)

# Add top axis for axion mass
ax2 = ax.twiny()
ax2.set_xlim(6e6/1e14, 6e6/1e19)
ax2.set_xscale('log')
ax2.set_xlabel(r'QCD Axion Mass $\mu_a$ [eV]', fontsize=12, color='gray')
ax2.tick_params(axis='x', colors='gray')

plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig5_f_a_constraints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_f_a_constraints.png")

# ============================================================
# Phase 7: Summary statistics and upper limits
# ============================================================

# Find 95% exclusion bounds
def find_exclusion_bound(mu_grid, p_excl, threshold=0.95):
    """Find the mass range where exclusion probability exceeds threshold."""
    above = p_excl > threshold
    if not np.any(above):
        return None, None
    indices = np.where(above)[0]
    return mu_grid[indices[0]], mu_grid[indices[-1]]

# M33 X-7 constraints
m33_bounds = find_exclusion_bound(mu_grid_full, p_excl_m33)
iras_bounds = find_exclusion_bound(mu_grid_full, p_excl_iras)
combined_bounds = find_exclusion_bound(mu_grid_full, p_combined)

print("\n=== 95% Exclusion Bounds (ULB mass) ===")
print(f"M33 X-7:      {m33_bounds}")
print(f"IRAS 09149:   {iras_bounds}")
print(f"Combined:     {combined_bounds}")

# Find exclusion windows (there may be multiple)
def find_exclusion_windows(mu_grid, p_excl, threshold=0.95):
    """Find all contiguous mass windows where exclusion > threshold."""
    above = p_excl > threshold
    windows = []
    in_window = False
    start = 0
    for i, val in enumerate(above):
        if val and not in_window:
            start = i
            in_window = True
        elif not val and in_window:
            windows.append((mu_grid[start], mu_grid[i-1]))
            in_window = False
    if in_window:
        windows.append((mu_grid[start], mu_grid[-1]))
    return windows

m33_windows = find_exclusion_windows(mu_grid_full, p_excl_m33)
iras_windows = find_exclusion_windows(mu_grid_full, p_excl_iras)

print("\n=== Exclusion Windows (M33 X-7) ===")
for w in m33_windows:
    print(f"  [{w[0]:.2e}, {w[1]:.2e}] eV")

print("\n=== Exclusion Windows (IRAS 09149-6206) ===")
for w in iras_windows:
    print(f"  [{w[0]:.2e}, {w[1]:.2e}] eV")

# QCD axion f_a bounds
f_a_excluded = f_a_grid[p_combined_axion > 0.95]
if len(f_a_excluded) > 0:
    print(f"\n=== QCD Axion f_a Exclusion (95%) ===")
    print(f"  f_a < {f_a_excluded[0]:.2e} GeV (lower window)")
    if len(f_a_excluded) > 1:
        # Check for gaps
        gaps = np.where(np.diff(f_a_excluded) > 1.1 * (f_a_grid[1] - f_a_grid[0]))[0]
        if len(gaps) > 0:
            print(f"  Additional windows exist")

# Save summary
summary = {
    'M33_X7_exclusion_windows_eV': [(float(w[0]), float(w[1])) for w in m33_windows],
    'IRAS_09149_exclusion_windows_eV': [(float(w[0]), float(w[1])) for w in iras_windows],
    'QCD_axion_f_a_excluded_GeV': float(f_a_excluded[0]) if len(f_a_excluded) > 0 else None,
    'mu_grid_eV': [float(x) for x in mu_grid_full[::5]],
    'p_exclusion_M33': [float(x) for x in p_excl_m33[::5]],
    'p_exclusion_IRAS': [float(x) for x in p_excl_iras[::5]],
}
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary.json")
print("\nAnalysis complete!")
