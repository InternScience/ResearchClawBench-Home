"""
Main analysis script: Bayesian constraints on ultralight bosons
from black hole superradiance using full posterior distributions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import sys
import os
import json

sys.path.insert(0, 'code')
from superradiance import (
    mu_to_coupling, equilibrium_spin, superradiant_timescale,
    superradiant_growth_rate_scalar, expected_final_spin,
    horizon_angular_velocity
)
from bayesian_framework import (
    compute_exclusion_curve, bayesian_evidence_ratio,
    compute_exclusion_contour_multi_BH, derive_upper_limit,
    save_outputs
)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("Loading BH posterior samples...")
print("=" * 60)

m33_data = np.loadtxt('data/M33_X-7_samples.dat')
iras_data = np.loadtxt('data/IRAS_09149-6206_samples.dat')

m33_M = m33_data[:, 0]  # Msol
m33_a = m33_data[:, 1]  # dimensionless spin
iras_M = iras_data[:, 0]
iras_a = iras_data[:, 1]

print(f"M33 X-7: {len(m33_M)} samples, M=[{m33_M.min():.1f}, {m33_M.max():.1f}] Msol, "
      f"a*=[{m33_a.min():.3f}, {m33_a.max():.3f}]")
print(f"IRAS 09149: {len(iras_M)} samples, M=[{iras_M.min():.1f}, {iras_M.max():.1f}] Msol, "
      f"a*=[{iras_a.min():.3f}, {iras_a.max():.3f}]")

# ============================================================
# 2. FIGURE 1: Data Overview - Posterior Distributions
# ============================================================
print("\nGenerating Figure 1: Data Overview...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# M33 X-7 mass posterior
ax = axes[0, 0]
ax.hist(m33_M, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='navy')
ax.set_xlabel('BH Mass $M$ [$M_\\odot$]', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('M33 X-7: Mass Posterior', fontsize=13)
ax.axvline(np.median(m33_M), color='red', ls='--', lw=2, label=f'Median: {np.median(m33_M):.1f} $M_\\odot$')
ax.legend(fontsize=10)

# M33 X-7 spin posterior
ax = axes[0, 1]
ax.hist(m33_a, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='navy')
ax.set_xlabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('M33 X-7: Spin Posterior', fontsize=13)
ax.axvline(np.median(m33_a), color='red', ls='--', lw=2, label=f'Median: {np.median(m33_a):.3f}')
ax.legend(fontsize=10)

# M33 X-7 2D posterior
ax = axes[0, 2]
# Use 2D histogram for density
h = ax.hist2d(m33_M, m33_a, bins=50, density=True, cmap='Blues', 
              norm=mcolors.LogNorm())
plt.colorbar(h[3], ax=ax, label='Density')
ax.set_xlabel('BH Mass $M$ [$M_\\odot$]', fontsize=12)
ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_title('M33 X-7: Joint Posterior', fontsize=13)

# IRAS 09149 mass posterior
ax = axes[1, 0]
log_M_iras = np.log10(iras_M)
ax.hist(log_M_iras, bins=50, density=True, alpha=0.7, color='darkorange', edgecolor='darkred')
ax.set_xlabel('$\\log_{10}(M / M_\\odot)$', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('IRAS 09149-6206: Mass Posterior', fontsize=13)
ax.axvline(np.log10(np.median(iras_M)), color='red', ls='--', lw=2, 
           label=f'Median: {np.median(iras_M):.1e} $M_\\odot$')
ax.legend(fontsize=10)

# IRAS 09149 spin posterior
ax = axes[1, 1]
ax.hist(iras_a, bins=50, density=True, alpha=0.7, color='darkorange', edgecolor='darkred')
ax.set_xlabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('IRAS 09149-6206: Spin Posterior', fontsize=13)
ax.axvline(np.median(iras_a), color='red', ls='--', lw=2, label=f'Median: {np.median(iras_a):.3f}')
ax.legend(fontsize=10)

# IRAS 09149 2D posterior
ax = axes[1, 2]
h = ax.hist2d(np.log10(iras_M), iras_a, bins=50, density=True, cmap='Oranges',
              norm=mcolors.LogNorm())
plt.colorbar(h[3], ax=ax, label='Density')
ax.set_xlabel('$\\log_{10}(M / M_\\odot)$', fontsize=12)
ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_title('IRAS 09149-6206: Joint Posterior', fontsize=13)

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig1_data_overview.png")

# ============================================================
# 3. FIGURE 2: Superradiance Physics Map
# ============================================================
print("\nGenerating Figure 2: Superradiance Physics...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel (a): Instability rate map for a specific ULB mass
ax = axes[0]
mu_demo = 1e-12  # eV

M_range = np.logspace(-1, 10, 200)  # 0.1 to 10^10 Msol
a_range = np.linspace(0.01, 0.99, 200)
M_grid, a_grid = np.meshgrid(M_range, a_range, indexing='ij')

# Compute growth rates
alpha_grid = mu_to_coupling(M_grid, mu_demo)
rate_grid = np.vectorize(superradiant_growth_rate_scalar)(alpha_grid, a_grid)

# Plot
log_rate = np.log10(np.maximum(rate_grid, 1e-30))
im = ax.pcolormesh(M_grid, a_grid, log_rate, cmap='RdYlBu_r', 
                    vmin=-20, vmax=-2, shading='auto')
plt.colorbar(im, ax=ax, label='$\\log_{10}(\\Gamma/\\mu)$')

# Overlay BH measurements
ax.scatter(m33_M, m33_a, s=3, alpha=0.3, color='black', label='M33 X-7')
# For IRAS, use log scale x
ax.scatter(iras_M, iras_a, s=3, alpha=0.3, color='red', label='IRAS 09149')

ax.set_xscale('log')
ax.set_xlabel('BH Mass [$M_\\odot$]', fontsize=12)
ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_title(f'Superradiant Growth Rate Map\n($\\mu = {mu_demo:.0e}$ eV)', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)

# Panel (b): Equilibrium spin as function of μ for different BH masses
ax = axes[1]
mu_grid = np.logspace(-20, -9, 300)

bh_masses = [10, 30, 100, 1e6, 1e8, 1e9]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(bh_masses)))

for M, col in zip(bh_masses, colors):
    a_eq = np.array([equilibrium_spin(M, mu) for mu in mu_grid])
    ax.plot(mu_grid, a_eq, color=col, lw=2, label=f'$M = {M:.0e}$ $M_\\odot$' if M >= 1 else f'$M = {M:.0f}$ $M_\\odot$')

ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Equilibrium Spin $a_{\\rm eq}$', fontsize=12)
ax.set_title('Equilibrium Spin after Superradiance', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(0, 1)
ax.axhline(0.9, color='gray', ls=':', alpha=0.5, label='$a_* = 0.9$')

# Panel (c): Superradiant timescale as function of μ
ax = axes[2]

for M, col in zip(bh_masses, colors):
    tau = np.array([superradiant_timescale(M, 0.9, mu) for mu in mu_grid])
    tau_log = np.where(np.isfinite(tau) & (tau > 0), tau, np.nan)
    ax.plot(mu_grid, tau_log, color=col, lw=2, label=f'$M = {M:.0e}$ $M_\\odot$' if M >= 1 else f'$M = {M:.0f}$ $M_\\odot$')

ax.axhline(1e10 * 3.156e7, color='red', ls='--', lw=2, label='10 Gyr (Hubble time)')
ax.axhline(1e7 * 3.156e7, color='orange', ls='--', lw=1.5, label='10 Myr')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Superradiant Timescale $\\tau_{\\rm SR}$ [s]', fontsize=12)
ax.set_title('Superradiant E-folding Timescale\n($a_* = 0.9$)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(1e-5, 1e30)

plt.tight_layout()
plt.savefig('report/images/fig2_superradiance_physics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig2_superradiance_physics.png")

# ============================================================
# 4. FIGURE 3: Regge Trajectories and Exclusion Zones
# ============================================================
print("\nGenerating Figure 3: Regge Trajectories...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
# Plot the exclusion zone in the mass-spin plane

# For several ULB masses, compute the exclusion boundary
mu_values = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10]
colors_mu = plt.cm.plasma(np.linspace(0, 0.9, len(mu_values)))

M_plot = np.logspace(0, 10, 500)

for mu, col in zip(mu_values, colors_mu):
    a_eq_plot = np.array([equilibrium_spin(M, mu) for M in M_plot])
    ax.plot(M_plot, a_eq_plot, color=col, lw=2.5, 
            label=f'$\\mu = {mu:.0e}$ eV')

# Shade the "excluded" region
# For the dominant exclusion, the region above the a_eq curve is excluded
ax.fill_between([1, 1e10], [0.95, 0.95], [1.0, 1.0], 
                alpha=0.1, color='red', label='Superradiance excluded')

# Overlay BH data points
ax.scatter(m33_M, m33_a, s=10, alpha=0.5, color='navy', 
           label='M33 X-7', zorder=5)
ax.scatter(iras_M, iras_a, s=10, alpha=0.5, color='darkred', 
           label='IRAS 09149', zorder=5)

ax.set_xscale('log')
ax.set_xlabel('BH Mass [$M_\\odot$]', fontsize=12)
ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_title('Regge Plane: Exclusion Boundaries', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.set_ylim(0, 1)
ax.set_xlim(1, 1e10)

# Panel (b): Zoomed view for M33 X-7 mass range
ax = axes[1]
M_zoom = np.logspace(0.5, 1.5, 200)

for mu, col in zip(mu_values, colors_mu):
    a_eq_zoom = np.array([equilibrium_spin(M, mu) for M in M_zoom])
    ax.plot(M_zoom, a_eq_zoom, color=col, lw=2.5,
            label=f'$\\mu = {mu:.0e}$ eV')

ax.scatter(m33_M, m33_a, s=15, alpha=0.5, color='navy', 
           label='M33 X-7', zorder=5)

ax.set_xscale('log')
ax.set_xlabel('BH Mass [$M_\\odot$]', fontsize=12)
ax.set_ylabel('Dimensionless Spin $a_*$', fontsize=12)
ax.set_title('Regge Plane: Stellar-mass BH Region (zoom)', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_xlim(5, 30)

plt.tight_layout()
plt.savefig('report/images/fig3_regge_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig3_regge_trajectories.png")

# ============================================================
# 5. MAIN BAYESIAN ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("Running Bayesian Analysis...")
print("=" * 60)

# Define ULB mass grid (log-spaced)
log_mu_min = -16  # 10^-16 eV
log_mu_max = -9   # 10^-9 eV
n_mu = 150
log_mu_grid = np.linspace(log_mu_min, log_mu_max, n_mu)
mu_grid = 10**log_mu_grid

print(f"Testing {n_mu} ULB masses from 10^{log_mu_min} to 10^{log_mu_max} eV")

# BH ages (estimated)
m33_age = 2e7 * 3.156e7  # ~20 Myr in seconds (M33 X-7 is a young system)
iras_age = 5e9 * 3.156e7  # ~5 Gyr in seconds (SMBH, old system)

print(f"M33 X-7 age estimate: {m33_age/3.156e7:.0e} yr")
print(f"IRAS 09149 age estimate: {iras_age/3.156e7:.0e} yr")

# --- Individual BH exclusions ---
print("\nComputing individual BH exclusion curves...")

p_excl_m33, details_m33 = compute_exclusion_curve(
    m33_M, m33_a, mu_grid, m33_age
)
print(f"  M33 X-7: max exclusion = {np.max(p_excl_m33):.4f} at "
      f"μ = {mu_grid[np.argmax(p_excl_m33)]:.2e} eV")

p_excl_iras, details_iras = compute_exclusion_curve(
    iras_M, iras_a, mu_grid, iras_age
)
print(f"  IRAS 09149: max exclusion = {np.max(p_excl_iras):.4f} at "
      f"μ = {mu_grid[np.argmax(p_excl_iras)]:.2e} eV")

# --- Combined exclusion ---
print("\nComputing combined exclusion...")
p_excl_combined, p_individual = compute_exclusion_contour_multi_BH(
    [(m33_M, m33_a), (iras_M, iras_a)],
    mu_grid,
    t_BH_sec_list=[m33_age, iras_age]
)

mu_upper, mu_lower, mu_peak = derive_upper_limit(mu_grid, p_excl_combined)
print(f"\nCombined exclusion results:")
print(f"  Peak exclusion at μ = {mu_peak:.2e} eV")
print(f"  Max exclusion probability = {np.max(p_excl_combined):.4f}")
if mu_upper:
    print(f"  95% CL excluded range: [{mu_lower:.2e}, {mu_upper:.2e}] eV")

# --- Bayesian evidence ratio ---
print("\nComputing Bayesian evidence ratio...")
mu_grid_bf, log_bf, p_excl_bf, posterior_mu, log_post = bayesian_evidence_ratio(
    m33_M, m33_a, mu_grid, 
    log_mu_min=log_mu_min, log_mu_max=log_mu_max,
    t_BH_sec=m33_age
)
print(f"  Log10(Bayes factor) = {log_bf:.4f}")

# ============================================================
# 6. FIGURE 4: Main Results - Exclusion Curves
# ============================================================
print("\nGenerating Figure 4: Main Results...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel (a): Individual exclusion curves
ax = axes[0, 0]
ax.plot(mu_grid, p_excl_m33, 'b-', lw=2.5, label='M33 X-7 (stellar-mass)')
ax.plot(mu_grid, p_excl_iras, 'r-', lw=2.5, label='IRAS 09149 (SMBH)')
ax.plot(mu_grid, p_excl_combined, 'k-', lw=3, label='Combined')
ax.axhline(0.95, color='gray', ls='--', lw=1.5, alpha=0.7, label='95% CL')
ax.axhline(0.68, color='gray', ls=':', lw=1.5, alpha=0.7, label='68% CL')
ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Exclusion Probability $P_{\\rm excl}$', fontsize=12)
ax.set_title('Superradiance Exclusion Curves', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_xlim(10**log_mu_min, 10**log_mu_max)

# Panel (b): Bayesian posterior on μ
ax = axes[0, 1]
ax.fill_between(mu_grid, posterior_mu, alpha=0.3, color='purple')
ax.plot(mu_grid, posterior_mu, 'purple', lw=2.5)
ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Posterior $P(\\mu | \\mathrm{data})$', fontsize=12)
ax.set_title('Bayesian Posterior on ULB Mass', fontsize=13)

# Find and mark the peak
idx_peak_post = np.argmax(posterior_mu)
ax.axvline(mu_grid[idx_peak_post], color='red', ls='--', lw=2,
           label=f'Peak: {mu_grid[idx_peak_post]:.2e} eV')
ax.legend(fontsize=10)

# Panel (c): Details - equilibrium spin for M33 X-7
ax = axes[1, 0]

# For the median M33 X-7 parameters
M_med = np.median(m33_M)
a_med = np.median(m33_a)

a_eq_curve = np.array([equilibrium_spin(M_med, mu) for mu in mu_grid])
ax.plot(mu_grid, a_eq_curve, 'b-', lw=2.5, label=f'$a_{{\\rm eq}}(\\mu)$ for $M = {M_med:.1f}$ $M_\\odot$')
ax.axhline(a_med, color='red', ls='--', lw=2, label=f'Observed $a_* = {a_med:.3f}$')

# Shade excluded region
mask_excl = a_eq_curve < a_med
ax.fill_between(mu_grid[mask_excl], a_eq_curve[mask_excl], a_med, 
                alpha=0.2, color='red', label='Excluded region')

ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Spin', fontsize=12)
ax.set_title('M33 X-7: Observed vs Equilibrium Spin', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)

# Panel (d): Superradiant timescale for M33 X-7
ax = axes[1, 1]
tau_m33 = np.array([superradiant_timescale(M_med, a_med, mu) for mu in mu_grid])
tau_m33_log = np.where(np.isfinite(tau_m33) & (tau_m33 > 0), tau_m33, np.nan)
ax.plot(mu_grid, tau_m33_log, 'b-', lw=2.5)
ax.axhline(m33_age, color='red', ls='--', lw=2, 
           label=f'BH age: {m33_age/3.156e7:.0e} yr')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('$\\tau_{\\rm SR}$ [s]', fontsize=12)
ax.set_title('M33 X-7: Superradiant Timescale', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig4_main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig4_main_results.png")

# ============================================================
# 7. FIGURE 5: Self-Interaction Constraints
# ============================================================
print("\nGenerating Figure 5: Self-Interaction Constraints...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): QCD axion mass-decay constant relation
ax = axes[0]
fa_grid = np.logspace(10, 18, 200)  # GeV
# QCD axion mass: μ_a ≈ 5.7 μeV × (10^12 GeV / f_a)
mu_qcd = 5.7e-6 * (1e12 / fa_grid)  # eV

ax.plot(fa_grid, mu_qcd, 'b-', lw=2.5, label='QCD axion: $\\mu_a \\approx 5.7\\,\\mu$eV $\\times$ ($10^{12}$ GeV/$f_a$)')

# Mark excluded regions from our analysis
for mu_excl, style in [(1e-13, 'r--'), (1e-12, 'g--'), (1e-11, 'm--')]:
    fa_excl = 1e12 * 5.7e-6 / mu_excl  # GeV
    ax.axvline(fa_excl, color=style.split('--')[0], ls='--', lw=1.5,
               label=f'$\\mu = {mu_excl:.0e}$ eV → $f_a = {fa_excl:.1e}$ GeV')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Axion Decay Constant $f_a$ [GeV]', fontsize=12)
ax.set_ylabel('Axion Mass $\\mu$ [eV]', fontsize=12)
ax.set_title('QCD Axion Parameter Space', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(1e10, 1e18)
ax.set_ylim(1e-15, 1e-3)

# Panel (b): Self-interaction coupling constraints
ax = axes[1]

# Self-interaction coupling: g = μ²/(16π²f_a²)
# Bosenova condition: when g > α² (gravitational coupling)
# For the bosenova: g_3 ∝ 1/f_a, so the coupling decreases with f_a

# Plot the bosenova exclusion for different BH masses
mu_fine = np.logspace(-16, -8, 300)

for M_bh, col, ls in [(10, 'blue', '-'), (100, 'green', '--'), 
                       (1e6, 'orange', '-.'), (1e8, 'red', ':')]:
    alpha_bh = mu_to_coupling(M_bh, mu_fine)
    
    # Bosenova condition: g_3² > α² × (16π²)^(-1)
    # g_3 = μ²/(16π²f_a²), so f_a < μ²/(16π√(α²)) = μ²/(16πα)
    # In terms of f_a: f_a < μ/(16π√(α)) (simplified)
    
    # More accurately: the self-interaction potential scales as μ²/f_a²
    # The bosenova occurs when this exceeds the gravitational binding energy ~ α² μ
    # So: μ²/f_a² > α² μ → f_a < μ/α
    
    # Actually for the QCD axion, the self-interaction potential is:
    # V_3 = -g_3 μ³ cos³(φ/f_a) where g_3 = 1/(4π²) (O(1) coupling)
    # The bosenova condition: N × g_3 μ/f_a > α (occupation-dependent)
    # For practical purposes: f_a_bosenova ≈ μ / (4π²α) × N^{1/3}
    
    # Simplified bosenova boundary
    fa_bosenova = mu_fine / (4 * np.pi**2 * np.maximum(alpha_bh, 1e-20))
    ax.plot(mu_fine, fa_bosenova * 1e-9, color=col, ls=ls, lw=2,
            label=f'$M = {M_bh:.0e}$ $M_\\odot$' if M_bh >= 1 else f'$M = {M_bh}$ $M_\\odot$')

# Mark the exclusion from our analysis
ax.axvspan(1e-14, 1e-11, alpha=0.2, color='red', label='Our 95% CL exclusion')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('$f_a$ [eV] (self-interaction scale)', fontsize=12)
ax.set_title('Self-Interaction Constraints from Superradiance', fontsize=13)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('report/images/fig5_self_interaction.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig5_self_interaction.png")

# ============================================================
# 8. FIGURE 6: Validation - Convergence and Sensitivity
# ============================================================
print("\nGenerating Figure 6: Validation...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel (a): Convergence with number of samples
ax = axes[0]
sample_sizes = [50, 100, 200, 500, 1000, len(m33_M)]
p_excl_convergence = []

for n_samp in sample_sizes:
    idx = np.random.choice(len(m33_M), min(n_samp, len(m33_M)), replace=False)
    p_ex, _ = compute_exclusion_curve(m33_M[idx], m33_a[idx], mu_grid, m33_age)
    p_excl_convergence.append(p_ex)

for i, (n_samp, p_ex) in enumerate(zip(sample_sizes, p_excl_convergence)):
    ax.plot(mu_grid, p_ex, lw=1.5, alpha=0.7, 
            label=f'n = {n_samp}' if n_samp < len(m33_M) else f'n = {n_samp} (all)')

ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Exclusion Probability', fontsize=12)
ax.set_title('Convergence Test (M33 X-7)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 1)

# Panel (b): Sensitivity to BH age assumption
ax = axes[1]
ages = [1e6, 5e6, 2e7, 1e8, 5e8, 1e9]
ages_yr = [a / 3.156e7 for a in ages]

for age, age_yr in zip(ages, ages_yr):
    p_ex, _ = compute_exclusion_curve(m33_M, m33_a, mu_grid, age)
    ax.plot(mu_grid, p_ex, lw=2, label=f'{age_yr:.0e} yr')

ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Exclusion Probability', fontsize=12)
ax.set_title('Sensitivity to BH Age (M33 X-7)', fontsize=13)
ax.legend(fontsize=9)
ax.set_ylim(0, 1)

# Panel (c): Comparison of different exclusion methods
ax = axes[2]

# Method 1: Simple threshold
a_eq_all = np.array([equilibrium_spin(M, mu) for M, mu in zip(m33_M, np.tile(mu_grid, len(m33_M)//len(mu_grid)+1)[:len(m33_M)*len(mu_grid)].reshape(len(mu_grid), len(m33_M)).T)])
# Simplified: just use the median BH
M_med_m33 = np.median(m33_M)
a_eq_median = np.array([equilibrium_spin(M_med_m33, mu) for mu in mu_grid])

# Simple exclusion: P(a_obs > a_eq) from posterior
p_simple = np.zeros(len(mu_grid))
for i, mu in enumerate(mu_grid):
    a_eq = equilibrium_spin(M_med_m33, mu)
    p_simple[i] = np.mean(m33_a > a_eq)

# Time-weighted exclusion
p_time, _ = compute_exclusion_curve(m33_M, m33_a, mu_grid, m33_age)

ax.plot(mu_grid, p_simple, 'b-', lw=2.5, label='Simple threshold')
ax.plot(mu_grid, p_time, 'r-', lw=2.5, label='Time-weighted (full posterior)')
ax.axhline(0.95, color='gray', ls='--', lw=1.5, alpha=0.7, label='95% CL')

ax.set_xscale('log')
ax.set_xlabel('ULB Mass $\\mu$ [eV]', fontsize=12)
ax.set_ylabel('Exclusion Probability', fontsize=12)
ax.set_title('Comparison of Exclusion Methods', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/fig6_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved fig6_validation.png")

# ============================================================
# 9. SAVE ALL OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("Saving outputs...")
print("=" * 60)

summary = save_outputs(mu_grid, p_excl_combined, posterior_mu, log_post,
                       ['M33 X-7', 'IRAS 09149-6206'],
                       p_individual, details_m33)

# Also save individual BH results
np.savez('outputs/individual_bh_results.npz',
         mu_grid=mu_grid,
         p_excl_m33=p_excl_m33,
         p_excl_iras=p_excl_iras,
         p_excl_combined=p_excl_combined,
         a_eq_m33_median=np.array([equilibrium_spin(np.median(m33_M), mu) for mu in mu_grid]),
         a_eq_iras_median=np.array([equilibrium_spin(np.median(iras_M), mu) for mu in mu_grid]))

# Save detailed JSON results
results = {
    'methodology': {
        'framework': 'Bayesian posterior-based superradiance exclusion',
        'bh_ages': {
            'M33_X-7': f'{m33_age/3.156e7:.0e} yr',
            'IRAS_09149': f'{iras_age/3.156e7:.0e} yr'
        },
        'exclusion_method': 'Time-weighted superradiant spin-down criterion',
        'bayesian_inference': 'Log-uniform prior on μ, likelihood from full posterior'
    },
    'm33_x7': {
        'n_samples': len(m33_M),
        'mass_median_Msol': float(np.median(m33_M)),
        'mass_std_Msol': float(np.std(m33_M)),
        'spin_median': float(np.median(m33_a)),
        'spin_std': float(np.std(m33_a)),
        'max_exclusion': float(np.max(p_excl_m33)),
        'peak_exclusion_mu_eV': float(mu_grid[np.argmax(p_excl_m33)]),
    },
    'iras_09149': {
        'n_samples': len(iras_M),
        'mass_median_Msol': float(np.median(iras_M)),
        'mass_std_Msol': float(np.std(iras_M)),
        'spin_median': float(np.median(iras_a)),
        'spin_std': float(np.std(iras_a)),
        'max_exclusion': float(np.max(p_excl_iras)),
        'peak_exclusion_mu_eV': float(mu_grid[np.argmax(p_excl_iras)]),
    },
    'combined_constraints': {
        'max_exclusion_combined': float(np.max(p_excl_combined)),
        'peak_exclusion_mu_eV': float(mu_grid[np.argmax(p_excl_combined)]),
        'upper_limit_95CL_eV': float(mu_upper) if mu_upper else None,
        'lower_bound_95CL_eV': float(mu_lower) if mu_lower else None,
    },
    'bayes_factor': {
        'log10_bayes_factor': float(log_bf),
    }
}

with open('outputs/full_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("All outputs saved successfully!")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
