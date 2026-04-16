"""
Generate all figures for the Bayesian superradiance constraint analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import brentq
import json
import os

# ============================================================
# Physical Constants
# ============================================================
G_N = 6.67430e-11
c = 2.99792458e8
hbar = 1.054571817e-34
M_sun_kg = 1.98892e30
eV_to_J = 1.602176634e-19
M_Pl_eV = 1.2209e28
yr_to_s = 365.25 * 24 * 3600

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

# ============================================================
# Physics Functions
# ============================================================
def alpha_param(M_bh_msun, mu_eV):
    M_eV = M_bh_msun * M_sun_kg * c**2 / eV_to_J
    return M_eV * mu_eV / M_Pl_eV**2

def regge_trajectory(alpha, l=1, m=1):
    n = l + 1
    alpha_eff = np.minimum(alpha, np.sqrt(2 * n**2) * 0.99)
    R = 2 * alpha_eff * (1 - alpha_eff**2 / (2 * n**2)) / m
    R = np.clip(R, 0, 1)
    a_crit = 2 * R / (1 + R**2)
    return a_crit

def superradiance_timescale(M_bh_msun, a_star, mu_eV, l=1, m=1):
    alpha = alpha_param(M_bh_msun, mu_eV)
    a_crit = regge_trajectory(alpha, l, m)
    delta_a = np.maximum(a_star - a_crit, 0)
    mu_J = mu_eV * eV_to_J
    mu_inv_s = mu_J / hbar
    gamma_rel = alpha**(4*l + 5) * delta_a / (24 * (l+1)**(4*l+5))
    gamma_abs = gamma_rel * mu_inv_s
    tau_s = np.where(gamma_abs > 0, 1.0 / gamma_abs, np.inf)
    return tau_s / yr_to_s

def compute_exclusion_probability(M_samples, a_samples, mu_eV, l=1, m=1,
                                   tau_max_years=5e7, use_timescale=True):
    alphas = alpha_param(M_samples, mu_eV)
    a_crits = regge_trajectory(alphas, l, m)
    spin_exceeded = a_samples > a_crits
    alpha_valid = (alphas > 0.005) & (alphas < 1.0)
    
    if use_timescale:
        tau_sr = superradiance_timescale(M_samples, a_samples, mu_eV, l, m)
        fast_enough = tau_sr < tau_max_years
        excluded = spin_exceeded & alpha_valid & fast_enough
    else:
        excluded = spin_exceeded & alpha_valid
    
    return np.mean(excluded)

def compute_exclusion_with_self_interaction(M_samples, a_samples, mu_eV, f_a_GeV,
                                             l=1, m=1, tau_max_years=5e7):
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

# ============================================================
# Compute all exclusion probabilities
# ============================================================
print("Computing exclusion probabilities...")

# M33 X-7 - stellar mass BH
mu_range_stellar = np.logspace(-13, -9.5, 1000)

# Conservative (with timescale cut at Salpeter time)
P_m33_l1_cons = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=1, m=1, tau_max_years=5e7, use_timescale=True)
                           for mu in mu_range_stellar])
P_m33_l2_cons = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=2, m=2, tau_max_years=5e7, use_timescale=True)
                           for mu in mu_range_stellar])

# Aggressive (without timescale cut - just Regge trajectory)
P_m33_l1_aggr = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=1, m=1, use_timescale=False)
                           for mu in mu_range_stellar])
P_m33_l2_aggr = np.array([compute_exclusion_probability(m33_mass, m33_spin, mu, l=2, m=2, use_timescale=False)
                           for mu in mu_range_stellar])

# IRAS 09149-6206 - SMBH
mu_range_smbh = np.logspace(-20.5, -17, 1000)

# For SMBHs, use longer timescale (1 Gyr)
P_iras_l1_cons = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=1, m=1, tau_max_years=1e9, use_timescale=True)
                            for mu in mu_range_smbh])
P_iras_l2_cons = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=2, m=2, tau_max_years=1e9, use_timescale=True)
                            for mu in mu_range_smbh])

# Aggressive (without timescale cut)
P_iras_l1_aggr = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=1, m=1, use_timescale=False)
                            for mu in mu_range_smbh])
P_iras_l2_aggr = np.array([compute_exclusion_probability(iras_mass, iras_spin, mu, l=2, m=2, use_timescale=False)
                            for mu in mu_range_smbh])

# Find exclusion limits
def find_exclusion_limits(mu_range, P_excl, confidence=0.95):
    excluded_mask = P_excl >= confidence
    if not np.any(excluded_mask):
        return None, None
    mu_excluded = mu_range[excluded_mask]
    return float(mu_excluded[0]), float(mu_excluded[-1])

print("\n95% Exclusion Limits:")
print("M33 X-7 (l=m=1, conservative):", find_exclusion_limits(mu_range_stellar, P_m33_l1_cons))
print("M33 X-7 (l=m=1, aggressive):", find_exclusion_limits(mu_range_stellar, P_m33_l1_aggr))
print("M33 X-7 (l=m=2, conservative):", find_exclusion_limits(mu_range_stellar, P_m33_l2_cons))
print("M33 X-7 (l=m=2, aggressive):", find_exclusion_limits(mu_range_stellar, P_m33_l2_aggr))
print("IRAS (l=m=1, conservative 1Gyr):", find_exclusion_limits(mu_range_smbh, P_iras_l1_cons))
print("IRAS (l=m=1, aggressive):", find_exclusion_limits(mu_range_smbh, P_iras_l1_aggr))
print("IRAS (l=m=2, conservative 1Gyr):", find_exclusion_limits(mu_range_smbh, P_iras_l2_cons))
print("IRAS (l=m=2, aggressive):", find_exclusion_limits(mu_range_smbh, P_iras_l2_aggr))

# ============================================================
# Self-interaction grids
# ============================================================
print("\nComputing self-interaction constraints...")
mu_scan_fa = np.logspace(-12.5, -10.0, 50)
fa_scan = np.logspace(12, 19, 50)
P_excl_fa = np.zeros((len(mu_scan_fa), len(fa_scan)))
for i, mu in enumerate(mu_scan_fa):
    for j, fa in enumerate(fa_scan):
        P_excl_fa[i, j] = compute_exclusion_with_self_interaction(
            m33_mass, m33_spin, mu, fa, l=1, m=1, tau_max_years=5e7)

mu_scan_fa_iras = np.logspace(-20.0, -17.5, 50)
fa_scan_iras = np.logspace(12, 19, 50)
P_excl_fa_iras = np.zeros((len(mu_scan_fa_iras), len(fa_scan_iras)))
for i, mu in enumerate(mu_scan_fa_iras):
    for j, fa in enumerate(fa_scan_iras):
        P_excl_fa_iras[i, j] = compute_exclusion_with_self_interaction(
            iras_mass, iras_spin, mu, fa, l=1, m=1, tau_max_years=1e9)

print("Self-interaction grids done.")

# ============================================================
# FIGURE 1: Posterior distributions of BH mass and spin
# ============================================================
print("Generating Figure 1...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# M33 X-7
ax = axes[0]
ax.scatter(m33_mass, m33_spin, s=1, alpha=0.3, c='steelblue')
from scipy.stats import gaussian_kde
xmin, xmax = np.min(m33_mass), np.max(m33_mass)
ymin, ymax = np.min(m33_spin), np.max(m33_spin)
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel = gaussian_kde(np.vstack([m33_mass, m33_spin]))
zz = np.reshape(kernel(positions), xx.shape)
ax.contour(xx, yy, zz, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='navy', linewidths=1.5)
ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]', fontsize=13)
ax.set_ylabel(r'Dimensionless Spin $a_*$', fontsize=13)
ax.set_title('M33 X-7 (Stellar-Mass BH)', fontsize=14)
ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(ymin - 0.02, ymax + 0.02)

# IRAS 09149-6206
ax = axes[1]
ax.scatter(iras_mass, iras_spin, s=1, alpha=0.3, c='darkorange')
xmin, xmax = np.min(iras_mass), np.max(iras_mass)
ymin, ymax = np.min(iras_spin), np.max(iras_spin)
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel = gaussian_kde(np.vstack([iras_mass, iras_spin]))
zz = np.reshape(kernel(positions), xx.shape)
ax.contour(xx, yy, zz, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='darkred', linewidths=1.5)
ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]', fontsize=13)
ax.set_ylabel(r'Dimensionless Spin $a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206 (SMBH)', fontsize=14)

plt.tight_layout()
plt.savefig('report/images/fig1_posterior_distributions.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Regge plane with exclusion zones
# ============================================================
print("Generating Figure 2...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# M33 X-7 Regge plane
ax = axes[0]
# Plot Regge trajectories
alpha_range = np.linspace(0.01, 0.99, 500)
for l, m, color, label in [(1,1,'blue', r'$\ell=m=1$'), (2,2,'red', r'$\ell=m=2$')]:
    a_crit = regge_trajectory(alpha_range, l, m)
    ax.plot(alpha_range, a_crit, color=color, linewidth=2, label=label)
    # Fill exclusion zone (above the Regge trajectory)
    ax.fill_between(alpha_range, a_crit, 1.0, alpha=0.15, color=color)

# Plot posterior samples
alphas_m33 = alpha_param(m33_mass, 1.0)  # dummy mu=1 to get scale
# Actually, alpha depends on mu, so we can't directly plot on alpha axis
# Instead, let's plot on a mass-spin plane and overlay the Regge trajectories
# converted to M-a* space for a specific mu

# Better approach: plot in the alpha-a* plane
# Each posterior sample maps to a different alpha for each mu
# Let's just show the Regge trajectories and indicate the BH position

# For M33 X-7 with mean mass, show where different mu values map
M_mean_m33 = np.mean(m33_mass)
a_mean_m33 = np.mean(m33_spin)
ax.axhline(y=a_mean_m33, color='gray', linestyle='--', alpha=0.5, label=r'Mean $a_*$')
ax.axhspan(np.percentile(m33_spin, 5), np.percentile(m33_spin, 95), alpha=0.1, color='gray')

ax.set_xlabel(r'$\alpha = G M \mu / \hbar c$', fontsize=13)
ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title('M33 X-7: Regge Plane', fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=11, loc='lower right')
ax.text(0.05, 0.95, f'$M = {M_mean_m33:.1f}\\ M_\\odot$\n$a_* = {a_mean_m33:.3f}$',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# IRAS 09149-6206 Regge plane
ax = axes[1]
for l, m, color, label in [(1,1,'blue', r'$\ell=m=1$'), (2,2,'red', r'$\ell=m=2$')]:
    a_crit = regge_trajectory(alpha_range, l, m)
    ax.plot(alpha_range, a_crit, color=color, linewidth=2, label=label)
    ax.fill_between(alpha_range, a_crit, 1.0, alpha=0.15, color=color)

M_mean_iras = np.mean(iras_mass)
a_mean_iras = np.mean(iras_spin)
ax.axhline(y=a_mean_iras, color='gray', linestyle='--', alpha=0.5, label=r'Mean $a_*$')
ax.axhspan(np.percentile(iras_spin, 5), np.percentile(iras_spin, 95), alpha=0.1, color='gray')

ax.set_xlabel(r'$\alpha = G M \mu / \hbar c$', fontsize=13)
ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206: Regge Plane', fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=11, loc='lower right')
ax.text(0.05, 0.95, f'$M = {M_mean_iras:.1e}\\ M_\\odot$\n$a_* = {a_mean_iras:.3f}$',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/fig2_regge_plane.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Exclusion probability vs ULB mass
# ============================================================
print("Generating Figure 3...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# M33 X-7
ax = axes[0]
ax.semilogx(mu_range_stellar, P_m33_l1_cons, 'b-', linewidth=2, label=r'$\ell=m=1$ (conservative)')
ax.semilogx(mu_range_stellar, P_m33_l1_aggr, 'b--', linewidth=2, label=r'$\ell=m=1$ (aggressive)')
ax.semilogx(mu_range_stellar, P_m33_l2_cons, 'r-', linewidth=2, label=r'$\ell=m=2$ (conservative)')
ax.semilogx(mu_range_stellar, P_m33_l2_aggr, 'r--', linewidth=2, label=r'$\ell=m=2$ (aggressive)')
ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95% CL')
ax.axhline(y=0.68, color='k', linestyle='-.', alpha=0.3, label='68% CL')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'Exclusion Probability $P_{\rm excl}$', fontsize=13)
ax.set_title('M33 X-7: ULB Mass Constraints', fontsize=14)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# IRAS 09149-6206
ax = axes[1]
ax.semilogx(mu_range_smbh, P_iras_l1_cons, 'b-', linewidth=2, label=r'$\ell=m=1$ (1 Gyr)')
ax.semilogx(mu_range_smbh, P_iras_l1_aggr, 'b--', linewidth=2, label=r'$\ell=m=1$ (aggressive)')
ax.semilogx(mu_range_smbh, P_iras_l2_cons, 'r-', linewidth=2, label=r'$\ell=m=2$ (1 Gyr)')
ax.semilogx(mu_range_smbh, P_iras_l2_aggr, 'r--', linewidth=2, label=r'$\ell=m=2$ (aggressive)')
ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95% CL')
ax.axhline(y=0.68, color='k', linestyle='-.', alpha=0.3, label='68% CL')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'Exclusion Probability $P_{\rm excl}$', fontsize=13)
ax.set_title('IRAS 09149-6206: ULB Mass Constraints', fontsize=14)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_exclusion_probability.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Mass-spin plane with superradiance exclusion bands
# ============================================================
print("Generating Figure 4...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# M33 X-7
ax = axes[0]
# Plot posterior samples
ax.scatter(m33_mass, m33_spin, s=2, alpha=0.3, c='steelblue', zorder=3)

# Overlay superradiance exclusion bands for l=m=1
# For each mu, the exclusion zone is M such that alpha is in the right range
# AND a* > a_crit
mu_values_l1 = np.logspace(-12, -10.5, 20)
for mu in mu_values_l1:
    M_range = np.linspace(10, 25, 200)
    alphas = alpha_param(M_range, mu)
    a_crits = regge_trajectory(alphas, l=1, m=1)
    valid = (alphas > 0.005) & (alphas < 1.0)
    if np.any(valid):
        ax.fill_between(M_range[valid], a_crits[valid], 1.0, alpha=0.03, color='blue')

# Overlay for l=m=2
mu_values_l2 = np.logspace(-12, -10.5, 20)
for mu in mu_values_l2:
    M_range = np.linspace(10, 25, 200)
    alphas = alpha_param(M_range, mu)
    a_crits = regge_trajectory(alphas, l=2, m=2)
    valid = (alphas > 0.005) & (alphas < 1.0)
    if np.any(valid):
        ax.fill_between(M_range[valid], a_crits[valid], 1.0, alpha=0.03, color='red')

# Add KDE contours
xmin, xmax = 10, 22
ymin, ymax = 0.6, 1.0
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel = gaussian_kde(np.vstack([m33_mass, m33_spin]))
zz = np.reshape(kernel(positions), xx.shape)
ax.contour(xx, yy, zz, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='navy', linewidths=1.5)

# Custom legend
legend_elements = [
    Patch(facecolor='blue', alpha=0.15, label=r'$\ell=m=1$ exclusion'),
    Patch(facecolor='red', alpha=0.15, label=r'$\ell=m=2$ exclusion'),
    Line2D([0], [0], color='navy', linewidth=1.5, label='Posterior contours'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower left')
ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]', fontsize=13)
ax.set_ylabel(r'Dimensionless Spin $a_*$', fontsize=13)
ax.set_title('M33 X-7: Mass-Spin Plane with SR Exclusion', fontsize=14)
ax.set_xlim(10, 22)
ax.set_ylim(0.6, 1.0)

# IRAS 09149-6206
ax = axes[1]
ax.scatter(iras_mass, iras_spin, s=1, alpha=0.2, c='darkorange', zorder=3)

mu_values_iras = np.logspace(-19.5, -17.5, 20)
for mu in mu_values_iras:
    M_range = np.logspace(7, 9, 200)
    alphas = alpha_param(M_range, mu)
    a_crits = regge_trajectory(alphas, l=1, m=1)
    valid = (alphas > 0.005) & (alphas < 1.0)
    if np.any(valid):
        ax.fill_between(M_range[valid], a_crits[valid], 1.0, alpha=0.03, color='blue')

for mu in mu_values_iras:
    M_range = np.logspace(7, 9, 200)
    alphas = alpha_param(M_range, mu)
    a_crits = regge_trajectory(alphas, l=2, m=2)
    valid = (alphas > 0.005) & (alphas < 1.0)
    if np.any(valid):
        ax.fill_between(M_range[valid], a_crits[valid], 1.0, alpha=0.03, color='red')

xmin, xmax = np.min(iras_mass)*0.5, np.max(iras_mass)*2
ymin, ymax = 0.8, 1.0
xx, yy = np.mgrid[np.log10(xmin):np.log10(xmax):100j, ymin:ymax:100j]
xx_lin = 10**xx
positions = np.vstack([iras_mass, iras_spin])
kernel = gaussian_kde(positions)
positions_grid = np.vstack([xx_lin.ravel(), yy.ravel()])
zz = np.reshape(kernel(positions_grid), xx.shape)
ax.contour(xx_lin, yy, zz, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='darkred', linewidths=1.5)

legend_elements = [
    Patch(facecolor='blue', alpha=0.15, label=r'$\ell=m=1$ exclusion'),
    Patch(facecolor='red', alpha=0.15, label=r'$\ell=m=2$ exclusion'),
    Line2D([0], [0], color='darkred', linewidth=1.5, label='Posterior contours'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower left')
ax.set_xscale('log')
ax.set_xlabel(r'Black Hole Mass $M$ [$M_\odot$]', fontsize=13)
ax.set_ylabel(r'Dimensionless Spin $a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206: Mass-Spin Plane with SR Exclusion', fontsize=14)
ax.set_xlim(xmin, xmax)
ax.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig('report/images/fig4_mass_spin_exclusion.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Self-interaction constraints (mu - f_a plane)
# ============================================================
print("Generating Figure 5...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# M33 X-7
ax = axes[0]
XX, YY = np.meshgrid(mu_scan_fa, fa_scan)
contour = ax.contourf(XX, YY, P_excl_fa.T, levels=np.arange(0, 1.1, 0.1),
                       cmap='RdYlBu_r', vmin=0, vmax=1)
ax.contour(XX, YY, P_excl_fa.T, levels=[0.68, 0.95], colors=['white', 'black'],
           linewidths=[1.5, 2.0], linestyles=['--', '-'])
plt.colorbar(contour, ax=ax, label=r'Exclusion Probability $P_{\rm excl}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'Decay Constant $f_a$ [GeV]', fontsize=13)
ax.set_title('M33 X-7: Self-Interaction Constraints', fontsize=14)

# Add QCD axion line: mu_a = 6e-10 * (1e16/f_a) eV => f_a = 6e6 / mu_a (in GeV, with mu in eV)
fa_qcd = 6e-10 * 1e16  # f_a in GeV when mu = 6e-10 eV
mu_qcd_range = np.logspace(-12.5, -10, 100)
fa_qcd_line = 6e-10 * 1e16 / mu_qcd_range * 1e-10  # This needs careful unit conversion
# mu_a [eV] = 6e-10 * (1e16 GeV / f_a [GeV]) eV
# => f_a [GeV] = 6e-10 * 1e16 / mu_a [eV] = 6e6 / mu_a [eV]
fa_qcd_line = 6e6 / mu_qcd_range
ax.plot(mu_qcd_range, fa_qcd_line, 'g--', linewidth=2, label='QCD axion')
ax.legend(fontsize=10)

# IRAS 09149-6206
ax = axes[1]
XX, YY = np.meshgrid(mu_scan_fa_iras, fa_scan_iras)
contour = ax.contourf(XX, YY, P_excl_fa_iras.T, levels=np.arange(0, 1.1, 0.1),
                       cmap='RdYlBu_r', vmin=0, vmax=1)
ax.contour(XX, YY, P_excl_fa_iras.T, levels=[0.68, 0.95], colors=['white', 'black'],
           linewidths=[1.5, 2.0], linestyles=['--', '-'])
plt.colorbar(contour, ax=ax, label=r'Exclusion Probability $P_{\rm excl}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'Decay Constant $f_a$ [GeV]', fontsize=13)
ax.set_title('IRAS 09149-6206: Self-Interaction Constraints', fontsize=14)

mu_qcd_range_iras = np.logspace(-20, -17.5, 100)
fa_qcd_line_iras = 6e6 / mu_qcd_range_iras
ax.plot(mu_qcd_range_iras, fa_qcd_line_iras, 'g--', linewidth=2, label='QCD axion')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig5_self_interaction.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6: Combined constraint summary
# ============================================================
print("Generating Figure 6...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot exclusion probability for both BHs on the same axis
# Use a common mu range spanning both
mu_full = np.logspace(-21, -9.5, 2000)

# Interpolate existing results
from scipy.interpolate import interp1d

# M33 conservative
P_m33_combined = np.maximum(P_m33_l1_cons, P_m33_l2_cons)
f_m33 = interp1d(np.log10(mu_range_stellar), P_m33_combined, bounds_error=False, fill_value=0)

# IRAS conservative
P_iras_combined = np.maximum(P_iras_l1_cons, P_iras_l2_cons)
f_iras = interp1d(np.log10(mu_range_smbh), P_iras_combined, bounds_error=False, fill_value=0)

P_m33_full = f_m33(np.log10(mu_full))
P_iras_full = f_iras(np.log10(mu_full))

# Combined (joint) exclusion: P_joint = 1 - (1-P1)*(1-P2)
P_joint = 1 - (1 - P_m33_full) * (1 - P_iras_full)

ax.semilogx(mu_full, P_m33_full, 'b-', linewidth=2, label='M33 X-7 (conservative)')
ax.semilogx(mu_full, P_iras_full, 'r-', linewidth=2, label='IRAS 09149-6206 (conservative)')
ax.semilogx(mu_full, P_joint, 'k-', linewidth=2.5, label='Combined (joint)')

ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5)
ax.axhline(y=0.68, color='k', linestyle='-.', alpha=0.3)

# Mark the QCD axion mass range
ax.axvspan(1e-12, 1e-10, alpha=0.1, color='green', label='QCD axion range')

ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=14)
ax.set_ylabel(r'Exclusion Probability $P_{\rm excl}$', fontsize=14)
ax.set_title('Combined Bayesian Constraints on Ultralight Boson Mass', fontsize=15)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add secondary x-axis showing alpha for a reference BH mass
ax2 = ax.twiny()
ax2.set_xscale('log')
# For M ~ 16 Msun: alpha = 0.117 * (mu / 1e-12)
alpha_ticks = [0.01, 0.1, 0.5, 1.0]
mu_ticks = [a * M_Pl_eV**2 / (16 * M_sun_kg * c**2 / eV_to_J) for a in alpha_ticks]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(mu_ticks)
ax2.set_xticklabels([f'{a}' for a in alpha_ticks])
ax2.set_xlabel(r'$\alpha$ (for $M = 16\ M_\odot$)', fontsize=12)

plt.tight_layout()
plt.savefig('report/images/fig6_combined_constraints.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 7: Superradiance timescale map
# ============================================================
print("Generating Figure 7...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# M33 X-7: timescale as function of mu and a*
ax = axes[0]
mu_grid = np.logspace(-12.5, -10, 100)
a_grid = np.linspace(0.5, 0.99, 100)
MU, A = np.meshgrid(mu_grid, a_grid)
M_ref = np.mean(m33_mass)
TAU = superradiance_timescale(M_ref, A, MU, l=1, m=1)

# Plot log10(tau) in years
log_tau = np.log10(np.clip(TAU, 1, 1e20))
contour = ax.contourf(MU, A, log_tau, levels=np.arange(-2, 16, 1), cmap='viridis_r')
plt.colorbar(contour, ax=ax, label=r'$\log_{10}(\tau_{\rm SR})$ [yr]')

# Overlay Regge trajectory
alphas_grid = alpha_param(M_ref, mu_grid)
a_crit_grid = regge_trajectory(alphas_grid, l=1, m=1)
ax.plot(mu_grid, a_crit_grid, 'w-', linewidth=2, label=r'Regge trajectory ($\ell=m=1$)')

# Mark Salpeter time
ax.contour(MU, A, log_tau, levels=[np.log10(5e7)], colors='red', linewidths=2, linestyles='--')

ax.set_xscale('log')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title(f'M33 X-7: SR Timescale ($M = {M_ref:.1f}\\ M_\\odot$)', fontsize=14)
ax.legend(fontsize=10, loc='lower right')

# IRAS 09149-6206
ax = axes[1]
mu_grid_iras = np.logspace(-20, -17.5, 100)
a_grid_iras = np.linspace(0.5, 0.99, 100)
MU, A = np.meshgrid(mu_grid_iras, a_grid_iras)
M_ref_iras = np.mean(iras_mass)
TAU = superradiance_timescale(M_ref_iras, A, MU, l=1, m=1)

log_tau = np.log10(np.clip(TAU, 1, 1e20))
contour = ax.contourf(MU, A, log_tau, levels=np.arange(-2, 16, 1), cmap='viridis_r')
plt.colorbar(contour, ax=ax, label=r'$\log_{10}(\tau_{\rm SR})$ [yr]')

alphas_grid = alpha_param(M_ref_iras, mu_grid_iras)
a_crit_grid = regge_trajectory(alphas_grid, l=1, m=1)
ax.plot(mu_grid_iras, a_crit_grid, 'w-', linewidth=2, label=r'Regge trajectory ($\ell=m=1$)')
ax.contour(MU, A, log_tau, levels=[np.log10(1e9)], colors='red', linewidths=2, linestyles='--')

ax.set_xscale('log')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title(f'IRAS 09149-6206: SR Timescale ($M = {M_ref_iras:.1e}\\ M_\\odot$)', fontsize=14)
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('report/images/fig7_timescale_map.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# Save all numerical results
# ============================================================
results = {
    "M33_X7": {
        "n_samples": int(len(m33_mass)),
        "mass_mean_msun": float(np.mean(m33_mass)),
        "mass_std_msun": float(np.std(m33_mass)),
        "spin_mean": float(np.mean(m33_spin)),
        "spin_std": float(np.std(m33_spin)),
        "l1_conservative_95_low": find_exclusion_limits(mu_range_stellar, P_m33_l1_cons)[0],
        "l1_conservative_95_high": find_exclusion_limits(mu_range_stellar, P_m33_l1_cons)[1],
        "l1_aggressive_95_low": find_exclusion_limits(mu_range_stellar, P_m33_l1_aggr)[0],
        "l1_aggressive_95_high": find_exclusion_limits(mu_range_stellar, P_m33_l1_aggr)[1],
        "l2_conservative_95_low": find_exclusion_limits(mu_range_stellar, P_m33_l2_cons)[0],
        "l2_conservative_95_high": find_exclusion_limits(mu_range_stellar, P_m33_l2_cons)[1],
        "l2_aggressive_95_low": find_exclusion_limits(mu_range_stellar, P_m33_l2_aggr)[0],
        "l2_aggressive_95_high": find_exclusion_limits(mu_range_stellar, P_m33_l2_aggr)[1],
    },
    "IRAS_09149": {
        "n_samples": int(len(iras_mass)),
        "mass_mean_msun": float(np.mean(iras_mass)),
        "mass_std_msun": float(np.std(iras_mass)),
        "spin_mean": float(np.mean(iras_spin)),
        "spin_std": float(np.std(iras_spin)),
        "l1_conservative_95_low": find_exclusion_limits(mu_range_smbh, P_iras_l1_cons)[0],
        "l1_conservative_95_high": find_exclusion_limits(mu_range_smbh, P_iras_l1_cons)[1],
        "l1_aggressive_95_low": find_exclusion_limits(mu_range_smbh, P_iras_l1_aggr)[0],
        "l1_aggressive_95_high": find_exclusion_limits(mu_range_smbh, P_iras_l1_aggr)[1],
        "l2_conservative_95_low": find_exclusion_limits(mu_range_smbh, P_iras_l2_cons)[0],
        "l2_conservative_95_high": find_exclusion_limits(mu_range_smbh, P_iras_l2_cons)[1],
        "l2_aggressive_95_low": find_exclusion_limits(mu_range_smbh, P_iras_l2_aggr)[0],
        "l2_aggressive_95_high": find_exclusion_limits(mu_range_smbh, P_iras_l2_aggr)[1],
    }
}

with open('outputs/constraint_results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.savez('outputs/exclusion_probabilities.npz',
         mu_stellar=mu_range_stellar,
         mu_smbh=mu_range_smbh,
         P_m33_l1_cons=P_m33_l1_cons,
         P_m33_l1_aggr=P_m33_l1_aggr,
         P_m33_l2_cons=P_m33_l2_cons,
         P_m33_l2_aggr=P_m33_l2_aggr,
         P_iras_l1_cons=P_iras_l1_cons,
         P_iras_l1_aggr=P_iras_l1_aggr,
         P_iras_l2_cons=P_iras_l2_cons,
         P_iras_l2_aggr=P_iras_l2_aggr,
         mu_scan_fa=mu_scan_fa,
         fa_scan=fa_scan,
         P_excl_fa=P_excl_fa,
         mu_scan_fa_iras=mu_scan_fa_iras,
         fa_scan_iras=fa_scan_iras,
         P_excl_fa_iras=P_excl_fa_iras)

print("\nAll figures and results saved!")
