#!/usr/bin/env python3
"""
Bayesian Constraints on Ultralight Bosons via Black Hole Superradiance
Version 3 - Corrected Kerr bound state frequencies and SR growth rates
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os, json

# Physical constants
G_N = 6.674e-11; c = 2.998e8; hbar = 1.055e-34; M_sun = 1.989e30
eV = 1.602e-19; yr = 3.156e7
r_g_sun = G_N * M_sun / c**2  # ~1477 m per solar mass

def Omega_H(a_star):
    """Horizon angular velocity in geometric units (1/length)."""
    a_star = np.clip(a_star, 0.0, 0.9999)
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))

def sr_timescale_yr(M_Msun, a_star, mu_eV):
    """
    Superradiance timescale for the dominant (l=m=1) mode.
    Uses:
    - Kerr bound state frequency fitting: omega_R/mu = 1 - alpha^2 - 2*alpha^4
    - Detweiler (1980) growth rate: Im(omega) = (a*/48) * alpha^9 * mu
    Returns timescale in years, or np.inf if not superradiant.
    """
    mu_SI = mu_eV * eV / (hbar * c)  # 1/m
    alpha = r_g_sun * M_Msun * mu_SI  # dimensionless

    if alpha < 0.05 or alpha > 1.0:
        return np.inf

    # Kerr bound state frequency (l=m=1, n=1) - fitting to numerical results
    omega_R_over_mu = 1.0 - alpha**2 - 2.0 * alpha**4
    if omega_R_over_mu <= 0:
        return np.inf

    OmH = Omega_H(a_star)

    # Superradiance condition: omega_R / (m * mu) < Omega_H, m=1
    if omega_R_over_mu >= OmH:
        return np.inf

    # Growth rate (Detweiler 1980): Im(omega) = (a*/48) * alpha^9 * mu
    # In physical units: Gamma = Im(omega) * c [s^-1]
    Gamma_SI = (a_star / 48.0) * alpha**9 * mu_SI * c  # s^-1
    if Gamma_SI <= 0:
        return np.inf

    tau_s = 1.0 / Gamma_SI
    return tau_s / yr

# Load data
data_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/data'

def load_samples(filename):
    samples = []
    with open(os.path.join(data_dir, filename), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split()
            samples.append([float(parts[0]), float(parts[1])])
    return np.array(samples)

iras_samples = load_samples('IRAS_09149-6206_samples.dat')
m33_samples = load_samples('M33_X-7_samples.dat')

# Quick validation
for mu in [1e-12, 3.74e-12, 1e-11]:
    tsr = sr_timescale_yr(15, 0.7, mu)
    alpha = r_g_sun * 15 * (mu * eV / (hbar * c))
    print(f'M33(15,0.7) mu={mu:.2e}: alpha={alpha:.3f}, T_SR={tsr:.2e} yr')

for mu in [1e-18, 5.6e-19, 1e-19]:
    tsr = sr_timescale_yr(1e8, 0.9, mu)
    alpha = r_g_sun * 1e8 * (mu * eV / (hbar * c))
    print(f'IRAS(1e8,0.9) mu={mu:.2e}: alpha={alpha:.3f}, T_SR={tsr:.2e} yr')

os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images', exist_ok=True)
os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs', exist_ok=True)

# ============================================================
# Figure 1: Data overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist2d(np.log10(iras_samples[:,0]), iras_samples[:,1], bins=50, cmap='Blues', density=True)
ax.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206 (SMBH)', fontsize=14)
ax = axes[1]
ax.hist2d(m33_samples[:,0], m33_samples[:,1], bins=50, cmap='Oranges', density=True)
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('M33 X-7 (Stellar-mass BH)', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Compute exclusion probabilities
# ============================================================
def compute_exclusion_prob(samples, mu_eV, age_yr=1e10):
    n_excl = 0
    for j in range(len(samples)):
        tsr = sr_timescale_yr(samples[j,0], samples[j,1], mu_eV)
        if tsr < age_yr:
            n_excl += 1
    return n_excl / len(samples)

mu_grid = np.logspace(-19, -9, 500)
p_m33 = np.zeros(len(mu_grid))
p_iras = np.zeros(len(mu_grid))
for i, mu in enumerate(mu_grid):
    p_m33[i] = compute_exclusion_prob(m33_samples, mu)
    p_iras[i] = compute_exclusion_prob(iras_samples, mu)
    if i % 100 == 0:
        print(f'mu=10^{np.log10(mu):.1f} eV: M33={p_m33[i]:.3f}, IRAS={p_iras[i]:.3f}')

p_comb = np.maximum(p_m33, p_iras)

# ============================================================
# Figure 3: Bayesian constraints
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax = axes[0]
ax.semilogx(mu_grid, p_m33, 'b-', lw=2, label='M33 X-7')
ax.semilogx(mu_grid, p_iras, 'r-', lw=2, label='IRAS 09149-6206')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7, label='95% threshold')
ax.fill_between(mu_grid, 0, p_m33, where=(p_m33>0.95), alpha=0.2, color='blue')
ax.fill_between(mu_grid, 0, p_iras, where=(p_iras>0.95), alpha=0.2, color='red')
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('Bayesian Exclusion Probability from BH Superradiance', fontsize=14)
ax.legend(fontsize=11); ax.set_ylim(-0.05,1.05); ax.grid(True,alpha=0.3)

ax = axes[1]
ax.semilogx(mu_grid, p_comb, 'k-', lw=2.5, label='Combined')
ax.semilogx(mu_grid, p_m33, 'b--', lw=1.5, alpha=0.6, label='M33 X-7')
ax.semilogx(mu_grid, p_iras, 'r--', lw=1.5, alpha=0.6, label='IRAS')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7)
ax.fill_between(mu_grid, 0, p_comb, where=(p_comb>0.95), alpha=0.15, color='green')
ax.set_xlabel(r'ULB Mass $\mu$ [eV]', fontsize=13)
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('Combined Bayesian Constraint', fontsize=14)
ax.legend(fontsize=11); ax.set_ylim(-0.05,1.05); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig3_bayesian_constraints.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: SR contours for multiple mu
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# Find mu values that give interesting exclusion zones
# For stellar BH (M~15 Msun), optimal mu ~ 3.7e-12 eV
mu_vals = [1e-12, 3e-12, 6e-12, 1e-11]
for idx, mu in enumerate(mu_vals):
    ax = axes.flatten()[idx]
    Mg = np.linspace(5, 80, 120); ag = np.linspace(0.01, 0.999, 120)
    excl = np.zeros((len(ag), len(Mg)))
    for i, a in enumerate(ag):
        for j, M in enumerate(Mg):
            excl[i,j] = 1.0 if sr_timescale_yr(M, a, mu) < 1e10 else 0.0
    ax.contourf(Mg, ag, excl, levels=[0.5,1.5], colors=['#ff9999'], alpha=0.5)
    ax.contour(Mg, ag, excl, levels=[0.5], colors=['darkred'], linewidths=2)
    ax.scatter(m33_samples[::5,0], m33_samples[::5,1], s=2, alpha=0.15, color='steelblue')
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12); ax.set_ylabel(r'$a_*$', fontsize=12)
    ax.set_title(r'$\mu = %.1e$ eV' % mu, fontsize=13)
    ax.set_ylim(0,1); ax.set_xlim(5,80)
    ax.legend(handles=[Patch(facecolor='#ff9999',alpha=0.6,label='SR zone')], fontsize=10)
plt.suptitle('SR Exclusion Zones (Stellar-mass BH regime)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig2_sr_contours.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Mass-spin plane with SR zones for both sources
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# M33
mu_demo = 4e-12
ax = axes[0]
Mg = np.linspace(5, 50, 150); ag = np.linspace(0.01, 0.999, 150)
excl = np.zeros((len(ag), len(Mg)))
for i, a in enumerate(ag):
    for j, M in enumerate(Mg):
        excl[i,j] = 1.0 if sr_timescale_yr(M, a, mu_demo) < 1e10 else 0.0
ax.contourf(Mg, ag, excl, levels=[0.5,1.5], colors=['#ff9999'], alpha=0.5)
ax.contour(Mg, ag, excl, levels=[0.5], colors=['darkred'], linewidths=2)
ax.scatter(m33_samples[::5,0], m33_samples[::5,1], s=3, alpha=0.3, color='steelblue', label='M33 X-7 posterior')
ax.plot(np.median(m33_samples[:,0]), np.median(m33_samples[:,1]), 'k*', ms=15, label='Median')
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13); ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title(f'M33 X-7: SR zone for $\\mu$={mu_demo:.0e} eV', fontsize=14)
ax.set_ylim(0,1); ax.set_xlim(5,50); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)

# IRAS
mu_demo2 = 6e-19
ax = axes[1]
Mg2 = np.logspace(7, 9.5, 100); ag2 = np.linspace(0.01, 0.999, 100)
excl2 = np.zeros((len(ag2), len(Mg2)))
for i, a in enumerate(ag2):
    for j, M in enumerate(Mg2):
        excl2[i,j] = 1.0 if sr_timescale_yr(M, a, mu_demo2) < 1e10 else 0.0
ax.contourf(Mg2, ag2, excl2, levels=[0.5,1.5], colors=['#ffcccc'], alpha=0.5)
ax.contour(Mg2, ag2, excl2, levels=[0.5], colors=['darkred'], linewidths=2)
ax.scatter(iras_samples[::10,0], iras_samples[::10,1], s=3, alpha=0.2, color='darkorange', label='IRAS posterior')
ax.plot(np.median(iras_samples[:,0]), np.median(iras_samples[:,1]), 'k*', ms=15, label='Median')
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13); ax.set_ylabel(r'$a_*$', fontsize=13)
ax.set_title(f'IRAS: SR zone for $\\mu$={mu_demo2:.0e} eV', fontsize=14)
ax.set_ylim(0,1); ax.set_xscale('log'); ax.set_xlim(1e7,3e9); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig4_mass_spin_exclusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 5: QCD axion f_a constraints
# ============================================================
f_a_grid = np.logspace(14, 19, 300)
mu_axion = 6e6 / f_a_grid  # QCD axion: mu_a [eV] = 6e-10 * (1e16/f_a[GeV])
p_m33_fa = np.array([compute_exclusion_prob(m33_samples, mu) for mu in mu_axion])
p_iras_fa = np.array([compute_exclusion_prob(iras_samples, mu) for mu in mu_axion])
p_comb_fa = np.maximum(p_m33_fa, p_iras_fa)

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogx(f_a_grid, p_m33_fa, 'b-', lw=2, label='M33 X-7')
ax.semilogx(f_a_grid, p_iras_fa, 'r-', lw=2, label='IRAS 09149-6206')
ax.semilogx(f_a_grid, p_comb_fa, 'k-', lw=2.5, label='Combined')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7, label='95% threshold')
ax.fill_between(f_a_grid, 0, p_comb_fa, where=(p_comb_fa>0.95), alpha=0.15, color='green')
ax.axvline(2e16, color='purple', ls='--', alpha=0.7, label=r'$M_{\rm GUT}$')
ax.set_xlabel(r'QCD Axion $f_a$ [GeV]', fontsize=13); ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('QCD Axion Constraints from BH Superradiance', fontsize=14)
ax.legend(fontsize=10); ax.set_ylim(-0.05,1.05); ax.set_xlim(1e14,1e19); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig5_f_a_constraints.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 6: SR timescale as function of alpha
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
alpha_arr = np.linspace(0.05, 0.95, 200)
for a_star in [0.5, 0.7, 0.9, 0.99]:
    tau_arr = []
    for al in alpha_arr:
        omega_R = 1.0 - al**2 - 2*al**4
        OmH = Omega_H(a_star)
        if omega_R < OmH and omega_R > 0:
            Gamma = (a_star/48) * al**9  # in units of mu
            tau_arr.append(1.0/Gamma)
        else:
            tau_arr.append(np.inf)
    tau_arr = np.array(tau_arr)
    mask = np.isfinite(tau_arr)
    if np.any(mask):
        ax.semilogy(alpha_arr[mask], tau_arr[mask], lw=2, label=f'$a_*={a_star}$')
ax.set_xlabel(r'Gravitational coupling $\alpha = G M \mu / (\hbar c)$', fontsize=13)
ax.set_ylabel(r'SR timescale $\tau_{\rm SR}$ [$1/\mu$]', fontsize=13)
ax.set_title('Superradiance Timescale vs Coupling', fontsize=14)
ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_xlim(0.05, 0.95)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig6_sr_timescale.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Find exclusion windows
# ============================================================
def find_windows(mu_grid, p, thr=0.95):
    above = p > thr
    windows = []; in_w = False; start = 0
    for i, v in enumerate(above):
        if v and not in_w: start = i; in_w = True
        elif not v and in_w: windows.append((mu_grid[start], mu_grid[i-1])); in_w = False
    if in_w: windows.append((mu_grid[start], mu_grid[-1]))
    return windows

m33_w = find_windows(mu_grid, p_m33)
iras_w = find_windows(mu_grid, p_iras)
print('\nM33 exclusion windows:', m33_w)
print('IRAS exclusion windows:', iras_w)

# Save everything
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/exclusion_probabilities.json','w') as f:
    json.dump({'mu_grid_eV':mu_grid.tolist(),'p_M33':p_m33.tolist(),'p_IRAS':p_iras.tolist()}, f)

summary = {
    'M33_X7_windows_eV': [(float(a),float(b)) for a,b in m33_w],
    'IRAS_windows_eV': [(float(a),float(b)) for a,b in iras_w],
}
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('\nAll figures saved. Analysis complete!')
