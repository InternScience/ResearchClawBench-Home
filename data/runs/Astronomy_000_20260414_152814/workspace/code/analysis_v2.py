#!/usr/bin/env python3
"""
Bayesian Constraints on Ultralight Bosons via Black Hole Superradiance - FIXED VERSION
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os, json

# Constants
G_N = 6.674e-11; c = 2.998e8; hbar = 1.055e-34; M_sun = 1.989e30; eV = 1.602e-19; yr = 3.156e7

def Omega_H(a_star):
    a_star = np.clip(a_star, 0, 0.9999)
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))

def sr_timescale_yr(M_Msun, a_star, mu_eV):
    """
    Superradiance timescale for dominant mode using Detweiler (1980) formula.
    For (n=2, l=m=1): tau ~ 24*G*M / (a* * (G*M*mu)^9) in geometric units.
    Returns timescale in years, or np.inf if not superradiant.
    """
    # alpha = G M mu / (hbar c) -- dimensionless coupling
    M_kg = M_Msun * M_sun
    mu_SI = mu_eV * eV / (hbar * c)  # 1/m
    alpha = G_N * M_kg * mu_SI / c**2
    
    if alpha < 0.01 or alpha > 2.0:
        return np.inf
    
    # Mode frequency (n=2, l=1): omega/mu = 1 - alpha^2/(2*2^2) = 1 - alpha^2/8
    omega_over_mu = 1.0 - alpha**2 / 8.0
    
    OmH = Omega_H(a_star)
    
    # Superradiance condition: omega/m < Omega_H  (m=1)
    # Since omega/mu ~ 1 and OmH < 0.5, this is almost never satisfied
    # with the simple hydrogenic formula for alpha << 1.
    # 
    # The correct approach: for small alpha, the SR condition becomes
    # 1 - alpha^2/(2n^2) < OmH, which requires alpha^2/2 > 1 - OmH
    # i.e., alpha > sqrt(2*(1-OmH)). For a*=0.9, OmH~0.31, need alpha>1.18
    # This means the simple n=2,l=1 mode only works for large alpha.
    #
    # For SMALL alpha, we need modes with LARGER m. The (n,l,m) mode has
    # omega_nm = mu*(1 - alpha^2/(2n^2)), and SR condition is omega/m < OmH.
    # For the ground state (n=1,l=0,m=0) there's no SR. 
    # For (n,l,m) with m>0: need mu*(1-alpha^2/(2n^2))/m < OmH
    # For small alpha: need 1/m < OmH, i.e., m > 1/OmH.
    # The growth rate for mode (n,l,m) scales as alpha^(4l+4) * mu.
    # The fastest growing superradiant mode has the smallest l,m satisfying SR.
    
    # Find the fastest-growing superradiant mode
    best_rate = 0.0
    for l in range(1, 6):
        for m in range(1, l+1):
            for n in range(max(l, 1), 4):
                # Hydrogenic energy
                omega_over_mu_nm = 1.0 - alpha**2 / (2.0 * n**2)
                # SR condition
                if omega_over_mu_nm / m < OmH:
                    # Growth rate ~ alpha^(4l+4) * mu * (OmH - omega/m)
                    # From Detweiler/Brito et al: Gamma_sr ~ 2*(OmH - omega/m)*alpha^(4l+4)*mu
                    delta = OmH - omega_over_mu_nm / m
                    Gamma = 2.0 * delta * alpha**(4*l + 4) * mu_SI * c  # 1/s
                    Gamma_yr = Gamma * yr
                    if Gamma_yr > best_rate:
                        best_rate = Gamma_yr
    
    if best_rate <= 0:
        return np.inf
    return 1.0 / best_rate

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
print(f"IRAS: {len(iras_samples)} samples, M=[{iras_samples[:,0].min():.2e},{iras_samples[:,0].max():.2e}], a*=[{iras_samples[:,1].min():.3f},{iras_samples[:,1].max():.3f}]")
print(f"M33: {len(m33_samples)} samples, M=[{m33_samples[:,0].min():.1f},{m33_samples[:,0].max():.1f}], a*=[{m33_samples[:,1].min():.3f},{m33_samples[:,1].max():.3f}]")

# Quick test of SR timescale
for mu in [1e-12, 1e-11, 1e-10]:
    tsr = sr_timescale_yr(15, 0.7, mu)
    print(f"  M33(15, a*=0.7), mu={mu:.0e}: T_SR={tsr:.2e} yr")

for mu in [1e-15, 1e-14, 1e-13]:
    tsr = sr_timescale_yr(1e8, 0.9, mu)
    print(f"  IRAS(1e8, a*=0.9), mu={mu:.0e}: T_SR={tsr:.2e} yr")

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
print("Saved fig1")

# ============================================================
# Figure 2: SR exclusion contours
# ============================================================
def compute_exclusion_prob(samples, mu_eV, age_yr=1e10):
    n_excl = 0
    for j in range(len(samples)):
        tsr = sr_timescale_yr(samples[j,0], samples[j,1], mu_eV)
        if tsr < age_yr:
            n_excl += 1
    return n_excl / len(samples)

mu_grid = np.logspace(-18, -9, 400)
p_m33 = np.zeros(len(mu_grid))
p_iras = np.zeros(len(mu_grid))
for i, mu in enumerate(mu_grid):
    p_m33[i] = compute_exclusion_prob(m33_samples, mu)
    p_iras[i] = compute_exclusion_prob(iras_samples, mu)
    if i % 80 == 0:
        print(f"  mu=10^{np.log10(mu):.1f} eV: M33={p_m33[i]:.3f}, IRAS={p_iras[i]:.3f}")

p_comb = np.maximum(p_m33, p_iras)

# Save
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/exclusion_probabilities.json','w') as f:
    json.dump({'mu_grid_eV':mu_grid.tolist(),'p_M33':p_m33.tolist(),'p_IRAS':p_iras.tolist()}, f)

# Main constraint figure
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
print("Saved fig3")

# ============================================================
# Figure 4: Mass-spin plane with SR zones
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# M33 demo
mu_demo = 5e-12
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
ax.set_title(f'M33 X-7: SR zone for μ={mu_demo:.0e} eV', fontsize=14)
ax.set_ylim(0,1); ax.set_xlim(5,50); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)

# IRAS demo
mu_demo2 = 5e-15
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
ax.set_title(f'IRAS: SR zone for μ={mu_demo2:.0e} eV', fontsize=14)
ax.set_ylim(0,1); ax.set_xscale('log'); ax.set_xlim(1e7,3e9); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig4_mass_spin_exclusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ============================================================
# Figure 5: QCD axion f_a constraints
# ============================================================
f_a_grid = np.logspace(14, 19, 300)
mu_axion = 6e6 / f_a_grid
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
ax.axvline(1.22e19, color='brown', ls='--', alpha=0.7, label=r'$M_{\rm Pl}$')
ax.set_xlabel(r'QCD Axion $f_a$ [GeV]', fontsize=13); ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('QCD Axion Constraints from BH Superradiance', fontsize=14)
ax.legend(fontsize=10); ax.set_ylim(-0.05,1.05); ax.set_xlim(1e14,1e19); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig5_f_a_constraints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ============================================================
# Figure 2: SR contours for multiple mu
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
mu_vals = [1e-13, 1e-12, 1e-11, 1e-10]
for idx, mu in enumerate(mu_vals):
    ax = axes.flatten()[idx]
    Mg = np.linspace(5, 80, 120); ag = np.linspace(0.01, 0.999, 120)
    excl = np.zeros((len(ag), len(Mg)))
    for i, a in enumerate(ag):
        for j, M in enumerate(Mg):
            excl[i,j] = 1.0 if sr_timescale_yr(M, a, mu) < 1e10 else 0.0
    ax.contourf(Mg, ag, excl, levels=[0.5,1.5], colors=['#ff9999'], alpha=0.5)
    ax.contour(Mg, ag, excl, levels=[0.5], colors=['red'], linewidths=2)
    ax.scatter(m33_samples[::10,0], m33_samples[::10,1], s=1, alpha=0.1, color='blue')
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12); ax.set_ylabel(r'$a_*$', fontsize=12)
    ax.set_title(r'$\mu = 10^{{{:.0f}}}$ eV'.format(np.log10(mu)), fontsize=13)
    ax.set_ylim(0,1); ax.set_xlim(5,80)
    ax.legend(handles=[Patch(facecolor='#ff9999',alpha=0.6,label='SR zone')], fontsize=10)
plt.suptitle('SR Exclusion Zones (Stellar-mass BH)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig2_sr_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

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
print("\nM33 exclusion windows:", m33_w)
print("IRAS exclusion windows:", iras_w)

summary = {
    'M33_X7_windows_eV': [(float(a),float(b)) for a,b in m33_w],
    'IRAS_windows_eV': [(float(a),float(b)) for a,b in iras_w],
}
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/summary.json','w') as f:
    json.dump(summary, f, indent=2)
print("Done!")
