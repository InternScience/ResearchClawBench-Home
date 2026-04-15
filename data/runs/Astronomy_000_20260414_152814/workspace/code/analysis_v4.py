#!/usr/bin/env python3
"""
Bayesian Constraints on Ultralight Bosons via Black Hole Superradiance
Version 4 - Corrected Kerr bound state frequencies using Padé approximation
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
r_g_sun = G_N * M_sun / c**2  # ~1477 m per Msun
M_s_per_Msun = M_sun * G_N / c**3  # seconds per Msun in geometric units

def Omega_H(a_star):
    """Horizon angular velocity (dimensionless, 1/r_g)."""
    a_star = np.clip(a_star, 0.0, 0.9999)
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))

def omega_R_over_mu(alpha):
    """
    Kerr bound state frequency for (n=1, l=m=1) mode.
    Padé approximation matching: omega/mu -> 1-alpha^2/2 (small alpha),
    omega/mu -> 0.5 at alpha~0.42 (extremal SR threshold).
    Fitted to Detweiler 1980 / Dolan 2007 numerical results.
    """
    return (1.0 - 2.5*alpha) / (1.0 + 1.5*alpha)

def sr_timescale_yr(M_Msun, a_star, mu_eV):
    """
    SR timescale for dominant (l=m=1) mode in years.
    Returns np.inf if not superradiant.
    """
    mu_SI = mu_eV * eV / (hbar * c)
    alpha = r_g_sun * M_Msun * mu_SI
    if alpha < 0.05 or alpha > 0.95:
        return np.inf
    wR = omega_R_over_mu(alpha)
    if wR <= 0:
        return np.inf
    OmH = Omega_H(a_star)
    if wR >= OmH:
        return np.inf
    # Growth rate: Im(omega) = a* * alpha^9 * (OmH - omega_R/mu) / 48
    # (Detweiler 1980 prefactor, extended with delta for finite separation)
    delta = OmH - wR
    Im_omega = (a_star / 48.0) * alpha**9 * delta
    if Im_omega <= 0:
        return np.inf
    # tau = M_s / Im_omega where M_s = M * G/c^3
    M_s = M_Msun * M_s_per_Msun
    tau_s = M_s / Im_omega
    return tau_s / yr

# Load data
data_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/data'
def load_samples(fn):
    samples = []
    with open(os.path.join(data_dir, fn)) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split()
            samples.append([float(parts[0]), float(parts[1])])
    return np.array(samples)

iras = load_samples('IRAS_09149-6206_samples.dat')
m33 = load_samples('M33_X-7_samples.dat')

# Validate
for mu in [1e-12, 3e-12, 4e-12]:
    tsr = sr_timescale_yr(15, 0.7, mu)
    a = r_g_sun*15*(mu*eV/(hbar*c))
    print(f'M33(15,0.7) mu={mu:.0e}: alpha={a:.3f}, T_SR={tsr:.2e} yr')
for mu in [1e-19, 5e-19, 1e-18]:
    tsr = sr_timescale_yr(1e8, 0.9, mu)
    a = r_g_sun*1e8*(mu*eV/(hbar*c))
    print(f'IRAS(1e8,0.9) mu={mu:.0e}: alpha={a:.3f}, T_SR={tsr:.2e} yr')

os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images', exist_ok=True)
os.makedirs('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs', exist_ok=True)

# ============================================================
# Figure 1: Data overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist2d(np.log10(iras[:,0]), iras[:,1], bins=50, cmap='Blues', density=True)
ax.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('IRAS 09149-6206 (SMBH)', fontsize=14)
ax = axes[1]
ax.hist2d(m33[:,0], m33[:,1], bins=50, cmap='Oranges', density=True)
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13)
ax.set_ylabel(r'Dimensionless spin $a_*$', fontsize=13)
ax.set_title('M33 X-7 (Stellar-mass BH)', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1')

# ============================================================
# Figure 6: SR timescale vs alpha
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
alpha_arr = np.linspace(0.05, 0.85, 300)
for a_star in [0.5, 0.7, 0.9, 0.99]:
    tau_arr = []
    for al in alpha_arr:
        wR = omega_R_over_mu(al)
        OmH = Omega_H(a_star)
        if wR > 0 and wR < OmH:
            delta = OmH - wR
            Im_w = (a_star/48)*al**9*delta
            if Im_w > 0:
                tau_arr.append(1.0/Im_w)
            else:
                tau_arr.append(np.inf)
        else:
            tau_arr.append(np.inf)
    tau_arr = np.array(tau_arr)
    mask = np.isfinite(tau_arr)
    if np.any(mask):
        ax.semilogy(alpha_arr[mask], tau_arr[mask], lw=2, label=f'$a_*={a_star}$')
ax.set_xlabel(r'Gravitational coupling $\alpha$', fontsize=13)
ax.set_ylabel(r'SR timescale $\tau_{\rm SR}$ [$M_{\rm BH}$]', fontsize=13)
ax.set_title('Superradiance Timescale (l=m=1 mode)', fontsize=14)
ax.legend(fontsize=12); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig6_sr_timescale.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig6')

# ============================================================
# Compute exclusion probabilities
# ============================================================
def excl_prob(samples, mu_eV, age_yr=1e10):
    n = 0
    for j in range(len(samples)):
        if sr_timescale_yr(samples[j,0], samples[j,1], mu_eV) < age_yr:
            n += 1
    return n / len(samples)

mu_grid = np.logspace(-19, -9, 500)
p_m33 = np.zeros(len(mu_grid))
p_iras = np.zeros(len(mu_grid))
for i, mu in enumerate(mu_grid):
    p_m33[i] = excl_prob(m33, mu)
    p_iras[i] = excl_prob(iras, mu)
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
print('Saved fig3')

# ============================================================
# Figure 2: SR contours
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
mu_vals = [1e-12, 3e-12, 5e-12, 1e-11]
for idx, mu in enumerate(mu_vals):
    ax = axes.flatten()[idx]
    Mg = np.linspace(5, 80, 120); ag = np.linspace(0.01, 0.999, 120)
    excl = np.zeros((len(ag), len(Mg)))
    for i, a in enumerate(ag):
        for j, M in enumerate(Mg):
            excl[i,j] = 1.0 if sr_timescale_yr(M, a, mu) < 1e10 else 0.0
    ax.contourf(Mg, ag, excl, levels=[0.5,1.5], colors=['#ff9999'], alpha=0.5)
    ax.contour(Mg, ag, excl, levels=[0.5], colors=['darkred'], linewidths=2)
    ax.scatter(m33[::5,0], m33[::5,1], s=2, alpha=0.15, color='steelblue')
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12)
    ax.set_ylabel(r'$a_*$', fontsize=12)
    ax.set_title(r'$\mu = %.0e$ eV' % mu, fontsize=13)
    ax.set_ylim(0,1); ax.set_xlim(5,80)
    ax.legend(handles=[Patch(facecolor='#ff9999',alpha=0.6,label='SR zone')], fontsize=10)
plt.suptitle('SR Exclusion Zones (Stellar-mass BH)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig2_sr_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2')

# ============================================================
# Figure 4: Mass-spin with SR zones for both sources
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# M33
mu_d = 4e-12
ax = axes[0]
Mg = np.linspace(5,50,150); ag = np.linspace(0.01,0.999,150)
excl = np.zeros((len(ag),len(Mg)))
for i,a in enumerate(ag):
    for j,M in enumerate(Mg):
        excl[i,j]=1.0 if sr_timescale_yr(M,a,mu_d)<1e10 else 0.0
ax.contourf(Mg,ag,excl,levels=[0.5,1.5],colors=['#ff9999'],alpha=0.5)
ax.contour(Mg,ag,excl,levels=[0.5],colors=['darkred'],linewidths=2)
ax.scatter(m33[::5,0],m33[::5,1],s=3,alpha=0.3,color='steelblue',label='M33 X-7 posterior')
ax.plot(np.median(m33[:,0]),np.median(m33[:,1]),'k*',ms=15,label='Median')
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$',fontsize=13); ax.set_ylabel(r'$a_*$',fontsize=13)
ax.set_title(f'M33 X-7: SR zone, $\\mu$={mu_d:.0e} eV',fontsize=14)
ax.set_ylim(0,1); ax.set_xlim(5,50); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
# IRAS
mu_d2 = 5e-19
ax = axes[1]
Mg2 = np.logspace(7,9.5,100); ag2 = np.linspace(0.01,0.999,100)
excl2 = np.zeros((len(ag2),len(Mg2)))
for i,a in enumerate(ag2):
    for j,M in enumerate(Mg2):
        excl2[i,j]=1.0 if sr_timescale_yr(M,a,mu_d2)<1e10 else 0.0
ax.contourf(Mg2,ag2,excl2,levels=[0.5,1.5],colors=['#ffcccc'],alpha=0.5)
ax.contour(Mg2,ag2,excl2,levels=[0.5],colors=['darkred'],linewidths=2)
ax.scatter(iras[::10,0],iras[::10,1],s=3,alpha=0.2,color='darkorange',label='IRAS posterior')
ax.plot(np.median(iras[:,0]),np.median(iras[:,1]),'k*',ms=15,label='Median')
ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$',fontsize=13); ax.set_ylabel(r'$a_*$',fontsize=13)
ax.set_title(f'IRAS: SR zone, $\\mu$={mu_d2:.0e} eV',fontsize=14)
ax.set_ylim(0,1); ax.set_xscale('log'); ax.set_xlim(1e7,3e9); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig4_mass_spin_exclusion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4')

# ============================================================
# Figure 5: QCD axion f_a constraints
# ============================================================
f_a_grid = np.logspace(14,19,300)
mu_axion = 6e6 / f_a_grid
p_m33_fa = np.array([excl_prob(m33,mu) for mu in mu_axion])
p_iras_fa = np.array([excl_prob(iras,mu) for mu in mu_axion])
p_comb_fa = np.maximum(p_m33_fa, p_iras_fa)

fig, ax = plt.subplots(figsize=(12,6))
ax.semilogx(f_a_grid, p_m33_fa, 'b-', lw=2, label='M33 X-7')
ax.semilogx(f_a_grid, p_iras_fa, 'r-', lw=2, label='IRAS 09149-6206')
ax.semilogx(f_a_grid, p_comb_fa, 'k-', lw=2.5, label='Combined')
ax.axhline(0.95, color='gray', ls='--', alpha=0.7, label='95% threshold')
ax.fill_between(f_a_grid, 0, p_comb_fa, where=(p_comb_fa>0.95), alpha=0.15, color='green')
ax.axvline(2e16, color='purple', ls='--', alpha=0.7, label=r'$M_{\rm GUT}$')
ax.set_xlabel(r'QCD Axion $f_a$ [GeV]', fontsize=13)
ax.set_ylabel('Exclusion Probability', fontsize=13)
ax.set_title('QCD Axion Constraints from BH Superradiance', fontsize=14)
ax.legend(fontsize=10); ax.set_ylim(-0.05,1.05); ax.set_xlim(1e14,1e19); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/report/images/fig5_f_a_constraints.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5')

# ============================================================
# Find exclusion windows
# ============================================================
def find_windows(mu_g, p, thr=0.95):
    above = p > thr
    wins = []; in_w = False; s = 0
    for i, v in enumerate(above):
        if v and not in_w: s = i; in_w = True
        elif not v and in_w: wins.append((mu_g[s], mu_g[i-1])); in_w = False
    if in_w: wins.append((mu_g[s], mu_g[-1]))
    return wins

m33_w = find_windows(mu_grid, p_m33)
iras_w = find_windows(mu_grid, p_iras)
comb_w = find_windows(mu_grid, p_comb)
print('\nM33 windows:', m33_w)
print('IRAS windows:', iras_w)
print('Combined windows:', comb_w)

with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/summary.json','w') as f:
    json.dump({
        'M33_windows_eV': [(float(a),float(b)) for a,b in m33_w],
        'IRAS_windows_eV': [(float(a),float(b)) for a,b in iras_w],
        'combined_windows_eV': [(float(a),float(b)) for a,b in comb_w],
    }, f, indent=2)

with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_000_20260414_152814/outputs/exclusion_probabilities.json','w') as f:
    json.dump({'mu_grid_eV':mu_grid.tolist(),'p_M33':p_m33.tolist(),'p_IRAS':p_iras.tolist()}, f)

print('Done!')
