"""
Bayesian superradiance constraint analysis for ultralight bosons.

This script loads posterior samples for two black holes (IRAS 09149-6206 and M33 X-7),
computes the posterior predictive exclusion probability in the (mu, f_a) plane,
and derives 95 % credible upper limits on the ULB decay constant f_a as a function
of mass mu, as well as excluded mass bands in the limit of negligible self-interactions.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, LogFormatter

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G = 6.67430e-11          # m^3 kg^-1 s^-2
c = 2.99792458e8         # m s^-1
hbar = 1.054571817e-34   # J s
M_sun = 1.98847e30       # kg
eV_to_J = 1.602176634e-19
M_pl_GeV = 1.22089e19    # Planck mass in GeV

# Dimensionless coupling constant: alpha = G M mu / (hbar c^3)
# For M = M_sun and mu = 1 eV:
ALPHA_CONST = G * M_sun * eV_to_J / (hbar * c**3)  # ≈ 7.483e9

# Age of each system (seconds)
T_AGE_IRAS = 1e10 * 365.25 * 24 * 3600   # 10 Gyr
T_AGE_M33  = 1e7  * 365.25 * 24 * 3600   # 10 Myr

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def alpha_of(M_msun, mu_eV):
    """Dimensionless coupling alpha = G M mu / (hbar c^3)."""
    return ALPHA_CONST * M_msun * mu_eV


def a_thr(alpha):
    """Threshold spin for l=m=1 superradiance (exact in non-relativistic limit)."""
    # Protect against overflow for very large alpha
    return np.where(alpha < 10.0, 2.0 * alpha / (1.0 + alpha**2), 0.2)


def t_inst(M_msun, a_star, alpha, cap_max_rate=True):
    """
    Instability e-folding time for the l=m=1 mode in the small-alpha limit.
    t_inst = 1 / Gamma, with M Gamma = (a_star / 48) alpha^9  (natural units).
    Converted to seconds.
    """
    alpha_safe = np.where(alpha > 1e-12, alpha, 1e-12)
    # Gamma = (a_star / 48) * alpha^9 * (c / M_len)  where M_len = G M / c^2
    Gamma = (a_star / 48.0) * (alpha_safe**9) * (c**3 / (G * M_msun * M_sun))
    if cap_max_rate:
        # Global maximum growth rate for l=m=1 scalar from Dolan (2013):
        # M * Gamma_max ≈ 1.72e-7  for a/M = 0.99.
        # Scale linearly with spin and convert to SI.
        # This prevents the small-alpha formula from overestimating the rate
        # at intermediate alpha ≳ 0.1 where the expansion breaks down.
        Gamma_max = 1.72e-7 * (a_star / 0.99) * (c**3 / (G * M_msun * M_sun))
        Gamma = np.where(Gamma > Gamma_max, Gamma_max, Gamma)
    return 1.0 / Gamma


def delta_a_BN(alpha, f_a_GeV):
    """
    Maximum spin-down per Bosenova cycle for the l=m=1 mode.
    From Eq. (48) of Arvanitaki & Dubovsky (2011):
        Delta a = 2 * (f_a / M_Pl)^2 / alpha^3 .
    """
    alpha_safe = np.where(alpha > 1e-12, alpha, 1e-12)
    return 2.0 * (f_a_GeV / M_pl_GeV)**2 / (alpha_safe**3)


# ---------------------------------------------------------------------------
# Bayesian exclusion computation
# ---------------------------------------------------------------------------
def compute_exclusion_grid(samples, mu_grid, fa_grid, t_age, chunk_size=1000):
    """
    For a given BH, compute the posterior predictive exclusion probability
    P_ex(mu, f_a) by averaging the exclusion indicator over posterior samples.

    Exclusion indicator = 1 if:
        (1) a > a_thr(alpha)   [kinematic superradiance]
        (2) t_inst < t_age     [cloud grows within BH lifetime]
        (3) delta_a_BN > a - a_thr   [cloud can spin down BH before Bosenova]
    otherwise 0.
    """
    M = samples[:, 0]
    a = samples[:, 1]
    Ns = len(M)
    Nmu = len(mu_grid)
    Nfa = len(fa_grid)

    exclusion_sum = np.zeros((Nmu, Nfa), dtype=np.float64)

    for start in range(0, Ns, chunk_size):
        end = min(start + chunk_size, Ns)
        M_chunk = M[start:end]
        a_chunk = a[start:end]

        # alpha shape: (chunk, Nmu)
        alpha = ALPHA_CONST * M_chunk[:, None] * mu_grid[None, :]
        a_t = a_thr(alpha)

        # Condition 1: kinematically superradiant
        # We restrict to alpha < 0.5 because our l=m=1 dominant-mode assumption
        # breaks down for larger couplings (higher-l modes would take over and
        # have different thresholds/growth rates).  Treating alpha > 0.5 as
        # allowed is conservative.
        cond1 = (a_chunk[:, None] > a_t) & (alpha < 0.5)

        # Condition 2: fast enough growth
        t = t_inst(M_chunk[:, None], a_chunk[:, None], alpha)
        cond2 = t < t_age

        active = cond1 & cond2  # (chunk, Nmu)

        # If no active points in this chunk, skip
        if not np.any(active):
            continue

        # Condition 3: Bosenova spin-down sufficient to reach threshold
        # We loop over f_a to keep memory usage low.
        for j, fa in enumerate(fa_grid):
            da = delta_a_BN(alpha, fa)          # (chunk, Nmu)
            cond3 = da > (a_chunk[:, None] - a_t)  # (chunk, Nmu)
            excluded = active & cond3
            exclusion_sum[:, j] += excluded.sum(axis=0)

    P_ex = exclusion_sum / Ns
    return P_ex


# ---------------------------------------------------------------------------
# Upper-limit extraction
# ---------------------------------------------------------------------------
def upper_limit_fa(mu_grid, fa_grid, P_ex, credibility=0.95):
    """
    For each mu, find the f_a such that P_ex(mu, f_a) = credibility.
    This is the Bayesian upper limit on f_a: values larger than this are
    excluded at the requested credibility level.

    Returns an array of f_a limits (GeV) with NaN where no limit can be set.
    """
    target = credibility
    limits = np.full(len(mu_grid), np.nan)
    for i in range(len(mu_grid)):
        # P_ex as a function of f_a is monotonically increasing (larger f_a -> weaker self-interaction -> more excluded)
        p = P_ex[i, :]
        if p[-1] < target:
            # Even at the largest f_a, exclusion probability is below target -> no limit
            continue
        if p[0] >= target:
            # Even at the smallest f_a, exclusion probability is above target
            limits[i] = fa_grid[0]
            continue
        # Find the crossing
        idx = np.searchsorted(p, target)
        # Linear interpolation in log-space
        log_fa_low = np.log10(fa_grid[idx - 1])
        log_fa_high = np.log10(fa_grid[idx])
        p_low = p[idx - 1]
        p_high = p[idx]
        frac = (target - p_low) / (p_high - p_low)
        log_fa = log_fa_low + frac * (log_fa_high - log_fa_low)
        limits[i] = 10.0**log_fa
    return limits


def excluded_band_mu(mu_grid, P_ex, credibility=0.95):
    """
    For f_a -> infinity (no self-interactions), P_ex(mu) is a 1D array.
    Return the intervals of mu where P_ex >= credibility.
    """
    excluded = P_ex >= credibility
    # Find contiguous intervals
    intervals = []
    inside = False
    for i, val in enumerate(excluded):
        if val and not inside:
            start = i
            inside = True
        if not val and inside:
            intervals.append((mu_grid[start], mu_grid[i - 1]))
            inside = False
    if inside:
        intervals.append((mu_grid[start], mu_grid[-1]))
    return intervals


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def setup_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
    })


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    setup_plot_style()

    # Load data
    samples_iras = np.loadtxt('data/IRAS_09149-6206_samples.dat')
    samples_m33 = np.loadtxt('data/M33_X-7_samples.dat')

    # Parameter grids
    mu_grid = np.logspace(-22, -9, 300)   # eV
    fa_grid = np.logspace(14, 20, 200)    # GeV

    print("Computing exclusion grid for IRAS 09149-6206 ...")
    P_ex_iras = compute_exclusion_grid(samples_iras, mu_grid, fa_grid, T_AGE_IRAS)
    np.save('outputs/P_ex_IRAS.npy', P_ex_iras)

    print("Computing exclusion grid for M33 X-7 ...")
    P_ex_m33 = compute_exclusion_grid(samples_m33, mu_grid, fa_grid, T_AGE_M33)
    np.save('outputs/P_ex_M33.npy', P_ex_m33)

    # Save grids metadata
    np.save('outputs/mu_grid.npy', mu_grid)
    np.save('outputs/fa_grid.npy', fa_grid)

    # Extract upper limits on f_a (95 %)
    print("Extracting upper limits on f_a ...")
    fa_lim_iras = upper_limit_fa(mu_grid, fa_grid, P_ex_iras, credibility=0.95)
    fa_lim_m33 = upper_limit_fa(mu_grid, fa_grid, P_ex_m33, credibility=0.95)

    # Extract excluded mu bands for f_a -> inf (use largest f_a in grid)
    print("Extracting excluded mass bands ...")
    band_iras = excluded_band_mu(mu_grid, P_ex_iras[:, -1], credibility=0.95)
    band_m33 = excluded_band_mu(mu_grid, P_ex_m33[:, -1], credibility=0.95)

    # Save limits to JSON
    results = {
        'mu_grid_eV': mu_grid.tolist(),
        'fa_limit_IRAS_GeV': fa_lim_iras.tolist(),
        'fa_limit_M33_GeV': fa_lim_m33.tolist(),
        'excluded_band_IRAS_eV': [[float(lo), float(hi)] for lo, hi in band_iras],
        'excluded_band_M33_eV': [[float(lo), float(hi)] for lo, hi in band_m33],
    }
    with open('outputs/upper_limits.json', 'w') as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------------
    # Figure 1: Data overview
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(samples_iras[:, 0] / 1e6, samples_iras[:, 1], s=2, alpha=0.3, c='C0')
    ax.set_xlabel(r'BH Mass $M \; [10^6 \, M_\odot]$')
    ax.set_ylabel(r'Dimensionless spin $a_*$')
    ax.set_title('IRAS 09149-6206')
    ax.set_xlim(0, None)
    ax.set_ylim(0.8, 1.0)

    ax = axes[1]
    ax.scatter(samples_m33[:, 0], samples_m33[:, 1], s=2, alpha=0.3, c='C1')
    ax.set_xlabel(r'BH Mass $M \; [M_\odot]$')
    ax.set_ylabel(r'Dimensionless spin $a_*$')
    ax.set_title('M33 X-7')
    ax.set_xlim(10, 16)
    ax.set_ylim(0.55, 0.9)

    plt.tight_layout()
    fig.savefig('report/images/fig_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: 2D exclusion contours for IRAS
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    X, Y = np.meshgrid(mu_grid, fa_grid)
    # Transpose because imshow/pcolormesh expects (M, N) with M rows = y, N cols = x
    # P_ex shape is (Nmu, Nfa). We want x=mu, y=fa.
    # For pcolormesh: C shape = (Nmu, Nfa). Actually pcolormesh(x, y, C) with C shape (len(y), len(x))? 
    # Let's use contourf with X, Y and transpose C.
    Z = P_ex_iras.T
    levels = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label(r'Exclusion probability $P_{\rm ex}(\mu, f_a)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'ULB mass $\mu$ [eV]')
    ax.set_ylabel(r'Decay constant $f_a$ [GeV]')
    ax.set_title('IRAS 09149-6206')
    ax.set_xlim(mu_grid.min(), mu_grid.max())
    ax.set_ylim(fa_grid.min(), fa_grid.max())
    # Mark 95 % contour line
    ax.contour(X, Y, Z, levels=[0.95], colors='k', linewidths=1.5)
    plt.tight_layout()
    fig.savefig('report/images/fig_exclusion_IRAS.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 3: 2D exclusion contours for M33
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    Z = P_ex_m33.T
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label(r'Exclusion probability $P_{\rm ex}(\mu, f_a)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'ULB mass $\mu$ [eV]')
    ax.set_ylabel(r'Decay constant $f_a$ [GeV]')
    ax.set_title('M33 X-7')
    ax.set_xlim(mu_grid.min(), mu_grid.max())
    ax.set_ylim(fa_grid.min(), fa_grid.max())
    ax.contour(X, Y, Z, levels=[0.95], colors='k', linewidths=1.5)
    plt.tight_layout()
    fig.savefig('report/images/fig_exclusion_M33.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 4: Upper limits on f_a as a function of mu
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_grid, fa_lim_iras, lw=2, label='IRAS 09149-6206 (SMBH)', color='C0')
    ax.plot(mu_grid, fa_lim_m33, lw=2, label='M33 X-7 (stellar-mass)', color='C1')
    ax.axhline(2e17, color='k', ls='--', lw=1, label=r'Literature bound $f_a \lesssim 2\times10^{17}\,{\rm GeV}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'ULB mass $\mu$ [eV]')
    ax.set_ylabel(r'95\% credible upper limit on $f_a$ [GeV]')
    ax.set_title(r'Bayesian upper limit on ULB decay constant $f_a$')
    ax.legend(loc='lower left')
    ax.set_xlim(mu_grid.min(), mu_grid.max())
    ax.set_ylim(fa_grid.min(), fa_grid.max())
    plt.tight_layout()
    fig.savefig('report/images/fig_upper_limits_fa.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 5: Excluded mu bands for f_a -> inf
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot P_ex vs mu for the largest f_a (weakest self-interactions)
    ax.plot(mu_grid, P_ex_iras[:, -1], lw=2, label='IRAS 09149-6206', color='C0')
    ax.plot(mu_grid, P_ex_m33[:, -1], lw=2, label='M33 X-7', color='C1')
    ax.axhline(0.95, color='k', ls='--', lw=1, label='95% credibility')
    ax.set_xscale('log')
    ax.set_xlabel(r'ULB mass $\mu$ [eV]')
    ax.set_ylabel(r'Exclusion probability $P_{\rm ex}$')
    ax.set_title(r'Excluded ULB mass bands ($f_a \to \infty$, no self-interactions)')
    ax.legend()
    ax.set_xlim(mu_grid.min(), mu_grid.max())
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig('report/images/fig_excluded_bands_mu.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 6: Combined 2D exclusion (overlay both BHs)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Combine by taking the maximum exclusion probability (either BH excludes)
    P_ex_combined = np.maximum(P_ex_iras, P_ex_m33)
    Z = P_ex_combined.T
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label(r'Combined exclusion probability $P_{\rm ex}(\mu, f_a)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'ULB mass $\mu$ [eV]')
    ax.set_ylabel(r'Decay constant $f_a$ [GeV]')
    ax.set_title('Combined constraint (IRAS + M33 X-7)')
    ax.set_xlim(mu_grid.min(), mu_grid.max())
    ax.set_ylim(fa_grid.min(), fa_grid.max())
    ax.contour(X, Y, Z, levels=[0.95], colors='k', linewidths=1.5)
    plt.tight_layout()
    fig.savefig('report/images/fig_exclusion_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Analysis complete. Outputs saved to outputs/ and report/images/.")


if __name__ == '__main__':
    main()
