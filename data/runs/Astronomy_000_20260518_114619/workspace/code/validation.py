"""
Validation script: compare the full Bayesian posterior-averaged exclusion
with the naive "plug-in" exclusion obtained from a single point estimate
(e.g., median M and a).  This highlights the information gain from using the
full posterior distribution.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    alpha_of, a_thr, t_inst, delta_a_BN,
    T_AGE_IRAS, T_AGE_M33,
)


def plugin_exclusion(M_msun, a_star, mu_grid, fa_grid, t_age):
    """Compute exclusion indicator for a single (M, a) point."""
    alpha = alpha_of(M_msun, mu_grid)
    a_t = a_thr(alpha)
    cond1 = (a_star > a_t) & (alpha < 0.5)
    t = t_inst(M_msun, a_star, alpha)
    cond2 = t < t_age
    active = cond1 & cond2
    P_ex = np.zeros((len(mu_grid), len(fa_grid)), dtype=np.float64)
    for j, fa in enumerate(fa_grid):
        da = delta_a_BN(alpha, fa)
        cond3 = da > (a_star - a_t)
        P_ex[:, j] = (active & cond3).astype(float)
    return P_ex


def main():
    samples_iras = np.loadtxt('data/IRAS_09149-6206_samples.dat')
    samples_m33 = np.loadtxt('data/M33_X-7_samples.dat')
    mu_grid = np.load('outputs/mu_grid.npy')
    fa_grid = np.load('outputs/fa_grid.npy')

    # Median point estimates
    M_iras_med = np.median(samples_iras[:, 0])
    a_iras_med = np.median(samples_iras[:, 1])
    M_m33_med = np.median(samples_m33[:, 0])
    a_m33_med = np.median(samples_m33[:, 1])

    P_plugin_iras = plugin_exclusion(M_iras_med, a_iras_med, mu_grid, fa_grid, T_AGE_IRAS)
    P_plugin_m33 = plugin_exclusion(M_m33_med, a_m33_med, mu_grid, fa_grid, T_AGE_M33)

    P_bayes_iras = np.load('outputs/P_ex_IRAS.npy')
    P_bayes_m33 = np.load('outputs/P_ex_M33.npy')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mu, P_b, P_p, title in zip(
        axes,
        [mu_grid, mu_grid],
        [P_bayes_iras, P_bayes_m33],
        [P_plugin_iras, P_plugin_m33],
        ['IRAS 09149-6206', 'M33 X-7']
    ):
        # 1D slice at f_a = 1e20 GeV (negligible self-interactions)
        ax.plot(mu, P_b[:, -1], lw=2, label='Full Bayesian')
        ax.plot(mu, P_p[:, -1], lw=2, ls='--', label='Plug-in (median M, a)')
        ax.axhline(0.95, color='k', ls=':', lw=1, label='95% credibility')
        ax.set_xscale('log')
        ax.set_xlabel(r'ULB mass $\mu$ [eV]')
        ax.set_ylabel(r'Exclusion probability')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(mu.min(), mu.max())
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig('report/images/fig_validation_plugin_vs_bayesian.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Validation plot saved.")


if __name__ == '__main__':
    main()
