"""
Main constraint analysis (vectorised).
"""
from __future__ import annotations
import os, json, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from superradiance import (
    alpha_coupling, omega_H_geom, horizon_radius_geom,
    spin_down_target_a, n_bosenova, n_extract_required,
    SEC_PER_YR, M_PL_RED, M_SUN, G, C, EV, HBAR, GEV_PER_EV,
)

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(OUTDIR, exist_ok=True)

SOURCES = {
    'M33_X-7': {
        'file': 'data/M33_X-7_samples.dat',
        'tau_BH_yr': 3.0e6,
        'tau_BH_yr_low': 1.0e6,
        'tau_BH_yr_high': 1.0e7,
        'mu_grid': np.geomspace(1e-13, 5e-11, 160),
    },
    'IRAS_09149-6206': {
        'file': 'data/IRAS_09149-6206_samples.dat',
        'tau_BH_yr': 4.5e7,
        'tau_BH_yr_low': 1.0e7,
        'tau_BH_yr_high': 1.0e9,
        'mu_grid': np.geomspace(1e-21, 1e-17, 160),
    },
}

LEVELS = [(1, 1, 2), (2, 2, 3), (3, 3, 4)]
ALPHA_MAX = 0.5
N_EFOLDS = 180.0  # ln(N_max) ~ 180 for full spin-down


def load_samples(path):
    arr = np.loadtxt(path, comments='#')
    return arr[:, 0], np.clip(arr[:, 1], -0.999, 0.999)


def gamma_geom_vec(M, a, mu, l, m, n):
    """Vectorised SR rate (Gamma * M) in geometric units. M (N,), a (N,), mu (G,).
    Returns (G, N) array."""
    from math import factorial
    M_b = M[None, :]
    a_b = a[None, :]
    mu_b = mu[:, None]
    al = alpha_coupling(M_b, mu_b)
    rp = 1.0 + np.sqrt(1.0 - a_b**2)
    omH_M = a_b / (2.0 * rp)
    omR_M = al * (1.0 - al**2 / (2.0 * n**2))
    if l == 1 and m == 1:
        return (1.0 / 24.0) * al**8 * (a_b - 2.0 * al * rp)
    Cnl = 2.0**(4*l + 1) * factorial(n + l) / (n**(2*l + 4) * factorial(max(n - l - 1, 0)))
    Cnl *= (factorial(l) / (factorial(2*l) * factorial(2*l + 1)))**2
    prod = 1.0
    for k in range(1, l + 1):
        prod = prod * (k**2 * (1.0 - a_b**2) + (a_b * m - 2.0 * rp * omR_M)**2)
    Cnl_arr = Cnl * prod
    diff = m * omH_M - omR_M
    return 2.0 * rp * diff * Cnl_arr * al**(4*l + 4)


def excluded_grid(M, a, mu_grid, tau_BH_s, fa_GeV=None):
    """Vectorised exclusion fraction over posterior samples for each mu in grid.
    Returns array P_excl(mu) of shape (G,)."""
    G_, N = len(mu_grid), len(M)
    excl_total = np.zeros((G_, N), dtype=bool)
    M_b = M[None, :]
    a_b = a[None, :]
    mu_b = mu_grid[:, None]
    al = alpha_coupling(M_b, mu_b)
    tg = 4.9255e-6 * M_b  # gravitational time in seconds, M in Msun
    for (l, m, n) in LEVELS:
        rp = 1.0 + np.sqrt(1.0 - a_b**2)
        omH_M = a_b / (2.0 * rp)
        omR_M = al * (1.0 - al**2 / (2.0 * n**2))
        sr_cond = omR_M < m * omH_M
        g_geom = gamma_geom_vec(M, a, mu_grid, l, m, n)
        valid = (al < ALPHA_MAX) & (g_geom > 0) & sr_cond
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_SR = np.where(valid, tg / g_geom, np.inf)
        time_ok = (tau_SR * N_EFOLDS) < tau_BH_s
        if fa_GeV is not None:
            # Bosenova check
            a_target = spin_down_target_a(M_b, mu_b, n=n, m=m)
            a_t = np.where(np.isnan(a_target), a_b, a_target)
            N_ext = n_extract_required(M_b, a_b, a_t, mu_b, n=n, m=m)
            N_bose = n_bosenova(M_b, mu_b, fa_GeV, alpha=al, n=n)
            bose_ok = (N_bose > N_ext) & np.isfinite(N_ext) & (N_ext > 0)
            this = valid & time_ok & bose_ok
        else:
            this = valid & time_ok
        excl_total = excl_total | this
    return excl_total.mean(axis=1)


def run_per_source(name, cfg):
    M, a = load_samples(cfg['file'])
    print(f'[{name}] N samples = {len(M)}, M median={np.median(M):.3e} Msun, a median={np.median(a):.3f}')
    tau_BH_s = cfg['tau_BH_yr'] * SEC_PER_YR
    mu_grid = cfg['mu_grid']
    Pexcl = excluded_grid(M, a, mu_grid, tau_BH_s)
    Pexcl_lo = excluded_grid(M, a, mu_grid, cfg['tau_BH_yr_low'] * SEC_PER_YR)
    Pexcl_hi = excluded_grid(M, a, mu_grid, cfg['tau_BH_yr_high'] * SEC_PER_YR)

    def intervals(p, level):
        mask = p >= level
        if not mask.any():
            return []
        runs, in_run, start = [], False, None
        for i, mk in enumerate(mask):
            if mk and not in_run:
                in_run = True; start = i
            elif not mk and in_run:
                runs.append((mu_grid[start], mu_grid[i-1])); in_run = False
        if in_run:
            runs.append((mu_grid[start], mu_grid[-1]))
        return runs

    excl_95 = intervals(Pexcl, 0.95)
    excl_68 = intervals(Pexcl, 0.68)
    print(f'[{name}] 95% excluded intervals (eV): {excl_95}')
    print(f'[{name}] 68% excluded intervals (eV): {excl_68}')
    return dict(M=M, a=a, mu_grid=mu_grid, Pexcl=Pexcl,
                Pexcl_lo=Pexcl_lo, Pexcl_hi=Pexcl_hi,
                excl_95=excl_95, excl_68=excl_68)


def run_2d_fa(name, cfg, mu_grid=None, fa_grid=None, subsample=2000):
    M, a = load_samples(cfg['file'])
    if subsample and len(M) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(M), subsample, replace=False)
        M, a = M[idx], a[idx]
    tau_BH_s = cfg['tau_BH_yr'] * SEC_PER_YR
    if mu_grid is None:
        mu_grid = cfg['mu_grid']
    if fa_grid is None:
        fa_grid = np.geomspace(1e10, 1e20, 80)
    Z = np.zeros((len(fa_grid), len(mu_grid)))
    for i, fa in enumerate(fa_grid):
        Z[i] = excluded_grid(M, a, mu_grid, tau_BH_s, fa_GeV=fa)
    return dict(mu_grid=mu_grid, fa_grid=fa_grid, Pexcl=Z)


def main():
    summary = {}
    grids_npz = {}
    t0 = time.time()
    for name, cfg in SOURCES.items():
        t1 = time.time()
        res = run_per_source(name, cfg)
        summary[name] = {
            'N_samples': int(len(res['M'])),
            'M_median_Msun': float(np.median(res['M'])),
            'M_mean_Msun': float(np.mean(res['M'])),
            'M_std_Msun': float(np.std(res['M'])),
            'a_median': float(np.median(res['a'])),
            'a_mean': float(np.mean(res['a'])),
            'a_std': float(np.std(res['a'])),
            'tau_BH_yr': cfg['tau_BH_yr'],
            'tau_BH_yr_range': [cfg['tau_BH_yr_low'], cfg['tau_BH_yr_high']],
            'excl_95_eV': [[float(x), float(y)] for x, y in res['excl_95']],
            'excl_68_eV': [[float(x), float(y)] for x, y in res['excl_68']],
            'mu_grid_range_eV': [float(cfg['mu_grid'][0]), float(cfg['mu_grid'][-1])],
        }
        grids_npz[f'{name}_mu'] = res['mu_grid']
        grids_npz[f'{name}_Pexcl'] = res['Pexcl']
        grids_npz[f'{name}_Pexcl_lo'] = res['Pexcl_lo']
        grids_npz[f'{name}_Pexcl_hi'] = res['Pexcl_hi']
        grids_npz[f'{name}_M'] = res['M']
        grids_npz[f'{name}_a'] = res['a']
        print(f'  per-source done in {time.time()-t1:.1f}s')

        t1 = time.time()
        res2d = run_2d_fa(name, cfg)
        grids_npz[f'{name}_fa_grid'] = res2d['fa_grid']
        grids_npz[f'{name}_2D_Pexcl'] = res2d['Pexcl']
        # f_a lower bound at peak mu (where 1D P_excl is maximal)
        peak = np.argmax(res['Pexcl'])
        if res['Pexcl'][peak] > 0.95:
            col = res2d['Pexcl'][:, peak] if peak < res2d['Pexcl'].shape[1] else None
            # find smallest f_a for which still P_excl > 0.95
            if col is not None:
                idx = np.where(col >= 0.95)[0]
                fa_min = float(res2d['fa_grid'][idx[0]]) if len(idx) else None
            else:
                fa_min = None
            summary[name]['fa_lower_bound_at_peak_GeV'] = fa_min
            summary[name]['mu_peak_eV'] = float(res['mu_grid'][peak])
        print(f'  2D done in {time.time()-t1:.1f}s')

    np.savez(os.path.join(OUTDIR, 'exclusion_grids.npz'), **grids_npz)
    with open(os.path.join(OUTDIR, 'summary_constraints.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nTotal: {time.time()-t0:.1f}s')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
