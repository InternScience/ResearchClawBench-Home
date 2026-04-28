"""
Workflow #3 — Autonomous synthesis optimization with Bayesian Optimization.

Setup (from M-AI-Synth block 3):
  Search space: T in [200, 500] (degC), t in [10, 30] (min)
  Hidden ground-truth optimum: (T*, t*) = (350, 20)
  Observation noise: sigma = 0.1
  Success threshold (in objective units): 10.0

Oracle:
  We define a noisy bell-shaped synthesis-yield oracle centred at
  (T*, t*):
      f(T, t) = 10 * exp( -((T - T*)/(0.4*span_T))^2
                          -((t - t*)/(0.4*span_t))^2 ) + N(0, sigma^2)
  i.e. peak yield = 10 (= the dataset's threshold) at (T*, t*),
  and yield = 0 far from the optimum. The peak value matching the
  threshold makes the threshold a meaningful "success" criterion.

Method:
  * Bayesian Optimization with a Gaussian Process (Matern 2.5) surrogate
    and Expected Improvement acquisition.
  * Baseline: uniform random search.
  * Evaluation: simple regret (best - current_best) and cumulative
    successes (observations >= threshold * 0.95) over the budget.

We average over R independent seeds (R=20) for statistical credibility.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def build_oracle(T_target, t_target, T_range, t_range, sigma):
    span_T = T_range[1] - T_range[0]
    span_t = t_range[1] - t_range[0]

    def oracle(T, t, rng=None):
        f = 10.0 * np.exp(
            -((T - T_target) / (0.4 * span_T)) ** 2
            -((t - t_target) / (0.4 * span_t)) ** 2
        )
        if rng is not None:
            f = f + rng.normal(0.0, sigma, size=np.shape(f))
        return f

    return oracle


def expected_improvement(mu, std, f_best, xi=0.01):
    std = np.maximum(std, 1e-9)
    z = (mu - f_best - xi) / std
    ei = (mu - f_best - xi) * norm.cdf(z) + std * norm.pdf(z)
    ei[std < 1e-8] = 0.0
    return ei


def run_bo(seed, oracle, T_range, t_range, n_init=4, budget=30):
    rng = np.random.default_rng(seed)
    Xs = []
    ys = []
    # Latin-hypercube-ish init
    for _ in range(n_init):
        T = rng.uniform(*T_range)
        t = rng.uniform(*t_range)
        y = float(oracle(T, t, rng=rng))
        Xs.append([T, t]); ys.append(y)
    grid_T = np.linspace(T_range[0], T_range[1], 51)
    grid_t = np.linspace(t_range[0], t_range[1], 51)
    GT, Gt = np.meshgrid(grid_T, grid_t)
    grid = np.column_stack([GT.ravel(), Gt.ravel()])
    bestcurve = []
    for it in range(budget - n_init):
        Xa = np.array(Xs); ya = np.array(ys)
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=[100.0, 5.0],
                           length_scale_bounds=(1e-1, 1e3), nu=2.5)
                  + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2,
                                      random_state=seed)
        gp.fit(Xa, ya)
        mu, std = gp.predict(grid, return_std=True)
        ei = expected_improvement(mu, std, ya.max())
        idx = int(np.argmax(ei))
        T, t = grid[idx]
        y = float(oracle(T, t, rng=rng))
        Xs.append([T, t]); ys.append(y)
        bestcurve.append(max(ys))
    # bestcurve length = budget - n_init; pre-pend init bests
    init_bests = np.maximum.accumulate(ys[:n_init]).tolist()
    full = init_bests + bestcurve
    return np.array(Xs), np.array(ys), np.array(full)


def run_random(seed, oracle, T_range, t_range, budget=30):
    rng = np.random.default_rng(seed + 9999)
    Ts = rng.uniform(T_range[0], T_range[1], budget)
    ts = rng.uniform(t_range[0], t_range[1], budget)
    ys = oracle(Ts, ts, rng=rng)
    Xs = np.column_stack([Ts, ts])
    full = np.maximum.accumulate(ys)
    return Xs, ys, full


def main():
    cfg = json.loads((OUT / "data_summary.json").read_text())["autonomous_optimization"]
    T_range = tuple(cfg["T_range"])
    t_range = tuple(cfg["t_range"])
    T_target = cfg["T_target"]
    t_target = cfg["t_target"]
    sigma = cfg["noise"]
    threshold = cfg["threshold"]   # peak yield = 10

    oracle = build_oracle(T_target, t_target, T_range, t_range, sigma)

    # Sanity: f(T*, t*) = 10 exactly (no noise).
    peak = float(oracle(T_target, t_target))
    print("noiseless peak:", peak)

    R = 20
    budget = 30
    bo_curves, rs_curves = [], []
    bo_xs, rs_xs = [], []
    bo_ys, rs_ys = [], []
    for seed in range(R):
        Xb, yb, cb = run_bo(seed, oracle, T_range, t_range, n_init=4, budget=budget)
        Xr, yr, cr = run_random(seed, oracle, T_range, t_range, budget=budget)
        bo_curves.append(cb); rs_curves.append(cr)
        bo_xs.append(Xb); rs_xs.append(Xr)
        bo_ys.append(yb); rs_ys.append(yr)
    bo_curves = np.stack(bo_curves)
    rs_curves = np.stack(rs_curves)

    # Success criterion: best_so_far >= 0.95 * threshold (i.e. >= 9.5)
    success_thr = 0.95 * threshold

    metrics = {
        "config": cfg,
        "noiseless_peak": peak,
        "budget": budget,
        "n_seeds": R,
        "BO": {
            "best_mean_at_end": float(bo_curves[:, -1].mean()),
            "best_std_at_end":  float(bo_curves[:, -1].std()),
            "first_hit_iter_mean": float(np.mean([
                int(np.argmax(c >= success_thr)) if (c >= success_thr).any() else budget
                for c in bo_curves
            ])),
            "success_rate": float(np.mean(bo_curves[:, -1] >= success_thr)),
            "mean_curve": bo_curves.mean(axis=0).tolist(),
            "std_curve": bo_curves.std(axis=0).tolist(),
        },
        "Random": {
            "best_mean_at_end": float(rs_curves[:, -1].mean()),
            "best_std_at_end":  float(rs_curves[:, -1].std()),
            "first_hit_iter_mean": float(np.mean([
                int(np.argmax(c >= success_thr)) if (c >= success_thr).any() else budget
                for c in rs_curves
            ])),
            "success_rate": float(np.mean(rs_curves[:, -1] >= success_thr)),
            "mean_curve": rs_curves.mean(axis=0).tolist(),
            "std_curve": rs_curves.std(axis=0).tolist(),
        },
    }
    (OUT / "autonomous_optimization_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(OUT / "autonomous_optimization_runs.npz",
             bo_curves=bo_curves, rs_curves=rs_curves,
             bo_X_seed0=bo_xs[0], bo_y_seed0=bo_ys[0],
             rs_X_seed0=rs_xs[0], rs_y_seed0=rs_ys[0])
    print(json.dumps({k: metrics[k] for k in ("BO", "Random") if k in metrics},
                     default=lambda x: list(x)[:3] if isinstance(x, list) else x,
                     indent=2)[:1500])


if __name__ == "__main__":
    main()
