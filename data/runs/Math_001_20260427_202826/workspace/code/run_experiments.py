"""Run all VOS experiments and produce figures + JSON outputs."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vos_framework import (LassoProblem, History,
                           proximal_gradient, fista, fista_restart,
                           heavy_ball_smooth, nag_sc_smooth,
                           admm_lasso,
                           integrate_nag_ode, integrate_nag_sc_ode,
                           lyapunov_nag_weak, lyapunov_nag_sc,
                           lyapunov_admm)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data():
    d = np.load(DATA / "complex_optimization_data.npy", allow_pickle=True).item()
    return d["A"], d["b"], d["x_true"], d.get("meta", "")


# =====================================================================
# Phase 1: Data overview figure
# =====================================================================
def fig_data_overview(A, b, x_true):
    sv = np.linalg.svd(A, compute_uv=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(sv, lw=1.6, color="C0")
    axes[0].set_title("Singular values of A (cond ≈ {:.1f})".format(sv.max() / sv.min()))
    axes[0].set_xlabel("index"); axes[0].set_ylabel("σ_i")
    axes[1].stem(np.arange(len(x_true)), x_true,
                 markerfmt=" ", basefmt=" ", linefmt="C2-")
    axes[1].set_title("Ground-truth sparse coefficients (nnz = {})".format((x_true != 0).sum()))
    axes[1].set_xlabel("coordinate"); axes[1].set_ylabel("x_true")
    axes[2].hist(b, bins=40, color="C3", alpha=0.85)
    axes[2].set_title("Response b distribution")
    axes[2].set_xlabel("b_i"); axes[2].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(IMG / "data_overview.png")
    plt.close(fig)
    return dict(L=float(sv.max() ** 2),
                sigma_max=float(sv.max()),
                sigma_min=float(sv.min()),
                cond=float(sv.max() / sv.min()),
                nnz=int((x_true != 0).sum()))


# =====================================================================
# Phase 2: Lasso comparison (full non-smooth Lasso)
# =====================================================================
def run_lasso_comparison(A, b, x_true):
    p = A.shape[1]
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))    # standard scaling
    prob = LassoProblem(A=A, b=b, lam=lam)
    print(f"Lipschitz L = {prob.L:.3f}, lambda = {lam:.4f}")

    x0 = np.zeros(p)

    # Reference solution: long FISTA-restart run
    x_star, _ = fista_restart(prob, x0, n_iters=4000)
    F_star = prob.F(x_star)
    print(f"F* (reference) = {F_star:.6f}")

    n_iters = 800
    t0 = time.perf_counter(); x_ista, h_ista = proximal_gradient(prob, x0, n_iters, x_ref=x_star); t_ista = time.perf_counter() - t0
    t0 = time.perf_counter(); x_fista, h_fista = fista(prob, x0, n_iters, x_ref=x_star);          t_fista = time.perf_counter() - t0
    t0 = time.perf_counter(); x_vnag,  h_vnag  = fista_restart(prob, x0, n_iters, x_ref=x_star); t_vnag  = time.perf_counter() - t0
    t0 = time.perf_counter(); x_admm,  h_admm, admm_state = admm_lasso(prob, x0, n_iters, rho=1.0, x_ref=x_star); t_admm = time.perf_counter() - t0

    timings = dict(ISTA=t_ista, FISTA=t_fista, VOS_NAG=t_vnag, VOS_ADMM=t_admm)

    np.savez(OUT / "lasso_comparison.npz",
             x_star=x_star, F_star=F_star,
             ISTA_F=np.array(h_ista.F), ISTA_err=np.array(h_ista.err),
             FISTA_F=np.array(h_fista.F), FISTA_err=np.array(h_fista.err),
             VOS_NAG_F=np.array(h_vnag.F), VOS_NAG_err=np.array(h_vnag.err),
             VOS_ADMM_F=np.array(h_admm.F), VOS_ADMM_err=np.array(h_admm.err),
             x_ista=x_ista, x_fista=x_fista, x_vnag=x_vnag, x_admm=x_admm)

    # Figure: convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h, c in [("ISTA", h_ista, "C0"),
                       ("FISTA (NAG)", h_fista, "C1"),
                       ("VOS-NAG (gradient restart)", h_vnag, "C2"),
                       ("VOS-ADMM", h_admm, "C3")]:
        gap = np.maximum(np.array(h.F) - F_star, 1e-16)
        axes[0].semilogy(gap, label=name, color=c, lw=1.6)
    axes[0].set_xlabel("iteration k"); axes[0].set_ylabel("F(x_k) - F*")
    axes[0].set_title("Objective sub-optimality")
    axes[0].legend(loc="best", fontsize=9)

    for name, h, c in [("ISTA", h_ista, "C0"),
                       ("FISTA (NAG)", h_fista, "C1"),
                       ("VOS-NAG", h_vnag, "C2"),
                       ("VOS-ADMM", h_admm, "C3")]:
        e = np.maximum(np.array(h.err), 1e-16)
        axes[1].semilogy(e, label=name, color=c, lw=1.6)
    axes[1].set_xlabel("iteration k"); axes[1].set_ylabel("||x_k - x*||")
    axes[1].set_title("Iterate distance to reference x*")
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "convergence_lasso.png")
    plt.close(fig)

    # Figure: support recovery
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, (name, x_) in zip(axes.flat, [("ISTA", x_ista), ("FISTA", x_fista),
                                          ("VOS-NAG", x_vnag), ("VOS-ADMM", x_admm)]):
        ax.stem(np.arange(p), x_true, markerfmt=" ", basefmt=" ", linefmt="C2-",
                label="x_true")
        ax.plot(np.arange(p), x_, ".", color="C3", ms=2.0, label=name)
        ax.set_title(f"{name}: ||x - x*||₂={np.linalg.norm(x_ - x_star):.3e}, "
                     f"nnz(>1e-3)={int((np.abs(x_) > 1e-3).sum())}")
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "support_recovery.png")
    plt.close(fig)

    summary = dict(F_star=F_star,
                   lam=lam,
                   L=prob.L,
                   n_iters=n_iters,
                   timings=timings,
                   final_F={k: float(v) for k, v in
                            dict(ISTA=h_ista.F[-1], FISTA=h_fista.F[-1],
                                 VOS_NAG=h_vnag.F[-1], VOS_ADMM=h_admm.F[-1]).items()},
                   final_err={k: float(np.linalg.norm(x - x_star)) for k, x in
                              dict(ISTA=x_ista, FISTA=x_fista,
                                   VOS_NAG=x_vnag, VOS_ADMM=x_admm).items()},
                   final_nnz={k: int((np.abs(x) > 1e-3).sum()) for k, x in
                              dict(ISTA=x_ista, FISTA=x_fista,
                                   VOS_NAG=x_vnag, VOS_ADMM=x_admm,
                                   x_true=x_true).items()})
    with open(OUT / "lasso_comparison.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    return summary, prob, x_star, dict(ista=h_ista, fista=h_fista, vnag=h_vnag,
                                        admm=h_admm, admm_state=admm_state)


# =====================================================================
# Phase 3: Strongly-convex smooth experiment + Lyapunov verification
# =====================================================================
def run_strongly_convex(A, b, x_true):
    """Use the smooth Tikhonov-regularized least squares
       f(x) = 0.5 || A x - b ||^2 + (mu/2) || x ||^2.
    Yields a μ-strongly-convex smooth problem on which we can numerically
    verify exponential decay of the strong Lyapunov function from VOS-NAG.
    """
    n, p = A.shape
    mu = 1.0
    # full Hessian: A^T A + mu I
    L = float(np.linalg.svd(A, compute_uv=False).max() ** 2 + mu)
    Atb = A.T @ b

    def f(x):
        r = A @ x - b
        return 0.5 * float(r @ r) + 0.5 * mu * float(x @ x)

    def grad(x):
        return A.T @ (A @ x - b) + mu * x

    # Closed-form minimizer: solve (A^T A + mu I) x = A^T b
    AAt = A @ A.T
    M = np.eye(n) + AAt / mu
    Lchol = np.linalg.cholesky(M)
    # x* = (1/mu) (Atb - A^T M^{-1} A Atb / mu)
    v = A @ Atb
    w = np.linalg.solve(Lchol.T, np.linalg.solve(Lchol, v))
    x_star = (Atb - A.T @ w / mu) / mu
    f_star = f(x_star)

    x0 = np.zeros(p)
    n_iters = 400

    x_gd, h_gd = proximal_gradient(LassoProblem(A=A, b=b, lam=0.0, L=L, mu=mu)
                                    .__class__(A=A, b=b, lam=0.0, L=L, mu=mu),
                                    x0, 1, x_ref=x_star)  # placeholder so type checker ok
    # We re-implement plain GD here to avoid the proximal soft-threshold:
    def gradient_descent(grad, x0, L, n_iters, x_ref):
        x = x0.copy()
        h = History()
        for k in range(n_iters):
            x = x - grad(x) / L
            h.time.append(k)
            h.F.append(float(f(x)))
            h.err.append(float(np.linalg.norm(x - x_ref)))
        return x, h

    x_gd, h_gd  = gradient_descent(grad, x0, L, n_iters, x_star)
    x_hb, h_hb  = heavy_ball_smooth(grad, x0, L, mu, n_iters, x_ref=x_star, f=f)
    x_nag, h_nag = nag_sc_smooth(grad, x0, L, mu, n_iters, x_ref=x_star, f=f)

    # ODE simulation for the strongly convex dynamics
    _, h_ode = integrate_nag_sc_ode(grad, x0, t_max=12.0, dt=1e-3, mu=mu,
                                    f=f, x_ref=x_star)

    # Lyapunov function along NAG-SC discrete trajectory
    s = 1.0 / L
    momentum = (1.0 - np.sqrt(mu * s)) / (1.0 + np.sqrt(mu * s))
    x_prev = x0.copy()
    x = x0.copy()
    Es = []
    err_seq = []
    for k in range(n_iters):
        y = x + momentum * (x - x_prev)
        x_new = y - s * grad(y)
        # discrete-time velocity proxy v ≈ (x_new - x)/sqrt(s)
        v = (x_new - x) / np.sqrt(s)
        Es.append(lyapunov_nag_sc(x_new, v, x_star, mu, f, f_star))
        err_seq.append(float(np.linalg.norm(x_new - x_star)))
        x_prev, x = x, x_new

    # Plot 1: convergence on smooth strongly-convex problem
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h, c in [("Gradient descent", h_gd, "C0"),
                       ("Heavy-ball (Polyak)", h_hb, "C3"),
                       ("VOS-NAG-SC (discrete)", h_nag, "C2")]:
        gap = np.maximum(np.array(h.F) - f_star, 1e-16)
        axes[0].semilogy(gap, label=name, color=c, lw=1.6)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("f(x_k) - f*")
    axes[0].set_title(f"Smooth μ-SC problem (μ={mu}, L={L:.1f}, κ={L/mu:.1f})")
    axes[0].legend()

    # ODE in continuous time
    axes[1].semilogy(h_ode.time, np.maximum(np.array(h_ode.F) - f_star, 1e-16),
                     color="C4", lw=1.6, label="NAG-SC ODE: ẍ + 2√μ ẋ + ∇f = 0")
    axes[1].set_xlabel("continuous time t"); axes[1].set_ylabel("f(x(t)) - f*")
    axes[1].set_title("Continuous-time NAG-SC dynamics")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMG / "strongly_convex.png")
    plt.close(fig)

    # Plot 2: Lyapunov function decay
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    Es_arr = np.array(Es)
    Es_arr = np.maximum(Es_arr, 1e-30)
    axes[0].semilogy(Es_arr, color="C2", lw=1.8,
                     label="E_k = f-f* + (μ/2)||x-x*||² + ½||v+√μ(x-x*)||²")
    # theoretical envelope:  E_k <= E_0 (1 - sqrt(mu/L))^k
    rate = 1.0 - np.sqrt(mu / L)
    env = Es_arr[0] * np.power(rate, np.arange(len(Es_arr)))
    axes[0].semilogy(env, "--", color="k", lw=1.0,
                     label=f"theoretical: E₀·(1-√(μ/L))^k, ρ={rate:.4f}")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("Lyapunov E_k")
    axes[0].set_title("Strong Lyapunov function (VOS-NAG-SC)")
    axes[0].legend(fontsize=9)

    err_arr = np.maximum(np.array(err_seq), 1e-16)
    axes[1].semilogy(err_arr, color="C2", lw=1.8, label="||x_k - x*||")
    env2 = err_arr[0] * np.power(np.sqrt(rate), np.arange(len(err_arr)))
    axes[1].semilogy(env2, "--", color="k", lw=1.0,
                     label=f"||x₀-x*||·(1-√(μ/L))^(k/2)")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("||x_k - x*||")
    axes[1].set_title("Iterate convergence vs. theoretical rate")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "lyapunov_strong_decay.png")
    plt.close(fig)

    # Numerical fit of empirical exponential rate
    K = len(Es_arr)
    if Es_arr[K // 2] > 0:
        emp_rate = (Es_arr[3 * K // 4] / Es_arr[K // 4]) ** (1.0 / (3 * K // 4 - K // 4))
    else:
        emp_rate = float('nan')

    sc_summary = dict(mu=mu, L=L, kappa=L / mu, f_star=f_star,
                      theoretical_rate=rate,
                      empirical_rate=float(emp_rate),
                      final_err_GD=float(np.linalg.norm(x_gd - x_star)),
                      final_err_HB=float(np.linalg.norm(x_hb - x_star)),
                      final_err_NAG=float(np.linalg.norm(x_nag - x_star)))
    with open(OUT / "strongly_convex_summary.json", "w") as fp:
        json.dump(sc_summary, fp, indent=2)

    np.savez(OUT / "lyapunov_trace.npz", E=Es_arr, err=err_arr,
             theoretical_rate=rate)
    return sc_summary


# =====================================================================
# Phase 4: ODE-iterates equivalence (Su-Boyd-Candes)
# =====================================================================
def run_ode_vs_iterates(A, b):
    """Compare continuous NAG-ODE x(t) with FISTA iterates x_k.
    Su-Boyd-Candes show  x_k ≈ x(k √s).
    """
    n, p = A.shape
    # use a small smooth problem for a clean ODE comparison
    L = float(np.linalg.svd(A, compute_uv=False).max() ** 2)

    def f(x):
        r = A @ x - b
        return 0.5 * float(r @ r)

    def grad(x):
        return A.T @ (A @ x - b)

    # Reference minimum (least-norm solution)
    x_star = np.linalg.lstsq(A, b, rcond=None)[0]
    f_star = f(x_star)

    x0 = np.zeros(p)
    s = 1.0 / L
    n_iters = 200

    # FISTA iterates (no regularization -> reduces to plain Nesterov)
    prob = LassoProblem(A=A, b=b, lam=0.0, L=L)
    x_fista, h_fista = fista(prob, x0, n_iters, x_ref=x_star)
    t_fista = np.sqrt(s) * np.arange(1, n_iters + 1)

    # ODE  ẍ + (3/t) ẋ + ∇f = 0
    _, h_ode = integrate_nag_ode(grad, x0,
                                 t_max=t_fista[-1] * 1.1,
                                 dt=s / 4.0,         # very fine
                                 a=3.0,
                                 f=f, x_ref=x_star,
                                 t_start=np.sqrt(s))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    gap_fista = np.maximum(np.array(h_fista.F) - f_star, 1e-12)
    gap_ode   = np.maximum(np.array(h_ode.F) - f_star, 1e-12)
    ax.loglog(t_fista, gap_fista, "o-", ms=3, lw=1.4, color="C1",
              label="FISTA iterate gap at t=k√s")
    ax.loglog(h_ode.time, gap_ode, "-", lw=1.6, color="C4",
              label="ODE  ẍ+(3/t)ẋ+∇f=0  trajectory")
    # 1/t^2 reference
    tt = np.geomspace(t_fista[1], t_fista[-1], 50)
    ax.loglog(tt, gap_fista[5] * (t_fista[5] / tt) ** 2, "--", color="k", lw=1.0,
              label="$O(1/t^2)$ reference")
    ax.set_xlabel("continuous time t (or k√s)")
    ax.set_ylabel("f - f*")
    ax.set_title("VOS continuous-time / discrete-time correspondence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "ode_vs_iterates.png")
    plt.close(fig)

    return dict(t_max=float(t_fista[-1]),
                f_star=float(f_star),
                fista_final_gap=float(gap_fista[-1]),
                ode_final_gap=float(gap_ode[-1]))


# =====================================================================
# Phase 5: ADMM Lyapunov empirical verification
# =====================================================================
def run_admm_lyapunov(prob, x_star, x0):
    """Run ADMM and track the Boyd-Parikh Lyapunov-type quantity
        V_k = (1/(2ρ))||u_k - u*||² + (ρ/2)||z_k - z*||²
    """
    rho = 1.0
    A, b = prob.A, prob.b
    p = A.shape[1]

    # First, run ADMM long to get fixed-point reference (z*, u*)
    z_ref, _, ref_state = admm_lasso(prob, x0, n_iters=4000, rho=rho)
    z_star = ref_state["z"]
    u_star = ref_state["u"]

    # Re-run, recording Lyapunov each iter
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(p)
    AAt = A @ A.T
    M = np.eye(A.shape[0]) + AAt / rho
    Lchol = np.linalg.cholesky(M)
    Atb = A.T @ b
    n_iters = 400
    Vs = []
    F_gap = []
    F_star = prob.F(z_star)
    for k in range(n_iters):
        rhs = Atb + rho * (z - u)
        v = A @ rhs
        w = np.linalg.solve(Lchol.T, np.linalg.solve(Lchol, v))
        x = (rhs - A.T @ w / rho) / rho
        z = prob.soft_threshold(x + u, prob.lam / rho)
        u = u + (x - z)
        Vs.append(lyapunov_admm(x, z, u, x, z_star, u_star, rho))
        F_gap.append(prob.F(z) - F_star)

    Vs = np.array(Vs)
    F_gap = np.maximum(np.array(F_gap), 1e-16)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(np.maximum(Vs, 1e-16), color="C3", lw=1.7,
                     label="V_k = ρ||z-z*||² + (1/ρ)||u-u*||²")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("V_k")
    axes[0].set_title("ADMM Lyapunov function (Boyd-Parikh)")
    axes[0].legend()

    axes[1].semilogy(F_gap, color="C3", lw=1.7, label="F(z_k) - F*")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("objective gap")
    axes[1].set_title("ADMM objective gap (composite Lasso)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMG / "admm_lyapunov.png")
    plt.close(fig)

    np.savez(OUT / "admm_lyapunov.npz", V=Vs, F_gap=F_gap)
    return dict(V_initial=float(Vs[0]), V_final=float(Vs[-1]),
                F_gap_initial=float(F_gap[0]), F_gap_final=float(F_gap[-1]))


# =====================================================================
# Main
# =====================================================================
def main():
    print(">> Loading data...")
    A, b, x_true, meta = load_data()
    print("   meta:", meta)
    overview = fig_data_overview(A, b, x_true)
    print("   data overview:", overview)
    with open(OUT / "data_overview.json", "w") as fp:
        json.dump(overview, fp, indent=2)

    print(">> Lasso comparison (ISTA / FISTA / VOS-NAG / VOS-ADMM)...")
    summary, prob, x_star, hists = run_lasso_comparison(A, b, x_true)
    print("   final F:", summary["final_F"])
    print("   final err:", summary["final_err"])

    print(">> Strongly convex smooth experiment + Lyapunov...")
    sc_summary = run_strongly_convex(A, b, x_true)
    print("   ", sc_summary)

    print(">> ODE vs iterates correspondence...")
    ode_summary = run_ode_vs_iterates(A, b)
    print("   ", ode_summary)

    print(">> ADMM Lyapunov empirical verification...")
    admm_summary = run_admm_lyapunov(prob, x_star, np.zeros(A.shape[1]))
    print("   ", admm_summary)

    final = dict(data_overview=overview,
                 lasso_comparison=summary,
                 strongly_convex=sc_summary,
                 ode_correspondence=ode_summary,
                 admm_lyapunov=admm_summary)
    with open(OUT / "all_results.json", "w") as fp:
        json.dump(final, fp, indent=2)
    print(">> Done.")


if __name__ == "__main__":
    main()
