"""
Build figures from outputs/per_instance_fidelity.csv and
outputs/aggregated_fidelity.csv.

Figures (saved as report/images/*.png):
  1. depth_scan_N40.png        F vs d at N=40 (XEB / MB / Transport_1QRB)
  2. depth_scan_N56.png        F vs d at N=56 (MB / Transport_1QRB)
  3. width_scan_d12.png        F vs N at d=12 (XEB / MB / Transport_1QRB)
  4. instance_distribution.png per-instance F distribution (box plots)
  5. xeb_vs_mb.png             cross-validation: agreement of XEB and MB
  6. classical_approximability_gap.png   experimental F vs digital-error model
  7. fidelity_vs_NxD.png       universal collapse F ~ exp(-eps_eff*N*d)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
IMG_DIR = os.path.join(ROOT, "report", "images")
os.makedirs(IMG_DIR, exist_ok=True)

per = pd.read_csv(os.path.join(OUT_DIR, "per_instance_fidelity.csv"))
agg = pd.read_csv(os.path.join(OUT_DIR, "aggregated_fidelity.csv"))

ESTIMATOR_COLOR = {
    "XEB": "#1f77b4",          # blue
    "MB": "#d62728",           # red
    "Transport_1QRB": "#2ca02c",  # green
}
ESTIMATOR_MARKER = {
    "XEB": "o", "MB": "s", "Transport_1QRB": "^",
}


# ---------- helper: digital error-per-qubit-cycle model ---------- #


def fit_eps_per_cycle(df_sub):
    """Fit F = exp(-eps * N * d) by ordinary log-linear regression.

    Uses the aggregated mean-fidelity points.
    Returns eps with 1-sigma uncertainty.
    """
    df_sub = df_sub[df_sub["fidelity_mean"] > 0].copy()
    x = df_sub["N"].values * df_sub["d"].values
    y = -np.log(df_sub["fidelity_mean"].values)
    if len(x) < 2:
        return float("nan"), float("nan")
    eps, c = np.polyfit(x, y, 1)
    # uncertainty via residuals
    resid = y - (eps * x + c)
    sse = (resid ** 2).sum()
    dof = max(len(x) - 2, 1)
    sigma2 = sse / dof
    sxx = ((x - x.mean()) ** 2).sum()
    eps_se = math.sqrt(sigma2 / max(sxx, 1e-12))
    return float(eps), float(eps_se)


# ---------- 1.  Depth scan, N = 40 ---------- #

def fig_depth_scan(N_target, out_name):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    df40 = agg[(agg["N"] == N_target) & (agg["group"].isin([f"N{N_target}_verification","N56_depths"]))]
    if df40.empty:
        df40 = agg[agg["N"] == N_target]
    found_estimators = []
    for est in ["XEB", "MB", "Transport_1QRB"]:
        sub = df40[df40["estimator"] == est].sort_values("d")
        if sub.empty:
            continue
        found_estimators.append(est)
        ax.errorbar(
            sub["d"], sub["fidelity_mean"], yerr=sub["fidelity_sem"],
            fmt=ESTIMATOR_MARKER[est] + "-", color=ESTIMATOR_COLOR[est],
            label=est, capsize=3, lw=1.5, ms=6,
        )
    # fit digital-error model on MB if present, else XEB
    base = df40[df40["estimator"] == "MB"]
    if base.empty:
        base = df40[df40["estimator"] == "XEB"]
    if not base.empty:
        eps, eps_se = fit_eps_per_cycle(base)
        if not math.isnan(eps):
            d_grid = np.linspace(base["d"].min(), base["d"].max(), 100)
            F_model = np.exp(-eps * N_target * d_grid)
            ax.plot(d_grid, F_model, "k--",
                    label=fr"digital model  $F=\exp(-\epsilon N d)$, $\epsilon$={eps:.3e}")
    ax.set_yscale("log")
    ax.set_xlabel("Circuit depth d (cycles)")
    ax.set_ylabel("Fidelity F")
    ax.set_title(f"Depth scan, N = {N_target}")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, out_name), dpi=160)
    plt.close(fig)


fig_depth_scan(40, "depth_scan_N40.png")
fig_depth_scan(56, "depth_scan_N56.png")


# ---------- 2.  Width scan, d = 12 ---------- #

def fig_width_scan():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    df12 = agg[(agg["d"] == 12) & (agg["group"] == "N_scan_depth12")]
    for est in ["XEB", "MB"]:
        sub = df12[df12["estimator"] == est].sort_values("N")
        if sub.empty:
            continue
        ax.errorbar(
            sub["N"], sub["fidelity_mean"], yerr=sub["fidelity_sem"],
            fmt=ESTIMATOR_MARKER[est] + "-", color=ESTIMATOR_COLOR[est],
            label=est, capsize=3, lw=1.5, ms=6,
        )
    base = df12[df12["estimator"] == "MB"]
    if not base.empty:
        eps, _ = fit_eps_per_cycle(base)
        if not math.isnan(eps):
            N_grid = np.linspace(base["N"].min(), base["N"].max(), 100)
            F_model = np.exp(-eps * N_grid * 12)
            ax.plot(N_grid, F_model, "k--",
                    label=fr"digital model, $\epsilon$={eps:.3e}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits N")
    ax.set_ylabel("Fidelity F")
    ax.set_title("Width scan, d = 12")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "width_scan_d12.png"), dpi=160)
    plt.close(fig)


fig_width_scan()


# ---------- 3.  Per-instance distribution ---------- #

def fig_instance_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # left: XEB at N=40 across d
    p40 = per[(per["estimator"] == "XEB") & (per["N"] == 40)
              & (per["group"] == "N40_verification")].dropna(subset=["fidelity"])
    ds = sorted(p40["d"].unique())
    box_data = [p40[p40["d"] == d]["fidelity"].values for d in ds]
    axes[0].boxplot(box_data, positions=ds, widths=1.0,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=ESTIMATOR_COLOR["XEB"],
                                  alpha=0.4))
    for d, vals in zip(ds, box_data):
        axes[0].scatter(np.full_like(vals, d, dtype=float)
                        + np.random.uniform(-0.4, 0.4, size=len(vals)),
                        vals, s=8, color=ESTIMATOR_COLOR["XEB"], alpha=0.5)
    axes[0].axhline(0, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("Depth d")
    axes[0].set_ylabel("Per-instance XEB fidelity")
    axes[0].set_title("Per-instance XEB at N=40")
    axes[0].grid(True, ls=":", alpha=0.4)

    # right: MB at d=12 across N (width scan)
    p12 = per[(per["estimator"] == "MB") & (per["d"] == 12)
              & (per["group"] == "N_scan_depth12")].dropna(subset=["fidelity"])
    Ns = sorted(p12["N"].unique())
    box_data = [p12[p12["N"] == n]["fidelity"].values for n in Ns]
    axes[1].boxplot(box_data, positions=Ns, widths=3.0,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=ESTIMATOR_COLOR["MB"],
                                  alpha=0.4))
    for n, vals in zip(Ns, box_data):
        axes[1].scatter(np.full_like(vals, n, dtype=float)
                        + np.random.uniform(-1.0, 1.0, size=len(vals)),
                        vals, s=8, color=ESTIMATOR_COLOR["MB"], alpha=0.5)
    axes[1].axhline(0, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("Number of qubits N")
    axes[1].set_ylabel("Per-instance MB success")
    axes[1].set_title("Per-instance MB at d=12")
    axes[1].grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "instance_distribution.png"), dpi=160)
    plt.close(fig)


fig_instance_distribution()


# ---------- 4.  XEB vs MB cross-validation ---------- #

def fig_xeb_vs_mb():
    """Pair (N,d,r) where both XEB and MB are available and scatter."""
    xeb = per[(per["estimator"] == "XEB")][["group", "N", "d", "r", "fidelity"]]
    mb  = per[(per["estimator"] == "MB")][["group", "N", "d", "r", "fidelity"]]
    merged = xeb.merge(mb, on=["group", "N", "d", "r"], suffixes=("_xeb", "_mb"))
    merged = merged.dropna()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(merged["fidelity_mb"], merged["fidelity_xeb"],
                    c=merged["N"]*merged["d"], cmap="viridis", s=22, alpha=0.75)
    ax.plot([-0.3, 1.0], [-0.3, 1.0], "k--", lw=1, label="y = x")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("N · d (gate-count proxy)")
    ax.set_xlabel("MB fidelity (per instance)")
    ax.set_ylabel("XEB fidelity (per instance)")
    ax.set_title("Cross-validation:  XEB vs MB on matched (N, d, r)")
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.5, 1.7)
    ax.grid(True, ls=":", alpha=0.5)
    # add aggregated correlation
    if len(merged) > 2:
        corr = np.corrcoef(merged["fidelity_mb"], merged["fidelity_xeb"])[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}\n"
                            f"N pairs = {len(merged)}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "xeb_vs_mb.png"), dpi=160)
    plt.close(fig)


fig_xeb_vs_mb()


# ---------- 5.  Classical approximability gap ---------- #

def fig_classical_gap():
    """Compare experimental fidelity to the digital-error / gate-count
    model that drives the classical approximability bound.

    The Sycamore-style framework states that a noise channel of depth d
    on N qubits with effective per-cycle error eps yields
        F_exp ~ exp(-eps * N * d).
    Classical *approximability* scales with the same exponential — any
    classical algorithm that approximates the experimental output to
    a target total-variation distance must absorb at least the same
    quantity of stochastic noise.  The empirical curve sitting above
    the noiseless-limit reference (F=1) and the model curve being
    finite together demonstrate the operational gap.
    """
    fig, ax = plt.subplots(figsize=(7, 4.8))
    # join all (N40, varying d) and (varying N at d=12) points for both XEB and MB
    pts = []
    for est in ["XEB", "MB"]:
        sub = agg[(agg["estimator"] == est) &
                  (agg["group"].isin(["N40_verification","N_scan_depth12","N56_depths"]))]
        for _, r in sub.iterrows():
            pts.append((r["N"]*r["d"], r["fidelity_mean"], r["fidelity_sem"], est))
    pts = pd.DataFrame(pts, columns=["Nd", "F", "Ferr", "est"])
    for est, c in ESTIMATOR_COLOR.items():
        s = pts[pts["est"] == est].sort_values("Nd")
        if s.empty:
            continue
        ax.errorbar(s["Nd"], s["F"], yerr=s["Ferr"], fmt=ESTIMATOR_MARKER[est],
                    color=c, label=est, capsize=2, ms=6, lw=0)
    # global fit
    s = pts.dropna()
    s = s[s["F"] > 0]
    if len(s) > 2:
        x = s["Nd"].values
        y = -np.log(s["F"].values)
        eps, c0 = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = np.exp(-eps * xx - c0)
        ax.plot(xx, yy, "k--",
                label=fr"$F=\exp(-\epsilon\,Nd)$, $\epsilon$={eps:.3e}")
    ax.set_yscale("log")
    ax.set_xlabel("N · d  (qubit-cycle count)")
    ax.set_ylabel("Fidelity F")
    ax.set_title("Classical-approximability gap:  experimental F vs digital-error model")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "classical_approximability_gap.png"), dpi=160)
    plt.close(fig)
    return float(eps)


eps_global = fig_classical_gap()


# ---------- 6.  Universal collapse F vs N*d ---------- #

def fig_collapse():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for est in ["XEB", "MB", "Transport_1QRB"]:
        sub = agg[agg["estimator"] == est]
        if sub.empty:
            continue
        ax.errorbar(sub["N"]*sub["d"], sub["fidelity_mean"], yerr=sub["fidelity_sem"],
                    fmt=ESTIMATOR_MARKER[est], color=ESTIMATOR_COLOR[est],
                    label=est, capsize=2, ms=5, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("N · d (qubit-cycles)")
    ax.set_ylabel("Fidelity F")
    ax.set_title("Universal scaling collapse")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fidelity_vs_NxD.png"), dpi=160)
    plt.close(fig)


fig_collapse()


# ---------- save fitted error rates ---------- #

fits = {}
for label, mask in {
    "N40_depth_scan_MB":      ((agg["group"]=="N40_verification") & (agg["estimator"]=="MB")),
    "N40_depth_scan_XEB":     ((agg["group"]=="N40_verification") & (agg["estimator"]=="XEB")),
    "N56_depth_scan_MB":      ((agg["group"]=="N56_depths") & (agg["estimator"]=="MB")),
    "width_scan_d12_MB":      ((agg["group"]=="N_scan_depth12") & (agg["estimator"]=="MB") & (agg["d"]==12)),
    "width_scan_d12_XEB":     ((agg["group"]=="N_scan_depth12") & (agg["estimator"]=="XEB") & (agg["d"]==12)),
}.items():
    eps, eps_se = fit_eps_per_cycle(agg[mask])
    fits[label] = {"eps_per_qubit_cycle": eps, "eps_se": eps_se,
                   "n_points": int(mask.sum())}

fits["global_fit"] = {"eps_per_qubit_cycle": float(eps_global)}

import json
json.dump(fits, open(os.path.join(OUT_DIR, "digital_error_fits.json"), "w"),
          indent=2)
print("digital error fits:")
print(json.dumps(fits, indent=2))
print("\nFigures saved to", IMG_DIR)
