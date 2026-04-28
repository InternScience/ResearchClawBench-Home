"""Generate all figures for the report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


# ---------- Figure 1: data overview ----------
def fig_data_overview():
    npz = np.load(OUT / "parsed_data.npz")
    Z = npz["pp_Z"]; X = npz["pp_X"]; y = npz["pp_y"]; edges = npz["pp_edges"]
    a = npz["sg_a"]; b = npz["sg_b"]
    cfg = json.loads((OUT / "data_summary.json").read_text())["autonomous_optimization"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    # 1a — atomic-number histogram & graph topology
    ax = axes[0, 0]
    ax.hist(Z.flatten(), bins=range(0, 12), color="C0", edgecolor="k")
    ax.set_title(f"Atomic numbers (uniform Z=5; B-like)\n{Z.shape[0]} graphs × {Z.shape[1]} atoms")
    ax.set_xlabel("Z"); ax.set_ylabel("count")

    ax = axes[0, 1]
    # draw the small graph topology (5 atoms)
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts = np.column_stack([np.cos(ang), np.sin(ang)])
    for (i, j) in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], "k-", alpha=0.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=300, c="C1", zorder=3, edgecolor="k")
    for k, p in enumerate(pts):
        ax.text(p[0], p[1], str(k), ha="center", va="center", fontsize=10, weight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Shared crystal-graph topology\n5 atoms · {edges.shape[0]} edges")

    ax = axes[0, 2]
    ax.hist(y, bins=20, color="C2", edgecolor="k")
    ax.set_title(f"Property targets y\nμ={y.mean():.3f}, σ={y.std():.3f}")
    ax.set_xlabel("y (a.u.)"); ax.set_ylabel("count")

    # 2a — lattice scatter
    ax = axes[1, 0]
    ax.scatter(a, b, s=18, alpha=0.7, c="C3")
    ax.set_xlabel("lattice parameter a"); ax.set_ylabel("lattice parameter b")
    ax.set_title(f"Lattice (a, b) samples (n={len(a)})")

    ax = axes[1, 1]
    ax.hist(a, bins=20, alpha=0.6, label="a", color="C3")
    ax.hist(b, bins=20, alpha=0.6, label="b", color="C4")
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.set_title("Marginal lattice distributions")
    ax.legend()

    # BO search space
    ax = axes[1, 2]
    T_range = cfg["T_range"]; t_range = cfg["t_range"]
    rect = plt.Rectangle((T_range[0], t_range[0]),
                         T_range[1]-T_range[0], t_range[1]-t_range[0],
                         fill=False, edgecolor="k", lw=1.5)
    ax.add_patch(rect)
    ax.scatter([cfg["T_target"]], [cfg["t_target"]], marker="*", s=300,
               c="gold", edgecolor="k", zorder=5, label="ground-truth optimum")
    ax.set_xlim(T_range[0]-20, T_range[1]+20); ax.set_ylim(t_range[0]-2, t_range[1]+2)
    ax.set_xlabel("Temperature T (°C)"); ax.set_ylabel("time t (min)")
    ax.set_title(f"Synthesis search space\nnoise σ={cfg['noise']}, threshold={cfg['threshold']}")
    ax.legend(loc="lower right")

    fig.suptitle("M-AI-Synth dataset — overview of three workflows", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(IMG / "01_data_overview.png")
    plt.close(fig)


# ---------- Figure 2: CGCNN training curves ----------
def fig_cgcnn_training():
    pd = np.load(OUT / "property_prediction_preds.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(pd["hist_cg_train"], label="CGCNN train", color="C0")
    ax.plot(pd["hist_cg_val"], label="CGCNN val", color="C0", linestyle="--")
    ax.plot(pd["hist_mlp_train"], label="MLP train", color="C1")
    ax.plot(pd["hist_mlp_val"], label="MLP val", color="C1", linestyle="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardized y)")
    ax.set_title("Training curves (early-stopped)")
    ax.legend()

    ax = axes[1]
    metrics = json.loads((OUT / "property_prediction_metrics.json").read_text())
    labels = ["RMSE", "MAE"]
    cg_tr = [metrics["CGCNN_lite"]["train"]["rmse"], metrics["CGCNN_lite"]["train"]["mae"]]
    cg_va = [metrics["CGCNN_lite"]["val"]["rmse"], metrics["CGCNN_lite"]["val"]["mae"]]
    ml_tr = [metrics["MLP_baseline"]["train"]["rmse"], metrics["MLP_baseline"]["train"]["mae"]]
    ml_va = [metrics["MLP_baseline"]["val"]["rmse"], metrics["MLP_baseline"]["val"]["mae"]]
    x = np.arange(len(labels)); w = 0.2
    ax.bar(x - 1.5*w, cg_tr, w, label="CGCNN train", color="C0", alpha=0.6)
    ax.bar(x - 0.5*w, cg_va, w, label="CGCNN val",   color="C0")
    ax.bar(x + 0.5*w, ml_tr, w, label="MLP train",   color="C1", alpha=0.6)
    ax.bar(x + 1.5*w, ml_va, w, label="MLP val",     color="C1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("error"); ax.set_title("Property-prediction error: CGCNN-lite vs MLP")
    ax.legend()

    fig.tight_layout()
    fig.savefig(IMG / "02_cgcnn_training.png")
    plt.close(fig)


# ---------- Figure 3: parity plot ----------
def fig_parity():
    pd = np.load(OUT / "property_prediction_preds.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model, color in zip(axes,
                                ["CGCNN-lite", "MLP baseline"],
                                ["C0", "C1"]):
        if model == "CGCNN-lite":
            tr_p, va_p = pd["cg_tr"], pd["cg_va"]
        else:
            tr_p, va_p = pd["mlp_tr"], pd["mlp_va"]
        ax.scatter(pd["y_tr"], tr_p, c=color, alpha=0.5, s=22, label="train")
        ax.scatter(pd["y_va"], va_p, c=color, edgecolor="k", s=40, label="val")
        lo = min(pd["y_tr"].min(), pd["y_va"].min(), tr_p.min(), va_p.min())
        hi = max(pd["y_tr"].max(), pd["y_va"].max(), tr_p.max(), va_p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true y"); ax.set_ylabel("pred y")
        ax.set_title(model); ax.legend()
    fig.suptitle("Parity plots — property prediction")
    fig.tight_layout()
    fig.savefig(IMG / "03_cgcnn_parity.png")
    plt.close(fig)


# ---------- Figure 4: VAE training ----------
def fig_vae_training():
    sg = np.load(OUT / "structure_generation_samples.npz")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(sg["history_loss"], label="total ELBO loss", color="k")
    ax.plot(sg["history_recon"], label="recon (MSE)", color="C0")
    ax.plot(sg["history_kl"], label="KL divergence", color="C2")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("VAE training (lattice generation)")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMG / "04_vae_training.png"); plt.close(fig)


# ---------- Figure 5: VAE generated vs real ----------
def fig_vae_samples():
    sg = np.load(OUT / "structure_generation_samples.npz")
    real = sg["real"]; vae = sg["vae"]; gauss = sg["gauss"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax = axes[0]
    ax.scatter(vae[:, 0], vae[:, 1], s=10, alpha=0.35, c="C0", label="VAE generated")
    ax.scatter(real[:, 0], real[:, 1], s=22, c="r", edgecolor="k", label="real")
    ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title("VAE samples vs real")
    ax.legend()

    ax = axes[1]
    ax.scatter(gauss[:, 0], gauss[:, 1], s=10, alpha=0.35, c="C2", label="Gaussian samples")
    ax.scatter(real[:, 0], real[:, 1], s=22, c="r", edgecolor="k", label="real")
    ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_title("Gaussian baseline samples")
    ax.legend()

    # marginals
    ax = axes[2]
    bins = 25
    ax.hist(real[:, 0], bins=bins, density=True, alpha=0.5, color="r",   label="real a")
    ax.hist(vae[:, 0],  bins=bins, density=True, alpha=0.5, color="C0",  label="VAE a")
    ax.hist(gauss[:, 0],bins=bins, density=True, alpha=0.4, color="C2",  label="Gauss a")
    ax.set_title("Marginal of a (real vs VAE vs Gauss)")
    ax.legend(); ax.set_xlabel("a"); ax.set_ylabel("density")
    fig.tight_layout(); fig.savefig(IMG / "05_vae_generated.png"); plt.close(fig)


# ---------- Figure 6: BO regret curves ----------
def fig_bo_regret():
    m = json.loads((OUT / "autonomous_optimization_metrics.json").read_text())
    bo_mean = np.array(m["BO"]["mean_curve"])
    bo_std  = np.array(m["BO"]["std_curve"])
    rs_mean = np.array(m["Random"]["mean_curve"])
    rs_std  = np.array(m["Random"]["std_curve"])
    peak    = m["noiseless_peak"]
    thr     = 0.95 * m["config"]["threshold"]
    it = np.arange(1, len(bo_mean) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(it, bo_mean, color="C0", label="Bayesian Optimization")
    ax.fill_between(it, bo_mean-bo_std, bo_mean+bo_std, color="C0", alpha=0.25)
    ax.plot(it, rs_mean, color="C1", label="Random search")
    ax.fill_between(it, rs_mean-rs_std, rs_mean+rs_std, color="C1", alpha=0.25)
    ax.axhline(peak, color="k", linestyle=":", label=f"noiseless peak ({peak:.1f})")
    ax.axhline(thr, color="grey", linestyle="--", label=f"success thr ({thr:.2f})")
    ax.set_xlabel("evaluations"); ax.set_ylabel("best yield observed")
    ax.set_title(f"Optimization progress (mean ± 1σ over {m['n_seeds']} seeds)")
    ax.legend(loc="lower right")

    ax = axes[1]
    regret_bo = peak - bo_mean
    regret_rs = peak - rs_mean
    ax.plot(it, regret_bo, color="C0", label="BO")
    ax.plot(it, regret_rs, color="C1", label="Random")
    ax.set_yscale("log")
    ax.set_xlabel("evaluations"); ax.set_ylabel("simple regret (peak − best)")
    ax.set_title("Simple regret (lower is better)")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMG / "06_bo_regret.png"); plt.close(fig)


# ---------- Figure 7: BO acquisition progress ----------
def fig_bo_progress():
    runs = np.load(OUT / "autonomous_optimization_runs.npz")
    cfg = json.loads((OUT / "data_summary.json").read_text())["autonomous_optimization"]
    Xb = runs["bo_X_seed0"]; yb = runs["bo_y_seed0"]
    Xr = runs["rs_X_seed0"]; yr = runs["rs_y_seed0"]
    T_target = cfg["T_target"]; t_target = cfg["t_target"]

    # noiseless oracle for the contour
    span_T = cfg["T_range"][1] - cfg["T_range"][0]
    span_t = cfg["t_range"][1] - cfg["t_range"][0]
    T = np.linspace(cfg["T_range"][0], cfg["T_range"][1], 80)
    t = np.linspace(cfg["t_range"][0], cfg["t_range"][1], 80)
    GT, Gt = np.meshgrid(T, t)
    Z = 10.0 * np.exp(-((GT - T_target) / (0.4 * span_T))**2
                      -((Gt - t_target) / (0.4 * span_t))**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X, y, name in [(axes[0], Xb, yb, "Bayesian Optimization"),
                           (axes[1], Xr, yr, "Random search")]:
        cs = ax.contourf(GT, Gt, Z, levels=15, cmap="viridis", alpha=0.85)
        sc = ax.scatter(X[:, 0], X[:, 1],
                        c=np.arange(len(X)), cmap="autumn",
                        s=42, edgecolor="k", zorder=4)
        ax.scatter([T_target], [t_target], marker="*", s=300,
                   c="white", edgecolor="k", zorder=5)
        ax.set_xlabel("T (°C)"); ax.set_ylabel("t (min)")
        ax.set_title(f"{name} — seed 0 trajectory")
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("evaluation index")
    fig.suptitle("Synthesis search space and sampled points (yield contour overlay)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(IMG / "07_bo_progress.png"); plt.close(fig)


if __name__ == "__main__":
    fig_data_overview()
    fig_cgcnn_training()
    fig_parity()
    fig_vae_training()
    fig_vae_samples()
    fig_bo_regret()
    fig_bo_progress()
    print("All figures written to", IMG)
