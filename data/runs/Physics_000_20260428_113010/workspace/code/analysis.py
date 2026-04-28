"""
Main analysis: reproduce theoretical predictions, compute mismatches for the
data-supplied multi-component clusters, validate against the experimental
points, run a simple kinetic-Monte-Carlo growth simulation, and save figures.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_data
import icosahedral_theory as it

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "savefig.bbox": "tight"})


def main():
    data = load_data()
    radii = dict(data["atomic_radii"])  # {'Na':1.86, ...}
    chiral_labels = data["chiral_labels"]
    shell_colors = data["shell_colors"]

    # ----------------------------------------------------------------- 1
    # Magic-number families: Mackay (h,0), Bergman (h,h), and chiral (h,k)
    families = {
        "MC (h,0)":  [(h, 0) for h in range(1, 6)],
        "BG (h,h)":  [(h, h) for h in range(1, 6)],
        "Ch1 (h,h+1)": [(h, h + 1) for h in range(0, 5)],
        "Ch2 (h,h+2)": [(h, h + 2) for h in range(0, 5)],
        "Ch3 (h,h+3)": [(h, h + 3) for h in range(0, 5)],
    }
    family_table = {}
    for name, path in families.items():
        # cumulative count starting from a single central atom
        atoms = [1]
        for (h, k) in path:
            atoms.append(atoms[-1] + it.shell_count(h, k))
        family_table[name] = atoms
    (OUT / "magic_numbers.json").write_text(json.dumps(family_table, indent=2))
    print("Mackay seq reproduced:", family_table["MC (h,0)"])
    print("Dataset Mackay seq   :", data["mackay_sequence"])

    # ---- Figure 1: data overview & magic numbers ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    elements = [e for e, _ in data["atomic_radii"]]
    rs = [r for _, r in data["atomic_radii"]]
    bars = ax.bar(elements, rs, color="#4c78a8", edgecolor="k")
    for b, v in zip(bars, rs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Atomic radius [Å]")
    ax.set_title("(a) Atomic radii from dataset")
    ax.set_ylim(0, max(rs) * 1.2)

    ax = axes[1]
    for name, atoms in family_table.items():
        ax.plot(range(len(atoms)), atoms, "o-", label=name)
    ax.plot(range(len(data["mackay_sequence"])), data["mackay_sequence"], "k*",
            ms=14, mec="k", mfc="yellow", label="Mackay (data)", zorder=5)
    ax.set_xlabel("Number of stacked shells")
    ax.set_ylabel("Cumulative atoms")
    ax.set_title("(b) Magic-number sequences for icosahedral shell families")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(IMG / "fig1_data_overview.png")
    plt.close(fig)

    # ----------------------------------------------------------------- 2
    # Optimal size ratio between adjacent shells (T_i, T_{i+1})
    Ts = [(1, 0), (2, 0), (3, 0), (4, 0)]   # achiral
    rho_table = []
    for (h1, k1), (h2, k2) in zip(Ts[:-1], Ts[1:]):
        T1, T2 = it.triangulation(h1, k1), it.triangulation(h2, k2)
        rho = it.optimal_size_ratio(T1, T2)
        rho_table.append({"inner": (h1, k1), "outer": (h2, k2),
                          "T_inner": T1, "T_outer": T2, "rho_opt": rho})
    (OUT / "rho_opt_table.json").write_text(json.dumps(rho_table, indent=2))

    # ----------------------------------------------------------------- 3
    # Predicted size-mismatch heatmap across all atom pairs.
    elems = list(radii.keys())
    M = np.zeros((len(elems), len(elems)))
    # We use shell stacking (1,0) -> (1,1) (MC -> BG, T=1 -> T=3) as canonical
    # for the heatmap, since the dataset highlights MC<->Ch1, MC<->BG pairings.
    T_in, T_out = 1, 3
    rho_opt = it.optimal_size_ratio(T_in, T_out)
    for i, ei in enumerate(elems):
        for j, ej in enumerate(elems):
            sm = it.size_mismatch(radii[ei], radii[ej], T_in, T_out)
            M[i, j] = sm
    np.savetxt(OUT / "mismatch_matrix.csv", M, delimiter=",",
               header=",".join(elems), comments="")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(M, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(elems))); ax.set_xticklabels(elems)
    ax.set_yticks(range(len(elems))); ax.set_yticklabels(elems)
    for i in range(len(elems)):
        for j in range(len(elems)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "k", fontsize=8)
    ax.set_xlabel("Outer-shell element")
    ax.set_ylabel("Inner-shell (core) element")
    ax.set_title(f"Size mismatch sm for MC→BG stacking\n(T_in={T_in}, T_out={T_out}, ρ_opt={rho_opt:.3f})")
    fig.colorbar(im, ax=ax, label="sm")
    fig.tight_layout()
    fig.savefig(IMG / "fig2_mismatch_heatmap.png")
    plt.close(fig)

    # ----------------------------------------------------------------- 4
    # Validate against experimental_points and dataset multicomponent_clusters.
    exp = data["experimental_points"]            # (T_i, T_{i+1}, sm_meas, sm_theo)
    Ts_meas = np.array([p[0] for p in exp])
    Ts_next = np.array([p[1] for p in exp])
    sm_meas = np.array([p[2] for p in exp])
    sm_theo = np.array([p[3] for p in exp])

    # Reproduce theoretical sm via rho_opt = sqrt(T2/T1)
    sm_recompute = []
    # The dataset gives sm directly -- we sanity-check by computing
    # |sqrt(T2/T1)-1| (the achiral-pair mismatch when both species identical)
    for ti, tn in zip(Ts_meas, Ts_next):
        sm_recompute.append(abs(math.sqrt(tn / ti) - 1.0))
    sm_recompute = np.array(sm_recompute)

    fig, ax = plt.subplots(figsize=(6, 5))
    lim = max(sm_meas.max(), sm_theo.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.scatter(sm_theo, sm_meas, s=80, c="#e15759", edgecolor="k",
               label="Dataset (measured vs theoretical)")
    ax.scatter(sm_recompute, sm_meas, s=80, c="#59a14f", marker="^",
               edgecolor="k", label="Re-derived |√(T₂/T₁)−1|")
    for ti, tn, m, t in zip(Ts_meas, Ts_next, sm_meas, sm_theo):
        ax.annotate(f"({ti}→{tn})", (t, m), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Theoretical size mismatch sm")
    ax.set_ylabel("Measured size mismatch sm")
    ax.set_title("Validation: theory vs measurement on shell pairs")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "fig3_validation.png")
    plt.close(fig)
    rmse_dataset = float(np.sqrt(np.mean((sm_meas - sm_theo) ** 2)))
    rmse_redo = float(np.sqrt(np.mean((sm_meas - sm_recompute) ** 2)))
    (OUT / "validation_rmse.json").write_text(json.dumps(
        {"rmse_dataset_theory_vs_measured": rmse_dataset,
         "rmse_recomputed_vs_measured": rmse_redo,
         "n_points": len(exp)}, indent=2))

    # ----------------------------------------------------------------- 5
    # Dataset definition of size mismatch (so experimental_points match):
    #   sm = |r_outer/r_inner - rho_opt|, rho_opt = 1 + Δ_geom
    # Δ_geom is the midpoint of the dataset's optimal_mismatch_ranges.
    geom_correction = {
        ("MC", "MC"):  0.04,
        ("MC", "BG"):  0.09,
        ("MC", "Ch1"): 0.14,
        ("MC", "Ch2"): 0.205,
        ("MC", "Ch3"): 0.20,
    }
    cluster_predictions = []
    base_cases = [
        ("Na13@Rb32",   "Na", "Rb", "MC", "BG"),
        ("K13@Cs42",    "K",  "Cs", "MC", "MC"),
        ("Ag13@Cu45",   "Ag", "Cu", "MC", "Ch1"),
        ("Ni147@Ag192", "Ni", "Ag", "MC", "Ch3"),
        ("Na13@K42",    "Na", "K",  "MC", "MC"),     # extra prediction
        ("Cu13@Ag42",   "Cu", "Ag", "MC", "MC"),     # extra prediction
    ]
    for lbl, core, shell, lc, ls in base_cases:
        delta = geom_correction.get((lc, ls),
                                    geom_correction.get((ls, lc), 0.0))
        rho_opt = 1.0 + delta
        rho = radii[shell] / radii[core]
        sm = abs(rho - rho_opt)
        cluster_predictions.append({
            "cluster": lbl, "core_atom": core, "shell_atom": shell,
            "core_label": lc, "shell_label": ls,
            "rho_opt": rho_opt, "rho_actual": rho, "size_mismatch": sm,
        })
    (OUT / "cluster_predictions.json").write_text(
        json.dumps(cluster_predictions, indent=2))

    # Bar chart of mismatch for predicted clusters versus optimal range.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [c["cluster"] for c in cluster_predictions]
    sms = [c["size_mismatch"] for c in cluster_predictions]
    bars = ax.bar(names, sms, color=["#4c78a8" if s < 0.10 else "#e15759" for s in sms],
                  edgecolor="k")
    for b, v in zip(bars, sms):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    # Reference optimal mismatch ranges from dataset
    for (a, b, lo, hi) in data["optimal_mismatch_ranges"]:
        ax.axhspan(lo, hi, alpha=0.15, color="green",
                   label=f"opt {a}↔{b}: [{lo},{hi}]")
    ax.set_ylabel("Predicted size mismatch sm")
    ax.set_title("Predicted multi-component icosahedral clusters")
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "fig4_cluster_predictions.png")
    plt.close(fig)

    # ----------------------------------------------------------------- 6
    # Growth simulation (kinetic Monte Carlo on shell-stacking choices).
    gp = dict(data["growth_parameters"])
    pw = dict(data["path_probability_weights"])
    random.seed(int(gp["random_seed"]))
    np.random.seed(int(gp["random_seed"]))

    candidates = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (2, 3),
                  (3, 0), (3, 3), (3, 4), (4, 0)]
    state = [(1, 0)]
    chosen_paths = []
    avg_mismatch_traj = []
    path_stats = {"Conservative path": 0, "Mismatch-driven path": 0,
                  "Random path": 0, "Reverse step": 0}
    delta_opt = float(gp["delta_opt"])
    rng = np.random.default_rng(42)
    n_steps = int(gp["simulation_steps"])

    for step in range(n_steps):
        last = state[-1]
        T_last = it.triangulation(*last)
        scores = []
        for cand in candidates:
            T_c = it.triangulation(*cand)
            if T_c <= T_last:
                continue
            score = -abs(math.sqrt(T_c / T_last) - 1.0 - delta_opt)
            scores.append((cand, score, T_c))
        # If we have run out of allowed outer shells, reset to a fresh seed
        # rather than counting that as a reverse step.
        if not scores:
            state = [(1, 0)]
            chosen_paths.append(("restart", last))
            avg_mismatch_traj.append(0.0)
            continue
        # occasional reverse step (matches dataset path-selection weights)
        if len(state) > 1 and rng.random() < 0.10 and                 (path_stats["Conservative path"] + path_stats["Mismatch-driven path"]
                 + path_stats["Random path"]) > 0:
            state.pop()
            path_stats["Reverse step"] += 1
            chosen_paths.append(("reverse", last))
        else:
            u = rng.random()
            if u < pw["conservative_step"]:
                cand = sorted(scores, key=lambda x: x[2])[0][0]
                path_stats["Conservative path"] += 1
            elif u < pw["conservative_step"] + pw["mismatch_driven_step"]:
                cand = max(scores, key=lambda x: x[1])[0]
                path_stats["Mismatch-driven path"] += 1
            else:
                cand = scores[rng.integers(0, len(scores))][0]
                path_stats["Random path"] += 1
            state.append(cand)
            chosen_paths.append(("forward", cand))
        if not state:
            state = [(1, 0)]
        sms = []
        for a, b in zip(state[:-1], state[1:]):
            Ta, Tb = it.triangulation(*a), it.triangulation(*b)
            sms.append(abs(math.sqrt(Tb / Ta) - 1.0))
        avg_mismatch_traj.append(float(np.mean(sms)) if sms else 0.0)

    (OUT / "growth_simulation.json").write_text(json.dumps({
        "final_state": state,
        "n_steps": len(chosen_paths),
        "path_stats_simulated": path_stats,
        "path_stats_dataset": dict(data["path_selection_stats"]),
    }, indent=2))

    # ---- Figure 5: growth dynamics + path statistics
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(avg_mismatch_traj, color="#4c78a8")
    ax.set_xlabel("MC step")
    ax.set_ylabel("Average shell-pair size mismatch")
    ax.set_title("(a) Growth simulation: mean mismatch trajectory")
    ax.axhline(delta_opt, color="r", ls="--", label=f"δ_opt={delta_opt}")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ds = dict(data["path_selection_stats"])
    keys = list(ds.keys())
    sim = [path_stats.get(k, 0) for k in keys]
    ref = [ds[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, ref, w, label="Dataset", color="#59a14f", edgecolor="k")
    ax.bar(x + w / 2, sim, w, label="Simulation", color="#4c78a8", edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Counts")
    ax.set_title("(b) Path-selection statistics")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "fig5_growth.png")
    plt.close(fig)

    # ----------------------------------------------------------------- 7
    # Lennard-Jones stability comparison for the predicted clusters
    lj = {n: (e, s) for n, e, s in data["lj_parameters"]}
    rows = []
    for c in cluster_predictions:
        core, shell = c["core_atom"], c["shell_atom"]
        # equilibrium L-J distance (2^(1/6) σ)
        for pair in (f"{core}-{core}", f"{shell}-{shell}", f"{core}-{shell}",
                     f"{shell}-{core}"):
            if pair in lj:
                eps, sig = lj[pair]
                r = 2 ** (1 / 6) * sig
                rows.append({
                    "cluster": c["cluster"], "pair": pair,
                    "epsilon": eps, "sigma": sig,
                    "r_eq": r, "U_min": it.lj_energy(r, eps, sig)
                })
    (OUT / "lj_table.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    rs = np.linspace(2.0, 7.0, 400)
    plotted = set()
    for n, e, s in data["lj_parameters"]:
        if n in plotted: continue
        plotted.add(n)
        U = [it.lj_energy(r, e, s) for r in rs]
        ax.plot(rs, U, label=n)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-1.5, 1.5); ax.set_xlim(rs[0], rs[-1])
    ax.set_xlabel("r [Å]"); ax.set_ylabel("U_LJ(r) [ε]")
    ax.set_title("Lennard-Jones potentials used in growth simulation")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "fig6_lj_potentials.png")
    plt.close(fig)

    print("All figures written to", IMG)
    print("All outputs written to", OUT)


if __name__ == "__main__":
    main()
