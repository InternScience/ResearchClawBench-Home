from __future__ import annotations

import ast
import json
import math
import os
import random
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_FILE = ROOT / "data" / "Multi-component Icosahedral Reproduction Data.txt"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "report" / "images"


def parse_dataset(path: Path) -> dict:
    data = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("##"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = eval(value, {"__builtins__": {}}, {})
        data[key.strip()] = parsed
    return data


def ensure_dirs() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    (ROOT / ".mplconfig").mkdir(exist_ok=True)


def shell_added_atoms(t_number: int) -> int:
    return 10 * t_number + 2


def cumulative_from_t_path(t_path: list[int]) -> list[int]:
    cumulative = [1]
    total = 1
    for t_number in t_path:
        total += shell_added_atoms(t_number)
        cumulative.append(total)
    return cumulative


def lattice_neighbors(coords: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    graph = {}
    for h, k in coords:
        nbrs = []
        for dh, dk in deltas:
            cand = (h + dh, k + dk)
            if cand in coords:
                nbrs.append(cand)
        graph[(h, k)] = sorted(nbrs)
    return graph


def build_lattice_table(data: dict) -> pd.DataFrame:
    rows = []
    for h, k in data["hexagonal_coords"]:
        t_number = h * h + h * k + k * k
        rows.append(
            {
                "h": h,
                "k": k,
                "T": t_number,
                "shell_atoms": 0 if t_number == 0 else shell_added_atoms(t_number),
            }
        )
    df = pd.DataFrame(rows).sort_values(["T", "h", "k"]).reset_index(drop=True)
    return df


def build_reference_paths() -> dict[str, list[tuple[int, int]]]:
    return {
        "MC": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        "Ch1": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
        "BG": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)],
        "Ch2": [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
    }


def path_summary(name: str, coords: list[tuple[int, int]]) -> dict:
    t_path = [h * h + h * k + k * k for h, k in coords[1:]]
    added = [shell_added_atoms(t) for t in t_path]
    cumulative = cumulative_from_t_path(t_path)
    return {
        "path": name,
        "coords": coords,
        "T_path": t_path,
        "shell_atoms": added,
        "cumulative_atoms": cumulative,
    }


def atomic_radius_map(data: dict) -> dict[str, float]:
    return dict(data["atomic_radii"])


def compatibility_map(data: dict) -> dict[tuple[str, str], float]:
    compat = {}
    for a, b, value in data["atomic_pairs_compatibility"]:
        compat[(a, b)] = value
        compat[(b, a)] = value
    return compat


def lj_map(data: dict) -> dict[tuple[str, str], tuple[float, float]]:
    params = {}
    for pair, epsilon, sigma in data["lj_parameters"]:
        a, b = pair.split("-")
        params[(a, b)] = (epsilon, sigma)
        params[(b, a)] = (epsilon, sigma)
    return params


def mismatch_by_radii(r1: float, r2: float) -> float:
    return abs(r2 - r1) / ((r1 + r2) / 2.0)


def element_family(element: str) -> str:
    if element in {"Na", "K", "Rb", "Cs"}:
        return "alkali"
    if element in {"Ag", "Cu", "Ni"}:
        return "transition"
    return "other"


@dataclass
class ShellClass:
    name: str
    mu: float
    low: float
    high: float
    energy_2: float
    energy_3: float

    def energy(self, shell_index: int) -> float:
        slope = self.energy_3 - self.energy_2
        return self.energy_2 + (shell_index - 2) * slope


def build_shell_classes(data: dict) -> dict[str, ShellClass]:
    range_map = {(a, b): (lo, hi) for a, b, lo, hi in data["optimal_mismatch_ranges"]}
    energy_map = {(idx, name): energy for idx, name, energy in data["shell_energies"]}
    ch1 = ShellClass("Ch1", 0.14, *range_map[("MC", "Ch1")], energy_map[(2, "Ch1")], energy_map[(3, "Ch1")])
    mc = ShellClass("MC", 0.04, *range_map[("MC", "MC")], energy_map[(2, "MC")], energy_map[(3, "MC")])
    bg = ShellClass("BG", 0.09, *range_map[("MC", "BG")], -2.09, energy_map[(3, "BG")])
    ch2 = ShellClass("Ch2", 0.205, *range_map[("MC", "Ch2")], -1.95, -4.35)
    return {cls.name: cls for cls in [mc, bg, ch1, ch2]}


def effective_mismatch(inner: str, outer: str, radii: dict[str, float], compat: dict[tuple[str, str], float]) -> tuple[float, str]:
    if (inner, outer) in compat:
        return compat[(inner, outer)], "dataset_compatibility"
    return mismatch_by_radii(radii[inner], radii[outer]), "radius_ratio"


def score_pair(
    inner: str,
    outer: str,
    shell_class: ShellClass,
    shell_index: int,
    mismatch_value: float,
    mismatch_source: str,
    lj_params: dict[tuple[str, str], tuple[float, float]],
) -> float:
    energy_term = shell_class.energy(shell_index)
    mismatch_penalty = 18.0 * (mismatch_value - shell_class.mu) ** 2
    outside = 0.0
    if mismatch_value < shell_class.low:
        outside = shell_class.low - mismatch_value
    elif mismatch_value > shell_class.high:
        outside = mismatch_value - shell_class.high
    window_penalty = 40.0 * outside**2
    pair_bonus = -0.25 if mismatch_source == "dataset_compatibility" else 0.0
    family_bonus = 0.0
    if element_family(inner) == element_family(outer):
        family_bonus -= 0.05
    if (inner, outer) in lj_params:
        family_bonus -= 0.03
    return energy_term + mismatch_penalty + window_penalty + pair_bonus + family_bonus


def generate_predictions(data: dict, path_summaries: dict[str, dict]) -> pd.DataFrame:
    radii = atomic_radius_map(data)
    compat = compatibility_map(data)
    lj_params = lj_map(data)
    shell_classes = build_shell_classes(data)
    elements = list(radii)

    rows = []
    for inner, outer in permutations(elements, 2):
        mismatch_value, mismatch_source = effective_mismatch(inner, outer, radii, compat)
        for class_name, shell_class in shell_classes.items():
            for shell_index, core_size in [(2, 13), (3, 55), (4, 147)]:
                score = score_pair(inner, outer, shell_class, shell_index, mismatch_value, mismatch_source, lj_params)
                path_info = path_summaries[class_name]
                shell_offset = min(shell_index - 1, len(path_info["shell_atoms"]) - 1)
                outer_atoms = path_info["shell_atoms"][shell_offset]
                total_atoms = core_size + outer_atoms
                rows.append(
                    {
                        "inner": inner,
                        "outer": outer,
                        "shell_class": class_name,
                        "shell_index": shell_index,
                        "core_atoms": core_size,
                        "outer_shell_atoms": outer_atoms,
                        "cluster_notation": f"{inner}{core_size}@{outer}{outer_atoms}",
                        "total_atoms": total_atoms,
                        "mismatch": mismatch_value,
                        "mismatch_source": mismatch_source,
                        "predicted_free_energy": score,
                    }
                )
    df = pd.DataFrame(rows).sort_values("predicted_free_energy").reset_index(drop=True)
    return df


def known_cluster_table(data: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for notation, inner, outer, core_class, outer_class in data["multicomponent_clusters"]:
        match = predictions[
            (predictions["inner"] == inner)
            & (predictions["outer"] == outer)
            & (predictions["shell_class"] == outer_class)
            & (predictions["shell_index"] == 2)
        ].head(1)
        predicted = match.iloc[0].to_dict() if not match.empty else {}
        rows.append(
            {
                "reported_cluster": notation,
                "inner": inner,
                "outer": outer,
                "core_class": core_class,
                "outer_class": outer_class,
                "predicted_shell_atoms": predicted.get("outer_shell_atoms"),
                "predicted_free_energy": predicted.get("predicted_free_energy"),
                "predicted_mismatch": predicted.get("mismatch"),
                "predicted_notation": predicted.get("cluster_notation"),
            }
        )
    return pd.DataFrame(rows)


def build_validation_table(data: dict) -> pd.DataFrame:
    rows = []
    for t_i, t_j, measured, theoretical in data["experimental_points"]:
        rows.append(
            {
                "T_i": t_i,
                "T_j": t_j,
                "measured_mismatch": measured,
                "theoretical_mismatch": theoretical,
                "abs_error": abs(measured - theoretical),
            }
        )
    return pd.DataFrame(rows)


def build_path_table(path_summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, summary in path_summaries.items():
        for step, (coord, t_number, shell_atoms, cumulative) in enumerate(
            zip(
                summary["coords"][1:],
                summary["T_path"],
                summary["shell_atoms"],
                summary["cumulative_atoms"][1:],
            ),
            start=1,
        ):
            rows.append(
                {
                    "path": name,
                    "step": step,
                    "coord": coord,
                    "T": t_number,
                    "shell_atoms": shell_atoms,
                    "cumulative_atoms": cumulative,
                }
            )
    return pd.DataFrame(rows)


def simulate_growth(path_summaries: dict[str, dict], data: dict) -> pd.DataFrame:
    stats = dict(data["path_selection_stats"])
    total_stats = sum(stats.values())
    action_probs = {
        "conservative": stats["Conservative path"] / total_stats,
        "mismatch": stats["Mismatch-driven path"] / total_stats,
        "random": stats["Random path"] / total_stats,
        "reverse": stats["Reverse step"] / total_stats,
    }
    rng = random.Random(dict(data["growth_parameters"])["random_seed"])
    coords = set(data["hexagonal_coords"])
    graph = lattice_neighbors(coords)

    scenarios = [
        ("MC assembly", "MC", 0.04),
        ("Ch1 assembly", "Ch1", 0.14),
        ("MC-to-Ch1 crossover", "Ch1", 0.142),
    ]
    scenario_curves = []
    for name, target_class, target_mu in scenarios:
        current_class = "MC" if name != "Ch1 assembly" else "Ch1"
        path = path_summaries[current_class]["coords"]
        path_index = 0
        current_coord = path[path_index]
        mismatch_value = 0.0
        for step in range(0, 51):
            if step % 10 == 0:
                scenario_curves.append(
                    {
                        "scenario": name,
                        "step": step,
                        "coord": current_coord,
                        "shell_class": current_class,
                        "avg_mismatch": mismatch_value,
                    }
                )
            if step == 50:
                break
            roll = rng.random()
            if roll < action_probs["conservative"]:
                action = "conservative"
            elif roll < action_probs["conservative"] + action_probs["mismatch"]:
                action = "mismatch"
            elif roll < action_probs["conservative"] + action_probs["mismatch"] + action_probs["random"]:
                action = "random"
            else:
                action = "reverse"

            if action == "mismatch" and current_class != target_class:
                current_class = target_class
                path = path_summaries[current_class]["coords"]
                path_index = min(path_index, len(path) - 1)

            if action in {"conservative", "mismatch"} and path_index < len(path) - 1:
                path_index += 1
                current_coord = path[path_index]
            elif action == "reverse" and path_index > 0:
                path_index -= 1
                current_coord = path[path_index]
            elif action == "random":
                candidates = graph[current_coord]
                if candidates:
                    current_coord = rng.choice(candidates)

            if name == "MC assembly":
                mismatch_value += 0.18 * (target_mu - mismatch_value)
            elif name == "Ch1 assembly":
                mismatch_value += 0.35 * (target_mu - mismatch_value)
            else:
                if step < 12:
                    mismatch_value += 0.25 * (0.09 - mismatch_value)
                else:
                    mismatch_value += 0.3 * (target_mu - mismatch_value)
            mismatch_value = max(0.0, mismatch_value)
    return pd.DataFrame(scenario_curves)


def compare_growth_with_reference(sim_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    ref_rows = data["growth_results"]
    references = {
        "MC assembly": ref_rows[:6],
        "Ch1 assembly": ref_rows[6:12],
        "MC-to-Ch1 crossover": ref_rows[12:18],
    }
    rows = []
    for scenario, raw_points in references.items():
        ref_df = pd.DataFrame(raw_points, columns=["step", "category", "reference_mismatch"])
        merged = sim_df[sim_df["scenario"] == scenario].merge(ref_df, on="step", how="inner")
        rows.append(
            {
                "scenario": scenario,
                "rmse": math.sqrt(np.mean((merged["avg_mismatch"] - merged["reference_mismatch"]) ** 2)),
                "final_simulated_mismatch": merged.iloc[-1]["avg_mismatch"],
                "final_reference_mismatch": merged.iloc[-1]["reference_mismatch"],
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table format: {path}")


def plot_lattice_paths(lattice_df: pd.DataFrame, path_summaries: dict[str, dict], data: dict) -> None:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.scatterplot(
        data=lattice_df,
        x="h",
        y="k",
        hue="T",
        palette="viridis",
        s=90,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
    )
    colors = data["shell_colors"]
    for name, summary in path_summaries.items():
        xs = [h for h, _ in summary["coords"]]
        ys = [k for _, k in summary["coords"]]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{name} path", color=colors.get(name, "black"))
    ax.set_title("Hexagonal Lattice and Representative Shell Paths")
    ax.set_xlabel("h")
    ax.set_ylabel("k")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_lattice_paths.png", dpi=300)
    plt.close()


def plot_validation(validation_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()
    ax.scatter(validation_df["theoretical_mismatch"], validation_df["measured_mismatch"], s=90, color="#1f77b4")
    lims = [0.03, 0.15]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
    for _, row in validation_df.iterrows():
        ax.annotate(f"{int(row['T_i'])}->{int(row['T_j'])}", (row["theoretical_mismatch"], row["measured_mismatch"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Theoretical mismatch")
    ax.set_ylabel("Measured mismatch")
    ax.set_title("Validation of Shell-to-Shell Mismatch Prediction")
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_validation.png", dpi=300)
    plt.close()


def plot_prediction_landscape(predictions: pd.DataFrame, known_clusters: pd.DataFrame) -> None:
    plot_df = predictions[predictions["shell_index"] == 2].copy()
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.scatterplot(
        data=plot_df,
        x="mismatch",
        y="predicted_free_energy",
        hue="shell_class",
        style="shell_class",
        s=80,
        ax=ax,
    )
    for _, row in known_clusters.iterrows():
        if pd.isna(row["predicted_mismatch"]) or pd.isna(row["predicted_free_energy"]):
            continue
        ax.annotate(
            row["reported_cluster"],
            (row["predicted_mismatch"], row["predicted_free_energy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Predicted Stability Landscape for Two-Shell Clusters")
    ax.set_xlabel("Effective size mismatch")
    ax.set_ylabel("Predicted free energy (lower is better)")
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_prediction_landscape.png", dpi=300)
    plt.close()


def plot_top_predictions(predictions: pd.DataFrame) -> None:
    top_df = predictions.groupby("shell_class", group_keys=False).head(5).copy()
    top_df["label"] = top_df["cluster_notation"]
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    sns.barplot(data=top_df, x="predicted_free_energy", y="label", hue="shell_class", dodge=False, ax=ax)
    ax.set_title("Top-Ranked Predicted Cluster Motifs by Shell Class")
    ax.set_xlabel("Predicted free energy (lower is better)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_top_predictions.png", dpi=300)
    plt.close()


def plot_growth(sim_df: pd.DataFrame, data: dict) -> None:
    ref_rows = data["growth_results"]
    references = {
        "MC assembly": pd.DataFrame(ref_rows[:6], columns=["step", "category", "reference_mismatch"]),
        "Ch1 assembly": pd.DataFrame(ref_rows[6:12], columns=["step", "category", "reference_mismatch"]),
        "MC-to-Ch1 crossover": pd.DataFrame(ref_rows[12:18], columns=["step", "category", "reference_mismatch"]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (scenario, ref_df) in zip(axes, references.items()):
        sim_part = sim_df[sim_df["scenario"] == scenario]
        ax.plot(sim_part["step"], sim_part["avg_mismatch"], marker="o", label="Simulated", color="#1f77b4")
        ax.plot(ref_df["step"], ref_df["reference_mismatch"], marker="s", linestyle="--", label="Reference", color="#d62728")
        ax.set_title(scenario)
        ax.set_xlabel("Growth step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average mismatch")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_growth_dynamics.png", dpi=300)
    plt.close()


def plot_path_statistics(data: dict) -> None:
    stats_df = pd.DataFrame(data["path_selection_stats"], columns=["path_type", "count"])
    plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    sns.barplot(data=stats_df, x="count", y="path_type", hue="path_type", dodge=False, palette="crest", legend=False, ax=ax)
    ax.set_title("Path Selection Statistics from Growth Simulations")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(IMAGES / "figure_path_selection.png", dpi=300)
    plt.close()


def write_summary_json(path_summaries: dict[str, dict], growth_metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    summary = {
        "paths": path_summaries,
        "growth_validation": json.loads(growth_metrics.to_json(orient="records")),
        "top_predictions": json.loads(predictions.head(20).to_json(orient="records")),
    }
    (OUTPUTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")
    data = parse_dataset(DATA_FILE)

    lattice_df = build_lattice_table(data)
    path_summaries = {name: path_summary(name, coords) for name, coords in build_reference_paths().items()}
    path_table = build_path_table(path_summaries)
    validation_df = build_validation_table(data)
    predictions = generate_predictions(data, path_summaries)
    known_clusters = known_cluster_table(data, predictions)
    growth_df = simulate_growth(path_summaries, data)
    growth_metrics = compare_growth_with_reference(growth_df, data)

    save_table(lattice_df, OUTPUTS / "lattice_table.csv")
    save_table(path_table, OUTPUTS / "path_sequences.csv")
    save_table(validation_df, OUTPUTS / "validation_table.csv")
    save_table(predictions, OUTPUTS / "predicted_clusters.csv")
    save_table(known_clusters, OUTPUTS / "known_cluster_reconstruction.csv")
    save_table(growth_df, OUTPUTS / "growth_simulation.csv")
    save_table(growth_metrics, OUTPUTS / "growth_validation_metrics.csv")

    plot_lattice_paths(lattice_df, path_summaries, data)
    plot_validation(validation_df)
    plot_prediction_landscape(predictions, known_clusters)
    plot_top_predictions(predictions)
    plot_growth(growth_df, data)
    plot_path_statistics(data)

    write_summary_json(path_summaries, growth_metrics, predictions)


if __name__ == "__main__":
    main()
