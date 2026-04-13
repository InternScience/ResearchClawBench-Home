#!/usr/bin/env python3
import ast
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Multi-component Icosahedral Reproduction Data.txt"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


def parse_dataset(path: Path) -> dict:
    data = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = eval(value.strip(), {"__builtins__": {}}, {})
    return data


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


def mismatch_ratio(inner_radius: float, outer_radius: float) -> float:
    return (outer_radius - inner_radius) / inner_radius


def category_target_map(dataset: dict) -> dict:
    result = {}
    for shell_i, shell_j, cat_i, cat_j, target in dataset["mismatch_params"]:
        result[(cat_i, cat_j)] = {
            "shell_pair": (shell_i, shell_j),
            "target_mismatch": target,
        }
    return result


def range_map(dataset: dict) -> dict:
    return {
        (cat_i, cat_j): (lower, upper)
        for cat_i, cat_j, lower, upper in dataset["optimal_mismatch_ranges"]
    }


def energy_map(dataset: dict) -> dict:
    return {(shell, category): energy for shell, category, energy in dataset["shell_energies"]}


def build_candidates(dataset: dict) -> list:
    radii = dict(dataset["atomic_radii"])
    target_map = category_target_map(dataset)
    target_ranges = range_map(dataset)
    energies = energy_map(dataset)
    candidates = []

    for cluster_name, core_element, shell_element, core_cat, shell_cat in dataset["multicomponent_clusters"]:
        inner_r = radii[core_element]
        outer_r = radii[shell_element]
        observed = mismatch_ratio(inner_r, outer_r)
        target = target_map.get((core_cat, shell_cat), {}).get("target_mismatch")
        target_low, target_high = target_ranges.get((core_cat, shell_cat), (None, None))
        core_energy = energies.get((1, core_cat), 0.0)
        shell_energy = energies.get((2, shell_cat), 0.0)
        mismatch_penalty = abs(observed - target) if target is not None else abs(observed)
        in_range = target_low is not None and target_low <= observed <= target_high
        energy_bonus = -(shell_energy + core_energy)
        range_penalty = 0.0 if in_range else min(abs(observed - target_low), abs(observed - target_high))
        score = mismatch_penalty + 2.5 * range_penalty - 0.03 * energy_bonus
        candidates.append(
            {
                "cluster": cluster_name,
                "core_element": core_element,
                "shell_element": shell_element,
                "core_category": core_cat,
                "shell_category": shell_cat,
                "observed_mismatch": observed,
                "target_mismatch": target,
                "target_low": target_low,
                "target_high": target_high,
                "core_energy": core_energy,
                "shell_energy": shell_energy,
                "stability_score": score,
                "within_target_range": in_range,
            }
        )

    category_options = [("MC", "MC"), ("MC", "Ch1"), ("MC", "Ch2"), ("MC", "BG")]
    shell_templates = [
        ("13@32", 1, 2),
        ("13@42", 1, 2),
        ("13@45", 1, 2),
        ("55@92", 2, 3),
        ("55@147", 2, 3),
        ("147@192", 3, 4),
    ]
    elements = list(radii.keys())

    for core_element in elements:
        for shell_element in elements:
            if core_element == shell_element:
                continue
            inner_r = radii[core_element]
            outer_r = radii[shell_element]
            observed = mismatch_ratio(inner_r, outer_r)
            for core_cat, shell_cat in category_options:
                target = target_map.get((core_cat, shell_cat), {}).get("target_mismatch")
                if target is None:
                    continue
                target_low, target_high = target_ranges.get((core_cat, shell_cat), (None, None))
                in_range = target_low is not None and target_low <= observed <= target_high
                mismatch_penalty = abs(observed - target)
                for label, core_shell_idx, outer_shell_idx in shell_templates:
                    core_energy = energies.get((core_shell_idx, core_cat), 0.0)
                    shell_energy = energies.get((outer_shell_idx, shell_cat), 0.0)
                    energy_bonus = -(shell_energy + core_energy)
                    range_penalty = 0.0 if in_range else min(abs(observed - target_low), abs(observed - target_high))
                    score = mismatch_penalty + 2.5 * range_penalty - 0.03 * energy_bonus
                    candidates.append(
                        {
                            "cluster": f"{core_element}{label.split('@')[0]}@{shell_element}{label.split('@')[1]}",
                            "core_element": core_element,
                            "shell_element": shell_element,
                            "core_category": core_cat,
                            "shell_category": shell_cat,
                            "observed_mismatch": observed,
                            "target_mismatch": target,
                            "target_low": target_low,
                            "target_high": target_high,
                            "core_energy": core_energy,
                            "shell_energy": shell_energy,
                            "stability_score": score,
                            "within_target_range": in_range,
                        }
                    )

    candidates.sort(
        key=lambda row: (
            row["stability_score"],
            abs(row["observed_mismatch"] - row["target_mismatch"]) if row["target_mismatch"] is not None else row["observed_mismatch"],
        )
    )
    deduped = []
    seen = set()
    for row in candidates:
        key = (row["cluster"], row["core_category"], row["shell_category"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def simulate_growth(dataset: dict, n_runs: int = 200) -> list:
    random.seed(dict(dataset["growth_parameters"])["random_seed"])
    weight_map = dict(dataset["path_probability_weights"])
    categories = ["MC", "BG", "Ch1", "Ch2"]
    target_map = category_target_map(dataset)
    mismatch_targets = {
        "MC": 0.04,
        "BG": target_map.get(("MC", "BG"), {}).get("target_mismatch", 0.09),
        "Ch1": target_map.get(("MC", "Ch1"), {}).get("target_mismatch", 0.14),
        "Ch2": target_map.get(("MC", "Ch2"), {}).get("target_mismatch", 0.21),
    }
    steps = list(range(0, 51, 10))
    trajectories = []

    for run_id in range(n_runs):
        mismatch = 0.0
        category = "MC"
        for step in steps:
            if step == 0:
                trajectories.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "category": category,
                        "mismatch": mismatch,
                    }
                )
                continue

            draw = random.random()
            if draw < weight_map["conservative_step"]:
                target_category = category if category != "MC" else "MC"
                mismatch += 0.35 * (mismatch_targets[target_category] - mismatch)
            elif draw < weight_map["conservative_step"] + weight_map["mismatch_driven_step"]:
                if mismatch < 0.08:
                    category = "Ch1"
                elif mismatch < 0.17:
                    category = "Ch2"
                mismatch += 0.7 * (mismatch_targets[category] - mismatch)
            else:
                category = random.choice(categories)
                mismatch = max(0.0, mismatch + random.uniform(-0.03, 0.03))

            trajectories.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "category": category,
                    "mismatch": mismatch,
                }
            )

    return trajectories


def aggregate_growth(trajectories: list) -> list:
    grouped = defaultdict(list)
    cat_counts = defaultdict(lambda: defaultdict(int))
    for row in trajectories:
        grouped[row["step"]].append(row["mismatch"])
        cat_counts[row["step"]][row["category"]] += 1

    summary = []
    for step in sorted(grouped):
        mismatches = grouped[step]
        dominant_category = max(cat_counts[step].items(), key=lambda x: x[1])[0]
        summary.append(
            {
                "step": step,
                "mean_mismatch": sum(mismatches) / len(mismatches),
                "std_mismatch": (
                    sum((x - (sum(mismatches) / len(mismatches))) ** 2 for x in mismatches) / len(mismatches)
                )
                ** 0.5,
                "dominant_category": dominant_category,
                "category_distribution": dict(cat_counts[step]),
            }
        )
    return summary


def experimental_validation(dataset: dict) -> list:
    rows = []
    for t_i, t_j, measured, theoretical in dataset["experimental_points"]:
        abs_err = abs(measured - theoretical)
        rel_err = abs_err / measured if measured else 0.0
        rows.append(
            {
                "t_i": t_i,
                "t_j": t_j,
                "measured": measured,
                "theoretical": theoretical,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            }
        )
    return rows


def summarize_sequences(dataset: dict) -> dict:
    return {
        "mackay_increments": [
            dataset["mackay_sequence"][i + 1] - dataset["mackay_sequence"][i]
            for i in range(len(dataset["mackay_sequence"]) - 1)
        ],
        "new_sequence_b5_increments": [
            dataset["new_sequence_b5"][i + 1] - dataset["new_sequence_b5"][i]
            for i in range(len(dataset["new_sequence_b5"]) - 1)
        ],
        "hexagonal_path_count": len(dataset["hexagonal_coords"]),
        "chiral_labels": dataset["chiral_labels"],
    }


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_size_mismatch(candidates: list) -> None:
    top = candidates[:12]
    labels = [row["cluster"] for row in top]
    scores = [row["stability_score"] for row in top]
    colors = ["#2ca02c" if row["within_target_range"] else "#d62728" for row in top]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top)), scores, color=colors)
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.ylabel("Composite stability score")
    plt.title("Top predicted multi-shell icosahedral candidates")
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "candidate_stability.png", dpi=200)
    plt.close()


def plot_validation(validation_rows: list) -> None:
    measured = [row["measured"] for row in validation_rows]
    theoretical = [row["theoretical"] for row in validation_rows]

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(measured, theoretical, s=70, color="#1f77b4")
    low = min(measured + theoretical) - 0.01
    high = max(measured + theoretical) + 0.01
    plt.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Measured mismatch")
    plt.ylabel("Theoretical mismatch")
    plt.title("Experimental validation of mismatch theory")
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "validation_scatter.png", dpi=200)
    plt.close()


def plot_growth(summary: list, dataset: dict) -> None:
    empirical = defaultdict(list)
    for step, category, mismatch in dataset["growth_results"]:
        empirical[step].append(mismatch)
    empirical_steps = sorted(empirical)
    empirical_means = [sum(empirical[s]) / len(empirical[s]) for s in empirical_steps]

    steps = [row["step"] for row in summary]
    means = [row["mean_mismatch"] for row in summary]
    stds = [row["std_mismatch"] for row in summary]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, means, marker="o", label="Local simulation mean", color="#d62728")
    plt.fill_between(steps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2, color="#d62728")
    plt.plot(empirical_steps, empirical_means, marker="s", label="Reproduction dataset mean", color="#1f77b4")
    plt.xlabel("Growth step")
    plt.ylabel("Average mismatch")
    plt.title("Growth trajectory mismatch evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "growth_dynamics.png", dpi=200)
    plt.close()


def plot_magic_numbers(dataset: dict) -> None:
    mackay = dataset["mackay_sequence"]
    new_b5 = dataset["new_sequence_b5"]
    x1 = list(range(1, len(mackay) + 1))
    x2 = list(range(1, len(new_b5) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(x1, mackay, marker="o", label="Mackay sequence", color="#1f77b4")
    plt.plot(x2, new_b5, marker="s", label="b=5 sequence", color="#ff7f0e")
    plt.xlabel("Shell index")
    plt.ylabel("Magic number")
    plt.title("Competing shell magic-number sequences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "magic_number_sequences.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    dataset = parse_dataset(DATA_PATH)

    candidates = build_candidates(dataset)
    validation_rows = experimental_validation(dataset)
    trajectories = simulate_growth(dataset)
    growth_summary = aggregate_growth(trajectories)
    sequence_summary = summarize_sequences(dataset)

    write_csv(
        OUTPUTS / "candidate_clusters.csv",
        candidates,
        list(candidates[0].keys()),
    )
    write_csv(
        OUTPUTS / "validation_metrics.csv",
        validation_rows,
        list(validation_rows[0].keys()),
    )
    write_csv(
        OUTPUTS / "growth_summary.csv",
        growth_summary,
        ["step", "mean_mismatch", "std_mismatch", "dominant_category", "category_distribution"],
    )

    (OUTPUTS / "sequence_summary.json").write_text(
        json.dumps(sequence_summary, indent=2),
        encoding="utf-8",
    )

    headline = {
        "best_candidates": candidates[:10],
        "validation_mae": sum(row["absolute_error"] for row in validation_rows) / len(validation_rows),
        "growth_terminal_mean_mismatch": growth_summary[-1]["mean_mismatch"],
        "growth_terminal_category": growth_summary[-1]["dominant_category"],
    }
    (OUTPUTS / "headline_results.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    plot_size_mismatch(candidates)
    plot_validation(validation_rows)
    plot_growth(growth_summary, dataset)
    plot_magic_numbers(dataset)


if __name__ == "__main__":
    main()
