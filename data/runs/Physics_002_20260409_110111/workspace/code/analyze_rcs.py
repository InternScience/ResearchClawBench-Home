#!/usr/bin/env python3
import ast
import csv
import glob
import json
import math
import os
import random
import re
from collections import defaultdict
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RESULTS = os.path.join(ROOT, "data", "results")
DATA_AMPLITUDES = os.path.join(ROOT, "data", "amplitudes")
OUT_DIR = os.path.join(ROOT, "outputs")
IMG_DIR = os.path.join(ROOT, "report", "images")

QUBIT_COUNTS_FOR_REFERENCE = [16, 24, 32, 40, 48, 56]
DEPTHS_FOR_REFERENCE = [8, 10, 12, 14, 16, 18, 20]
BOOTSTRAP_SAMPLES = 2000
RNG = random.Random(20260409)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


def canonicalize_bitstring(key):
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.startswith("("):
            seq = ast.literal_eval(stripped)
            return "".join(str(int(bit)) for bit in seq)
        if set(stripped).issubset({"0", "1"}):
            return stripped
    if isinstance(key, (list, tuple)):
        return "".join(str(int(bit)) for bit in key)
    raise ValueError(f"Unsupported bitstring key format: {key!r}")


def complex_or_real_to_probability(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = complex(text.replace("i", "j"))
            return abs(parsed) ** 2
        except ValueError:
            return float(text)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parsed = complex(value[0], value[1])
        return abs(parsed) ** 2
    raise ValueError(f"Unsupported amplitude/probability format: {value!r}")


def parse_instance_metadata(path):
    match = re.search(r"N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_XEB_", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse metadata from {path}")
    return {k: int(v) for k, v in match.groupdict().items()}


def load_counts(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    counts = {}
    for key, value in raw.items():
        counts[canonicalize_bitstring(key)] = int(value)
    return counts


def load_ideal_probabilities(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    probs = {}
    for key, value in raw.items():
        probs[canonicalize_bitstring(key)] = complex_or_real_to_probability(value)
    return probs


def bootstrap_mean(values, n_boot=BOOTSTRAP_SAMPLES):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return values[0], values[0], 0.0
    draws = []
    for _ in range(n_boot):
        sample = [values[RNG.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    low = draws[int(0.025 * n_boot)]
    high = draws[min(n_boot - 1, int(0.975 * n_boot))]
    avg = sum(draws) / n_boot
    variance = sum((x - avg) ** 2 for x in draws) / (n_boot - 1)
    return low, high, math.sqrt(variance)


def analyze_xeb_instances():
    pattern = os.path.join(DATA_AMPLITUDES, "N40_verification", "N40_d*_XEB", "*_amplitudes.json")
    amp_paths = sorted(glob.glob(pattern))
    instance_rows = []
    for amp_path in amp_paths:
        meta = parse_instance_metadata(amp_path)
        counts_path = os.path.join(
            DATA_RESULTS,
            "N40_verification",
            f"N{meta['N']}_d{meta['d']}_XEB",
            f"N{meta['N']}_d{meta['d']}_r{meta['r']}_XEB_counts.json",
        )
        if not os.path.exists(counts_path):
            continue
        counts = load_counts(counts_path)
        ideal_probs = load_ideal_probabilities(amp_path)
        matched = sorted(set(counts) & set(ideal_probs))
        total_shots = sum(counts.values())
        weighted_probs = []
        per_shot_terms = []
        missing_count = total_shots - sum(counts[k] for k in matched)
        for bitstring in matched:
            p = ideal_probs[bitstring]
            weighted_probs.append((counts[bitstring], p))
            per_shot_terms.extend([2 ** meta["N"] * p - 1.0] * counts[bitstring])
        if total_shots == 0 or not per_shot_terms:
            continue
        mean_ideal_prob = sum(count * p for count, p in weighted_probs) / total_shots
        linear_xeb = (2 ** meta["N"]) * mean_ideal_prob - 1.0
        ci_low, ci_high, boot_se = bootstrap_mean(per_shot_terms)
        instance_rows.append(
            {
                "N": meta["N"],
                "depth": meta["d"],
                "instance": meta["r"],
                "shots": total_shots,
                "matched_keys": len(matched),
                "missing_shots": missing_count,
                "mean_ideal_prob": mean_ideal_prob,
                "linear_xeb": linear_xeb,
                "xeb_ci_low": ci_low,
                "xeb_ci_high": ci_high,
                "xeb_bootstrap_se": boot_se,
                "subset_prob_sum": sum(ideal_probs.values()),
                "subset_prob_median": median(ideal_probs.values()),
                "subset_prob_max": max(ideal_probs.values()),
            }
        )
    return instance_rows


def aggregate_by_depth(instance_rows):
    grouped = defaultdict(list)
    for row in instance_rows:
        grouped[row["depth"]].append(row)
    summary_rows = []
    for depth in sorted(grouped):
        rows = grouped[depth]
        xebs = [row["linear_xeb"] for row in rows]
        summary_rows.append(
            {
                "N": 40,
                "depth": depth,
                "instances": len(rows),
                "mean_linear_xeb": mean(xebs),
                "median_linear_xeb": median(xebs),
                "std_linear_xeb": sample_std(xebs),
                "sem_linear_xeb": sample_std(xebs) / math.sqrt(len(xebs)),
                "mean_shots": mean([row["shots"] for row in rows]),
                "mean_matched_keys": mean([row["matched_keys"] for row in rows]),
                "mean_subset_prob_sum": mean([row["subset_prob_sum"] for row in rows]),
            }
        )
    return summary_rows


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def plot_depth_curve(summary_rows):
    depths = [row["depth"] for row in summary_rows]
    means = [row["mean_linear_xeb"] for row in summary_rows]
    sems = [row["sem_linear_xeb"] for row in summary_rows]
    plt.figure(figsize=(7.2, 4.6))
    plt.errorbar(depths, means, yerr=sems, marker="o", linewidth=1.8, capsize=4, color="#145A32")
    plt.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("Circuit depth d")
    plt.ylabel("Mean linear XEB fidelity estimate")
    plt.title("40-qubit arbitrary-geometry RCS: fidelity vs depth")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "xeb_vs_depth.png"), dpi=220)
    plt.close()


def plot_instance_distribution(instance_rows):
    grouped = defaultdict(list)
    for row in instance_rows:
        grouped[row["depth"]].append(row["linear_xeb"])
    depths = sorted(grouped)
    data = [grouped[depth] for depth in depths]
    plt.figure(figsize=(8.0, 4.8))
    bp = plt.boxplot(data, tick_labels=[str(depth) for depth in depths], patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#AED6F1", alpha=0.8)
    for med in bp["medians"]:
        med.set(color="#1B4F72", linewidth=1.8)
    plt.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("Circuit depth d")
    plt.ylabel("Per-instance linear XEB fidelity")
    plt.title("Distribution across 50 random instances per depth")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "xeb_instance_distribution.png"), dpi=220)
    plt.close()


def plot_local_gap(summary_rows):
    # Local benchmark substitute for the paper's hardness gap: ideal brute-force state size grows
    # exponentially with qubit count while the measured fidelity decays only gradually over depth.
    depths = [row["depth"] for row in summary_rows]
    mean_xeb = [max(row["mean_linear_xeb"], 1e-12) for row in summary_rows]
    classical_log10 = [40 * math.log10(2.0)] * len(depths)
    quantum_signal = [math.log10(x) for x in mean_xeb]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(depths, classical_log10, marker="s", color="#7D3C98", label=r"$\log_{10}(2^{40})$ state-space scale")
    plt.plot(depths, quantum_signal, marker="o", color="#B03A2E", label=r"$\log_{10}(\mathrm{mean\ XEB})$")
    plt.xlabel("Circuit depth d")
    plt.ylabel("Log10 scale")
    plt.title("Local comparison: state-space scale vs measured fidelity signal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "local_gap_proxy.png"), dpi=220)
    plt.close()


def build_data_overview(instance_rows, summary_rows):
    return {
        "analysis_scope": {
            "qubits_analyzed_with_ideal_subset": 40,
            "depths_analyzed": [row["depth"] for row in summary_rows],
            "instances_total": len(instance_rows),
            "instances_per_depth": {str(row["depth"]): row["instances"] for row in summary_rows},
        },
        "local_dataset_reference": {
            "results_qubit_counts_found": QUBIT_COUNTS_FOR_REFERENCE,
            "depths_with_xeb_amplitudes_for_n40": DEPTHS_FOR_REFERENCE,
        },
    }


def main():
    ensure_dirs()
    instance_rows = analyze_xeb_instances()
    summary_rows = aggregate_by_depth(instance_rows)
    instance_csv = os.path.join(OUT_DIR, "xeb_instance_estimates.csv")
    summary_csv = os.path.join(OUT_DIR, "xeb_depth_summary.csv")
    overview_json = os.path.join(OUT_DIR, "analysis_overview.json")
    write_csv(instance_csv, instance_rows, list(instance_rows[0].keys()))
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()))
    write_json(overview_json, build_data_overview(instance_rows, summary_rows))
    plot_depth_curve(summary_rows)
    plot_instance_distribution(instance_rows)
    plot_local_gap(summary_rows)
    print(f"Wrote {instance_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {overview_json}")
    print(f"Wrote figures to {IMG_DIR}")


if __name__ == "__main__":
    main()
