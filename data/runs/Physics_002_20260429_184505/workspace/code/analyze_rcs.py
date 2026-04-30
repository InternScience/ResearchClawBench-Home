#!/usr/bin/env python3
"""Reproducible RCS verification analysis for the workspace data.

Computes subset cross-entropy benchmarking (XEB) estimates from measured counts
and ideal amplitudes/probabilities, plus mirror-benchmark/1Q transport success
rates when ideal bitstrings are provided.
"""
import ast
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

META_RE = re.compile(r"N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_(?P<kind>[A-Za-z0-9]+)")


def parse_meta(path: Path):
    m = META_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse metadata from {path}")
    return {"N": int(m.group("N")), "d": int(m.group("d")), "r": int(m.group("r")), "kind": m.group("kind")}


def bit_key(obj):
    """Normalize a bitstring key/list to a tuple-string used by the JSON files."""
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("(") or s.startswith("["):
            try:
                vals = ast.literal_eval(s)
                return str(tuple(int(x) for x in vals))
            except Exception:
                pass
        if set(s) <= {"0", "1"}:
            return str(tuple(int(ch) for ch in s))
        return s
    if isinstance(obj, (list, tuple)):
        return str(tuple(int(x) for x in obj))
    raise TypeError(type(obj))


def load_counts(path: Path):
    raw = json.loads(path.read_text())
    return {bit_key(k): int(v) for k, v in raw.items()}


def value_to_prob(v):
    """Convert stored ideal value to probability.

    Values in these files are complex amplitudes encoded as strings. If a real
    nonnegative number is ever stored, treat values <=1 as probabilities.
    """
    if isinstance(v, str):
        z = complex(v.replace("i", "j"))
        return float((z.real * z.real) + (z.imag * z.imag))
    if isinstance(v, (int, float)):
        x = float(v)
        if x < 0:
            return x * x
        return x if x <= 1.0 else x * x
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]) ** 2 + float(v[1]) ** 2
    raise TypeError(f"Unsupported amplitude/probability value: {v!r}")


def load_probs(path: Path):
    raw = json.loads(path.read_text())
    return {bit_key(k): value_to_prob(v) for k, v in raw.items()}


def compute_xeb_row(count_path: Path):
    meta = parse_meta(count_path)
    amp_path = Path(str(count_path).replace("/data/results/", "/data/amplitudes/").replace("_counts.json", "_amplitudes.json"))
    if not amp_path.exists():
        return None
    counts = load_counts(count_path)
    probs = load_probs(amp_path)
    matched = sorted(set(counts) & set(probs))
    total_shots = sum(counts.values())
    matched_shots = sum(counts[k] for k in matched)
    if matched_shots == 0:
        raise ValueError(f"No matched shots for {count_path}")
    dim = 2 ** meta["N"]
    y = []
    weights = []
    for k in matched:
        weights.append(counts[k])
        y.append(dim * probs[k] - 1.0)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    expanded_mean = float(np.average(y, weights=weights))
    # Multinomial plug-in standard error for the sample mean over measured shots.
    # Since counts are aggregate categories, the weighted sample variance is
    # computed over the matched shot distribution with finite-sample correction.
    if matched_shots > 1:
        expanded_var = float(np.average((y - expanded_mean) ** 2, weights=weights) * matched_shots / (matched_shots - 1))
        se = math.sqrt(max(expanded_var, 0.0) / matched_shots)
    else:
        expanded_var = float("nan")
        se = float("nan")
    ci95 = 1.96 * se if math.isfinite(se) else float("nan")
    pvals = np.asarray([probs[k] for k in matched], dtype=float)
    row = {
        **meta,
        "count_file": str(count_path.relative_to(ROOT)),
        "amplitude_file": str(amp_path.relative_to(ROOT)),
        "n_count_keys": len(counts),
        "n_prob_keys": len(probs),
        "n_matched_keys": len(matched),
        "total_shots": total_shots,
        "matched_shots": matched_shots,
        "match_fraction_shots": matched_shots / total_shots if total_shots else np.nan,
        "xeb_fidelity": expanded_mean,
        "xeb_se": se,
        "xeb_ci95": ci95,
        "mean_dim_p_matched_unweighted": float(np.mean(dim * pvals)),
        "std_dim_p_matched_unweighted": float(np.std(dim * pvals, ddof=1)) if len(pvals) > 1 else 0.0,
        "subset_prob_mass": float(np.sum(pvals)),
        "min_dim_p": float(np.min(dim * pvals)),
        "max_dim_p": float(np.max(dim * pvals)),
    }
    return row


def compute_success_rows(pattern: str, label: str):
    rows = []
    for count_path in sorted((DATA / "results").glob(pattern)):
        meta = parse_meta(count_path)
        ideal_path = Path(str(count_path).replace("_counts.json", "_ideal_bitstring.json"))
        if not ideal_path.exists():
            continue
        counts = load_counts(count_path)
        ideal = bit_key(json.loads(ideal_path.read_text()))
        shots = sum(counts.values())
        success = counts.get(ideal, 0)
        p = success / shots if shots else np.nan
        se = math.sqrt(p * (1 - p) / shots) if shots else np.nan
        rows.append({**meta, "benchmark": label, "total_shots": shots, "n_count_keys": len(counts), "success_count": success, "success_rate": p, "success_se": se, "ideal_file": str(ideal_path.relative_to(ROOT)), "count_file": str(count_path.relative_to(ROOT))})
    return rows


def update_inventory():
    inv = {
        "primary_tables": [
            {"artifact": "outputs/fidelity_estimates.csv", "status": "satisfied", "description": "Per (N,d,r) XEB fidelity estimate, uncertainty, shot/match diagnostics"},
            {"artifact": "outputs/depth_summary.csv", "status": "satisfied", "description": "Depth-level aggregate XEB summary"},
            {"artifact": "outputs/n_scan_summary.csv", "status": "satisfied", "description": "N-scan aggregate XEB summary at d=12"},
            {"artifact": "outputs/benchmark_success_estimates.csv", "status": "satisfied", "description": "MB/Transport success-rate estimates from ideal bitstrings"},
            {"artifact": "outputs/data_overview.json", "status": "satisfied", "description": "Dataset counts, depth/r coverage, shot and key diagnostics"}
        ],
        "figures": [
            {"artifact": "report/images/data_overview.png", "status": "satisfied", "description": "Data coverage and measurement volume overview"},
            {"artifact": "report/images/xeb_by_depth.png", "status": "satisfied", "description": "Main XEB fidelity curve vs depth with uncertainty"},
            {"artifact": "report/images/n_scan_xeb.png", "status": "satisfied", "description": "XEB fidelity vs qubit count at d=12"},
            {"artifact": "report/images/instance_fidelity_heatmap.png", "status": "satisfied", "description": "Per-instance fidelity structure across depth and instance index"},
            {"artifact": "report/images/validation_diagnostics.png", "status": "satisfied", "description": "Match/uncertainty diagnostics validating subset-XEB workflow"},
            {"artifact": "report/images/benchmark_success.png", "status": "satisfied", "description": "MB/Transport ideal-bitstring success rates"}
        ],
        "report": {"artifact": "report/report.md", "status": "pending"}
    }
    (OUT / "target_artifact_inventory.json").write_text(json.dumps(inv, indent=2))


def main():
    sns.set_theme(style="whitegrid", context="paper")
    xeb_rows = []
    for count_path in sorted((DATA / "results").glob("**/*_XEB_counts.json")):
        row = compute_xeb_row(count_path)
        if row is not None:
            xeb_rows.append(row)
    xeb = pd.DataFrame(xeb_rows).sort_values(["N", "d", "r", "count_file"])
    xeb.to_csv(OUT / "fidelity_estimates.csv", index=False)

    # Aggregates by N,d. Distinguish instance-to-instance standard deviation from mean uncertainty.
    grp = xeb.groupby(["N", "d"], as_index=False).agg(
        n_instances=("xeb_fidelity", "size"),
        mean_xeb=("xeb_fidelity", "mean"),
        std_xeb=("xeb_fidelity", "std"),
        median_xeb=("xeb_fidelity", "median"),
        mean_xeb_se=("xeb_se", "mean"),
        mean_total_shots=("total_shots", "mean"),
        mean_matched_keys=("n_matched_keys", "mean"),
        min_match_fraction=("match_fraction_shots", "min"),
        max_match_fraction=("match_fraction_shots", "max"),
    )
    grp["sem_across_instances"] = grp["std_xeb"] / np.sqrt(grp["n_instances"])
    grp["ci95_across_instances"] = 1.96 * grp["sem_across_instances"]
    grp.to_csv(OUT / "xeb_summary_by_N_d.csv", index=False)
    grp[grp.N.isin([40, 56])].to_csv(OUT / "depth_summary.csv", index=False)
    grp[grp.d.eq(12)].to_csv(OUT / "n_scan_summary.csv", index=False)

    bench = pd.DataFrame(compute_success_rows("**/*_MB_counts.json", "MB") + compute_success_rows("**/*_Transport_1QRB_counts.json", "Transport_1QRB"))
    if not bench.empty:
        bench = bench.sort_values(["benchmark", "N", "d", "r", "count_file"])
        bench.to_csv(OUT / "benchmark_success_estimates.csv", index=False)
        bsum = bench.groupby(["benchmark", "N", "d"], as_index=False).agg(
            n_instances=("success_rate", "size"), mean_success=("success_rate", "mean"), std_success=("success_rate", "std"), mean_shots=("total_shots", "mean")
        )
        bsum["sem_success"] = bsum["std_success"] / np.sqrt(bsum["n_instances"])
        bsum.to_csv(OUT / "benchmark_success_summary.csv", index=False)
    else:
        bsum = pd.DataFrame()

    overview = {
        "xeb_paired_instances": int(len(xeb)),
        "xeb_coverage": {f"N{int(N)}_d{int(d)}": int(n) for (N, d), n in xeb.groupby(["N", "d"]).size().items()},
        "all_xeb_pairs_have_20_count_keys": bool((xeb.n_count_keys == 20).all()),
        "all_xeb_pairs_have_20_prob_keys": bool((xeb.n_prob_keys == 20).all()),
        "all_xeb_pairs_have_20_matched_keys": bool((xeb.n_matched_keys == 20).all()),
        "xeb_total_shots_unique": sorted(int(x) for x in xeb.total_shots.unique()),
        "amplitude_limited_XEB": "XEB fidelities computed only for instances with matching amplitude files; N56 XEB counts lack amplitudes in this workspace.",
        "benchmark_instances": int(len(bench)) if not bench.empty else 0,
    }
    (OUT / "data_overview.json").write_text(json.dumps(overview, indent=2))

    related = {
        "extracted_from_related_work": "ReadPDF failed and no pdftotext/PyPDF2/pdfminer was available initially; extraction relies on task instructions plus PDF metadata/strings fallback.",
        "task_relevant_method_facts": [
            "Cross-entropy benchmarking estimates circuit fidelity using F_XEB = 2^N <P_ideal(x)> - 1 over experimentally sampled bitstrings.",
            "For finite samples, uncertainty can be estimated from the variance of per-shot linear-XEB values.",
            "Mirror/transport benchmark files in the workspace encode ideal bitstrings and support a direct success-rate validation, but not a full gate-count error model without circuit/gate metadata."
        ],
        "limitations": [
            "No circuit topology/gate count metadata was found in the data tree, so gate-count/error propagation is discussed qualitatively rather than fit as a physical error model.",
            "Amplitude files are present for N=16/24/32/40 at d=12 and N=40 at d=8..20; N=48/N=56 XEB counts are not converted to XEB fidelities because corresponding amplitudes are absent."
        ]
    }
    (OUT / "related_work_contract.json").write_text(json.dumps(related, indent=2))

    fidelity_check = {
        "method": "linear XEB",
        "definition": "F_XEB = 2^N * mean_sample[p_ideal(x)] - 1",
        "implemented_steps": [
            "Parse N,d,r from filenames.",
            "Pair each XEB counts file to its ideal amplitude file.",
            "Convert complex amplitudes to probabilities via |a|^2.",
            "Match measured bitstrings to ideal probabilities exactly after tuple-key normalization.",
            "Compute counts-weighted mean of 2^N p_ideal(x)-1.",
            "Compute finite-shot standard error from weighted sample variance over matched shots."
        ],
        "invariants_verified": {
            "all_paired_instances_have_20_matched_keys": bool((xeb.n_matched_keys == 20).all()),
            "all_paired_instances_have_full_shot_match": bool((xeb.match_fraction_shots == 1.0).all()),
            "probabilities_nonnegative": bool((xeb.min_dim_p >= 0).all())
        },
        "deviations": [
            "The workflow is subset-XEB because only 20 measured bitstrings/probabilities per instance are provided.",
            "MB regression probability and gate-count error propagation are not fit exactly due to missing circuit-level metadata."
        ]
    }
    (OUT / "method_fidelity_checklist.json").write_text(json.dumps(fidelity_check, indent=2))

    # Figure 1 data overview: coverage and shots.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    cov = grp.pivot(index="N", columns="d", values="n_instances").fillna(0)
    sns.heatmap(cov, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "paired XEB instances"}, ax=axes[0])
    axes[0].set_title("XEB amplitude/count coverage")
    axes[0].set_xlabel("depth d")
    axes[0].set_ylabel("qubits N")
    sns.barplot(data=xeb, x="N", y="total_shots", hue="d", estimator=np.mean, errorbar=None, ax=axes[1])
    axes[1].set_title("Shots per paired XEB instance")
    axes[1].set_ylabel("shots")
    axes[1].legend(title="d", fontsize=7, ncol=2)
    fig.savefig(IMG / "data_overview.png", dpi=200)
    plt.close(fig)

    # Figure 2 depth scan for N=40 (and N=56 note unavailable).
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for N, sub in grp[grp.N.isin([40])].groupby("N"):
        sub = sub.sort_values("d")
        ax.errorbar(sub.d, sub.mean_xeb, yerr=sub.ci95_across_instances, marker="o", capsize=3, label=f"N={N}, mean ±95% CI across r")
    raw40 = xeb[xeb.N.eq(40)]
    ax.scatter(raw40.d + np.random.default_rng(2).normal(0, 0.045, len(raw40)), raw40.xeb_fidelity, s=12, alpha=0.25, color="tab:blue", label="instances")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("depth d")
    ax.set_ylabel("XEB fidelity")
    ax.set_title("Fixed-N depth scan from subset XEB")
    ax.legend(fontsize=8)
    fig.savefig(IMG / "xeb_by_depth.png", dpi=200)
    plt.close(fig)

    # Figure 3 N scan at d=12.
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    sub = grp[grp.d.eq(12)].sort_values("N")
    ax.errorbar(sub.N, sub.mean_xeb, yerr=sub.ci95_across_instances, marker="o", capsize=3, color="tab:green")
    raw = xeb[xeb.d.eq(12)]
    ax.scatter(raw.N + np.random.default_rng(3).normal(0, 0.08, len(raw)), raw.xeb_fidelity, s=10, alpha=0.25, color="tab:green")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("qubits N")
    ax.set_ylabel("XEB fidelity")
    ax.set_title("Fixed-depth N scan (d=12)")
    fig.savefig(IMG / "n_scan_xeb.png", dpi=200)
    plt.close(fig)

    # Figure 4 heatmap per instance for N=40 verification depths.
    heat = xeb[xeb.N.eq(40)].pivot_table(index="r", columns="d", values="xeb_fidelity", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sns.heatmap(heat, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "XEB fidelity"})
    ax.set_title("N=40 per-instance XEB fidelity")
    ax.set_xlabel("depth d")
    ax.set_ylabel("instance r")
    fig.savefig(IMG / "instance_fidelity_heatmap.png", dpi=200)
    plt.close(fig)

    # Figure 5 validation diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    sns.histplot(xeb["n_matched_keys"], bins=np.arange(19.5, 21.6, 0.5), ax=axes[0, 0])
    axes[0, 0].set_title("Matched ideal/measured keys")
    sns.histplot(xeb["match_fraction_shots"], bins=10, ax=axes[0, 1])
    axes[0, 1].set_title("Matched shot fraction")
    sns.scatterplot(data=xeb, x="xeb_fidelity", y="xeb_se", hue="N", palette="viridis", s=18, ax=axes[1, 0])
    axes[1, 0].set_title("Finite-shot uncertainty vs estimate")
    sns.scatterplot(data=xeb, x="mean_dim_p_matched_unweighted", y="xeb_fidelity", hue="N", palette="viridis", s=18, ax=axes[1, 1], legend=False)
    axes[1, 1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1, 1].set_title("Unweighted subset probability vs weighted XEB")
    fig.savefig(IMG / "validation_diagnostics.png", dpi=200)
    plt.close(fig)

    # Figure 6 benchmark success rates for ideal-bitstring data.
    if not bsum.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        mb = bsum[bsum.benchmark.eq("MB")]
        sns.lineplot(data=mb, x="d", y="mean_success", hue="N", marker="o", ax=axes[0])
        axes[0].set_title("Mirror-benchmark ideal-bitstring success")
        axes[0].set_ylabel("mean success rate")
        tr = bsum[bsum.benchmark.eq("Transport_1QRB")]
        sns.lineplot(data=tr, x="d", y="mean_success", hue="N", marker="o", ax=axes[1])
        axes[1].set_title("Transport 1Q RB ideal-bitstring success")
        axes[1].set_ylabel("mean success rate")
        for ax in axes:
            ax.set_ylim(0, 1.05)
        fig.savefig(IMG / "benchmark_success.png", dpi=200)
        plt.close(fig)

    # Claims table.
    claims = [
        {"claim": "All XEB fidelity estimates are computed on exact count/probability key matches.", "supporting_artifact": "outputs/data_overview.json; outputs/method_fidelity_checklist.json", "evidence": f"{overview['xeb_paired_instances']} paired instances; matched-key invariant = {overview['all_xeb_pairs_have_20_matched_keys']}."},
        {"claim": "For N=40, mean XEB fidelity decays over the measured depth range.", "supporting_artifact": "outputs/depth_summary.csv; report/images/xeb_by_depth.png", "evidence": "Depth summary contains mean_xeb and ci95_across_instances for d=8..20."},
        {"claim": "At d=12, subset-XEB is available across N=16,24,32,40 and supports an N-scan.", "supporting_artifact": "outputs/n_scan_summary.csv; report/images/n_scan_xeb.png", "evidence": "n_scan_summary.csv gives one row for each available N at d=12."},
        {"claim": "MB/Transport files validate the separate ideal-bitstring success-rate workflow but do not by themselves provide ideal probabilities for XEB.", "supporting_artifact": "outputs/benchmark_success_estimates.csv; report/images/benchmark_success.png", "evidence": "Counts are compared with ideal_bitstring JSON files to compute direct success rates."},
        {"claim": "A full classical-approximability boundary cannot be re-derived from these files alone.", "supporting_artifact": "outputs/related_work_contract.json; report/report.md", "evidence": "Circuit topology/gate-count/classical simulation cost metadata are absent from data/."}
    ]
    pd.DataFrame(claims).to_csv(OUT / "claim_recovery_table.csv", index=False)

    update_inventory()
    print(json.dumps({"xeb_rows": len(xeb), "summary_rows": len(grp), "benchmark_rows": len(bench), "figures": len(list(IMG.glob('*.png')))}, indent=2))

if __name__ == "__main__":
    main()
