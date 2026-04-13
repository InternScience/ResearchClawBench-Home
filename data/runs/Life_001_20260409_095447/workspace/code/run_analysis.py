from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyPDF2 import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"
RELATED_WORK_DIR = ROOT / "related_work"


def parse_rep_from_name(name: str) -> int:
    match = re.search(r"rep-(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse replicate from {name}")
    return int(match.group(1))


def parse_population_rep(population: str) -> int:
    return int(str(population).split(",")[-1].strip())


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_literature_notes() -> pd.DataFrame:
    records = []
    for pdf_path in sorted(RELATED_WORK_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
        compact = re.sub(r"\s+", " ", text).strip()
        records.append(
            {
                "paper": pdf_path.name,
                "pages": len(reader.pages),
                "excerpt": compact[:900],
            }
        )
    literature = pd.DataFrame(records)
    literature.to_csv(OUTPUT_DIR / "literature_notes.csv", index=False)
    return literature


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "cells": pd.read_csv(DATA_DIR / "cell-populations.csv"),
        "final": pd.read_csv(DATA_DIR / "final-response-likelihoods.csv"),
        "runtime": pd.read_csv(DATA_DIR / "optimization_runtime_data.csv"),
        "selected": pd.read_csv(
            DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv"
        ),
        "sim_specific": pd.read_csv(DATA_DIR / "sim-specific-response-likelihoods.csv"),
        "vaccine_budget": pd.read_csv(DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv"),
    }
    score_frames = []
    for score_path in sorted(DATA_DIR.glob("vaccine-elements.scores.*.csv")):
        rep = parse_rep_from_name(score_path.name)
        frame = pd.read_csv(score_path)
        frame["rep"] = rep
        score_frames.append(frame)
    data["scores"] = pd.concat(score_frames, ignore_index=True)
    return data


def analyze_vaccine_consistency(selected: pd.DataFrame, vaccine_budget: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_sets = (
        selected.groupby("repetition")["peptide"].apply(lambda s: sorted(set(s))).reset_index()
    )
    canonical_set = set(vaccine_budget["peptide"])
    rows = []
    for _, row_i in selected_sets.iterrows():
        rep_i = int(row_i["repetition"])
        set_i = set(row_i["peptide"])
        rows.append(
            {
                "rep": rep_i,
                "set_size": len(set_i),
                "iou_vs_canonical": len(set_i & canonical_set) / len(set_i | canonical_set),
            }
        )
    pairwise = []
    for _, row_i in selected_sets.iterrows():
        rep_i = int(row_i["repetition"])
        set_i = set(row_i["peptide"])
        for _, row_j in selected_sets.iterrows():
            rep_j = int(row_j["repetition"])
            if rep_j <= rep_i:
                continue
            set_j = set(row_j["peptide"])
            iou = len(set_i & set_j) / len(set_i | set_j)
            pairwise.append({"rep_i": rep_i, "rep_j": rep_j, "iou": iou})
    summary_df = pd.DataFrame(rows)
    pairwise_df = pd.DataFrame(pairwise)
    summary_df.to_csv(OUTPUT_DIR / "vaccine_consistency_summary.csv", index=False)
    pairwise_df.to_csv(OUTPUT_DIR / "pairwise_iou.csv", index=False)
    return summary_df, pairwise_df


def aggregate_cell_response(scores: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_map = selected.groupby("repetition")["peptide"].apply(set).to_dict()
    rows = []
    for rep, rep_scores in scores.groupby("rep"):
        chosen = selected_map[int(rep)]
        rep_scores = rep_scores.copy()
        rep_scores["is_selected"] = rep_scores["vaccine_element"].isin(chosen)
        selected_scores = rep_scores[rep_scores["is_selected"]]
        cell_summary = (
            selected_scores.groupby("cell_id")
            .agg(
                mean_selected_response=("p_response", "mean"),
                max_selected_response=("p_response", "max"),
                activated_elements=("p_response", lambda s: int((s > 0.5).sum())),
            )
            .reset_index()
        )
        cell_summary["rep"] = int(rep)
        cell_summary["coverage_0_5"] = cell_summary["max_selected_response"] >= 0.5
        cell_summary["coverage_0_9"] = cell_summary["max_selected_response"] >= 0.9
        rows.append(cell_summary)
    all_cells = pd.concat(rows, ignore_index=True)
    all_cells.to_csv(OUTPUT_DIR / "cell_level_selected_response.csv", index=False)
    return all_cells


def analyze_final_response(final_df: pd.DataFrame) -> pd.DataFrame:
    final_df = final_df.copy()
    final_df["rep"] = final_df["population"].map(parse_population_rep)
    final_df["covered_0_5"] = final_df["p_response"] >= 0.5
    final_df["covered_0_9"] = final_df["p_response"] >= 0.9
    summary = (
        final_df.groupby("rep")
        .agg(
            mean_p_response=("p_response", "mean"),
            median_p_response=("p_response", "median"),
            min_p_response=("p_response", "min"),
            coverage_ratio_0_5=("covered_0_5", "mean"),
            coverage_ratio_0_9=("covered_0_9", "mean"),
            mean_presented_peptides=("num_presented_peptides", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "final_response_summary.csv", index=False)
    return final_df


def analyze_cell_populations(cells: pd.DataFrame) -> pd.DataFrame:
    cell_counts = (
        cells.groupby("repetition")
        .agg(
            num_rows=("cell_ids", "size"),
            num_cells=("cell_ids", "nunique"),
            unique_mutations=("mutation", "nunique"),
            unique_peptides=("presented_peptides", "nunique"),
            mean_peptides_per_cell=("presented_peptides", lambda s: len(s) / s.nunique()),
        )
        .reset_index()
    )
    cell_counts.to_csv(OUTPUT_DIR / "cell_population_summary.csv", index=False)
    return cell_counts


def analyze_runtime(runtime: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    runtime = runtime.copy()
    runtime_summary = (
        runtime.groupby("PopulationSize")
        .agg(
            mean_runtime=("RunTime", "mean"),
            std_runtime=("RunTime", "std"),
            min_runtime=("RunTime", "min"),
            max_runtime=("RunTime", "max"),
        )
        .reset_index()
    )
    runtime_summary.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)

    x = np.log10(runtime["PopulationSize"].to_numpy(dtype=float))
    y = np.log10(runtime["RunTime"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    runtime_fit = {
        "log10_slope": float(slope),
        "log10_intercept": float(intercept),
        "r2": float(r2),
    }
    with open(OUTPUT_DIR / "runtime_fit.json", "w", encoding="utf-8") as f:
        json.dump(runtime_fit, f, indent=2)
    return runtime_summary, runtime_fit


def make_figures(
    final_df: pd.DataFrame,
    cell_summary: pd.DataFrame,
    pairwise_iou: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    runtime_raw: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(9, 6))
    sns.violinplot(data=final_df, x="rep", y="p_response", inner="quartile", color="#4C72B0")
    plt.xlabel("Simulation replicate")
    plt.ylabel("Per-cell immune response probability")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "response_probability_by_replicate.png", dpi=200)
    plt.close()

    coverage = (
        cell_summary.groupby("rep")[["coverage_0_5", "coverage_0_9"]]
        .mean()
        .reset_index()
        .melt(id_vars="rep", var_name="threshold", value_name="coverage_ratio")
    )
    coverage["threshold"] = coverage["threshold"].map(
        {"coverage_0_5": "max p >= 0.5", "coverage_0_9": "max p >= 0.9"}
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(data=coverage, x="rep", y="coverage_ratio", hue="threshold", palette="deep")
    plt.ylim(0, 1.0)
    plt.xlabel("Simulation replicate")
    plt.ylabel("Coverage ratio")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "coverage_ratio_by_replicate.png", dpi=200)
    plt.close()

    iou_matrix = np.ones((10, 10))
    for _, row in pairwise_iou.iterrows():
        i = int(row["rep_i"])
        j = int(row["rep_j"])
        iou_matrix[i, j] = row["iou"]
        iou_matrix[j, i] = row["iou"]
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(
        iou_matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
    )
    plt.xlabel("Replicate")
    plt.ylabel("Replicate")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "pairwise_iou_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    sns.lineplot(
        data=runtime_summary,
        x="PopulationSize",
        y="mean_runtime",
        marker="o",
        linewidth=2.5,
        color="#DD8452",
    )
    sns.scatterplot(
        data=runtime_raw,
        x="PopulationSize",
        y="RunTime",
        hue="SampleID",
        palette="tab10",
        s=90,
        alpha=0.8,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Population size")
    plt.ylabel("Runtime (seconds)")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "runtime_scaling.png", dpi=200)
    plt.close()


def write_key_results(
    final_df: pd.DataFrame,
    cell_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    pairwise_iou: pd.DataFrame,
    runtime_fit: dict[str, float],
    vaccine_budget: pd.DataFrame,
) -> None:
    result = {
        "budget": int(vaccine_budget["counts"].iloc[0]),
        "selected_elements": vaccine_budget["peptide"].tolist(),
        "mean_final_response": float(final_df["p_response"].mean()),
        "median_final_response": float(final_df["p_response"].median()),
        "min_final_response": float(final_df["p_response"].min()),
        "coverage_ratio_p_ge_0_5": float((final_df["p_response"] >= 0.5).mean()),
        "coverage_ratio_p_ge_0_9": float((final_df["p_response"] >= 0.9).mean()),
        "cell_level_max_selected_ge_0_5": float(cell_summary["coverage_0_5"].mean()),
        "cell_level_max_selected_ge_0_9": float(cell_summary["coverage_0_9"].mean()),
        "mean_iou_vs_canonical": float(consistency_summary["iou_vs_canonical"].mean()),
        "pairwise_iou_mean": float(pairwise_iou["iou"].mean()),
        "pairwise_iou_min": float(pairwise_iou["iou"].min()),
        "runtime_loglog_slope": float(runtime_fit["log10_slope"]),
        "runtime_loglog_r2": float(runtime_fit["r2"]),
    }
    with open(OUTPUT_DIR / "key_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    ensure_dirs()
    literature = load_literature_notes()
    data = load_data()

    consistency_summary, pairwise_iou = analyze_vaccine_consistency(
        data["selected"], data["vaccine_budget"]
    )
    cell_summary = aggregate_cell_response(data["scores"], data["selected"])
    final_df = analyze_final_response(data["final"])
    analyze_cell_populations(data["cells"])
    runtime_summary, runtime_fit = analyze_runtime(data["runtime"])

    make_figures(
        final_df=final_df,
        cell_summary=cell_summary,
        pairwise_iou=pairwise_iou,
        runtime_summary=runtime_summary,
        runtime_raw=data["runtime"],
    )
    write_key_results(
        final_df=final_df,
        cell_summary=cell_summary,
        consistency_summary=consistency_summary,
        pairwise_iou=pairwise_iou,
        runtime_fit=runtime_fit,
        vaccine_budget=data["vaccine_budget"],
    )
    literature.to_csv(OUTPUT_DIR / "literature_summary.csv", index=False)


if __name__ == "__main__":
    main()
