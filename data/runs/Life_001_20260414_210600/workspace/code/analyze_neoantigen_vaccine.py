import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def load_data():
    cell_pop = pd.read_csv(DATA / "cell-populations.csv")
    final_resp = pd.read_csv(DATA / "final-response-likelihoods.csv")
    sim_resp = pd.read_csv(DATA / "sim-specific-response-likelihoods.csv")
    runtime = pd.read_csv(DATA / "optimization_runtime_data.csv")
    selected = pd.read_csv(DATA / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
    vaccine_summary = pd.read_csv(DATA / "vaccine.budget-10.minsum.adaptive.csv")

    score_frames = []
    for path in sorted(DATA.glob("vaccine-elements.scores.100-cells.10x.rep-*.csv")):
        rep = int(re.search(r"rep-(\d+)", path.name).group(1))
        df = pd.read_csv(path)
        df["repetition"] = rep
        score_frames.append(df)
    scores = pd.concat(score_frames, ignore_index=True)
    return cell_pop, final_resp, sim_resp, runtime, selected, vaccine_summary, scores


def add_population_fields(df):
    split = df["population"].str.extract(r"(?P<simulation>[^,]+),\s*(?P<rep>\d+)")
    out = df.join(split)
    out["rep"] = out["rep"].astype(int)
    return out


def make_tables(cell_pop, final_resp, sim_resp, runtime, selected, vaccine_summary, scores):
    data_schema = []
    for name, df in [
        ("cell-populations", cell_pop),
        ("final-response-likelihoods", final_resp),
        ("sim-specific-response-likelihoods", sim_resp),
        ("optimization-runtime", runtime),
        ("selected-vaccine-elements", selected),
        ("vaccine-budget-summary", vaccine_summary),
        ("vaccine-element-scores-all-reps", scores),
    ]:
        data_schema.append({"dataset": name, "rows": int(len(df)), "columns": list(df.columns)})
    pd.DataFrame(data_schema).to_csv(OUT / "data_schema_summary.csv", index=False)

    composition_by_rep = (
        selected.sort_values(["repetition", "peptide"])
        .groupby(["simulation_name", "repetition"])["peptide"]
        .agg(lambda x: ";".join(sorted(map(str, x))))
        .reset_index()
        .rename(columns={"peptide": "selected_peptides"})
    )
    composition_by_rep["num_elements"] = composition_by_rep["selected_peptides"].str.split(";").apply(len)
    composition_by_rep.to_csv(OUT / "patient_vaccine_composition_table.csv", index=False)

    rep_sets = {int(rep): set(g["peptide"]) for rep, g in selected.groupby("repetition")}
    iou_rows = []
    for r1, r2 in combinations(sorted(rep_sets), 2):
        s1, s2 = rep_sets[r1], rep_sets[r2]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        iou_rows.append({"rep_a": r1, "rep_b": r2, "intersection": inter, "union": union, "iou": inter / union if union else np.nan})
    iou_df = pd.DataFrame(iou_rows)
    iou_df.to_csv(OUT / "iou_agreement_table.csv", index=False)

    reps = sorted(rep_sets)
    iou_matrix = pd.DataFrame(np.eye(len(reps)), index=reps, columns=reps)
    for _, row in iou_df.iterrows():
        iou_matrix.loc[row.rep_a, row.rep_b] = row.iou
        iou_matrix.loc[row.rep_b, row.rep_a] = row.iou
    iou_matrix.to_csv(OUT / "iou_matrix.csv")

    final_resp2 = add_population_fields(final_resp)
    response_summary = final_resp2.groupby("rep")["p_response"].agg(["mean", "median", "std", "min", "max"]).reset_index()
    response_summary.to_csv(OUT / "response_probability_summary_table.csv", index=False)

    overall_response_summary = {
        "mean_p_response": float(final_resp2["p_response"].mean()),
        "median_p_response": float(final_resp2["p_response"].median()),
        "std_p_response": float(final_resp2["p_response"].std()),
        "min_p_response": float(final_resp2["p_response"].min()),
        "max_p_response": float(final_resp2["p_response"].max()),
    }
    (OUT / "overall_response_summary.json").write_text(json.dumps(overall_response_summary, indent=2))

    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    coverage_rows = []
    for rep, g in final_resp2.groupby("rep"):
        for t in thresholds:
            coverage_rows.append(
                {
                    "rep": int(rep),
                    "threshold": t,
                    "coverage_ratio": float((g["p_response"] >= t).mean()),
                    "covered_cells": int((g["p_response"] >= t).sum()),
                    "total_cells": int(len(g)),
                }
            )
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(OUT / "coverage_ratio_table.csv", index=False)

    selected_map = {int(rep): set(g["peptide"]) for rep, g in selected.groupby("repetition")}
    per_cell_rows = []
    selected_cov_rows = []
    for rep, g in scores.groupby("repetition"):
        chosen = selected_map[int(rep)]
        gs = g[g["vaccine_element"].isin(chosen)].copy()
        agg = gs.groupby("cell_id")["p_no_response"].prod().reset_index(name="combined_p_no_response")
        agg["combined_p_response"] = 1 - agg["combined_p_no_response"]
        agg["repetition"] = int(rep)
        per_cell_rows.append(agg)
        for t in thresholds:
            selected_cov_rows.append(
                {
                    "rep": int(rep),
                    "threshold": t,
                    "coverage_ratio_selected_scores": float((agg["combined_p_response"] >= t).mean()),
                    "covered_cells": int((agg["combined_p_response"] >= t).sum()),
                    "total_cells": int(len(agg)),
                }
            )
    per_cell_df = pd.concat(per_cell_rows, ignore_index=True)
    per_cell_df.to_csv(OUT / "cell_level_selected_vaccine_response.csv", index=False)
    pd.DataFrame(selected_cov_rows).to_csv(OUT / "coverage_ratio_from_selected_scores.csv", index=False)

    runtime_summary = runtime.groupby("PopulationSize")["RunTime"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    runtime_summary.to_csv(OUT / "runtime_summary_table.csv", index=False)

    peptide_frequency = (
        selected.groupby("peptide").size().reset_index(name="times_selected").sort_values(["times_selected", "peptide"], ascending=[False, True])
    )
    peptide_frequency.to_csv(OUT / "peptide_selection_frequency.csv", index=False)

    immune_mean_by_element = scores.groupby("vaccine_element")["p_response"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)
    immune_mean_by_element.to_csv(OUT / "element_response_rankings.csv", index=False)

    claim_df = pd.DataFrame(
        [
            ["Each repetition contains a 10-element optimized vaccine set.", "outputs/patient_vaccine_composition_table.csv"],
            ["The optimized vaccine induces high per-cell response probabilities overall.", "outputs/response_probability_summary_table.csv; outputs/overall_response_summary.json"],
            ["Coverage decreases as the required response threshold becomes more stringent.", "outputs/coverage_ratio_table.csv; outputs/coverage_ratio_from_selected_scores.csv"],
            ["Composition stability is perfect across repetitions in this dataset (IoU = 1).", "outputs/iou_agreement_table.csv; outputs/iou_matrix.csv"],
            ["Optimization runtime increases with simulated population size.", "outputs/runtime_summary_table.csv"],
        ],
        columns=["claim", "supporting_artifacts"],
    )
    claim_df.to_csv(OUT / "claim_recovery_table.csv", index=False)

    analysis_summary = {
        "num_repetitions": int(selected["repetition"].nunique()),
        "elements_per_repetition": sorted(selected.groupby("repetition").size().astype(int).unique().tolist()),
        "selected_peptides": sorted(selected["peptide"].unique().tolist()),
        "mean_pairwise_iou": float(iou_df["iou"].mean()) if len(iou_df) else None,
        "median_pairwise_iou": float(iou_df["iou"].median()) if len(iou_df) else None,
        "overall_mean_p_response": float(final_resp2["p_response"].mean()),
        "overall_std_p_response": float(final_resp2["p_response"].std()),
        "coverage_final_threshold_05_mean": float(coverage_df.query("threshold == 0.5")["coverage_ratio"].mean()),
        "coverage_final_threshold_09_mean": float(coverage_df.query("threshold == 0.9")["coverage_ratio"].mean()),
        "coverage_selected_threshold_05_mean": float(pd.read_csv(OUT / "coverage_ratio_from_selected_scores.csv").query("threshold == 0.5")["coverage_ratio_selected_scores"].mean()),
        "runtime_slope_seconds_per_1000_cells": float(np.polyfit(runtime["PopulationSize"], runtime["RunTime"], 1)[0] * 1000),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2))

    return final_resp2, coverage_df, per_cell_df, runtime_summary, iou_matrix, immune_mean_by_element


def make_figures(final_resp2, coverage_df, per_cell_df, runtime_summary, iou_matrix, immune_mean_by_element):
    plt.figure(figsize=(8, 5))
    sns.histplot(final_resp2["p_response"], bins=30, kde=True, color="#4C72B0")
    plt.xlabel("Per-cell immune response probability")
    plt.ylabel("Count")
    plt.title("Distribution of final immune response probabilities")
    plt.tight_layout()
    plt.savefig(IMG / "response_distribution.png", dpi=200)
    plt.close()

    cov_sel = pd.read_csv(OUT / "coverage_ratio_from_selected_scores.csv")
    curve = coverage_df.groupby("threshold")["coverage_ratio"].agg(["mean", "std"]).reset_index()
    curve2 = cov_sel.groupby("threshold")["coverage_ratio_selected_scores"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(curve["threshold"], curve["mean"], marker="o", label="Final response file")
    plt.fill_between(curve["threshold"], curve["mean"] - curve["std"], curve["mean"] + curve["std"], alpha=0.2)
    plt.plot(curve2["threshold"], curve2["mean"], marker="s", label="Reconstructed from selected element scores")
    plt.fill_between(curve2["threshold"], curve2["mean"] - curve2["std"], curve2["mean"] + curve2["std"], alpha=0.2)
    plt.xlabel("Response probability threshold")
    plt.ylabel("Coverage ratio")
    plt.title("Tumor-cell coverage under increasing response thresholds")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / "coverage_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(iou_matrix, annot=True, vmin=0, vmax=1, cmap="viridis", cbar_kws={"label": "IoU"})
    plt.title("IoU of optimized vaccine compositions across repetitions")
    plt.xlabel("Repetition")
    plt.ylabel("Repetition")
    plt.tight_layout()
    plt.savefig(IMG / "iou_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(runtime_summary["PopulationSize"], runtime_summary["mean"], yerr=runtime_summary["std"].fillna(0), marker="o", capsize=4)
    plt.xlabel("Simulated population size")
    plt.ylabel("Optimization runtime (s)")
    plt.title("Runtime scaling of vaccine optimization")
    plt.tight_layout()
    plt.savefig(IMG / "runtime_scaling.png", dpi=200)
    plt.close()

    top = immune_mean_by_element.head(10).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top, x="vaccine_element", y="mean", color="#55A868")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Vaccine element")
    plt.ylabel("Mean single-element response probability")
    plt.title("Average response contribution by vaccine element")
    plt.tight_layout()
    plt.savefig(IMG / "element_response_rankings.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=per_cell_df, x="repetition", y="combined_p_response", color="#C44E52")
    plt.xlabel("Repetition")
    plt.ylabel("Combined per-cell response probability")
    plt.title("Per-cell response distributions after combining selected vaccine elements")
    plt.tight_layout()
    plt.savefig(IMG / "combined_response_by_repetition.png", dpi=200)
    plt.close()


def main():
    loaded = load_data()
    outputs = make_tables(*loaded)
    make_figures(*outputs)


if __name__ == "__main__":
    main()
