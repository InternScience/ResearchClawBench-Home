"""
Analysis pipeline for personalized neoantigen vaccine optimization
==================================================================

Reproduces the key quantitative outputs of a MinSum / budget=10 / adaptive
neoantigen vaccine selection pipeline:

  1. Per-cell immune response probability distribution.
  2. Tumor-cell coverage curve (fraction of cells with p_response >= t).
  3. IoU between selected vaccine compositions across replicates.
  4. Vaccine composition (mutation counts / weights).
  5. Optimization runtime vs. population size (per patient sample).
  6. Aggregated cell response probabilities directly from rep-0..9 score files.

All inputs are read from `data/`; all outputs go to `outputs/` and figures go
to `report/images/`.
"""
import itertools
import json
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
OUT  = ROOT / "outputs"
IMG  = ROOT / "report" / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# ---------------------------------------------------------------------------
# 1. Data overview
# ---------------------------------------------------------------------------
cell_pop = pd.read_csv(DATA / "cell-populations.csv")
selected = pd.read_csv(DATA / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
vaccine  = pd.read_csv(DATA / "vaccine.budget-10.minsum.adaptive.csv")
final_lk = pd.read_csv(DATA / "final-response-likelihoods.csv")
sim_lk   = pd.read_csv(DATA / "sim-specific-response-likelihoods.csv")
runtime  = pd.read_csv(DATA / "optimization_runtime_data.csv")

reps = sorted(cell_pop.repetition.unique())
print(f"Repetitions: {reps}")
print(f"Total cell-peptide presentations: {len(cell_pop)}")
print(f"Unique mutations across pop: {cell_pop.mutation.nunique()}")
print(f"Unique HLAs: {cell_pop.presented_hlas.nunique()} -> {sorted(cell_pop.presented_hlas.unique())}")
print(f"Vaccine elements (counts/weights):\n{vaccine}")

# ---------------------------------------------------------------------------
# 2. Per-cell immune response probability distribution
# ---------------------------------------------------------------------------
# final-response-likelihoods.csv = "final" probability per cell aggregated over
# repetitions; sim-specific-response-likelihoods.csv = same per replicate.
final_summary = final_lk["p_response"].describe()
final_summary.to_csv(OUT / "per_cell_response_stats.csv")
print(f"\nFinal p_response summary:\n{final_summary}")

sim_lk["repetition"] = sim_lk["vaccine"].str.extract(r"rep-(\d+)").astype(int)
sim_summary = (sim_lk.groupby("repetition")["p_response"]
                       .agg(["count","mean","std","median",
                             lambda x: (x >= 0.5).mean(),
                             lambda x: (x >= 0.9).mean()])
                       .rename(columns={"<lambda_0>": "frac_ge_0.5",
                                        "<lambda_1>": "frac_ge_0.9"}))
sim_summary.to_csv(OUT / "per_cell_response_stats_per_rep.csv")
print(f"\nPer-rep p_response summary:\n{sim_summary}")

# ---------------------------------------------------------------------------
# 3. Coverage curve
# ---------------------------------------------------------------------------
thresholds = np.linspace(0, 1, 101)
cov_final = [(final_lk["p_response"] >= t).mean() for t in thresholds]
cov_per_rep = {}
for r, grp in sim_lk.groupby("repetition"):
    cov_per_rep[r] = [(grp["p_response"] >= t).mean() for t in thresholds]
cov_df = pd.DataFrame({"threshold": thresholds, "coverage_final": cov_final})
for r, vals in cov_per_rep.items():
    cov_df[f"coverage_rep{r}"] = vals
cov_df.to_csv(OUT / "coverage_curve.csv", index=False)
print(f"\nCoverage at p>=0.5: {(final_lk.p_response>=0.5).mean():.3f}")
print(f"Coverage at p>=0.9: {(final_lk.p_response>=0.9).mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Selected-vaccine composition per replicate, IoU matrix
# ---------------------------------------------------------------------------
selected_per_rep = (selected.groupby("repetition")["peptide"]
                            .apply(lambda s: sorted(set(s)))
                            .to_dict())
selected_df = pd.DataFrame(
    [{"repetition": r, "n_elements": len(v),
      "elements": ";".join(v)} for r, v in selected_per_rep.items()]
).sort_values("repetition")
selected_df.to_csv(OUT / "selected_vaccine_per_replicate.csv", index=False)

reps_sorted = sorted(selected_per_rep.keys())
n = len(reps_sorted)
iou = np.zeros((n, n))
for i, ri in enumerate(reps_sorted):
    for j, rj in enumerate(reps_sorted):
        a, b = set(selected_per_rep[ri]), set(selected_per_rep[rj])
        iou[i, j] = len(a & b) / len(a | b) if a | b else 1.0
iou_df = pd.DataFrame(iou, index=[f"rep{r}" for r in reps_sorted],
                      columns=[f"rep{r}" for r in reps_sorted])
iou_df.to_csv(OUT / "iou_matrix.csv")
upper = iou[np.triu_indices(n, k=1)]
print(f"\nIoU off-diagonal: mean={upper.mean():.3f}, min={upper.min():.3f}, max={upper.max():.3f}")
with open(OUT / "iou_summary.json", "w") as f:
    json.dump({"mean_off_diag_iou": float(upper.mean()),
               "min_off_diag_iou":  float(upper.min()),
               "max_off_diag_iou":  float(upper.max()),
               "n_replicates": int(n)}, f, indent=2)

# Composition table (counts of each mutation across replicates)
pep_counts = (selected.groupby("peptide")["repetition"].nunique()
                       .rename("rep_count").reset_index()
                       .sort_values("rep_count", ascending=False))
pep_counts.to_csv(OUT / "vaccine_composition.csv", index=False)
print(f"\nMutation rep_counts:\n{pep_counts}")

# ---------------------------------------------------------------------------
# 5. Aggregated cell-level response probabilities from score files
# ---------------------------------------------------------------------------
score_files = sorted(DATA.glob("vaccine-elements.scores.100-cells.10x.rep-*.csv"))
agg_rows = []
for sf in score_files:
    rep = int(re.search(r"rep-(\d+)", sf.name).group(1))
    df = pd.read_csv(sf)
    # Aggregate cell-level response across vaccine elements (independent events):
    # P(no_response | cell) = prod_e P(no_response | cell, element)
    # log P(no_response | cell) = sum_e log P(no_response | cell, element)
    by_cell = df.groupby("cell_id")["log_p_no_response"].sum()
    p_resp = 1 - np.exp(by_cell)
    rep_df = pd.DataFrame({
        "repetition": rep,
        "cell_id": p_resp.index,
        "p_response_agg": p_resp.values
    })
    agg_rows.append(rep_df)
agg_cell = pd.concat(agg_rows, ignore_index=True)
agg_cell.to_csv(OUT / "aggregated_cell_response.csv", index=False)
print(f"\nAggregated p_response per rep (mean):\n"
      f"{agg_cell.groupby('repetition').p_response_agg.mean()}")

# ---------------------------------------------------------------------------
# 6. Runtime summary
# ---------------------------------------------------------------------------
runtime_summary = (runtime.groupby("PopulationSize")["RunTime"]
                          .agg(["mean","std","min","max"]))
runtime_summary.to_csv(OUT / "runtime_summary.csv")
print(f"\nRuntime summary:\n{runtime_summary}")

# ===========================================================================
# Figures
# ===========================================================================
# Fig 1 — Data overview (presented peptides per cell, mutation frequency, HLA)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
peps_per_cell = cell_pop.groupby(["repetition","cell_ids"]).size()
axes[0].hist(peps_per_cell.values, bins=20, color="steelblue", edgecolor="black")
axes[0].set_xlabel("Presented peptides per cell")
axes[0].set_ylabel("Number of (rep, cell) pairs")
axes[0].set_title("Presentation load")

mut_freq = cell_pop.mutation.value_counts()
axes[1].bar(range(len(mut_freq)), mut_freq.values, color="indianred")
axes[1].set_xticks(range(len(mut_freq)))
axes[1].set_xticklabels(mut_freq.index, rotation=90, fontsize=8)
axes[1].set_ylabel("Total presentations")
axes[1].set_title("Mutation presentation frequency")

hla_freq = cell_pop.presented_hlas.value_counts()
axes[2].bar(hla_freq.index, hla_freq.values, color="seagreen")
axes[2].set_xticklabels(hla_freq.index, rotation=45, ha="right")
axes[2].set_ylabel("Total presentations")
axes[2].set_title("HLA-allele usage")

plt.tight_layout()
plt.savefig(IMG / "fig1_data_overview.png", dpi=150)
plt.close()

# Fig 2 — Per-cell response distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(final_lk["p_response"], bins=30, color="navy", alpha=0.85,
             edgecolor="white")
axes[0].axvline(final_lk.p_response.mean(), color="red", linestyle="--",
                label=f"mean = {final_lk.p_response.mean():.3f}")
axes[0].axvline(final_lk.p_response.median(), color="orange", linestyle=":",
                label=f"median = {final_lk.p_response.median():.3f}")
axes[0].set_xlabel("Per-cell P(immune response)")
axes[0].set_ylabel("Number of cells")
axes[0].set_title("Final response likelihood (vaccine = MinSum.budget-10.adaptive)")
axes[0].legend()

# Per-rep box / violin
sns.violinplot(data=sim_lk, x="repetition", y="p_response",
               ax=axes[1], inner="quartile", cut=0, palette="viridis")
axes[1].set_xlabel("Replicate")
axes[1].set_ylabel("Per-cell P(response)")
axes[1].set_title("Replicate-specific p_response distributions")
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(IMG / "fig2_response_distribution.png", dpi=150)
plt.close()

# Fig 3 — Coverage curves
fig, ax = plt.subplots(figsize=(7.5, 5))
for r in reps_sorted:
    ax.plot(thresholds, cov_per_rep[r], color="lightgray", lw=1)
ax.plot(thresholds, cov_final, color="crimson", lw=2.5, label="Final (all reps)")
ax.axvline(0.5, color="black", linestyle=":", alpha=0.5)
ax.axvline(0.9, color="black", linestyle=":", alpha=0.5)
ax.set_xlabel("Response-probability threshold t")
ax.set_ylabel("Fraction of tumor cells with P(response) ≥ t")
ax.set_title("Tumor-cell coverage vs. response-probability threshold")
ax.legend()
plt.tight_layout()
plt.savefig(IMG / "fig3_coverage_curve.png", dpi=150)
plt.close()

# Fig 4 — IoU heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(iou_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
            cbar_kws={"label": "IoU"}, ax=ax)
ax.set_title("Pairwise IoU of selected vaccine elements across replicates")
plt.tight_layout()
plt.savefig(IMG / "fig4_iou_heatmap.png", dpi=150)
plt.close()

# Fig 5 — Vaccine composition
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(vaccine.peptide, vaccine.counts, color="purple")
axes[0].set_xlabel("Vaccine element (mutation)")
axes[0].set_ylabel("Times selected (out of 10 reps)")
axes[0].set_title("Consensus MinSum / budget=10 vaccine composition")
axes[0].tick_params(axis="x", rotation=45)

# Element x replicate matrix
mat = (selected.assign(present=1)
                .pivot_table(index="peptide", columns="repetition",
                             values="present", aggfunc="max", fill_value=0))
sns.heatmap(mat, ax=axes[1], cmap="Greens", cbar=False,
            linewidths=0.5, linecolor="white",
            annot=False)
axes[1].set_xlabel("Replicate")
axes[1].set_ylabel("Mutation")
axes[1].set_title("Per-replicate inclusion (1 = selected)")

plt.tight_layout()
plt.savefig(IMG / "fig5_vaccine_composition.png", dpi=150)
plt.close()

# Fig 6 — Runtime vs population size
fig, ax = plt.subplots(figsize=(7.5, 5))
samples = sorted(runtime.SampleID.unique())
palette = sns.color_palette("tab10", n_colors=len(samples))
for i, s in enumerate(samples):
    sub = runtime[runtime.SampleID == s]
    ax.plot(sub.PopulationSize, sub.RunTime, marker="o",
            label=f"Sample {s}", color=palette[i])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Cell population size")
ax.set_ylabel("Optimization runtime [s]")
ax.set_title("Runtime scaling of MinSum-budget-10-adaptive optimizer")
ax.legend(loc="best", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(IMG / "fig6_runtime_vs_population.png", dpi=150)
plt.close()

print("\nAll figures written to", IMG)
print("All artifacts written to", OUT)
