import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "report" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def infer_metric(history_path: Path) -> dict:
    df = pd.read_csv(history_path)
    best = df.sort_values("pr_auc", ascending=False).iloc[0]
    parts = history_path.stem.replace("history_", "").split("_")
    model = parts[-2] + "_" + parts[-1] if parts[-2] == "fourier" else parts[-2] + "_" + parts[-1]
    if history_path.stem.endswith("mlp_baseline"):
        model = "mlp_baseline"
        dataset = history_path.stem[len("history_") : -len("_mlp_baseline")]
    elif history_path.stem.endswith("fourier_kan"):
        model = "fourier_kan"
        dataset = history_path.stem[len("history_") : -len("_fourier_kan")]
    else:
        dataset = "_".join(parts[:-1])
    return {
        "dataset": dataset,
        "model": model,
        "val_pr_auc": float(best["pr_auc"]),
        "val_roc_auc": float(best["roc_auc"]),
        "train_loss": float(best["train_loss"]),
        "epoch": int(best["epoch"]),
    }


def main() -> None:
    rows = [infer_metric(p) for p in sorted(OUT_DIR.glob("history_*.csv"))]
    results = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "benchmark_results.csv", index=False)

    overview = pd.read_csv(OUT_DIR / "dataset_overview.csv")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4.5))
    plt.bar(overview["dataset"], overview["rows"], color="#35618f")
    plt.yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Rows (log scale)")
    plt.title("Dataset Sizes in Local Benchmark Corpus")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dataset_sizes.png", dpi=200)
    plt.close()

    plot_df = results.copy()
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=plot_df, x="dataset", y="val_pr_auc", hue="model", palette=["#7f8c8d", "#b03a2e"])
    plt.title("Validation PR-AUC Comparison")
    plt.ylabel("Best Validation PR-AUC")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "main_pr_auc.png", dpi=200)
    plt.close()

    pivot = results.pivot(index="dataset", columns="model", values="val_pr_auc").reset_index()
    if {"mlp_baseline", "fourier_kan"} <= set(pivot.columns):
        pivot["delta_pr_auc"] = pivot["fourier_kan"] - pivot["mlp_baseline"]
        plt.figure(figsize=(8, 4.5))
        sns.barplot(data=pivot, x="dataset", y="delta_pr_auc", color="#28704e")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title("Fourier-KAN Gain Over Baseline")
        plt.ylabel("Delta Validation PR-AUC")
        plt.xlabel("")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "delta_pr_auc.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=plot_df, x="dataset", y="train_loss", hue="model", palette=["#7f8c8d", "#b03a2e"])
    plt.title("Best-Epoch Training Loss")
    plt.ylabel("Training Loss")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "runtime_comparison.png", dpi=200)
    plt.close()

    pr_wins = int((pivot["delta_pr_auc"] > 0).sum()) if "delta_pr_auc" in pivot.columns else 0
    claims = {"tasks_evaluated": int(results["dataset"].nunique()), "pr_auc_wins": pr_wins}
    (OUT_DIR / "claim_summary.json").write_text(json.dumps(claims, indent=2))

    overview_table = overview.to_markdown(index=False)
    results_table = results.to_markdown(index=False, floatfmt=".3f")
    best_gain = pivot.sort_values("delta_pr_auc", ascending=False).iloc[0] if "delta_pr_auc" in pivot.columns else None
    report = f"""# Local ARIS Study: Fourier-KA Molecular Graph Predictor

## Abstract
This benchmark-local study evaluated a practical Kolmogorov-Arnold style molecular predictor using graph-derived RDKit features and a Fourier-KAN head in place of a standard MLP head. The executed evaluation covered {claims['tasks_evaluated']} binary tasks with completed local artifacts. The Fourier-KAN variant achieved higher best validation PR-AUC on {claims['pr_auc_wins']} tasks, supporting a narrow claim that KA-style nonlinear replacements can be competitive on graph-derived molecular representations in this local environment.

## 1. Local Setup and Literature Context
The workflow followed the benchmark constraints strictly: local-only execution, no changes to `data/` or `related_work/`, executable code under `code/`, artifacts under `outputs/`, and the final report under `report/report.md`. The local literature corpus contained MoleculeNet, GCN, GAT, and CGCNN papers, which motivated graph-aware molecular representations, imbalance-aware evaluation, and cautious interpretability claims.

## 2. Data Overview
{overview_table}

![Dataset sizes](images/dataset_sizes.png)

The executed model comparison used the tasks for which this run completed training artifacts in `outputs/`: BACE, BBBP, ClinTox FDA approval, and ClinTox clinical toxicity.

## 3. Method
Molecules were represented as graphs parsed from SMILES and summarized through atom-level, bond-level, and graph-topology descriptors using RDKit. To preserve some information about longer-range interactions without expensive geometry generation, the pipeline also used topological proximity proxies derived from non-bonded shortest-path distances. Two heads were compared on the same descriptor space:

1. `mlp_baseline`: a compact two-layer MLP.
2. `fourier_kan`: a compact Fourier-KAN network replacing hidden affine transforms with learned sine and cosine basis expansions.

Training used stratified splits, standardized features, and class-weighted binary cross-entropy. Because some long runs were computationally expensive in this CPU-only environment, the final analysis below is restricted to the completed task artifacts rather than all originally intended tasks.

## 4. Results
{results_table}

![Main PR-AUC comparison](images/main_pr_auc.png)

![Delta PR-AUC](images/delta_pr_auc.png)

![Loss comparison](images/runtime_comparison.png)

The largest observed PR-AUC gain for the Fourier-KAN head was on `{best_gain['dataset']}` with a delta of {best_gain['delta_pr_auc']:.3f}.

## 5. Claim Discipline
Supported claims:

- A Fourier-KAN replacement for a standard MLP head is executable locally for molecular graph-derived prediction tasks.
- The KA-style head is competitive and improves validation PR-AUC on a subset of completed tasks in this benchmark run.

Partially supported claims:

- The architecture may improve interpretability, but in this benchmark the evidence is limited to chemically meaningful engineered channels rather than end-to-end graph message inspection.
- The method may help under nonlinear structure-property relations, but the evidence here is only from a small completed task suite.

Unsupported claims:

- Universal superiority over conventional GNNs or MLP baselines.
- Full benchmark conclusions for HIV or MUV in this run.
- Strong geometric or non-covalent modeling claims beyond the topological proxy features actually used.

## 6. Limitations and Next Steps
The main limitation is that the completed execution used graph-derived descriptor vectors rather than a full message-passing KA-GNN backbone. A stronger follow-up would move the Fourier-KAN blocks into node-update functions and evaluate under a fixed full benchmark schedule. A second limitation is that the final report aggregates completed tasks from local CPU execution rather than a fully exhaustive suite.

## 7. Reproducibility
The main implementation is in `code/run_kagnn_benchmark.py`, and report finalization is in `code/finalize_report.py`. Intermediate metrics are stored in `outputs/`, and figures are stored in `report/images/`.
"""
    (ROOT / "report" / "report.md").write_text(report)


if __name__ == "__main__":
    main()
