from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def base_counts(kmer: str) -> dict[str, int]:
    return {base: kmer.count(base) for base in "ACGT"}


def add_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    k = len(out["kmer"].iloc[0])
    for i in range(k):
        out[f"pos_{i+1}"] = out["kmer"].str[i]
    for base in "ACGT":
        out[f"count_{base}"] = out["kmer"].str.count(base)
    return out


def summarize_pore_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_specs = [
        ("dna_r9.4.1_400bps_6mer_uncalled4.csv", "DNA R9.4.1", "DNA", 6),
        ("dna_r10.4.1_400bps_9mer_uncalled4.csv", "DNA R10.4.1", "DNA", 9),
        ("rna_r9.4.1_70bps_5mer_uncalled4.csv", "RNA R9.4.1", "RNA", 5),
        ("rna004_130bps_9mer_uncalled4.csv", "RNA004", "RNA", 9),
    ]

    summary_rows = []
    position_rows = []

    for filename, chemistry, molecule, k in model_specs:
        df = pd.read_csv(DATA_DIR / filename)
        df = add_position_columns(df)
        summary_rows.append(
            {
                "chemistry": chemistry,
                "molecule": molecule,
                "k": k,
                "n_kmers": len(df),
                "current_mean_mean": df["current_mean"].mean(),
                "current_mean_std": df["current_mean"].std(),
                "current_std_mean": df["current_std"].mean(),
                "dwell_time_mean": df["dwell_time"].mean(),
                "dwell_time_std": df["dwell_time"].std(),
            }
        )
        for pos in range(1, k + 1):
            grouped = (
                df.groupby(f"pos_{pos}")
                .agg(
                    mean_current_mean=("current_mean", "mean"),
                    mean_current_std=("current_mean", "std"),
                    mean_dwell_time=("dwell_time", "mean"),
                )
                .reset_index()
                .rename(columns={f"pos_{pos}": "base"})
            )
            for _, row in grouped.iterrows():
                position_rows.append(
                    {
                        "chemistry": chemistry,
                        "molecule": molecule,
                        "k": k,
                        "position": pos,
                        "base": row["base"],
                        "mean_current_mean": row["mean_current_mean"],
                        "mean_current_std": row["mean_current_std"],
                        "mean_dwell_time": row["mean_dwell_time"],
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    position_df = pd.DataFrame(position_rows)
    summary_df.to_csv(OUTPUT_DIR / "pore_model_summary.csv", index=False)
    position_df.to_csv(OUTPUT_DIR / "pore_model_position_effects.csv", index=False)
    return summary_df, position_df


def analyze_performance() -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(DATA_DIR / "performance_summary.csv")
    baseline = perf[perf["Tool"] == "Uncalled4"][
        ["Chemistry", "Time_min", "FileSize_MB"]
    ].rename(
        columns={
            "Time_min": "uncalled4_time_min",
            "FileSize_MB": "uncalled4_size_mb",
        }
    )
    merged = perf.merge(baseline, on="Chemistry", how="left")
    merged["speedup_vs_uncalled4"] = merged["Time_min"] / merged["uncalled4_time_min"]
    merged["size_ratio_vs_uncalled4"] = merged["FileSize_MB"] / merged["uncalled4_size_mb"]
    merged.to_csv(OUTPUT_DIR / "performance_with_ratios.csv", index=False)

    summary = (
        merged[merged["Tool"] != "Uncalled4"]
        .groupby("Tool", as_index=False)[["speedup_vs_uncalled4", "size_ratio_vs_uncalled4"]]
        .median()
        .rename(
            columns={
                "speedup_vs_uncalled4": "median_runtime_ratio_vs_uncalled4",
                "size_ratio_vs_uncalled4": "median_filesize_ratio_vs_uncalled4",
            }
        )
    )
    summary.to_csv(OUTPUT_DIR / "performance_ratio_summary.csv", index=False)
    return perf, merged


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        vals.append(metric_fn(sample_y, y_score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_m6a() -> tuple[pd.DataFrame, dict]:
    labels = pd.read_csv(DATA_DIR / "m6a_labels.csv")
    unc = pd.read_csv(DATA_DIR / "m6a_predictions_uncalled4.csv").rename(
        columns={"probability": "uncalled4_probability"}
    )
    nano = pd.read_csv(DATA_DIR / "m6a_predictions_nanopolish.csv").rename(
        columns={"probability": "nanopolish_probability"}
    )
    merged = labels.merge(unc, on="site_id").merge(nano, on="site_id")

    y = merged["label"].to_numpy()
    scores = {
        "Uncalled4": merged["uncalled4_probability"].to_numpy(),
        "Nanopolish": merged["nanopolish_probability"].to_numpy(),
    }

    metric_rows = []
    curve_payload: dict[str, dict] = {}
    prevalence = float(merged["label"].mean())

    for name, score in scores.items():
        auprc = average_precision_score(y, score)
        auroc = roc_auc_score(y, score)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y, score)
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y, score)
        metric_rows.append(
            {
                "model": name,
                "auprc": auprc,
                "auroc": auroc,
                "positive_prevalence": prevalence,
                "auprc_ci_low": bootstrap_metric(y, score, average_precision_score)[0],
                "auprc_ci_high": bootstrap_metric(y, score, average_precision_score)[1],
                "auroc_ci_low": bootstrap_metric(y, score, roc_auc_score)[0],
                "auroc_ci_high": bootstrap_metric(y, score, roc_auc_score)[1],
            }
        )
        curve_payload[name] = {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "pr_thresholds": pr_thresholds.tolist(),
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
        }

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["auprc_lift_vs_prevalence"] = metrics_df["auprc"] / prevalence
    metrics_df.to_csv(OUTPUT_DIR / "m6a_model_metrics.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "m6a_joined_predictions.csv", index=False)
    with open(OUTPUT_DIR / "m6a_curves.json", "w", encoding="utf-8") as f:
        json.dump(curve_payload, f)
    return merged, curve_payload


def plot_performance(perf: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=perf, x="Chemistry", y="Time_min", hue="Tool", ax=axes[0])
    axes[0].set_title("Runtime Across Chemistries")
    axes[0].set_ylabel("Alignment time (minutes)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=perf, x="Chemistry", y="FileSize_MB", hue="Tool", ax=axes[1])
    axes[1].set_title("Output File Size Across Chemistries")
    axes[1].set_ylabel("File size (MB)")
    axes[1].tick_params(axis="x", rotation=30)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, title="Tool", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "performance_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_m6a(curves: dict, metrics_path: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for model in ["Uncalled4", "Nanopolish"]:
        color = "#1b9e77" if model == "Uncalled4" else "#d95f02"
        pr = curves[model]
        auprc = metrics.loc[metrics["model"] == model, "auprc"].iloc[0]
        auroc = metrics.loc[metrics["model"] == model, "auroc"].iloc[0]
        axes[0].plot(pr["recall"], pr["precision"], label=f"{model} AUPRC={auprc:.3f}", color=color, linewidth=2.5)
        axes[1].plot(pr["fpr"], pr["tpr"], label=f"{model} AUROC={auroc:.3f}", color=color, linewidth=2.5)
    prevalence = metrics["positive_prevalence"].iloc[0]
    axes[0].axhline(prevalence, linestyle="--", color="black", alpha=0.7, label=f"Prevalence={prevalence:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.7)
    axes[0].set_title("m6A Precision-Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_title("m6A ROC")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "m6a_detection_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pore_effects(position_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1.3]})

    sns.barplot(
        data=summary_df,
        x="chemistry",
        y="current_mean_std",
        hue="molecule",
        ax=axes[0],
        palette={"DNA": "#7570b3", "RNA": "#e7298a"},
    )
    axes[0].set_title("Signal Dynamic Range by Pore Chemistry")
    axes[0].set_ylabel("SD of k-mer current mean")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    dna9 = position_df[position_df["chemistry"] == "DNA R10.4.1"]
    pivot = dna9.pivot(index="base", columns="position", values="mean_current_mean").loc[list("ACGT")]
    sns.heatmap(pivot, cmap="vlag", center=pivot.values.mean(), annot=True, fmt=".2f", ax=axes[1], cbar_kws={"label": "Mean current"})
    axes[1].set_title("Position-Specific Base Effects in DNA R10.4.1 9-mers")
    axes[1].set_xlabel("Position in k-mer")
    axes[1].set_ylabel("Base")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "pore_model_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report_inputs(summary_df: pd.DataFrame, perf_ratios: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    key_findings = {
        "best_auprc_model": metrics_df.sort_values("auprc", ascending=False).iloc[0]["model"],
        "best_auroc_model": metrics_df.sort_values("auroc", ascending=False).iloc[0]["model"],
        "median_runtime_ratio": perf_ratios.loc[perf_ratios["Tool"] != "Uncalled4", "speedup_vs_uncalled4"].median(),
        "median_filesize_ratio": perf_ratios.loc[perf_ratios["Tool"] != "Uncalled4", "size_ratio_vs_uncalled4"].median(),
        "widest_current_dynamic_range": summary_df.sort_values("current_mean_std", ascending=False).iloc[0]["chemistry"],
    }
    with open(OUTPUT_DIR / "key_findings.json", "w", encoding="utf-8") as f:
        json.dump(key_findings, f, indent=2)


def main() -> None:
    ensure_dirs()
    summary_df, position_df = summarize_pore_models()
    perf, perf_ratios = analyze_performance()
    _, curves = analyze_m6a()
    metrics_df = pd.read_csv(OUTPUT_DIR / "m6a_model_metrics.csv")

    plot_performance(perf)
    plot_m6a(curves, OUTPUT_DIR / "m6a_model_metrics.csv")
    plot_pore_effects(position_df, summary_df)
    write_report_inputs(summary_df, perf_ratios, metrics_df)


if __name__ == "__main__":
    main()
