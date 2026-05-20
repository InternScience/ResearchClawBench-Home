#!/usr/bin/env python3
"""
Uncalled4 Nanopore Analysis Script
- Loads performance and m6A prediction data
- Generates benchmark bar plots (time + file size)
- Generates precision-recall curves comparing Uncalled4 vs Nanopolish
- Saves figures to report/images/ and stats to outputs/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

def load_data():
    perf = pd.read_csv(f"{DATA_DIR}/performance_summary.csv")
    m6a_unc = pd.read_csv(f"{DATA_DIR}/m6a_predictions_uncalled4.csv")
    m6a_nano = pd.read_csv(f"{DATA_DIR}/m6a_predictions_nanopolish.csv")
    labels = pd.read_csv(f"{DATA_DIR}/m6a_labels.csv")
    return perf, m6a_unc, m6a_nano, labels

def plot_performance(perf):
    # Clean data for plotting
    perf_clean = perf.dropna(subset=["Time_min", "FileSize_MB"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time plot
    sns.barplot(data=perf_clean, x="Chemistry", y="Time_min", hue="Tool", ax=axes[0])
    axes[0].set_title("Alignment Time by Chemistry and Tool")
    axes[0].set_ylabel("Time (minutes)")
    axes[0].set_xlabel("Sequencing Chemistry")
    axes[0].legend(title="Tool")

    # File size plot
    sns.barplot(data=perf_clean, x="Chemistry", y="FileSize_MB", hue="Tool", ax=axes[1])
    axes[1].set_title("Output File Size by Chemistry and Tool")
    axes[1].set_ylabel("File Size (MB)")
    axes[1].set_xlabel("Sequencing Chemistry")
    axes[1].legend(title="Tool")

    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/performance_benchmarks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved performance_benchmarks.png")

    # Save summary stats
    summary = perf_clean.groupby(["Chemistry", "Tool"])[["Time_min", "FileSize_MB"]].mean().reset_index()
    summary.to_csv(f"{OUTPUT_DIR}/performance_summary_stats.csv", index=False)
    print("Saved performance_summary_stats.csv")

def plot_pr_curves(m6a_unc, m6a_nano, labels):
    y_true = labels["label"].values
    scores_unc = m6a_unc["probability"].values
    scores_nano = m6a_nano["probability"].values

    # Uncalled4
    prec_u, rec_u, _ = precision_recall_curve(y_true, scores_unc)
    ap_u = average_precision_score(y_true, scores_unc)

    # Nanopolish
    prec_n, rec_n, _ = precision_recall_curve(y_true, scores_nano)
    ap_n = average_precision_score(y_true, scores_nano)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_u, prec_u, label=f"Uncalled4 (AP={ap_u:.3f})", linewidth=2)
    ax.plot(rec_n, prec_n, label=f"Nanopolish (AP={ap_n:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: m6A Detection")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{FIGURE_DIR}/pr_curves_m6a.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved pr_curves_m6a.png")

    # Save PR data
    min_len = min(len(rec_u), len(rec_n))
    pr_df = pd.DataFrame({
        "recall_uncalled4": rec_u[:min_len],
        "precision_uncalled4": prec_u[:min_len],
        "recall_nanopolish": rec_n[:min_len],
        "precision_nanopolish": prec_n[:min_len]
    })
    pr_df.to_csv(f"{OUTPUT_DIR}/pr_curve_data.csv", index=False)
    print("Saved pr_curve_data.csv")

    # Save AP scores
    ap_df = pd.DataFrame({"tool": ["Uncalled4", "Nanopolish"], "average_precision": [ap_u, ap_n]})
    ap_df.to_csv(f"{OUTPUT_DIR}/average_precision_scores.csv", index=False)
    print("Saved average_precision_scores.csv")

def main():
    print("Loading data...")
    perf, m6a_unc, m6a_nano, labels = load_data()
    print(f"Performance rows: {len(perf)}")
    print(f"m6A sites: {len(labels)}")

    print("Generating performance benchmarks...")
    plot_performance(perf)

    print("Generating PR curves...")
    plot_pr_curves(m6a_unc, m6a_nano, labels)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
