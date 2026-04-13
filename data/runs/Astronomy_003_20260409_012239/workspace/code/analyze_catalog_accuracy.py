from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fig6 = pd.read_csv(DATA_DIR / "fig6_data.csv")
    fig7 = pd.read_csv(DATA_DIR / "fig7_data.csv")
    fig8 = pd.read_csv(DATA_DIR / "fig8_data.csv")
    return fig6, fig7, fig8


def summarize_series(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(series.shape[0]),
        "min": float(series.min()),
        "q10": float(q.loc[0.1]),
        "q25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "q90": float(q.loc[0.9]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
    }


def make_fig6_panels(fig6: pd.DataFrame) -> dict[str, float]:
    values = fig6["waveform_difference"]
    thresholds = [1e-3, 3e-3, 1e-2, 5e-2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bins = np.logspace(np.log10(values.min() * 0.8), np.log10(values.max() * 1.2), 45)
    axes[0].hist(values, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
    axes[0].axvline(values.median(), color="#d62728", linestyle="--", linewidth=2, label="Median")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Waveform difference")
    axes[0].set_ylabel("Simulation count")
    axes[0].set_title("Resolution-error distribution")
    axes[0].legend(frameon=False)

    sorted_vals = np.sort(values.to_numpy())
    survival = 1.0 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, survival, color="#ff7f0e", linewidth=2)
    for thr in thresholds:
        axes[1].axvline(thr, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Waveform difference threshold")
    axes[1].set_ylabel("Fraction above threshold")
    axes[1].set_title("Tail risk of large mismatches")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "resolution_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    coverage = {f"share_below_{thr:.0e}": float((values < thr).mean()) for thr in thresholds}
    return coverage


def make_fig7_plots(fig7: pd.DataFrame) -> pd.DataFrame:
    long_df = fig7.melt(var_name="mode", value_name="difference")
    long_df["ell"] = long_df["mode"].str.replace("ell", "", regex=False).astype(int)

    stats = (
        long_df.groupby("ell")["difference"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            q90=lambda s: s.quantile(0.90),
            mean="mean",
        )
        .reset_index()
    )
    log_median = np.log10(stats["median"])
    slope, intercept = np.polyfit(stats["ell"], log_median, 1)
    stats["fit_log10_median"] = intercept + slope * stats["ell"]
    stats["fit_median"] = 10 ** stats["fit_log10_median"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    positions = np.arange(1, len(fig7.columns) + 1)
    axes[0].boxplot(
        [fig7[c].to_numpy() for c in fig7.columns],
        positions=positions,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#9ecae1", alpha=0.9),
        medianprops=dict(color="#d62728", linewidth=2),
    )
    axes[0].set_xticks(positions, [c.replace("ell", "l=") for c in fig7.columns])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mode-wise waveform difference")
    axes[0].set_title("Accuracy degrades toward higher multipoles")

    axes[1].plot(stats["ell"], stats["median"], marker="o", linewidth=2, color="#2ca02c", label="Median")
    axes[1].fill_between(stats["ell"], stats["q25"], stats["q75"], color="#2ca02c", alpha=0.15, label="IQR")
    axes[1].plot(stats["ell"], stats["fit_median"], linestyle="--", color="#9467bd", label="Log-linear fit")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Spherical-harmonic index l")
    axes[1].set_ylabel("Waveform difference")
    axes[1].set_title("Median error scaling by mode")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "mode_error_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    stats["median_growth_l2_to_l8"] = float(
        stats.loc[stats["ell"] == 8, "median"].iloc[0] / stats.loc[stats["ell"] == 2, "median"].iloc[0]
    )
    stats["log10_median_slope_per_l"] = float(slope)
    return stats


def make_fig8_plots(fig8: pd.DataFrame) -> dict[str, float]:
    ratio = fig8["N2vsN4"] / fig8["N2vsN3"]
    worse_share = float((fig8["N2vsN4"] > fig8["N2vsN3"]).mean())
    median_ratio = float(ratio.median())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bins = np.logspace(
        np.log10(min(fig8["N2vsN3"].min(), fig8["N2vsN4"].min()) * 0.8),
        np.log10(max(fig8["N2vsN3"].max(), fig8["N2vsN4"].max()) * 1.2),
        40,
    )
    axes[0].hist(fig8["N2vsN3"], bins=bins, alpha=0.7, label="N=2 vs N=3", color="#1f77b4")
    axes[0].hist(fig8["N2vsN4"], bins=bins, alpha=0.6, label="N=2 vs N=4", color="#ff7f0e")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Extrapolation-order difference")
    axes[0].set_ylabel("Case count")
    axes[0].set_title("Distribution of extrapolation discrepancies")
    axes[0].legend(frameon=False)

    axes[1].scatter(fig8["N2vsN3"], fig8["N2vsN4"], s=12, alpha=0.35, color="#2ca02c")
    diagonal = np.logspace(
        np.log10(min(fig8.min().min(), 1e-7)),
        np.log10(max(fig8.max().max(), 1e-2)),
        200,
    )
    axes[1].plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N=2 vs N=3")
    axes[1].set_ylabel("N=2 vs N=4")
    axes[1].set_title("Most cases worsen at higher-order comparison")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "extrapolation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "share_N2vsN4_greater_than_N2vsN3": worse_share,
        "median_ratio_N2vsN4_over_N2vsN3": median_ratio,
        "pearson_correlation": float(fig8.corr().iloc[0, 1]),
    }


def build_quality_index(fig6: pd.DataFrame, fig7: pd.DataFrame, fig8: pd.DataFrame) -> pd.DataFrame:
    n = min(len(fig6), len(fig7), len(fig8))
    joined = pd.DataFrame(
        {
            "resolution": fig6["waveform_difference"].iloc[:n].to_numpy(),
            "mode_median": fig7.iloc[:n].median(axis=1).to_numpy(),
            "mode_max": fig7.iloc[:n].max(axis=1).to_numpy(),
            "extrap_n23": fig8["N2vsN3"].iloc[:n].to_numpy(),
            "extrap_n24": fig8["N2vsN4"].iloc[:n].to_numpy(),
        }
    )
    log_cols = np.log10(joined)
    normalized = (log_cols - log_cols.median()) / log_cols.std(ddof=1)
    joined["quality_index"] = normalized.mean(axis=1)
    joined["quality_tier"] = pd.qcut(
        joined["quality_index"],
        q=4,
        labels=["Tier A", "Tier B", "Tier C", "Tier D"],
    )
    joined["high_mode_penalty"] = np.log10(joined["mode_max"] / joined["mode_median"])

    tier_summary = (
        joined.groupby("quality_tier", observed=False)
        .agg(
            count=("quality_index", "size"),
            median_resolution=("resolution", "median"),
            median_mode_max=("mode_max", "median"),
            median_extrap_n24=("extrap_n24", "median"),
        )
        .reset_index()
    )
    tier_summary.to_csv(OUTPUT_DIR / "quality_tier_summary.csv", index=False)
    joined.to_csv(OUTPUT_DIR / "catalog_quality_index.csv", index=False)
    return joined


def main() -> None:
    ensure_dirs()
    fig6, fig7, fig8 = load_data()

    summary = {
        "fig6_summary": summarize_series(fig6["waveform_difference"]),
        "fig7_summary_by_mode": {col: summarize_series(fig7[col]) for col in fig7.columns},
        "fig8_summary_by_column": {col: summarize_series(fig8[col]) for col in fig8.columns},
    }

    summary["fig6_coverage"] = make_fig6_panels(fig6)
    mode_stats = make_fig7_plots(fig7)
    summary["fig7_mode_stats"] = mode_stats.to_dict(orient="records")
    summary["fig8_comparison"] = make_fig8_plots(fig8)

    quality = build_quality_index(fig6, fig7, fig8)
    summary["quality_index_summary"] = {
        "count": int(len(quality)),
        "median": float(quality["quality_index"].median()),
        "q25": float(quality["quality_index"].quantile(0.25)),
        "q75": float(quality["quality_index"].quantile(0.75)),
        "share_tier_a": float((quality["quality_tier"] == "Tier A").mean()),
        "share_tier_d": float((quality["quality_tier"] == "Tier D").mean()),
        "median_high_mode_penalty": float(quality["high_mode_penalty"].median()),
    }

    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    mode_stats.to_csv(OUTPUT_DIR / "mode_error_stats.csv", index=False)


if __name__ == "__main__":
    main()
