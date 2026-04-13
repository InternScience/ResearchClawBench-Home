from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "glambie"
INPUT_DIR = DATA_DIR / "input"
RESULT_CAL_DIR = DATA_DIR / "results" / "calendar_years"
RESULT_HYDRO_DIR = DATA_DIR / "results" / "hydrological_years"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

START_YEAR = 2000
END_YEAR = 2023


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def parse_input_catalog() -> pd.DataFrame:
    rows = []
    for path in sorted(INPUT_DIR.glob("*/*.csv")):
        region_folder = path.parent.name
        region_name = region_folder.split("_", 1)[1]
        stem = path.stem
        prefix = f"{region_name}_"
        remainder = stem[len(prefix):] if stem.startswith(prefix) else stem
        data_group, _, dataset_suffix = remainder.partition("_")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "region_folder": region_folder,
                "region": region_name,
                "data_group": data_group,
                "dataset_id": dataset_suffix or stem,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "input_catalog.csv", index=False)
    return df


def load_calendar_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULT_CAL_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["start_dates"].round().astype(int)
    out = out[(out["year"] >= START_YEAR) & (out["year"] <= END_YEAR)].copy()
    return out


def load_hydrological_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULT_HYDRO_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["hydro_year"] = np.floor(out["end_dates"]).astype(int)
    return out


def annualize_input_series(input_catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, meta in input_catalog.iterrows():
        df = pd.read_csv(ROOT / meta["path"])
        if df.empty:
            continue
        df = df.copy()
        df["duration"] = df["end_dates"] - df["start_dates"]
        df = df[df["duration"] > 0].copy()
        if df.empty:
            continue
        for year in range(START_YEAR, END_YEAR + 1):
            overlap_start = np.maximum(df["start_dates"].to_numpy(), year)
            overlap_end = np.minimum(df["end_dates"].to_numpy(), year + 1)
            overlap = np.clip(overlap_end - overlap_start, 0, None)
            mask = overlap > 0
            if not mask.any():
                continue
            frac = overlap[mask] / df.loc[mask, "duration"].to_numpy()
            changes = (df.loc[mask, "changes"].to_numpy() * frac).sum()
            errors = np.sqrt(((df.loc[mask, "errors"].to_numpy() * frac) ** 2).sum())
            rows.append(
                {
                    "region": meta["region"],
                    "region_folder": meta["region_folder"],
                    "data_group": meta["data_group"],
                    "dataset_id": meta["dataset_id"],
                    "year": year,
                    "unit": df.loc[mask, "unit"].iloc[0],
                    "annual_change": changes,
                    "annual_error": errors,
                    "n_segments": int(mask.sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "annualized_input_series.csv", index=False)
    return out


def summarize_global(calendar_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    global_df = calendar_df[calendar_df["region"] == "global"].copy()
    global_df["cumulative_gt"] = global_df["combined_gt"].cumsum()
    global_df["cumulative_gt_error"] = np.sqrt((global_df["combined_gt_errors"] ** 2).cumsum())
    global_df["cumulative_mwe"] = global_df["combined_mwe"].cumsum()
    global_df.to_csv(OUTPUT_DIR / "global_annual_series.csv", index=False)

    period = global_df[(global_df["year"] >= START_YEAR) & (global_df["year"] <= END_YEAR)]
    stats = {
        "period": f"{START_YEAR}-{END_YEAR}",
        "years": int(len(period)),
        "mean_annual_gt": float(period["combined_gt"].mean()),
        "mean_annual_mwe": float(period["combined_mwe"].mean()),
        "total_gt": float(period["combined_gt"].sum()),
        "total_mwe": float(period["combined_mwe"].sum()),
        "mean_annual_uncertainty_gt": float(period["combined_gt_errors"].mean()),
        "max_loss_year": int(period.loc[period["combined_gt"].idxmin(), "year"]),
        "max_loss_gt": float(period["combined_gt"].min()),
        "least_negative_year": int(period.loc[period["combined_gt"].idxmax(), "year"]),
        "least_negative_gt": float(period["combined_gt"].max()),
    }
    with open(OUTPUT_DIR / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return global_df, stats


def summarize_regions(calendar_df: pd.DataFrame) -> pd.DataFrame:
    regions = calendar_df[calendar_df["region"] != "global"].copy()
    summary = (
        regions.groupby("region", as_index=False)
        .agg(
            area_2000_km2=("glacier_area", "first"),
            area_2023_km2=("glacier_area", "last"),
            total_gt_2000_2023=("combined_gt", "sum"),
            mean_gt_per_year=("combined_gt", "mean"),
            total_mwe_2000_2023=("combined_mwe", "sum"),
            mean_mwe_per_year=("combined_mwe", "mean"),
            mean_annual_uncertainty_gt=("combined_gt_errors", "mean"),
        )
    )
    summary["area_change_pct"] = 100 * (summary["area_2023_km2"] - summary["area_2000_km2"]) / summary["area_2000_km2"]
    summary = summary.sort_values("total_gt_2000_2023")
    summary.to_csv(OUTPUT_DIR / "regional_summary.csv", index=False)
    return summary


def method_spread_summary(hydro_df: pd.DataFrame) -> pd.DataFrame:
    method_cols = {
        "altimetry": "altimetry_gt",
        "gravimetry": "gravimetry_gt",
        "demdiff_and_glaciological": "demdiff_and_glaciological_gt",
    }
    rows = []
    hydro = hydro_df[(hydro_df["hydro_year"] >= START_YEAR) & (hydro_df["hydro_year"] <= END_YEAR)].copy()
    hydro = hydro[hydro["region"] != "global"].copy()
    for (region, year), grp in hydro.groupby(["region", "hydro_year"]):
        record = {"region": region, "year": int(year)}
        vals = []
        for label, col in method_cols.items():
            val = grp.iloc[0][col]
            record[label] = val
            if pd.notna(val):
                vals.append(val)
        record["method_count"] = len(vals)
        record["spread_gt"] = float(np.max(vals) - np.min(vals)) if len(vals) >= 2 else np.nan
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "method_spread_by_region_year.csv", index=False)
    summary = (
        out.groupby("region", as_index=False)
        .agg(
            years_with_multi_method=("method_count", lambda s: int((s >= 2).sum())),
            mean_spread_gt=("spread_gt", "mean"),
            max_spread_gt=("spread_gt", "max"),
        )
        .sort_values("mean_spread_gt", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "method_spread_summary.csv", index=False)
    return summary


def input_coverage_summary(annualized_input: pd.DataFrame) -> pd.DataFrame:
    summary = (
        annualized_input.groupby(["region", "data_group"], as_index=False)
        .agg(
            datasets=("dataset_id", "nunique"),
            covered_years=("year", "nunique"),
            unit=("unit", lambda s: s.mode().iat[0]),
        )
        .sort_values(["region", "data_group"])
    )
    summary.to_csv(OUTPUT_DIR / "input_coverage_summary.csv", index=False)
    return summary


def plot_global_series(global_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    x = global_df["year"]
    y = global_df["combined_gt"]
    err = global_df["combined_gt_errors"]
    ax.fill_between(x, y - err, y + err, color="#9ecae1", alpha=0.45, linewidth=0)
    ax.plot(x, y, color="#08519c", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title("Global annual glacier mass change")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass change (Gt yr$^{-1}$)")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "global_annual_mass_change.png")
    plt.close(fig)


def plot_regional_totals(region_summary: pd.DataFrame) -> None:
    top = region_summary.nsmallest(10, "total_gt_2000_2023").copy()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    sns.barplot(data=top, y="region", x="total_gt_2000_2023", color="#3182bd", ax=ax)
    ax.set_title("Largest regional cumulative mass losses, 2000-2023")
    ax.set_xlabel("Cumulative mass change (Gt)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "regional_cumulative_mass_loss_top10.png")
    plt.close(fig)


def plot_specific_vs_total(region_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    sns.scatterplot(
        data=region_summary,
        x="total_mwe_2000_2023",
        y="total_gt_2000_2023",
        size="area_2000_km2",
        hue="area_2000_km2",
        palette="Blues",
        ax=ax,
        legend=False,
    )
    for _, row in region_summary.iterrows():
        ax.text(row["total_mwe_2000_2023"], row["total_gt_2000_2023"], row["region"], fontsize=7)
    ax.set_title("Specific versus total glacier mass loss by region")
    ax.set_xlabel("Cumulative specific mass change (m w.e.)")
    ax.set_ylabel("Cumulative mass change (Gt)")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "regional_specific_vs_total_loss.png")
    plt.close(fig)


def plot_method_spread(method_summary: pd.DataFrame) -> None:
    top = method_summary.head(10).copy()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    sns.barplot(data=top, y="region", x="mean_spread_gt", color="#e6550d", ax=ax)
    ax.set_title("Mean cross-method spread in hydrological-year estimates")
    ax.set_xlabel("Mean spread across available methods (Gt yr$^{-1}$)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "method_spread_top10.png")
    plt.close(fig)


def plot_method_coverage(input_coverage: pd.DataFrame) -> None:
    pivot = (
        input_coverage.pivot(index="region", columns="data_group", values="datasets")
        .fillna(0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Dataset count"}, ax=ax)
    ax.set_title("Input dataset coverage by region and method")
    ax.set_xlabel("Method group")
    ax.set_ylabel("Region")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "input_dataset_coverage_heatmap.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    input_catalog = parse_input_catalog()
    calendar_df = load_calendar_results()
    hydro_df = load_hydrological_results()
    annualized_input = annualize_input_series(input_catalog)
    global_df, _ = summarize_global(calendar_df)
    region_summary = summarize_regions(calendar_df)
    method_summary = method_spread_summary(hydro_df)
    input_coverage = input_coverage_summary(annualized_input)

    plot_global_series(global_df)
    plot_regional_totals(region_summary)
    plot_specific_vs_total(region_summary)
    plot_method_spread(method_summary)
    plot_method_coverage(input_coverage)


if __name__ == "__main__":
    main()
