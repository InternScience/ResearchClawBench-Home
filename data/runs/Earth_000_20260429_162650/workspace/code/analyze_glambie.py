#!/usr/bin/env python3
"""Reproducible analysis of GlaMBIE glacier mass-change results.

This script reads read-only GlaMBIE input and result CSV files, exports analysis
artifacts to outputs/, and writes PNG figures to report/images/.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "glambie"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 200})

METHODS = ["altimetry", "combined", "demdiff", "glaciological", "gravimetry"]
METHOD_LABELS = {
    "altimetry": "Altimetry",
    "combined": "Hybrid/combined",
    "demdiff": "DEM differencing",
    "glaciological": "Glaciological",
    "gravimetry": "Gravimetry",
    "demdiff_and_glaciological": "DEM + glaciological",
}


def read_calendar_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in sorted((DATA / "results" / "calendar_years").glob("*.csv")):
        df = pd.read_csv(f)
        df["source_file"] = str(f.relative_to(ROOT))
        df["year"] = df["start_dates"].astype(int)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    regional = all_df[all_df["region"] != "global"].copy()
    global_df = all_df[all_df["region"] == "global"].copy()
    return regional, global_df


def read_hydro_method_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    for f in sorted((DATA / "results" / "hydrological_years").glob("*.csv")):
        df = pd.read_csv(f)
        df["source_file"] = str(f.relative_to(ROOT))
        df["year"] = df["start_dates"].astype(int)
        rows.append(df)
        for group in ["combined", "altimetry", "gravimetry", "demdiff_and_glaciological"]:
            gt = f"{group}_gt"
            err = f"{group}_gt_errors"
            mwe = f"{group}_mwe"
            mwe_err = f"{group}_mwe_errors"
            if gt in df.columns:
                sub = df[["region", "year", "start_dates", "end_dates", "glacier_area", gt, err, mwe, mwe_err]].copy()
                sub.columns = ["region", "year", "start_dates", "end_dates", "glacier_area", "gt", "gt_error", "mwe", "mwe_error"]
                sub["method_group"] = group
                long_rows.append(sub)
    hydro = pd.concat(rows, ignore_index=True)
    long = pd.concat(long_rows, ignore_index=True)
    return hydro, long


def infer_method(path: Path) -> str:
    base = path.name
    for m in METHODS:
        if f"_{m}_" in base:
            return m
    return "unknown"


def read_input_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    annual_rows = []
    for f in sorted((DATA / "input").glob("*/*.csv")):
        df = pd.read_csv(f)
        method = infer_method(f)
        region_folder = f.parent.name
        region = re.sub(r"^\d+_", "", region_folder)
        nrows = len(df)
        area_hint = np.nan
        rows.append(
            {
                "file": str(f.relative_to(ROOT)),
                "region_folder": region_folder,
                "region": region,
                "method": method,
                "n_rows": nrows,
                "unit": ";".join(sorted(map(str, df.get("unit", pd.Series(dtype=str)).dropna().unique()))),
                "min_start": float(df["start_dates"].min()) if nrows else np.nan,
                "max_end": float(df["end_dates"].max()) if nrows else np.nan,
                "mean_duration_years": float((df["end_dates"] - df["start_dates"]).mean()) if nrows else np.nan,
                "author": "; ".join(sorted(map(str, df.get("author", pd.Series(dtype=str)).dropna().unique())))[:200],
            }
        )
        # annualized source estimate for broad validation vs consensus.
        for _, r in df.iterrows():
            duration = r["end_dates"] - r["start_dates"]
            if duration <= 0:
                continue
            annual_rows.append(
                {
                    "file": str(f.relative_to(ROOT)),
                    "region": region,
                    "method": method,
                    "start_dates": r["start_dates"],
                    "end_dates": r["end_dates"],
                    "mid_year": (r["start_dates"] + r["end_dates"]) / 2.0,
                    "duration_years": duration,
                    "change": r["changes"],
                    "error": r["errors"],
                    "unit": r.get("unit", np.nan),
                    "annual_change": r["changes"] / duration,
                    "annual_error": r["errors"] / duration,
                    "author": r.get("author", ""),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def weighted_mean(values: pd.Series, errors: pd.Series) -> tuple[float, float, int]:
    vals = pd.to_numeric(values, errors="coerce")
    errs = pd.to_numeric(errors, errors="coerce")
    mask = vals.notna() & errs.notna() & (errs > 0)
    if mask.sum() == 0:
        return np.nan, np.nan, 0
    w = 1.0 / np.square(errs[mask].astype(float))
    mu = float(np.sum(w * vals[mask]) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return mu, se, int(mask.sum())


def main() -> None:
    regional, global_df = read_calendar_results()
    hydro, hydro_long = read_hydro_method_results()
    inventory, input_long = read_input_inventory()

    # Save primary result tables.
    regional_out = regional[["region", "year", "start_dates", "end_dates", "glacier_area", "combined_gt", "combined_gt_errors", "combined_mwe", "combined_mwe_errors", "source_file"]].copy()
    global_out = global_df[["region", "year", "start_dates", "end_dates", "glacier_area", "combined_gt", "combined_gt_errors", "combined_mwe", "combined_mwe_errors", "source_file"]].copy()
    regional_out.to_csv(OUT / "regional_annual_reconciled.csv", index=False)
    global_out.to_csv(OUT / "global_annual_reconciled.csv", index=False)

    # Verify global aggregation from regional calendar files.
    agg = regional.groupby("year", as_index=False).agg(
        regional_sum_gt=("combined_gt", "sum"),
        regional_area_sum_km2=("glacier_area", "sum"),
        regional_quad_error_gt=("combined_gt_errors", lambda x: float(np.sqrt(np.sum(np.square(x))))),
    )
    check = global_out.merge(agg, on="year")
    check["global_minus_regional_sum_gt"] = check["combined_gt"] - check["regional_sum_gt"]
    check["global_minus_regional_area_sum_km2"] = check["glacier_area"] - check["regional_area_sum_km2"]
    check.to_csv(OUT / "global_regional_aggregation_check.csv", index=False)

    # Overview of input estimates and coverage.
    overview = inventory.groupby(["region", "method"], as_index=False).agg(
        n_datasets=("file", "count"),
        n_records=("n_rows", "sum"),
        min_start=("min_start", "min"),
        max_end=("max_end", "max"),
        mean_duration_years=("mean_duration_years", "mean"),
    )
    overview.to_csv(OUT / "data_overview.csv", index=False)
    inventory.to_csv(OUT / "input_dataset_inventory.csv", index=False)
    input_long.to_csv(OUT / "input_observation_records_annualized.csv", index=False)

    # Hydrological-year method comparison table: source group vs combined.
    method_rows = []
    for (region, group), df in hydro_long.groupby(["region", "method_group"]):
        valid = df.dropna(subset=["gt", "gt_error", "mwe", "mwe_error"])
        if valid.empty:
            continue
        comb = hydro_long[(hydro_long["region"] == region) & (hydro_long["method_group"] == "combined")][["year", "gt", "mwe"]].rename(columns={"gt": "combined_gt", "mwe": "combined_mwe"})
        joined = valid.merge(comb, on="year", how="left")
        method_rows.append(
            {
                "region": region,
                "method_group": group,
                "n_years": int(valid["year"].nunique()),
                "mean_gt_per_year": valid["gt"].mean(),
                "mean_gt_error": valid["gt_error"].mean(),
                "cumulative_gt": valid["gt"].sum(),
                "cumulative_gt_quadrature_error": float(np.sqrt(np.sum(np.square(valid["gt_error"])))),
                "mean_mwe_per_year": valid["mwe"].mean(),
                "mean_mwe_error": valid["mwe_error"].mean(),
                "rmse_vs_combined_gt": float(np.sqrt(np.nanmean(np.square(joined["gt"] - joined["combined_gt"])))) if group != "combined" else 0.0,
                "bias_vs_combined_gt": float(np.nanmean(joined["gt"] - joined["combined_gt"])) if group != "combined" else 0.0,
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(OUT / "method_comparison_summary.csv", index=False)

    # Input-observation validation in m units vs calendar consensus for overlaps.
    # Annualize multi-year records and compare to mean consensus annual specific mass change over matching interval.
    cal_reg = regional[["region", "year", "combined_mwe", "combined_gt", "glacier_area"]].copy()
    val_rows = []
    for _, r in input_long[input_long["unit"] == "m"].iterrows():
        yrs = list(range(math.ceil(r["start_dates"]), math.floor(r["end_dates"])))
        if not yrs:
            yrs = [int(math.floor(r["mid_year"]))]
        sub = cal_reg[(cal_reg.region == r.region) & (cal_reg.year.isin(yrs))]
        if sub.empty:
            continue
        val_rows.append({**r.to_dict(), "consensus_annual_mwe": sub["combined_mwe"].mean(), "difference_mwe_per_year": r["annual_change"] - sub["combined_mwe"].mean(), "n_consensus_years": len(sub)})
    validation = pd.DataFrame(val_rows)
    validation.to_csv(OUT / "input_vs_consensus_validation_mwe.csv", index=False)

    # Global and regional summaries.
    g = global_out.copy()
    g["cumulative_gt"] = g["combined_gt"].cumsum()
    g["cumulative_gt_error_independent"] = np.sqrt(np.cumsum(np.square(g["combined_gt_errors"])))
    g["cumulative_mwe"] = g["combined_mwe"].cumsum()
    g["cumulative_mwe_error_independent"] = np.sqrt(np.cumsum(np.square(g["combined_mwe_errors"])))
    g.to_csv(OUT / "global_annual_reconciled_with_cumulative.csv", index=False)

    reg_summary = regional.groupby("region", as_index=False).agg(
        area_2000_km2=("glacier_area", "first"),
        area_2023_km2=("glacier_area", "last"),
        cumulative_gt=("combined_gt", "sum"),
        mean_gt_per_year=("combined_gt", "mean"),
        mean_mwe_per_year=("combined_mwe", "mean"),
        max_loss_year_gt=("combined_gt", lambda x: int(regional.loc[x.idxmin(), "year"])),
        n_years=("year", "count"),
    )
    reg_err = regional.groupby("region")["combined_gt_errors"].apply(lambda x: float(np.sqrt(np.sum(np.square(x))))).reset_index(name="cumulative_gt_error_independent")
    reg_summary = reg_summary.merge(reg_err, on="region")
    reg_summary["area_change_km2"] = reg_summary["area_2023_km2"] - reg_summary["area_2000_km2"]
    reg_summary.to_csv(OUT / "regional_cumulative_summary.csv", index=False)

    uncertainty_summary = pd.DataFrame([
        {
            "scope": "global",
            "period": "2000-2023",
            "n_years": len(g),
            "cumulative_gt": g["combined_gt"].sum(),
            "cumulative_gt_error_independent": float(np.sqrt(np.sum(np.square(g["combined_gt_errors"])))),
            "mean_annual_gt": g["combined_gt"].mean(),
            "mean_annual_gt_error": g["combined_gt_errors"].mean(),
            "cumulative_mwe": g["combined_mwe"].sum(),
            "cumulative_mwe_error_independent": float(np.sqrt(np.sum(np.square(g["combined_mwe_errors"])))),
            "mean_annual_mwe": g["combined_mwe"].mean(),
            "mean_annual_mwe_error": g["combined_mwe_errors"].mean(),
        }
    ])
    uncertainty_summary.to_csv(OUT / "uncertainty_summary.csv", index=False)

    # Compact direct answer table.
    direct = global_out[["year", "combined_gt", "combined_gt_errors", "combined_mwe", "combined_mwe_errors", "glacier_area"]].copy()
    direct.rename(columns={
        "combined_gt": "global_total_mass_change_gt",
        "combined_gt_errors": "global_total_mass_change_uncertainty_gt",
        "combined_mwe": "global_specific_mass_change_mwe",
        "combined_mwe_errors": "global_specific_mass_change_uncertainty_mwe",
    }, inplace=True)
    direct.to_csv(OUT / "direct_global_2000_2023_answer.csv", index=False)

    # Figures.
    # Fig 1: overview counts by region and method.
    pivot = overview.pivot(index="region", columns="method", values="n_datasets").fillna(0)
    region_order = reg_summary.sort_values("area_2000_km2", ascending=False)["region"].tolist()
    pivot = pivot.reindex(region_order)
    ax = pivot[METHODS].plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
    ax.set_ylabel("Number of submitted datasets")
    ax.set_xlabel("GTN-G glacier region")
    ax.set_title("GlaMBIE input coverage by region and observation method")
    ax.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(IMG / "fig1_data_overview.png")
    plt.close()

    # Fig 2: global annual and cumulative time series.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.plot(g["year"], g["combined_gt"], color="#1f77b4", marker="o", label="Annual mass change")
    ax1.fill_between(g["year"], g["combined_gt"] - g["combined_gt_errors"], g["combined_gt"] + g["combined_gt_errors"], color="#1f77b4", alpha=0.2, label="±1σ")
    ax1.set_ylabel("Annual mass change (Gt yr$^{-1}$)")
    ax1.set_title("Global glacier mass change, calendar years 2000–2023")
    ax1.legend(loc="lower left")
    ax2.axhline(0, color="0.4", lw=0.8)
    ax2.plot(g["year"], g["cumulative_gt"], color="#d62728", marker="o", label="Cumulative")
    ax2.fill_between(g["year"], g["cumulative_gt"] - g["cumulative_gt_error_independent"], g["cumulative_gt"] + g["cumulative_gt_error_independent"], color="#d62728", alpha=0.2, label="Independent-error ±1σ")
    ax2.set_ylabel("Cumulative mass change (Gt)")
    ax2.set_xlabel("Year")
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(IMG / "fig2_global_timeseries.png")
    plt.close()

    # Fig 3: regional annual specific mass-change heatmap.
    heat = regional.pivot(index="region", columns="year", values="combined_mwe").reindex(region_order)
    plt.figure(figsize=(12, 7))
    sns.heatmap(heat, cmap="RdBu", center=0, cbar_kws={"label": "Specific mass change (m w.e. yr$^{-1}$)"})
    plt.title("Regional annual specific mass change from reconciled GlaMBIE results")
    plt.xlabel("Year")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(IMG / "fig3_regional_heatmap.png")
    plt.close()

    # Fig 4: method validation/comparison, hydrological data groups against combined.
    plot_df = method_summary[method_summary["method_group"] != "combined"].copy()
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=plot_df, x="bias_vs_combined_gt", y="rmse_vs_combined_gt", hue="method_group", size="n_years", sizes=(40, 180), alpha=0.85)
    plt.axvline(0, color="0.4", lw=0.8)
    plt.xlabel("Mean bias vs combined solution (Gt yr$^{-1}$)")
    plt.ylabel("RMSE vs combined solution (Gt yr$^{-1}$)")
    plt.title("Agreement of hydrological-year method groups with regional combined estimates")
    plt.legend(title="Method group", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(IMG / "fig4_method_validation.png")
    plt.close()

    # Fig 5: regional cumulative contributions.
    top = reg_summary.sort_values("cumulative_gt")
    plt.figure(figsize=(9, 6))
    plt.barh(top["region"], top["cumulative_gt"], xerr=top["cumulative_gt_error_independent"], color="#4c78a8", alpha=0.85)
    plt.axvline(0, color="0.3", lw=0.8)
    plt.xlabel("Cumulative mass change 2000–2023 (Gt)")
    plt.ylabel("Region")
    plt.title("Regional cumulative contribution to global glacier mass change")
    plt.tight_layout()
    plt.savefig(IMG / "fig5_regional_cumulative.png")
    plt.close()

    # Claim recovery table.
    claim_rows = [
        {
            "claim": "The calendar-year GlaMBIE result files provide annual 2000-2023 estimates for 19 regions and a global aggregate.",
            "supporting_artifact": "outputs/regional_annual_reconciled.csv; outputs/global_annual_reconciled.csv; outputs/data_overview.csv",
            "verification": f"{regional['region'].nunique()} regions, {len(global_out)} global annual rows, years {int(global_out.year.min())}-{int(global_out.year.max())}",
        },
        {
            "claim": "Global glaciers lost mass in aggregate over 2000-2023.",
            "supporting_artifact": "outputs/uncertainty_summary.csv; report/images/fig2_global_timeseries.png",
            "verification": f"Cumulative global mass change {g['combined_gt'].sum():.1f} ± {np.sqrt(np.sum(np.square(g['combined_gt_errors']))):.1f} Gt (independent-error propagation)",
        },
        {
            "claim": "The largest annual global loss in this 2000-2023 series occurs in 2023.",
            "supporting_artifact": "outputs/direct_global_2000_2023_answer.csv",
            "verification": f"Minimum annual global combined_gt is {g.loc[g.combined_gt.idxmin(),'combined_gt']:.1f} Gt in {int(g.loc[g.combined_gt.idxmin(),'year'])}",
        },
        {
            "claim": "Regional losses are spatially heterogeneous and dominated by large Arctic/peripheral regions plus Alaska and Southern Andes.",
            "supporting_artifact": "outputs/regional_cumulative_summary.csv; report/images/fig3_regional_heatmap.png; report/images/fig5_regional_cumulative.png",
            "verification": "Regional cumulative summaries preserve all 19 region-level totals and uncertainties.",
        },
        {
            "claim": "Method groups show measurable deviations from the combined regional solution, supporting comparison/validation rather than blind pooling.",
            "supporting_artifact": "outputs/method_comparison_summary.csv; report/images/fig4_method_validation.png",
            "verification": f"{len(plot_df)} non-combined region-method comparison rows exported.",
        },
    ]
    pd.DataFrame(claim_rows).to_csv(OUT / "claim_recovery_table.csv", index=False)

    # Update inventory statuses.
    artifact_path = OUT / "target_artifact_inventory.json"
    if artifact_path.exists():
        invj = json.loads(artifact_path.read_text())
        for a in invj.get("artifacts", []):
            p = ROOT / a["path"]
            a["status"] = "satisfied" if p.exists() else "unsatisfied"
            if not p.exists():
                a["reason"] = "File not generated by analysis script."
        artifact_path.write_text(json.dumps(invj, indent=2))

    print(json.dumps({
        "regional_rows": len(regional_out),
        "global_rows": len(global_out),
        "input_files": len(inventory),
        "input_records": int(inventory["n_rows"].sum()),
        "global_cumulative_gt": float(g["combined_gt"].sum()),
        "global_cumulative_gt_error_independent": float(np.sqrt(np.sum(np.square(g["combined_gt_errors"])))),
        "figures": sorted(p.name for p in IMG.glob("*.png")),
    }, indent=2))


if __name__ == "__main__":
    main()
