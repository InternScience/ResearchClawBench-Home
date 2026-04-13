from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from shapely.geometry import Point
from sklearn.neighbors import BallTree


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_IMG = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORT_IMG.mkdir(parents=True, exist_ok=True)


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    clipped = s.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


def load_mangroves() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA / "mangroves" / "gmw_v4_ref_smpls_qad_v12.gpkg")
    gdf = gdf[["uid", "geometry"]].copy()
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


def load_countries() -> gpd.GeoDataFrame:
    countries = gpd.read_file(DATA / "ecosystem" / "UCSC_CWON_countrybounds.gpkg")
    cols = ["Country", "Country_2020", "ISO3", "Mang_Ha_2020", "geometry"]
    countries = countries[cols].copy()
    countries["country_name"] = countries["Country_2020"].fillna(countries["Country"])
    return countries[["country_name", "ISO3", "Mang_Ha_2020", "geometry"]]


def assign_countries(mangroves: gpd.GeoDataFrame, countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(mangroves, countries, how="left", predicate="intersects")
    joined["country_name"] = joined["country_name"].fillna("Offshore/Unassigned")
    joined["ISO3"] = joined["ISO3"].fillna("UNK")
    return joined.drop(columns=["index_right"])


def load_slr_scenario(path: Path, scenario: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    q_idx = int(np.abs(ds["quantiles"].values - 0.5).argmin())
    years = ds["years"].values
    yr_mask = (years >= 2090) & (years <= 2100)
    rates = ds["sea_level_change_rate"].isel(quantiles=q_idx, years=yr_mask).mean(dim="years").values
    df = pd.DataFrame(
        {
            "slr_lat": ds["lat"].values.astype(float),
            "slr_lon": ds["lon"].values.astype(float),
            "slr_rate_mm_yr": rates.astype(float),
            "scenario": scenario,
        }
    )
    return df


def haversine_tree(lat_lon_deg: np.ndarray) -> BallTree:
    return BallTree(np.deg2rad(lat_lon_deg), metric="haversine")


def nearest_slr_to_mangroves(mangroves: pd.DataFrame, slr_df: pd.DataFrame) -> pd.DataFrame:
    tree = haversine_tree(slr_df[["slr_lat", "slr_lon"]].to_numpy())
    dist, ind = tree.query(np.deg2rad(mangroves[["lat", "lon"]].to_numpy()), k=1)
    matched = slr_df.iloc[ind[:, 0]].reset_index(drop=True)
    out = mangroves.reset_index(drop=True).copy()
    out["slr_rate_mm_yr"] = matched["slr_rate_mm_yr"].values
    out["slr_match_km"] = dist[:, 0] * 6371.0
    return out


def build_tc_grid() -> pd.DataFrame:
    ds = xr.open_dataset(DATA / "tc" / "tracks_mit_mpi-esm1-2-hr_historical_reduced.nc")
    df = pd.DataFrame(
        {
            "lat": ds["lat"].values.astype(float),
            "lon": ds["lon"].values.astype(float),
            "wind": ds["wind"].values.astype(float),
        }
    ).dropna()
    df = df[(df["lat"].between(-40, 40)) & (df["lon"].between(-180, 180))]
    bins = 2.0
    df["lat_bin"] = np.floor(df["lat"] / bins) * bins + bins / 2
    df["lon_bin"] = np.floor(df["lon"] / bins) * bins + bins / 2
    df["cat45"] = (df["wind"] >= 58.0).astype(float)
    grid = (
        df.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            tc_point_count=("wind", "size"),
            tc_mean_wind=("wind", "mean"),
            tc_p90_wind=("wind", lambda x: float(np.percentile(x, 90))),
            tc_cat45_share=("cat45", "mean"),
        )
    )
    grid["tc_exposure_raw"] = (
        np.log1p(grid["tc_point_count"]) * (grid["tc_mean_wind"] / 33.0) * (1.0 + 2.0 * grid["tc_cat45_share"])
    )
    return grid


def nearest_tc_to_mangroves(mangroves: pd.DataFrame, tc_grid: pd.DataFrame) -> pd.DataFrame:
    tree = haversine_tree(tc_grid[["lat_bin", "lon_bin"]].to_numpy())
    dist, ind = tree.query(np.deg2rad(mangroves[["lat", "lon"]].to_numpy()), k=1)
    matched = tc_grid.iloc[ind[:, 0]].reset_index(drop=True)
    out = mangroves.reset_index(drop=True).copy()
    for col in ["tc_point_count", "tc_mean_wind", "tc_p90_wind", "tc_cat45_share", "tc_exposure_raw"]:
        out[col] = matched[col].values
    out["tc_match_km"] = dist[:, 0] * 6371.0
    return out


def aggregate_country_risk(mangrove_risk: pd.DataFrame, scenario: str, slr_bounds: tuple[float, float], tc_bounds: tuple[float, float], high_threshold: float) -> pd.DataFrame:
    df = mangrove_risk.copy()
    slr_lo, slr_hi = slr_bounds
    tc_lo, tc_hi = tc_bounds
    df["slr_norm"] = ((df["slr_rate_mm_yr"].clip(slr_lo, slr_hi) - slr_lo) / max(slr_hi - slr_lo, 1e-9)).clip(0, 1)
    df["tc_norm"] = ((df["tc_exposure_raw"].clip(tc_lo, tc_hi) - tc_lo) / max(tc_hi - tc_lo, 1e-9)).clip(0, 1)
    df["compound_risk"] = 0.45 * df["slr_norm"] + 0.35 * df["tc_norm"] + 0.20 * (df["slr_norm"] * df["tc_norm"])
    df["slr_ge_4"] = (df["slr_rate_mm_yr"] >= 4.0).astype(float)
    df["slr_ge_7"] = (df["slr_rate_mm_yr"] >= 7.0).astype(float)
    df["high_compound"] = (df["compound_risk"] >= high_threshold).astype(float)
    grouped = (
        df.groupby(["country_name", "ISO3"], as_index=False)
        .agg(
            mangrove_points=("uid", "count"),
            mean_slr_rate_mm_yr=("slr_rate_mm_yr", "mean"),
            p90_slr_rate_mm_yr=("slr_rate_mm_yr", lambda x: float(np.percentile(x, 90))),
            mean_tc_exposure=("tc_exposure_raw", "mean"),
            tc_cat45_share=("tc_cat45_share", "mean"),
            compound_risk=("compound_risk", "mean"),
            share_slr_ge_4=("slr_ge_4", "mean"),
            share_slr_ge_7=("slr_ge_7", "mean"),
            share_high_compound=("high_compound", "mean"),
        )
        .sort_values("compound_risk", ascending=False)
    )
    grouped["scenario"] = scenario
    grouped["risk_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def save_figures(country_results: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(12, 8))
    top = country_results[country_results["scenario"] == "SSP5-8.5"].nlargest(15, "compound_risk").sort_values("compound_risk")
    ax.barh(top["country_name"], top["compound_risk"], color="#c43c39")
    ax.set_xlabel("Composite risk index")
    ax.set_ylabel("")
    ax.set_title("Highest composite mangrove risk countries under SSP5-8.5")
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "top_risk_countries_ssp585.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    s585 = country_results[country_results["scenario"] == "SSP5-8.5"]
    sc = ax.scatter(
        s585["mean_slr_rate_mm_yr"],
        s585["mean_tc_exposure"],
        s=np.sqrt(s585["mangrove_points"]) * 10,
        c=s585["compound_risk"],
        cmap="YlOrRd",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axvline(4.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(7.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean 2090-2100 relative sea-level-rise rate (mm/yr)")
    ax.set_ylabel("Historical TC exposure proxy")
    ax.set_title("Compound risk emerges where high SLR overlaps intense cyclone exposure")
    fig.colorbar(sc, ax=ax, label="Composite risk index")
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "slr_tc_scatter_ssp585.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = (
        country_results.groupby("scenario", as_index=False)
        .agg(
            mean_risk=("compound_risk", "mean"),
            mean_share_slr_ge_4=("share_slr_ge_4", "mean"),
            mean_share_slr_ge_7=("share_slr_ge_7", "mean"),
        )
        .sort_values("scenario")
    )
    x = np.arange(len(plot_df))
    width = 0.25
    ax.bar(x - width, plot_df["mean_risk"], width=width, label="Mean composite risk", color="#dd8452")
    ax.bar(x, plot_df["mean_share_slr_ge_4"], width=width, label="Share with SLR >= 4 mm/yr", color="#55a868")
    ax.bar(x + width, plot_df["mean_share_slr_ge_7"], width=width, label="Share with SLR >= 7 mm/yr", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario"])
    ax.set_title("Scenario escalation increases SLR-driven exposure")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "scenario_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    mangroves = load_mangroves()
    countries = load_countries()
    mangroves = assign_countries(mangroves, countries)
    tc_grid = build_tc_grid()
    mangroves_tc = nearest_tc_to_mangroves(mangroves, tc_grid)

    scenario_files = {
        "SSP2-4.5": DATA / "slr" / "total_ssp245_medium_confidence_rates.nc",
        "SSP3-7.0": DATA / "slr" / "total_ssp370_medium_confidence_rates.nc",
        "SSP5-8.5": DATA / "slr" / "total_ssp585_medium_confidence_rates.nc",
    }

    scenario_frames = []
    for scenario, path in scenario_files.items():
        slr_df = load_slr_scenario(path, scenario)
        merged = nearest_slr_to_mangroves(mangroves_tc, slr_df)
        merged["scenario"] = scenario
        scenario_frames.append(merged)

    all_points = pd.concat(scenario_frames, ignore_index=True)
    slr_bounds = tuple(np.nanpercentile(all_points["slr_rate_mm_yr"], [5, 95]))
    tc_bounds = tuple(np.nanpercentile(all_points["tc_exposure_raw"], [5, 95]))
    all_points["slr_norm"] = ((all_points["slr_rate_mm_yr"].clip(*slr_bounds) - slr_bounds[0]) / max(slr_bounds[1] - slr_bounds[0], 1e-9)).clip(0, 1)
    all_points["tc_norm"] = ((all_points["tc_exposure_raw"].clip(*tc_bounds) - tc_bounds[0]) / max(tc_bounds[1] - tc_bounds[0], 1e-9)).clip(0, 1)
    all_points["compound_risk"] = 0.45 * all_points["slr_norm"] + 0.35 * all_points["tc_norm"] + 0.20 * (all_points["slr_norm"] * all_points["tc_norm"])
    high_threshold = float(all_points["compound_risk"].quantile(0.9))

    scenario_results = []
    point_outputs = []
    for merged in scenario_frames:
        scenario = str(merged["scenario"].iloc[0])
        enriched = merged.copy()
        point_outputs.append(
            enriched[
                [
                    "uid",
                    "country_name",
                    "ISO3",
                    "lat",
                    "lon",
                    "slr_rate_mm_yr",
                    "slr_match_km",
                    "tc_exposure_raw",
                    "tc_cat45_share",
                    "tc_match_km",
                    "scenario",
                ]
            ]
        )
        scenario_results.append(aggregate_country_risk(enriched, scenario, slr_bounds, tc_bounds, high_threshold))

    point_df = pd.concat(point_outputs, ignore_index=True)
    country_results = pd.concat(scenario_results, ignore_index=True)

    point_df.to_csv(OUTPUTS / "mangrove_point_risk_samples.csv", index=False)
    country_results.to_csv(OUTPUTS / "country_composite_risk.csv", index=False)

    summary = {
        "n_mangrove_points": int(len(mangroves)),
        "n_countries_with_mangroves": int(country_results["ISO3"].nunique()),
        "tc_grid_cells": int(len(tc_grid)),
        "scenario_mean_risk": country_results.groupby("scenario")["compound_risk"].mean().to_dict(),
        "scenario_top10_mean_risk": country_results.groupby("scenario").apply(lambda x: float(x.nlargest(10, "compound_risk")["compound_risk"].mean())).to_dict(),
    }
    (OUTPUTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    save_figures(country_results)


if __name__ == "__main__":
    main()
