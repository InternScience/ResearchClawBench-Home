from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "code" / "vendor"
LIB_DIRS = [
    VENDOR / "netcdf4.libs",
    VENDOR / "pyogrio.libs",
    VENDOR / "shapely.libs",
    VENDOR / "pyproj.libs",
]
existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [str(p) for p in LIB_DIRS if p.exists()] + ([existing_ld] if existing_ld else [])
)
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import json
from dataclasses import dataclass

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point


MPL_DIR = ROOT / "outputs" / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_IMG = ROOT / "report" / "images"
REPORT_IMG.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    slr_path: Path
    tc_multiplier: float


SCENARIOS = [
    ScenarioConfig("ssp245", DATA / "slr" / "total_ssp245_medium_confidence_rates.nc", 1.00),
    ScenarioConfig("ssp370", DATA / "slr" / "total_ssp370_medium_confidence_rates.nc", 1.08),
    ScenarioConfig("ssp585", DATA / "slr" / "total_ssp585_medium_confidence_rates.nc", 1.16),
]


def load_mangroves() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA / "mangroves" / "gmw_v4_ref_smpls_qad_v12.gpkg")
    gdf = gdf[gdf["ref_cls"] == 1].copy()
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf


def load_countries() -> gpd.GeoDataFrame:
    countries = gpd.read_file(DATA / "ecosystem" / "UCSC_CWON_countrybounds.gpkg")
    keep = [
        "ISO3",
        "Country_2020",
        "Mang_Ha_2020",
        "Risk_Pop_2020",
        "Risk_Stock_2020",
        "Ben_Pop_2020",
        "Ben_Stock_2020",
        "geometry",
    ]
    countries = countries[keep].rename(columns={"Country_2020": "Country"})
    return countries


def read_slr_points(path: Path) -> pd.DataFrame:
    ds = netCDF4.Dataset(path)
    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    lon = np.asarray(ds.variables["lon"][:], dtype=float)
    years = np.asarray(ds.variables["years"][:], dtype=int)
    quantiles = np.asarray(ds.variables["quantiles"][:], dtype=float)
    q_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    rate = np.asarray(ds.variables["sea_level_change_rate"][q_idx, :, :], dtype=float)
    fill = getattr(ds.variables["sea_level_change_rate"], "_FillValue", -32768)
    rate[rate == fill] = np.nan
    ds.close()

    df = pd.DataFrame({"lon": lon, "lat": lat})
    for i, year in enumerate(years):
        df[f"slr_{year}"] = rate[i, :]
    df["slr_mean_2020_2100"] = np.nanmean(rate, axis=0)
    df["slr_2100"] = rate[-1, :]
    return df


def load_tc_points() -> pd.DataFrame:
    ds = netCDF4.Dataset(DATA / "tc" / "tracks_mit_mpi-esm1-2-hr_historical_reduced.nc")
    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    lon = np.asarray(ds.variables["lon"][:], dtype=float)
    wind = np.asarray(ds.variables["wind"][:], dtype=float)
    ds.close()
    lon = ((lon + 180) % 360) - 180
    return pd.DataFrame({"lon": lon, "lat": lat, "wind": wind})


def compute_tc_grid(tc: pd.DataFrame, resolution: float = 1.0) -> pd.DataFrame:
    tc = tc.dropna().copy()
    tc["lon_bin"] = np.floor(tc["lon"] / resolution) * resolution + resolution / 2
    tc["lat_bin"] = np.floor(tc["lat"] / resolution) * resolution + resolution / 2
    grid = (
        tc.groupby(["lon_bin", "lat_bin"], observed=True)
        .agg(
            tc_records=("wind", "size"),
            mean_wind=("wind", "mean"),
            p90_wind=("wind", lambda x: float(np.nanpercentile(x, 90))),
            intense_share=("wind", lambda x: float(np.mean(x >= 60.0))),
            major_share=("wind", lambda x: float(np.mean(x >= 50.0))),
        )
        .reset_index()
        .rename(columns={"lon_bin": "lon", "lat_bin": "lat"})
    )
    max_records = grid["tc_records"].max()
    grid["freq_norm"] = grid["tc_records"] / max_records
    grid["wind_norm"] = grid["mean_wind"] / grid["mean_wind"].max()
    grid["baseline_tc_hazard"] = (
        0.45 * grid["freq_norm"]
        + 0.25 * grid["wind_norm"]
        + 0.20 * grid["major_share"]
        + 0.10 * grid["intense_share"]
    )
    return grid


def spatial_join_nearest(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_lon: str = "lon",
    left_lat: str = "lat",
    right_lon: str = "lon",
    right_lat: str = "lat",
    max_distance_deg: float | None = None,
) -> pd.DataFrame:
    left = gpd.GeoDataFrame(
        left_df.copy(),
        geometry=gpd.points_from_xy(left_df[left_lon], left_df[left_lat]),
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        right_df.copy(),
        geometry=gpd.points_from_xy(right_df[right_lon], right_df[right_lat]),
        crs="EPSG:4326",
    )
    left_proj = left.to_crs("EPSG:3857")
    right_proj = right.to_crs("EPSG:3857")
    joined = gpd.sjoin_nearest(
        left_proj,
        right_proj,
        how="left",
        distance_col="join_distance",
        max_distance=None if max_distance_deg is None else max_distance_deg * 111_000,
    )
    joined = pd.DataFrame(joined.drop(columns="geometry")).reset_index(drop=True)
    if "index_right" in joined.columns:
        right_reset = right_df.reset_index(drop=True).add_prefix("right_")
        joined = joined.merge(
            right_reset,
            left_on="index_right",
            right_index=True,
            how="left",
        )
        joined = joined.drop(columns=[c for c in joined.columns if c in right_df.columns])
        rename_map = {f"right_{c}": c for c in right_df.columns}
        joined = joined.rename(columns=rename_map)
    return joined.reset_index(drop=True)


def assign_basin(row: pd.Series) -> str:
    lon = row["lon"]
    lat = row["lat"]
    if lat >= 0 and -100 <= lon <= -30:
        return "north_america_central_america"
    if lat >= 0 and 100 <= lon <= 180:
        return "northwest_pacific"
    if lat < 0 and 20 <= lon <= 120:
        return "south_indian"
    if lat < 0 and 120 < lon <= 180:
        return "oceania"
    if lat >= 0 and 30 <= lon < 100:
        return "north_indian"
    if lat < 0 and -60 <= lon < 20:
        return "south_atlantic_margin"
    return "other_tropics"


def future_basin_multiplier(basin: str, scenario: str) -> float:
    base = {
        "north_america_central_america": 1.15,
        "northwest_pacific": 1.08,
        "south_indian": 1.07,
        "north_indian": 1.05,
        "south_atlantic_margin": 1.03,
        "oceania": 0.88,
        "other_tropics": 1.03,
    }
    scenario_scale = {"ssp245": 1.00, "ssp370": 1.05, "ssp585": 1.10}[scenario]
    value = 1.0 + (base[basin] - 1.0) * scenario_scale
    return value


def normalize_series(x: pd.Series) -> pd.Series:
    denom = x.max() - x.min()
    if denom == 0 or np.isnan(denom):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / denom


def build_point_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mangroves = load_mangroves()
    countries = load_countries()
    mangroves = gpd.sjoin(
        gpd.GeoDataFrame(mangroves, geometry="geometry", crs="EPSG:4326"),
        countries[["ISO3", "Country", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    tc_grid = compute_tc_grid(load_tc_points(), resolution=1.0)
    tc_join = spatial_join_nearest(mangroves[["lon", "lat"]], tc_grid, max_distance_deg=2.0)
    slr_join = {}
    for cfg in SCENARIOS:
        slr_points = read_slr_points(cfg.slr_path)
        slr_join[cfg.name] = spatial_join_nearest(
            mangroves[["lon", "lat"]],
            slr_points[["lon", "lat", "slr_mean_2020_2100", "slr_2100"]],
            max_distance_deg=2.0,
        )
    points = mangroves.reset_index(drop=True).copy()
    points["tc_join_distance"] = tc_join["join_distance"]
    for col in ["tc_records", "mean_wind", "p90_wind", "intense_share", "major_share", "baseline_tc_hazard"]:
        points[col] = tc_join[col]
    points["basin"] = points.apply(assign_basin, axis=1)
    for cfg in SCENARIOS:
        points[f"{cfg.name}_slr_mean"] = slr_join[cfg.name]["slr_mean_2020_2100"]
        points[f"{cfg.name}_slr_2100"] = slr_join[cfg.name]["slr_2100"]
    return points, countries, tc_grid


def compute_risk(points: pd.DataFrame) -> pd.DataFrame:
    points = points.copy()
    tc_base = points["baseline_tc_hazard"].fillna(0.0)
    intense = points["intense_share"].fillna(0.0)
    for cfg in SCENARIOS:
        slr = points[f"{cfg.name}_slr_2100"]
        points[f"{cfg.name}_slr_stress_4"] = (slr >= 4.0).astype(int)
        points[f"{cfg.name}_slr_stress_7"] = (slr >= 7.0).astype(int)
        points[f"{cfg.name}_tc_multiplier"] = points["basin"].map(
            lambda basin: future_basin_multiplier(basin, cfg.name)
        ) * cfg.tc_multiplier
        points[f"{cfg.name}_future_tc_hazard"] = tc_base * points[f"{cfg.name}_tc_multiplier"]
    all_slr = pd.concat([points[f"{cfg.name}_slr_2100"] for cfg in SCENARIOS], ignore_index=True)
    slr_min, slr_max = float(all_slr.min()), float(all_slr.max())
    all_tc = pd.concat([points[f"{cfg.name}_future_tc_hazard"] for cfg in SCENARIOS], ignore_index=True)
    tc_min, tc_max = float(all_tc.min()), float(all_tc.max())

    for cfg in SCENARIOS:
        slr = points[f"{cfg.name}_slr_2100"].fillna(points[f"{cfg.name}_slr_2100"].median())
        future_tc = points[f"{cfg.name}_future_tc_hazard"].fillna(0.0)
        points[f"{cfg.name}_slr_norm"] = (slr - slr_min) / (slr_max - slr_min)
        points[f"{cfg.name}_tc_shift_norm"] = (future_tc - tc_min) / (tc_max - tc_min)
        points[f"{cfg.name}_recovery_pressure"] = 0.6 * points[f"{cfg.name}_tc_shift_norm"] + 0.4 * intense
        points[f"{cfg.name}_composite_risk"] = (
            0.55 * points[f"{cfg.name}_slr_norm"]
            + 0.35 * points[f"{cfg.name}_tc_shift_norm"]
            + 0.10 * points[f"{cfg.name}_recovery_pressure"]
        )
        points[f"{cfg.name}_risk_class"] = pd.qcut(
            points[f"{cfg.name}_composite_risk"],
            q=5,
            labels=["Very low", "Low", "Moderate", "High", "Very high"],
            duplicates="drop",
        )
    return points


def aggregate_country_metrics(points: pd.DataFrame, countries: gpd.GeoDataFrame) -> pd.DataFrame:
    summaries = []
    for cfg in SCENARIOS:
        global_high = float(points[f"{cfg.name}_composite_risk"].quantile(0.8))
        agg = (
            points.groupby(["ISO3", "Country"], dropna=False)
            .agg(
                n_points=("uid", "size"),
                mean_scenario_risk=(f"{cfg.name}_composite_risk", "mean"),
                high_risk_share=(f"{cfg.name}_composite_risk", lambda x: float(np.mean(x >= global_high))),
                slr_ge_4_share=(f"{cfg.name}_slr_stress_4", "mean"),
                slr_ge_7_share=(f"{cfg.name}_slr_stress_7", "mean"),
                baseline_tc_hazard=("baseline_tc_hazard", "mean"),
                future_tc_hazard=(f"{cfg.name}_future_tc_hazard", "mean"),
            )
            .reset_index()
        )
        agg["scenario"] = cfg.name
        summaries.append(agg)
    country = pd.concat(summaries, ignore_index=True)
    country = country.merge(
        pd.DataFrame(countries.drop(columns="geometry")),
        on=["ISO3", "Country"],
        how="left",
    )
    country["risk_pop_proxy"] = country["mean_scenario_risk"] * country["Risk_Pop_2020"].fillna(0)
    country["risk_stock_proxy"] = country["mean_scenario_risk"] * country["Risk_Stock_2020"].fillna(0)
    return country


def save_outputs(points: pd.DataFrame, country: pd.DataFrame, tc_grid: pd.DataFrame) -> None:
    points.to_csv(OUTPUTS / "mangrove_point_risk.csv", index=False)
    country.to_csv(OUTPUTS / "country_risk_summary.csv", index=False)
    tc_grid.to_csv(OUTPUTS / "tc_grid_baseline.csv", index=False)

    metadata = {
        "n_mangrove_points": int(len(points)),
        "scenarios": [cfg.name for cfg in SCENARIOS],
        "slr_thresholds_mm_per_year": [4.0, 7.0],
        "tc_recovery_threshold_ms": 60.0,
        "notes": [
            "Mangrove input is a 10% sampled point layer rather than polygons.",
            "Future tropical cyclone regime shifts are approximated using literature-informed basin multipliers applied to historical hazard baselines.",
            "Sea-level rise uses the median (0.5 quantile) regional relative rate at the nearest AR6 coastal point.",
        ],
    }
    (OUTPUTS / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_data_overview(points: pd.DataFrame, tc_grid: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].scatter(tc_grid["lon"], tc_grid["lat"], c=tc_grid["baseline_tc_hazard"], s=4, cmap="magma", alpha=0.7)
    axes[0].scatter(points["lon"], points["lat"], s=1, color="#0a7f5a", alpha=0.35)
    axes[0].set_title("Mangrove Samples and Baseline TC Hazard Grid")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    axes[1].hist(points["ssp585_slr_2100"].dropna(), bins=40, color="#2b6cb0", alpha=0.85)
    axes[1].axvline(4, color="#d97706", linestyle="--", linewidth=2)
    axes[1].axvline(7, color="#b91c1c", linestyle="--", linewidth=2)
    axes[1].set_title("Nearest-Point Relative SLR Rates by 2100 (SSP5-8.5)")
    axes[1].set_xlabel("SLR rate (mm/yr)")
    axes[1].set_ylabel("Mangrove sample count")

    fig.savefig(REPORT_IMG / "figure_1_data_overview.png", dpi=220)
    plt.close(fig)


def plot_scenario_distributions(points: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, cfg in zip(axes, SCENARIOS):
        ax.hist(points[f"{cfg.name}_composite_risk"], bins=35, color="#2563eb", alpha=0.85)
        ax.set_title(cfg.name.upper())
        ax.set_xlabel("Composite risk index")
        ax.set_ylabel("Mangrove sample count")
    fig.savefig(REPORT_IMG / "figure_2_risk_distributions.png", dpi=220)
    plt.close(fig)


def plot_global_risk_map(points: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for ax, cfg in zip(axes, SCENARIOS):
        sc = ax.scatter(
            points["lon"],
            points["lat"],
            c=points[f"{cfg.name}_composite_risk"],
            s=2,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.7,
        )
        ax.set_title(f"Composite Risk {cfg.name.upper()}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Composite risk")
    fig.savefig(REPORT_IMG / "figure_3_global_risk_map.png", dpi=240)
    plt.close(fig)


def plot_country_rankings(country: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for ax, cfg in zip(axes, SCENARIOS):
        subset = country[country["scenario"] == cfg.name].copy()
        subset = subset.sort_values("mean_scenario_risk", ascending=False).head(15)
        sns.barplot(data=subset, y="Country", x="mean_scenario_risk", color="#be123c", ax=ax)
        ax.set_title(f"Highest Mean Risk Countries: {cfg.name.upper()}")
        ax.set_xlabel("Mean composite risk")
        ax.set_ylabel("")
    fig.savefig(REPORT_IMG / "figure_4_country_rankings.png", dpi=220)
    plt.close(fig)


def plot_validation_relationship(points: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].scatter(points["ssp585_slr_2100"], points["ssp585_composite_risk"], s=3, alpha=0.2, color="#0369a1")
    axes[0].axvline(4, color="#d97706", linestyle="--")
    axes[0].axvline(7, color="#b91c1c", linestyle="--")
    axes[0].set_xlabel("SLR rate by 2100 (mm/yr)")
    axes[0].set_ylabel("Composite risk (SSP5-8.5)")
    axes[0].set_title("Risk Increases with End-Century SLR Stress")

    axes[1].scatter(points["baseline_tc_hazard"], points["ssp585_future_tc_hazard"], s=3, alpha=0.2, color="#166534")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("Baseline TC hazard")
    axes[1].set_ylabel("Future TC hazard proxy (SSP5-8.5)")
    axes[1].set_title("Regional TC Shift Proxy Preserves Baseline Hotspots")
    fig.savefig(REPORT_IMG / "figure_5_validation.png", dpi=220)
    plt.close(fig)


def build_report_tables(points: pd.DataFrame, country: pd.DataFrame) -> None:
    scenario_summary = []
    for cfg in SCENARIOS:
        s = {
            "scenario": cfg.name,
            "mean_risk": float(points[f"{cfg.name}_composite_risk"].mean()),
            "p90_risk": float(points[f"{cfg.name}_composite_risk"].quantile(0.9)),
            "share_slr_ge_4": float(points[f"{cfg.name}_slr_stress_4"].mean()),
            "share_slr_ge_7": float(points[f"{cfg.name}_slr_stress_7"].mean()),
            "mean_future_tc_hazard": float(points[f"{cfg.name}_future_tc_hazard"].mean()),
        }
        scenario_summary.append(s)
    pd.DataFrame(scenario_summary).to_csv(OUTPUTS / "scenario_summary.csv", index=False)

    top = (
        country[country["scenario"] == "ssp585"]
        .sort_values("mean_scenario_risk", ascending=False)
        .head(20)
    )
    top.to_csv(OUTPUTS / "top20_countries_ssp585.csv", index=False)


def main() -> None:
    points, countries, tc_grid = build_point_dataset()
    points = compute_risk(points)
    country = aggregate_country_metrics(points, countries)
    save_outputs(points, country, tc_grid)
    build_report_tables(points, country)
    plot_data_overview(points, tc_grid)
    plot_scenario_distributions(points)
    plot_global_risk_map(points)
    plot_country_rankings(country)
    plot_validation_relationship(points)


if __name__ == "__main__":
    main()
