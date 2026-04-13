from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def annuity_factor(wacc: float, lifetime: int) -> float:
    if wacc == 0:
        return 1 / lifetime
    return (wacc * (1 + wacc) ** lifetime) / (((1 + wacc) ** lifetime) - 1)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def build_assumptions() -> dict:
    component_capex = {
        "electrolyzer": 700,
        "renewables": 900,
        "battery": 180,
        "h2_storage": 120,
        "ammonia_synthesis": 250,
        "reconversion": 350,
        "port_export": 120,
        "port_import": 80,
    }
    component_life = {
        "electrolyzer": 20,
        "renewables": 25,
        "battery": 15,
        "h2_storage": 25,
        "ammonia_synthesis": 25,
        "reconversion": 25,
        "port_export": 30,
        "port_import": 30,
    }
    component_fixed_opex_rate = {
        "electrolyzer": 0.03,
        "renewables": 0.025,
        "battery": 0.02,
        "h2_storage": 0.02,
        "ammonia_synthesis": 0.04,
        "reconversion": 0.04,
        "port_export": 0.03,
        "port_import": 0.03,
    }

    scenarios = {
        "africa_high_risk": {
            "description": "Higher African financing costs and a still-favorable European rate environment.",
            "africa_wacc": 0.14,
            "europe_wacc": 0.06,
            "policy_credit": 0.00,
        },
        "de_risked_africa": {
            "description": "Public guarantees and concessional finance compress African WACC.",
            "africa_wacc": 0.08,
            "europe_wacc": 0.06,
            "policy_credit": 0.00,
        },
        "global_high_rate": {
            "description": "Higher global interest-rate regime affecting both Africa and Europe.",
            "africa_wacc": 0.18,
            "europe_wacc": 0.10,
            "policy_credit": 0.00,
        },
        "de_risked_plus_policy": {
            "description": "African de-risking plus delivered-hydrogen support at the European border.",
            "africa_wacc": 0.08,
            "europe_wacc": 0.06,
            "policy_credit": 0.75,
        },
    }

    europe_cases = {
        "europe_low_rate": {"wacc": 0.05, "renewable_cf": 0.26},
        "europe_high_rate": {"wacc": 0.09, "renewable_cf": 0.26},
    }

    return {
        "component_capex_usd_per_kw": component_capex,
        "component_life_years": component_life,
        "component_fixed_opex_rate": component_fixed_opex_rate,
        "electrolyzer_specific_consumption_kwh_per_kg": 52,
        "ammonia_conversion_and_reconversion_usd_per_kg_h2": 0.90,
        "shipping_usd_per_kg_h2_base": 0.42,
        "shipping_usd_per_kg_h2_per_1000km": 0.06,
        "distance_costs_usd_per_kg_h2_per_km": {
            "road": 0.0013,
            "grid": 0.0005,
            "ocean": 0.0010,
            "water": 0.0007,
        },
        "scenario_definitions": scenarios,
        "europe_cases": europe_cases,
    }


def renewable_mix_and_cf(df: pd.DataFrame) -> pd.DataFrame:
    mix = df["theo_pv"] / (df["theo_pv"] + df["theo_wind"])
    cf = 0.18 + 0.22 * df["theo_pv"] + 0.28 * df["theo_wind"] + 0.04 * np.minimum(df["theo_pv"], df["theo_wind"])
    cf = cf.clip(0.22, 0.62)
    out = df.copy()
    out["pv_share"] = mix.clip(0.2, 0.8)
    out["hybrid_cf"] = cf
    return out


def renewable_lcoe_per_kwh(cf: pd.Series, wacc: float) -> pd.Series:
    assumptions = build_assumptions()
    capex = assumptions["component_capex_usd_per_kw"]
    life = assumptions["component_life_years"]
    opex = assumptions["component_fixed_opex_rate"]

    annualized = (
        capex["renewables"] * annuity_factor(wacc, life["renewables"])
        + capex["renewables"] * opex["renewables"]
        + capex["battery"] * annuity_factor(wacc, life["battery"]) * 0.18
        + capex["battery"] * opex["battery"] * 0.18
    )
    annual_generation = 8760 * cf
    return annualized / annual_generation


def africa_costs(df: pd.DataFrame, scenario_name: str, scenario: dict) -> pd.DataFrame:
    assumptions = build_assumptions()
    kwh_per_kg = assumptions["electrolyzer_specific_consumption_kwh_per_kg"]
    capex = assumptions["component_capex_usd_per_kw"]
    life = assumptions["component_life_years"]
    opex = assumptions["component_fixed_opex_rate"]

    out = renewable_mix_and_cf(df)
    wacc = scenario["africa_wacc"]
    out["renewable_lcoe_usd_per_kwh"] = renewable_lcoe_per_kwh(out["hybrid_cf"], wacc)

    electrolysis_capex = (
        capex["electrolyzer"] * annuity_factor(wacc, life["electrolyzer"])
        + capex["electrolyzer"] * opex["electrolyzer"]
    ) / (8760 * 0.72 / kwh_per_kg)
    storage_and_synthesis = (
        (capex["h2_storage"] * annuity_factor(wacc, life["h2_storage"]) + capex["h2_storage"] * opex["h2_storage"])
        + (capex["ammonia_synthesis"] * annuity_factor(wacc, life["ammonia_synthesis"]) + capex["ammonia_synthesis"] * opex["ammonia_synthesis"])
        + (capex["port_export"] * annuity_factor(wacc, life["port_export"]) + capex["port_export"] * opex["port_export"])
    ) / 1000

    out["power_cost_usd_per_kg"] = out["renewable_lcoe_usd_per_kwh"] * kwh_per_kg
    out["electrolysis_capex_usd_per_kg"] = electrolysis_capex
    out["water_access_cost_usd_per_kg"] = 0.08 + out["waterbody_dist_km"] * assumptions["distance_costs_usd_per_kg_h2_per_km"]["water"]
    out["road_cost_usd_per_kg"] = out["road_dist_km"] * assumptions["distance_costs_usd_per_kg_h2_per_km"]["road"]
    out["grid_backup_cost_usd_per_kg"] = out["grid_dist_km"] * assumptions["distance_costs_usd_per_kg_h2_per_km"]["grid"]
    out["ocean_to_port_cost_usd_per_kg"] = out["ocean_dist_km"] * assumptions["distance_costs_usd_per_kg_h2_per_km"]["ocean"]
    out["ammonia_chain_usd_per_kg"] = assumptions["ammonia_conversion_and_reconversion_usd_per_kg_h2"]
    out["shipping_usd_per_kg"] = assumptions["shipping_usd_per_kg_h2_base"] + assumptions["shipping_usd_per_kg_h2_per_1000km"] * 5.6
    out["processing_capex_usd_per_kg"] = storage_and_synthesis
    out["delivered_cost_usd_per_kg"] = (
        out["power_cost_usd_per_kg"]
        + out["electrolysis_capex_usd_per_kg"]
        + out["processing_capex_usd_per_kg"]
        + out["water_access_cost_usd_per_kg"]
        + out["road_cost_usd_per_kg"]
        + out["grid_backup_cost_usd_per_kg"]
        + out["ocean_to_port_cost_usd_per_kg"]
        + out["ammonia_chain_usd_per_kg"]
        + out["shipping_usd_per_kg"]
        - scenario["policy_credit"]
    )
    out["scenario"] = scenario_name
    out["africa_wacc"] = wacc
    out["europe_wacc"] = scenario["europe_wacc"]
    out["policy_credit_usd_per_kg"] = scenario["policy_credit"]
    return out


def europe_benchmark() -> pd.DataFrame:
    assumptions = build_assumptions()
    capex = assumptions["component_capex_usd_per_kw"]
    life = assumptions["component_life_years"]
    opex = assumptions["component_fixed_opex_rate"]
    kwh_per_kg = assumptions["electrolyzer_specific_consumption_kwh_per_kg"]

    rows = []
    for name, case in assumptions["europe_cases"].items():
        wacc = case["wacc"]
        cf = case["renewable_cf"]
        renewable_cost = renewable_lcoe_per_kwh(pd.Series([cf]), wacc).iloc[0] * kwh_per_kg
        electrolysis_capex = (
            capex["electrolyzer"] * annuity_factor(wacc, life["electrolyzer"])
            + capex["electrolyzer"] * opex["electrolyzer"]
        ) / (8760 * 0.72 / kwh_per_kg)
        storage = (
            capex["h2_storage"] * annuity_factor(wacc, life["h2_storage"])
            + capex["h2_storage"] * opex["h2_storage"]
        ) / 1300
        rows.append(
            {
                "scenario": name,
                "wacc": wacc,
                "renewable_cf": cf,
                "delivered_cost_usd_per_kg": renewable_cost + electrolysis_capex + storage + 0.24,
            }
        )
    return pd.DataFrame(rows)


def map_country_names(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    world = gpd.read_file(DATA_DIR / "africa_map" / "ne_10m_admin_0_countries.shp")
    africa = world[world["CONTINENT"] == "Africa"][["ADMIN", "geometry"]].to_crs(points.crs)
    joined = gpd.sjoin(points, africa, how="left", predicate="within")
    joined["country"] = joined["ADMIN"].fillna("Offshore/Unknown")
    return joined.drop(columns=["index_right", "ADMIN"], errors="ignore")


def create_figures(results: pd.DataFrame, europe: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    gdf = gpd.GeoDataFrame(
        results[results["scenario"] == "africa_high_risk"].copy(),
        geometry=gpd.points_from_xy(results[results["scenario"] == "africa_high_risk"]["lon"], results[results["scenario"] == "africa_high_risk"]["lat"]),
        crs="EPSG:4326",
    )
    world = gpd.read_file(DATA_DIR / "africa_map" / "ne_10m_admin_0_countries.shp")
    africa = world[world["CONTINENT"] == "Africa"].to_crs(gdf.crs)

    fig, ax = plt.subplots(figsize=(9, 8))
    africa.plot(ax=ax, color="#f0efe9", edgecolor="#9d998c", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column="delivered_cost_usd_per_kg",
        cmap="viridis_r",
        markersize=50,
        legend=True,
        legend_kwds={"label": "Delivered cost to Europe (USD/kg H2)"},
    )
    ax.set_title("African candidate sites: delivered green hydrogen cost in the baseline scenario")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "africa_cost_map.png", dpi=220)
    plt.close(fig)

    top = (
        results.groupby("scenario", as_index=False)["delivered_cost_usd_per_kg"]
        .min()
        .rename(columns={"delivered_cost_usd_per_kg": "best_site_cost"})
    )
    eu = europe.rename(columns={"scenario": "case", "delivered_cost_usd_per_kg": "cost"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, x="scenario", y="best_site_cost", hue="scenario", dodge=False, palette="deep", legend=False, ax=ax)
    ax.axhline(eu.loc[eu["case"] == "europe_low_rate", "cost"].iloc[0], color="black", linestyle="--", linewidth=1.2, label="Europe low-rate benchmark")
    ax.axhline(eu.loc[eu["case"] == "europe_high_rate", "cost"].iloc[0], color="black", linestyle=":", linewidth=1.2, label="Europe high-rate benchmark")
    ax.set_ylabel("Lowest delivered cost (USD/kg H2)")
    ax.set_xlabel("")
    ax.set_title("Best African site under each financing and policy scenario")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "scenario_comparison.png", dpi=220)
    plt.close(fig)

    baseline = results[results["scenario"] == "africa_high_risk"].copy()
    derisked = results[results["scenario"] == "de_risked_africa"].copy()[["hex_id", "delivered_cost_usd_per_kg"]].rename(
        columns={"delivered_cost_usd_per_kg": "de_risked_cost"}
    )
    merged = baseline.merge(derisked, on="hex_id")
    merged["savings_usd_per_kg"] = merged["delivered_cost_usd_per_kg"] - merged["de_risked_cost"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=merged, x="delivered_cost_usd_per_kg", y="savings_usd_per_kg", hue="hybrid_cf", palette="viridis", s=70, ax=ax)
    ax.set_xlabel("Baseline delivered cost (USD/kg H2)")
    ax.set_ylabel("Savings from de-risking (USD/kg H2)")
    ax.set_title("Financing relief matters most for low-cost, high-quality sites")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "derisking_sensitivity.png", dpi=220)
    plt.close(fig)

    component_cols = [
        "power_cost_usd_per_kg",
        "electrolysis_capex_usd_per_kg",
        "processing_capex_usd_per_kg",
        "water_access_cost_usd_per_kg",
        "road_cost_usd_per_kg",
        "grid_backup_cost_usd_per_kg",
        "ocean_to_port_cost_usd_per_kg",
        "ammonia_chain_usd_per_kg",
        "shipping_usd_per_kg",
    ]
    best_hex = baseline.nsmallest(1, "delivered_cost_usd_per_kg").iloc[0]
    component_series = best_hex[component_cols].rename(
        {
            "power_cost_usd_per_kg": "Renewable power",
            "electrolysis_capex_usd_per_kg": "Electrolyzer capex",
            "processing_capex_usd_per_kg": "Processing capex",
            "water_access_cost_usd_per_kg": "Water access",
            "road_cost_usd_per_kg": "Road link",
            "grid_backup_cost_usd_per_kg": "Grid backup",
            "ocean_to_port_cost_usd_per_kg": "Port connection",
            "ammonia_chain_usd_per_kg": "Ammonia conversion/reconversion",
            "shipping_usd_per_kg": "Shipping",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=component_series.values, y=component_series.index, orient="h", color="#4c72b0", ax=ax)
    ax.set_xlabel("USD/kg H2")
    ax.set_ylabel("")
    ax.set_title("Delivered-cost composition for the lowest-cost African site")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "cost_breakdown.png", dpi=220)
    plt.close(fig)


def write_outputs(results: pd.DataFrame, europe: pd.DataFrame, assumptions: dict) -> None:
    points = gpd.GeoDataFrame(results.copy(), geometry=gpd.points_from_xy(results["lon"], results["lat"]), crs="EPSG:4326")
    points = map_country_names(points)
    results = pd.DataFrame(points.drop(columns="geometry"))

    results.to_csv(OUTPUT_DIR / "site_results.csv", index=False)
    europe.to_csv(OUTPUT_DIR / "europe_benchmark.csv", index=False)

    scenario_summary = (
        results.groupby("scenario")
        .agg(
            best_cost_usd_per_kg=("delivered_cost_usd_per_kg", "min"),
            median_cost_usd_per_kg=("delivered_cost_usd_per_kg", "median"),
            p90_cost_usd_per_kg=("delivered_cost_usd_per_kg", lambda s: s.quantile(0.9)),
            best_country=("country", lambda s: results.loc[s.index, :].sort_values("delivered_cost_usd_per_kg").iloc[0]["country"]),
        )
        .reset_index()
    )
    scenario_summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)

    top_sites = (
        results.sort_values(["scenario", "delivered_cost_usd_per_kg"])
        .groupby("scenario")
        .head(5)
    )
    top_sites.to_csv(OUTPUT_DIR / "top_sites_by_scenario.csv", index=False)

    comparison_rows = []
    for _, africa_row in scenario_summary.iterrows():
        for _, europe_row in europe.iterrows():
            comparison_rows.append(
                {
                    "africa_scenario": africa_row["scenario"],
                    "europe_case": europe_row["scenario"],
                    "africa_best_cost_usd_per_kg": africa_row["best_cost_usd_per_kg"],
                    "europe_cost_usd_per_kg": europe_row["delivered_cost_usd_per_kg"],
                    "africa_advantage_usd_per_kg": europe_row["delivered_cost_usd_per_kg"] - africa_row["best_cost_usd_per_kg"],
                }
            )
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "africa_vs_europe.csv", index=False)

    with open(OUTPUT_DIR / "assumptions.json", "w", encoding="utf-8") as f:
        json.dump(assumptions, f, indent=2)


def main() -> None:
    ensure_dirs()
    assumptions = build_assumptions()
    sites = pd.read_csv(DATA_DIR / "hex_final_NA_min.csv")

    frames = []
    for scenario_name, scenario in assumptions["scenario_definitions"].items():
        frames.append(africa_costs(sites, scenario_name, scenario))
    results = pd.concat(frames, ignore_index=True)
    europe = europe_benchmark()

    write_outputs(results, europe, assumptions)
    create_figures(results, europe)


if __name__ == "__main__":
    main()
