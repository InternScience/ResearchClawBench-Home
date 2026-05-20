#!/usr/bin/env python3
"""
Transparent geospatial LCOH model for African green hydrogen delivered to Europe (via NH3).
Scenarios: base, de-risked (low WACC), high-interest, policy-subsidy.
Outputs: CSV results, figures (map, bar, box), comparison table.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point

# Paths
DATA_PATH = "data/hex_final_NA_min.csv"
SHAPEFILE = "data/africa_map/ne_10m_admin_0_countries.shp"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# 2030 parameters (transparent, literature-aligned)
ELECTRO_CAPEX = 500          # $/kW
ELEC_EFF = 50                # kWh/kg H2
LIFETIME = 25                # years
OPEX_RATIO = 0.03            # % of CAPEX
PV_CF_SCALE = 0.22           # realistic CF from theo_pv
WIND_CF_SCALE = 0.28         # realistic CF from theo_wind
INFRA_COST_PER_KM = 0.05     # $/kg per 100km (grid/road/water)
OCEAN_DIST_COST = 0.8        # $/kg fixed shipping + reconversion
EUROPE_LCOH = 4.5            # $/kg reference

def compute_lcoh(row, wacc, subsidy=0.0):
    """Compute delivered LCOH ($/kg) for one site."""
    # Renewable potential
    pv_cf = min(row["theo_pv"] * PV_CF_SCALE, 0.35)
    wind_cf = min(row["theo_wind"] * WIND_CF_SCALE, 0.45)
    cf = max(pv_cf, wind_cf)  # pick better resource

    # Annual production (assume 1 MW electrolyzer)
    annual_h2_kg = (1000 * cf * 8760) / ELEC_EFF

    # CAPEX after subsidy
    capex = ELECTRO_CAPEX * (1 - subsidy)
    crf = wacc * (1 + wacc) ** LIFETIME / ((1 + wacc) ** LIFETIME - 1)
    capex_ann = capex * crf

    # OPEX
    opex = capex * OPEX_RATIO

    # Infrastructure adder
    infra_dist = (row["grid_dist_km"] + row["road_dist_km"] +
                  row["waterbody_dist_km"]) / 3
    infra_cost = infra_dist * INFRA_COST_PER_KM

    # LCOH before delivery
    lcoh_prod = (capex_ann + opex) / annual_h2_kg * 1000  # $/kg

    # Delivered cost
    delivered = lcoh_prod + infra_cost + OCEAN_DIST_COST
    return round(delivered, 2)

def main():
    df = pd.read_csv(DATA_PATH)

    # Scenarios
    scenarios = {
        "Base (WACC=8%)": {"wacc": 0.08, "subsidy": 0.0},
        "De-risked (WACC=4%)": {"wacc": 0.04, "subsidy": 0.0},
        "High-interest (WACC=12%)": {"wacc": 0.12, "subsidy": 0.0},
        "Policy subsidy (30%)": {"wacc": 0.08, "subsidy": 0.30},
    }

    results = []
    for name, params in scenarios.items():
        df[name] = df.apply(
            lambda r: compute_lcoh(r, params["wacc"], params["subsidy"]), axis=1
        )
        results.append({
            "Scenario": name,
            "Mean LCOH": round(df[name].mean(), 2),
            "Min LCOH": round(df[name].min(), 2),
            "Max LCOH": round(df[name].max(), 2),
            "Sites < Europe": (df[name] < EUROPE_LCOH).sum(),
        })

    # Save results
    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUTPUT_DIR}/scenario_summary.csv", index=False)
    df.to_csv(f"{OUTPUT_DIR}/lcoh_results.csv", index=False)

    # ========== Figure 1: Geospatial map ==========
    africa = gpd.read_file(SHAPEFILE)
    africa = africa[africa["CONTINENT"] == "Africa"]

    fig, ax = plt.subplots(figsize=(10, 8))
    africa.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)

    # Color by base LCOH
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    scatter = ax.scatter(
        df["lon"], df["lat"],
        c=df["Base (WACC=8%)"], cmap="RdYlGn_r", s=80, edgecolors="black"
    )
    plt.colorbar(scatter, ax=ax, label="LCOH ($/kg)")
    ax.set_title("African Green H₂ Delivered Cost (Base Scenario, 2030)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/figure1_map.png", dpi=300)
    plt.close()

    # ========== Figure 2: Scenario comparison boxplot ==========
    lcoh_cols = list(scenarios.keys())
    df_melt = df[lcoh_cols].melt(var_name="Scenario", value_name="LCOH")
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_melt, x="Scenario", y="LCOH", palette="Set2")
    plt.axhline(EUROPE_LCOH, color="red", linestyle="--", label="Europe reference")
    plt.title("Delivered LCOH by Financing & Policy Scenario")
    plt.ylabel("LCOH ($/kg)")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/figure2_boxplot.png", dpi=300)
    plt.close()

    # ========== Figure 3: Min-cost sites bar ==========
    min_sites = df.nsmallest(5, "Base (WACC=8%)")[["hex_id", "Base (WACC=8%)"]]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=min_sites, x="hex_id", y="Base (WACC=8%)", palette="viridis")
    plt.title("5 Least-Cost African Sites (Base Scenario)")
    plt.ylabel("LCOH ($/kg)")
    plt.xlabel("Site ID")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/figure3_least_cost.png", dpi=300)
    plt.close()

    print("Analysis complete. Results and figures saved.")
    print(summary)

if __name__ == "__main__":
    main()
