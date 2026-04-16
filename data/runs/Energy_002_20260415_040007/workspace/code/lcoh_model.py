#!/usr/bin/env python3
"""
Geospatial Levelized-Cost Model for African Green Hydrogen Delivered to Europe
via Ammonia Shipping and Reconversion — 2030 Projections

Revised version with corrected infrastructure cost model.
Infrastructure costs are amortized per-kg over a realistic plant scale.
"""

import pandas as pd
import numpy as np
import json
import os

# ============================================================
# 1. TECHNO-ECONOMIC PARAMETERS FOR 2030
# ============================================================

params_2030 = {
    # --- Renewable Generation ---
    "pv_capex": 550,           # $/kW installed (2030 projection)
    "pv_opex_frac": 0.02,      # OPEX as fraction of CAPEX
    "pv_lifetime": 25,
    "pv_capacity_factor_base": 0.25,  # base CF, scaled by theo_pv
    
    "wind_capex": 1050,        # $/kW installed (2030 projection)
    "wind_opex_frac": 0.02,
    "wind_lifetime": 25,
    "wind_capacity_factor_base": 0.35,  # base CF, scaled by theo_wind
    
    # --- Electrolysis ---
    "electrolyzer_capex": 450,      # $/kW (2030 projection for PEM)
    "electrolyzer_opex_frac": 0.02, # OPEX as fraction of CAPEX
    "electrolyzer_efficiency": 0.70, # LHV efficiency
    "electrolyzer_lifetime": 20,     # years
    "electrolyzer_stack_replacement_frac": 0.15,  # fraction of CAPEX for stack replacement at mid-life
    
    # --- Hydrogen Storage (compressed gas) ---
    "h2_storage_capex": 500,        # $/kg H2 stored
    "h2_storage_opex_frac": 0.01,
    "h2_storage_hours": 8,          # hours of production stored
    
    # --- Ammonia Synthesis (Haber-Bosch) ---
    "nh3_synthesis_capex": 900,     # $/tNH3/yr capacity (2030 projection, large scale)
    "nh3_synthesis_opex_frac": 0.03,
    "nh3_synthesis_electricity": 0.35,  # kWh_el/kg_NH3
    "nh3_synthesis_h2_consumption": 0.197,  # kg_H2/kg_NH3
    "nh3_synthesis_lifetime": 25,
    "nh3_synthesis_efficiency": 0.88,  # overall H2-to-NH3 conversion efficiency
    
    # --- Ammonia Storage & Port Infrastructure ---
    "nh3_storage_capex": 150,       # $/tNH3 storage capacity
    "nh3_port_capex": 30,           # $/tNH3/yr throughput
    
    # --- Shipping (Ammonia Carrier) ---
    "shipping_distance_base": 10500,  # km, Namibia coast to Rotterdam via sea
    "shipping_cost_per_ton_km": 0.004,  # $/tNH3/km (2030 large-scale ammonia carrier)
    "shipping_boiloff": 0.015,         # 1.5% boiloff during shipping
    "shipping_fixed": 20,              # $/tNH3 fixed port/handling costs
    
    # --- Ammonia Cracking (Reconversion to H2) ---
    "nh3_cracking_capex": 1200,     # $/tNH3/yr capacity
    "nh3_cracking_opex_frac": 0.03,
    "nh3_cracking_heat": 4.2,       # kWh_th/kg_H2
    "nh3_cracking_electricity": 0.3, # kWh_el/kg_H2
    "nh3_cracking_efficiency": 0.86, # H2 recovery efficiency
    "nh3_cracking_lifetime": 25,
    "nh3_cracking_heat_cost": 0.03,  # $/kWh_th
    
    # --- Water ---
    "water_desalination_cost": 2.0,  # $/m3 desalinated water
    "water_consumption_electrolysis": 9,  # L/kg_H2
    
    # --- Infrastructure Connection ---
    "grid_connection_cost_per_km": 15000,  # $/km
    "road_connection_cost_per_km": 30000,  # $/km  
    "pipeline_cost_per_km": 300000,        # $/km for NH3 pipeline to coast
    
    # --- Plant Scale ---
    "plant_capacity_t_h2_yr": 100000,  # 100 kt H2/yr plant
    
    # --- Financial ---
    "wacc_base": 0.08,            # 8% base WACC for African projects
    "wacc_derisked": 0.05,        # 5% WACC with de-risking
    "wacc_high": 0.12,            # 12% WACC high-risk
    "wacc_europe": 0.05,          # 5% WACC for European projects
    
    # --- European H2 Production Reference ---
    "eu_pv_capacity_factor": 0.14,
    "eu_wind_capacity_factor": 0.28,
    "eu_pv_capex": 600,
    "eu_wind_capex": 1200,
    "eu_electrolyzer_capex": 500,
    
    # --- General ---
    "plant_lifetime": 25,
    "h2_lhv": 33.33,             # kWh/kg H2 (lower heating value)
    "annual_hours": 8760,
    "degradation": 0.005,
}

# ============================================================
# 2. LCOH CALCULATION ENGINE
# ============================================================

def annuity_factor(wacc, lifetime):
    """Calculate capital recovery factor (annuity factor)."""
    if wacc == 0:
        return 1.0 / lifetime
    return (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def calc_renewable_lcoe(capex, opex_frac, capacity_factor, wacc, lifetime, degradation=0.005):
    """Calculate levelized cost of electricity from renewable source.
    Returns LCOE in $/kWh.
    """
    opex = capex * opex_frac
    crf = annuity_factor(wacc, lifetime)
    avg_cf = capacity_factor * (1 - degradation * (lifetime - 1) / 2)
    annual_energy = avg_cf * 8760
    if annual_energy == 0:
        return float('inf')
    annualized_capex = capex * crf
    annual_cost = annualized_capex + opex
    lcoe = annual_cost / annual_energy
    return lcoe

def calc_production_lcoh(pv_lcoe, wind_lcoe, pv_cf, wind_cf, params, wacc):
    """Calculate production LCOH (at plant gate) given renewable electricity costs.
    Returns dict with cost components in $/kg_H2.
    """
    h2_lhv = params["h2_lhv"]
    efficiency = params["electrolyzer_efficiency"]
    
    # Optimal RE mix
    if wind_lcoe < pv_lcoe * 1.3:
        wind_fraction = 0.55
        solar_fraction = 0.45
    else:
        wind_fraction = 0.30
        solar_fraction = 0.70
    
    blended_lcoe = wind_fraction * wind_lcoe + solar_fraction * pv_lcoe
    blended_cf = wind_fraction * wind_cf + solar_fraction * pv_cf
    
    electrolyzer_cf = min(blended_cf * 0.95, 0.60)
    
    # Electricity cost per kg H2
    elec_per_kg = h2_lhv / efficiency
    electricity_cost_per_kg = elec_per_kg * blended_lcoe
    
    # Electrolyzer CAPEX per kg H2
    capex = params["electrolyzer_capex"]
    opex = capex * params["electrolyzer_opex_frac"]
    lifetime = params["electrolyzer_lifetime"]
    crf = annuity_factor(wacc, lifetime)
    
    annual_h2_per_kw = electrolyzer_cf * 8760 * efficiency / h2_lhv
    annualized_capex_per_kw = capex * crf
    capex_per_kg = annualized_capex_per_kw / annual_h2_per_kw
    opex_per_kg = opex / annual_h2_per_kw
    
    # Stack replacement
    stack_cost = capex * params["electrolyzer_stack_replacement_frac"]
    mid_life = lifetime // 2
    pv_stack = stack_cost / (1 + wacc)**mid_life
    stack_per_kg = pv_stack * annuity_factor(wacc, params["plant_lifetime"]) / annual_h2_per_kw
    
    cost_electrolysis = electricity_cost_per_kg + capex_per_kg + opex_per_kg + stack_per_kg
    
    # H2 Storage
    storage_capex = params["h2_storage_capex"]
    storage_hours = params["h2_storage_hours"]
    storage_per_kw = storage_hours * efficiency / h2_lhv
    storage_capex_per_kw = storage_per_kw * storage_capex
    annualized_storage = storage_capex_per_kw * annuity_factor(wacc, params["plant_lifetime"])
    storage_opex_per_kw = storage_capex_per_kw * params["h2_storage_opex_frac"]
    cost_storage = (annualized_storage + storage_opex_per_kw) / annual_h2_per_kw
    
    # Water
    water_per_kg = params["water_consumption_electrolysis"] / 1000
    cost_water = water_per_kg * params["water_desalination_cost"]
    
    cost_production = cost_electrolysis + cost_storage + cost_water
    
    return {
        "blended_lcoe": blended_lcoe,
        "blended_cf": blended_cf,
        "electrolyzer_cf": electrolyzer_cf,
        "wind_fraction": wind_fraction,
        "solar_fraction": solar_fraction,
        "elec_per_kg": elec_per_kg,
        "electricity_cost_per_kg": electricity_cost_per_kg,
        "capex_per_kg": capex_per_kg,
        "opex_per_kg": opex_per_kg,
        "stack_per_kg": stack_per_kg,
        "cost_electrolysis": cost_electrolysis,
        "cost_storage": cost_storage,
        "cost_water": cost_water,
        "cost_production": cost_production,
    }

def calc_ammonia_conversion_cost(cost_production, params, wacc):
    """Calculate additional cost of converting H2 to NH3.
    Returns cost in $/kg_H2.
    """
    capex = params["nh3_synthesis_capex"]
    opex_frac = params["nh3_synthesis_opex_frac"]
    electricity = params["nh3_synthesis_electricity"]
    h2_consumption = params["nh3_synthesis_h2_consumption"]
    lifetime = params["nh3_synthesis_lifetime"]
    efficiency = params["nh3_synthesis_efficiency"]
    
    crf = annuity_factor(wacc, lifetime)
    utilization = 0.90
    
    annualized_capex = capex * crf
    opex = capex * opex_frac
    
    # Electricity for synthesis
    h2_lhv = params["h2_lhv"]
    elec_cost_approx = cost_production * params["electrolyzer_efficiency"] / h2_lhv * 0.7
    electricity_cost_per_kg_nh3 = electricity * max(elec_cost_approx, 0.03) / 1000
    
    # CAPEX+OPEX per kg H2 (through NH3 pathway)
    nh3_capex_opex_per_kg_h2 = ((annualized_capex + opex) / (utilization * 1000) + electricity_cost_per_kg_nh3) / h2_consumption
    
    # H2 lost in conversion
    h2_loss_cost = cost_production * (1.0 / efficiency - 1.0)
    
    return nh3_capex_opex_per_kg_h2 + h2_loss_cost

def calc_shipping_cost(params, ocean_dist_km):
    """Calculate ammonia shipping cost per kg H2 delivered."""
    distance = params["shipping_distance_base"]
    h2_consumption = params["nh3_synthesis_h2_consumption"]
    shipping_per_tnh3 = distance * params["shipping_cost_per_ton_km"] + params["shipping_fixed"]
    boiloff = params["shipping_boiloff"]
    cost_per_kg_h2 = shipping_per_tnh3 / (h2_consumption * 1000) * (1 + boiloff)
    return cost_per_kg_h2

def calc_cracking_cost(params, wacc_eu):
    """Calculate ammonia cracking cost per kg H2 at European destination."""
    capex = params["nh3_cracking_capex"]
    opex_frac = params["nh3_cracking_opex_frac"]
    heat = params["nh3_cracking_heat"]
    electricity = params["nh3_cracking_electricity"]
    efficiency = params["nh3_cracking_efficiency"]
    lifetime = params["nh3_cracking_lifetime"]
    h2_consumption = params["nh3_synthesis_h2_consumption"]
    
    crf = annuity_factor(wacc_eu, lifetime)
    utilization = 0.90
    
    h2_per_tnh3 = h2_consumption * efficiency * 1000
    annualized_capex = capex * crf
    opex = capex * opex_frac
    
    capex_per_kg_h2 = annualized_capex / (utilization * h2_per_tnh3)
    opex_per_kg_h2 = opex / (utilization * h2_per_tnh3)
    
    heat_cost = heat * params["nh3_cracking_heat_cost"]
    elec_cost = electricity * 0.06
    
    return capex_per_kg_h2 + opex_per_kg_h2 + heat_cost + elec_cost

def calc_infrastructure_cost(row, params, wacc):
    """Calculate infrastructure connection costs per kg H2."""
    crf = annuity_factor(wacc, params["plant_lifetime"])
    plant_capacity = params["plant_capacity_t_h2_yr"] * 1000  # kg H2/yr
    
    grid_cost = row["grid_dist_km"] * params["grid_connection_cost_per_km"]
    road_cost = row["road_dist_km"] * params["road_connection_cost_per_km"]
    
    if row["ocean_dist_km"] > 20:
        pipeline_cost = row["ocean_dist_km"] * params["pipeline_cost_per_km"]
    else:
        pipeline_cost = row["ocean_dist_km"] * params["pipeline_cost_per_km"] * 0.5
    
    total_infra = grid_cost + road_cost + pipeline_cost
    annual_cost = total_infra * crf
    cost_per_kg = annual_cost / plant_capacity
    
    return cost_per_kg

def calc_site_lcoh(row, params, wacc, scenario_name="base"):
    """Calculate full delivered LCOH for a single site."""
    pv_cf = row["theo_pv"] * params["pv_capacity_factor_base"] / 0.5
    wind_cf = row["theo_wind"] * params["wind_capacity_factor_base"] / 0.5
    pv_cf = min(pv_cf, 0.35)
    wind_cf = min(wind_cf, 0.50)
    
    pv_lcoe = calc_renewable_lcoe(params["pv_capex"], params["pv_opex_frac"], pv_cf, wacc, params["pv_lifetime"])
    wind_lcoe = calc_renewable_lcoe(params["wind_capex"], params["wind_opex_frac"], wind_cf, wacc, params["wind_lifetime"])
    
    prod = calc_production_lcoh(pv_lcoe, wind_lcoe, pv_cf, wind_cf, params, wacc)
    
    cost_infrastructure = calc_infrastructure_cost(row, params, wacc)
    cost_production_total = prod["cost_production"] + cost_infrastructure
    
    cost_ammonia_conversion = calc_ammonia_conversion_cost(cost_production_total, params, wacc)
    cost_shipping = calc_shipping_cost(params, row["ocean_dist_km"])
    cost_cracking = calc_cracking_cost(params, params["wacc_europe"])
    
    cost_delivered = cost_production_total + cost_ammonia_conversion + cost_shipping + cost_cracking
    
    return {
        "hex_id": row["hex_id"],
        "lat": row["lat"],
        "lon": row["lon"],
        "scenario": scenario_name,
        "wacc": wacc,
        "pv_cf": pv_cf,
        "wind_cf": wind_cf,
        "pv_lcoe": pv_lcoe,
        "wind_lcoe": wind_lcoe,
        "blended_lcoe": prod["blended_lcoe"],
        "blended_cf": prod["blended_cf"],
        "electrolyzer_cf": prod["electrolyzer_cf"],
        "wind_fraction": prod["wind_fraction"],
        "solar_fraction": prod["solar_fraction"],
        "theo_pv": row["theo_pv"],
        "theo_wind": row["theo_wind"],
        "ocean_dist_km": row["ocean_dist_km"],
        "grid_dist_km": row["grid_dist_km"],
        "road_dist_km": row["road_dist_km"],
        "cost_electricity": prod["electricity_cost_per_kg"],
        "cost_electrolyzer_capex": prod["capex_per_kg"],
        "cost_electrolyzer_opex": prod["opex_per_kg"],
        "cost_electrolysis": prod["cost_electrolysis"],
        "cost_storage": prod["cost_storage"],
        "cost_water": prod["cost_water"],
        "cost_infrastructure": cost_infrastructure,
        "cost_production": cost_production_total,
        "cost_ammonia_conversion": cost_ammonia_conversion,
        "cost_shipping": cost_shipping,
        "cost_cracking": cost_cracking,
        "cost_delivered": cost_delivered,
    }

def calc_european_lcoh(params, wacc_eu=None):
    """Calculate reference LCOH for European green hydrogen production."""
    if wacc_eu is None:
        wacc_eu = params["wacc_europe"]
    
    pv_lcoe = calc_renewable_lcoe(params["eu_pv_capex"], 0.02, 
                                   params["eu_pv_capacity_factor"], wacc_eu, params["pv_lifetime"])
    wind_lcoe = calc_renewable_lcoe(params["eu_wind_capex"], 0.02,
                                     params["eu_wind_capacity_factor"], wacc_eu, params["wind_lifetime"])
    
    eu_params = params.copy()
    eu_params["electrolyzer_capex"] = params["eu_electrolyzer_capex"]
    
    prod = calc_production_lcoh(pv_lcoe, wind_lcoe, 
                                params["eu_pv_capacity_factor"], 
                                params["eu_wind_capacity_factor"],
                                eu_params, wacc_eu)
    
    water_cost = 0.01
    cost_production = prod["cost_production"] + water_cost
    
    return {
        "wacc": wacc_eu,
        "pv_lcoe": pv_lcoe,
        "wind_lcoe": wind_lcoe,
        "blended_lcoe": prod["blended_lcoe"],
        "cost_electrolysis": prod["cost_electrolysis"],
        "cost_storage": prod["cost_storage"],
        "cost_water": water_cost,
        "cost_production": cost_production,
        "cost_delivered": cost_production,
    }

def main():
    df = pd.read_csv("data/hex_final_NA_min.csv")
    print(f"Loaded {len(df)} sites")
    
    scenarios = {
        "base": params_2030["wacc_base"],
        "derisked": params_2030["wacc_derisked"],
        "high_risk": params_2030["wacc_high"],
    }
    
    results = []
    for scenario_name, wacc in scenarios.items():
        for _, row in df.iterrows():
            result = calc_site_lcoh(row, params_2030, wacc, scenario_name)
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    eu_results = {}
    for eu_scenario, eu_wacc in [("eu_base", 0.05), ("eu_low", 0.04), ("eu_high", 0.06)]:
        eu_results[eu_scenario] = calc_european_lcoh(params_2030, eu_wacc)
    
    # WACC sensitivity
    wacc_range = np.arange(0.03, 0.16, 0.01)
    best_site_base = results_df[results_df["scenario"]=="base"]
    best_site_idx = best_site_base["cost_delivered"].idxmin()
    best_hex_id = results_df.loc[best_site_idx, "hex_id"]
    best_site_num = int(best_hex_id.replace("hex_", ""))
    best_site_row = df.iloc[best_site_num]
    
    wacc_sensitivity = []
    for w in wacc_range:
        result = calc_site_lcoh(best_site_row, params_2030, w, f"wacc_{w:.0%}")
        wacc_sensitivity.append({
            "wacc": w,
            "cost_delivered": result["cost_delivered"],
            "cost_production": result["cost_production"],
            "cost_ammonia_conversion": result["cost_ammonia_conversion"],
            "cost_shipping": result["cost_shipping"],
            "cost_cracking": result["cost_cracking"],
        })
    wacc_sensitivity_df = pd.DataFrame(wacc_sensitivity)
    
    # Save results
    results_df.to_csv("outputs/lcoh_results.csv", index=False)
    
    eu_df = pd.DataFrame(eu_results).T
    eu_df.to_csv("outputs/european_lcoh.csv")
    
    wacc_sensitivity_df.to_csv("outputs/wacc_sensitivity.csv", index=False)
    
    cost_cols = ["cost_electrolysis", "cost_storage", "cost_water", "cost_infrastructure",
                 "cost_ammonia_conversion", "cost_shipping", "cost_cracking", "cost_delivered"]
    breakdown = results_df.groupby("scenario")[cost_cols].agg(["mean", "min", "max"]).round(3)
    breakdown.to_csv("outputs/cost_breakdown.csv")
    
    scenario_summary = results_df.groupby("scenario")[["cost_production", "cost_delivered"]].agg(["mean", "min", "max"]).round(3)
    scenario_summary.to_csv("outputs/scenario_comparison.csv")
    
    key_metrics = {
        "base_min_delivered": float(results_df[results_df["scenario"]=="base"]["cost_delivered"].min()),
        "base_mean_delivered": float(results_df[results_df["scenario"]=="base"]["cost_delivered"].mean()),
        "base_max_delivered": float(results_df[results_df["scenario"]=="base"]["cost_delivered"].max()),
        "derisked_min_delivered": float(results_df[results_df["scenario"]=="derisked"]["cost_delivered"].min()),
        "derisked_mean_delivered": float(results_df[results_df["scenario"]=="derisked"]["cost_delivered"].mean()),
        "high_risk_min_delivered": float(results_df[results_df["scenario"]=="high_risk"]["cost_delivered"].min()),
        "high_risk_mean_delivered": float(results_df[results_df["scenario"]=="high_risk"]["cost_delivered"].mean()),
        "eu_base_production": float(eu_results["eu_base"]["cost_production"]),
        "eu_low_production": float(eu_results["eu_low"]["cost_production"]),
        "eu_high_production": float(eu_results["eu_high"]["cost_production"]),
        "best_site_base": best_hex_id,
        "n_sites": len(df),
    }
    with open("outputs/key_metrics.json", "w") as f:
        json.dump(key_metrics, f, indent=2)
    
    print("\n=== KEY RESULTS ===")
    for s in ["base", "derisked", "high_risk"]:
        s_data = results_df[results_df["scenario"]==s]
        wacc_key = f"wacc_{s}" if s != "high_risk" else "wacc_high"
        print(f"\n{s.upper()} scenario ({params_2030[wacc_key]:.0%} WACC):")
        print(f"  Min delivered LCOH: ${s_data['cost_delivered'].min():.2f}/kg")
        print(f"  Mean delivered LCOH: ${s_data['cost_delivered'].mean():.2f}/kg")
        print(f"  Max delivered LCOH: ${s_data['cost_delivered'].max():.2f}/kg")
        print(f"  Min production LCOH: ${s_data['cost_production'].min():.2f}/kg")
    
    print(f"\nEuropean domestic LCOH (5% WACC): ${eu_results['eu_base']['cost_production']:.2f}/kg")
    print(f"Best site (base): {best_hex_id}")
    
    eu_lcoh = eu_results["eu_base"]["cost_production"]
    for s in ["base", "derisked", "high_risk"]:
        s_data = results_df[results_df["scenario"]==s]
        competitive = (s_data["cost_delivered"] < eu_lcoh).sum()
        print(f"\n{s.upper()}: {competitive}/{len(s_data)} sites competitive with European domestic H2")
    
    return results_df, eu_results, wacc_sensitivity_df

if __name__ == "__main__":
    results_df, eu_results, wacc_sensitivity_df = main()
