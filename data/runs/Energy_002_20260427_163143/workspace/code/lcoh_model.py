"""
Africa -> Europe green hydrogen delivered-cost model (2030).

Approach (transparent, GeoH2 / Mueller-Kenya style component LCOH):
  For each African hexagon i and scenario s:
    1. Convert theoretical PV / wind potential -> annual capacity factors.
    2. Compute LCOE for PV-only, wind-only, hybrid (cheapest mix) using
       annuitised CAPEX (CRF) + OPEX over annual generation.
    3. LCOH_at_plant = electrolyser annuitised CAPEX/(8760 * CF_eff) +
                        electrolyser OPEX + electricity_cost / efficiency
                        + water + storage premium.
    4. LCOA_synth   = LCOH_to_NH3 + ammonia synthesis CAPEX/OPEX + electricity for synth.
    5. Inland transport from hex to nearest African export port (use ocean_dist_km
       as proxy distance to coast; trucking cost per km).
    6. Shipping NH3 by sea: average port-to-EU distance (Rotterdam) ~ great-circle
       from hex centroid + 6000 km buffer for routing + canal -- we use a
       representative shipping distance derived from the centroid latitude of
       Africa export ports (~ 7500 km).
    7. NH3 cracking back to H2 in Europe: CAPEX + OPEX + heat penalty (~13% LHV).
  Delivered LCOH (eur/kg H2) reported under baseline / de-risked / EU-ref /
  EU-rising-IR scenarios.

Parameters drawn from Halloran et al. 2024 (GeoH2, Namibia case),
Mueller et al. 2023 (Kenya), Steffen 2020 (cost of capital), Schmidt et al. 2019
(rising IR scenarios).
"""

from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data"
OUT  = WORK / "outputs"
OUT.mkdir(exist_ok=True)

# ----------------------------- Parameters -----------------------------

# Techno-economic parameters (2030, EUR, real terms unless noted).
# Grounded in GeoH2 Namibia case (Halloran et al. 2024) and Mueller-Kenya 2023.
TE = {
    # Solar PV
    "pv_capex_eur_per_kw":       650.0,   # 2030 (IRENA / GeoH2 baseline ~1470 today, 600/kW future per Mueller)
    "pv_opex_pct":               0.02,    # % CAPEX/yr
    "pv_lifetime":               25,
    # Wind onshore
    "wind_capex_eur_per_kw":     1100.0,  # 2030
    "wind_opex_pct":             0.03,
    "wind_lifetime":             25,
    # Electrolyser (alkaline/PEM blended)
    "ely_capex_eur_per_kw":      600.0,   # 2030 (Mueller 2023 future)
    "ely_opex_pct":              0.04,
    "ely_lifetime":              20,
    "ely_efficiency_LHV":        0.70,    # 2030 (Mueller 2023 future). 47.6 kWh/kg LHV @ 70%.
    "h2_LHV_kwh_per_kg":         33.33,
    # Water (desalinated, transport included)
    "water_eur_per_kg_h2":       0.05,
    # Compressed-storage premium (small buffer on plant)
    "storage_eur_per_kg_h2":     0.10,
    # Ammonia synthesis
    "nh3_synth_capex_eur_per_t_nh3_yr": 1100.0,  # GeoH2 / IEA ammonia roadmap
    "nh3_synth_opex_pct":        0.015,
    "nh3_synth_lifetime":        25,
    "nh3_synth_kwh_e_per_kg_h2": 2.81,    # 2.809 kWh/kgH2 (GeoH2)
    "nh3_h2_mass_ratio":         0.178,   # kg H2 in 1 kg NH3 (NH3 is 17.65% H2 by mass)
    # NH3 sea shipping (very large gas carrier, NH3-ready)
    "nh3_ship_eur_per_t_per_km": 0.000018,  # ~0.018 €/t/km for long-haul NH3 (IEA, Hampp et al.)
    "ship_distance_km":          7500.0,    # representative SADC -> Rotterdam via Cape
    "boil_off_pct":              0.04,      # 4% lost in shipping (NH3 long voyage round-trip)
    # NH3 cracking (reconversion to H2 in EU)
    "nh3_crack_capex_eur_per_t_h2_yr": 1900.0,  # GeoH2: 17.26 kWh/g/h coefficient -> we approximate
    "nh3_crack_opex_pct":        0.02,
    "nh3_crack_lifetime":        25,
    "nh3_crack_h2_yield":        0.87,    # 13% energy penalty (heat from cracked H2)
    # Inland trucking from plant to export port (NH3 truck)
    "truck_eur_per_t_per_km":    0.10,    # NH3 trucking proxy
    # Reference annual H2 demand at European user (only used for amortising fixed transport infra)
    "demand_h2_t_per_yr":        100000.0, # 100 kt H2/yr (≈ ~ Lürssen-scale industrial cluster)
}

# Scenario WACCs (real, after-tax). Anchored to Steffen 2020 (developing countries
# ~7-12% for RE, industrialised ~3-6%) and Schmidt et al. 2019 (EU rising-IR).
SCENARIOS = {
    "S1_AFR_BASELINE": {
        "label": "Africa baseline (high WACC, no de-risking)",
        "wacc_re": 0.10, "wacc_ely": 0.10, "wacc_infra": 0.10,
        "region": "Africa",
    },
    "S2_AFR_MODERATE": {
        "label": "Africa moderate (some de-risking)",
        "wacc_re": 0.08, "wacc_ely": 0.08, "wacc_infra": 0.08,
        "region": "Africa",
    },
    "S3_AFR_DERISKED": {
        "label": "Africa de-risked (concessional/blended)",
        "wacc_re": 0.06, "wacc_ely": 0.06, "wacc_infra": 0.06,
        "region": "Africa",
    },
    "S4_EU_LOW_IR": {
        "label": "Europe green H2 reference (low IR)",
        "wacc_re": 0.04, "wacc_ely": 0.04, "wacc_infra": 0.04,
        "region": "Europe",
    },
    "S5_EU_RISING_IR": {
        "label": "Europe green H2 reference (rising IR, Schmidt 2019)",
        "wacc_re": 0.07, "wacc_ely": 0.07, "wacc_infra": 0.07,
        "region": "Europe",
    },
}

# European reference (no shipping, slightly higher CAPEX, lower CF)
EU_REF = {
    "pv_cf":     0.13,  # Northern/central Europe utility PV
    "wind_cf":   0.30,  # Onshore Europe avg
    "pv_capex_eur_per_kw":   700.0,
    "wind_capex_eur_per_kw": 1300.0,
}

# ----------------------------- Helpers -----------------------------

def crf(wacc: float, lifetime: int) -> float:
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)

def cf_from_theo_pv(theo_pv: float) -> float:
    """Map 'theoretical PV potential' index in [0,1] to capacity factor.
    Excellent SADC sites reach ~26% CF; poor ones ~16%. Linear map."""
    return 0.16 + 0.10 * theo_pv

def cf_from_theo_wind(theo_wind: float) -> float:
    """Map 'theoretical wind potential' index to onshore CF.
    Range typically 0.10 (poor) to 0.45 (excellent SADC coast)."""
    return 0.10 + 0.35 * theo_wind

def lcoe_re(capex_kw: float, opex_pct: float, lifetime: int, wacc: float, cf: float) -> float:
    """Return €/kWh."""
    annuity = capex_kw * crf(wacc, lifetime)
    annual_opex = capex_kw * opex_pct
    annual_gen_kwh = 8760.0 * cf
    if annual_gen_kwh <= 0:
        return float("inf")
    return (annuity + annual_opex) / annual_gen_kwh

# ----------------------------- Core LCOH -----------------------------

def hex_lcoh_at_plant(row: pd.Series, sc: dict) -> dict:
    """Return component costs (€/kg H2) at the plant gate (no shipping)."""
    cf_pv  = cf_from_theo_pv(row["theo_pv"])
    cf_wd  = cf_from_theo_wind(row["theo_wind"])
    # LCOEs
    lcoe_pv = lcoe_re(TE["pv_capex_eur_per_kw"], TE["pv_opex_pct"],
                      TE["pv_lifetime"], sc["wacc_re"], cf_pv)
    lcoe_wd = lcoe_re(TE["wind_capex_eur_per_kw"], TE["wind_opex_pct"],
                      TE["wind_lifetime"], sc["wacc_re"], cf_wd)
    # Hybrid: weighted blend by inverse LCOE (favours cheaper resource);
    # cap PV share at 70% to acknowledge diurnal limit without storage co-opt.
    inv = np.array([1/lcoe_pv, 1/lcoe_wd])
    w   = inv / inv.sum()
    if w[0] > 0.70:
        w = np.array([0.70, 0.30])
    # Effective CF (electrolyser sees combined plant)
    cf_eff = w[0]*cf_pv + w[1]*cf_wd
    lcoe_blend = w[0]*lcoe_pv + w[1]*lcoe_wd
    # Electrolyser
    ely_kwh_per_kg = TE["h2_LHV_kwh_per_kg"] / TE["ely_efficiency_LHV"]
    ely_annuity_per_kw = TE["ely_capex_eur_per_kw"] * crf(sc["wacc_ely"], TE["ely_lifetime"])
    ely_opex_per_kw    = TE["ely_capex_eur_per_kw"] * TE["ely_opex_pct"]
    ely_annual_kwh     = 8760.0 * cf_eff
    ely_capex_per_kg   = (ely_annuity_per_kw + ely_opex_per_kw) / ely_annual_kwh * ely_kwh_per_kg
    elec_per_kg        = lcoe_blend * ely_kwh_per_kg
    water_per_kg       = TE["water_eur_per_kg_h2"]
    storage_per_kg     = TE["storage_eur_per_kg_h2"]
    return {
        "cf_pv": cf_pv, "cf_wd": cf_wd, "cf_eff": cf_eff,
        "lcoe_pv_eur_kwh": lcoe_pv, "lcoe_wd_eur_kwh": lcoe_wd, "lcoe_blend_eur_kwh": lcoe_blend,
        "w_pv": w[0], "w_wd": w[1],
        "c_elec_per_kg":     elec_per_kg,
        "c_ely_capex_per_kg": ely_capex_per_kg,
        "c_water_per_kg":    water_per_kg,
        "c_storage_per_kg":  storage_per_kg,
        "lcoh_plant_per_kg": elec_per_kg + ely_capex_per_kg + water_per_kg + storage_per_kg,
    }

def add_export_chain(row: pd.Series, sc: dict, plant: dict) -> dict:
    """Add NH3 synthesis, inland transport, ocean shipping, EU cracking."""
    # NH3 synthesis CAPEX/OPEX expressed per kg H2 (assuming continuous operation
    # buffered, sized to plant H2 throughput; use plant capacity factor 0.85 typical
    # for synthesis loop).
    nh3_synth_cf = 0.85
    # Convert per-tNH3-yr CAPEX to per-kg-H2 cost: 1 kg H2 -> 1/0.178 kg NH3 = 5.62 kg NH3/kg H2
    kg_nh3_per_kg_h2 = 1.0 / TE["nh3_h2_mass_ratio"]
    annual_h2_per_kw_synth = nh3_synth_cf * 8760.0 / 1000.0  # not quite right, instead:
    # simpler: capex per kg-H2 = capex_per_t_NH3_yr * kg_NH3 per kg_H2 / (1000 kg/t * cf normalization)
    nh3_synth_capex_per_kg_h2_yr = TE["nh3_synth_capex_eur_per_t_nh3_yr"] * kg_nh3_per_kg_h2 / 1000.0
    nh3_synth_annuity = nh3_synth_capex_per_kg_h2_yr * crf(sc["wacc_infra"], TE["nh3_synth_lifetime"])
    nh3_synth_opex    = nh3_synth_capex_per_kg_h2_yr * TE["nh3_synth_opex_pct"]
    # synthesis electricity
    elec_synth_per_kg = TE["nh3_synth_kwh_e_per_kg_h2"] * plant["lcoe_blend_eur_kwh"]
    c_synth_per_kg = nh3_synth_annuity + nh3_synth_opex + elec_synth_per_kg

    # Inland trucking from plant to nearest African port (proxy: ocean_dist_km)
    truck_dist_km = float(row["ocean_dist_km"])
    truck_per_kg_nh3 = TE["truck_eur_per_t_per_km"] * truck_dist_km / 1000.0
    truck_per_kg_h2  = truck_per_kg_nh3 * kg_nh3_per_kg_h2

    # Sea shipping
    ship_per_kg_nh3 = TE["nh3_ship_eur_per_t_per_km"] * TE["ship_distance_km"] * 1000.0 / 1000.0
    # (kg of NH3 -> per t -> per km; clarification below)
    # eur_per_t_per_km * km * (1 t / 1000 kg)  = eur per kg
    ship_per_kg_nh3 = TE["nh3_ship_eur_per_t_per_km"] * TE["ship_distance_km"]
    ship_per_kg_h2  = ship_per_kg_nh3 * kg_nh3_per_kg_h2
    # Boil-off / losses: scale up upstream cost by 1/(1-loss)
    loss_factor = 1.0 / (1.0 - TE["boil_off_pct"])

    # EU cracking
    crack_capex_per_kg_h2 = TE["nh3_crack_capex_eur_per_t_h2_yr"] / 1000.0  # eur/kg-H2/yr (per kg-H2 capacity)
    crack_annuity = crack_capex_per_kg_h2 * crf(sc["wacc_infra"], TE["nh3_crack_lifetime"])
    crack_opex    = crack_capex_per_kg_h2 * TE["nh3_crack_opex_pct"]
    # Adjust for cracking yield: need 1/yield kg of cracked-feed H2 to deliver 1 kg useful H2
    yield_factor = 1.0 / TE["nh3_crack_h2_yield"]
    c_crack_per_kg = (crack_annuity + crack_opex) * yield_factor

    return {
        "c_nh3_synth_per_kg":  c_synth_per_kg,
        "c_truck_per_kg":      truck_per_kg_h2,
        "c_ship_per_kg":       ship_per_kg_h2,
        "c_crack_per_kg":      c_crack_per_kg,
        "loss_factor":         loss_factor,
        "yield_factor":        yield_factor,
    }

def european_reference_lcoh(sc: dict) -> dict:
    """Compute EU green H2 LCOH using simple reference CFs and CAPEX."""
    # Hybrid PV/wind 50/50
    lcoe_pv = lcoe_re(EU_REF["pv_capex_eur_per_kw"], TE["pv_opex_pct"],
                      TE["pv_lifetime"], sc["wacc_re"], EU_REF["pv_cf"])
    lcoe_wd = lcoe_re(EU_REF["wind_capex_eur_per_kw"], TE["wind_opex_pct"],
                      TE["wind_lifetime"], sc["wacc_re"], EU_REF["wind_cf"])
    lcoe_blend = 0.4*lcoe_pv + 0.6*lcoe_wd
    cf_eff = 0.4*EU_REF["pv_cf"] + 0.6*EU_REF["wind_cf"]
    ely_kwh_per_kg = TE["h2_LHV_kwh_per_kg"] / TE["ely_efficiency_LHV"]
    ely_annuity_per_kw = TE["ely_capex_eur_per_kw"] * crf(sc["wacc_ely"], TE["ely_lifetime"])
    ely_opex_per_kw    = TE["ely_capex_eur_per_kw"] * TE["ely_opex_pct"]
    ely_annual_kwh     = 8760.0 * cf_eff
    ely_capex_per_kg   = (ely_annuity_per_kw + ely_opex_per_kw) / ely_annual_kwh * ely_kwh_per_kg
    elec_per_kg = lcoe_blend * ely_kwh_per_kg
    water = TE["water_eur_per_kg_h2"]; storage = TE["storage_eur_per_kg_h2"]
    return {
        "lcoe_pv": lcoe_pv, "lcoe_wd": lcoe_wd, "lcoe_blend": lcoe_blend,
        "cf_eff": cf_eff,
        "c_elec_per_kg": elec_per_kg, "c_ely_capex_per_kg": ely_capex_per_kg,
        "c_water_per_kg": water, "c_storage_per_kg": storage,
        "lcoh_eu_per_kg": elec_per_kg + ely_capex_per_kg + water + storage,
    }

# ----------------------------- Main run -----------------------------

def main():
    df = pd.read_csv(DATA / "hex_final_NA_min.csv")
    rows = []
    for sc_name, sc in SCENARIOS.items():
        for _, row in df.iterrows():
            plant = hex_lcoh_at_plant(row, sc)
            chain = add_export_chain(row, sc, plant)
            # Delivered LCOH including upstream losses
            upstream = (plant["lcoh_plant_per_kg"] + chain["c_nh3_synth_per_kg"] +
                        chain["c_truck_per_kg"]) * chain["loss_factor"]
            delivered = (upstream + chain["c_ship_per_kg"]) * chain["yield_factor"] + chain["c_crack_per_kg"]
            rec = {
                "scenario":  sc_name,
                "scenario_label": sc["label"],
                "region":    sc["region"],
                "hex_id":    row["hex_id"],
                "lat":       row["lat"], "lon": row["lon"],
                **plant,
                **chain,
                "lcoh_plant_per_kg":     plant["lcoh_plant_per_kg"],
                "lcoh_delivered_per_kg": delivered,
                "ocean_dist_km":         row["ocean_dist_km"],
                "road_dist_km":          row["road_dist_km"],
                "grid_dist_km":          row["grid_dist_km"],
                "waterbody_dist_km":     row["waterbody_dist_km"],
                "theo_pv":               row["theo_pv"],
                "theo_wind":             row["theo_wind"],
            }
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "lcoh_delivered_per_hex.csv", index=False)
    print(f"Wrote {len(out)} rows -> outputs/lcoh_delivered_per_hex.csv")

    # Scenario summary
    summary = out.groupby("scenario").agg(
        scenario_label=("scenario_label","first"),
        region=("region","first"),
        n_hex=("hex_id","count"),
        lcoh_delivered_min=("lcoh_delivered_per_kg","min"),
        lcoh_delivered_p10=("lcoh_delivered_per_kg",lambda x: x.quantile(0.10)),
        lcoh_delivered_median=("lcoh_delivered_per_kg","median"),
        lcoh_delivered_mean=("lcoh_delivered_per_kg","mean"),
        lcoh_delivered_p90=("lcoh_delivered_per_kg",lambda x: x.quantile(0.90)),
        lcoh_delivered_max=("lcoh_delivered_per_kg","max"),
        lcoh_plant_min=("lcoh_plant_per_kg","min"),
        lcoh_plant_median=("lcoh_plant_per_kg","median"),
        cf_eff_median=("cf_eff","median"),
    ).reset_index()
    # EU reference scenarios
    eu_rows = []
    for sc_name in ["S4_EU_LOW_IR", "S5_EU_RISING_IR"]:
        sc = SCENARIOS[sc_name]
        eu = european_reference_lcoh(sc)
        eu_rows.append({"scenario": sc_name, "lcoh_eu_per_kg": eu["lcoh_eu_per_kg"], **eu})
    eu_df = pd.DataFrame(eu_rows)
    eu_df.to_csv(OUT / "eu_reference_lcoh.csv", index=False)
    summary.to_csv(OUT / "scenario_summary.csv", index=False)
    print("Wrote outputs/scenario_summary.csv and outputs/eu_reference_lcoh.csv")

    # Save TE + scenarios for traceability
    with open(OUT / "model_parameters.json","w") as f:
        json.dump({"TE": TE, "SCENARIOS": SCENARIOS, "EU_REF": EU_REF}, f, indent=2)

if __name__ == "__main__":
    main()
