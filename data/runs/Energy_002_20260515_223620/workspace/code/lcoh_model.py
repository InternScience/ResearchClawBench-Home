#!/usr/bin/env python3
"""
Geospatial Levelized-Cost Model for African Green Hydrogen to Europe
via Ammonia Shipping and Reconversion

Based on GeoH2 methodology (Halloran et al., Müller et al.) and
extended with financing scenarios and interest-rate analysis.

Author: Autonomous Research Agent
Date: 2026-05-15
"""

import numpy as np
import pandas as pd
import json
import os

# ============================================================
# CONSTANTS AND TECHNO-ECONOMIC PARAMETERS
# ============================================================

# --- 2030 Projected Technology Costs (from literature review) ---
# PV CAPEX: declining from 1,470,000 EUR/MW (2020) at ~10%/yr learning → ~520,000 EUR/MW by 2030
PV_CAPEX_2030 = 520000  # EUR/MW
# Wind CAPEX: declining from 1,580,000 EUR/MW → ~900,000 EUR/MW by 2030
WIND_CAPEX_2030 = 900000  # EUR/MW
# Electrolyzer CAPEX: declining from 1,250,000 EUR/MW → ~500,000 EUR/MW by 2030
ELECTROLYZER_CAPEX_2030 = 500000  # EUR/MW
# Electrolyzer efficiency: 0.59 MWh_H2 / MWh_el → improving to 0.55 by 2030 (lower is better)
ELECTROLYZER_EFFICIENCY = 0.55  # MWh_el / MWh_H2 (i.e., 55 kWh per kg H2)

# O&M costs (% of CAPEX per year)
PV_OM = 0.015  # 1.5% of CAPEX
WIND_OM = 0.025  # 2.5% of CAPEX
ELECTROLYZER_OM = 0.03  # 3% of CAPEX

# Lifetimes (years)
PV_LIFETIME = 25
WIND_LIFETIME = 25
ELECTROLYZER_LIFETIME = 20

# --- Water Costs ---
# Water desalination + treatment cost, scaled by distance
WATER_BASE_COST = 0.05  # EUR/kg_H2 base water cost
WATER_DISTANCE_FACTOR = 0.0005  # EUR/kg_H2 per km from waterbody

# --- Grid Connection Cost ---
GRID_CONNECTION_COST_PER_KM = 50000  # EUR/km for transmission line
# Annualized grid cost added to LCOH

# --- Road Transport (to port) ---
# Ammonia trucking cost per tonne-km
AMMONIA_TRUCK_COST = 0.08  # EUR/tonne-km
# Ammonia contains ~0.177 tonnes H2 per tonne NH3
NH3_H2_RATIO = 0.177  # kg H2 per kg NH3 (H2 in NH3: 3*2.016/(14.007+3*1.008) = 6.048/17.031 = 0.355... wait, let me compute)
# Actually: NH3 molecular weight = 17.031, H2 molecular weight = 2.016
# 2 NH3 → N2 + 3 H2, so 34.062 g NH3 → 6.048 g H2
# Ratio: 6.048/34.062 = 0.1776 kg H2 per kg NH3
# But ammonia is transported as NH3, so 1 kg H2 requires 1/0.1776 = 5.63 kg NH3
KG_NH3_PER_KG_H2 = 1.0 / 0.1776  # ~5.63 kg NH3 per kg H2

# --- Ammonia Synthesis (Haber-Bosch) ---
# Electricity demand: 2.809 kWh/kg_H2
NH3_SYNTH_ELEC = 2.809  # kWh_el / kg_H2
# Capex: 0.75717 EUR/(kg_H2/year) → for 2030, ~0.50 EUR/(kg_H2/year)
NH3_SYNTH_CAPEX = 0.50  # EUR per kg_H2 annual capacity
NH3_SYNTH_OM = 0.015  # 1.5% of CAPEX
NH3_SYNTH_LIFETIME = 25

# --- Maritime Shipping ---
# Rotterdam port coordinates
ROTTERDAM_LAT = 51.909
ROTTERDAM_LON = 4.482

# Shipping cost: ~0.015-0.03 EUR/tonne-km for bulk ammonia
SHIPPING_COST_PER_TONNE_KM = 0.02  # EUR/tonne-km
# Average shipping speed: 14 knots = ~26 km/h
# Port handling: fixed cost per tonne
PORT_HANDLING_COST = 15.0  # EUR/tonne NH3

# --- Ammonia Cracking (at destination) ---
# Heat demand: 4.2 kWh/kg_H2
NH3_CRACK_HEAT = 4.2  # kWh_heat / kg_H2
# Heat cost (natural gas): ~0.03 EUR/kWh
HEAT_COST = 0.03  # EUR/kWh_heat
# Capex: 17,262,450 EUR / (kg_H2/hour)
# For a large-scale plant (100 kt_H2/year = ~11.4 t_H2/hr):
NH3_CRACK_CAPEX = 0.20  # EUR per kg_H2 annual capacity (scaled down for 2030)
NH3_CRACK_OM = 0.02  # 2% of CAPEX
NH3_CRACK_LIFETIME = 25

# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

# Financing scenarios (WACC for African projects)
SCENARIOS = {
    'high_risk': {
        'name': 'High Risk (Baseline)',
        'wacc': 0.12,
        'description': 'WACC 12% — typical for emerging-market RE projects without de-risking'
    },
    'moderate_derisk': {
        'name': 'Moderate De-risking',
        'wacc': 0.08,
        'description': 'WACC 8% — partial de-risking via guarantees/MDB involvement'
    },
    'strong_derisk': {
        'name': 'Strong De-risking',
        'wacc': 0.06,
        'description': 'WACC 6% — strong policy support, concessional finance'
    },
    'european_benchmark': {
        'name': 'European Benchmark',
        'wacc': 0.04,
        'description': 'WACC 4% — European green hydrogen production benchmark'
    }
}

# Interest rate sub-scenarios (based on Schmidt et al. 2019)
INTEREST_SCENARIOS = {
    'low_rates': {
        'name': 'Persistently Low Rates',
        'wacc_adjustment': 0.0,  # no change from baseline WACC
        'description': 'Interest rates remain at historically low levels'
    },
    'moderate_rise': {
        'name': 'Moderate Rate Rise',
        'wacc_adjustment': 0.02,  # +2pp WACC
        'description': 'Rates rise moderately (cf. Schmidt et al. moderate scenario)'
    },
    'extreme_rise': {
        'name': 'Extreme Rate Rise',
        'wacc_adjustment': 0.04,  # +4pp WACC
        'description': 'Rates rise sharply (cf. Schmidt et al. extreme scenario)'
    }
}

# European green hydrogen production cost benchmarks (EUR/kg H2)
# Based on literature: Europe 2030 LCOH ~3.0-5.5 EUR/kg
EUROPE_LCOH_BENCHMARKS = {
    'low': 3.0,    # best-case solar in Spain/Portugal
    'mid': 4.5,    # typical Germany/Netherlands
    'high': 5.5    # conservative estimate
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def crf(rate, years):
    """Capital recovery factor: annualize capital cost over lifetime."""
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate)**years / ((1 + rate)**years - 1)

def annualize_capex(capex, wacc, lifetime):
    """Annualize capital expenditure."""
    return capex * crf(wacc, lifetime)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def shipping_distance_km(ocean_lat, ocean_lon):
    """Approximate maritime shipping distance to Rotterdam.
    For African west coast, use direct route. For east coast, add Suez/Cape detour.
    Simplified: use great-circle * 1.35 factor for realistic shipping routes."""
    direct = haversine_distance(ocean_lat, ocean_lon, ROTTERDAM_LAT, ROTTERDAM_LON)
    # Shipping routes are ~30-40% longer than great circle
    # For East Africa, Suez adds more. Let's use a simplified model based on longitude.
    # West Africa (lon < 20): ~1.3x
    # Southern/East Africa: longer detour
    if ocean_lon < 15:
        factor = 1.30  # West Africa, fairly direct
    elif ocean_lon < 30:
        factor = 1.45  # Southern Africa
    else:
        factor = 1.75  # East Africa via Suez
    return direct * factor

# ============================================================
# MAIN LCOH CALCULATION
# ============================================================

def compute_lcoh(row, wacc, include_shipping=True):
    """
    Compute the delivered levelized cost of hydrogen (EUR/kg_H2)
    for a given hexagon site.
    
    Returns a dictionary with cost breakdown.
    """
    theo_pv = row['theo_pv']
    theo_wind = row['theo_wind']
    road_dist = row['road_dist_km']
    ocean_dist = row['ocean_dist_km']
    water_dist = row['waterbody_dist_km']
    grid_dist = row['grid_dist_km']
    lat = row['lat']
    lon = row['lon']
    
    # --- Step 1: Optimal RE mix ---
    # For each site, use the better resource between PV and wind,
    # or a mix. Simplification: use weighted blend favoring the stronger resource.
    # Capacity factor for each technology
    cf_pv = theo_pv
    cf_wind = theo_wind
    
    # Annual generation per MW installed (MWh)
    annual_gen_pv = cf_pv * 8760  # MWh/MW/year
    annual_gen_wind = cf_wind * 8760
    
    # Determine optimal share (simplified: proportional to capacity factor)
    total_cf = cf_pv + cf_wind
    if total_cf > 0:
        share_pv = cf_pv / total_cf
        share_wind = cf_wind / total_cf
    else:
        share_pv = share_wind = 0.5
    
    # LCOE for each technology (EUR/MWh)
    annualized_pv_capex = annualize_capex(PV_CAPEX_2030, wacc, PV_LIFETIME)
    lcoe_pv = (annualized_pv_capex + PV_CAPEX_2030 * PV_OM) / max(annual_gen_pv, 1)
    
    annualized_wind_capex = annualize_capex(WIND_CAPEX_2030, wacc, WIND_LIFETIME)
    lcoe_wind = (annualized_wind_capex + WIND_CAPEX_2030 * WIND_OM) / max(annual_gen_wind, 1)
    
    # Weighted LCOE
    lcoe_weighted = share_pv * lcoe_pv + share_wind * lcoe_wind
    
    # --- Step 2: Hydrogen Production ---
    # Electricity needed per kg H2
    elec_per_kg_h2 = ELECTROLYZER_EFFICIENCY * 1000  # kWh_el / kg_H2 (0.55 MWh/ MWh → 55 kWh/kg)
    # Wait - efficiency is 0.55 MWh_el / MWh_H2 means 1 MWh_H2 requires 0.55 MWh_el? No.
    # Let me re-read: "Efficiency 0.59 MWh_H2/MWh_el" means 1 MWh electricity → 0.59 MWh hydrogen
    # H2 energy content: 33.33 kWh/kg (LHV), so 1 MWh_H2 = 30 kg H2
    # 0.59 MWh_H2 / MWh_el means 1 MWh_el → 0.59 * 30 = 17.7 kg H2, so 56.5 kWh_el/kg_H2
    # For 2030 with improved efficiency: let's say 52 kWh_el / kg_H2
    ELEC_PER_KG_H2 = 52.0  # kWh_el per kg H2 (2030 PEM electrolyzer)
    
    # Electricity cost per kg H2
    elec_cost_per_kg = (lcoe_weighted / 1000) * ELEC_PER_KG_H2  # EUR/kg
    
    # Electrolyzer CAPEX contribution
    # Capacity: 1 MW electrolyzer → at 52 kWh/kg, hourly output = 1000/52 = 19.23 kg/h
    # Annual output at 100% CF = 19.23 * 8760 = 168,462 kg/year
    # But electrolyzer runs at RE CF
    combined_cf = share_pv * cf_pv + share_wind * cf_wind
    annual_h2_per_mw_elec = (1000 / ELEC_PER_KG_H2) * 8760 * combined_cf  # kg/year per MW
    
    annualized_elec_capex = annualize_capex(ELECTROLYZER_CAPEX_2030, wacc, ELECTROLYZER_LIFETIME)
    elec_capex_per_kg = annualized_elec_capex / max(annual_h2_per_mw_elec, 1)
    elec_om_per_kg = (ELECTROLYZER_CAPEX_2030 * ELECTROLYZER_OM) / max(annual_h2_per_mw_elec, 1)
    
    # --- Step 3: Water Cost ---
    water_cost = WATER_BASE_COST + WATER_DISTANCE_FACTOR * water_dist
    
    # --- Step 4: Grid Connection (optional, if far from grid) ---
    # Annualized grid connection cost per kg H2
    grid_connection_cost = 0.0
    if grid_dist > 0:
        grid_capex = grid_dist * GRID_CONNECTION_COST_PER_KM  # EUR
        annual_h2_total = annual_h2_per_mw_elec  # kg
        grid_connection_cost = annualize_capex(grid_capex, wacc, 30) / max(annual_h2_total, 1)
    
    # --- LCOH at production site ---
    lcoh_production = elec_cost_per_kg + elec_capex_per_kg + elec_om_per_kg + water_cost + grid_connection_cost
    
    result = {
        'lcoe_weighted': lcoe_weighted,
        'lcoh_production': lcoh_production,
        'elec_cost_per_kg': elec_cost_per_kg,
        'elec_capex_per_kg': elec_capex_per_kg,
        'elec_om_per_kg': elec_om_per_kg,
        'water_cost': water_cost,
        'grid_connection_cost': grid_connection_cost,
        'cf_pv': cf_pv,
        'cf_wind': cf_wind,
        'share_pv': share_pv,
        'share_wind': share_wind,
        'lcoe_pv': lcoe_pv,
        'lcoe_wind': lcoe_wind,
    }
    
    if not include_shipping:
        return result
    
    # --- Step 5: Ammonia Synthesis ---
    # Electricity for synthesis
    nh3_synth_elec_cost = (lcoe_weighted / 1000) * NH3_SYNTH_ELEC
    
    # Capex for synthesis plant
    annualized_synth_capex = annualize_capex(NH3_SYNTH_CAPEX, wacc, NH3_SYNTH_LIFETIME)
    nh3_synth_capex_cost = annualized_synth_capex
    nh3_synth_om_cost = NH3_SYNTH_CAPEX * NH3_SYNTH_OM
    
    nh3_synth_total = nh3_synth_elec_cost + nh3_synth_capex_cost + nh3_synth_om_cost
    
    # --- Step 6: Transport to Port ---
    # Road distance to ocean (port)
    # Ammonia transported: KG_NH3_PER_KG_H2 kg NH3 per kg H2
    # Trucking cost per kg H2
    truck_cost_per_kg_h2 = (road_dist * AMMONIA_TRUCK_COST / 1000) * KG_NH3_PER_KG_H2  # dist(km) * EUR/t-km / 1000 * kg_NH3/kg_H2
    
    # --- Step 7: Maritime Shipping ---
    # Distance to Rotterdam by sea
    sea_dist = shipping_distance_km(lat, lon)
    # Shipping cost per kg H2
    shipping_cost = (sea_dist * SHIPPING_COST_PER_TONNE_KM / 1000) * KG_NH3_PER_KG_H2
    
    # Port handling (loading + unloading)
    port_cost = (PORT_HANDLING_COST * 2 / 1000) * KG_NH3_PER_KG_H2  # *2 for both ports, /1000 for per tonne
    
    # --- Step 8: Ammonia Cracking ---
    crack_heat_cost = NH3_CRACK_HEAT * HEAT_COST
    annualized_crack_capex = annualize_capex(NH3_CRACK_CAPEX, wacc, NH3_CRACK_LIFETIME)
    crack_capex_cost = annualized_crack_capex
    crack_om_cost = NH3_CRACK_CAPEX * NH3_CRACK_OM
    cracking_total = crack_heat_cost + crack_capex_cost + crack_om_cost
    
    # --- Total Delivered Cost ---
    total_delivered = (lcoh_production + nh3_synth_total + truck_cost_per_kg_h2 + 
                       shipping_cost + port_cost + cracking_total)
    
    result.update({
        'nh3_synth_elec': nh3_synth_elec_cost,
        'nh3_synth_capex': nh3_synth_capex_cost,
        'nh3_synth_om': nh3_synth_om_cost,
        'nh3_synth_total': nh3_synth_total,
        'truck_cost': truck_cost_per_kg_h2,
        'sea_distance_km': sea_dist,
        'shipping_cost': shipping_cost,
        'port_cost': port_cost,
        'crack_heat': crack_heat_cost,
        'crack_capex': crack_capex_cost,
        'crack_om': crack_om_cost,
        'cracking_total': cracking_total,
        'total_delivered': total_delivered,
    })
    
    return result


def run_all_scenarios(data_df):
    """Run LCOH calculation for all hexagons under all financing x interest scenarios."""
    results = []
    
    for scen_name, scen_params in SCENARIOS.items():
        base_wacc = scen_params['wacc']
        
        for rate_name, rate_params in INTEREST_SCENARIOS.items():
            wacc = base_wacc + rate_params['wacc_adjustment']
            scenario_label = f"{scen_name}_{rate_name}"
            
            for _, row in data_df.iterrows():
                r = compute_lcoh(row, wacc, include_shipping=True)
                r['hex_id'] = row['hex_id']
                r['lat'] = row['lat']
                r['lon'] = row['lon']
                r['scenario'] = scenario_label
                r['financing'] = scen_name
                r['interest_rate_scenario'] = rate_name
                r['wacc'] = wacc
                r['financing_name'] = scen_params['name']
                r['interest_name'] = rate_params['name']
                results.append(r)
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Load data
    data_path = 'data/hex_final_NA_min.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} hexagons")
    print(df.columns.tolist())
    
    # Run scenarios
    results_df = run_all_scenarios(df)
    
    # Save
    os.makedirs('outputs', exist_ok=True)
    results_df.to_csv('outputs/lcoh_results.csv', index=False)
    
    # Print summary
    print("\n=== SUMMARY BY FINANCING SCENARIO (Low Rates) ===")
    for scen in SCENARIOS:
        subset = results_df[(results_df['financing'] == scen) & 
                           (results_df['interest_rate_scenario'] == 'low_rates')]
        if len(subset) > 0:
            print(f"\n{SCENARIOS[scen]['name']}:")
            print(f"  Min delivered LCOH: {subset['total_delivered'].min():.2f} EUR/kg")
            print(f"  Median delivered LCOH: {subset['total_delivered'].median():.2f} EUR/kg")
            print(f"  Top-5 sites:")
            top5 = subset.nsmallest(5, 'total_delivered')
            for _, r in top5.iterrows():
                print(f"    {r['hex_id']}: lat={r['lat']:.1f}, lon={r['lon']:.1f}, LCOH={r['total_delivered']:.2f} EUR/kg")
    
    print("\nDone! Results saved to outputs/lcoh_results.csv")
