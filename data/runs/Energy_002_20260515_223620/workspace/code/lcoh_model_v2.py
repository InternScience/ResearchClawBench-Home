#!/usr/bin/env python3
"""
Improved Geospatial Levelized-Cost Model for African Green Hydrogen to Europe.
Corrected shipping costs, added detailed cost breakdowns and scenario analysis.
"""

import numpy as np
import pandas as pd
import json
import os

# ============================================================
# TECHNO-ECONOMIC PARAMETERS (2030 projections)
# ============================================================

# --- Renewable Generation ---
PV_CAPEX_2030 = 520000       # EUR/MW
WIND_CAPEX_2030 = 900000     # EUR/MW
PV_OM = 0.015                # % CAPEX/year
WIND_OM = 0.025
PV_LIFETIME = 25
WIND_LIFETIME = 25
ELEC_PER_KG_H2 = 52.0        # kWh_el / kg H2 (2030 PEM)

# --- Electrolyzer ---
ELECTROLYZER_CAPEX_2030 = 500000  # EUR/MW_el
ELECTROLYZER_OM = 0.03
ELECTROLYZER_LIFETIME = 20

# --- Water ---
WATER_BASE_COST = 0.03       # EUR/kg_H2
WATER_DISTANCE_FACTOR = 0.0003  # EUR/kg_H2 per km

# --- Grid Connection ---
GRID_CONNECTION_COST_PER_KM = 50000  # EUR/km
GRID_LIFETIME = 30

# --- Ammonia Synthesis ---
NH3_SYNTH_ELEC = 2.809       # kWh_el / kg_H2
NH3_SYNTH_CAPEX_2030 = 0.40  # EUR per kg_H2 annual capacity
NH3_SYNTH_OM = 0.015
NH3_SYNTH_LIFETIME = 25
KG_NH3_PER_KG_H2 = 1.0 / 0.1776  # ~5.63

# --- Truck Transport ---
AMMONIA_TRUCK_COST = 0.06    # EUR/tonne-km (heavy truck)

# --- Maritime Shipping ---
# Bulk ammonia shipping: ~$40-80/tonne intercontinental, ~0.003-0.006 EUR/tonne-km
SHIPPING_COST_PER_TONNE_KM = 0.004  # EUR/tonne-km (corrected)
PORT_HANDLING_COST = 15.0    # EUR/tonne NH3 per port call

# --- Ammonia Cracking ---
NH3_CRACK_HEAT = 4.2         # kWh_heat / kg_H2
HEAT_COST = 0.03             # EUR/kWh_heat
NH3_CRACK_CAPEX_2030 = 0.15  # EUR per kg_H2 annual capacity
NH3_CRACK_OM = 0.02
NH3_CRACK_LIFETIME = 25

# --- Europe ---
ROTTERDAM_LAT = 51.909
ROTTERDAM_LON = 4.482

# --- European Green H2 Production Benchmark ---
# 2030 estimates: solar in Spain ~3.0, wind in North Sea ~3.5-4.5, Netherlands mix ~4.5
EUROPE_LCOH = {
    'spain_solar': 3.0,
    'north_sea_wind': 4.0,
    'netherlands_mix': 4.5,
    'germany_mix': 5.0,
}

# ============================================================
# SCENARIOS
# ============================================================

FINANCING_SCENARIOS = {
    'high_risk':    {'name': 'High Risk (Baseline)',    'wacc': 0.12},
    'moderate_derisk': {'name': 'Moderate De-risking',  'wacc': 0.08},
    'strong_derisk':   {'name': 'Strong De-risking',    'wacc': 0.06},
}

INTEREST_SCENARIOS = {
    'low_rates':     {'name': 'Low Rates',     'wacc_delta': 0.00},
    'moderate_rise': {'name': 'Moderate Rise', 'wacc_delta': 0.02},
    'extreme_rise':  {'name': 'Extreme Rise',  'wacc_delta': 0.04},
}

# ============================================================
# FUNCTIONS
# ============================================================

def crf(r, n):
    if r == 0:
        return 1.0 / n
    return r * (1+r)**n / ((1+r)**n - 1)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) *
         np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def shipping_dist(lat, lon):
    """Approximate maritime distance to Rotterdam."""
    direct = haversine(lat, lon, ROTTERDAM_LAT, ROTTERDAM_LON)
    if lon < 15:
        factor = 1.30
    elif lon < 30:
        factor = 1.45
    else:
        factor = 1.75
    return direct * factor

def compute_lcoh(row, wacc, eu_crack_wacc=0.04):
    """
    Compute delivered LCOH for one hexagon.
    Cracking plant uses European WACC since it's built in Europe.
    """
    cf_pv, cf_wind = row['theo_pv'], row['theo_wind']
    rd, gd, wd = row['road_dist_km'], row['grid_dist_km'], row['waterbody_dist_km']
    lat, lon = row['lat'], row['lon']

    total_cf = cf_pv + cf_wind
    sp = cf_pv / total_cf if total_cf > 0 else 0.5
    sw = cf_wind / total_cf if total_cf > 0 else 0.5
    combined_cf = sp * cf_pv + sw * cf_wind

    # LCOE (EUR/MWh)
    apv = PV_CAPEX_2030 * crf(wacc, PV_LIFETIME)
    lcoe_pv = (apv + PV_CAPEX_2030 * PV_OM) / max(cf_pv * 8760, 1)
    aw = WIND_CAPEX_2030 * crf(wacc, WIND_LIFETIME)
    lcoe_wind = (aw + WIND_CAPEX_2030 * WIND_OM) / max(cf_wind * 8760, 1)
    lcoe = sp * lcoe_pv + sw * lcoe_wind

    # H2 production
    elec_cost_kg = (lcoe / 1000) * ELEC_PER_KG_H2
    annual_h2_per_mw = (1000 / ELEC_PER_KG_H2) * 8760 * combined_cf
    ae = ELECTROLYZER_CAPEX_2030 * crf(wacc, ELECTROLYZER_LIFETIME)
    elec_capex_kg = ae / max(annual_h2_per_mw, 1)
    elec_om_kg = (ELECTROLYZER_CAPEX_2030 * ELECTROLYZER_OM) / max(annual_h2_per_mw, 1)
    water_kg = WATER_BASE_COST + WATER_DISTANCE_FACTOR * wd

    # Grid connection
    grid_kg = 0.0
    if gd > 0:
        gc = gd * GRID_CONNECTION_COST_PER_KM
        grid_kg = gc * crf(wacc, GRID_LIFETIME) / max(annual_h2_per_mw, 1)

    lcoh_prod = elec_cost_kg + elec_capex_kg + elec_om_kg + water_kg + grid_kg

    # Ammonia synthesis
    syn_elec = (lcoe / 1000) * NH3_SYNTH_ELEC
    syn_capex = NH3_SYNTH_CAPEX_2030 * crf(wacc, NH3_SYNTH_LIFETIME)
    syn_om = NH3_SYNTH_CAPEX_2030 * NH3_SYNTH_OM
    syn_total = syn_elec + syn_capex + syn_om

    # Truck to port
    truck = (rd * AMMONIA_TRUCK_COST / 1000) * KG_NH3_PER_KG_H2

    # Shipping
    sdist = shipping_dist(lat, lon)
    ship = (sdist * SHIPPING_COST_PER_TONNE_KM / 1000) * KG_NH3_PER_KG_H2
    port = (PORT_HANDLING_COST * 2 / 1000) * KG_NH3_PER_KG_H2

    # Cracking (European WACC)
    crack_heat = NH3_CRACK_HEAT * HEAT_COST
    crack_capex = NH3_CRACK_CAPEX_2030 * crf(eu_crack_wacc, NH3_CRACK_LIFETIME)
    crack_om = NH3_CRACK_CAPEX_2030 * NH3_CRACK_OM
    crack_total = crack_heat + crack_capex + crack_om

    total = lcoh_prod + syn_total + truck + ship + port + crack_total

    return {
        'lcoe': lcoe, 'lcoe_pv': lcoe_pv, 'lcoe_wind': lcoe_wind,
        'cf_combined': combined_cf, 'share_pv': sp, 'share_wind': sw,
        'elec_cost_kg': elec_cost_kg, 'elec_capex_kg': elec_capex_kg,
        'elec_om_kg': elec_om_kg, 'water_kg': water_kg, 'grid_kg': grid_kg,
        'lcoh_production': lcoh_prod,
        'syn_elec': syn_elec, 'syn_capex': syn_capex, 'syn_om': syn_om,
        'syn_total': syn_total,
        'truck_kg': truck, 'sea_dist_km': sdist,
        'ship_kg': ship, 'port_kg': port,
        'crack_heat': crack_heat, 'crack_capex': crack_capex,
        'crack_om': crack_om, 'crack_total': crack_total,
        'total_delivered': total,
    }

def run_all():
    df = pd.read_csv('data/hex_final_NA_min.csv')
    rows = []
    for fin_k, fin_v in FINANCING_SCENARIOS.items():
        for int_k, int_v in INTEREST_SCENARIOS.items():
            wacc = fin_v['wacc'] + int_v['wacc_delta']
            for _, row in df.iterrows():
                r = compute_lcoh(row, wacc)
                r['hex_id'] = row['hex_id']
                r['lat'] = row['lat']
                r['lon'] = row['lon']
                r['financing'] = fin_k
                r['interest'] = int_k
                r['wacc'] = wacc
                r['fin_name'] = fin_v['name']
                r['int_name'] = int_v['name']
                r['scenario'] = f"{fin_k}_{int_k}"
                rows.append(r)
    results = pd.DataFrame(rows)
    os.makedirs('outputs', exist_ok=True)
    results.to_csv('outputs/lcoh_results_v2.csv', index=False)
    
    # Also compute European LCOH for comparison
    eu_data = []
    for fin_k, fin_v in FINANCING_SCENARIOS.items():
        for int_k, int_v in INTEREST_SCENARIOS.items():
            wacc_eu = 0.04 + int_v['wacc_delta']  # European base WACC ~4%
            # European production (simplified)
            # Using similar model but with EU-specific parameters
            for loc, base_lcoh in EUROPE_LCOH.items():
                # Scale LCOH by WACC ratio
                eu_lcoh = base_lcoh * (crf(wacc_eu, 25) / crf(0.04, 25))
                eu_data.append({
                    'location': loc,
                    'financing': fin_k,
                    'interest': int_k,
                    'eu_wacc': wacc_eu,
                    'eu_lcoh': eu_lcoh,
                })
    eu_df = pd.DataFrame(eu_data)
    eu_df.to_csv('outputs/europe_lcoh_estimates.csv', index=False)
    
    print("Done. Results saved.")
    return results, eu_df

if __name__ == '__main__':
    results, eu = run_all()
    
    # Quick summary
    for fin in ['high_risk', 'moderate_derisk', 'strong_derisk']:
        sub = results[(results['financing'] == fin) & (results['interest'] == 'low_rates')]
        print(f"\n{fin}: min={sub['total_delivered'].min():.2f}, "
              f"med={sub['total_delivered'].median():.2f}, "
              f"max={sub['total_delivered'].max():.2f} EUR/kg")
