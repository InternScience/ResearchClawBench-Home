#!/usr/bin/env python3
"""
Geospatial Levelized Cost Model for African Green Hydrogen Delivered to Europe
via Ammonia Shipping and Reconversion (2030 Projections)

Based on GeoH2 methodology (Halloran et al., 2024; Müller et al., 2023)
with financing scenarios informed by Steffen (2020) and Schmidt et al. (2019)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'hex_final_NA_min.csv'))
print(f"Loaded {len(df)} hexagon sites")
print(df.describe())

# ============================================================
# 2. TECHNO-ECONOMIC PARAMETERS (2030 projections)
# ============================================================

PARAMS = {
    # Solar PV (2030 projections)
    'pv_capex_per_kw': 550,          # €/kW
    'pv_opex_pct': 0.02,             # % of CAPEX per year
    'pv_lifetime': 25,               # years
    'pv_degradation': 0.005,         # annual degradation rate
    
    # Onshore Wind (2030 projections)
    'wind_capex_per_kw': 1100,       # €/kW
    'wind_opex_pct': 0.03,           # % of CAPEX per year
    'wind_lifetime': 25,             # years
    
    # Electrolyzer PEM (2030 projections)
    'elec_capex_per_kw': 500,        # €/kW
    'elec_opex_pct': 0.03,           # % of CAPEX per year
    'elec_efficiency': 0.65,         # MWh_H2/MWh_el (improved from 0.59)
    'elec_lifetime': 20,             # years
    
    # Battery Storage
    'batt_capex_per_kwh': 120,       # €/kWh
    'batt_opex_pct': 0.02,
    'batt_lifetime': 15,
    'batt_efficiency': 0.90,
    
    # H2 Storage (compressed)
    'h2_store_capex_per_kwh': 15,    # €/kWh_H2
    'h2_store_lifetime': 25,
    
    # Water
    'water_demand_l_per_kg': 21,     # L water / kg H2
    'water_cost_per_m3': 1.25,       # €/m3
    'desal_electricity': 3.7,        # kWh/m3 for seawater
    'freshwater_electricity': 0.4,   # kWh/m3 for freshwater
    'water_transport_cost': 0.001,   # €/km/m3
    
    # Ammonia Synthesis (at production site)
    # Based on IEA Ammonia Technology Roadmap and GeoH2 parameters
    'nh3_synth_capex_per_kgH2yr': 8.0,  # €/(kgH2/year capacity)
    'nh3_synth_opex_pct': 0.015,
    'nh3_synth_lifetime': 25,
    'nh3_synth_elec': 2.809,         # kWh/kgH2
    
    # Ammonia Cracking (at European destination)
    'nh3_crack_capex_per_kgH2yr': 6.0,  # €/(kgH2/year capacity)
    'nh3_crack_opex_pct': 0.02,
    'nh3_crack_lifetime': 25,
    'nh3_crack_heat': 4.2,           # kWh/kgH2
    'eu_heat_price': 0.04,           # €/kWh
    'eu_elec_price': 0.08,           # €/kWh
    
    # Transport to port (trucking as NH3)
    'trucking_cost_per_kg_km': 0.0003,  # €/kgH2/km (NH3 trucking)
    
    # Shipping (Africa to Europe via ammonia tanker)
    # Rotterdam is ~6000-10000 km from various African ports
    'shipping_base_cost': 0.40,      # €/kgH2 (base shipping cost for ~7000 km)
    'shipping_var_cost_per_1000km': 0.05,  # additional cost per 1000 km deviation
    'port_handling_cost': 0.15,      # €/kgH2 (loading + unloading)
    'avg_shipping_dist_km': 7000,    # reference distance
    
    # H2 energy content
    'h2_lhv_kwh_per_kg': 33.33,     # kWh/kg (LHV)
    
    # European production benchmark
    'eu_pv_cf': 0.14,               # European average PV capacity factor
    'eu_wind_cf': 0.28,             # European average wind capacity factor
}

# ============================================================
# 3. FINANCING SCENARIOS
# ============================================================

SCENARIOS = {
    'baseline': {
        'name': 'Baseline (High Risk)',
        'africa_wacc': 0.10,
        'europe_wacc': 0.07,
        'description': 'Current financing conditions, high perceived risk in Africa',
        'tech_factor': 1.0,
    },
    'moderate_derisking': {
        'name': 'Moderate De-risking',
        'africa_wacc': 0.08,
        'europe_wacc': 0.06,
        'description': 'Partial de-risking through blended finance and policy support',
        'tech_factor': 1.0,
    },
    'full_derisking': {
        'name': 'Full De-risking',
        'africa_wacc': 0.06,
        'europe_wacc': 0.05,
        'description': 'Strong de-risking: guarantees, concessional finance, stable policy',
        'tech_factor': 1.0,
    },
    'rising_ir': {
        'name': 'Rising Interest Rates',
        'africa_wacc': 0.12,
        'europe_wacc': 0.09,
        'description': 'Interest rates rise to pre-2008 levels (Schmidt et al. 2019 extreme scenario)',
        'tech_factor': 1.0,
    },
    'optimistic_2030': {
        'name': 'Optimistic 2030',
        'africa_wacc': 0.06,
        'europe_wacc': 0.05,
        'description': 'Full de-risking + aggressive technology cost reductions',
        'tech_factor': 0.80,  # 20% further cost reduction
    }
}

# ============================================================
# 4. CORE COST FUNCTIONS
# ============================================================

def capital_recovery_factor(wacc, lifetime):
    """Annuity factor: converts CAPEX to annual payment."""
    if wacc == 0:
        return 1.0 / lifetime
    return (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def lcoe_solar(capacity_factor, wacc, capex=None):
    """LCOE of solar PV in €/kWh."""
    if capex is None:
        capex = PARAMS['pv_capex_per_kw']
    crf = capital_recovery_factor(wacc, PARAMS['pv_lifetime'])
    annual_gen = capacity_factor * 8760  # kWh/kW/year
    avg_degradation = 1 - PARAMS['pv_degradation'] * PARAMS['pv_lifetime'] / 2
    annual_gen *= avg_degradation
    annual_cost = capex * (crf + PARAMS['pv_opex_pct'])
    return annual_cost / annual_gen

def lcoe_wind(capacity_factor, wacc, capex=None):
    """LCOE of onshore wind in €/kWh."""
    if capex is None:
        capex = PARAMS['wind_capex_per_kw']
    crf = capital_recovery_factor(wacc, PARAMS['wind_lifetime'])
    annual_gen = capacity_factor * 8760  # kWh/kW/year
    annual_cost = capex * (crf + PARAMS['wind_opex_pct'])
    return annual_cost / annual_gen

def lcoh_production(pv_cf, wind_cf, wacc, tech_cost_factor=1.0):
    """
    Calculate LCOH for green hydrogen production (€/kgH2).
    
    Uses a simplified hybrid plant model with optimized PV/Wind mix.
    """
    pv_capex = PARAMS['pv_capex_per_kw'] * tech_cost_factor
    wind_capex = PARAMS['wind_capex_per_kw'] * tech_cost_factor
    elec_capex = PARAMS['elec_capex_per_kw'] * tech_cost_factor
    
    # Calculate LCOE for each source
    pv_lcoe = lcoe_solar(pv_cf, wacc, pv_capex)
    wind_lcoe = lcoe_wind(wind_cf, wacc, wind_capex)
    
    # Optimal mix based on cost and capacity factor
    total_cf = pv_cf + wind_cf
    if total_cf == 0:
        return np.inf, {}
    
    if pv_lcoe < wind_lcoe:
        pv_share = min(0.8, pv_cf / total_cf + 0.2)
        wind_share = 1 - pv_share
    else:
        wind_share = min(0.8, wind_cf / total_cf + 0.2)
        pv_share = 1 - wind_share
    
    hybrid_lcoe = pv_share * pv_lcoe + wind_share * wind_lcoe
    
    # Effective capacity factor with complementarity bonus
    complementarity = 1.15
    effective_cf = min(0.60, (pv_share * pv_cf + wind_share * wind_cf) * complementarity)
    
    # Electrolyzer cost component
    elec_crf = capital_recovery_factor(wacc, PARAMS['elec_lifetime'])
    elec_annual_per_kw = elec_capex * (elec_crf + PARAMS['elec_opex_pct'])
    elec_utilization = effective_cf
    # kWh_el per year per kW electrolyzer
    elec_gen_per_kw = elec_utilization * 8760
    # kgH2 per kW electrolyzer per year
    kg_h2_per_kw_yr = elec_gen_per_kw * PARAMS['elec_efficiency'] / PARAMS['h2_lhv_kwh_per_kg']
    electrolyzer_cost_per_kg = elec_annual_per_kw / kg_h2_per_kw_yr
    
    # Battery storage cost
    batt_hours = max(2, 8 * (1 - effective_cf))
    batt_capex = PARAMS['batt_capex_per_kwh'] * tech_cost_factor * batt_hours
    batt_crf = capital_recovery_factor(wacc, PARAMS['batt_lifetime'])
    batt_annual = batt_capex * (batt_crf + PARAMS['batt_opex_pct'])
    # Spread over annual H2 production per kW
    batt_cost_per_kg = batt_annual / kg_h2_per_kw_yr
    
    # H2 storage cost
    h2_store_hours = max(24, 72 * (1 - effective_cf))
    h2_store_capex = PARAMS['h2_store_capex_per_kwh'] * tech_cost_factor * h2_store_hours
    h2_store_crf = capital_recovery_factor(wacc, PARAMS['h2_store_lifetime'])
    h2_store_annual = h2_store_capex * h2_store_crf
    h2_store_cost_per_kg = h2_store_annual / kg_h2_per_kw_yr
    
    # Electricity cost per kgH2
    kwh_el_per_kg = PARAMS['h2_lhv_kwh_per_kg'] / PARAMS['elec_efficiency']
    electricity_cost_per_kg = hybrid_lcoe * kwh_el_per_kg
    
    # Total LCOH
    lcoh = electricity_cost_per_kg + electrolyzer_cost_per_kg + batt_cost_per_kg + h2_store_cost_per_kg
    
    components = {
        'electricity': electricity_cost_per_kg,
        'electrolyzer': electrolyzer_cost_per_kg,
        'battery': batt_cost_per_kg,
        'h2_storage': h2_store_cost_per_kg,
        'hybrid_lcoe': hybrid_lcoe,
        'effective_cf': effective_cf,
        'pv_share': pv_share,
        'wind_share': wind_share,
        'pv_lcoe': pv_lcoe,
        'wind_lcoe': wind_lcoe,
    }
    
    return lcoh, components

def water_cost(ocean_dist_km, waterbody_dist_km, elec_price=0.05):
    """Calculate water cost for H2 production (€/kgH2)."""
    water_per_kg = PARAMS['water_demand_l_per_kg'] / 1000  # m3/kgH2
    
    # Freshwater option
    fw_treatment = PARAMS['freshwater_electricity'] * elec_price
    fw_transport = PARAMS['water_transport_cost'] * waterbody_dist_km
    fw_total = (PARAMS['water_cost_per_m3'] + fw_treatment + fw_transport) * water_per_kg
    
    # Ocean water option (desalination)
    ow_treatment = PARAMS['desal_electricity'] * elec_price
    ow_transport = PARAMS['water_transport_cost'] * ocean_dist_km
    ow_total = (PARAMS['water_cost_per_m3'] + ow_treatment + ow_transport) * water_per_kg
    
    return min(fw_total, ow_total)

def ammonia_conversion_cost(wacc, elec_price=0.05):
    """Cost of converting H2 to ammonia at production site (€/kgH2)."""
    crf = capital_recovery_factor(wacc, PARAMS['nh3_synth_lifetime'])
    capex_annual = PARAMS['nh3_synth_capex_per_kgH2yr'] * (crf + PARAMS['nh3_synth_opex_pct'])
    elec_cost = PARAMS['nh3_synth_elec'] * elec_price
    return capex_annual + elec_cost

def transport_to_port_cost(ocean_dist_km):
    """Cost of transporting NH3 from production site to nearest port (€/kgH2)."""
    return PARAMS['trucking_cost_per_kg_km'] * ocean_dist_km

def shipping_cost_func(lat):
    """
    Cost of shipping ammonia from African port to Europe (€/kgH2).
    Adjusted by latitude (southern sites have longer shipping routes).
    """
    # Approximate shipping distance based on latitude
    # Northern Africa (~15°S): ~5000 km to Rotterdam
    # Southern Africa (~28°S): ~10000 km to Rotterdam
    base_dist = 5000 + (-lat - 15) * (5000 / 13)  # linear interpolation
    base_dist = max(4000, min(12000, base_dist))
    
    # Cost
    base_cost = PARAMS['shipping_base_cost']
    deviation = (base_dist - PARAMS['avg_shipping_dist_km']) / 1000
    shipping = base_cost + deviation * PARAMS['shipping_var_cost_per_1000km']
    
    return shipping + PARAMS['port_handling_cost']

def reconversion_cost(wacc_eu):
    """Cost of cracking ammonia back to H2 in Europe (€/kgH2)."""
    crf = capital_recovery_factor(wacc_eu, PARAMS['nh3_crack_lifetime'])
    capex_annual = PARAMS['nh3_crack_capex_per_kgH2yr'] * (crf + PARAMS['nh3_crack_opex_pct'])
    heat_cost = PARAMS['nh3_crack_heat'] * PARAMS['eu_heat_price']
    return capex_annual + heat_cost

def european_benchmark(wacc_eu, tech_cost_factor=1.0):
    """Calculate European domestic green H2 production cost (€/kgH2)."""
    lcoh, components = lcoh_production(
        PARAMS['eu_pv_cf'], 
        PARAMS['eu_wind_cf'], 
        wacc_eu,
        tech_cost_factor
    )
    water = 0.05  # €/kgH2 (minimal in Europe)
    return lcoh + water, components

# ============================================================
# 5. CALCULATE COSTS FOR ALL SITES AND SCENARIOS
# ============================================================

results = []

for scenario_key, scenario in SCENARIOS.items():
    africa_wacc = scenario['africa_wacc']
    europe_wacc = scenario['europe_wacc']
    tech_factor = scenario['tech_factor']
    
    # European benchmark
    eu_lcoh, eu_components = european_benchmark(europe_wacc, tech_factor)
    
    for _, row in df.iterrows():
        # Production LCOH
        lcoh, prod_components = lcoh_production(
            row['theo_pv'], row['theo_wind'], africa_wacc, tech_factor
        )
        
        # Local electricity price
        local_elec_price = prod_components['hybrid_lcoe']
        
        # Water cost
        w_cost = water_cost(row['ocean_dist_km'], row['waterbody_dist_km'], local_elec_price)
        
        # Ammonia conversion at source
        nh3_conv = ammonia_conversion_cost(africa_wacc, local_elec_price)
        
        # Transport to port
        transport = transport_to_port_cost(row['ocean_dist_km'])
        
        # Shipping to Europe
        ship = shipping_cost_func(row['lat'])
        
        # Reconversion in Europe
        reconv = reconversion_cost(europe_wacc)
        
        # Total delivered cost
        total_delivered = lcoh + w_cost + nh3_conv + transport + ship + reconv
        
        # Cost advantage vs Europe
        cost_advantage = eu_lcoh - total_delivered
        competitive = total_delivered < eu_lcoh
        
        results.append({
            'hex_id': row['hex_id'],
            'lat': row['lat'],
            'lon': row['lon'],
            'theo_pv': row['theo_pv'],
            'theo_wind': row['theo_wind'],
            'ocean_dist_km': row['ocean_dist_km'],
            'road_dist_km': row['road_dist_km'],
            'waterbody_dist_km': row['waterbody_dist_km'],
            'grid_dist_km': row['grid_dist_km'],
            'scenario': scenario_key,
            'scenario_name': scenario['name'],
            'africa_wacc': africa_wacc,
            'europe_wacc': europe_wacc,
            'lcoh_production': lcoh,
            'water_cost': w_cost,
            'nh3_conversion': nh3_conv,
            'transport_to_port': transport,
            'shipping': ship,
            'reconversion': reconv,
            'total_delivered': total_delivered,
            'eu_benchmark': eu_lcoh,
            'cost_advantage': cost_advantage,
            'competitive': competitive,
            'pv_share': prod_components['pv_share'],
            'wind_share': prod_components['wind_share'],
            'effective_cf': prod_components['effective_cf'],
            'electricity_cost': prod_components['electricity'],
            'electrolyzer_cost': prod_components['electrolyzer'],
            'h2_storage_cost': prod_components['h2_storage'],
            'battery_cost': prod_components['battery'],
            'pv_lcoe': prod_components['pv_lcoe'],
            'wind_lcoe': prod_components['wind_lcoe'],
        })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT, 'full_results.csv'), index=False)
print(f"\nSaved {len(results_df)} results to outputs/full_results.csv")

# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print("\n" + "="*80)
print("SUMMARY BY SCENARIO")
print("="*80)

summary_data = []
for scenario_key in SCENARIOS:
    sdf = results_df[results_df['scenario'] == scenario_key]
    summary = {
        'scenario': scenario_key,
        'scenario_name': SCENARIOS[scenario_key]['name'],
        'africa_wacc': SCENARIOS[scenario_key]['africa_wacc'],
        'europe_wacc': SCENARIOS[scenario_key]['europe_wacc'],
        'min_delivered': round(sdf['total_delivered'].min(), 2),
        'max_delivered': round(sdf['total_delivered'].max(), 2),
        'mean_delivered': round(sdf['total_delivered'].mean(), 2),
        'median_delivered': round(sdf['total_delivered'].median(), 2),
        'std_delivered': round(sdf['total_delivered'].std(), 2),
        'min_production': round(sdf['lcoh_production'].min(), 2),
        'max_production': round(sdf['lcoh_production'].max(), 2),
        'mean_production': round(sdf['lcoh_production'].mean(), 2),
        'eu_benchmark': round(sdf['eu_benchmark'].iloc[0], 2),
        'n_competitive': int(sdf['competitive'].sum()),
        'pct_competitive': round(sdf['competitive'].mean() * 100, 1),
    }
    summary_data.append(summary)
    print(f"\n--- {summary['scenario_name']} ---")
    print(f"  Africa WACC: {summary['africa_wacc']*100:.0f}%, Europe WACC: {summary['europe_wacc']*100:.0f}%")
    print(f"  LCOH Production: {summary['min_production']:.2f} - {summary['max_production']:.2f} €/kgH2 (mean: {summary['mean_production']:.2f})")
    print(f"  Delivered Cost:  {summary['min_delivered']:.2f} - {summary['max_delivered']:.2f} €/kgH2 (mean: {summary['mean_delivered']:.2f})")
    print(f"  EU Benchmark:    {summary['eu_benchmark']:.2f} €/kgH2")
    print(f"  Competitive sites: {summary['n_competitive']}/{len(sdf)} ({summary['pct_competitive']:.0f}%)")

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUT, 'scenario_summary.csv'), index=False)

# ============================================================
# 7. TOP SITES PER SCENARIO
# ============================================================

print("\n" + "="*80)
print("TOP 5 LEAST-COST SITES PER SCENARIO")
print("="*80)

for scenario_key in SCENARIOS:
    sdf = results_df[results_df['scenario'] == scenario_key].sort_values('total_delivered')
    top5 = sdf.head(5)[['hex_id', 'lat', 'lon', 'total_delivered', 'lcoh_production', 
                          'nh3_conversion', 'shipping', 'reconversion', 'eu_benchmark', 'cost_advantage']]
    print(f"\n--- {SCENARIOS[scenario_key]['name']} ---")
    print(top5.to_string(index=False))

# ============================================================
# 8. COST BREAKDOWN FOR BEST SITE
# ============================================================

print("\n" + "="*80)
print("COST BREAKDOWN - BEST SITE (BASELINE SCENARIO)")
print("="*80)

baseline = results_df[results_df['scenario'] == 'baseline'].sort_values('total_delivered').iloc[0]
print(f"Site: {baseline['hex_id']} (lat={baseline['lat']:.2f}, lon={baseline['lon']:.2f})")
print(f"  LCOH Production:    {baseline['lcoh_production']:.3f} €/kgH2")
print(f"    - Electricity:    {baseline['electricity_cost']:.3f}")
print(f"    - Electrolyzer:   {baseline['electrolyzer_cost']:.3f}")
print(f"    - H2 Storage:     {baseline['h2_storage_cost']:.3f}")
print(f"    - Battery:        {baseline['battery_cost']:.3f}")
print(f"  Water:              {baseline['water_cost']:.3f}")
print(f"  NH3 Conversion:     {baseline['nh3_conversion']:.3f}")
print(f"  Transport to Port:  {baseline['transport_to_port']:.3f}")
print(f"  Shipping:           {baseline['shipping']:.3f}")
print(f"  Reconversion:       {baseline['reconversion']:.3f}")
print(f"  TOTAL DELIVERED:    {baseline['total_delivered']:.3f} €/kgH2")
print(f"  EU Benchmark:       {baseline['eu_benchmark']:.3f} €/kgH2")

# ============================================================
# 9. WACC SENSITIVITY ANALYSIS
# ============================================================

print("\n" + "="*80)
print("WACC SENSITIVITY ANALYSIS")
print("="*80)

# Pick the best site from baseline
best_site_id = baseline['hex_id']
best_site = df[df['hex_id'] == best_site_id].iloc[0]

wacc_range = np.arange(0.04, 0.16, 0.005)
sensitivity_data = []

for wacc in wacc_range:
    lcoh, comps = lcoh_production(best_site['theo_pv'], best_site['theo_wind'], wacc)
    w = water_cost(best_site['ocean_dist_km'], best_site['waterbody_dist_km'])
    nh3 = ammonia_conversion_cost(wacc)
    trans = transport_to_port_cost(best_site['ocean_dist_km'])
    ship = shipping_cost_func(best_site['lat'])
    reconv = reconversion_cost(0.06)  # mid-range EU WACC
    total = lcoh + w + nh3 + trans + ship + reconv
    
    # EU benchmark at same interest rate environment
    eu_wacc = max(0.04, wacc - 0.03)  # EU typically 3% lower
    eu_bench, _ = european_benchmark(eu_wacc)
    
    sensitivity_data.append({
        'wacc': wacc,
        'lcoh_production': lcoh,
        'total_delivered': total,
        'eu_benchmark': eu_bench,
        'competitive': total < eu_bench,
        'electricity': comps['electricity'],
        'electrolyzer': comps['electrolyzer'],
        'h2_storage': comps['h2_storage'],
        'battery': comps['battery'],
    })

sensitivity_df = pd.DataFrame(sensitivity_data)
sensitivity_df.to_csv(os.path.join(OUT, 'wacc_sensitivity.csv'), index=False)
print(sensitivity_df[['wacc', 'lcoh_production', 'total_delivered', 'eu_benchmark', 'competitive']].to_string(index=False))

# ============================================================
# 10. EXPORT KEY RESULTS AS JSON
# ============================================================

key_results = {
    'scenarios': {},
    'best_site_baseline': {
        'hex_id': baseline['hex_id'],
        'lat': round(baseline['lat'], 2),
        'lon': round(baseline['lon'], 2),
        'total_delivered': round(baseline['total_delivered'], 2),
        'lcoh_production': round(baseline['lcoh_production'], 2),
    }
}

for scenario_key in SCENARIOS:
    sdf = results_df[results_df['scenario'] == scenario_key]
    best = sdf.sort_values('total_delivered').iloc[0]
    key_results['scenarios'][scenario_key] = {
        'name': SCENARIOS[scenario_key]['name'],
        'min_delivered_cost': round(best['total_delivered'], 2),
        'best_site': best['hex_id'],
        'mean_delivered_cost': round(sdf['total_delivered'].mean(), 2),
        'eu_benchmark': round(sdf['eu_benchmark'].iloc[0], 2),
        'n_competitive': int(sdf['competitive'].sum()),
        'pct_competitive': round(sdf['competitive'].mean() * 100, 1),
    }

with open(os.path.join(OUT, 'key_results.json'), 'w') as f:
    json.dump(key_results, f, indent=2)

print("\n\nAnalysis complete! Results saved to outputs/")
