"""
Geospatial Levelized Cost Model for African Green Hydrogen to Europe (2030)

This module computes the delivered cost of green hydrogen from African production
sites to European demand centers via ammonia shipping and reconversion.

Cost chain:
  PV/Wind -> Electricity -> Electrolysis -> H2 -> NH3 synthesis -> Shipping -> Cracking -> H2 delivered

Financing scenarios vary WACC to capture de-risking effects.
Policy scenarios vary carbon prices affecting European production competitiveness.
"""

import pandas as pd
import numpy as np
import json
import os

# =============================================================================
# 1. TECHNO-ECONOMIC PARAMETERS (2030 projections)
# =============================================================================

class TechParams:
    """Technology and economic parameters for 2030 green hydrogen supply chain."""
    
    # --- Renewable Energy ---
    # Capacity factors derived from potentials (normalized 0-1)
    # PV: typical CF range 0.15-0.30 in Africa; Wind: 0.25-0.55
    pv_cf_min = 0.15
    pv_cf_max = 0.30
    wind_cf_min = 0.25
    wind_cf_max = 0.55
    
    # LCOE parameters ($/kW installed, 2030)
    pv_capex = 400        # $/kW
    pv_opex_pct = 0.015   # % of CAPEX/year
    pv_lifetime = 30      # years
    
    wind_capex = 900      # $/kW (onshore)
    wind_opex_pct = 0.02  # % of CAPEX/year
    wind_lifetime = 25    # years
    
    # --- Electrolysis ---
    electrolyzer_capex = 500     # $/kW (PEM, 2030 projection)
    electrolyzer_opex_pct = 0.04 # % of CAPEX/year
    electrolyzer_lifetime = 20   # years
    electrolyzer_efficiency = 50 # kWh/kg H2 (LHV basis)
    electrolyzer_load_factor = 0.6  # average capacity factor
    
    # --- Ammonia Synthesis ---
    nh3_synthesis_capex = 1200   # $/t-NH3/day capacity
    nh3_synthesis_opex_pct = 0.04
    nh3_synthesis_lifetime = 25
    nh3_energy_intensity = 8.0   # MWh/t-NH3 (compression + synthesis)
    nh3_h2_intensity = 0.176     # t-H2 per t-NH3 (stoichiometric)
    nh3_plant_capacity_factor = 0.9
    
    # --- Shipping (Liquid Ammonia) ---
    ship_capex_per_t = 150       # $/t-NH3 shipping capacity (annualized vessel cost)
    ship_opex_per_t = 30         # $/t-NH3/year operating
    ship_speed = 350             # km/day (average sailing speed ~14.5 knots)
    port_loading_cost = 5        # $/t-NH3 loading
    port_unloading_cost = 5      # $/t-NH3 unloading
    boil_off_rate = 0.001        # % per day (ammonia has lower boil-off than LH2)
    
    # --- Reconversion (NH3 cracking) ---
    cracking_capex = 800         # $/t-H2/day capacity
    cracking_opex_pct = 0.04
    cracking_lifetime = 25
    cracking_energy = 8.0        # MWh/t-H2 (thermal + electrical)
    cracking_efficiency = 0.95   # H2 recovery rate
    
    # --- European Production Baseline ---
    eu_pv_cf = 0.12              # Lower solar resource in Europe
    eu_wind_cf = 0.35            # Good wind in Northern Europe
    eu_land_cost_premium = 1.5   # Multiplier on equipment costs for EU
    
    # --- Financial ---
    currency = "USD"
    exchange_rate_eur_usd = 1.08  # 1 EUR = 1.08 USD
    
    # --- Carbon Pricing ---
    co2_intensity_gray_nh3 = 1.85  # t-CO2 per t-NH3 (gray ammonia baseline)
    co2_intensity_gray_h2 = 10.5   # t-CO2 per t-H2 (SMR baseline)


def compute_crF(wacc, lifetime):
    """Capital Recovery Factor: annualizes capital cost over lifetime at given WACC."""
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc)**lifetime / ((1 + wacc)**lifetime - 1)


def compute_lcoe(capex, opex_pct, lifetime, wacc, cf):
    """Levelized Cost of Electricity ($/MWh)."""
    crf = compute_crF(wacc, lifetime)
    annual_capex = capex * crf  # $/kW/year
    annual_opex = capex * opex_pct  # $/kW/year
    annual_generation = cf * 8760  # MWh/kW/year
    lcoe = (annual_capex + annual_opex) / annual_generation * 1000  # $/MWh
    return lcoe


def compute_lcoh_site(pv_potential, wind_potential, wacc, 
                      grid_dist_km=None, road_dist_km=None):
    """
    Compute Levelized Cost of Hydrogen production at site ($/kg H2).
    
    Uses optimal mix of PV and wind based on potentials.
    """
    params = TechParams()
    
    # Map potentials to capacity factors (linear interpolation)
    pv_cf = params.pv_cf_min + pv_potential * (params.pv_cf_max - params.pv_cf_min)
    wind_cf = params.wind_cf_min + wind_potential * (params.wind_cf_max - params.wind_cf_min)
    
    # Compute LCOE for each source
    lcoe_pv = compute_lcoe(params.pv_capex, params.pv_opex_pct, 
                           params.pv_lifetime, wacc, pv_cf)
    lcoe_wind = compute_lcoe(params.wind_capex, params.wind_opex_pct,
                             params.wind_lifetime, wacc, wind_cf)
    
    # Optimal mix: use cheaper source primarily, blend for higher load factor
    if lcoe_pv <= lcoe_wind:
        blended_lcoe = lcoe_pv * 0.7 + lcoe_wind * 0.3
        effective_cf = pv_cf * 0.7 + wind_cf * 0.3
    else:
        blended_lcoe = lcoe_wind * 0.7 + lcoe_pv * 0.3
        effective_cf = wind_cf * 0.7 + pv_cf * 0.3
    
    # Infrastructure penalty (remote sites need more infrastructure investment)
    infra_penalty = 0.0
    if grid_dist_km is not None:
        # Grid connection cost: ~$10k/km, amortized over production
        grid_connection_annual = grid_dist_km * 10000 * compute_crF(wacc, 25)
        # Assume 100 MW plant producing ~15,000 t-H2/year
        annual_h2_production = 15000 * 1000  # kg/year
        infra_penalty += grid_connection_annual / annual_h2_production
    
    if road_dist_km is not None and road_dist_km > 50:
        # Road access penalty for construction/maintenance
        road_penalty = min(road_dist_km / 500, 0.5)  # Up to $0.50/kg
        infra_penalty += road_penalty
    
    # Electrolysis cost
    crf_el = compute_crF(wacc, params.electrolyzer_lifetime)
    el_annual_capex = params.electrolyzer_capex * crf_el  # $/kW/year
    el_annual_opex = params.electrolyzer_capex * params.electrolyzer_opex_pct  # $/kW/year
    
    # Annual H2 production per kW electrolyzer capacity
    annual_h2_per_kw = params.electrolyzer_load_factor * 8760 / params.electrolyzer_efficiency  # kg/kW/year
    
    el_capex_cost = el_annual_capex / annual_h2_per_kw  # $/kg
    el_opex_cost = el_annual_opex / annual_h2_per_kw    # $/kg
    el_electricity_cost = blended_lcoe / 1000 * params.electrolyzer_efficiency  # $/kg
    
    lcoh_production = el_capex_cost + el_opex_cost + el_electricity_cost + infra_penalty
    
    return {
        'lcoh_production': lcoh_production,
        'lcoe_pv': lcoe_pv,
        'lcoe_wind': lcoe_wind,
        'blended_lcoe': blended_lcoe,
        'pv_cf': pv_cf,
        'wind_cf': wind_cf,
        'el_capex_cost': el_capex_cost,
        'el_opex_cost': el_opex_cost,
        'el_electricity_cost': el_electricity_cost,
        'infra_penalty': infra_penalty,
    }


def compute_ammonia_cost(lcoh_production, wacc, h2_mass_kg=1.0):
    """
    Convert H2 production cost to ammonia production cost.
    Returns cost per kg H2-equivalent in ammonia.
    """
    params = TechParams()
    
    # NH3 synthesis cost per ton NH3
    crf_nh3 = compute_crF(wacc, params.nh3_synthesis_lifetime)
    nh3_annual_capex = params.nh3_synthesis_capex * 365 * crf_nh3  # $/year per t/day capacity
    nh3_annual_opex = params.nh3_synthesis_capex * 365 * params.nh3_synthesis_opex_pct
    nh3_annual_production = 365 * params.nh3_plant_capacity_factor  # t-NH3/year per t/day capacity
    
    nh3_capex_cost = nh3_annual_capex / nh3_annual_production  # $/t-NH3
    nh3_opex_cost = nh3_annual_opex / nh3_annual_production    # $/t-NH3
    
    # Energy cost for synthesis
    nh3_energy_cost = params.nh3_energy_intensity * params.blended_lcoe / 1000 if hasattr(params, 'blended_lcoe') else 0
    
    # H2 cost component: need 0.176 t-H2 per t-NH3
    h2_cost_per_t_nh3 = lcoh_production * 1000 * params.nh3_h2_intensity  # $/t-NH3
    
    # Total NH3 cost per ton
    total_nh3_cost = h2_cost_per_t_nh3 + nh3_capex_cost + nh3_opex_cost
    
    # Convert back to $/kg H2-equivalent
    nh3_cost_per_kg_h2eq = total_nh3_cost / (params.nh3_h2_intensity * 1000)
    
    return {
        'nh3_capex_cost': nh3_capex_cost / (params.nh3_h2_intensity * 1000),
        'nh3_opex_cost': nh3_opex_cost / (params.nh3_h2_intensity * 1000),
        'nh3_total_addition': nh3_cost_per_kg_h2eq - lcoh_production,
    }


def compute_shipping_cost(ocean_dist_km, wacc):
    """
    Compute shipping cost for liquid ammonia from African port to Europe.
    Returns $/kg H2-equivalent.
    """
    params = TechParams()
    
    # Round trip distance (Africa to Rotterdam and back)
    round_trip_km = ocean_dist_km * 2
    
    # Voyage time
    voyage_days = round_trip_km / params.ship_speed
    
    # Daily vessel cost (annualized)
    crf_ship = compute_crF(wacc, 20)  # Ship lifetime ~20 years
    daily_capex = params.ship_capex_per_t * crf_ship / 365  # $/t-NH3/day
    daily_opex = params.ship_opex_per_t / 365  # $/t-NH3/day
    
    # Shipping cost per ton NH3
    shipping_cost_per_t = (daily_capex + daily_opex) * voyage_days + \
                          params.port_loading_cost + params.port_unloading_cost
    
    # Boil-off loss (additional cost to replace lost ammonia)
    boiloff_loss = voyage_days * params.boil_off_rate
    shipping_cost_per_t *= (1 + boiloff_loss)
    
    # Convert to $/kg H2-equivalent
    params_obj = TechParams()
    shipping_cost_per_kg_h2eq = shipping_cost_per_t / (params_obj.nh3_h2_intensity * 1000)
    
    return {
        'shipping_cost': shipping_cost_per_kg_h2eq,
        'voyage_days': voyage_days,
        'boiloff_loss': boiloff_loss,
    }


def compute_reconversion_cost(wacc):
    """
    Compute NH3 cracking cost at European port.
    Returns $/kg H2 delivered.
    """
    params = TechParams()
    
    crf_crack = compute_crF(wacc, params.cracking_lifetime)
    crack_annual_capex = params.cracking_capex * 365 * crf_crack
    crack_annual_opex = params.cracking_capex * 365 * params.cracking_opex_pct
    crack_annual_production = 365 * 0.9  # t-H2/year per t/day capacity
    
    crack_capex_cost = crack_annual_capex / (crack_annual_production * 1000)  # $/kg
    crack_opex_cost = crack_annual_opex / (crack_annual_production * 1000)    # $/kg
    
    # Energy cost for cracking (assume European electricity price)
    eu_electricity_price = 80  # $/MWh (European industrial average)
    crack_energy_cost = params.cracking_energy * eu_electricity_price / 1000  # $/kg
    
    # Efficiency loss: need more NH3 input to get 1 kg H2 output
    efficiency_loss = (1 - params.cracking_efficiency) / params.cracking_efficiency
    
    total_cracking = crack_capex_cost + crack_opex_cost + crack_energy_cost
    
    return {
        'cracking_capex': crack_capex_cost,
        'cracking_opex': crack_opex_cost,
        'cracking_energy': crack_energy_cost,
        'cracking_total': total_cracking,
        'efficiency_loss_factor': efficiency_loss,
    }


def compute_eu_production_cost(wacc, carbon_price=0):
    """
    Compute green hydrogen production cost in Europe (baseline for comparison).
    """
    params = TechParams()
    
    # European renewable resources
    eu_pv_cf = params.eu_pv_cf
    eu_wind_cf = params.eu_wind_cf
    
    # Higher installation costs in Europe
    eu_pv_capex = params.pv_capex * params.eu_land_cost_premium
    eu_wind_capex = params.wind_capex * params.eu_land_cost_premium
    
    lcoe_pv = compute_lcoe(eu_pv_capex, params.pv_opex_pct, 
                           params.pv_lifetime, wacc, eu_pv_cf)
    lcoe_wind = compute_lcoe(eu_wind_capex, params.wind_opex_pct,
                             params.wind_lifetime, wacc, eu_wind_cf)
    
    # Wind is typically better in Europe
    blended_lcoe = lcoe_wind * 0.6 + lcoe_pv * 0.4
    effective_cf = eu_wind_cf * 0.6 + eu_pv_cf * 0.4
    
    # Electrolysis (same tech but potentially higher EU costs)
    eu_el_capex = params.electrolyzer_capex * params.eu_land_cost_premium
    crf_el = compute_crF(wacc, params.electrolyzer_lifetime)
    el_annual_capex = eu_el_capex * crf_el
    el_annual_opex = eu_el_capex * params.electrolyzer_opex_pct
    annual_h2_per_kw = params.electrolyzer_load_factor * 8760 / params.electrolyzer_efficiency
    
    el_cost = (el_annual_capex + el_annual_opex) / annual_h2_per_kw
    el_elec_cost = blended_lcoe / 1000 * params.electrolyzer_efficiency
    
    lcoh_eu = el_cost + el_elec_cost
    
    return {
        'lcoh_eu': lcoh_eu,
        'lcoe_pv': lcoe_pv,
        'lcoe_wind': lcoe_wind,
        'blended_lcoe': blended_lcoe,
    }


# =============================================================================
# 2. SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    'baseline_africa': {
        'wacc': 0.10,  # 10% - typical African project finance
        'description': 'Baseline African financing (WACC 10%)',
    },
    'de_risked_africa': {
        'wacc': 0.05,  # 5% - with MDB guarantees, political risk insurance
        'description': 'De-risked African financing (WACC 5%)',
    },
    'optimistic_africa': {
        'wacc': 0.03,  # 3% - concessional/European-level financing
        'description': 'Optimistic African financing (WACC 3%)',
    },
    'europe_baseline': {
        'wacc': 0.04,  # 4% - typical European project finance
        'description': 'European production baseline (WACC 4%)',
    },
}

POLICY_SCENARIOS = {
    'no_carbon': {
        'carbon_price': 0,
        'description': 'No carbon price',
    },
    'moderate_carbon': {
        'carbon_price': 50,  # $/tCO2
        'description': 'Moderate carbon price ($50/tCO2)',
    },
    'high_carbon': {
        'carbon_price': 100,  # $/tCO2
        'description': 'High carbon price ($100/tCO2)',
    },
}


# =============================================================================
# 3. MAIN COMPUTATION
# =============================================================================

def run_full_analysis():
    """Run complete geospatial LCOH analysis for all sites and scenarios."""
    
    # Load data
    df = pd.read_csv('data/hex_final_NA_min.csv')
    print(f"Loaded {len(df)} production sites")
    
    # Store all results
    all_results = []
    
    for scenario_name, scenario in SCENARIOS.items():
        wacc = scenario['wacc']
        
        for site_idx, row in df.iterrows():
            # Production cost
            prod_result = compute_lcoh_site(
                row['theo_pv'], row['theo_wind'], wacc,
                grid_dist_km=row['grid_dist_km'],
                road_dist_km=row['road_dist_km']
            )
            
            # Ammonia synthesis
            nh3_result = compute_ammonia_cost(prod_result['lcoh_production'], wacc)
            
            # Shipping (ocean distance to Europe)
            ship_result = compute_shipping_cost(row['ocean_dist_km'], wacc)
            
            # Reconversion
            crack_result = compute_reconversion_cost(wacc)
            
            # Total delivered cost (Africa -> Europe)
            if scenario_name != 'europe_baseline':
                total_delivered = (
                    prod_result['lcoh_production'] +
                    nh3_result['nh3_total_addition'] +
                    ship_result['shipping_cost'] +
                    crack_result['cracking_total'] * (1 + crack_result['efficiency_loss_factor'])
                )
                
                result = {
                    'hex_id': row['hex_id'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'country_region': infer_region(row['lat'], row['lon']),
                    'scenario': scenario_name,
                    'wacc': wacc,
                    'theo_pv': row['theo_pv'],
                    'theo_wind': row['theo_wind'],
                    'ocean_dist_km': row['ocean_dist_km'],
                    'grid_dist_km': row['grid_dist_km'],
                    'road_dist_km': row['road_dist_km'],
                    'lcoh_production': prod_result['lcoh_production'],
                    'lcoe_pv': prod_result['lcoe_pv'],
                    'lcoe_wind': prod_result['lcoe_wind'],
                    'blended_lcoe': prod_result['blended_lcoe'],
                    'pv_cf': prod_result['pv_cf'],
                    'wind_cf': prod_result['wind_cf'],
                    'nh3_synthesis_cost': nh3_result['nh3_total_addition'],
                    'shipping_cost': ship_result['shipping_cost'],
                    'reconversion_cost': crack_result['cracking_total'] * (1 + crack_result['efficiency_loss_factor']),
                    'total_delivered_cost': total_delivered,
                    'is_eu_production': False,
                }
            else:
                eu_result = compute_eu_production_cost(wacc)
                result = {
                    'hex_id': 'EU_baseline',
                    'lat': 52.0,  # Approximate Northern Europe
                    'lon': 5.0,
                    'country_region': 'Europe',
                    'scenario': scenario_name,
                    'wacc': wacc,
                    'theo_pv': 0.5,
                    'theo_wind': 0.7,
                    'ocean_dist_km': 0,
                    'grid_dist_km': 0,
                    'road_dist_km': 0,
                    'lcoh_production': eu_result['lcoh_eu'],
                    'lcoe_pv': eu_result['lcoe_pv'],
                    'lcoe_wind': eu_result['lcoe_wind'],
                    'blended_lcoe': eu_result['blended_lcoe'],
                    'pv_cf': TechParams().eu_pv_cf,
                    'wind_cf': TechParams().eu_wind_cf,
                    'nh3_synthesis_cost': 0,
                    'shipping_cost': 0,
                    'reconversion_cost': 0,
                    'total_delivered_cost': eu_result['lcoh_eu'],
                    'is_eu_production': True,
                }
            
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('outputs/full_results.csv', index=False)
    print(f"\nResults saved: {len(results_df)} rows")
    print(f"Scenarios: {results_df['scenario'].unique()}")
    print(f"\nSummary statistics:")
    print(results_df.groupby('scenario')['total_delivered_cost'].describe())
    
    return results_df


def infer_region(lat, lon):
    """Infer African region from coordinates."""
    if lat > 15:
        return 'North Africa'
    elif lat > 0 and lon < 20:
        return 'West Africa'
    elif lat > 0 and lon >= 20:
        return 'Central Africa'
    elif lat <= 0 and lon < 20:
        return 'Southern Africa (West)'
    else:
        return 'Southern Africa (East)'


if __name__ == '__main__':
    results = run_full_analysis()
