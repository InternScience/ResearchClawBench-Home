#!/usr/bin/env python3
"""
Geospatial Levelized Cost of Hydrogen (LCOH) Model for African Green Hydrogen to Europe

This module implements a transparent LCOH calculation based on the GeoH2 methodology,
including production costs, ammonia conversion, shipping, and reconversion.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# ============================================================================
# TECHNO-ECONOMIC PARAMETERS FOR 2030
# Based on related work and IEA/IRENA projections
# ============================================================================

@dataclass
class TechnoEconomicParams:
    """Techno-economic parameters for green hydrogen system (2030 projections)"""
    
    # Electrolyzer (PEM/Alkaline average for 2030)
    electrolyzer_capex_eur_per_kw: float = 500.0  # EUR/kW (down from ~800-1000 in 2023)
    electrolyzer_lifetime_years: float = 25.0
    electrolyzer_efficiency_kwh_per_kg: float = 45.0  # kWh/kg H2 (improved from ~50-55)
    
    # Renewable energy (2030 projections)
    pv_capex_eur_per_kw: float = 400.0  # EUR/kW
    wind_capex_eur_per_kw: float = 900.0  # EUR/kW (onshore)
    pv_lifetime_years: float = 25.0
    wind_lifetime_years: float = 25.0
    
    # Storage and balance of plant
    battery_storage_capex_eur_per_kwh: float = 150.0  # For daily cycling
    h2_storage_capex_eur_per_kg: float = 10.0  # Compressed H2 storage
    bos_fraction: float = 0.15  # Balance of system as fraction of capex
    
    # Ammonia conversion
    nh3_synthesis_capex_eur_per_kg_h2_per_day: float = 800.0  # Per kg H2/day capacity
    nh3_conversion_efficiency: float = 0.78  # kg H2 per kg NH3 (theoretical: 0.176 kg H2/kg NH3, so 1/0.176*efficiency)
    nh3_reconversion_efficiency: float = 0.90  # Efficiency of cracking NH3 back to H2
    
    # Shipping
    nh3_shipping_cost_eur_per_tonne_1000km: float = 5.0  # per tonne NH3 per 1000 km
    nh3_h2_energy_ratio: float = 5.9  # kg NH3 needed per kg H2 (1/0.176 * efficiency losses)
    
    # Grid connection
    grid_connection_cost_eur_per_km: float = 50000.0  # EUR/km for transmission line
    
    # Water
    water_treatment_cost_eur_per_m3: float = 0.50
    water_consumption_m3_per_kg_h2: float = 0.018  # ~18 liters per kg H2
    
    # O&M costs (annual as % of capex)
    o_and_m_rate: float = 0.02  # 2% of capex per year
    
    # Capacity factors (will be calculated from resource data)
    # These are multipliers applied to theoretical potentials


@dataclass
class FinancingParams:
    """Financing parameters for different scenarios"""
    wacc: float  # Weighted Average Cost of Capital
    debt_ratio: float = 0.70  # 70% debt financing typical for projects
    debt_interest_rate: float = None  # If None, derived from WACC
    equity_return: float = None  # If None, derived from WACC
    project_lifetime: float = 25.0  # years
    
    def __post_init__(self):
        if self.debt_interest_rate is None:
            # Assume debt rate is WACC - 2% (typical spread)
            self.debt_interest_rate = max(0.03, self.wacc - 0.02)
        if self.equity_return is None:
            # Equity return higher than WACC
            self.equity_return = self.wacc + 0.04


# Standard financing scenarios for 2030
FINANCING_SCENARIOS = {
    "baseline_wacc": FinancingParams(
        wacc=0.08,  # 8% typical for developing country renewable projects
        debt_ratio=0.70
    ),
    "de_risked": FinancingParams(
        wacc=0.05,  # 5% with guarantees/de-risking (approaching developed country rates)
        debt_ratio=0.80  # Higher debt ratio possible with guarantees
    ),
    "high_interest_rate": FinancingParams(
        wacc=0.12,  # 12% in high interest rate environment
        debt_ratio=0.60  # Lower debt ratio due to higher rates
    ),
    "europe_production": FinancingParams(
        wacc=0.04,  # 4% typical for European projects
        debt_ratio=0.80
    )
}


def calculate_capital_recovery_factor(wacc: float, lifetime: float) -> float:
    """Calculate the capital recovery factor (CRF) for annuitization"""
    if wacc == 0:
        return 1.0 / lifetime
    crf = (wacc * (1 + wacc) ** lifetime) / ((1 + wacc) ** lifetime - 1)
    return crf


def calculate_capacity_factor(theo_pv: float, theo_wind: float, 
                              pv_cf_base: float = 0.20, 
                              wind_cf_base: float = 0.35) -> Tuple[float, float, float]:
    """
    Calculate capacity factors from theoretical potential values.
    
    The theoretical potential values (theo_pv, theo_wind) are normalized scores.
    We convert them to realistic capacity factors.
    
    Returns: (pv_cf, wind_cf, combined_cf)
    """
    # Scale theoretical potentials to realistic capacity factors
    # Assuming theo values range roughly 0.3-0.9 based on data inspection
    pv_cf = pv_cf_base * (theo_pv / 0.5)  # Normalize around 0.5
    pv_cf = np.clip(pv_cf, 0.12, 0.28)  # Cap between 12% and 28%
    
    wind_cf = wind_cf_base * (theo_wind / 0.5)
    wind_cf = np.clip(wind_cf, 0.20, 0.50)  # Cap between 20% and 50%
    
    # Combined CF assuming optimal mix (simplified: weighted average)
    # In practice, would optimize the PV:wind ratio
    combined_cf = 0.6 * pv_cf + 0.4 * wind_cf  # Assume 60:40 PV:wind mix
    
    return pv_cf, wind_cf, combined_cf


def calculate_lcoh_production(lat: float, lon: float,
                              theo_pv: float, theo_wind: float,
                              grid_dist_km: float, road_dist_km: float,
                              ocean_dist_km: float, waterbody_dist_km: float,
                              tech_params: TechnoEconomicParams,
                              fin_params: FinancingParams) -> Dict:
    """
    Calculate levelized cost of hydrogen production at a specific location.
    
    Returns a dictionary with detailed cost breakdown.
    """
    # Calculate capacity factors
    pv_cf, wind_cf, combined_cf = calculate_capacity_factor(theo_pv, theo_wind)
    
    # Assume 1 kg H2/hour nominal capacity as reference (8760 kg/year at 100% CF)
    nominal_capacity_kg_h2_per_hour = 1.0
    annual_h2_production_kg = nominal_capacity_kg_h2_per_hour * 8760 * combined_cf
    
    # Electrolyzer capacity needed (kW)
    electrolyzer_capacity_kw = nominal_capacity_kg_h2_per_hour * tech_params.electrolyzer_efficiency_kwh_per_kg
    
    # Renewable capacity needed (accounting for CF and efficiency losses)
    # Total annual electricity needed
    annual_electricity_kwh = annual_h2_production_kg * tech_params.electrolyzer_efficiency_kwh_per_kg
    
    # PV and wind capacity (assuming 60:40 split)
    pv_capacity_kw = (0.6 * annual_electricity_kwh) / (8760 * pv_cf)
    wind_capacity_kw = (0.4 * annual_electricity_kwh) / (8760 * wind_cf)
    
    # Capital costs
    electrolyzer_capex = electrolyzer_capacity_kw * tech_params.electrolyzer_capex_eur_per_kw
    pv_capex = pv_capacity_kw * tech_params.pv_capex_eur_per_kw
    wind_capex = wind_capacity_kw * tech_params.wind_capex_eur_per_kw
    
    # Balance of system
    total_renewable_capex = pv_capex + wind_capex
    bos_cost = (electrolyzer_capex + total_renewable_capex) * tech_params.bos_fraction
    
    # Grid connection cost (if needed - simplified assumption)
    grid_connection_cost = grid_dist_km * tech_params.grid_connection_cost_eur_per_km * 0.001  # Scale down
    
    # Battery storage (for daily cycling, simplified)
    # Assume 4 hours of storage for smoothing
    battery_capacity_kwh = electrolyzer_capacity_kw * 4
    battery_capex = battery_capacity_kwh * tech_params.battery_storage_capex_eur_per_kwh
    
    # H2 storage
    h2_storage_capacity_kg = nominal_capacity_kg_h2_per_hour * 24  # 1 day storage
    h2_storage_capex = h2_storage_capacity_kg * tech_params.h2_storage_capex_eur_per_kg
    
    # Total capital cost
    total_capex = electrolyzer_capex + pv_capex + wind_capex + bos_cost + grid_connection_cost + battery_capex + h2_storage_capex
    
    # Annualized capital cost
    crf = calculate_capital_recovery_factor(fin_params.wacc, fin_params.project_lifetime)
    annualized_capex = total_capex * crf
    
    # O&M costs
    annual_om = total_capex * tech_params.o_and_m_rate
    
    # Water costs
    annual_water_cost = annual_h2_production_kg * tech_params.water_consumption_m3_per_kg_h2 * tech_params.water_treatment_cost_eur_per_m3
    
    # Total annual cost
    total_annual_cost = annualized_capex + annual_om + annual_water_cost
    
    # LCOH (EUR/kg)
    lcoh_production = total_annual_cost / annual_h2_production_kg
    
    return {
        'lcoh_production_eur_per_kg': lcoh_production,
        'annual_h2_production_kg': annual_h2_production_kg,
        'combined_capacity_factor': combined_cf,
        'pv_capacity_factor': pv_cf,
        'wind_capacity_factor': wind_cf,
        'electrolyzer_capacity_kw': electrolyzer_capacity_kw,
        'pv_capacity_kw': pv_capacity_kw,
        'wind_capacity_kw': wind_capacity_kw,
        'total_capex_eur': total_capex,
        'annualized_capex_eur': annualized_capex,
        'annual_om_eur': annual_om,
        'annual_water_cost_eur': annual_water_cost,
        'capex_breakdown': {
            'electrolyzer': electrolyzer_capex,
            'pv': pv_capex,
            'wind': wind_capex,
            'bos': bos_cost,
            'grid_connection': grid_connection_cost,
            'battery': battery_capex,
            'h2_storage': h2_storage_capex
        }
    }


def calculate_ammonia_conversion_costs(annual_h2_kg: float,
                                       tech_params: TechnoEconomicParams,
                                       fin_params: FinancingParams) -> Dict:
    """Calculate costs for converting H2 to NH3"""
    # Ammonia synthesis plant capacity (kg H2 per day)
    daily_h2_kg = annual_h2_kg / 365
    nh3_plant_capacity = daily_h2_kg  # kg H2/day
    
    # Capital cost for ammonia synthesis
    nh3_capex = nh3_plant_capacity * tech_params.nh3_synthesis_capex_eur_per_kg_h2_per_day
    
    # Annualized cost
    crf = calculate_capital_recovery_factor(fin_params.wacc, 25)  # 25 year lifetime
    annualized_nh3_capex = nh3_capex * crf
    
    # Operating cost (energy for Haber-Bosch process)
    # Approximately 8-10 MWh per tonne NH3, or ~0.037 MWh per kg H2 equivalent
    nh3_energy_kwh_per_kg_h2 = 3.0  # kWh per kg H2 converted
    electricity_cost_eur_per_kwh = 0.03  # Low cost from dedicated renewables
    annual_nh3_opex = annual_h2_kg * nh3_energy_kwh_per_kg_h2 * electricity_cost_eur_per_kwh
    
    # O&M
    annual_nh3_om = nh3_capex * 0.02
    
    # Total annual cost
    total_annual_nh3_cost = annualized_nh3_capex + annual_nh3_opex + annual_nh3_om
    
    # Cost per kg H2
    nh3_conversion_cost = total_annual_nh3_cost / annual_h2_kg
    
    return {
        'nh3_conversion_cost_eur_per_kg_h2': nh3_conversion_cost,
        'nh3_capex_eur': nh3_capex,
        'annualized_nh3_capex_eur': annualized_nh3_capex,
        'annual_nh3_opex_eur': annual_nh3_opex
    }


def calculate_shipping_costs(ocean_dist_km: float, annual_h2_kg: float,
                             tech_params: TechnoEconomicParams,
                             fin_params: FinancingParams) -> Dict:
    """Calculate shipping costs for ammonia transport"""
    # Convert H2 to NH3 mass
    annual_nh3_tonnes = (annual_h2_kg * tech_params.nh3_h2_energy_ratio) / 1000
    
    # Shipping cost (simplified linear model)
    # Base port costs + distance-based costs
    base_port_cost_eur_per_tonne = 20.0  # Loading/unloading
    distance_cost = (ocean_dist_km / 1000) * tech_params.nh3_shipping_cost_eur_per_tonne_1000km
    
    total_shipping_cost_per_tonne = base_port_cost_eur_per_tonne + distance_cost
    total_annual_shipping_cost = annual_nh3_tonnes * total_shipping_cost_per_tonne
    
    shipping_cost_per_kg_h2 = total_annual_shipping_cost / annual_h2_kg
    
    return {
        'shipping_cost_eur_per_kg_h2': shipping_cost_per_kg_h2,
        'annual_nh3_tonnes': annual_nh3_tonnes,
        'total_shipping_cost_per_tonne': total_shipping_cost_per_tonne,
        'distance_km': ocean_dist_km
    }


def calculate_reconversion_costs(annual_h2_kg: float,
                                 tech_params: TechnoEconomicParams,
                                 fin_params: FinancingParams) -> Dict:
    """Calculate costs for reconverting NH3 back to H2 at destination"""
    # Cracking plant capacity
    daily_h2_kg = annual_h2_kg / 365
    
    # Similar capex structure to synthesis
    cracking_capex = daily_h2_kg * tech_params.nh3_synthesis_capex_eur_per_kg_h2_per_day * 0.8  # Slightly cheaper
    
    # Annualized cost
    crf = calculate_capital_recovery_factor(fin_params.wacc, 25)
    annualized_cracking_capex = cracking_capex * crf
    
    # Energy for cracking (endothermic process)
    cracking_energy_kwh_per_kg_h2 = 4.0  # kWh per kg H2 recovered
    electricity_cost_eur_per_kwh = 0.05  # European electricity price
    
    # Losses in reconversion
    effective_h2_output = annual_h2_kg * tech_params.nh3_reconversion_efficiency
    
    annual_cracking_opex = annual_h2_kg * cracking_energy_kwh_per_kg_h2 * electricity_cost_eur_per_kwh
    annual_cracking_om = cracking_capex * 0.02
    
    total_annual_cracking_cost = annualized_cracking_capex + annual_cracking_opex + annual_cracking_om
    
    # Cost per kg H2 delivered
    reconversion_cost = total_annual_cracking_cost / effective_h2_output
    
    return {
        'reconversion_cost_eur_per_kg_h2': reconversion_cost,
        'cracking_capex_eur': cracking_capex,
        'effective_h2_output_kg': effective_h2_output
    }


def calculate_total_delivered_cost(row: pd.Series,
                                   tech_params: TechnoEconomicParams,
                                   fin_params: FinancingParams,
                                   scenario_name: str) -> Dict:
    """
    Calculate complete delivered cost from production to Europe.
    
    Returns comprehensive cost breakdown.
    """
    # Production costs
    prod_results = calculate_lcoh_production(
        lat=row['lat'],
        lon=row['lon'],
        theo_pv=row['theo_pv'],
        theo_wind=row['theo_wind'],
        grid_dist_km=row['grid_dist_km'],
        road_dist_km=row['road_dist_km'],
        ocean_dist_km=row['ocean_dist_km'],
        waterbody_dist_km=row['waterbody_dist_km'],
        tech_params=tech_params,
        fin_params=fin_params
    )
    
    annual_h2 = prod_results['annual_h2_production_kg']
    
    # Ammonia conversion
    nh3_results = calculate_ammonia_conversion_costs(annual_h2, tech_params, fin_params)
    
    # Shipping (assume Rotterdam as destination)
    ship_results = calculate_shipping_costs(row['ocean_dist_km'], annual_h2, tech_params, fin_params)
    
    # Reconversion at destination
    recon_results = calculate_reconversion_costs(annual_h2, tech_params, fin_params)
    
    # Account for reconversion losses
    delivery_factor = tech_params.nh3_reconversion_efficiency
    
    # Total delivered cost (EUR/kg H2 delivered)
    production_cost = prod_results['lcoh_production_eur_per_kg']
    nh3_cost = nh3_results['nh3_conversion_cost_eur_per_kg_h2']
    shipping_cost = ship_results['shipping_cost_eur_per_kg_h2']
    reconversion_cost = recon_results['reconversion_cost_eur_per_kg_h2']
    
    # Total cost per kg H2 delivered (accounting for losses)
    total_delivered_cost = (production_cost + nh3_cost + shipping_cost) / delivery_factor + reconversion_cost
    
    return {
        'hex_id': row['hex_id'],
        'lat': row['lat'],
        'lon': row['lon'],
        'scenario': scenario_name,
        'wacc': fin_params.wacc,
        'lcoh_production': production_cost,
        'nh3_conversion_cost': nh3_cost,
        'shipping_cost': shipping_cost,
        'reconversion_cost': reconversion_cost,
        'total_delivered_cost': total_delivered_cost,
        'delivery_factor': delivery_factor,
        'capacity_factor': prod_results['combined_capacity_factor'],
        'annual_h2_production_kg': annual_h2,
        'total_capex_eur': prod_results['total_capex_eur'],
        'capex_breakdown': prod_results['capex_breakdown']
    }


def run_scenario_analysis(df: pd.DataFrame,
                          tech_params: TechnoEconomicParams,
                          financing_scenarios: Dict[str, FinancingParams]) -> pd.DataFrame:
    """
    Run LCOH analysis for all locations and scenarios.
    
    Returns DataFrame with results.
    """
    all_results = []
    
    for scenario_name, fin_params in financing_scenarios.items():
        for idx, row in df.iterrows():
            result = calculate_total_delivered_cost(row, tech_params, fin_params, scenario_name)
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    return results_df


if __name__ == "__main__":
    # Test the model
    print("Testing LCOH model...")
    
    # Load data
    df = pd.read_csv('data/hex_final_NA_min.csv')
    
    # Initialize parameters
    tech_params = TechnoEconomicParams()
    
    # Run analysis
    results = run_scenario_analysis(df.head(3), tech_params, FINANCING_SCENARIOS)
    
    print("\nSample results:")
    print(results[['hex_id', 'scenario', 'lcoh_production', 'total_delivered_cost']].to_string())
