import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_002_20260415_132706/data')
OUTPUTS_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_002_20260415_132706/outputs')
REPORT_IMG_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_002_20260415_132706/report/images')
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(exist_ok=True)

# Technology parameters for 2030
TECH_2030 = {
    'electrolyzer_capex': 650,
    'electrolyzer_opex': 0.02,
    'electrolyzer_efficiency': 0.65,
    'electrolyzer_lifetime': 25,
    'pv_capex': 450,
    'pv_opex': 15,
    'pv_lifetime': 30,
    'pv_capacity_factor_base': 0.25,
    'wind_capex': 900,
    'wind_opex': 35,
    'wind_lifetime': 25,
    'wind_capacity_factor_base': 0.35,
    'storage_capex': 15,
    'storage_lifetime': 30,
}

SUPPLY_CHAIN = {
    'ammonia_conversion_cost': 0.25,
    'shipping_cost_per_kg': 0.35,
    'reconversion_cost': 0.60,
    'ammonia_efficiency': 0.88,
    'reconversion_efficiency': 0.70,
}

FINANCING_SCENARIOS = {
    'Optimistic_DeRisked': {'wacc': 0.05},
    'Moderate_Standard': {'wacc': 0.08},
    'High_Risk': {'wacc': 0.12},
    'VeryHigh_Constrained': {'wacc': 0.15},
}

EU_FINANCING = {'wacc': 0.04}

def load_hexagon_data():
    df = pd.read_csv(DATA_DIR / 'hex_final_NA_min.csv')
    return df

def calculate_crf(wacc, lifetime):
    if wacc == 0:
        return 1 / lifetime
    return (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def calculate_lcoh_production(df, scenario_name, scenario_params, tech_params):
    wacc = scenario_params['wacc']
    results = df.copy()
    
    # Capacity factors based on resource quality
    pv_cf = tech_params['pv_capacity_factor_base'] * (1 + 0.5 * (results['theo_pv'] - 0.7))
    pv_cf = pv_cf.clip(lower=0.15, upper=0.35)
    
    wind_cf = tech_params['wind_capacity_factor_base'] * (1 + 0.8 * (results['theo_wind'] - 0.5))
    wind_cf = wind_cf.clip(lower=0.20, upper=0.55)
    
    # Optimal mix: 70% solar, 30% wind for Africa
    solar_share = 0.7
    wind_share = 0.3
    
    # Calculate LCOE for each technology
    crf_pv = calculate_crf(wacc, tech_params['pv_lifetime'])
    lcoe_pv = (tech_params['pv_capex'] * crf_pv + tech_params['pv_opex']) / (pv_cf * 8760)
    
    crf_wind = calculate_crf(wacc, tech_params['wind_lifetime'])
    lcoe_wind = (tech_params['wind_capex'] * crf_wind + tech_params['wind_opex']) / (wind_cf * 8760)
    
    lcoe_mix = solar_share * lcoe_pv + wind_share * lcoe_wind
    
    # Infrastructure penalty
    infra_penalty = 0.0001 * results['grid_dist_km'] + 0.00005 * results['road_dist_km']
    lcoe_mix += infra_penalty
    
    # Electrolyzer costs
    crf_elec = calculate_crf(wacc, tech_params['electrolyzer_lifetime'])
    elec_capex_annual = tech_params['electrolyzer_capex'] * crf_elec
    elec_opex_annual = tech_params['electrolyzer_capex'] * tech_params['electrolyzer_opex']
    
    flh = 4000  # full load hours
    h2_output_kg_per_kw = (flh * tech_params['electrolyzer_efficiency']) / 33.3
    
    lcoh_elec = (elec_capex_annual + elec_opex_annual) / h2_output_kg_per_kw
    
    # Electricity cost
    electricity_kwh_per_kg_h2 = 33.3 / tech_params['electrolyzer_efficiency']
    lcoh_electricity = lcoe_mix * electricity_kwh_per_kg_h2
    
    # Storage costs
    crf_storage = calculate_crf(wacc, tech_params['storage_lifetime'])
    storage_hours = 8
    storage_kwh = storage_hours / 33.3
    storage_capex_annual = tech_params['storage_capex'] * storage_kwh * crf_storage
    lcoh_storage = storage_capex_annual * 33.3 / flh
    
    # Water cost
    water_cost_per_kg = 1.5 * 0.009
    
    # Total LCOH at production site
    lcoh_production = lcoh_elec + lcoh_electricity + lcoh_storage + water_cost_per_kg
    
    results['lcoh_production'] = lcoh_production
    results['lcoe_mix'] = lcoe_mix
    results['pv_cf'] = pv_cf
    results['wind_cf'] = wind_cf
    results['scenario'] = scenario_name
    results['wacc'] = wacc
    
    return results

def calculate_delivered_cost(df, supply_chain):
    # Ammonia conversion cost (€/kg H2)
    conversion_cost = supply_chain['ammonia_conversion_cost']
    
    # Shipping cost to Europe (€/kg H2)
    shipping_cost = supply_chain['shipping_cost_per_kg']
    
    # Reconversion in Europe (€/kg H2)
    reconversion_cost = supply_chain['reconversion_cost']
    
    # Efficiency losses
    ammonia_eff = supply_chain['ammonia_efficiency']
    reconversion_eff = supply_chain['reconversion_efficiency']
    total_efficiency = ammonia_eff * reconversion_eff
    
    # Delivered cost = production / efficiency + conversion + shipping + reconversion
    df['conversion_cost'] = conversion_cost
    df['shipping_cost'] = shipping_cost
    df['reconversion_cost'] = reconversion_cost
    df['total_efficiency'] = total_efficiency
    df['lcoh_delivered'] = df['lcoh_production'] / total_efficiency + conversion_cost + shipping_cost + reconversion_cost
    
    return df

def calculate_european_cost(wacc, tech_params):
    # European production with good resources (North Sea wind + Southern Europe solar)
    pv_cf = 0.22
    wind_cf = 0.45
    
    solar_share = 0.4
    wind_share = 0.6
    
    crf_pv = calculate_crf(wacc, tech_params['pv_lifetime'])
    lcoe_pv = (tech_params['pv_capex'] * crf_pv + tech_params['pv_opex']) / (pv_cf * 8760)
    
    crf_wind = calculate_crf(wacc, tech_params['wind_lifetime'])
    lcoe_wind = (tech_params['wind_capex'] * crf_wind + tech_params['wind_opex']) / (wind_cf * 8760)
    
    lcoe_mix = solar_share * lcoe_pv + wind_share * lcoe_wind
    
    crf_elec = calculate_crf(wacc, tech_params['electrolyzer_lifetime'])
    elec_capex_annual = tech_params['electrolyzer_capex'] * crf_elec
    elec_opex_annual = tech_params['electrolyzer_capex'] * tech_params['electrolyzer_opex']
    
    flh = 4500  # Better grid integration in Europe
    h2_output_kg_per_kw = (flh * tech_params['electrolyzer_efficiency']) / 33.3
    lcoh_elec = (elec_capex_annual + elec_opex_annual) / h2_output_kg_per_kw
    
    electricity_kwh_per_kg_h2 = 33.3 / tech_params['electrolyzer_efficiency']
    lcoh_electricity = lcoe_mix * electricity_kwh_per_kg_h2
    
    water_cost_per_kg = 1.5 * 0.009
    
    # No shipping/reconversion for domestic production
    return lcoh_elec + lcoh_electricity + water_cost_per_kg

def run_all_scenarios():
    print('Loading data...')
    df = load_hexagon_data()
    
    all_results = []
    
    print('Running scenarios...')
    for scenario_name, scenario_params in FINANCING_SCENARIOS.items():
        print(f'  Processing {scenario_name}...')
        results = calculate_lcoh_production(df, scenario_name, scenario_params, TECH_2030)
        results = calculate_delivered_cost(results, SUPPLY_CHAIN)
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate European reference costs
    eu_cost = calculate_european_cost(EU_FINANCING['wacc'], TECH_2030)
    combined_results['eu_production_cost'] = eu_cost
    combined_results['cost_difference'] = combined_results['lcoh_delivered'] - eu_cost
    combined_results['competitive'] = combined_results['lcoh_delivered'] < eu_cost
    
    # Save results
    combined_results.to_csv(OUTPUTS_DIR / 'lcoh_results.csv', index=False)
    print(f'Results saved to {OUTPUTS_DIR}/lcoh_results.csv')
    
    return combined_results

if __name__ == '__main__':
    results = run_all_scenarios()
    print('Done!')
