import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

# Load data
df = pd.read_csv('data/hex_final_NA_min.csv')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')

# Model parameters for 2030 (in USD, sources: IRENA, literature approximations)
params = {
    'scenarios': {
        'base_africa': {'wacc': 0.08, 'name': 'Base Africa (8% WACC)'},
        'derisk_africa': {'wacc': 0.06, 'name': 'De-risked (6% WACC)'},
        'lowint_africa': {'wacc': 0.04, 'name': 'Low Interest (4% WACC)'},
        'europe': {'wacc': 0.05, 'name': 'Europe Baseline (5% WACC)'}
    },
    'common': {
        'lifetime': 25,  # years for RES, 15 for electrolyzer avg
        'el_capex': 350,  # $/kW_el 2030
        'el_opex': 0.02,  # fraction CAPEX
        'el_eff': 50,  # kWh/kg H2
        'el_load': 0.85,  # avg load factor
        'pv_capex': 300,  # $/kW_pv
        'wind_capex': 800,  # $/kW_wind
        'pv_opex': 0.015,
        'wind_opex': 0.02,
        'hybrid_ratio': 0.7,  # PV fraction in hybrid
        'bos_fixed': 200,  # $/kW_el fixed BOS
        'grid_cost': 1000,  # $/km per MW? Normalize later
        'road_cost': 500,
        'water_cost': 0.5,  # $/km? approx
        'nh3_conv': 0.8,  # $/kg H2
        'shipping': 0.0003,  # $/kg per km (approx for NH3 ship ~0.2-0.5 $/tonne-km -> per kg km)
        'shipping_dist': 8000,  # km sea to Rotterdam
        'reconv': 1.2,  # $/kg H2 reconversion
    }
}

def crf(r, n):
    return r * (1 + r)**n / ((1 + r)**n - 1)

def compute_lcoh(row, wacc, is_europe=False):
    theo_pv, theo_wind = row['theo_pv'], row['theo_wind']
    grid_d, road_d, ocean_d, water_d = row['grid_dist_km'], row['road_dist_km'], row['ocean_dist_km'], row['waterbody_dist_km']
    
    # Effective CF hybrid
    cf_hybrid = params['common']['hybrid_ratio'] * theo_pv + (1 - params['common']['hybrid_ratio']) * theo_wind
    
    if is_europe:
        cf_hybrid = 0.12 * theo_pv + 0.28 * theo_wind  # Lower Europe CFs, but use site theo as proxy adjust
        el_capex = 400  # Higher
        pv_capex = 400
        wind_capex = 1000
    
    # Size RES for el 1 MW
    P_el = 1  # MW
    prod_h2 = P_el * 8760 * params['common']['el_load'] / params['common']['el_eff']  # tonnes? kg/MWh * MWh = kg/year /1e6 = tonnes?
    # 50 kWh/kg, 8760*0.85 /50 ~ 149 tonnes/year per MW_el approx
    
    # RES overbuild factor ~1 / cf_hybrid * el_load / eff? But simplify, assume sized for avg
    overbuild = params['common']['el_load'] / cf_hybrid  # rough
    P_pv = overbuild * P_el * params['common']['hybrid_ratio']
    P_wind = overbuild * P_el * (1 - params['common']['hybrid_ratio'])
    
    # CAPEX
    capex_el = params['common']['el_capex'] * P_el
    capex_pv = params['common']['pv_capex'] * P_pv
    capex_wind = params['common']['wind_capex'] * P_wind
    capex_bos = params['common']['bos_fixed'] * P_el + params['common']['grid_cost'] * grid_d /100 + params['common']['road_cost'] * road_d /100 + params['common']['water_cost'] * water_d
    capex_total = capex_el + capex_pv + capex_wind + capex_bos
    
    # OPEX annual fraction avg
    opex_frac = 0.018  # weighted
    opex_annual = capex_total * opex_frac
    
    # Annuitized CAPEX
    crf_val = crf(wacc, params['common']['lifetime'])
    capex_annual = capex_total * crf_val
    
    # Production LCOH $/kg
    lcoh_prod = (capex_annual + opex_annual) / (prod_h2 * 1000)  # kg/year
    
    # Transport chain
    pipe_to_port = params['common']['road_cost'] * 0.5 * ocean_d /100  # rough pipe
    nh3_conv_capex_ann = params['common']['nh3_conv'] * prod_h2 * 1000 * crf_val * 0.2  # rough, 20% el size equiv
    shipping_annual = params['common']['shipping'] * params['common']['shipping_dist'] * prod_h2 * 1000
    reconv_annual = params['common']['reconv'] * prod_h2 * 1000
    
    total_annual = capex_annual + opex_annual + nh3_conv_capex_ann + pipe_to_port * crf_val + shipping_annual + reconv_annual
    lcoh_deliv = total_annual / (prod_h2 * 1000)
    
    return lcoh_prod, lcoh_deliv

# Compute for sites
results = []
for sc in params['scenarios']:
    wacc = params['scenarios'][sc]['wacc']
    is_eu = 'europe' in sc
    df[['lcoh_prod', 'lcoh_deliv']] = df.apply(lambda row: compute_lcoh(row, wacc, is_eu), axis=1, result_type='expand')
    df[f'lcoh_prod_{sc}'] = df['lcoh_prod']
    df[f'lcoh_deliv_{sc}'] = df['lcoh_deliv']
    best_site = df.loc[df[f'lcoh_deliv_{sc}'].idxmin()]
    results.append({'scenario': params['scenarios'][sc]['name'], 'min_deliv': best_site[f'lcoh_deliv_{sc}'], 'site': best_site['hex_id']})

res_df = pd.DataFrame(results)
print(res_df)

# Save
df.to_csv('outputs/lcoh_results.csv', index=False)
res_df.to_csv('outputs/top_sites.csv', index=False)
gdf = gdf.merge(df[['hex_id', 'lcoh_deliv_base_africa']], on='hex_id')
gdf.plot(column='lcoh_deliv_base_africa', cmap='Reds_r', legend=True, figsize=(12,12))
plt.title('Delivered LCOH Base Scenario ($/kg)')
plt.savefig('report/images/lcoh_map.png', dpi=300, bbox_inches='tight')
plt.close()

# Europe single point avg
eu_lcoh_prod, eu_lcoh_deliv = compute_lcoh(df.iloc[0], params['scenarios']['europe']['wacc'], True)
print(f'Europe avg delivered LCOH: {eu_lcoh_deliv:.2f} $/kg')

print('Model run complete')