#!/usr/bin/env python3
"""
African Green Hydrogen Levelized-Cost Model (Corrected)
Estimates delivered cost of African green hydrogen to Europe via ammonia shipping
by 2030 under multiple financing and policy scenarios.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv('data/hex_final_NA_min.csv')
print(f"Loaded {len(df)} hexagonal sites")

africa = gpd.read_file('data/africa_map/ne_10m_admin_0_countries.shp')
africa = africa[africa['CONTINENT'] == 'Africa']
print(f"Loaded {len(africa)} African countries")

# ============================================================
# 2. TECHNO-ECONOMIC PARAMETERS (2030 projections)
# ============================================================
print("\nDefining 2030 techno-economic parameters...")

params = {
    # Electrolyzer (PEM) - €/kW
    'electrolyzer_capex_per_kw': 500,     # €/kW (2030 projection)
    'electrolyzer_efficiency': 55,         # kWh_el/kg_H2 (LHV basis)
    'electrolyzer_lifetime': 20,
    'electrolyzer_opex_pct': 0.02,

    # Solar PV - €/kW
    'pv_capex_per_kw': 450,
    'pv_opex_pct': 0.015,
    'pv_lifetime': 30,

    # Wind onshore - €/kW
    'wind_capex_per_kw': 1200,
    'wind_opex_pct': 0.02,
    'wind_lifetime': 25,

    # Battery storage (4h) - €/kWh
    'battery_capex_per_kwh': 150,
    'battery_hours': 4,
    'battery_opex_pct': 0.015,
    'battery_lifetime': 15,

    # Water desalination
    'water_cost_per_m3': 1.5,
    'water_inland_cost': 3.0,
    'water_per_kg_h2': 0.01,

    # Ammonia synthesis
    'nh3_capex_per_ton_h2_yr': 800,
    'nh3_opex_pct': 0.015,
    'nh3_lifetime': 25,
    'nh3_elec_kwh_per_kg_h2': 1.67,

    # Ammonia cracking
    'cracking_heat_kwh_per_kg_h2': 4.2,
    'cracking_capex_per_ton_h2_yr': 1200,
    'cracking_opex_pct': 0.02,
    'cracking_lifetime': 25,

    # Shipping
    'shipping_cost_per_ton_km': 0.00004,
    'port_handling_cost_per_kg': 0.02,

    # Infrastructure
    'grid_connection_eur_per_mw_km': 1000,
    'road_cost_eur_per_km': 300000,
    'water_pipeline_eur_per_km': 50000,

    # Plant sizing
    'electrolyzer_capacity_mw': 100,
    'annual_hours': 8760,
    'availability': 0.92,
}

# ============================================================
# 3. FINANCING SCENARIOS
# ============================================================
scenarios = {
    'base': {'name': 'Base Case (Moderate)', 'wacc': 0.08},
    'derisked': {'name': 'De-risked (Policy Support)', 'wacc': 0.05},
    'high_risk': {'name': 'High Risk (No Support)', 'wacc': 0.12},
}
eu_wacc = 0.04

# ============================================================
# 4. LCOH CALCULATION
# ============================================================

def crf(wacc, n):
    if wacc == 0: return 1.0 / n
    return (wacc * (1 + wacc)**n) / ((1 + wacc)**n - 1)

def annualized_cost(capex_total, wacc, lifetime, opex_pct):
    return capex_total * crf(wacc, lifetime) + capex_total * opex_pct

def calc_lcoh(row, wacc, p, include_nh3_shipping=True):
    pv_pot = row['theo_pv']
    wind_pot = row['theo_wind']
    ocean_dist = row['ocean_dist_km']
    water_dist = row['waterbody_dist_km']
    grid_dist = row['grid_dist_km']
    road_dist = row['road_dist_km']
    lat, lon = row['lat'], row['lon']

    pv_share = pv_pot / (pv_pot + wind_pot) if (pv_pot + wind_pot) > 0 else 0.5
    pv_cf = 0.15 + pv_pot * 0.13
    wind_cf = 0.20 + wind_pot * 0.25

    # RE costs
    pv_mw = p['electrolyzer_capacity_mw'] * pv_share
    wind_mw = p['electrolyzer_capacity_mw'] * (1 - pv_share)
    pv_gen = pv_mw * pv_cf * 8760
    wind_gen = wind_mw * wind_cf * 8760
    pv_cost = annualized_cost(p['pv_capex_per_kw'] * pv_mw * 1000, wacc, p['pv_lifetime'], p['pv_opex_pct'])
    wind_cost = annualized_cost(p['wind_capex_per_kw'] * wind_mw * 1000, wacc, p['wind_lifetime'], p['wind_opex_pct'])
    total_gen = pv_gen + wind_gen
    lcoe = (pv_cost + wind_cost) / total_gen if total_gen > 0 else 1e6  # €/MWh

    elec_to_el = total_gen * p['availability']
    h2_kg_yr = (elec_to_el * 1000) / p['electrolyzer_efficiency']
    if h2_kg_yr <= 0: return np.nan, {}

    # 1. Electricity
    elec_per_kg = lcoe * p['electrolyzer_efficiency'] / 1000  # €/kg

    # 2. Electrolyzer
    el_capex = p['electrolyzer_capex_per_kw'] * p['electrolyzer_capacity_mw'] * 1000
    el_annual = annualized_cost(el_capex, wacc, p['electrolyzer_lifetime'], p['electrolyzer_opex_pct'])
    el_per_kg = el_annual / h2_kg_yr

    # 3. Battery
    batt_kwh = p['electrolyzer_capacity_mw'] * 1000 * p['battery_hours']
    batt_annual = annualized_cost(batt_kwh * p['battery_capex_per_kwh'], wacc, p['battery_lifetime'], p['battery_opex_pct'])
    batt_per_kg = batt_annual / h2_kg_yr

    # 4. Water
    water_unit = p['water_cost_per_m3'] if ocean_dist < 100 else p['water_inland_cost']
    water_infra = annualized_cost(water_dist * p['water_pipeline_eur_per_km'] * 0.1, wacc, 20, 0.02)
    water_per_kg = water_unit * p['water_per_kg_h2'] + water_infra / h2_kg_yr

    # 5. Infrastructure
    infra_capex = (grid_dist * p['grid_connection_eur_per_mw_km'] * p['electrolyzer_capacity_mw'] * 0.01 +
                   road_dist * p['road_cost_eur_per_km'] * 0.02 +
                   ocean_dist * 20000 * 0.05)
    infra_annual = annualized_cost(infra_capex, wacc, 20, 0.02)
    infra_per_kg = infra_annual / h2_kg_yr

    if include_nh3_shipping:
        # 6. NH3 synthesis
        nh3_capex = (h2_kg_yr / 1000) * p['nh3_capex_per_ton_h2_yr'] * 1000
        nh3_annual = annualized_cost(nh3_capex, wacc, p['nh3_lifetime'], p['nh3_opex_pct'])
        nh3_elec = p['nh3_elec_kwh_per_kg_h2'] * lcoe / 1000
        nh3_per_kg = nh3_annual / h2_kg_yr + nh3_elec

        # 7. Shipping
        dlat = np.radians(51.9 - lat)
        dlon = np.radians(4.5 - lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(51.9)) * np.sin(dlon/2)**2
        gc_km = 2 * 6371 * np.arcsin(np.sqrt(a))
        sea_km = gc_km * 1.3 + ocean_dist
        shipping_per_kg = sea_km * p['shipping_cost_per_ton_km'] + 2 * p['port_handling_cost_per_kg']

        # 8. Cracking
        crack_capex = 1000 * p['cracking_capex_per_ton_h2_yr'] * 1000
        crack_annual = annualized_cost(crack_capex, wacc, p['cracking_lifetime'], p['cracking_opex_pct'])
        crack_ref = 1000 * 1000 * 0.9
        crack_per_kg = crack_annual / crack_ref + p['cracking_heat_kwh_per_kg_h2'] * 0.04
    else:
        nh3_per_kg = shipping_per_kg = crack_per_kg = sea_km = 0

    total = elec_per_kg + el_per_kg + batt_per_kg + water_per_kg + infra_per_kg + nh3_per_kg + shipping_per_kg + crack_per_kg
    breakdown = {
        'electricity': elec_per_kg, 'electrolyzer': el_per_kg, 'battery': batt_per_kg,
        'water': water_per_kg, 'infrastructure': infra_per_kg,
        'ammonia_synthesis': nh3_per_kg, 'shipping': shipping_per_kg,
        'ammonia_cracking': crack_per_kg, 'total': total,
        'lcoe_mwh': lcoe, 'pv_cf': pv_cf, 'wind_cf': wind_cf,
        'h2_kg_yr': h2_kg_yr, 'sea_km': sea_km,
    }
    return total, breakdown

# ============================================================
# 5. RUN ALL SCENARIOS
# ============================================================
print("\nRunning LCOH calculations...")
results = {}

for sk, scen in scenarios.items():
    print(f"  {scen['name']} (WACC={scen['wacc']*100:.0f}%)")
    lcohs, bds = [], []
    for _, row in df.iterrows():
        l, b = calc_lcoh(row, scen['wacc'], params)
        lcohs.append(l); bds.append(b)
    sdf = df.copy()
    sdf['lcoh'] = lcohs
    for key in bds[0]: sdf[key] = [b[key] for b in bds]
    results[sk] = sdf
    print(f"    Mean={sdf['lcoh'].mean():.2f}  Min={sdf['lcoh'].min():.2f}  Max={sdf['lcoh'].max():.2f} €/kg")

print(f"  European Domestic (WACC={eu_wacc*100:.0f}%)")
eu_lcohs = []
for _, row in df.iterrows():
    l, _ = calc_lcoh(row, eu_wacc, params, include_nh3_shipping=False)
    eu_lcohs.append(l)
results['european'] = df.copy()
results['european']['lcoh'] = eu_lcohs
eu_mean = np.nanmean(eu_lcohs)
print(f"    Mean={eu_mean:.2f}  Min={np.nanmin(eu_lcohs):.2f}  Max={np.nanmax(eu_lcohs):.2f} €/kg")

# ============================================================
# 6. SAVE RESULTS
# ============================================================
print("\nSaving results...")
for sk, sdf in results.items():
    sdf.to_csv(f'outputs/lcoh_{sk}.csv', index=False)

summary_rows = []
for sk in list(scenarios.keys()) + ['european']:
    sdf = results[sk]
    label = scenarios.get(sk, {}).get('name', 'European Domestic')
    summary_rows.append({
        'scenario': label,
        'wacc': scenarios.get(sk, {}).get('wacc', eu_wacc),
        'mean': sdf['lcoh'].mean(), 'median': sdf['lcoh'].median(),
        'min': sdf['lcoh'].min(), 'p10': sdf['lcoh'].quantile(0.10),
        'p90': sdf['lcoh'].quantile(0.90), 'max': sdf['lcoh'].max(),
    })
pd.DataFrame(summary_rows).to_csv('outputs/scenario_summary.csv', index=False)

# ============================================================
# 7. FIGURES
# ============================================================
print("\nGenerating figures...")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# Fig 1: Spatial LCOH base case
fig, ax = plt.subplots(figsize=(12, 10))
africa.plot(ax=ax, color='#f0f0f0', edgecolor='#999', linewidth=0.5)
bdf = results['base']
geom = [Point(lo, la) for lo, la in zip(bdf['lon'], bdf['lat'])]
gdf = gpd.GeoDataFrame(bdf, geometry=geom, crs='EPSG:4326')
vmin, vmax = 3, 12
gdf.plot(ax=ax, column='lcoh', cmap='RdYlGn_r', vmin=vmin, vmax=vmax, markersize=40, alpha=0.85, edgecolor='none')
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin, vmax))
plt.colorbar(sm, ax=ax, shrink=0.7, label='LCOH (€/kg H₂)')
ax.set_xlim(-20, 55); ax.set_ylim(-40, 40)
ax.plot(4.5, 51.9, 'r*', ms=15, label='Rotterdam')
ax.legend(loc='upper left')
ax.set_title('African Green H₂ to Europe – Base Case (2030)')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
plt.savefig('report/images/spatial_lcoh_base.png'); plt.close()

# Fig 2: Scenario comparison maps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, sk in zip(axes, ['base', 'derisked', 'high_risk']):
    africa.plot(ax=ax, color='#f0f0f0', edgecolor='#999', lw=0.3)
    sdf = results[sk]
    geom = [Point(lo, la) for lo, la in zip(sdf['lon'], sdf['lat'])]
    gdf = gpd.GeoDataFrame(sdf, geometry=geom, crs='EPSG:4326')
    gdf.plot(ax=ax, column='lcoh', cmap='RdYlGn_r', vmin=2, vmax=14, markersize=25, alpha=0.85, edgecolor='none')
    ax.set_xlim(-20, 55); ax.set_ylim(-40, 40)
    ax.set_title(f"{scenarios[sk]['name']}\nMean={sdf['lcoh'].mean():.1f} €/kg")
    ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(2, 14))
plt.colorbar(sm, ax=axes, shrink=0.8, label='LCOH (€/kg H₂)')
plt.savefig('report/images/scenario_comparison_maps.png'); plt.close()

# Fig 3: Cost breakdown
fig, ax = plt.subplots(figsize=(12, 7))
comps = ['electricity', 'electrolyzer', 'battery', 'water', 'infrastructure', 'ammonia_synthesis', 'shipping', 'ammonia_cracking']
labels = ['Electricity', 'Electrolyzer', 'Battery', 'Water', 'Infrastructure', 'NH₃ Synthesis', 'Shipping', 'NH₃ Cracking']
colors_bar = ['#2196F3', '#FF9800', '#8BC34A', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548']
x = np.arange(len(comps)); w = 0.25
for i, sk in enumerate(['base', 'derisked', 'high_risk']):
    means = [results[sk][c].mean() for c in comps]
    ax.bar(x + i*w, means, w, label=scenarios[sk]['name'], color=colors_bar[i], alpha=0.85)
ax.set_xticks(x + w); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('€/kg H₂'); ax.set_title('LCOH Cost Breakdown by Scenario')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.savefig('report/images/cost_breakdown.png'); plt.close()

# Fig 4: LCOH distribution
fig, ax = plt.subplots(figsize=(10, 6))
for sk, col, lab in [('base', '#2196F3', 'Base'), ('derisked', '#4CAF50', 'De-risked'), ('high_risk', '#F44336', 'High Risk')]:
    ax.hist(results[sk]['lcoh'].dropna(), bins=15, alpha=0.5, color=col, label=lab, edgecolor='w')
ax.axvline(eu_mean, color='k', ls='--', lw=2, label=f'EU Domestic: {eu_mean:.1f} €/kg')
ax.set_xlabel('LCOH (€/kg H₂)'); ax.set_ylabel('Sites')
ax.set_title('LCOH Distribution by Scenario'); ax.legend(); ax.grid(alpha=0.3)
plt.savefig('report/images/lcoh_distribution.png'); plt.close()

# Fig 5: Competitiveness map
fig, ax = plt.subplots(figsize=(12, 10))
africa.plot(ax=ax, color='#f0f0f0', edgecolor='#999', lw=0.5)
bdf2 = results['base'].copy()
bdf2['eu_lcoh'] = results['european']['lcoh']
bdf2['advantage'] = bdf2['eu_lcoh'] - bdf2['lcoh']
geom = [Point(lo, la) for lo, la in zip(bdf2['lon'], bdf2['lat'])]
gdf = gpd.GeoDataFrame(bdf2, geometry=geom, crs='EPSG:4326')
gdf.plot(ax=ax, column='advantage', cmap='RdYlGn', vmin=-5, vmax=5, markersize=40, alpha=0.85, edgecolor='none')
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-5, 5))
plt.colorbar(sm, ax=ax, shrink=0.7, label='Cost Advantage vs EU (€/kg)\n+ = Africa cheaper')
ax.set_xlim(-20, 55); ax.set_ylim(-40, 40)
ax.set_title('African H₂ Cost Advantage vs European Domestic (Base Case)')
plt.savefig('report/images/competitiveness_map.png'); plt.close()

# Fig 6: WACC sensitivity
fig, ax = plt.subplots(figsize=(10, 6))
wacc_range = np.arange(0.03, 0.16, 0.005)
mean_lcoh_wacc = []
for w in wacc_range:
    vals = [calc_lcoh(row, w, params)[0] for _, row in df.iterrows()]
    mean_lcoh_wacc.append(np.nanmean(vals))
ax.plot(wacc_range*100, mean_lcoh_wacc, 'b-o', ms=4, lw=2)
for wv, col, lab in [(8, 'gray', 'Base'), (5, 'green', 'De-risked'), (12, 'red', 'High Risk')]:
    ax.axvline(wv, color=col, ls='--', alpha=0.7, label=f'{lab} ({wv}%)')
ax.axhline(eu_mean, color='k', ls=':', alpha=0.7, label=f'EU Domestic ({eu_mean:.1f})')
ax.set_xlabel('WACC (%)'); ax.set_ylabel('Mean LCOH (€/kg H₂)')
ax.set_title('LCOH Sensitivity to Financing Cost'); ax.legend(); ax.grid(alpha=0.3)
plt.savefig('report/images/wacc_sensitivity.png'); plt.close()

# Fig 7: Top 10 sites
fig, ax = plt.subplots(figsize=(12, 8))
africa.plot(ax=ax, color='#f0f0f0', edgecolor='#999', lw=0.5)
top10 = results['base'].nsmallest(10, 'lcoh')
geom = [Point(lo, la) for lo, la in zip(top10['lon'], top10['lat'])]
gdf = gpd.GeoDataFrame(top10, geometry=geom, crs='EPSG:4326')
gdf.plot(ax=ax, color='#E53935', markersize=120, edgecolor='k', lw=1)
for _, r in top10.iterrows():
    ax.annotate(f"{r['lcoh']:.1f}", (r['lon'], r['lat']),
                textcoords="offset points", xytext=(8, 5), fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8))
ax.set_xlim(-20, 55); ax.set_ylim(-40, 40)
ax.set_title('Top 10 Least-Cost African H₂ Sites (Base Case, 2030)')
plt.savefig('report/images/top10_sites.png'); plt.close()

# Fig 8: Shipping distance vs LCOH
fig, ax = plt.subplots(figsize=(10, 6))
bdf3 = results['base']
ax.scatter(bdf3['sea_km'], bdf3['lcoh'], c=bdf3['lcoh'], cmap='RdYlGn_r', s=50, alpha=0.8, edgecolor='k', lw=0.3)
ax.set_xlabel('Sea Distance to Rotterdam (km)'); ax.set_ylabel('LCOH (€/kg H₂)')
ax.set_title('Shipping Distance vs LCOH (Base Case)'); ax.grid(alpha=0.3)
plt.savefig('report/images/shipping_vs_lcoh.png'); plt.close()

# ============================================================
# 8. PRINT SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for sk in ['base', 'derisked', 'high_risk']:
    sdf = results[sk]
    print(f"\n{scenarios[sk]['name']}:")
    print(f"  Mean LCOH: {sdf['lcoh'].mean():.2f} €/kg")
    print(f"  Best site: {sdf.loc[sdf['lcoh'].idxmin(), 'hex_id']} at {sdf['lcoh'].min():.2f} €/kg")
    print(f"  P10-P90: {sdf['lcoh'].quantile(0.10):.2f} – {sdf['lcoh'].quantile(0.90):.2f} €/kg")

print(f"\nEuropean Domestic: {eu_mean:.2f} €/kg (mean)")
for sk in ['base', 'derisked', 'high_risk']:
    bdf_c = results[sk].copy()
    bdf_c['eu_lcoh'] = results['european']['lcoh']
    n_comp = (bdf_c['lcoh'] < bdf_c['eu_lcoh']).sum()
    print(f"{scenarios[sk]['name']}: {n_comp}/{len(bdf_c)} sites competitive vs EU")
print("\nDone!")
