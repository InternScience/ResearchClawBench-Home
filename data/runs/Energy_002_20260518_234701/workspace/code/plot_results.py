"""
Generate figures for the LCOH analysis report.
"""
import os
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Paths
out_dir = 'outputs'
img_dir = 'report/images'
os.makedirs(img_dir, exist_ok=True)

# Load data
results = pd.read_csv(os.path.join(out_dir, 'lcoh_results.csv'))
with open(os.path.join(out_dir, 'summary.json')) as f:
    summary = json.load(f)
# Merge original input columns back in
raw = pd.read_csv('data/hex_final_NA_min.csv')
results = results.merge(raw[['hex_id','theo_pv','theo_wind']], on='hex_id', how='left')

eu_benchmark = summary['european_benchmark_eur_per_kg']
scenarios = {
    'strong_de-risking': 'Strong de-risking (5%)',
    'moderate_de-risking': 'Moderate de-risking (8%)',
    'no_de-risking': 'No de-risking (12%)',
}

# ---------------------------------------------------------------------------
# Figure 1: Maps of delivered cost per scenario
# ---------------------------------------------------------------------------
world = gpd.read_file('data/africa_map/ne_10m_admin_0_countries.shp')
africa = world[world['CONTINENT'] == 'Africa']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (scen, title) in zip(axes, scenarios.items()):
    africa.boundary.plot(ax=ax, linewidth=0.5, color='grey')
    col = f'delivered_{scen}'
    vmin = results[col].min()
    vmax = results[col].max()
    sc = ax.scatter(results['lon'], results['lat'], c=results[col],
                    cmap='RdYlGn_r', s=80, edgecolor='k', linewidth=0.3,
                    vmin=3.0, vmax=6.0)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(results['lon'].min() - 2, results['lon'].max() + 2)
    ax.set_ylim(results['lat'].min() - 2, results['lat'].max() + 2)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Delivered cost (€/kg H₂)')
    # annotate benchmark
    ax.axhline(eu_benchmark, color='blue', linestyle='--', linewidth=1)
    ax.text(0.02, 0.02, f'EU benchmark = {eu_benchmark:.2f} €/kg', transform=ax.transAxes,
            fontsize=9, color='blue', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig1_maps_delivered_cost.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Cost stack for least-cost site under each scenario
# ---------------------------------------------------------------------------
# Recompute component costs for the least-cost hex in each scenario
# We'll use the model logic inline for transparency
from lcoh_model import lcoe, lcoh_production, water_cost, DOWNSTREAM_COST, NH3_SYNTHESIS, NH3_SHIPPING, NH3_CRACKING, LOCAL_TRUCK, PV_PR, WIND_PR, PV_CAPEX, PV_OPEX, PV_LIFETIME, WIND_CAPEX, WIND_OPEX, WIND_LIFETIME, ELEC_CAPEX, ELEC_OPEX_RATE, ELEC_LIFETIME, ELEC_EFF

fig, ax = plt.subplots(figsize=(10, 6))
labels = []
bottoms = []
width = 0.6
x = np.arange(len(scenarios))

# components
components = ['Renewable electricity', 'Electrolyzer CAPEX/OPEX', 'Water', 'Ammonia synthesis', 'Shipping', 'Cracking', 'Local truck']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

stack = {c: [] for c in components}

for scen in scenarios:
    r = 0.05 if scen == 'strong_de-risking' else (0.08 if scen == 'moderate_de-risking' else 0.12)
    row = results.loc[results[f'delivered_{scen}'].idxmin()]
    cf_pv = row['theo_pv'] * PV_PR
    cf_wind = row['theo_wind'] * WIND_PR
    lcoe_pv = lcoe(PV_CAPEX, PV_OPEX, cf_pv, r, PV_LIFETIME)
    lcoe_wind = lcoe(WIND_CAPEX, WIND_OPEX, cf_wind, r, WIND_LIFETIME)
    best_lcoe = min(lcoe_pv, lcoe_wind)
    best_cf = cf_pv if lcoe_pv < lcoe_wind else cf_wind

    # Renewable electricity cost
    elec_energy_cost = best_lcoe * ELEC_EFF / 1000
    # Electrolyzer cost
    annual_h2 = best_cf * 8760 / ELEC_EFF
    elec_capex_annual = ELEC_CAPEX / (( (1+r)**ELEC_LIFETIME - 1) / ((1+r)**ELEC_LIFETIME * r))
    elec_opex_annual = ELEC_OPEX_RATE * ELEC_CAPEX
    elec_cost_per_kg = (elec_capex_annual + elec_opex_annual) / annual_h2

    w_cost = water_cost(row, best_lcoe)

    stack['Renewable electricity'].append(elec_energy_cost)
    stack['Electrolyzer CAPEX/OPEX'].append(elec_cost_per_kg)
    stack['Water'].append(w_cost)
    stack['Ammonia synthesis'].append(NH3_SYNTHESIS)
    stack['Shipping'].append(NH3_SHIPPING)
    stack['Cracking'].append(NH3_CRACKING)
    stack['Local truck'].append(LOCAL_TRUCK)

bottom = np.zeros(len(scenarios))
for comp, color in zip(components, colors):
    vals = np.array(stack[comp])
    ax.bar(x, vals, width, bottom=bottom, label=comp, color=color)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([scenarios[s] for s in scenarios], rotation=15, ha='right')
ax.set_ylabel('Cost (€/kg H₂)')
ax.set_title('Cost stack for least-cost African site per scenario (2030)')
ax.axhline(eu_benchmark, color='blue', linestyle='--', linewidth=2, label=f'EU benchmark ({eu_benchmark:.2f} €/kg)')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 6)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig2_cost_stack.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Sensitivity – delivered cost vs WACC for best / median / worst site
# ---------------------------------------------------------------------------
waccs = np.linspace(0.03, 0.15, 50)
# identify best, median, worst hex IDs based on moderate scenario
moderate = results['delivered_moderate_de-risking'].values
best_idx = np.argmin(moderate)
median_idx = np.argsort(moderate)[len(moderate)//2]
worst_idx = np.argmax(moderate)

fig, ax = plt.subplots(figsize=(9, 6))
for idx, label, color in [(best_idx, 'Best site', 'green'), (median_idx, 'Median site', 'orange'), (worst_idx, 'Worst site', 'red')]:
    row = results.iloc[idx]
    costs = []
    for r in waccs:
        cf_pv = row['theo_pv'] * PV_PR
        cf_wind = row['theo_wind'] * WIND_PR
        lcoe_pv = lcoe(PV_CAPEX, PV_OPEX, cf_pv, r, PV_LIFETIME)
        lcoe_wind = lcoe(WIND_CAPEX, WIND_OPEX, cf_wind, r, WIND_LIFETIME)
        best_lcoe = min(lcoe_pv, lcoe_wind)
        best_cf = cf_pv if lcoe_pv < lcoe_wind else cf_wind
        prod = lcoh_production(best_lcoe, best_cf, r, ELEC_CAPEX, ELEC_OPEX_RATE, ELEC_LIFETIME, ELEC_EFF)
        prod += water_cost(row, best_lcoe)
        costs.append(prod + DOWNSTREAM_COST)
    ax.plot(waccs*100, costs, label=label, color=color, linewidth=2)

ax.axhline(eu_benchmark, color='blue', linestyle='--', linewidth=2, label=f'EU benchmark ({eu_benchmark:.2f} €/kg)')
ax.set_xlabel('WACC (%)')
ax.set_ylabel('Delivered cost (€/kg H₂)')
ax.set_title('Sensitivity of delivered African H₂ cost to financing cost (WACC)')
ax.legend()
ax.set_xlim(3, 15)
ax.set_ylim(2.5, 7)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig3_wacc_sensitivity.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Competitiveness – number of sites below EU benchmark per scenario
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
competitive = [summary['scenarios'][s]['competitive_sites'] for s in scenarios]
ax.bar(range(len(scenarios)), competitive, color=['#2ca02c', '#ffcc00', '#d62728'], edgecolor='black')
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels([scenarios[s] for s in scenarios], rotation=15, ha='right')
ax.set_ylabel('Number of competitive sites')
ax.set_title('Count of African sites with delivered cost < European benchmark')
ax.set_ylim(0, len(results))
for i, v in enumerate(competitive):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig4_competitiveness.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Delivered cost distribution (boxplot) per scenario vs EU benchmark
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
data_for_box = [results[f'delivered_{s}'].values for s in scenarios]
bp = ax.boxplot(data_for_box, labels=[scenarios[s] for s in scenarios], patch_artist=True)
colors_box = ['#2ca02c', '#ffcc00', '#d62728']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.axhline(eu_benchmark, color='blue', linestyle='--', linewidth=2, label=f'EU benchmark ({eu_benchmark:.2f} €/kg)')
ax.set_ylabel('Delivered cost (€/kg H₂)')
ax.set_title('Distribution of delivered African H₂ costs by financing scenario')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig5_boxplot.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Figure 6: Scatter – delivered cost vs ocean distance (moderate scenario)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
scen = 'moderate_de-risking'
ax.scatter(results['ocean_dist_km'], results[f'delivered_{scen}'],
           c=results['theo_pv'], cmap='YlOrRd', s=80, edgecolor='k', linewidth=0.3)
ax.set_xlabel('Distance to ocean (km)')
ax.set_ylabel('Delivered cost (€/kg H₂)')
ax.set_title(f'Delivered cost vs. port distance ({scenarios[scen]})')
ax.axhline(eu_benchmark, color='blue', linestyle='--', linewidth=1)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('PV capacity factor (practical)')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig6_distance_scatter.png'), dpi=300)
plt.close()

print("Figures saved to", img_dir)
