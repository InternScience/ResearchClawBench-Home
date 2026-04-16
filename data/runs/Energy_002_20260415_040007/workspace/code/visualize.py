#!/usr/bin/env python3
"""
Visualization script for the Geospatial LCOH Model.
Generates all figures for the research report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json
import os

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.figsize': (10, 7),
})

# Load data
results_df = pd.read_csv("outputs/lcoh_results.csv")
eu_df = pd.read_csv("outputs/european_lcoh.csv")
wacc_sens = pd.read_csv("outputs/wacc_sensitivity.csv")

with open("outputs/key_metrics.json") as f:
    key_metrics = json.load(f)

# Load shapefile
world = gpd.read_file("data/africa_map/ne_10m_admin_0_countries.shp")
africa = world[world['CONTINENT'] == 'Africa']

# Create GeoDataFrame from results
from shapely.geometry import Point
base_results = results_df[results_df['scenario'] == 'base'].copy()
geometry = [Point(xy) for xy in zip(base_results['lon'], base_results['lat'])]
gdf = gpd.GeoDataFrame(base_results, geometry=geometry, crs='EPSG:4326')

os.makedirs("report/images", exist_ok=True)

# ============================================================
# Figure 1: Geospatial LCOH Map
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

scenarios = [("base", "Base (8% WACC)"), ("derisked", "De-risked (5% WACC)"), ("high_risk", "High Risk (12% WACC)")]

for idx, (scenario, title) in enumerate(scenarios):
    ax = axes[idx]
    scenario_data = results_df[results_df['scenario'] == scenario].copy()
    geo = [Point(xy) for xy in zip(scenario_data['lon'], scenario_data['lat'])]
    sgdf = gpd.GeoDataFrame(scenario_data, geometry=geo, crs='EPSG:4326')
    
    africa.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
    
    vmin = 3.5
    vmax = 7.0
    sgdf.plot(ax=ax, column='cost_delivered', cmap='RdYlGn_r', 
              markersize=60, edgecolor='black', linewidth=0.5,
              vmin=vmin, vmax=vmax, legend=(idx==2))
    
    ax.set_xlim(8, 28)
    ax.set_ylim(-32, -14)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Longitude')
    if idx == 0:
        ax.set_ylabel('Latitude')

# Add colorbar
sm = ScalarMappable(cmap='RdYlGn_r', norm=Normalize(vmin=3.5, vmax=7.0))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Delivered LCOH ($/kg H₂)')

fig.suptitle('Delivered Cost of Green Hydrogen from Africa to Europe via Ammonia Shipping', 
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/lcoh_map.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 1 saved: lcoh_map.png")

# ============================================================
# Figure 2: Cost Breakdown by Component
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Use base scenario, sorted by delivered cost
base = results_df[results_df['scenario'] == 'base'].sort_values('cost_delivered')

components = ['cost_electricity', 'cost_electrolyzer_capex', 'cost_electrolyzer_opex',
              'cost_storage', 'cost_water', 'cost_infrastructure',
              'cost_ammonia_conversion', 'cost_shipping', 'cost_cracking']
labels = ['RE Electricity', 'Electrolyzer CAPEX', 'Electrolyzer OPEX', 
          'H₂ Storage', 'Water', 'Infrastructure',
          'NH₃ Conversion', 'Shipping', 'NH₃ Cracking']
colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#00BCD4', '#795548',
          '#9C27B0', '#F44336', '#FF5722']

bottom = np.zeros(len(base))
for comp, label, color in zip(components, labels, colors):
    values = base[comp].values
    ax.bar(range(len(base)), values, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.3)
    bottom += values

# Add European reference line
eu_lcoh = eu_df[eu_df['Unnamed: 0']=='eu_base']['cost_production'].values[0]
ax.axhline(y=eu_lcoh, color='navy', linestyle='--', linewidth=2, label=f'European domestic LCOH (${eu_lcoh:.2f}/kg)')

ax.set_xlabel('Production Site (ranked by delivered cost)')
ax.set_ylabel('Cost ($/kg H₂)')
ax.set_title('Delivered Cost Breakdown by Supply Chain Component (Base Scenario, 8% WACC)', fontweight='bold')
ax.legend(loc='upper left', ncol=2, fontsize=8)
ax.set_xticks(range(0, len(base), 5))
ax.set_xticklabels([base.iloc[i]['hex_id'] for i in range(0, len(base), 5)], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig('report/images/cost_breakdown.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 saved: cost_breakdown.png")

# ============================================================
# Figure 3: Scenario Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

scenario_data = []
for scenario, label, color in [("base", "Base (8% WACC)", "#2196F3"), 
                                 ("derisked", "De-risked (5% WACC)", "#4CAF50"),
                                 ("high_risk", "High Risk (12% WACC)", "#F44336")]:
    s_data = results_df[results_df['scenario'] == scenario]['cost_delivered']
    scenario_data.append({
        'scenario': label,
        'min': s_data.min(),
        'mean': s_data.mean(),
        'max': s_data.max(),
        'p25': s_data.quantile(0.25),
        'p75': s_data.quantile(0.75),
    })

x = range(len(scenario_data))
means = [d['mean'] for d in scenario_data]
mins = [d['min'] for d in scenario_data]
maxs = [d['max'] for d in scenario_data]
p25s = [d['p25'] for d in scenario_data]
p75s = [d['p75'] for d in scenario_data]

colors = ['#2196F3', '#4CAF50', '#F44336']
bars = ax.bar(x, means, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8, width=0.5)

# Error bars for range
for i in range(len(scenario_data)):
    ax.plot([i, i], [mins[i], maxs[i]], color='black', linewidth=2, zorder=5)
    ax.plot([i-0.1, i+0.1], [mins[i], mins[i]], color='black', linewidth=2, zorder=5)
    ax.plot([i-0.1, i+0.1], [maxs[i], maxs[i]], color='black', linewidth=2, zorder=5)

# European reference
eu_base = eu_df[eu_df['Unnamed: 0']=='eu_base']['cost_production'].values[0]
eu_low = eu_df[eu_df['Unnamed: 0']=='eu_low']['cost_production'].values[0]
eu_high = eu_df[eu_df['Unnamed: 0']=='eu_high']['cost_production'].values[0]

ax.axhline(y=eu_base, color='navy', linestyle='--', linewidth=2, label=f'European domestic LCOH (5% WACC): ${eu_base:.2f}/kg')
ax.fill_between([-0.5, 2.5], eu_low, eu_high, alpha=0.15, color='navy', label=f'European LCOH range (4-6% WACC)')

ax.set_xticks(x)
ax.set_xticklabels([d['scenario'] for d in scenario_data])
ax.set_ylabel('Delivered LCOH ($/kg H₂)')
ax.set_title('Delivered Cost of African Green H₂ to Europe: Scenario Comparison', fontweight='bold')
ax.legend(loc='upper left')

# Add value labels
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
            f'${means[i]:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/scenario_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 saved: scenario_comparison.png")

# ============================================================
# Figure 4: Competitiveness Comparison (African vs European)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# For each scenario, show distribution of African delivered costs vs European
scenarios_plot = [("base", "Base (8%)", "#2196F3"), 
                  ("derisked", "De-risked (5%)", "#4CAF50"),
                  ("high_risk", "High Risk (12%)", "#F44336")]

positions = []
data_sets = []
colors_violin = []
labels_violin = []

pos = 0
for scenario, label, color in scenarios_plot:
    s_data = results_df[results_df['scenario'] == scenario]['cost_delivered']
    positions.append(pos)
    data_sets.append(s_data.values)
    colors_violin.append(color)
    labels_violin.append(f'African\n{label}')
    pos += 1

# Add European points
eu_vals = [eu_df[eu_df['Unnamed: 0']==s]['cost_production'].values[0] 
           for s in ['eu_low', 'eu_base', 'eu_high']]
positions.append(pos)
data_sets.append(np.array(eu_vals))
colors_violin.append('navy')
labels_violin.append('European\nDomestic')

parts = ax.violinplot(data_sets, positions=positions, showmeans=True, showmedians=True, widths=0.7)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_violin[i])
    pc.set_alpha(0.6)

parts['cmeans'].set_color('black')
parts['cmedians'].set_color('red')

ax.set_xticks(positions)
ax.set_xticklabels(labels_violin)
ax.set_ylabel('LCOH ($/kg H₂)')
ax.set_title('African Delivered H₂ Cost vs European Domestic H₂ Production', fontweight='bold')

# Add reference line for European base
ax.axhline(y=eu_base, color='navy', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/competitiveness.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 4 saved: competitiveness.png")

# ============================================================
# Figure 5: WACC Sensitivity Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(wacc_sens['wacc'] * 100, wacc_sens['cost_delivered'], 'o-', color='#2196F3', 
        linewidth=2, markersize=5, label='Total Delivered Cost')
ax.plot(wacc_sens['wacc'] * 100, wacc_sens['cost_production'], 's-', color='#4CAF50', 
        linewidth=2, markersize=5, label='Production Cost')
ax.plot(wacc_sens['wacc'] * 100, wacc_sens['cost_ammonia_conversion'], '^-', color='#9C27B0', 
        linewidth=1.5, markersize=4, label='NH₃ Conversion')
ax.plot(wacc_sens['wacc'] * 100, wacc_sens['cost_shipping'], 'D-', color='#F44336', 
        linewidth=1.5, markersize=4, label='Shipping')
ax.plot(wacc_sens['wacc'] * 100, wacc_sens['cost_cracking'], 'v-', color='#FF9800', 
        linewidth=1.5, markersize=4, label='NH₃ Cracking')

# European reference band
ax.axhspan(eu_low, eu_high, alpha=0.15, color='navy', label=f'European LCOH range (4-6% WACC)')
ax.axhline(y=eu_base, color='navy', linestyle='--', linewidth=1.5, alpha=0.7, label=f'European LCOH (5% WACC)')

# Mark de-risked scenario
ax.axvline(x=5, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
ax.annotate('De-risked\nscenario', xy=(5, wacc_sens[wacc_sens['wacc']==0.05]['cost_delivered'].values[0]),
            xytext=(6.5, 4.0), fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

ax.axvline(x=8, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
ax.annotate('Base\nscenario', xy=(8, wacc_sens[wacc_sens['wacc']==0.08]['cost_delivered'].values[0]),
            xytext=(9.5, 4.5), fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'))

ax.set_xlabel('WACC (%)')
ax.set_ylabel('Cost ($/kg H₂)')
ax.set_title('WACC Sensitivity: Delivered Cost of African Green H₂ to Europe (Best Site)', fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/wacc_sensitivity.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 5 saved: wacc_sensitivity.png")

# ============================================================
# Figure 6: Resource Potential vs Delivered Cost
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

base = results_df[results_df['scenario'] == 'base']

# PV potential vs cost
ax = axes[0]
scatter = ax.scatter(base['theo_pv'], base['cost_delivered'], 
                     c=base['cost_infrastructure'], cmap='YlOrRd',
                     s=80, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Theoretical PV Potential (normalized)')
ax.set_ylabel('Delivered LCOH ($/kg H₂)')
ax.set_title('PV Potential vs Delivered Cost', fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Infrastructure Cost ($/kg H₂)')

# Wind potential vs cost
ax = axes[1]
scatter = ax.scatter(base['theo_wind'], base['cost_delivered'],
                     c=base['cost_infrastructure'], cmap='YlOrRd',
                     s=80, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Theoretical Wind Potential (normalized)')
ax.set_ylabel('Delivered LCOH ($/kg H₂)')
ax.set_title('Wind Potential vs Delivered Cost', fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Infrastructure Cost ($/kg H₂)')

fig.suptitle('Renewable Resource Potential and Delivered Cost (Base Scenario)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/resource_vs_cost.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 6 saved: resource_vs_cost.png")

# ============================================================
# Figure 7: Production vs Delivered Cost Decomposition
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Average cost breakdown across scenarios
scenarios_list = ["base", "derisked", "high_risk"]
scenario_labels = ["Base (8% WACC)", "De-risked (5% WACC)", "High Risk (12% WACC)"]

prod_components = ['cost_electrolysis', 'cost_storage', 'cost_water', 'cost_infrastructure']
delivery_adds = ['cost_ammonia_conversion', 'cost_shipping', 'cost_cracking']

x = np.arange(len(scenarios_list))
width = 0.35

# Production costs
prod_bottom = np.zeros(len(scenarios_list))
for comp, label, color in zip(prod_components,
                               ['Electrolysis', 'Storage', 'Water', 'Infrastructure'],
                               ['#2196F3', '#FF9800', '#00BCD4', '#795548']):
    vals = [results_df[results_df['scenario']==s][comp].mean() for s in scenarios_list]
    ax.bar(x - width/2, vals, width, bottom=prod_bottom, label=f'Production: {label}', color=color, edgecolor='white')
    prod_bottom += vals

# Delivery additions
del_bottom = np.zeros(len(scenarios_list))
for comp, label, color in zip(delivery_adds,
                               ['NH₃ Conversion', 'Shipping', 'NH₃ Cracking'],
                               ['#9C27B0', '#F44336', '#FF5722']):
    vals = [results_df[results_df['scenario']==s][comp].mean() for s in scenarios_list]
    ax.bar(x + width/2, vals, width, bottom=del_bottom, label=f'Delivery: {label}', color=color, edgecolor='white')
    del_bottom += vals

# European reference
ax.axhline(y=eu_base, color='navy', linestyle='--', linewidth=2, label=f'European domestic LCOH')

ax.set_xticks(x)
ax.set_xticklabels(scenario_labels)
ax.set_ylabel('Cost ($/kg H₂)')
ax.set_title('Production Cost vs Delivery Chain Cost by Scenario', fontweight='bold')
ax.legend(loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('report/images/production_vs_delivery.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 7 saved: production_vs_delivery.png")

print("\nAll figures generated successfully!")
