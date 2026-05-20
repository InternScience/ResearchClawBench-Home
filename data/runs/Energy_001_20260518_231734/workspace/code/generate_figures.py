"""
Generate all figures for the GB power system dispatch analysis report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import networkx as nx
from matplotlib.patches import Circle
import os

# Setup
os.makedirs('report/images', exist_ok=True)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 9

# Load data
buses = pd.read_csv('data/buses.csv')
links = pd.read_csv('data/links.csv')
generators = pd.read_csv('data/generators.csv')
storage = pd.read_csv('data/storage.csv')
demand = pd.read_csv('data/demand.csv')
wind_cf = pd.read_csv('data/wind_cf.csv')

with open('outputs/scenario_summary.json') as f:
    scenario_summary = json.load(f)

base_hourly = pd.read_csv('outputs/base_case_hourly.csv')

bus_names = list(buses['name'].values)
n_hours = len(demand)
hours = np.arange(1, n_hours + 1)

# ========================================================================
# Figure 1: Network Topology Map
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Draw links
for _, l in links.iterrows():
    b0 = buses[buses['name'] == l['bus0']].iloc[0]
    b1 = buses[buses['name'] == l['bus1']].iloc[0]
    color = '#2ca02c' if l['p_nom'] >= 4000 else '#ff7f0e'
    lw = 2 if l['p_nom'] >= 4000 else 1.5
    ax.plot([b0['x'], b1['x']], [b0['y'], b1['y']], color=color, lw=lw, alpha=0.7, zorder=1)
    mid_x = (b0['x'] + b1['x']) / 2
    mid_y = (b0['y'] + b1['y']) / 2
    ax.text(mid_x, mid_y, f"{l['p_nom']:.0f}", fontsize=6, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

# Draw buses
colors = []
sizes = []
for _, b in buses.iterrows():
    wind_cap = generators[(generators['bus'] == b['name']) & (generators['carrier'] == 'onshore wind')]['p_nom'].sum()
    gas_cap = generators[(generators['bus'] == b['name']) & (generators['carrier'] == 'gas')]['p_nom'].sum()
    nuc_cap = generators[(generators['bus'] == b['name']) & (generators['carrier'] == 'nuclear')]['p_nom'].sum()
    
    if nuc_cap > 0:
        colors.append('#d62728')
    elif wind_cap > 5000:
        colors.append('#1f77b4')
    else:
        colors.append('#9467bd')
    
    total_demand = demand[b['name']].sum()
    sizes.append(50 + total_demand / 500)

scatter = ax.scatter(buses['x'], buses['y'], c=colors, s=sizes, edgecolors='black', linewidths=0.5, zorder=2)

for _, b in buses.iterrows():
    ax.text(b['x'], b['y'] + 0.3, b['name'], fontsize=7, ha='center', fontweight='bold')

ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title('GB 20-Bus Test System Network Topology\n(Line labels: nominal capacity in MW)', fontsize=11)
ax.set_aspect('equal')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='Major wind hub'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='Nuclear site'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=8, label='Other bus'),
    Line2D([0], [0], color='#2ca02c', lw=2, label='High-capacity link (≥4000 MW)'),
    Line2D([0], [0], color='#ff7f0e', lw=1.5, label='Lower-capacity link (<4000 MW)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/figure1_network_topology.png')
plt.close()
print('Saved figure1_network_topology.png')

# ========================================================================
# Figure 2: Weekly Demand and Wind Profiles
# ========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Total demand
total_demand = demand[bus_names].sum(axis=1)
axes[0].fill_between(hours, total_demand, alpha=0.3, color='#d62728', label='Total demand')
axes[0].plot(hours, total_demand, color='#d62728', lw=1)
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('System Demand and Renewable Availability (One Week)', fontsize=11)
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, max(total_demand) * 1.1)

# Total wind potential
total_wind_pot = np.zeros(n_hours)
for _, g in generators[generators['carrier'] == 'onshore wind'].iterrows():
    total_wind_pot += g['p_nom'] * wind_cf[g['bus']].values
axes[1].fill_between(hours, total_wind_pot, alpha=0.3, color='#1f77b4', label='Wind potential')
axes[1].plot(hours, total_wind_pot, color='#1f77b4', lw=1)
axes[1].set_ylabel('Power (MW)')
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, max(total_wind_pot) * 1.1)

# Net load (demand - wind - nuclear)
total_nuc_cap = generators[generators['carrier'] == 'nuclear']['p_nom'].sum()
net_load = total_demand - total_wind_pot - total_nuc_cap
axes[2].fill_between(hours, net_load, alpha=0.3, color='#ff7f0e', label='Net load (demand − wind − nuclear)')
axes[2].plot(hours, net_load, color='#ff7f0e', lw=1)
axes[2].axhline(y=generators[generators['carrier'] == 'gas']['p_nom'].sum(), color='green', ls='--', lw=1, label='Total gas capacity')
axes[2].set_ylabel('Power (MW)')
axes[2].set_xlabel('Hour')
axes[2].legend(loc='upper right')
axes[2].set_ylim(min(net_load) * 1.1, max(net_load) * 1.1)

plt.tight_layout()
plt.savefig('report/images/figure2_demand_wind_profiles.png')
plt.close()
print('Saved figure2_demand_wind_profiles.png')

# ========================================================================
# Figure 3: Base Case Dispatch Stack
# ========================================================================
fig, ax = plt.subplots(figsize=(12, 5))

y1 = base_hourly['nuclear_MW'].values
y2 = y1 + base_hourly['gas_MW'].values
y3 = y2 + base_hourly['wind_MW'].values
y4 = y3 + base_hourly['storage_discharge_MW'].values - base_hourly['storage_charge_MW'].values

ax.fill_between(hours, 0, y1, label='Nuclear', color='#d62728', alpha=0.8)
ax.fill_between(hours, y1, y2, label='Gas', color='#ff7f0e', alpha=0.8)
ax.fill_between(hours, y2, y3, label='Wind', color='#1f77b4', alpha=0.8)
net_storage = base_hourly['storage_discharge_MW'].values - base_hourly['storage_charge_MW'].values
ax.fill_between(hours, y3, y3 + net_storage, where=(net_storage > 0), label='Storage discharge', color='#2ca02c', alpha=0.8)
ax.fill_between(hours, y3, y3 + net_storage, where=(net_storage < 0), label='Storage charge', color='#9467bd', alpha=0.8)

ax.plot(hours, base_hourly['demand_MW'].values, color='black', lw=1.5, label='Demand')
ax.plot(hours, base_hourly['unserved_MW'].values + base_hourly['demand_MW'].values, color='gray', ls='--', lw=1, label='Demand + unserved')

ax.set_xlabel('Hour')
ax.set_ylabel('Power (MW)')
ax.set_title('Base Case Hourly Dispatch Stack', fontsize=11)
ax.legend(loc='upper left', ncol=3, fontsize=8)
ax.set_xlim(1, n_hours)
plt.tight_layout()
plt.savefig('report/images/figure3_dispatch_stack.png')
plt.close()
print('Saved figure3_dispatch_stack.png')

# ========================================================================
# Figure 4: Scenario Comparison - Generation Mix and Costs
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scenarios = list(scenario_summary.keys())
gen_data = {
    'Wind': [scenario_summary[s]['wind_generation_MWh'] / 1e3 for s in scenarios],
    'Gas': [scenario_summary[s]['gas_generation_MWh'] / 1e3 for s in scenarios],
    'Nuclear': [scenario_summary[s]['nuclear_generation_MWh'] / 1e3 for s in scenarios],
    'Unserved': [scenario_summary[s]['unserved_load_MWh'] / 1e3 for s in scenarios],
}

x = np.arange(len(scenarios))
width = 0.6
bottom = np.zeros(len(scenarios))
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#8c564b']
labels = ['Wind', 'Gas', 'Nuclear', 'Unserved load']

for i, (label, values) in enumerate(zip(labels, [gen_data['Wind'], gen_data['Gas'], gen_data['Nuclear'], gen_data['Unserved']])):
    axes[0].bar(x, values, width, label=label, bottom=bottom, color=colors[i], alpha=0.85)
    bottom += values

axes[0].set_xticks(x)
axes[0].set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
axes[0].set_ylabel('Energy (GWh)')
axes[0].set_title('Weekly Generation Mix by Scenario', fontsize=11)
axes[0].legend(loc='upper right', fontsize=8)

# Costs
costs = [scenario_summary[s]['total_cost_GBP'] / 1e6 for s in scenarios]
axes[1].bar(x, costs, width=0.5, color='#2ca02c', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
axes[1].set_ylabel('Total Cost (M£)')
axes[1].set_title('Total System Cost by Scenario', fontsize=11)

# Add value labels
for i, v in enumerate(costs):
    axes[1].text(i, v + max(costs)*0.01, f'{v:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('report/images/figure4_scenario_comparison.png')
plt.close()
print('Saved figure4_scenario_comparison.png')

# ========================================================================
# Figure 5: Storage Operation Profile (Base Case)
# ========================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].fill_between(hours, base_hourly['storage_soc_MWh'].values, alpha=0.4, color='#2ca02c')
axes[0].plot(hours, base_hourly['storage_soc_MWh'].values, color='#2ca02c', lw=1.5)
axes[0].set_ylabel('SOC (MWh)')
axes[0].set_title('Pumped Hydro Storage Operation (Base Case)', fontsize=11)

axes[1].bar(hours, base_hourly['storage_charge_MW'].values, color='#9467bd', alpha=0.7, label='Charge')
axes[1].bar(hours, -base_hourly['storage_discharge_MW'].values, color='#2ca02c', alpha=0.7, label='Discharge')
axes[1].axhline(y=0, color='black', lw=0.5)
axes[1].set_ylabel('Power (MW)')
axes[1].legend(loc='upper right')

# Net storage contribution to system
net_storage = base_hourly['storage_discharge_MW'].values - base_hourly['storage_charge_MW'].values
axes[2].bar(hours, net_storage, color='#17becf', alpha=0.7)
axes[2].axhline(y=0, color='black', lw=0.5)
axes[2].set_ylabel('Net injection (MW)')
axes[2].set_xlabel('Hour')

plt.tight_layout()
plt.savefig('report/images/figure5_storage_operation.png')
plt.close()
print('Saved figure5_storage_operation.png')

# ========================================================================
# Figure 6: Unserved Load and Curtailment Analysis
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Unserved load duration curve for all scenarios
for sname in scenarios:
    # We only have summary data; need to reload from saved files if we want hourly
    # For now, just plot base case
    pass

# Base case unserved load duration curve
unserved_sorted = np.sort(base_hourly['unserved_MW'].values)[::-1]
axes[0].plot(hours, unserved_sorted, color='#d62728', lw=2, label='Base Case')
axes[0].fill_between(hours, unserved_sorted, alpha=0.3, color='#d62728')
axes[0].set_xlabel('Hour (sorted by unserved load)')
axes[0].set_ylabel('Unserved Load (MW)')
axes[0].set_title('Unserved Load Duration Curve (Base Case)', fontsize=11)
axes[0].legend()

# Curtailment vs wind potential
axes[1].scatter(base_hourly['wind_MW'].values, base_hourly['curtailment_MW'].values, 
                alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1].set_xlabel('Wind Dispatch (MW)')
axes[1].set_ylabel('Curtailment (MW)')
axes[1].set_title('Curtailment vs Wind Dispatch (Base Case)', fontsize=11)

plt.tight_layout()
plt.savefig('report/images/figure6_unserved_curtailment.png')
plt.close()
print('Saved figure6_unserved_curtailment.png')

# ========================================================================
# Figure 7: Spatial Generation and Unserved Load Maps
# ========================================================================
# Need to load per-bus results - let's compute from saved scenario data
# For simplicity, use bus-level data from the model run
import sys
sys.path.insert(0, 'code')
from dispatch_model import load_data, build_dispatch_model, solve_scenario

buses, links, generators, storage, demand, wind_cf = load_data()
model = build_dispatch_model(buses, links, generators, storage, demand, wind_cf)
base_res = solve_scenario(model, 'Base Case')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Average generation by bus
avg_wind = base_res['bus']['wind'].mean(axis=1)
avg_gas = base_res['bus']['gas'].mean(axis=1)
avg_nuc = base_res['bus']['nuclear'].mean(axis=1)

# Wind generation map
sc1 = axes[0].scatter(buses['x'], buses['y'], c=avg_wind, s=100, cmap='Blues', edgecolors='black', linewidths=0.5)
for _, b in buses.iterrows():
    axes[0].text(b['x'], b['y'] + 0.3, b['name'], fontsize=7, ha='center')
axes[0].set_title('Average Wind Dispatch (MW)', fontsize=10)
plt.colorbar(sc1, ax=axes[0])
axes[0].set_aspect('equal')

# Gas generation map
sc2 = axes[1].scatter(buses['x'], buses['y'], c=avg_gas, s=100, cmap='Oranges', edgecolors='black', linewidths=0.5)
for _, b in buses.iterrows():
    axes[1].text(b['x'], b['y'] + 0.3, b['name'], fontsize=7, ha='center')
axes[1].set_title('Average Gas Dispatch (MW)', fontsize=10)
plt.colorbar(sc2, ax=axes[1])
axes[1].set_aspect('equal')

# Unserved load map
avg_unserved = base_res['bus']['unserved'].mean(axis=1)
sc3 = axes[2].scatter(buses['x'], buses['y'], c=avg_unserved, s=100, cmap='Reds', edgecolors='black', linewidths=0.5)
for _, b in buses.iterrows():
    axes[2].text(b['x'], b['y'] + 0.3, b['name'], fontsize=7, ha='center')
axes[2].set_title('Average Unserved Load (MW)', fontsize=10)
plt.colorbar(sc3, ax=axes[2])
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('report/images/figure7_spatial_maps.png')
plt.close()
print('Saved figure7_spatial_maps.png')

# ========================================================================
# Figure 8: Marginal Price Heatmap
# ========================================================================
marginal_prices = base_res['marginal_prices']

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(marginal_prices, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
ax.set_yticks(range(len(buses)))
ax.set_yticklabels(buses['name'].values)
ax.set_xlabel('Hour')
ax.set_ylabel('Bus')
ax.set_title('Nodal Marginal Price (£/MWh) - Base Case', fontsize=11)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Price (£/MWh)')
plt.tight_layout()
plt.savefig('report/images/figure8_marginal_prices.png')
plt.close()
print('Saved figure8_marginal_prices.png')

# ========================================================================
# Figure 9: Transmission Utilization Heatmap
# ========================================================================
link_flow = base_res['link_flow']
link_names = [f"{l['bus0']}-{l['bus1']}" for l in model['params']['link_list']]
link_caps = np.array([l['p_nom'] for l in model['params']['link_list']])
utilization = np.abs(link_flow) / link_caps[:, None] * 100

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(utilization, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=100)
ax.set_yticks(range(len(link_names)))
ax.set_yticklabels(link_names, fontsize=7)
ax.set_xlabel('Hour')
ax.set_ylabel('Transmission Link')
ax.set_title('Transmission Link Utilization (%) - Base Case', fontsize=11)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Utilization (%)')
plt.tight_layout()
plt.savefig('report/images/figure9_transmission_utilization.png')
plt.close()
print('Saved figure9_transmission_utilization.png')

# ========================================================================
# Figure 10: Scenario Sensitivity - Key Metrics Table as Bar Chart
# ========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Unserved load
unserved_vals = [scenario_summary[s]['unserved_load_MWh'] / 1e3 for s in scenarios]
axes[0, 0].barh(scenarios, unserved_vals, color='#d62728', alpha=0.8)
axes[0, 0].set_xlabel('Unserved Load (GWh)')
axes[0, 0].set_title('Unserved Load by Scenario')

# Curtailment
curt_vals = [scenario_summary[s]['curtailment_MWh'] / 1e3 for s in scenarios]
axes[0, 1].barh(scenarios, curt_vals, color='#1f77b4', alpha=0.8)
axes[0, 1].set_xlabel('Curtailment (GWh)')
axes[0, 1].set_title('Wind Curtailment by Scenario')

# Wind generation
wind_vals = [scenario_summary[s]['wind_generation_MWh'] / 1e3 for s in scenarios]
axes[1, 0].barh(scenarios, wind_vals, color='#2ca02c', alpha=0.8)
axes[1, 0].set_xlabel('Wind Generation (GWh)')
axes[1, 0].set_title('Wind Generation by Scenario')

# System cost
cost_vals = [scenario_summary[s]['total_cost_GBP'] / 1e6 for s in scenarios]
axes[1, 1].barh(scenarios, cost_vals, color='#ff7f0e', alpha=0.8)
axes[1, 1].set_xlabel('Total Cost (M£)')
axes[1, 1].set_title('System Cost by Scenario')

plt.tight_layout()
plt.savefig('report/images/figure10_sensitivity_metrics.png')
plt.close()
print('Saved figure10_sensitivity_metrics.png')

print("\nAll figures generated successfully.")
