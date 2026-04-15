"""
Visualization and Scenario Analysis for GB Energy System
=========================================================
Generates publication-quality figures and runs sensitivity scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================
# LOAD DATA AND RESULTS
# ============================================================
buses = pd.read_csv(f'{DATA_DIR}/buses.csv')
links = pd.read_csv(f'{DATA_DIR}/links.csv')
generators = pd.read_csv(f'{DATA_DIR}/generators.csv')
storage_units = pd.read_csv(f'{DATA_DIR}/storage.csv')
demand = pd.read_csv(f'{DATA_DIR}/demand.csv')
wind_cf = pd.read_csv(f'{DATA_DIR}/wind_cf.csv')

with open(f'{OUTPUT_DIR}/results.json') as f:
    results = json.load(f)
with open(f'{OUTPUT_DIR}/meta.json') as f:
    meta = json.load(f)

T = meta['T']
N_buses = len(buses)
bus_names = meta['bus_names']
gen_keys = [(int(b), c) for b, c in meta['gen_keys']]
stor_bus_indices = meta['stor_bus_indices']
N_gen_groups = meta['N_gen_groups']
N_stor = meta['N_stor']

demand_matrix = np.load(f'{OUTPUT_DIR}/demand_matrix.npy')
dispatch = np.load(f'{OUTPUT_DIR}/dispatch.npy')
link_flows = np.load(f'{OUTPUT_DIR}/link_flows.npy')
stor_disch = np.load(f'{OUTPUT_DIR}/stor_disch.npy')
stor_ch = np.load(f'{OUTPUT_DIR}/stor_ch.npy')
stor_e = np.load(f'{OUTPUT_DIR}/stor_e.npy')
curtailment = np.load(f'{OUTPUT_DIR}/curtailment.npy')
wind_available = np.load(f'{OUTPUT_DIR}/wind_available.npy')
wind_dispatch = np.load(f'{OUTPUT_DIR}/wind_dispatch.npy')
load_shedding = np.load(f'{OUTPUT_DIR}/load_shedding.npy')

hours = np.arange(T)
colors = {'gas': '#E69F00', 'nuclear': '#56B4E9', 'onshore wind': '#009E73', 
          'curtailment': '#D55E00', 'shedding': '#CC79A7'}

print("Data loaded successfully")

# ============================================================
# FIGURE 1: System-wide dispatch stack plot over time
# ============================================================
print("Creating Figure 1: System dispatch stack...")

gas_disp = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'gas')
nuc_disp = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'nuclear')
wind_disp_total = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'onshore wind')
total_demand = demand_matrix.sum(axis=1)
total_served = total_demand - load_shedding.sum(axis=0)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(hours, 0, nuc_disp, color=colors['nuclear'], alpha=0.85, label='Nuclear')
ax.fill_between(hours, nuc_disp, nuc_disp + gas_disp, color=colors['gas'], alpha=0.85, label='Gas')
ax.fill_between(hours, nuc_disp + gas_disp, nuc_disp + gas_disp + wind_disp_total,
                color=colors['onshore wind'], alpha=0.85, label='Onshore Wind')
ax.plot(hours, total_demand, 'k-', linewidth=1.5, label='Total Demand')
ax.plot(hours, total_served, 'r--', linewidth=1.5, label='Served Demand')
ax.set_xlabel('Hour')
ax.set_ylabel('Power (MW)')
ax.set_title('System-Wide Generation Dispatch and Demand Over 168 Hours')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, T-1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig01_dispatch_stack.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig01_dispatch_stack.png")

# ============================================================
# FIGURE 2: Per-carrier energy contribution pie/bar chart
# ============================================================
print("Creating Figure 2: Energy contribution by carrier...")

carrier_energy = {}
for g, (b, c) in enumerate(gen_keys):
    if c not in carrier_energy:
        carrier_energy[c] = 0
    carrier_energy[c] += dispatch[g, :].sum()
carrier_energy['Load Shedding'] = load_shedding.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
carriers_list = ['nuclear', 'gas', 'onshore wind', 'Load Shedding']
energy_vals = [carrier_energy[c] / 1000 for c in carriers_list]  # GWh
bar_colors = [colors.get(c, '#999999') for c in carriers_list]
bars = axes[0].bar(carriers_list, energy_vals, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)
for bar, val in zip(bars, energy_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0].set_ylabel('Energy (GWh)')
axes[0].set_title('Total Energy by Source (168h)')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart (generation only, no shedding)
gen_carriers = ['nuclear', 'gas', 'onshore wind']
gen_vals = [carrier_energy[c] for c in gen_carriers]
gen_labels = [f'{c}\n{v/1000:.0f} GWh' for c, v in zip(gen_carriers, gen_vals)]
explode = [0.05] * len(gen_carriers)
axes[1].pie(gen_vals, labels=gen_labels, colors=[colors[c] for c in gen_carriers],
            autopct='%1.1f%%', startangle=90, explode=explode, textprops={'fontsize': 9})
axes[1].set_title('Generation Mix Share')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig02_energy_contribution.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig02_energy_contribution.png")

# ============================================================
# FIGURE 3: Spatial demand and generation map
# ============================================================
print("Creating Figure 3: Spatial distribution...")

bus_x = buses['x'].values.astype(float)
bus_y = buses['y'].values.astype(float)
bus_demand = demand_matrix.sum(axis=0)  # Total demand per bus over 168h

# Generation per bus
bus_generation = np.zeros(N_buses)
for g, (b, c) in enumerate(gen_keys):
    bus_generation[b] += dispatch[g, :].sum()

bus_curtailment_val = curtailment.sum(axis=1).astype(float)  # sum over time -> per bus
bus_shedding = load_shedding.sum(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Demand
sc1 = axes[0, 0].scatter(bus_x, bus_y, s=bus_demand/50, c=bus_demand, cmap='Reds',
                          edgecolors='black', linewidth=0.5, alpha=0.8)
axes[0, 0].set_title('Total Demand per Bus (MWh)')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')
plt.colorbar(sc1, ax=axes[0, 0], shrink=0.8, label='MWh')

# Generation
sc2 = axes[0, 1].scatter(bus_x, bus_y, s=bus_generation/50, c=bus_generation, cmap='Greens',
                          edgecolors='black', linewidth=0.5, alpha=0.8)
axes[0, 1].set_title('Total Generation per Bus (MWh)')
axes[0, 1].set_xlabel('Longitude')
axes[0, 1].set_ylabel('Latitude')
plt.colorbar(sc2, ax=axes[0, 1], shrink=0.8, label='MWh')

# Curtailment
sc3 = axes[1, 0].scatter(bus_x, bus_y, s=np.maximum(bus_curtailment_val, 100).astype(float), c=bus_curtailment_val,
                          cmap='Oranges', edgecolors='black', linewidth=0.5, alpha=0.8)
axes[1, 0].set_title('Wind Curtailment per Bus (MWh)')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Latitude')
plt.colorbar(sc3, ax=axes[1, 0], shrink=0.8, label='MWh')

# Load shedding
bus_shedding_per_bus = load_shedding.sum(axis=1).astype(float)  # per bus
sc4 = axes[1, 1].scatter(bus_x, bus_y, s=np.maximum(bus_shedding_per_bus/50, 30), c=bus_shedding_per_bus, cmap='Purples',
                          edgecolors='black', linewidth=0.5, alpha=0.8)
axes[1, 1].set_title('Load Shedding per Bus (MWh)')
axes[1, 1].set_xlabel('Longitude')
axes[1, 1].set_ylabel('Latitude')
plt.colorbar(sc4, ax=axes[1, 1], shrink=0.8, label='MWh')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig03_spatial_distribution.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig03_spatial_distribution.png")

# ============================================================
# FIGURE 4: Storage behavior over time
# ============================================================
print("Creating Figure 4: Storage dynamics...")

stor_labels = [f"Bus{b+1} PHS" for b in stor_bus_indices]
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for s_idx in range(N_stor):
    # Power (discharge positive, charge negative)
    net_power = stor_disch[s_idx, :] - stor_ch[s_idx, :]
    axes[0].plot(hours, net_power, linewidth=1.2, label=stor_labels[s_idx])
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_ylabel('Net Power (MW)')
axes[0].set_title('Storage Net Power (Discharge > 0)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for s_idx in range(N_stor):
    axes[1].plot(hours, stor_disch[s_idx, :], '--', linewidth=1, label=f'{stor_labels[s_idx]} Discharge')
    axes[1].plot(hours, stor_ch[s_idx, :], ':', linewidth=1, label=f'{stor_labels[s_idx]} Charge')
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Power (MW)')
axes[1].set_title('Storage Charge/Discharge')
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.3)

for s_idx in range(N_stor):
    axes[2].plot(hours, stor_e[s_idx, :], linewidth=1.2, label=stor_labels[s_idx])
axes[2].set_ylabel('Energy (MWh)')
axes[2].set_title('Storage Energy Level')
axes[2].set_xlabel('Hour')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig04_storage_dynamics.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig04_storage_dynamics.png")

# ============================================================
# FIGURE 5: Wind availability vs dispatch and curtailment
# ============================================================
print("Creating Figure 5: Wind analysis...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Time series
axes[0].fill_between(hours, 0, wind_available.sum(axis=0), color=colors['onshore wind'],
                      alpha=0.4, label='Available Wind')
axes[0].fill_between(hours, 0, wind_dispatch.sum(axis=0), color=colors['onshore wind'],
                      alpha=0.8, label='Dispatched Wind')
axes[0].fill_between(hours, wind_dispatch.sum(axis=0), wind_available.sum(axis=0),
                      color=colors['curtailment'], alpha=0.6, label='Curtailment')
axes[0].set_ylabel('Wind Power (MW)')
axes[0].set_title('System-Wind: Available vs Dispatched vs Curtailed')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Histogram of capacity factors
cf_flat = wind_cf.values.flatten()
axes[1].hist(cf_flat, bins=30, color=colors['onshore wind'], edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Capacity Factor')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Wind Capacity Factor Distribution (All Buses, All Hours)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig05_wind_analysis.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig05_wind_analysis.png")

# ============================================================
# FIGURE 6: Load shedding analysis
# ============================================================
print("Creating Figure 6: Load shedding analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hourly shedding
shedding_hourly = load_shedding.sum(axis=0)
axes[0, 0].plot(hours, shedding_hourly, color=colors['shedding'], linewidth=1.5)
axes[0, 0].fill_between(hours, 0, shedding_hourly, color=colors['shedding'], alpha=0.3)
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Shedding (MW)')
axes[0, 0].set_title('Hourly Load Shedding')
axes[0, 0].grid(True, alpha=0.3)

# Shedding vs demand
axes[0, 1].scatter(total_demand, shedding_hourly, c=hours, cmap='viridis', s=20, alpha=0.7)
axes[0, 1].set_xlabel('Total Demand (MW)')
axes[0, 1].set_ylabel('Shedding (MW)')
axes[0, 1].set_title('Shedding vs Demand Level')
axes[0, 1].grid(True, alpha=0.3)

# Per-bus shedding
shedding_pct_per_bus = []
for b in range(N_buses):
    pct = 100 * load_shedding[b, :].sum() / max(demand_matrix[:, b].sum(), 1)
    shedding_pct_per_bus.append(pct)

bars = axes[1, 0].bar(range(N_buses), shedding_pct_per_bus, color=colors['shedding'], edgecolor='black', linewidth=0.5)
axes[1, 0].set_xlabel('Bus Index')
axes[1, 0].set_ylabel('Shedding (%)')
axes[1, 0].set_title('Load Shedding Percentage by Bus')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Supply adequacy ratio
adequacy = total_served / total_demand * 100
axes[1, 1].plot(hours, adequacy, 'b-', linewidth=1.5)
axes[1, 1].axhline(y=100, color='g', linestyle='--', linewidth=1, label='Full Supply')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Supply Adequacy (%)')
axes[1, 1].set_title('Demand Served Ratio Over Time')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig06_load_shedding.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig06_load_shedding.png")

# ============================================================
# FIGURE 7: Network flow patterns
# ============================================================
print("Creating Figure 7: Network flows...")
N_links = len(links)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Show top 5 most used links
link_usage = np.abs(link_flows).sum(axis=1)
top_links = np.argsort(link_usage)[-5:][::-1]

for l_idx in top_links:
    link_name = f"{links.iloc[l_idx]['bus0']}-{links.iloc[l_idx]['bus1']}"
    axes[0].plot(hours, link_flows[l_idx, :], linewidth=1, label=link_name)
axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_ylabel('Flow (MW)')
axes[0].set_title('Top 5 Most Utilized Transmission Links')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Link utilization histogram
link_util = np.abs(link_flows).max(axis=1) / links['p_nom'].values * 100
axes[1].bar(range(N_links), link_util, color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.7)
axes[1].axhline(y=100, color='r', linestyle='--', linewidth=1, label='Full Capacity')
axes[1].set_xlabel('Link Index')
axes[1].set_ylabel('Peak Utilization (%)')
axes[1].set_title('Transmission Link Peak Utilization')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig07_network_flows.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig07_network_flows.png")

# ============================================================
# FIGURE 8: Cost breakdown
# ============================================================
print("Creating Figure 8: Cost breakdown...")

fig, ax = plt.subplots(figsize=(10, 6))

cost_data = {
    'Gas Fuel': results['carrier_summary']['gas']['cost'],
    'Nuclear Fuel': results['carrier_summary']['nuclear']['cost'],
    'Wind (zero marginal)': 0,
    'Value of Lost Load': results['total_shedding_cost'],
}

categories = list(cost_data.keys())
values = list(cost_data.values())
bar_colors_plot = [colors['gas'], colors['nuclear'], colors['onshore wind'], colors['shedding']]

bars = ax.barh(categories, values, color=bar_colors_plot, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, values):
    if val > 0:
        ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f'${val/1e6:.1f}M', va='center', fontweight='bold', fontsize=9)

ax.set_xlabel('Cost ($)')
ax.set_title('System Cost Breakdown (168-Hour Period)')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig08_cost_breakdown.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig08_cost_breakdown.png")

# ============================================================
# FIGURE 9: Hourly supply-demand balance
# ============================================================
print("Creating Figure 9: Supply-demand balance...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Supply vs demand
axes[0].plot(hours, total_demand, 'k-', linewidth=1.5, label='Demand')
axes[0].plot(hours, total_served, 'b-', linewidth=1.5, label='Served')
axes[0].fill_between(hours, total_served, total_demand, color='red', alpha=0.3, label='Unmet')
axes[0].set_ylabel('Power (MW)')
axes[0].set_title('Supply vs Demand Balance')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Marginal price proxy (shadow prices from demand constraint would be ideal, 
# but we show residual as proxy)
residual = total_demand - total_served
axes[1].plot(hours, residual, 'r-', linewidth=1.5)
axes[1].fill_between(hours, 0, residual, color='red', alpha=0.3)
axes[1].set_ylabel('Unmet Demand (MW)')
axes[1].set_title('Unmet Demand Over Time')
axes[1].grid(True, alpha=0.3)

# Cumulative energy
cum_demand = np.cumsum(total_demand)
cum_served = np.cumsum(total_served)
cum_shed = np.cumsum(shedding_hourly)
axes[2].plot(hours, cum_demand/1000, 'k-', linewidth=1.5, label='Cumulative Demand')
axes[2].plot(hours, cum_served/1000, 'b-', linewidth=1.5, label='Cumulative Served')
axes[2].plot(hours, cum_shed/1000, 'r--', linewidth=1.5, label='Cumulative Shedding')
axes[2].set_ylabel('Cumulative Energy (GWh)')
axes[2].set_xlabel('Hour')
axes[2].set_title('Cumulative Energy Balance')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig09_supply_demand_balance.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig09_supply_demand_balance.png")

# ============================================================
# FIGURE 10: Per-bus detailed generation mix
# ============================================================
print("Creating Figure 10: Per-bus generation mix...")

fig, ax = plt.subplots(figsize=(14, 6))

bus_gen_by_carrier = {c: np.zeros(N_buses) for c in ['gas', 'nuclear', 'onshore wind']}
for g, (b, c) in enumerate(gen_keys):
    bus_gen_by_carrier[c][b] = dispatch[g, :].sum()

x = np.arange(N_buses)
width = 0.25
ax.bar(x - width, bus_gen_by_carrier['nuclear']/1000, width, label='Nuclear', color=colors['nuclear'])
ax.bar(x, bus_gen_by_carrier['gas']/1000, width, label='Gas', color=colors['gas'])
ax.bar(x + width, bus_gen_by_carrier['onshore wind']/1000, width, label='Wind', color=colors['onshore wind'])

ax.set_xlabel('Bus')
ax.set_ylabel('Energy (GWh)')
ax.set_title('Per-Bus Generation Mix by Carrier')
ax.set_xticks(x)
ax.set_xticklabels([f'B{i+1}' for i in range(N_buses)], fontsize=7)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig10_per_bus_generation.png', dpi=150)
plt.close()
print(f"  Saved: {IMG_DIR}/fig10_per_bus_generation.png")

print("\n=== All figures generated ===")
print(f"Figures saved to: {IMG_DIR}/")
