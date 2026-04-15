"""
GB Power System Visualization and Analysis
==========================================
Generates figures for the research report from saved outputs.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = "outputs"
FIG_DIR = "report/images"
os.makedirs(FIG_DIR, exist_ok=True)

# Load results
with open(f"{OUT_DIR}/results_summary.json") as f:
    summary = json.load(f)

hourly = pd.read_csv(f"{OUT_DIR}/hourly_dispatch.csv")
bus_results = pd.read_csv(f"{OUT_DIR}/bus_results.csv")
link_results = pd.read_csv(f"{OUT_DIR}/link_results.csv")
gen_results = pd.read_csv(f"{OUT_DIR}/generator_results.csv")
buses = pd.read_csv("data/buses.csv")
gens = pd.read_csv("data/generators.csv")
demand = pd.read_csv("data/demand.csv")
wind_cf = pd.read_csv("data/wind_cf.csv")
links = pd.read_csv("data/links.csv")
storage = pd.read_csv("data/storage.csv")

bus_list = buses["name"].tolist()

COLORS = {
    'onshore wind': '#2ecc71',
    'gas': '#e74c3c',
    'nuclear': '#9b59b6',
    'demand': '#2c3e50',
    'curtailment': '#f39c12',
    'shedding': '#e74c3c',
    'storage_charge': '#3498db',
    'storage_discharge': '#1abc9c',
}

# ── Figure 1: System Overview - Stacked Generation ───────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

hours = hourly['hour'].values
demand_vals = hourly['total_demand_mw'].values

ax = axes[0]
carriers = ['onshore wind', 'gas', 'nuclear']
bottom = np.zeros(len(hours))
for carrier in carriers:
    col = f'gen_{carrier}_mw'
    if col in hourly.columns:
        vals = hourly[col].values
        ax.fill_between(hours, bottom, bottom + vals, alpha=0.8, label=carrier, color=COLORS.get(carrier, '#95a5a6'))
        bottom += vals
shed_vals = hourly['total_load_shedding_mw'].values
ax.fill_between(hours, bottom, bottom + shed_vals, alpha=0.6, label='Load shedding', color='#e74c3c', hatch='//')
bottom += shed_vals
ax.plot(hours, demand_vals, 'k-', linewidth=1.5, label='Demand', alpha=0.7)
ax.set_ylabel('Power (MW)')
ax.set_title('Hourly Generation Dispatch and Demand')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(hours, hourly['total_storage_charge_mw'].values, color=COLORS['storage_charge'], label='Charging', linewidth=1.2)
ax.plot(hours, hourly['total_storage_discharge_mw'].values, color=COLORS['storage_discharge'], label='Discharging', linewidth=1.2)
ax.fill_between(hours, 0, hourly['total_storage_charge_mw'].values, alpha=0.3, color=COLORS['storage_charge'])
ax.fill_between(hours, 0, hourly['total_storage_discharge_mw'].values, alpha=0.3, color=COLORS['storage_discharge'])
ax.set_ylabel('Power (MW)')
ax.set_title('Storage Operation')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.fill_between(hours, 0, hourly['total_curtailment_mw'].values, alpha=0.6, color=COLORS['curtailment'], label='Curtailment')
ax.fill_between(hours, 0, hourly['total_load_shedding_mw'].values, alpha=0.6, color=COLORS['shedding'], label='Load shedding')
ax.set_ylabel('Power (MW)')
ax.set_xlabel('Hour')
ax.set_title('Curtailment and Load Shedding')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/hourly_dispatch.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: hourly_dispatch.png")

# ── Figure 2: Generation Mix Pie Chart ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

gen_by_carrier = summary['total_generation_by_carrier_gwh']
labels = list(gen_by_carrier.keys())
sizes = list(gen_by_carrier.values())
colors = [COLORS.get(l, '#95a5a6') for l in labels]
ax = axes[0]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Generation Mix by Carrier')

total_gen = sum(sizes)
total_demand = summary['total_demand_gwh']
shedding = summary['total_load_shedding_gwh']
met = total_demand - shedding
ax = axes[1]
ax.pie([met, shedding], labels=['Met demand', 'Load shedding'], autopct='%1.1f%%',
       colors=['#2ecc71', '#e74c3c'], startangle=90)
ax.set_title('Demand Satisfaction')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/generation_mix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: generation_mix.png")

# ── Figure 3: Bus-level Results ──────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

bus_names = bus_results['bus'].values
x_pos = np.arange(len(bus_names))
width = 0.35

ax = axes[0]
ax.bar(x_pos - width/2, bus_results['demand_gwh'], width, label='Demand', color='#2c3e50', alpha=0.8)
ax.bar(x_pos + width/2, bus_results['generation_gwh'], width, label='Local generation', color='#2ecc71', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Energy (GWh)')
ax.set_title('Bus-level Demand vs Local Generation')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x_pos - width/2, bus_results['curtailment_gwh'], width, label='Curtailment', color=COLORS['curtailment'], alpha=0.8)
ax.bar(x_pos + width/2, bus_results['load_shedding_gwh'], width, label='Load shedding', color=COLORS['shedding'], alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Energy (GWh)')
ax.set_title('Bus-level Curtailment and Load Shedding')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/bus_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: bus_results.png")

# ── Figure 4: Network Topology and Link Utilization ──────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

bus_coords = {row['name']: (row['x'], row['y']) for _, row in buses.iterrows()}
for _, bus in buses.iterrows():
    ax.plot(bus['x'], bus['y'], 'o', markersize=15, color='#3498db', zorder=5)
    ax.annotate(bus['name'], (bus['x'], bus['y']), fontsize=7, ha='center', va='bottom',
                xytext=(0, 10), textcoords='offset points')

for _, link in links.iterrows():
    x0, y0 = bus_coords[link['bus0']]
    x1, y1 = bus_coords[link['bus1']]
    link_name = f"{link['bus0']}-{link['bus1']}"
    util_row = link_results[link_results['link'] == link_name]
    if len(util_row) > 0:
        util = util_row['utilization_pct'].values[0] / 100
    else:
        util = 0
    color = plt.cm.RdYlGn_r(util)
    lw = 1 + util * 4
    ax.plot([x0, x1], [y0, y1], '-', color=color, linewidth=lw, alpha=0.7, zorder=1)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GB Power System Network Topology\n(Line color/thickness indicates utilization)')
ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label('Link Utilization (%)')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/network_topology.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: network_topology.png")

# ── Figure 5: Load Duration Curve ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

demand_sorted = np.sort(demand_vals)[::-1]
wind_total = hourly['gen_onshore wind_mw'].values
wind_sorted = np.sort(wind_total)[::-1]

ax.plot(range(1, len(demand_sorted)+1), demand_sorted, 'k-', linewidth=2, label='Demand')
ax.plot(range(1, len(wind_sorted)+1), wind_sorted, '-', color=COLORS['onshore wind'], linewidth=2, label='Wind generation')
ax.fill_between(range(1, len(demand_sorted)+1), 0, demand_sorted, alpha=0.1, color='black')
ax.fill_between(range(1, len(wind_sorted)+1), 0, wind_sorted, alpha=0.2, color=COLORS['onshore wind'])

gas_cap = gens[gens.carrier=='gas']['p_nom'].sum()
nuc_cap = gens[gens.carrier=='nuclear']['p_nom'].sum()
ax.axhline(y=gas_cap + nuc_cap, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Gas+Nuclear capacity ({gas_cap+nuc_cap:.0f} MW)')
ax.axhline(y=gas_cap, color='#e74c3c', linestyle=':', alpha=0.5, label=f'Gas capacity ({gas_cap:.0f} MW)')

ax.set_xlabel('Hours (sorted)')
ax.set_ylabel('Power (MW)')
ax.set_title('Load Duration Curve and Wind Availability')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/load_duration_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: load_duration_curve.png")

# ── Figure 6: Wind Capacity Factors Heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

cf_matrix = wind_cf.values.T
im = ax.imshow(cf_matrix, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
ax.set_xlabel('Hour')
ax.set_ylabel('Bus')
ax.set_yticks(range(len(bus_list)))
ax.set_yticklabels(buses['name'].values, fontsize=7)
ax.set_title('Wind Capacity Factors by Bus and Hour')
plt.colorbar(im, ax=ax, label='Capacity Factor', shrink=0.8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/wind_capacity_factors.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: wind_capacity_factors.png")

# ── Figure 7: Demand Profile by Bus ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

for bus_name in bus_list:
    ax.plot(hours, demand[bus_name].values, alpha=0.5, linewidth=0.8)

ax.set_xlabel('Hour')
ax.set_ylabel('Demand (MW)')
ax.set_title('Hourly Demand Profiles by Bus')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/demand_profiles.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: demand_profiles.png")

# ── Figure 8: Scenario Comparison ────────────────────────────────────────
# Create a comparison showing the capacity adequacy gap
fig, ax = plt.subplots(figsize=(12, 6))

# Total available generation per hour
wind_avail = np.zeros(168)
for _, g in gens[gens.carrier=="onshore wind"].iterrows():
    wind_avail += wind_cf[g["bus"]].values * g["p_nom"]

gas_cap = gens[gens.carrier=="gas"]["p_nom"].sum()
nuc_cap = gens[gens.carrier=="nuclear"]["p_nom"].sum()
total_avail = wind_avail + gas_cap + nuc_cap

ax.fill_between(hours, 0, demand_vals, alpha=0.3, color='#2c3e50', label='Demand')
ax.fill_between(hours, 0, wind_avail, alpha=0.5, color=COLORS['onshore wind'], label='Wind available')
ax.axhline(y=gas_cap, color=COLORS['gas'], linestyle='--', alpha=0.7, label=f'Gas capacity ({gas_cap:.0f} MW)')
ax.axhline(y=gas_cap+nuc_cap, color=COLORS['nuclear'], linestyle='--', alpha=0.7, label=f'Gas+Nuclear ({gas_cap+nuc_cap:.0f} MW)')

# Show gap
gap = demand_vals - total_avail
gap_positive = np.maximum(gap, 0)
ax.fill_between(hours, total_avail, total_avail + gap_positive, alpha=0.3, color='#e74c3c', label='Capacity gap')

ax.set_xlabel('Hour')
ax.set_ylabel('Power (MW)')
ax.set_title('Capacity Adequacy Analysis')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/capacity_adequacy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved: capacity_adequacy.png")

print("\nAll figures generated successfully!")
