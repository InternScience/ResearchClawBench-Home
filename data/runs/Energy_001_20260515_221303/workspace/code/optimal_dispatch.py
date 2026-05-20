#!/usr/bin/env python3
"""
GB Power System Optimal Dispatch Model
=======================================
A high-resolution (20-node, hourly) optimal power dispatch model for Great Britain
using linear programming.

Includes load shedding with Value of Lost Load (VoLL) penalty to handle
generation capacity shortfall in heavily electrified future scenarios.

Author: Autonomous Research Agent
Date: 2026-05-15
"""

import numpy as np
import pandas as pd
from pulp import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os, sys, json, warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
REPORT_IMG_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# Value of Lost Load (VoLL) - penalty for unserved energy
VOLL = 10000.0  # €/MWh

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

buses = pd.read_csv(f'{DATA_DIR}/buses.csv')
links = pd.read_csv(f'{DATA_DIR}/links.csv')
demand = pd.read_csv(f'{DATA_DIR}/demand.csv')
generators = pd.read_csv(f'{DATA_DIR}/generators.csv')
wind_cf = pd.read_csv(f'{DATA_DIR}/wind_cf.csv')
storage = pd.read_csv(f'{DATA_DIR}/storage.csv')

bus_names = buses['name'].tolist()
n_buses = len(bus_names)
n_hours = len(demand)
n_links = len(links)

print(f"  Buses: {n_buses}")
print(f"  Links: {n_links}")
print(f"  Generators: {len(generators)}")
print(f"  Storage units: {len(storage)}")
print(f"  Time periods: {n_hours} hours")

bus_to_idx = {name: i for i, name in enumerate(bus_names)}

# ============================================================
# 2. DATA SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)

gen_summary = generators.groupby('carrier').agg(
    total_capacity=('p_nom', 'sum'),
    count=('p_nom', 'count'),
    avg_marginal_cost=('marginal_cost', 'mean')
)
print("\nGenerator Summary:")
print(gen_summary)

total_demand_per_hour = demand.sum(axis=1)
total_demand_mwh = total_demand_per_hour.sum()
print(f"\nTotal Demand: min={total_demand_per_hour.min():.0f} MW, "
      f"max={total_demand_per_hour.max():.0f} MW, "
      f"mean={total_demand_per_hour.mean():.0f} MW")
print(f"Total Energy Demand: {total_demand_mwh:,.0f} MWh")

print(f"\nWind CF: min={wind_cf.values.min():.3f}, max={wind_cf.values.max():.3f}, "
      f"mean={wind_cf.values.mean():.3f}")

# ============================================================
# 3. BUILD OPTIMIZATION MODEL
# ============================================================
print("\n" + "=" * 60)
print("BUILDING OPTIMIZATION MODEL")
print("=" * 60)

prob = LpProblem("GB_Power_Dispatch", LpMinimize)

T = range(n_hours)
B = range(n_buses)
G = range(len(generators))
S = range(len(storage))
L = range(n_links)

gen_bus_idx = [bus_to_idx[b] for b in generators['bus']]
gen_carrier = generators['carrier'].tolist()
gen_p_nom = generators['p_nom'].values
gen_mc = generators['marginal_cost'].values
is_wind = np.array([c == 'onshore wind' for c in gen_carrier])

sto_bus_idx = [bus_to_idx[b] for b in storage['bus']]
sto_p_nom = storage['p_nom'].values
sto_e_nom = storage['e_nom'].values
sto_eff = storage['efficiency'].values

link_bus0_idx = [bus_to_idx[b] for b in links['bus0']]
link_bus1_idx = [bus_to_idx[b] for b in links['bus1']]
link_p_nom = links['p_nom'].values

demand_array = demand.values
wind_cf_array = wind_cf.values

# ---- Decision Variables ----
print("  Creating decision variables...")

p_gen = {}
for g in G:
    for t in T:
        p_gen[g, t] = LpVariable(f"p_gen_{g}_{t}", lowBound=0)

c_sto = {}; d_sto = {}; soc = {}
for s in S:
    for t in T:
        c_sto[s, t] = LpVariable(f"c_sto_{s}_{t}", lowBound=0)
        d_sto[s, t] = LpVariable(f"d_sto_{s}_{t}", lowBound=0)
        soc[s, t] = LpVariable(f"soc_{s}_{t}", lowBound=0, upBound=sto_e_nom[s])

f_link = {}
for l in L:
    for t in T:
        f_link[l, t] = LpVariable(f"f_link_{l}_{t}",
                                  lowBound=-link_p_nom[l],
                                  upBound=link_p_nom[l])

shed = {}
for b in B:
    for t in T:
        shed[b, t] = LpVariable(f"shed_{b}_{t}", lowBound=0,
                                upBound=demand_array[t, b])

# ---- Constraints ----
print("  Adding constraints...")

for g in G:
    for t in T:
        if is_wind[g]:
            bus = gen_bus_idx[g]
            avail = gen_p_nom[g] * wind_cf_array[t, bus]
            p_gen[g, t].upBound = avail
        else:
            p_gen[g, t].upBound = gen_p_nom[g]

for s in S:
    for t in T:
        c_sto[s, t].upBound = sto_p_nom[s]
        d_sto[s, t].upBound = sto_p_nom[s]

for b in B:
    for t in T:
        gen_at_bus = [p_gen[g, t] for g in G if gen_bus_idx[g] == b]
        charge_at_bus = [c_sto[s, t] for s in S if sto_bus_idx[s] == b]
        discharge_at_bus = [d_sto[s, t] for s in S if sto_bus_idx[s] == b]
        link_inflow = []
        for l in L:
            if link_bus0_idx[l] == b:
                link_inflow.append(-f_link[l, t])
            elif link_bus1_idx[l] == b:
                link_inflow.append(f_link[l, t])
        lhs = lpSum(gen_at_bus) + lpSum(discharge_at_bus) + lpSum(link_inflow) + shed[b, t]
        rhs = demand_array[t, b] + lpSum(charge_at_bus)
        prob += (lhs == rhs), f"bal_b{b}_t{t}"

for s in S:
    prob += (soc[s, 0] == 0.5 * sto_e_nom[s]), f"soc_init_{s}"
    for t in range(1, n_hours):
        prob += (soc[s, t] == soc[s, t-1] +
                 sto_eff[s] * c_sto[s, t] - d_sto[s, t]), f"soc_{s}_{t}"
    prob += (soc[s, n_hours-1] == 0.5 * sto_e_nom[s]), f"soc_final_{s}"

# ---- Objective ----
print("  Building objective...")
obj = lpSum([gen_mc[g] * p_gen[g, t] for g in G for t in T])
obj += lpSum([VOLL * shed[b, t] for b in B for t in T])
prob += obj, "total_cost"

# ============================================================
# 4. SOLVE
# ============================================================
print("\n" + "=" * 60)
print("SOLVING OPTIMIZATION MODEL")
print("=" * 60)
print(f"  Variables: {len(prob.variables())}")
print(f"  Constraints: {len(prob.constraints)}")

prob.solve(PULP_CBC_CMD(msg=True, timeLimit=300, gapRel=0.001))

print(f"\n  Status: {LpStatus[prob.status]}")
print(f"  Objective value: {value(prob.objective):.2f}")

# ============================================================
# 5. EXTRACT RESULTS
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING RESULTS")
print("=" * 60)

gen_dispatch = np.zeros((len(G), n_hours))
for g in G:
    for t in T:
        gen_dispatch[g, t] = value(p_gen[g, t])

sto_charge = np.zeros((len(S), n_hours))
sto_discharge = np.zeros((len(S), n_hours))
sto_soc_vals = np.zeros((len(S), n_hours))
for s in S:
    for t in T:
        sto_charge[s, t] = value(c_sto[s, t])
        sto_discharge[s, t] = value(d_sto[s, t])
        sto_soc_vals[s, t] = value(soc[s, t])

link_flows = np.zeros((len(L), n_hours))
for l in L:
    for t in T:
        link_flows[l, t] = value(f_link[l, t])

shedding = np.zeros((n_buses, n_hours))
for b in B:
    for t in T:
        shedding[b, t] = value(shed[b, t])

# ============================================================
# 6. COMPUTE DERIVED METRICS
# ============================================================
print("Computing derived metrics...")

gen_by_carrier = {}
for carrier in ['onshore wind', 'gas', 'nuclear']:
    mask = np.array([c == carrier for c in gen_carrier])
    gen_by_carrier[carrier] = gen_dispatch[mask].sum(axis=0)

total_gen_per_hour = gen_dispatch.sum(axis=0)
total_shed_per_hour = shedding.sum(axis=0)

wind_available = np.zeros((sum(is_wind), n_hours))
wind_idx = 0
for g in G:
    if is_wind[g]:
        bus = gen_bus_idx[g]
        wind_available[wind_idx] = gen_p_nom[g] * wind_cf_array[:, bus]
        wind_idx += 1
total_wind_available = wind_available.sum(axis=0)
total_wind_dispatched = gen_dispatch[is_wind].sum(axis=0)
wind_curtailment = total_wind_available - total_wind_dispatched

total_charge = sto_charge.sum(axis=0)
total_discharge = sto_discharge.sum(axis=0)

gen_per_bus = np.zeros((n_buses, n_hours))
for g in G:
    gen_per_bus[gen_bus_idx[g]] += gen_dispatch[g]

gen_cost = sum(gen_dispatch[g].sum() * gen_mc[g] for g in G)
shed_cost = shedding.sum() * VOLL
total_cost = gen_cost + shed_cost

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("Saving results...")

results_ts = pd.DataFrame({
    'hour': range(n_hours),
    'total_demand': total_demand_per_hour.values,
    'total_generation': total_gen_per_hour,
    'wind_generation': gen_by_carrier['onshore wind'],
    'gas_generation': gen_by_carrier['gas'],
    'nuclear_generation': gen_by_carrier['nuclear'],
    'wind_available': total_wind_available,
    'wind_curtailment': wind_curtailment,
    'storage_charge': total_charge,
    'storage_discharge': total_discharge,
    'load_shedding': total_shed_per_hour,
})
results_ts.to_csv(f'{OUTPUT_DIR}/dispatch_timeseries.csv', index=False)

bus_results = pd.DataFrame(gen_per_bus.T, columns=bus_names)
bus_results.index.name = 'hour'
bus_results.to_csv(f'{OUTPUT_DIR}/generation_per_bus.csv')

link_flows_df = pd.DataFrame(link_flows.T,
    columns=[f"{links.iloc[l]['bus0']}-{links.iloc[l]['bus1']}" for l in range(n_links)])
link_flows_df.index.name = 'hour'
link_flows_df.to_csv(f'{OUTPUT_DIR}/link_flows.csv')

shedding_df = pd.DataFrame(shedding.T, columns=bus_names)
shedding_df.index.name = 'hour'
shedding_df.to_csv(f'{OUTPUT_DIR}/load_shedding.csv')

summary = {
    'total_cost_eur': float(total_cost),
    'generation_cost_eur': float(gen_cost),
    'shedding_cost_eur': float(shed_cost),
    'total_demand_mwh': float(total_demand_per_hour.sum()),
    'total_generation_mwh': float(total_gen_per_hour.sum()),
    'total_shedding_mwh': float(total_shed_per_hour.sum()),
    'total_wind_gen_mwh': float(gen_by_carrier['onshore wind'].sum()),
    'total_gas_gen_mwh': float(gen_by_carrier['gas'].sum()),
    'total_nuclear_gen_mwh': float(gen_by_carrier['nuclear'].sum()),
    'total_wind_available_mwh': float(total_wind_available.sum()),
    'total_wind_curtailed_mwh': float(wind_curtailment.sum()),
    'wind_curtailment_pct': float(wind_curtailment.sum() / max(1, total_wind_available.sum()) * 100),
    'wind_share_of_demand_pct': float(gen_by_carrier['onshore wind'].sum() / total_demand_per_hour.sum() * 100),
    'gas_share_of_demand_pct': float(gen_by_carrier['gas'].sum() / total_demand_per_hour.sum() * 100),
    'nuclear_share_of_demand_pct': float(gen_by_carrier['nuclear'].sum() / total_demand_per_hour.sum() * 100),
    'load_shedding_pct': float(total_shed_per_hour.sum() / total_demand_per_hour.sum() * 100),
    'shedding_hours': int(np.sum(total_shed_per_hour > 1)),
    'max_shedding_mw': float(total_shed_per_hour.max()),
    'min_demand_mw': float(total_demand_per_hour.min()),
    'max_demand_mw': float(total_demand_per_hour.max()),
    'avg_demand_mw': float(total_demand_per_hour.mean()),
    'total_gen_capacity_mw': float(gen_p_nom.sum()),
    'total_wind_capacity_mw': float(gen_p_nom[is_wind].sum()),
    'total_gas_capacity_mw': float(gen_p_nom[~is_wind & (np.array(gen_carrier) == 'gas')].sum()),
    'total_nuclear_capacity_mw': float(gen_p_nom[np.array(gen_carrier) == 'nuclear'].sum()),
}

with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSummary Statistics:")
for k, v in summary.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 8. GENERATE FIGURES
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

colors = {
    'onshore wind': '#2ecc71',
    'gas': '#e74c3c',
    'nuclear': '#9b59b6',
    'storage_discharge': '#3498db',
    'storage_charge': '#1abc9c',
    'curtailment': '#f39c12',
    'shedding': '#c0392b',
    'demand': '#2c3e50',
}

hours = np.arange(n_hours)

# ---- Figure 1: Dispatch Stack ----
print("  Figure 1: Dispatch stack with load shedding")
fig, ax = plt.subplots(figsize=(14, 6))
bottom = np.zeros(n_hours)
for carrier, color_key in [('nuclear', 'nuclear'), ('onshore wind', 'onshore wind'), ('gas', 'gas')]:
    ax.fill_between(hours, bottom, bottom + gen_by_carrier[carrier],
                    label=carrier.title(), color=colors[color_key], alpha=0.85)
    bottom += gen_by_carrier[carrier]
ax.fill_between(hours, bottom, bottom + total_discharge,
                label='Storage Discharge', color=colors['storage_discharge'], alpha=0.7)
bottom += total_discharge
ax.fill_between(hours, bottom, bottom + total_shed_per_hour,
                label='Load Shedding', color=colors['shedding'], alpha=0.6, hatch='//')
ax.plot(hours, total_demand_per_hour.values, 'k-', linewidth=1.5, label='Demand', alpha=0.8)
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Power (MW)')
ax.set_title('Optimal Power Dispatch: Generation Stack with Load Shedding')
ax.legend(loc='upper left', ncol=2, fontsize=9)
ax.set_xlim(0, n_hours - 1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/dispatch_stack.png')
plt.close()

# ---- Figure 2: Wind Curtailment ----
print("  Figure 2: Wind curtailment")
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.fill_between(hours, 0, total_wind_available, alpha=0.5, color=colors['onshore wind'],
                  label='Available Wind')
ax1.fill_between(hours, 0, total_wind_dispatched, alpha=0.8, color=colors['onshore wind'],
                  label='Dispatched Wind')
ax1.set_xlabel('Hour of Week')
ax1.set_ylabel('Wind Power (MW)')
ax1.set_title('Wind Generation: Available vs. Dispatched')
ax1.legend(loc='upper left')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
ax2 = ax1.twinx()
ax2.plot(hours, wind_curtailment, color=colors['curtailment'], linewidth=1, alpha=0.7)
ax2.set_ylabel('Curtailment (MW)', color=colors['curtailment'])
ax2.tick_params(axis='y', labelcolor=colors['curtailment'])
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/wind_curtailment.png')
plt.close()

# ---- Figure 3: Storage Operation ----
print("  Figure 3: Storage operation")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for s in S:
    ax1.plot(hours, sto_soc_vals[s] / sto_e_nom[s] * 100,
             label=f'{storage.iloc[s]["bus"]} ({storage.iloc[s]["carrier"]})', linewidth=1.5)
ax1.set_ylabel('State of Charge (%)')
ax1.set_title('Storage State of Charge')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(0, 105)
ax2.fill_between(hours, 0, total_charge, alpha=0.7, color=colors['storage_charge'], label='Charging')
ax2.fill_between(hours, 0, -total_discharge, alpha=0.7, color=colors['storage_discharge'], label='Discharging')
ax2.set_xlabel('Hour of Week')
ax2.set_ylabel('Power (MW)')
ax2.set_title('Aggregate Storage Charge/Discharge')
ax2.legend(loc='upper left', fontsize=9)
ax2.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/storage_operation.png')
plt.close()

# ---- Figure 4: Generation Mix Pie Chart ----
print("  Figure 4: Generation mix pie chart")
fig, ax = plt.subplots(figsize=(7, 7))
labels = ['Onshore Wind', 'Gas', 'Nuclear']
sizes = [gen_by_carrier[c].sum() for c in ['onshore wind', 'gas', 'nuclear']]
colors_pie = [colors['onshore wind'], colors['gas'], colors['nuclear']]
wedges, texts, autotexts = ax.pie(sizes, explode=(0.02, 0.02, 0.02), labels=labels,
                                  colors=colors_pie, autopct='%1.1f%%',
                                  startangle=90, pctdistance=0.6)
for t in autotexts: t.set_fontsize(11)
ax.set_title('Generation Mix Over One Week')
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/generation_mix_pie.png')
plt.close()

# ---- Figure 5: Load Shedding by Hour ----
print("  Figure 5: Load shedding")
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(hours, 0, total_shed_per_hour, alpha=0.7, color=colors['shedding'])
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Load Shedding (MW)')
ax.set_title('Unserved Energy (Load Shedding) by Hour')
ax.set_xlim(0, n_hours - 1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/load_shedding.png')
plt.close()

# ---- Figure 6: Cost Breakdown ----
print("  Figure 6: Cost breakdown")
fig, ax = plt.subplots(figsize=(10, 5))
cost_per_hour_nuclear = gen_by_carrier['nuclear'] * 10
cost_per_hour_gas = gen_by_carrier['gas'] * 50
cost_per_hour_shed = total_shed_per_hour * VOLL
bottom = np.zeros(n_hours)
for vals, label, col in [
    (cost_per_hour_nuclear/1e6, 'Nuclear (€10/MWh)', colors['nuclear']),
    (cost_per_hour_gas/1e6, 'Gas (€50/MWh)', colors['gas']),
    (cost_per_hour_shed/1e6, 'Load Shedding (€10,000/MWh)', colors['shedding']),
]:
    ax.fill_between(hours, bottom, bottom + vals, label=label, color=col, alpha=0.8)
    bottom += vals
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Cost (M€/h)')
ax.set_title('Hourly System Cost by Component')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, n_hours - 1)
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/cost_breakdown.png')
plt.close()

# ---- Figure 7: Link Flow Utilization ----
print("  Figure 7: Link flow utilization")
fig, ax = plt.subplots(figsize=(14, 6))
link_labels = [f"{links.iloc[l]['bus0']}-{links.iloc[l]['bus1']}" for l in range(n_links)]
avg_util = np.mean(np.abs(link_flows), axis=1) / link_p_nom * 100
max_util = np.max(np.abs(link_flows), axis=1) / link_p_nom * 100
x_pos = np.arange(n_links)
width = 0.35
ax.bar(x_pos - width/2, avg_util, width, label='Average', color='steelblue', alpha=0.8)
ax.bar(x_pos + width/2, max_util, width, label='Peak', color='darkorange', alpha=0.8)
ax.set_xlabel('Transmission Link')
ax.set_ylabel('Utilization (%)')
ax.set_title('Transmission Link Utilization')
ax.set_xticks(x_pos)
ax.set_xticklabels(link_labels, rotation=45, ha='right', fontsize=7)
ax.legend(loc='upper right')
ax.axhline(y=100, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/link_utilization.png')
plt.close()

# ---- Figure 8: Power Balance Check ----
print("  Figure 8: Power balance check")
fig, ax = plt.subplots(figsize=(14, 5))
net_supply = total_gen_per_hour + total_discharge - total_charge + total_shed_per_hour
ax.plot(hours, total_demand_per_hour.values, 'k-', linewidth=2, label='Demand')
ax.plot(hours, net_supply, 'b--', linewidth=1.5, label='Supply (Gen+Storage+Shedding)')
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Power (MW)')
ax.set_title('Power Balance Check')
ax.legend(loc='upper right')
ax.set_xlim(0, n_hours - 1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/power_balance.png')
plt.close()

# ---- Figure 9: Network Topology ----
print("  Figure 9: Network topology map")
fig, ax = plt.subplots(figsize=(10, 10))
bus_x = buses['x'].values
bus_y = buses['y'].values
ax.scatter(bus_x, bus_y, c='steelblue', s=200, zorder=5, edgecolors='black', linewidth=1)
for l in range(n_links):
    b0, b1 = link_bus0_idx[l], link_bus1_idx[l]
    ax.plot([bus_x[b0], bus_x[b1]], [bus_y[b0], bus_y[b1]], 'gray', linewidth=1.5, alpha=0.6, zorder=2)
for i, name in enumerate(bus_names):
    ax.annotate(name, (bus_x[i], bus_y[i]), textcoords="offset points",
                xytext=(8, 8), fontsize=8, ha='left')
for s in S:
    bi = sto_bus_idx[s]
    ax.scatter(bus_x[bi], bus_y[bi], c='cyan', s=150, marker='^', zorder=7,
               edgecolors='black', linewidth=1)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markersize=10, label='Bus (400kV)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan',
           markersize=10, markeredgecolor='black', label='Pumped Hydro Storage'),
    Line2D([0], [0], color='gray', linewidth=1.5, label='Transmission Line'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
ax.set_xlabel('Longitude (approx)')
ax.set_ylabel('Latitude (approx)')
ax.set_title('GB Power System Network Topology (20-Bus, 400kV)')
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/network_topology.png')
plt.close()

# ---- Figure 10: Per-bus generation heatmap ----
print("  Figure 10: Per-bus generation heatmap")
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(gen_per_bus.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Bus')
ax.set_title('Generation by Bus and Hour (MW)')
ax.set_yticks(range(n_buses))
ax.set_yticklabels(bus_names, fontsize=8)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('MW')
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/generation_heatmap.png')
plt.close()

# ---- Figure 11: Per-Bus Demand vs Generation ----
print("  Figure 11: Per-bus demand vs generation")
fig, ax = plt.subplots(figsize=(12, 6))
avg_demand_per_bus = demand_array.mean(axis=0)
avg_gen_per_bus = gen_per_bus.mean(axis=1)
avg_shed_per_bus = shedding.mean(axis=1)
x_pos = np.arange(n_buses)
width = 0.3
ax.bar(x_pos - width, avg_demand_per_bus, width, label='Avg Demand', color='steelblue', alpha=0.8)
ax.bar(x_pos, avg_gen_per_bus, width, label='Avg Generation', color='darkorange', alpha=0.8)
ax.bar(x_pos + width, avg_shed_per_bus, width, label='Avg Shedding', color=colors['shedding'], alpha=0.7)
ax.set_xlabel('Bus')
ax.set_ylabel('Power (MW)')
ax.set_title('Average Power by Bus: Demand, Generation, and Shedding')
ax.set_xticks(x_pos)
ax.set_xticklabels(bus_names, fontsize=8)
ax.legend(loc='upper right', fontsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/bus_demand_gen.png')
plt.close()

# ---- Figure 12: Capacity factors ----
print("  Figure 12: Generator capacity factors")
fig, ax = plt.subplots(figsize=(12, 5))
gen_labels = [f"{generators.iloc[g]['bus']}/{gen_carrier[g]}" for g in G]
cfs = []
for g in G:
    total = gen_dispatch[g].sum()
    max_possible = gen_p_nom[g] * n_hours
    cfs.append(total / max_possible * 100)
colors_cf = [colors.get(carrier, 'gray') for carrier in gen_carrier]
ax.barh(range(len(G)), cfs, color=colors_cf, alpha=0.8)
ax.set_yticks(range(len(G)))
ax.set_yticklabels(gen_labels, fontsize=7)
ax.set_xlabel('Capacity Factor (%)')
ax.set_title('Generator Capacity Factors')
ax.axvline(x=100, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/capacity_factors.png')
plt.close()

# ============================================================
# 9. SAVE ADDITIONAL OUTPUTS
# ============================================================
print("\nSaving additional outputs...")

gen_output = []
for g in G:
    total_gen = gen_dispatch[g].sum()
    cf = total_gen / max(1, gen_p_nom[g] * n_hours) * 100
    gen_output.append({
        'bus': generators.iloc[g]['bus'],
        'carrier': gen_carrier[g],
        'capacity_mw': gen_p_nom[g],
        'total_generation_mwh': total_gen,
        'capacity_factor_pct': cf,
    })
pd.DataFrame(gen_output).to_csv(f'{OUTPUT_DIR}/generator_summary.csv', index=False)

sto_output = []
for s in S:
    sto_output.append({
        'bus': storage.iloc[s]['bus'],
        'type': storage.iloc[s]['carrier'],
        'total_charge_mwh': sto_charge[s].sum(),
        'total_discharge_mwh': sto_discharge[s].sum(),
        'round_trip_loss_mwh': sto_charge[s].sum() - sto_discharge[s].sum(),
        'avg_soc_pct': sto_soc_vals[s].mean() / sto_e_nom[s] * 100,
    })
pd.DataFrame(sto_output).to_csv(f'{OUTPUT_DIR}/storage_summary.csv', index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Total system cost: €{total_cost:,.0f}")
print(f"  Generation cost: €{gen_cost:,.0f}")
print(f"  Shedding cost: €{shed_cost:,.0f}")
print(f"Wind share of demand: {summary['wind_share_of_demand_pct']:.1f}%")
print(f"Wind curtailment: {summary['wind_curtailment_pct']:.1f}%")
print(f"Load shedding: {summary['load_shedding_pct']:.1f}% of demand")
print(f"Shedding hours: {summary['shedding_hours']}")
