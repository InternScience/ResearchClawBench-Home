#!/usr/bin/env python3
"""
Scenario comparison: Base case vs. Copper-plate (no network constraints)
"""

import numpy as np
import pandas as pd
from pulp import *
import json, os

DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
VOLL = 10000.0

def run_scenario(name, network_constrained=True):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    
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
    bus_to_idx = {name: i for i, name in enumerate(bus_names)}
    
    prob = LpProblem(f"GB_Dispatch_{name}", LpMinimize)
    
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
    
    link_p_nom = links['p_nom'].values
    
    demand_array = demand.values
    wind_cf_array = wind_cf.values
    total_demand_per_hour = demand.sum(axis=1).values
    
    # Variables
    p_gen = {}
    for g in G:
        for t in T:
            p_gen[g, t] = LpVariable(f"p_{g}_{t}", lowBound=0)
    
    c_sto = {}; d_sto = {}; soc = {}
    for s in S:
        for t in T:
            c_sto[s, t] = LpVariable(f"c_{s}_{t}", lowBound=0)
            d_sto[s, t] = LpVariable(f"d_{s}_{t}", lowBound=0)
            soc[s, t] = LpVariable(f"soc_{s}_{t}", lowBound=0, upBound=sto_e_nom[s])
    
    if network_constrained:
        f_link = {}
        for l in L:
            for t in T:
                f_link[l, t] = LpVariable(f"f_{l}_{t}", lowBound=-link_p_nom[l], upBound=link_p_nom[l])
    
    shed = {}
    for b in B:
        for t in T:
            shed[b, t] = LpVariable(f"shed_{b}_{t}", lowBound=0, upBound=demand_array[t,b])
    
    # Generator limits
    for g in G:
        for t in T:
            if is_wind[g]:
                bus = gen_bus_idx[g]
                avail = gen_p_nom[g] * wind_cf_array[t, bus]
                p_gen[g, t].upBound = avail
            else:
                p_gen[g, t].upBound = gen_p_nom[g]
    
    # Storage limits
    for s in S:
        for t in T:
            c_sto[s, t].upBound = sto_p_nom[s]
            d_sto[s, t].upBound = sto_p_nom[s]
    
    if network_constrained:
        link_bus0_idx = [bus_to_idx[b] for b in links['bus0']]
        link_bus1_idx = [bus_to_idx[b] for b in links['bus1']]
        
        # Power balance per bus
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
    else:
        # Copper-plate: single power balance for total system
        for t in T:
            total_gen = lpSum([p_gen[g, t] for g in G])
            total_charge = lpSum([c_sto[s, t] for s in S])
            total_discharge = lpSum([d_sto[s, t] for s in S])
            total_shed = lpSum([shed[b, t] for b in B])
            lhs = total_gen + total_discharge + total_shed
            rhs = total_demand_per_hour[t] + total_charge
            prob += (lhs == rhs), f"bal_t{t}"
    
    # Storage dynamics
    for s in S:
        prob += (soc[s, 0] == 0.5 * sto_e_nom[s]), f"soc_init_{s}"
        for t in range(1, n_hours):
            prob += (soc[s, t] == soc[s, t-1] + sto_eff[s]*c_sto[s,t] - d_sto[s,t]), f"soc_{s}_{t}"
        prob += (soc[s, n_hours-1] == 0.5 * sto_e_nom[s]), f"soc_final_{s}"
    
    # Objective
    obj = lpSum([gen_mc[g] * p_gen[g, t] for g in G for t in T])
    obj += lpSum([VOLL * shed[b, t] for b in B for t in T])
    prob += obj, "total_cost"
    
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=300))
    
    # Extract
    gen_dispatch = np.zeros((len(G), n_hours))
    for g in G:
        for t in T:
            gen_dispatch[g, t] = value(p_gen[g, t])
    
    shedding = np.zeros((n_buses, n_hours))
    for b in B:
        for t in T:
            shedding[b, t] = value(shed[b, t])
    
    total_shed_per_hour = shedding.sum(axis=0)
    
    gen_by_carrier = {}
    for carrier in ['onshore wind', 'gas', 'nuclear']:
        mask = np.array([c == carrier for c in gen_carrier])
        gen_by_carrier[carrier] = gen_dispatch[mask].sum(axis=0)
    
    # Wind curtailment
    wind_available_arr = np.zeros((sum(is_wind), n_hours))
    widx = 0
    for g in G:
        if is_wind[g]:
            bus = gen_bus_idx[g]
            wind_available_arr[widx] = gen_p_nom[g] * wind_cf_array[:, bus]
            widx += 1
    total_wind_avail = wind_available_arr.sum(axis=0)
    total_wind_disp = gen_dispatch[is_wind].sum(axis=0)
    wind_curt = total_wind_avail - total_wind_disp
    
    gen_cost = sum(gen_dispatch[g].sum() * gen_mc[g] for g in G)
    shed_cost = shedding.sum() * VOLL
    
    result = {
        'scenario': name,
        'total_cost_eur': float(gen_cost + shed_cost),
        'generation_cost_eur': float(gen_cost),
        'shedding_cost_eur': float(shed_cost),
        'total_demand_mwh': float(total_demand_per_hour.sum()),
        'total_generation_mwh': float(gen_dispatch.sum()),
        'total_shedding_mwh': float(shedding.sum()),
        'wind_gen_mwh': float(gen_by_carrier['onshore wind'].sum()),
        'gas_gen_mwh': float(gen_by_carrier['gas'].sum()),
        'nuclear_gen_mwh': float(gen_by_carrier['nuclear'].sum()),
        'wind_available_mwh': float(total_wind_avail.sum()),
        'wind_curtailed_mwh': float(wind_curt.sum()),
        'wind_curtailment_pct': float(wind_curt.sum() / max(1, total_wind_avail.sum()) * 100),
        'load_shedding_pct': float(shedding.sum() / total_demand_per_hour.sum() * 100),
        
        # Per-hour data for comparison plots
        'total_shed_per_hour': total_shed_per_hour.tolist(),
        'wind_gen_per_hour': gen_by_carrier['onshore wind'].tolist(),
        'gas_gen_per_hour': gen_by_carrier['gas'].tolist(),
        'nuclear_gen_per_hour': gen_by_carrier['nuclear'].tolist(),
        'wind_curt_per_hour': wind_curt.tolist(),
    }
    
    return result

# Run both scenarios
print("Running scenario comparison...")
base_result = run_scenario("Base (Network Constrained)", network_constrained=True)
copper_result = run_scenario("Copper-plate (No Network)", network_constrained=False)

# Save comparison
comparison = {
    'base': base_result,
    'copper_plate': copper_result,
}

with open(f'{OUTPUT_DIR}/scenario_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# Print comparison
print("\n" + "=" * 60)
print("SCENARIO COMPARISON")
print("=" * 60)
for key in ['total_cost_eur', 'generation_cost_eur', 'shedding_cost_eur',
            'wind_gen_mwh', 'gas_gen_mwh', 'nuclear_gen_mwh',
            'wind_curtailed_mwh', 'wind_curtailment_pct', 'load_shedding_pct']:
    bv = base_result[key]
    cv = copper_result[key]
    print(f"  {key}:")
    print(f"    Base:       {bv:,.2f}")
    print(f"    Copper:     {cv:,.2f}")
    if isinstance(bv, float) and bv != 0:
        pct = (cv - bv) / abs(bv) * 100
        print(f"    Δ:          {cv - bv:+,.2f} ({pct:+.1f}%)")

# Generate comparison figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPORT_IMG_DIR = 'report/images'
hours = np.arange(168)

# Figure 13: Scenario comparison - shedding
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(hours, base_result['total_shed_per_hour'], 'r-', linewidth=1.5, alpha=0.8, label='Base (Network Constrained)')
ax.plot(hours, copper_result['total_shed_per_hour'], 'b--', linewidth=1.5, alpha=0.8, label='Copper-plate (No Network)')
ax.fill_between(hours, copper_result['total_shed_per_hour'], base_result['total_shed_per_hour'],
                alpha=0.2, color='red', label='Network-induced additional shedding')
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Load Shedding (MW)')
ax.set_title('Scenario Comparison: Load Shedding with and without Network Constraints')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 167)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/scenario_shedding.png')
plt.close()
print("  Saved: report/images/scenario_shedding.png")

# Figure 14: Scenario comparison - wind curtailment
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(hours, base_result['wind_curt_per_hour'], 'orange', linewidth=1.5, alpha=0.8, label='Base (Network Constrained)')
ax.plot(hours, copper_result['wind_curt_per_hour'], 'g--', linewidth=1.5, alpha=0.8, label='Copper-plate (No Network)')
ax.set_xlabel('Hour of Week')
ax.set_ylabel('Wind Curtailment (MW)')
ax.set_title('Scenario Comparison: Wind Curtailment with and without Network Constraints')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 167)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
plt.tight_layout()
fig.savefig(f'{REPORT_IMG_DIR}/scenario_curtailment.png')
plt.close()
print("  Saved: report/images/scenario_curtailment.png")

print("\nScenario comparison complete!")
