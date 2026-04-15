"""
Optimal Power Dispatch for Great Britain Energy System
======================================================
Linear programming formulation for hourly economic dispatch
with network constraints, storage, and renewable generation.

Solves: min sum_t sum_g c_g * p_{g,t} + VOLL * sum_{b,t} shed_{b,t}
subject to:
  - Power balance at each node (with load shedding slack)
  - Generator capacity limits
  - Renewable availability (capacity factors)
  - Transmission line capacity limits  
  - Storage dynamics and limits
  - Non-negativity

Uses scipy.optimize.linprog (HiGHS solver) for efficient LP solution.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================
buses = pd.read_csv(f'{DATA_DIR}/buses.csv')
links = pd.read_csv(f'{DATA_DIR}/links.csv')
generators = pd.read_csv(f'{DATA_DIR}/generators.csv')
storage_units = pd.read_csv(f'{DATA_DIR}/storage.csv')
demand = pd.read_csv(f'{DATA_DIR}/demand.csv')
wind_cf = pd.read_csv(f'{DATA_DIR}/wind_cf.csv')

T = len(demand)
N_buses = len(buses)
N_links = len(links)
bus_names = buses['name'].tolist()
bus_idx = {name: i for i, name in enumerate(bus_names)}

print(f"System: {N_buses} buses, {N_links} links, {T} timesteps")
print(f"Generators: {len(generators)} units")
print(f"Storage: {len(storage_units)} units")

# ============================================================
# 2. GENERATOR ORGANIZATION
# ============================================================
gen_data = {}
for _, row in generators.iterrows():
    b = bus_idx[row['bus']]
    carrier = row['carrier']
    key = (b, carrier)
    if key not in gen_data:
        gen_data[key] = {'p_nom': 0, 'marginal_cost': row['marginal_cost']}
    gen_data[key]['p_nom'] += row['p_nom']

gen_keys = sorted(gen_data.keys(), key=lambda x: (x[0], x[1]))
N_gen_groups = len(gen_keys)
carriers = sorted(set(k[1] for k in gen_keys))
print(f"Generator groups: {N_gen_groups}")
print(f"Carrier types: {carriers}")

# ============================================================
# 3. STORAGE ORGANIZATION
# ============================================================
stor_data = {}
for _, row in storage_units.iterrows():
    b = bus_idx[row['bus']]
    stor_data[b] = {
        'p_nom': row['p_nom'],
        'e_nom': row['e_nom'],
        'efficiency': row['efficiency']
    }
stor_bus_indices = sorted(stor_data.keys())
N_stor = len(stor_bus_indices)
print(f"Storage at buses: {stor_bus_indices}")

# ============================================================
# 4. DEMAND AND WIND DATA
# ============================================================
demand_matrix = demand[bus_names].values
wind_cf_matrix = wind_cf[bus_names].values
print(f"Demand range: {demand_matrix.sum(axis=1).min():.0f} - {demand_matrix.sum(axis=1).max():.0f} MW")

# ============================================================
# 5. LP FORMULATION
# ============================================================
# Blocks:
# 0: gen dispatch [N_gen_groups, T]
# 1: link flows [N_links, T]
# 2: stor discharge [N_stor, T]
# 3: stor charge [N_stor, T]
# 4: stor energy [N_stor, T]
# 5: load shedding [N_buses, T]

B0 = 0
B1 = N_gen_groups * T
B2 = B1 + N_links * T
B3 = B2 + N_stor * T
B4 = B3 + N_stor * T
B5 = B4 + N_stor * T
N_total = B5 + N_buses * T

def gi(block, idx1, t):
    return block + idx1 * T + t

VOLL = 5000

# Objective
c = np.zeros(N_total)
for g, (b, carrier) in enumerate(gen_keys):
    cost = gen_data[(b, carrier)]['marginal_cost']
    for t in range(T):
        c[gi(B0, g, t)] = cost
for b in range(N_buses):
    for t in range(T):
        c[gi(B5, b, t)] = VOLL

# Inequality constraints
A_ub_rows = []
b_ub_vals = []

# Gen capacity
for g, (b, carrier) in enumerate(gen_keys):
    pn = gen_data[(b, carrier)]['p_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B0, g, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(pn)

# Wind availability
for g, (b, carrier) in enumerate(gen_keys):
    if carrier != 'onshore wind':
        continue
    pn = gen_data[(b, carrier)]['p_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B0, g, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(wind_cf_matrix[t, b] * pn)

# Link capacity
for l in range(N_links):
    pn = links.iloc[l]['p_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B1, l, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(pn)
        row = np.zeros(N_total)
        row[gi(B1, l, t)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(pn)

# Stor discharge
for s, b in enumerate(stor_bus_indices):
    pn = stor_data[b]['p_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B2, s, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(pn)

# Stor charge
for s, b in enumerate(stor_bus_indices):
    pn = stor_data[b]['p_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B3, s, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(pn)

# Stor energy
for s, b in enumerate(stor_bus_indices):
    en = stor_data[b]['e_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B4, s, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(en)

# Shedding limit
for b in range(N_buses):
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B5, b, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(demand_matrix[t, b])

A_ub = np.array(A_ub_rows)
b_ub = np.array(b_ub_vals)
print(f"Ineq constraints: {len(b_ub):,}")

# Equality constraints
A_eq_rows = []
b_eq_vals = []

# Power balance
for t in range(T):
    for b in range(N_buses):
        row = np.zeros(N_total)
        # Generation
        for g, (gb, carrier) in enumerate(gen_keys):
            if gb == b:
                row[gi(B0, g, t)] = 1.0
        # Links
        for l in range(N_links):
            link = links.iloc[l]
            l0 = bus_idx[link['bus0']]
            l1 = bus_idx[link['bus1']]
            if l0 == b:
                row[gi(B1, l, t)] = -1.0
            elif l1 == b:
                row[gi(B1, l, t)] = 1.0
        # Storage
        for s, sb in enumerate(stor_bus_indices):
            if sb == b:
                row[gi(B2, s, t)] = 1.0
                row[gi(B3, s, t)] = -1.0
        # Shedding
        row[gi(B5, b, t)] = 1.0
        
        A_eq_rows.append(row)
        b_eq_vals.append(demand_matrix[t, b])

# Storage dynamics
for s, b in enumerate(stor_bus_indices):
    eta = stor_data[b]['efficiency']
    e_init = 0.5 * stor_data[b]['e_nom']
    for t in range(T):
        row = np.zeros(N_total)
        row[gi(B4, s, t)] = 1.0
        if t == 0:
            row[gi(B3, s, 0)] = -eta
            row[gi(B2, s, 0)] = 1.0 / eta
            rhs = e_init
        else:
            row[gi(B4, s, t-1)] = -1.0
            row[gi(B3, s, t)] = -eta
            row[gi(B2, s, t)] = 1.0 / eta
            rhs = 0.0
        A_eq_rows.append(row)
        b_eq_vals.append(rhs)

# Periodic constraint
for s, b in enumerate(stor_bus_indices):
    row = np.zeros(N_total)
    row[gi(B4, s, T-1)] = 1.0
    row[gi(B4, s, 0)] = -1.0
    A_eq_rows.append(row)
    b_eq_vals.append(0.0)

A_eq = np.array(A_eq_rows)
b_eq = np.array(b_eq_vals)
print(f"Eq constraints: {len(b_eq):,}")

bounds = [(0, None)] * N_total

# ============================================================
# 6. SOLVE
# ============================================================
print("\nSolving LP...")
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method='highs',
                 options={'disp': True, 'time_limit': 600})

if result.success:
    print(f"\nOptimization successful!")
    print(f"Total cost: ${result.fun:,.2f}")
else:
    print(f"\nFailed: {result.message}")
    raise RuntimeError("LP failed")

# ============================================================
# 7. EXTRACT RESULTS
# ============================================================
dispatch = np.zeros((N_gen_groups, T))
for g in range(N_gen_groups):
    for t in range(T):
        dispatch[g, t] = result.x[gi(B0, g, t)]

link_flows = np.zeros((N_links, T))
for l in range(N_links):
    for t in range(T):
        link_flows[l, t] = result.x[gi(B1, l, t)]

stor_disch = np.zeros((N_stor, T))
stor_ch = np.zeros((N_stor, T))
stor_e = np.zeros((N_stor, T))
for s in range(N_stor):
    for t in range(T):
        stor_disch[s, t] = result.x[gi(B2, s, t)]
        stor_ch[s, t] = result.x[gi(B3, s, t)]
        stor_e[s, t] = result.x[gi(B4, s, t)]

load_shedding = np.zeros((N_buses, T))
for b in range(N_buses):
    for t in range(T):
        load_shedding[b, t] = result.x[gi(B5, b, t)]

# Wind analysis
wind_dispatch = np.zeros((N_buses, T))
wind_available = np.zeros((N_buses, T))
for g, (b, carrier) in enumerate(gen_keys):
    if carrier == 'onshore wind':
        wind_dispatch[b, :] = dispatch[g, :]
        wind_available[b, :] = wind_cf_matrix[:, b] * gen_data[(b, carrier)]['p_nom']

curtailment = np.maximum(0, wind_available - wind_dispatch)

# ============================================================
# 8. SAVE RESULTS
# ============================================================
total_fuel_cost = sum(dispatch[g, :].sum() * gen_data[(b, carrier)]['marginal_cost']
                      for g, (b, carrier) in enumerate(gen_keys))
total_shedding_cost = load_shedding.sum() * VOLL

results = {
    'total_cost': float(result.fun),
    'total_fuel_cost': float(total_fuel_cost),
    'total_shedding_cost': float(total_shedding_cost),
    'total_demand_mwh': float(demand_matrix.sum()),
    'total_served_mwh': float(demand_matrix.sum() - load_shedding.sum()),
    'total_shedding_mwh': float(load_shedding.sum()),
    'shedding_pct': float(100 * load_shedding.sum() / demand_matrix.sum()),
    'total_generation_mwh': float(dispatch.sum()),
    'total_wind_available_mwh': float(wind_available.sum()),
    'total_wind_dispatched_mwh': float(wind_dispatch.sum()),
    'total_curtailment_mwh': float(curtailment.sum()),
    'curtailment_pct': float(100 * curtailment.sum() / wind_available.sum()),
    'total_storage_throughput_mwh': float(stor_disch.sum()),
}

carrier_summary = {}
for g, (b, carrier) in enumerate(gen_keys):
    if carrier not in carrier_summary:
        carrier_summary[carrier] = {'total_mwh': 0, 'mean_mw': 0, 'max_mw': 0, 'cost': 0, 'n_units': 0}
    carrier_summary[carrier]['total_mwh'] += float(dispatch[g, :].sum())
    carrier_summary[carrier]['mean_mw'] += float(dispatch[g, :].mean())
    carrier_summary[carrier]['max_mw'] += float(dispatch[g, :].max())
    carrier_summary[carrier]['cost'] += float(dispatch[g, :].sum() * gen_data[(b, carrier)]['marginal_cost'])
    carrier_summary[carrier]['n_units'] += 1
results['carrier_summary'] = carrier_summary

gas_disp = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'gas')
nuc_disp = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'nuclear')
wind_disp_total = sum(dispatch[g, :] for g, (_, c) in enumerate(gen_keys) if c == 'onshore wind')

hourly_totals = {
    'total_demand_mw': demand_matrix.sum(axis=1).tolist(),
    'total_served_mw': (demand_matrix.sum(axis=1) - load_shedding.sum(axis=0)).tolist(),
    'shedding_mw': load_shedding.sum(axis=0).tolist(),
    'gas_generation_mw': gas_disp.tolist(),
    'nuclear_generation_mw': nuc_disp.tolist(),
    'wind_generation_mw': wind_disp_total.tolist(),
    'curtailment_mw': curtailment.sum(axis=0).tolist(),
    'wind_available_mw': wind_available.sum(axis=0).tolist(),
    'stor_disch_mw': stor_disch.sum(axis=0).tolist(),
    'stor_ch_mw': stor_ch.sum(axis=0).tolist(),
}
results['hourly_totals'] = hourly_totals

bus_results = {}
for b, bn in enumerate(bus_names):
    bus_results[bn] = {
        'total_demand_mwh': float(demand_matrix[:, b].sum()),
        'total_served_mwh': float(demand_matrix[:, b].sum() - load_shedding[b, :].sum()),
        'total_shedding_mwh': float(load_shedding[b, :].sum()),
        'shedding_pct': float(100 * load_shedding[b, :].sum() / max(demand_matrix[:, b].sum(), 1)),
    }
results['bus_results'] = bus_results

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save(f'{OUTPUT_DIR}/dispatch.npy', dispatch)
np.save(f'{OUTPUT_DIR}/link_flows.npy', link_flows)
np.save(f'{OUTPUT_DIR}/stor_disch.npy', stor_disch)
np.save(f'{OUTPUT_DIR}/stor_ch.npy', stor_ch)
np.save(f'{OUTPUT_DIR}/stor_e.npy', stor_e)
np.save(f'{OUTPUT_DIR}/curtailment.npy', curtailment)
np.save(f'{OUTPUT_DIR}/wind_available.npy', wind_available)
np.save(f'{OUTPUT_DIR}/wind_dispatch.npy', wind_dispatch)
np.save(f'{OUTPUT_DIR}/demand_matrix.npy', demand_matrix)
np.save(f'{OUTPUT_DIR}/load_shedding.npy', load_shedding)

meta = {
    'gen_keys': [(int(b), c) for b, c in gen_keys],
    'stor_bus_indices': [int(b) for b in stor_bus_indices],
    'bus_names': bus_names,
    'N_gen_groups': N_gen_groups,
    'N_stor': N_stor,
    'T': T,
    'VOLL': VOLL,
}
with open(f'{OUTPUT_DIR}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nResults saved to outputs/")
print(f"Fuel cost: ${results['total_fuel_cost']:,.2f}")
print(f"Shedding cost: ${results['total_shedding_cost']:,.2f}")
print(f"Total cost: ${results['total_cost']:,.2f}")
print(f"Demand: {results['total_demand_mwh']:,.0f} MWh")
print(f"Served: {results['total_served_mwh']:,.0f} MWh")
print(f"Shedding: {results['total_shedding_mwh']:,.0f} MWh ({results['shedding_pct']:.1f}%)")
print(f"Wind dispatched: {results['total_wind_dispatched_mwh']:,.0f} MWh")
print(f"Wind curtailed: {results['total_curtailment_mwh']:,.0f} MWh ({results['curtailment_pct']:.1f}%)")
