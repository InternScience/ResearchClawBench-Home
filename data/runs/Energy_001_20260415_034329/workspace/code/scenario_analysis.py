"""
Scenario Analysis for GB Energy System
=======================================
Runs alternative scenarios:
1. Base case (already solved)
2. High wind capacity (2x wind)
3. No storage
4. Increased transmission capacity
5. Higher demand (future growth)
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
# LOAD DATA
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

def solve_dispatch(generators_df, storage_df, demand_df, wind_cf_df, links_df, voll=5000):
    """Generic LP solver for economic dispatch with load shedding."""
    gen_data = {}
    for _, row in generators_df.iterrows():
        b = bus_idx[row['bus']]
        carrier = row['carrier']
        key = (b, carrier)
        if key not in gen_data:
            gen_data[key] = {'p_nom': 0, 'marginal_cost': row['marginal_cost']}
        gen_data[key]['p_nom'] += row['p_nom']
    
    gen_keys = sorted(gen_data.keys(), key=lambda x: (x[0], x[1]))
    N_gen_groups = len(gen_keys)
    
    stor_data = {}
    for _, row in storage_df.iterrows():
        b = bus_idx[row['bus']]
        stor_data[b] = {
            'p_nom': row['p_nom'],
            'e_nom': row['e_nom'],
            'efficiency': row['efficiency']
        }
    stor_bus_indices = sorted(stor_data.keys())
    N_stor = len(stor_bus_indices)
    
    demand_matrix = demand_df[bus_names].values
    wind_cf_matrix = wind_cf_df[bus_names].values
    
    B0 = 0
    B1 = N_gen_groups * T
    B2 = B1 + N_links * T
    B3 = B2 + N_stor * T
    B4 = B3 + N_stor * T
    B5 = B4 + N_stor * T
    N_total = B5 + N_buses * T
    
    def gi(block, idx1, t):
        return block + idx1 * T + t
    
    c = np.zeros(N_total)
    for g, (b, carrier) in enumerate(gen_keys):
        cost = gen_data[(b, carrier)]['marginal_cost']
        for t in range(T):
            c[gi(B0, g, t)] = cost
    for b in range(N_buses):
        for t in range(T):
            c[gi(B5, b, t)] = voll
    
    A_ub_rows = []
    b_ub_vals = []
    
    for g, (b, carrier) in enumerate(gen_keys):
        pn = gen_data[(b, carrier)]['p_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B0, g, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(pn)
    
    for g, (b, carrier) in enumerate(gen_keys):
        if carrier != 'onshore wind':
            continue
        pn = gen_data[(b, carrier)]['p_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B0, g, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(wind_cf_matrix[t, b] * pn)
    
    for l in range(N_links):
        pn = links_df.iloc[l]['p_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B1, l, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(pn)
            row = np.zeros(N_total)
            row[gi(B1, l, t)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(pn)
    
    for s, b in enumerate(stor_bus_indices):
        pn = stor_data[b]['p_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B2, s, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(pn)
    
    for s, b in enumerate(stor_bus_indices):
        pn = stor_data[b]['p_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B3, s, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(pn)
    
    for s, b in enumerate(stor_bus_indices):
        en = stor_data[b]['e_nom']
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B4, s, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(en)
    
    for b in range(N_buses):
        for t in range(T):
            row = np.zeros(N_total)
            row[gi(B5, b, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(demand_matrix[t, b])
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    A_eq_rows = []
    b_eq_vals = []
    
    for t in range(T):
        for b in range(N_buses):
            row = np.zeros(N_total)
            for g, (gb, carrier) in enumerate(gen_keys):
                if gb == b:
                    row[gi(B0, g, t)] = 1.0
            for l in range(N_links):
                link = links_df.iloc[l]
                l0 = bus_idx[link['bus0']]
                l1 = bus_idx[link['bus1']]
                if l0 == b:
                    row[gi(B1, l, t)] = -1.0
                elif l1 == b:
                    row[gi(B1, l, t)] = 1.0
            for s, sb in enumerate(stor_bus_indices):
                if sb == b:
                    row[gi(B2, s, t)] = 1.0
                    row[gi(B3, s, t)] = -1.0
            row[gi(B5, b, t)] = 1.0
            A_eq_rows.append(row)
            b_eq_vals.append(demand_matrix[t, b])
    
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
    
    for s, b in enumerate(stor_bus_indices):
        row = np.zeros(N_total)
        row[gi(B4, s, T-1)] = 1.0
        row[gi(B4, s, 0)] = -1.0
        A_eq_rows.append(row)
        b_eq_vals.append(0.0)
    
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_vals)
    bounds = [(0, None)] * N_total
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs',
                     options={'disp': False, 'time_limit': 600})
    
    if not result.success:
        return None
    
    dispatch = np.zeros((N_gen_groups, T))
    for g in range(N_gen_groups):
        for t in range(T):
            dispatch[g, t] = result.x[gi(B0, g, t)]
    
    load_shedding = np.zeros((N_buses, T))
    for b in range(N_buses):
        for t in range(T):
            load_shedding[b, t] = result.x[gi(B5, b, t)]
    
    stor_disch = np.zeros((N_stor, T))
    stor_ch = np.zeros((N_stor, T))
    for s in range(N_stor):
        for t in range(T):
            stor_disch[s, t] = result.x[gi(B2, s, t)]
            stor_ch[s, t] = result.x[gi(B3, s, t)]
    
    wind_available = np.zeros((N_buses, T))
    for g, (b, carrier) in enumerate(gen_keys):
        if carrier == 'onshore wind':
            wind_available[b, :] = wind_cf_matrix[:, b] * gen_data[(b, carrier)]['p_nom']
    
    total_fuel_cost = sum(dispatch[g, :].sum() * gen_data[(b, carrier)]['marginal_cost']
                          for g, (b, carrier) in enumerate(gen_keys))
    
    return {
        'total_cost': float(result.fun),
        'fuel_cost': float(total_fuel_cost),
        'shedding_cost': float(load_shedding.sum() * voll),
        'total_demand_mwh': float(demand_matrix.sum()),
        'total_served_mwh': float(demand_matrix.sum() - load_shedding.sum()),
        'total_shedding_mwh': float(load_shedding.sum()),
        'shedding_pct': float(100 * load_shedding.sum() / demand_matrix.sum()),
        'dispatch': dispatch,
        'gen_keys': gen_keys,
        'gen_data': gen_data,
        'stor_disch': stor_disch,
        'stor_ch': stor_ch,
        'wind_available': wind_available,
        'N_gen_groups': N_gen_groups,
    }

# ============================================================
# SCENARIO DEFINITIONS
# ============================================================
scenarios = {}

# Scenario 1: Base case (already solved, re-solve for consistency)
scenarios['Base Case'] = {
    'generators': generators.copy(),
    'storage': storage_units.copy(),
    'demand': demand.copy(),
    'wind_cf': wind_cf.copy(),
    'links': links.copy(),
}

# Scenario 2: High wind (2x onshore wind capacity)
gen_high_wind = generators.copy()
mask = gen_high_wind['carrier'] == 'onshore wind'
gen_high_wind.loc[mask, 'p_nom'] *= 2.0
scenarios['High Wind (2x)'] = {
    'generators': gen_high_wind,
    'storage': storage_units.copy(),
    'demand': demand.copy(),
    'wind_cf': wind_cf.copy(),
    'links': links.copy(),
}

# Scenario 3: No storage
scenarios['No Storage'] = {
    'generators': generators.copy(),
    'storage': pd.DataFrame(columns=storage_units.columns),
    'demand': demand.copy(),
    'wind_cf': wind_cf.copy(),
    'links': links.copy(),
}

# Scenario 4: Enhanced transmission (2x link capacity)
links_enhanced = links.copy()
links_enhanced['p_nom'] *= 2.0
scenarios['Enhanced Transmission (2x)'] = {
    'generators': generators.copy(),
    'storage': storage_units.copy(),
    'demand': demand.copy(),
    'wind_cf': wind_cf.copy(),
    'links': links_enhanced,
}

# Scenario 5: High demand (+20%)
demand_high = demand.copy()
for col in demand_high.columns:
    demand_high[col] *= 1.2
scenarios['High Demand (+20%)'] = {
    'generators': generators.copy(),
    'storage': storage_units.copy(),
    'demand': demand_high,
    'wind_cf': wind_cf.copy(),
    'links': links.copy(),
}

# ============================================================
# RUN SCENARIOS
# ============================================================
print("Running scenario analysis...")
scenario_results = {}

for name, config in scenarios.items():
    print(f"\n--- {name} ---")
    res = solve_dispatch(config['generators'], config['storage'], 
                         config['demand'], config['wind_cf'], config['links'])
    if res is not None:
        scenario_results[name] = res
        print(f"  Cost: ${res['total_cost']:,.0f}")
        print(f"  Shedding: {res['shedding_pct']:.1f}%")
        print(f"  Served: {res['total_served_mwh']:,.0f} MWh")
    else:
        print(f"  FAILED")

# ============================================================
# SAVE COMPARISON
# ============================================================
comparison = {}
for name, res in scenario_results.items():
    comparison[name] = {
        'total_cost': res['total_cost'],
        'fuel_cost': res['fuel_cost'],
        'shedding_cost': res['shedding_cost'],
        'total_demand_mwh': res['total_demand_mwh'],
        'total_served_mwh': res['total_served_mwh'],
        'total_shedding_mwh': res['total_shedding_mwh'],
        'shedding_pct': res['shedding_pct'],
    }

with open(f'{OUTPUT_DIR}/scenario_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# ============================================================
# GENERATE COMPARISON FIGURE
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
names = list(scenario_results.keys())

# Cost comparison
costs = [scenario_results[n]['total_cost'] for n in names]
axes[0, 0].barh(names, costs, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
axes[0, 0].set_xlabel('Total Cost ($)')
axes[0, 0].set_title('Total System Cost Comparison')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Shedding comparison
shedding_pcts = [scenario_results[n]['shedding_pct'] for n in names]
axes[0, 1].barh(names, shedding_pcts, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
axes[0, 1].set_xlabel('Load Shedding (%)')
axes[0, 1].set_title('Load Shedding Percentage')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Served energy
served = [scenario_results[n]['total_served_mwh']/1000 for n in names]
axes[1, 0].barh(names, served, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
axes[1, 0].set_xlabel('Energy Served (GWh)')
axes[1, 0].set_title('Total Energy Served')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Fuel cost vs shedding cost
fuel_costs = [scenario_results[n]['fuel_cost']/1e6 for n in names]
shed_costs = [scenario_results[n]['shedding_cost']/1e6 for n in names]
x = np.arange(len(names))
width = 0.35
axes[1, 1].bar(x - width/2, fuel_costs, width, label='Fuel Cost', color='#FF9800')
axes[1, 1].bar(x + width/2, shed_costs, width, label='Shedding Cost', color='#F44336')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
axes[1, 1].set_ylabel('Cost ($M)')
axes[1, 1].set_title('Cost Components by Scenario')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/fig11_scenario_comparison.png', dpi=150)
plt.close()

print("\n=== Scenario analysis complete ===")
print(f"Comparison saved to: {OUTPUT_DIR}/scenario_comparison.json")
print(f"Figure saved to: report/images/fig11_scenario_comparison.png")
