"""
Optimal Power Dispatch Model for GB 20-Bus System

Formulates and solves a linear programming (LP) optimal dispatch problem
with transport-model power flow, storage dynamics, and scenario analysis.

Author: Research Agent
Date: 2026-05-18
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    buses = pd.read_csv('data/buses.csv')
    links = pd.read_csv('data/links.csv')
    generators = pd.read_csv('data/generators.csv')
    storage = pd.read_csv('data/storage.csv')
    demand = pd.read_csv('data/demand.csv')
    wind_cf = pd.read_csv('data/wind_cf.csv')
    return buses, links, generators, storage, demand, wind_cf

# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
def build_dispatch_model(buses, links, generators, storage_units, demand, wind_cf,
                         voLL=6000.0, curtailment_cost=0.1,
                         line_capacity_factor=1.0, wind_capacity_factor=1.0,
                         storage_enabled=True, wind_enabled=True,
                         gas_enabled=True, nuclear_enabled=True):
    """
    Build a CVXPY LP model for multi-period optimal dispatch.
    Uses a transport model (Kirchhoff Current Law only) for robustness.
    """
    n_buses = len(buses)
    n_hours = len(demand)
    bus_names = list(buses['name'].values)
    bus_idx = {name: i for i, name in enumerate(bus_names)}
    
    # Generator sets by type and bus
    gen_by_bus_type = {}
    for _, g in generators.iterrows():
        b = g['bus']
        t = g['carrier']
        if (b, t) not in gen_by_bus_type:
            gen_by_bus_type[(b, t)] = []
        gen_by_bus_type[(b, t)].append(g.to_dict())
    
    # Storage by bus
    storage_by_bus = {}
    for _, s in storage_units.iterrows():
        b = s['bus']
        if b not in storage_by_bus:
            storage_by_bus[b] = []
        storage_by_bus[b].append(s.to_dict())
    
    n_storage = len(storage_units)
    
    # Link incidence
    link_list = []
    for _, l in links.iterrows():
        link_list.append({
            'bus0': l['bus0'],
            'bus1': l['bus1'],
            'p_nom': l['p_nom'] * line_capacity_factor,
            'length': l['length'],
            'idx0': bus_idx[l['bus0']],
            'idx1': bus_idx[l['bus1']]
        })
    n_links = len(link_list)
    
    # Wind potential by bus and hour
    wind_potential = np.zeros((n_buses, n_hours))
    for (b, t), gens in gen_by_bus_type.items():
        if t == 'onshore wind' and wind_enabled:
            bi = bus_idx[b]
            cap = sum(g['p_nom'] for g in gens) * wind_capacity_factor
            wind_potential[bi, :] = cap * wind_cf[b].values
    
    # Gas capacities by bus
    gas_cap = np.zeros(n_buses)
    gas_mc = 50.0
    for (b, t), gens in gen_by_bus_type.items():
        if t == 'gas' and gas_enabled:
            bi = bus_idx[b]
            gas_cap[bi] = sum(g['p_nom'] for g in gens)
    
    # Nuclear capacities by bus
    nuclear_cap = np.zeros(n_buses)
    nuclear_mc = 10.0
    for (b, t), gens in gen_by_bus_type.items():
        if t == 'nuclear' and nuclear_enabled:
            bi = bus_idx[b]
            nuclear_cap[bi] = sum(g['p_nom'] for g in gens)
    
    # Demand matrix
    demand_mat = demand[bus_names].values.T  # shape (n_buses, n_hours)
    
    # Storage parameters
    storage_p_nom = np.zeros(n_storage)
    storage_e_nom = np.zeros(n_storage)
    storage_bus_idx = np.zeros(n_storage, dtype=int)
    storage_eta = np.sqrt(0.75)  # single-trip efficiency assuming 0.75 round-trip
    
    for i, s in enumerate(storage_units.to_dict('records')):
        storage_p_nom[i] = s['p_nom']
        storage_e_nom[i] = s['e_nom']
        storage_bus_idx[i] = bus_idx[s['bus']]
    
    # -----------------------------------------------------------------------
    # Decision Variables
    # -----------------------------------------------------------------------
    wind_dispatch = cp.Variable((n_buses, n_hours), nonneg=True)
    gas_dispatch = cp.Variable((n_buses, n_hours), nonneg=True)
    nuclear_dispatch = cp.Variable((n_buses, n_hours), nonneg=True)
    
    if storage_enabled and n_storage > 0:
        storage_charge = cp.Variable((n_storage, n_hours), nonneg=True)
        storage_discharge = cp.Variable((n_storage, n_hours), nonneg=True)
        soc = cp.Variable((n_storage, n_hours), nonneg=True)
    else:
        storage_charge = cp.Constant(np.zeros((n_storage, n_hours)))
        storage_discharge = cp.Constant(np.zeros((n_storage, n_hours)))
        soc = cp.Constant(np.zeros((n_storage, n_hours)))
    
    link_flow = cp.Variable((n_links, n_hours))
    unserved = cp.Variable((n_buses, n_hours), nonneg=True)
    
    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    cost_gas = gas_mc * cp.sum(gas_dispatch)
    cost_nuclear = nuclear_mc * cp.sum(nuclear_dispatch)
    cost_unserved = voLL * cp.sum(unserved)
    cost_curtailment = curtailment_cost * cp.sum(wind_potential - wind_dispatch)
    
    objective = cp.Minimize(cost_gas + cost_nuclear + cost_unserved + cost_curtailment)
    
    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------
    constraints = []
    
    # Generator limits
    constraints += [wind_dispatch <= wind_potential]
    constraints += [gas_dispatch <= gas_cap[:, None]]
    constraints += [nuclear_dispatch <= nuclear_cap[:, None]]
    
    # Link flow limits
    for li in range(n_links):
        constraints += [link_flow[li, :] <= link_list[li]['p_nom']]
        constraints += [link_flow[li, :] >= -link_list[li]['p_nom']]
    
    # Power balance at each bus and hour
    power_balance_constraints = []
    for bi in range(n_buses):
        for t in range(n_hours):
            net_flow = 0
            for li in range(n_links):
                if link_list[li]['idx0'] == bi:
                    net_flow -= link_flow[li, t]
                if link_list[li]['idx1'] == bi:
                    net_flow += link_flow[li, t]
            
            storage_injection = 0
            if storage_enabled and n_storage > 0:
                for si in range(n_storage):
                    if storage_bus_idx[si] == bi:
                        storage_injection += storage_discharge[si, t] - storage_charge[si, t]
            
            cstr = (wind_dispatch[bi, t] + gas_dispatch[bi, t] + nuclear_dispatch[bi, t]
                    + net_flow + storage_injection + unserved[bi, t] == demand_mat[bi, t])
            power_balance_constraints.append(cstr)
    constraints += power_balance_constraints
    
    # Storage dynamics
    if storage_enabled and n_storage > 0:
        for si in range(n_storage):
            # Power limits
            constraints += [storage_charge[si, :] <= storage_p_nom[si]]
            constraints += [storage_discharge[si, :] <= storage_p_nom[si]]
            # Energy limits
            constraints += [soc[si, :] <= storage_e_nom[si]]
            # Initial SOC (start at 50%)
            constraints += [soc[si, 0] == 0.5 * storage_e_nom[si]]
            # SOC dynamics
            for t in range(1, n_hours):
                constraints += [
                    soc[si, t] == soc[si, t-1] 
                    + storage_eta * storage_charge[si, t-1]
                    - storage_discharge[si, t-1] / storage_eta
                ]
            # End SOC = start SOC (periodic)
            constraints += [
                soc[si, 0] == soc[si, n_hours-1]
                + storage_eta * storage_charge[si, n_hours-1]
                - storage_discharge[si, n_hours-1] / storage_eta
            ]
    
    return {
        'objective': objective,
        'constraints': constraints,
        'power_balance_constraints': power_balance_constraints,
        'vars': {
            'wind_dispatch': wind_dispatch,
            'gas_dispatch': gas_dispatch,
            'nuclear_dispatch': nuclear_dispatch,
            'storage_charge': storage_charge,
            'storage_discharge': storage_discharge,
            'soc': soc,
            'link_flow': link_flow,
            'unserved': unserved,
        },
        'params': {
            'n_buses': n_buses,
            'n_hours': n_hours,
            'n_links': n_links,
            'n_storage': n_storage,
            'bus_names': bus_names,
            'bus_idx': bus_idx,
            'link_list': link_list,
            'wind_potential': wind_potential,
            'gas_cap': gas_cap,
            'nuclear_cap': nuclear_cap,
            'demand_mat': demand_mat,
            'storage_p_nom': storage_p_nom,
            'storage_e_nom': storage_e_nom,
            'storage_bus_idx': storage_bus_idx,
        }
    }


def solve_scenario(model_dict, scenario_name):
    """Solve the LP model and return results."""
    prob = cp.Problem(model_dict['objective'], model_dict['constraints'])
    prob.solve(solver=cp.HIGHS, verbose=False)
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"WARNING: Scenario {scenario_name} status = {prob.status}")
    
    v = model_dict['vars']
    p = model_dict['params']
    
    results = {
        'scenario': scenario_name,
        'status': prob.status,
        'total_cost': float(prob.value),
        'wind_generation': float(cp.sum(v['wind_dispatch']).value),
        'gas_generation': float(cp.sum(v['gas_dispatch']).value),
        'nuclear_generation': float(cp.sum(v['nuclear_dispatch']).value),
        'unserved_load': float(cp.sum(v['unserved']).value),
        'curtailment': float(np.sum(p['wind_potential'] - v['wind_dispatch'].value)),
    }
    
    # Hourly totals
    results['hourly'] = {
        'wind': np.sum(v['wind_dispatch'].value, axis=0),
        'gas': np.sum(v['gas_dispatch'].value, axis=0),
        'nuclear': np.sum(v['nuclear_dispatch'].value, axis=0),
        'unserved': np.sum(v['unserved'].value, axis=0),
        'demand': np.sum(p['demand_mat'], axis=0),
        'curtailment': np.sum(p['wind_potential'] - v['wind_dispatch'].value, axis=0),
    }
    
    if p['n_storage'] > 0 and hasattr(v['storage_charge'], 'value'):
        results['hourly']['storage_charge'] = np.sum(v['storage_charge'].value, axis=0)
        results['hourly']['storage_discharge'] = np.sum(v['storage_discharge'].value, axis=0)
        results['hourly']['soc'] = np.sum(v['soc'].value, axis=0)
        results['storage_charge_total'] = float(cp.sum(v['storage_charge']).value)
        results['storage_discharge_total'] = float(cp.sum(v['storage_discharge']).value)
    else:
        results['hourly']['storage_charge'] = np.zeros(p['n_hours'])
        results['hourly']['storage_discharge'] = np.zeros(p['n_hours'])
        results['hourly']['soc'] = np.zeros(p['n_hours'])
        results['storage_charge_total'] = 0.0
        results['storage_discharge_total'] = 0.0
    
    # Per-bus results
    results['bus'] = {
        'wind': v['wind_dispatch'].value,
        'gas': v['gas_dispatch'].value,
        'nuclear': v['nuclear_dispatch'].value,
        'unserved': v['unserved'].value,
        'demand': p['demand_mat'],
    }
    
    # Link flows
    results['link_flow'] = v['link_flow'].value
    
    # Marginal prices from power balance duals
    marginal_prices = np.zeros((p['n_buses'], p['n_hours']))
    pb_constraints = model_dict['power_balance_constraints']
    idx = 0
    for bi in range(p['n_buses']):
        for t in range(p['n_hours']):
            dual = pb_constraints[idx].dual_value
            if dual is not None:
                marginal_prices[bi, t] = float(dual)
            idx += 1
    results['marginal_prices'] = marginal_prices
    
    return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    buses, links, generators, storage, demand, wind_cf = load_data()
    
    scenarios = {
        'Base Case': {},
        'No Storage': {'storage_enabled': False},
        'Constrained Transmission': {'line_capacity_factor': 0.5},
        'Double Wind': {'wind_capacity_factor': 2.0},
        'No Wind': {'wind_enabled': False},
        'Nuclear Only (No Gas)': {'gas_enabled': False},
    }
    
    all_results = {}
    for name, kwargs in scenarios.items():
        print(f"Solving scenario: {name}...")
        model = build_dispatch_model(buses, links, generators, storage, demand, wind_cf, **kwargs)
        res = solve_scenario(model, name)
        all_results[name] = res
        print(f"  Status: {res['status']}, Cost: {res['total_cost']/1e6:.2f} M£, Unserved: {res['unserved_load']:.1f} MWh")
    
    # Save results
    import os
    os.makedirs('outputs', exist_ok=True)
    
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            'total_cost_GBP': res['total_cost'],
            'wind_generation_MWh': res['wind_generation'],
            'gas_generation_MWh': res['gas_generation'],
            'nuclear_generation_MWh': res['nuclear_generation'],
            'unserved_load_MWh': res['unserved_load'],
            'curtailment_MWh': res['curtailment'],
            'storage_charge_MWh': res.get('storage_charge_total', 0),
            'storage_discharge_MWh': res.get('storage_discharge_total', 0),
        }
    
    with open('outputs/scenario_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save hourly data for base case
    base = all_results['Base Case']
    hourly_df = pd.DataFrame({
        'hour': range(1, len(demand)+1),
        'demand_MW': base['hourly']['demand'],
        'wind_MW': base['hourly']['wind'],
        'gas_MW': base['hourly']['gas'],
        'nuclear_MW': base['hourly']['nuclear'],
        'unserved_MW': base['hourly']['unserved'],
        'curtailment_MW': base['hourly']['curtailment'],
        'storage_charge_MW': base['hourly']['storage_charge'],
        'storage_discharge_MW': base['hourly']['storage_discharge'],
        'storage_soc_MWh': base['hourly']['soc'],
    })
    hourly_df.to_csv('outputs/base_case_hourly.csv', index=False)
    
    print("\nAll scenarios solved and saved.")
