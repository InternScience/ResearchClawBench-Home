"""
GB 20-Bus Power System: Optimal Dispatch Analysis
===================================================
Uses PyPSA to model the GB power system with 20 buses, 23 transmission lines,
onshore wind, gas, nuclear generators, and pumped hydro storage.

Scenarios:
1. Base case - current capacity mix
2. High wind - 3x wind capacity to study renewable integration
3. Constrained transmission - 50% line capacity reduction
"""

import pandas as pd
import numpy as np
import pypsa
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_001_20260518_002218'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 1. Load Data
# ============================
print("Loading data...")
buses = pd.read_csv(os.path.join(DATA_DIR, 'buses.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))
generators = pd.read_csv(os.path.join(DATA_DIR, 'generators.csv'))
storage_df = pd.read_csv(os.path.join(DATA_DIR, 'storage.csv'))
demand = pd.read_csv(os.path.join(DATA_DIR, 'demand.csv'))
wind_cf = pd.read_csv(os.path.join(DATA_DIR, 'wind_cf.csv'))

n_snapshots = len(demand)
print(f"  Buses: {len(buses)}, Links: {len(links)}, Generators: {len(generators)}")
print(f"  Snapshots: {n_snapshots}")

# System totals
total_wind_nom = generators.loc[generators['carrier'] == 'onshore wind', 'p_nom'].sum()
total_gas_nom = generators.loc[generators['carrier'] == 'gas', 'p_nom'].sum()
total_nuclear_nom = generators.loc[generators['carrier'] == 'nuclear', 'p_nom'].sum()
total_demand_peak = demand.sum(axis=1).max()
total_demand_mean = demand.sum(axis=1).mean()

print(f"\nSystem summary:")
print(f"  Wind: {total_wind_nom/1000:.1f} GW, Gas: {total_gas_nom/1000:.1f} GW, Nuclear: {total_nuclear_nom/1000:.1f} GW")
print(f"  Storage: {storage_df['p_nom'].sum():.0f} MW / {storage_df['e_nom'].sum():.0f} MWh")
print(f"  Peak demand: {total_demand_peak/1000:.1f} GW, Mean: {total_demand_mean/1000:.1f} GW")

# ============================
# 2. Build Network Function
# ============================
VOLL = 6000  # Value of Lost Load (£/MWh) - standard for GB (National Grid)

def build_network(wind_scaling=1.0, line_scaling=1.0):
    """Build a PyPSA network with the given scenario parameters."""
    n = pypsa.Network()
    n.set_snapshots(range(n_snapshots))
    
    # Add carriers
    for c in ['wind', 'gas', 'nuclear', 'PHS', 'load_shedding']:
        n.add("Carrier", c)
    
    # Add buses
    for _, bus in buses.iterrows():
        n.add("Bus", bus['name'], v_nom=bus['v_nom'], x=bus['x'], y=bus['y'])
    
    # Add links (transmission lines)
    for _, link in links.iterrows():
        n.add("Link", f"{link['bus0']}_{link['bus1']}",
              bus0=link['bus0'], bus1=link['bus1'],
              p_nom=link['p_nom'] * line_scaling,
              length=link['length'],
              p_min_pu=-1.0)
    
    # Add loads
    for bus_name in demand.columns:
        n.add("Load", f"load_{bus_name}", bus=bus_name,
              p_set=demand[bus_name].values)
    
    # Add generators
    for _, gen in generators.iterrows():
        carrier = gen['carrier']
        p_nom_scaled = gen['p_nom'] * (wind_scaling if carrier == 'onshore wind' else 1.0)
        
        if carrier == 'onshore wind':
            p_max_pu = wind_cf[gen['bus']].values
        else:
            p_max_pu = 1.0
        
        n.add("Generator", f"{carrier}_{gen['bus']}",
              bus=gen['bus'],
              carrier=carrier,
              p_nom=p_nom_scaled,
              marginal_cost=gen['marginal_cost'],
              p_max_pu=p_max_pu)
    
    # Add storage units
    for _, stor in storage_df.iterrows():
        n.add("StorageUnit", f"PHS_{stor['bus']}",
              bus=stor['bus'],
              carrier="PHS",
              p_nom=stor['p_nom'],
              max_hours=stor['e_nom'] / stor['p_nom'],
              efficiency_store=stor['efficiency'],
              efficiency_dispatch=stor['efficiency'],
              standing_loss=0.0,
              cyclic_state_of_charge=True)
    
    # Add load shedding generators at every bus
    for _, bus in buses.iterrows():
        n.add("Generator", f"LS_{bus['name']}",
              bus=bus['name'],
              carrier="load_shedding",
              p_nom=1e6,
              marginal_cost=VOLL,
              p_max_pu=1.0)
    
    return n


def run_optimization(n, scenario_name):
    """Run linear optimal power flow."""
    print(f"\nRunning: {scenario_name}")
    status = n.optimize(solver_name='cbc')
    print(f"  Status: {status[0]}, Termination: {status[1]}")
    
    if status[1] in ['infeasible']:
        print(f"  WARNING: Optimization failed!")
        return n, False
    
    print(f"  Objective: {n.objective:,.0f}")
    return n, True


def extract_results(n, scenario_name):
    """Extract and save results from optimization."""
    gen_dispatch = n.generators_t.p.copy()
    load_data = n.loads_t.p_set.copy()
    
    has_storage = len(n.storage_units) > 0
    if has_storage:
        storage_p = n.storage_units_t.p.copy()
        storage_soc = n.storage_units_t.state_of_charge.copy()
    
    link_flows = n.links_t.p0.copy()
    
    # Separate generation by type
    wind_gens = [c for c in gen_dispatch.columns if 'onshore wind' in c]
    gas_gens = [c for c in gen_dispatch.columns if 'gas_' in c]
    nuclear_gens = [c for c in gen_dispatch.columns if 'nuclear' in c]
    ls_gens = [c for c in gen_dispatch.columns if 'LS_' in c]
    
    total_wind = gen_dispatch[wind_gens].sum(axis=1) if wind_gens else pd.Series(0, index=gen_dispatch.index)
    total_gas = gen_dispatch[gas_gens].sum(axis=1) if gas_gens else pd.Series(0, index=gen_dispatch.index)
    total_nuclear = gen_dispatch[nuclear_gens].sum(axis=1) if nuclear_gens else pd.Series(0, index=gen_dispatch.index)
    total_ls = gen_dispatch[ls_gens].sum(axis=1) if ls_gens else pd.Series(0, index=gen_dispatch.index)
    total_demand = load_data.sum(axis=1)
    
    # Wind curtailment
    wind_available = np.zeros(n_snapshots)
    for gen_name in wind_gens:
        gen = n.generators.loc[gen_name]
        wind_available += gen['p_max_pu'] * gen['p_nom']
    wind_dispatched = total_wind.values
    wind_curtailment = np.maximum(wind_available - wind_dispatched, 0)
    total_curtailment = wind_curtailment.sum()
    curtailment_pct = (total_curtailment / wind_available.sum() * 100) if wind_available.sum() > 0 else 0
    
    # Storage summary
    if has_storage:
        storage_charge = float(storage_p.clip(lower=0).sum().sum())
        storage_discharge = float(storage_p.clip(upper=0).sum().abs().sum())
        storage_e_total = float(storage_df['e_nom'].sum())
        storage_cycles = storage_discharge / storage_e_total if storage_e_total > 0 else 0
    else:
        storage_charge = storage_discharge = storage_cycles = 0
    
    # Per-generator total dispatch by type
    gen_totals = {}
    for gen_name in gen_dispatch.columns:
        carrier = n.generators.loc[gen_name, 'carrier']
        if carrier not in gen_totals:
            gen_totals[carrier] = 0.0
        gen_totals[carrier] += float(gen_dispatch[gen_name].sum())
    
    # Cost breakdown
    cost_breakdown = {}
    for gen_name in gen_dispatch.columns:
        carrier = n.generators.loc[gen_name, 'carrier']
        mc = n.generators.loc[gen_name, 'marginal_cost']
        cost = float((gen_dispatch[gen_name] * mc).sum())
        if carrier not in cost_breakdown:
            cost_breakdown[carrier] = 0.0
        cost_breakdown[carrier] += cost
    
    # Link utilization
    link_util = {}
    for link_name in link_flows.columns:
        link = n.links.loc[link_name]
        flow = link_flows[link_name].abs()
        max_cap = link['p_nom']
        link_util[link_name] = {
            'avg_util_pct': float(flow.mean() / max_cap * 100) if max_cap > 0 else 0,
            'peak_util_pct': float(flow.max() / max_cap * 100) if max_cap > 0 else 0,
            'avg_flow_mw': float(flow.mean()),
            'peak_flow_mw': float(flow.max()),
        }
    
    # Per-bus generation summary
    bus_gen_summary = {}
    for _, bus in buses.iterrows():
        bname = bus['name']
        bus_gens = [c for c in gen_dispatch.columns if c.endswith(f"_{bname}")]
        load_col = f"load_{bname}"
        bus_gen_summary[bname] = {
            'total_gen_mwh': float(gen_dispatch[bus_gens].sum().sum()) if bus_gens else 0.0,
            'total_load_mwh': float(load_data[load_col].sum()) if load_col in load_data.columns else 0.0,
        }
    
    summary = {
        'scenario': scenario_name,
        'total_objective': float(n.objective),
        'total_generation_mwh': float(gen_dispatch.sum().sum()),
        'total_wind_mwh': float(total_wind.sum()),
        'total_gas_mwh': float(total_gas.sum()),
        'total_nuclear_mwh': float(total_nuclear.sum()),
        'total_load_shedding_mwh': float(total_ls.sum()),
        'total_demand_mwh': float(total_demand.sum()),
        'wind_available_mwh': float(wind_available.sum()),
        'wind_curtailment_mwh': float(total_curtailment),
        'curtailment_pct': float(curtailment_pct),
        'storage_charge_mwh': storage_charge,
        'storage_discharge_mwh': storage_discharge,
        'storage_cycles': float(storage_cycles),
        'wind_penetration_pct': float(total_wind.sum() / (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) * 100) if (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) > 0 else 0,
        'gas_fraction_pct': float(total_gas.sum() / (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) * 100) if (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) > 0 else 0,
        'nuclear_fraction_pct': float(total_nuclear.sum() / (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) * 100) if (gen_dispatch[wind_gens + gas_gens + nuclear_gens].sum().sum()) > 0 else 0,
        'cost_breakdown': cost_breakdown,
        'gen_totals_mwh': gen_totals,
        'link_utilization': link_util,
        'load_shed_pct': float(total_ls.sum() / total_demand.sum() * 100) if total_demand.sum() > 0 else 0,
    }
    
    # Save time series
    ts_data = pd.DataFrame({
        'total_wind_mw': total_wind.values,
        'total_gas_mw': total_gas.values,
        'total_nuclear_mw': total_nuclear.values,
        'total_demand_mw': total_demand.values,
        'total_load_shed_mw': total_ls.values,
        'wind_available_mw': wind_available,
        'wind_curtailment_mw': wind_curtailment,
    })
    ts_data.to_csv(os.path.join(OUTPUT_DIR, f'{scenario_name}_timeseries.csv'), index=False)
    gen_dispatch.to_csv(os.path.join(OUTPUT_DIR, f'{scenario_name}_gen_dispatch.csv'))
    link_flows.to_csv(os.path.join(OUTPUT_DIR, f'{scenario_name}_link_flows.csv'))
    
    if has_storage:
        storage_p.to_csv(os.path.join(OUTPUT_DIR, f'{scenario_name}_storage_dispatch.csv'))
        storage_soc.to_csv(os.path.join(OUTPUT_DIR, f'{scenario_name}_storage_soc.csv'))
    
    return summary, {
        'gen_dispatch': gen_dispatch,
        'total_wind': total_wind,
        'total_gas': total_gas,
        'total_nuclear': total_nuclear,
        'total_ls': total_ls,
        'total_demand': total_demand,
        'wind_available': wind_available,
        'wind_curtailment': wind_curtailment,
        'link_flows': link_flows,
        'storage_p': storage_p if has_storage else None,
        'storage_soc': storage_soc if has_storage else None,
        'bus_gen_summary': bus_gen_summary,
    }


# ============================
# 3. Run All Scenarios
# ============================
scenarios = {
    'base': {'wind_scaling': 1.0, 'line_scaling': 1.0},
    'high_wind': {'wind_scaling': 3.0, 'line_scaling': 1.0},
    'constrained_tx': {'wind_scaling': 1.0, 'line_scaling': 0.5},
}

all_summaries = {}
all_results = {}

for scenario_name, params in scenarios.items():
    n = build_network(wind_scaling=params['wind_scaling'], 
                      line_scaling=params['line_scaling'])
    n, success = run_optimization(n, scenario_name)
    
    if success:
        summary, results = extract_results(n, scenario_name)
        all_summaries[scenario_name] = summary
        all_results[scenario_name] = results
        
        print(f"\n  Summary for {scenario_name}:")
        print(f"    Total cost: {summary['total_objective']:,.0f}")
        print(f"    Wind: {summary['total_wind_mwh']:,.0f} MWh ({summary['wind_penetration_pct']:.1f}%)")
        print(f"    Gas: {summary['total_gas_mwh']:,.0f} MWh ({summary['gas_fraction_pct']:.1f}%)")
        print(f"    Nuclear: {summary['total_nuclear_mwh']:,.0f} MWh ({summary['nuclear_fraction_pct']:.1f}%)")
        print(f"    Curtailment: {summary['wind_curtailment_mwh']:,.0f} MWh ({summary['curtailment_pct']:.1f}%)")
        print(f"    Load shedding: {summary['total_load_shedding_mwh']:,.0f} MWh ({summary['load_shed_pct']:.1f}%)")
        print(f"    Storage cycles: {summary['storage_cycles']:.2f}")
    else:
        print(f"  FAILED for {scenario_name}")

# Save all summaries
with open(os.path.join(OUTPUT_DIR, 'system_costs.json'), 'w') as f:
    json.dump(all_summaries, f, indent=2, default=str)

print("\n\n=== All scenarios complete ===")
for name, s in all_summaries.items():
    print(f"\n{name}:")
    print(f"  Cost: {s['total_objective']:,.0f}")
    print(f"  Wind: {s['total_wind_mwh']:,.0f} MWh ({s['wind_penetration_pct']:.1f}%)")
    print(f"  Gas: {s['total_gas_mwh']:,.0f} MWh ({s['gas_fraction_pct']:.1f}%)")
    print(f"  Curtailment: {s['curtailment_pct']:.1f}%")
    print(f"  Load shedding: {s['load_shed_pct']:.1f}%")
