#!/usr/bin/env python3
"""
GB Power System Optimal Dispatch Model using PyPSA
===================================================
Builds a 20-bus GB power system model and runs optimal power flow
(transport model / LOPF) to determine optimal generation dispatch,
storage operation, and transmission flows under multiple scenarios.

The model uses a transport formulation (Links) for transmission,
which is appropriate for this dataset and avoids Kirchhoff voltage
law constraints that would over-constrain the network.

Load shedding generators (VOLL) are added at each bus to ensure
feasibility — this is standard practice in power system optimization
and represents the Value of Lost Load.
"""

import pypsa
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_001_20260416_184602'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOLL = 3000.0  # Value of Lost Load (£/MWh) - standard GB assumption

def load_data():
    """Load all CSV data files."""
    buses = pd.read_csv(os.path.join(DATA_DIR, 'buses.csv'))
    links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))
    generators = pd.read_csv(os.path.join(DATA_DIR, 'generators.csv'))
    demand = pd.read_csv(os.path.join(DATA_DIR, 'demand.csv'))
    wind_cf = pd.read_csv(os.path.join(DATA_DIR, 'wind_cf.csv'))
    storage = pd.read_csv(os.path.join(DATA_DIR, 'storage.csv'))
    return buses, links, generators, demand, wind_cf, storage

def build_network(buses_df, links_df, generators_df, demand_df, wind_cf_df, storage_df,
                  transmission_scale=1.0, wind_scale=1.0, storage_enabled=True,
                  add_load_shedding=True):
    """
    Build a PyPSA network from the data files.
    Uses transport model (Links) for transmission.
    """
    network = pypsa.Network()
    
    # Set snapshots (168 hours = 1 week)
    snapshots = pd.date_range('2050-01-01', periods=len(demand_df), freq='h')
    network.set_snapshots(snapshots)
    
    # Add buses
    for _, row in buses_df.iterrows():
        network.add("Bus", row['name'], v_nom=row['v_nom'],
                     x=row['x'], y=row['y'])
    
    # Add transmission links (transport model - bidirectional)
    for idx, row in links_df.iterrows():
        cap = row['p_nom'] * transmission_scale
        network.add("Link", f"Link_{row['bus0']}_{row['bus1']}",
                     bus0=row['bus0'], bus1=row['bus1'],
                     p_nom=cap,
                     p_min_pu=-1,  # bidirectional
                     length=row['length'],
                     marginal_cost=0.0,
                     efficiency=1.0)
    
    # Add generators
    for idx, row in generators_df.iterrows():
        gen_name = f"{row['bus']}_{row['carrier']}_{idx}"
        if row['carrier'] == 'onshore wind':
            bus_col = row['bus']
            cf_series = wind_cf_df[bus_col].values * wind_scale
            cf_series = np.clip(cf_series, 0, 1)
            network.add("Generator", gen_name,
                        bus=row['bus'],
                        carrier='onshore wind',
                        p_nom=row['p_nom'],
                        marginal_cost=row['marginal_cost'],
                        p_max_pu=cf_series)
        else:
            network.add("Generator", gen_name,
                        bus=row['bus'],
                        carrier=row['carrier'],
                        p_nom=row['p_nom'],
                        marginal_cost=row['marginal_cost'])
    
    # Add storage units
    if storage_enabled:
        for idx, row in storage_df.iterrows():
            network.add("StorageUnit", f"{row['bus']}_{row['carrier']}_{idx}",
                        bus=row['bus'],
                        carrier='PHS',
                        p_nom=row['p_nom'],
                        max_hours=row['e_nom'] / row['p_nom'],
                        efficiency_store=np.sqrt(row['efficiency']),
                        efficiency_dispatch=np.sqrt(row['efficiency']),
                        cyclic_state_of_charge=True,
                        marginal_cost=0.0)
    
    # Add load shedding generators (VOLL) at each bus
    if add_load_shedding:
        for bus_col in demand_df.columns:
            max_demand = demand_df[bus_col].max()
            network.add("Generator", f"VOLL_{bus_col}",
                        bus=bus_col,
                        carrier='load_shedding',
                        p_nom=max_demand * 1.1,  # slightly more than max demand
                        marginal_cost=VOLL)
    
    # Add loads
    for bus_col in demand_df.columns:
        network.add("Load", f"Load_{bus_col}",
                     bus=bus_col,
                     p_set=demand_df[bus_col].values)
    
    return network


def run_scenario(name, buses_df, links_df, generators_df, demand_df, wind_cf_df, storage_df, **kwargs):
    """Build and solve a scenario, return results dict."""
    print(f"\n{'='*60}")
    print(f"Running scenario: {name}")
    print(f"  params: {kwargs}")
    print(f"{'='*60}")
    
    network = build_network(buses_df, links_df, generators_df, demand_df, wind_cf_df, storage_df, **kwargs)
    
    # Solve LOPF
    status = network.optimize(solver_name='highs')
    print(f"Solver status: {status}")
    
    if 'infeasible' in str(status):
        print("WARNING: Model infeasible!")
        return network, None
    
    results = extract_results(network, name)
    return network, results


def extract_results(network, scenario_name):
    """Extract key results from a solved network."""
    gen_dispatch = network.generators_t.p.copy()
    
    # Generation by carrier
    gen_by_carrier = {}
    for carrier in network.generators.carrier.unique():
        mask = network.generators.carrier == carrier
        gen_names = network.generators.index[mask]
        cols = [c for c in gen_names if c in gen_dispatch.columns]
        if cols:
            gen_by_carrier[carrier] = gen_dispatch[cols].sum(axis=1)
    
    gen_carrier_df = pd.DataFrame(gen_by_carrier)
    total_gen_by_carrier = gen_carrier_df.sum()
    
    # Storage dispatch
    if len(network.storage_units) > 0:
        storage_dispatch = network.storage_units_t.p.sum(axis=1)
        storage_state = network.storage_units_t.state_of_charge.sum(axis=1)
    else:
        storage_dispatch = pd.Series(0, index=network.snapshots)
        storage_state = pd.Series(0, index=network.snapshots)
    
    # Total demand
    total_demand = network.loads_t.p_set.sum(axis=1)
    
    # System cost
    total_cost = network.objective
    
    # Wind curtailment
    wind_gens = network.generators[network.generators.carrier == 'onshore wind']
    wind_gen_names = [c for c in wind_gens.index if c in gen_dispatch.columns]
    if wind_gen_names:
        wind_available = (network.generators_t.p_max_pu[wind_gen_names] * wind_gens.loc[wind_gen_names, 'p_nom']).sum(axis=1)
        wind_dispatched = gen_carrier_df.get('onshore wind', pd.Series(0, index=network.snapshots))
        wind_curtailment = wind_available - wind_dispatched
        curtailment_rate = wind_curtailment.sum() / wind_available.sum() * 100 if wind_available.sum() > 0 else 0
    else:
        wind_available = pd.Series(0, index=network.snapshots)
        wind_dispatched = pd.Series(0, index=network.snapshots)
        wind_curtailment = pd.Series(0, index=network.snapshots)
        curtailment_rate = 0
    
    # Load shedding
    load_shed_total = gen_carrier_df.get('load_shedding', pd.Series(0, index=network.snapshots))
    load_shed_pct = load_shed_total.sum() / total_demand.sum() * 100 if total_demand.sum() > 0 else 0
    
    # Link flows
    link_flows = network.links_t.p0.copy()
    
    # Nodal prices (marginal prices at buses)
    bus_prices = network.buses_t.marginal_price.copy()
    
    # Cost breakdown
    # Generation cost by carrier
    cost_by_carrier = {}
    for carrier in network.generators.carrier.unique():
        mask = network.generators.carrier == carrier
        gen_names_c = network.generators.index[mask]
        cols = [c for c in gen_names_c if c in gen_dispatch.columns]
        if cols:
            mc = network.generators.loc[cols, 'marginal_cost']
            cost = (gen_dispatch[cols] * mc).sum().sum()
            cost_by_carrier[carrier] = cost
    
    results = {
        'scenario': scenario_name,
        'total_cost': total_cost,
        'gen_carrier_df': gen_carrier_df,
        'total_gen_by_carrier': total_gen_by_carrier,
        'cost_by_carrier': cost_by_carrier,
        'total_demand': total_demand,
        'storage_dispatch': storage_dispatch,
        'storage_state': storage_state,
        'wind_available': wind_available,
        'wind_dispatched': wind_dispatched,
        'wind_curtailment': wind_curtailment,
        'curtailment_rate': curtailment_rate,
        'load_shed_total': load_shed_total,
        'load_shed_pct': load_shed_pct,
        'link_flows': link_flows,
        'bus_prices': bus_prices,
        'network': network,
    }
    
    print(f"\nResults for {scenario_name}:")
    print(f"  Total system cost: £{total_cost:,.0f}")
    print(f"  Wind curtailment rate: {curtailment_rate:.2f}%")
    print(f"  Load shedding: {load_shed_pct:.2f}% of demand")
    print(f"  Total demand: {total_demand.sum():,.0f} MWh")
    print(f"  Generation mix (MWh):")
    for carrier, val in total_gen_by_carrier.items():
        share = val / total_demand.sum() * 100
        print(f"    {carrier}: {val:,.0f} MWh ({share:.1f}%)")
    print(f"  Cost breakdown:")
    for carrier, cost in cost_by_carrier.items():
        print(f"    {carrier}: £{cost:,.0f}")
    
    return results


def main():
    print("Loading data...")
    buses_df, links_df, generators_df, demand_df, wind_cf_df, storage_df = load_data()
    
    print(f"\nData summary:")
    print(f"  Buses: {len(buses_df)}")
    print(f"  Links: {len(links_df)}")
    print(f"  Generators: {len(generators_df)}")
    print(f"  Storage units: {len(storage_df)}")
    print(f"  Time steps: {len(demand_df)}")
    print(f"  Total peak demand: {demand_df.sum(axis=1).max():,.0f} MW")
    print(f"  Total min demand: {demand_df.sum(axis=1).min():,.0f} MW")
    
    # Define scenarios
    scenarios = {
        'Base Case': {'transmission_scale': 1.0, 'wind_scale': 1.0, 'storage_enabled': True},
        'High Wind': {'transmission_scale': 1.0, 'wind_scale': 1.5, 'storage_enabled': True},
        'Low Wind': {'transmission_scale': 1.0, 'wind_scale': 0.5, 'storage_enabled': True},
        'No Storage': {'transmission_scale': 1.0, 'wind_scale': 1.0, 'storage_enabled': False},
        'Constrained Transmission': {'transmission_scale': 0.5, 'wind_scale': 1.0, 'storage_enabled': True},
        'Enhanced Transmission': {'transmission_scale': 2.0, 'wind_scale': 1.0, 'storage_enabled': True},
    }
    
    all_results = {}
    all_networks = {}
    
    for name, params in scenarios.items():
        network, results = run_scenario(name, buses_df, links_df, generators_df, demand_df, wind_cf_df, storage_df, **params)
        if results is not None:
            all_results[name] = results
            all_networks[name] = network
    
    # Save summary results
    summary = []
    for name, res in all_results.items():
        total_demand_mwh = res['total_demand'].sum()
        summary.append({
            'Scenario': name,
            'Total Cost (£)': res['total_cost'],
            'Cost per MWh (£/MWh)': res['total_cost'] / total_demand_mwh,
            'Wind Generation (MWh)': res['total_gen_by_carrier'].get('onshore wind', 0),
            'Gas Generation (MWh)': res['total_gen_by_carrier'].get('gas', 0),
            'Nuclear Generation (MWh)': res['total_gen_by_carrier'].get('nuclear', 0),
            'Load Shedding (MWh)': res['total_gen_by_carrier'].get('load_shedding', 0),
            'Wind Share (%)': res['total_gen_by_carrier'].get('onshore wind', 0) / total_demand_mwh * 100,
            'Gas Share (%)': res['total_gen_by_carrier'].get('gas', 0) / total_demand_mwh * 100,
            'Nuclear Share (%)': res['total_gen_by_carrier'].get('nuclear', 0) / total_demand_mwh * 100,
            'Load Shedding (%)': res['load_shed_pct'],
            'Wind Curtailment (%)': res['curtailment_rate'],
            'Total Demand (MWh)': total_demand_mwh,
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), index=False)
    print("\n\nScenario Summary:")
    print(summary_df.to_string(index=False))
    
    # Save detailed results for base case
    if 'Base Case' in all_results:
        base = all_results['Base Case']
        base['gen_carrier_df'].to_csv(os.path.join(OUTPUT_DIR, 'base_case_generation.csv'))
        base['bus_prices'].to_csv(os.path.join(OUTPUT_DIR, 'base_case_bus_prices.csv'))
        base['link_flows'].to_csv(os.path.join(OUTPUT_DIR, 'base_case_link_flows.csv'))
        
        bn = all_networks['Base Case']
        if len(bn.storage_units) > 0:
            bn.storage_units_t.state_of_charge.to_csv(os.path.join(OUTPUT_DIR, 'base_case_storage_soc.csv'))
            bn.storage_units_t.p.to_csv(os.path.join(OUTPUT_DIR, 'base_case_storage_dispatch.csv'))
    
    # Save all results as JSON
    json_summary = {}
    for name, res in all_results.items():
        json_summary[name] = {
            'total_cost': float(res['total_cost']),
            'curtailment_rate': float(res['curtailment_rate']),
            'load_shed_pct': float(res['load_shed_pct']),
            'total_demand_mwh': float(res['total_demand'].sum()),
            'generation_mwh': {k: float(v) for k, v in res['total_gen_by_carrier'].items()},
            'cost_by_carrier': {k: float(v) for k, v in res['cost_by_carrier'].items()},
        }
    
    with open(os.path.join(OUTPUT_DIR, 'all_scenarios_summary.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print("\n\nAll results saved to outputs/")
    return all_results, all_networks


if __name__ == '__main__':
    all_results, all_networks = main()
