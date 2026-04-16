"""
Sensitivity Analysis: Wind Capacity Multiplier Sweep
=====================================================
Vary wind capacity from 1x to 5x to understand how increasing
renewable penetration affects curtailment, costs, and load shedding.
"""

import pandas as pd
import numpy as np
import pypsa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'
OUTPUT_DIR = '../outputs'
IMAGE_DIR = '../report/images'

buses_df = pd.read_csv(f'{DATA_DIR}/buses.csv', index_col='name')
generators_df = pd.read_csv(f'{DATA_DIR}/generators.csv')
links_df = pd.read_csv(f'{DATA_DIR}/links.csv')
storage_df = pd.read_csv(f'{DATA_DIR}/storage.csv')
demand_df = pd.read_csv(f'{DATA_DIR}/demand.csv')
wind_cf_df = pd.read_csv(f'{DATA_DIR}/wind_cf.csv')

def build_network(wind_multiplier=1.0, line_multiplier=1.0, include_storage=True):
    n = pypsa.Network()
    n.set_snapshots(range(168))

    for carrier in ['AC', 'onshore wind', 'gas', 'nuclear', 'PHS', 'load shedding']:
        n.add("Carrier", carrier)

    for bus_name, bus_row in buses_df.iterrows():
        n.add("Bus", bus_name, v_nom=bus_row['v_nom'], carrier='AC',
               x=bus_row['x'], y=bus_row['y'])

    for bus_name in demand_df.columns:
        n.add("Load", f"{bus_name}_load", bus=bus_name,
               p_set=demand_df[bus_name].values)

    for idx, gen in generators_df.iterrows():
        gen_name = f"{gen['bus']}_{gen['carrier']}_{idx}"
        p_nom = gen['p_nom']
        if gen['carrier'] == 'onshore wind':
            p_nom *= wind_multiplier
        n.add("Generator", gen_name, bus=gen['bus'], carrier=gen['carrier'],
               p_nom=p_nom, marginal_cost=gen['marginal_cost'])
        if gen['carrier'] == 'onshore wind' and gen['bus'] in wind_cf_df.columns:
            n.generators_t.p_max_pu[gen_name] = wind_cf_df[gen['bus']].values

    for bus_name in buses_df.index:
        n.add("Generator", f"{bus_name}_load_shedding", bus=bus_name,
               carrier='load shedding', p_nom=50000, marginal_cost=6000)

    for idx, link in links_df.iterrows():
        n.add("Link", f"{link['bus0']}_{link['bus1']}",
               bus0=link['bus0'], bus1=link['bus1'],
               p_nom=link['p_nom'] * line_multiplier,
               length=link['length'], carrier=link['carrier'], efficiency=1.0)

    if include_storage:
        for idx, stor in storage_df.iterrows():
            n.add("StorageUnit", f"{stor['bus']}_{stor['carrier']}_{idx}",
                   bus=stor['bus'], carrier='PHS', p_nom=stor['p_nom'],
                   max_hours=stor['e_nom'] / stor['p_nom'],
                   efficiency_store=stor['efficiency'],
                   efficiency_dispatch=stor['efficiency'],
                   cyclic_state_of_charge=True)
    return n

# Sweep wind multiplier
multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
sweep_results = []

for mult in multipliers:
    print(f"Running wind multiplier = {mult}x...")
    n = build_network(wind_multiplier=mult)
    status, condition = n.optimize(solver_name='highs')

    wind_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'onshore wind']
    gas_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'gas']
    nuclear_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'nuclear']
    shed_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'load shedding']

    total_wind = n.generators_t.p[wind_gens].sum().sum()
    total_gas = n.generators_t.p[gas_gens].sum().sum()
    total_nuclear = n.generators_t.p[nuclear_gens].sum().sum()
    total_shed = n.generators_t.p[shed_gens].sum().sum()
    total_demand = n.loads_t.p.sum().sum().sum()

    wind_available = pd.Series(0, index=n.snapshots)
    for g in wind_gens:
        if g in n.generators_t.p_max_pu.columns:
            wind_available += n.generators.at[g, 'p_nom'] * n.generators_t.p_max_pu[g]
        else:
            wind_available += n.generators.at[g, 'p_nom']
    wind_dispatched = n.generators_t.p[wind_gens].sum(axis=1)
    curtailment = (wind_available - wind_dispatched).clip(lower=0).sum()

    sweep_results.append({
        'wind_multiplier': mult,
        'total_cost_GBP_M': n.objective / 1e6,
        'wind_share_pct': total_wind / total_demand * 100,
        'gas_share_pct': total_gas / total_demand * 100,
        'nuclear_share_pct': total_nuclear / total_demand * 100,
        'shedding_share_pct': total_shed / total_demand * 100,
        'curtailment_rate_pct': curtailment / (total_wind + curtailment) * 100 if (total_wind + curtailment) > 0 else 0,
        'total_wind_GWh': total_wind / 1000,
        'total_curtailment_GWh': curtailment / 1000,
        'total_shedding_GWh': total_shed / 1000,
    })

sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(f'{OUTPUT_DIR}/wind_sensitivity.csv', index=False)

# --- Figure 13: Wind Sensitivity Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(sweep_df['wind_multiplier'], sweep_df['wind_share_pct'], 'g-o', label='Wind')
axes[0,0].plot(sweep_df['wind_multiplier'], sweep_df['gas_share_pct'], '-', color='gray', marker='o', label='Gas')
axes[0,0].plot(sweep_df['wind_multiplier'], sweep_df['nuclear_share_pct'], '-', color='purple', marker='o', label='Nuclear')
axes[0,0].plot(sweep_df['wind_multiplier'], sweep_df['shedding_share_pct'], '-', color='red', marker='o', label='Load Shedding')
axes[0,0].set_xlabel('Wind Capacity Multiplier')
axes[0,0].set_ylabel('Share of Demand (%)')
axes[0,0].set_title('Generation Mix vs Wind Capacity')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(sweep_df['wind_multiplier'], sweep_df['curtailment_rate_pct'], '-', color='red', marker='o')
axes[0,1].set_xlabel('Wind Capacity Multiplier')
axes[0,1].set_ylabel('Curtailment Rate (%)')
axes[0,1].set_title('Wind Curtailment Rate vs Wind Capacity')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(sweep_df['wind_multiplier'], sweep_df['total_cost_GBP_M'], '-', color='blue', marker='o')
axes[1,0].set_xlabel('Wind Capacity Multiplier')
axes[1,0].set_ylabel('Total System Cost (GBP M)')
axes[1,0].set_title('System Cost vs Wind Capacity')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(sweep_df['wind_multiplier'], sweep_df['total_wind_GWh'], '-', color='green', marker='o', label='Wind Dispatched')
axes[1,1].plot(sweep_df['wind_multiplier'], sweep_df['total_curtailment_GWh'], '-', color='red', marker='o', label='Curtailed')
axes[1,1].plot(sweep_df['wind_multiplier'], sweep_df['total_shedding_GWh'], '-', color='black', marker='o', label='Load Shedding')
axes[1,1].set_xlabel('Wind Capacity Multiplier')
axes[1,1].set_ylabel('Energy (GWh)')
axes[1,1].set_title('Energy Balance vs Wind Capacity')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Sensitivity Analysis: Wind Capacity Multiplier Sweep', fontsize=14)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/wind_sensitivity.png', bbox_inches='tight')
plt.close()

print("Sensitivity analysis complete. Figure saved.")
print(sweep_df.to_string())
