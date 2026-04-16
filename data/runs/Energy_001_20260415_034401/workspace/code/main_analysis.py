"""
GB Power System Optimal Dispatch Model
=======================================
A high-resolution (20-node, hourly) linear optimal power flow model
for the Great Britain power system using PyPSA.

Scenarios:
1. Baseline (current mix)
2. High Renewable (3x wind capacity)
3. Constrained Network (50% line capacities)
4. No Storage (storage units removed)
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

# Paths
DATA_DIR = '../data'
OUTPUT_DIR = '../outputs'
IMAGE_DIR = '../report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
buses_df = pd.read_csv(f'{DATA_DIR}/buses.csv', index_col='name')
generators_df = pd.read_csv(f'{DATA_DIR}/generators.csv')
links_df = pd.read_csv(f'{DATA_DIR}/links.csv')
storage_df = pd.read_csv(f'{DATA_DIR}/storage.csv')
demand_df = pd.read_csv(f'{DATA_DIR}/demand.csv')
wind_cf_df = pd.read_csv(f'{DATA_DIR}/wind_cf.csv')

# Quick data summary
total_demand_peak = demand_df.sum(axis=1).max()
total_demand_avg = demand_df.sum(axis=1).mean()
total_wind_cap = generators_df[generators_df['carrier']=='onshore wind']['p_nom'].sum()
total_gas_cap = generators_df[generators_df['carrier']=='gas']['p_nom'].sum()
total_nuclear_cap = generators_df[generators_df['carrier']=='nuclear']['p_nom'].sum()

print(f"System Summary:")
print(f"  Peak demand: {total_demand_peak:.0f} MW")
print(f"  Avg demand: {total_demand_avg:.0f} MW")
print(f"  Wind capacity: {total_wind_cap:.0f} MW")
print(f"  Gas capacity: {total_gas_cap:.0f} MW")
print(f"  Nuclear capacity: {total_nuclear_cap:.0f} MW")
print(f"  Total conventional+wind nameplate: {total_wind_cap+total_gas_cap+total_nuclear_cap:.0f} MW")

# ============================================================
# 2. Build PyPSA Network
# ============================================================
def build_network(scenario='baseline', wind_multiplier=1.0, line_multiplier=1.0, include_storage=True):
    """Build a PyPSA network from the data files."""
    n = pypsa.Network()
    n.set_snapshots(range(168))

    # Add carriers
    for carrier in ['AC', 'onshore wind', 'gas', 'nuclear', 'PHS', 'load shedding']:
        n.add("Carrier", carrier)

    # Add buses
    for bus_name, bus_row in buses_df.iterrows():
        n.add("Bus", bus_name, v_nom=bus_row['v_nom'], carrier='AC',
               x=bus_row['x'], y=bus_row['y'])

    # Add loads (demand)
    for bus_name in demand_df.columns:
        n.add("Load", f"{bus_name}_load", bus=bus_name,
               p_set=demand_df[bus_name].values)

    # Add generators
    for idx, gen in generators_df.iterrows():
        gen_name = f"{gen['bus']}_{gen['carrier']}_{idx}"
        p_nom = gen['p_nom']

        # Apply scenario modifications
        if gen['carrier'] == 'onshore wind':
            p_nom *= wind_multiplier

        n.add("Generator", gen_name,
               bus=gen['bus'],
               carrier=gen['carrier'],
               p_nom=p_nom,
               marginal_cost=gen['marginal_cost'])

        # Set wind capacity factor as p_max_pu time series
        if gen['carrier'] == 'onshore wind' and gen['bus'] in wind_cf_df.columns:
            n.generators_t.p_max_pu[gen_name] = wind_cf_df[gen['bus']].values

    # Add load shedding at each bus (very expensive backup)
    # Value of Lost Load = 6000 GBP/MWh (standard GB figure from Zeyringer et al.)
    for bus_name in buses_df.index:
        shed_name = f"{bus_name}_load_shedding"
        n.add("Generator", shed_name,
               bus=bus_name,
               carrier='load shedding',
               p_nom=50000,  # Large enough to cover any deficit
               marginal_cost=6000,  # VoLL
               p_nom_extendable=False)

    # Add links (transmission lines) - bidirectional
    for idx, link in links_df.iterrows():
        link_name = f"{link['bus0']}_{link['bus1']}"
        p_nom = link['p_nom'] * line_multiplier
        n.add("Link", link_name,
               bus0=link['bus0'],
               bus1=link['bus1'],
               p_nom=p_nom,
               length=link['length'],
               carrier=link['carrier'],
               efficiency=1.0)  # Lossless for simplicity

    # Add storage units
    if include_storage:
        for idx, stor in storage_df.iterrows():
            stor_name = f"{stor['bus']}_{stor['carrier']}_{idx}"
            n.add("StorageUnit", stor_name,
                   bus=stor['bus'],
                   carrier='PHS',
                   p_nom=stor['p_nom'],
                   max_hours=stor['e_nom'] / stor['p_nom'],
                   efficiency_store=stor['efficiency'],
                   efficiency_dispatch=stor['efficiency'],
                   cyclic_state_of_charge=True)

    return n

# ============================================================
# 3. Run scenarios
# ============================================================
scenarios = {
    'Baseline': {'wind_multiplier': 1.0, 'line_multiplier': 1.0, 'include_storage': True},
    'High Renewable': {'wind_multiplier': 3.0, 'line_multiplier': 1.0, 'include_storage': True},
    'Constrained Network': {'wind_multiplier': 1.0, 'line_multiplier': 0.5, 'include_storage': True},
    'No Storage': {'wind_multiplier': 1.0, 'line_multiplier': 1.0, 'include_storage': False},
}

results = {}

for scenario_name, params in scenarios.items():
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"{'='*60}")

    n = build_network(
        wind_multiplier=params['wind_multiplier'],
        line_multiplier=params['line_multiplier'],
        include_storage=params['include_storage']
    )

    status, condition = n.optimize(solver_name='highs')
    print(f"  Optimization status: {status}, {condition}")

    # Extract results
    total_cost = n.objective if n.objective is not None and not np.isnan(n.objective) else 0
    generators_p = n.generators_t.p

    # Storage
    if params['include_storage'] and len(n.storage_units) > 0:
        storage_p = n.storage_units_t.p
    else:
        storage_p = pd.DataFrame(0, index=n.snapshots, columns=['dummy'])

    loads_p = n.loads_t.p

    # Classify generators
    wind_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'onshore wind']
    gas_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'gas']
    nuclear_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'nuclear']
    shed_gens = [g for g in n.generators.index if n.generators.at[g, 'carrier'] == 'load shedding']

    # Calculate curtailment
    wind_available = pd.Series(0, index=n.snapshots)
    wind_dispatched = pd.Series(0, index=n.snapshots)
    for g in wind_gens:
        if g in n.generators_t.p_max_pu.columns:
            wind_available += n.generators.at[g, 'p_nom'] * n.generators_t.p_max_pu[g]
        else:
            wind_available += n.generators.at[g, 'p_nom']
        wind_dispatched += generators_p[g]

    curtailment = (wind_available - wind_dispatched).clip(lower=0)

    # Total generation by carrier
    total_wind = generators_p[wind_gens].sum().sum() if wind_gens else 0
    total_gas = generators_p[gas_gens].sum().sum() if gas_gens else 0
    total_nuclear = generators_p[nuclear_gens].sum().sum() if nuclear_gens else 0
    total_shedding = generators_p[shed_gens].sum().sum() if shed_gens else 0
    total_demand = loads_p.sum().sum().sum()
    total_curtailment = curtailment.sum()
    total_storage_discharge = storage_p.clip(lower=0).sum().sum() if params['include_storage'] else 0
    total_storage_charge = storage_p.clip(upper=0).sum().sum() if params['include_storage'] else 0

    # Link flows
    link_flows = n.links_t.p0 if hasattr(n.links_t, 'p0') and len(n.links_t.p0.columns) > 0 else pd.DataFrame()

    results[scenario_name] = {
        'network': n,
        'total_cost': total_cost,
        'total_wind_MWh': total_wind,
        'total_gas_MWh': total_gas,
        'total_nuclear_MWh': total_nuclear,
        'total_shedding_MWh': total_shedding,
        'total_demand_MWh': total_demand,
        'total_curtailment_MWh': total_curtailment,
        'total_storage_discharge_MWh': total_storage_discharge,
        'total_storage_charge_MWh': total_storage_charge,
        'generators_p': generators_p,
        'storage_p': storage_p,
        'curtailment': curtailment,
        'wind_available': wind_available,
        'wind_dispatched': wind_dispatched,
        'link_flows': link_flows,
        'wind_gens': wind_gens,
        'gas_gens': gas_gens,
        'nuclear_gens': nuclear_gens,
        'shed_gens': shed_gens,
    }

    print(f"  Total cost: GBP {total_cost:,.0f}")
    print(f"  Total demand: {total_demand:,.0f} MWh")
    print(f"  Wind generation: {total_wind:,.0f} MWh")
    print(f"  Gas generation: {total_gas:,.0f} MWh")
    print(f"  Nuclear generation: {total_nuclear:,.0f} MWh")
    print(f"  Load shedding: {total_shedding:,.0f} MWh")
    print(f"  Curtailment: {total_curtailment:,.0f} MWh")
    print(f"  Storage discharge: {total_storage_discharge:,.0f} MWh")

# ============================================================
# 4. Save summary results
# ============================================================
summary = {}
for name, r in results.items():
    wind_avail_total = r['total_wind_MWh'] + r['total_curtailment_MWh']
    summary[name] = {
        'total_cost_GBP': float(r['total_cost']),
        'total_demand_MWh': float(r['total_demand_MWh']),
        'total_wind_MWh': float(r['total_wind_MWh']),
        'total_gas_MWh': float(r['total_gas_MWh']),
        'total_nuclear_MWh': float(r['total_nuclear_MWh']),
        'total_shedding_MWh': float(r['total_shedding_MWh']),
        'total_curtailment_MWh': float(r['total_curtailment_MWh']),
        'total_storage_discharge_MWh': float(r['total_storage_discharge_MWh']),
        'wind_share_pct': float(r['total_wind_MWh'] / r['total_demand_MWh'] * 100),
        'gas_share_pct': float(r['total_gas_MWh'] / r['total_demand_MWh'] * 100),
        'nuclear_share_pct': float(r['total_nuclear_MWh'] / r['total_demand_MWh'] * 100),
        'curtailment_rate_pct': float(r['total_curtailment_MWh'] / wind_avail_total * 100) if wind_avail_total > 0 else 0,
        'shedding_rate_pct': float(r['total_shedding_MWh'] / r['total_demand_MWh'] * 100),
    }

with open(f'{OUTPUT_DIR}/scenario_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to outputs/scenario_summary.json")

# ============================================================
# 5. Generate Figures
# ============================================================
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

scenario_names = list(results.keys())
total_demand_ts = demand_df.sum(axis=1)

def plot_dispatch_stack(ax, scenario_name, results_dict):
    r = results_dict[scenario_name]
    gens_p = r['generators_p']

    wind_total = pd.Series(0, index=range(168))
    gas_total = pd.Series(0, index=range(168))
    nuclear_total = pd.Series(0, index=range(168))
    shed_total = pd.Series(0, index=range(168))

    for g in r['wind_gens']:
        if g in gens_p.columns:
            wind_total += gens_p[g]
    for g in r['gas_gens']:
        if g in gens_p.columns:
            gas_total += gens_p[g]
    for g in r['nuclear_gens']:
        if g in gens_p.columns:
            nuclear_total += gens_p[g]
    for g in r['shed_gens']:
        if g in gens_p.columns:
            shed_total += gens_p[g]

    storage_dispatch = r['storage_p'].clip(lower=0).sum(axis=1) if isinstance(r['storage_p'], pd.DataFrame) and len(r['storage_p'].columns) > 0 and r['storage_p'].columns[0] != 'dummy' else pd.Series(0, index=range(168))

    ax.fill_between(range(168), 0, nuclear_total, alpha=0.7, color='purple', label='Nuclear')
    ax.fill_between(range(168), nuclear_total, nuclear_total + wind_total, alpha=0.7, color='green', label='Wind')
    ax.fill_between(range(168), nuclear_total + wind_total,
                    nuclear_total + wind_total + storage_dispatch, alpha=0.7, color='orange', label='Storage Discharge')
    ax.fill_between(range(168), nuclear_total + wind_total + storage_dispatch,
                    nuclear_total + wind_total + storage_dispatch + gas_total, alpha=0.7, color='gray', label='Gas')
    ax.fill_between(range(168), nuclear_total + wind_total + storage_dispatch + gas_total,
                    nuclear_total + wind_total + storage_dispatch + gas_total + shed_total,
                    alpha=0.7, color='red', label='Load Shedding')

    ax.plot(range(168), total_demand_ts, 'k-', linewidth=1.5, label='Total Demand')
    ax.set_ylabel('Power (MW)')
    ax.set_title(f'{scenario_name}: Generation Dispatch Stack')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 167)

# --- Figure 1: Network Topology ---
fig, ax = plt.subplots(figsize=(12, 10))
n_base = results['Baseline']['network']
bus_positions = {bus: (n_base.buses.at[bus, 'x'], n_base.buses.at[bus, 'y']) for bus in n_base.buses.index}

for link_name in n_base.links.index:
    bus0 = n_base.links.at[link_name, 'bus0']
    bus1 = n_base.links.at[link_name, 'bus1']
    x0, y0 = bus_positions[bus0]
    x1, y1 = bus_positions[bus1]
    ax.plot([x0, x1], [y0, y1], 'k-', alpha=0.5, linewidth=1)

bus_demand = demand_df.sum()
for bus_name in n_base.buses.index:
    x, y = bus_positions[bus_name]
    demand_val = bus_demand.get(bus_name, 0)
    size = demand_val / bus_demand.max() * 500
    ax.scatter(x, y, s=size, c='steelblue', edgecolors='navy', linewidth=0.5, zorder=5)
    ax.annotate(bus_name, (x, y), fontsize=7, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

for idx, gen in generators_df.iterrows():
    x, y = bus_positions[gen['bus']]
    if gen['carrier'] == 'nuclear':
        ax.scatter(x, y, s=100, c='red', marker='^', zorder=6, edgecolors='darkred')
    elif gen['carrier'] == 'onshore wind' and gen['p_nom'] >= 1000:
        ax.scatter(x, y, s=60, c='green', marker='s', zorder=6, edgecolors='darkgreen')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Bus (size ~ demand)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Nuclear'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='Wind (>=1 GW)'),
    Line2D([0], [0], color='k', linewidth=1, label='Transmission line'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GB Power System Network Topology (20-Bus Model)')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/network_topology.png', bbox_inches='tight')
plt.close()
print("Figure 1: Network topology saved")

# --- Figure 2: Demand and Wind Profiles ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].fill_between(range(168), total_demand_ts, alpha=0.4, color='steelblue')
axes[0].plot(range(168), total_demand_ts, color='navy', linewidth=0.8)
axes[0].set_ylabel('Total Demand (MW)')
axes[0].set_title('System-Wide Hourly Demand Profile (1 Week)')
axes[0].set_xlim(0, 167)

avg_wind_cf = wind_cf_df.mean(axis=1)
axes[1].fill_between(range(168), avg_wind_cf, alpha=0.4, color='green')
axes[1].plot(range(168), avg_wind_cf, color='darkgreen', linewidth=0.8)
axes[1].set_ylabel('Avg Wind Capacity Factor')
axes[1].set_xlabel('Hour')
axes[1].set_title('Average Wind Capacity Factor Profile (1 Week)')
axes[1].set_xlim(0, 167)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/demand_wind_profiles.png', bbox_inches='tight')
plt.close()
print("Figure 2: Demand and wind profiles saved")

# --- Figure 3: Generation Dispatch Stack (Baseline) ---
fig, ax = plt.subplots(figsize=(14, 7))
plot_dispatch_stack(ax, 'Baseline', results)
ax.set_xlabel('Hour')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/dispatch_stack_baseline.png', bbox_inches='tight')
plt.close()
print("Figure 3: Dispatch stack (baseline) saved")

# --- Figure 4: Scenario Comparison - Generation Mix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

wind_shares = [results[s]['total_wind_MWh'] / results[s]['total_demand_MWh'] * 100 for s in scenario_names]
gas_shares = [results[s]['total_gas_MWh'] / results[s]['total_demand_MWh'] * 100 for s in scenario_names]
nuclear_shares = [results[s]['total_nuclear_MWh'] / results[s]['total_demand_MWh'] * 100 for s in scenario_names]
shed_shares = [results[s]['total_shedding_MWh'] / results[s]['total_demand_MWh'] * 100 for s in scenario_names]

x = np.arange(len(scenario_names))
width = 0.6

axes[0].bar(x, nuclear_shares, width, label='Nuclear', color='purple', alpha=0.8)
axes[0].bar(x, wind_shares, width, bottom=nuclear_shares, label='Wind', color='green', alpha=0.8)
bottoms = [n + w for n, w in zip(nuclear_shares, wind_shares)]
axes[0].bar(x, gas_shares, width, bottom=bottoms, label='Gas', color='gray', alpha=0.8)
bottoms2 = [b + g for b, g in zip(bottoms, gas_shares)]
axes[0].bar(x, shed_shares, width, bottom=bottoms2, label='Load Shedding', color='red', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[0].set_ylabel('Share of Demand (%)')
axes[0].set_title('Generation Mix by Scenario')
axes[0].legend()

costs = [results[s]['total_cost'] / 1e6 for s in scenario_names]
colors = ['steelblue', 'green', 'orange', 'red']
bars = axes[1].bar(x, costs, width, color=colors, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[1].set_ylabel('Total System Cost (GBP M)')
axes[1].set_title('Total System Cost by Scenario')
for bar, cost in zip(bars, costs):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{cost:.1f}M', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/scenario_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 4: Scenario comparison saved")

# --- Figure 5: Curtailment Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for s_name in scenario_names:
    curt = results[s_name]['curtailment']
    axes[0].plot(range(168), curt, label=s_name, alpha=0.8, linewidth=1)

axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Curtailment (MW)')
axes[0].set_title('Hourly Wind Curtailment by Scenario')
axes[0].legend()
axes[0].set_xlim(0, 167)

curt_rates = []
for s_name in scenario_names:
    r = results[s_name]
    total_available = r['total_wind_MWh'] + r['total_curtailment_MWh']
    rate = r['total_curtailment_MWh'] / total_available * 100 if total_available > 0 else 0
    curt_rates.append(rate)

bars = axes[1].bar(x, curt_rates, width, color=colors, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenario_names, rotation=15, ha='right')
axes[1].set_ylabel('Curtailment Rate (%)')
axes[1].set_title('Wind Curtailment Rate by Scenario')
for bar, rate in zip(bars, curt_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/curtailment_analysis.png', bbox_inches='tight')
plt.close()
print("Figure 5: Curtailment analysis saved")

# --- Figure 6: High Renewable Dispatch Stack ---
fig, ax = plt.subplots(figsize=(14, 7))
plot_dispatch_stack(ax, 'High Renewable', results)
ax.set_xlabel('Hour')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/dispatch_stack_high_renewable.png', bbox_inches='tight')
plt.close()
print("Figure 6: High Renewable dispatch stack saved")

# --- Figure 7: Storage State of Charge ---
fig, ax = plt.subplots(figsize=(14, 6))
n_base = results['Baseline']['network']
if hasattr(n_base, 'storage_units_t') and hasattr(n_base.storage_units_t, 'state_of_charge'):
    soc = n_base.storage_units_t.state_of_charge
    for col in soc.columns:
        ax.plot(range(168), soc[col], label=col, linewidth=1.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('State of Charge (MWh)')
    ax.set_title('Storage Unit State of Charge (Baseline Scenario)')
    ax.legend()
    ax.set_xlim(0, 167)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/storage_soc.png', bbox_inches='tight')
plt.close()
print("Figure 7: Storage SOC saved")

# --- Figure 8: Transmission Line Utilization (Baseline) ---
fig, ax = plt.subplots(figsize=(14, 7))
n_base = results['Baseline']['network']
if hasattr(n_base.links_t, 'p0') and len(n_base.links_t.p0.columns) > 0:
    link_util = (n_base.links_t.p0.abs().mean() / n_base.links.p_nom * 100).sort_values(ascending=True)
    link_util.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
    ax.set_xlabel('Average Utilization (%)')
    ax.set_title('Transmission Line Average Utilization (Baseline)')
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Capacity limit')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/line_utilization.png', bbox_inches='tight')
plt.close()
print("Figure 8: Line utilization saved")

# --- Figure 9: Bus-level Capacity Map ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

bus_demand_total = demand_df.sum()
for bus_name, row in buses_df.iterrows():
    demand_val = bus_demand_total.get(bus_name, 0)
    axes[0].scatter(row['x'], row['y'], s=demand_val/500, c='steelblue', edgecolors='navy', linewidth=0.5)
    axes[0].annotate(bus_name, (row['x'], row['y']), fontsize=6, ha='center', va='bottom',
                     xytext=(0, 3), textcoords='offset points')
axes[0].set_title('Total Weekly Demand by Bus')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

wind_cap = generators_df[generators_df['carrier'] == 'onshore wind'].groupby('bus')['p_nom'].sum()
for bus_name, row in buses_df.iterrows():
    cap = wind_cap.get(bus_name, 0)
    axes[1].scatter(row['x'], row['y'], s=cap/30, c='green', edgecolors='darkgreen', linewidth=0.5)
    axes[1].annotate(bus_name, (row['x'], row['y']), fontsize=6, ha='center', va='bottom',
                     xytext=(0, 3), textcoords='offset points')
axes[1].set_title('Wind Capacity by Bus')
axes[1].set_xlabel('Longitude')

gas_cap = generators_df[generators_df['carrier'] == 'gas'].groupby('bus')['p_nom'].sum()
for bus_name, row in buses_df.iterrows():
    cap = gas_cap.get(bus_name, 0)
    axes[2].scatter(row['x'], row['y'], s=cap/5, c='gray', edgecolors='black', linewidth=0.5)
    axes[2].annotate(bus_name, (row['x'], row['y']), fontsize=6, ha='center', va='bottom',
                     xytext=(0, 3), textcoords='offset points')
axes[2].set_title('Gas Capacity by Bus')
axes[2].set_xlabel('Longitude')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/bus_level_capacity.png', bbox_inches='tight')
plt.close()
print("Figure 9: Bus-level capacity map saved")

# --- Figure 10: Constrained Network vs Baseline Dispatch Comparison ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

for scenario_name, ax in zip(['Baseline', 'Constrained Network'], axes):
    plot_dispatch_stack(ax, scenario_name, results)

axes[1].set_xlabel('Hour')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/constrained_vs_baseline.png', bbox_inches='tight')
plt.close()
print("Figure 10: Constrained vs Baseline saved")

# --- Figure 11: Load Shedding by Scenario ---
fig, ax = plt.subplots(figsize=(14, 6))
for s_name in scenario_names:
    r = results[s_name]
    shed_total = pd.Series(0, index=range(168))
    for g in r['shed_gens']:
        if g in r['generators_p'].columns:
            shed_total += r['generators_p'][g]
    ax.plot(range(168), shed_total, label=s_name, alpha=0.8, linewidth=1)

ax.set_xlabel('Hour')
ax.set_ylabel('Load Shedding (MW)')
ax.set_title('Hourly Load Shedding by Scenario')
ax.legend()
ax.set_xlim(0, 167)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/load_shedding.png', bbox_inches='tight')
plt.close()
print("Figure 11: Load shedding saved")

# --- Figure 12: Wind Generation vs Available Wind ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, s_name in enumerate(scenario_names):
    ax = axes[i // 2][i % 2]
    r = results[s_name]
    ax.fill_between(range(168), r['wind_available'], alpha=0.3, color='green', label='Available Wind')
    ax.fill_between(range(168), r['wind_dispatched'], alpha=0.7, color='green', label='Dispatched Wind')
    ax.plot(range(168), r['curtailment'], 'r-', linewidth=0.8, label='Curtailment')
    ax.set_title(s_name)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power (MW)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 167)
plt.suptitle('Wind Availability, Dispatch, and Curtailment by Scenario', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/wind_availability_dispatch.png', bbox_inches='tight')
plt.close()
print("Figure 12: Wind availability vs dispatch saved")

# ============================================================
# 6. Save detailed results tables
# ============================================================
mix_data = []
for s_name in scenario_names:
    r = results[s_name]
    mix_data.append({
        'Scenario': s_name,
        'Wind (GWh)': r['total_wind_MWh'] / 1000,
        'Gas (GWh)': r['total_gas_MWh'] / 1000,
        'Nuclear (GWh)': r['total_nuclear_MWh'] / 1000,
        'Load Shedding (GWh)': r['total_shedding_MWh'] / 1000,
        'Curtailment (GWh)': r['total_curtailment_MWh'] / 1000,
        'Demand (GWh)': r['total_demand_MWh'] / 1000,
        'Total Cost (GBP M)': r['total_cost'] / 1e6,
        'Wind Share (%)': r['total_wind_MWh'] / r['total_demand_MWh'] * 100,
        'Gas Share (%)': r['total_gas_MWh'] / r['total_demand_MWh'] * 100,
        'Nuclear Share (%)': r['total_nuclear_MWh'] / r['total_demand_MWh'] * 100,
        'Curtailment Rate (%)': r['total_curtailment_MWh'] / (r['total_wind_MWh'] + r['total_curtailment_MWh']) * 100 if (r['total_wind_MWh'] + r['total_curtailment_MWh']) > 0 else 0,
    })

mix_df = pd.DataFrame(mix_data)
mix_df.to_csv(f'{OUTPUT_DIR}/generation_mix.csv', index=False)

# Bus-level results for baseline
n_base = results['Baseline']['network']
bus_results = []
for bus_name in buses_df.index:
    bus_demand_total_val = demand_df[bus_name].sum() if bus_name in demand_df.columns else 0

    bus_wind = 0
    for g in results['Baseline']['wind_gens']:
        if g.startswith(bus_name + '_onshore wind'):
            if g in results['Baseline']['generators_p'].columns:
                bus_wind += results['Baseline']['generators_p'][g].sum()

    bus_gas = 0
    for g in results['Baseline']['gas_gens']:
        if g.startswith(bus_name + '_gas'):
            if g in results['Baseline']['generators_p'].columns:
                bus_gas += results['Baseline']['generators_p'][g].sum()

    bus_shed = 0
    for g in results['Baseline']['shed_gens']:
        if g.startswith(bus_name + '_load_shedding'):
            if g in results['Baseline']['generators_p'].columns:
                bus_shed += results['Baseline']['generators_p'][g].sum()

    bus_results.append({
        'Bus': bus_name,
        'Total_Demand_MWh': bus_demand_total_val,
        'Wind_Generation_MWh': bus_wind,
        'Gas_Generation_MWh': bus_gas,
        'Load_Shedding_MWh': bus_shed,
        'Net_Import_MWh': bus_demand_total_val - bus_wind - bus_gas - bus_shed,
    })

bus_df = pd.DataFrame(bus_results)
bus_df.to_csv(f'{OUTPUT_DIR}/bus_results.csv', index=False)

# Save method contract and artifact inventory
method_contract = {
    "objective": "Optimal power dispatch for GB 20-node system under multiple scenarios",
    "framework": "PyPSA v1.1+ with HiGHS solver",
    "formulation": "Linear Optimal Power Flow (LOPF)",
    "temporal_resolution": "hourly, 168 hours (1 week)",
    "spatial_resolution": "20 buses, 23 transmission links",
    "scenarios": ["Baseline", "High Renewable (3x wind)", "Constrained Network (50% line capacity)", "No Storage"],
    "load_shedding": "Enabled at 6000 GBP/MWh (VoLL) to ensure feasibility",
    "storage_modeling": "PHS with cyclic SOC, round-trip efficiency 0.75",
    "network_model": "Lossless DC power flow approximation via links",
}

with open(f'{OUTPUT_DIR}/method_contract.json', 'w') as f:
    json.dump(method_contract, f, indent=2)

artifact_inventory = {
    "figures": [
        "report/images/network_topology.png",
        "report/images/demand_wind_profiles.png",
        "report/images/dispatch_stack_baseline.png",
        "report/images/scenario_comparison.png",
        "report/images/curtailment_analysis.png",
        "report/images/dispatch_stack_high_renewable.png",
        "report/images/storage_soc.png",
        "report/images/line_utilization.png",
        "report/images/bus_level_capacity.png",
        "report/images/constrained_vs_baseline.png",
        "report/images/load_shedding.png",
        "report/images/wind_availability_dispatch.png",
    ],
    "data_tables": [
        "outputs/generation_mix.csv",
        "outputs/bus_results.csv",
        "outputs/scenario_summary.json",
    ],
    "code": ["code/main_analysis.py"],
}

with open(f'{OUTPUT_DIR}/target_artifact_inventory.json', 'w') as f:
    json.dump(artifact_inventory, f, indent=2)

print("\nAll results and figures saved successfully!")
print(f"\nScenario Summary:")
for name, s in summary.items():
    print(f"\n{name}:")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
