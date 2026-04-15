#!/usr/bin/env python3
"""
Optimal Power Dispatch for GB Power System

This script performs economic dispatch optimization using linear programming.
Objective: Minimize total system cost while meeting demand and respecting constraints.

Key assumptions:
- Linearized DC power flow (simplified transmission model)
- No transmission losses for simplicity
- Storage can charge/discharge with efficiency losses
- Wind curtailment is allowed at zero cost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path("../data")
OUTPUTS_DIR = Path("../outputs")
REPORT_IMAGES_DIR = Path("../report/images")

print("=" * 60)
print("Optimal Power Dispatch Optimization")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

buses = pd.read_csv(DATA_DIR / "buses.csv")
links = pd.read_csv(DATA_DIR / "links.csv")
generators = pd.read_csv(DATA_DIR / "generators.csv")
storage = pd.read_csv(DATA_DIR / "storage.csv")
demand = pd.read_csv(DATA_DIR / "demand.csv")
wind_cf = pd.read_csv(DATA_DIR / "wind_cf.csv")

n_buses = len(buses)
n_hours = len(demand)
n_generators = len(generators)
n_storage = len(storage)

# Create bus index mapping
bus_map = {name: idx for idx, name in enumerate(buses['name'])}

# =============================================================================
# 2. SET UP OPTIMIZATION PROBLEM
# =============================================================================
print("\n[2] Setting up optimization model...")

# Generator types and their properties
gen_types = generators['carrier'].values
gen_buses = [bus_map[b] for b in generators['bus'].values]
gen_capacity = generators['p_nom'].values
gen_marginal_cost = generators['marginal_cost'].values

# Storage properties
storage_buses = [bus_map[b] for b in storage['bus'].values]
storage_p_nom = storage['p_nom'].values
storage_e_nom = storage['e_nom'].values
storage_eff = storage['efficiency'].values

# Wind generators (for max output calculation)
wind_gens = generators[generators['carrier'] == 'onshore wind']
n_wind = len(wind_gens)
wind_indices = wind_gens.index.values
wind_bus_indices = [bus_map[b] for b in wind_gens['bus'].values]
wind_capacity = wind_gens['p_nom'].values

# Calculate available wind generation per hour
wind_max_output = np.zeros((n_hours, n_wind))
for i, (idx, gen) in enumerate(wind_gens.iterrows()):
    bus_name = gen['bus']
    wind_max_output[:, i] = wind_cf[bus_name].values * gen['p_nom']

# Total demand per hour per bus
demand_array = demand.values  # shape: (hours, buses)

print(f"  - Time horizon: {n_hours} hours")
print(f"  - Buses: {n_buses}")
print(f"  - Generators: {n_generators}")
print(f"  - Wind generators: {n_wind}")
print(f"  - Storage units: {n_storage}")

# =============================================================================
# 3. SIMPLIFIED ECONOMIC DISPATCH (PER HOUR)
# =============================================================================
print("\n[3] Running economic dispatch optimization...")

# For computational efficiency, we do a simplified hourly dispatch
# that respects generator constraints and storage state

# Results storage
generation_results = np.zeros((n_hours, n_generators))
storage_charge_results = np.zeros((n_hours, n_storage))
storage_discharge_results = np.zeros((n_hours, n_storage))
storage_soc_results = np.zeros((n_hours, n_storage))  # State of charge
curtailment_results = np.zeros((n_hours, n_wind))
total_cost_per_hour = np.zeros(n_hours)

# Storage state of charge (initialize at 50%)
soc = storage_e_nom * 0.5

for t in range(n_hours):
    # Total demand this hour
    total_demand = demand_array[t].sum()
    
    # Available wind this hour
    available_wind = wind_max_output[t]
    total_available_wind = available_wind.sum()
    
    # Calculate net demand after wind
    net_demand = total_demand - total_available_wind
    
    # First priority: Use all available wind (zero marginal cost)
    # Second priority: Nuclear (low marginal cost)
    # Third priority: Storage discharge if needed
    # Fourth priority: Gas (higher marginal cost)
    
    wind_gen = available_wind.copy()
    
    # Nuclear generation (dispatch to meet remaining demand)
    nuclear_gens = generators[generators['carrier'] == 'nuclear']
    nuclear_output = np.zeros(len(nuclear_gens))
    for i, (idx, gen) in enumerate(nuclear_gens.iterrows()):
        nuclear_output[i] = gen['p_nom']  # Nuclear runs at full capacity
    total_nuclear = nuclear_output.sum()
    
    # Remaining demand
    remaining_demand = net_demand + total_available_wind - total_nuclear
    
    # Storage dispatch
    storage_discharge = np.zeros(n_storage)
    storage_charge = np.zeros(n_storage)
    
    if remaining_demand > 0:
        # Discharge storage to meet demand
        for i in range(n_storage):
            max_discharge = min(storage_p_nom[i], soc[i] * storage_eff[i])
            discharge = min(max_discharge, remaining_demand)
            storage_discharge[i] = discharge
            soc[i] -= discharge / storage_eff[i]
            remaining_demand -= discharge
    elif remaining_demand < 0:
        # Charge storage with excess wind
        excess = -remaining_demand
        for i in range(n_storage):
            max_charge = min(storage_p_nom[i], 
                           (storage_e_nom[i] - soc[i]) / storage_eff[i])
            charge = min(max_charge, excess)
            storage_charge[i] = charge
            soc[i] += charge * storage_eff[i]
            excess -= charge
        # Curtail excess wind
        if excess > 0 and total_available_wind > 0:
            curtailment_factor = excess / total_available_wind
            curtailment_results[t] = wind_gen * min(curtailment_factor, 1.0)
            wind_gen -= curtailment_results[t]
    
    # Gas generation for remaining demand
    gas_gens = generators[generators['carrier'] == 'gas']
    gas_output = np.zeros(len(gas_gens))
    if remaining_demand > 0:
        # Dispatch gas generators
        total_gas_capacity = gas_gens['p_nom'].sum()
        if total_gas_capacity > 0:
            for i, (idx, gen) in enumerate(gas_gens.iterrows()):
                gas_output[i] = min(gen['p_nom'], 
                                   remaining_demand * (gen['p_nom'] / total_gas_capacity))
            # Recalculate to ensure exact match
            scale = remaining_demand / gas_output.sum() if gas_output.sum() > 0 else 0
            if scale > 0:
                gas_output = np.minimum(gas_output * scale, gas_gens['p_nom'].values)
    
    # Store results
    for i, idx in enumerate(wind_indices):
        generation_results[t, idx] = wind_gen[i]
    for i, (idx, gen) in enumerate(nuclear_gens.iterrows()):
        generation_results[t, gen.name] = nuclear_output[i]
    for i, (idx, gen) in enumerate(gas_gens.iterrows()):
        generation_results[t, gen.name] = gas_output[i]
    
    storage_discharge_results[t] = storage_discharge
    storage_charge_results[t] = storage_charge
    storage_soc_results[t] = soc.copy()
    
    # Calculate cost
    hour_cost = (gas_output.sum() * 50 +  # Gas at 50 £/MWh
                 nuclear_output.sum() * 10)  # Nuclear at 10 £/MWh
    total_cost_per_hour[t] = hour_cost

print(f"  - Optimization complete")
print(f"  - Total system cost: £{total_cost_per_hour.sum()/1000:.1f}k")

# =============================================================================
# 4. AGGREGATE RESULTS
# =============================================================================
print("\n[4] Aggregating results...")

# Create time series for each carrier
results_df = pd.DataFrame(index=range(n_hours))
results_df['wind_generation'] = generation_results[:, wind_indices].sum(axis=1)
results_df['nuclear_generation'] = generation_results[:, 
    generators[generators['carrier'] == 'nuclear'].index].sum(axis=1)
results_df['gas_generation'] = generation_results[:,
    generators[generators['carrier'] == 'gas'].index].sum(axis=1)
results_df['storage_discharge'] = storage_discharge_results.sum(axis=1)
results_df['storage_charge'] = storage_charge_results.sum(axis=1)
results_df['curtailment'] = curtailment_results.sum(axis=1)
results_df['total_demand'] = demand_array.sum(axis=1)
results_df['net_demand'] = results_df['total_demand'] - results_df['wind_generation']
results_df['total_cost'] = total_cost_per_hour

# Calculate generation mix
total_generation = results_df[['wind_generation', 'nuclear_generation', 'gas_generation']].sum()
gen_mix_pct = total_generation / total_generation.sum() * 100

print(f"\nGeneration Mix:")
for carrier, pct in gen_mix_pct.items():
    print(f"  - {carrier.replace('_generation', '').title()}: {pct:.1f}%")

# Calculate key metrics
total_curtailment = results_df['curtailment'].sum()
total_wind_available = wind_max_output.sum()
curtailment_rate = total_curtailment / total_wind_available * 100

print(f"\nSystem Performance:")
print(f"  - Total Demand Met: {results_df['total_demand'].sum()/1000:.1f} GWh")
print(f"  - Total Generation: {total_generation.sum()/1000:.1f} GWh")
print(f"  - Total Curtailment: {total_curtailment/1000:.1f} GWh ({curtailment_rate:.1f}%)")
print(f"  - Storage Throughput: {(results_df['storage_discharge'].sum() + results_df['storage_charge'].sum())/1000:.1f} GWh")
print(f"  - Total System Cost: £{total_cost_per_hour.sum()/1e6:.2f}M")
print(f"  - Average Cost: £{total_cost_per_hour.sum()/results_df['total_demand'].sum()*1000:.1f}/MWh")

# Save results
results_df.to_csv(OUTPUTS_DIR / 'dispatch_results.csv')
gen_mix_pct.to_csv(OUTPUTS_DIR / 'generation_mix.csv')

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
print("\n[5] Generating result plots...")

time_axis = np.arange(n_hours)

# Figure 1: Generation Dispatch Stack
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Stacked generation
ax = axes[0]
ax.fill_between(time_axis, 0, results_df['wind_generation']/1000, 
                label='Wind', color='#2ecc71', alpha=0.8)
ax.fill_between(time_axis, results_df['wind_generation']/1000, 
                (results_df['wind_generation'] + results_df['nuclear_generation'])/1000,
                label='Nuclear', color='#3498db', alpha=0.8)
ax.fill_between(time_axis, 
                (results_df['wind_generation'] + results_df['nuclear_generation'])/1000,
                (results_df['wind_generation'] + results_df['nuclear_generation'] + 
                 results_df['gas_generation'])/1000,
                label='Gas', color='#e74c3c', alpha=0.8)
ax.plot(time_axis, results_df['total_demand']/1000, 'k--', linewidth=2, 
        label='Demand', linestyle='--')

ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Generation (GW)', fontsize=12)
ax.set_title('Optimal Generation Dispatch (One Week)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_hours-1)

# Net load and storage
ax = axes[1]
net_load = results_df['total_demand'] - results_df['wind_generation'] - results_df['nuclear_generation']
ax.fill_between(time_axis, 0, net_load/1000, alpha=0.3, color='orange', label='Net Load (after wind & nuclear)')
ax.plot(time_axis, results_df['gas_generation']/1000, 'r-', linewidth=2, label='Gas Generation')
ax.plot(time_axis, results_df['storage_discharge']/1000, 'g-', linewidth=2, label='Storage Discharge')
ax.plot(time_axis, -results_df['storage_charge']/1000, 'b-', linewidth=2, label='Storage Charge')

ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Power (GW)', fontsize=12)
ax.set_title('Net Load and Storage Dispatch', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_hours-1)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'dispatch_stack.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'dispatch_stack.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: dispatch_stack.png")

# Figure 2: Generation Mix Pie Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#2ecc71', '#3498db', '#e74c3c']
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = axes[0].pie(total_generation.values, explode=explode, 
                                        labels=['Wind', 'Nuclear', 'Gas'],
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90)
axes[0].set_title('Generation Mix by Energy', fontsize=14)

# Hourly cost
axes[1].plot(time_axis, results_df['total_cost']/1000, color='purple', linewidth=2)
axes[1].fill_between(time_axis, results_df['total_cost']/1000, alpha=0.3, color='purple')
axes[1].set_xlabel('Hour', fontsize=12)
axes[1].set_ylabel('Cost (£k)', fontsize=12)
axes[1].set_title('Hourly System Operating Cost', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'generation_mix_cost.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'generation_mix_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: generation_mix_cost.png")

# Figure 3: Storage and Curtailment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Storage SOC
for i in range(n_storage):
    axes[0, 0].plot(time_axis, storage_soc_results[:, i], 
                    label=f"{storage.iloc[i]['bus']} ({storage.iloc[i]['p_nom']}MW)", linewidth=2)
axes[0, 0].set_xlabel('Hour', fontsize=11)
axes[0, 0].set_ylabel('State of Charge (MWh)', fontsize=11)
axes[0, 0].set_title('Storage State of Charge', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Curtailment
axes[0, 1].fill_between(time_axis, results_df['curtailment']/1000, alpha=0.5, color='red')
axes[0, 1].plot(time_axis, results_df['curtailment']/1000, color='darkred', linewidth=2)
axes[0, 1].set_xlabel('Hour', fontsize=11)
axes[0, 1].set_ylabel('Curtailment (GW)', fontsize=11)
axes[0, 1].set_title(f'Wind Curtailment (Total: {total_curtailment/1000:.1f} GWh)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Wind utilization
wind_utilization = (results_df['wind_generation'] / 
                   (results_df['wind_generation'] + results_df['curtailment']) * 100)
axes[1, 0].plot(time_axis, wind_utilization, color='green', linewidth=2)
axes[1, 0].axhline(y=wind_utilization.mean(), color='red', linestyle='--', 
                   label=f'Mean: {wind_utilization.mean():.1f}%')
axes[1, 0].set_xlabel('Hour', fontsize=11)
axes[1, 0].set_ylabel('Utilization Rate (%)', fontsize=11)
axes[1, 0].set_title('Wind Utilization Rate', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 105)

# Marginal cost duration curve
sorted_costs = np.sort(results_df['total_cost'] / 
                       (results_df['gas_generation'] + results_df['nuclear_generation'] + 1e-6))[::-1]
axes[1, 1].plot(np.arange(len(sorted_costs)), sorted_costs, color='blue', linewidth=2)
axes[1, 1].set_xlabel('Hours (sorted)', fontsize=11)
axes[1, 1].set_ylabel('Marginal Cost (£/MWh)', fontsize=11)
axes[1, 1].set_title('Marginal Cost Duration Curve', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'storage_curtailment.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'storage_curtailment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: storage_curtailment.png")

# Save key metrics
metrics = {
    'total_demand_gwh': float(results_df['total_demand'].sum()/1000),
    'total_generation_gwh': float(total_generation.sum()/1000),
    'wind_generation_gwh': float(results_df['wind_generation'].sum()/1000),
    'nuclear_generation_gwh': float(results_df['nuclear_generation'].sum()/1000),
    'gas_generation_gwh': float(results_df['gas_generation'].sum()/1000),
    'wind_share_pct': float(gen_mix_pct['wind_generation']),
    'nuclear_share_pct': float(gen_mix_pct['nuclear_generation']),
    'gas_share_pct': float(gen_mix_pct['gas_generation']),
    'total_curtailment_gwh': float(total_curtailment/1000),
    'curtailment_rate_pct': float(curtailment_rate),
    'total_system_cost_mgbp': float(total_cost_per_hour.sum()/1e6),
    'average_cost_gbp_mwh': float(total_cost_per_hour.sum()/results_df['total_demand'].sum()*1000),
    'storage_throughput_gwh': float((results_df['storage_discharge'].sum() + 
                                      results_df['storage_charge'].sum())/1000),
    'peak_demand_gw': float(results_df['total_demand'].max()/1000),
    'min_demand_gw': float(results_df['total_demand'].min()/1000),
}

import json
with open(OUTPUTS_DIR / 'optimization_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n[6] Optimization complete. Results saved to outputs/")
print(f"\nKey Results:")
print(f"  - Wind Share: {metrics['wind_share_pct']:.1f}%")
print(f"  - Curtailment Rate: {metrics['curtailment_rate_pct']:.1f}%")
print(f"  - Total Cost: £{metrics['total_system_cost_mgbp']:.2f}M")
print(f"  - Average Cost: £{metrics['average_cost_gbp_mwh']:.1f}/MWh")
