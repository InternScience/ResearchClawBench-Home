#!/usr/bin/env python3
"""Optimal Power Dispatch for GB Power System - v2"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = Path("../data")
OUTPUTS_DIR = Path("../outputs")
REPORT_IMAGES_DIR = Path("../report/images")

print("=" * 60)
print("Optimal Power Dispatch Optimization - v2")
print("=" * 60)

# Load data
buses = pd.read_csv(DATA_DIR / "buses.csv")
generators = pd.read_csv(DATA_DIR / "generators.csv")
storage = pd.read_csv(DATA_DIR / "storage.csv")
demand = pd.read_csv(DATA_DIR / "demand.csv")
wind_cf = pd.read_csv(DATA_DIR / "wind_cf.csv")

n_hours = len(demand)
wind_gens = generators[generators['carrier'] == 'onshore wind']
gas_gens = generators[generators['carrier'] == 'gas']
nuclear_gens = generators[generators['carrier'] == 'nuclear']

wind_capacity = wind_gens['p_nom'].values
gas_capacity = gas_gens['p_nom'].values
nuclear_capacity = nuclear_gens['p_nom'].values
wind_bus_names = wind_gens['bus'].values
storage_p_nom = storage['p_nom'].values
storage_e_nom = storage['e_nom'].values
storage_eff = storage['efficiency'].values

wind_available = np.zeros((n_hours, len(wind_gens)))
for i, bus_name in enumerate(wind_bus_names):
    wind_available[:, i] = wind_cf[bus_name].values * wind_capacity[i]

total_demand = demand.sum(axis=1).values

print(f"\nSystem: {n_hours}h | Peak: {total_demand.max()/1000:.1f}GW | Avg: {total_demand.mean()/1000:.1f}GW")
print(f"Capacities: Wind {wind_capacity.sum()/1000:.1f}GW | Gas {gas_capacity.sum()/1000:.1f}GW | Nuclear {nuclear_capacity.sum()/1000:.1f}GW")

MC_WIND, MC_NUCLEAR, MC_GAS = 0, 10, 50
n_wind, n_gas, n_nuclear, n_storage = len(wind_gens), len(gas_gens), len(nuclear_gens), len(storage)

wind_gen = np.zeros((n_hours, n_wind))
gas_gen = np.zeros((n_hours, n_gas))
nuclear_gen = np.zeros((n_hours, n_nuclear))
storage_discharge = np.zeros((n_hours, n_storage))
storage_charge = np.zeros((n_hours, n_storage))
storage_soc = np.zeros((n_hours, n_storage))
wind_curtail = np.zeros((n_hours, n_wind))
load_shed = np.zeros(n_hours)

soc = storage_e_nom * 0.5
for t in range(n_hours):
    D = total_demand[t]
    nuclear_gen[t] = nuclear_capacity
    residual = D - nuclear_gen[t].sum()
    wind_gen[t] = wind_available[t]
    residual -= wind_gen[t].sum()
    
    if residual > 0:
        for i in range(n_storage):
            max_discharge = min(storage_p_nom[i], soc[i] * storage_eff[i])
            discharge = min(max_discharge, residual)
            storage_discharge[t, i] = discharge
            soc[i] -= discharge / storage_eff[i]
            residual -= discharge
    else:
        excess = -residual
        for i in range(n_storage):
            max_charge = min(storage_p_nom[i], (storage_e_nom[i] - soc[i]) / storage_eff[i])
            charge = min(max_charge, excess)
            storage_charge[t, i] = charge
            soc[i] += charge * storage_eff[i]
            excess -= charge
        if excess > 0 and wind_gen[t].sum() > 0:
            curtail_ratio = min(excess / wind_gen[t].sum(), 1.0)
            wind_curtail[t] = wind_gen[t] * curtail_ratio
            wind_gen[t] -= wind_curtail[t]
            residual = -excess + wind_curtail[t].sum()
    
    if residual > 0:
        total_gas_cap = gas_capacity.sum()
        if total_gas_cap > 0:
            for i in range(n_gas):
                gas_gen[t, i] = min(gas_capacity[i], residual * (gas_capacity[i] / total_gas_cap))
            scale = min(1, residual / gas_gen[t].sum()) if gas_gen[t].sum() > 0 else 0
            gas_gen[t] *= scale
            residual -= gas_gen[t].sum()
    
    if residual > 0:
        load_shed[t] = residual
    storage_soc[t] = soc.copy()

wind_total = wind_gen.sum(axis=1)
nuclear_total = nuclear_gen.sum(axis=1)
gas_total = gas_gen.sum(axis=1)
storage_discharge_total = storage_discharge.sum(axis=1)
storage_charge_total = storage_charge.sum(axis=1)
wind_curtail_total = wind_curtail.sum(axis=1)
total_gen = wind_total + nuclear_total + gas_total + storage_discharge_total

total_cost = gas_total.sum() * MC_GAS + nuclear_total.sum() * MC_NUCLEAR

print(f"\nResults: Wind {wind_total.sum()/1000:.1f}GWh | Nuclear {nuclear_total.sum()/1000:.1f}GWh | Gas {gas_total.sum()/1000:.1f}GWh")
print(f"  Storage discharge: {storage_discharge_total.sum()/1000:.1f}GWh | Curtail: {wind_curtail_total.sum()/1000:.1f}GWh | Shed: {load_shed.sum()/1000:.1f}GWh")
print(f"  Total cost: £{total_cost/1e6:.2f}M")

results_df = pd.DataFrame({
    'hour': range(n_hours), 'demand': total_demand, 'wind_gen': wind_total,
    'nuclear_gen': nuclear_total, 'gas_gen': gas_total,
    'storage_discharge': storage_discharge_total, 'storage_charge': storage_charge_total,
    'wind_curtail': wind_curtail_total, 'load_shed': load_shed,
})
results_df.to_csv(OUTPUTS_DIR / 'dispatch_results.csv', index=False)

# Plots
time_axis = np.arange(n_hours)
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax.fill_between(time_axis, 0, wind_total/1000, label='Wind', color='#2ecc71', alpha=0.8)
ax.fill_between(time_axis, wind_total/1000, (wind_total+nuclear_total)/1000, label='Nuclear', color='#3498db', alpha=0.8)
ax.fill_between(time_axis, (wind_total+nuclear_total)/1000, (wind_total+nuclear_total+gas_total)/1000, label='Gas', color='#e74c3c', alpha=0.8)
ax.fill_between(time_axis, (wind_total+nuclear_total+gas_total)/1000, (wind_total+nuclear_total+gas_total+storage_discharge_total)/1000, label='Storage', color='orange', alpha=0.8)
ax.plot(time_axis, total_demand/1000, 'k--', linewidth=2, label='Demand')
ax.set_ylabel('Power (GW)')
ax.set_title('Optimal Generation Dispatch (One Week)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(time_axis, storage_soc.sum(axis=1)/1000, alpha=0.5, color='green')
ax.plot(time_axis, storage_soc.sum(axis=1)/1000, color='darkgreen', linewidth=2)
ax.set_ylabel('Storage SOC (GWh)')
ax.set_title('Total Storage State of Charge')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.fill_between(time_axis, wind_curtail_total/1000, alpha=0.5, color='red', label='Curtailment')
ax.fill_between(time_axis, load_shed/1000, alpha=0.5, color='black', label='Load Shed')
ax.set_xlabel('Hour')
ax.set_ylabel('Power (GW)')
ax.set_title('Curtailment and Load Shedding')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'dispatch_results.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'dispatch_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Generation mix pie chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
gen_totals = [wind_total.sum(), nuclear_total.sum(), gas_total.sum()]
labels = ['Wind', 'Nuclear', 'Gas']
colors = ['#2ecc71', '#3498db', '#e74c3c']
axes[0].pie(gen_totals, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Generation Mix by Energy')
axes[1].bar(range(n_hours), results_df['demand']/1000, alpha=0.3, color='gray', label='Demand')
axes[1].plot(range(n_hours), total_gen/1000, 'b-', linewidth=2, label='Generation')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Power (GW)')
axes[1].set_title('Demand vs Generation Matching')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'generation_mix.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'generation_mix.png', dpi=150, bbox_inches='tight')
plt.close()

metrics = {
    'total_demand_gwh': float(total_demand.sum()/1000),
    'wind_gen_gwh': float(wind_total.sum()/1000),
    'nuclear_gen_gwh': float(nuclear_total.sum()/1000),
    'gas_gen_gwh': float(gas_total.sum()/1000),
    'storage_discharge_gwh': float(storage_discharge_total.sum()/1000),
    'wind_curtail_gwh': float(wind_curtail_total.sum()/1000),
    'load_shed_gwh': float(load_shed.sum()/1000),
    'total_cost_mgbp': float(total_cost/1e6),
    'avg_cost_gbp_mwh': float(total_cost/total_demand.sum()*1000),
    'wind_share_pct': float(wind_total.sum()/total_gen.sum()*100),
    'curtailment_rate_pct': float(wind_curtail_total.sum()/(wind_total.sum()+wind_curtail_total.sum())*100) if (wind_total.sum()+wind_curtail_total.sum()) > 0 else 0,
}
with open(OUTPUTS_DIR / 'optimization_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nComplete! Files saved to outputs/ and report/images/")
