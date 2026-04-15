#!/usr/bin/env python3
"""
GB Power System Analysis
This script analyzes the Great Britain power system using optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_DIR = Path("../data")
OUTPUTS_DIR = Path("../outputs")
REPORT_IMAGES_DIR = Path("../report/images")

OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_IMAGES_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("GB Power System Analysis - Optimal Dispatch Model")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data files...")

buses = pd.read_csv(DATA_DIR / "buses.csv")
links = pd.read_csv(DATA_DIR / "links.csv")
generators = pd.read_csv(DATA_DIR / "generators.csv")
storage = pd.read_csv(DATA_DIR / "storage.csv")
demand = pd.read_csv(DATA_DIR / "demand.csv")
wind_cf = pd.read_csv(DATA_DIR / "wind_cf.csv")
hours = len(demand)

print(f"  - Buses: {len(buses)} nodes")
print(f"  - Links: {len(links)} transmission lines")
print(f"  - Generators: {len(generators)} units")
print(f"  - Storage: {len(storage)} units")
print(f"  - Demand/Wind CF: {hours} hours")

# Create snapshots index
snapshots = pd.date_range(start='2050-01-01', periods=hours, freq='h')
demand.index = snapshots
wind_cf.index = snapshots

# =============================================================================
# 2. DATA EXPLORATION AND VISUALIZATION
# =============================================================================
print("\n[2] Generating data overview plots...")

# Figure 1: Network Topology
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(buses['x'], buses['y'], s=300, c='steelblue', edgecolors='black', linewidth=2, zorder=3)
for idx, row in buses.iterrows():
    ax.annotate(row['name'], (row['x'], row['y']), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, ha='left')

for idx, row in links.iterrows():
    bus0 = buses[buses['name'] == row['bus0']].iloc[0]
    bus1 = buses[buses['name'] == row['bus1']].iloc[0]
    color = 'red' if row['p_nom'] > 4000 else 'darkgreen'
    linewidth = 3 if row['p_nom'] > 4000 else 1.5
    ax.plot([bus0['x'], bus1['x']], [bus0['y'], bus1['y']], 
            color=color, linewidth=linewidth, alpha=0.7, zorder=1)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('GB Power System Network Topology (20 Buses, 23 Transmission Lines)', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'network_topology.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'network_topology.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: network_topology.png")

# Figure 2: Generator capacity by type
gen_by_carrier = generators.groupby('carrier')['p_nom'].sum() / 1000
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'onshore wind': '#2ecc71', 'gas': '#e74c3c', 'nuclear': '#3498db'}
bar_colors = [colors.get(c, 'gray') for c in gen_by_carrier.index]
bars = ax.bar(gen_by_carrier.index, gen_by_carrier.values, color=bar_colors, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f} GW', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Generator Type', fontsize=12)
ax.set_ylabel('Installed Capacity (GW)', fontsize=12)
ax.set_title('Generator Capacity by Type', fontsize=14)
ax.set_ylim(0, max(gen_by_carrier.values) * 1.2)
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'generator_capacity.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'generator_capacity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: generator_capacity.png")

# Figure 3: Demand and Wind Profiles
time_axis = np.arange(hours)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

total_demand = demand.sum(axis=1)
axes[0, 0].plot(time_axis, total_demand / 1000, color='steelblue', linewidth=2)
axes[0, 0].fill_between(time_axis, total_demand / 1000, alpha=0.3, color='steelblue')
axes[0, 0].set_xlabel('Hour', fontsize=11)
axes[0, 0].set_ylabel('Demand (GW)', fontsize=11)
axes[0, 0].set_title('Total System Demand (One Week)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

mean_wind_cf = wind_cf.mean(axis=1)
axes[0, 1].plot(time_axis, mean_wind_cf, color='green', linewidth=2)
axes[0, 1].fill_between(time_axis, wind_cf.min(axis=1), wind_cf.max(axis=1), 
                        alpha=0.3, color='green', label='Min-Max Range')
axes[0, 1].plot(time_axis, mean_wind_cf, color='darkgreen', linewidth=2, label='Mean')
axes[0, 1].set_xlabel('Hour', fontsize=11)
axes[0, 1].set_ylabel('Capacity Factor', fontsize=11)
axes[0, 1].set_title('Wind Capacity Factors (All Buses)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Storage capacity
axes[1, 0].bar(storage['bus'], storage['p_nom'], color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Bus', fontsize=11)
axes[1, 0].set_ylabel('Power Capacity (MW)', fontsize=11)
axes[1, 0].set_title('Pumped Hydro Storage Capacity by Bus', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# Marginal costs
marginal_costs = generators.groupby('carrier')['marginal_cost'].first()
axes[1, 1].bar(marginal_costs.index, marginal_costs.values, 
               color=[colors.get(c, 'gray') for c in marginal_costs.index], edgecolor='black')
axes[1, 1].set_xlabel('Generator Type', fontsize=11)
axes[1, 1].set_ylabel('Marginal Cost (£/MWh)', fontsize=11)
axes[1, 1].set_title('Marginal Costs by Generator Type', fontsize=12)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'demand_wind_profiles.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTPUTS_DIR / 'demand_wind_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: demand_wind_profiles.png")

# Save summary statistics
summary_stats = {
    'total_demand_mwh': float(total_demand.sum()),
    'peak_demand_mw': float(total_demand.max()),
    'min_demand_mw': float(total_demand.min()),
    'avg_demand_mw': float(total_demand.mean()),
    'total_generation_capacity_mw': float(generators['p_nom'].sum()),
    'wind_capacity_mw': float(generators[generators['carrier'] == 'onshore wind']['p_nom'].sum()),
    'gas_capacity_mw': float(generators[generators['carrier'] == 'gas']['p_nom'].sum()),
    'nuclear_capacity_mw': float(generators[generators['carrier'] == 'nuclear']['p_nom'].sum()),
    'storage_power_capacity_mw': float(storage['p_nom'].sum()),
    'storage_energy_capacity_mwh': float(storage['e_nom'].sum()),
    'total_transmission_capacity_mw': float(links['p_nom'].sum()),
}

with open(OUTPUTS_DIR / 'system_summary.json', 'w') as f:
    import json
    json.dump(summary_stats, f, indent=2)

print(f"\nSystem Summary:")
print(f"  - Total Demand: {summary_stats['total_demand_mwh']/1000:.1f} GWh")
print(f"  - Peak Demand: {summary_stats['peak_demand_mw']/1000:.1f} GW")
print(f"  - Total Generation Capacity: {summary_stats['total_generation_capacity_mw']/1000:.1f} GW")
print(f"  - Wind: {summary_stats['wind_capacity_mw']/1000:.1f} GW")
print(f"  - Gas: {summary_stats['gas_capacity_mw']/1000:.1f} GW")
print(f"  - Nuclear: {summary_stats['nuclear_capacity_mw']/1000:.1f} GW")

# Save processed data for optimization
demand.to_csv(OUTPUTS_DIR / 'processed_demand.csv')
wind_cf.to_csv(OUTPUTS_DIR / 'processed_wind_cf.csv')

print("\n[3] Data processing complete. Ready for optimization.")
