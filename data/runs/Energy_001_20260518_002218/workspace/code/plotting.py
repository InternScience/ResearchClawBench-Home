"""
GB 20-Bus Power System: Visualization Script
=============================================
Generates all figures for the research report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_001_20260518_002218'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
IMG_DIR = os.path.join(BASE_DIR, 'report', 'images')

os.makedirs(IMG_DIR, exist_ok=True)

# Load data
buses = pd.read_csv(os.path.join(DATA_DIR, 'buses.csv'))
links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))

# Load results
with open(os.path.join(OUTPUT_DIR, 'system_costs.json'), 'r') as f:
    summaries = json.load(f)

# Color scheme
COLORS = {
    'onshore wind': '#2196F3',
    'gas': '#FF9800',
    'nuclear': '#9C27B0',
    'load_shedding': '#F44336',
    'curtailment': '#795548',
    'storage_charge': '#4CAF50',
    'storage_discharge': '#009688',
}

SCENARIO_COLORS = {
    'base': '#1976D2',
    'high_wind': '#43A047',
    'constrained_tx': '#E53935',
}

SCENARIO_LABELS = {
    'base': 'Base Case',
    'high_wind': 'High Wind (3x)',
    'constrained_tx': 'Constrained TX (50%)',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

# ============================
# Figure 1: Network Topology
# ============================
print("Generating Figure 1: Network Topology...")

fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Draw links
for _, link in links.iterrows():
    bus0 = buses[buses['name'] == link['bus0']].iloc[0]
    bus1 = buses[buses['name'] == link['bus1']].iloc[0]
    color = '#B0BEC5' if link['bus0'] in ['Bus1', 'Bus2', 'Bus3', 'Bus4', 'Bus5'] and link['bus1'] in ['Bus6', 'Bus7', 'Bus8', 'Bus9', 'Bus10'] else '#78909C'
    width = 3 if link['p_nom'] == 5000 else 1.5
    ax.plot([bus0['x'], bus1['x']], [bus0['y'], bus1['y']], 
            color=color, linewidth=width, alpha=0.7, zorder=1)
    mid_x = (bus0['x'] + bus1['x']) / 2
    mid_y = (bus0['y'] + bus1['y']) / 2
    cap = f"{link['p_nom']/1000:.0f} GW"
    ax.annotate(cap, (mid_x, mid_y), fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

# Draw buses
for _, bus in buses.iterrows():
    # Color by generator presence
    has_wind = bus['name'] in [f"Bus{i}" for i in range(1, 6)]  # Large wind
    has_nuclear = bus['name'] in ['Bus2', 'Bus8', 'Bus14']
    
    if has_nuclear:
        color = '#9C27B0'
        size = 300
    elif has_wind:
        color = '#2196F3'
        size = 250
    else:
        color = '#FF9800'
        size = 200
    
    ax.scatter(bus['x'], bus['y'], c=color, s=size, zorder=3, edgecolors='white', linewidth=1.5)
    ax.annotate(bus['name'], (bus['x'], bus['y']), fontsize=9, ha='center', va='bottom',
                xytext=(0, 10), textcoords='offset points', fontweight='bold')

# Legend
legend_elements = [
    plt.scatter([], [], c='#2196F3', s=100, label='Large Wind Bus (10 GW)'),
    plt.scatter([], [], c='#FF9800', s=100, label='Standard Bus (0.5 GW wind + gas)'),
    plt.scatter([], [], c='#9C27B0', s=100, label='Nuclear Bus'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)

ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('GB 20-Bus Power System Network Topology', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_network_topology.png'))
plt.close()
print("  Done.")

# ============================
# Figure 2: Dispatch Stacked Area (Base Case)
# ============================
print("Generating Figure 2: Base Case Dispatch...")

ts_base = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_timeseries.csv'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

# Stacked area: wind, nuclear, gas, load_shedding
hours = np.arange(len(ts_base))
ax1.fill_between(hours, 0, ts_base['total_nuclear_mw']/1000, 
                 color=COLORS['nuclear'], alpha=0.8, label='Nuclear')
ax1.fill_between(hours, ts_base['total_nuclear_mw']/1000, 
                 (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'])/1000,
                 color=COLORS['gas'], alpha=0.8, label='Gas')
ax1.fill_between(hours, 
                 (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'])/1000,
                 (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'] + ts_base['total_wind_mw'])/1000,
                 color=COLORS['onshore wind'], alpha=0.8, label='Wind')
ax1.fill_between(hours,
                 (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'] + ts_base['total_wind_mw'])/1000,
                 (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'] + ts_base['total_wind_mw'] + ts_base['total_load_shed_mw'])/1000,
                 color=COLORS['load_shedding'], alpha=0.8, label='Load Shedding')
ax1.plot(hours, ts_base['total_demand_mw']/1000, 'k--', linewidth=1.5, label='Demand')
ax1.plot(hours, ts_base['wind_available_mw']/1000, color='#0D47A1', linewidth=1, alpha=0.5, linestyle=':', label='Wind Available')
ax1.plot(hours, ts_base['wind_curtailment_mw']/1000 + (ts_base['total_nuclear_mw'] + ts_base['total_gas_mw'] + ts_base['total_wind_mw'])/1000,
         color=COLORS['curtailment'], linewidth=0.5, alpha=0.3)

ax1.set_ylabel('Power (GW)')
ax1.set_title('Optimal Dispatch - Base Case (168 Hours)', fontweight='bold')
ax1.legend(loc='upper right', fontsize=9, ncol=2)
ax1.set_xlim(0, len(ts_base)-1)
ax1.grid(True, alpha=0.3)

# Wind curtailment bar chart
ax2.bar(hours, ts_base['wind_curtailment_mw']/1000, color=COLORS['curtailment'], alpha=0.7, width=1.0)
ax2.set_xlabel('Hour')
ax2.set_ylabel('Curtailment (GW)')
ax2.set_title('Wind Curtailment', fontweight='bold')
ax2.set_xlim(0, len(ts_base)-1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_dispatch_stacked.png'))
plt.close()
print("  Done.")

# ============================
# Figure 3: Generation Mix Comparison
# ============================
print("Generating Figure 3: Generation Mix Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for idx, (sc, label) in enumerate(SCENARIO_LABELS.items()):
    s = summaries[sc]
    techs = ['onshore wind', 'gas', 'nuclear', 'load_shedding']
    values = [
        s['total_wind_mwh'] / 1e6,
        s['total_gas_mwh'] / 1e6,
        s['total_nuclear_mwh'] / 1e6,
        s['total_load_shedding_mwh'] / 1e6,
    ]
    colors = [COLORS[t] for t in techs]
    
    bars = axes[idx].bar(techs, values, color=colors, alpha=0.85, edgecolor='white')
    axes[idx].set_title(label, fontweight='bold')
    axes[idx].set_ylabel('Energy (TWh)' if idx == 0 else '')
    axes[idx].set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.02,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Generation Mix by Scenario (168 Hours)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_generation_mix_comparison.png'))
plt.close()
print("  Done.")

# ============================
# Figure 4: Curtailment Analysis
# ============================
print("Generating Figure 4: Curtailment Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Curtailment percentage comparison
scenarios_list = list(SCENARIO_LABELS.keys())
curtailment_pcts = [summaries[sc]['curtailment_pct'] for sc in scenarios_list]
wind_pen_pcts = [summaries[sc]['wind_penetration_pct'] for sc in scenarios_list]

x = np.arange(len(scenarios_list))
width = 0.35
bars1 = ax1.bar(x - width/2, curtailment_pcts, width, label='Curtailment %', color=COLORS['curtailment'], alpha=0.8)
bars2 = ax1.bar(x + width/2, wind_pen_pcts, width, label='Wind Penetration %', color=COLORS['onshore wind'], alpha=0.8)

ax1.set_ylabel('Percentage (%)')
ax1.set_title('Wind Curtailment vs Penetration', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([SCENARIO_LABELS[sc] for sc in scenarios_list], fontsize=9)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

# Right: Hourly wind curtailment for base case
ts = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_timeseries.csv'))
hours = np.arange(len(ts))
ax2.fill_between(hours, 0, ts['wind_available_mw']/1000, alpha=0.3, color=COLORS['onshore wind'], label='Available Wind')
ax2.fill_between(hours, 0, ts['total_wind_mw']/1000, alpha=0.7, color=COLORS['onshore wind'], label='Dispatched Wind')
ax2.fill_between(hours, ts['total_wind_mw']/1000, ts['wind_available_mw']/1000, alpha=0.7, color=COLORS['curtailment'], label='Curtailment')
ax2.set_xlabel('Hour')
ax2.set_ylabel('Power (GW)')
ax2.set_title('Hourly Wind Curtailment (Base Case)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim(0, len(ts)-1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_curtailment_analysis.png'))
plt.close()
print("  Done.")

# ============================
# Figure 5: Storage Utilization
# ============================
print("Generating Figure 5: Storage Utilization...")

try:
    storage_base = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_storage_dispatch.csv'))
    storage_soc_base = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_storage_soc.csv'))
    storage_high = pd.read_csv(os.path.join(OUTPUT_DIR, 'high_wind_storage_dispatch.csv'))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    hours = np.arange(len(storage_base))
    
    # Top: Storage dispatch
    for col in storage_base.columns:
        axes[0].plot(hours, storage_base[col]/1000, linewidth=1.2, label=col)
    axes[0].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('Power (GW)')
    axes[0].set_title('Storage Dispatch - Base Case', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, len(hours)-1)
    
    # Bottom: State of charge
    for col in storage_soc_base.columns:
        axes[1].plot(hours, storage_soc_base[col]/1000, linewidth=1.2, label=col)
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('State of Charge (GWh)')
    axes[1].set_title('Storage State of Charge - Base Case', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, len(hours)-1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig5_storage_utilization.png'))
    plt.close()
    print("  Done.")
except Exception as e:
    print(f"  Warning: {e}")
    # Create placeholder
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Storage utilization data not available', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Storage Utilization')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig5_storage_utilization.png'))
    plt.close()

# ============================
# Figure 6: Line Utilization
# ============================
print("Generating Figure 6: Line Utilization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (sc, label) in enumerate(SCENARIO_LABELS.items()):
    link_util = summaries[sc]['link_utilization']
    link_names = sorted(link_util.keys())
    avg_utils = [link_util[l]['avg_util_pct'] for l in link_names]
    peak_utils = [link_util[l]['peak_util_pct'] for l in link_names]
    
    x = np.arange(len(link_names))
    width = 0.35
    axes[idx].bar(x - width/2, avg_utils, width, label='Avg Util.', color=SCENARIO_COLORS[sc], alpha=0.7)
    axes[idx].bar(x + width/2, peak_utils, width, label='Peak Util.', color=SCENARIO_COLORS[sc], alpha=0.4)
    axes[idx].set_title(label, fontweight='bold')
    axes[idx].set_ylabel('Utilization (%)' if idx == 0 else '')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(link_names, rotation=90, fontsize=7)
    axes[idx].set_ylim(0, max(max(avg_utils), max(peak_utils)) * 1.2 + 1)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('Transmission Line Utilization by Scenario', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_line_utilization.png'))
plt.close()
print("  Done.")

# ============================
# Figure 7: System Cost Comparison
# ============================
print("Generating Figure 7: System Cost Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Total cost comparison
scenarios_list = list(SCENARIO_LABELS.keys())
total_costs = [summaries[sc]['total_objective'] / 1e9 for sc in scenarios_list]
bars = ax1.bar([SCENARIO_LABELS[sc] for sc in scenarios_list], total_costs, 
               color=[SCENARIO_COLORS[sc] for sc in scenarios_list], alpha=0.85)

for bar, cost in zip(bars, total_costs):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'€{cost:.1f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Total System Cost (€ Billion)')
ax1.set_title('Total System Cost by Scenario', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Right: Cost breakdown
cost_types = ['onshore wind', 'gas', 'nuclear', 'load_shedding']
bottom = np.zeros(len(scenarios_list))

for ct in cost_types:
    vals = [summaries[sc]['cost_breakdown'].get(ct, 0) / 1e9 for sc in scenarios_list]
    ax2.bar([SCENARIO_LABELS[sc] for sc in scenarios_list], vals, bottom=bottom,
            label=ct, color=COLORS[ct], alpha=0.85)
    bottom += vals

ax2.set_ylabel('Cost Component (€ Billion)')
ax2.set_title('Cost Breakdown by Scenario', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_cost_comparison.png'))
plt.close()
print("  Done.")

# ============================
# Figure 8: Demand vs Supply
# ============================
print("Generating Figure 8: Demand vs Supply...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, (sc, label) in enumerate(SCENARIO_LABELS.items()):
    ts = pd.read_csv(os.path.join(OUTPUT_DIR, f'{sc}_timeseries.csv'))
    hours = np.arange(len(ts))
    
    axes[idx].plot(hours, ts['total_demand_mw']/1000, 'k--', linewidth=1.5, label='Demand')
    axes[idx].plot(hours, (ts['total_wind_mw'] + ts['total_gas_mw'] + ts['total_nuclear_mw'])/1000,
                   color='#43A047', linewidth=1.5, label='Generation')
    axes[idx].fill_between(hours, 
                          (ts['total_wind_mw'] + ts['total_gas_mw'] + ts['total_nuclear_mw'])/1000,
                          ts['total_demand_mw']/1000,
                          where=ts['total_demand_mw'] > (ts['total_wind_mw'] + ts['total_gas_mw'] + ts['total_nuclear_mw']),
                          color=COLORS['load_shedding'], alpha=0.5, label='Load Shedding')
    axes[idx].set_title(label, fontweight='bold')
    axes[idx].set_xlabel('Hour')
    axes[idx].set_xlim(0, len(ts)-1)
    axes[idx].grid(True, alpha=0.3)
    if idx == 0:
        axes[idx].set_ylabel('Power (GW)')
    axes[idx].legend(fontsize=9)

plt.suptitle('Demand vs. Supply by Scenario', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig8_demand_vs_supply.png'))
plt.close()
print("  Done.")

print("\nAll figures generated successfully!")
print(f"Figures saved to: {IMG_DIR}")
