#!/usr/bin/env python3
"""
Generate all figures for the GB Power System report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_001_20260416_184602'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
IMG_DIR = os.path.join(BASE_DIR, 'report', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# Color scheme
COLORS = {
    'onshore wind': '#2196F3',
    'gas': '#FF9800',
    'nuclear': '#9C27B0',
    'load_shedding': '#F44336',
    'storage': '#4CAF50',
    'demand': '#333333',
    'curtailment': '#90CAF9',
}

def load_data():
    buses = pd.read_csv(os.path.join(DATA_DIR, 'buses.csv'))
    links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))
    generators = pd.read_csv(os.path.join(DATA_DIR, 'generators.csv'))
    demand = pd.read_csv(os.path.join(DATA_DIR, 'demand.csv'))
    wind_cf = pd.read_csv(os.path.join(DATA_DIR, 'wind_cf.csv'))
    storage = pd.read_csv(os.path.join(DATA_DIR, 'storage.csv'))
    return buses, links, generators, demand, wind_cf, storage


# ============================================================
# FIGURE 1: Network Topology Map
# ============================================================
def plot_network_topology(buses, links, generators, storage):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    for _, row in links.iterrows():
        b0 = buses[buses.name == row['bus0']].iloc[0]
        b1 = buses[buses.name == row['bus1']].iloc[0]
        lw = 3 if row['p_nom'] >= 5000 else 1.5
        color = '#1565C0' if row['p_nom'] >= 5000 else '#E65100'
        ax.plot([b0.x, b1.x], [b0.y, b1.y], color=color, linewidth=lw, alpha=0.7, zorder=1)
    
    gen_by_bus = generators.groupby('bus')['p_nom'].sum()
    
    for _, row in buses.iterrows():
        bus_name = row['name']
        total_cap = gen_by_bus.get(bus_name, 0)
        has_nuclear = bus_name in generators[generators.carrier == 'nuclear'].bus.values
        has_storage = bus_name in storage.bus.values
        
        size = max(50, min(500, total_cap / 30))
        
        if has_nuclear:
            color = COLORS['nuclear']
            marker = 's'
        elif has_storage:
            color = COLORS['storage']
            marker = 'D'
        else:
            color = COLORS['onshore wind']
            marker = 'o'
        
        ax.scatter(row.x, row.y, s=size, c=color, marker=marker, edgecolors='black', 
                   linewidths=1, zorder=3)
        ax.annotate(bus_name, (row.x, row.y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1565C0', linewidth=3, label='High capacity links (5 GW)'),
        Line2D([0], [0], color='#E65100', linewidth=1.5, label='Cross links (1.5 GW)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['onshore wind'], 
               markersize=10, markeredgecolor='black', label='Wind + Gas bus'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['nuclear'], 
               markersize=10, markeredgecolor='black', label='Nuclear bus'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['storage'], 
               markersize=10, markeredgecolor='black', label='Storage bus'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GB Power System Network Topology (20-Bus Model)')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(IMG_DIR, 'network_topology.png'))
    plt.close()
    print("Saved: network_topology.png")


# ============================================================
# FIGURE 2: Demand Profile
# ============================================================
def plot_demand_profile(demand):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    hours = np.arange(len(demand))
    total_demand = demand.sum(axis=1)
    
    ax1.fill_between(hours, total_demand / 1000, alpha=0.3, color=COLORS['demand'])
    ax1.plot(hours, total_demand / 1000, color=COLORS['demand'], linewidth=2)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Total Demand (GW)')
    ax1.set_title('Total System Demand Over One Week')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 167)
    
    for d in range(7):
        ax1.axvline(x=d*24, color='gray', linestyle='--', alpha=0.3)
        ax1.text(d*24 + 12, ax1.get_ylim()[1] * 0.95, f'Day {d+1}', ha='center', fontsize=9, alpha=0.6)
    
    north_buses = [f'Bus{i}' for i in range(1, 6)]
    south_buses = [f'Bus{i}' for i in range(6, 21)]
    
    north_demand = demand[north_buses].sum(axis=1) / 1000
    south_demand = demand[south_buses].sum(axis=1) / 1000
    
    ax2.fill_between(hours, south_demand, alpha=0.3, color='#E65100', label='South (Bus6-20)')
    ax2.plot(hours, south_demand, color='#E65100', linewidth=2)
    ax2.fill_between(hours, north_demand, alpha=0.3, color='#1565C0', label='North (Bus1-5)')
    ax2.plot(hours, north_demand, color='#1565C0', linewidth=2)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Demand (GW)')
    ax2.set_title('Demand by Region: North vs South')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 167)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'demand_profile.png'))
    plt.close()
    print("Saved: demand_profile.png")


# ============================================================
# FIGURE 3: Wind Capacity Factors
# ============================================================
def plot_wind_capacity_factors(wind_cf):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    sns.heatmap(wind_cf.T, ax=ax1, cmap='YlGnBu', vmin=0, vmax=1,
                xticklabels=12, yticklabels=True, cbar_kws={'label': 'Capacity Factor'})
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Bus')
    ax1.set_title('Wind Capacity Factors by Bus and Hour')
    
    hours = np.arange(len(wind_cf))
    for bus in ['Bus1', 'Bus5', 'Bus10', 'Bus15', 'Bus20']:
        ax2.plot(hours, wind_cf[bus], label=bus, alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Capacity Factor')
    ax2.set_title('Wind Capacity Factor Time Series (Selected Buses)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 167)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'wind_capacity_factors.png'))
    plt.close()
    print("Saved: wind_capacity_factors.png")


# ============================================================
# FIGURE 4: Generation Dispatch (Base Case) - Stacked Area
# ============================================================
def plot_generation_dispatch():
    gen_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_generation.csv'), index_col=0, parse_dates=True)
    demand_df = pd.read_csv(os.path.join(DATA_DIR, 'demand.csv'))
    
    hours = np.arange(len(gen_df))
    total_demand = demand_df.sum(axis=1).values / 1000
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    carriers = ['nuclear', 'onshore wind', 'gas', 'load_shedding']
    carrier_labels = ['Nuclear', 'Onshore Wind', 'Gas (CCGT)', 'Load Shedding (VOLL)']
    carrier_colors = [COLORS['nuclear'], COLORS['onshore wind'], COLORS['gas'], COLORS['load_shedding']]
    
    bottom = np.zeros(len(gen_df))
    for carrier, label, color in zip(carriers, carrier_labels, carrier_colors):
        if carrier in gen_df.columns:
            values = gen_df[carrier].values / 1000
            ax.fill_between(hours, bottom, bottom + values, alpha=0.7, color=color, label=label)
            bottom += values
    
    ax.plot(hours, total_demand, color='black', linewidth=2, linestyle='--', label='Total Demand')
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power (GW)')
    ax.set_title('Base Case: Hourly Generation Dispatch')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 167)
    
    for d in range(7):
        ax.axvline(x=d*24, color='gray', linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(IMG_DIR, 'generation_dispatch.png'))
    plt.close()
    print("Saved: generation_dispatch.png")


# ============================================================
# FIGURE 5: Scenario Comparison Bar Chart
# ============================================================
def plot_scenario_comparison():
    summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenario_summary.csv'))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = summary['Scenario'].values
    x = np.arange(len(scenarios))
    width = 0.6
    
    # (a) Generation mix stacked bar
    ax = axes[0, 0]
    wind = summary['Wind Generation (MWh)'].values / 1e6
    gas = summary['Gas Generation (MWh)'].values / 1e6
    nuclear = summary['Nuclear Generation (MWh)'].values / 1e6
    shed = summary['Load Shedding (MWh)'].values / 1e6
    
    ax.bar(x, nuclear, width, label='Nuclear', color=COLORS['nuclear'])
    ax.bar(x, wind, width, bottom=nuclear, label='Wind', color=COLORS['onshore wind'])
    ax.bar(x, gas, width, bottom=nuclear+wind, label='Gas', color=COLORS['gas'])
    ax.bar(x, shed, width, bottom=nuclear+wind+gas, label='Load Shedding', color=COLORS['load_shedding'])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Energy (TWh)')
    ax.set_title('(a) Generation Mix by Scenario')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # (b) Total system cost
    ax = axes[0, 1]
    costs = summary['Total Cost (£)'].values / 1e9
    bars = ax.bar(x, costs, width, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4'])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Total Cost (£ Billion)')
    ax.set_title('(b) Total System Cost by Scenario')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'£{cost:.1f}B', ha='center', va='bottom', fontsize=8)
    
    # (c) Wind curtailment rate
    ax = axes[1, 0]
    curt = summary['Wind Curtailment (%)'].values
    bars = ax.bar(x, curt, width, color=COLORS['curtailment'], edgecolor='#1565C0')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Wind Curtailment (%)')
    ax.set_title('(c) Wind Curtailment Rate by Scenario')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, c in zip(bars, curt):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{c:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # (d) Load shedding percentage
    ax = axes[1, 1]
    ls = summary['Load Shedding (%)'].values
    bars = ax.bar(x, ls, width, color=COLORS['load_shedding'], alpha=0.7, edgecolor='#B71C1C')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Load Shedding (%)')
    ax.set_title('(d) Load Shedding by Scenario')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, l in zip(bars, ls):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{l:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'scenario_comparison.png'))
    plt.close()
    print("Saved: scenario_comparison.png")


# ============================================================
# FIGURE 6: Storage Operation
# ============================================================
def plot_storage_operation():
    try:
        soc = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_storage_soc.csv'), index_col=0, parse_dates=True)
        dispatch = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_storage_dispatch.csv'), index_col=0, parse_dates=True)
    except:
        print("No storage data available, skipping storage plot")
        return
    
    hours = np.arange(len(soc))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    for col in soc.columns:
        bus_name = col.split('_')[0]
        ax1.plot(hours, soc[col], linewidth=2, label=f'{bus_name} PHS')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('State of Charge (MWh)')
    ax1.set_title('Base Case: Pumped Hydro Storage - State of Charge')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 167)
    
    for col in dispatch.columns:
        bus_name = col.split('_')[0]
        ax2.plot(hours, dispatch[col], linewidth=2, label=f'{bus_name} PHS')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Power (MW) [+discharge / -charge]')
    ax2.set_title('Base Case: Pumped Hydro Storage - Dispatch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 167)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'storage_operation.png'))
    plt.close()
    print("Saved: storage_operation.png")


# ============================================================
# FIGURE 7: Nodal Prices
# ============================================================
def plot_nodal_prices():
    prices = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_bus_prices.csv'), index_col=0, parse_dates=True)
    
    hours = np.arange(len(prices))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    selected = ['Bus1', 'Bus5', 'Bus10', 'Bus15', 'Bus20']
    for bus in selected:
        if bus in prices.columns:
            ax1.plot(hours, prices[bus], label=bus, alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Marginal Price (£/MWh)')
    ax1.set_title('Base Case: Nodal Marginal Prices (Selected Buses)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 167)
    
    avg_prices = prices.mean()
    colors_bar = ['#1565C0' if int(b.replace('Bus','')) <= 5 else '#E65100' for b in avg_prices.index]
    ax2.bar(range(len(avg_prices)), avg_prices.values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(avg_prices)))
    ax2.set_xticklabels(avg_prices.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Average Marginal Price (£/MWh)')
    ax2.set_title('Base Case: Average Nodal Prices (Blue=North, Orange=South)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'nodal_prices.png'))
    plt.close()
    print("Saved: nodal_prices.png")


# ============================================================
# FIGURE 8: Transmission Flow Analysis
# ============================================================
def plot_transmission_flows():
    flows = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_link_flows.csv'), index_col=0, parse_dates=True)
    links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'))
    
    hours = np.arange(len(flows))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    cross_link_names = []
    for idx, row in links.iterrows():
        link_name = f"Link_{row['bus0']}_{row['bus1']}"
        if row['p_nom'] == 1500:
            cross_link_names.append(link_name)
    
    for link in cross_link_names:
        if link in flows.columns:
            short_name = link.replace('Link_', '')
            ax1.plot(hours, flows[link], label=short_name, linewidth=1.5, alpha=0.8)
    ax1.axhline(y=1500, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Capacity limit')
    ax1.axhline(y=-1500, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power Flow (MW)')
    ax1.set_title('Base Case: Cross-Link Flows (North-South, 1.5 GW capacity)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 167)
    
    link_caps = {}
    for idx, row in links.iterrows():
        link_name = f"Link_{row['bus0']}_{row['bus1']}"
        link_caps[link_name] = row['p_nom']
    
    utilizations = []
    link_labels = []
    bar_colors = []
    for link in flows.columns:
        if link in link_caps:
            util = flows[link].abs().mean() / link_caps[link] * 100
            utilizations.append(util)
            link_labels.append(link.replace('Link_', ''))
            bar_colors.append('#E65100' if link_caps[link] == 1500 else '#1565C0')
    
    ax2.barh(range(len(utilizations)), utilizations, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(link_labels)))
    ax2.set_yticklabels(link_labels, fontsize=8)
    ax2.set_xlabel('Average Utilization (%)')
    ax2.set_title('Base Case: Average Link Utilization (Orange=Cross-links, Blue=Sequential)')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=100, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'transmission_flows.png'))
    plt.close()
    print("Saved: transmission_flows.png")


# ============================================================
# FIGURE 9: Wind Curtailment Analysis
# ============================================================
def plot_wind_curtailment():
    gen_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'base_case_generation.csv'), index_col=0, parse_dates=True)
    wind_cf = pd.read_csv(os.path.join(DATA_DIR, 'wind_cf.csv'))
    generators = pd.read_csv(os.path.join(DATA_DIR, 'generators.csv'))
    
    hours = np.arange(len(gen_df))
    
    wind_gens = generators[generators.carrier == 'onshore wind']
    wind_available = np.zeros(168)
    for _, row in wind_gens.iterrows():
        wind_available += row['p_nom'] * wind_cf[row['bus']].values
    
    wind_dispatched = gen_df['onshore wind'].values if 'onshore wind' in gen_df.columns else np.zeros(168)
    wind_curtailed = wind_available - wind_dispatched
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.fill_between(hours, wind_available / 1000, alpha=0.3, color=COLORS['curtailment'], label='Curtailed Wind')
    ax1.fill_between(hours, wind_dispatched / 1000, alpha=0.7, color=COLORS['onshore wind'], label='Dispatched Wind')
    ax1.plot(hours, wind_available / 1000, color='#1565C0', linewidth=1.5, linestyle='--', label='Available Wind')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (GW)')
    ax1.set_title('Base Case: Wind Generation - Available vs Dispatched')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 167)
    
    curt_rate = np.where(wind_available > 0, wind_curtailed / wind_available * 100, 0)
    ax2.fill_between(hours, curt_rate, alpha=0.5, color=COLORS['load_shedding'])
    ax2.plot(hours, curt_rate, color='#B71C1C', linewidth=1.5)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Curtailment Rate (%)')
    ax2.set_title('Base Case: Hourly Wind Curtailment Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 167)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'wind_curtailment.png'))
    plt.close()
    print("Saved: wind_curtailment.png")


# ============================================================
# FIGURE 10: Cost Breakdown
# ============================================================
def plot_cost_breakdown():
    with open(os.path.join(OUTPUT_DIR, 'all_scenarios_summary.json')) as f:
        data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    base = data['Base Case']['cost_by_carrier']
    labels = []
    sizes = []
    colors = []
    for carrier, cost in base.items():
        if cost > 0:
            labels.append(carrier.replace('_', ' ').title())
            sizes.append(cost)
            colors.append(COLORS.get(carrier, '#999999'))
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.85)
    for text in autotexts:
        text.set_fontsize(9)
    ax1.set_title('Base Case: Cost Breakdown by Source')
    
    scenarios = list(data.keys())
    carriers_to_plot = ['onshore wind', 'gas', 'nuclear']
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, carrier in enumerate(carriers_to_plot):
        costs = [data[s]['cost_by_carrier'].get(carrier, 0) / 1e6 for s in scenarios]
        ax2.bar(x + i * width, costs, width, label=carrier.replace('_', ' ').title(),
                color=COLORS.get(carrier, '#999999'))
    
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Generation Cost (£ Million)')
    ax2.set_title('Generation Cost by Carrier (Excl. Load Shedding)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'cost_breakdown.png'))
    plt.close()
    print("Saved: cost_breakdown.png")


# ============================================================
# FIGURE 11: Generation Capacity vs Demand by Bus
# ============================================================
def plot_capacity_vs_demand(buses, generators, demand, storage):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    bus_names = buses['name'].values
    x = np.arange(len(bus_names))
    width = 0.35
    
    peak_demand = demand.max().values / 1000
    
    wind_cap = []
    gas_cap = []
    nuc_cap = []
    stor_cap = []
    
    for bus in bus_names:
        wind_cap.append(generators[(generators.bus == bus) & (generators.carrier == 'onshore wind')].p_nom.sum() / 1000)
        gas_cap.append(generators[(generators.bus == bus) & (generators.carrier == 'gas')].p_nom.sum() / 1000)
        nuc_cap.append(generators[(generators.bus == bus) & (generators.carrier == 'nuclear')].p_nom.sum() / 1000)
        stor_cap.append(storage[storage.bus == bus].p_nom.sum() / 1000)
    
    wind_cap = np.array(wind_cap)
    gas_cap = np.array(gas_cap)
    nuc_cap = np.array(nuc_cap)
    stor_cap = np.array(stor_cap)
    
    ax.bar(x - width/2, nuc_cap, width, label='Nuclear', color=COLORS['nuclear'])
    ax.bar(x - width/2, gas_cap, width, bottom=nuc_cap, label='Gas', color=COLORS['gas'])
    ax.bar(x - width/2, wind_cap, width, bottom=nuc_cap+gas_cap, label='Wind', color=COLORS['onshore wind'])
    ax.bar(x - width/2, stor_cap, width, bottom=nuc_cap+gas_cap+wind_cap, label='Storage', color=COLORS['storage'])
    
    ax.bar(x + width/2, peak_demand, width, label='Peak Demand', color=COLORS['demand'], alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bus_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Capacity / Peak Demand (GW)')
    ax.set_title('Generation Capacity vs Peak Demand by Bus')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'capacity_vs_demand.png'))
    plt.close()
    print("Saved: capacity_vs_demand.png")


# ============================================================
# FIGURE 12: Transmission Impact Analysis
# ============================================================
def plot_transmission_impact():
    with open(os.path.join(OUTPUT_DIR, 'all_scenarios_summary.json')) as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    scenarios = ['Constrained\nTransmission\n(0.5x)', 'Base Case\n(1.0x)', 'Enhanced\nTransmission\n(2.0x)']
    costs = [data['Constrained Transmission']['total_cost'] / 1e9,
             data['Base Case']['total_cost'] / 1e9,
             data['Enhanced Transmission']['total_cost'] / 1e9]
    shed_pct = [data['Constrained Transmission']['load_shed_pct'],
                data['Base Case']['load_shed_pct'],
                data['Enhanced Transmission']['load_shed_pct']]
    curt_pct = [data['Constrained Transmission']['curtailment_rate'],
                data['Base Case']['curtailment_rate'],
                data['Enhanced Transmission']['curtailment_rate']]
    
    x = np.arange(len(scenarios))
    
    bars = ax.bar(x - 0.15, costs, 0.3, label='Total Cost (£B)', color='#2196F3', alpha=0.8)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'£{cost:.1f}B', ha='center', va='bottom', fontsize=9)
    
    ax2 = ax.twinx()
    ax2.plot(x, shed_pct, 'o-', color=COLORS['load_shedding'], linewidth=2.5, markersize=10, label='Load Shedding (%)')
    ax2.plot(x, curt_pct, 's-', color='#1565C0', linewidth=2.5, markersize=10, label='Wind Curtailment (%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel('Total Cost (£ Billion)', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Impact of Transmission Capacity on System Performance', fontsize=14)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'transmission_impact.png'))
    plt.close()
    print("Saved: transmission_impact.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading data...")
    buses, links, generators, demand, wind_cf, storage = load_data()
    
    print("\nGenerating figures...")
    
    plot_network_topology(buses, links, generators, storage)
    plot_demand_profile(demand)
    plot_wind_capacity_factors(wind_cf)
    plot_generation_dispatch()
    plot_scenario_comparison()
    plot_storage_operation()
    plot_nodal_prices()
    plot_transmission_flows()
    plot_wind_curtailment()
    plot_cost_breakdown()
    plot_capacity_vs_demand(buses, generators, demand, storage)
    plot_transmission_impact()
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
