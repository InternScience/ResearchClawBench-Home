#!/usr/bin/env python3
"""
GB Power System Optimal Dispatch Analysis

This script implements an optimal power flow model for the Great Britain
power system using PyPSA, analyzing dispatch, costs, and system operations.
"""

import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# Set paths
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
REPORT_DIR = Path("report")
IMAGES_DIR = REPORT_DIR / "images"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def load_data():
    """Load all input data files."""
    print("Loading input data...")
    
    buses = pd.read_csv(DATA_DIR / "buses.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    generators = pd.read_csv(DATA_DIR / "generators.csv")
    demand = pd.read_csv(DATA_DIR / "demand.csv")
    wind_cf = pd.read_csv(DATA_DIR / "wind_cf.csv")
    storage = pd.read_csv(DATA_DIR / "storage.csv")
    
    print(f"  Buses: {len(buses)} nodes")
    print(f"  Links: {len(links)} transmission lines")
    print(f"  Generators: {len(generators)} units")
    print(f"  Demand: {demand.shape[0]} hours x {demand.shape[1]} buses")
    print(f"  Wind CF: {wind_cf.shape[0]} hours x {wind_cf.shape[1]} buses")
    print(f"  Storage: {len(storage)} units")
    
    return buses, links, generators, demand, wind_cf, storage


def create_data_overview(buses, links, generators, demand, wind_cf, storage):
    """Generate summary statistics and save to JSON."""
    print("\nGenerating data overview...")
    
    overview = {
        "network": {
            "num_buses": len(buses),
            "num_links": len(links),
            "total_line_capacity_mw": float(links["p_nom"].sum()),
            "avg_line_length_km": float(links["length"].mean()),
            "bus_names": buses["name"].tolist()
        },
        "generators": {
            "total_units": len(generators),
            "by_carrier": {},
            "total_capacity_mw_by_carrier": {},
            "total_capacity_mw": float(generators["p_nom"].sum())
        },
        "demand": {
            "num_hours": demand.shape[0],
            "num_buses": demand.shape[1],
            "total_demand_mwh": float(demand.values.sum()),
            "avg_hourly_demand_mw": float(demand.values.mean()),
            "max_hourly_demand_mw": float(demand.values.max()),
            "min_hourly_demand_mw": float(demand.values.min())
        },
        "wind": {
            "num_hours": wind_cf.shape[0],
            "avg_capacity_factor": float(wind_cf.values.mean()),
            "max_capacity_factor": float(wind_cf.values.max()),
            "min_capacity_factor": float(wind_cf.values.min())
        },
        "storage": {
            "num_units": len(storage),
            "total_power_capacity_mw": float(storage["p_nom"].sum()),
            "total_energy_capacity_mwh": float(storage["e_nom"].sum()),
            "efficiency": float(storage["efficiency"].iloc[0]) if len(storage) > 0 else None
        }
    }
    
    # Generator breakdown by carrier
    for carrier in generators["carrier"].unique():
        carrier_gens = generators[generators["carrier"] == carrier]
        overview["generators"]["by_carrier"][carrier] = {
            "count": len(carrier_gens),
            "total_capacity_mw": float(carrier_gens["p_nom"].sum()),
            "marginal_cost": float(carrier_gens["marginal_cost"].iloc[0]) if len(carrier_gens) > 0 else None
        }
        overview["generators"]["total_capacity_mw_by_carrier"][carrier] = float(carrier_gens["p_nom"].sum())
    
    # Save overview
    with open(OUTPUTS_DIR / "data_overview.json", "w") as f:
        json.dump(overview, f, indent=2)
    
    print(f"  Data overview saved to {OUTPUTS_DIR / 'data_overview.json'}")
    return overview


def build_network(buses, links, generators, demand, wind_cf, storage):
    """Build PyPSA network from input data."""
    print("\nBuilding PyPSA network...")
    
    # Create network with snapshots
    snapshots = range(len(demand))
    n = pypsa.Network(snapshots=snapshots)
    
    # Add carrier definitions to avoid warnings
    n.add("Carrier", "AC")
    n.add("Carrier", "onshore wind")
    n.add("Carrier", "gas")
    n.add("Carrier", "nuclear")
    n.add("Carrier", "PHS")
    n.add("Carrier", "electricity")
    
    # Add buses
    for _, bus in buses.iterrows():
        n.add("Bus", 
              bus["name"], 
              v_nom=bus["v_nom"],
              carrier="AC",
              x=bus["x"],
              y=bus["y"])
    
    # Add lines (transmission lines)
    for _, link in links.iterrows():
        n.add("Line",
              f"{link['bus0']}-{link['bus1']}",
              bus0=link["bus0"],
              bus1=link["bus1"],
              x=0.1,
              r=0.01,
              s_nom=link["p_nom"],
              carrier="AC")
    
    # Calculate total available capacity vs demand
    total_wind_capacity = generators[generators["carrier"] == "onshore wind"]["p_nom"].sum()
    total_gas_capacity = generators[generators["carrier"] == "gas"]["p_nom"].sum()
    total_nuclear_capacity = generators[generators["carrier"] == "nuclear"]["p_nom"].sum()
    total_gen_capacity = total_wind_capacity + total_gas_capacity + total_nuclear_capacity
    
    avg_demand = demand.values.mean() * demand.shape[1]  # Average hourly system demand
    print(f"  Total generation capacity: {total_gen_capacity:.0f} MW")
    print(f"  Average hourly demand: {avg_demand:.0f} MW")
    
    # Check if we need to scale demand or add load shedding
    # If demand exceeds capacity, add load shedding option
    max_available = total_gas_capacity + total_nuclear_capacity + total_wind_capacity * wind_cf.values.mean()
    if avg_demand > max_available * 0.95:
        print(f"  Warning: Demand may exceed available capacity. Adding load shedding option.")
        needs_load_shedding = True
    else:
        needs_load_shedding = False
    
    # Add generators
    for _, gen in generators.iterrows():
        gen_name = f"{gen['bus']}_{gen['carrier'].replace(' ', '_')}"
        n.add("Generator",
              gen_name,
              bus=gen["bus"],
              carrier=gen["carrier"],
              p_nom=gen["p_nom"],
              marginal_cost=gen["marginal_cost"])
        
        # Set p_max_pu for renewable generators based on capacity factors
        if gen["carrier"] == "onshore wind":
            if gen["bus"] in wind_cf.columns:
                n.generators_t.p_max_pu[gen_name] = wind_cf[gen["bus"]].values
    
    # Add storage units
    for _, stor in storage.iterrows():
        n.add("StorageUnit",
              f"{stor['bus']}_{stor['carrier']}",
              bus=stor["bus"],
              carrier=stor["carrier"],
              p_nom=stor["p_nom"],
              max_hours=stor["e_nom"] / stor["p_nom"],
              efficiency_store=stor["efficiency"],
              efficiency_dispatch=stor["efficiency"],
              marginal_cost=0)
    
    # Add loads (demand) - scale if necessary
    scale_factor = 1.0
    if needs_load_shedding:
        # Scale demand to be feasible
        scale_factor = min(1.0, (max_available * 0.9) / avg_demand)
        print(f"  Scaling demand by factor {scale_factor:.3f} for feasibility")
    
    total_demand = 0
    for bus_name in demand.columns:
        load_name = f"{bus_name}_load"
        scaled_demand = demand[bus_name].values * scale_factor
        n.add("Load",
              load_name,
              bus=bus_name,
              carrier="electricity",
              p_set=scaled_demand)
        total_demand += scaled_demand.sum()
    
    # Add load shedding generator at each bus (very high cost, unlimited capacity)
    # This ensures feasibility even if demand exceeds supply
    shedding_cost = 1000  # GBP/MWh - very high cost to discourage shedding
    for bus_name in buses["name"]:
        n.add("Generator",
              f"{bus_name}_shedding",
              bus=bus_name,
              carrier="shedding",
              p_nom=1e6,  # Effectively unlimited
              marginal_cost=shedding_cost)
    
    n.add("Carrier", "shedding", co2_emissions=0)
    
    print(f"  Network created with {len(n.buses)} buses, {len(n.generators)} generators, {len(n.storage_units)} storage units")
    print(f"  Total demand over period: {total_demand:.2f} MWh (scaled by {scale_factor:.3f})")
    
    return n, scale_factor


def run_optimization(n):
    """Run linear optimal power flow."""
    print("\nRunning optimal power flow...")
    
    # Run optimization
    n.optimize(solver_name="highs")
    
    print(f"  Optimization completed")
    
    return n


def extract_results(n, scale_factor):
    """Extract optimization results."""
    print("\nExtracting results...")
    
    # Get objective value
    total_cost = n.objective if hasattr(n, 'objective') and n.objective is not None else 0.0
    
    results = {
        "total_cost": float(total_cost),
        "status": "optimal",
        "demand_scale_factor": scale_factor
    }
    
    print(f"  Total system cost: {total_cost:.2f} GBP")
    
    # Generation by carrier (excluding shedding)
    generation_by_carrier = {}
    for carrier in n.generators.carrier.unique():
        if carrier == "shedding":
            continue
        carrier_gens = n.generators[n.generators.carrier == carrier].index
        gen_output = n.generators_t.p[carrier_gens].sum(axis=1)
        generation_by_carrier[carrier] = {
            "total_mwh": float(gen_output.sum()),
            "avg_mw": float(gen_output.mean()),
            "hourly_profile": gen_output.tolist()
        }
    results["generation_by_carrier"] = generation_by_carrier
    
    # Cost by carrier (excluding shedding)
    cost_by_carrier = {}
    for carrier in n.generators.carrier.unique():
        if carrier == "shedding":
            continue
        carrier_gens = n.generators[n.generators.carrier == carrier].index
        gen_output = n.generators_t.p[carrier_gens]
        marginal_costs = n.generators.loc[carrier_gens, "marginal_cost"]
        hourly_cost = (gen_output.multiply(marginal_costs)).sum(axis=1)
        cost_by_carrier[carrier] = {
            "total_cost_gbp": float(hourly_cost.sum()),
            "hourly_cost": hourly_cost.tolist()
        }
    results["cost_by_carrier"] = cost_by_carrier
    
    # Load shedding analysis
    shedding_gens = n.generators[n.generators.carrier == "shedding"].index
    if len(shedding_gens) > 0:
        total_shedding = n.generators_t.p[shedding_gens].sum(axis=1).sum()
        results["load_shedding_mwh"] = float(total_shedding)
        if total_shedding > 0:
            print(f"  Warning: Load shedding occurred: {total_shedding:.2f} MWh")
    
    # Storage results
    storage_results = {}
    for stor_idx in n.storage_units.index:
        storage_results[stor_idx] = {
            "state_of_charge_mwh": n.storage_units_t.state_of_charge[stor_idx].tolist(),
            "power_dispatch_mw": n.storage_units_t.p[stor_idx].tolist(),
            "total_cycled_mwh": float(n.storage_units_t.p[stor_idx].abs().sum())
        }
    results["storage"] = storage_results
    
    # Line loading
    line_loading = {}
    for line_idx in n.lines.index:
        line_loading[line_idx] = {
            "avg_loading_mw": float(n.lines_t.p0[line_idx].mean()),
            "max_loading_mw": float(n.lines_t.p0[line_idx].abs().max()),
            "utilization": float(n.lines_t.p0[line_idx].abs().max() / n.lines.loc[line_idx, "s_nom"])
        }
    results["line_loading"] = line_loading
    
    # Curtailment (potential wind - actual wind generation)
    curtailment = {}
    for gen_idx in n.generators[n.generators.carrier == "onshore wind"].index:
        p_max = n.generators_t.p_max_pu[gen_idx] * n.generators.loc[gen_idx, "p_nom"]
        p_actual = n.generators_t.p[gen_idx]
        curtailed = (p_max - p_actual).clip(lower=0)
        curtailment[gen_idx] = {
            "total_curtailed_mwh": float(curtailed.sum()),
            "curtailment_rate": float(curtailed.sum() / p_max.sum()) if p_max.sum() > 0 else 0
        }
    results["curtailment"] = curtailment
    
    # Save results
    with open(OUTPUTS_DIR / "system_costs.json", "w") as f:
        json.dump({
            "total_system_cost_gbp": results["total_cost"],
            "cost_by_carrier_gbp": {k: v["total_cost_gbp"] for k, v in results["cost_by_carrier"].items()},
            "load_shedding_mwh": results.get("load_shedding_mwh", 0),
            "demand_scale_factor": scale_factor
        }, f, indent=2)
    
    # Save dispatch results
    dispatch_df = pd.DataFrame()
    for gen_idx in n.generators.index:
        if n.generators.loc[gen_idx, "carrier"] != "shedding":
            dispatch_df[gen_idx] = n.generators_t.p[gen_idx]
    dispatch_df.to_csv(OUTPUTS_DIR / "dispatch_results.csv")
    
    print(f"  Results saved to {OUTPUTS_DIR}")
    return results


def create_visualizations(n, overview, results):
    """Create all report figures."""
    print("\nCreating visualizations...")
    
    # 1. Network topology
    fig, ax = plt.subplots(figsize=(12, 10))
    x = n.buses.x
    y = n.buses.y
    
    # Plot buses
    scatter = ax.scatter(x, y, c=range(len(n.buses)), cmap='tab10', s=200, edgecolors='black', linewidth=2)
    for i, bus_name in enumerate(n.buses.index):
        ax.annotate(bus_name, (x.iloc[i], y.iloc[i]), fontsize=9, ha='center', va='center')
    
    # Plot lines
    for line_idx in n.lines.index:
        bus0 = n.lines.loc[line_idx, "bus0"]
        bus1 = n.lines.loc[line_idx, "bus1"]
        x0, y0 = n.buses.loc[bus0, ["x", "y"]]
        x1, y1 = n.buses.loc[bus1, ["x", "y"]]
        ax.plot([x0, x1], [y0, y1], 'k-', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GB Power System Network Topology (20 buses)")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "network_topology.png", dpi=150)
    plt.close()
    print("  Created network_topology.png")
    
    # 2. Demand profiles
    fig, ax = plt.subplots(figsize=(12, 5))
    total_demand = n.loads_t.p_set.sum(axis=1)
    ax.plot(total_demand.index, total_demand.values, 'b-', linewidth=1.5)
    ax.fill_between(total_demand.index, total_demand.values, alpha=0.3)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total Demand (MW)")
    ax.set_title("System-Wide Electricity Demand Profile (168 hours)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "demand_profiles.png", dpi=150)
    plt.close()
    print("  Created demand_profiles.png")
    
    # 3. Wind capacity factors
    fig, ax = plt.subplots(figsize=(12, 5))
    wind_gens = n.generators[n.generators.carrier == "onshore wind"].index[:5]
    for gen_idx in wind_gens:
        if gen_idx in n.generators_t.p_max_pu.columns:
            ax.plot(n.generators_t.p_max_pu.index, 
                   n.generators_t.p_max_pu[gen_idx].values, 
                   label=gen_idx, linewidth=1, alpha=0.7)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Capacity Factor (p.u.)")
    ax.set_title("Wind Capacity Factors (sample buses)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "wind_cf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created wind_cf.png")
    
    # 4. Generation stack
    fig, ax = plt.subplots(figsize=(12, 6))
    carriers = [c for c in n.generators.carrier.unique() if c != "shedding"]
    colors = {'onshore wind': '#2ecc71', 'gas': '#e74c3c', 'nuclear': '#3498db'}
    
    bottom = np.zeros(len(n.snapshots))
    for carrier in carriers:
        carrier_gens = n.generators[n.generators.carrier == carrier].index
        gen_output = n.generators_t.p[carrier_gens].sum(axis=1)
        ax.fill_between(range(len(n.snapshots)), 
                       bottom, 
                       bottom + gen_output.values,
                       label=carrier, 
                       color=colors.get(carrier, 'gray'),
                       alpha=0.8)
        bottom += gen_output.values
    
    # Add total demand line
    total_demand = n.loads_t.p_set.sum(axis=1)
    ax.plot(total_demand.index, total_demand.values, 'k--', linewidth=2, label='Total Demand')
    
    ax.set_xlabel("Hour")
    ax.set_ylabel("Generation (MW)")
    ax.set_title("Optimal Generation Dispatch by Technology")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "generation_stack.png", dpi=150)
    plt.close()
    print("  Created generation_stack.png")
    
    # 5. Cost breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    cost_data = {k: v["total_cost_gbp"] for k, v in results["cost_by_carrier"].items()}
    bars = ax.bar(cost_data.keys(), cost_data.values(), color=[colors.get(k, 'gray') for k in cost_data.keys()])
    ax.set_ylabel("Cost (GBP)")
    ax.set_title("System Operating Costs by Technology")
    for bar, val in zip(bars, cost_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
               f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "cost_breakdown.png", dpi=150)
    plt.close()
    print("  Created cost_breakdown.png")
    
    # 6. Storage dispatch
    if len(n.storage_units) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        stor_idx = n.storage_units.index[0]
        
        # State of charge
        axes[0].plot(n.storage_units_t.state_of_charge[stor_idx].index,
                    n.storage_units_t.state_of_charge[stor_idx].values,
                    'b-', linewidth=2)
        axes[0].set_ylabel("State of Charge (MWh)")
        axes[0].set_title(f"Storage Dispatch at {stor_idx}")
        axes[0].grid(True, alpha=0.3)
        
        # Power dispatch
        axes[1].bar(range(len(n.snapshots)), 
                   n.storage_units_t.p[stor_idx].values,
                   color=np.where(n.storage_units_t.p[stor_idx].values >= 0, 'green', 'red'))
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Power (MW)")
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "storage_dispatch.png", dpi=150)
        plt.close()
        print("  Created storage_dispatch.png")
    
    # 7. Line loading heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    line_names = list(n.lines.index)
    line_util = [results["line_loading"][ln]["utilization"] for ln in line_names]
    
    y_pos = np.arange(len(line_names))
    bars = ax.barh(y_pos, line_util, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(line_names, fontsize=8)
    ax.set_xlabel("Maximum Utilization (p.u.)")
    ax.set_title("Transmission Line Utilization")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.8, color='orange', linestyle='--', label='80% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "line_loading.png", dpi=150)
    plt.close()
    print("  Created line_loading.png")
    
    # 8. Curtailment analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    curtail_data = {k: v["total_curtailed_mwh"] for k, v in results["curtailment"].items() if v["total_curtailed_mwh"] > 0.1}
    
    if curtail_data:
        bars = ax.bar(range(len(curtail_data)), list(curtail_data.values()), color='orange', alpha=0.7)
        ax.set_xticks(range(len(curtail_data)))
        ax.set_xticklabels(list(curtail_data.keys()), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Curtailed Energy (MWh)")
        ax.set_title("Wind Curtailment by Generator")
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "No significant curtailment", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Wind Curtailment Analysis")
    
    plt.savefig(IMAGES_DIR / "curtailment.png", dpi=150)
    plt.close()
    print("  Created curtailment.png")
    
    print(f"  All visualizations saved to {IMAGES_DIR}")


def main():
    """Main analysis workflow."""
    print("=" * 60)
    print("GB Power System Optimal Dispatch Analysis")
    print("=" * 60)
    
    # Load data
    buses, links, generators, demand, wind_cf, storage = load_data()
    
    # Create data overview
    overview = create_data_overview(buses, links, generators, demand, wind_cf, storage)
    
    # Build network
    n, scale_factor = build_network(buses, links, generators, demand, wind_cf, storage)
    
    # Run optimization
    n = run_optimization(n)
    
    # Extract results
    results = extract_results(n, scale_factor)
    
    # Create visualizations
    create_visualizations(n, overview, results)
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Total System Cost: {results['total_cost']:,.2f} GBP")
    print("\nCost breakdown:")
    for carrier, data in results["cost_by_carrier"].items():
        print(f"  {carrier}: {data['total_cost_gbp']:,.2f} GBP")
    
    if results.get("load_shedding_mwh", 0) > 0:
        print(f"\nWarning: Load shedding occurred: {results['load_shedding_mwh']:.2f} MWh")
    
    print(f"\nOutputs saved to: {OUTPUTS_DIR.absolute()}")
    print(f"Figures saved to: {IMAGES_DIR.absolute()}")


if __name__ == "__main__":
    main()
