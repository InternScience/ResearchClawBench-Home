#!/usr/bin/env python3
"""
Main analysis script for African Green Hydrogen LCOH Model

This script runs the complete analysis and generates all outputs and figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import json
import os

# Add code directory to path
import sys
sys.path.insert(0, 'code')

from lcoh_model import (
    TechnoEconomicParams, 
    FINANCING_SCENARIOS, 
    run_scenario_analysis
)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 60)
print("AFRICAN GREEN HYDROGEN TO EUROPE - LCOH ANALYSIS")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

df = pd.read_csv('data/hex_final_NA_min.csv')
print(f"  Loaded {len(df)} hexagon sites from Africa")

gdf_countries = gpd.read_file('data/africa_map/ne_10m_admin_0_countries.shp')
africa_countries = gdf_countries[gdf_countries['CONTINENT'] == 'Africa'].copy()
print(f"  Loaded {len(africa_countries)} African country boundaries")

# ============================================================================
# 2. RUN SCENARIO ANALYSIS
# ============================================================================
print("\n[2/6] Running LCOH scenario analysis...")

tech_params = TechnoEconomicParams()
results_df = run_scenario_analysis(df, tech_params, FINANCING_SCENARIOS)

# Save results
results_df.to_csv('outputs/lcoh_results.csv', index=False)
print(f"  Saved results to outputs/lcoh_results.csv")
print(f"  Scenarios: {results_df['scenario'].unique().tolist()}")

# Summary statistics
print("\n  LCOH Production Cost Summary (EUR/kg H2):")
for scenario in results_df['scenario'].unique():
    scen_data = results_df[results_df['scenario'] == scenario]
    print(f"    {scenario:20s}: min={scen_data['lcoh_production'].min():.2f}, "
          f"mean={scen_data['lcoh_production'].mean():.2f}, "
          f"max={scen_data['lcoh_production'].max():.2f}")

print("\n  Total Delivered Cost Summary (EUR/kg H2):")
for scenario in results_df['scenario'].unique():
    scen_data = results_df[results_df['scenario'] == scenario]
    print(f"    {scenario:20s}: min={scen_data['total_delivered_cost'].min():.2f}, "
          f"mean={scen_data['total_delivered_cost'].mean():.2f}, "
          f"max={scen_data['total_delivered_cost'].max():.2f}")

# ============================================================================
# 3. IDENTIFY LEAST-COST LOCATIONS
# ============================================================================
print("\n[3/6] Identifying least-cost locations...")

baseline_results = results_df[results_df['scenario'] == 'baseline_wacc'].copy()
baseline_results = baseline_results.sort_values('total_delivered_cost')

top_5_locations = baseline_results.head(5)[['hex_id', 'lat', 'lon', 'lcoh_production', 
                                              'nh3_conversion_cost', 'shipping_cost',
                                              'reconversion_cost', 'total_delivered_cost',
                                              'capacity_factor']].copy()
top_5_locations.to_csv('outputs/least_cost_locations.csv', index=False)
print("  Top 5 least-cost locations saved to outputs/least_cost_locations.csv")
print("\n  Top 5 Least-Cost Locations (Baseline WACC):")
for idx, row in top_5_locations.iterrows():
    print(f"    {row['hex_id']}: {row['total_delivered_cost']:.2f} EUR/kg "
          f"(prod={row['lcoh_production']:.2f}, ship={row['shipping_cost']:.2f})")

# ============================================================================
# 4. GENERATE FIGURES
# ============================================================================
print("\n[4/6] Generating figures...")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure 1: Data Overview - PV and Wind Potential
print("  Creating data overview plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PV potential distribution
axes[0].hist(df['theo_pv'], bins=15, edgecolor='black', alpha=0.7, color='orange')
axes[0].set_xlabel('Theoretical PV Potential (normalized)')
axes[0].set_ylabel('Number of Sites')
axes[0].set_title('PV Potential Distribution')
axes[0].axvline(df['theo_pv'].mean(), color='red', linestyle='--', 
                label=f"Mean: {df['theo_pv'].mean():.2f}")
axes[0].legend()

# Wind potential distribution
axes[1].hist(df['theo_wind'], bins=15, edgecolor='black', alpha=0.7, color='blue')
axes[1].set_xlabel('Theoretical Wind Potential (normalized)')
axes[1].set_ylabel('Number of Sites')
axes[1].set_title('Wind Potential Distribution')
axes[1].axvline(df['theo_wind'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['theo_wind'].mean():.2f}")
axes[1].legend()

# Ocean distance distribution
axes[2].hist(df['ocean_dist_km'], bins=15, edgecolor='black', alpha=0.7, color='green')
axes[2].set_xlabel('Distance to Ocean (km)')
axes[2].set_ylabel('Number of Sites')
axes[2].set_title('Distance to Ocean Distribution')
axes[2].axvline(df['ocean_dist_km'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['ocean_dist_km'].mean():.0f} km")
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/data_overview.png")

# Figure 2: LCOH Production Map
print("  Creating LCOH production map...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot African countries
africa_countries.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

# Create scatter plot of LCOH at each hexagon
scatter = ax.scatter(
    baseline_results['lon'], 
    baseline_results['lat'],
    c=baseline_results['lcoh_production'],
    cmap='RdYlGn_r',  # Red-Yellow-Green reversed (green = low cost)
    s=150,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8
)

cbar = plt.colorbar(scatter, ax=ax, label='LCOH Production (EUR/kg H2)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Levelized Cost of Hydrogen Production Across African Sites (2030, Baseline WACC)')
ax.set_xlim(-20, 35)
ax.set_ylim(-35, 15)

plt.tight_layout()
plt.savefig('report/images/lcoh_production_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/lcoh_production_map.png")

# Figure 3: Total Delivered Cost Map
print("  Creating delivered cost map...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot African countries
africa_countries.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

# Create scatter plot of delivered cost
scatter = ax.scatter(
    baseline_results['lon'], 
    baseline_results['lat'],
    c=baseline_results['total_delivered_cost'],
    cmap='RdYlGn_r',
    s=150,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8
)

cbar = plt.colorbar(scatter, ax=ax, label='Delivered Cost to Europe (EUR/kg H2)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Total Delivered Cost of African H2 to Europe via Ammonia (2030, Baseline WACC)')
ax.set_xlim(-20, 35)
ax.set_ylim(-35, 15)

plt.tight_layout()
plt.savefig('report/images/delivered_cost_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/delivered_cost_map.png")

# Figure 4: Scenario Comparison
print("  Creating scenario comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data for plotting
scenario_order = ['baseline_wacc', 'de_risked', 'high_interest_rate', 'europe_production']
scenario_labels = ['Baseline\n(8% WACC)', 'De-risked\n(5% WACC)', 
                   'High Interest\n(12% WACC)', 'Europe\n(4% WACC)']

# Production cost comparison
prod_data = []
for scenario, label in zip(scenario_order, scenario_labels):
    scen_data = results_df[results_df['scenario'] == scenario]
    prod_data.append({
        'scenario': label,
        'min': scen_data['lcoh_production'].min(),
        'mean': scen_data['lcoh_production'].mean(),
        'max': scen_data['lcoh_production'].max()
    })
prod_df = pd.DataFrame(prod_data)

x = np.arange(len(scenario_labels))
width = 0.25

axes[0].bar(x - width, prod_df['min'], width, label='Min', alpha=0.8)
axes[0].bar(x, prod_df['mean'], width, label='Mean', alpha=0.8)
axes[0].bar(x + width, prod_df['max'], width, label='Max', alpha=0.8)
axes[0].set_ylabel('LCOH Production (EUR/kg H2)')
axes[0].set_title('Production Cost by Financing Scenario')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenario_labels, fontsize=9)
axes[0].legend()
axes[0].axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='IEA 2030 Target')
axes[0].legend(loc='upper right')

# Delivered cost comparison
deliv_data = []
for scenario, label in zip(scenario_order, scenario_labels):
    scen_data = results_df[results_df['scenario'] == scenario]
    deliv_data.append({
        'scenario': label,
        'min': scen_data['total_delivered_cost'].min(),
        'mean': scen_data['total_delivered_cost'].mean(),
        'max': scen_data['total_delivered_cost'].max()
    })
deliv_df = pd.DataFrame(deliv_data)

axes[1].bar(x - width, deliv_df['min'], width, label='Min', alpha=0.8)
axes[1].bar(x, deliv_df['mean'], width, label='Mean', alpha=0.8)
axes[1].bar(x + width, deliv_df['max'], width, label='Max', alpha=0.8)
axes[1].set_ylabel('Delivered Cost (EUR/kg H2)')
axes[1].set_title('Total Delivered Cost to Europe by Scenario')
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenario_labels, fontsize=9)
axes[1].axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='Competitiveness Threshold')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('report/images/scenario_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/scenario_comparison.png")

# Figure 5: WACC Sensitivity Analysis
print("  Creating WACC sensitivity plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate LCOH at different WACC values for a representative site
representative_site = df.iloc[10]  # Pick a middle site
wacc_values = np.linspace(0.03, 0.15, 13)
lcoh_values = []
deliv_values = []

for wacc in wacc_values:
    fin_params = type('FinancingParams', (), {'wacc': wacc, 'project_lifetime': 25})()
    # Quick calculation
    from lcoh_model import calculate_total_delivered_cost, TechnoEconomicParams
    tech_p = TechnoEconomicParams()
    
    result = calculate_total_delivered_cost(
        representative_site, tech_p, 
        type('FinancingParams', (), {'wacc': wacc, 'project_lifetime': 25, 'debt_ratio': 0.7})(),
        'sensitivity'
    )
    lcoh_values.append(result['lcoh_production'])
    deliv_values.append(result['total_delivered_cost'])

ax.plot(wacc_values * 100, lcoh_values, 'o-', linewidth=2, markersize=8, label='Production Cost')
ax.plot(wacc_values * 100, deliv_values, 's-', linewidth=2, markersize=8, label='Delivered Cost')
ax.set_xlabel('WACC (%)')
ax.set_ylabel('Cost (EUR/kg H2)')
ax.set_title('Sensitivity of LCOH to Weighted Average Cost of Capital')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotation for key points
ax.annotate(f'5% WACC: {deliv_values[6]:.2f} EUR/kg', 
            xy=(5, deliv_values[6]), xytext=(7, deliv_values[6]+0.5),
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'12% WACC: {deliv_values[-1]:.2f} EUR/kg',
            xy=(12, deliv_values[-1]), xytext=(8, deliv_values[-1]-0.8),
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('report/images/wacc_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/wacc_sensitivity.png")

# Figure 6: Africa vs Europe Comparison
print("  Creating Africa vs Europe comparison...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Get European production costs
europe_baseline = results_df[results_df['scenario'] == 'europe_production']['lcoh_production'].mean()
europe_delivered = results_df[results_df['scenario'] == 'europe_production']['total_delivered_cost'].mean()

# Get African costs (best sites)
africa_best_prod = baseline_results.nsmallest(5, 'lcoh_production')['lcoh_production'].mean()
africa_best_deliv = baseline_results.nsmallest(5, 'total_delivered_cost')['total_delivered_cost'].mean()
africa_avg_prod = baseline_results['lcoh_production'].mean()
africa_avg_deliv = baseline_results['total_delivered_cost'].mean()

categories = ['Production\n(Best African Sites)', 'Production\n(African Average)', 
              'Production\n(Europe)', 'Delivered\n(Best African)', 
              'Delivered\n(African Avg)', 'Delivered\n(Europe)']
values = [africa_best_prod, africa_avg_prod, europe_baseline,
          africa_best_deliv, africa_avg_deliv, europe_delivered]

colors = ['#2ecc71', '#27ae60', '#3498db', '#e74c3c', '#c0392b', '#2980b9']

bars = ax.bar(categories, values, color=colors, edgecolor='black')
ax.set_ylabel('Cost (EUR/kg H2)')
ax.set_title('Comparison: African vs European Green Hydrogen Costs (2030)')
ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='Target Competitiveness')
ax.legend()

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('report/images/africa_vs_europe.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/africa_vs_europe.png")

# Figure 7: Cost Breakdown Stacked Bar
print("  Creating cost breakdown plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Get cost components for best African site
best_site = baseline_results.iloc[0]
components = {
    'Production': best_site['lcoh_production'],
    'NH3 Conversion': best_site['nh3_conversion_cost'],
    'Shipping': best_site['shipping_cost'],
    'Reconversion': best_site['reconversion_cost']
}

# Also get average site
avg_site = baseline_results.iloc[len(baseline_results)//2]
components_avg = {
    'Production': avg_site['lcoh_production'],
    'NH3 Conversion': avg_site['nh3_conversion_cost'],
    'Shipping': avg_site['shipping_cost'],
    'Reconversion': avg_site['reconversion_cost']
}

x = np.arange(2)
width = 0.6

# Stack the components
bottom_vals_best = [0, 0, 0, 0]
bottom_vals_avg = [0, 0, 0, 0]

colors_comp = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
labels_comp = ['Production', 'NH3 Conversion', 'Shipping', 'Reconversion']

for i, (label, color) in enumerate(zip(labels_comp, colors_comp)):
    ax.bar(x[0], list(components.values())[i], bottom=sum(list(components.values())[:i]), 
           color=color, label=label if i == 0 else "", width=width)
    ax.bar(x[1], list(components_avg.values())[i], bottom=sum(list(components_avg.values())[:i]),
           color=color, width=width)

ax.set_xticks(x)
ax.set_xticklabels(['Best African Site', 'Average African Site'])
ax.set_ylabel('Cost (EUR/kg H2)')
ax.set_title('Cost Breakdown: Best vs Average African Site (Baseline WACC)')

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors_comp, labels_comp)]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('report/images/cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: report/images/cost_breakdown.png")

# ============================================================================
# 5. SAVE ADDITIONAL OUTPUTS
# ============================================================================
print("\n[5/6] Saving additional outputs...")

# Save dependency check
dependency_check = {
    "python_packages": ["pandas", "numpy", "matplotlib", "seaborn", "geopandas"],
    "model_method": "GeoH2-inspired geospatial LCOH optimization",
    "temporal_scope": "2030 projections",
    "spatial_scope": "30 African hexagon sites",
    "financing_scenarios": list(FINANCING_SCENARIOS.keys()),
    "key_assumptions": {
        "electrolyzer_capex_2030": "500 EUR/kW",
        "pv_capex_2030": "400 EUR/kW",
        "wind_capex_2030": "900 EUR/kW",
        "wacc_baseline": "8%",
        "wacc_de_risked": "5%",
        "wacc_high_interest": "12%",
        "wacc_europe": "4%"
    }
}
with open('outputs/dependency_check.json', 'w') as f:
    json.dump(dependency_check, f, indent=2)
print("  Saved: outputs/dependency_check.json")

# Update target artifact inventory
artifact_status = {
    "target_artifacts": [
        {"id": "A1", "name": "LCOH_production_map", "status": "completed", "path": "report/images/lcoh_production_map.png"},
        {"id": "A2", "name": "delivered_cost_map", "status": "completed", "path": "report/images/delivered_cost_map.png"},
        {"id": "A3", "name": "scenario_comparison_bar", "status": "completed", "path": "report/images/scenario_comparison.png"},
        {"id": "A4", "name": "wacc_sensitivity_plot", "status": "completed", "path": "report/images/wacc_sensitivity.png"},
        {"id": "A5", "name": "africa_vs_europe_comparison", "status": "completed", "path": "report/images/africa_vs_europe.png"},
        {"id": "A6", "name": "cost_breakdown_stacked", "status": "completed", "path": "report/images/cost_breakdown.png"},
        {"id": "A7", "name": "least_cost_locations_table", "status": "completed", "path": "outputs/least_cost_locations.csv"},
        {"id": "A8", "name": "lcoh_results_all_scenarios", "status": "completed", "path": "outputs/lcoh_results.csv"},
        {"id": "A9", "name": "data_overview_plots", "status": "completed", "path": "report/images/data_overview.png"}
    ],
    "completion_checklist": {
        "methodology_documented": True,
        "all_figures_generated": True,
        "all_tables_generated": True,
        "report_written": False,
        "claim_recovery_complete": False
    }
}
with open('outputs/target_artifact_inventory.json', 'w') as f:
    json.dump(artifact_status, f, indent=2)
print("  Updated: outputs/target_artifact_inventory.json")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n[6/6] Analysis Complete!")
print("=" * 60)
print("\nKEY FINDINGS:")
print(f"  - Best African sites can produce H2 at ~{africa_best_prod:.2f} EUR/kg (2030)")
print(f"  - Delivered to Europe (via NH3): ~{africa_best_deliv:.2f} EUR/kg")
print(f"  - European production reference: ~{europe_delivered:.2f} EUR/kg")
print(f"  - De-risking (5% WACC) reduces costs by ~{(africa_best_deliv - deliv_values[6])/africa_best_deliv*100:.1f}%")
print(f"  - High interest rates (12%) increase costs by ~{(deliv_values[-1] - africa_best_deliv)/africa_best_deliv*100:.1f}%")
print("\nFIGURES GENERATED:")
print("  - report/images/data_overview.png")
print("  - report/images/lcoh_production_map.png")
print("  - report/images/delivered_cost_map.png")
print("  - report/images/scenario_comparison.png")
print("  - report/images/wacc_sensitivity.png")
print("  - report/images/africa_vs_europe.png")
print("  - report/images/cost_breakdown.png")
print("\nOUTPUT FILES:")
print("  - outputs/lcoh_results.csv")
print("  - outputs/least_cost_locations.csv")
print("  - outputs/dependency_check.json")
print("=" * 60)
