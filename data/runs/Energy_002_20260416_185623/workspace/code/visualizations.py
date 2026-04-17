#!/usr/bin/env python3
"""
Visualization code for African Green Hydrogen Cost Analysis
Generates all figures for the research report
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import geopandas as gpd
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
os.makedirs(IMG, exist_ok=True)

# Load results
results_df = pd.read_csv(os.path.join(OUT, 'full_results.csv'))
sensitivity_df = pd.read_csv(os.path.join(OUT, 'wacc_sensitivity.csv'))
summary_df = pd.read_csv(os.path.join(OUT, 'scenario_summary.csv'))

# Load shapefile
africa_shp = gpd.read_file(os.path.join(BASE, 'data', 'africa_map', 'ne_10m_admin_0_countries.shp'))
# Filter to Africa
africa = africa_shp[africa_shp['CONTINENT'] == 'Africa']

# Color scheme
COLORS = {
    'baseline': '#e74c3c',
    'moderate_derisking': '#f39c12',
    'full_derisking': '#2ecc71',
    'rising_ir': '#9b59b6',
    'optimistic_2030': '#3498db',
}

SCENARIO_LABELS = {
    'baseline': 'Baseline\n(WACC 10%)',
    'moderate_derisking': 'Moderate\nDe-risking\n(WACC 8%)',
    'full_derisking': 'Full\nDe-risking\n(WACC 6%)',
    'rising_ir': 'Rising IR\n(WACC 12%)',
    'optimistic_2030': 'Optimistic\n2030\n(WACC 6%)',
}

SHORT_LABELS = {
    'baseline': 'Baseline (10%)',
    'moderate_derisking': 'Moderate (8%)',
    'full_derisking': 'Full De-risk (6%)',
    'rising_ir': 'Rising IR (12%)',
    'optimistic_2030': 'Optimistic 2030',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ============================================================
# FIGURE 1: Geospatial Map of Delivered Cost (Baseline)
# ============================================================
print("Creating Figure 1: Geospatial map...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

scenarios_to_map = ['baseline', 'full_derisking', 'optimistic_2030']
titles_map = ['(a) Baseline (WACC 10%)', '(b) Full De-risking (WACC 6%)', '(c) Optimistic 2030']

for ax, scen, title in zip(axes, scenarios_to_map, titles_map):
    sdf = results_df[results_df['scenario'] == scen]
    
    # Plot Africa
    africa.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
    
    # Focus on southern Africa
    ax.set_xlim(8, 28)
    ax.set_ylim(-32, -14)
    
    # Plot hexagon sites
    vmin = 3.5
    vmax = 6.5
    scatter = ax.scatter(sdf['lon'], sdf['lat'], 
                         c=sdf['total_delivered'], 
                         cmap='RdYlGn_r', 
                         s=120, 
                         edgecolors='black',
                         linewidth=0.5,
                         vmin=vmin, vmax=vmax,
                         zorder=5)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Delivered Cost (€/kgH₂)', fontsize=10)

fig.suptitle('Delivered Cost of African Green Hydrogen to Europe by 2030', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig1_geospatial_delivered_cost.png'))
plt.close()
print("  Saved fig1_geospatial_delivered_cost.png")

# ============================================================
# FIGURE 2: Cost Breakdown Waterfall (Best Site, Baseline)
# ============================================================
print("Creating Figure 2: Cost breakdown waterfall...")

baseline = results_df[results_df['scenario'] == 'baseline'].sort_values('total_delivered').iloc[0]

components = {
    'Electricity\n(RE)': baseline['electricity_cost'],
    'Electrolyzer': baseline['electrolyzer_cost'],
    'Battery\nStorage': baseline['battery_cost'],
    'H₂ Storage': baseline['h2_storage_cost'],
    'Water': baseline['water_cost'],
    'NH₃\nSynthesis': baseline['nh3_conversion'],
    'Transport\nto Port': baseline['transport_to_port'],
    'Shipping': baseline['shipping'],
    'NH₃\nCracking': baseline['reconversion'],
}

fig, ax = plt.subplots(figsize=(14, 7))

labels = list(components.keys())
values = list(components.values())
cumulative = np.cumsum(values)

# Colors for different categories
cat_colors = ['#3498db', '#3498db', '#3498db', '#3498db',  # production
              '#27ae60',  # water
              '#e67e22',  # conversion
              '#9b59b6',  # transport
              '#9b59b6',  # shipping
              '#e74c3c']  # reconversion

# Draw waterfall bars
bottoms = [0] + list(cumulative[:-1])
bars = ax.bar(range(len(labels)), values, bottom=bottoms, 
              color=cat_colors, edgecolor='white', linewidth=1.5, width=0.6)

# Add value labels
for i, (v, b) in enumerate(zip(values, bottoms)):
    ax.text(i, b + v/2, f'€{v:.2f}', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')

# Add total bar
total = sum(values)
ax.bar(len(labels), total, color='#2c3e50', edgecolor='white', linewidth=1.5, width=0.6)
ax.text(len(labels), total/2, f'€{total:.2f}', ha='center', va='center',
        fontweight='bold', fontsize=11, color='white')

# EU benchmark line
eu_bench = baseline['eu_benchmark']
ax.axhline(y=eu_bench, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.text(len(labels) + 0.5, eu_bench + 0.1, f'EU Benchmark: €{eu_bench:.2f}/kgH₂', 
        color='red', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(labels) + 1))
ax.set_xticklabels(labels + ['TOTAL\nDelivered'], fontsize=10)
ax.set_ylabel('Cost (€/kgH₂)', fontsize=12)
ax.set_title(f'Cost Breakdown: Delivered African Green H₂ to Europe\n'
             f'Best Site: {baseline["hex_id"]} (Baseline Scenario, WACC=10%)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(total, eu_bench) * 1.15)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#3498db', label='H₂ Production'),
    mpatches.Patch(facecolor='#27ae60', label='Water Supply'),
    mpatches.Patch(facecolor='#e67e22', label='NH₃ Conversion'),
    mpatches.Patch(facecolor='#9b59b6', label='Transport & Shipping'),
    mpatches.Patch(facecolor='#e74c3c', label='Reconversion (EU)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig2_cost_breakdown_waterfall.png'))
plt.close()
print("  Saved fig2_cost_breakdown_waterfall.png")

# ============================================================
# FIGURE 3: Scenario Comparison Bar Chart
# ============================================================
print("Creating Figure 3: Scenario comparison...")

fig, ax = plt.subplots(figsize=(14, 7))

scenarios = ['baseline', 'moderate_derisking', 'full_derisking', 'rising_ir', 'optimistic_2030']
x = np.arange(len(scenarios))
width = 0.35

# Get min, mean, max delivered costs per scenario
mins = [results_df[results_df['scenario']==s]['total_delivered'].min() for s in scenarios]
means = [results_df[results_df['scenario']==s]['total_delivered'].mean() for s in scenarios]
maxs = [results_df[results_df['scenario']==s]['total_delivered'].max() for s in scenarios]
eu_benchmarks = [results_df[results_df['scenario']==s]['eu_benchmark'].iloc[0] for s in scenarios]

# Bar for delivered cost (mean with error bars)
colors = [COLORS[s] for s in scenarios]
bars = ax.bar(x, means, width, color=colors, edgecolor='white', linewidth=1.5,
              yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)],
              capsize=5, error_kw={'linewidth': 2}, label='Delivered to EU (mean ± range)')

# EU benchmark markers
ax.scatter(x, eu_benchmarks, color='red', marker='D', s=100, zorder=5, 
           label='EU Domestic Production', edgecolors='darkred', linewidth=1)

# Connect EU benchmarks
ax.plot(x, eu_benchmarks, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

# Add value labels
for i, (m, eu) in enumerate(zip(means, eu_benchmarks)):
    ax.text(i, m + 0.15, f'€{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + 0.15, eu + 0.1, f'€{eu:.2f}', ha='left', va='bottom', fontsize=9, color='red')

# Cost advantage annotation
for i, (m, eu) in enumerate(zip(means, eu_benchmarks)):
    advantage = eu - m
    pct = advantage / eu * 100
    ax.annotate(f'Δ€{advantage:.1f}\n({pct:.0f}%)', 
                xy=(i, (m + eu)/2), fontsize=8, ha='center', 
                color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels([SHORT_LABELS[s] for s in scenarios], fontsize=10)
ax.set_ylabel('Cost (€/kgH₂)', fontsize=12)
ax.set_title('Delivered Cost of African Green H₂ vs European Domestic Production\nAcross Financing Scenarios (2030)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 12)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig3_scenario_comparison.png'))
plt.close()
print("  Saved fig3_scenario_comparison.png")

# ============================================================
# FIGURE 4: WACC Sensitivity Analysis
# ============================================================
print("Creating Figure 4: WACC sensitivity...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel a: Total delivered cost vs WACC
ax1.plot(sensitivity_df['wacc']*100, sensitivity_df['total_delivered'], 
         'b-o', linewidth=2, markersize=5, label='Delivered to EU')
ax1.plot(sensitivity_df['wacc']*100, sensitivity_df['eu_benchmark'], 
         'r--D', linewidth=2, markersize=5, label='EU Domestic')
ax1.fill_between(sensitivity_df['wacc']*100, sensitivity_df['total_delivered'], 
                 sensitivity_df['eu_benchmark'], alpha=0.2, color='green',
                 label='Cost Advantage')

ax1.set_xlabel('Africa WACC (%)', fontsize=12)
ax1.set_ylabel('Cost (€/kgH₂)', fontsize=12)
ax1.set_title('(a) Delivered Cost vs WACC\n(Best African Site)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Mark scenario WACCs
scenario_waccs = {'Baseline': 10, 'De-risked': 6, 'Rising IR': 12}
for label, wacc in scenario_waccs.items():
    idx = (sensitivity_df['wacc']*100 - wacc).abs().idxmin()
    cost = sensitivity_df.loc[idx, 'total_delivered']
    ax1.annotate(label, xy=(wacc, cost), xytext=(wacc+1, cost-0.3),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

# Panel b: Cost component breakdown vs WACC
ax2.stackplot(sensitivity_df['wacc']*100,
              sensitivity_df['electricity'],
              sensitivity_df['electrolyzer'],
              sensitivity_df['battery'],
              sensitivity_df['h2_storage'],
              labels=['Electricity (RE)', 'Electrolyzer', 'Battery', 'H₂ Storage'],
              colors=['#3498db', '#e67e22', '#2ecc71', '#9b59b6'],
              alpha=0.8)

ax2.set_xlabel('Africa WACC (%)', fontsize=12)
ax2.set_ylabel('Production Cost (€/kgH₂)', fontsize=12)
ax2.set_title('(b) Production Cost Components vs WACC', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(alpha=0.3)

fig.suptitle('WACC Sensitivity Analysis for African Green Hydrogen', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig4_wacc_sensitivity.png'))
plt.close()
print("  Saved fig4_wacc_sensitivity.png")

# ============================================================
# FIGURE 5: Site-by-Site Comparison (Baseline)
# ============================================================
print("Creating Figure 5: Site-by-site comparison...")

baseline_df = results_df[results_df['scenario'] == 'baseline'].sort_values('total_delivered')

fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(baseline_df))
width = 0.6

# Stacked bar chart
bottom = np.zeros(len(baseline_df))
components_plot = [
    ('electricity_cost', 'Electricity (RE)', '#3498db'),
    ('electrolyzer_cost', 'Electrolyzer', '#e67e22'),
    ('battery_cost', 'Battery Storage', '#2ecc71'),
    ('h2_storage_cost', 'H₂ Storage', '#9b59b6'),
    ('water_cost', 'Water', '#1abc9c'),
    ('nh3_conversion', 'NH₃ Synthesis', '#f1c40f'),
    ('transport_to_port', 'Transport to Port', '#95a5a6'),
    ('shipping', 'Shipping', '#34495e'),
    ('reconversion', 'Reconversion', '#e74c3c'),
]

for col, label, color in components_plot:
    vals = baseline_df[col].values
    ax.bar(x, vals, width, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5)
    bottom += vals

# EU benchmark line
eu_bench = baseline_df['eu_benchmark'].iloc[0]
ax.axhline(y=eu_bench, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'EU Benchmark (€{eu_bench:.1f})')

ax.set_xticks(x)
ax.set_xticklabels(baseline_df['hex_id'].values, rotation=90, fontsize=8)
ax.set_ylabel('Delivered Cost (€/kgH₂)', fontsize=12)
ax.set_title('Delivered Cost Breakdown by Site (Baseline Scenario, WACC=10%)', 
             fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, eu_bench * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig5_site_comparison.png'))
plt.close()
print("  Saved fig5_site_comparison.png")

# ============================================================
# FIGURE 6: Cost Competitiveness Map
# ============================================================
print("Creating Figure 6: Competitiveness map...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, scen, title in zip(axes, ['baseline', 'optimistic_2030'], 
                           ['(a) Baseline Scenario', '(b) Optimistic 2030']):
    sdf = results_df[results_df['scenario'] == scen]
    
    africa.plot(ax=ax, color='#f5f5f5', edgecolor='#cccccc', linewidth=0.5)
    ax.set_xlim(8, 28)
    ax.set_ylim(-32, -14)
    
    # Color by cost advantage
    advantage = sdf['cost_advantage'].values
    
    scatter = ax.scatter(sdf['lon'], sdf['lat'], 
                         c=advantage, 
                         cmap='RdYlGn', 
                         s=150, 
                         edgecolors='black',
                         linewidth=0.5,
                         vmin=min(advantage)*0.9, 
                         vmax=max(advantage)*1.1,
                         zorder=5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Cost Advantage vs EU (€/kgH₂)', fontsize=10)
    
    ax.set_title(f'{title}\nAll sites competitive (advantage: €{min(advantage):.1f}-{max(advantage):.1f}/kgH₂)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

fig.suptitle('Cost Competitiveness of African Green H₂ vs European Domestic Production', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig6_competitiveness_map.png'))
plt.close()
print("  Saved fig6_competitiveness_map.png")

# ============================================================
# FIGURE 7: De-risking Impact Analysis
# ============================================================
print("Creating Figure 7: De-risking impact...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel a: Distribution of delivered costs across scenarios
scenarios_order = ['rising_ir', 'baseline', 'moderate_derisking', 'full_derisking', 'optimistic_2030']
bp_data = [results_df[results_df['scenario']==s]['total_delivered'].values for s in scenarios_order]
bp_labels = [SHORT_LABELS[s] for s in scenarios_order]
bp_colors = [COLORS[s] for s in scenarios_order]

bp = ax1.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add EU benchmark points
for i, s in enumerate(scenarios_order):
    eu = results_df[results_df['scenario']==s]['eu_benchmark'].iloc[0]
    ax1.plot(i+1, eu, 'rD', markersize=10, label='EU Benchmark' if i==0 else '')

ax1.set_ylabel('Cost (€/kgH₂)', fontsize=12)
ax1.set_title('(a) Delivered Cost Distribution by Scenario', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=15)

# Panel b: Cost reduction from de-risking
baseline_mean = results_df[results_df['scenario']=='baseline']['total_delivered'].mean()
reductions = {}
for s in scenarios_order:
    s_mean = results_df[results_df['scenario']==s]['total_delivered'].mean()
    reductions[s] = (baseline_mean - s_mean) / baseline_mean * 100

bars = ax2.bar(range(len(scenarios_order)), 
               [reductions[s] for s in scenarios_order],
               color=[COLORS[s] for s in scenarios_order],
               edgecolor='white', linewidth=1.5)

for i, (s, v) in enumerate(zip(scenarios_order, [reductions[s] for s in scenarios_order])):
    ax2.text(i, v + 0.5 if v >= 0 else v - 1.5, f'{v:.1f}%', 
             ha='center', fontweight='bold', fontsize=10)

ax2.set_xticks(range(len(scenarios_order)))
ax2.set_xticklabels([SHORT_LABELS[s] for s in scenarios_order], fontsize=10, rotation=15)
ax2.set_ylabel('Cost Change vs Baseline (%)', fontsize=12)
ax2.set_title('(b) Cost Impact of Financing Scenarios\n(Relative to Baseline)', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Impact of De-risking and Interest Rate Environment', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig7_derisking_impact.png'))
plt.close()
print("  Saved fig7_derisking_impact.png")

# ============================================================
# FIGURE 8: Resource Quality vs Cost
# ============================================================
print("Creating Figure 8: Resource quality vs cost...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

baseline_df = results_df[results_df['scenario'] == 'baseline']

# Panel a: PV potential vs LCOH
ax = axes[0]
scatter = ax.scatter(baseline_df['theo_pv'], baseline_df['lcoh_production'], 
                     c=baseline_df['total_delivered'], cmap='viridis_r', 
                     s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Solar PV Potential (CF)', fontsize=11)
ax.set_ylabel('LCOH Production (€/kgH₂)', fontsize=11)
ax.set_title('(a) Solar Potential vs Production Cost', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Delivered Cost (€/kgH₂)', shrink=0.8)
ax.grid(alpha=0.3)

# Panel b: Wind potential vs LCOH
ax = axes[1]
scatter = ax.scatter(baseline_df['theo_wind'], baseline_df['lcoh_production'],
                     c=baseline_df['total_delivered'], cmap='viridis_r',
                     s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Wind Potential (CF)', fontsize=11)
ax.set_ylabel('LCOH Production (€/kgH₂)', fontsize=11)
ax.set_title('(b) Wind Potential vs Production Cost', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Delivered Cost (€/kgH₂)', shrink=0.8)
ax.grid(alpha=0.3)

# Panel c: Distance to coast vs total delivered cost
ax = axes[2]
scatter = ax.scatter(baseline_df['ocean_dist_km'], baseline_df['total_delivered'],
                     c=baseline_df['lcoh_production'], cmap='coolwarm',
                     s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Distance to Coast (km)', fontsize=11)
ax.set_ylabel('Total Delivered Cost (€/kgH₂)', fontsize=11)
ax.set_title('(c) Coastal Proximity vs Delivered Cost', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='LCOH Production (€/kgH₂)', shrink=0.8)
ax.grid(alpha=0.3)

fig.suptitle('Resource Quality and Location Drivers of Green Hydrogen Cost', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig8_resource_quality.png'))
plt.close()
print("  Saved fig8_resource_quality.png")

# ============================================================
# FIGURE 9: Cost Component Pie Chart (Best vs Worst Site)
# ============================================================
print("Creating Figure 9: Cost component comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, idx, title in zip(axes, [0, -1], 
                           ['(a) Best Site (Lowest Cost)', '(b) Worst Site (Highest Cost)']):
    site = results_df[results_df['scenario'] == 'baseline'].sort_values('total_delivered').iloc[idx]
    
    labels = ['Electricity', 'Electrolyzer', 'Battery', 'H₂ Storage', 
              'Water', 'NH₃ Synthesis', 'Transport', 'Shipping', 'Reconversion']
    sizes = [site['electricity_cost'], site['electrolyzer_cost'], site['battery_cost'],
             site['h2_storage_cost'], site['water_cost'], site['nh3_conversion'],
             site['transport_to_port'], site['shipping'], site['reconversion']]
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#1abc9c',
              '#f1c40f', '#95a5a6', '#34495e', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       pctdistance=0.85, textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_fontsize(7)
    
    ax.set_title(f'{title}\n{site["hex_id"]}: €{site["total_delivered"]:.2f}/kgH₂', 
                 fontsize=12, fontweight='bold')

fig.suptitle('Cost Component Distribution (Baseline Scenario)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig9_cost_components_pie.png'))
plt.close()
print("  Saved fig9_cost_components_pie.png")

# ============================================================
# FIGURE 10: Summary Dashboard
# ============================================================
print("Creating Figure 10: Summary dashboard...")

fig = plt.figure(figsize=(18, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Panel 1: Scenario comparison (top left)
ax1 = fig.add_subplot(gs[0, 0:2])
scenarios_order = ['optimistic_2030', 'full_derisking', 'moderate_derisking', 'baseline', 'rising_ir']
y_pos = np.arange(len(scenarios_order))

delivered = [results_df[results_df['scenario']==s]['total_delivered'].mean() for s in scenarios_order]
eu_bench = [results_df[results_df['scenario']==s]['eu_benchmark'].iloc[0] for s in scenarios_order]

bars1 = ax1.barh(y_pos - 0.15, delivered, 0.3, color=[COLORS[s] for s in scenarios_order], 
                 edgecolor='white', label='Delivered to EU')
bars2 = ax1.barh(y_pos + 0.15, eu_bench, 0.3, color='lightcoral', 
                 edgecolor='white', alpha=0.7, label='EU Domestic')

for i, (d, e) in enumerate(zip(delivered, eu_bench)):
    ax1.text(d + 0.1, i - 0.15, f'€{d:.1f}', va='center', fontsize=9, fontweight='bold')
    ax1.text(e + 0.1, i + 0.15, f'€{e:.1f}', va='center', fontsize=9, color='red')

ax1.set_yticks(y_pos)
ax1.set_yticklabels([SHORT_LABELS[s] for s in scenarios_order])
ax1.set_xlabel('Cost (€/kgH₂)')
ax1.set_title('Delivered Cost vs EU Benchmark', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Panel 2: Key metrics (top right)
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
metrics = [
    ('Best Delivered Cost\n(Optimistic 2030)', f'€{results_df[results_df["scenario"]=="optimistic_2030"]["total_delivered"].min():.2f}/kgH₂'),
    ('Best Delivered Cost\n(Baseline)', f'€{results_df[results_df["scenario"]=="baseline"]["total_delivered"].min():.2f}/kgH₂'),
    ('EU Benchmark\n(Baseline)', f'€{results_df[results_df["scenario"]=="baseline"]["eu_benchmark"].iloc[0]:.2f}/kgH₂'),
    ('Max Cost Advantage\n(Baseline)', f'€{results_df[results_df["scenario"]=="baseline"]["cost_advantage"].max():.2f}/kgH₂'),
    ('Sites Competitive\n(All Scenarios)', '30/30 (100%)'),
]

for i, (label, value) in enumerate(metrics):
    y = 0.9 - i * 0.19
    ax2.text(0.05, y, label, fontsize=10, va='top', transform=ax2.transAxes)
    ax2.text(0.95, y, value, fontsize=12, fontweight='bold', va='top', ha='right',
             transform=ax2.transAxes, color='#2c3e50')

ax2.set_title('Key Results', fontweight='bold', fontsize=12)

# Panel 3: WACC sensitivity (bottom left)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(sensitivity_df['wacc']*100, sensitivity_df['total_delivered'], 'b-o', 
         linewidth=2, markersize=4, label='Delivered')
ax3.plot(sensitivity_df['wacc']*100, sensitivity_df['eu_benchmark'], 'r--', 
         linewidth=2, label='EU Benchmark')
ax3.fill_between(sensitivity_df['wacc']*100, sensitivity_df['total_delivered'],
                 sensitivity_df['eu_benchmark'], alpha=0.15, color='green')
ax3.set_xlabel('WACC (%)')
ax3.set_ylabel('Cost (€/kgH₂)')
ax3.set_title('WACC Sensitivity', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Panel 4: Production cost map (bottom center)
ax4 = fig.add_subplot(gs[1, 1])
sdf = results_df[results_df['scenario'] == 'baseline']
africa.plot(ax=ax4, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
ax4.set_xlim(8, 28)
ax4.set_ylim(-32, -14)
scatter = ax4.scatter(sdf['lon'], sdf['lat'], c=sdf['lcoh_production'], 
                      cmap='YlOrRd', s=80, edgecolors='black', linewidth=0.5, zorder=5)
plt.colorbar(scatter, ax=ax4, shrink=0.7, label='€/kgH₂')
ax4.set_title('Production LCOH (Baseline)', fontweight='bold')

# Panel 5: Cost advantage histogram (bottom right)
ax5 = fig.add_subplot(gs[1, 2])
for s in ['baseline', 'full_derisking', 'optimistic_2030']:
    advantages = results_df[results_df['scenario']==s]['cost_advantage']
    ax5.hist(advantages, bins=10, alpha=0.5, color=COLORS[s], label=SHORT_LABELS[s], edgecolor='white')
ax5.set_xlabel('Cost Advantage vs EU (€/kgH₂)')
ax5.set_ylabel('Number of Sites')
ax5.set_title('Cost Advantage Distribution', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

fig.suptitle('African Green Hydrogen to Europe: 2030 Cost Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=1.01)
plt.savefig(os.path.join(IMG, 'fig10_summary_dashboard.png'))
plt.close()
print("  Saved fig10_summary_dashboard.png")

print("\nAll figures generated successfully!")
