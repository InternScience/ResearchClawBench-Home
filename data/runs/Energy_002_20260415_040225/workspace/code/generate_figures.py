"""
Generate all figures for the African Green Hydrogen LCOH report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
import os

# Try geopandas for map
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

os.makedirs('report/images', exist_ok=True)

# Load results
df = pd.read_csv('outputs/full_results.csv')

# =============================================================================
# Figure 1: Map of Africa with LCOH by site
# =============================================================================

def plot_map_lcoh():
    """Create a map showing production sites colored by delivered cost."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    if HAS_GEOPANDAS:
        try:
            gdf = gpd.read_file('data/africa_map/ne_10m_admin_0_countries.shp')
            # Filter to African countries (rough bounding box)
            africa_mask = (
                (gdf.geometry.bounds['minx'] > -20) & 
                (gdf.geometry.bounds['maxx'] < 55) & 
                (gdf.geometry.bounds['miny'] > -38) & 
                (gdf.geometry.bounds['maxy'] < 40)
            )
            gdf_africa = gdf[africa_mask]
            gdf_africa.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, zorder=1)
        except Exception as e:
            print(f"Geopandas map loading issue: {e}")
            ax.set_facecolor('lightblue')
    else:
        ax.set_facecolor('lightblue')
    
    # Plot baseline scenario sites
    baseline = df[df['scenario'] == 'baseline_africa'].copy()
    
    # Color by total delivered cost
    sc = ax.scatter(
        baseline['lon'], baseline['lat'],
        c=baseline['total_delivered_cost'],
        cmap='RdYlGn_r',  # Red (high cost) to Green (low cost)
        s=200,
        edgecolors='black',
        linewidths=1,
        zorder=3,
        vmin=3.8, vmax=4.4
    )
    
    # Add labels for top 5 cheapest sites
    cheapest = baseline.nsmallest(5, 'total_delivered_cost')
    for _, row in cheapest.iterrows():
        ax.annotate(
            f"${row['total_delivered_cost']:.2f}",
            (row['lon'], row['lat']),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
            fontweight='bold',
            zorder=4
        )
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, label='Delivered Cost ($/kg H₂)')
    
    ax.set_xlim(-20, 55)
    ax.set_ylim(-38, 40)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(
        'Green Hydrogen Delivered Cost to Europe (Baseline Scenario, WACC 10%)\n'
        'via Ammonia Shipping and Reconversion — 2030 Projection',
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/map_lcoh_baseline.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: map_lcoh_baseline.png")


# =============================================================================
# Figure 2: Cost breakdown by scenario (stacked bar chart)
# =============================================================================

def plot_cost_breakdown():
    """Stacked bar chart showing cost components by financing scenario."""
    scenarios_order = ['baseline_africa', 'de_risked_africa', 'optimistic_africa', 'europe_baseline']
    scenario_labels = ['Africa\nBaseline\n(WACC 10%)', 'Africa\nDe-risked\n(WACC 5%)', 
                       'Africa\nOptimistic\n(WACC 3%)', 'Europe\nBaseline\n(WACC 4%)']
    
    components = ['lcoh_production', 'nh3_synthesis_cost', 'shipping_cost', 'reconversion_cost']
    component_labels = ['H₂ Production', 'NH₃ Synthesis', 'Shipping', 'Reconversion']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(scenarios_order))
    bottom = np.zeros(len(scenarios_order))
    
    for comp, label, color in zip(components, component_labels, colors):
        values = []
        for s in scenarios_order:
            if s == 'europe_baseline':
                # For EU, only production cost matters
                if comp == 'lcoh_production':
                    val = df[df['scenario'] == s]['total_delivered_cost'].mean()
                else:
                    val = 0
            else:
                val = df[df['scenario'] == s][comp].mean()
            values.append(val)
        
        bars = ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5, width=0.6)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            if v > 0.1:
                ax.text(x[i], bottom[i] + v/2, f'${v:.2f}', 
                       ha='center', va='center', fontsize=9, fontweight='bold')
        
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_ylabel('Cost ($/kg H₂)', fontsize=12)
    ax.set_title(
        'Levelized Cost Breakdown: African Green Hydrogen Delivered to Europe\n'
        'by Financing Scenario — 2030 Projection',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 5.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/cost_breakdown_scenarios.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: cost_breakdown_scenarios.png")


# =============================================================================
# Figure 3: Delivered cost distribution (violin/box plot)
# =============================================================================

def plot_cost_distribution():
    """Distribution of delivered costs across sites by scenario."""
    african_df = df[df['scenario'] != 'europe_baseline'].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violin plot
    parts = ax.violinplot(
        [african_df[african_df['scenario']==s]['total_delivered_cost'].values 
         for s in ['baseline_africa', 'de_risked_africa', 'optimistic_africa']],
        positions=[1, 2, 3],
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )
    
    for pc in parts['bodies']:
        pc.set_facecolor('#2196F3')
        pc.set_alpha(0.6)
    
    # Add individual points
    for i, s in enumerate(['baseline_africa', 'de_risked_africa', 'optimistic_africa'], 1):
        vals = african_df[african_df['scenario']==s]['total_delivered_cost'].values
        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(i + jitter, vals, alpha=0.5, color='black', s=20, zorder=3)
    
    # European baseline line
    eu_cost = df[df['scenario']=='europe_baseline']['total_delivered_cost'].mean()
    ax.axhline(y=eu_cost, color='red', linestyle='--', linewidth=2, 
               label=f'EU Production Baseline (${eu_cost:.2f}/kg)')
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Baseline\n(WACC 10%)', 'De-risked\n(WACC 5%)', 'Optimistic\n(WACC 3%)'],
                      fontsize=11)
    ax.set_ylabel('Delivered Cost ($/kg H₂)', fontsize=12)
    ax.set_title(
        'Distribution of Delivered Costs Across 30 African Sites\n'
        'by Financing Scenario vs. European Production — 2030',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(2.5, 4.5)
    
    plt.tight_layout()
    plt.savefig('report/images/cost_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: cost_distribution.png")


# =============================================================================
# Figure 4: WACC sensitivity analysis
# =============================================================================

def plot_wacc_sensitivity():
    """Sensitivity of delivered cost to WACC for best/worst/average sites."""
    wacc_values = np.linspace(0.02, 0.15, 50)
    
    import sys
    sys.path.insert(0, 'code')
    from lcoh_model import (
        compute_lcoh_site, compute_ammonia_cost, 
        compute_shipping_cost, compute_reconversion_cost
    )
    
    # Pick representative sites
    baseline = df[df['scenario'] == 'baseline_africa'].copy()
    best_site = baseline.loc[baseline['total_delivered_cost'].idxmin()]
    worst_site = baseline.loc[baseline['total_delivered_cost'].idxmax()]
    median_site = baseline.iloc[len(baseline)//2]
    
    sites = {
        'Best site (hex_020)': best_site,
        'Median site': median_site,
        'Worst site (hex_021)': worst_site,
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, site in sites.items():
        costs = []
        for wacc in wacc_values:
            prod = compute_lcoh_site(site['theo_pv'], site['theo_wind'], wacc,
                                     site['grid_dist_km'], site['road_dist_km'])
            nh3 = compute_ammonia_cost(prod['lcoh_production'], wacc)
            ship = compute_shipping_cost(site['ocean_dist_km'], wacc)
            crack = compute_reconversion_cost(wacc)
            
            total = (prod['lcoh_production'] + nh3['nh3_total_addition'] + 
                    ship['shipping_cost'] + 
                    crack['cracking_total'] * (1 + crack['efficiency_loss_factor']))
            costs.append(total)
        
        ax.plot(wacc_values * 100, costs, linewidth=2.5, label=label)
    
    # EU baseline
    eu_cost = df[df['scenario']=='europe_baseline']['total_delivered_cost'].mean()
    ax.axhline(y=eu_cost, color='red', linestyle='--', linewidth=2,
               label=f'EU Production (${eu_cost:.2f}/kg)')
    
    ax.set_xlabel('WACC (%)', fontsize=12)
    ax.set_ylabel('Delivered Cost ($/kg H₂)', fontsize=12)
    ax.set_title(
        'WACC Sensitivity: How Financing Costs Affect Competitiveness\n'
        'African Green Hydrogen vs. European Production — 2030',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 15)
    
    plt.tight_layout()
    plt.savefig('report/images/wacc_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: wacc_sensitivity.png")


# =============================================================================
# Figure 5: Competitiveness ranking across scenarios
# =============================================================================

def plot_competitiveness():
    """Ranking of sites by competitiveness relative to EU production."""
    eu_cost = df[df['scenario']=='europe_baseline']['total_delivered_cost'].mean()
    
    african_df = df[df['scenario'] != 'europe_baseline'].copy()
    
    # Compute cost advantage (negative = cheaper than EU)
    african_df['cost_advantage'] = african_df['total_delivered_cost'] - eu_cost
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    
    scenario_map = {
        'baseline_africa': ('Baseline (WACC 10%)', 0),
        'de_risked_africa': ('De-risked (WACC 5%)', 1),
        'optimistic_africa': ('Optimistic (WACC 3%)', 2),
    }
    
    for scenario, (label, ax_idx) in scenario_map.items():
        sub = african_df[african_df['scenario'] == scenario].sort_values('total_delivered_cost')
        
        colors = ['#4CAF50' if v < 0 else '#F44336' for v in sub['cost_advantage']]
        
        axes[ax_idx].barh(range(len(sub)), sub['total_delivered_cost'], 
                         color=colors, edgecolor='white', linewidth=0.5)
        axes[ax_idx].axvline(x=eu_cost, color='red', linestyle='--', linewidth=2,
                            label=f'EU: ${eu_cost:.2f}')
        axes[ax_idx].set_yticks(range(len(sub)))
        axes[ax_idx].set_yticklabels(sub['hex_id'], fontsize=7)
        axes[ax_idx].set_xlabel('Delivered Cost ($/kg H₂)', fontsize=10)
        axes[ax_idx].set_title(label, fontsize=12, fontweight='bold')
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(axis='x', alpha=0.3)
    
    fig.suptitle(
        'Site Ranking by Delivered Cost to Europe\n'
        'Green = Competitive with EU | Red = More Expensive than EU — 2030',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig('report/images/competitiveness_ranking.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: competitiveness_ranking.png")


# =============================================================================
# Figure 6: Carbon price impact on competitiveness
# =============================================================================

def plot_carbon_price_impact():
    """How carbon pricing affects the competitiveness gap."""
    carbon_prices = [0, 25, 50, 75, 100, 150, 200]  # $/tCO2
    
    # Gray H2 emissions: ~10.5 tCO2/t-H2 (SMR)
    co2_per_kg_h2 = 0.0105  # tCO2 per kg H2
    
    eu_cost = df[df['scenario']=='europe_baseline']['total_delivered_cost'].mean()
    
    african_costs = {}
    for s in ['baseline_africa', 'de_risked_africa', 'optimistic_africa']:
        sub = df[df['scenario'] == s]
        african_costs[s] = {
            'min': sub['total_delivered_cost'].min(),
            'mean': sub['total_delivered_cost'].mean(),
            'max': sub['total_delivered_cost'].max(),
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'baseline_africa': '#F44336', 'de_risked_africa': '#FF9800', 'optimistic_africa': '#4CAF50'}
    labels = {'baseline_africa': 'Africa Baseline (WACC 10%)',
              'de_risked_africa': 'Africa De-risked (WACC 5%)',
              'optimistic_africa': 'Africa Optimistic (WACC 3%)'}
    
    for s in ['baseline_africa', 'de_risked_africa', 'optimistic_africa']:
        # Green H2 has zero carbon cost; gray H2 gets more expensive
        adjusted_eu = [eu_cost + cp * co2_per_kg_h2 for cp in carbon_prices]
        
        ax.plot(carbon_prices, adjusted_eu, '--', color=colors[s], linewidth=2, alpha=0.5)
        ax.fill_between(carbon_prices, 
                       [african_costs[s]['min']]*len(carbon_prices),
                       [african_costs[s]['max']]*len(carbon_prices),
                       alpha=0.15, color=colors[s])
        ax.axhline(y=african_costs[s]['mean'], color=colors[s], linewidth=2,
                  label=f'{labels[s]}: ${african_costs[s]["mean"]:.2f}/kg')
    
    ax.set_xlabel('Carbon Price ($/tCO₂)', fontsize=12)
    ax.set_ylabel('Delivered Cost ($/kg H₂)', fontsize=12)
    ax.set_title(
        'Carbon Pricing Impact on Competitiveness\n'
        'Green H₂ (fixed) vs. Gray H₂ (dashed lines increase with carbon price) — 2030',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/carbon_price_impact.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: carbon_price_impact.png")


# =============================================================================
# Figure 7: Supply chain cost waterfall
# =============================================================================

def plot_waterfall():
    """Waterfall chart showing cumulative cost buildup along the supply chain."""
    baseline = df[df['scenario'] == 'baseline_africa'].copy()
    
    stages = ['H₂ Production', 'NH₃ Synthesis', 'Shipping', 'Reconversion', 'Total Delivered']
    mean_values = [
        baseline['lcoh_production'].mean(),
        baseline['nh3_synthesis_cost'].mean(),
        baseline['shipping_cost'].mean(),
        baseline['reconversion_cost'].mean(),
        baseline['total_delivered_cost'].mean(),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#333333']
    
    cumulative = 0
    for i, (stage, value, color) in enumerate(zip(stages, mean_values, colors)):
        if i < len(stages) - 1:
            ax.bar(i, value, color=color, edgecolor='white', linewidth=0.5, width=0.6)
            ax.text(i, value/2, f'${value:.2f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            cumulative += value
        else:
            ax.bar(i, value, color=color, edgecolor='white', linewidth=0.5, width=0.6)
            ax.text(i, value/2, f'${value:.2f}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white')
    
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('Cost ($/kg H₂)', fontsize=12)
    ax.set_title(
        'Supply Chain Cost Waterfall: African Green Hydrogen to Europe\n'
        'Baseline Scenario (WACC 10%) — 2030 Projection',
        fontsize=14, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/cost_waterfall.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: cost_waterfall.png")


# =============================================================================
# Run all figure generation
# =============================================================================

if __name__ == '__main__':
    plot_map_lcoh()
    plot_cost_breakdown()
    plot_cost_distribution()
    plot_wacc_sensitivity()
    plot_competitiveness()
    plot_carbon_price_impact()
    plot_waterfall()
    print("\nAll figures generated successfully!")
