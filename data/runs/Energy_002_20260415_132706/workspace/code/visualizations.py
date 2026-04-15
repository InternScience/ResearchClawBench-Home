
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUTS_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_002_20260415_132706/outputs')
REPORT_IMG_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_002_20260415_132706/report/images')
REPORT_IMG_DIR.mkdir(exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_results():
    return pd.read_csv(OUTPUTS_DIR / 'lcoh_results.csv')

def plot_data_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    scatter = ax.scatter(df['lon'], df['lat'], c=df['theo_pv'], cmap='YlOrRd', s=150, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Solar PV Potential by Location')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('PV Potential (normalized)')
    
    ax = axes[0, 1]
    scatter = ax.scatter(df['lon'], df['lat'], c=df['theo_wind'], cmap='Blues', s=150, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Wind Potential by Location')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Wind Potential (normalized)')
    
    ax = axes[1, 0]
    scatter = ax.scatter(df['lon'], df['lat'], c=df['ocean_dist_km'], cmap='Greens', s=150, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Distance to Ocean (km)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance (km)')
    
    ax = axes[1, 1]
    scatter = ax.scatter(df['lon'], df['lat'], c=df['grid_dist_km'], cmap='Purples', s=150, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Distance to Grid (km)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance (km)')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'fig1_data_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved fig1_data_overview.png')

def plot_lcoh_maps(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scenarios = ['Optimistic_DeRisked', 'Moderate_Standard', 'High_Risk', 'VeryHigh_Constrained']
    titles = ['Optimistic (5% WACC)', 'Moderate (8% WACC)', 'High Risk (12% WACC)', 'Very High Risk (15% WACC)']
    
    vmin, vmax = 3.5, 12
    
    for idx, (scenario, title) in enumerate(zip(scenarios, titles)):
        ax = axes[idx // 2, idx % 2]
        scenario_df = df[df['scenario'] == scenario]
        
        scatter = ax.scatter(scenario_df['lon'], scenario_df['lat'], 
                           c=scenario_df['lcoh_delivered'], cmap='RdYlGn_r', 
                           s=150, edgecolors='black', linewidth=0.5, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'LCOH - {title}')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('LCOH (EUR/kg H2)')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'fig2_lcoh_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved fig2_lcoh_maps.png')

def plot_cost_breakdown(df):
    moderate_df = df[df['scenario'] == 'Moderate_Standard']
    best_idx = moderate_df['lcoh_delivered'].idxmin()
    best_row = moderate_df.loc[best_idx]
    
    production = best_row['lcoh_production']
    production_adjusted = production / best_row['total_efficiency']
    conversion = best_row['conversion_cost']
    shipping = best_row['shipping_cost']
    reconversion = best_row['reconversion_cost']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ['Production (adjusted)', 'Ammonia Conversion', 'Shipping', 'Reconversion (Europe)']
    costs = [production_adjusted, conversion, shipping, reconversion]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    bars = ax.bar(components, costs, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Cost (EUR/kg H2)')
    ax.set_title('Cost Breakdown for Best African Location (Moderate Scenario)')
    ax.set_ylim(0, max(costs) * 1.3)
    
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{cost:.2f}', ha='center', va='bottom', fontweight='bold')
    
    total = sum(costs)
    ax.axhline(y=total, color='red', linestyle='--', linewidth=2, label=f'Total: {total:.2f} EUR/kg')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'fig3_cost_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved fig3_cost_breakdown.png')

def plot_scenario_comparison(df):
    scenarios = ['Optimistic_DeRisked', 'Moderate_Standard', 'High_Risk', 'VeryHigh_Constrained']
    scenario_labels = ['Optimistic (5% WACC)', 'Moderate (8% WACC)', 'High Risk (12% WACC)', 'Very High (15% WACC)']
    
    summary = []
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        summary.append({
            'scenario': scenario,
            'min': scenario_df['lcoh_delivered'].min(),
            'mean': scenario_df['lcoh_delivered'].mean(),
            'max': scenario_df['lcoh_delivered'].max(),
        })
    
    summary_df = pd.DataFrame(summary)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    means = summary_df['mean']
    mins = summary_df['min']
    maxs = summary_df['max']
    
    bars = ax.bar(x, means, width, label='Mean', color='steelblue', edgecolor='black')
    ax.errorbar(x, means, yerr=[means - mins, maxs - means], fmt='none', 
               color='black', capsize=5, capthick=2, label='Range (min-max)')
    
    ax.set_xlabel('Financing Scenario')
    ax.set_ylabel('LCOH (EUR/kg H2)')
    ax.set_title('Delivered Hydrogen Cost by Financing Scenario (2030)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
               f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'fig4_scenario_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved fig4_scenario_comparison.png')

def plot_competitiveness_analysis(df):
    scenarios = ['Optimistic_DeRisked', 'Moderate_Standard', 'High_Risk', 'VeryHigh_Constrained']
    scenario_labels = ['Optimistic (5%)', 'Moderate (8%)', 'High Risk (12%)', 'Very High (15%)']
    
    african_costs = []
    eu_costs = []
    
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        african_costs.append(scenario_df['lcoh_delivered'].min())
        eu_costs.append(scenario_df['eu_production_cost'].iloc[0])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, african_costs, width, label='African Green H2 (Best Location)', 
                  color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, eu_costs, width, label='European Domestic H2', 
                  color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Financing Scenario')
    ax.set_ylabel('Cost (EUR/kg H2)')
    ax.set_title('African vs European Green Hydrogen Cost Competitiveness (2030)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=eu_costs[0], color='red', linestyle='--', alpha=0.5, 
              label=f'EU Cost: {eu_costs[0]:.2f} EUR/kg')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'fig5_competitiveness.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved fig5_competitiveness.png')

def generate_all_plots():
    df = load_results()
    df_base = df[df['scenario'] == 'Moderate_Standard'].drop_duplicates(subset=['hex_id'])
    
    plot_data_overview(df_base)
    plot_lcoh_maps(df)
    plot_cost_breakdown(df)
    plot_scenario_comparison(df)
    plot_competitiveness_analysis(df)
    
    print('All visualizations generated!')

if __name__ == '__main__':
    generate_all_plots()
