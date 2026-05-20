#!/usr/bin/env python3
"""
Visualization for African Green Hydrogen to Europe analysis.
Generates all figures for the research report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import json

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
})
sns.set_style("whitegrid")

OUTDIR = 'report/images'
os.makedirs(OUTDIR, exist_ok=True)

# Color palettes
FIN_COLORS = {
    'high_risk': '#d62728',        # red
    'moderate_derisk': '#ff7f0e',  # orange
    'strong_derisk': '#2ca02c',    # green
}
INT_STYLES = {
    'low_rates': '-',
    'moderate_rise': '--',
    'extreme_rise': ':',
}

# Load data
results = pd.read_csv('outputs/lcoh_results_v2.csv')
eu = pd.read_csv('outputs/europe_lcoh_estimates.csv')

# ============================================================
# FIGURE 1: Map of LCOH across African sites
# ============================================================
def fig1_lcoh_map():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use low_rates, moderate_derisk as baseline
    df = results[(results['financing']=='moderate_derisk') & (results['interest']=='low_rates')]
    
    sc = ax.scatter(df['lon'], df['lat'], c=df['total_delivered'], 
                    cmap='RdYlGn_r', s=200, edgecolors='black', linewidth=0.5,
                    vmin=2, vmax=12)
    
    # Label top sites
    top5 = df.nsmallest(5, 'total_delivered')
    for _, r in top5.iterrows():
        ax.annotate(f"{r['hex_id']}\n{r['total_delivered']:.1f}€", 
                    (r['lon'], r['lat']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add Rotterdam marker
    ax.scatter([4.482], [51.909], c='blue', s=300, marker='*', 
               edgecolors='black', linewidth=0.5, zorder=10)
    ax.annotate('Rotterdam\n(Demand)', (4.482, 51.909),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='blue')
    
    # Coastline outline approximation
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Delivered LCOH from African Sites to Rotterdam\n(Moderate De-risking, Low Interest Rates, 2030)')
    
    cbar = plt.colorbar(sc, ax=ax, label='Delivered LCOH (EUR/kg H₂)')
    
    # Add country labels
    country_labels = {
        'Namibia': (17.0, -22.0),
        'Botswana': (24.0, -22.0),
        'South Africa': (22.0, -30.0),
        'Angola': (17.0, -12.0),
    }
    for name, (lon, lat) in country_labels.items():
        ax.annotate(name, (lon, lat), fontsize=9, fontstyle='italic', color='gray',
                   ha='center', va='center')
    
    ax.set_xlim(8, 28)
    ax.set_ylim(-32, -14)
    
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig1_lcoh_map.png')
    plt.close()
    print("Figure 1 saved.")

# ============================================================
# FIGURE 2: Financing scenario comparison (box plots)
# ============================================================
def fig2_financing_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_low = results[results['interest'] == 'low_rates'].copy()
    
    # Panel A: Box plot by financing
    ax = axes[0]
    order = ['high_risk', 'moderate_derisk', 'strong_derisk']
    labels = ['High Risk\n(WACC 12%)', 'Moderate\nDe-risking\n(WACC 8%)', 'Strong\nDe-risking\n(WACC 6%)']
    
    bp = ax.boxplot([df_low[df_low['financing']==o]['total_delivered'] for o in order],
                     patch_artist=True, widths=0.5)
    for patch, o in zip(bp['boxes'], order):
        patch.set_facecolor(FIN_COLORS[o])
        patch.set_alpha(0.7)
    
    # Add European benchmarks
    for i, (loc, val) in enumerate([
        ('Spain Solar', 3.0), ('North Sea Wind', 4.0), 
        ('Netherlands', 4.5), ('Germany', 5.0)
    ]):
        ax.axhline(y=val, color='blue', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(3.5, val, f'EU: {loc} ({val:.1f}€)', fontsize=8, 
                color='blue', alpha=0.7, va='center')
    
    ax.set_xticklabels(labels)
    ax.set_ylabel('Delivered LCOH (EUR/kg H₂)')
    ax.set_title('A. Impact of Financing / De-risking')
    ax.set_ylim(1, 18)
    
    # Panel B: Competitive sites count
    ax = axes[1]
    thresholds = [3.0, 4.0, 5.0, 6.0]
    x = np.arange(len(order))
    width = 0.2
    
    for i, thresh in enumerate(thresholds):
        counts = [(df_low[(df_low['financing']==o)]['total_delivered'] < thresh).sum() 
                  for o in order]
        bars = ax.bar(x + i*width, counts, width, label=f'< {thresh} EUR/kg',
                     alpha=0.8)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['High Risk', 'Moderate\nDe-risking', 'Strong\nDe-risking'])
    ax.set_ylabel('Number of Competitive Sites')
    ax.set_title('B. Number of Sites Competitive with European H₂')
    ax.legend(fontsize=8, loc='upper left')
    
    plt.suptitle('Green Hydrogen Cost Competitiveness Under Different Financing Scenarios (2030)', 
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig2_financing_comparison.png')
    plt.close()
    print("Figure 2 saved.")

# ============================================================
# FIGURE 3: Interest rate sensitivity
# ============================================================
def fig3_interest_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: LCOH vs WACC for best sites
    ax = axes[0]
    
    # Get all unique WACC values
    wacc_vals = sorted(results['wacc'].unique())
    best_sites = results.groupby('hex_id')['total_delivered'].mean().nsmallest(5).index
    
    for hex_id in best_sites:
        site_data = results[results['hex_id'] == hex_id].groupby('wacc')['total_delivered'].mean()
        ax.plot(site_data.index * 100, site_data.values, 'o-', 
                label=hex_id, linewidth=2, markersize=6)
    
    # Add European benchmark bands
    ax.axhspan(3.0, 5.0, alpha=0.1, color='blue', label='EU H₂ range (3-5 €/kg)')
    
    ax.set_xlabel('Weighted Average Cost of Capital (%)')
    ax.set_ylabel('Delivered LCOH (EUR/kg H₂)')
    ax.set_title('A. LCOH Sensitivity to WACC (Top-5 Sites)')
    ax.legend(fontsize=8, ncol=2)
    
    # Panel B: De-risking premium
    ax = axes[1]
    
    # Compute the LCOH reduction from de-risking
    hr = results[(results['financing']=='high_risk') & (results['interest']=='low_rates')]
    sd = results[(results['financing']=='strong_derisk') & (results['interest']=='low_rates')]
    
    merged = hr.merge(sd, on='hex_id', suffixes=('_hr', '_sd'))
    merged['savings'] = merged['total_delivered_hr'] - merged['total_delivered_sd']
    merged['pct_savings'] = merged['savings'] / merged['total_delivered_hr'] * 100
    
    merged_sorted = merged.sort_values('savings', ascending=False)
    
    colors = [FIN_COLORS['strong_derisk'] if s > merged['savings'].median() 
              else FIN_COLORS['moderate_derisk'] for s in merged_sorted['savings']]
    bars = ax.barh(range(len(merged_sorted)), merged_sorted['savings'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(merged_sorted)))
    ax.set_yticklabels([f"{r['hex_id']} ({r['total_delivered_sd']:.1f}€)" 
                        for _, r in merged_sorted.iterrows()], fontsize=7)
    ax.set_xlabel('Cost Reduction from De-risking (EUR/kg H₂)')
    ax.set_title('B. De-risking Benefit: High Risk → Strong De-risking')
    ax.axvline(x=merged['savings'].mean(), color='red', linestyle='--', 
               label=f'Mean: {merged["savings"].mean():.2f} EUR/kg')
    ax.legend(fontsize=9)
    
    plt.suptitle('Interest Rate and De-risking Impacts on Green Hydrogen Cost Competitiveness',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig3_interest_sensitivity.png')
    plt.close()
    print("Figure 3 saved.")

# ============================================================
# FIGURE 4: Cost breakdown waterfall
# ============================================================
def fig4_cost_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Cost breakdown for best site
    ax = axes[0]
    
    best = results[(results['financing']=='moderate_derisk') & 
                   (results['interest']=='low_rates')].nsmallest(1, 'total_delivered').iloc[0]
    
    components = [
        ('Electricity\n(Renewables)', best['elec_cost_kg']),
        ('Electrolyzer\n(CAPEX+OM)', best['elec_capex_kg'] + best['elec_om_kg']),
        ('Water', best['water_kg']),
        ('Grid\nConnection', best['grid_kg']),
        ('NH₃ Synthesis\n(Energy+CAPEX)', best['syn_total']),
        ('Truck to Port', best['truck_kg']),
        ('Maritime\nShipping', best['ship_kg']),
        ('Port\nHandling', best['port_kg']),
        ('NH₃ Cracking\n(Heat+CAPEX)', best['crack_total']),
    ]
    
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    
    # Build waterfall
    bottom = 0
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.bar(i, val, bottom=bottom, color=colors[i], edgecolor='white', linewidth=0.5)
        ax.text(i, bottom + val/2, f'{val:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += val
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_ylabel('Cost (EUR/kg H₂)')
    ax.set_title(f'A. Cost Breakdown — Best Site ({best["hex_id"]})\n'
                 f'Total: {best["total_delivered"]:.2f} EUR/kg H₂')
    ax.set_ylim(0, bottom * 1.1)
    
    # Panel B: Stacked bar for top 5 sites
    ax = axes[1]
    
    top5 = results[(results['financing']=='moderate_derisk') & 
                   (results['interest']=='low_rates')].nsmallest(5, 'total_delivered')
    
    categories = ['Production', 'NH₃ Synthesis', 'Logistics', 'Shipping', 'Cracking']
    for i, (_, r) in enumerate(top5.iterrows()):
        prod = r['lcoh_production']
        synth = r['syn_total']
        logistics = r['truck_kg'] + r['port_kg']
        shipping = r['ship_kg']
        cracking = r['crack_total']
        
        vals = [prod, synth, logistics, shipping, cracking]
        bottom = 0
        for j, (cat, val) in enumerate(zip(categories, vals)):
            ax.bar(i, val, bottom=bottom, color=plt.cm.Set2(j/len(categories)),
                   edgecolor='white', linewidth=0.5)
            bottom += val
    
    ax.set_xticks(range(len(top5)))
    ax.set_xticklabels([f"{r['hex_id']}\n{r['total_delivered']:.1f}€" 
                        for _, r in top5.iterrows()], fontsize=9)
    ax.set_ylabel('Cost (EUR/kg H₂)')
    ax.set_title('B. Cost Structure — Top-5 Sites')
    ax.legend(categories, fontsize=8, loc='upper right')
    
    plt.suptitle('Delivered Green Hydrogen Cost Breakdown (Moderate De-risking, Low Rates, 2030)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig4_cost_breakdown.png')
    plt.close()
    print("Figure 4 saved.")

# ============================================================
# FIGURE 5: Africa vs Europe competitiveness matrix
# ============================================================
def fig5_competitiveness():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Compute African best/median for each financing x interest combo
    africa_summary = results.groupby(['financing', 'interest']).agg(
        best=('total_delivered', 'min'),
        median=('total_delivered', 'median'),
        worst=('total_delivered', 'max'),
    ).reset_index()
    
    # European estimates (median of all EU locations under each interest scenario)
    eu_summary = eu.groupby('interest').agg(
        eu_min=('eu_lcoh', 'min'),
        eu_max=('eu_lcoh', 'max'),
        eu_mid=('eu_lcoh', 'median'),
    ).reset_index()
    
    x_positions = []
    y_values = []
    colors_list = []
    sizes = []
    labels = []
    
    x_idx = 0
    xtick_labels = []
    xtick_positions = []
    
    for int_k in ['low_rates', 'moderate_rise', 'extreme_rise']:
        int_name = {'low_rates': 'Low Rates', 'moderate_rise': 'Moderate Rise', 
                     'extreme_rise': 'Extreme Rise'}[int_k]
        
        # European range
        eu_row = eu_summary[eu_summary['interest'] == int_k].iloc[0]
        ax.axhspan(eu_row['eu_min'], eu_row['eu_max'], alpha=0.15, color='blue')
        ax.text(x_idx + 1.5, eu_row['eu_max'] + 0.3, f'EU H₂\n({eu_row["eu_min"]:.1f}–{eu_row["eu_max"]:.1f}€)',
                fontsize=8, color='blue', ha='center')
        
        for fin_k in ['high_risk', 'moderate_derisk', 'strong_derisk']:
            af_row = africa_summary[(africa_summary['financing']==fin_k) & 
                                     (africa_summary['interest']==int_k)].iloc[0]
            
            # Plot as point range: best to median
            ax.errorbar(x_idx, af_row['median'], 
                       yerr=[[af_row['median']-af_row['best']], [af_row['worst']-af_row['median']]],
                       fmt='o', color=FIN_COLORS[fin_k], markersize=8, 
                       capsize=5, linewidth=2, markeredgecolor='black')
            
            x_positions.append(x_idx)
            x_idx += 1
        
        # Separator
        if int_k != 'extreme_rise':
            ax.axvline(x=x_idx - 0.5, color='gray', linestyle=':', alpha=0.5)
        
        xtick_labels.append(int_name)
        xtick_positions.append(x_idx - 1.5)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FIN_COLORS['high_risk'], label='High Risk (WACC 12%)'),
        Patch(facecolor=FIN_COLORS['moderate_derisk'], label='Moderate De-risking (WACC 8%)'),
        Patch(facecolor=FIN_COLORS['strong_derisk'], label='Strong De-risking (WACC 6%)'),
        Patch(facecolor='blue', alpha=0.15, label='European H₂ Range'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    
    # Add x-axis minor ticks for financing
    fin_labels = ['HR', 'MD', 'SD'] * 3
    for i in range(9):
        ax.text(i, ax.get_ylim()[0] - 0.3, fin_labels[i], ha='center', fontsize=7, color='gray')
    
    ax.set_ylabel('Levelized Cost of Hydrogen (EUR/kg H₂)')
    ax.set_title('Africa vs Europe: Green Hydrogen Cost Competitiveness by 2030\n'
                 '(Points = median, bars = min–max range across African sites)')
    ax.set_ylim(0, 22)
    
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig5_competitiveness.png')
    plt.close()
    print("Figure 5 saved.")

# ============================================================
# FIGURE 6: Resource quality vs LCOH scatter
# ============================================================
def fig6_resource_quality():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = results[(results['financing']=='moderate_derisk') & (results['interest']=='low_rates')]
    
    # Add raw data
    raw = pd.read_csv('data/hex_final_NA_min.csv')
    df = df.merge(raw[['hex_id', 'theo_pv', 'theo_wind', 'ocean_dist_km']], on='hex_id')
    
    # Panel A: PV potential vs LCOH
    ax = axes[0]
    sc = ax.scatter(df['theo_pv'], df['total_delivered'], c=df['ocean_dist_km'], 
                    cmap='coolwarm', s=100, edgecolors='black', linewidth=0.3)
    # Annotate best
    for _, r in df.nsmallest(5, 'total_delivered').iterrows():
        ax.annotate(r['hex_id'], (r['theo_pv'], r['total_delivered']),
                   fontsize=7, fontweight='bold')
    
    ax.set_xlabel('PV Capacity Factor')
    ax.set_ylabel('Delivered LCOH (EUR/kg H₂)')
    ax.set_title('A. PV Potential vs Delivered LCOH')
    cbar = plt.colorbar(sc, ax=ax, label='Distance to Ocean (km)')
    
    # Panel B: Wind potential vs LCOH
    ax = axes[1]
    sc = ax.scatter(df['theo_wind'], df['total_delivered'], c=df['ocean_dist_km'], 
                    cmap='coolwarm', s=100, edgecolors='black', linewidth=0.3)
    for _, r in df.nsmallest(5, 'total_delivered').iterrows():
        ax.annotate(r['hex_id'], (r['theo_wind'], r['total_delivered']),
                   fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Wind Capacity Factor')
    ax.set_ylabel('Delivered LCOH (EUR/kg H₂)')
    ax.set_title('B. Wind Potential vs Delivered LCOH')
    cbar = plt.colorbar(sc, ax=ax, label='Distance to Ocean (km)')
    
    plt.suptitle('Resource Quality and Proximity to Port as Cost Drivers',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig6_resource_quality.png')
    plt.close()
    print("Figure 6 saved.")

# ============================================================
# FIGURE 7: De-risking impact by interest rate scenario
# ============================================================
def fig7_derisking_heatmap():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Pivot: rows=financing, cols=interest, values=min LCOH
    pivot = results.pivot_table(
        values='total_delivered', index='financing', columns='interest', aggfunc='min'
    )
    
    # Reorder
    pivot = pivot.loc[['high_risk', 'moderate_derisk', 'strong_derisk'],
                       ['low_rates', 'moderate_rise', 'extreme_rise']]
    
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=2, vmax=8)
    
    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 4.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=13, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(['Low Rates', 'Moderate Rise\n(+2pp WACC)', 'Extreme Rise\n(+4pp WACC)'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(['High Risk\n(WACC 12%)', 'Moderate\nDe-risking\n(WACC 8%)', 
                         'Strong\nDe-risking\n(WACC 6%)'])
    
    # Add European comparison annotations
    ax.text(2.7, 0.2, 'EU min:\n3.0 €/kg', fontsize=9, color='blue', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im, ax=ax, label='Best-Site Delivered LCOH (EUR/kg H₂)')
    ax.set_title('Best-Site Delivered LCOH: Financing × Interest Rate Matrix\n'
                 '(Green = competitive with EU production)', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig7_derisking_heatmap.png')
    plt.close()
    print("Figure 7 saved.")

# ============================================================
# FIGURE 8: Distance sensitivity
# ============================================================
def fig8_distance_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = results[(results['financing']=='moderate_derisk') & (results['interest']=='low_rates')]
    raw = pd.read_csv('data/hex_final_NA_min.csv')
    df = df.merge(raw[['hex_id', 'road_dist_km', 'ocean_dist_km', 'grid_dist_km']], on='hex_id')
    
    # Panel A: Road distance vs transport cost
    ax = axes[0]
    ax.scatter(df['road_dist_km'], df['truck_kg'], c=FIN_COLORS['moderate_derisk'], 
              s=80, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    # Fit line
    z = np.polyfit(df['road_dist_km'], df['truck_kg'], 1)
    x_fit = np.linspace(0, df['road_dist_km'].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), '--', color='darkred', linewidth=2)
    
    ax.set_xlabel('Road Distance to Port (km)')
    ax.set_ylabel('Truck Transport Cost (EUR/kg H₂)')
    ax.set_title('A. Road Distance vs Truck Transport Cost')
    
    for _, r in df.nsmallest(5, 'total_delivered').iterrows():
        ax.annotate(r['hex_id'], (r['road_dist_km'], r['truck_kg']), fontsize=7)
    
    # Panel B: Total delivered vs shipping distance
    ax = axes[1]
    ax.scatter(df['sea_dist_km'], df['total_delivered'], c=df['lcoh_production'], 
              cmap='viridis', s=100, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel('Shipping Distance to Rotterdam (km)')
    ax.set_ylabel('Total Delivered LCOH (EUR/kg H₂)')
    ax.set_title('B. Shipping Distance vs Total Delivered Cost')
    
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Production LCOH (EUR/kg)')
    
    for _, r in df.nsmallest(5, 'total_delivered').iterrows():
        ax.annotate(r['hex_id'], (r['sea_dist_km'], r['total_delivered']), fontsize=7)
    
    plt.suptitle('Logistics Cost Drivers for African Green Hydrogen Export',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTDIR}/fig8_distance_sensitivity.png')
    plt.close()
    print("Figure 8 saved.")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    fig1_lcoh_map()
    fig2_financing_comparison()
    fig3_interest_sensitivity()
    fig4_cost_breakdown()
    fig5_competitiveness()
    fig6_resource_quality()
    fig7_derisking_heatmap()
    fig8_distance_sensitivity()
    
    print("\nAll figures generated successfully!")
