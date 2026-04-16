import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('outputs/top_locations.csv')

# Cost Breakdown Calculation
def crf(w, l):
    if w == 0: return 1/l
    return (w * (1+w)**l) / ((1+w)**l - 1)

def calculate_breakdown(row, wacc):
    annual_h2_production = 8760 * 0.7 * 20
    
    # PV
    ann_capex_pv = 1.5 * 1470000.0 * crf(wacc, 20)
    opex_pv = 1.5 * 0.02 * 1470000.0
    cost_pv = (ann_capex_pv + opex_pv) / annual_h2_production
    
    # Wind
    ann_capex_wind = 1.0 * 1580000.0 * crf(wacc, 20)
    opex_wind = 1.0 * 0.03 * 1580000.0
    cost_wind = (ann_capex_wind + opex_wind) / annual_h2_production
    
    # Electrolyzer
    ann_capex_ely = 1.0 * 1250000.0 * crf(wacc, 20)
    opex_ely = 1.0 * 0.02 * 1250000.0
    cost_ely = (ann_capex_ely + opex_ely) / annual_h2_production
    
    # Water
    water_cost = 0.001 * row['ocean_dist_km'] * annual_h2_production
    cost_water = water_cost / annual_h2_production
    
    # Ammonia Synthesis
    cost_nh3 = 750 * crf(wacc, 25) + 0.015 * 750
    
    # Shipping
    shipping = 0.39
    
    # Cracking
    cost_cracking = 500 * crf(wacc, 25) + 0.02 * 500
    
    return [cost_pv, cost_wind, cost_ely, cost_water, cost_nh3, shipping, cost_cracking]

labels = ['PV', 'Wind', 'Electrolyzer', 'Water/Desal', 'Ammonia Synthesis', 'Shipping', 'Cracking']
breakdown_base = np.array([calculate_breakdown(row, 0.08) for _, row in df.iterrows()])
breakdown_derisked = np.array([calculate_breakdown(row, 0.04) for _, row in df.iterrows()])

x = np.arange(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

bottom_base = np.zeros(len(df))
bottom_derisked = np.zeros(len(df))

colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

for i, label in enumerate(labels):
    ax.bar(x - width/2, breakdown_base[:, i], width, bottom=bottom_base, color=colors[i], label=label)
    bottom_base += breakdown_base[:, i]

for i, label in enumerate(labels):
    ax.bar(x + width/2, breakdown_derisked[:, i], width, bottom=bottom_derisked, color=colors[i], alpha=0.5)
    bottom_derisked += breakdown_derisked[:, i]

ax.set_ylabel('Delivered Cost (€/kgH2)')
ax.set_title('Cost Breakdown for Top 5 Locations (Left: Base 8% WACC, Right: De-risked 4% WACC)')
ax.set_xticks(x)
ax.set_xticklabels(df['hex_id'])

# Add European benchmark line
ax.axhline(y=5.0, color='r', linestyle='--', label='European Benchmark (5.0 €/kgH2)')

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('report/images/cost_breakdown.png', dpi=300)

print("Breakdown chart saved to report/images/cost_breakdown.png")
