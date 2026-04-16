"""
Step 6: Additional figures - Country-level risk and ecosystem service figures.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_csv('outputs/mangrove_composite_risk_with_region.csv')
with open('outputs/country_risk_summary.json') as f:
    country_risk = json.load(f)

# ============================================================
# Figure 10: Top 20 at-risk countries by CRI (SSP5-8.5)
# ============================================================
print("Creating Figure 10: Top at-risk countries...")

countries_ssp585 = country_risk['SSP5-8.5']
# Filter to countries with at least 50 mangrove points
filtered = [c for c in countries_ssp585 if c['n_points'] >= 50]
filtered.sort(key=lambda x: x['mean_cri'], reverse=True)
top20 = filtered[:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Mean CRI
names = [c['country_name'][:20] for c in top20]
cri_vals = [c['mean_cri'] for c in top20]
colors = ['#d62728' if v >= 0.6 else '#ff7f0e' if v >= 0.5 else '#ffcc00' if v >= 0.4 else '#2ca02c' for v in cri_vals]

ax1.barh(range(len(names)), cri_vals, color=colors, edgecolor='white')
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlabel('Mean Composite Risk Index')
ax1.set_title('Top 20 Countries by Mean CRI\n(SSP5-8.5)', fontweight='bold')
ax1.invert_yaxis()
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

# % High+Very High
pct_vals = [c['pct_high_vh'] for c in top20]
colors2 = ['#d62728' if v >= 80 else '#ff7f0e' if v >= 50 else '#ffcc00' if v >= 30 else '#2ca02c' for v in pct_vals]

ax2.barh(range(len(names)), pct_vals, color=colors2, edgecolor='white')
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=9)
ax2.set_xlabel('% Mangrove Points at High or Very High Risk')
ax2.set_title('Top 20 Countries by % High+Very High Risk\n(SSP5-8.5)', fontweight='bold')
ax2.invert_yaxis()
ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/fig10_top_countries.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig10_top_countries.png")

# ============================================================
# Figure 11: Ecosystem services at risk
# ============================================================
print("Creating Figure 11: Ecosystem services at risk...")

with open('outputs/ecosystem_service_risk.json') as f:
    es_risk = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
categories = ['Low', 'Moderate', 'High', 'Very High']
risk_colors = {'Low': '#2ca02c', 'Moderate': '#ffcc00', 'High': '#ff7f0e', 'Very High': '#d62728'}

# Population at risk by category and scenario
ax = axes[0, 0]
for idx, scenario in enumerate(scenarios):
    vals = [es_risk[scenario]['by_category'][cat]['total_risk_pop'] / 1e6 for cat in categories]
    x = np.arange(len(categories)) + idx * 0.25
    ax.bar(x, vals, 0.25, label=scenario, alpha=0.8)
ax.set_xticks(np.arange(len(categories)) + 0.25)
ax.set_xticklabels(categories)
ax.set_ylabel('Population at Risk (millions)')
ax.set_title('Population at Risk by Category', fontweight='bold')
ax.legend(fontsize=8)

# Capital stock at risk
ax = axes[0, 1]
for idx, scenario in enumerate(scenarios):
    vals = [es_risk[scenario]['by_category'][cat]['total_risk_stock_usd'] / 1e9 for cat in categories]
    x = np.arange(len(categories)) + idx * 0.25
    ax.bar(x, vals, 0.25, label=scenario, alpha=0.8)
ax.set_xticks(np.arange(len(categories)) + 0.25)
ax.set_xticklabels(categories)
ax.set_ylabel('Capital Stock at Risk (billion USD)')
ax.set_title('Capital Stock at Risk by Category', fontweight='bold')
ax.legend(fontsize=8)

# At-risk area by scenario
ax = axes[1, 0]
area_hv = [es_risk[s]['high_very_high']['area_km2'] / 1000 for s in scenarios]
area_total = [150.0] * 3  # total in thousands km2
area_safe = [t - h for t, h in zip(area_total, area_hv)]

ax.bar(scenarios, area_safe, label='Low+Moderate Risk', color='#2ca02c', alpha=0.7)
ax.bar(scenarios, area_hv, bottom=area_safe, label='High+Very High Risk', color='#d62728', alpha=0.7)
ax.set_ylabel('Mangrove Area (×1000 km²)')
ax.set_title('At-Risk Mangrove Area by Scenario', fontweight='bold')
ax.legend()

# Population benefiting at risk
ax = axes[1, 1]
for idx, scenario in enumerate(scenarios):
    vals = [es_risk[scenario]['by_category'][cat]['total_ben_pop'] / 1e6 for cat in categories]
    x = np.arange(len(categories)) + idx * 0.25
    ax.bar(x, vals, 0.25, label=scenario, alpha=0.8)
ax.set_xticks(np.arange(len(categories)) + 0.25)
ax.set_xticklabels(categories)
ax.set_ylabel('Benefiting Population (millions)')
ax.set_title('Population Benefiting from Mangroves by Risk Category', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('report/images/fig11_ecosystem_services.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig11_ecosystem_services.png")

# ============================================================
# Figure 12: SLR risk vs TC risk contribution
# ============================================================
print("Creating Figure 12: Risk component contribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    slr_col = f'slr_risk_{scenario}'
    tc_col = f'tc_risk_{scenario}'
    cri_col = f'cri_{scenario}'
    
    # Classify by dominant risk driver
    slr_dominant = df[slr_col] > df[tc_col]
    tc_dominant = df[tc_col] > df[slr_col]
    equal = df[slr_col] == df[tc_col]
    
    n_slr = slr_dominant.sum()
    n_tc = tc_dominant.sum()
    n_eq = equal.sum()
    
    labels = ['SLR-Dominant', 'TC-Dominant', 'Equal']
    sizes = [n_slr, n_tc, n_eq]
    colors_pie = ['steelblue', 'coral', 'gray']
    
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')

fig.suptitle('Dominant Risk Driver at Mangrove Locations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig12_risk_contribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig12_risk_contribution.png")

print("\nAll additional figures generated!")
