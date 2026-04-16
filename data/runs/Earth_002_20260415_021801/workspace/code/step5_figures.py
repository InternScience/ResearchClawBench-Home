"""
Step 5: Generate all figures for the report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import json
import os

os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_csv('outputs/mangrove_composite_risk.csv')

# Load summaries
with open('outputs/composite_risk_summary.json') as f:
    risk_summary = json.load(f)
with open('outputs/slr_summary.json') as f:
    slr_summary = json.load(f)
with open('outputs/tc_summary.json') as f:
    tc_summary = json.load(f)

# Color scheme for risk categories
risk_colors = {
    'Low': '#2ca02c',       # green
    'Moderate': '#ffcc00',   # yellow
    'High': '#ff7f0e',       # orange
    'Very High': '#d62728',  # red
}

scenarios = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
scenario_labels = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

# ============================================================
# Figure 1: Global Composite Risk Index Map (SSP5-8.5)
# ============================================================
print("Creating Figure 1: Global CRI map...")

fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={'projection': None})

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    cri_col = f'cri_{scenario}'
    cat_col = f'cri_cat_{scenario}'
    
    # Plot by category with colors
    for cat in ['Low', 'Moderate', 'High', 'Very High']:
        mask = df[cat_col] == cat
        if mask.sum() > 0:
            # Subsample for plotting efficiency
            subset = df[mask]
            if len(subset) > 10000:
                subset = subset.sample(10000, random_state=42)
            ax.scatter(subset['lon'], subset['lat'], c=risk_colors[cat], 
                      s=0.5, alpha=0.6, label=cat, rasterized=True)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    if idx == 0:
        legend_elements = [Patch(facecolor=risk_colors[c], label=c) for c in risk_colors]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8, title='Risk Category')

fig.suptitle('Global Composite Risk Index for Mangroves by SSP Scenario', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig1_global_cri_map.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig1_global_cri_map.png")

# ============================================================
# Figure 2: SLR Rate Distribution by Scenario
# ============================================================
print("Creating Figure 2: SLR rate distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    slr_col = f'slr_rate_{scenario}'
    rates = df[slr_col].values
    
    ax.hist(rates, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=4, color='orange', linestyle='--', linewidth=2, label='4 mm/yr (deficit likely)')
    ax.axvline(x=7, color='red', linestyle='--', linewidth=2, label='7 mm/yr (deficit highly likely)')
    ax.set_xlabel('SLR Rate (mm/yr)', fontsize=11)
    ax.set_ylabel('Number of Mangrove Points', fontsize=11)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Add text with key stats
    pct4 = np.mean(rates >= 4) * 100
    pct7 = np.mean(rates >= 7) * 100
    ax.text(0.95, 0.95, f'≥4 mm/yr: {pct4:.1f}%\n≥7 mm/yr: {pct7:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Distribution of Sea Level Rise Rates at Mangrove Locations (2020-2100)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig2_slr_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig2_slr_distribution.png")

# ============================================================
# Figure 3: TC Frequency Map
# ============================================================
print("Creating Figure 3: TC frequency map...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Major TC frequency at mangrove points
ax = axes[0]
freq = df['tc_freq_major_tc'].values
sc = ax.scatter(df['lon'], df['lat'], c=freq, s=0.5, cmap='YlOrRd', 
                alpha=0.6, rasterized=True, vmin=0, vmax=0.3)
plt.colorbar(sc, ax=ax, label='Major TC Frequency (events/yr)')
ax.set_xlim(-180, 180)
ax.set_ylim(-40, 35)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Historical Major TC (Cat 3+) Frequency', fontweight='bold')
ax.set_aspect('equal')

# Max wind at mangrove points
ax = axes[1]
wind = df['tc_maxwind_all_tc'].values
sc = ax.scatter(df['lon'], df['lat'], c=wind, s=0.5, cmap='YlOrRd',
                alpha=0.6, rasterized=True, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label='Max Wind Speed (m/s)')
ax.set_xlim(-180, 180)
ax.set_ylim(-40, 35)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Historical Maximum TC Wind Speed', fontweight='bold')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('report/images/fig3_tc_frequency_map.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig3_tc_frequency_map.png")

# ============================================================
# Figure 4: Risk Category Distribution by Scenario
# ============================================================
print("Creating Figure 4: Risk category distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    cat_col = f'cri_cat_{scenario}'
    cats = df[cat_col].value_counts().reindex(['Low', 'Moderate', 'High', 'Very High'])
    
    bars = ax.bar(cats.index, cats.values, color=[risk_colors[c] for c in cats.index], edgecolor='white')
    ax.set_xlabel('Risk Category', fontsize=11)
    ax.set_ylabel('Number of Mangrove Points', fontsize=11)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    
    # Add percentage labels
    for bar, val in zip(bars, cats.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle('Composite Risk Index Distribution by Scenario', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig4_risk_category_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig4_risk_category_distribution.png")

# ============================================================
# Figure 5: SLR vs TC Risk Scatter
# ============================================================
print("Creating Figure 5: SLR vs TC risk scatter...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    slr_col = f'slr_risk_{scenario}'
    tc_col = f'tc_risk_{scenario}'
    cri_col = f'cri_{scenario}'
    
    # Subsample for plotting
    sample = df.sample(min(20000, len(df)), random_state=42)
    
    sc = ax.scatter(sample[slr_col], sample[tc_col], c=sample[cri_col],
                    s=1, cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Composite Risk Index')
    
    # Add threshold lines
    ax.axhline(y=0.4, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('SLR Risk Score', fontsize=11)
    ax.set_ylabel('TC Risk Score', fontsize=11)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

fig.suptitle('SLR Risk vs TC Risk at Mangrove Locations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig5_slr_tc_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig5_slr_tc_scatter.png")

# ============================================================
# Figure 6: Regional Risk Comparison
# ============================================================
print("Creating Figure 6: Regional risk comparison...")

# Define regions based on longitude/latitude
def assign_region(lat, lon):
    if lat > 23.5:  # Northern subtropics/extratropics
        if lon > -30 and lon < 60:
            return 'W. Africa & Med'
        elif lon >= 60 and lon < 150:
            return 'S. Asia & Pacific'
        else:
            return 'Caribbean & Americas'
    elif lon >= -80 and lon <= -30:
        return 'Americas'
    elif lon > -30 and lon < 50:
        return 'W. & E. Africa'
    elif lon >= 50 and lon < 100:
        return 'Middle East & S. Asia'
    elif lon >= 100 and lon < 150:
        return 'SE Asia & Australia'
    elif lon >= 150:
        return 'Pacific Islands'
    else:
        return 'Other'

df['region'] = df.apply(lambda r: assign_region(r['lat'], r['lon']), axis=1)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    cri_col = f'cri_{scenario}'
    
    region_order = df.groupby('region')[cri_col].mean().sort_values(ascending=False).index
    
    region_data = [df[df['region'] == r][cri_col].values for r in region_order]
    
    bp = ax.boxplot(region_data, labels=region_order, vert=True, patch_artist=True)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(region_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Composite Risk Index', fontsize=11)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High risk threshold')
    ax.legend(fontsize=9)

fig.suptitle('Regional Comparison of Composite Risk Index', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig6_regional_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig6_regional_comparison.png")

# ============================================================
# Figure 7: Scenario Comparison - At-Risk Area
# ============================================================
print("Creating Figure 7: Scenario comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stacked bar chart by category
categories = ['Low', 'Moderate', 'High', 'Very High']
x = np.arange(len(scenarios))
width = 0.6

bottom = np.zeros(len(scenarios))
for cat in categories:
    vals = [risk_summary[s]['cri_categories'][cat] / 1000 for s in scenarios]
    ax1.bar(x, vals, width, bottom=bottom, label=cat, color=risk_colors[cat], edgecolor='white')
    bottom += vals

ax1.set_xlabel('Scenario', fontsize=11)
ax1.set_ylabel('Number of Mangrove Points (thousands)', fontsize=11)
ax1.set_title('Risk Category Distribution by Scenario', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenario_labels)
ax1.legend()

# Mean CRI comparison
slr_means = [risk_summary[s]['slr_risk_mean'] for s in scenarios]
tc_means = [risk_summary[s]['tc_risk_mean'] for s in scenarios]
cri_means = [risk_summary[s]['cri_mean'] for s in scenarios]

x = np.arange(len(scenarios))
width = 0.25

ax2.bar(x - width, slr_means, width, label='SLR Risk', color='steelblue')
ax2.bar(x, tc_means, width, label='TC Risk', color='coral')
ax2.bar(x + width, cri_means, width, label='Composite Risk', color='purple')

ax2.set_xlabel('Scenario', fontsize=11)
ax2.set_ylabel('Mean Risk Score', fontsize=11)
ax2.set_title('Mean Risk Scores by Scenario', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(scenario_labels)
ax2.legend()
ax2.set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig('report/images/fig7_scenario_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig7_scenario_comparison.png")

# ============================================================
# Figure 8: SLR Rate Map
# ============================================================
print("Creating Figure 8: SLR rate map...")

fig, axes = plt.subplots(1, 3, figsize=(24, 5))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    slr_col = f'slr_rate_{scenario}'
    
    sample = df.sample(min(30000, len(df)), random_state=42)
    sc = ax.scatter(sample['lon'], sample['lat'], c=sample[slr_col],
                    s=0.5, cmap='YlOrRd', alpha=0.6, vmin=0, vmax=15, rasterized=True)
    plt.colorbar(sc, ax=ax, label='SLR Rate (mm/yr)')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

fig.suptitle('Sea Level Rise Rates at Mangrove Locations (2020-2100 Median)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig8_slr_rate_map.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig8_slr_rate_map.png")

# ============================================================
# Figure 9: Latitudinal Risk Profile
# ============================================================
print("Creating Figure 9: Latitudinal risk profile...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Lat bins
lat_bins = np.arange(-35, 35, 2)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

ax = axes[0]
for scenario in scenarios:
    cri_col = f'cri_{scenario}'
    mean_cri = []
    for i in range(len(lat_bins)-1):
        mask = (df['lat'] >= lat_bins[i]) & (df['lat'] < lat_bins[i+1])
        if mask.sum() > 10:
            mean_cri.append(df.loc[mask, cri_col].mean())
        else:
            mean_cri.append(np.nan)
    ax.plot(lat_centers, mean_cri, 'o-', label=scenario, markersize=4)

ax.set_xlabel('Latitude (°)', fontsize=11)
ax.set_ylabel('Mean Composite Risk Index', fontsize=11)
ax.set_title('Latitudinal Risk Profile', fontweight='bold')
ax.legend()
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)

# Mangrove density by latitude
ax = axes[1]
counts, _, _ = ax.hist(df['lat'], bins=lat_bins, color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel('Latitude (°)', fontsize=11)
ax.set_ylabel('Number of Mangrove Points', fontsize=11)
ax.set_title('Mangrove Distribution by Latitude', fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/fig9_latitudinal_profile.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig9_latitudinal_profile.png")

# ============================================================
# Save regional summary
# ============================================================
regional_summary = {}
for scenario in scenarios:
    cri_col = f'cri_{scenario}'
    cat_col = f'cri_cat_{scenario}'
    
    reg = df.groupby('region').agg(
        n_points=('region', 'count'),
        mean_cri=(cri_col, 'mean'),
        pct_high_vh=(cat_col, lambda x: (x.isin(['High', 'Very High'])).mean() * 100),
    ).to_dict('index')
    regional_summary[scenario] = reg

with open('outputs/regional_summary.json', 'w') as f:
    json.dump(regional_summary, f, indent=2, default=str)
print("Saved regional summary to outputs/regional_summary.json")

# Save updated dataframe with region
df.to_csv('outputs/mangrove_composite_risk_with_region.csv', index=False)
print("Saved updated data with region column")

print("\nAll figures generated successfully!")
