"""
Visualization Script: Generate Figures for Mangrove Risk Analysis
Creates publication-quality figures for the research report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("Loading results...")
df = pd.read_csv('outputs/mangrove_risk_results.csv')

with open('outputs/risk_summary.json', 'r') as f:
    summary = json.load(f)

tc_freq = np.load('outputs/tc_freq_grid.npy')
tc_max_wind = np.load('outputs/tc_max_wind_grid.npy')
lat_bins = np.load('outputs/lat_bins.npy')
lon_bins = np.load('outputs/lon_bins.npy')

print(f"Loaded {len(df)} mangrove points")

# ============================================================
# Figure 1: Global Map of Mangrove Locations with TC Frequency
# ============================================================
print("\nGenerating Figure 1: Global Map of TC Frequency...")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot TC frequency as background
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

# Only plot non-zero cells
mask = tc_freq > 0
lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

# Plot using pcolormesh
im = ax.pcolormesh(lon_bins, lat_bins, tc_freq, 
                    cmap='YlOrRd', vmin=0, vmax=tc_freq.max(),
                    shading='flat', alpha=0.8)

# Plot mangrove points
scatter = ax.scatter(df['longitude'], df['latitude'], 
                     s=1, c='black', alpha=0.3, label='Mangrove points')

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='TC Events per Year')

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Global Distribution of Tropical Cyclone Frequency (1850-2014)\nwith Mangrove Locations', fontsize=14)
ax.set_xlim(-180, 180)
ax.set_ylim(-40, 50)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/figure1_global_tc_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure1_global_tc_frequency.png")

# ============================================================
# Figure 2: Composite Risk Maps for Three SSP Scenarios
# ============================================================
print("\nGenerating Figure 2: Composite Risk Maps...")

fig, axes = plt.subplots(3, 1, figsize=(16, 18))

scenarios = ['ssp245', 'ssp370', 'ssp585']
scenario_labels = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

# Custom colormap
colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#8e44ad']
cmap = mcolors.LinearSegmentedColormap.from_list('risk', colors, N=256)

for idx, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
    ax = axes[idx]
    
    # Plot mangrove points colored by risk
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                         c=df[f'composite_risk_{scenario}'], 
                         cmap=cmap, s=2, alpha=0.6,
                         vmin=0, vmax=1)
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f'Composite Risk Index - {label}', fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 50)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, label='Composite Risk Index')
cbar.set_ticks([0, 0.2, 0.5, 0.8, 1.0])
cbar.set_ticklabels(['Low\n(0)', 'Moderate\n(0.2)', 'High\n(0.5)', 'Very High\n(0.8)', 'Extreme\n(1.0)'])

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('report/images/figure2_composite_risk_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure2_composite_risk_maps.png")

# ============================================================
# Figure 3: Risk Distribution Comparison
# ============================================================
print("\nGenerating Figure 3: Risk Distribution Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

risk_levels = ['low', 'moderate', 'high', 'very_high']
risk_colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#8e44ad']
risk_labels = ['Low', 'Moderate', 'High', 'Very High']

for idx, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
    ax = axes[idx]
    
    # Get percentages
    pcts = [summary['risk_statistics'][scenario][level] for level in risk_levels]
    
    # Create bar chart
    bars = ax.bar(risk_labels, pcts, color=risk_colors, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for bar, pct in zip(bars, pcts):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Percentage of Mangroves (%)', fontsize=10)
    ax.set_title(f'{label}', fontsize=12)
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', rotation=0)

plt.suptitle('Distribution of Mangrove Risk Levels by SSP Scenario', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure3_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure3_risk_distribution.png")

# ============================================================
# Figure 4: SLR Risk vs TC Risk Scatter
# ============================================================
print("\nGenerating Figure 4: SLR Risk vs TC Risk Scatter...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (scenario, label) in enumerate(zip(scenarios, scenario_labels)):
    ax = axes[idx]
    
    scatter = ax.scatter(df['slr_risk_' + scenario], df['tc_risk'],
                         c=df[f'composite_risk_{scenario}'],
                         cmap=cmap, s=5, alpha=0.5, vmin=0, vmax=1)
    
    # Add diagonal lines for composite risk levels
    for level, ls in [(0.2, '--'), (0.5, '-'), (0.8, '--')]:
        x = np.linspace(0, 1, 100)
        y = 2 * level - x
        valid = (y >= 0) & (y <= 1)
        ax.plot(x[valid], y[valid], ls, color='gray', alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel('Sea Level Rise Risk', fontsize=10)
    ax.set_ylabel('Tropical Cyclone Risk', fontsize=10)
    ax.set_title(f'{label}', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

plt.suptitle('Relationship Between SLR Risk and TC Risk Components', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure4_slr_vs_tc_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure4_slr_vs_tc_risk.png")

# ============================================================
# Figure 5: Regional Risk Analysis
# ============================================================
print("\nGenerating Figure 5: Regional Risk Analysis...")

# Define regions
def get_region(lat, lon):
    """Assign region based on coordinates."""
    if -30 <= lat <= 30 and -100 <= lon <= -60:
        return 'Caribbean &\nCentral America'
    elif -30 <= lat <= 30 and 60 <= lon <= 120:
        return 'South &\nSoutheast Asia'
    elif -30 <= lat <= 30 and 120 <= lon <= 180:
        return 'Western\nPacific'
    elif -30 <= lat <= 30 and -180 <= lon <= -100:
        return 'Eastern\nPacific'
    elif -30 <= lat <= 30 and -20 <= lon <= 50:
        return 'Africa &\nIndian Ocean'
    elif 30 < lat <= 50 and -130 <= lon <= -60:
        return 'North\nAmerica'
    elif -40 <= lat < -30 and 110 <= lon <= 180:
        return 'Australia &\nOceania'
    else:
        return 'Other'

df['region'] = df.apply(lambda row: get_region(row['latitude'], row['longitude']), axis=1)

# Calculate regional statistics
regional_stats = df.groupby('region').agg({
    'composite_risk_ssp245': ['mean', 'std', 'count'],
    'composite_risk_ssp370': ['mean', 'std'],
    'composite_risk_ssp585': ['mean', 'std'],
    'slr_risk_ssp245': ['mean'],
    'slr_risk_ssp370': ['mean'],
    'slr_risk_ssp585': ['mean'],
    'tc_risk': ['mean']
}).round(3)

# Filter regions with enough points
region_counts = df['region'].value_counts()
valid_regions = region_counts[region_counts > 100].index.tolist()

df_regions = df[df['region'].isin(valid_regions)]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Regional composite risk comparison
ax = axes[0, 0]
region_means = df_regions.groupby('region')[['composite_risk_ssp245', 'composite_risk_ssp370', 'composite_risk_ssp585']].mean()
region_means.plot(kind='bar', ax=ax, width=0.8)
ax.set_ylabel('Mean Composite Risk Index', fontsize=10)
ax.set_title('Regional Composite Risk by Scenario', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.legend(scenario_labels, fontsize=9)
ax.set_ylim(0, 1)

# Plot 2: Regional SLR risk
ax = axes[0, 1]
region_slr = df_regions.groupby('region')[['slr_risk_ssp245', 'slr_risk_ssp370', 'slr_risk_ssp585']].mean()
region_slr.plot(kind='bar', ax=ax, width=0.8, color=['#3498db', '#e67e22', '#e74c3c'])
ax.set_ylabel('Mean SLR Risk', fontsize=10)
ax.set_title('Regional SLR Risk by Scenario', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.legend(scenario_labels, fontsize=9)
ax.set_ylim(0, 1)

# Plot 3: Regional TC risk
ax = axes[1, 0]
region_tc = df_regions.groupby('region')['tc_risk'].mean().sort_values(ascending=True)
region_tc.plot(kind='barh', ax=ax, color='#e74c3c')
ax.set_xlabel('Mean TC Risk', fontsize=10)
ax.set_title('Regional TC Risk', fontsize=12)
ax.set_xlim(0, 1)

# Plot 4: Number of mangroves by region
ax = axes[1, 1]
region_counts_valid = df_regions['region'].value_counts()
region_counts_valid.plot(kind='barh', ax=ax, color='#2ecc71')
ax.set_xlabel('Number of Mangrove Points', fontsize=10)
ax.set_title('Mangrove Distribution by Region', fontsize=12)

plt.suptitle('Regional Analysis of Mangrove Risk', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure5_regional_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure5_regional_analysis.png")

# ============================================================
# Figure 6: Heatmap of Risk by Region and Scenario
# ============================================================
print("\nGenerating Figure 6: Heatmap of Risk by Region and Scenario...")

fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap data
heatmap_data = pd.DataFrame({
    'SSP2-4.5': df_regions.groupby('region')['composite_risk_ssp245'].mean(),
    'SSP3-7.0': df_regions.groupby('region')['composite_risk_ssp370'].mean(),
    'SSP5-8.5': df_regions.groupby('region')['composite_risk_ssp585'].mean()
})

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Composite Risk Index'})
ax.set_title('Mean Composite Risk Index by Region and Scenario', fontsize=14)
ax.set_ylabel('Region', fontsize=12)
ax.set_xlabel('SSP Scenario', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('report/images/figure6_risk_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure6_risk_heatmap.png")

# ============================================================
# Figure 7: Cumulative Risk Distribution
# ============================================================
print("\nGenerating Figure 7: Cumulative Risk Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

for scenario, label, color in zip(scenarios, scenario_labels, ['#3498db', '#e67e22', '#e74c3c']):
    # Sort risk values
    sorted_risk = np.sort(df[f'composite_risk_{scenario}'].values)
    cdf = np.arange(1, len(sorted_risk) + 1) / len(sorted_risk)
    ax.plot(sorted_risk, cdf, color=color, linewidth=2, label=label)

# Add reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Median')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

# Add risk level boundaries
ax.axvline(x=0.2, color='#2ecc71', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='#f1c40f', linestyle='--', alpha=0.3)
ax.axvline(x=0.8, color='#8e44ad', linestyle='--', alpha=0.3)

# Add risk level labels
ax.text(0.1, 1.02, 'Low', ha='center', fontsize=9, color='#2ecc71')
ax.text(0.35, 1.02, 'Moderate', ha='center', fontsize=9, color='#f1c40f')
ax.text(0.65, 1.02, 'High', ha='center', fontsize=9, color='#e74c3c')
ax.text(0.9, 1.02, 'Very High', ha='center', fontsize=9, color='#8e44ad')

ax.set_xlabel('Composite Risk Index', fontsize=12)
ax.set_ylabel('Cumulative Proportion', fontsize=12)
ax.set_title('Cumulative Distribution of Composite Risk Index', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('report/images/figure7_cumulative_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure7_cumulative_distribution.png")

# ============================================================
# Figure 8: Summary Statistics Table
# ============================================================
print("\nGenerating Figure 8: Summary Statistics...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Create summary table
table_data = []
headers = ['Metric', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

# Add rows
table_data.append(['Mean SLR Rate (mm/yr)', 
                   f"{summary['slr_rates_mean']['ssp245']:.2f}",
                   f"{summary['slr_rates_mean']['ssp370']:.2f}",
                   f"{summary['slr_rates_mean']['ssp585']:.2f}"])

table_data.append(['Mean TC Risk', 
                   f"{summary['tc_risk_mean']:.3f}",
                   f"{summary['tc_risk_mean']:.3f}",
                   f"{summary['tc_risk_mean']:.3f}"])

table_data.append(['Mean Composite Risk',
                   f"{summary['composite_risk_mean']['ssp245']:.3f}",
                   f"{summary['composite_risk_mean']['ssp370']:.3f}",
                   f"{summary['composite_risk_mean']['ssp585']:.3f}"])

table_data.append(['% Low Risk',
                   f"{summary['risk_statistics']['ssp245']['low']:.1f}%",
                   f"{summary['risk_statistics']['ssp370']['low']:.1f}%",
                   f"{summary['risk_statistics']['ssp585']['low']:.1f}%"])

table_data.append(['% Moderate Risk',
                   f"{summary['risk_statistics']['ssp245']['moderate']:.1f}%",
                   f"{summary['risk_statistics']['ssp370']['moderate']:.1f}%",
                   f"{summary['risk_statistics']['ssp585']['moderate']:.1f}%"])

table_data.append(['% High Risk',
                   f"{summary['risk_statistics']['ssp245']['high']:.1f}%",
                   f"{summary['risk_statistics']['ssp370']['high']:.1f}%",
                   f"{summary['risk_statistics']['ssp585']['high']:.1f}%"])

table_data.append(['% Very High Risk',
                   f"{summary['risk_statistics']['ssp245']['very_high']:.1f}%",
                   f"{summary['risk_statistics']['ssp370']['very_high']:.1f}%",
                   f"{summary['risk_statistics']['ssp585']['very_high']:.1f}%"])

# Create table
table = ax.table(cellText=table_data, colLabels=headers, 
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor('#3498db')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style rows
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#ecf0f1')

ax.set_title('Summary Statistics: Mangrove Risk Analysis', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('report/images/figure8_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure8_summary_table.png")

print("\n" + "=" * 60)
print("Visualization Complete!")
print("=" * 60)
print("\nAll figures saved to report/images/")
print("Files generated:")
for f in sorted(os.listdir('report/images')):
    if f.endswith('.png'):
        print(f"  - {f}")