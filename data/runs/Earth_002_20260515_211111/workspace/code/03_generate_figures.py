#!/usr/bin/env python3
"""
Generate publication-quality figures for the mangrove composite risk analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

print("=== Loading data ===")
gdf = gpd.read_parquet('outputs/mangrove_risk_final.parquet')
print(f"Loaded {len(gdf)} mangrove points")

# ============================================================
# Figure 1: Global overview - SLR rates and TC frequency maps
# ============================================================
print("\n=== Figure 1: Global overview maps ===")

fig, axes = plt.subplots(2, 2, figsize=(18, 12), subplot_kw={'projection': ccrs.Robinson()})

# Panel a: SLR rates SSP5-8.5 2080-2100
ax = axes[0, 0]
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
ax.coastlines(linewidth=0.3)

slr_vals = gdf['ssp5_8_5_slr_2080_2100'].values
sc = ax.scatter(gdf.geometry.x.values, gdf.geometry.y.values,
                c=slr_vals, s=0.5, cmap='YlOrRd', 
                norm=mcolors.Normalize(vmin=0, vmax=20),
                transform=ccrs.PlateCarree(), alpha=0.7)
plt.colorbar(sc, ax=ax, shrink=0.5, label='RSLR Rate (mm/yr)', orientation='horizontal', pad=0.05)
ax.set_title('(a) Projected RSLR Rate 2080-2100 (SSP5-8.5)', fontweight='bold')

# Panel b: TC major cyclone frequency
ax = axes[0, 1]
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
ax.coastlines(linewidth=0.3)

tc_vals = np.log10(gdf['tc_major_freq'].values + 0.001)
sc = ax.scatter(gdf.geometry.x.values, gdf.geometry.y.values,
                c=tc_vals, s=0.5, cmap='Purples', 
                vmin=-3, vmax=0,
                transform=ccrs.PlateCarree(), alpha=0.7)
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, label='log10(Annual Major TC Frequency)', 
                     orientation='horizontal', pad=0.05)
ax.set_title('(b) Historical Major TC Frequency (Cat 3-5)', fontweight='bold')

# Panel c: Composite Risk Index SSP5-8.5
ax = axes[1, 0]
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
ax.coastlines(linewidth=0.3)

cri_colors = {'Very Low': '#2ecc71', 'Low': '#27ae60', 'Moderate': '#f39c12', 
              'High': '#e74c3c', 'Very High': '#8e44ad'}
cri_cats = gdf['ssp5_8_5_cri_cat'].values
for cat, color in cri_colors.items():
    mask = cri_cats == cat
    if mask.sum() > 0:
        ax.scatter(gdf.geometry.x.values[mask], gdf.geometry.y.values[mask],
                   c=color, s=0.8, transform=ccrs.PlateCarree(), alpha=0.8, label=cat)
ax.legend(loc='lower left', fontsize=7, markerscale=3, framealpha=0.8)
ax.set_title('(c) Composite Risk Index (SSP5-8.5)', fontweight='bold')

# Panel d: Risk comparison across SSPs
ax = axes[1, 1]
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
ax.coastlines(linewidth=0.3)

# Highlight high-risk areas that are consistent across all three SSPs
cri_ssp245 = gdf['ssp2_4_5_cri'].values
cri_ssp370 = gdf['ssp3_7_0_cri'].values
cri_ssp585 = gdf['ssp5_8_5_cri'].values

high_all = (cri_ssp245 >= 4) & (cri_ssp370 >= 4) & (cri_ssp585 >= 4)
high_any = (cri_ssp245 >= 3) | (cri_ssp370 >= 3) | (cri_ssp585 >= 3)

# Plot
ax.scatter(gdf.geometry.x.values[~high_all], gdf.geometry.y.values[~high_all],
           c='gray', s=0.3, transform=ccrs.PlateCarree(), alpha=0.3)
ax.scatter(gdf.geometry.x.values[high_all], gdf.geometry.y.values[high_all],
           c='#8e44ad', s=1.5, transform=ccrs.PlateCarree(), alpha=0.9, 
           label='Very High across all SSPs')
ax.legend(loc='lower left', fontsize=8, markerscale=3)
ax.set_title('(d) Consistent Very High Risk (All SSPs)', fontweight='bold')

plt.suptitle('Global Mangrove Risk Assessment: Combined SLR and Tropical Cyclone Hazards', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/figure1_global_overview.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Risk decomposition and component analysis
# ============================================================
print("\n=== Figure 2: Risk decomposition ===")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel a: Distribution of SLR rates by SSP
ax = axes[0, 0]
for ssp_label, ssp_col, color in [
    ('SSP2-4.5', 'ssp2_4_5_slr_2080_2100', '#3498db'),
    ('SSP3-7.0', 'ssp3_7_0_slr_2080_2100', '#e67e22'),
    ('SSP5-8.5', 'ssp5_8_5_slr_2080_2100', '#e74c3c'),
]:
    ax.hist(gdf[ssp_col], bins=50, alpha=0.5, label=ssp_label, color=color, density=True)
ax.axvline(x=4, color='orange', linestyle='--', linewidth=1.5, label='4 mm/yr (likely retreat)')
ax.axvline(x=7, color='red', linestyle='--', linewidth=1.5, label='7 mm/yr (highly likely retreat)')
ax.set_xlabel('RSLR Rate (mm/yr)')
ax.set_ylabel('Density')
ax.set_title('(a) RSLR Rate Distribution (2080-2100)', fontweight='bold')
ax.legend(fontsize=7)

# Panel b: TC frequency components
ax = axes[0, 1]
tc_data = {
    'Total (Cat 1-5)': gdf['tc_total_freq'],
    'Major (Cat 3-5)': gdf['tc_major_freq'],
    'Intense (Cat 4-5)': gdf['tc_intense_freq'],
}
box_data = [tc_data[k] for k in tc_data]
bp = ax.boxplot(box_data, labels=list(tc_data.keys()), patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Annual Frequency')
ax.set_title('(b) TC Frequency by Category', fontweight='bold')
ax.set_yscale('log')

# Panel c: SLR risk categories
ax = axes[0, 2]
risk_dist = pd.DataFrame({
    'SSP2-4.5': gdf['ssp2_4_5_slr_risk_cat'].value_counts(),
    'SSP3-7.0': gdf['ssp3_7_0_slr_risk_cat'].value_counts(),
    'SSP5-8.5': gdf['ssp5_8_5_slr_risk_cat'].value_counts(),
}).fillna(0)
risk_dist = risk_dist.reindex(['Low', 'Moderate', 'High'])
risk_dist_pct = risk_dist / risk_dist.sum() * 100
colors_slr = ['#2ecc71', '#f39c12', '#e74c3c']
risk_dist_pct.T.plot(kind='bar', stacked=True, ax=ax, color=colors_slr)
ax.set_ylabel('Percentage of Mangrove Sites')
ax.set_title('(c) SLR Risk Classification', fontweight='bold')
ax.legend(loc='upper left', fontsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Panel d: Composite risk by SSP
ax = axes[1, 0]
cri_dist = pd.DataFrame({
    'SSP2-4.5': gdf['ssp2_4_5_cri_cat'].value_counts(),
    'SSP3-7.0': gdf['ssp3_7_0_cri_cat'].value_counts(),
    'SSP5-8.5': gdf['ssp5_8_5_cri_cat'].value_counts(),
}).fillna(0)
cri_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
cri_dist = cri_dist.reindex(cri_order)
cri_dist_pct = cri_dist / cri_dist.sum() * 100
cri_colors_list = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
cri_dist_pct.T.plot(kind='bar', stacked=True, ax=ax, color=cri_colors_list)
ax.set_ylabel('Percentage of Mangrove Sites')
ax.set_title('(d) Composite Risk Index Distribution', fontweight='bold')
ax.legend(loc='upper left', fontsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Panel e: Risk matrix (SLR vs TC)
ax = axes[1, 1]
# Create a 2D histogram
slr_score = gdf['ssp5_8_5_slr_risk_score'].values
tc_score = gdf['tc_projected_risk_score'].values
hist2d, xedges, yedges = np.histogram2d(slr_score, tc_score, bins=[4, 5], 
                                          range=[[-0.5, 3.5], [-0.5, 4.5]])
im = ax.imshow(hist2d.T, origin='lower', aspect='auto', cmap='YlOrRd',
               extent=[-0.5, 3.5, -0.5, 4.5])
# Add text annotations
for i in range(4):
    for j in range(5):
        if hist2d[i, j] > 0:
            ax.text(i, j, f'{hist2d[i,j]:.0f}', ha='center', va='center', 
                   fontsize=7, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Low', 'Moderate', 'High'])
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['None', 'Low', 'Moderate', 'High'])
ax.set_xlabel('SLR Risk')
ax.set_ylabel('TC Risk')
ax.set_title('(e) Risk Matrix (SSP5-8.5)', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel f: Country-level risk for top countries
ax = axes[1, 2]
country_risk = pd.read_csv('outputs/country_risk_SSP5_8_5.csv')
top_countries = country_risk[country_risk['n_points'] >= 100].nlargest(15, 'mean_cri')

colors_bar = [cri_colors_list[min(int(s), 4)] for s in np.clip(np.floor(top_countries['mean_cri']).astype(int), 0, 4)]
ax.barh(range(len(top_countries)), top_countries['mean_cri'], color=colors_bar)
ax.set_yticks(range(len(top_countries)))
ax.set_yticklabels(top_countries['Country'], fontsize=7)
ax.set_xlabel('Mean Composite Risk Index')
ax.set_title('(f) Top Countries by Risk (SSP5-8.5)', fontweight='bold')
ax.invert_yaxis()

plt.suptitle('Risk Decomposition and Component Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure2_risk_decomposition.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Regional focus maps
# ============================================================
print("\n=== Figure 3: Regional focus maps ===")

# Define regions of interest
regions = {
    'Southeast Asia': (95, 145, -12, 25),
    'Caribbean & Central America': (-95, -55, 5, 30),
    'South Asia & Bay of Bengal': (65, 100, 0, 25),
    'Oceania & SW Pacific': (130, 180, -35, 5),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for ax, (region_name, (lon_min, lon_max, lat_min, lat_max)) in zip(axes.flat, regions.items()):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    
    # Filter points in region
    mask = ((gdf.geometry.x >= lon_min) & (gdf.geometry.x <= lon_max) &
            (gdf.geometry.y >= lat_min) & (gdf.geometry.y <= lat_max))
    region_gdf = gdf[mask]
    
    if len(region_gdf) > 0:
        for cat, color in cri_colors.items():
            cmask = region_gdf['ssp5_8_5_cri_cat'] == cat
            if cmask.sum() > 0:
                ax.scatter(region_gdf.geometry.x.values[cmask], 
                          region_gdf.geometry.y.values[cmask],
                          c=color, s=3, alpha=0.8, label=cat if ax == axes.flat[0] else '')
    
    if ax == axes.flat[0]:
        ax.legend(loc='lower left', fontsize=6, markerscale=2)
    
    # Add stats
    if len(region_gdf) > 0:
        high_pct = ((region_gdf['ssp5_8_5_cri_cat'] == 'High') | 
                    (region_gdf['ssp5_8_5_cri_cat'] == 'Very High')).mean() * 100
        ax.set_title(f'{region_name}\n{len(region_gdf)} sites, {high_pct:.1f}% High/Very High Risk', 
                    fontweight='bold', fontsize=10)

plt.suptitle('Regional Mangrove Risk Patterns (SSP5-8.5)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure3_regional_maps.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: Country-level risk and ecosystem services
# ============================================================
print("\n=== Figure 4: Country risk vs ecosystem services ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel a: Country risk bar chart
ax = axes[0, 0]
country_risk_all = pd.read_csv('outputs/country_risk_SSP5_8_5.csv')
top30 = country_risk_all[country_risk_all['n_points'] >= 50].nlargest(20, 'mean_cri')
colors_bar = [cri_colors_list[min(int(s), 4)] for s in np.clip(np.floor(top30['mean_cri']).astype(int), 0, 4)]
ax.barh(range(len(top30)), top30['mean_cri'], color=colors_bar, edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['Country'], fontsize=7)
ax.set_xlabel('Mean Composite Risk Index')
ax.set_title('(a) Top 20 Countries by Mangrove Risk', fontweight='bold')
for i, (v, pct) in enumerate(zip(top30['mean_cri'], top30['high_risk_pct'])):
    ax.text(v + 0.05, i, f'{pct:.0f}%', va='center', fontsize=6, fontweight='bold')
ax.invert_yaxis()

# Panel b: SLR rates latitudinal profile
ax = axes[0, 1]
lat_bins = np.arange(-40, 35, 5)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
slr_means = []
slr_stds = []
for i in range(len(lat_bins)-1):
    mask = (gdf.geometry.y >= lat_bins[i]) & (gdf.geometry.y < lat_bins[i+1])
    if mask.sum() > 0:
        slr_means.append(gdf.loc[mask, 'ssp5_8_5_slr_2080_2100'].mean())
        slr_stds.append(gdf.loc[mask, 'ssp5_8_5_slr_2080_2100'].std())
    else:
        slr_means.append(np.nan)
        slr_stds.append(np.nan)

slr_means = np.array(slr_means)
slr_stds = np.array(slr_stds)

ax.fill_between(lat_centers, slr_means - slr_stds, slr_means + slr_stds, alpha=0.3, color='#e74c3c')
ax.plot(lat_centers, slr_means, 'o-', color='#e74c3c', linewidth=2)
ax.axhline(y=7, color='red', linestyle='--', alpha=0.7, label='7 mm/yr threshold')
ax.axhline(y=4, color='orange', linestyle='--', alpha=0.7, label='4 mm/yr threshold')
ax.set_xlabel('Latitude')
ax.set_ylabel('Mean RSLR Rate (mm/yr)')
ax.set_title('(b) RSLR Rate by Latitude (SSP5-8.5)', fontweight='bold')
ax.legend(fontsize=7)

# Panel c: Risk vs TC frequency scatter
ax = axes[1, 0]
sample = gdf.sample(min(5000, len(gdf)), random_state=42)
sc = ax.scatter(np.log10(sample['tc_major_freq'] + 0.001), 
                sample['ssp5_8_5_slr_2080_2100'],
                c=sample['ssp5_8_5_cri'], s=3, alpha=0.5, cmap='RdYlGn_r', vmin=0, vmax=5)
ax.set_xlabel('log10(Major TC Frequency + 0.001)')
ax.set_ylabel('RSLR Rate 2080-2100 (mm/yr)')
ax.set_title('(c) SLR vs TC Hazard Space', fontweight='bold')
plt.colorbar(sc, ax=ax, label='Composite Risk Index')

# Panel d: Risk by region bar chart
ax = axes[1, 1]
regions_broad = {
    'Southeast Asia': (95, 145, -12, 25),
    'South Asia': (65, 100, 0, 30),
    'Caribbean/Central America': (-95, -55, 5, 30),
    'Oceania': (130, 180, -35, 5),
    'West Africa': (-20, 15, -5, 15),
    'East Africa': (30, 55, -30, 5),
    'South America (Atlantic)': (-55, -30, -25, 5),
    'North America (Gulf)': (-100, -80, 20, 35),
    'Middle East': (35, 60, 15, 35),
}

region_stats = []
for rname, (lon_min, lon_max, lat_min, lat_max) in regions_broad.items():
    mask = ((gdf.geometry.x >= lon_min) & (gdf.geometry.x <= lon_max) &
            (gdf.geometry.y >= lat_min) & (gdf.geometry.y <= lat_max))
    rgdf = gdf[mask]
    if len(rgdf) > 0:
        region_stats.append({
            'Region': rname,
            'N': len(rgdf),
            'Mean CRI': rgdf['ssp5_8_5_cri'].mean(),
            'Mean SLR': rgdf['ssp5_8_5_slr_2080_2100'].mean(),
            'High Risk %': ((rgdf['ssp5_8_5_cri_cat'] == 'High') | (rgdf['ssp5_8_5_cri_cat'] == 'Very High')).mean() * 100,
            'Mean TC Freq': rgdf['tc_major_freq'].mean(),
        })

region_df = pd.DataFrame(region_stats).sort_values('Mean CRI', ascending=True)
colors_reg = [cri_colors_list[min(int(s), 4)] for s in np.clip(np.floor(region_df['Mean CRI']).astype(int), 0, 4)]
ax.barh(range(len(region_df)), region_df['Mean CRI'], color=colors_reg, edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(region_df)))
ax.set_yticklabels(region_df['Region'], fontsize=8)
ax.set_xlabel('Mean Composite Risk Index')
ax.set_title('(d) Regional Risk Comparison (SSP5-8.5)', fontweight='bold')

plt.suptitle('Country and Regional Risk Profiles', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure4_country_risk.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: TC-SLR synergy and adaptation implications
# ============================================================
print("\n=== Figure 5: Synergy and policy implications ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel a: Ecosystem services at risk - conceptual
ax = axes[0, 0]
# Categorize: high risk has both high SLR AND high TC
high_slr = gdf['ssp5_8_5_slr_risk_score'] >= 2
high_tc = gdf['tc_projected_risk_score'] >= 2

categories = {
    'SLR Only Risk': high_slr & ~high_tc,
    'TC Only Risk': ~high_slr & high_tc,
    'Both Hazards': high_slr & high_tc,
    'Low Risk': ~high_slr & ~high_tc,
}
cat_counts = {k: v.sum() for k, v in categories.items()}
colors_pie = ['#f39c12', '#3498db', '#8e44ad', '#95a5a6']
wedges, texts, autotexts = ax.pie(cat_counts.values(), labels=cat_counts.keys(), 
                                   autopct='%1.1f%%', colors=colors_pie,
                                   explode=(0, 0, 0.05, 0))
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax.set_title('(a) Hazard Exposure Overlap (SSP5-8.5)', fontweight='bold')

# Panel b: Risk transition across SSPs
ax = axes[1, 0]
risk_levels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
ssp_data = {}
for ssp_label, col_prefix in [('SSP2-4.5', 'ssp2_4_5'), ('SSP3-7.0', 'ssp3_7_0'), ('SSP5-8.5', 'ssp5_8_5')]:
    counts = gdf[f'{col_prefix}_cri_cat'].value_counts()
    ssp_data[ssp_label] = [counts.get(rl, 0) / len(gdf) * 100 for rl in risk_levels]

x = np.arange(len(risk_levels))
width = 0.25
for i, (ssp_label, values) in enumerate(ssp_data.items()):
    ax.bar(x + i * width, values, width, label=ssp_label, alpha=0.8,
           color=['#3498db', '#e67e22', '#e74c3c'][i])
ax.set_xticks(x + width)
ax.set_xticklabels(risk_levels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Percentage of Mangrove Sites')
ax.set_title('(b) Risk Distribution Across Scenarios', fontweight='bold')
ax.legend(fontsize=8)

# Panel c: Composite risk index formula / schematic
ax = axes[0, 1]
ax.axis('off')
formula_text = (
    "Composite Risk Index (CRI) Framework\n\n"
    "CRI = SLR_Risk + TC_Projected_Risk\n\n"
    "SLR Risk (0-2):\n"
    "  • Low (0): RSLR < 4 mm/yr\n"
    "  • Moderate (1): 4 ≤ RSLR < 7 mm/yr\n"
    "  • High (2): RSLR ≥ 7 mm/yr\n\n"
    "TC Risk (0-3):\n"
    "  • None (0): < 0.005 major TCs/yr\n"
    "  • Low (1): 0.005-0.02 /yr\n"
    "  • Moderate (2): 0.02-0.1 /yr\n"
    "  • High (3): > 0.1 /yr\n\n"
    "Projected TC Risk = Baseline × Shift Factor\n"
    "(based on Mo et al. 2023; Kropf et al. 2023)\n\n"
    "CRI Categories: Very Low (0), Low (1),\n"
    "Moderate (2), High (3), Very High (4-5)"
)
ax.text(0.05, 0.95, formula_text, transform=ax.transAxes, fontsize=9, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel d: Adaptation priority table
ax = axes[1, 1]
ax.axis('off')

# Create a summary table of top at-risk countries
top10 = country_risk_all[country_risk_all['n_points'] >= 100].nlargest(10, 'mean_cri')
table_data = top10[['Country', 'mean_cri', 'mean_slr', 'high_risk_pct', 'n_points']].copy()
table_data.columns = ['Country', 'Mean CRI', 'Mean SLR\n(mm/yr)', 'High Risk\n(%)', 'Sites']
table_data = table_data.round(1)

table = ax.table(cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.3, 0.15, 0.15, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.5)

# Color-code rows by risk
for i in range(len(table_data)):
    cri_val = table_data['Mean CRI'].iloc[i]
    color_idx = min(int(cri_val), 4)
    color = cri_colors_list[color_idx]
    for j in range(len(table_data.columns)):
        cell = table[i+1, j]
        cell.set_facecolor(color)
        cell.set_alpha(0.3)

ax.set_title('(d) Top 10 At-Risk Countries (SSP5-8.5)', fontweight='bold', y=1.05)

plt.suptitle('Risk Synthesis and Policy Implications', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure5_synthesis.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Validation - consistency with literature
# ============================================================
print("\n=== Figure 6: Validation plots ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel a: SLR thresholds validation
ax = axes[0, 0]
slr_bins = np.arange(0, 25, 1)
slr_counts, _ = np.histogram(gdf['ssp5_8_5_slr_2080_2100'], bins=slr_bins)
slr_cumfrac = np.cumsum(slr_counts) / len(gdf)

ax.fill_between(slr_bins[:-1], 0, slr_cumfrac, alpha=0.3, color='#e74c3c')
ax.plot(slr_bins[:-1], slr_cumfrac, 'o-', color='#e74c3c', linewidth=2, markersize=3)
ax.axvline(x=4, color='orange', linestyle='--', linewidth=1.5, label='4 mm/yr (Saintilan et al. 2023)')
ax.axvline(x=7, color='red', linestyle='--', linewidth=1.5, label='7 mm/yr (Saintilan et al. 2023)')
ax.set_xlabel('RSLR Rate (mm/yr)')
ax.set_ylabel('Cumulative Fraction of Mangrove Sites')
ax.set_title('(a) SLR Threshold Exposure', fontweight='bold')
ax.legend(fontsize=7)
ax.set_xlim(0, 25)

# Add annotations
for thresh, label in [(4, '4 mm/yr'), (7, '7 mm/yr')]:
    frac = (gdf['ssp5_8_5_slr_2080_2100'] >= thresh).mean()
    ax.annotate(f'{frac*100:.0f}% ≥ {label}', 
                xy=(thresh, 1-frac), fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel b: TC damage categories validation (from Mo et al. 2023)
ax = axes[0, 1]
cat_labels = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
cat_freqs = [gdf[f'tc_{c}_freq'].sum() for c in ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']]
total_freq = sum(cat_freqs)
cat_pcts = [f/total_freq*100 for f in cat_freqs]
bars = ax.bar(cat_labels, cat_pcts, color=['#f1c40f', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad'], 
              edgecolor='black', linewidth=0.5)
ax.set_ylabel('Percentage of All TC Encounters')
ax.set_title('(b) TC Category Distribution at Mangrove Sites', fontweight='bold')
# Add Mo et al. 2023 comparison
ax.axhline(y=97, color='#c0392b', linestyle='--', alpha=0.5, 
           label='Cat 3-5 = 97% of risk (Mo et al. 2023)')
major_pct = sum(cat_pcts[2:])
ax.annotate(f'Cat 3-5: {major_pct:.0f}%', xy=(2, major_pct), fontsize=8)
ax.legend(fontsize=7)

# Panel c: Projected vs baseline TC risk
ax = axes[1, 0]
# Compare baseline vs projected risk scores
sample = gdf.sample(min(5000, len(gdf)), random_state=42)
ax.scatter(sample['tc_baseline_risk_score'] + np.random.uniform(-0.15, 0.15, len(sample)),
          sample['tc_projected_risk_score'] + np.random.uniform(-0.15, 0.15, len(sample)),
          c=sample['tc_shift_factor'], s=2, alpha=0.5, cmap='RdBu_r', vmin=0.85, vmax=1.15)
ax.plot([-0.5, 3.5], [-0.5, 3.5], 'k--', alpha=0.3)
ax.set_xlabel('Baseline TC Risk Score')
ax.set_ylabel('Projected TC Risk Score')
ax.set_title('(c) TC Risk Regime Shift (SSP5-8.5)', fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='Shift Factor', shrink=0.8)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)

# Panel d: CRI comparison across SSPs
ax = axes[1, 1]
ssp_means = [
    gdf['ssp2_4_5_cri'].mean(),
    gdf['ssp3_7_0_cri'].mean(),
    gdf['ssp5_8_5_cri'].mean(),
]
ssp_vh = [
    (gdf['ssp2_4_5_cri_cat'] == 'Very High').mean() * 100,
    (gdf['ssp3_7_0_cri_cat'] == 'Very High').mean() * 100,
    (gdf['ssp5_8_5_cri_cat'] == 'Very High').mean() * 100,
]
ssp_hh = [
    ((gdf['ssp2_4_5_cri_cat'] == 'High') | (gdf['ssp2_4_5_cri_cat'] == 'Very High')).mean() * 100,
    ((gdf['ssp3_7_0_cri_cat'] == 'High') | (gdf['ssp3_7_0_cri_cat'] == 'Very High')).mean() * 100,
    ((gdf['ssp5_8_5_cri_cat'] == 'High') | (gdf['ssp5_8_5_cri_cat'] == 'Very High')).mean() * 100,
]

x = np.arange(3)
width = 0.25
ax.bar(x - width, ssp_means, width, color=['#3498db', '#e67e22', '#e74c3c'], label='Mean CRI')
ax2 = ax.twinx()
ax2.bar(x, ssp_vh, width, color='#8e44ad', alpha=0.7, label='Very High %')
ax2.bar(x + width, ssp_hh, width, color='#e74c3c', alpha=0.7, label='High+Very High %')
ax.set_xticks(x)
ax.set_xticklabels(['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'])
ax.set_ylabel('Mean CRI')
ax2.set_ylabel('Percentage of Sites')
ax.set_title('(d) Risk Escalation Across Scenarios', fontweight='bold')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

plt.suptitle('Validation and Scenario Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure6_validation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 6 saved.")

print("\n=== All figures generated! ===")
