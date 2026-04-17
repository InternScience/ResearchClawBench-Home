#!/usr/bin/env python3
"""
Visualization script for the Composite Risk Index analysis.
Generates all figures for the report.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_002_20260417_110902'
OUT = f'{BASE}/outputs'
IMG = f'{BASE}/report/images'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

ssp_scenarios = ['ssp245', 'ssp370', 'ssp585']
ssp_labels = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}
ssp_colors = {'ssp245': '#2166ac', 'ssp370': '#f4a582', 'ssp585': '#b2182b'}

# Load processed data
print("Loading processed data...")
mangroves = pd.read_pickle(f'{OUT}/mangroves_processed.pkl')
mang_country = pd.read_pickle(f'{OUT}/mang_country_processed.pkl')
cstats = pd.read_csv(f'{OUT}/country_risk_stats.csv')
region_df = pd.read_csv(f'{OUT}/regional_risk_stats.csv')

# Load world boundaries for basemap
try:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
except:
    world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')

print(f"Loaded {len(mangroves)} mangrove points, {len(cstats)} countries")

# ============================================================
# FIGURE 1: Global Composite Risk Index Map
# ============================================================
print("\n--- Figure 1: Global CRI Map ---")

fig, axes = plt.subplots(3, 1, figsize=(16, 18), subplot_kw={'projection': None})

for i, ssp in enumerate(ssp_scenarios):
    ax = axes[i]
    world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
    
    cri = mangroves[f'cri_{ssp}'].values
    lons = mangroves['mang_lon'].values
    lats = mangroves['mang_lat'].values
    
    # Sort by CRI so high values are plotted on top
    sort_idx = np.argsort(cri)
    
    sc = ax.scatter(lons[sort_idx], lats[sort_idx], c=cri[sort_idx],
                    cmap='RdYlGn_r', vmin=0, vmax=1, s=1.5, alpha=0.7,
                    rasterized=True)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-50, 40)
    ax.set_title(f'Composite Risk Index — {ssp_labels[ssp]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('CRI')

plt.tight_layout()
plt.savefig(f'{IMG}/fig1_global_cri_map.png', dpi=150)
plt.close()
print("  Saved fig1_global_cri_map.png")

# ============================================================
# FIGURE 2: TC Risk Component Map
# ============================================================
print("\n--- Figure 2: TC Risk Map ---")

fig, ax = plt.subplots(1, 1, figsize=(16, 7))
world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)

tc_score = mangroves['tc_score'].values
lons = mangroves['mang_lon'].values
lats = mangroves['mang_lat'].values
sort_idx = np.argsort(tc_score)

sc = ax.scatter(lons[sort_idx], lats[sort_idx], c=tc_score[sort_idx],
                cmap='YlOrRd', vmin=0, vmax=1, s=2, alpha=0.7, rasterized=True)

ax.set_xlim(-180, 180)
ax.set_ylim(-50, 40)
ax.set_title('Tropical Cyclone Risk Component (Normalized TCRI)', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('TC Risk Score')

plt.tight_layout()
plt.savefig(f'{IMG}/fig2_tc_risk_map.png', dpi=150)
plt.close()
print("  Saved fig2_tc_risk_map.png")

# ============================================================
# FIGURE 3: SLR Risk Component Map
# ============================================================
print("\n--- Figure 3: SLR Risk Map ---")

fig, axes = plt.subplots(3, 1, figsize=(16, 18))

for i, ssp in enumerate(ssp_scenarios):
    ax = axes[i]
    world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
    
    slr_rate = mangroves[f'slr_rate_2100_{ssp}'].values
    sort_idx = np.argsort(slr_rate)
    
    sc = ax.scatter(lons[sort_idx], lats[sort_idx], c=slr_rate[sort_idx],
                    cmap='Blues', vmin=0, vmax=20, s=1.5, alpha=0.7, rasterized=True)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-50, 40)
    ax.set_title(f'Sea Level Rise Rate at 2100 — {ssp_labels[ssp]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('SLR Rate (mm/yr)')
    
    # Add threshold lines in colorbar
    for thresh, label in [(4, '4'), (7, '7')]:
        cbar.ax.axhline(y=thresh, color='red', linewidth=1.5, linestyle='--')

plt.tight_layout()
plt.savefig(f'{IMG}/fig3_slr_risk_map.png', dpi=150)
plt.close()
print("  Saved fig3_slr_risk_map.png")

# ============================================================
# FIGURE 4: Country-level CRI Rankings
# ============================================================
print("\n--- Figure 4: Country CRI Rankings ---")

# Get CRI mean columns
cri_mean_cols = {}
for ssp in ssp_scenarios:
    candidates = [c for c in cstats.columns if f'cri_{ssp}' in c and 'mean' in c]
    cri_mean_cols[ssp] = candidates[0] if candidates else None

# Top 25 countries by SSP585 CRI
top25 = cstats.nlargest(25, cri_mean_cols['ssp585']).copy()
top25 = top25.sort_values(cri_mean_cols['ssp585'], ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(top25))
bar_height = 0.25

for j, ssp in enumerate(ssp_scenarios):
    col = cri_mean_cols[ssp]
    ax.barh(y_pos + j * bar_height, top25[col], bar_height,
            label=ssp_labels[ssp], color=ssp_colors[ssp], alpha=0.85)

ax.set_yticks(y_pos + bar_height)
ax.set_yticklabels(top25['Country'])
ax.set_xlabel('Mean Composite Risk Index (CRI)')
ax.set_title('Top 25 Countries by Composite Risk Index', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5, label='Moderate threshold')
ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='High threshold')
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f'{IMG}/fig4_country_cri_rankings.png', dpi=150)
plt.close()
print("  Saved fig4_country_cri_rankings.png")

# ============================================================
# FIGURE 5: TC vs SLR Contribution Scatter
# ============================================================
print("\n--- Figure 5: TC vs SLR Scatter ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, ssp in enumerate(ssp_scenarios):
    ax = axes[i]
    
    tc = mangroves['tc_score'].values
    slr = mangroves[f'slr_score_{ssp}'].values
    cri = mangroves[f'cri_{ssp}'].values
    
    # Subsample for visibility
    np.random.seed(42)
    idx = np.random.choice(len(tc), min(10000, len(tc)), replace=False)
    
    sc = ax.scatter(tc[idx], slr[idx], c=cri[idx], cmap='RdYlGn_r',
                    vmin=0, vmax=1, s=5, alpha=0.3, rasterized=True)
    
    ax.set_xlabel('TC Risk Score')
    ax.set_ylabel('SLR Risk Score')
    ax.set_title(f'{ssp_labels[ssp]}')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('CRI')

fig.suptitle('TC Risk vs SLR Risk Components', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMG}/fig5_tc_vs_slr_scatter.png', dpi=150)
plt.close()
print("  Saved fig5_tc_vs_slr_scatter.png")

# ============================================================
# FIGURE 6: Regional Risk Comparison
# ============================================================
print("\n--- Figure 6: Regional Risk ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Panel A: CRI by region
ax = axes[0]
regions = region_df[region_df['ssp'] == 'ssp585'].sort_values('cri_mean', ascending=True)['region'].values
for j, ssp in enumerate(ssp_scenarios):
    rd = region_df[region_df['ssp'] == ssp].set_index('region').reindex(regions)
    y_pos = np.arange(len(regions))
    ax.barh(y_pos + j * 0.25, rd['cri_mean'], 0.25,
            label=ssp_labels[ssp], color=ssp_colors[ssp], alpha=0.85)
ax.set_yticks(np.arange(len(regions)) + 0.25)
ax.set_yticklabels(regions, fontsize=9)
ax.set_xlabel('Mean CRI')
ax.set_title('(a) CRI by Region')
ax.legend(fontsize=8)

# Panel B: TC component by region
ax = axes[1]
rd = region_df[region_df['ssp'] == 'ssp585'].set_index('region').reindex(regions)
ax.barh(np.arange(len(regions)), rd['tc_mean'], 0.6, color='#d73027', alpha=0.7)
ax.set_yticks(np.arange(len(regions)))
ax.set_yticklabels(regions, fontsize=9)
ax.set_xlabel('Mean TC Risk Score')
ax.set_title('(b) TC Risk by Region')

# Panel C: SLR component by region
ax = axes[2]
for j, ssp in enumerate(ssp_scenarios):
    rd = region_df[region_df['ssp'] == ssp].set_index('region').reindex(regions)
    ax.barh(np.arange(len(regions)) + j * 0.25, rd['slr_rate'], 0.25,
            label=ssp_labels[ssp], color=ssp_colors[ssp], alpha=0.85)
ax.set_yticks(np.arange(len(regions)) + 0.25)
ax.set_yticklabels(regions, fontsize=9)
ax.set_xlabel('Mean SLR Rate at 2100 (mm/yr)')
ax.set_title('(c) SLR Rate by Region')
ax.axvline(x=4, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(x=7, color='red', linestyle='--', alpha=0.7, linewidth=1)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{IMG}/fig6_regional_risk.png', dpi=150)
plt.close()
print("  Saved fig6_regional_risk.png")

# ============================================================
# FIGURE 7: SSP Scenario Comparison (Distributions)
# ============================================================
print("\n--- Figure 7: SSP Comparison ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: CRI distributions
ax = axes[0]
for ssp in ssp_scenarios:
    cri = mangroves[f'cri_{ssp}'].values
    ax.hist(cri, bins=50, alpha=0.5, label=ssp_labels[ssp], color=ssp_colors[ssp], density=True)
ax.set_xlabel('Composite Risk Index')
ax.set_ylabel('Density')
ax.set_title('(a) CRI Distribution')
ax.legend()
ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5)

# Panel B: SLR rate distributions
ax = axes[1]
for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    ax.hist(rate, bins=50, alpha=0.5, label=ssp_labels[ssp], color=ssp_colors[ssp], density=True)
ax.set_xlabel('SLR Rate at 2100 (mm/yr)')
ax.set_ylabel('Density')
ax.set_title('(b) SLR Rate Distribution at Mangrove Locations')
ax.axvline(x=4, color='orange', linestyle='--', alpha=0.7, label='4 mm/yr')
ax.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='7 mm/yr')
ax.legend()

# Panel C: % mangroves in risk categories
ax = axes[2]
categories = ['Very Low\n(0-0.2)', 'Low\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 
              'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)']
x_pos = np.arange(len(categories))
width = 0.25

for j, ssp in enumerate(ssp_scenarios):
    cri = mangroves[f'cri_{ssp}'].values
    pcts = [
        100 * np.mean(cri <= 0.2),
        100 * np.mean((cri > 0.2) & (cri <= 0.4)),
        100 * np.mean((cri > 0.4) & (cri <= 0.6)),
        100 * np.mean((cri > 0.6) & (cri <= 0.8)),
        100 * np.mean(cri > 0.8)
    ]
    ax.bar(x_pos + j * width, pcts, width, label=ssp_labels[ssp], color=ssp_colors[ssp], alpha=0.85)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('% of Mangrove Points')
ax.set_title('(c) Risk Category Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f'{IMG}/fig7_ssp_comparison.png', dpi=150)
plt.close()
print("  Saved fig7_ssp_comparison.png")

# ============================================================
# FIGURE 8: Ecosystem Services at Risk
# ============================================================
print("\n--- Figure 8: Ecosystem Services at Risk ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Get top countries by area
top_area = cstats.nlargest(15, 'Mang_Ha_2020').copy()
top_area = top_area.sort_values('Mang_Ha_2020', ascending=True)

# Panel A: Mangrove area by country with risk overlay
ax = axes[0, 0]
y_pos = np.arange(len(top_area))
cri_col = cri_mean_cols['ssp585']
colors = plt.cm.RdYlGn_r(top_area[cri_col].values / top_area[cri_col].max())
ax.barh(y_pos, top_area['Mang_Ha_2020'] / 1000, color=colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_area['Country'], fontsize=9)
ax.set_xlabel('Mangrove Area (×1000 ha)')
ax.set_title('(a) Mangrove Area (color = CRI SSP5-8.5)')

# Panel B: Population benefiting from mangroves
ax = axes[0, 1]
top_pop = cstats.nlargest(15, 'Ben_Pop_2020').copy()
top_pop = top_pop.sort_values('Ben_Pop_2020', ascending=True)
y_pos = np.arange(len(top_pop))
cri_vals = top_pop[cri_col].values
colors = plt.cm.RdYlGn_r(cri_vals / max(cri_vals.max(), 0.001))
ax.barh(y_pos, top_pop['Ben_Pop_2020'] / 1e6, color=colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_pop['Country'], fontsize=9)
ax.set_xlabel('Benefiting Population (millions)')
ax.set_title('(b) Population Benefiting (color = CRI SSP5-8.5)')

# Panel C: Property value at risk
ax = axes[1, 0]
for j, ssp in enumerate(ssp_scenarios):
    col = f'stock_risk_{ssp}'
    if col in cstats.columns:
        top_stock = cstats.nlargest(15, col).copy()
        total = top_stock[col].sum()
        ax.bar(j, total / 1e9, color=ssp_colors[ssp], alpha=0.85, label=ssp_labels[ssp])
ax.set_xticks(range(3))
ax.set_xticklabels([ssp_labels[s] for s in ssp_scenarios])
ax.set_ylabel('Property Value at Risk (billion USD)')
ax.set_title('(c) Total Property Value at Risk (Top 15 Countries)')
ax.legend()

# Panel D: Mangrove area at risk
ax = axes[1, 1]
for j, ssp in enumerate(ssp_scenarios):
    col = f'area_risk_{ssp}'
    if col in cstats.columns:
        total = cstats[col].sum()
        ax.bar(j, total / 1e6, color=ssp_colors[ssp], alpha=0.85, label=ssp_labels[ssp])
ax.set_xticks(range(3))
ax.set_xticklabels([ssp_labels[s] for s in ssp_scenarios])
ax.set_ylabel('Mangrove Area at Risk (million ha)')
ax.set_title('(d) Total Mangrove Area at Risk')
ax.legend()

plt.tight_layout()
plt.savefig(f'{IMG}/fig8_ecosystem_services.png', dpi=150)
plt.close()
print("  Saved fig8_ecosystem_services.png")

# ============================================================
# FIGURE 9: SLR Threshold Exceedance
# ============================================================
print("\n--- Figure 9: SLR Threshold Analysis ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: % mangroves exceeding thresholds
ax = axes[0]
thresholds = np.arange(0, 20.5, 0.5)
for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    pcts = [100 * np.mean(rate >= t) for t in thresholds]
    ax.plot(thresholds, pcts, label=ssp_labels[ssp], color=ssp_colors[ssp], linewidth=2)

ax.axvline(x=4, color='orange', linestyle='--', alpha=0.7, label='4 mm/yr (deficit likely)')
ax.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='7 mm/yr (highly likely loss)')
ax.set_xlabel('SLR Rate Threshold (mm/yr)')
ax.set_ylabel('% of Mangrove Locations Exceeding Threshold')
ax.set_title('(a) SLR Threshold Exceedance at 2100')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: CRI vs SLR rate
ax = axes[1]
for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    cri = mangroves[f'cri_{ssp}'].values
    # Bin by SLR rate
    bins = np.arange(0, 22, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    for b in range(len(bins) - 1):
        mask = (rate >= bins[b]) & (rate < bins[b+1])
        if mask.sum() > 10:
            bin_means.append(cri[mask].mean())
            bin_stds.append(cri[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    bm = np.array(bin_means)
    bs = np.array(bin_stds)
    valid = ~np.isnan(bm)
    ax.plot(bin_centers[valid], bm[valid], color=ssp_colors[ssp], linewidth=2, label=ssp_labels[ssp])
    ax.fill_between(bin_centers[valid], (bm-bs)[valid], (bm+bs)[valid], 
                     color=ssp_colors[ssp], alpha=0.15)

ax.axvline(x=4, color='orange', linestyle='--', alpha=0.7)
ax.axvline(x=7, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('SLR Rate at 2100 (mm/yr)')
ax.set_ylabel('Mean CRI (±1 SD)')
ax.set_title('(b) CRI vs SLR Rate')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG}/fig9_slr_threshold.png', dpi=150)
plt.close()
print("  Saved fig9_slr_threshold.png")

# ============================================================
# FIGURE 10: Compound Risk Hotspots
# ============================================================
print("\n--- Figure 10: Compound Risk Hotspots ---")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)

# Identify compound hotspots: high TC AND high SLR
tc = mangroves['tc_score'].values
slr = mangroves[f'slr_score_ssp585'].values
cri = mangroves['cri_ssp585'].values
lons = mangroves['mang_lon'].values
lats = mangroves['mang_lat'].values

# Classify into quadrants
colors_arr = np.full(len(tc), '#cccccc')  # default gray
# Low TC, Low SLR
mask_ll = (tc < 0.3) & (slr < 0.7)
colors_arr[mask_ll] = '#4daf4a'  # green
# Low TC, High SLR
mask_lh = (tc < 0.3) & (slr >= 0.7)
colors_arr[mask_lh] = '#377eb8'  # blue
# High TC, Low SLR
mask_hl = (tc >= 0.3) & (slr < 0.7)
colors_arr[mask_hl] = '#ff7f00'  # orange
# High TC, High SLR (compound)
mask_hh = (tc >= 0.3) & (slr >= 0.7)
colors_arr[mask_hh] = '#e41a1c'  # red

# Plot in order: gray first, then colored
for mask, color, label in [
    (mask_ll, '#4daf4a', f'Low TC + Low SLR ({mask_ll.sum()})'),
    (mask_lh, '#377eb8', f'Low TC + High SLR ({mask_lh.sum()})'),
    (mask_hl, '#ff7f00', f'High TC + Low SLR ({mask_hl.sum()})'),
    (mask_hh, '#e41a1c', f'High TC + High SLR ({mask_hh.sum()})'),
]:
    if mask.sum() > 0:
        ax.scatter(lons[mask], lats[mask], c=color, s=3, alpha=0.6, 
                  label=label, rasterized=True)

ax.set_xlim(-180, 180)
ax.set_ylim(-50, 40)
ax.set_title('Compound Risk Hotspots (SSP5-8.5): TC × SLR', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc='lower left', fontsize=10, markerscale=3)

plt.tight_layout()
plt.savefig(f'{IMG}/fig10_compound_hotspots.png', dpi=150)
plt.close()
print("  Saved fig10_compound_hotspots.png")

# ============================================================
# FIGURE 11: Sensitivity Analysis (Weight Comparison)
# ============================================================
print("\n--- Figure 11: Sensitivity Analysis ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Compute CRI with different weights
weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
ssp = 'ssp585'

ax = axes[0]
for w_tc, w_slr in weights:
    cri_w = w_tc * mangroves['tc_score'] + w_slr * mangroves[f'slr_score_{ssp}']
    ax.hist(cri_w, bins=50, alpha=0.4, density=True,
            label=f'TC:{w_tc:.1f}, SLR:{w_slr:.1f}')
ax.set_xlabel('CRI')
ax.set_ylabel('Density')
ax.set_title(f'(a) CRI Sensitivity to Weights ({ssp_labels[ssp]})')
ax.legend(fontsize=8)

# Panel B: Mean CRI by weight across SSPs
ax = axes[1]
for ssp in ssp_scenarios:
    means = []
    w_tc_vals = np.arange(0, 1.05, 0.05)
    for w_tc in w_tc_vals:
        w_slr = 1 - w_tc
        cri_w = w_tc * mangroves['tc_score'] + w_slr * mangroves[f'slr_score_{ssp}']
        means.append(cri_w.mean())
    ax.plot(w_tc_vals, means, label=ssp_labels[ssp], color=ssp_colors[ssp], linewidth=2)

ax.set_xlabel('TC Weight (SLR Weight = 1 - TC Weight)')
ax.set_ylabel('Mean CRI')
ax.set_title('(b) Mean CRI vs Weight Allocation')
ax.legend()
ax.grid(alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{IMG}/fig11_sensitivity.png', dpi=150)
plt.close()
print("  Saved fig11_sensitivity.png")

# ============================================================
# FIGURE 12: Data Overview
# ============================================================
print("\n--- Figure 12: Data Overview ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

def classify_ss(wind_ms):
    cats = np.zeros(len(wind_ms), dtype=int)
    cats[(wind_ms >= 33) & (wind_ms < 42.5)] = 1
    cats[(wind_ms >= 42.5) & (wind_ms < 49.5)] = 2
    cats[(wind_ms >= 49.5) & (wind_ms < 58)] = 3
    cats[(wind_ms >= 58) & (wind_ms < 70)] = 4
    cats[wind_ms >= 70] = 5
    return cats

# Panel A: Mangrove distribution
ax = axes[0, 0]
world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
ax.scatter(mangroves['mang_lon'], mangroves['mang_lat'], c='#2ca02c', s=0.5, alpha=0.3, rasterized=True)
ax.set_xlim(-180, 180)
ax.set_ylim(-50, 40)
ax.set_title('(a) Global Mangrove Distribution (10% sample)')

# Panel B: TC track density
ax = axes[0, 1]
world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
# Plot TC track points colored by category
# Reload TC data
import xarray as xr
tc_ds = xr.open_dataset(f'{BASE}/data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lat_all = tc_ds['lat'].values
tc_lon_all = tc_ds['lon'].values
tc_wind_all = tc_ds['wind'].values
tc_ds.close()

# Subsample for plotting
np.random.seed(42)
tc_plot_idx = np.random.choice(len(tc_lat_all), 20000, replace=False)
tc_cats_plot = classify_ss(tc_wind_all[tc_plot_idx])
cat_colors = {1: '#fee08b', 2: '#fdae61', 3: '#f46d43', 4: '#d73027', 5: '#a50026'}

for cat in range(1, 6):
    mask = tc_cats_plot == cat
    ax.scatter(tc_lon_all[tc_plot_idx][mask], tc_lat_all[tc_plot_idx][mask],
              c=cat_colors[cat], s=0.5, alpha=0.3, label=f'Cat {cat}', rasterized=True)
ax.set_xlim(-180, 180)
ax.set_ylim(-50, 50)
ax.set_title('(b) TC Track Points (20k sample)')
ax.legend(fontsize=8, markerscale=5, loc='lower left')

# Panel C: TC category distribution
ax = axes[1, 0]
tc_cats_all = classify_ss(tc_wind_all)
cats, counts = np.unique(tc_cats_all, return_counts=True)
cat_labels = [f'Cat {c}' for c in cats]
bars = ax.bar(cat_labels, counts, color=[cat_colors[c] for c in cats])
ax.set_xlabel('Saffir-Simpson Category')
ax.set_ylabel('Number of Track Points')
ax.set_title('(c) TC Category Distribution')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count:,}', ha='center', va='bottom', fontsize=9)

# Panel D: SLR rate distribution at mangrove locations
ax = axes[1, 1]
for ssp in ssp_scenarios:
    rate = mangroves[f'slr_rate_2100_{ssp}'].values
    ax.hist(rate, bins=50, alpha=0.5, label=ssp_labels[ssp], color=ssp_colors[ssp])
ax.axvline(x=4, color='orange', linestyle='--', linewidth=2, label='4 mm/yr threshold')
ax.axvline(x=7, color='red', linestyle='--', linewidth=2, label='7 mm/yr threshold')
ax.set_xlabel('SLR Rate at 2100 (mm/yr)')
ax.set_ylabel('Count')
ax.set_title('(d) SLR Rate Distribution at Mangrove Locations')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{IMG}/fig12_data_overview.png', dpi=150)
plt.close()
print("  Saved fig12_data_overview.png")

print("\n\nAll figures generated successfully!")
print(f"Figures saved to: {IMG}")
