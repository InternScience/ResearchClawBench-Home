#!/usr/bin/env python3
"""
Step 2: Build composite risk index and generate figures.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

os.makedirs('report/images', exist_ok=True)

# Load data
d = np.load('outputs/mangrove_risk_data.npz')
lon = d['lon']
lat = d['lat']
slr245 = d['slr_ssp245']
slr370 = d['slr_ssp370']
slr585 = d['slr_ssp585']
tc_freq = d['tc_annual_freq']
tc_wind = d['tc_max_wind']

n = len(lon)

# ============================================================
# 1. Normalize risk components to [0, 1]
# ============================================================

# SLR risk: based on thresholds from Saintilan et al. 2023
# 4 mm/yr = deficit likely, 7 mm/yr = deficit highly likely
# Risk = min(rate / 10, 1) — saturates at 10 mm/yr
def slr_risk(rate):
    return np.clip(rate / 10.0, 0, 1)

slr_risk_245 = slr_risk(slr245)
slr_risk_370 = slr_risk(slr370)
slr_risk_585 = slr_risk(slr585)

# TC risk: based on annual frequency and max wind speed
# Normalize frequency by global max, wind by 70 m/s (Cat 5)
# Risk = 0.5 * (freq_norm) + 0.5 * (wind_norm)
freq_max = np.percentile(tc_freq[tc_freq > 0], 99) if np.any(tc_freq > 0) else 1.0
freq_norm = np.clip(tc_freq / freq_max, 0, 1)
wind_norm = np.clip(tc_wind / 70.0, 0, 1)
tc_risk = 0.5 * freq_norm + 0.5 * wind_norm

# Composite risk index: weighted combination
# Following the task: combine TC regime shifts and SLR
# Weight: 0.6 SLR + 0.4 TC (SLR is the primary driver of permanent loss)
def composite_risk(slr_r, tc_r, w_slr=0.6, w_tc=0.4):
    return w_slr * slr_r + w_tc * tc_r

comp_245 = composite_risk(slr_risk_245, tc_risk)
comp_370 = composite_risk(slr_risk_370, tc_risk)
comp_585 = composite_risk(slr_risk_585, tc_risk)

# ============================================================
# Save composite risk results
# ============================================================
np.savez_compressed('outputs/composite_risk.npz',
    lon=lon, lat=lat,
    slr_risk_245=slr_risk_245, slr_risk_370=slr_risk_370, slr_risk_585=slr_risk_585,
    tc_risk=tc_risk,
    comp_245=comp_245, comp_370=comp_370, comp_585=comp_585,
    slr245=slr245, slr370=slr370, slr585=slr585,
    tc_freq=tc_freq, tc_wind=tc_wind,
)

# ============================================================
# 2. Figure 1: Global composite risk maps by SSP
# ============================================================
print("Generating Figure 1: Global composite risk maps...")

fig, axes = plt.subplots(3, 1, figsize=(14, 18),
    subplot_kw={'projection': ccrs.Robinson()})

scenarios = [
    ('SSP2-4.5', comp_245, slr245),
    ('SSP3-7.0', comp_370, slr370),
    ('SSP5-8.5', comp_585, slr585),
]

cmap = plt.cm.YlOrRd
for ax, (name, comp, slr) in zip(axes, scenarios):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f8ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    
    sc = ax.scatter(lon, lat, c=comp, cmap=cmap, s=0.5, alpha=0.8,
        vmin=0, vmax=1, transform=ccrs.PlateCarree())
    ax.set_title(f'{name} — Composite Risk Index (2080–2100)', fontsize=14, fontweight='bold')

cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
cbar.set_label('Composite Risk Index', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/fig1_global_composite_risk.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig1_global_composite_risk.png")

# ============================================================
# 3. Figure 2: Risk decomposition (SLR vs TC contribution)
# ============================================================
print("Generating Figure 2: Risk decomposition...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: SLR risk vs TC risk scatter
ax = axes[0]
sc = ax.scatter(slr_risk_585, tc_risk, c=comp_585, cmap='YlOrRd', s=2, alpha=0.5, vmin=0, vmax=1)
ax.set_xlabel('SLR Risk (SSP5-8.5)', fontsize=12)
ax.set_ylabel('TC Risk', fontsize=12)
ax.set_title('Risk Decomposition: SLR vs TC', fontsize=14, fontweight='bold')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
fig.colorbar(sc, ax=ax, label='Composite Risk')

# Right: histogram of composite risk by SSP
ax2 = axes[1]
bins = np.linspace(0, 1, 50)
ax2.hist(comp_245, bins=bins, alpha=0.5, label='SSP2-4.5', color='steelblue', density=True)
ax2.hist(comp_370, bins=bins, alpha=0.5, label='SSP3-7.0', color='orange', density=True)
ax2.hist(comp_585, bins=bins, alpha=0.5, label='SSP5-8.5', color='red', density=True)
ax2.set_xlabel('Composite Risk Index', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Distribution of Composite Risk by Scenario', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('report/images/fig2_risk_decomposition.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig2_risk_decomposition.png")

# ============================================================
# 4. Figure 3: Regional risk rankings
# ============================================================
print("Generating Figure 3: Regional risk rankings...")

# Assign regions based on lat/lon
def assign_region(lat, lon):
    if 0 <= lat <= 30 and -100 <= lon <= -50:
        return 'Caribbean'
    elif 20 <= lat <= 45 and 100 <= lon <= 150:
        return 'East Asia'
    elif -10 <= lat <= 25 and 60 <= lon <= 100:
        return 'South Asia'
    elif -15 <= lat <= 10 and 95 <= lon <= 140:
        return 'Southeast Asia'
    elif -30 <= lat <= 0 and 110 <= lon <= 160:
        return 'Oceania'
    elif -30 <= lat <= 0 and -80 <= lon <= -35:
        return 'South America'
    elif 0 <= lat <= 30 and -10 <= lon <= 50:
        return 'West Africa'
    elif -30 <= lat <= 0 and 30 <= lon <= 55:
        return 'East Africa'
    elif 20 <= lat <= 50 and -130 <= lon <= -60:
        return 'North America'
    elif 5 <= lat <= 25 and 115 <= lon <= 130:
        return 'Pacific Islands'
    else:
        return 'Other'

regions = np.array([assign_region(la, lo) for la, lo in zip(lat, lon)])
unique_regions = sorted(set(regions))

# Compute mean composite risk per region
region_stats = {}
for reg in unique_regions:
    mask = regions == reg
    if np.sum(mask) < 10:
        continue
    region_stats[reg] = {
        'n_points': int(np.sum(mask)),
        'mean_comp_245': float(np.mean(comp_245[mask])),
        'mean_comp_370': float(np.mean(comp_370[mask])),
        'mean_comp_585': float(np.mean(comp_585[mask])),
        'mean_slr_585': float(np.mean(slr585[mask])),
        'mean_tc_freq': float(np.mean(tc_freq[mask])),
    }

# Sort by SSP585 composite risk
sorted_regions = sorted(region_stats.items(), key=lambda x: x[1]['mean_comp_585'], reverse=True)

fig, ax = plt.subplots(figsize=(12, 8))
reg_names = [r[0] for r in sorted_regions]
y_pos = np.arange(len(reg_names))
bar_h = 0.25

bars1 = ax.barh(y_pos - bar_h, [r[1]['mean_comp_245'] for r in sorted_regions], bar_h, label='SSP2-4.5', color='steelblue')
bars2 = ax.barh(y_pos, [r[1]['mean_comp_370'] for r in sorted_regions], bar_h, label='SSP3-7.0', color='orange')
bars3 = ax.barh(y_pos + bar_h, [r[1]['mean_comp_585'] for r in sorted_regions], bar_h, label='SSP5-8.5', color='red')

ax.set_yticks(y_pos)
ax.set_yticklabels(reg_names, fontsize=11)
ax.set_xlabel('Mean Composite Risk Index', fontsize=12)
ax.set_title('Regional Composite Risk Rankings (2080–2100)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('report/images/fig3_regional_risk_rankings.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig3_regional_risk_rankings.png")

# ============================================================
# 5. Figure 4: Scenario comparison heatmap
# ============================================================
print("Generating Figure 4: Scenario comparison heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
data_matrix = np.array([
    [r[1]['mean_comp_245'], r[1]['mean_comp_370'], r[1]['mean_comp_585']]
    for r in sorted_regions
])

im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'], fontsize=12)
ax.set_yticks(range(len(reg_names)))
ax.set_yticklabels(reg_names, fontsize=11)
ax.set_title('Composite Risk by Region and Scenario', fontsize=14, fontweight='bold')

for i in range(len(reg_names)):
    for j in range(3):
        ax.text(j, i, f'{data_matrix[i, j]:.2f}', ha='center', va='center', fontsize=9,
                color='white' if data_matrix[i, j] > 0.5 else 'black')

fig.colorbar(im, ax=ax, label='Composite Risk Index')
plt.tight_layout()
plt.savefig('report/images/fig4_scenario_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig4_scenario_heatmap.png")

# ============================================================
# 6. Figure 5: SLR threshold exposure
# ============================================================
print("Generating Figure 5: SLR threshold exposure...")

thresholds = [4, 7, 10]
scenarios_slr = [('SSP2-4.5', slr245), ('SSP3-7.0', slr370), ('SSP5-8.5', slr585)]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = ['steelblue', 'orange', 'red']

for ax, (name, slr), color in zip(axes, scenarios_slr, colors):
    pcts = [100 * np.mean(slr > t) for t in thresholds]
    bars = ax.bar(['4 mm/yr\n(deficit likely)', '7 mm/yr\n(deficit highly likely)', '10 mm/yr\n(severe)'], pcts, color=color, alpha=0.8)
    ax.set_ylabel('% of mangrove points', fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{pct:.1f}%', ha='center', fontsize=10)

plt.suptitle('Mangrove Exposure to SLR Thresholds (2080–2100)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig5_slr_threshold_exposure.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig5_slr_threshold_exposure.png")

# ============================================================
# Print summary statistics
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for name, comp, slr in scenarios:
    print(f"\n{name}:")
    print(f"  Composite Risk: mean={np.mean(comp):.3f}, median={np.median(comp):.3f}, 90th pct={np.percentile(comp, 90):.3f}")
    print(f"  SLR rate (mm/yr): mean={np.mean(slr):.1f}, median={np.median(slr):.1f}")
    print(f"  High risk (>0.7): {100*np.mean(comp>0.7):.1f}%")
    print(f"  Very high risk (>0.85): {100*np.mean(comp>0.85):.1f}%")

print("\nTC Risk:")
print(f"  Mean TC risk: {np.mean(tc_risk):.3f}")
print(f"  Points with TC risk > 0.3: {100*np.mean(tc_risk>0.3):.1f}%")

print("\nTop 5 highest risk regions (SSP5-8.5):")
for reg, stats in sorted_regions[:5]:
    print(f"  {reg}: {stats['mean_comp_585']:.3f} (SLR={stats['mean_slr_585']:.1f} mm/yr, TC_freq={stats['mean_tc_freq']:.2f}/yr)")

print("\nDone!")
