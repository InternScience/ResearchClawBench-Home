#!/usr/bin/env python3
"""
Step 3: Additional validation and ecosystem service figures.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

os.makedirs('report/images', exist_ok=True)

d = np.load('outputs/composite_risk.npz')
lon = d['lon']
lat = d['lat']
slr245 = d['slr245']
slr370 = d['slr370']
slr585 = d['slr585']
tc_freq = d['tc_freq']
tc_wind = d['tc_wind']
comp_245 = d['comp_245']
comp_370 = d['comp_370']
comp_585 = d['comp_585']
tc_risk = d['tc_risk']

# ============================================================
# Figure 6: SLR rate maps by SSP
# ============================================================
print("Generating Figure 6: SLR rate maps...")

fig, axes = plt.subplots(3, 1, figsize=(14, 18),
    subplot_kw={'projection': ccrs.Robinson()})

scenarios = [('SSP2-4.5', slr245), ('SSP3-7.0', slr370), ('SSP5-8.5', slr585)]
cmap = plt.cm.viridis

for ax, (name, slr) in zip(axes, scenarios):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f8ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    sc = ax.scatter(lon, lat, c=slr, cmap=cmap, s=0.5, alpha=0.8,
        vmin=0, vmax=25, transform=ccrs.PlateCarree())
    ax.set_title(f'{name} — SLR Rate (mm/yr, 2080–2100)', fontsize=14, fontweight='bold')

cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
cbar.set_label('SLR Rate (mm/yr)', fontsize=12)
plt.savefig('report/images/fig6_slr_rate_maps.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig6_slr_rate_maps.png")

# ============================================================
# Figure 7: TC exposure map
# ============================================================
print("Generating Figure 7: TC exposure map...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8),
    subplot_kw={'projection': ccrs.Robinson()})
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#f0f8ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

# Show only points with TC exposure
mask = tc_freq > 0
sc = ax.scatter(lon[mask], lat[mask], c=tc_freq[mask], cmap='YlOrRd', s=1, alpha=0.7,
    vmin=0, vmax=np.percentile(tc_freq[mask], 95), transform=ccrs.PlateCarree())
ax.set_title('Tropical Cyclone Annual Frequency at Mangrove Sites (Historical 1850–2014)', fontsize=14, fontweight='bold')
fig.colorbar(sc, ax=ax, label='Annual TC frequency (events/yr)', shrink=0.6)
plt.savefig('report/images/fig7_tc_exposure_map.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig7_tc_exposure_map.png")

# ============================================================
# Figure 8: Risk change from SSP245 to SSP585
# ============================================================
print("Generating Figure 8: Risk change map...")

risk_change = comp_585 - comp_245

fig, ax = plt.subplots(1, 1, figsize=(14, 8),
    subplot_kw={'projection': ccrs.Robinson()})
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#f0f8ff')
ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

cmap = plt.cm.RdYlBu_r
sc = ax.scatter(lon, lat, c=risk_change, cmap=cmap, s=0.5, alpha=0.8,
    vmin=-0.1, vmax=0.4, transform=ccrs.PlateCarree())
ax.set_title('Composite Risk Change: SSP5-8.5 minus SSP2-4.5', fontsize=14, fontweight='bold')
fig.colorbar(sc, ax=ax, label='Risk Index Change', shrink=0.6)
plt.savefig('report/images/fig8_risk_change_map.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig8_risk_change_map.png")

# ============================================================
# Figure 9: Cumulative risk distribution
# ============================================================
print("Generating Figure 9: Cumulative risk distribution...")

fig, ax = plt.subplots(figsize=(10, 6))
sorted_245 = np.sort(comp_245)
sorted_370 = np.sort(comp_370)
sorted_585 = np.sort(comp_585)
cdf = np.linspace(0, 1, len(sorted_245))

ax.plot(sorted_245, cdf, 'b-', linewidth=2, label='SSP2-4.5')
ax.plot(sorted_370, cdf, color='orange', linewidth=2, label='SSP3-7.0')
ax.plot(sorted_585, cdf, 'r-', linewidth=2, label='SSP5-8.5')
ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.7, label='High risk threshold')
ax.axvline(x=0.85, color='black', linestyle='--', alpha=0.7, label='Very high risk threshold')
ax.set_xlabel('Composite Risk Index', fontsize=12)
ax.set_ylabel('Cumulative Proportion', fontsize=12)
ax.set_title('Cumulative Distribution of Composite Risk by Scenario', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('report/images/fig9_cumulative_risk.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig9_cumulative_risk.png")

# ============================================================
# Figure 10: Risk category pie charts
# ============================================================
print("Generating Figure 10: Risk category breakdown...")

def categorize(risk):
    cats = np.zeros_like(risk, dtype=int)
    cats[risk < 0.3] = 0  # Low
    cats[(risk >= 0.3) & (risk < 0.5)] = 1  # Moderate
    cats[(risk >= 0.5) & (risk < 0.7)] = 2  # High
    cats[(risk >= 0.7) & (risk < 0.85)] = 3  # Very High
    cats[risk >= 0.85] = 4  # Extreme
    return cats

cat_labels = ['Low\n(<0.3)', 'Moderate\n(0.3–0.5)', 'High\n(0.5–0.7)', 'Very High\n(0.7–0.85)', 'Extreme\n(≥0.85)']
cat_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, comp) in zip(axes, [('SSP2-4.5', comp_245), ('SSP3-7.0', comp_370), ('SSP5-8.5', comp_585)]):
    cats = categorize(comp)
    counts = [np.sum(cats == i) for i in range(5)]
    pcts = [100 * c / len(comp) for c in counts]
    
    wedges, texts, autotexts = ax.pie(counts, labels=None, colors=cat_colors,
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=90)
    ax.set_title(name, fontsize=14, fontweight='bold')

fig.legend(cat_labels, loc='lower center', ncol=5, fontsize=11, 
    bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Risk Category Distribution by Scenario', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig10_risk_categories.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig10_risk_categories.png")

print("\nAll additional figures generated!")
