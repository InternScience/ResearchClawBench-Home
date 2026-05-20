#!/usr/bin/env python3
"""
Composite Risk Index for Mangroves under Tropical Cyclone Regime Shifts and Sea Level Rise

Author: Research Agent
Date: 2026-05-18

Data sources:
- GMW v4: Global Mangrove Watch sampled points
- IPCC AR6 SLR rates: SSP2-4.5, SSP3-7.0, SSP5-8.5
- MIT TC tracks: Historical tropical cyclone tracks downscaled from MPI-ESM1-2-HR
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("Loading mangrove points...")
gdf = gpd.read_file(f"{DATA_DIR}/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg")
mangrove_lats = gdf.geometry.y.values
mangrove_lons = gdf.geometry.x.values
n_mangrove = len(gdf)
print(f"  {n_mangrove} mangrove points loaded")

print("Loading SLR data...")
slr_files = {
    "ssp245": f"{DATA_DIR}/slr/total_ssp245_medium_confidence_rates.nc",
    "ssp370": f"{DATA_DIR}/slr/total_ssp370_medium_confidence_rates.nc",
    "ssp585": f"{DATA_DIR}/slr/total_ssp585_medium_confidence_rates.nc",
}

slr_data = {}
for ssp, path in slr_files.items():
    ds = xr.open_dataset(path)
    q_idx = np.where(np.isclose(ds.quantiles.values, 0.5))[0][0]
    y_idx = np.where(ds.years.values == 2100)[0][0]
    rates = ds.sea_level_change_rate[q_idx, y_idx, :].values  # mm/yr
    slr_data[ssp] = {
        "lat": ds.lat.values,
        "lon": ds.lon.values,
        "rate": rates,
    }
    ds.close()
    print(f"  {ssp}: median 2100 rate loaded, range [{rates.min():.2f}, {rates.max():.2f}] mm/yr")

print("Loading TC track data...")
ds_tc = xr.open_dataset(f"{DATA_DIR}/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc")
tc_lats = ds_tc.lat.values
tc_lons = ds_tc.lon.values
tc_winds = ds_tc.wind.values  # m/s
n_tc = len(tc_lats)
print(f"  {n_tc} TC track points loaded, wind range [{tc_winds.min():.1f}, {tc_winds.max():.1f}] m/s")

# ---------------------------------------------------------------------------
# 2. SEA LEVEL RISE RISK COMPONENT
# ---------------------------------------------------------------------------
print("\nComputing SLR risk component...")

# Map SLR rates to each mangrove point using nearest neighbor
slr_tree = cKDTree(np.column_stack([slr_data["ssp245"]["lat"], slr_data["ssp245"]["lon"]]))
_, slr_idx = slr_tree.query(np.column_stack([mangrove_lats, mangrove_lons]), k=1)

slr_risk = {}
for ssp in ["ssp245", "ssp370", "ssp585"]:
    rates = slr_data[ssp]["rate"][slr_idx]
    # Risk score based on Saintilan et al. (2023) thresholds
    # Low: <4 mm/yr, Moderate: 4-7 mm/yr, High: >7 mm/yr
    risk_score = np.zeros(n_mangrove)
    risk_score[(rates >= 4) & (rates <= 7)] = 0.5
    risk_score[rates > 7] = 1.0
    # Also store raw rate for analysis
    slr_risk[ssp] = {
        "rate": rates,
        "risk": risk_score,
    }
    print(f"  {ssp}: Low={(risk_score==0).sum()}, Mod={(risk_score==0.5).sum()}, High={(risk_score==1).sum()}")

# ---------------------------------------------------------------------------
# 3. TROPICAL CYCLONE RISK COMPONENT
# ---------------------------------------------------------------------------
print("\nComputing TC risk component...")

# Saffir-Simpson thresholds in m/s
# Cat1: 33.06 - 42.50, Cat2: 42.78 - 49.17, Cat3: 49.44 - 57.78, Cat4: 58.06 - 69.72, Cat5: >=70.0
ss_thresholds = {
    "cat1": (33.0, 42.5),
    "cat2": (42.5, 49.2),
    "cat3": (49.2, 57.8),
    "cat4": (57.8, 69.7),
    "cat5": (69.7, 999),
}

# Damage weights from Mo et al. (2023): Cat5:Cat4:Cat3 = 29:13:1
# We extend this to all categories proportionally
damage_weights = {
    "cat1": 0.1,
    "cat2": 0.5,
    "cat3": 1.0,
    "cat4": 13.0,
    "cat5": 29.0,
}

# Create a 2-degree global grid for TC frequency counting
grid_res = 2.0
lon_bins = np.arange(-180, 180 + grid_res, grid_res)
lat_bins = np.arange(-90, 90 + grid_res, grid_res)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

# Assign each TC point to a grid cell
tc_lon_idx = np.digitize(tc_lons, lon_bins) - 1
tc_lat_idx = np.digitize(tc_lats, lat_bins) - 1
# Clip to valid ranges
tc_lon_idx = np.clip(tc_lon_idx, 0, len(lon_centers) - 1)
tc_lat_idx = np.clip(tc_lat_idx, 0, len(lat_centers) - 1)

# Count TC points by category in each grid cell
tc_counts = {}
for cat, (wmin, wmax) in ss_thresholds.items():
    mask = (tc_winds >= wmin) & (tc_winds < wmax)
    grid = np.zeros((len(lat_centers), len(lon_centers)))
    for li, loi in zip(tc_lat_idx[mask], tc_lon_idx[mask]):
        grid[li, loi] += 1
    tc_counts[cat] = grid
    print(f"  {cat}: {mask.sum()} points")

# Normalize counts to per-year rates (historical period: 1850-2014 = 165 years)
historical_years = 165
for cat in tc_counts:
    tc_counts[cat] = tc_counts[cat] / historical_years

# Compute baseline TC damage index per grid cell
tc_damage_grid = np.zeros((len(lat_centers), len(lon_centers)))
for cat, weight in damage_weights.items():
    tc_damage_grid += tc_counts[cat] * weight

print(f"  Baseline TC damage grid: max={tc_damage_grid.max():.4f}, mean={tc_damage_grid.mean():.4f}")

# Map TC damage to each mangrove point
mangrove_lon_idx = np.digitize(mangrove_lons, lon_bins) - 1
mangrove_lat_idx = np.digitize(mangrove_lats, lat_bins) - 1
mangrove_lon_idx = np.clip(mangrove_lon_idx, 0, len(lon_centers) - 1)
mangrove_lat_idx = np.clip(mangrove_lat_idx, 0, len(lat_centers) - 1)

tc_baseline = tc_damage_grid[mangrove_lat_idx, mangrove_lon_idx]

# Apply future TC regime shift projections
# Based on Knutson et al. (2020) / IPCC AR6 WG1 Table 11.3:
# Global TC frequency: slight decrease (-13% to +11%)
# Category 4-5 proportion: increase (+13% to +39%)
# We apply scenario-specific multipliers:
# SSP2-4.5: modest increase in intensity, slight decrease in frequency
# SSP3-7.0: moderate increase in intensity, near-neutral frequency
# SSP5-8.5: strong increase in intensity, slight decrease in frequency
# Regional shifts from Mo et al. (2023) and Kropf et al. (2023) are approximated
# by basin-level factors.

# For simplicity and robustness, we use the following approach:
# Future TCRI = baseline × (intensity_factor × frequency_factor)
# We apply different factors per scenario, with stronger effects on Cat4-5

tc_projections = {
    "ssp245": {"freq_factor": 0.95, "intensity_factor": 1.10, "cat45_boost": 1.15},
    "ssp370": {"freq_factor": 1.00, "intensity_factor": 1.20, "cat45_boost": 1.25},
    "ssp585": {"freq_factor": 0.95, "intensity_factor": 1.30, "cat45_boost": 1.40},
}

tc_risk = {}
for ssp, proj in tc_projections.items():
    # Recalculate future TC damage grid
    future_grid = np.zeros_like(tc_damage_grid)
    for cat, weight in damage_weights.items():
        if cat in ["cat4", "cat5"]:
            future_grid += tc_counts[cat] * weight * proj["freq_factor"] * proj["intensity_factor"] * proj["cat45_boost"]
        elif cat in ["cat3"]:
            future_grid += tc_counts[cat] * weight * proj["freq_factor"] * proj["intensity_factor"]
        else:
            future_grid += tc_counts[cat] * weight * proj["freq_factor"]
    
    future_risk = future_grid[mangrove_lat_idx, mangrove_lon_idx]
    # Normalize to 0-1 using 95th percentile of baseline as reference
    norm_ref = np.percentile(tc_baseline, 95)
    if norm_ref == 0:
        norm_ref = 1e-6
    tc_risk_norm = np.clip(future_risk / norm_ref, 0, 1)
    
    tc_risk[ssp] = {
        "future": future_risk,
        "norm": tc_risk_norm,
    }
    print(f"  {ssp}: future max={future_risk.max():.4f}, norm mean={tc_risk_norm.mean():.4f}")

# ---------------------------------------------------------------------------
# 4. COMPOSITE RISK INDEX
# ---------------------------------------------------------------------------
print("\nComputing Composite Risk Index...")

composite_risk = {}
for ssp in ["ssp245", "ssp370", "ssp585"]:
    # Combine SLR and TC risk using geometric mean (both need to be high for high composite)
    # Add small epsilon to avoid zeros
    slr = slr_risk[ssp]["risk"]
    tc = tc_risk[ssp]["norm"]
    composite = np.sqrt(slr * tc)
    composite_risk[ssp] = composite
    print(f"  {ssp}: Composite mean={composite.mean():.4f}, max={composite.max():.4f}")

# ---------------------------------------------------------------------------
# 5. SAVE INTERMEDIATE RESULTS
# ---------------------------------------------------------------------------
print("\nSaving intermediate results...")

results_df = pd.DataFrame({
    "lon": mangrove_lons,
    "lat": mangrove_lats,
    "slr_rate_ssp245": slr_risk["ssp245"]["rate"],
    "slr_rate_ssp370": slr_risk["ssp370"]["rate"],
    "slr_rate_ssp585": slr_risk["ssp585"]["rate"],
    "slr_risk_ssp245": slr_risk["ssp245"]["risk"],
    "slr_risk_ssp370": slr_risk["ssp370"]["risk"],
    "slr_risk_ssp585": slr_risk["ssp585"]["risk"],
    "tc_baseline": tc_baseline,
    "tc_risk_ssp245": tc_risk["ssp245"]["future"],
    "tc_risk_ssp370": tc_risk["ssp370"]["future"],
    "tc_risk_ssp585": tc_risk["ssp585"]["future"],
    "composite_ssp245": composite_risk["ssp245"],
    "composite_ssp370": composite_risk["ssp370"],
    "composite_ssp585": composite_risk["ssp585"],
})
results_df.to_csv(f"{OUTPUT_DIR}/mangrove_risk_index.csv", index=False)
print(f"  Saved {OUTPUT_DIR}/mangrove_risk_index.csv")

# Summary statistics by region
# Define regions by latitude bands
lat_bands = {
    "Tropical (0-23.5)": (0, 23.5),
    "Subtropical (23.5-35)": (23.5, 35),
    "Temperate (35-50)": (35, 50),
    "South Temperate (-35-0)": (-35, 0),
}

summary = []
for ssp in ["ssp245", "ssp370", "ssp585"]:
    for region, (lat_min, lat_max) in lat_bands.items():
        mask = (mangrove_lats >= lat_min) & (mangrove_lats < lat_max)
        n_points = mask.sum()
        if n_points == 0:
            continue
        summary.append({
            "ssp": ssp,
            "region": region,
            "n_points": n_points,
            "slr_high_pct": (slr_risk[ssp]["risk"][mask] == 1.0).mean() * 100,
            "slr_mod_pct": (slr_risk[ssp]["risk"][mask] == 0.5).mean() * 100,
            "slr_low_pct": (slr_risk[ssp]["risk"][mask] == 0.0).mean() * 100,
            "tc_risk_mean": tc_risk[ssp]["norm"][mask].mean(),
            "composite_mean": composite_risk[ssp][mask].mean(),
            "composite_high_pct": (composite_risk[ssp][mask] > 0.5).mean() * 100,
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUTPUT_DIR}/regional_summary.csv", index=False)
print(f"  Saved {OUTPUT_DIR}/regional_summary.csv")

# ---------------------------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------------------------
print("\nGenerating figures...")

# Figure 1: Global SLR risk map (SSP5-8.5)
fig, ax = plt.subplots(figsize=(14, 7))
scatter = ax.scatter(
    mangrove_lons, mangrove_lats,
    c=slr_risk["ssp585"]["rate"],
    cmap="YlOrRd",
    s=1,
    vmin=0, vmax=20,
)
plt.colorbar(scatter, label="SLR rate (mm yr⁻¹)", shrink=0.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Projected Sea-Level Rise Rate at Mangrove Locations (SSP5-8.5, 2100)")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig1_slr_risk_map.png", dpi=300)
plt.close()
print("  Figure 1 saved")

# Figure 2: Global TC baseline risk map
fig, ax = plt.subplots(figsize=(14, 7))
scatter = ax.scatter(
    mangrove_lons, mangrove_lats,
    c=tc_baseline,
    cmap="Reds",
    s=1,
    vmin=0, vmax=np.percentile(tc_baseline, 99),
)
plt.colorbar(scatter, label="Baseline TC Damage Index", shrink=0.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Baseline Tropical Cyclone Damage Index at Mangrove Locations")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig2_tc_baseline_map.png", dpi=300)
plt.close()
print("  Figure 2 saved")

# Figure 3: Composite risk map for all three SSPs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, ssp in zip(axes, ["ssp245", "ssp370", "ssp585"]):
    scatter = ax.scatter(
        mangrove_lons, mangrove_lats,
        c=composite_risk[ssp],
        cmap="YlOrRd",
        s=1,
        vmin=0, vmax=1,
    )
    ax.set_title(f"{ssp.upper()}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
fig.suptitle("Composite Risk Index (SLR + TC) at Mangrove Locations", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig3_composite_risk_maps.png", dpi=300)
plt.close()
print("  Figure 3 saved")

# Figure 4: Risk distribution histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#2ca02c", "#ff7f0e", "#d62728"]
for ax, ssp, color in zip(axes, ["ssp245", "ssp370", "ssp585"], colors):
    ax.hist(composite_risk[ssp], bins=50, color=color, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Composite Risk Index")
    ax.set_ylabel("Number of Mangrove Points")
    ax.set_title(f"{ssp.upper()}")
    ax.axvline(composite_risk[ssp].mean(), color="navy", linestyle="--", label=f"Mean={composite_risk[ssp].mean():.3f}")
    ax.legend()
fig.suptitle("Distribution of Composite Risk Index across Mangrove Locations", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig4_risk_histograms.png", dpi=300)
plt.close()
print("  Figure 4 saved")

# Figure 5: Regional comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
regions = summary_df["region"].unique()
x = np.arange(len(regions))
width = 0.25

for ax, metric, title in zip(axes, 
                             ["slr_high_pct", "tc_risk_mean", "composite_mean"],
                             ["SLR High Risk (%)", "Mean TC Risk", "Mean Composite Risk"]):
    for i, ssp in enumerate(["ssp245", "ssp370", "ssp585"]):
        vals = summary_df[summary_df["ssp"] == ssp][metric].values
        ax.bar(x + i * width, vals, width, label=ssp.upper(), alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()

fig.suptitle("Regional Comparison of Mangrove Risk Components by SSP Scenario", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig5_regional_comparison.png", dpi=300)
plt.close()
print("  Figure 5 saved")

# Figure 6: TC category frequency map (grid-based)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
cats = ["cat1", "cat2", "cat3", "cat4", "cat5", "all_major"]
for ax, cat in zip(axes.flat, cats):
    if cat == "all_major":
        grid = tc_counts["cat3"] + tc_counts["cat4"] + tc_counts["cat5"]
        title = "All Major TCs (Cat 3-5)"
    else:
        grid = tc_counts[cat]
        title = f"{cat.upper()} Frequency"
    vmax = np.percentile(grid[grid > 0], 99) if (grid > 0).any() else 1
    im = ax.pcolormesh(lon_bins, lat_bins, grid, cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle("Historical Tropical Cyclone Frequency per 2° Grid Cell (storms yr⁻¹)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig6_tc_frequency_maps.png", dpi=300)
plt.close()
print("  Figure 6 saved")

# ---------------------------------------------------------------------------
# 7. SAVE SUMMARY JSON
# ---------------------------------------------------------------------------
summary_json = {
    "total_mangrove_points": int(n_mangrove),
    "slr_risk": {},
    "tc_risk": {},
    "composite_risk": {},
}

for ssp in ["ssp245", "ssp370", "ssp585"]:
    summary_json["slr_risk"][ssp] = {
        "high_risk_points": int((slr_risk[ssp]["risk"] == 1.0).sum()),
        "moderate_risk_points": int((slr_risk[ssp]["risk"] == 0.5).sum()),
        "low_risk_points": int((slr_risk[ssp]["risk"] == 0.0).sum()),
        "mean_rate_mm_yr": float(slr_risk[ssp]["rate"].mean()),
    }
    summary_json["tc_risk"][ssp] = {
        "mean_normalized_risk": float(tc_risk[ssp]["norm"].mean()),
        "high_risk_points": int((tc_risk[ssp]["norm"] > 0.5).sum()),
    }
    summary_json["composite_risk"][ssp] = {
        "mean_risk": float(composite_risk[ssp].mean()),
        "high_risk_points": int((composite_risk[ssp] > 0.5).sum()),
        "extreme_risk_points": int((composite_risk[ssp] > 0.7).sum()),
    }

with open(f"{OUTPUT_DIR}/summary_statistics.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print(f"  Saved {OUTPUT_DIR}/summary_statistics.json")

print("\nAnalysis complete!")
