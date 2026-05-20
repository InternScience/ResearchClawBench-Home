#!/usr/bin/env python3
"""
Enhanced analysis: country-level ecosystem services at risk
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "outputs"
IMG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

print("Loading data...")
# Load mangrove risk results
risk_df = pd.read_csv(f"{OUTPUT_DIR}/mangrove_risk_index.csv")

# Load country bounds with ecosystem service data
countries = gpd.read_file("data/ecosystem/UCSC_CWON_countrybounds.gpkg")

# Load mangrove points for spatial join
mangrove_gdf = gpd.read_file("data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg")

print(f"Countries: {len(countries)}")
print(f"Mangrove points: {len(mangrove_gdf)}")

# ---------------------------------------------------------------------------
# Spatial join: assign each mangrove point to a country
# ---------------------------------------------------------------------------
print("\nPerforming spatial join...")
# Ensure same CRS
mangrove_gdf = mangrove_gdf.set_crs(epsg=4326, allow_override=True)
countries = countries.to_crs(epsg=4326)

joined = gpd.sjoin(mangrove_gdf, countries[["Country", "ISO3", "Mang_Ha_2020", "Risk_Pop_2020", "Risk_Stock_2020", "Ben_Pop_2020", "Ben_Stock_2020", "geometry"]], how="left", predicate="within")
print(f"Joined points: {len(joined)}")
print(f"Points with country: {joined['Country'].notna().sum()}")

# Add risk data
for col in ["slr_rate_ssp245", "slr_rate_ssp370", "slr_rate_ssp585",
            "slr_risk_ssp245", "slr_risk_ssp370", "slr_risk_ssp585",
            "tc_baseline", "tc_risk_ssp245", "tc_risk_ssp370", "tc_risk_ssp585",
            "composite_ssp245", "composite_ssp370", "composite_ssp585"]:
    joined[col] = risk_df[col].values

# ---------------------------------------------------------------------------
# Country-level aggregation
# ---------------------------------------------------------------------------
print("\nAggregating by country...")

country_summary = []
for country in joined["Country"].dropna().unique():
    mask = joined["Country"] == country
    n_pts = mask.sum()
    if n_pts == 0:
        continue
    
    row = {"country": country, "n_points": n_pts}
    
    # Get country-level ecosystem service data
    cdata = countries[countries["Country"] == country].iloc[0]
    row["mangrove_ha"] = cdata["Mang_Ha_2020"] if pd.notna(cdata["Mang_Ha_2020"]) else 0
    row["risk_pop"] = cdata["Risk_Pop_2020"] if pd.notna(cdata["Risk_Pop_2020"]) else 0
    row["risk_stock_usd"] = cdata["Risk_Stock_2020"] if pd.notna(cdata["Risk_Stock_2020"]) else 0
    row["ben_pop"] = cdata["Ben_Pop_2020"] if pd.notna(cdata["Ben_Pop_2020"]) else 0
    row["ben_stock_usd"] = cdata["Ben_Stock_2020"] if pd.notna(cdata["Ben_Stock_2020"]) else 0
    
    for ssp in ["ssp245", "ssp370", "ssp585"]:
        comp = joined[f"composite_{ssp}"][mask].values
        slr = joined[f"slr_risk_{ssp}"][mask].values
        tc = joined[f"tc_risk_{ssp}"][mask].values
        
        row[f"composite_mean_{ssp}"] = comp.mean()
        row[f"composite_high_pct_{ssp}"] = (comp > 0.5).mean() * 100
        row[f"composite_extreme_pct_{ssp}"] = (comp > 0.7).mean() * 100
        
        # Ecosystem services at risk: scale by proportion of high-risk points
        high_risk_frac = (comp > 0.5).mean()
        row[f"mangrove_ha_at_risk_{ssp}"] = row["mangrove_ha"] * high_risk_frac
        row[f"risk_pop_at_risk_{ssp}"] = row["risk_pop"] * high_risk_frac
        row[f"risk_stock_at_risk_usd_{ssp}"] = row["risk_stock_usd"] * high_risk_frac
    
    country_summary.append(row)

country_df = pd.DataFrame(country_summary)
country_df.to_csv(f"{OUTPUT_DIR}/country_risk_summary.csv", index=False)
print(f"Saved {OUTPUT_DIR}/country_risk_summary.csv")

# ---------------------------------------------------------------------------
# Global ecosystem services at risk
# ---------------------------------------------------------------------------
print("\nGlobal ecosystem services at risk:")
for ssp in ["ssp245", "ssp370", "ssp585"]:
    total_ha = country_df["mangrove_ha"].sum()
    ha_at_risk = country_df[f"mangrove_ha_at_risk_{ssp}"].sum()
    pop_at_risk = country_df[f"risk_pop_at_risk_{ssp}"].sum()
    stock_at_risk = country_df[f"risk_stock_at_risk_usd_{ssp}"].sum()
    print(f"  {ssp}: {ha_at_risk:,.0f} ha at risk ({ha_at_risk/total_ha*100:.1f}%)")
    print(f"         {pop_at_risk:,.0f} people at risk")
    print(f"         ${stock_at_risk/1e9:,.1f}B coastal stock at risk")

# ---------------------------------------------------------------------------
# Top 20 countries by composite risk
# ---------------------------------------------------------------------------
print("\nTop 10 countries by composite risk (SSP585):")
top = country_df.nlargest(10, "composite_mean_ssp585")[["country", "composite_mean_ssp585", "composite_high_pct_ssp585", "mangrove_ha"]]
print(top.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 7: Top countries by mangrove area at risk
# ---------------------------------------------------------------------------
print("\nGenerating additional figures...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, ssp in zip(axes, ["ssp245", "ssp370", "ssp585"]):
    top20 = country_df.nlargest(20, f"mangrove_ha_at_risk_{ssp}").sort_values(f"mangrove_ha_at_risk_{ssp}")
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20[f"mangrove_ha_at_risk_{ssp}"], color="forestgreen", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["country"], fontsize=8)
    ax.set_xlabel("Mangrove area at risk (ha)")
    ax.set_title(f"{ssp.upper()}")
fig.suptitle("Top 20 Countries by Mangrove Area at High Composite Risk", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig7_country_area_at_risk.png", dpi=300)
plt.close()
print("  Figure 7 saved")

# Figure 8: Ecosystem services at risk comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = [
    ("mangrove_ha_at_risk", "Mangrove Area at Risk (ha)", 1),
    ("risk_pop_at_risk", "Population at Risk (millions)", 1e6),
    ("risk_stock_at_risk_usd", "Coastal Stock at Risk (US$ billions)", 1e9),
]
for ax, (base_col, label, scale) in zip(axes, metrics):
    vals = []
    for ssp in ["ssp245", "ssp370", "ssp585"]:
        vals.append(country_df[f"{base_col}_{ssp}"].sum() / scale)
    bars = ax.bar(["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"], vals, color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.8)
    ax.set_ylabel(label)
    ax.set_title(label)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01, 
                f"{val:,.1f}", ha="center", va="bottom", fontsize=10)
fig.suptitle("Global Ecosystem Services at High Composite Risk by Scenario (2100)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig8_ecosystem_services_at_risk.png", dpi=300)
plt.close()
print("  Figure 8 saved")

# Figure 9: Scatter plot of TC risk vs SLR risk
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ssp in zip(axes, ["ssp245", "ssp370", "ssp585"]):
    slr = risk_df[f"slr_risk_{ssp}"].values
    tc = risk_df[f"tc_risk_{ssp}"].values
    # Normalize tc for visualization
    tc_norm = np.clip(tc / np.percentile(tc, 95), 0, 1)
    scatter = ax.scatter(slr, tc_norm, c=risk_df[f"composite_{ssp}"], cmap="YlOrRd", s=2, alpha=0.5)
    ax.set_xlabel("SLR Risk Score")
    ax.set_ylabel("Normalized TC Risk")
    ax.set_title(f"{ssp.upper()}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(scatter, ax=ax, label="Composite Risk")
fig.suptitle("Relationship between SLR Risk and TC Risk at Mangrove Locations", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig9_slr_vs_tc_scatter.png", dpi=300)
plt.close()
print("  Figure 9 saved")

# Figure 10: Global heatmap of composite risk (2D histogram)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, ssp in zip(axes, ["ssp245", "ssp370", "ssp585"]):
    h, xedges, yedges = np.histogram2d(risk_df["lon"], risk_df["lat"], bins=[180, 90], weights=risk_df[f"composite_{ssp}"])
    h_counts, _, _ = np.histogram2d(risk_df["lon"], risk_df["lat"], bins=[180, 90])
    h = np.divide(h, h_counts, out=np.zeros_like(h), where=h_counts > 0)
    im = ax.imshow(h.T, origin="lower", extent=[-180, 180, -90, 90], cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{ssp.upper()}")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean Composite Risk")
fig.suptitle("Global Heatmap of Mean Composite Risk in 2° Bins", fontsize=14)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig10_composite_heatmap.png", dpi=300)
plt.close()
print("  Figure 10 saved")

print("\nEnhanced analysis complete!")
