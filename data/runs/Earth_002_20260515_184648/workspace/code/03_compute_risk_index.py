#!/usr/bin/env python3
"""
Compute composite mangrove risk index from SLR + TC frequency.
Generates summary stats, figures, and final outputs.
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
INPUT_GPKG = "outputs/mangrove_risk_points.gpkg"
OUTPUT_GPKG = "outputs/mangrove_final_risk.gpkg"
IMAGES_DIR = Path("report/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def standardize(x):
    """Min-max standardization to [0,1]"""
    return (x - x.min()) / (x.max() - x.min())

def main():
    print("Loading points...")
    gdf = gpd.read_file(INPUT_GPKG)
    print(f"  {len(gdf)} points loaded")

    # Standardize variables
    gdf['slr_norm_ssp245'] = standardize(gdf['slr_rate_ssp245'])
    gdf['slr_norm_ssp370'] = standardize(gdf['slr_rate_ssp370'])
    gdf['slr_norm_ssp585'] = standardize(gdf['slr_rate_ssp585'])
    gdf['tc_norm'] = standardize(gdf['tc_freq'])

    # Composite risk index (weighted average)
    # Weights: SLR 60%, TC 40% (SLR dominant driver at end-of-century)
    w_slr, w_tc = 0.6, 0.4
    gdf['risk_ssp245'] = w_slr * gdf['slr_norm_ssp245'] + w_tc * gdf['tc_norm']
    gdf['risk_ssp370'] = w_slr * gdf['slr_norm_ssp370'] + w_tc * gdf['tc_norm']
    gdf['risk_ssp585'] = w_slr * gdf['slr_norm_ssp585'] + w_tc * gdf['tc_norm']

    # Save final dataset
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"Saved final risk dataset to {OUTPUT_GPKG}")

    # Summary statistics
    print("\nRisk index summary (SSP5-8.5):")
    print(gdf[['risk_ssp245', 'risk_ssp370', 'risk_ssp585']].describe())

    # === Figure 1: Risk distribution histograms ===
    plt.figure(figsize=(10, 6))
    sns.histplot(gdf['risk_ssp585'], bins=50, color='darkred', label='SSP5-8.5', alpha=0.7)
    sns.histplot(gdf['risk_ssp370'], bins=50, color='orange', label='SSP3-7.0', alpha=0.6)
    sns.histplot(gdf['risk_ssp245'], bins=50, color='steelblue', label='SSP2-4.5', alpha=0.5)
    plt.xlabel("Composite Risk Index (0-1)")
    plt.ylabel("Number of Mangrove Points")
    plt.title("Distribution of Mangrove Risk Index under Three SSP Scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "figure1_risk_histograms.png", dpi=150)
    plt.close()
    print("Saved figure1_risk_histograms.png")

    # === Figure 2: Scatter SLR vs TC (color by risk) ===
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(gdf['slr_rate_ssp585'], gdf['tc_freq'], 
                          c=gdf['risk_ssp585'], cmap='RdYlBu_r', s=8, alpha=0.6)
    plt.colorbar(scatter, label='Risk Index (SSP5-8.5)')
    plt.xlabel("Sea Level Rise Rate (mm/century, SSP5-8.5)")
    plt.ylabel("TC Frequency (events/year)")
    plt.title("SLR vs TC Frequency Colored by Composite Risk (SSP5-8.5)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "figure2_slr_tc_scatter.png", dpi=150)
    plt.close()
    print("Saved figure2_slr_tc_scatter.png")

    # === Figure 3: Boxplot of risk by scenario ===
    risk_df = gdf[['risk_ssp245', 'risk_ssp370', 'risk_ssp585']].melt(
        var_name='Scenario', value_name='Risk'
    )
    risk_df['Scenario'] = risk_df['Scenario'].str.replace('risk_', '')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=risk_df, x='Scenario', y='Risk', palette=['steelblue', 'orange', 'darkred'])
    plt.ylabel("Composite Risk Index")
    plt.title("Mangrove Risk Index by Climate Scenario")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "figure3_risk_boxplots.png", dpi=150)
    plt.close()
    print("Saved figure3_risk_boxplots.png")

    # === Figure 4: Risk vs area (if area column exists) ===
    if 'area' in gdf.columns or 'AREA' in gdf.columns:
        area_col = 'area' if 'area' in gdf.columns else 'AREA'
        plt.figure(figsize=(8, 6))
        plt.scatter(gdf[area_col], gdf['risk_ssp585'], s=10, alpha=0.4, color='darkgreen')
        plt.xlabel("Mangrove Patch Area (ha)")
        plt.ylabel("Risk Index (SSP5-8.5)")
        plt.title("Risk vs Mangrove Patch Size")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "figure4_risk_vs_area.png", dpi=150)
        plt.close()
        print("Saved figure4_risk_vs_area.png")

    print("\nAll figures generated successfully.")
    print(f"Images saved to: {IMAGES_DIR}")

if __name__ == "__main__":
    main()