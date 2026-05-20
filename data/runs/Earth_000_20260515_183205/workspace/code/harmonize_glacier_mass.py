#!/usr/bin/env python3
"""
GlaMBIE Glacier Mass Change Harmonization Script
Produces 2000-2023 annual regional and global time series (m w.e. and Gt)
with uncertainties from GlaMBIE calendar-year consensus estimates.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("data/glambie/results/calendar_years")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR = Path("report/images")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Regions (excluding global)
REGIONS = [
    "1_alaska", "2_western_canada_us", "3_arctic_canada_north",
    "4_arctic_canada_south", "5_greenland_periphery", "6_iceland",
    "7_svalbard", "8_scandinavia", "9_russian_arctic",
    "10_north_asia", "11_central_europe", "12_caucasus_middle_east",
    "13_central_asia", "14_south_asia_west", "15_south_asia_east",
    "16_low_latitudes", "17_southern_andes", "18_new_zealand",
    "19_antarctic_and_subantarctic"
]

def load_region_csv(region):
    """Load a single region CSV and add integer year column."""
    fp = DATA_DIR / f"{region}.csv"
    df = pd.read_csv(fp)
    df["year"] = df["start_dates"].astype(int)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2023)]
    df["region_name"] = region.split("_", 1)[1].replace("_", " ").title()
    return df

def create_annual_series():
    """Create harmonized annual time series for all regions + global."""
    all_dfs = []
    for region in REGIONS:
        df = load_region_csv(region)
        all_dfs.append(df)

    regional = pd.concat(all_dfs, ignore_index=True)

    # Global from 0_global.csv
    global_df = pd.read_csv(DATA_DIR / "0_global.csv")
    global_df["year"] = global_df["start_dates"].astype(int)
    global_df = global_df[(global_df["year"] >= 2000) & (global_df["year"] <= 2023)]
    global_df["region_name"] = "Global"

    # Save processed data
    regional.to_csv(OUTPUT_DIR / "regional_annual_mass_change.csv", index=False)
    global_df.to_csv(OUTPUT_DIR / "global_annual_mass_change.csv", index=False)

    return regional, global_df

def plot_global_time_series(global_df):
    """Main global mass change time series figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    years = global_df["year"].values
    mwe = global_df["combined_mwe"].values
    mwe_err = global_df["combined_mwe_errors"].values
    gt = global_df["combined_gt"].values
    gt_err = global_df["combined_gt_errors"].values

    # m w.e. plot
    ax.errorbar(years, mwe, yerr=mwe_err, fmt="o-", color="#1f77b4",
                capsize=3, label="Specific mass change (m w.e.)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Specific mass change (m w.e. yr⁻¹)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    # Gt on secondary axis
    ax2 = ax.twinx()
    ax2.errorbar(years, gt, yerr=gt_err, fmt="s--", color="#d62728",
                 capsize=3, label="Total mass change (Gt yr⁻¹)")
    ax2.set_ylabel("Total mass change (Gt yr⁻¹)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax.set_title("Global Glacier Mass Change 2000–2023 (GlaMBIE Consensus)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "global_mass_change_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_regional_overview(regional_df):
    """Heatmap of regional trends."""
    pivot = regional_df.pivot(index="region_name", columns="year", values="combined_mwe")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "m w.e. yr⁻¹"})
    ax.set_title("Regional Glacier Specific Mass Change (m w.e. yr⁻¹)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regional_mass_change_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_cumulative_global(global_df):
    """Cumulative mass loss figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    years = global_df["year"].values
    cum_gt = global_df["combined_gt"].cumsum()
    cum_err = np.sqrt((global_df["combined_gt_errors"]**2).cumsum())

    ax.fill_between(years, cum_gt - cum_err, cum_gt + cum_err,
                    alpha=0.3, color="#d62728", label="±1σ uncertainty")
    ax.plot(years, cum_gt, "o-", color="#d62728", label="Cumulative Gt")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative mass change (Gt)")
    ax.set_title("Cumulative Global Glacier Mass Loss 2000–2023")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cumulative_global_mass_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Harmonizing GlaMBIE glacier mass change data...")
    regional, global_df = create_annual_series()
    print(f"Regional data: {len(regional)} rows")
    print(f"Global data: {len(global_df)} rows (2000-2023)")

    print("Generating figures...")
    plot_global_time_series(global_df)
    plot_regional_overview(regional)
    plot_cumulative_global(global_df)

    print("All outputs saved to outputs/ and report/images/")
    print("Done.")