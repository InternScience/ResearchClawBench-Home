#!/usr/bin/env python3
"""
HEEW Mini-Dataset Analysis Script
- Data loading and cleaning
- Correlation analysis
- Hierarchical aggregation verification
- Figure generation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("data/HEEW_Mini-Dataset")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("report/images")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BUILDINGS = [f"BN{str(i).zfill(3)}" for i in range(1, 11)]
ALL_LEVELS = BUILDINGS + ["CN01", "Total"]

# Column mapping
ENERGY_COLS = [
    "Electricity [kW]",
    "Heat [mmBTU]",
    "Cooling Energy [Ton]",
    "PV Power Generation [kW]",
    "Greenhouse Gas Emission [Ton]",
]
WEATHER_COLS = [
    "Temperature [°F]",
    "Dew Point [°F]",
    "Humidity [%]",
    "Wind Speed [mph]",
    "Wind Gust [mph]",
    "Pressure [in]",
    "Precipitation [in]",
]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


def load_energy(bldg: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{bldg}_energy.csv")
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H",
    )
    df = df.set_index("datetime")
    return df


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "Total_weather.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")
    return df


def clean_data(df: pd.DataFrame, is_energy: bool = True) -> pd.DataFrame:
    """Basic cleaning: missing values, negative values, outliers."""
    df = df.copy()
    if is_energy:
        # Negative values should not exist
        for col in ENERGY_COLS:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        # Simple IQR outlier capping
        for col in ENERGY_COLS:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                upper = q3 + 3 * iqr
                df[col] = df[col].clip(upper=upper)
    else:
        for col in WEATHER_COLS:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df[col] = df[col].clip(lower=q1 - 3 * iqr, upper=q3 + 3 * iqr)
    return df


def verify_hierarchy(df_dict: dict) -> pd.DataFrame:
    """Verify that sum(BN001..BN010) ≈ CN01 ≈ Total"""
    results = []
    for col in ENERGY_COLS:
        building_sum = sum(df_dict[b][col] for b in BUILDINGS)
        cn01 = df_dict["CN01"][col]
        total = df_dict["Total"][col]

        err_cn01 = np.abs(building_sum - cn01) / (cn01 + 1e-8)
        err_total = np.abs(cn01 - total) / (total + 1e-8)

        results.append(
            {
                "Variable": col,
                "Mean_Building_Sum": building_sum.mean(),
                "Mean_CN01": cn01.mean(),
                "Mean_Total": total.mean(),
                "Rel_Error_Buildings_to_CN01": err_cn01.mean(),
                "Rel_Error_CN01_to_Total": err_total.mean(),
            }
        )
    return pd.DataFrame(results)


def plot_correlation(df: pd.DataFrame, title: str, filename: str):
    """Correlation heatmap for energy variables."""
    corr = df[ENERGY_COLS].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename)
    plt.close()


def plot_hierarchy_comparison(df_dict: dict, var: str, filename: str):
    """Plot monthly mean comparison for hierarchy verification."""
    monthly = {}
    for level in ["Total", "CN01"] + BUILDINGS[:3]:  # show first 3 buildings for clarity
        monthly[level] = df_dict[level][var].resample("M").mean()

    plt.figure(figsize=(10, 5))
    for level, series in monthly.items():
        plt.plot(series.index, series.values, label=level, linewidth=1.5)
    plt.title(f"Monthly Mean {var} - Hierarchical Comparison")
    plt.xlabel("Month")
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename)
    plt.close()


def main():
    print("Loading data...")
    data = {}
    for b in BUILDINGS + ["CN01", "Total"]:
        data[b] = load_energy(b)
        data[b] = clean_data(data[b], is_energy=True)

    weather = load_weather()
    weather = clean_data(weather, is_energy=False)

    print("Verifying hierarchical aggregation...")
    hierarchy_results = verify_hierarchy(data)
    hierarchy_results.to_csv(OUTPUT_DIR / "hierarchy_verification.csv", index=False)
    print(hierarchy_results)

    print("Generating figures...")
    # Correlation for Total
    plot_correlation(
        data["Total"],
        "Correlation Matrix - Total Area (2014)",
        "figure1_correlation_total.png",
    )

    # Correlation for one building
    plot_correlation(
        data["BN001"],
        "Correlation Matrix - Building BN001 (2014)",
        "figure2_correlation_bn001.png",
    )

    # Hierarchy comparison for Electricity
    plot_hierarchy_comparison(
        data, "Electricity [kW]", "figure3_hierarchy_electricity.png"
    )

    # Weather overview
    plt.figure(figsize=(10, 4))
    weather["Temperature [°F]"].resample("D").mean().plot()
    plt.title("Daily Mean Temperature - 2014")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_temperature.png")
    plt.close()

    print("Analysis complete. Figures saved to report/images/")


if __name__ == "__main__":
    main()