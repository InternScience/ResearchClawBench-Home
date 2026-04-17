#!/usr/bin/env python3
"""
Regional analysis: Aggregate risk by country/region using ecosystem boundaries.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")

def load_ecosystem_boundaries():
    """Load country/region boundaries for mangrove areas."""
    filepath = DATA_DIR / "ecosystem" / "UCSC_CWON_countrybounds.gpkg"
    gdf = gpd.read_file(filepath, engine="pyogrio")
    print(f"Loaded {len(gdf)} country/region boundaries")
    print(f"Columns: {list(gdf.columns)}")
    return gdf


def assign_countries_to_mangroves(mangrove_df, bounds_gdf):
    """Assign country/region to each mangrove grid cell using spatial join."""
    # Create GeoDataFrame from mangrove points
    mangrove_gdf = gpd.GeoDataFrame(
        mangrove_df,
        geometry=gpd.points_from_xy(mangrove_df["lon_mean"], mangrove_df["lat_mean"]),
        crs="EPSG:4326"
    )
    
    # Spatial join - use 'Country' column from bounds
    joined = gpd.sjoin(mangrove_gdf, bounds_gdf, how="left", predicate="within")
    
    # Check what columns we have
    print(f"Joined columns (sample): {[c for c in joined.columns if 'Country' in c or 'ISO' in c]}")
    
    # Use the 'Country' column (or first country-like column)
    country_col = None
    for col in ["Country", "Country_2020", "Country_2010", "Country_1996"]:
        if col in joined.columns:
            country_col = col
            break
    
    if country_col:
        print(f"Using column '{country_col}' for country assignment")
        print(f"Mangrove cells with country assignment: {joined[country_col].notna().sum()} / {len(joined)}")
    else:
        print("WARNING: No country column found!")
        joined["Country"] = "Unknown"
        country_col = "Country"
    
    return joined, country_col


def aggregate_by_region(mangrove_with_country, scenario, country_col):
    """Aggregate risk metrics by country/region."""
    df = mangrove_with_country[mangrove_with_country["scenario"] == scenario].copy()
    
    # Group by country - use mangrove_count instead of uid
    regional_stats = df.groupby(country_col).agg(
        n_mangrove_cells=("mangrove_count", "sum"),
        mean_slr_rate=("slr_rate", "mean"),
        mean_slr_risk=("slr_risk", "mean"),
        mean_tc_risk=("tc_risk", "mean"),
        mean_composite_risk=("composite_risk", "mean"),
        std_composite_risk=("composite_risk", "std"),
        n_high_risk=("risk_class", lambda x: (x == "High").sum()),
        n_medium_risk=("risk_class", lambda x: (x == "Medium").sum()),
        n_low_risk=("risk_class", lambda x: (x == "Low").sum())
    ).reset_index()
    
    # Calculate percentages
    regional_stats["pct_high_risk"] = regional_stats["n_high_risk"] / regional_stats["n_mangrove_cells"] * 100
    regional_stats["pct_medium_risk"] = regional_stats["n_medium_risk"] / regional_stats["n_mangrove_cells"] * 100
    regional_stats["pct_low_risk"] = regional_stats["n_low_risk"] / regional_stats["n_mangrove_cells"] * 100
    
    # Rename country column for consistency
    regional_stats = regional_stats.rename(columns={country_col: "COUNTRY"})
    
    return regional_stats


def main():
    print("=" * 60)
    print("Loading ecosystem boundaries...")
    print("=" * 60)
    
    bounds_gdf = load_ecosystem_boundaries()
    
    print("\n" + "=" * 60)
    print("Loading and combining risk data...")
    print("=" * 60)
    
    # Load all scenario data and combine
    all_data = []
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        df = pd.read_csv(OUTPUT_DIR / f"risk_assessment_{scenario}.csv")
        df["scenario"] = scenario
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {len(combined)} rows")
    
    # Assign countries
    print("\nAssigning countries to mangrove cells...")
    mangrove_with_country, country_col = assign_countries_to_mangroves(combined, bounds_gdf)
    
    # Save intermediate
    mangrove_with_country.to_csv(OUTPUT_DIR / "mangrove_risk_with_country.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Aggregating by region for each scenario...")
    print("=" * 60)
    
    regional_results = {}
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        print(f"\n{scenario}:")
        regional = aggregate_by_region(mangrove_with_country, scenario, country_col)
        regional_results[scenario] = regional
        regional.to_csv(OUTPUT_DIR / f"regional_risk_{scenario}.csv", index=False)
        
        # Show top 10 high-risk countries
        top_high = regional.nlargest(10, "pct_high_risk")
        print(f"Top 10 countries by % high-risk mangroves:")
        print(top_high[["COUNTRY", "n_mangrove_cells", "pct_high_risk", "mean_composite_risk"]].to_string(index=False))
    
    # Create comparison table across scenarios
    print("\n" + "=" * 60)
    print("Creating cross-scenario comparison...")
    print("=" * 60)
    
    # Merge all scenarios
    comparison = regional_results["ssp245"][["COUNTRY", "n_mangrove_cells"]].copy()
    
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        regional = regional_results[scenario]
        comparison = comparison.merge(
            regional[["COUNTRY", "mean_composite_risk", "pct_high_risk"]],
            on="COUNTRY",
            suffixes=("", f"_{scenario}")
        )
    
    comparison.to_csv(OUTPUT_DIR / "regional_comparison_all_scenarios.csv", index=False)
    
    print(f"Saved regional comparison: {OUTPUT_DIR / 'regional_comparison_all_scenarios.csv'}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Regional Analysis Summary")
    print("=" * 60)
    
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        regional = regional_results[scenario]
        print(f"\n{scenario.upper()}:")
        print(f"  Countries with mangroves: {len(regional)}")
        print(f"  Mean composite risk (across countries): {regional['mean_composite_risk'].mean():.3f}")
        print(f"  Countries with >50% high-risk mangroves: {(regional['pct_high_risk'] > 50).sum()}")
    
    print("\n" + "=" * 60)
    print("Regional analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
