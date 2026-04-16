"""
Step 4: Ecosystem service assessment - join risk with country-level data.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os

os.makedirs('outputs', exist_ok=True)

# Load mangrove risk data
df = pd.read_csv('outputs/mangrove_composite_risk.csv')
print(f"Loaded {len(df)} mangrove points")

# Load country boundaries with ecosystem service data
countries = gpd.read_file('data/ecosystem/UCSC_CWON_countrybounds.gpkg')
print(f"Loaded {len(countries)} countries")

# Create mangrove GeoDataFrame
from shapely.geometry import Point
mangrove_gdf = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])],
    crs='EPSG:4326'
)

# Spatial join: assign each mangrove point to a country
print("Performing spatial join with country boundaries...")
mangrove_with_country = gpd.sjoin(mangrove_gdf, countries[['ISO3', 'Country_2020', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020', 'geometry']], how='left', predicate='within')
print(f"  {mangrove_with_country['ISO3'].notna().sum()} points matched to countries")
print(f"  {mangrove_with_country['ISO3'].isna().sum()} points unmatched")

# For unmatched points, try nearest country
unmatched = mangrove_with_country[mangrove_with_country['ISO3'].isna()].copy()
if len(unmatched) > 0:
    print(f"  Assigning {len(unmatched)} unmatched points to nearest country...")
    matched = mangrove_with_country[mangrove_with_country['ISO3'].notna()].copy()
    
    # Simple nearest assignment using centroids
    country_centroids = countries.copy()
    country_centroids['centroid'] = country_centroids.geometry.centroid
    
    for idx in unmatched.index:
        point = mangrove_gdf.loc[idx, 'geometry']
        min_dist = float('inf')
        best_iso = None
        best_country = None
        for _, crow in country_centroids.iterrows():
            d = point.distance(crow['centroid'])
            if d < min_dist:
                min_dist = d
                best_iso = crow['ISO3']
                best_country = crow['Country_2020']
        mangrove_with_country.loc[idx, 'ISO3'] = best_iso
        mangrove_with_country.loc[idx, 'Country_2020'] = best_country

# Save with country assignment
mangrove_with_country.to_csv('outputs/mangrove_with_country.csv', index=False)

# ============================================================
# Ecosystem service at-risk analysis by scenario and risk category
# ============================================================
scenarios = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
risk_cats = ['Low', 'Moderate', 'High', 'Very High']

# Each mangrove point represents approximately equal area (sampled from GMW)
# Total mangrove area from GMW ~150,000 km2
# With 100k sample points, each point ~ 1.5 km2
total_mangrove_area_km2 = 150000  # approximate
area_per_point = total_mangrove_area_km2 / len(df)

# Country-level ecosystem service data
# Mang_Ha_2020: mangrove area in hectares
# Risk_Pop_2020: population at risk (people)
# Risk_Stock_2020: capital stock at risk (USD)
# Ben_Pop_2020: population benefiting (people)
# Ben_Stock_2020: capital stock benefiting (USD)

es_results = {}

for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"Ecosystem service analysis for {scenario}...")
    
    cri_col = f'cri_{scenario}'
    cat_col = f'cri_cat_{scenario}'
    
    # By risk category
    cat_summary = {}
    for cat in risk_cats:
        mask = mangrove_with_country[cat_col] == cat
        n_points = mask.sum()
        area_km2 = n_points * area_per_point
        
        # Get countries for these points
        subset = mangrove_with_country[mask]
        
        # Aggregate ecosystem services by country
        country_es = subset.groupby('ISO3').agg(
            n_points=('ISO3', 'count'),
            country_name=('Country_2020', 'first'),
        ).reset_index()
        
        # Merge with country-level ES data
        country_es = country_es.merge(
            countries[['ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020']],
            on='ISO3', how='left'
        )
        
        cat_summary[cat] = {
            'n_points': int(n_points),
            'area_km2': float(area_km2),
            'pct_total': float(n_points / len(df) * 100),
            'n_countries': int(country_es['ISO3'].nunique()),
            'total_risk_pop': float(country_es['Risk_Pop_2020'].sum()),
            'total_risk_stock_usd': float(country_es['Risk_Stock_2020'].sum()),
            'total_ben_pop': float(country_es['Ben_Pop_2020'].sum()),
            'total_ben_stock_usd': float(country_es['Ben_Stock_2020'].sum()),
        }
    
    # High + Very High combined
    hv_mask = mangrove_with_country[cat_col].isin(['High', 'Very High'])
    hv_points = hv_mask.sum()
    
    es_results[scenario] = {
        'by_category': cat_summary,
        'high_very_high': {
            'n_points': int(hv_points),
            'area_km2': float(hv_points * area_per_point),
            'pct_total': float(hv_points / len(df) * 100),
        }
    }
    
    print(f"  High+Very High: {hv_points} points ({hv_points/len(df)*100:.1f}%), "
          f"~{hv_points * area_per_point:.0f} km2")

with open('outputs/ecosystem_service_risk.json', 'w') as f:
    json.dump(es_results, f, indent=2)
print("\nSaved ecosystem service risk to outputs/ecosystem_service_risk.json")

# ============================================================
# Country-level risk summary
# ============================================================
country_risk = {}
for scenario in scenarios:
    cri_col = f'cri_{scenario}'
    cat_col = f'cri_cat_{scenario}'
    
    country_agg = mangrove_with_country.groupby('ISO3').agg(
        country_name=('Country_2020', 'first'),
        n_points=('ISO3', 'count'),
        mean_cri=(cri_col, 'mean'),
        n_high_vh=(cat_col, lambda x: (x.isin(['High', 'Very High'])).sum()),
    ).reset_index()
    
    country_agg['pct_high_vh'] = country_agg['n_high_vh'] / country_agg['n_points'] * 100
    
    # Merge with ES data
    country_agg = country_agg.merge(
        countries[['ISO3', 'Mang_Ha_2020', 'Risk_Pop_2020', 'Risk_Stock_2020', 'Ben_Pop_2020', 'Ben_Stock_2020']],
        on='ISO3', how='left'
    )
    
    # Top 20 most at-risk countries
    top20 = country_agg.nlargest(20, 'pct_high_vh')[
        ['ISO3', 'country_name', 'n_points', 'mean_cri', 'pct_high_vh', 'Mang_Ha_2020', 'Risk_Pop_2020']
    ]
    
    country_risk[scenario] = top20.to_dict('records')
    
    print(f"\nTop 10 most at-risk countries ({scenario}):")
    print(top10 := country_agg.nlargest(10, 'pct_high_vh')[
        ['ISO3', 'country_name', 'pct_high_vh', 'mean_cri', 'Mang_Ha_2020']
    ].to_string(index=False))

with open('outputs/country_risk_summary.json', 'w') as f:
    json.dump(country_risk, f, indent=2)
print("\nSaved country risk summary to outputs/country_risk_summary.json")
