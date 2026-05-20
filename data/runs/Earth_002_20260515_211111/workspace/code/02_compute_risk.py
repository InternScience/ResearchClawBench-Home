#!/usr/bin/env python3
"""
Compute composite risk index combining SLR and TC risks for global mangroves.
Based on methodology from Saintilan et al. (2023), Mo et al. (2023), and Kropf et al. (2023).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

print("=== Loading extracted data ===")
gdf = gpd.read_parquet('outputs/mangrove_risk_data.parquet')
print(f"Loaded {len(gdf)} mangrove points")

# === SLR Risk Classification ===
# Based on Saintilan et al. (2023): 
# - RSLR < 4 mm/yr: low probability of retreat
# - RSLR 4-7 mm/yr: likely retreat (P > 0.66)
# - RSLR > 7 mm/yr: highly likely retreat (P > 0.90)

print("\n=== Computing SLR Risk ===")

for ssp_label, ssp_col in [
    ('SSP2-4.5', 'ssp2_4_5_slr_2080_2100'),
    ('SSP3-7.0', 'ssp3_7_0_slr_2080_2100'),
    ('SSP5-8.5', 'ssp5_8_5_slr_2080_2100'),
]:
    slr = gdf[ssp_col].values
    
    # Numerical score: 0 (low), 1 (moderate), 2 (high)
    conditions = [
        slr < 4.0,
        (slr >= 4.0) & (slr < 7.0),
        slr >= 7.0
    ]
    choices = [0, 1, 2]
    
    risk_col = ssp_col.replace('_slr_2080_2100', '_slr_risk_score')
    cat_col = ssp_col.replace('_slr_2080_2100', '_slr_risk_cat')
    
    gdf[risk_col] = np.select(conditions, choices, default=-1)
    
    cat_labels = ['Low', 'Moderate', 'High']
    gdf[cat_col] = pd.Categorical.from_codes(gdf[risk_col], categories=cat_labels)
    
    print(f"\n{ssp_label} SLR Risk Distribution:")
    print(gdf[cat_col].value_counts())

# === TC Risk Classification ===
# Based on Mo et al. (2023) and Kropf et al. (2023):
# - TC Risk depends on baseline frequency of major/intense cyclones AND projected shifts
# - Regions with high baseline TC frequency are considered "adapted" but face risk from regime shifts
# - Regions with low baseline but projected increase in TC activity are "vulnerable"

print("\n=== Computing TC Risk ===")

# TC baseline risk based on intense cyclone frequency (Cat 4-5)
tc_intense_freq = gdf['tc_intense_freq'].values
tc_major_freq = gdf['tc_major_freq'].values

# TC Risk Score based on major cyclone frequency (Cat 3-5):
# 0: No major TCs (freq < 0.005/yr, i.e. < 1 per 200 years)
# 1: Low frequency (0.005-0.02/yr, i.e. 1 per 50-200 years)
# 2: Moderate frequency (0.02-0.1/yr, i.e. 1 per 10-50 years)
# 3: High frequency (> 0.1/yr, i.e. > 1 per 10 years)

tc_conditions = [
    tc_major_freq < 0.005,
    (tc_major_freq >= 0.005) & (tc_major_freq < 0.02),
    (tc_major_freq >= 0.02) & (tc_major_freq < 0.1),
    tc_major_freq >= 0.1
]
tc_choices = [0, 1, 2, 3]
gdf['tc_baseline_risk_score'] = np.select(tc_conditions, tc_choices, default=0)

tc_cat_labels = ['None', 'Low', 'Moderate', 'High']
gdf['tc_baseline_risk_cat'] = pd.Categorical.from_codes(gdf['tc_baseline_risk_score'], categories=tc_cat_labels)

print("\nTC Baseline Risk Distribution:")
print(gdf['tc_baseline_risk_cat'].value_counts())

# TC Regime Shift Risk
# Based on Mo et al. (2023): with 2°C warming, risk increases in North Atlantic, North Indian Ocean,
# decreases in NW and SW Pacific
# Based on Kropf et al. (2023): regions facing significant frequency increases are at risk

# We apply a simplified regional projection based on Mo et al. (2023) Fig. 3:
# - North America/Caribbean: +10% risk
# - East Asia (China, Japan, Philippines): mixed (NW Pacific decreases, but intensity increases)
# - Oceania: -10% risk
# - Indian Ocean (East Africa, South Asia): increase
# - South America, Africa (Atlantic): minimal change

def get_tc_shift_factor(lon, lat):
    """Return projected TC frequency shift factor based on region.
    Values derived from Mo et al. (2023) and Kropf et al. (2023).
    Positive = increase in risk, Negative = decrease.
    """
    # North America / Caribbean / Gulf of Mexico
    if (-100 <= lon <= -60) and (10 <= lat <= 35):
        return 1.10  # +10% risk
    # Central America / northern South America
    if (-90 <= lon <= -50) and (-5 <= lat <= 15):
        return 1.05
    # East Asia - Northwest Pacific (mixed region)
    if (100 <= lon <= 150) and (15 <= lat <= 45):
        return 0.95  # slight decrease in NW Pacific
    # Southeast Asia / Philippines
    if (110 <= lon <= 130) and (-10 <= lat <= 15):
        return 1.02  # slight increase
    # South Asia / Bay of Bengal / North Indian Ocean
    if (65 <= lon <= 100) and (5 <= lat <= 30):
        return 1.12  # increase
    # East Africa / Madagascar / SW Indian Ocean
    if (30 <= lon <= 60) and (-30 <= lat <= 0):
        return 1.08  # increase
    # Oceania / SW Pacific
    if (140 <= lon <= 180) and (-35 <= lat <= -5):
        return 0.90  # -10% risk
    # Australia
    if (110 <= lon <= 155) and (-40 <= lat <= -10):
        return 0.95
    # West Africa / Atlantic
    if (-20 <= lon <= 10) and (-10 <= lat <= 15):
        return 1.02
    # South America Atlantic coast
    if (-50 <= lon <= -30) and (-25 <= lat <= 0):
        return 1.01
    # Default: minimal change
    return 1.00

shift_factors = np.array([get_tc_shift_factor(lon, lat) for lon, lat in zip(gdf.geometry.x.values, gdf.geometry.y.values)])
gdf['tc_shift_factor'] = shift_factors

# Projected TC risk = baseline * shift factor
gdf['tc_projected_risk'] = gdf['tc_major_freq'] * gdf['tc_shift_factor']

# TC projected risk score
tc_proj = gdf['tc_projected_risk'].values
proj_conditions = [
    tc_proj < 0.005,
    (tc_proj >= 0.005) & (tc_proj < 0.02),
    (tc_proj >= 0.02) & (tc_proj < 0.1),
    tc_proj >= 0.1
]
gdf['tc_projected_risk_score'] = np.select(proj_conditions, tc_choices, default=0)
gdf['tc_projected_risk_cat'] = pd.Categorical.from_codes(gdf['tc_projected_risk_score'], categories=tc_cat_labels)

print("\nTC Projected Risk Distribution:")
print(gdf['tc_projected_risk_cat'].value_counts())

# === Composite Risk Index ===
# Combine SLR risk (score 0-2) and TC risk (score 0-3) into a composite 0-5 scale
# CRI = SLR_score + TC_projected_score (range 0-5)

print("\n=== Computing Composite Risk Index ===")

for ssp_label, ssp_col in [
    ('SSP2-4.5', 'ssp2_4_5_slr_2080_2100'),
    ('SSP3-7.0', 'ssp3_7_0_slr_2080_2100'),
    ('SSP5-8.5', 'ssp5_8_5_slr_2080_2100'),
]:
    slr_score_col = ssp_col.replace('_slr_2080_2100', '_slr_risk_score')
    cri_col = ssp_col.replace('_slr_2080_2100', '_cri')
    cri_cat_col = ssp_col.replace('_slr_2080_2100', '_cri_cat')
    
    # Composite score = SLR (0-2) + TC projected (0-3)
    gdf[cri_col] = gdf[slr_score_col] + gdf['tc_projected_risk_score']
    
    # Map to categories
    cri_conditions = [
        gdf[cri_col] <= 0,
        gdf[cri_col] == 1,
        gdf[cri_col] == 2,
        gdf[cri_col] == 3,
        gdf[cri_col] >= 4
    ]
    cri_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    gdf[cri_cat_col] = np.select(cri_conditions, 
                                  [0, 1, 2, 3, 4], default=4)
    gdf[cri_cat_col] = pd.Categorical.from_codes(gdf[cri_cat_col], categories=cri_labels)
    
    print(f"\n{ssp_label} Composite Risk Distribution:")
    print(gdf[cri_cat_col].value_counts())

# === Country-level aggregation ===
print("\n=== Country-level Aggregation ===")

# Load country boundaries
country_gdf = gpd.read_file('data/ecosystem/UCSC_CWON_countrybounds.gpkg')
print(f"Loaded {len(country_gdf)} countries")

# Spatial join: assign each mangrove point to a country
# Use the 'Country' column from country bounds
gdf_mangrove = gdf.copy()
gdf_mangrove.crs = 'EPSG:4326'
country_gdf.crs = 'EPSG:4326'

# Spatial join
gdf_with_country = gpd.sjoin(gdf_mangrove, country_gdf[['Country', 'geometry']], how='left', predicate='within')
print(f"Mangroves assigned to countries: {gdf_with_country['Country'].notna().sum()} / {len(gdf_with_country)}")

# Count mangroves per country
country_counts = gdf_with_country.groupby('Country').size().sort_values(ascending=False)
print(f"\nTop 20 countries by mangrove count:")
print(country_counts.head(20))

# For SSP5-8.5, compute country-level risk summary
for ssp_label, ssp_col in [
    ('SSP2-4.5', 'ssp2_4_5'),
    ('SSP3-7.0', 'ssp3_7_0'),
    ('SSP5-8.5', 'ssp5_8_5'),
]:
    cri_col = f'{ssp_col}_cri'
    cri_cat_col = f'{ssp_col}_cri_cat'
    slr_col = f'{ssp_col}_slr_2080_2100'
    
    country_risk = gdf_with_country.groupby('Country').agg(
        mean_cri=(cri_col, 'mean'),
        mean_slr=(slr_col, 'mean'),
        high_risk_pct=(cri_cat_col, lambda x: ((x == 'High') | (x == 'Very High')).mean() * 100),
        very_high_pct=(cri_cat_col, lambda x: (x == 'Very High').mean() * 100),
        n_points=('uid', 'count')
    ).reset_index()
    
    country_risk = country_risk.sort_values('mean_cri', ascending=False)
    
    outpath = f'outputs/country_risk_{ssp_label.replace(".", "_").replace("-", "_")}.csv'
    country_risk.to_csv(outpath, index=False)
    print(f"\nSaved country risk to {outpath}")
    print(country_risk.head(15))

# === Save final data ===
print("\n=== Saving final risk data ===")
gdf_with_country.to_parquet('outputs/mangrove_risk_final.parquet')
print("Saved to outputs/mangrove_risk_final.parquet")

# Also save a CSV with key columns
key_cols = ['uid', 'ref_cls', 'Country']
for ssp in ['ssp2_4_5', 'ssp3_7_0', 'ssp5_8_5']:
    key_cols.extend([f'{ssp}_slr_2080_2100', f'{ssp}_slr_risk_cat', f'{ssp}_cri', f'{ssp}_cri_cat'])
key_cols.extend(['tc_total_freq', 'tc_major_freq', 'tc_intense_freq', 
                  'tc_baseline_risk_cat', 'tc_projected_risk_cat', 'tc_shift_factor'])

gdf_with_country[key_cols].to_csv('outputs/mangrove_risk_summary.csv', index=False)

print("\n=== Risk computation complete! ===")
