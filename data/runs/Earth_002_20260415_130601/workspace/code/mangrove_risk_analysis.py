"""
Mangrove Composite Risk Index Analysis
=====================================
Develops a composite risk index combining tropical cyclone regime shifts 
and sea level rise for global mangrove ecosystems.

Author: Research Analysis
Date: 2024
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("MANGROVE COMPOSITE RISK INDEX ANALYSIS")
print("=" * 60)

# =============================================================================
# PART 1: LOAD AND PROCESS SEA LEVEL RISE DATA
# =============================================================================
print("\n[1] Loading Sea Level Rise Data...")

slr_scenarios = {
    'SSP2-4.5': 'data/slr/total_ssp245_medium_confidence_rates.nc',
    'SSP3-7.0': 'data/slr/total_ssp370_medium_confidence_rates.nc',
    'SSP5-8.5': 'data/slr/total_ssp585_medium_confidence_rates.nc'
}

slr_data = {}
for scenario, filepath in slr_scenarios.items():
    print(f"  Loading {scenario}...")
    ds = xr.open_dataset(filepath)
    
    # Extract median (50th percentile) rates for end of century
    # Find the index closest to 0.5 quantile
    quantiles = ds['quantiles'].values
    median_idx = np.argmin(np.abs(quantiles - 0.5))
    
    # Get year 2100 (last time step typically)
    years = ds['years'].values
    year_2100_idx = -1  # Last year
    
    # Extract median SLR rates for 2100
    median_rates = ds['sea_level_change_rate'].values[median_idx, year_2100_idx, :]
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    slr_data[scenario] = {
        'lat': lats,
        'lon': lons,
        'rate': median_rates,
        'rates_all': ds['sea_level_change_rate'].values[:, year_2100_idx, :]
    }
    ds.close()

print(f"  SLR data loaded for {len(slr_data)} scenarios")
print(f"  Grid points: {len(slr_data['SSP2-4.5']['lat'])}")

# =============================================================================
# PART 2: LOAD AND PROCESS TROPICAL CYCLONE DATA
# =============================================================================
print("\n[2] Loading Tropical Cyclone Track Data...")

tc_ds = xr.open_dataset('data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc')
tc_lats = tc_ds['lat'].values
tc_lons = tc_ds['lon'].values
tc_winds = tc_ds['wind'].values
tc_ds.close()

# Filter valid data (remove NaNs)
valid_mask = ~(np.isnan(tc_lats) | np.isnan(tc_lons) | np.isnan(tc_winds))
tc_lats = tc_lats[valid_mask]
tc_lons = tc_lons[valid_mask]
tc_winds = tc_winds[valid_mask]

print(f"  Historical TC records: {len(tc_lats):,}")
print(f"  Wind speed range: {np.min(tc_winds):.1f} - {np.max(tc_winds):.1f} m/s")
print(f"  Period: 1850-2014 (~165 years)")

# Categorize cyclones by intensity (Saffir-Simpson scale approximation)
# Category 1: 33-42 m/s, Cat 2: 43-49, Cat 3: 50-58, Cat 4: 59-70, Cat 5: >70
def categorize_cyclone(wind_ms):
    """Convert wind speed (m/s) to Saffir-Simpson category"""
    wind_kmh = wind_ms * 3.6
    if wind_kmh < 119:
        return 0  # Tropical storm
    elif wind_kmh < 154:
        return 1
    elif wind_kmh < 177:
        return 2
    elif wind_kmh < 208:
        return 3
    elif wind_kmh < 252:
        return 4
    else:
        return 5

tc_categories = np.array([categorize_cyclone(w) for w in tc_winds])
for cat in range(6):
    count = np.sum(tc_categories == cat)
    print(f"    Category {cat}: {count:,} records")

# =============================================================================
# PART 3: LOAD MANGROVE DATA
# =============================================================================
print("\n[3] Loading Mangrove Distribution Data...")

mangroves = gpd.read_file('data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg')
print(f"  Mangrove sample points: {len(mangroves):,}")
print(f"  CRS: {mangroves.crs}")

# Extract coordinates
mg_lons = mangroves.geometry.x.values
mg_lats = mangroves.geometry.y.values

# Calculate approximate area per point (assuming sample represents 10%)
mangrove_area_km2 = len(mangroves) * 0.25  # Approximate area estimate
print(f"  Approximate global mangrove area: {mangrove_area_km2:,.0f} km²")

# =============================================================================
# PART 4: CALCULATE SEA LEVEL RISE RISK
# =============================================================================
print("\n[4] Calculating Sea Level Rise Risk...")

# Based on Saintilan et al. (2023) thresholds:
# - Deficit likely at 4 mm/yr
# - Highly likely/deficit very likely at 7 mm/yr
def calculate_slr_risk(slr_rate):
    """
    Calculate SLR risk score based on Saintilan et al. (2023)
    Returns risk score from 0-1
    """
    risk = np.zeros_like(slr_rate, dtype=float)
    
    # Low risk: < 4 mm/yr
    mask_low = slr_rate < 4
    risk[mask_low] = slr_rate[mask_low] / 4 * 0.3  # Scale to 0-0.3
    
    # Moderate risk: 4-7 mm/yr
    mask_med = (slr_rate >= 4) & (slr_rate < 7)
    risk[mask_med] = 0.3 + (slr_rate[mask_med] - 4) / 3 * 0.4  # Scale to 0.3-0.7
    
    # High risk: >= 7 mm/yr
    mask_high = slr_rate >= 7
    risk[mask_high] = 0.7 + np.minimum((slr_rate[mask_high] - 7) / 5, 1) * 0.3  # Scale to 0.7-1.0
    
    return np.clip(risk, 0, 1)

# Build KD-tree for SLR data to interpolate to mangrove locations
from scipy.spatial import cKDTree

def interpolate_to_points(lats_src, lons_src, values, lats_tgt, lons_tgt):
    """Interpolate values from source grid to target points"""
    # Convert to 3D Cartesian coordinates for accurate distance calculation
    def latlon_to_cartesian(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.column_stack([x, y, z])
    
    src_coords = latlon_to_cartesian(lats_src, lons_src)
    tgt_coords = latlon_to_cartesian(lats_tgt, lons_tgt)
    
    tree = cKDTree(src_coords)
    distances, indices = tree.query(tgt_coords, k=3)
    
    # Inverse distance weighting
    weights = 1 / (distances + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    
    interpolated = np.sum(values[indices] * weights, axis=1)
    return interpolated

# Calculate SLR risk for each scenario at mangrove locations
slr_risk_results = {}
for scenario, data in slr_data.items():
    print(f"  Processing {scenario}...")
    
    # Interpolate SLR rates to mangrove locations
    slr_at_mangroves = interpolate_to_points(
        data['lat'], data['lon'], data['rate'],
        mg_lats, mg_lons
    )
    
    # Calculate risk scores
    risk_scores = calculate_slr_risk(slr_at_mangroves)
    
    slr_risk_results[scenario] = {
        'rate': slr_at_mangroves,
        'risk': risk_scores
    }
    
    print(f"    Mean SLR rate: {np.mean(slr_at_mangroves):.2f} mm/yr")
    print(f"    High risk (>0.7): {np.sum(risk_scores > 0.7) / len(risk_scores) * 100:.1f}%")

# =============================================================================
# PART 5: CALCULATE TROPICAL CYCLONE RISK
# =============================================================================
print("\n[5] Calculating Tropical Cyclone Risk...")

# Calculate TC frequency/intensity within radius of each mangrove point
# Use a 1-degree radius (~111 km at equator)

def calculate_tc_risk(mg_lats, mg_lons, tc_lats, tc_lons, tc_winds, tc_categories):
    """
    Calculate TC risk at mangrove locations based on:
    - Frequency of TC passage
    - Maximum intensity experienced
    - Weighted by distance to track
    """
    n_points = len(mg_lats)
    tc_risk = np.zeros(n_points)
    tc_frequency = np.zeros(n_points)
    tc_max_intensity = np.zeros(n_points)
    
    # For efficiency, process in batches
    batch_size = 1000
    n_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_points)
        
        for i in range(start_idx, end_idx):
            lat, lon = mg_lats[i], mg_lons[i]
            
            # Find TC points within 2 degrees (~220 km)
            dist_lat = np.abs(tc_lats - lat)
            dist_lon = np.abs(tc_lons - lon)
            
            # Handle longitude wrap-around
            dist_lon = np.minimum(dist_lon, 360 - dist_lon)
            
            within_radius = (dist_lat < 2) & (dist_lon < 2)
            
            if np.any(within_radius):
                # Calculate great circle distance
                tc_lats_near = tc_lats[within_radius]
                tc_lons_near = tc_lons[within_radius]
                tc_winds_near = tc_winds[within_radius]
                tc_cats_near = tc_categories[within_radius]
                
                # Simple Euclidean approximation for small distances
                dist_deg = np.sqrt((tc_lats_near - lat)**2 + (tc_lons_near - lon)**2)
                dist_km = dist_deg * 111  # Approximate km
                
                # Weight by inverse distance (closer = higher impact)
                weights = np.exp(-dist_km / 100)  # Decay with 100km characteristic distance
                
                # Calculate weighted risk
                # Higher category cyclones cause exponentially more damage
                damage_weights = np.power(2, tc_cats_near)  # Cat 5 = 32x damage of Cat 0
                
                tc_risk[i] = np.sum(weights * damage_weights) / 165  # Normalize by years
                tc_frequency[i] = np.sum(within_radius) / 165  # Passages per year
                tc_max_intensity[i] = np.max(tc_cats_near)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Progress: {(batch_idx + 1) / n_batches * 100:.0f}%")
    
    return tc_risk, tc_frequency, tc_max_intensity

# Calculate TC risk
tc_risk, tc_freq, tc_max = calculate_tc_risk(
    mg_lats, mg_lons, tc_lats, tc_lons, tc_winds, tc_categories
)

# Normalize TC risk to 0-1 scale
tc_risk_norm = (tc_risk - np.min(tc_risk)) / (np.max(tc_risk) - np.min(tc_risk) + 1e-10)

print(f"  TC risk calculation complete")
print(f"  Mean TC frequency: {np.mean(tc_freq):.3f} passages/year")
print(f"  Max observed intensity: Category {int(np.max(tc_max))}")
print(f"  High TC risk (>0.7): {np.sum(tc_risk_norm > 0.7) / len(tc_risk_norm) * 100:.1f}%")

# =============================================================================
# PART 6: CALCULATE COMPOSITE RISK INDEX
# =============================================================================
print("\n[6] Calculating Composite Risk Index...")

# Weights for combining SLR and TC risk
# Based on literature: both are significant but SLR may be more pervasive
SLR_WEIGHT = 0.6
TC_WEIGHT = 0.4

composite_risk = {}
for scenario in slr_risk_results.keys():
    # Combine risks (can use different combination methods)
    # Method 1: Weighted average
    combined_avg = SLR_WEIGHT * slr_risk_results[scenario]['risk'] + TC_WEIGHT * tc_risk_norm
    
    # Method 2: Maximum of components (conservative)
    combined_max = np.maximum(slr_risk_results[scenario]['risk'], tc_risk_norm)
    
    # Method 3: Geometric mean
    combined_geom = np.sqrt(slr_risk_results[scenario]['risk'] * tc_risk_norm)
    
    composite_risk[scenario] = {
        'weighted_avg': combined_avg,
        'max_risk': combined_max,
        'geometric_mean': combined_geom,
        'slr_risk': slr_risk_results[scenario]['risk'],
        'tc_risk': tc_risk_norm,
        'slr_rate': slr_risk_results[scenario]['rate']
    }

# Create summary dataframe
results_df = pd.DataFrame({
    'lon': mg_lons,
    'lat': mg_lats,
    'tc_risk': tc_risk_norm,
    'tc_frequency': tc_freq,
    'tc_max_category': tc_max
})

for scenario in slr_risk_results.keys():
    results_df[f'slr_rate_{scenario}'] = composite_risk[scenario]['slr_rate']
    results_df[f'slr_risk_{scenario}'] = composite_risk[scenario]['slr_risk']
    results_df[f'composite_risk_{scenario}'] = composite_risk[scenario]['weighted_avg']

print(f"  Results DataFrame shape: {results_df.shape}")
print(f"\n  Risk Summary (Weighted Average Method):")
for scenario in slr_risk_results.keys():
    risk_col = f'composite_risk_{scenario}'
    high_risk = np.sum(results_df[risk_col] > 0.7) / len(results_df) * 100
    print(f"    {scenario}: Mean={results_df[risk_col].mean():.3f}, High Risk={high_risk:.1f}%")

# Save results
results_df.to_csv('outputs/mangrove_risk_results.csv', index=False)
print(f"\n  Results saved to outputs/mangrove_risk_results.csv")

# =============================================================================
# PART 7: GENERATE VISUALIZATIONS
# =============================================================================
print("\n[7] Generating Visualizations...")

# Create figure directory
import os
os.makedirs('report/images', exist_ok=True)

# Figure 1: Global SLR Risk Maps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
scenarios = ['SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    risk_col = f'slr_risk_{scenario}'
    
    scatter = ax.scatter(results_df['lon'], results_df['lat'], 
                        c=results_df[risk_col], cmap='YlOrRd',
                        s=1, alpha=0.6, vmin=0, vmax=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'SLR Risk - {scenario}')
    plt.colorbar(scatter, ax=ax, label='Risk Score')

plt.tight_layout()
plt.savefig('report/images/slr_risk_maps.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/slr_risk_maps.png")

# Figure 2: Global TC Risk Map
fig, ax = plt.subplots(figsize=(14, 7))
scatter = ax.scatter(results_df['lon'], results_df['lat'], 
                    c=results_df['tc_risk'], cmap='YlOrRd',
                    s=1, alpha=0.6, vmin=0, vmax=1)
ax.set_xlim(-180, 180)
ax.set_ylim(-40, 35)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Tropical Cyclone Risk to Global Mangroves', fontsize=14)
plt.colorbar(scatter, ax=ax, label='Risk Score')
plt.tight_layout()
plt.savefig('report/images/tc_risk_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/tc_risk_map.png")

# Figure 3: Composite Risk Maps (all scenarios)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    risk_col = f'composite_risk_{scenario}'
    
    scatter = ax.scatter(results_df['lon'], results_df['lat'], 
                        c=results_df[risk_col], cmap='YlOrRd',
                        s=1, alpha=0.6, vmin=0, vmax=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Composite Risk - {scenario}')
    plt.colorbar(scatter, ax=ax, label='Risk Score')

plt.tight_layout()
plt.savefig('report/images/composite_risk_maps.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/composite_risk_maps.png")

# Figure 4: Risk Distribution Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SLR rates by scenario
ax = axes[0, 0]
for scenario in scenarios:
    ax.hist(results_df[f'slr_rate_{scenario}'], bins=50, alpha=0.5, label=scenario)
ax.axvline(4, color='orange', linestyle='--', label='Threshold (4 mm/yr)')
ax.axvline(7, color='red', linestyle='--', label='Critical (7 mm/yr)')
ax.set_xlabel('SLR Rate (mm/yr)')
ax.set_ylabel('Frequency')
ax.set_title('Sea Level Rise Rate Distribution')
ax.legend()

# SLR risk
ax = axes[0, 1]
for scenario in scenarios:
    ax.hist(results_df[f'slr_risk_{scenario}'], bins=50, alpha=0.5, label=scenario)
ax.set_xlabel('SLR Risk Score')
ax.set_ylabel('Frequency')
ax.set_title('SLR Risk Score Distribution')
ax.legend()

# TC risk
ax = axes[1, 0]
ax.hist(results_df['tc_risk'], bins=50, color='purple', alpha=0.6)
ax.set_xlabel('TC Risk Score')
ax.set_ylabel('Frequency')
ax.set_title('Tropical Cyclone Risk Distribution')

# Composite risk comparison
ax = axes[1, 1]
for scenario in scenarios:
    ax.hist(results_df[f'composite_risk_{scenario}'], bins=50, alpha=0.5, label=scenario)
ax.set_xlabel('Composite Risk Score')
ax.set_ylabel('Frequency')
ax.set_title('Composite Risk Score Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/risk_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/risk_distributions.png")

# Figure 5: Regional Risk Analysis
print("  Creating regional summaries...")

# Define regions based on longitude bands
def assign_region(lon, lat):
    if lat > 23.5:
        return 'Northern Tropics'
    elif lat < -23.5:
        return 'Southern Tropics'
    else:
        if -100 < lon < -30:
            return 'Americas'
        elif -30 <= lon < 60:
            return 'Africa/ME'
        elif 60 <= lon < 150:
            return 'Asia-Pacific'
        else:
            return 'Oceania/Americas'

results_df['region'] = [assign_region(lon, lat) for lon, lat in zip(results_df['lon'], results_df['lat'])]

# Regional summary
regional_summary = results_df.groupby('region').agg({
    f'composite_risk_SSP5-8.5': ['mean', 'std', 'count'],
    f'slr_rate_SSP5-8.5': 'mean',
    'tc_risk': 'mean',
    'tc_frequency': 'mean'
}).round(3)

regional_summary.columns = ['_'.join(col).strip() for col in regional_summary.columns]
regional_summary.to_csv('outputs/regional_summary.csv')
print("  Saved: outputs/regional_summary.csv")

# Figure 6: Regional Risk Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
regions = results_df['region'].unique()
x = np.arange(len(regions))
width = 0.25

for idx, scenario in enumerate(scenarios):
    means = [results_df[results_df['region'] == r][f'composite_risk_{scenario}'].mean() for r in regions]
    ax.bar(x + idx * width, means, width, label=scenario)

ax.set_xlabel('Region')
ax.set_ylabel('Mean Composite Risk Score')
ax.set_title('Regional Mangrove Risk Assessment by Scenario')
ax.set_xticks(x + width)
ax.set_xticklabels(regions, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('report/images/regional_risk_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/regional_risk_comparison.png")

# Figure 7: TC vs SLR Risk Scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    ax.scatter(results_df['tc_risk'], results_df[f'slr_risk_{scenario}'], 
              c=results_df[f'composite_risk_{scenario}'], cmap='YlOrRd',
              s=1, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('TC Risk')
    ax.set_ylabel('SLR Risk')
    ax.set_title(f'{scenario}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/tc_vs_slr_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/tc_vs_slr_risk.png")

# Figure 8: Risk Category Maps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    risk_col = f'composite_risk_{scenario}'
    
    # Categorize risk
    risk_cats = pd.cut(results_df[risk_col], 
                      bins=[0, 0.3, 0.5, 0.7, 1.0],
                      labels=['Low', 'Moderate', 'High', 'Very High'])
    
    colors = {'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Very High': 'red'}
    for cat in ['Low', 'Moderate', 'High', 'Very High']:
        mask = risk_cats == cat
        ax.scatter(results_df.loc[mask, 'lon'], results_df.loc[mask, 'lat'],
                  c=colors[cat], s=1, alpha=0.6, label=cat)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Risk Categories - {scenario}')
    ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('report/images/risk_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/risk_categories.png")

# =============================================================================
# PART 8: ECOSYSTEM SERVICE RISK ASSESSMENT
# =============================================================================
print("\n[8] Ecosystem Service Risk Assessment...")

# Based on Dabalà et al. (2023) - high value services include:
# 1. Coastal protection (property value at risk)
# 2. Carbon storage
# 3. Fisheries support

# Estimate ecosystem service value at risk
# Using risk-weighted area approach

# Calculate area at risk by risk category
def calculate_area_at_risk(risk_scores):
    """Calculate proportion of mangroves in each risk category"""
    low = np.sum(risk_scores < 0.3) / len(risk_scores)
    moderate = np.sum((risk_scores >= 0.3) & (risk_scores < 0.5)) / len(risk_scores)
    high = np.sum((risk_scores >= 0.5) & (risk_scores < 0.7)) / len(risk_scores)
    very_high = np.sum(risk_scores >= 0.7) / len(risk_scores)
    return {'Low': low, 'Moderate': moderate, 'High': high, 'Very High': very_high}

# Global estimates (based on literature)
GLOBAL_MANGROVE_AREA_KM2 = 147000  # Total global mangrove area
CARBON_STORAGE_TG = 2800  # Total carbon storage in Tg
PEOPLE_PROTECTED_MILLIONS = 15  # Millions of people protected
FISHER_DAYS_MILLIONS = 100  # Million fisher days per year

ecosystem_services_risk = {}
for scenario in scenarios:
    risk_col = f'composite_risk_{scenario}'
    risk_dist = calculate_area_at_risk(results_df[risk_col].values)
    
    # Weighted risk calculation
    weighted_risk = (
        risk_dist['Low'] * 0.1 +
        risk_dist['Moderate'] * 0.3 +
        risk_dist['High'] * 0.6 +
        risk_dist['Very High'] * 1.0
    )
    
    ecosystem_services_risk[scenario] = {
        'area_at_risk_km2': GLOBAL_MANGROVE_AREA_KM2 * weighted_risk,
        'carbon_at_risk_tg': CARBON_STORAGE_TG * weighted_risk,
        'people_at_risk_millions': PEOPLE_PROTECTED_MILLIONS * weighted_risk,
        'fisheries_at_risk_millions': FISHER_DAYS_MILLIONS * weighted_risk,
        'risk_distribution': risk_dist,
        'weighted_risk': weighted_risk
    }
    
    print(f"\n  {scenario}:")
    print(f"    Weighted risk factor: {weighted_risk:.3f}")
    print(f"    Area at high/very high risk: {(risk_dist['High'] + risk_dist['Very High'])*100:.1f}%")
    print(f"    Estimated carbon at risk: {ecosystem_services_risk[scenario]['carbon_at_risk_tg']:.0f} Tg C")

# Save ecosystem services risk
es_df = pd.DataFrame({
    scenario: {
        'Weighted Risk Factor': v['weighted_risk'],
        'Area at Risk (km²)': v['area_at_risk_km2'],
        'Carbon at Risk (Tg C)': v['carbon_at_risk_tg'],
        'People at Risk (millions)': v['people_at_risk_millions'],
        'Fisheries at Risk (million days/yr)': v['fisheries_at_risk_millions'],
        'High Risk %': (v['risk_distribution']['High'] + v['risk_distribution']['Very High']) * 100
    }
    for scenario, v in ecosystem_services_risk.items()
}).T

es_df.to_csv('outputs/ecosystem_services_risk.csv')
print("\n  Saved: outputs/ecosystem_services_risk.csv")

# Figure 9: Ecosystem Services at Risk
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = [
    ('Area at Risk (km²)', 'Area at Risk (1000 km²)', 1000),
    ('Carbon at Risk (Tg C)', 'Carbon at Risk (Tg C)', 1),
    ('People at Risk (millions)', 'People at Risk (millions)', 1),
    ('Fisheries at Risk (million days/yr)', 'Fisheries at Risk (million days/yr)', 1)
]

for idx, (col, label, scale) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = [es_df.loc[s, col] / scale for s in scenarios]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(scenarios, values, color=colors, alpha=0.8)
    ax.set_ylabel(label)
    ax.set_title(f'Ecosystem Service at Risk by Scenario')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
               f'{val:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('report/images/ecosystem_services_at_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: report/images/ecosystem_services_at_risk.png")

# =============================================================================
# PART 9: SUMMARY STATISTICS
# =============================================================================
print("\n[9] Generating Summary Statistics...")

summary_stats = pd.DataFrame({
    'Metric': [
        'Total Mangrove Points',
        'Mean TC Risk',
        'Mean TC Frequency (passages/yr)',
        'SSP2-4.5 Mean SLR Rate (mm/yr)',
        'SSP3-7.0 Mean SLR Rate (mm/yr)',
        'SSP5-8.5 Mean SLR Rate (mm/yr)',
        'SSP2-4.5 Mean Composite Risk',
        'SSP3-7.0 Mean Composite Risk',
        'SSP5-8.5 Mean Composite Risk',
        'SSP2-4.5 High Risk (>0.7) %',
        'SSP3-7.0 High Risk (>0.7) %',
        'SSP5-8.5 High Risk (>0.7) %'
    ],
    'Value': [
        len(results_df),
        f"{results_df['tc_risk'].mean():.3f}",
        f"{results_df['tc_frequency'].mean():.3f}",
        f"{results_df['slr_rate_SSP2-4.5'].mean():.2f}",
        f"{results_df['slr_rate_SSP3-7.0'].mean():.2f}",
        f"{results_df['slr_rate_SSP5-8.5'].mean():.2f}",
        f"{results_df['composite_risk_SSP2-4.5'].mean():.3f}",
        f"{results_df['composite_risk_SSP3-7.0'].mean():.3f}",
        f"{results_df['composite_risk_SSP5-8.5'].mean():.3f}",
        f"{(results_df['composite_risk_SSP2-4.5'] > 0.7).sum() / len(results_df) * 100:.1f}%",
        f"{(results_df['composite_risk_SSP3-7.0'] > 0.7).sum() / len(results_df) * 100:.1f}%",
        f"{(results_df['composite_risk_SSP5-8.5'] > 0.7).sum() / len(results_df) * 100:.1f}%"
    ]
})

summary_stats.to_csv('outputs/summary_statistics.csv', index=False)
print("  Saved: outputs/summary_statistics.csv")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nKey Findings:")
print(f"- Total mangrove sample points analyzed: {len(results_df):,}")
print(f"- Mean SLR rate (SSP5-8.5): {results_df['slr_rate_SSP5-8.5'].mean():.2f} mm/yr")
print(f"- High/Very high risk mangroves (SSP5-8.5): {(results_df['composite_risk_SSP5-8.5'] > 0.7).sum() / len(results_df) * 100:.1f}%")
print(f"- Estimated carbon at risk (SSP5-8.5): {ecosystem_services_risk['SSP5-8.5']['carbon_at_risk_tg']:.0f} Tg C")
