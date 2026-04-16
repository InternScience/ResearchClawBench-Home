import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import geopandas as gpd

# 1. Constants & Assumptions based on Literature (2030 projections)
# WACC assumptions
WACC_BASE = 0.08      # 8% baseline cost of capital (from paper_001.pdf)
WACC_DERISKED = 0.04  # 4% de-risked cost of capital (assuming lower rate for EU-backed/de-risked projects)

# CAPEX assumptions (from paper_000.pdf)
CAPEX_PV = 1470000.0        # €/MW
CAPEX_WIND = 1580000.0      # €/MW
CAPEX_ELECTROLYZER = 1250000.0 # €/MW
CAPEX_DESAL = 2000000.0     # €/MW (assumption if not found)

# OPEX assumptions
OPEX_PV = 0.02 * CAPEX_PV
OPEX_WIND = 0.03 * CAPEX_WIND
OPEX_ELECTROLYZER = 0.02 * CAPEX_ELECTROLYZER

# Lifetimes
LIFETIME_PV = 20
LIFETIME_WIND = 20
LIFETIME_ELECTROLYZER = 20

# Ammonia synthesis (from paper_000.pdf)
# Ammonia synthesis Electricity demand 2.809 kWh/kgH2
# Ammonia synthesis Capex coefficient 0.75717 kWh/gH2/year -> 757.17 €/kgH2/year capacity (assuming typo in paper, meaning €/(kgH2/year))
# Let's use standard values for Ammonia synthesis and cracking
CAPEX_AMMONIA_SYNTHESIS = 750  # €/(kgH2/year)
OPEX_AMMONIA_SYNTHESIS = 0.015 * CAPEX_AMMONIA_SYNTHESIS
LIFETIME_AMMONIA = 25

CAPEX_AMMONIA_CRACKING = 500   # €/(kgH2/year)
OPEX_AMMONIA_CRACKING = 0.02 * CAPEX_AMMONIA_CRACKING
LIFETIME_CRACKING = 25

# Shipping to Europe
SHIPPING_COST_AMMONIA = 0.39 # €/kgH2 (from paper_001.pdf)

# European benchmark LCOH (from paper_001.pdf or general knowledge for 2030)
EU_LCOH_BENCHMARK = 5.0 # €/kgH2

def crf(wacc, lifetime):
    """Capital Recovery Factor"""
    if wacc == 0:
        return 1 / lifetime
    return (wacc * (1 + wacc)**lifetime) / ((1 + wacc)**lifetime - 1)

def calculate_lcoh(row, wacc):
    # Simplified LCOH calculation
    # Assume 1 MW Electrolyzer, optimal mix of PV and Wind based on theo_pv and theo_wind
    
    # Capacity factors
    cf_pv = row['theo_pv']
    cf_wind = row['theo_wind']
    
    # To run 1 MW electrolyzer at high utilization (e.g., 70%), need more PV/Wind
    # Let's assume a sizing ratio: 1 MW Electrolyzer, 1.5 MW PV, 1 MW Wind
    # This is a simplification for the model
    size_pv = 1.5
    size_wind = 1.0
    size_ely = 1.0
    
    annual_h2_production = 8760 * 0.7 * 20 # kg H2 per MW electrolyzer per year (approx 50 kWh/kg)
    
    # Annualized CAPEX
    ann_capex_pv = size_pv * CAPEX_PV * crf(wacc, LIFETIME_PV)
    ann_capex_wind = size_wind * CAPEX_WIND * crf(wacc, LIFETIME_WIND)
    ann_capex_ely = size_ely * CAPEX_ELECTROLYZER * crf(wacc, LIFETIME_ELECTROLYZER)
    
    # OPEX
    opex_pv = size_pv * OPEX_PV
    opex_wind = size_wind * OPEX_WIND
    opex_ely = size_ely * OPEX_ELECTROLYZER
    
    # Water cost (desalination + transport)
    # Assume 10 L water per kg H2
    water_dist = row['ocean_dist_km']
    water_cost = 0.001 * water_dist * annual_h2_production # simplified €/kg
    
    total_annual_cost = (ann_capex_pv + ann_capex_wind + ann_capex_ely +
                         opex_pv + opex_wind + opex_ely + water_cost)
                         
    lcoh = total_annual_cost / annual_h2_production
    return lcoh

def calculate_lcoa(lcoh, wacc):
    # LCOA (Levelized Cost of Ammonia delivered and cracked)
    # Ammonia synthesis
    ann_capex_nh3 = CAPEX_AMMONIA_SYNTHESIS * crf(wacc, LIFETIME_AMMONIA)
    opex_nh3 = OPEX_AMMONIA_SYNTHESIS
    cost_nh3_synthesis = ann_capex_nh3 + opex_nh3
    
    # Cracking
    ann_capex_crack = CAPEX_AMMONIA_CRACKING * crf(wacc, LIFETIME_CRACKING)
    opex_crack = OPEX_AMMONIA_CRACKING
    cost_cracking = ann_capex_crack + opex_crack
    
    # Total delivered cost of H2 via Ammonia
    delivered_cost = lcoh + cost_nh3_synthesis + SHIPPING_COST_AMMONIA + cost_cracking
    return delivered_cost

# Load data
df = pd.read_csv('data/hex_final_NA_min.csv')

# Calculate costs
df['LCOH_Base'] = df.apply(lambda row: calculate_lcoh(row, WACC_BASE), axis=1)
df['LCOA_Base'] = df.apply(lambda row: calculate_lcoa(row['LCOH_Base'], WACC_BASE), axis=1)

df['LCOH_Derisked'] = df.apply(lambda row: calculate_lcoh(row, WACC_DERISKED), axis=1)
df['LCOA_Derisked'] = df.apply(lambda row: calculate_lcoa(row['LCOH_Derisked'], WACC_DERISKED), axis=1)

df.to_csv('outputs/results.csv', index=False)

# Identify least-cost locations
top_locations = df.nsmallest(5, 'LCOA_Derisked')
top_locations.to_csv('outputs/top_locations.csv', index=False)

print("Processing complete. Results saved to outputs/results.csv")
