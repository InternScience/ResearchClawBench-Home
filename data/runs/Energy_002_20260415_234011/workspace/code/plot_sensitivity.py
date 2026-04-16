import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('outputs/results.csv')

# Calculate LCOA for different WACC rates
wacc_range = np.linspace(0.02, 0.12, 11)

def crf(w, l):
    if w == 0: return 1/l
    return (w * (1+w)**l) / ((1+w)**l - 1)

def calc_lcoa_for_wacc(row, wacc):
    annual_h2_production = 8760 * 0.7 * 20
    
    # PV
    ann_capex_pv = 1.5 * 1470000.0 * crf(wacc, 20)
    opex_pv = 1.5 * 0.02 * 1470000.0
    cost_pv = (ann_capex_pv + opex_pv) / annual_h2_production
    
    # Wind
    ann_capex_wind = 1.0 * 1580000.0 * crf(wacc, 20)
    opex_wind = 1.0 * 0.03 * 1580000.0
    cost_wind = (ann_capex_wind + opex_wind) / annual_h2_production
    
    # Electrolyzer
    ann_capex_ely = 1.0 * 1250000.0 * crf(wacc, 20)
    opex_ely = 1.0 * 0.02 * 1250000.0
    cost_ely = (ann_capex_ely + opex_ely) / annual_h2_production
    
    # Water
    water_cost = 0.001 * row['ocean_dist_km'] * annual_h2_production
    cost_water = water_cost / annual_h2_production
    
    # Ammonia Synthesis
    cost_nh3 = 750 * crf(wacc, 25) + 0.015 * 750
    
    # Shipping
    shipping = 0.39
    
    # Cracking
    cost_cracking = 500 * crf(wacc, 25) + 0.02 * 500
    
    return cost_pv + cost_wind + cost_ely + cost_water + cost_nh3 + shipping + cost_cracking

# Select top location
top_loc = df.nsmallest(1, 'LCOA_Derisked').iloc[0]

lcoa_values = [calc_lcoa_for_wacc(top_loc, w) for w in wacc_range]

plt.figure(figsize=(8, 5))
plt.plot(wacc_range * 100, lcoa_values, marker='o', linestyle='-', color='b')
plt.axhline(y=5.0, color='r', linestyle='--', label='European Benchmark (5.0 €/kgH2)')
plt.xlabel('WACC (%)')
plt.ylabel('Delivered Cost (€/kgH2)')
plt.title(f'Sensitivity of Delivered Cost to WACC for Best Location ({top_loc["hex_id"]})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('report/images/wacc_sensitivity.png', dpi=300)

print("Sensitivity chart saved to report/images/wacc_sensitivity.png")
