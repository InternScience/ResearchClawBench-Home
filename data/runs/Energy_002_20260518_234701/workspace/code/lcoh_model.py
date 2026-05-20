"""
Transparent geospatial LCOH model for African green hydrogen export to Europe
via ammonia shipping and reconversion (2030).
"""
import csv
import json
import math
import os

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Technology assumptions for 2030 (sources: IRENA 2024, IEA GHR 2023, GeoH2)
PV_CAPEX = 400          # €/kW
PV_OPEX = 10            # €/kW/year
PV_LIFETIME = 25        # years

WIND_CAPEX = 800        # €/kW
WIND_OPEX = 25          # €/kW/year
WIND_LIFETIME = 25      # years

ELEC_CAPEX = 500        # €/kW
ELEC_OPEX_RATE = 0.02   # fraction of CAPEX/year
ELEC_LIFETIME = 20      # years
ELEC_EFF = 50           # kWh electricity per kg H2

# Practical performance ratios (theoretical -> practical capacity factor)
PV_PR = 0.35
WIND_PR = 0.45

# Water
WATER_DEMAND = 21e-3    # m3/kg H2
WATER_COST_BASE = 1.25  # €/m3
WATER_TRANS_COST = 0.1 / 100  # €/km/m3
FRESH_TREAT_E = 0.4     # kWh/m3
OCEAN_TREAT_E = 3.7     # kWh/m3

# Downstream ammonia chain (€/kg H2) – midpoint estimates from literature
NH3_SYNTHESIS = 0.40
NH3_SHIPPING = 0.50
NH3_CRACKING = 0.60
LOCAL_TRUCK = 0.10      # flat adder for inland transport to port
DOWNSTREAM_COST = NH3_SYNTHESIS + NH3_SHIPPING + NH3_CRACKING + LOCAL_TRUCK

# European benchmark (same model, different resource quality)
EU_PV_CF = 0.15
EU_WIND_CF = 0.30
EU_PV_CAPEX = 500
EU_WIND_CAPEX = 1000
EU_ELEC_CAPEX = 600
EU_WACC = 0.05
EU_LOCAL_DIST = 0.20    # local distribution in Europe

# Scenarios
SCENARIOS = {
    "strong_de-risking": {"wacc": 0.05, "label": "Strong de-risking (5% WACC)"},
    "moderate_de-risking": {"wacc": 0.08, "label": "Moderate de-risking (8% WACC)"},
    "no_de-risking": {"wacc": 0.12, "label": "No de-risking (12% WACC)"},
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def pvf(r, n):
    return ((1 + r) ** n - 1) / ((1 + r) ** n * r)

def lcoe(capex, opex, cf, r, n):
    annual_gen = cf * 8760  # kWh/kW/year
    if annual_gen <= 0:
        return float('inf')
    return (capex / pvf(r, n) + opex) / annual_gen * 1000  # €/MWh

def lcoh_production(lcoe_mwh, cf, r, elec_capex, elec_opex_rate, elec_lifetime, elec_eff):
    annual_h2 = cf * 8760 / elec_eff  # kg/kW electrolyzer/year
    if annual_h2 <= 0:
        return float('inf')
    elec_capex_annual = elec_capex / pvf(r, elec_lifetime)
    elec_opex_annual = elec_opex_rate * elec_capex
    elec_cost_per_kg = (elec_capex_annual + elec_opex_annual) / annual_h2
    elec_energy_cost = lcoe_mwh * elec_eff / 1000
    return elec_cost_per_kg + elec_energy_cost

def water_cost(row, lcoe_mwh):
    # choose nearest source: freshwater body or ocean
    d_water = float(row['waterbody_dist_km'])
    d_ocean = float(row['ocean_dist_km'])
    if d_water <= d_ocean:
        source = 'fresh'
        treat_e = FRESH_TREAT_E
        dist = d_water
    else:
        source = 'ocean'
        treat_e = OCEAN_TREAT_E
        dist = d_ocean
    water_specific = WATER_COST_BASE + WATER_TRANS_COST * dist
    water_e_cost = lcoe_mwh * treat_e * WATER_DEMAND / 1000
    return (water_specific + water_e_cost) * WATER_DEMAND

# ---------------------------------------------------------------------------
# European benchmark
# ---------------------------------------------------------------------------

def european_benchmark():
    lcoe_pv = lcoe(EU_PV_CAPEX, PV_OPEX, EU_PV_CF, EU_WACC, PV_LIFETIME)
    lcoe_wind = lcoe(EU_WIND_CAPEX, WIND_OPEX, EU_WIND_CF, EU_WACC, WIND_LIFETIME)
    best_lcoe = min(lcoe_pv, lcoe_wind)
    best_cf = EU_PV_CF if lcoe_pv < lcoe_wind else EU_WIND_CF
    lcoh = lcoh_production(best_lcoe, best_cf, EU_WACC,
                           EU_ELEC_CAPEX, ELEC_OPEX_RATE, ELEC_LIFETIME, ELEC_EFF)
    return lcoh + EU_LOCAL_DIST

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = 'data/hex_final_NA_min.csv'
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    eu_cost = european_benchmark()

    with open(data_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        out = {
            'hex_id': row['hex_id'],
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'grid_dist_km': float(row['grid_dist_km']),
            'road_dist_km': float(row['road_dist_km']),
            'ocean_dist_km': float(row['ocean_dist_km']),
            'waterbody_dist_km': float(row['waterbody_dist_km']),
        }
        cf_pv = float(row['theo_pv']) * PV_PR
        cf_wind = float(row['theo_wind']) * WIND_PR
        for scen, cfg in SCENARIOS.items():
            r = cfg['wacc']
            lcoe_pv = lcoe(PV_CAPEX, PV_OPEX, cf_pv, r, PV_LIFETIME)
            lcoe_wind = lcoe(WIND_CAPEX, WIND_OPEX, cf_wind, r, WIND_LIFETIME)
            best_lcoe = min(lcoe_pv, lcoe_wind)
            best_tech = 'PV' if lcoe_pv < lcoe_wind else 'Wind'
            best_cf = cf_pv if lcoe_pv < lcoe_wind else cf_wind
            lcoh_prod = lcoh_production(best_lcoe, best_cf, r,
                                        ELEC_CAPEX, ELEC_OPEX_RATE, ELEC_LIFETIME, ELEC_EFF)
            # add water cost
            w_cost = water_cost(row, best_lcoe)
            lcoh_prod += w_cost
            delivered = lcoh_prod + DOWNSTREAM_COST
            out[f'lcoh_prod_{scen}'] = round(lcoh_prod, 4)
            out[f'delivered_{scen}'] = round(delivered, 4)
            out[f'tech_{scen}'] = best_tech
            out[f'lcoe_{scen}'] = round(best_lcoe, 4)
        results.append(out)

    # Write CSV
    fieldnames = list(results[0].keys())
    with open(os.path.join(out_dir, 'lcoh_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary JSON
    summary = {
        'european_benchmark_eur_per_kg': round(eu_cost, 4),
        'downstream_cost_eur_per_kg': round(DOWNSTREAM_COST, 4),
        'scenarios': {}
    }
    for scen in SCENARIOS:
        vals = [r[f'delivered_{scen}'] for r in results]
        prod_vals = [r[f'lcoh_prod_{scen}'] for r in results]
        summary['scenarios'][scen] = {
            'min_delivered_eur_per_kg': round(min(vals), 4),
            'max_delivered_eur_per_kg': round(max(vals), 4),
            'mean_delivered_eur_per_kg': round(sum(vals)/len(vals), 4),
            'min_production_eur_per_kg': round(min(prod_vals), 4),
            'max_production_eur_per_kg': round(max(prod_vals), 4),
            'mean_production_eur_per_kg': round(sum(prod_vals)/len(prod_vals), 4),
            'competitive_sites': sum(1 for v in vals if v < eu_cost),
            'least_cost_hex': min(results, key=lambda x: x[f'delivered_{scen}'])['hex_id']
        }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Model run complete.")
    print("European benchmark:", round(eu_cost, 2), "€/kg")
    for scen, cfg in SCENARIOS.items():
        s = summary['scenarios'][scen]
        print(f"{cfg['label']}: delivered {s['min_delivered_eur_per_kg']}-{s['max_delivered_eur_per_kg']} €/kg, competitive sites {s['competitive_sites']}/{len(results)}")

if __name__ == '__main__':
    main()
