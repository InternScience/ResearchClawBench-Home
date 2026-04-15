import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapefile
from shapely.geometry import Point, shape as shapely_shape

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid')

ANNUAL_H2_KG = 100_000_000  # 100 kt/y reference project scale
ELECTROLYZER_KWH_PER_KG = 52.0
WATER_L_PER_KG = 15.0
EUR_PER_M3_WATER = 1.0

ASSUMPTIONS = {
    'technology': {
        'pv_capex_eur_per_kw_2030': 420,
        'pv_fixed_opex_frac': 0.018,
        'wind_capex_eur_per_kw_2030': 980,
        'wind_fixed_opex_frac': 0.03,
        'electrolyzer_capex_eur_per_kw_2030': 480,
        'electrolyzer_fixed_opex_frac': 0.03,
        'electrolyzer_lifetime_years': 20,
        'renewable_lifetime_years': 25,
        'ammonia_synthesis_eur_per_kg_h2': 0.85,
        'ammonia_shipping_to_europe_eur_per_kg_h2': 0.35,
        'ammonia_reconversion_eur_per_kg_h2': 0.95,
        'europe_terminal_eur_per_kg_h2': 0.15,
    },
    'infrastructure': {
        'grid_connection_capex_eur_per_km': 350_000,
        'grid_connection_opex_frac': 0.02,
        'road_connection_capex_eur_per_km': 500_000,
        'road_connection_opex_frac': 0.03,
        'port_connection_capex_eur_per_km': 800_000,
        'port_connection_opex_frac': 0.025,
        'water_pipeline_capex_eur_per_km': 150_000,
        'water_pipeline_opex_frac': 0.02,
        'water_treatment_energy_kwh_per_m3': 4.0,
    },
    'capacity_factor_mapping': {
        'africa_pv_cf_min': 0.22,
        'africa_pv_cf_max': 0.33,
        'africa_wind_cf_min': 0.26,
        'africa_wind_cf_max': 0.55,
        'europe_pv_cf': 0.16,
        'europe_wind_cf': 0.38,
    },
    'europe_benchmark': {
        'grid_balancing_and_land_eur_per_kg': 0.20,
        'water_and_misc_eur_per_kg': 0.05,
    },
    'scenarios': {
        'base_highrisk': {
            'label': 'Base/high-risk financing',
            'africa_wacc': 0.14,
            'europe_wacc': 0.06,
            'infra_multiplier': 1.00,
            'ammonia_chain_credit_eur_per_kg': 0.0,
        },
        'derisked': {
            'label': 'De-risked Africa',
            'africa_wacc': 0.08,
            'europe_wacc': 0.06,
            'infra_multiplier': 1.00,
            'ammonia_chain_credit_eur_per_kg': 0.0,
        },
        'high_rates': {
            'label': 'High-rate environment',
            'africa_wacc': 0.18,
            'europe_wacc': 0.10,
            'infra_multiplier': 1.00,
            'ammonia_chain_credit_eur_per_kg': 0.0,
        },
        'corridor_policy': {
            'label': 'De-risked + export corridor policy',
            'africa_wacc': 0.08,
            'europe_wacc': 0.06,
            'infra_multiplier': 0.75,
            'ammonia_chain_credit_eur_per_kg': 0.35,
        },
    },
}


def crf(rate: float, years: int) -> float:
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def map_range(series: pd.Series, low: float, high: float) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series(np.full(len(series), (low + high) / 2), index=series.index)
    return low + (series - smin) * (high - low) / (smax - smin)


def lcoe_from_cf(capex, opex_frac, wacc, life, cf):
    annualized = capex * crf(wacc, life) + capex * opex_frac
    return annualized / (8760 * cf)


def infra_cost_per_kg(distance_km, capex_per_km, opex_frac, wacc, life, multiplier=1.0):
    annualized = (distance_km * capex_per_km * multiplier) * (crf(wacc, life) + opex_frac)
    return annualized / ANNUAL_H2_KG


def assign_countries(df: pd.DataFrame) -> pd.Series:
    shp = shapefile.Reader(str(DATA / 'africa_map' / 'ne_10m_admin_0_countries.shp'))
    name_idx = [f[0] for f in shp.fields[1:]].index('NAME')
    africa_shapes = []
    for sr in shp.iterShapeRecords():
        rec = sr.record[name_idx]
        geom = shapely_shape(sr.shape.__geo_interface__)
        minx, miny, maxx, maxy = geom.bounds
        if maxy < -40 or miny > 40 or maxx < -30 or minx > 60:
            continue
        africa_shapes.append((rec, geom))
    names = []
    for _, row in df.iterrows():
        pt = Point(float(row['lon']), float(row['lat']))
        found = 'Unknown'
        for nm, geom in africa_shapes:
            if geom.contains(pt):
                found = nm
                break
        names.append(found)
    return pd.Series(names, index=df.index)


def europe_benchmark_cost(europe_wacc: float) -> dict:
    tech = ASSUMPTIONS['technology']
    cfm = ASSUMPTIONS['capacity_factor_mapping']
    pv_lcoe = lcoe_from_cf(tech['pv_capex_eur_per_kw_2030'], tech['pv_fixed_opex_frac'], europe_wacc, tech['renewable_lifetime_years'], cfm['europe_pv_cf'])
    wind_lcoe = lcoe_from_cf(tech['wind_capex_eur_per_kw_2030'], tech['wind_fixed_opex_frac'], europe_wacc, tech['renewable_lifetime_years'], cfm['europe_wind_cf'])
    chosen_lcoe = min(pv_lcoe, wind_lcoe)
    chosen_tech = 'wind' if wind_lcoe <= pv_lcoe else 'pv'
    electrolyzer_annual = tech['electrolyzer_capex_eur_per_kw_2030'] * (crf(europe_wacc, tech['electrolyzer_lifetime_years']) + tech['electrolyzer_fixed_opex_frac'])
    electrolyzer_kg_per_kw = 8760 * 0.55 / ELECTROLYZER_KWH_PER_KG
    electrolyzer_cost = electrolyzer_annual / electrolyzer_kg_per_kw
    total = chosen_lcoe * ELECTROLYZER_KWH_PER_KG + electrolyzer_cost + ASSUMPTIONS['europe_benchmark']['grid_balancing_and_land_eur_per_kg'] + ASSUMPTIONS['europe_benchmark']['water_and_misc_eur_per_kg']
    return {
        'europe_wacc': europe_wacc,
        'europe_chosen_power': chosen_tech,
        'europe_power_lcoe_eur_per_kwh': chosen_lcoe,
        'europe_electrolyzer_cost_eur_per_kg': electrolyzer_cost,
        'europe_total_cost_eur_per_kg': total,
    }


def build_results():
    df = pd.read_csv(DATA / 'hex_final_NA_min.csv')
    df['country'] = assign_countries(df)
    cfm = ASSUMPTIONS['capacity_factor_mapping']
    df['pv_cf'] = map_range(df['theo_pv'], cfm['africa_pv_cf_min'], cfm['africa_pv_cf_max'])
    df['wind_cf'] = map_range(df['theo_wind'], cfm['africa_wind_cf_min'], cfm['africa_wind_cf_max'])

    records = []
    europe_rows = []
    tech = ASSUMPTIONS['technology']
    infra = ASSUMPTIONS['infrastructure']

    for scen_key, scen in ASSUMPTIONS['scenarios'].items():
        africa_wacc = scen['africa_wacc']
        europe = europe_benchmark_cost(scen['europe_wacc'])
        europe['scenario'] = scen_key
        europe['scenario_label'] = scen['label']
        europe_rows.append(europe)

        pv_lcoe = lcoe_from_cf(tech['pv_capex_eur_per_kw_2030'], tech['pv_fixed_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], df['pv_cf'])
        wind_lcoe = lcoe_from_cf(tech['wind_capex_eur_per_kw_2030'], tech['wind_fixed_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], df['wind_cf'])
        renewable_choice = np.where(wind_lcoe <= pv_lcoe, 'wind', 'pv')
        chosen_lcoe = np.minimum(pv_lcoe, wind_lcoe)
        chosen_cf = np.where(wind_lcoe <= pv_lcoe, df['wind_cf'], df['pv_cf'])
        electrolyzer_annual = tech['electrolyzer_capex_eur_per_kw_2030'] * (crf(africa_wacc, tech['electrolyzer_lifetime_years']) + tech['electrolyzer_fixed_opex_frac'])
        electrolyzer_kg_per_kw = 8760 * chosen_cf / ELECTROLYZER_KWH_PER_KG
        electrolyzer_cost = electrolyzer_annual / electrolyzer_kg_per_kw
        electricity_cost = chosen_lcoe * ELECTROLYZER_KWH_PER_KG
        water_cost = (WATER_L_PER_KG / 1000.0) * EUR_PER_M3_WATER + (WATER_L_PER_KG / 1000.0) * infra['water_treatment_energy_kwh_per_m3'] * chosen_lcoe

        grid_adder = infra_cost_per_kg(df['grid_dist_km'], infra['grid_connection_capex_eur_per_km'], infra['grid_connection_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], scen['infra_multiplier'])
        road_adder = infra_cost_per_kg(df['road_dist_km'], infra['road_connection_capex_eur_per_km'], infra['road_connection_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], scen['infra_multiplier'])
        port_adder = infra_cost_per_kg(df['ocean_dist_km'], infra['port_connection_capex_eur_per_km'], infra['port_connection_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], scen['infra_multiplier'])
        water_adder = infra_cost_per_kg(df['waterbody_dist_km'], infra['water_pipeline_capex_eur_per_km'], infra['water_pipeline_opex_frac'], africa_wacc, tech['renewable_lifetime_years'], scen['infra_multiplier'])

        production = electricity_cost + electrolyzer_cost + water_cost + grid_adder + road_adder + water_adder
        ammonia_chain = (
            tech['ammonia_synthesis_eur_per_kg_h2']
            + tech['ammonia_shipping_to_europe_eur_per_kg_h2']
            + tech['ammonia_reconversion_eur_per_kg_h2']
            + tech['europe_terminal_eur_per_kg_h2']
            - scen['ammonia_chain_credit_eur_per_kg']
        )
        delivered = production + port_adder + ammonia_chain
        competitive = delivered <= europe['europe_total_cost_eur_per_kg']

        temp = df.copy()
        temp['scenario'] = scen_key
        temp['scenario_label'] = scen['label']
        temp['africa_wacc'] = africa_wacc
        temp['europe_wacc'] = scen['europe_wacc']
        temp['selected_power'] = renewable_choice
        temp['selected_cf'] = chosen_cf
        temp['selected_lcoe_eur_per_kwh'] = chosen_lcoe
        temp['electricity_cost_eur_per_kg'] = electricity_cost
        temp['electrolyzer_cost_eur_per_kg'] = electrolyzer_cost
        temp['water_cost_eur_per_kg'] = water_cost
        temp['grid_adder_eur_per_kg'] = grid_adder
        temp['road_adder_eur_per_kg'] = road_adder
        temp['port_adder_eur_per_kg'] = port_adder
        temp['water_infra_adder_eur_per_kg'] = water_adder
        temp['africa_production_cost_eur_per_kg'] = production
        temp['ammonia_chain_cost_eur_per_kg'] = ammonia_chain
        temp['delivered_cost_eur_per_kg'] = delivered
        temp['europe_benchmark_eur_per_kg'] = europe['europe_total_cost_eur_per_kg']
        temp['competitive_vs_europe'] = competitive
        temp['cost_gap_vs_europe_eur_per_kg'] = delivered - europe['europe_total_cost_eur_per_kg']
        records.append(temp)

    results = pd.concat(records, ignore_index=True)
    europe_df = pd.DataFrame(europe_rows)
    return results, europe_df


def make_maps(results: pd.DataFrame):
    shp = shapefile.Reader(str(DATA / 'africa_map' / 'ne_10m_admin_0_countries.shp'))
    fig_scenarios = [
        ('base_highrisk', 'africa_delivered_cost_map_base.png'),
        ('derisked', 'africa_delivered_cost_map_derisked.png'),
    ]
    for scen, fname in fig_scenarios:
        sub = results[results['scenario'] == scen]
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        for s in shp.shapes():
            pts = np.asarray(s.points)
            if len(pts) == 0:
                continue
            parts = list(s.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                seg = pts[parts[i]:parts[i+1]]
                ax.plot(seg[:, 0], seg[:, 1], color='lightgray', linewidth=0.4, zorder=1)
        sc = ax.scatter(sub['lon'], sub['lat'], c=sub['delivered_cost_eur_per_kg'], s=80, cmap='viridis_r', edgecolor='black', linewidth=0.4, zorder=3)
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('Delivered cost to Europe (€/kg H$_2$)')
        ax.set_title(sub['scenario_label'].iloc[0])
        ax.set_xlim(-20, 55)
        ax.set_ylim(-38, 38)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(IMG / fname, dpi=200)
        plt.close(fig)


def make_distribution_plot(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=results, x='scenario_label', y='delivered_cost_eur_per_kg', ax=ax, color='#a6cee3')
    sns.stripplot(data=results, x='scenario_label', y='delivered_cost_eur_per_kg', ax=ax, color='#1f78b4', size=4, alpha=0.7)
    ax.set_ylabel('Delivered cost to Europe (€/kg H$_2$)')
    ax.set_xlabel('Scenario')
    ax.set_title('Scenario distribution of delivered African hydrogen costs')
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    plt.savefig(IMG / 'scenario_cost_distributions.png', dpi=200)
    plt.close(fig)


def make_competitiveness_plot(results: pd.DataFrame, europe_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette('Set2', n_colors=results['scenario'].nunique())
    for color, (scen, sub) in zip(palette, results.groupby('scenario')):
        x = np.sort(sub['delivered_cost_eur_per_kg'].values)
        y = np.arange(1, len(x) + 1) / len(x)
        label = sub['scenario_label'].iloc[0]
        ax.plot(x, y, label=label, linewidth=2, color=color)
        bench = float(europe_df.loc[europe_df['scenario'] == scen, 'europe_total_cost_eur_per_kg'].iloc[0])
        ax.axvline(bench, color=color, linestyle='--', alpha=0.65)
    ax.set_xlabel('Delivered African hydrogen cost (€/kg H$_2$)')
    ax.set_ylabel('Share of African sites at or below cost')
    ax.set_title('Competitiveness curves vs Europe benchmark (dashed lines)')
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(IMG / 'competitiveness_vs_europe.png', dpi=200)
    plt.close(fig)


def make_sensitivity_plot(df: pd.DataFrame):
    waccs = np.arange(0.06, 0.181, 0.01)
    tech = ASSUMPTIONS['technology']
    cfm = ASSUMPTIONS['capacity_factor_mapping']
    tmp = df.copy()
    tmp['pv_cf'] = map_range(tmp['theo_pv'], cfm['africa_pv_cf_min'], cfm['africa_pv_cf_max'])
    tmp['wind_cf'] = map_range(tmp['theo_wind'], cfm['africa_wind_cf_min'], cfm['africa_wind_cf_max'])
    rows = []
    for wacc in waccs:
        pv_lcoe = lcoe_from_cf(tech['pv_capex_eur_per_kw_2030'], tech['pv_fixed_opex_frac'], wacc, tech['renewable_lifetime_years'], tmp['pv_cf'])
        wind_lcoe = lcoe_from_cf(tech['wind_capex_eur_per_kw_2030'], tech['wind_fixed_opex_frac'], wacc, tech['renewable_lifetime_years'], tmp['wind_cf'])
        chosen_lcoe = np.minimum(pv_lcoe, wind_lcoe)
        chosen_cf = np.where(wind_lcoe <= pv_lcoe, tmp['wind_cf'], tmp['pv_cf'])
        electrolyzer_annual = tech['electrolyzer_capex_eur_per_kw_2030'] * (crf(wacc, tech['electrolyzer_lifetime_years']) + tech['electrolyzer_fixed_opex_frac'])
        electrolyzer_kg_per_kw = 8760 * chosen_cf / ELECTROLYZER_KWH_PER_KG
        electrolyzer_cost = electrolyzer_annual / electrolyzer_kg_per_kw
        electricity_cost = chosen_lcoe * ELECTROLYZER_KWH_PER_KG
        production = electricity_cost + electrolyzer_cost
        rows.append(pd.DataFrame({'africa_wacc': wacc, 'production_cost': production}))
    sens = pd.concat(rows, ignore_index=True)
    q = sens.groupby('africa_wacc')['production_cost'].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    q.columns = ['africa_wacc', 'p10', 'p50', 'p90']
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(q['africa_wacc'] * 100, q['p50'], color='#d95f02', linewidth=2, label='Median production cost')
    ax.fill_between(q['africa_wacc'] * 100, q['p10'], q['p90'], color='#fdb863', alpha=0.35, label='P10-P90 range')
    ax.set_xlabel('African WACC (%)')
    ax.set_ylabel('Production cost before ammonia chain (€/kg H$_2$)')
    ax.set_title('Financing sensitivity of African production cost')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG / 'financing_sensitivity.png', dpi=200)
    plt.close(fig)
    return q


def main():
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    results, europe_df = build_results()

    scenario_assumptions = ASSUMPTIONS
    (OUT / 'scenario_assumptions.json').write_text(json.dumps(scenario_assumptions, indent=2))
    results.to_csv(OUT / 'site_cost_results.csv', index=False)
    europe_df.to_csv(OUT / 'europe_benchmark_results.csv', index=False)

    comp = results.groupby(['scenario', 'scenario_label']).agg(
        min_delivered_cost_eur_per_kg=('delivered_cost_eur_per_kg', 'min'),
        median_delivered_cost_eur_per_kg=('delivered_cost_eur_per_kg', 'median'),
        mean_delivered_cost_eur_per_kg=('delivered_cost_eur_per_kg', 'mean'),
        competitive_share=('competitive_vs_europe', 'mean'),
        best_site=('hex_id', lambda s: results.loc[s.index, 'hex_id'].iloc[results.loc[s.index, 'delivered_cost_eur_per_kg'].argmin()]),
    ).reset_index()
    comp = comp.merge(europe_df[['scenario', 'europe_total_cost_eur_per_kg']], on='scenario', how='left')
    comp.to_csv(OUT / 'competitive_summary.csv', index=False)

    for scen, fname in [('base_highrisk', 'map_data_base.csv'), ('derisked', 'map_data_derisked.csv')]:
        results.loc[results['scenario'] == scen, ['hex_id','country','lat','lon','delivered_cost_eur_per_kg','competitive_vs_europe']].to_csv(OUT / fname, index=False)
    results[['scenario','scenario_label','hex_id','delivered_cost_eur_per_kg']].to_csv(OUT / 'scenario_boxplot_data.csv', index=False)

    curve_rows = []
    for scen, sub in results.groupby('scenario'):
        x = np.sort(sub['delivered_cost_eur_per_kg'].values)
        y = np.arange(1, len(x) + 1) / len(x)
        curve_rows.append(pd.DataFrame({'scenario': scen, 'cost_eur_per_kg': x, 'share_of_sites': y}))
    pd.concat(curve_rows, ignore_index=True).to_csv(OUT / 'competitiveness_curve_data.csv', index=False)

    sens_q = make_sensitivity_plot(pd.read_csv(DATA / 'hex_final_NA_min.csv'))
    sens_q.to_csv(OUT / 'sensitivity_data.csv', index=False)

    make_maps(results)
    make_distribution_plot(results)
    make_competitiveness_plot(results, europe_df)

    summary_rows = []
    for scen, sub in results.groupby('scenario'):
        best = sub.nsmallest(5, 'delivered_cost_eur_per_kg')[['hex_id','country','delivered_cost_eur_per_kg','cost_gap_vs_europe_eur_per_kg']]
        for _, row in best.iterrows():
            summary_rows.append({'scenario': scen, **row.to_dict()})
    pd.DataFrame(summary_rows).to_csv(OUT / 'best_sites_top5_by_scenario.csv', index=False)

    validation = {
        'verified_directly_from_workspace': [
            'Input dataset has 30 sites and nine columns.',
            'Shapefile is readable with pyshp.',
            'Scenario-wise site results, benchmark results, and figure data were generated locally.'
        ],
        'related_work_used_for_method_shape': [
            'GeoH2 papers motivated geospatial site-wise cost modelling and explicit transport/conversion representation.',
            'Financing literature motivated scenario-specific WACC changes and interest-rate sensitivity.'
        ],
        'assumptions_and_limitations': [
            'The site table includes renewable-potential proxies rather than hourly profiles, so the model uses a capacity-factor mapping rather than full temporal optimization.',
            'The Europe benchmark is stylized and does not represent one exact European location.',
            'Ammonia shipping and reconversion costs are represented with transparent fixed 2030 adders instead of vessel-by-vessel logistics optimization.'
        ]
    }
    (OUT / 'validation_summary.json').write_text(json.dumps(validation, indent=2))

    checklist = json.loads((OUT / 'method_fidelity_checklist.json').read_text())
    checklist['status'] = {
        'implemented_exactly': False,
        'implemented_faithfully_with_simplifications': True,
        'notes': 'Implemented transparent site-wise production plus ammonia-chain delivery with financing scenarios and Europe benchmark, but used simplified capacity-factor mapping and fixed ammonia-chain adders due available data.'
    }
    (OUT / 'method_fidelity_checklist.json').write_text(json.dumps(checklist, indent=2))

if __name__ == '__main__':
    main()
