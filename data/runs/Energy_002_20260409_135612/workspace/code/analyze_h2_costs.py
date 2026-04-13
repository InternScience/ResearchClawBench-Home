import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')

EUROPEAN_H2_BENCHMARK = {
    'optimistic': 4.0,
    'base': 5.0,
    'high_rate': 6.0,
}

SCENARIOS = {
    'Africa_base_8pct': {
        'wacc': 0.08,
        'pv_capex_kw': 500,
        'wind_capex_kw': 1100,
        'electrolyzer_capex_kw': 500,
        'battery_capex_kwh': 150,
        'h2_storage_capex_kwh': 15,
        'ammonia_capex_per_kgpy': 1.2,
        'reconversion_capex_per_kgpy': 0.9,
        'grid_connection_capex_per_kwkm': 0.7,
        'road_capex_per_tkm_year': 0.02,
        'desal_capex_factor': 1.0,
        'policy_credit_eurkg': 0.0,
    },
    'Africa_derisked_5pct': {
        'wacc': 0.05,
        'pv_capex_kw': 500,
        'wind_capex_kw': 1100,
        'electrolyzer_capex_kw': 500,
        'battery_capex_kwh': 150,
        'h2_storage_capex_kwh': 15,
        'ammonia_capex_per_kgpy': 1.2,
        'reconversion_capex_per_kgpy': 0.9,
        'grid_connection_capex_per_kwkm': 0.7,
        'road_capex_per_tkm_year': 0.02,
        'desal_capex_factor': 1.0,
        'policy_credit_eurkg': 0.0,
    },
    'Africa_high_rate_12pct': {
        'wacc': 0.12,
        'pv_capex_kw': 500,
        'wind_capex_kw': 1100,
        'electrolyzer_capex_kw': 500,
        'battery_capex_kwh': 150,
        'h2_storage_capex_kwh': 15,
        'ammonia_capex_per_kgpy': 1.2,
        'reconversion_capex_per_kgpy': 0.9,
        'grid_connection_capex_per_kwkm': 0.7,
        'road_capex_per_tkm_year': 0.02,
        'desal_capex_factor': 1.0,
        'policy_credit_eurkg': 0.0,
    },
    'Africa_derisked_plus_policy': {
        'wacc': 0.05,
        'pv_capex_kw': 450,
        'wind_capex_kw': 1000,
        'electrolyzer_capex_kw': 450,
        'battery_capex_kwh': 140,
        'h2_storage_capex_kwh': 14,
        'ammonia_capex_per_kgpy': 1.1,
        'reconversion_capex_per_kgpy': 0.85,
        'grid_connection_capex_per_kwkm': 0.6,
        'road_capex_per_tkm_year': 0.018,
        'desal_capex_factor': 0.95,
        'policy_credit_eurkg': 0.5,
    },
}

TECH = {
    'pv_life': 25,
    'wind_life': 25,
    'electrolyzer_life': 18,
    'battery_life': 15,
    'storage_life': 20,
    'ammonia_life': 25,
    'reconversion_life': 25,
    'shipping_life': 25,
}

H2_ENERGY_KWH_PER_KG = 52.0
WATER_M3_PER_KG = 0.018
NH3_CONV_ELEC_KWH_PER_KG = 9.0
NH3_RECONV_ELEC_KWH_PER_KG = 9.5
NH3_SHIPPING_EUR_PER_KG = 0.45
PORT_TERMINAL_EUR_PER_KG = 0.12
FIXED_OPEX_SHARE = 0.02
VAR_OPEX_EUR_PER_KG = 0.08
WATER_BASE_EUR_M3 = 0.6
DESAL_EXTRA_EUR_M3 = 0.4
ROAD_COST_EUR_PER_KGKM = 0.00018
GRID_LINE_CAPACITY_KW = 100000.0
PLANT_CF = 0.95
ANNUAL_KG = 600_000_000
HOURS = 8760


def crf(r, n):
    return r / (1 - (1 + r) ** (-n))


def normalize(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def load_data():
    df = pd.read_csv(DATA / 'hex_final_NA_min.csv')
    df['pv_cf'] = 0.18 + 0.12 * normalize(df['theo_pv'])
    df['wind_cf'] = 0.20 + 0.25 * normalize(df['theo_wind'])
    df['hybrid_cf'] = 0.55 * df['pv_cf'] + 0.45 * df['wind_cf']
    df['storage_penalty'] = 0.45 * normalize(df['road_dist_km']) + 0.55 * (1 - normalize(df['theo_wind']))
    df['storage_multiplier'] = 0.22 + 0.28 * df['storage_penalty']
    return df


def annual_h2_per_kw(cf):
    return HOURS * cf * PLANT_CF / H2_ENERGY_KWH_PER_KG


def component_cost_per_kg(capex_per_kw, cf, wacc, life, fixed_opex_share=FIXED_OPEX_SHARE):
    ann = capex_per_kw * (crf(wacc, life) + fixed_opex_share)
    return ann / annual_h2_per_kw(cf)


def component_cost_per_kg_from_kwh(capex_per_kwh, storage_hours, cf, wacc, life, fixed_opex_share=0.02):
    capex_per_kw_equiv = capex_per_kwh * storage_hours
    ann = capex_per_kw_equiv * (crf(wacc, life) + fixed_opex_share)
    return ann / annual_h2_per_kw(cf)


def process_site(row, params):
    wacc = params['wacc']
    pv_cost = component_cost_per_kg(params['pv_capex_kw'], row['pv_cf'], wacc, TECH['pv_life'])
    wind_cost = component_cost_per_kg(params['wind_capex_kw'], row['wind_cf'], wacc, TECH['wind_life'])
    renew_cost = 0.6 * pv_cost + 0.4 * wind_cost

    electrolyzer_cf = min(0.78, max(0.45, row['hybrid_cf'] + 0.18))
    elec_per_kw_year = HOURS * electrolyzer_cf * PLANT_CF
    electrolyzer_annual = params['electrolyzer_capex_kw'] * (crf(wacc, TECH['electrolyzer_life']) + 0.03)
    electrolyzer_cost = electrolyzer_annual / (elec_per_kw_year / H2_ENERGY_KWH_PER_KG)

    battery_hours = 1.5 + 4.0 * row['storage_multiplier']
    battery_cost = component_cost_per_kg_from_kwh(params['battery_capex_kwh'], battery_hours, row['hybrid_cf'], wacc, TECH['battery_life'])
    h2_storage_hours = 8 + 12 * row['storage_multiplier']
    h2_storage_cost = component_cost_per_kg_from_kwh(params['h2_storage_capex_kwh'], h2_storage_hours, row['hybrid_cf'], wacc, TECH['storage_life'])

    road_cost = ROAD_COST_EUR_PER_KGKM * row['road_dist_km']
    grid_ann = params['grid_connection_capex_per_kwkm'] * row['grid_dist_km'] * GRID_LINE_CAPACITY_KW * crf(wacc, 30)
    grid_cost = grid_ann / ANNUAL_KG

    water_cost = WATER_M3_PER_KG * (WATER_BASE_EUR_M3 + params['desal_capex_factor'] * DESAL_EXTRA_EUR_M3 * (1/(1+np.exp(-(50-row['ocean_dist_km'])/20))))
    water_transport = 0.00008 * min(row['waterbody_dist_km'], row['ocean_dist_km'])

    nh3_conv_ann = params['ammonia_capex_per_kgpy'] * ANNUAL_KG * (crf(wacc, TECH['ammonia_life']) + 0.02)
    nh3_conv = nh3_conv_ann / ANNUAL_KG + NH3_CONV_ELEC_KWH_PER_KG / H2_ENERGY_KWH_PER_KG * renew_cost

    reconv_ann = params['reconversion_capex_per_kgpy'] * ANNUAL_KG * (crf(wacc, TECH['reconversion_life']) + 0.025)
    reconv = reconv_ann / ANNUAL_KG + 0.22

    shipping = NH3_SHIPPING_EUR_PER_KG + 0.00025 * row['ocean_dist_km']
    port = PORT_TERMINAL_EUR_PER_KG

    production = renew_cost + electrolyzer_cost + battery_cost + h2_storage_cost + water_cost + water_transport + grid_cost + road_cost + VAR_OPEX_EUR_PER_KG
    delivered = production + nh3_conv + shipping + port + reconv - params['policy_credit_eurkg']

    return {
        'renew_cost': renew_cost,
        'electrolyzer_cost': electrolyzer_cost,
        'battery_cost': battery_cost,
        'h2_storage_cost': h2_storage_cost,
        'water_cost': water_cost + water_transport,
        'grid_road_cost': grid_cost + road_cost,
        'ammonia_conv_cost': nh3_conv,
        'shipping_port_cost': shipping + port,
        'reconversion_cost': reconv,
        'production_cost': production,
        'delivered_cost': delivered,
        'electrolyzer_cf': electrolyzer_cf,
        'battery_hours': battery_hours,
        'h2_storage_hours': h2_storage_hours,
    }


def run():
    df = load_data()
    records = []
    for scen, params in SCENARIOS.items():
        tmp = df.copy()
        metrics = tmp.apply(lambda r: pd.Series(process_site(r, params)), axis=1)
        tmp = pd.concat([tmp, metrics], axis=1)
        tmp['scenario'] = scen
        tmp['europe_benchmark'] = EUROPEAN_H2_BENCHMARK['base']
        if 'high_rate' in scen:
            tmp['europe_benchmark'] = EUROPEAN_H2_BENCHMARK['high_rate']
        elif 'derisked' in scen:
            tmp['europe_benchmark'] = EUROPEAN_H2_BENCHMARK['optimistic'] if 'policy' in scen else 4.5
        tmp['advantage_vs_europe'] = tmp['europe_benchmark'] - tmp['delivered_cost']
        tmp['competitive_vs_europe'] = tmp['delivered_cost'] <= tmp['europe_benchmark']
        tmp['rank_in_scenario'] = tmp['delivered_cost'].rank(method='first')
        records.append(tmp)
    res = pd.concat(records, ignore_index=True)
    res.to_csv(OUT / 'site_results_by_scenario.csv', index=False)

    summary = res.groupby('scenario').agg(
        min_delivered=('delivered_cost', 'min'),
        p10_delivered=('delivered_cost', lambda x: np.quantile(x, 0.1)),
        median_delivered=('delivered_cost', 'median'),
        mean_delivered=('delivered_cost', 'mean'),
        competitive_sites=('competitive_vs_europe', 'sum'),
        total_sites=('competitive_vs_europe', 'size'),
        avg_advantage=('advantage_vs_europe', 'mean'),
    ).reset_index()
    summary['competitive_share'] = summary['competitive_sites'] / summary['total_sites']
    summary.to_csv(OUT / 'scenario_summary.csv', index=False)

    best = res.sort_values(['scenario', 'delivered_cost']).groupby('scenario').head(5)
    best.to_csv(OUT / 'best_sites.csv', index=False)

    baseline = res[res['scenario'] == 'Africa_base_8pct'][['hex_id','delivered_cost']].rename(columns={'delivered_cost':'base_cost'})
    derisk = res[res['scenario'] == 'Africa_derisked_5pct'][['hex_id','delivered_cost']].rename(columns={'delivered_cost':'derisk_cost'})
    high = res[res['scenario'] == 'Africa_high_rate_12pct'][['hex_id','delivered_cost']].rename(columns={'delivered_cost':'high_cost'})
    comp = baseline.merge(derisk, on='hex_id').merge(high, on='hex_id')
    comp['derisk_savings'] = comp['base_cost'] - comp['derisk_cost']
    comp['highrate_penalty'] = comp['high_cost'] - comp['base_cost']
    comp.to_csv(OUT / 'financing_sensitivity_by_site.csv', index=False)

    world = gpd.read_file(DATA / 'africa_map' / 'ne_10m_admin_0_countries.shp')
    africa = world[world['CONTINENT'].isin(['Africa','Europe'])].copy()

    # Figure 1: site map baseline
    fig, ax = plt.subplots(figsize=(11,10))
    africa[africa['CONTINENT']=='Africa'].plot(ax=ax, color='#f3efe0', edgecolor='gray', linewidth=0.5)
    base = res[res['scenario']=='Africa_base_8pct']
    sc = ax.scatter(base['lon'], base['lat'], c=base['delivered_cost'], s=110, cmap='viridis_r', edgecolor='black', linewidth=0.4)
    plt.colorbar(sc, ax=ax, label='Delivered H2 cost to Europe (€/kg)')
    ax.set_title('African candidate sites: baseline delivered green hydrogen cost in 2030')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout(); plt.savefig(IMG / 'africa_baseline_cost_map.png', dpi=220); plt.close()

    # Figure 2 scenario comparison
    fig, ax = plt.subplots(figsize=(12,7))
    order = summary.sort_values('min_delivered')['scenario']
    sns.boxplot(data=res, x='scenario', y='delivered_cost', order=order, ax=ax, color='#b7d4ea')
    sns.stripplot(data=res, x='scenario', y='delivered_cost', order=order, ax=ax, color='black', size=4, alpha=0.55)
    ax.set_title('Delivered cost distribution across financing and policy scenarios')
    ax.set_xlabel('Scenario'); ax.set_ylabel('Delivered H2 cost (€/kg)')
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.savefig(IMG / 'scenario_cost_distribution.png', dpi=220); plt.close()

    # Figure 3 cost breakdown top sites
    top_base = base.nsmallest(5, 'delivered_cost').copy()
    comp_cols = ['renew_cost','electrolyzer_cost','battery_cost','h2_storage_cost','water_cost','grid_road_cost','ammonia_conv_cost','shipping_port_cost','reconversion_cost']
    melt = top_base[['hex_id'] + comp_cols].melt(id_vars='hex_id', var_name='component', value_name='eur_per_kg')
    fig, ax = plt.subplots(figsize=(13,7))
    pivot = melt.pivot(index='hex_id', columns='component', values='eur_per_kg')[comp_cols]
    pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_title('Cost breakdown of five least-cost baseline sites')
    ax.set_ylabel('€/kg delivered H2')
    ax.set_xlabel('Site')
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', frameon=True, title='Component')
    plt.tight_layout(); plt.savefig(IMG / 'top_sites_cost_breakdown.png', dpi=220); plt.close()

    # Figure 4 financing effect scatter
    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(comp['base_cost'], comp['derisk_savings'], s=80, c=comp['highrate_penalty'], cmap='magma', edgecolor='black', linewidth=0.4)
    for _, r in comp.nsmallest(5, 'base_cost').iterrows():
        ax.annotate(r['hex_id'], (r['base_cost'], r['derisk_savings']), fontsize=9)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('High-rate penalty vs base (€/kg)')
    ax.set_xlabel('Baseline delivered cost (€/kg)')
    ax.set_ylabel('Savings from de-risking 8%→5% WACC (€/kg)')
    ax.set_title('Financing conditions reshape site-level competitiveness')
    plt.tight_layout(); plt.savefig(IMG / 'financing_sensitivity_scatter.png', dpi=220); plt.close()

    # Figure 5 validation/comparison with Europe benchmark
    compare = summary.copy()
    europe_map = {'Africa_base_8pct':EUROPEAN_H2_BENCHMARK['base'],'Africa_derisked_5pct':4.5,'Africa_high_rate_12pct':EUROPEAN_H2_BENCHMARK['high_rate'],'Africa_derisked_plus_policy':EUROPEAN_H2_BENCHMARK['optimistic']}
    compare['europe_cost'] = compare['scenario'].map(europe_map)
    x = np.arange(len(compare))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11,7))
    ax.bar(x-w/2, compare['min_delivered'], width=w, label='African min delivered cost', color='#1b9e77')
    ax.bar(x+w/2, compare['europe_cost'], width=w, label='European green H2 benchmark', color='#d95f02')
    ax.set_xticks(x)
    ax.set_xticklabels(compare['scenario'], rotation=20)
    ax.set_ylabel('€/kg H2')
    ax.set_title('Best African delivered cost versus stylized European production benchmark')
    ax.legend()
    plt.tight_layout(); plt.savefig(IMG / 'africa_vs_europe_benchmark.png', dpi=220); plt.close()

    # Figure 6 resource overview
    fig, axes = plt.subplots(1,2, figsize=(13,5.5))
    sns.scatterplot(data=df, x='theo_pv', y='theo_wind', size='ocean_dist_km', hue='grid_dist_km', palette='crest', ax=axes[0])
    axes[0].set_title('Resource endowment and infrastructure context')
    axes[0].set_xlabel('PV potential index')
    axes[0].set_ylabel('Wind potential index')
    sns.histplot(base['delivered_cost'], bins=10, kde=True, ax=axes[1], color='#4c78a8')
    axes[1].axvline(base['delivered_cost'].min(), color='red', linestyle='--', label='Minimum')
    axes[1].axvline(base['delivered_cost'].median(), color='black', linestyle=':', label='Median')
    axes[1].set_title('Baseline delivered cost distribution')
    axes[1].set_xlabel('€/kg delivered H2')
    axes[1].legend()
    plt.tight_layout(); plt.savefig(IMG / 'data_overview.png', dpi=220); plt.close()

if __name__ == '__main__':
    run()
