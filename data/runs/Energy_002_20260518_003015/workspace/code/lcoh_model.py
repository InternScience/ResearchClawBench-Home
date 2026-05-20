#!/usr/bin/env python3
"""
Geospatial Levelized Cost of Hydrogen (LCOH) Model - CORRECTED
African Green Hydrogen Delivered to Europe via Ammonia Shipping (2030)

Based on methodology from:
- GeoH2 (Halloran et al., 2023)
- Muller et al. (2023) - Kenya case study  
- Steffen (2020) - Cost of capital for RE
- Schmidt et al. (2019) - Interest rate effects

Key: 1 kg H2 = 33.33 kWh (LHV), needs ~51.3 kWh electricity at 65% efficiency
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

KWH_PER_KG_H2 = 33.33
NH3_RATIO = 34.0 / 6.0  # kg NH3 per kg H2 (stoichiometric)
EU_HUBS = {'Rotterdam': 10800, 'Hamburg': 11200, 'Barcelona': 8500}

P = {
    'pv_capex_kw': 900, 'pv_om_pct': 0.015, 'pv_lifetime': 25,
    'wind_capex_kw': 1100, 'wind_om_pct': 0.02, 'wind_lifetime': 25,
    'elec_capex_kw': 600, 'elec_efficiency': 0.65, 'elec_om_pct': 0.03,
    'elec_lifetime': 15, 'elec_cf': 0.50,
    'water_demand_l_kg': 21, 'water_cost_eur_m3': 1.25,
    'water_treat_kwh_m3': 0.4, 'desal_kwh_m3': 3.7, 'water_transport': 0.1,
    'nh3_elec_kwh_kg_h2': 1.5, 'nh3_capex_kw': 450, 'nh3_om_pct': 0.02, 'nh3_lifetime': 25,
    'crack_heat_kwh_kg_h2': 4.2, 'crack_capex_kw': 350, 'crack_om_pct': 0.025,
    'crack_lifetime': 20, 'crack_heat_price': 0.06,
    'nh3_ship_eur_tkm': 0.008, 'nh3_boiloff': 0.005,
    'nh3_stor_capex': 15, 'nh3_stor_om': 0.02,
    'nh3_port_eur_t': 8,
    'eu_pv_capex': 1200, 'eu_wind_capex': 1400, 'eu_elec_capex': 700,
    'eu_pv_cf': 0.14, 'eu_wind_cf': 0.28, 'eu_water_demand_l_kg': 15,
}

SCENARIOS = {
    'Optimistic (4%)':  {'wacc': 0.04},
    'De-risked (6%)':   {'wacc': 0.06},
    'Base (8%)':        {'wacc': 0.08},
    'High-risk (12%)':  {'wacc': 0.12},
    'EU baseline (4%)': {'wacc': 0.04},
}

def crf(r, n):
    if r <= 0: return 1.0 / n
    return r * (1 + r)**n / ((1 + r)**n - 1)

def compute_lcoe(capex, om_pct, life, cf, wacc):
    return (capex * crf(wacc, life) + capex * om_pct) / (cf * 8760)

def h2kg_per_kw(eff, cf):
    return eff * cf * 8760 / KWH_PER_KG_H2

def site_costs(site, wacc):
    pv_cf = site['theo_pv']
    ws = site['theo_wind'] * 10
    wind_cf = min(max(0.30 * (ws / 7.5)**3, 0.05), 0.55)
    lcoe_pv = compute_lcoe(P['pv_capex_kw'], P['pv_om_pct'], P['pv_lifetime'], pv_cf, wacc)
    lcoe_wd = compute_lcoe(P['wind_capex_kw'], P['wind_om_pct'], P['wind_lifetime'], wind_cf, wacc)
    if lcoe_pv <= lcoe_wd:
        lcoe_re, tech, cf = lcoe_pv, 'PV', pv_cf
    else:
        lcoe_re, tech, cf = lcoe_wd, 'Wind', wind_cf
    h2kg = h2kg_per_kw(P['elec_efficiency'], P['elec_cf'])
    kwh_el_per_kg = KWH_PER_KG_H2 / P['elec_efficiency']
    elec_cost = kwh_el_per_kg * lcoe_re
    elec_cap = P['elec_capex_kw'] * crf(wacc, P['elec_lifetime']) / h2kg
    elec_om = P['elec_capex_kw'] * P['elec_om_pct'] / h2kg
    wd = site['waterbody_dist_km']
    if wd < 50:
        wt_kwh = P['water_treat_kwh_m3']
    else:
        wt_kwh = P['desal_kwh_m3']
    water_m3_per_kg = P['water_demand_l_kg'] / 1000
    water_treat_elec = wt_kwh * lcoe_re
    water_transport = P['water_transport'] * wd / 100
    wt_cost = (P['water_cost_eur_m3'] + water_transport + water_treat_elec) * water_m3_per_kg
    scale = 100000
    rd = site['road_dist_km']
    gd = site['grid_dist_km']
    road_capex = 482000 * rd * crf(wacc, 30) / (scale * 1000)
    road_om = 7150 * rd / (scale * 1000)
    grid_capex = 90000 * gd * crf(wacc, 30) / (scale * 1000)
    infra = road_capex + road_om + grid_capex
    prod = elec_cost + elec_cap + elec_om + wt_cost + infra
    nh3_elec = P['nh3_elec_kwh_kg_h2'] * lcoe_re
    nh3_cap = P['nh3_capex_kw'] * crf(wacc, P['nh3_lifetime']) / h2kg
    nh3_om = P['nh3_capex_kw'] * P['nh3_om_pct'] / h2kg
    nh3_conv = nh3_elec + nh3_cap + nh3_om
    stor_tonnes = scale * NH3_RATIO * 7 / 365
    stor = (P['nh3_stor_capex'] * stor_tonnes * crf(wacc, 20) + P['nh3_stor_capex'] * P['nh3_stor_om'] * stor_tonnes) / (scale * 1000)
    port = P['nh3_port_eur_t'] * NH3_RATIO / 1000
    avg_d = np.mean(list(EU_HUBS.values()))
    ship = P['nh3_ship_eur_tkm'] * avg_d * NH3_RATIO / 1000
    t_days = avg_d / (15 * 24)
    boil = P['nh3_boiloff'] * t_days * NH3_RATIO * 200 / 1000
    conv_trans = nh3_conv + stor + port + ship + boil
    crack_heat = P['crack_heat_kwh_kg_h2'] * P['crack_heat_price']
    crack_cap = P['crack_capex_kw'] * crf(wacc, P['crack_lifetime']) / h2kg
    crack_om = P['crack_capex_kw'] * P['crack_om_pct'] / h2kg
    crack = crack_heat + crack_cap + crack_om
    total = prod + conv_trans + crack
    return {
        're_tech': tech, 're_cf': round(cf, 4), 'lcoe_re': round(lcoe_re, 6),
        'electricity_cost': round(elec_cost, 4), 'electrolyzer_capex': round(elec_cap, 4),
        'electrolyzer_om': round(elec_om, 4), 'water_cost': round(wt_cost, 4),
        'infrastructure': round(infra, 4), 'ammonia_conversion': round(nh3_conv, 4),
        'storage': round(stor, 4), 'port_handling': round(port, 4),
        'shipping': round(ship + boil, 4), 'cracking': round(crack, 4),
        'lcoh_production': round(prod, 4), 'total_delivered': round(total, 4),
    }

def eu_baseline_cost(wacc):
    lcoe_pv = compute_lcoe(P['eu_pv_capex'], P['pv_om_pct'], P['pv_lifetime'], P['eu_pv_cf'], wacc)
    lcoe_wd = compute_lcoe(P['eu_wind_capex'], P['wind_om_pct'], P['wind_lifetime'], P['eu_wind_cf'], wacc)
    lcoe_re = min(lcoe_pv, lcoe_wd)
    tech = 'PV' if lcoe_pv <= lcoe_wd else 'Wind'
    h2kg = h2kg_per_kw(P['elec_efficiency'], P['elec_cf'])
    kwh_el = KWH_PER_KG_H2 / P['elec_efficiency']
    elec_cost = kwh_el * lcoe_re
    elec_cap = P['eu_elec_capex'] * crf(wacc, P['elec_lifetime']) / h2kg
    elec_om = P['eu_elec_capex'] * P['elec_om_pct'] / h2kg
    water = P['water_cost_eur_m3'] * P['eu_water_demand_l_kg'] / 1000
    lcoh = elec_cost + elec_cap + elec_om + water
    return {'lcoh': round(lcoh, 4), 'lcoe_re': round(lcoe_re, 6), 're_tech': tech}

def country(lat, lon):
    if lon < 12:
        return 'Namibia' if lat < -17 else 'Angola'
    elif lon < 18:
        return 'South Africa' if lat < -28 else ('Namibia' if lat < -17 else 'Angola')
    elif lon < 25:
        return 'Botswana' if lat < -22 else ('Namibia' if lat < -17 else 'South Africa')
    else:
        return 'South Africa'

df = pd.read_csv('data/hex_final_NA_min.csv')
df['country'] = df.apply(lambda r: country(r['lat'], r['lon']), axis=1)
print(f"Loaded {len(df)} sites: {df['country'].value_counts().to_dict()}")

results = {}
for sname, sp in SCENARIOS.items():
    wacc = sp['wacc']
    rows = []
    for _, site in df.iterrows():
        if 'EU baseline' in sname:
            eb = eu_baseline_cost(wacc)
            h2kg = h2kg_per_kw(P['elec_efficiency'], P['elec_cf'])
            kwh_el = KWH_PER_KG_H2 / P['elec_efficiency']
            r = {
                're_tech': eb['re_tech'], 're_cf': 0, 'lcoe_re': eb['lcoe_re'],
                'electricity_cost': round(kwh_el * eb['lcoe_re'], 4),
                'electrolyzer_capex': round(P['eu_elec_capex']*crf(wacc,P['elec_lifetime'])/h2kg, 4),
                'electrolyzer_om': round(P['eu_elec_capex']*P['elec_om_pct']/h2kg, 4),
                'water_cost': round(P['water_cost_eur_m3']*P['eu_water_demand_l_kg']/1000, 4),
                'infrastructure': 0, 'ammonia_conversion': 0, 'storage': 0,
                'port_handling': 0, 'shipping': 0, 'cracking': 0,
                'lcoh_production': eb['lcoh'], 'total_delivered': eb['lcoh'],
            }
        else:
            r = site_costs(site, wacc)
        r.update({'hex_id': site['hex_id'], 'lat': site['lat'], 'lon': site['lon'],
                   'country': site['country'], 'scenario': sname})
        rows.append(r)
    rdf = pd.DataFrame(rows)
    results[sname] = rdf
    if 'EU baseline' not in sname:
        print(f"{sname}: mean={rdf['total_delivered'].mean():.2f}, min={rdf['total_delivered'].min():.2f}, max={rdf['total_delivered'].max():.2f}, std={rdf['total_delivered'].std():.2f}")
    else:
        print(f"{sname}: {rdf['total_delivered'].iloc[0]:.2f}")

combined = pd.concat(results.values(), ignore_index=True)
combined.to_csv('outputs/lcoh_per_site.csv', index=False)

stats = {}
for nm, rd in results.items():
    stats[nm] = {
        'mean': round(rd['total_delivered'].mean(), 4),
        'median': round(rd['total_delivered'].median(), 4),
        'min': round(rd['total_delivered'].min(), 4),
        'max': round(rd['total_delivered'].max(), 4),
        'std': round(rd['total_delivered'].std(), 4),
        'p25': round(rd['total_delivered'].quantile(0.25), 4),
        'p75': round(rd['total_delivered'].quantile(0.75), 4),
        'n': len(rd)
    }
with open('outputs/scenario_comparison.json', 'w') as f:
    json.dump(stats, f, indent=2)

base = results['Base (8%)']
cols = ['hex_id','lat','lon','country','re_tech','re_cf','lcoe_re','electricity_cost',
        'electrolyzer_capex','electrolyzer_om','water_cost','infrastructure',
        'ammonia_conversion','storage','port_handling','shipping','cracking',
        'lcoh_production','total_delivered']
base[cols].to_json('outputs/cost_breakdown_base.json', orient='records', indent=2)

top10 = base.sort_values('total_delivered').head(10)
top10[['hex_id','lat','lon','country','re_tech','re_cf','lcoh_production','total_delivered']].to_csv('outputs/top10_sites.csv', index=False)

cg = base.groupby('country').agg(
    mean_lcoh=('total_delivered', 'mean'), min_lcoh=('total_delivered', 'min'),
    max_lcoh=('total_delivered', 'max'), std_lcoh=('total_delivered', 'std'),
    count=('total_delivered', 'count'), mean_prod=('lcoh_production', 'mean')
).round(4)
cg.to_csv('outputs/country_summary.csv')

print(f"\nTop 5 cheapest (Base 8%):")
print(top10.head(5)[['hex_id','country','re_tech','re_cf','total_delivered']].to_string(index=False))
eu_val = results['EU baseline (4%)']['total_delivered'].iloc[0]
print(f"\nEU baseline: {eu_val:.2f} EUR/kg")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
eu_mean = eu_val

# FIG 1
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(base['lon'], base['lat'], c=base['total_delivered'], cmap='RdYlGn_r', s=140, edgecolors='black', linewidth=0.5, zorder=5)
top5 = base.nsmallest(5, 'total_delivered')
ax.scatter(top5['lon'], top5['lat'], facecolors='none', edgecolors='blue', s=280, linewidths=2, zorder=6, label='Top 5 lowest cost')
plt.colorbar(sc, ax=ax, shrink=0.8, label='Delivered Cost to Europe (EUR/kg H$_2$)')
ax.set_xlabel('Longitude ($^\\circ$E)', fontsize=12)
ax.set_ylabel('Latitude ($^\\circ$N)', fontsize=12)
ax.set_title('Delivered Cost of Green Hydrogen to Europe via Ammonia\n(African Production Sites, Base Scenario: 8% WACC)', fontsize=13)
ax.legend(loc='lower left', fontsize=10)
for _, r in top5.head(3).iterrows():
    ax.annotate(f"EUR{r['total_delivered']:.2f}", (r['lon'], r['lat']), xytext=(10,10), textcoords='offset points', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), arrowprops=dict(arrowstyle='->', color='blue'))
plt.tight_layout()
plt.savefig('report/images/fig1_spatial_lcoh_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1")

# FIG 2
fig, ax = plt.subplots(figsize=(16, 7))
bs = base.sort_values('total_delivered').reset_index(drop=True)
comps = ['electricity_cost','electrolyzer_capex','electrolyzer_om','water_cost','infrastructure','ammonia_conversion','storage','port_handling','shipping','cracking']
labels = ['Electricity','Electrolyzer CAPEX','Electrolyzer O&M','Water','Infrastructure','NH$_3$ Conversion','Storage','Port Handling','Shipping','Cracking']
ccolors = sns.color_palette("Set3", len(comps))
bot = np.zeros(len(bs))
x = np.arange(len(bs))
for c, lb, cl in zip(comps, labels, ccolors):
    ax.bar(x, bs[c].values, bottom=bot, label=lb, color=cl, edgecolor='white', linewidth=0.3)
    bot += bs[c].values
ax.set_xlabel('Production Site (ranked by total delivered cost)', fontsize=11)
ax.set_ylabel('Cost (EUR/kg H$_2$)', fontsize=11)
ax.set_title('Cost Breakdown of Delivered Green Hydrogen to Europe\n(Base Scenario: 8% WACC)', fontsize=12)
ax.set_xticks(x[::3])
ax.set_xticklabels([r['hex_id'] for _, r in bs.iloc[::3].iterrows()], rotation=45, fontsize=8)
ax.legend(loc='upper left', ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig('report/images/fig2_cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

# FIG 3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
so = ['High-risk (12%)','Base (8%)','De-risked (6%)','Optimistic (4%)']
sd = [results[s]['total_delivered'].values for s in so]
bcolors = ['#e74c3c','#f39c12','#2ecc71','#3498db']
bp = axes[0].boxplot(sd, labels=['12%\n(High-risk)','8%\n(Base)','6%\n(De-risked)','4%\n(Optimistic)'], patch_artist=True, widths=0.6)
for p, c in zip(bp['boxes'], bcolors):
    p.set_facecolor(c); p.set_alpha(0.7)
for m in bp['medians']:
    m.set_color('black'); m.set_linewidth(2)
axes[0].axhline(y=eu_mean, color='red', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}/kg')
axes[0].set_ylabel('Delivered Cost (EUR/kg H$_2$)', fontsize=11)
axes[0].set_title('LCOH Sensitivity to Discount Rate\n(WACC Scenarios)', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(bottom=0)
means = [results[s]['total_delivered'].mean() for s in so]
stds = [results[s]['total_delivered'].std() for s in so]
waccs = [SCENARIOS[s]['wacc']*100 for s in so]
axes[1].errorbar(waccs, means, yerr=stds, fmt='o-', color='#2c3e50', markersize=10, capsize=5, linewidth=2, markerfacecolor='#3498db', markeredgecolor='black')
axes[1].axhline(y=eu_mean, color='red', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}/kg')
axes[1].set_xlabel('WACC / Discount Rate (%)', fontsize=11)
axes[1].set_ylabel('Mean Delivered Cost (EUR/kg H$_2$)', fontsize=11)
axes[1].set_title('Mean LCOH vs. Discount Rate', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].set_ylim(bottom=0)
axes[1].axhspan(0, eu_mean, alpha=0.1, color='green')
plt.tight_layout()
plt.savefig('report/images/fig3_scenario_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3")

# FIG 4
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bs2 = base.sort_values('total_delivered').reset_index(drop=True)
cbar = ['#2ecc71' if v <= eu_mean else '#e74c3c' for v in bs2['total_delivered']]
axes[0].bar(range(len(bs2)), bs2['total_delivered'], color=cbar, edgecolor='white', linewidth=0.3)
axes[0].axhline(y=eu_mean, color='red', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}/kg')
axes[0].set_xlabel('Site (ranked by cost)', fontsize=11)
axes[0].set_ylabel('Delivered Cost (EUR/kg H$_2$)', fontsize=11)
axes[0].set_title('Site-level Cost Competitiveness\nvs European Green H$_2$ Baseline', fontsize=12)
axes[0].legend(fontsize=9)
nc = (bs2['total_delivered'] <= eu_mean).sum()
axes[0].text(0.95, 0.95, f'{nc}/{len(bs2)} sites\ncompetitive', transform=axes[0].transAxes, fontsize=12, fontweight='bold', va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
fracs = [(results[s]['total_delivered'] <= eu_mean).mean()*100 for s in so]
bars = axes[1].bar(range(len(so)), fracs, color=bcolors, edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(so)))
axes[1].set_xticklabels(['12%\n(High-risk)','8%\n(Base)','6%\n(De-risked)','4%\n(Optimistic)'])
axes[1].set_ylabel('% of Sites Competitive\nvs EU Baseline', fontsize=11)
axes[1].set_title('Fraction of Competitive Sites\nby Financing Scenario', fontsize=12)
axes[1].set_ylim(0, 105)
for b, f in zip(bars, fracs):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{f:.0f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig4_competitiveness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# FIG 5
fig, ax = plt.subplots(figsize=(12, 7))
cheapest = df.iloc[base['total_delivered'].idxmin()]
base_cost = base['total_delivered'].mean()
sens_items = [
    ('Electrolyzer CAPEX', 'elec_capex_kw', 600, 0.25),
    ('PV CAPEX', 'pv_capex_kw', 900, 0.25),
    ('Wind CAPEX', 'wind_capex_kw', 1100, 0.25),
    ('Electrolyzer Efficiency', 'elec_efficiency', 0.65, 0.20),
    ('RE Capacity Factor', 'elec_cf', 0.50, 0.20),
    ('NH$_3$ Shipping Cost', 'nh3_ship_eur_tkm', 0.008, 0.40),
    ('Discount Rate (WACC)', 'wacc', 0.08, 0.50),
    ('NH$_3$ Synth. Electricity', 'nh3_elec_kwh_kg_h2', 1.5, 0.30),
]
tornado = []
for name, pk, bv, var in sens_items:
    if pk == 'wacc':
        lo = site_costs(cheapest, bv*(1-var))['total_delivered']
        hi = site_costs(cheapest, bv*(1+var))['total_delivered']
    else:
        orig = P[pk]
        P[pk] = bv*(1-var); lo = site_costs(cheapest, 0.08)['total_delivered']
        P[pk] = bv*(1+var); hi = site_costs(cheapest, 0.08)['total_delivered']
        P[pk] = orig
    tornado.append({'name': name, 'lo': lo, 'hi': hi, 'spread': abs(hi-lo)})
tdf = pd.DataFrame(tornado).sort_values('spread')
yp = range(len(tdf))
ax.barh(yp, tdf['hi']-base_cost, left=base_cost, height=0.6, color='#e74c3c', alpha=0.8, label='Parameter increase')
ax.barh(yp, tdf['lo']-base_cost, left=base_cost, height=0.6, color='#3498db', alpha=0.8, label='Parameter decrease')
ax.set_yticks(yp); ax.set_yticklabels(tdf['name'], fontsize=10)
ax.axvline(x=base_cost, color='black', linewidth=1.5, label=f'Base mean: EUR{base_cost:.2f}')
ax.axvline(x=eu_mean, color='green', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}')
ax.set_xlabel('Delivered Cost (EUR/kg H$_2$)', fontsize=11)
ax.set_title('Tornado Sensitivity Analysis\n(Parameter variation on cheapest site)', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/fig5_tornado_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# FIG 6
fig, ax = plt.subplots(figsize=(10, 6))
countries = sorted(base['country'].unique())
cdata = [base[base['country']==c]['total_delivered'].values for c in countries]
bp = ax.boxplot(cdata, labels=countries, patch_artist=True, widths=0.5)
cpal = sns.color_palette("Set2", len(countries))
for p, c in zip(bp['boxes'], cpal):
    p.set_facecolor(c); p.set_alpha(0.7)
ax.axhline(y=eu_mean, color='red', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}')
ax.set_ylabel('Delivered Cost (EUR/kg H$_2$)', fontsize=11)
ax.set_title('Cost Distribution by Country (Base Scenario: 8% WACC)', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('report/images/fig6_country_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6")

# FIG 7
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
wacc_vals = [0.04, 0.06, 0.08, 0.12]
wacc_labels = ['4% WACC', '6% WACC', '8% WACC', '12% WACC']
vlabs = ['Electricity', 'Electrolyzer', 'Water+Infra', 'NH$_3$ Conv.', 'Shipping+Port', 'Cracking']
vcols = sns.color_palette("Set2", 6)
for i, (w, lbl) in enumerate(zip(wacc_vals, wacc_labels)):
    rdf = pd.DataFrame([site_costs(cheapest, w)])
    vals = [rdf['electricity_cost'].iloc[0], rdf['electrolyzer_capex'].iloc[0]+rdf['electrolyzer_om'].iloc[0],
            rdf['water_cost'].iloc[0]+rdf['infrastructure'].iloc[0],
            rdf['ammonia_conversion'].iloc[0]+rdf['storage'].iloc[0],
            rdf['shipping'].iloc[0]+rdf['port_handling'].iloc[0],
            rdf['cracking'].iloc[0]]
    axes[i].pie(vals, labels=vlabs, colors=vcols, autopct='%1.0f%%', textprops={'fontsize':8}, pctdistance=0.75)
    axes[i].set_title(f'{lbl}\n(Total: EUR{sum(vals):.2f}/kg)', fontsize=11, fontweight='bold')
fig.suptitle('Cost Component Shares at Cheapest Site by Discount Rate', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig7_component_shares.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7")

# FIG 8
fig, ax = plt.subplots(figsize=(10, 6))
distances = np.arange(2000, 15000, 500)
cheapest_site = df.iloc[base['total_delivered'].idxmin()]
costs_by_dist = []
for d in distances:
    ship = P['nh3_ship_eur_tkm'] * d * NH3_RATIO / 1000
    t_days = d / (15*24)
    boil = P['nh3_boiloff'] * t_days * NH3_RATIO * 200 / 1000
    s_costs = site_costs(cheapest_site, 0.08)
    adjusted = s_costs['total_delivered'] - s_costs['shipping'] + ship + boil
    costs_by_dist.append(adjusted)
ax.plot(distances/1000, costs_by_dist, 'o-', color='#2c3e50', linewidth=2, markersize=4)
ax.axhline(y=eu_mean, color='red', linestyle='--', linewidth=2, label=f'EU baseline: EUR{eu_mean:.2f}')
ylim_lo = min(costs_by_dist) - 0.5
for port, d in EU_HUBS.items():
    ax.axvline(x=d/1000, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(d/1000, ylim_lo + 0.1, port, rotation=90, fontsize=9, color='gray', va='bottom')
ax.set_xlabel('Shipping Distance (x1000 km)', fontsize=11)
ax.set_ylabel('Delivered Cost (EUR/kg H$_2$)', fontsize=11)
ax.set_title('Sensitivity of Delivered Cost to Shipping Distance\n(Cheapest Site, Base WACC)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(2, 14)
plt.tight_layout()
plt.savefig('report/images/fig8_shipping_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8")

# FIG 9
fig, ax = plt.subplots(figsize=(14, 6))
cheapest_r = base.loc[base['total_delivered'].idxmin()]
waterfall_items = [
    ('Electricity', cheapest_r['electricity_cost']),
    ('Electrolyzer', cheapest_r['electrolyzer_capex'] + cheapest_r['electrolyzer_om']),
    ('Water+Infra', cheapest_r['water_cost'] + cheapest_r['infrastructure']),
    ('NH$_3$ Conversion', cheapest_r['ammonia_conversion']),
    ('Storage', cheapest_r['storage']),
    ('Port+Shipping', cheapest_r['port_handling'] + cheapest_r['shipping']),
    ('Cracking (EU)', cheapest_r['cracking']),
]
labels_w = [x[0] for x in waterfall_items]
vals_w = [x[1] for x in waterfall_items]
running = 0
wcols = sns.color_palette("Set2", len(vals_w))
for i, v in enumerate(vals_w):
    ax.bar(i, v, bottom=running, color=wcols[i], edgecolor='white', linewidth=0.5, width=0.7)
    ax.text(i, running + v/2, f'EUR{v:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
    running += v
ax.bar(len(vals_w)+0.5, running, color='#2c3e50', edgecolor='white', linewidth=0.5, width=0.7)
ax.text(len(vals_w)+0.5, running/2, f'TOTAL\nEUR{running:.2f}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.set_xticks(list(range(len(labels_w))) + [len(vals_w)+0.5])
ax.set_xticklabels(labels_w + ['Total'], rotation=30, fontsize=10)
ax.set_ylabel('Cost (EUR/kg H$_2$)', fontsize=11)
ax.set_title(f'Cost Waterfall for Cheapest Site ({cheapest_r["hex_id"]}, {cheapest_r["country"]})\nDelivered to Europe via Ammonia', fontsize=12)
ax.axhline(y=eu_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'EU baseline: EUR{eu_mean:.2f}')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('report/images/fig9_cost_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9")

print(f"\n=== ALL ANALYSIS COMPLETE ===")
print(f"Outputs: {sorted(os.listdir('outputs/'))}")
print(f"Images: {sorted(os.listdir('report/images/'))}")
