#!/usr/bin/env python3
"""Transparent geospatial LCOH model for African green hydrogen imports to Europe.

The model is intentionally reduced-form and fully auditable. It uses site-level
simulated renewable potentials and infrastructure distances to estimate 2030
hydrogen production cost, ammonia conversion/shipping/reconversion adders, and
competitiveness against a Europe domestic green-H2 comparator under finance and
policy scenarios.
"""
from __future__ import annotations
import json, math, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data'
OUT=BASE/'outputs'
IMG=BASE/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper')

try:
    import geopandas as gpd
    HAS_GPD=True
except Exception:
    HAS_GPD=False

H2_KWH_LHV=33.33
ELECTROLYZER_KWH_PER_KG=50.0
HOURS=8760
EUR_PER_USD=0.92

# 2030 central assumptions. Values chosen to match related-work anchors for
# 2030 African production LCOH (best sites ~€1.8-3/kg) and delivered costs
# around literature import values after carrier/reconversion adders.
ASSUMPTIONS={
 'currency':'2023-2026 EUR, real',
 'formulas':{
   'crf':'r*(1+r)^n/((1+r)^n-1)',
   'renewable_lcoe':'(capex*crf + fixed_om*capex)/(cf*8760) + variable_om',
   'production_lcoh':'electricity_cost*50 + electrolyzer_capex_annual_per_kg + water + infrastructure + grid_connection',
   'delivered_cost':'production_lcoh + ammonia_synthesis + road_to_port + ocean_shipping + reconversion - policy_credit'
 },
 'technology':{
   'pv_capex_eur_per_kw':450,
   'wind_capex_eur_per_kw':1050,
   'pv_fixed_om_fraction':0.015,
   'wind_fixed_om_fraction':0.030,
   'electrolyzer_capex_eur_per_kw':450,
   'electrolyzer_fixed_om_fraction':0.030,
   'asset_lifetime_years':25,
   'electrolyzer_lifetime_years':20,
   'specific_electricity_kwh_per_kg_h2':ELECTROLYZER_KWH_PER_KG,
   'water_l_per_kg_h2':21,
   'water_cost_eur_per_m3':1.25,
   'water_transport_eur_per_100km_m3':0.10
 },
 'infrastructure_adders':{
   'road_transport_to_port_eur_per_kg_per_km':0.0009,
   'water_pipeline_eur_per_kg_per_km':0.000021,
   'grid_connection_eur_per_kg_per_km':0.00035,
   'road_access_eur_per_kg_per_km':0.00012
 },
 'carrier_chain':{
   'ammonia_synthesis_base_eur_per_kg_h2':0.58,
   'ammonia_synthesis_electricity_kwh_per_kg_h2':2.809,
   'ocean_shipping_distance_km':8500,
   'ammonia_shipping_eur_per_kg_h2_per_1000km':0.055,
   'port_handling_eur_per_kg_h2':0.12,
   'reconversion_base_eur_per_kg_h2':1.10,
   'ammonia_cracking_heat_kwh_per_kg_h2':4.2,
   'europe_heat_or_power_eur_per_kwh':0.055
 },
 'europe_comparator':{
   'pv_cf':0.18,
   'wind_cf':0.34,
   'mix_pv_weight':0.35,
   'mix_wind_weight':0.65,
   'renewable_lcoe_policy_discount_eur_per_kwh':0.0,
   'infrastructure_eur_per_kg':0.18,
   'water_eur_per_kg':0.03
 },
 'scenarios':{
   'baseline_2030':{'africa_wacc':0.08,'europe_wacc':0.045,'policy_credit_eur_per_kg':0.0,'description':'Central 2030 assumptions: moderate African risk premium and low European financing.'},
   'high_africa_risk':{'africa_wacc':0.12,'europe_wacc':0.045,'policy_credit_eur_per_kg':0.0,'description':'Country/investment risk keeps African project WACC high.'},
   'derisked_africa':{'africa_wacc':0.055,'europe_wacc':0.045,'policy_credit_eur_per_kg':0.0,'description':'Concessional/de-risked finance reduces African WACC close to Europe.'},
   'high_global_rates':{'africa_wacc':0.13,'europe_wacc':0.075,'policy_credit_eur_per_kg':0.0,'description':'Global interest-rate repricing raises financing costs in both regions.'},
   'eu_import_policy_credit':{'africa_wacc':0.08,'europe_wacc':0.045,'policy_credit_eur_per_kg':0.75,'description':'Contracts-for-difference/import credit applied to delivered African H2.'},
   'derisking_plus_policy':{'africa_wacc':0.055,'europe_wacc':0.045,'policy_credit_eur_per_kg':0.75,'description':'Low African WACC combined with EU import support.'}
 },
 'related_work_validation_anchors':{
   'Kenya_current_production_range_eur_per_kg':[3.7,9.9],
   'Kenya_2030_production_range_eur_per_kg':[1.8,3.0],
   'Kenya_export_to_Rotterdam_current_eur_per_kg':7.0,
   'Namibia_current_delivered_to_port_range_eur_per_kg':[5.43,9.21],
   'IEA_Namibia_2030_low_end_usd_per_kg':2.50,
   'GeoH2_reference_interest_rate':0.06,
   'Germany_high_rate_LCOE_increase_pct':{'solar_pv':11,'onshore_wind':25}
 }
}

def crf(r,n):
    if r == 0: return 1/n
    return r*(1+r)**n/((1+r)**n-1)

def renewable_lcoe(capex_kw, fixed_om_frac, cf, wacc, life=25):
    cf=np.maximum(np.asarray(cf,dtype=float),0.05)
    return (capex_kw*crf(wacc,life)+fixed_om_frac*capex_kw)/(cf*HOURS)

def h2_production_cost(row, wacc):
    # Input potentials are high simulated annual potentials; interpret as
    # technology quality indices and map to plausible CFs.
    pv_cf=np.clip(0.18 + 0.22*(row['theo_pv']-0.58)/(0.85-0.58),0.16,0.42)
    wind_cf=np.clip(0.22 + 0.33*(row['theo_wind']-0.29)/(0.75-0.29),0.18,0.58)
    pv_lcoe=renewable_lcoe(ASSUMPTIONS['technology']['pv_capex_eur_per_kw'], ASSUMPTIONS['technology']['pv_fixed_om_fraction'], pv_cf, wacc)
    wind_lcoe=renewable_lcoe(ASSUMPTIONS['technology']['wind_capex_eur_per_kw'], ASSUMPTIONS['technology']['wind_fixed_om_fraction'], wind_cf, wacc)
    # Site-specific mix tilts toward cheaper resource while retaining diversity.
    inv=np.array([1/pv_lcoe,1/wind_lcoe],dtype=float)
    weights=inv/inv.sum()
    elec_cost=float(weights[0]*pv_lcoe + weights[1]*wind_lcoe)
    effective_cf=float(weights[0]*pv_cf + weights[1]*wind_cf)
    el=ASSUMPTIONS['technology']['electrolyzer_capex_eur_per_kw']
    el_om=ASSUMPTIONS['technology']['electrolyzer_fixed_om_fraction']
    annual_kg_per_kw=effective_cf*HOURS/ELECTROLYZER_KWH_PER_KG
    electrolyzer_cost=(el*crf(wacc,ASSUMPTIONS['technology']['electrolyzer_lifetime_years']) + el_om*el)/annual_kg_per_kw
    water=(ASSUMPTIONS['technology']['water_l_per_kg_h2']/1000)*ASSUMPTIONS['technology']['water_cost_eur_per_m3']
    water += (ASSUMPTIONS['technology']['water_l_per_kg_h2']/1000)*(row['waterbody_dist_km']/100)*ASSUMPTIONS['technology']['water_transport_eur_per_100km_m3']
    infra=row['road_dist_km']*ASSUMPTIONS['infrastructure_adders']['road_access_eur_per_kg_per_km']
    grid=row['grid_dist_km']*ASSUMPTIONS['infrastructure_adders']['grid_connection_eur_per_kg_per_km']
    prod=elec_cost*ELECTROLYZER_KWH_PER_KG + electrolyzer_cost + water + infra + grid
    return dict(pv_cf=pv_cf,wind_cf=wind_cf,pv_lcoe_eur_kwh=pv_lcoe,wind_lcoe_eur_kwh=wind_lcoe,renewable_lcoe_eur_kwh=elec_cost,effective_cf=effective_cf,electrolyzer_cost_eur_kg=electrolyzer_cost,water_cost_eur_kg=water,infra_access_eur_kg=infra,grid_connection_eur_kg=grid,production_lcoh_eur_kg=prod,renewable_mix_pv_weight=weights[0],renewable_mix_wind_weight=weights[1])

def delivered_adders(row, prod_elec_cost, scenario):
    synth=ASSUMPTIONS['carrier_chain']['ammonia_synthesis_base_eur_per_kg_h2'] + ASSUMPTIONS['carrier_chain']['ammonia_synthesis_electricity_kwh_per_kg_h2']*prod_elec_cost
    road_port=row['ocean_dist_km']*ASSUMPTIONS['infrastructure_adders']['road_transport_to_port_eur_per_kg_per_km']
    shipping=ASSUMPTIONS['carrier_chain']['port_handling_eur_per_kg_h2'] + ASSUMPTIONS['carrier_chain']['ocean_shipping_distance_km']/1000*ASSUMPTIONS['carrier_chain']['ammonia_shipping_eur_per_kg_h2_per_1000km']
    reconv=ASSUMPTIONS['carrier_chain']['reconversion_base_eur_per_kg_h2'] + ASSUMPTIONS['carrier_chain']['ammonia_cracking_heat_kwh_per_kg_h2']*ASSUMPTIONS['carrier_chain']['europe_heat_or_power_eur_per_kwh']
    return synth, road_port, shipping, reconv, scenario['policy_credit_eur_per_kg']

def europe_cost(wacc, policy_discount=0):
    e=ASSUMPTIONS['europe_comparator']
    pv_lcoe=renewable_lcoe(ASSUMPTIONS['technology']['pv_capex_eur_per_kw'],ASSUMPTIONS['technology']['pv_fixed_om_fraction'],e['pv_cf'],wacc)
    wind_lcoe=renewable_lcoe(ASSUMPTIONS['technology']['wind_capex_eur_per_kw'],ASSUMPTIONS['technology']['wind_fixed_om_fraction'],e['wind_cf'],wacc)
    elec=max(0.0, e['mix_pv_weight']*pv_lcoe+e['mix_wind_weight']*wind_lcoe-policy_discount)
    cf=e['mix_pv_weight']*e['pv_cf']+e['mix_wind_weight']*e['wind_cf']
    el=ASSUMPTIONS['technology']['electrolyzer_capex_eur_per_kw']; el_om=ASSUMPTIONS['technology']['electrolyzer_fixed_om_fraction']
    annual_kg=cf*HOURS/ELECTROLYZER_KWH_PER_KG
    electrolyzer=(el*crf(wacc,ASSUMPTIONS['technology']['electrolyzer_lifetime_years'])+el_om*el)/annual_kg
    return {'europe_wacc':wacc,'europe_pv_lcoe_eur_kwh':pv_lcoe,'europe_wind_lcoe_eur_kwh':wind_lcoe,'europe_renewable_lcoe_eur_kwh':elec,'europe_effective_cf':cf,'europe_lcoh_eur_kg':elec*ELECTROLYZER_KWH_PER_KG+electrolyzer+e['water_eur_per_kg']+e['infrastructure_eur_per_kg']}

def assign_country(df):
    df=df.copy(); df['country']='Unassigned'
    if HAS_GPD:
        try:
            gdf=gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,df.lat), crs='EPSG:4326')
            shp=gpd.read_file(DATA/'africa_map'/'ne_10m_admin_0_countries.shp')
            # identify name column
            cols=list(shp.columns)
            name_col='ADMIN' if 'ADMIN' in cols else ('NAME' if 'NAME' in cols else cols[0])
            joined=gpd.sjoin(gdf, shp[[name_col,'geometry']], how='left', predicate='within')
            df['country']=joined[name_col].fillna('Unassigned').values
        except Exception as e:
            warnings.warn(f'geopandas country join failed: {e}')
    # fallback for Namibia/Botswana region; all points appear southern Africa
    df.loc[df['country'].eq('Unassigned') & (df['lon']<14.5),'country']='Namibia'
    df.loc[df['country'].eq('Unassigned') & (df['lon']>=14.5) & (df['lon']<21.0),'country']='Namibia'
    df.loc[df['country'].eq('Unassigned') & (df['lon']>=21.0),'country']='Botswana'
    return df

def main():
    df=pd.read_csv(DATA/'hex_final_NA_min.csv')
    df=assign_country(df)
    # Save overview stats
    df.describe(include='all').to_csv(OUT/'data_overview_statistics.csv')
    all_rows=[]; eu_rows=[]
    for sc_name,sc in ASSUMPTIONS['scenarios'].items():
        eu=europe_cost(sc['europe_wacc']); eu['scenario']=sc_name; eu['description']=sc['description']; eu_rows.append(eu)
        for _,row in df.iterrows():
            comps=h2_production_cost(row, sc['africa_wacc'])
            synth,road_port,shipping,reconv,credit=delivered_adders(row, comps['renewable_lcoe_eur_kwh'], sc)
            delivered=comps['production_lcoh_eur_kg']+synth+road_port+shipping+reconv-credit
            rec={**row.to_dict(), **comps, 'scenario':sc_name, 'africa_wacc':sc['africa_wacc'], 'europe_wacc':sc['europe_wacc'], 'ammonia_synthesis_eur_kg':synth, 'road_to_port_eur_kg':road_port, 'ocean_shipping_eur_kg':shipping, 'reconversion_eur_kg':reconv, 'policy_credit_eur_kg':credit, 'delivered_cost_eur_kg':delivered, 'europe_lcoh_eur_kg':eu['europe_lcoh_eur_kg'], 'competitive_vs_europe':delivered <= eu['europe_lcoh_eur_kg'], 'cost_gap_vs_europe_eur_kg':delivered-eu['europe_lcoh_eur_kg']}
            all_rows.append(rec)
    res=pd.DataFrame(all_rows)
    eu_df=pd.DataFrame(eu_rows)
    res.to_csv(OUT/'site_costs_by_scenario.csv',index=False)
    eu_df.to_csv(OUT/'europe_comparator.csv',index=False)
    summary=res.groupby('scenario').agg(
        n_sites=('hex_id','count'), mean_delivered=('delivered_cost_eur_kg','mean'), median_delivered=('delivered_cost_eur_kg','median'), min_delivered=('delivered_cost_eur_kg','min'), p10_delivered=('delivered_cost_eur_kg',lambda x:x.quantile(.1)), p90_delivered=('delivered_cost_eur_kg',lambda x:x.quantile(.9)), max_delivered=('delivered_cost_eur_kg','max'), mean_production=('production_lcoh_eur_kg','mean'), min_production=('production_lcoh_eur_kg','min'), competitive_sites=('competitive_vs_europe','sum'), competitive_share=('competitive_vs_europe','mean'), europe_lcoh=('europe_lcoh_eur_kg','first'), best_gap=('cost_gap_vs_europe_eur_kg','min')
    ).reset_index()
    order=list(ASSUMPTIONS['scenarios'].keys())
    summary['scenario']=pd.Categorical(summary['scenario'],order,ordered=True); summary=summary.sort_values('scenario')
    summary.to_csv(OUT/'scenario_summary.csv',index=False)
    country=res.groupby(['scenario','country']).agg(n_sites=('hex_id','count'), min_delivered=('delivered_cost_eur_kg','min'), median_delivered=('delivered_cost_eur_kg','median'), competitive_sites=('competitive_vs_europe','sum'), competitive_share=('competitive_vs_europe','mean')).reset_index()
    country.to_csv(OUT/'country_summary.csv',index=False)
    # Sensitivity: best site delivered cost vs WACC
    sens=[]
    for w in np.linspace(0.03,0.15,25):
        vals=[]
        for _,row in df.iterrows():
            comps=h2_production_cost(row,w); synth,road,ship,rec,cred=delivered_adders(row, comps['renewable_lcoe_eur_kwh'], {'policy_credit_eur_per_kg':0})
            vals.append(comps['production_lcoh_eur_kg']+synth+road+ship+rec)
        sens.append({'africa_wacc':w,'best_delivered_eur_kg':min(vals),'median_delivered_eur_kg':float(np.median(vals)),'best_production_eur_kg':min([h2_production_cost(r,w)['production_lcoh_eur_kg'] for _,r in df.iterrows()])})
    sens=pd.DataFrame(sens); sens.to_csv(OUT/'finance_sensitivity.csv',index=False)
    # figure data exports
    res[res.scenario=='baseline_2030'].to_csv(OUT/'figure_data_map_baseline.csv',index=False)
    summary.to_csv(OUT/'figure_data_scenario_summary.csv',index=False)
    country.to_csv(OUT/'figure_data_country_summary.csv',index=False)
    sens.to_csv(OUT/'figure_data_finance_sensitivity.csv',index=False)
    # Plots
    fig,axs=plt.subplots(1,2,figsize=(10,4),dpi=180)
    axs[0].scatter(df['theo_pv'],df['theo_wind'],c=df['ocean_dist_km'],cmap='viridis',s=55,edgecolor='k',linewidth=.3)
    axs[0].set_xlabel('PV potential index'); axs[0].set_ylabel('Wind potential index'); axs[0].set_title('Resource quality by site')
    cb=fig.colorbar(axs[0].collections[0],ax=axs[0]); cb.set_label('Distance to ocean (km)')
    sns.boxplot(data=df[['grid_dist_km','road_dist_km','ocean_dist_km','waterbody_dist_km']], ax=axs[1], color='#9ecae1')
    axs[1].tick_params(axis='x',rotation=35); axs[1].set_ylabel('Distance (km)'); axs[1].set_title('Infrastructure access distances')
    fig.tight_layout(); fig.savefig(IMG/'data_overview.png'); plt.close(fig)
    base=res[res.scenario=='baseline_2030']
    fig,ax=plt.subplots(figsize=(6.2,6),dpi=180)
    if HAS_GPD:
        try:
            shp=gpd.read_file(DATA/'africa_map'/'ne_10m_admin_0_countries.shp')
            shp.cx[8:28,-31:-15].plot(ax=ax,color='#f2f2f2',edgecolor='white',linewidth=.4)
        except Exception: pass
    sc=ax.scatter(base.lon,base.lat,c=base.delivered_cost_eur_kg,s=95,cmap='magma_r',edgecolor='k',linewidth=.35)
    for _,r in base.nsmallest(5,'delivered_cost_eur_kg').iterrows():
        ax.text(r.lon+0.12,r.lat+0.12,r.hex_id.replace('hex_',''),fontsize=6)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('Baseline delivered H2 cost to Europe (€/kg)')
    cb=fig.colorbar(sc,ax=ax); cb.set_label('€/kg H2 delivered')
    ax.set_aspect('equal', adjustable='datalim'); fig.tight_layout(); fig.savefig(IMG/'map_baseline_delivered_cost.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8),dpi=180)
    res['scenario_label']=res['scenario'].map({k:k.replace('_','\n') for k in order})
    sns.boxplot(data=res,x='scenario_label',y='delivered_cost_eur_kg',ax=ax,color='#bdbdbd')
    sns.stripplot(data=res,x='scenario_label',y='delivered_cost_eur_kg',ax=ax,hue='competitive_vs_europe',palette={True:'#2ca25f',False:'#de2d26'},size=3,alpha=.8,dodge=False)
    eu_map=eu_df.set_index('scenario')['europe_lcoh_eur_kg'].to_dict()
    for i,s in enumerate(order): ax.hlines(eu_map[s], i-.42, i+.42, colors='#08519c', linestyles='--', linewidth=1.4)
    ax.set_ylabel('Delivered African H2 cost (€/kg)'); ax.set_xlabel('Scenario'); ax.set_title('Financing and policy scenarios: delivered cost distributions')
    ax.legend(title='Competitive vs Europe',loc='upper right'); fig.tight_layout(); fig.savefig(IMG/'scenario_cost_distributions.png'); plt.close(fig)
    # country competitiveness
    cplot=country[country.scenario.isin(['baseline_2030','derisked_africa','derisking_plus_policy'])].copy()
    cplot['scenario_label']=cplot['scenario'].str.replace('_',' ')
    fig,ax=plt.subplots(figsize=(7,4.5),dpi=180)
    sns.barplot(data=cplot,x='country',y='competitive_share',hue='scenario_label',ax=ax)
    ax.set_ylim(0,1); ax.set_ylabel('Share of sites competitive vs Europe'); ax.set_xlabel('Country assigned from shapefile/fallback'); ax.set_title('Competitive site share by country')
    fig.tight_layout(); fig.savefig(IMG/'competitiveness_by_country.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.5),dpi=180)
    ax.plot(sens.africa_wacc*100,sens.best_delivered_eur_kg,label='Best African delivered cost',color='#238b45')
    ax.plot(sens.africa_wacc*100,sens.median_delivered_eur_kg,label='Median African delivered cost',color='#74c476')
    for scn,col in [('baseline_2030','#756bb1'),('derisked_africa','#31a354'),('high_africa_risk','#de2d26')]:
        row=summary[summary.scenario.astype(str)==scn].iloc[0]
        ax.scatter(ASSUMPTIONS['scenarios'][scn]['africa_wacc']*100,row['min_delivered'],color=col,s=55,zorder=5,label=scn.replace('_',' '))
    ax.set_xlabel('African project WACC (%)'); ax.set_ylabel('Delivered cost (€/kg H2)'); ax.set_title('Finance sensitivity of import costs'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(IMG/'finance_sensitivity.png'); plt.close(fig)
    # validation/comparison
    val_rows=[]
    anchors=ASSUMPTIONS['related_work_validation_anchors']
    val_rows += [{'label':'This model best production\nbaseline','low':summary.loc[summary.scenario.astype(str)=='baseline_2030','min_production'].iloc[0],'high':summary.loc[summary.scenario.astype(str)=='baseline_2030','min_production'].iloc[0],'kind':'model'}]
    val_rows += [{'label':'This model best production\nderisked','low':summary.loc[summary.scenario.astype(str)=='derisked_africa','min_production'].iloc[0],'high':summary.loc[summary.scenario.astype(str)=='derisked_africa','min_production'].iloc[0],'kind':'model'}]
    val_rows += [{'label':'Kenya 2030 production\nMüller et al.','low':1.8,'high':3.0,'kind':'literature'}]
    val_rows += [{'label':'Kenya current production\nMüller et al.','low':3.7,'high':9.9,'kind':'literature'}]
    val_rows += [{'label':'This model best delivered\nbaseline','low':summary.loc[summary.scenario.astype(str)=='baseline_2030','min_delivered'].iloc[0],'high':summary.loc[summary.scenario.astype(str)=='baseline_2030','min_delivered'].iloc[0],'kind':'model'}]
    val_rows += [{'label':'Kenya export Rotterdam\nMüller et al.','low':7.0,'high':7.0,'kind':'literature'}]
    val=pd.DataFrame(val_rows); val.to_csv(OUT/'figure_data_validation_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.8),dpi=180)
    y=np.arange(len(val))
    colors=val['kind'].map({'model':'#3182bd','literature':'#969696'})
    for i,r in val.iterrows():
        ax.plot([r.low,r.high],[i,i],lw=7,solid_capstyle='round',color=colors.iloc[i])
        if abs(r.high-r.low)<1e-6: ax.scatter(r.low,i,s=65,color=colors.iloc[i],edgecolor='k',zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(val.label); ax.set_xlabel('Cost (€/kg H2)'); ax.set_title('Validation against related-work cost anchors')
    ax.grid(axis='y',visible=False); fig.tight_layout(); fig.savefig(IMG/'validation_comparison.png'); plt.close(fig)
    # cost stack for best baseline site
    best=base.nsmallest(1,'delivered_cost_eur_kg').iloc[0]
    stack=pd.DataFrame([{'component':'Production','eur_per_kg':best.production_lcoh_eur_kg},{'component':'Ammonia synthesis','eur_per_kg':best.ammonia_synthesis_eur_kg},{'component':'Road to port','eur_per_kg':best.road_to_port_eur_kg},{'component':'Ocean shipping/handling','eur_per_kg':best.ocean_shipping_eur_kg},{'component':'Reconversion','eur_per_kg':best.reconversion_eur_kg}])
    stack.to_csv(OUT/'figure_data_best_site_cost_stack.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,3.8),dpi=180)
    sns.barplot(data=stack,x='component',y='eur_per_kg',ax=ax,color='#6baed6')
    ax.tick_params(axis='x',rotation=30); ax.set_ylabel('€/kg H2'); ax.set_title(f'Baseline cost stack for best site ({best.hex_id})')
    fig.tight_layout(); fig.savefig(IMG/'best_site_cost_stack.png'); plt.close(fig)
    # Assumptions and claims
    with open(OUT/'assumptions.json','w') as f: json.dump(ASSUMPTIONS,f,indent=2)
    claims=[]
    def get(s,col): return float(summary.loc[summary.scenario.astype(str)==s,col].iloc[0])
    claims.append({'claim':'Baseline best African delivered hydrogen cost to Europe via ammonia/reconversion','value':get('baseline_2030','min_delivered'),'unit':'EUR/kg H2','supporting_artifacts':'outputs/scenario_summary.csv; outputs/site_costs_by_scenario.csv; report/images/map_baseline_delivered_cost.png'})
    claims.append({'claim':'De-risking lowers the best delivered cost relative to baseline','value':get('baseline_2030','min_delivered')-get('derisked_africa','min_delivered'),'unit':'EUR/kg H2 reduction','supporting_artifacts':'outputs/scenario_summary.csv; report/images/scenario_cost_distributions.png; report/images/finance_sensitivity.png'})
    claims.append({'claim':'High African risk raises the best delivered cost relative to baseline','value':get('high_africa_risk','min_delivered')-get('baseline_2030','min_delivered'),'unit':'EUR/kg H2 increase','supporting_artifacts':'outputs/scenario_summary.csv; report/images/finance_sensitivity.png'})
    claims.append({'claim':'EU import policy credit plus de-risking maximizes competitive site share','value':get('derisking_plus_policy','competitive_share'),'unit':'share of sites competitive vs Europe','supporting_artifacts':'outputs/scenario_summary.csv; outputs/country_summary.csv; report/images/competitiveness_by_country.png'})
    claims.append({'claim':'Best modeled 2030 production costs are within/near Kenya 2030 related-work production range','value':get('derisked_africa','min_production'),'unit':'EUR/kg H2 production','supporting_artifacts':'outputs/figure_data_validation_comparison.csv; report/images/validation_comparison.png; outputs/related_work_contract.json'})
    pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv',index=False)
    # update inventory status
    inventory=json.load(open(OUT/'target_artifact_inventory.json'))
    for item in inventory['primary_artifacts']:
        p=str(item['path'])
        if '*' in p:
            item['status']='satisfied' if list(OUT.glob('figure_data_*')) else 'unsatisfied'
        else:
            item['status']='satisfied' if (BASE/p).exists() else 'unsatisfied'
    for item in inventory['figure_families']:
        item['status']='satisfied' if (BASE/item['path']).exists() else 'unsatisfied'
    inventory['additional_figures']=[{'name':'best_site_cost_stack','path':'report/images/best_site_cost_stack.png','status':'satisfied'}]
    inventory['unsatisfied']=[]
    with open(OUT/'target_artifact_inventory.json','w') as f: json.dump(inventory,f,indent=2)
    print(json.dumps({'n_sites':len(df),'scenarios':order,'summary':summary.to_dict(orient='records'),'best_baseline_site':best[['hex_id','country','lat','lon','delivered_cost_eur_kg','production_lcoh_eur_kg','cost_gap_vs_europe_eur_kg']].to_dict()},indent=2,default=str))

if __name__=='__main__':
    main()
