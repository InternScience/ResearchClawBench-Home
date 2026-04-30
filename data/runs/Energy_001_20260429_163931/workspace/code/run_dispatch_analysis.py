#!/usr/bin/env python3
"""Reproducible GB-style nodal dispatch analysis for the ResearchClawBench Energy task.

Model: linear transport economic dispatch over all hourly snapshots. Decision variables
include generator output, link flows, storage charge/discharge/state of charge,
curtailed wind, and high-penalty load shedding.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

LOAD_SHED_COST = 10000.0  # GBP/MWh, deliberately above all generation costs
CHARGE_EPS_COST = 0.01    # tiny cost to avoid pointless simultaneous cycling

SCENARIOS = {
    'baseline_2025': {
        'description': 'Input capacities and demand as supplied.',
        'demand_mult': 1.0, 'wind_mult': 1.0, 'gas_mult': 1.0,
        'nuclear_mult': 1.0, 'storage_power_mult': 1.0, 'storage_energy_mult': 1.0,
        'link_mult': 1.0,
    },
    'high_renewable_2035': {
        'description': 'Electrification increases demand while wind and storage expand.',
        'demand_mult': 1.12, 'wind_mult': 1.8, 'gas_mult': 0.85,
        'nuclear_mult': 1.0, 'storage_power_mult': 2.0, 'storage_energy_mult': 3.0,
        'link_mult': 1.25,
    },
    'net_zero_2050': {
        'description': 'Stronger electrification, large wind buildout, reduced gas fleet, expanded storage and grid.',
        'demand_mult': 1.30, 'wind_mult': 2.8, 'gas_mult': 0.55,
        'nuclear_mult': 1.10, 'storage_power_mult': 4.0, 'storage_energy_mult': 6.0,
        'link_mult': 1.6,
    },
}


def read_inputs():
    buses = pd.read_csv(DATA / 'buses.csv')
    links = pd.read_csv(DATA / 'links.csv')
    demand = pd.read_csv(DATA / 'demand.csv')
    gens = pd.read_csv(DATA / 'generators.csv')
    wind_cf = pd.read_csv(DATA / 'wind_cf.csv')
    storage = pd.read_csv(DATA / 'storage.csv')
    return buses, links, demand, gens, wind_cf, storage


def solve_scenario(name: str, sc: dict, inputs: tuple):
    buses, links0, demand0, gens0, wind_cf, storage0 = inputs
    bus_names = list(buses['name'])
    B = len(bus_names)
    T = len(demand0)
    L = len(links0)
    S = len(storage0)
    G = len(gens0)
    bus_index = {b: i for i, b in enumerate(bus_names)}

    demand = demand0[bus_names].to_numpy(float) * sc['demand_mult']
    gens = gens0.copy()
    # Capacity multipliers by carrier
    mult = []
    for c in gens['carrier']:
        if 'wind' in c:
            mult.append(sc['wind_mult'])
        elif c == 'gas':
            mult.append(sc['gas_mult'])
        elif c == 'nuclear':
            mult.append(sc['nuclear_mult'])
        else:
            mult.append(1.0)
    gens['p_nom_s'] = gens['p_nom'].to_numpy(float) * np.array(mult)
    links = links0.copy()
    links['p_nom_s'] = links['p_nom'].to_numpy(float) * sc['link_mult']
    storage = storage0.copy()
    storage['p_nom_s'] = storage['p_nom'].to_numpy(float) * sc['storage_power_mult']
    storage['e_nom_s'] = storage['e_nom'].to_numpy(float) * sc['storage_energy_mult']

    # Variable blocks flattened by hour then component.
    n_pg = T * G
    n_flow = T * L
    n_ch = T * S
    n_dis = T * S
    n_soc = T * S
    n_shed = T * B
    off_pg = 0
    off_flow = off_pg + n_pg
    off_ch = off_flow + n_flow
    off_dis = off_ch + n_ch
    off_soc = off_dis + n_dis
    off_shed = off_soc + n_soc
    N = off_shed + n_shed

    def idx_pg(t,g): return off_pg + t*G + g
    def idx_flow(t,l): return off_flow + t*L + l
    def idx_ch(t,s): return off_ch + t*S + s
    def idx_dis(t,s): return off_dis + t*S + s
    def idx_soc(t,s): return off_soc + t*S + s
    def idx_shed(t,b): return off_shed + t*B + b

    c = np.zeros(N)
    bounds = [(0, None)] * N

    # Generator bounds and costs. Wind availability depends on local CF.
    for t in range(T):
        for g, row in gens.iterrows():
            cap = float(row['p_nom_s'])
            if 'wind' in row['carrier']:
                cap *= float(wind_cf.loc[t, row['bus']])
            bounds[idx_pg(t,g)] = (0, cap)
            c[idx_pg(t,g)] = float(row['marginal_cost'])
        for l, row in links.iterrows():
            cap = float(row['p_nom_s'])
            bounds[idx_flow(t,l)] = (-cap, cap)
        for s, row in storage.iterrows():
            bounds[idx_ch(t,s)] = (0, float(row['p_nom_s']))
            bounds[idx_dis(t,s)] = (0, float(row['p_nom_s']))
            bounds[idx_soc(t,s)] = (0, float(row['e_nom_s']))
            c[idx_ch(t,s)] = CHARGE_EPS_COST
        for b in range(B):
            bounds[idx_shed(t,b)] = (0, demand[t,b])
            c[idx_shed(t,b)] = LOAD_SHED_COST

    # Nodal balances and storage dynamics as sparse COO triplets.
    rows = []
    cols = []
    vals = []
    beq = []
    r = 0
    gen_at_bus = {b: [] for b in bus_names}
    for g, grow in gens.iterrows():
        gen_at_bus[grow['bus']].append(g)
    store_at_bus = {b: [] for b in bus_names}
    for s_i, srow in storage.iterrows():
        store_at_bus[srow['bus']].append(s_i)
    links_at_bus = {b: [] for b in bus_names}
    for l, lrow in links.iterrows():
        links_at_bus[lrow['bus0']].append((l, -1.0))
        links_at_bus[lrow['bus1']].append((l, 1.0))

    for t in range(T):
        for bname in bus_names:
            b = bus_index[bname]
            for g in gen_at_bus[bname]:
                rows.append(r); cols.append(idx_pg(t,g)); vals.append(1.0)
            for l, sign in links_at_bus[bname]:
                rows.append(r); cols.append(idx_flow(t,l)); vals.append(sign)
            for s_i in store_at_bus[bname]:
                rows.append(r); cols.append(idx_dis(t,s_i)); vals.append(1.0)
                rows.append(r); cols.append(idx_ch(t,s_i)); vals.append(-1.0)
            rows.append(r); cols.append(idx_shed(t,b)); vals.append(1.0)
            beq.append(demand[t,b])
            r += 1

    # Storage dynamics with cyclic terminal condition. eta split as sqrt round-trip.
    for t in range(T):
        prev = T-1 if t == 0 else t-1
        for s_i, srow in storage.iterrows():
            eta = float(srow['efficiency']) ** 0.5
            for col, val in [(idx_soc(t,s_i), 1.0), (idx_soc(prev,s_i), -1.0), (idx_ch(t,s_i), -eta), (idx_dis(t,s_i), 1.0/eta)]:
                rows.append(r); cols.append(col); vals.append(val)
            beq.append(0.0)
            r += 1

    from scipy.sparse import coo_matrix
    Aeq = coo_matrix((vals, (rows, cols)), shape=(r, N), dtype=float).tocsr()
    beq = np.array(beq)

    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs-ds', options={'presolve': True})
    if not res.success:
        raise RuntimeError(f"{name} optimisation failed: {res.message}")
    x = res.x

    # Extract results
    pg = np.array([[x[idx_pg(t,g)] for g in range(G)] for t in range(T)])
    flows = np.array([[x[idx_flow(t,l)] for l in range(L)] for t in range(T)])
    ch = np.array([[x[idx_ch(t,s)] for s in range(S)] for t in range(T)]) if S else np.zeros((T,0))
    dis = np.array([[x[idx_dis(t,s)] for s in range(S)] for t in range(T)]) if S else np.zeros((T,0))
    soc = np.array([[x[idx_soc(t,s)] for s in range(S)] for t in range(T)]) if S else np.zeros((T,0))
    shed = np.array([[x[idx_shed(t,b)] for b in range(B)] for t in range(T)])

    wind_avail = np.zeros((T,G))
    for t in range(T):
        for g, row in gens.iterrows():
            cap = float(row['p_nom_s'])
            if 'wind' in row['carrier']:
                cap *= float(wind_cf.loc[t, row['bus']])
            wind_avail[t,g] = cap if 'wind' in row['carrier'] else np.nan
    curtailed_by_g = np.nan_to_num(wind_avail - np.where(np.isnan(wind_avail), np.nan, pg), nan=0.0)

    gen_rows=[]
    for carrier, idxs in gens.groupby('carrier').groups.items():
        idxs=list(idxs)
        gen_rows.append({
            'scenario': name, 'carrier': carrier,
            'generation_MWh': pg[:,idxs].sum(),
            'capacity_MW': gens.loc[idxs,'p_nom_s'].sum(),
            'variable_cost_GBP': (pg[:,idxs] * gens.loc[idxs,'marginal_cost'].to_numpy()).sum(),
            'curtailment_MWh': curtailed_by_g[:,idxs].sum() if 'wind' in carrier else 0.0,
            'available_MWh': np.nan_to_num(wind_avail[:,idxs], nan=0.0).sum() if 'wind' in carrier else np.nan,
        })

    hourly=[]
    for t in range(T):
        rec={'scenario': name, 'hour': t, 'demand_MWh': demand[t].sum(), 'load_shed_MWh': shed[t].sum(),
             'storage_charge_MWh': ch[t].sum(), 'storage_discharge_MWh': dis[t].sum(), 'wind_curtailment_MWh': curtailed_by_g[t].sum()}
        for carrier, idxs in gens.groupby('carrier').groups.items():
            rec[f'{carrier}_MWh'] = pg[t,list(idxs)].sum()
        hourly.append(rec)

    line_rows=[]
    for l, row in links.iterrows():
        cap=row['p_nom_s']
        absf=np.abs(flows[:,l])
        line_rows.append({'scenario':name, 'line':f"{row['bus0']}-{row['bus1']}", 'bus0':row['bus0'], 'bus1':row['bus1'],
                          'capacity_MW':cap, 'mean_abs_flow_MW':absf.mean(), 'max_abs_flow_MW':absf.max(),
                          'utilisation_max_pct':100*absf.max()/cap if cap else np.nan,
                          'congested_hours_95pct':int((absf >= 0.95*cap - 1e-6).sum()),
                          'net_flow_MWh_bus0_to_bus1':flows[:,l].sum()})

    store_rows=[]
    for t in range(T):
        for s, row in storage.iterrows():
            store_rows.append({'scenario':name, 'hour':t, 'storage':f"{row['carrier']}_{s}_{row['bus']}", 'bus':row['bus'],
                               'charge_MW':ch[t,s], 'discharge_MW':dis[t,s], 'soc_MWh':soc[t,s]})

    total_cost = float(res.fun)
    gen_cost = sum(r['variable_cost_GBP'] for r in gen_rows)
    shed_cost = shed.sum()*LOAD_SHED_COST
    summary = {
        'scenario': name, 'description': sc['description'], 'status': 'optimal',
        'objective_GBP': total_cost, 'generation_cost_GBP': gen_cost, 'load_shed_penalty_GBP': shed_cost,
        'demand_MWh': demand.sum(), 'served_demand_MWh': demand.sum()-shed.sum(), 'unserved_energy_MWh': shed.sum(),
        'unserved_pct_of_demand': 100*shed.sum()/demand.sum(),
        'generation_MWh': pg.sum(), 'wind_curtailment_MWh': curtailed_by_g.sum(),
        'wind_available_MWh': np.nan_to_num(wind_avail, nan=0.0).sum(),
        'wind_curtailment_pct_of_available': 100*curtailed_by_g.sum()/max(np.nan_to_num(wind_avail, nan=0.0).sum(),1e-9),
        'storage_charge_MWh': ch.sum(), 'storage_discharge_MWh': dis.sum(),
        'max_line_utilisation_pct': max(r['utilisation_max_pct'] for r in line_rows),
        'line_hours_congested_95pct': sum(r['congested_hours_95pct'] for r in line_rows),
        'total_installed_generation_MW': gens['p_nom_s'].sum(), 'total_wind_capacity_MW': gens.loc[gens['carrier'].str.contains('wind'),'p_nom_s'].sum(),
        'total_storage_power_MW': storage['p_nom_s'].sum(), 'total_storage_energy_MWh': storage['e_nom_s'].sum(),
        'link_capacity_multiplier': sc['link_mult'],
    }
    return summary, pd.DataFrame(gen_rows), pd.DataFrame(hourly), pd.DataFrame(line_rows), pd.DataFrame(store_rows), flows, pg, demand


def make_figures(inputs, scenario_summary, gen_by_carrier, hourly, line_util, storage_ts):
    buses, links, demand, gens, wind_cf, storage = inputs
    sns.set_theme(style='whitegrid')

    # Figure 1: data overview (demand, capacities, wind CF distribution)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    total_demand = demand.sum(axis=1)
    axes[0,0].plot(total_demand.index, total_demand.values/1000, color='black')
    axes[0,0].set_title('Hourly GB-wide demand sample')
    axes[0,0].set_xlabel('Hour'); axes[0,0].set_ylabel('Demand (GW)')
    cap = gens.groupby('carrier')['p_nom'].sum().sort_values(ascending=False)/1000
    axes[0,1].bar(cap.index, cap.values, color=['#4daf4a','#e41a1c','#377eb8'][:len(cap)])
    axes[0,1].set_title('Installed generation capacity in input data')
    axes[0,1].set_ylabel('Capacity (GW)'); axes[0,1].tick_params(axis='x', rotation=25)
    axes[1,0].hist(wind_cf.to_numpy().ravel(), bins=30, color='#4daf4a', alpha=0.8)
    axes[1,0].set_title('Wind capacity factor distribution')
    axes[1,0].set_xlabel('Capacity factor'); axes[1,0].set_ylabel('Bus-hours')
    axes[1,1].scatter(buses['x'], buses['y'], s=demand.mean().reindex(buses['name']).to_numpy()/30, color='#984ea3', alpha=0.7)
    for _, r in links.iterrows():
        a=buses.set_index('name').loc[r['bus0']]; b=buses.set_index('name').loc[r['bus1']]
        axes[1,1].plot([a['x'],b['x']],[a['y'],b['y']], color='gray', lw=1, alpha=0.6)
    axes[1,1].set_title('20-node network and mean demand bubble size')
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
    fig.tight_layout(); fig.savefig(IMG/'figure_1_data_overview.png', dpi=180); plt.close(fig)

    # Figure 2: scenario results
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    gen_pivot = gen_by_carrier.pivot(index='scenario', columns='carrier', values='generation_MWh').reindex(scenario_summary['scenario'])/1000
    gen_pivot.plot(kind='bar', stacked=True, ax=axes[0,0], colormap='Set2')
    axes[0,0].set_title('Generation mix by scenario'); axes[0,0].set_ylabel('GWh over week'); axes[0,0].tick_params(axis='x', rotation=30)
    ss=scenario_summary.set_index('scenario')
    axes[0,1].bar(ss.index, ss['wind_curtailment_MWh']/1000, color='#4daf4a')
    axes[0,1].set_title('Wind curtailment'); axes[0,1].set_ylabel('GWh'); axes[0,1].tick_params(axis='x', rotation=30)
    axes[1,0].bar(ss.index, ss['generation_cost_GBP']/1e6, label='generation cost')
    axes[1,0].bar(ss.index, ss['load_shed_penalty_GBP']/1e6, bottom=ss['generation_cost_GBP']/1e6, label='unserved-energy penalty', color='#e41a1c')
    axes[1,0].set_title('Optimisation objective components'); axes[1,0].set_ylabel('Million GBP'); axes[1,0].tick_params(axis='x', rotation=30); axes[1,0].legend()
    axes[1,1].bar(ss.index, ss['unserved_energy_MWh']/1000, color='#e41a1c')
    axes[1,1].set_title('Unserved energy diagnostic'); axes[1,1].set_ylabel('GWh'); axes[1,1].tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(IMG/'figure_2_scenario_results.png', dpi=180); plt.close(fig)

    # Figure 3: validation/comparison: hourly balance for baseline and validation residuals.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    h=hourly[hourly['scenario']=='baseline_2025'].copy()
    comps=[c for c in ['onshore wind_MWh','nuclear_MWh','gas_MWh','storage_discharge_MWh','load_shed_MWh'] if c in h.columns]
    axes[0].stackplot(h['hour'], [h[c].values/1000 for c in comps], labels=[c.replace('_MWh','') for c in comps], alpha=0.85)
    axes[0].plot(h['hour'], h['demand_MWh']/1000 + h['storage_charge_MWh']/1000, color='black', lw=1.5, label='demand + storage charging')
    axes[0].set_title('Baseline hourly supply balance check'); axes[0].set_ylabel('GW'); axes[0].legend(ncol=3, fontsize=8, loc='upper right')
    h2=hourly.copy()
    gen_cols=[c for c in h2.columns if c.endswith('_MWh') and c not in ['demand_MWh','load_shed_MWh','storage_charge_MWh','storage_discharge_MWh','wind_curtailment_MWh']]
    h2['balance_residual_MWh']=h2[gen_cols].sum(axis=1)+h2['storage_discharge_MWh']+h2['load_shed_MWh']-h2['demand_MWh']-h2['storage_charge_MWh']
    sns.boxplot(data=h2, x='scenario', y='balance_residual_MWh', ax=axes[1])
    axes[1].axhline(0, color='black', lw=1)
    axes[1].set_title('Aggregate hourly energy-balance residuals (should be near zero)')
    axes[1].set_ylabel('MWh'); axes[1].tick_params(axis='x', rotation=25)
    fig.tight_layout(); fig.savefig(IMG/'figure_3_validation.png', dpi=180); plt.close(fig)

    # Figure 4: network map with max line utilisation in net zero scenario
    fig, ax = plt.subplots(figsize=(8, 9))
    bidx=buses.set_index('name')
    scenario='net_zero_2050' if 'net_zero_2050' in set(line_util['scenario']) else line_util['scenario'].iloc[0]
    lu=line_util[line_util['scenario']==scenario].set_index('line')
    norm=plt.Normalize(0, max(100, line_util['utilisation_max_pct'].max()))
    cmap=plt.cm.plasma
    for _, r in links.iterrows():
        a=bidx.loc[r['bus0']]; b=bidx.loc[r['bus1']]
        line=f"{r['bus0']}-{r['bus1']}"
        u=lu.loc[line,'utilisation_max_pct'] if line in lu.index else 0
        ax.plot([a['x'],b['x']],[a['y'],b['y']], color=cmap(norm(u)), lw=1+3*min(u,100)/100, alpha=0.9)
    sc=ax.scatter(buses['x'], buses['y'], s=80, color='white', edgecolor='black', zorder=3)
    for _, r in buses.iterrows():
        ax.text(r['x'], r['y'], r['name'].replace('Bus',''), fontsize=8, ha='center', va='center', zorder=4)
    sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar=fig.colorbar(sm, ax=ax); cbar.set_label('Max line utilisation (%)')
    ax.set_title(f'Network constraints: maximum utilisation in {scenario}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout(); fig.savefig(IMG/'figure_4_network_flows.png', dpi=180); plt.close(fig)


def main():
    inputs = read_inputs()
    all_summary=[]; all_gen=[]; all_hourly=[]; all_lines=[]; all_store=[]
    for name, sc in SCENARIOS.items():
        print(f'Solving {name}...', flush=True)
        summary, gen, hourly, lines, store, flows, pg, demand = solve_scenario(name, sc, inputs)
        all_summary.append(summary); all_gen.append(gen); all_hourly.append(hourly); all_lines.append(lines); all_store.append(store)
    scenario_summary=pd.DataFrame(all_summary)
    gen_by_carrier=pd.concat(all_gen, ignore_index=True)
    hourly=pd.concat(all_hourly, ignore_index=True)
    line_util=pd.concat(all_lines, ignore_index=True)
    storage_ts=pd.concat(all_store, ignore_index=True)

    scenario_summary.to_csv(OUT/'scenario_summary.csv', index=False)
    gen_by_carrier.to_csv(OUT/'generation_by_carrier.csv', index=False)
    hourly.to_csv(OUT/'hourly_dispatch_by_carrier.csv', index=False)
    line_util.to_csv(OUT/'line_utilisation.csv', index=False)
    storage_ts.to_csv(OUT/'storage_timeseries.csv', index=False)

    # Direct constraint and validation outputs
    validation = {
        'scenarios_solved': list(SCENARIOS.keys()),
        'max_unserved_pct': float(scenario_summary['unserved_pct_of_demand'].max()),
        'max_balance_residual_abs_MWh_aggregate': None,
        'all_solver_status_optimal': True,
        'line_constraints_included': True,
        'storage_cyclic_soc': True,
        'load_shed_penalty_GBP_per_MWh': LOAD_SHED_COST,
    }
    h=hourly.copy()
    gen_cols=[c for c in h.columns if c.endswith('_MWh') and c not in ['demand_MWh','load_shed_MWh','storage_charge_MWh','storage_discharge_MWh','wind_curtailment_MWh']]
    h['balance_residual_MWh']=h[gen_cols].sum(axis=1)+h['storage_discharge_MWh']+h['load_shed_MWh']-h['demand_MWh']-h['storage_charge_MWh']
    validation['max_balance_residual_abs_MWh_aggregate']=float(h['balance_residual_MWh'].abs().max())
    with open(OUT/'validation_metrics.json','w') as f: json.dump(validation, f, indent=2)

    # Scenario definitions artifact
    with open(OUT/'scenario_definitions.json','w') as f: json.dump(SCENARIOS, f, indent=2)

    make_figures(inputs, scenario_summary, gen_by_carrier, hourly, line_util, storage_ts)

    # Claim recovery table
    claim_rows = [
        {'claim':'All reported dispatch scenarios were solved to optimality with a linear program.', 'artifact':'outputs/scenario_summary.csv; outputs/validation_metrics.json', 'support':'Solver status optimal for each scenario.'},
        {'claim':'The model preserves hourly and nodal resolution from the input data.', 'artifact':'data/demand.csv; code/run_dispatch_analysis.py; outputs/hourly_dispatch_by_carrier.csv', 'support':'168 hourly snapshots and all buses from buses.csv used in nodal balance constraints.'},
        {'claim':'Wind curtailment and network congestion are scenario-dependent.', 'artifact':'outputs/scenario_summary.csv; outputs/line_utilisation.csv; report/images/figure_2_scenario_results.png; report/images/figure_4_network_flows.png', 'support':'Curtailment totals and line utilisation/congestion hours exported by scenario.'},
        {'claim':'Storage operation is represented with power, energy, efficiency, and cyclic state-of-charge constraints.', 'artifact':'outputs/storage_timeseries.csv; code/run_dispatch_analysis.py', 'support':'Charge, discharge, SOC variables and cyclic equations are included.'},
        {'claim':'FES-specific numerical reproduction is not exact.', 'artifact':'outputs/dependency_check.json; outputs/scenario_definitions.json', 'support':'No FES trajectory file is present; future cases are transparent sensitivity scenarios.'},
    ]
    pd.DataFrame(claim_rows).to_csv(OUT/'claim_recovery_table.csv', index=False)

    # Update artifact inventory statuses after outputs exist
    inv_path=OUT/'target_artifact_inventory.json'
    inv=json.load(open(inv_path))
    for item in inv['required_artifacts']:
        p=ROOT/item['path']
        item['status']='satisfied' if p.exists() else ('unsatisfied' if item['name']=='related_work_contract' else item['status'])
    json.dump(inv, open(inv_path,'w'), indent=2)

    print('Wrote outputs and figures.', flush=True)
    print(scenario_summary[['scenario','demand_MWh','generation_cost_GBP','unserved_energy_MWh','wind_curtailment_MWh','max_line_utilisation_pct']].to_string(index=False))

if __name__ == '__main__':
    main()
