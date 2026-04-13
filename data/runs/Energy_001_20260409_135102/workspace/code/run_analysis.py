
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def load_data():
    buses = pd.read_csv(DATA / 'buses.csv')
    links = pd.read_csv(DATA / 'links.csv')
    demand = pd.read_csv(DATA / 'demand.csv')
    generators = pd.read_csv(DATA / 'generators.csv')
    wind_cf = pd.read_csv(DATA / 'wind_cf.csv')
    storage = pd.read_csv(DATA / 'storage.csv')
    demand.index.name = 'hour'
    wind_cf.index.name = 'hour'
    return buses, links, demand, generators, wind_cf, storage


def build_generator_table(generators, wind_cf, hours):
    rows = []
    for i, row in generators.iterrows():
        if row['carrier'] == 'onshore wind':
            avail = wind_cf[row['bus']].values * row['p_nom']
        else:
            avail = np.full(hours, row['p_nom'])
        rows.append({
            'gen_id': f'g{i}',
            'bus': row['bus'],
            'carrier': row['carrier'],
            'p_nom': float(row['p_nom']),
            'marginal_cost': float(row['marginal_cost']),
            'avail': avail,
        })
    return rows


def solve_dispatch(link_scale=1.0, gas_cost_adder=0.0, curtailment_cost=1.0, unmet_cost=10000.0):
    buses, links, demand, generators, wind_cf, storage = load_data()
    hours = len(demand)
    bus_names = buses['name'].tolist()
    bus_idx = {b: i for i, b in enumerate(bus_names)}
    n_b, n_t = len(bus_names), hours
    gens = build_generator_table(generators, wind_cf, hours)
    n_g = len(gens)
    n_s = len(storage)
    n_l = len(links)

    incidence = np.zeros((n_b, n_l))
    line_caps = []
    for l, row in links.iterrows():
        incidence[bus_idx[row['bus0']], l] = -1
        incidence[bus_idx[row['bus1']], l] = 1
        line_caps.append(float(row['p_nom']) * link_scale)
    line_caps = np.array(line_caps)

    gen_bus = np.zeros((n_b, n_g))
    gen_cost = np.zeros(n_g)
    gen_avail = np.zeros((n_g, n_t))
    for g, row in enumerate(gens):
        gen_bus[bus_idx[row['bus']], g] = 1
        gen_cost[g] = row['marginal_cost'] + (gas_cost_adder if row['carrier'] == 'gas' else 0.0)
        gen_avail[g, :] = row['avail']

    stor_bus = np.zeros((n_b, n_s))
    p_nom = storage['p_nom'].to_numpy(dtype=float)
    e_nom = storage['e_nom'].to_numpy(dtype=float)
    eff = storage['efficiency'].to_numpy(dtype=float)
    for s, row in storage.iterrows():
        stor_bus[bus_idx[row['bus']], s] = 1

    D = demand[bus_names].to_numpy(dtype=float).T

    g = cp.Variable((n_g, n_t), nonneg=True)
    f = cp.Variable((n_l, n_t))
    ch = cp.Variable((n_s, n_t), nonneg=True)
    dis = cp.Variable((n_s, n_t), nonneg=True)
    soc = cp.Variable((n_s, n_t), nonneg=True)
    curt = cp.Variable((n_b, n_t), nonneg=True)
    unmet = cp.Variable((n_b, n_t), nonneg=True)

    constraints = []
    constraints += [g <= gen_avail]
    constraints += [f <= line_caps[:, None], f >= -line_caps[:, None]]
    constraints += [ch <= p_nom[:, None], dis <= p_nom[:, None]]
    constraints += [soc <= e_nom[:, None]]

    for t in range(n_t):
        prev = soc[:, t-1] if t > 0 else 0.5 * e_nom
        constraints += [soc[:, t] == prev + cp.multiply(eff, ch[:, t]) - cp.multiply(1/eff, dis[:, t])]
    constraints += [soc[:, n_t-1] == 0.5 * e_nom]

    for t in range(n_t):
        inj = gen_bus @ g[:, t] + stor_bus @ (dis[:, t] - ch[:, t]) - D[:, t] - curt[:, t] + unmet[:, t]
        constraints += [inj + incidence @ f[:, t] == 0]

    objective = cp.Minimize(
        cp.sum(cp.multiply(gen_cost[:, None], g))
        + curtailment_cost * cp.sum(curt)
        + unmet_cost * cp.sum(unmet)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError(f'Solver failed with status {prob.status}')

    gen_df = pd.DataFrame(g.value.T, columns=[row['gen_id'] for row in gens])
    flow_df = pd.DataFrame(f.value.T, columns=[f'l{i}' for i in range(n_l)])
    soc_df = pd.DataFrame(soc.value.T, columns=[f's{i}' for i in range(n_s)])
    ch_df = pd.DataFrame(ch.value.T, columns=[f's{i}' for i in range(n_s)])
    dis_df = pd.DataFrame(dis.value.T, columns=[f's{i}' for i in range(n_s)])
    curt_df = pd.DataFrame(curt.value.T, columns=bus_names)
    unmet_df = pd.DataFrame(unmet.value.T, columns=bus_names)

    gen_meta = pd.DataFrame([{k:v for k,v in row.items() if k!='avail'} for row in gens])
    carrier_dispatch = {}
    for carrier, sub in gen_meta.groupby('carrier'):
        carrier_dispatch[carrier] = gen_df[sub['gen_id']].sum(axis=1)
    carrier_dispatch = pd.DataFrame(carrier_dispatch)
    wind_available = gen_meta.loc[gen_meta['carrier']=='onshore wind', 'gen_id'].tolist()
    wind_avail_ts = gen_df[wind_available].copy()
    for gid in wind_available:
        idx = int(gid[1:])
        wind_avail_ts[gid] = gens[idx]['avail']
    wind_available_total = wind_avail_ts.sum(axis=1)
    wind_dispatch_total = gen_df[wind_available].sum(axis=1)
    wind_curtailment = wind_available_total - wind_dispatch_total

    line_loading = flow_df.abs().copy()
    for i, cap in enumerate(line_caps):
        line_loading[f'l{i}'] /= cap

    summary = {
        'scenario': {'link_scale': link_scale, 'gas_cost_adder': gas_cost_adder},
        'objective_gbp': float(prob.value),
        'total_demand_mwh': float(D.sum()),
        'total_generation_mwh': float(gen_df.sum().sum()),
        'total_curtailment_mwh': float(curt_df.sum().sum()),
        'total_unserved_mwh': float(unmet_df.sum().sum()),
        'wind_available_mwh': float(wind_available_total.sum()),
        'wind_dispatched_mwh': float(wind_dispatch_total.sum()),
        'wind_curtailed_mwh': float(wind_curtailment.sum()),
        'storage_charge_mwh': float(ch_df.sum().sum()),
        'storage_discharge_mwh': float(dis_df.sum().sum()),
        'peak_demand_mw': float(D.sum(axis=0).max()),
        'max_line_loading': float(line_loading.max().max()),
    }
    for carrier in carrier_dispatch.columns:
        summary[f'dispatch_{carrier}_mwh'] = float(carrier_dispatch[carrier].sum())

    tag = f'ls{link_scale:.2f}_ga{gas_cost_adder:.0f}'.replace('.','p')
    gen_df.to_csv(OUT / f'generation_{tag}.csv', index=False)
    flow_df.to_csv(OUT / f'flows_{tag}.csv', index=False)
    soc_df.to_csv(OUT / f'soc_{tag}.csv', index=False)
    ch_df.to_csv(OUT / f'charge_{tag}.csv', index=False)
    dis_df.to_csv(OUT / f'discharge_{tag}.csv', index=False)
    curt_df.to_csv(OUT / f'curtailment_{tag}.csv', index=False)
    unmet_df.to_csv(OUT / f'unserved_{tag}.csv', index=False)
    pd.DataFrame([summary]).to_csv(OUT / f'summary_{tag}.csv', index=False)

    return {
        'summary': summary,
        'carrier_dispatch': carrier_dispatch,
        'wind_available_total': wind_available_total,
        'wind_dispatch_total': wind_dispatch_total,
        'wind_curtailment': wind_curtailment,
        'soc_df': soc_df,
        'flow_df': flow_df,
        'line_loading': line_loading,
        'buses': buses,
        'links': links,
        'demand': demand,
        'gen_meta': gen_meta,
        'tag': tag,
    }


def make_plots(base, constrained):
    buses = base['buses']
    links = base['links']
    demand = base['demand']

    # Figure 1: network map
    fig, ax = plt.subplots(figsize=(8, 8))
    G = nx.Graph()
    for _, row in buses.iterrows():
        G.add_node(row['name'], pos=(row['x'], row['y']))
    for _, row in links.iterrows():
        G.add_edge(row['bus0'], row['bus1'])
    pos = {row['name']: (row['x'], row['y']) for _, row in buses.iterrows()}
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=1.5)
    sizes = demand.sum()/demand.sum().max()*800
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes.loc[buses['name']].values, node_color='tab:blue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title('GB test system topology and relative weekly demand by bus')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.tight_layout()
    fig.savefig(IMG / 'network_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: dispatch stack base
    for res, name in [(base, 'dispatch_base.png'), (constrained, 'dispatch_constrained.png')]:
        cd = res['carrier_dispatch'].copy()
        total_load = res['demand'].sum(axis=1)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.stackplot(cd.index, [cd[c] for c in cd.columns], labels=cd.columns, alpha=0.85)
        ax.plot(total_load.values, color='black', linewidth=1.5, label='Demand')
        ax.plot(res['wind_available_total'].values, color='tab:green', linestyle='--', linewidth=1, label='Wind available')
        ax.set_title(f'Hourly dispatch stack: {res["tag"]}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('MW')
        ax.legend(ncol=5, fontsize=8, loc='upper right')
        fig.tight_layout()
        fig.savefig(IMG / name, dpi=200)
        plt.close(fig)

    # Figure 3: storage SOC
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in base['soc_df'].columns:
        ax.plot(base['soc_df'][col], label=f'{col} base', alpha=0.8)
    for col in constrained['soc_df'].columns:
        ax.plot(constrained['soc_df'][col], linestyle='--', label=f'{col} constrained', alpha=0.8)
    ax.set_title('Storage state of charge trajectories')
    ax.set_xlabel('Hour')
    ax.set_ylabel('MWh')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / 'storage_soc.png', dpi=200)
    plt.close(fig)

    # Figure 4: line loading duration
    fig, ax = plt.subplots(figsize=(8, 4))
    for res, label in [(base, 'Base'), (constrained, 'Constrained')]:
        vals = np.sort(res['line_loading'].values.ravel())[::-1]
        ax.plot(vals, label=label)
    ax.set_title('Line loading duration curve')
    ax.set_xlabel('Line-hour rank')
    ax.set_ylabel('Loading (p.u.)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'line_loading_duration.png', dpi=200)
    plt.close(fig)

    # Figure 5: scenario comparison bars
    comp = pd.DataFrame([
        base['summary'] | {'case':'Base'},
        constrained['summary'] | {'case':'Constrained'}
    ])
    metrics = ['objective_gbp','dispatch_onshore wind_mwh','dispatch_gas_mwh','wind_curtailed_mwh','max_line_loading']
    labels = ['System cost (£)','Wind dispatch (MWh)','Gas dispatch (MWh)','Wind curtailment (MWh)','Max line loading (p.u.)']
    fig, axes = plt.subplots(1, len(metrics), figsize=(16,4))
    for ax, m, lab in zip(axes, metrics, labels):
        sns.barplot(data=comp, x='case', y=m, ax=ax, palette='Set2')
        ax.set_title(lab)
        ax.set_xlabel('')
        if m != 'max_line_loading':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.suptitle('Scenario comparison')
    fig.tight_layout()
    fig.savefig(IMG / 'scenario_comparison.png', dpi=200)
    plt.close(fig)


def main():
    base = solve_dispatch(link_scale=1.0, gas_cost_adder=0.0)
    constrained = solve_dispatch(link_scale=0.6, gas_cost_adder=20.0)
    make_plots(base, constrained)
    summary = pd.DataFrame([base['summary'] | {'case':'base'}, constrained['summary'] | {'case':'constrained'}])
    summary.to_csv(OUT / 'scenario_summary.csv', index=False)
    with open(OUT / 'scenario_summary.json', 'w') as f:
        json.dump(summary.to_dict(orient='records'), f, indent=2)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
