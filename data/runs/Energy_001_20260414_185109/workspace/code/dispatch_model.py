"""
GB Power System Optimal Dispatch Model (v3)
============================================
Correct sign conventions for power balance.
Generation + discharge + load_shedding = demand + charge + curtailment
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import linprog

DATA_DIR = "data"
OUT_DIR = "outputs"
FIG_DIR = "report/images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

buses = pd.read_csv(f"{DATA_DIR}/buses.csv")
links = pd.read_csv(f"{DATA_DIR}/links.csv")
gens = pd.read_csv(f"{DATA_DIR}/generators.csv")
storage = pd.read_csv(f"{DATA_DIR}/storage.csv")
demand = pd.read_csv(f"{DATA_DIR}/demand.csv")
wind_cf = pd.read_csv(f"{DATA_DIR}/wind_cf.csv")

bus_list = buses["name"].tolist()
n_bus = len(bus_list)
bus_idx = {b: i for i, b in enumerate(bus_list)}
n_t = len(demand)
n_gen = len(gens)
n_link = len(links)
n_sto = len(storage)

VOLL = 10000.0  # £/MWh

# Variable indices
def gen_var(g, t):    return g * n_t + t
def link_var(l, t):   return n_gen * n_t + l * n_t + t
def ch_var(s, t):     return n_gen * n_t + n_link * n_t + s * n_t + t
def dis_var(s, t):    return n_gen * n_t + n_link * n_t + n_sto * n_t + s * n_t + t
def soc_var(s, t):    return n_gen * n_t + n_link * n_t + 2 * n_sto * n_t + s * n_t + t
def cur_var(b, t):    return n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + b * n_t + t
def shed_var(b, t):   return n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + n_bus * n_t + b * n_t + t

n_var = n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + 2 * n_bus * n_t
print(f"Buses:{n_bus} Hours:{n_t} Gens:{n_gen} Links:{n_link} Sto:{n_sto} Vars:{n_var}")

# Objective: min fuel cost + VOLL * shedding
c = np.zeros(n_var)
for g in range(n_gen):
    mc = gens.loc[g, "marginal_cost"]
    for t in range(n_t):
        c[gen_var(g, t)] = mc
for b in range(n_bus):
    for t in range(n_t):
        c[shed_var(b, t)] = VOLL

# Equality: power balance (n_bus*n_t) + storage energy (n_sto*n_t)
n_eq = n_bus * n_t + n_sto * n_t
# Inequality: gen_upper + 2*link + ch_upper + dis_upper + soc_upper + soc_lower
n_ub = n_gen * n_t + 2 * n_link * n_t + 4 * n_sto * n_t

A_eq = np.zeros((n_eq, n_var))
b_eq = np.zeros(n_eq)
A_ub = np.zeros((n_ub, n_var))
b_ub = np.zeros(n_ub)

# Power balance: gen + dis + shed + link_in = demand + ch + cur + link_out
# Rearranged: gen + dis + shed - ch - cur + (link_in - link_out) = demand
eq_row = 0
for t in range(n_t):
    for b in range(n_bus):
        bus_name = bus_list[b]
        b_eq[eq_row] = demand.loc[t, bus_name]
        # Generation adds to supply
        for g in range(n_gen):
            if bus_idx[gens.loc[g, "bus"]] == b:
                A_eq[eq_row, gen_var(g, t)] = 1.0
        # Discharge adds to supply, charge consumes
        for s in range(n_sto):
            if bus_idx[storage.loc[s, "bus"]] == b:
                A_eq[eq_row, dis_var(s, t)] = 1.0
                A_eq[eq_row, ch_var(s, t)] = -1.0
        # Curtailment consumes generation
        A_eq[eq_row, cur_var(b, t)] = -1.0
        # Load shedding adds to supply
        A_eq[eq_row, shed_var(b, t)] = 1.0
        # Link flows: flow from bus0->bus1 means bus0 loses power, bus1 gains
        for l in range(n_link):
            b0 = bus_idx[links.loc[l, "bus0"]]
            b1 = bus_idx[links.loc[l, "bus1"]]
            if b0 == b:
                A_eq[eq_row, link_var(l, t)] = -1.0  # power leaving
            if b1 == b:
                A_eq[eq_row, link_var(l, t)] = 1.0   # power entering
        eq_row += 1

# Storage energy balance: soc(t) = soc(t-1) + eff*ch(t) - dis(t)/eff
for t in range(n_t):
    for s in range(n_sto):
        eff = storage.loc[s, "efficiency"]
        e_nom = storage.loc[s, "e_nom"]
        A_eq[eq_row, soc_var(s, t)] = 1.0
        A_eq[eq_row, ch_var(s, t)] = -eff
        A_eq[eq_row, dis_var(s, t)] = 1.0 / eff
        if t > 0:
            A_eq[eq_row, soc_var(s, t - 1)] = -1.0
        else:
            b_eq[eq_row] = e_nom / 2.0  # initial SOC
        eq_row += 1

# Inequality: generator upper bounds
ub_row = 0
for g in range(n_gen):
    bus_name = gens.loc[g, "bus"]
    carrier = gens.loc[g, "carrier"]
    p_nom = gens.loc[g, "p_nom"]
    for t in range(n_t):
        A_ub[ub_row, gen_var(g, t)] = 1.0
        if carrier == "onshore wind":
            b_ub[ub_row] = p_nom * wind_cf.loc[t, bus_name]
        else:
            b_ub[ub_row] = p_nom
        ub_row += 1

# Link flow upper bounds
for l in range(n_link):
    pn = links.loc[l, "p_nom"]
    for t in range(n_t):
        A_ub[ub_row, link_var(l, t)] = 1.0
        b_ub[ub_row] = pn
        ub_row += 1
# Link flow lower bounds
for l in range(n_link):
    pn = links.loc[l, "p_nom"]
    for t in range(n_t):
        A_ub[ub_row, link_var(l, t)] = -1.0
        b_ub[ub_row] = pn
        ub_row += 1

# Storage charge upper
for s in range(n_sto):
    pn = storage.loc[s, "p_nom"]
    for t in range(n_t):
        A_ub[ub_row, ch_var(s, t)] = 1.0
        b_ub[ub_row] = pn
        ub_row += 1
# Storage discharge upper
for s in range(n_sto):
    pn = storage.loc[s, "p_nom"]
    for t in range(n_t):
        A_ub[ub_row, dis_var(s, t)] = 1.0
        b_ub[ub_row] = pn
        ub_row += 1
# SOC upper
for s in range(n_sto):
    en = storage.loc[s, "e_nom"]
    for t in range(n_t):
        A_ub[ub_row, soc_var(s, t)] = 1.0
        b_ub[ub_row] = en
        ub_row += 1
# SOC lower (>=0)
for s in range(n_sto):
    for t in range(n_t):
        A_ub[ub_row, soc_var(s, t)] = -1.0
        b_ub[ub_row] = 0.0
        ub_row += 1

# Bounds
bounds = []
for _ in range(n_gen * n_t): bounds.append((0, None))  # gen
for _ in range(n_link * n_t): bounds.append((None, None))  # link
for _ in range(n_sto * n_t): bounds.append((0, None))  # ch
for _ in range(n_sto * n_t): bounds.append((0, None))  # dis
for _ in range(n_sto * n_t): bounds.append((0, None))  # soc
for _ in range(n_bus * n_t): bounds.append((0, None))  # cur
for _ in range(n_bus * n_t): bounds.append((0, None))  # shed

print("Solving LP...")
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                 method='highs', options={'maxiter': 500000})

if not result.success:
    print(f"LP failed: {result.message}")
    raise RuntimeError("LP infeasible")

x = result.x
print(f"Optimal total cost: £{result.fun:,.0f}")

# Extract results
gen_dispatch = np.zeros((n_gen, n_t))
for g in range(n_gen):
    for t in range(n_t):
        gen_dispatch[g, t] = x[gen_var(g, t)]

link_flow = np.zeros((n_link, n_t))
for l in range(n_link):
    for t in range(n_t):
        link_flow[l, t] = x[link_var(l, t)]

storage_charge = np.zeros((n_sto, n_t))
storage_discharge = np.zeros((n_sto, n_t))
storage_soc = np.zeros((n_sto, n_t))
for s in range(n_sto):
    for t in range(n_t):
        storage_charge[s, t] = x[ch_var(s, t)]
        storage_discharge[s, t] = x[dis_var(s, t)]
        storage_soc[s, t] = x[soc_var(s, t)]

curtailment = np.zeros((n_bus, n_t))
load_shedding = np.zeros((n_bus, n_t))
for b in range(n_bus):
    for t in range(n_t):
        curtailment[b, t] = x[cur_var(b, t)]
        load_shedding[b, t] = x[shed_var(b, t)]

# Aggregate by carrier
carrier_gen = {}
for g in range(n_gen):
    carrier = gens.loc[g, "carrier"]
    if carrier not in carrier_gen:
        carrier_gen[carrier] = np.zeros(n_t)
    carrier_gen[carrier] += gen_dispatch[g, :]

total_demand = demand.sum(axis=1).values

results_summary = {
    "total_cost_gbp": float(result.fun),
    "fuel_cost_gbp": float(sum(carrier_gen[c].sum() * gens[gens.carrier==c]["marginal_cost"].iloc[0] for c in carrier_gen)),
    "shedding_cost_gbp": float(load_shedding.sum() * VOLL),
    "total_demand_gwh": float(total_demand.sum() / 1e3),
    "total_generation_by_carrier_gwh": {k: float(v.sum() / 1e3) for k, v in carrier_gen.items()},
    "total_storage_charge_gwh": float(storage_charge.sum() / 1e3),
    "total_storage_discharge_gwh": float(storage_discharge.sum() / 1e3),
    "total_curtailment_gwh": float(curtailment.sum() / 1e3),
    "total_load_shedding_gwh": float(load_shedding.sum() / 1e3),
    "shedding_pct_of_demand": float(load_shedding.sum() / total_demand.sum() * 100),
}

with open(f"{OUT_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

hourly_df = pd.DataFrame({
    "hour": range(n_t),
    "total_demand_mw": total_demand,
    "total_curtailment_mw": curtailment.sum(axis=0),
    "total_load_shedding_mw": load_shedding.sum(axis=0),
    "total_storage_charge_mw": storage_charge.sum(axis=0),
    "total_storage_discharge_mw": storage_discharge.sum(axis=0),
})
for carrier, vals in carrier_gen.items():
    hourly_df[f"gen_{carrier}_mw"] = vals
hourly_df.to_csv(f"{OUT_DIR}/hourly_dispatch.csv", index=False)

gen_df = gens.copy()
gen_df["total_generation_gwh"] = gen_dispatch.sum(axis=1) / 1e3
gen_df.to_csv(f"{OUT_DIR}/generator_results.csv", index=False)

bus_results = []
for b in range(n_bus):
    bus_name = bus_list[b]
    bus_demand_val = demand[bus_name].sum() / 1e3
    bus_gen_val = sum(gen_dispatch[g, :].sum() / 1e3 for g in range(n_gen) if bus_idx[gens.loc[g, "bus"]] == b)
    bus_curt = curtailment[b, :].sum() / 1e3
    bus_shed = load_shedding[b, :].sum() / 1e3
    bus_results.append({
        "bus": bus_name,
        "demand_gwh": round(bus_demand_val, 2),
        "generation_gwh": round(bus_gen_val, 2),
        "curtailment_gwh": round(bus_curt, 2),
        "load_shedding_gwh": round(bus_shed, 2),
    })
pd.DataFrame(bus_results).to_csv(f"{OUT_DIR}/bus_results.csv", index=False)

link_results = []
for l in range(n_link):
    link_results.append({
        "link": f"{links.loc[l,'bus0']}-{links.loc[l,'bus1']}",
        "p_nom_mw": links.loc[l, "p_nom"],
        "mean_flow_mw": round(link_flow[l, :].mean(), 1),
        "max_flow_mw": round(link_flow[l, :].max(), 1),
        "min_flow_mw": round(link_flow[l, :].min(), 1),
        "utilization_pct": round(abs(link_flow[l, :]).max() / links.loc[l, "p_nom"] * 100, 1),
    })
pd.DataFrame(link_results).to_csv(f"{OUT_DIR}/link_results.csv", index=False)

print("\n=== Results Summary ===")
for k, v in results_summary.items():
    print(f"  {k}: {v}")
print("\nDone.")
