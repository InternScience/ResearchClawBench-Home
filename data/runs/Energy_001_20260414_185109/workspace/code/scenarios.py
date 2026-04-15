"""
GB Power System Scenario Analysis
==================================
Compares three scenarios:
1. Base case (current data)
2. High wind (2x wind capacity)
3. High storage (5x storage capacity)
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import linprog

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

buses = pd.read_csv(f"{DATA_DIR}/buses.csv")
links = pd.read_csv(f"{DATA_DIR}/links.csv")
gens_base = pd.read_csv(f"{DATA_DIR}/generators.csv")
storage_base = pd.read_csv(f"{DATA_DIR}/storage.csv")
demand = pd.read_csv(f"{DATA_DIR}/demand.csv")
wind_cf = pd.read_csv(f"{DATA_DIR}/wind_cf.csv")

bus_list = buses["name"].tolist()
n_bus = len(bus_list)
bus_idx = {b: i for i, b in enumerate(bus_list)}
n_t = len(demand)
n_link = len(links)

VOLL = 10000.0

def solve_dispatch(gens, storage, label=""):
    n_gen = len(gens)
    n_sto = len(storage)
    
    def gen_var(g, t):    return g * n_t + t
    def link_var(l, t):   return n_gen * n_t + l * n_t + t
    def ch_var(s, t):     return n_gen * n_t + n_link * n_t + s * n_t + t
    def dis_var(s, t):    return n_gen * n_t + n_link * n_t + n_sto * n_t + s * n_t + t
    def soc_var(s, t):    return n_gen * n_t + n_link * n_t + 2 * n_sto * n_t + s * n_t + t
    def cur_var(b, t):    return n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + b * n_t + t
    def shed_var(b, t):   return n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + n_bus * n_t + b * n_t + t
    
    n_var = n_gen * n_t + n_link * n_t + 3 * n_sto * n_t + 2 * n_bus * n_t
    
    c = np.zeros(n_var)
    for g in range(n_gen):
        mc = gens.loc[g, "marginal_cost"]
        for t in range(n_t):
            c[gen_var(g, t)] = mc
    for b in range(n_bus):
        for t in range(n_t):
            c[shed_var(b, t)] = VOLL
    
    n_eq = n_bus * n_t + n_sto * n_t
    n_ub = n_gen * n_t + 2 * n_link * n_t + 4 * n_sto * n_t
    
    A_eq = np.zeros((n_eq, n_var))
    b_eq = np.zeros(n_eq)
    A_ub = np.zeros((n_ub, n_var))
    b_ub = np.zeros(n_ub)
    
    eq_row = 0
    for t in range(n_t):
        for b in range(n_bus):
            bus_name = bus_list[b]
            b_eq[eq_row] = demand.loc[t, bus_name]
            for g in range(n_gen):
                if bus_idx[gens.loc[g, "bus"]] == b:
                    A_eq[eq_row, gen_var(g, t)] = 1.0
            for s in range(n_sto):
                if bus_idx[storage.loc[s, "bus"]] == b:
                    A_eq[eq_row, dis_var(s, t)] = 1.0
                    A_eq[eq_row, ch_var(s, t)] = -1.0
            A_eq[eq_row, cur_var(b, t)] = -1.0
            A_eq[eq_row, shed_var(b, t)] = 1.0
            for l in range(n_link):
                b0 = bus_idx[links.loc[l, "bus0"]]
                b1 = bus_idx[links.loc[l, "bus1"]]
                if b0 == b:
                    A_eq[eq_row, link_var(l, t)] = -1.0
                if b1 == b:
                    A_eq[eq_row, link_var(l, t)] = 1.0
            eq_row += 1
    
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
                b_eq[eq_row] = e_nom / 2.0
            eq_row += 1
    
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
    
    for l in range(n_link):
        pn = links.loc[l, "p_nom"]
        for t in range(n_t):
            A_ub[ub_row, link_var(l, t)] = 1.0
            b_ub[ub_row] = pn
            ub_row += 1
    for l in range(n_link):
        pn = links.loc[l, "p_nom"]
        for t in range(n_t):
            A_ub[ub_row, link_var(l, t)] = -1.0
            b_ub[ub_row] = pn
            ub_row += 1
    
    for s in range(n_sto):
        pn = storage.loc[s, "p_nom"]
        for t in range(n_t):
            A_ub[ub_row, ch_var(s, t)] = 1.0
            b_ub[ub_row] = pn
            ub_row += 1
    for s in range(n_sto):
        pn = storage.loc[s, "p_nom"]
        for t in range(n_t):
            A_ub[ub_row, dis_var(s, t)] = 1.0
            b_ub[ub_row] = pn
            ub_row += 1
    for s in range(n_sto):
        en = storage.loc[s, "e_nom"]
        for t in range(n_t):
            A_ub[ub_row, soc_var(s, t)] = 1.0
            b_ub[ub_row] = en
            ub_row += 1
    for s in range(n_sto):
        for t in range(n_t):
            A_ub[ub_row, soc_var(s, t)] = -1.0
            b_ub[ub_row] = 0.0
            ub_row += 1
    
    bounds = []
    for _ in range(n_gen * n_t): bounds.append((0, None))
    for _ in range(n_link * n_t): bounds.append((None, None))
    for _ in range(n_sto * n_t): bounds.append((0, None))
    for _ in range(n_sto * n_t): bounds.append((0, None))
    for _ in range(n_sto * n_t): bounds.append((0, None))
    for _ in range(n_bus * n_t): bounds.append((0, None))
    for _ in range(n_bus * n_t): bounds.append((0, None))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                     method='highs', options={'maxiter': 500000})
    
    if not result.success:
        print(f"  {label}: FAILED - {result.message}")
        return None
    
    x = result.x
    
    gen_dispatch = np.zeros((n_gen, n_t))
    for g in range(n_gen):
        for t in range(n_t):
            gen_dispatch[g, t] = x[gen_var(g, t)]
    
    curtailment = np.zeros((n_bus, n_t))
    load_shedding = np.zeros((n_bus, n_t))
    for b in range(n_bus):
        for t in range(n_t):
            curtailment[b, t] = x[cur_var(b, t)]
            load_shedding[b, t] = x[shed_var(b, t)]
    
    storage_charge = np.zeros((n_sto, n_t))
    storage_discharge = np.zeros((n_sto, n_t))
    for s in range(n_sto):
        for t in range(n_t):
            storage_charge[s, t] = x[ch_var(s, t)]
            storage_discharge[s, t] = x[dis_var(s, t)]
    
    carrier_gen = {}
    for g in range(n_gen):
        carrier = gens.loc[g, "carrier"]
        if carrier not in carrier_gen:
            carrier_gen[carrier] = 0.0
        carrier_gen[carrier] += gen_dispatch[g, :].sum() / 1e3
    
    total_demand_gwh = demand.sum().sum() / 1e3
    
    res = {
        "label": label,
        "total_cost_gbp": float(result.fun),
        "fuel_cost_gbp": float(sum(carrier_gen[c] * 1e3 * gens[gens.carrier==c]["marginal_cost"].iloc[0] for c in carrier_gen)),
        "shedding_cost_gbp": float(load_shedding.sum() * VOLL),
        "total_demand_gwh": float(total_demand_gwh),
        "generation_by_carrier_gwh": carrier_gen,
        "total_curtailment_gwh": float(curtailment.sum() / 1e3),
        "total_load_shedding_gwh": float(load_shedding.sum() / 1e3),
        "shedding_pct": float(load_shedding.sum() / demand.sum().sum() * 100),
        "total_storage_charge_gwh": float(storage_charge.sum() / 1e3),
        "total_storage_discharge_gwh": float(storage_discharge.sum() / 1e3),
    }
    
    print(f"  {label}: Cost=£{result.fun/1e9:.2f}B, Shedding={res['shedding_pct']:.1f}%, "
          f"Wind={carrier_gen.get('onshore wind',0):.0f} GWh, Gas={carrier_gen.get('gas',0):.0f} GWh")
    
    return res

# Scenario 1: Base case
print("Scenario 1: Base case")
res_base = solve_dispatch(gens_base, storage_base, "Base Case")

# Scenario 2: High wind (2x wind capacity)
gens_hw = gens_base.copy()
wind_mask = gens_hw['carrier'] == 'onshore wind'
gens_hw.loc[wind_mask, 'p_nom'] *= 2
print("\nScenario 2: High Wind (2x)")
res_hw = solve_dispatch(gens_hw, storage_base, "High Wind (2x)")

# Scenario 3: High storage (5x storage capacity)
storage_hs = storage_base.copy()
storage_hs['p_nom'] *= 5
storage_hs['e_nom'] *= 5
print("\nScenario 3: High Storage (5x)")
res_hs = solve_dispatch(gens_base, storage_hs, "High Storage (5x)")

# Scenario 4: High wind + High storage
print("\nScenario 4: High Wind + High Storage")
res_hw_hs = solve_dispatch(gens_hw, storage_hs, "High Wind + High Storage")

# Save scenario comparison
scenarios = [res_base, res_hw, res_hs, res_hw_hs]
comparison = []
for s in scenarios:
    if s:
        comparison.append({
            "scenario": s["label"],
            "total_cost_billion_gbp": round(s["total_cost_gbp"] / 1e9, 3),
            "fuel_cost_million_gbp": round(s["fuel_cost_gbp"] / 1e6, 1),
            "shedding_cost_billion_gbp": round(s["shedding_cost_gbp"] / 1e9, 3),
            "shedding_pct": round(s["shedding_pct"], 2),
            "wind_generation_gwh": round(s["generation_by_carrier_gwh"].get("onshore wind", 0), 1),
            "gas_generation_gwh": round(s["generation_by_carrier_gwh"].get("gas", 0), 1),
            "nuclear_generation_gwh": round(s["generation_by_carrier_gwh"].get("nuclear", 0), 1),
            "curtailment_gwh": round(s["total_curtailment_gwh"], 1),
            "storage_charge_gwh": round(s["total_storage_charge_gwh"], 1),
            "storage_discharge_gwh": round(s["total_storage_discharge_gwh"], 1),
        })

pd.DataFrame(comparison).to_csv(f"{OUT_DIR}/scenario_comparison.csv", index=False)
with open(f"{OUT_DIR}/scenario_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("\n=== Scenario Comparison ===")
for c in comparison:
    print(f"  {c['scenario']}: Cost=£{c['total_cost_billion_gbp']}B, Shed={c['shedding_pct']}%")

print("\nDone.")
