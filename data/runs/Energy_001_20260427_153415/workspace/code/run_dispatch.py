"""Run optimal dispatch for a list of scenarios on the GB 20-node model.

Solves a linear OPF (PyPSA `optimize`) for each scenario and exports
result tables (per-snapshot dispatch, per-bus mix, prices, storage
state of energy) plus a high-level scenario summary used by the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_network import build_network


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


SCENARIOS = {
    "S0_base": dict(),
    "S1_grid_minus50": dict(line_capacity_factor=0.5),
    "S2_wind_plus50": dict(wind_capacity_factor=1.5),
    "S3_no_storage": dict(include_storage=False),
    "S4_high_gas": dict(gas_cost_factor=3.0),
    "S5_wind_drought": dict(wind_cf_factor=0.5),
}


def _carrier_of(name: str, n) -> str:
    if name in n.generators.index:
        return n.generators.at[name, "carrier"]
    return "?"


def solve_scenario(name: str, kwargs: dict) -> dict:
    print(f"\n=== {name} -> {kwargs} ===")
    n = build_network(**kwargs)
    status, cond = n.optimize(solver_name="highs")
    print("status:", status, cond)

    # Dispatch by generator
    p = n.generators_t.p.copy()
    carriers = n.generators.carrier
    dispatch_by_carrier = p.T.groupby(carriers).sum().T
    dispatch_by_carrier.to_csv(OUT / f"dispatch_{name}.csv")

    # Generation mix per bus & carrier (energy MWh over the week)
    gen_total_per_bus_carrier = p.sum(axis=0).groupby(
        [n.generators.bus, n.generators.carrier]
    ).sum().unstack(fill_value=0.0)
    gen_total_per_bus_carrier.to_csv(OUT / f"mix_{name}.csv")

    # Curtailment for wind: max available - dispatched
    wind_gens = n.generators.index[n.generators.carrier == "onshore wind"]
    p_max_pu_t = n.generators_t.p_max_pu.reindex(columns=wind_gens).fillna(1.0)
    p_nom = n.generators.p_nom.loc[wind_gens]
    available = p_max_pu_t * p_nom
    dispatched = p[wind_gens]
    curtailment = (available - dispatched).clip(lower=0.0)
    curtailment.sum(axis=1).to_csv(OUT / f"curtailment_{name}.csv", header=["MW"])
    curtailed_total_mwh = curtailment.values.sum()

    # Marginal prices
    prices = n.buses_t.marginal_price.copy()
    prices.to_csv(OUT / f"prices_{name}.csv")

    # Storage SoC
    if len(n.storage_units) > 0 and not n.storage_units_t.state_of_charge.empty:
        soc = n.storage_units_t.state_of_charge.copy()
        soc.to_csv(OUT / f"storage_soc_{name}.csv")

    # Link flows
    fl = n.links_t.p0.copy()
    fl.to_csv(OUT / f"flows_{name}.csv")
    p_nom_links = n.links.p_nom
    util = (fl.abs().div(p_nom_links, axis=1)).clip(upper=1.0)
    util.to_csv(OUT / f"line_utilisation_{name}.csv")

    # Load shedding
    shed_gens = n.generators.index[n.generators.carrier == "shed"]
    shed_total_mwh = float(p[shed_gens].values.sum())

    # Costs
    obj = float(n.objective)
    energy_total = float(n.loads_t.p_set.values.sum())

    summary = {
        "scenario": name,
        "status": status,
        "condition": cond,
        "objective_GBP": obj,
        "demand_total_MWh": energy_total,
        "wind_MWh": float(dispatch_by_carrier.get("onshore wind", pd.Series([0])).sum()),
        "gas_MWh": float(dispatch_by_carrier.get("gas", pd.Series([0])).sum()),
        "nuclear_MWh": float(dispatch_by_carrier.get("nuclear", pd.Series([0])).sum()),
        "shed_MWh": shed_total_mwh,
        "curtail_MWh": float(curtailed_total_mwh),
        "mean_price_GBP_per_MWh": float(prices.values.mean()),
        "max_price_GBP_per_MWh": float(prices.values.max()),
        "min_price_GBP_per_MWh": float(prices.values.min()),
        "mean_line_loading": float(util.values.mean()),
        "max_line_loading": float(util.values.max()),
        "n_buses": int(len(n.buses)),
        "n_generators": int(len(n.generators)),
        "n_links": int(len(n.links)),
        "n_storage": int(len(n.storage_units)),
    }
    with open(OUT / f"summary_{name}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("status", "condition")}, indent=2, default=str))
    return summary


def main():
    rows = []
    for name, kwargs in SCENARIOS.items():
        rows.append(solve_scenario(name, kwargs))
    pd.DataFrame(rows).to_csv(OUT / "scenario_summary.csv", index=False)
    print("\nWrote outputs/scenario_summary.csv")


if __name__ == "__main__":
    main()
