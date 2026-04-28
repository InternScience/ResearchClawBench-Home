"""Build a PyPSA network for the GB 20-node power system from CSV data.

The same builder is used by `run_dispatch.py` for several scenario
variants (transmission, wind, gas-cost, storage and demand sweeps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_inputs():
    buses = pd.read_csv(DATA_DIR / "buses.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    demand = pd.read_csv(DATA_DIR / "demand.csv")
    generators = pd.read_csv(DATA_DIR / "generators.csv")
    wind_cf = pd.read_csv(DATA_DIR / "wind_cf.csv")
    storage = pd.read_csv(DATA_DIR / "storage.csv")
    return buses, links, demand, generators, wind_cf, storage


def build_network(
    line_capacity_factor: float = 1.0,
    wind_capacity_factor: float = 1.0,
    wind_cf_factor: float = 1.0,
    gas_cost_factor: float = 1.0,
    include_storage: bool = True,
    voll: float = 3000.0,
    backstop_cost: float = 150.0,
    demand_scale: float = 0.5,
) -> pypsa.Network:
    """Return a configured PyPSA Network.

    Parameters
    ----------
    line_capacity_factor : float
        Multiplier applied to all link nominal power capacities.
    wind_capacity_factor : float
        Multiplier applied to all wind generator p_nom.
    wind_cf_factor : float
        Multiplier applied to wind capacity-factor time series (clipped
        to [0, 1]).
    gas_cost_factor : float
        Multiplier applied to the marginal cost of gas generators.
    include_storage : bool
        If False, remove all storage units from the network.
    voll : float
        Value-of-lost-load (£/MWh) used as the marginal cost of a slack
        load-shedding generator at every bus.
    backstop_cost : float
        Marginal cost (£/MWh) of an unlimited backstop generator placed
        at every bus.  Represents imports and unmodelled peaking
        capacity (biomass, OCGT, interconnectors).
    demand_scale : float
        Multiplier applied to the demand time series.  The supplied
        ``data/demand.csv`` peaks at ~140 GW, well above realistic GB
        peak load (~50 GW); the default of 0.5 brings the model into a
        physically reasonable regime while preserving the temporal and
        spatial pattern of the input data.
    """
    buses, links, demand, generators, wind_cf, storage = _load_inputs()
    demand = demand * demand_scale

    n = pypsa.Network()
    snapshots = pd.date_range("2025-01-06 00:00", periods=len(demand), freq="h")
    n.set_snapshots(snapshots)

    # Buses ------------------------------------------------------------
    for _, b in buses.iterrows():
        n.add(
            "Bus",
            b["name"],
            v_nom=b["v_nom"],
            carrier=b["carrier"],
            x=b["x"],
            y=b["y"],
        )

    # Carriers ---------------------------------------------------------
    for c in ["AC", "onshore wind", "gas", "nuclear", "PHS",
              "backstop", "shed"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # Loads ------------------------------------------------------------
    demand.index = snapshots
    for bus in demand.columns:
        n.add(
            "Load",
            f"L_{bus}",
            bus=bus,
            p_set=demand[bus].values,
        )

    # Links (transmission) --------------------------------------------
    for i, l in links.iterrows():
        n.add(
            "Link",
            f"{l['bus0']}-{l['bus1']}",
            bus0=l["bus0"],
            bus1=l["bus1"],
            p_nom=float(l["p_nom"]) * line_capacity_factor,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            length=l["length"],
            carrier=l["carrier"],
        )

    # Generators -------------------------------------------------------
    wind_cf.index = snapshots
    for i, g in generators.iterrows():
        carrier = g["carrier"]
        bus = g["bus"]
        gname = f"{bus}_{carrier.replace(' ', '_')}_{i}"
        kwargs = dict(bus=bus, carrier=carrier)

        if carrier == "onshore wind":
            cf = wind_cf[bus].clip(lower=0.0, upper=1.0).values * wind_cf_factor
            cf = np.clip(cf, 0.0, 1.0)
            n.add(
                "Generator",
                gname,
                p_nom=float(g["p_nom"]) * wind_capacity_factor,
                marginal_cost=float(g["marginal_cost"]),
                p_max_pu=cf,
                **kwargs,
            )
        elif carrier == "gas":
            n.add(
                "Generator",
                gname,
                p_nom=float(g["p_nom"]),
                marginal_cost=float(g["marginal_cost"]) * gas_cost_factor,
                **kwargs,
            )
        else:
            n.add(
                "Generator",
                gname,
                p_nom=float(g["p_nom"]),
                marginal_cost=float(g["marginal_cost"]),
                **kwargs,
            )

    # Storage ----------------------------------------------------------
    if include_storage:
        for i, s in storage.iterrows():
            eta = float(s["efficiency"])
            n.add(
                "StorageUnit",
                f"S_{s['bus']}_{i}",
                bus=s["bus"],
                carrier=s["carrier"],
                p_nom=float(s["p_nom"]),
                max_hours=float(s["e_nom"]) / float(s["p_nom"]),
                efficiency_store=eta,
                efficiency_dispatch=eta,
                cyclic_state_of_charge=True,
                marginal_cost=0.01,
            )

    # Backstop (imports / unmodelled peaking capacity) ----------------
    for bus in n.buses.index:
        n.add(
            "Generator",
            f"backstop_{bus}",
            bus=bus,
            carrier="backstop",
            p_nom=1e5,
            marginal_cost=backstop_cost,
        )

    # Slack load-shedding generator (only used if even backstop fails)
    for bus in n.buses.index:
        n.add(
            "Generator",
            f"shed_{bus}",
            bus=bus,
            carrier="shed",
            p_nom=1e6,
            marginal_cost=voll,
        )

    return n


if __name__ == "__main__":
    n = build_network()
    print(n)
    print("snapshots:", len(n.snapshots))
    print("buses:", len(n.buses))
    print("generators:", len(n.generators))
    print("storage:", len(n.storage_units))
    print("links:", len(n.links))
