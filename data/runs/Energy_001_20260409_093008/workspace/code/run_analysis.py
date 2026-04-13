from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"

LOAD_SHED_COST = 10000.0


@dataclass
class Scenario:
    name: str
    line_scale: float = 1.0
    storage_scale: float = 1.0
    wind_scale: float = 1.0


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "buses": pd.read_csv(DATA / "buses.csv"),
        "links": pd.read_csv(DATA / "links.csv"),
        "demand": pd.read_csv(DATA / "demand.csv"),
        "generators": pd.read_csv(DATA / "generators.csv"),
        "wind_cf": pd.read_csv(DATA / "wind_cf.csv"),
        "storage": pd.read_csv(DATA / "storage.csv"),
    }


def shortest_path_hops(buses: list[str], links: pd.DataFrame) -> np.ndarray:
    n = len(buses)
    idx = {b: i for i, b in enumerate(buses)}
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    for _, row in links.iterrows():
        i = idx[row["bus0"]]
        j = idx[row["bus1"]]
        dist[i, j] = 1
        dist[j, i] = 1
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def prepare_static(inputs: dict[str, pd.DataFrame], scenario: Scenario) -> dict[str, object]:
    buses = inputs["buses"]["name"].tolist()
    demand = inputs["demand"][buses].to_numpy()
    generators = inputs["generators"].copy().reset_index(drop=True)
    storage = inputs["storage"].copy().reset_index(drop=True)
    links = inputs["links"].copy()

    wind_mask = generators["carrier"] == "onshore wind"
    generators.loc[wind_mask, "p_nom"] *= scenario.wind_scale
    storage["p_nom"] *= scenario.storage_scale
    storage["e_nom"] *= scenario.storage_scale
    links["p_nom"] *= scenario.line_scale

    bus_idx = {b: i for i, b in enumerate(buses)}
    hops = shortest_path_hops(buses, links)
    transfer_cost = np.where(np.isfinite(hops), 0.15 * hops, 1e4)

    gen_bus = generators["bus"].map(bus_idx).to_numpy()
    store_bus = storage["bus"].map(bus_idx).to_numpy() if len(storage) else np.array([], dtype=int)

    wind_cf = inputs["wind_cf"][buses].to_numpy()
    avail = np.zeros((len(demand), len(generators)))
    for g, row in generators.iterrows():
        if row["carrier"] == "onshore wind":
            avail[:, g] = row["p_nom"] * wind_cf[:, bus_idx[row["bus"]]]
        else:
            avail[:, g] = row["p_nom"]

    return {
        "buses": buses,
        "bus_idx": bus_idx,
        "demand": demand,
        "generators": generators,
        "storage": storage,
        "links": links,
        "avail": avail,
        "transfer_cost": transfer_cost,
        "gen_bus": gen_bus,
        "store_bus": store_bus,
        "line_cap_total": float(links["p_nom"].sum()),
    }


def solve_hour(
    demand_t: np.ndarray,
    avail_t: np.ndarray,
    generators: pd.DataFrame,
    buses: list[str],
    gen_bus: np.ndarray,
    transfer_cost: np.ndarray,
) -> dict[str, np.ndarray | float]:
    B = len(buses)
    G = len(generators)
    n_gen = G
    n_trade = B * B
    n_shed = B
    total = n_gen + n_trade + n_shed

    def trade_ix(i: int, j: int) -> int:
        return n_gen + i * B + j

    def shed_ix(i: int) -> int:
        return n_gen + n_trade + i

    c = np.zeros(total)
    c[:G] = generators["marginal_cost"].to_numpy()
    for i in range(B):
        for j in range(B):
            c[trade_ix(i, j)] = transfer_cost[i, j]
    for i in range(B):
        c[shed_ix(i)] = LOAD_SHED_COST

    bounds = [(0.0, float(avail_t[g])) for g in range(G)]
    bounds += [(0.0, None) for _ in range(n_trade)]
    bounds += [(0.0, None) for _ in range(n_shed)]

    A_eq = []
    b_eq = []
    for b in range(B):
        row = np.zeros(total)
        for g in range(G):
            if gen_bus[g] == b:
                row[g] += 1.0
        for j in range(B):
            row[trade_ix(j, b)] += 1.0
            row[trade_ix(b, j)] -= 1.0
        row[shed_ix(b)] += 1.0
        A_eq.append(row)
        b_eq.append(float(demand_t[b]))

    res = linprog(c, A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    x = res.x
    gen = x[:G]
    trade = x[n_gen : n_gen + n_trade].reshape(B, B)
    shed = x[n_gen + n_trade :]
    return {
        "objective": float(res.fun),
        "generation": gen,
        "trade": trade,
        "shed": shed,
    }


def solve_scenario(inputs: dict[str, pd.DataFrame], scenario: Scenario) -> dict[str, object]:
    static = prepare_static(inputs, scenario)
    buses = static["buses"]
    demand = static["demand"]
    generators = static["generators"]
    storage = static["storage"]
    avail = static["avail"]
    transfer_cost = static["transfer_cost"]
    gen_bus = static["gen_bus"]
    store_bus = static["store_bus"]

    T = len(demand)
    B = len(buses)
    G = len(generators)
    S = len(storage)

    gen_out = np.zeros((T, G))
    trade_out = np.zeros((T, B, B))
    shed_out = np.zeros((T, B))
    charge_out = np.zeros((T, S))
    discharge_out = np.zeros((T, S))
    soc_out = np.zeros((T, S))
    objective = 0.0

    if S:
        eta = np.sqrt(storage["efficiency"].to_numpy())
        soc = 0.5 * storage["e_nom"].to_numpy()
        storage_headroom = storage["p_nom"].to_numpy()
    else:
        eta = np.array([])
        soc = np.array([])
        storage_headroom = np.array([])

    for t in range(T):
        residual = demand[t].copy()

        if S:
            discharge_cap = np.minimum(storage_headroom, soc * eta)
            for s in range(S):
                bus = store_bus[s]
                dispatch = min(discharge_cap[s], max(residual[bus], 0.0) * 0.2)
                discharge_out[t, s] = dispatch
                soc[s] -= dispatch / eta[s]
                residual[bus] -= dispatch

        hour = solve_hour(
            demand_t=residual,
            avail_t=avail[t],
            generators=generators,
            buses=buses,
            gen_bus=gen_bus,
            transfer_cost=transfer_cost,
        )
        gen_out[t] = hour["generation"]
        trade_out[t] = hour["trade"]
        shed_out[t] = hour["shed"]
        objective += hour["objective"]

        if S:
            wind_dispatch_by_bus = np.zeros(B)
            wind_avail_by_bus = np.zeros(B)
            for g, row in generators.iterrows():
                if row["carrier"] == "onshore wind":
                    wind_dispatch_by_bus[gen_bus[g]] += gen_out[t, g]
                    wind_avail_by_bus[gen_bus[g]] += avail[t, g]
            surplus = np.maximum(wind_avail_by_bus - wind_dispatch_by_bus, 0.0)
            for s in range(S):
                bus = store_bus[s]
                charge_cap = min(storage_headroom[s], (storage.loc[s, "e_nom"] - soc[s]) / max(eta[s], 1e-9))
                charge = min(charge_cap, surplus[bus] * 0.25)
                charge_out[t, s] = charge
                soc[s] += charge * eta[s]
            soc_out[t] = soc

    carrier_dispatch = {}
    for carrier in generators["carrier"].unique():
        carrier_dispatch[carrier] = gen_out[:, generators["carrier"] == carrier].sum(axis=1)
    wind_curt = np.maximum(avail[:, generators["carrier"] == "onshore wind"].sum(axis=1) - carrier_dispatch.get("onshore wind", np.zeros(T)), 0.0)

    transfer_volume = trade_out.sum(axis=(1, 2))
    summary = {
        "scenario": scenario.name,
        "objective": float(objective),
        "total_demand_mwh": float(demand.sum()),
        "total_generation_mwh": float(gen_out.sum()),
        "total_load_shed_mwh": float(shed_out.sum()),
        "total_curtailment_mwh": float(wind_curt.sum()),
        "wind_share": float(carrier_dispatch.get("onshore wind", np.zeros(T)).sum() / max(gen_out.sum(), 1e-9)),
        "storage_discharge_mwh": float(discharge_out.sum()),
        "storage_charge_mwh": float(charge_out.sum()),
        "peak_transfer_mw": float(transfer_volume.max()),
        "mean_transfer_mw": float(transfer_volume.mean()),
        "transfer_stress": float(transfer_volume.max() / max(static["line_cap_total"], 1e-9)),
    }

    hourly = pd.DataFrame(
        {
            "hour": np.arange(T),
            "demand_mw": demand.sum(axis=1),
            "generation_mw": gen_out.sum(axis=1),
            "load_shed_mw": shed_out.sum(axis=1),
            "wind_dispatch_mw": carrier_dispatch.get("onshore wind", np.zeros(T)),
            "gas_dispatch_mw": carrier_dispatch.get("gas", np.zeros(T)),
            "nuclear_dispatch_mw": carrier_dispatch.get("nuclear", np.zeros(T)),
            "wind_curtailment_mw": wind_curt,
            "storage_charge_mw": charge_out.sum(axis=1) if S else np.zeros(T),
            "storage_discharge_mw": discharge_out.sum(axis=1) if S else np.zeros(T),
            "transfer_volume_mw": transfer_volume,
        }
    )

    bus_balance = pd.DataFrame(
        {
            "bus": buses,
            "total_demand_mwh": demand.sum(axis=0),
            "unserved_mwh": shed_out.sum(axis=0),
            "avg_net_import_mw": trade_out.sum(axis=0).sum(axis=0) / T - trade_out.sum(axis=0).sum(axis=1) / T,
        }
    )

    links = static["links"].copy()
    link_stats = []
    for _, row in links.iterrows():
        i = static["bus_idx"][row["bus0"]]
        j = static["bus_idx"][row["bus1"]]
        directional = trade_out[:, i, j] + trade_out[:, j, i]
        link_stats.append(
            {
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "p_nom_mw": row["p_nom"],
                "proxy_flow_mw": float(directional.mean()),
                "proxy_peak_mw": float(directional.max()),
                "proxy_loading_ratio": float(directional.max() / max(row["p_nom"], 1e-9)),
            }
        )
    line_loading = pd.DataFrame(link_stats)

    storage_state = pd.DataFrame(soc_out, columns=storage["bus"].tolist()) if S else pd.DataFrame()

    return {
        "summary": summary,
        "hourly": hourly,
        "bus_balance": bus_balance,
        "line_loading": line_loading,
        "storage_state": storage_state,
        "buses": buses,
    }


def save_scenario_outputs(res: dict[str, object], scenario: Scenario) -> None:
    stem = scenario.name
    pd.DataFrame([res["summary"]]).to_csv(OUTPUTS / f"{stem}_summary.csv", index=False)
    res["hourly"].to_csv(OUTPUTS / f"{stem}_hourly.csv", index=False)
    res["bus_balance"].to_csv(OUTPUTS / f"{stem}_bus_balance.csv", index=False)
    res["line_loading"].to_csv(OUTPUTS / f"{stem}_line_loading.csv", index=False)
    if not res["storage_state"].empty:
        res["storage_state"].to_csv(OUTPUTS / f"{stem}_storage_state.csv", index=False)


def plot_network_overview(inputs: dict[str, pd.DataFrame]) -> None:
    buses = inputs["buses"]
    links = inputs["links"]
    demand = inputs["demand"].sum()
    gen = inputs["generators"].groupby("bus")["p_nom"].sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, line in links.iterrows():
        b0 = buses.loc[buses["name"] == line["bus0"]].iloc[0]
        b1 = buses.loc[buses["name"] == line["bus1"]].iloc[0]
        ax.plot([b0["x"], b1["x"]], [b0["y"], b1["y"]], color="#b0b7c3", linewidth=1 + line["p_nom"] / 2500, alpha=0.7)
    sizes = 30 + 0.03 * demand[buses["name"]].to_numpy()
    colors = gen.reindex(buses["name"]).fillna(0).to_numpy()
    sc = ax.scatter(buses["x"], buses["y"], s=sizes, c=colors, cmap="viridis", edgecolor="black", linewidth=0.5)
    for _, row in buses.iterrows():
        ax.text(row["x"] + 0.15, row["y"] + 0.1, row["name"], fontsize=8)
    ax.set_title("GB benchmark network overview")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Installed generation capacity (MW)")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "network_overview.png", dpi=200)
    plt.close(fig)


def plot_dispatch(results: dict[str, dict[str, object]]) -> None:
    base = results["baseline"]["hourly"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        base["hour"],
        base["nuclear_dispatch_mw"],
        base["gas_dispatch_mw"],
        base["wind_dispatch_mw"],
        base["storage_discharge_mw"],
        labels=["Nuclear", "Gas", "Wind", "Storage discharge"],
        colors=["#4c78a8", "#e45756", "#72b7b2", "#f2cf5b"],
        alpha=0.9,
    )
    ax.plot(base["hour"], base["demand_mw"], color="black", linewidth=1.5, label="Demand")
    ax.plot(base["hour"], base["storage_charge_mw"], color="#7f7f7f", linewidth=1.0, linestyle="--", label="Storage charge")
    ax.set_title("Baseline hourly dispatch")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "baseline_dispatch.png", dpi=200)
    plt.close(fig)


def plot_scenario_comparison(summary_df: pd.DataFrame) -> None:
    metrics = summary_df[["scenario", "objective", "total_curtailment_mwh", "storage_discharge_mwh", "transfer_stress"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics.plot.bar(x="scenario", y="objective", ax=axes[0, 0], color="#4c78a8", legend=False)
    axes[0, 0].set_title("System operating cost")
    axes[0, 0].set_ylabel("Cost units")
    metrics.plot.bar(x="scenario", y="total_curtailment_mwh", ax=axes[0, 1], color="#72b7b2", legend=False)
    axes[0, 1].set_title("Wind curtailment")
    axes[0, 1].set_ylabel("MWh")
    metrics.plot.bar(x="scenario", y="storage_discharge_mwh", ax=axes[1, 0], color="#f2cf5b", legend=False)
    axes[1, 0].set_title("Storage discharge")
    axes[1, 0].set_ylabel("MWh")
    metrics.plot.bar(x="scenario", y="transfer_stress", ax=axes[1, 1], color="#e45756", legend=False)
    axes[1, 1].set_title("Peak transfer stress")
    axes[1, 1].set_ylabel("Share of aggregate line capacity")
    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "scenario_comparison.png", dpi=200)
    plt.close(fig)


def plot_congestion_map(results: dict[str, object]) -> None:
    line_loading = results["line_loading"].sort_values("proxy_loading_ratio", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = line_loading["bus0"] + "-" + line_loading["bus1"]
    ax.bar(labels, line_loading["proxy_loading_ratio"], color="#e45756")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Most stressed transmission links in baseline")
    ax.set_ylabel("Proxy loading ratio")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "baseline_congestion.png", dpi=200)
    plt.close(fig)


def write_metadata(summaries: list[dict[str, object]]) -> None:
    metadata = {"scenarios": summaries, "created_by": "code/run_analysis.py"}
    (OUTPUTS / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    ensure_dirs()
    inputs = load_inputs()
    scenarios = [
        Scenario("baseline", line_scale=1.0, storage_scale=1.0, wind_scale=1.0),
        Scenario("no_network_constraints", line_scale=10.0, storage_scale=1.0, wind_scale=1.0),
        Scenario("no_storage", line_scale=1.0, storage_scale=0.0, wind_scale=1.0),
        Scenario("high_wind", line_scale=1.0, storage_scale=1.0, wind_scale=1.5),
    ]

    results = {}
    summaries = []
    for scenario in scenarios:
        res = solve_scenario(inputs, scenario)
        save_scenario_outputs(res, scenario)
        results[scenario.name] = res
        summaries.append(res["summary"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUTS / "scenario_summary.csv", index=False)
    write_metadata(summaries)

    plot_network_overview(inputs)
    plot_dispatch(results)
    plot_scenario_comparison(summary_df)
    plot_congestion_map(results["baseline"])


if __name__ == "__main__":
    main()
