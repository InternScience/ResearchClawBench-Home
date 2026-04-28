"""Plot figures for the GB power-system dispatch report.

Reads outputs/* produced by run_dispatch.py and saves PNG figures
into report/images/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
IMG.mkdir(parents=True, exist_ok=True)

CARRIER_COLOR = {
    "onshore wind": "#41ab5d",
    "wind": "#41ab5d",
    "nuclear": "#984ea3",
    "gas": "#fb6a4a",
    "backstop": "#969696",
    "shed": "#000000",
}

SCENARIOS = [
    "S0_base",
    "S1_grid_minus50",
    "S2_wind_plus50",
    "S3_no_storage",
    "S4_high_gas",
    "S5_wind_drought",
]


def fig_network_map():
    buses = pd.read_csv(ROOT / "data" / "buses.csv")
    links = pd.read_csv(ROOT / "data" / "links.csv")
    gens = pd.read_csv(ROOT / "data" / "generators.csv")
    gen_per_bus = gens.groupby(["bus", "carrier"]).p_nom.sum().unstack(fill_value=0.0)
    gen_per_bus = gen_per_bus.reindex(buses["name"]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 9))
    for _, l in links.iterrows():
        b0 = buses[buses.name == l.bus0].iloc[0]
        b1 = buses[buses.name == l.bus1].iloc[0]
        lw = 0.5 + 2.0 * float(l.p_nom) / float(links.p_nom.max())
        ax.plot([b0.x, b1.x], [b0.y, b1.y], color="#888", lw=lw, alpha=0.6, zorder=1)

    sizes = gen_per_bus.sum(axis=1) / 50  # MW -> dot size
    ax.scatter(buses.x, buses.y, s=sizes, c="#41ab5d", alpha=0.7,
               edgecolor="k", lw=0.5, zorder=2, label="installed capacity")

    for _, b in buses.iterrows():
        ax.text(b.x + 0.05, b.y + 0.05, b["name"], fontsize=8)

    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_title("GB 20-node network: buses, transmission links, and total\n"
                 "generator capacity (dot size proportional to installed MW)")
    ax.set_aspect(1.4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG / "fig_network_map.png", dpi=140)
    plt.close()


def fig_demand_wind_overview():
    d = pd.read_csv(ROOT / "data" / "demand.csv")
    w = pd.read_csv(ROOT / "data" / "wind_cf.csv")
    t = np.arange(len(d))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, d.sum(axis=1) / 1e3, color="#1f77b4")
    axes[0].set_ylabel("Total demand (GW)")
    axes[0].set_title("System-wide demand profile (raw input, sum across 20 buses)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, w.mean(axis=1), color="#41ab5d")
    axes[1].fill_between(t, w.min(axis=1), w.max(axis=1),
                         color="#41ab5d", alpha=0.2, label="bus min/max")
    axes[1].set_ylabel("Wind capacity factor (–)")
    axes[1].set_xlabel("hour of week")
    axes[1].set_title("Wind capacity factor: mean across buses (line) "
                      "and bus-wise min/max envelope (band)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IMG / "fig_demand_wind_overview.png", dpi=140)
    plt.close()


def fig_dispatch_stack(scenario: str = "S0_base"):
    disp = pd.read_csv(OUT / f"dispatch_{scenario}.csv", index_col=0, parse_dates=True)
    cols = [c for c in ["onshore wind", "nuclear", "gas", "backstop", "shed"]
            if c in disp.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [CARRIER_COLOR.get(c, "#cccccc") for c in cols]
    ax.stackplot(disp.index, [disp[c].values / 1e3 for c in cols],
                 labels=cols, colors=colors, alpha=0.95)
    ax.set_ylabel("Dispatch (GW)")
    ax.set_xlabel("snapshot")
    ax.set_title(f"Hourly system-wide dispatch by carrier — {scenario}")
    ax.legend(loc="upper right", ncol=len(cols))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG / f"fig_dispatch_stack_{scenario}.png", dpi=140)
    plt.close()


def fig_mix_by_scenario():
    sums = {}
    for s in SCENARIOS:
        d = pd.read_csv(OUT / f"dispatch_{s}.csv", index_col=0)
        sums[s] = d.sum(axis=0)
    df = pd.DataFrame(sums).T.fillna(0.0) / 1e3  # GWh
    cols = [c for c in ["onshore wind", "nuclear", "gas", "backstop", "shed"]
            if c in df.columns]
    df = df[cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for c in cols:
        ax.bar(x, df[c].values, bottom=bottom,
               color=CARRIER_COLOR.get(c, "#cccccc"), label=c, edgecolor="white")
        bottom += df[c].values
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.set_ylabel("Energy over the week (GWh)")
    ax.set_title("Weekly generation mix by scenario")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG / "fig_mix_by_scenario.png", dpi=140)
    plt.close()


def fig_cost_curtailment():
    summ = pd.read_csv(OUT / "scenario_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(summ.scenario, summ.objective_GBP / 1e6, color="#1f77b4")
    axes[0].set_ylabel("Total operating cost (£M / week)")
    axes[0].set_title("Total system operating cost")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(summ.scenario, summ.curtail_MWh / 1e3, color="#41ab5d")
    axes[1].set_ylabel("Wind curtailment (GWh / week)")
    axes[1].set_title("Wind curtailment")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(summ.scenario, summ.mean_price_GBP_per_MWh, color="#fb6a4a")
    axes[2].set_ylabel("Mean nodal price (£/MWh)")
    axes[2].set_title("Average locational marginal price")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(IMG / "fig_cost_curtailment.png", dpi=140)
    plt.close()


def fig_storage_soc():
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False
    for s in SCENARIOS:
        path = OUT / f"storage_soc_{s}.csv"
        if not path.exists():
            continue
        soc = pd.read_csv(path, index_col=0, parse_dates=True)
        total = soc.sum(axis=1)
        ax.plot(total.index, total.values, label=s)
        plotted = True
    ax.set_ylabel("Total PHS state of energy (MWh)")
    ax.set_xlabel("snapshot")
    ax.set_title("Pumped hydro storage state of energy across scenarios")
    if plotted:
        ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG / "fig_storage_soc.png", dpi=140)
    plt.close()


def fig_prices_box():
    data = []
    labels = []
    for s in SCENARIOS:
        p = pd.read_csv(OUT / f"prices_{s}.csv", index_col=0)
        data.append(p.values.flatten())
        labels.append(s)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Nodal marginal price (£/MWh)")
    ax.set_title("Distribution of locational marginal prices "
                 "(all buses × all hours)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG / "fig_prices_box.png", dpi=140)
    plt.close()


def fig_line_loading():
    fig, ax = plt.subplots(figsize=(10, 5))
    means = []
    p95 = []
    for s in SCENARIOS:
        u = pd.read_csv(OUT / f"line_utilisation_{s}.csv", index_col=0)
        means.append(u.values.mean())
        p95.append(np.percentile(u.values, 95))
    x = np.arange(len(SCENARIOS))
    w = 0.4
    ax.bar(x - w / 2, means, width=w, label="mean", color="#1f77b4")
    ax.bar(x + w / 2, p95, width=w, label="95th percentile", color="#fb6a4a")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=20, ha="right")
    ax.set_ylabel("Line loading (–)")
    ax.set_title("Transmission line utilisation across scenarios")
    ax.axhline(1.0, ls="--", color="k", lw=1, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG / "fig_line_loading.png", dpi=140)
    plt.close()


def fig_congestion_heatmap(scenario: str = "S0_base"):
    u = pd.read_csv(OUT / f"line_utilisation_{scenario}.csv",
                    index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(u.T.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_yticks(np.arange(u.shape[1]))
    ax.set_yticklabels(u.columns, fontsize=7)
    ax.set_xlabel("hour of week")
    ax.set_title(f"Transmission line loading heatmap — {scenario}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|p_link| / p_nom")
    plt.tight_layout()
    plt.savefig(IMG / f"fig_line_heatmap_{scenario}.png", dpi=140)
    plt.close()


def fig_curtailment_timeseries():
    fig, ax = plt.subplots(figsize=(11, 5))
    for s in SCENARIOS:
        c = pd.read_csv(OUT / f"curtailment_{s}.csv", index_col=0,
                        parse_dates=True)
        ax.plot(c.index, c.iloc[:, 0].values / 1e3, label=s, lw=1)
    ax.set_ylabel("System wind curtailment (GW)")
    ax.set_xlabel("snapshot")
    ax.set_title("Hourly wind curtailment by scenario")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG / "fig_curtailment_ts.png", dpi=140)
    plt.close()


def fig_per_bus_mix(scenario: str = "S0_base"):
    df = pd.read_csv(OUT / f"mix_{scenario}.csv", index_col=0)
    cols = [c for c in ["onshore wind", "nuclear", "gas", "backstop", "shed"]
            if c in df.columns]
    df = df[cols] / 1e3  # GWh
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for c in cols:
        ax.bar(x, df[c].values, bottom=bottom,
               color=CARRIER_COLOR.get(c, "#cccccc"), label=c, edgecolor="white")
        bottom += df[c].values
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.set_ylabel("Weekly energy (GWh)")
    ax.set_title(f"Per-bus generation mix — {scenario}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG / f"fig_per_bus_mix_{scenario}.png", dpi=140)
    plt.close()


def main():
    fig_network_map()
    fig_demand_wind_overview()
    for s in SCENARIOS:
        fig_dispatch_stack(s)
    fig_mix_by_scenario()
    fig_cost_curtailment()
    fig_storage_soc()
    fig_prices_box()
    fig_line_loading()
    fig_congestion_heatmap("S0_base")
    fig_congestion_heatmap("S1_grid_minus50")
    fig_curtailment_timeseries()
    fig_per_bus_mix("S0_base")
    print("All figures written to", IMG)


if __name__ == "__main__":
    main()
