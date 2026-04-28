"""Generate figures and comparison artefacts for the report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / "outputs"
IMG = WORK / "report" / "images"
IMG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160,
                     "font.family": "DejaVu Sans"})

df = pd.read_csv(OUT / "lcoh_delivered_per_hex.csv")
summary = pd.read_csv(OUT / "scenario_summary.csv")
eu_ref = pd.read_csv(OUT / "eu_reference_lcoh.csv")

# --- Africa shape ---
import geopandas as gpd
shp = gpd.read_file(WORK / "data" / "africa_map" / "ne_10m_admin_0_countries.shp")
africa = shp[shp["CONTINENT"] == "Africa"]

EU_LOW   = float(eu_ref.loc[eu_ref.scenario=="S4_EU_LOW_IR","lcoh_eu_per_kg"].iloc[0])
EU_HIGH  = float(eu_ref.loc[eu_ref.scenario=="S5_EU_RISING_IR","lcoh_eu_per_kg"].iloc[0])

# ============================================================
# Fig 1: Data overview — resource map (PV, wind, ocean dist)
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
hex_locs = df[df.scenario=="S1_AFR_BASELINE"]
for ax, col, ttl, cmap in zip(
    axs, ["theo_pv","theo_wind","ocean_dist_km"],
    ["PV resource index", "Wind resource index", "Distance to coast [km]"],
    ["YlOrRd","Blues","viridis"]):
    africa.plot(ax=ax, color="#f3f3f3", edgecolor="0.5", linewidth=0.4)
    sc = ax.scatter(hex_locs["lon"], hex_locs["lat"], c=hex_locs[col],
                    cmap=cmap, s=60, edgecolor="k", linewidth=0.4)
    plt.colorbar(sc, ax=ax, fraction=0.04, label=ttl)
    ax.set_title(ttl); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(8, 30); ax.set_ylim(-32, -14)
fig.suptitle("Fig. 1 — African production hexagons: solar, wind, and coast distance", fontsize=12)
fig.savefig(IMG / "fig_data_overview.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig 2: Map of delivered LCOH — baseline (S1) and de-risked (S3)
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
vmin = df["lcoh_delivered_per_kg"].min()
vmax = df["lcoh_delivered_per_kg"].max()
for ax, sc_name, title in zip(
    axs, ["S1_AFR_BASELINE", "S3_AFR_DERISKED"],
    ["S1 Africa baseline (WACC=10%)", "S3 Africa de-risked (WACC=6%)"]):
    africa.plot(ax=ax, color="#f3f3f3", edgecolor="0.5", linewidth=0.4)
    sub = df[df.scenario==sc_name]
    sc_pl = ax.scatter(sub.lon, sub.lat, c=sub["lcoh_delivered_per_kg"],
                       cmap="RdYlGn_r", vmin=vmin, vmax=vmax, s=80,
                       edgecolor="k", linewidth=0.4)
    plt.colorbar(sc_pl, ax=ax, fraction=0.04,
                 label="Delivered LCOH (€/kg H₂)")
    # mark cheapest 3
    best = sub.nsmallest(3, "lcoh_delivered_per_kg")
    ax.scatter(best.lon, best.lat, marker="*", s=260, facecolor="gold",
               edgecolor="k", linewidth=0.7, zorder=5, label="Cheapest 3")
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(8, 30); ax.set_ylim(-32, -14)
    ax.legend(loc="lower right")
fig.suptitle("Fig. 2 — Delivered green H₂ cost (Africa→Europe via NH₃) under two financing scenarios",
             fontsize=12)
fig.savefig(IMG / "fig_map_baseline_vs_derisked.png", bbox_inches="tight")
plt.close(fig)

# Save individual maps for inventory
for sc_name, fname in [("S1_AFR_BASELINE","fig_map_baseline.png"),
                       ("S3_AFR_DERISKED","fig_map_derisked.png")]:
    fig, ax = plt.subplots(figsize=(7,7))
    africa.plot(ax=ax, color="#f3f3f3", edgecolor="0.5", linewidth=0.4)
    sub = df[df.scenario==sc_name]
    sc_pl = ax.scatter(sub.lon, sub.lat, c=sub["lcoh_delivered_per_kg"],
                       cmap="RdYlGn_r", s=80, edgecolor="k", linewidth=0.4)
    plt.colorbar(sc_pl, ax=ax, fraction=0.04, label="Delivered LCOH (€/kg H₂)")
    ax.set_title(f"Delivered LCOH — {sc_name}")
    ax.set_xlim(8, 30); ax.set_ylim(-32, -14)
    fig.tight_layout(); fig.savefig(IMG / fname, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Fig 3: Cost-component stacked bars (median hex per scenario)
# ============================================================
comp_cols_local = ["c_elec_per_kg", "c_ely_capex_per_kg",
                   "c_water_per_kg", "c_storage_per_kg"]
comp_cols_chain = ["c_nh3_synth_per_kg", "c_truck_per_kg",
                   "c_ship_per_kg", "c_crack_per_kg"]
all_cols = comp_cols_local + comp_cols_chain
labels = ["Electricity (RE)", "Electrolyser CAPEX", "Water", "On-site storage",
          "NH₃ synthesis", "Inland truck", "Sea shipping", "EU NH₃ cracking"]
colors = ["#ffd166","#ef476f","#06aed5","#a3a3a3","#118ab2","#8ac926","#3a86ff","#7209b7"]

med = df.groupby("scenario")[all_cols].median().loc[
        ["S1_AFR_BASELINE","S2_AFR_MODERATE","S3_AFR_DERISKED",
         "S4_EU_LOW_IR","S5_EU_RISING_IR"]]
# For EU rows, suppress chain costs (no shipping); use EU reference instead.
# Replace EU chain with zeros and add the EU reference total stack:
eu_low = eu_ref[eu_ref.scenario=="S4_EU_LOW_IR"].iloc[0]
eu_hi  = eu_ref[eu_ref.scenario=="S5_EU_RISING_IR"].iloc[0]
med.loc["S4_EU_LOW_IR",  comp_cols_local]  = [eu_low.c_elec_per_kg, eu_low.c_ely_capex_per_kg,
                                              eu_low.c_water_per_kg, eu_low.c_storage_per_kg]
med.loc["S5_EU_RISING_IR", comp_cols_local] = [eu_hi.c_elec_per_kg, eu_hi.c_ely_capex_per_kg,
                                               eu_hi.c_water_per_kg, eu_hi.c_storage_per_kg]
med.loc["S4_EU_LOW_IR",  comp_cols_chain] = 0.0
med.loc["S5_EU_RISING_IR", comp_cols_chain] = 0.0

fig, ax = plt.subplots(figsize=(11, 5.6))
xs = np.arange(len(med))
bottom = np.zeros(len(med))
for col, lab, c in zip(all_cols, labels, colors):
    ax.bar(xs, med[col].values, bottom=bottom, label=lab, color=c, edgecolor="white", linewidth=0.5)
    bottom += med[col].values
ax.set_xticks(xs)
ax.set_xticklabels([s.replace("_","\n") for s in med.index], fontsize=9)
ax.set_ylabel("Levelized cost (€/kg H₂)")
ax.axhline(EU_LOW, color="#1d3557", linestyle="--", linewidth=1.2,
           label=f"EU green H₂ low IR ({EU_LOW:.2f} €/kg)")
ax.axhline(EU_HIGH, color="#e63946", linestyle="--", linewidth=1.2,
           label=f"EU green H₂ rising IR ({EU_HIGH:.2f} €/kg)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.set_title("Fig. 3 — Median delivered cost stack by scenario\n(Africa: incl. NH₃ chain; EU: production only)")
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(IMG / "fig_cost_stack.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig 4: WACC sensitivity (delivered cost vs WACC, for cheapest hex)
# ============================================================
import importlib.util
spec = importlib.util.spec_from_file_location("lcoh_model", WORK/"code"/"lcoh_model.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
TE = M.TE
hex_lcoh_at_plant = M.hex_lcoh_at_plant
add_export_chain = M.add_export_chain
european_reference_lcoh = M.european_reference_lcoh
SCENARIOS = M.SCENARIOS

# Pick representative cheapest hex from baseline
cheap_hex_id = df[df.scenario=="S1_AFR_BASELINE"].nsmallest(1,"lcoh_delivered_per_kg")["hex_id"].iloc[0]
hex_row = pd.read_csv(WORK/"data"/"hex_final_NA_min.csv").set_index("hex_id").loc[cheap_hex_id]

wacc_grid = np.linspace(0.03, 0.13, 21)
afr_costs = []; eu_costs = []
for w in wacc_grid:
    sc = {"wacc_re": w, "wacc_ely": w, "wacc_infra": w}
    plant = hex_lcoh_at_plant(hex_row, sc)
    chain = add_export_chain(hex_row, sc, plant)
    upstream = (plant["lcoh_plant_per_kg"] + chain["c_nh3_synth_per_kg"]
                + chain["c_truck_per_kg"]) * chain["loss_factor"]
    delivered = (upstream + chain["c_ship_per_kg"]) * chain["yield_factor"] + chain["c_crack_per_kg"]
    afr_costs.append(delivered)
    eu_costs.append(european_reference_lcoh(sc)["lcoh_eu_per_kg"])

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(wacc_grid*100, afr_costs, marker="o", label="Africa delivered (best hex, NH₃ chain)",
        color="#e76f51", linewidth=2)
ax.plot(wacc_grid*100, eu_costs, marker="s", label="Europe green H₂ (production only)",
        color="#264653", linewidth=2)
# Highlight scenario WACCs
for w_pct, txt, col in [(10, "S1 AFR baseline", "#e76f51"),
                        (8,  "S2 AFR moderate",  "#f4a261"),
                        (6,  "S3 AFR de-risked", "#2a9d8f"),
                        (4,  "S4 EU low IR",     "#264653"),
                        (7,  "S5 EU rising IR",  "#e63946")]:
    ax.axvline(w_pct, color=col, linestyle=":", alpha=0.4)
ax.set_xlabel("Project WACC (% real)")
ax.set_ylabel("Delivered LCOH (€/kg H₂)")
ax.set_title("Fig. 4 — Delivered green H₂ cost vs cost of capital\n(Africa best hex via NH₃ to EU vs Europe in-region)")
ax.grid(linestyle=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(IMG / "fig_wacc_sensitivity.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig 5: Africa vs Europe direct comparison — boxplots + EU lines
# ============================================================
order = ["S1_AFR_BASELINE","S2_AFR_MODERATE","S3_AFR_DERISKED"]
data_box = [df[df.scenario==s]["lcoh_delivered_per_kg"].values for s in order]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
bp = ax.boxplot(data_box, tick_labels=["S1 baseline\n(WACC 10%)",
                                       "S2 moderate\n(WACC 8%)",
                                       "S3 de-risked\n(WACC 6%)"],
                patch_artist=True, widths=0.55)
for patch, c in zip(bp["boxes"], ["#e76f51","#f4a261","#2a9d8f"]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(EU_LOW, color="#264653", linestyle="--", linewidth=1.4,
           label=f"EU green H₂ low IR ({EU_LOW:.2f} €/kg)")
ax.axhline(EU_HIGH, color="#e63946", linestyle="--", linewidth=1.4,
           label=f"EU green H₂ rising IR ({EU_HIGH:.2f} €/kg)")
# Annotate medians
for i, s in enumerate(order, start=1):
    med_val = df[df.scenario==s]["lcoh_delivered_per_kg"].median()
    ax.text(i, med_val, f" {med_val:.2f}", va="center", fontsize=9, color="black")
ax.set_ylabel("Delivered LCOH (€/kg H₂)")
ax.set_title("Fig. 5 — Africa→Europe delivered green H₂ vs European-produced reference")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(IMG / "fig_africa_vs_eu.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig 6: Tornado — sensitivity of S3 best hex delivered cost to ±20% on key parameters
# ============================================================
sc = dict(SCENARIOS["S3_AFR_DERISKED"])
hex_row = pd.read_csv(WORK/"data"/"hex_final_NA_min.csv").set_index("hex_id").loc[cheap_hex_id]
def delivered(TE_local, sc_local):
    # Temporarily override TE
    saved = dict(M.TE); M.TE.update(TE_local)
    plant = M.hex_lcoh_at_plant(hex_row, sc_local)
    chain = M.add_export_chain(hex_row, sc_local, plant)
    upstream = (plant["lcoh_plant_per_kg"] + chain["c_nh3_synth_per_kg"]
                + chain["c_truck_per_kg"]) * chain["loss_factor"]
    d = (upstream + chain["c_ship_per_kg"]) * chain["yield_factor"] + chain["c_crack_per_kg"]
    M.TE.clear(); M.TE.update(saved)
    return d
base = delivered({}, sc)

params = {
    "PV CAPEX": ("pv_capex_eur_per_kw", 0.20),
    "Wind CAPEX": ("wind_capex_eur_per_kw", 0.20),
    "Electrolyser CAPEX": ("ely_capex_eur_per_kw", 0.20),
    "Electrolyser η (LHV)": ("ely_efficiency_LHV", 0.10),
    "WACC (RE+ely+infra)": ("__wacc__", 0.20),
    "NH₃ shipping €/t/km": ("nh3_ship_eur_per_t_per_km", 0.30),
    "Ship distance km":   ("ship_distance_km", 0.20),
    "NH₃ cracker CAPEX":  ("nh3_crack_capex_eur_per_t_h2_yr", 0.20),
    "NH₃ synthesis CAPEX":("nh3_synth_capex_eur_per_t_nh3_yr", 0.20),
}
records = []
for label, (param, frac) in params.items():
    if param == "__wacc__":
        sc_lo = {k: v*(1-frac) if k.startswith("wacc") else v for k,v in sc.items()}
        sc_hi = {k: v*(1+frac) if k.startswith("wacc") else v for k,v in sc.items()}
        lo = delivered({}, sc_lo); hi = delivered({}, sc_hi)
    else:
        cur = TE[param]
        lo = delivered({param: cur*(1-frac)}, sc)
        hi = delivered({param: cur*(1+frac)}, sc)
    records.append({"param": label, "lo": lo, "hi": hi, "base": base,
                    "delta_lo": lo-base, "delta_hi": hi-base, "frac": frac})

td = pd.DataFrame(records)
td["span"] = td.hi - td.lo
td = td.sort_values("span")
fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(td))
ax.barh(y, td["delta_lo"], color="#2a9d8f", label="Parameter ↓")
ax.barh(y, td["delta_hi"], color="#e76f51", label="Parameter ↑")
ax.set_yticks(y); ax.set_yticklabels(td["param"])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Δ delivered LCOH from de-risked baseline (€/kg H₂)")
ax.set_title(f"Fig. 6 — Tornado sensitivity (S3 de-risked, best hex; baseline = {base:.2f} €/kg)")
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(IMG / "fig_tornado.png", bbox_inches="tight")
plt.close(fig)

# Save tornado data
td.to_csv(OUT/"tornado_S3_best_hex.csv", index=False)

# ============================================================
# Save: Africa vs Europe gap table
# ============================================================
gap_rows = []
for sc_a in ["S1_AFR_BASELINE","S2_AFR_MODERATE","S3_AFR_DERISKED"]:
    a_med = df[df.scenario==sc_a]["lcoh_delivered_per_kg"].median()
    a_min = df[df.scenario==sc_a]["lcoh_delivered_per_kg"].min()
    for sc_e, eu_val in [("S4_EU_LOW_IR", EU_LOW), ("S5_EU_RISING_IR", EU_HIGH)]:
        gap_rows.append({
            "africa_scenario": sc_a,
            "europe_scenario": sc_e,
            "afr_median_delivered": a_med,
            "afr_min_delivered": a_min,
            "eu_lcoh": eu_val,
            "gap_median": a_med - eu_val,
            "gap_min": a_min - eu_val,
            "afr_competitive_at_min": a_min < eu_val,
            "afr_competitive_at_median": a_med < eu_val,
        })
gap = pd.DataFrame(gap_rows)
gap.to_csv(OUT/"africa_vs_europe_gap.csv", index=False)
print(gap.to_string(index=False))

print("\nFigures saved to", IMG)
