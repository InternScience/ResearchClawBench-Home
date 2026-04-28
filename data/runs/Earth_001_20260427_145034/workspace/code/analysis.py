"""
Cloud-seeding records (US, 2000-2025) reproducibility analysis.

Generates tables (outputs/) and figures (report/images/) supporting:
  1) spatial concentration
  2) annual activity dynamics
  3) purpose composition
  4) agent-apparatus deployment patterns
"""
from __future__ import annotations
import os, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "dataset1_cloud_seeding_records" / "cloud_seeding_us_2000_2025.csv"
GEO  = WS / "data" / "dataset1_cloud_seeding_records" / "us_states.geojson"
OUT  = WS / "outputs"; OUT.mkdir(exist_ok=True)
IMG  = WS / "report" / "images"; IMG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# ----------------------------------------------------------------------- load
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().str.lower()

print(f"records: {len(df)}  fields: {list(df.columns)}")

# ----------------------------------------------------- helpers: split lists
def split_multi(series: pd.Series) -> list[str]:
    """Split comma-separated multi-label cells into a flat normalized list."""
    out = []
    for v in series.dropna():
        for tok in re.split(r",\s*", str(v)):
            t = tok.strip().lower()
            if t and t != "nan":
                out.append(t)
    return out

def normalize_purpose(tok: str) -> str:
    t = tok.strip().lower()
    repl = {
        "augment snowpack": "augment snowpack",
        "increase precipitation": "increase precipitation",
        "increase runoff": "increase runoff",
        "suppress hail": "suppress hail",
        "suppress fog": "suppress fog",
        "research": "research",
    }
    return repl.get(t, t)

def normalize_agent(tok: str) -> str:
    t = tok.strip().lower()
    if "hygroscopic" in t: return "hygroscopic agents"
    if t in ("silver iodide", "agi", "silver iodide (agi)"): return "silver iodide"
    return t

# ============================================================ S1: spatial
state_counts = df.groupby("state").size().sort_values(ascending=False)
state_counts.to_csv(OUT / "table_state_counts.csv", header=["records"])
total = int(state_counts.sum())
top_states = state_counts.head(8)
hhi = float(((state_counts / total) ** 2).sum() * 10000)  # Herfindahl-Hirschman index in points
top3_share = float(state_counts.head(3).sum() / total)
top5_share = float(state_counts.head(5).sum() / total)
top8_share = float(state_counts.head(8).sum() / total)

# Active operating years per state
years_per_state = df.groupby("state")["year"].nunique().sort_values(ascending=False)
years_per_state.to_csv(OUT / "table_active_years_per_state.csv", header=["active_years"])

print(f"HHI={hhi:.1f}  top3={top3_share:.2%}  top5={top5_share:.2%}  top8={top8_share:.2%}")

# Bar chart: records per state
fig, ax = plt.subplots(figsize=(8, 4.5))
state_counts.plot(kind="bar", ax=ax, color="#3b6ea5")
for i, v in enumerate(state_counts.values):
    ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Project-year records")
ax.set_xlabel("State")
ax.set_title("Cloud-seeding project-year records by state (2000-2025)")
plt.xticks(rotation=35, ha="right")
fig.savefig(IMG / "fig01_state_records.png")
plt.close(fig)

# ----------- choropleth using us_states.geojson
import json as _json
try:
    with open(GEO) as f:
        gj = _json.load(f)
    # build state name -> count map; geojson likely has 'NAME' or 'name'
    feat0 = gj["features"][0]
    print("geojson props sample:", list(feat0["properties"].keys())[:10])
except Exception as e:
    gj = None
    print("geojson load err:", e)

if gj is not None:
    # detect name field
    name_field = None
    for k in ("NAME", "name", "STATE_NAME", "State", "STUSPS", "state_name"):
        if k in gj["features"][0]["properties"]:
            name_field = k; break
    print("using name field:", name_field)

    # build a count dict; states normalized lowercase
    counts = state_counts.to_dict()
    # also produce top-state summary
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("YlOrRd")
    vmax = float(state_counts.max())
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    drawn = 0
    # draw shapes
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    patches = []
    facecolors = []
    edgecolors = []
    for feat in gj["features"]:
        nm = str(feat["properties"].get(name_field, "")).lower()
        cnt = counts.get(nm, 0)
        color = cmap(norm(cnt)) if cnt > 0 else "#f5f5f5"
        geom = feat["geometry"]
        if geom is None: continue
        polys = geom["coordinates"]
        if geom["type"] == "Polygon":
            polys = [polys]
        for poly in polys:
            ring = poly[0]
            arr = np.array(ring)
            if arr.ndim < 2 or arr.shape[1] < 2: continue
            patches.append(MplPoly(arr, closed=True))
            facecolors.append(color)
            edgecolors.append("#666")
            drawn += 1
    pc = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=0.5)
    ax.add_collection(pc)
    # set extent to CONUS
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 51)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Spatial concentration of US cloud-seeding records (2000-2025)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.01)
    cb.set_label("Project-year records")
    # label top states
    top_state_names = set(state_counts.head(8).index)
    for feat in gj["features"]:
        nm = str(feat["properties"].get(name_field, "")).lower()
        if nm not in top_state_names: continue
        # centroid (rough, mean of first ring)
        geom = feat["geometry"]
        polys = geom["coordinates"]
        if geom["type"] == "Polygon": polys = [polys]
        ring = max(polys, key=lambda p: len(p[0]))[0]
        arr = np.array(ring)
        cx, cy = arr[:,0].mean(), arr[:,1].mean()
        ax.text(cx, cy, f"{nm.title()}\n{counts[nm]}", ha="center", va="center", fontsize=8,
                color="black", weight="bold")
    fig.savefig(IMG / "fig02_state_choropleth.png")
    plt.close(fig)
    print("choropleth drawn polys:", drawn)

# ============================================================ S2: annual
year_counts = df.groupby("year").size().sort_index()
year_counts.to_csv(OUT / "table_yearly_counts.csv", header=["records"])

# season -> primary winter share
def has_winter(s): return "winter" in str(s)
df["winter_flag"] = df["season"].apply(has_winter)
winter_share = df["winter_flag"].mean()
print(f"winter share: {winter_share:.2%}")

# yearly stacked by season type
def season_type(s):
    s = str(s)
    if s == "winter": return "winter only"
    if "winter" in s: return "winter + other"
    return "non-winter"
df["season_type"] = df["season"].apply(season_type)
year_season = df.groupby(["year","season_type"]).size().unstack(fill_value=0)
year_season = year_season.reindex(columns=["winter only","winter + other","non-winter"], fill_value=0)
year_season.to_csv(OUT / "table_yearly_by_season.csv")

fig, ax = plt.subplots(figsize=(9, 4.5))
year_season.plot(kind="bar", stacked=True, ax=ax,
                 color=["#1f4e79","#5b9bd5","#ed7d31"], width=0.85)
ax.set_ylabel("Project-year records")
ax.set_xlabel("Year")
ax.set_title("Annual cloud-seeding activity by season type (2000-2025)")
ax.legend(title="Season composition", loc="upper left")
plt.xticks(rotation=0, fontsize=8)
fig.savefig(IMG / "fig03_yearly_activity.png")
plt.close(fig)

# 3-year rolling line
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(year_counts.index, year_counts.values, marker="o", color="#3b6ea5", label="Annual records")
roll = year_counts.rolling(3, center=True, min_periods=1).mean()
ax.plot(roll.index, roll.values, color="#c0392b", lw=2, label="3-yr rolling mean")
ax.set_xlabel("Year"); ax.set_ylabel("Records")
ax.set_title("Temporal dynamics of US cloud-seeding records, 2000-2025")
ax.grid(alpha=0.3); ax.legend()
fig.savefig(IMG / "fig04_yearly_trend.png")
plt.close(fig)

# Year x state heatmap (top 8)
top8 = state_counts.head(8).index.tolist()
ys = (df[df["state"].isin(top8)]
        .groupby(["state","year"]).size()
        .unstack(fill_value=0)
        .reindex(top8))
ys.to_csv(OUT / "table_state_year_heatmap.csv")
fig, ax = plt.subplots(figsize=(11, 4.2))
im = ax.imshow(ys.values, aspect="auto", cmap="YlGnBu")
ax.set_yticks(range(len(top8))); ax.set_yticklabels([s.title() for s in top8])
ax.set_xticks(range(ys.shape[1])); ax.set_xticklabels(ys.columns, rotation=0, fontsize=8)
ax.set_title("Top-8 states: project-year records by year")
ax.set_xlabel("Year"); ax.set_ylabel("State")
for i in range(ys.shape[0]):
    for j in range(ys.shape[1]):
        v = ys.values[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > ys.values.max()*0.55 else "black", fontsize=7)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Records")
fig.savefig(IMG / "fig05_state_year_heatmap.png")
plt.close(fig)

# ============================================================ S3: purpose
purpose_tokens = [normalize_purpose(t) for t in split_multi(df["purpose"])]
purpose_freq = pd.Series(purpose_tokens).value_counts()
purpose_freq.to_csv(OUT / "table_purpose_tokens.csv", header=["count"])
purpose_share = (purpose_freq / len(df)).round(4)
purpose_share.to_csv(OUT / "table_purpose_share_per_record.csv", header=["share_per_record"])
# raw multi-label categorical
raw_purpose = df["purpose"].value_counts()
raw_purpose.to_csv(OUT / "table_raw_purpose_strings.csv", header=["count"])
print("top purpose tokens:"); print(purpose_freq.head(10))

fig, ax = plt.subplots(figsize=(8, 4.2))
purpose_freq.plot(kind="barh", ax=ax, color="#2ca02c")
ax.invert_yaxis()
for i, v in enumerate(purpose_freq.values):
    ax.text(v + 3, i, str(v), va="center", fontsize=8)
ax.set_xlabel("Mention count (multi-label)")
ax.set_title("Stated purposes of US cloud-seeding records (2000-2025)")
fig.savefig(IMG / "fig06_purpose_tokens.png")
plt.close(fig)

# Purpose by state (presence of dominant purpose tokens)
key_purposes = ["augment snowpack", "increase precipitation", "suppress hail",
                "increase runoff", "suppress fog", "research"]
purpose_by_state = pd.DataFrame(0, index=state_counts.index, columns=key_purposes, dtype=int)
for _, row in df.iterrows():
    toks = [normalize_purpose(t) for t in re.split(r",\s*", str(row["purpose"]))]
    for kp in key_purposes:
        if kp in toks:
            purpose_by_state.loc[row["state"], kp] += 1
purpose_by_state.to_csv(OUT / "table_purpose_by_state.csv")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
im = ax.imshow(purpose_by_state.values, aspect="auto", cmap="Purples")
ax.set_yticks(range(len(purpose_by_state.index)))
ax.set_yticklabels([s.title() for s in purpose_by_state.index])
ax.set_xticks(range(len(key_purposes)))
ax.set_xticklabels(key_purposes, rotation=25, ha="right")
ax.set_title("Purpose composition by state (project-year mentions)")
for i in range(purpose_by_state.shape[0]):
    for j in range(purpose_by_state.shape[1]):
        v = purpose_by_state.values[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > purpose_by_state.values.max()*0.55 else "black")
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Mentions")
fig.savefig(IMG / "fig07_purpose_by_state.png")
plt.close(fig)

# ============================================================ S4: agent / apparatus
agent_tokens = [normalize_agent(t) for t in split_multi(df["agent"])]
agent_freq = pd.Series(agent_tokens).value_counts()
agent_freq.to_csv(OUT / "table_agent_tokens.csv", header=["count"])
print("top agents:"); print(agent_freq.head(10))

# share of records using silver iodide
def has_agi(s): return "silver iodide" in str(s).lower()
agi_share = df["agent"].apply(has_agi).mean()
mono_agi_share = (df["agent"] == "silver iodide").mean()
print(f"AgI presence share: {agi_share:.2%}; AgI-only share: {mono_agi_share:.2%}")

# apparatus distribution
appar_freq = df["apparatus"].fillna("unknown").value_counts()
appar_freq.to_csv(OUT / "table_apparatus.csv", header=["count"])

# agent x apparatus crosstab (using top agents tokens) - but agent is multi-label.
# Build: for each record, derive (apparatus, has_agi, has_hygroscopic, has_dryice/co2, has_other)
def agent_class(a):
    a = str(a).lower()
    classes = []
    if "silver iodide" in a: classes.append("silver iodide")
    if "hygroscopic" in a: classes.append("hygroscopic")
    if "dry ice" in a or "carbon dioxide" in a: classes.append("dry ice / CO2")
    if "ammonium iodide" in a: classes.append("ammonium iodide")
    if "sodium iodide" in a: classes.append("sodium iodide")
    if "calcium chloride" in a: classes.append("calcium chloride")
    if "ionized air" in a: classes.append("ionized air")
    if not classes: classes.append("other / unspecified")
    return classes

class_list = ["silver iodide","hygroscopic","ammonium iodide","sodium iodide",
              "calcium chloride","dry ice / CO2","ionized air","other / unspecified"]

ct = pd.DataFrame(0, index=class_list, columns=appar_freq.index, dtype=int)
for _, row in df.iterrows():
    apparatus = row["apparatus"] if pd.notna(row["apparatus"]) else "unknown"
    for ac in agent_class(row["agent"]):
        if ac in ct.index:
            ct.loc[ac, apparatus] += 1
ct.to_csv(OUT / "table_agent_x_apparatus.csv")

fig, ax = plt.subplots(figsize=(8, 4.6))
im = ax.imshow(ct.values, aspect="auto", cmap="OrRd")
ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, rotation=20, ha="right")
ax.set_title("Agent × apparatus deployment matrix (record counts)")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > ct.values.max()*0.55 else "black")
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Records")
fig.savefig(IMG / "fig08_agent_apparatus.png")
plt.close(fig)

# Apparatus by year (stacked area)
year_app = df.groupby(["year","apparatus"]).size().unstack(fill_value=0)
order = [c for c in ["ground","airborne","ground, airborne"] if c in year_app.columns]
year_app = year_app.reindex(columns=order)
year_app.to_csv(OUT / "table_apparatus_by_year.csv")
fig, ax = plt.subplots(figsize=(9, 4.0))
year_app.plot(kind="area", stacked=True, ax=ax,
              color=["#8c564b","#1f77b4","#9467bd"], alpha=0.85)
ax.set_xlabel("Year"); ax.set_ylabel("Records")
ax.set_title("Apparatus deployment over time")
ax.legend(title="Apparatus")
fig.savefig(IMG / "fig09_apparatus_by_year.png")
plt.close(fig)

# Operator concentration
op_counts = df["operator_affiliation"].value_counts()
op_counts.to_csv(OUT / "table_operator_counts.csv", header=["records"])
op_top10 = op_counts.head(10)
op_share_top5 = op_counts.head(5).sum() / op_counts.sum()
op_share_top10 = op_counts.head(10).sum() / op_counts.sum()
op_hhi = float(((op_counts / op_counts.sum()) ** 2).sum() * 10000)
print(f"operator HHI={op_hhi:.1f}; top5 share={op_share_top5:.2%}; top10 share={op_share_top10:.2%}")

fig, ax = plt.subplots(figsize=(8, 4.5))
op_top10.iloc[::-1].plot(kind="barh", ax=ax, color="#d62728")
for i, v in enumerate(op_top10.iloc[::-1].values):
    ax.text(v + 1, i, str(v), va="center", fontsize=8)
ax.set_xlabel("Records")
ax.set_title("Top-10 operator affiliations (2000-2025)")
fig.savefig(IMG / "fig10_top_operators.png")
plt.close(fig)

# ============================================================ summary stats
summary = {
    "n_records": int(len(df)),
    "n_unique_projects": int(df["project"].nunique()),
    "n_states": int(df["state"].nunique()),
    "year_min": int(df["year"].min()),
    "year_max": int(df["year"].max()),
    "winter_share": round(float(df["winter_flag"].mean()), 4),
    "winter_only_share": round(float((df["season"]=="winter").mean()), 4),
    "agi_presence_share": round(float(agi_share), 4),
    "agi_only_share": round(float(mono_agi_share), 4),
    "ground_only_share": round(float((df["apparatus"]=="ground").mean()), 4),
    "airborne_only_share": round(float((df["apparatus"]=="airborne").mean()), 4),
    "ground_and_airborne_share": round(float((df["apparatus"]=="ground, airborne").mean()), 4),
    "state_HHI": round(hhi, 2),
    "state_top3_share": round(top3_share, 4),
    "state_top5_share": round(top5_share, 4),
    "state_top8_share": round(top8_share, 4),
    "operator_HHI": round(op_hhi, 2),
    "operator_top5_share": round(float(op_share_top5), 4),
    "operator_top10_share": round(float(op_share_top10), 4),
    "top_states_records": state_counts.head(8).astype(int).to_dict(),
    "top_purposes_mentions": purpose_freq.head(8).astype(int).to_dict(),
    "top_agents_mentions": agent_freq.head(8).astype(int).to_dict(),
    "apparatus_distribution": appar_freq.astype(int).to_dict(),
}
with open(OUT / "summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)
print("summary saved.")
print(json.dumps(summary, indent=2))
