"""
GlaMBIE Glacier Mass Change Analysis
=====================================
Reconciles diverse observational methods to produce consistent regional and
global glacier mass change time series (2000-2023) with uncertainties.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07"
INPUT_DIR = DATA / "glambie_input_20240716"
RESULTS_CAL = DATA / "glambie_results_20240716" / "calendar_years"
RESULTS_HYDRO = DATA / "glambie_results_20240716" / "hydrological_years"
OUT = BASE / "outputs"
IMG = BASE / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

REGION_NAMES = {
    "1_alaska": "Alaska",
    "2_western_canada_us": "Western Canada & US",
    "3_arctic_canada_north": "Arctic Canada North",
    "4_arctic_canada_south": "Arctic Canada South",
    "5_greenland_periphery": "Greenland Periphery",
    "6_iceland": "Iceland",
    "7_svalbard": "Svalbard",
    "8_scandinavia": "Scandinavia",
    "9_russian_arctic": "Russian Arctic",
    "10_north_asia": "North Asia",
    "11_central_europe": "Central Europe",
    "12_caucasus_middle_east": "Caucasus & Middle East",
    "13_central_asia": "Central Asia",
    "14_south_asia_west": "South Asia West",
    "15_south_asia_east": "South Asia East",
    "16_low_latitudes": "Low Latitudes",
    "17_southern_andes": "Southern Andes",
    "18_new_zealand": "New Zealand",
    "19_antarctic_and_subantarctic": "Antarctic & Subantarctic",
}

# ── 1. Load calendar-year results ────────────────────────────────────────────
def load_calendar_results():
    """Load all calendar-year regional + global results."""
    frames = {}
    for fp in sorted(RESULTS_CAL.glob("*.csv")):
        df = pd.read_csv(fp)
        key = fp.stem  # e.g. "0_global", "1_alaska"
        frames[key] = df
    return frames

def load_hydro_results():
    """Load hydrological-year results (with per-data-group columns)."""
    frames = {}
    for fp in sorted(RESULTS_HYDRO.glob("*.csv")):
        df = pd.read_csv(fp)
        frames[fp.stem] = df
    return frames

def load_input_data():
    """Load all input CSV files, tagging each with region and method."""
    rows = []
    for region_dir in sorted(INPUT_DIR.iterdir()):
        if not region_dir.is_dir():
            continue
        region = region_dir.name
        for fp in sorted(region_dir.glob("*.csv")):
            fname = fp.stem
            # Determine method from filename
            for method in ["glaciological", "demdiff", "altimetry", "gravimetry", "combined"]:
                if f"_{method}_" in fname:
                    break
            else:
                method = "unknown"
            df = pd.read_csv(fp)
            df["region"] = region
            df["method"] = method
            df["source"] = fname
            rows.append(df)
    return pd.concat(rows, ignore_index=True)

# ── 2. Analysis functions ────────────────────────────────────────────────────
def compute_cumulative(df, col="combined_gt"):
    """Compute cumulative mass change from annual increments."""
    return np.cumsum(df[col].values)

def compute_rates(df, col="combined_gt", window=5):
    """Compute rolling mean rate of mass change (Gt/yr)."""
    return df[col].rolling(window, center=True).mean()

# ── 3. Figures ───────────────────────────────────────────────────────────────

def fig1_global_timeseries(cal):
    """Figure 1: Global annual and cumulative mass change."""
    gl = cal["0_global"].copy()
    gl["year"] = gl["start_dates"].astype(int)
    gl["cumul_gt"] = compute_cumulative(gl)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Annual
    ax = axes[0]
    ax.bar(gl["year"], gl["combined_gt"], color="steelblue", alpha=0.8,
           yerr=gl["combined_gt_errors"], capsize=2, ecolor="grey", label="Annual mass change")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Mass change (Gt yr$^{-1}$)")
    ax.set_title("Global Glacier Mass Change 2000–2023 (Calendar Years)")
    ax.legend()

    # Cumulative
    ax = axes[1]
    cum_err = np.sqrt(np.cumsum(gl["combined_gt_errors"].values**2))
    ax.fill_between(gl["year"], gl["cumul_gt"] - cum_err,
                    gl["cumul_gt"] + cum_err, alpha=0.25, color="steelblue")
    ax.plot(gl["year"], gl["cumul_gt"], "o-", color="steelblue", label="Cumulative mass change")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative mass change (Gt)")
    ax.set_xlabel("Year")
    ax.legend()

    plt.tight_layout()
    fig.savefig(IMG / "fig1_global_timeseries.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig1_global_timeseries.png")

def fig2_regional_cumulative(cal):
    """Figure 2: Cumulative mass change for all 19 regions."""
    fig, axes = plt.subplots(4, 5, figsize=(22, 16), sharex=True)
    axes = axes.flat

    for i, (key, name) in enumerate(REGION_NAMES.items()):
        ax = axes[i]
        df = cal[key].copy()
        df["year"] = df["start_dates"].astype(int)
        cum = compute_cumulative(df)
        cum_err = np.sqrt(np.cumsum(df["combined_gt_errors"].values**2))
        ax.fill_between(df["year"], cum - cum_err, cum + cum_err,
                        alpha=0.25, color="steelblue")
        ax.plot(df["year"], cum, "-", color="steelblue", lw=1.5)
        ax.set_title(name, fontsize=9)
        ax.axhline(0, color="black", lw=0.3)
        if i >= 15:
            ax.set_xlabel("Year", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide last empty subplot
    axes[19].set_visible(False)
    fig.suptitle("Cumulative Glacier Mass Change by Region (Gt), 2000–2023", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(IMG / "fig2_regional_cumulative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig2_regional_cumulative.png")

def fig3_regional_bar_ranking(cal):
    """Figure 3: Mean annual mass change ranked across regions."""
    records = []
    for key, name in REGION_NAMES.items():
        df = cal[key]
        mean_gt = df["combined_gt"].mean()
        mean_err = np.sqrt(np.sum(df["combined_gt_errors"]**2)) / len(df)
        records.append({"region": name, "mean_gt": mean_gt, "mean_err": mean_err})
    rdf = pd.DataFrame(records).sort_values("mean_gt")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in rdf["mean_gt"]]
    ax.barh(rdf["region"], rdf["mean_gt"], xerr=rdf["mean_err"],
            color=colors, capsize=3, ecolor="grey")
    ax.set_xlabel("Mean annual mass change (Gt yr$^{-1}$)")
    ax.set_title("Mean Annual Glacier Mass Change by Region (2000–2023)")
    ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    fig.savefig(IMG / "fig3_regional_ranking.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig3_regional_ranking.png")

def fig4_datagroup_comparison(hydro):
    """Figure 4: Comparison of data groups (altimetry, gravimetry, demdiff+glaciological) for selected regions."""
    sel_regions = ["1_alaska", "13_central_asia", "7_svalbard", "5_greenland_periphery",
                   "17_southern_andes", "19_antarctic_and_subantarctic"]
    groups = [
        ("altimetry_gt", "altimetry_gt_errors", "Altimetry", "tab:blue"),
        ("gravimetry_gt", "gravimetry_gt_errors", "Gravimetry", "tab:green"),
        ("demdiff_and_glaciological_gt", "demdiff_and_glaciological_gt_errors", "DEM diff + Glaciol.", "tab:orange"),
        ("combined_gt", "combined_gt_errors", "Combined", "black"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flat
    for i, rkey in enumerate(sel_regions):
        ax = axes[i]
        df = hydro[rkey].copy()
        df["year"] = df["start_dates"].astype(float)
        for col, ecol, label, color in groups:
            if col not in df.columns:
                continue
            mask = df[col].notna()
            ax.plot(df.loc[mask, "year"], df.loc[mask, col], "-", color=color, lw=1.2 if label != "Combined" else 2, label=label, alpha=0.8 if label != "Combined" else 1.0)
        ax.axhline(0, color="black", lw=0.3)
        ax.set_title(REGION_NAMES.get(rkey, rkey), fontsize=10)
        ax.set_ylabel("Mass change (Gt yr$^{-1}$)")
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Comparison of Observation Groups per Region (Hydrological Years)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(IMG / "fig4_datagroup_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig4_datagroup_comparison.png")

def fig5_global_area_evolution(cal):
    """Figure 5: Global glacier area evolution."""
    gl = cal["0_global"].copy()
    gl["year"] = gl["start_dates"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gl["year"], gl["glacier_area"] / 1e3, "s-", color="teal", markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Glacier area (×10$^3$ km$^2$)")
    ax.set_title("Global Glacier Area Evolution, 2000–2023")
    plt.tight_layout()
    fig.savefig(IMG / "fig5_global_area.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig5_global_area.png")

def fig6_specific_mass_change(cal):
    """Figure 6: Regional specific mass change (m w.e. yr-1) heatmap."""
    years = cal["0_global"]["start_dates"].astype(int).values
    regions = list(REGION_NAMES.keys())
    mat = np.zeros((len(regions), len(years)))
    for i, rk in enumerate(regions):
        mat[i, :] = cal[rk]["combined_mwe"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-2, vmax=1,
                   extent=[years[0]-0.5, years[-1]+0.5, len(regions)-0.5, -0.5])
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels([REGION_NAMES[r] for r in regions], fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("Specific Mass Change by Region (m w.e. yr$^{-1}$)")
    plt.colorbar(im, ax=ax, label="m w.e. yr$^{-1}$", shrink=0.8)
    plt.tight_layout()
    fig.savefig(IMG / "fig6_specific_mass_heatmap.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig6_specific_mass_heatmap.png")

def fig7_input_data_overview(inputs):
    """Figure 7: Number of input datasets per region and method."""
    ct = inputs.groupby(["region", "method"])["source"].nunique().reset_index()
    ct.columns = ["region", "method", "count"]
    pivot = ct.pivot_table(index="region", columns="method", values="count", fill_value=0)
    # reorder rows
    ordered = [k for k in REGION_NAMES.keys() if k in pivot.index]
    pivot = pivot.reindex(ordered)
    pivot.index = [REGION_NAMES.get(r, r) for r in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    pivot.plot.barh(stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("Number of input datasets")
    ax.set_title("Input Dataset Count by Region and Observation Method")
    ax.legend(title="Method", fontsize=8)
    plt.tight_layout()
    fig.savefig(IMG / "fig7_input_overview.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig7_input_overview.png")

def fig8_sea_level_contribution(cal):
    """Figure 8: Glacier contribution to sea level rise."""
    OCEAN_AREA_M2 = 3.625e14  # m^2
    GT_TO_M3 = 1e9  # 1 Gt water ≈ 1e9 m^3
    gl = cal["0_global"].copy()
    gl["year"] = gl["start_dates"].astype(int)
    gl["slr_mm"] = -gl["combined_gt"] * GT_TO_M3 / OCEAN_AREA_M2 * 1e3  # mm
    gl["cumul_slr"] = np.cumsum(gl["slr_mm"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(gl["year"], gl["slr_mm"], color="coral", alpha=0.8, label="Annual SLR contribution")
    ax2 = ax.twinx()
    ax2.plot(gl["year"], gl["cumul_slr"], "k-o", markersize=4, label="Cumulative SLR")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual SLR contribution (mm yr$^{-1}$)")
    ax2.set_ylabel("Cumulative SLR contribution (mm)")
    ax.set_title("Global Glacier Contribution to Sea Level Rise, 2000–2023")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    fig.savefig(IMG / "fig8_sea_level_contribution.png", dpi=200)
    plt.close(fig)
    print("  [saved] fig8_sea_level_contribution.png")


# ── 4. Summary statistics ───────────────────────────────────────────────────
def compute_summary(cal):
    """Compute summary statistics and save to CSV."""
    gl = cal["0_global"]
    total_loss = gl["combined_gt"].sum()
    total_err = np.sqrt(np.sum(gl["combined_gt_errors"]**2))
    mean_annual = gl["combined_gt"].mean()
    mean_annual_err = total_err / len(gl)

    # Per-region totals
    records = []
    for key, name in REGION_NAMES.items():
        df = cal[key]
        rtotal = df["combined_gt"].sum()
        rerr = np.sqrt(np.sum(df["combined_gt_errors"]**2))
        mean_mwe = df["combined_mwe"].mean()
        records.append({
            "Region": name,
            "Total mass change (Gt)": round(rtotal, 1),
            "Uncertainty (Gt)": round(rerr, 1),
            "Mean specific mass change (m w.e./yr)": round(mean_mwe, 3),
            "Glacier area 2000 (km2)": round(df["glacier_area"].iloc[0], 0),
            "Glacier area 2023 (km2)": round(df["glacier_area"].iloc[-1], 0),
            "Fraction of global loss (%)": round(rtotal / total_loss * 100, 1),
        })
    rdf = pd.DataFrame(records).sort_values("Total mass change (Gt)")
    rdf.to_csv(OUT / "regional_summary.csv", index=False)

    summary = {
        "Total global mass loss 2000-2023 (Gt)": round(total_loss, 1),
        "Total uncertainty (Gt)": round(total_err, 1),
        "Mean annual loss (Gt/yr)": round(mean_annual, 1),
        "Mean annual uncertainty (Gt/yr)": round(mean_annual_err, 1),
        "Global glacier area 2000 (km2)": round(gl["glacier_area"].iloc[0], 0),
        "Global glacier area 2023 (km2)": round(gl["glacier_area"].iloc[-1], 0),
    }
    pd.Series(summary).to_csv(OUT / "global_summary.csv")

    # Decadal comparison
    d1 = gl[(gl["start_dates"] >= 2000) & (gl["start_dates"] < 2010)]
    d2 = gl[(gl["start_dates"] >= 2010) & (gl["start_dates"] < 2020)]
    d3 = gl[gl["start_dates"] >= 2020]
    decadal = pd.DataFrame({
        "Period": ["2000-2009", "2010-2019", "2020-2023"],
        "Mean annual loss (Gt/yr)": [round(d1["combined_gt"].mean(), 1),
                                      round(d2["combined_gt"].mean(), 1),
                                      round(d3["combined_gt"].mean(), 1)],
        "Mean specific change (m w.e./yr)": [round(d1["combined_mwe"].mean(), 3),
                                              round(d2["combined_mwe"].mean(), 3),
                                              round(d3["combined_mwe"].mean(), 3)],
    })
    decadal.to_csv(OUT / "decadal_comparison.csv", index=False)

    return summary, rdf, decadal


# ── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    cal = load_calendar_results()
    hydro = load_hydro_results()
    inputs = load_input_data()

    print("Computing summary statistics...")
    summary, rdf, decadal = compute_summary(cal)
    print(f"  Global total mass loss: {summary['Total global mass loss 2000-2023 (Gt)']} ± {summary['Total uncertainty (Gt)']} Gt")
    print(f"  Mean annual loss: {summary['Mean annual loss (Gt/yr)']} ± {summary['Mean annual uncertainty (Gt/yr)']} Gt/yr")
    print("\nDecadal comparison:")
    print(decadal.to_string(index=False))

    print("\nGenerating figures...")
    fig1_global_timeseries(cal)
    fig2_regional_cumulative(cal)
    fig3_regional_bar_ranking(cal)
    fig4_datagroup_comparison(hydro)
    fig5_global_area_evolution(cal)
    fig6_specific_mass_change(cal)
    fig7_input_data_overview(inputs)
    fig8_sea_level_contribution(cal)

    print("\nAll outputs saved.")
    print(f"  Figures: {IMG}")
    print(f"  Summaries: {OUT}")

if __name__ == "__main__":
    main()
