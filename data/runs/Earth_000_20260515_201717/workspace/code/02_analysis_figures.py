#!/usr/bin/env python3
"""Main analysis and figure generation for GlaMBIE glacier mass change."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path("data/glambie/results/calendar_years")
OUTPUT_DIR = Path("outputs")
IMG_DIR = Path("report/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

REGION_NAMES = {
    "1_alaska": "Alaska", "2_western_canada_us": "W. Canada & US",
    "3_arctic_canada_north": "Arctic Canada N", "4_arctic_canada_south": "Arctic Canada S",
    "5_greenland_periphery": "Greenland Periph.", "6_iceland": "Iceland",
    "7_svalbard": "Svalbard", "8_scandinavia": "Scandinavia",
    "9_russian_arctic": "Russian Arctic", "10_north_asia": "North Asia",
    "11_central_europe": "Central Europe", "12_caucasus_middle_east": "Caucasus & ME",
    "13_central_asia": "Central Asia", "14_south_asia_west": "S. Asia West",
    "15_south_asia_east": "S. Asia East", "16_low_latitudes": "Low Latitudes",
    "17_southern_andes": "Southern Andes", "18_new_zealand": "New Zealand",
    "19_antarctic_and_subantarctic": "Antarctic & Subantarctic",
}

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
                      'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})

def load_all_results():
    dfs = {}
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        key = f.stem
        df["region_name"] = "Global" if key == "0_global" else REGION_NAMES.get(key, key)
        dfs[key] = df
    return dfs

# ========== Figure 1: Global cumulative mass change ==========
def fig_global_timeseries(dfs):
    df = dfs["0_global"].copy()
    mask = (df["start_dates"] >= 2000) & (df["start_dates"] < 2024)
    df = df[mask]
    
    df["cum_gt"] = df["combined_gt"].cumsum()
    df["cum_gt_err"] = np.sqrt((df["combined_gt_errors"]**2).cumsum())
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Panel A: Annual rates
    ax = axes[0]
    ax.fill_between(df["end_dates"], -df["combined_gt_errors"], df["combined_gt_errors"],
                     alpha=0.2, color='steelblue')
    ax.bar(df["end_dates"]-0.4, df["combined_gt"], width=0.8, color='steelblue', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel("Mass Change (Gt yr⁻¹)")
    ax.set_title("Annual Global Glacier Mass Change (2000–2023)")
    ax.grid(axis='y', alpha=0.3)
    mean_rate = df["combined_gt"].mean()
    ax.axhline(y=mean_rate, color='darkred', linestyle='--', linewidth=1,
               label=f'Mean: {mean_rate:.1f} Gt yr⁻¹')
    ax.legend(loc='lower left')
    
    # Panel B: Cumulative
    ax = axes[1]
    ax.fill_between(df["end_dates"], 
                     df["cum_gt"] - df["cum_gt_err"],
                     df["cum_gt"] + df["cum_gt_err"],
                     alpha=0.2, color='darkred')
    ax.plot(df["end_dates"], df["cum_gt"], 'o-', color='darkred', linewidth=2, markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Mass Change (Gt)")
    ax.set_title("Cumulative Global Glacier Mass Change (2000–2023)")
    ax.grid(alpha=0.3)
    
    total = df["cum_gt"].iloc[-1]
    total_err = df["cum_gt_err"].iloc[-1]
    ax.text(0.02, 0.05, f'Total: {total:.0f} ± {total_err:.0f} Gt\n'
            f'({total/362.5:.1f} ± {total_err/362.5:.1f} mm SLE)',
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig1_global_timeseries.png")
    plt.close()
    print("Figure 1 saved.")

# ========== Figure 2: Regional mass change comparison ==========
def fig_regional_comparison(dfs):
    region_data = []
    for key, df in dfs.items():
        if key == "0_global":
            continue
        mask = (df["start_dates"] >= 2000) & (df["start_dates"] < 2024)
        d = df[mask]
        total_gt = d["combined_gt"].sum()
        total_err = np.sqrt((d["combined_gt_errors"]**2).sum())
        mean_mwe = d["combined_mwe"].mean()
        area = d["glacier_area"].mean()
        region_data.append({
            "region": d["region_name"].iloc[0],
            "total_gt": total_gt,
            "total_err": total_err,
            "mean_mwe": mean_mwe,
            "area": area,
            "key": key
        })
    
    rdf = pd.DataFrame(region_data).sort_values("total_gt")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    
    # Panel A: Total mass change bar chart
    ax = axes[0]
    colors = ['#d73027' if x < -300 else '#fc8d59' if x < -100 else '#fee090' for x in rdf["total_gt"]]
    bars = ax.barh(range(len(rdf)), rdf["total_gt"], xerr=rdf["total_err"], 
                   color=colors, edgecolor='gray', linewidth=0.5, capsize=2)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf["region"])
    ax.set_xlabel("Total Mass Change 2000–2023 (Gt)")
    ax.set_title("Regional Total Glacier Mass Change")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    for i, (v, e) in enumerate(zip(rdf["total_gt"], rdf["total_err"])):
        ax.text(v - 10, i, f'{v:.0f}', va='center', ha='right' if v < 0 else 'left', fontsize=7)
    
    # Panel B: Specific mass change
    ax = axes[1]
    rdf2 = rdf.sort_values("mean_mwe")
    colors2 = ['#d73027' if x < -1.0 else '#fc8d59' if x < -0.5 else '#fee090' for x in rdf2["mean_mwe"]]
    ax.barh(range(len(rdf2)), rdf2["mean_mwe"], color=colors2, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(rdf2)))
    ax.set_yticklabels(rdf2["region"])
    ax.set_xlabel("Mean Specific Mass Change (m w.e. yr⁻¹)")
    ax.set_title("Regional Mean Specific Mass Change Rate")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig2_regional_comparison.png")
    plt.close()
    
    rdf.to_csv(OUTPUT_DIR / "regional_summary.csv", index=False)
    print("Figure 2 saved.")

# ========== Figure 3: Regional stacked area time series ==========
def fig_regional_timeseries(dfs):
    years = np.arange(2000, 2024)
    region_keys = sorted([k for k in dfs.keys() if k != "0_global"])
    
    data_matrix = np.zeros((len(years), len(region_keys)))
    for j, rk in enumerate(region_keys):
        df = dfs[rk]
        for i, yr in enumerate(years):
            row = df[df["end_dates"] == float(yr)]
            if len(row) > 0:
                data_matrix[i, j] = row["combined_gt"].values[0]
    
    cumsum = np.cumsum(data_matrix, axis=0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(region_keys)))
    
    # Sort by total contribution
    total_contrib = data_matrix.sum(axis=0)
    order = np.argsort(total_contrib)
    
    bottom = np.zeros(len(years))
    for j in order:
        ax.fill_between(years, bottom, bottom + cumsum[:, j] - (cumsum[:, j-1] if j > 0 else 0) if False else bottom + data_matrix[:, j],
                         alpha=0.8)
        bottom += data_matrix[:, j]
    
    # Simplified: plot cumulative sum
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(years))
    labels = []
    for j in order:
        vals = data_matrix[:, j]
        label = REGION_NAMES.get(region_keys[j], region_keys[j])
        ax.fill_between(years, bottom, bottom + vals, alpha=0.8, label=label)
        bottom += vals
        labels.append(label)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mass Change (Gt yr⁻¹)")
    ax.set_title("Regional Contributions to Global Glacier Mass Change")
    ax.legend(loc='lower left', ncol=3, fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig3_regional_stacked.png")
    plt.close()
    print("Figure 3 saved.")

# ========== Figure 4: Regional cumulative time series (faceted) ==========
def fig_regional_cumulative_facets(dfs):
    region_keys = sorted([k for k in dfs.keys() if k != "0_global"])
    n_regions = len(region_keys)
    ncols = 5
    nrows = (n_regions + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14))
    axes_flat = axes.flatten()
    
    for i, rk in enumerate(region_keys):
        ax = axes_flat[i]
        df = dfs[rk]
        mask = (df["start_dates"] >= 2000) & (df["start_dates"] < 2024)
        d = df[mask].copy()
        d["cum_gt"] = d["combined_gt"].cumsum()
        d["cum_err"] = np.sqrt((d["combined_gt_errors"]**2).cumsum())
        
        ax.fill_between(d["end_dates"], d["cum_gt"] - d["cum_err"], d["cum_gt"] + d["cum_err"],
                         alpha=0.2, color='steelblue')
        ax.plot(d["end_dates"], d["cum_gt"], 'o-', color='steelblue', linewidth=1.5, markersize=3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(REGION_NAMES.get(rk, rk), fontsize=9)
        ax.grid(alpha=0.3)
        
        total = d["cum_gt"].iloc[-1]
        ax.text(0.05, 0.05, f'{total:.0f} Gt', transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    fig.supxlabel("Year", fontsize=12)
    fig.supylabel("Cumulative Mass Change (Gt)", fontsize=12)
    fig.suptitle("Regional Cumulative Glacier Mass Change (2000–2023)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig4_regional_cumulative_facets.png")
    plt.close()
    print("Figure 4 saved.")

# ========== Figure 5: Method coverage analysis ==========
def fig_method_coverage():
    input_df = pd.read_csv(OUTPUT_DIR / "compiled_input_data.csv")
    
    method_order = ["Glaciological", "DEM Differencing", "Altimetry", "Gravimetry", "Combined/Hybrid"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Files per method per region
    ax = axes[0]
    pivot = input_df.groupby(["region_name", "method_name"])["source_file"].nunique().unstack(fill_value=0)
    pivot = pivot.reindex(columns=method_order, fill_value=0)
    
    im = ax.imshow(pivot.T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=90, fontsize=7)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=8)
    ax.set_title("Number of Datasets per Method and Region")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    for i in range(len(method_order)):
        for j in range(len(pivot.index)):
            val = pivot.iloc[j, i]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center', fontsize=7,
                        color='white' if val > 3 else 'black')
    
    # Panel B: Temporal coverage
    ax = axes[1]
    input_df["mid_year"] = (input_df["start_dates"] + input_df["end_dates"]) / 2
    for i, method in enumerate(method_order):
        subset = input_df[input_df["method_name"] == method]
        if len(subset) > 0:
            years_covered = set()
            for _, row in subset.iterrows():
                sy = int(np.floor(row["start_dates"]))
                ey = int(np.ceil(row["end_dates"]))
                years_covered.update(range(max(1950, sy), min(2025, ey+1)))
            years_sorted = sorted(years_covered)
            if years_sorted:
                ax.scatter(years_sorted, [i]*len(years_sorted), marker='|', s=100, 
                          label=method)
    
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_xlabel("Year")
    ax.set_title("Temporal Coverage by Observation Method")
    ax.set_xlim(1945, 2025)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig5_method_coverage.png")
    plt.close()
    print("Figure 5 saved.")

# ========== Figure 6: Uncertainty evolution ==========
def fig_uncertainty(dfs):
    df = dfs["0_global"].copy()
    mask = (df["start_dates"] >= 2000) & (df["start_dates"] < 2024)
    df = df[mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Relative uncertainty over time
    ax = axes[0]
    rel_err = np.abs(df["combined_gt_errors"].values / df["combined_gt"].values) * 100
    ax.bar(df["end_dates"]-0.4, rel_err, width=0.8, color='darkorange', alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Uncertainty (%)")
    ax.set_title("Annual Relative Uncertainty in Global Mass Change")
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=np.mean(rel_err), color='darkred', linestyle='--', 
               label=f'Mean: {np.mean(rel_err):.1f}%')
    ax.legend()
    
    # Panel B: Regional uncertainty
    ax = axes[1]
    region_errs = []
    for key, df_r in dfs.items():
        if key == "0_global":
            continue
        m = (df_r["start_dates"] >= 2000) & (df_r["start_dates"] < 2024)
        d = df_r[m]
        mean_abs_rate = np.abs(d["combined_gt"]).mean()
        mean_err = d["combined_gt_errors"].mean()
        region_errs.append({
            "region": REGION_NAMES.get(key, key),
            "mean_abs_rate": mean_abs_rate,
            "mean_err": mean_err,
            "rel_err_pct": (mean_err / mean_abs_rate * 100) if mean_abs_rate > 0 else 0
        })
    
    redf = pd.DataFrame(region_errs).sort_values("rel_err_pct")
    ax.barh(range(len(redf)), redf["rel_err_pct"], color='darkorange', alpha=0.7)
    ax.set_yticks(range(len(redf)))
    ax.set_yticklabels(redf["region"], fontsize=8)
    ax.set_xlabel("Mean Relative Uncertainty (%)")
    ax.set_title("Regional Mean Relative Uncertainty")
    ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig6_uncertainty.png")
    plt.close()
    print("Figure 6 saved.")

# ========== Figure 7: Data availability heatmap ==========
def fig_data_availability():
    input_df = pd.read_csv(OUTPUT_DIR / "compiled_input_data.csv")
    
    input_df["start_year"] = input_df["start_dates"].apply(lambda x: int(np.floor(x)))
    input_df["end_year"] = input_df["end_dates"].apply(lambda x: int(np.ceil(x)))
    
    years = range(1950, 2025)
    region_order = sorted(REGION_NAMES.keys())
    
    matrix = np.zeros((len(region_order), len(years)))
    for i, rk in enumerate(region_order):
        subset = input_df[input_df["region_key"] == rk]
        for j, yr in enumerate(years):
            count = 0
            for _, row in subset.iterrows():
                if row["start_year"] <= yr <= row["end_year"]:
                    count += 1
            matrix[i, j] = count
    
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels([REGION_NAMES[rk] for rk in region_order], fontsize=8)
    ax.set_xticks(range(0, len(years), 5))
    ax.set_xticklabels([str(y) for y in years][::5])
    ax.set_xlabel("Year")
    ax.set_title("Data Availability: Number of Datasets per Region and Year")
    plt.colorbar(im, ax=ax, label="Number of Datasets")
    
    fig.tight_layout()
    fig.savefig(IMG_DIR / "fig7_data_availability.png")
    plt.close()
    print("Figure 7 saved.")


if __name__ == "__main__":
    print("Loading results...")
    dfs = load_all_results()
    print(f"Loaded {len(dfs)} result files.")
    
    print("Generating figures...")
    fig_global_timeseries(dfs)
    fig_regional_comparison(dfs)
    fig_regional_timeseries(dfs)
    fig_regional_cumulative_facets(dfs)
    fig_method_coverage()
    fig_uncertainty(dfs)
    fig_data_availability()
    
    # Export final time series tables
    global_df = dfs["0_global"].copy()
    mask = (global_df["start_dates"] >= 2000) & (global_df["start_dates"] < 2024)
    global_df = global_df[mask]
    
    # Export as requested: 2000-2023, annual resolution, m.w.e. and Gt
    ts_export = global_df[["start_dates", "end_dates", "combined_mwe", "combined_mwe_errors",
                            "combined_gt", "combined_gt_errors", "glacier_area"]].copy()
    ts_export.columns = ["year_start", "year_end", "specific_mass_change_mwe", 
                          "specific_mass_change_mwe_error", "total_mass_change_Gt",
                          "total_mass_change_Gt_error", "glacier_area_km2"]
    ts_export.to_csv(OUTPUT_DIR / "global_annual_timeseries_2000_2023.csv", index=False, float_format='%.3f')
    
    # Regional export
    for key, df in dfs.items():
        if key == "0_global":
            continue
        d = df[(df["start_dates"] >= 2000) & (df["start_dates"] < 2024)].copy()
        d_out = d[["start_dates", "end_dates", "region", "combined_mwe", "combined_mwe_errors",
                    "combined_gt", "combined_gt_errors", "glacier_area"]]
        d_out.to_csv(OUTPUT_DIR / f"regional_annual_timeseries_{key}.csv", index=False, float_format='%.3f')
    
    print("\nAll figures and data exported.")
