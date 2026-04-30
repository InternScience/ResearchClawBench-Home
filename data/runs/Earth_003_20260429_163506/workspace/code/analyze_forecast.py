#!/usr/bin/env python3
"""Reproducible diagnostics for the available ERA5-like input and FuXi one-step forecast.

The workspace does not include trained U-Transformer model weights, a 15-day forecast
sequence, ECMWF ensemble forecasts, or verifying future ERA5 targets. This script
therefore produces structural data summaries and one-step self-consistency diagnostics
for the available NetCDF artifacts, plus figures and protocol files for a faithful
cascade evaluation when complete data are available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "20231012-06_input_netcdf.nc"
FORECAST_FILE = DATA_DIR / "006.nc"


def decode_levels(arr):
    return ["".join(ch.decode() if isinstance(ch, bytes) else str(ch) for ch in row).strip() for row in arr]


def family_level(name):
    for fam in ["MSL", "T2M", "U10", "V10", "TP"]:
        if name == fam:
            return fam, "surface"
    fam = name[0]
    lev = name[1:]
    return fam, lev


def load_nc(path):
    ds = Dataset(path)
    levels = decode_levels(ds.variables["level"][:])
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    data = np.asarray(ds.variables["data"][:], dtype=np.float32)
    times = np.asarray(ds.variables["time"][:])
    time_var = ds.variables["time"]
    dates = [str(x) for x in num2date(times, time_var.units, getattr(time_var, "calendar", "standard"))]
    step = np.asarray(ds.variables["step"][:]).tolist() if "step" in ds.variables else None
    meta = {
        "path": str(path.relative_to(ROOT)),
        "dimensions": {k: len(v) for k, v in ds.dimensions.items()},
        "variables": {
            k: {"dims": list(v.dimensions), "shape": list(v.shape), "dtype": str(v.dtype), "attrs": {a: str(getattr(v, a)) for a in v.ncattrs()}}
            for k, v in ds.variables.items()
        },
        "level_names": levels,
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "time_values": times.tolist(),
        "decoded_times": dates,
        "step_values": step,
        "finite_min": float(np.nanmin(data)),
        "finite_max": float(np.nanmax(data)),
        "nan_count": int(np.isnan(data).sum()),
    }
    ds.close()
    return data, levels, lat, lon, meta


def lat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    w = np.clip(w, 0, None)
    return w / w.mean()


def weighted_stats(field, w_lat):
    # field: lat x lon
    w = w_lat[:, None]
    mean = np.sum(field * w) / np.sum(w * np.ones_like(field))
    var = np.sum(((field - mean) ** 2) * w) / np.sum(w * np.ones_like(field))
    return float(mean), float(np.sqrt(var))


def weighted_rmse(a, b, w_lat):
    diff = a - b
    w = w_lat[:, None]
    return float(np.sqrt(np.sum((diff ** 2) * w) / np.sum(w * np.ones_like(diff))))


def weighted_corr(a, b, w_lat):
    w = w_lat[:, None]
    denom = np.sum(w * np.ones_like(a))
    ma = np.sum(a * w) / denom
    mb = np.sum(b * w) / denom
    aa = a - ma
    bb = b - mb
    cov = np.sum(aa * bb * w) / denom
    va = np.sum(aa * aa * w) / denom
    vb = np.sum(bb * bb * w) / denom
    if va <= 0 or vb <= 0:
        return np.nan
    return float(cov / np.sqrt(va * vb))


def hemisphere_mean(field, lat, w_lat, selector):
    mask = selector(lat)
    w = w_lat[mask, None]
    sub = field[mask, :]
    return float(np.sum(sub * w) / np.sum(w * np.ones_like(sub)))


def main():
    sns.set_theme(style="whitegrid", context="paper")
    input_data, levels, lat, lon, input_meta = load_nc(INPUT_FILE)
    fc_data, fc_levels, fc_lat, fc_lon, fc_meta = load_nc(FORECAST_FILE)
    assert levels == fc_levels
    assert np.allclose(lat, fc_lat) and np.allclose(lon, fc_lon)
    w_lat = lat_weights(lat)

    # Normalize shapes to channel x lat x lon for states.
    state_t0 = input_data[0]
    state_t6_in = input_data[1]
    fc_t12 = fc_data[0, 0]

    metadata = {
        "input": input_meta,
        "forecast": fc_meta,
        "grid_note": "The files are 1 degree grids (181 x 360), despite the task text describing intended 0.25 degree data.",
        "channel_count": len(levels),
        "families": sorted(set(family_level(n)[0] for n in levels)),
    }
    (OUT / "netcdf_metadata.json").write_text(json.dumps(metadata, indent=2))

    rows = []
    transitions = []
    for i, name in enumerate(levels):
        fam, plevel = family_level(name)
        for label, arr in [("input_t0", state_t0), ("input_t6", state_t6_in), ("forecast_t12", fc_t12)]:
            m, s = weighted_stats(arr[i], w_lat)
            rows.append({
                "channel_index": i, "channel": name, "family": fam, "pressure_or_surface": plevel,
                "state": label, "weighted_mean": m, "weighted_std": s,
                "min": float(np.nanmin(arr[i])), "max": float(np.nanmax(arr[i])),
                "nh_weighted_mean": hemisphere_mean(arr[i], lat, w_lat, lambda x: x > 0),
                "tropics_weighted_mean": hemisphere_mean(arr[i], lat, w_lat, lambda x: np.abs(x) <= 20),
                "sh_weighted_mean": hemisphere_mean(arr[i], lat, w_lat, lambda x: x < 0),
            })
        persistence_rmse = weighted_rmse(state_t6_in[i], state_t0[i], w_lat)
        forecast_increment_rmse = weighted_rmse(fc_t12[i], state_t6_in[i], w_lat)
        forecast_vs_t0_rmse = weighted_rmse(fc_t12[i], state_t0[i], w_lat)
        corr_with_t6 = weighted_corr(fc_t12[i], state_t6_in[i], w_lat)
        corr_in_6h = weighted_corr(state_t6_in[i], state_t0[i], w_lat)
        transitions.append({
            "channel_index": i, "channel": name, "family": fam, "pressure_or_surface": plevel,
            "input_6h_increment_rmse": persistence_rmse,
            "forecast_6h_increment_rmse": forecast_increment_rmse,
            "forecast_vs_t0_rmse": forecast_vs_t0_rmse,
            "forecast_vs_input_t6_weighted_corr": corr_with_t6,
            "input_t6_vs_t0_weighted_corr": corr_in_6h,
            "forecast_increment_minus_input_increment_rmse": forecast_increment_rmse - persistence_rmse,
        })

    stats_df = pd.DataFrame(rows)
    trans_df = pd.DataFrame(transitions)
    stats_df.to_csv(OUT / "channel_statistics.csv", index=False)
    trans_df.to_csv(OUT / "transition_metrics.csv", index=False)

    family_summary = trans_df.groupby("family").agg(
        n_channels=("channel", "count"),
        mean_input_increment_rmse=("input_6h_increment_rmse", "mean"),
        mean_forecast_increment_rmse=("forecast_6h_increment_rmse", "mean"),
        mean_forecast_corr_with_t6=("forecast_vs_input_t6_weighted_corr", "mean"),
    ).reset_index()
    family_summary.to_csv(OUT / "family_summary.csv", index=False)

    # Figure 1: data overview.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    counts = pd.Series([family_level(n)[0] for n in levels]).value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color="#4C78A8")
    axes[0].set_title("70-channel variable inventory")
    axes[0].set_ylabel("channels")
    shape_text = (
        f"Input: {tuple(input_data.shape)}\nForecast: {tuple(fc_data.shape)}\n"
        f"Grid: {len(lat)} lat × {len(lon)} lon ({abs(lat[1]-lat[0]):.0f}°)\n"
        f"Input times: {', '.join(input_meta['decoded_times'])}\n"
        f"Forecast step: {fc_meta['step_values']} h from {fc_meta['decoded_times'][0]}"
    )
    axes[1].axis("off")
    axes[1].text(0.02, 0.95, shape_text, va="top", ha="left", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7"))
    fig.suptitle("Available NetCDF artifacts and resolution check")
    fig.savefig(IMG / "figure_01_data_overview.png", dpi=180)
    plt.close(fig)

    # Figure 2: maps for key channels (Z500/T2M/TP) input_t6, forecast, increment.
    key_names = ["Z500", "T2M", "TP"]
    fig, axes = plt.subplots(len(key_names), 3, figsize=(13, 8), constrained_layout=True)
    for r, name in enumerate(key_names):
        idx = levels.index(name)
        fields = [state_t6_in[idx], fc_t12[idx], fc_t12[idx] - state_t6_in[idx]]
        titles = [f"{name} input at 06Z", f"{name} FuXi +6h", f"{name} forecast increment"]
        for c, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[r, c]
            cmap = "RdBu_r" if c < 2 and name != "TP" else ("viridis" if c < 2 else "RdBu_r")
            vmax = np.nanpercentile(np.abs(field), 98) if c == 2 or name != "TP" else np.nanpercentile(field, 98)
            if c == 2 or name != "TP":
                im = ax.imshow(field, origin="upper", extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
            else:
                im = ax.imshow(field, origin="upper", extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Spatial structure of available one-step forecast")
    fig.savefig(IMG / "figure_02_forecast_increment_maps.png", dpi=180)
    plt.close(fig)

    # Figure 3: RMSE by channel/family.
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    plot_df = trans_df.sort_values(["family", "pressure_or_surface", "channel_index"]).copy()
    x = np.arange(len(plot_df))
    ax.plot(x, plot_df["input_6h_increment_rmse"], marker="o", lw=1.2, label="observed input 00Z→06Z increment")
    ax.plot(x, plot_df["forecast_6h_increment_rmse"], marker="s", lw=1.2, label="FuXi 06Z→+6h increment")
    colors = {fam: col for fam, col in zip(sorted(plot_df.family.unique()), sns.color_palette("tab10", n_colors=plot_df.family.nunique()))}
    pos = {idx: k for k, idx in enumerate(plot_df.index)}
    for fam, group in plot_df.groupby("family", sort=False):
        start = min(pos[idx] for idx in group.index) - 0.5
        end = max(pos[idx] for idx in group.index) + 0.5
        ax.axvspan(start, end, alpha=0.06, color=colors[fam])
    ax.set_xticks(x[::2])
    ax.set_xticklabels(plot_df["channel"].iloc[::2], rotation=90, fontsize=7)
    ax.set_ylabel("latitude-weighted RMSE (standardized units)")
    ax.set_title("One-step increment magnitude by channel")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(IMG / "figure_03_channel_rmse_by_family.png", dpi=180)
    plt.close(fig)

    # Figure 4: cascade design schematic using matplotlib primitives.
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.16, 0.28, "ERA5 state t-6h\nERA5 state t"),
        (0.25, 0.55, 0.17, 0.28, "Stage 1\nshort-lead U-Transformer\n6h–1d"),
        (0.48, 0.55, 0.17, 0.28, "Stage 2\nmedium-lead U-Transformer\n1–7d"),
        (0.71, 0.55, 0.17, 0.28, "Stage 3\nlong-lead U-Transformer\n7–15d"),
        (0.37, 0.08, 0.30, 0.25, "Verification protocol\nlat-weighted RMSE/ACC vs ERA5\nlead curves vs ECMWF ensemble mean"),
    ]
    for x0, y0, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fc="#F5F7FA", ec="#333333", lw=1.5))
        ax.text(x0+w/2, y0+h/2, text, ha="center", va="center", fontsize=10)
    for start, end in [((0.18,0.69),(0.25,0.69)), ((0.42,0.69),(0.48,0.69)), ((0.65,0.69),(0.71,0.69)), ((0.795,0.55),(0.52,0.33))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.94, "Faithful cascade forecast design implied by the task", ha="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.40, "Available workspace: input context + one FuXi 6h forecast only; trained cascade and long-lead verification unavailable.", ha="center", fontsize=10, color="#8B0000")
    fig.savefig(IMG / "figure_04_cascade_design.png", dpi=180)
    plt.close(fig)

    # Figure 5: validation matrix.
    validation_items = [
        ("NetCDF readable", 1), ("70 channels", 1), ("two input states", 1), ("forecast step", 1),
        ("0.25° grid", 0), ("15-day sequence", 0), ("trained models", 0), ("future ERA5 truth", 0), ("ECMWF ensemble", 0)
    ]
    val_df = pd.DataFrame(validation_items, columns=["item", "available"])
    val_df.to_csv(OUT / "validation_matrix.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    mat = val_df[["available"]].T
    sns.heatmap(mat, cmap=sns.color_palette(["#D95F02", "#1B9E77"], as_cmap=True), cbar=False,
                xticklabels=val_df["item"], yticklabels=["workspace"], linewidths=1, linecolor="white", ax=ax, vmin=0, vmax=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Validation of task-critical ingredients")
    for j, val in enumerate(val_df["available"]):
        ax.text(j+0.5, 0.5, "yes" if val else "no", ha="center", va="center", color="white", weight="bold")
    fig.savefig(IMG / "figure_05_validation_matrix.png", dpi=180)
    plt.close(fig)

    comparison_protocol = {
        "if_full_data_available": [
            "Run the three U-Transformer cascade autoregressively for 60 six-hour steps from each initialization.",
            "For each lead and channel, compute latitude-weighted RMSE and ACC against verifying ERA5 analysis.",
            "For key variables (Z500, T850/T2M, U10/V10, MSL, TP), compare lead-time curves to ECMWF ensemble mean initialized at the same time.",
            "Report skillful lead time as the last lead where Z500 ACC exceeds 0.6, matching the FengWu literature convention.",
            "Stratify by variable family, pressure level, hemisphere, and tropics/extratropics."
        ],
        "implemented_now": "Only one-step structural and increment diagnostics because data/006.nc contains shape (1,1,70,181,360)."
    }
    (OUT / "comparison_protocol.json").write_text(json.dumps(comparison_protocol, indent=2))

    claim_rows = [
        {"claim": "Workspace input contains two consecutive atmospheric states", "support": "outputs/netcdf_metadata.json input dimensions time=2", "status": "directly verified"},
        {"claim": "Workspace files are 1 degree, not 0.25 degree", "support": "lat/lon dimensions 181 x 360 and 1 degree coordinate spacing", "status": "directly verified"},
        {"claim": "Forecast artifact is a single 6-hour FuXi output step", "support": "outputs/netcdf_metadata.json forecast dimensions time=1, step=1, step_values=[6]", "status": "directly verified"},
        {"claim": "Latitude-weighted RMSE/ACC are appropriate core metrics", "support": "outputs/related_work_contract.json extracts FourCastNet/FengWu evaluation practice", "status": "related-work supported"},
        {"claim": "15-day ECMWF-comparable skill cannot be concluded", "support": "outputs/validation_matrix.csv shows no 15-day sequence, truth, ECMWF ensemble, or trained models", "status": "limitation"},
    ]
    pd.DataFrame(claim_rows).to_csv(OUT / "claim_recovery_table.csv", index=False)

    # Update artifact inventory statuses.
    inventory = json.loads((OUT / "target_artifact_inventory.json").read_text())
    for section in ["primary_artifacts", "figure_artifacts"]:
        for item in inventory[section]:
            p = ROOT / item["path"]
            item["status"] = "satisfied" if p.exists() else "unsatisfied"
            if not p.exists():
                item["reason"] = "file not produced"
    (OUT / "target_artifact_inventory.json").write_text(json.dumps(inventory, indent=2))

    summary = {
        "n_channels": len(levels),
        "grid_shape": [len(lat), len(lon)],
        "grid_spacing_deg": float(abs(lat[1] - lat[0])),
        "forecast_steps_available": int(fc_data.shape[1]),
        "mean_forecast_increment_rmse": float(trans_df["forecast_6h_increment_rmse"].mean()),
        "mean_input_increment_rmse": float(trans_df["input_6h_increment_rmse"].mean()),
        "mean_forecast_corr_with_input_t6": float(trans_df["forecast_vs_input_t6_weighted_corr"].mean()),
        "top5_forecast_increment_channels": trans_df.sort_values("forecast_6h_increment_rmse", ascending=False).head(5)[["channel", "forecast_6h_increment_rmse"]].to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
