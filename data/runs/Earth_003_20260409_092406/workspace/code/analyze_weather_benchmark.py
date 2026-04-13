import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def classify_variable(level_name: str) -> str:
    surface = {"T2M", "U10", "V10", "MSL", "TP"}
    if level_name in surface:
        return "surface"
    return level_name.rstrip("0123456789")


def weighted_mean(field: np.ndarray, weights_2d: np.ndarray) -> float:
    return float(np.sum(field * weights_2d) / np.sum(weights_2d))


def weighted_rmse(diff: np.ndarray, weights_2d: np.ndarray) -> float:
    return float(np.sqrt(np.sum((diff ** 2) * weights_2d) / np.sum(weights_2d)))


def weighted_mae(diff: np.ndarray, weights_2d: np.ndarray) -> float:
    return float(np.sum(np.abs(diff) * weights_2d) / np.sum(weights_2d))


def weighted_corr(a: np.ndarray, b: np.ndarray, weights_2d: np.ndarray) -> float:
    w = weights_2d / np.sum(weights_2d)
    a_mean = np.sum(w * a)
    b_mean = np.sum(w * b)
    a_anom = a - a_mean
    b_anom = b - b_mean
    cov = np.sum(w * a_anom * b_anom)
    a_std = np.sqrt(np.sum(w * a_anom * a_anom))
    b_std = np.sqrt(np.sum(w * b_anom * b_anom))
    if a_std == 0 or b_std == 0:
        return np.nan
    return float(cov / (a_std * b_std))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    ds_in = xr.open_dataset(DATA_DIR / "20231012-06_input_netcdf.nc")
    ds_fc = xr.open_dataset(DATA_DIR / "006.nc")

    levels = [str(v) for v in ds_in["level"].values]
    lats = ds_in["lat"].values.astype(np.float64)
    lons = ds_in["lon"].values.astype(np.float64)

    x0 = ds_in["data"].isel(time=0).values.astype(np.float64)
    x1 = ds_in["data"].isel(time=1).values.astype(np.float64)
    y = ds_fc["data"].isel(time=0, step=0).values.astype(np.float64)

    persistence = x1
    linear = x1 + (x1 - x0)
    increment = y - x1
    tendency = x1 - x0

    lat_weights = np.cos(np.deg2rad(lats))
    weights_2d = lat_weights[:, None] * np.ones((len(lats), len(lons)), dtype=np.float64)

    rows = []
    map_rmse_persistence = np.sqrt(np.mean((y - persistence) ** 2, axis=0))
    map_rmse_linear = np.sqrt(np.mean((y - linear) ** 2, axis=0))
    map_skill = (map_rmse_persistence - map_rmse_linear) / np.maximum(map_rmse_persistence, 1e-8)
    mean_abs_increment = np.mean(np.abs(increment), axis=0)

    for idx, level_name in enumerate(levels):
        truth = y[idx]
        pers = persistence[idx]
        lin = linear[idx]
        inc = increment[idx]
        ten = tendency[idx]

        diff_p = truth - pers
        diff_l = truth - lin

        rmse_p = weighted_rmse(diff_p, weights_2d)
        rmse_l = weighted_rmse(diff_l, weights_2d)
        mae_p = weighted_mae(diff_p, weights_2d)
        mae_l = weighted_mae(diff_l, weights_2d)
        corr_p = weighted_corr(truth, pers, weights_2d)
        corr_l = weighted_corr(truth, lin, weights_2d)
        inc_energy = weighted_mean(np.abs(inc), weights_2d)
        ten_energy = weighted_mean(np.abs(ten), weights_2d)
        improvement = (rmse_p - rmse_l) / rmse_p if rmse_p != 0 else 0.0

        rows.append(
            {
                "level": level_name,
                "family": classify_variable(level_name),
                "rmse_persistence": rmse_p,
                "rmse_linear": rmse_l,
                "mae_persistence": mae_p,
                "mae_linear": mae_l,
                "corr_persistence": corr_p,
                "corr_linear": corr_l,
                "increment_mean_abs": inc_energy,
                "input_tendency_mean_abs": ten_energy,
                "linear_rmse_improvement_frac": improvement,
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["family", "level"]).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "variable_metrics.csv", index=False)

    family_summary = (
        metrics.groupby("family", as_index=False)[
            [
                "rmse_persistence",
                "rmse_linear",
                "mae_persistence",
                "mae_linear",
                "corr_persistence",
                "corr_linear",
                "increment_mean_abs",
                "input_tendency_mean_abs",
                "linear_rmse_improvement_frac",
            ]
        ]
        .mean()
        .sort_values("rmse_linear")
    )
    family_summary.to_csv(OUTPUT_DIR / "family_summary.csv", index=False)

    best_levels = metrics.sort_values("linear_rmse_improvement_frac", ascending=False).head(12)
    worst_levels = metrics.sort_values("linear_rmse_improvement_frac", ascending=True).head(12)
    pd.concat(
        [
            best_levels.assign(rank_group="best"),
            worst_levels.assign(rank_group="worst"),
        ]
    ).to_csv(OUTPUT_DIR / "improvement_rankings.csv", index=False)

    summary = {
        "input_shape": list(ds_in["data"].shape),
        "forecast_shape": list(ds_fc["data"].shape),
        "lat_count": int(len(lats)),
        "lon_count": int(len(lons)),
        "level_count": int(len(levels)),
        "forecast_steps": [int(v) for v in ds_fc["step"].values.tolist()],
        "time_values_input": [str(v) for v in ds_in["time"].values.tolist()],
        "time_values_forecast": [str(v) for v in ds_fc["time"].values.tolist()],
        "global_mean_abs_increment": weighted_mean(np.mean(np.abs(increment), axis=0), weights_2d),
        "global_mean_abs_input_tendency": weighted_mean(np.mean(np.abs(tendency), axis=0), weights_2d),
        "mean_persistence_rmse": float(metrics["rmse_persistence"].mean()),
        "mean_linear_rmse": float(metrics["rmse_linear"].mean()),
        "mean_linear_improvement_frac": float(metrics["linear_rmse_improvement_frac"].mean()),
        "best_improvement_level": str(best_levels.iloc[0]["level"]),
        "best_improvement_value": float(best_levels.iloc[0]["linear_rmse_improvement_frac"]),
        "worst_improvement_level": str(worst_levels.iloc[0]["level"]),
        "worst_improvement_value": float(worst_levels.iloc[0]["linear_rmse_improvement_frac"]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(12, 6))
    ordered = metrics.sort_values("rmse_linear")
    x = np.arange(len(ordered))
    plt.bar(x - 0.2, ordered["rmse_persistence"], width=0.4, label="Persistence")
    plt.bar(x + 0.2, ordered["rmse_linear"], width=0.4, label="Linear extrapolation")
    plt.xticks(x, ordered["level"], rotation=90, fontsize=7)
    plt.ylabel("Latitude-weighted RMSE")
    plt.title("6-hour forecast RMSE by variable channel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "rmse_by_variable.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    family_plot = family_summary.sort_values("linear_rmse_improvement_frac", ascending=False)
    sns.barplot(data=family_plot, x="family", y="linear_rmse_improvement_frac", color="#2a9d8f")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Mean RMSE improvement over persistence")
    plt.xlabel("Variable family")
    plt.title("Benefit of simple trend extrapolation by variable family")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "family_improvement.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    vmax1 = np.percentile(map_rmse_persistence, 98)
    vmax2 = np.percentile(np.abs(map_skill), 98)
    vmax3 = np.percentile(mean_abs_increment, 98)
    im0 = axes[0].imshow(map_rmse_persistence, origin="upper", cmap="viridis", vmin=0, vmax=vmax1)
    axes[0].set_title("Persistence RMSE map")
    axes[1].imshow(map_skill, origin="upper", cmap="coolwarm", vmin=-vmax2, vmax=vmax2)
    axes[1].set_title("Relative skill of linear baseline")
    axes[2].imshow(mean_abs_increment, origin="upper", cmap="magma", vmin=0, vmax=vmax3)
    axes[2].set_title("Mean absolute 6-hour increment")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    plt.savefig(REPORT_IMG_DIR / "global_maps.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=metrics,
        x="input_tendency_mean_abs",
        y="linear_rmse_improvement_frac",
        hue="family",
        s=60,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Mean absolute input tendency")
    plt.ylabel("RMSE improvement over persistence")
    plt.title("Variables with stronger short-term dynamics benefit more from trend extrapolation")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "tendency_vs_skill.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    top = metrics.sort_values("linear_rmse_improvement_frac", ascending=False).head(15)
    sns.barplot(data=top, x="linear_rmse_improvement_frac", y="level", color="#e76f51")
    plt.xlabel("RMSE improvement over persistence")
    plt.ylabel("Level / variable")
    plt.title("Top channels helped by trend extrapolation")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "top_improving_variables.png", dpi=180)
    plt.close()

    ds_in.close()
    ds_fc.close()


if __name__ == "__main__":
    main()
