#!/usr/bin/env python3
"""Reproducible analysis of Floquet-Bloch replica signatures in graphene tr-ARPES data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"
REPORT_DIR = BASE / "report"
IMAGE_DIR = REPORT_DIR / "images"

PUMP_WAVELENGTH_UM = 5.0
PHOTON_ENERGY_EV = 1.239841984 / PUMP_WAVELENGTH_UM
HBAR_EV_S = 6.582119569e-16


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Dict[str, object]:
    with open(DATA_DIR / "processed_band_data.json", "r", encoding="utf-8") as f:
        processed = json.load(f)
    pol = pd.read_csv(DATA_DIR / "polarization_dependence_data.csv")
    with h5py.File(DATA_DIR / "raw_trARPES_data.h5", "r") as h5:
        energy = h5["energy_axis"][:]
        kx = h5["kx_axis"][:]
        time_delays = h5["time_delays"][:]
        polarization_angles = h5["polarization_angles"][:]
        pump_off = h5["pump_off_spectrum"][:]
        pump_on = {int(angle): h5[f"pump_on_angle_{int(angle)}"][:] for angle in polarization_angles}
    return {
        "processed": processed,
        "polarization": pol,
        "energy": energy,
        "kx": kx,
        "time_delays": time_delays,
        "polarization_angles": polarization_angles,
        "pump_off": pump_off,
        "pump_on": pump_on,
    }


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 4000, seed: int = 7) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        slopes[i] = slope_through_origin(xb, yb)
    return {
        "median": float(np.median(slopes)),
        "mean": float(np.mean(slopes)),
        "std": float(np.std(slopes, ddof=1)),
        "ci95_low": float(np.percentile(slopes, 2.5)),
        "ci95_high": float(np.percentile(slopes, 97.5)),
    }


def harmonic_design(theta_rad: np.ndarray, model: str) -> np.ndarray:
    if model == "constant":
        return np.column_stack([np.ones_like(theta_rad)])
    if model == "2fold":
        return np.column_stack([
            np.ones_like(theta_rad),
            np.cos(2 * theta_rad),
            np.sin(2 * theta_rad),
        ])
    if model == "4fold":
        return np.column_stack([
            np.ones_like(theta_rad),
            np.cos(2 * theta_rad),
            np.sin(2 * theta_rad),
            np.cos(4 * theta_rad),
            np.sin(4 * theta_rad),
        ])
    raise ValueError(f"Unknown model: {model}")


def fit_harmonic_models(angle_deg: Iterable[float], values: Iterable[float], dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, dict], str]:
    theta_rad = np.deg2rad(np.asarray(list(angle_deg), dtype=float))
    y = np.asarray(list(values), dtype=float)
    n = len(y)
    rows: List[dict] = []
    predictions: Dict[str, dict] = {}
    for model in ["constant", "2fold", "4fold"]:
        X = harmonic_design(theta_rad, model)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X @ beta
        rss = float(np.sum((y - yhat) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 1.0
        k = X.shape[1]
        safe_rss = max(rss, 1e-15)
        aic = float(n * np.log(safe_rss / n) + 2 * k)
        bic = float(n * np.log(safe_rss / n) + k * np.log(n))
        # Leave-one-out cross validation to guard against overfitting the small angle grid.
        loocv_preds = []
        for i in range(n):
            train = np.arange(n) != i
            beta_i = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
            loocv_preds.append(float((X[i : i + 1] @ beta_i)[0]))
        loocv_preds = np.asarray(loocv_preds)
        loocv_rmse = float(np.sqrt(np.mean((loocv_preds - y) ** 2)))
        modulation_depth = float((yhat.max() - yhat.min()) / yhat.mean())
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "n_parameters": k,
                "rss": rss,
                "r2": r2,
                "aic": aic,
                "bic": bic,
                "loocv_rmse": loocv_rmse,
                "modulation_depth": modulation_depth,
            }
        )
        predictions[model] = {
            "beta": [float(v) for v in beta],
            "yhat": [float(v) for v in yhat],
            "loocv_predictions": [float(v) for v in loocv_preds],
        }
    comparison = pd.DataFrame(rows).sort_values(["bic", "aic"]).reset_index(drop=True)
    best_model = str(comparison.iloc[0]["model"])
    return comparison, predictions, best_model


def raw_main_ridge(pump_off: np.ndarray, energy: np.ndarray, kx: np.ndarray, k_window: float = 0.12) -> pd.DataFrame:
    left_mask = (kx < 0) & (kx > -k_window)
    right_mask = (kx > 0) & (kx < k_window)
    records = []
    for row, e in zip(pump_off, energy):
        left_vals = row[left_mask]
        right_vals = row[right_mask]
        lk = float(kx[left_mask][np.argmax(left_vals)])
        rk = float(kx[right_mask][np.argmax(right_vals)])
        records.append(
            {
                "energy": float(e),
                "left_k": lk,
                "right_k": rk,
                "center_k": 0.5 * (lk + rk),
                "mean_abs_k": 0.5 * (abs(lk) + abs(rk)),
                "left_intensity": float(np.max(left_vals)),
                "right_intensity": float(np.max(right_vals)),
            }
        )
    return pd.DataFrame(records)


def make_masks(energy: np.ndarray, kx: np.ndarray, dirac_energy: float) -> Dict[str, np.ndarray]:
    main_mask = (np.abs(energy[:, None] - dirac_energy) < 0.08) & (np.abs(kx[None, :]) < 0.06)
    plus_mask = (np.abs(energy[:, None] - (dirac_energy + PHOTON_ENERGY_EV)) < 0.08) & (np.abs(kx[None, :]) < 0.08)
    minus_mask = (np.abs(energy[:, None] - (dirac_energy - PHOTON_ENERGY_EV)) < 0.08) & (np.abs(kx[None, :]) < 0.08)
    return {"main": main_mask, "plus": plus_mask, "minus": minus_mask}


def shifted_template_correlation(
    pump_off: np.ndarray,
    pump_on: Dict[int, np.ndarray],
    energy: np.ndarray,
    kx: np.ndarray,
    dirac_energy: float,
) -> pd.DataFrame:
    interpolator = interp1d(energy, pump_off, axis=0, bounds_error=False, fill_value=0.0)
    shifted_template = interpolator(energy - PHOTON_ENERGY_EV) + interpolator(energy + PHOTON_ENERGY_EV)
    sideband_mask = (
        (((energy[:, None] > dirac_energy + 0.10) & (energy[:, None] < dirac_energy + 0.34))
         | ((energy[:, None] > dirac_energy - 0.34) & (energy[:, None] < dirac_energy - 0.10)))
        & (np.abs(kx[None, :]) < 0.12)
    )
    template_values = shifted_template[sideband_mask]
    positive_template = template_values[template_values > 0]
    threshold = float(np.percentile(positive_template, 25))
    rows = []
    for angle, spectrum in pump_on.items():
        diff = np.maximum(spectrum - pump_off, 0.0)
        diff_values = diff[sideband_mask]
        keep = template_values > threshold
        corr = float(np.corrcoef(diff_values[keep], template_values[keep])[0, 1])
        high = float(diff_values[template_values >= np.quantile(template_values, 0.9)].mean())
        low = float(diff_values[template_values <= np.quantile(template_values, 0.5)].mean())
        rows.append(
            {
                "angle_deg": int(angle),
                "template_correlation": corr,
                "template_high_mean": high,
                "template_low_mean": low,
                "template_contrast_ratio": high / low,
            }
        )
    return pd.DataFrame(rows).sort_values("angle_deg").reset_index(drop=True)


def spectral_summary(
    pump_off: np.ndarray,
    pump_on: Dict[int, np.ndarray],
    energy: np.ndarray,
    kx: np.ndarray,
    dirac_energy: float,
) -> pd.DataFrame:
    masks = make_masks(energy, kx, dirac_energy)
    rows = []
    for angle, spectrum in pump_on.items():
        diff = spectrum - pump_off
        rows.append(
            {
                "angle_deg": int(angle),
                "pump_on_mean": float(spectrum.mean()),
                "delta_mean": float(diff.mean()),
                "main_mean_on": float(spectrum[masks["main"]].mean()),
                "main_mean_off": float(pump_off[masks["main"]].mean()),
                "main_delta_mean": float(diff[masks["main"]].mean()),
                "replica_plus_delta_mean": float(diff[masks["plus"]].mean()),
                "replica_minus_delta_mean": float(diff[masks["minus"]].mean()),
            }
        )
    df = pd.DataFrame(rows).sort_values("angle_deg").reset_index(drop=True)
    df["replica_symmetry_diff"] = df["replica_plus_delta_mean"] - df["replica_minus_delta_mean"]
    df["replica_avg_delta_mean"] = 0.5 * (df["replica_plus_delta_mean"] + df["replica_minus_delta_mean"])
    df["replica_to_main_delta_ratio"] = df["replica_avg_delta_mean"] / df["main_delta_mean"]
    return df


def edc_around_k(spectrum: np.ndarray, kx: np.ndarray, target_abs_k: float, dk: float = 0.0065) -> np.ndarray:
    mask = (np.abs(np.abs(kx) - target_abs_k) <= dk)
    return spectrum[:, mask].mean(axis=1)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_figures(data: Dict[str, object], results: Dict[str, object]) -> None:
    processed = data["processed"]
    pol = data["polarization"]
    energy = data["energy"]
    kx = data["kx"]
    pump_off = data["pump_off"]
    pump_on = data["pump_on"]
    dirac_energy = results["dirac_energy_eV"]
    dirac_k = results["dirac_k_Ainv"]
    slope = results["main_cone_fit"]["k_per_eV"]
    predicted_replica_k = results["predicted_replica_abs_k_Ainv"]
    corr_df = results["template_correlation_df"]
    spectral_df = results["spectral_summary_df"]
    csv_fit = results["polarization_fits"]["csv"]
    raw_fit = results["polarization_fits"]["raw_replica_delta"]

    def draw_map(ax, matrix, title, cmap="magma"):
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[float(kx.min()), float(kx.max()), float(energy.min()), float(energy.max())],
            cmap=cmap,
        )
        ax.axhline(dirac_energy, color="white", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(dirac_energy + PHOTON_ENERGY_EV, color="cyan", linestyle=":", linewidth=1.0, alpha=0.9)
        ax.axhline(dirac_energy - PHOTON_ENERGY_EV, color="cyan", linestyle=":", linewidth=1.0, alpha=0.9)
        ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(title)
        return im

    # Figure 1: overview maps.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    im0 = draw_map(axes[0], pump_off, "Pump off")
    im1 = draw_map(axes[1], pump_on[90], "Pump on (90°)")
    im2 = draw_map(axes[2], pump_on[90] - pump_off, "Pump on - off (90°)", cmap="coolwarm")
    replica_points = np.array([[r["kx"], r["energy"]] for r in processed["replica_bands"]], dtype=float)
    for ax in axes:
        ax.scatter(replica_points[:, 0], replica_points[:, 1], s=45, facecolor="none", edgecolor="lime", linewidth=1.2, label="Processed replicas")
        ax.scatter([dirac_k], [dirac_energy], s=40, color="white", marker="x", linewidth=1.2, label="Dirac point")
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Intensity (arb. units)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Intensity (arb. units)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="Δ intensity")
    fig.suptitle("tr-ARPES overview: equilibrium cone and pump-induced sidebands", fontsize=14)
    fig.savefig(IMAGE_DIR / "figure1_overview_maps.png", dpi=220)
    plt.close(fig)

    # Figure 2: validation panels.
    band_df = pd.DataFrame(processed["band_dispersion"])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), constrained_layout=True)
    ax = axes[0]
    ax.scatter(band_df["kx"], band_df["energy"], s=11, color="black", alpha=0.5, label="Main-cone ridge")
    e_grid = np.linspace(float(energy.min()), float(energy.max()), 500)
    for offset, label, color in [(0.0, "Main cone fit", "tab:blue"), (PHOTON_ENERGY_EV, "+1 guide", "tab:orange"), (-PHOTON_ENERGY_EV, "-1 guide", "tab:green")]:
        valid = np.abs(e_grid - (dirac_energy + offset)) <= (abs(e_grid - (dirac_energy + offset)).max())
        k_curve = slope * np.abs(e_grid - (dirac_energy + offset))
        ax.plot(+k_curve, e_grid, color=color, linewidth=1.6, label=label)
        ax.plot(-k_curve, e_grid, color=color, linewidth=1.6)
    ax.scatter(replica_points[:, 0], replica_points[:, 1], s=55, color="red", zorder=3, label="Replica markers")
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(float(energy.min()), float(energy.max()))
    ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Main-cone fit and ±1 Floquet guides")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    target_k = predicted_replica_k
    edc_off = edc_around_k(pump_off, kx, target_k)
    edc_on = edc_around_k(pump_on[90], kx, target_k)
    edc_diff = edc_on - edc_off
    ax.plot(energy, edc_off, color="black", linewidth=1.4, label="Pump off")
    ax.plot(energy, edc_on, color="tab:red", linewidth=1.4, label="Pump on (90°)")
    ax.plot(energy, edc_diff, color="tab:purple", linewidth=1.2, label="Difference")
    for ref_e, txt in [(dirac_energy - PHOTON_ENERGY_EV, r"$E_D-\hbar\Omega$"), (dirac_energy, r"$E_D$"), (dirac_energy + PHOTON_ENERGY_EV, r"$E_D+\hbar\Omega$")]:
        ax.axvline(ref_e, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.set_title(fr"EDC averaged near $|k_x|={target_k:.3f}\,\AA^{{-1}}$")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    ax.bar(corr_df["angle_deg"].astype(str), corr_df["template_correlation"], color="tab:blue", alpha=0.8)
    ax.set_ylim(0.94, 1.0)
    ax.set_ylabel("Correlation with energy-shifted template")
    ax.set_xlabel("Pump polarization angle (deg)")
    ax.set_title("Replica dispersion remains a shifted copy of the main cone")
    mean_corr = corr_df["template_correlation"].mean()
    std_corr = corr_df["template_correlation"].std(ddof=1)
    ax.text(0.02, 0.05, f"mean = {mean_corr:.4f}\nstd = {std_corr:.4f}", transform=ax.transAxes, fontsize=9, va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.suptitle("Quantitative validation of Floquet-like replica bands", fontsize=14)
    fig.savefig(IMAGE_DIR / "figure2_replica_validation.png", dpi=220)
    plt.close(fig)

    # Figure 3: polarization dependence.
    angle_fine = np.linspace(0, 180, 361)
    theta_fine = np.deg2rad(angle_fine)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    ax = axes[0]
    for model, style, color in [("constant", "--", "gray"), ("2fold", ":", "tab:green"), ("4fold", "-", "tab:red")]:
        beta = np.asarray(csv_fit["predictions"][model]["beta"], dtype=float)
        X_fine = harmonic_design(theta_fine, model)
        ax.plot(angle_fine, X_fine @ beta, linestyle=style, color=color, linewidth=1.8, label=model)
    ax.scatter(pol["angle_degrees"], pol["intensity"], s=45, color="black", zorder=3, label="Measured replica intensity")
    ax.set_xlabel("Pump polarization angle (deg)")
    ax.set_ylabel("Replica intensity (arb. units)")
    ax.set_title("CSV polarization scan")
    ax.legend(fontsize=8, loc="best")
    best_csv = csv_fit["best_model"]
    best_csv_row = pd.DataFrame(csv_fit["comparison"]).query("model == @best_csv").iloc[0]
    ax.text(0.03, 0.05, f"Best model: {best_csv}\nLOOCV RMSE = {best_csv_row['loocv_rmse']:.2e}", transform=ax.transAxes, fontsize=9, va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax = axes[1]
    raw_angles = spectral_df["angle_deg"].to_numpy(dtype=float)
    raw_theta = np.deg2rad(raw_angles)
    raw_replica = spectral_df["replica_avg_delta_mean"].to_numpy(dtype=float)
    raw_main = spectral_df["main_delta_mean"].to_numpy(dtype=float)
    raw_replica_norm = raw_replica / raw_replica.mean()
    raw_main_norm = raw_main / raw_main.mean()
    beta_raw = np.asarray(raw_fit["predictions"]["4fold"]["beta"], dtype=float)
    raw_fit_curve = harmonic_design(theta_fine, "4fold") @ beta_raw
    ax.plot(angle_fine, raw_fit_curve / raw_replica.mean(), color="tab:red", linewidth=1.8, label="4fold fit (replica Δ)")
    ax.scatter(raw_angles, raw_replica_norm, s=48, color="tab:red", label="Replica Δ / mean")
    ax.scatter(raw_angles, raw_main_norm, s=48, color="tab:blue", marker="s", label="Main-cone Δ / mean")
    ax.set_xlabel("Pump polarization angle (deg)")
    ax.set_ylabel("Normalized pump-induced enhancement")
    ax.set_title("Raw HDF5 ROI analysis")
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("Angular modulation favors a fourfold matrix-element envelope", fontsize=14)
    fig.savefig(IMAGE_DIR / "figure3_polarization_dependence.png", dpi=220)
    plt.close(fig)

    # Figure 4: all angles difference maps.
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.6), constrained_layout=True)
    all_axes = axes.ravel()
    for ax, angle in zip(all_axes, sorted(pump_on)):
        im = draw_map(ax, pump_on[angle] - pump_off, f"Δ map, {angle}°", cmap="coolwarm")
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.42, 0.32)
    all_axes[-1].axis("off")
    cbar = fig.colorbar(im, ax=all_axes[:-1], shrink=0.92, label="Δ intensity")
    fig.suptitle("Pump-induced replica weight across polarization angle", fontsize=14)
    fig.savefig(IMAGE_DIR / "figure4_all_angle_difference_maps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    data = load_data()
    processed = data["processed"]
    pol = data["polarization"]
    energy = data["energy"]
    kx = data["kx"]
    pump_off = data["pump_off"]
    pump_on = data["pump_on"]
    time_delays = data["time_delays"]

    dirac_energy = float(processed["dirac_point"][1])
    ridge_df = raw_main_ridge(pump_off, energy, kx)
    dirac_k = float(ridge_df.loc[np.abs(ridge_df["energy"] - dirac_energy) < 0.1, "center_k"].median())

    fit_mask = np.abs(ridge_df["energy"].to_numpy() - dirac_energy) > 0.05
    fit_x = np.abs(ridge_df.loc[fit_mask, "energy"].to_numpy() - dirac_energy)
    fit_y = ridge_df.loc[fit_mask, "mean_abs_k"].to_numpy()
    slope = slope_through_origin(fit_x, fit_y)
    slope_boot = bootstrap_slope(fit_x, fit_y)
    dE_dk = 1.0 / slope
    vf = dE_dk * 1e-10 / HBAR_EV_S

    replica_rows = []
    replica_abs_ks = []
    energy_residuals = []
    for rep in processed["replica_bands"]:
        dE = float(rep["energy"] - dirac_energy)
        energy_residual = abs(dE) - PHOTON_ENERGY_EV
        energy_residuals.append(energy_residual)
        replica_abs_ks.append(abs(float(rep["kx"])))
        replica_rows.append(
            {
                "order": int(rep["order"]),
                "kx_Ainv": float(rep["kx"]),
                "energy_eV": float(rep["energy"]),
                "delta_energy_eV": dE,
                "energy_residual_vs_photon_eV": float(energy_residual),
                "intensity": float(rep["intensity"]),
            }
        )
    replica_df = pd.DataFrame(replica_rows)
    predicted_replica_abs_k = slope * PHOTON_ENERGY_EV
    observed_replica_abs_k_mean = float(np.mean(replica_abs_ks))

    inventory = {
        "raw_h5_keys": [
            "energy_axis",
            "kx_axis",
            "polarization_angles",
            "pump_off_spectrum",
            "pump_on_angle_0",
            "pump_on_angle_30",
            "pump_on_angle_60",
            "pump_on_angle_90",
            "pump_on_angle_120",
            "pump_on_angle_150",
            "pump_on_angle_180",
            "time_delays",
        ],
        "energy_axis": {
            "n": int(len(energy)),
            "min_eV": float(energy.min()),
            "max_eV": float(energy.max()),
            "step_eV": float(np.mean(np.diff(energy))),
        },
        "kx_axis": {
            "n": int(len(kx)),
            "min_Ainv": float(kx.min()),
            "max_Ainv": float(kx.max()),
            "step_Ainv": float(np.mean(np.diff(kx))),
        },
        "time_delays_ps": [float(v) for v in time_delays],
        "polarization_angles_deg": [int(v) for v in data["polarization_angles"]],
        "spectra_shapes": {
            "pump_off": list(map(int, pump_off.shape)),
            "pump_on_each_angle": list(map(int, next(iter(pump_on.values())).shape)),
        },
    }
    save_json(OUTPUT_DIR / "data_inventory.json", inventory)

    quality_notes = f"""# Data quality and convention notes

- The supplied HDF5 file exposes 2D spectra `pump_off_spectrum` and `pump_on_angle_*` for seven pump polarization angles, plus separate `time_delays` values. No explicit 3D/4D time-resolved intensity cube was present in the file tree, so the present analysis treats the spectra as angle-resolved snapshots rather than a full delay-resolved movie.
- `processed_band_data.json` provides a useful Dirac-point energy ({dirac_energy:.6f} eV) because it makes the replica spacing exactly match the 5 μm photon energy. However, its `dirac_point[0]` momentum coordinate is inconsistent with the raw spectra (listed as the left boundary, -0.3 Å^-1). Momentum was therefore re-centered using the raw pump-off cone symmetry, which yields `k_D ≈ {dirac_k:.3e}` Å^-1.
- `polarization_dependence_data.csv` uses `target_energy ≈ 0.2487 eV`, which is consistent with a replica energy measured relative to the Dirac point rather than the raw absolute energy axis. Adding the processed Dirac energy gives an absolute target energy near {dirac_energy + PHOTON_ENERGY_EV:.6f} eV, consistent with the +1 replica in the raw spectra.
- Because the dataset does not expose a clean time stack or a geometry where the Volkov channel is independently nulled, the final interpretation can confirm replica bands and polarization-dependent matrix-element effects, but it cannot fully isolate pure initial-state Floquet weight from final-state dressing.
"""
    (OUTPUT_DIR / "data_quality_notes.md").write_text(quality_notes, encoding="utf-8")

    spectral_df = spectral_summary(pump_off, pump_on, energy, kx, dirac_energy)
    template_corr_df = shifted_template_correlation(pump_off, pump_on, energy, kx, dirac_energy)
    spectral_df = spectral_df.merge(template_corr_df, on="angle_deg", how="left")
    spectral_df.to_csv(OUTPUT_DIR / "spectral_summaries.csv", index=False)
    template_corr_df.to_csv(OUTPUT_DIR / "template_correlation.csv", index=False)

    csv_comp, csv_predictions, csv_best = fit_harmonic_models(pol["angle_degrees"], pol["intensity"], "csv_replica_intensity")
    raw_comp, raw_predictions, raw_best = fit_harmonic_models(spectral_df["angle_deg"], spectral_df["replica_avg_delta_mean"], "raw_replica_delta")
    model_comparison = pd.concat([csv_comp, raw_comp], ignore_index=True)
    model_comparison.to_csv(OUTPUT_DIR / "polarization_model_comparison.csv", index=False)
    save_json(
        OUTPUT_DIR / "polarization_fit_results.json",
        {
            "csv_replica_intensity": {
                "best_model": csv_best,
                "comparison": csv_comp.to_dict(orient="records"),
                "predictions": csv_predictions,
            },
            "raw_replica_delta": {
                "best_model": raw_best,
                "comparison": raw_comp.to_dict(orient="records"),
                "predictions": raw_predictions,
            },
        },
    )

    grouped = spectral_df.assign(group=np.where(spectral_df["angle_deg"].isin([0, 90, 180]), "axial (0/90/180)", "oblique (30/60/120/150)"))
    group_summary = grouped.groupby("group")[["replica_avg_delta_mean", "main_delta_mean"]].mean().reset_index()
    group_summary.to_csv(OUTPUT_DIR / "polarization_group_summary.csv", index=False)

    replica_metrics = {
        "pump_wavelength_um": PUMP_WAVELENGTH_UM,
        "photon_energy_eV": PHOTON_ENERGY_EV,
        "dirac_energy_eV": dirac_energy,
        "dirac_k_Ainv": dirac_k,
        "energy_step_eV": float(np.mean(np.diff(energy))),
        "kx_step_Ainv": float(np.mean(np.diff(kx))),
        "main_cone_fit": {
            "k_per_eV": float(slope),
            "k_per_eV_bootstrap": slope_boot,
            "dE_dk_eV_A": float(dE_dk),
            "fermi_velocity_m_per_s": float(vf),
        },
        "replicas": replica_df.to_dict(orient="records"),
        "mean_abs_replica_k_Ainv": observed_replica_abs_k_mean,
        "predicted_replica_abs_k_Ainv": float(predicted_replica_abs_k),
        "mean_abs_replica_k_relative_error": float((observed_replica_abs_k_mean - predicted_replica_abs_k) / predicted_replica_abs_k),
        "energy_residual_vs_photon_eV": {
            "mean": float(np.mean(energy_residuals)),
            "max_abs": float(np.max(np.abs(energy_residuals))),
        },
        "template_correlation": {
            "mean": float(template_corr_df["template_correlation"].mean()),
            "std": float(template_corr_df["template_correlation"].std(ddof=1)),
            "min": float(template_corr_df["template_correlation"].min()),
            "max": float(template_corr_df["template_correlation"].max()),
        },
        "replica_enhancement_by_angle": spectral_df[["angle_deg", "replica_avg_delta_mean", "main_delta_mean", "replica_to_main_delta_ratio"]].to_dict(orient="records"),
    }
    save_json(OUTPUT_DIR / "replica_metrics.json", replica_metrics)

    # Minimal related-work notes based on the supplied PDFs and interpretation search.
    related_notes = """# Related-work notes

- **Oka & Aoki (2009), _Photovoltaic Hall effect in graphene_**: early Floquet theory for Dirac cones under strong periodic driving; establishes that circular driving can open light-induced gaps in graphene.
- **Wang et al. (2013), _Observation of Floquet-Bloch states on the surface of a topological insulator_**: experimental proof that tr-ARPES can directly image photon-dressed replica bands and polarization-dependent avoided crossings.
- **Sentef et al. (2015), _Theory of Floquet band formation and local pseudospin textures in pump-probe photoemission of graphene_**: predicts that realistic pump pulses in graphene generate Floquet-like sidebands observable in pump-probe photoemission, even when gaps are broadened.
- **Selective scattering between Floquet-Bloch and Volkov states**: emphasizes that pump-probe ARPES sidebands can reflect both initial-state Floquet dressing and final-state Volkov/LAPE dressing; geometry and polarization are critical for distinguishing them.
"""
    (OUTPUT_DIR / "related_work_notes.md").write_text(related_notes, encoding="utf-8")

    results = {
        "dirac_energy_eV": dirac_energy,
        "dirac_k_Ainv": dirac_k,
        "main_cone_fit": replica_metrics["main_cone_fit"],
        "predicted_replica_abs_k_Ainv": float(predicted_replica_abs_k),
        "template_correlation_df": template_corr_df,
        "spectral_summary_df": spectral_df,
        "polarization_fits": {
            "csv": {
                "best_model": csv_best,
                "comparison": csv_comp.to_dict(orient="records"),
                "predictions": csv_predictions,
            },
            "raw_replica_delta": {
                "best_model": raw_best,
                "comparison": raw_comp.to_dict(orient="records"),
                "predictions": raw_predictions,
            },
        },
    }
    create_figures(data, results)

    summary = {
        "dirac_energy_eV": dirac_energy,
        "photon_energy_eV": PHOTON_ENERGY_EV,
        "main_dE_dk_eV_A": float(dE_dk),
        "fermi_velocity_m_per_s": float(vf),
        "template_correlation_mean": float(template_corr_df["template_correlation"].mean()),
        "best_csv_polarization_model": csv_best,
        "best_raw_polarization_model": raw_best,
        "axial_to_oblique_replica_enhancement_ratio": float(
            group_summary.loc[group_summary["group"] == "axial (0/90/180)", "replica_avg_delta_mean"].iloc[0]
            / group_summary.loc[group_summary["group"] == "oblique (30/60/120/150)", "replica_avg_delta_mean"].iloc[0]
        ),
    }
    save_json(OUTPUT_DIR / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
