#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def load_processed():
    with (DATA_DIR / "processed_band_data.json").open() as f:
        return json.load(f)


def load_polarization_table():
    rows = []
    with (DATA_DIR / "polarization_dependence_data.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def nearest_idx(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def fit_linear_dirac_band(points):
    k = np.array([p["kx"] for p in points], dtype=float)
    e = np.array([p["energy"] for p in points], dtype=float)
    slope, intercept = np.polyfit(k, e, 1)
    return slope, intercept


def fit_polarization_model(angles_deg: np.ndarray, intensity: np.ndarray):
    theta = np.deg2rad(angles_deg)
    X = np.column_stack(
        [
            np.ones_like(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X, intensity, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((intensity - pred) ** 2)
    ss_tot = np.sum((intensity - intensity.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    amp = float(np.hypot(coeffs[1], coeffs[2]))
    phase = 0.5 * math.atan2(coeffs[2], coeffs[1])
    return {
        "baseline": float(coeffs[0]),
        "cos2_coeff": float(coeffs[1]),
        "sin2_coeff": float(coeffs[2]),
        "amplitude": amp,
        "phase_radians": float(phase),
        "phase_degrees": float(np.rad2deg(phase)),
        "predicted_intensity": pred.tolist(),
        "r_squared": float(r2),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    processed = load_processed()
    pol_rows = load_polarization_table()

    with h5py.File(DATA_DIR / "raw_trARPES_data.h5", "r") as f:
        energy = f["energy_axis"][:]
        kx = f["kx_axis"][:]
        off = f["pump_off_spectrum"][:]
        angles = f["polarization_angles"][:]
        on_maps = {int(a): f[f"pump_on_angle_{int(a)}"][:] for a in angles}

    dirac_energy, dirac_kx = processed["dirac_point"]
    replica_bands = processed["replica_bands"]
    band_dispersion = processed["band_dispersion"]
    pump_energy = float(processed["pump_energy"])

    left_branch = [p for p in band_dispersion if p["kx"] <= 0]
    right_branch = [p for p in band_dispersion if p["kx"] >= 0]
    left_slope, left_intercept = fit_linear_dirac_band(left_branch)
    right_slope, right_intercept = fit_linear_dirac_band(right_branch)

    replica_offsets = []
    replica_raw_metrics = []
    energies_by_order = {}
    for rb in replica_bands:
        energies_by_order.setdefault(int(rb["order"]), []).append(float(rb["energy"]))
        replica_offsets.append(
            {
                "order": int(rb["order"]),
                "kx": float(rb["kx"]),
                "energy": float(rb["energy"]),
                "intensity": float(rb["intensity"]),
            }
        )
        e_idx = nearest_idx(energy, rb["energy"])
        k_idx = nearest_idx(kx, rb["kx"])
        window = (
            slice(max(0, e_idx - 2), min(len(energy), e_idx + 3)),
            slice(max(0, k_idx - 2), min(len(kx), k_idx + 3)),
        )
        angle_metrics = {}
        for angle, arr in on_maps.items():
            local_on = float(np.mean(arr[window]))
            local_off = float(np.mean(off[window]))
            angle_metrics[str(angle)] = {
                "pump_on_mean": local_on,
                "pump_off_mean": local_off,
                "difference": local_on - local_off,
                "ratio": local_on / local_off if local_off else None,
            }
        replica_raw_metrics.append(
            {
                "order": int(rb["order"]),
                "kx": float(rb["kx"]),
                "energy": float(rb["energy"]),
                "angle_metrics": angle_metrics,
            }
        )

    pol_angles = np.array([row["angle_degrees"] for row in pol_rows], dtype=float)
    pol_intensity = np.array([row["intensity"] for row in pol_rows], dtype=float)
    pol_fit = fit_polarization_model(pol_angles, pol_intensity)

    raw_angle_summary = []
    target_energy = pol_rows[0]["target_energy"]
    target_kx = pol_rows[0]["target_kx"]
    e_idx = nearest_idx(energy, target_energy)
    k_idx = nearest_idx(kx, target_kx)
    window = (
        slice(max(0, e_idx - 2), min(len(energy), e_idx + 3)),
        slice(max(0, k_idx - 2), min(len(kx), k_idx + 3)),
    )
    for angle in sorted(on_maps):
        arr = on_maps[angle]
        raw_angle_summary.append(
            {
                "angle_degrees": int(angle),
                "local_replica_intensity": float(np.mean(arr[window])),
                "local_background": float(np.mean(off[window])),
                "local_difference": float(np.mean(arr[window] - off[window])),
                "global_difference_mean": float(np.mean(arr - off)),
                "global_difference_max": float(np.max(arr - off)),
            }
        )

    order_mean_energy = {
        str(order): float(np.mean(vals)) for order, vals in sorted(energies_by_order.items())
    }
    adjacent_order_spacing = {}
    sorted_orders = sorted(energies_by_order)
    for lower, upper in zip(sorted_orders[:-1], sorted_orders[1:]):
        spacing = order_mean_energy[str(upper)] - order_mean_energy[str(lower)]
        adjacent_order_spacing[f"{lower}_to_{upper}"] = {
            "observed_spacing_eV": float(spacing),
            "pump_energy_eV": float(pump_energy),
            "spacing_error_eV": float(spacing - pump_energy),
        }

    summary = {
        "pump_energy_eV": pump_energy,
        "dirac_point": {
            "energy_eV": float(dirac_energy),
            "kx_Ainv": float(dirac_kx),
        },
        "main_band_fits": {
            "left_branch": {
                "slope_eV_per_Ainv": float(left_slope),
                "intercept_eV": float(left_intercept),
            },
            "right_branch": {
                "slope_eV_per_Ainv": float(right_slope),
                "intercept_eV": float(right_intercept),
            },
        },
        "replica_offsets": replica_offsets,
        "order_mean_energy": order_mean_energy,
        "adjacent_order_spacing": adjacent_order_spacing,
        "replica_raw_metrics": replica_raw_metrics,
        "polarization_fit": pol_fit,
        "raw_angle_summary": raw_angle_summary,
    }

    with (OUTPUT_DIR / "analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (OUTPUT_DIR / "literature_notes.md").open("w") as f:
        f.write(
            "# Local literature notes\n\n"
            "- `paper_000.pdf`: theoretical Floquet response in graphene under strong light; motivates photon-dressed Dirac physics and pump-induced band modifications.\n"
            "- `paper_001.pdf`: tr-ARPES observation of Floquet-Bloch states in a Dirac surface system; key signatures are replica bands spaced by the pump photon energy and polarization-dependent avoided-crossing behavior.\n"
            "- `paper_002.pdf`: broader Floquet-engineering context in Dirac materials; supports conservative interpretation of driven band replicas as light-dressed quasibands.\n"
            "- `paper_003.pdf`: graphene-specific Floquet photoemission theory; supports interpreting sidebands and local spectral modifications under realistic pulses.\n"
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(
        off,
        origin="lower",
        aspect="auto",
        extent=[kx.min(), kx.max(), energy.min(), energy.max()],
        cmap="magma",
    )
    axes[0].set_title("Pump off spectrum")
    axes[0].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].scatter([dirac_kx], [dirac_energy], c="cyan", s=30, label="Dirac point")
    axes[0].legend(loc="upper right")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    ref_angle = 90 if 90 in on_maps else sorted(on_maps)[0]
    diff = on_maps[ref_angle] - off
    im1 = axes[1].imshow(
        diff,
        origin="lower",
        aspect="auto",
        extent=[kx.min(), kx.max(), energy.min(), energy.max()],
        cmap="coolwarm",
    )
    axes[1].set_title(f"Pump-on minus off ({ref_angle} deg)")
    axes[1].set_xlabel(r"$k_x$ (1/Angstrom)")
    axes[1].set_ylabel("Energy (eV)")
    for rb in replica_bands:
        axes[1].scatter(rb["kx"], rb["energy"], c="lime", s=28)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    axes[2].scatter(pol_angles, pol_intensity, s=45, color="black", label="Processed intensity")
    fit_curve_deg = np.linspace(0, 180, 361)
    fit_curve_theta = np.deg2rad(fit_curve_deg)
    fit_curve = (
        pol_fit["baseline"]
        + pol_fit["cos2_coeff"] * np.cos(2 * fit_curve_theta)
        + pol_fit["sin2_coeff"] * np.sin(2 * fit_curve_theta)
    )
    axes[2].plot(fit_curve_deg, fit_curve, color="tab:red", label="cos(2theta) fit")
    axes[2].set_title("Replica intensity vs polarization")
    axes[2].set_xlabel("Polarization angle (deg)")
    axes[2].set_ylabel("Replica intensity (a.u.)")
    axes[2].legend(loc="best")
    fig.savefig(REPORT_IMG_DIR / "figure_overview.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for angle in sorted(on_maps):
        axes[0].plot(
            energy,
            on_maps[angle][:, nearest_idx(kx, target_kx)],
            label=f"{angle} deg",
            alpha=0.8,
        )
    axes[0].plot(energy, off[:, nearest_idx(kx, target_kx)], color="black", linewidth=2, label="pump off")
    axes[0].axvline(target_energy, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"Energy cuts near kx={target_kx:.3f}")
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("Intensity (a.u.)")
    axes[0].legend(fontsize=8, ncol=2)

    local_diffs = [row["local_difference"] for row in raw_angle_summary]
    axes[1].plot(pol_angles, local_diffs, marker="o", color="tab:blue")
    axes[1].set_title("Local raw-map enhancement at replica window")
    axes[1].set_xlabel("Polarization angle (deg)")
    axes[1].set_ylabel("Pump-on minus off (a.u.)")
    fig.savefig(REPORT_IMG_DIR / "figure_polarization_validation.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    orders = [item["order"] for item in replica_offsets]
    energies = [item["energy"] for item in replica_offsets]
    x = np.arange(len(replica_offsets))
    ax.scatter(x, energies, color="tab:green", s=50, label="Observed replica energy")
    for i, item in enumerate(replica_offsets):
        ax.text(i, energies[i] + 0.012, f"n={item['order']}", ha="center", fontsize=9)
    for order, mean_energy in order_mean_energy.items():
        ax.axhline(mean_energy, linestyle="--", linewidth=1, label=f"mean n={order}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{item['kx']:.3f}" for item in replica_offsets], rotation=45)
    ax.set_xlabel(r"Replica momentum $k_x$ (1/Angstrom)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Replica order energies and spacing")
    ax.legend(loc="best")
    fig.savefig(REPORT_IMG_DIR / "figure_replica_offsets.png", dpi=200)
    plt.close(fig)

    print("Wrote analysis outputs and figures.")


if __name__ == "__main__":
    main()
