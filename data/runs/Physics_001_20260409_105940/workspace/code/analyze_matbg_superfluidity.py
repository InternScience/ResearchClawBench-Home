from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "MATBG Superfluid Stiffness Core Dataset.txt"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float


def parse_array_block(text: str, label: str) -> np.ndarray:
    pattern = re.escape(label) + r"[^\n]*\n(\[[\s\S]*?\])"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing block for {label}")
    array_text = match.group(1)
    cleaned = array_text.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if "," in cleaned:
        values = ast.literal_eval(cleaned)
        return np.array(values, dtype=float)
    inner = cleaned.strip()[1:-1].strip()
    if not inner:
        return np.array([], dtype=float)
    return np.fromstring(inner, sep=" ")


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return FitResult(slope=slope, intercept=intercept, r2=r2)


def align_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    return x[:n], y[:n]


def fit_power_law_temperature(
    temperature: np.ndarray, ds: np.ndarray, tc: float = 1.0, tmax_fraction: float = 0.35
) -> dict:
    temperature, ds = align_xy(temperature, ds)
    mask = (temperature > 0) & (temperature <= tc * tmax_fraction) & (ds > 0)
    t = temperature[mask]
    y = 1.0 - ds[mask] / ds[0]
    keep = y > 0
    t = t[keep]
    y = y[keep]
    lx = np.log(t / tc)
    ly = np.log(y)
    fit = fit_linear(lx, ly)
    amplitude = math.exp(fit.intercept)
    pred = amplitude * (t / tc) ** fit.slope
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "fit_points": int(t.size),
        "n": float(fit.slope),
        "amplitude": float(amplitude),
        "r2": float(fit.r2),
        "rmse": rmse,
        "t_fit": t,
        "y_fit": y,
        "y_pred": pred,
    }


def fit_current_quadratic(current: np.ndarray, ds: np.ndarray, ic_guess: float = 50.0) -> dict:
    current, ds = align_xy(current, ds)
    mask = current <= 0.6 * ic_guess
    i = current[mask]
    y = 1.0 - ds[mask] / ds[0]
    x = i ** 2
    fit = fit_linear(x, y)
    pred = fit.slope * x + fit.intercept
    ic_est = math.sqrt(1.0 / fit.slope) if fit.slope > 0 else float("inf")
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "fit_points": int(i.size),
        "quadratic_coeff": float(fit.slope),
        "offset": float(fit.intercept),
        "r2": float(fit.r2),
        "rmse": rmse,
        "ic_est_nA": float(ic_est),
        "i_fit": i,
        "y_fit": y,
        "y_pred": pred,
    }


def ensure_dirs() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def make_density_figure(data: dict) -> None:
    n_cm2 = data["n_eff"] / 1e4
    plt.figure(figsize=(8, 5.2))
    plt.plot(n_cm2, data["D_s_conv"] / 1e9, label="Conventional Fermi-liquid scale", lw=2.2)
    plt.plot(n_cm2, data["D_s_geom"] / 1e9, label="Quantum geometric scale", lw=2.2)
    plt.plot(n_cm2, data["D_s_exp_hole"] / 1e9, label="Experiment hole-doped", lw=2.4)
    plt.plot(n_cm2, data["D_s_exp_electron"] / 1e9, label="Experiment electron-doped", lw=2.4)
    plt.xlabel(r"Carrier density $n_{\mathrm{eff}}$ (cm$^{-2}$)")
    plt.ylabel(r"Superfluid stiffness $D_s$ ($10^9$ arb. units)")
    plt.title("Carrier-density dependence of MATBG superfluid stiffness")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "density_dependence.png", dpi=220)
    plt.close()


def make_enhancement_figure(data: dict) -> None:
    n_cm2 = data["n_eff"] / 1e4
    hole_conv = data["D_s_exp_hole"] / data["D_s_conv"]
    hole_geom = data["D_s_exp_hole"] / data["D_s_geom"]
    elec_conv = data["D_s_exp_electron"] / data["D_s_conv"]
    elec_geom = data["D_s_exp_electron"] / data["D_s_geom"]
    plt.figure(figsize=(8, 5.2))
    plt.plot(n_cm2, hole_conv, label="Hole / conventional", lw=2.2)
    plt.plot(n_cm2, elec_conv, label="Electron / conventional", lw=2.2)
    plt.plot(n_cm2, hole_geom, label="Hole / geometric", lw=2.2, ls="--")
    plt.plot(n_cm2, elec_geom, label="Electron / geometric", lw=2.2, ls="--")
    plt.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    plt.xlabel(r"Carrier density $n_{\mathrm{eff}}$ (cm$^{-2}$)")
    plt.ylabel("Enhancement factor")
    plt.title("Experimental stiffness enhancement over conventional and geometric scales")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "enhancement_factors.png", dpi=220)
    plt.close()


def make_temperature_figure(data: dict, fits: dict) -> None:
    t_bcs, ds_bcs = align_xy(data["temperature"], data["D_s_bcs"])
    t_nodal, ds_nodal = align_xy(data["temperature"], data["D_s_nodal"])
    t_exp, ds_exp = align_xy(data["temperature"], data["D_s_experimental"])
    t_p25, ds_p25 = align_xy(data["temperature"], data["D_s_power_n2_5"])
    plt.figure(figsize=(8, 5.2))
    plt.plot(t_bcs, ds_bcs, label="BCS/full gap", lw=2.0)
    plt.plot(t_nodal, ds_nodal, label="Nodal linear-T", lw=2.0)
    plt.plot(t_exp, ds_exp, label="Experimental", lw=2.4)
    plt.plot(t_p25, ds_p25, label="Power law n=2.5", lw=2.0, ls="--")
    plt.axvline(1.0, color="black", lw=1.0, alpha=0.5)
    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Superfluid stiffness $D_s / D_s(0)$ (%)")
    plt.title("Temperature dependence and power-law comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "temperature_dependence.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    t = fits["experimental"]["t_fit"]
    y = fits["experimental"]["y_fit"]
    plt.scatter(np.log(t), np.log(y), s=24, label="Experimental low-T data")
    plt.plot(np.log(t), np.log(fits["experimental"]["y_pred"]), lw=2.2, label=f"Fit n={fits['experimental']['n']:.2f}")
    plt.xlabel(r"$\log(T/T_c)$")
    plt.ylabel(r"$\log(1 - D_s(T)/D_s(0))$")
    plt.title("Low-temperature power-law extraction")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "temperature_powerlaw_fit.png", dpi=220)
    plt.close()


def make_current_figure(data: dict, fit: dict) -> None:
    i_dc, ds_dc = align_xy(data["I_dc"], data["D_s_current"])
    i_mw, ds_mw = align_xy(data["I_mw"], data["D_s_mw_exp"])
    plt.figure(figsize=(8, 5.2))
    plt.plot(i_dc, ds_dc, label="DC-current response", lw=2.4)
    plt.plot(i_mw, ds_mw, label="Microwave-current response", lw=2.4)
    plt.xlabel("Current amplitude (nA)")
    plt.ylabel(r"Superfluid stiffness $D_s / D_s(0)$ (%)")
    plt.title("Current-induced suppression of superfluid stiffness")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "current_dependence.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    i = fit["i_fit"]
    y = fit["y_fit"]
    plt.scatter(i**2, y, s=24, label="Low-current data")
    plt.plot(i**2, fit["y_pred"], lw=2.2, label=f"Linear fit in $I^2$, $R^2$={fit['r2']:.4f}")
    plt.xlabel(r"$I^2$ (nA$^2$)")
    plt.ylabel(r"$1 - D_s(I)/D_s(0)$")
    plt.title("Quadratic current scaling at low bias")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "current_quadratic_fit.png", dpi=220)
    plt.close()


def write_summary(data: dict, metrics: dict) -> None:
    lines = []
    lines.append("MATBG superfluid stiffness analysis summary")
    lines.append("")
    lines.append("Carrier-density enhancement metrics")
    lines.append(
        f"- Mean experimental/conventional ratio: hole={metrics['density']['hole_over_conv_mean']:.2f}, electron={metrics['density']['electron_over_conv_mean']:.2f}"
    )
    lines.append(
        f"- Mean experimental/geometric ratio: hole={metrics['density']['hole_over_geom_mean']:.2f}, electron={metrics['density']['electron_over_geom_mean']:.2f}"
    )
    lines.append(
        f"- Hole/electron asymmetry (mean relative difference): {metrics['density']['doping_asymmetry_mean_pct']:.2f}%"
    )
    lines.append("")
    lines.append("Temperature fits")
    for name, fit in metrics["temperature"].items():
        lines.append(
            f"- {name}: n={fit['n']:.3f}, amplitude={fit['amplitude']:.4f}, R2={fit['r2']:.5f}, RMSE={fit['rmse']:.5f}, points={fit['fit_points']}"
        )
    lines.append("")
    lines.append("Current fit")
    cur = metrics["current"]
    lines.append(
        f"- Low-current quadratic coefficient={cur['quadratic_coeff']:.6f}, offset={cur['offset']:.6f}, R2={cur['r2']:.5f}, Ic_est={cur['ic_est_nA']:.2f} nA"
    )
    (OUTPUTS / "analysis_summary.txt").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    text = DATA_FILE.read_text()

    density_data = {
        "n_eff": parse_array_block(text, "Carrier Density Data"),
        "D_s_conv": parse_array_block(text, "Conventional Superfluid Stiffness"),
        "D_s_geom": parse_array_block(text, "Quantum Geometric Superfluid Stiffness"),
        "D_s_exp_hole": parse_array_block(text, "Experimental Superfluid Stiffness Hole-doped"),
        "D_s_exp_electron": parse_array_block(text, "Experimental Superfluid Stiffness Electron-doped"),
    }
    temperature_data = {
        "temperature": parse_array_block(text, "Temperature Array"),
        "D_s_bcs": parse_array_block(text, "BCS Model Data"),
        "D_s_nodal": parse_array_block(text, "Nodal Superconductor Data"),
        "D_s_power_n2": parse_array_block(text, "Power Law n=2.0 Data"),
        "D_s_power_n2_5": parse_array_block(text, "Power Law n=2.5 Data"),
        "D_s_power_n3": parse_array_block(text, "Power Law n=3.0 Data"),
        "D_s_experimental": parse_array_block(text, "Experimental Data with Noise"),
    }
    current_data = {
        "I_dc": parse_array_block(text, "DC Current Array"),
        "D_s_current": parse_array_block(text, "Experimental DC Data"),
        "I_mw": parse_array_block(text, "Microwave Current Amplitude"),
        "D_s_mw_exp": parse_array_block(text, "Experimental Microwave Data"),
    }

    temp_fits = {
        "bcs_reference": fit_power_law_temperature(temperature_data["temperature"], temperature_data["D_s_bcs"]),
        "power_n2_5_reference": fit_power_law_temperature(
            temperature_data["temperature"], temperature_data["D_s_power_n2_5"]
        ),
        "experimental": fit_power_law_temperature(
            temperature_data["temperature"], temperature_data["D_s_experimental"]
        ),
    }
    current_fit = fit_current_quadratic(current_data["I_dc"], current_data["D_s_current"])

    metrics = {
        "density": {
            "hole_over_conv_mean": float(np.mean(density_data["D_s_exp_hole"] / density_data["D_s_conv"])),
            "electron_over_conv_mean": float(np.mean(density_data["D_s_exp_electron"] / density_data["D_s_conv"])),
            "hole_over_geom_mean": float(np.mean(density_data["D_s_exp_hole"] / density_data["D_s_geom"])),
            "electron_over_geom_mean": float(np.mean(density_data["D_s_exp_electron"] / density_data["D_s_geom"])),
            "doping_asymmetry_mean_pct": float(
                np.mean(
                    np.abs(density_data["D_s_exp_hole"] - density_data["D_s_exp_electron"])
                    / ((density_data["D_s_exp_hole"] + density_data["D_s_exp_electron"]) / 2.0)
                )
                * 100.0
            ),
        },
        "temperature": {
            key: {
                k: v
                for k, v in value.items()
                if k not in {"t_fit", "y_fit", "y_pred"}
            }
            for key, value in temp_fits.items()
        },
        "current": {
            k: v
            for k, v in current_fit.items()
            if k not in {"i_fit", "y_fit", "y_pred"}
        },
    }

    save_json(OUTPUTS / "metrics.json", metrics)
    write_summary({**density_data, **temperature_data, **current_data}, metrics)
    make_density_figure(density_data)
    make_enhancement_figure(density_data)
    make_temperature_figure(temperature_data, temp_fits)
    make_current_figure(current_data, current_fit)


if __name__ == "__main__":
    main()
