#!/usr/bin/env python3
"""Posterior-driven ultralight-boson constraints from black-hole superradiance.

This script implements a lightweight Bayesian exclusion framework based on the
analytic superradiance rate formula in Arvanitaki & Dubovsky (2011). It uses
the full posterior samples of black-hole mass and spin rather than point
estimates, evaluates whether each sample would be incompatible with the
existence of a boson of mass ``mu``, and then marginalizes over the posterior.

Outputs:
  - CSV tables in ``outputs/``
  - Figures in ``report/images/``
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_superradiance")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


# SI constants
G_SI = 6.67430e-11
C_SI = 299792458.0
HBAR_SI = 1.054571817e-34
EV_J = 1.602176634e-19
M_SUN_KG = 1.98847e30
YEAR_S = 365.25 * 24.0 * 3600.0

# Natural-unit conversion for alpha = mu * r_g.
ALPHA_COEFF = G_SI * M_SUN_KG / C_SI**3 * (EV_J / HBAR_SI)
RG_TIME_PER_MSUN_S = G_SI * M_SUN_KG / C_SI**3

# Planck mass used in the reference-paper scaling relations.
MPL_GEV = 1.2209e19
MPL_EV = MPL_GEV * 1.0e9

EDDINGTON_TIME_YR = 4.0e8
HUBBLE_TIME_YR = 1.38e10
N_EFOLD = 100.0

MU_GRID = np.logspace(-21.5, -9.0, 500)
MODES: Tuple[Tuple[int, int], ...] = ((1, 0), (2, 0), (3, 0), (4, 0), (4, 1))


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path
    label: str


SOURCES = (
    SourceConfig(
        name="M33_X-7",
        path=DATA_DIR / "M33_X-7_samples.dat",
        label="M33 X-7",
    ),
    SourceConfig(
        name="IRAS_09149-6206",
        path=DATA_DIR / "IRAS_09149-6206_samples.dat",
        label="IRAS 09149-6206",
    ),
)


def load_samples(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def quantile_summary(values: np.ndarray) -> Dict[str, float]:
    q = np.quantile(values, [0.05, 0.16, 0.5, 0.84, 0.95])
    return {
        "q05": float(q[0]),
        "q16": float(q[1]),
        "q50": float(q[2]),
        "q84": float(q[3]),
        "q95": float(q[4]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def omega_horizon_hat(spin: np.ndarray) -> np.ndarray:
    """Dimensionless horizon frequency rg * Omega_H."""
    spin = np.clip(spin, 0.0, 0.999999)
    return spin / (2.0 * (1.0 + np.sqrt(1.0 - spin**2)))


def r_plus_hat(spin: np.ndarray) -> np.ndarray:
    return 1.0 + np.sqrt(1.0 - np.clip(spin, 0.0, 0.999999) ** 2)


def critical_spin(alpha: np.ndarray, m: int) -> np.ndarray:
    """Critical dimensionless spin where alpha = m * rg * Omega_H."""
    x = alpha / float(m)
    crit = np.full_like(alpha, np.nan, dtype=float)
    mask = (x > 0.0) & (x < 0.5)
    crit[mask] = 4.0 * x[mask] / (1.0 + 4.0 * x[mask] ** 2)
    return crit


def clmn_factor(alpha: np.ndarray, spin: np.ndarray, m: int, n: int) -> np.ndarray:
    l = m
    rp = r_plus_hat(spin)
    omega_hat = omega_horizon_hat(spin)
    delta = m * omega_hat - alpha
    prefactor = (
        2.0 ** (4 * l + 2)
        * math.factorial(2 * l + n + 1)
        / ((l + n + 1) ** (2 * l + 4) * math.factorial(n))
        * (math.factorial(l) / (math.factorial(2 * l) * math.factorial(2 * l + 1))) ** 2
    )
    product = np.ones_like(alpha)
    spin_term = 1.0 - np.clip(spin, 0.0, 0.999999) ** 2
    for j in range(1, l + 1):
        product *= j**2 * spin_term + 4.0 * rp**2 * delta**2
    return prefactor * product


def gamma_rg(alpha: np.ndarray, spin: np.ndarray, m: int, n: int) -> np.ndarray:
    """Dimensionless rate Gamma * r_g from Eq. (18)."""
    rp = r_plus_hat(spin)
    omega_hat = omega_horizon_hat(spin)
    delta = m * omega_hat - alpha
    clmn = clmn_factor(alpha, spin, m, n)
    gamma = 2.0 * alpha ** (4 * m + 5) * rp * delta * clmn
    gamma = np.where(np.isfinite(gamma), gamma, 0.0)
    return np.maximum(gamma, 0.0)


def rg_time_years(mass_msun: np.ndarray) -> np.ndarray:
    return RG_TIME_PER_MSUN_S * mass_msun / YEAR_S


def largest_true_interval(mu_grid: np.ndarray, mask: np.ndarray) -> Tuple[float, float] | Tuple[None, None]:
    if not np.any(mask):
        return (None, None)

    indices = np.where(mask)[0]
    blocks: List[np.ndarray] = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    block = max(blocks, key=len)
    return float(mu_grid[block[0]]), float(mu_grid[block[-1]])


def all_true_intervals(mu_grid: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    if not np.any(mask):
        return []
    indices = np.where(mask)[0]
    blocks: List[np.ndarray] = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [(float(mu_grid[block[0]]), float(mu_grid[block[-1]])) for block in blocks if len(block)]


def point_estimate_curve(samples: np.ndarray, mu_grid: np.ndarray, timescale_years: float) -> np.ndarray:
    median_mass = np.median(samples[:, 0])
    median_spin = np.median(samples[:, 1])
    mass = np.array([median_mass])
    spin = np.array([median_spin])
    result = evaluate_exclusion(samples=np.column_stack([mass, spin]), mu_grid=mu_grid, timescale_years=timescale_years)
    return result["probability"]


def evaluate_exclusion(samples: np.ndarray, mu_grid: np.ndarray, timescale_years: float) -> Dict[str, np.ndarray]:
    """Evaluate sample-wise exclusion probabilities for one source and one timescale."""
    mass = samples[:, 0]
    spin = samples[:, 1]
    ns = len(samples)
    nm = len(mu_grid)

    alpha = ALPHA_COEFF * mass[:, None] * mu_grid[None, :]
    rg_years = rg_time_years(mass)[:, None]
    spin_2d = spin[:, None]

    excluded_any = np.zeros((ns, nm), dtype=bool)
    best_rate = np.zeros((ns, nm), dtype=float)
    best_mode = np.zeros((ns, nm), dtype=int)
    best_crit_spin = np.full((ns, nm), np.nan)
    best_fa_min = np.full((ns, nm), np.inf)

    for mode_index, (m, n) in enumerate(MODES, start=1):
        crit = critical_spin(alpha, m)
        above_trajectory = spin_2d > crit
        gamma = gamma_rg(alpha, spin_2d, m=m, n=n)
        active = gamma > 0.0
        spin_down_time = np.where(active, N_EFOLD * rg_years / gamma, np.inf)
        fast_enough = spin_down_time < timescale_years
        excluded_mode = active & fast_enough & above_trajectory & np.isfinite(crit)

        better = excluded_mode & (gamma > best_rate)
        best_rate = np.where(better, gamma, best_rate)
        best_mode = np.where(better, m, best_mode)
        best_crit_spin = np.where(better, crit, best_crit_spin)

        delta_spin = np.maximum(spin_2d - crit, 0.0)
        # Estimated cloud fraction needed to remove the excess spin:
        # M_cloud / M_BH ~ alpha * Delta a_* / m.
        required_fraction = alpha * delta_spin / float(m)
        fa_min = MPL_EV * np.sqrt(np.maximum(required_fraction, 0.0) * alpha**2 / (2.0 * m**4))
        fa_min = np.where(excluded_mode, fa_min, np.inf)
        best_fa_min = np.minimum(best_fa_min, fa_min)

        excluded_any |= excluded_mode

    probability = excluded_any.mean(axis=0)
    return {
        "probability": probability,
        "excluded_any": excluded_any,
        "best_mode": best_mode,
        "best_rate": best_rate,
        "best_crit_spin": best_crit_spin,
        "best_fa_min": best_fa_min,
    }


def fa_quantile_for_target_probability(fa_min: np.ndarray, target_probability: float) -> np.ndarray:
    """Return the minimum f_a that yields the requested exclusion probability."""
    ns, nm = fa_min.shape
    result = np.full(nm, np.inf)
    for i in range(nm):
        finite = np.isfinite(fa_min[:, i])
        p0 = finite.mean()
        if p0 < target_probability:
            continue
        q = target_probability / p0
        q = min(max(q, 0.0), 1.0)
        result[i] = np.quantile(fa_min[finite, i], q)
    return result


def combined_fa_limit_for_target_probability(
    fa_min_by_source: Dict[str, np.ndarray],
    target_probability: float,
) -> np.ndarray:
    """Combine independent sources at fixed coupling and solve for the target probability."""
    source_names = list(fa_min_by_source)
    nm = fa_min_by_source[source_names[0]].shape[1]
    result = np.full(nm, np.inf)

    for i in range(nm):
        per_source = {}
        source_pmax = []
        for name in source_names:
            vals = fa_min_by_source[name][:, i]
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size:
                vals = np.sort(vals)
                per_source[name] = vals
                source_pmax.append(vals.size / fa_min_by_source[name].shape[0])
            else:
                per_source[name] = np.array([], dtype=float)
                source_pmax.append(0.0)

        if all(arr.size == 0 for arr in per_source.values()):
            continue

        pmax_total = 1.0 - np.prod(1.0 - np.array(source_pmax))
        if pmax_total < target_probability:
            continue

        lo = min(arr[0] for arr in per_source.values() if arr.size)
        hi = max(arr[-1] for arr in per_source.values() if arr.size)

        for _ in range(60):
            mid = math.sqrt(lo * hi)
            source_probs = []
            for name in source_names:
                vals = per_source[name]
                if vals.size == 0:
                    source_probs.append(0.0)
                else:
                    count = np.searchsorted(vals, mid, side="right")
                    source_probs.append(count / fa_min_by_source[name].shape[0])
            total_prob = 1.0 - np.prod(1.0 - np.array(source_probs))
            if total_prob >= target_probability:
                hi = mid
            else:
                lo = mid
        result[i] = hi
    return result


def summarize_source(name: str, label: str, samples: np.ndarray) -> pd.DataFrame:
    mass_stats = quantile_summary(samples[:, 0])
    spin_stats = quantile_summary(samples[:, 1])
    rows = []
    for param, stats in [("mass_msun", mass_stats), ("spin", spin_stats)]:
        row = {"source": name, "label": label, "parameter": param}
        row.update(stats)
        rows.append(row)
    corr = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    rows.append(
        {
            "source": name,
            "label": label,
            "parameter": "mass_spin_correlation",
            "q05": np.nan,
            "q16": np.nan,
            "q50": np.nan,
            "q84": np.nan,
            "q95": np.nan,
            "mean": corr,
            "std": np.nan,
        }
    )
    return pd.DataFrame(rows)


def build_probability_table(
    source_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mu_grid: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for source_name, source_dict in source_results.items():
        for benchmark_name, res in source_dict.items():
            frames.append(
                pd.DataFrame(
                    {
                        "source": source_name,
                        "benchmark": benchmark_name,
                        "mu_eV": mu_grid,
                        "exclusion_probability": res["probability"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def plot_posteriors(samples_map: Dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig)
    colors = {"M33_X-7": "#1f77b4", "IRAS_09149-6206": "#d62728"}

    for idx, source in enumerate(SOURCES):
        ax = fig.add_subplot(gs[0, idx])
        samples = samples_map[source.name]
        step = max(1, len(samples) // 2500)
        ax.scatter(
            samples[::step, 0],
            samples[::step, 1],
            s=8,
            alpha=0.35,
            color=colors[source.name],
            edgecolors="none",
        )
        ax.set_xlabel(r"Black-hole mass [$M_\odot$]")
        ax.set_ylabel(r"Spin $a_*$")
        ax.set_title(source.label)
        if samples[:, 0].max() / samples[:, 0].min() > 50:
            ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle("Posterior samples used in the superradiance analysis", fontsize=13)
    fig.savefig(REPORT_IMG_DIR / "posterior_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_exclusion_curves(
    source_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    combined_results: Dict[str, Dict[str, np.ndarray]],
    mu_grid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True, constrained_layout=True)
    colors = {"M33_X-7": "#1f77b4", "IRAS_09149-6206": "#d62728", "Combined": "#111111"}
    labels = {"eddington": "Eddington benchmark", "hubble": "Hubble benchmark"}

    for ax, benchmark in zip(axes, ["eddington", "hubble"]):
        for source in SOURCES:
            ax.plot(
                mu_grid,
                source_results[source.name][benchmark]["probability"],
                color=colors[source.name],
                lw=2.0,
                label=source.label,
            )
        ax.plot(
            mu_grid,
            combined_results[benchmark]["probability"],
            color=colors["Combined"],
            lw=2.4,
            label="Combined",
        )
        ax.axhline(0.95, color="0.45", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r"Boson mass $\mu$ [eV]")
        ax.set_title(labels[benchmark])
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Posterior exclusion probability")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Posterior-integrated exclusion probability for ultralight boson masses", fontsize=13)
    fig.savefig(REPORT_IMG_DIR / "mass_exclusion_probability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regge_diagnostics(
    samples_map: Dict[str, np.ndarray],
    source_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    mu_grid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for ax, source in zip(axes, SOURCES):
        result = source_results[source.name]["eddington"]["probability"]
        mu_peak = mu_grid[np.argmax(result)]
        samples = samples_map[source.name]
        step = max(1, len(samples) // 2500)
        ax.scatter(samples[::step, 0], samples[::step, 1], s=9, alpha=0.3, color="0.2", edgecolors="none")

        mass_line = np.logspace(np.log10(samples[:, 0].min() * 0.85), np.log10(samples[:, 0].max() * 1.15), 400)
        alpha_line = ALPHA_COEFF * mass_line * mu_peak
        for color, m in zip(palette, [1, 2, 3, 4]):
            curve = critical_spin(alpha_line, m)
            ax.plot(mass_line, curve, lw=2.0, color=color, label=fr"$m=l={m}$")

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"Black-hole mass [$M_\odot$]")
        ax.set_ylabel(r"Spin $a_*$")
        ax.set_title(f"{source.label}: representative mass $\\mu={mu_peak:.2e}$ eV")
        ax.grid(alpha=0.25, lw=0.4)
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Regge-plane diagnostics at the most strongly excluded boson mass", fontsize=13)
    fig.savefig(REPORT_IMG_DIR / "regge_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_self_interaction_limits(mu_grid: np.ndarray, fa_95: np.ndarray, lambda_max: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.6), sharex=True, constrained_layout=True)

    mask_fa = np.isfinite(fa_95) & (fa_95 > 0.0)
    axes[0].plot(mu_grid[mask_fa], fa_95[mask_fa] / 1.0e9, color="#2c3e50", lw=2.2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$f_{a,95}^{\min}$ [GeV]")
    axes[0].set_title("Self-interaction benchmark from the cloud-collapse criterion")
    axes[0].grid(alpha=0.25, lw=0.4)

    mask_lam = np.isfinite(lambda_max) & (lambda_max > 0.0)
    axes[1].plot(mu_grid[mask_lam], lambda_max[mask_lam], color="#8e2d2d", lw=2.2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Boson mass $\mu$ [eV]")
    axes[1].set_ylabel(r"$\lambda_{95}^{\max}\simeq(\mu/f_a)^2$")
    axes[1].grid(alpha=0.25, lw=0.4)

    fig.savefig(REPORT_IMG_DIR / "self_interaction_limits.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation(
    source_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    point_curves: Dict[str, np.ndarray],
    mu_grid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
    colors = {"posterior": "#111111", "point": "#d95f02"}

    for ax, source in zip(axes, SOURCES):
        ax.plot(
            mu_grid,
            source_results[source.name]["eddington"]["probability"],
            color=colors["posterior"],
            lw=2.2,
            label="Posterior-integrated",
        )
        ax.plot(
            mu_grid,
            point_curves[source.name],
            color=colors["point"],
            lw=2.0,
            ls="--",
            label="Point estimate",
        )
        ax.axhline(0.95, color="0.45", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel(r"Boson mass $\mu$ [eV]")
        ax.set_title(source.label)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Exclusion probability")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Validation: full-posterior inference versus median-parameter approximation", fontsize=13)
    fig.savefig(REPORT_IMG_DIR / "validation_posterior_vs_point.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    samples_map = {source.name: load_samples(source.path) for source in SOURCES}

    summary_df = pd.concat(
        [summarize_source(source.name, source.label, samples_map[source.name]) for source in SOURCES],
        ignore_index=True,
    )
    summary_df.to_csv(OUTPUT_DIR / "posterior_summary.csv", index=False)

    source_results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    point_curves: Dict[str, np.ndarray] = {}
    for source in SOURCES:
        samples = samples_map[source.name]
        source_results[source.name] = {
            "eddington": evaluate_exclusion(samples, MU_GRID, EDDINGTON_TIME_YR),
            "hubble": evaluate_exclusion(samples, MU_GRID, HUBBLE_TIME_YR),
        }
        point_curves[source.name] = point_estimate_curve(samples, MU_GRID, EDDINGTON_TIME_YR)

    probability_df = build_probability_table(source_results, MU_GRID)

    combined_results: Dict[str, Dict[str, np.ndarray]] = {}
    for benchmark in ["eddington", "hubble"]:
        p_stack = np.vstack([source_results[source.name][benchmark]["probability"] for source in SOURCES])
        combined_probability = 1.0 - np.prod(1.0 - p_stack, axis=0)
        combined_results[benchmark] = {"probability": combined_probability}
        probability_df = pd.concat(
            [
                probability_df,
                pd.DataFrame(
                    {
                        "source": "Combined",
                        "benchmark": benchmark,
                        "mu_eV": MU_GRID,
                        "exclusion_probability": combined_probability,
                    }
                ),
            ],
            ignore_index=True,
        )
    probability_df.to_csv(OUTPUT_DIR / "mass_exclusion_probabilities.csv", index=False)

    # Conservative self-interaction limit from the Eddington benchmark.
    fa_min_by_source = {
        source.name: source_results[source.name]["eddington"]["best_fa_min"] for source in SOURCES
    }
    fa_95 = combined_fa_limit_for_target_probability(fa_min_by_source, target_probability=0.95)
    lambda_max = np.where(np.isfinite(fa_95) & (fa_95 > 0.0), (MU_GRID / fa_95) ** 2, np.nan)
    pd.DataFrame(
        {
            "mu_eV": MU_GRID,
            "fa95_min_GeV": fa_95 / 1.0e9,
            "lambda95_max": lambda_max,
        }
    ).to_csv(OUTPUT_DIR / "self_interaction_limits.csv", index=False)

    intervals = []
    interval_rows = []
    for source_name in ["M33_X-7", "IRAS_09149-6206", "Combined"]:
        for benchmark in ["eddington", "hubble"]:
            if source_name == "Combined":
                prob = combined_results[benchmark]["probability"]
            else:
                prob = source_results[source_name][benchmark]["probability"]
            mu_lo, mu_hi = largest_true_interval(MU_GRID, prob >= 0.95)
            bands = all_true_intervals(MU_GRID, prob >= 0.95)
            intervals.append(
                {
                    "source": source_name,
                    "benchmark": benchmark,
                    "mu95_lower_eV": mu_lo,
                    "mu95_upper_eV": mu_hi,
                    "peak_probability": float(prob.max()),
                    "peak_mu_eV": float(MU_GRID[np.argmax(prob)]),
                }
            )
            for band_index, (band_lo, band_hi) in enumerate(bands, start=1):
                interval_rows.append(
                    {
                        "source": source_name,
                        "benchmark": benchmark,
                        "band_index": band_index,
                        "mu95_lower_eV": band_lo,
                        "mu95_upper_eV": band_hi,
                    }
                )
    intervals_df = pd.DataFrame(intervals)
    intervals_df.to_csv(OUTPUT_DIR / "mass_exclusion_intervals.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(OUTPUT_DIR / "mass_exclusion_bands.csv", index=False)

    plot_posteriors(samples_map)
    plot_exclusion_curves(source_results, combined_results, MU_GRID)
    plot_regge_diagnostics(samples_map, source_results, MU_GRID)
    plot_self_interaction_limits(MU_GRID, fa_95, lambda_max)
    plot_validation(source_results, point_curves, MU_GRID)

    summary_payload = {
        "alpha_coeff": ALPHA_COEFF,
        "eddington_time_yr": EDDINGTON_TIME_YR,
        "hubble_time_yr": HUBBLE_TIME_YR,
        "n_efold_factor": N_EFOLD,
        "sources": [source.name for source in SOURCES],
    }
    with (OUTPUT_DIR / "analysis_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)


if __name__ == "__main__":
    main()
