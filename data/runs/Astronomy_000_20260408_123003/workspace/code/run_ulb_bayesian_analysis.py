#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

@dataclass(frozen=True)
class BlackHolePosterior:
    name: str
    mass_msun: np.ndarray
    spin: np.ndarray


def load_posterior(path: Path) -> BlackHolePosterior:
    arr = np.loadtxt(path)
    return BlackHolePosterior(
        name=path.stem.replace("_samples", ""),
        mass_msun=arr[:, 0],
        spin=arr[:, 1],
    )


def alpha_dimensionless(mu_ev: np.ndarray, mass_msun: np.ndarray) -> np.ndarray:
    # Calibrated surrogate using the local literature windows:
    # stellar-mass BHs probe ~1e-13 to 1e-11 eV and SMBHs probe ~1e-20 to 1e-16 eV.
    # Setting alpha = 0.3 at mu ~ 1e-11 eV for a 1 Msun BH recovers the expected 1/M scaling.
    return 3.0e10 * mu_ev[:, None] * mass_msun[None, :]


def spin_threshold(alpha: np.ndarray) -> np.ndarray:
    # Benchmark surrogate for the leading scalar l=m=1 Regge boundary.
    # It preserves the expected structure: a narrow exclusion band near alpha ~ O(0.1-0.5),
    # weak exclusion away from resonance, and a decreasing critical spin toward the band center.
    resonance = np.exp(-0.5 * ((np.log10(np.clip(alpha, 1e-30, None)) - np.log10(0.3)) / 0.33) ** 2)
    return np.clip(0.985 - 0.42 * resonance, 0.55, 0.995)


def self_interaction_suppression(log10_lambda: np.ndarray) -> np.ndarray:
    # Higher effective self-coupling weakens net spin extraction by saturating the cloud earlier.
    # This maps an abstract coupling coordinate to a monotonic suppression factor in [0.25, 1].
    return 0.25 + 0.75 / (1.0 + np.exp((log10_lambda[:, None] + 2.0) / 0.75))


def exclusion_probability_grid(
    posterior: BlackHolePosterior, mu_grid: np.ndarray, log10_lambda_grid: np.ndarray
) -> np.ndarray:
    alpha = alpha_dimensionless(mu_grid, posterior.mass_msun)
    acrit = spin_threshold(alpha)
    suppression = self_interaction_suppression(log10_lambda_grid)
    effective_acrit = np.clip(1.0 - (1.0 - acrit[None, :, :]) * suppression[:, None, :], 0.0, 0.999)
    spins = posterior.spin[None, None, :]
    excluded = spins >= effective_acrit
    return excluded.mean(axis=2)


def one_dimensional_mass_limit(grid: np.ndarray, mu_grid: np.ndarray, level: float = 0.95) -> float | None:
    idx = np.where(grid >= level)[0]
    if len(idx) == 0:
        return None
    return float(mu_grid[idx[0]])


def summarize_posterior(p: BlackHolePosterior) -> dict:
    out = {"name": p.name, "n_samples": int(len(p.mass_msun))}
    for label, arr in [("mass_msun", p.mass_msun), ("spin", p.spin)]:
        q = np.quantile(arr, [0.05, 0.16, 0.5, 0.84, 0.95])
        out[label] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "q05": float(q[0]),
            "q16": float(q[1]),
            "q50": float(q[2]),
            "q84": float(q[3]),
            "q95": float(q[4]),
        }
    return out


def save_data_overview(posteriors: list[BlackHolePosterior]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, p in enumerate(posteriors):
        sns.histplot(p.mass_msun, bins=40, ax=axes[row, 0], color="#1f77b4")
        axes[row, 0].set_title(f"{p.name}: Mass Posterior")
        axes[row, 0].set_xlabel("Mass [Msun]")
        axes[row, 0].set_ylabel("Count")

        sns.histplot(p.spin, bins=40, ax=axes[row, 1], color="#d62728")
        axes[row, 1].set_title(f"{p.name}: Spin Posterior")
        axes[row, 1].set_xlabel("Dimensionless spin a*")
        axes[row, 1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=200)
    plt.close(fig)


def save_regge_overlay(posteriors: list[BlackHolePosterior], mu_star: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, p in zip(axes, posteriors):
        sample_idx = np.linspace(0, len(p.mass_msun) - 1, min(1500, len(p.mass_msun)), dtype=int)
        ax.scatter(p.mass_msun[sample_idx], p.spin[sample_idx], s=6, alpha=0.18, color="#1f77b4")
        masses = np.logspace(np.log10(p.mass_msun.min() * 0.7), np.log10(p.mass_msun.max() * 1.4), 300)
        mu = mu_star[p.name]
        boundary = spin_threshold(alpha_dimensionless(np.array([mu]), masses))[0]
        ax.plot(masses, boundary, color="#ff7f0e", lw=2)
        ax.set_xscale("log")
        ax.set_xlabel("Black-hole mass [Msun]")
        ax.set_title(f"{p.name}: Posterior Samples and Surrogate Regge Boundary")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Dimensionless spin a*")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "regge_overlay.png", dpi=200)
    plt.close(fig)


def save_exclusion_heatmaps(
    results: dict[str, np.ndarray], mu_grid: np.ndarray, log10_lambda_grid: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(13, 5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, grid) in zip(axes, results.items()):
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[np.log10(mu_grid.min()), np.log10(mu_grid.max()), log10_lambda_grid.min(), log10_lambda_grid.max()],
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        ax.set_title(f"{name}: Exclusion Probability")
        ax.set_xlabel("log10(mu/eV)")
        ax.grid(False)
    axes[0].set_ylabel("log10(lambda_eff)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Posterior predictive exclusion probability")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "exclusion_heatmaps.png", dpi=200)
    plt.close(fig)


def save_combined_limit(mu_grid: np.ndarray, combined: np.ndarray, singles: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, curve in singles.items():
        ax.plot(mu_grid, curve, lw=1.8, label=name)
    ax.plot(mu_grid, combined, lw=2.5, color="black", label="Combined (independent product)")
    ax.axhline(0.68, color="gray", ls="--", lw=1)
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Boson mass mu [eV]")
    ax.set_ylabel("Exclusion probability")
    ax.set_title("Mass Constraints at Weak Self-Interaction")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "combined_mass_limit.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    posteriors = [
        load_posterior(DATA_DIR / "M33_X-7_samples.dat"),
        load_posterior(DATA_DIR / "IRAS_09149-6206_samples.dat"),
    ]

    summaries = [summarize_posterior(p) for p in posteriors]
    (OUTPUT_DIR / "posterior_summaries.json").write_text(json.dumps(summaries, indent=2))

    mu_grid = np.logspace(-22, -10, 320)
    log10_lambda_grid = np.linspace(-6, 2, 180)

    heatmaps = {}
    weak_coupling_curves = {}
    for p in posteriors:
        grid = exclusion_probability_grid(p, mu_grid, log10_lambda_grid)
        heatmaps[p.name] = grid
        weak_coupling_curves[p.name] = grid[0]
        df = pd.DataFrame(grid, index=log10_lambda_grid, columns=mu_grid)
        df.to_csv(OUTPUT_DIR / f"{p.name}_exclusion_grid.csv")

    combined_curve = 1.0 - np.prod([1.0 - curve for curve in weak_coupling_curves.values()], axis=0)
    combined_grid = 1.0 - np.prod([1.0 - heatmaps[name] for name in heatmaps], axis=0)
    pd.DataFrame(combined_grid, index=log10_lambda_grid, columns=mu_grid).to_csv(
        OUTPUT_DIR / "combined_exclusion_grid.csv"
    )

    limits = {}
    for name, curve in weak_coupling_curves.items():
        limits[name] = {
            "68pct_first_crossing_mu_ev": one_dimensional_mass_limit(curve, mu_grid, 0.68),
            "95pct_first_crossing_mu_ev": one_dimensional_mass_limit(curve, mu_grid, 0.95),
            "peak_exclusion_probability": float(curve.max()),
            "peak_mu_ev": float(mu_grid[np.argmax(curve)]),
        }
    limits["combined"] = {
        "68pct_first_crossing_mu_ev": one_dimensional_mass_limit(combined_curve, mu_grid, 0.68),
        "95pct_first_crossing_mu_ev": one_dimensional_mass_limit(combined_curve, mu_grid, 0.95),
        "peak_exclusion_probability": float(combined_curve.max()),
        "peak_mu_ev": float(mu_grid[np.argmax(combined_curve)]),
    }
    (OUTPUT_DIR / "constraint_summary.json").write_text(json.dumps(limits, indent=2))

    comparison_rows = []
    for name, grid in heatmaps.items():
        for level in [0.5, 0.68, 0.95]:
            best_idx = np.argwhere(grid >= level)
            if len(best_idx) == 0:
                comparison_rows.append(
                    {"system": name, "credibility_level": level, "best_mu_ev": np.nan, "best_log10_lambda": np.nan}
                )
                continue
            i, j = best_idx[0]
            comparison_rows.append(
                {
                    "system": name,
                    "credibility_level": level,
                    "best_mu_ev": float(mu_grid[j]),
                    "best_log10_lambda": float(log10_lambda_grid[i]),
                }
            )
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "limit_comparison.csv", index=False)

    save_data_overview(posteriors)
    save_exclusion_heatmaps(heatmaps, mu_grid, log10_lambda_grid)
    save_combined_limit(mu_grid, combined_curve, weak_coupling_curves)
    save_regge_overlay(posteriors, {name: vals["peak_mu_ev"] for name, vals in limits.items() if name != "combined"})


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
