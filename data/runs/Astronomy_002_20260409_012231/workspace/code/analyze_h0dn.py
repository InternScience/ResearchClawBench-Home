#!/usr/bin/env python3
"""Local Distance Network reconstruction for the minimal H0DN dataset."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "H0DN_MinimalDataset.txt"
OUTPUTS = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


def load_dataset(path: Path) -> dict:
    namespace: dict = {}
    exec(path.read_text(), {}, namespace)
    return namespace


def dm_to_distance_mpc(mu: np.ndarray | float) -> np.ndarray | float:
    return 10 ** ((np.asarray(mu) - 25.0) / 5.0)


def redshift_distance_modulus(z: float, h0: float, c_km: float) -> float:
    d_mpc = (c_km * z) / h0
    return 5.0 * math.log10(d_mpc) + 25.0


def hubble_mu_error(z: float, mag_err: float, pv_kms: float, c_km: float) -> float:
    frac = pv_kms / (c_km * z)
    pv_mag = (5.0 / math.log(10.0)) * frac
    return math.sqrt(mag_err**2 + pv_mag**2)


def weighted_mean(values: list[float], errors: list[float]) -> tuple[float, float]:
    weights = 1.0 / np.square(errors)
    mean = float(np.sum(weights * np.asarray(values)) / np.sum(weights))
    err = float(np.sqrt(1.0 / np.sum(weights)))
    return mean, err


def build_host_dataframe(ds: dict) -> pd.DataFrame:
    rows = []
    for host, method, anchor, mu_meas, err_meas in ds["host_measurements"]:
        sys = ds["method_anchor_err"].get((method, anchor), 0.0)
        anchor_err = ds["anchors"][anchor]["err"]
        total_err = math.sqrt(err_meas**2 + sys**2 + anchor_err**2)
        rows.append(
            {
                "host": host,
                "method": method,
                "anchor": anchor,
                "mu": mu_meas,
                "err_stat": err_meas,
                "err_sys": sys,
                "anchor_err": anchor_err,
                "err_total": total_err,
            }
        )
    return pd.DataFrame(rows)


def aggregate_host_distances(
    host_df: pd.DataFrame,
    allowed_methods: list[str] | None = None,
    allowed_anchors: list[str] | None = None,
) -> pd.DataFrame:
    df = host_df.copy()
    if allowed_methods is not None:
        df = df[df["method"].isin(allowed_methods)]
    if allowed_anchors is not None:
        df = df[df["anchor"].isin(allowed_anchors)]

    grouped = []
    for host, sub in df.groupby("host"):
        mu, err = weighted_mean(sub["mu"].tolist(), sub["err_total"].tolist())
        grouped.append(
            {
                "host": host,
                "mu": mu,
                "err": err,
                "n_measurements": int(len(sub)),
                "methods": ",".join(sorted(set(sub["method"]))),
                "anchors": ",".join(sorted(set(sub["anchor"]))),
            }
        )
    return pd.DataFrame(grouped).sort_values("host").reset_index(drop=True)


def calibrate_sne_absolute_magnitude(host_mu: pd.DataFrame, ds: dict) -> tuple[pd.DataFrame, float, float]:
    mu_map = dict(zip(host_mu["host"], host_mu["mu"]))
    err_map = dict(zip(host_mu["host"], host_mu["err"]))
    rows = []
    for host, m_b, err_m in ds["sneia_calibrators"]:
        if host not in mu_map:
            continue
        abs_mag = m_b - mu_map[host]
        err_abs = math.sqrt(err_m**2 + err_map[host] ** 2)
        rows.append({"host": host, "m_b": m_b, "mu": mu_map[host], "M_B": abs_mag, "err": err_abs})
    calib = pd.DataFrame(rows).sort_values("host").reset_index(drop=True)
    mb, mb_err = weighted_mean(calib["M_B"].tolist(), calib["err"].tolist())
    return calib, mb, mb_err


def fit_h0_from_hubble_sne(ds: dict, mb: float, mb_err: float) -> tuple[pd.DataFrame, float, float]:
    rows = []
    h0_values = []
    h0_errs = []
    for z, m_b, err_m, pv in ds["hubble_flow_sneia"]:
        mu_obs = m_b - mb
        mu_err = math.sqrt(err_m**2 + mb_err**2 + hubble_mu_error(z, 0.0, pv, ds["c_km"]) ** 2)
        d_mpc = dm_to_distance_mpc(mu_obs)
        h0_i = ds["c_km"] * z / d_mpc
        h0_err_i = h0_i * (math.log(10.0) / 5.0) * mu_err
        h0_values.append(h0_i)
        h0_errs.append(h0_err_i)
        rows.append(
            {
                "indicator": "SN Ia",
                "z": z,
                "m_obs": m_b,
                "mu_obs": mu_obs,
                "mu_err": mu_err,
                "h0_i": h0_i,
                "h0_err_i": h0_err_i,
            }
        )
    h0, h0_err = weighted_mean(h0_values, h0_errs)
    return pd.DataFrame(rows), h0, h0_err


def estimate_sbf_absolute_magnitude(ds: dict, h0_ref: float) -> tuple[pd.DataFrame, float, float]:
    group_mu = {}
    for host, mag, err_mag in ds["sbf_calibrators"]:
        group = ds["host_group"][host]
        z_guess = 0.0045 if group == "Fornax" else 0.0036
        mu_group = redshift_distance_modulus(z_guess, h0_ref, ds["c_km"])
        total_err = math.sqrt(err_mag**2 + ds["depth_scatter"] ** 2 + 0.15**2)
        group_mu.setdefault(group, []).append((host, mag, total_err, mu_group))

    rows = []
    mags = []
    errs = []
    for group, items in group_mu.items():
        for host, mag, total_err, mu_group in items:
            abs_mag = mag - mu_group
            rows.append(
                {
                    "host": host,
                    "group": group,
                    "m_sbf": mag,
                    "mu_group": mu_group,
                    "M_sbf": abs_mag,
                    "err": total_err,
                }
            )
            mags.append(abs_mag)
            errs.append(total_err)
    calib = pd.DataFrame(rows).sort_values(["group", "host"]).reset_index(drop=True)
    msbf, msbf_err = weighted_mean(mags, errs)
    return calib, msbf, msbf_err


def fit_h0_from_hubble_sbf(ds: dict, msbf: float, msbf_err: float) -> tuple[pd.DataFrame, float, float]:
    rows = []
    values = []
    errs = []
    for z, mag, err_mag, pv in ds["hubble_flow_sbf"]:
        mu_obs = mag - msbf
        mu_err = math.sqrt(err_mag**2 + msbf_err**2 + hubble_mu_error(z, 0.0, pv, ds["c_km"]) ** 2)
        d_mpc = dm_to_distance_mpc(mu_obs)
        h0_i = ds["c_km"] * z / d_mpc
        h0_err_i = h0_i * (math.log(10.0) / 5.0) * mu_err
        values.append(h0_i)
        errs.append(h0_err_i)
        rows.append(
            {
                "indicator": "SBF",
                "z": z,
                "m_obs": mag,
                "mu_obs": mu_obs,
                "mu_err": mu_err,
                "h0_i": h0_i,
                "h0_err_i": h0_err_i,
            }
        )
    h0, h0_err = weighted_mean(values, errs)
    return pd.DataFrame(rows), h0, h0_err


def combined_h0(indicator_estimates: list[tuple[str, float, float]]) -> tuple[float, float]:
    vals = [v for _, v, _ in indicator_estimates]
    errs = [e for _, _, e in indicator_estimates]
    return weighted_mean(vals, errs)


def residual_objective(h0: float, rows: pd.DataFrame, c_km: float) -> float:
    model = np.array([redshift_distance_modulus(z, h0, c_km) for z in rows["z"]])
    return float(np.sum(np.square((rows["mu_obs"] - model) / rows["mu_err"])))


def fit_direct_h0(all_rows: pd.DataFrame, c_km: float, bounds: tuple[float, float] = (50.0, 150.0)) -> tuple[float, float]:
    result = minimize_scalar(
        lambda h: residual_objective(h, all_rows, c_km),
        bounds=bounds,
        method="bounded",
    )
    h0 = float(result.x)
    chi2_min = residual_objective(h0, all_rows, c_km)
    grid = np.linspace(max(bounds[0], h0 - 20.0), min(bounds[1], h0 + 20.0), 5000)
    chi2 = np.array([residual_objective(h, all_rows, c_km) for h in grid])
    mask = chi2 <= chi2_min + 1.0
    err = float(max(h0 - grid[mask][0], grid[mask][-1] - h0))
    return h0, err


def run_variant(
    name: str,
    ds: dict,
    host_df: pd.DataFrame,
    methods: list[str] | None = None,
    anchors: list[str] | None = None,
    include_sbf: bool = True,
) -> dict:
    host_mu = aggregate_host_distances(host_df, methods, anchors)
    sne_cal, mb, mb_err = calibrate_sne_absolute_magnitude(host_mu, ds)
    flow_sn, h0_sn, h0_sn_err = fit_h0_from_hubble_sne(ds, mb, mb_err)

    estimates = [("SN Ia", h0_sn, h0_sn_err)]
    direct_rows = [flow_sn]
    sbf_cal = None
    flow_sbf = None
    msbf = None
    msbf_err = None
    h0_sbf = None
    h0_sbf_err = None
    if include_sbf:
        sbf_cal, msbf, msbf_err = estimate_sbf_absolute_magnitude(ds, h0_sn)
        flow_sbf, h0_sbf, h0_sbf_err = fit_h0_from_hubble_sbf(ds, msbf, msbf_err)
        estimates.append(("SBF", h0_sbf, h0_sbf_err))
        direct_rows.append(flow_sbf)

    h0_weighted, h0_weighted_err = combined_h0(estimates)
    combined_rows = pd.concat(direct_rows, ignore_index=True)
    h0_direct, h0_direct_err = fit_direct_h0(combined_rows, ds["c_km"])
    h0_sn_direct, h0_sn_direct_err = fit_direct_h0(flow_sn, ds["c_km"])
    return {
        "name": name,
        "host_mu": host_mu,
        "sne_cal": sne_cal,
        "flow_sn": flow_sn,
        "sbf_cal": sbf_cal,
        "flow_sbf": flow_sbf,
        "M_B": mb,
        "M_B_err": mb_err,
        "M_sbf": msbf,
        "M_sbf_err": msbf_err,
        "H0_SN": h0_sn,
        "H0_SN_err": h0_sn_err,
        "H0_SBF": h0_sbf,
        "H0_SBF_err": h0_sbf_err,
        "H0_weighted": h0_weighted,
        "H0_weighted_err": h0_weighted_err,
        "H0_direct": h0_direct,
        "H0_direct_err": h0_direct_err,
        "H0_SN_direct": h0_sn_direct,
        "H0_SN_direct_err": h0_sn_direct_err,
        "n_hosts": int(len(host_mu)),
        "n_sne_cal": int(len(sne_cal)),
    }


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def make_figures(ds: dict, baseline: dict, variants_df: pd.DataFrame, host_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    method_counts = Counter(host_df["method"])
    plt.figure(figsize=(6, 4))
    plt.bar(method_counts.keys(), method_counts.values(), color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Number of host measurements")
    plt.title("Primary-indicator coverage in the minimal dataset")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dataset_overview.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    flow = baseline["flow_sn"]
    ax.errorbar(flow["z"], flow["mu_obs"], yerr=flow["mu_err"], fmt="o", label="SN Ia flow", color="#1f77b4")
    z_grid = np.linspace(0.02, 0.09, 200)
    mu_grid = [redshift_distance_modulus(z, baseline["H0_direct"], ds["c_km"]) for z in z_grid]
    ax.plot(z_grid, mu_grid, color="black", label=f"Best fit H0={baseline['H0_direct']:.2f}")
    if baseline["flow_sbf"] is not None:
        flow_sbf = baseline["flow_sbf"]
        ax.errorbar(flow_sbf["z"], flow_sbf["mu_obs"], yerr=flow_sbf["mu_err"], fmt="s", label="SBF flow", color="#d62728")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance modulus")
    ax.set_title("Hubble-flow relation from the local distance network")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hubble_flow_fit.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = variants_df["variant"]
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        variants_df["H0_SN_direct"],
        yerr=variants_df["H0_SN_direct_err"],
        fmt="o",
        color="#2ca02c",
        ecolor="#2ca02c",
        capsize=3,
    )
    ax.axhline(73.50, color="black", linestyle="--", linewidth=1.2, label="Task baseline 73.50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Analysis variants and local robustness checks")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "variant_comparison.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    labels = ["This work", "Planck-like early universe"]
    vals = [baseline["H0_SN_direct"], 67.4]
    errs = [baseline["H0_SN_direct_err"], 0.5]
    colors = ["#1f77b4", "#9467bd"]
    ax.bar(labels, vals, yerr=errs, color=colors, capsize=5)
    ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Late- vs early-universe comparison")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cmb_comparison.png", dpi=200)
    plt.close()


def build_summary(ds: dict, host_df: pd.DataFrame, baseline: dict, variants_df: pd.DataFrame) -> dict:
    delta = baseline["H0_SN_direct"] - 67.4
    sigma = delta / math.sqrt(baseline["H0_SN_direct_err"] ** 2 + 0.5**2)
    return {
        "dataset": {
            "n_anchor_entries": len(ds["anchors"]),
            "n_host_measurements": int(len(ds["host_measurements"])),
            "n_unique_hosts": int(host_df["host"].nunique()),
            "methods": sorted(host_df["method"].unique().tolist()),
            "anchors": sorted(host_df["anchor"].unique().tolist()),
            "n_sneia_calibrators": int(len(ds["sneia_calibrators"])),
            "n_hubble_flow_sneia": int(len(ds["hubble_flow_sneia"])),
            "n_sbf_calibrators": int(len(ds["sbf_calibrators"])),
            "n_hubble_flow_sbf": int(len(ds["hubble_flow_sbf"])),
        },
        "baseline": {
            "H0_direct": baseline["H0_direct"],
            "H0_direct_err": baseline["H0_direct_err"],
            "H0_SN_direct": baseline["H0_SN_direct"],
            "H0_SN_direct_err": baseline["H0_SN_direct_err"],
            "H0_weighted": baseline["H0_weighted"],
            "H0_weighted_err": baseline["H0_weighted_err"],
            "H0_SN": baseline["H0_SN"],
            "H0_SN_err": baseline["H0_SN_err"],
            "H0_SBF": baseline["H0_SBF"],
            "H0_SBF_err": baseline["H0_SBF_err"],
            "M_B": baseline["M_B"],
            "M_B_err": baseline["M_B_err"],
            "M_sbf": baseline["M_sbf"],
            "M_sbf_err": baseline["M_sbf_err"],
            "planck_reference": 67.4,
            "planck_err": 0.5,
            "tension_sigma_vs_67p4": sigma,
        },
        "variants": variants_df.replace({np.nan: None}).to_dict(orient="records"),
    }


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATA_PATH)
    host_df = build_host_dataframe(ds)
    baseline = run_variant("baseline", ds, host_df, include_sbf=True)

    variant_specs = [
        ("baseline", None, None, True),
        ("cepheid_only", ["Cepheid"], None, True),
        ("trgb_only", ["TRGB"], None, True),
        ("n4258_only", None, ["N4258"], True),
        ("lmc_only", None, ["LMC"], False),
        ("no_sbf", None, None, False),
    ]
    variants = []
    for name, methods, anchors, include_sbf in variant_specs:
        result = run_variant(name, ds, host_df, methods=methods, anchors=anchors, include_sbf=include_sbf)
        variants.append(
            {
                "variant": name,
                "n_hosts": result["n_hosts"],
                "n_sne_cal": result["n_sne_cal"],
                "H0_SN": result["H0_SN"],
                "H0_SN_err": result["H0_SN_err"],
                "H0_SBF": result["H0_SBF"],
                "H0_SBF_err": result["H0_SBF_err"],
                "H0_weighted": result["H0_weighted"],
                "H0_weighted_err": result["H0_weighted_err"],
                "H0_direct": result["H0_direct"],
                "H0_direct_err": result["H0_direct_err"],
                "H0_SN_direct": result["H0_SN_direct"],
                "H0_SN_direct_err": result["H0_SN_direct_err"],
            }
        )
    variants_df = pd.DataFrame(variants)

    save_table(host_df, OUTPUTS / "host_measurements_expanded.csv")
    save_table(baseline["host_mu"], OUTPUTS / "baseline_host_distances.csv")
    save_table(baseline["sne_cal"], OUTPUTS / "sneia_calibrators.csv")
    save_table(baseline["flow_sn"], OUTPUTS / "hubble_flow_sneia.csv")
    if baseline["sbf_cal"] is not None:
        save_table(baseline["sbf_cal"], OUTPUTS / "sbf_calibrators.csv")
    if baseline["flow_sbf"] is not None:
        save_table(baseline["flow_sbf"], OUTPUTS / "hubble_flow_sbf.csv")
    save_table(variants_df, OUTPUTS / "variant_summary.csv")

    make_figures(ds, baseline, variants_df, host_df)
    summary = build_summary(ds, host_df, baseline, variants_df)
    (OUTPUTS / "summary_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
