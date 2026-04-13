#!/usr/bin/env python3
"""Local offline reproduction analysis for the DESI/ACT EDE benchmark task."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "DESI_EDE_Repro_Data.txt"
OUTPUTS = ROOT / "outputs"
REPORT_IMG = ROOT / "report" / "images"


def load_structured_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    env: dict[str, object] = {}
    exec(text, {}, env)
    return env


def model_table(data: dict) -> pd.DataFrame:
    rows = []
    mapping = {
        "lcdm_params": "LambdaCDM",
        "ede_params": "EDE",
        "w0wa_params": "w0wa",
    }
    for key, model in mapping.items():
        params = data[key]
        for param, (mean, sigma) in params.items():
            rows.append(
                {
                    "model": model,
                    "parameter": param,
                    "mean": float(mean),
                    "sigma": float(sigma),
                }
            )
    return pd.DataFrame(rows)


def points_table(points: list[tuple[float, float, float]], kind: str) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["z", "value", "error"]).assign(dataset=kind)


def compute_parameter_shifts(param_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        param_df[param_df["model"] == "LambdaCDM"][["parameter", "mean", "sigma"]]
        .rename(columns={"mean": "lcdm_mean", "sigma": "lcdm_sigma"})
        .set_index("parameter")
    )
    rows = []
    for model in ["EDE", "w0wa"]:
        subset = param_df[param_df["model"] == model].set_index("parameter")
        common = subset.join(base, how="inner")
        for param, row in common.iterrows():
            delta = row["mean"] - row["lcdm_mean"]
            combined_sigma = math.sqrt(row["sigma"] ** 2 + row["lcdm_sigma"] ** 2)
            rows.append(
                {
                    "model": model,
                    "parameter": param,
                    "delta_vs_lcdm": delta,
                    "shift_in_sigma": delta / combined_sigma if combined_sigma else np.nan,
                    "model_mean": row["mean"],
                    "lcdm_mean": row["lcdm_mean"],
                }
            )
    return pd.DataFrame(rows).sort_values(["parameter", "model"]).reset_index(drop=True)


def derive_summary_metrics(data: dict, param_df: pd.DataFrame) -> dict:
    params = {
        model: {
            row.parameter: {"mean": row.mean, "sigma": row.sigma}
            for row in group.itertuples(index=False)
        }
        for model, group in param_df.groupby("model")
    }

    h0_l = params["LambdaCDM"]["H0"]["mean"]
    h0_e = params["EDE"]["H0"]["mean"]
    h0_w = params["w0wa"]["H0"]["mean"]
    om_l = params["LambdaCDM"]["omega_m"]["mean"]
    om_e = params["EDE"]["omega_m"]["mean"]
    om_w = params["w0wa"]["omega_m"]["mean"]
    s8_l = params["LambdaCDM"]["sigma8"]["mean"]
    s8_e = params["EDE"]["sigma8"]["mean"]
    s8_w = params["w0wa"]["sigma8"]["mean"]

    desi = points_table(data["desi_dvrd_points"], "DESI_DV_over_rd")
    fap = points_table(data["desi_fap_points"], "DESI_FAP")
    sne = points_table(data["sne_mu_points"], "Union3_mu")

    # Offline proxy metric: the BAO digitization crosses zero at high z, so we summarize
    # low-z negative offsets and the pivot redshift of the sign change.
    sign_change_idx = np.where(np.sign(desi["value"]).diff().fillna(0) != 0)[0]
    pivot_z = float(desi.iloc[sign_change_idx[0]]["z"]) if len(sign_change_idx) else np.nan

    metrics = {
        "H0_shift_EDE_vs_LambdaCDM": h0_e - h0_l,
        "H0_shift_w0wa_vs_LambdaCDM": h0_w - h0_l,
        "omega_m_shift_EDE_vs_LambdaCDM": om_e - om_l,
        "omega_m_shift_w0wa_vs_LambdaCDM": om_w - om_l,
        "sigma8_shift_EDE_vs_LambdaCDM": s8_e - s8_l,
        "sigma8_shift_w0wa_vs_LambdaCDM": s8_w - s8_l,
        "fractional_H0_shift_EDE_percent": 100.0 * (h0_e / h0_l - 1.0),
        "fractional_H0_shift_w0wa_percent": 100.0 * (h0_w / h0_l - 1.0),
        "EDE_f_EDE_mean": params["EDE"]["f_EDE"]["mean"],
        "EDE_log10_ac_mean": params["EDE"]["log10_ac"]["mean"],
        "DESI_DVrd_weighted_mean_offset": float(
            np.average(desi["value"], weights=1.0 / desi["error"] ** 2)
        ),
        "DESI_FAP_weighted_mean_offset": float(
            np.average(fap["value"], weights=1.0 / fap["error"] ** 2)
        ),
        "Union3_weighted_mean_offset_mag": float(
            np.average(sne["value"], weights=1.0 / sne["error"] ** 2)
        ),
        "DESI_DVrd_sign_change_pivot_z": pivot_z,
        "DESI_DVrd_absmax_offset_sigma": float(np.max(np.abs(desi["value"] / desi["error"]))),
    }
    return metrics


def save_tables(param_df: pd.DataFrame, shift_df: pd.DataFrame, data: dict, metrics: dict) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    param_df.to_csv(OUTPUTS / "parameter_constraints.csv", index=False)
    shift_df.to_csv(OUTPUTS / "parameter_shifts.csv", index=False)

    combined_points = pd.concat(
        [
            points_table(data["desi_dvrd_points"], "desi_dvrd"),
            points_table(data["desi_fap_points"], "desi_fap"),
            points_table(data["sne_mu_points"], "union3_mu"),
        ],
        ignore_index=True,
    )
    combined_points.to_csv(OUTPUTS / "distance_points.csv", index=False)
    (OUTPUTS / "summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def plot_parameter_constraints(param_df: pd.DataFrame) -> None:
    focus = ["omega_m", "H0", "sigma8", "ns", "ombh2", "ln10As", "tau"]
    plot_df = param_df[param_df["parameter"].isin(focus)].copy()
    order = focus[::-1]

    sns.set_theme(style="whitegrid", context="talk")
    palette = {"LambdaCDM": "#1f3b73", "EDE": "#c0392b", "w0wa": "#1e8449"}

    fig, ax = plt.subplots(figsize=(11, 7))
    y_base = np.arange(len(order))
    offsets = {"LambdaCDM": -0.22, "EDE": 0.0, "w0wa": 0.22}
    for model in ["LambdaCDM", "EDE", "w0wa"]:
        subset = plot_df[plot_df["model"] == model].set_index("parameter").loc[order].reset_index()
        ypos = y_base + offsets[model]
        ax.errorbar(
            subset["mean"],
            ypos,
            xerr=subset["sigma"],
            fmt="o",
            capsize=4,
            lw=2,
            ms=7,
            color=palette[model],
            label=model,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(order)
    ax.set_xlabel("Posterior mean with 1σ interval")
    ax.set_title("Parameter constraints from the local reproduction dataset")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "parameter_constraints.png", dpi=200)
    plt.close(fig)


def plot_standardized_shifts(shift_df: pd.DataFrame) -> None:
    focus = ["omega_m", "H0", "sigma8", "ns", "ombh2", "ln10As", "tau"]
    plot_df = shift_df[shift_df["parameter"].isin(focus)].copy()
    order = focus
    palette = {"EDE": "#c0392b", "w0wa": "#1e8449"}

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.38
    x = np.arange(len(order))
    for idx, model in enumerate(["EDE", "w0wa"]):
        subset = plot_df[plot_df["model"] == model].set_index("parameter").loc[order].reset_index()
        ax.bar(
            x + (idx - 0.5) * width,
            subset["shift_in_sigma"],
            width=width,
            color=palette[model],
            label=model,
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Shift relative to LambdaCDM [combined σ]")
    ax.set_title("Distinct parameter response of EDE and w0wa")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "parameter_shift_sigma.png", dpi=200)
    plt.close(fig)


def plot_distance_proxies(data: dict) -> None:
    desi = points_table(data["desi_dvrd_points"], "DESI Δ(DV/rd)")
    fap = points_table(data["desi_fap_points"], "DESI ΔF_AP")
    sne = points_table(data["sne_mu_points"], "Union3 Δμ")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    series = [
        (desi, "DESI BAO radial-volume proxy", "Δ(DV/rd)"),
        (fap, "DESI BAO Alcock-Paczynski proxy", "ΔF_AP"),
        (sne, "Union3 supernova proxy", "Δμ [mag]"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (df, title, ylabel), color in zip(axes, series, colors):
        ax.errorbar(df["z"], df["value"], yerr=df["error"], fmt="o-", capsize=3, color=color)
        ax.axhline(0, color="black", lw=1, alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Redshift z")
        ax.set_ylabel(ylabel)
    fig.suptitle("Digitized distance residuals from the reproduction file", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "distance_residuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ede_posterior_proxy(data: dict) -> None:
    ede = data["ede_params"]
    mean_f, sig_f = ede["f_EDE"]
    mean_a, sig_a = ede["log10_ac"]

    cov = np.array([[sig_f ** 2, 0.15 * sig_f * sig_a], [0.15 * sig_f * sig_a, sig_a ** 2]])
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal([mean_f, mean_a], cov, size=3000)
    df = pd.DataFrame(samples, columns=["f_EDE", "log10_ac"])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.kdeplot(
        data=df,
        x="f_EDE",
        y="log10_ac",
        fill=True,
        levels=6,
        thresh=0.05,
        cmap="Reds",
        ax=ax,
    )
    ax.scatter([mean_f], [mean_a], color="black", s=30, label="Posterior mean")
    ax.set_title("Approximate local posterior proxy for EDE parameters")
    ax.set_xlabel("f_EDE")
    ax.set_ylabel("log10(a_c)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG / "ede_posterior_proxy.png", dpi=200)
    plt.close(fig)


def write_summary_note(metrics: dict, shift_df: pd.DataFrame) -> None:
    top = (
        shift_df.assign(abs_shift=shift_df["shift_in_sigma"].abs())
        .sort_values("abs_shift", ascending=False)
        .head(8)
    )
    lines = [
        "Local reproduction summary",
        "",
        "This note is generated from the structured benchmark input only.",
        "",
        "Key metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}")
    lines.extend(["", "Largest standardized shifts vs LambdaCDM:"])
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.model} {row.parameter}: delta={row.delta_vs_lcdm:.4g}, shift={row.shift_in_sigma:.3f} sigma"
        )
    (OUTPUTS / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_IMG.mkdir(parents=True, exist_ok=True)

    data = load_structured_data(DATA_FILE)
    param_df = model_table(data)
    shift_df = compute_parameter_shifts(param_df)
    metrics = derive_summary_metrics(data, param_df)

    save_tables(param_df, shift_df, data, metrics)
    plot_parameter_constraints(param_df)
    plot_standardized_shifts(shift_df)
    plot_distance_proxies(data)
    plot_ede_posterior_proxy(data)
    write_summary_note(metrics, shift_df)


if __name__ == "__main__":
    main()
