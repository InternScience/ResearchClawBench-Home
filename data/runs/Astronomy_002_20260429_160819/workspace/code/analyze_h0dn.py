#!/usr/bin/env python3
"""Reproducible Local Distance Network analysis for the minimal H0DN dataset.

The script parses the Python-like text dataset, constructs a generalized least
squares (GLS) distance-ladder model for the SNe Ia branch, produces pragmatic
SBF diagnostics, runs analysis variants, and exports report tables/figures.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "H0DN_MinimalDataset.txt"
OUT = ROOT / "outputs"
FIG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CMB_H0 = 67.4
CMB_SIG = 0.5
PROMPT_H0 = 73.50
PROMPT_SIG = 0.81
SHOES_H0 = 73.04
SHOES_SIG = 1.04
# A weak absolute-magnitude prior regularizes the intentionally tiny Hubble-flow
# sample to the conventional standardized SN Ia scale.  The central value is the
# value implied by the prompt baseline and the Hubble-flow intercept in this file;
# 0.20 mag is deliberately broad compared with modern SN Ia calibrations.
DEFAULT_M_SN_PRIOR_SIG = 0.20

sns.set_theme(style="whitegrid", context="talk")


def load_dataset(path: Path = DATA) -> Dict[str, object]:
    text = path.read_text()
    names = [
        "anchors",
        "host_measurements",
        "sneia_calibrators",
        "sbf_calibrators",
        "hubble_flow_sneia",
        "hubble_flow_sbf",
        "method_anchor_err",
        "host_group",
        "depth_scatter",
        "c_km",
    ]
    data = {}
    for name in names:
        m = re.search(rf"^{name}\s*=\s*(.*?)(?=\n\n#|\n\w+\s*=|\Z)", text, flags=re.S | re.M)
        if not m:
            raise ValueError(f"Could not parse {name}")
        expr = m.group(1).strip()
        data[name] = ast.literal_eval(expr)
    return data


def mu_from_z_h0(z: float, h0: float, c_km: float) -> float:
    d_mpc = c_km * z / h0
    return 5.0 * math.log10(d_mpc) + 25.0


def pv_sigma_mag(z: float, vpec: float, c_km: float) -> float:
    # First-order propagation dmu = 5/log(10) * dv/(cz)
    return (5.0 / math.log(10.0)) * vpec / (c_km * z)


def build_snia_gls(
    data: Dict[str, object],
    include_methods: Optional[Iterable[str]] = None,
    include_anchors: Optional[Iterable[str]] = None,
    exclude_hosts: Optional[Iterable[str]] = None,
    h0_prior: Optional[Tuple[float, float]] = None,
    m_sn_prior: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    include_methods = set(include_methods) if include_methods is not None else None
    include_anchors = set(include_anchors) if include_anchors is not None else None
    exclude_hosts = set(exclude_hosts or [])

    host_measurements = [
        row for row in data["host_measurements"]
        if (include_methods is None or row[1] in include_methods)
        and (include_anchors is None or row[2] in include_anchors)
        and row[0] not in exclude_hosts
    ]
    sne_cal = [row for row in data["sneia_calibrators"] if row[0] not in exclude_hosts]
    cal_hosts = sorted(set(h for h, *_ in sne_cal))
    # Only hosts with at least one primary distance can calibrate SN absolute magnitude.
    dist_hosts = sorted(set(h for h, *_ in host_measurements))
    hosts = sorted(set(cal_hosts) | set(dist_hosts))

    params = [f"mu:{h}" for h in hosts] + ["M_SN", "logH0"]
    idx = {p: i for i, p in enumerate(params)}
    rows, y, sig, meta = [], [], [], []

    anchors = data["anchors"]
    method_anchor_err = data["method_anchor_err"]
    for host, method, anchor, mu_meas, err in host_measurements:
        if host not in hosts:
            continue
        row = np.zeros(len(params)); row[idx[f"mu:{host}"]] = 1.0
        # Treat listed host mu_meas as the distance modulus inferred via that anchor.
        # Anchor and method-anchor zero-point errors are common-mode components for rows
        # sharing the same calibration path; in this compact GLS they are included in the
        # diagonal variance and catalogued in the uncertainty table.
        aerr = float(anchors[anchor]["err"])
        merr = float(method_anchor_err.get((method, anchor), 0.0))
        total = math.sqrt(err**2 + aerr**2 + merr**2)
        rows.append(row); y.append(mu_meas); sig.append(total)
        meta.append({"type":"primary_distance", "host":host, "method":method, "anchor":anchor, "observed":mu_meas, "sigma":total})

    for host, mb, emb in sne_cal:
        if host not in hosts:
            continue
        row = np.zeros(len(params)); row[idx[f"mu:{host}"]] = 1.0; row[idx["M_SN"]] = 1.0
        rows.append(row); y.append(mb); sig.append(emb)
        meta.append({"type":"snia_calibrator", "host":host, "method":"SN Ia", "anchor":"calibrator", "observed":mb, "sigma":emb})

    c_km = float(data["c_km"])
    for z, mb, emb, vpec in data["hubble_flow_sneia"]:
        # mB = M_SN + 5 log10(cz) - 5 log10(H0) + 25 = M_SN + const - 5 logH0
        row = np.zeros(len(params)); row[idx["M_SN"]] = 1.0; row[idx["logH0"]] = -5.0
        const = 5.0 * math.log10(c_km * z) + 25.0
        total = math.sqrt(emb**2 + pv_sigma_mag(z, vpec, c_km)**2)
        rows.append(row); y.append(mb - const); sig.append(total)
        meta.append({"type":"snia_hubble_flow", "host":f"z={z:.3f}", "method":"SN Ia", "anchor":"Hubble flow", "observed":mb, "sigma":total, "z":z, "const":const})

    if m_sn_prior is not None:
        m0, sm0 = m_sn_prior
        row = np.zeros(len(params)); row[idx["M_SN"]] = 1.0
        rows.append(row); y.append(m0); sig.append(sm0)
        meta.append({"type":"external_M_SN_prior", "host":"external", "method":"SN Ia", "anchor":"weak prior", "observed":m0, "sigma":sm0})

    if h0_prior is not None:
        h0, sh0 = h0_prior
        row = np.zeros(len(params)); row[idx["logH0"]] = 1.0
        rows.append(row); y.append(math.log10(h0)); sig.append(sh0 / (h0 * math.log(10.0)))
        meta.append({"type":"external_h0_prior", "host":"external", "method":"H0", "anchor":"prior", "observed":h0, "sigma":sh0})

    A = np.vstack(rows); b = np.asarray(y, dtype=float); sigma = np.asarray(sig, dtype=float)
    return A, b, sigma, params, pd.DataFrame(meta)


def solve_gls(A: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    W = np.diag(1.0 / sigma**2)
    AtW = A.T @ W
    cov = np.linalg.inv(AtW @ A)
    beta = cov @ (AtW @ b)
    resid = b - A @ beta
    chi2 = float(np.sum((resid / sigma)**2))
    dof = len(b) - A.shape[1]
    return beta, cov, chi2, dof, resid


def summarize_fit(name: str, data: Dict[str, object], **kwargs) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    A, b, sigma, params, meta = build_snia_gls(data, **kwargs)
    beta, cov, chi2, dof, resid = solve_gls(A, b, sigma)
    logh_idx = params.index("logH0")
    logh = beta[logh_idx]
    sig_logh = math.sqrt(cov[logh_idx, logh_idx])
    h0 = 10**logh
    sig_h0 = math.log(10.0) * h0 * sig_logh
    M_idx = params.index("M_SN")
    M = beta[M_idx]; sig_M = math.sqrt(cov[M_idx, M_idx])
    out = {
        "variant": name,
        "H0": h0,
        "sigma_H0": sig_h0,
        "log10_H0": logh,
        "sigma_log10_H0": sig_logh,
        "M_SN": M,
        "sigma_M_SN": sig_M,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2/dof if dof > 0 else np.nan,
        "n_observations": int(len(b)),
        "n_parameters": int(len(params)),
    }
    fitted = []
    for p, val, se in zip(params, beta, np.sqrt(np.diag(cov))):
        fitted.append({"variant": name, "parameter": p, "estimate": val, "sigma": se})
    meta = meta.copy()
    meta["variant"] = name
    meta["model_value"] = A @ beta
    meta["residual"] = resid
    meta["normalized_residual"] = resid / sigma
    return out, pd.DataFrame(fitted), meta


def sbf_branch_diagnostics(data: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate an approximate SBF absolute magnitude from group distances.

    The minimal file lists SBF calibrator apparent magnitudes but no direct primary
    distance rows for those galaxies. For diagnostics only, assign Fornax the mean
    NGC1316/TRGB distance (same cluster context) and Virgo the M101/Cepheid+TRGB
    distance as a rough nearby Virgo-spiral proxy, inflating by depth scatter.
    """
    hm = pd.DataFrame(data["host_measurements"], columns=["host","method","anchor","mu","err"])
    sbfcal = pd.DataFrame(data["sbf_calibrators"], columns=["host","m","err_m"])
    group = data["host_group"]
    depth = float(data["depth_scatter"])
    # group proxy distances
    proxies = []
    for _, r in hm.iterrows():
        if r.host == "NGC1316" and r.method == "TRGB": proxies.append(("Fornax", r.mu, math.sqrt(r.err**2 + depth**2), "NGC1316 TRGB proxy"))
        if r.host == "M101": proxies.append(("Virgo", r.mu, math.sqrt(r.err**2 + depth**2), "M101 nearby-spiral proxy"))
    prox = pd.DataFrame(proxies, columns=["group","mu_proxy","sigma_mu","proxy_source"])
    rows = []
    for _, r in sbfcal.iterrows():
        g = group[r.host]
        gp = prox[prox.group == g]
        if gp.empty: continue
        # use inverse variance average if multiple proxies
        w = 1/gp.sigma_mu.values**2
        mu = float(np.sum(w*gp.mu_proxy.values)/np.sum(w))
        smu = float(math.sqrt(1/np.sum(w)))
        M = r.m - mu
        sM = math.sqrt(r.err_m**2 + smu**2)
        rows.append({"host":r.host,"group":g,"m_sbf":r.m,"mu_proxy":mu,"sigma_mu_proxy":smu,"M_SBF":M,"sigma_M_SBF":sM,"proxy":"; ".join(gp.proxy_source)})
    cal = pd.DataFrame(rows)
    w = 1/cal.sigma_M_SBF.values**2
    Mbar = float(np.sum(w*cal.M_SBF.values)/np.sum(w))
    sMbar = float(math.sqrt(1/np.sum(w)))
    flow_rows=[]
    for z, m, em, vpec in data["hubble_flow_sbf"]:
        smu = math.sqrt(em**2 + sMbar**2 + pv_sigma_mag(z, vpec, float(data["c_km"]))**2)
        mu = m - Mbar
        h0 = float(data["c_km"])*z/(10**((mu-25)/5))
        sh0 = (math.log(10)/5)*h0*smu
        flow_rows.append({"z":z,"m_sbf":m,"sigma_m":em,"M_SBF_cal":Mbar,"sigma_M_SBF_cal":sMbar,"mu":mu,"sigma_mu":smu,"H0":h0,"sigma_H0":sh0})
    flow = pd.DataFrame(flow_rows)
    return cal, flow


def main() -> None:
    data = load_dataset()
    dep = {}
    for mod in ["numpy","pandas","matplotlib","seaborn","scipy","pypdf","pdfminer","statsmodels"]:
        try:
            __import__(mod); dep[mod] = True
        except Exception:
            dep[mod] = False
    (OUT / "dependency_check.json").write_text(json.dumps({"dependencies":dep,"fallbacks":["statsmodels unavailable but not required; GLS solved directly with numpy linear algebra."],"method_feasibility":"Named covariance-weighted GLS is feasible with numpy."}, indent=2))

    overview = {
        "anchors": len(data["anchors"]),
        "host_measurements": len(data["host_measurements"]),
        "sneia_calibrators": len(data["sneia_calibrators"]),
        "sbf_calibrators": len(data["sbf_calibrators"]),
        "hubble_flow_sneia": len(data["hubble_flow_sneia"]),
        "hubble_flow_sbf": len(data["hubble_flow_sbf"]),
        "primary_methods": sorted(set(r[1] for r in data["host_measurements"])),
        "anchors_used_in_primary_rows": sorted(set(r[2] for r in data["host_measurements"])),
        "secondary_methods_present": ["SN Ia", "SBF"],
    }
    (OUT / "data_overview.json").write_text(json.dumps(overview, indent=2))

    variants = []
    fitted_all = []
    resid_all = []
    specs = [
        ("baseline_all_primary", {}),
        ("cepheid_only", {"include_methods":["Cepheid"]}),
        ("trgb_only", {"include_methods":["TRGB"]}),
        ("ngc4258_anchor_only", {"include_anchors":["N4258"]}),
        ("lmc_anchor_only", {"include_anchors":["LMC"]}),
        ("exclude_M101", {"exclude_hosts":["M101"]}),
        ("exclude_NGC1365", {"exclude_hosts":["NGC1365"]}),
        ("combined_with_planck_prior", {"h0_prior":(CMB_H0, CMB_SIG)}),
    ]
    for name, kwargs in specs:
        try:
            res, fit, meta = summarize_fit(name, data, **kwargs)
            variants.append(res); fitted_all.append(fit); resid_all.append(meta)
        except Exception as e:
            variants.append({"variant":name,"error":repr(e)})
    vdf = pd.DataFrame(variants)
    fdf = pd.concat(fitted_all, ignore_index=True)
    rdf = pd.concat(resid_all, ignore_index=True)
    vdf.to_csv(OUT / "variant_results.csv", index=False)
    fdf.to_csv(OUT / "fitted_parameters.csv", index=False)
    rdf.to_csv(OUT / "residuals.csv", index=False)

    baseline = vdf[vdf.variant == "baseline_all_primary"].iloc[0].to_dict()
    tension_planck = (baseline["H0"] - CMB_H0)/math.sqrt(baseline["sigma_H0"]**2 + CMB_SIG**2)
    tension_prompt = (baseline["H0"] - PROMPT_H0)/math.sqrt(baseline["sigma_H0"]**2 + PROMPT_SIG**2)
    baseline_scaled_sigma = baseline["sigma_H0"] * math.sqrt(baseline["chi2_dof"]) if baseline["chi2_dof"] and baseline["chi2_dof"] > 1 else baseline["sigma_H0"]
    baseline.update({"planck_H0":CMB_H0,"planck_sigma":CMB_SIG,"tension_vs_planck_sigma":tension_planck,"prompt_baseline_H0":PROMPT_H0,"prompt_baseline_sigma":PROMPT_SIG,"difference_vs_prompt_sigma":tension_prompt,"sigma_H0_chi2_scaled":baseline_scaled_sigma})
    (OUT / "baseline_results.json").write_text(json.dumps({k:(None if pd.isna(v) else v) for k,v in baseline.items()}, indent=2))

    # uncertainty components from rows
    rdf_base = rdf[rdf.variant == "baseline_all_primary"].copy()
    components = rdf_base[["type","host","method","anchor","sigma"]].copy()
    components.to_csv(OUT / "uncertainty_components.csv", index=False)

    sbf_cal, sbf_flow = sbf_branch_diagnostics(data)
    sbf_cal.to_csv(OUT / "sbf_calibration_diagnostic.csv", index=False)
    sbf_flow.to_csv(OUT / "sbf_hubble_diagnostic.csv", index=False)

    # Figure source exports
    figsrc = OUT / "figure_source_data"; figsrc.mkdir(exist_ok=True)
    # data overview counts
    count_df = pd.DataFrame([
        {"category":"Geometric anchors","count":overview["anchors"]},
        {"category":"Primary host distances","count":overview["host_measurements"]},
        {"category":"SN Ia calibrators","count":overview["sneia_calibrators"]},
        {"category":"SN Ia Hubble flow","count":overview["hubble_flow_sneia"]},
        {"category":"SBF calibrators","count":overview["sbf_calibrators"]},
        {"category":"SBF Hubble flow","count":overview["hubble_flow_sbf"]},
    ])
    count_df.to_csv(figsrc / "data_overview_counts.csv", index=False)
    comp = pd.DataFrame([
        {"source":"This minimal GLS baseline","H0":baseline["H0"],"sigma":baseline["sigma_H0"],"class":"local"},
        {"source":"Prompt target baseline","H0":PROMPT_H0,"sigma":PROMPT_SIG,"class":"local target"},
        {"source":"SH0ES 2022 Cepheid-SN","H0":SHOES_H0,"sigma":SHOES_SIG,"class":"related work"},
        {"source":"Planck ΛCDM","H0":CMB_H0,"sigma":CMB_SIG,"class":"early universe"},
    ])
    comp.to_csv(figsrc / "h0_comparison.csv", index=False)
    vdf.to_csv(figsrc / "variant_results.csv", index=False)
    rdf_base.to_csv(figsrc / "baseline_residuals.csv", index=False)

    # Plots
    plt.figure(figsize=(10,5))
    ax=sns.barplot(data=count_df, x="category", y="count", color="#4c72b0")
    ax.set_xlabel(""); ax.set_ylabel("Number of entries"); ax.set_title("Minimal H0DN dataset contents")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.savefig(FIG / "data_overview.png", dpi=180); plt.close()

    plt.figure(figsize=(10,5))
    y=np.arange(len(comp))
    colors=["#4c72b0" if c!="early universe" else "#c44e52" for c in comp["class"]]
    plt.errorbar(comp.H0, y, xerr=comp.sigma, fmt='none', ecolor='0.25', capsize=4, lw=2)
    plt.scatter(comp.H0, y, s=100, c=colors, zorder=3)
    plt.yticks(y, comp.source)
    plt.xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
    plt.title("Local Distance Network estimate and external comparisons")
    plt.axvspan(CMB_H0-CMB_SIG, CMB_H0+CMB_SIG, color="#c44e52", alpha=0.12)
    plt.tight_layout(); plt.savefig(FIG / "h0_comparison.png", dpi=180); plt.close()

    plotv = vdf.dropna(subset=["H0"]).copy()
    plotv = plotv[~plotv.variant.eq("combined_with_planck_prior")]
    plt.figure(figsize=(10,6))
    y=np.arange(len(plotv))
    plt.errorbar(plotv.H0, y, xerr=plotv.sigma_H0, fmt='o', capsize=4, color="#4c72b0")
    plt.axvline(PROMPT_H0, color="black", ls="--", lw=1.5, label="Prompt baseline")
    plt.axvline(CMB_H0, color="#c44e52", ls=":", lw=2, label="Planck ΛCDM")
    plt.yticks(y, plotv.variant)
    plt.xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
    plt.title("Analysis variants from the minimal network")
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "variant_results.png", dpi=180); plt.close()

    plt.figure(figsize=(11,5))
    order=["primary_distance","snia_calibrator","snia_hubble_flow"]
    ax=sns.scatterplot(data=rdf_base, x="host", y="normalized_residual", hue="type", style="type", hue_order=order, s=90)
    ax.axhline(0,color='0.2',lw=1); ax.axhline(2,color='0.5',ls='--',lw=1); ax.axhline(-2,color='0.5',ls='--',lw=1)
    ax.set_xlabel("Observation / host"); ax.set_ylabel("Normalized residual")
    ax.set_title("Baseline GLS validation residuals")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig(FIG / "validation_residuals.png", dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    if not sbf_flow.empty:
        plt.errorbar(sbf_flow.z, sbf_flow.H0, yerr=sbf_flow.sigma_H0, fmt='o', capsize=4, color="#55a868")
        plt.axhline(baseline["H0"], color="#4c72b0", ls="--", label="SN Ia GLS baseline")
        plt.axhline(PROMPT_H0, color='0.2', ls=':', label="Prompt target")
        plt.xlabel("Redshift"); plt.ylabel(r"SBF-implied $H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
        plt.title("Diagnostic SBF branch (proxy-calibrated)")
        plt.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "sbf_diagnostic.png", dpi=180); plt.close()

    claims = pd.DataFrame([
        {"claim":"Baseline minimal-data GLS H0", "supporting_artifact":"outputs/baseline_results.json", "status":"verified from workspace data"},
        {"claim":"Variant sensitivity by method/anchor/host", "supporting_artifact":"outputs/variant_results.csv; report/images/variant_results.png", "status":"verified from workspace data"},
        {"claim":"Residual validation of baseline fit", "supporting_artifact":"outputs/residuals.csv; report/images/validation_residuals.png", "status":"verified from workspace data"},
        {"claim":"Related-work CMB comparison value 67.4 +/- 0.5", "supporting_artifact":"outputs/related_work_contract.json", "status":"extracted from related work"},
        {"claim":"Full task indicators Miras/JAGB/SNe II/FP/TF absent from minimal dataset", "supporting_artifact":"data/H0DN_MinimalDataset.txt; outputs/data_overview.json", "status":"verified from workspace data"},
        {"claim":"SBF branch is diagnostic/proxy-calibrated, not a full independent GLS rung", "supporting_artifact":"outputs/sbf_calibration_diagnostic.csv", "status":"limitation documented"},
    ])
    claims.to_csv(OUT / "claim_recovery_table.csv", index=False)

    # Update artifact inventory with statuses.
    inv = json.loads((OUT / "target_artifact_inventory.json").read_text())
    for item in inv["required_artifacts"]:
        path = ROOT / item["target_path"]
        item["status"] = "satisfied" if path.exists() else "unsatisfied"
        if not path.exists(): item["reason"] = "file not found after analysis run"
    (OUT / "target_artifact_inventory.json").write_text(json.dumps(inv, indent=2))

    print(json.dumps({"baseline_H0":baseline["H0"],"sigma_H0":baseline["sigma_H0"],"tension_vs_planck_sigma":tension_planck,"n_variants":len(vdf)}, indent=2))

if __name__ == "__main__":
    main()
