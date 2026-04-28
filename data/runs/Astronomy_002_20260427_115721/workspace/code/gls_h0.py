"""
Local Distance Network — Generalized Least Squares (GLS) solver for H0.

Linear model:
    y = A @ theta + e,    e ~ N(0, C)

Parameters (theta):
    mu_h[h]   — distance modulus of each primary host h
    mu_g[g]   — distance modulus of each SBF group g
    M_B       — SN Ia absolute B-mag (peak)
    M_SBF     — SBF F110W absolute mag
    a_H       == 5 * log10(H0)              (so H0 = 10**(a_H/5) km/s/Mpc)

Observation blocks (rows of A,y) with their covariance contributions:

(1) Primary host measurements via Cepheid/TRGB anchored on N4258/LMC/MW:
        y_i = mu_meas + mu_anchor    (we add anchor mu so left side is on absolute scale)
        prediction: mu_h
        covariance:
            diag: err_meas^2
            shared per-anchor: err_anchor^2 (block correlation across rows sharing the same anchor)
            shared per (method,anchor): method_anchor_err^2
        For MW the anchor mu is 0 with 0 error, but the (method, anchor) systematic still applies.

(2) SN Ia calibrators (one per host):
        y = mB
        prediction: mu_h + M_B
        covariance: diag err_mB^2

(3) SBF calibrators (one per group host):
        y = mF110W
        prediction: mu_g + M_SBF      (host -> group mapping; group depth scatter handled by adding extra row noise)
        covariance: diag (err_mF110W^2 + depth_scatter^2)

(4) Hubble-flow SNe Ia:
        y = mB - 5*log10(c*z) - 25
        prediction: M_B - a_H
        covariance: diag (err_mB^2 + (5/ln10 * sigma_v/(c*z))^2)

(5) Hubble-flow SBF:
        y = mF110W - 5*log10(c*z) - 25
        prediction: M_SBF - a_H
        covariance: diag (err_mF110W^2 + (5/ln10 * sigma_v/(c*z))^2)

GLS solution:
        theta_hat = (A^T C^{-1} A)^{-1} A^T C^{-1} y
        Cov(theta) = (A^T C^{-1} A)^{-1}

H0 = 10**(a_H/5);   sigma_H0 = H0 * (ln10/5) * sigma_aH.
"""

from __future__ import annotations
import json
import math
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------
# Hard-coded copy of the minimal dataset (also stored in data/)
# ------------------------------------------------------------------
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC':   {'mu': 18.477, 'err': 0.024},
    'MW':    {'mu': 0.0,    'err': 0.0},
}

host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101',    'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB',    'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB',    'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB',    'N4258', 30.53, 0.09),
    ('M101',    'TRGB',    'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC',   32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC',   31.34, 0.09),
]

sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101',     9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06),
]

sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12),
]

hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250.0),
    (0.042, 15.68, 0.05, 250.0),
    (0.055, 16.35, 0.05, 250.0),
    (0.068, 17.02, 0.05, 250.0),
    (0.082, 17.55, 0.06, 250.0),
]

hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250.0),
    (0.031, 31.02, 0.15, 250.0),
    (0.045, 31.89, 0.16, 250.0),
]

method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'):   0.03,
    ('Cepheid', 'MW'):    0.02,
    ('TRGB',    'N4258'): 0.05,
}

host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo',
}

depth_scatter = 0.10
c_km = 299792.458


# ------------------------------------------------------------------
# Build / solve the GLS system
# ------------------------------------------------------------------
def build_system(use_methods=("Cepheid", "TRGB"),
                 use_anchors=("N4258", "LMC", "MW"),
                 use_secondaries=("SNeIa", "SBF"),
                 drop_host=None):
    """Return (A, y, C, theta_names, row_meta).

    use_methods   : tuple of primary indicators to use
    use_anchors   : tuple of geometric anchors to use
    use_secondaries: tuple of secondary indicators to use ("SNeIa", "SBF")
    drop_host     : if not None, exclude rows referring to this primary host (jackknife)
    """
    # -- filter primary host measurements --
    hm = [r for r in host_measurements
          if (r[1] in use_methods) and (r[2] in use_anchors)
          and (drop_host is None or r[0] != drop_host)]

    # -- filter secondary calibrators --
    use_sneia = "SNeIa" in use_secondaries
    use_sbf   = "SBF"   in use_secondaries

    # primary hosts present after filtering
    primary_hosts = sorted({r[0] for r in hm})

    # If SN Ia is not used, primary host μ-parameters become disconnected
    # (no equation links them to H0 in this minimal dataset).
    if not use_sneia:
        hm = []
        primary_hosts = []

    sn_cals = [r for r in sneia_calibrators if r[0] in primary_hosts] if use_sneia else []
    sbf_cals = sbf_calibrators if use_sbf else []
    sbf_groups = sorted({host_group[h] for h, *_ in sbf_cals}) if use_sbf else []

    # -- parameter index --
    theta_names = []
    for h in primary_hosts:
        theta_names.append(f"mu_{h}")
    for g in sbf_groups:
        theta_names.append(f"mu_grp_{g}")
    if use_sneia:
        theta_names.append("M_B")
    if use_sbf:
        theta_names.append("M_SBF")
    theta_names.append("a_H")          # 5*log10(H0)
    n_par = len(theta_names)
    idx = {n: i for i, n in enumerate(theta_names)}

    rows_A, rows_y, row_meta = [], [], []

    # ---- Block 1: primary host measurements ----
    for (h, m, a, mu_meas, err_meas) in hm:
        row = np.zeros(n_par)
        row[idx[f"mu_{h}"]] = 1.0
        # observation = mu_meas + mu_anchor (we shift the data to absolute scale)
        rows_A.append(row)
        rows_y.append(mu_meas)  # already a distance modulus on absolute scale
        row_meta.append(dict(block="primary", host=h, method=m, anchor=a,
                             err_meas=err_meas))

    # ---- Block 2: SNe Ia calibrators ----
    for (h, mB, err_mB) in sn_cals:
        row = np.zeros(n_par)
        row[idx[f"mu_{h}"]] = 1.0
        row[idx["M_B"]]      = 1.0
        rows_A.append(row)
        rows_y.append(mB)
        row_meta.append(dict(block="sneia_cal", host=h, err_meas=err_mB))

    # ---- Block 3: SBF calibrators ----
    for (h, mF, err_mF) in sbf_cals:
        g = host_group[h]
        row = np.zeros(n_par)
        row[idx[f"mu_grp_{g}"]] = 1.0
        row[idx["M_SBF"]]        = 1.0
        rows_A.append(row)
        rows_y.append(mF)
        row_meta.append(dict(block="sbf_cal", host=h, group=g,
                             err_meas=err_mF, depth=depth_scatter))

    # ---- Block 4: Hubble flow SNe Ia ----
    if use_sneia:
        for (z, mB, err_mB, sig_v) in hubble_flow_sneia:
            row = np.zeros(n_par)
            row[idx["M_B"]] = 1.0
            row[idx["a_H"]] = -1.0
            rows_A.append(row)
            rows_y.append(mB - 5.0*math.log10(c_km*z) - 25.0)
            sig_pec = (5.0/math.log(10.0)) * sig_v / (c_km*z)
            row_meta.append(dict(block="hf_sneia", z=z, err_meas=err_mB,
                                 sig_pec=sig_pec))

    # ---- Block 5: Hubble flow SBF ----
    if use_sbf:
        for (z, mF, err_mF, sig_v) in hubble_flow_sbf:
            row = np.zeros(n_par)
            row[idx["M_SBF"]] = 1.0
            row[idx["a_H"]]   = -1.0
            rows_A.append(row)
            rows_y.append(mF - 5.0*math.log10(c_km*z) - 25.0)
            sig_pec = (5.0/math.log(10.0)) * sig_v / (c_km*z)
            row_meta.append(dict(block="hf_sbf", z=z, err_meas=err_mF,
                                 sig_pec=sig_pec))

    n = len(rows_A)
    A = np.vstack(rows_A)
    y = np.array(rows_y)

    # ---- Build covariance matrix C ----
    C = np.zeros((n, n))
    for i, m in enumerate(row_meta):
        if m["block"] == "primary":
            C[i, i] += m["err_meas"]**2
        elif m["block"] in ("sneia_cal", "hf_sneia"):
            C[i, i] += m["err_meas"]**2
            if "sig_pec" in m:
                C[i, i] += m["sig_pec"]**2
        elif m["block"] == "sbf_cal":
            C[i, i] += m["err_meas"]**2 + m["depth"]**2
        elif m["block"] == "hf_sbf":
            C[i, i] += m["err_meas"]**2 + m["sig_pec"]**2

    # Block-correlated terms for primary host measurements:
    #  - anchor error shared across all rows with the same anchor
    #  - method-anchor systematic shared across rows with same (method, anchor)
    primary_idx = [i for i, m in enumerate(row_meta) if m["block"] == "primary"]
    # anchor-shared
    for a in use_anchors:
        sigma_a = anchors[a]['err']
        if sigma_a == 0.0:
            continue
        ix = [i for i in primary_idx if row_meta[i]["anchor"] == a]
        for i in ix:
            for j in ix:
                C[i, j] += sigma_a**2
    # method-anchor shared
    for (mth, a), sig in method_anchor_err.items():
        if mth not in use_methods or a not in use_anchors:
            continue
        ix = [i for i in primary_idx if row_meta[i]["method"] == mth and row_meta[i]["anchor"] == a]
        for i in ix:
            for j in ix:
                C[i, j] += sig**2

    return A, y, C, theta_names, row_meta


def solve_gls(A, y, C):
    Cinv = np.linalg.inv(C)
    AtCi = A.T @ Cinv
    Cov = np.linalg.inv(AtCi @ A)
    theta = Cov @ AtCi @ y
    resid = y - A @ theta
    chi2 = float(resid @ Cinv @ resid)
    dof = A.shape[0] - A.shape[1]
    return theta, Cov, resid, chi2, dof


def report_h0(theta, Cov, names):
    i = names.index("a_H")
    a_H = theta[i]
    s_aH = math.sqrt(Cov[i, i])
    H0 = 10.0**(a_H/5.0)
    sH = H0 * math.log(10.0)/5.0 * s_aH
    return H0, sH, a_H, s_aH


def variant(name, **kwargs):
    A, y, C, names, meta = build_system(**kwargs)
    theta, Cov, resid, chi2, dof = solve_gls(A, y, C)
    H0, sH, aH, saH = report_h0(theta, Cov, names)
    return dict(name=name, kwargs=kwargs, n=len(y), n_par=len(names),
                H0=H0, sigma_H0=sH, a_H=aH, sigma_a_H=saH,
                chi2=chi2, dof=dof, chi2_red=chi2/max(dof, 1),
                theta=theta, Cov=Cov, names=names, resid=resid, meta=meta,
                A=A, y=y, C=C)


# ------------------------------------------------------------------
# Run baseline + variants
# ------------------------------------------------------------------
def main():
    variants = []

    # Baseline: all primaries, all anchors with calibration data, both secondaries
    variants.append(variant("baseline",
                            use_methods=("Cepheid","TRGB"),
                            use_anchors=("N4258","LMC","MW"),
                            use_secondaries=("SNeIa","SBF")))

    # Anchor variants
    variants.append(variant("only_N4258",  use_anchors=("N4258",)))
    variants.append(variant("only_LMC",    use_anchors=("LMC",)))
    variants.append(variant("N4258+LMC",   use_anchors=("N4258","LMC")))

    # Primary-indicator variants
    variants.append(variant("Cepheids_only", use_methods=("Cepheid",)))
    variants.append(variant("TRGB_only",     use_methods=("TRGB",)))

    # Secondary-indicator variants
    variants.append(variant("SNeIa_only", use_secondaries=("SNeIa",)))
    # SBF_only is not solvable in this minimal dataset because SBF group hosts
    # (NGC1399/NGC1404/NGC4472) are not linked to any geometric anchor through a
    # primary indicator measurement, leaving M_SBF, μ_grp and a_H mutually
    # degenerate. We therefore omit SBF_only and report SNeIa_only as the
    # secondary-indicator variant.

    # Jackknife (drop each primary host one at a time)
    primary_hosts = sorted({r[0] for r in host_measurements})
    for h in primary_hosts:
        variants.append(variant(f"drop_{h}", drop_host=h))

    # ----- Save summary -----
    rows = []
    for v in variants:
        rows.append(dict(name=v["name"], n_obs=v["n"], n_par=v["n_par"],
                         H0=v["H0"], sigma_H0=v["sigma_H0"],
                         chi2=v["chi2"], dof=v["dof"], chi2_red=v["chi2_red"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "h0_variants.csv"), index=False)

    base = variants[0]
    with open(os.path.join(OUT, "h0_baseline.json"), "w") as f:
        json.dump({
            "H0":        base["H0"],
            "sigma_H0":  base["sigma_H0"],
            "a_H":       base["a_H"],
            "sigma_a_H": base["sigma_a_H"],
            "chi2":      base["chi2"],
            "dof":       base["dof"],
            "chi2_red":  base["chi2_red"],
            "n_obs":     base["n"],
            "n_par":     base["n_par"],
        }, f, indent=2)

    # parameter table for baseline
    par_rows = []
    for nm, val, var in zip(base["names"], base["theta"], np.diag(base["Cov"])):
        par_rows.append(dict(parameter=nm, value=val, sigma=math.sqrt(var)))
    pd.DataFrame(par_rows).to_csv(os.path.join(OUT, "gls_parameters.csv"), index=False)

    # residuals
    rrows = []
    for i, m in enumerate(base["meta"]):
        Cii = base["C"][i, i]
        rrows.append(dict(
            block=m["block"],
            host=m.get("host", ""),
            method=m.get("method", ""),
            anchor=m.get("anchor", ""),
            z=m.get("z", float("nan")),
            y=base["y"][i],
            pred=(base["A"] @ base["theta"])[i],
            resid=base["resid"][i],
            sigma=math.sqrt(Cii),
            std_resid=base["resid"][i]/math.sqrt(Cii)
        ))
    pd.DataFrame(rrows).to_csv(os.path.join(OUT, "residuals.csv"), index=False)

    # ----- Information weights for H0 (per row) -----
    # Influence of each row on a_H: w_i = ((C^-1 A) (A^T C^-1 A)^-1)[i, a_H_idx]
    Cinv = np.linalg.inv(base["C"])
    AtCi = base["A"].T @ Cinv
    Cov  = np.linalg.inv(AtCi @ base["A"])
    a_idx = base["names"].index("a_H")
    # Each observation's leverage on a_H:
    Linv = Cinv @ base["A"] @ Cov   # (n x p)
    leverage_aH = Linv[:, a_idx]    # row weight on a_H
    # Information contribution per row toward H0 variance reduction (relative)
    info = leverage_aH**2 / np.diag(base["C"])  # higher = more informative
    info_norm = info / info.sum()
    irows = []
    for i, m in enumerate(base["meta"]):
        irows.append(dict(block=m["block"],
                          host=m.get("host", ""),
                          method=m.get("method", ""),
                          anchor=m.get("anchor", ""),
                          z=m.get("z", float("nan")),
                          weight_on_aH=leverage_aH[i],
                          info_share=info_norm[i]))
    pd.DataFrame(irows).to_csv(os.path.join(OUT, "info_weights.csv"), index=False)

    # ----- per-anchor table (for figure) -----
    per_anchor = []
    for vname in ("only_N4258", "only_LMC", "N4258+LMC", "baseline"):
        v = next(x for x in variants if x["name"] == vname)
        per_anchor.append(dict(label=vname, H0=v["H0"], sigma_H0=v["sigma_H0"]))
    pd.DataFrame(per_anchor).to_csv(os.path.join(OUT, "h0_by_anchor.csv"), index=False)

    # per-indicator
    per_ind = []
    for vname in ("Cepheids_only", "TRGB_only", "SNeIa_only", "baseline"):
        v = next(x for x in variants if x["name"] == vname)
        per_ind.append(dict(label=vname, H0=v["H0"], sigma_H0=v["sigma_H0"]))
    pd.DataFrame(per_ind).to_csv(os.path.join(OUT, "h0_by_indicator.csv"), index=False)

    print("Baseline H0 = {:.2f} +- {:.2f} km/s/Mpc  (chi2/dof = {:.2f}/{} = {:.2f})".format(
        base["H0"], base["sigma_H0"], base["chi2"], base["dof"], base["chi2_red"]))
    print("Saved: outputs/h0_baseline.json, h0_variants.csv, gls_parameters.csv, residuals.csv, info_weights.csv")
    return variants


if __name__ == "__main__":
    main()
