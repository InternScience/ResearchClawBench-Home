"""Main analysis: three core MATBG superfluid-stiffness experiments.

Outputs:
  outputs/density_summary.csv
  outputs/temperature_fits.json
  outputs/current_fits.json
  report/images/fig1_density_dependence.png
  report/images/fig2_geometric_enhancement.png
  report/images/fig3_temperature_dependence.png
  report/images/fig4_powerlaw_fit.png
  report/images/fig5_current_dc.png
  report/images/fig6_GL_quadratic.png
  report/images/fig7_microwave.png
"""
from __future__ import annotations

import json
import os
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "outputs", "matbg_data.npz")
IMG = os.path.join(ROOT, "report", "images")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.8,
})

DATA = np.load(NPZ)


def get(name: str) -> np.ndarray:
    return DATA[name]


# ---------------------------------------------------------------------------
# 1) Carrier-density dependence
# ---------------------------------------------------------------------------
def analysis_density() -> dict:
    n = get("n_eff in m^-2")
    Ds_conv = get("d_s_conv")
    Ds_geom = get("d_s_geom")
    Ds_h = get("d_s_exp_hole")
    Ds_e = get("d_s_exp_electron")

    # Save tabulated CSV
    csv_path = os.path.join(OUT, "density_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_eff_m^-2", "D_s_conv", "D_s_geom", "D_s_exp_hole", "D_s_exp_electron",
                    "ratio_hole_over_conv", "ratio_geom_over_conv"])
        for i in range(len(n)):
            w.writerow([
                n[i], Ds_conv[i], Ds_geom[i], Ds_h[i], Ds_e[i],
                Ds_h[i] / Ds_conv[i], Ds_geom[i] / Ds_conv[i],
            ])

    # Figure 1 — D_s vs n
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(n / 1e15, Ds_conv / 1e9, "C0--", label=r"Conventional FL ($v_F=700$ m/s)")
    ax.plot(n / 1e15, Ds_geom / 1e9, "C1-.", label=r"Quantum geometric ($v_F=3000$ m/s)")
    ax.plot(n / 1e15, Ds_h / 1e9, "C3o-", ms=4, label="Experimental hole-doped")
    ax.plot(n / 1e15, Ds_e / 1e9, "C2s-", ms=4, label="Experimental electron-doped")
    ax.set_xlabel(r"Carrier density $n_{\rm eff}$  $(10^{15}\;{\rm m}^{-2})$")
    ax.set_ylabel(r"Superfluid stiffness $D_s$  $(10^9\;{\rm s}^{-1})$")
    ax.set_yscale("log")
    ax.set_title("MATBG superfluid stiffness vs carrier density")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(IMG, "fig1_density_dependence.png"))
    plt.close(fig)

    # Figure 2 — geometric enhancement
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(n / 1e15, Ds_geom / Ds_conv, "C1-", label=r"$D_s^{\rm geom}/D_s^{\rm conv}$ (theory)")
    ax.plot(n / 1e15, Ds_h / Ds_conv, "C3o-", ms=4, label=r"$D_s^{\rm exp,h}/D_s^{\rm conv}$")
    ax.plot(n / 1e15, Ds_e / Ds_conv, "C2s-", ms=4, label=r"$D_s^{\rm exp,e}/D_s^{\rm conv}$")
    ax.set_xlabel(r"Carrier density $n_{\rm eff}$  $(10^{15}\;{\rm m}^{-2})$")
    ax.set_ylabel(r"Enhancement ratio $D_s/D_s^{\rm conv}$")
    ax.set_title("Quantum-geometric enhancement of MATBG superfluid stiffness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(IMG, "fig2_geometric_enhancement.png"))
    plt.close(fig)

    summary = {
        "v_f_conventional_m_per_s": 700.0,
        "v_f_geometric_m_per_s": 3000.0,
        "n_min_m^-2": float(n.min()),
        "n_max_m^-2": float(n.max()),
        "ratio_geom_over_conv": {
            "min": float((Ds_geom / Ds_conv).min()),
            "max": float((Ds_geom / Ds_conv).max()),
            "mean": float((Ds_geom / Ds_conv).mean()),
        },
        "ratio_hole_over_conv": {
            "min": float((Ds_h / Ds_conv).min()),
            "max": float((Ds_h / Ds_conv).max()),
            "mean": float((Ds_h / Ds_conv).mean()),
        },
        "ratio_electron_over_conv": {
            "min": float((Ds_e / Ds_conv).min()),
            "max": float((Ds_e / Ds_conv).max()),
            "mean": float((Ds_e / Ds_conv).mean()),
        },
        "csv": csv_path,
    }
    return summary


# ---------------------------------------------------------------------------
# 2) Temperature dependence
# ---------------------------------------------------------------------------
def power_law(T, D0, Tc, n):
    val = D0 * (1.0 - (T / Tc) ** n)
    return np.where(T < Tc, val, 0.0)


def analysis_temperature() -> dict:
    T = get("t in k")  # 100 pts on [0, 1.2]
    Tc_known = 1.0
    D0_known = 100.0

    bcs = get("d_s_bcs")
    nodal = get("d_s_nodal")
    p2 = get("d_s_power_n2")
    p25 = get("d_s_power_n2_5")
    p3 = get("d_s_power_n3")
    exp = get("d_s_experimental")

    # Theory curves are 95 / 95 / 95 / 90 / 90 long; pad with zeros to length 100
    def pad(y: np.ndarray, length: int) -> np.ndarray:
        if len(y) >= length:
            return y[:length]
        out = np.zeros(length, dtype=float)
        out[: len(y)] = y
        return out

    bcs100 = pad(bcs, len(T))
    nodal100 = pad(nodal, len(T))
    p2_100 = pad(p2, len(T))
    p25_100 = pad(p25, len(T))
    p3_100 = pad(p3, len(T))

    # Experimental data has 110 points -> use its own dense T grid
    T_exp = np.linspace(T[0], T[-1], len(exp))

    # ---- Fit low-T power-law to experimental data
    # The experimental curve does not vanish at T_c=1K but instead approaches a
    # residual plateau ~67% (consistent with the published two-coil mutual-inductance
    # measurement where the change Delta D_s(T) = D_s(0) - D_s(T) follows a power law
    # in T at low T and reflects the gap structure).
    # Standard low-T form for a superconductor with a gap minimum delta_min (or nodes):
    #     D_s(T)/D_s(0) = 1 - c (T/Tc)^n
    # n=1 -> line nodes (d-wave); n=2 -> anisotropic / shallow gap minimum;
    # exponential -> fully gapped s-wave (BCS).
    # We fit on the low-T region 0 <= T <= 0.7 K (T < 0.7 Tc).
    # Fix Tc=1 K (degenerate with c at low T) and fit only D0, c, n.
    mask = T_exp <= 0.70
    Tf = T_exp[mask]
    Yf = exp[mask]

    def model_lowT(T, D0, c, n):
        return D0 * (1.0 - c * (T / 1.0) ** n)

    try:
        popt, pcov = curve_fit(
            model_lowT, Tf, Yf, p0=[100.0, 0.3, 2.5],
            bounds=([90, 0.0, 0.3], [110, 5.0, 6.0]),
        )
        perr = np.sqrt(np.diag(pcov))
        D0_fit, c_fit, n_fit = popt
        D0_err, c_err, n_err = perr
        yhat = model_lowT(Tf, *popt)
        ss_res = float(np.sum((Yf - yhat) ** 2))
        ss_tot = float(np.sum((Yf - Yf.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    except Exception:
        D0_fit = c_fit = n_fit = D0_err = c_err = n_err = float("nan")
        r2 = float("nan")

    # 4-parameter fit on the FULL experimental curve (including residual plateau)
    def model_full(T, D0, residual, n, Tc):
        ratio = np.clip(T / Tc, 0.0, 1.0)
        delta = D0 - residual
        return residual + delta * (1.0 - ratio ** n)

    try:
        popt2, pcov2 = curve_fit(
            model_full, T_exp, exp, p0=[100, 67, 2.5, 1.0],
            bounds=([90, 50, 0.5, 0.5], [110, 80, 6.0, 1.5]),
        )
        D0f, resf, nf, Tcf = popt2
        perr2 = np.sqrt(np.diag(pcov2))
        D0fe, rese, nfe, Tcfe = perr2
        yhat2 = model_full(T_exp, *popt2)
        ss_res2 = float(np.sum((exp - yhat2) ** 2))
        ss_tot2 = float(np.sum((exp - exp.mean()) ** 2))
        r2_full = 1.0 - ss_res2 / ss_tot2
    except Exception:
        D0f = resf = nf = Tcf = D0fe = rese = nfe = Tcfe = float("nan")
        r2_full = float("nan")

    # ---- Compare residuals at the model curves ----
    def rmse(theory_full: np.ndarray) -> float:
        # Interpolate theory onto T_exp
        th = np.interp(T_exp, T, theory_full)
        return float(np.sqrt(np.mean((exp - th) ** 2)))

    rmse_summary = {
        "BCS_s_wave": rmse(bcs100),
        "nodal_d_wave_linear": rmse(nodal100),
        "power_n=2.0": rmse(p2_100),
        "power_n=2.5": rmse(p25_100),
        "power_n=3.0": rmse(p3_100),
    }

    # Figure 3 — overlay all theory curves + experimental
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(T, bcs100, label="BCS s-wave", color="black", lw=1.8)
    ax.plot(T, nodal100, label="Nodal (linear, line-node)", color="C0", lw=1.6)
    ax.plot(T, p2_100, "--", label=r"Power law $n=2.0$", color="C2", lw=1.5)
    ax.plot(T, p25_100, "--", label=r"Power law $n=2.5$", color="C3", lw=1.5)
    ax.plot(T, p3_100, "--", label=r"Power law $n=3.0$", color="C4", lw=1.5)
    ax.plot(T_exp, exp, "o", ms=3, color="C1", label="Experimental (with noise)")
    ax.axvline(Tc_known, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel(r"Superfluid stiffness $D_s/D_{s,0}$ (%)")
    ax.set_title("Temperature dependence: theory vs experiment")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.savefig(os.path.join(IMG, "fig3_temperature_dependence.png"))
    plt.close(fig)

    # Figure 4 — fit
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(T_exp, exp, "o", ms=3, color="C1", label="Experimental")
    Tplot = np.linspace(0, Tc_known * 1.1, 300)
    if not np.isnan(n_fit):
        ax.plot(Tplot, model_lowT(Tplot, D0_fit, c_fit, n_fit), "-",
                color="C3", lw=2.2,
                label=(fr"Low-$T$ fit ($T<0.7$ K): $n={n_fit:.2f}\pm{n_err:.2f}$, "
                       fr"$c={c_fit:.2f}$"))
    if not np.isnan(nf):
        ax.plot(Tplot, model_full(Tplot, D0f, resf, nf, Tcf), "-",
                color="black", lw=1.4, alpha=0.7,
                label=(fr"Full fit: $n={nf:.2f}\pm{nfe:.2f}$, "
                       fr"$D_s^{{\rm res}}={resf:.1f}$"))
    ax.plot(T, p2_100, "--", color="C2", alpha=0.7, label="$n=2.0$ ref")
    ax.plot(T, p25_100, "--", color="C3", alpha=0.7, label="$n=2.5$ ref")
    ax.plot(T, p3_100, "--", color="C4", alpha=0.7, label="$n=3.0$ ref")
    ax.axvline(Tc_known, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel(r"$D_s/D_{s,0}$ (%)")
    ax.set_title("Power-law fit to experimental temperature dependence")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(IMG, "fig4_powerlaw_fit.png"))
    plt.close(fig)

    out = {
        "Tc_known_K": Tc_known,
        "D0_known": D0_known,
        "fit_range": "0 <= T <= 0.7 K (experimental)",
        "low_T_fit_Tc_fixed": {
            "model": "D_s(T) = D0*(1 - c*(T/Tc)^n) with Tc fixed at 1 K, fit on T<=0.7 K",
            "D0": float(D0_fit), "D0_err": float(D0_err),
            "c": float(c_fit), "c_err": float(c_err),
            "n_exponent": float(n_fit), "n_err": float(n_err),
            "R2": float(r2),
            "interpretation": (
                "n=1 -> line nodes; n=2 -> nodeless anisotropic gap; "
                "BCS s-wave is exponential at low T (effective n>>3). "
                "n<1 indicates strong sub-linear suppression typical of nodal pairing."
            ),
        },
        "full_curve_fit": {
            "model": "D_s(T) = D_res + (D0-D_res)*(1-(T/Tc)^n) for T<Tc",
            "D0": float(D0f), "D0_err": float(D0fe),
            "D_s_residual_pct": float(resf), "D_s_residual_err": float(rese),
            "n_exponent": float(nf), "n_err": float(nfe),
            "Tc_K": float(Tcf), "Tc_err": float(Tcfe),
            "R2": float(r2_full),
        },
        "rmse_vs_theory_models": rmse_summary,
    }
    with open(os.path.join(OUT, "temperature_fits.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# 3) Current dependence
# ---------------------------------------------------------------------------
def analysis_current() -> dict:
    I_dc = get("i_dc in na")  # 50 pts
    gl = get("d_s_gl")
    lin = get("d_s_linear")
    dc_exp = get("d_s_dc_exp")  # 85 points
    P_mw = get("p_mw normalized")
    I_mw = get("i_mw_amplitude in na")
    mw_exp = get("d_s_mw_exp")

    Ic_known = 50.0
    D0_known = 100.0

    # The dc theory arrays have len 60 / 50, but I_dc has 50.
    # Both gl and dc_exp are sampled on extended grids; use the appropriate axis.
    I_dc_60 = np.linspace(I_dc[0], I_dc[1] * (60 - 1), 60)  # GL has 60 pts; assume same step
    # Actually, GL has 60 points; safer to assume linspace from 0 to (60/50)*60 ~ 72 nA
    # Build it more robustly:
    step = I_dc[1] - I_dc[0]
    I_gl = I_dc[0] + step * np.arange(len(gl))
    I_lin = I_dc[0] + step * np.arange(len(lin))
    I_exp = I_dc[0] + step * np.arange(len(dc_exp))

    # Test the leading-order quadratic Ginzburg-Landau prediction:
    #   1 - D_s/D0 = (I/Ic)^2 + O(I^4)
    # using only the small-current regime (I <= 0.4*Ic) where the quadratic
    # expansion is dominant. Slope (vs (I/Ic)^2) is expected to be 1.
    Ismall = 0.4 * Ic_known
    mask_gl_small = (I_gl > 0) & (I_gl <= Ismall)
    x_gl = (I_gl[mask_gl_small] / Ic_known) ** 2
    y_gl = (D0_known - gl[mask_gl_small]) / D0_known
    slope_gl_small, intercept_gl_small = np.polyfit(x_gl, y_gl, 1)

    # Same diagnostic for the experimental DC curve.
    mask_exp_small = (I_exp > 0) & (I_exp <= Ismall)
    x_e = (I_exp[mask_exp_small] / Ic_known) ** 2
    y_e = (D0_known - dc_exp[mask_exp_small]) / D0_known
    slope_exp_small, intercept_exp_small = np.polyfit(x_e, y_e, 1)

    # Also do a global GL theory fit to identify pair-breaking critical current
    # via D_s vanishing.  We fit a flexible (1 - (I/Ic_eff)^2)^p form.
    nonz = gl > 0
    def gl_model(I, D0, Ic, p):
        v = D0 * np.maximum(1.0 - (I / Ic) ** 2, 0.0) ** p
        return v
    try:
        popt_gl, _ = curve_fit(gl_model, I_gl[nonz], gl[nonz],
                               p0=[100, 60, 1.0],
                               bounds=([50, 30, 0.3], [150, 100, 3.0]))
        D0_glfit, Ic_glfit, p_glfit = popt_gl
    except Exception:
        D0_glfit = Ic_glfit = p_glfit = float("nan")

    # Identify experimental "minimum" — re-entrant rise indicates joule heating /
    # vortex flow regime in the simulator.  Report the first minimum.
    i_min = int(np.argmin(dc_exp))
    Ic_exp_est = float(I_exp[i_min])
    # Backwards-compatibility variables for plotting block below
    mask_gl = (gl > 0) & (I_gl <= Ic_known)
    mask_e = I_exp <= Ic_exp_est

    # Fit microwave: D_s_mw = D0*(1 - alpha * I_mw^2)
    # Use small amplitude part where it's still monotone
    mask_mw = I_mw < I_mw.max() * 0.9
    A = np.vstack([I_mw[mask_mw] ** 2, np.ones(mask_mw.sum())]).T
    coef, *_ = np.linalg.lstsq(A, mw_exp[mask_mw], rcond=None)
    alpha_mw, c0_mw = coef

    # Figure 5 — DC current dependence (overlay)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(I_gl, gl, "C0-", label="Ginzburg-Landau model")
    ax.plot(I_lin, lin, "C2--", label="Linear Meissner model")
    ax.plot(I_exp, dc_exp, "C3o-", ms=3, label="Experimental DC")
    ax.axvline(Ic_known, color="grey", ls=":", alpha=0.7, label=r"$I_c=50$ nA (theory)")
    ax.set_xlabel(r"DC bias current $I_{\rm dc}$ (nA)")
    ax.set_ylabel(r"$D_s/D_{s,0}$ (%)")
    ax.set_title("DC current dependence of superfluid stiffness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(IMG, "fig5_current_dc.png"))
    plt.close(fig)

    # Figure 6 — quadratic GL test in small-current regime
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a = axes[0]
    nonz_dc = (I_exp <= Ic_known) & (I_exp >= 0)
    a.plot((I_gl[mask_gl] / Ic_known) ** 2, (D0_known - gl[mask_gl]) / D0_known,
           "C0o", ms=4, label="GL theory")
    a.plot((I_exp[nonz_dc] / Ic_known) ** 2,
           (D0_known - dc_exp[nonz_dc]) / D0_known,
           "C3s", ms=3.5, label="Experimental DC")
    xx = np.linspace(0, 1, 50)
    a.plot(xx, xx, "k--", label="$y=x$ (leading-order GL)")
    a.set_xlabel(r"$(I/I_c)^2$")
    a.set_ylabel(r"$1 - D_s/D_{s,0}$")
    a.set_title("Quadratic GL: full range")
    a.legend()
    a.grid(True, alpha=0.3)

    a = axes[1]
    a.plot(x_gl, y_gl, "C0o", ms=5,
           label=fr"GL theory: slope$={slope_gl_small:.3f}$")
    a.plot(x_e, y_e, "C3s", ms=4.5,
           label=fr"Experiment: slope$={slope_exp_small:.3f}$")
    xs = np.linspace(0, max(x_gl.max(), x_e.max()) * 1.05, 50)
    a.plot(xs, xs, "k--", label="ideal slope $=1$")
    a.set_xlabel(r"$(I/I_c)^2$  (small-current regime)")
    a.set_ylabel(r"$1 - D_s/D_{s,0}$")
    a.set_title("Small-current GL slope test")
    a.legend()
    a.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig6_GL_quadratic.png"))
    plt.close(fig)

    # Figure 7 — microwave
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a = axes[0]
    a.plot(P_mw, mw_exp, "C4o-", ms=3.5)
    a.set_xlabel("Normalized microwave power $P_{\\rm mw}$")
    a.set_ylabel(r"$D_s/D_{s,0}$ (%)")
    a.set_title("Microwave-induced suppression vs power")
    a.grid(True, alpha=0.3)
    a = axes[1]
    a.plot(I_mw, mw_exp, "C5o-", ms=3.5, label="Experiment")
    Ifit = np.linspace(0, I_mw.max(), 200)
    a.plot(Ifit, c0_mw + alpha_mw * Ifit ** 2, "k--",
           label=fr"Quadratic fit: $\alpha={alpha_mw:.4g}$")
    a.set_xlabel("Microwave current amplitude $I_{\\rm mw}$ (nA)")
    a.set_ylabel(r"$D_s/D_{s,0}$ (%)")
    a.set_title("Microwave-induced suppression vs current")
    a.legend()
    a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig7_microwave.png"))
    plt.close(fig)

    out = {
        "Ic_theory_nA": Ic_known,
        "GL_smallI_quadratic_test": {
            "fit_range_nA": [0.0, float(Ismall)],
            "slope_GL_theory": float(slope_gl_small),
            "intercept_GL_theory": float(intercept_gl_small),
            "slope_experiment": float(slope_exp_small),
            "intercept_experiment": float(intercept_exp_small),
            "expected_slope": 1.0,
            "expected_intercept": 0.0,
        },
        "GL_full_fit_to_theory": {
            "model": "D_s(I) = D0 * (1 - (I/Ic_eff)^2)^p",
            "D0": float(D0_glfit), "Ic_eff_nA": float(Ic_glfit),
            "p_exponent": float(p_glfit),
        },
        "experimental_DC": {
            "first_minimum_at_I_nA": Ic_exp_est,
            "min_Ds_value_pct": float(dc_exp.min()),
            "rises_after_minimum": bool(dc_exp[-1] > dc_exp[i_min]),
            "comment": "Experiment shows re-entrant rise above the first minimum, "
                       "consistent with vortex-flow / heating regime above Ic.",
        },
        "microwave_quadratic": {
            "alpha_per_nA2": float(alpha_mw),
            "intercept": float(c0_mw),
            "I_mw_max_nA": float(I_mw.max()),
            "delta_Ds_at_max_pct": float(100 - mw_exp.min()),
        },
    }
    with open(os.path.join(OUT, "current_fits.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    s_density = analysis_density()
    s_temp = analysis_temperature()
    s_curr = analysis_current()
    full = {"density": s_density, "temperature": s_temp, "current": s_curr}
    with open(os.path.join(OUT, "analysis_summary.json"), "w") as f:
        json.dump(full, f, indent=2)
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
