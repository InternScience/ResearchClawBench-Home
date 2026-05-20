import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from pathlib import Path

# Paths
workspace = Path(__file__).resolve().parent.parent
npz_path = workspace / "outputs" / "parsed_data.npz"
img_dir = workspace / "report" / "images"
img_dir.mkdir(parents=True, exist_ok=True)
out_dir = workspace / "outputs"
out_dir.mkdir(exist_ok=True)

data = np.load(npz_path)

# Unpack arrays
n_eff = data["n_eff"]
D_s_conv = data["D_s_conv"]
D_s_geom = data["D_s_geom"]
D_s_exp_hole = data["D_s_exp_hole"]
D_s_exp_electron = data["D_s_exp_electron"]

T_common = data["T"]  # length 100, 0..1.2 K
D_s_bcs_raw = data["D_s_bcs"]          # length 95
D_s_nodal_raw = data["D_s_nodal"]      # length 95
D_s_power_n2_raw = data["D_s_power_n2"] # length 95
D_s_power_n2_5_raw = data["D_s_power_n2_5"] # length 90
D_s_power_n3_raw = data["D_s_power_n3"]     # length 90
D_s_exp_temp_raw = data["D_s_experimental"] # length 110

I_dc = data["I_dc"]  # length 50, 0..60 nA
D_s_gl_raw = data["D_s_gl"]        # length 60
D_s_linear = data["D_s_linear"]    # length 50
D_s_dc_exp_raw = data["D_s_dc_exp"] # length 85
P_mw = data["P_mw"]
I_mw_amplitude = data["I_mw_amplitude"]
D_s_mw_exp = data["D_s_mw_exp"]

results = {}

# Helper to interpolate onto a common axis
def align_to_common(common_axis, raw_array, raw_max):
    raw_axis = np.linspace(0, raw_max, len(raw_array))
    return np.interp(common_axis, raw_axis, raw_array, left=raw_array[0], right=raw_array[-1])

T_max = float(T_common[-1])  # 1.2 K
I_max = float(I_dc[-1])      # 60 nA

# Align temperature arrays to T_common
D_s_bcs = align_to_common(T_common, D_s_bcs_raw, T_max)
D_s_nodal = align_to_common(T_common, D_s_nodal_raw, T_max)
D_s_power_n2 = align_to_common(T_common, D_s_power_n2_raw, T_max)
D_s_power_n2_5 = align_to_common(T_common, D_s_power_n2_5_raw, T_max)
D_s_power_n3 = align_to_common(T_common, D_s_power_n3_raw, T_max)
D_s_exp_temp = align_to_common(T_common, D_s_exp_temp_raw, T_max)

# Align current arrays to I_dc
D_s_gl = align_to_common(I_dc, D_s_gl_raw, I_max)
D_s_dc_exp = align_to_common(I_dc, D_s_dc_exp_raw, I_max)

# ============================================================
# 1. Carrier Density Dependence
# ============================================================
ratio_geom_conv = D_s_geom / D_s_conv
ratio_hole_conv = D_s_exp_hole / D_s_conv
ratio_electron_conv = D_s_exp_electron / D_s_conv

results["carrier_density"] = {
    "mean_ratio_geom_conv": float(np.mean(ratio_geom_conv)),
    "mean_ratio_hole_conv": float(np.mean(ratio_hole_conv)),
    "mean_ratio_electron_conv": float(np.mean(ratio_electron_conv)),
    "max_ratio_geom_conv": float(np.max(ratio_geom_conv)),
    "max_ratio_hole_conv": float(np.max(ratio_hole_conv)),
    "max_ratio_electron_conv": float(np.max(ratio_electron_conv)),
}

# Figure 1: D_s vs n_eff (log scale)
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(n_eff, D_s_conv, label="Conventional (Fermi liquid)", lw=2)
ax.semilogy(n_eff, D_s_geom, label="Quantum geometric", lw=2)
ax.semilogy(n_eff, D_s_exp_hole, label="Exp. hole-doped", lw=2, marker='o', markersize=3)
ax.semilogy(n_eff, D_s_exp_electron, label="Exp. electron-doped", lw=2, marker='s', markersize=3)
ax.set_xlabel(r"Carrier density $n_{\mathrm{eff}}$ (m$^{-2}$)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (H$^{-1}$)")
ax.set_title("Carrier-density dependence of superfluid stiffness")
ax.legend()
ax.grid(True, ls="--", alpha=0.4, which='both')
fig.tight_layout()
fig.savefig(img_dir / "fig1_carrier_density.png", dpi=300)
plt.close(fig)

# Figure 2: Enhancement ratio
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_eff * 1e-14, ratio_geom_conv, label=r"$D_s^{\mathrm{geom}} / D_s^{\mathrm{conv}}$", lw=2)
ax.plot(n_eff * 1e-14, ratio_hole_conv, label=r"$D_s^{\mathrm{hole}} / D_s^{\mathrm{conv}}$", lw=2)
ax.plot(n_eff * 1e-14, ratio_electron_conv, label=r"$D_s^{\mathrm{elec}} / D_s^{\mathrm{conv}}$", lw=2)
ax.set_xlabel(r"Carrier density $n_{\mathrm{eff}}$ ($10^{14}$ m$^{-2}$)")
ax.set_ylabel("Enhancement ratio")
ax.set_title("Quantum-geometric enhancement of superfluid stiffness")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig2_enhancement_ratio.png", dpi=300)
plt.close(fig)

# ============================================================
# 2. Temperature Dependence
# ============================================================
T_c = 1.0
D_s0 = 100.0

# Fit experimental temperature data (use raw to avoid interpolation bias)
T_exp_raw = np.linspace(0, T_max, len(D_s_exp_temp_raw))
mask_T = T_exp_raw <= T_c
T_fit = T_exp_raw[mask_T]
D_fit = D_s_exp_temp_raw[mask_T]

def power_law_model(T, n):
    return D_s0 * np.maximum(0.0, 1.0 - (T / T_c) ** n)

popt, pcov = curve_fit(power_law_model, T_fit, D_fit, p0=[2.5], bounds=([1.0], [5.0]))
n_fit = float(popt[0])
n_err = float(np.sqrt(pcov[0, 0]))

# Chi-squared on aligned data for comparison
mask_common = T_common <= T_c
chi2_bcs = float(np.sum((D_s_exp_temp[mask_common] - D_s_bcs[mask_common]) ** 2))
chi2_nodal = float(np.sum((D_s_exp_temp[mask_common] - D_s_nodal[mask_common]) ** 2))
chi2_power_fit = float(np.sum((D_s_exp_temp[mask_common] - power_law_model(T_common[mask_common], n_fit)) ** 2))

results["temperature_dependence"] = {
    "fitted_exponent_n": n_fit,
    "fitted_exponent_n_err": n_err,
    "chi2_vs_bcs": chi2_bcs,
    "chi2_vs_nodal": chi2_nodal,
    "chi2_vs_power_fit": chi2_power_fit,
}

# Figure 3: D_s vs T
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(T_common, D_s_bcs, label="BCS (s-wave)", lw=2)
ax.plot(T_common, D_s_nodal, label="Nodal (linear)", lw=2)
ax.plot(T_common, D_s_power_n2, label="Power law n=2", lw=2, ls="--")
ax.plot(T_common, D_s_power_n2_5, label="Power law n=2.5", lw=2, ls="--")
ax.plot(T_common, D_s_power_n3, label="Power law n=3", lw=2, ls="--")
ax.plot(T_common, D_s_exp_temp, label="Experiment", lw=2, color="black", alpha=0.7)
fit_label = f"Fit n = {n_fit:.2f} ± {n_err:.2f}"
ax.plot(T_common[mask_common], power_law_model(T_common[mask_common], n_fit), label=fit_label, lw=2, color="red", ls=":")
ax.set_xlabel(r"Temperature $T$ (K)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (normalized)")
ax.set_title("Temperature dependence of superfluid stiffness")
ax.legend(loc="upper right")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig3_temperature_dependence.png", dpi=300)
plt.close(fig)

# Figure 4: Log-log plot of (1 - D_s/D_s0) vs T (only for T>0 and D_s<D_s0)
mask_log = (T_common > 0) & (T_common < T_c) & (D_s_exp_temp < D_s0) & (D_s_exp_temp > 0)
log_T = np.log10(T_common[mask_log])
log_y_exp = np.log10(1.0 - D_s_exp_temp[mask_log] / D_s0)
log_y_bcs = np.log10(1.0 - D_s_bcs[mask_log] / D_s0)
log_y_nodal = np.log10(1.0 - D_s_nodal[mask_log] / D_s0)
log_y_n2 = np.log10(1.0 - D_s_power_n2[mask_log] / D_s0)
log_y_n25 = np.log10(1.0 - D_s_power_n2_5[mask_log] / D_s0)
log_y_n3 = np.log10(1.0 - D_s_power_n3[mask_log] / D_s0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(log_T, log_y_bcs, label="BCS", lw=2)
ax.plot(log_T, log_y_nodal, label="Nodal", lw=2)
ax.plot(log_T, log_y_n2, label="n=2", lw=2, ls="--")
ax.plot(log_T, log_y_n25, label="n=2.5", lw=2, ls="--")
ax.plot(log_T, log_y_n3, label="n=3", lw=2, ls="--")
ax.plot(log_T, log_y_exp, label="Experiment", lw=2, color="black", alpha=0.7)
slope, intercept = np.polyfit(log_T, log_y_exp, 1)
ax.plot(log_T, slope * log_T + intercept, color="red", ls=":", lw=2, label=f"Fit slope = {slope:.2f}")
ax.set_xlabel(r"$\log_{10}(T)$")
ax.set_ylabel(r"$\log_{10}(1 - D_s/D_{s0})$")
ax.set_title("Power-law scaling of superfluid stiffness suppression")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig4_loglog_temperature.png", dpi=300)
plt.close(fig)

# ============================================================
# 3. Current Dependence
# ============================================================
I_c = 50.0

# Fit DC experimental data (use raw to avoid interpolation bias)
I_dc_raw = np.linspace(0, I_max, len(D_s_dc_exp_raw))
mask_I = I_dc_raw <= I_c

def gl_model(I, Ds0, Ic):
    return Ds0 * np.maximum(0.0, 1.0 - (I / Ic) ** 2)

popt_dc, pcov_dc = curve_fit(gl_model, I_dc_raw[mask_I], D_s_dc_exp_raw[mask_I], p0=[100.0, 50.0])
Ds0_dc, Ic_dc = float(popt_dc[0]), float(popt_dc[1])

# Fit microwave data (already aligned)
def mw_model(I, Ds0, a):
    return Ds0 - a * I ** 2

popt_mw, pcov_mw = curve_fit(mw_model, I_mw_amplitude, D_s_mw_exp, p0=[100.0, 0.01])
Ds0_mw, a_mw = float(popt_mw[0]), float(popt_mw[1])

results["current_dependence"] = {
    "Dc_fit_Ds0": Ds0_dc,
    "Dc_fit_Ic": Ic_dc,
    "MW_fit_Ds0": Ds0_mw,
    "MW_fit_quadratic_coeff_a": a_mw,
}

# Figure 5: DC current dependence
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(I_dc, D_s_gl, label="Ginzburg-Landau", lw=2)
ax.plot(I_dc, D_s_linear, label="Linear Meissner", lw=2)
ax.plot(I_dc, D_s_dc_exp, label="Experiment", lw=2, color="black", alpha=0.7, marker='o', markersize=3)
ax.plot(I_dc_raw[mask_I], gl_model(I_dc_raw[mask_I], Ds0_dc, Ic_dc), label="Fit to exp.", lw=2, color="red", ls=":")
ax.set_xlabel(r"DC current $I_{\mathrm{dc}}$ (nA)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (normalized)")
ax.set_title("DC current dependence of superfluid stiffness")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig5_dc_current_dependence.png", dpi=300)
plt.close(fig)

# Figure 6: Microwave current dependence
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(I_mw_amplitude, D_s_mw_exp, label="Experiment", lw=2, color="black", alpha=0.7, marker='o', markersize=3)
ax.plot(I_mw_amplitude, mw_model(I_mw_amplitude, Ds0_mw, a_mw), label="Quadratic fit", lw=2, color="red", ls=":")
ax.set_xlabel(r"Microwave current amplitude $I_{\mathrm{mw}}$ (nA)")
ax.set_ylabel(r"Superfluid stiffness $D_s$ (normalized)")
ax.set_title("Microwave-induced suppression of superfluid stiffness")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(img_dir / "fig6_microwave_dependence.png", dpi=300)
plt.close(fig)

# Save results
with open(out_dir / "analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Analysis complete. Results saved to", out_dir / "analysis_results.json")
print("Figures saved to", img_dir)
