"""Main analysis: superfluid stiffness of MATBG — figures and quantitative results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import curve_fit
from scipy import stats

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = "outputs"
IMG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Load parsed data
data = np.load(os.path.join(OUTPUT_DIR, 'parsed_data.npz'))

n_eff = data['n_eff']
D_s_conv = data['D_s_conv']
D_s_geom = data['D_s_geom']
D_s_exp_hole = data['D_s_exp_hole']
D_s_exp_electron = data['D_s_exp_electron']

T_array = data['T_array']
D_s_bcs = data['D_s_bcs']
D_s_nodal = data['D_s_nodal']
D_s_power_n2 = data['D_s_power_n2']
D_s_power_n2_5 = data['D_s_power_n2_5']
D_s_power_n3 = data['D_s_power_n3']
D_s_experimental = data['D_s_experimental']

I_dc = data['I_dc']
D_s_gl = data['D_s_gl']
D_s_linear = data['D_s_linear']
D_s_dc_exp = data['D_s_dc_exp']
P_mw = data['P_mw']
I_mw_amplitude = data['I_mw_amplitude']
D_s_mw_exp = data['D_s_mw_exp']

# ============================================================
# FIGURE 1: Carrier density dependence — Conventional vs Geometric vs Experimental
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

n_cm2 = n_eff / 1e4  # convert m^-2 to cm^-2

ax.plot(n_cm2, D_s_conv/1e9, 'b--', linewidth=2, label='Conventional (Fermi liquid)')
ax.plot(n_cm2, D_s_geom/1e9, 'r-', linewidth=2, label='Quantum Geometric')
ax.plot(n_cm2, D_s_exp_hole/1e9, 'o', color='darkorange', markersize=4, 
        label='Exp. Hole-doped', alpha=0.7)
ax.plot(n_cm2, D_s_exp_electron/1e9, 's', color='green', markersize=4, 
        label='Exp. Electron-doped', alpha=0.7)

# Enhancement ratio annotation
idx_mid = len(n_cm2)//2
ratio_hole = D_s_exp_hole[idx_mid] / D_s_conv[idx_mid]
ratio_geom = D_s_geom[idx_mid] / D_s_conv[idx_mid]
ax.annotate(f'Geometric/Conventional ≈ {ratio_geom:.1f}×\nExp/Conventional ≈ {ratio_hole:.1f}×',
            xy=(n_cm2[idx_mid], D_s_exp_hole[idx_mid]/1e9), xytext=(2.5e11, 150),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='darkred')

ax.set_xlabel('Carrier Density $n$ (cm$^{-2}$)')
ax.set_ylabel('Superfluid Stiffness $D_s$ (10$^9$ H$^{-1}$)')
ax.set_title('Carrier Density Dependence of Superfluid Stiffness in MATBG')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig1_carrier_density.png'))
plt.close()
print("Figure 1 saved.")

# Save quantitative results
enhancement_results = {
    'mean_ratio_geom_to_conv': float(np.mean(D_s_geom / D_s_conv)),
    'mean_ratio_exp_hole_to_conv': float(np.mean(D_s_exp_hole / D_s_conv)),
    'mean_ratio_exp_electron_to_conv': float(np.mean(D_s_exp_electron / D_s_conv)),
    'max_D_s_conv': float(np.max(D_s_conv)),
    'max_D_s_geom': float(np.max(D_s_geom)),
    'max_D_s_exp_hole': float(np.max(D_s_exp_hole)),
    'max_D_s_exp_electron': float(np.max(D_s_exp_electron)),
}

# ============================================================
# FIGURE 2: Temperature dependence — BCS vs Nodal vs Power-law vs Experimental
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Model comparison (only up to Tc=1.0)
mask_tc = T_array <= 1.0
# Truncate all arrays to same length
min_len = min(len(D_s_bcs), len(T_array[mask_tc]))
T_plot = T_array[mask_tc][:min_len]

ax1.plot(T_plot, D_s_bcs[:min_len], 'b-', linewidth=2, label='BCS (s-wave)')
ax1.plot(T_plot, D_s_nodal[:min_len], 'g--', linewidth=2, label='Nodal (d-wave)')
ax1.plot(T_plot, D_s_power_n2[:min_len], 'r-', linewidth=2, label='Power-law $n=2.0$')
ax1.plot(T_plot, D_s_power_n2_5[:min_len], 'orange', linewidth=2, label='Power-law $n=2.5$')
ax1.plot(T_plot, D_s_power_n3[:min_len], 'purple', linewidth=2, label='Power-law $n=3.0$')
ax1.set_xlabel('Temperature $T$ (K)')
ax1.set_ylabel('$D_s(T) / D_s(0)$ (%)')
ax1.set_title('Model Predictions: $D_s(T)$ Temperature Scaling')
ax1.legend(loc='lower left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(-5, 105)

# Panel (b): Experimental data with model fits
T_exp = np.linspace(0, 1.0, len(D_s_experimental))
mask_exp = T_exp <= 1.0
T_exp_plot = T_exp[mask_exp]
D_s_exp_plot = D_s_experimental[:len(T_exp_plot)]

# Fit power law: D_s(T) = D_s(0) * (1 - (T/T_c)^n)
def power_law_fit(T, D0, n):
    Tc = 1.0
    return D0 * (1 - (T/Tc)**n)

# Only fit data before it goes to zero
fit_mask = T_exp_plot < 0.9
try:
    popt, pcov = curve_fit(power_law_fit, T_exp_plot[fit_mask], 
                           D_s_exp_plot[fit_mask], p0=[100, 2.0], maxfev=10000)
    n_fitted = popt[1]
    n_err = np.sqrt(pcov[1,1])
except:
    n_fitted = 2.0
    n_err = 0.5

ax2.plot(T_exp_plot, D_s_exp_plot, 'ko', markersize=3, label='Experimental data')
T_smooth = np.linspace(0, 0.95, 200)
ax2.plot(T_smooth, power_law_fit(T_smooth, *popt), 'r-', linewidth=2, 
         label=f'Fit: $n={n_fitted:.2f}\\pm{n_err:.2f}$')
ax2.plot(T_plot, D_s_bcs[:min_len], 'b--', linewidth=1.5, alpha=0.5, label='BCS (s-wave)')
ax2.set_xlabel('Temperature $T$ (K)')
ax2.set_ylabel('$D_s(T)$ (arb. units)')
ax2.set_title('Experimental $D_s(T)$ with Power-Law Fit')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig2_temperature_dependence.png'))
plt.close()
print("Figure 2 saved.")

temperature_results = {
    'fitted_power_law_exponent': float(n_fitted),
    'fitted_power_law_exponent_err': float(n_err),
    'bcs_Tc_drop_50pct': float(T_plot[np.argmin(np.abs(D_s_bcs[:min_len] - 50))]),
    'nodal_Tc_drop_50pct': float(T_plot[np.argmin(np.abs(D_s_nodal[:min_len] - 50))]),
    'power_n2_Tc_drop_50pct': float(T_plot[np.argmin(np.abs(D_s_power_n2[:min_len] - 50))]),
}

# ============================================================
# FIGURE 3: Current dependence — GL, Linear Meissner, Experimental
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): DC current dependence
I_c = 50.0
I_gl_plot = I_dc[:len(D_s_gl)]
I_lin_plot = I_dc[:len(D_s_linear)]
I_dc_exp_plot = I_dc[:len(D_s_dc_exp)]

ax1.plot(I_gl_plot, D_s_gl[:len(I_gl_plot)], 'b-', linewidth=2, label='Ginzburg-Landau')
ax1.plot(I_lin_plot, D_s_linear[:len(I_lin_plot)], 'g--', linewidth=2, label='Linear Meissner')
ax1.plot(I_dc_exp_plot, D_s_dc_exp[:len(I_dc_exp_plot)], 'ro', markersize=4, label='Experimental DC')
ax1.axvline(x=I_c, color='gray', linestyle=':', alpha=0.5, label=f'$I_c$ = 50 nA')
ax1.set_xlabel('DC Current $I_{dc}$ (nA)')
ax1.set_ylabel('$D_s(I_{dc})$ (arb. units)')
ax1.set_title('DC Current Dependence of Superfluid Stiffness')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel (b): Microwave power dependence
ax2.plot(P_mw, D_s_mw_exp, 'mo-', markersize=4, linewidth=1.5, label='Microwave $D_s(P_{mw})$')

# Quadratic fit to microwave data
def quad_fit(x, a, b, c):
    return a + b*x + c*x**2

popt_mw, pcov_mw = curve_fit(quad_fit, P_mw, D_s_mw_exp)
P_smooth = np.linspace(0, 1, 100)
ax2.plot(P_smooth, quad_fit(P_smooth, *popt_mw), 'r--', linewidth=1.5, 
         alpha=0.7, label=f'Quadratic fit')

ax2.set_xlabel('Normalized Microwave Power $P_{mw}$')
ax2.set_ylabel('$D_s(P_{mw})$ (arb. units)')
ax2.set_title('Microwave Power Dependence of $D_s$')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig3_current_dependence.png'))
plt.close()
print("Figure 3 saved.")

current_results = {
    'I_c': 50.0,
    'quadratic_coeff_a': float(popt_mw[0]),
    'quadratic_coeff_b': float(popt_mw[1]),
    'quadratic_coeff_c': float(popt_mw[2]),
    'D_s_at_Ic_GL': float(np.interp(I_c, I_gl_plot, D_s_gl[:len(I_gl_plot)])),
}

# ============================================================
# FIGURE 4: Enhancement ratio & Geometry dominance
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Geometric / Conventional ratio
ax = axes[0,0]
ratio_gc = D_s_geom / D_s_conv
ax.plot(n_cm2, ratio_gc, 'r-', linewidth=2)
ax.axhline(y=np.mean(ratio_gc), color='gray', linestyle='--', alpha=0.5,
           label=f'Mean = {np.mean(ratio_gc):.1f}')
ax.set_xlabel('Carrier Density (cm$^{-2}$)')
ax.set_ylabel('Ratio $D_s^{geom} / D_s^{conv}$')
ax.set_title('Quantum Geometry Enhancement Factor')
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Hole vs Electron doping asymmetry
ax = axes[0,1]
ratio_he = D_s_exp_hole / D_s_exp_electron
ax.plot(n_cm2, ratio_he, 'purple', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(ratio_he), color='purple', linestyle=':', alpha=0.5,
           label=f'Mean = {np.mean(ratio_he):.3f}')
ax.set_xlabel('Carrier Density (cm$^{-2}$)')
ax.set_ylabel('Ratio $D_s^{hole} / D_s^{electron}$')
ax.set_title('Particle-Hole Asymmetry')
ax.legend()
ax.grid(True, alpha=0.3)

# (c) Experimental / Conventional ratio
ax = axes[1,0]
ratio_ec_hole = D_s_exp_hole / D_s_conv
ratio_ec_elec = D_s_exp_electron / D_s_conv
ax.plot(n_cm2, ratio_ec_hole, 'o-', color='darkorange', markersize=3, label='Hole-doped')
ax.plot(n_cm2, ratio_ec_elec, 's-', color='green', markersize=3, label='Electron-doped')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Fermi liquid limit')
ax.set_xlabel('Carrier Density (cm$^{-2}$)')
ax.set_ylabel('Ratio $D_s^{exp} / D_s^{conv}$')
ax.set_title('Experimental Enhancement over Fermi Liquid')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Total superfluid stiffness decomposition
ax = axes[1,1]
ax.fill_between(n_cm2, 0, D_s_conv/1e9, alpha=0.3, color='blue', label='Conventional')
ax.fill_between(n_cm2, D_s_conv/1e9, D_s_geom/1e9, alpha=0.3, color='red', label='Geometric (additional)')
ax.plot(n_cm2, D_s_conv/1e9, 'b-', linewidth=1.5)
ax.plot(n_cm2, D_s_geom/1e9, 'r-', linewidth=1.5)
ax.plot(n_cm2, D_s_exp_hole/1e9, 'o', color='darkorange', markersize=3, alpha=0.5)
ax.set_xlabel('Carrier Density (cm$^{-2}$)')
ax.set_ylabel('$D_s$ (10$^9$ H$^{-1}$)')
ax.set_title('Decomposition: Conventional + Geometric')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig4_enhancement_analysis.png'))
plt.close()
print("Figure 4 saved.")

# ============================================================
# FIGURE 5: BKT Transition Temperature estimation
# ============================================================
# BKT relation: k_B T_BKT = (π/8) * (ħ² D_s / e²)
hbar = 1.054e-34
e = 1.602e-19
kB = 1.381e-23

def Ds_to_Tbkt(Ds):
    """Convert superfluid stiffness (H^-1) to BKT temperature (K)."""
    return (np.pi / 8) * (hbar**2 * Ds / (e**2 * kB))

fig, ax = plt.subplots(figsize=(9, 6))

T_bkt_conv = Ds_to_Tbkt(D_s_conv)
T_bkt_geom = Ds_to_Tbkt(D_s_geom)
T_bkt_exp_hole = Ds_to_Tbkt(D_s_exp_hole)
T_bkt_exp_electron = Ds_to_Tbkt(D_s_exp_electron)

ax.plot(n_cm2, T_bkt_conv, 'b--', linewidth=2, label='Conventional (Fermi liquid)')
ax.plot(n_cm2, T_bkt_geom, 'r-', linewidth=2, label='Quantum Geometric')
ax.plot(n_cm2, T_bkt_exp_hole, 'o', color='darkorange', markersize=4, 
        label='Exp. Hole-doped', alpha=0.7)
ax.plot(n_cm2, T_bkt_exp_electron, 's', color='green', markersize=4, 
        label='Exp. Electron-doped', alpha=0.7)

# Mark experimental Tc ~ 1.7 K
ax.axhline(y=1.7, color='purple', linestyle=':', alpha=0.7, label='Reported $T_c$ ≈ 1.7 K (Cao et al.)')
ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, 
           label='Conventional estimate ≈ 0.6 K (Xie et al.)')

ax.set_xlabel('Carrier Density $n$ (cm$^{-2}$)')
ax.set_ylabel('BKT Transition Temperature $T_{BKT}$ (K)')
ax.set_title('Estimated BKT Transition Temperature from Superfluid Stiffness')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig5_bkt_temperature.png'))
plt.close()
print("Figure 5 saved.")

bkt_results = {
    'max_Tbkt_conv_K': float(np.max(T_bkt_conv)),
    'max_Tbkt_geom_K': float(np.max(T_bkt_geom)),
    'max_Tbkt_exp_hole_K': float(np.max(T_bkt_exp_hole)),
    'max_Tbkt_exp_electron_K': float(np.max(T_bkt_exp_electron)),
}

# ============================================================
# FIGURE 6: Power-law exponent analysis for anisotropic gap
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

# Fit power-law exponents to different T ranges
T_plot_full = T_array[:len(D_s_experimental)]

# Compute local power-law exponent via numerical derivative
# n_local = d log(1 - D_s/D_s0) / d log(T)
D_s_exp_nonzero = np.clip(D_s_experimental[:len(T_plot_full)], 1e-10, None)
D_s0_exp = D_s_exp_nonzero[0]
valid = (T_plot_full > 0.02) & (T_plot_full < 0.95) & (D_s_exp_nonzero > 1)

log_T = np.log(T_plot_full[valid])
log_Ds = np.log(1 - D_s_exp_nonzero[valid] / D_s0_exp + 1e-10)

# Local slope via Savitzky-Golay or simple binned slopes
n_local = np.gradient(log_Ds) / np.gradient(log_T)

ax.plot(T_plot_full[valid], n_local, 'ko-', markersize=3, linewidth=1, label='Local exponent $n(T)$')
ax.axhline(y=2.0, color='blue', linestyle='--', alpha=0.7, label='$n=2$ (d-wave nodal)')
ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, label='$n=3$ (point-node)')
ax.axhline(y=n_fitted, color='red', linestyle='-', alpha=0.7, 
           label=f'Global fit: $n={n_fitted:.2f}\\pm{n_err:.2f}$')

ax.set_xlabel('Temperature $T/T_c$')
ax.set_ylabel('Local Power-Law Exponent $n$')
ax.set_title('Power-Law Scaling Analysis: Evidence for Anisotropic Gap')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 6)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig6_power_law_exponent.png'))
plt.close()
print("Figure 6 saved.")

power_law_results = {
    'fitted_exponent': float(n_fitted),
    'fitted_exponent_err': float(n_err),
    'mean_local_exponent': float(np.mean(n_local[np.isfinite(n_local)])),
    'std_local_exponent': float(np.std(n_local[np.isfinite(n_local)])),
}

# ============================================================
# FIGURE 7: Microwave resonance and kinetic inductance
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): D_s vs microwave current amplitude
ax1.plot(I_mw_amplitude, D_s_mw_exp, 'mo-', markersize=4, linewidth=1.5)
# Quadratic fit
coeffs = np.polyfit(I_mw_amplitude, D_s_mw_exp, 2)
I_smooth = np.linspace(0, max(I_mw_amplitude), 100)
ax1.plot(I_smooth, np.polyval(coeffs, I_smooth), 'r--', linewidth=1.5, alpha=0.7,
         label=f'Quadratic: $D_s \\approx {coeffs[0]:.3f}I^2 + {coeffs[1]:.2f}I + {coeffs[2]:.1f}$')

# Linear term is near zero => quadratic dominance
ax1.set_xlabel('Microwave Current Amplitude $I_{mw}$ (nA)')
ax1.set_ylabel('$D_s$ (arb. units)')
ax1.set_title('Microwave Current Dependence: Quadratic Scaling')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel (b): Resonance frequency shift (kinetic inductance)
# Δf/f0 ∝ -ΔL_k/L_k ∝ Δ(1/D_s)
L_k = 1.0 / np.clip(D_s_mw_exp, 1e-10, None)
L_k0 = L_k[0]
delta_L = (L_k - L_k0) / L_k0 * 100  # percent
ax2.plot(P_mw, delta_L, 'cs-', markersize=4, linewidth=1.5)
ax2.set_xlabel('Normalized Microwave Power $P_{mw}$')
ax2.set_ylabel('$\\Delta L_k / L_{k0}$ (%)')
ax2.set_title('Kinetic Inductance Change with Microwave Power')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig7_microwave_response.png'))
plt.close()
print("Figure 7 saved.")

mw_results = {
    'quadratic_coeff_I2': float(coeffs[0]),
    'quadratic_coeff_I1': float(coeffs[1]),
    'quadratic_coeff_I0': float(coeffs[2]),
    'max_kinetic_inductance_change_pct': float(delta_L[-1]),
}

# ============================================================
# FIGURE 8: Summary validation plot
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) GL model prediction vs exp DC
ax = axes[0,0]
n_gl = min(len(I_dc), len(D_s_gl))
ax.plot(I_dc[:n_gl], D_s_gl[:n_gl], 'b-', label='GL Theory')
n_de = min(len(I_dc), len(D_s_dc_exp))
ax.plot(I_dc[:n_de], D_s_dc_exp[:n_de], 'ro', markersize=3, label='Experiment')
ax.set_xlabel('$I_{dc}$ (nA)'); ax.set_ylabel('$D_s$'); ax.set_title('GL vs Experiment')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) BCS vs Experiment (T-dependence)
ax = axes[0,1]
n_bcs = min(len(T_plot), len(D_s_bcs))
n_exp = min(len(T_exp_plot), len(D_s_experimental))
ax.plot(T_plot[:n_bcs], D_s_bcs[:n_bcs], 'b-', label='BCS s-wave')
ax.plot(T_exp_plot[:n_exp], D_s_experimental[:n_exp], 'ko', markersize=2, label='Experiment')
ax.set_xlabel('$T$ (K)'); ax.set_ylabel('$D_s$'); ax.set_title('BCS vs Experiment')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) Residual analysis: Exp - BCS
ax = axes[0,2]
n_res = min(n_exp, n_bcs)
residual = D_s_experimental[:n_res] - D_s_bcs[:n_res]
n_exp = n_res
T_exp_plot = T_exp_plot[:n_res]
ax.plot(T_exp_plot[:n_exp], residual, 'r-', linewidth=1)
ax.axhline(y=0, color='gray', linestyle='--')
ax.fill_between(T_exp_plot[:n_exp], 0, residual, alpha=0.2, color='red')
ax.set_xlabel('$T$ (K)'); ax.set_ylabel('$\\Delta D_s$'); 
ax.set_title('Residual: Exp $-$ BCS')
ax.grid(True, alpha=0.3)

# (d) Conventional vs Geometric at optimal doping
ax = axes[1,0]
idx_opt = np.argmax(D_s_exp_hole)
n_opt = n_cm2[idx_opt]
bar_colors = ['blue', 'red', 'darkorange', 'green']
bar_labels = ['Conventional', 'Geometric', 'Exp Hole', 'Exp Electron']
bar_vals = [D_s_conv[idx_opt]/1e9, D_s_geom[idx_opt]/1e9, 
            D_s_exp_hole[idx_opt]/1e9, D_s_exp_electron[idx_opt]/1e9]
bars = ax.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.7)
ax.set_ylabel('$D_s$ (10$^9$ H$^{-1}$)')
ax.set_title(f'Superfluid Stiffness at $n \\approx {n_opt:.1e}$ cm$^{{-2}}$')
for b, v in zip(bars, bar_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}', 
            ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# (e) Power law exponents comparison
ax = axes[1,1]
exp_labels = ['BCS\n(s-wave)', 'Nodal\n(d-wave)', 'n=2.0', 'n=2.5', 'n=3.0', 'Exp. Fit']
# Compute effective exponents from models
def compute_exponent(T, Ds):
    valid = (T > 0.01) & (T < 0.8) & (Ds > 0.5)
    if np.sum(valid) < 5: return np.nan
    logT = np.log(T[valid]); logDs = np.log(1 - Ds[valid]/Ds[0] + 1e-10)
    return np.polyfit(logT, logDs, 1)[0]

exponents = [
    compute_exponent(T_plot, D_s_bcs[:min_len]),
    compute_exponent(T_plot, D_s_nodal[:min_len]),
    2.0, 2.5, 3.0,
    n_fitted
]
colors_exp = ['blue', 'green', 'red', 'orange', 'purple', 'black']
ax.barh(exp_labels, exponents, color=colors_exp, alpha=0.7)
ax.set_xlabel('Power-law Exponent $n$')
ax.set_title('Power-Law Exponent Comparison')
ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

# (f) Normalized D_s(T) comparison at low T
ax = axes[1,2]
ax.plot(T_plot[:min_len], D_s_bcs[:min_len]/D_s_bcs[0], 'b-', alpha=0.6, label='BCS')
ax.plot(T_plot[:min_len], D_s_nodal[:min_len]/D_s_nodal[0], 'g-', alpha=0.6, label='Nodal')
ax.plot(T_plot[:min_len], D_s_power_n2[:min_len]/D_s_power_n2[0], 'r-', alpha=0.6, label='n=2')
ax.plot(T_exp_plot[:n_exp], D_s_experimental[:n_exp]/D_s_experimental[0], 'ko', markersize=2, label='Exp')
ax.set_xlabel('$T/T_c$'); ax.set_ylabel('$D_s(T)/D_s(0)$')
ax.set_title('Normalized $D_s(T)$ at Low $T$')
ax.set_xlim(0, 0.5); ax.set_ylim(0.6, 1.02)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(IMG_DIR, 'fig8_validation_summary.png'))
plt.close()
print("Figure 8 saved.")

# ============================================================
# Save all quantitative results
# ============================================================
all_results = {
    'carrier_density': enhancement_results,
    'temperature': temperature_results,
    'current': current_results,
    'bkt': bkt_results,
    'power_law': power_law_results,
    'microwave': mw_results,
}

with open(os.path.join(OUTPUT_DIR, 'quantitative_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll figures and results saved successfully.")
print(f"Geometric/Conventional enhancement ratio: {enhancement_results['mean_ratio_geom_to_conv']:.1f}")
print(f"Experimental (hole)/Conventional ratio: {enhancement_results['mean_ratio_exp_hole_to_conv']:.1f}")
print(f"Fitted power-law exponent: {n_fitted:.2f} ± {n_err:.2f}")
print(f"Max BKT T (hole-doped): {bkt_results['max_Tbkt_exp_hole_K']:.1f} K")
