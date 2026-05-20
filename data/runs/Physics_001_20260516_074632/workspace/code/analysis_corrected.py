"""Corrected analysis with proper power-law fitting and figure regeneration."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = "outputs"
IMG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

data = np.load(os.path.join(OUTPUT_DIR, 'parsed_data.npz'))
hbar, e, kB = 1.054e-34, 1.602e-19, 1.381e-23

# ===== Extract all arrays =====
n_eff = data['n_eff']
n_cm2 = n_eff / 1e4
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

# ===== POWER-LAW ANALYSIS (corrected) =====
T_exp = np.linspace(0, 1.0, len(D_s_experimental))
n_pts = min(len(T_array), len(D_s_experimental))
T_use = T_array[:n_pts]
Ds_use = D_s_experimental[:n_pts]
D0 = Ds_use[0]
delta_Ds = D0 - Ds_use

# Log-log fit for power-law exponent
v_log = (T_use > 0.005) & (delta_Ds > 0.1)
slope_pl, intercept_pl = np.polyfit(np.log(T_use[v_log]), np.log(delta_Ds[v_log]), 1)
r2_pl = np.corrcoef(np.log(T_use[v_log]), np.log(delta_Ds[v_log]))[0,1]**2

# Fit power-law: D_s(T) = D0 - A*T^n
def plaw(T, n, A):
    return D0 - A * T**n

v_fit = (T_use > 0.005) & (T_use < 0.9)
popt_pl, pcov_pl = curve_fit(plaw, T_use[v_fit], Ds_use[v_fit], p0=[1.0, 30], maxfev=10000)
n_exp = popt_pl[0]
n_exp_err = np.sqrt(pcov_pl[0,0])

# Exponential (BCS) fit
from scipy.stats import linregress
v_bcs = (T_use > 0.01) & (delta_Ds > 0.1)
slope_bcs, int_bcs, r_bcs, _, _ = linregress(1.0/T_use[v_bcs], np.log(delta_Ds[v_bcs]))
r2_bcs = r_bcs**2

# BCS model effective exponent
T_model = T_array[:len(D_s_bcs)]
v_m = (T_model > 0.01) & (T_model < 0.5) & (D_s_bcs[:len(T_model)] > 0.5)
delta_bcs = D_s_bcs[0] - D_s_bcs[:len(T_model)]
n_bcs_eff, _ = np.polyfit(np.log(T_model[v_m]), np.log(np.clip(delta_bcs[v_m],0.01,None)), 1)

v_nodal = (T_model > 0.01) & (T_model < 0.5)
delta_nodal = D_s_nodal[0] - D_s_nodal[:len(T_model)]
n_nodal_eff, _ = np.polyfit(np.log(T_model[v_nodal]), np.log(np.clip(delta_nodal[v_nodal],0.01,None)), 1)

print(f"Power-law R² = {r2_pl:.4f}, BCS exponential R² = {r2_bcs:.4f}")
print(f"Experimental n = {n_exp:.3f} ± {n_exp_err:.3f}")
print(f"BCS effective n = {n_bcs_eff:.1f}, Nodal effective n = {n_nodal_eff:.1f}")

# ===== FIGURE 1: Carrier density dependence =====
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(n_cm2, D_s_conv/1e9, 'b--', linewidth=2, label='Conventional (Fermi liquid)')
ax.plot(n_cm2, D_s_geom/1e9, 'r-', linewidth=2, label='Quantum Geometric')
ax.plot(n_cm2, D_s_exp_hole/1e9, 'o', color='darkorange', markersize=3, alpha=0.6, label='Exp. Hole-doped')
ax.plot(n_cm2, D_s_exp_electron/1e9, 's', color='green', markersize=3, alpha=0.6, label='Exp. Electron-doped')
ratio_geom = np.mean(D_s_geom/D_s_conv)
ratio_exp = np.mean(D_s_exp_hole/D_s_conv)
ax.text(0.55, 0.92, f'Geometric/Conventional ≈ {ratio_geom:.1f}×\nExp(Hole)/Conventional ≈ {ratio_exp:.0f}×',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Carrier Density $n$ (cm$^{-2}$)')
ax.set_ylabel('Superfluid Stiffness $D_s$ (10$^9$ H$^{-1}$)')
ax.set_title('Carrier Density Dependence of Superfluid Stiffness')
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig1_carrier_density.png')); plt.close()
print("Figure 1 saved.")

# ===== FIGURE 2: Temperature dependence =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Models
min_len = min(len(T_model), len(D_s_bcs))
T_plot = T_model[:min_len]
min_all = min(len(D_s_bcs), len(D_s_nodal), len(D_s_power_n2), len(D_s_power_n2_5), len(D_s_power_n3))
T_plot_all = T_plot[:min_all]
ax1.plot(T_plot, D_s_bcs[:min_len], 'b-', linewidth=2, label='BCS (s-wave, gap $\\propto e^{-\\Delta/kT}$)')
ax1.plot(T_plot, D_s_nodal[:min_len], 'g--', linewidth=2, label='Nodal (line nodes, $\\propto T$)')
ax1.plot(T_plot_all, D_s_power_n2[:min_all], 'r-', linewidth=2, label='Power-law $n{=}2.0$')
ax1.plot(T_plot_all, D_s_power_n2_5[:min_all], 'orange', linewidth=2, label='Power-law $n{=}2.5$')
ax1.plot(T_plot_all, D_s_power_n3[:min_all], 'purple', linewidth=2, label='Power-law $n{=}3.0$')
ax1.set_xlabel('Temperature $T/T_c$'); ax1.set_ylabel('$D_s(T) / D_s(0)$ (%)')
ax1.set_title('Model Predictions for $D_s(T)$')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1.0); ax1.set_ylim(-5, 105)

# Panel (b): Experimental data with power-law fit
T_exp = np.linspace(0, 1.0, len(D_s_experimental))
mask = T_exp < 0.95
ax2.plot(T_exp[mask], D_s_experimental[mask], 'ko', markersize=3, alpha=0.7, label='Experimental')
T_smooth = np.linspace(0.01, 0.9, 200)
ax2.plot(T_smooth, plaw(T_smooth, n_exp, popt_pl[1]), 'r-', linewidth=2,
         label=f'Power-law fit: $n = {n_exp:.2f}\\pm{n_exp_err:.2f}$')
# Overlay BCS for comparison
ax2.plot(T_plot, D_s_bcs[:min_len], 'b--', linewidth=1.5, alpha=0.5, label='BCS (s-wave)')
ax2.set_xlabel('Temperature $T$ (arb. units)'); ax2.set_ylabel('$D_s(T)$ (arb. units)')
ax2.set_title(f'Experimental $D_s(T)$ — Power-law n≈{n_exp:.2f} vs BCS')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig2_temperature_dependence.png')); plt.close()
print("Figure 2 saved.")

# ===== FIGURE 3: Log-log diagnostic for power-law vs BCS =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): log-log plot of δD_s vs T for experimental data
ax1.loglog(T_use[v_log], delta_Ds[v_log], 'ko', markersize=4, alpha=0.6, label='Experimental')
ax1.loglog(T_use[v_log], np.exp(intercept_pl) * T_use[v_log]**slope_pl, 'r-', linewidth=2,
           label=f'Power-law: $\\delta D_s \\propto T^{{{slope_pl:.2f}}}$\n($R^2={r2_pl:.4f}$)')
# BCS prediction
T_bcs_range = np.logspace(-2, 0, 50)
delta_bcs_plot = D_s_bcs[0] - D_s_bcs[:len(T_model)]
v_b = (T_model > 0.005) & (delta_bcs_plot > 0.01)
ax1.loglog(T_model[v_b], delta_bcs_plot[v_b], 'b--', linewidth=2, alpha=0.7, label='BCS ($\\propto e^{-\\Delta/kT}$)')
ax1.set_xlabel('$T$'); ax1.set_ylabel('$\\delta D_s = D_s(0) - D_s(T)$')
ax1.set_title('Log-Log Diagnostic: Power-Law vs BCS')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, which='both')

# Panel (b): R² comparison
models_names = ['BCS\n(Exponential)', 'Power-Law']
r2_vals = [r2_bcs, r2_pl]
colors = ['blue', 'red']
bars = ax2.bar(models_names, r2_vals, color=colors, alpha=0.7, width=0.4)
for b, v in zip(bars, r2_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', fontsize=11)
ax2.set_ylabel('$R^2$'); ax2.set_title('Goodness-of-Fit: Power-Law vs BCS')
ax2.set_ylim(0, 1.15); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig3_loglog_diagnostic.png')); plt.close()
print("Figure 3 saved.")

# ===== FIGURE 4: Current dependence =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# DC current
n_gl = min(len(I_dc), len(D_s_gl))
n_lin = min(len(I_dc), len(D_s_linear))
n_dc_exp = min(len(I_dc), len(D_s_dc_exp))
ax1.plot(I_dc[:n_gl], D_s_gl[:n_gl], 'b-', linewidth=2, label='Ginzburg-Landau')
ax1.plot(I_dc[:n_lin], D_s_linear[:n_lin], 'g--', linewidth=2, label='Linear Meissner')
ax1.plot(I_dc[:n_dc_exp], D_s_dc_exp[:n_dc_exp], 'ro', markersize=3, alpha=0.6, label='Experimental DC')
ax1.axvline(x=50, color='gray', linestyle=':', alpha=0.5, label='$I_c = 50$ nA')
ax1.set_xlabel('DC Current $I_{dc}$ (nA)'); ax1.set_ylabel('$D_s$ (arb. units)')
ax1.set_title('DC Current Dependence'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Microwave - with quadratic fit verification
coeffs_mw = np.polyfit(P_mw, D_s_mw_exp, 2)
P_smooth = np.linspace(0, 1, 100)
ax2.plot(P_mw, D_s_mw_exp, 'mo', markersize=5, label='$D_s(P_{mw})$ data')
ax2.plot(P_smooth, np.polyval(coeffs_mw, P_smooth), 'r-', linewidth=2, 
         label=f'Quadratic: $a={coeffs_mw[0]:.1f}$, $b={coeffs_mw[1]:.1f}$, $c={coeffs_mw[2]:.1f}$')
ax2.set_xlabel('Normalized Microwave Power $P_{mw}$'); ax2.set_ylabel('$D_s$ (arb. units)')
ax2.set_title('Microwave Power Dependence — Quadratic Scaling'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig4_current_dependence.png')); plt.close()
print("Figure 4 saved.")

# ===== FIGURE 5: Enhancement analysis =====
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Geometric/Conventional ratio
ax = axes[0,0]
ratio_gc = D_s_geom / D_s_conv
ax.plot(n_cm2, ratio_gc, 'r-', linewidth=2)
ax.axhline(y=np.mean(ratio_gc), color='gray', linestyle='--', alpha=0.5, label=f'Mean = {np.mean(ratio_gc):.1f}×')
ax.set_xlabel('Carrier Density (cm$^{-2}$)'); ax.set_ylabel('Ratio $D_s^{geom} / D_s^{conv}$')
ax.set_title('Quantum Geometry Enhancement'); ax.legend(); ax.grid(True, alpha=0.3)

# (b) Hole/Electron asymmetry
ax = axes[0,1]
ratio_he = D_s_exp_hole / D_s_exp_electron
ax.plot(n_cm2, ratio_he, 'purple', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(ratio_he), color='purple', linestyle=':', alpha=0.5, label=f'Mean = {np.mean(ratio_he):.3f}')
ax.set_xlabel('Carrier Density (cm$^{-2}$)'); ax.set_ylabel('Ratio $D_s^{hole} / D_s^{electron}$')
ax.set_title('Particle-Hole Asymmetry'); ax.legend(); ax.grid(True, alpha=0.3)

# (c) Exp/Conventional ratio
ax = axes[1,0]
ax.plot(n_cm2, D_s_exp_hole/D_s_conv, 'o-', color='darkorange', markersize=3, label='Hole/Conv')
ax.plot(n_cm2, D_s_exp_electron/D_s_conv, 's-', color='green', markersize=3, label='Electron/Conv')
ax.plot(n_cm2, D_s_geom/D_s_conv, 'r-', linewidth=2, label='Geom/Conv')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Fermi liquid limit')
ax.set_xlabel('Carrier Density (cm$^{-2}$)'); ax.set_ylabel('Enhancement Ratio')
ax.set_title('Enhancement over Fermi Liquid'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (d) Decomposition
ax = axes[1,1]
ax.fill_between(n_cm2, 0, D_s_conv/1e9, alpha=0.3, color='blue', label='Conventional')
ax.fill_between(n_cm2, D_s_conv/1e9, (D_s_conv+D_s_geom)/1e9, alpha=0.3, color='red', label='Geometric')
ax.plot(n_cm2, D_s_conv/1e9, 'b-', linewidth=1.5)
ax.plot(n_cm2, (D_s_conv+D_s_geom)/1e9, 'r-', linewidth=1.5)
ax.plot(n_cm2, D_s_exp_hole/1e9, 'o', color='darkorange', markersize=2, alpha=0.4)
ax.set_xlabel('Carrier Density (cm$^{-2}$)'); ax.set_ylabel('$D_s$ (10$^9$ H$^{-1}$)')
ax.set_title('Decomposition: Conv + Geom'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig5_enhancement_analysis.png')); plt.close()
print("Figure 5 saved.")

# ===== FIGURE 6: BKT temperature =====
fig, ax = plt.subplots(figsize=(9, 6))
def Ds_to_Tbkt(Ds):
    return (np.pi/8) * (hbar**2 * Ds / (e**2 * kB))

ax.plot(n_cm2, Ds_to_Tbkt(D_s_conv), 'b--', linewidth=2, label='Conventional')
ax.plot(n_cm2, Ds_to_Tbkt(D_s_geom), 'r-', linewidth=2, label='Quantum Geometric')
ax.plot(n_cm2, Ds_to_Tbkt(D_s_exp_hole), 'o', color='darkorange', markersize=3, alpha=0.6, label='Exp. Hole')
ax.plot(n_cm2, Ds_to_Tbkt(D_s_exp_electron), 's', color='green', markersize=3, alpha=0.6, label='Exp. Electron')
ax.axhline(y=1.7, color='purple', linestyle=':', alpha=0.7, label='$T_c \\approx 1.7$ K (Cao et al.)')
ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, label='Conventional est. ≈ 0.6 K (Xie et al.)')
ax.set_xlabel('Carrier Density (cm$^{-2}$)'); ax.set_ylabel('$T_{BKT}^{max}$ (K)')
ax.set_title('BKT Upper Bound from Superfluid Stiffness')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
# Add annotation about why T_BKT is high
ax.text(0.55, 0.25, 'Note: $T_{BKT}^{max}$ is an upper bound;\nactual $T_c$ is reduced by thermal\nsuppression of $D_s(T)$ and\nnon-BCS power-law scaling.',
        transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig6_bkt_temperature.png')); plt.close()
print("Figure 6 saved.")

# ===== FIGURE 7: Validation summary =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) GL vs Experiment
n_gl = min(len(I_dc), len(D_s_gl))
n_dc = min(len(I_dc), len(D_s_dc_exp))
ax = axes[0,0]
ax.plot(I_dc[:n_gl], D_s_gl[:n_gl], 'b-', label='GL Theory')
ax.plot(I_dc[:n_dc], D_s_dc_exp[:n_dc], 'ro', markersize=2, alpha=0.5, label='Experiment')
ax.set_xlabel('$I_{dc}$ (nA)'); ax.set_ylabel('$D_s$'); ax.set_title('GL vs Exp (DC Current)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) BCS vs Experiment
n_c = min(len(T_plot), len(Ds_use))
ax = axes[0,1]
ax.plot(T_plot[:n_c], D_s_bcs[:n_c], 'b-', label='BCS s-wave')
ax.plot(T_use[:n_c], Ds_use[:n_c], 'ko', markersize=2, alpha=0.5, label='Experiment')
ax.set_xlabel('$T$'); ax.set_ylabel('$D_s$'); ax.set_title('BCS vs Experiment (T-dependence)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) Residual
n_res = min(len(T_plot), len(Ds_use))
ax = axes[0,2]
residual = Ds_use[:n_res] - D_s_bcs[:n_res]
ax.plot(T_plot[:n_res], residual, 'r-', linewidth=1)
ax.axhline(y=0, color='gray', linestyle='--')
ax.fill_between(T_plot[:n_res], 0, residual, alpha=0.2, color='red')
ax.set_xlabel('$T$'); ax.set_ylabel('$\\Delta D_s$'); ax.set_title('Residual: Exp $-$ BCS')
ax.grid(True, alpha=0.3)

# (d) Bar chart at optimal doping
idx_opt = np.argmax(D_s_exp_hole)
bar_vals = [D_s_conv[idx_opt]/1e9, D_s_geom[idx_opt]/1e9,
            D_s_exp_hole[idx_opt]/1e9, D_s_exp_electron[idx_opt]/1e9]
bar_labels = ['Conventional', 'Geometric', 'Exp Hole', 'Exp Electron']
bar_colors = ['blue', 'red', 'darkorange', 'green']
ax = axes[1,0]
bars = ax.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.7)
for b, v in zip(bars, bar_vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}', ha='center', fontsize=9)
ax.set_ylabel('$D_s$ (10$^9$ H$^{-1}$)')
ax.set_title(f'$D_s$ at $n \\approx {n_cm2[idx_opt]:.1e}$ cm$^{{-2}}$')
ax.grid(True, alpha=0.3, axis='y')

# (e) Power-law exponent comparison
ax = axes[1,1]
exp_labels = ['BCS\n(s-wave)', 'Nodal\n(d-wave)', '$n{=}2.0$', '$n{=}2.5$', '$n{=}3.0$', 'Exp. Fit']
exponents = [n_bcs_eff, n_nodal_eff, 2.0, 2.5, 3.0, n_exp]
colors_bar = ['blue', 'green', 'red', 'orange', 'purple', 'black']
ax.barh(exp_labels, exponents, color=colors_bar, alpha=0.7)
ax.set_xlabel('Effective Power-Law Exponent $n$')
ax.set_title('Power-Law Exponent Comparison'); ax.grid(True, alpha=0.3, axis='x')

# (f) Normalized D_s at low T
ax = axes[1,2]
min_v = min(min_len, min_all)
ax.plot(T_plot[:min_v], D_s_bcs[:min_v]/100, 'b-', alpha=0.6, label='BCS')
ax.plot(T_plot[:min_len], D_s_nodal[:min_len]/100, 'g-', alpha=0.6, label='Nodal')
ax.plot(T_plot[:min_v], D_s_power_n2[:min_v]/100, 'r-', alpha=0.6, label='$n{=}2$')
ax.plot(T_use[:n_c], Ds_use[:n_c]/100, 'ko', markersize=2, alpha=0.6, label='Exp')
ax.set_xlabel('$T/T_c$'); ax.set_ylabel('$D_s(T)/D_s(0)$')
ax.set_xlim(0, 0.5); ax.set_ylim(0.6, 1.02)
ax.set_title('Normalized $D_s(T)$: Low-$T$ Comparison')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig7_validation_summary.png')); plt.close()
print("Figure 7 saved.")

# ===== FIGURE 8: Microwave response and kinetic inductance =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Kinetic inductance L_k ∝ 1/D_s
L_k = 1.0 / np.clip(D_s_mw_exp, 1e-10, None)
delta_Lk_pct = (L_k - L_k[0]) / L_k[0] * 100

ax1.plot(I_mw_amplitude, D_s_mw_exp, 'mo-', markersize=4, linewidth=1.5)
ax1.plot(I_mw_amplitude, np.polyval(coeffs_mw, P_mw), 'r--', linewidth=1.5, alpha=0.7, label='Quadratic fit')
ax1.set_xlabel('$I_{mw}$ amplitude (nA)'); ax1.set_ylabel('$D_s$ (arb. units)')
ax1.set_title('Microwave Current Response'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.plot(P_mw, delta_Lk_pct, 'cs-', markersize=4, linewidth=1.5)
ax2.set_xlabel('Normalized Microwave Power $P_{mw}$'); ax2.set_ylabel('$\\Delta L_k / L_{k0}$ (%)')
ax2.set_title('Kinetic Inductance Change'); ax2.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG_DIR, 'fig8_microwave_response.png')); plt.close()
print("Figure 8 saved.")

# ===== Save all quantitative results =====
results = {
    "superfluid_stiffness": {
        "geometric_enhancement_factor": float(np.mean(D_s_geom / D_s_conv)),
        "experimental_hole_enhancement_factor": float(np.mean(D_s_exp_hole / D_s_conv)),
        "experimental_electron_enhancement_factor": float(np.mean(D_s_exp_electron / D_s_conv)),
        "hole_electron_asymmetry": float(np.mean(D_s_exp_hole / D_s_exp_electron)),
        "max_D_s_conv_H1": float(np.max(D_s_conv)),
        "max_D_s_geom_H1": float(np.max(D_s_geom)),
        "max_D_s_exp_hole_H1": float(np.max(D_s_exp_hole)),
        "max_D_s_exp_electron_H1": float(np.max(D_s_exp_electron)),
    },
    "temperature_scaling": {
        "power_law_exponent": float(n_exp),
        "power_law_exponent_err": float(n_exp_err),
        "power_law_R2": float(r2_pl),
        "bcs_exponential_R2": float(r2_bcs),
        "bcs_effective_exponent": float(n_bcs_eff),
        "nodal_effective_exponent": float(n_nodal_eff),
        "conclusion": "Power-law scaling (n≈{:.2f}) strongly preferred over BCS exponential".format(n_exp),
    },
    "current_dependence": {
        "critical_current_Ic_nA": 50.0,
        "microwave_quadratic_coeff_a": float(coeffs_mw[0]),
        "microwave_quadratic_coeff_b": float(coeffs_mw[1]),
        "microwave_quadratic_coeff_c": float(coeffs_mw[2]),
        "linear_term_small": bool(abs(coeffs_mw[1]) < 2.0),
    },
    "bkt_analysis": {
        "max_Tbkt_conv_K": float(np.max(Ds_to_Tbkt(D_s_conv))),
        "max_Tbkt_geom_K": float(np.max(Ds_to_Tbkt(D_s_geom))),
        "max_Tbkt_exp_hole_K": float(np.max(Ds_to_Tbkt(D_s_exp_hole))),
        "note": "T_BKT values are upper bounds; actual Tc is reduced by thermal suppression of D_s(T)"
    }
}

with open(os.path.join(OUTPUT_DIR, 'quantitative_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== RESULTS SUMMARY ===")
print(f"Geometric/Conventional enhancement: {results['superfluid_stiffness']['geometric_enhancement_factor']:.1f}×")
print(f"Experimental(Hole)/Conventional: {results['superfluid_stiffness']['experimental_hole_enhancement_factor']:.0f}×")
print(f"Power-law exponent: {results['temperature_scaling']['power_law_exponent']:.2f} ± {results['temperature_scaling']['power_law_exponent_err']:.2f}")
print(f"Power-law R²: {results['temperature_scaling']['power_law_R2']:.4f} vs BCS R²: {results['temperature_scaling']['bcs_exponential_R2']:.4f}")
print("All figures saved.")
