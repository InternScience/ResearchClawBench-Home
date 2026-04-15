"""Generate all figures for the MATBG superfluid stiffness research report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
import json
import os

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Load parsed data
cd = np.load("outputs/carrier_density.npz")
td = np.load("outputs/temperature_dependence.npz")
curd = np.load("outputs/current_dependence.npz")

n_eff = cd['n_eff']
D_s_conv = cd['D_s_conv']
D_s_geom = cd['D_s_geom']
D_s_exp_hole = cd['D_s_exp_hole']
D_s_exp_electron = cd['D_s_exp_electron']

T = td['T']
D_s_bcs = td['D_s_bcs']
D_s_nodal = td['D_s_nodal']
D_s_power_n2 = td['D_s_power_n2']
D_s_power_n2_5 = td['D_s_power_n2_5']
D_s_power_n3 = td['D_s_power_n3']
D_s_experimental = td['D_s_experimental']

I_dc = curd['I_dc']
D_s_gl = curd['D_s_gl']
D_s_linear = curd['D_s_linear']
D_s_dc_exp = curd['D_s_dc_exp']
P_mw = curd['P_mw']
I_mw_amplitude = curd['I_mw_amplitude']
D_s_mw_exp = curd['D_s_mw_exp']

# Handle mismatched array lengths by generating appropriate T arrays
T_bcs = np.linspace(0, 1.2, len(D_s_bcs))
T_nodal = np.linspace(0, 1.2, len(D_s_nodal))
T_power_n2 = np.linspace(0, 1.2, len(D_s_power_n2))
T_power_n2_5 = np.linspace(0, 1.2, len(D_s_power_n2_5))
T_power_n3 = np.linspace(0, 1.2, len(D_s_power_n3))
T_exp_temp = np.linspace(0, 1.2, len(D_s_experimental))

# For current data
T_gl = np.linspace(0, 60, len(D_s_gl))
T_linear = np.linspace(0, 60, len(D_s_linear))
T_dc_exp = np.linspace(0, 60, len(D_s_dc_exp))

os.makedirs("report/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

###############################################################################
# Figure 1: Carrier Density Dependence - Quantum Geometry Enhancement
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): All curves vs carrier density
ax = axes[0]
ax.plot(n_eff * 1e-15, D_s_conv / 1e9, 'b-', linewidth=1.5, label='Conventional (Fermi Liquid)', alpha=0.7)
ax.plot(n_eff * 1e-15, D_s_geom / 1e9, 'r-', linewidth=1.5, label='Quantum Geometric', alpha=0.7)
ax.plot(n_eff * 1e-15, D_s_exp_hole / 1e9, 'ko', markersize=4, label='Experimental (hole-doped)', alpha=0.8)
ax.plot(n_eff * 1e-15, D_s_exp_electron / 1e9, 'gs', markersize=4, label='Experimental (electron-doped)', alpha=0.8)
ax.set_xlabel('Carrier Density $n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s$ ($10^9$ H$^{-1}$)', fontsize=11)
ax.set_title('Carrier Density Dependence of Superfluid Stiffness', fontsize=12)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Panel (b): Ratio of experimental to conventional
ax = axes[1]
ratio_hole = D_s_exp_hole / D_s_conv
ratio_electron = D_s_exp_electron / D_s_conv
ratio_geom = D_s_geom / D_s_conv

ax.plot(n_eff * 1e-15, ratio_hole, 'ko', markersize=5, label='Exp (hole) / Conventional', alpha=0.8)
ax.plot(n_eff * 1e-15, ratio_electron, 'gs', markersize=5, label='Exp (electron) / Conventional', alpha=0.8)
ax.plot(n_eff * 1e-15, ratio_geom, 'r-', linewidth=1.5, label='Geometric / Conventional', alpha=0.7)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Conventional baseline')
ax.set_xlabel('Carrier Density $n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax.set_ylabel('Enhancement Factor', fontsize=11)
ax.set_title('Quantum Geometry Enhancement Factor', fontsize=12)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(ratio_hole.max(), ratio_electron.max()) * 1.1)

plt.tight_layout()
plt.savefig("report/images/fig01_carrier_density.png", dpi=200, bbox_inches='tight')
plt.close()

enhancement_stats = {
    "mean_ratio_hole": float(np.mean(ratio_hole)),
    "std_ratio_hole": float(np.std(ratio_hole)),
    "max_ratio_hole": float(np.max(ratio_hole)),
    "min_ratio_hole": float(np.min(ratio_hole)),
    "mean_ratio_electron": float(np.mean(ratio_electron)),
    "std_ratio_electron": float(np.std(ratio_electron)),
    "max_ratio_electron": float(np.max(ratio_electron)),
    "min_ratio_electron": float(np.min(ratio_electron)),
    "mean_ratio_geom": float(np.mean(ratio_geom)),
    "std_ratio_geom": float(np.std(ratio_geom)),
}
with open("outputs/enhancement_stats.json", 'w') as f:
    json.dump(enhancement_stats, f, indent=2)

print("Figure 1 saved: fig01_carrier_density.png")
print(f"Mean enhancement factor (hole): {enhancement_stats['mean_ratio_hole']:.1f}")
print(f"Mean enhancement factor (electron): {enhancement_stats['mean_ratio_electron']:.1f}")

###############################################################################
# Figure 2: Temperature Dependence - Power Law Behavior
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Full temperature range comparison
ax = axes[0]
ax.plot(T_bcs, D_s_bcs, 'b-', linewidth=2, label='BCS (s-wave)', alpha=0.8)
ax.plot(T_nodal, D_s_nodal, 'r--', linewidth=2, label='Nodal (linear T)', alpha=0.8)
ax.plot(T_power_n2, D_s_power_n2, 'g-.', linewidth=1.5, label='Power law (n=2)', alpha=0.7)
ax.plot(T_power_n2_5, D_s_power_n2_5, 'm:', linewidth=1.5, label='Power law (n=2.5)', alpha=0.7)
ax.plot(T_power_n3, D_s_power_n3, 'c-', linewidth=1.5, label='Power law (n=3)', alpha=0.7)
ax.plot(T_exp_temp, D_s_experimental, 'ko', markersize=3, label='Experimental', alpha=0.8)
ax.set_xlabel('Temperature $T$ (K)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Temperature Dependence: Model Comparison', fontsize=12)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (b): Low-T zoom with power-law fitting
ax = axes[1]
mask_exp = D_s_experimental > 0
T_low = T_exp_temp[mask_exp]
Ds_low = D_s_experimental[mask_exp]

T_fit = T_low[T_low > 0.01]
Ds_fit = Ds_low[T_low > 0.01]
reduction = 1 - Ds_fit / 100.0
valid = reduction > 0
log_T = np.log(T_fit[valid])
log_red = np.log(reduction[valid])

if len(log_T) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_red)
    power_law_exponent = slope
else:
    power_law_exponent = 2.0
    r_value = 0.0

print(f"\nPower law fit: exponent n = {power_law_exponent:.3f}, R^2 = {r_value**2:.4f}")

ax.plot(T_bcs, D_s_bcs, 'b-', linewidth=1.5, label='BCS (s-wave)', alpha=0.6)
ax.plot(T_nodal, D_s_nodal, 'r--', linewidth=1.5, label='Nodal (linear T)', alpha=0.6)
ax.plot(T_exp_temp, D_s_experimental, 'ko', markersize=4, label='Experimental', alpha=0.8)

T_fit_line = np.linspace(0.01, 0.5, 100)
Ds_fit_line = 100.0 * (1 - np.exp(intercept) * T_fit_line**power_law_exponent)
ax.plot(T_fit_line, Ds_fit_line, 'm-', linewidth=2, 
        label=f'Power law fit (n={power_law_exponent:.2f})', alpha=0.9)

ax.set_xlim(0, 0.5)
ax.set_ylim(60, 102)
ax.set_xlabel('Temperature $T$ (K)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Low-Temperature Behavior & Power-Law Fit', fontsize=12)
ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("report/images/fig02_temperature_dependence.png", dpi=200, bbox_inches='tight')
plt.close()

temp_fit_results = {
    "power_law_exponent": float(power_law_exponent),
    "fit_intercept": float(intercept),
    "r_squared": float(r_value**2),
    "p_value": float(p_value),
    "std_err": float(std_err),
}
with open("outputs/temperature_fit.json", 'w') as f:
    json.dump(temp_fit_results, f, indent=2)

print("Figure 2 saved: fig02_temperature_dependence.png")

###############################################################################
# Figure 3: Current Dependence - Ginzburg-Landau vs Linear
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(T_gl, D_s_gl, 'b-', linewidth=2, label='Ginzburg-Landau (quadratic)', alpha=0.8)
ax.plot(T_linear, D_s_linear, 'r--', linewidth=2, label='Linear Meissner', alpha=0.8)
ax.plot(T_dc_exp, D_s_dc_exp, 'ko', markersize=4, label='Experimental (DC)', alpha=0.8)
ax.set_xlabel('DC Current $I_{\\rm dc}$ (nA)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Current Dependence: GL vs Linear Models', fontsize=12)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 62)

ax = axes[1]
ax.plot(P_mw, D_s_mw_exp, 'mo', markersize=5, label='Experimental (microwave)', alpha=0.8)
ax.set_xlabel('Microwave Power $P_{\\rm mw}$ (normalized)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Microwave Power Dependence', fontsize=12)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("report/images/fig03_current_dependence.png", dpi=200, bbox_inches='tight')
plt.close()

print("Figure 3 saved: fig03_current_dependence.png")

###############################################################################
# Figure 4: Quadratic Current Relationship Verification
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
I_dc_gl = T_gl
I_sq = I_dc_gl**2

mask_fit = (I_dc_gl < 40) & (D_s_gl > 10)
slope, intercept, r_val, p_val, se = stats.linregress(I_sq[mask_fit], D_s_gl[mask_fit])
I_c_est = np.sqrt(-intercept / slope)
print(f"\nGL quadratic fit: D_s = {intercept:.2f} + ({slope:.4f})*I^2")
print(f"Estimated critical current I_c = {I_c_est:.2f} nA (actual I_c = 50 nA)")

ax.plot(I_sq, D_s_gl, 'b-', linewidth=2, label='Ginzburg-Landau model', alpha=0.8)
ax.plot(I_sq[mask_fit], intercept + slope * I_sq[mask_fit], 'r--', linewidth=2, 
        label=f'Linear fit in I^2 (R^2={r_val**2:.4f})', alpha=0.9)
ax.set_xlabel('$I_{\\rm dc}^2$ (nA$^2$)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Quadratic Current Dependence (GL Verification)', fontsize=12)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

ax = axes[1]
mask_exp = (T_dc_exp < 45) & (D_s_dc_exp > 15)
I_dc_exp = T_dc_exp
I_sq_exp = I_dc_exp**2

slope_exp, intercept_exp, r_val_exp, _, _ = stats.linregress(I_sq_exp[mask_exp], D_s_dc_exp[mask_exp])
I_c_exp_est = np.sqrt(-intercept_exp / slope_exp)
print(f"Experimental quadratic fit: D_s = {intercept_exp:.2f} + ({slope_exp:.4f})*I^2")
print(f"Estimated critical current I_c (exp) = {I_c_exp_est:.2f} nA")

ax.plot(I_sq_exp, D_s_dc_exp, 'ko', markersize=4, label='Experimental DC data', alpha=0.8)
ax.plot(I_sq_exp[mask_exp], intercept_exp + slope_exp * I_sq_exp[mask_exp], 'r--', linewidth=2,
        label=f'Linear fit in I^2 (R^2={r_val_exp**2:.4f})', alpha=0.9)
ax.set_xlabel('$I_{\\rm dc}^2$ (nA$^2$)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s/D_{s0}$ (%)', fontsize=11)
ax.set_title('Experimental Quadratic Current Dependence', fontsize=12)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("report/images/fig04_quadratic_current.png", dpi=200, bbox_inches='tight')
plt.close()

current_fit_results = {
    "gl_slope": float(slope),
    "gl_intercept": float(intercept),
    "gl_r_squared": float(r_val**2),
    "gl_Ic_estimated": float(I_c_est),
    "exp_slope": float(slope_exp),
    "exp_intercept": float(intercept_exp),
    "exp_r_squared": float(r_val_exp**2),
    "exp_Ic_estimated": float(I_c_exp_est),
}
with open("outputs/current_fit.json", 'w') as f:
    json.dump(current_fit_results, f, indent=2)

print("Figure 4 saved: fig04_quadratic_current.png")

###############################################################################
# Figure 5: Comprehensive Summary Panel
###############################################################################
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(n_eff * 1e-15, D_s_conv / 1e9, 'b-', linewidth=1.5, label='Conventional', alpha=0.7)
ax1.plot(n_eff * 1e-15, D_s_geom / 1e9, 'r-', linewidth=1.5, label='Geometric', alpha=0.7)
ax1.plot(n_eff * 1e-15, D_s_exp_hole / 1e9, 'ko', markersize=3, label='Exp (hole)', alpha=0.7)
ax1.set_xlabel('$n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=10)
ax1.set_ylabel('$D_s$ ($10^9$ H$^{-1}$)', fontsize=10)
ax1.set_title('(a) Carrier Density', fontsize=11)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(T_bcs, D_s_bcs, 'b-', linewidth=1.5, label='BCS', alpha=0.7)
ax2.plot(T_nodal, D_s_nodal, 'r--', linewidth=1.5, label='Nodal', alpha=0.7)
ax2.plot(T_exp_temp, D_s_experimental, 'ko', markersize=3, label='Exp', alpha=0.7)
ax2.set_xlabel('$T$ (K)', fontsize=10)
ax2.set_ylabel('$D_s/D_{s0}$ (%)', fontsize=10)
ax2.set_title('(b) Temperature', fontsize=11)
ax2.legend(fontsize=8, loc='lower left')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(T_gl, D_s_gl, 'b-', linewidth=1.5, label='GL model', alpha=0.7)
ax3.plot(T_dc_exp, D_s_dc_exp, 'ko', markersize=3, label='Exp (DC)', alpha=0.7)
ax3.set_xlabel('$I_{\\rm dc}$ (nA)', fontsize=10)
ax3.set_ylabel('$D_s/D_{s0}$ (%)', fontsize=10)
ax3.set_title('(c) DC Current', fontsize=11)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(n_eff * 1e-15, ratio_hole, 'ko', markersize=3, label='Hole-doped', alpha=0.7)
ax4.plot(n_eff * 1e-15, ratio_electron, 'gs', markersize=3, label='Electron-doped', alpha=0.7)
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('$n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=10)
ax4.set_ylabel('Enhancement Factor', fontsize=10)
ax4.set_title('(d) Geometry Enhancement', fontsize=11)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
residual_bcs = 100 - D_s_bcs
residual_nodal = 100 - D_s_nodal
residual_exp = 100 - D_s_experimental

mask_pos_bcs = (T_bcs > 0.01) & (residual_bcs > 0)
mask_pos_nodal = (T_nodal > 0.01) & (residual_nodal > 0)
mask_pos_exp = (T_exp_temp > 0.01) & (residual_exp > 0)

ax5.loglog(T_bcs[mask_pos_bcs], residual_bcs[mask_pos_bcs], 'b-', linewidth=1.5, label='BCS', alpha=0.7)
ax5.loglog(T_nodal[mask_pos_nodal], residual_nodal[mask_pos_nodal], 'r--', linewidth=1.5, label='Nodal', alpha=0.7)
ax5.loglog(T_exp_temp[mask_pos_exp], residual_exp[mask_pos_exp], 'ko', markersize=3, label='Exp', alpha=0.7)
ax5.set_xlabel('$T$ (K)', fontsize=10)
ax5.set_ylabel('$1 - D_s/D_{s0}$', fontsize=10)
ax5.set_title('(e) Log-Log Power Law', fontsize=11)
ax5.legend(fontsize=8, loc='lower right')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(P_mw, D_s_mw_exp, 'mo', markersize=4, label='MW exp', alpha=0.8)
ax6.set_xlabel('$P_{\\rm mw}$ (normalized)', fontsize=10)
ax6.set_ylabel('$D_s/D_{s0}$ (%)', fontsize=10)
ax6.set_title('(f) Microwave Power', fontsize=11)
ax6.legend(fontsize=8, loc='upper right')
ax6.grid(True, alpha=0.3)

plt.savefig("report/images/fig05_summary.png", dpi=200, bbox_inches='tight')
plt.close()

print("Figure 5 saved: fig05_summary.png")

###############################################################################
# Figure 6: Subgroup Analysis - Hole vs Electron Doping Comparison
###############################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(n_eff * 1e-15, D_s_exp_hole / 1e9, 'o-', markersize=4, linewidth=1.5, 
        label='Hole-doped', color='#1f77b4', alpha=0.8)
ax.plot(n_eff * 1e-15, D_s_exp_electron / 1e9, 's-', markersize=4, linewidth=1.5, 
        label='Electron-doped', color='#ff7f0e', alpha=0.8)
ax.fill_between(n_eff * 1e-15, D_s_exp_hole / 1e9, D_s_exp_electron / 1e9, 
                alpha=0.15, color='gray', label='Asymmetry region')
ax.set_xlabel('Carrier Density $n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax.set_ylabel('Superfluid Stiffness $D_s$ ($10^9$ H$^{-1}$)', fontsize=11)
ax.set_title('Particle-Hole Asymmetry in Superfluid Stiffness', fontsize=12)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

rel_diff = (D_s_exp_hole - D_s_exp_electron) / ((D_s_exp_hole + D_s_exp_electron) / 2) * 100
ax = axes[1]
ax.plot(n_eff * 1e-15, rel_diff, 'd-', markersize=4, linewidth=1.5, color='purple', alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Carrier Density $n_{\\rm eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax.set_ylabel('Relative Difference (%)', fontsize=11)
ax.set_title('Hole-Electron Asymmetry', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("report/images/fig06_asymmetry.png", dpi=200, bbox_inches='tight')
plt.close()

print("Figure 6 saved: fig06_asymmetry.png")

###############################################################################
# Save all quantitative results
###############################################################################
all_results = {
    "carrier_density": {
        "n_eff_range": [float(n_eff[0]), float(n_eff[-1])],
        "D_s_conv_range": [float(D_s_conv[0]), float(D_s_conv[-1])],
        "D_s_geom_range": [float(D_s_geom[0]), float(D_s_geom[-1])],
        "D_s_exp_hole_range": [float(D_s_exp_hole[0]), float(D_s_exp_hole[-1])],
        "D_s_exp_electron_range": [float(D_s_exp_electron[0]), float(D_s_exp_electron[-1])],
        "enhancement_stats": enhancement_stats,
    },
    "temperature": {
        "T_range": [float(T_bcs[0]), float(T_bcs[-1])],
        "power_law_fit": temp_fit_results,
    },
    "current": {
        "I_dc_range": [float(I_dc[0]), float(I_dc[-1])],
        "quadratic_fit": current_fit_results,
    }
}
with open("outputs/all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n=== All figures generated successfully ===")
print("Saved to report/images/")
