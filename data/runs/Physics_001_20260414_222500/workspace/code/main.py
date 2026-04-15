import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
import os

os.makedirs('report/images', exist_ok=True)

# Load data from json
with open('outputs/data_parsed.json', 'r') as f:
    data = json.load(f)

# Carrier density
carrier = data['carrier_density']
n_eff = np.array(carrier['n_eff'])
ds_conv = np.array(carrier['D_s_conv'])
ds_geom = np.array(carrier['D_s_geom'])
ds_exp_hole = np.array(carrier['D_s_exp_hole'])
ds_exp_elec = np.array(carrier['D_s_exp_electron'])

# Ratios
ratio_geom_conv = np.max(ds_geom / ds_conv)
ratio_hole_conv = np.max(ds_exp_hole / ds_conv)
ratio_elec_conv = np.max(ds_exp_elec / ds_conv)

print(f'Max ratios: geom/conv={ratio_geom_conv:.1f}, hole/conv={ratio_hole_conv:.0f}, elec/conv={ratio_elec_conv:.0f}')

# Plot 1: Carrier density
plt.figure(figsize=(10,6))
plt.semilogy(n_eff*1e-15, ds_conv*1e-9, 'b-', linewidth=2, label='Conventional (FL)')
plt.semilogy(n_eff*1e-15, ds_geom*1e-9, 'g-', linewidth=2, label='Quantum Geometric')
plt.semilogy(n_eff*1e-15, ds_exp_hole*1e-9, 'ro', markersize=6, label='Exp Hole-doped')
plt.semilogy(n_eff*1e-15, ds_exp_elec*1e-9, 'r^', markersize=6, label='Exp Electron-doped')
plt.xlabel(r'Effective Carrier Density $n_\mathrm{eff}$ ($10^{15}$ m$^{-2}$)')
plt.ylabel(r'Superfluid Stiffness $D_s$ (GPa)')
plt.legend()
plt.title('Superfluid Stiffness vs Carrier Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig1_carrier.png', dpi=300, bbox_inches='tight')
plt.close()

# Temperature
temp = data['temperature']
T = np.array(temp['T'])
ds_bcs = np.array(temp['D_s_bcs'])
ds_nodal = np.array(temp['D_s_nodal'])
ds_power_n2 = np.array(temp['D_s_power_n2'])
ds_power_n25 = np.array(temp['D_s_power_n2_5'])
ds_power_n3 = np.array(temp['D_s_power_n3'])
ds_exp_temp = np.array(temp['D_s_experimental'])

Tc = 1.0
norm_bcs = ds_bcs / ds_bcs[0]
norm_nodal = ds_nodal / ds_nodal[0]
norm_p2 = ds_power_n2 / ds_power_n2[0]
norm_p25 = ds_power_n25 / ds_power_n25[0]
norm_p3 = ds_power_n3 / ds_power_n3[0]
norm_exp = ds_exp_temp / ds_exp_temp[0]

# Power-law fit
def power_law(t_norm, alpha):
    return 1 - t_norm**alpha

mask = T < Tc
popt, pcov = curve_fit(power_law, T[mask]/Tc, norm_exp[mask], p0=[2.0], bounds=(1,4))
alpha_fit = popt[0]
alpha_err = np.sqrt(pcov[0,0])
print(f'Power-law exponent alpha={alpha_fit:.2f} ± {alpha_err:.2f}')

# Plot 2: Temperature
plt.figure(figsize=(10,6))
plt.loglog(T/Tc, norm_bcs, 'b-', linewidth=2, label='BCS s-wave')
plt.loglog(T/Tc, norm_nodal, 'm-', linewidth=2, label='Nodal (linear)')
plt.loglog(T/Tc, norm_p2, 'orange', linewidth=2, label='Power-law $\\alpha=2.0$')
plt.loglog(T/Tc, norm_p25, 'g-', linewidth=2, label='Power-law $\\alpha=2.5$')
plt.loglog(T/Tc, norm_p3, 'r-', linewidth=2, label='Power-law $\\alpha=3.0$')
plt.loglog(T/Tc, norm_exp, 'k.', markersize=4, label='Experimental (noisy)')
t_fit = np.linspace(0,1,100)
plt.loglog(t_fit, power_law(t_fit, alpha_fit), 'k--', linewidth=2, label=f'Fit $\\alpha={alpha_fit:.1f}$')
plt.axvline(1, color='gray', linestyle='--', alpha=0.7)
plt.xlabel(r'$T / T_c$')
plt.ylabel(r'$D_s(T) / D_s(0)$')
plt.ylim(1e-3, 1.1)
plt.legend()
plt.title('Temperature Dependence of Normalized Superfluid Stiffness')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig2_temp.png', dpi=300, bbox_inches='tight')
plt.close()

# Current DC
curr = data['current']
I_dc = np.array(curr['I_dc'])
ds_gl = np.array(curr['D_s_gl'])
ds_linear = np.array(curr['D_s_linear'])
ds_dc_exp = np.array(curr['D_s_dc_exp'])

# Plot 3: DC current
plt.figure(figsize=(10,6))
plt.plot(I_dc, ds_gl, 'b-', linewidth=2, label='Ginzburg-Landau')
plt.plot(I_dc, ds_linear, 'g-', linewidth=2, label='Linear Meissner')
plt.plot(I_dc, ds_dc_exp, 'ro-', linewidth=2, markersize=4, label='Experimental DC')
plt.xlabel(r'DC Current $I_\mathrm{dc}$ (nA)')
plt.ylabel(r'$D_s$ (arb. units)')
plt.legend()
plt.title('Superfluid Stiffness vs DC Bias Current')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig3_dc_current.png', dpi=300, bbox_inches='tight')
plt.close()

# Microwave
P_mw = np.array(curr['P_mw'])
I_mw = np.array(curr['I_mw_amplitude'])
ds_mw_exp = np.array(curr['D_s_mw_exp'])

plt.figure(figsize=(10,6))
plt.plot(I_mw, ds_mw_exp, 'm^-', linewidth=2, markersize=6, label='Experimental Microwave')
plt.xlabel(r'Microwave Current Amplitude $I_\mathrm{mw}$ (nA)')
plt.ylabel(r'$D_s$ (arb. units)')
plt.legend()
plt.title('Superfluid Stiffness vs Microwave Probe Current')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig4_mw.png', dpi=300, bbox_inches='tight')
plt.close()

# Quantitative results
quant = {
    'max_ratio_geom_conv': float(ratio_geom_conv),
    'max_ratio_exp_hole_conv': float(ratio_hole_conv),
    'max_ratio_exp_elec_conv': float(ratio_elec_conv),
    'power_law_alpha_fit': float(alpha_fit),
    'alpha_fit_error': float(alpha_err)
}

with open('outputs/quant_results.json', 'w') as f:
    json.dump(quant, f, indent=2)

print('Analysis complete. Figures saved to report/images/.')
print('Quantitative results:', quant)
