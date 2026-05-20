#!/usr/bin/env python3
"""MATBG Superfluid Stiffness Analysis - Part 1: Data loading and Figure 1"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'lines.linewidth': 1.5,
    'axes.grid': True, 'grid.alpha': 0.3,
})
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# === Part 1: Carrier Density Dependence ===
n_eff = np.array([5.00000000e+14, 5.91836735e+14, 6.83673469e+14, 7.75510204e+14,
    8.67346939e+14, 9.59183673e+14, 1.05102041e+15, 1.14285714e+15,
    1.23469388e+15, 1.32653061e+15, 1.41836735e+15, 1.51020408e+15,
    1.60204082e+15, 1.69387755e+15, 1.78571429e+15, 1.87755102e+15,
    1.96938776e+15, 2.06122449e+15, 2.15306122e+15, 2.24489796e+15,
    2.33673469e+15, 2.42857143e+15, 2.52040816e+15, 2.61224490e+15,
    2.70408163e+15, 2.79591837e+15, 2.88775510e+15, 2.97959184e+15,
    3.07142857e+15, 3.16326531e+15, 3.25510204e+15, 3.34693878e+15,
    3.43877551e+15, 3.53061224e+15, 3.62244898e+15, 3.71428571e+15,
    3.80612245e+15, 3.89795918e+15, 3.98979592e+15, 4.08163265e+15,
    4.17346939e+15, 4.26530612e+15, 4.35714286e+15, 4.44897959e+15,
    4.54081633e+15, 4.63265306e+15, 4.72448980e+15, 4.81632653e+15,
    4.90816327e+15, 5.00000000e+15])

D_s_conv = np.array([1.14642368e+09, 1.24696564e+09, 1.34039778e+09, 1.42782172e+09,
    1.51002634e+09, 1.58760949e+09, 1.66103385e+09, 1.73066529e+09,
    1.79679961e+09, 1.85967941e+09, 1.91950426e+09, 1.97643991e+09,
    2.03062413e+09, 2.08217148e+09, 2.13117727e+09, 2.17772057e+09,
    2.22186633e+09, 2.26366730e+09, 2.30316590e+09, 2.34039589e+09,
    2.37538376e+09, 2.40815000e+09, 2.43871016e+09, 2.46707582e+09,
    2.49325536e+09, 2.51725466e+09, 2.53907765e+09, 2.55872678e+09,
    2.57620348e+09, 2.59250847e+09, 2.60764211e+09, 2.62160464e+09,
    2.63439640e+09, 2.64601800e+09, 2.65647044e+09, 2.66575522e+09,
    2.67387435e+09, 2.68083038e+09, 2.68662640e+09, 2.69126601e+09,
    2.69475331e+09, 2.69709289e+09, 2.69828981e+09, 2.69834957e+09,
    2.69727810e+09, 2.69508174e+09, 2.69176722e+09, 2.68734165e+09,
    2.68181248e+09, 2.67518747e+09])

D_s_geom = np.array([4.91324433e+09, 5.34628159e+09, 5.74627790e+09, 6.11923597e+09,
    6.46968559e+09, 6.80104047e+09, 7.11599637e+09, 7.41669926e+09,
    7.70485534e+09, 7.98182318e+09, 8.24866141e+09, 8.50618546e+09,
    8.75503198e+09, 8.99569258e+09, 9.22854793e+09, 9.45388790e+09,
    9.67192698e+09, 9.88281486e+09, 1.00866482e+10, 1.02831122e+10,
    1.04723646e+10, 1.06546435e+10, 1.08301850e+10, 1.09992225e+10,
    1.11619857e+10, 1.13187004e+10, 1.14695875e+10, 1.16148630e+10,
    1.17547375e+10, 1.18894159e+10, 1.20190975e+10, 1.21439763e+10,
    1.22642406e+10, 1.23800735e+10, 1.24916525e+10, 1.25991504e+10,
    1.27027346e+10, 1.28025679e+10, 1.28988080e+10, 1.29916079e+10,
    1.30811160e+10, 1.31674761e+10, 1.32508276e+10, 1.33313054e+10,
    1.34090404e+10, 1.34841593e+10, 1.35567847e+10, 1.36270356e+10,
    1.36950268e+10, 1.37608697e+10])

D_s_exp_hole = np.array([3.85604343e+10, 4.24265821e+10, 4.52423238e+10, 4.93808532e+10,
    5.19704020e+10, 5.57956448e+10, 5.90341377e+10, 6.32891365e+10,
    6.66534759e+10, 7.05280091e+10, 7.38961211e+10, 7.86183663e+10,
    8.14731800e+10, 8.59709197e+10, 8.94166552e+10, 9.42492950e+10,
    9.76904386e+10, 1.02049782e+11, 1.06107094e+11, 1.10086450e+11,
    1.14387728e+11, 1.18191820e+11, 1.22351351e+11, 1.26525799e+11,
    1.30167348e+11, 1.34355941e+11, 1.37852551e+11, 1.42465133e+11,
    1.45881941e+11, 1.50437589e+11, 1.54495487e+11, 1.58821246e+11,
    1.63023454e+11, 1.67471364e+11, 1.71755524e+11, 1.75901845e+11,
    1.80239352e+11, 1.84604118e+11, 1.88613997e+11, 1.92878937e+11,
    1.96996621e+11, 2.01188574e+11, 2.05285732e+11, 2.09374636e+11,
    2.13632989e+11, 2.17696229e+11, 2.21767737e+11, 2.25876875e+11,
    2.29888452e+11, 2.33911617e+11])

D_s_exp_electron = np.array([3.66324126e+10, 4.03052529e+10, 4.29802076e+10, 4.69118005e+10,
    4.93718819e+10, 5.30058625e+10, 5.60824308e+10, 6.01246797e+10,
    6.33208021e+10, 6.70016086e+10, 7.02013150e+10, 7.46874480e+10,
    7.73995210e+10, 8.16723737e+10, 8.49458224e+10, 8.95368203e+10,
    9.28059167e+10, 9.69472943e+10, 1.00801739e+11, 1.04582127e+11,
    1.08668342e+11, 1.12282229e+11, 1.16233784e+11, 1.20199509e+11,
    1.23658981e+11, 1.27638144e+11, 1.30959923e+11, 1.35341876e+11,
    1.38587844e+11, 1.42915709e+11, 1.46750713e+11, 1.50880184e+11,
    1.54872282e+11, 1.59097796e+11, 1.63167748e+11, 1.67106753e+11,
    1.71247384e+11, 1.75373912e+11, 1.79183298e+11, 1.83234991e+11,
    1.87146790e+11, 1.91129146e+11, 1.95021445e+11, 1.98905904e+11,
    2.02951340e+11, 2.06811417e+11, 2.10679350e+11, 2.14583031e+11,
    2.18394030e+11, 2.22216036e+11])

n_eff_1012 = n_eff / 1e12
ratio_geom_conv = D_s_geom / D_s_conv
ratio_exp_hole = D_s_exp_hole / D_s_conv
ratio_exp_elec = D_s_exp_electron / D_s_conv

print(f"Conventional D_s range: [{D_s_conv.min():.2e}, {D_s_conv.max():.2e}]")
print(f"Quantum Geometric D_s range: [{D_s_geom.min():.2e}, {D_s_geom.max():.2e}]")
print(f"Mean ratio geom/conv: {ratio_geom_conv.mean():.1f}x")
print(f"Mean ratio exp_hole/conv: {ratio_exp_hole.mean():.1f}x")

# Figure 1: Carrier density dependence
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.semilogy(n_eff_1012, D_s_conv, 'b-', label='Conventional', lw=2)
ax.semilogy(n_eff_1012, D_s_geom, 'r-', label='Quantum Geometric', lw=2)
ax.semilogy(n_eff_1012, D_s_exp_hole, 'gs', ms=4, alpha=0.7, label='Experimental (hole)')
ax.semilogy(n_eff_1012, D_s_exp_electron, 'm^', ms=4, alpha=0.7, label='Experimental (electron)')
ax.set_xlabel(r'$n_{\mathrm{eff}}$ ($10^{12}$ cm$^{-2}$)')
ax.set_ylabel(r'$D_s$ (H$^{-1}$)')
ax.set_title('(a) Superfluid Stiffness vs Carrier Density')
ax.legend(loc='upper left')

ax = axes[0, 1]
ax.semilogy(n_eff_1012, D_s_conv, 'b-', label='Conventional', lw=2)
ax.semilogy(n_eff_1012, D_s_geom, 'r-', label='Quantum Geometric', lw=2)
ax.semilogy(n_eff_1012, D_s_exp_hole, 'gs', ms=5, alpha=0.8, label='Experimental (hole)')
ax.set_xlabel(r'$n_{\mathrm{eff}}$ ($10^{12}$ cm$^{-2}$)')
ax.set_ylabel(r'$D_s$ (H$^{-1}$)')
ax.set_title('(b) Low-Density Regime')
ax.legend(loc='upper left')
ax.set_xlim([0.5, 1.5])

ax = axes[1, 0]
ax.plot(n_eff_1012, ratio_geom_conv, 'r-', lw=2, label=r'$D_s^{\mathrm{geom}}/D_s^{\mathrm{conv}}$')
ax.plot(n_eff_1012, ratio_exp_hole, 'gs', ms=4, alpha=0.7, label=r'$D_s^{\mathrm{exp,hole}}/D_s^{\mathrm{conv}}$')
ax.plot(n_eff_1012, ratio_exp_elec, 'm^', ms=4, alpha=0.7, label=r'$D_s^{\mathrm{exp,e}^-}/D_s^{\mathrm{conv}}$')
ax.axhline(y=1, color='k', ls='--', alpha=0.5, label='Unity')
ax.set_xlabel(r'$n_{\mathrm{eff}}$ ($10^{12}$ cm$^{-2}$)')
ax.set_ylabel('Enhancement Ratio')
ax.set_title('(c) Quantum Geometry Enhancement Factor')
ax.legend(loc='upper left')

ax = axes[1, 1]
sel = [0, 12, 25, 37, 49]
sel_n = n_eff_1012[sel]
xp = np.arange(len(sel_n))
w = 0.2
ax.bar(xp-1.5*w, D_s_conv[sel]/1e9, w, label='Conv. (×10⁹)', color='blue', alpha=0.8)
ax.bar(xp-0.5*w, D_s_geom[sel]/1e9, w, label='Geom. (×10⁹)', color='red', alpha=0.8)
ax.bar(xp+0.5*w, D_s_exp_hole[sel]/1e11, w, label='Exp. hole (×10¹¹)', color='green', alpha=0.8)
ax.bar(xp+1.5*w, D_s_exp_electron[sel]/1e11, w, label='Exp. e⁻ (×10¹¹)', color='purple', alpha=0.8)
ax.set_xlabel(r'$n_{\mathrm{eff}}$ ($10^{12}$ cm$^{-2}$)')
ax.set_ylabel(r'$D_s$ (scaled)')
ax.set_title('(d) Comparison at Selected Densities')
ax.set_xticks(xp)
ax.set_xticklabels([f'{n:.1f}' for n in sel_n], rotation=45)
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig('report/images/fig1_carrier_density.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig1_carrier_density.png")

# Save numerical results
results1 = {
    'n_eff_1012': n_eff_1012.tolist(),
    'D_s_conv': D_s_conv.tolist(),
    'D_s_geom': D_s_geom.tolist(),
    'D_s_exp_hole': D_s_exp_hole.tolist(),
    'D_s_exp_electron': D_s_exp_electron.tolist(),
    'mean_ratio_geom_conv': float(ratio_geom_conv.mean()),
    'mean_ratio_exp_hole_conv': float(ratio_exp_hole.mean()),
    'mean_ratio_exp_electron_conv': float(ratio_exp_elec.mean()),
}
with open('outputs/carrier_density_results.json', 'w') as f:
    json.dump(results1, f, indent=2)
print("Saved: outputs/carrier_density_results.json")
