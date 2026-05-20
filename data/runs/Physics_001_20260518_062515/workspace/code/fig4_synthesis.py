#!/usr/bin/env python3
"""MATBG Synthesis Figure: Summary of all key results"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'lines.linewidth': 1.5,
    'axes.grid': True, 'grid.alpha': 0.3,
})

# Load all results
with open('outputs/carrier_density_results.json') as f:
    r1 = json.load(f)
with open('outputs/temperature_dependence_results.json') as f:
    r2 = json.load(f)
with open('outputs/current_dependence_results.json') as f:
    r3 = json.load(f)

n_eff = np.array(r1['n_eff_1012'])
D_s_conv = np.array(r1['D_s_conv'])
D_s_geom = np.array(r1['D_s_geom'])
D_s_exp_hole = np.array(r1['D_s_exp_hole'])
D_s_exp_elec = np.array(r1['D_s_exp_electron'])

T_arr = np.array(r2['T_arr'])
D_s_exp_temp = np.array(r2['D_s_exp_temp'])
alpha_fit = r2['alpha_fit']
alpha_err = r2['alpha_err']

I_dc = np.array(r3['I_dc'])
D_s_dc_exp = np.array(r3['D_s_dc_exp'])
Ic_fit = r3['Ic_fit']

# Figure 4: 3-panel synthesis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Enhancement
ax = axes[0]
ax.semilogy(n_eff, D_s_geom / D_s_conv, 'r-', lw=2, label=r'$D_s^{\rm geom}/D_s^{\rm conv}$')
ax.semilogy(n_eff, D_s_exp_hole / D_s_conv, 'gs', ms=4, alpha=0.7, label=r'$D_s^{\rm exp}/D_s^{\rm conv}$')
ax.axhline(y=1, color='k', ls='--', alpha=0.4, label='Unity')
ax.set_xlabel(r'$n_{\rm eff}$ ($10^{12}$ cm$^{-2}$)')
ax.set_ylabel('Enhancement Ratio')
ax.set_title('(a) Quantum Geometry Enhancement')
ax.legend(loc='upper left')

# Panel B: Temperature
ax = axes[1]
ax.plot(T_arr, D_s_exp_temp, 'ko', ms=5, alpha=0.8, label='Experimental')
mask_fit = (T_arr > 0.05) & (T_arr < 1.0)
Ds_fit_full = 100.0 * (1.0 - (T_arr / 1.0)**alpha_fit)
ax.plot(T_arr[mask_fit], Ds_fit_full[mask_fit], 'r-', lw=2,
        label=fr'Fit: $\alpha={alpha_fit:.2f}\pm{alpha_err:.2f}$')
ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Temperature $T$ (K)')
ax.set_ylabel('$D_s(T)/D_s(0)$ (%)')
ax.set_title('(b) Power-Law Temperature Dependence')
ax.legend(loc='upper right')

# Panel C: Current
ax = axes[2]
ax.plot(I_dc, D_s_dc_exp, 'ko', ms=5, alpha=0.8, label='Experimental DC')
I_plot = np.linspace(0, 60, 200)
ax.plot(I_plot, 100*(1 - (I_plot/Ic_fit)**2), 'b-', lw=2,
        label=f'Fit: $I_c={Ic_fit:.1f}$ nA')
ax.axvline(x=Ic_fit, color='gray', ls='--', alpha=0.5)
ax.set_xlabel(r'DC Current $I_{\rm dc}$ (nA)')
ax.set_ylabel('$D_s/D_s(0)$ (%)')
ax.set_title('(c) Current-Driven Suppression')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('report/images/fig4_synthesis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig4_synthesis.png")

# Figure 5: Detailed gap structure comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# BCS vs Nodal gap comparison
ax = axes[0]
T_arr = np.array(r2['T_arr'])
D_s_bcs = np.array(r2['D_s_bcs'])
D_s_nodal = np.array(r2['D_s_nodal'])
D_s_pn2 = np.array(r2['D_s_power_n2'])
D_s_pn25 = np.array(r2['D_s_power_n25'])
D_s_pn3 = np.array(r2['D_s_power_n3'])

ax.plot(T_arr, D_s_bcs, 'b-', lw=2, label='BCS')
ax.plot(T_arr, D_s_nodal, 'r-', lw=2, label='Nodal (linear)')
ax.plot(T_arr, D_s_pn2, 'c--', lw=1.5, label=r'Power $\alpha=2$')
ax.plot(T_arr, D_s_pn25, 'm--', lw=1.5, label=r'Power $\alpha=2.5$')
ax.plot(T_arr, D_s_pn3, 'g--', lw=1.5, label=r'Power $\alpha=3$')
ax.plot(T_arr, D_s_exp_temp, 'ko', ms=4, alpha=0.6, label='Exp.')
ax.set_xlabel('$T/T_c$')
ax.set_ylabel('$D_s/D_s(0)$ (%)')
ax.set_title('(a) Gap Structure Comparison')
ax.legend(fontsize=7)

# Residuals from different models
ax = axes[1]
models = {
    'BCS': D_s_bcs,
    'Nodal': D_s_nodal,
    r'Power $\alpha=2$': D_s_pn2,
    r'Power $\alpha=2.5$': D_s_pn25,
    r'Power $\alpha=3$': D_s_pn3,
}
chi2 = {}
for name, model in models.items():
    diff = D_s_exp_temp - model
    chi2[name] = np.sum(diff**2) / len(diff)
    ax.plot(T_arr, diff, lw=1.5, label=f'{name} ($\\chi^2$={chi2[name]:.1f})')
ax.axhline(y=0, color='k', ls='--', alpha=0.3)
ax.set_xlabel('$T/T_c$')
ax.set_ylabel('Residual (%)')
ax.set_title('(b) Model Residuals')
ax.legend(fontsize=7)

# Critical current comparison
ax = axes[2]
I_mw = np.array(r3['I_mw'])
D_s_mw = np.array(r3['D_s_mw'])
ax.plot(I_dc, D_s_dc_exp, 'b-o', ms=4, lw=1.5, label='DC bias')
ax.plot(I_mw, D_s_mw, 'r-s', ms=4, lw=1.5, label='MW probe')
ax.axvline(x=Ic_fit, color='b', ls=':', alpha=0.5, label=f'$I_c^{{DC}}$={Ic_fit:.0f} nA')
ax.set_xlabel('Current (nA)')
ax.set_ylabel('$D_s/D_s(0)$ (%)')
ax.set_title('(c) DC vs Microwave Current')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/fig5_gap_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig5_gap_comparison.png")

# Figure 6: Summary table as figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table_data = [
    ['Quantity', 'Value', 'Significance'],
    [r'$D_s^{\rm geom}/D_s^{\rm conv}$', f'{r1["mean_ratio_geom_conv"]:.1f}x',
     'Quantum geometric enhancement'],
    [r'$D_s^{\rm exp}/D_s^{\rm conv}$', f'{r1["mean_ratio_exp_hole_conv"]:.0f}x',
     'Total experimental enhancement'],
    [r'$\alpha$ (power law)', f'{alpha_fit:.2f} $\\pm$ {alpha_err:.2f}',
     'Anisotropic gap exponent'],
    [r'$I_c$', f'{Ic_fit:.1f} nA',
     'Critical current for depairing'],
    [r'$T_c$', '1.0 K', 'Superconducting transition'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.35, 0.25, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#D9E2F3')
ax.set_title('Key Results Summary: MATBG Superfluid Stiffness', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('report/images/fig6_summary_table.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig6_summary_table.png")

# Save all final results
final = {
    'carrier_density': r1,
    'temperature': r2,
    'current': r3,
    'key_findings': {
        'quantum_geometry_enhancement_factor': r1['mean_ratio_geom_conv'],
        'total_enhancement_factor': r1['mean_ratio_exp_hole_conv'],
        'power_law_exponent': alpha_fit,
        'power_law_exponent_error': alpha_err,
        'critical_current_nA': Ic_fit,
    }
}
with open('outputs/all_results.json', 'w') as f:
    json.dump(final, f, indent=2)
print("Saved: outputs/all_results.json")
print("\nAll figures generated successfully!")
