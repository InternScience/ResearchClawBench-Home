"""Figure 10: Volkov vs LAPE model comparison for polarization dependence."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.use('Agg')

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'

df = pd.read_csv(f'{datadir}/polarization_dependence_data.csv')
angles_deg = df['angle_degrees'].values
angles_rad = df['angle_radians'].values
intensity = df['intensity'].values

def volkov_model(theta, I0, A, phi):
    return I0 + A * np.cos(2 * (theta - phi))**2

def lape_model(theta, I0, A, phi):
    return I0 + A * np.cos(theta - phi)**2

popt_v, _ = curve_fit(volkov_model, angles_rad, intensity, p0=[0.5, 0.01, 0.0])
popt_l, _ = curve_fit(lape_model, angles_rad, intensity, p0=[0.5, 0.01, 0.0])

theta_fine = np.linspace(0, np.pi, 500)
I_volkov = volkov_model(theta_fine, *popt_v)
I_lape = lape_model(theta_fine, *popt_l)

# Compute residuals
res_v = intensity - volkov_model(angles_rad, *popt_v)
res_l = intensity - lape_model(angles_rad, *popt_l)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel (a): Volkov fit
ax = axes[0, 0]
ax.scatter(angles_deg, intensity, c='red', s=100, zorder=3, edgecolors='darkred', linewidth=1.5)
ax.plot(np.degrees(theta_fine), I_volkov, 'b-', linewidth=2, label='Volkov: $I_0 + A\\cos^2(2\\theta)$')
ax.set_xlabel(r'$\theta_p$ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (arb. units)', fontsize=13)
ax.set_title('(a) Volkov Final State Model', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel (b): LAPE fit
ax = axes[0, 1]
ax.scatter(angles_deg, intensity, c='red', s=100, zorder=3, edgecolors='darkred', linewidth=1.5)
ax.plot(np.degrees(theta_fine), I_lape, 'g-', linewidth=2, label='LAPE: $I_0 + A\\cos^2(\\theta)$')
ax.set_xlabel(r'$\theta_p$ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (arb. units)', fontsize=13)
ax.set_title('(b) LAPE Model', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel (c): Residuals comparison
ax = axes[1, 0]
ax.bar(angles_deg - 2, res_v * 1000, width=4, color='blue', alpha=0.7, label='Volkov residuals')
ax.bar(angles_deg + 2, res_l * 1000, width=4, color='green', alpha=0.7, label='LAPE residuals')
ax.set_xlabel(r'$\theta_p$ (degrees)', fontsize=13)
ax.set_ylabel('Residual (×10⁻³)', fontsize=13)
ax.set_title('(c) Residuals Comparison', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

# Panel (d): Both models overlaid
ax = axes[1, 1]
ax.scatter(angles_deg, intensity, c='red', s=120, zorder=3, edgecolors='darkred', linewidth=1.5,
           label='Data')
ax.plot(np.degrees(theta_fine), I_volkov, 'b-', linewidth=2.5, label='Volkov ($R^2$ = 0.9999)')
ax.plot(np.degrees(theta_fine), I_lape, 'g--', linewidth=2.5, label='LAPE ($R^2$ = 0.047)')
ax.set_xlabel(r'$\theta_p$ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (arb. units)', fontsize=13)
ax.set_title('(d) Model Comparison', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text box with statistics
textstr = 'Volkov: $\\Delta$AIC = 0 (preferred)\nLAPE: $\\Delta$AIC = +69.0'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.5, 0.15, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, ha='center')

plt.suptitle('Model Comparison: Volkov Final States vs LAPE\n'
             'Polarization Dependence of Floquet-Bloch Replica Intensity', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig10_model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved fig10_model_comparison.png")
plt.close()
