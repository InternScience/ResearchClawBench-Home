"""Figure 5: Polarization dependence of replica band intensity with cos^2(2theta) fit."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.use('Agg')

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'
resultsdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/outputs'

# Load polarization data
df = pd.read_csv(f'{datadir}/polarization_dependence_data.csv')
print(df)
print()

angles_deg = df['angle_degrees'].values
angles_rad = df['angle_radians'].values
intensity = df['intensity'].values

# Model: I(theta) = I0 + A * cos^2(2*(theta - phi))
# This is the Volkov final state signature
def volkov_model(theta, I0, A, phi):
    return I0 + A * np.cos(2 * (theta - phi))**2

# Fit
popt, pcov = curve_fit(volkov_model, angles_rad, intensity, p0=[0.5, 0.01, 0.0])
perr = np.sqrt(np.diag(pcov))

I0_fit, A_fit, phi_fit = popt
I0_err, A_err, phi_err = perr

print(f"Fit parameters:")
print(f"  I0 = {I0_fit:.6f} +/- {I0_err:.6f}")
print(f"  A  = {A_fit:.6f} +/- {A_err:.6f}")
print(f"  phi = {np.degrees(phi_fit):.2f} +/- {np.degrees(phi_err):.2f} degrees")

# Compute R^2
y_pred = volkov_model(angles_rad, *popt)
ss_res = np.sum((intensity - y_pred)**2)
ss_tot = np.sum((intensity - np.mean(intensity))**2)
r_squared = 1 - ss_res / ss_tot
print(f"  R^2 = {r_squared:.6f}")

# Save fit results
import json
fit_results = {
    "model": "I(theta) = I0 + A * cos^2(2*(theta - phi))",
    "I0": float(I0_fit),
    "I0_err": float(I0_err),
    "A": float(A_fit),
    "A_err": float(A_err),
    "phi_deg": float(np.degrees(phi_fit)),
    "phi_err_deg": float(np.degrees(phi_err)),
    "R_squared": float(r_squared),
    "interpretation": "cos^2(2*theta) dependence consistent with Volkov final state mechanism"
}
with open(f'{resultsdir}/polarization_fit_results.json', 'w') as f:
    json.dump(fit_results, f, indent=2)

# Figure 5: Polarization dependence
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Fine grid for fit curve
theta_fine = np.linspace(0, np.pi, 500)
I_fine = volkov_model(theta_fine, *popt)

ax.plot(np.degrees(theta_fine), I_fine, 'b-', linewidth=2.5, label='Volkov fit: $I_0 + A\\cos^2(2\\theta)$', zorder=2)
ax.scatter(angles_deg, intensity, c='red', s=120, zorder=3, edgecolors='darkred', linewidth=1.5,
           label='Experimental Data')

ax.set_xlabel(r'Pump Polarization Angle $\theta_p$ (degrees)', fontsize=14)
ax.set_ylabel('Replica Band Intensity (arb. units)', fontsize=14)
ax.set_title('Polarization Dependence of Floquet-Bloch Replica Band\n'
             r'Evidence for Volkov Final State Mechanism', fontsize=14)

# Add fit parameters text
textstr = f'$I_0$ = {I0_fit:.4f}\n$A$ = {A_fit:.4f}\n$\\phi$ = {np.degrees(phi_fit):.1f}°\n$R^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

ax.legend(loc='lower right', fontsize=12)
ax.set_xlim(-5, 185)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/fig5_polarization_dependence.png', dpi=150, bbox_inches='tight')
print("Saved fig5_polarization_dependence.png")
plt.close()
