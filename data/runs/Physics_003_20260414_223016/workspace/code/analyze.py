import h5py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os
plt.rc('text', usetex=False)

# Load data
with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy = np.array(f['energy_axis'])
    kx = np.array(f['kx_axis'])
    pol_angles = np.array(f['polarization_angles'])
    pump_off = np.array(f['pump_off_spectrum'])
    pump_on_keys = [k for k in f.keys() if k.startswith('pump_on_angle_')]
    pump_on_list = [np.array(f[k]) for k in pump_on_keys]
    pump_on_avg = np.mean(pump_on_list, axis=0)

df_pol = pd.read_csv('data/polarization_dependence_data.csv')
with open('data/processed_band_data.json') as f:
    band_data = json.load(f)
dirac_E, dirac_kx = band_data['dirac_point']
pump_energy = band_data['pump_energy']
replicas = band_data['replica_bands']

print('Energy range:', energy[0], energy[-1])
print('Kx range:', kx[0], kx[-1])
print('Dirac:', dirac_E, dirac_kx)
print('Pump energy:', pump_energy)
print('Replicas:')
for r in replicas:
    print(r)

# Fit pol dep
def cos2_fit(theta_deg, I0, A, phi_deg):
    theta = np.deg2rad(theta_deg - phi_deg)
    return I0 + A * (np.cos(2 * theta))**2

popt, pcov = curve_fit(cos2_fit, df_pol['angle_degrees'], df_pol['intensity'], p0=[0.5, 0.01, 0])

# Figures
os.makedirs('report/images', exist_ok=True)

# Fig1: Data overview
fig, axs = plt.subplots(1,2, figsize=(12,5))
im1 = axs[0].pcolormesh(kx, energy, pump_off.T, cmap='hot', shading='gouraud')
axs[0].set_xlabel(r'$kx$ (Å$^{-1}$)')
axs[0].set_ylabel('Energy (eV)')
axs[0].set_title('Pump Off')
plt.colorbar(im1, ax=axs[0])
im2 = axs[1].pcolormesh(kx, energy, pump_on_avg.T, cmap='hot', shading='gouraud')
axs[1].set_title('Pump On (avg)')
plt.colorbar(im2, ax=axs[1])
axs[1].set_xlabel(r'$kx$ (Å$^{-1}$)')
plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig2: Band dispersion
fig, ax = plt.subplots(figsize=(10,6))
disp_points = np.array([[d['kx'], d['energy'], d['intensity']] for d in band_data['band_dispersion']])
scatter = ax.scatter(disp_points[:,0], disp_points[:,1], c=disp_points[:,2], s=20, cmap='viridis', alpha=0.7)
ax.scatter(dirac_kx, dirac_E, c='red', s=200, marker='*', label='Dirac point', zorder=5)
colors = {1:'green', -1:'blue'}
for r in replicas:
    color = colors.get(r['order'], 'orange')
    label = r'Replica $n={}$'.format(r['order'])
    ax.scatter(r['kx'], r['energy'], c=color, s=200, marker='o', linewidth=2, edgecolors='black',
               label=label, zorder=5)
ax.set_xlabel(r'$kx$ (Å$^{-1}$)')
ax.set_ylabel('Energy (eV)')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Intensity')
plt.savefig('report/images/band_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig3: Pol dependence
fig, ax = plt.subplots(figsize=(8,6))
theta = np.linspace(0,180,200)
ax.errorbar(df_pol['angle_degrees'], df_pol['intensity'], fmt='ro', label=r'Data (target $E=0.249$ eV, $kx=0.042$ Å$^{-1}$)')
ax.plot(theta, cos2_fit(theta, *popt), 'b-', linewidth=2, label=r'$\\cos^2$ fit')
ax.set_xlabel(r'Pump pol. angle $\\theta_p$ (deg)')
ax.set_ylabel('Replica intensity (arb. u.)')
ax.legend()
ax.set_title(r'Polarization dep.: $I_0$={:.3f}, $A$={:.3f}, $\\phi$={:.1f}$^\\circ$'.format(popt[0], popt[1], popt[2]))
plt.savefig('report/images/polarization_dependence.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig4: Replica validation
fig, ax = plt.subplots(figsize=(8,6))
n_orders = [r['order'] for r in replicas]
delta_E = [r['energy'] - dirac_E for r in replicas]
ax.scatter(n_orders, delta_E, s=100, label='Measured $\\Delta E$')
ax.plot([-2,2], np.array([-2,2])*pump_energy, 'r--', linewidth=2, label=r'$n \\times \\hbar \\omega$ ($\\omega$={:.3f} eV)'.format(pump_energy))
ax.set_xlabel('Replica order $n$')
ax.set_ylabel(r'$\\Delta E = E_{{\\rm rep}} - E_{{\\rm Dirac}}$ (eV)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('report/images/replica_validation.png', dpi=300, bbox_inches='tight')
plt.close()

# Save outputs
results = {
    'replicas': replicas,
    'dirac_point': [dirac_E, dirac_kx],
    'pump_energy': float(pump_energy),
    'pol_fit_params': popt.tolist(),
    'pol_fit_cov': np.diag(pcov).tolist()
}
with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)
df_pol.to_json('outputs/pol_data.json', orient='records')
print('Polarization fit params:', popt)
print('All figures saved to report/images/ and outputs updated.')
