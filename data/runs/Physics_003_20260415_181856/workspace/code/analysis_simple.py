"""
Floquet-Bloch States Analysis - Simplified Version
"""

import numpy as np
import h5py
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Paths
DATA_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/data'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/outputs'
FIGURE_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("Loading data...")
# Load data
with h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r') as f:
    energy = np.array(f['energy_axis'])
    kx = np.array(f['kx_axis'])
    pump_off = np.array(f['pump_off_spectrum'])
    pump_on_0 = np.array(f['pump_on_angle_0'])
    pump_on_90 = np.array(f['pump_on_angle_90'])
    angles = np.array(f['polarization_angles'])

with open(os.path.join(DATA_DIR, 'processed_band_data.json'), 'r') as f:
    processed = json.load(f)

pol_data = pd.read_csv(os.path.join(DATA_DIR, 'polarization_dependence_data.csv'))

print("Data loaded successfully!")

# Figure 1: Raw spectra comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im0 = axes[0,0].pcolormesh(kx, energy, pump_off, shading='auto', cmap='hot')
axes[0,0].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[0,0].set_ylabel('Energy (eV)')
axes[0,0].set_title('Pump-Off Spectrum (Equilibrium)')
axes[0,0].axhline(y=-0.3, color='cyan', linestyle='--', alpha=0.7)
plt.colorbar(im0, ax=axes[0,0])

im1 = axes[0,1].pcolormesh(kx, energy, pump_on_0, shading='auto', cmap='hot')
axes[0,1].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[0,1].set_ylabel('Energy (eV)')
axes[0,1].set_title(r'Pump-On ($\theta_p = 0°$)')
axes[0,1].axhline(y=-0.3, color='cyan', linestyle='--', alpha=0.7)
axes[0,1].axhline(y=-0.3+0.248, color='lime', linestyle=':', alpha=0.7)
axes[0,1].axhline(y=-0.3-0.248, color='lime', linestyle=':', alpha=0.7)
plt.colorbar(im1, ax=axes[0,1])

diff_0 = pump_on_0 - pump_off
vmax = np.abs(diff_0).max()
im2 = axes[1,0].pcolormesh(kx, energy, diff_0, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1,0].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[1,0].set_ylabel('Energy (eV)')
axes[1,0].set_title(r'Pump-Induced Change ($\theta_p = 0°$)')
plt.colorbar(im2, ax=axes[1,0])

im3 = axes[1,1].pcolormesh(kx, energy, pump_on_90, shading='auto', cmap='hot')
axes[1,1].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[1,1].set_ylabel('Energy (eV)')
axes[1,1].set_title(r'Pump-On ($\theta_p = 90°$)')
plt.colorbar(im3, ax=axes[1,1])

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_01_raw_spectra.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# Figure 2: Floquet bands
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

band_disp = processed['band_dispersion']
energies = np.array([p['energy'] for p in band_disp])
kxs = np.array([p['kx'] for p in band_disp])
replicas = processed['replica_bands']

axes[0].scatter(kxs, energies, c='gray', s=10, alpha=0.5, label='Main band')
for rb in replicas:
    color = 'blue' if rb['order'] < 0 else 'red'
    axes[0].scatter(rb['kx'], rb['energy'], c=color, s=200, marker='*', 
                    edgecolors='black', linewidths=1, zorder=5)
axes[0].scatter(processed['dirac_point'][0], processed['dirac_point'][1], 
                c='green', s=300, marker='X', edgecolors='black', linewidths=2, zorder=5)
axes[0].axhline(y=-0.3+0.248, color='red', linestyle=':', alpha=0.5)
axes[0].axhline(y=-0.3-0.248, color='blue', linestyle=':', alpha=0.5)
axes[0].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[0].set_ylabel('Energy (eV)')
axes[0].set_title('Extracted Band Structure with Replica Bands')
axes[0].legend(['Main band', 'n=-1 replica', 'n=+1 replica', 'Dirac point'])
axes[0].grid(True, alpha=0.3)

# Schematic
k_schem = np.linspace(-0.08, 0.08, 100)
E_dirac = -0.3
v_f = 6
E_cone = E_dirac + v_f * np.abs(k_schem)
axes[1].plot(k_schem, E_cone, 'k-', linewidth=2, label='n=0')
axes[1].plot(k_schem, -E_cone + 2*E_dirac, 'k-', linewidth=2)
axes[1].plot(k_schem, E_cone + 0.248, 'r--', linewidth=1.5, label='n=+1')
axes[1].plot(k_schem, E_cone - 0.248, 'b--', linewidth=1.5, label='n=-1')
axes[1].plot(k_schem, -E_cone + 2*E_dirac + 0.248, 'r--', linewidth=1.5)
axes[1].plot(k_schem, -E_cone + 2*E_dirac - 0.248, 'b--', linewidth=1.5)
for rb in replicas:
    axes[1].scatter([rb['kx']], [rb['energy']], c='cyan', s=100, marker='o', 
                    edgecolors='black', zorder=5)
axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[1].set_ylabel('Energy (eV)')
axes[1].set_title('Schematic of Floquet Sidebands')
axes[1].legend()
axes[1].set_xlim([-0.08, 0.08])
axes[1].set_ylim([-0.6, 0.3])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_02_floquet_bands.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# Figure 3: Polarization dependence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

angles_rad = pol_data['angle_radians'].values
intensities = pol_data['intensity'].values

def sinusoidal(theta, I0, A, phi):
    return I0 + A * np.cos(2 * (theta - phi))

popt, _ = curve_fit(sinusoidal, angles_rad, intensities, 
                    p0=[np.mean(intensities), 0.01, 0])
theta_fine = np.linspace(0, np.pi, 200)

axes[0].plot(np.degrees(theta_fine), sinusoidal(theta_fine, *popt), 'b-', linewidth=2)
axes[0].scatter(pol_data['angle_degrees'], intensities, c='red', s=100, zorder=5)
axes[0].set_xlabel('Pump Polarization Angle (degrees)')
axes[0].set_ylabel('Replica Band Intensity')
axes[0].set_title('Polarization Dependence of Replica Intensity')
axes[0].grid(True, alpha=0.3)

# Polar plot
ax_polar = fig.add_subplot(122, projection='polar')
theta_full = np.linspace(0, 2*np.pi, 400)
ax_polar.plot(theta_full, sinusoidal(theta_full, *popt), 'b-', linewidth=2)
ax_polar.scatter(angles_rad, intensities, c='red', s=100, zorder=5)
ax_polar.scatter(angles_rad + np.pi, intensities, c='red', s=100, zorder=5)
ax_polar.set_theta_zero_location('E')
ax_polar.set_title('Polar Plot', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_03_polarization.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# Figure 4: EDC and MDC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# EDC at kx ≈ 0
kx_center = np.argmin(np.abs(kx))
kx_slice = slice(kx_center-3, kx_center+4)
colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
for i, ang in enumerate(angles):
    with h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r') as f:
        pump_on = np.array(f[f'pump_on_angle_{ang}'])
    edc = np.mean(pump_on[:, kx_slice], axis=1)
    axes[0].plot(energy, edc, label=f'{ang}°', color=colors[i])
axes[0].axvline(x=-0.3, color='k', linestyle='--', alpha=0.5)
axes[0].axvline(x=-0.3+0.248, color='r', linestyle=':', alpha=0.5)
axes[0].axvline(x=-0.3-0.248, color='b', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Energy (eV)')
axes[0].set_ylabel('Intensity')
axes[0].set_title('Energy Distribution Curves at $k_x \\approx 0$')
axes[0].legend(title='Angle', fontsize=8)
axes[0].grid(True, alpha=0.3)

# MDC at different energies
idx_dirac = np.argmin(np.abs(energy - (-0.3)))
idx_plus = np.argmin(np.abs(energy - (-0.3 + 0.248)))
idx_minus = np.argmin(np.abs(energy - (-0.3 - 0.248)))

axes[1].plot(kx, pump_on_0[idx_dirac, :], 'k-', linewidth=2, label='n=0 (Dirac)')
axes[1].plot(kx, pump_on_0[idx_plus, :], 'r--', linewidth=2, label='n=+1')
axes[1].plot(kx, pump_on_0[idx_minus, :], 'b:', linewidth=2, label='n=-1')
axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)')
axes[1].set_ylabel('Intensity')
axes[1].set_title('Momentum Distribution Curves ($\\theta_p = 0°$)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_04_edc_mdc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved")

# Figure 5: Multiple angles comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, ang in enumerate(angles):
    with h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r') as f:
        pump_on = np.array(f[f'pump_on_angle_{ang}'])
    diff = pump_on - pump_off
    vmax = np.abs(diff).max()
    im = axes[i].pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[i].axhline(y=-0.3, color='k', linestyle='--', alpha=0.5)
    axes[i].set_xlabel(r'$k_x$ (Å$^{-1}$)')
    axes[i].set_ylabel('Energy (eV)')
    axes[i].set_title(f'{ang}°')
    plt.colorbar(im, ax=axes[i], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'figure_05_angle_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved")

# Save results
results = {
    'floquet_photon_energy_eV': 0.248,
    'pump_wavelength_um': 5.0,
    'dirac_point': processed['dirac_point'],
    'replica_bands': processed['replica_bands'],
    'polarization_fit': {
        'I0': float(popt[0]),
        'amplitude': float(popt[1]),
        'phase_rad': float(popt[2])
    }
}
with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("All figures and results saved successfully!")
