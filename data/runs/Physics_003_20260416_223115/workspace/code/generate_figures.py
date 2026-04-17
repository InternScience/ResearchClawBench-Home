#!/usr/bin/env python3
"""Generate figures for Floquet-Bloch analysis."""

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")

OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_IMAGES_DIR.mkdir(exist_ok=True)

print("Loading data...")

# Load raw data
with h5py.File(DATA_DIR / 'raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on_0 = f['pump_on_angle_0'][:]
    pump_on_90 = f['pump_on_angle_90'][:]

# Load processed data
with open(DATA_DIR / 'processed_band_data.json', 'r') as f:
    processed = json.load(f)

# Load polarization data
pol_data = pd.read_csv(DATA_DIR / 'polarization_dependence_data.csv')

print(f"Energy: {len(energy)} points, Kx: {len(kx)} points")
KX, ENERGY = np.meshgrid(kx, energy)

# Figure 1: Data Overview
print("Creating Figure 1...")
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
im = ax.pcolormesh(KX, ENERGY, pump_off, shading='gouraud', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Momentum k_x (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Pump-Off: Equilibrium Dirac Cone')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
plt.colorbar(im, ax=ax, label='Intensity (a.u.)')

ax = axes[0, 1]
im = ax.pcolormesh(KX, ENERGY, pump_on_0, shading='gouraud', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('Momentum k_x (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Pump-On (θ=0°): Floquet Replica Bands')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
plt.colorbar(im, ax=ax, label='Intensity (a.u.)')

ax = axes[1, 0]
diff = pump_on_0 - pump_off
im = ax.pcolormesh(KX, ENERGY, diff, shading='gouraud', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('Momentum k_x (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Difference Spectrum')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.colorbar(im, ax=ax, label='Δ Intensity')

ax = axes[1, 1]
dirac_idx = len(kx) // 2
ax.plot(energy, pump_off[:, dirac_idx], 'k-', label='Pump-Off', lw=2)
ax.plot(energy, pump_on_0[:, dirac_idx], 'r-', label='Pump-On', lw=2)
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title('EDC at k_x ≈ 0')
ax.legend()
ax.set_ylim([-0.6, 0.4])

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'fig1_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig1_data_overview.png")

# Figure 2: Replica Bands
print("Creating Figure 2...")
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.pcolormesh(KX, ENERGY, pump_on_0, shading='gouraud', cmap='viridis', vmin=0, vmax=0.7, alpha=0.6)

dirac = processed['dirac_point']
ax.plot(dirac[1], dirac[0], 'wo', ms=12, markeredgecolor='black', markeredgewidth=2, label='Dirac Point')

replicas = processed['replica_bands']
for r in replicas:
    color = 'blue' if r['order'] == -1 else 'red'
    ax.plot(r['kx'], r['energy'], 'o', color=color, ms=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.annotate(f"n={r['order']}", xy=(r['kx'], r['energy']),
               xytext=(r['kx'] + 0.01*np.sign(r['kx']), r['energy'] + 0.05*np.sign(r['order'])),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('Momentum k_x (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Floquet Replica Bands')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax.legend()
plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'fig2_replica_bands.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig2_replica_bands.png")

# Figure 3: Polarization Dependence
print("Creating Figure 3...")
fig, ax = plt.subplots(figsize=(10, 6))

angles_deg = pol_data['angle_degrees'].values
angles_rad = pol_data['angle_radians'].values
intensities = pol_data['intensity'].values

ax.scatter(angles_deg, intensities, s=100, c='darkblue', zorder=3, label='Measured')

def cos2_model(theta, A, B, theta0):
    return A * np.cos(theta - theta0)**2 + B

try:
    popt, pcov = curve_fit(cos2_model, angles_rad, intensities, p0=[0.05, 0.47, 0])
    theta_fit = np.linspace(-np.pi/2, 3*np.pi/2, 200)
    intensity_fit = cos2_model(theta_fit, *popt)
    ax.plot(np.degrees(theta_fit), intensity_fit, 'r-', lw=2, 
            label=f'Fit: A·cos²(θ-θ₀)+B\nA={popt[0]:.4f}, θ₀={np.degrees(popt[2]):.1f}°')
    
    fit_results = {
        'model': 'A*cos^2(theta - theta0) + B',
        'parameters': {'A': float(popt[0]), 'B': float(popt[1]), 
                      'theta0_rad': float(popt[2]), 'theta0_deg': float(np.degrees(popt[2]))},
        'covariance': pcov.tolist()
    }
    with open(OUTPUTS_DIR / 'polarization_fit_results.json', 'w') as f:
        json.dump(fit_results, f, indent=2)
except Exception as e:
    print(f"  Fit failed: {e}")

ax.set_xlabel('Pump Polarization Angle θ_p (degrees)')
ax.set_ylabel('Replica Band Intensity (a.u.)')
ax.set_title('Polarization Dependence of Floquet Replica Band')
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'fig3_polarization_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig3_polarization_dependence.png")

# Figure 4: Band Dispersion
print("Creating Figure 4...")
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.pcolormesh(KX, ENERGY, pump_off, shading='gouraud', cmap='magma', vmin=0, vmax=0.8)

band_disp = processed['band_dispersion']
energies = [p['energy'] for p in band_disp]
kxs = [p['kx'] for p in band_disp]
ints = [p['intensity'] for p in band_disp]

scatter = ax.scatter(kxs, energies, c=ints, s=50, cmap='coolwarm', 
                     edgecolors='black', linewidth=0.5, label='Band Dispersion')

ax.plot(dirac[1], dirac[0], 'w*', ms=20, markeredgecolor='gold', markeredgewidth=2, label='Dirac Point')

for r in replicas:
    color = 'cyan' if r['order'] == -1 else 'yellow'
    ax.plot(r['kx'], r['energy'], 's', color=color, ms=12, 
            markeredgecolor='black', markeredgewidth=1.5, label=f"Replica n={r['order']}")

ax.set_xlabel('Momentum k_x (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Band Dispersion: Dirac Cone and Floquet-Bloch States')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='Fermi Level')
ax.legend(loc='lower right')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Spectral Weight (a.u.)')
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'fig4_band_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved fig4_band_dispersion.png")

# Save data overview
print("Saving data overview...")
overview = {
    'raw_trarpes_data': {
        'energy_range_eV': [float(energy.min()), float(energy.max())],
        'energy_points': int(len(energy)),
        'kx_range_Angstrom_inv': [float(kx.min()), float(kx.max())],
        'kx_points': int(len(kx)),
        'spectrum_shape': list(pump_off.shape)
    },
    'processed_band_data': {
        'dirac_point': processed['dirac_point'],
        'n_replica_bands': len(processed['replica_bands']),
        'n_band_dispersion_points': len(processed['band_dispersion'])
    },
    'polarization_data': {
        'n_angles': len(pol_data),
        'angle_range_degrees': [int(pol_data['angle_degrees'].min()), int(pol_data['angle_degrees'].max())]
    }
}
with open(OUTPUTS_DIR / 'data_overview.json', 'w') as f:
    json.dump(overview, f, indent=2)

# Compute replica analysis
print("Computing replica band analysis...")
replica_analysis = {
    'dirac_point': {'energy_eV': dirac[0], 'kx_Angstrom_inv': dirac[1]},
    'replica_bands': [],
    'photon_energy_estimate_eV': None
}

energy_spacings = []
for r in replicas:
    delta_e = r['energy'] - dirac[0]
    energy_spacings.append(abs(delta_e))
    replica_analysis['replica_bands'].append({
        'order': r['order'], 'energy_eV': r['energy'], 'kx_Angstrom_inv': r['kx'],
        'intensity': r['intensity'], 'delta_energy_eV': delta_e
    })

if len(energy_spacings) >= 2:
    photon_est = np.mean(energy_spacings)
    replica_analysis['photon_energy_estimate_eV'] = float(photon_est)
    expected = 1.2398 / 5.0  # 5 μm pump
    replica_analysis['expected_photon_energy_eV'] = float(expected)
    replica_analysis['deviation_percent'] = float(abs(photon_est - expected) / expected * 100)

with open(OUTPUTS_DIR / 'replica_band_analysis.json', 'w') as f:
    json.dump(replica_analysis, f, indent=2)

print("\nAnalysis complete!")
print(f"Outputs: {OUTPUTS_DIR.absolute()}")
print(f"Figures: {REPORT_IMAGES_DIR.absolute()}")
