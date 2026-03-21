#!/usr/bin/env python3
"""
Floquet-Bloch States in Monolayer Epitaxial Graphene: tr-ARPES Analysis
=======================================================================
Analyzes time-resolved ARPES data to observe Floquet-Bloch replica bands,
characterize their dispersion, and study polarization dependence.
"""

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Custom colormap for ARPES (blue-black-yellow)
arpes_cmap = LinearSegmentedColormap.from_list(
    'arpes', ['#000020', '#000060', '#0000A0', '#4040C0', '#8080E0',
              '#C0C0FF', '#FFFFFF', '#FFFFC0', '#FFE080', '#FFC040',
              '#FF8000', '#FF4000', '#C00000']
)

# ============================================================
# 1. Load all data
# ============================================================
print("Loading data...")

# Raw HDF5
h5 = h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r')
energy = h5['energy_axis'][:]
kx = h5['kx_axis'][:]
time_delays = h5['time_delays'][:]
pol_angles = h5['polarization_angles'][:]
pump_off = h5['pump_off_spectrum'][:]
pump_energy_eV = h5.attrs['pump_energy_eV']
pump_wavelength = h5.attrs['pump_wavelength_um']

# Load all pump-on spectra
pump_on = {}
for angle in pol_angles:
    pump_on[angle] = h5[f'pump_on_angle_{angle}'][:]

# Processed band data
with open(os.path.join(DATA_DIR, 'processed_band_data.json')) as f:
    band_data = json.load(f)

# Polarization dependence
pol_df = pd.read_csv(os.path.join(DATA_DIR, 'polarization_dependence_data.csv'))

print(f"Energy range: [{energy[0]:.3f}, {energy[-1]:.3f}] eV, {len(energy)} points")
print(f"kx range: [{kx[0]:.3f}, {kx[-1]:.3f}] Å⁻¹, {len(kx)} points")
print(f"Pump energy: {pump_energy_eV} eV (λ = {pump_wavelength} μm)")
print(f"Time delays: {time_delays} ps")
print(f"Polarization angles: {pol_angles}°")

# ============================================================
# 2. Figure 1: Pump-off equilibrium band structure (Dirac cone)
# ============================================================
print("\nGenerating Figure 1: Equilibrium Dirac cone...")

fig, ax = plt.subplots(figsize=(6, 7))
extent = [kx[0], kx[-1], energy[0], energy[-1]]
im = ax.imshow(pump_off, aspect='auto', origin='lower', extent=extent,
               cmap=arpes_cmap, vmin=np.percentile(pump_off, 5),
               vmax=np.percentile(pump_off, 99))
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=13)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=13)
ax.set_title('Equilibrium Band Structure\n(Pump Off)', fontsize=14)
plt.colorbar(im, ax=ax, label='Photoemission Intensity (arb. units)', shrink=0.8)

# Mark Dirac point
dp = band_data['dirac_point']
ax.plot(dp[0], dp[1], 'w+', markersize=12, markeredgewidth=2)
ax.annotate(f'Dirac Point\n({dp[0]:.2f}, {dp[1]:.2f} eV)',
            xy=(dp[0], dp[1]), xytext=(dp[0]+0.08, dp[1]+0.15),
            color='white', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_equilibrium_dirac_cone.png'), dpi=200)
plt.close()

# ============================================================
# 3. Figure 2: Pump-on spectrum showing Floquet replicas
# ============================================================
print("Generating Figure 2: Pump-on spectrum with replicas...")

pump_on_0 = pump_on[0]
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# (a) Pump off
ax = axes[0]
im0 = ax.imshow(pump_off, aspect='auto', origin='lower', extent=extent,
                cmap=arpes_cmap, vmin=np.percentile(pump_off, 5),
                vmax=np.percentile(pump_off, 99))
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=12)
ax.set_title('(a) Pump Off', fontsize=13)

# (b) Pump on
ax = axes[1]
im1 = ax.imshow(pump_on_0, aspect='auto', origin='lower', extent=extent,
                cmap=arpes_cmap, vmin=np.percentile(pump_on_0, 5),
                vmax=np.percentile(pump_on_0, 99))
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=12)
ax.set_title(r'(b) Pump On ($\theta_p = 0°$)', fontsize=13)

# Mark replica bands
for rb in band_data['replica_bands']:
    ax.plot(rb['kx'], rb['energy'], 'ro', markersize=8, markerfacecolor='none',
            markeredgewidth=1.5)
    label = f"n={rb['order']:+d}"
    ax.annotate(label, xy=(rb['kx'], rb['energy']),
                xytext=(rb['kx']+0.02, rb['energy']+0.03),
                color='red', fontsize=9, fontweight='bold')

# (c) Difference
diff = pump_on_0 - pump_off
diff_smooth = gaussian_filter(diff, sigma=1.0)
ax = axes[2]
vmax_diff = np.percentile(np.abs(diff_smooth), 98)
im2 = ax.imshow(diff_smooth, aspect='auto', origin='lower', extent=extent,
                cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=12)
ax.set_title('(c) Difference (On − Off)', fontsize=13)
plt.colorbar(im2, ax=ax, label='ΔI (arb. units)', shrink=0.8)

# Mark expected replica positions
for rb in band_data['replica_bands']:
    ax.plot(rb['kx'], rb['energy'], 'ko', markersize=8, markerfacecolor='none',
            markeredgewidth=1.5)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_floquet_replicas.png'), dpi=200)
plt.close()

# ============================================================
# 4. Figure 3: Replica band positions and Floquet gap analysis
# ============================================================
print("Generating Figure 3: Replica band energy-momentum map...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# (a) Band dispersion with replica annotations
ax = axes[0]
disp = band_data['band_dispersion']
e_disp = [p['energy'] for p in disp]
k_disp = [p['kx'] for p in disp]
I_disp = [p['intensity'] for p in disp]

# Plot as scatter with intensity as color
sc = ax.scatter(k_disp, e_disp, c=I_disp, cmap=arpes_cmap, s=4, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Intensity', shrink=0.8)

# Overlay replica positions
for rb in band_data['replica_bands']:
    marker = 's' if rb['order'] == -1 else '^'
    color = 'blue' if rb['order'] == -1 else 'red'
    ax.plot(rb['kx'], rb['energy'], marker, color=color, markersize=10,
            markerfacecolor='none', markeredgewidth=2,
            label=f"n={rb['order']:+d} replica")

# Draw horizontal lines for replica energies
dp_energy = dp[1]
ax.axhline(y=dp_energy - pump_energy_eV, color='blue', ls='--', alpha=0.5,
           label=f'$E_D - \\hbar\\omega$ = {dp_energy - pump_energy_eV:.3f} eV')
ax.axhline(y=dp_energy + pump_energy_eV, color='red', ls='--', alpha=0.5,
           label=f'$E_D + \\hbar\\omega$ = {dp_energy + pump_energy_eV:.3f} eV')
ax.axhline(y=dp_energy, color='gray', ls=':', alpha=0.5)

ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=12)
ax.set_title('(a) Band Dispersion with Floquet Replicas', fontsize=12)
ax.legend(fontsize=8, loc='upper left')

# (b) Energy distribution curves at specific kx
ax = axes[1]
# EDC at Dirac point kx
dp_kx = dp[0]
dp_idx = np.argmin(np.abs(kx - dp_kx))

edc_off = pump_off[:, dp_idx]
edc_on = pump_on_0[:, dp_idx]
edc_diff = edc_on - edc_off

ax.plot(energy, edc_off / edc_off.max(), 'b-', lw=1.5, label='Pump off', alpha=0.7)
ax.plot(energy, edc_on / edc_on.max(), 'r-', lw=1.5, label='Pump on', alpha=0.7)
ax.plot(energy, edc_diff / np.abs(edc_diff).max() * 0.3 + 1.2, 'g-', lw=1.5,
        label='Difference (×0.3, offset)')

# Mark expected replica energies
ax.axvline(x=dp_energy - pump_energy_eV, color='blue', ls='--', alpha=0.5,
           label=f'$E_D - \\hbar\\omega$')
ax.axvline(x=dp_energy + pump_energy_eV, color='red', ls='--', alpha=0.5,
           label=f'$E_D + \\hbar\\omega$')

ax.set_xlabel(r'$E - E_F$ (eV)', fontsize=12)
ax.set_ylabel('Normalized Intensity', fontsize=12)
ax.set_title(f'(b) EDC at $k_x$ = {dp_kx:.3f} Å⁻¹', fontsize=12)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_replica_analysis.png'), dpi=200)
plt.close()

# ============================================================
# 5. Figure 4: Polarization dependence
# ============================================================
print("Generating Figure 4: Polarization dependence...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Polar plot
ax = axes[0]
ax.remove()
ax = fig.add_subplot(1, 2, 1, projection='polar')

theta = pol_df['angle_radians'].values
intensity = pol_df['intensity'].values

# Extend to full circle by mirroring
theta_full = np.concatenate([theta, theta + np.pi])
intensity_full = np.concatenate([intensity, intensity])
# Sort
sort_idx = np.argsort(theta_full)
theta_full = theta_full[sort_idx]
intensity_full = intensity_full[sort_idx]
# Close the loop
theta_full = np.append(theta_full, theta_full[0])
intensity_full = np.append(intensity_full, intensity_full[0])

ax.plot(theta_full, intensity_full, 'bo-', lw=2, markersize=5)
ax.fill(theta_full, intensity_full, alpha=0.15, color='blue')
ax.set_title('(a) Polar Plot of Replica\nBand Intensity', fontsize=12, pad=15)
ax.set_rticks([])

# (b) Cartesian plot with fit
ax = axes[1]
theta_deg = pol_df['angle_degrees'].values
ax.plot(theta_deg, intensity, 'ko', markersize=8, label='Measured')

# Fit: I(θ) = A + B*cos²(θ - θ₀)
# For Volkov final states, expect weak polarization dependence
def cos2_model(theta_rad, A, B, theta0):
    return A + B * np.cos(theta_rad - theta0)**2

theta_rad = pol_df['angle_radians'].values
popt, pcov = curve_fit(cos2_model, theta_rad, intensity,
                       p0=[intensity.mean(), 0.01, 0])
perr = np.sqrt(np.diag(pcov))

theta_fine = np.linspace(0, np.pi, 200)
ax.plot(np.degrees(theta_fine), cos2_model(theta_fine, *popt), 'r-', lw=2,
        label=f'Fit: $A + B\\cos^2(\\theta - \\theta_0)$\n'
              f'$A = {popt[0]:.4f} \\pm {perr[0]:.4f}$\n'
              f'$B = {popt[1]:.4f} \\pm {perr[1]:.4f}$\n'
              f'$\\theta_0 = {np.degrees(popt[2]):.1f}°$')

ax.set_xlabel(r'Pump Polarization Angle $\theta_p$ (°)', fontsize=12)
ax.set_ylabel('Replica Band Intensity (arb. units)', fontsize=12)
ax.set_title('(b) Polarization Dependence of Replica Band', fontsize=12)
ax.legend(fontsize=9)

# Calculate anisotropy
I_max = intensity.max()
I_min = intensity.min()
anisotropy = (I_max - I_min) / (I_max + I_min)
ax.text(0.95, 0.05, f'Anisotropy: {anisotropy:.4f}\n(~isotropic)',
        transform=ax.transAxes, ha='right', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_polarization_dependence.png'), dpi=200)
plt.close()

# ============================================================
# 6. Figure 5: Multi-angle comparison
# ============================================================
print("Generating Figure 5: Multi-angle pump-on spectra...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

# First panel: pump off
ax = axes[0]
im = ax.imshow(pump_off, aspect='auto', origin='lower', extent=extent,
               cmap=arpes_cmap, vmin=np.percentile(pump_off, 5),
               vmax=np.percentile(pump_off, 99))
ax.set_title('Pump Off', fontsize=11)
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=9)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=9)

# Remaining panels: pump on at different angles
for i, angle in enumerate(pol_angles):
    ax = axes[i+1]
    data = pump_on[angle]
    im = ax.imshow(data, aspect='auto', origin='lower', extent=extent,
                   cmap=arpes_cmap, vmin=np.percentile(data, 5),
                   vmax=np.percentile(data, 99))
    ax.set_title(f'$\\theta_p = {angle}°$', fontsize=11)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=9)
    ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=9)

    # Mark replica positions
    for rb in band_data['replica_bands']:
        ax.plot(rb['kx'], rb['energy'], 'r+', markersize=6, markeredgewidth=1)

plt.suptitle('tr-ARPES Spectra at Different Pump Polarization Angles', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_multi_angle_spectra.png'), dpi=200,
            bbox_inches='tight')
plt.close()

# ============================================================
# 7. Figure 6: Difference maps for all angles
# ============================================================
print("Generating Figure 6: Difference maps...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, angle in enumerate(pol_angles):
    ax = axes[i]
    diff = pump_on[angle] - pump_off
    diff_s = gaussian_filter(diff, sigma=1.0)
    vmax_d = np.percentile(np.abs(diff_s), 98)
    im = ax.imshow(diff_s, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    ax.set_title(f'$\\Delta I$ ($\\theta_p = {angle}°$)', fontsize=11)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=9)
    ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=9)

    for rb in band_data['replica_bands']:
        ax.plot(rb['kx'], rb['energy'], 'ko', markersize=6, markerfacecolor='none',
                markeredgewidth=1)

# Hide last panel
axes[7].axis('off')

plt.suptitle('Difference Maps (Pump On − Pump Off) at Various Polarizations', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_difference_maps.png'), dpi=200,
            bbox_inches='tight')
plt.close()

# ============================================================
# 8. Figure 7: Schematic of Floquet-Bloch mechanism
# ============================================================
print("Generating Figure 7: Floquet-Bloch schematic...")

fig, ax = plt.subplots(figsize=(7, 7))

# Draw Dirac cone
k_cone = np.linspace(-0.2, 0.2, 200)
v_F = 5.5  # eV·Å (approximate Fermi velocity for graphene)
E_upper = v_F * np.abs(k_cone) + dp[1]
E_lower = -v_F * np.abs(k_cone) + dp[1]

ax.plot(k_cone, E_upper, 'k-', lw=2.5, label='Main Dirac cone')
ax.plot(k_cone, E_lower, 'k-', lw=2.5)

# Draw replica cones (n = ±1)
hw = pump_energy_eV
E_upper_p1 = v_F * np.abs(k_cone) + dp[1] + hw
E_lower_p1 = -v_F * np.abs(k_cone) + dp[1] + hw
E_upper_m1 = v_F * np.abs(k_cone) + dp[1] - hw
E_lower_m1 = -v_F * np.abs(k_cone) + dp[1] - hw

ax.plot(k_cone, E_upper_p1, 'r--', lw=1.5, alpha=0.7, label=f'n=+1 replica ($+\\hbar\\omega$)')
ax.plot(k_cone, E_lower_p1, 'r--', lw=1.5, alpha=0.7)
ax.plot(k_cone, E_upper_m1, 'b--', lw=1.5, alpha=0.7, label=f'n=−1 replica ($-\\hbar\\omega$)')
ax.plot(k_cone, E_lower_m1, 'b--', lw=1.5, alpha=0.7)

# Mark Dirac point
ax.plot(0, dp[1], 'ko', markersize=8)
ax.annotate('$E_D$', xy=(0, dp[1]), xytext=(0.03, dp[1]-0.02), fontsize=12)

# Mark photon energy arrows
ax.annotate('', xy=(0.12, dp[1]+hw), xytext=(0.12, dp[1]),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.14, dp[1]+hw/2, f'$\\hbar\\omega$ = {hw} eV', fontsize=11, color='green')

# Mark Fermi level
ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
ax.text(0.18, 0.01, '$E_F$', fontsize=11, color='gray')

# Highlight replica band regions
for rb in band_data['replica_bands']:
    color = 'blue' if rb['order'] == -1 else 'red'
    ax.plot(rb['kx'], rb['energy'], 'o', color=color, markersize=10,
            markerfacecolor='none', markeredgewidth=2)

ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=13)
ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=13)
ax.set_title('Floquet-Bloch Band Structure\nof Monolayer Graphene', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.6, 0.6)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_floquet_schematic.png'), dpi=200)
plt.close()

# ============================================================
# 9. Save quantitative results
# ============================================================
print("\nSaving quantitative results...")

results = {
    'pump_parameters': {
        'energy_eV': pump_energy_eV,
        'wavelength_um': pump_wavelength,
        'sample': 'monolayer_epitaxial_graphene'
    },
    'dirac_point': {
        'kx': dp[0],
        'energy_eV': dp[1]
    },
    'replica_bands': band_data['replica_bands'],
    'expected_replica_energies': {
        'n_minus_1': dp[1] - pump_energy_eV,
        'n_plus_1': dp[1] + pump_energy_eV
    },
    'measured_replica_energies': {
        'n_minus_1': band_data['replica_bands'][0]['energy'],
        'n_plus_1': band_data['replica_bands'][2]['energy']
    },
    'energy_shift_verification': {
        'expected_shift': pump_energy_eV,
        'measured_shift_minus': abs(dp[1] - band_data['replica_bands'][0]['energy']),
        'measured_shift_plus': abs(band_data['replica_bands'][2]['energy'] - dp[1]),
    },
    'polarization_fit': {
        'A': float(popt[0]),
        'B': float(popt[1]),
        'theta0_deg': float(np.degrees(popt[2])),
        'A_err': float(perr[0]),
        'B_err': float(perr[1]),
        'anisotropy': float(anisotropy),
    },
    'intensity_statistics': {
        'pump_off_mean': float(pump_off.mean()),
        'pump_on_mean': float(pump_on_0.mean()),
        'difference_mean': float((pump_on_0 - pump_off).mean()),
    }
}

with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
print(f"Dirac point: kx = {dp[0]:.3f} Å⁻¹, E = {dp[1]:.3f} eV")
print(f"Pump photon energy: {pump_energy_eV} eV")
print(f"Expected replica energies: n=-1 at {dp[1]-pump_energy_eV:.3f} eV, "
      f"n=+1 at {dp[1]+pump_energy_eV:.3f} eV")
print(f"Measured replica energies: n=-1 at {band_data['replica_bands'][0]['energy']:.3f} eV, "
      f"n=+1 at {band_data['replica_bands'][2]['energy']:.3f} eV")
print(f"Polarization anisotropy: {anisotropy:.4f} (near-isotropic → Volkov mechanism)")
print(f"Fit: A={popt[0]:.4f}±{perr[0]:.4f}, B={popt[1]:.4f}±{perr[1]:.4f}")
print("\nAll figures saved to report/images/")
print("Results saved to outputs/analysis_results.json")

h5.close()
print("Done.")
