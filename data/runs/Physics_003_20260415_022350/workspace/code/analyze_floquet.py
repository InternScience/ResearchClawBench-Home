#!/usr/bin/env python3
"""
Floquet-Bloch States Analysis in Monolayer Graphene
tr-ARPES data analysis for mid-infrared pump excitation
"""

import numpy as np
import h5py
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

# Create output directories
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================
print("Loading data...")

# Load raw tr-ARPES data
with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy_axis = f['energy_axis'][:]
    kx_axis = f['kx_axis'][:]
    polarization_angles = f['polarization_angles'][:]
    pump_off_spectrum = f['pump_off_spectrum'][:]
    time_delays = f['time_delays'][:]
    
    # Load pump-on spectra for different polarization angles
    pump_on_spectra = {}
    for angle in polarization_angles:
        key = f'pump_on_angle_{angle}'
        pump_on_spectra[int(angle)] = f[key][:]

# Load processed band data
with open('data/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

# Load polarization dependence data
pol_data = []
with open('data/polarization_dependence_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pol_data.append({
            'angle_deg': float(row['angle_degrees']),
            'angle_rad': float(row['angle_radians']),
            'intensity': float(row['intensity']),
            'target_energy': float(row['target_energy']),
            'target_kx': float(row['target_kx'])
        })

print(f"Energy axis: {energy_axis.min():.3f} to {energy_axis.max():.3f} eV ({len(energy_axis)} points)")
print(f"Kx axis: {kx_axis.min():.3f} to {kx_axis.max():.3f} Å⁻¹ ({len(kx_axis)} points)")
print(f"Polarization angles: {polarization_angles}")
print(f"Time delays: {time_delays}")
print(f"Dirac point: E={band_data['dirac_point'][0]:.3f} eV, kx={band_data['dirac_point'][1]:.3f} Å⁻¹")
print(f"Number of replica bands: {len(band_data['replica_bands'])}")

# =============================================================================
# 2. Figure 1: ARPES Spectra Overview (Pump-off vs Pump-on)
# =============================================================================
print("\nGenerating Figure 1: ARPES Spectra Overview...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Pump-off spectrum
im1 = axes[0].pcolormesh(kx_axis, energy_axis, pump_off_spectrum, 
                          shading='auto', cmap='hot')
axes[0].set_xlabel('kx (Å⁻¹)')
axes[0].set_ylabel('Energy (eV)')
axes[0].set_title('Pump-off Spectrum')
plt.colorbar(im1, ax=axes[0], label='Intensity (arb. units)')

# Pump-on spectrum (0° polarization)
im2 = axes[1].pcolormesh(kx_axis, energy_axis, pump_on_spectra[0], 
                          shading='auto', cmap='hot')
axes[1].set_xlabel('kx (Å⁻¹)')
axes[1].set_ylabel('Energy (eV)')
axes[1].set_title('Pump-on Spectrum (θ = 0°)')
plt.colorbar(im2, ax=axes[1], label='Intensity (arb. units)')

# Difference map (pump-on minus pump-off)
diff_spectrum = pump_on_spectra[0] - pump_off_spectrum
vmax = np.max(np.abs(diff_spectrum))
im3 = axes[2].pcolormesh(kx_axis, energy_axis, diff_spectrum, 
                          shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_xlabel('kx (Å⁻¹)')
axes[2].set_ylabel('Energy (eV)')
axes[2].set_title('Difference (Pump-on - Pump-off)')
plt.colorbar(im3, ax=axes[2], label='Δ Intensity')

# Mark Dirac point
for ax in axes:
    ax.axhline(y=band_data['dirac_point'][0], color='cyan', linestyle='--', alpha=0.5, label='Dirac point')
    ax.axvline(x=band_data['dirac_point'][1], color='cyan', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/figure1_arpes_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure1_arpes_overview.png")

# =============================================================================
# 3. Figure 2: Replica Bands Identification
# =============================================================================
print("\nGenerating Figure 2: Replica Bands...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot pump-on spectrum with replica band positions marked
im = axes[0].pcolormesh(kx_axis, energy_axis, pump_on_spectra[0], 
                         shading='auto', cmap='hot')
axes[0].set_xlabel('kx (Å⁻¹)')
axes[0].set_ylabel('Energy (eV)')
axes[0].set_title('Pump-on Spectrum with Replica Bands')
plt.colorbar(im, ax=axes[0], label='Intensity (arb. units)')

# Mark replica band positions
colors = ['cyan', 'lime', 'yellow', 'magenta']
for i, rb in enumerate(band_data['replica_bands']):
    label = f"Order {rb['order']}"
    axes[0].plot(rb['kx'], rb['energy'], 'o', color=colors[i % len(colors)], 
                 markersize=10, markeredgecolor='white', markeredgewidth=1.5, label=label)

# Mark Dirac point
axes[0].plot(band_data['dirac_point'][1], band_data['dirac_point'][0], '*', 
             color='white', markersize=15, label='Dirac point')
axes[0].legend(loc='upper right', fontsize=8)

# Plot difference map with enhanced visibility
diff_spectrum = pump_on_spectra[0] - pump_off_spectrum
vmax = np.max(np.abs(diff_spectrum)) * 0.5  # Scale for better visibility
im2 = axes[1].pcolormesh(kx_axis, energy_axis, diff_spectrum, 
                          shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_xlabel('kx (Å⁻¹)')
axes[1].set_ylabel('Energy (eV)')
axes[1].set_title('Difference Map (Enhanced)')
plt.colorbar(im2, ax=axes[1], label='Δ Intensity')

# Mark photon energy offset
photon_energy = 0.248  # 5 μm ≈ 248 meV
axes[1].axhline(y=band_data['dirac_point'][0] + photon_energy, color='cyan', 
                linestyle=':', alpha=0.7, label=f'+ℏω (+{photon_energy:.3f} eV)')
axes[1].axhline(y=band_data['dirac_point'][0] - photon_energy, color='lime', 
                linestyle=':', alpha=0.7, label=f'-ℏω (-{photon_energy:.3f} eV)')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('report/images/figure2_replica_bands.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure2_replica_bands.png")

# =============================================================================
# 4. Figure 3: Energy Distribution Curves (EDCs)
# =============================================================================
print("\nGenerating Figure 3: Energy Distribution Curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Find indices for key k-points
kx_zero_idx = np.argmin(np.abs(kx_axis - 0))
kx_dirac_idx = np.argmin(np.abs(kx_axis - band_data['dirac_point'][1]))

# EDC at kx = 0
axes[0, 0].plot(energy_axis, pump_off_spectrum[:, kx_zero_idx], 'b-', 
                label='Pump-off', linewidth=2)
axes[0, 0].plot(energy_axis, pump_on_spectra[0][:, kx_zero_idx], 'r-', 
                label='Pump-on', linewidth=2)
axes[0, 0].set_xlabel('Energy (eV)')
axes[0, 0].set_ylabel('Intensity (arb. units)')
axes[0, 0].set_title('EDC at kx = 0')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# EDC at Dirac point kx
axes[0, 1].plot(energy_axis, pump_off_spectrum[:, kx_dirac_idx], 'b-', 
                label='Pump-off', linewidth=2)
axes[0, 1].plot(energy_axis, pump_on_spectra[0][:, kx_dirac_idx], 'r-', 
                label='Pump-on', linewidth=2)
axes[0, 1].set_xlabel('Energy (eV)')
axes[0, 1].set_ylabel('Intensity (arb. units)')
axes[0, 1].set_title(f'EDC at kx = {kx_axis[kx_dirac_idx]:.3f} Å⁻¹')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# EDC at replica band kx (positive)
rb_pos = band_data['replica_bands'][2]  # Order +1, positive kx
rb_pos_idx = np.argmin(np.abs(kx_axis - rb_pos['kx']))
axes[1, 0].plot(energy_axis, pump_off_spectrum[:, rb_pos_idx], 'b-', 
                label='Pump-off', linewidth=2)
axes[1, 0].plot(energy_axis, pump_on_spectra[0][:, rb_pos_idx], 'r-', 
                label='Pump-on', linewidth=2)
axes[1, 0].axvline(x=rb_pos['energy'], color='green', linestyle='--', 
                    label=f'Replica at {rb_pos["energy"]:.3f} eV')
axes[1, 0].set_xlabel('Energy (eV)')
axes[1, 0].set_ylabel('Intensity (arb. units)')
axes[1, 0].set_title(f'EDC at replica kx = {rb_pos["kx"]:.3f} Å⁻¹')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Difference EDC at kx = 0
diff_edc = pump_on_spectra[0][:, kx_zero_idx] - pump_off_spectrum[:, kx_zero_idx]
axes[1, 1].plot(energy_axis, diff_edc, 'g-', linewidth=2)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].axvline(x=photon_energy, color='cyan', linestyle='--', 
                    label=f'+ℏω ({photon_energy:.3f} eV)')
axes[1, 1].axvline(x=-photon_energy, color='lime', linestyle='--', 
                    label=f'-ℏω ({-photon_energy:.3f} eV)')
axes[1, 1].set_xlabel('Energy (eV)')
axes[1, 1].set_ylabel('Δ Intensity')
axes[1, 1].set_title('Difference EDC at kx = 0')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure3_edcs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure3_edcs.png")

# =============================================================================
# 5. Figure 4: Momentum Distribution Curves (MDCs)
# =============================================================================
print("\nGenerating Figure 4: Momentum Distribution Curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Find indices for key energies
energy_zero_idx = np.argmin(np.abs(energy_axis - 0))
energy_dirac_idx = np.argmin(np.abs(energy_axis - band_data['dirac_point'][0]))
energy_replica_pos_idx = np.argmin(np.abs(energy_axis - 0.205))  # Order +1 replica
energy_replica_neg_idx = np.argmin(np.abs(energy_axis - -0.291))  # Order -1 replica

# MDC at E = 0
axes[0, 0].plot(kx_axis, pump_off_spectrum[energy_zero_idx, :], 'b-', 
                label='Pump-off', linewidth=2)
axes[0, 0].plot(kx_axis, pump_on_spectra[0][energy_zero_idx, :], 'r-', 
                label='Pump-on', linewidth=2)
axes[0, 0].set_xlabel('kx (Å⁻¹)')
axes[0, 0].set_ylabel('Intensity (arb. units)')
axes[0, 0].set_title('MDC at E = 0')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MDC at Dirac point energy
axes[0, 1].plot(kx_axis, pump_off_spectrum[energy_dirac_idx, :], 'b-', 
                label='Pump-off', linewidth=2)
axes[0, 1].plot(kx_axis, pump_on_spectra[0][energy_dirac_idx, :], 'r-', 
                label='Pump-on', linewidth=2)
axes[0, 1].set_xlabel('kx (Å⁻¹)')
axes[0, 1].set_ylabel('Intensity (arb. units)')
axes[0, 1].set_title(f'MDC at E = {energy_axis[energy_dirac_idx]:.3f} eV')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# MDC at replica band energy (positive)
axes[1, 0].plot(kx_axis, pump_off_spectrum[energy_replica_pos_idx, :], 'b-', 
                label='Pump-off', linewidth=2)
axes[1, 0].plot(kx_axis, pump_on_spectra[0][energy_replica_pos_idx, :], 'r-', 
                label='Pump-on', linewidth=2)
# Mark expected replica positions
for rb in band_data['replica_bands']:
    if rb['order'] == 1:
        axes[1, 0].axvline(x=rb['kx'], color='green', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('kx (Å⁻¹)')
axes[1, 0].set_ylabel('Intensity (arb. units)')
axes[1, 0].set_title(f'MDC at E = {energy_axis[energy_replica_pos_idx]:.3f} eV (Replica)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# MDC at replica band energy (negative)
axes[1, 1].plot(kx_axis, pump_off_spectrum[energy_replica_neg_idx, :], 'b-', 
                label='Pump-off', linewidth=2)
axes[1, 1].plot(kx_axis, pump_on_spectra[0][energy_replica_neg_idx, :], 'r-', 
                label='Pump-on', linewidth=2)
# Mark expected replica positions
for rb in band_data['replica_bands']:
    if rb['order'] == -1:
        axes[1, 1].axvline(x=rb['kx'], color='green', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('kx (Å⁻¹)')
axes[1, 1].set_ylabel('Intensity (arb. units)')
axes[1, 1].set_title(f'MDC at E = {energy_axis[energy_replica_neg_idx]:.3f} eV (Replica)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure4_mdcs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure4_mdcs.png")

# =============================================================================
# 6. Figure 5: Polarization Dependence
# =============================================================================
print("\nGenerating Figure 5: Polarization Dependence...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Extract angles and intensities
angles_deg = [d['angle_deg'] for d in pol_data]
intensities = [d['intensity'] for d in pol_data]

# Plot polarization dependence
axes[0].plot(angles_deg, intensities, 'bo-', markersize=8, linewidth=2)
axes[0].set_xlabel('Polarization Angle (degrees)')
axes[0].set_ylabel('Replica Band Intensity (arb. units)')
axes[0].set_title('Polarization Dependence of Replica Band')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-10, 190)

# Fit to cos(2θ) + offset (expected for linear polarization dependence)
from scipy.optimize import curve_fit

def pol_func(theta, A, B, C):
    """Polarization dependence: A + B*cos(2*(theta-theta0))"""
    return A + B * np.cos(2 * np.radians(theta - C))

angles_rad = np.array([d['angle_rad'] for d in pol_data])
intensities_arr = np.array(intensities)

try:
    popt, pcov = curve_fit(pol_func, angles_deg, intensities, p0=[0.5, 0.01, 0])
    theta_fit = np.linspace(0, 180, 100)
    axes[0].plot(theta_fit, pol_func(theta_fit, *popt), 'r--', 
                 label=f'Fit: {popt[0]:.3f} + {popt[1]:.3f}·cos(2(θ-{popt[2]:.1f}°))')
    axes[0].legend()
except:
    print("Warning: Polarization fit failed")

# Polar plot
ax_polar = fig.add_subplot(122, projection='polar')
angles_rad_plot = np.array(angles_deg) * np.pi / 180
ax_polar.plot(angles_rad_plot, intensities, 'bo-', markersize=8, linewidth=2)
ax_polar.set_title('Polar Plot of Polarization Dependence', pad=20)
ax_polar.set_ylim(0.49, 0.51)

plt.tight_layout()
plt.savefig('report/images/figure5_polarization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure5_polarization.png")

# =============================================================================
# 7. Figure 6: Band Dispersion with Floquet Sidebands
# =============================================================================
print("\nGenerating Figure 6: Band Dispersion with Floquet Sidebands...")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Extract band dispersion from processed data
band_disp = band_data['band_dispersion']
band_energies = [p['energy'] for p in band_disp]
band_kx = [p['kx'] for p in band_disp]
band_intensities = [p['intensity'] for p in band_disp]

# Plot as scatter with intensity coloring
scatter = ax.scatter(band_kx, band_energies, c=band_intensities, cmap='hot', 
                      s=20, alpha=0.7)
plt.colorbar(scatter, ax=ax, label='Intensity (arb. units)')

# Mark Dirac point
ax.plot(band_data['dirac_point'][1], band_data['dirac_point'][0], '*', 
        color='cyan', markersize=20, label='Dirac point')

# Mark replica bands
for rb in band_data['replica_bands']:
    color = 'lime' if rb['order'] > 0 else 'magenta'
    ax.plot(rb['kx'], rb['energy'], 'o', color=color, markersize=12, 
            markeredgecolor='white', markeredgewidth=2,
            label=f'Replica order {rb["order"]}')

# Draw expected Floquet sideband structure
# Main Dirac cone (linear dispersion)
kx_theory = np.linspace(-0.1, 0.1, 100)
v_F = 1.0  # Fermi velocity (in appropriate units)
E_dirac = band_data['dirac_point'][0]
kx_dirac = band_data['dirac_point'][1]

# Main band
E_main = E_dirac + v_F * np.abs(kx_theory - kx_dirac)
ax.plot(kx_theory, E_main, 'c--', alpha=0.5, label='Main Dirac cone')

# Floquet sidebands at ±ℏω
E_upper = E_main + photon_energy
E_lower = E_main - photon_energy
ax.plot(kx_theory, E_upper, 'y--', alpha=0.5, label=f'+ℏω sideband')
ax.plot(kx_theory, E_lower, 'g--', alpha=0.5, label=f'-ℏω sideband')

ax.set_xlabel('kx (Å⁻¹)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Band Dispersion with Floquet Sidebands')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure6_floquet_bands.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure6_floquet_bands.png")

# =============================================================================
# 8. Figure 7: Multi-angle comparison
# =============================================================================
print("\nGenerating Figure 7: Multi-angle Comparison...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Plot pump-off
im = axes[0, 0].pcolormesh(kx_axis, energy_axis, pump_off_spectrum, 
                            shading='auto', cmap='hot')
axes[0, 0].set_title('Pump-off')
axes[0, 0].set_xlabel('kx (Å⁻¹)')
axes[0, 0].set_ylabel('Energy (eV)')

# Plot pump-on for each angle
for i, angle in enumerate(polarization_angles):
    row = (i + 1) // 4
    col = (i + 1) % 4
    if i < 7:
        im = axes[row, col].pcolormesh(kx_axis, energy_axis, pump_on_spectra[int(angle)], 
                                        shading='auto', cmap='hot')
        axes[row, col].set_title(f'θ = {angle}°')
        axes[row, col].set_xlabel('kx (Å⁻¹)')
        if col == 0:
            axes[row, col].set_ylabel('Energy (eV)')

plt.tight_layout()
plt.savefig('report/images/figure7_multi_angle.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure7_multi_angle.png")

# =============================================================================
# 9. Figure 8: Time-resolved evolution
# =============================================================================
print("\nGenerating Figure 8: Analysis Summary...")

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig)

# Summary statistics
replica_intensities = [rb['intensity'] for rb in band_data['replica_bands']]
replica_orders = [rb['order'] for rb in band_data['replica_bands']]
replica_energies = [rb['energy'] for rb in band_data['replica_bands']]

# Replica band intensities by order
ax1 = fig.add_subplot(gs[0, 0])
colors = ['magenta' if o < 0 else 'lime' for o in replica_orders]
ax1.bar(range(len(replica_orders)), replica_intensities, color=colors)
ax1.set_xlabel('Replica Band Index')
ax1.set_ylabel('Intensity (arb. units)')
ax1.set_title('Replica Band Intensities')
ax1.set_xticks(range(len(replica_orders)))
ax1.set_xticklabels([f'n={o}' for o in replica_orders])

# Energy offsets from Dirac point
ax2 = fig.add_subplot(gs[0, 1])
energy_offsets = [e - band_data['dirac_point'][0] for e in replica_energies]
ax2.bar(range(len(energy_offsets)), energy_offsets, color=colors)
ax2.axhline(y=photon_energy, color='cyan', linestyle='--', label=f'±ℏω = ±{photon_energy:.3f} eV')
ax2.axhline(y=-photon_energy, color='cyan', linestyle='--')
ax2.set_xlabel('Replica Band Index')
ax2.set_ylabel('Energy Offset from Dirac Point (eV)')
ax2.set_title('Replica Band Energy Offsets')
ax2.set_xticks(range(len(replica_orders)))
ax2.set_xticklabels([f'n={o}' for o in replica_orders])
ax2.legend()

# Momentum positions
ax3 = fig.add_subplot(gs[0, 2])
kx_positions = [rb['kx'] for rb in band_data['replica_bands']]
ax3.bar(range(len(kx_positions)), kx_positions, color=colors)
ax3.set_xlabel('Replica Band Index')
ax3.set_ylabel('kx (Å⁻¹)')
ax3.set_title('Replica Band Momentum Positions')
ax3.set_xticks(range(len(replica_orders)))
ax3.set_xticklabels([f'n={o}' for o in replica_orders])

# Difference spectrum at key energies
ax4 = fig.add_subplot(gs[1, :])
diff_spectrum_full = pump_on_spectra[0] - pump_off_spectrum
plot_angles = [0, 60, 120, 180]
for i, angle in enumerate(plot_angles):
    diff = pump_on_spectra[int(angle)] - pump_off_spectrum
    edc_diff = np.mean(diff[:, 60:90], axis=1)  # Average over central kx region
    ax4.plot(energy_axis, edc_diff + i*0.01, label=f'θ = {angle}°', linewidth=1.5)
ax4.axvline(x=photon_energy, color='gray', linestyle=':', alpha=0.5)
ax4.axvline(x=-photon_energy, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Energy (eV)')
ax4.set_ylabel('Δ Intensity (offset)')
ax4.set_title('Difference EDCs for Different Polarizations')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Polarization dependence summary
ax5 = fig.add_subplot(gs[2, :2])
ax5.errorbar(angles_deg, intensities, fmt='o-', capsize=5, markersize=8)
ax5.set_xlabel('Polarization Angle (degrees)')
ax5.set_ylabel('Replica Band Intensity')
ax5.set_title('Polarization Dependence Summary')
ax5.grid(True, alpha=0.3)

# Key parameters text
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
info_text = f"""
Key Parameters:
---------------
Pump wavelength: 5 μm
Photon energy: {photon_energy*1000:.1f} meV
Dirac point: E = {band_data['dirac_point'][0]:.3f} eV
             kx = {band_data['dirac_point'][1]:.3f} Å⁻¹

Replica bands: {len(band_data['replica_bands'])}
  Order -1: {sum(1 for rb in band_data['replica_bands'] if rb['order']==-1)} bands
  Order +1: {sum(1 for rb in band_data['replica_bands'] if rb['order']==1)} bands

Mean intensity variation: 
  {np.std(intensities)/np.mean(intensities)*100:.2f}%
"""
ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/figure8_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure8_summary.png")

# =============================================================================
# 10. Save analysis results
# =============================================================================
print("\nSaving analysis results...")

results = {
    'photon_energy_eV': photon_energy,
    'photon_energy_meV': photon_energy * 1000,
    'pump_wavelength_um': 5.0,
    'dirac_point': {
        'energy_eV': band_data['dirac_point'][0],
        'kx_per_angstrom': band_data['dirac_point'][1]
    },
    'replica_bands': [],
    'polarization_analysis': {
        'angles_deg': angles_deg,
        'intensities': intensities,
        'mean_intensity': np.mean(intensities),
        'std_intensity': np.std(intensities),
        'relative_variation': np.std(intensities) / np.mean(intensities) * 100
    }
}

for rb in band_data['replica_bands']:
    results['replica_bands'].append({
        'order': rb['order'],
        'energy_eV': rb['energy'],
        'kx_per_angstrom': rb['kx'],
        'intensity': rb['intensity'],
        'energy_offset_from_dirac_eV': rb['energy'] - band_data['dirac_point'][0]
    })

with open('outputs/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Saved: outputs/analysis_results.json")

# =============================================================================
# 11. Compute additional metrics
# =============================================================================
print("\nComputing additional metrics...")

# Compute replica band intensity ratios
main_band_intensity = np.max(pump_off_spectrum)
replica_intensity_ratio = np.mean(replica_intensities) / main_band_intensity

# Compute energy gap at Dirac point (if any)
dirac_idx_e = np.argmin(np.abs(energy_axis - band_data['dirac_point'][0]))
dirac_idx_k = np.argmin(np.abs(kx_axis - band_data['dirac_point'][1]))
window = 5
edc_dirac_off = pump_off_spectrum[dirac_idx_e-window:dirac_idx_e+window, dirac_idx_k]
edc_dirac_on = pump_on_spectra[0][dirac_idx_e-window:dirac_idx_e+window, dirac_idx_k]

metrics = {
    'main_band_max_intensity': float(main_band_intensity),
    'replica_mean_intensity': float(np.mean(replica_intensities)),
    'replica_to_main_ratio': float(replica_intensity_ratio),
    'energy_resolution_eV': float(energy_axis[1] - energy_axis[0]),
    'momentum_resolution_per_angstrom': float(kx_axis[1] - kx_axis[0]),
    'num_energy_points': len(energy_axis),
    'num_momentum_points': len(kx_axis),
    'num_polarization_angles': len(polarization_angles),
    'energy_range_eV': [float(energy_axis.min()), float(energy_axis.max())],
    'momentum_range_per_angstrom': [float(kx_axis.min()), float(kx_axis.max())]
}

with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Saved: outputs/metrics.json")
print("\nAnalysis complete!")
print(f"\nKey Results:")
print(f"  Photon energy: {photon_energy*1000:.1f} meV")
print(f"  Number of replica bands: {len(band_data['replica_bands'])}")
print(f"  Replica/main intensity ratio: {replica_intensity_ratio:.4f}")
print(f"  Polarization variation: {results['polarization_analysis']['relative_variation']:.2f}%")
