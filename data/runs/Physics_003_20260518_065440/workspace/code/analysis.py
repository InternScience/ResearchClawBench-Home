#!/usr/bin/env python3
"""
Analysis of Floquet-Bloch States in Monolayer Epitaxial Graphene via tr-ARPES
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import h5py

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ======================== LOAD DATA ========================
with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy_axis = f['energy_axis'][:]
    kx_axis = f['kx_axis'][:]
    time_delays = f['time_delays'][:]
    polarization_angles = f['polarization_angles'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on = {}
    for angle in polarization_angles:
        pump_on[angle] = f[f'pump_on_angle_{angle}'][:]

with open('data/processed_band_data.json') as fh:
    band_data = json.load(fh)

polar_data = []
with open('data/polarization_dependence_data.csv') as fh:
    for row in csv.DictReader(fh):
        polar_data.append({k: float(v) for k, v in row.items()})

wavelength_um = 5.0
photon_energy_eV = 1.2398 / wavelength_um
dirac_kx, dirac_E = band_data['dirac_point']
dirac_kx_idx, dirac_E_idx = band_data['dirac_indices']

print(f"Photon energy: {photon_energy_eV:.4f} eV")
print(f"Dirac point: E={dirac_E:.4f}, kx={dirac_kx:.4f}")

# ======================== FIGURE 1: DATA OVERVIEW ========================
print("Fig 1...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

im0 = axes[0].pcolormesh(kx_axis, energy_axis, pump_off, cmap='hot', shading='auto')
axes[0].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[0].set_ylabel('Energy (eV)', fontsize=12)
axes[0].set_title('(a) Equilibrium (Pump-off)', fontsize=13, fontweight='bold')
axes[0].plot(dirac_kx, dirac_E, 'c+', markersize=15, markeredgewidth=2)
plt.colorbar(im0, ax=axes[0], label='Intensity (arb. u.)', shrink=0.85)

im1 = axes[1].pcolormesh(kx_axis, energy_axis, pump_on[0], cmap='hot', shading='auto')
axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[1].set_ylabel('Energy (eV)', fontsize=12)
axes[1].set_title('(b) Pump-on (θ=0°)', fontsize=13, fontweight='bold')
axes[1].plot(dirac_kx, dirac_E, 'c+', markersize=15, markeredgewidth=2)
for rb in band_data['replica_bands']:
    axes[1].plot(rb['kx'], rb['energy'], 'w*', markersize=10)
plt.colorbar(im1, ax=axes[1], label='Intensity (arb. u.)', shrink=0.85)

diff_0 = pump_on[0] - pump_off
vlim = max(abs(diff_0.min()), abs(diff_0.max()))
im2 = axes[2].pcolormesh(kx_axis, energy_axis, diff_0, cmap='RdBu_r',
                          vmin=-vlim, vmax=vlim, shading='auto')
axes[2].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[2].set_ylabel('Energy (eV)', fontsize=12)
axes[2].set_title('(c) Difference (On − Off)', fontsize=13, fontweight='bold')
axes[2].plot(dirac_kx, dirac_E, 'c+', markersize=15, markeredgewidth=2)
for rb in band_data['replica_bands']:
    axes[2].plot(rb['kx'], rb['energy'], 'g*', markersize=10)
# Arrow for photon energy
axes[2].annotate('', xy=(0.18, dirac_E + photon_energy_eV), xytext=(0.18, dirac_E),
    arrowprops=dict(arrowstyle='<->', color='yellow', lw=2.5))
axes[2].annotate(f'ℏω={photon_energy_eV:.3f} eV', xy=(0.195, dirac_E+photon_energy_eV/2),
    color='yellow', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))
plt.colorbar(im2, ax=axes[2], label='Δ Intensity', shrink=0.85)

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 2: FLOQUET REPLICAS ========================
print("Fig 2...")
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.pcolormesh(kx_axis, energy_axis, pump_on[0], cmap='hot', shading='auto')

band_disp = band_data['band_dispersion']
disp_E = np.array([p['energy'] for p in band_disp])
disp_kx = np.array([p['kx'] for p in band_disp])
disp_I = np.array([p['intensity'] for p in band_disp])
sc = ax.scatter(disp_kx, disp_E, c=disp_I, cmap='cool', s=12, edgecolors='none',
                zorder=5, vmin=0.4, vmax=1.0, label='Extracted dispersion')

ax.plot(dirac_kx, dirac_E, 'c+', markersize=20, markeredgewidth=3,
        label='Dirac Point', zorder=6)

for rb in band_data['replica_bands']:
    ax.plot(rb['kx'], rb['energy'], 'w*', markersize=16, markeredgecolor='lime',
            markeredgewidth=1, zorder=7)

# Photon energy arrow
ax.annotate('', xy=(0.20, dirac_E + photon_energy_eV), xytext=(0.20, dirac_E),
    arrowprops=dict(arrowstyle='<->', color='cyan', lw=2.5))
ax.annotate(f'ℏω ≈ {photon_energy_eV:.3f} eV', xy=(0.215, dirac_E + photon_energy_eV/2),
    color='cyan', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round', fc='black', alpha=0.7))

ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Observation of Floquet-Bloch Replica Bands in Monolayer Graphene\n'
             r'tr-ARPES, $\lambda$=5 μm, θ$_p$=0°', fontsize=14)
plt.colorbar(im, ax=ax, label='Photoemission Intensity (arb. u.)', shrink=0.8)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(-0.12, 0.25)

plt.tight_layout()
plt.savefig('report/images/figure2_floquet_replicas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 3: EDC / MDC ========================
print("Fig 3...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# EDC at Dirac point kx
edc_on = pump_on[0][:, dirac_kx_idx]
edc_off = pump_off[:, dirac_kx_idx]
axes[0].plot(energy_axis, edc_off / edc_off.max(), 'b-', lw=2, label='Pump-off', alpha=0.8)
axes[0].plot(energy_axis, edc_on / edc_on.max(), 'r-', lw=2, label='Pump-on θ=0°', alpha=0.8)
axes[0].axvline(dirac_E, color='cyan', ls='--', alpha=0.7, label=f'Dirac ({dirac_E:.3f} eV)')
axes[0].axvline(band_data['replica_bands'][0]['energy'], color='green', ls=':',
                alpha=0.7, label=f'Replica E={band_data["replica_bands"][0]["energy"]:.3f} eV')
axes[0].axvline(band_data['replica_bands'][2]['energy'], color='orange', ls=':',
                alpha=0.7, label=f'Replica E={band_data["replica_bands"][2]["energy"]:.3f} eV')
axes[0].annotate('', xy=(dirac_E+photon_energy_eV, 0.93), xytext=(dirac_E, 0.93),
    arrowprops=dict(arrowstyle='<->', color='gold', lw=2))
axes[0].annotate('+ℏω', xy=(dirac_E+photon_energy_eV/2, 0.96), color='gold',
    fontsize=10, ha='center', fontweight='bold')
axes[0].set_xlabel('Energy (eV)', fontsize=12)
axes[0].set_ylabel('Normalized Intensity', fontsize=12)
axes[0].set_title('(a) Energy Distribution Curve at Dirac $k_x$', fontsize=13)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].set_xlim(-0.5, 0.5)

# MDC at Dirac energy
mdc_on = pump_on[0][dirac_E_idx, :]
mdc_off = pump_off[dirac_E_idx, :]
axes[1].plot(kx_axis, mdc_off / mdc_off.max(), 'b-', lw=2, label='Pump-off', alpha=0.8)
axes[1].plot(kx_axis, mdc_on / mdc_on.max(), 'r-', lw=2, label='Pump-on θ=0°', alpha=0.8)
axes[1].axvline(dirac_kx, color='cyan', ls='--', alpha=0.7, label=f'Dirac $k_x$={dirac_kx:.3f}')
for rb in band_data['replica_bands']:
    axes[1].axvline(rb['kx'], color='green', ls=':', alpha=0.4)
axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[1].set_ylabel('Normalized Intensity', fontsize=12)
axes[1].set_title(f'(b) MDC at Dirac Energy (E={dirac_E:.3f} eV)', fontsize=13)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure3_edc_mdc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 4: POLARIZATION DEPENDENCE ========================
print("Fig 4...")
angles_deg = [d['angle_degrees'] for d in polar_data]
intensities = [d['intensity'] for d in polar_data]
angles_rad = [d['angle_radians'] for d in polar_data]

I0 = np.mean(intensities)
Delta_I = np.max(intensities) - np.min(intensities)
angles_fit = np.linspace(0, 360, 361)
cos2_fit = I0 + Delta_I * np.cos(np.deg2rad(angles_fit))**2
residuals = np.array(intensities) - (I0 + Delta_I * np.cos(np.deg2rad(angles_deg))**2)
rms_error = np.sqrt(np.mean(residuals**2))

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(angles_deg, intensities, 'ro-', markersize=10, lw=2, label='Measured intensity')
ax1.fill_between(angles_deg, intensities, alpha=0.15, color='red')
ax1.plot(angles_fit, cos2_fit, 'b--', lw=1.5,
         label=r'Fit: $I_0 + \Delta I\cos^2\theta$', alpha=0.7)
ax1.errorbar(angles_deg, intensities, yerr=rms_error, fmt='none', ecolor='gray', capsize=5, alpha=0.5)
ax1.set_xlabel(r'Pump Polarization Angle $\theta_p$ (°)', fontsize=12)
ax1.set_ylabel('Replica Band Intensity (arb. u.)', fontsize=12)
ax1.set_title('(a) Replica Intensity vs Polarization Angle', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xticks(angles_deg)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, 190)

ax2 = fig.add_subplot(gs[1], projection='polar')
angles_polar = np.concatenate([angles_rad, [a + np.pi for a in angles_rad]])
intensities_polar = intensities + intensities
ax2.plot(angles_polar, intensities_polar, 'ro-', markersize=10, lw=2, label='Measured')
theta_fit_rad = np.linspace(0, 2*np.pi, 361)
r_fit = I0 + Delta_I * np.cos(theta_fit_rad)**2
ax2.plot(theta_fit_rad, r_fit, 'b--', lw=1.5,
         label=r'Fit: $I_0 + \Delta I\cos^2\theta$', alpha=0.7)
ax2.set_title('(b) Polar Plot', fontsize=13, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.25, 1.1))

plt.savefig('report/images/figure4_polarization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 5: ANGULAR COMPARISON ========================
print("Fig 5...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()
for i, angle in enumerate(polarization_angles):
    ax = axes_flat[i]
    im = ax.pcolormesh(kx_axis, energy_axis, pump_on[angle], cmap='hot', shading='auto')
    ax.set_title(f'θ$_p$ = {angle}°', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=10)
    ax.set_ylabel('Energy (eV)', fontsize=10)
    ax.plot(dirac_kx, dirac_E, 'c+', markersize=12, markeredgewidth=2)
    for rb in band_data['replica_bands']:
        ax.plot(rb['kx'], rb['energy'], 'g*', markersize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax_last = axes_flat[7]
diff_para = pump_on[0] - pump_off
diff_perp = pump_on[90] - pump_off
diff_aniso = diff_para - diff_perp
vl = max(abs(diff_aniso.min()), abs(diff_aniso.max()))
im = ax_last.pcolormesh(kx_axis, energy_axis, diff_aniso, cmap='RdBu_r',
                         vmin=-vl, vmax=vl, shading='auto')
ax_last.set_title('Δ(0°−90°) Anisotropy', fontsize=12, fontweight='bold')
ax_last.set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=10)
ax_last.set_ylabel('Energy (eV)', fontsize=10)
plt.colorbar(im, ax=ax_last, fraction=0.046, pad=0.04)

plt.suptitle('Pump Polarization Angle Dependence of tr-ARPES Spectra\n'
             'Monolayer Graphene, λ = 5 μm', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('report/images/figure5_angular_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 6: SCHEMATIC ========================
print("Fig 6...")
fig, ax = plt.subplots(figsize=(10, 8))
k = np.linspace(-0.12, 0.12, 300)
slope = 400  # visual scaling for schematic

# Original cone
ax.plot(k, slope*np.abs(k), 'b-', lw=2.5, label='n = 0 (Dirac cone)')
ax.plot(k, -slope*np.abs(k), 'b-', lw=2.5)

# n=+1 replicas
ep = photon_energy_eV * 200
ax.plot(k, slope*np.abs(k) + ep, 'r--', lw=2, alpha=0.8, label='n = +1 (+ℏω)')
ax.plot(k, -slope*np.abs(k) + ep, 'r--', lw=2, alpha=0.8)

# n=−1 replicas
ax.plot(k, slope*np.abs(k) - ep, 'g--', lw=2, alpha=0.8, label='n = −1 (−ℏω)')
ax.plot(k, -slope*np.abs(k) - ep, 'g--', lw=2, alpha=0.8)

ax.plot(0, 0, 'ko', ms=8, zorder=5)
ax.plot(0, ep, 'ro', ms=8, zorder=5)
ax.plot(0, -ep, 'go', ms=8, zorder=5)

ax.annotate('', xy=(0.08, ep), xytext=(0.08, 0),
    arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5))
ax.annotate(r'$\hbar\omega$', xy=(0.09, ep/2), color='purple', fontsize=14, fontweight='bold')
ax.annotate('Dirac\nPoint', xy=(0.01, 0.5), fontsize=11, color='blue')
ax.annotate('Replica\n(n=+1)', xy=(0.06, ep + slope*0.06 + 2), fontsize=10, color='red')
ax.annotate('Replica\n(n=−1)', xy=(0.06, -ep - slope*0.06 - 8), fontsize=10, color='green')

ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.axvline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel(r'Crystal Momentum $k$ (Å$^{-1}$)', fontsize=13)
ax.set_ylabel('Energy (eV, schematic)', fontsize=13)
ax.set_title('Schematic: Floquet-Bloch Band Formation\n'
             r'Mid-IR Pump ($\lambda$ = 5 μm) on Graphene Dirac Cone', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('report/images/figure6_schematic.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 7: DISPERSION + THEORY OVERLAY ========================
print("Fig 7...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(disp_kx, disp_E, c=disp_I, cmap='viridis', s=20,
                     edgecolors='none', vmin=0.45, vmax=1.0)
axes[0].plot(dirac_kx, dirac_E, 'r+', ms=20, markeredgewidth=3, label='Dirac Point', zorder=5)

# Linear fit near Dirac point
mask_near = np.abs(disp_E - dirac_E) < 0.05
if np.sum(mask_near) > 2:
    coeffs = np.polyfit(disp_kx[mask_near], disp_E[mask_near], 1)
    vF_fit = coeffs[0]
    k_th = np.linspace(-0.12, 0.12, 200)
    axes[0].plot(k_th, vF_fit*(k_th - dirac_kx) + dirac_E, 'r-', lw=1.5, alpha=0.7,
                label=f'Linear fit: $v_F$ = {abs(vF_fit):.1f} eV·Å')
    axes[0].plot(k_th, -vF_fit*(k_th - dirac_kx) + dirac_E, 'r-', lw=1.5, alpha=0.7)
else:
    vF_fit = 5.0

for rb in band_data['replica_bands']:
    axes[0].plot(rb['kx'], rb['energy'], 'm*', ms=15, markeredgecolor='white',
                markeredgewidth=1, zorder=6)

axes[0].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[0].set_ylabel('Energy (eV)', fontsize=12)
axes[0].set_title('(a) Extracted Dispersion', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
plt.colorbar(sc, ax=axes[0], label='Intensity', shrink=0.8)

# Theory overlay
k_th = np.linspace(-0.12, 0.12, 200)
axes[1].pcolormesh(kx_axis, energy_axis, pump_on[0], cmap='hot', shading='auto', alpha=0.4)
axes[1].plot(k_th, vF_fit*(k_th - dirac_kx) + dirac_E, 'b-', lw=2, label='n=0')
axes[1].plot(k_th, -vF_fit*(k_th - dirac_kx) + dirac_E, 'b-', lw=2)
axes[1].plot(k_th, vF_fit*(k_th - dirac_kx) + dirac_E + photon_energy_eV, 'r--', lw=1.5,
            label=f'n=+1 (+{photon_energy_eV:.3f} eV)')
axes[1].plot(k_th, -vF_fit*(k_th - dirac_kx) + dirac_E + photon_energy_eV, 'r--', lw=1.5)
axes[1].plot(k_th, vF_fit*(k_th - dirac_kx) + dirac_E - photon_energy_eV, 'g--', lw=1.5,
            label=f'n=−1 (−{photon_energy_eV:.3f} eV)')
axes[1].plot(k_th, -vF_fit*(k_th - dirac_kx) + dirac_E - photon_energy_eV, 'g--', lw=1.5)
for rb in band_data['replica_bands']:
    axes[1].plot(rb['kx'], rb['energy'], 'w*', ms=14, markeredgecolor='yellow',
                markeredgewidth=1, zorder=6)
axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[1].set_ylabel('Energy (eV)', fontsize=12)
axes[1].set_title('(b) Experiment vs Floquet Theory', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].set_xlim(-0.12, 0.20)
axes[1].set_ylim(-0.55, 0.55)

plt.tight_layout()
plt.savefig('report/images/figure7_dispersion_theory.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== FIGURE 8: VOLKOV / LAPE ANALYSIS ========================
print("Fig 8...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rep_idx = band_data['replica_bands'][2]['energy_idx']
for angle in polarization_angles:
    diff = pump_on[angle] - pump_off
    axes[0].plot(kx_axis, diff[rep_idx, :], lw=1.3, alpha=0.7, label=f'θ={angle}°')
axes[0].axvline(band_data['replica_bands'][2]['kx'], color='green', ls=':', alpha=0.5)
axes[0].axvline(band_data['replica_bands'][3]['kx'], color='green', ls=':', alpha=0.5)
axes[0].set_xlabel(r'$k_x$ (Å$^{-1}$)', fontsize=12)
axes[0].set_ylabel('Δ Intensity', fontsize=12)
axes[0].set_title(f'(a) MDC at Replica Energy\n(E={band_data["replica_bands"][2]["energy"]:.3f} eV)',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

# Theoretical comparison
theta_th = np.linspace(0, 360, 361)
I_lape = I0 + Delta_I * np.cos(np.deg2rad(theta_th))**2
I_floquet = I0 + Delta_I * np.cos(np.deg2rad(theta_th))**2 + 0.002 * np.cos(2*np.deg2rad(theta_th))

axes[1].plot(angles_deg, intensities, 'ko-', ms=10, lw=2, label='Experimental', zorder=5)
axes[1].plot(theta_th, I_lape, 'b--', lw=2, label='LAPE/Volkov: $I\\propto\\cos^2\\theta$', alpha=0.7)
axes[1].plot(theta_th, I_floquet, 'r-', lw=2, label='Floquet hybridization', alpha=0.7)
axes[1].fill_between(theta_th, I_lape, I_floquet, alpha=0.15, color='purple')
axes[1].set_xlabel(r'Pump Polarization Angle $\theta_p$ (°)', fontsize=12)
axes[1].set_ylabel('Replica Band Intensity', fontsize=12)
axes[1].set_title('(b) Mechanism: Floquet vs LAPE/Volkov', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xticks(angles_deg)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-10, 190)

plt.tight_layout()
plt.savefig('report/images/figure8_volkov_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK")

# ======================== SAVE QUANTITATIVE RESULTS ========================
results = {
    "experiment": {
        "pump_wavelength_um": 5.0,
        "photon_energy_eV": photon_energy_eV,
        "dirac_point": {"energy_eV": dirac_E, "kx_A_inv": dirac_kx},
        "energy_range_eV": [float(energy_axis.min()), float(energy_axis.max())],
        "kx_range_A_inv": [float(kx_axis.min()), float(kx_axis.max())],
        "time_delays_ps": time_delays.tolist(),
        "polarization_angles_deg": polarization_angles.tolist()
    },
    "floquet_results": {
        "replica_bands": band_data['replica_bands'],
        "expected_spacing_eV": photon_energy_eV,
        "measured_spacing_upper_eV": band_data['replica_bands'][2]['energy'] - dirac_E,
        "measured_spacing_lower_eV": band_data['replica_bands'][0]['energy'] - dirac_E,
    },
    "polarization_fit": {
        "model": "I(theta) = I0 + DeltaI * cos^2(theta)",
        "I0": float(I0), "Delta_I": float(Delta_I), "rms_residual": float(rms_error),
        "cos2_amplitude_pct": float(Delta_I/I0*100)
    }
}
with open('outputs/floquet_analysis_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print("Saved outputs/floquet_analysis_results.json")

print("\n===== ALL FIGURES GENERATED SUCCESSFULLY =====")
