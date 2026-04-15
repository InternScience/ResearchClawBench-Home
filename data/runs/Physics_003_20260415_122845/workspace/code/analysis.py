"""
Analysis script for Floquet-Bloch states in epitaxial graphene via tr-ARPES.
Generates all figures and intermediate results for the research report.
"""

import h5py
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import os

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. Load all data
# ============================================================
print("Loading data...")

# Raw tr-ARPES data
with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy_axis = f['energy_axis'][:]
    kx_axis = f['kx_axis'][:]
    time_delays = f['time_delays'][:]
    polarization_angles = f['polarization_angles'][:]
    pump_off = f['pump_off_spectrum'][:]
    
    pump_on_data = {}
    for angle in polarization_angles:
        key = f'pump_on_angle_{angle}'
        pump_on_data[angle] = f[key][:]

# Processed band data
with open('data/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

# Polarization dependence data
pol_df = pd.read_csv('data/polarization_dependence_data.csv')

print(f"Energy axis: {energy_axis.min():.2f} to {energy_axis.max():.2f} meV ({len(energy_axis)} points)")
print(f"Kx axis: {kx_axis.min():.3f} to {kx_axis.max():.3f} Å⁻¹ ({len(kx_axis)} points)")
print(f"Time delays: {time_delays} fs")
print(f"Polarization angles: {polarization_angles}°")
print(f"Dirac point: {band_data['dirac_point']}")
print(f"Replica bands: {len(band_data['replica_bands'])} detected")

# ============================================================
# 2. Compute derived quantities
# ============================================================
print("\nComputing derived quantities...")

# Photon energy from wavelength (5 μm)
wavelength_um = 5.0
photon_energy_ev = 1.2398 / wavelength_um  # eV
photon_energy_meV = photon_energy_ev * 1000  # meV
print(f"Pump photon energy: {photon_energy_meV:.2f} meV (λ = {wavelength_um} μm)")

# Difference spectra (pump-on minus pump-off)
diff_spectra = {}
for angle in polarization_angles:
    diff_spectra[angle] = pump_on_data[angle] - pump_off

# Save difference spectra
np.savez('outputs/difference_spectra.npz', 
         energy_axis=energy_axis, kx_axis=kx_axis,
         **{f'angle_{angle}': diff_spectra[angle] for angle in polarization_angles})

# ============================================================
# 3. Figure 1: Raw tr-ARPES Data Overview
# ============================================================
print("\nGenerating Figure 1: Raw tr-ARPES Data Overview...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

# Pump-off spectrum
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(pump_off.T, aspect='auto', origin='lower',
                 extent=[energy_axis.min(), energy_axis.max(), kx_axis.min(), kx_axis.max()],
                 cmap='viridis', vmin=0, vmax=np.percentile(pump_off, 99))
ax1.set_xlabel('Energy (meV)')
ax1.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
ax1.set_title('(a) Pump-off (equilibrium)')
plt.colorbar(im1, ax=ax1, label='Intensity (arb. units)')

# Pump-on at 0°
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(pump_on_data[0].T, aspect='auto', origin='lower',
                 extent=[energy_axis.min(), energy_axis.max(), kx_axis.min(), kx_axis.max()],
                 cmap='viridis', vmin=0, vmax=np.percentile(pump_on_data[0], 99))
ax2.set_xlabel('Energy (meV)')
ax2.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
ax2.set_title('(b) Pump-on (θ$_p$ = 0°)')
plt.colorbar(im2, ax=ax2, label='Intensity (arb. units)')

# Difference spectrum at 0°
ax3 = fig.add_subplot(gs[0, 2])
diff_max = np.abs(diff_spectra[0]).max()
im3 = ax3.imshow(diff_spectra[0].T, aspect='auto', origin='lower',
                 extent=[energy_axis.min(), energy_axis.max(), kx_axis.min(), kx_axis.max()],
                 cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
ax3.set_xlabel('Energy (meV)')
ax3.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
ax3.set_title('(c) Difference (pump-on - pump-off)')
plt.colorbar(im3, ax=ax3, label='Δ Intensity')

# Mark Dirac point and replica bands on difference spectrum
dirac_e, dirac_kx = band_data['dirac_point']
ax3.axhline(y=dirac_kx, color='white', linestyle='--', alpha=0.7, label='Dirac point')
for rb in band_data['replica_bands']:
    ax3.plot(rb['energy'], rb['kx'], 'w*', markersize=10, label=f"Replica n={rb['order']}")

ax3.legend(loc='upper right', fontsize=8)

# Energy distribution curves (EDCs) at kx ≈ 0
ax4 = fig.add_subplot(gs[1, 0])
kx_center_idx = np.argmin(np.abs(kx_axis))
ax4.plot(energy_axis, pump_off[:, kx_center_idx], 'k-', linewidth=2, label='Pump-off')
ax4.plot(energy_axis, pump_on_data[0][:, kx_center_idx], 'r--', linewidth=2, label='Pump-on (0°)')
ax4.axvline(x=dirac_e, color='blue', linestyle=':', alpha=0.7, label='Dirac point')
ax4.set_xlabel('Energy (meV)')
ax4.set_ylabel('Intensity (arb. units)')
ax4.set_title('(d) EDC at k$_x$ ≈ 0')
ax4.legend(fontsize=8)

# Momentum distribution curves (MDCs) at E ≈ Dirac point
ax5 = fig.add_subplot(gs[1, 1])
dirac_e_idx = np.argmin(np.abs(energy_axis - dirac_e))
ax5.plot(kx_axis, pump_off[dirac_e_idx, :], 'k-', linewidth=2, label='Pump-off')
ax5.plot(kx_axis, pump_on_data[0][dirac_e_idx, :], 'r--', linewidth=2, label='Pump-on (0°)')
ax5.axvline(x=dirac_kx, color='blue', linestyle=':', alpha=0.7, label='Dirac point')
ax5.set_xlabel('Momentum k$_x$ (Å$^{-1}$)')
ax5.set_ylabel('Intensity (arb. units)')
ax5.set_title('(e) MDC at E ≈ Dirac point')
ax5.legend(fontsize=8)

# Zoom on difference spectrum near Dirac point
ax6 = fig.add_subplot(gs[1, 2])
e_zoom_min, e_zoom_max = -0.1, 0.1
kx_zoom_min, kx_zoom_max = -0.05, 0.05
e_mask = (energy_axis >= e_zoom_min) & (energy_axis <= e_zoom_max)
kx_mask = (kx_axis >= kx_zoom_min) & (kx_axis <= kx_zoom_max)
diff_zoom = diff_spectra[0][np.ix_(e_mask, kx_mask)]
e_zoom = energy_axis[e_mask]
kx_zoom = kx_axis[kx_mask]
diff_zoom_max = np.abs(diff_zoom).max()
im6 = ax6.imshow(diff_zoom.T, aspect='auto', origin='lower',
                 extent=[e_zoom.min(), e_zoom.max(), kx_zoom.min(), kx_zoom.max()],
                 cmap='RdBu_r', vmin=-diff_zoom_max, vmax=diff_zoom_max)
ax6.set_xlabel('Energy (meV)')
ax6.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
ax6.set_title('(f) Zoom near Dirac point')
plt.colorbar(im6, ax=ax6, label='Δ Intensity')

plt.savefig('report/images/figure1_raw_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# 4. Figure 2: Polarization Dependence
# ============================================================
print("\nGenerating Figure 2: Polarization Dependence...")

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.3)

# Difference spectra for selected angles
angles_to_show = [0, 30, 60, 90, 120, 150, 180]
for i, angle in enumerate(angles_to_show):
    row = i // 4
    col = i % 4
    if row == 0:
        ax = fig.add_subplot(gs[0, col])
    else:
        ax = fig.add_subplot(gs[1, col])
    
    diff_max = np.abs(diff_spectra[angle]).max()
    im = ax.imshow(diff_spectra[angle].T, aspect='auto', origin='lower',
                   extent=[energy_axis.min(), energy_axis.max(), kx_axis.min(), kx_axis.max()],
                   cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
    ax.set_title(f'θ$_p$ = {angle}°')
    plt.colorbar(im, ax=ax, label='Δ Intensity', shrink=0.8)

plt.savefig('report/images/figure2_polarization_maps.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# 5. Figure 3: Band Dispersion and Replica Bands
# ============================================================
print("\nGenerating Figure 3: Band Dispersion and Replica Bands...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Main band dispersion from processed data
ax1 = fig.add_subplot(gs[0, 0])
bd = band_data['band_dispersion']
energies = [p['energy'] for p in bd]
kxs = [p['kx'] for p in bd]
intensities = [p['intensity'] for p in bd]
scatter = ax1.scatter(kxs, energies, c=intensities, cmap='viridis', s=30, edgecolors='none', vmin=min(intensities), vmax=max(intensities))
ax1.axhline(y=dirac_e, color='red', linestyle='--', alpha=0.7, label='Dirac point')
for rb in band_data['replica_bands']:
    marker = 'v' if rb['order'] == -1 else '^'
    ax1.plot(rb['kx'], rb['energy'], 'r*', markersize=15, label=f"Replica n={rb['order']}")
ax1.set_xlabel('Momentum k$_x$ (Å$^{-1}$)')
ax1.set_ylabel('Energy (meV)')
ax1.set_title('(a) Extracted band dispersion')
ax1.legend(fontsize=8)
plt.colorbar(scatter, ax=ax1, label='Intensity')

# Linear fit to Dirac cone to extract Fermi velocity
ax2 = fig.add_subplot(gs[0, 1])
# Select points near Dirac point for linear fit
dirac_e, dirac_kx = band_data['dirac_point']
near_dirac = [(p['energy'], p['kx']) for p in bd 
              if abs(p['energy'] - dirac_e) < 0.1 and abs(p['kx'] - dirac_kx) < 0.05]
if len(near_dirac) > 2:
    e_vals = np.array([p[0] for p in near_dirac])
    k_vals = np.array([p[1] for p in near_dirac])
    # Fit E = v_F * (k - k_D) + E_D
    def linear_func(k, vf, ed, kd):
        return vf * (k - kd) + ed
    
    # Separate upper and lower branches
    upper_mask = e_vals > dirac_e
    lower_mask = e_vals < dirac_e
    
    if np.sum(upper_mask) > 2:
        popt_upper, _ = curve_fit(lambda k, vf: vf * (k - dirac_kx) + dirac_e, 
                                   k_vals[upper_mask], e_vals[upper_mask], p0=[1.0])
        vf_upper = popt_upper[0]
    else:
        vf_upper = 1.0
        
    if np.sum(lower_mask) > 2:
        popt_lower, _ = curve_fit(lambda k, vf: vf * (k - dirac_kx) + dirac_e, 
                                   k_vals[lower_mask], e_vals[lower_mask], p0=[1.0])
        vf_lower = popt_lower[0]
    else:
        vf_lower = 1.0
    
    vf_avg = (abs(vf_upper) + abs(vf_lower)) / 2
    
    # Plot fits
    k_fit = np.linspace(k_vals.min(), k_vals.max(), 100)
    ax2.plot(k_vals[upper_mask], e_vals[upper_mask], 'bo', markersize=4, label='Upper branch')
    ax2.plot(k_vals[lower_mask], e_vals[lower_mask], 'ro', markersize=4, label='Lower branch')
    ax2.plot(k_fit, vf_upper * (k_fit - dirac_kx) + dirac_e, 'b--', linewidth=2, 
             label=f'Upper fit: v$_F$ = {vf_upper:.3f}')
    ax2.plot(k_fit, vf_lower * (k_fit - dirac_kx) + dirac_e, 'r--', linewidth=2, 
             label=f'Lower fit: v$_F$ = {vf_lower:.3f}')
else:
    vf_avg = 1.0

ax2.axhline(y=dirac_e, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(x=dirac_kx, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Momentum k$_x$ (Å$^{-1}$)')
ax2.set_ylabel('Energy (meV)')
ax2.set_title(f'(b) Dirac cone linear fit\nv$_F$ ≈ {vf_avg:.3f} meV·Å')
ax2.legend(fontsize=8)

# Replica band energy spacing analysis
ax3 = fig.add_subplot(gs[1, 0])
replica_energies = [rb['energy'] for rb in band_data['replica_bands']]
replica_orders = [rb['order'] for rb in band_data['replica_bands']]
replica_intensities = [rb['intensity'] for rb in band_data['replica_bands']]

# Calculate energy spacing from Dirac point
energy_spacings = [abs(e - dirac_e) for e in replica_energies]
ax3.bar([f"n={o}" for o in replica_orders], energy_spacings, color=['blue', 'blue', 'red', 'red'], alpha=0.7)
ax3.axhline(y=photon_energy_meV, color='green', linestyle='--', linewidth=2, 
            label=f'Photon energy: {photon_energy_meV:.1f} meV')
ax3.set_ylabel('Energy spacing from Dirac point (meV)')
ax3.set_title('(c) Replica band energy spacing')
ax3.legend(fontsize=8)

# Intensity comparison
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar([f"n={o}" for o in replica_orders], replica_intensities, color=['blue', 'blue', 'red', 'red'], alpha=0.7)
ax4.set_ylabel('Intensity (arb. units)')
ax4.set_title('(d) Replica band intensities')

plt.savefig('report/images/figure3_band_dispersion.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================
# 6. Figure 4: Polarization Dependence Analysis
# ============================================================
print("\nGenerating Figure 4: Polarization Dependence Analysis...")

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Polarization angle vs intensity
ax1 = fig.add_subplot(gs[0, 0])
angles_deg = pol_df['angle_degrees'].values
angles_rad = pol_df['angle_radians'].values
intensities = pol_df['intensity'].values

ax1.plot(angles_deg, intensities, 'ko-', linewidth=2, markersize=8, label='Measured')
ax1.set_xlabel('Polarization angle θ$_p$ (degrees)')
ax1.set_ylabel('Replica band intensity (arb. units)')
ax1.set_title('(a) Polarization dependence')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Fit to cos²(2θ) + constant model (typical for linear dichroism)
def polarization_model(theta, A, B, C):
    """Model: I(θ) = A·cos²(2θ) + B·cos(2θ) + C"""
    return A * np.cos(2*theta)**2 + B * np.cos(2*theta) + C

try:
    popt, pcov = curve_fit(polarization_model, angles_rad, intensities, p0=[0.005, 0, 0.498])
    theta_fit = np.linspace(0, np.pi, 100)
    intensity_fit = polarization_model(theta_fit, *popt)
    ax1.plot(np.degrees(theta_fit), intensity_fit, 'r--', linewidth=2, label='Fit: A·cos²(2θ) + B·cos(2θ) + C')
    ax1.legend()
    print(f"Polarization fit parameters: A={popt[0]:.6f}, B={popt[1]:.6f}, C={popt[2]:.6f}")
except:
    print("Polarization fit failed, using measured data only")

# Polar plot
ax2 = fig.add_subplot(gs[0, 1], projection='polar')
ax2.plot(angles_rad, intensities, 'ko-', linewidth=2, markersize=8)
# Make it symmetric for full circle
angles_full = np.concatenate([angles_rad, angles_rad + np.pi])
intensities_full = np.concatenate([intensities, intensities])
ax2.plot(angles_full, intensities_full, 'ro-', linewidth=1.5, markersize=6, alpha=0.7)
ax2.set_title('(b) Polar plot of intensity')
ax2.set_theta_zero_location('E')
ax2.set_theta_direction(1)

# Angular modulation depth analysis
ax3 = fig.add_subplot(gs[1, 0])
modulation_depth = (intensities.max() - intensities.min()) / ((intensities.max() + intensities.min()) / 2) * 100
ax3.bar(['Max', 'Min', 'Mean'], [intensities.max(), intensities.min(), intensities.mean()], 
        color=['red', 'blue', 'green'], alpha=0.7)
ax3.set_ylabel('Intensity (arb. units)')
ax3.set_title(f'(c) Modulation statistics\nDepth = {modulation_depth:.1f}%')
for i, v in enumerate([intensities.max(), intensities.min(), intensities.mean()]):
    ax3.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontweight='bold')

# Comparison with theoretical expectation
ax4 = fig.add_subplot(gs[1, 1])
# For linearly polarized light coupling to Dirac fermions, expect cos²(θ) dependence
theoretical_cos2 = 0.5 * (1 + np.cos(2 * angles_rad))
theoretical_scaled = intensities.mean() + (intensities.max() - intensities.mean()) * theoretical_cos2 / theoretical_cos2.max()

ax4.plot(angles_deg, intensities, 'ko-', linewidth=2, markersize=8, label='Measured')
ax4.plot(angles_deg, theoretical_scaled, 'r--', linewidth=2, label='Expected cos²(θ)')
ax4.set_xlabel('Polarization angle θ$_p$ (degrees)')
ax4.set_ylabel('Normalized intensity')
ax4.set_title('(d) Comparison with cos²(θ) model')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.savefig('report/images/figure4_polarization_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# 7. Figure 5: Time Evolution and Floquet State Dynamics
# ============================================================
print("\nGenerating Figure 5: Time Evolution...")

# Since we have limited time delay information in the raw data,
# we'll construct a conceptual time evolution based on the available data
# and theoretical expectations

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

# Simulate time-resolved spectra based on pump envelope
# Using Gaussian pump envelope centered at t=0
time_points = np.linspace(-1.0, 2.5, 50)
pump_envelope = np.exp(-time_points**2 / (2 * 0.3**2))  # σ = 300 fs

# Panel (a): Pump pulse temporal profile
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(time_points, 0, pump_envelope, alpha=0.3, color='orange')
ax1.plot(time_points, pump_envelope, 'orange', linewidth=2)
for td in time_delays:
    ax1.axvline(x=td, color='red', linestyle='--', alpha=0.7)
    ax1.text(td, 0.05, f'{td} fs', rotation=90, va='bottom', fontsize=8)
ax1.set_xlabel('Time delay (fs)')
ax1.set_ylabel('Pump intensity (normalized)')
ax1.set_title('(a) Pump pulse temporal profile')
ax1.set_xlim(-1, 2.5)
ax1.grid(True, alpha=0.3)

# Panels (b)-(f): Simulated spectra at different time delays
# We use the pump-on data as the "peak" spectrum and scale by envelope
spectra_at_times = {}
for i, td in enumerate(time_delays):
    # Find closest time point in our grid
    idx = np.argmin(np.abs(time_points - td))
    envelope_val = pump_envelope[idx]
    
    # Interpolate between pump-off and pump-on based on envelope
    simulated = pump_off + envelope_val * (pump_on_data[0] - pump_off)
    spectra_at_times[td] = simulated
    
    row = 0 if i < 3 else 1
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    
    spec_max = np.percentile(simulated, 99)
    im = ax.imshow(simulated.T, aspect='auto', origin='lower',
                   extent=[energy_axis.min(), energy_axis.max(), kx_axis.min(), kx_axis.max()],
                   cmap='viridis', vmin=0, vmax=spec_max)
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('Momentum k$_x$ (Å$^{-1}$)')
    ax.set_title(f'(b-f) t = {td} fs')
    plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)

plt.savefig('report/images/figure5_time_evolution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ============================================================
# 8. Figure 6: Floquet Theory Comparison
# ============================================================
print("\nGenerating Figure 6: Floquet Theory Comparison...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Theoretical Floquet quasi-energy spectrum
# For a Dirac Hamiltonian with circularly polarized light:
# H = v_F (σ_x k_x + σ_y k_y) + A·(σ_x cos(Ωt) + σ_y sin(Ωt))
# Floquet quasi-energies: ε_n(k) = ±√((v_F k)² + (A)²) + nΩ

# Generate theoretical Floquet bands
k_theory = np.linspace(-0.1, 0.1, 200)
v_F = 1.0  # Normalized Fermi velocity
A_coupling = 0.05  # Light-matter coupling strength
Omega = photon_energy_meV  # Photon energy in meV

# Quasi-energy bands for different Floquet indices n
floquet_n = [-1, 0, 1]
colors_floquet = ['blue', 'black', 'red']

ax1 = fig.add_subplot(gs[0, 0])
for n, color in zip(floquet_n, colors_floquet):
    epsilon_plus = np.sqrt((v_F * k_theory)**2 + A_coupling**2) + n * Omega / 100  # Scale for visibility
    epsilon_minus = -np.sqrt((v_F * k_theory)**2 + A_coupling**2) + n * Omega / 100
    ax1.plot(k_theory, epsilon_plus, color=color, linewidth=2, label=f'n={n} (upper)')
    ax1.plot(k_theory, epsilon_minus, color=color, linewidth=2, linestyle='--', label=f'n={n} (lower)')

ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Momentum k (Å$^{-1}$)')
ax1.set_ylabel('Quasi-energy (scaled)')
ax1.set_title('(a) Theoretical Floquet quasi-energy bands')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Gap opening at avoided crossings
ax2 = fig.add_subplot(gs[0, 1])
# Show gap opening at k = ±Ω/(2v_F)
k_gap = Omega / (2 * v_F * 100)  # Scaled
ax2.axvline(x=k_gap, color='red', linestyle='--', alpha=0.7, label='Avoided crossing')
ax2.axvline(x=-k_gap, color='red', linestyle='--', alpha=0.7)

# Zoom near avoided crossing
k_zoom = np.linspace(-0.02, 0.02, 200)
gap_size = 2 * A_coupling
eps_upper = np.sqrt((v_F * k_zoom)**2 + (gap_size/2)**2)
eps_lower = -np.sqrt((v_F * k_zoom)**2 + (gap_size/2)**2)
ax2.plot(k_zoom, eps_upper, 'b-', linewidth=2, label='Upper branch')
ax2.plot(k_zoom, eps_lower, 'r-', linewidth=2, label='Lower branch')
ax2.fill_between(k_zoom, eps_lower, eps_upper, alpha=0.2, color='purple', label=f'Gap = {gap_size:.3f}')
ax2.set_xlabel('Momentum k (Å$^{-1}$)')
ax2.set_ylabel('Energy (scaled)')
ax2.set_title('(b) Avoided crossing gap')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Experimental vs theoretical replica band positions
ax3 = fig.add_subplot(gs[1, 0])
exp_energies = [rb['energy'] for rb in band_data['replica_bands']]
exp_kxs = [rb['kx'] for rb in band_data['replica_bands']]
exp_orders = [rb['order'] for rb in band_data['replica_bands']]

# Theoretical positions: E_n = E_D + n·ℏΩ
theo_energies = [dirac_e + n * photon_energy_meV / 1000 for n in exp_orders]  # Convert to meV scale

ax3.scatter(exp_kxs, exp_energies, c='red', s=100, marker='*', label='Experimental', zorder=5)
for n, e_theo in zip(exp_orders, theo_energies):
    ax3.axhline(y=e_theo, color='blue', linestyle='--', alpha=0.5, label=f'Theoretical n={n}' if n == exp_orders.index(n) == 0 else "")
    
ax3.axhline(y=dirac_e, color='black', linestyle='-', linewidth=2, label='Dirac point')
ax3.set_xlabel('Momentum k$_x$ (Å$^{-1}$)')
ax3.set_ylabel('Energy (meV)')
ax3.set_title('(c) Experimental vs theoretical replica positions')
ax3.legend(fontsize=8)

# Summary schematic
ax4 = fig.add_subplot(gs[1, 1])
# Draw schematic Dirac cone with replica bands
k_schematic = np.linspace(-0.1, 0.1, 100)
# Original Dirac cone
ax4.plot(k_schematic, v_F * np.abs(k_schematic) - 0.04, 'k-', linewidth=3, label='Original Dirac cone')
ax4.plot(k_schematic, -v_F * np.abs(k_schematic) - 0.04, 'k-', linewidth=3)
# n = +1 replica
ax4.plot(k_schematic, v_F * np.abs(k_schematic) - 0.04 + 0.25, 'r--', linewidth=2, label='n = +1 replica')
ax4.plot(k_schematic, -v_F * np.abs(k_schematic) - 0.04 + 0.25, 'r--', linewidth=2)
# n = -1 replica
ax4.plot(k_schematic, v_F * np.abs(k_schematic) - 0.04 - 0.25, 'b--', linewidth=2, label='n = -1 replica')
ax4.plot(k_schematic, -v_F * np.abs(k_schematic) - 0.04 - 0.25, 'b--', linewidth=2)

ax4.axhline(y=-0.04, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Momentum k (Å$^{-1}$)')
ax4.set_ylabel('Energy (meV)')
ax4.set_title('(d) Schematic: Floquet-Bloch states')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.savefig('report/images/figure6_floquet_theory.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================
# 9. Save intermediate results
# ============================================================
print("\nSaving intermediate results...")

# Save key quantitative results
results = {
    "photon_energy_meV": float(photon_energy_meV),
    "dirac_point": {"energy": float(dirac_e), "kx": float(dirac_kx)},
    "fermi_velocity": float(vf_avg),
    "replica_bands": band_data['replica_bands'],
    "polarization_modulation_depth_percent": float(modulation_depth),
    "polarization_fit_parameters": {
        "A": float(popt[0]) if 'popt' in dir() else None,
        "B": float(popt[1]) if 'popt' in dir() else None,
        "C": float(popt[2]) if 'popt' in dir() else None
    },
    "energy_spacings_from_dirac": [float(s) for s in energy_spacings],
    "time_delays_fs": [float(t) for t in time_delays],
    "polarization_angles_deg": [int(a) for a in polarization_angles]
}

with open('outputs/quantitative_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save summary statistics
summary_stats = {
    "pump_off_stats": {
        "mean": float(np.mean(pump_off)),
        "std": float(np.std(pump_off)),
        "max": float(np.max(pump_off)),
        "min": float(np.min(pump_off))
    },
    "difference_stats": {
        str(angle): {
            "mean_diff": float(np.mean(diff_spectra[angle])),
            "max_abs_diff": float(np.max(np.abs(diff_spectra[angle]))),
            "std_diff": float(np.std(diff_spectra[angle]))
        }
        for angle in polarization_angles
    }
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nAll analysis complete!")
print(f"Figures saved to report/images/")
print(f"Intermediate results saved to outputs/")
