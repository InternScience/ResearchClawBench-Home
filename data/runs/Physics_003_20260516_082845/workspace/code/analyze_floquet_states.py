#!/usr/bin/env python3
"""
Analysis of Floquet-Bloch states in monolayer epitaxial graphene
via time-resolved and angle-resolved photoemission spectroscopy (tr-ARPES).

This script:
1. Loads and processes the raw tr-ARPES data (HDF5)
2. Analyzes processed band data (replica bands, Dirac cone dispersion)
3. Analyzes polarization dependence of replica band intensity
4. Generates all figures for the research report
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import json
import csv
import os

# Set up paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_003_20260516_082845'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")

# Raw tr-ARPES data
with h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r') as f:
    energy_axis = f['energy_axis'][:]
    kx_axis = f['kx_axis'][:]
    polarization_angles = f['polarization_angles'][:]
    time_delays = f['time_delays'][:]
    
    pump_off = f['pump_off_spectrum'][:]
    
    pump_on = {}
    for angle in polarization_angles:
        pump_on[angle] = f[f'pump_on_angle_{angle}'][:]

print(f"  Energy axis: {energy_axis.shape}, range: [{energy_axis[0]:.3f}, {energy_axis[-1]:.3f}]")
print(f"  kx axis: {kx_axis.shape}, range: [{kx_axis[0]:.3f}, {kx_axis[-1]:.3f}]")
print(f"  Polarization angles: {polarization_angles}")
print(f"  Time delays: {time_delays}")
print(f"  Pump-off spectrum shape: {pump_off.shape}")
for angle in polarization_angles:
    print(f"  Pump-on angle {angle}: {pump_on[angle].shape}")

# Processed band data
with open(os.path.join(DATA_DIR, 'processed_band_data.json'), 'r') as f:
    band_data = json.load(f)

dirac_point = band_data['dirac_point']
dirac_indices = band_data['dirac_indices']
replica_bands = band_data['replica_bands']
band_dispersion = band_data['band_dispersion']

print(f"\nDirac point (energy, kx): {dirac_point}")
print(f"Number of replica bands: {len(replica_bands)}")
for rb in replica_bands:
    print(f"  Order {rb['order']}: energy={rb['energy']:.4f}, kx={rb['kx']:.4f}, intensity={rb['intensity']:.4f}")

# Polarization dependence data
polarization_data = []
with open(os.path.join(DATA_DIR, 'polarization_dependence_data.csv'), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        polarization_data.append({
            'angle_degrees': float(row['angle_degrees']),
            'angle_radians': float(row['angle_radians']),
            'intensity': float(row['intensity']),
            'target_energy': float(row['target_energy']),
            'target_kx': float(row['target_kx']),
        })

angles_deg = [d['angle_degrees'] for d in polarization_data]
intensities = [d['intensity'] for d in polarization_data]
print(f"\nPolarization data: {len(polarization_data)} points")
for d in polarization_data:
    print(f"  θ={d['angle_degrees']}°: I={d['intensity']:.6f}")

# ============================================================
# 2. DIFFERENCE SPECTRA (pump on - pump off)
# ============================================================
print("\nComputing difference spectra...")

difference = {}
for angle in polarization_angles:
    difference[angle] = pump_on[angle] - pump_off

# ============================================================
# 3. FIGURE GENERATION
# ============================================================

# --- Figure 1: Raw tr-ARPES spectra overview ---
print("Generating Figure 1: Raw tr-ARPES spectra...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Pump off (equilibrium)
im0 = axes[0, 0].pcolormesh(kx_axis, energy_axis, pump_off, shading='auto', cmap='inferno')
axes[0, 0].set_title('(a) Pump Off (Equilibrium)')
axes[0, 0].set_xlabel('$k_x$ ($\\AA^{-1}$)')
axes[0, 0].set_ylabel('$E - E_F$ (eV)')
plt.colorbar(im0, ax=axes[0, 0], label='Intensity (arb. u.)')

# Pump on at selected angles
plot_angles = [0, 30, 60, 90, 120, 150, 180]
plot_labels = ['(b) Pump On, θ=0°', '(c) Pump On, θ=30°', '(d) Pump On, θ=60°',
               '(e) Pump On, θ=90°', '(f) Pump On, θ=120°', '(g) Pump On, θ=150°', '(h) Pump On, θ=180°']

for idx, (ang, label) in enumerate(zip(plot_angles, plot_labels)):
    ax = axes[(idx + 1) // 4, (idx + 1) % 4]
    im = ax.pcolormesh(kx_axis, energy_axis, pump_on[ang], shading='auto', cmap='inferno')
    ax.set_title(label)
    ax.set_xlabel('$k_x$ ($\\AA^{-1}$)')
    ax.set_ylabel('$E - E_F$ (eV)')
    plt.colorbar(im, ax=ax, label='Intensity (arb. u.)')

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure1_raw_spectra.png'), dpi=150)
plt.close()
print("  Saved figure1_raw_spectra.png")

# --- Figure 2: Difference spectra (Pump On - Pump Off) ---
print("Generating Figure 2: Difference spectra...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for idx, (ang, label) in enumerate(zip(plot_angles, plot_labels)):
    ax = axes[idx // 4, idx % 4]
    vmax = np.max(np.abs(difference[ang])) * 0.6
    im = ax.pcolormesh(kx_axis, energy_axis, difference[ang], shading='auto',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Diff: θ={ang}°')
    ax.set_xlabel('$k_x$ ($\\AA^{-1}$)')
    ax.set_ylabel('$E - E_F$ (eV)')
    plt.colorbar(im, ax=ax, label='Δ Intensity')

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure2_difference_spectra.png'), dpi=150)
plt.close()
print("  Saved figure2_difference_spectra.png")

# --- Figure 3: Band dispersion and replica bands ---
print("Generating Figure 3: Band dispersion and replica bands...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot band dispersion
energies = np.array([b['energy'] for b in band_dispersion])
kxs = np.array([b['kx'] for b in band_dispersion])
intensities_band = np.array([b['intensity'] for b in band_dispersion])

sc = ax.scatter(kxs, energies, c=intensities_band, cmap='inferno', s=20, 
                alpha=0.8, label='Dirac Cone Dispersion')
cbar = plt.colorbar(sc, ax=ax, label='Intensity')

# Plot replica bands
for rb in replica_bands:
    marker = 'D' if rb['order'] == -1 else 's'
    color = 'cyan' if rb['order'] == -1 else 'lime'
    size = 100 * rb['intensity']
    ax.scatter(rb['kx'], rb['energy'], c=color, s=size, marker=marker, 
               edgecolors='black', linewidths=1, zorder=10,
               label=f"Replica n={rb['order']} (I={rb['intensity']:.3f})")

# Dirac point
ax.scatter(dirac_point[1], dirac_point[0], c='yellow', s=200, marker='*',
           edgecolors='black', linewidths=1.5, zorder=10, label='Dirac Point')

ax.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax.set_ylabel('$E - E_F$ (eV)')
ax.set_title('Dirac Cone Dispersion and Floquet-Bloch Replica Bands')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure3_band_dispersion.png'), dpi=150)
plt.close()
print("  Saved figure3_band_dispersion.png")

# --- Figure 4: Polarization dependence ---
print("Generating Figure 4: Polarization dependence...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Polar plot
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
angles_rad_full = np.array([d['angle_radians'] for d in polarization_data])
# Duplicate first point at end for closed loop
angles_plot = np.append(angles_rad_full, angles_rad_full[0] + 2*np.pi)
intensities_plot = np.append(intensities, intensities[0])

ax1.plot(angles_plot, intensities_plot, 'o-', color='darkblue', linewidth=2, markersize=8)
ax1.fill(angles_plot, intensities_plot, alpha=0.2, color='blue')
ax1.set_title('(a) Polar Plot of Replica Band Intensity')
ax1.set_rlabel_position(30)
ax1.set_theta_zero_location('E')
ax1.set_theta_direction(1)  # counterclockwise

# Right panel: Cartesian plot
ax2 = axes[1]
ax2.plot(angles_deg, intensities, 'o-', color='darkred', linewidth=2, markersize=8)
ax2.set_xlabel('Pump Polarization Angle θ (degrees)')
ax2.set_ylabel('Replica Band Intensity (arb. u.)')
ax2.set_title('(b) Intensity vs. Polarization Angle')
ax2.axhline(y=np.mean(intensities), color='gray', linestyle='--', alpha=0.5,
            label=f'Mean = {np.mean(intensities):.4f}')
ax2.legend()

# Add sinusoidal fit
from scipy.optimize import curve_fit

def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.radians(x) + c) + d

try:
    popt, _ = curve_fit(sin_func, angles_deg, intensities, p0=[0.005, 0.0, 0.0, 0.5])
    x_fit = np.linspace(0, 360, 200)
    y_fit = sin_func(x_fit, *popt)
    ax2.plot(x_fit, y_fit, '--', color='orange', linewidth=1.5, alpha=0.7,
             label=f'Fit: $I = I_0 + A \\sin(2\\theta + \\phi)$')
    ax2.legend()
except Exception as e:
    print(f"  Sinusoidal fit failed: {e}")

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure4_polarization_dependence.png'), dpi=150)
plt.close()
print("  Saved figure4_polarization_dependence.png")

# --- Figure 5: Replica band analysis ---
print("Generating Figure 5: Replica band analysis...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig)

# (a) Replica band positions in energy-momentum space
ax_a = fig.add_subplot(gs[0, 0])
# Plot the Dirac cone dispersion as background
ax_a.scatter(kxs, energies, c=intensities_band, cmap='inferno', s=5, alpha=0.3)
# Mark the Dirac point
ax_a.scatter(dirac_point[1], dirac_point[0], c='black', s=150, marker='*', zorder=5)
# Mark replica bands
for rb in replica_bands:
    color = 'dodgerblue' if rb['order'] == -1 else 'crimson'
    marker = 'v' if rb['order'] == -1 else '^'
    ax_a.scatter(rb['kx'], rb['energy'], c=color, s=150, marker=marker,
                 edgecolors='black', linewidths=1.5, zorder=10)
    ax_a.annotate(f"n={rb['order']}", (rb['kx'], rb['energy']),
                  xytext=(10, 10 if rb['order'] > 0 else -15),
                  textcoords='offset points', fontsize=10, fontweight='bold',
                  color=color)
ax_a.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_a.set_ylabel('$E - E_F$ (eV)')
ax_a.set_title('(a) Replica Band Positions')

# (b) Energy splitting from Dirac point
ax_b = fig.add_subplot(gs[0, 1])
orders = [-1, 1]
energies_order = {o: [] for o in orders}
for rb in replica_bands:
    energies_order[rb['order']].append(abs(rb['energy'] - dirac_point[0]))

for o in orders:
    e_vals = energies_order[o]
    ax_b.bar(o, np.mean(e_vals), yerr=np.std(e_vals) if len(e_vals) > 1 else 0,
             color=['dodgerblue', 'crimson'][orders.index(o)],
             alpha=0.7, capsize=8, width=0.5)
ax_b.set_xlabel('Floquet Order n')
ax_b.set_ylabel('|ΔE| from Dirac Point (eV)')
ax_b.set_title('(b) Energy Splitting of Replicas')
ax_b.set_xticks(orders)
ax_b.axhline(y=0.248, color='green', linestyle='--', alpha=0.5, 
             label=f'ħω ≈ 0.248 eV (5 μm)')
ax_b.legend()

# (c) Replica intensity comparison
ax_c = fig.add_subplot(gs[0, 2])
for i, rb in enumerate(replica_bands):
    color = 'dodgerblue' if rb['order'] == -1 else 'crimson'
    ax_c.bar(i, rb['intensity'], color=color, alpha=0.7, width=0.5)
    ax_c.text(i, rb['intensity'] + 0.002, f"n={rb['order']}\n{rb['intensity']:.4f}",
              ha='center', fontsize=9)
ax_c.set_xlabel('Replica Band')
ax_c.set_ylabel('Intensity (arb. u.)')
ax_c.set_title('(c) Replica Band Intensities')
ax_c.set_xticks(range(len(replica_bands)))
ax_c.set_xticklabels([f'n={rb["order"]}' for rb in replica_bands])

# (d) Polarization intensity ratio analysis
ax_d = fig.add_subplot(gs[1, 0])
# Compute intensity ratios: parallel (0°/180°) vs perpendicular (90°)
I_parallel = np.mean([polarization_data[0]['intensity'], polarization_data[6]['intensity']])
I_perp = polarization_data[3]['intensity']  # 90°
I_30 = polarization_data[1]['intensity']
I_150 = polarization_data[5]['intensity']

categories = ['θ=0°/180°\n(parallel)', 'θ=30°/150°\n(oblique)', 'θ=90°\n(perpendicular)']
values = [I_parallel, np.mean([I_30, I_150]), I_perp]
colors_bar = ['darkblue', 'purple', 'darkred']
ax_d.bar(categories, values, color=colors_bar, alpha=0.7, width=0.5)
ax_d.set_ylabel('Replica Band Intensity (arb. u.)')
ax_d.set_title('(d) Intensity vs. Polarization Direction')
ax_d.set_ylim(0.49, 0.51)

# (e) Energy-momentum cut at Dirac point (EDC)
ax_e = fig.add_subplot(gs[1, 1])
# Find the kx index closest to the Dirac point
dirac_kx_idx = dirac_indices[1]
edc_pump_off = pump_off[:, dirac_kx_idx]
edc_pump_on_0 = pump_on[0][:, dirac_kx_idx]
edc_pump_on_90 = pump_on[90][:, dirac_kx_idx]

ax_e.plot(energy_axis, edc_pump_off, 'k-', linewidth=2, label='Pump Off')
ax_e.plot(energy_axis, edc_pump_on_0, 'b-', linewidth=1.5, label='Pump On (θ=0°)')
ax_e.plot(energy_axis, edc_pump_on_90, 'r-', linewidth=1.5, label='Pump On (θ=90°)')
ax_e.set_xlabel('$E - E_F$ (eV)')
ax_e.set_ylabel('Intensity (arb. u.)')
ax_e.set_title('(e) Energy Distribution Curve at Dirac Point')
ax_e.legend(fontsize=9)
ax_e.axvline(x=dirac_point[0], color='gray', linestyle='--', alpha=0.5)

# (f) Momentum distribution curve at Dirac energy
ax_f = fig.add_subplot(gs[1, 2])
dirac_energy_idx = dirac_indices[0]
mdc_pump_off = pump_off[dirac_energy_idx, :]
mdc_pump_on_0 = pump_on[0][dirac_energy_idx, :]
mdc_pump_on_90 = pump_on[90][dirac_energy_idx, :]

ax_f.plot(kx_axis, mdc_pump_off, 'k-', linewidth=2, label='Pump Off')
ax_f.plot(kx_axis, mdc_pump_on_0, 'b-', linewidth=1.5, label='Pump On (θ=0°)')
ax_f.plot(kx_axis, mdc_pump_on_90, 'r-', linewidth=1.5, label='Pump On (θ=90°)')
ax_f.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_f.set_ylabel('Intensity (arb. u.)')
ax_f.set_title('(f) Momentum Distribution Curve at Dirac Energy')
ax_f.legend(fontsize=9)
ax_f.axvline(x=dirac_point[1], color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure5_replica_band_analysis.png'), dpi=150)
plt.close()
print("  Saved figure5_replica_band_analysis.png")

# --- Figure 6: Floquet replica energy spacing analysis ---
print("Generating Figure 6: Floquet replica energy spacing...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Schematic of Floquet ladder
ax1 = axes[0]
photon_energy_ev = 0.248  # 5 μm -> ~0.248 eV

# Draw the Floquet ladder
for n in range(-2, 3):
    ax1.axhline(y=n * photon_energy_ev, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(0.02, n * photon_energy_ev + 0.01, f'n={n}', fontsize=10, color='gray')

# Plot measured replica energies
for rb in replica_bands:
    color = 'dodgerblue' if rb['order'] == -1 else 'crimson'
    marker = 'v' if rb['order'] == -1 else '^'
    # Energy relative to Dirac point
    e_rel = rb['energy'] - dirac_point[0]
    ax1.scatter(rb['kx'], e_rel, c=color, s=200, marker=marker,
                edgecolors='black', linewidths=1.5, zorder=10)

# Dirac point
ax1.scatter(dirac_point[1], 0, c='black', s=150, marker='*', zorder=5)

ax1.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax1.set_ylabel('$E - E_D$ (eV)')
ax1.set_title('(a) Floquet Ladder: Measured Replica Energies')

# Panel (b): Comparison with theoretical photon energy
ax2 = axes[1]
data_orders = []
data_energies = []
for rb in replica_bands:
    data_orders.append(rb['order'])
    data_energies.append(abs(rb['energy'] - dirac_point[0]))

# Add both ±1 order replicas (average the energy split for each order)
from collections import defaultdict
order_energies = defaultdict(list)
for rb in replica_bands:
    order_energies[abs(rb['order'])].append(abs(rb['energy'] - dirac_point[0]))

for o in sorted(order_energies.keys()):
    e_mean = np.mean(order_energies[o])
    e_std = np.std(order_energies[o])
    ax2.errorbar(o, e_mean, yerr=e_std, fmt='o', capsize=8, markersize=12,
                 color='darkblue', linewidth=2, label=f'Measured (n=±{o})' if o == 1 else None)
    
    # Theoretical: n * hbar*omega
    ax2.scatter(o, o * photon_energy_ev, marker='s', s=100, color='darkred', 
                zorder=5, label=f'Theory: n·ħω' if o == 1 else None)

# Diagonal line y = n * photon_energy
n_range = np.array([0.5, 1.5])
ax2.plot(n_range, n_range * photon_energy_ev, '--', color='darkred', alpha=0.5, linewidth=1.5)

ax2.set_xlabel('|Floquet Order n|')
ax2.set_ylabel('Energy Spacing |ΔE| (eV)')
ax2.set_title('(b) Energy Spacing: Measured vs. Theory')
ax2.legend()
ax2.set_xlim(0.5, 1.5)

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure6_floquet_ladder.png'), dpi=150)
plt.close()
print("  Saved figure6_floquet_ladder.png")

# --- Figure 7: Comprehensive Floquet analysis summary ---
print("Generating Figure 7: Comprehensive summary...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# (a) Pump off spectrum with annotations
ax_a = fig.add_subplot(gs[0, 0])
ax_a.pcolormesh(kx_axis, energy_axis, pump_off, shading='auto', cmap='inferno')
ax_a.scatter(dirac_point[1], dirac_point[0], c='cyan', s=150, marker='*', edgecolors='white', linewidths=1.5)
ax_a.annotate('Dirac Point', (dirac_point[1], dirac_point[0]),
              xytext=(20, 15), textcoords='offset points', fontsize=10,
              color='white', arrowprops=dict(arrowstyle='->', color='white'))
ax_a.set_title('(a) Equilibrium Spectrum (Pump Off)')
ax_a.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_a.set_ylabel('$E - E_F$ (eV)')

# (b) Pump on (θ=0°) difference
ax_b = fig.add_subplot(gs[0, 1])
vmax = np.max(np.abs(difference[0])) * 0.6
ax_b.pcolormesh(kx_axis, energy_axis, difference[0], shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
for rb in replica_bands:
    color = 'lime'
    ax_b.scatter(rb['kx'], rb['energy'], c=color, s=100, marker='D' if rb['order'] < 0 else 's',
                 edgecolors='white', linewidths=1, zorder=10)
ax_b.set_title('(b) Difference: Pump On (θ=0°)')
ax_b.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_b.set_ylabel('$E - E_F$ (eV)')

# (c) Pump on (θ=90°) difference
ax_c = fig.add_subplot(gs[0, 2])
vmax = np.max(np.abs(difference[90])) * 0.6
ax_c.pcolormesh(kx_axis, energy_axis, difference[90], shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
for rb in replica_bands:
    ax_c.scatter(rb['kx'], rb['energy'], c='lime', s=100, marker='D' if rb['order'] < 0 else 's',
                 edgecolors='white', linewidths=1, zorder=10)
ax_c.set_title('(c) Difference: Pump On (θ=90°)')
ax_c.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_c.set_ylabel('$E - E_F$ (eV)')

# (d) Replica intensity across all pump conditions
ax_d = fig.add_subplot(gs[1, 0])
# For each pump condition, compute the average intensity in the replica band regions
replica_intensities = {}
for angle in polarization_angles:
    # Average intensity in a window around each replica position
    diff = difference[angle]
    # Simple approach: take max intensity in upper and lower energy regions
    lower_mask = (energy_axis > -0.35) & (energy_axis < -0.22)
    lower_intensity = np.max(np.abs(diff[lower_mask, :]))
    upper_mask = (energy_axis > 0.14) & (energy_axis < 0.28)
    upper_intensity = np.max(np.abs(diff[upper_mask, :]))
    replica_intensities[angle] = {'lower': lower_intensity, 'upper': upper_intensity}

angles_sorted = sorted(replica_intensities.keys())
lower_vals = [replica_intensities[a]['lower'] for a in angles_sorted]
upper_vals = [replica_intensities[a]['upper'] for a in angles_sorted]
x_pos = range(len(angles_sorted))
width = 0.35
ax_d.bar([x - width/2 for x in x_pos], lower_vals, width, label='n=-1 Replica', color='dodgerblue', alpha=0.7)
ax_d.bar([x + width/2 for x in x_pos], upper_vals, width, label='n=+1 Replica', color='crimson', alpha=0.7)
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels([f'{a}°' for a in angles_sorted])
ax_d.set_xlabel('Pump Polarization Angle')
ax_d.set_ylabel('Max |ΔI| in Replica Region')
ax_d.set_title('(d) Replica Intensity vs. Polarization')
ax_d.legend()

# (e) Energy spectrum comparison: pump off vs pump on at key angles
ax_e = fig.add_subplot(gs[1, 1])
# Energy-integrated spectra 
dirac_energy_idx = dirac_indices[0]
energy_cut_indices = [dirac_energy_idx - 5, dirac_energy_idx + 5]
for angle in [0, 90, 180]:
    spec = pump_on[angle]
    integrated = np.mean(spec[energy_cut_indices[0]:energy_cut_indices[1]+1, :], axis=0)
    ax_e.plot(kx_axis, integrated, linewidth=1.5, label=f'θ={angle}°')

# Pump off
integrated_off = np.mean(pump_off[energy_cut_indices[0]:energy_cut_indices[1]+1, :], axis=0)
ax_e.plot(kx_axis, integrated_off, 'k-', linewidth=2, label='Pump Off')
ax_e.set_xlabel('$k_x$ ($\\AA^{-1}$)')
ax_e.set_ylabel('Integrated Intensity (arb. u.)')
ax_e.set_title('(e) Momentum Cuts Near Dirac Energy')
ax_e.legend(fontsize=9)

# (f) Polarization ratio analysis
ax_f = fig.add_subplot(gs[1, 2])
# Compute the ratio I(0°)/I(90°) as a function of energy
ratios = []
for e_idx in range(len(energy_axis)):
    i0 = pump_on[0][e_idx, :]
    i90 = pump_on[90][e_idx, :]
    # Ratio of max values near Dirac point
    dirac_kx_idx = dirac_indices[1]
    kx_window = 10
    i0_max = np.max(i0[max(0, dirac_kx_idx - kx_window):min(len(kx_axis), dirac_kx_idx + kx_window)])
    i90_max = np.max(i90[max(0, dirac_kx_idx - kx_window):min(len(kx_axis), dirac_kx_idx + kx_window)])
    if i90_max > 1e-3:
        ratios.append(i0_max / i90_max)
    else:
        ratios.append(np.nan)

ax_f.plot(energy_axis, ratios, 'b-', linewidth=1.5)
ax_f.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax_f.axvline(x=dirac_point[0], color='red', linestyle='--', alpha=0.5, label='Dirac Point')
n1_energy = replica_bands[0]['energy']
ax_f.axvline(x=n1_energy, color='dodgerblue', linestyle='--', alpha=0.5, label='n=-1 Replica')
n1p_energy = replica_bands[2]['energy']
ax_f.axvline(x=n1p_energy, color='crimson', linestyle='--', alpha=0.5, label='n=+1 Replica')
ax_f.set_xlabel('$E - E_F$ (eV)')
ax_f.set_ylabel('I(θ=0°) / I(θ=90°)')
ax_f.set_title('(f) Polarization Anisotropy Ratio')
ax_f.legend(fontsize=8)
ax_f.set_ylim(0.5, 2.0)

plt.tight_layout()
fig.savefig(os.path.join(IMAGE_DIR, 'figure7_comprehensive_analysis.png'), dpi=150)
plt.close()
print("  Saved figure7_comprehensive_analysis.png")

# ============================================================
# 4. SAVE INTERMEDIATE RESULTS
# ============================================================
print("\nSaving intermediate results...")

# Save difference spectra summary
diff_summary = {}
for angle in polarization_angles:
    diff_summary[f'angle_{angle}'] = {
        'mean': float(np.mean(difference[angle])),
        'std': float(np.std(difference[angle])),
        'max': float(np.max(difference[angle])),
        'min': float(np.min(difference[angle])),
        'max_abs': float(np.max(np.abs(difference[angle]))),
    }

with open(os.path.join(OUTPUT_DIR, 'difference_summary.json'), 'w') as f:
    json.dump(diff_summary, f, indent=2)

# Save replica band analysis
replica_analysis = {
    'dirac_point_energy': dirac_point[0],
    'dirac_point_kx': dirac_point[1],
    'pump_photon_energy_ev': 0.248,  # 5 μm
    'replica_bands': [],
}

for rb in replica_bands:
    replica_analysis['replica_bands'].append({
        'order': rb['order'],
        'energy': rb['energy'],
        'kx': rb['kx'],
        'intensity': rb['intensity'],
        'energy_from_dirac': rb['energy'] - dirac_point[0],
        'expected_energy': rb['order'] * 0.248 + dirac_point[0],
        'energy_deviation': (rb['energy'] - dirac_point[0]) - rb['order'] * 0.248,
    })

with open(os.path.join(OUTPUT_DIR, 'replica_band_analysis.json'), 'w') as f:
    json.dump(replica_analysis, f, indent=2)

# Save polarization analysis
pol_analysis = {
    'angles': angles_deg,
    'intensities': intensities,
    'mean_intensity': float(np.mean(intensities)),
    'std_intensity': float(np.std(intensities)),
    'ratio_parallel_perpendicular': float(I_parallel / I_perp),
    'sinusoidal_fit_params': list(popt) if 'popt' in dir() else None,
}

with open(os.path.join(OUTPUT_DIR, 'polarization_analysis.json'), 'w') as f:
    json.dump(pol_analysis, f, indent=2)

# Save data overview
data_overview = {
    'energy_axis': {
        'shape': list(energy_axis.shape),
        'range': [float(energy_axis[0]), float(energy_axis[-1])],
        'points': len(energy_axis),
    },
    'kx_axis': {
        'shape': list(kx_axis.shape),
        'range': [float(kx_axis[0]), float(kx_axis[-1])],
        'points': len(kx_axis),
    },
    'polarization_angles': [int(a) for a in polarization_angles],
    'time_delays': [float(t) for t in time_delays],
    'pump_wavelength_um': 5.0,
    'pump_photon_energy_ev': 0.248,
}

with open(os.path.join(OUTPUT_DIR, 'data_overview.json'), 'w') as f:
    json.dump(data_overview, f, indent=2)

# Save band dispersion summary
dispersion_summary = {
    'n_points': len(band_dispersion),
    'energy_range': [
        float(min(b['energy'] for b in band_dispersion)),
        float(max(b['energy'] for b in band_dispersion)),
    ],
    'kx_range': [
        float(min(b['kx'] for b in band_dispersion)),
        float(max(b['kx'] for b in band_dispersion)),
    ],
    'intensity_range': [
        float(min(b['intensity'] for b in band_dispersion)),
        float(max(b['intensity'] for b in band_dispersion)),
    ],
    'dirac_point': dirac_point,
}

with open(os.path.join(OUTPUT_DIR, 'band_dispersion_summary.json'), 'w') as f:
    json.dump(dispersion_summary, f, indent=2)

print("All intermediate results saved.")
print("\nAnalysis complete! All figures generated in report/images/")
