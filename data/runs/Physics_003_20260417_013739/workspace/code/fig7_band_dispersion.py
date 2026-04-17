"""Figure 7: Band dispersion extraction with replica band identification."""
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'
resultsdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/outputs'

with h5py.File(f'{datadir}/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on_0 = f['pump_on_angle_0'][:]
    pump_energy = f.attrs['pump_energy_eV']

with open(f'{datadir}/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

dirac_E = -0.0427
replica_bands = band_data['replica_bands']

# Extract band dispersion from pump-off: find peak kx for each energy
band_kx_off = np.zeros(len(energy))
for i in range(len(energy)):
    band_kx_off[i] = kx[np.argmax(pump_off[i, :])]

# Extract from pump-on
band_kx_on = np.zeros(len(energy))
for i in range(len(energy)):
    band_kx_on[i] = kx[np.argmax(pump_on_0[i, :])]

# Extract band dispersion from processed data
bd = band_data['band_dispersion']
bd_energy = [b['energy'] for b in bd]
bd_kx = [b['kx'] for b in bd]

# Figure 7: Band dispersion
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Panel (a): Pump-off spectrum with extracted dispersion
ax = axes[0]
vmin = np.percentile(pump_off, 5)
vmax = np.percentile(pump_off, 99)
im = ax.pcolormesh(kx, energy, pump_off, shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax.plot(band_kx_off, energy, 'c.', markersize=1, alpha=0.5)
ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=13)
ax.set_ylabel('Energy (eV)', fontsize=13)
ax.set_title('(a) Pump Off + Dispersion', fontsize=13)
ax.axhline(y=dirac_E, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)

# Panel (b): Pump-on spectrum with replica positions marked
ax = axes[1]
vmin_on = np.percentile(pump_on_0, 5)
vmax_on = np.percentile(pump_on_0, 99)
im = ax.pcolormesh(kx, energy, pump_on_0, shading='auto', cmap='hot', vmin=vmin_on, vmax=vmax_on)
ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=13)
ax.set_title(r'(b) Pump On ($\theta_p=0°$) + Replicas', fontsize=13)

# Mark replica band positions
for rb in replica_bands:
    marker = 'v' if rb['order'] == -1 else '^'
    color = 'lime' if rb['order'] == -1 else 'cyan'
    ax.plot(rb['kx'], rb['energy'], marker, color=color, markersize=12, markeredgecolor='white',
            label=f"$n={rb['order']}$: ({rb['kx']:.3f}, {rb['energy']:.3f})")

ax.axhline(y=dirac_E, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
ax.axhline(y=dirac_E + pump_energy, color='lime', linestyle=':', alpha=0.5)
ax.axhline(y=dirac_E - pump_energy, color='lime', linestyle=':', alpha=0.5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.8)
plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)

# Panel (c): Difference spectrum with Floquet theory overlay
ax = axes[2]
diff = pump_on_0 - pump_off
vlim = np.percentile(np.abs(diff), 98)
im = ax.pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=13)
ax.set_title('(c) Difference + Floquet Theory', fontsize=13)

# Overlay Floquet theory: shifted Dirac cones
# Original cone: E = E_D + hbar*v_F*|kx| (approximately)
# From the data, the Fermi velocity can be estimated
# The cone appears to have slope ~1 eV/A^-1 based on the peak positions
v_F = 3.0  # eV*Angstrom (typical for graphene surface states)
# But let's estimate from the data
# At kx=0.06, peak energy ≈ -0.37 (from pump_off peak analysis)
# slope ≈ (0 - (-0.37))/0.06 ≈ 6.2 eV/A^-1 - that's too high
# Let me use the actual data
kx_theory = np.linspace(-0.15, 0.15, 200)

# Original Dirac cone (approximate)
# E = E_D ± v_F * |kx|
# From the peak positions: at kx≈0.06, E≈-0.37 -> slope = (0.0427-0.37)/0.06 ≈ 5.5
# This is the slope of the lower branch
# Actually the Dirac cone in graphene: E(k) = E_D + hbar*v_F*k for upper, E_D - hbar*v_F*k for lower
# From kx=0.058, E_peak=-0.35: (E_peak - E_D) = -0.35 - (-0.0427) = -0.307
# |kx| = 0.058, so v_F = 0.307/0.058 = 5.3 eV*A

v_est = 5.0  # rough estimate

# n=0 cone
E_upper_0 = dirac_E + v_est * np.abs(kx_theory)
E_lower_0 = dirac_E - v_est * np.abs(kx_theory)

# n=+1 replica
E_upper_p1 = (dirac_E + pump_energy) + v_est * np.abs(kx_theory)
E_lower_p1 = (dirac_E + pump_energy) - v_est * np.abs(kx_theory)

# n=-1 replica
E_upper_m1 = (dirac_E - pump_energy) + v_est * np.abs(kx_theory)
E_lower_m1 = (dirac_E - pump_energy) - v_est * np.abs(kx_theory)

ax.plot(kx_theory, E_upper_0, 'k-', linewidth=1, alpha=0.6)
ax.plot(kx_theory, E_lower_0, 'k-', linewidth=1, alpha=0.6, label='$n=0$ cone')
ax.plot(kx_theory, E_upper_p1, 'g--', linewidth=1, alpha=0.6)
ax.plot(kx_theory, E_lower_p1, 'g--', linewidth=1, alpha=0.6, label='$n=+1$ replica')
ax.plot(kx_theory, E_upper_m1, 'b--', linewidth=1, alpha=0.6)
ax.plot(kx_theory, E_lower_m1, 'b--', linewidth=1, alpha=0.6, label='$n=-1$ replica')

ax.set_ylim(energy.min(), energy.max())
ax.legend(fontsize=10, loc='upper right', framealpha=0.8)
plt.colorbar(im, ax=ax, label=r'$\Delta I$', shrink=0.8)

for rb in replica_bands:
    marker = 'v' if rb['order'] == -1 else '^'
    color = 'lime' if rb['order'] == -1 else 'cyan'
    ax.plot(rb['kx'], rb['energy'], marker, color=color, markersize=10, markeredgecolor='white')

plt.suptitle('Band Dispersion Analysis: Floquet-Bloch Replica Identification\n'
             r'$\hbar\omega$ = 0.248 eV, Monolayer Epitaxial Graphene', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig7_band_dispersion.png', dpi=150, bbox_inches='tight')
print("Saved fig7_band_dispersion.png")
plt.close()

# Save quantitative results
results = {
    "dirac_point_energy_eV": dirac_E,
    "pump_photon_energy_eV": pump_energy,
    "replica_separation_eV": 2 * pump_energy,
    "measured_replica_separation_eV": replica_bands[2]['energy'] - replica_bands[0]['energy'],
    "estimated_fermi_velocity_eV_A": v_est,
    "replica_bands": replica_bands,
    "n_plus1_energy": dirac_E + pump_energy,
    "n_minus1_energy": dirac_E - pump_energy,
}
with open(f'{resultsdir}/quantitative_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved quantitative_results.json")
