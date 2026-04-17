"""Figure 6: EDC analysis showing replica bands."""
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'

with h5py.File(f'{datadir}/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on_0 = f['pump_on_angle_0'][:]
    pump_energy = f.attrs['pump_energy_eV']

with open(f'{datadir}/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

# Dirac point is at kx≈0, E≈-0.0427
dirac_E = -0.0427
replica_plus = dirac_E + pump_energy   # ~0.205
replica_minus = dirac_E - pump_energy  # ~-0.291

# EDC at several kx positions near the replica bands
# Replica bands are at kx ≈ ±0.04
kx_positions = [0.0, 0.04, -0.04, 0.08, -0.08]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel (a): EDCs at kx=0 (Dirac point)
kx_idx = np.argmin(np.abs(kx - 0.0))
edc_off = pump_off[:, kx_idx]
edc_on = pump_on_0[:, kx_idx]
edc_diff = edc_on - edc_off

ax = axes[0]
ax.plot(energy, edc_off, 'b-', linewidth=1.5, label='Pump Off', alpha=0.8)
ax.plot(energy, edc_on, 'r-', linewidth=1.5, label='Pump On ($\\theta_p=0°$)', alpha=0.8)
ax.axvline(x=dirac_E, color='black', linestyle='--', alpha=0.5, label=f'$E_D$ = {dirac_E:.3f} eV')
ax.axvline(x=replica_plus, color='green', linestyle=':', alpha=0.7, label=f'$E_D + \\hbar\\omega$ = {replica_plus:.3f} eV')
ax.axvline(x=replica_minus, color='green', linestyle=':', alpha=0.7, label=f'$E_D - \\hbar\\omega$ = {replica_minus:.3f} eV')
ax.set_xlabel('Energy (eV)', fontsize=14)
ax.set_ylabel('Intensity (arb. units)', fontsize=14)
ax.set_title(f'(a) EDC at $k_x$ = 0 $\\AA^{{-1}}$', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel (b): EDCs at multiple kx positions showing replica structure
ax = axes[1]
colors = ['blue', 'red', 'green', 'purple', 'orange']
offsets = [0, 15, 30, 45, 60]

for i, kx_val in enumerate([0.0, 0.03, 0.05, 0.08, 0.12]):
    kx_idx = np.argmin(np.abs(kx - kx_val))
    diff = pump_on_0[:, kx_idx] - pump_off[:, kx_idx]
    ax.plot(energy, diff + offsets[i], color=colors[i], linewidth=1.2,
            label=f'$k_x$ = {kx[kx_idx]:.3f} $\\AA^{{-1}}$')

ax.axvline(x=dirac_E, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=replica_plus, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=replica_minus, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Energy (eV)', fontsize=14)
ax.set_ylabel('$\\Delta I$ + offset (arb. units)', fontsize=14)
ax.set_title('(b) Difference EDCs at Various $k_x$', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle('Energy Distribution Curves: Evidence for Floquet-Bloch Replica Bands', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig6_edc_analysis.png', dpi=150, bbox_inches='tight')
print("Saved fig6_edc_analysis.png")
plt.close()
