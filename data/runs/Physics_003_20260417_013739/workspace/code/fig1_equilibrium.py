"""Figure 1: Equilibrium (pump-off) ARPES spectrum showing Dirac cone."""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'

with h5py.File(f'{datadir}/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on_0 = f['pump_on_angle_0'][:]
    pump_energy = f.attrs['pump_energy_eV']

print(f"Energy range: {energy.min():.3f} to {energy.max():.3f} eV")
print(f"kx range: {kx.min():.3f} to {kx.max():.3f} A^-1")
print(f"Pump energy: {pump_energy} eV")
print(f"Pump-off shape: {pump_off.shape}")
print(f"Pump-off intensity range: {pump_off.min():.3f} to {pump_off.max():.3f}")

# Figure 1: Equilibrium Dirac cone
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.pcolormesh(kx, energy, pump_off, shading='auto', cmap='hot', 
                    vmin=np.percentile(pump_off, 5), vmax=np.percentile(pump_off, 99))
ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Equilibrium ARPES Spectrum (Pump Off)\nMonolayer Epitaxial Graphene', fontsize=14)
plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='Fermi Level')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{outdir}/fig1_equilibrium_dirac.png', dpi=150, bbox_inches='tight')
print("Saved fig1_equilibrium_dirac.png")
plt.close()
