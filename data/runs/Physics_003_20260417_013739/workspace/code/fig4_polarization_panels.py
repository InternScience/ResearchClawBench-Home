"""Figure 4: Multi-panel comparison across polarization angles."""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
outdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/report/images'

with h5py.File(f'{datadir}/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    angles = f['polarization_angles'][:]
    pump_energy = f.attrs['pump_energy_eV']
    
    pump_on_data = {}
    for angle in angles:
        pump_on_data[angle] = f[f'pump_on_angle_{angle}'][:]

dirac_E = -0.0427

# Figure 4: Difference spectra for all polarization angles
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True, sharex=True)
axes_flat = axes.flatten()

# First panel: pump-off reference
ax = axes_flat[0]
vmin = np.percentile(pump_off, 5)
vmax = np.percentile(pump_off, 99)
im = ax.pcolormesh(kx, energy, pump_off, shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax.set_title('Pump Off', fontsize=12)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.axhline(y=dirac_E, color='cyan', linestyle='--', alpha=0.5, linewidth=1)

# Remaining panels: difference spectra for each angle
for i, angle in enumerate(angles):
    ax = axes_flat[i+1]
    diff = pump_on_data[angle] - pump_off
    vlim = np.percentile(np.abs(diff), 98)
    im = ax.pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_title(f'$\\theta_p = {angle}°$ (Diff)', fontsize=12)
    ax.axhline(y=dirac_E + pump_energy, color='green', linestyle=':', alpha=0.7, linewidth=1)
    ax.axhline(y=dirac_E - pump_energy, color='green', linestyle=':', alpha=0.7, linewidth=1)
    ax.axhline(y=dirac_E, color='black', linestyle='--', alpha=0.5, linewidth=1)

for ax in axes[1,:]:
    ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=12)
for ax in axes[:,0]:
    ax.set_ylabel('Energy (eV)', fontsize=12)

plt.suptitle('Polarization-Dependent tr-ARPES: Difference Spectra\n'
             r'Mid-IR Pump $\hbar\omega$ = 0.248 eV, Monolayer Graphene', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig4_polarization_panels.png', dpi=150, bbox_inches='tight')
print("Saved fig4_polarization_panels.png")
plt.close()
