"""Figure 2: Pump-on ARPES spectrum showing Floquet-Bloch replica bands."""
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
    pump_on_0 = f['pump_on_angle_0'][:]
    pump_energy = f.attrs['pump_energy_eV']

# Figure 2: Side-by-side pump-off and pump-on
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

vmin_off = np.percentile(pump_off, 5)
vmax_off = np.percentile(pump_off, 99)

im0 = axes[0].pcolormesh(kx, energy, pump_off, shading='auto', cmap='hot', 
                          vmin=vmin_off, vmax=vmax_off)
axes[0].set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=14)
axes[0].set_ylabel('Energy (eV)', fontsize=14)
axes[0].set_title('(a) Pump Off (Equilibrium)', fontsize=14)
axes[0].axhline(y=0, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
plt.colorbar(im0, ax=axes[0], label='Intensity')

vmin_on = np.percentile(pump_on_0, 5)
vmax_on = np.percentile(pump_on_0, 99)

im1 = axes[1].pcolormesh(kx, energy, pump_on_0, shading='auto', cmap='hot',
                          vmin=vmin_on, vmax=vmax_on)
axes[1].set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=14)
axes[1].set_title(r'(b) Pump On ($\theta_p = 0°$)', fontsize=14)
axes[1].axhline(y=0, color='cyan', linestyle='--', alpha=0.7, linewidth=1)

# Mark expected replica positions
dirac_E = -0.0427  # Dirac point energy from processed data
axes[1].axhline(y=dirac_E + pump_energy, color='lime', linestyle=':', alpha=0.8, 
                label=f'$E_D + \\hbar\\omega$ = {dirac_E + pump_energy:.3f} eV')
axes[1].axhline(y=dirac_E - pump_energy, color='lime', linestyle=':', alpha=0.8,
                label=f'$E_D - \\hbar\\omega$ = {dirac_E - pump_energy:.3f} eV')
axes[1].axhline(y=dirac_E, color='yellow', linestyle='--', alpha=0.7,
                label=f'$E_D$ = {dirac_E:.3f} eV')
axes[1].legend(loc='upper right', fontsize=9, framealpha=0.8)

plt.colorbar(im1, ax=axes[1], label='Intensity')
plt.suptitle('tr-ARPES Spectra: Observation of Floquet-Bloch Replica Bands\n'
             r'Mid-IR Pump: $\lambda$ = 5 μm, $\hbar\omega$ = 0.248 eV', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig2_pump_on_comparison.png', dpi=150, bbox_inches='tight')
print("Saved fig2_pump_on_comparison.png")
plt.close()
