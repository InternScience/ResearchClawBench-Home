"""Figure 3: Difference spectrum highlighting Floquet-Bloch replica bands."""
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

# Compute difference
diff = pump_on_0 - pump_off

print(f"Difference range: {diff.min():.4f} to {diff.max():.4f}")
print(f"Difference mean: {diff.mean():.4f}, std: {diff.std():.4f}")

# Figure 3: Difference spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

vlim = np.percentile(np.abs(diff), 98)
im = ax.pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r',
                    vmin=-vlim, vmax=vlim)
ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title(r'Difference Spectrum: Pump On ($\theta_p=0°$) $-$ Pump Off' + '\n'
             r'Revealing Floquet-Bloch Replica Bands', fontsize=14)
plt.colorbar(im, ax=ax, label=r'$\Delta I$ (arb. units)')

dirac_E = -0.0427
ax.axhline(y=dirac_E, color='black', linestyle='--', alpha=0.7, linewidth=1.5,
           label=f'Dirac Point ($E_D$ = {dirac_E:.3f} eV)')
ax.axhline(y=dirac_E + pump_energy, color='green', linestyle='--', alpha=0.8, linewidth=1.5,
           label=f'$n=+1$ replica ($E_D + \\hbar\\omega$)')
ax.axhline(y=dirac_E - pump_energy, color='green', linestyle='--', alpha=0.8, linewidth=1.5,
           label=f'$n=-1$ replica ($E_D - \\hbar\\omega$)')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{outdir}/fig3_difference_spectrum.png', dpi=150, bbox_inches='tight')
print("Saved fig3_difference_spectrum.png")
plt.close()
