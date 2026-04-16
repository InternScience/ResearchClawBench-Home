import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('data/raw_trARPES_data.h5', 'r')
energy = f['energy_axis'][:]
kx = f['kx_axis'][:]
pump_off = f['pump_off_spectrum'][:]
pump_on_0 = f['pump_on_angle_0'][:]
pump_on_90 = f['pump_on_angle_90'][:]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

im0 = axes[0].pcolormesh(kx, energy, pump_off, shading='auto', cmap='magma')
axes[0].set_title('Pump OFF')
axes[0].set_xlabel('$k_x$ (Å$^{-1}$)')
axes[0].set_ylabel('$E - E_F$ (eV)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(kx, energy, pump_on_0, shading='auto', cmap='magma')
axes[1].set_title('Pump ON (0°)')
axes[1].set_xlabel('$k_x$ (Å$^{-1}$)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(kx, energy, pump_on_90, shading='auto', cmap='magma')
axes[2].set_title('Pump ON (90°)')
axes[2].set_xlabel('$k_x$ (Å$^{-1}$)')
plt.colorbar(im2, ax=axes[2])

diff = pump_on_0 - pump_off
im3 = axes[3].pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
axes[3].set_title('Difference (ON 0° - OFF)')
axes[3].set_xlabel('$k_x$ (Å$^{-1}$)')
plt.colorbar(im3, ax=axes[3])

plt.tight_layout()
plt.savefig('report/images/spectra_comparison.png')

