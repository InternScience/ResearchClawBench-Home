import json
import matplotlib.pyplot as plt
import numpy as np
import h5py

with open('data/processed_band_data.json', 'r') as f:
    d = json.load(f)

with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_on = f['pump_on_angle_0'][:]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(kx, energy, pump_on, shading='auto', cmap='magma')
plt.colorbar(im, ax=ax, label='Intensity')

# Plot extracted dispersion
disp_k = [p['kx'] for p in d['band_dispersion']]
disp_e = [p['energy'] for p in d['band_dispersion']]
ax.plot(disp_k, disp_e, 'w.', markersize=2, label='Extracted Dispersion')

# Plot replica bands
for i, b in enumerate(d['replica_bands']):
    label = 'Replica Band' if i == 0 else None
    ax.plot(b['kx'], b['energy'], 'r+', markersize=10, markeredgewidth=2, label=label)

ax.plot(d['dirac_point'][1], d['dirac_point'][0], 'y*', markersize=15, label='Dirac Point')

ax.set_title('Extracted Band Structure (Pump ON 0°)')
ax.set_xlabel('Momentum $k_x$ (Å$^{-1}$)')
ax.set_ylabel('Energy $E - E_F$ (eV)')
ax.legend()
plt.tight_layout()
plt.savefig('report/images/band_dispersion.png')

