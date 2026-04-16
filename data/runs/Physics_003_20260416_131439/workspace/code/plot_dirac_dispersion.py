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

# Dirac point
dp_e, dp_k = d['dirac_point']
ax.plot(dp_k, dp_e, 'y*', markersize=15, label='Dirac Point')

# Plot extracted dispersion
disp_k = [p['kx'] for p in d['band_dispersion']]
disp_e = [p['energy'] for p in d['band_dispersion']]
ax.plot(disp_k, disp_e, 'w.', markersize=2, label='Extracted Dispersion')

pump_energy = d['pump_energy']

for i, rb in enumerate(d['replica_bands']):
    label = 'Replica Band Peaks' if i == 0 else None
    ax.plot(rb['kx'], rb['energy'], 'r+', markersize=10, markeredgewidth=2, label=label)
    
    # Plot expected replica bands
    order = rb['order']
    expected_disp_e = [e + order * pump_energy for e in disp_e]
    label_expected = f'Expected Replica (n={order})' if i in [0, 2] else None
    ax.plot(disp_k, expected_disp_e, 'c--', linewidth=1, label=label_expected)

ax.set_title('Band Structure and Floquet Replicas (Pump ON 0°)')
ax.set_xlabel('Momentum $k_x$ (Å$^{-1}$)')
ax.set_ylabel('Energy $E - E_F$ (eV)')
ax.legend()
plt.tight_layout()
plt.savefig('report/images/band_dispersion_with_expected.png')

