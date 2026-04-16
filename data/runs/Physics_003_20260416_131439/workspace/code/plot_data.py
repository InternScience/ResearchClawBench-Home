import h5py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
with open('data/processed_band_data.json', 'r') as f:
    processed = json.load(f)

# Load raw data
with h5py.File('data/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_on = f['pump_on_angle_0'][:]

# Plot 1: Pump off vs Pump on
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

im1 = ax1.pcolormesh(kx, energy, pump_off, shading='auto', cmap='magma')
ax1.set_title('Pump OFF')
ax1.set_xlabel('Momentum $k_x$ (Å$^{-1}$)')
ax1.set_ylabel('Energy $E - E_F$ (eV)')
plt.colorbar(im1, ax=ax1)

im2 = ax2.pcolormesh(kx, energy, pump_on, shading='auto', cmap='magma')
ax2.set_title('Pump ON (0°)')
ax2.set_xlabel('Momentum $k_x$ (Å$^{-1}$)')
ax2.set_ylabel('Energy $E - E_F$ (eV)')
plt.colorbar(im2, ax=ax2)

# Overlay replica bands
for b in processed['replica_bands']:
    ax2.plot(b['kx'], b['energy'], 'w+', markersize=10)

plt.tight_layout()
plt.savefig('report/images/trarpes_spectra.png')
plt.close()

# Plot 2: Polarization dependence
df = pd.read_csv('data/polarization_dependence_data.csv')
plt.figure(figsize=(6, 4))
plt.plot(df['angle_degrees'], df['intensity'], 'ko-')
plt.title('Replica Band Intensity vs Polarization Angle')
plt.xlabel('Polarization Angle (°)')
plt.ylabel('Intensity (arb. units)')
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/polarization_dependence.png')
plt.close()

