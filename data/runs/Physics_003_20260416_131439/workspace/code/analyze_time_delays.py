import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('data/raw_trARPES_data.h5', 'r')
energy = f['energy_axis'][:]
kx = f['kx_axis'][:]
time_delays = f['time_delays'][:]

# Find the peak intensity of a replica band over time
# We know from processed_band_data.json that one replica band is at:
# kx: -0.046, energy: -0.29
# Let's find the indices:
kx_idx = np.argmin(np.abs(kx - (-0.046)))
e_idx = np.argmin(np.abs(energy - (-0.29)))

# Wait, the time-resolved data isn't explicitly provided as a 4D array in raw_trARPES_data.h5. 
# Let's check the keys again.
print(list(f.keys()))
