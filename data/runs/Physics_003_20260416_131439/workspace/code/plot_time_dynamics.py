import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

with open('data/processed_band_data.json', 'r') as f:
    d = json.load(f)

f = h5py.File('data/raw_trARPES_data.h5', 'r')
time_delays = f['time_delays'][:]
energy = f['energy_axis'][:]
kx = f['kx_axis'][:]

# The datasets are pump_on_angle_0, etc., but time_delays are also there?
# Let's check the shape of pump_on_angle_0. It's (200, 150), so it's a single time delay?
# Or maybe the time_delays correspond to something else?
print("pump_on_angle_0 shape:", f['pump_on_angle_0'].shape)
print("time_delays:", time_delays)

