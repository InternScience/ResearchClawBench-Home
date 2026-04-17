"""Figure 6: Energy Distribution Curves (EDCs) through key momentum positions."""
import numpy as np
import h5py
import json
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

with open(f'{datadir}/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

dirac_E = band_data['dirac_point'][0]  # -0.3 (actually this is kx of Dirac point)
dirac_kx = band_data['dirac_point'][1]  # -0.0427 (energy)

# Wait - let me check the convention
print(f"Dirac point from JSON: {band_data['dirac_point']}")
print(f"Dirac indices: {band_data['dirac_indices']}")

# The dirac_point is [energy_value, kx_value] or [kx_value, energy_value]?
# dirac_indices = [91, 0] -> energy_idx=91, kx_idx=0
dirac_energy_idx = band_data['dirac_indices'][0]
dirac_kx_idx = band_data['dirac_indices'][1]
print(f"Energy at idx 91: {energy[91]:.4f}")
print(f"kx at idx 0: {kx[0]:.4f}")

# So dirac_point = [energy_at_idx91, kx_at_idx0] = [-0.3, -0.042...]
# Wait, energy[91] should be around 0 if Dirac is near Fermi level
# Let me check
print(f"Energy[91] = {energy[91]:.4f}")
print(f"kx[0] = {kx[0]:.4f}")

# Hmm, energy[91] = -0.0427 and kx[0] = -0.3
# So dirac_point = [kx_value, energy_value] or [energy, kx]?
# dirac_point = [-0.3, -0.0427]
# energy[91] ≈ -0.0427, kx[0] = -0.3
# So dirac_point = [kx, energy] where kx=-0.3 is actually energy[91]=-0.0427
# Actually: dirac_indices = [91, 0] means energy_idx=91, kx_idx=0
# energy[91] = some value, kx[0] = -0.3
# dirac_point[0] = -0.3 = kx[0], dirac_point[1] = -0.0427 = energy[91]
# So dirac_point = [kx_value, energy_value]
# But that seems odd. Let me verify:
print(f"\nVerification:")
print(f"energy[91] = {energy[91]:.6f}")
print(f"kx[0] = {kx[0]:.6f}")
print(f"dirac_point = {band_data['dirac_point']}")
# OK so dirac_point[0] = -0.3 and kx[0] = -0.3 -> dirac_point[0] is kx value
# dirac_point[1] = -0.0427 and energy[91] = -0.0427 -> dirac_point[1] is energy value

# Actually wait, let me look at the pump_off spectrum to find the actual Dirac cone
# The Dirac point should be where intensity is maximum along the cone
# kx[0] = -0.3 is at the edge - that's unusual for a Dirac point

# Let me look at the spectrum at kx=0 (center)
kx_center_idx = np.argmin(np.abs(kx))
print(f"\nkx center idx: {kx_center_idx}, kx={kx[kx_center_idx]:.4f}")
edc_center = pump_off[:, kx_center_idx]
print(f"Max intensity at kx=0: {edc_center.max():.2f} at energy={energy[np.argmax(edc_center)]:.4f}")

# Find where the Dirac cone is brightest
max_idx = np.unravel_index(np.argmax(pump_off), pump_off.shape)
print(f"\nGlobal max: energy={energy[max_idx[0]]:.4f}, kx={kx[max_idx[1]]:.4f}, I={pump_off[max_idx[0], max_idx[1]]:.2f}")

