import json
import numpy as np

with open('data/processed_band_data.json', 'r') as f:
    d = json.load(f)

pump_energy = d['pump_energy']
dirac_point = d['dirac_point']
replica_bands = d['replica_bands']

print(f"Pump energy: {pump_energy} eV")
print(f"Dirac point: E = {dirac_point[0]} eV, kx = {dirac_point[1]}")

for i, rb in enumerate(replica_bands):
    print(f"Replica Band {i+1}: E = {rb['energy']:.4f} eV, kx = {rb['kx']:.4f}, order = {rb['order']}")
    expected_e = dirac_point[0] + rb['order'] * pump_energy
    print(f"  Expected E = {expected_e:.4f} eV")
    print(f"  Difference = {np.abs(rb['energy'] - expected_e):.4f} eV")

