"""
Parse and explore the M-AI-Synth dataset.
"""
import json
import numpy as np

# Read raw data
with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

# Section 1: property_prediction.py data (lines 1-4)
lattice_dim = json.loads(lines[1].strip())
x_coords = json.loads(lines[2].strip())
atom_types = json.loads(lines[3].strip())
targets = json.loads(lines[4].strip())

print(f"Lattice dimensions: {len(lattice_dim)} values")
print(f"X coordinates: {len(x_coords)} values, range [{min(x_coords):.2f}, {max(x_coords):.2f}]")
print(f"Atom types: {len(atom_types)} values, unique: {sorted(set(atom_types))}")
print(f"Targets: {len(targets)} values, range [{min(targets):.4f}, {max(targets):.4f}]")

# Section 2: structure_generation.py data (lines 7-8)
a_vals = json.loads(lines[7].strip())
b_vals = json.loads(lines[8].strip())

print(f"Lattice a: {len(a_vals)} values, mean={np.mean(a_vals):.4f}, std={np.std(a_vals):.4f}")
print(f"Lattice b: {len(b_vals)} values, mean={np.mean(b_vals):.4f}, std={np.std(b_vals):.4f}")

# Section 3: autonomous_optimization.py data (lines 11-16)
temp_range = json.loads(lines[11].strip())
pres_range = json.loads(lines[12].strip())
target_temp = json.loads(lines[13].strip())
target_pres = json.loads(lines[14].strip())
target_yield = json.loads(lines[15].strip())
target_time = json.loads(lines[16].strip())

print(f"Temp range: {temp_range}")
print(f"Pressure range: {pres_range}")
print(f"Target temp: {target_temp}")
print(f"Target pressure: {target_pres}")
print(f"Target yield: {target_yield}")
print(f"Target time: {target_time}")

# Save parsed data
np.savez('outputs/parsed_data.npz',
         lattice_dim=np.array(lattice_dim),
         x_coords=np.array(x_coords),
         atom_types=np.array(atom_types),
         targets=np.array(targets),
         a_vals=np.array(a_vals),
         b_vals=np.array(b_vals),
         temp_range=np.array(temp_range),
         pres_range=np.array(pres_range),
         target_temp=np.array(target_temp),
         target_pres=np.array(target_pres),
         target_yield=np.array(target_yield),
         target_time=np.array(target_time))

print("Saved parsed data to outputs/parsed_data.npz")
