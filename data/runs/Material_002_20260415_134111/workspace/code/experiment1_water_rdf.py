"""
MACE-MP-0 Foundation Model Reproduction
Experiment 1: Liquid Water RDF Simulation (reduced steps for CPU feasibility)
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import MACECalculator

# ---- Parameters from dataset ----
n_water = 32
box_size = 12.0  # Angstrom, cubic
temperature = 330  # K
timestep = 0.5  # fs
n_steps = 300  # Reduced for CPU feasibility
friction = 0.01  # fs^-1

# ---- Build water box ----
single_water_positions = np.array([
    [0.000000, 0.000000, 0.119262],  # O
    [0.000000, 0.763239, -0.477047],  # H
    [0.000000, -0.763239, -0.477047], # H
])

np.random.seed(42)
all_positions = []
all_symbols = []
for i in range(n_water):
    offset = np.random.uniform(1.5, box_size - 1.5, size=3)
    mol_pos = single_water_positions + offset
    mol_pos = mol_pos % box_size
    all_positions.append(mol_pos)
    all_symbols.extend(['O', 'H', 'H'])

all_positions = np.vstack(all_positions)
water_system = Atoms(symbols=all_symbols,
                     positions=all_positions,
                     cell=[box_size, box_size, box_size],
                     pbc=True)

print(f"Water system: {len(water_system)} atoms, {n_water} molecules")

# ---- Load MACE calculator ----
calc = MACECalculator(model_paths='mace_mp_0.model', device='cpu')
water_system.calc = calc

E = water_system.get_potential_energy()
print(f"Initial energy: {E:.4f} eV")

# ---- Run MD, saving frames ----
os.makedirs('outputs', exist_ok=True)

frames = []
step_count = [0]

def save_frame():
    step_count[0] += 1
    if step_count[0] % 5 == 0:
        frames.append(water_system.get_positions().copy())
        if step_count[0] % 50 == 0:
            E = water_system.get_potential_energy()
            print(f"Step {step_count[0]}/{n_steps}, E={E:.4f} eV")

dyn = Langevin(water_system, timestep * units.fs, temperature_K=temperature,
               friction=friction)
dyn.attach(save_frame, interval=1)

print("Running MD simulation...")
dyn.run(n_steps)
print(f"MD simulation complete! Saved {len(frames)} frames")

# ---- Compute RDFs ----
from ase.geometry.analysis import Analysis

# Reconstruct trajectory as list of Atoms objects
trajectory = []
for i, pos in enumerate(frames):
    atoms = Atoms(symbols=all_symbols,
                  positions=pos,
                  cell=[box_size, box_size, box_size],
                  pbc=True)
    trajectory.append(atoms)

print(f"Computing RDFs from {len(trajectory)} frames...")

# Compute RDFs using manual method
def compute_rdf(trajectory, pair, r_max=6.0, n_bins=100, cell=None):
    """Compute RDF for a pair of element types."""
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)
    
    elem1, elem2 = pair
    n_frames = len(trajectory)
    
    for atoms in trajectory:
        symbols = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        cell_mat = atoms.get_cell()[:]
        
        idx1 = [i for i, s in enumerate(symbols) if s == elem1]
        idx2 = [i for i, s in enumerate(symbols) if s == elem2]
        
        n1 = len(idx1)
        n2 = len(idx2)
        
        for i in idx1:
            for j in idx2:
                if i == j:
                    continue
                # Minimum image convention
                delta = pos[j] - pos[i]
                # Apply PBC
                delta -= np.round(delta @ np.linalg.inv(cell_mat)) @ cell_mat
                r = np.linalg.norm(delta)
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 1
    
    # Normalize
    volume = np.linalg.det(cell_mat)
    rho2 = n2 / volume  # number density of type 2
    
    for i in range(n_bins):
        r = r_centers[i]
        shell_volume = 4.0 / 3.0 * np.pi * ((r + dr/2)**3 - (r - dr/2)**3)
        hist[i] /= (n1 * rho2 * shell_volume * n_frames)
    
    return r_centers, hist

# Compute O-O, O-H, H-H RDFs
r_oo, g_oo = compute_rdf(trajectory, ('O', 'O'), r_max=6.0, n_bins=100)
r_oh, g_oh = compute_rdf(trajectory, ('O', 'H'), r_max=6.0, n_bins=100)
r_hh, g_hh = compute_rdf(trajectory, ('H', 'H'), r_max=6.0, n_bins=100)

# Save RDF data
rdf_data = {
    'r_oo': r_oo.tolist(),
    'g_oo': g_oo.tolist(),
    'r_oh': r_oh.tolist(),
    'g_oh': g_oh.tolist(),
    'r_hh': r_hh.tolist(),
    'g_hh': g_hh.tolist(),
    'n_frames': len(trajectory),
    'n_steps': n_steps,
    'temperature': temperature,
    'box_size': box_size,
    'n_water': n_water
}

with open('outputs/water_rdf_data.json', 'w') as f:
    json.dump(rdf_data, f, indent=2)

print("RDF data saved to outputs/water_rdf_data.json")
print(f"O-O RDF peak: r={r_oo[np.argmax(g_oo)]:.2f} Å, g={np.max(g_oo):.2f}")
print(f"O-H RDF peak: r={r_oh[np.argmax(g_oh)]:.2f} Å, g={np.max(g_oh):.2f}")
