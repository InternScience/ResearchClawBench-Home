#!/usr/bin/env python3
"""
Experiment 1: Liquid Water RDF Simulation using MACE-MP-0
Reproduces the water radial distribution function with a 32-molecule system.
Optimized for speed: fewer steps, less frequent trajectory saving.
"""

import os
import sys
import json
import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp
import time

# ── Parameters from the dataset ──────────────────────────────────
N_MOLECULES = 32
BOX_SIZE = 12.0  # Å, cubic
TEMPERATURE = 330  # K
TIMESTEP = 0.5  # fs
N_STEPS = 500  # Reduced for computational feasibility
FRICTION = 0.01  # fs⁻¹

# Water molecule coordinates
H2O_COORDS = np.array([
    [0.000000, 0.000000, 0.119262],   # O
    [0.000000, 0.763239, -0.477047],  # H
    [0.000000, -0.763239, -0.477047], # H
])

def create_water_box(n_molecules, box_size):
    """Create a box of water molecules with random orientations."""
    from scipy.spatial.transform import Rotation as R
    
    positions = []
    symbols = []
    
    n_per_side = int(np.ceil(n_molecules ** (1/3)))
    spacing = box_size / n_per_side
    
    count = 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            for k in range(n_per_side):
                if count >= n_molecules:
                    break
                center = np.array([i, j, k]) * spacing + spacing / 2
                rot = R.random().as_matrix()
                mol_coords = H2O_COORDS @ rot.T + center
                
                positions.extend(mol_coords.tolist())
                symbols.extend(['O', 'H', 'H'])
                count += 1
    
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([box_size, box_size, box_size])
    atoms.set_pbc([True, True, True])
    return atoms

def compute_rdf(atoms_trajectory, r_max=6.0, n_bins=200, element='O'):
    """Compute O-O radial distribution function from trajectory."""
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    hist = np.zeros(n_bins)
    
    o_indices = np.array([i for i, s in enumerate(atoms_trajectory[0].get_chemical_symbols()) if s == element])
    n_o = len(o_indices)
    
    for atoms in atoms_trajectory:
        cell_diag = atoms.get_cell().diagonal()
        positions = atoms.get_positions()
        o_positions = positions[o_indices]
        
        for i in range(n_o):
            for j in range(i + 1, n_o):
                delta = o_positions[i] - o_positions[j]
                delta = delta - cell_diag * np.round(delta / cell_diag)
                dist = np.sqrt(np.sum(delta**2))
                if dist < r_max:
                    bin_idx = int(dist / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2
    
    n_frames = len(atoms_trajectory)
    V = atoms_trajectory[0].get_volume()
    rho = n_o / V
    
    for i in range(n_bins):
        r_inner = r_edges[i]
        r_outer = r_edges[i + 1]
        shell_vol = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        hist[i] = hist[i] / (n_frames * n_o * rho * shell_vol)
    
    return r_centers, hist

def main():
    print("=" * 60)
    print("Experiment 1: Water RDF Simulation")
    print("=" * 60)
    
    t_start = time.time()
    
    # Load MACE model
    print("Loading MACE-MP-0 model...")
    calc = mace_mp(model='medium', device='cpu', default_dtype='float64')
    
    # Create water box
    print("Creating water box with 32 molecules...")
    atoms = create_water_box(N_MOLECULES, BOX_SIZE)
    atoms.calc = calc
    
    # Initial energy
    e0 = atoms.get_potential_energy()
    print(f"Initial potential energy: {e0:.3f} eV")
    
    # Set up MD
    print("Setting up Langevin dynamics...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    
    timestep_ASE = TIMESTEP * units.fs
    friction_ASE = FRICTION / units.fs
    
    dyn = Langevin(atoms, timestep_ASE, temperature_K=TEMPERATURE, 
                   friction=friction_ASE)
    
    # Run MD
    save_interval = 25
    print(f"Running MD for {N_STEPS} steps (saving every {save_interval} steps)...")
    trajectory = []
    energies = []
    
    for step in range(N_STEPS):
        dyn.run(1)
        if step % save_interval == 0 or step == N_STEPS - 1:
            e = atoms.get_potential_energy()
            energies.append(e)
            trajectory.append(atoms.copy())
            elapsed = time.time() - t_start
            print(f"  Step {step}/{N_STEPS}, Energy: {e:.3f} eV, Time: {elapsed:.1f}s")
    
    e_final = atoms.get_potential_energy()
    print(f"Final potential energy: {e_final:.3f} eV")
    
    # Compute RDF
    print("Computing O-O RDF...")
    r, g_r = compute_rdf(trajectory, r_max=6.0, n_bins=200, element='O')
    
    # Save results
    results = {
        'r': r.tolist(),
        'g_r': g_r.tolist(),
        'energies': [float(e) for e in energies],
        'initial_energy': float(e0),
        'final_energy': float(e_final),
        'parameters': {
            'n_molecules': N_MOLECULES,
            'box_size': BOX_SIZE,
            'temperature': TEMPERATURE,
            'timestep': TIMESTEP,
            'n_steps': N_STEPS,
            'friction': FRICTION,
        }
    }
    
    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/water_rdf_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to outputs/water_rdf_results.json")
    print(f"Total time: {time.time() - t_start:.1f}s")

if __name__ == '__main__':
    main()
