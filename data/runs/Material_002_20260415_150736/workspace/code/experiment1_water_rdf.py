"""
Experiment 1: Liquid Water Structure - Radial Distribution Function

This script simulates liquid water using MACE-MP-0 and computes the 
radial distribution function (RDF) to validate the model's accuracy
for liquid systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import molecule
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write
from ase import units
import torch
from mace.calculators import mace_mp
import json
import os

# Simulation parameters from dataset
N_MOLECULES = 32
BOX_SIZE = 12.0  # Angstrom
TEMPERATURE = 330  # K
TIME_STEP = 0.5  # fs
N_STEPS = 2000
FRICTION = 0.01  # fs^-1

# Water molecule coordinates (from ASE molecule('H2O') after centering)
# O: [0.000000, 0.000000, 0.119262]
# H: [0.000000, 0.763239, -0.477047]
# H: [0.000000, -0.763239, -0.477047]

def create_water_box():
    """Create a cubic box of water molecules."""
    # Single water molecule positions
    water_pos = np.array([
        [0.000000, 0.000000, 0.119262],  # O
        [0.000000, 0.763239, -0.477047],  # H
        [0.000000, -0.763239, -0.477047],  # H
    ])
    
    # Create positions for N molecules in a cubic arrangement
    positions = []
    symbols = []
    
    # Simple cubic arrangement
    n_side = int(np.ceil(N_MOLECULES ** (1/3)))
    spacing = BOX_SIZE / n_side
    
    mol_count = 0
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if mol_count >= N_MOLECULES:
                    break
                # Random rotation for each molecule would be better
                # but for simplicity, we use fixed orientation with small random offset
                offset = np.array([i, j, k]) * spacing + spacing/2
                # Add small random displacement
                offset += np.random.uniform(-0.5, 0.5, 3)
                
                for atom_pos in water_pos:
                    positions.append(atom_pos + offset)
                symbols.extend(['O', 'H', 'H'])
                mol_count += 1
            if mol_count >= N_MOLECULES:
                break
        if mol_count >= N_MOLECULES:
            break
    
    atoms = Atoms(symbols=symbols, positions=positions, cell=[BOX_SIZE]*3, pbc=True)
    return atoms

def compute_rdf(atoms, r_max=6.0, n_bins=100, elements=None):
    """
    Compute radial distribution function.
    
    Args:
        atoms: ASE Atoms object
        r_max: Maximum radius for RDF
        n_bins: Number of bins
        elements: Tuple of (element1, element2) for partial RDF, or None for total
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    # Get indices for element pair
    if elements is None:
        indices1 = list(range(len(atoms)))
        indices2 = list(range(len(atoms)))
    else:
        indices1 = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == elements[0]]
        indices2 = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == elements[1]]
    
    # Compute distances using minimum image convention
    dr = r_max / n_bins
    rdf = np.zeros(n_bins)
    
    for i in indices1:
        for j in indices2:
            if i == j and (elements is None or elements[0] == elements[1]):
                continue
            
            # Vector from i to j
            r_ij = positions[j] - positions[i]
            # Minimum image convention
            r_ij -= np.round(r_ij @ np.linalg.inv(cell)) @ cell
            r = np.linalg.norm(r_ij)
            
            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < n_bins:
                    rdf[bin_idx] += 1
    
    # Normalize RDF
    r_bins = np.linspace(0, r_max, n_bins, endpoint=False) + dr/2
    
    # Volume factor for normalization
    V = atoms.get_volume()
    N1 = len(indices1)
    N2 = len(indices2)
    
    # Normalize by shell volume and density
    for i in range(n_bins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        shell_volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
        
        if elements is None:
            # Total RDF normalization
            rho = (N1 - 1) / V  # Exclude self
        else:
            rho = N2 / V
        
        if elements is not None and elements[0] == elements[1]:
            # Correction for same-element pairs (i != j)
            rdf[i] /= (N1 * rho * shell_volume)
        else:
            rdf[i] /= (N1 * rho * shell_volume)
    
    return r_bins, rdf

def run_water_simulation():
    """Run MD simulation of water and compute RDF."""
    print("="*60)
    print("Experiment 1: Liquid Water Structure - RDF Simulation")
    print("="*60)
    
    # Initialize MACE calculator
    print("\n[1/5] Loading MACE-MP-0 foundation model...")
    calc = mace_mp(model="/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/models/mace-mp-0b3-medium.model", device='cpu')
    
    # Create water box
    print(f"[2/5] Creating water box with {N_MOLECULES} molecules...")
    atoms = create_water_box()
    atoms.calc = calc
    
    # Minimize energy
    print("[3/5] Minimizing initial structure...")
    from ase.optimize import BFGS
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.1, steps=50)
    
    # Set up Langevin dynamics
    print(f"[4/5] Setting up Langevin dynamics at {TEMPERATURE}K...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    dyn = Langevin(atoms, TIME_STEP * units.fs, temperature_K=TEMPERATURE, friction=FRICTION)
    
    # Storage for RDF calculation
    positions_trajectory = []
    energies = []
    
    def record_frame():
        positions_trajectory.append(atoms.get_positions().copy())
        energies.append(atoms.get_potential_energy())
    
    # Run simulation
    print(f"[5/5] Running {N_STEPS} steps of MD...")
    dyn.attach(record_frame, interval=10)
    record_frame()  # Record initial frame
    
    dyn.run(N_STEPS)
    
    print(f"\nSimulation complete!")
    print(f"  Total frames recorded: {len(positions_trajectory)}")
    print(f"  Average energy: {np.mean(energies):.3f} eV")
    print(f"  Energy std: {np.std(energies):.3f} eV")
    
    # Compute RDF from trajectory (use last 100 frames for equilibrated structure)
    print("\n[6/6] Computing radial distribution functions...")
    
    # Use equilibrated part of trajectory
    equil_frames = positions_trajectory[-100:]
    
    # O-O RDF
    r_bins_oo = None
    rdf_oo_frames = []
    
    # O-H RDF
    r_bins_oh = None
    rdf_oh_frames = []
    
    # H-H RDF
    r_bins_hh = None
    rdf_hh_frames = []
    
    for positions in equil_frames:
        frame_atoms = Atoms(symbols=atoms.get_chemical_symbols(), 
                           positions=positions, 
                           cell=atoms.get_cell(), 
                           pbc=True)
        
        r_bins, rdf_oo = compute_rdf(frame_atoms, elements=('O', 'O'))
        r_bins_oo = r_bins
        rdf_oo_frames.append(rdf_oo)
        
        r_bins, rdf_oh = compute_rdf(frame_atoms, elements=('O', 'H'))
        r_bins_oh = r_bins
        rdf_oh_frames.append(rdf_oh)
        
        r_bins, rdf_hh = compute_rdf(frame_atoms, elements=('H', 'H'))
        r_bins_hh = r_bins
        rdf_hh_frames.append(rdf_hh)
    
    # Average RDFs
    rdf_oo_avg = np.mean(rdf_oo_frames, axis=0)
    rdf_oh_avg = np.mean(rdf_oh_frames, axis=0)
    rdf_hh_avg = np.mean(rdf_hh_frames, axis=0)
    
    # Save results
    results = {
        'r_bins_oo': r_bins_oo.tolist(),
        'rdf_oo': rdf_oo_avg.tolist(),
        'r_bins_oh': r_bins_oh.tolist(),
        'rdf_oh': rdf_oh_avg.tolist(),
        'r_bins_hh': r_bins_hh.tolist(),
        'rdf_hh': rdf_hh_avg.tolist(),
        'simulation_params': {
            'n_molecules': N_MOLECULES,
            'box_size': BOX_SIZE,
            'temperature': TEMPERATURE,
            'time_step': TIME_STEP,
            'n_steps': N_STEPS,
            'friction': FRICTION
        },
        'energy_stats': {
            'mean': float(np.mean(energies)),
            'std': float(np.std(energies)),
            'min': float(np.min(energies)),
            'max': float(np.max(energies))
        }
    }
    
    os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs', exist_ok=True)
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/water_rdf_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to outputs/water_rdf_results.json")
    
    return results

if __name__ == '__main__':
    results = run_water_simulation()
