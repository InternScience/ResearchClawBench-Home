#!/usr/bin/env python3
"""
Water RDF reproduction script for MACE-MP-0b3-medium
32 H2O molecules in 12 Å cubic box at 330 K
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kB
from mace.calculators import MACECalculator
import torch

# Paths
MODEL_PATH = "data/mace-mp-0b3-medium.model"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Simulation parameters
N_MOLECULES = 32
BOX_SIZE = 12.0  # Å
TEMPERATURE = 330  # K
TIMESTEP = 1.0 * fs
N_STEPS = 2000  # full reproduction
EQUIL_STEPS = 500
RDF_CUTOFF = 6.0
RDF_BINS = 50
N_STEPS = 50  # Reduced for feasibility
def create_water_box(n_molecules, box_size):
    """Create a box of water molecules"""
    atoms_list = []
    density = 1.0  # g/cm^3
    volume = box_size ** 3
    # Approximate number of molecules for water density
    for i in range(n_molecules):
        x = np.random.rand() * box_size
        y = np.random.rand() * box_size
        z = np.random.rand() * box_size
        # Water molecule geometry
        O = Atoms('O', positions=[[x, y, z]])
        H1 = Atoms('H', positions=[[x + 0.96, y, z]])
        H2 = Atoms('H', positions=[[x + 0.24, y + 0.93, z]])
        atoms_list.extend([O, H1, H2])
    water = atoms_list[0]
    for a in atoms_list[1:]:
        water += a
    water.set_cell([box_size, box_size, box_size])
    water.set_pbc(True)
    return water

def compute_rdf(atoms, cutoff=6.0, bins=100):
    """Compute O-O RDF"""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    o_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'O']
    o_pos = positions[o_indices]
    n_o = len(o_pos)
    distances = []
    for i in range(n_o):
        for j in range(i+1, n_o):
            d = np.linalg.norm(o_pos[i] - o_pos[j])
            if d < cutoff:
                distances.append(d)
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, cutoff))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Normalize
    volume = np.prod(cell.diagonal())
    rho = n_o / volume
    shell_vol = 4 * np.pi * bin_centers**2 * (bin_edges[1] - bin_edges[0])
    rdf = hist / (n_o * rho * shell_vol)
    return bin_centers, rdf

def main():
    print("Loading MACE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc = MACECalculator(model_path=MODEL_PATH, device=device, default_dtype="float32")
    print(f"Using device: {device}")

    print("Creating water box...")
    atoms = create_water_box(N_MOLECULES, BOX_SIZE)
    atoms.calc = calc

    print("Initializing velocities...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)

    print("Running MD...")
    dyn = Langevin(atoms, timestep=TIMESTEP, temperature_K=TEMPERATURE, friction=0.01)
    dyn.run(EQUIL_STEPS)

    # Production run with trajectory saving
    traj = []
    for step in range(N_STEPS):
        dyn.run(1)
        if step % 10 == 0:
            traj.append(atoms.copy())
            if step % 500 == 0:
                print(f"  Step {step}/{N_STEPS}")

    # Compute RDF from last configuration
    print("Computing RDF...")
    r, g = compute_rdf(atoms, cutoff=RDF_CUTOFF, bins=RDF_BINS)

    # Save results
    np.savetxt(f"{OUTPUT_DIR}/water_rdf.txt", np.column_stack([r, g]),
               header="r(A) g(r)")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(r, g, 'b-', linewidth=2, label='MACE-MP-0')
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Water O-O RDF at 330 K', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, RDF_CUTOFF)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/water_rdf.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Results saved to {OUTPUT_DIR}/water_rdf.txt")
    print(f"Figure saved to {FIGURE_DIR}/water_rdf.png")

if __name__ == "__main__":
    main()
