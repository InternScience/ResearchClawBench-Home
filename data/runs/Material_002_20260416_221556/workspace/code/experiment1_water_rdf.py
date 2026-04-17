"""
Experiment 1: Liquid Water Radial Distribution Function (RDF)
Using MACE-MP-0b3-medium model to simulate 32 water molecules
and compute O-O, O-H, and H-H RDFs.
"""
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Setup paths
WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_002_20260416_221556"
MODEL_PATH = os.path.join(WORKSPACE, "models/mace-mp-0b3-medium.model")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report/images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
from mace.calculators import MACECalculator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Setting up water box...")

# Parameters from data file
n_molecules = 32
box_size = 12.0  # Angstrom, cubic
temperature = 330  # K
timestep = 0.5  # fs
n_steps = 2000
friction = 0.01  # fs^-1

# Water molecule coordinates (centered)
O_pos = np.array([0.000000, 0.000000, 0.119262])
H1_pos = np.array([0.000000, 0.763239, -0.477047])
H2_pos = np.array([0.000000, -0.763239, -0.477047])

# Place 32 water molecules in a cubic box
# Use a grid arrangement: 4x4x2 = 32 molecules
positions = []
symbols = []
nx, ny, nz = 4, 4, 2
spacing = box_size / max(nx, ny, nz)

mol_count = 0
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            if mol_count >= n_molecules:
                break
            # Center of molecule
            center = np.array([
                (ix + 0.5) * box_size / nx,
                (iy + 0.5) * box_size / ny,
                (iz + 0.5) * box_size / nz
            ])
            # Add random rotation for more realistic starting config
            np.random.seed(42 + mol_count)
            # Simple random rotation using Euler angles
            alpha, beta, gamma = np.random.uniform(0, 2*np.pi, 3)
            # Rotation matrix
            Rz1 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                           [np.sin(alpha), np.cos(alpha), 0],
                           [0, 0, 1]])
            Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                          [0, 1, 0],
                          [-np.sin(beta), 0, np.cos(beta)]])
            Rz2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                           [np.sin(gamma), np.cos(gamma), 0],
                           [0, 0, 1]])
            R = Rz1 @ Ry @ Rz2
            
            o_pos = center + R @ O_pos
            h1_pos = center + R @ H1_pos
            h2_pos = center + R @ H2_pos
            
            positions.extend([o_pos, h1_pos, h2_pos])
            symbols.extend(['O', 'H', 'H'])
            mol_count += 1

positions = np.array(positions)
# Wrap positions into box
positions = positions % box_size

water_box = Atoms(
    symbols=symbols,
    positions=positions,
    cell=[box_size, box_size, box_size],
    pbc=True
)

print(f"Created water box with {len(water_box)} atoms ({n_molecules} molecules)")
print(f"Box size: {box_size} Å")

# Setup MACE calculator
print("Loading MACE-MP-0b3 model...")
calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
water_box.calc = calc

# Get initial energy
e0 = water_box.get_potential_energy()
print(f"Initial energy: {e0:.4f} eV")

# Setup Langevin dynamics
print(f"Setting up Langevin MD at {temperature} K...")
dyn = Langevin(
    water_box,
    timestep=timestep * units.fs,
    temperature_K=temperature,
    friction=friction / units.fs,
    fixcm=True
)

# Run MD and collect trajectory
print(f"Running {n_steps} MD steps...")
energies = []
temperatures_list = []
trajectory_positions = []

# Collect snapshots every 10 steps for RDF
collect_interval = 10
equilibration_steps = 500  # Skip first 500 steps for equilibration

for step in range(n_steps):
    dyn.run(1)
    
    if step % 100 == 0:
        e = water_box.get_potential_energy()
        T = water_box.get_kinetic_energy() / (1.5 * units.kB * len(water_box))
        energies.append(float(e))
        temperatures_list.append(float(T))
        print(f"  Step {step}: E = {e:.4f} eV, T = {T:.1f} K")
    
    if step >= equilibration_steps and step % collect_interval == 0:
        trajectory_positions.append(water_box.get_positions().copy())

print(f"MD complete. Collected {len(trajectory_positions)} snapshots for RDF analysis.")

# Compute RDF
print("Computing radial distribution functions...")

def compute_rdf(positions_list, cell, symbols, pair_types, r_max=6.0, n_bins=200):
    """Compute RDF for specified atom pair types."""
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    # Get indices for each element
    symbols_arr = np.array(symbols)
    
    histograms = {}
    for pair in pair_types:
        histograms[pair] = np.zeros(n_bins)
    
    cell_arr = np.array(cell)
    
    for positions in positions_list:
        for pair in pair_types:
            elem1, elem2 = pair.split('-')
            idx1 = np.where(symbols_arr == elem1)[0]
            idx2 = np.where(symbols_arr == elem2)[0]
            
            for i in idx1:
                for j in idx2:
                    if i == j:
                        continue
                    if elem1 == elem2 and j <= i:
                        continue
                    
                    dr_vec = positions[j] - positions[i]
                    # Minimum image convention
                    dr_vec = dr_vec - cell_arr * np.round(dr_vec / cell_arr)
                    dist = np.linalg.norm(dr_vec)
                    
                    if dist < r_max:
                        bin_idx = int(dist / (r_max / n_bins))
                        if bin_idx < n_bins:
                            histograms[pair][bin_idx] += 1
    
    # Normalize
    n_frames = len(positions_list)
    volume = np.prod(cell_arr)
    
    rdfs = {}
    for pair in pair_types:
        elem1, elem2 = pair.split('-')
        n1 = np.sum(symbols_arr == elem1)
        n2 = np.sum(symbols_arr == elem2)
        
        if elem1 == elem2:
            n_pairs = n1 * (n1 - 1) / 2
        else:
            n_pairs = n1 * n2
        
        rho = n_pairs / volume
        
        shell_volumes = 4.0/3.0 * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
        
        g_r = histograms[pair] / (n_frames * shell_volumes * rho)
        rdfs[pair] = g_r
    
    return r_centers, rdfs

cell_diag = [box_size, box_size, box_size]
pair_types = ['O-O', 'O-H', 'H-H']

r_centers, rdfs = compute_rdf(
    trajectory_positions, cell_diag, symbols, pair_types, 
    r_max=6.0, n_bins=200
)

# Save RDF data
rdf_data = {
    'r': r_centers.tolist(),
    'g_OO': rdfs['O-O'].tolist(),
    'g_OH': rdfs['O-H'].tolist(),
    'g_HH': rdfs['H-H'].tolist(),
    'n_frames': len(trajectory_positions),
    'temperature_K': temperature,
    'n_molecules': n_molecules,
    'box_size_A': box_size,
    'n_steps': n_steps,
    'equilibration_steps': equilibration_steps
}

with open(os.path.join(OUTPUT_DIR, 'water_rdf_data.json'), 'w') as f:
    json.dump(rdf_data, f, indent=2)

# Save MD trajectory summary
md_summary = {
    'energies_eV': energies,
    'temperatures_K': temperatures_list,
    'n_steps': n_steps,
    'timestep_fs': timestep,
    'friction_fs_inv': friction,
    'target_temperature_K': temperature,
    'n_molecules': n_molecules,
    'box_size_A': box_size,
    'n_snapshots_for_rdf': len(trajectory_positions)
}

with open(os.path.join(OUTPUT_DIR, 'water_md_trajectory.json'), 'w') as f:
    json.dump(md_summary, f, indent=2)

# Plot RDF
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# O-O RDF
axes[0].plot(r_centers, rdfs['O-O'], 'b-', linewidth=2, label='MACE-MP-0')
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
# Experimental O-O first peak at ~2.8 Å
axes[0].axvline(x=2.8, color='r', linestyle=':', alpha=0.5, label='Exp. peak (~2.8 Å)')
axes[0].set_xlabel('r (Å)', fontsize=12)
axes[0].set_ylabel('g(r)', fontsize=12)
axes[0].set_title('O-O RDF', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].set_xlim(0, 6)

# O-H RDF
axes[1].plot(r_centers, rdfs['O-H'], 'g-', linewidth=2, label='MACE-MP-0')
axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=1.0, color='r', linestyle=':', alpha=0.5, label='Intramol. O-H (~1.0 Å)')
axes[1].axvline(x=1.8, color='orange', linestyle=':', alpha=0.5, label='Intermol. O-H (~1.8 Å)')
axes[1].set_xlabel('r (Å)', fontsize=12)
axes[1].set_ylabel('g(r)', fontsize=12)
axes[1].set_title('O-H RDF', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 6)

# H-H RDF
axes[2].plot(r_centers, rdfs['H-H'], 'r-', linewidth=2, label='MACE-MP-0')
axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(x=1.5, color='orange', linestyle=':', alpha=0.5, label='Intramol. H-H (~1.5 Å)')
axes[2].set_xlabel('r (Å)', fontsize=12)
axes[2].set_ylabel('g(r)', fontsize=12)
axes[2].set_title('H-H RDF', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].set_xlim(0, 6)

plt.suptitle('Radial Distribution Functions of Liquid Water (MACE-MP-0, 330 K)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'water_rdf.png'), dpi=150, bbox_inches='tight')
print(f"Saved water RDF plot to {os.path.join(IMAGE_DIR, 'water_rdf.png')}")

# Also plot MD convergence
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
steps_recorded = [i*100 for i in range(len(energies))]
ax1.plot(steps_recorded, energies, 'b-')
ax1.set_xlabel('MD Step')
ax1.set_ylabel('Potential Energy (eV)')
ax1.set_title('MD Energy Convergence')
ax1.axvline(x=equilibration_steps, color='r', linestyle='--', label=f'Equilibration ({equilibration_steps} steps)')
ax1.legend()

ax2.plot(steps_recorded, temperatures_list, 'r-')
ax2.axhline(y=temperature, color='k', linestyle='--', label=f'Target T = {temperature} K')
ax2.set_xlabel('MD Step')
ax2.set_ylabel('Temperature (K)')
ax2.set_title('MD Temperature')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'water_md_convergence.png'), dpi=150, bbox_inches='tight')
print(f"Saved MD convergence plot to {os.path.join(IMAGE_DIR, 'water_md_convergence.png')}")

# Report key RDF features
oo_peak_idx = np.argmax(rdfs['O-O'][5:]) + 5  # Skip first few bins
oo_peak_r = r_centers[oo_peak_idx]
oo_peak_g = rdfs['O-O'][oo_peak_idx]
print(f"\nO-O RDF first peak: r = {oo_peak_r:.2f} Å, g(r) = {oo_peak_g:.2f}")
print(f"(Experimental: ~2.8 Å)")

print("\nExperiment 1 complete!")
