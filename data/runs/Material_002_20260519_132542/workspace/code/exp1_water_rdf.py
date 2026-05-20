"""
Experiment 1: Liquid water MD and O-O RDF using MACE-MP-0.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import molecule
from ase.io import write, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.geometry.analysis import Analysis
from mace.calculators import mace_mp
import torch

# Parameters from dataset
N_WATER = 32
BOX = 12.0  # Å cubic
TEMPERATURE_K = 330
TIMESTEP_FS = 0.5
N_STEPS = 2000
FRICTION = 0.01  # fs^-1

# Load MACE-MP-0 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = mace_mp(model="outputs/mace-mp-0b3-medium.model", device=device, default_dtype='float32')

# Build a single water molecule and center it
h2o = molecule('H2O')
h2o.center()
# Ensure coordinates match dataset roughly (optional)
# print(h2o.get_positions())

# Place N_WATER copies randomly in the box with minimum O-O distance
np.random.seed(42)
molecules = []
min_oo = 2.2  # Å
max_trials = 10000

for i in range(N_WATER):
    for trial in range(max_trials):
        # Random translation
        pos = np.random.rand(3) * BOX
        # Random rotation
        angle = np.random.rand(3) * 2 * np.pi
        mol = h2o.copy()
        mol.rotate(angle[0], 'x', center=(0, 0, 0))
        mol.rotate(angle[1], 'y', center=(0, 0, 0))
        mol.rotate(angle[2], 'z', center=(0, 0, 0))
        mol.translate(pos)
        # Check periodic boundaries: wrap O positions
        oo_positions = np.array([m.get_positions()[0] for m in molecules] + [mol.get_positions()[0]])
        # Minimum image distance
        if len(molecules) == 0:
            molecules.append(mol)
            break
        diffs = oo_positions[:-1] - oo_positions[-1]
        diffs -= BOX * np.round(diffs / BOX)
        dists = np.linalg.norm(diffs, axis=1)
        if np.all(dists > min_oo):
            molecules.append(mol)
            break
    else:
        raise RuntimeError(f"Could not place molecule {i} after {max_trials} trials")

water = sum(molecules[1:], molecules[0])
water.set_cell([BOX, BOX, BOX])
water.set_pbc(True)
water.calc = calc

# Initialize velocities
MaxwellBoltzmannDistribution(water, temperature_K=TEMPERATURE_K)

# Langevin dynamics
dyn = Langevin(water, timestep=TIMESTEP_FS * units.fs, temperature_K=TEMPERATURE_K,
               friction=FRICTION / units.fs, fixcm=False)

# Run MD and save trajectory
traj_file = 'outputs/water_md.traj'
from ase.io import Trajectory
traj = Trajectory(traj_file, 'w', water)
dyn.attach(traj.write, interval=10)

print(f"Running MD for {N_STEPS} steps...")
dyn.run(N_STEPS)
traj.close()

print("MD finished. Computing O-O RDF...")

# Compute RDF from trajectory
# Read trajectory
images = read(traj_file, index=':')
# Use last 50% of trajectory for averaging (after equilibration)
start_idx = len(images) // 2
rdf_data = []

# Manually compute RDF with bins
r_max = BOX / 2
n_bins = 100
bin_edges = np.linspace(0, r_max, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dr = bin_edges[1] - bin_edges[0]
volume = BOX ** 3
rho = N_WATER / volume  # number density of O atoms

g_r = np.zeros(n_bins)
count_frames = 0

for atoms in images[start_idx:]:
    pos = atoms.get_positions()
    # Oxygen indices: every 3rd atom starting from 0 (O, H, H)
    o_pos = pos[0::3]
    n_o = len(o_pos)
    # Pairwise distances with minimum image convention
    for i in range(n_o):
        for j in range(i + 1, n_o):
            diff = o_pos[i] - o_pos[j]
            diff -= BOX * np.round(diff / BOX)
            r = np.linalg.norm(diff)
            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < n_bins:
                    g_r[bin_idx] += 2  # count both i,j and j,i
    count_frames += 1

# Normalize RDF
for k in range(n_bins):
    shell_volume = 4.0 / 3.0 * np.pi * ((bin_edges[k+1])**3 - (bin_edges[k])**3)
    g_r[k] /= count_frames * N_WATER * shell_volume * rho

# Save RDF data
np.savetxt('outputs/water_oo_rdf.csv', np.column_stack([bin_centers, g_r]),
           header='r(Å),g(r)', delimiter=',', comments='')

# Plot
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, g_r, label='MACE-MP-0')
plt.xlabel('r (Å)')
plt.ylabel('g$_{OO}$(r)')
plt.title('O–O Radial Distribution Function (330 K)')
plt.legend()
plt.xlim(0, r_max)
plt.ylim(0, max(g_r)*1.1)
plt.tight_layout()
plt.savefig('report/images/water_oo_rdf.png', dpi=300)
plt.close()

print("Experiment 1 complete. Figure saved to report/images/water_oo_rdf.png")
