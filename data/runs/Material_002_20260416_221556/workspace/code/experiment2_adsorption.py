"""
Experiment 2: Adsorption Energy Scaling Relations
Compute O and OH adsorption energies on fcc(111) transition metal surfaces
using MACE-MP-0b3-medium model.
"""
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_002_20260416_221556"
MODEL_PATH = os.path.join(WORKSPACE, "models/mace-mp-0b3-medium.model")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report/images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading MACE-MP-0b3 model...")
calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")

# Metal parameters from data file
metals = {
    'Ni': 3.52,
    'Cu': 3.61,
    'Rh': 3.80,
    'Pd': 3.89,
    'Ir': 3.84,
    'Pt': 3.92
}

# Slab parameters
miller = (1, 1, 1)
size = (2, 2, 3)
vacuum = 10.0
fmax = 0.05  # eV/Å

# Gas phase references
print("\n=== Computing gas phase references ===")

# O atom in a box
o_atom = Atoms('O', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
o_atom.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
E_O_gas = o_atom.get_potential_energy()
print(f"E(O atom) = {E_O_gas:.4f} eV")

# OH molecule in a box
oh_mol = Atoms('OH', positions=[[5, 5, 5], [5, 5, 6.0]], cell=[10, 10, 10], pbc=True)
oh_mol.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
# Relax OH
opt = BFGS(oh_mol, logfile=None)
opt.run(fmax=0.01)
E_OH_gas = oh_mol.get_potential_energy()
print(f"E(OH molecule) = {E_OH_gas:.4f} eV")

results = {}

for metal, a in metals.items():
    print(f"\n=== Processing {metal} (a = {a} Å) ===")
    
    # Build clean slab
    slab = fcc111(metal, size=size, a=a, vacuum=vacuum, periodic=True)
    
    # Fix bottom 2 layers
    # In ASE fcc111, tags are 0 for bottom layer, increasing upward
    # We need to fix atoms with tag >= 2 (bottom 2 layers)
    # Actually, let's check the tags
    tags = slab.get_tags()
    print(f"  Slab tags: {np.unique(tags)}")
    # Fix bottom 2 layers (tags 0 and 1, or tags >= 2 depending on convention)
    # In ASE, for fcc111 with 3 layers, tags are 1, 2, 3 from bottom to top
    # Fix bottom 2 layers: tags 1 and 2
    # But data file says "Fixed layers: bottom 2 layers (tags ≥ 2)"
    # This means fix atoms where tag >= 2
    constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag >= 2])
    slab.set_constraint(constraint)
    
    # Clean slab energy
    slab.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    
    # Relax clean slab
    print(f"  Relaxing clean slab...")
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=fmax, steps=200)
    E_slab = slab.get_potential_energy()
    print(f"  E(clean slab) = {E_slab:.4f} eV")
    
    # O adsorption on fcc hollow site
    print(f"  Computing O adsorption...")
    slab_O = slab.copy()
    add_adsorbate(slab_O, 'O', height=1.5, position='fcc')
    constraint_O = FixAtoms(indices=[atom.index for atom in slab_O if atom.tag >= 2])
    slab_O.set_constraint(constraint_O)
    slab_O.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    
    opt = BFGS(slab_O, logfile=None)
    opt.run(fmax=fmax, steps=200)
    E_slab_O = slab_O.get_potential_energy()
    E_ads_O = E_slab_O - E_slab - E_O_gas
    print(f"  E(slab+O) = {E_slab_O:.4f} eV, E_ads(O) = {E_ads_O:.4f} eV")
    
    # OH adsorption on fcc hollow site
    print(f"  Computing OH adsorption...")
    slab_OH = slab.copy()
    # Add OH molecule at fcc site
    add_adsorbate(slab_OH, 'O', height=1.5, position='fcc')
    # Add H on top of O
    o_idx = len(slab_OH) - 1
    o_pos = slab_OH.positions[o_idx]
    slab_OH.append(Atoms('H', positions=[o_pos + np.array([0, 0, 1.0])]))
    
    constraint_OH = FixAtoms(indices=[atom.index for atom in slab_OH if atom.tag >= 2])
    slab_OH.set_constraint(constraint_OH)
    slab_OH.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    
    opt = BFGS(slab_OH, logfile=None)
    opt.run(fmax=fmax, steps=200)
    E_slab_OH = slab_OH.get_potential_energy()
    E_ads_OH = E_slab_OH - E_slab - E_OH_gas
    print(f"  E(slab+OH) = {E_slab_OH:.4f} eV, E_ads(OH) = {E_ads_OH:.4f} eV")
    
    results[metal] = {
        'lattice_constant': a,
        'E_slab': float(E_slab),
        'E_slab_O': float(E_slab_O),
        'E_slab_OH': float(E_slab_OH),
        'E_ads_O': float(E_ads_O),
        'E_ads_OH': float(E_ads_OH)
    }

# Save results
results['gas_phase'] = {
    'E_O': float(E_O_gas),
    'E_OH': float(E_OH_gas)
}

with open(os.path.join(OUTPUT_DIR, 'adsorption_energies.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Results Summary ===")
print(f"{'Metal':>5s} {'E_ads(O)':>10s} {'E_ads(OH)':>10s}")
print("-" * 30)
for metal in metals:
    print(f"{metal:>5s} {results[metal]['E_ads_O']:>10.3f} {results[metal]['E_ads_OH']:>10.3f}")

# Plot scaling relation
fig, ax = plt.subplots(figsize=(8, 6))
E_O_list = [results[m]['E_ads_O'] for m in metals]
E_OH_list = [results[m]['E_ads_OH'] for m in metals]

for i, metal in enumerate(metals):
    ax.scatter(results[metal]['E_ads_O'], results[metal]['E_ads_OH'], 
              s=100, zorder=5, label=metal)
    ax.annotate(metal, (results[metal]['E_ads_O'], results[metal]['E_ads_OH']),
               textcoords="offset points", xytext=(10, 5), fontsize=12)

# Linear fit
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(E_O_list, E_OH_list, 1)
x_fit = np.linspace(min(E_O_list) - 0.5, max(E_O_list) + 0.5, 100)
y_fit = np.polyval(coeffs, x_fit)
ax.plot(x_fit, y_fit, 'k--', alpha=0.7, 
        label=f'Linear fit: slope={coeffs[0]:.2f}, R²={np.corrcoef(E_O_list, E_OH_list)[0,1]**2:.3f}')

ax.set_xlabel('E$_{ads}$(O) (eV)', fontsize=14)
ax.set_ylabel('E$_{ads}$(OH) (eV)', fontsize=14)
ax.set_title('Adsorption Energy Scaling: OH vs O on fcc(111) Surfaces\n(MACE-MP-0)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'adsorption_scaling.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved scaling plot to {os.path.join(IMAGE_DIR, 'adsorption_scaling.png')}")

# Compute correlation
R2 = np.corrcoef(E_O_list, E_OH_list)[0,1]**2
print(f"\nScaling relation: E_ads(OH) = {coeffs[0]:.3f} * E_ads(O) + {coeffs[1]:.3f}")
print(f"R² = {R2:.4f}")
print(f"(Expected slope ~0.5-0.7 from literature)")

print("\nExperiment 2 complete!")
