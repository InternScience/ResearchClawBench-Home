"""
MACE-MP-0 Foundation Model Reproduction
Experiment 2: Adsorption Energy Scaling Relations on Transition Metal Surfaces
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from ase import Atoms, Atom
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator

# ---- Parameters from dataset ----
metals = {
    'Ni': 3.52,
    'Cu': 3.61,
    'Rh': 3.80,
    'Pd': 3.89,
    'Ir': 3.84,
    'Pt': 3.92,
}

slab_size = (2, 2, 3)
vacuum = 10.0
ads_height = 1.5
force_tol = 0.05

calc = MACECalculator(model_paths='mace_mp_0.model', device='cpu')

results = {}

for metal, lattice_const in metals.items():
    print(f"\n{'='*50}")
    print(f"Processing {metal} (a = {lattice_const} Å)")
    print(f"{'='*50}")
    
    # Build fcc(111) slab
    slab = fcc111(metal, size=slab_size, a=lattice_const, vacuum=vacuum)
    slab.calc = calc
    
    n_atoms_per_layer = slab_size[0] * slab_size[1]
    n_fixed_layers = 2
    
    tags = np.ones(len(slab), dtype=int)
    for i in range(n_fixed_layers * n_atoms_per_layer):
        tags[i] = 2
    slab.set_tags(tags)
    
    constraint = FixAtoms(mask=[tag >= 2 for tag in slab.get_tags()])
    slab.set_constraint(constraint)
    
    # Relax clean slab
    print(f"  Relaxing clean slab ({len(slab)} atoms)...")
    try:
        opt = BFGS(slab, logfile=None)
        opt.run(fmax=force_tol, steps=100)
        E_slab = slab.get_potential_energy()
        print(f"  Clean slab energy: {E_slab:.4f} eV")
    except Exception as e:
        print(f"  Error relaxing slab: {e}")
        continue
    
    # ---- O adsorption ----
    slab_O = slab.copy()
    slab_O.calc = calc
    slab_O.set_constraint(constraint)
    
    add_adsorbate(slab_O, 'O', ads_height, position='fcc')
    slab_O.calc = calc
    
    print(f"  Relaxing O on {metal} ({len(slab_O)} atoms)...")
    try:
        opt_O = BFGS(slab_O, logfile=None)
        opt_O.run(fmax=force_tol, steps=100)
        E_slab_O = slab_O.get_potential_energy()
        print(f"  Slab+O energy: {E_slab_O:.4f} eV")
    except Exception as e:
        print(f"  Error: {e}")
        E_slab_O = None
    
    # ---- OH adsorption ----
    slab_OH = slab.copy()
    slab_OH.calc = calc
    slab_OH.set_constraint(constraint)
    
    add_adsorbate(slab_OH, 'O', ads_height, position='fcc')
    
    # Add H above the adsorbed O
    o_pos = slab_OH.get_positions()[-1].copy()
    h_pos = o_pos + np.array([0, 0, 1.0])
    slab_OH.append(Atom('H', position=h_pos))
    slab_OH.calc = calc
    
    print(f"  Relaxing OH on {metal} ({len(slab_OH)} atoms)...")
    try:
        opt_OH = BFGS(slab_OH, logfile=None)
        opt_OH.run(fmax=force_tol, steps=100)
        E_slab_OH = slab_OH.get_potential_energy()
        print(f"  Slab+OH energy: {E_slab_OH:.4f} eV")
    except Exception as e:
        print(f"  Error: {e}")
        E_slab_OH = None
    
    # ---- Gas phase references ----
    O_gas = Atoms('O', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    O_gas.calc = calc
    E_O = O_gas.get_potential_energy()
    print(f"  O gas energy: {E_O:.4f} eV")
    
    OH_gas = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=True)
    OH_gas.calc = calc
    E_OH = OH_gas.get_potential_energy()
    print(f"  OH gas energy: {E_OH:.4f} eV")
    
    # ---- Compute adsorption energies ----
    if E_slab_O is not None:
        E_ads_O = E_slab_O - E_slab - E_O
        print(f"  E_ads(O) = {E_ads_O:.4f} eV")
    else:
        E_ads_O = None
    
    if E_slab_OH is not None:
        E_ads_OH = E_slab_OH - E_slab - E_OH
        print(f"  E_ads(OH) = {E_ads_OH:.4f} eV")
    else:
        E_ads_OH = None
    
    results[metal] = {
        'lattice_const': lattice_const,
        'E_slab': E_slab,
        'E_slab_O': E_slab_O,
        'E_slab_OH': E_slab_OH,
        'E_O': E_O,
        'E_OH': E_OH,
        'E_ads_O': E_ads_O,
        'E_ads_OH': E_ads_OH,
    }

with open('outputs/adsorption_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nAdsorption results saved to outputs/adsorption_results.json")
print("\nSummary:")
print(f"{'Metal':>6} {'E_ads(O) (eV)':>14} {'E_ads(OH) (eV)':>16}")
for metal, r in results.items():
    e_o = f"{r['E_ads_O']:.4f}" if r['E_ads_O'] is not None else 'N/A'
    e_oh = f"{r['E_ads_OH']:.4f}" if r['E_ads_OH'] is not None else 'N/A'
    print(f"{metal:>6} {e_o:>14} {e_oh:>16}")
