#!/usr/bin/env python3
"""
Experiment 2: Adsorption Energy Scaling Relations on Transition Metal Surfaces
Uses MACE-MP-0 to compute O and OH adsorption energies on fcc(111) surfaces
of Ni, Cu, Rh, Pd, Ir, Pt.
"""

import os
import json
import time
import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from mace.calculators import mace_mp

# ── Parameters from the dataset ──────────────────────────────────
METALS = {
    'Ni': 3.52,
    'Cu': 3.61,
    'Rh': 3.80,
    'Pd': 3.89,
    'Ir': 3.84,
    'Pt': 3.92,
}

SLAB_SIZE = (2, 2, 3)  # 2x2 surface unit cell, 3 layers
VACUUM = 10.0  # Å
ADSORBATE_HEIGHT = 1.5  # Å above surface
SITE = 'fcc'  # hollow site
FIXED_LAYERS = 2  # bottom 2 layers fixed
FMAX = 0.05  # eV/Å force convergence

# Gas phase molecules
O_ATOM = Atoms('O', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=[True, True, True])
OH_MOLECULE = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=[True, True, True])


def relax_atoms(atoms, fmax=FMAX, steps=200):
    """Relax atomic positions with BFGS optimizer."""
    opt = BFGS(atoms)
    opt.run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy()


def compute_slab_energy(metal, lattice_constant, calc):
    """Build slab and compute its energy."""
    # Build fcc(111) slab
    from ase.build import fcc111
    slab = fcc111(metal, a=lattice_constant, size=SLAB_SIZE, vacuum=VACUUM)
    slab.calc = calc
    
    # Fix bottom layers
    mask = [atom.tag >= FIXED_LAYERS for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    
    e_slab = relax_atoms(slab)
    return e_slab, slab


def compute_adsorption_energy(metal, lattice_constant, adsorbate_atoms, calc):
    """Compute adsorption energy: E_ads = E(slab+ads) - E(slab) - E(gas)"""
    # Build clean slab
    from ase.build import fcc111
    slab = fcc111(metal, a=lattice_constant, size=SLAB_SIZE, vacuum=VACUUM)
    slab.calc = calc
    
    # Fix bottom layers
    mask = [atom.tag >= FIXED_LAYERS for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    
    # Relax clean slab
    e_slab = relax_atoms(slab)
    
    # Add adsorbate at fcc hollow site
    slab_with_ads = slab.copy()
    add_adsorbate(slab_with_ads, adsorbate_atoms, ADSORBATE_HEIGHT, position=SITE)
    slab_with_ads.calc = calc
    
    # Fix slab atoms, relax adsorbate
    n_slab = len(slab)
    mask_ads = [True] * len(slab_with_ads)
    for i in range(n_slab):
        mask_ads[i] = (slab_with_ads[i].tag < FIXED_LAYERS)
    slab_with_ads.set_constraint(FixAtoms(mask=[not m for m in mask_ads]))
    
    # Actually, let's constrain all slab atoms except maybe top layer
    # Better: fix bottom 2 layers, relax top layer + adsorbate
    mask_fix = [atom.tag < FIXED_LAYERS for atom in slab_with_ads]
    slab_with_ads.set_constraint(FixAtoms(mask=mask_fix))
    
    e_slab_ads = relax_atoms(slab_with_ads)
    
    # Gas phase energy
    gas_atoms = adsorbate_atoms.copy()
    gas_atoms.set_cell([10, 10, 10])
    gas_atoms.set_pbc([True, True, True])
    gas_atoms.calc = calc
    e_gas = gas_atoms.get_potential_energy()
    
    e_ads = e_slab_ads - e_slab - e_gas
    
    return e_ads, e_slab, e_slab_ads, e_gas


def main():
    print("=" * 60)
    print("Experiment 2: Adsorption Energy Scaling Relations")
    print("=" * 60)
    
    t_start = time.time()
    
    # Load MACE model
    print("Loading MACE-MP-0 model...")
    calc = mace_mp(model='medium', device='cpu', default_dtype='float64')
    print(f"Model loaded in {time.time()-t_start:.1f}s")
    
    results = {'metals': {}, 'parameters': {
        'slab_size': SLAB_SIZE,
        'vacuum': VACUUM,
        'adsorbate_height': ADSORBATE_HEIGHT,
        'site': SITE,
        'fixed_layers': FIXED_LAYERS,
        'fmax': FMAX,
    }}
    
    for metal, a in METALS.items():
        print(f"\n--- {metal} (a={a} Å) ---")
        t_metal = time.time()
        
        # O adsorption
        print("  Computing O adsorption...")
        e_ads_O, e_slab_O, e_slab_ads_O, e_gas_O = compute_adsorption_energy(
            metal, a, O_ATOM, calc)
        print(f"  O: E_slab={e_slab_O:.3f}, E_gas={e_gas_O:.3f}, "
              f"E_slab+ads={e_slab_ads_O:.3f}, E_ads={e_ads_O:.3f} eV")
        
        # OH adsorption
        print("  Computing OH adsorption...")
        e_ads_OH, e_slab_OH, e_slab_ads_OH, e_gas_OH = compute_adsorption_energy(
            metal, a, OH_MOLECULE, calc)
        print(f"  OH: E_slab={e_slab_OH:.3f}, E_gas={e_gas_OH:.3f}, "
              f"E_slab+ads={e_slab_ads_OH:.3f}, E_ads={e_ads_OH:.3f} eV")
        
        results['metals'][metal] = {
            'lattice_constant': a,
            'O': {
                'E_slab': float(e_slab_O),
                'E_gas': float(e_gas_O),
                'E_slab_ads': float(e_slab_ads_O),
                'E_ads': float(e_ads_O),
            },
            'OH': {
                'E_slab': float(e_slab_OH),
                'E_gas': float(e_gas_OH),
                'E_slab_ads': float(e_slab_ads_OH),
                'E_ads': float(e_ads_OH),
            }
        }
        
        print(f"  Metal done in {time.time()-t_metal:.1f}s")
    
    # Compute scaling relation: E_ads(OH) vs E_ads(O)
    e_ads_O_list = [results['metals'][m]['O']['E_ads'] for m in METALS]
    e_ads_OH_list = [results['metals'][m]['OH']['E_ads'] for m in METALS]
    
    # Linear fit
    from numpy.polynomial.polynomial import polyfit
    slope, intercept = polyfit(e_ads_O_list, e_ads_OH_list, 1)
    results['scaling_relation'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'E_ads_O': e_ads_O_list,
        'E_ads_OH': e_ads_OH_list,
        'metals': list(METALS.keys()),
    }
    
    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/adsorption_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to outputs/adsorption_results.json")
    print(f"Scaling relation: E_ads(OH) = {slope:.3f} * E_ads(O) + {intercept:.3f}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
