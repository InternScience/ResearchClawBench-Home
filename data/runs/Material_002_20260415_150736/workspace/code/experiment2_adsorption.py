"""
Experiment 2: Adsorption Energy Scaling Relations on Transition Metal Surfaces

This script computes adsorption energies of O and OH on transition metal 
fcc(111) surfaces to validate scaling relations predicted by MACE-MP-0.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator
import torch
from mace.calculators import mace_mp
import json
import os

# Metal lattice constants (from dataset)
METALS = {
    'Ni': 3.52,
    'Cu': 3.61,
    'Rh': 3.80,
    'Pd': 3.89,
    'Ir': 3.84,
    'Pt': 3.92
}

# Slab parameters
MILLER_INDICES = (1, 1, 1)
SLAB_SIZE = (2, 2, 3)  # 2x2 surface unit cell, 3 layers
VACUUM_GAP = 10.0  # Angstrom
ADS_HEIGHT = 1.5  # Angstrom above surface
FIXED_LAYERS = 2  # Fix bottom 2 layers

def create_slab(metal, lattice_constant):
    """Create fcc(111) slab with specified parameters."""
    slab = fcc111(metal, size=SLAB_SIZE, a=lattice_constant, vacuum=VACUUM_GAP)
    
    # Fix bottom layers
    c = FixAtoms(indices=[atom.index for atom in slab if atom.tag >= FIXED_LAYERS])
    slab.set_constraint(c)
    
    return slab

def create_adsorbed_system(metal, lattice_constant, adsorbate_type):
    """
    Create slab with adsorbate.
    
    Args:
        metal: Metal symbol
        lattice_constant: Lattice constant in Angstrom
        adsorbate_type: 'O' or 'OH'
    """
    slab = create_slab(metal, lattice_constant)
    
    if adsorbate_type == 'O':
        ads = Atoms('O', positions=[[0, 0, 0]])
    elif adsorbate_type == 'OH':
        # OH molecule coordinates (approximate)
        ads = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.98]])
    else:
        raise ValueError(f"Unknown adsorbate: {adsorbate_type}")
    
    # Add adsorbate at fcc hollow site
    add_adsorbate(slab, ads, height=ADS_HEIGHT, position='fcc')
    
    return slab

def create_gas_phase(adsorbate_type):
    """Create gas phase molecule in a box."""
    if adsorbate_type == 'O':
        # O atom in 10 Å box
        atoms = Atoms('O', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    elif adsorbate_type == 'OH':
        # OH molecule in 10 Å box
        atoms = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=True)
    else:
        raise ValueError(f"Unknown adsorbate: {adsorbate_type}")
    
    return atoms

def compute_adsorption_energy(calc, metal, lattice_constant, adsorbate_type):
    """
    Compute adsorption energy: E_ads = E_slab+ads - E_slab - E_gas
    """
    print(f"  Computing {adsorbate_type} adsorption on {metal}...")
    
    # Clean slab energy
    slab = create_slab(metal, lattice_constant)
    slab.calc = calc
    
    opt_slab = BFGS(slab, logfile=None)
    opt_slab.run(fmax=0.05)
    E_slab = slab.get_potential_energy()
    
    # Adsorbed system energy
    slab_ads = create_adsorbed_system(metal, lattice_constant, adsorbate_type)
    slab_ads.calc = calc
    
    opt_ads = BFGS(slab_ads, logfile=None)
    opt_ads.run(fmax=0.05)
    E_slab_ads = slab_ads.get_potential_energy()
    
    # Gas phase energy
    gas = create_gas_phase(adsorbate_type)
    gas.calc = calc
    E_gas = gas.get_potential_energy()
    
    # Adsorption energy
    E_ads = E_slab_ads - E_slab - E_gas
    
    return {
        'E_ads': E_ads,
        'E_slab': E_slab,
        'E_slab_ads': E_slab_ads,
        'E_gas': E_gas
    }

def run_adsorption_experiment():
    """Run adsorption energy calculations for all metals."""
    print("="*60)
    print("Experiment 2: Adsorption Energy Scaling Relations")
    print("="*60)
    
    # Initialize MACE calculator
    print("\n[1/3] Loading MACE-MP-0 foundation model...")
    calc = mace_mp(model="/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/models/mace-mp-0b3-medium.model", device='cpu')
    
    # Compute adsorption energies
    print("[2/3] Computing adsorption energies...")
    results = {}
    
    for metal, lc in METALS.items():
        print(f"\nProcessing {metal} (a = {lc} Å)...")
        
        results[metal] = {
            'lattice_constant': lc,
            'O': compute_adsorption_energy(calc, metal, lc, 'O'),
            'OH': compute_adsorption_energy(calc, metal, lc, 'OH')
        }
    
    # Extract data for scaling relation
    print("\n[3/3] Analyzing scaling relations...")
    
    metals_list = list(METALS.keys())
    E_O = [results[m]['O']['E_ads'] for m in metals_list]
    E_OH = [results[m]['OH']['E_ads'] for m in metals_list]
    
    # Linear fit for scaling relation: E_OH = a * E_O + b
    fit_coeffs = np.polyfit(E_O, E_OH, 1)
    slope, intercept = fit_coeffs
    
    # Compute R^2
    E_OH_pred = np.polyval(fit_coeffs, E_O)
    ss_res = np.sum((np.array(E_OH) - E_OH_pred)**2)
    ss_tot = np.sum((np.array(E_OH) - np.mean(E_OH))**2)
    r_squared = 1 - ss_res / ss_tot
    
    print("\n" + "="*60)
    print("Adsorption Energy Results (eV):")
    print("="*60)
    print(f"{'Metal':<8} {'E_O':>12} {'E_OH':>12}")
    print("-"*32)
    for i, metal in enumerate(metals_list):
        print(f"{metal:<8} {E_O[i]:>12.3f} {E_OH[i]:>12.3f}")
    
    print(f"\nScaling Relation: E_OH = {slope:.3f} * E_O + {intercept:.3f}")
    print(f"R² = {r_squared:.4f}")
    
    # Save results
    output = {
        'metals': metals_list,
        'E_O': E_O,
        'E_OH': E_OH,
        'scaling_relation': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared)
        },
        'detailed_results': results
    }
    
    os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs', exist_ok=True)
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/adsorption_energies.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to outputs/adsorption_energies.json")
    
    return output

if __name__ == '__main__':
    results = run_adsorption_experiment()
