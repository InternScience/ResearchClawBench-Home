#!/usr/bin/env python3
"""
Adsorption energy scaling relations on transition metal fcc(111) surfaces
Metals: Ni, Cu, Rh, Pd, Ir, Pt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import torch

MODEL_PATH = "data/mace-mp-0b3-medium.model"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

METALS = ['Ni', 'Cu', 'Rh', 'Pd', 'Ir', 'Pt']
LATTICE_CONSTANTS = {'Ni': 3.52, 'Cu': 3.61, 'Rh': 3.80, 'Pd': 3.89, 'Ir': 3.84, 'Pt': 3.92}
ADSORBATES = ['O', 'OH', 'OOH']

def get_slab(metal, a):
    slab = fcc111(metal, size=(3, 3, 4), a=a, vacuum=15.0)
    slab.set_pbc(True)
    return slab

def compute_adsorption_energy(slab, adsorbate, calc):
    slab = slab.copy()
    slab.calc = calc
    e_slab = slab.get_potential_energy()
    if adsorbate == 'O':
        ads = Atoms('O', positions=[[0, 0, 0]])
    elif adsorbate == 'OH':
        ads = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.96]])
    elif adsorbate == 'OOH':
        ads = Atoms('OOH', positions=[[0, 0, 0], [0, 0, 1.3], [0, 0, 2.2]])
    else:
        raise ValueError(f"Unknown adsorbate {adsorbate}")
    add_adsorbate(slab, ads, height=1.8, position='ontop')
    slab.calc = calc
    opt = BFGS(slab, logfile=None)
    opt.run(fmax=0.05, steps=50)
    e_ads = slab.get_potential_energy()
    return e_ads - e_slab

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc = MACECalculator(model_path=MODEL_PATH, device=device, default_dtype="float32")

    results = {}
    for metal in METALS:
        a = LATTICE_CONSTANTS[metal]
        slab = get_slab(metal, a)
        results[metal] = {}
        for ads in ADSORBATES:
            e = compute_adsorption_energy(slab, ads, calc)
            results[metal][ads] = e
            print(f"{metal} {ads}: {e:.3f} eV")

    # Save
    with open(f"{OUTPUT_DIR}/adsorption_energies.txt", "w") as f:
        f.write("Metal O OH OOH\n")
        for m in METALS:
            f.write(f"{m} {results[m]['O']:.4f} {results[m]['OH']:.4f} {results[m]['OOH']:.4f}\n")

    # Plot scaling
    plt.figure(figsize=(8, 6))
    x = [results[m]['O'] for m in METALS]
    y = [results[m]['OH'] for m in METALS]
    plt.scatter(x, y, s=100, c='blue', label='OH vs O')
    for i, m in enumerate(METALS):
        plt.annotate(m, (x[i], y[i]), fontsize=10)
    plt.xlabel('E_O (eV)', fontsize=12)
    plt.ylabel('E_OH (eV)', fontsize=12)
    plt.title('Adsorption Scaling on fcc(111)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/adsorption_scaling.png", dpi=150)
    plt.close()

    print("Adsorption results saved.")

if __name__ == "__main__":
    main()
