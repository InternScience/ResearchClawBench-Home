#!/usr/bin/env python3
"""Inspect the three .xyz datasets for the LES-style ML potential task."""
import os
from ase.io import read
import numpy as np

DATA_DIR = "data"
FILES = [
    "random_charges.xyz",
    "charged_dimer.xyz",
    "ag3_chargestates.xyz",
]

def summarize(atoms_list, name):
    n_frames = len(atoms_list)
    first = atoms_list[0]
    n_atoms = len(first)
    species = sorted(set(first.get_chemical_symbols()))
    has_energy = "energy" in first.info
    has_forces = (first.calc is not None and
                  first.calc.results.get("forces") is not None)
    has_charges = "charges" in first.arrays or "charge" in first.info
    energies = [a.info.get("energy") for a in atoms_list if "energy" in a.info]
    print(f"\n=== {name} ===")
    print(f"Frames: {n_frames}")
    print(f"Atoms per frame: {n_atoms}")
    print(f"Species: {species}")
    print(f"Has energy: {has_energy}  |  Has forces: {has_forces}  |  Has charges: {has_charges}")
    if energies:
        print(f"Energy range: {min(energies):.4f} ... {max(energies):.4f} eV")
    if "charges" in first.arrays:
        ch = first.arrays["charges"]
        print(f"Charge range (first frame): {ch.min():.2f} ... {ch.max():.2f}")
    if "total_charge" in first.info:
        print(f"Total charge (first frame): {first.info['total_charge']}")

def main():
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        atoms_list = read(path, index=":")
        summarize(atoms_list, fname)

if __name__ == "__main__":
    main()