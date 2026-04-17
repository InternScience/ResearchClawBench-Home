#!/usr/bin/env python3
"""
Data analysis module for Chemistry_003 task.
Parses XYZ files and extracts structures, energies, forces, and charges.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def parse_xyz_file(filepath: str) -> List[Dict]:
    """
    Parse an XYZ file and extract configuration data.
    
    Returns a list of dictionaries, each containing:
    - n_atoms: number of atoms
    - species: list of element symbols
    - positions: Nx3 array of atomic positions
    - energy: total energy (if available)
    - forces: Nx3 array of forces (if available)
    - charges: true charges from properties line (if available)
    - pbc: periodic boundary conditions
    - charge_state: charge state label (if available)
    - total_charge: total system charge (if available)
    """
    configurations = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read number of atoms
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        
        # Read properties line
        props_line = lines[i + 1].strip()
        
        # Parse properties
        config = {'n_atoms': n_atoms}
        
        # Extract species and positions
        species = []
        positions = []
        forces = []
        
        for j in range(n_atoms):
            atom_line = lines[i + 2 + j].strip().split()
            species.append(atom_line[0])
            positions.append([float(atom_line[1]), float(atom_line[2]), float(atom_line[3])])
            if len(atom_line) >= 7:
                forces.append([float(atom_line[4]), float(atom_line[5]), float(atom_line[6])])
        
        config['species'] = species
        config['positions'] = np.array(positions)
        
        if forces:
            config['forces'] = np.array(forces)
        
        # Parse properties line for metadata
        # Look for energy=
        energy_match = re.search(r'energy=(-?\d+\.?\d*)', props_line)
        if energy_match:
            config['energy'] = float(energy_match.group(1))
        
        # Look for pbc=
        pbc_match = re.search(r'pbc="([^"]*)"', props_line)
        if pbc_match:
            config['pbc'] = pbc_match.group(1)
        
        # Look for true_charges=
        charges_match = re.search(r'true_charges="([^"]*)"', props_line)
        if charges_match:
            charges_str = charges_match.group(1).split()
            config['true_charges'] = np.array([float(c) for c in charges_str])
        
        # Look for charge_state=
        charge_state_match = re.search(r'charge_state=(-?\d+)', props_line)
        if charge_state_match:
            config['charge_state'] = int(charge_state_match.group(1))
        
        # Look for total_charge=
        total_charge_match = re.search(r'total_charge=(-?\d+)', props_line)
        if total_charge_match:
            config['total_charge'] = int(total_charge_match.group(1))
        
        configurations.append(config)
        i += 2 + n_atoms
    
    return configurations


def compute_pairwise_distances(positions: np.ndarray, pbc: str = "F F F", 
                                box_size: Optional[float] = None) -> np.ndarray:
    """
    Compute pairwise distances between atoms.
    
    Args:
        positions: Nx3 array of atomic positions
        pbc: periodic boundary conditions string
        box_size: box size for PBC (if applicable)
    
    Returns:
        NxN array of pairwise distances
    """
    n_atoms = len(positions)
    distances = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dx = positions[i] - positions[j]
            
            # Apply minimum image convention if PBC is enabled
            if pbc == "T T T" and box_size is not None:
                dx = dx - box_size * np.round(dx / box_size)
            
            dist = np.linalg.norm(dx)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def compute_coulomb_energy(charges: np.ndarray, positions: np.ndarray,
                           pbc: str = "F F F") -> float:
    """
    Compute Coulomb energy for a system of point charges.
    
    E = sum_{i<j} q_i * q_j / r_ij
    
    In atomic units where 4*pi*epsilon_0 = 1.
    """
    n_atoms = len(charges)
    energy = 0.0
    
    distances = compute_pairwise_distances(positions, pbc)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] > 1e-10:  # Avoid self-interaction
                energy += charges[i] * charges[j] / distances[i, j]
    
    return energy


def compute_lj_energy(positions: np.ndarray, epsilon: float = 0.1,
                      sigma: float = 1.0, cutoff: float = None,
                      pbc: str = "F F F") -> float:
    """
    Compute Lennard-Jones energy for a system.
    
    V_LJ = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    """
    n_atoms = len(positions)
    
    if cutoff is None:
        cutoff = 2.5 * sigma
    
    energy = 0.0
    
    distances = compute_pairwise_distances(positions, pbc)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = distances[i, j]
            if r < cutoff and r > 1e-10:
                sr = sigma / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                energy += 4 * epsilon * (sr12 - sr6)
    
    return energy


if __name__ == "__main__":
    import sys
    
    # Test parsing
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        configs = parse_xyz_file(filepath)
        print(f"Parsed {len(configs)} configurations from {filepath}")
        
        if configs:
            c = configs[0]
            print(f"First config: {c['n_atoms']} atoms")
            print(f"Species: {c['species'][:5]}...")
            print(f"Positions shape: {c['positions'].shape}")
            if 'energy' in c:
                print(f"Energy: {c['energy']}")
            if 'true_charges' in c:
                print(f"True charges: min={c['true_charges'].min()}, max={c['true_charges'].max()}")
                print(f"Net charge: {c['true_charges'].sum()}")
