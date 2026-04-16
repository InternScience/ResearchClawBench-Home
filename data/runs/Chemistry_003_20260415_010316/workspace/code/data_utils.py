"""
Data loading and utility functions for the LES benchmark datasets.
"""
import numpy as np
from ase.io import read
from ase import Atoms
import re
import os


def parse_xyz_file(filepath):
    """Parse extended XYZ file with custom properties."""
    structures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())
        comment_line = lines[i + 1].strip()
        
        # Parse properties from comment line
        props = parse_comment_line(comment_line)
        
        atoms_data = []
        for j in range(natoms):
            parts = lines[i + 2 + j].split()
            species = parts[0]
            # Parse based on what properties are available
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            atom_info = {'species': species, 'pos': coords}
            
            if len(parts) > 4:
                # Has forces
                forces = [float(parts[4]), float(parts[5]), float(parts[6])]
                atom_info['forces'] = forces
            
            atoms_data.append(atom_info)
        
        structure = {
            'natoms': natoms,
            'atoms': atoms_data,
            'properties': props
        }
        structures.append(structure)
        i += natoms + 2
    
    return structures


def parse_comment_line(line):
    """Parse the comment line of an extended XYZ file."""
    props = {}
    
    # Parse key=value pairs
    # Handle quoted values
    pattern = r'(\w+)="([^"]*)"|(\w+)=([^\s]+)'
    for match in re.finditer(pattern, line):
        if match.group(1):
            props[match.group(1)] = match.group(2)
        elif match.group(3):
            props[match.group(3)] = match.group(4)
    
    # Parse Properties= field to understand column layout
    if 'Properties' in props:
        prop_str = props['Properties']
        prop_defs = prop_str.split(':')
        # Parse property definitions like "species:S:1:pos:R:3:forces:R:3"
        # This tells us the column layout
        props['_property_layout'] = prop_str
    
    return props


def load_random_charges(filepath):
    """Load the random_charges dataset.
    
    Returns list of dicts with:
    - positions: (N, 3) array
    - true_charges: (N,) array of ±1
    - species: list of element symbols
    - pbc: bool
    """
    structures = parse_xyz_file(filepath)
    dataset = []
    
    for struct in structures:
        natoms = struct['natoms']
        positions = np.array([a['pos'] for a in struct['atoms']])
        species = [a['species'] for a in struct['atoms']]
        
        # Parse true_charges from properties
        true_charges = None
        if 'true_charges' in struct['properties']:
            charge_str = struct['properties']['true_charges']
            true_charges = np.array([float(x) for x in charge_str.split()])
        
        pbc = struct['properties'].get('pbc', 'F F F') == 'T T T'
        
        dataset.append({
            'positions': positions,
            'true_charges': true_charges,
            'species': species,
            'natoms': natoms,
            'pbc': pbc,
            'cell': None  # Non-periodic
        })
    
    return dataset


def load_charged_dimer(filepath):
    """Load the charged_dimer dataset.
    
    Returns list of dicts with:
    - positions: (N, 3) array
    - forces: (N, 3) array
    - energy: float
    - species: list of element symbols
    """
    structures = parse_xyz_file(filepath)
    dataset = []
    
    for struct in structures:
        natoms = struct['natoms']
        positions = np.array([a['pos'] for a in struct['atoms']])
        forces = np.array([a['forces'] for a in struct['atoms']])
        species = [a['species'] for a in struct['atoms']]
        energy = float(struct['properties'].get('energy', 0.0))
        
        # Compute center-of-mass separation between the two dimers
        # First dimer: atoms 0-3 (C + 3H), second dimer: atoms 4-7 (C + 3H)
        com1 = positions[:4].mean(axis=0)
        com2 = positions[4:].mean(axis=0)
        separation = np.linalg.norm(com2 - com1)
        
        dataset.append({
            'positions': positions,
            'forces': forces,
            'energy': energy,
            'species': species,
            'natoms': natoms,
            'separation': separation,
            'pbc': False,
            'cell': None
        })
    
    return dataset


def load_ag3_chargestates(filepath):
    """Load the ag3_chargestates dataset.
    
    Returns list of dicts with:
    - positions: (N, 3) array
    - forces: (N, 3) array
    - energy: float
    - charge_state: int (+1 or -1)
    - total_charge: int
    - species: list of element symbols
    """
    structures = parse_xyz_file(filepath)
    dataset = []
    
    for struct in structures:
        natoms = struct['natoms']
        positions = np.array([a['pos'] for a in struct['atoms']])
        forces = np.array([a['forces'] for a in struct['atoms']])
        species = [a['species'] for a in struct['atoms']]
        energy = float(struct['properties'].get('energy', 0.0))
        charge_state = int(struct['properties'].get('charge_state', 0))
        total_charge = int(struct['properties'].get('total_charge', 0))
        
        # Compute Ag-Ag bond lengths
        # Ag-Ag distances
        d01 = np.linalg.norm(positions[1] - positions[0])
        d02 = np.linalg.norm(positions[2] - positions[0])
        d12 = np.linalg.norm(positions[2] - positions[1])
        
        dataset.append({
            'positions': positions,
            'forces': forces,
            'energy': energy,
            'species': species,
            'natoms': natoms,
            'charge_state': charge_state,
            'total_charge': total_charge,
            'bond_lengths': [d01, d02, d12],
            'pbc': False,
            'cell': None
        })
    
    return dataset


def compute_coulomb_energy(positions, charges, epsilon=0.0):
    """Compute Coulomb energy for a set of point charges.
    
    E = sum_{i<j} q_i * q_j / r_ij
    
    Args:
        positions: (N, 3) array
        charges: (N,) array
        epsilon: dielectric constant (default 1, vacuum)
    
    Returns:
        Total Coulomb energy
    """
    N = len(charges)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = np.linalg.norm(positions[i] - positions[j])
            energy += charges[i] * charges[j] / rij
    return energy


def compute_coulomb_forces(positions, charges):
    """Compute Coulomb forces for a set of point charges.
    
    F_i = sum_{j!=i} q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
    
    Returns:
        (N, 3) force array
    """
    N = len(charges)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij_vec = positions[i] - positions[j]
            rij = np.linalg.norm(rij_vec)
            forces[i] += charges[i] * charges[j] * rij_vec / (rij ** 3)
    return forces


def compute_lj_energy(positions, epsilon_lj=1.0, sigma_lj=1.0):
    """Compute Lennard-Jones repulsive energy.
    
    U_LJ = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    We use only the repulsive part: U_rep = 4 * epsilon * (sigma/r)^12
    
    Args:
        positions: (N, 3) array
        epsilon_lj: LJ epsilon parameter
        sigma_lj: LJ sigma parameter
    
    Returns:
        Total LJ energy
    """
    N = len(positions)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = np.linalg.norm(positions[i] - positions[j])
            sr6 = (sigma_lj / rij) ** 6
            energy += 4.0 * epsilon_lj * (sr6 * sr6 - sr6)
    return energy


def compute_lj_forces(positions, epsilon_lj=1.0, sigma_lj=1.0):
    """Compute LJ forces.
    
    Returns:
        (N, 3) force array
    """
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i+1, N):
            rij_vec = positions[i] - positions[j]
            rij = np.linalg.norm(rij_vec)
            sr6 = (sigma_lj / rij) ** 6
            # F = 24 * epsilon * (2 * sr6^2 - sr6) / r * r_hat
            f_mag = 24.0 * epsilon_lj * (2.0 * sr6 * sr6 - sr6) / rij
            f_vec = f_mag * rij_vec / rij
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces


if __name__ == '__main__':
    # Test data loading
    print("Loading random_charges...")
    rc_data = load_random_charges('data/random_charges.xyz')
    print(f"  Loaded {len(rc_data)} structures")
    print(f"  First structure: {rc_data[0]['natoms']} atoms")
    print(f"  True charges: {rc_data[0]['true_charges'][:5]}... (sum={rc_data[0]['true_charges'].sum()})")
    
    print("\nLoading charged_dimer...")
    cd_data = load_charged_dimer('data/charged_dimer.xyz')
    print(f"  Loaded {len(cd_data)} structures")
    print(f"  First structure: {cd_data[0]['natoms']} atoms, energy={cd_data[0]['energy']:.4f}")
    print(f"  Separation range: {min(d['separation'] for d in cd_data):.2f} - {max(d['separation'] for d in cd_data):.2f}")
    
    print("\nLoading ag3_chargestates...")
    ag_data = load_ag3_chargestates('data/ag3_chargestates.xyz')
    print(f"  Loaded {len(ag_data)} structures")
    cs_pos = sum(1 for d in ag_data if d['charge_state'] == 1)
    cs_neg = sum(1 for d in ag_data if d['charge_state'] == -1)
    print(f"  Charge state +1: {cs_pos}, -1: {cs_neg}")
    print(f"  Energy range: {min(d['energy'] for d in ag_data):.4f} - {max(d['energy'] for d in ag_data):.4f}")
