"""Compute energies and forces for random_charges dataset using Coulomb + LJ potential."""
import numpy as np
import json
import re
import os

def parse_xyz(filepath):
    """Parse extended XYZ file with properties in comment line."""
    structures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip().replace('\r', '')
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        
        comment = lines[i+1].strip().replace('\r', '')
        
        props = {}
        m = re.search(r'energy=([-\d.eE+]+)', comment)
        if m:
            props['energy'] = float(m.group(1))
        m = re.search(r'pbc="([^"]*)"', comment)
        if m:
            props['pbc'] = m.group(1).split()
        m = re.search(r'true_charges="([^"]*)"', comment)
        if m:
            props['true_charges'] = [float(x) for x in m.group(1).split()]
        m = re.search(r'charge_state=([-\d]+)', comment)
        if m:
            props['charge_state'] = int(m.group(1))
        m = re.search(r'total_charge=([-\d.eE+]+)', comment)
        if m:
            props['total_charge'] = float(m.group(1))
        
        has_forces = 'forces:R:3' in comment
        
        positions = []
        species = []
        forces = []
        
        for j in range(i+2, i+2+n_atoms):
            parts = lines[j].strip().replace('\r', '').split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if has_forces and len(parts) >= 7:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        struct = {
            'n_atoms': n_atoms,
            'species': species,
            'positions': np.array(positions),
            'comment': comment,
            'props': props,
        }
        if has_forces and forces:
            struct['forces'] = np.array(forces)
        
        structures.append(struct)
        i = i + 2 + n_atoms
    
    return structures


def coulomb_lj_energy(positions, charges, sigma=1.0, epsilon=0.1, cutoff=None):
    """Compute Coulomb + repulsive LJ energy for point charges."""
    n = len(positions)
    energy = 0.0
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if cutoff is not None and r > cutoff:
                continue
            
            # Coulomb: q_i * q_j / r
            e_coul = charges[i] * charges[j] / r
            
            # Repulsive LJ: 4*epsilon*(sigma/r)^12
            sr6 = (sigma / r) ** 6
            e_lj = 4 * epsilon * sr6 * sr6  # (sigma/r)^12
            
            energy += e_coul + e_lj
            
            # Forces
            f_mag = charges[i] * charges[j] / r**3 + 4 * epsilon * 12 * sr6 * sr6 / r
            fij = f_mag * rij
            forces[i] += fij
            forces[j] -= fij
    
    return energy, forces


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    structures = parse_xyz(os.path.join(base, 'data/random_charges.xyz'))
    
    print(f"Loaded {len(structures)} structures")
    
    energies = []
    all_forces = []
    
    for idx, s in enumerate(structures):
        charges = np.array(s['props']['true_charges'])
        pos = s['positions']
        e, f = coulomb_lj_energy(pos, charges)
        energies.append(e)
        all_forces.append(f)
    
    energies = np.array(energies)
    print(f"Computed energies: min={energies.min():.4f}, max={energies.max():.4f}, mean={energies.mean():.4f}")
    
    # Save computed energies
    results = {
        'energies': energies.tolist(),
        'n_structures': len(structures),
    }
    
    os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)
    with open(os.path.join(base, 'outputs/random_charges_energies.json'), 'w') as f:
        json.dump(results, f)
    
    print("Saved to outputs/random_charges_energies.json")
