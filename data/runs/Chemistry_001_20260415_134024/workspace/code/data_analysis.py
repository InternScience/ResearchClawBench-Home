"""
Data Analysis Module for Biomolecular Complex Structure Prediction
Analyzes the FKBP12 protein (2L3R) and FK506 ligand structures.
"""

import json
import numpy as np
import os

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_001_20260415_134024"
DATA_DIR = os.path.join(WORKSPACE, "data/sample/2l3r")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

def parse_pdb(pdb_path):
    """Parse PDB file and extract atom information."""
    atoms = []
    ca_atoms = []
    residues = {}
    sequence = []
    
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip() if len(line) > 76 else ''
                
                atom_info = {
                    'atom_name': atom_name,
                    'res_name': res_name,
                    'chain_id': chain_id,
                    'res_seq': res_seq,
                    'x': x, 'y': y, 'z': z,
                    'element': element
                }
                atoms.append(atom_info)
                
                if atom_name == 'CA':
                    ca_atoms.append(atom_info)
                
                if res_seq not in residues:
                    residues[res_seq] = {
                        'res_name': res_name,
                        'chain_id': chain_id,
                        'atoms': []
                    }
                residues[res_seq]['atoms'].append(atom_info)
    
    # Build sequence
    for res_seq in sorted(residues.keys()):
        res_name = residues[res_seq]['res_name']
        one_letter = three_to_one.get(res_name, 'X')
        sequence.append(one_letter)
    
    return {
        'atoms': atoms,
        'ca_atoms': ca_atoms,
        'residues': residues,
        'sequence': ''.join(sequence),
        'n_atoms': len(atoms),
        'n_residues': len(residues),
        'n_ca': len(ca_atoms)
    }


def parse_sdf(sdf_path):
    """Parse SDF file and extract molecular information."""
    atoms = []
    bonds = []
    
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    # Header
    mol_name = lines[0].strip()
    
    # Counts line (line 3, 0-indexed) - V2000 fixed-width format
    counts_line = lines[3]
    n_atoms = int(counts_line[0:3])
    n_bonds = int(counts_line[3:6])
    
    # Atom block
    for i in range(4, 4 + n_atoms):
        parts = lines[i].split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        element = parts[3]
        atoms.append({
            'index': i - 4,
            'x': x, 'y': y, 'z': z,
            'element': element
        })
    
    # Bond block
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        parts = lines[i].split()
        atom1 = int(parts[0]) - 1  # 0-indexed
        atom2 = int(parts[1]) - 1
        bond_type = int(parts[2])
        bonds.append({
            'atom1': atom1,
            'atom2': atom2,
            'bond_type': bond_type
        })
    
    # Count elements
    element_counts = {}
    for atom in atoms:
        el = atom['element']
        element_counts[el] = element_counts.get(el, 0) + 1
    
    # Compute center of mass (unweighted)
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    center = coords.mean(axis=0)
    
    # Compute radius of gyration
    diffs = coords - center
    rg = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    
    return {
        'mol_name': mol_name,
        'atoms': atoms,
        'bonds': bonds,
        'n_atoms': n_atoms,
        'n_bonds': n_bonds,
        'element_counts': element_counts,
        'center_of_mass': center.tolist(),
        'radius_of_gyration': float(rg),
        'coords': coords
    }


def compute_distance_matrix(coords):
    """Compute pairwise distance matrix."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def compute_contact_map(ca_coords, threshold=8.0):
    """Compute contact map from CA coordinates."""
    n = len(ca_coords)
    contacts = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if d < threshold:
                contacts[i][j] = 1
                contacts[j][i] = 1
    return contacts


def analyze_binding_interface(protein_data, ligand_data, threshold=5.0):
    """Identify protein residues near the ligand."""
    ligand_coords = ligand_data['coords']
    interface_residues = []
    
    for res_seq, res_info in protein_data['residues'].items():
        min_dist = float('inf')
        for atom in res_info['atoms']:
            pcoord = np.array([atom['x'], atom['y'], atom['z']])
            for lcoord in ligand_coords:
                d = np.linalg.norm(pcoord - lcoord)
                if d < min_dist:
                    min_dist = d
        if min_dist < threshold:
            interface_residues.append({
                'res_seq': res_seq,
                'res_name': res_info['res_name'],
                'min_distance': float(min_dist)
            })
    
    return interface_residues


def compute_rmsd(coords1, coords2):
    """Compute RMSD between two coordinate sets."""
    assert len(coords1) == len(coords2)
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def kabsch_align(P, Q):
    """Align P onto Q using Kabsch algorithm. Returns aligned P."""
    # Center
    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)
    P_centered = P - p_center
    Q_centered = Q - q_center
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)
    
    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T
    
    # Apply rotation and translation
    P_aligned = (P_centered @ R.T) + q_center
    
    return P_aligned, R, p_center, q_center


def hungarian_rmsd(coords1, coords2):
    """Compute symmetry-aware RMSD using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment
    
    n1, n2 = len(coords1), len(coords2)
    if n1 != n2:
        # Use the smaller set
        n = min(n1, n2)
    else:
        n = n1
    
    # Compute cost matrix
    cost = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost[i][j] = np.sum((coords1[i] - coords2[j])**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matched_dist_sq = cost[row_ind, col_ind]
    rmsd = np.sqrt(np.mean(matched_dist_sq))
    
    return float(rmsd), row_ind, col_ind


if __name__ == "__main__":
    print("=" * 60)
    print("Data Analysis: FKBP12-FK506 Complex (PDB: 2L3R)")
    print("=" * 60)
    
    # Parse protein
    pdb_path = os.path.join(DATA_DIR, "2l3r_protein.pdb")
    protein = parse_pdb(pdb_path)
    print(f"\n--- Protein Analysis ---")
    print(f"Total atoms: {protein['n_atoms']}")
    print(f"Total residues: {protein['n_residues']}")
    print(f"CA atoms: {protein['n_ca']}")
    print(f"Sequence length: {len(protein['sequence'])}")
    print(f"Sequence: {protein['sequence']}")
    
    # Parse ligand
    sdf_path = os.path.join(DATA_DIR, "2l3r_ligand.sdf")
    ligand = parse_sdf(sdf_path)
    print(f"\n--- Ligand Analysis ---")
    print(f"Molecule: {ligand['mol_name']}")
    print(f"Total atoms: {ligand['n_atoms']}")
    print(f"Total bonds: {ligand['n_bonds']}")
    print(f"Element counts: {ligand['element_counts']}")
    print(f"Center of mass: {[f'{c:.2f}' for c in ligand['center_of_mass']]}")
    print(f"Radius of gyration: {ligand['radius_of_gyration']:.2f} Å")
    
    # Heavy atoms only
    heavy_atoms = [a for a in ligand['atoms'] if a['element'] != 'H']
    print(f"Heavy atoms: {len(heavy_atoms)}")
    
    # Compute CA coordinate statistics
    ca_coords = np.array([[a['x'], a['y'], a['z']] for a in protein['ca_atoms']])
    ca_center = ca_coords.mean(axis=0)
    ca_rg = np.sqrt(np.mean(np.sum((ca_coords - ca_center)**2, axis=1)))
    print(f"\n--- Protein Geometry ---")
    print(f"CA center of mass: {[f'{c:.2f}' for c in ca_center]}")
    print(f"CA radius of gyration: {ca_rg:.2f} Å")
    
    # Contact map
    contacts = compute_contact_map(ca_coords, threshold=8.0)
    n_contacts = int(np.sum(contacts) / 2)
    print(f"Number of CA-CA contacts (8Å): {n_contacts}")
    
    # Binding interface
    interface = analyze_binding_interface(protein, ligand, threshold=5.0)
    print(f"\n--- Binding Interface ---")
    print(f"Interface residues (within 5Å): {len(interface)}")
    for res in sorted(interface, key=lambda x: x['min_distance']):
        print(f"  {res['res_name']}{res['res_seq']}: {res['min_distance']:.2f} Å")
    
    # Save results
    results = {
        'protein': {
            'n_atoms': protein['n_atoms'],
            'n_residues': protein['n_residues'],
            'n_ca': protein['n_ca'],
            'sequence': protein['sequence'],
            'ca_center_of_mass': [float(c) for c in ca_center],
            'ca_radius_of_gyration': float(ca_rg),
            'n_contacts_8A': n_contacts
        },
        'ligand': {
            'mol_name': ligand['mol_name'],
            'n_atoms': ligand['n_atoms'],
            'n_bonds': ligand['n_bonds'],
            'n_heavy_atoms': len(heavy_atoms),
            'element_counts': ligand['element_counts'],
            'center_of_mass': ligand['center_of_mass'],
            'radius_of_gyration': ligand['radius_of_gyration']
        },
        'interface': {
            'n_interface_residues': len(interface),
            'residues': interface
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "data_analysis.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to outputs/data_analysis.json")
