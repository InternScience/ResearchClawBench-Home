"""
Data preprocessing utilities for the unified diffusion-based biomolecular complex
structure prediction framework.

Handles parsing of PDB protein structures and SDF ligand structures,
featurization of sequences and molecular graphs, and preparation of training data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import re


# Amino acid one-letter code mapping
AA_TO_INT = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}

INT_TO_AA = {v: k for k, v in AA_TO_INT.items()}

# Atom type encoding for ligands
ATOM_TYPE_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5,
    'Cl': 6, 'Br': 7, 'I': 8, 'H': 9, 'B': 10,
}

NUM_ATOM_TYPES = len(ATOM_TYPE_MAP)


def parse_pdb(pdb_path: str) -> Dict:
    """
    Parse a PDB file and extract atom coordinates, residue information.
    
    Returns:
        Dictionary with:
            - 'ca_coords': numpy array of CA atom coordinates (N_ca x 3)
            - 'all_coords': numpy array of all atom coordinates (N_atoms x 3)
            - 'residue_names': list of residue names
            - 'residue_ids': list of residue sequence IDs
            - 'atom_names': list of atom names
            - 'sequence': protein sequence string (one-letter codes)
    """
    ca_coords = []
    all_coords = []
    residue_names = []
    residue_ids = []
    atom_names = []
    chain_ids = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                all_coords.append([x, y, z])
                residue_names.append(res_name)
                residue_ids.append(res_id)
                atom_names.append(atom_name)
                chain_ids.append(chain_id)
                
                if atom_name == 'CA':
                    ca_coords.append([x, y, z])
    
    # Build sequence from residues (using CA atoms as reference)
    sequence = []
    seen_residues = set()
    for res_name, res_id in zip(residue_names, residue_ids):
        key = (res_id, res_name)
        if key not in seen_residues:
            seen_residues.add(key)
            if res_name in AA_TO_INT:
                sequence.append(res_name)
    
    return {
        'ca_coords': np.array(ca_coords, dtype=np.float32),
        'all_coords': np.array(all_coords, dtype=np.float32),
        'residue_names': residue_names,
        'residue_ids': residue_ids,
        'atom_names': atom_names,
        'chain_ids': chain_ids,
        'sequence': sequence,
        'n_residues': len(sequence),
        'n_ca_atoms': len(ca_coords),
        'n_all_atoms': len(all_coords),
    }


def parse_sdf(sdf_path: str) -> Dict:
    """
    Parse an SDF file and extract molecular graph information.
    
    Returns:
        Dictionary with:
            - 'atom_coords': numpy array of atom coordinates (N_atoms x 3)
            - 'atom_types': list of atom element symbols
            - 'atom_type_indices': numpy array of atom type indices
            - 'bonds': list of (atom_idx1, atom_idx2, bond_order) tuples
            - 'n_atoms': number of atoms
            - 'n_bonds': number of bonds
            - 'heavy_atom_mask': boolean mask for non-hydrogen atoms
    """
    atom_coords = []
    atom_types = []
    bonds = []
    
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    # Find the counts line (typically line 4, 0-indexed line 3)
    counts_line_idx = None
    for i, line in enumerate(lines):
        if 'V2000' in line or 'V3000' in line:
            counts_line_idx = i
            break
    
    if counts_line_idx is None:
        raise ValueError("Cannot find V2000/V3000 marker in SDF file")
    
    counts_line = lines[counts_line_idx]
    n_atoms = int(counts_line[0:3].strip())
    n_bonds = int(counts_line[3:6].strip())
    
    # Parse atom block
    atom_start = counts_line_idx + 1
    for i in range(n_atoms):
        line = lines[atom_start + i]
        x = float(line[0:10])
        y = float(line[10:20])
        z = float(line[20:30])
        element = line[31:34].strip()
        
        atom_coords.append([x, y, z])
        atom_types.append(element)
    
    # Parse bond block
    bond_start = atom_start + n_atoms
    for i in range(n_bonds):
        line = lines[bond_start + i]
        atom1 = int(line[0:3]) - 1  # 0-indexed
        atom2 = int(line[3:6]) - 1
        bond_order = int(line[6:9])
        bonds.append((atom1, atom2, bond_order))
    
    # Convert atom types to indices
    atom_type_indices = []
    for elem in atom_types:
        if elem in ATOM_TYPE_MAP:
            atom_type_indices.append(ATOM_TYPE_MAP[elem])
        else:
            atom_type_indices.append(0)  # Default to carbon
    
    # Heavy atom mask
    heavy_atom_mask = [elem != 'H' for elem in atom_types]
    
    return {
        'atom_coords': np.array(atom_coords, dtype=np.float32),
        'atom_types': atom_types,
        'atom_type_indices': np.array(atom_type_indices, dtype=np.int64),
        'bonds': bonds,
        'n_atoms': n_atoms,
        'n_bonds': n_bonds,
        'heavy_atom_mask': np.array(heavy_atom_mask),
        'n_heavy_atoms': sum(heavy_atom_mask),
    }


def sequence_to_onehot(sequence: List[str]) -> np.ndarray:
    """Convert residue name sequence to one-hot encoding."""
    n_res = len(sequence)
    onehot = np.zeros((n_res, len(AA_TO_INT)), dtype=np.float32)
    for i, res_name in enumerate(sequence):
        if res_name in AA_TO_INT:
            onehot[i, AA_TO_INT[res_name]] = 1.0
    return onehot


def compute_adjacency_matrix(n_atoms: int, bonds: List[Tuple]) -> np.ndarray:
    """Compute adjacency matrix from bond list."""
    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for atom1, atom2, order in bonds:
        adj[atom1, atom2] = order
        adj[atom2, atom1] = order
    return adj


def center_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center coordinates at origin and return the centroid."""
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    return centered, centroid


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)


def compute_protein_features(protein_data: Dict) -> Dict:
    """Compute additional features from parsed protein data."""
    ca_coords = protein_data['ca_coords']
    centered_ca, centroid = center_coordinates(ca_coords)
    pairwise_dist = compute_pairwise_distances(centered_ca)
    onehot = sequence_to_onehot(protein_data['sequence'])
    
    # Compute CA-CA distance statistics
    upper_tri_indices = np.triu_indices(len(ca_coords), k=1)
    dist_values = pairwise_dist[upper_tri_indices]
    
    return {
        'centered_ca_coords': centered_ca,
        'centroid': centroid,
        'pairwise_distances': pairwise_dist,
        'onehot_sequence': onehot,
        'mean_ca_distance': float(np.mean(dist_values)),
        'std_ca_distance': float(np.std(dist_values)),
        'min_ca_distance': float(np.min(dist_values)),
        'max_ca_distance': float(np.max(dist_values)),
    }


def compute_ligand_features(ligand_data: Dict) -> Dict:
    """Compute additional features from parsed ligand data."""
    atom_coords = ligand_data['atom_coords']
    centered_coords, centroid = center_coordinates(atom_coords)
    heavy_mask = ligand_data['heavy_atom_mask']
    heavy_coords = atom_coords[heavy_mask]
    centered_heavy, heavy_centroid = center_coordinates(heavy_coords)
    
    adj = compute_adjacency_matrix(ligand_data['n_atoms'], ligand_data['bonds'])
    pairwise_dist = compute_pairwise_distances(centered_coords)
    
    return {
        'centered_coords': centered_coords,
        'centroid': centroid,
        'centered_heavy_coords': centered_heavy,
        'heavy_centroid': heavy_centroid,
        'adjacency_matrix': adj,
        'pairwise_distances': pairwise_dist,
        'molecular_radius': float(np.max(np.linalg.norm(centered_heavy, axis=1))),
    }


if __name__ == '__main__':
    import json
    
    # Test parsing
    protein = parse_pdb('data/sample/2l3r/2l3r_protein.pdb')
    ligand = parse_sdf('data/sample/2l3r/2l3r_ligand.sdf')
    
    print(f"Protein: {protein['n_residues']} residues, {protein['n_ca_atoms']} CA atoms")
    print(f"Protein CA coords shape: {protein['ca_coords'].shape}")
    print(f"Ligand: {ligand['n_atoms']} atoms, {ligand['n_bonds']} bonds")
    print(f"Ligand heavy atoms: {ligand['n_heavy_atoms']}")
    print(f"Ligand atom types: {ligand['atom_types'][:10]}...")
    
    # Compute features
    prot_feats = compute_protein_features(protein)
    lig_feats = compute_ligand_features(ligand)
    
    print(f"\nProtein features:")
    print(f"  Mean CA distance: {prot_feats['mean_ca_distance']:.3f} A")
    print(f"  Std CA distance: {prot_feats['std_ca_distance']:.3f} A")
    print(f"  Sequence length: {len(protein['sequence'])}")
    
    print(f"\nLigand features:")
    print(f"  Molecular radius: {lig_feats['molecular_radius']:.3f} A")
    print(f"  Number of atoms: {ligand['n_atoms']}")
    
    # Save processed data
    np.savez('outputs/protein_data.npz',
             ca_coords=protein['ca_coords'],
             all_coords=protein['all_coords'],
             centered_ca=prot_feats['centered_ca_coords'],
             onehot_sequence=prot_feats['onehot_sequence'],
             pairwise_distances=prot_feats['pairwise_distances'])
    
    np.savez('outputs/ligand_data.npz',
             atom_coords=ligand['atom_coords'],
             atom_type_indices=ligand['atom_type_indices'],
             centered_coords=lig_feats['centered_coords'],
             adjacency_matrix=lig_feats['adjacency_matrix'],
             heavy_atom_mask=ligand['heavy_atom_mask'])
    
    # Save metadata
    metadata = {
        'protein': {
            'n_residues': protein['n_residues'],
            'n_ca_atoms': protein['n_ca_atoms'],
            'n_all_atoms': protein['n_all_atoms'],
            'sequence_length': len(protein['sequence']),
            'mean_ca_distance': prot_feats['mean_ca_distance'],
            'std_ca_distance': prot_feats['std_ca_distance'],
            'pdb_id': '2L3R',
            'protein_name': 'FKBP12',
        },
        'ligand': {
            'n_atoms': ligand['n_atoms'],
            'n_bonds': ligand['n_bonds'],
            'n_heavy_atoms': int(ligand['n_heavy_atoms']),
            'molecular_radius': lig_feats['molecular_radius'],
            'ligand_name': 'FK506',
        },
    }
    
    with open('outputs/data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nProcessed data saved to outputs/")
