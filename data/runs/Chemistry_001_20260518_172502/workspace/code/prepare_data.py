"""
Data preparation for UniDiff-Complex on the 2L3R protein-ligand complex.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os

# Amino acid mapping
AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']

def parse_pdb(pdb_path: str):
    """Parse PDB file and extract CA coordinates and sequence."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    ca_coords = []
    sequence = []
    residue_indices = []
    all_coords = []
    all_atoms = []
    
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            
            all_coords.append([x, y, z])
            all_atoms.append({
                'name': atom_name,
                'residue': res_name,
                'res_seq': res_seq,
                'element': element,
                'coord': [x, y, z]
            })
            
            if atom_name == 'CA':
                ca_coords.append([x, y, z])
                if res_name in AA_MAP:
                    sequence.append(AA_MAP[res_name])
                else:
                    sequence.append(20)  # UNK
                residue_indices.append(res_seq)
    
    return {
        'ca_coords': np.array(ca_coords),
        'sequence': np.array(sequence, dtype=np.int64),
        'all_coords': np.array(all_coords),
        'all_atoms': all_atoms,
        'residue_indices': np.array(residue_indices)
    }


def parse_sdf(sdf_path: str):
    """Parse SDF file and extract coordinates and molecular features."""
    mol = Chem.SDMolSupplier(sdf_path)[0]
    if mol is None:
        raise ValueError(f"Could not parse SDF: {sdf_path}")
    
    # Get coordinates
    conf = mol.GetConformer()
    coords = []
    atom_features = []
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
        
        # One-hot atom type
        symbol = atom.GetSymbol()
        atom_type = [1 if t == symbol else 0 for t in ATOM_TYPES]
        if sum(atom_type) == 0:
            atom_type = [0] * len(ATOM_TYPES)
            atom_type[0] = 1  # Default to C
        
        # Additional features
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hybridization = int(atom.GetHybridization())
        aromatic = 1 if atom.GetIsAromatic() else 0
        num_h = atom.GetTotalNumHs()
        
        features = atom_type + [degree, formal_charge, hybridization, aromatic, num_h]
        atom_features.append(features)
    
    coords = np.array(coords)
    atom_features = np.array(atom_features, dtype=np.float32)
    
    # Build edge index from bonds
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = int(bond.GetBondType())
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([bond_type])
        edge_attr.append([bond_type])
    
    edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 1), dtype=np.float32)
    
    # Molecular descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    return {
        'coords': coords,
        'atom_features': atom_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'mol': mol,
        'descriptors': {
            'mw': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'tpsa': tpsa
        }
    }


def build_complex_graph(protein_data: dict, ligand_data: dict, distance_threshold: float = 8.0):
    """
    Build a unified graph for the protein-ligand complex.
    Returns combined coordinates, edge index, and node type labels.
    """
    protein_coords = protein_data['ca_coords']
    ligand_coords = ligand_data['coords']
    
    n_protein = len(protein_coords)
    n_ligand = len(ligand_coords)
    n_total = n_protein + n_ligand
    
    # Combine coordinates
    combined_coords = np.vstack([protein_coords, ligand_coords])
    
    # Node types: 0=protein, 1=ligand
    node_types = np.array([0] * n_protein + [1] * n_ligand)
    
    # Build edges: within-protein (sequential neighbors), within-ligand (bonds), cross-interface
    edges = []
    
    # Protein backbone edges
    for i in range(n_protein - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    
    # Ligand bond edges (offset by n_protein)
    ligand_edges = ligand_data['edge_index']
    for i in range(ligand_edges.shape[1]):
        edges.append([n_protein + ligand_edges[0, i], n_protein + ligand_edges[1, i]])
    
    # Cross-interface edges (within distance threshold)
    for i in range(n_protein):
        for j in range(n_ligand):
            dist = np.linalg.norm(protein_coords[i] - ligand_coords[j])
            if dist < distance_threshold:
                edges.append([i, n_protein + j])
                edges.append([n_protein + j, i])
    
    edge_index = np.array(edges, dtype=np.int64).T
    
    # Edge distances as attributes
    edge_attr = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        dist = np.linalg.norm(combined_coords[src] - combined_coords[dst])
        edge_attr.append([dist])
    edge_attr = np.array(edge_attr, dtype=np.float32)
    
    return {
        'coords': combined_coords,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_types': node_types,
        'n_protein': n_protein,
        'n_ligand': n_ligand
    }


def main():
    os.makedirs('outputs', exist_ok=True)
    
    # Parse data
    protein_data = parse_pdb('data/sample/2l3r/2l3r_protein.pdb')
    ligand_data = parse_sdf('data/sample/2l3r/2l3r_ligand.sdf')
    complex_graph = build_complex_graph(protein_data, ligand_data)
    
    print(f"Protein: {len(protein_data['ca_coords'])} CA atoms, sequence length {len(protein_data['sequence'])}")
    print(f"Ligand: {len(ligand_data['coords'])} atoms")
    print(f"Complex graph: {len(complex_graph['coords'])} nodes, {complex_graph['edge_index'].shape[1]} edges")
    print(f"Ligand descriptors: MW={ligand_data['descriptors']['mw']:.1f}, LogP={ligand_data['descriptors']['logp']:.2f}")
    
    # Save processed data
    np.save('outputs/protein_sequence.npy', protein_data['sequence'])
    np.save('outputs/protein_ca_coords.npy', protein_data['ca_coords'])
    np.save('outputs/ligand_coords.npy', ligand_data['coords'])
    np.save('outputs/ligand_atom_features.npy', ligand_data['atom_features'])
    np.save('outputs/ligand_edge_index.npy', ligand_data['edge_index'])
    np.save('outputs/ligand_edge_attr.npy', ligand_data['edge_attr'])
    np.save('outputs/complex_coords.npy', complex_graph['coords'])
    np.save('outputs/complex_edge_index.npy', complex_graph['edge_index'])
    np.save('outputs/complex_edge_attr.npy', complex_graph['edge_attr'])
    np.save('outputs/complex_node_types.npy', complex_graph['node_types'])
    
    # Save as PyTorch tensors
    torch.save({
        'protein_seq': torch.from_numpy(protein_data['sequence']).long(),
        'protein_coords': torch.from_numpy(protein_data['ca_coords']).float(),
        'ligand_coords': torch.from_numpy(ligand_data['coords']).float(),
        'ligand_features': torch.from_numpy(ligand_data['atom_features']).float(),
        'ligand_edge_index': torch.from_numpy(ligand_data['edge_index']).long(),
        'ligand_edge_attr': torch.from_numpy(ligand_data['edge_attr']).float(),
        'complex_coords': torch.from_numpy(complex_graph['coords']).float(),
        'complex_edge_index': torch.from_numpy(complex_graph['edge_index']).long(),
        'complex_edge_attr': torch.from_numpy(complex_graph['edge_attr']).float(),
        'node_types': torch.from_numpy(complex_graph['node_types']).long(),
        'n_protein': complex_graph['n_protein'],
        'n_ligand': complex_graph['n_ligand'],
        'descriptors': ligand_data['descriptors']
    }, 'outputs/processed_data.pt')
    
    print("Data preparation complete. Saved to outputs/")


if __name__ == '__main__':
    main()
