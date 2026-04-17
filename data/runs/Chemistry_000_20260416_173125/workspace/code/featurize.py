"""
Molecular Graph Featurization using RDKit
Converts SMILES strings to PyG Data objects with atom and bond features.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch_geometric.data import Data

# Atom feature dimensions
ATOM_FEATURES = {
    'atom_type': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Na', 'K', 'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Other'],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'num_h': [0, 1, 2, 3, 4],
    'chirality': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}

def one_hot(val, allowed_set):
    """One-hot encoding with unknown category."""
    encoding = [0] * (len(allowed_set) + 1)
    if val in allowed_set:
        encoding[allowed_set.index(val)] = 1
    else:
        encoding[-1] = 1  # unknown
    return encoding

def get_atom_features(atom):
    """Extract atom-level features."""
    features = []
    features += one_hot(atom.GetSymbol(), ATOM_FEATURES['atom_type'])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_h'])
    features += one_hot(atom.GetChiralTag(), ATOM_FEATURES['chirality'])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.GetMass() / 100.0)  # normalized mass
    return features

def get_bond_features(bond):
    """Extract bond-level features."""
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features

def smiles_to_graph(smiles, y=None):
    """Convert a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Bond features and edge index
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)
    
    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    
    return data

def get_atom_feature_dim():
    """Return the total dimension of atom features."""
    dim = 0
    dim += len(ATOM_FEATURES['atom_type']) + 1  # +1 for unknown
    dim += len(ATOM_FEATURES['degree']) + 1
    dim += len(ATOM_FEATURES['formal_charge']) + 1
    dim += len(ATOM_FEATURES['hybridization']) + 1
    dim += len(ATOM_FEATURES['num_h']) + 1
    dim += len(ATOM_FEATURES['chirality']) + 1
    dim += 1  # aromatic
    dim += 1  # mass
    return dim

def get_bond_feature_dim():
    """Return the total dimension of bond features."""
    dim = 0
    dim += len(BOND_FEATURES['bond_type']) + 1
    dim += len(BOND_FEATURES['stereo']) + 1
    dim += 1  # conjugated
    dim += 1  # in ring
    return dim

if __name__ == '__main__':
    print(f"Atom feature dim: {get_atom_feature_dim()}")
    print(f"Bond feature dim: {get_bond_feature_dim()}")
    # Test
    data = smiles_to_graph('CCO', 1.0)
    print(f"Test graph: {data}")
