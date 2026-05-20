"""
Data loading and featurization for molecular property prediction.

Converts SMILES strings to molecular graphs with:
- Atom features (type, degree, formal charge, etc.)
- Bond features (type, conjugation, ring membership, etc.)
- Non-covalent interaction edges (distance-based for 3D conformers)
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# Atom feature definitions
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se', 'Te', 'As', 'Sn', 'other']
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ATOM_FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ATOM_CHIRALITIES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
NUM_HYDROGENS = [0, 1, 2, 3, 4]

# Bond feature definitions
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def get_atom_features(atom):
    """Extract atom features as a dictionary of categorical indices."""
    features = {}
    
    # Atom type
    atom_symbol = atom.GetSymbol()
    features['atom_type'] = ATOM_TYPES.index(atom_symbol) if atom_symbol in ATOM_TYPES else len(ATOM_TYPES) - 1
    
    # Degree
    degree = atom.GetDegree()
    features['degree'] = min(degree, 6)
    
    # Formal charge
    fc = atom.GetFormalCharge()
    features['formal_charge'] = max(-3, min(3, fc)) + 3
    
    # Hybridization
    hyb = atom.GetHybridization()
    features['hybridization'] = ATOM_HYBRIDIZATIONS.index(hyb) if hyb in ATOM_HYBRIDIZATIONS else 0
    
    # Chirality
    chiral = atom.GetChiralTag()
    features['chirality'] = ATOM_CHIRALITIES.index(chiral) if chiral in ATOM_CHIRALITIES else 0
    
    # Number of hydrogens
    num_h = atom.GetTotalNumHs()
    features['num_hydrogens'] = min(num_h, 4)
    
    # Is in ring
    features['is_in_ring'] = 1 if atom.IsInRing() else 0
    
    # Is aromatic
    features['is_aromatic'] = 1 if atom.GetIsAromatic() else 0
    
    # Atomic mass
    features['atomic_mass'] = min(int(atom.GetMass()), 200)
    
    return features


def get_bond_features(bond):
    """Extract bond features as a dictionary of categorical indices."""
    features = {}
    
    # Bond type
    bond_type = bond.GetBondType()
    features['bond_type'] = BOND_TYPES.index(bond_type) if bond_type in BOND_TYPES else 0
    
    # Bond stereo
    stereo = bond.GetStereo()
    features['bond_stereo'] = BOND_STEREOS.index(stereo) if stereo in BOND_STEREOS else 0
    
    # Is conjugated
    features['is_conjugated'] = 1 if bond.GetIsConjugated() else 0
    
    # Is in ring
    features['is_in_ring'] = 1 if bond.IsInRing() else 0
    
    # Is aromatic
    features['is_aromatic'] = 1 if bond.GetIsAromatic() else 0
    
    return features


def smiles_to_graph(smiles, add_non_covalent=True, nc_cutoff=5.0):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        add_non_covalent: Whether to add non-covalent interaction edges
        nc_cutoff: Distance cutoff for non-covalent interactions (Angstroms)
    
    Returns:
        PyG Data object with x (atom features), edge_index, edge_attr (bond features)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 3D coordinates for non-covalent interactions
    if add_non_covalent:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            add_non_covalent = False
    
    # Get atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))
    
    # Build node feature tensor
    num_atoms = len(atom_features_list)
    # Combine all categorical features into a single index
    # We'll use separate indices per feature type
    # For simplicity, use a vocabulary-based encoding
    x_list = []
    for af in atom_features_list:
        # Each atom encoded by concatenating feature indices
        feat_vec = [
            af['atom_type'],
            af['degree'],
            af['formal_charge'],
            af['hybridization'],
            af['chirality'],
            af['num_hydrogens'],
            af['is_in_ring'],
            af['is_aromatic'],
            af['atomic_mass'],
        ]
        x_list.append(feat_vec)
    
    x = torch.tensor(x_list, dtype=torch.long)  # (N, 9)
    
    # Get bond features (covalent)
    edge_index_list = []
    edge_attr_list = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        
        # Add both directions
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        
        feat_vec = [
            bf['bond_type'],
            bf['bond_stereo'],
            bf['is_conjugated'],
            bf['is_in_ring'],
            bf['is_aromatic'],
        ]
        edge_attr_list.append(feat_vec)
        edge_attr_list.append(feat_vec)
    
    # Add non-covalent interactions based on 3D distances
    if add_non_covalent and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                # Skip if already covalently bonded
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    continue
                
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < nc_cutoff:
                    # Non-covalent edge
                    edge_index_list.append([i, j])
                    edge_index_list.append([j, i])
                    
                    # Non-covalent bond features
                    nc_feat = [
                        len(BOND_TYPES),  # special type for non-covalent
                        0,  # no stereo
                        0,  # not conjugated
                        0,  # not in ring
                        0,  # not aromatic
                    ]
                    edge_attr_list.append(nc_feat)
                    edge_attr_list.append(nc_feat)
    
    if len(edge_index_list) == 0:
        # Self-loop for isolated atoms
        edge_index_list.append([0, 0])
        edge_attr_list.append([0, 0, 0, 0, 0])
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_molecule_dataset(csv_path, smiles_col='smiles', label_col='label', 
                          task_type='classification', max_mols=None):
    """
    Load a molecular dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        smiles_col: Column name for SMILES strings
        label_col: Column name for labels
        task_type: 'classification' or 'regression'
        max_mols: Maximum number of molecules to load
    
    Returns:
        List of PyG Data objects, list of labels
    """
    df = pd.read_csv(csv_path)
    
    # Handle multi-task datasets (like ClinTox and MUV)
    if label_col not in df.columns:
        # Try to find label columns
        if 'FDA_APPROVED' in df.columns and 'CT_TOX' in df.columns:
            # ClinTox multi-task
            labels = df[['FDA_APPROVED', 'CT_TOX']].values.astype(float)
        elif 'activity' in df.columns:
            labels = df['label'].values.astype(float) if 'label' in df.columns else None
        else:
            # MUV: multiple binary columns
            label_cols = [c for c in df.columns if c.startswith('MUV-')]
            if label_cols:
                labels = df[label_cols].values.astype(float)
                # Replace empty strings with NaN
                labels = np.where(labels == '', np.nan, labels)
            else:
                labels = df[label_col].values.astype(float) if label_col in df.columns else None
    else:
        labels = df[label_col].values.astype(float)
    
    # Handle NaN labels
    if labels.ndim == 2:
        mask = ~np.isnan(labels).any(axis=1)
    else:
        mask = ~np.isnan(labels)
    
    df = df[mask].reset_index(drop=True)
    if labels.ndim == 2:
        labels = labels[mask]
    else:
        labels = labels[mask]
    
    if max_mols is not None and len(df) > max_mols:
        df = df.iloc[:max_mols]
        labels = labels[:max_mols]
    
    graphs = []
    valid_indices = []
    
    for i, row in df.iterrows():
        smiles = row[smiles_col]
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(i)
    
    if labels.ndim == 2:
        labels = labels[valid_indices]
    else:
        labels = labels[valid_indices]
    
    return graphs, labels, df.iloc[valid_indices]


class MoleculeDataset(Dataset):
    """PyTorch Geometric Dataset for molecular property prediction."""
    
    def __init__(self, graphs, labels, task_type='classification'):
        super().__init__()
        self.graphs = graphs
        self.labels = labels
        self.task_type = task_type
        
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        data = self.graphs[idx].clone()
        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            data.y = torch.tensor(label, dtype=torch.float32)
        else:
            data.y = torch.tensor([label], dtype=torch.float32 if self.task_type == 'regression' else torch.long)
        return data


def create_data_loaders(graphs, labels, batch_size=32, train_ratio=0.8, 
                        val_ratio=0.1, task_type='classification', seed=42):
    """Create train/val/test data loaders."""
    from torch_geometric.loader import DataLoader
    
    # Split indices
    num_data = len(graphs)
    indices = np.arange(num_data)
    
    train_idx, temp_idx = train_test_split(indices, test_size=1 - train_ratio, 
                                           random_state=seed, stratify=labels if task_type == 'classification' else None)
    
    if val_ratio > 0:
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_size,
                                            random_state=seed, 
                                            stratify=labels[temp_idx] if task_type == 'classification' and labels.ndim == 1 else None)
    else:
        val_idx = np.array([])
        test_idx = temp_idx
    
    # Create datasets
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx] if len(val_idx) > 0 else []
    test_graphs = [graphs[i] for i in test_idx]
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx] if len(val_idx) > 0 else np.array([])
    test_labels = labels[test_idx]
    
    train_dataset = MoleculeDataset(train_graphs, train_labels, task_type)
    val_dataset = MoleculeDataset(val_graphs, val_labels, task_type) if len(val_graphs) > 0 else None
    test_dataset = MoleculeDataset(test_graphs, test_labels, task_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx
