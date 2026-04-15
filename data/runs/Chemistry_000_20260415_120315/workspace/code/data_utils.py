"""
Data utilities for molecular property prediction.
Handles loading datasets, converting SMILES to graphs, and feature extraction.
"""

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch.utils.data import Dataset, DataLoader
import os

# Atom feature extraction
def get_atom_features(atom):
    """Extract features for an atom."""
    features = []
    # Atomic number
    features.append(atom.GetAtomicNum())
    # Degree
    features.append(atom.GetDegree())
    # Formal charge
    features.append(atom.GetFormalCharge())
    # Hybridization
    features.append(atom.GetHybridization())
    # Aromatic
    features.append(int(atom.GetIsAromatic()))
    # Num hydrogens
    features.append(atom.GetTotalNumHs())
    # In ring
    features.append(int(atom.IsInRing()))
    return features

# Bond feature extraction
def get_bond_features(bond):
    """Extract features for a bond."""
    features = []
    # Bond type
    bond_type = bond.GetBondType()
    features.append(int(bond_type == Chem.BondType.SINGLE))
    features.append(int(bond_type == Chem.BondType.DOUBLE))
    features.append(int(bond_type == Chem.BondType.TRIPLE))
    features.append(int(bond_type == Chem.BondType.AROMATIC))
    # Conjugated
    features.append(int(bond.GetIsConjugated()))
    # In ring
    features.append(int(bond.IsInRing()))
    # Stereo
    features.append(int(bond.GetStereo()))
    return features

class MoleculeGraph:
    """Represents a molecule as a graph."""
    def __init__(self, smiles, label):
        self.smiles = smiles
        self.label = label
        self.mol = Chem.MolFromSmiles(smiles)
        
        if self.mol is None:
            self.valid = False
            return
        
        self.valid = True
        self.atom_features = []
        self.bond_features = []
        self.edge_index = []
        
        # Extract atom features
        for atom in self.mol.GetAtoms():
            self.atom_features.append(get_atom_features(atom))
        
        # Extract bond features and edge indices
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            self.edge_index.append([i, j])
            self.edge_index.append([j, i])  # Undirected
            bond_feat = get_bond_features(bond)
            self.bond_features.append(bond_feat)
            self.bond_features.append(bond_feat)
        
        self.atom_features = np.array(self.atom_features, dtype=np.float32)
        self.bond_features = np.array(self.bond_features, dtype=np.float32)
        self.edge_index = np.array(self.edge_index, dtype=np.int64).T

class MoleculeDataset(Dataset):
    """PyTorch dataset for molecular graphs."""
    def __init__(self, graphs):
        self.graphs = [g for g in graphs if g.valid]
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            'atom_features': torch.FloatTensor(g.atom_features),
            'bond_features': torch.FloatTensor(g.bond_features) if len(g.bond_features) > 0 else torch.zeros((0, 7)),
            'edge_index': torch.LongTensor(g.edge_index),
            'num_atoms': g.atom_features.shape[0],
            'label': torch.FloatTensor([g.label]),
            'smiles': g.smiles
        }

def collate_fn(batch):
    """Collate function for batching molecular graphs."""
    # Separate features
    atom_features_list = [item['atom_features'] for item in batch]
    bond_features_list = [item['bond_features'] for item in batch]
    edge_index_list = [item['edge_index'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Calculate cumulative atom counts for edge index adjustment
    num_atoms_list = [item['num_atoms'] for item in batch]
    cumsum_atoms = np.cumsum([0] + num_atoms_list[:-1])
    
    # Adjust edge indices
    adjusted_edge_indices = []
    for i, edge_index in enumerate(edge_index_list):
        adjusted_edge_index = edge_index + cumsum_atoms[i]
        adjusted_edge_indices.append(adjusted_edge_index)
    
    # Concatenate all features
    atom_features = torch.cat(atom_features_list, dim=0)
    bond_features = torch.cat(bond_features_list, dim=0) if len(bond_features_list[0]) > 0 else torch.zeros((0, 7))
    edge_index = torch.cat(adjusted_edge_indices, dim=1) if len(adjusted_edge_indices) > 0 else torch.zeros((2, 0), dtype=torch.long)
    
    # Create batch assignment
    batch_assign = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_atoms_list)])
    
    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'edge_index': edge_index,
        'batch': batch_assign,
        'num_graphs': len(batch),
        'labels': labels
    }

def load_dataset(dataset_name, data_dir='data'):
    """Load a dataset by name."""
    filepath = os.path.join(data_dir, f'{dataset_name}.csv')
    
    if dataset_name == 'bace':
        df = pd.read_csv(filepath)
        smiles_col = 'smiles'
        label_col = 'label'
    elif dataset_name == 'bbbp':
        df = pd.read_csv(filepath)
        smiles_col = 'smiles'
        label_col = 'label'
    elif dataset_name == 'clintox':
        df = pd.read_csv(filepath)
        smiles_col = 'smiles'
        label_cols = ['FDA_APPROVED', 'CT_TOX']
        # Use both labels
        df = df.dropna(subset=label_cols)
        df['label'] = df[label_cols].values.tolist()
        smiles_col = 'smiles'
        label_col = 'label'
    elif dataset_name == 'hiv':
        df = pd.read_csv(filepath)
        smiles_col = 'smiles'
        label_col = 'label'
    elif dataset_name == 'muv':
        df = pd.read_csv(filepath)
        smiles_col = 'smiles'
        # MUV has multiple tasks, use the first non-null
        task_cols = [c for c in df.columns if c.startswith('MUV')]
        df = df.dropna(subset=task_cols, how='all')
        # Use first task for simplicity
        df['label'] = df[task_cols[0]].fillna(0)
        smiles_col = 'smiles'
        label_col = 'label'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create molecular graphs
    graphs = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[label_col]
        if isinstance(label, list):
            label = [float(l) for l in label]
        else:
            label = float(label)
        
        graph = MoleculeGraph(smiles, label)
        if graph.valid:
            graphs.append(graph)
    
    return graphs, df

def get_dataloader(dataset_name, batch_size=32, data_dir='data'):
    """Get a DataLoader for a dataset."""
    graphs, df = load_dataset(dataset_name, data_dir)
    dataset = MoleculeDataset(graphs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader, dataset, graphs, df

# Test the data loading
if __name__ == '__main__':
    print("Testing data loading...")
    for dataset_name in ['bace', 'bbbp', 'clintox', 'hiv', 'muv']:
        try:
            loader, dataset, graphs, df = get_dataloader(dataset_name, batch_size=16)
            batch = next(iter(loader))
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total graphs: {len(graphs)}")
            print(f"  Valid graphs: {len(dataset)}")
            print(f"  Atom features shape: {batch['atom_features'].shape}")
            print(f"  Edge index shape: {batch['edge_index'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
        except Exception as e:
            print(f"\n{dataset_name.upper()}: Error - {e}")
