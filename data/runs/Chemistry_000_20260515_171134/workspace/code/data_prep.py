#!/usr/bin/env python3
"""Data preparation utilities for KA-GNN molecular property prediction."""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Dict, Any


# Atomic number to feature mapping (common atoms in molecules)
ATOM_FEATURES = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    15: 5,  # P
    16: 6,  # S
    17: 7,  # Cl
    35: 8,  # Br
    53: 9,  # I
}

BOND_FEATURES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features for graph node."""
    atomic_num = atom.GetAtomicNum()
    atom_type = ATOM_FEATURES.get(atomic_num, len(ATOM_FEATURES))
    
    # One-hot encoding for atom type (10 common + 1 unknown)
    atom_type_onehot = np.zeros(len(ATOM_FEATURES) + 1)
    atom_type_onehot[atom_type] = 1.0
    
    # Additional features
    degree = atom.GetDegree()
    degree_onehot = np.zeros(6)
    if degree < 6:
        degree_onehot[degree] = 1.0
    
    formal_charge = atom.GetFormalCharge()
    charge_feat = np.array([formal_charge / 4.0])  # Normalize
    
    hybridization = atom.GetHybridization()
    hyb_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }
    hyb_feat = np.zeros(5)
    hyb_feat[hyb_map.get(hybridization, 0)] = 1.0
    
    # Aromaticity
    aromatic = np.array([float(atom.GetIsAromatic())])
    
    # Hydrogen count
    num_h = atom.GetTotalNumHs()
    h_feat = np.zeros(5)
    if num_h < 5:
        h_feat[num_h] = 1.0
    
    return np.concatenate([
        atom_type_onehot,
        degree_onehot,
        charge_feat,
        hyb_feat,
        aromatic,
        h_feat
    ]).astype(np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features for graph edge."""
    bond_type = bond.GetBondType()
    bond_type_idx = BOND_FEATURES.get(bond_type, 0)
    
    # One-hot encoding for bond type
    bond_type_onehot = np.zeros(len(BOND_FEATURES))
    bond_type_onehot[bond_type_idx] = 1.0
    
    # Additional features
    is_conjugated = float(bond.GetIsConjugated())
    is_aromatic = float(bond.GetIsAromatic())
    stereo = bond.GetStereo()
    stereo_feat = np.zeros(4)
    stereo_map = {
        Chem.BondStereo.STEREONONE: 0,
        Chem.BondStereo.STEREOE: 1,
        Chem.BondStereo.STEREOZ: 2,
        Chem.BondStereo.STEREOCIS: 3,
    }
    stereo_feat[stereo_map.get(stereo, 0)] = 1.0
    
    return np.concatenate([
        bond_type_onehot,
        np.array([is_conjugated, is_aromatic]),
        stereo_feat
    ]).astype(np.float32)


def smiles_to_graph(smiles: str, label: Optional[np.ndarray] = None) -> Optional[Data]:
    """Convert SMILES string to PyTorch Geometric Data object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Atom features
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(atom_features(atom))
        x = torch.tensor(np.array(atom_feats), dtype=torch.float)
        
        # Edge indices and features
        edge_index = []
        edge_attr = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            feat = bond_features(bond)
            edge_attr.append(feat)
            edge_attr.append(feat)
        
        if len(edge_index) == 0:
            # Single atom molecule - create self-loop
            edge_index = [[0, 0]]
            edge_attr = [np.zeros(9, dtype=np.float32)]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if label is not None:
            data.y = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        
        return data
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def load_dataset(csv_path: str, task_name: str = None) -> Tuple[List[Data], List[int]]:
    """Load molecular dataset from CSV and convert to graph format.
    
    Returns:
        graphs: List of PyTorch Geometric Data objects
        indices: Original indices for tracking
    """
    df = pd.read_csv(csv_path)
    
    # Determine label column(s) based on dataset
    if 'label' in df.columns:
        label_cols = ['label']
    elif 'activity' in df.columns:
        label_cols = ['activity']
    elif 'FDA_APPROVED' in df.columns and 'CT_TOX' in df.columns:
        label_cols = ['FDA_APPROVED', 'CT_TOX']
    elif any(col.startswith('MUV-') for col in df.columns):
        label_cols = [col for col in df.columns if col.startswith('MUV-')]
    else:
        # Fallback: last column(s) as label
        label_cols = df.columns[-1:]
    
    graphs = []
    indices = []
    
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col])
        
        # Extract label(s)
        if len(label_cols) == 1:
            label = np.array([row[label_cols[0]]], dtype=np.float32)
        else:
            label = np.array([row[col] for col in label_cols], dtype=np.float32)
        
        # Convert to graph
        graph = smiles_to_graph(smiles, label)
        
        if graph is not None:
            graphs.append(graph)
            indices.append(idx)
    
    print(f"Loaded {len(graphs)} valid graphs from {len(df)} molecules")
    return graphs, indices


def create_dataloaders(graphs: List[Data], batch_size: int = 32, 
                       train_ratio: float = 0.8, val_ratio: float = 0.1,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split graphs into train/val/test and create DataLoaders."""
    np.random.seed(seed)
    n = len(graphs)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_graphs = [graphs[i] for i in indices[:train_end]]
    val_graphs = [graphs[i] for i in indices[train_end:val_end]]
    test_graphs = [graphs[i] for i in indices[val_end:]]
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    data_dir = "data"
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            print(f"\nTesting {fname}...")
            graphs, idx = load_dataset(os.path.join(data_dir, fname))
            if graphs:
                loader, _, _ = create_dataloaders(graphs, batch_size=4)
                batch = next(iter(loader))
                print(f"Batch: {batch}")
                print(f"Node features shape: {batch.x.shape}")
                print(f"Edge index shape: {batch.edge_index.shape}")