"""
Kolmogorov-Arnold Graph Neural Networks (KA-GNNs) for Molecular Property Prediction.

This module implements:
1. Fourier-KAN Layer: replaces conventional MLP with Fourier basis function expansions
2. KA-GNN: Graph neural network using KAN layers for message passing
3. GCN Baseline: Standard graph convolutional network with MLP layers
4. Molecular featurization from SMILES strings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


# ============================================================
# Atom and Bond Featurization
# ============================================================

ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'As']
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
NUM_HYDROGENS = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
HYBRIDIZATIONS = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    from rdkit import Chem
    features = []
    features.extend(one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOLS + ['Unknown']))
    features.extend(one_of_k_encoding(atom.GetDegree(), ATOM_DEGREES + ['Unknown']))
    features.extend(one_of_k_encoding(atom.GetTotalNumHs(), NUM_HYDROGENS + ['Unknown']))
    features.extend(one_of_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGES + ['Unknown']))
    features.extend(one_of_k_encoding(str(atom.GetHybridization()), HYBRIDIZATIONS + ['Unknown']))
    features.append(atom.GetIsAromatic())
    features.append(atom.GetMass() * 0.01)
    return torch.tensor(features, dtype=torch.float)

BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_STEREO = ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE']
BOND_DIRS = ['NONE', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT', 'ENDUPRIGHT']

def get_bond_features(bond):
    features = []
    bt = bond.GetBondType()
    features.extend(one_of_k_encoding(str(bt), BOND_TYPES))
    features.extend(one_of_k_encoding(str(bond.GetStereo()), BOND_STEREO))
    features.append(bond.GetIsConjugated())
    features.append(bond.IsInRing())
    features.extend(one_of_k_encoding(str(bond.GetBondDir()), BOND_DIRS))
    return torch.tensor(features, dtype=torch.float)


# ============================================================
# Fourier-KAN Layer (Improved)
# ============================================================

class FourierKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer using Fourier basis functions.
    
    Per the Kolmogorov-Arnold representation theorem, any multivariate continuous
    function can be represented as a composition of univariate functions. We approximate
    these univariate functions using Fourier series:
    
        f(x) = sum_{k=1}^{G} [a_k * sin(2*pi*k*x) + b_k * cos(2*pi*k*x)]
    
    This provides stronger expressive power than fixed activation MLPs.
    """
    def __init__(self, in_features, out_features, grid_size=5, inner_dim=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        if inner_dim is None:
            inner_dim = min(in_features, out_features, 32)
        self.inner_dim = inner_dim
        
        # Project to inner dim
        self.proj_in = nn.Linear(in_features, inner_dim)
        
        # Fourier coefficients: [out_features, inner_dim, 2*grid_size]
        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_features, inner_dim, 2 * grid_size) * 0.01
        )
        
        # Base linear transformation (residual path)
        self.base_weight = nn.Parameter(torch.randn(out_features, inner_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable temperature for scaling input to Fourier basis
        self.temperature = nn.Parameter(torch.ones(inner_dim) * 0.5)
        
        self._freqs = None
    
    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        
        # Project to inner dimension with SiLU
        z = F.silu(self.proj_in(x))  # [N, inner_dim]
        
        # Scale by learnable temperature
        z_scaled = z * self.temperature
        
        # Base linear
        base_out = z @ self.base_weight.T  # [N, out_features]
        
        # Fourier basis
        if self._freqs is None or self._freqs.device != x.device:
            self._freqs = torch.arange(1, self.grid_size + 1, device=x.device).float()
        
        freqs = self._freqs
        z_expanded = z_scaled.unsqueeze(1)  # [N, 1, inner_dim]
        kz = z_expanded * freqs.view(1, -1, 1)  # [N, G, inner_dim]
        sin_kz = torch.sin(kz)
        cos_kz = torch.cos(kz)
        fourier_basis = torch.cat([sin_kz, cos_kz], dim=1)  # [N, 2G, inner_dim]
        
        # Apply coefficients: fourier_basis [N, 2G, inner], coeffs [out, inner, 2G]
        fourier_out = torch.einsum('ngi,oig->no', fourier_basis, self.fourier_coeffs)
        
        # Combine with residual scaling factor
        alpha = 0.5  # Blend between base and fourier
        out = (1 - alpha) * base_out + alpha * fourier_out + self.bias
        return out.reshape(*original_shape, self.out_features)


# ============================================================
# Message Passing Layers
# ============================================================

class KANConv(MessagePassing):
    """
    Graph convolution layer using Fourier-KAN for message transformation.
    Replaces the standard linear transform in GCN with a KAN layer.
    """
    def __init__(self, in_channels, out_channels, edge_dim=None, grid_size=5, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        msg_dim = in_channels + (edge_dim if edge_dim else 0)
        self.kan = FourierKANLayer(msg_dim, out_channels, grid_size=grid_size)
        self.root_lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        num_edges_before = edge_index.size(1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        num_edges_after = edge_index.size(1)
        if edge_attr is not None and num_edges_after > num_edges_before:
            n_self = num_edges_after - num_edges_before
            zero_edges = torch.zeros(n_self, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, zero_edges], dim=0)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        root = self.root_lin(x)
        out = out + root
        out = self.norm(out)
        return out
    
    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            msg_input = torch.cat([x_j, edge_attr], dim=-1)
        else:
            msg_input = x_j
        return self.kan(msg_input)


class GCNConvBaseline(nn.Module):
    """Standard GCN layer with linear transformation (baseline)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x_j = self.lin(x)
        msg = norm.view(-1, 1) * x_j[col]
        
        out = torch.zeros_like(x_j)
        out.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)
        out = out + self.lin(x)
        out = self.norm(out)
        return out


# ============================================================
# Full Model Architectures
# ============================================================

class KAGNN(nn.Module):
    """
    Kolmogorov-Arnold Graph Neural Network for molecular property prediction.
    
    Architecture:
    - Node embedding via linear projection
    - Multiple KAN-Conv layers for message passing
    - Global pooling (mean + add)
    - KAN-based readout MLP for final prediction
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=128, num_layers=3, 
                 grid_size=5, dropout=0.2, num_tasks=1):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim // 2)
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                KANConv(hidden_dim, hidden_dim, edge_dim=hidden_dim // 2, grid_size=grid_size)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        concat_dim = hidden_dim * 2
        self.readout = nn.Sequential(
            FourierKANLayer(concat_dim, hidden_dim, grid_size=grid_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            FourierKANLayer(hidden_dim, hidden_dim // 2, grid_size=grid_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.silu(self.node_encoder(x))
        if edge_attr is not None:
            edge_attr = F.silu(self.edge_encoder(edge_attr))
        
        for i, conv in enumerate(self.conv_layers):
            x_before = x
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.silu(x)
            x = self.dropout(x)
            x = x + x_before
        
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_graph = torch.cat([x_mean, x_add], dim=-1)
        
        out = self.readout(x_graph)
        return out


class GCNBaseline(nn.Module):
    """
    Standard GCN baseline with MLP transformations.
    Same architecture as KAGNN but uses linear/MLP instead of KAN layers.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=128, num_layers=3,
                 dropout=0.2, num_tasks=1):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim // 2)
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(GCNConvBaseline(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        concat_dim = hidden_dim * 2
        self.readout = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.relu(self.node_encoder(x))
        if edge_attr is not None:
            edge_attr = F.relu(self.edge_encoder(edge_attr))
        
        for i, conv in enumerate(self.conv_layers):
            x_before = x
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_before
        
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_graph = torch.cat([x_mean, x_add], dim=-1)
        
        out = self.readout(x_graph)
        return out


# ============================================================
# Molecular Graph Conversion
# ============================================================

from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    x = torch.stack(node_features)
    
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_indices.append([i, j])
        edge_features.append(bf)
        edge_indices.append([j, i])
        edge_features.append(bf)
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        ref_mol = Chem.MolFromSmiles('CC')
        ref_mol = Chem.AddHs(ref_mol)
        edge_attr = torch.empty((0, len(get_bond_features(ref_mol.GetBondWithIdx(0)))))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================
# Scaffold Splitting
# ============================================================

from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def scaffold_split(smiles_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split molecules by scaffold to ensure structural diversity across splits."""
    np.random.seed(seed)
    
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except:
            scaffold = "unknown"
        scaffolds[scaffold].append(idx)
    
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    
    train_idx, val_idx, test_idx = [], [], []
    for scaffold_group in scaffold_sets:
        r = np.random.random()
        if len(train_idx) / max(len(smiles_list), 1) < train_ratio:
            train_idx.extend(scaffold_group)
        elif len(val_idx) / max(len(smiles_list), 1) < val_ratio:
            val_idx.extend(scaffold_group)
        else:
            test_idx.extend(scaffold_group)
    
    return train_idx, val_idx, test_idx
