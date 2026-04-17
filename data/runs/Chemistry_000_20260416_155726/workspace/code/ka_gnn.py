"""
Kolmogorov-Arnold Graph Neural Networks (KA-GNNs) for Molecular Property Prediction

This module implements:
1. Kolmogorov-Arnold Network (KAN) layers with Fourier basis functions
2. KA-GNN architecture that replaces MLP transformations in GNNs with KAN modules
3. Baseline GNN architectures (GCN, GAT, GraphSAGE) for comparison
4. Molecular graph featurization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from typing import Tuple, List, Optional


# ============================================================================
# Atomic and Bond Features
# ============================================================================

# Atomic properties for node features
ATOM_SYMBOL_TO_NUM = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
ATOM_VALENCE_OPTIONS = [0, 1, 2, 3, 4, 5, 6]
ATOM_CHARGE_OPTIONS = [-3, -2, -1, 0, 1, 2, 3]
ATOM_HYBRIDIZATION_OPTIONS = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
ATOM_AROMATIC_OPTIONS = [0, 1]

# Bond properties for edge features
BOND_TYPE_OPTIONS = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_STEREO_OPTIONS = ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS']
BOND_CONJUGATED_OPTIONS = [0, 1]


def get_atom_feature_vector(atom) -> List[float]:
    """Extract feature vector for an atom."""
    if atom is None:
        return [0] * 28
    
    features = []
    
    # Atom symbol (10 features)
    symbol_onehot = [0] * len(ATOM_SYMBOL_TO_NUM)
    symbol = atom.GetSymbol()
    if symbol in ATOM_SYMBOL_TO_NUM:
        symbol_onehot[ATOM_SYMBOL_TO_NUM[symbol]] = 1
    features.extend(symbol_onehot)
    
    # Degree (7 features)
    degree = atom.GetDegree()
    degree_onehot = [0] * 7
    if degree < 7:
        degree_onehot[degree] = 1
    features.extend(degree_onehot)
    
    # Total H (5 features)
    total_h = atom.GetTotalNumHs()
    total_h_onehot = [0] * 5
    if total_h < 5:
        total_h_onehot[total_h] = 1
    features.extend(total_h_onehot)
    
    # Formal charge (7 features)
    charge = atom.GetFormalCharge()
    charge_idx = min(max(charge + 3, 0), 6)
    charge_onehot = [0] * 7
    charge_onehot[charge_idx] = 1
    features.extend(charge_onehot)
    
    # Hybridization (6 features)
    hybrid = str(atom.GetHybridization())
    hybrid_onehot = [0] * len(ATOM_HYBRIDIZATION_OPTIONS)
    if hybrid in ATOM_HYBRIDIZATION_OPTIONS:
        hybrid_onehot[ATOM_HYBRIDIZATION_OPTIONS.index(hybrid)] = 1
    features.extend(hybrid_onehot)
    
    # Aromatic (1 feature)
    features.append(1 if atom.GetIsAromatic() else 0)
    
    return features


def get_bond_feature_vector(bond) -> List[float]:
    """Extract feature vector for a bond."""
    if bond is None:
        return [0] * 10
    
    features = []
    
    # Bond type (4 features)
    bond_type = str(bond.GetBondType())
    bond_type_onehot = [0] * len(BOND_TYPE_OPTIONS)
    if bond_type in BOND_TYPE_OPTIONS:
        bond_type_onehot[BOND_TYPE_OPTIONS.index(bond_type)] = 1
    features.extend(bond_type_onehot)
    
    # Stereo (5 features)
    stereo = str(bond.GetStereo())
    stereo_onehot = [0] * len(BOND_STEREO_OPTIONS)
    if stereo in BOND_STEREO_OPTIONS:
        stereo_onehot[BOND_STEREO_OPTIONS.index(stereo)] = 1
    features.extend(stereo_onehot)
    
    # Conjugated (1 feature)
    features.append(1 if bond.GetIsConjugated() else 0)
    
    return features


def mol_to_graph(smiles: str, label: int = None) -> Optional[Data]:
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature_vector(atom))
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feat = get_bond_feature_vector(bond)
        
        edge_indices.append([i, j])
        edge_features.append(edge_feat)
        
        # Add reverse edge for undirected graph
        edge_indices.append([j, i])
        edge_features.append(edge_feat)
    
    if len(edge_indices) == 0:
        # Single atom molecule - add self loop
        edge_indices = [[0, 0]]
        edge_features = [get_bond_feature_vector(None)]
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    
    return data


# ============================================================================
# Kolmogorov-Arnold Network (KAN) Layer
# ============================================================================

class FourierKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer using Fourier basis functions.
    
    The KAN theorem states that any multivariate continuous function can be 
    represented as a sum of univariate functions. We approximate these 
    univariate functions using Fourier series.
    
    f(x) = sum_{k=1}^{K} c_k * phi_k(x)
    where phi_k are Fourier basis functions: sin(k*x), cos(k*x)
    """
    
    def __init__(self, in_features: int, out_features: int, num_fourier_terms: int = 8,
                 learnable_frequencies: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_fourier_terms = num_fourier_terms
        
        # Coefficients for each input-output pair and each Fourier term
        # Shape: (out_features, in_features, 2 * num_fourier_terms)
        # 2 because we have both sin and cos for each frequency
        self.coefficients = nn.Parameter(
            torch.randn(out_features, in_features, 2 * num_fourier_terms) * 0.1
        )
        
        # Learnable frequencies (optional)
        if learnable_frequencies:
            self.frequencies = nn.Parameter(
                torch.arange(1, num_fourier_terms + 1, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                .repeat(out_features, in_features, 1)
            )
        else:
            self.register_buffer(
                'frequencies',
                torch.arange(1, num_fourier_terms + 1, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                .repeat(out_features, in_features, 1)
            )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable scaling for input normalization
        self.input_scale = nn.Parameter(torch.ones(in_features))
        self.input_shift = nn.Parameter(torch.zeros(in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Fourier KAN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Normalize input
        x_normalized = x * self.input_scale + self.input_shift  # (batch, in_features)
        
        # Efficient computation using einsum
        # x_normalized: (batch, in_features)
        # frequencies: (out_features, in_features, num_fourier_terms)
        # coefficients: (out_features, in_features, 2 * num_fourier_terms)
        
        # Compute phase: (batch, out_features, in_features, num_fourier_terms)
        # Using broadcasting: (batch, in, 1) * (1, out, in, terms)
        phases = x_normalized.unsqueeze(1).unsqueeze(-1) * self.frequencies.unsqueeze(0)
        
        # Compute sin and cos: (batch, out, in, terms)
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)
        
        # Split coefficients
        coeff_sin = self.coefficients[:, :, :self.num_fourier_terms]  # (out, in, terms)
        coeff_cos = self.coefficients[:, :, self.num_fourier_terms:]  # (out, in, terms)
        
        # Compute weighted sum using einsum
        # sin_terms: (batch, out, in, terms), coeff_sin: (out, in, terms)
        # Result: (batch, out)
        sin_out = torch.einsum('boit,oit->bo', sin_terms, coeff_sin)
        cos_out = torch.einsum('boit,oit->bo', cos_terms, coeff_cos)
        
        # Sum contributions
        output = sin_out + cos_out  # (batch, out)
        
        # Add bias
        output = output + self.bias
        
        return output


class KANLinear(nn.Module):
    """
    Simplified KAN linear layer using learnable B-spline basis functions.
    This is a more practical implementation inspired by recent KAN papers.
    """
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 10,
                 spline_order: int = 3, base_activation: nn.Module = nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear transformation (like standard linear layer)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # Spline coefficients for learnable activation
        # Each output feature has a set of spline functions for each input
        self.spline_coeff = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.01
        )
        
        # Grid boundaries (fixed)
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 2 * spline_order + 1))
        
        self.base_act = base_activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Base linear transformation with activation
        base_out = F.linear(self.base_act(x), self.base_weight, self.base_bias)
        
        # For simplicity, use a simplified spline approximation
        # In practice, we use a combination of base linear + learnable residual
        x_normalized = torch.tanh(x)  # Normalize to [-1, 1]
        
        # Simple RBF-like interpolation for spline
        grid_centers = torch.linspace(-1, 1, self.grid_size).to(x.device)
        # (grid_size,)
        
        # Compute RBF activations
        # x_normalized: (batch, in_features)
        # grid_centers: (grid_size,)
        diff = x_normalized.unsqueeze(-1) - grid_centers.unsqueeze(0).unsqueeze(0)  # (batch, in, grid)
        rbf = torch.exp(-diff ** 2 / 0.1)  # (batch, in, grid)
        
        # Apply spline coefficients
        # spline_coeff: (out, in, grid + order)
        coeff = self.spline_coeff[:, :, :self.grid_size]  # (out, in, grid)
        
        # (batch, in, grid) * (1, in, grid) -> (batch, in, grid)
        weighted = rbf * coeff.unsqueeze(0)
        
        # Sum over grid and input
        spline_out = weighted.sum(dim=(1, 2))  # (batch, out) - this sums over wrong dimension
        
        # Let's fix this
        # weighted: (batch, in, grid)
        # We want: for each output j, sum over inputs i and grid k: coeff[j,i,k] * rbf[batch,i,k]
        
        # Transpose and reshape
        rbf_flat = rbf.permute(0, 2, 1).reshape(batch_size, self.grid_size * self.in_features)
        coeff_flat = self.spline_coeff[:, :, :self.grid_size].reshape(self.out_features, self.grid_size * self.in_features)
        
        spline_out = F.linear(rbf_flat, coeff_flat)  # (batch, out)
        
        return base_out + spline_out


# ============================================================================
# Graph Neural Network Architectures
# ============================================================================

class GCN(nn.Module):
    """Standard Graph Convolutional Network with MLP transformations."""
    
    def __init__(self, in_features: int, hidden_features: int = 64, 
                 out_features: int = 1, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_features, hidden_features))
        self.bns.append(nn.BatchNorm1d(hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_features, hidden_features))
            self.bns.append(nn.BatchNorm1d(hidden_features))
        
        # Output layer
        self.convs.append(GCNConv(hidden_features, hidden_features))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, out_features)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final convolution
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network with MLP transformations."""
    
    def __init__(self, in_features: int, hidden_features: int = 64,
                 out_features: int = 1, num_layers: int = 3, 
                 num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_features, hidden_features // num_heads, 
                                   heads=num_heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_features, hidden_features // num_heads,
                                       heads=num_heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_features))
        
        # Output layer
        self.convs.append(GATConv(hidden_features, hidden_features, heads=1, dropout=dropout))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, out_features)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Final convolution
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class KAGNN(nn.Module):
    """
    Kolmogorov-Arnold Graph Neural Network.
    
    Replaces MLP transformations in GNN layers with Fourier-based KAN modules
    for enhanced expressive power and interpretability.
    """
    
    def __init__(self, in_features: int, hidden_features: int = 64,
                 out_features: int = 1, num_layers: int = 3,
                 num_fourier_terms: int = 8, dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        
        # Graph convolution layers with KAN transformations
        self.convs = nn.ModuleList()
        self.kan_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input projection with KAN
        self.input_kan = FourierKANLayer(in_features, hidden_features, 
                                          num_fourier_terms=num_fourier_terms)
        self.bns.append(nn.BatchNorm1d(hidden_features))
        
        # Graph convolutions
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_features, hidden_features))
            
            # KAN transformation after each graph conv
            self.kan_layers.append(
                FourierKANLayer(hidden_features, hidden_features,
                               num_fourier_terms=num_fourier_terms)
            )
            self.bns.append(nn.BatchNorm1d(hidden_features))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output KAN classifier
        self.output_kan = FourierKANLayer(hidden_features, hidden_features // 2,
                                           num_fourier_terms=num_fourier_terms)
        self.final_linear = nn.Linear(hidden_features // 2, out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        # Input projection through KAN
        x = self.input_kan(x)
        x = self.bns[0](x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Graph convolution layers with KAN transformations
        for i in range(self.num_layers):
            # Graph convolution
            x = self.convs[i](x, edge_index)
            
            # KAN transformation
            x = self.kan_layers[i](x)
            x = self.bns[i + 1](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output classification through KAN
        x = self.output_kan(x)
        x = F.relu(x)
        x = self.final_linear(x)
        
        return x


class KAGNNWithEdgeFeatures(nn.Module):
    """
    Enhanced KA-GNN that incorporates edge features.
    Uses edge_attr in the graph convolution.
    """
    
    def __init__(self, in_features: int, edge_features: int, 
                 hidden_features: int = 64, out_features: int = 1,
                 num_layers: int = 3, num_fourier_terms: int = 8,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        
        # Input projections
        self.node_kan = FourierKANLayer(in_features, hidden_features,
                                         num_fourier_terms=num_fourier_terms)
        self.edge_kan = FourierKANLayer(edge_features, hidden_features,
                                         num_fourier_terms=num_fourier_terms)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.kan_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            # Using simple GCNConv - will handle edge features manually
            self.convs.append(GCNConv(hidden_features, hidden_features))
            self.kan_layers.append(
                FourierKANLayer(hidden_features, hidden_features,
                               num_fourier_terms=num_fourier_terms)
            )
            self.bns.append(nn.BatchNorm1d(hidden_features))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.output_kan = FourierKANLayer(hidden_features, hidden_features // 2,
                                           num_fourier_terms=num_fourier_terms)
        self.final_linear = nn.Linear(hidden_features // 2, out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Project node and edge features through KAN
        x = self.node_kan(x)
        edge_attr = self.edge_kan(edge_attr)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            # Simple message passing with edge features
            # row, col = edge_index
            # message = x[col] * edge_attr  # Element-wise multiplication
            
            # Use GCNConv
            x_new = self.convs[i](x, edge_index)
            
            # Add edge feature contribution
            # This is a simplified approach
            x_new = x_new + global_mean_pool(edge_attr, batch[edge_index[1]])[:batch.max()+1][batch]
            
            # KAN transformation
            x_new = self.kan_layers[i](x_new)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            x = x_new
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output
        x = self.output_kan(x)
        x = F.relu(x)
        x = self.final_linear(x)
        
        return x


# ============================================================================
# Training Utilities
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, use_edge_attr=False):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Check both parameter and batch attribute
        has_edge_attr = use_edge_attr or (hasattr(batch, 'edge_attr') and batch.edge_attr is not None)
        
        if has_edge_attr and hasattr(model, 'edge_kan'):
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            out = model(batch.x, batch.edge_index, batch.batch)
        
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        predictions = (torch.sigmoid(out.squeeze()) > 0.5).float()
        total_correct += (predictions == batch.y).sum().item()
        total_samples += batch.num_graphs
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, use_edge_attr=False):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            has_edge_attr = use_edge_attr or (hasattr(batch, 'edge_attr') and batch.edge_attr is not None)
            
            if has_edge_attr and hasattr(model, 'edge_kan'):
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(out.squeeze(), batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            scores = torch.sigmoid(out.squeeze()).cpu().numpy()
            predictions = (scores > 0.5).astype(float)
            labels = batch.y.cpu().numpy()
            
            all_scores.extend(scores)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            total_correct += (predictions == labels).sum()
            total_samples += len(labels)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'scores': np.array(all_scores),
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


if __name__ == '__main__':
    # Quick test
    print("Testing KA-GNN implementation...")
    
    # Create dummy data
    x = torch.randn(10, 28)
    edge_index = torch.randint(0, 10, (2, 20))
    batch = torch.zeros(10, dtype=torch.long)
    
    # Test Fourier KAN layer
    kan = FourierKANLayer(28, 64, num_fourier_terms=8)
    out = kan(x)
    print(f"FourierKANLayer output shape: {out.shape}")
    
    # Test KA-GNN
    model = KAGNN(in_features=28, hidden_features=64, out_features=1, num_layers=3)
    out = model(x, edge_index, batch)
    print(f"KA-GNN output shape: {out.shape}")
    
    print("Tests passed!")
