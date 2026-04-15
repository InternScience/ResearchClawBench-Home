"""
Kolmogorov-Arnold Graph Neural Network (KA-GNN) implementation.

The Kolmogorov-Arnold representation theorem states that any multivariate continuous function
can be represented as a superposition of continuous functions of one variable:
f(x1, x2, ..., xn) = Σ g_i(φ_i(x_i))

This implementation uses Fourier basis functions for the learnable activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FourierKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer using Fourier basis functions.
    
    Instead of using fixed activation functions (like ReLU) in MLPs,
    we use learnable activation functions represented as Fourier series:
    φ(x) = a0 + Σ [a_k * cos(kωx) + b_k * sin(kωx)]
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_frequencies: Number of Fourier basis functions (degree)
        omega: Base frequency for Fourier basis
    """
    def __init__(self, in_features, out_features, num_frequencies=8, omega=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.omega = omega
        
        # Learnable coefficients for Fourier basis
        # Shape: (out_features, in_features, num_frequencies, 2)
        # Last dimension: [cos_coeff, sin_coeff]
        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, num_frequencies, 2) * 0.1
        )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x):
        """
        Forward pass using Fourier basis activation.
        
        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        # x shape: (batch, in_features)
        batch_size = x.shape[0]
        
        # Expand x for broadcasting: (batch, 1, in_features, 1)
        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_features, 1)
        
        # Frequencies: (1, 1, 1, num_frequencies)
        k = torch.arange(1, self.num_frequencies + 1, device=x.device).float()
        k = k.view(1, 1, 1, -1) * self.omega
        
        # Compute arguments: k * x
        args = k * x_expanded  # (batch, 1, in_features, num_frequencies)
        
        # Compute Fourier basis
        cos_basis = torch.cos(args)  # (batch, 1, in_features, num_frequencies)
        sin_basis = torch.sin(args)  # (batch, 1, in_features, num_frequencies)
        
        # Apply coefficients
        # fourier_coeffs: (out_features, in_features, num_frequencies, 2)
        cos_coeffs = self.fourier_coeffs[..., 0]  # (out_features, in_features, num_frequencies)
        sin_coeffs = self.fourier_coeffs[..., 1]  # (out_features, in_features, num_frequencies)
        
        # Weighted sum over frequencies and inputs
        cos_contrib = (cos_basis * cos_coeffs.unsqueeze(0)).sum(dim=(-2, -1))  # (batch, out_features)
        sin_contrib = (sin_basis * sin_coeffs.unsqueeze(0)).sum(dim=(-2, -1))  # (batch, out_features)
        
        output = cos_contrib + sin_contrib + self.bias
        
        return self.layer_norm(output)

class MLP(nn.Module):
    """Standard MLP for comparison."""
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class KAN(nn.Module):
    """Kolmogorov-Arnold Network using Fourier basis."""
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, 
                 num_frequencies=8, omega=1.0):
        super().__init__()
        layers = []
        dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        
        for i in range(len(dims) - 1):
            layers.append(FourierKANLayer(dims[i], dims[i+1], num_frequencies, omega))
            if i < len(dims) - 2:
                layers.append(nn.Dropout(0.1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MessagePassingLayer(nn.Module):
    """
    Message passing layer that can use either MLP or KAN for message and update functions.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, use_kan=True, num_frequencies=8):
        super().__init__()
        self.use_kan = use_kan
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Message function: combines node and edge features
        msg_input_dim = 2 * node_dim + edge_dim
        
        if use_kan:
            self.msg_fn = KAN(msg_input_dim, hidden_dim, hidden_dim, num_layers=2, 
                             num_frequencies=num_frequencies)
            self.update_fn = KAN(node_dim + hidden_dim, hidden_dim, node_dim, num_layers=2,
                                num_frequencies=num_frequencies)
        else:
            self.msg_fn = MLP(msg_input_dim, hidden_dim, hidden_dim, num_layers=2)
            self.update_fn = MLP(node_dim + hidden_dim, hidden_dim, node_dim, num_layers=2)
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: (E, edge_dim)
        Returns:
            Updated node features: (N, node_dim)
        """
        num_nodes = node_features.shape[0]
        
        # Get source and target nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Compute messages
        src_features = node_features[src]  # (E, node_dim)
        dst_features = node_features[dst]  # (E, node_dim)
        
        # Concatenate source, destination, and edge features
        msg_input = torch.cat([src_features, dst_features, edge_features], dim=-1)  # (E, 2*node_dim + edge_dim)
        messages = self.msg_fn(msg_input)  # (E, hidden_dim)
        
        # Aggregate messages (sum aggregation)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=node_features.device)
        aggregated.index_add_(0, dst, messages)
        
        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=-1)  # (N, node_dim + hidden_dim)
        updated = self.update_fn(update_input)  # (N, node_dim)
        
        # Residual connection
        return node_features + updated

class KAGNN(nn.Module):
    """
    Kolmogorov-Arnold Graph Neural Network for molecular property prediction.
    """
    def __init__(self, 
                 node_feature_dim=7,
                 edge_feature_dim=7,
                 hidden_dim=64,
                 num_layers=3,
                 num_classes=1,
                 use_kan=True,
                 num_frequencies=8,
                 omega=1.0,
                 pooling='mean'):
        super().__init__()
        
        self.use_kan = use_kan
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input embedding
        if use_kan:
            self.node_embedding = FourierKANLayer(node_feature_dim, hidden_dim, num_frequencies, omega)
            self.edge_embedding = FourierKANLayer(edge_feature_dim, hidden_dim, num_frequencies, omega)
        else:
            self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
            self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim, use_kan, num_frequencies)
            for _ in range(num_layers)
        ])
        
        # Readout layers
        readout_input_dim = hidden_dim * (num_layers + 1)  # Concatenate all layer outputs
        
        if use_kan:
            self.readout = KAN(readout_input_dim, hidden_dim, hidden_dim, num_layers=2, 
                              num_frequencies=num_frequencies)
            self.classifier = KAN(hidden_dim, hidden_dim // 2, num_classes, num_layers=2,
                                 num_frequencies=num_frequencies)
        else:
            self.readout = MLP(readout_input_dim, hidden_dim, hidden_dim, num_layers=2)
            self.classifier = MLP(hidden_dim, hidden_dim // 2, num_classes, num_layers=2)
    
    def forward(self, atom_features, bond_features, edge_index, batch):
        """
        Args:
            atom_features: (N, node_feature_dim)
            bond_features: (E, edge_feature_dim)
            edge_index: (2, E)
            batch: (N,) - batch assignment for each node
        Returns:
            predictions: (batch_size, num_classes)
        """
        # Embed features
        x = self.node_embedding(atom_features)  # (N, hidden_dim)
        edge_emb = self.edge_embedding(bond_features) if bond_features.shape[0] > 0 else torch.zeros(
            (edge_index.shape[1], x.shape[1]), device=x.device)
        
        # Store all layer outputs for skip connections
        layer_outputs = [x]
        
        # Message passing
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_emb)
            x = F.relu(x)
            layer_outputs.append(x)
        
        # Concatenate all layer outputs (Jumping Knowledge)
        x = torch.cat(layer_outputs, dim=-1)  # (N, hidden_dim * (num_layers + 1))
        
        # Readout
        x = self.readout(x)  # (N, hidden_dim)
        x = F.relu(x)
        
        # Graph-level pooling
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.shape[1], device=x.device)
        
        if self.pooling == 'mean':
            # Mean pooling
            pooled.index_add_(0, batch, x)
            counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
            pooled = pooled / counts.clamp(min=1)
        elif self.pooling == 'sum':
            pooled.index_add_(0, batch, x)
        elif self.pooling == 'max':
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    pooled[i] = x[mask].max(dim=0)[0]
        
        # Classification
        out = self.classifier(pooled)  # (num_graphs, num_classes)
        
        return out.squeeze(-1) if out.shape[-1] == 1 else out

class GCN(nn.Module):
    """
    Graph Convolutional Network baseline for comparison.
    Based on Kipf & Welling (2016).
    """
    def __init__(self, node_feature_dim=7, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, atom_features, bond_features, edge_index, batch):
        x = atom_features
        
        # Simple graph convolution
        for i, conv in enumerate(self.convs):
            # Aggregate neighbor features
            num_nodes = x.shape[0]
            agg = torch.zeros_like(x)
            agg.index_add_(0, edge_index[1], x[edge_index[0]])
            
            # Add self-loops and normalize
            x = conv(x + agg)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        num_graphs = batch.max().item() + 1
        pooled = torch.zeros(num_graphs, x.shape[1], device=x.device)
        pooled.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
        pooled = pooled / counts.clamp(min=1)
        
        return self.classifier(pooled).squeeze(-1)

# Test the models
if __name__ == '__main__':
    print("Testing KA-GNN models...")
    
    # Create dummy data
    atom_features = torch.randn(100, 7)
    bond_features = torch.randn(200, 7)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.cat([torch.zeros(30), torch.ones(30), torch.full((40,), 2)]).long()
    
    # Test KA-GNN
    model_kan = KAGNN(node_feature_dim=7, edge_feature_dim=7, hidden_dim=64, 
                      num_layers=3, num_classes=1, use_kan=True, num_frequencies=8)
    out_kan = model_kan(atom_features, bond_features, edge_index, batch)
    print(f"KA-GNN output shape: {out_kan.shape}")
    
    # Test GCN
    model_gcn = GCN(node_feature_dim=7, hidden_dim=64, num_layers=3, num_classes=1)
    out_gcn = model_gcn(atom_features, bond_features, edge_index, batch)
    print(f"GCN output shape: {out_gcn.shape}")
    
    # Test standard MLP-based GNN
    model_mlp = KAGNN(node_feature_dim=7, edge_feature_dim=7, hidden_dim=64,
                      num_layers=3, num_classes=1, use_kan=False)
    out_mlp = model_mlp(atom_features, bond_features, edge_index, batch)
    print(f"MLP-GNN output shape: {out_mlp.shape}")
    
    print("\nAll models tested successfully!")
