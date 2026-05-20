"""
KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction
CPU-optimized version with reduced parameter counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import numpy as np
import math


class FourierKANLayer(nn.Module):
    """
    Fourier-based KAN layer with learnable univariate functions.
    CPU-optimized: computes using matrix operations rather than expanding all dims.
    
    φ(x) = Σ_{k=1}^{gridsize} (a_k * sin(k * π * x) + b_k * cos(k * π * x))
    """
    
    def __init__(self, in_features, out_features, gridsize=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gridsize = gridsize
        
        scale = 0.1 / math.sqrt(gridsize)
        # (out_features, in_features, gridsize) -> we'll compute differently for CPU
        self.fourier_coeffs_a = nn.Parameter(torch.randn(out_features, in_features, gridsize) * scale)
        self.fourier_coeffs_b = nn.Parameter(torch.randn(out_features, in_features, gridsize) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Frequencies: k * π
        self.register_buffer('freqs', math.pi * torch.arange(1, gridsize + 1).float())
        
    def forward(self, x):
        """
        Efficient CPU implementation.
        x: (N, in_features)
        returns: (N, out_features)
        """
        # Compute sin/cos for each frequency
        # x: (N, in) -> (N, in, 1) * freqs(1, 1, G) -> (N, in, G)
        x_ext = x.unsqueeze(-1)  # (N, in, 1)
        angles = x_ext * self.freqs.view(1, 1, -1)  # (N, in, G)
        
        sin_vals = torch.sin(angles)  # (N, in, G)
        cos_vals = torch.cos(angles)  # (N, in, G)
        
        # coeffs: (out, in, G), sin_vals: (N, in, G)
        # Need: sum over in and G -> (N, out)
        # einsum: 'nig,oig->no'
        a_out = torch.einsum('nig,oig->no', sin_vals, self.fourier_coeffs_a)
        b_out = torch.einsum('nig,oig->no', cos_vals, self.fourier_coeffs_b)
        
        return a_out + b_out + self.bias


class MultiFeatureEmbedding(nn.Module):
    """Embeds multiple categorical features per atom/bond."""
    
    def __init__(self, feature_sizes, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim, padding_idx=0)
            for size in feature_sizes
        ])
        self.total_dim = embedding_dim * len(feature_sizes)
        
    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=-1)


class KAGNNConv(nn.Module):
    """KA-GNN Convolution Layer with Fourier KAN message passing."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, kan_gridsize=4, dropout=0.1):
        super().__init__()
        
        msg_in_dim = node_dim * 2 + edge_dim
        
        # Message function
        self.message_fn = nn.Sequential(
            FourierKANLayer(msg_in_dim, hidden_dim, gridsize=kan_gridsize),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Edge gate
        self.edge_gate = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Update function
        self.update_fn = nn.Sequential(
            FourierKANLayer(node_dim + hidden_dim, hidden_dim, gridsize=kan_gridsize),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        msg_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_fn(msg_input)
        gates = self.edge_gate(msg_input)
        messages = messages * gates
        
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, messages.size(1), device=x.device, dtype=x.dtype)
        aggregated = aggregated.index_add(0, col, messages)
        
        update_input = torch.cat([x, aggregated], dim=-1)
        new_x = self.update_fn(update_input)
        new_x = x + self.dropout(new_x)  # Residual
        
        return new_x


class BaselineGCNConv(nn.Module):
    """MLP-based graph convolution for baseline comparison."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        msg_in_dim = node_dim * 2 + edge_dim
        self.message_fn = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.edge_gate = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        msg_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_fn(msg_input)
        gates = self.edge_gate(msg_input)
        messages = messages * gates
        
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, messages.size(1), device=x.device, dtype=x.dtype)
        aggregated = aggregated.index_add(0, col, messages)
        
        update_input = torch.cat([x, aggregated], dim=-1)
        new_x = self.update_fn(update_input)
        new_x = x + self.dropout(new_x)
        
        return new_x


class KAGNN(nn.Module):
    """Kolmogorov-Arnold Graph Neural Network for molecular property prediction."""
    
    def __init__(
        self,
        atom_feature_sizes=(17, 7, 7, 6, 5, 5, 3, 3, 201),
        atom_embed_dim=4,
        bond_feature_sizes=(6, 6, 3, 3, 3),
        bond_embed_dim=4,
        hidden_dim=64,
        num_layers=2,
        kan_gridsize=4,
        dropout=0.2,
        pool='mean',
        num_tasks=1,
    ):
        super().__init__()
        
        atom_total_dim = atom_embed_dim * len(atom_feature_sizes)
        bond_total_dim = bond_embed_dim * len(bond_feature_sizes)
        
        self.atom_embed = MultiFeatureEmbedding(atom_feature_sizes, atom_embed_dim)
        self.bond_embed = MultiFeatureEmbedding(bond_feature_sizes, bond_embed_dim)
        
        self.node_init = nn.Sequential(
            nn.Linear(atom_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                KAGNNConv(hidden_dim, bond_total_dim, hidden_dim, kan_gridsize, dropout)
            )
        
        self.pool = pool
        
        pred_in_dim = hidden_dim * 2 if pool == 'meanmax' else hidden_dim
        
        # KAN-based prediction head
        self.pred_head = nn.Sequential(
            FourierKANLayer(pred_in_dim, hidden_dim, gridsize=kan_gridsize),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )
        
        self.num_tasks = num_tasks
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.atom_embed(x)
        edge_attr = self.bond_embed(edge_attr)
        x = self.node_init(x)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.silu(x)
        
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        elif self.pool == 'meanmax':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)
        
        out = self.pred_head(x)
        if self.num_tasks == 1:
            out = out.squeeze(-1)
        return out


class BaselineGCN(nn.Module):
    """Baseline GCN with MLP-based message passing."""
    
    def __init__(
        self,
        atom_feature_sizes=(17, 7, 7, 6, 5, 5, 3, 3, 201),
        atom_embed_dim=4,
        bond_feature_sizes=(6, 6, 3, 3, 3),
        bond_embed_dim=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        pool='mean',
        num_tasks=1,
    ):
        super().__init__()
        
        atom_total_dim = atom_embed_dim * len(atom_feature_sizes)
        bond_total_dim = bond_embed_dim * len(bond_feature_sizes)
        
        self.atom_embed = MultiFeatureEmbedding(atom_feature_sizes, atom_embed_dim)
        self.bond_embed = MultiFeatureEmbedding(bond_feature_sizes, bond_embed_dim)
        
        self.node_init = nn.Sequential(
            nn.Linear(atom_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                BaselineGCNConv(hidden_dim, bond_total_dim, hidden_dim, dropout)
            )
        
        self.pool = pool
        
        pred_in_dim = hidden_dim * 2 if pool == 'meanmax' else hidden_dim
        
        self.pred_head = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )
        
        self.num_tasks = num_tasks
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.atom_embed(x)
        edge_attr = self.bond_embed(edge_attr)
        x = self.node_init(x)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.silu(x)
        
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool == 'add':
            x = global_add_pool(x, batch)
        elif self.pool == 'max':
            x = global_max_pool(x, batch)
        elif self.pool == 'meanmax':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=-1)
        
        out = self.pred_head(x)
        if self.num_tasks == 1:
            out = out.squeeze(-1)
        return out
