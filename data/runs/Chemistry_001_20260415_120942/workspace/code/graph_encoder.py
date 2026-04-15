"""
Graph Neural Network encoder for biomolecular structures.
Implements geometric deep learning architectures for protein-ligand complex encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """Graph attention layer (GAT)."""
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.size()
        
        # Linear transformation
        h = torch.matmul(x, self.W)  # (batch_size, num_nodes, out_features)
        
        # Compute attention scores
        a_input = torch.cat([
            h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1),
            h.repeat(1, num_nodes, 1)
        ], dim=-1).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)
        
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1), negative_slope=self.alpha)
        
        # Mask attention scores with adjacency matrix
        mask = adj == 0
        e = e.masked_fill(mask, float('-inf'))
        
        # Softmax
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)
        
        return h_prime, attention


class GeometricGraphEncoder(nn.Module):
    """
    Graph encoder that incorporates geometric information.
    """
    
    def __init__(self, node_feature_dim, hidden_dim=256, num_layers=4, dropout=0.1):
        super(GeometricGraphEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Edge feature projection for geometric information
        self.edge_proj = nn.Linear(1, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def build_distance_adjacency(self, coords, threshold=8.0):
        """
        Build adjacency matrix based on distance threshold.
        
        Args:
            coords: Node coordinates (batch_size, num_nodes, 3)
            threshold: Distance threshold for edges
        Returns:
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = coords.size()
        
        # Compute pairwise distances
        coords_i = coords.unsqueeze(2)  # (batch_size, num_nodes, 1, 3)
        coords_j = coords.unsqueeze(1)  # (batch_size, 1, num_nodes, 3)
        
        dists = torch.sqrt(torch.sum((coords_i - coords_j) ** 2, dim=-1) + 1e-8)
        
        # Create adjacency matrix (1 if distance < threshold, 0 otherwise)
        adj = (dists < threshold).float()
        
        # Add self-connections
        eye = torch.eye(num_nodes, device=coords.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        adj = (adj > 0).float()
        
        return adj
    
    def forward(self, node_features, coords):
        """
        Args:
            node_features: (batch_size, num_nodes, node_feature_dim)
            coords: (batch_size, num_nodes, 3)
        Returns:
            node_embeddings: (batch_size, num_nodes, hidden_dim)
        """
        # Build adjacency matrix from coordinates
        adj = self.build_distance_adjacency(coords)
        
        # Initial projection
        x = self.input_proj(node_features)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Graph convolution and attention layers
        for gcn_layer, gat_layer, layer_norm in zip(self.gcn_layers, self.gat_layers, self.layer_norms):
            # GCN
            x_gcn = gcn_layer(x, adj)
            x_gcn = F.relu(x_gcn)
            
            # GAT
            x_gat, _ = gat_layer(x, adj)
            x_gat = F.relu(x_gat)
            
            # Combine and normalize
            x = layer_norm(x + x_gcn + x_gat)
            x = self.dropout(x)
        
        return x


class HeterogeneousGraphEncoder(nn.Module):
    """
    Encoder for heterogeneous graphs (protein + ligand).
    """
    
    def __init__(self, protein_feature_dim=20, ligand_feature_dim=103, 
                 hidden_dim=256, num_layers=4):
        super(HeterogeneousGraphEncoder, self).__init__()
        
        # Separate encoders for protein and ligand
        self.protein_encoder = GeometricGraphEncoder(
            protein_feature_dim, hidden_dim, num_layers
        )
        self.ligand_encoder = GeometricGraphEncoder(
            ligand_feature_dim, hidden_dim, num_layers
        )
        
        # Cross-attention for protein-ligand interactions
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, protein_features, protein_coords, ligand_features, ligand_coords):
        """
        Args:
            protein_features: (batch_size, num_protein_nodes, protein_feature_dim)
            protein_coords: (batch_size, num_protein_nodes, 3)
            ligand_features: (batch_size, num_ligand_nodes, ligand_feature_dim)
            ligand_coords: (batch_size, num_ligand_nodes, 3)
        Returns:
            protein_embedding: (batch_size, num_protein_nodes, hidden_dim)
            ligand_embedding: (batch_size, num_ligand_nodes, hidden_dim)
            complex_embedding: (batch_size, hidden_dim)
        """
        # Encode protein
        protein_emb = self.protein_encoder(protein_features, protein_coords)
        
        # Encode ligand
        ligand_emb = self.ligand_encoder(ligand_features, ligand_coords)
        
        # Cross-attention between protein and ligand
        protein_attended, _ = self.cross_attention(
            protein_emb, ligand_emb, ligand_emb
        )
        ligand_attended, _ = self.cross_attention(
            ligand_emb, protein_emb, protein_emb
        )
        
        # Combine embeddings
        protein_output = protein_emb + protein_attended
        ligand_output = ligand_emb + ligand_attended
        
        # Global pooling
        protein_global = torch.mean(protein_output, dim=1)
        ligand_global = torch.mean(ligand_output, dim=1)
        
        complex_embedding = (protein_global + ligand_global) / 2
        
        return protein_output, ligand_output, complex_embedding


if __name__ == "__main__":
    # Test the encoder
    batch_size = 2
    num_protein_nodes = 107
    num_ligand_nodes = 50
    
    protein_features = torch.randn(batch_size, num_protein_nodes, 20)
    protein_coords = torch.randn(batch_size, num_protein_nodes, 3)
    ligand_features = torch.randn(batch_size, num_ligand_nodes, 103)
    ligand_coords = torch.randn(batch_size, num_ligand_nodes, 3)
    
    encoder = HeterogeneousGraphEncoder()
    protein_emb, ligand_emb, complex_emb = encoder(
        protein_features, protein_coords, ligand_features, ligand_coords
    )
    
    print(f"Protein embedding shape: {protein_emb.shape}")
    print(f"Ligand embedding shape: {ligand_emb.shape}")
    print(f"Complex embedding shape: {complex_emb.shape}")
