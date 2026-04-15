"""
Crystal Graph Neural Network for Altermagnetic Material Discovery
==================================================================
This module implements a GNN-based approach for discovering altermagnetic materials.
Architecture: Graph Convolutional Network with self-supervised pre-training
and supervised fine-tuning for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader
import numpy as np
from typing import Tuple, Optional


class CrystalGCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for crystal structures.
    
    Processes crystal graphs through multiple GCN layers with residual connections,
    followed by global pooling to produce a fixed-size graph embedding.
    """
    
    def __init__(
        self,
        node_features: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN encoder.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            
            if self.use_residual and h_new.shape == h.shape:
                h_new = h_new + h
            
            h = F.relu(h_new)
            h = self.dropout(h)
        
        # Global pooling (concatenate mean and max)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        
        return torch.cat([mean_pool, max_pool], dim=-1)


class AltermagnetClassifier(nn.Module):
    """
    Binary classifier for altermagnetic materials.
    
    Combines a GCN encoder with a classification head.
    """
    
    def __init__(
        self,
        node_features: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = CrystalGCNEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classification head
        combined_dim = hidden_dim * 2  # mean + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            Logits [batch_size, 1]
        """
        graph_emb = self.encoder(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits
    
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        
        Returns:
            Probabilities [batch_size, 1]
        """
        logits = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)


class SelfSupervisedPretrainer(nn.Module):
    """
    Self-supervised pre-training via graph reconstruction and contrastive learning.
    
    Tasks:
    1. Node feature reconstruction from corrupted inputs
    2. Edge prediction (link prediction)
    3. Graph-level contrastive learning
    """
    
    def __init__(self, encoder: CrystalGCNEncoder):
        super().__init__()
        self.encoder = encoder
        
        # Node reconstruction head
        self.node_recon = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 2, encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder.hidden_dim, encoder.node_features)
        )
        
        # Edge prediction head
        self.edge_pred = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 4, encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder.hidden_dim, 1)
        )
        
    def forward_node_reconstruction(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct node features from graph embeddings."""
        graph_emb = self.encoder(x, edge_index, batch)
        # Broadcast graph embedding to nodes
        node_emb = graph_emb[batch]
        recon = self.node_recon(node_emb)
        return recon
    
    def forward_edge_prediction(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
        src_nodes: torch.Tensor, dst_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Predict existence of edges between node pairs."""
        graph_emb = self.encoder(x, edge_index, batch)
        src_emb = graph_emb[batch[src_nodes]]
        dst_emb = graph_emb[batch[dst_nodes]]
        combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb, 
                              torch.abs(src_emb - dst_emb)], dim=-1)
        return self.edge_pred(combined).squeeze(-1)
    
    def contrastive_loss(
        self, x1: torch.Tensor, edge_index1: torch.Tensor, batch1: torch.Tensor,
        x2: torch.Tensor, edge_index2: torch.Tensor, batch2: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """InfoNCE contrastive loss between augmented views."""
        emb1 = self.encoder(x1, edge_index1, batch1)
        emb2 = self.encoder(x2, edge_index2, batch2)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(emb1, emb2.T) / temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(emb1.size(0), device=emb1.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def create_dataloader(dataset, batch_size=64, shuffle=True):
    """Create a DataLoader from a RealisticCrystalDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(dataset, num_classes=2):
    """Compute class weights for imbalanced datasets."""
    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    print("Testing model architecture...")
    
    # Test encoder
    encoder = CrystalGCNEncoder(node_features=28, hidden_dim=64, num_layers=3)
    x = torch.randn(100, 28)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)
    
    emb = encoder(x, edge_index, batch)
    print(f"Encoder output shape: {emb.shape}")
    
    # Test classifier
    model = AltermagnetClassifier(node_features=28, hidden_dim=64, num_layers=3)
    logits = model(x, edge_index, batch)
    print(f"Classifier output shape: {logits.shape}")
    
    print("Model architecture test passed!")
