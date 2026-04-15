"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection System
with Multi-scale Fusion Learning

Based on concepts from:
- 3D-IDS (Doubly Disentangled Dynamic Intrusion Detection) [1]
- DisenLink (Disentangled Representation Learning) [2]
- E-GraphSAGE (GNN-based NIDS) [3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from sklearn.mutual_info import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class StatisticalDisentanglement(nn.Module):
    """
    Step 1: Statistical Feature Disentanglement
    Uses mutual information-based feature weighting to differentiate features
    """
    def __init__(self, n_features, n_factors=4):
        super().__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        
        # Learnable factor assignment
        self.factor_assignment = nn.Parameter(torch.randn(n_features, n_factors))
        
        # Feature importance scoring
        self.feature_scorer = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features),
            nn.Sigmoid()
        )
        
    def compute_mi_weights(self, features, labels):
        """Compute mutual information based weights"""
        # Convert to numpy for sklearn
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
        else:
            features_np = features
            labels_np = labels
            
        # Compute mutual information for each feature
        mi_scores = mutual_info_classif(features_np, labels_np, random_state=42)
        mi_scores = torch.tensor(mi_scores, dtype=torch.float32, device=features.device)
        
        # Normalize
        mi_scores = mi_scores / (mi_scores.sum() + 1e-8)
        return mi_scores
    
    def forward(self, x, labels=None):
        """
        Args:
            x: [batch_size, n_features]
            labels: optional labels for MI computation
        Returns:
            disentangled_features: [batch_size, n_factors, n_features]
        """
        batch_size = x.size(0)
        
        # Compute feature importance
        feature_weights = self.feature_scorer(x.mean(dim=0, keepdim=True)).squeeze()
        
        # Apply MI-based weighting if labels available
        if labels is not None and self.training:
            mi_weights = self.compute_mi_weights(x, labels)
            feature_weights = feature_weights * mi_weights
        
        # Generate factor-specific features
        factor_probs = F.softmax(self.factor_assignment, dim=1)  # [n_features, n_factors]
        
        disentangled = []
        for k in range(self.n_factors):
            # Weight features by factor assignment
            factor_weight = factor_probs[:, k] * feature_weights  # [n_features]
            factor_features = x * factor_weight.unsqueeze(0)  # [batch_size, n_features]
            disentangled.append(factor_features)
        
        # Stack to get [batch_size, n_factors, n_features]
        disentangled = torch.stack(disentangled, dim=1)
        return disentangled


class MemoryDisentanglement(nn.Module):
    """
    Step 2: Representational Disentanglement with Memory
    Uses memory network to generate and disentangle representations
    """
    def __init__(self, n_features, hidden_dim=128, memory_size=100):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Memory bank for prototypes
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention for memory addressing
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Disentanglement heads
        self.disentangle_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            ) for _ in range(4)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, n_factors, n_features]
        Returns:
            representations: [batch_size, n_factors, hidden_dim]
        """
        batch_size, n_factors, n_features = x.shape
        
        # Encode features
        x_flat = x.view(-1, n_features)  # [batch_size * n_factors, n_features]
        encoded = self.encoder(x_flat)  # [batch_size * n_factors, hidden_dim]
        encoded = encoded.view(batch_size, n_factors, self.hidden_dim)
        
        # Memory addressing with attention
        queries = self.query_proj(encoded)  # [batch_size, n_factors, hidden_dim]
        keys = self.key_proj(self.memory)  # [memory_size, hidden_dim]
        
        # Compute attention scores
        attn_scores = torch.einsum('bnd,kd->bnk', queries, keys)  # [batch_size, n_factors, memory_size]
        attn_weights = F.softmax(attn_scores / np.sqrt(self.hidden_dim), dim=-1)
        
        # Retrieve from memory
        memory_values = torch.einsum('bnk,kd->bnd', attn_weights, self.memory)  # [batch_size, n_factors, hidden_dim]
        
        # Combine encoded features with memory
        combined = encoded + memory_values
        
        # Apply disentanglement heads
        disentangled = []
        for k, head in enumerate(self.disentangle_heads[:n_factors]):
            factor_repr = head(combined[:, k, :])  # [batch_size, hidden_dim]
            disentangled.append(factor_repr)
        
        disentangled = torch.stack(disentangled, dim=1)  # [batch_size, n_factors, hidden_dim]
        return disentangled


class DynamicGraphDiffusion(MessagePassing):
    """
    Dynamic Graph Diffusion for Spatiotemporal Aggregation
    Learns to dynamically fuse network topology
    """
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        self.query = nn.Linear(in_channels, out_channels * num_heads)
        self.key = nn.Linear(in_channels, out_channels * num_heads)
        self.value = nn.Linear(in_channels, out_channels * num_heads)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_heads)
        )
        
        self.output_proj = nn.Linear(out_channels * num_heads, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_time=None):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_time: [num_edges] temporal information
        """
        # Add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Prepare edge time features
        if edge_time is not None:
            # Add dummy time for self-loops
            self_loop_time = torch.zeros(edge_index.size(1) - edge_time.size(0), 
                                         device=edge_time.device, dtype=edge_time.dtype)
            edge_time = torch.cat([edge_time, self_loop_time])
            edge_time = edge_time.unsqueeze(-1)  # [num_edges, 1]
        else:
            edge_time = torch.zeros(edge_index.size(1), 1, device=x.device)
        
        # Propagate
        out = self.propagate(edge_index, x=x, edge_time=edge_time, size=None)
        out = self.output_proj(out)
        return out
    
    def message(self, x_i, x_j, edge_time, index, ptr, size_i):
        """Compute messages with attention"""
        # x_i: [num_edges, in_channels] - source
        # x_j: [num_edges, in_channels] - target
        
        query = self.query(x_i).view(-1, self.num_heads, self.out_channels)
        key = self.key(x_j).view(-1, self.num_heads, self.out_channels)
        value = self.value(x_j).view(-1, self.num_heads, self.out_channels)
        
        # Compute attention scores
        attn_scores = (query * key).sum(dim=-1) / np.sqrt(self.out_channels)  # [num_edges, num_heads]
        
        # Incorporate temporal edge features
        edge_attn = self.edge_encoder(edge_time)  # [num_edges, num_heads]
        attn_scores = attn_scores + edge_attn
        
        attn_weights = softmax(attn_scores, index, ptr, size_i)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        messages = value * attn_weights.unsqueeze(-1)  # [num_edges, num_heads, out_channels]
        messages = messages.view(-1, self.num_heads * self.out_channels)
        
        return messages


class MultiScaleFusion(nn.Module):
    """
    Multi-scale Representation Fusion for Few-shot Learning
    """
    def __init__(self, hidden_dim, n_classes, n_scales=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_scales = n_scales
        
        # Multi-scale feature extractors
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(n_scales)
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * n_scales, n_scales),
            nn.Softmax(dim=-1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, n_factors, hidden_dim]
        Returns:
            logits: [batch_size, n_classes]
            fused_features: [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        
        # Pool across factors for each sample
        x_pooled = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Extract multi-scale features
        scale_features = []
        for extractor in self.scale_extractors:
            scale_feat = extractor(x_pooled)
            scale_features.append(scale_feat)
        
        # Stack scales
        multi_scale = torch.stack(scale_features, dim=1)  # [batch_size, n_scales, hidden_dim]
        
        # Compute scale attention
        scale_input = torch.cat(scale_features, dim=-1)  # [batch_size, hidden_dim * n_scales]
        scale_weights = self.scale_attention(scale_input)  # [batch_size, n_scales]
        
        # Fuse scales with attention
        fused = torch.einsum('bsh,bs->bh', multi_scale, scale_weights)  # [batch_size, hidden_dim]
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, fused


class DIDS_MFL(nn.Module):
    """
    Complete DIDS-MFL Framework
    """
    def __init__(self, n_features, n_classes=10, hidden_dim=128, n_factors=4):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_factors = n_factors
        
        # Disentanglement modules
        self.stat_disentangle = StatisticalDisentanglement(n_features, n_factors)
        self.mem_disentangle = MemoryDisentanglement(n_features, hidden_dim)
        
        # Graph diffusion (optional - for node-level tasks)
        self.graph_diffusion = DynamicGraphDiffusion(hidden_dim, hidden_dim)
        
        # Multi-scale fusion
        self.multi_scale = MultiScaleFusion(hidden_dim, n_classes)
        
    def forward(self, x, edge_index=None, edge_time=None, labels=None):
        """
        Args:
            x: [batch_size, n_features] or [num_nodes, n_features]
            edge_index: optional for graph diffusion
            edge_time: optional temporal edge features
            labels: optional for MI computation during training
        Returns:
            logits: [batch_size, n_classes]
        """
        # Step 1: Statistical Disentanglement
        stat_features = self.stat_disentangle(x, labels)  # [batch_size, n_factors, n_features]
        
        # Step 2: Representational Disentanglement
        repr_features = self.mem_disentangle(stat_features)  # [batch_size, n_factors, hidden_dim]
        
        # Step 3: Optional Graph Diffusion (if edge_index provided)
        if edge_index is not None:
            # Aggregate across factors for graph processing
            graph_input = repr_features.mean(dim=1)  # [batch_size, hidden_dim]
            graph_output = self.graph_diffusion(graph_input, edge_index, edge_time)
            # Expand back to factor dimension
            repr_features = graph_output.unsqueeze(1).expand(-1, self.n_factors, -1)
        
        # Step 4: Multi-scale Fusion and Classification
        logits, fused = self.multi_scale(repr_features)
        
        return logits
    
    def get_embeddings(self, x):
        """Get learned embeddings for visualization"""
        stat_features = self.stat_disentangle(x)
        repr_features = self.mem_disentangle(stat_features)
        logits, fused = self.multi_scale(repr_features)
        return fused


class PrototypicalLoss(nn.Module):
    """Prototypical loss for few-shot learning"""
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        
    def forward(self, embeddings, labels, support_mask=None):
        """
        Args:
            embeddings: [batch_size, hidden_dim]
            labels: [batch_size]
            support_mask: [batch_size] boolean mask for support set
        """
        if support_mask is None:
            # Use all as support
            support_mask = torch.ones_like(labels, dtype=torch.bool)
        
        # Compute prototypes
        prototypes = []
        for c in range(self.n_classes):
            class_mask = (labels == c) & support_mask
            if class_mask.sum() > 0:
                class_proto = embeddings[class_mask].mean(dim=0)
            else:
                class_proto = torch.zeros(embeddings.size(1), device=embeddings.device)
            prototypes.append(class_proto)
        prototypes = torch.stack(prototypes)  # [n_classes, hidden_dim]
        
        # Compute distances to prototypes
        distances = torch.cdist(embeddings, prototypes)  # [batch_size, n_classes]
        logits = -distances
        
        loss = F.cross_entropy(logits, labels)
        return loss, logits


if __name__ == '__main__':
    # Test model
    batch_size = 32
    n_features = 40
    n_classes = 10
    
    model = DIDS_MFL(n_features, n_classes)
    
    # Test forward pass
    x = torch.randn(batch_size, n_features)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    logits = model(x, labels=labels)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
