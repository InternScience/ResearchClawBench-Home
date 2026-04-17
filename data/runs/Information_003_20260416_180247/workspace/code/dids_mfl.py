#!/usr/bin/env python3
"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Feature Learning

This module implements a disentangled dynamic intrusion detection framework that:
1. Disentangles entangled feature distributions through statistical and representational disentanglement
2. Incorporates dynamic graph diffusion for spatiotemporal aggregation
3. Enhances few-shot learning via multi-scale representation fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
from typing import Tuple, Dict, Optional


class StatisticalDisentanglement(nn.Module):
    """
    Statistical disentanglement module based on mutual information optimization.
    Separates features into attack-specific and benign components without parameters.
    """
    
    def __init__(self, input_dim: int, num_factors: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        # Learnable projection to disentangled factors
        self.projection = nn.Linear(input_dim, num_factors * input_dim)
        self.factor_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_factors)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch, input_dim]
        Returns:
            disentangled_features: [batch, num_factors, input_dim]
            factor_weights: [batch, num_factors]
        """
        batch_size = x.shape[0]
        # Project to factor space
        projected = self.projection(x).view(batch_size, self.num_factors, self.input_dim)
        # Compute attention weights for each factor
        factor_weights = F.softmax(self.factor_attention(x), dim=-1)
        # Apply attention
        disentangled = projected * factor_weights.unsqueeze(-1)
        return disentangled, factor_weights


class RepresentationalDisentanglement(nn.Module):
    """
    Memory-based representational disentanglement module.
    Generates attack-specific representations using a memory matrix.
    """
    
    def __init__(self, input_dim: int, memory_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.memory_size = memory_size
        self.num_classes = num_classes
        # Memory matrix for storing prototype representations
        self.memory = nn.Parameter(torch.randn(memory_size, input_dim))
        # Query network to read from memory
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, memory_size)
        )
        # Class-specific memory addressing
        self.class_memory = nn.Parameter(torch.randn(num_classes, input_dim))
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]
            labels: Optional class labels for supervised memory access
        Returns:
            memory_enhanced: Memory-enhanced representations [batch, input_dim]
        """
        # Compute query weights
        query_weights = F.softmax(self.query_net(x), dim=-1)
        # Read from memory
        memory_read = torch.matmul(query_weights, self.memory)
        # Combine with input
        enhanced = x + memory_read
        
        if labels is not None:
            # Add class-specific memory enhancement
            class_enhancement = self.class_memory[labels]
            enhanced = enhanced + 0.1 * class_enhancement
            
        return enhanced


class DynamicGraphDiffusion(MessagePassing):
    """
    Dynamic graph diffusion module for spatiotemporal aggregation.
    Adapts message passing based on temporal dynamics and edge features.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int = 16):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time-aware edge weight computation
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, 1)
        )
        
        # Edge feature processor
        self.edge_processor = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        
        # Node transformation
        self.node_transform = nn.Linear(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, in_channels]
            time: Timestamps [num_edges,] or [num_nodes,]
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Reshape time for edge-wise processing
        if time.dim() == 1 and time.shape[0] == edge_index.shape[1]:
            time_edge = time.unsqueeze(-1)
        else:
            # Use source node times
            time_edge = time[edge_index[0]].unsqueeze(-1)
            
        # Compute dynamic edge weights
        time_weight = torch.sigmoid(self.time_encoder(time_edge.float()))
        edge_weight = torch.sigmoid(self.edge_processor(edge_attr))
        dynamic_weight = time_weight * edge_weight
        
        # Transform node features
        x_trans = self.node_transform(x)
        
        # Propagate with dynamic weights
        out = self.propagate(edge_index, x=x_trans, weight=dynamic_weight.squeeze(-1))
        
        return out + x_trans  # Residual connection
    
    def message(self, x_j: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight.unsqueeze(-1) * x_j


class MultiScaleFusion(nn.Module):
    """
    Multi-scale representation fusion for enhanced few-shot learning.
    Combines features at different scales for robust classification.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales
        
        # Multi-scale feature extractors
        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.Linear(hidden_dim // (2 ** i), hidden_dim)
            )
            for i in range(num_scales)
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]
        Returns:
            Fused features [batch, input_dim]
        """
        # Extract multi-scale features
        scale_features = []
        for scale_net in self.scale_nets:
            scale_features.append(scale_net(x))
        
        # Concatenate all scales
        concat_features = torch.cat(scale_features, dim=-1)
        
        # Compute scale attention weights
        scale_weights = F.softmax(self.scale_attention(concat_features), dim=-1)
        
        # Weighted fusion
        weighted_features = []
        for i, sf in enumerate(scale_features):
            weighted_features.append(sf * scale_weights[:, i:i+1])
        
        fused = torch.cat(weighted_features, dim=-1)
        output = self.fusion(fused)
        
        return output + x  # Residual connection


class DIDS_MFL(nn.Module):
    """
    Complete DIDS-MFL architecture for intrusion detection.
    """
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, 
                 num_classes: int = 10, num_factors: int = 8):
        super().__init__()
        
        # Initial feature embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Statistical disentanglement
        self.stat_disentangle = StatisticalDisentanglement(hidden_dim, num_factors)
        
        # Representational disentanglement
        self.rep_disentangle = RepresentationalDisentanglement(hidden_dim, num_classes=num_classes)
        
        # Dynamic graph diffusion
        self.graph_diffusion = DynamicGraphDiffusion(hidden_dim, hidden_dim)
        
        # Multi-scale fusion
        self.multi_scale = MultiScaleFusion(hidden_dim)
        
        # Classification heads
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.multiclass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, msg: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, time: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                task: str = 'binary') -> torch.Tensor:
        """
        Args:
            msg: Message/edge features [num_edges, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, input_dim]
            time: Timestamps [num_edges,]
            labels: Optional labels for memory enhancement
            task: 'binary' or 'multiclass'
        Returns:
            Logits [num_edges, num_classes]
        """
        # Initial embedding
        x = self.input_embed(msg)
        
        # Statistical disentanglement
        disentangled, factor_weights = self.stat_disentangle(x)
        # Aggregate factors
        x = disentangled.mean(dim=1)
        
        # Representational disentanglement
        x = self.rep_disentangle(x, labels)
        
        # Note: For edge-level prediction, we process edge features directly
        # In a full implementation, we would build a proper graph structure
        
        # Multi-scale fusion
        x = self.multi_scale(x)
        
        # Classification
        if task == 'binary':
            logits = self.binary_head(x)
        else:
            logits = self.multiclass_head(x)
            
        return logits
    
    def forward_with_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_attr: torch.Tensor, time: torch.Tensor,
                           node_labels: Optional[torch.Tensor] = None,
                           task: str = 'binary') -> torch.Tensor:
        """
        Forward pass with explicit graph diffusion.
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, input_dim]
            time: Node timestamps [num_nodes,]
            node_labels: Optional node labels
            task: 'binary' or 'multiclass'
        Returns:
            Node logits [num_nodes, num_classes]
        """
        # Initial embedding
        h = self.input_embed(x)
        
        # Statistical disentanglement
        disentangled, factor_weights = self.stat_disentangle(h)
        h = disentangled.mean(dim=1)
        
        # Representational disentanglement
        h = self.rep_disentangle(h, node_labels)
        
        # Dynamic graph diffusion
        h = self.graph_diffusion(h, edge_index, edge_attr, time)
        
        # Multi-scale fusion
        h = self.multi_scale(h)
        
        # Classification
        if task == 'binary':
            logits = self.binary_head(h)
        else:
            logits = self.multiclass_head(h)
            
        return logits


def prepare_data(temporal_data) -> Dict:
    """
    Prepare data from PyG TemporalData for training.
    """
    store = temporal_data.stores[0]
    
    data_dict = {
        'msg': store.msg,  # Edge features
        'edge_index': temporal_data.edge_index,
        'src': store.src,
        'dst': store.dst,
        'time': store.t,
        'label': store.label,  # Binary labels
        'attack': store.attack,  # Multi-class labels
        'dt': store.dt,
    }
    
    return data_dict


def compute_class_distribution(labels: torch.Tensor) -> Dict[int, float]:
    """Compute class distribution."""
    unique, counts = torch.unique(labels, return_counts=True)
    total = len(labels)
    return {int(u.item()): float(c.item()) / total for u, c in zip(unique, counts)}


if __name__ == '__main__':
    # Test the model
    print("Testing DIDS-MFL model...")
    
    # Create dummy data
    batch_size = 64
    input_dim = 40
    hidden_dim = 128
    num_classes = 10
    num_edges = 100
    num_nodes = 50
    
    # Test edge-level forward pass
    model = DIDS_MFL(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    
    msg = torch.randn(num_edges, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, input_dim)
    time = torch.randn(num_edges)
    
    binary_logits = model(msg, edge_index, edge_attr, time, task='binary')
    multiclass_logits = model(msg, edge_index, edge_attr, time, task='multiclass')
    
    print(f"Binary logits shape: {binary_logits.shape}")
    print(f"Multiclass logits shape: {multiclass_logits.shape}")
    
    # Test node-level forward pass
    x = torch.randn(num_nodes, input_dim)
    node_time = torch.randn(num_nodes)
    
    binary_logits_node = model.forward_with_graph(x, edge_index, edge_attr, node_time, task='binary')
    print(f"Node-level binary logits shape: {binary_logits_node.shape}")
    
    print("Model test completed successfully!")
