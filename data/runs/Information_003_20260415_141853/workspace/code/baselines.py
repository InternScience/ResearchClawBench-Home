"""
Baseline Models for Comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import numpy as np

class MLPBaseline(nn.Module):
    """Multi-layer Perceptron baseline"""
    def __init__(self, n_features, n_classes, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LSTMBaseline(nn.Module):
    """LSTM-based sequential model"""
    def __init__(self, n_features, n_classes, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features] -> treat each sample as seq of 1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline for edge/node classification"""
    def __init__(self, n_features, n_classes, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(n_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out


class EGraphSAGE(nn.Module):
    """
    E-GraphSAGE: Edge-featured GraphSAGE for NIDS
    Based on: E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT
    """
    def __init__(self, n_features, n_classes, hidden_dim=128, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(n_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, n_features]
            edge_index: Edge indices [2, num_edges]
        Returns:
            logits: [num_nodes, n_classes]
        """
        hidden_states = []
        h = x
        
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=0.3, training=self.training)
            hidden_states.append(h)
        
        # Concatenate all layer outputs (skip connections)
        h_concat = torch.cat(hidden_states, dim=-1)
        out = self.classifier(h_concat)
        return out


class GATBaseline(nn.Module):
    """Graph Attention Network baseline"""
    def __init__(self, n_features, n_classes, hidden_dim=128, heads=4):
        super().__init__()
        self.conv1 = GATConv(n_features, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out


class TabNetBaseline(nn.Module):
    """TabNet-inspired attention-based model for tabular data"""
    def __init__(self, n_features, n_classes, hidden_dim=128, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        
        # Feature transformer
        self.feature_transform = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention transformer for each step
        self.attention_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_features),
                nn.Sigmoid()
            ) for _ in range(n_steps)
        ])
        
        # Decision transformers
        self.decision_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_steps)
        ])
        
        self.final_classifier = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initial feature transform
        h = self.feature_transform(x)
        
        # Accumulate decisions from each step
        output = 0
        for i in range(self.n_steps):
            # Compute attention mask
            mask = self.attention_transforms[i](h)
            masked_features = x * mask
            
            # Transform and accumulate
            h_step = self.feature_transform(masked_features)
            h_step = self.decision_transforms[i](h_step)
            output = output + h_step
            h = h_step
        
        out = self.final_classifier(output)
        return out


if __name__ == '__main__':
    # Test models
    batch_size = 32
    n_features = 40
    n_classes = 10
    num_nodes = 100
    
    # Test non-graph models
    x = torch.randn(batch_size, n_features)
    
    models = {
        'MLP': MLPBaseline(n_features, n_classes),
        'LSTM': LSTMBaseline(n_features, n_classes),
        'TabNet': TabNetBaseline(n_features, n_classes)
    }
    
    print("Testing non-graph models:")
    for name, model in models.items():
        out = model(x)
        print(f"  {name}: {x.shape} -> {out.shape}")
    
    # Test graph models
    x_graph = torch.randn(num_nodes, n_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    graph_models = {
        'GraphSAGE': GraphSAGEBaseline(n_features, n_classes),
        'E-GraphSAGE': EGraphSAGE(n_features, n_classes),
        'GAT': GATBaseline(n_features, n_classes)
    }
    
    print("\nTesting graph models:")
    for name, model in graph_models.items():
        out = model(x_graph, edge_index)
        print(f"  {name}: {x_graph.shape} -> {out.shape}")
