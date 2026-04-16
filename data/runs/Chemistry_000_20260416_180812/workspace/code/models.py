import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import numpy as np
import math

class KANLayer(nn.Module):
    """
    Fourier-based Kolmogorov-Arnold Network Layer.
    Instead of MLPs, we use Fourier features for transformations.
    """
    def __init__(self, in_features, out_features, num_frequencies=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # We will map each input feature x_i to sum_k (a_ik * cos(k*x_i) + b_ik * sin(k*x_i))
        # Then sum over i for each output j.
        # This is equivalent to a linear layer over the Fourier features.
        
        # Frequencies: 1, 2, ..., num_frequencies
        self.register_buffer('frequencies', torch.arange(1, num_frequencies + 1, dtype=torch.float32))
        
        # Weights for cos and sin
        # Shape: (out_features, in_features, num_frequencies)
        self.weight_cos = nn.Parameter(torch.Tensor(out_features, in_features, num_frequencies))
        self.weight_sin = nn.Parameter(torch.Tensor(out_features, in_features, num_frequencies))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_cos, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_sin, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features * self.num_frequencies)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # x: (batch_size, in_features)
        batch_size = x.shape[0]
        
        # x_freq: (batch_size, in_features, num_frequencies)
        x_freq = x.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        
        x_cos = torch.cos(x_freq)
        x_sin = torch.sin(x_freq)
        
        # Einsum: 
        # x_cos: (B, I, F)
        # weight_cos: (O, I, F)
        # output: (B, O)
        out = torch.einsum('bif,oif->bo', x_cos, self.weight_cos) + \
              torch.einsum('bif,oif->bo', x_sin, self.weight_sin) + \
              self.bias
              
        return out

class KAGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, num_frequencies=4):
        super().__init__()
        # Initial transformation
        self.node_emb = KANLayer(node_features, hidden_dim, num_frequencies)
        
        # GNN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.kan1 = KANLayer(hidden_dim, hidden_dim, num_frequencies)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, hidden_dim, num_frequencies)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.kan3 = KANLayer(hidden_dim, hidden_dim, num_frequencies)
        
        # Readout KAN
        self.readout_kan = KANLayer(hidden_dim, num_classes, num_frequencies)

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x)
        
        x = self.conv1(x, edge_index)
        x = self.kan1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.kan2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.kan3(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Readout
        out = self.readout_kan(x)
        return out

class BaselineGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes):
        super().__init__()
        self.node_emb = nn.Linear(node_features, hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.readout_mlp = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x)
        
        x = self.conv1(x, edge_index)
        x = self.mlp1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.mlp2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.mlp3(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Readout
        out = self.readout_mlp(x)
        return out
