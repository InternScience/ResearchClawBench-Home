"""
Models: Fourier-based KAN Layer, GCN-MLP Baseline, and KA-GNN.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool


class FourierKANLayer(nn.Module):
    """
    Fourier-based Kolmogorov-Arnold Network (KAN) Layer.
    
    Instead of using fixed activation functions like ReLU in MLPs,
    this layer learns activation functions as Fourier series:
    
    phi(x) = a_0 + sum_{k=1}^{K} [a_k * cos(k * x) + b_k * sin(k * x)]
    
    Each input-output pair (i, j) has its own learnable Fourier coefficients,
    following the Kolmogorov-Arnold representation theorem.
    """
    def __init__(self, in_features, out_features, num_frequencies=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # Learnable Fourier coefficients for each (input, output) pair
        # a_0: bias term
        self.a0 = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        # a_k: cosine coefficients  
        self.a_cos = nn.Parameter(torch.randn(out_features, in_features, num_frequencies) * 0.1)
        # b_k: sine coefficients
        self.b_sin = nn.Parameter(torch.randn(out_features, in_features, num_frequencies) * 0.1)
        
        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x):
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Scale input
        x_scaled = x * self.freq_scale
        
        # Compute Fourier basis: cos(k*x) and sin(k*x) for k=1..K
        # x_scaled: (batch, in_features)
        k = torch.arange(1, self.num_frequencies + 1, device=x.device, dtype=x.dtype)
        # (batch, in_features, num_frequencies)
        kx = x_scaled.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0)
        
        cos_kx = torch.cos(kx)  # (batch, in_features, K)
        sin_kx = torch.sin(kx)  # (batch, in_features, K)
        
        # Compute output: sum over input features of phi_ij(x_i)
        # phi_ij(x_i) = a0[j,i] + sum_k a_cos[j,i,k]*cos(k*x_i) + b_sin[j,i,k]*sin(k*x_i)
        
        # Constant term: (batch, out_features) = (batch, in_features) @ (in_features, out_features)
        out = torch.matmul(x, self.a0.t())
        
        # Cosine terms: sum over k and i
        # a_cos: (out, in, K), cos_kx: (batch, in, K)
        # For each output j: sum_i sum_k a_cos[j,i,k] * cos_kx[batch,i,k]
        cos_contrib = torch.einsum('bik,jik->bj', cos_kx, self.a_cos)
        sin_contrib = torch.einsum('bik,jik->bj', sin_kx, self.b_sin)
        
        out = out + cos_contrib + sin_contrib
        out = self.layer_norm(out)
        
        return out


class FourierKANBlock(nn.Module):
    """A block of Fourier KAN layers with residual connection."""
    def __init__(self, in_features, out_features, num_frequencies=8):
        super().__init__()
        self.kan = FourierKANLayer(in_features, out_features, num_frequencies)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        return self.kan(x) + self.residual(x)


class GCN_MLP(nn.Module):
    """
    Baseline: GCN with MLP readout for graph-level prediction.
    Uses standard GCN message passing + MLP for classification.
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=3, num_tasks=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # MLP readout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP readout
        x = self.mlp(x)
        return x


class KA_GNN(nn.Module):
    """
    Kolmogorov-Arnold Graph Neural Network (KA-GNN).
    
    Replaces MLP transformations in GNN with Fourier-based KAN layers.
    Key differences from GCN-MLP:
    1. Node transformation uses KAN instead of linear+ReLU
    2. Readout uses KAN layers instead of MLP
    3. Learnable activation functions via Fourier series
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=3, num_tasks=1, 
                 dropout=0.2, num_frequencies=8):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection with KAN
        self.input_kan = FourierKANLayer(in_features, hidden_dim, num_frequencies)
        
        # GCN layers for message passing
        self.convs = nn.ModuleList()
        self.kan_transforms = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.kan_transforms.append(FourierKANBlock(hidden_dim, hidden_dim, num_frequencies))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # KAN readout (replaces MLP)
        self.readout = nn.Sequential(
            FourierKANLayer(hidden_dim, hidden_dim, num_frequencies),
            nn.Dropout(dropout),
            FourierKANLayer(hidden_dim, hidden_dim // 2, num_frequencies),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection with KAN
        x = self.input_kan(x)
        
        # GCN message passing with KAN transformations
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self.kan_transforms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # KAN readout
        x = self.readout(x)
        return x
    
    def get_fourier_coefficients(self):
        """Extract Fourier coefficients for interpretability analysis."""
        coefficients = {}
        for name, module in self.named_modules():
            if isinstance(module, FourierKANLayer):
                coefficients[name] = {
                    'a0': module.a0.detach().cpu().numpy(),
                    'a_cos': module.a_cos.detach().cpu().numpy(),
                    'b_sin': module.b_sin.detach().cpu().numpy(),
                    'freq_scale': module.freq_scale.detach().cpu().item(),
                }
        return coefficients


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from featurize import get_atom_feature_dim
    in_dim = get_atom_feature_dim()
    
    # Test models
    gcn_mlp = GCN_MLP(in_dim, hidden_dim=64, num_layers=3, num_tasks=1)
    ka_gnn = KA_GNN(in_dim, hidden_dim=64, num_layers=3, num_tasks=1, num_frequencies=8)
    
    print(f"GCN-MLP parameters: {count_parameters(gcn_mlp):,}")
    print(f"KA-GNN parameters: {count_parameters(ka_gnn):,}")
    
    # Test forward pass
    from featurize import smiles_to_graph
    from torch_geometric.data import Batch
    
    graphs = [smiles_to_graph('CCO', 1.0), smiles_to_graph('c1ccccc1', 0.0)]
    batch = Batch.from_data_list(graphs)
    
    out1 = gcn_mlp(batch)
    out2 = ka_gnn(batch)
    print(f"GCN-MLP output shape: {out1.shape}")
    print(f"KA-GNN output shape: {out2.shape}")
