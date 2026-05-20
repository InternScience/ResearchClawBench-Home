"""Graph construction and GNN models for dynamic intrusion detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Data
import numpy as np
from collections import defaultdict

def build_static_graphs(data, time_window=3600):
    """Build static graph snapshots from temporal data using time windows."""
    timestamps = data.t.numpy()
    src = data.src.numpy()
    dst = data.dst.numpy()
    features = data.msg
    labels = data.label
    attacks = data.attack
    
    max_t = timestamps.max()
    num_windows = int(np.ceil((max_t + 1) / time_window))
    
    graphs = []
    for w in range(num_windows):
        t_start = w * time_window
        t_end = min((w + 1) * time_window, max_t + 1)
        mask = (timestamps >= t_start) & (timestamps < t_end)
        
        if mask.sum() < 2:
            continue
            
        edge_index = torch.stack([data.src[mask], data.dst[mask]], dim=0)
        # Remap node IDs locally
        unique_nodes = torch.unique(edge_index)
        node_map = {n.item(): i for i, n in enumerate(unique_nodes)}
        
        src_mapped = torch.tensor([node_map[s.item()] for s in edge_index[0]], dtype=torch.long)
        dst_mapped = torch.tensor([node_map[d.item()] for d in edge_index[1]], dtype=torch.long)
        edge_index_mapped = torch.stack([src_mapped, dst_mapped], dim=0)
        
        g = Data(
            edge_index=edge_index_mapped,
            edge_attr=data.msg[mask],
            edge_label=data.label[mask],
            edge_attack=data.attack[mask],
            num_nodes=len(unique_nodes)
        )
        graphs.append(g)
    
    return graphs

def build_edge_graph(edge_features, k_nn=10):
    """Build a KNN graph among edges (flows) for message passing."""
    from sklearn.neighbors import NearestNeighbors
    
    feats = edge_features.numpy()
    nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, len(feats)), algorithm='auto').fit(feats)
    distances, indices = nbrs.kneighbors(feats)
    
    edge_index_list = []
    for i in range(len(feats)):
        for j_idx in range(1, len(indices[i])):
            edge_index_list.append([i, indices[i][j_idx]])
    
    if len(edge_index_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    ei = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return ei


class DisentangledGNN(nn.Module):
    """GNN with representational disentanglement."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_factors=4):
        super().__init__()
        self.num_factors = num_factors
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        self.factor_convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_factors, heads=1, dropout=0.1)
            for _ in range(num_factors)
        ])
        
        self.factor_out = nn.ModuleList([
            nn.Linear(hidden_dim // num_factors, out_dim // num_factors)
            for _ in range(num_factors)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x, edge_index):
        x = F.relu(self.in_proj(x))
        
        factor_outs = []
        for i in range(self.num_factors):
            h = self.factor_convs[i](x, edge_index)
            h = F.relu(h)
            h = self.factor_out[i](h)
            factor_outs.append(h)
        
        z = torch.cat(factor_outs, dim=-1)
        z = self.fusion(z)
        return z
    
    def disentanglement_loss(self, z):
        """Covariance regularization to encourage factor independence."""
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
        
        # Off-diagonal penalty
        mask = 1 - torch.eye(cov.shape[0], device=cov.device)
        off_diag = (cov * mask).pow(2).sum()
        return off_diag / cov.shape[0]


class EdgeClassifier(nn.Module):
    """Edge-level classifier for intrusion detection."""
    def __init__(self, in_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class DIDS_MFL(nn.Module):
    """Full DIDS-MFL model: Disentangled Dynamic Intrusion Detection with Multi-scale Feature Learning."""
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, num_classes=2, num_factors=4):
        super().__init__()
        self.gnn = DisentangledGNN(in_dim, hidden_dim, latent_dim, num_factors)
        self.classifier = EdgeClassifier(latent_dim, hidden_dim, num_classes)
        self.disent_weight = 0.01
        
    def forward(self, x, edge_index, return_latent=False):
        z = self.gnn(x, edge_index)
        logits = self.classifier(z)
        if return_latent:
            return logits, z
        return logits
    
    def compute_loss(self, logits, labels, z):
        ce_loss = F.cross_entropy(logits, labels)
        disent_loss = self.gnn.disentanglement_loss(z)
        return ce_loss + self.disent_weight * disent_loss
