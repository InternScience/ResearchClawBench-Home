"""Shared model definitions and data loading helpers."""
from __future__ import annotations
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


def load_dataset(name: str):
    return torch.load(os.path.join(ROOT, "data", name), weights_only=False)


class GNNEncoder(nn.Module):
    """A small GINE-based crystal-graph encoder.

    Inputs: x [N, 28] one-hot, edge_index [2, E], edge_attr [E, 2].
    Output: graph embedding [B, hidden] and node embedding [N, hidden].
    """

    def __init__(self, in_dim: int = 28, edge_dim: int = 2,
                 hidden: int = 96, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.atom_embed = nn.Linear(in_dim, hidden)
        self.edge_embed = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.hidden = hidden

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        e = self.edge_embed(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g_mean = global_mean_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_mean, g_add], dim=-1)  # [B, 2*hidden]
        return g, h


class PretrainHead(nn.Module):
    """Masked node-feature reconstruction head + projection for SimCLR-style
    contrastive view."""

    def __init__(self, hidden: int, in_dim: int = 28, proj: int = 64):
        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, proj)
        )


class Classifier(nn.Module):
    def __init__(self, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, g):
        return self.net(g).squeeze(-1)
