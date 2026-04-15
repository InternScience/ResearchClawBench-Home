import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import dropout_edge

class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64, num_layers=3, out_dim=128):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.mlp1 = torch.nn.Linear(input_dim, hidden_dim)
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.mlp_out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.mlp1(x))
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index))
            x = F.relu(x)
        if batch is None:
            x = global_add_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            x = global_add_pool(x, batch)
        x = self.mlp_out(x)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.batch)
        out = self.head(h).squeeze(-1)
        return out
