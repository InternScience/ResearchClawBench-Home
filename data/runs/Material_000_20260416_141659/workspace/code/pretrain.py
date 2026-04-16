import sys
import types
import torch
from torch.utils.data import Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv, BatchNorm
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

sys.modules['data_prepare'] = types.ModuleType('data_prepare')
class RealisticCrystalDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
sys.modules['data_prepare'].RealisticCrystalDataset = RealisticCrystalDataset

class GNNEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=False)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn3 = BatchNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = global_mean_pool(x, batch)
        return x

class PretrainModel(nn.Module):
    def __init__(self, encoder, hidden_channels):
        super(PretrainModel, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index, batch)
        z = self.projector(h)
        return z

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)
    
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
    return loss.mean()

def train_pretrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pretrain_data = torch.load('data/pretrain_data.pt', map_location=device, weights_only=False)
    loader = DataLoader(pretrain_data, batch_size=128, shuffle=True)

    num_node_features = pretrain_data[0].x.shape[1]
    hidden_channels = 128
    
    encoder = GNNEncoder(num_node_features, hidden_channels).to(device)
    model = PretrainModel(encoder, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting pretraining...")
    epochs = 20
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Data augmentation: Node dropping and feature masking
            drop_mask1 = (torch.rand(data.x.size(0), 1, device=device) > 0.1).float()
            x1 = data.x * drop_mask1
            
            drop_mask2 = (torch.rand(data.x.size(0), 1, device=device) > 0.1).float()
            x2 = data.x * drop_mask2
            
            z1 = model(x1, data.edge_index, data.batch)
            z2 = model(x2, data.edge_index, data.batch)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(loss_history)
    plt.title('Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/pretrain_loss.png')
    
    torch.save(encoder.state_dict(), 'outputs/pretrained_encoder.pth')
    print("Pretraining complete.")

if __name__ == "__main__":
    train_pretrain()
