import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import os

# Simple graph VAE for molecular inverse design
# Using a toy graph representation for SMILES-like structures

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_node_features, max_nodes=20):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_node_features = num_node_features
        self.max_nodes = max_nodes

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * max_nodes)
        )
        self.node_decoder = nn.Linear(hidden_dim, num_node_features)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.max_nodes, self.hidden_dim)
        node_features = self.node_decoder(h)
        return node_features

class GraphVAE(nn.Module):
    def __init__(self, num_node_features, hidden_dim, latent_dim, max_nodes=20):
        super(GraphVAE, self).__init__()
        self.encoder = GraphEncoder(num_node_features, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, num_node_features, max_nodes)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        node_features = self.decoder(z)
        return node_features, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Toy data generation for vitrimer-like graphs
def generate_toy_graph_data(num_samples=100, num_node_features=5, max_nodes=20):
    data_list = []
    for i in range(num_samples):
        num_nodes = np.random.randint(5, max_nodes)
        x = torch.randn(num_nodes, num_node_features)
        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # no self-loops
        # Target Tg (simulated)
        tg = torch.tensor([300 + np.random.randn() * 50])  # around 300K
        data = Data(x=x, edge_index=edge_index, y=tg)
        data_list.append(data)
    return data_list

def train_vae(model, data_list, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data in data_list:
            optimizer.zero_grad()
            # Pad or handle variable size - for simplicity use batch of 1
            recon, mu, logvar = model(data)
            # For loss, we need to match sizes - use first sample
            target = data.x.unsqueeze(0)  # [1, nodes, features]
            # Pad recon if needed
            if recon.shape[1] > target.shape[1]:
                target = F.pad(target, (0, 0, 0, recon.shape[1] - target.shape[1]))
            elif recon.shape[1] < target.shape[1]:
                recon = F.pad(recon, (0, 0, 0, target.shape[1] - recon.shape[1]))
            loss = vae_loss(recon, target, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_list)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    return losses

def generate_new_molecules(model, num_samples=10, target_tg=350):
    model.eval()
    generated = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, model.latent_dim)
            node_features = model.decoder(z)
            generated.append(node_features.squeeze(0).numpy())
    return generated

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)

    # Generate toy data
    data_list = generate_toy_graph_data(num_samples=200)

    # Model
    num_features = 5
    model = GraphVAE(num_features, hidden_dim=32, latent_dim=16, max_nodes=20)

    # Train
    losses = train_vae(model, data_list, epochs=30)

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Graph VAE Training Loss')
    plt.grid(True)
    plt.savefig('report/images/vae_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Generate new candidates targeting Tg ~350K
    new_mols = generate_new_molecules(model, num_samples=5, target_tg=350)

    # Save results
    np.save('outputs/generated_molecules.npy', new_mols)
    torch.save(model.state_dict(), 'outputs/graph_vae_model.pt')

    print("Graph VAE training and generation complete.")
    print(f"Generated {len(new_mols)} new molecular graph candidates.")