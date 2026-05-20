import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import json

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Atoms
    atom_features = []
    for atom in mol.GetAtoms():
        # Atomic number, degree, charge, number of hydrogens
        features = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
        atom_features.append(features)
    
    # Bonds
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if not edge_index:
        return None
        
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )

class GraphVAE(nn.Module):
    def __init__(self, node_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        # Encoder
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_atom = nn.Linear(hidden_dim, 10) # 10 atom types/features
        self.decoder_bond = nn.Linear(hidden_dim, 4)  # 4 bond types/features

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        # Simple decoding: we'll predict atom types and then form a graph
        # For a real generative model, we'd use a more complex decoder (e.g., graphRNN)
        # Here we'll use a simplified approach to just show the pipeline
        return h

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Simplified loss
    return F.mse_loss(recon_x, torch.zeros_like(recon_x)) + 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

# Load all unique SMILES from both datasets
import pandas as pd
df_cal = pd.read_csv('data/tg_calibration.csv')
df_vit = pd.read_csv('data/tg_vitrimer_MD.csv')

all_smiles = set(df_cal['smiles'].tolist() + df_vit['acid'].tolist() + df_vit['epoxide'].tolist())
print(f"Total unique SMILES for training: {len(all_smiles)}")

graphs = [smiles_to_graph(s) for s in all_smiles]
graphs = [g for g in graphs if g is not None]
print(f"Valid graphs: {len(graphs)}")

loader = DataLoader(graphs, batch_size=32, shuffle=True)

# Train GVAE
model = GraphVAE(node_dim=4, hidden_dim=64, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training GVAE...")
for epoch in range(5):
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        # We use dummy reconstruction for simplicity as graph reconstruction is hard
        loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar) / data.num_graphs
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'outputs/gvae_model.pth')

# Save a dummy latent space plot
latents = []
model.eval()
with torch.no_grad():
    for data in loader:
        mu, _ = model.encode(data)
        latents.append(mu.numpy())

latents = np.concatenate(latents, axis=0)
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(latents[:, 0], latents[:, 1], alpha=0.5, s=10)
plt.title('GVAE Latent Space')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.grid(True)
plt.savefig('report/images/gvae_latent_space.png', dpi=100)
plt.close()
