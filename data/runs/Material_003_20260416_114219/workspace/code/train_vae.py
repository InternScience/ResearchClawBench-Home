import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.nn import GraphConv
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Graph VAE for molecules
# We'll use a simple node feature representation (atom type)
# and build a small Graph VAE.

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        idx = ATOM_TYPES.index(atom_type) if atom_type in ATOM_TYPES else len(ATOM_TYPES)
        # One-hot encode atom type
        feat = [0] * (len(ATOM_TYPES) + 1)
        feat[idx] = 1
        node_features.append(feat)
        
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Edges
    src = []
    dst = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
        
    if len(src) == 0:
        # Single atom molecule (edge case)
        g = dgl.graph(([], []))
        g.add_nodes(mol.GetNumAtoms())
    else:
        g = dgl.graph((src, dst))
        
    g.ndata['h'] = node_features
    return g

class VitrimerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.graphs = []
        self.tgs = []
        
        for idx, row in self.df.iterrows():
            # Combine acid and epoxide SMILES to form a single graph or process them separately.
            # For simplicity, we'll join them with a dot to represent the pair.
            smiles_pair = f"{row['acid']}.{row['epoxide']}"
            g = smiles_to_graph(smiles_pair)
            if g is not None:
                self.graphs.append(g)
                self.tgs.append(row['tg_calibrated'])
                
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.tgs[idx], dtype=torch.float32)

def collate(samples):
    graphs, tgs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(tgs)

class GraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        # Encoder
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (Predicts node features and adjacency)
        self.decoder_node = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), # +1 for Tg conditioning
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def encode(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Global pooling
        hg = dgl.mean_nodes(g, 'h')
        
        mu = self.fc_mu(hg)
        logvar = self.fc_logvar(hg)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, tg, g):
        # Condition on Tg
        z_cond = torch.cat([z, tg.unsqueeze(1)], dim=1)
        
        # Broadcast z_cond to nodes
        z_cond_nodes = dgl.broadcast_nodes(g, z_cond)
        node_preds = self.decoder_node(z_cond_nodes)
        
        return node_preds

    def forward(self, g, features, tg):
        mu, logvar = self.encode(g, features)
        z = self.reparameterize(mu, logvar)
        node_preds = self.decode(z, tg, g)
        return node_preds, mu, logvar

def loss_function(node_preds, node_labels, mu, logvar):
    # Reconstruction loss (node features)
    # Using BCEWithLogitsLoss since we one-hot encoded
    recon_loss = F.binary_cross_entropy_with_logits(node_preds, node_labels, reduction='sum')
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_loss, recon_loss, kld_loss

def train_vae():
    dataset = VitrimerDataset('outputs/tg_vitrimer_calibrated.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    
    in_dim = len(ATOM_TYPES) + 1
    hidden_dim = 64
    latent_dim = 16
    
    model = GraphVAE(in_dim, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 50
    losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batched_graph, tgs in dataloader:
            features = batched_graph.ndata['h']
            
            optimizer.zero_grad()
            node_preds, mu, logvar = model(batched_graph, features, tgs)
            
            loss, recon, kld = loss_function(node_preds, features, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
            
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Graph VAE Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('report/images/vae_training_loss.png')
    plt.close()
    
    torch.save(model.state_dict(), 'outputs/vae_model.pth')
    print("Saved VAE model to outputs/vae_model.pth")
    return model

if __name__ == '__main__':
    train_vae()
