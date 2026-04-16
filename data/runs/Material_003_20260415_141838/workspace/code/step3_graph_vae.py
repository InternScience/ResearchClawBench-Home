"""
Step 3: Graph Variational Autoencoder (Graph VAE)
- Build a graph VAE model using PyTorch and PyTorch Geometric
- Train on vitrimer molecular structures
- Generate latent representations and new candidate molecules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
import pickle
import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# Load graph data
# ==============================
with open('../outputs/all_molecule_graphs.pkl', 'rb') as f:
    graph_data = pickle.load(f)

all_graphs = graph_data['graphs']
all_smiles = graph_data['smiles']
print(f"Loaded {len(all_graphs)} molecular graphs")

# ==============================
# Prepare PyG Data objects
# ==============================
MAX_ATOMS = 80  # Maximum atoms per molecule

def prepare_pyg_data(graph, max_atoms=MAX_ATOMS):
    """Convert graph dict to PyG Data with padding."""
    n_atoms = graph['n_atoms']
    atom_feat = graph['atom_features']
    edge_list = graph['edge_list']
    edge_feat = graph['edge_features']
    
    # Pad atom features
    padded_atom_feat = np.zeros((max_atoms, atom_feat.shape[1]), dtype=np.float32)
    padded_atom_feat[:n_atoms] = atom_feat
    
    # Offset edges (no offset needed for single graph)
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feat, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_feat.shape[1] if len(edge_feat.shape) > 1 else 1), dtype=torch.float32)
    
    data = Data(
        x=torch.tensor(padded_atom_feat, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),
    )
    return data

pyg_dataset = []
valid_smiles_list = []
for i, g in enumerate(all_graphs):
    if g['n_atoms'] <= MAX_ATOMS:
        data = prepare_pyg_data(g)
        pyg_dataset.append(data)
        valid_smiles_list.append(all_smiles[i])

print(f"Dataset size after filtering (n_atoms <= {MAX_ATOMS}): {len(pyg_dataset)}")

# ==============================
# Graph VAE Model
# ==============================
class GraphEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h_pool = global_mean_pool(h, batch)
        mu = self.fc_mu(h_pool)
        logvar = self.fc_logvar(h_pool)
        return mu, logvar

class GraphDecoder(nn.Module):
    """Decode latent vector to molecular fingerprint (for property prediction and reconstruction)."""
    def __init__(self, latent_dim=64, hidden_dim=128, output_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))
        return out

class GraphVAE(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, latent_dim=64, fp_dim=1024):
        super().__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, fp_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        recon_fp = self.decoder(z)
        return recon_fp, mu, logvar
    
    def encode(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index, batch)
        return mu
    
    def decode(self, z):
        return self.decoder(z)

# ==============================
# Compute target fingerprints for reconstruction loss
# ==============================
print("Computing target fingerprints for VAE training...")
target_fps = []
for smi in valid_smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        target_fps.append(np.array(fp, dtype=np.float32))
    else:
        target_fps.append(np.zeros(1024, dtype=np.float32))

target_fps = np.array(target_fps)
print(f"Target fingerprints shape: {target_fps.shape}")

# ==============================
# Training
# ==============================
LATENT_DIM = 64
HIDDEN_DIM = 128
FP_DIM = 1024
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

model = GraphVAE(input_dim=7, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, fp_dim=FP_DIM).to(device)

# Custom collate for PyG
def collate_fn(batch):
    return Batch.from_data_list(batch)

# Split into train/val
n_total = len(pyg_dataset)
n_train = int(0.9 * n_total)
indices = np.random.RandomState(42).permutation(n_total)
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_dataset = [pyg_dataset[i] for i in train_indices]
val_dataset = [pyg_dataset[i] for i in val_indices]
train_fps = target_fps[train_indices]
val_fps = target_fps[val_indices]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def loss_function(recon_fp, target_fp, mu, logvar, beta=0.5):
    """VAE loss = reconstruction loss + KL divergence."""
    recon_loss = F.binary_cross_entropy(recon_fp, target_fp, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')

print(f"\nTraining Graph VAE...")
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"  Latent dim: {LATENT_DIM}, Hidden dim: {HIDDEN_DIM}")

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    n_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(train_fps))
        fp_target = torch.tensor(train_fps[batch_start:batch_end], dtype=torch.float32).to(device)
        
        if fp_target.shape[0] != batch_data.num_graphs:
            # Adjust for last batch
            fp_target = fp_target[:batch_data.num_graphs]
        
        recon_fp, mu, logvar = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        loss, recon_l, kl_l = loss_function(recon_fp, fp_target, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        n_batches += 1
    
    avg_train_loss = epoch_train_loss / n_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    epoch_val_loss = 0
    n_val_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(val_fps))
            fp_target = torch.tensor(val_fps[batch_start:batch_end], dtype=torch.float32).to(device)
            
            if fp_target.shape[0] != batch_data.num_graphs:
                fp_target = fp_target[:batch_data.num_graphs]
            
            recon_fp, mu, logvar = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss, _, _ = loss_function(recon_fp, fp_target, mu, logvar)
            epoch_val_loss += loss.item()
            n_val_batches += 1
    
    avg_val_loss = epoch_val_loss / n_val_batches
    val_losses.append(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '../outputs/graph_vae_best.pt')
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: Train Loss={avg_train_loss:.2f}, Val Loss={avg_val_loss:.2f}")

print(f"Best validation loss: {best_val_loss:.2f}")

# Save training history
training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss,
    'latent_dim': LATENT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'epochs': EPOCHS,
}
with open('../outputs/vae_training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)

# ==============================
# Encode all molecules to latent space
# ==============================
print("\nEncoding molecules to latent space...")
model.load_state_dict(torch.load('../outputs/graph_vae_best.pt', weights_only=True))
model.eval()

all_loader = DataLoader(pyg_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
latent_vectors = []

with torch.no_grad():
    for batch_data in all_loader:
        batch_data = batch_data.to(device)
        mu = model.encode(batch_data.x, batch_data.edge_index, batch_data.batch)
        latent_vectors.append(mu.cpu().numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)
np.save('../outputs/latent_vectors.npy', latent_vectors)
print(f"Latent vectors shape: {latent_vectors.shape}")

# Save mapping from SMILES to latent vectors
smiles_to_latent = dict(zip(valid_smiles_list, latent_vectors))
with open('../outputs/smiles_to_latent.pkl', 'wb') as f:
    pickle.dump(smiles_to_latent, f)

# ==============================
# Figure 3: VAE Training Curves
# ==============================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', color='steelblue')
ax.plot(range(1, EPOCHS+1), val_losses, label='Val Loss', color='coral')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Graph VAE Training Curves', fontsize=13)
ax.legend(fontsize=11)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('../report/images/fig3_vae_training.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_vae_training.png")

# ==============================
# Figure 4: Latent Space Visualization (t-SNE)
# ==============================
from sklearn.manifold import TSNE

print("Computing t-SNE...")
# Subsample for visualization
n_vis = min(3000, len(latent_vectors))
vis_idx = np.random.RandomState(42).choice(len(latent_vectors), n_vis, replace=False)
vis_latent = latent_vectors[vis_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(vis_latent)

# Color by molecular weight
vis_mw = []
for idx in vis_idx:
    smi = valid_smiles_list[idx]
    mol = Chem.MolFromSmiles(smi)
    if mol:
        vis_mw.append(Descriptors.MolWt(mol))
    else:
        vis_mw.append(0)

fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=vis_mw, cmap='viridis', alpha=0.5, s=8)
plt.colorbar(scatter, label='Molecular Weight (Da)')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('Graph VAE Latent Space (t-SNE)', fontsize=13)
plt.tight_layout()
plt.savefig('../report/images/fig4_latent_space_tsne.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_latent_space_tsne.png")

print("\nStep 3 complete.")
