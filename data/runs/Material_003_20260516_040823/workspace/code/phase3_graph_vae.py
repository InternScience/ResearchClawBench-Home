#!/usr/bin/env python3
"""
Phase 3: Graph Variational Autoencoder for Vitrimer Molecules (Fixed)
Convert SMILES to molecular graphs, train a graph VAE with fingerprint reconstruction,
and analyze the latent space.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260516_040823')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
IMAGES_DIR = WORKSPACE / 'report' / 'images'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)

# ============================================================
# Load calibrated vitrimer data
# ============================================================
print("Loading data...")
vitrimer_df = pd.read_csv(OUTPUTS_DIR / 'vitrimer_calibrated.csv')
print(f"Vitrimer data: {len(vitrimer_df)} entries")

# ============================================================
# SMILES to Molecular Graph Conversion
# ============================================================
ATOM_TYPES = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', '*']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

def get_atom_features(mol):
    """Get rich atom features for each atom in the molecule."""
    features = []
    for atom in mol.GetAtoms():
        feat = []
        symbol = atom.GetSymbol()
        atom_type = np.zeros(NUM_ATOM_TYPES)
        if symbol in ATOM_TO_IDX:
            atom_type[ATOM_TO_IDX[symbol]] = 1.0
        else:
            atom_type[0] = 1.0
        feat.extend(atom_type)
        
        degree = atom.GetDegree()
        deg_feat = np.zeros(5)
        deg_feat[min(degree, 4)] = 1.0
        feat.extend(deg_feat)
        
        valence = atom.GetImplicitValence()
        val_feat = np.zeros(5)
        val_feat[min(valence, 4)] = 1.0
        feat.extend(val_feat)
        
        feat.append(float(atom.GetIsAromatic()))
        feat.append(float(atom.IsInRing()))
        
        features.append(feat)
    return np.array(features, dtype=np.float32)

NUM_NODE_FEATURES = NUM_ATOM_TYPES + 5 + 5 + 1 + 1

def get_morgan_fingerprint(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp, dtype=np.float32)

def smiles_to_graph_with_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    x = torch.tensor(get_atom_features(mol), dtype=torch.float32)
    
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])
    
    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    fp = get_morgan_fingerprint(smiles)
    fp_tensor = torch.tensor(fp, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, num_nodes=len(x), fp=fp_tensor)

print("Converting SMILES to molecular graphs...")
acid_graphs = []
epoxide_graphs = []
valid_indices = []
for i, row in vitrimer_df.iterrows():
    g_a = smiles_to_graph_with_fp(row['acid'])
    g_e = smiles_to_graph_with_fp(row['epoxide'])
    if g_a is not None and g_e is not None:
        acid_graphs.append(g_a)
        epoxide_graphs.append(g_e)
        valid_indices.append(i)

print(f"Valid pairs: {len(valid_indices)} / {len(vitrimer_df)}")

# ============================================================
# Graph VAE with fingerprint reconstruction
# ============================================================
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        mu = self.fc_mu(x_pooled)
        logvar = self.fc_logvar(x_pooled)
        return mu, logvar

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, fp_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, fp_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return torch.sigmoid(self.fc_out(h))

class GraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, fp_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, fp_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    def encode(self, data):
        mu, logvar = self.encoder(data)
        return mu

# ============================================================
# Prepare data
# ============================================================
print("Preparing training data...")
all_graphs = acid_graphs + epoxide_graphs
print(f"Total graphs: {len(all_graphs)}")

batch_size = 256
dataloader = DataLoader(all_graphs, batch_size=batch_size, shuffle=True)

# ============================================================
# Train VAE
# ============================================================
print("Training Graph VAE...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

in_dim = NUM_NODE_FEATURES
hidden_dim = 128
latent_dim = 32
fp_dim = 512

model = GraphVAE(in_dim, hidden_dim, latent_dim, fp_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def vae_loss(recon, data, mu, logvar, beta=0.01):
    fp_target = data.fp.view(-1, fp_dim)
    recon_loss = F.binary_cross_entropy(recon, fp_target, reduction='sum') / data.num_graphs
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.num_graphs
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

n_epochs = 80
train_losses = []
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, z = model(batch)
        loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_l.item()
        total_kl += kl_l.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    train_losses.append(avg_loss)
    if (epoch + 1) % 16 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Recon: {total_recon/n_batches:.4f}, KL: {total_kl/n_batches:.4f}")

print(f"VAE training complete. Final loss: {train_losses[-1]:.4f}")

# ============================================================
# Encode all molecules
# ============================================================
print("Encoding molecules to latent space...")
model.eval()
acid_latent_list = []
epoxide_latent_list = []
with torch.no_grad():
    for g in DataLoader(acid_graphs, batch_size=512, shuffle=False):
        g = g.to(device)
        mu = model.encode(g)
        acid_latent_list.append(mu.cpu().numpy())
    for g in DataLoader(epoxide_graphs, batch_size=512, shuffle=False):
        g = g.to(device)
        mu = model.encode(g)
        epoxide_latent_list.append(mu.cpu().numpy())

acid_latent = np.vstack(acid_latent_list)
epoxide_latent = np.vstack(epoxide_latent_list)
combined_latent = np.concatenate([acid_latent, epoxide_latent], axis=1)
print(f"Acid latent: {acid_latent.shape}, Epoxide latent: {epoxide_latent.shape}")
print(f"Combined latent: {combined_latent.shape}")

# ============================================================
# Train property predictor
# ============================================================
print("Training property predictor on latent space...")
valid_tg = vitrimer_df.iloc[valid_indices]['tg_calibrated'].values

X_train, X_test, y_train, y_test = train_test_split(
    combined_latent, valid_tg, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Property Predictor Performance:")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²:  {test_r2:.4f}")
print(f"  Test MAE: {test_mae:.2f} K")
print(f"  Test RMSE: {test_rmse:.2f} K")

# ============================================================
# Save artifacts
# ============================================================
np.savez(OUTPUTS_DIR / 'latent_representations.npz',
         acid_latent=acid_latent,
         epoxide_latent=epoxide_latent,
         combined_latent=combined_latent,
         valid_indices=np.array(valid_indices),
         calibrated_tg=valid_tg)

torch.save({
    'model_state_dict': model.state_dict(),
    'in_dim': in_dim,
    'hidden_dim': hidden_dim,
    'latent_dim': latent_dim,
    'fp_dim': fp_dim,
}, OUTPUTS_DIR / 'graph_vae_model.pt')

with open(OUTPUTS_DIR / 'property_predictor.pkl', 'wb') as f:
    pickle.dump(rf, f)

vae_metrics = {
    'n_valid_graphs': len(valid_indices),
    'total_graphs': len(all_graphs),
    'latent_dim': latent_dim,
    'property_predictor_train_r2': float(train_r2),
    'property_predictor_test_r2': float(test_r2),
    'property_predictor_test_mae': float(test_mae),
    'property_predictor_test_rmse': float(test_rmse),
    'final_vae_loss': float(train_losses[-1]),
}

with open(OUTPUTS_DIR / 'vae_metrics.json', 'w') as f:
    json.dump(vae_metrics, f, indent=2)

# ============================================================
# Figure 5: Latent Space Visualization
# ============================================================
print("Generating latent space visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

pca = PCA(n_components=2)
combined_pca = pca.fit_transform(combined_latent)
sc = axes[0, 0].scatter(combined_pca[:, 0], combined_pca[:, 1], 
                         c=valid_tg, cmap='RdYlBu_r', s=3, alpha=0.6)
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0, 0].set_title('A: PCA of Combined Latent Space')
cbar = plt.colorbar(sc, ax=axes[0, 0])
cbar.set_label('Calibrated Tg (K)')

tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=500)
combined_tsne = tsne.fit_transform(combined_latent)
sc = axes[0, 1].scatter(combined_tsne[:, 0], combined_tsne[:, 1], 
                         c=valid_tg, cmap='RdYlBu_r', s=3, alpha=0.6)
axes[0, 1].set_xlabel('t-SNE 1')
axes[0, 1].set_ylabel('t-SNE 2')
axes[0, 1].set_title('B: t-SNE of Combined Latent Space')
cbar = plt.colorbar(sc, ax=axes[0, 1])
cbar.set_label('Calibrated Tg (K)')

axes[1, 0].scatter(y_test, y_pred_test, c='steelblue', s=8, alpha=0.4)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1.5)
axes[1, 0].set_xlabel('True Calibrated Tg (K)')
axes[1, 0].set_ylabel('Predicted Tg (K)')
axes[1, 0].set_title(f'C: Latent Predictor Parity (Test R²={test_r2:.3f}, MAE={test_mae:.1f} K)')

axes[1, 1].plot(train_losses, 'b-', linewidth=1)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('D: VAE Training Loss')

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure5_vae_latent_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure5_vae_latent_space.png")

# ============================================================
# Figure 6: Analysis plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].hist(valid_tg, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 0].axvline(valid_tg.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {valid_tg.mean():.1f} K')
axes[0, 0].set_xlabel('Calibrated Tg (K)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('A: Calibrated Tg Distribution')
axes[0, 0].legend()

importances = rf.feature_importances_
axes[0, 1].bar(range(len(importances)), importances, color='teal', alpha=0.8)
axes[0, 1].set_xlabel('Latent Dimension Index')
axes[0, 1].set_ylabel('Importance')
axes[0, 1].set_title('B: Latent Feature Importance for Tg Prediction')

axes[1, 0].scatter(acid_latent[:, 0], acid_latent[:, 1], c='coral', s=3, alpha=0.3, label='Acid')
axes[1, 0].scatter(epoxide_latent[:, 0], epoxide_latent[:, 1], c='steelblue', s=3, alpha=0.3, label='Epoxide')
axes[1, 0].set_xlabel('Latent Dim 1')
axes[1, 0].set_ylabel('Latent Dim 2')
axes[1, 0].set_title('C: Acid vs Epoxide Latent Distribution')
axes[1, 0].legend(markerscale=5)

# Mean Tg by PCA region
tg_matrix = np.zeros((10, 10))
count_matrix = np.zeros((10, 10))
for i in range(len(valid_tg)):
    xi = combined_pca[:, 0][i]
    yi = combined_pca[:, 1][i]
    bx = min(int((xi - combined_pca[:, 0].min()) / (combined_pca[:, 0].max() - combined_pca[:, 0].min() + 1e-8) * 10), 9)
    by = min(int((yi - combined_pca[:, 1].min()) / (combined_pca[:, 1].max() - combined_pca[:, 1].min() + 1e-8) * 10), 9)
    tg_matrix[by, bx] += valid_tg[i]
    count_matrix[by, bx] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    tg_matrix = np.where(count_matrix > 3, tg_matrix / count_matrix, np.nan)

im = axes[1, 1].imshow(tg_matrix, cmap='RdYlBu_r', aspect='auto', origin='lower')
axes[1, 1].set_xlabel('PCA1 Bin')
axes[1, 1].set_ylabel('PCA2 Bin')
axes[1, 1].set_title('D: Mean Tg by PCA Region')
plt.colorbar(im, ax=axes[1, 1], label='Mean Tg (K)')

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure6_tg_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure6_tg_analysis.png")

print("\nPhase 3 complete!")
