import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import json

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load vitrimer data
md = pd.read_csv('outputs/vitrimer_calibrated.csv')

# Use subset for training to keep it manageable
md_sample = md.sample(n=min(4000, len(md)), random_state=42).reset_index(drop=True)

# Compute fingerprints and descriptors
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fp = np.zeros(512)
        desc = np.zeros(10)
    else:
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 512))
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.HeavyAtomCount(mol),
        ])
    return np.concatenate([fp, desc])

acid_features = np.array([get_features(s) for s in md_sample['acid']])
epoxide_features = np.array([get_features(s) for s in md_sample['epoxide']])
tg_values = md_sample['tg_calibrated'].values.reshape(-1, 1)

# Standardize
scaler_acid = StandardScaler()
scaler_epoxide = StandardScaler()
scaler_tg = StandardScaler()

acid_features_s = scaler_acid.fit_transform(acid_features)
epoxide_features_s = scaler_epoxide.fit_transform(epoxide_features)
tg_values_s = scaler_tg.fit_transform(tg_values).flatten()

# Dataset
class VitrimerDataset(Dataset):
    def __init__(self, acid_feat, epoxide_feat, tg):
        self.acid_feat = torch.tensor(acid_feat, dtype=torch.float32)
        self.epoxide_feat = torch.tensor(epoxide_feat, dtype=torch.float32)
        self.tg = torch.tensor(tg, dtype=torch.float32)
    
    def __len__(self):
        return len(self.tg)
    
    def __getitem__(self, idx):
        return self.acid_feat[idx], self.epoxide_feat[idx], self.tg[idx]

dataset = VitrimerDataset(acid_features_s, epoxide_features_s, tg_values_s)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# VAE Model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, z_acid, z_epoxide):
        z = torch.cat([z_acid, z_epoxide], dim=1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)

class VitrimerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.acid_encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.epoxide_encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.acid_decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.epoxide_decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.predictor = PropertyPredictor(latent_dim, hidden_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, acid_feat, epoxide_feat):
        acid_mu, acid_logvar = self.acid_encoder(acid_feat)
        epoxide_mu, epoxide_logvar = self.epoxide_encoder(epoxide_feat)
        
        z_acid = self.reparameterize(acid_mu, acid_logvar)
        z_epoxide = self.reparameterize(epoxide_mu, epoxide_logvar)
        
        acid_recon = self.acid_decoder(z_acid)
        epoxide_recon = self.epoxide_decoder(z_epoxide)
        tg_pred = self.predictor(z_acid, z_epoxide)
        
        return acid_recon, epoxide_recon, tg_pred, acid_mu, acid_logvar, epoxide_mu, epoxide_logvar

input_dim = acid_features_s.shape[1]
hidden_dim = 256
latent_dim = 32

model = VitrimerVAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Loss function
def loss_function(acid_recon, epoxide_recon, tg_pred, acid_feat, epoxide_feat, tg,
                  acid_mu, acid_logvar, epoxide_mu, epoxide_logvar, beta=0.01):
    recon_loss = F.mse_loss(acid_recon, acid_feat, reduction='sum') + \
                 F.mse_loss(epoxide_recon, epoxide_feat, reduction='sum')
    
    kld_loss = -0.5 * torch.sum(1 + acid_logvar - acid_mu.pow(2) - acid_logvar.exp()) + \
               -0.5 * torch.sum(1 + epoxide_logvar - epoxide_mu.pow(2) - epoxide_logvar.exp())
    
    pred_loss = F.mse_loss(tg_pred, tg, reduction='sum')
    
    return recon_loss + beta * kld_loss + pred_loss, recon_loss, kld_loss, pred_loss

# Training
def train_epoch(model, loader, optimizer, beta=0.01):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    total_pred = 0
    for acid_feat, epoxide_feat, tg in loader:
        acid_feat = acid_feat.to(device)
        epoxide_feat = epoxide_feat.to(device)
        tg = tg.to(device)
        
        optimizer.zero_grad()
        acid_recon, epoxide_recon, tg_pred, acid_mu, acid_logvar, epoxide_mu, epoxide_logvar = model(acid_feat, epoxide_feat)
        loss, recon, kld, pred = loss_function(acid_recon, epoxide_recon, tg_pred, acid_feat, epoxide_feat, tg,
                                               acid_mu, acid_logvar, epoxide_mu, epoxide_logvar, beta)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
        total_pred += pred.item()
    
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kld/n, total_pred/n

def test_epoch(model, loader, beta=0.01):
    model.eval()
    total_loss = 0
    all_tg_pred = []
    all_tg_true = []
    with torch.no_grad():
        for acid_feat, epoxide_feat, tg in loader:
            acid_feat = acid_feat.to(device)
            epoxide_feat = epoxide_feat.to(device)
            tg = tg.to(device)
            
            acid_recon, epoxide_recon, tg_pred, acid_mu, acid_logvar, epoxide_mu, epoxide_logvar = model(acid_feat, epoxide_feat)
            loss, _, _, _ = loss_function(acid_recon, epoxide_recon, tg_pred, acid_feat, epoxide_feat, tg,
                                          acid_mu, acid_logvar, epoxide_mu, epoxide_logvar, beta)
            total_loss += loss.item()
            all_tg_pred.append(tg_pred.cpu())
            all_tg_true.append(tg.cpu())
    
    all_tg_pred = torch.cat(all_tg_pred).numpy()
    all_tg_true = torch.cat(all_tg_true).numpy()
    tg_rmse = np.sqrt(np.mean((all_tg_pred - all_tg_true)**2))
    
    return total_loss/len(loader.dataset), tg_rmse, all_tg_pred, all_tg_true

# Train
num_epochs = 150
best_test_loss = float('inf')
train_losses = []
test_losses = []
tg_rmses = []

for epoch in range(num_epochs):
    beta = min(0.01, epoch / 100 * 0.01)  # KL annealing
    train_loss, train_recon, train_kld, train_pred = train_epoch(model, train_loader, optimizer, beta)
    test_loss, test_rmse, _, _ = test_epoch(model, test_loader, beta)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    tg_rmses.append(test_rmse)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'outputs/vae_model.pt')
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f} (recon={train_recon:.4f}, kld={train_kld:.4f}, pred={train_pred:.4f}), Test Loss={test_loss:.4f}, Tg RMSE={test_rmse:.4f}")

print("Training complete.")

# Load best model
model.load_state_dict(torch.load('outputs/vae_model.pt'))
model.eval()

# Evaluate on test set
test_loss, test_rmse, test_pred, test_true = test_epoch(model, test_loader, beta=0.01)
test_pred_tg = scaler_tg.inverse_transform(test_pred.reshape(-1, 1)).flatten()
test_true_tg = scaler_tg.inverse_transform(test_true.reshape(-1, 1)).flatten()
test_rmse_tg = np.sqrt(np.mean((test_pred_tg - test_true_tg)**2))
test_r2 = 1 - np.sum((test_pred_tg - test_true_tg)**2) / np.sum((test_true_tg - test_true_tg.mean())**2)

print(f"\nTest Tg RMSE: {test_rmse_tg:.2f} K, R²: {test_r2:.3f}")

# Save metrics
vae_metrics = {
    'test_rmse_scaled': float(test_rmse),
    'test_rmse_K': float(test_rmse_tg),
    'test_r2': float(test_r2),
    'best_test_loss': float(best_test_loss),
    'latent_dim': latent_dim,
    'hidden_dim': hidden_dim,
}
with open('outputs/vae_metrics.json', 'w') as f:
    json.dump(vae_metrics, f, indent=2)

# Extract latent representations for all data
model.eval()
all_acid_mu = []
all_epoxide_mu = []
all_tg = []
all_acid_smiles = []
all_epoxide_smiles = []

with torch.no_grad():
    for i in range(0, len(md_sample), 128):
        acid_feat = torch.tensor(acid_features_s[i:i+128], dtype=torch.float32).to(device)
        epoxide_feat = torch.tensor(epoxide_features_s[i:i+128], dtype=torch.float32).to(device)
        acid_mu, _ = model.acid_encoder(acid_feat)
        epoxide_mu, _ = model.epoxide_encoder(epoxide_feat)
        all_acid_mu.append(acid_mu.cpu().numpy())
        all_epoxide_mu.append(epoxide_mu.cpu().numpy())
        all_tg.append(tg_values_s[i:i+128])
        all_acid_smiles.extend(md_sample['acid'].values[i:i+128])
        all_epoxide_smiles.extend(md_sample['epoxide'].values[i:i+128])

all_acid_mu = np.vstack(all_acid_mu)
all_epoxide_mu = np.vstack(all_epoxide_mu)
all_tg = np.concatenate(all_tg)

# Save latent space
latent_df = pd.DataFrame({
    'acid_smiles': all_acid_smiles,
    'epoxide_smiles': all_epoxide_smiles,
    'tg_calibrated': md_sample['tg_calibrated'].values[:len(all_acid_mu)],
})
for i in range(latent_dim):
    latent_df[f'acid_z{i}'] = all_acid_mu[:, i]
    latent_df[f'epoxide_z{i}'] = all_epoxide_mu[:, i]
latent_df.to_csv('outputs/latent_representations.csv', index=False)

print(f"Saved latent representations for {len(latent_df)} molecules.")

# Figure 3: VAE Training and Performance
def create_figures():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training curves
    ax = axes[0]
    ax.plot(train_losses, label='Train Loss', color='steelblue')
    ax.plot(test_losses, label='Test Loss', color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('VAE Training Curves')
    ax.legend()
    ax.set_yscale('log')
    
    # Tg prediction parity
    ax = axes[1]
    ax.scatter(test_true_tg, test_pred_tg, alpha=0.5, c='darkgreen', edgecolors='k', linewidths=0.3)
    ax.plot([test_true_tg.min(), test_true_tg.max()], [test_true_tg.min(), test_true_tg.max()], 'r--', lw=1.5)
    ax.set_xlabel('True Calibrated Tg (K)')
    ax.set_ylabel('Predicted Tg (K)')
    ax.set_title(f'VAE Property Prediction (R²={test_r2:.3f})')
    
    # Latent space PCA visualization
    from sklearn.decomposition import PCA
    combined_z = np.hstack([all_acid_mu, all_epoxide_mu])
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(combined_z)
    tg_colors = md_sample['tg_calibrated'].values[:len(z_pca)]
    
    ax = axes[2]
    scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=tg_colors, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Latent Space PCA (colored by Tg)')
    plt.colorbar(scatter, ax=ax, label='Calibrated Tg (K)')
    
    plt.tight_layout()
    plt.savefig('report/images/fig03_vae_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")

create_figures()

# Save model config and scalers
model_config = {
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'latent_dim': latent_dim,
}
import pickle
with open('outputs/vae_config.pkl', 'wb') as f:
    pickle.dump({'config': model_config, 'scaler_acid': scaler_acid, 'scaler_epoxide': scaler_epoxide, 'scaler_tg': scaler_tg}, f)

print("VAE training and evaluation complete.")
