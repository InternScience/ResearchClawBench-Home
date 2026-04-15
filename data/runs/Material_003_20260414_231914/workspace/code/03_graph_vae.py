"""
Step 3: Graph VAE for vitrimer inverse design.
Simplified molecular VAE using fingerprint-based encoding for latent space.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import pickle

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load calibrated vitrimer data
vit = pd.read_csv('outputs/vitrimer_calibrated.csv')
vit_desc = pd.read_csv('outputs/vitrimer_descriptors.csv')

print(f"Vitrimer samples: {len(vit)}")

# Compute Morgan fingerprints for acid and epoxide
def get_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

print("Computing fingerprints...")
acid_fps = np.array([get_fingerprint(s) for s in vit['acid']])
epox_fps = np.array([get_fingerprint(s) for s in vit['epoxide']])
combined_fps = np.hstack([acid_fps, epox_fps])  # 2048 bits

print(f"Combined fingerprint shape: {combined_fps.shape}")

# Convert to float tensor
X = torch.FloatTensor(combined_fps)
y_tg = torch.FloatTensor(vit['tg_calibrated'].values)
y_std = torch.FloatTensor(vit['tg_calibrated_std'].values)

# VAE Architecture
class MolecularVAE(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=64, hidden_dim=512):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Property predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict_tg(self, z):
        return self.predictor(z).squeeze(-1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        tg_pred = self.predict_tg(mu)  # Use mu for deterministic prediction
        return x_recon, mu, logvar, tg_pred

# Loss function
def vae_loss(x, x_recon, mu, logvar, tg_pred, tg_true, beta=0.001, lambda_prop=1.0):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    prop_loss = nn.functional.mse_loss(tg_pred, tg_true)
    return recon_loss + beta * kl_loss + lambda_prop * prop_loss, recon_loss.item(), kl_loss.item(), prop_loss.item()

# Training
device = torch.device('cpu')
model = MolecularVAE(input_dim=2048, latent_dim=64, hidden_dim=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

dataset = TensorDataset(X, y_tg)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

n_epochs = 100
losses = []
print("Training VAE...")
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    epoch_prop = 0
    for batch_x, batch_tg in dataloader:
        optimizer.zero_grad()
        x_recon, mu, logvar, tg_pred = model(batch_x)
        loss, recon_l, kl_l, prop_l = vae_loss(batch_x, x_recon, mu, logvar, tg_pred, batch_tg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_recon += recon_l
        epoch_kl += kl_l
        epoch_prop += prop_l
    
    avg_loss = epoch_loss / len(dataloader)
    scheduler.step(avg_loss)
    losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.2f}, Recon: {epoch_recon/len(dataloader):.2f}, KL: {epoch_kl/len(dataloader):.2f}, Prop: {epoch_prop/len(dataloader):.2f}")

# Encode all vitrimers to latent space
model.eval()
with torch.no_grad():
    mu_all, logvar_all = model.encode(X)
    z_all = mu_all.numpy()
    tg_pred_all = model.predict_tg(mu_all).numpy()
    x_recon_all = model.decode(mu_all).numpy()

# Reconstruction accuracy
recon_acc = np.mean((x_recon_all > 0.5).astype(float) == combined_fps)
print(f"\nReconstruction accuracy: {recon_acc:.4f}")

# Tg prediction from latent space
tg_mae = mean_absolute_error(vit['tg_calibrated'].values, tg_pred_all)
tg_r2 = r2_score(vit['tg_calibrated'].values, tg_pred_all)
print(f"Tg prediction from latent space - MAE: {tg_mae:.2f} K, R2: {tg_r2:.4f}")

# Save latent representations
latent_df = pd.DataFrame(z_all, columns=[f'z_{i}' for i in range(64)])
latent_df['tg_calibrated'] = vit['tg_calibrated'].values
latent_df['tg_predicted'] = tg_pred_all
latent_df['acid'] = vit['acid'].values
latent_df['epoxide'] = vit['epoxide'].values
latent_df.to_csv('outputs/vitrimer_latent.csv', index=False)

# Save model
torch.save(model.state_dict(), 'outputs/vae_model.pt')

# Save results
vae_results = {
    'reconstruction_accuracy': float(recon_acc),
    'tg_prediction_mae': float(tg_mae),
    'tg_prediction_r2': float(tg_r2),
    'latent_dim': 64,
    'n_epochs': n_epochs,
    'final_loss': float(losses[-1]),
}
with open('outputs/vae_results.json', 'w') as f:
    json.dump(vae_results, f, indent=2)

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(16, 11))

# Plot 1: Training loss
ax = axes[0, 0]
ax.plot(losses, c='steelblue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('VAE Training Loss')

# Plot 2: Latent space PCA colored by Tg
ax = axes[0, 1]
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_all)
sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=vit['tg_calibrated'].values, s=3, alpha=0.5, cmap='viridis')
plt.colorbar(sc, ax=ax, label='Calibrated Tg (K)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Latent Space (PCA) Colored by Tg')

# Plot 3: Predicted vs calibrated Tg from latent space
ax = axes[0, 2]
ax.scatter(vit['tg_calibrated'].values, tg_pred_all, s=5, alpha=0.3, c='steelblue')
lims = [vit['tg_calibrated'].min() - 10, vit['tg_calibrated'].max() + 10]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('GP Calibrated Tg (K)')
ax.set_ylabel('VAE Predicted Tg (K)')
ax.set_title(f'VAE Tg Prediction (R2={tg_r2:.3f}, MAE={tg_mae:.1f} K)')

# Plot 4: Reconstruction accuracy by bit
ax = axes[1, 0]
bit_acc = (x_recon_all > 0.5).astype(float) == combined_fps
bit_acc_mean = bit_acc.mean(axis=0)
ax.hist(bit_acc_mean, bins=50, alpha=0.7, color='seagreen')
ax.set_xlabel('Per-bit Reconstruction Accuracy')
ax.set_ylabel('Count')
ax.set_title(f'Mean Reconstruction Accuracy: {recon_acc:.3f}')

# Plot 5: Latent space density
ax = axes[1, 1]
ax.hist2d(z_pca[:, 0], z_pca[:, 1], bins=50, cmap='Blues')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Latent Space Density')

# Plot 6: Tg prediction residuals
ax = axes[1, 2]
residuals = tg_pred_all - vit['tg_calibrated'].values
ax.hist(residuals, bins=50, alpha=0.7, color='coral')
ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('VAE Tg Prediction Residual (K)')
ax.set_ylabel('Count')
ax.set_title(f'Tg Residuals (Mean: {residuals.mean():.1f} K, Std: {residuals.std():.1f} K)')

plt.tight_layout()
plt.savefig('report/images/vae_latent_space.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVAE training complete.")
print("Plots saved to report/images/vae_latent_space.png")
