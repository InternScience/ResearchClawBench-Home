#!/usr/bin/env python3
"""Phase 3: Train VAE (uses cached features)"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings, pickle
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')

df_vit = pd.read_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'))

print("Loading cached features...")
data = np.load(os.path.join(OUT_DIR, 'vitrimer_features.npz'), allow_pickle=True)
X_combined = data['X_combined']
valid_indices = data['valid_indices']
print(f"Features: {X_combined.shape}, Valid: {len(valid_indices)}")

tg_values = df_vit.loc[valid_indices, 'tg_calibrated'].values
tg_md_values = df_vit.loc[valid_indices, 'tg'].values

scaler_vae = StandardScaler()
X_scaled = scaler_vae.fit_transform(X_combined)

# Save scaler params
np.savez(os.path.join(OUT_DIR, 'scaler_params.npz'), 
         mean=scaler_vae.mean_, scale=scaler_vae.scale_)

class GraphVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
        self.prop_pred = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.prop_pred(z)

print("\nTraining Graph VAE...")
X_tensor = torch.FloatTensor(X_scaled)
tg_mean, tg_std_v = float(tg_values.mean()), float(tg_values.std())
tg_norm = torch.FloatTensor((tg_values - tg_mean) / tg_std_v)

dataset = TensorDataset(X_tensor, tg_norm)
loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

latent_dim = 32
model = GraphVAE(X_scaled.shape[1], latent_dim=latent_dim)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

losses_hist = {'total': [], 'recon': [], 'kl': [], 'prop': []}
n_epochs = 80

for epoch in range(n_epochs):
    model.train()
    ep_loss, ep_r, ep_k, ep_p, nb = 0, 0, 0, 0, 0
    beta = min(1.0, epoch / 20.0) * 0.5
    
    for bx, bt in loader:
        opt.zero_grad()
        xr, mu, lv, tp = model(bx)
        recon = F.mse_loss(xr, bx)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        prop = F.mse_loss(tp.squeeze(), bt)
        loss = recon + beta * kl + 10.0 * prop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item(); ep_r += recon.item(); ep_k += kl.item(); ep_p += prop.item(); nb += 1
    
    for k, v in zip(['total','recon','kl','prop'], [ep_loss, ep_r, ep_k, ep_p]):
        losses_hist[k].append(v/nb)
    sched.step(ep_loss/nb)
    
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Loss={ep_loss/nb:.4f}")

print("Training complete!")
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'vae_model.pt'))

# Training loss figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(losses_hist['total'], 'b-'); axes[0].set_title('(a) Total Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[1].plot(losses_hist['recon'], 'r-', label='Recon'); axes[1].plot(losses_hist['kl'], 'g-', label='KL')
axes[1].set_title('(b) Recon & KL Loss'); axes[1].legend(); axes[1].set_xlabel('Epoch')
axes[2].plot(losses_hist['prop'], 'm-'); axes[2].set_title('(c) Property Loss'); axes[2].set_xlabel('Epoch')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig5_vae_training.png')); plt.close()
print("Saved fig5_vae_training.png")

# Latent space
model.eval()
with torch.no_grad():
    mu_all, _ = model.encode(X_tensor)
    z_all = mu_all.numpy()
    tg_pred_all = model.prop_pred(mu_all).squeeze().numpy() * tg_std_v + tg_mean

np.savez(os.path.join(OUT_DIR, 'latent_space.npz'), 
         z_all=z_all, tg_values=tg_values, tg_md_values=tg_md_values,
         tg_pred_all=tg_pred_all, tg_mean=tg_mean, tg_std_v=tg_std_v,
         valid_indices=valid_indices)

pca_lat = PCA(n_components=2)
z_2d = pca_lat.fit_transform(z_all)
r2_vae = r2_score(tg_values, tg_pred_all)
mae_vae = mean_absolute_error(tg_values, tg_pred_all)
print(f"VAE property: R²={r2_vae:.3f}, MAE={mae_vae:.1f} K")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sc = axes[0].scatter(z_2d[:,0], z_2d[:,1], c=tg_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[0].set_xlabel('Latent PC1'); axes[0].set_ylabel('Latent PC2')
axes[0].set_title('(a) Latent Space (Calibrated Tg)'); plt.colorbar(sc, ax=axes[0], label='Tg (K)')

sc2 = axes[1].scatter(z_2d[:,0], z_2d[:,1], c=tg_md_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[1].set_xlabel('Latent PC1'); axes[1].set_ylabel('Latent PC2')
axes[1].set_title('(b) Latent Space (MD Tg)'); plt.colorbar(sc2, ax=axes[1], label='MD Tg (K)')

axes[2].scatter(tg_values, tg_pred_all, c='steelblue', alpha=0.3, s=5)
lims = [min(tg_values.min(), tg_pred_all.min())-10, max(tg_values.max(), tg_pred_all.max())+10]
axes[2].plot(lims, lims, 'r--', lw=2)
axes[2].set_xlabel('True Calibrated Tg (K)'); axes[2].set_ylabel('VAE Predicted Tg (K)')
axes[2].set_title(f'(c) VAE Property Prediction\nR²={r2_vae:.3f}, MAE={mae_vae:.1f} K')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig6_latent_space.png')); plt.close()
print("Saved fig6_latent_space.png")

# Save metrics
with open(os.path.join(OUT_DIR, 'vae_metrics.json'), 'w') as f:
    json.dump({'r2': float(r2_vae), 'mae': float(mae_vae), 'latent_dim': latent_dim,
               'n_epochs': n_epochs, 'n_samples': len(valid_indices)}, f, indent=2)

print("Phase 3 complete!")
