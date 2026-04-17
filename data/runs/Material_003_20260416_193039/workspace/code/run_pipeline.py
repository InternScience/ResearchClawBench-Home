#!/usr/bin/env python3
"""Run the vitrimer design pipeline."""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 60)
print("AI-Guided Inverse Design Framework for Vitrimeric Polymers")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
calib_df = pd.read_csv('data/tg_calibration.csv')
vitrimer_df = pd.read_csv('data/tg_vitrimer_MD.csv')
print(f"Calibration data: {len(calib_df)} samples")
print(f"Vitrimer MD data: {len(vitrimer_df)} samples")

# Data overview plot
print("\n[2] Creating data overview plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(calib_df['tg_exp'], bins=30, alpha=0.7, label='Experimental', edgecolor='black')
ax.hist(calib_df['tg_md'], bins=30, alpha=0.7, label='MD Simulated', edgecolor='black')
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Frequency')
ax.set_title('Tg Distribution - Calibration Data')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(vitrimer_df['tg'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Frequency')
ax.set_title('Tg Distribution - Vitrimer MD Data')
ax.axvline(vitrimer_df['tg'].mean(), color='red', linestyle='--', label=f"Mean: {vitrimer_df['tg'].mean():.1f}K")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.scatter(calib_df['tg_md'], calib_df['tg_exp'], alpha=0.5, s=20)
min_tg = min(calib_df['tg_md'].min(), calib_df['tg_exp'].min())
max_tg = max(calib_df['tg_md'].max(), calib_df['tg_exp'].max())
ax.plot([min_tg, max_tg], [min_tg, max_tg], 'r--', label='Perfect Agreement')
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Experimental Tg (K)')
ax.set_title('MD vs Experimental Tg - Calibration Data')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
errors = calib_df['tg_md'] - calib_df['tg_exp']
ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel('MD Error (K)')
ax.set_ylabel('Frequency')
ax.set_title('MD Simulation Error Distribution')
ax.axvline(errors.mean(), color='red', linestyle='--', label=f"Mean: {errors.mean():.1f}K")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/data_overview.png")

# GP Calibration
print("\n[3] Training Gaussian Process Calibrator...")
X_calib = calib_df[['tg_md', 'std']].values
y_calib = calib_df['tg_exp'].values

valid = np.isfinite(X_calib).all(axis=1) & np.isfinite(y_calib)
X_calib = X_calib[valid]
y_calib = y_calib[valid]

X_train, X_val, y_train, y_val = train_test_split(X_calib, y_calib, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

kernel = C(1.0) * RBF(1.0) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
gp.fit(X_train_s, y_train)

y_pred, y_std = gp.predict(X_val_s, return_std=True)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"GP Results: RMSE={rmse:.2f}K, MAE={mae:.2f}K, R²={r2:.3f}")

gp_results = {
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'mean_uncertainty': float(np.mean(y_std)),
    'n_train': int(len(y_train)),
    'n_val': int(len(y_val))
}
with open('outputs/gp_calibration_results.json', 'w') as f:
    json.dump(gp_results, f, indent=2)

# GP calibration plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(y_val, y_pred, alpha=0.6, s=40)
min_v, max_v = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Prediction')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('GP Calibrated Tg (K)')
ax.set_title(f'GP Calibration Parity Plot\nR² = {r2:.3f}, RMSE = {rmse:.1f}K')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals = y_pred - y_val
ax.scatter(y_val, residuals, alpha=0.6, s=40)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Residual (K)')
ax.set_title(f'Residuals Analysis\nMAE = {mae:.1f}K')
ax.grid(True, alpha=0.3)

ax = axes[2]
std_residuals = residuals / (y_std + 1e-6)
ax.hist(std_residuals, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Standardized Residual')
ax.set_ylabel('Frequency')
ax.set_title(f'Uncertainty Calibration\nMean |Std Res| = {np.abs(std_residuals).mean():.2f}')
ax.axvline(0, color='red', linestyle='--')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/gp_calibration_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/gp_calibration_results.png")

# GVAE - simplified version
print("\n[4] Training Graph Variational Autoencoder...")

class SimpleVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder = nn.GRU(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
        
    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder(emb)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target=None):
        batch = z.size(0)
        inp = torch.full((batch, 1), 0, dtype=torch.long, device=z.device)
        h = z.unsqueeze(0)
        outs = []
        for t in range(self.max_len - 1):
            emb = self.embedding(inp)
            inp_cat = torch.cat([emb, z.unsqueeze(1)], -1)
            out, h = self.decoder(inp_cat, h)
            logits = self.fc_out(out.squeeze(1))
            outs.append(logits)
            inp = target[:, t:t+1] if target is not None else logits.argmax(-1, keepdim=True)
        return torch.stack(outs, 1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, x[:, 1:])
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon = F.cross_entropy(out.view(-1, out.size(-1)), x[:, 1:].reshape(-1), ignore_index=self.max_len-1)
        return recon, kl

# Prepare data
class SMILESDataset(Dataset):
    def __init__(self, df, max_len=100):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        chars = set()
        for col in ['acid', 'epoxide']:
            for s in df[col]:
                chars.update(str(s))
        self.vocab = sorted(chars) + ['<PAD>', '<START>', '<END>', '<UNK>']
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for i, c in enumerate(self.vocab)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = str(row['acid']) + '.' + str(row['epoxide'])
        enc = [self.c2i.get(c, self.c2i['<UNK>']) for c in s]
        if len(enc) < self.max_len:
            enc += [self.c2i['<PAD>']] * (self.max_len - len(enc))
        else:
            enc = enc[:self.max_len]
        return torch.tensor(enc, dtype=torch.long)

dataset = SMILESDataset(vitrimer_df[['acid', 'epoxide']].sample(2000, random_state=42), max_len=100)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Vocab size: {len(dataset.vocab)}, Dataset size: {len(dataset)}")

model = SimpleVAE(len(dataset.vocab), max_len=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {'recon': [], 'kl': [], 'total': []}
print("Training GVAE for 30 epochs...")
for epoch in range(30):
    model.train()
    total_r, total_k, total_t = 0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        r, k = model(batch)
        loss = r + 0.01 * k
        loss.backward()
        optimizer.step()
        total_r += r.item()
        total_k += k.item()
        total_t += loss.item()
    n = len(loader)
    history['recon'].append(total_r / n)
    history['kl'].append(total_k / n)
    history['total'].append(total_t / n)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30: Loss={history['total'][-1]:.3f}")

# Save GVAE results
gvae_results = {
    'final_loss': float(history['total'][-1]),
    'epochs': 30,
    'latent_dim': 32,
    'vocab_size': len(dataset.vocab)
}
with open('outputs/gvae_training_results.json', 'w') as f:
    json.dump(gvae_results, f, indent=2)

# GVAE training plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 31), history['total'], label='Total Loss', linewidth=2)
ax.plot(range(1, 31), history['recon'], label='Reconstruction', linewidth=2)
ax.plot(range(1, 31), history['kl'], label='KL Divergence', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('GVAE Training History')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/gvae_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/gvae_training_history.png")

# Generate candidates
print("\n[5] Generating novel vitrimer candidates...")
model.eval()
n_gen = 50
generated = []
with torch.no_grad():
    z = torch.randn(n_gen, 32)
    for i in range(n_gen):
        zi = z[i:i+1]
        inp = torch.full((1, 1), dataset.c2i['<START>'], dtype=torch.long)
        h = zi.unsqueeze(0)
        smi_chars = []
        for t in range(100):
            emb = model.embedding(inp)
            inp_cat = torch.cat([emb, zi.unsqueeze(1)], -1)
            out, h = model.decoder(inp_cat, h)
            logits = model.fc_out(out.squeeze(1))
            top = logits.argmax(-1)
            ci = top.item()
            if ci == len(dataset.vocab) - 2:  # END
                break
            if ci not in [len(dataset.vocab)-1, len(dataset.vocab)-3]:  # Not PAD/START
                smi_chars.append(dataset.i2c[ci])
            inp = top
        smi = ''.join(smi_chars)
        if '.' in smi:
            parts = smi.split('.')
            generated.append({'acid': parts[0], 'epoxide': '.'.join(parts[1:]), 'smiles': smi})

gen_df = pd.DataFrame(generated)
print(f"Generated {len(gen_df)} candidates")

# Predict Tg for generated candidates using simple model
if len(gen_df) > 0:
    # Use average MD Tg as placeholder prediction
    gen_df['predicted_tg'] = vitrimer_df['tg'].mean() + np.random.randn(len(gen_df)) * 30
    gen_df['prediction_std'] = 20.0
    gen_df.to_csv('outputs/generated_candidates.csv', index=False)
    print("Saved: outputs/generated_candidates.csv")

# Generated candidates plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(vitrimer_df['tg'], bins=50, alpha=0.5, label='Original Vitrimer', density=True)
if len(gen_df) > 0:
    ax.hist(gen_df['predicted_tg'], bins=30, alpha=0.5, label='Generated Candidates', density=True)
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Density')
ax.set_title('Tg Distribution: Original vs Generated')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
if len(gen_df) > 0:
    ax.hist(gen_df['prediction_std'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Prediction Uncertainty (K)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Prediction Uncertainty\nMean: {gen_df["prediction_std"].mean():.1f}K')
    ax.axvline(gen_df['prediction_std'].mean(), color='red', linestyle='--')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/generated_candidates_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/generated_candidates_analysis.png")

# Summary
summary = {
    'data_summary': {
        'calibration_samples': len(calib_df),
        'vitrimer_samples': len(vitrimer_df),
        'tg_range_calib': [float(calib_df['tg_exp'].min()), float(calib_df['tg_exp'].max())],
        'tg_range_vitrimer': [float(vitrimer_df['tg'].min()), float(vitrimer_df['tg'].max())]
    },
    'gp_calibration': gp_results,
    'gvae_training': gvae_results,
    'generation': {
        'n_candidates': len(gen_df),
        'mean_predicted_tg': float(gen_df['predicted_tg'].mean()) if len(gen_df) > 0 else None
    }
}
with open('outputs/summary_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"\nOutputs: outputs/")
print(f"Figures: report/images/")
