#!/usr/bin/env python3
"""
Phase 3: Graph Variational Autoencoder for Vitrimer Generation
Phase 4: Inverse Design & Candidate Selection

Uses molecular graph features (fingerprints) as input to a VAE
for generating new vitrimer chemistries.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Chem import Draw
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (10, 8),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')

# ============================================================
# Load and prepare data
# ============================================================
print("=" * 60)
print("PHASE 3: Graph VAE for Vitrimer Generation")
print("=" * 60)

df_vit = pd.read_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'))
df_cal = pd.read_csv(os.path.join(BASE, 'data', 'tg_calibration.csv'))

FP_BITS = 256

def smiles_to_fingerprint(smiles, radius=2, nbits=FP_BITS):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)

def get_descriptors(smiles):
    """Get basic molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 8
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
    ]

# Build graph-based features for all vitrimers
print("\nBuilding molecular graph features for vitrimers...")
acid_fps = []
epoxide_fps = []
acid_descs = []
epoxide_descs = []
valid_indices = []

for idx, row in df_vit.iterrows():
    try:
        afp = smiles_to_fingerprint(row['acid'])
        efp = smiles_to_fingerprint(row['epoxide'])
        ad = get_descriptors(row['acid'])
        ed = get_descriptors(row['epoxide'])
        acid_fps.append(afp)
        epoxide_fps.append(efp)
        acid_descs.append(ad)
        epoxide_descs.append(ed)
        valid_indices.append(idx)
    except:
        pass

acid_fps = np.array(acid_fps)
epoxide_fps = np.array(epoxide_fps)
acid_descs = np.array(acid_descs)
epoxide_descs = np.array(epoxide_descs)

# Combined features: acid_fp + epoxide_fp + acid_desc + epoxide_desc
X_combined = np.column_stack([acid_fps, epoxide_fps, acid_descs, epoxide_descs])
tg_values = df_vit.loc[valid_indices, 'tg_calibrated'].values
tg_md_values = df_vit.loc[valid_indices, 'tg'].values

print(f"Combined feature matrix: {X_combined.shape}")
print(f"Valid samples: {len(valid_indices)}")

# Scale features
scaler_vae = StandardScaler()
X_scaled = scaler_vae.fit_transform(X_combined)

# ============================================================
# Graph VAE Architecture
# ============================================================
class GraphVAE(nn.Module):
    """
    Variational Autoencoder for molecular graph features.
    Uses molecular fingerprints as a graph-level representation.
    Includes a property predictor branch for conditional generation.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
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
        )
        
        # Property predictor (from latent space)
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
    
    def predict_property(self, z):
        return self.property_predictor(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        tg_pred = self.predict_property(z)
        return x_recon, mu, logvar, tg_pred

def vae_loss(x_recon, x, mu, logvar, tg_pred, tg_true, beta=1.0, gamma=10.0):
    """Combined VAE loss: reconstruction + KL divergence + property prediction."""
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    prop_loss = F.mse_loss(tg_pred.squeeze(), tg_true, reduction='mean')
    return recon_loss + beta * kl_loss + gamma * prop_loss, recon_loss, kl_loss, prop_loss

# ============================================================
# Train VAE
# ============================================================
print("\nTraining Graph VAE...")

# Prepare data
X_tensor = torch.FloatTensor(X_scaled)
tg_tensor = torch.FloatTensor(tg_values)

# Normalize Tg for training
tg_mean = tg_values.mean()
tg_std_val = tg_values.std()
tg_normalized = (tg_values - tg_mean) / tg_std_val
tg_norm_tensor = torch.FloatTensor(tg_normalized)

dataset = TensorDataset(X_tensor, tg_norm_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

input_dim = X_scaled.shape[1]
latent_dim = 32
model = GraphVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

n_epochs = 150
train_losses = []
recon_losses = []
kl_losses = []
prop_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    epoch_prop = 0
    n_batches = 0
    
    # Anneal beta
    beta = min(1.0, epoch / 30.0) * 0.5
    
    for batch_x, batch_tg in dataloader:
        optimizer.zero_grad()
        x_recon, mu, logvar, tg_pred = model(batch_x)
        loss, recon, kl, prop = vae_loss(x_recon, batch_x, mu, logvar, tg_pred, batch_tg, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon += recon.item()
        epoch_kl += kl.item()
        epoch_prop += prop.item()
        n_batches += 1
    
    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)
    recon_losses.append(epoch_recon / n_batches)
    kl_losses.append(epoch_kl / n_batches)
    prop_losses.append(epoch_prop / n_batches)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Recon={epoch_recon/n_batches:.4f}, "
              f"KL={epoch_kl/n_batches:.4f}, Prop={epoch_prop/n_batches:.4f}")

print("Training complete!")

# --- Figure 5: Training Loss ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('(a) Total Training Loss')
axes[0].set_yscale('log')

axes[1].plot(recon_losses, 'r-', label='Reconstruction', linewidth=1.5)
axes[1].plot(kl_losses, 'g-', label='KL Divergence', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('(b) Reconstruction & KL Loss')
axes[1].legend()

axes[2].plot(prop_losses, 'm-', linewidth=1.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Property Loss')
axes[2].set_title('(c) Tg Property Prediction Loss')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_vae_training.png'))
plt.close()
print("Saved fig5_vae_training.png")

# ============================================================
# Latent Space Analysis
# ============================================================
print("\nAnalyzing latent space...")
model.eval()
with torch.no_grad():
    mu_all, logvar_all = model.encode(X_tensor)
    z_all = mu_all.numpy()
    tg_pred_all = model.predict_property(mu_all).squeeze().numpy()
    tg_pred_all = tg_pred_all * tg_std_val + tg_mean

# PCA for visualization
pca_latent = PCA(n_components=2)
z_2d = pca_latent.fit_transform(z_all)
print(f"Latent space PCA: {pca_latent.explained_variance_ratio_[:2].sum()*100:.1f}% variance explained")

# --- Figure 6: Latent Space Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 6a: Colored by calibrated Tg
scatter = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=tg_values, cmap='RdYlBu_r', 
                          alpha=0.3, s=5)
axes[0].set_xlabel('Latent PC1')
axes[0].set_ylabel('Latent PC2')
axes[0].set_title('(a) Latent Space Colored by Calibrated Tg')
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Calibrated Tg (K)')

# 6b: Colored by MD Tg
scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=tg_md_values, cmap='RdYlBu_r', 
                           alpha=0.3, s=5)
axes[1].set_xlabel('Latent PC1')
axes[1].set_ylabel('Latent PC2')
axes[1].set_title('(b) Latent Space Colored by MD Tg')
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('MD Tg (K)')

# 6c: Property prediction accuracy
axes[2].scatter(tg_values, tg_pred_all, c='steelblue', alpha=0.3, s=5)
min_v = min(tg_values.min(), tg_pred_all.min()) - 10
max_v = max(tg_values.max(), tg_pred_all.max()) + 10
axes[2].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
r2 = r2_score(tg_values, tg_pred_all)
mae = mean_absolute_error(tg_values, tg_pred_all)
axes[2].set_xlabel('True Calibrated Tg (K)')
axes[2].set_ylabel('VAE-Predicted Tg (K)')
axes[2].set_title(f'(c) VAE Property Prediction\nR²={r2:.3f}, MAE={mae:.1f} K')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_latent_space.png'))
plt.close()
print("Saved fig6_latent_space.png")

# ============================================================
# PHASE 4: Inverse Design - Generate New Candidates
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Inverse Design & Candidate Generation")
print("=" * 60)

# Strategy: Sample from latent space regions with desired Tg
# Target Tg ranges for different applications
target_ranges = {
    'High Tg (>480 K)': (480, 600),
    'Medium-High Tg (420-480 K)': (420, 480),
    'Medium Tg (360-420 K)': (360, 420),
}

# Find latent vectors for existing vitrimers in each target range
all_generated_candidates = []

for target_name, (tg_low, tg_high) in target_ranges.items():
    print(f"\n--- Target: {target_name} ---")
    
    # Find existing vitrimers in target range
    mask = (tg_values >= tg_low) & (tg_values <= tg_high)
    n_in_range = mask.sum()
    print(f"  Existing vitrimers in range: {n_in_range}")
    
    if n_in_range < 5:
        print(f"  Too few samples, skipping...")
        continue
    
    # Get latent vectors for vitrimers in target range
    z_target = z_all[mask]
    
    # Generate new candidates by:
    # 1. Interpolating between existing latent vectors
    # 2. Perturbing existing latent vectors
    # 3. Random sampling near target region
    
    n_generate = 500
    generated_z = []
    
    # Method 1: Interpolation
    for _ in range(n_generate // 3):
        idx1, idx2 = np.random.choice(len(z_target), 2, replace=False)
        alpha = np.random.uniform(0.2, 0.8)
        z_interp = alpha * z_target[idx1] + (1 - alpha) * z_target[idx2]
        generated_z.append(z_interp)
    
    # Method 2: Perturbation
    for _ in range(n_generate // 3):
        idx = np.random.choice(len(z_target))
        noise = np.random.randn(latent_dim) * 0.3
        z_perturb = z_target[idx] + noise
        generated_z.append(z_perturb)
    
    # Method 3: Random sampling near centroid
    centroid = z_target.mean(axis=0)
    cov = np.cov(z_target.T)
    for _ in range(n_generate // 3):
        z_random = np.random.multivariate_normal(centroid, cov * 0.5)
        generated_z.append(z_random)
    
    generated_z = np.array(generated_z)
    
    # Predict Tg for generated candidates
    with torch.no_grad():
        z_tensor = torch.FloatTensor(generated_z)
        tg_pred_gen = model.predict_property(z_tensor).squeeze().numpy()
        tg_pred_gen = tg_pred_gen * tg_std_val + tg_mean
        
        # Decode to feature space
        x_decoded = model.decode(z_tensor).numpy()
    
    # Filter candidates in target range
    in_range = (tg_pred_gen >= tg_low) & (tg_pred_gen <= tg_high)
    n_valid = in_range.sum()
    print(f"  Generated candidates in target range: {n_valid}/{len(generated_z)}")
    
    for i in range(len(generated_z)):
        all_generated_candidates.append({
            'target': target_name,
            'tg_predicted': float(tg_pred_gen[i]),
            'in_target_range': bool(in_range[i]),
            'latent_vector': generated_z[i].tolist(),
        })

print(f"\nTotal generated candidates: {len(all_generated_candidates)}")

# ============================================================
# Find nearest neighbors in training set for generated candidates
# ============================================================
print("\nFinding nearest neighbors for top candidates...")

# For each generated candidate, find the nearest vitrimer in the training set
from scipy.spatial.distance import cdist

# Get top candidates from each range
top_candidates = []
for target_name, (tg_low, tg_high) in target_ranges.items():
    cands = [c for c in all_generated_candidates if c['target'] == target_name and c['in_target_range']]
    # Sort by how close to center of target range
    target_center = (tg_low + tg_high) / 2
    cands.sort(key=lambda c: abs(c['tg_predicted'] - target_center))
    top_candidates.extend(cands[:20])

print(f"Top candidates selected: {len(top_candidates)}")

# Find nearest neighbors
top_z = np.array([c['latent_vector'] for c in top_candidates])
distances = cdist(top_z, z_all, metric='euclidean')

for i, cand in enumerate(top_candidates):
    nn_idx = np.argmin(distances[i])
    real_idx = valid_indices[nn_idx]
    cand['nearest_acid'] = df_vit.loc[real_idx, 'acid']
    cand['nearest_epoxide'] = df_vit.loc[real_idx, 'epoxide']
    cand['nearest_tg_calibrated'] = float(df_vit.loc[real_idx, 'tg_calibrated'])
    cand['nearest_tg_md'] = float(df_vit.loc[real_idx, 'tg'])
    cand['nn_distance'] = float(distances[i, nn_idx])

# Save top candidates
candidates_df = pd.DataFrame(top_candidates)
candidates_df_export = candidates_df.drop(columns=['latent_vector'])
candidates_df_export.to_csv(os.path.join(OUT_DIR, 'top_candidates.csv'), index=False)
print(f"Saved top candidates to outputs/top_candidates.csv")

# ============================================================
# Propose novel vitrimer chemistries
# ============================================================
print("\n--- Proposing Novel Vitrimer Chemistries ---")

# Strategy: For each target range, find the best acid-epoxide combinations
# by looking at which acids and epoxides appear most frequently in high-Tg vitrimers

# Analyze acid and epoxide distributions by Tg range
unique_acids = df_vit['acid'].unique()
unique_epoxides = df_vit['epoxide'].unique()
print(f"Unique acids: {len(unique_acids)}")
print(f"Unique epoxides: {len(unique_epoxides)}")

# High Tg vitrimers (top 10%)
tg_threshold_high = df_vit['tg_calibrated'].quantile(0.90)
high_tg_vit = df_vit[df_vit['tg_calibrated'] >= tg_threshold_high]
print(f"\nHigh Tg vitrimers (>= {tg_threshold_high:.1f} K): {len(high_tg_vit)}")

# Most common acids in high Tg
acid_counts = high_tg_vit['acid'].value_counts().head(10)
epoxide_counts = high_tg_vit['epoxide'].value_counts().head(10)

# Generate novel combinations from top acids and epoxides
novel_combinations = []
existing_pairs = set(zip(df_vit['acid'], df_vit['epoxide']))

top_acids = acid_counts.index.tolist()[:5]
top_epoxides = epoxide_counts.index.tolist()[:5]

for acid in top_acids:
    for epoxide in top_epoxides:
        if (acid, epoxide) not in existing_pairs:
            novel_combinations.append({
                'acid': acid,
                'epoxide': epoxide,
                'source': 'combinatorial_high_tg'
            })

print(f"Novel combinations proposed: {len(novel_combinations)}")

# Predict Tg for novel combinations
if len(novel_combinations) > 0:
    novel_features = []
    for combo in novel_combinations:
        afp = smiles_to_fingerprint(combo['acid'])
        efp = smiles_to_fingerprint(combo['epoxide'])
        ad = get_descriptors(combo['acid'])
        ed = get_descriptors(combo['epoxide'])
        feat = np.concatenate([afp, efp, ad, ed])
        novel_features.append(feat)
    
    novel_features = np.array(novel_features)
    novel_scaled = scaler_vae.transform(novel_features)
    
    with torch.no_grad():
        novel_tensor = torch.FloatTensor(novel_scaled)
        mu_novel, _ = model.encode(novel_tensor)
        tg_novel = model.predict_property(mu_novel).squeeze().numpy()
        tg_novel = tg_novel * tg_std_val + tg_mean
    
    for i, combo in enumerate(novel_combinations):
        combo['tg_predicted'] = float(tg_novel[i])
    
    novel_df = pd.DataFrame(novel_combinations)
    novel_df = novel_df.sort_values('tg_predicted', ascending=False)
    novel_df.to_csv(os.path.join(OUT_DIR, 'novel_vitrimer_candidates.csv'), index=False)
    print(f"\nNovel candidates saved. Tg range: {tg_novel.min():.1f} - {tg_novel.max():.1f} K")
    print(novel_df[['acid', 'epoxide', 'tg_predicted']].head(10).to_string())

# ============================================================
# Comprehensive Figures
# ============================================================

# --- Figure 7: Inverse Design Results ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 7a: Distribution of generated candidates Tg
gen_tg = [c['tg_predicted'] for c in all_generated_candidates]
axes[0, 0].hist(gen_tg, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label='Generated')
axes[0, 0].hist(tg_values, bins=50, color='coral', edgecolor='black', alpha=0.4, label='Training')
for target_name, (tg_low, tg_high) in target_ranges.items():
    axes[0, 0].axvspan(tg_low, tg_high, alpha=0.1, color='green')
axes[0, 0].set_xlabel('Predicted Tg (K)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('(a) Generated vs Training Tg Distribution')
axes[0, 0].legend()

# 7b: Latent space with generated candidates
gen_z = np.array([c['latent_vector'] for c in all_generated_candidates])
gen_z_2d = pca_latent.transform(gen_z)
axes[0, 1].scatter(z_2d[:, 0], z_2d[:, 1], c='lightgray', alpha=0.2, s=3, label='Training')
scatter = axes[0, 1].scatter(gen_z_2d[:, 0], gen_z_2d[:, 1], 
                              c=[c['tg_predicted'] for c in all_generated_candidates],
                              cmap='RdYlBu_r', alpha=0.5, s=10, label='Generated')
axes[0, 1].set_xlabel('Latent PC1')
axes[0, 1].set_ylabel('Latent PC2')
axes[0, 1].set_title('(b) Generated Candidates in Latent Space')
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Predicted Tg (K)')

# 7c: Target range success rates
success_rates = {}
for target_name, (tg_low, tg_high) in target_ranges.items():
    cands = [c for c in all_generated_candidates if c['target'] == target_name]
    in_range = sum(1 for c in cands if c['in_target_range'])
    success_rates[target_name] = in_range / len(cands) * 100 if cands else 0

bars = axes[1, 0].bar(range(len(success_rates)), list(success_rates.values()), 
                       color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[1, 0].set_xticks(range(len(success_rates)))
axes[1, 0].set_xticklabels([k.split('(')[0].strip() for k in success_rates.keys()], rotation=15)
axes[1, 0].set_ylabel('Success Rate (%)')
axes[1, 0].set_title('(c) Target Range Success Rates')
for bar, val in zip(bars, success_rates.values()):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12)

# 7d: Top candidates by target range
if len(novel_combinations) > 0:
    novel_tg = [c['tg_predicted'] for c in novel_combinations]
    axes[1, 1].barh(range(min(15, len(novel_combinations))), 
                     novel_df['tg_predicted'].head(15).values,
                     color='teal', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Tg (K)')
    axes[1, 1].set_ylabel('Candidate Index')
    axes[1, 1].set_title('(d) Top 15 Novel Vitrimer Candidates')
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_inverse_design.png'))
plt.close()
print("\nSaved fig7_inverse_design.png")

# --- Figure 8: Chemical Diversity Analysis ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 8a: Molecular weight distribution
acid_mw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df_vit.loc[valid_indices, 'acid'] if Chem.MolFromSmiles(s)]
epoxide_mw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df_vit.loc[valid_indices, 'epoxide'] if Chem.MolFromSmiles(s)]
axes[0].hist(acid_mw, bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid')
axes[0].hist(epoxide_mw, bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide')
axes[0].set_xlabel('Molecular Weight (g/mol)')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Molecular Weight Distribution')
axes[0].legend()

# 8b: Tg vs molecular weight (average of acid + epoxide)
avg_mw = [(a + e) / 2 for a, e in zip(acid_mw[:len(tg_values)], epoxide_mw[:len(tg_values)])]
if len(avg_mw) >= len(tg_values):
    axes[1].scatter(avg_mw[:len(tg_values)], tg_values, c='steelblue', alpha=0.2, s=5)
    axes[1].set_xlabel('Average Molecular Weight (g/mol)')
    axes[1].set_ylabel('Calibrated Tg (K)')
    axes[1].set_title('(b) Tg vs Molecular Weight')

# 8c: LogP distribution
acid_logp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in df_vit.loc[valid_indices, 'acid'] if Chem.MolFromSmiles(s)]
epoxide_logp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in df_vit.loc[valid_indices, 'epoxide'] if Chem.MolFromSmiles(s)]
axes[2].hist(acid_logp, bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid')
axes[2].hist(epoxide_logp, bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide')
axes[2].set_xlabel('LogP')
axes[2].set_ylabel('Count')
axes[2].set_title('(c) LogP Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig8_chemical_diversity.png'))
plt.close()
print("Saved fig8_chemical_diversity.png")

# --- Figure 9: Tg Heatmap for top acid-epoxide combinations ---
print("\nGenerating Tg heatmap for top acid-epoxide combinations...")

# Get top acids and epoxides by frequency in high-Tg region
top_n = 15
top_acids_heat = high_tg_vit['acid'].value_counts().head(top_n).index.tolist()
top_epoxides_heat = high_tg_vit['epoxide'].value_counts().head(top_n).index.tolist()

# Build heatmap matrix
heatmap_data = np.full((len(top_acids_heat), len(top_epoxides_heat)), np.nan)
for i, acid in enumerate(top_acids_heat):
    for j, epoxide in enumerate(top_epoxides_heat):
        match = df_vit[(df_vit['acid'] == acid) & (df_vit['epoxide'] == epoxide)]
        if len(match) > 0:
            heatmap_data[i, j] = match['tg_calibrated'].values[0]

fig, ax = plt.subplots(figsize=(14, 10))
# Truncate labels
acid_labels = [s[:30] + '...' if len(s) > 30 else s for s in top_acids_heat]
epoxide_labels = [s[:30] + '...' if len(s) > 30 else s for s in top_epoxides_heat]

mask = np.isnan(heatmap_data)
sns.heatmap(heatmap_data, ax=ax, cmap='RdYlBu_r', annot=True, fmt='.0f',
            xticklabels=epoxide_labels, yticklabels=acid_labels,
            mask=mask, cbar_kws={'label': 'Calibrated Tg (K)'})
ax.set_xlabel('Epoxide Component')
ax.set_ylabel('Acid Component')
ax.set_title('Calibrated Tg (K) for Top Acid-Epoxide Combinations')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig9_tg_heatmap.png'))
plt.close()
print("Saved fig9_tg_heatmap.png")

# Save all results
all_results = {
    'n_vitrimers': len(df_vit),
    'n_generated': len(all_generated_candidates),
    'n_novel_combinations': len(novel_combinations),
    'success_rates': success_rates,
    'tg_calibrated_range': [float(tg_values.min()), float(tg_values.max())],
    'vae_latent_dim': latent_dim,
    'vae_property_r2': float(r2),
    'vae_property_mae': float(mae),
}
with open(os.path.join(OUT_DIR, 'generation_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nPhase 3 & 4 complete!")
