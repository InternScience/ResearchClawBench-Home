"""
Phase 3: Variational Autoencoder for Inverse Design of Vitrimers
Uses VAE latent space for nearest-neighbor retrieval and interpolation
to generate valid novel vitrimer candidates targeting specific Tg ranges.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
TMP = '/tmp/vitrimer_work'

cal_df = pd.read_csv('data/tg_calibration.csv')
vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')
gp_cal_results = pd.read_csv('outputs/gp_calibration_results.csv')
gp_vit_results = pd.read_csv('outputs/gp_vitrimer_predictions.csv')

print("Preparing SMILES corpus...")

# Use calibration + subset of vitrimer SMILES
all_smiles = set()
for s in cal_df['smiles']: all_smiles.add(s)
for s in vit_df['acid'].sample(n=min(2000, len(vit_df)), random_state=42): all_smiles.add(s)
for s in vit_df['epoxide'].sample(n=min(2000, len(vit_df)), random_state=42): all_smiles.add(s)
all_smiles = list(all_smiles)
print(f"Total unique SMILES: {len(all_smiles)}")

chars = sorted(set(c for s in all_smiles for c in s))
char_to_idx = {c: i+1 for i, c in enumerate(chars)}
idx_to_char = {i+1: c for i, c in enumerate(chars)}
vocab_size = len(chars) + 1
max_len = max(len(s) for s in all_smiles)

valid_smiles = [s for s in all_smiles if Chem.MolFromSmiles(s) is not None]
print(f"Valid SMILES: {len(valid_smiles)}")

def smiles_to_indices(smiles):
    indices = np.zeros(max_len, dtype=np.int64)
    for i, c in enumerate(smiles):
        if i < max_len: indices[i] = char_to_idx[c]
    return indices

indices = np.array([smiles_to_indices(s) for s in valid_smiles])
lengths = np.array([len(s) for s in valid_smiles], dtype=np.int64)

vocab_info = {
    'vocab_size': vocab_size, 'max_len': max_len,
    'char_to_idx': char_to_idx,
    'idx_to_char': {str(k): v for k, v in idx_to_char.items()},
    'n_valid_smiles': len(valid_smiles),
}
with open('outputs/vae_vocab.json', 'w') as f:
    json.dump(vocab_info, f, indent=2)

np.save(os.path.join(TMP, 'vae_indices.npy'), indices)
np.save(os.path.join(TMP, 'vae_lengths.npy'), lengths)
np.save(os.path.join(TMP, 'valid_smiles.npy'), np.array(valid_smiles))

# ---- VAE Model ----
class SimpleVAE(nn.Module):
    def __init__(self, vocab_size, max_len, latent_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.latent_dim = latent_dim
        
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.enc_conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.enc_conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        
        enc_out_size = 256 * max_len
        self.fc_mu = nn.Linear(enc_out_size, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_size, latent_dim)
        
        self.dec_fc = nn.Linear(latent_dim, 128 * max_len)
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2)
        self.fc_out = nn.Linear(32, vocab_size)
    
    def encode(self, x):
        emb = self.embedding(x).transpose(1, 2)
        h = F.relu(self.enc_conv1(emb))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.dec_fc(z)).view(-1, 128, self.max_len)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = h.transpose(1, 2)
        return self.fc_out(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

indices_t = torch.from_numpy(indices)
dataset = TensorDataset(indices_t)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

latent_dim = 32
model = SimpleVAE(vocab_size, max_len, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

def vae_loss(logits, x, mu, logvar):
    recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), reduction='none').view(*x.shape)
    mask = (x > 0).float()
    recon_loss = (recon_loss * mask).sum() / mask.sum()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return recon_loss + 0.05 * kl_loss, recon_loss, kl_loss

print("\nTraining VAE...")
n_epochs = 20
best_loss = float('inf')
for epoch in range(n_epochs):
    model.train()
    total_loss = total_recon = total_kl = n_batches = 0
    for (batch_x,) in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(batch_x)
        loss, recon, kl = vae_loss(logits, batch_x, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); total_recon += recon.item(); total_kl += kl.item(); n_batches += 1
    
    avg_loss = total_loss / n_batches
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Recon={total_recon/n_batches:.4f}, KL={total_kl/n_batches:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(TMP, 'vae_best.pt'))

torch.save(model.state_dict(), os.path.join(TMP, 'vae_final.pt'))

# ---- Latent Space ----
print("\nEncoding SMILES into latent space...")
model.eval()
with torch.no_grad():
    all_mu, _ = model.encode(indices_t.to(device))
    latent_vectors = all_mu.cpu().numpy()
np.save(os.path.join(TMP, 'latent_vectors.npy'), latent_vectors)

smiles_to_latent = {s: latent_vectors[i] for i, s in enumerate(valid_smiles)}

# ---- Property Predictor in Latent Space ----
cal_latents, cal_tgs = [], []
for _, row in cal_df.iterrows():
    s = row['smiles']
    if s in smiles_to_latent:
        cal_latents.append(smiles_to_latent[s])
        cal_tgs.append(row['tg_exp'])
cal_latents, cal_tgs = np.array(cal_latents), np.array(cal_tgs)
print(f"Calibration molecules in latent space: {len(cal_latents)}")

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(cal_latents, cal_tgs)
cv_scores = cross_val_score(mlp, cal_latents, cal_tgs, cv=5, scoring='r2')
print(f"Latent->Tg predictor CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ---- Inverse Design via Latent Space Nearest Neighbor Retrieval ----
print("\nPerforming inverse design via latent space exploration...")

# Build nearest neighbor index over all valid SMILES
nn_index = NearestNeighbors(n_neighbors=20, metric='euclidean')
nn_index.fit(latent_vectors)

target_tg_ranges = [(300, 350, "low_Tg"), (350, 400, "mid_Tg"), (400, 450, "high_Tg")]
generated_candidates = []

for t_min, t_max, label in target_tg_ranges:
    # Strategy 1: Sample random points in latent space, find nearest valid SMILES
    n_samples = 5000
    random_latents = np.random.randn(n_samples, latent_dim) * np.std(latent_vectors, axis=0)
    random_latents += np.mean(latent_vectors, axis=0)
    
    pred_tg = mlp.predict(random_latents)
    mask = (pred_tg >= t_min) & (pred_tg <= t_max)
    selected = random_latents[mask]
    selected_pred = pred_tg[mask]
    
    print(f"\n  Target {label} ({t_min}-{t_max}K): {len(selected)} latent points")
    
    # Find nearest valid SMILES for each selected point
    found_smiles = set()
    for i in range(min(200, len(selected))):
        distances, indices_nn = nn_index.kneighbors(selected[i:i+1])
        for j in range(min(10, len(indices_nn[0]))):
            idx = indices_nn[0][j]
            s = valid_smiles[idx]
            if s not in found_smiles:
                found_smiles.add(s)
                generated_candidates.append({
                    'smiles': s,
                    'predicted_tg': float(selected_pred[i]),
                    'target_range': label,
                    'method': 'latent_nn',
                    'distance': float(distances[0][j]),
                })
    
    # Strategy 2: Interpolate between pairs of molecules with Tg near target range
    # Find molecules in vitrimer dataset with calibrated Tg near target
    vit_mask = (gp_vit_results['tg_calibrated'] >= t_min - 20) & (gp_vit_results['tg_calibrated'] <= t_max + 20)
    vit_near = gp_vit_results[vit_mask].head(50)
    
    for idx1 in range(min(10, len(vit_near))):
        for idx2 in range(idx1+1, min(10, len(vit_near))):
            s1 = vit_near.iloc[idx1]['acid']
            s2 = vit_near.iloc[idx2]['acid']
            if s1 in smiles_to_latent and s2 in smiles_to_latent:
                # Interpolate
                z1 = smiles_to_latent[s1]
                z2 = smiles_to_latent[s2]
                for alpha in [0.3, 0.5, 0.7]:
                    z_interp = alpha * z1 + (1 - alpha) * z2
                    pred = mlp.predict(z_interp.reshape(1, -1))[0]
                    if t_min <= pred <= t_max:
                        # Find nearest valid SMILES
                        dists, idxs = nn_index.kneighbors(z_interp.reshape(1, -1), n_neighbors=5)
                        for k in range(5):
                            nn_idx = idxs[0][k]
                            s = valid_smiles[nn_idx]
                            if s not in found_smiles:
                                found_smiles.add(s)
                                generated_candidates.append({
                                    'smiles': s,
                                    'predicted_tg': float(pred),
                                    'target_range': label,
                                    'method': 'interpolation',
                                    'distance': float(dists[0][k]),
                                })

    print(f"  Found {len(found_smiles)} unique candidates for {label}")

# Validate all candidates
valid_candidates = []
seen = set()
for c in generated_candidates:
    s = c['smiles']
    if s in seen: continue
    seen.add(s)
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        c['mol_wt'] = Descriptors.MolWt(mol)
        c['logp'] = Descriptors.MolLogP(mol)
        c['num_atoms'] = mol.GetNumAtoms()
        c['num_rings'] = Descriptors.RingCount(mol)
        c['is_valid'] = True
        valid_candidates.append(c)

print(f"\nTotal generated: {len(generated_candidates)}, Valid unique: {len(valid_candidates)}")

if valid_candidates:
    cand_df = pd.DataFrame(valid_candidates)
    cand_df.to_csv('outputs/generated_candidates.csv', index=False)
    for label in ["low_Tg", "mid_Tg", "high_Tg"]:
        subset = cand_df[cand_df['target_range'] == label]
        if len(subset) > 0:
            print(f"  {label}: {len(subset)} valid candidates")
            print(f"    Tg range: {subset['predicted_tg'].min():.1f} - {subset['predicted_tg'].max():.1f} K")
            print(f"    Mean MW: {subset['mol_wt'].mean():.1f}, Mean LogP: {subset['logp'].mean():.2f}")

# Save latent space data
np.save(os.path.join(TMP, 'cal_latents.npy'), cal_latents)
np.save(os.path.join(TMP, 'cal_tgs.npy'), cal_tgs)

# Save all latent vectors with corresponding SMILES for visualization
latent_meta = pd.DataFrame({
    'smiles': valid_smiles,
    **{'latent_dim_' + str(i): latent_vectors[:, i] for i in range(latent_dim)}
})
latent_meta.to_csv('outputs/latent_space_data.csv', index=False)

print("\nVAE inverse design complete.")
