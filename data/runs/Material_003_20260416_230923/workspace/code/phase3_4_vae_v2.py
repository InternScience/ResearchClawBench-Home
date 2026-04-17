#!/usr/bin/env python3
"""
Phase 3 & 4: Optimized Graph VAE + Inverse Design
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
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from scipy.spatial.distance import cdist
import json, os, warnings, pickle
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                     'savefig.dpi': 150, 'savefig.bbox': 'tight'})

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')

df_vit = pd.read_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'))
df_cal = pd.read_csv(os.path.join(BASE, 'data', 'tg_calibration.csv'))

FP_BITS = 128  # Reduced for speed

print("=" * 60)
print("PHASE 3: Graph VAE for Vitrimer Generation")
print("=" * 60)

# ============================================================
# Fast feature computation
# ============================================================
def smiles_to_fp(smiles, nbits=FP_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits))

def get_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*6
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)]

cache_file = os.path.join(OUT_DIR, 'vitrimer_features.npz')
if os.path.exists(cache_file):
    print("Loading cached features...")
    data = np.load(cache_file, allow_pickle=True)
    X_combined = data['X_combined']
    valid_indices = data['valid_indices']
else:
    print("Computing features for vitrimers...")
    features_list = []
    valid_indices = []
    for idx, row in df_vit.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(df_vit)}...")
        try:
            afp = smiles_to_fp(row['acid'])
            efp = smiles_to_fp(row['epoxide'])
            ad = get_desc(row['acid'])
            ed = get_desc(row['epoxide'])
            feat = np.concatenate([afp, efp, ad, ed])
            features_list.append(feat)
            valid_indices.append(idx)
        except:
            pass
    
    X_combined = np.array(features_list)
    valid_indices = np.array(valid_indices)
    np.savez(cache_file, X_combined=X_combined, valid_indices=valid_indices)
    print(f"Saved cached features.")

print(f"Feature matrix: {X_combined.shape}, Valid: {len(valid_indices)}")

tg_values = df_vit.loc[valid_indices, 'tg_calibrated'].values
tg_md_values = df_vit.loc[valid_indices, 'tg'].values

scaler_vae = StandardScaler()
X_scaled = scaler_vae.fit_transform(X_combined)

# ============================================================
# Graph VAE
# ============================================================
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

# Train
print("\nTraining Graph VAE...")
X_tensor = torch.FloatTensor(X_scaled)
tg_mean, tg_std_v = tg_values.mean(), tg_values.std()
tg_norm = torch.FloatTensor((tg_values - tg_mean) / tg_std_v)

dataset = TensorDataset(X_tensor, tg_norm)
loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

latent_dim = 32
model = GraphVAE(X_scaled.shape[1], latent_dim=latent_dim)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

losses_hist = {'total': [], 'recon': [], 'kl': [], 'prop': []}
n_epochs = 100

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
    
    if (epoch+1) % 25 == 0:
        print(f"  Epoch {epoch+1}: Loss={ep_loss/nb:.4f}, R={ep_r/nb:.4f}, KL={ep_k/nb:.4f}, P={ep_p/nb:.4f}")

print("Training complete!")

# Save model
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'vae_model.pt'))

# --- Figure 5: Training Loss ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(losses_hist['total'], 'b-'); axes[0].set_title('(a) Total Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[1].plot(losses_hist['recon'], 'r-', label='Recon'); axes[1].plot(losses_hist['kl'], 'g-', label='KL')
axes[1].set_title('(b) Recon & KL Loss'); axes[1].legend(); axes[1].set_xlabel('Epoch')
axes[2].plot(losses_hist['prop'], 'm-'); axes[2].set_title('(c) Property Loss'); axes[2].set_xlabel('Epoch')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig5_vae_training.png')); plt.close()
print("Saved fig5_vae_training.png")

# ============================================================
# Latent Space Analysis
# ============================================================
print("\nAnalyzing latent space...")
model.eval()
with torch.no_grad():
    mu_all, _ = model.encode(X_tensor)
    z_all = mu_all.numpy()
    tg_pred_all = model.prop_pred(mu_all).squeeze().numpy() * tg_std_v + tg_mean

pca_lat = PCA(n_components=2)
z_2d = pca_lat.fit_transform(z_all)

r2_vae = r2_score(tg_values, tg_pred_all)
mae_vae = mean_absolute_error(tg_values, tg_pred_all)
print(f"VAE property prediction: R²={r2_vae:.3f}, MAE={mae_vae:.1f} K")

# --- Figure 6: Latent Space ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=tg_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[0].set_xlabel('Latent PC1'); axes[0].set_ylabel('Latent PC2')
axes[0].set_title('(a) Latent Space (Calibrated Tg)'); plt.colorbar(sc, ax=axes[0], label='Tg (K)')

sc2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=tg_md_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[1].set_xlabel('Latent PC1'); axes[1].set_ylabel('Latent PC2')
axes[1].set_title('(b) Latent Space (MD Tg)'); plt.colorbar(sc2, ax=axes[1], label='MD Tg (K)')

axes[2].scatter(tg_values, tg_pred_all, c='steelblue', alpha=0.3, s=5)
lims = [min(tg_values.min(), tg_pred_all.min())-10, max(tg_values.max(), tg_pred_all.max())+10]
axes[2].plot(lims, lims, 'r--', lw=2)
axes[2].set_xlabel('True Calibrated Tg (K)'); axes[2].set_ylabel('VAE Predicted Tg (K)')
axes[2].set_title(f'(c) VAE Property Prediction\nR²={r2_vae:.3f}, MAE={mae_vae:.1f} K')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig6_latent_space.png')); plt.close()
print("Saved fig6_latent_space.png")

# ============================================================
# PHASE 4: Inverse Design
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Inverse Design")
print("=" * 60)

target_ranges = {
    'High Tg (>480 K)': (480, 600),
    'Medium-High Tg (420-480 K)': (420, 480),
    'Medium Tg (360-420 K)': (360, 420),
}

all_gen = []
for tname, (tl, th) in target_ranges.items():
    print(f"\n--- {tname} ---")
    mask = (tg_values >= tl) & (tg_values <= th)
    n_in = mask.sum()
    print(f"  Existing in range: {n_in}")
    if n_in < 5: continue
    
    z_target = z_all[mask]
    centroid = z_target.mean(0)
    cov = np.cov(z_target.T) + np.eye(latent_dim)*0.01
    
    gen_z = []
    # Interpolation
    for _ in range(150):
        i1, i2 = np.random.choice(len(z_target), 2, replace=True)
        gen_z.append(np.random.uniform(0.2, 0.8) * z_target[i1] + (1-np.random.uniform(0.2, 0.8)) * z_target[i2])
    # Perturbation
    for _ in range(150):
        gen_z.append(z_target[np.random.choice(len(z_target))] + np.random.randn(latent_dim)*0.3)
    # Sampling
    for _ in range(150):
        gen_z.append(np.random.multivariate_normal(centroid, cov*0.5))
    
    gen_z = np.array(gen_z)
    with torch.no_grad():
        tg_gen = model.prop_pred(torch.FloatTensor(gen_z)).squeeze().numpy() * tg_std_v + tg_mean
    
    in_range = (tg_gen >= tl) & (tg_gen <= th)
    print(f"  In target range: {in_range.sum()}/{len(gen_z)}")
    
    for i in range(len(gen_z)):
        all_gen.append({'target': tname, 'tg_predicted': float(tg_gen[i]),
                       'in_target_range': bool(in_range[i]), 'latent_vector': gen_z[i].tolist()})

print(f"\nTotal generated: {len(all_gen)}")

# Find nearest neighbors
top_cands = []
for tname, (tl, th) in target_ranges.items():
    cands = [c for c in all_gen if c['target'] == tname and c['in_target_range']]
    tc = (tl + th) / 2
    cands.sort(key=lambda c: abs(c['tg_predicted'] - tc))
    top_cands.extend(cands[:20])

print(f"Top candidates: {len(top_cands)}")

if top_cands:
    top_z = np.array([c['latent_vector'] for c in top_cands])
    dists = cdist(top_z, z_all)
    for i, c in enumerate(top_cands):
        nn_idx = np.argmin(dists[i])
        ri = valid_indices[nn_idx]
        c['nearest_acid'] = df_vit.loc[ri, 'acid']
        c['nearest_epoxide'] = df_vit.loc[ri, 'epoxide']
        c['nearest_tg_cal'] = float(df_vit.loc[ri, 'tg_calibrated'])
        c['nearest_tg_md'] = float(df_vit.loc[ri, 'tg'])
        c['nn_dist'] = float(dists[i, nn_idx])
    
    cdf = pd.DataFrame(top_cands)
    cdf_export = cdf.drop(columns=['latent_vector'])
    cdf_export.to_csv(os.path.join(OUT_DIR, 'top_candidates.csv'), index=False)

# Novel combinations
print("\n--- Novel Vitrimer Chemistries ---")
tg_thresh = df_vit['tg_calibrated'].quantile(0.90)
high_vit = df_vit[df_vit['tg_calibrated'] >= tg_thresh]
print(f"High Tg threshold (90th pct): {tg_thresh:.1f} K, n={len(high_vit)}")

acid_top = high_vit['acid'].value_counts().head(8).index.tolist()
epox_top = high_vit['epoxide'].value_counts().head(8).index.tolist()
existing = set(zip(df_vit['acid'], df_vit['epoxide']))

novel = []
for a in acid_top:
    for e in epox_top:
        if (a, e) not in existing:
            novel.append({'acid': a, 'epoxide': e})

print(f"Novel combinations: {len(novel)}")

if novel:
    nf = []
    for c in novel:
        feat = np.concatenate([smiles_to_fp(c['acid']), smiles_to_fp(c['epoxide']),
                              get_desc(c['acid']), get_desc(c['epoxide'])])
        nf.append(feat)
    nf = np.array(nf)
    nf_scaled = scaler_vae.transform(nf)
    with torch.no_grad():
        mu_n, _ = model.encode(torch.FloatTensor(nf_scaled))
        tg_n = model.prop_pred(mu_n).squeeze().numpy() * tg_std_v + tg_mean
    
    for i, c in enumerate(novel):
        c['tg_predicted'] = float(tg_n[i])
    
    ndf = pd.DataFrame(novel).sort_values('tg_predicted', ascending=False)
    ndf.to_csv(os.path.join(OUT_DIR, 'novel_vitrimer_candidates.csv'), index=False)
    print(f"Novel Tg range: {tg_n.min():.1f} - {tg_n.max():.1f} K")
    print(ndf[['acid', 'epoxide', 'tg_predicted']].head(10).to_string())

# ============================================================
# Figures
# ============================================================
# --- Figure 7: Inverse Design Results ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

gen_tg = [c['tg_predicted'] for c in all_gen]
axes[0,0].hist(gen_tg, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label='Generated')
axes[0,0].hist(tg_values, bins=50, color='coral', edgecolor='black', alpha=0.4, label='Training', density=True)
# Normalize generated too
axes[0,0].clear()
axes[0,0].hist(tg_values, bins=50, color='coral', edgecolor='black', alpha=0.5, label='Training data')
axes[0,0].hist(gen_tg, bins=50, color='steelblue', edgecolor='black', alpha=0.5, label='Generated')
for _, (tl, th) in target_ranges.items():
    axes[0,0].axvspan(tl, th, alpha=0.1, color='green')
axes[0,0].set_xlabel('Predicted Tg (K)'); axes[0,0].set_ylabel('Count')
axes[0,0].set_title('(a) Generated vs Training Tg'); axes[0,0].legend()

gen_z_arr = np.array([c['latent_vector'] for c in all_gen])
gen_z_2d = pca_lat.transform(gen_z_arr)
axes[0,1].scatter(z_2d[:,0], z_2d[:,1], c='lightgray', alpha=0.2, s=3, label='Training')
sc = axes[0,1].scatter(gen_z_2d[:,0], gen_z_2d[:,1], c=[c['tg_predicted'] for c in all_gen],
                       cmap='RdYlBu_r', alpha=0.5, s=10)
axes[0,1].set_xlabel('Latent PC1'); axes[0,1].set_ylabel('Latent PC2')
axes[0,1].set_title('(b) Generated in Latent Space'); plt.colorbar(sc, ax=axes[0,1], label='Tg (K)')

# Success rates
sr = {}
for tn, (tl, th) in target_ranges.items():
    cs = [c for c in all_gen if c['target'] == tn]
    sr[tn] = sum(c['in_target_range'] for c in cs) / len(cs) * 100 if cs else 0
bars = axes[1,0].bar(range(len(sr)), list(sr.values()), color=['#e74c3c','#f39c12','#2ecc71'])
axes[1,0].set_xticks(range(len(sr)))
axes[1,0].set_xticklabels([k.split('(')[0].strip() for k in sr.keys()], rotation=15)
axes[1,0].set_ylabel('Success Rate (%)'); axes[1,0].set_title('(c) Target Range Success')
for b, v in zip(bars, sr.values()):
    axes[1,0].text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}%', ha='center', fontsize=11)

if novel:
    axes[1,1].barh(range(min(15, len(ndf))), ndf['tg_predicted'].head(15).values, color='teal', edgecolor='black')
    axes[1,1].set_xlabel('Predicted Tg (K)'); axes[1,1].set_ylabel('Candidate #')
    axes[1,1].set_title('(d) Top 15 Novel Candidates'); axes[1,1].invert_yaxis()

plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig7_inverse_design.png')); plt.close()
print("Saved fig7_inverse_design.png")

# --- Figure 8: Chemical Diversity ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sample for speed
sample_idx = np.random.choice(len(valid_indices), min(2000, len(valid_indices)), replace=False)
sample_acids = df_vit.loc[valid_indices[sample_idx], 'acid'].values
sample_epox = df_vit.loc[valid_indices[sample_idx], 'epoxide'].values

amw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in sample_acids if Chem.MolFromSmiles(s)]
emw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in sample_epox if Chem.MolFromSmiles(s)]
axes[0].hist(amw, bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid')
axes[0].hist(emw, bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide')
axes[0].set_xlabel('Molecular Weight'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) MW Distribution'); axes[0].legend()

avg_mw = [(a+e)/2 for a, e in zip(amw[:len(sample_idx)], emw[:len(sample_idx)])]
stg = tg_values[sample_idx[:len(avg_mw)]]
axes[1].scatter(avg_mw, stg, c='steelblue', alpha=0.3, s=10)
axes[1].set_xlabel('Avg MW'); axes[1].set_ylabel('Calibrated Tg (K)')
axes[1].set_title('(b) Tg vs MW')

alp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in sample_acids if Chem.MolFromSmiles(s)]
elp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in sample_epox if Chem.MolFromSmiles(s)]
axes[2].hist(alp, bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid')
axes[2].hist(elp, bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide')
axes[2].set_xlabel('LogP'); axes[2].set_ylabel('Count'); axes[2].set_title('(c) LogP Distribution'); axes[2].legend()

plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig8_chemical_diversity.png')); plt.close()
print("Saved fig8_chemical_diversity.png")

# --- Figure 9: Tg Heatmap ---
print("Generating Tg heatmap...")
top_n = 12
ta = high_vit['acid'].value_counts().head(top_n).index.tolist()
te = high_vit['epoxide'].value_counts().head(top_n).index.tolist()

hm = np.full((len(ta), len(te)), np.nan)
for i, a in enumerate(ta):
    for j, e in enumerate(te):
        m = df_vit[(df_vit['acid']==a) & (df_vit['epoxide']==e)]
        if len(m) > 0: hm[i,j] = m['tg_calibrated'].values[0]

fig, ax = plt.subplots(figsize=(14, 10))
al = [s[:25]+'...' if len(s)>25 else s for s in ta]
el = [s[:25]+'...' if len(s)>25 else s for s in te]
sns.heatmap(hm, ax=ax, cmap='RdYlBu_r', annot=True, fmt='.0f',
            xticklabels=el, yticklabels=al, mask=np.isnan(hm),
            cbar_kws={'label': 'Calibrated Tg (K)'})
ax.set_xlabel('Epoxide'); ax.set_ylabel('Acid')
ax.set_title('Calibrated Tg for Top Acid-Epoxide Combinations')
plt.xticks(rotation=45, ha='right', fontsize=7); plt.yticks(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig9_tg_heatmap.png')); plt.close()
print("Saved fig9_tg_heatmap.png")

# Save results
results = {
    'n_vitrimers': int(len(df_vit)),
    'n_valid': int(len(valid_indices)),
    'n_generated': len(all_gen),
    'n_novel': len(novel) if novel else 0,
    'success_rates': sr,
    'vae_latent_dim': latent_dim,
    'vae_r2': float(r2_vae),
    'vae_mae': float(mae_vae),
    'tg_calibrated_stats': {
        'mean': float(tg_values.mean()),
        'std': float(tg_values.std()),
        'min': float(tg_values.min()),
        'max': float(tg_values.max())
    }
}
with open(os.path.join(OUT_DIR, 'generation_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nPhase 3 & 4 complete!")
