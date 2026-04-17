#!/usr/bin/env python3
"""Phase 3: Train VAE - optimized for speed"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')

df_vit = pd.read_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'))

print("Loading cached features...")
data = np.load(os.path.join(OUT_DIR, 'vitrimer_features.npz'), allow_pickle=True)
X_combined = data['X_combined'].astype(np.float32)
valid_indices = data['valid_indices']
print(f"Features: {X_combined.shape}")

tg_values = df_vit.loc[valid_indices, 'tg_calibrated'].values.astype(np.float32)
tg_md_values = df_vit.loc[valid_indices, 'tg'].values.astype(np.float32)

scaler_vae = StandardScaler()
X_scaled = scaler_vae.fit_transform(X_combined).astype(np.float32)
np.savez(os.path.join(OUT_DIR, 'scaler_params.npz'), mean=scaler_vae.mean_, scale=scaler_vae.scale_)

class GraphVAE(nn.Module):
    def __init__(self, d_in, d_lat=32, d_hid=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid//2), nn.ReLU())
        self.fc_mu = nn.Linear(d_hid//2, d_lat)
        self.fc_lv = nn.Linear(d_hid//2, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, d_hid//2), nn.ReLU(), nn.Linear(d_hid//2, d_hid), nn.ReLU(), nn.Linear(d_hid, d_in))
        self.pp = nn.Sequential(nn.Linear(d_lat, 32), nn.ReLU(), nn.Linear(32, 1))
    def encode(self, x):
        h = self.enc(x); return self.fc_mu(h), self.fc_lv(h)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.dec(z), mu, lv, self.pp(z)

X_t = torch.from_numpy(X_scaled)
tg_m, tg_s = float(tg_values.mean()), float(tg_values.std())
tg_n = torch.from_numpy((tg_values - tg_m) / tg_s)

loader = DataLoader(TensorDataset(X_t, tg_n), batch_size=1024, shuffle=True, drop_last=True)

latent_dim = 32
model = GraphVAE(X_scaled.shape[1], d_lat=latent_dim, d_hid=128)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

losses = {'total':[], 'recon':[], 'kl':[], 'prop':[]}
print("Training...")
for ep in range(50):
    model.train()
    tl, tr, tk, tp, nb = 0,0,0,0,0
    beta = min(1.0, ep/15.0)*0.5
    for bx, bt in loader:
        opt.zero_grad()
        xr, mu, lv, tpred = model(bx)
        r = F.mse_loss(xr, bx)
        k = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        p = F.mse_loss(tpred.squeeze(), bt)
        loss = r + beta*k + 10*p
        loss.backward(); opt.step()
        tl+=loss.item(); tr+=r.item(); tk+=k.item(); tp+=p.item(); nb+=1
    for key, v in zip(['total','recon','kl','prop'],[tl,tr,tk,tp]):
        losses[key].append(v/nb)
    if (ep+1)%10==0: print(f"  Ep {ep+1}: L={tl/nb:.4f} R={tr/nb:.4f} K={tk/nb:.4f} P={tp/nb:.4f}")

print("Done!")
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'vae_model.pt'))

# Figures
fig, axes = plt.subplots(1,3,figsize=(18,5))
axes[0].plot(losses['total'],'b-'); axes[0].set_title('(a) Total Loss'); axes[0].set_xlabel('Epoch')
axes[1].plot(losses['recon'],'r-',label='Recon'); axes[1].plot(losses['kl'],'g-',label='KL')
axes[1].set_title('(b) Recon & KL'); axes[1].legend(); axes[1].set_xlabel('Epoch')
axes[2].plot(losses['prop'],'m-'); axes[2].set_title('(c) Property Loss'); axes[2].set_xlabel('Epoch')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR,'fig5_vae_training.png')); plt.close()
print("Saved fig5")

model.eval()
with torch.no_grad():
    mu_all, _ = model.encode(X_t)
    z_all = mu_all.numpy()
    tp_all = model.pp(mu_all).squeeze().numpy()*tg_s+tg_m

np.savez(os.path.join(OUT_DIR,'latent_space.npz'), z_all=z_all, tg_values=tg_values,
         tg_md_values=tg_md_values, tg_pred_all=tp_all, tg_mean=tg_m, tg_std_v=tg_s, valid_indices=valid_indices)

pca2 = PCA(2); z2d = pca2.fit_transform(z_all)
r2v = r2_score(tg_values, tp_all); maev = mean_absolute_error(tg_values, tp_all)
print(f"VAE: R²={r2v:.3f}, MAE={maev:.1f}")

fig, axes = plt.subplots(1,3,figsize=(20,6))
sc=axes[0].scatter(z2d[:,0],z2d[:,1],c=tg_values,cmap='RdYlBu_r',alpha=0.3,s=5)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].set_title('(a) Latent (Cal. Tg)')
plt.colorbar(sc,ax=axes[0],label='Tg(K)')
sc2=axes[1].scatter(z2d[:,0],z2d[:,1],c=tg_md_values,cmap='RdYlBu_r',alpha=0.3,s=5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2'); axes[1].set_title('(b) Latent (MD Tg)')
plt.colorbar(sc2,ax=axes[1],label='MD Tg(K)')
axes[2].scatter(tg_values,tp_all,c='steelblue',alpha=0.3,s=5)
lm=[min(tg_values.min(),tp_all.min())-10,max(tg_values.max(),tp_all.max())+10]
axes[2].plot(lm,lm,'r--',lw=2)
axes[2].set_xlabel('True Tg(K)'); axes[2].set_ylabel('Pred Tg(K)')
axes[2].set_title(f'(c) VAE Prop\nR²={r2v:.3f}, MAE={maev:.1f}K')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR,'fig6_latent_space.png')); plt.close()
print("Saved fig6")

with open(os.path.join(OUT_DIR,'vae_metrics.json'),'w') as f:
    json.dump({'r2':float(r2v),'mae':float(maev),'latent_dim':latent_dim,'n_epochs':50},f,indent=2)
print("Phase 3 complete!")
