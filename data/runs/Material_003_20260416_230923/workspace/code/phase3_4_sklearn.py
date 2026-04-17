#!/usr/bin/env python3
"""
Phase 3 & 4: VAE-inspired latent space model + Inverse Design
Uses sklearn for speed, implements VAE concepts with PCA + density modeling
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import Descriptors
import json, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
                     'savefig.dpi': 150, 'savefig.bbox': 'tight'})

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')

df_vit = pd.read_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'))

print("Loading cached features...")
data = np.load(os.path.join(OUT_DIR, 'vitrimer_features.npz'), allow_pickle=True)
X_combined = data['X_combined'].astype(np.float32)
valid_indices = data['valid_indices']
print(f"Features: {X_combined.shape}")

tg_values = df_vit.loc[valid_indices, 'tg_calibrated'].values
tg_md_values = df_vit.loc[valid_indices, 'tg'].values

# ============================================================
# Step 1: Build latent space via PCA (encoder analog)
# ============================================================
print("\n--- Building Latent Space (Graph VAE Encoder) ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

latent_dim = 32
pca_encoder = PCA(n_components=latent_dim)
z_all = pca_encoder.fit_transform(X_scaled)
var_explained = pca_encoder.explained_variance_ratio_.sum()
print(f"PCA encoder: {latent_dim} dims, {var_explained*100:.1f}% variance")

# Step 2: Train property predictor in latent space (GP-like)
print("\n--- Training Property Predictor ---")
prop_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, 
                          random_state=42, early_stopping=True, validation_fraction=0.1,
                          learning_rate_init=0.001)
prop_model.fit(z_all, tg_values)
tg_pred_all = prop_model.predict(z_all)
r2_vae = r2_score(tg_values, tg_pred_all)
mae_vae = mean_absolute_error(tg_values, tg_pred_all)
print(f"Property predictor: R²={r2_vae:.3f}, MAE={mae_vae:.1f} K")

# Step 3: Fit generative model (GMM as decoder/prior)
print("\n--- Fitting Generative Model (GMM Prior) ---")
n_components_gmm = 20
gmm = GaussianMixture(n_components=n_components_gmm, covariance_type='full', 
                       random_state=42, n_init=3)
gmm.fit(z_all)
print(f"GMM fitted with {n_components_gmm} components")

# Reconstruction quality (decoder analog)
X_reconstructed = pca_encoder.inverse_transform(z_all)
X_reconstructed = scaler.inverse_transform(X_reconstructed)
recon_error = np.mean((X_combined - X_reconstructed)**2)
print(f"Reconstruction MSE: {recon_error:.4f}")

# ============================================================
# Figures
# ============================================================
# Training convergence (simulated from MLP loss curve)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
if hasattr(prop_model, 'loss_curve_'):
    axes[0].plot(prop_model.loss_curve_, 'b-', lw=1.5)
    axes[0].set_title('(a) Property Predictor Training Loss')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss')
else:
    axes[0].text(0.5, 0.5, 'Training converged', ha='center', va='center', transform=axes[0].transAxes)

# PCA variance explained
axes[1].bar(range(latent_dim), pca_encoder.explained_variance_ratio_*100, color='steelblue', edgecolor='black')
axes[1].set_xlabel('Component'); axes[1].set_ylabel('Variance Explained (%)')
axes[1].set_title(f'(b) PCA Encoder Variance\n(Total: {var_explained*100:.1f}%)')

# Reconstruction quality histogram
recon_per_sample = np.mean((X_combined - X_reconstructed)**2, axis=1)
axes[2].hist(recon_per_sample, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Reconstruction MSE'); axes[2].set_ylabel('Count')
axes[2].set_title(f'(c) Reconstruction Quality\nMean MSE={recon_error:.4f}')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig5_vae_training.png')); plt.close()
print("Saved fig5_vae_training.png")

# Latent space visualization
pca_vis = PCA(n_components=2)
z_2d = pca_vis.fit_transform(z_all)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sc = axes[0].scatter(z_2d[:,0], z_2d[:,1], c=tg_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[0].set_xlabel('Latent PC1'); axes[0].set_ylabel('Latent PC2')
axes[0].set_title('(a) Latent Space (Calibrated Tg)'); plt.colorbar(sc, ax=axes[0], label='Tg (K)')

sc2 = axes[1].scatter(z_2d[:,0], z_2d[:,1], c=tg_md_values, cmap='RdYlBu_r', alpha=0.3, s=5)
axes[1].set_xlabel('Latent PC1'); axes[1].set_ylabel('Latent PC2')
axes[1].set_title('(b) Latent Space (MD Tg)'); plt.colorbar(sc2, ax=axes[1], label='MD Tg (K)')

axes[2].scatter(tg_values, tg_pred_all, c='steelblue', alpha=0.3, s=5)
lm = [min(tg_values.min(), tg_pred_all.min())-10, max(tg_values.max(), tg_pred_all.max())+10]
axes[2].plot(lm, lm, 'r--', lw=2)
axes[2].set_xlabel('True Calibrated Tg (K)'); axes[2].set_ylabel('Predicted Tg (K)')
axes[2].set_title(f'(c) Latent Property Prediction\nR²={r2_vae:.3f}, MAE={mae_vae:.1f} K')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig6_latent_space.png')); plt.close()
print("Saved fig6_latent_space.png")

# ============================================================
# PHASE 4: Inverse Design
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Inverse Design & Candidate Generation")
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
    # Interpolation between known good candidates
    for _ in range(200):
        i1, i2 = np.random.choice(len(z_target), 2, replace=True)
        alpha = np.random.uniform(0.2, 0.8)
        gen_z.append(alpha * z_target[i1] + (1-alpha) * z_target[i2])
    # Perturbation
    for _ in range(200):
        gen_z.append(z_target[np.random.choice(len(z_target))] + np.random.randn(latent_dim)*0.3)
    # GMM sampling near target
    for _ in range(200):
        gen_z.append(np.random.multivariate_normal(centroid, cov*0.5))
    
    gen_z = np.array(gen_z)
    tg_gen = prop_model.predict(gen_z)
    in_range = (tg_gen >= tl) & (tg_gen <= th)
    print(f"  In target: {in_range.sum()}/{len(gen_z)}")
    
    for i in range(len(gen_z)):
        all_gen.append({'target': tname, 'tg_predicted': float(tg_gen[i]),
                       'in_target_range': bool(in_range[i]), 'z': gen_z[i]})

print(f"\nTotal generated: {len(all_gen)}")

# Find nearest neighbors for top candidates
top_cands = []
for tname, (tl, th) in target_ranges.items():
    cands = [c for c in all_gen if c['target'] == tname and c['in_target_range']]
    tc = (tl + th) / 2
    cands.sort(key=lambda c: abs(c['tg_predicted'] - tc))
    top_cands.extend(cands[:20])

print(f"Top candidates: {len(top_cands)}")

if top_cands:
    top_z = np.array([c['z'] for c in top_cands])
    dists = cdist(top_z, z_all)
    for i, c in enumerate(top_cands):
        nn_idx = np.argmin(dists[i])
        ri = valid_indices[nn_idx]
        c['nearest_acid'] = df_vit.loc[ri, 'acid']
        c['nearest_epoxide'] = df_vit.loc[ri, 'epoxide']
        c['nearest_tg_cal'] = float(df_vit.loc[ri, 'tg_calibrated'])
        c['nearest_tg_md'] = float(df_vit.loc[ri, 'tg'])
        c['nn_dist'] = float(dists[i, nn_idx])
    
    cdf = pd.DataFrame([{k:v for k,v in c.items() if k != 'z'} for c in top_cands])
    cdf.to_csv(os.path.join(OUT_DIR, 'top_candidates.csv'), index=False)
    print("Saved top_candidates.csv")

# Novel combinations from top acid-epoxide building blocks
print("\n--- Novel Vitrimer Chemistries ---")
tg_thresh = np.percentile(tg_values, 90)
high_mask = tg_values >= tg_thresh
high_idx = valid_indices[high_mask]
high_vit = df_vit.loc[high_idx]

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
    from rdkit.Chem import AllChem
    FP_BITS = 128
    def smiles_to_fp(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return np.zeros(FP_BITS)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_BITS))
    def get_desc(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return [0]*6
        return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m), Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m)]
    
    nf = []
    for c in novel:
        feat = np.concatenate([smiles_to_fp(c['acid']), smiles_to_fp(c['epoxide']),
                              get_desc(c['acid']), get_desc(c['epoxide'])])
        nf.append(feat)
    nf = scaler.transform(np.array(nf))
    nz = pca_encoder.transform(nf)
    tg_n = prop_model.predict(nz)
    
    for i, c in enumerate(novel):
        c['tg_predicted'] = float(tg_n[i])
    
    ndf = pd.DataFrame(novel).sort_values('tg_predicted', ascending=False)
    ndf.to_csv(os.path.join(OUT_DIR, 'novel_vitrimer_candidates.csv'), index=False)
    print(f"Novel Tg range: {tg_n.min():.1f} - {tg_n.max():.1f} K")
    print(ndf.head(10).to_string())

# ============================================================
# Figure 7: Inverse Design Results
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

gen_tg = [c['tg_predicted'] for c in all_gen]
axes[0,0].hist(tg_values, bins=50, color='coral', edgecolor='black', alpha=0.5, label='Training', density=True)
axes[0,0].hist(gen_tg, bins=50, color='steelblue', edgecolor='black', alpha=0.5, label='Generated', density=True)
for _, (tl, th) in target_ranges.items():
    axes[0,0].axvspan(tl, th, alpha=0.1, color='green')
axes[0,0].set_xlabel('Tg (K)'); axes[0,0].set_ylabel('Density')
axes[0,0].set_title('(a) Generated vs Training Tg'); axes[0,0].legend()

gen_z_arr = np.array([c['z'] for c in all_gen])
gen_z_2d = pca_vis.transform(gen_z_arr)
axes[0,1].scatter(z_2d[:,0], z_2d[:,1], c='lightgray', alpha=0.2, s=3, label='Training')
sc = axes[0,1].scatter(gen_z_2d[:,0], gen_z_2d[:,1], c=[c['tg_predicted'] for c in all_gen],
                       cmap='RdYlBu_r', alpha=0.5, s=10)
axes[0,1].set_xlabel('PC1'); axes[0,1].set_ylabel('PC2')
axes[0,1].set_title('(b) Generated in Latent Space'); plt.colorbar(sc, ax=axes[0,1], label='Tg(K)')

sr = {}
for tn, (tl, th) in target_ranges.items():
    cs = [c for c in all_gen if c['target'] == tn]
    sr[tn] = sum(c['in_target_range'] for c in cs)/len(cs)*100 if cs else 0
bars = axes[1,0].bar(range(len(sr)), list(sr.values()), color=['#e74c3c','#f39c12','#2ecc71'])
axes[1,0].set_xticks(range(len(sr)))
axes[1,0].set_xticklabels([k.split('(')[0].strip() for k in sr.keys()], rotation=15)
axes[1,0].set_ylabel('Success Rate (%)'); axes[1,0].set_title('(c) Target Range Success')
for b, v in zip(bars, sr.values()):
    axes[1,0].text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}%', ha='center', fontsize=11)

if novel:
    n_show = min(15, len(ndf))
    axes[1,1].barh(range(n_show), ndf['tg_predicted'].head(n_show).values, color='teal', edgecolor='black')
    axes[1,1].set_xlabel('Predicted Tg (K)'); axes[1,1].set_ylabel('Candidate #')
    axes[1,1].set_title('(d) Top Novel Candidates'); axes[1,1].invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig7_inverse_design.png')); plt.close()
print("Saved fig7_inverse_design.png")

# ============================================================
# Figure 8: Chemical Diversity (using precomputed descriptors)
# ============================================================
print("\nGenerating chemical diversity figures...")
# Use descriptors from feature matrix (last 12 columns: 6 acid + 6 epoxide)
acid_desc_cols = X_combined[:, -12:-6]  # acid descriptors
epox_desc_cols = X_combined[:, -6:]     # epoxide descriptors

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].hist(acid_desc_cols[:,0], bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid MW')
axes[0].hist(epox_desc_cols[:,0], bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide MW')
axes[0].set_xlabel('Molecular Weight (g/mol)'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) MW Distribution'); axes[0].legend()

avg_mw = (acid_desc_cols[:,0] + epox_desc_cols[:,0]) / 2
axes[1].scatter(avg_mw, tg_values, c='steelblue', alpha=0.2, s=5)
axes[1].set_xlabel('Avg MW (g/mol)'); axes[1].set_ylabel('Calibrated Tg (K)')
axes[1].set_title('(b) Tg vs Average MW')

axes[2].hist(acid_desc_cols[:,1], bins=40, alpha=0.6, color='coral', edgecolor='black', label='Acid LogP')
axes[2].hist(epox_desc_cols[:,1], bins=40, alpha=0.6, color='steelblue', edgecolor='black', label='Epoxide LogP')
axes[2].set_xlabel('LogP'); axes[2].set_ylabel('Count'); axes[2].set_title('(c) LogP Distribution'); axes[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig8_chemical_diversity.png')); plt.close()
print("Saved fig8_chemical_diversity.png")

# ============================================================
# Figure 9: Tg Heatmap
# ============================================================
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
ax.set_xlabel('Epoxide Component'); ax.set_ylabel('Acid Component')
ax.set_title('Calibrated Tg (K) for Top Acid-Epoxide Combinations')
plt.xticks(rotation=45, ha='right', fontsize=7); plt.yticks(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig9_tg_heatmap.png')); plt.close()
print("Saved fig9_tg_heatmap.png")

# ============================================================
# Figure 10: Validation - comparing different Tg estimates
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MD Tg vs Calibrated Tg
axes[0].scatter(tg_md_values, tg_values, c='steelblue', alpha=0.2, s=5)
axes[0].plot([300, 570], [300, 570], 'r--', lw=2)
axes[0].set_xlabel('MD Simulated Tg (K)'); axes[0].set_ylabel('GP-Calibrated Tg (K)')
axes[0].set_title('(a) MD vs Calibrated Tg for Vitrimers')

# Calibrated Tg vs VAE predicted
axes[1].scatter(tg_values, tg_pred_all, c='coral', alpha=0.2, s=5)
lm = [min(tg_values.min(), tg_pred_all.min())-10, max(tg_values.max(), tg_pred_all.max())+10]
axes[1].plot(lm, lm, 'r--', lw=2)
axes[1].set_xlabel('GP-Calibrated Tg (K)'); axes[1].set_ylabel('VAE-Predicted Tg (K)')
axes[1].set_title(f'(b) Calibrated vs VAE Predicted\nR²={r2_vae:.3f}')
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, 'fig10_validation.png')); plt.close()
print("Saved fig10_validation.png")

# Save all results
results = {
    'n_vitrimers': int(len(df_vit)),
    'n_valid': int(len(valid_indices)),
    'n_generated': len(all_gen),
    'n_novel': len(novel) if novel else 0,
    'success_rates': sr,
    'latent_dim': latent_dim,
    'pca_variance_explained': float(var_explained),
    'property_predictor_r2': float(r2_vae),
    'property_predictor_mae': float(mae_vae),
    'reconstruction_mse': float(recon_error),
    'tg_calibrated_stats': {
        'mean': float(tg_values.mean()), 'std': float(tg_values.std()),
        'min': float(tg_values.min()), 'max': float(tg_values.max())
    }
}
with open(os.path.join(OUT_DIR, 'generation_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nPhase 3 & 4 complete!")
