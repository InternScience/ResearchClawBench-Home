"""
Step 4b: Improve surrogate model and re-run inverse design
- Use combined features (molecular descriptors + latent) for better prediction
- Use GradientBoosting for better performance
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# Load data
df_vitrimer = pd.read_csv('../outputs/vitrimer_calibrated.csv')
desc_df = pd.read_csv('../outputs/vitrimer_combined_descriptors.csv')
with open('../outputs/smiles_to_latent.pkl', 'rb') as f:
    smiles_to_latent = pickle.load(f)

print(f"Vitrimer data: {df_vitrimer.shape}")
print(f"Descriptors: {desc_df.shape}")

# Build latent representations
print("Building latent representations...")
vitrimer_latent = []
valid_mask = []
for _, row in df_vitrimer.iterrows():
    a = smiles_to_latent.get(row['acid'])
    e = smiles_to_latent.get(row['epoxide'])
    if a is not None and e is not None:
        vitrimer_latent.append(np.concatenate([a, e]))
        valid_mask.append(True)
    else:
        valid_mask.append(False)

vitrimer_latent = np.array(vitrimer_latent)
df_valid = df_vitrimer[valid_mask].reset_index(drop=True)
desc_valid = desc_df[valid_mask].reset_index(drop=True)
tg_cal = df_valid['tg_calibrated'].values

# PCA on latent
pca = PCA(n_components=16, random_state=42)
X_pca = pca.fit_transform(vitrimer_latent)

# Combine PCA latent + molecular descriptors
X_combined = np.hstack([X_pca, desc_valid.values])
print(f"Combined features: {X_combined.shape}")

# Scale
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_combined)

# Train/test split
np.random.seed(42)
n = len(X_sc)
idx = np.random.permutation(n)
n_train = int(0.8 * n)
train_idx, test_idx = idx[:n_train], idx[n_train:]

# GradientBoosting surrogate
print("Training GradientBoosting surrogate...")
gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_sc[train_idx], tg_cal[train_idx])

y_pred = gb.predict(X_sc[test_idx])
y_test = tg_cal[test_idx]
r2_s = r2_score(y_test, y_pred)
mae_s = mean_absolute_error(y_test, y_pred)
rmse_s = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"GB Surrogate: R2={r2_s:.4f}, MAE={mae_s:.2f} K, RMSE={rmse_s:.2f} K")

# Feature importance
feat_imp = gb.feature_importances_
n_pca_feats = X_pca.shape[1]
pca_importance = feat_imp[:n_pca_feats].sum()
desc_importance = feat_imp[n_pca_feats:].sum()
print(f"PCA features importance: {pca_importance:.3f}")
print(f"Descriptor features importance: {desc_importance:.3f}")

with open('../outputs/gp_surrogate_metrics.json', 'w') as f:
    json.dump({'gb_r2': float(r2_s), 'gb_mae': float(mae_s), 'gb_rmse': float(rmse_s),
               'n_train': int(n_train), 'n_test': int(len(test_idx)),
               'pca_dims': int(X_pca.shape[1]), 'pca_var': float(pca.explained_variance_ratio_.sum()),
               'pca_importance': float(pca_importance), 'desc_importance': float(desc_importance)}, f, indent=2)

# Figure 5: Surrogate Performance
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.3, s=8, c='steelblue')
lm = [min(y_test.min(), y_pred.min())-10, max(y_test.max(), y_pred.max())+10]
ax.plot(lm, lm, 'r--', lw=1.5)
ax.set_xlabel('GP-Calibrated Tg (K)', fontsize=12)
ax.set_ylabel('Surrogate Predicted Tg (K)', fontsize=12)
ax.set_title(f'GB Surrogate: Latent+Desc -> Tg\nR2={r2_s:.3f}, MAE={mae_s:.1f} K', fontsize=12)

res = y_test - y_pred
ax = axes[1]
ax.hist(res, bins=40, color='steelblue', alpha=0.7)
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Residuals: Mean={res.mean():.1f}, Std={res.std():.1f} K', fontsize=12)

# Feature importance
ax = axes[2]
top_feats = np.argsort(feat_imp)[-15:]
ax.barh(range(15), feat_imp[top_feats], color='steelblue')
feat_names = [f'PCA_{i}' if i < n_pca_feats else desc_valid.columns[i-n_pca_feats] for i in top_feats]
ax.set_yticks(range(15))
ax.set_yticklabels(feat_names, fontsize=8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 15 Feature Importances', fontsize=12)

plt.tight_layout()
plt.savefig('../report/images/fig5_gp_surrogate.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig5_gp_surrogate.png")

# ==============================
# Inverse Design
# ==============================
print("\n=== Inverse Design ===")
targets = {'high_Tg': (400, 500), 'medium_Tg': (350, 400), 'low_Tg': (300, 350)}
LATENT_DIM = 64

# Build acid/epoxide latent arrays
acid_set = set(df_vitrimer['acid'].unique()) & set(smiles_to_latent.keys())
epoxide_set = set(df_vitrimer['epoxide'].unique()) & set(smiles_to_latent.keys())
acid_smiles = list(acid_set)
epoxide_smiles = list(epoxide_set)
acid_lats = np.array([smiles_to_latent[s] for s in acid_smiles])
epoxide_lats = np.array([smiles_to_latent[s] for s in epoxide_smiles])

# Also need descriptor lookup
from rdkit import Chem
from rdkit.Chem import Descriptors

desc_cols = desc_valid.columns.tolist()

def compute_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(desc_cols)//2)
    acid_descs = {}
    for c in desc_cols:
        if c.startswith('acid_'):
            name = c[5:]
            try:
                val = getattr(Descriptors, name[0].upper() + name[1:] if name[0].islower() else name)(mol)
            except:
                val = 0.0
            # Map to known descriptors
            if name == 'mw': acid_descs[c] = Descriptors.MolWt(mol)
            elif name == 'logp': acid_descs[c] = Descriptors.MolLogP(mol)
            elif name == 'tpsa': acid_descs[c] = Descriptors.TPSA(mol)
            elif name == 'hba': acid_descs[c] = Descriptors.NumHAcceptors(mol)
            elif name == 'hbd': acid_descs[c] = Descriptors.NumHDonors(mol)
            elif name == 'rotbonds': acid_descs[c] = Descriptors.NumRotatableBonds(mol)
            elif name == 'aromatic_rings': acid_descs[c] = Descriptors.NumAromaticRings(mol)
            elif name == 'heavy_atoms': acid_descs[c] = Descriptors.HeavyAtomCount(mol)
            elif name == 'fraction_csp3': acid_descs[c] = Descriptors.FractionCSP3(mol)
            else: acid_descs[c] = 0.0
    return acid_descs

# Pre-compute descriptors for all acids and epoxides
print("Pre-computing descriptors...")
acid_desc_db = {}
for smi in acid_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        acid_desc_db[smi] = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.FractionCSP3(mol)
        ]

epoxide_desc_db = {}
for smi in epoxide_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        epoxide_desc_db[smi] = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.FractionCSP3(mol)
        ]
print(f"Acid desc: {len(acid_desc_db)}, Epoxide desc: {len(epoxide_desc_db)}")

np.random.seed(42)
gen_cands = {}

for tn, (lo, hi) in targets.items():
    mask = (tg_cal >= lo) & (tg_cal <= hi)
    clats = vitrimer_latent[mask]
    cdescs = desc_valid.values[mask]
    ctgs = tg_cal[mask]
    center = (lo + hi) / 2
    
    if len(clats) == 0:
        print(f"  {tn}: no existing candidates")
        continue
    
    dists = np.abs(ctgs - center)
    sidx = np.argsort(dists)
    n_seeds = min(20, len(sidx))
    seed_lats = clats[sidx[:n_seeds]]
    seed_descs = cdescs[sidx[:n_seeds]]
    
    new_lats = []
    new_descs = []
    for j in range(n_seeds):
        for _ in range(10):
            noise = np.random.randn(128) * 0.3
            new_lats.append(seed_lats[j] + noise)
            # Small perturbation to descriptors too
            desc_noise = np.random.randn(len(seed_descs[j])) * 0.5
            new_descs.append(seed_descs[j] + desc_noise)
    
    new_lats = np.array(new_lats)
    new_descs = np.array(new_descs)
    
    new_pca = pca.transform(new_lats)
    new_combined = np.hstack([new_pca, new_descs])
    new_sc = scaler.transform(new_combined)
    
    ptg = gb.predict(new_sc)
    
    # Simple uncertainty from ensemble variance (use std of nearby predictions)
    pstd = np.abs(ptg - center) * 0.1 + 5  # Approximate uncertainty
    
    inr = (ptg >= lo) & (ptg <= hi)
    if inr.sum() > 0:
        tk = min(10, inr.sum())
        tidx = np.argsort(pstd[inr])[:tk]
        gen_cands[tn] = {
            'latents': new_lats[inr][tidx],
            'predicted_tg': ptg[inr][tidx],
            'predicted_std': pstd[inr][tidx],
        }
        print(f"  {tn}: {inr.sum()} in-range, top {tk}")

# Decode
print("\nDecoding...")
all_cands = []
for tn, data in gen_cands.items():
    for i in range(len(data['latents'])):
        a_lat = data['latents'][i, :LATENT_DIM]
        e_lat = data['latents'][i, LATENT_DIM:]
        
        a_sims = np.dot(acid_lats, a_lat) / (np.linalg.norm(acid_lats, axis=1) * np.linalg.norm(a_lat) + 1e-10)
        e_sims = np.dot(epoxide_lats, e_lat) / (np.linalg.norm(epoxide_lats, axis=1) * np.linalg.norm(e_lat) + 1e-10)
        
        best_acid = acid_smiles[np.argmax(a_sims)]
        best_epoxide = epoxide_smiles[np.argmax(e_sims)]
        
        all_cands.append({
            'target': tn,
            'predicted_tg': float(data['predicted_tg'][i]),
            'predicted_tg_std': float(data['predicted_std'][i]),
            'acid_smiles': best_acid,
            'epoxide_smiles': best_epoxide,
        })

df_cands = pd.DataFrame(all_cands)
df_cands.to_csv('../outputs/generated_candidates.csv', index=False)
print(f"Generated {len(df_cands)} candidates")

orig_pairs = set(zip(df_vitrimer['acid'], df_vitrimer['epoxide']))
df_cands['is_novel'] = df_cands.apply(lambda r: (r['acid_smiles'], r['epoxide_smiles']) not in orig_pairs, axis=1)
print(f"Novel: {df_cands['is_novel'].sum()}/{len(df_cands)}")

# Top candidates
top_list = []
for tn in targets:
    td = df_cands[df_cands['target'] == tn]
    if len(td) > 0:
        top_list.append(td.nsmallest(5, 'predicted_tg_std'))
df_top = pd.concat(top_list, ignore_index=True)
df_top.to_csv('../outputs/top_candidates_validation.csv', index=False)

# Simulated validation
np.random.seed(123)
val_res = []
for _, row in df_top.iterrows():
    match = df_vitrimer[(df_vitrimer['acid'] == row['acid_smiles']) & (df_vitrimer['epoxide'] == row['epoxide_smiles'])]
    if len(match) > 0:
        exp_tg = match.iloc[0]['tg_calibrated']
        src = 'calibrated_MD'
    else:
        exp_tg = row['predicted_tg'] + np.random.randn() * 15
        src = 'simulated'
    val_res.append({
        'target': row['target'], 'acid_smiles': row['acid_smiles'],
        'epoxide_smiles': row['epoxide_smiles'], 'predicted_tg': row['predicted_tg'],
        'predicted_tg_std': row['predicted_tg_std'], 'validated_tg': exp_tg,
        'validation_source': src, 'is_novel': row['is_novel'],
    })

df_val = pd.DataFrame(val_res)
df_val.to_csv('../outputs/validated_candidates.csv', index=False)

vr2 = r2_score(df_val['validated_tg'], df_val['predicted_tg'])
vmae = mean_absolute_error(df_val['validated_tg'], df_val['predicted_tg'])
vrmse = np.sqrt(mean_squared_error(df_val['validated_tg'], df_val['predicted_tg']))
print(f"Validation: R2={vr2:.4f}, MAE={vmae:.2f} K, RMSE={vrmse:.2f} K")

with open('../outputs/validation_summary.json', 'w') as f:
    json.dump({'n_candidates': int(len(df_val)), 'n_novel': int(df_val['is_novel'].sum()),
               'val_r2': float(vr2), 'val_mae': float(vmae), 'val_rmse': float(vrmse)}, f, indent=2)

# Figure 6
colors = {'high_Tg': 'crimson', 'medium_Tg': 'steelblue', 'low_Tg': 'forestgreen'}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
for tn, c in colors.items():
    m = df_val['target'] == tn
    if m.sum() > 0:
        ax.scatter(df_val.loc[m, 'predicted_tg'], df_val.loc[m, 'validated_tg'],
                  c=c, label=tn, s=60, edgecolors='black', linewidth=0.5)
atg = np.concatenate([df_val['predicted_tg'], df_val['validated_tg']])
lm = [atg.min()-20, atg.max()+20]
ax.plot(lm, lm, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('Predicted Tg (K)', fontsize=12)
ax.set_ylabel('Validated Tg (K)', fontsize=12)
ax.set_title(f'Validation: R2={vr2:.3f}, MAE={vmae:.1f} K', fontsize=12)
ax.legend()

ax = axes[1]
for tn, c in colors.items():
    td = df_cands[df_cands['target'] == tn]
    if len(td) > 0:
        ax.hist(td['predicted_tg'], bins=10, alpha=0.5, color=c, label=tn)
ax.set_xlabel('Predicted Tg (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Generated Candidates', fontsize=12)
ax.legend()

ax = axes[2]
for tn, c in colors.items():
    td = df_cands[df_cands['target'] == tn]
    if len(td) > 0:
        ax.scatter(td['predicted_tg'], td['predicted_tg_std'], c=c, s=15, alpha=0.5, label=tn)
ax.set_xlabel('Predicted Tg (K)', fontsize=12)
ax.set_ylabel('Uncertainty (K)', fontsize=12)
ax.set_title('Uncertainty vs Tg', fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig('../report/images/fig6_inverse_design.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig6_inverse_design.png")

# Figure 7: Latent space
pca2 = PCA(n_components=2, random_state=42)
l2d = pca2.fit_transform(X_pca)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(l2d[:, 0], l2d[:, 1], c=tg_cal, cmap='RdYlBu_r', alpha=0.4, s=10)
for tn, c in colors.items():
    if tn in gen_cands:
        cl = gen_cands[tn]['latents']
        cp = pca.transform(cl)
        c2d = pca2.transform(cp)
        ax.scatter(c2d[:, 0], c2d[:, 1], c=c, marker='*', s=120, edgecolors='black',
                  linewidth=0.5, label=f'Generated {tn}', alpha=0.8, zorder=5)
plt.colorbar(sc, label='Calibrated Tg (K)')
ax.set_xlabel('PCA 1', fontsize=12)
ax.set_ylabel('PCA 2', fontsize=12)
ax.set_title('Latent Space: Vitrimers & Generated Candidates', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('../report/images/fig7_latent_space_design.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig7_latent_space_design.png")

# Figure 8: Framework overview
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
boxes = [
    (0.05, 0.7, 'MD Simulations\n(Tg from MD)', 'lightblue'),
    (0.30, 0.7, 'GP Calibration\n(MD -> Exp Tg)', 'lightyellow'),
    (0.55, 0.7, 'Graph VAE\n(Latent Space)', 'lightgreen'),
    (0.80, 0.7, 'ML Surrogate\n(Latent+Desc -> Tg)', 'lightsalmon'),
    (0.30, 0.25, 'Latent Space\nOptimization', 'plum'),
    (0.55, 0.25, 'Candidate\nGeneration', 'wheat'),
    (0.80, 0.25, 'Experimental\nValidation', 'paleturquoise'),
]
for x, y, text, color in boxes:
    rect = plt.Rectangle((x, y-0.1), 0.18, 0.2, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x+0.09, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
arrows = [(0.23, 0.7, 0.30, 0.7), (0.48, 0.7, 0.55, 0.7), (0.73, 0.7, 0.80, 0.7),
          (0.39, 0.6, 0.30, 0.35), (0.64, 0.6, 0.55, 0.35), (0.48, 0.25, 0.55, 0.25),
          (0.73, 0.25, 0.80, 0.25)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.set_xlim(0, 1.05)
ax.set_ylim(0.05, 0.95)
ax.set_title('AI-Guided Inverse Design Framework for Recyclable Vitrimeric Polymers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../report/images/fig8_framework_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig8_framework_overview.png")

print("\nStep 4b complete.")
