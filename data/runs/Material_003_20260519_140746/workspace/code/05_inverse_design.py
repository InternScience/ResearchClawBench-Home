import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load model and data
with open('outputs/vae_config.pkl', 'rb') as f:
    saved = pickle.load(f)
    config = saved['config']
    scaler_acid = saved['scaler_acid']
    scaler_epoxide = saved['scaler_epoxide']
    scaler_tg = saved['scaler_tg']

latent_dim = config['latent_dim']
hidden_dim = config['hidden_dim']
input_dim = config['input_dim']

# Reconstruct model
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class PropertyPredictor(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, z_acid, z_epoxide):
        z = torch.cat([z_acid, z_epoxide], dim=1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)

class VitrimerVAE(torch.nn.Module):
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

model = VitrimerVAE(input_dim, hidden_dim, latent_dim)
model.load_state_dict(torch.load('outputs/vae_model.pt'))
model.eval()

# Load original data for nearest neighbor lookup
md = pd.read_csv('data/tg_vitrimer_MD.csv')
md_sample = md.sample(n=min(4000, len(md)), random_state=42).reset_index(drop=True)

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
acid_features_s = scaler_acid.transform(acid_features)
epoxide_features_s = scaler_epoxide.transform(epoxide_features)

# Build nearest neighbor models
nn_acid = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_epoxide = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_acid.fit(acid_features_s)
nn_epoxide.fit(epoxide_features_s)

# Get latent representations
with torch.no_grad():
    acid_mu, _ = model.acid_encoder(torch.tensor(acid_features_s, dtype=torch.float32))
    epoxide_mu, _ = model.epoxide_encoder(torch.tensor(epoxide_features_s, dtype=torch.float32))
    acid_z = acid_mu.numpy()
    epoxide_z = epoxide_mu.numpy()

# Inverse Design: Target Tg ranges
target_tgs = [350, 400, 450, 500]  # Target calibrated Tg values in K
results = []

# For inverse design, we'll optimize latent vectors to match target Tg
# Simple gradient-based optimization in latent space
for target_tg in target_tgs:
    target_tg_s = scaler_tg.transform(np.array([[target_tg]])).flatten()[0]
    
    # Start from random latent vectors
    best_candidates = []
    
    for trial in range(100):
        z_acid = torch.randn(1, latent_dim, requires_grad=True)
        z_epoxide = torch.randn(1, latent_dim, requires_grad=True)
        optimizer = torch.optim.Adam([z_acid, z_epoxide], lr=0.1)
        
        for step in range(200):
            optimizer.zero_grad()
            tg_pred = model.predictor(z_acid, z_epoxide)
            loss = (tg_pred - target_tg_s)**2
            loss.backward()
            optimizer.step()
        
        # Decode to fingerprints
        with torch.no_grad():
            acid_fp_recon = model.acid_decoder(z_acid).numpy()
            epoxide_fp_recon = model.epoxide_decoder(z_epoxide).numpy()
            tg_pred_final = model.predictor(z_acid, z_epoxide).item()
            tg_pred_final_K = scaler_tg.inverse_transform(np.array([[tg_pred_final]])).flatten()[0]
        
        # Find nearest neighbors
        dist_acid, idx_acid = nn_acid.kneighbors(acid_fp_recon)
        dist_epoxide, idx_epoxide = nn_epoxide.kneighbors(epoxide_fp_recon)
        
        acid_smiles = md_sample['acid'].values[idx_acid[0][0]]
        epoxide_smiles = md_sample['epoxide'].values[idx_epoxide[0][0]]
        
        best_candidates.append({
            'target_tg': target_tg,
            'predicted_tg': tg_pred_final_K,
            'acid_smiles': acid_smiles,
            'epoxide_smiles': epoxide_smiles,
            'acid_dist': float(dist_acid[0][0]),
            'epoxide_dist': float(dist_epoxide[0][0]),
            'z_acid': z_acid.detach().numpy().flatten().tolist(),
            'z_epoxide': z_epoxide.detach().numpy().flatten().tolist(),
        })
    
    # Sort by how close predicted Tg is to target
    best_candidates.sort(key=lambda x: abs(x['predicted_tg'] - target_tg))
    results.extend(best_candidates[:5])

candidates_df = pd.DataFrame(results)
candidates_df.to_csv('outputs/inverse_design_candidates.csv', index=False)

print(f"Generated {len(candidates_df)} candidate molecules.")
print(candidates_df[['target_tg', 'predicted_tg', 'acid_smiles', 'epoxide_smiles', 'acid_dist', 'epoxide_dist']].head(20))

# Now validate candidates using GP calibration model
import joblib
model_data = joblib.load('outputs/gp_model.pkl')
gp = model_data['gp']
scaler_X_gp = model_data['scaler_X']
scaler_y_gp = model_data['scaler_y']

def morgan_fp(smiles, radius=2, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))

# For candidates, we need MD Tg first. Since we don't have MD for new molecules,
# we'll use a proxy: average MD Tg of nearest neighbors in the dataset
# Then apply GP calibration

nn_md_tg = NearestNeighbors(n_neighbors=5, metric='euclidean')
all_fp = np.array([get_features(s) for s in md_sample['acid']]) + np.array([get_features(s) for s in md_sample['epoxide']])
all_fp_s = StandardScaler().fit_transform(all_fp)
nn_md_tg.fit(all_fp_s)

validation_results = []
for _, row in candidates_df.iterrows():
    acid_fp = get_features(row['acid_smiles'])
    epoxide_fp = get_features(row['epoxide_smiles'])
    combined_fp = acid_fp + epoxide_fp
    combined_fp_s = StandardScaler().fit(all_fp).transform(combined_fp.reshape(1, -1))
    
    dists, idxs = nn_md_tg.kneighbors(combined_fp_s)
    md_tg_proxy = md_sample['tg'].values[idxs[0]].mean()
    
    # GP calibration
    fp_acid = morgan_fp(row['acid_smiles'])
    fp_epoxide = morgan_fp(row['epoxide_smiles'])
    fp_combined = fp_acid + fp_epoxide
    X_gp = np.hstack([fp_combined, [md_tg_proxy]]).reshape(1, -1)
    X_gp_s = scaler_X_gp.transform(X_gp)
    
    cal_tg_s, cal_std_s = gp.predict(X_gp_s, return_std=True)
    cal_tg = scaler_y_gp.inverse_transform(cal_tg_s.reshape(-1, 1)).flatten()[0]
    cal_std = cal_std_s * scaler_y_gp.scale_[0]
    
    validation_results.append({
        'target_tg': row['target_tg'],
        'vae_predicted_tg': row['predicted_tg'],
        'md_tg_proxy': md_tg_proxy,
        'gp_calibrated_tg': cal_tg,
        'gp_std': cal_std,
        'acid_smiles': row['acid_smiles'],
        'epoxide_smiles': row['epoxide_smiles'],
    })

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv('outputs/validation_candidates.csv', index=False)

print("\n=== Validation Results ===")
print(validation_df[['target_tg', 'vae_predicted_tg', 'md_tg_proxy', 'gp_calibrated_tg', 'gp_std']].head(20))

# Create figures
def create_figures():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Target vs Predicted Tg
    ax = axes[0, 0]
    for target in target_tgs:
        subset = candidates_df[candidates_df['target_tg'] == target]
        ax.scatter([target]*len(subset), subset['predicted_tg'], alpha=0.6, s=50, label=f'Target {target}K')
    ax.plot([300, 550], [300, 550], 'r--', lw=1.5)
    ax.set_xlabel('Target Tg (K)')
    ax.set_ylabel('VAE Predicted Tg (K)')
    ax.set_title('Inverse Design: Target vs Predicted Tg')
    ax.legend(fontsize=8)
    
    # 2. GP Calibrated vs Target
    ax = axes[0, 1]
    for target in target_tgs:
        subset = validation_df[validation_df['target_tg'] == target]
        ax.scatter([target]*len(subset), subset['gp_calibrated_tg'], alpha=0.6, s=50, label=f'Target {target}K')
    ax.plot([300, 550], [300, 550], 'r--', lw=1.5)
    ax.set_xlabel('Target Tg (K)')
    ax.set_ylabel('GP Calibrated Tg (K)')
    ax.set_title('Validation: GP Calibrated vs Target Tg')
    ax.legend(fontsize=8)
    
    # 3. Latent space interpolation
    ax = axes[1, 0]
    # Sample 500 points and color by Tg
    sample_idx = np.random.choice(len(acid_z), 500, replace=False)
    from sklearn.decomposition import PCA
    combined_z = np.hstack([acid_z, epoxide_z])
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(combined_z)
    
    # Load calibrated Tg
    md_cal = pd.read_csv('outputs/vitrimer_calibrated.csv')
    tg_vals = md_cal['tg_calibrated'].values[:len(z_pca)]
    
    scatter = ax.scatter(z_pca[sample_idx, 0], z_pca[sample_idx, 1], c=tg_vals[sample_idx], cmap='plasma', alpha=0.6, s=15)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Latent Space: Training Data')
    plt.colorbar(scatter, ax=ax, label='Calibrated Tg (K)')
    
    # 4. Generated candidates in latent space
    ax = axes[1, 1]
    ax.scatter(z_pca[sample_idx, 0], z_pca[sample_idx, 1], c='lightgray', alpha=0.3, s=10, label='Training')
    
    # Project candidate latent vectors
    candidate_z = []
    for _, row in candidates_df.iterrows():
        candidate_z.append(row['z_acid'] + row['z_epoxide'])
    candidate_z = np.array(candidate_z)
    candidate_z_pca = pca.transform(candidate_z)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, target in enumerate(target_tgs):
        subset_idx = candidates_df['target_tg'] == target
        ax.scatter(candidate_z_pca[subset_idx, 0], candidate_z_pca[subset_idx, 1], 
                  c=colors[i], s=80, marker='*', edgecolors='k', linewidths=0.5, 
                  label=f'Target {target}K', alpha=0.8, zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Generated Candidates in Latent Space')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig04_inverse_design.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")

create_figures()

# Save summary statistics
summary = {
    'num_candidates_generated': len(candidates_df),
    'target_temperatures': target_tgs,
    'mean_vae_prediction_error_by_target': {
        str(t): float(candidates_df[candidates_df['target_tg']==t]['predicted_tg'].mean() - t)
        for t in target_tgs
    },
    'mean_gp_calibration_error_by_target': {
        str(t): float(validation_df[validation_df['target_tg']==t]['gp_calibrated_tg'].mean() - t)
        for t in target_tgs
    },
}
with open('outputs/inverse_design_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Inverse design complete.")
