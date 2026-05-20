import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt

# --- GVAE Definition (reused from gvae.py) ---
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        features = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
        atom_features.append(features)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    if not edge_index:
        return None
    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )

class GraphVAE(nn.Module):
    def __init__(self, node_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_atom = nn.Linear(hidden_dim, 10)
        self.decoder_bond = nn.Linear(hidden_dim, 4)

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        return h

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# --- Load Models ---
model = GraphVAE(node_dim=4, hidden_dim=64, latent_dim=32)
model.load_state_dict(torch.load('outputs/gvae_model.pth'))
model.eval()

with open('outputs/rf_vit_md_model.pkl', 'rb') as f:
    rf_vit = pickle.load(f)
with open('outputs/gp_model.pkl', 'rb') as f:
    gp = pickle.load(f)

def get_features_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        fp_arr = np.array(list(fp.ToBitString()), dtype=int)
        descs = [Descriptors.MolWt(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
        return np.concatenate([fp_arr, descs])
    return None

# 1. Encode all existing acids and epoxides
df_vit = pd.read_csv('data/tg_vitrimer_MD.csv')
acids = df_vit['acid'].unique()
epoxides = df_vit['epoxide'].unique()

def encode_smiles_list(smiles_list):
    graphs = [smiles_to_graph(s) for s in smiles_list]
    valid = [(s, g) for s, g in zip(smiles_list, graphs) if g is not None]
    if not valid:
        return [], []
    s_list, g_list = zip(*valid)
    loader = DataLoader(list(g_list), batch_size=64, shuffle=False)
    latents = []
    with torch.no_grad():
        for data in loader:
            mu, _ = model.encode(data)
            latents.append(mu.numpy())
    return list(s_list), np.concatenate(latents, axis=0)

acid_smiles, acid_latents = encode_smiles_list(acids)
epox_smiles, epox_latents = encode_smiles_list(epoxides)

# 2. Inverse Design via Latent Space Search
target_tg_exp = 450.0
print("Performing Latent Space Search...")

n_samples = 2000
acid_indices = np.random.randint(0, len(acid_smiles), n_samples)
epox_indices = np.random.randint(0, len(epox_smiles), n_samples)

X_pred = []
for a_idx, e_idx in zip(acid_indices, epox_indices):
    f_a = get_features_fp(acid_smiles[a_idx])
    f_e = get_features_fp(epox_smiles[e_idx])
    if f_a is not None and f_e is not None:
        X_pred.append(np.concatenate([f_a, f_e]))

X_pred = np.array(X_pred)
tg_md_pred = rf_vit.predict(X_pred)

# Fix GP input dimension: We use the features of the acid + tg_md_pred
# This matches the training dimension of the GP (260 features)
X_gp_pred = []
for i in range(len(X_pred)):
    # X_pred[i] has 518 features (acid 259 + epoxide 259)
    # We take the first 259 (acid features) and append tg_md_pred
    acid_feats = X_pred[i, :259]
    tg_md = tg_md_pred[i]
    X_gp_pred.append(np.concatenate([acid_feats, [tg_md]]))

X_gp_pred = np.array(X_gp_pred)
tg_exp_pred, tg_exp_std = gp.predict(X_gp_pred, return_std=True)

distances = np.abs(tg_exp_pred - target_tg_exp)
best_idx = np.argmin(distances)

best_acid = acid_smiles[acid_indices[best_idx]]
best_epox = epox_smiles[epox_indices[best_idx]]
final_tg_md = tg_md_pred[best_idx]
final_tg_exp = tg_exp_pred[best_idx]
final_std = tg_exp_std[best_idx]

print(f"Target Tg: {target_tg_exp}K")
print(f"Best Acid: {best_acid}")
print(f"Best Epoxide: {best_epox}")
print(f"Predicted MD Tg: {final_tg_md:.2f}K")
print(f"Predicted Exp Tg: {final_tg_exp:.2f} +/- {final_std:.2f}K")

candidates = {
    "target_tg": target_tg_exp,
    "acid": best_acid,
    "epoxide": best_epox,
    "pred_tg_md": float(final_tg_md),
    "pred_tg_exp": float(final_tg_exp),
    "std": float(final_std)
}
with open('outputs/optimized_candidates.json', 'w') as f:
    json.dump(candidates, f, indent=2)

# Plotting
plt.figure(figsize=(8, 6))
plt.hist(tg_exp_pred, bins=30, alpha=0.6, label='Generated Candidates')
plt.axvline(target_tg_exp, color='r', linestyle='--', label=f'Target Tg = {target_tg_exp}K')
plt.axvline(final_tg_exp, color='g', linestyle='-', linewidth=2, label=f'Best Candidate Tg = {final_tg_exp:.1f}K')
plt.xlabel('Predicted Experimental Tg (K)')
plt.ylabel('Count')
plt.title('Inverse Design: Distribution of Generated Vitrimer Candidates')
plt.legend()
plt.grid(True)
plt.savefig('report/images/inverse_design_results.png', dpi=100)
plt.close()

df_cal = pd.read_csv('data/tg_calibration.csv')
plt.figure(figsize=(6, 6))
plt.scatter(df_cal['tg_md'], df_cal['tg_exp'], alpha=0.3, label='Existing Data')
plt.scatter(final_tg_md, final_tg_exp, color='red', s=100, label='New Candidate', zorder=5)
plt.errorbar(final_tg_md, final_tg_exp, yerr=final_std, fmt='none', color='red')
plt.xlabel('MD Simulated Tg (K)')
plt.ylabel('Experimental Tg (K)')
plt.title('Validation of New Vitrimer Candidate')
plt.legend()
plt.grid(True)
plt.savefig('report/images/candidate_validation.png', dpi=100)
plt.close()

print("Done.")
