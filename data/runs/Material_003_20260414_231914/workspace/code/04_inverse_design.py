"""
Step 4: Inverse design - generate candidates with target Tg values.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
vit = pd.read_csv('outputs/vitrimer_calibrated.csv')
latent_df = pd.read_csv('outputs/vitrimer_latent.csv')

# Define VAE model (same architecture)
class MolecularVAE(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=64, hidden_dim=512):
        super().__init__()
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
            nn.Sigmoid()
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict_tg(self, z):
        return self.predictor(z).squeeze(-1)

# Load model
model = MolecularVAE()
model.load_state_dict(torch.load('outputs/vae_model.pt', weights_only=True))
model.eval()

# Define target Tg ranges
targets = [
    {'name': 'Low_Tg_RoomTemp', 'min': 300, 'max': 350, 'desc': 'Room-temperature reprocessable'},
    {'name': 'Mid_Tg_Automotive', 'min': 380, 'max': 420, 'desc': 'Automotive/engineering'},
    {'name': 'High_Tg_Structural', 'min': 450, 'max': 500, 'desc': 'High-temperature structural'},
]

# Inverse design via latent space optimization
z_cols = [f'z_{i}' for i in range(64)]
z_all = latent_df[z_cols].values
tg_all = latent_df['tg_calibrated'].values

# Use nearest-neighbor search in latent space for each target
nn_model = NearestNeighbors(n_neighbors=50, metric='euclidean')
nn_model.fit(z_all)

results = {}
for target in targets:
    # Find candidates within target Tg range
    mask = (tg_all >= target['min']) & (tg_all <= target['max'])
    candidates_idx = np.where(mask)[0]
    
    if len(candidates_idx) == 0:
        # Find closest
        center_tg = (target['min'] + target['max']) / 2
        distances = np.abs(tg_all - center_tg)
        candidates_idx = np.argsort(distances)[:100]
    
    # Get their latent vectors and find diverse set
    candidate_z = z_all[candidates_idx]
    candidate_tg = tg_all[candidates_idx]
    
    # Sort by distance to target center
    center_tg = (target['min'] + target['max']) / 2
    sort_idx = np.argsort(np.abs(candidate_tg - center_tg))
    
    # Select top candidates with diversity (pick every Nth to ensure spread)
    n_select = min(20, len(sort_idx))
    step = max(1, len(sort_idx) // n_select)
    selected_idx = sort_idx[::step][:n_select]
    
    selected_candidates = []
    for idx in selected_idx:
        orig_idx = candidates_idx[idx]
        selected_candidates.append({
            'acid': vit.iloc[orig_idx]['acid'],
            'epoxide': vit.iloc[orig_idx]['epoxide'],
            'tg_md': float(vit.iloc[orig_idx]['tg']),
            'tg_calibrated': float(vit.iloc[orig_idx]['tg_calibrated']),
            'tg_calibrated_std': float(vit.iloc[orig_idx]['tg_calibrated_std']),
        })
    
    results[target['name']] = {
        'target_range': [target['min'], target['max']],
        'description': target['desc'],
        'n_candidates_in_range': int(mask.sum()),
        'top_candidates': selected_candidates,
    }
    
    print(f"\n{target['name']} (Tg: {target['min']}-{target['max']} K):")
    print(f"  Candidates in range: {mask.sum()}")
    print(f"  Top candidate Tg: {selected_candidates[0]['tg_calibrated']:.1f} K")

# Save results
with open('outputs/inverse_design_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# --- Inverse design plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Tg distribution with target ranges
ax = axes[0, 0]
ax.hist(tg_all, bins=60, alpha=0.5, color='gray', label='All vitrimers')
colors = ['steelblue', 'coral', 'seagreen']
for i, target in enumerate(targets):
    ax.axvspan(target['min'], target['max'], alpha=0.2, color=colors[i], label=target['name'])
ax.set_xlabel('Calibrated Tg (K)')
ax.set_ylabel('Count')
ax.set_title('Vitrimer Tg Distribution with Target Ranges')
ax.legend(fontsize=8)

# Plot 2: Candidates per target
ax = axes[0, 1]
target_names = [t['name'].replace('_', '\n') for t in targets]
n_cands = [results[t['name']]['n_candidates_in_range'] for t in targets]
bars = ax.bar(target_names, n_cands, color=colors, alpha=0.7)
ax.set_ylabel('Number of Candidates')
ax.set_title('Candidates per Target Tg Range')
for bar, n in zip(bars, n_cands):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(n), ha='center', fontsize=10)

# Plot 3: Latent space with target candidates highlighted
ax = axes[1, 0]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_all)
ax.scatter(z_pca[:, 0], z_pca[:, 1], c='lightgray', s=2, alpha=0.3, label='All')
for i, target in enumerate(targets):
    mask = (tg_all >= target['min']) & (tg_all <= target['max'])
    ax.scatter(z_pca[mask, 0], z_pca[mask, 1], c=colors[i], s=10, alpha=0.5, label=target['name'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Latent Space: Target Candidates Highlighted')
ax.legend(fontsize=8)

# Plot 4: Top candidates Tg comparison
ax = axes[1, 1]
x_pos = 0
tick_positions = []
tick_labels = []
for i, target in enumerate(targets):
    cands = results[target['name']]['top_candidates'][:5]
    for j, c in enumerate(cands):
        ax.bar(x_pos, c['tg_calibrated'], color=colors[i], alpha=0.7, 
               yerr=c['tg_calibrated_std'], capsize=3)
        tick_positions.append(x_pos)
        tick_labels.append(f"T{i+1}-{j+1}")
        x_pos += 1
    x_pos += 0.5  # gap between targets
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
ax.set_ylabel('Calibrated Tg (K)')
ax.set_title('Top Candidates: Calibrated Tg')
# Add target range lines
for i, target in enumerate(targets):
    ax.axhspan(target['min'], target['max'], alpha=0.05, color=colors[i])

plt.tight_layout()
plt.savefig('report/images/inverse_design.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Candidate analysis plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, target in enumerate(targets):
    ax = axes[i]
    cands = results[target['name']]['top_candidates']
    tgs = [c['tg_calibrated'] for c in cands]
    stds = [c['tg_calibrated_std'] for c in cands]
    
    y_pos = range(len(tgs))
    ax.barh(y_pos, tgs, xerr=stds, color=colors[i], alpha=0.7, capsize=3)
    ax.axvspan(target['min'], target['max'], alpha=0.1, color='green')
    ax.set_xlabel('Calibrated Tg (K)')
    ax.set_ylabel('Candidate Rank')
    ax.set_title(f"{target['name'].replace('_', ' ')}\n({target['min']}-{target['max']} K)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{j+1}" for j in y_pos], fontsize=8)

plt.tight_layout()
plt.savefig('report/images/candidate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nInverse design complete.")
print("Results saved to outputs/inverse_design_results.json")
print("Plots saved to report/images/inverse_design.png and candidate_analysis.png")
