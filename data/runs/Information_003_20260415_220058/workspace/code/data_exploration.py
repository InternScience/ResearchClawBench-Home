"""
DIDS-MFL: Data Exploration and Visualization
Generates initial data overview figures for the NF-UNSW-NB15-v2 dataset.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Paths
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt')
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
torch.serialization.add_safe_globals([__import__('torch_geometric.data.temporal').data.temporal.TemporalData])
data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)

# Extract arrays
msg = data.msg.numpy()       # (148774, 40)
attack = data.attack.numpy() # attack types 0-9
label = data.label.numpy()   # binary 0/1
t = data.t.numpy()           # timestamps
dt = data.dt.numpy()         # duration
src = data.src.numpy()
dst = data.dst.numpy()

# Attack type mapping (based on NF-UNSW-NB15-v2 dataset)
ATTACK_NAMES = {
    0: 'Backdoor',
    1: 'Analysis',
    2: 'Benign',
    3: 'DoS',
    4: 'Exploits',
    5: 'Fuzzers',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms'
}

# Few-shot threshold: <1500 samples
FEW_SHOT_TYPES = [0, 1, 4, 5, 8, 9]
NORMAL_TYPES = [3, 6, 7]

# ===================== Figure 1: Attack Type Distribution =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary distribution
benign_count = (label == 0).sum()
attack_count = (label == 1).sum()
axes[0].bar(['Benign', 'Attack'], [benign_count, attack_count], color=['#4CAF50', '#F44336'])
axes[0].set_title('Binary Label Distribution', fontsize=14)
axes[0].set_ylabel('Number of Flows')
for i, v in enumerate([benign_count, attack_count]):
    axes[0].text(i, v + 1000, f'{v} ({v/len(label)*100:.1f}%)', ha='center', fontsize=11)

# Multi-class distribution
attack_counts = {}
for atype in sorted(ATTACK_NAMES.keys()):
    count = (attack == atype).sum()
    attack_counts[ATTACK_NAMES[atype]] = count

names = list(attack_counts.keys())
counts = list(attack_counts.values())
colors = ['#4CAF50' if n == 'Benign' else ('#FF9800' if counts[i] < 1500 else '#F44336') 
          for i, n in enumerate(names)]
bars = axes[1].bar(names, counts, color=colors)
axes[1].set_title('Attack Type Distribution', fontsize=14)
axes[1].set_ylabel('Number of Flows')
axes[1].tick_params(axis='x', rotation=45)
# Add few-shot annotation
for i, (n, c) in enumerate(zip(names, counts)):
    if c < 1500 and n != 'Benign':
        axes[1].annotate('few-shot', xy=(i, c), fontsize=8, color='#FF9800', 
                        ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig1_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 2: Feature Distribution by Attack Type =====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
feature_indices = [0, 1, 2, 6, 11, 12]  # Key features

for idx, fi in enumerate(feature_indices):
    ax = axes[idx // 3, idx % 3]
    for atype in [2, 3, 6, 7, 0, 9]:  # Benign + major attacks + few-shot
        mask = attack == atype
        if mask.sum() > 0:
            feat_vals = msg[mask, fi]
            name = ATTACK_NAMES[atype]
            alpha = 0.6 if atype == 2 else 0.4
            ax.hist(feat_vals, bins=50, alpha=alpha, label=name, density=True)
    ax.set_title(f'Feature {fi} Distribution', fontsize=12)
    ax.set_xlabel(f'Feature {fi} Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig2_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 3: Temporal Pattern =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Timeline of events
time_bins = np.linspace(0, 86400, 100)
bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

benign_time_hist = np.histogram(t[label == 0], bins=time_bins)[0]
attack_time_hist = np.histogram(t[label == 1], bins=time_bins)[0]

axes[0].plot(bin_centers, benign_time_hist, label='Benign', color='#4CAF50', alpha=0.7)
axes[0].plot(bin_centers, attack_time_hist, label='Attack', color='#F44336', alpha=0.7)
axes[0].set_title('Temporal Distribution of Flows', fontsize=14)
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Number of Flows')
axes[0].legend()

# Duration distribution by attack type
for atype in [2, 3, 6, 7]:
    mask = attack == atype
    dt_vals = dt[mask]
    name = ATTACK_NAMES[atype]
    axes[1].hist(dt_vals, bins=50, alpha=0.5, label=name, density=True)
axes[1].set_title('Duration Distribution by Attack Type', fontsize=14)
axes[1].set_xlabel('Duration (normalized)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig3_temporal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 4: Feature Correlation Heatmaps =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, atype in enumerate([2, 6, 7]):  # Benign, Generic, Reconnaissance
    mask = attack == atype
    feat_subset = msg[mask, :20]  # First 20 features for readability
    corr = np.corrcoef(feat_subset.T)
    name = ATTACK_NAMES[atype]
    sns.heatmap(corr, ax=axes[idx], cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True,
                xticklabels=False, yticklabels=False)
    axes[idx].set_title(f'{name} (n={mask.sum()})', fontsize=12)

plt.suptitle('Feature Correlation Heatmaps (First 20 Features)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig4_correlation_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 5: Entangled Distribution Visualization =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA visualization showing entangled vs separated distributions
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(msg)

# Plot benign vs all attacks
for atype in [2, 0, 3, 6, 7, 9]:
    mask = attack == atype
    name = ATTACK_NAMES[atype]
    n_samples = min(mask.sum(), 500)
    indices = np.where(mask)[0][:n_samples]
    axes[0].scatter(pca_result[indices, 0], pca_result[indices, 1], 
                    alpha=0.3, s=10, label=f'{name} ({mask.sum()})')
axes[0].set_title('PCA: All Attack Types (Entangled)', fontsize=14)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(fontsize=8, markerscale=3)

# Highlight few-shot attacks specifically
for atype in [2, 0, 1, 9]:
    mask = attack == atype
    name = ATTACK_NAMES[atype]
    n_samples = min(mask.sum(), 500)
    indices = np.where(mask)[0][:n_samples]
    axes[1].scatter(pca_result[indices, 0], pca_result[indices, 1], 
                    alpha=0.5, s=15, label=f'{name} ({mask.sum()})')
axes[1].set_title('PCA: Few-shot Attacks vs Benign', fontsize=14)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(fontsize=8, markerscale=3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig5_entangled_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Save Data Statistics =====================
stats = {
    'total_events': len(label),
    'benign_count': int(benign_count),
    'attack_count': int(attack_count),
    'num_features': msg.shape[1],
    'num_nodes': len(torch.cat([data.src, data.dst]).unique()),
    'attack_type_counts': {ATTACK_NAMES[k]: int((attack == k).sum()) for k in ATTACK_NAMES},
    'few_shot_types': FEW_SHOT_TYPES,
    'normal_attack_types': NORMAL_TYPES,
    'timestamp_range': [int(t.min()), int(t.max())],
    'feature_range': [float(msg.min()), float(msg.max())]
}

with open(os.path.join(OUTPUT_DIR, 'data_statistics.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("Data exploration complete. Figures saved.")