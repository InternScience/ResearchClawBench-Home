"""
Data Overview and Exploratory Data Analysis
- Generates comprehensive data overview figures
- Statistical summaries and correlation analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = 'report/images'

# Load data
features = np.load('outputs/features.npy')
targets = np.load('outputs/targets.npy')
feature_names = list(np.load('outputs/feature_names.npy', allow_pickle=True))
target_names = list(np.load('outputs/target_names.npy', allow_pickle=True))

# ============================================================
# FIGURE 15: Data overview - Feature distributions
# ============================================================
fig, axes = plt.subplots(4, 5, figsize=(24, 16))
fig.suptitle('Feature Distributions in Materials Dataset', fontsize=16, fontweight='bold', y=1.02)

for idx in range(min(20, features.shape[1])):
    ax = axes[idx // 5, idx % 5]
    ax.hist(features[:, idx], bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title(feature_names[idx], fontsize=10)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(labelsize=7)

# Hide extra axes
for idx in range(features.shape[1], 20):
    axes[idx // 5, idx % 5].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig15_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig15_feature_distributions.png")

# ============================================================
# FIGURE 16: Target property distributions
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Target Property Distributions', fontsize=14, fontweight='bold')

nice_names = ['Formation Energy (eV)', 'Band Gap (eV)', 'Bulk Modulus (GPa)', 'Thermal Conductivity (W/mK)']
for idx in range(4):
    ax = axes[idx]
    ax.hist(targets[:, idx], bins=30, alpha=0.7, color=f'C{idx}', edgecolor='white')
    ax.axvline(x=np.mean(targets[:, idx]), color='red', linestyle='--', label=f'Mean: {np.mean(targets[:, idx]):.2f}')
    ax.set_xlabel(nice_names[idx])
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.set_title(nice_names[idx])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig16_target_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig16_target_distributions.png")

# ============================================================
# FIGURE 17: Correlation heatmap
# ============================================================
# Combine features and targets for correlation analysis
selected_features = [8, 9, 10, 11, 14, 15, 16, 17]  # Key structural and chemical features
selected_names = [feature_names[i] for i in selected_features] + target_names
combined = np.column_stack([features[:, selected_features], targets])

corr_matrix = np.corrcoef(combined.T)

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=selected_names, yticklabels=selected_names, ax=ax,
            vmin=-1, vmax=1, square=True, linewidths=0.5)
ax.set_title('Feature-Target Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig17_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig17_correlation_heatmap.png")

# ============================================================
# FIGURE 18: Pairwise scatter plots for key features
# ============================================================
key_features = [15, 16, 14, 8]  # electronegativity, atomic_radius, volume, lattice_a
key_names = [feature_names[i] for i in key_features]
key_data = features[:, key_features]

fig, axes = plt.subplots(len(key_features), len(key_features), figsize=(16, 16))
fig.suptitle('Pairwise Feature Relationships', fontsize=14, fontweight='bold', y=1.02)

for i in range(len(key_features)):
    for j in range(len(key_features)):
        ax = axes[i, j]
        if i == j:
            ax.hist(key_data[:, i], bins=25, alpha=0.7, color='steelblue', edgecolor='white')
        else:
            ax.scatter(key_data[:, j], key_data[:, i], alpha=0.3, s=10, c='steelblue')
        if i == len(key_features) - 1:
            ax.set_xlabel(key_names[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(key_names[i], fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig18_pairwise_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig18_pairwise_scatter.png")

# Save data summary
data_summary = {
    'n_samples': int(features.shape[0]),
    'n_features': int(features.shape[1]),
    'n_targets': int(targets.shape[1]),
    'feature_names': feature_names,
    'target_names': target_names,
    'target_statistics': {}
}

for idx, tname in enumerate(target_names):
    data_summary['target_statistics'][tname] = {
        'mean': float(np.mean(targets[:, idx])),
        'std': float(np.std(targets[:, idx])),
        'min': float(np.min(targets[:, idx])),
        'max': float(np.max(targets[:, idx])),
        'median': float(np.median(targets[:, idx]))
    }

with open('outputs/data_summary.json', 'w') as f:
    json.dump(data_summary, f, indent=2)

print("\nEDA complete!")
