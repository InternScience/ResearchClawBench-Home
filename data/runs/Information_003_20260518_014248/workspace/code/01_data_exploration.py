"""
Phase 1: Data Exploration and Visualization for NF-UNSW-NB15-v2
Generates data overview figures and statistics.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os

# Load data
print("Loading data...")
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)

msg = data.msg.numpy()  # [148774, 40]
label = data.label.numpy()  # binary labels
attack = data.attack.numpy()  # multi-class labels
t = data.t.numpy()  # timestamps
src = data.src.numpy()
dst = data.dst.numpy()
dt = data.dt.numpy()

print(f"Data loaded: {msg.shape[0]} flows, {msg.shape[1]} features")
print(f"Nodes: {data.num_nodes}")

# ===================== Figure 1: Data Overview =====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1a: Binary class distribution
ax = axes[0, 0]
benign_count = (label == 0).sum()
attack_count = (label == 1).sum()
bars = ax.bar(['Benign', 'Attack'], [benign_count, attack_count], 
              color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, [benign_count, attack_count]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
            f'{count:,}\n({count/len(label)*100:.1f}%)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Flows', fontsize=12)
ax.set_title('Binary Class Distribution', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(benign_count, attack_count) * 1.25)

# 1b: Multi-class attack distribution
ax = axes[0, 1]
attack_names = {
    0: 'Type-0', 1: 'Type-1', 2: 'Benign', 3: 'Type-3', 4: 'Type-4',
    5: 'Type-5', 6: 'Type-6', 7: 'Type-7', 8: 'Type-8', 9: 'Type-9'
}
attack_counts = []
attack_labels_list = []
for a in sorted(np.unique(attack)):
    cnt = (attack == a).sum()
    attack_counts.append(cnt)
    attack_labels_list.append(f'{attack_names[a]}\n(A{a})')
colors = ['#95a5a6' if a == 2 else plt.cm.Set2(i/9) 
          for i, a in enumerate(sorted(np.unique(attack)))]
bars = ax.bar(range(len(attack_counts)), attack_counts, color=colors, 
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(attack_labels_list)))
ax.set_xticklabels(attack_labels_list, fontsize=8)
ax.set_ylabel('Number of Flows', fontsize=12)
ax.set_title('Multi-class Distribution (10 types)', fontsize=13, fontweight='bold')
ax.set_yscale('log')
for bar, count in zip(bars, attack_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.15,
            f'{count:,}', ha='center', va='bottom', fontsize=8)

# 1c: Temporal distribution
ax = axes[0, 2]
# Create time bins
time_bins = np.linspace(0, t.max(), 50)
benign_times = t[label == 0]
attack_times = t[label == 1]
ax.hist(benign_times, bins=time_bins, alpha=0.7, label='Benign', color='#2ecc71', density=True)
ax.hist(attack_times, bins=time_bins, alpha=0.7, label='Attack', color='#e74c3c', density=True)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Temporal Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

# 1d: Feature variance distribution
ax = axes[1, 0]
feature_vars = msg.var(axis=0)
ax.bar(range(40), feature_vars, color=plt.cm.viridis(feature_vars / feature_vars.max()), 
       edgecolor='none')
ax.set_xlabel('Feature Index', fontsize=12)
ax.set_ylabel('Variance', fontsize=12)
ax.set_title('Feature Variance (40 features)', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 40, 5))

# 1e: Feature correlation heatmap (top 15 features by variance)
ax = axes[1, 1]
top_features = np.argsort(feature_vars)[-15:][::-1]
corr_matrix = np.corrcoef(msg[:, top_features].T)
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(15))
ax.set_xticklabels([f'F{i}' for i in top_features], fontsize=7, rotation=45)
ax.set_yticks(range(15))
ax.set_yticklabels([f'F{i}' for i in top_features], fontsize=7)
ax.set_title('Feature Correlation (Top-15)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# 1f: Graph structure - degree distribution (sampled)
ax = axes[1, 2]
# Compute degree for sampled nodes
np.random.seed(42)
sample_nodes = np.random.choice(data.num_nodes, min(5000, data.num_nodes), replace=False)
degree = np.zeros(data.num_nodes)
np.add.at(degree, src, 1)
np.add.at(degree, dst, 1)
sample_degrees = degree[sample_nodes]
sample_degrees = sample_degrees[sample_degrees > 0]
ax.hist(sample_degrees, bins=np.logspace(0, np.log10(sample_degrees.max()+1), 30), 
        color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Node Degree (log scale)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Node Degree Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/data_overview.png")

# ===================== Figure 2: Feature Distributions by Class =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Feature distributions for benign vs attack (selected features)
ax = axes[0, 0]
selected_feats = [0, 6, 7, 8, 10, 15, 20, 30]
feat_names = [f'Feature {i}' for i in selected_feats]
x_pos = np.arange(len(selected_feats))
width = 0.35
benign_means = [msg[label==0, f].mean() for f in selected_feats]
attack_means = [msg[label==1, f].mean() for f in selected_feats]
ax.bar(x_pos - width/2, benign_means, width, label='Benign', color='#2ecc71', alpha=0.8)
ax.bar(x_pos + width/2, attack_means, width, label='Attack', color='#e74c3c', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' for i in selected_feats], fontsize=9)
ax.set_ylabel('Mean Value', fontsize=12)
ax.set_title('Feature Means: Benign vs Attack', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

# 2b: Boxplot of key features
ax = axes[0, 1]
key_feats = [6, 7, 8]  # Most distinctive features
data_for_box = []
labels_for_box = []
for f in key_feats:
    data_for_box.extend(msg[label==0, f].tolist()[:1000])
    labels_for_box.extend([f'F{f}\nBenign'] * 1000)
    data_for_box.extend(msg[label==1, f].tolist()[:1000])
    labels_for_box.extend([f'F{f}\nAttack'] * 1000)
df_box = pd.DataFrame({'Value': data_for_box, 'Feature': labels_for_box})
positions = []
colors_box = []
for i, f in enumerate(key_feats):
    positions.extend([i*5, i*5+1])
    colors_box.extend(['#2ecc71', '#e74c3c'])
sns.boxplot(x='Feature', y='Value', data=df_box, ax=ax, palette=['#2ecc71', '#e74c3c'] * 3)
ax.set_title('Key Feature Distributions', fontsize=13, fontweight='bold')

# 2c: t-SNE of raw features
ax = axes[1, 0]
from sklearn.manifold import TSNE
np.random.seed(42)
n_sample = min(5000, msg.shape[0])
idx = np.random.choice(msg.shape[0], n_sample, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb = tsne.fit_transform(msg[idx])
scatter = ax.scatter(emb[:, 0], emb[:, 1], c=label[idx], cmap='RdYlGn_r', 
                     s=5, alpha=0.6, edgecolors='none')
ax.set_title('t-SNE of Raw Features (colored by class)', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(['Benign', 'Attack'])

# 2d: t-SNE colored by attack type
ax = axes[1, 1]
scatter = ax.scatter(emb[:, 0], emb[:, 1], c=attack[idx], cmap='tab10', 
                     s=5, alpha=0.6, edgecolors='none')
ax.set_title('t-SNE of Raw Features (colored by attack type)', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
cbar.set_ticklabels([attack_names[i] for i in range(10)], fontsize=8)

plt.tight_layout()
plt.savefig('report/images/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/feature_distributions.png")

# Save data statistics
stats = {
    "total_flows": int(msg.shape[0]),
    "feature_dim": int(msg.shape[1]),
    "num_nodes": int(data.num_nodes),
    "num_attack_types": 10,
    "binary_distribution": {
        "benign": int(benign_count),
        "attack": int(attack_count),
        "attack_ratio": float(attack_count / len(label))
    },
    "attack_type_distribution": {
        attack_names[int(a)]: {"count": int((attack == a).sum()), 
                                "ratio": float((attack == a).sum() / len(attack))}
        for a in sorted(np.unique(attack))
    },
    "temporal_range": {"min": int(t.min()), "max": int(t.max())},
    "feature_variance_top5": feature_vars.argsort()[-5:][::-1].tolist(),
    "unique_src_nodes": int(data.src.unique().shape[0]),
    "unique_dst_nodes": int(data.dst.unique().shape[0])
}
with open('outputs/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved: outputs/data_statistics.json")

print("\nPhase 1 complete!")
