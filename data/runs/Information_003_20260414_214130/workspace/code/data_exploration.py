"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-Scale Fusion and Few-Shot Learning
Data Exploration and Preprocessing
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os

# Create directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
print("Loading data...")
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)

# Extract components
src = data.src.numpy()
dst = data.dst.numpy()
t = data.t.numpy()
msg = data.msg.numpy()  # 40-dim features
labels = data.label.numpy()  # binary: 0=attack, 1=benign
attacks = data.attack.numpy()  # multi-class: 0-9

print(f"Total edges: {len(src)}")
print(f"Total nodes: {data.num_nodes}")
print(f"Feature dim: {msg.shape[1]}")
print(f"Time range: {t.min()} - {t.max()}")

# Binary label distribution
print("\n=== Binary Classification ===")
benign_count = np.sum(labels == 1)
attack_count = np.sum(labels == 0)
print(f"Benign: {benign_count} ({100*benign_count/len(labels):.1f}%)")
print(f"Attack: {attack_count} ({100*attack_count/len(labels):.1f}%)")

# Multi-class distribution
print("\n=== Multi-class Classification ===")
attack_types = {
    0: 'Analysis', 1: 'Backdoor', 2: 'Benign', 3: 'DoS',
    4: 'Exploits', 5: 'Fuzzers', 6: 'Generic', 7: 'Reconnaissance',
    8: 'Shellcode', 9: 'Worms'
}
attack_counts = Counter(attacks)
for k in sorted(attack_counts.keys()):
    name = attack_types.get(k, f'Unknown_{k}')
    count = attack_counts[k]
    pct = 100 * count / len(attacks)
    print(f"  {name} (class {k}): {count} ({pct:.2f}%)")

# Feature statistics
print("\n=== Feature Statistics ===")
print(f"Feature mean: {msg.mean():.4f}")
print(f"Feature std: {msg.std():.4f}")
print(f"Feature min: {msg.min():.4f}")
print(f"Feature max: {msg.max():.4f}")

# Per-class feature means for disentanglement analysis
print("\n=== Per-class Feature Analysis ===")
class_feature_means = {}
for cls in np.unique(attacks):
    mask = attacks == cls
    class_feature_means[int(cls)] = msg[mask].mean(axis=0).tolist()

# Save statistics
stats = {
    'total_edges': int(len(src)),
    'total_nodes': int(data.num_nodes),
    'feature_dim': int(msg.shape[1]),
    'binary_distribution': {'benign': int(benign_count), 'attack': int(attack_count)},
    'multi_class_distribution': {str(k): int(v) for k, v in attack_counts.items()},
    'attack_type_names': attack_types,
    'feature_stats': {
        'mean': float(msg.mean()),
        'std': float(msg.std()),
        'min': float(msg.min()),
        'max': float(msg.max())
    }
}
with open('outputs/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n=== Generating Visualizations ===")

# Figure 1: Binary class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Benign', 'Attack'], [benign_count, attack_count], color=colors)
axes[0].set_ylabel('Count')
axes[0].set_title('Binary Class Distribution')
for i, v in enumerate([benign_count, attack_count]):
    axes[0].text(i, v + 500, str(v), ha='center', fontweight='bold')

# Multi-class distribution
class_names = [attack_types.get(k, f'?{k}') for k in sorted(attack_counts.keys())]
class_vals = [attack_counts[k] for k in sorted(attack_counts.keys())]
colors_multi = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
bars = axes[1].barh(class_names, class_vals, color=colors_multi)
axes[1].set_xlabel('Count')
axes[1].set_title('Multi-class Distribution (Attack Types)')
for bar, val in zip(bars, class_vals):
    axes[1].text(val + 100, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: class_distribution.png")

# Figure 2: Feature distribution per class (heatmap of mean features)
fig, ax = plt.subplots(figsize=(14, 6))
class_order = sorted(class_feature_means.keys())
feature_matrix = np.array([class_feature_means[c] for c in class_order])
class_labels = [attack_types.get(c, f'?{c}') for c in class_order]
sns.heatmap(feature_matrix, yticklabels=class_labels, cmap='YlOrRd', ax=ax)
ax.set_xlabel('Feature Index')
ax.set_title('Mean Feature Values per Attack Type')
plt.tight_layout()
plt.savefig('report/images/feature_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_heatmap.png")

# Figure 3: Temporal distribution
fig, ax = plt.subplots(figsize=(12, 4))
time_bins = np.linspace(t.min(), t.max(), 100)
for cls in [2, 3, 6, 7]:  # Benign, DoS, Generic, Reconnaissance
    mask = attacks == cls
    ax.hist(t[mask], bins=time_bins, alpha=0.5, label=attack_types.get(cls, str(cls)))
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Count')
ax.set_title('Temporal Distribution of Traffic by Attack Type')
ax.legend()
plt.tight_layout()
plt.savefig('report/images/temporal_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: temporal_distribution.png")

# Figure 4: Feature variance analysis for disentanglement
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
feature_vars = msg.var(axis=0)
axes[0].bar(range(40), feature_vars)
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Variance')
axes[0].set_title('Feature Variance (Identifying Disentanglement Candidates)')

# Inter-class feature separability
inter_class_var = np.var([class_feature_means[c] for c in class_order], axis=0)
axes[1].bar(range(40), inter_class_var)
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Inter-class Variance')
axes[1].set_title('Inter-class Feature Variance (Discriminative Features)')
plt.tight_layout()
plt.savefig('report/images/feature_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_variance.png")

# Figure 5: Class imbalance visualization (log scale)
fig, ax = plt.subplots(figsize=(10, 5))
sorted_classes = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)
names = [attack_types.get(k, f'?{k}') for k, v in sorted_classes]
vals = [v for k, v in sorted_classes]
colors = ['#e74c3c' if v < 500 else '#3498db' for v in vals]
bars = ax.bar(names, vals, color=colors)
ax.set_yscale('log')
ax.set_ylabel('Count (log scale)')
ax.set_title('Class Distribution (Few-shot classes highlighted in red)')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.1, str(val), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/class_imbalance_log.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: class_imbalance_log.png")

print("\nData exploration complete!")
