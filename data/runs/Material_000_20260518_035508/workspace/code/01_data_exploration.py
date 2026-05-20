"""
Step 1: Data exploration and visualization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
pretrain = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)

print(f"Pretrain: {len(pretrain)} samples")
print(f"Finetune: {len(finetune)} samples")
print(f"Candidate: {len(candidate)} samples")

# Collect statistics
def get_stats(dataset):
    node_counts = [dataset[i].x.shape[0] for i in range(len(dataset))]
    edge_counts = [dataset[i].edge_index.shape[1] for i in range(len(dataset))]
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    features = torch.cat([dataset[i].x for i in range(len(dataset))], dim=0)
    return node_counts, edge_counts, labels, features

pt_nodes, pt_edges, pt_labels, pt_feat = get_stats(pretrain)
ft_nodes, ft_edges, ft_labels, ft_feat = get_stats(finetune)
cd_nodes, cd_edges, cd_labels, cd_feat = get_stats(candidate)

# Figure 1: Dataset Overview
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Node count distributions
axes[0,0].hist(pt_nodes, bins=range(2, 26), alpha=0.7, label='Pretrain', color='steelblue', edgecolor='black')
axes[0,0].set_xlabel('Number of Nodes (Atoms)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Node Count Distribution (Pretrain)')
axes[0,0].legend()

axes[0,1].hist(ft_nodes, bins=range(2, 26), alpha=0.7, label='Finetune', color='coral', edgecolor='black')
axes[0,1].set_xlabel('Number of Nodes (Atoms)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Node Count Distribution (Finetune)')
axes[0,1].legend()

axes[0,2].hist(cd_nodes, bins=range(2, 26), alpha=0.7, label='Candidate', color='mediumseagreen', edgecolor='black')
axes[0,2].set_xlabel('Number of Nodes (Atoms)')
axes[0,2].set_ylabel('Count')
axes[0,2].set_title('Node Count Distribution (Candidate)')
axes[0,2].legend()

# Edge count distributions
axes[1,0].hist(pt_edges, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1,0].set_xlabel('Number of Edges')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Edge Count Distribution (Pretrain)')

axes[1,1].hist(ft_edges, bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[1,1].set_xlabel('Number of Edges')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Edge Count Distribution (Finetune)')

axes[1,2].hist(cd_edges, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
axes[1,2].set_xlabel('Number of Edges')
axes[1,2].set_ylabel('Count')
axes[1,2].set_title('Edge Count Distribution (Candidate)')

plt.tight_layout()
plt.savefig('report/images/fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_dataset_overview.png")

# Figure 2: Label Balance and Feature Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Finetune label distribution
ft_pos = sum(ft_labels)
ft_neg = len(ft_labels) - ft_pos
axes[0].bar(['Negative', 'Positive\n(Altermagnet)'], [ft_neg, ft_pos], 
            color=['#ff7f0e', '#1f77b4'], edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Finetune Label Distribution\n({ft_pos/len(ft_labels)*100:.1f}% positive)')
for i, v in enumerate([ft_neg, ft_pos]):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Candidate hidden labels (for reference)
cd_pos = sum(cd_labels)
cd_neg = len(cd_labels) - cd_pos
axes[1].bar(['Negative', 'Positive\n(Altermagnet)'], [cd_neg, cd_pos],
            color=['#ff7f0e', '#1f77b4'], edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Candidate Hidden Labels\n({cd_pos/len(cd_labels)*100:.1f}% positive)')
for i, v in enumerate([cd_neg, cd_pos]):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Feature heatmap (mean of each feature across all datasets)
all_features = torch.cat([pt_feat.mean(dim=0), ft_feat.mean(dim=0), cd_feat.mean(dim=0)]).reshape(3, 28)
im = axes[2].imshow(all_features.numpy(), cmap='viridis', aspect='auto')
axes[2].set_xlabel('Feature Index')
axes[2].set_ylabel('Dataset')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['Pretrain', 'Finetune', 'Candidate'])
axes[2].set_title('Mean Feature Values')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig('report/images/fig2_label_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_label_features.png")

# Figure 3: Feature distribution comparison between positive and negative in finetune
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
ft_feat_pos = torch.cat([finetune[i].x for i in range(len(finetune)) if ft_labels[i]==1])
ft_feat_neg = torch.cat([finetune[i].x for i in range(len(finetune)) if ft_labels[i]==0])

for idx in range(12):
    ax = axes[idx // 4, idx % 4]
    ax.hist(ft_feat_neg[:, idx].numpy(), bins=30, alpha=0.5, label='Negative', color='steelblue', density=True)
    if ft_feat_pos.shape[0] > 0:
        ax.hist(ft_feat_pos[:, idx].numpy(), bins=30, alpha=0.5, label='Positive', color='coral', density=True)
    ax.set_title(f'Feature {idx}')
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: Positive vs Negative (Finetune)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig3_feature_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_feature_comparison.png")

# Save statistics to outputs
stats = {
    'pretrain': {'n': len(pretrain), 'mean_nodes': np.mean(pt_nodes), 'mean_edges': np.mean(pt_edges),
                 'feat_dim': 28, 'edge_feat_dim': 2},
    'finetune': {'n': len(finetune), 'mean_nodes': np.mean(ft_nodes), 'mean_edges': np.mean(ft_edges),
                 'n_positive': ft_pos, 'n_negative': ft_neg},
    'candidate': {'n': len(candidate), 'mean_nodes': np.mean(cd_nodes), 'mean_edges': np.mean(cd_edges),
                  'hidden_positive': cd_pos, 'hidden_negative': cd_neg}
}
import json
with open('outputs/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved dataset_stats.json")
print("\nData exploration complete!")
