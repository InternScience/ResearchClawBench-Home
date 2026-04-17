#!/usr/bin/env python3
"""Part 1: Data loading and visualization"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)
sns.set_style('whitegrid')

WORKSPACE = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260416_211933'
DATA_DIR = os.path.join(WORKSPACE, 'data')
REPORT_IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
train_df.columns = feature_cols + ['label', 'degradation']
test_df.columns = feature_cols + ['label', 'degradation']

y_train = train_df['label'].values.astype(int)
y_test = test_df['label'].values.astype(int)

print(f"Train: {len(y_train)}, Test: {len(y_test)}")
print(f"Train class dist: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")

# Fig 1: Label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
train_counts = [np.sum(y_train==0), np.sum(y_train==1)]
test_counts = [np.sum(y_test==0), np.sum(y_test==1)]
labels = ['Different (0)', 'Same (1)']

axes[0].bar(labels, train_counts, color=['#2E86AB', '#E94F37'])
axes[0].set_title('Training Set Label Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(train_counts):
    axes[0].text(i, v + 1000, str(v), ha='center', fontsize=10)

axes[1].bar(labels, test_counts, color=['#2E86AB', '#E94F37'])
axes[1].set_title('Test Set Label Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(test_counts):
    axes[1].text(i, v + 500, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig01_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig01")

# Fig 2: Degradation distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
deg_order = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
train_deg = train_df['degradation'].value_counts().reindex(deg_order)
test_deg = test_df['degradation'].value_counts().reindex(deg_order)
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

axes[0].bar(deg_order, train_deg.values, color=colors)
axes[0].set_title('Training Set - Degradation Types', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(deg_order, test_deg.values, color=colors)
axes[1].set_title('Test Set - Degradation Types', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig02_degradation_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig02")

# Fig 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
corr = train_df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=False, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig03_feature_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig03")

# Fig 4: Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sel_feat = ['0', '5', '10', '15']
feat_names = ['Feature 0', 'Feature 5', 'Feature 10', 'Feature 15']

for idx, (feat, name) in enumerate(zip(sel_feat, feat_names)):
    ax = axes[idx // 2, idx % 2]
    feat_0 = train_df[train_df['label'] == 0][feat]
    feat_1 = train_df[train_df['label'] == 1][feat]
    ax.hist(feat_0, bins=50, alpha=0.6, label='Different (0)', color='#2E86AB', density=True)
    ax.hist(feat_1, bins=50, alpha=0.6, label='Same (1)', color='#E94F37', density=True)
    ax.set_xlabel(name)
    ax.set_ylabel('Density')
    ax.set_title(f'{name} by Label', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig04_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig04")

print("Part 1 complete!")
