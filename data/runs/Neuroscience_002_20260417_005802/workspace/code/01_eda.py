"""
Exploratory Data Analysis for Connectomics Proofreading Binary Classification
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Paths
DATA_DIR = 'data'
IMG_DIR = 'report/images'
OUT_DIR = 'outputs'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train label distribution:\n{train['label'].value_counts()}")
print(f"Test label distribution:\n{test['label'].value_counts()}")

# ============================================================
# Figure 1: Data Overview - Feature distributions
# ============================================================
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for idx, col in enumerate(feature_cols):
    ax = axes[idx // 5, idx % 5]
    ax.hist(train.loc[train['label']==0, col], bins=50, alpha=0.6, label='No merge (0)', density=True, color='steelblue')
    ax.hist(train.loc[train['label']==1, col], bins=50, alpha=0.6, label='Merge (1)', density=True, color='coral')
    ax.set_title(f'Feature {col}', fontsize=10)
    ax.set_xlabel('')
    if idx == 0:
        ax.legend(fontsize=8)
plt.suptitle('Feature Distributions by Class Label', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_distributions.png")

# ============================================================
# Figure 2: Correlation heatmap
# ============================================================
corr = train[feature_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            xticklabels=feature_cols, yticklabels=feature_cols, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved correlation_heatmap.png")

# ============================================================
# Figure 3: Per-degradation analysis
# ============================================================
deg_types = train['degradation'].unique()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, deg in enumerate(sorted(deg_types)):
    subset = train[train['degradation'] == deg]
    counts = subset['label'].value_counts()
    pos_rate = counts.get(1.0, 0) / len(subset) * 100
    axes[i].bar(['No merge (0)', 'Merge (1)'], [counts.get(0.0, 0), counts.get(1.0, 0)],
                color=['steelblue', 'coral'])
    axes[i].set_title(f'{deg}\n(Positive rate: {pos_rate:.1f}%)', fontsize=12)
    axes[i].set_ylabel('Count')
plt.suptitle('Label Distribution by Degradation Type', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'degradation_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved degradation_label_distribution.png")

# ============================================================
# Figure 4: Feature means by degradation and label
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for i, deg in enumerate(sorted(deg_types)):
    ax = axes[i // 2, i % 2]
    subset = train[train['degradation'] == deg]
    means_0 = subset[subset['label'] == 0][feature_cols].mean()
    means_1 = subset[subset['label'] == 1][feature_cols].mean()
    x = np.arange(20)
    width = 0.35
    ax.bar(x - width/2, means_0, width, label='No merge (0)', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, means_1, width, label='Merge (1)', color='coral', alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Value')
    ax.set_title(f'{deg}', fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=8)
plt.suptitle('Feature Means by Degradation Type and Label', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'feature_means_by_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_means_by_degradation.png")

# ============================================================
# Summary statistics
# ============================================================
summary = {}
for deg in sorted(deg_types):
    subset = train[train['degradation'] == deg]
    pos = (subset['label'] == 1).sum()
    neg = (subset['label'] == 0).sum()
    summary[deg] = {'total': len(subset), 'positive': int(pos), 'negative': int(neg),
                    'positive_rate': float(pos / len(subset))}

with open(os.path.join(OUT_DIR, 'data_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved data_summary.json")

# Feature statistics
feat_stats = train[feature_cols].describe().to_dict()
with open(os.path.join(OUT_DIR, 'feature_statistics.json'), 'w') as f:
    json.dump(feat_stats, f, indent=2)
print("Saved feature_statistics.json")

# Correlation with label
label_corr = train[feature_cols].corrwith(train['label']).sort_values(ascending=False)
print("\nFeature correlation with label:")
print(label_corr)
label_corr.to_json(os.path.join(OUT_DIR, 'feature_label_correlation.json'))

# t-SNE or PCA visualization (PCA for speed)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# Sample for speed
sample_idx = np.random.RandomState(42).choice(len(train), 10000, replace=False)
X_sample = train.iloc[sample_idx][feature_cols].values
y_sample = train.iloc[sample_idx]['label'].values
deg_sample = train.iloc[sample_idx]['degradation'].values

X_pca = pca.fit_transform(X_sample)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# By label
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='coolwarm', alpha=0.3, s=5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('PCA - Colored by Label')
plt.colorbar(scatter, ax=axes[0])

# By degradation
deg_map = {d: i for i, d in enumerate(sorted(deg_types))}
deg_colors = [deg_map[d] for d in deg_sample]
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=deg_colors, cmap='tab10', alpha=0.3, s=5)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('PCA - Colored by Degradation Type')
cbar = plt.colorbar(scatter2, ax=axes[1], ticks=list(deg_map.values()))
cbar.ax.set_yticklabels(sorted(deg_types))

plt.suptitle('PCA Visualization of Feature Space', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'pca_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pca_visualization.png")

print("\n=== EDA Complete ===")
