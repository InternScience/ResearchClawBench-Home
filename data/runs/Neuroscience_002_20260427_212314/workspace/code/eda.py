"""EDA for connectomics merge-prediction dataset."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

tr = pd.read_csv('data/train_simulated.csv')
te = pd.read_csv('data/test_simulated.csv')

# Modality groupings (inferred from feature statistics)
# - features 0-4  -> morphology (mean ~0.28, std ~0.33)
# - features 5-9  -> intensity   (mean ~0.38, std ~0.39)
# - features 10-19-> embedding   (mean ~0.57, std ~0.58)
modalities = {
    'morphology': [str(i) for i in range(0, 5)],
    'intensity':  [str(i) for i in range(5, 10)],
    'embedding':  [str(i) for i in range(10, 20)],
}
with open('outputs/modality_groups.json', 'w') as f:
    json.dump(modalities, f, indent=2)

feat_cols = [str(i) for i in range(20)]

# 1) Overview figure: label balance + degradation distribution + per-degradation positive rate
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
counts = tr['label'].value_counts().sort_index()
ax.bar(['0 (no merge)', '1 (merge)'], counts.values, color=['#4C72B0', '#DD8452'])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({100*v/len(tr):.1f}%)", ha='center', va='bottom')
ax.set_title('Train label distribution')
ax.set_ylabel('count')

ax = axes[1]
deg_counts = tr['degradation'].value_counts()
ax.bar(deg_counts.index, deg_counts.values, color=sns.color_palette('Set2', 4))
ax.set_title('Train degradation distribution')
ax.set_ylabel('count')
ax.tick_params(axis='x', rotation=20)

ax = axes[2]
pr = tr.groupby('degradation')['label'].mean().sort_values()
ax.bar(pr.index, pr.values, color=sns.color_palette('Set2', 4))
ax.set_title('Positive (merge) rate per degradation, train')
ax.set_ylabel('P(label=1)')
ax.tick_params(axis='x', rotation=20)
for i, v in enumerate(pr.values):
    ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150)
plt.close()

# 2) Feature distributions per class for each modality (boxplots of group means)
for mod, cols in modalities.items():
    tr[f'{mod}_mean'] = tr[cols].mean(axis=1)
    te[f'{mod}_mean'] = te[cols].mean(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, mod in zip(axes, modalities):
    sns.violinplot(data=tr, x='label', y=f'{mod}_mean', ax=ax,
                   palette=['#4C72B0', '#DD8452'], cut=0, inner='quartile')
    ax.set_title(f'{mod} mean by class (train)')
    ax.set_xlabel('label')
plt.tight_layout()
plt.savefig('report/images/feature_dist_per_class.png', dpi=150)
plt.close()

# 3) Correlation heatmap of the 20 features
corr = tr[feat_cols].corr()
plt.figure(figsize=(8.5, 7))
sns.heatmap(corr, cmap='RdBu_r', center=0, square=True, cbar_kws={'shrink': .7},
            vmin=-1, vmax=1)
plt.title('Feature correlation matrix (train, 20 features)')
plt.tight_layout()
plt.savefig('report/images/feature_correlation.png', dpi=150)
plt.close()

# 4) Mean-difference per feature between classes
mean_pos = tr.loc[tr.label == 1, feat_cols].mean()
mean_neg = tr.loc[tr.label == 0, feat_cols].mean()
diff = (mean_pos - mean_neg)
plt.figure(figsize=(11, 4))
colors = ['#DD8452' if d > 0 else '#4C72B0' for d in diff.values]
plt.bar(feat_cols, diff.values, color=colors)
plt.axhline(0, color='k', lw=0.6)
plt.title('Mean(positive) - Mean(negative) per feature, train')
plt.ylabel('Δ mean')
plt.xlabel('feature index')
for i, v in enumerate(diff.values):
    plt.text(i, v, f"{v:+.2f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/feature_mean_difference.png', dpi=150)
plt.close()

# 5) PCA scatter coloured by label
X = tr[feat_cols].values
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(X)
plt.figure(figsize=(7, 6))
sub = np.random.RandomState(0).choice(len(Z), size=10000, replace=False)
plt.scatter(Z[sub, 0], Z[sub, 1], c=tr['label'].values[sub], cmap='coolwarm',
            s=6, alpha=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA of 20 features (subsample = 10k), train')
cbar = plt.colorbar()
cbar.set_label('label')
plt.tight_layout()
plt.savefig('report/images/pca_train.png', dpi=150)
plt.close()

print('PCA explained variance ratio:', pca.explained_variance_ratio_.round(4).tolist())
print('mean diff (pos-neg) per feature:')
print(diff.round(4).to_string())

# Save EDA summary
summary = {
    'n_train': int(len(tr)),
    'n_test': int(len(te)),
    'pos_rate_train': float(tr.label.mean()),
    'pos_rate_test': float(te.label.mean()),
    'pos_rate_per_degradation_train': tr.groupby('degradation').label.mean().round(4).to_dict(),
    'pos_rate_per_degradation_test': te.groupby('degradation').label.mean().round(4).to_dict(),
    'pca_explained': pca.explained_variance_ratio_.round(4).tolist(),
    'feature_mean_diff': diff.round(4).to_dict(),
    'modalities': modalities,
}
with open('outputs/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('EDA done')
