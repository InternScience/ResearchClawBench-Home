"""
EDA Analysis for Neuron Segment Merging Task
Generates exploratory data analysis figures and statistics.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
train = pd.read_csv('data/train_simulated.csv')
test = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]

# ============================================================
# Figure 1: Label Distribution (Train & Test)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, df, title in zip(axes, [train, test], ['Training Set', 'Test Set']):
    counts = df['label'].value_counts().sort_index()
    colors = ['#e74c3c', '#3498db']
    bars = ax.bar(['Negative (0)', 'Positive (1)'], counts.values, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)

plt.suptitle('Label Distribution: Binary Classification Task\n(Same Neuron vs Different Neuron)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig1_label_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_label_distribution.png")

# ============================================================
# Figure 2: Feature Distributions by Label (subset of features)
# ============================================================
# Select representative features from different modality groups
# Features 0-4: morphology, 5-9: intensity, 10-19: embedding
selected_features = ['0', '3', '5', '8', '10', '15', '18', '19']
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for idx, feat in enumerate(selected_features):
    ax = axes[idx]
    for label_val, color, label_name in [(0, '#e74c3c', 'Different'), (1, '#3498db', 'Same')]:
        subset = train[train['label'] == label_val][feat]
        ax.hist(subset.values, bins=50, alpha=0.6, color=color, label=label_name, density=True)
    ax.set_title(f'Feature {feat}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)

plt.suptitle('Feature Distributions by Class Label (Training Set)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig2_feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_feature_distributions.png")

# ============================================================
# Figure 3: Correlation Heatmap
# ============================================================
corr_matrix = train[feature_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (Training Set)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('report/images/fig3_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_correlation_heatmap.png")

# ============================================================
# Figure 4: Degradation Type Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Degradation distribution
deg_counts = train['degradation'].value_counts()
colors_deg = ['#2ecc71', '#e67e22', '#9b59b6', '#3498db']
bars = axes[0].bar(deg_counts.index, deg_counts.values, color=colors_deg, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, deg_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                 f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_title('Samples per Degradation Type', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
axes[0].tick_params(axis='x', rotation=15)

# Positive rate by degradation type
pos_rates = train.groupby('degradation')['label'].mean()
bars2 = axes[1].bar(pos_rates.index, pos_rates.values * 100, color=colors_deg, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars2, pos_rates.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_title('Positive Rate by Degradation Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Positive Rate (%)', fontsize=11)
axes[1].tick_params(axis='x', rotation=15)
axes[1].set_ylim(0, max(pos_rates.values) * 120)

plt.suptitle('Degradation Type Analysis', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_degradation_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_degradation_analysis.png")

# ============================================================
# Figure 5: Feature Statistics Summary
# ============================================================
stats = train[feature_cols].describe().T
stats['skew'] = train[feature_cols].skew()
stats['kurtosis'] = train[feature_cols].kurtosis()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mean by feature group
groups = {'Morphology (0-4)': list(range(5)), 'Intensity (5-9)': list(range(5, 10)), 
          'Embedding (10-19)': list(range(10, 20))}
group_means = []
group_stds = []
group_names = []
for name, indices in groups.items():
    vals = train[[str(i) for i in indices]].values.flatten()
    group_means.append(np.mean(vals))
    group_stds.append(np.std(vals))
    group_names.append(name)

axes[0].bar(group_names, group_means, yerr=group_stds, capsize=5, color=['#e74c3c', '#f39c12', '#3498db'], edgecolor='black')
axes[0].set_title('Mean Feature Value by Modality Group', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Mean Value', fontsize=10)
axes[0].tick_params(axis='x', rotation=10)

# Skewness distribution
axes[1].hist(stats['skew'].values, bins=20, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[1].set_title('Feature Skewness Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Skewness', fontsize=10)
axes[1].set_ylabel('Count', fontsize=10)

# Kurtosis distribution
axes[2].hist(stats['kurtosis'].values, bins=20, color='#9b59b6', edgecolor='black', alpha=0.8)
axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[2].set_title('Feature Kurtosis Distribution', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Kurtosis', fontsize=10)
axes[2].set_ylabel('Count', fontsize=10)

plt.suptitle('Feature Statistical Properties', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig5_feature_statistics.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig5_feature_statistics.png")

# Save EDA stats
eda_stats = {
    'feature_means': stats['mean'].to_dict(),
    'feature_stds': stats['std'].to_dict(),
    'feature_skew': stats['skew'].to_dict(),
    'feature_kurtosis': stats['kurtosis'].to_dict(),
    'class_balance_train': float(train['label'].mean()),
    'class_balance_test': float(test['label'].mean()),
}
with open('outputs/eda_statistics.json', 'w') as f:
    json.dump(eda_stats, f, indent=2)
print("Saved outputs/eda_statistics.json")
print("\nEDA complete!")
