"""
Part 1: Data visualization and model training
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
deg_train = train_df['degradation'].values

X_test = test_df[feature_cols].values
y_test = test_df['label'].values
deg_test = test_df['degradation'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']

# ==========================================
# Figure 1: Data Overview
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1a: Label distribution
ax = axes[0, 0]
train_counts = train_df['label'].value_counts()
ax.bar(['Different (0)', 'Same Neuron (1)'], train_counts.values, color=['#e74c3c', '#2ecc71'])
ax.set_title('Training Set: Label Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(train_counts.values):
    ax.text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# 1b: Test set distribution
ax = axes[0, 1]
test_counts = test_df['label'].value_counts()
ax.bar(['Different (0)', 'Same Neuron (1)'], test_counts.values, color=['#e74c3c', '#2ecc71'])
ax.set_title('Test Set: Label Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(test_counts.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# 1c: Feature mean comparison
ax = axes[0, 2]
feat_means_0 = X_train[y_train == 0].mean(axis=0)
feat_means_1 = X_train[y_train == 1].mean(axis=0)
x_pos = np.arange(20)
width = 0.35
ax.bar(x_pos - width/2, feat_means_0, width, label='Different (0)', alpha=0.7, color='#3498db')
ax.bar(x_pos + width/2, feat_means_1, width, label='Same (1)', alpha=0.7, color='#e67e22')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Mean Value')
ax.set_title('Mean Feature Values by Class', fontsize=12, fontweight='bold')
ax.legend()

# 1d: Degradation distribution
ax = axes[1, 0]
deg_counts = pd.crosstab(train_df['degradation'], train_df['label'])
deg_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_title('Training Set: Degradation by Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.legend(['Different (0)', 'Same (1)'])

# 1e: Correlation heatmap
ax = axes[1, 1]
corr = train_df[feature_cols[:10]].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Feature Correlation (first 10)', fontsize=12, fontweight='bold')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
plt.colorbar(im, ax=ax)

# 1f: Feature distribution variance by class
ax = axes[1, 2]
feat_var_0 = X_train[y_train == 0].std(axis=0)
feat_var_1 = X_train[y_train == 1].std(axis=0)
ax.bar(x_pos - width/2, feat_var_0, width, label='Different (0)', alpha=0.7, color='#3498db')
ax.bar(x_pos + width/2, feat_var_1, width, label='Same (1)', alpha=0.7, color='#e67e22')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Std Dev')
ax.set_title('Feature Variance by Class', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: data overview")

# ==========================================
# Figure 2: Mutual information
# ==========================================
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42, n_neighbors=3)
mi_sorted_idx = np.argsort(mi_scores)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
bars = ax.bar(range(20), mi_scores[mi_sorted_idx], color=colors[mi_sorted_idx])
ax.set_xlabel('Feature Index (sorted by importance)', fontsize=12)
ax.set_ylabel('Mutual Information', fontsize=12)
ax.set_title('Feature Importance via Mutual Information with Label', fontsize=13, fontweight='bold')
ax.set_xticks(range(20))
ax.set_xticklabels([str(i) for i in mi_sorted_idx])
for i, (v, idx) in enumerate(zip(mi_scores[mi_sorted_idx], mi_sorted_idx)):
    ax.text(i, v + 0.0003, f'{v:.4f}', ha='center', fontsize=8, rotation=60)

plt.tight_layout()
plt.savefig('report/images/figure2_mutual_information.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: mutual information")

# ==========================================
# Figure 3: Degradation-specific distributions
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, deg in enumerate(degradations):
    ax = axes[idx // 2, idx % 2]
    mask = deg_train == deg
    # PCA-like projection using top features
    for feat_idx in [0, 5, 10, 15]:
        vals_0 = X_train[(mask) & (y_train == 0), feat_idx]
        vals_1 = X_train[(mask) & (y_train == 1), feat_idx]
        ax.hist(vals_0, bins=30, alpha=0.3, label=f'F{feat_idx}(diff)', density=True)
        ax.hist(vals_1, bins=30, alpha=0.5, label=f'F{feat_idx}(same)', density=True, linestyle='--')
    ax.set_title(f'{deg}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)

plt.suptitle('Feature Distributions by Degradation Type', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure3_degradation_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: degradation features")

print("Part 1 complete. Starting models...")
