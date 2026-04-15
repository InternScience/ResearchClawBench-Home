import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
out_dir = Path('report/images')
out_dir.mkdir(exist_ok=True)
outputs_dir = Path('outputs')
outputs_dir.mkdir(exist_ok=True)

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

# Features
feature_cols = [str(i) for i in range(20)]
X_train, y_train = train_df[feature_cols], train_df['label']
degr_train = train_df['degradation']
X_test, y_test = test_df[feature_cols], test_df['label']
degr_test = test_df['degradation']

# Save stats
stats = {
    'train_shape': train_df.shape,
    'test_shape': test_df.shape,
    'label_balance_train': y_train.value_counts(normalize=True).to_dict(),
    'label_balance_test': y_test.value_counts(normalize=True).to_dict(),
    'degr_dist_train': degr_train.value_counts(normalize=True).to_dict(),
    'degr_dist_test': degr_test.value_counts(normalize=True).to_dict()
}
pd.DataFrame(X_train.describe()).to_json(outputs_dir / 'feature_stats_train.json')
pd.DataFrame(X_test.describe()).to_json(outputs_dir / 'feature_stats_test.json')
with open(outputs_dir / 'eda_stats.json', 'w') as f:
    import json
    json.dump(stats, f, indent=2)

# Plots
fig, axes = plt.subplots(2, 2, figsize=(15,12))

# Label balance per degradation
pd.crosstab(degr_train, y_train, normalize='index').plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Train Label Balance per Degradation')
axes[0,0].legend(title='Label')

pd.crosstab(degr_test, y_test, normalize='index').plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Test Label Balance per Degradation')

# Feature distributions (boxplot sample)
sample_cols = feature_cols[:4]
X_train[sample_cols].boxplot(ax=axes[1,0])
axes[1,0].set_title('Train Feature Distributions (first 4)')

X_test[sample_cols].boxplot(ax=axes[1,1])
axes[1,1].set_title('Test Feature Distributions (first 4)')

plt.tight_layout()
plt.savefig(out_dir / 'data_overview.png', dpi=300, bbox_inches='tight')

# Correlation heatmap (train)
plt.figure(figsize=(12,10))
sns.heatmap(X_train.corr(), center=0, cmap='RdBu_r', ax=plt.gca())
plt.title('Feature Correlation Heatmap (Train)')
plt.tight_layout()
plt.savefig(out_dir / 'corr_heatmap_train.png', dpi=300, bbox_inches='tight')

print('EDA complete. Plots saved.')