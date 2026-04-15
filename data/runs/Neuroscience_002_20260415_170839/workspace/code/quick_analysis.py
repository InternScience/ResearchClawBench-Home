"""Quick analysis for connectomics data."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

# Paths
DATA_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/data"
OUTPUT_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/outputs"
REPORT_IMG_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/report/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_simulated.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_simulated.csv"))
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Feature columns
feature_cols = [str(i) for i in range(20)]

# Data distributions
print("Creating distribution plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_counts = train_df['label'].value_counts()
test_counts = test_df['label'].value_counts()
axes[0].bar(['Different (0)', 'Same (1)'], train_counts.values, color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Training Set Label Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(train_counts.values):
    axes[0].text(i, v + 2000, f'{v}\n({v/len(train_df)*100:.1f}%)', ha='center')
axes[1].bar(['Different (0)', 'Same (1)'], test_counts.values, color=['#e74c3c', '#2ecc71'])
axes[1].set_title('Test Set Label Distribution')
axes[1].set_ylabel('Count')
for i, v in enumerate(test_counts.values):
    axes[1].text(i, v + 1000, f'{v}\n({v/len(test_df)*100:.1f}%)', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Degradation distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_deg = train_df['degradation'].value_counts()
test_deg = test_df['degradation'].value_counts()
colors_deg = {'Misalignment': '#3498db', 'Missing Sections': '#9b59b6', 'Mixed': '#e67e22', 'Average': '#1abc9c'}
axes[0].bar(train_deg.index, train_deg.values, color=[colors_deg.get(d, '#95a5a6') for d in train_deg.index])
axes[0].set_title('Training Set: Degradation Type Distribution')
axes[0].tick_params(axis='x', rotation=45)
axes[1].bar(test_deg.index, test_deg.values, color=[colors_deg.get(d, '#95a5a6') for d in test_deg.index])
axes[1].set_title('Test Set: Degradation Type Distribution')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'degradation_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Correlation heatmap
print("Creating correlation heatmap...")
corr_matrix = train_df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# Feature distributions (sample)
print("Creating feature distribution plots...")
sample_df = train_df.sample(n=5000, random_state=42)
fig, axes = plt.subplots(4, 5, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    ax = axes[i]
    sample_df[sample_df['label'] == 0][col].hist(ax=ax, alpha=0.6, bins=30, label='Different (0)', color='#e74c3c')
    sample_df[sample_df['label'] == 1][col].hist(ax=ax, alpha=0.6, bins=30, label='Same (1)', color='#2ecc71')
    ax.set_title(f'Feature {col}')
    ax.legend(fontsize=7)
plt.suptitle('Feature Distributions by Label (Sample)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'feature_distributions_train.png'), dpi=150, bbox_inches='tight')
plt.close()

# Prepare data
print("Preparing data...")
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("Training models...")
results = {}
trained_models = {}

# Logistic Regression
print("  Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
trained_models['Logistic Regression'] = lr
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob)
}

# Random Forest
print("  Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
trained_models['Random Forest'] = rf
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob)
}

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
print("\nModel Comparison:")
print(results_df)

# Model comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, results_df.loc['Logistic Regression', metrics], width, label='Logistic Regression', color='#3498db')
ax.bar(x + width/2, results_df.loc['Random Forest', metrics], width, label='Random Forest', color='#2ecc71')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics])
ax.legend()
ax.set_ylim(0, 1)
for i, m in enumerate(metrics):
    ax.text(i - width/2, results_df.loc['Logistic Regression', m] + 0.02, f"{results_df.loc['Logistic Regression', m]:.3f}", ha='center', fontsize=8)
    ax.text(i + width/2, results_df.loc['Random Forest', m] + 0.02, f"{results_df.loc['Random Forest', m]:.3f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    axes[1].plot(recall, precision, label=f'{name} (AP={pr_auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

baseline = np.sum(y_test) / len(y_test)
axes[1].axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
               xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
print("Analyzing feature importance...")
importances = rf.feature_importances_
importance_df = pd.DataFrame({'feature': [f'Feature_{i}' for i in range(20)], 'importance': importances}).sort_values('importance', ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
print(importance_df.head(10))

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color=colors[::-1])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance Ranking (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Degradation analysis
print("Analyzing by degradation type...")
best_model = rf
degradation_results = {}
for deg_type in test_df['degradation'].unique():
    subset = test_df[test_df['degradation'] == deg_type]
    X_sub = scaler.transform(subset[feature_cols].values)
    y_sub = subset['label'].values
    y_pred = best_model.predict(X_sub)
    y_prob = best_model.predict_proba(X_sub)[:, 1]
    degradation_results[deg_type] = {
        'n_samples': len(subset),
        'accuracy': accuracy_score(y_sub, y_pred),
        'f1': f1_score(y_sub, y_pred),
        'roc_auc': roc_auc_score(y_sub, y_prob)
    }

deg_df = pd.DataFrame(degradation_results).T
print(deg_df)
deg_df.to_csv(os.path.join(OUTPUT_DIR, 'degradation_analysis.csv'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['accuracy', 'f1', 'roc_auc']
colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(deg_df.index, deg_df[metric], color=colors[:len(deg_df)])
    ax.set_ylim(0, 1)
    ax.set_title(f'{metric.upper()} by Degradation Type')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('Model Performance Across Degradation Types (Random Forest)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'degradation_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save summary
summary = {
    'dataset_info': {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': 20,
        'train_positive_ratio': float(train_counts.get(1.0, 0) / len(train_df)),
        'test_positive_ratio': float(test_counts.get(1.0, 0) / len(test_df))
    },
    'best_model': 'Random Forest',
    'model_comparison': results,
    'degradation_results': degradation_results
}

with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nAnalysis complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Figures saved to: {REPORT_IMG_DIR}")
