#!/usr/bin/env python3
"""SimBA Behavior Classification Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("SIMBA BEHAVIOR CLASSIFICATION")
print("="*60)

# Load data
features_df = pd.read_csv('data/Together_1_features_extracted.csv', index_col=0)
targets_df = pd.read_csv('data/Together_1_targets_inserted.csv', index_col=0)
reference_df = pd.read_csv('data/Together_1_machine_results_reference.csv', index_col=0)

print(f"Features: {features_df.shape}, Targets: {targets_df.shape}, Reference: {reference_df.shape}")

# Extract features and targets
feature_cols = [c for c in features_df.columns if c not in ['Feature_1', 'Feature_2']]
X = features_df[feature_cols].values
y_attack = targets_df['Attack'].values
y_sniffing = targets_df['Sniffing'].values

print(f"X shape: {X.shape}")
print(f"Attack: {y_attack.sum()} positive, {len(y_attack)-y_attack.sum()} negative")
print(f"Sniffing: {y_sniffing.sum()} positive, {len(y_sniffing)-y_sniffing.sum()} negative")

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
attack_counts = pd.Series(y_attack).value_counts()
axes[0].bar(['No Attack', 'Attack'], attack_counts.values, color=['skyblue', 'salmon'])
axes[0].set_title('Attack Behavior Distribution')
axes[0].set_ylabel('Frame Count')
for i, v in enumerate(attack_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=12)

sniffing_counts = pd.Series(y_sniffing).value_counts()
axes[1].bar(['No Sniffing', 'Sniffing'], sniffing_counts.values, color=['lightgreen', 'orange'])
axes[1].set_title('Sniffing Behavior Distribution')
axes[1].set_ylabel('Frame Count')
for i, v in enumerate(sniffing_counts.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('report/images/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: class_distribution.png")

# Temporal patterns
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
frames = np.arange(len(y_attack))
axes[0].fill_between(frames, y_attack, alpha=0.7, color='salmon')
axes[0].set_ylabel('Attack')
axes[0].set_title('Attack Behavior Over Time')
axes[0].set_ylim(-0.1, 1.1)
axes[1].fill_between(frames, y_sniffing, alpha=0.7, color='orange')
axes[1].set_ylabel('Sniffing')
axes[1].set_xlabel('Frame Number')
axes[1].set_title('Sniffing Behavior Over Time')
axes[1].set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig('report/images/temporal_behavior.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: temporal_behavior.png")

# Co-occurrence matrix
fig, ax = plt.subplots(figsize=(8, 6))
cooccurrence = pd.crosstab(pd.Series(y_attack, name='Attack'), pd.Series(y_sniffing, name='Sniffing'))
sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('Behavior Co-occurrence Matrix')
plt.tight_layout()
plt.savefig('report/images/cooccurrence_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cooccurrence_matrix.png")

# Train/test split (temporal to avoid leakage)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_attack_train, y_attack_test = y_attack[:split_idx], y_attack[split_idx:]
y_sniffing_train, y_sniffing_test = y_sniffing[:split_idx], y_sniffing[split_idx:]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

def train_and_evaluate(X_train, X_test, y_train, y_test, behavior_name):
    print(f"\n--- {behavior_name} ---")
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                  min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    avg_precision = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    
    return {'classifier': clf, 'predictions': y_pred, 'probabilities': y_prob,
            'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
                       'f1_score': f1, 'roc_auc': roc_auc, 'avg_precision': avg_precision}}

results['Attack'] = train_and_evaluate(X_train_scaled, X_test_scaled, y_attack_train, y_attack_test, 'Attack')
results['Sniffing'] = train_and_evaluate(X_train_scaled, X_test_scaled, y_sniffing_train, y_sniffing_test, 'Sniffing')

# Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, data['probabilities'])
    avg_precision = data['metrics']['avg_precision']
    axes[idx].plot(recall_vals, precision_vals, linewidth=2, label=f'AP={avg_precision:.3f}')
    axes[idx].set_xlabel('Recall')
    axes[idx].set_ylabel('Precision')
    axes[idx].set_title(f'Precision-Recall: {behavior}')
    axes[idx].legend()
    axes[idx].grid(True)
plt.tight_layout()
plt.savefig('report/images/precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_curves.png")

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    fpr, tpr, _ = roc_curve(y_test, data['probabilities'])
    roc_auc = data['metrics']['roc_auc']
    axes[idx].plot(fpr, tpr, linewidth=2, label=f'AUC={roc_auc:.3f}')
    axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[idx].set_xlabel('False Positive Rate')
    axes[idx].set_ylabel('True Positive Rate')
    axes[idx].set_title(f'ROC Curve: {behavior}')
    axes[idx].legend()
    axes[idx].grid(True)
plt.tight_layout()
plt.savefig('report/images/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    cm = confusion_matrix(y_test, data['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[idx].set_title(f'Confusion Matrix: {behavior}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('report/images/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# Feature importance
feature_names = feature_cols
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for idx, (behavior, data) in enumerate(results.items()):
    importances = data['classifier'].feature_importances_
    indices = np.argsort(importances)[-20:]
    axes[idx].barh(range(len(indices)), importances[indices], align='center')
    axes[idx].set_yticks(range(len(indices)))
    axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    axes[idx].set_xlabel('Feature Importance')
    axes[idx].set_title(f'Top 20 Features: {behavior}')
plt.tight_layout()
plt.savefig('report/images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# Save metrics
metrics_summary = {}
for behavior, data in results.items():
    metrics_summary[behavior] = data['metrics']

with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved: outputs/metrics.json")

# Save predictions
predictions_df = pd.DataFrame({
    'frame': np.arange(len(y_attack_test)),
    'attack_true': y_attack_test,
    'attack_pred': results['Attack']['predictions'],
    'attack_prob': results['Attack']['probabilities'],
    'sniffing_true': y_sniffing_test,
    'sniffing_pred': results['Sniffing']['predictions'],
    'sniffing_prob': results['Sniffing']['probabilities']
})
predictions_df.to_csv('outputs/predictions.csv', index=False)
print("Saved: outputs/predictions.csv")

# Feature importance tables
for behavior, data in results.items():
    importances = data['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feat_imp_df.to_csv(f'outputs/feature_importance_{behavior.lower()}.csv', index=False)
    print(f"Saved: feature_importance_{behavior.lower()}.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
