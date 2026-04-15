#!/usr/bin/env python3
"""SimBA Behavior Classification Analysis - Improved with stratified splits"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

print("="*60)
print("SIMBA BEHAVIOR CLASSIFICATION ANALYSIS")
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

print(f"Feature matrix X shape: {X.shape}")
print(f"Attack: {y_attack.sum()} positive ({100*y_attack.mean():.1f}%), {len(y_attack)-y_attack.sum()} negative")
print(f"Sniffing: {y_sniffing.sum()} positive ({100*y_sniffing.mean():.1f}%), {len(y_sniffing)-y_sniffing.sum()} negative")

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
attack_counts = pd.Series(y_attack).value_counts()
axes[0].bar(['No Attack', 'Attack'], attack_counts.values, color=['skyblue', 'salmon'], edgecolor='black')
axes[0].set_title('Attack Behavior Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Frame Count')
for i, v in enumerate(attack_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontsize=12, fontweight='bold')

sniffing_counts = pd.Series(y_sniffing).value_counts()
axes[1].bar(['No Sniffing', 'Sniffing'], sniffing_counts.values, color=['lightgreen', 'orange'], edgecolor='black')
axes[1].set_title('Sniffing Behavior Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frame Count')
for i, v in enumerate(sniffing_counts.values):
    axes[1].text(i, v + 30, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/class_distribution.png")

# Temporal patterns
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
frames = np.arange(len(y_attack))
axes[0].fill_between(frames, y_attack, alpha=0.7, color='salmon', label='Attack')
axes[0].set_ylabel('Attack', fontsize=12)
axes[0].set_title('Attack Behavior Over Time (Frames)', fontsize=14, fontweight='bold')
axes[0].set_ylim(-0.1, 1.1)
axes[0].legend(loc='upper right')

axes[1].fill_between(frames, y_sniffing, alpha=0.7, color='orange', label='Sniffing')
axes[1].set_ylabel('Sniffing', fontsize=12)
axes[1].set_xlabel('Frame Number', fontsize=12)
axes[1].set_title('Sniffing Behavior Over Time (Frames)', fontsize=14, fontweight='bold')
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend(loc='upper right')
plt.tight_layout()
plt.savefig('report/images/temporal_behavior.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/temporal_behavior.png")

# Co-occurrence matrix
fig, ax = plt.subplots(figsize=(8, 6))
cooccurrence = pd.crosstab(pd.Series(y_attack, name='Attack'), pd.Series(y_sniffing, name='Sniffing'))
sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Behavior Co-occurrence Matrix (Frame Counts)', fontsize=14, fontweight='bold')
ax.set_ylabel('Attack', fontsize=12)
ax.set_xlabel('Sniffing', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/cooccurrence_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/cooccurrence_matrix.png")

# Train/test split using stratified sampling for reproducible evaluation
X_train_a, X_test_a, y_attack_train, y_attack_test = train_test_split(
    X, y_attack, test_size=0.3, random_state=42, stratify=y_attack)

X_train_s, X_test_s, y_sniffing_train, y_sniffing_test = train_test_split(
    X, y_sniffing, test_size=0.3, random_state=42, stratify=y_sniffing)

print(f"\nAttack - Train: {len(y_attack_train)} ({y_attack_train.sum()} pos), Test: {len(y_attack_test)} ({y_attack_test.sum()} pos)")
print(f"Sniffing - Train: {len(y_sniffing_train)} ({y_sniffing_train.sum()} pos), Test: {len(y_sniffing_test)} ({y_sniffing_test.sum()} pos)")

# Scale features
scaler_a = StandardScaler()
X_train_a_scaled = scaler_a.fit_transform(X_train_a)
X_test_a_scaled = scaler_a.transform(X_test_a)

scaler_s = StandardScaler()
X_train_s_scaled = scaler_s.fit_transform(X_train_s)
X_test_s_scaled = scaler_s.transform(X_test_s)

results = {}

def train_and_evaluate(X_train, X_test, y_train, y_test, behavior_name):
    print(f"\n{'='*40}")
    print(f"Training {behavior_name} Classifier")
    print('='*40)
    
    clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Avg Prec:  {avg_precision:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'classifier': clf, 
        'predictions': y_pred, 
        'probabilities': y_prob,
        'metrics': {
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1_score': f1, 
            'roc_auc': roc_auc, 
            'avg_precision': avg_precision
        },
        'confusion_matrix': cm.tolist()
    }

results['Attack'] = train_and_evaluate(X_train_a_scaled, X_test_a_scaled, y_attack_train, y_attack_test, 'Attack')
results['Sniffing'] = train_and_evaluate(X_train_s_scaled, X_test_s_scaled, y_sniffing_train, y_sniffing_test, 'Sniffing')

# Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, data['probabilities'])
    avg_precision = data['metrics']['avg_precision']
    axes[idx].plot(recall_vals, precision_vals, linewidth=2.5, color='darkblue', label=f'AP = {avg_precision:.3f}')
    axes[idx].set_xlabel('Recall', fontsize=12)
    axes[idx].set_ylabel('Precision', fontsize=12)
    axes[idx].set_title(f'Precision-Recall Curve: {behavior}', fontsize=14, fontweight='bold')
    axes[idx].legend(loc='lower left', fontsize=11)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('report/images/precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: report/images/precision_recall_curves.png")

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    fpr, tpr, _ = roc_curve(y_test, data['probabilities'])
    roc_auc = data['metrics']['roc_auc']
    axes[idx].plot(fpr, tpr, linewidth=2.5, color='darkred', label=f'AUC = {roc_auc:.3f}')
    axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    axes[idx].set_xlabel('False Positive Rate', fontsize=12)
    axes[idx].set_ylabel('True Positive Rate', fontsize=12)
    axes[idx].set_title(f'ROC Curve: {behavior}', fontsize=14, fontweight='bold')
    axes[idx].legend(loc='lower right', fontsize=11)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('report/images/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/roc_curves.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (behavior, data) in enumerate(results.items()):
    y_test = y_attack_test if behavior == 'Attack' else y_sniffing_test
    cm = confusion_matrix(y_test, data['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 14, "weight": "bold"})
    axes[idx].set_title(f'Confusion Matrix: {behavior}', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=12)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/confusion_matrices.png")

# Feature importance
feature_names = feature_cols
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
for idx, (behavior, data) in enumerate(results.items()):
    importances = data['classifier'].feature_importances_
    indices = np.argsort(importances)[-20:]
    axes[idx].barh(range(len(indices)), importances[indices], align='center', color='steelblue', edgecolor='black')
    axes[idx].set_yticks(range(len(indices)))
    axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    axes[idx].set_xlabel('Feature Importance', fontsize=12)
    axes[idx].set_title(f'Top 20 Feature Importances: {behavior}', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('report/images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/feature_importance.png")

# Save metrics
with open('outputs/metrics.json', 'w') as f:
    json.dump({k: v['metrics'] for k, v in results.items()}, f, indent=2)
print("Saved: outputs/metrics.json")

# Save predictions
predictions_df = pd.DataFrame({
    'attack_true': y_attack_test,
    'attack_pred': results['Attack']['predictions'],
    'attack_prob': results['Attack']['probabilities'],
})
predictions_df.to_csv('outputs/predictions_attack.csv', index=False)

predictions_df_s = pd.DataFrame({
    'sniffing_true': y_sniffing_test,
    'sniffing_pred': results['Sniffing']['predictions'],
    'sniffing_prob': results['Sniffing']['probabilities']
})
predictions_df_s.to_csv('outputs/predictions_sniffing.csv', index=False)
print("Saved: outputs/predictions_*.csv")

# Feature importance tables
for behavior, data in results.items():
    importances = data['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feat_imp_df.to_csv(f'outputs/feature_importance_{behavior.lower()}.csv', index=False)
    print(f"Saved: outputs/feature_importance_{behavior.lower()}.csv")

# Comparison bar chart of metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_names))
width = 0.35

attack_metrics = [results['Attack']['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
sniffing_metrics = [results['Sniffing']['metrics'][k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]

bars1 = ax.bar(x - width/2, attack_metrics, width, label='Attack', color='salmon', edgecolor='black')
bars2 = ax.bar(x + width/2, sniffing_metrics, width, label='Sniffing', color='orange', edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/metrics_comparison.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
