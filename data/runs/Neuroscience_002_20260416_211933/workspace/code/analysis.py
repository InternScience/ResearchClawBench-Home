#!/usr/bin/env python3
"""
Neuron Segment Merge Prediction Analysis
=========================================
Binary classification for connectomics proofreading automation.

This script processes over-segmented EM neuron fragment data to predict
whether two adjacent segments belong to the same neuron and should be merged.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
WORKSPACE = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260416_211933'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')

# Ensure output directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

print("=" * 60)
print("NEURON SEGMENT MERGE PREDICTION ANALYSIS")
print("=" * 60)

# =============================================================================
# 1. DATA LOADING AND EXPLORATION
# =============================================================================
print("\n[1] Loading data...")

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

# Rename feature columns for easier handling
feature_cols = [str(i) for i in range(20)]
train_df.columns = feature_cols + ['label', 'degradation']
test_df.columns = feature_cols + ['label', 'degradation']

X_train = train_df[feature_cols].values
y_train = train_df['label'].values.astype(int)
X_test = test_df[feature_cols].values
y_test = test_df['label'].values.astype(int)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Class distribution (train): 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
print(f"Class distribution (test): 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")

# =============================================================================
# 2. DATA VISUALIZATION - OVERVIEW PLOTS
# =============================================================================
print("\n[2] Generating data overview plots...")

# Figure 1: Label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_train = ['Different Neuron (0)', 'Same Neuron (1)']
labels_test = ['Different Neuron (0)', 'Same Neuron (1)']

train_counts = [np.sum(y_train==0), np.sum(y_train==1)]
test_counts = [np.sum(y_test==0), np.sum(y_test==1)]

axes[0].bar(labels_train, train_counts, color=['#2E86AB', '#E94F37'])
axes[0].set_title('Training Set Label Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(train_counts):
    axes[0].text(i, v + 1000, str(v), ha='center', fontsize=10)

axes[1].bar(labels_test, test_counts, color=['#2E86AB', '#E94F37'])
axes[1].set_title('Test Set Label Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(test_counts):
    axes[1].text(i, v + 500, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig01_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig01_label_distribution.png")

# Figure 2: Degradation type distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

degradation_order = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
train_deg_counts = train_df['degradation'].value_counts().reindex(degradation_order)
test_deg_counts = test_df['degradation'].value_counts().reindex(degradation_order)

colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

axes[0].bar(degradation_order, train_deg_counts.values, color=colors)
axes[0].set_title('Training Set - Degradation Type Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(train_deg_counts.values):
    axes[0].text(i, v + 500, str(v), ha='center', fontsize=9)

axes[1].bar(degradation_order, test_deg_counts.values, color=colors)
axes[1].set_title('Test Set - Degradation Type Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(test_deg_counts.values):
    axes[1].text(i, v + 200, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig02_degradation_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig02_degradation_distribution.png")

# Figure 3: Feature correlation heatmap (sample of features)
fig, ax = plt.subplots(figsize=(14, 12))

# Compute correlation matrix for all features
corr_matrix = train_df[feature_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=False, ax=ax)
ax.set_title('Feature Correlation Matrix (Training Set)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig03_feature_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig03_feature_correlation.png")

# Figure 4: Feature distributions by label (selected features)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
selected_features = ['0', '5', '10', '15']
feature_names = ['Feature 0', 'Feature 5', 'Feature 10', 'Feature 15']

for idx, (feat, name) in enumerate(zip(selected_features, feature_names)):
    ax = axes[idx // 2, idx % 2]
    
    # Separate by label
    feat_0 = train_df[train_df['label'] == 0][feat]
    feat_1 = train_df[train_df['label'] == 1][feat]
    
    ax.hist(feat_0, bins=50, alpha=0.6, label='Different Neuron (0)', color='#2E86AB', density=True)
    ax.hist(feat_1, bins=50, alpha=0.6, label='Same Neuron (1)', color='#E94F37', density=True)
    ax.set_xlabel(name)
    ax.set_ylabel('Density')
    ax.set_title(f'{name} Distribution by Label', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig04_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig04_feature_distributions.png")

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================
print("\n[3] Training classification models...")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with class weights
class_weight_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"  Class imbalance ratio: {class_weight_ratio:.2f}:1")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Metrics
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    Accuracy: {results[name]['accuracy']:.4f}")
    print(f"    Precision: {results[name]['precision']:.4f}")
    print(f"    Recall: {results[name]['recall']:.4f}")
    print(f"    F1 Score: {results[name]['f1']:.4f}")
    print(f"    ROC-AUC: {results[name]['roc_auc']:.4f}")

# Select best model based on F1 score (important for imbalanced data)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
print(f"\n  Best model by F1 score: {best_model_name}")

# Save model results
with open(os.path.join(OUTPUTS_DIR, 'model_results.json'), 'w') as f:
    # Convert to serializable format
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {k: float(v) for k, v in res.items() 
                                       if k not in ['y_pred', 'y_pred_proba']}
    json.dump(serializable_results, f, indent=2)
print("  - Saved: model_results.json")

# =============================================================================
# 4. RESULTS VISUALIZATION
# =============================================================================
print("\n[4] Generating result visualization plots...")

# Figure 5: Model comparison bar chart
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
colors = ['#3498DB', '#E74C3C', '#2ECC71']

for idx, (metric, mname) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx // 3, idx % 3]
    values = [results[name][metric] for name in models.keys()]
    bars = ax.bar(models.keys(), values, color=colors)
    ax.set_title(f'{mname} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(mname)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)

# Empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig05_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig05_model_comparison.png")

# Figure 6: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, color in zip(models.keys(), colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    ax.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.4f})', 
            color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig06_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig06_roc_curves.png")

# Figure 7: Confusion Matrix (Best Model)
fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred: Different (0)', 'Pred: Same (1)'],
            yticklabels=['True: Different (0)', 'True: Same (1)'],
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig07_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig07_confusion_matrix.png")

# Figure 8: Precision-Recall Curve
fig, ax = plt.subplots(figsize=(10, 8))

for name, color in zip(models.keys(), colors):
    prec, rec, thresholds = precision_recall_curve(y_test, results[name]['y_pred_proba'])
    ax.plot(rec, prec, label=f'{name}', color=color, linewidth=2)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig08_precision_recall.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig08_precision_recall.png")

# =============================================================================
# 5. PERFORMANCE BY DEGRADATION TYPE
# =============================================================================
print("\n[5] Analyzing performance by degradation type...")

# Get predictions from best model
best_y_pred = results[best_model_name]['y_pred']
test_df_copy = test_df.copy()
test_df_copy['prediction'] = best_y_pred

degradation_metrics = {}
for deg_type in degradation_order:
    mask = test_df_copy['degradation'] == deg_type
    if mask.sum() > 0:
        y_true_deg = test_df_copy.loc[mask, 'label'].values.astype(int)
        y_pred_deg = test_df_copy.loc[mask, 'prediction'].values
        
        degradation_metrics[deg_type] = {
            'accuracy': accuracy_score(y_true_deg, y_pred_deg),
            'precision': precision_score(y_true_deg, y_pred_deg, zero_division=0),
            'recall': recall_score(y_true_deg, y_pred_deg, zero_division=0),
            'f1': f1_score(y_true_deg, y_pred_deg, zero_division=0),
            'n_samples': mask.sum()
        }

# Figure 9: Performance by degradation type
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[idx // 2, idx % 2]
    values = [degradation_metrics[deg][metric] for deg in degradation_order]
    bars = ax.bar(degradation_order, values, color=colors)
    ax.set_title(f'{title} by Degradation Type ({best_model_name})', fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig09_performance_by_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig09_performance_by_degradation.png")

# Save degradation metrics
with open(os.path.join(OUTPUTS_DIR, 'degradation_metrics.json'), 'w') as f:
    json.dump(degradation_metrics, f, indent=2)
print("  - Saved: degradation_metrics.json")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
print("\n[6] Computing feature importance...")

# Use Random Forest for feature importance (most interpretable)
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Figure 10: Feature importance
fig, ax = plt.subplots(figsize=(12, 10))

top_n = 15
top_features = feature_importance.head(top_n)

bars = ax.barh(range(top_n), top_features['importance'].values, color='#3498DB')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f'Feature {int(f)}' for f in top_features['feature'].values])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, top_features['importance'].values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig10_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: fig10_feature_importance.png")

# Save feature importance
feature_importance.to_csv(os.path.join(OUTPUTS_DIR, 'feature_importance.csv'), index=False)
print("  - Saved: feature_importance.csv")

# =============================================================================
# 7. SUMMARY AND OUTPUT
# =============================================================================
print("\n[7] Generating summary...")

summary = {
    'best_model': best_model_name,
    'best_metrics': {k: float(v) for k, v in results[best_model_name].items() 
                     if k not in ['y_pred', 'y_pred_proba']},
    'all_models': {name: {k: float(v) for k, v in res.items() 
                          if k not in ['y_pred', 'y_pred_proba']} 
                   for name, res in results.items()},
    'degradation_performance': degradation_metrics,
    'top_features': feature_importance.head(10).to_dict('records'),
    'data_summary': {
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'n_features': X_train.shape[1],
        'class_imbalance_ratio': float(class_weight_ratio),
        'degradation_types': degradation_order
    }
}

with open(os.path.join(OUTPUTS_DIR, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  - Saved: analysis_summary.json")

# Print final summary
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"  Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision: {results[best_model_name]['precision']:.4f}")
print(f"  Recall:    {results[best_model_name]['recall']:.4f}")
print(f"  F1 Score:  {results[best_model_name]['f1']:.4f}")
print(f"  ROC-AUC:   {results[best_model_name]['roc_auc']:.4f}")
print(f"\nAll figures saved to: {REPORT_IMAGES_DIR}")
print(f"All outputs saved to: {OUTPUTS_DIR}")
print("=" * 60)
