#!/usr/bin/env python3
"""
Neuron Segment Merge Prediction - Full Analysis Pipeline
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, average_precision_score)
from sklearn.inspection import permutation_importance

# Try xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

# Try lightgbm
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except:
    HAS_LGB = False

OUT_DIR = "outputs"
IMG_DIR = "report/images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("="*60)
print("1. LOADING DATA")
print("="*60)

train_df = pd.read_csv("data/train_simulated.csv")
test_df = pd.read_csv("data/test_simulated.csv")

feature_cols = [str(i) for i in range(20)]
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Feature columns: {feature_cols}")
print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTrain degradation distribution:\n{train_df['degradation'].value_counts()}")
print(f"\nTest label distribution:\n{test_df['label'].value_counts()}")
print(f"\nTest degradation distribution:\n{test_df['degradation'].value_counts()}")

# Save data summary
data_summary = {
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "n_features": 20,
    "train_label_counts": train_df['label'].value_counts().to_dict(),
    "test_label_counts": test_df['label'].value_counts().to_dict(),
    "train_degradation_counts": train_df['degradation'].value_counts().to_dict(),
    "test_degradation_counts": test_df['degradation'].value_counts().to_dict(),
    "train_class_balance": float(train_df['label'].mean()),
    "test_class_balance": float(test_df['label'].mean()),
}
with open(f"{OUT_DIR}/data_summary.json", "w") as f:
    json.dump(data_summary, f, indent=2)

# ============================================================
# 2. EDA FIGURES
# ============================================================
print("\n" + "="*60)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*60)

# Figure 1: Class distribution by degradation type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class balance
ax = axes[0]
labels = ['Different Neuron (0)', 'Same Neuron (1)']
counts = [int(data_summary["train_label_counts"][0.0]), int(data_summary["train_label_counts"][1.0])]
colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{count}\n({count/sum(counts)*100:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Training Set Class Distribution', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(counts)*1.15)

# Degradation breakdown
ax = axes[1]
deg_counts = train_df.groupby(['degradation', 'label']).size().unstack(fill_value=0)
deg_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Samples by Degradation Type & Label', fontsize=13, fontweight='bold')
ax.legend(['Different (0)', 'Same (1)'], fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig1_data_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_data_overview.png")

# Figure 2: Feature distributions
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    ax = axes[i]
    for label, color in [(0, '#e74c3c'), (1, '#2ecc71')]:
        subset = train_df[train_df['label'] == label][col]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=f'Class {int(label)}', density=True)
    ax.set_title(f'Feature {i}', fontsize=10)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)
plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig2_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_feature_distributions.png")

# Figure 3: Correlation matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr = train_df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix (Training Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig3_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_correlation_matrix.png")

# ============================================================
# 3. MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("3. MODEL TRAINING")
print("="*60)

X_train = train_df[feature_cols].values
y_train = train_df['label'].values.astype(int)
X_test = test_df[feature_cols].values
y_test = test_df['label'].values.astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                            random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                     random_state=42),
}

if HAS_XGB:
    models['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False, eval_metric='logloss')
if HAS_LGB:
    models['LightGBM'] = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                          random_state=42, verbose=-1)

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    use_scaled = name == 'Logistic Regression'
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc_roc': auc, 'avg_precision': ap
    }
    predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Save results
with open(f"{OUT_DIR}/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Cross-validation for best model
best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\nBest model by F1: {best_model_name}")

# Figure 4: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of metrics
ax = axes[0]
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
x = np.arange(len(metrics))
width = 0.15
model_names = list(results.keys())
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=name, edgecolor='black', linewidth=0.3)
ax.set_xticks(x + width*(len(model_names)-1)/2)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'], fontsize=10)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)

# ROC curves
ax = axes[1]
for name in model_names:
    y_prob = predictions[name]['y_prob']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = results[name]['auc_roc']
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax.plot([0,1], [0,1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig4_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_model_comparison.png")

# Figure 5: Precision-Recall curves
fig, ax = plt.subplots(figsize=(8, 6))
for name in model_names:
    y_prob = predictions[name]['y_prob']
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ap = results[name]['avg_precision']
    ax.plot(rec_arr, prec_arr, label=f'{name} (AP={ap:.3f})', linewidth=2)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig5_pr_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_pr_curves.png")

# ============================================================
# 4. CONFUSION MATRICES FOR BEST MODEL
# ============================================================
print("\n" + "="*60)
print("4. DETAILED ANALYSIS OF BEST MODEL")
print("="*60)

best_pred = predictions[best_model_name]
cm = confusion_matrix(y_test, best_pred['y_pred'])
print(f"\nConfusion Matrix ({best_model_name}):")
print(cm)
print(f"\nClassification Report:\n{classification_report(y_test, best_pred['y_pred'])}")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Diff (0)', 'Same (1)'],
            yticklabels=['Diff (0)', 'Same (1)'], ax=ax, cbar_kws={'shrink': 0.8})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig6_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_confusion_matrix.png")

# ============================================================
# 5. PERFORMANCE BY DEGRADATION TYPE
# ============================================================
print("\n" + "="*60)
print("5. PERFORMANCE BY DEGRADATION TYPE")
print("="*60)

use_scaled = best_model_name == 'Logistic Regression'
deg_results = {}
for deg in test_df['degradation'].unique():
    mask = test_df['degradation'] == deg
    y_true_deg = y_test[mask]
    y_pred_deg = best_pred['y_pred'][mask]
    y_prob_deg = best_pred['y_prob'][mask]
    deg_results[deg] = {
        'n_samples': int(mask.sum()),
        'accuracy': accuracy_score(y_true_deg, y_pred_deg),
        'precision': precision_score(y_true_deg, y_pred_deg, zero_division=0),
        'recall': recall_score(y_true_deg, y_pred_deg, zero_division=0),
        'f1': f1_score(y_true_deg, y_pred_deg, zero_division=0),
        'auc_roc': roc_auc_score(y_true_deg, y_prob_deg) if len(np.unique(y_true_deg)) > 1 else 0,
        'class_balance': float(y_true_deg.mean()),
    }
    print(f"  {deg}: n={deg_results[deg]['n_samples']}, F1={deg_results[deg]['f1']:.4f}, AUC={deg_results[deg]['auc_roc']:.4f}")

with open(f"{OUT_DIR}/degradation_results.json", "w") as f:
    json.dump(deg_results, f, indent=2)

# Figure 7: Performance by degradation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
degs = sorted(deg_results.keys())
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
x = np.arange(len(degs))
width = 0.15
for i, m in enumerate(metrics_to_plot):
    vals = [deg_results[d][m] for d in degs]
    ax.bar(x + i*width, vals, width, label=m.replace('_', '-').title(), edgecolor='black', linewidth=0.3)
ax.set_xticks(x + width*(len(metrics_to_plot)-1)/2)
ax.set_xticklabels(degs, fontsize=10, rotation=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Performance by Degradation Type ({best_model_name})', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)

# Sample counts
ax = axes[1]
counts_diff = [test_df[(test_df['degradation']==d) & (test_df['label']==0)].shape[0] for d in degs]
counts_same = [test_df[(test_df['degradation']==d) & (test_df['label']==1)].shape[0] for d in degs]
x = np.arange(len(degs))
ax.bar(x - 0.15, counts_diff, 0.3, label='Different (0)', color='#e74c3c', edgecolor='black', linewidth=0.3)
ax.bar(x + 0.15, counts_same, 0.3, label='Same (1)', color='#2ecc71', edgecolor='black', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(degs, fontsize=10, rotation=15)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Test Set Composition by Degradation', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig7_degradation_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_degradation_analysis.png")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Use best tree-based model for feature importance
tree_models = {k: v for k, v in models.items() if k != 'Logistic Regression'}
if tree_models:
    best_tree = max(tree_models, key=lambda k: results[k]['f1'])
    model = models[best_tree]
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(f"\nFeature Importance ({best_tree}):")
    print(feat_imp.to_string(index=False))
    feat_imp.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_imp)))
    bars = ax.barh(range(len(feat_imp)), feat_imp['importance'].values, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels([f"Feature {f}" for f in feat_imp['feature'].values], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance ({best_tree})', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig8_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig8_feature_importance.png")

# Permutation importance for best model
print("\nComputing permutation importance...")
best_model = models[best_model_name]
use_scaled = best_model_name == 'Logistic Regression'
Xte = X_test_scaled if use_scaled else X_test
perm_imp = permutation_importance(best_model, Xte, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)
perm_df.to_csv(f"{OUT_DIR}/permutation_importance.csv", index=False)
print(perm_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(perm_df)), perm_df['importance_mean'].values,
        xerr=perm_df['importance_std'].values, color='#3498db', edgecolor='black',
        linewidth=0.3, capsize=3)
ax.set_yticks(range(len(perm_df)))
ax.set_yticklabels([f"Feature {f}" for f in perm_df['feature'].values], fontsize=10)
ax.set_xlabel('Mean Decrease in Accuracy', fontsize=12)
ax.set_title(f'Permutation Importance ({best_model_name})', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig9_permutation_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_permutation_importance.png")

# ============================================================
# 7. PROBABILITY DISTRIBUTION & CALIBRATION
# ============================================================
print("\n" + "="*60)
print("7. PREDICTION PROBABILITY ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Probability distribution
ax = axes[0]
for label, color, name in [(0, '#e74c3c', 'Different (0)'), (1, '#2ecc71', 'Same (1)')]:
    probs = best_pred['y_prob'][y_test == label]
    ax.hist(probs, bins=50, alpha=0.6, color=color, label=name, density=True)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold=0.5')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Prediction Probability Distribution ({best_model_name})', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Calibration-style plot
ax = axes[1]
prob_true = []
prob_pred = []
bins = np.linspace(0, 1, 11)
for i in range(len(bins)-1):
    mask = (best_pred['y_prob'] >= bins[i]) & (best_pred['y_prob'] < bins[i+1])
    if mask.sum() > 0:
        prob_true.append(y_test[mask].mean())
        prob_pred.append(best_pred['y_prob'][mask].mean())
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.plot(prob_pred, prob_true, 'o-', color='#3498db', linewidth=2, markersize=8, label=best_model_name)
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Plot', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig10_probability_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_probability_analysis.png")

# ============================================================
# 8. SAVE FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("8. FINAL SUMMARY")
print("="*60)

final_summary = {
    "best_model": best_model_name,
    "best_model_results": results[best_model_name],
    "all_model_results": results,
    "degradation_results": deg_results,
    "top_features": feat_imp.head(5).to_dict('records') if 'feat_imp' in dir() else [],
}
with open(f"{OUT_DIR}/final_summary.json", "w") as f:
    json.dump(final_summary, f, indent=2)

print(f"\nBest Model: {best_model_name}")
print(f"  Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision: {results[best_model_name]['precision']:.4f}")
print(f"  Recall:    {results[best_model_name]['recall']:.4f}")
print(f"  F1 Score:  {results[best_model_name]['f1']:.4f}")
print(f"  AUC-ROC:   {results[best_model_name]['auc_roc']:.4f}")
print("\nAll outputs saved to outputs/ and report/images/")
print("DONE!")
