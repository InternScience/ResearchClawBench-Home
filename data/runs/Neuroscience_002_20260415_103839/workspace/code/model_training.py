"""
Main Modeling Pipeline for Neuron Segment Merging
Trains multiple classifiers, evaluates them, and saves results.
Optimized for speed with reduced hyperparameters.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, classification_report, f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('data/train_simulated.csv')
test = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]

X_train = train[feature_cols].values
y_train = train['label'].values
X_test = test[feature_cols].values
y_test = test['label'].values

degradation_test = test['degradation'].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# ============================================================
# Scale features
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Model Definitions (reduced for speed)
# ============================================================
def get_models():
    spw = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=500, class_weight='balanced', random_state=42, solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss', random_state=42, n_jobs=-1,
            use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            class_weight='balanced', random_state=42, n_jobs=-1,
            verbose=-1
        ),
    }
    return models

# ============================================================
# Cross-Validation on Training Set (3-fold for speed)
# ============================================================
print("\n" + "="*60)
print("Cross-Validation on Training Set (3-fold stratified)")
print("="*60)

models = get_models()
cv_results = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)

        y_pred = model_clone.predict(X_fold_val)
        y_prob = model_clone.predict_proba(X_fold_val)[:, 1]

        fpr_cv, tpr_cv, _ = roc_curve(y_fold_val, y_prob)
        metrics = {
            'accuracy': accuracy_score(y_fold_val, y_pred),
            'precision': precision_score(y_fold_val, y_pred),
            'recall': recall_score(y_fold_val, y_pred),
            'f1': f1_score(y_fold_val, y_pred),
            'roc_auc': auc(fpr_cv, tpr_cv),
            'pr_auc': average_precision_score(y_fold_val, y_prob),
        }
        fold_metrics.append(metrics)

    elapsed = time.time() - start_time

    avg_metrics = {}
    for key in fold_metrics[0].keys():
        vals = [m[key] for m in fold_metrics]
        avg_metrics[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    cv_results[name] = avg_metrics
    print(f"  {name}: F1={avg_metrics['f1']['mean']:.4f}+/-{avg_metrics['f1']['std']:.4f}, "
          f"ROC-AUC={avg_metrics['roc_auc']['mean']:.4f}+/-{avg_metrics['roc_auc']['std']:.4f}, "
          f"Time={elapsed:.1f}s")

with open('outputs/cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)
print("\nSaved outputs/cv_results.json")

# ============================================================
# Train on Full Training Set & Evaluate on Test
# ============================================================
print("\n" + "="*60)
print("Training on Full Training Set & Evaluating on Test Set")
print("="*60)

test_results = {}
best_models = {}

for name, model in models.items():
    print(f"\nTraining {name} on full training set...")
    start_time = time.time()

    model_full = type(model)(**model.get_params())
    model_full.fit(X_train_scaled, y_train)

    y_pred = model_full.predict(X_test_scaled)
    y_prob = model_full.predict_proba(X_test_scaled)[:, 1]

    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(auc(fpr_t, tpr_t)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
    }

    elapsed = time.time() - start_time
    metrics['training_time'] = float(elapsed)

    test_results[name] = metrics
    best_models[name] = model_full

    print(f"  {name}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
          f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

# ============================================================
# Ensemble: Soft voting with top 3 models
# ============================================================
print("\nBuilding ensemble (soft voting with top 3 by F1)...")
sorted_by_f1 = sorted(test_results.items(), key=lambda x: x[1]['f1'], reverse=True)
top3_names = [x[0] for x in sorted_by_f1[:3]]
print(f"Top 3 models: {top3_names}")

ensemble = VotingClassifier(
    estimators=[(name, best_models[name]) for name in top3_names],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)

y_pred_ens = ensemble.predict(X_test_scaled)
y_prob_ens = ensemble.predict_proba(X_test_scaled)[:, 1]

fpr_e, tpr_e, _ = roc_curve(y_test, y_prob_ens)
ensemble_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred_ens)),
    'precision': float(precision_score(y_test, y_pred_ens)),
    'recall': float(recall_score(y_test, y_pred_ens)),
    'f1': float(f1_score(y_test, y_pred_ens)),
    'roc_auc': float(auc(fpr_e, tpr_e)),
    'pr_auc': float(average_precision_score(y_test, y_prob_ens)),
}
test_results['Ensemble (Top 3)'] = ensemble_metrics
print(f"  Ensemble: Acc={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1']:.4f}, "
      f"ROC-AUC={ensemble_metrics['roc_auc']:.4f}, PR-AUC={ensemble_metrics['pr_auc']:.4f}")

with open('outputs/test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print("\nSaved outputs/test_results.json")

# ============================================================
# Find Best Model
# ============================================================
best_name = max(test_results, key=lambda k: test_results[k]['f1'])
if best_name == 'Ensemble (Top 3)':
    best_model = ensemble
    y_prob_best = y_prob_ens
    y_pred_best = y_pred_ens
else:
    best_model = best_models[best_name]
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_best = best_model.predict(X_test_scaled)
print(f"\nBest model by F1: {best_name} (F1={test_results[best_name]['f1']:.4f})")

# ============================================================
# Generate Figures
# ============================================================
print("\n" + "="*60)
print("Generating Result Figures")
print("="*60)

all_model_names = list(models.keys()) + ['Ensemble (Top 3)']

# Figure 6: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name in all_model_names:
    if name == 'Ensemble (Top 3)':
        y_prob_cur = y_prob_ens
    else:
        y_prob_cur = best_models[name].predict_proba(X_test_scaled)[:, 1]

    fpr_c, tpr_c, _ = roc_curve(y_test, y_prob_cur)
    roc_auc_c = auc(fpr_c, tpr_c)
    ax.plot(fpr_c, tpr_c, linewidth=2, label=f'{name} (AUC={roc_auc_c:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig6_roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig6_roc_curves.png")

# Figure 7: Precision-Recall Curves
fig, ax = plt.subplots(figsize=(8, 6))
for name in all_model_names:
    if name == 'Ensemble (Top 3)':
        y_prob_cur = y_prob_ens
    else:
        y_prob_cur = best_models[name].predict_proba(X_test_scaled)[:, 1]

    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob_cur)
    pr_auc_c = average_precision_score(y_test, y_prob_cur)
    ax.plot(rec_c, prec_c, linewidth=2, label=f'{name} (AP={pr_auc_c:.4f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - All Models (Test Set)', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig7_pr_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig7_pr_curves.png")

# Figure 8: Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
metric_names_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

for idx, (metric, label) in enumerate(zip(metric_names_list, metric_labels)):
    ax = axes[idx]
    names = list(test_results.keys())
    values = [test_results[n][metric] for n in names]
    colors_b = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.barh(names, values, color=colors_b, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Score', fontsize=10)

plt.suptitle('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig8_performance_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig8_performance_comparison.png")

# Figure 9: Confusion Matrix for Best Model
cm = confusion_matrix(y_test, y_pred_best)

fig, axes_cm = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0],
            xticklabels=['Different (0)', 'Same (1)'],
            yticklabels=['Different (0)', 'Same (1)'])
axes_cm[0].set_title(f'Confusion Matrix - {best_name}\n(F1={test_results[best_name]["f1"]:.4f})',
                  fontsize=12, fontweight='bold')
axes_cm[0].set_ylabel('True Label', fontsize=11)
axes_cm[0].set_xlabel('Predicted Label', fontsize=11)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Greens', ax=axes_cm[1],
            xticklabels=['Different (0)', 'Same (1)'],
            yticklabels=['Different (0)', 'Same (1)'])
axes_cm[1].set_title(f'Normalized Confusion Matrix - {best_name}',
                  fontsize=12, fontweight='bold')
axes_cm[1].set_ylabel('True Label', fontsize=11)
axes_cm[1].set_xlabel('Predicted Label', fontsize=11)

plt.suptitle('Confusion Matrix Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig9_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig9_confusion_matrix.png")

# Figure 10: Feature Importance (from Random Forest)
rf_model = best_models['Random Forest']
importances = rf_model.feature_importances_
indices_imp = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 7))
feat_names_ordered = [feature_cols[i] for i in indices_imp]
imp_ordered = importances[indices_imp]
colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_names_ordered)))
bars_fi = ax.barh(range(len(feat_names_ordered)), imp_ordered, color=colors_imp, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(feat_names_ordered)))
ax.set_yticklabels(feat_names_ordered)
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for bar_fi, val_fi in zip(bars_fi, imp_ordered):
    ax.text(val_fi + 0.002, bar_fi.get_y() + bar_fi.get_height()/2.,
            f'{val_fi:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/fig10_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig10_feature_importance.png")

# Figure 11: Performance by Degradation Type
print("\nComputing per-degradation performance...")
deg_types = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
per_deg_results = {}

for name in all_model_names:
    if name == 'Ensemble (Top 3)':
        y_prob_cur = y_prob_ens
        y_pred_cur = y_pred_ens
    else:
        y_prob_cur = best_models[name].predict_proba(X_test_scaled)[:, 1]
        y_pred_cur = best_models[name].predict(X_test_scaled)

    per_deg_results[name] = {}
    for deg in deg_types:
        mask = degradation_test == deg
        if mask.sum() > 0:
            fpr_d, tpr_d, _ = roc_curve(y_test[mask], y_prob_cur[mask])
            per_deg_results[name][deg] = {
                'f1': float(f1_score(y_test[mask], y_pred_cur[mask])),
                'roc_auc': float(auc(fpr_d, tpr_d)),
                'precision': float(precision_score(y_test[mask], y_pred_cur[mask], zero_division=0)),
                'recall': float(recall_score(y_test[mask], y_pred_cur[mask], zero_division=0)),
                'n_samples': int(mask.sum()),
            }

with open('outputs/per_degradation_results.json', 'w') as f:
    json.dump(per_deg_results, f, indent=2)

# Plot: F1 score by degradation type for all models
fig, ax = plt.subplots(figsize=(12, 6))
x_deg = np.arange(len(deg_types))
width_deg = 0.12
n_models = len(all_model_names)
colors_bar = plt.cm.tab10(np.linspace(0, 1, n_models))

for i, name in enumerate(all_model_names):
    vals = [per_deg_results[name][deg]['f1'] for deg in deg_types]
    offset = (i - n_models/2 + 0.5) * width_deg
    ax.bar(x_deg + offset, vals, width_deg, label=name, color=colors_bar[i], edgecolor='black', linewidth=0.3)

ax.set_xlabel('Degradation Type', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('F1-Score by Degradation Type (Test Set)', fontsize=13, fontweight='bold')
ax.set_xticks(x_deg)
ax.set_xticklabels(deg_types, rotation=15)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/fig11_per_degradation_f1.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig11_per_degradation_f1.png")

# Figure 12: Threshold Analysis for Best Model
thresholds = np.arange(0.1, 0.91, 0.05)
threshold_metrics = []
for thresh in thresholds:
    y_pred_t = (y_prob_best >= thresh).astype(int)
    threshold_metrics.append({
        'threshold': float(thresh),
        'accuracy': float(accuracy_score(y_test, y_pred_t)),
        'precision': float(precision_score(y_test, y_pred_t, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_t, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_t, zero_division=0)),
    })

fig, ax = plt.subplots(figsize=(8, 5))
for metric in ['precision', 'recall', 'f1', 'accuracy']:
    vals = [m[metric] for m in threshold_metrics]
    ax.plot(thresholds, vals, marker='o', markersize=4, linewidth=2, label=metric.capitalize())

ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default Threshold')
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Threshold Analysis - {best_name}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.95)
plt.tight_layout()
plt.savefig('report/images/fig12_threshold_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig12_threshold_analysis.png")

# Save predictions
predictions_df = pd.DataFrame({
    'sample_id': range(len(y_test)),
    'true_label': y_test,
    'predicted_label': y_pred_best,
    'probability': y_prob_best,
    'degradation': degradation_test,
})
predictions_df.to_csv('outputs/predictions_best_model.csv', index=False)
print(f"\nSaved outputs/predictions_best_model.csv")

# Final summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
for name, metrics in sorted(test_results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"{name:30s} | Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
          f"Rec={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
          f"PR-AUC={metrics['pr_auc']:.4f}")

print(f"\nBest model: {best_name}")
print(f"Best F1: {test_results[best_name]['f1']:.4f}")
print("\nAll figures saved to report/images/")
print("All results saved to outputs/")
