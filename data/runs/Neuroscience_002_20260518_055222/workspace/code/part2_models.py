"""
Part 2: Model training and evaluation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve, 
                              confusion_matrix, average_precision_score,
                              f1_score, precision_score, recall_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
import json

# Load and preprocess
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
# Model training
# ==========================================
print("Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, 
                                           class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                                    learning_rate=0.1, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                        random_state=42, early_stopping=True, 
                                        validation_fraction=0.15),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': roc_auc_score(y_test, y_prob),
        'ap': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    print(f"  AUC-ROC: {results[name]['auc']:.4f}, F1: {results[name]['f1']:.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} (AUC={results[best_model_name]['auc']:.4f})")

# ==========================================
# Figure 4: ROC and PR Curves
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(rec, prec, label=f"{name} (AP={res['ap']:.4f})", linewidth=2)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure4_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: ROC and PR curves")

# ==========================================
# Figure 5: Model Comparison Bar Chart
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_names = list(results.keys())
metrics_df = pd.DataFrame({
    'AUC-ROC': [results[n]['auc'] for n in model_names],
    'AP': [results[n]['ap'] for n in model_names],
    'F1': [results[n]['f1'] for n in model_names],
    'Precision': [results[n]['precision'] for n in model_names],
    'Recall': [results[n]['recall'] for n in model_names],
}, index=model_names)

ax = axes[0]
metrics_df[['AUC-ROC', 'AP', 'F1']].plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: AUC, AP, and F1', fontsize=13, fontweight='bold')
ax.set_ylim([0.4, 1.0])
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='lower right')

ax = axes[1]
metrics_df[['Precision', 'Recall']].plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: Precision and Recall', fontsize=13, fontweight='bold')
ax.set_ylim([0.4, 1.0])
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('report/images/figure5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: model comparison")

# ==========================================
# Figure 6: Confusion Matrices
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
top_models = sorted(results.keys(), key=lambda x: results[x]['auc'], reverse=True)[:3]

for idx, name in enumerate(top_models):
    ax = axes[idx]
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
    ax.set_title(f'{name}\n(AUC={results[name]["auc"]:.4f})', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices - Top 3 Models', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('report/images/figure6_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: confusion matrices")

# ==========================================
# Degradation-specific performance
# ==========================================
print("\nDegradation-specific performance:")
deg_results = {}

for deg in degradations:
    mask = deg_test == deg
    X_deg = X_test_scaled[mask]
    y_deg = y_test[mask]
    
    y_prob_deg = best_model.predict_proba(X_deg)[:, 1]
    y_pred_deg = best_model.predict(X_deg)
    
    deg_results[deg] = {
        'auc': float(roc_auc_score(y_deg, y_prob_deg)),
        'f1': float(f1_score(y_deg, y_pred_deg)),
        'ap': float(average_precision_score(y_deg, y_prob_deg)),
        'precision': float(precision_score(y_deg, y_pred_deg)),
        'recall': float(recall_score(y_deg, y_pred_deg)),
        'n_samples': int(len(y_deg)),
        'n_positives': int(y_deg.sum())
    }
    print(f"  {deg}: AUC={deg_results[deg]['auc']:.4f}, F1={deg_results[deg]['f1']:.4f}")

# ==========================================
# Figure 7: Degradation-specific performance
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
deg_names = list(deg_results.keys())
x_pos = np.arange(len(deg_names))
width = 0.25
auc_vals = [deg_results[d]['auc'] for d in deg_names]
f1_vals = [deg_results[d]['f1'] for d in deg_names]
ap_vals = [deg_results[d]['ap'] for d in deg_names]

ax.bar(x_pos - width, auc_vals, width, label='AUC-ROC', color='#3498db', alpha=0.8)
ax.bar(x_pos, f1_vals, width, label='F1 Score', color='#2ecc71', alpha=0.8)
ax.bar(x_pos + width, ap_vals, width, label='AP', color='#e67e22', alpha=0.8)
ax.set_xlabel('Degradation Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Performance by Degradation ({best_model_name})', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(deg_names, rotation=15)
ax.legend()
ax.set_ylim([0.4, 1.0])

ax = axes[1]
colors_deg = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for idx, deg in enumerate(degradations):
    mask = deg_test == deg
    y_prob_deg = best_model.predict_proba(X_test_scaled[mask])[:, 1]
    y_deg = y_test[mask]
    fpr, tpr, _ = roc_curve(y_deg, y_prob_deg)
    ax.plot(fpr, tpr, label=f'{deg} (AUC={deg_results[deg]["auc"]:.4f})', 
            linewidth=2, color=colors_deg[idx])
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC by Degradation ({best_model_name})', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure7_degradation_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: degradation performance")

# ==========================================
# Figure 8: Feature importance
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_model = results['Random Forest']['model']
rf_importance = rf_model.feature_importances_
rf_sorted_idx = np.argsort(rf_importance)[::-1]

ax = axes[0]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, 20))
ax.bar(range(20), rf_importance[rf_sorted_idx], color=colors[rf_sorted_idx])
ax.set_xlabel('Feature Index (sorted)', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Random Forest: Feature Importance', fontsize=13, fontweight='bold')
ax.set_xticks(range(20))
ax.set_xticklabels([str(i) for i in rf_sorted_idx])

gb_model = results['Gradient Boosting']['model']
gb_importance = gb_model.feature_importances_
gb_sorted_idx = np.argsort(gb_importance)[::-1]

ax = axes[1]
colors = plt.cm.magma(np.linspace(0.2, 0.9, 20))
ax.bar(range(20), gb_importance[gb_sorted_idx], color=colors[gb_sorted_idx])
ax.set_xlabel('Feature Index (sorted)', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Gradient Boosting: Feature Importance', fontsize=13, fontweight='bold')
ax.set_xticks(range(20))
ax.set_xticklabels([str(i) for i in gb_sorted_idx])

plt.tight_layout()
plt.savefig('report/images/figure8_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved: feature importance")

# ==========================================
# Cross-validation
# ==========================================
print("\nRunning cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

# Use simpler models for CV
cv_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}

for name, model in cv_models.items():
    fold_aucs = []
    fold_f1s = []
    
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        m = type(model)(**model.get_params())
        m.fit(X_train_scaled[train_idx], y_train[train_idx])
        y_prob = m.predict_proba(X_train_scaled[val_idx])[:, 1]
        y_pred = m.predict(X_train_scaled[val_idx])
        fold_aucs.append(roc_auc_score(y_train[val_idx], y_prob))
        fold_f1s.append(f1_score(y_train[val_idx], y_pred))
    
    cv_results[name] = {
        'auc_mean': float(np.mean(fold_aucs)),
        'auc_std': float(np.std(fold_aucs)),
        'f1_mean': float(np.mean(fold_f1s)),
        'f1_std': float(np.std(fold_f1s)),
        'auc_folds': [float(x) for x in fold_aucs]
    }
    print(f"  {name}: AUC={np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}")

# ==========================================
# Figure 9: Cross-validation
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
cv_data = [cv_results[n]['auc_folds'] for n in cv_results]
cv_labels = [n.replace(' ', '\n') for n in cv_results]
bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)
colors_bp = ['#3498db', '#2ecc71', '#e67e22']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('5-Fold Cross-Validation AUC Distribution', fontsize=13, fontweight='bold')

ax = axes[1]
model_names_cv = list(cv_results.keys())
test_aucs = [results[n]['auc'] for n in model_names_cv]
cv_aucs_m = [cv_results[n]['auc_mean'] for n in model_names_cv]
cv_stds = [cv_results[n]['auc_std'] for n in model_names_cv]
x_pos = np.arange(len(model_names_cv))
width = 0.35
ax.bar(x_pos - width/2, test_aucs, width, label='Test Set AUC', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, cv_aucs_m, width, yerr=cv_stds, label='CV AUC (mean±std)', 
       color='#e67e22', alpha=0.8, capsize=5)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Test vs Cross-Validation Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names_cv])
ax.legend()
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig('report/images/figure9_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved: cross-validation")

# Save all results
results_summary = {
    'model_comparison': {name: {
        'auc': float(results[name]['auc']),
        'ap': float(results[name]['ap']),
        'f1': float(results[name]['f1']),
        'precision': float(results[name]['precision']),
        'recall': float(results[name]['recall']),
        'accuracy': float(results[name]['accuracy'])
    } for name in results},
    'best_model': best_model_name,
    'degradation_results': deg_results,
    'cross_validation': {k: {kk: vv for kk, vv in v.items() if kk != 'auc_folds'} 
                         for k, v in cv_results.items()}
}

with open('outputs/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ All analysis complete!")
