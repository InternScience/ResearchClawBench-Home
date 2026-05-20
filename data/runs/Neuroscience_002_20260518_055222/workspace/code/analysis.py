"""
Neuron Segment Connectivity Prediction
Analysis of EM brain data for predicting whether two segments belong to the same neuron.
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
                              classification_report, confusion_matrix, average_precision_score,
                              f1_score, precision_score, recall_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

# Separate features and labels
feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
deg_train = train_df['degradation'].values

X_test = test_df[feature_cols].values
y_test = test_df['label'].values
deg_test = test_df['degradation'].values

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Train label distribution: {np.sum(y_train==0)} negatives, {np.sum(y_train==1)} positives")
print(f"Test label distribution: {np.sum(y_test==0)} negatives, {np.sum(y_test==1)} positives")

# Encode degradation
le = LabelEncoder()
deg_train_enc = le.fit_transform(deg_train)
deg_test_enc = le.transform(deg_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Figure 1: Data Overview
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1a: Label distribution
ax = axes[0, 0]
train_counts = train_df['label'].value_counts()
ax.bar(['Same Neuron (1)', 'Different (0)'], train_counts.values, color=['#2ecc71', '#e74c3c'])
ax.set_title('Training Set: Label Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(train_counts.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# 1b: Test set distribution
ax = axes[0, 1]
test_counts = test_df['label'].value_counts()
ax.bar(['Same Neuron (1)', 'Different (0)'], test_counts.values, color=['#2ecc71', '#e74c3c'])
ax.set_title('Test Set: Label Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(test_counts.values):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')

# 1c: Feature distributions by class (first 10 features)
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
ax.set_xticks(range(0, 20, 2))

# 1d: Degradation type distribution
ax = axes[1, 0]
deg_train_df = pd.DataFrame({'degradation': deg_train, 'label': y_train})
deg_counts = deg_train_df.groupby(['degradation', 'label']).size().unstack()
deg_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_title('Training Set: Degradation by Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.legend(['Different (0)', 'Same (1)'])

# 1e: Feature correlation heatmap (selected features)
ax = axes[1, 1]
corr = train_df[feature_cols[:10]].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Feature Correlation (first 10)', fontsize=12, fontweight='bold')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
plt.colorbar(im, ax=ax)

# 1f: Boxplot of key features
ax = axes[1, 2]
data_plot = []
labels_plot = []
for feat_idx in [0, 5, 10, 15]:
    for cls in [0, 1]:
        vals = X_train[y_train == cls, feat_idx]
        data_plot.append(vals[:500])  # Subsample for plotting
        labels_plot.append(f'F{feat_idx}\nC{cls}')
ax.boxplot(data_plot, labels=labels_plot)
ax.set_title('Selected Features Distribution', fontsize=12, fontweight='bold')
ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('report/images/figure1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: data overview")

# ==========================================
# Figure 2: Feature importance by mutual information
# ==========================================
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
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
    ax.text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig('report/images/figure2_mutual_information.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: mutual information")

# ==========================================
# Figure 3: Degradation-specific feature distributions
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']

for idx, deg in enumerate(degradations):
    ax = axes[idx // 2, idx % 2]
    mask = deg_train == deg
    for feat_idx in [0, 5, 10, 15]:
        vals_0 = X_train[(mask) & (y_train == 0), feat_idx]
        vals_1 = X_train[(mask) & (y_train == 1), feat_idx]
        ax.hist(vals_0, bins=30, alpha=0.4, label=f'F{feat_idx} (diff)', density=True)
        ax.hist(vals_1, bins=30, alpha=0.6, label=f'F{feat_idx} (same)', density=True, linestyle='--')
    ax.set_title(f'{deg}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

plt.suptitle('Feature Distributions by Degradation Type and Label', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/figure3_degradation_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: degradation features")

# ==========================================
# Model Training and Evaluation
# ==========================================
print("\nTraining models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, 
                                           class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, 
                                                    learning_rate=0.1, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                        random_state=42, early_stopping=True, 
                                        validation_fraction=0.15),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': auc,
        'ap': ap,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc
    }
    
    print(f"  AUC-ROC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

# ==========================================
# Figure 4: Model Comparison - ROC and PR Curves
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curves
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# Precision-Recall Curves
ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(rec, prec, label=f"{name} (AP={res['ap']:.4f})", linewidth=2)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('report/images/figure4_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: ROC and PR curves")

# ==========================================
# Figure 5: Model Comparison Bar Chart
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Performance metrics
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
ax.set_ylim([0.5, 1.0])
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='lower right')

ax = axes[1]
metrics_df[['Precision', 'Recall']].plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: Precision and Recall', fontsize=13, fontweight='bold')
ax.set_ylim([0.5, 1.0])
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('report/images/figure5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: model comparison")

# ==========================================
# Figure 6: Confusion Matrices for top models
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
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")

for deg in degradations:
    mask = deg_test == deg
    X_deg = X_test_scaled[mask]
    y_deg = y_test[mask]
    
    y_prob_deg = best_model.predict_proba(X_deg)[:, 1]
    y_pred_deg = best_model.predict(X_deg)
    
    auc_deg = roc_auc_score(y_deg, y_prob_deg)
    f1_deg = f1_score(y_deg, y_pred_deg)
    ap_deg = average_precision_score(y_deg, y_prob_deg)
    
    deg_results[deg] = {
        'auc': auc_deg,
        'f1': f1_deg,
        'ap': ap_deg,
        'n_samples': len(y_deg),
        'n_positives': int(y_deg.sum())
    }
    print(f"  {deg}: AUC={auc_deg:.4f}, F1={f1_deg:.4f}, AP={ap_deg:.4f}, n={len(y_deg)}")

# Save degradation results
with open('outputs/degradation_results.json', 'w') as f:
    json.dump(deg_results, f, indent=2)

# ==========================================
# Figure 7: Degradation-specific performance
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance by degradation
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
ax.set_title(f'Performance by Degradation Type ({best_model_name})', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(deg_names, rotation=15)
ax.legend()
ax.set_ylim([0.5, 1.0])
for i, (a, f, ap) in enumerate(zip(auc_vals, f1_vals, ap_vals)):
    ax.text(i - width, a + 0.01, f'{a:.3f}', ha='center', fontsize=8)
    ax.text(i, f + 0.01, f'{f:.3f}', ha='center', fontsize=8)
    ax.text(i + width, ap + 0.01, f'{ap:.3f}', ha='center', fontsize=8)

# ROC curves by degradation
ax = axes[1]
colors_deg = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for idx, deg in enumerate(degradations):
    mask = deg_test == deg
    y_prob_deg = best_model.predict_proba(X_test_scaled[mask])[:, 1]
    y_deg = y_test[mask]
    fpr, tpr, _ = roc_curve(y_deg, y_prob_deg)
    auc_val = deg_results[deg]['auc']
    ax.plot(fpr, tpr, label=f'{deg} (AUC={auc_val:.4f})', 
            linewidth=2, color=colors_deg[idx])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves by Degradation Type ({best_model_name})', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('report/images/figure7_degradation_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: degradation performance")

# ==========================================
# Figure 8: Feature importance from best tree-based model
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest feature importance
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

# Gradient Boosting feature importance
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
# Figure 9: Cross-validation analysis
# ==========================================
print("\nRunning cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    fold_aucs = []
    fold_aps = []
    fold_f1s = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_train[val_idx]
        
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        
        y_prob_val = model_clone.predict_proba(X_fold_val)[:, 1]
        y_pred_val = model_clone.predict(X_fold_val)
        
        fold_aucs.append(roc_auc_score(y_fold_val, y_prob_val))
        fold_aps.append(average_precision_score(y_fold_val, y_prob_val))
        fold_f1s.append(f1_score(y_fold_val, y_pred_val))
    
    cv_results[name] = {
        'auc_mean': np.mean(fold_aucs),
        'auc_std': np.std(fold_aucs),
        'ap_mean': np.mean(fold_aps),
        'ap_std': np.std(fold_aps),
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s),
        'auc_folds': fold_aucs
    }
    print(f"  {name}: AUC={np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}")

# Save CV results
with open('outputs/cross_validation_results.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'auc_folds'} 
               for k, v in cv_results.items()}, f, indent=2)

# ==========================================
# Figure 10: Cross-validation results
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CV boxplots
ax = axes[0]
cv_data = []
cv_labels = []
for name in cv_results:
    cv_data.append(cv_results[name]['auc_folds'])
    cv_labels.append(name.replace(' ', '\n'))
bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)
colors_bp = plt.cm.Set2(np.linspace(0, 1, len(cv_results)))
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('5-Fold Cross-Validation AUC Distribution', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=0)

# Comparison of test vs CV performance
ax = axes[1]
model_names_list = list(results.keys())
test_aucs = [results[n]['auc'] for n in model_names_list]
cv_aucs = [cv_results[n]['auc_mean'] for n in model_names_list]
cv_stds = [cv_results[n]['auc_std'] for n in model_names_list]

x_pos = np.arange(len(model_names_list))
width = 0.35
ax.bar(x_pos - width/2, test_aucs, width, label='Test Set AUC', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, cv_aucs, width, yerr=cv_stds, label='CV AUC (mean±std)', 
       color='#e67e22', alpha=0.8, capsize=5)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Test vs Cross-Validation Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names_list])
ax.legend()
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig('report/images/figure9_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved: cross-validation")

# ==========================================
# Save main results
# ==========================================
results_summary = {
    'model_comparison': {name: {
        'auc': results[name]['auc'],
        'ap': results[name]['ap'],
        'f1': results[name]['f1'],
        'precision': results[name]['precision'],
        'recall': results[name]['recall'],
        'accuracy': results[name]['accuracy']
    } for name in results},
    'best_model': best_model_name,
    'degradation_results': deg_results,
    'cross_validation': {k: {kk: vv for kk, vv in v.items() if kk != 'auc_folds'} 
                         for k, v in cv_results.items()},
    'feature_importance': {
        'random_forest': {str(i): float(v) for i, v in enumerate(rf_importance)},
        'gradient_boosting': {str(i): float(v) for i, v in enumerate(gb_importance)}
    }
}

with open('outputs/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ All analysis complete!")
print(f"Best model: {best_model_name} (AUC={results[best_model_name]['auc']:.4f})")
print(f"All results saved to outputs/")
print(f"All figures saved to report/images/")
