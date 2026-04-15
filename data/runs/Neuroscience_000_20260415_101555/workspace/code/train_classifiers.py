"""
SimBA-style Behavior Classification Pipeline
=============================================
Reproduces the SimBA workflow: pose-derived features -> supervised classifiers
-> quantitative evaluation with precision-recall diagnostics, confusion matrices,
and feature-importance tables.

Behaviors: Attack, Sniffing
Classifiers: Random Forest, Gradient Boosting, SVM, Logistic Regression, XGBoost
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data')
OUT_DIR = os.path.join(BASE, '..', 'outputs')
IMG_DIR = os.path.join(BASE, '..', 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 1: Loading Data")
print("=" * 60)

feat_df = pd.read_csv(os.path.join(DATA_DIR, 'Together_1_features_extracted.csv'))
targ_df = pd.read_csv(os.path.join(DATA_DIR, 'Together_1_targets_inserted.csv'))
ref_df = pd.read_csv(os.path.join(DATA_DIR, 'Together_1_machine_results_reference.csv'))

# Drop unnamed index column if present
if 'Unnamed: 0' in feat_df.columns:
    feat_df = feat_df.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in targ_df.columns:
    targ_df = targ_df.drop(columns=['Unnamed: 0'])

# Feature columns: everything except Feature_1 and Feature_2 (which are just frame indices)
feature_cols = [c for c in feat_df.columns if c not in ('Feature_1', 'Feature_2')]
X = feat_df[feature_cols].values
y_attack = targ_df['Attack'].values
y_sniffing = targ_df['Sniffing'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Attack labels: {y_attack.sum()} positive / {(~y_attack.astype(bool)).sum()} negative")
print(f"Sniffing labels: {y_sniffing.sum()} positive / {(~y_sniffing.astype(bool)).sum()} negative")

# Save feature names
with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_cols, f, indent=2)

# ── 2. Data Overview Statistics ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Data Overview")
print("=" * 60)

feat_stats = feat_df[feature_cols].describe().T
feat_stats.to_csv(os.path.join(OUT_DIR, 'feature_statistics.csv'))
print("Feature statistics saved to outputs/feature_statistics.csv")

# Class balance overview
class_balance = {
    'Attack': {'positive': int(y_attack.sum()), 'negative': int((~y_attack.astype(bool)).sum()),
               'pos_ratio': float(y_attack.mean())},
    'Sniffing': {'positive': int(y_sniffing.sum()), 'negative': int((~y_sniffing.astype(bool)).sum()),
                 'pos_ratio': float(y_sniffing.mean())}
}
with open(os.path.join(OUT_DIR, 'class_balance.json'), 'w') as f:
    json.dump(class_balance, f, indent=2)
print(f"Class balance: {json.dumps(class_balance, indent=2)}")

# ── 3. Train / Test Split ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3: Train/Test Split")
print("=" * 60)

X_train, X_test, y_atk_train, y_atk_test, y_sni_train, y_sni_test = train_test_split(
    X, y_attack, y_sniffing, test_size=0.25, random_state=RANDOM_STATE, stratify=y_attack
)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Train Attack pos ratio: {y_atk_train.mean():.3f}")
print(f"Test  Attack pos ratio: {y_atk_test.mean():.3f}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Classifier Definitions ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 4: Training Classifiers")
print("=" * 60)

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, C=1.0),
}

try:
    import xgboost as xgb
    classifiers['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False
    )
except ImportError:
    pass

# ── 5. Training & Evaluation Loop ────────────────────────────────────────────
results = {}

for behavior_name, y_train, y_test in [('Attack', y_atk_train, y_atk_test),
                                         ('Sniffing', y_sni_train, y_sni_test)]:
    print(f"\n--- Training for {behavior_name} ---")
    results[behavior_name] = {}
    
    for clf_name, clf in classifiers.items():
        print(f"  Training {clf_name}...")
        
        # Use scaled data for SVM and LR, raw for tree-based
        if clf_name in ('SVM (RBF)', 'Logistic Regression'):
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        
        # Probability predictions
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_te)[:, 1]
        else:
            y_prob = y_pred.astype(float)
        
        # Metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'avg_precision': float(average_precision_score(y_test, y_prob)),
        }
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics['roc_auc'] = None
        
        results[behavior_name][clf_name] = {
            'metrics': metrics,
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'y_test': y_test.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"    Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
              f"AP={metrics['avg_precision']:.4f}")

# Save results
with open(os.path.join(OUT_DIR, 'classifier_results.json'), 'w') as f:
    # Convert numpy types for JSON serialization
    serializable = {}
    for beh, cls_dict in results.items():
        serializable[beh] = {}
        for cn, res in cls_dict.items():
            serializable[beh][cn] = {k: v for k, v in res.items() if k not in ('y_pred', 'y_prob', 'y_test')}
    json.dump(serializable, f, indent=2)
print("\nResults saved to outputs/classifier_results.json")

# ── 6. Comparison Table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 5: Generating Comparison Tables")
print("=" * 60)

for behavior_name in ['Attack', 'Sniffing']:
    rows = []
    for clf_name, res in results[behavior_name].items():
        m = res['metrics']
        rows.append({
            'Classifier': clf_name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1']:.4f}",
            'Avg Precision': f"{m['avg_precision']:.4f}",
            'ROC AUC': f"{m['roc_auc']:.4f}" if m['roc_auc'] else 'N/A'
        })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(OUT_DIR, f'{behavior_name.lower()}_comparison.csv'), index=False)
    print(f"\n{behavior_name} comparison table:")
    print(comp_df.to_string(index=False))

# ── 7. Feature Importance (Random Forest) ────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 6: Feature Importance Analysis")
print("=" * 60)

for behavior_name, y_train in [('Attack', y_atk_train), ('Sniffing', y_sni_train)]:
    rf = classifiers['Random Forest']
    # Re-fit on full training data for feature importance
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fi_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in indices],
        'Importance': importances[indices]
    })
    fi_df.to_csv(os.path.join(OUT_DIR, f'{behavior_name.lower()}_feature_importance.csv'), index=False)
    
    print(f"\nTop 15 features for {behavior_name}:")
    print(fi_df.head(15).to_string(index=False))

# ── 8. Generate Figures ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 7: Generating Figures")
print("=" * 60)

# Figure 1: Class Balance Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
behaviors = ['Attack', 'Sniffing']
pos_counts = [class_balance[b]['positive'] for b in behaviors]
neg_counts = [class_balance[b]['negative'] for b in behaviors]
x = np.arange(len(behaviors))
width = 0.35
ax.bar(x - width/2, neg_counts, width, label='Negative (0)', color='#4a90d9')
ax.bar(x + width/2, pos_counts, width, label='Positive (1)', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels(behaviors)
ax.set_ylabel('Frame Count')
ax.set_title('Class Distribution per Behavior')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_01_class_distribution.png'), dpi=150)
plt.close()
print("Saved figure_01_class_distribution.png")

# Figure 2: Confusion Matrices
fig, axes = plt.subplots(2, len(classifiers), figsize=(4*len(classifiers), 8))
if len(classifiers) == 1:
    axes = axes.reshape(2, 1)

for j, (clf_name, _) in enumerate(classifiers.items()):
    for i, behavior_name in enumerate(['Attack', 'Sniffing']):
        cm = np.array(results[behavior_name][clf_name]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, j],
                   xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        axes[i, j].set_title(f'{clf_name}\n{behavior_name}')
        axes[i, j].set_xlabel('Predicted')
        axes[i, j].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_02_confusion_matrices.png'), dpi=150)
plt.close()
print("Saved figure_02_confusion_matrices.png")

# Figure 3: Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('husl', len(classifiers))

for i, behavior_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[i]
    for j, (clf_name, _) in enumerate(classifiers.items()):
        y_test = np.array(results[behavior_name][clf_name]['y_test'])
        y_prob = np.array(results[behavior_name][clf_name]['y_prob'])
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f'{clf_name} (AP={ap:.3f})',
               color=colors[j], linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve: {behavior_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_03_pr_curves.png'), dpi=150)
plt.close()
print("Saved figure_03_pr_curves.png")

# Figure 4: ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, behavior_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[i]
    for j, (clf_name, _) in enumerate(classifiers.items()):
        y_test = np.array(results[behavior_name][clf_name]['y_test'])
        y_prob = np.array(results[behavior_name][clf_name]['y_prob'])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{clf_name} (AUC={roc_auc:.3f})',
               color=colors[j], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {behavior_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_04_roc_curves.png'), dpi=150)
plt.close()
print("Saved figure_04_roc_curves.png")

# Figure 5: Feature Importance Horizontal Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for i, behavior_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[i]
    fi_df = pd.read_csv(os.path.join(OUT_DIR, f'{behavior_name.lower()}_feature_importance.csv'))
    top_n = 15
    top_fi = fi_df.head(top_n).sort_values('Importance', ascending=True)
    ax.barh(top_fi['Feature'], top_fi['Importance'], color=colors[0])
    ax.set_xlabel('Feature Importance (Gini Impurity Reduction)')
    ax.set_title(f'Top {top_n} Features: {behavior_name}')
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_05_feature_importance.png'), dpi=150)
plt.close()
print("Saved figure_05_feature_importance.png")

# Figure 6: Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metric_names = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
clf_names = list(classifiers.keys())

for idx, (metric, title) in enumerate(zip(metric_names, titles)):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(clf_names))
    width = 0.35
    
    vals_attack = [results['Attack'][cn]['metrics'][metric] for cn in clf_names]
    vals_sniff = [results['Sniffing'][cn]['metrics'][metric] for cn in clf_names]
    
    ax.bar(x - width/2, vals_attack, width, label='Attack', color='#4a90d9')
    ax.bar(x + width/2, vals_sniff, width, label='Sniffing', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(title)
    ax.set_title(f'{title} Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_06_performance_comparison.png'), dpi=150)
plt.close()
print("Saved figure_06_performance_comparison.png")

# Figure 7: SHAP-like permutation importance (using RF)
print("\nComputing permutation importance...")
from sklearn.inspection import permutation_importance

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for i, behavior_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[i]
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_atk_train if behavior_name == 'Attack' else y_sni_train)
    
    perm_imp = permutation_importance(rf, X_test, 
                                       y_atk_test if behavior_name == 'Attack' else y_sni_test,
                                       n_repeats=10, random_state=RANDOM_STATE, scoring='f1')
    
    mean_imp = perm_imp.importances_mean
    indices = np.argsort(mean_imp)[::-1]
    top_idx = indices[:15]
    
    ax.barh([feature_cols[j] for j in top_idx][::-1],
           [mean_imp[j] for j in top_idx][::-1], color='#2ecc71')
    ax.set_xlabel('Decrease in F1 Score (Permutation Importance)')
    ax.set_title(f'Permutation Importance: {behavior_name}')
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_07_permutation_importance.png'), dpi=150)
plt.close()
print("Saved figure_07_permutation_importance.png")

# ── 9. Cross-Validation Results ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 8: Cross-Validation")
print("=" * 60)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for behavior_name, y_full in [('Attack', y_attack), ('Sniffing', y_sniffing)]:
    cv_results[behavior_name] = {}
    for clf_name, clf in classifiers.items():
        if clf_name in ('SVM (RBF)', 'Logistic Regression'):
            scores = cross_val_score(clf, X_train_scaled, y_full[:len(X_train)], cv=skf, scoring='f1')
        else:
            scores = cross_val_score(clf, X_train, y_full[:len(X_train)], cv=skf, scoring='f1')
        cv_results[behavior_name][clf_name] = {
            'mean_f1': float(np.mean(scores)),
            'std_f1': float(np.std(scores)),
            'scores': scores.tolist()
        }
        print(f"  {behavior_name} - {clf_name}: F1 = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

with open(os.path.join(OUT_DIR, 'cross_validation_results.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

# Figure 8: CV Results Boxplot
fig, ax = plt.subplots(figsize=(12, 5))
clf_names = list(classifiers.keys())
x = np.arange(len(clf_names))
width = 0.35

f1_attack_cv = [cv_results['Attack'][cn]['scores'] for cn in clf_names]
f1_sniff_cv = [cv_results['Sniffing'][cn]['scores'] for cn in clf_names]

bp1 = ax.boxplot(f1_attack_cv, positions=x - width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='#4a90d9', alpha=0.7))
bp2 = ax.boxplot(f1_sniff_cv, positions=x + width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='#e74c3c', alpha=0.7))

ax.set_xticks(x)
ax.set_xticklabels(clf_names, rotation=30, ha='right')
ax.set_ylabel('F1 Score (5-fold CV)')
ax.set_title('Cross-Validation F1 Score Distribution')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Attack', 'Sniffing'])
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'figure_08_cv_boxplot.png'), dpi=150)
plt.close()
print("Saved figure_08_cv_boxplot.png")

# ── 10. Summary Export ───────────────────────────────────────────────────────
summary = {
    'best_attack': max(results['Attack'].items(), key=lambda x: x[1]['metrics']['f1']),
    'best_sniffing': max(results['Sniffing'].items(), key=lambda x: x[1]['metrics']['f1']),
    'total_features': len(feature_cols),
    'total_samples': len(y_attack),
    'train_size': len(X_train),
    'test_size': len(X_test),
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"Best Attack classifier: {summary['best_attack'][0]} (F1={summary['best_attack'][1]['metrics']['f1']:.4f})")
print(f"Best Sniffing classifier: {summary['best_sniffing'][0]} (F1={summary['best_sniffing'][1]['metrics']['f1']:.4f})")
