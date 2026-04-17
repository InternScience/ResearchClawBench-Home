"""
Advanced models (XGBoost, LightGBM) and SHAP interpretability analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             average_precision_score, precision_score, recall_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import pickle

# Paths
DATA_DIR = 'data'
IMG_DIR = 'report/images'
OUT_DIR = 'outputs'

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
X_train = train[feature_cols].values
y_train = train['label'].values
X_test = test[feature_cols].values
y_test = test['label'].values
deg_test = test['degradation'].values

# Scale ratio for class imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Positive weight: {pos_weight:.2f}")

# ============================================================
# XGBoost
# ============================================================
print("\n" + "="*60)
print("Training: XGBoost")
print("="*60)
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    eval_metric='logloss', random_state=42,
    use_label_encoder=False, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - t0

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# ============================================================
# LightGBM
# ============================================================
print("\n" + "="*60)
print("Training: LightGBM")
print("="*60)
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - t0

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

# Evaluate both
for name, y_pred, y_prob, train_time in [
    ('XGBoost', y_pred_xgb, y_prob_xgb, xgb_time),
    ('LightGBM', y_pred_lgb, y_prob_lgb, lgb_time)
]:
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC:  {auc_roc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Time:      {train_time:.1f}s")

# Load previous results
with open(os.path.join(OUT_DIR, 'model_results.json'), 'r') as f:
    all_results = json.load(f)

# Add new results
for name, y_pred, y_prob, train_time, model in [
    ('XGBoost', y_pred_xgb, y_prob_xgb, xgb_time, xgb_model),
    ('LightGBM', y_pred_lgb, y_prob_lgb, lgb_time, lgb_model)
]:
    all_results[name] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_prob)),
        'auc_pr': float(average_precision_score(y_test, y_prob)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'train_time': float(train_time),
        'per_degradation': {}
    }
    
    for deg in sorted(np.unique(deg_test)):
        mask = deg_test == deg
        y_t = y_test[mask]
        y_p = y_pred[mask]
        y_pb = y_prob[mask]
        all_results[name]['per_degradation'][deg] = {
            'accuracy': float(accuracy_score(y_t, y_p)),
            'f1_score': float(f1_score(y_t, y_p)),
            'auc_roc': float(roc_auc_score(y_t, y_pb)),
            'auc_pr': float(average_precision_score(y_t, y_pb)),
            'precision': float(precision_score(y_t, y_p)),
            'recall': float(recall_score(y_t, y_p)),
        }

with open(os.path.join(OUT_DIR, 'model_results_all.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# ============================================================
# Updated ROC and PR curves with all models
# ============================================================
# Reload previous models
model_preds = {}
scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)

for mname in ['logistic_regression', 'random_forest', 'gradient_boosting', 'mlp']:
    with open(os.path.join(OUT_DIR, f'model_{mname}.pkl'), 'rb') as f:
        m = pickle.load(f)
    if mname in ['logistic_regression', 'mlp']:
        model_preds[mname] = m.predict_proba(X_test_scaled)[:, 1]
    else:
        model_preds[mname] = m.predict_proba(X_test)[:, 1]

model_preds['xgboost'] = y_prob_xgb
model_preds['lightgbm'] = y_prob_lgb

display_names = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'mlp': 'MLP',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
for i, (key, probs) in enumerate(model_preds.items()):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{display_names[key]} (AUC={auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'roc_curves_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved roc_curves_all.png")

# PR curves
fig, ax = plt.subplots(figsize=(9, 7))
for i, (key, probs) in enumerate(model_preds.items()):
    prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(rec_c, prec_c, color=colors[i], lw=2, label=f'{display_names[key]} (AP={ap:.4f})')
baseline_rate = y_test.sum() / len(y_test)
ax.axhline(y=baseline_rate, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({baseline_rate:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - All Models', fontsize=14)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'pr_curves_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pr_curves_all.png")

# ============================================================
# Feature Importance - XGBoost native
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost feature importance
xgb_imp = xgb_model.feature_importances_
sorted_idx = np.argsort(xgb_imp)
axes[0].barh(range(20), xgb_imp[sorted_idx], color='steelblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels([f'Feature {feature_cols[i]}' for i in sorted_idx])
axes[0].set_xlabel('Importance (gain)')
axes[0].set_title('XGBoost Feature Importance')

# LightGBM feature importance
lgb_imp = lgb_model.feature_importances_
sorted_idx2 = np.argsort(lgb_imp)
axes[1].barh(range(20), lgb_imp[sorted_idx2], color='coral')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f'Feature {feature_cols[i]}' for i in sorted_idx2])
axes[1].set_xlabel('Importance (split)')
axes[1].set_title('LightGBM Feature Importance')

plt.suptitle('Tree-Based Feature Importance', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'tree_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved tree_feature_importance.png")

# ============================================================
# SHAP Analysis on XGBoost
# ============================================================
print("\nComputing SHAP values...")
import shap

# Use a sample for SHAP (faster)
np.random.seed(42)
shap_sample_idx = np.random.choice(len(X_test), 2000, replace=False)
X_shap = X_test[shap_sample_idx]

# Fix for XGBoost 3.x + SHAP compatibility: use LightGBM instead
# or use KernelExplainer. Let's use LightGBM TreeExplainer which works.
try:
    explainer = shap.TreeExplainer(xgb_model)
except Exception as e:
    print(f"XGBoost SHAP failed ({e}), using LightGBM for SHAP instead")
    explainer = shap.TreeExplainer(lgb_model)

shap_values = explainer.shap_values(X_shap)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=[f'Feature {i}' for i in range(20)],
                  show=False, max_display=20)
plt.title('SHAP Summary Plot (XGBoost)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# SHAP bar plot (mean absolute SHAP)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, feature_names=[f'Feature {i}' for i in range(20)],
                  plot_type='bar', show=False, max_display=20)
plt.title('Mean |SHAP| Values (XGBoost)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_bar.png")

# Save SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = {f'Feature {i}': float(mean_shap[i]) for i in range(20)}
shap_importance_sorted = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
with open(os.path.join(OUT_DIR, 'shap_importance.json'), 'w') as f:
    json.dump(shap_importance_sorted, f, indent=2)
print("Saved shap_importance.json")

# ============================================================
# SHAP per degradation type
# ============================================================
deg_test_sample = deg_test[shap_sample_idx]
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

for i, deg in enumerate(sorted(np.unique(deg_test))):
    ax = axes[i // 2, i % 2]
    mask = deg_test_sample == deg
    if mask.sum() > 0:
        mean_shap_deg = np.abs(shap_values[mask]).mean(axis=0)
        sorted_idx = np.argsort(mean_shap_deg)
        ax.barh(range(20), mean_shap_deg[sorted_idx], color=colors[i])
        ax.set_yticks(range(20))
        ax.set_yticklabels([f'Feature {feature_cols[j]}' for j in sorted_idx])
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'{deg} (n={mask.sum()})', fontsize=12)

plt.suptitle('SHAP Feature Importance by Degradation Type', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'shap_per_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_per_degradation.png")

# ============================================================
# Permutation Importance for MLP (best model)
# ============================================================
print("\nComputing Permutation Importance for MLP...")
with open(os.path.join(OUT_DIR, 'model_mlp.pkl'), 'rb') as f:
    mlp_model = pickle.load(f)

# Use subset for speed
perm_sample_idx = np.random.choice(len(X_test), 5000, replace=False)
X_perm = X_test_scaled[perm_sample_idx]  # MLP uses scaled data
y_perm = y_test[perm_sample_idx]

perm_result = permutation_importance(mlp_model, X_perm, y_perm, 
                                     n_repeats=10, random_state=42, 
                                     scoring='f1', n_jobs=-1)

fig, ax = plt.subplots(figsize=(10, 8))
sorted_idx = perm_result.importances_mean.argsort()
ax.boxplot(perm_result.importances[sorted_idx].T, vert=False,
           labels=[f'Feature {feature_cols[i]}' for i in sorted_idx])
ax.set_xlabel('Decrease in F1 Score')
ax.set_title('Permutation Importance (MLP - Best Model)', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'permutation_importance_mlp.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved permutation_importance_mlp.png")

# Save permutation importance
perm_imp = {f'Feature {i}': float(perm_result.importances_mean[i]) for i in range(20)}
perm_imp_sorted = dict(sorted(perm_imp.items(), key=lambda x: x[1], reverse=True))
with open(os.path.join(OUT_DIR, 'permutation_importance_mlp.json'), 'w') as f:
    json.dump(perm_imp_sorted, f, indent=2)

# ============================================================
# Final comprehensive comparison table
# ============================================================
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
final_table = []
for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'MLP']:
    r = all_results[name]
    final_table.append({
        'Model': name,
        'Accuracy': f"{r['accuracy']:.4f}",
        'F1': f"{r['f1_score']:.4f}",
        'AUC-ROC': f"{r['auc_roc']:.4f}",
        'AUC-PR': f"{r['auc_pr']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'Time (s)': f"{r['train_time']:.1f}"
    })
final_df = pd.DataFrame(final_table)
print(final_df.to_string(index=False))
final_df.to_csv(os.path.join(OUT_DIR, 'final_comparison_table.csv'), index=False)

print("\n=== Advanced Modeling & Interpretability Complete ===")
