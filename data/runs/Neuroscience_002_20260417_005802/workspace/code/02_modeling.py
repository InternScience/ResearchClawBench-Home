"""
Model Training and Evaluation for Connectomics Proofreading
Multiple classifiers with comprehensive evaluation
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             average_precision_score, precision_score, recall_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)
import pickle

# Paths
DATA_DIR = 'data'
IMG_DIR = 'report/images'
OUT_DIR = 'outputs'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
X_train = train[feature_cols].values
y_train = train['label'].values
X_test = test[feature_cols].values
y_test = test['label'].values
deg_train = train['degradation'].values
deg_test = test['degradation'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Define models
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=42, batch_size=512, learning_rate_init=0.001
    )
}

# ============================================================
# Train and evaluate
# ============================================================
results = {}
roc_data = {}
pr_data = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Use scaled data for LR and MLP, raw for tree methods
    if name in ['Logistic Regression', 'MLP']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    train_time = time.time() - t0
    
    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'precision': float(prec),
        'recall': float(rec),
        'train_time': float(train_time)
    }
    
    # ROC and PR curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr)
    pr_data[name] = (prec_curve, rec_curve)
    
    # Per-degradation metrics
    deg_results = {}
    for deg in sorted(np.unique(deg_test)):
        mask = deg_test == deg
        y_t = y_test[mask]
        y_p = y_pred[mask]
        y_pb = y_prob[mask]
        
        deg_results[deg] = {
            'accuracy': float(accuracy_score(y_t, y_p)),
            'f1_score': float(f1_score(y_t, y_p)),
            'auc_roc': float(roc_auc_score(y_t, y_pb)),
            'auc_pr': float(average_precision_score(y_t, y_pb)),
            'precision': float(precision_score(y_t, y_p)),
            'recall': float(recall_score(y_t, y_p)),
            'n_samples': int(mask.sum()),
            'n_positive': int(y_t.sum())
        }
    
    results[name]['per_degradation'] = deg_results
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC:  {auc_roc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Time:      {train_time:.1f}s")
    
    # Save model
    with open(os.path.join(OUT_DIR, f'model_{name.lower().replace(" ", "_")}.pkl'), 'wb') as f:
        pickle.dump(model, f)

# Save results
with open(os.path.join(OUT_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved model_results.json")

# ============================================================
# Figure: ROC Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
    auc = results[name]['auc_roc']
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC={auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved roc_curves.png")

# ============================================================
# Figure: PR Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
for i, (name, (prec_c, rec_c)) in enumerate(pr_data.items()):
    auc = results[name]['auc_pr']
    ax.plot(rec_c, prec_c, color=colors[i], lw=2, label=f'{name} (AP={auc:.4f})')
baseline_rate = y_test.sum() / len(y_test)
ax.axhline(y=baseline_rate, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline ({baseline_rate:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pr_curves.png")

# ============================================================
# Figure: Model Comparison Bar Chart
# ============================================================
metrics_to_plot = ['accuracy', 'f1_score', 'auc_roc', 'auc_pr', 'precision', 'recall']
model_names = list(results.keys())

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(metrics_to_plot))
width = 0.18
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall'])
ax.legend(fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_comparison.png")

# ============================================================
# Figure: Confusion Matrices for best model
# ============================================================
# Find best model by F1
best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
print(f"\nBest model by F1: {best_model_name}")

# Reload best model
with open(os.path.join(OUT_DIR, f'model_{best_model_name.lower().replace(" ", "_")}.pkl'), 'rb') as f:
    best_model = pickle.load(f)

if best_model_name in ['Logistic Regression', 'MLP']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))

# Overall confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No merge', 'Merge'], yticklabels=['No merge', 'Merge'])
axes[0].set_title(f'Overall ({best_model_name})', fontsize=11)
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Per-degradation confusion matrices
for i, deg in enumerate(sorted(np.unique(deg_test))):
    mask = deg_test == deg
    cm_deg = confusion_matrix(y_test[mask], y_pred_best[mask])
    sns.heatmap(cm_deg, annot=True, fmt='d', cmap='Blues', ax=axes[i+1],
                xticklabels=['No merge', 'Merge'], yticklabels=['No merge', 'Merge'])
    axes[i+1].set_title(f'{deg}', fontsize=11)
    axes[i+1].set_ylabel('True')
    axes[i+1].set_xlabel('Predicted')

plt.suptitle(f'Confusion Matrices - {best_model_name}', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved confusion_matrices.png")

# ============================================================
# Figure: Per-degradation performance heatmap
# ============================================================
deg_types_sorted = sorted(np.unique(deg_test))
metrics_for_heatmap = ['accuracy', 'f1_score', 'auc_roc', 'auc_pr']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for mi, metric in enumerate(metrics_for_heatmap):
    ax = axes[mi // 2, mi % 2]
    data_matrix = []
    for name in model_names:
        row = [results[name]['per_degradation'][deg][metric] for deg in deg_types_sorted]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    
    sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                xticklabels=deg_types_sorted, yticklabels=model_names)
    ax.set_title(metric.replace('_', ' ').title(), fontsize=12)

plt.suptitle('Per-Degradation Performance by Model', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'per_degradation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_degradation_heatmap.png")

# ============================================================
# Save comparison table
# ============================================================
comparison_table = []
for name in model_names:
    row = {'Model': name}
    for m in ['accuracy', 'f1_score', 'auc_roc', 'auc_pr', 'precision', 'recall', 'train_time']:
        row[m] = results[name][m]
    comparison_table.append(row)

comp_df = pd.DataFrame(comparison_table)
comp_df.to_csv(os.path.join(OUT_DIR, 'model_comparison_table.csv'), index=False)
print("\nModel Comparison Table:")
print(comp_df.to_string(index=False))

# Per-degradation table
deg_table = []
for name in model_names:
    for deg in deg_types_sorted:
        row = {'Model': name, 'Degradation': deg}
        for m in ['accuracy', 'f1_score', 'auc_roc', 'auc_pr', 'precision', 'recall']:
            row[m] = results[name]['per_degradation'][deg][m]
        deg_table.append(row)

deg_df = pd.DataFrame(deg_table)
deg_df.to_csv(os.path.join(OUT_DIR, 'per_degradation_table.csv'), index=False)
print("\nPer-degradation table saved")

print("\n=== Model Training & Evaluation Complete ===")
