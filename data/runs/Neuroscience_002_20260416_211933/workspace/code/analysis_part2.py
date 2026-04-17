#!/usr/bin/env python3
"""Part 2: Model training and evaluation - simplified"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import json
import os

np.random.seed(42)

WORKSPACE = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260416_211933'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
train_df.columns = feature_cols + ['label', 'degradation']
test_df.columns = feature_cols + ['label', 'degradation']

X_train = train_df[feature_cols].values
y_train = train_df['label'].values.astype(int)
X_test = test_df[feature_cols].values
y_test = test_df['label'].values.astype(int)

print("Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Train Logistic Regression (faster)
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

lr_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_proba_lr)
}
print(f"LR: F1={lr_metrics['f1']:.4f}, AUC={lr_metrics['roc_auc']:.4f}")

# Train Random Forest with fewer trees
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=4)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_proba_rf)
}
print(f"RF: F1={rf_metrics['f1']:.4f}, AUC={rf_metrics['roc_auc']:.4f}")

results = {'Logistic Regression': lr_metrics, 'Random Forest': rf_metrics}
models = {'Logistic Regression': (lr, y_pred_lr, y_proba_lr), 'Random Forest': (rf, y_pred_rf, y_proba_rf)}
best_name = 'Random Forest' if rf_metrics['f1'] > lr_metrics['f1'] else 'Logistic Regression'
print(f"Best: {best_name}")

# Save results
with open(os.path.join(OUTPUTS_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Fig 5: Model comparison
print("Plotting...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
mets = ['f1', 'roc_auc', 'recall']
titles = ['F1 Score', 'ROC-AUC', 'Recall']
colors = ['#3498DB', '#E74C3C']

for idx, (m, t) in enumerate(zip(mets, titles)):
    ax = axes[idx]
    vals = [results[n][m] for n in models.keys()]
    bars = ax.bar(models.keys(), vals, color=colors)
    ax.set_title(t, fontsize=12, fontweight='bold')
    ax.set_ylabel(t)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig05_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig05")

# Fig 6: ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, color in zip(models.keys(), colors):
    _, y_proba, _ = models[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'{name} ({results[name]["roc_auc"]:.4f})', color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig06_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig06")

# Fig 7: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
model, y_pred, _ = models[best_name]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'], ax=ax)
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig07_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig07")

# Degradation analysis
print("Analyzing by degradation...")
deg_order = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
test_df_copy = test_df.copy()
test_df_copy['pred'] = y_pred_rf

deg_metrics = {}
for deg in deg_order:
    mask = test_df_copy['degradation'] == deg
    y_t = test_df_copy.loc[mask, 'label'].values.astype(int)
    y_p = test_df_copy.loc[mask, 'pred'].values
    deg_metrics[deg] = {
        'f1': float(f1_score(y_t, y_p, zero_division=0)),
        'precision': float(precision_score(y_t, y_p, zero_division=0)),
        'recall': float(recall_score(y_t, y_p, zero_division=0)),
        'n': int(mask.sum())
    }

# Fig 9: By degradation
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
mets = ['precision', 'recall', 'f1']
titles = ['Precision', 'Recall', 'F1 Score']
cols = ['#3498DB', '#E74C3C', '#2ECC71']

for idx, (m, t) in enumerate(zip(mets, titles)):
    ax = axes[idx // 2, idx % 2]
    vals = [deg_metrics[d][m] for d in deg_order]
    bars = ax.bar(deg_order, vals, color=cols[idx])
    ax.set_title(f'{t} by Degradation', fontsize=11, fontweight='bold')
    ax.set_ylabel(t)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=8)

axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig09_performance_by_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig09")

with open(os.path.join(OUTPUTS_DIR, 'degradation_metrics.json'), 'w') as f:
    json.dump(deg_metrics, f, indent=2)

# Feature importance
print("Feature importance...")
fi = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

# Fig 10
fig, ax = plt.subplots(figsize=(8, 7))
top_n = 12
top_fi = fi.head(top_n)
bars = ax.barh(range(top_n), top_fi['importance'].values, color='#3498DB')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f'Feature {int(f)}' for f in top_fi['feature'].values])
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'fig10_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10")

fi.to_csv(os.path.join(OUTPUTS_DIR, 'feature_importance.csv'), index=False)

# Summary
summary = {
    'best_model': best_name,
    'best_metrics': results[best_name],
    'all_models': results,
    'degradation': deg_metrics
}
with open(os.path.join(OUTPUTS_DIR, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*50)
print("COMPLETE")
print(f"Best: {best_name}, F1={results[best_name]['f1']:.4f}")
print("="*50)
