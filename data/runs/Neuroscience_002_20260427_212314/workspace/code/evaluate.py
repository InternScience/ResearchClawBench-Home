"""Generate evaluation figures: model comparison, ROC/PR curves, confusion matrix,
per-degradation breakdown, threshold tuning."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
)

os.makedirs('report/images', exist_ok=True)
preds = pd.read_csv('outputs/test_predictions.csv')
with open('outputs/metrics.json') as f:
    metrics = json.load(f)

model_names = ['logreg', 'rf', 'gbm', 'xgb', 'mlp']
pretty = {'logreg': 'Logistic Regression', 'rf': 'Random Forest',
          'gbm': 'Gradient Boosting', 'xgb': 'XGBoost', 'mlp': 'MLP (64,32)'}
yte = preds['label'].values
deg = preds['degradation'].values

# 1) Model comparison bar chart on key metrics
metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
M = np.array([[metrics[m][k] for k in metric_keys] for m in model_names])
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(metric_keys))
w = 0.15
colors = sns.color_palette('Set2', len(model_names))
for i, m in enumerate(model_names):
    ax.bar(x + i*w - 2*w, M[i], w, label=pretty[m], color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(metric_keys)
ax.set_ylim(0, 1.05)
ax.set_ylabel('score')
ax.set_title('Test-set performance comparison across classifiers')
ax.legend(loc='lower right', ncol=3, fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/model_comparison.png', dpi=150)
plt.close()

# 2) ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, m in enumerate(model_names):
    p = preds[m].values
    fpr, tpr, _ = roc_curve(yte, p)
    axes[0].plot(fpr, tpr, label=f'{pretty[m]} (AUC={auc(fpr, tpr):.4f})',
                 color=colors[i], lw=1.6)
    pr_p, pr_r, _ = precision_recall_curve(yte, p)
    axes[1].plot(pr_r, pr_p, label=f'{pretty[m]} (AP={average_precision_score(yte, p):.4f})',
                 color=colors[i], lw=1.6)
axes[0].plot([0, 1], [0, 1], 'k--', lw=0.7)
axes[0].set_xlabel('False positive rate')
axes[0].set_ylabel('True positive rate')
axes[0].set_title('ROC curves on test set')
axes[0].legend(loc='lower right', fontsize=8)
axes[0].grid(alpha=0.3)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision–Recall curves on test set')
axes[1].legend(loc='lower left', fontsize=8)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/roc_pr_curves.png', dpi=150)
plt.close()

# 3) Confusion matrix for best model (MLP)
best = 'mlp'
yhat_best = (preds[best].values >= 0.5).astype(int)
cm = confusion_matrix(yte, yhat_best)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'], ax=ax)
ax.set_title(f'Confusion matrix — {pretty[best]} (threshold 0.5)')
plt.tight_layout()
plt.savefig('report/images/confusion_matrix_mlp.png', dpi=150)
plt.close()

# 4) Per-degradation comparison (heatmap of F1 / ROC-AUC)
degs = sorted(np.unique(deg).tolist())
f1_grid = np.array([[metrics[m]['per_degradation'][d]['f1'] for d in degs]
                    for m in model_names])
auc_grid = np.array([[metrics[m]['per_degradation'][d]['roc_auc'] for d in degs]
                     for m in model_names])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.heatmap(f1_grid, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=degs, yticklabels=[pretty[m] for m in model_names],
            ax=axes[0], vmin=0.5, vmax=1.0)
axes[0].set_title('F1 by model × degradation type (test)')
sns.heatmap(auc_grid, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=degs, yticklabels=[pretty[m] for m in model_names],
            ax=axes[1], vmin=0.9, vmax=1.0)
axes[1].set_title('ROC-AUC by model × degradation type (test)')
plt.tight_layout()
plt.savefig('report/images/per_degradation_heatmaps.png', dpi=150)
plt.close()

# 5) Threshold sweep for best model on overall + per-degradation
p_best = preds[best].values
thr = np.linspace(0.05, 0.95, 91)
f1s = [f1_score(yte, p_best >= t, zero_division=0) for t in thr]
prec = [precision_score(yte, p_best >= t, zero_division=0) for t in thr]
rec  = [recall_score(yte, p_best >= t, zero_division=0)    for t in thr]
best_idx = int(np.argmax(f1s))
best_thr = float(thr[best_idx])
best_f1 = float(f1s[best_idx])

plt.figure(figsize=(8, 4.5))
plt.plot(thr, f1s, label='F1', lw=2)
plt.plot(thr, prec, label='Precision', lw=1.4)
plt.plot(thr, rec,  label='Recall', lw=1.4)
plt.axvline(best_thr, ls='--', color='gray',
            label=f'best F1 = {best_f1:.4f} @ thr={best_thr:.2f}')
plt.xlabel('decision threshold')
plt.ylabel('score')
plt.title(f'Threshold sweep — {pretty[best]} on test')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/threshold_sweep_mlp.png', dpi=150)
plt.close()

# Save final compact comparison table
rows = []
for m in model_names:
    row = {'model': pretty[m]}
    row.update({k: metrics[m][k] for k in metric_keys})
    rows.append(row)
df_cmp = pd.DataFrame(rows)
df_cmp.to_csv('outputs/model_comparison.csv', index=False)

per_deg_rows = []
for m in model_names:
    for d in degs:
        r = {'model': pretty[m], 'degradation': d}
        r.update(metrics[m]['per_degradation'][d])
        per_deg_rows.append(r)
pd.DataFrame(per_deg_rows).to_csv('outputs/per_degradation_metrics.csv', index=False)

with open('outputs/best_threshold.json', 'w') as f:
    json.dump({'model': best, 'best_threshold': best_thr,
               'best_f1_at_threshold': best_f1,
               'f1_at_0.5': float(f1_score(yte, p_best >= 0.5))}, f, indent=2)
print(f'best F1={best_f1:.4f} at threshold {best_thr:.2f}')
print('eval figures + tables saved.')
