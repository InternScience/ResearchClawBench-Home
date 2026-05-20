#!/usr/bin/env python3
"""m6A modification detection analysis with PR/ROC curves."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import json
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
labels = pd.read_csv('data/m6a_labels.csv')
pred_u4 = pd.read_csv('data/m6a_predictions_uncalled4.csv')
pred_np = pd.read_csv('data/m6a_predictions_nanopolish.csv')

# Merge on site_id
df = labels.merge(pred_u4, on='site_id', suffixes=('', '_u4'))
df = df.merge(pred_np, on='site_id', suffixes=('_u4', '_np'))
df.columns = ['site_id', 'label', 'prob_uncalled4', 'prob_nanopolish']

print(f"Total sites: {len(df)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
print(f"Label proportion: {df['label'].mean():.4f}")

# Basic stats
print(f"\nUncalled4 predictions:")
print(f"  Mean: {df['prob_uncalled4'].mean():.4f}")
print(f"  Std:  {df['prob_uncalled4'].std():.4f}")
print(f"  Min:  {df['prob_uncalled4'].min():.4f}")
print(f"  Max:  {df['prob_uncalled4'].max():.4f}")

print(f"\nNanopolish predictions:")
print(f"  Mean: {df['prob_nanopolish'].mean():.4f}")
print(f"  Std:  {df['prob_nanopolish'].std():.4f}")
print(f"  Min:  {df['prob_nanopolish'].min():.4f}")
print(f"  Max:  {df['prob_nanopolish'].max():.4f}")

# === PR and ROC curves ===
y_true = df['label'].values
y_score_u4 = df['prob_uncalled4'].values
y_score_np = df['prob_nanopolish'].values

# Compute PR curves
precision_u4, recall_u4, thresholds_u4 = precision_recall_curve(y_true, y_score_u4)
precision_np, recall_np, thresholds_np = precision_recall_curve(y_true, y_score_np)
auprc_u4 = average_precision_score(y_true, y_score_u4)
auprc_np = average_precision_score(y_true, y_score_np)

# Compute ROC curves
fpr_u4, tpr_u4, _ = roc_curve(y_true, y_score_u4)
fpr_np, tpr_np, _ = roc_curve(y_true, y_score_np)
auroc_u4 = roc_auc_score(y_true, y_score_u4)
auroc_np = roc_auc_score(y_true, y_score_np)

print(f"\nAUPRC - Uncalled4: {auprc_u4:.4f}, Nanopolish: {auprc_np:.4f}")
print(f"AUROC - Uncalled4: {auroc_u4:.4f}, Nanopolish: {auroc_np:.4f}")

# === Figure 8: PR and ROC curves ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PR Curve
ax = axes[0]
ax.plot(recall_u4, precision_u4, 'b-', linewidth=2, label=f'Uncalled4 (AUPRC={auprc_u4:.3f})')
ax.plot(recall_np, precision_np, 'r-', linewidth=2, label=f'Nanopolish (AUPRC={auprc_np:.3f})')
# Baseline (random)
baseline = y_true.mean()
ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Random (={baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve: m6A Detection')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

# ROC Curve
ax = axes[1]
ax.plot(fpr_u4, tpr_u4, 'b-', linewidth=2, label=f'Uncalled4 (AUROC={auroc_u4:.3f})')
ax.plot(fpr_np, tpr_np, 'r-', linewidth=2, label=f'Nanopolish (AUROC={auroc_np:.3f})')
ax.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5, label='Random (0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: m6A Detection')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig8_m6a_pr_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved: fig8_m6a_pr_roc.png")

# === Figure 9: Prediction score distributions by label ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (scores, title) in zip(axes, [
    (y_score_u4, 'Uncalled4 Alignments'),
    (y_score_np, 'Nanopolish Alignments')
]):
    for label_val, color, name in [(0, 'steelblue', 'Unmodified (0)'), (1, 'coral', 'Modified (1)')]:
        mask = y_true == label_val
        ax.hist(scores[mask], bins=50, alpha=0.5, color=color, label=f'{name} (n={mask.sum()})', density=True)
    ax.set_xlabel('m6Anet Prediction Probability')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig9_prediction_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved: fig9_prediction_distributions.png")

# === Figure 10: F1 score vs threshold ===
thresholds = np.linspace(0.01, 0.99, 99)
f1_u4_vals = []
f1_np_vals = []
prec_u4_vals = []
prec_np_vals = []
rec_u4_vals = []
rec_np_vals = []

for t in thresholds:
    pred_u4_bin = (y_score_u4 >= t).astype(int)
    pred_np_bin = (y_score_np >= t).astype(int)
    f1_u4_vals.append(f1_score(y_true, pred_u4_bin))
    f1_np_vals.append(f1_score(y_true, pred_np_bin))
    prec_u4_vals.append(precision_score(y_true, pred_u4_bin, zero_division=0))
    prec_np_vals.append(precision_score(y_true, pred_np_bin, zero_division=0))
    rec_u4_vals.append(recall_score(y_true, pred_u4_bin))
    rec_np_vals.append(recall_score(y_true, pred_np_bin))

f1_u4_vals = np.array(f1_u4_vals)
f1_np_vals = np.array(f1_np_vals)

best_t_u4 = thresholds[np.argmax(f1_u4_vals)]
best_t_np = thresholds[np.argmax(f1_np_vals)]
best_f1_u4 = np.max(f1_u4_vals)
best_f1_np = np.max(f1_np_vals)

print(f"\nBest F1 - Uncalled4: {best_f1_u4:.4f} at threshold {best_t_u4:.2f}")
print(f"Best F1 - Nanopolish: {best_f1_np:.4f} at threshold {best_t_np:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# F1 vs threshold
ax = axes[0]
ax.plot(thresholds, f1_u4_vals, 'b-', linewidth=2, label=f'Uncalled4 (max={best_f1_u4:.3f} @ t={best_t_u4:.2f})')
ax.plot(thresholds, f1_np_vals, 'r-', linewidth=2, label=f'Nanopolish (max={best_f1_np:.3f} @ t={best_t_np:.2f})')
ax.axvline(x=best_t_u4, color='b', linestyle=':', alpha=0.5)
ax.axvline(x=best_t_np, color='r', linestyle=':', alpha=0.5)
ax.set_xlabel('Probability Threshold')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs. Decision Threshold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Precision & Recall vs threshold for Uncalled4
ax = axes[1]
ax.plot(thresholds, prec_u4_vals, 'b-', linewidth=1.5, label='Precision (Uncalled4)')
ax.plot(thresholds, rec_u4_vals, 'b--', linewidth=1.5, label='Recall (Uncalled4)')
ax.plot(thresholds, prec_np_vals, 'r-', linewidth=1.5, label='Precision (Nanopolish)')
ax.plot(thresholds, rec_np_vals, 'r--', linewidth=1.5, label='Recall (Nanopolish)')
ax.set_xlabel('Probability Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision/Recall vs. Decision Threshold')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig10_f1_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10 saved: fig10_f1_threshold.png")

# === Figure 11: Score agreement between Uncalled4 and Nanopolish ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter
ax = axes[0]
for label_val, color, name in [(0, 'steelblue', 'Unmodified'), (1, 'coral', 'Modified')]:
    mask = y_true == label_val
    ax.scatter(df.loc[mask, 'prob_uncalled4'], df.loc[mask, 'prob_nanopolish'], 
               alpha=0.3, s=2, c=color, label=name, rasterized=True)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='y=x')
ax.set_xlabel('Uncalled4 Probability')
ax.set_ylabel('Nanopolish Probability')
ax.set_title(f'Prediction Agreement\n(r={df["prob_uncalled4"].corr(df["prob_nanopolish"]):.3f})')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Difference histogram
ax = axes[1]
df['prob_diff'] = df['prob_uncalled4'] - df['prob_nanopolish']
for label_val, color, name in [(0, 'steelblue', 'Unmodified'), (1, 'coral', 'Modified')]:
    mask = y_true == label_val
    ax.hist(df.loc[mask, 'prob_diff'], bins=50, alpha=0.5, color=color, label=name, density=True)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Uncalled4 - Nanopolish Probability Difference')
ax.set_ylabel('Density')
ax.set_title('Prediction Score Difference Distribution')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig11_prediction_agreement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 11 saved: fig11_prediction_agreement.png")

# Save metrics JSON
metrics = {
    'n_sites': len(df),
    'n_positive': int(y_true.sum()),
    'n_negative': int((1-y_true).sum()),
    'label_proportion': float(y_true.mean()),
    'uncalled4': {
        'auprc': float(auprc_u4),
        'auroc': float(auroc_u4),
        'best_f1': float(best_f1_u4),
        'best_threshold': float(best_t_u4),
        'mean_prob': float(y_score_u4.mean()),
        'std_prob': float(y_score_u4.std()),
    },
    'nanopolish': {
        'auprc': float(auprc_np),
        'auroc': float(auroc_np),
        'best_f1': float(best_f1_np),
        'best_threshold': float(best_t_np),
        'mean_prob': float(y_score_np.mean()),
        'std_prob': float(y_score_np.std()),
    },
    'correlation': float(df['prob_uncalled4'].corr(df['prob_nanopolish'])),
}

with open('outputs/m6a_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# === Contingency tables at optimal thresholds ===
pred_u4_best = (y_score_u4 >= best_t_u4).astype(int)
pred_np_best = (y_score_np >= best_t_np).astype(int)

print(f"\nUncalled4 at threshold {best_t_u4:.2f}:")
print(f"  Accuracy: {accuracy_score(y_true, pred_u4_best):.4f}")
print(f"  Precision: {precision_score(y_true, pred_u4_best, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_true, pred_u4_best):.4f}")
print(f"  F1: {f1_score(y_true, pred_u4_best):.4f}")

print(f"\nNanopolish at threshold {best_t_np:.2f}:")
print(f"  Accuracy: {accuracy_score(y_true, pred_np_best):.4f}")
print(f"  Precision: {precision_score(y_true, pred_np_best, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_true, pred_np_best):.4f}")
print(f"  F1: {f1_score(y_true, pred_np_best):.4f}")

# Save PR points for later use
pr_data = {
    'uncalled4': {'precision': precision_u4.tolist(), 'recall': recall_u4.tolist()},
    'nanopolish': {'precision': precision_np.tolist(), 'recall': recall_np.tolist()},
}
with open('outputs/pr_curve_data.json', 'w') as f:
    json.dump(pr_data, f, indent=2)

print("\nm6A analysis complete.")
