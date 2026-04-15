#!/usr/bin/env python3
"""Generate all figures for the SimBA behavior classification report."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, confusion_matrix,
    roc_curve, roc_auc_score
)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Load results
with open('outputs/classification_metrics.json') as f:
    metrics = json.load(f)

pred_df = pd.read_csv('outputs/test_predictions.csv')
fi_df = pd.read_csv('outputs/feature_importance_all.csv')

os_mkdir = __import__('os').makedirs
os_mkdir('report/images', exist_ok=True)

# ============================================================
# Figure 1: Data Overview - Label distributions
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
targ = pd.read_csv('data/Together_1_targets_inserted.csv', index_col=0)

for ax, col, color in zip(axes, ['Attack', 'Sniffing'], ['#e74c3c', '#3498db']):
    counts = targ[col].value_counts().sort_index()
    bars = ax.bar(['Absent (0)', 'Present (1)'], counts.values, color=['#bdc3c7', color], edgecolor='white')
    ax.set_title(f'{col} Label Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frame Count')
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 20, str(v), ha='center', fontweight='bold')
    total = counts.sum()
    ax.text(0.5, 0.9, f'Prevalence: {counts.get(1,0)/total*100:.1f}%', transform=ax.transAxes,
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Figure 1: Dataset Label Overview (Together_1, N=1738 frames)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig1_label_distribution.png', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Confusion Matrices
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
titles = ['Attack - Random Forest', 'Attack - Gradient Boosting',
          'Sniffing - Random Forest', 'Sniffing - Gradient Boosting']
keys = ['Attack_RF', 'Attack_GB', 'Sniffing_RF', 'Sniffing_GB']

for ax, title, key in zip(axes.flat, titles, keys):
    cm = np.array(metrics[key]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

fig.suptitle('Figure 2: Confusion Matrices (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig2_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Precision-Recall Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, behavior, y_true_col in zip(axes, ['Attack', 'Sniffing'], ['y_true_attack', 'y_true_sniffing']):
    y_true = pred_df[y_true_col].values
    for clf_name, prob_col, color, ls in [
        ('RF', f'prob_{behavior.lower()}_rf', '#e74c3c', '-'),
        ('GB', f'prob_{behavior.lower()}_gb', '#2980b9', '--')
    ]:
        y_prob = pred_df[prob_col].values
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, color=color, ls=ls, lw=2, label=f'{clf_name} (AP={ap:.3f})')
    
    baseline = y_true.mean()
    ax.axhline(baseline, color='gray', ls=':', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{behavior}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])

fig.suptitle('Figure 3: Precision-Recall Curves', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig3_precision_recall.png', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: ROC Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, behavior, y_true_col in zip(axes, ['Attack', 'Sniffing'], ['y_true_attack', 'y_true_sniffing']):
    y_true = pred_df[y_true_col].values
    for clf_name, prob_col, color, ls in [
        ('RF', f'prob_{behavior.lower()}_rf', '#e74c3c', '-'),
        ('GB', f'prob_{behavior.lower()}_gb', '#2980b9', '--')
    ]:
        y_prob = pred_df[prob_col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f'{clf_name} (AUC={auc:.3f})')
    
    ax.plot([0,1], [0,1], 'k:', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{behavior}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

fig.suptitle('Figure 4: ROC Curves', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig4_roc_curves.png', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Feature Importance (Top 15)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, behavior, imp_col in zip(axes, ['Attack', 'Sniffing'], ['importance_attack', 'importance_sniffing']):
    top = fi_df.nlargest(15, imp_col)
    ax.barh(range(len(top)), top[imp_col].values, color='#3498db', edgecolor='white')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (MDI)', fontsize=11)
    ax.set_title(f'{behavior} - Top 15 Features', fontsize=13, fontweight='bold')

fig.suptitle('Figure 5: Random Forest Feature Importance', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig5_feature_importance.png', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Model Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    rf_key = f'{behavior}_RF'
    gb_key = f'{behavior}_GB'
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AP', 'AUC-ROC']
    rf_vals = [metrics[rf_key][k] for k in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'auc_roc']]
    gb_vals = [metrics[gb_key][k] for k in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'auc_roc']]
    
    x = np.arange(len(metric_names))
    w = 0.35
    ax.bar(x - w/2, rf_vals, w, label='Random Forest', color='#e74c3c', edgecolor='white')
    ax.bar(x + w/2, gb_vals, w, label='Gradient Boosting', color='#2980b9', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{behavior}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylabel('Score')

fig.suptitle('Figure 6: Classifier Performance Comparison (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig6_model_comparison.png', bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================
# Figure 7: Probability Distribution Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, behavior, y_true_col in zip(axes, ['Attack', 'Sniffing'], ['y_true_attack', 'y_true_sniffing']):
    y_true = pred_df[y_true_col].values
    prob_col = f'prob_{behavior.lower()}_rf'
    y_prob = pred_df[prob_col].values
    
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, color='#bdc3c7', label='Negative', density=True)
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, color='#e74c3c', label='Positive', density=True)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{behavior} (RF)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

fig.suptitle('Figure 7: Predicted Probability Distributions by True Label', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig7_probability_distributions.png', bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

print("\nAll figures generated successfully!")
