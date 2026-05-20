"""Generate figures for the DIDS-MFL report."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load results
with open('outputs/results.json', 'r') as f:
    results = json.load(f)

preds = np.load('outputs/test_predictions.npz')

# Color palette
colors = sns.color_palette('tab10')

# =============================================================================
# Figure 1: Dataset Distribution
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Class distribution
class_counts = [380, 341, 114716, 3666, 1473, 1009, 14688, 10910, 1427, 164]
class_names = [f'C{i}' for i in range(10)]
axes[0].bar(class_names, class_counts, color=colors[:10])
axes[0].set_yscale('log')
axes[0].set_ylabel('Count (log scale)')
axes[0].set_xlabel('Class')
axes[0].set_title('Class Distribution in NF-UNSW-NB15-v2')
axes[0].tick_params(axis='x', rotation=45)

# Binary distribution
bin_counts = [114716, 34058]
axes[1].bar(['Benign', 'Attack'], bin_counts, color=['#2ecc71', '#e74c3c'])
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Label')
axes[1].set_title('Binary Label Distribution')
for i, v in enumerate(bin_counts):
    axes[1].text(i, v + 1000, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('report/images/fig1_data_distribution.png', bbox_inches='tight')
plt.close()
print("Saved fig1_data_distribution.png")

# =============================================================================
# Figure 2: Training History
# =============================================================================
history = results['standard']['history']
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train Loss', color=colors[0])
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

val_epochs = np.arange(0, len(history['train_loss']), 5)
if len(val_epochs) > len(history['val_bin_f1']):
    val_epochs = val_epochs[:len(history['val_bin_f1'])]
val_bin = history['val_bin_f1'][:len(val_epochs)]
val_multi = history['val_multi_f1'][:len(val_epochs)]
axes[1].plot(val_epochs[:len(val_bin)], val_bin, marker='o', label='Binary F1', color=colors[1])
axes[1].plot(val_epochs[:len(val_multi)], val_multi, marker='s', label='Multi F1', color=colors[2])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Validation F1 Scores')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_training_history.png', bbox_inches='tight')
plt.close()
print("Saved fig2_training_history.png")

# =============================================================================
# Figure 3: Confusion Matrices
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary CM
y_bin = preds['y_bin']
preds_bin = preds['preds_bin']
cm_bin = confusion_matrix(y_bin, preds_bin)
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[0].set_title('Binary Classification Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Multi-class CM
y_multi = preds['y_multi']
preds_multi = preds['preds_multi']
cm_multi = confusion_matrix(y_multi, preds_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            xticklabels=[f'C{i}' for i in range(10)],
            yticklabels=[f'C{i}' for i in range(10)])
axes[1].set_title('Multi-class Classification Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('report/images/fig3_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Saved fig3_confusion_matrices.png")

# =============================================================================
# Figure 4: ROC and PR Curves
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

probs_bin = preds['probs_bin']
fpr, tpr, _ = roc_curve(y_bin, probs_bin)
axes[0].plot(fpr, tpr, color=colors[0], lw=2, label=f'ROC curve (AUC = {results["standard"]["test_binary"]["auc"]:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (Binary)')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

precision, recall, _ = precision_recall_curve(y_bin, probs_bin)
axes[1].plot(recall, precision, color=colors[1], lw=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve (Binary)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig4_roc_pr_curves.png', bbox_inches='tight')
plt.close()
print("Saved fig4_roc_pr_curves.png")

# =============================================================================
# Figure 5: Comparison with Baselines
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

methods = ['DIDS-MFL', 'Random Forest', 'Logistic Regression']
bin_f1 = [
    results['standard']['test_binary']['f1'],
    results['baselines']['RF_binary']['f1'],
    results['baselines']['LR_binary']['f1']
]
bin_auc = [
    results['standard']['test_binary']['auc'],
    results['baselines']['RF_binary']['auc'],
    results['baselines']['LR_binary']['auc']
]
multi_f1 = [
    results['standard']['test_multi']['f1'],
    results['baselines']['RF_multi']['f1'],
    0  # LR multi not computed
]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, bin_f1, width, label='Binary F1', color=colors[0])
bars2 = ax.bar(x, bin_auc, width, label='Binary AUC', color=colors[1])
bars3 = ax.bar(x + width, multi_f1, width, label='Multi F1 (weighted)', color=colors[2])

ax.set_ylabel('Score')
ax.set_title('Performance Comparison: DIDS-MFL vs Baselines')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('report/images/fig5_baseline_comparison.png', bbox_inches='tight')
plt.close()
print("Saved fig5_baseline_comparison.png")

# =============================================================================
# Figure 6: Unknown Attack Detection
# =============================================================================
unknown = results['unknown']
if unknown:
    attack_ids = sorted([int(k.split('_')[1]) for k in unknown.keys()])
    bin_f1s = [unknown.get(f'attack_{aid}', {}).get('binary', {}).get('f1', 0) for aid in attack_ids]
    multi_f1s = [unknown.get(f'attack_{aid}', {}).get('multi', {}).get('f1', 0) for aid in attack_ids]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(attack_ids))
    width = 0.35
    ax.bar(x - width/2, bin_f1s, width, label='Binary F1', color=colors[3])
    ax.bar(x + width/2, multi_f1s, width, label='Multi F1', color=colors[4])
    ax.set_xlabel('Left-out Attack Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Unknown Attack Detection (Leave-One-Out)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {aid}' for aid in attack_ids])
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/fig6_unknown_attacks.png', bbox_inches='tight')
    plt.close()
    print("Saved fig6_unknown_attacks.png")

# =============================================================================
# Figure 7: Few-Shot Learning
# =============================================================================
fewshot = results['fewshot']
shot_labels = ['1-shot', '5-shot', '10-shot', '20-shot']
shots = [fewshot.get(s, {}) for s in shot_labels]

bin_f1s = [s.get('binary', {}).get('f1', 0) for s in shots]
bin_aucs = [s.get('binary', {}).get('auc', 0) for s in shots]
multi_f1s = [s.get('multi', {}).get('f1', 0) for s in shots]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(shot_labels))
width = 0.25

ax.plot(x, bin_f1s, marker='o', label='Binary F1', color=colors[0], linewidth=2)
ax.plot(x, bin_aucs, marker='s', label='Binary AUC', color=colors[1], linewidth=2)
ax.plot(x, multi_f1s, marker='^', label='Multi F1', color=colors[2], linewidth=2)

ax.set_xlabel('Number of Shots per Class')
ax.set_ylabel('Score')
ax.set_title('Few-Shot Learning Performance')
ax.set_xticks(x)
ax.set_xticklabels(shot_labels)
ax.legend()
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig7_few_shot.png', bbox_inches='tight')
plt.close()
print("Saved fig7_few_shot.png")

# =============================================================================
# Figure 8: Per-Class F1 Comparison (Standard)
# =============================================================================
report = results['standard']['classification_report']
class_f1s = [report.get(f'Class_{i}', {}).get('f1-score', 0) for i in range(10)]
class_supports = [report.get(f'Class_{i}', {}).get('support', 0) for i in range(10)]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar([f'C{i}' for i in range(10)], class_f1s, color=colors[:10])
ax.set_xlabel('Class')
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Score (Standard Evaluation)')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

for bar, sup in zip(bars, class_supports):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}\n(n={int(sup)})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('report/images/fig8_per_class_f1.png', bbox_inches='tight')
plt.close()
print("Saved fig8_per_class_f1.png")

print("\nAll figures generated successfully.")
