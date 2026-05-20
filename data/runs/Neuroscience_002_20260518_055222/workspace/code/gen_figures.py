"""
Generate all remaining figures
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve, 
                              precision_recall_curve, average_precision_score, f1_score)
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
y_test = test_df['label'].values
deg_test = test_df['degradation'].values

# Load model results
with open('outputs/all_results.json') as f:
    all_data = json.load(f)

all_results = all_data['model_results']
best_name = all_data['best_model']
deg_results = all_data['degradation_results']
cv_results = all_data['cross_validation']
rf_imp = all_data['feature_importances']['rf']
gb_imp = all_data['feature_importances']['gb']

degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']

# Load probabilities
y_prob_lr = np.load('outputs/y_prob_lr.npy')
y_prob_rf = np.load('outputs/y_prob_rf.npy')
y_prob_gb = np.load('outputs/y_prob_gb.npy')
y_prob_ab = np.load('outputs/y_prob_ab.npy')
y_prob_mlp = np.load('outputs/y_prob_mlp.npy')

y_prob_map = {
    'Logistic Regression': y_prob_lr,
    'Random Forest': y_prob_rf,
    'Gradient Boosting': y_prob_gb,
    'AdaBoost': y_prob_ab,
    'MLP': y_prob_mlp
}

y_pred_rf = np.load('outputs/y_pred_rf.npy')
y_pred_gb = np.load('outputs/y_pred_gb.npy')
y_pred_mlp = np.load('outputs/y_pred_mlp.npy')
y_pred_ab = np.load('outputs/y_pred_ab.npy')
y_pred_lr = (y_prob_lr > 0.5).astype(int)
y_pred_map = {
    'Logistic Regression': y_pred_lr,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb,
    'AdaBoost': y_pred_ab,
    'MLP': y_pred_mlp
}

# ==========================================
# Figure 4: ROC and PR Curves
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for name in all_results:
    y_prob = y_prob_map[name]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={all_results[name]['auc']:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

ax = axes[1]
for name in all_results:
    y_prob = y_prob_map[name]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = all_results[name]['ap']
    ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", linewidth=2)
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

model_names = list(all_results.keys())
metrics_df = pd.DataFrame({
    'AUC-ROC': [all_results[n]['auc'] for n in model_names],
    'AP': [all_results[n]['ap'] for n in model_names],
    'F1': [all_results[n]['f1'] for n in model_names],
    'Precision': [all_results[n]['precision'] for n in model_names],
    'Recall': [all_results[n]['recall'] for n in model_names],
}, index=model_names)

ax = axes[0]
metrics_df[['AUC-ROC', 'AP', 'F1']].plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: AUC, AP, and F1', fontsize=13, fontweight='bold')
ax.set_ylim([0.4, 1.05])
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='lower right')

ax = axes[1]
metrics_df[['Precision', 'Recall']].plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: Precision and Recall', fontsize=13, fontweight='bold')
ax.set_ylim([0.4, 1.05])
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
top_models = sorted(all_results.keys(), key=lambda x: all_results[x]['auc'], reverse=True)[:3]

for idx, name in enumerate(top_models):
    ax = axes[idx]
    cm = confusion_matrix(y_test, y_pred_map[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
    ax.set_title(f'{name}\n(AUC={all_results[name]["auc"]:.4f})', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices - Top 3 Models', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('report/images/figure6_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: confusion matrices")

# ==========================================
# Figure 7: Degradation-specific performance
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use best model (MLP)
y_prob_best = y_prob_mlp

ax = axes[0]
deg_names = degradations
x_pos = np.arange(len(deg_names))
width = 0.25

auc_vals = [deg_results[d]['MLP']['auc'] for d in deg_names]
f1_vals = [deg_results[d]['MLP']['f1'] for d in deg_names]

ax.bar(x_pos - width/2, auc_vals, width, label='AUC-ROC', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, f1_vals, width, label='F1 Score', color='#2ecc71', alpha=0.8)
ax.set_xlabel('Degradation Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Performance by Degradation (MLP)', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(deg_names, rotation=15)
ax.legend()
ax.set_ylim([0.4, 1.05])
for i, (a, f) in enumerate(zip(auc_vals, f1_vals)):
    ax.text(i - width/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
    ax.text(i + width/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9)

# ROC curves by degradation
ax = axes[1]
colors_deg = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for idx, deg in enumerate(degradations):
    mask = deg_test == deg
    y_prob_d = y_prob_best[mask]
    y_d = y_test[mask]
    fpr, tpr, _ = roc_curve(y_d, y_prob_d)
    auc_val = deg_results[deg]['MLP']['auc']
    ax.plot(fpr, tpr, label=f'{deg} (AUC={auc_val:.4f})', 
            linewidth=2, color=colors_deg[idx])
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves by Degradation Type (MLP)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure7_degradation_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: degradation performance")

# ==========================================
# Figure 8: Feature importance
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_sorted_idx = np.argsort(rf_imp)[::-1]
ax = axes[0]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, 20))
ax.bar(range(20), np.array(rf_imp)[rf_sorted_idx], color=colors)
ax.set_xlabel('Feature Index (sorted)', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Random Forest: Feature Importance', fontsize=13, fontweight='bold')
ax.set_xticks(range(20))
ax.set_xticklabels([str(i) for i in rf_sorted_idx], fontsize=8)

gb_sorted_idx = np.argsort(gb_imp)[::-1]
ax = axes[1]
colors = plt.cm.magma(np.linspace(0.2, 0.9, 20))
ax.bar(range(20), np.array(gb_imp)[gb_sorted_idx], color=colors)
ax.set_xlabel('Feature Index (sorted)', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Gradient Boosting: Feature Importance', fontsize=13, fontweight='bold')
ax.set_xticks(range(20))
ax.set_xticklabels([str(i) for i in gb_sorted_idx], fontsize=8)

plt.tight_layout()
plt.savefig('report/images/figure8_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved: feature importance")

# ==========================================
# Figure 9: Cross-validation
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
cv_model_names = list(cv_results.keys())
cv_aucs = [cv_results[n]['auc_mean'] for n in cv_model_names]
cv_stds = [cv_results[n]['auc_std'] for n in cv_model_names]
test_aucs = [all_results[n]['auc'] for n in cv_model_names]

x_pos = np.arange(len(cv_model_names))
width = 0.35
ax.bar(x_pos - width/2, test_aucs, width, label='Test Set AUC', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, cv_aucs, width, yerr=cv_stds, label='CV AUC (mean±std)', 
       color='#e67e22', alpha=0.8, capsize=5)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Test vs Cross-Validation Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in cv_model_names])
ax.legend()
ax.set_ylim([0.85, 1.0])

# Degradation heatmap
ax = axes[1]
deg_model_data = []
for deg in degradations:
    row = [deg_results[deg][n]['auc'] for n in ['Logistic Regression', 'Random Forest', 
                                                  'Gradient Boosting', 'AdaBoost', 'MLP']]
    deg_model_data.append(row)

deg_model_df = pd.DataFrame(deg_model_data, 
                           columns=['LR', 'RF', 'GB', 'AB', 'MLP'],
                           index=degradations)
sns.heatmap(deg_model_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
            vmin=0.7, vmax=1.0, linewidths=0.5)
ax.set_title('AUC-ROC: Degradation × Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Degradation Type')
ax.set_xlabel('Model')

plt.tight_layout()
plt.savefig('report/images/figure9_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9 saved: cross-validation heatmap")

# ==========================================
# Figure 10: Additional analysis - Score distributions
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Score distribution for MLP
ax = axes[0]
y_prob_best = y_prob_mlp
ax.hist(y_prob_best[y_test == 0], bins=50, alpha=0.6, label='Different (0)', density=True, color='#e74c3c')
ax.hist(y_prob_best[y_test == 1], bins=50, alpha=0.6, label='Same (1)', density=True, color='#2ecc71')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('MLP: Score Distribution by True Label', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold=0.5')

# Per-degradation score distribution
ax = axes[1]
colors_deg = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for idx, deg in enumerate(degradations):
    mask = deg_test == deg
    y_prob_d = y_prob_best[mask]
    y_d = y_test[mask]
    # Plot only same-neuron scores
    same_scores = y_prob_d[y_d == 1]
    ax.hist(same_scores, bins=30, alpha=0.5, label=f'{deg} (n={len(same_scores)})', 
            density=True, color=colors_deg[idx])
ax.set_xlabel('Predicted Probability (Same Neuron)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('MLP: Score Distribution for True Same-Neuron Pairs', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure10_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10 saved: score distributions")

# Save final summary
summary = {
    'model_results': all_results,
    'best_model': best_name,
    'degradation_results': deg_results,
    'feature_importances': {'rf': rf_imp, 'gb': gb_imp},
    'cross_validation': cv_results,
    'test_accuracy': float(all_results[best_name]['accuracy'])
}
with open('outputs/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ All figures generated!")
print(f"Best model: {best_name} (AUC={all_results[best_name]['auc']:.4f})")
