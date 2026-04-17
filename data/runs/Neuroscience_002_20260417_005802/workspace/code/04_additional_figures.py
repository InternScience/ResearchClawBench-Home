"""
Create additional summary/validation figures for the report
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

IMG_DIR = 'report/images'
OUT_DIR = 'outputs'

# Load results
with open(os.path.join(OUT_DIR, 'model_results_all.json')) as f:
    all_results = json.load(f)

# ============================================================
# Figure: Comprehensive per-degradation F1 comparison
# ============================================================
model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'MLP']
deg_types = ['Average', 'Misalignment', 'Missing Sections', 'Mixed']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# F1 by degradation
x = np.arange(len(deg_types))
width = 0.12
for i, name in enumerate(model_names):
    vals = [all_results[name]['per_degradation'][deg]['f1_score'] for deg in deg_types]
    axes[0].bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[0].set_xlabel('Degradation Type', fontsize=12)
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('F1 Score by Degradation Type', fontsize=14)
axes[0].set_xticks(x + width * 2.5)
axes[0].set_xticklabels(deg_types, fontsize=10)
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, 1.05)

# AUC-ROC by degradation
for i, name in enumerate(model_names):
    vals = [all_results[name]['per_degradation'][deg]['auc_roc'] for deg in deg_types]
    axes[1].bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[1].set_xlabel('Degradation Type', fontsize=12)
axes[1].set_ylabel('AUC-ROC', fontsize=12)
axes[1].set_title('AUC-ROC by Degradation Type', fontsize=14)
axes[1].set_xticks(x + width * 2.5)
axes[1].set_xticklabels(deg_types, fontsize=10)
axes[1].legend(fontsize=8, loc='lower right')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0.9, 1.005)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'degradation_performance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved degradation_performance_comparison.png")

# ============================================================
# Figure: Radar chart for best model (MLP)
# ============================================================
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 7))
metrics = ['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall']
metric_keys = ['accuracy', 'f1_score', 'auc_roc', 'auc_pr', 'precision', 'recall']

# Create grouped bar chart of all models
x = np.arange(len(model_names))
width = 0.12
for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
    vals = [all_results[name][key] for name in model_names]
    ax.bar(x + i * width, vals, width, label=metric, alpha=0.85)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('All Models - All Metrics Overview', fontsize=14)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(model_names, fontsize=9, rotation=15)
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'all_models_all_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved all_models_all_metrics.png")

# ============================================================
# Figure: Training time vs performance trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
for i, name in enumerate(model_names):
    r = all_results[name]
    ax.scatter(r['train_time'], r['f1_score'], s=200, color=colors[i], 
               label=name, zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(name, (r['train_time'], r['f1_score']), 
                textcoords="offset points", xytext=(10, 5), fontsize=9)

ax.set_xlabel('Training Time (seconds)', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Training Time vs. F1 Score Trade-off', fontsize=14)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'time_vs_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved time_vs_performance.png")

# ============================================================
# Figure: Feature group analysis
# ============================================================
# Features 0-9 seem to be one group (higher correlation with label)
# Features 10-19 seem to be another group
train = pd.read_csv('data/train_simulated.csv')
feature_cols = [str(i) for i in range(20)]

# Feature groups based on correlation patterns
group1 = [str(i) for i in range(10)]  # Features 0-9 (higher correlation)
group2 = [str(i) for i in range(10, 20)]  # Features 10-19 (lower correlation)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Group 1 mean by label
g1_means = train.groupby('label')[group1].mean()
g1_means.T.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
axes[0].set_title('Features 0-9 (Group 1) Mean by Label', fontsize=12)
axes[0].set_xlabel('Feature')
axes[0].set_ylabel('Mean Value')
axes[0].legend(['No merge (0)', 'Merge (1)'])
axes[0].tick_params(axis='x', rotation=0)

# Group 2 mean by label
g2_means = train.groupby('label')[group2].mean()
g2_means.T.plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
axes[1].set_title('Features 10-19 (Group 2) Mean by Label', fontsize=12)
axes[1].set_xlabel('Feature')
axes[1].set_ylabel('Mean Value')
axes[1].legend(['No merge (0)', 'Merge (1)'])
axes[1].tick_params(axis='x', rotation=0)

# Effect size (Cohen's d) per feature
from scipy import stats
effect_sizes = []
for col in feature_cols:
    g0 = train[train['label'] == 0][col]
    g1 = train[train['label'] == 1][col]
    pooled_std = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
    d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else 0
    effect_sizes.append(d)

axes[2].bar(range(20), effect_sizes, color=['steelblue' if i < 10 else 'coral' for i in range(20)])
axes[2].set_xlabel('Feature Index')
axes[2].set_ylabel("Cohen's d")
axes[2].set_title("Effect Size (Cohen's d) per Feature", fontsize=12)
axes[2].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
axes[2].axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium effect')
axes[2].legend(fontsize=8)
axes[2].set_xticks(range(20))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'feature_group_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_group_analysis.png")

print("\n=== Additional figures complete ===")
