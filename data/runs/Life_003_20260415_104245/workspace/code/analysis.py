#!/usr/bin/env python3
"""
Uncalled4 Research Analysis Pipeline
=====================================
Comprehensive analysis of nanopore signal alignment performance and m6A modification detection.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    average_precision_score, roc_auc_score
)

# ─── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ─── Style ──────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

###############################################################################
# 1. PORE MODEL ANALYSIS
###############################################################################
print("=" * 60)
print("1. PORE MODEL ANALYSIS")
print("=" * 60)

pore_models = {
    'DNA r9.4.1 (6-mer)': f'{DATA_DIR}/dna_r9.4.1_400bps_6mer_uncalled4.csv',
    'DNA r10.4.1 (9-mer)': f'{DATA_DIR}/dna_r10.4.1_400bps_9mer_uncalled4.csv',
    'RNA001 (5-mer)': f'{DATA_DIR}/rna_r9.4.1_70bps_5mer_uncalled4.csv',
    'RNA004 (9-mer)': f'{DATA_DIR}/rna004_130bps_9mer_uncalled4.csv',
}

pore_dfs = {}
for name, path in pore_models.items():
    df = pd.read_csv(path)
    pore_dfs[name] = df
    print(f"\n{name}:")
    print(f"  k-mers: {len(df)}")
    print(f"  current_mean: {df['current_mean'].mean():.3f} ± {df['current_mean'].std():.3f}")
    print(f"  current_std: {df['current_std'].mean():.3f} ± {df['current_std'].std():.3f}")
    print(f"  dwell_time: {df['dwell_time'].mean():.1f} ± {df['dwell_time'].std():.1f}")

# Save pore model statistics
pore_stats = {}
for name, df in pore_dfs.items():
    pore_stats[name] = {
        'n_kmers': len(df),
        'current_mean_mean': float(df['current_mean'].mean()),
        'current_mean_std': float(df['current_mean'].std()),
        'current_std_mean': float(df['current_std'].mean()),
        'current_std_std': float(df['current_std'].std()),
        'dwell_time_mean': float(df['dwell_time'].mean()),
        'dwell_time_std': float(df['dwell_time'].std()),
    }

with open(f'{OUTPUTS_DIR}/pore_model_stats.json', 'w') as f:
    json.dump(pore_stats, f, indent=2)

# Figure 1: Current distribution by chemistry
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, df) in enumerate(pore_dfs.items()):
    ax = axes[idx]
    ax.hist(df['current_mean'], bins=50, alpha=0.7, edgecolor='white', color='#2196F3')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Mean Current (pA)')
    ax.set_ylabel('Count')
    ax.axvline(df['current_mean'].mean(), color='red', linestyle='--', 
               label=f'Mean={df["current_mean"].mean():.2f}')
    ax.legend(fontsize=9)

plt.suptitle('Nanopore K-mer Current Distributions by Chemistry', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig1_current_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {IMAGES_DIR}/fig1_current_distributions.png")

# Figure 2: Comparative statistics across chemistries
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Current mean comparison
labels = list(pore_dfs.keys())
means = [pore_dfs[l]['current_mean'].mean() for l in labels]
stds = [pore_dfs[l]['current_mean'].std() for l in labels]
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
bars = axes[0].bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='white', alpha=0.85)
axes[0].set_title('Mean Current by Chemistry', fontsize=12)
axes[0].set_ylabel('Mean Current (pA)')
axes[0].tick_params(axis='x', rotation=15)

# Current std comparison
std_means = [pore_dfs[l]['current_std'].mean() for l in labels]
std_stds = [pore_dfs[l]['current_std'].std() for l in labels]
axes[1].bar(labels, std_means, yerr=std_stds, capsize=5, color=colors, edgecolor='white', alpha=0.85)
axes[1].set_title('Mean Current Std by Chemistry', fontsize=12)
axes[1].set_ylabel('Current Std (pA)')
axes[1].tick_params(axis='x', rotation=15)

# Dwell time comparison
dt_means = [pore_dfs[l]['dwell_time'].mean() for l in labels]
dt_stds = [pore_dfs[l]['dwell_time'].std() for l in labels]
axes[2].bar(labels, dt_means, yerr=dt_stds, capsize=5, color=colors, edgecolor='white', alpha=0.85)
axes[2].set_title('Mean Dwell Time by Chemistry', fontsize=12)
axes[2].set_ylabel('Dwell Time (ms)')
axes[2].tick_params(axis='x', rotation=15)

plt.suptitle('Pore Model Statistics Comparison Across Sequencing Chemistries', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig2_pore_statistics_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig2_pore_statistics_comparison.png")

# Figure 3: DNA vs RNA current distributions overlay
fig, ax = plt.subplots(figsize=(8, 5))
for name, df in pore_dfs.items():
    if 'DNA' in name:
        sns.kdeplot(df['current_mean'], ax=ax, label=name, linewidth=2)
ax.set_title('DNA Pore Current Distributions', fontsize=12)
ax.set_xlabel('Mean Current (pA)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig3_dna_current_overlay.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig3_dna_current_overlay.png")

fig, ax = plt.subplots(figsize=(8, 5))
for name, df in pore_dfs.items():
    if 'RNA' in name:
        sns.kdeplot(df['current_mean'], ax=ax, label=name, linewidth=2)
ax.set_title('RNA Pore Current Distributions', fontsize=12)
ax.set_xlabel('Mean Current (pA)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig4_rna_current_overlay.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig4_rna_current_overlay.png")

###############################################################################
# 2. PERFORMANCE BENCHMARK REPRODUCTION
###############################################################################
print("\n" + "=" * 60)
print("2. PERFORMANCE BENCHMARK ANALYSIS")
print("=" * 60)

perf_df = pd.read_csv(f'{DATA_DIR}/performance_summary.csv')
print(perf_df.to_string(index=False))

# Save performance data
perf_df.to_csv(f'{OUTPUTS_DIR}/performance_summary_processed.csv', index=False)

# Figure 5: Speed comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

chemistries = perf_df['Chemistry'].unique()
tools = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']
colors_map = {'Uncalled4': '#2196F3', 'f5c': '#FF9800', 'Nanopolish': '#4CAF50', 'Tombo': '#E91E63'}

x = np.arange(len(chemistries))
width = 0.2

for i, tool in enumerate(tools):
    tool_data = perf_df[perf_df['Tool'] == tool]
    times = []
    for chem in chemistries:
        val = tool_data[tool_data['Chemistry'] == chem]['Time_min'].values
        if len(val) > 0 and not np.isnan(val[0]):
            times.append(val[0])
        else:
            times.append(np.nan)
    
    bars = ax.bar(x + i * width, times, width, label=tool, 
                  color=colors_map[tool], edgecolor='white', alpha=0.85)
    # Add value labels on bars
    for j, (xi, t) in enumerate(zip(x + i * width, times)):
        if not np.isnan(t):
            if t >= 100:
                ax.text(xi, t + 30, f'{t:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)
            elif t >= 10:
                ax.text(xi, t + 15, f'{t:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)
            else:
                ax.text(xi, t + 5, f'{t:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chemistries)
ax.set_ylabel('Alignment Time (minutes)', fontsize=11)
ax.set_title('Uncalled4 vs Competing Tools: Alignment Speed\n(Lower is Better)', fontsize=13)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.set_ylim(bottom=10, top=5000)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig5_speed_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig5_speed_comparison.png")

# Figure 6: File size comparison
fig, ax = plt.subplots(figsize=(10, 6))

for i, tool in enumerate(tools):
    tool_data = perf_df[perf_df['Tool'] == tool]
    sizes = []
    for chem in chemistries:
        val = tool_data[tool_data['Chemistry'] == chem]['FileSize_MB'].values
        if len(val) > 0 and not np.isnan(val[0]):
            sizes.append(val[0])
        else:
            sizes.append(np.nan)
    
    ax.bar(x + i * width, sizes, width, label=tool, 
           color=colors_map[tool], edgecolor='white', alpha=0.85)
    for j, (xi, s) in enumerate(zip(x + i * width, sizes)):
        if not np.isnan(s):
            if s >= 1000:
                ax.text(xi, s + 50, f'{s/1024:.1f}GB', ha='center', va='bottom', fontsize=7, rotation=90)
            else:
                ax.text(xi, s + 30, f'{s:.0f}MB', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chemistries)
ax.set_ylabel('Output File Size (MB)', fontsize=11)
ax.set_title('Uncalled4 vs Competing Tools: Output File Size\n(Lower is Better)', fontsize=13)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.set_ylim(bottom=10, top=5000)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig6_filesize_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig6_filesize_comparison.png")

# Speedup calculations
print("\nSpeedup of Uncalled4 over other tools:")
for chem in chemistries:
    u4_time = perf_df[(perf_df['Chemistry'] == chem) & (perf_df['Tool'] == 'Uncalled4')]['Time_min'].values[0]
    for tool in ['f5c', 'Nanopolish', 'Tombo']:
        t_time_row = perf_df[(perf_df['Chemistry'] == chem) & (perf_df['Tool'] == tool)]
        if len(t_time_row) > 0:
            t_time = t_time_row['Time_min'].values[0]
            if not np.isnan(t_time):
                speedup = t_time / u4_time
                print(f"  {chem}: Uncalled4 is {speedup:.1f}x faster than {tool}")

# File size reduction
print("\nFile size reduction of Uncalled4:")
for chem in chemistries:
    u4_size = perf_df[(perf_df['Chemistry'] == chem) & (perf_df['Tool'] == 'Uncalled4')]['FileSize_MB'].values[0]
    for tool in ['f5c', 'Nanopolish', 'Tombo']:
        t_size_row = perf_df[(perf_df['Chemistry'] == chem) & (perf_df['Tool'] == tool)]
        if len(t_size_row) > 0:
            t_size = t_size_row['FileSize_MB'].values[0]
            if not np.isnan(t_size):
                reduction = t_size / u4_size
                print(f"  {chem}: Uncalled4 uses {reduction:.1f}x less storage than {tool}")

###############################################################################
# 3. M6A DETECTION EVALUATION
###############################################################################
print("\n" + "=" * 60)
print("3. M6A DETECTION EVALUATION")
print("=" * 60)

# Load data
labels_df = pd.read_csv(f'{DATA_DIR}/m6a_labels.csv')
uncalled4_pred = pd.read_csv(f'{DATA_DIR}/m6a_predictions_uncalled4.csv')
nanopolish_pred = pd.read_csv(f'{DATA_DIR}/m6a_predictions_nanopolish.csv')

# Merge
merged = labels_df.merge(uncalled4_pred, on='site_id', suffixes=('', '_uncalled4'))
merged = merged.merge(nanopolish_pred, on='site_id', suffixes=('', '_nanopolish'))
merged.columns = ['site_id', 'label', 'uncalled4_prob', 'nanopolish_prob']

print(f"Total sites: {len(merged)}")
print(f"Positive sites (m6A): {merged['label'].sum()} ({merged['label'].mean()*100:.1f}%)")
print(f"Negative sites: {(merged['label']==0).sum()} ({(1-merged['label'].mean())*100:.1f}%)")

y_true = merged['label'].values
y_uncalled4 = merged['uncalled4_prob'].values
y_nanopolish = merged['nanopolish_prob'].values

# Precision-Recall curves
pr_uncalled4 = precision_recall_curve(y_true, y_uncalled4)
pr_nanopolish = precision_recall_curve(y_true, y_nanopolish)

ap_uncalled4 = average_precision_score(y_true, y_uncalled4)
ap_nanopolish = average_precision_score(y_true, y_nanopolish)

print(f"\nAverage Precision (AP):")
print(f"  Uncalled4: {ap_uncalled4:.4f}")
print(f"  Nanopolish: {ap_nanopolish:.4f}")

# ROC curves
fpr_uncalled4, tpr_uncalled4, _ = roc_curve(y_true, y_uncalled4)
fpr_nanopolish, tpr_nanopolish, _ = roc_curve(y_true, y_nanopolish)

roc_auc_uncalled4 = roc_auc_score(y_true, y_uncalled4)
roc_auc_nanopolish = roc_auc_score(y_true, y_nanopolish)

print(f"\nROC AUC:")
print(f"  Uncalled4: {roc_auc_uncalled4:.4f}")
print(f"  Nanopolish: {roc_auc_nanopolish:.4f}")

# Save metrics
metrics = {
    'uncalled4': {
        'average_precision': float(ap_uncalled4),
        'roc_auc': float(roc_auc_uncalled4),
    },
    'nanopolish': {
        'average_precision': float(ap_nanopolish),
        'roc_auc': float(roc_auc_nanopolish),
    }
}
with open(f'{OUTPUTS_DIR}/m6a_detection_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save PR curve data
pr_data = {
    'uncalled4': {
        'precision': pr_uncalled4[0].tolist(),
        'recall': pr_uncalled4[1].tolist(),
    },
    'nanopolish': {
        'precision': pr_nanopolish[0].tolist(),
        'recall': pr_nanopolish[1].tolist(),
    }
}
with open(f'{OUTPUTS_DIR}/pr_curve_data.json', 'w') as f:
    json.dump(pr_data, f, indent=2)

# Save ROC curve data
roc_data = {
    'uncalled4': {
        'fpr': fpr_uncalled4.tolist(),
        'tpr': tpr_uncalled4.tolist(),
    },
    'nanopolish': {
        'fpr': fpr_nanopolish.tolist(),
        'tpr': tpr_nanopolish.tolist(),
    }
}
with open(f'{OUTPUTS_DIR}/roc_curve_data.json', 'w') as f:
    json.dump(roc_data, f, indent=2)

# Figure 7: Precision-Recall Curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pr_uncalled4[1], pr_uncalled4[0], linewidth=2.5, 
        label=f'Uncalled4 (AP={ap_uncalled4:.3f})', color='#2196F3')
ax.plot(pr_nanopolish[1], pr_nanopolish[0], linewidth=2.5,
        label=f'Nanopolish (AP={ap_nanopolish:.3f})', color='#4CAF50')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('m6A Detection: Precision-Recall Curves', fontsize=14)
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig7_pr_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {IMAGES_DIR}/fig7_pr_curves.png")

# Figure 8: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_uncalled4, tpr_uncalled4, linewidth=2.5,
        label=f'Uncalled4 (AUC={roc_auc_uncalled4:.3f})', color='#2196F3')
ax.plot(fpr_nanopolish, tpr_nanopolish, linewidth=2.5,
        label=f'Nanopolish (AUC={roc_auc_nanopolish:.3f})', color='#4CAF50')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('m6A Detection: ROC Curves', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig8_roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig8_roc_curves.png")

# Figure 9: Prediction probability distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Uncalled4
ax = axes[0]
pos_mask = y_true == 1
neg_mask = y_true == 0
ax.hist(y_uncalled4[neg_mask], bins=50, alpha=0.6, label='Negative (no m6A)', 
        color='#B0BEC5', edgecolor='white', density=True)
ax.hist(y_uncalled4[pos_mask], bins=50, alpha=0.6, label='Positive (m6A)',
        color='#2196F3', edgecolor='white', density=True)
ax.set_title('Uncalled4 Prediction Probabilities', fontsize=12)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.legend()
ax.set_xlim([0, 1.05])

# Nanopolish
ax = axes[1]
ax.hist(y_nanopolish[neg_mask], bins=50, alpha=0.6, label='Negative (no m6A)',
        color='#B0BEC5', edgecolor='white', density=True)
ax.hist(y_nanopolish[pos_mask], bins=50, alpha=0.6, label='Positive (m6A)',
        color='#4CAF50', edgecolor='white', density=True)
ax.set_title('Nanopolish Prediction Probabilities', fontsize=12)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.legend()
ax.set_xlim([0, 1.05])

plt.suptitle('m6A Prediction Probability Distributions by Tool', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig9_prediction_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig9_prediction_distributions.png")

# Figure 10: Box plot comparison
fig, ax = plt.subplots(figsize=(8, 5))
box_data = [
    y_uncalled4[neg_mask], y_uncalled4[pos_mask],
    y_nanopolish[neg_mask], y_nanopolish[pos_mask]
]
labels_box = ['Uncalled4\nNegative', 'Uncalled4\nPositive', 
              'Nanopolish\nNegative', 'Nanopolish\nPositive']
bp = ax.boxplot(box_data, labels=labels_box, patch_artist=True, widths=0.5)
colors_box = ['#B0BEC5', '#2196F3', '#B0BEC5', '#4CAF50']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Predicted Probability', fontsize=11)
ax.set_title('Prediction Probability by Tool and True Label', fontsize=13)
ax.set_ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig10_boxplot_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig10_boxplot_comparison.png")

# Figure 11: Threshold analysis
fig, ax = plt.subplots(figsize=(8, 5))
thresholds = np.linspace(0, 1, 100)
for tool_name, y_pred, color in [('Uncalled4', y_uncalled4, '#2196F3'), 
                                   ('Nanopolish', y_nanopolish, '#4CAF50')]:
    precisions = []
    recalls = []
    fprs = []
    for t in thresholds:
        y_pred_t = (y_pred >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_true == 1)).sum()
        fp = ((y_pred_t == 1) & (y_true == 0)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        tn = ((y_pred_t == 0) & (y_true == 0)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        fprs.append(fpr_val)
    
    ax.plot(thresholds, precisions, label=f'{tool_name} Precision', 
            color=color, linestyle='-', linewidth=2)
    ax.plot(thresholds, recalls, label=f'{tool_name} Recall',
            color=color, linestyle='--', linewidth=2)

ax.set_xlabel('Classification Threshold', fontsize=11)
ax.set_ylabel('Metric Value', fontsize=11)
ax.set_title('Precision and Recall vs Classification Threshold', fontsize=13)
ax.legend(loc='best', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig11_threshold_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig11_threshold_analysis.png")

# Find optimal thresholds
print("\nOptimal threshold analysis:")
for tool_name, y_pred in [('Uncalled4', y_uncalled4), ('Nanopolish', y_nanopolish)]:
    best_f1 = 0
    best_thresh = 0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred_t = (y_pred >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_true == 1)).sum()
        fp = ((y_pred_t == 1) & (y_true == 0)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    y_pred_opt = (y_pred >= best_thresh).astype(int)
    tp = ((y_pred_opt == 1) & (y_true == 1)).sum()
    fp = ((y_pred_opt == 1) & (y_true == 0)).sum()
    fn = ((y_pred_opt == 0) & (y_true == 1)).sum()
    tn = ((y_pred_opt == 0) & (y_true == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = (tp + tn) / len(y_true)
    
    print(f"  {tool_name}: threshold={best_thresh:.2f}, F1={best_f1:.4f}, "
          f"Precision={prec:.4f}, Recall={rec:.4f}, Accuracy={acc:.4f}")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Save optimal threshold results
opt_results = {}
for tool_name, y_pred in [('Uncalled4', y_uncalled4), ('Nanopolish', y_nanopolish)]:
    best_f1 = 0
    best_thresh = 0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred_t = (y_pred >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_true == 1)).sum()
        fp = ((y_pred_t == 1) & (y_true == 0)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    y_pred_opt = (y_pred >= best_thresh).astype(int)
    tp = int(((y_pred_opt == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_opt == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_opt == 0) & (y_true == 1)).sum())
    tn = int(((y_pred_opt == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = (tp + tn) / len(y_true)
    
    opt_results[tool_name] = {
        'threshold': float(best_thresh),
        'f1': float(best_f1),
        'precision': float(prec),
        'recall': float(rec),
        'accuracy': float(acc),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }

with open(f'{OUTPUTS_DIR}/optimal_thresholds.json', 'w') as f:
    json.dump(opt_results, f, indent=2)

###############################################################################
# 4. K-MER SUBSTITUTION ANALYSIS
###############################################################################
print("\n" + "=" * 60)
print("4. K-MER SUBSTITUTION ANALYSIS")
print("=" * 60)

# Analyze current variation by k-mer composition
for name, df in pore_dfs.items():
    df_copy = df.copy()
    df_copy['kmer_length'] = df_copy['kmer'].str.len()
    df_copy['gc_content'] = df_copy['kmer'].str.count('[GC]') / df_copy['kmer_length']
    
    # Correlation between GC content and current
    corr = df_copy['gc_content'].corr(df_copy['current_mean'])
    print(f"\n{name}:")
    print(f"  GC content vs current mean correlation: {corr:.4f}")
    
    # Save per-kmer analysis
    df_copy.to_csv(f'{OUTPUTS_DIR}/{name.replace(" ", "_").replace("(", "").replace(")", "")}_analysis.csv', index=False)

# Figure 12: GC content vs current mean
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, df) in enumerate(pore_dfs.items()):
    ax = axes[idx]
    df_copy = df.copy()
    df_copy['gc_content'] = df_copy['kmer'].str.count('[GC]') / df_copy['kmer'].str.len()
    
    scatter = ax.scatter(df_copy['gc_content'], df_copy['current_mean'], 
                         c=df_copy['current_std'], cmap='viridis', 
                         alpha=0.6, s=10, edgecolors='none')
    corr = df_copy['gc_content'].corr(df_copy['current_mean'])
    ax.set_title(f'{name}\n(r={corr:.3f})', fontsize=10)
    ax.set_xlabel('GC Content')
    ax.set_ylabel('Mean Current (pA)')
    plt.colorbar(scatter, ax=ax, label='Current Std')

plt.suptitle('GC Content vs Mean Current by Chemistry', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/fig12_gc_vs_current.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/fig12_gc_vs_current.png")

###############################################################################
# 5. SUMMARY STATISTICS TABLE
###############################################################################
print("\n" + "=" * 60)
print("5. SUMMARY TABLE GENERATION")
print("=" * 60)

# Create Table 1: Performance comparison
table1_data = []
for chem in chemistries:
    for tool in tools:
        row = perf_df[(perf_df['Chemistry'] == chem) & (perf_df['Tool'] == tool)]
        if len(row) > 0:
            time_val = row['Time_min'].values[0]
            size_val = row['FileSize_MB'].values[0]
            table1_data.append({
                'Chemistry': chem,
                'Tool': tool,
                'Time (min)': f"{time_val:.1f}" if not np.isnan(time_val) else "N/A",
                'File Size (MB)': f"{size_val:.1f}" if not np.isnan(size_val) else "N/A",
            })

table1_df = pd.DataFrame(table1_data)
table1_df.to_csv(f'{OUTPUTS_DIR}/table1_performance.csv', index=False)
print("Table 1 saved to outputs/table1_performance.csv")

# Create Table 2: m6A detection metrics
table2_data = []
for tool_name, m in opt_results.items():
    table2_data.append({
        'Tool': tool_name,
        'ROC AUC': f"{metrics[tool_name.lower().replace(' ', '')]['roc_auc']:.4f}",
        'Average Precision': f"{metrics[tool_name.lower().replace(' ', '')]['average_precision']:.4f}",
        'Optimal Threshold': f"{m['threshold']:.2f}",
        'F1 Score': f"{m['f1']:.4f}",
        'Precision': f"{m['precision']:.4f}",
        'Recall': f"{m['recall']:.4f}",
        'Accuracy': f"{m['accuracy']:.4f}",
    })

table2_df = pd.DataFrame(table2_data)
table2_df.to_csv(f'{OUTPUTS_DIR}/table2_m6a_metrics.csv', index=False)
print("Table 2 saved to outputs/table2_m6a_metrics.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFigures saved to: {IMAGES_DIR}/")
print(f"Intermediate results saved to: {OUTPUTS_DIR}/")
