#!/usr/bin/env python3
"""Analysis of Uncalled4 nanopore signal alignment toolkit."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. PORE MODEL ANALYSIS
# ============================================================

print("=== Loading pore model data ===")
dna_r9 = pd.read_csv('data/dna_r9.4.1_400bps_6mer_uncalled4.csv')
dna_r10 = pd.read_csv('data/dna_r10.4.1_400bps_9mer_uncalled4.csv')
rna_r9 = pd.read_csv('data/rna_r9.4.1_70bps_5mer_uncalled4.csv')
rna004 = pd.read_csv('data/rna004_130bps_9mer_uncalled4.csv')

print(f"DNA r9.4.1 (6-mer): {len(dna_r9)} kmers")
print(f"DNA r10.4.1 (9-mer): {len(dna_r10)} kmers")
print(f"RNA r9.4.1 (5-mer): {len(rna_r9)} kmers")
print(f"RNA004 (9-mer): {len(rna004)} kmers")

# Figure 1: Pore model current distributions across chemistries
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(dna_r9['current_mean'], bins=80, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0,0].set_title('DNA r9.4.1 (6-mer)', fontsize=13, fontweight='bold')
axes[0,0].set_xlabel('Mean Current (pA)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].axvline(dna_r9['current_mean'].median(), color='red', linestyle='--', label=f"Median={dna_r9['current_mean'].median():.2f}")
axes[0,0].legend()

axes[0,1].hist(dna_r10['current_mean'], bins=80, color='#4CAF50', alpha=0.8, edgecolor='white')
axes[0,1].set_title('DNA r10.4.1 (9-mer)', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Mean Current (pA)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].axvline(dna_r10['current_mean'].median(), color='red', linestyle='--', label=f"Median={dna_r10['current_mean'].median():.2f}")
axes[0,1].legend()

axes[1,0].hist(rna_r9['current_mean'], bins=50, color='#FF9800', alpha=0.8, edgecolor='white')
axes[1,0].set_title('RNA r9.4.1 / RNA001 (5-mer)', fontsize=13, fontweight='bold')
axes[1,0].set_xlabel('Mean Current (pA)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].axvline(rna_r9['current_mean'].median(), color='red', linestyle='--', label=f"Median={rna_r9['current_mean'].median():.2f}")
axes[1,0].legend()

axes[1,1].hist(rna004['current_mean'], bins=80, color='#9C27B0', alpha=0.8, edgecolor='white')
axes[1,1].set_title('RNA004 (9-mer)', fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Mean Current (pA)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].axvline(rna004['current_mean'].median(), color='red', linestyle='--', label=f"Median={rna004['current_mean'].median():.2f}")
axes[1,1].legend()

plt.suptitle('Figure 1: Current Distributions Across Nanopore Chemistries', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig1_current_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_current_distributions.png")

# Figure 2: Current mean vs std scatter with dwell time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc1 = axes[0].scatter(dna_r9['current_mean'], dna_r9['current_std'], 
                       c=np.log10(dna_r9['dwell_time']), cmap='viridis', alpha=0.5, s=5)
axes[0].set_title('DNA r9.4.1: Current Mean vs Std', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Mean Current (pA)')
axes[0].set_ylabel('Std Current (pA)')
plt.colorbar(sc1, ax=axes[0], label='log10(Dwell Time)')

sc2 = axes[1].scatter(dna_r10['current_mean'], dna_r10['current_std'],
                       c=np.log10(dna_r10['dwell_time'].clip(lower=1)), cmap='viridis', alpha=0.3, s=2)
axes[1].set_title('DNA r10.4.1: Current Mean vs Std', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Mean Current (pA)')
axes[1].set_ylabel('Std Current (pA)')
plt.colorbar(sc2, ax=axes[1], label='log10(Dwell Time)')

plt.suptitle('Figure 2: Signal Variability vs Mean Current', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig2_signal_variability.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_signal_variability.png")

# Figure 3: Base-position effects (substitution profiles)
# For DNA r9.4.1 (6-mer): analyze effect of each position
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for pos in range(6):
    col = f'pos{pos}'
    dna_r9[col] = dna_r9['kmer'].str[pos]

for pos in range(6):
    ax_idx = pos // 3
    ax_sub = pos % 3
    if ax_idx < 2 and ax_sub < 2:
        ax = axes[ax_idx, ax_sub] if ax_idx * 2 + ax_sub < 4 else None
        if ax is None:
            continue
    else:
        continue

# Simplified: show base composition effects at each position for r9.4.1
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for pos in range(6):
    ax = axes[pos // 3, pos % 3]
    dna_r9[f'pos{pos}'] = dna_r9['kmer'].str[pos]
    groups = dna_r9.groupby(f'pos{pos}')['current_mean'].agg(['mean', 'std']).reindex(['A','C','G','T'])
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    ax.bar(groups.index, groups['mean'], yerr=groups['std'], color=colors, capsize=3, edgecolor='white')
    ax.set_title(f'Position {pos+1}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Current (pA)')
    ax.set_xlabel('Base')

plt.suptitle('Figure 3: Base-Position Effects on Current Signal (DNA r9.4.1)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('report/images/fig3_base_position_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_base_position_effects.png")

# Figure 4: RNA chemistry comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RNA001 base effects at each position
for pos in range(5):
    rna_r9[f'pos{pos}'] = rna_r9['kmer'].str[pos]

# Show first and middle position effects
for i, pos in enumerate([0, 2]):
    ax = axes[i]
    groups = rna_r9.groupby(f'pos{pos}')['current_mean'].agg(['mean', 'std']).reindex(['A','C','G','U'])
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    valid = groups.dropna()
    ax.bar(valid.index, valid['mean'], yerr=valid['std'], color=[colors[['A','C','G','U'].index(b)] for b in valid.index], capsize=3, edgecolor='white')
    ax.set_title(f'RNA001: Position {pos+1}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Current (pA)')
    ax.set_xlabel('Base')

plt.suptitle('Figure 4: Base Effects in RNA Pore Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_rna_base_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_rna_base_effects.png")

# ============================================================
# 2. PERFORMANCE BENCHMARKS
# ============================================================

print("\n=== Performance Benchmarks ===")
perf = pd.read_csv('data/performance_summary.csv')
print(perf.to_string(index=False))

# Figure 5: Performance comparison bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

chem_order = ['DNA r9.4', 'DNA r10.4', 'RNA001', 'RNA004']
tool_colors = {'Uncalled4': '#2196F3', 'f5c': '#FF9800', 'Nanopolish': '#4CAF50', 'Tombo': '#9C27B0'}

# Time comparison
perf_time = perf.pivot(index='Chemistry', columns='Tool', values='Time_min')
perf_time = perf_time.reindex(chem_order)

x = np.arange(len(chem_order))
width = 0.2
for i, tool in enumerate(['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']):
    if tool in perf_time.columns:
        vals = perf_time[tool].values
        mask = ~np.isnan(vals)
        axes[0].bar(x[mask] + i*width, vals[mask], width, label=tool, color=tool_colors[tool], edgecolor='white')

axes[0].set_xticks(x + 1.5*width)
axes[0].set_xticklabels(chem_order, rotation=15)
axes[0].set_ylabel('Alignment Time (minutes)')
axes[0].set_title('Alignment Time by Tool and Chemistry', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].set_yscale('log')

# File size comparison
perf_size = perf.pivot(index='Chemistry', columns='Tool', values='FileSize_MB')
perf_size = perf_size.reindex(chem_order)

for i, tool in enumerate(['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']):
    if tool in perf_size.columns:
        vals = perf_size[tool].values
        mask = ~np.isnan(vals)
        axes[1].bar(x[mask] + i*width, vals[mask], width, label=tool, color=tool_colors[tool], edgecolor='white')

axes[1].set_xticks(x + 1.5*width)
axes[1].set_xticklabels(chem_order, rotation=15)
axes[1].set_ylabel('File Size (MB)')
axes[1].set_title('Output File Size by Tool and Chemistry', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].set_yscale('log')

plt.suptitle('Figure 5: Performance Benchmarks Across Tools and Chemistries', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig5_performance_benchmarks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_performance_benchmarks.png")

# Speedup table
print("\n=== Speedup Analysis ===")
for chem in chem_order:
    sub = perf[perf['Chemistry'] == chem]
    u4_time = sub[sub['Tool'] == 'Uncalled4']['Time_min'].values
    u4_size = sub[sub['Tool'] == 'Uncalled4']['FileSize_MB'].values
    for tool in ['f5c', 'Nanopolish', 'Tombo']:
        t = sub[sub['Tool'] == tool]['Time_min'].values
        s = sub[sub['Tool'] == tool]['FileSize_MB'].values
        if len(u4_time) > 0 and len(t) > 0 and not np.isnan(t[0]):
            print(f"{chem}: Uncalled4 is {t[0]/u4_time[0]:.1f}x faster than {tool}")
            if len(u4_size) > 0 and len(s) > 0:
                print(f"  File size reduction: {s[0]/u4_size[0]:.1f}x smaller")

# ============================================================
# 3. m6A MODIFICATION DETECTION
# ============================================================

print("\n=== m6A Modification Detection ===")
labels = pd.read_csv('data/m6a_labels.csv')
pred_u4 = pd.read_csv('data/m6a_predictions_uncalled4.csv')
pred_nano = pd.read_csv('data/m6a_predictions_nanopolish.csv')

y_true = labels['label'].values
y_u4 = pred_u4['probability'].values
y_nano = pred_nano['probability'].values

print(f"Total sites: {len(y_true)}")
print(f"Positive sites (m6A): {y_true.sum()} ({100*y_true.mean():.1f}%)")
print(f"Negative sites: {(1-y_true).sum()} ({100*(1-y_true.mean()):.1f}%)")

# Compute PR curves
prec_u4, rec_u4, _ = precision_recall_curve(y_true, y_u4)
ap_u4 = average_precision_score(y_true, y_u4)

prec_nano, rec_nano, _ = precision_recall_curve(y_true, y_nano)
ap_nano = average_precision_score(y_true, y_nano)

# Compute ROC curves
fpr_u4, tpr_u4, _ = roc_curve(y_true, y_u4)
roc_auc_u4 = auc(fpr_u4, tpr_u4)

fpr_nano, tpr_nano, _ = roc_curve(y_true, y_nano)
roc_auc_nano = auc(fpr_nano, tpr_nano)

print(f"\nUncalled4:  AUPRC={ap_u4:.4f}, AUROC={roc_auc_u4:.4f}")
print(f"Nanopolish: AUPRC={ap_nano:.4f}, AUROC={roc_auc_nano:.4f}")

# Figure 6: Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rec_u4, prec_u4, color='#2196F3', lw=2.5, label=f'Uncalled4 (AP={ap_u4:.3f})')
axes[0].plot(rec_nano, prec_nano, color='#FF9800', lw=2.5, label=f'Nanopolish (AP={ap_nano:.3f})')
axes[0].set_xlabel('Recall', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim([0, 1.02])
axes[0].set_ylim([0, 1.05])
axes[0].grid(alpha=0.3)

axes[1].plot(fpr_u4, tpr_u4, color='#2196F3', lw=2.5, label=f'Uncalled4 (AUC={roc_auc_u4:.3f})')
axes[1].plot(fpr_nano, tpr_nano, color='#FF9800', lw=2.5, label=f'Nanopolish (AUC={roc_auc_nano:.3f})')
axes[1].plot([0,1],[0,1], 'k--', alpha=0.3)
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 6: m6A Detection Performance — Uncalled4 vs Nanopolish', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig6_m6a_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_m6a_detection.png")

# Figure 7: Score distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, scores, title, color in [(axes[0], y_u4, 'Uncalled4', '#2196F3'), (axes[1], y_nano, 'Nanopolish', '#FF9800')]:
    ax.hist(scores[y_true==0], bins=50, alpha=0.6, color='#95a5a6', label='Unmodified', density=True, edgecolor='white')
    ax.hist(scores[y_true==1], bins=50, alpha=0.6, color=color, label='m6A Modified', density=True, edgecolor='white')
    ax.set_xlabel('Prediction Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{title}: Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)

plt.suptitle('Figure 7: m6A Prediction Score Distributions by Ground Truth', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig7_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_score_distributions.png")

# Figure 8: Sensitivity at fixed FPR thresholds
thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
fig, ax = plt.subplots(figsize=(10, 6))

sens_u4 = []
sens_nano = []
for t in thresholds:
    tp_u4 = ((y_u4 >= t) & (y_true == 1)).sum()
    tp_nano = ((y_nano >= t) & (y_true == 1)).sum()
    sens_u4.append(tp_u4 / y_true.sum())
    sens_nano.append(tp_nano / y_true.sum())

x_pos = np.arange(len(thresholds))
width = 0.35
ax.bar(x_pos - width/2, sens_u4, width, label='Uncalled4', color='#2196F3', edgecolor='white')
ax.bar(x_pos + width/2, sens_nano, width, label='Nanopolish', color='#FF9800', edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
ax.set_xlabel('Probability Threshold', fontsize=12)
ax.set_ylabel('Sensitivity (Recall)', fontsize=12)
ax.set_title('Sensitivity at Different Probability Thresholds', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig8_sensitivity_thresholds.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_sensitivity_thresholds.png")

# ============================================================
# 4. SAVE SUMMARY STATISTICS
# ============================================================

summary = {
    'pore_models': {
        'dna_r9.4.1': {'kmer_count': len(dna_r9), 'kmer_length': 6, 'current_range': [float(dna_r9['current_mean'].min()), float(dna_r9['current_mean'].max())]},
        'dna_r10.4.1': {'kmer_count': len(dna_r10), 'kmer_length': 9, 'current_range': [float(dna_r10['current_mean'].min()), float(dna_r10['current_mean'].max())]},
        'rna_r9.4.1': {'kmer_count': len(rna_r9), 'kmer_length': 5, 'current_range': [float(rna_r9['current_mean'].min()), float(rna_r9['current_mean'].max())]},
        'rna004': {'kmer_count': len(rna004), 'kmer_length': 9, 'current_range': [float(rna004['current_mean'].min()), float(rna004['current_mean'].max())]},
    },
    'm6a_detection': {
        'total_sites': int(len(y_true)),
        'positive_sites': int(y_true.sum()),
        'uncalled4_auprc': float(ap_u4),
        'uncalled4_auroc': float(roc_auc_u4),
        'nanopolish_auprc': float(ap_nano),
        'nanopolish_auroc': float(roc_auc_nano),
    }
}

import json
with open('outputs/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved outputs/analysis_summary.json")

# Performance table as CSV
perf.to_csv('outputs/performance_table.csv', index=False)
print("Saved outputs/performance_table.csv")

print("\n=== ALL ANALYSIS COMPLETE ===")
