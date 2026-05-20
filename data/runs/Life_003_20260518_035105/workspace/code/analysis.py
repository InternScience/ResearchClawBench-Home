#!/usr/bin/env python3
"""
Comprehensive analysis of Uncalled4 nanopore signal alignment toolkit.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("Uncalled4 Nanopore Analysis")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading data...")

# Pore models
dna_r9 = pd.read_csv('data/dna_r9.4.1_400bps_6mer_uncalled4.csv')
dna_r10 = pd.read_csv('data/dna_r10.4.1_400bps_9mer_uncalled4.csv')
rna_r9 = pd.read_csv('data/rna_r9.4.1_70bps_5mer_uncalled4.csv')
rna004 = pd.read_csv('data/rna004_130bps_9mer_uncalled4.csv')

# Performance
perf = pd.read_csv('data/performance_summary.csv')

# m6A
labels = pd.read_csv('data/m6a_labels.csv')
pred_u4 = pd.read_csv('data/m6a_predictions_uncalled4.csv')
pred_nano = pd.read_csv('data/m6a_predictions_nanopolish.csv')

print(f"  DNA R9.4.1 6-mer: {len(dna_r9)} kmers")
print(f"  DNA R10.4.1 9-mer: {len(dna_r10)} kmers")
print(f"  RNA R9.4.1 5-mer: {len(rna_r9)} kmers")
print(f"  RNA004 9-mer: {len(rna004)} kmers")
print(f"  Performance entries: {len(perf)}")
print(f"  m6A sites: {len(labels)}")

# ============================================================
# 2. PORE MODEL ANALYSIS
# ============================================================
print("\n[2] Pore model analysis...")

def analyze_pore_model(df, name, k):
    """Analyze pore model statistics."""
    print(f"\n  {name} ({k}-mer model):")
    print(f"    Current mean range: [{df['current_mean'].min():.2f}, {df['current_mean'].max():.2f}] pA")
    print(f"    Current std range: [{df['current_std'].min():.4f}, {df['current_std'].max():.4f}]")
    print(f"    Dwell time range: [{df['dwell_time'].min()}, {df['dwell_time'].max()}]")
    print(f"    Mean current: {df['current_mean'].mean():.2f} ± {df['current_mean'].std():.2f}")
    print(f"    Mean std: {df['current_std'].mean():.4f} ± {df['current_std'].std():.4f}")
    return df['current_mean'].mean()

m_dna_r9 = analyze_pore_model(dna_r9, "DNA R9.4.1", 6)
m_dna_r10 = analyze_pore_model(dna_r10, "DNA R10.4.1", 9)
m_rna_r9 = analyze_pore_model(rna_r9, "RNA R9.4.1", 5)
m_rna004 = analyze_pore_model(rna004, "RNA004", 9)

# ============================================================
# FIGURE 1: Pore Model Current Distribution Comparison
# ============================================================
print("\n[3] Generating Figure 1: Pore model current distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Current Distribution Across Pore Models', fontsize=16, fontweight='bold')

models = [
    (dna_r9, "DNA R9.4.1 (6-mer)", axes[0, 0], '#2196F3'),
    (dna_r10, "DNA R10.4.1 (9-mer)", axes[0, 1], '#4CAF50'),
    (rna_r9, "RNA R9.4.1 (5-mer)", axes[1, 0], '#FF9800'),
    (rna004, "RNA004 (9-mer)", axes[1, 1], '#9C27B0'),
]

for df, name, ax, color in models:
    ax.hist(df['current_mean'], bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    ax.axvline(df['current_mean'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {df["current_mean"].mean():.1f}')
    ax.set_xlabel('Current Mean (pA)', fontsize=11)
    ax.set_ylabel('Number of k-mers', fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig1_pore_model_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_pore_model_distributions.png")

# ============================================================
# FIGURE 2: Base Position Effects - Current Mean by Position
# ============================================================
print("\n[4] Generating Figure 2: Base position effects...")

def extract_position_effects(df, k, label, color):
    """Extract how each base at each position affects current."""
    results = []
    bases = ['A', 'C', 'G', 'T']
    
    for pos in range(k):
        for base in bases:
            mask = df['kmer'].str[pos] == base
            subset = df[mask]
            if len(subset) > 0:
                results.append({
                    'Position': pos,
                    'Base': base,
                    'Mean_Current': subset['current_mean'].mean(),
                    'Std': subset['current_std'].mean(),
                    'Count': len(subset)
                })
    return pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Base Position Effects on Current Mean', fontsize=16, fontweight='bold')

for (df, name, _, color), ax in zip(models, axes.flat):
    k = len(df['kmer'].iloc[0])
    pos_df = extract_position_effects(df, k, name, color)
    
    bases = ['A', 'C', 'G', 'T']
    colors_bases = {'A': '#e74c3c', 'C': '#3498db', 'G': '#2ecc71', 'T': '#9b59b6'}
    
    for base in bases:
        base_data = pos_df[pos_df['Base'] == base]
        if len(base_data) > 0:
            ax.plot(base_data['Position'], base_data['Mean_Current'], 
                   marker='o', color=colors_bases[base], label=base, 
                   linewidth=2, markersize=6)
            ax.fill_between(base_data['Position'], 
                          base_data['Mean_Current'] - base_data['Std'],
                          base_data['Mean_Current'] + base_data['Std'],
                          alpha=0.2, color=colors_bases[base])
    
    ax.set_xlabel(f'Position in k-mer (0={k}-mer)', fontsize=11)
    ax.set_ylabel('Mean Current (pA)', fontsize=11)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.legend(title='Base', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig2_base_position_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_base_position_effects.png")

# ============================================================
# FIGURE 3: Substitution Profile Heatmaps
# ============================================================
print("\n[5] Generating Figure 3: Substitution profiles...")

def compute_substitution_profile(df, k):
    """Compute mean current substitution profile for each position."""
    profile = []
    bases = ['A', 'C', 'G', 'T']
    
    # For each position, compute the average current difference when a base changes
    for pos in range(k):
        for base in bases:
            mask = df['kmer'].str[pos] == base
            subset = df[mask]
            if len(subset) > 0:
                profile.append({
                    'Position': pos,
                    'Base': base,
                    'Mean_Current': subset['current_mean'].mean(),
                    'Mean_Std': subset['current_std'].mean()
                })
    return pd.DataFrame(profile)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Substitution Profiles: Mean Current by Position and Base', fontsize=16, fontweight='bold')

for (df, name, _, color), ax in zip(models, axes.flat):
    k = len(df['kmer'].iloc[0])
    profile = compute_substitution_profile(df, k)
    
    pivot = profile.pivot_table(values='Mean_Current', index='Base', columns='Position')
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax, 
                linewidths=0.5, center=pivot.values.mean(),
                cbar_kws={'label': 'Mean Current (pA)'})
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Position in k-mer', fontsize=11)
    ax.set_ylabel('Base', fontsize=11)

plt.tight_layout()
plt.savefig('report/images/fig3_substitution_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_substitution_profiles.png")

# ============================================================
# FIGURE 4: Dwell Time Analysis
# ============================================================
print("\n[6] Generating Figure 4: Dwell time analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dwell Time Distribution Across Pore Models', fontsize=16, fontweight='bold')

for (df, name, _, color), ax in zip(models, axes.flat):
    ax.hist(df['dwell_time'], bins=60, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    ax.axvline(df['dwell_time'].mean(), color='red', linestyle='--', linewidth=1.5, 
              label=f'Mean: {df["dwell_time"].mean():.1f}')
    ax.set_xlabel('Dwell Time', fontsize=11)
    ax.set_ylabel('Number of k-mers', fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig4_dwell_time_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_dwell_time_analysis.png")

# ============================================================
# 7. PERFORMANCE BENCHMARK ANALYSIS
# ============================================================
print("\n[7] Performance benchmark analysis...")

# Clean data
perf_clean = perf.dropna(subset=['Time_min', 'FileSize_MB'])
print("\n  Performance Summary:")
print(perf_clean.to_string(index=False))

# Speedup analysis for Uncalled4
u4 = perf_clean[perf_clean['Tool'] == 'Uncalled4']
for chem in perf_clean['Chemistry'].unique():
    chem_data = perf_clean[perf_clean['Chemistry'] == chem]
    u4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
    if len(u4_time) > 0:
        for tool in chem_data['Tool'].unique():
            if tool != 'Uncalled4':
                tool_time = chem_data[chem_data['Tool'] == tool]['Time_min'].values
                if len(tool_time) > 0 and tool_time[0] > 0:
                    speedup = tool_time[0] / u4_time[0]
                    print(f"  {chem}: Uncalled4 vs {tool}: {speedup:.1f}x faster")

# ============================================================
# FIGURE 5: Performance Benchmark - Time Comparison
# ============================================================
print("\n[8] Generating Figure 5: Performance benchmarks...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Time comparison
chem_order = ['DNA r9.4', 'DNA r10.4', 'RNA001', 'RNA004']
tool_colors = {'Uncalled4': '#2196F3', 'f5c': '#4CAF50', 'Nanopolish': '#FF9800', 'Tombo': '#9C27B0'}

# Filter out NaN values for plotting
perf_plot = perf_clean.copy()

ax1 = axes[0]
x = np.arange(len(chem_order))
width = 0.18
tools = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']

for i, tool in enumerate(tools):
    tool_data = perf_plot[perf_plot['Tool'] == tool]
    times = []
    for chem in chem_order:
        val = tool_data[tool_data['Chemistry'] == chem]['Time_min']
        times.append(val.values[0] if len(val) > 0 and pd.notna(val.values[0]) else 0)
    ax1.bar(x + i * width, times, width, label=tool, color=tool_colors.get(tool, 'gray'), edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Sequencing Chemistry', fontsize=12)
ax1.set_ylabel('Alignment Time (minutes)', fontsize=12)
ax1.set_title('Alignment Time by Tool and Chemistry', fontsize=14, fontweight='bold')
ax1.set_xticks(x + 1.5 * width)
ax1.set_xticklabels(chem_order, fontsize=10)
ax1.legend(fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# File size comparison
ax2 = axes[1]
for i, tool in enumerate(tools):
    tool_data = perf_plot[perf_plot['Tool'] == tool]
    sizes = []
    for chem in chem_order:
        val = tool_data[tool_data['Chemistry'] == chem]['FileSize_MB']
        sizes.append(val.values[0] if len(val) > 0 and pd.notna(val.values[0]) else 0)
    ax2.bar(x + i * width, sizes, width, label=tool, color=tool_colors.get(tool, 'gray'), edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Sequencing Chemistry', fontsize=12)
ax2.set_ylabel('Output File Size (MB)', fontsize=12)
ax2.set_title('Output File Size by Tool and Chemistry', fontsize=14, fontweight='bold')
ax2.set_xticks(x + 1.5 * width)
ax2.set_xticklabels(chem_order, fontsize=10)
ax2.legend(fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_performance_benchmarks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_performance_benchmarks.png")

# ============================================================
# FIGURE 6: Speedup Factor (Uncalled4 vs others)
# ============================================================
print("\n[9] Generating Figure 6: Speedup factors...")

fig, ax = plt.subplots(figsize=(12, 6))

speedup_data = []
for chem in chem_order:
    chem_data = perf_plot[perf_plot['Chemistry'] == chem]
    u4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
    if len(u4_time) == 0:
        continue
    u4_t = u4_time[0]
    
    for tool in ['f5c', 'Nanopolish', 'Tombo']:
        tool_data = chem_data[chem_data['Tool'] == tool]['Time_min'].values
        if len(tool_data) > 0 and pd.notna(tool_data[0]) and tool_data[0] > 0:
            speedup_data.append({
                'Chemistry': chem,
                'Comparison': f'vs {tool}',
                'Speedup': tool_data[0] / u4_t
            })

if speedup_data:
    speedup_df = pd.DataFrame(speedup_data)
    speedup_pivot = speedup_df.pivot(index='Chemistry', columns='Comparison', values='Speedup')
    speedup_pivot = speedup_pivot.reindex(chem_order)
    
    speedup_pivot.plot(kind='bar', ax=ax, color=['#FF9800', '#9C27B0', '#4CAF50'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Speedup Factor (x)', fontsize=12)
    ax.set_title('Uncalled4 Speedup vs. Other Tools', fontsize=14, fontweight='bold')
    ax.set_xlabel('Chemistry', fontsize=12)
    ax.legend(title='Comparison', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fx', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/fig6_speedup_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_speedup_factors.png")

# ============================================================
# 10. m6A MODIFICATION DETECTION ANALYSIS
# ============================================================
print("\n[10] m6A modification detection analysis...")

# Merge data
merged = labels.merge(pred_u4, on='site_id').merge(pred_nano, on='site_id', suffixes=('_u4', '_nano'))
print(f"  Merged sites: {len(merged)}")
print(f"  Positive labels: {merged['label'].sum()} ({merged['label'].mean()*100:.1f}%)")
print(f"  Negative labels: {(merged['label']==0).sum()} ({(1-merged['label'].mean())*100:.1f}%)")

# ROC and PR curves
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

fpr_u4, tpr_u4, _ = roc_curve(merged['label'], merged['probability_u4'])
roc_auc_u4 = auc(fpr_u4, tpr_u4)

fpr_nano, tpr_nano, _ = roc_curve(merged['label'], merged['probability_nano'])
roc_auc_nano = auc(fpr_nano, tpr_nano)

precision_u4, recall_u4, _ = precision_recall_curve(merged['label'], merged['probability_u4'])
ap_u4 = average_precision_score(merged['label'], merged['probability_u4'])

precision_nano, recall_nano, _ = precision_recall_curve(merged['label'], merged['probability_nano'])
ap_nano = average_precision_score(merged['label'], merged['probability_nano'])

print(f"\n  Uncalled4:   ROC AUC = {roc_auc_u4:.4f},  PR AUC = {ap_u4:.4f}")
print(f"  Nanopolish:  ROC AUC = {roc_auc_nano:.4f},  PR AUC = {ap_nano:.4f}")

# ============================================================
# FIGURE 7: ROC Curves
# ============================================================
print("\n[11] Generating Figure 7: ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
ax1 = axes[0]
ax1.plot(fpr_u4, tpr_u4, color='#2196F3', linewidth=2.5, label=f'Uncalled4 (AUC={roc_auc_u4:.3f})')
ax1.plot(fpr_nano, tpr_nano, color='#FF9800', linewidth=2.5, label=f'Nanopolish (AUC={roc_auc_nano:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve: m6A Detection', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.grid(True, alpha=0.3)

# PR
ax2 = axes[1]
ax2.plot(recall_u4, precision_u4, color='#2196F3', linewidth=2.5, label=f'Uncalled4 (AP={ap_u4:.3f})')
ax2.plot(recall_nano, precision_nano, color='#FF9800', linewidth=2.5, label=f'Nanopolish (AP={ap_nano:.3f})')
baseline = merged['label'].mean()
ax2.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.2f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve: m6A Detection', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([0, 1.05])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig7_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig7_roc_pr_curves.png")

# ============================================================
# FIGURE 8: Prediction Score Distribution
# ============================================================
print("\n[12] Generating Figure 8: Score distributions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Uncalled4
ax1 = axes[0]
pos_u4 = merged[merged['label'] == 1]['probability_u4']
neg_u4 = merged[merged['label'] == 0]['probability_u4']
ax1.hist(neg_u4, bins=50, alpha=0.6, color='#3498db', label='Unmodified', density=True, edgecolor='black', linewidth=0.5)
ax1.hist(pos_u4, bins=50, alpha=0.6, color='#e74c3c', label='Modified (m6A)', density=True, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Prediction Probability', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Uncalled4 + m6Anet Predictions', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)

# Nanopolish
ax2 = axes[1]
pos_nano = merged[merged['label'] == 1]['probability_nano']
neg_nano = merged[merged['label'] == 0]['probability_nano']
ax2.hist(neg_nano, bins=50, alpha=0.6, color='#3498db', label='Unmodified', density=True, edgecolor='black', linewidth=0.5)
ax2.hist(pos_nano, bins=50, alpha=0.6, color='#e74c3c', label='Modified (m6A)', density=True, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Prediction Probability', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Nanopolish + m6Anet Predictions', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('report/images/fig8_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig8_score_distributions.png")

# ============================================================
# FIGURE 9: Threshold Analysis
# ============================================================
print("\n[13] Generating Figure 9: Threshold analysis...")

thresholds = np.arange(0, 1.01, 0.01)
metrics_u4 = []
metrics_nano = []

for t in thresholds:
    pred_u4_binary = (merged['probability_u4'] >= t).astype(int)
    pred_nano_binary = (merged['probability_nano'] >= t).astype(int)
    
    tp_u4 = ((pred_u4_binary == 1) & (merged['label'] == 1)).sum()
    fp_u4 = ((pred_u4_binary == 1) & (merged['label'] == 0)).sum()
    tn_u4 = ((pred_u4_binary == 0) & (merged['label'] == 0)).sum()
    fn_u4 = ((pred_u4_binary == 0) & (merged['label'] == 1)).sum()
    
    tp_nano = ((pred_nano_binary == 1) & (merged['label'] == 1)).sum()
    fp_nano = ((pred_nano_binary == 1) & (merged['label'] == 0)).sum()
    tn_nano = ((pred_nano_binary == 0) & (merged['label'] == 0)).sum()
    fn_nano = ((pred_nano_binary == 0) & (merged['label'] == 1)).sum()
    
    acc_u4 = (tp_u4 + tn_u4) / (tp_u4 + fp_u4 + tn_u4 + fn_u4) if (tp_u4 + fp_u4 + tn_u4 + fn_u4) > 0 else 0
    acc_nano = (tp_nano + tn_nano) / (tp_nano + fp_nano + tn_nano + fn_nano) if (tp_nano + fp_nano + tn_nano + fn_nano) > 0 else 0
    
    prec_u4 = tp_u4 / (tp_u4 + fp_u4) if (tp_u4 + fp_u4) > 0 else 0
    prec_nano = tp_nano / (tp_nano + fp_nano) if (tp_nano + fp_nano) > 0 else 0
    
    rec_u4 = tp_u4 / (tp_u4 + fn_u4) if (tp_u4 + fn_u4) > 0 else 0
    rec_nano = tp_nano / (tp_nano + fn_nano) if (tp_nano + fn_nano) > 0 else 0
    
    f1_u4 = 2 * prec_u4 * rec_u4 / (prec_u4 + rec_u4) if (prec_u4 + rec_u4) > 0 else 0
    f1_nano = 2 * prec_nano * rec_nano / (prec_nano + rec_nano) if (prec_nano + rec_nano) > 0 else 0
    
    metrics_u4.append({'threshold': t, 'accuracy': acc_u4, 'precision': prec_u4, 'recall': rec_u4, 'f1': f1_u4})
    metrics_nano.append({'threshold': t, 'accuracy': acc_nano, 'precision': prec_nano, 'recall': rec_nano, 'f1': f1_nano})

metrics_u4_df = pd.DataFrame(metrics_u4)
metrics_nano_df = pd.DataFrame(metrics_nano)

# Find best F1 thresholds
best_u4_idx = metrics_u4_df['f1'].idxmax()
best_nano_idx = metrics_nano_df['f1'].idxmax()
print(f"  Uncalled4 best F1 threshold: {metrics_u4_df.loc[best_u4_idx, 'threshold']:.2f} (F1={metrics_u4_df.loc[best_u4_idx, 'f1']:.4f})")
print(f"  Nanopolish best F1 threshold: {metrics_nano_df.loc[best_nano_idx, 'threshold']:.2f} (F1={metrics_nano_df.loc[best_nano_idx, 'f1']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Uncalled4 threshold analysis
ax1 = axes[0]
ax1.plot(metrics_u4_df['threshold'], metrics_u4_df['accuracy'], label='Accuracy', linewidth=2)
ax1.plot(metrics_u4_df['threshold'], metrics_u4_df['precision'], label='Precision', linewidth=2)
ax1.plot(metrics_u4_df['threshold'], metrics_u4_df['recall'], label='Recall', linewidth=2)
ax1.plot(metrics_u4_df['threshold'], metrics_u4_df['f1'], label='F1 Score', linewidth=2, color='red')
ax1.axvline(metrics_u4_df.loc[best_u4_idx, 'threshold'], color='red', linestyle='--', alpha=0.5, label=f'Best F1 ({metrics_u4_df.loc[best_u4_idx, "threshold"]:.2f})')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Uncalled4: Metric vs Threshold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Nanopolish threshold analysis
ax2 = axes[1]
ax2.plot(metrics_nano_df['threshold'], metrics_nano_df['accuracy'], label='Accuracy', linewidth=2)
ax2.plot(metrics_nano_df['threshold'], metrics_nano_df['precision'], label='Precision', linewidth=2)
ax2.plot(metrics_nano_df['threshold'], metrics_nano_df['recall'], label='Recall', linewidth=2)
ax2.plot(metrics_nano_df['threshold'], metrics_nano_df['f1'], label='F1 Score', linewidth=2, color='red')
ax2.axvline(metrics_nano_df.loc[best_nano_idx, 'threshold'], color='red', linestyle='--', alpha=0.5, label=f'Best F1 ({metrics_nano_df.loc[best_nano_idx, "threshold"]:.2f})')
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Nanopolish: Metric vs Threshold', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig9_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig9_threshold_analysis.png")

# ============================================================
# FIGURE 10: Comprehensive Comparison Summary
# ============================================================
print("\n[14] Generating Figure 10: Summary comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: k-mer space size comparison
ax = axes[0, 0]
kmer_sizes = [4096, 262144, 1024, 262144]
labels_k = ['DNA R9.4.1\n(6-mer)', 'DNA R10.4.1\n(9-mer)', 'RNA R9.4.1\n(5-mer)', 'RNA004\n(9-mer)']
colors_k = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.bar(labels_k, kmer_sizes, color=colors_k, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Number of k-mers', fontsize=11)
ax.set_title('A. k-mer Space Size', fontsize=13, fontweight='bold')
ax.set_yscale('log')
for bar, val in zip(bars, kmer_sizes):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1, 
           f'{val:,}', ha='center', va='bottom', fontsize=10)

# Panel B: Current range comparison
ax = axes[0, 1]
ranges = [
    (dna_r9['current_mean'].min(), dna_r9['current_mean'].max()),
    (dna_r10['current_mean'].min(), dna_r10['current_mean'].max()),
    (rna_r9['current_mean'].min(), rna_r9['current_mean'].max()),
    (rna004['current_mean'].min(), rna004['current_mean'].max()),
]
labels_r = ['DNA R9.4.1', 'DNA R10.4.1', 'RNA R9.4.1', 'RNA004']
for i, ((lo, hi), label, color) in enumerate(zip(ranges, labels_r, colors_k)):
    ax.barh(i, hi - lo, left=lo, height=0.6, color=color, edgecolor='black', linewidth=0.5)
    ax.text((lo + hi) / 2, i, f'{hi - lo:.1f} pA', ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_yticks(range(4))
ax.set_yticklabels(labels_r)
ax.set_xlabel('Current Mean Range (pA)', fontsize=11)
ax.set_title('B. Current Range', fontsize=13, fontweight='bold')

# Panel C: Performance heatmap
ax = axes[1, 0]
perf_pivot_time = perf_clean.pivot_table(values='Time_min', index='Tool', columns='Chemistry')
perf_pivot_time = perf_pivot_time[['DNA r9.4', 'DNA r10.4', 'RNA001', 'RNA004']]
sns.heatmap(perf_pivot_time, annot=True, fmt='.0f', cmap='YlOrRd_r', ax=ax, 
           linewidths=0.5, cbar_kws={'label': 'Time (min)'})
ax.set_title('C. Alignment Time (minutes)', fontsize=13, fontweight='bold')
ax.set_ylabel('')

# Panel D: m6A metrics summary
ax = axes[1, 1]
metrics_summary = {
    'Metric': ['ROC AUC', 'PR AUC', 'Best F1', 'Best F1 Threshold'],
    'Uncalled4': [roc_auc_u4, ap_u4, metrics_u4_df.loc[best_u4_idx, 'f1'], metrics_u4_df.loc[best_u4_idx, 'threshold']],
    'Nanopolish': [roc_auc_nano, ap_nano, metrics_nano_df.loc[best_nano_idx, 'f1'], metrics_nano_df.loc[best_nano_idx, 'threshold']],
}
summary_df = pd.DataFrame(metrics_summary)
table = ax.table(cellText=[[f'{v:.4f}' if isinstance(v, float) else str(v) for v in row] 
                           for row in summary_df.values],
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
ax.set_title('D. m6A Detection Metrics', fontsize=13, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('report/images/fig10_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig10_summary.png")

# ============================================================
# SAVE INTERMEDIATE RESULTS
# ============================================================
print("\n[15] Saving intermediate results...")

# Save merged m6A data
merged.to_csv('outputs/m6a_merged_predictions.csv', index=False)

# Save performance summary with speedups
perf_clean.to_csv('outputs/performance_benchmark.csv', index=False)

# Save metrics
metrics_u4_df.to_csv('outputs/threshold_metrics_uncalled4.csv', index=False)
metrics_nano_df.to_csv('outputs/threshold_metrics_nanopolish.csv', index=False)

# Save summary
summary_df.to_csv('outputs/m6a_metrics_summary.csv', index=False)

print("  Saved intermediate results to outputs/")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print(f"\nFigures generated:")
print("  fig1_pore_model_distributions.png")
print("  fig2_base_position_effects.png")
print("  fig3_substitution_profiles.png")
print("  fig4_dwell_time_analysis.png")
print("  fig5_performance_benchmarks.png")
print("  fig6_speedup_factors.png")
print("  fig7_roc_pr_curves.png")
print("  fig8_score_distributions.png")
print("  fig9_threshold_analysis.png")
print("  fig10_summary.png")
