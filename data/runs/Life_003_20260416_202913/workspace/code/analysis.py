#!/usr/bin/env python3
"""
Comprehensive analysis for Uncalled4 nanopore signal alignment toolkit.
Reproduces key results: performance benchmarks, pore model analysis,
and m6A modification detection comparison.
"""

import os
import sys
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
from scipy import stats

# Setup paths
WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_003_20260416_202913'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
})
sns.set_style("whitegrid")

# ============================================================================
# 1. PERFORMANCE BENCHMARK ANALYSIS (Table 1)
# ============================================================================
print("=" * 60)
print("1. PERFORMANCE BENCHMARK ANALYSIS")
print("=" * 60)

perf = pd.read_csv(os.path.join(DATA_DIR, 'performance_summary.csv'))
print(perf.to_string())

# Save performance table
perf_table = perf.pivot_table(index='Chemistry', columns='Tool', values=['Time_min', 'FileSize_MB'])
perf_table.to_csv(os.path.join(OUTPUT_DIR, 'performance_table.csv'))
print("\nPerformance table saved.")

# Figure 1a: Alignment Time Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Time comparison
ax = axes[0]
chemistries = perf['Chemistry'].unique()
tools = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']
colors = {'Uncalled4': '#2196F3', 'f5c': '#FF9800', 'Nanopolish': '#4CAF50', 'Tombo': '#F44336'}

x = np.arange(len(chemistries))
width = 0.2

for i, tool in enumerate(tools):
    tool_data = perf[perf['Tool'] == tool]
    times = []
    for chem in chemistries:
        row = tool_data[tool_data['Chemistry'] == chem]
        if len(row) > 0 and pd.notna(row['Time_min'].values[0]):
            times.append(row['Time_min'].values[0])
        else:
            times.append(0)
    bars = ax.bar(x + i * width, times, width, label=tool, color=colors[tool], alpha=0.85)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('Alignment Time (minutes)')
ax.set_title('Alignment Time Comparison')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(chemistries, rotation=15)
ax.legend()
ax.set_yscale('log')
ax.set_ylim(bottom=1)

# File size comparison
ax = axes[1]
for i, tool in enumerate(tools):
    tool_data = perf[perf['Tool'] == tool]
    sizes = []
    for chem in chemistries:
        row = tool_data[tool_data['Chemistry'] == chem]
        if len(row) > 0 and pd.notna(row['FileSize_MB'].values[0]):
            sizes.append(row['FileSize_MB'].values[0])
        else:
            sizes.append(0)
    ax.bar(x + i * width, sizes, width, label=tool, color=colors[tool], alpha=0.85)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('Output File Size (MB)')
ax.set_title('Output File Size Comparison')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(chemistries, rotation=15)
ax.legend()
ax.set_yscale('log')
ax.set_ylim(bottom=1)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'performance_comparison.png'), bbox_inches='tight')
plt.close()
print("Figure 1: Performance comparison saved.")

# Compute speedup factors
print("\nSpeedup of Uncalled4 vs other tools:")
for chem in chemistries:
    chem_data = perf[perf['Chemistry'] == chem]
    uc4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
    if len(uc4_time) > 0 and pd.notna(uc4_time[0]):
        for tool in ['f5c', 'Nanopolish', 'Tombo']:
            other_time = chem_data[chem_data['Tool'] == tool]['Time_min'].values
            if len(other_time) > 0 and pd.notna(other_time[0]):
                speedup = other_time[0] / uc4_time[0]
                print(f"  {chem}: Uncalled4 is {speedup:.1f}x faster than {tool}")

# ============================================================================
# 2. PORE MODEL ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("2. PORE MODEL ANALYSIS")
print("=" * 60)

# Load all pore models
pore_models = {
    'DNA r9.4 (6-mer)': pd.read_csv(os.path.join(DATA_DIR, 'dna_r9.4.1_400bps_6mer_uncalled4.csv')),
    'DNA r10.4 (9-mer)': pd.read_csv(os.path.join(DATA_DIR, 'dna_r10.4.1_400bps_9mer_uncalled4.csv')),
    'RNA001 (5-mer)': pd.read_csv(os.path.join(DATA_DIR, 'rna_r9.4.1_70bps_5mer_uncalled4.csv')),
    'RNA004 (9-mer)': pd.read_csv(os.path.join(DATA_DIR, 'rna004_130bps_9mer_uncalled4.csv')),
}

for name, df in pore_models.items():
    kmer_len = len(df['kmer'].iloc[0])
    print(f"\n{name}: {len(df)} k-mers (k={kmer_len})")
    print(f"  Current mean: {df['current_mean'].mean():.4f} ± {df['current_mean'].std():.4f}")
    print(f"  Current std:  {df['current_std'].mean():.4f} ± {df['current_std'].std():.4f}")
    print(f"  Dwell time:   {df['dwell_time'].mean():.2f} ± {df['dwell_time'].std():.2f}")

# Figure 2: Current Mean Distributions across chemistries
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_list = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    ax.hist(df['current_mean'], bins=80, color=colors_list[idx], alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Current Mean (normalized)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}')
    ax.axvline(df['current_mean'].mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean={df["current_mean"].mean():.2f}')
    ax.legend()

plt.suptitle('Distribution of K-mer Current Means Across Pore Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'pore_model_distributions.png'), bbox_inches='tight')
plt.close()
print("\nFigure 2: Pore model distributions saved.")

# Figure 3: Current Mean vs Standard Deviation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    scatter = ax.scatter(df['current_mean'], df['current_std'], 
                        c=df['dwell_time'], cmap='viridis', alpha=0.3, s=5)
    ax.set_xlabel('Current Mean')
    ax.set_ylabel('Current Std')
    ax.set_title(f'{name}')
    plt.colorbar(scatter, ax=ax, label='Dwell Time')

plt.suptitle('Current Mean vs Standard Deviation (colored by Dwell Time)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'current_mean_vs_std.png'), bbox_inches='tight')
plt.close()
print("Figure 3: Current mean vs std saved.")

# ============================================================================
# 3. SUBSTITUTION PROFILE ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("3. SUBSTITUTION PROFILE ANALYSIS")
print("=" * 60)

def compute_substitution_effects(df, model_name):
    """Compute the effect of single-base substitutions on current mean."""
    kmer_len = len(df['kmer'].iloc[0])
    bases = ['A', 'C', 'G', 'T']
    
    # Create kmer -> current_mean lookup
    kmer_to_current = dict(zip(df['kmer'], df['current_mean']))
    
    # For each position, compute mean absolute change when substituting
    position_effects = np.zeros((4, kmer_len))  # 4 bases x k positions
    position_counts = np.zeros((4, kmer_len))
    
    for kmer, current in kmer_to_current.items():
        for pos in range(kmer_len):
            original_base = kmer[pos]
            for bi, new_base in enumerate(bases):
                if new_base != original_base:
                    new_kmer = kmer[:pos] + new_base + kmer[pos+1:]
                    if new_kmer in kmer_to_current:
                        delta = kmer_to_current[new_kmer] - current
                        position_effects[bi, pos] += abs(delta)
                        position_counts[bi, pos] += 1
    
    # Average
    mask = position_counts > 0
    position_effects[mask] /= position_counts[mask]
    
    return position_effects, bases

# Compute for all models
sub_results = {}
for name, df in pore_models.items():
    effects, bases = compute_substitution_effects(df, name)
    sub_results[name] = effects
    kmer_len = len(df['kmer'].iloc[0])
    print(f"\n{name} - Mean absolute current change by position:")
    for bi, base in enumerate(bases):
        vals = [f"{effects[bi, p]:.4f}" for p in range(kmer_len)]
        print(f"  To {base}: {', '.join(vals)}")

# Figure 4: Substitution profile heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
bases = ['A', 'C', 'G', 'T']

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    effects = sub_results[name]
    kmer_len = len(df['kmer'].iloc[0])
    
    im = ax.imshow(effects, cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(4))
    ax.set_yticklabels(bases)
    ax.set_xticks(range(kmer_len))
    ax.set_xticklabels([f'Pos {i}' for i in range(kmer_len)], rotation=45)
    ax.set_title(f'{name}')
    ax.set_ylabel('Substituted Base')
    ax.set_xlabel('Position in K-mer')
    plt.colorbar(im, ax=ax, label='Mean |ΔCurrent|')

plt.suptitle('Single-Base Substitution Effects on Current Mean', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'substitution_profiles.png'), bbox_inches='tight')
plt.close()
print("\nFigure 4: Substitution profiles saved.")

# ============================================================================
# 4. BASE-POSITION EFFECT ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("4. BASE-POSITION EFFECT ANALYSIS")
print("=" * 60)

def compute_position_base_effect(df):
    """Compute mean current for each base at each position."""
    kmer_len = len(df['kmer'].iloc[0])
    bases = ['A', 'C', 'G', 'T']
    
    results = np.zeros((4, kmer_len))
    counts = np.zeros((4, kmer_len))
    
    for _, row in df.iterrows():
        kmer = row['kmer']
        current = row['current_mean']
        for pos in range(kmer_len):
            bi = bases.index(kmer[pos])
            results[bi, pos] += current
            counts[bi, pos] += 1
    
    mask = counts > 0
    results[mask] /= counts[mask]
    return results, bases

# Figure 5: Base-position effects
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
base_colors = {'A': '#E53935', 'C': '#1E88E5', 'G': '#43A047', 'T': '#FDD835'}

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    effects, bases = compute_position_base_effect(df)
    kmer_len = len(df['kmer'].iloc[0])
    
    for bi, base in enumerate(bases):
        ax.plot(range(kmer_len), effects[bi], 'o-', label=base, 
                color=base_colors[base], linewidth=2, markersize=6)
    
    ax.set_xlabel('Position in K-mer')
    ax.set_ylabel('Mean Current')
    ax.set_title(f'{name}')
    ax.legend()
    ax.set_xticks(range(kmer_len))

plt.suptitle('Base-Position Effects on Ionic Current', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'base_position_effects.png'), bbox_inches='tight')
plt.close()
print("Figure 5: Base-position effects saved.")

# ============================================================================
# 5. NUCLEOTIDE COMPOSITION vs CURRENT ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("5. NUCLEOTIDE COMPOSITION ANALYSIS")
print("=" * 60)

def compute_base_fraction(kmer, base):
    return kmer.count(base) / len(kmer)

# Figure 6: Base composition vs current
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    
    for base, color in base_colors.items():
        fractions = df['kmer'].apply(lambda k: compute_base_fraction(k, base))
        # Group by fraction and compute mean current
        df_temp = pd.DataFrame({'fraction': fractions, 'current': df['current_mean']})
        grouped = df_temp.groupby('fraction')['current'].agg(['mean', 'std']).reset_index()
        ax.errorbar(grouped['fraction'], grouped['mean'], yerr=grouped['std']/2,
                    fmt='o-', label=base, color=color, alpha=0.8, markersize=5, capsize=3)
    
    ax.set_xlabel('Base Fraction in K-mer')
    ax.set_ylabel('Mean Current')
    ax.set_title(f'{name}')
    ax.legend()

plt.suptitle('Nucleotide Composition vs Ionic Current', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'composition_vs_current.png'), bbox_inches='tight')
plt.close()
print("Figure 6: Composition vs current saved.")

# ============================================================================
# 6. DWELL TIME ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("6. DWELL TIME ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    ax.hist(df['dwell_time'], bins=50, color=colors_list[idx], alpha=0.7, edgecolor='white')
    ax.set_xlabel('Dwell Time')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}')
    ax.axvline(df['dwell_time'].mean(), color='red', linestyle='--', 
              label=f'Mean={df["dwell_time"].mean():.1f}')
    ax.axvline(df['dwell_time'].median(), color='blue', linestyle='--',
              label=f'Median={df["dwell_time"].median():.1f}')
    ax.legend()

plt.suptitle('Dwell Time Distributions Across Pore Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'dwell_time_distributions.png'), bbox_inches='tight')
plt.close()
print("Figure 7: Dwell time distributions saved.")

# Dwell time vs current
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    ax.scatter(df['dwell_time'], df['current_mean'], alpha=0.1, s=3, color=colors_list[idx])
    ax.set_xlabel('Dwell Time')
    ax.set_ylabel('Current Mean')
    ax.set_title(f'{name}')
    # Compute correlation
    r, p = stats.pearsonr(df['dwell_time'], df['current_mean'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Dwell Time vs Current Mean', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'dwell_time_vs_current.png'), bbox_inches='tight')
plt.close()
print("Figure 8: Dwell time vs current saved.")

# ============================================================================
# 7. m6A MODIFICATION DETECTION ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("7. m6A MODIFICATION DETECTION ANALYSIS")
print("=" * 60)

# Load data
labels = pd.read_csv(os.path.join(DATA_DIR, 'm6a_labels.csv'))
pred_uc4 = pd.read_csv(os.path.join(DATA_DIR, 'm6a_predictions_uncalled4.csv'))
pred_np = pd.read_csv(os.path.join(DATA_DIR, 'm6a_predictions_nanopolish.csv'))

y_true = labels['label'].values
y_uc4 = pred_uc4['probability'].values
y_np = pred_np['probability'].values

print(f"Total sites: {len(y_true)}")
print(f"Positive sites: {sum(y_true)}")
print(f"Negative sites: {len(y_true) - sum(y_true)}")

# Compute metrics
auprc_uc4 = average_precision_score(y_true, y_uc4)
auprc_np = average_precision_score(y_true, y_np)
auroc_uc4 = roc_auc_score(y_true, y_uc4)
auroc_np = roc_auc_score(y_true, y_np)

print(f"\nUncalled4 AUPRC: {auprc_uc4:.4f}")
print(f"Nanopolish AUPRC: {auprc_np:.4f}")
print(f"Uncalled4 AUROC: {auroc_uc4:.4f}")
print(f"Nanopolish AUROC: {auroc_np:.4f}")

# Save metrics
metrics = {
    'Uncalled4': {'AUPRC': round(auprc_uc4, 4), 'AUROC': round(auroc_uc4, 4)},
    'Nanopolish': {'AUPRC': round(auprc_np, 4), 'AUROC': round(auroc_np, 4)},
    'AUPRC_improvement': round(auprc_uc4 - auprc_np, 4),
    'AUROC_improvement': round(auroc_uc4 - auroc_np, 4)
}
with open(os.path.join(OUTPUT_DIR, 'm6a_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved to m6a_metrics.json")

# Precision-Recall Curves
prec_uc4, rec_uc4, _ = precision_recall_curve(y_true, y_uc4)
prec_np, rec_np, _ = precision_recall_curve(y_true, y_np)

# ROC Curves
fpr_uc4, tpr_uc4, _ = roc_curve(y_true, y_uc4)
fpr_np, tpr_np, _ = roc_curve(y_true, y_np)

# Figure 9: Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(rec_uc4, prec_uc4, color='#2196F3', linewidth=2.5, 
        label=f'Uncalled4 (AUPRC={auprc_uc4:.3f})')
ax.plot(rec_np, prec_np, color='#FF9800', linewidth=2.5, 
        label=f'Nanopolish (AUPRC={auprc_np:.3f})')
baseline = sum(y_true) / len(y_true)
ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve: m6A Detection')
ax.legend(loc='upper right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# ROC Curves
ax = axes[1]
ax.plot(fpr_uc4, tpr_uc4, color='#2196F3', linewidth=2.5,
        label=f'Uncalled4 (AUROC={auroc_uc4:.3f})')
ax.plot(fpr_np, tpr_np, color='#FF9800', linewidth=2.5,
        label=f'Nanopolish (AUROC={auroc_np:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: m6A Detection')
ax.legend(loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'm6a_detection_curves.png'), bbox_inches='tight')
plt.close()
print("Figure 9: m6A detection curves saved.")

# Figure 10: Prediction distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Uncalled4 predictions
ax = axes[0]
pos_mask = y_true == 1
neg_mask = y_true == 0
ax.hist(y_uc4[neg_mask], bins=50, alpha=0.6, color='#1E88E5', label='Negative sites', density=True)
ax.hist(y_uc4[pos_mask], bins=50, alpha=0.6, color='#E53935', label='Positive sites', density=True)
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Density')
ax.set_title('Uncalled4 m6Anet Predictions')
ax.legend()

# Nanopolish predictions
ax = axes[1]
ax.hist(y_np[neg_mask], bins=50, alpha=0.6, color='#1E88E5', label='Negative sites', density=True)
ax.hist(y_np[pos_mask], bins=50, alpha=0.6, color='#E53935', label='Positive sites', density=True)
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Density')
ax.set_title('Nanopolish m6Anet Predictions')
ax.legend()

plt.suptitle('Distribution of m6A Prediction Probabilities', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'm6a_prediction_distributions.png'), bbox_inches='tight')
plt.close()
print("Figure 10: m6A prediction distributions saved.")

# ============================================================================
# 8. THRESHOLD ANALYSIS FOR m6A DETECTION
# ============================================================================
print("\n" + "=" * 60)
print("8. THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
print(f"\n{'Threshold':>10} {'Tool':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)

threshold_results = []
for thresh in thresholds:
    for tool_name, y_pred in [('Uncalled4', y_uc4), ('Nanopolish', y_np)]:
        predicted = (y_pred >= thresh).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        fn = np.sum((predicted == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{thresh:>10.1f} {tool_name:>12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
        threshold_results.append({
            'threshold': thresh, 'tool': tool_name,
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)
        })

pd.DataFrame(threshold_results).to_csv(os.path.join(OUTPUT_DIR, 'threshold_analysis.csv'), index=False)

# ============================================================================
# 9. CHEMISTRY COMPARISON - VIOLIN PLOTS
# ============================================================================
print("\n" + "=" * 60)
print("9. CHEMISTRY COMPARISON")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Prepare data for violin plots
all_data = []
for name, df in pore_models.items():
    for _, row in df.iterrows():
        all_data.append({
            'Chemistry': name,
            'Current Mean': row['current_mean'],
            'Current Std': row['current_std'],
            'Dwell Time': row['dwell_time']
        })

# Sample for violin plot (too many points)
df_all = pd.DataFrame(all_data)
if len(df_all) > 50000:
    df_sample = df_all.sample(50000, random_state=42)
else:
    df_sample = df_all

ax = axes[0]
sns.violinplot(data=df_sample, x='Chemistry', y='Current Mean', ax=ax, 
               palette=colors_list, inner='box', cut=0)
ax.set_title('Current Mean Distribution')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

ax = axes[1]
sns.violinplot(data=df_sample, x='Chemistry', y='Current Std', ax=ax,
               palette=colors_list, inner='box', cut=0)
ax.set_title('Current Std Distribution')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

ax = axes[2]
sns.violinplot(data=df_sample, x='Chemistry', y='Dwell Time', ax=ax,
               palette=colors_list, inner='box', cut=0)
ax.set_title('Dwell Time Distribution')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

plt.suptitle('Comparison of Pore Model Parameters Across Chemistries', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'chemistry_comparison_violin.png'), bbox_inches='tight')
plt.close()
print("Figure 11: Chemistry comparison violin plots saved.")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 60)
print("10. SUMMARY STATISTICS")
print("=" * 60)

summary_stats = {}
for name, df in pore_models.items():
    kmer_len = len(df['kmer'].iloc[0])
    summary_stats[name] = {
        'k': kmer_len,
        'n_kmers': len(df),
        'current_mean_avg': round(df['current_mean'].mean(), 4),
        'current_mean_std': round(df['current_mean'].std(), 4),
        'current_mean_range': [round(df['current_mean'].min(), 4), round(df['current_mean'].max(), 4)],
        'current_std_avg': round(df['current_std'].mean(), 4),
        'dwell_time_avg': round(df['dwell_time'].mean(), 2),
        'dwell_time_median': round(float(df['dwell_time'].median()), 2),
    }

with open(os.path.join(OUTPUT_DIR, 'pore_model_summary.json'), 'w') as f:
    json.dump(summary_stats, f, indent=2)
print("Summary statistics saved.")

# ============================================================================
# 11. ADDITIONAL: GC CONTENT EFFECT
# ============================================================================
print("\n" + "=" * 60)
print("11. GC CONTENT EFFECT")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, df) in enumerate(pore_models.items()):
    ax = axes[idx // 2][idx % 2]
    
    gc_content = df['kmer'].apply(lambda k: (k.count('G') + k.count('C')) / len(k))
    
    df_gc = pd.DataFrame({'gc': gc_content, 'current': df['current_mean']})
    grouped = df_gc.groupby('gc')['current'].agg(['mean', 'std', 'count']).reset_index()
    
    ax.errorbar(grouped['gc'], grouped['mean'], yerr=grouped['std']/np.sqrt(grouped['count']),
                fmt='o-', color=colors_list[idx], markersize=6, capsize=4, linewidth=2)
    ax.set_xlabel('GC Content')
    ax.set_ylabel('Mean Current')
    ax.set_title(f'{name}')
    
    r, p = stats.pearsonr(gc_content, df['current_mean'])
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('GC Content vs Ionic Current', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'gc_content_effect.png'), bbox_inches='tight')
plt.close()
print("Figure 12: GC content effect saved.")

# ============================================================================
# 12. COMPREHENSIVE PERFORMANCE SUMMARY FIGURE
# ============================================================================
print("\n" + "=" * 60)
print("12. COMPREHENSIVE PERFORMANCE FIGURE")
print("=" * 60)

# Heatmap-style performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time heatmap
time_pivot = perf.pivot(index='Chemistry', columns='Tool', values='Time_min')
time_pivot = time_pivot[tools]  # reorder
ax = axes[0]
mask = time_pivot.isna()
sns.heatmap(time_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, mask=mask,
            linewidths=0.5, cbar_kws={'label': 'Time (min)'})
ax.set_title('Alignment Time (minutes)')

# Size heatmap
size_pivot = perf.pivot(index='Chemistry', columns='Tool', values='FileSize_MB')
size_pivot = size_pivot[tools]
ax = axes[1]
mask = size_pivot.isna()
sns.heatmap(size_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, mask=mask,
            linewidths=0.5, cbar_kws={'label': 'Size (MB)'})
ax.set_title('Output File Size (MB)')

plt.suptitle('Performance Benchmark Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'performance_heatmap.png'), bbox_inches='tight')
plt.close()
print("Figure 13: Performance heatmap saved.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll figures saved to: {IMAGE_DIR}")
print(f"All outputs saved to: {OUTPUT_DIR}")

# List all generated files
print("\nGenerated figures:")
for f in sorted(os.listdir(IMAGE_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(IMAGE_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size/1024:.1f} KB)")

print("\nGenerated outputs:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f} ({size/1024:.1f} KB)")
