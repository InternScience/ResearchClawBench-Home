#!/usr/bin/env python3
"""
Uncalled4 Analysis: Performance Benchmarking, Pore Model Analysis, and m6A Detection Comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMAGE_DIR = "report/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# Phase 1: Load Data
# ============================================================
print("Loading data...")

perf = pd.read_csv(f"{DATA_DIR}/performance_summary.csv")
dna_r9 = pd.read_csv(f"{DATA_DIR}/dna_r9.4.1_400bps_6mer_uncalled4.csv")
dna_r10 = pd.read_csv(f"{DATA_DIR}/dna_r10.4.1_400bps_9mer_uncalled4.csv")
rna_r9 = pd.read_csv(f"{DATA_DIR}/rna_r9.4.1_70bps_5mer_uncalled4.csv")
rna004 = pd.read_csv(f"{DATA_DIR}/rna004_130bps_9mer_uncalled4.csv")
m6a_uc4 = pd.read_csv(f"{DATA_DIR}/m6a_predictions_uncalled4.csv")
m6a_np = pd.read_csv(f"{DATA_DIR}/m6a_predictions_nanopolish.csv")
m6a_lab = pd.read_csv(f"{DATA_DIR}/m6a_labels.csv")

print(f"Performance: {perf.shape}")
print(f"DNA r9.4.1: {dna_r9.shape}, DNA r10.4.1: {dna_r10.shape}")
print(f"RNA r9.4.1: {rna_r9.shape}, RNA004: {rna004.shape}")
print(f"m6A predictions: {m6a_uc4.shape}, labels: {m6a_lab.shape}")

# ============================================================
# Phase 2: Performance Benchmarking
# ============================================================
print("\n=== Performance Benchmarking ===")

# Create performance summary table
perf_pivot_time = perf.pivot_table(index='Chemistry', columns='Tool', values='Time_min')
perf_pivot_size = perf.pivot_table(index='Chemistry', columns='Tool', values='FileSize_MB')

print("\nAlignment Time (minutes):")
print(perf_pivot_time.to_string())
print("\nFile Size (MB):")
print(perf_pivot_size.to_string())

# Calculate speedup ratios
speedups = {}
for chem in perf['Chemistry'].unique():
    chem_data = perf[perf['Chemistry'] == chem]
    uc4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
    if len(uc4_time) > 0:
        uc4_time = uc4_time[0]
        for tool in chem_data['Tool'].unique():
            if tool != 'Uncalled4':
                tool_time = chem_data[chem_data['Tool'] == tool]['Time_min'].values
                if len(tool_time) > 0 and not np.isnan(tool_time[0]):
                    speedups[f"{chem}_{tool}"] = tool_time[0] / uc4_time

print("\nSpeedup ratios (tool_time / Uncalled4_time):")
for k, v in speedups.items():
    print(f"  {k}: {v:.1f}x")

# Save performance summary
perf_summary = {
    'time_table': perf_pivot_time.to_dict(),
    'size_table': perf_pivot_size.to_dict(),
    'speedups': speedups
}
with open(f"{OUTPUT_DIR}/performance_summary.json", 'w') as f:
    json.dump(perf_summary, f, indent=2, default=str)

# Figure 1: Performance comparison - Time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time comparison
chem_order = ['DNA r9.4', 'DNA r10.4', 'RNA001', 'RNA004']
tool_order = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']
colors = {'Uncalled4': '#2196F3', 'f5c': '#FF9800', 'Nanopolish': '#4CAF50', 'Tombo': '#9C27B0'}

ax = axes[0]
x = np.arange(len(chem_order))
width = 0.2
for i, tool in enumerate(tool_order):
    vals = []
    for chem in chem_order:
        row = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == tool)]
        if len(row) > 0:
            v = row['Time_min'].values[0]
            vals.append(v if not np.isnan(v) else 0)
        else:
            vals.append(0)
    bars = ax.bar(x + i * width, vals, width, label=tool, color=colors[tool], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                   f'{v:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('Alignment Time (minutes)')
ax.set_title('A) Alignment Time Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chem_order)
ax.set_yscale('log')
ax.legend(title='Tool', fontsize=9)
ax.set_ylim(bottom=1)

# File size comparison
ax = axes[1]
for i, tool in enumerate(tool_order):
    vals = []
    for chem in chem_order:
        row = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == tool)]
        if len(row) > 0:
            v = row['FileSize_MB'].values[0]
            vals.append(v if not np.isnan(v) else 0)
        else:
            vals.append(0)
    bars = ax.bar(x + i * width, vals, width, label=tool, color=colors[tool], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                   f'{v:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('File Size (MB)')
ax.set_title('B) Output File Size Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chem_order)
ax.set_yscale('log')
ax.legend(title='Tool', fontsize=9)
ax.set_ylim(bottom=1)

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig1_performance_benchmark.png", bbox_inches='tight')
plt.close()
print("Saved fig1_performance_benchmark.png")

# Figure 2: Speedup ratio heatmap
fig, ax = plt.subplots(figsize=(8, 4))
speedup_matrix = []
tools_for_heatmap = ['f5c', 'Nanopolish', 'Tombo']
for chem in chem_order:
    row = []
    for tool in tools_for_heatmap:
        key = f"{chem}_{tool}"
        row.append(speedups.get(key, np.nan))
    speedup_matrix.append(row)
speedup_df = pd.DataFrame(speedup_matrix, index=chem_order, columns=tools_for_heatmap)

sns.heatmap(speedup_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=1, linecolor='white', cbar_kws={'label': 'Speedup (×) vs Uncalled4'})
ax.set_title('Speedup Ratio: Other Tools vs Uncalled4')
ax.set_ylabel('Chemistry')
ax.set_xlabel('Tool')
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig2_speedup_heatmap.png", bbox_inches='tight')
plt.close()
print("Saved fig2_speedup_heatmap.png")

# ============================================================
# Phase 3: Pore Model Analysis
# ============================================================
print("\n=== Pore Model Analysis ===")

def analyze_pore_model(df, name, k):
    """Analyze a pore model dataframe"""
    stats = {
        'name': name,
        'k': k,
        'n_kmers': len(df),
        'mean_current': df['current_mean'].mean(),
        'std_current': df['current_mean'].std(),
        'mean_std': df['current_std'].mean(),
        'mean_dwell': df['dwell_time'].mean(),
    }
    return stats

pore_stats = []
pore_stats.append(analyze_pore_model(dna_r9, 'DNA r9.4.1 (6-mer)', 6))
pore_stats.append(analyze_pore_model(dna_r10, 'DNA r10.4.1 (9-mer)', 9))
pore_stats.append(analyze_pore_model(rna_r9, 'RNA001 (5-mer)', 5))
pore_stats.append(analyze_pore_model(rna004, 'RNA004 (9-mer)', 9))

pore_stats_df = pd.DataFrame(pore_stats)
print(pore_stats_df.to_string(index=False))
pore_stats_df.to_csv(f"{OUTPUT_DIR}/pore_model_stats.csv", index=False)

# Figure 3: Current distribution comparison for pore models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DNA comparison
ax = axes[0, 0]
ax.hist(dna_r9['current_mean'], bins=80, alpha=0.6, label='DNA r9.4.1 (6-mer)', color='#2196F3', density=True)
ax.hist(dna_r10['current_mean'], bins=80, alpha=0.6, label='DNA r10.4.1 (9-mer)', color='#F44336', density=True)
ax.set_xlabel('Mean Current (pA)')
ax.set_ylabel('Density')
ax.set_title('A) DNA Pore Model: Current Distribution')
ax.legend()

ax = axes[0, 1]
ax.hist(rna_r9['current_mean'], bins=50, alpha=0.6, label='RNA001 (5-mer)', color='#4CAF50', density=True)
ax.hist(rna004['current_mean'], bins=80, alpha=0.6, label='RNA004 (9-mer)', color='#FF9800', density=True)
ax.set_xlabel('Mean Current (pA)')
ax.set_ylabel('Density')
ax.set_title('B) RNA Pore Model: Current Distribution')
ax.legend()

# Current std comparison
ax = axes[1, 0]
ax.hist(dna_r9['current_std'], bins=80, alpha=0.6, label='DNA r9.4.1 (6-mer)', color='#2196F3', density=True)
ax.hist(dna_r10['current_std'], bins=80, alpha=0.6, label='DNA r10.4.1 (9-mer)', color='#F44336', density=True)
ax.set_xlabel('Current Standard Deviation (pA)')
ax.set_ylabel('Density')
ax.set_title('C) DNA Pore Model: Current Variability')
ax.legend()

ax = axes[1, 1]
ax.hist(rna_r9['current_std'], bins=50, alpha=0.6, label='RNA001 (5-mer)', color='#4CAF50', density=True)
ax.hist(rna004['current_std'], bins=80, alpha=0.6, label='RNA004 (9-mer)', color='#FF9800', density=True)
ax.set_xlabel('Current Standard Deviation (pA)')
ax.set_ylabel('Density')
ax.set_title('D) RNA Pore Model: Current Variability')
ax.legend()

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig3_pore_model_distributions.png", bbox_inches='tight')
plt.close()
print("Saved fig3_pore_model_distributions.png")

# Figure 4: Base-position effect analysis for DNA r9.4.1 (6-mer)
def analyze_base_position_effect(df, k):
    """Analyze how each base position affects the current signal"""
    position_effects = {i: {base: [] for base in 'ACGT'} for i in range(k)}
    
    for _, row in df.iterrows():
        kmer = row['kmer']
        for i, base in enumerate(kmer):
            position_effects[i][base].append(row['current_mean'])
    
    return position_effects

# DNA r9.4.1 base position effects
pos_effects_dna_r9 = analyze_base_position_effect(dna_r9, 6)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DNA r9.4.1
ax = axes[0]
base_colors = {'A': '#4CAF50', 'C': '#2196F3', 'G': '#FF9800', 'T': '#F44336'}
positions = range(6)
for base in 'ACGT':
    means = [np.mean(pos_effects_dna_r9[i][base]) for i in positions]
    stds = [np.std(pos_effects_dna_r9[i][base]) for i in positions]
    ax.errorbar(positions, means, yerr=stds, label=base, color=base_colors[base],
               marker='o', capsize=3, linewidth=2, markersize=6)

ax.set_xlabel('Position in 6-mer')
ax.set_ylabel('Mean Current (pA)')
ax.set_title('A) DNA r9.4.1: Base-Position Effect on Current')
ax.set_xticks(positions)
ax.set_xticklabels([f'Pos {i+1}' for i in positions])
ax.legend(title='Base')

# RNA001 base position effects
pos_effects_rna_r9 = analyze_base_position_effect(rna_r9, 5)

ax = axes[1]
positions = range(5)
for base in 'ACGU':
    base_key = base
    if base == 'U':
        base_key = 'T'  # stored as T in kmer
    means = [np.mean(pos_effects_rna_r9[i][base_key]) for i in positions]
    stds = [np.std(pos_effects_rna_r9[i][base_key]) for i in positions]
    label = 'U' if base == 'U' else base
    ax.errorbar(positions, means, yerr=stds, label=label, 
               color=base_colors.get(base_key, '#9C27B0'),
               marker='o', capsize=3, linewidth=2, markersize=6)

ax.set_xlabel('Position in 5-mer')
ax.set_ylabel('Mean Current (pA)')
ax.set_title('B) RNA001: Base-Position Effect on Current')
ax.set_xticks(positions)
ax.set_xticklabels([f'Pos {i+1}' for i in positions])
ax.legend(title='Base')

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig4_base_position_effects.png", bbox_inches='tight')
plt.close()
print("Saved fig4_base_position_effects.png")

# Figure 5: Substitution profile - effect of single nucleotide substitution on current
# For DNA r9.4.1, compare k-mers that differ by one base at center position
def compute_substitution_profile(df, k):
    """Compute the current difference when substituting the central base"""
    center = k // 2
    kmer_dict = dict(zip(df['kmer'], df['current_mean']))
    
    substitutions = {}
    for ref_base in 'ACGT':
        for sub_base in 'ACGT':
            if ref_base == sub_base:
                continue
            diffs = []
            for _, row in df.iterrows():
                kmer = row['kmer']
                if kmer[center] == ref_base:
                    substituted = kmer[:center] + sub_base + kmer[center+1:]
                    if substituted in kmer_dict:
                        diffs.append(kmer_dict[substituted] - row['current_mean'])
            if diffs:
                substitutions[(ref_base, sub_base)] = {
                    'mean_diff': np.mean(diffs),
                    'std_diff': np.std(diffs),
                    'n': len(diffs)
                }
    return substitutions

sub_profile_dna_r9 = compute_substitution_profile(dna_r9, 6)
sub_profile_rna_r9 = compute_substitution_profile(rna_r9, 5)

# Create substitution matrix
def sub_profile_to_matrix(sub_profile):
    bases = 'ACGT'
    matrix = np.zeros((4, 4))
    for i, ref in enumerate(bases):
        for j, sub in enumerate(bases):
            if ref == sub:
                matrix[i, j] = 0
            else:
                key = (ref, sub)
                if key in sub_profile:
                    matrix[i, j] = sub_profile[key]['mean_diff']
    return matrix

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# DNA r9.4.1 substitution profile
mat_dna = sub_profile_to_matrix(sub_profile_dna_r9)
ax = axes[0]
sns.heatmap(mat_dna, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            xticklabels=list('ACGT'), yticklabels=list('ACGT'),
            linewidths=1, linecolor='white',
            cbar_kws={'label': 'Current Shift (pA)'})
ax.set_xlabel('Substituted Base')
ax.set_ylabel('Reference Base')
ax.set_title('A) DNA r9.4.1: Central Base Substitution Profile')

# RNA001 substitution profile
mat_rna = sub_profile_to_matrix(sub_profile_rna_r9)
ax = axes[1]
sns.heatmap(mat_rna, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            xticklabels=list('ACGU'), yticklabels=list('ACGU'),
            linewidths=1, linecolor='white',
            cbar_kws={'label': 'Current Shift (pA)'})
ax.set_xlabel('Substituted Base')
ax.set_ylabel('Reference Base')
ax.set_title('B) RNA001: Central Base Substitution Profile')

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig5_substitution_profiles.png", bbox_inches='tight')
plt.close()
print("Saved fig5_substitution_profiles.png")

# Save substitution profiles
with open(f"{OUTPUT_DIR}/substitution_profiles.json", 'w') as f:
    # Convert tuple keys to strings
    out = {}
    for k, v in sub_profile_dna_r9.items():
        out[f"DNA_r9.4.1_{k[0]}->{k[1]}"] = v
    for k, v in sub_profile_rna_r9.items():
        out[f"RNA001_{k[0]}->{k[1]}"] = v
    json.dump(out, f, indent=2)

# Figure 6: Dwell time distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(dna_r9['dwell_time'], bins=50, alpha=0.6, label='DNA r9.4.1 (6-mer)', color='#2196F3', density=True)
ax.hist(dna_r10['dwell_time'], bins=50, alpha=0.6, label='DNA r10.4.1 (9-mer)', color='#F44336', density=True)
ax.set_xlabel('Dwell Time (samples)')
ax.set_ylabel('Density')
ax.set_title('A) DNA Pore Model: Dwell Time Distribution')
ax.legend()

ax = axes[1]
ax.hist(rna_r9['dwell_time'], bins=50, alpha=0.6, label='RNA001 (5-mer)', color='#4CAF50', density=True)
ax.hist(rna004['dwell_time'], bins=50, alpha=0.6, label='RNA004 (9-mer)', color='#FF9800', density=True)
ax.set_xlabel('Dwell Time (samples)')
ax.set_ylabel('Density')
ax.set_title('B) RNA Pore Model: Dwell Time Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig6_dwell_time_distributions.png", bbox_inches='tight')
plt.close()
print("Saved fig6_dwell_time_distributions.png")

# ============================================================
# Phase 4: m6A Modification Detection Comparison
# ============================================================
print("\n=== m6A Modification Detection Comparison ===")

# Merge predictions with labels
merged = m6a_uc4.merge(m6a_np, on='site_id', suffixes=('_uc4', '_np')).merge(m6a_lab, on='site_id')
y_true = merged['label'].values
y_uc4 = merged['probability_uc4'].values
y_np = merged['probability_np'].values

n_positive = y_true.sum()
n_negative = len(y_true) - n_positive
print(f"Total sites: {len(y_true)}, Positive: {n_positive}, Negative: {n_negative}")
print(f"Positive rate: {n_positive/len(y_true)*100:.1f}%")

# Precision-Recall curves
precision_uc4, recall_uc4, thresholds_uc4 = precision_recall_curve(y_true, y_uc4)
precision_np, recall_np, thresholds_np = precision_recall_curve(y_true, y_np)

ap_uc4 = average_precision_score(y_true, y_uc4)
ap_np = average_precision_score(y_true, y_np)

# ROC curves
fpr_uc4, tpr_uc4, _ = roc_curve(y_true, y_uc4)
fpr_np, tpr_np, _ = roc_curve(y_true, y_np)

auc_roc_uc4 = auc(fpr_uc4, tpr_uc4)
auc_roc_np = auc(fpr_np, tpr_np)

print(f"\nUncalled4 - PR AUC: {ap_uc4:.4f}, ROC AUC: {auc_roc_uc4:.4f}")
print(f"Nanopolish - PR AUC: {ap_np:.4f}, ROC AUC: {auc_roc_np:.4f}")
print(f"Improvement in PR AUC: {(ap_uc4 - ap_np)*100:.2f} percentage points")
print(f"Improvement in ROC AUC: {(auc_roc_uc4 - auc_roc_np)*100:.2f} percentage points")

# Save m6A metrics
m6a_metrics = {
    'n_sites': int(len(y_true)),
    'n_positive': int(n_positive),
    'n_negative': int(n_negative),
    'uncalled4': {
        'pr_auc': float(ap_uc4),
        'roc_auc': float(auc_roc_uc4),
    },
    'nanopolish': {
        'pr_auc': float(ap_np),
        'roc_auc': float(auc_roc_np),
    },
    'pr_auc_improvement': float(ap_uc4 - ap_np),
    'roc_auc_improvement': float(auc_roc_uc4 - auc_roc_np),
}
with open(f"{OUTPUT_DIR}/m6a_metrics.json", 'w') as f:
    json.dump(m6a_metrics, f, indent=2)

# Figure 7: PR and ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PR curve
ax = axes[0]
ax.plot(recall_uc4, precision_uc4, color='#2196F3', linewidth=2,
        label=f'Uncalled4 (AP = {ap_uc4:.3f})')
ax.plot(recall_np, precision_np, color='#4CAF50', linewidth=2,
        label=f'Nanopolish (AP = {ap_np:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('A) Precision-Recall Curve for m6A Detection')
ax.legend(loc='best', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

# ROC curve
ax = axes[1]
ax.plot(fpr_uc4, tpr_uc4, color='#2196F3', linewidth=2,
        label=f'Uncalled4 (AUC = {auc_roc_uc4:.3f})')
ax.plot(fpr_np, tpr_np, color='#4CAF50', linewidth=2,
        label=f'Nanopolish (AUC = {auc_roc_np:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('B) ROC Curve for m6A Detection')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig7_m6a_pr_roc_curves.png", bbox_inches='tight')
plt.close()
print("Saved fig7_m6a_pr_roc_curves.png")

# Figure 8: Prediction probability distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
pos_uc4 = y_uc4[y_true == 1]
neg_uc4 = y_uc4[y_true == 0]
ax.hist(neg_uc4, bins=50, alpha=0.6, label=f'Unmodified (n={len(neg_uc4)})', color='#9E9E9E', density=True)
ax.hist(pos_uc4, bins=50, alpha=0.6, label=f'm6A Modified (n={len(pos_uc4)})', color='#F44336', density=True)
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Density')
ax.set_title('A) Uncalled4: Prediction Distribution')
ax.legend()

ax = axes[1]
pos_np = y_np[y_true == 1]
neg_np = y_np[y_true == 0]
ax.hist(neg_np, bins=50, alpha=0.6, label=f'Unmodified (n={len(neg_np)})', color='#9E9E9E', density=True)
ax.hist(pos_np, bins=50, alpha=0.6, label=f'm6A Modified (n={len(pos_np)})', color='#F44336', density=True)
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Density')
ax.set_title('B) Nanopolish: Prediction Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig8_m6a_prediction_distributions.png", bbox_inches='tight')
plt.close()
print("Saved fig8_m6a_prediction_distributions.png")

# Figure 9: Scatter plot comparing Uncalled4 vs Nanopolish predictions
fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter(y_np, y_uc4, c=y_true, cmap='RdYlBu_r', alpha=0.3, s=10, edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Nanopolish Prediction Probability')
ax.set_ylabel('Uncalled4 Prediction Probability')
ax.set_title('Prediction Probability: Uncalled4 vs Nanopolish')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Ground Truth Label')
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig9_m6a_scatter_comparison.png", bbox_inches='tight')
plt.close()
print("Saved fig9_m6a_scatter_comparison.png")

# Figure 10: Sensitivity at fixed precision thresholds
precision_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def get_recall_at_precision(precision_arr, recall_arr, target_precision):
    """Get the maximum recall achievable at or above a given precision"""
    valid = precision_arr >= target_precision
    if valid.any():
        return recall_arr[valid].max()
    return 0.0

recall_uc4_at_prec = [get_recall_at_precision(precision_uc4, recall_uc4, p) for p in precision_levels]
recall_np_at_prec = [get_recall_at_precision(precision_np, recall_np, p) for p in precision_levels]

sensitivity_df = pd.DataFrame({
    'Precision Threshold': precision_levels,
    'Uncalled4 Recall': recall_uc4_at_prec,
    'Nanopolish Recall': recall_np_at_prec,
})
sensitivity_df['Improvement'] = sensitivity_df['Uncalled4 Recall'] - sensitivity_df['Nanopolish Recall']
print("\nSensitivity at fixed precision thresholds:")
print(sensitivity_df.to_string(index=False))
sensitivity_df.to_csv(f"{OUTPUT_DIR}/sensitivity_at_precision.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(precision_levels))
width = 0.35
ax.bar(x - width/2, recall_uc4_at_prec, width, label='Uncalled4', color='#2196F3', edgecolor='white')
ax.bar(x + width/2, recall_np_at_prec, width, label='Nanopolish', color='#4CAF50', edgecolor='white')
ax.set_xlabel('Precision Threshold')
ax.set_ylabel('Recall (Sensitivity)')
ax.set_title('Sensitivity at Fixed Precision Thresholds')
ax.set_xticks(x)
ax.set_xticklabels([f'{p:.2f}' for p in precision_levels])
ax.legend()
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig10_sensitivity_at_precision.png", bbox_inches='tight')
plt.close()
print("Saved fig10_sensitivity_at_precision.png")

# ============================================================
# Additional: DNA r10.4.1 vs r9.4.1 comparison (9-mer vs 6-mer)
# ============================================================
print("\n=== DNA Chemistry Comparison ===")

# For DNA r10.4.1, analyze the central 6 bases of 9-mers to compare with r9.4.1 6-mers
dna_r10['central_6mer'] = dna_r10['kmer'].apply(lambda x: x[1:7])
central6_mean = dna_r10.groupby('central_6mer')['current_mean'].mean().reset_index()
central6_mean.columns = ['kmer', 'r10_central_mean']

# Merge with r9 data
dna_compare = dna_r9[['kmer', 'current_mean']].merge(central6_mean, on='kmer', how='inner')
dna_compare.columns = ['kmer', 'r9_mean', 'r10_central_mean']

correlation = dna_compare['r9_mean'].corr(dna_compare['r10_central_mean'])
print(f"Correlation between r9.4.1 6-mer current and r10.4.1 central 6-mer current: {correlation:.4f}")

# Figure 11: Cross-chemistry correlation
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(dna_compare['r9_mean'], dna_compare['r10_central_mean'], alpha=0.1, s=5, color='#2196F3')
# Add regression line
z = np.polyfit(dna_compare['r9_mean'], dna_compare['r10_central_mean'], 1)
p = np.poly1d(z)
x_line = np.linspace(dna_compare['r9_mean'].min(), dna_compare['r9_mean'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
ax.set_xlabel('DNA r9.4.1 6-mer Current (pA)')
ax.set_ylabel('DNA r10.4.1 Central 6-mer Current (pA)')
ax.set_title(f'Cross-Chemistry Current Correlation (r = {correlation:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/fig11_cross_chemistry_correlation.png", bbox_inches='tight')
plt.close()
print("Saved fig11_cross_chemistry_correlation.png")

# Save cross-chemistry comparison
dna_compare.to_csv(f"{OUTPUT_DIR}/cross_chemistry_comparison.csv", index=False)

# ============================================================
# Summary statistics for all pore models
# ============================================================
all_pore_summary = pd.DataFrame([
    {'Chemistry': 'DNA r9.4.1', 'K-mer Length': 6, 'N K-mers': len(dna_r9),
     'Mean Current': dna_r9['current_mean'].mean(), 'Std Current': dna_r9['current_mean'].std(),
     'Mean Signal Std': dna_r9['current_std'].mean(), 'Mean Dwell Time': dna_r9['dwell_time'].mean()},
    {'Chemistry': 'DNA r10.4.1', 'K-mer Length': 9, 'N K-mers': len(dna_r10),
     'Mean Current': dna_r10['current_mean'].mean(), 'Std Current': dna_r10['current_mean'].std(),
     'Mean Signal Std': dna_r10['current_std'].mean(), 'Mean Dwell Time': dna_r10['dwell_time'].mean()},
    {'Chemistry': 'RNA001', 'K-mer Length': 5, 'N K-mers': len(rna_r9),
     'Mean Current': rna_r9['current_mean'].mean(), 'Std Current': rna_r9['current_mean'].std(),
     'Mean Signal Std': rna_r9['current_std'].mean(), 'Mean Dwell Time': rna_r9['dwell_time'].mean()},
    {'Chemistry': 'RNA004', 'K-mer Length': 9, 'N K-mers': len(rna004),
     'Mean Current': rna004['current_mean'].mean(), 'Std Current': rna004['current_mean'].std(),
     'Mean Signal Std': rna004['current_std'].mean(), 'Mean Dwell Time': rna004['dwell_time'].mean()},
])
all_pore_summary.to_csv(f"{OUTPUT_DIR}/pore_model_summary_table.csv", index=False)
print("\nPore Model Summary:")
print(all_pore_summary.to_string(index=False))

print("\n=== All analyses complete ===")
