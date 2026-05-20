#!/usr/bin/env python3
"""
Comprehensive analysis of Uncalled4 nanopore signal alignment toolkit.

Analyses:
1. Pore model comparison across chemistries (DNA r9.4.1, DNA r10.4.1, RNA001, RNA004)
2. Performance benchmarks (alignment time, file size)
3. m6A modification detection comparison (Uncalled4 vs Nanopolish)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
REPORT_IMG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

print("=" * 60)
print("UNCALLED4 COMPREHENSIVE ANALYSIS")
print("=" * 60)

# ========================================================================
# 1. LOAD PORE MODEL DATA
# ========================================================================
print("\n[1] Loading pore model data...")

dna_r9 = pd.read_csv(f"{DATA_DIR}/dna_r9.4.1_400bps_6mer_uncalled4.csv")
dna_r10 = pd.read_csv(f"{DATA_DIR}/dna_r10.4.1_400bps_9mer_uncalled4.csv")
rna_r9 = pd.read_csv(f"{DATA_DIR}/rna_r9.4.1_70bps_5mer_uncalled4.csv")
rna004 = pd.read_csv(f"{DATA_DIR}/rna004_130bps_9mer_uncalled4.csv")

print(f"  DNA r9.4.1 (6-mer): {len(dna_r9):,} k-mers")
print(f"  DNA r10.4.1 (9-mer): {len(dna_r10):,} k-mers")
print(f"  RNA r9.4.1 (5-mer): {len(rna_r9):,} k-mers")
print(f"  RNA004 (9-mer): {len(rna004):,} k-mers")

# ========================================================================
# 2. PORE MODEL STATISTICS
# ========================================================================
print("\n[2] Computing pore model statistics...")

pore_stats = []
for name, df in [("DNA r9.4.1", dna_r9), ("DNA r10.4.1", dna_r10), ("RNA001", rna_r9), ("RNA004", rna004)]:
    stats = {
        "Chemistry": name,
        "kmer_size": len(df['kmer'].iloc[0]),
        "num_kmers": len(df),
        "mean_current_mean": df['current_mean'].mean(),
        "std_current_mean": df['current_mean'].std(),
        "mean_current_std": df['current_std'].mean(),
        "mean_dwell_time": df['dwell_time'].mean(),
        "min_current": df['current_mean'].min(),
        "max_current": df['current_mean'].max(),
        "current_range": df['current_mean'].max() - df['current_mean'].min(),
    }
    pore_stats.append(stats)
    print(f"  {name}: mean={stats['mean_current_mean']:.3f}, std={stats['mean_current_std']:.3f}, "
          f"range={stats['current_range']:.3f}, dwell={stats['mean_dwell_time']:.1f}")

pore_stats_df = pd.DataFrame(pore_stats)
pore_stats_df.to_csv(f"{OUTPUT_DIR}/pore_model_statistics.csv", index=False)

# ========================================================================
# 3. PORE MODEL VISUALIZATIONS
# ========================================================================
print("\n[3] Generating pore model visualizations...")

# Figure 1: Current mean distributions across chemistries
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

chemistries = [("DNA r9.4.1", dna_r9), ("DNA r10.4.1", dna_r10), ("RNA001", rna_r9), ("RNA004", rna004)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (name, df) in enumerate(chemistries):
    ax = axes[i]
    ax.hist(df['current_mean'], bins=50, color=colors[i], edgecolor='black', alpha=0.7)
    ax.axvline(df['current_mean'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["current_mean"].mean():.2f}')
    ax.set_xlabel('Normalized Current Mean', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{name} ({len(df.iloc[0]["kmer"])}-mer)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

fig.suptitle('K-mer Current Mean Distributions Across Chemistries', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{REPORT_IMG_DIR}/figure1_current_distributions.png", bbox_inches='tight')
plt.close()
print("  Saved: figure1_current_distributions.png")

# Figure 2: Current std vs mean scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, df) in enumerate(chemistries):
    ax = axes[i]
    # Sample for performance if large
    if len(df) > 5000:
        df_sample = df.sample(5000, random_state=42)
    else:
        df_sample = df
    scatter = ax.scatter(df_sample['current_mean'], df_sample['current_std'], c=df_sample['dwell_time'], 
                         cmap='viridis', s=5, alpha=0.6)
    ax.set_xlabel('Current Mean', fontsize=11)
    ax.set_ylabel('Current Std Dev', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dwell Time', fontsize=9)

fig.suptitle('Current Standard Deviation vs Mean (colored by Dwell Time)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{REPORT_IMG_DIR}/figure2_std_vs_mean.png", bbox_inches='tight')
plt.close()
print("  Saved: figure2_std_vs_mean.png")

# Figure 3: Comparison of current ranges and means across chemistries
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot of mean current
ax1 = axes[0]
bars = ax1.bar([s['Chemistry'] for s in pore_stats], [s['mean_current_mean'] for s in pore_stats], 
               color=colors, edgecolor='black', alpha=0.8)
ax1.set_ylabel('Mean Current', fontsize=12)
ax1.set_title('Mean K-mer Current by Chemistry', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, [s['mean_current_mean'] for s in pore_stats]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Bar plot of current range
ax2 = axes[1]
bars = ax2.bar([s['Chemistry'] for s in pore_stats], [s['current_range'] for s in pore_stats], 
               color=colors, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Current Range (max - min)', fontsize=12)
ax2.set_title('K-mer Current Range by Chemistry', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, [s['current_range'] for s in pore_stats]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{REPORT_IMG_DIR}/figure3_chemistry_comparison.png")
plt.close()
print("  Saved: figure3_chemistry_comparison.png")

# ========================================================================
# 4. PERFORMANCE BENCHMARKS
# ========================================================================
print("\n[4] Analyzing performance benchmarks...")

perf = pd.read_csv(f"{DATA_DIR}/performance_summary.csv")
print(perf.to_string(index=False))

# Calculate speedups
speedups = []
for chem in perf['Chemistry'].unique():
    chem_data = perf[perf['Chemistry'] == chem]
    uncalled4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
    if len(uncalled4_time) == 0:
        continue
    uncalled4_time = uncalled4_time[0]
    for _, row in chem_data.iterrows():
        if row['Tool'] != 'Uncalled4' and pd.notna(row['Time_min']):
            speedups.append({
                'Chemistry': chem,
                'Tool': row['Tool'],
                'Uncalled4_time': uncalled4_time,
                'Tool_time': row['Time_min'],
                'Speedup': row['Time_min'] / uncalled4_time,
                'FileSize_MB': row['FileSize_MB'],
                'Uncalled4_FileSize_MB': chem_data[chem_data['Tool'] == 'Uncalled4']['FileSize_MB'].values[0]
            })

speedup_df = pd.DataFrame(speedups)
if len(speedup_df) > 0:
    speedup_df.to_csv(f"{OUTPUT_DIR}/speedup_analysis.csv", index=False)
    print(f"\n  Speedup summary saved to outputs/speedup_analysis.csv")
    print(speedup_df[['Chemistry', 'Tool', 'Speedup']].to_string(index=False))

# Figure 4: Performance benchmark bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Time comparison
ax1 = axes[0]
tools = perf['Tool'].unique()
chems = perf['Chemistry'].unique()
x = np.arange(len(chems))
width = 0.2
tool_colors = {'Uncalled4': '#2ca02c', 'f5c': '#1f77b4', 'Nanopolish': '#ff7f0e', 'Tombo': '#d62728'}

for i, tool in enumerate(tools):
    times = []
    for chem in chems:
        t = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == tool)]['Time_min'].values
        times.append(t[0] if len(t) > 0 else np.nan)
    ax1.bar(x + i*width, times, width, label=tool, color=tool_colors.get(tool, 'gray'), edgecolor='black')

ax1.set_xlabel('Chemistry', fontsize=12)
ax1.set_ylabel('Alignment Time (minutes)', fontsize=12)
ax1.set_title('Alignment Time Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(chems, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.set_yscale('log')

# File size comparison
ax2 = axes[1]
for i, tool in enumerate(tools):
    sizes = []
    for chem in chems:
        s = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == tool)]['FileSize_MB'].values
        sizes.append(s[0] if len(s) > 0 else np.nan)
    ax2.bar(x + i*width, sizes, width, label=tool, color=tool_colors.get(tool, 'gray'), edgecolor='black')

ax2.set_xlabel('Chemistry', fontsize=12)
ax2.set_ylabel('File Size (MB)', fontsize=12)
ax2.set_title('Output File Size Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(chems, rotation=15, ha='right')
ax2.legend(fontsize=10)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(f"{REPORT_IMG_DIR}/figure4_performance_benchmarks.png")
plt.close()
print("  Saved: figure4_performance_benchmarks.png")

# ========================================================================
# 5. M6A MODIFICATION DETECTION ANALYSIS
# ========================================================================
print("\n[5] Analyzing m6A modification detection...")

labels = pd.read_csv(f"{DATA_DIR}/m6a_labels.csv")
uncalled4_pred = pd.read_csv(f"{DATA_DIR}/m6a_predictions_uncalled4.csv")
nanopolish_pred = pd.read_csv(f"{DATA_DIR}/m6a_predictions_nanopolish.csv")

# Merge data
m6a_data = labels.merge(uncalled4_pred, on='site_id').merge(nanopolish_pred, on='site_id', suffixes=('_uncalled4', '_nanopolish'))
print(f"  Total sites: {len(m6a_data):,}")
print(f"  Positive sites (m6A): {m6a_data['label'].sum():,} ({100*m6a_data['label'].mean():.1f}%)")

# Compute PR and ROC curves
y_true = m6a_data['label'].values
y_uncalled4 = m6a_data['probability_uncalled4'].values
y_nanopolish = m6a_data['probability_nanopolish'].values

# Precision-Recall
precision_u, recall_u, _ = precision_recall_curve(y_true, y_uncalled4)
precision_n, recall_n, _ = precision_recall_curve(y_true, y_nanopolish)

ap_u = average_precision_score(y_true, y_uncalled4)
ap_n = average_precision_score(y_true, y_nanopolish)

# ROC
fpr_u, tpr_u, _ = roc_curve(y_true, y_uncalled4)
fpr_n, tpr_n, _ = roc_curve(y_true, y_nanopolish)

auc_u = auc(fpr_u, tpr_u)
auc_n = auc(fpr_n, tpr_n)

print(f"\n  Uncalled4: AUC-PR = {ap_u:.4f}, AUC-ROC = {auc_u:.4f}")
print(f"  Nanopolish: AUC-PR = {ap_n:.4f}, AUC-ROC = {auc_n:.4f}")

# Save metrics
metrics = {
    "uncalled4": {"auc_pr": float(ap_u), "auc_roc": float(auc_u)},
    "nanopolish": {"auc_pr": float(ap_n), "auc_roc": float(auc_n)},
    "num_sites": len(m6a_data),
    "num_positive": int(m6a_data['label'].sum()),
    "num_negative": int((m6a_data['label'] == 0).sum()),
}
with open(f"{OUTPUT_DIR}/m6a_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Figure 5: Precision-Recall curves
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(recall_u, precision_u, color='#2ca02c', linewidth=2.5, label=f'Uncalled4 (AP = {ap_u:.3f})')
ax.plot(recall_n, precision_n, color='#ff7f0e', linewidth=2.5, label=f'Nanopolish (AP = {ap_n:.3f})')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve: m6A Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG_DIR}/figure5_precision_recall.png")
plt.close()
print("  Saved: figure5_precision_recall.png")

# Figure 6: ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_u, tpr_u, color='#2ca02c', linewidth=2.5, label=f'Uncalled4 (AUC = {auc_u:.3f})')
ax.plot(fpr_n, tpr_n, color='#ff7f0e', linewidth=2.5, label=f'Nanopolish (AUC = {auc_n:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve: m6A Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG_DIR}/figure6_roc_curve.png")
plt.close()
print("  Saved: figure6_roc_curve.png")

# ========================================================================
# 6. ADDITIONAL ANALYSES
# ========================================================================
print("\n[6] Additional analyses...")

# Figure 7: Prediction probability distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, probs) in enumerate([("Uncalled4", y_uncalled4), ("Nanopolish", y_nanopolish)]):
    ax = axes[i]
    pos_probs = probs[y_true == 1]
    neg_probs = probs[y_true == 0]
    ax.hist(neg_probs, bins=30, color='steelblue', alpha=0.6, label='Unmodified', edgecolor='black')
    ax.hist(pos_probs, bins=30, color='coral', alpha=0.6, label='m6A', edgecolor='black')
    ax.set_xlabel('Prediction Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

fig.suptitle('m6A Prediction Probability Distributions', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{REPORT_IMG_DIR}/figure7_probability_distributions.png")
plt.close()
print("  Saved: figure7_probability_distributions.png")

# Figure 8: Speedup heatmap
if len(speedup_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = speedup_df.pivot(index='Tool', columns='Chemistry', values='Speedup')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=1, 
                cbar_kws={'label': 'Speedup (x times)'}, ax=ax)
    ax.set_title('Uncalled4 Speedup Factor Over Competitors', fontsize=13, fontweight='bold')
    ax.set_xlabel('Chemistry', fontsize=12)
    ax.set_ylabel('Tool', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{REPORT_IMG_DIR}/figure8_speedup_heatmap.png")
    plt.close()
    print("  Saved: figure8_speedup_heatmap.png")

# ========================================================================
# 7. SUMMARY STATISTICS EXPORT
# ========================================================================
print("\n[7] Exporting summary statistics...")

summary = {
    "pore_models": pore_stats,
    "performance": {
        "speedups": speedups,
    },
    "m6a_detection": metrics,
}

with open(f"{OUTPUT_DIR}/summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Outputs saved to: {OUTPUT_DIR}/")
print(f"Figures saved to: {REPORT_IMG_DIR}/")
