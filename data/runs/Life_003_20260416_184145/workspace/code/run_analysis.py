#!/usr/bin/env python3
"""
Uncalled4 Nanopore Signal Alignment and m6A Detection Analysis
Modular version for better execution control
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

WORKSPACE = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_003_20260416_184145'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("Uncalled4 Analysis Pipeline")
print("=" * 60)

# Step 1: Load Data
print("\n[1] Loading data...")
perf_df = pd.read_csv(os.path.join(DATA_DIR, 'performance_summary.csv'))
uncalled4_pred = pd.read_csv(os.path.join(DATA_DIR, 'm6a_predictions_uncalled4.csv'))
nanopolish_pred = pd.read_csv(os.path.join(DATA_DIR, 'm6a_predictions_nanopolish.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, 'm6a_labels.csv'))

dna_r9 = pd.read_csv(os.path.join(DATA_DIR, 'dna_r9.4.1_400bps_6mer_uncalled4.csv'))
dna_r10 = pd.read_csv(os.path.join(DATA_DIR, 'dna_r10.4.1_400bps_9mer_uncalled4.csv'))
rna_001 = pd.read_csv(os.path.join(DATA_DIR, 'rna_r9.4.1_70bps_5mer_uncalled4.csv'))
rna_004 = pd.read_csv(os.path.join(DATA_DIR, 'rna004_130bps_9mer_uncalled4.csv'))

print(f"  Performance: {len(perf_df)} rows")
print(f"  Predictions: {len(uncalled4_pred)} sites")
print(f"  Pore models: r9={len(dna_r9)}, r10={len(dna_r10)}, rna001={len(rna_001)}, rna004={len(rna_004)}")

# Step 2: Merge predictions
print("\n[2] Merging predictions with labels...")
merged = labels.merge(uncalled4_pred, on='site_id')
merged = merged.merge(nanopolish_pred, on='site_id', suffixes=('_uncalled4', '_nanopolish'))
y_true = merged['label'].values
y_prob_u = merged['probability_uncalled4'].values
y_prob_n = merged['probability_nanopolish'].values
print(f"  Positive: {y_true.sum()}, Negative: {(y_true==0).sum()}")

# Step 3: Performance benchmarks
print("\n[3] Processing performance benchmarks...")
tools = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']
chemistries = perf_df['Chemistry'].unique()

speedup_data = []
for chem in chemistries:
    chem_data = perf_df[perf_df['Chemistry'] == chem]
    u_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values[0]
    u_size = chem_data[chem_data['Tool'] == 'Uncalled4']['FileSize_MB'].values[0]
    
    for tool in tools:
        td = chem_data[chem_data['Tool'] == tool]
        if len(td) > 0 and pd.notna(td['Time_min'].values[0]):
            t = td['Time_min'].values[0]
            s = td['FileSize_MB'].values[0]
            speedup_data.append({
                'Chemistry': chem, 'Tool': tool, 'Time_min': t, 'FileSize_MB': s,
                'Speedup': t/u_time if u_time > 0 else np.nan,
                'SizeRatio': s/u_size if u_size > 0 else np.nan
            })
speedup_df = pd.DataFrame(speedup_data)

with open(os.path.join(OUTPUTS_DIR, "performance_comparison.json"), 'w') as f:
    json.dump({'benchmarks': perf_df.to_dict('records'), 'speedup': speedup_df.to_dict('records')}, f, indent=2)
print(f"  Saved performance_comparison.json")

# Step 4: Performance figures
print("\n[4] Generating performance figures...")
colors = {'Uncalled4': '#2ecc71', 'f5c': '#3498db', 'Nanopolish': '#e74c3c', 'Tombo': '#9b59b6'}

# Time comparison
fig, ax = plt.subplots(figsize=(12, 6))
for i, chem in enumerate(chemistries):
    cd = perf_df[perf_df['Chemistry'] == chem]
    bars = ax.bar(np.arange(len(cd)) + i*0.9, cd['Time_min'], 
                  color=[colors.get(t, '#95a5a6') for t in cd['Tool']], label=chem, alpha=0.8)
ax.set_xlabel('Tool')
ax.set_ylabel('Alignment Time (minutes)')
ax.set_title('Uncalled4: Alignment Time Comparison')
ax.set_xticks(np.arange(len(tools)) + 1.35)
ax.set_xticklabels(tools, rotation=45, ha='right')
ax.legend(title='Chemistry')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "performance_time.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved performance_time.png")

# File size comparison
fig, ax = plt.subplots(figsize=(12, 6))
for i, chem in enumerate(chemistries):
    cd = perf_df[perf_df['Chemistry'] == chem]
    vd = cd[pd.notna(cd['FileSize_MB'])]
    if len(vd) > 0:
        ax.bar(np.arange(len(vd)) + i*0.9, vd['FileSize_MB'], 
               color=[colors.get(t, '#95a5a6') for t in vd['Tool']], label=chem, alpha=0.8)
ax.set_xlabel('Tool')
ax.set_ylabel('File Size (MB)')
ax.set_title('Uncalled4: Output File Size Comparison')
ax.set_xticks(np.arange(len(tools)) + 1.35)
ax.set_xticklabels(tools, rotation=45, ha='right')
ax.legend(title='Chemistry')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "performance_filesize.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved performance_filesize.png")

# Speedup heatmap
pivot = speedup_df.pivot(index='Chemistry', columns='Tool', values='Speedup')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Speedup vs Uncalled4'})
ax.set_title('Speedup Factor Relative to Uncalled4')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "speedup_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved speedup_heatmap.png")

# Step 5: m6A detection analysis
print("\n[5] Computing PR and ROC curves...")
pr_u = precision_recall_curve(y_true, y_prob_u)
pr_n = precision_recall_curve(y_true, y_prob_n)
fpr_u, tpr_u, _ = roc_curve(y_true, y_prob_u)
fpr_n, tpr_n, _ = roc_curve(y_true, y_prob_n)

auc_u = auc(fpr_u, tpr_u)
auc_n = auc(fpr_n, tpr_n)
ap_u = average_precision_score(y_true, y_prob_u)
ap_n = average_precision_score(y_true, y_prob_n)

print(f"  Uncalled4: AUC={auc_u:.4f}, AP={ap_u:.4f}")
print(f"  Nanopolish: AUC={auc_n:.4f}, AP={ap_n:.4f}")

with open(os.path.join(OUTPUTS_DIR, "pr_roc_analysis.json"), 'w') as f:
    json.dump({
        'uncalled4': {'auc': auc_u, 'ap': ap_u, 'fpr': fpr_u.tolist(), 'tpr': tpr_u.tolist()},
        'nanopolish': {'auc': auc_n, 'ap': ap_n, 'fpr': fpr_n.tolist(), 'tpr': tpr_n.tolist()},
        'summary': {'total': len(merged), 'positive': int(y_true.sum()), 'negative': int((y_true==0).sum()),
                    'auc_delta': auc_u - auc_n, 'ap_delta': ap_u - ap_n}
    }, f, indent=2)
print("  Saved pr_roc_analysis.json")

# Step 6: m6A figures
print("\n[6] Generating m6A detection figures...")

# PR curves
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(pr_u[1], pr_u[0], 'b-', lw=2.5, label=f'Uncalled4 (AP={ap_u:.3f})')
ax.plot(pr_n[1], pr_n[0], 'r--', lw=2.5, label=f'Nanopolish (AP={ap_n:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('m6A Detection: Precision-Recall Curves')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pr_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pr_curves.png")

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_u, tpr_u, 'b-', lw=2.5, label=f'Uncalled4 (AUC={auc_u:.3f})')
ax.plot(fpr_n, tpr_n, 'r--', lw=2.5, label=f'Nanopolish (AUC={auc_n:.3f})')
ax.plot([0,1], [0,1], 'k:', lw=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('m6A Detection: ROC Curves')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved roc_curves.png")

# Prediction distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(merged[merged['label']==1]['probability_uncalled4'], bins=50, alpha=0.6, color='red', label='Positive', density=True)
axes[0].hist(merged[merged['label']==0]['probability_uncalled4'], bins=50, alpha=0.6, color='blue', label='Negative', density=True)
axes[0].set_xlabel('Prediction Probability')
axes[0].set_ylabel('Density')
axes[0].set_title('Uncalled4 Score Distribution')
axes[0].legend()
axes[0].axvline(0.5, color='black', ls='--', alpha=0.5)

axes[1].hist(merged[merged['label']==1]['probability_nanopolish'], bins=50, alpha=0.6, color='red', label='Positive', density=True)
axes[1].hist(merged[merged['label']==0]['probability_nanopolish'], bins=50, alpha=0.6, color='blue', label='Negative', density=True)
axes[1].set_xlabel('Prediction Probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Nanopolish Score Distribution')
axes[1].legend()
axes[1].axvline(0.5, color='black', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "prediction_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved prediction_distributions.png")

# Step 7: Pore model analysis
print("\n[7] Analyzing pore models...")
pore_stats = {}
for name, df in [('DNA_r9.4.1_6mer', dna_r9), ('DNA_r10.4.1_9mer', dna_r10),
                  ('RNA001_5mer', rna_001), ('RNA004_9mer', rna_004)]:
    stats = {
        'n_kmers': len(df), 'kmer_length': len(df['kmer'].iloc[0]),
        'current_mean_mean': float(df['current_mean'].mean()),
        'current_mean_std': float(df['current_mean'].std()),
        'current_mean_min': float(df['current_mean'].min()),
        'current_mean_max': float(df['current_mean'].max()),
        'dwell_time_mean': float(df['dwell_time'].mean()),
        'dwell_time_median': float(df['dwell_time'].median())
    }
    pore_stats[name] = stats
    print(f"  {name}: n={stats['n_kmers']}, mean_current={stats['current_mean_mean']:.3f}")

with open(os.path.join(OUTPUTS_DIR, "pore_model_stats.json"), 'w') as f:
    json.dump(pore_stats, f, indent=2)
print("  Saved pore_model_stats.json")

# Step 8: Pore model figures
print("\n[8] Generating pore model figures...")
pore_models = [
    ('DNA r9.4.1 (6-mer)', dna_r9, '#3498db'),
    ('DNA r10.4.1 (9-mer)', dna_r10, '#2ecc71'),
    ('RNA001 (5-mer)', rna_001, '#e74c3c'),
    ('RNA004 (9-mer)', rna_004, '#9b59b6')
]

# Current distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (title, df, color) in enumerate(pore_models):
    ax = axes[idx//2, idx%2]
    ax.hist(df['current_mean'], bins=100, color=color, alpha=0.7, density=True)
    ax.axvline(df['current_mean'].mean(), color='black', ls='--', lw=2, label=f"Mean: {df['current_mean'].mean():.3f}")
    ax.set_xlabel('Current Mean (pA)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nn={len(df)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pore_current_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pore_current_dist.png")

# Dwell time distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (title, df, color) in enumerate(pore_models):
    ax = axes[idx//2, idx%2]
    ax.hist(df['dwell_time'], bins=50, color=color, alpha=0.7)
    ax.axvline(df['dwell_time'].median(), color='black', ls='--', lw=2, label=f"Median: {df['dwell_time'].median():.1f}")
    ax.set_xlabel('Dwell Time')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pore_dwell_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pore_dwell_dist.png")

# Current std vs mean
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (title, df, color) in enumerate(pore_models):
    ax = axes[idx//2, idx%2]
    sample = df.sample(min(10000, len(df)), random_state=42)
    ax.scatter(sample['current_mean'], sample['current_std'], alpha=0.3, s=10, color=color)
    ax.set_xlabel('Current Mean (pA)')
    ax.set_ylabel('Current Std (pA)')
    ax.set_title(f'{title}\nCorr: {sample["current_mean"].corr(sample["current_std"]):.3f}')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "pore_std_vs_mean.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pore_std_vs_mean.png")

# Step 9: Summary
print("\n[9] Creating summary...")
summary = {
    'analysis_date': '2026-04-16',
    'performance': {
        'tools': tools,
        'chemistries': list(chemistries),
        'speedup_range': f"{speedup_df['Speedup'].min():.1f}x - {speedup_df['Speedup'].max():.1f}x"
    },
    'm6a_detection': {
        'total_sites': len(merged),
        'positive': int(y_true.sum()),
        'uncalled4_auc': round(auc_u, 4),
        'uncalled4_ap': round(ap_u, 4),
        'nanopolish_auc': round(auc_n, 4),
        'nanopolish_ap': round(ap_n, 4),
        'auc_improvement': round(auc_u - auc_n, 4),
        'ap_improvement': round(ap_u - ap_n, 4)
    },
    'pore_models': list(pore_stats.keys()),
    'figures': [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')],
    'outputs': [f for f in os.listdir(OUTPUTS_DIR) if f.endswith('.json')]
}

with open(os.path.join(OUTPUTS_DIR, "analysis_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved analysis_summary.json")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print(f"Outputs: {OUTPUTS_DIR}")
print(f"Figures: {IMAGES_DIR}")
print(f"\nKey Results:")
print(f"  Uncalled4 AUC: {auc_u:.4f} vs Nanopolish AUC: {auc_n:.4f}")
print(f"  Uncalled4 AP: {ap_u:.4f} vs Nanopolish AP: {ap_n:.4f}")
print(f"  Figures generated: {len(summary['figures'])}")
