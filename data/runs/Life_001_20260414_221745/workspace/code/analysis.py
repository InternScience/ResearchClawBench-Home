#!/usr/bin/env python3
"""
Personalized Neoantigen Vaccine Optimization Analysis
=====================================================
Analyzes simulated neoantigen vaccine optimization data to compute:
1. Per-cell immune response probability distributions
2. Coverage ratio of tumor cells
3. IoU of optimal vaccine compositions across replicates
4. Optimization runtime scaling
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.2)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

cell_pop = pd.read_csv(f'{DATA_DIR}/cell-populations.csv')
final_resp = pd.read_csv(f'{DATA_DIR}/final-response-likelihoods.csv')
runtime = pd.read_csv(f'{DATA_DIR}/optimization_runtime_data.csv')
selected_elements = pd.read_csv(f'{DATA_DIR}/selected-vaccine-elements.budget-10.minsum.adaptive.csv')
sim_resp = pd.read_csv(f'{DATA_DIR}/sim-specific-response-likelihoods.csv')
vaccine_simple = pd.read_csv(f'{DATA_DIR}/vaccine.budget-10.minsum.adaptive.csv')

# Load all 10 replicate score files
rep_dfs = {}
for i in range(10):
    path = f'{DATA_DIR}/vaccine-elements.scores.100-cells.10x.rep-{i}.csv'
    rep_dfs[i] = pd.read_csv(path)
    print(f"  Loaded rep-{i}: {len(rep_dfs[i])} rows")

print(f"\ncell-populations.csv: {cell_pop.shape}")
print(f"final-response-likelihoods.csv: {final_resp.shape}")
print(f"optimization_runtime_data.csv: {runtime.shape}")
print(f"selected-vaccine-elements: {selected_elements.shape}")
print(f"sim-specific-response-likelihoods: {sim_resp.shape}")
print(f"vaccine.budget-10: {vaccine_simple.shape}")

# ============================================================
# 2. DATA OVERVIEW
# ============================================================
print("\n" + "=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

# Cell populations summary
print("\nCell populations:")
print(f"  Repetitions: {cell_pop['repetition'].nunique()}")
print(f"  Unique cells per rep: {cell_pop.groupby('repetition')['cell_ids'].nunique().values[:5]}...")
print(f"  Unique peptides: {cell_pop['presented_peptides'].nunique()}")
print(f"  Unique mutations: {cell_pop['mutation'].nunique()}")
print(f"  Simulation names: {cell_pop['simulation_name'].unique()}")

# Selected vaccine elements
print("\nSelected vaccine elements (budget=10, MinSum, adaptive):")
selected_muts = selected_elements['peptide'].unique()
print(f"  Unique mutations: {len(selected_muts)} -> {sorted(selected_muts)}")
print(f"  Repetitions covered: {selected_elements['repetition'].nunique()}")

# Response likelihoods overview
print("\nFinal response likelihoods:")
print(f"  Mean p_response: {final_resp['p_response'].mean():.4f}")
print(f"  Median p_response: {final_resp['p_response'].median():.4f}")
print(f"  Min p_response: {final_resp['p_response'].min():.4f}")
print(f"  Max p_response: {final_resp['p_response'].max():.4f}")

# ============================================================
# 3. PER-CELL RESPONSE PROBABILITY DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: Per-cell response probability distributions")
print("=" * 60)

# From final-response-likelihoods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of p_response
axes[0].hist(final_resp['p_response'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Per-cell Immune Response Probability (p_response)')
axes[0].set_ylabel('Number of Cells')
axes[0].set_title('Distribution of Per-Cell Response Probabilities\n(MinSum Budget=10, Adaptive)')
axes[0].axvline(final_resp['p_response'].mean(), color='red', linestyle='--', label=f'Mean={final_resp["p_response"].mean():.3f}')
axes[0].axvline(final_resp['p_response'].median(), color='orange', linestyle='--', label=f'Median={final_resp["p_response"].median():.3f}')
axes[0].legend()

# ECDF
sorted_p = np.sort(final_resp['p_response'])
ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
axes[1].plot(sorted_p, ecdf, color='steelblue', linewidth=2)
axes[1].set_xlabel('Per-cell Immune Response Probability (p_response)')
axes[1].set_ylabel('Cumulative Fraction')
axes[1].set_title('ECDF of Per-Cell Response Probabilities')
axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig1_response_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_response_distribution.png")

# Save summary stats
response_stats = {
    'mean_p_response': float(final_resp['p_response'].mean()),
    'median_p_response': float(final_resp['p_response'].median()),
    'std_p_response': float(final_resp['p_response'].std()),
    'min_p_response': float(final_resp['p_response'].min()),
    'max_p_response': float(final_resp['p_response'].max()),
    'q25_p_response': float(final_resp['p_response'].quantile(0.25)),
    'q75_p_response': float(final_resp['p_response'].quantile(0.75)),
    'n_cells': int(len(final_resp))
}
import json
with open(f'{OUTPUT_DIR}/response_stats.json', 'w') as f:
    json.dump(response_stats, f, indent=2)
print(f"  Response stats: mean={response_stats['mean_p_response']:.4f}, median={response_stats['median_p_response']:.4f}")

# ============================================================
# 4. COVERAGE RATIO ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Coverage ratio of tumor cells")
print("=" * 60)

# Coverage = fraction of cells with p_response >= threshold
thresholds = np.arange(0.0, 1.01, 0.05)
coverage_ratios = []
for t in thresholds:
    cov = (final_resp['p_response'] >= t).mean()
    coverage_ratios.append({'threshold': t, 'coverage': cov})

coverage_df = pd.DataFrame(coverage_ratios)
coverage_df.to_csv(f'{OUTPUT_DIR}/coverage_ratios.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(coverage_df['threshold'], coverage_df['coverage'], 'o-', color='darkgreen', linewidth=2, markersize=4)
ax.set_xlabel('Response Probability Threshold')
ax.set_ylabel('Coverage Ratio (Fraction of Cells)')
ax.set_title('Coverage Curve: Fraction of Cells Above Response Threshold')
ax.fill_between(coverage_df['threshold'], coverage_df['coverage'], alpha=0.15, color='green')

# Annotate key thresholds
for t_val in [0.5, 0.8, 0.9, 0.95]:
    cov_val = coverage_df.loc[coverage_df['threshold'].sub(t_val).abs().idxmin(), 'coverage']
    ax.annotate(f'thresh={t_val:.2f}\ncov={cov_val:.3f}',
                xy=(t_val, cov_val), xytext=(t_val+0.05, cov_val-0.05),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='darkgreen')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig2_coverage_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_coverage_curve.png")

# Key coverage values
key_thresholds = [0.5, 0.8, 0.9, 0.95]
coverage_summary = {}
for t in key_thresholds:
    cov = (final_resp['p_response'] >= t).mean()
    coverage_summary[f'coverage_at_thresh_{t}'] = float(cov)
    print(f"  Coverage at threshold={t}: {cov:.4f}")

with open(f'{OUTPUT_DIR}/coverage_summary.json', 'w') as f:
    json.dump(coverage_summary, f, indent=2)

# ============================================================
# 5. IoU OF VACCINE COMPOSITIONS ACROSS REPLICATES
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: IoU of vaccine compositions across replicates")
print("=" * 60)

# Get selected mutations per replicate from selected-vaccine-elements
rep_mut_sets = {}
for rep in selected_elements['repetition'].unique():
    muts = set(selected_elements.loc[selected_elements['repetition'] == rep, 'peptide'].unique())
    rep_mut_sets[rep] = muts
    print(f"  Rep {rep}: {sorted(muts)}")

# Compute pairwise IoU
reps = sorted(rep_mut_sets.keys())
iou_matrix = np.zeros((len(reps), len(reps)))
for i, r1 in enumerate(reps):
    for j, r2 in enumerate(reps):
        intersection = len(rep_mut_sets[r1] & rep_mut_sets[r2])
        union = len(rep_mut_sets[r1] | rep_mut_sets[r2])
        iou_matrix[i, j] = intersection / union if union > 0 else 0

# Mean IoU (off-diagonal)
off_diag_ious = []
for i in range(len(reps)):
    for j in range(i+1, len(reps)):
        off_diag_ious.append(iou_matrix[i, j])
mean_iou = np.mean(off_diag_ious)
std_iou = np.std(off_diag_ious)
print(f"\n  Mean pairwise IoU: {mean_iou:.4f} ± {std_iou:.4f}")

# Also compute IoU from the 10 replicate score files
# Each replicate has cell-level scores; the "selected" elements per rep are the same (budget-10 MinSum)
# Let's check what vaccine elements appear in each replicate score file
rep_score_elements = {}
for i in range(10):
    elements = set(rep_dfs[i]['vaccine_element'].unique())
    rep_score_elements[i] = elements

# IoU across score replicates
score_ious = []
for i in range(10):
    for j in range(i+1, 10):
        inter = len(rep_score_elements[i] & rep_score_elements[j])
        union = len(rep_score_elements[i] | rep_score_elements[j])
        score_ious.append(inter / union if union > 0 else 0)
mean_score_iou = np.mean(score_ious)
print(f"  Mean pairwise IoU (score replicates): {mean_score_iou:.4f}")

# Plot IoU heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(iou_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=[f'Rep {r}' for r in reps],
            yticklabels=[f'Rep {r}' for r in reps], ax=ax)
ax.set_title('IoU of Selected Vaccine Elements Across Replicates\n(MinSum Budget=10, Adaptive)')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig3_iou_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_iou_heatmap.png")

iou_results = {
    'mean_pairwise_iou': float(mean_iou),
    'std_pairwise_iou': float(std_iou),
    'mean_score_iou': float(mean_score_iou),
    'n_replicates': len(reps),
    'selected_mutations_per_rep': {str(k): sorted(list(v)) for k, v in rep_mut_sets.items()}
}
with open(f'{OUTPUT_DIR}/iou_results.json', 'w') as f:
    json.dump(iou_results, f, indent=2)

# ============================================================
# 6. OPTIMIZATION RUNTIME ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Optimization runtime vs population size")
print("=" * 60)

print(f"  Samples: {runtime['SampleID'].nunique()}")
print(f"  Population sizes: {sorted(runtime['PopulationSize'].unique())}")

fig, ax = plt.subplots(figsize=(10, 6))
for sample_id in sorted(runtime['SampleID'].unique()):
    subset = runtime[runtime['SampleID'] == sample_id].sort_values('PopulationSize')
    ax.plot(subset['PopulationSize'], subset['RunTime'], 'o-', label=f'Sample {sample_id}', linewidth=1.5, markersize=6)

ax.set_xlabel('Tumor Cell Population Size')
ax.set_ylabel('Optimization Runtime (seconds)')
ax.set_title('Optimization Runtime vs. Tumor Population Size')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig4_runtime_scaling.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_runtime_scaling.png")

# Runtime summary stats
runtime_summary = runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std', 'min', 'max']).reset_index()
runtime_summary.to_csv(f'{OUTPUT_DIR}/runtime_summary.csv', index=False)
print(f"\n  Runtime summary:\n{runtime_summary.to_string(index=False)}")

# ============================================================
# 7. VACCINE ELEMENT SCORE ANALYSIS (from rep-0)
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 5: Vaccine element score analysis")
print("=" * 60)

rep0 = rep_dfs[0]
# Pivot: for each cell, what's the max p_response across all vaccine elements?
cell_max_response = rep0.groupby('cell_id')['p_response'].max().reset_index()
cell_max_response.columns = ['cell_id', 'max_p_response']

# Per-element average response across cells
element_avg = rep0.groupby('vaccine_element')['p_response'].mean().reset_index()
element_avg.columns = ['vaccine_element', 'mean_p_response']
element_avg = element_avg.sort_values('mean_p_response', ascending=False)

print(f"  Elements ranked by mean p_response:")
for _, row in element_avg.iterrows():
    print(f"    {row['vaccine_element']}: {row['mean_p_response']:.4f}")

# Save element ranking
element_avg.to_csv(f'{OUTPUT_DIR}/element_ranking.csv', index=False)

# Plot element ranking
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['darkred' if e in selected_muts else 'steelblue' for e in element_avg['vaccine_element']]
bars = ax.bar(range(len(element_avg)), element_avg['mean_p_response'], color=colors, edgecolor='white')
ax.set_xticks(range(len(element_avg)))
ax.set_xticklabels(element_avg['vaccine_element'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean Per-Cell Response Probability')
ax.set_title('Vaccine Element Ranking by Mean Response Probability\n(Red = selected in optimal vaccine)')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig5_element_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_element_ranking.png")

# ============================================================
# 8. RESPONSE DISTRIBUTION BY NUMBER OF PRESENTED PEPTIDES
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 6: Response by number of presented peptides")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(final_resp['num_presented_peptides'], final_resp['p_response'], alpha=0.5, s=15, color='steelblue')
ax.set_xlabel('Number of Presented Peptides per Cell')
ax.set_ylabel('Per-Cell Response Probability')
ax.set_title('Response Probability vs. Number of Presented Peptides')

# Add trend line
z = np.polyfit(final_resp['num_presented_peptides'], final_resp['p_response'], 1)
p = np.poly1d(z)
x_line = np.linspace(final_resp['num_presented_peptides'].min(), final_resp['num_presented_peptides'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit (slope={z[0]:.4f})')
ax.legend()

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig6_response_vs_peptides.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_response_vs_peptides.png")

# Correlation
corr = final_resp['num_presented_peptides'].corr(final_resp['p_response'])
print(f"  Correlation (num_peptides vs p_response): {corr:.4f}")

# ============================================================
# 9. CELL POPULATION HETEROGENEITY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 7: Cell population heterogeneity")
print("=" * 60)

# Number of unique peptides per cell
cell_peptide_counts = cell_pop.groupby(['repetition', 'cell_ids'])['presented_peptides'].nunique().reset_index()
cell_peptide_counts.columns = ['repetition', 'cell_id', 'n_unique_peptides']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of peptides per cell
axes[0].hist(cell_peptide_counts['n_unique_peptides'], bins=20, color='coral', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Number of Unique Peptides per Cell')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Unique Peptides per Cell')

# Mutation frequency
mut_counts = cell_pop['mutation'].value_counts()
axes[1].bar(range(len(mut_counts)), mut_counts.values, color='teal', edgecolor='white')
axes[1].set_xticks(range(len(mut_counts)))
axes[1].set_xticklabels(mut_counts.index, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Total Presentation Count')
axes[1].set_title('Mutation Presentation Frequency')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig7_population_heterogeneity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_population_heterogeneity.png")

# ============================================================
# 10. COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 8: Summary dashboard")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Response distribution
axes[0,0].hist(final_resp['p_response'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0,0].set_xlabel('p_response')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('A) Response Probability Distribution')
axes[0,0].axvline(final_resp['p_response'].mean(), color='red', linestyle='--')

# (0,1) Coverage curve
axes[0,1].plot(coverage_df['threshold'], coverage_df['coverage'], 'o-', color='darkgreen', markersize=3)
axes[0,1].set_xlabel('Threshold')
axes[0,1].set_ylabel('Coverage')
axes[0,1].set_title('B) Coverage Curve')
axes[0,1].fill_between(coverage_df['threshold'], coverage_df['coverage'], alpha=0.15, color='green')

# (0,2) IoU heatmap
sns.heatmap(iou_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,2],
            xticklabels=[f'R{r}' for r in reps], yticklabels=[f'R{r}' for r in reps],
            cbar_kws={'shrink': 0.8})
axes[0,2].set_title(f'C) IoU Matrix (mean={mean_iou:.3f})')

# (1,0) Runtime scaling
for sample_id in sorted(runtime['SampleID'].unique()):
    subset = runtime[runtime['SampleID'] == sample_id].sort_values('PopulationSize')
    axes[1,0].plot(subset['PopulationSize'], subset['RunTime'], 'o-', markersize=4, linewidth=1)
axes[1,0].set_xlabel('Population Size')
axes[1,0].set_ylabel('Runtime (s)')
axes[1,0].set_title('D) Runtime Scaling')
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')

# (1,1) Element ranking
colors = ['darkred' if e in selected_muts else 'steelblue' for e in element_avg['vaccine_element']]
axes[1,1].bar(range(len(element_avg)), element_avg['mean_p_response'], color=colors, edgecolor='white')
axes[1,1].set_xticks(range(len(element_avg)))
axes[1,1].set_xticklabels(element_avg['vaccine_element'], rotation=45, ha='right', fontsize=7)
axes[1,1].set_ylabel('Mean p_response')
axes[1,1].set_title('E) Element Ranking (red=selected)')

# (1,2) Response vs peptides
axes[1,2].scatter(final_resp['num_presented_peptides'], final_resp['p_response'], alpha=0.4, s=10, color='steelblue')
axes[1,2].set_xlabel('Num Presented Peptides')
axes[1,2].set_ylabel('p_response')
axes[1,2].set_title(f'F) Response vs Peptides (r={corr:.3f})')

plt.suptitle('Personalized Neoantigen Vaccine Optimization Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig8_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8_dashboard.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nKey Results:")
print(f"  Mean per-cell response probability: {response_stats['mean_p_response']:.4f}")
print(f"  Coverage at threshold=0.9: {coverage_summary['coverage_at_thresh_0.9']:.4f}")
print(f"  Mean IoU across replicates: {mean_iou:.4f}")
print(f"  Selected vaccine elements: {sorted(selected_muts)}")
print(f"  Runtime at 10000 cells: {runtime[runtime['PopulationSize']==10000]['RunTime'].mean():.2f}s")

print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"Figures saved to: {IMG_DIR}/")
print("Done!")
