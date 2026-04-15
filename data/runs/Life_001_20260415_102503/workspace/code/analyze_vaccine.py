#!/usr/bin/env python3
"""
Comprehensive analysis of personalized neoantigen vaccine optimization data.
Generates all figures and saves intermediate results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────
print("Loading data...")

cell_pop = pd.read_csv(os.path.join(DATA_DIR, 'cell-populations.csv'))
final_resp = pd.read_csv(os.path.join(DATA_DIR, 'final-response-likelihoods.csv'))
runtime_data = pd.read_csv(os.path.join(DATA_DIR, 'optimization_runtime_data.csv'))
selected_elements = pd.read_csv(os.path.join(DATA_DIR, 'selected-vaccine-elements.budget-10.minsum.adaptive.csv'))
sim_resp = pd.read_csv(os.path.join(DATA_DIR, 'sim-specific-response-likelihoods.csv'))
vaccine_comp = pd.read_csv(os.path.join(DATA_DIR, 'vaccine.budget-10.minsum.adaptive.csv'))

# Load all replicate score files
score_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('vaccine-elements.scores.')])
score_dfs = []
for sf in score_files:
    df = pd.read_csv(os.path.join(DATA_DIR, sf))
    rep_id = int(sf.split('rep-')[1].split('.')[0])
    df['replicate'] = rep_id
    score_dfs.append(df)
    print(f"  Loaded {sf}: {len(df)} rows")

scores_all = pd.concat(score_dfs, ignore_index=True)

print(f"\nData summary:")
print(f"  cell-populations: {cell_pop.shape}")
print(f"  final-response-likelihoods: {final_resp.shape}")
print(f"  optimization-runtime: {runtime_data.shape}")
print(f"  selected-elements: {selected_elements.shape}")
print(f"  sim-response-likelihoods: {sim_resp.shape}")
print(f"  vaccine-composition: {vaccine_comp.shape}")
print(f"  scores (all reps): {scores_all.shape}")

# ── 2. Basic Statistics ───────────────────────────────────────────────
print("\n=== Basic Statistics ===")

# Unique mutations / peptides
unique_mutations = cell_pop['mutation'].unique()
print(f"Unique mutations in cell populations: {len(unique_mutations)}")
print(f"Mutations: {sorted(unique_mutations)}")

# Vaccine composition
print(f"\nVaccine composition (budget-10 MinSum):")
print(vaccine_comp.to_string(index=False))

# Response probability distribution
resp_mean = final_resp['p_response'].mean()
resp_std = final_resp['p_response'].std()
resp_median = final_resp['p_response'].median()
resp_min = final_resp['p_response'].min()
resp_max = final_resp['p_response'].max()
print(f"\nResponse probability stats:")
print(f"  Mean: {resp_mean:.4f} ± {resp_std:.4f}")
print(f"  Median: {resp_median:.4f}")
print(f"  Range: [{resp_min:.4f}, {resp_max:.4f}]")

# Cells with high response probability (>0.9)
high_resp = (final_resp['p_response'] > 0.9).sum()
print(f"  Cells with p_response > 0.9: {high_resp}/{len(final_resp)} ({100*high_resp/len(final_resp):.1f}%)")

# Per-repetition vaccine selection stability
print(f"\nVaccine selection across repetitions:")
for rep in sorted(selected_elements['repetition'].unique()):
    rep_elements = selected_elements[selected_elements['repetition'] == rep]['peptide'].tolist()
    print(f"  Rep {rep}: {sorted(rep_elements)}")

# ── 3. Save Intermediate Results ──────────────────────────────────────
print("\nSaving intermediate results...")

# Response statistics
resp_stats = {
    'mean': float(resp_mean),
    'std': float(resp_std),
    'median': float(resp_median),
    'min': float(resp_min),
    'max': float(resp_max),
    'n_cells': len(final_resp),
    'high_response_ratio': float(high_resp / len(final_resp)),
    'vaccine_elements': vaccine_comp['peptide'].tolist(),
    'budget': 10,
    'objective': 'MinSum'
}
with open(os.path.join(OUTPUTS_DIR, 'response_statistics.json'), 'w') as f:
    json.dump(resp_stats, f, indent=2)

# Vaccine composition
vaccine_comp.to_csv(os.path.join(OUTPUTS_DIR, 'vaccine_composition.csv'), index=False)

# Selected elements per repetition
sel_summary = selected_elements.groupby(['peptide', 'repetition']).agg(
    weight=('weight', 'first'),
    run_time=('run_time', 'first')
).reset_index()
sel_summary.to_csv(os.path.join(OUTPUTS_DIR, 'selected_elements_summary.csv'), index=False)

# Aggregate scores across replicates
agg_scores = scores_all.groupby(['cell_id', 'vaccine_element']).agg(
    p_response_mean=('p_response', 'mean'),
    p_response_std=('p_response', 'std'),
    p_response_median=('p_response', 'median'),
    n_reps=('replicate', 'count')
).reset_index()
agg_scores.to_csv(os.path.join(OUTPUTS_DIR, 'aggregated_cell_element_scores.csv'), index=False)

# Per-cell coverage (union of responding elements)
# A cell "responds" to an element if p_response > threshold
threshold = 0.5
cell_coverage = scores_all[scores_all['p_response'] > threshold].groupby(
    ['cell_id', 'replicate']
).agg(n_responding=('vaccine_element', 'nunique')).reset_index()
cell_coverage.to_csv(os.path.join(OUTPUTS_DIR, 'cell_coverage.csv'), index=False)

print("Intermediate results saved.")

# ── 4. Generate Figures ───────────────────────────────────────────────
print("\nGenerating figures...")

plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})

# ── Figure 1: Response Probability Distribution ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram
ax = axes[0]
ax.hist(final_resp['p_response'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(resp_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {resp_mean:.3f}')
ax.axvline(resp_median, color='orange', linestyle='--', linewidth=2, label=f'Median = {resp_median:.3f}')
ax.set_xlabel('Per-Cell Response Probability', fontsize=12)
ax.set_ylabel('Cell Count', fontsize=12)
ax.set_title('Distribution of Immune Response Probabilities\n(MinSum Budget-10 Adaptive)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ECDF
ax = axes[1]
sorted_p = np.sort(final_resp['p_response'])
ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
ax.plot(sorted_p, ecdf, color='steelblue', linewidth=2.5)
ax.axhline(0.9, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(np.percentile(final_resp['p_response'], 90), color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Response Probability', fontsize=12)
ax.set_ylabel('Cumulative Fraction', fontsize=12)
ax.set_title('Empirical CDF of Response Probabilities', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.7, 1.0)

# Box plot by number of presented peptides
ax = axes[2]
final_resp['peptide_bin'] = pd.cut(final_resp['num_presented_peptides'], 
                                     bins=[0, 10, 15, 20, 30], 
                                     labels=['≤10', '11-15', '16-20', '>20'])
bp = final_resp.boxplot(column='p_response', by='peptide_bin', ax=ax, patch_artist=True,
                         return_type='dict')
for patch in bp['p_response']['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
ax.set_xlabel('Number of Presented Peptides', fontsize=12)
ax.set_ylabel('Response Probability', fontsize=12)
ax.set_title('Response Probability vs Peptide Count', fontsize=13, fontweight='bold')
plt.suptitle('', fontsize=0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure1_response_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure1_response_distribution.png")

# ── Figure 2: Cell × Mutation Response Heatmap ────────────────────────
# Use first replicate for clarity
rep0 = scores_all[scores_all['replicate'] == 0].copy()
pivot = rep0.pivot_table(index='cell_id', columns='vaccine_element', values='p_response')
# Sort cells by mean response
pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(pivot.values.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_yticks(range(len(pivot.columns)))
ax.set_yticklabels(pivot.columns, fontsize=10)
ax.set_xlabel('Cell ID', fontsize=12)
ax.set_ylabel('Vaccine Element (Mutation)', fontsize=12)
ax.set_title('Cell × Vaccine Element Response Probability Matrix\n(Replicate 0, 100 Cells)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Response Probability', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure2_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure2_heatmap.png")

# ── Figure 3: Per-Mutation Response Across Replicates ─────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})

# Violin plot of per-mutation response probabilities
ax = axes[0]
sns.violinplot(data=scores_all, x='vaccine_element', y='p_response', 
               inner='quartile', palette='Blues', ax=ax, cut=0)
ax.set_xlabel('Vaccine Element (Mutation)', fontsize=12)
ax.set_ylabel('Response Probability', fontsize=12)
ax.set_title('Per-Mutation Response Probability Distribution Across 10 Replicates', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)

# Mean response with error bars
ax = axes[1]
mut_stats = scores_all.groupby('vaccine_element')['p_response'].agg(['mean', 'std', 'median']).reset_index()
mut_stats = mut_stats.sort_values('mean', ascending=True)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(mut_stats)))
bars = ax.barh(mut_stats['vaccine_element'], mut_stats['mean'], 
               xerr=mut_stats['std'], capsize=4, color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('Mean Response Probability ± SD', fontsize=12)
ax.set_ylabel('Vaccine Element (Mutation)', fontsize=12)
ax.set_title('Mean Per-Mutation Response Probability (±1 SD) Across Replicates', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure3_per_mutation_response.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure3_per_mutation_response.png")

# ── Figure 4: Coverage Analysis ───────────────────────────────────────
# Compute cumulative coverage as we add vaccine elements
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# For each replicate, compute what fraction of cells respond to at least one element
coverage_by_rep = []
for rep in range(10):
    rep_data = scores_all[scores_all['replicate'] == rep]
    total_cells = rep_data['cell_id'].nunique()
    
    # Sort elements by mean response
    elem_means = rep_data.groupby('vaccine_element')['p_response'].mean().sort_values(ascending=False)
    
    cumulative_coverage = []
    covered_cells = set()
    for elem in elem_means.index:
        responding = set(rep_data[rep_data['vaccine_element'] == elem][
            rep_data[rep_data['vaccine_element'] == elem]['p_response'] > 0.5
        ]['cell_id'])
        covered_cells |= responding
        cumulative_coverage.append(len(covered_cells) / total_cells)
    
    coverage_by_rep.append(cumulative_coverage)

coverage_arr = np.array(coverage_by_rep)
n_elements = coverage_arr.shape[1]

ax = axes[0]
x = np.arange(1, n_elements + 1)
mean_cov = coverage_arr.mean(axis=0)
std_cov = coverage_arr.std(axis=0)
ax.plot(x, mean_cov, 'o-', color='steelblue', linewidth=2.5, markersize=6, label='Mean Coverage')
ax.fill_between(x, mean_cov - std_cov, mean_cov + std_cov, alpha=0.2, color='steelblue')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Vaccine Elements Added', fontsize=12)
ax.set_ylabel('Fraction of Cells Covered', fontsize=12)
ax.set_title('Cumulative Tumor Cell Coverage vs Vaccine Budget\n(MinSum Greedy Selection, 10 Replicates)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(x)
ax.set_xlim(0.5, n_elements + 0.5)

# IoU between optimal sets across replicates
ax = axes[1]
# Compute pairwise IoU of selected elements
ious = []
for i in range(10):
    for j in range(i+1, 10):
        set_i = set(selected_elements[selected_elements['repetition'] == i]['peptide'])
        set_j = set(selected_elements[selected_elements['repetition'] == j]['peptide'])
        iou = len(set_i & set_j) / len(set_i | set_j)
        ious.append(iou)

ax.hist(ious, bins=10, color='darkgreen', edgecolor='white', alpha=0.8)
ax.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean IoU = {np.mean(ious):.3f}')
ax.set_xlabel('Jaccard Index (IoU) Between Replicate Pairs', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Vaccine Composition Stability Across Replicates\nPairwise Intersection-over-Union', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure4_coverage_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure4_coverage_analysis.png")

# ── Figure 5: Vaccine Element Importance / Recall ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Selection frequency across repetitions
ax = axes[0]
sel_counts = selected_elements.groupby('peptide').size().reset_index(name='count')
sel_counts = sel_counts.sort_values('count', ascending=True)
colors_sel = plt.cm.Greens(np.linspace(0.3, 0.9, len(sel_counts)))
ax.barh(sel_counts['peptide'], sel_counts['count'], color=colors_sel, edgecolor='white', height=0.6)
ax.set_xlabel('Selection Frequency (out of 10 repetitions)', fontsize=12)
ax.set_ylabel('Vaccine Element (Mutation)', fontsize=12)
ax.set_title('Vaccine Element Selection Stability\nAcross 10 Optimization Runs', fontsize=13, fontweight='bold')
ax.set_xlim(0, 11)
ax.grid(True, alpha=0.3, axis='x')

# Weight distribution
ax = axes[1]
weights = selected_elements.groupby('peptide')['weight'].mean().sort_values(ascending=True)
colors_wt = plt.cm.Purples(np.linspace(0.3, 0.9, len(weights)))
ax.barh(weights.index, weights.values, color=colors_wt, edgecolor='white', height=0.6)
ax.set_xlabel('Average Weight', fontsize=12)
ax.set_ylabel('Vaccine Element (Mutation)', fontsize=12)
ax.set_title('Average Vaccine Element Weights\n(MinSum Objective)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure5_vaccine_importance.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure5_vaccine_importance.png")

# ── Figure 6: Runtime Scaling ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-log plot
ax = axes[0]
sample_ids = runtime_data['SampleID'].unique()
colors_rt = sns.color_palette('husl', len(sample_ids))
for i, sid in enumerate(sorted(sample_ids)):
    sub = runtime_data[runtime_data['SampleID'] == sid].sort_values('PopulationSize')
    ax.loglog(sub['PopulationSize'], sub['RunTime'], 'o-', 
              color=colors_rt[i], linewidth=2, markersize=8, label=f'Patient {sid}')

ax.set_xlabel('Cell Population Size', fontsize=12)
ax.set_ylabel('Optimization Runtime (seconds)', fontsize=12)
ax.set_title('Optimization Runtime vs Population Size\n(Log-Log Scale)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# Power law fit
ax = axes[1]
all_sizes = runtime_data['PopulationSize'].values
all_times = runtime_data['RunTime'].values
# Fit log(time) = a * log(size) + b
log_sizes = np.log(all_sizes)
log_times = np.log(all_times)
coeffs = np.polyfit(log_sizes, log_times, 1)
a, b = coeffs
fit_line = np.exp(b) * all_sizes ** a

sample_codes = pd.Categorical(runtime_data['SampleID']).codes
scatter = ax.scatter(all_sizes, all_times, c=sample_codes,
                     cmap='viridis', s=80, edgecolors='black', linewidth=0.5, zorder=5)
ax.plot(np.unique(all_sizes), np.exp(b) * np.unique(all_sizes) ** a, 
        'r--', linewidth=2.5, label=f'Power Law: t ∝ N^{a:.2f}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cell Population Size', fontsize=12)
ax.set_ylabel('Optimization Runtime (seconds)', fontsize=12)
ax.set_title('Scaling Behavior with Power-Law Fit\nAll Patients Combined', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure6_runtime_scaling.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure6_runtime_scaling.png")

# ── Figure 7: Subgroup Analysis by Mutation Type ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Response probability by mutation in cell populations
ax = axes[0]
mut_cell_counts = cell_pop.groupby('mutation')['cell_ids'].nunique().sort_values(ascending=True)
colors_mc = plt.cm.RdYlBu(np.linspace(0.2, 0.9, len(mut_cell_counts)))
ax.barh(mut_cell_counts.index, mut_cell_counts.values, color=colors_mc, edgecolor='white', height=0.6)
ax.set_xlabel('Number of Cells Presenting Mutation', fontsize=12)
ax.set_ylabel('Mutation', fontsize=12)
ax.set_title('Mutation Prevalence Across Cell Population\n(Number of Cells per Mutation)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# HLA allele distribution
ax = axes[1]
hla_counts = cell_pop['presented_hlas'].value_counts().sort_values(ascending=True)
colors_hla = plt.cm.Set2(np.linspace(0.2, 0.9, len(hla_counts)))
ax.barh(hla_counts.index, hla_counts.values, color=colors_hla, edgecolor='white', height=0.6)
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('HLA Allele', fontsize=12)
ax.set_title('HLA Allele Distribution in Cell Population', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'figure7_subgroup_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figure7_subgroup_analysis.png")

# ── 5. Save Key Metrics ───────────────────────────────────────────────
print("\nSaving key metrics...")

# Coverage metrics
coverage_metrics = {
    'mean_final_coverage': float(mean_cov[-1]),
    'std_final_coverage': float(std_cov[-1]),
    'coverage_at_budget_5': float(coverage_arr[:, 4].mean()),
    'coverage_at_budget_10': float(coverage_arr[:, 9].mean()),
    'mean_iou': float(np.mean(ious)),
    'std_iou': float(np.std(ious)),
    'min_iou': float(np.min(ious)),
    'max_iou': float(np.max(ious)),
    'n_replicate_pairs': len(ious)
}

# Runtime metrics
runtime_metrics = {}
for sid in sorted(runtime_data['SampleID'].unique()):
    sub = runtime_data[runtime_data['SampleID'] == sid]
    runtime_metrics[str(sid)] = {
        'runtime_100': float(sub[sub['PopulationSize'] == 100]['RunTime'].values[0]),
        'runtime_10000': float(sub[sub['PopulationSize'] == 10000]['RunTime'].values[0]),
        'speedup_factor': float(sub[sub['PopulationSize'] == 10000]['RunTime'].values[0] / 
                                sub[sub['PopulationSize'] == 100]['RunTime'].values[0])
    }

# Power law fit
power_law_fit = {
    'exponent': float(a),
    'intercept_log': float(b),
    'interpretation': f'Runtime scales as O(N^{a:.2f})'
}

key_metrics = {
    'response_statistics': resp_stats,
    'coverage_metrics': coverage_metrics,
    'runtime_metrics': runtime_metrics,
    'power_law_fit': power_law_fit,
    'vaccine_composition': vaccine_comp.to_dict(orient='records'),
    'selection_stability': {
        'fully_stable_elements': int((sel_counts['count'] == 10).sum()),
        'partially_stable_elements': int(((sel_counts['count'] > 0) & (sel_counts['count'] < 10)).sum()),
        'total_unique_elements': len(sel_counts)
    }
}

with open(os.path.join(OUTPUTS_DIR, 'key_metrics.json'), 'w') as f:
    json.dump(key_metrics, f, indent=2)

print("Key metrics saved to outputs/key_metrics.json")

print("\n=== Analysis Complete ===")
print(f"Figures saved to: {IMAGES_DIR}/")
print(f"Outputs saved to: {OUTPUTS_DIR}/")
