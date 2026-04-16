#!/usr/bin/env python3
"""
Main analysis script for personalized neoantigen vaccine optimization.
Generates all figures and intermediate results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = Path("report/images")
CODE_DIR = Path("code")

for d in [OUTPUT_DIR, IMAGE_DIR, CODE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")

cell_pop = pd.read_csv(DATA_DIR / "cell-populations.csv")
final_resp = pd.read_csv(DATA_DIR / "final-response-likelihoods.csv")
sim_resp = pd.read_csv(DATA_DIR / "sim-specific-response-likelihoods.csv")
opt_runtime = pd.read_csv(DATA_DIR / "optimization_runtime_data.csv")
selected_elements = pd.read_csv(DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
vaccine_comp = pd.read_csv(DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv")

# Load all replicate score files
score_dfs = []
for i in range(10):
    df = pd.read_csv(DATA_DIR / f"vaccine-elements.scores.100-cells.10x.rep-{i}.csv")
    df['replication'] = i
    score_dfs.append(df)
all_scores = pd.concat(score_dfs, ignore_index=True)

print(f"Cell population: {len(cell_pop)} rows")
print(f"Final response: {len(final_resp)} rows")
print(f"All scores: {len(all_scores)} rows")

# ============================================================
# 2. Data Overview
# ============================================================
print("\n=== Data Overview ===")

# Cell population summary
n_cells = cell_pop['cell_ids'].nunique()
n_mutations = cell_pop['mutation'].nunique()
n_peptides = cell_pop['presented_peptides'].nunique()
n_repetitions = cell_pop['repetition'].nunique()

overview = {
    'n_cells': n_cells,
    'n_mutations': n_mutations,
    'n_peptides': n_peptides,
    'n_repetitions': n_repetitions,
    'hla_alleles': cell_pop['presented_hlas'].unique().tolist(),
    'selected_vaccine_elements': vaccine_comp['peptide'].tolist(),
    'budget': len(vaccine_comp),
}
print(f"Cells: {n_cells}, Mutations: {n_mutations}, Peptides: {n_peptides}, Repetitions: {n_repetitions}")
print(f"Selected vaccine elements: {overview['selected_vaccine_elements']}")

# Save overview
import json
with open(OUTPUT_DIR / "data_overview.json", 'w') as f:
    json.dump(overview, f, indent=2, default=str)

# ============================================================
# 3. Vaccine Composition Consistency (IoU across repetitions)
# ============================================================
print("\n=== Vaccine Composition IoU ===")

# Get selected elements per repetition
selected_per_rep = {}
for rep in range(10):
    rep_elements = set(selected_elements[selected_elements['repetition'] == rep]['peptide'].tolist())
    selected_per_rep[rep] = rep_elements

# Compute pairwise IoU
n_reps = 10
iou_matrix = np.zeros((n_reps, n_reps))
for i in range(n_reps):
    for j in range(n_reps):
        intersection = len(selected_per_rep[i] & selected_per_rep[j])
        union = len(selected_per_rep[i] | selected_per_rep[j])
        iou_matrix[i, j] = intersection / union if union > 0 else 0

# Save IoU matrix
iou_df = pd.DataFrame(iou_matrix, index=[f"rep-{i}" for i in range(n_reps)], 
                       columns=[f"rep-{i}" for i in range(n_reps)])
iou_df.to_csv(OUTPUT_DIR / "iou_matrix.csv")

# Compute mean IoU (excluding diagonal)
mean_iou = iou_matrix[np.triu_indices(n_reps, k=1)].mean()
print(f"Mean pairwise IoU: {mean_iou:.4f}")

# Also compute IoU against the consensus vaccine
consensus_elements = set(vaccine_comp['peptide'].tolist())
ious_vs_consensus = []
for rep in range(n_reps):
    intersection = len(selected_per_rep[rep] & consensus_elements)
    union = len(selected_per_rep[rep] | consensus_elements)
    iou = intersection / union if union > 0 else 0
    ious_vs_consensus.append(iou)
    
mean_iou_consensus = np.mean(ious_vs_consensus)
print(f"Mean IoU vs consensus: {mean_iou_consensus:.4f}")

# ============================================================
# Figure 1: IoU Heatmap across repetitions
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.zeros_like(iou_matrix, dtype=bool)
# No mask - show full matrix
sns.heatmap(iou_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
            ax=ax, linewidths=0.5, square=True,
            cbar_kws={'label': 'IoU'})
ax.set_title(f"Vaccine Composition IoU Across Repetitions\n(Mean pairwise IoU = {mean_iou:.3f})")
ax.set_xlabel("Repetition")
ax.set_ylabel("Repetition")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure1_iou_heatmap.png")
plt.close()
print("Saved figure1_iou_heatmap.png")

# ============================================================
# 4. Per-Cell Immune Response Probability Distribution
# ============================================================
print("\n=== Response Probability Distribution ===")

# Aggregate across all repetitions
p_response_all = final_resp['p_response'].values
mean_p_response = p_response_all.mean()
median_p_response = np.median(p_response_all)
std_p_response = p_response_all.std()

print(f"Mean p_response: {mean_p_response:.4f}")
print(f"Median p_response: {median_p_response:.4f}")
print(f"Std p_response: {std_p_response:.4f}")

# Per-repetition statistics
rep_stats = final_resp.copy()
rep_stats['repetition'] = rep_stats['population'].str.extract(r'(\d+)$').astype(int)
rep_summary = rep_stats.groupby('repetition')['p_response'].agg(['mean', 'std', 'median']).reset_index()
rep_summary.to_csv(OUTPUT_DIR / "per_repetition_response_stats.csv", index=False)

# ============================================================
# Figure 2: Response Probability Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: Histogram
axes[0].hist(p_response_all, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(mean_p_response, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_p_response:.3f}')
axes[0].axvline(median_p_response, color='orange', linestyle=':', linewidth=2, label=f'Median = {median_p_response:.3f}')
axes[0].set_xlabel("Per-Cell Immune Response Probability (p_response)")
axes[0].set_ylabel("Count")
axes[0].set_title("A) Distribution of Per-Cell Response Probabilities")
axes[0].legend(fontsize=10)

# Panel B: Box plot by repetition
rep_data = []
for rep in range(10):
    rep_mask = rep_stats['repetition'] == rep
    rep_data.append(rep_stats.loc[rep_mask, 'p_response'].values)

bp = axes[1].boxplot(rep_data, patch_artist=True, labels=[f"Rep {i}" for i in range(10)])
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1].set_xlabel("Repetition")
axes[1].set_ylabel("Per-Cell Response Probability")
axes[1].set_title("B) Response Probability by Repetition")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure2_response_distribution.png")
plt.close()
print("Saved figure2_response_distribution.png")

# ============================================================
# 5. Coverage Ratio Analysis
# ============================================================
print("\n=== Coverage Ratio ===")

thresholds = np.arange(0, 1.01, 0.05)
coverage_per_rep = {}

for rep in range(10):
    rep_mask = rep_stats['repetition'] == rep
    rep_p = rep_stats.loc[rep_mask, 'p_response'].values
    coverages = []
    for t in thresholds:
        coverage = np.mean(rep_p >= t)
        coverages.append(coverage)
    coverage_per_rep[rep] = coverages

# Mean coverage across repetitions
mean_coverage = np.mean([coverage_per_rep[r] for r in range(10)], axis=0)
std_coverage = np.std([coverage_per_rep[r] for r in range(10)], axis=0)

# Save coverage data
coverage_df = pd.DataFrame({
    'threshold': thresholds,
    'mean_coverage': mean_coverage,
    'std_coverage': std_coverage
})
coverage_df.to_csv(OUTPUT_DIR / "coverage_curve.csv", index=False)

# Key coverage metrics
coverage_50 = np.mean([np.mean(rep_stats.loc[rep_stats['repetition']==r, 'p_response'].values >= 0.5) for r in range(10)])
coverage_90 = np.mean([np.mean(rep_stats.loc[rep_stats['repetition']==r, 'p_response'].values >= 0.9) for r in range(10)])
coverage_95 = np.mean([np.mean(rep_stats.loc[rep_stats['repetition']==r, 'p_response'].values >= 0.95) for r in range(10)])
print(f"Coverage at p>=0.5: {coverage_50:.4f}")
print(f"Coverage at p>=0.9: {coverage_90:.4f}")
print(f"Coverage at p>=0.95: {coverage_95:.4f}")

# ============================================================
# Figure 3: Coverage Curve
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(thresholds, mean_coverage, 'b-', linewidth=2.5, label='Mean Coverage')
ax.fill_between(thresholds, mean_coverage - std_coverage, mean_coverage + std_coverage, 
                alpha=0.2, color='blue', label='±1 SD')
ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label=f'p≥0.5: {coverage_50:.1%}')
ax.axvline(0.9, color='orange', linestyle='--', alpha=0.7, label=f'p≥0.9: {coverage_90:.1%}')
ax.axvline(0.95, color='red', linestyle='--', alpha=0.7, label=f'p≥0.95: {coverage_95:.1%}')
ax.set_xlabel("Response Probability Threshold")
ax.set_ylabel("Fraction of Cells Covered")
ax.set_title("Tumor Cell Coverage vs. Response Probability Threshold")
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure3_coverage_curve.png")
plt.close()
print("Saved figure3_coverage_curve.png")

# ============================================================
# 6. Vaccine Element Contribution Analysis
# ============================================================
print("\n=== Vaccine Element Contribution ===")

# For each vaccine element, compute mean p_response across cells and repetitions
# Only for the selected elements
selected_muts = vaccine_comp['peptide'].tolist()

element_stats = all_scores[all_scores['vaccine_element'].isin(selected_muts)].groupby('vaccine_element').agg(
    mean_p_response=('p_response', 'mean'),
    std_p_response=('p_response', 'std'),
    median_p_response=('p_response', 'median'),
    max_p_response=('p_response', 'max'),
    min_p_response=('p_response', 'min'),
    n_cells_responded=('p_response', lambda x: (x > 0.01).sum()),
).reset_index()

element_stats['fraction_cells_responded'] = element_stats['n_cells_responded'] / (100 * 10)  # 100 cells × 10 reps
element_stats = element_stats.sort_values('mean_p_response', ascending=False)
element_stats.to_csv(OUTPUT_DIR / "vaccine_element_contribution.csv", index=False)
print(element_stats.to_string())

# ============================================================
# Figure 4: Vaccine Element Contribution Heatmap
# ============================================================
# Create cell × element heatmap for one representative repetition (rep-0)
rep0_scores = pd.read_csv(DATA_DIR / "vaccine-elements.scores.100-cells.10x.rep-0.csv")
selected_scores = rep0_scores[rep0_scores['vaccine_element'].isin(selected_muts)]

# Pivot to cell × element matrix
heatmap_data = selected_scores.pivot_table(index='cell_id', columns='vaccine_element', values='p_response')
heatmap_data = heatmap_data[sorted(heatmap_data.columns, key=lambda x: int(x.replace('mut', '')))]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap="YlOrRd", vmin=0, vmax=1,
            ax=ax, linewidths=0, 
            cbar_kws={'label': 'Response Probability'})
ax.set_title("Per-Cell Response Probability by Vaccine Element (Rep 0)")
ax.set_xlabel("Vaccine Element")
ax.set_ylabel("Cell ID")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure4_element_heatmap.png")
plt.close()
print("Saved figure4_element_heatmap.png")

# ============================================================
# Figure 5: Vaccine Element Mean Response Bar Plot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
element_stats_sorted = element_stats.sort_values('mean_p_response', ascending=True)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(element_stats_sorted)))
bars = ax.barh(element_stats_sorted['vaccine_element'], element_stats_sorted['mean_p_response'], 
               xerr=element_stats_sorted['std_p_response'], color=colors, edgecolor='gray', linewidth=0.5)
ax.set_xlabel("Mean Per-Cell Response Probability")
ax.set_ylabel("Vaccine Element")
ax.set_title("Contribution of Each Vaccine Element to Immune Response")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure5_element_contribution.png")
plt.close()
print("Saved figure5_element_contribution.png")

# ============================================================
# 7. Runtime Scaling Analysis
# ============================================================
print("\n=== Runtime Scaling ===")

# Fit power law: runtime = a * N^b
from scipy.optimize import curve_fit

def power_law(N, a, b):
    return a * N**b

samples = opt_runtime['SampleID'].unique()
runtime_params = {}

for sample in samples:
    sample_data = opt_runtime[opt_runtime['SampleID'] == sample]
    pops = sample_data['PopulationSize'].values
    times = sample_data['RunTime'].values
    
    # Fit on positive times
    mask = times > 0
    if mask.sum() >= 2:
        try:
            popt, _ = curve_fit(power_law, pops[mask], times[mask], p0=[1e-5, 1.5], maxfev=10000)
            runtime_params[sample] = {'a': popt[0], 'b': popt[1]}
        except:
            runtime_params[sample] = {'a': np.nan, 'b': np.nan}

runtime_params_df = pd.DataFrame(runtime_params).T
runtime_params_df.index.name = 'SampleID'
runtime_params_df.to_csv(OUTPUT_DIR / "runtime_scaling_params.csv")

mean_exponent = runtime_params_df['b'].mean()
print(f"Mean scaling exponent: {mean_exponent:.3f}")
print(runtime_params_df)

# ============================================================
# Figure 6: Runtime vs Population Size
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

colors_samples = plt.cm.tab10(np.linspace(0, 1, len(samples)))
for idx, sample in enumerate(samples):
    sample_data = opt_runtime[opt_runtime['SampleID'] == sample]
    ax.plot(sample_data['PopulationSize'], sample_data['RunTime'], 'o-', 
            color=colors_samples[idx], label=f'Sample {sample}', linewidth=2, markersize=6)
    
    # Plot fitted curve
    if sample in runtime_params and not np.isnan(runtime_params[sample]['b']):
        a, b = runtime_params[sample]['a'], runtime_params[sample]['b']
        N_fit = np.linspace(100, 10000, 100)
        t_fit = power_law(N_fit, a, b)
        ax.plot(N_fit, t_fit, '--', color=colors_samples[idx], alpha=0.5)

ax.set_xlabel("Cell Population Size")
ax.set_ylabel("Optimization Runtime (seconds)")
ax.set_title(f"Optimization Runtime vs. Population Size\n(Mean scaling exponent ≈ N^{mean_exponent:.2f})")
ax.legend(fontsize=9, loc='upper left')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure6_runtime_scaling.png")
plt.close()
print("Saved figure6_runtime_scaling.png")

# ============================================================
# 8. Additional Analysis: Mutation Presentation in Cell Population
# ============================================================
print("\n=== Mutation Presentation Analysis ===")

# For each cell, which mutations are presented?
cell_mutation = cell_pop.groupby(['repetition', 'cell_ids'])['mutation'].apply(set).reset_index()
cell_mutation.columns = ['repetition', 'cell_ids', 'mutations_presented']

# For each mutation, how many cells present it?
mut_cell_count = cell_pop.groupby(['repetition', 'mutation'])['cell_ids'].nunique().reset_index()
mut_cell_count.columns = ['repetition', 'mutation', 'n_cells_presenting']

# Average across repetitions
mut_avg = mut_cell_count.groupby('mutation')['n_cells_presenting'].agg(['mean', 'std']).reset_index()
mut_avg = mut_avg.sort_values('mean', ascending=False)
mut_avg.to_csv(OUTPUT_DIR / "mutation_cell_presentation.csv", index=False)
print(mut_avg.to_string())

# ============================================================
# Figure 7: Mutation Presentation Across Cells
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
mut_avg_sorted = mut_avg.sort_values('mean', ascending=True)
colors_mut = ['steelblue' if m in selected_muts else 'lightgray' for m in mut_avg_sorted['mutation']]
ax.barh(mut_avg_sorted['mutation'], mut_avg_sorted['mean'], 
        xerr=mut_avg_sorted['std'], color=colors_mut, edgecolor='gray', linewidth=0.5)
ax.set_xlabel("Mean Number of Cells Presenting Mutation")
ax.set_ylabel("Mutation")
ax.set_title("Mutation Presentation Across Cell Population\n(Blue = Selected for Vaccine)")
ax.legend(['Not selected', 'Selected'], 
          handles=[plt.Rectangle((0,0),1,1,facecolor='lightgray',edgecolor='gray'),
                   plt.Rectangle((0,0),1,1,facecolor='steelblue',edgecolor='gray')])
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure7_mutation_presentation.png")
plt.close()
print("Saved figure7_mutation_presentation.png")

# ============================================================
# 9. Comprehensive Efficacy Summary
# ============================================================
print("\n=== Comprehensive Efficacy Summary ===")

efficacy_summary = {
    'vaccine_type': 'MinSum.budget-10.adaptive',
    'budget': 10,
    'n_cells': int(n_cells),
    'n_mutations_total': int(n_mutations),
    'n_vaccine_elements': len(selected_muts),
    'selected_elements': selected_muts,
    'mean_p_response': float(mean_p_response),
    'median_p_response': float(median_p_response),
    'std_p_response': float(std_p_response),
    'coverage_p50': float(coverage_50),
    'coverage_p90': float(coverage_90),
    'coverage_p95': float(coverage_95),
    'mean_iou_pairwise': float(mean_iou),
    'mean_iou_consensus': float(mean_iou_consensus),
    'mean_runtime_exponent': float(mean_exponent),
    'runtime_100cells': float(opt_runtime[opt_runtime['PopulationSize']==100]['RunTime'].mean()),
    'runtime_10000cells': float(opt_runtime[opt_runtime['PopulationSize']==10000]['RunTime'].mean()),
}

with open(OUTPUT_DIR / "efficacy_summary.json", 'w') as f:
    json.dump(efficacy_summary, f, indent=2)

print(json.dumps(efficacy_summary, indent=2))
print("\n=== Analysis Complete ===")
