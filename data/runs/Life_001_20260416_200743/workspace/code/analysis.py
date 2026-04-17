#!/usr/bin/env python3
"""
Personalized Neoantigen Vaccine Optimization Analysis
=====================================================
This script performs comprehensive analysis of neoantigen vaccine optimization
using the MinSum objective with adaptive budget-constrained selection.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 7),
})
sns.set_style("whitegrid")

DATA_DIR = "data"
IMG_DIR = "report/images"
OUT_DIR = "outputs"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Load all data
# ============================================================
print("Loading data...")

cell_pop = pd.read_csv(f"{DATA_DIR}/cell-populations.csv")
final_resp = pd.read_csv(f"{DATA_DIR}/final-response-likelihoods.csv")
runtime_data = pd.read_csv(f"{DATA_DIR}/optimization_runtime_data.csv")
selected_elements = pd.read_csv(f"{DATA_DIR}/selected-vaccine-elements.budget-10.minsum.adaptive.csv")
sim_resp = pd.read_csv(f"{DATA_DIR}/sim-specific-response-likelihoods.csv")
vaccine_comp = pd.read_csv(f"{DATA_DIR}/vaccine.budget-10.minsum.adaptive.csv")

# Load all replicate score files
rep_scores = {}
for i in range(10):
    rep_scores[i] = pd.read_csv(f"{DATA_DIR}/vaccine-elements.scores.100-cells.10x.rep-{i}.csv")

print("Data loaded successfully.")

# ============================================================
# 2. Data Overview Analysis
# ============================================================
print("\n=== DATA OVERVIEW ===")

# Cell population statistics
n_reps = cell_pop['repetition'].nunique()
n_mutations = cell_pop['mutation'].nunique()
n_peptides = cell_pop['presented_peptides'].nunique()
n_hlas = cell_pop['presented_hlas'].nunique()
cells_per_rep = cell_pop.groupby('repetition')['cell_ids'].nunique()

print(f"Number of replicates: {n_reps}")
print(f"Number of unique mutations: {n_mutations}")
print(f"Number of unique peptides: {n_peptides}")
print(f"Number of unique HLAs: {n_hlas}")
print(f"Cells per replicate: {cells_per_rep.values}")

# Mutation presentation frequency
mut_freq = cell_pop.groupby(['repetition', 'mutation']).agg(
    n_cells=('cell_ids', 'nunique'),
    n_peptides=('presented_peptides', 'nunique')
).reset_index()

print(f"\nMutation presentation frequency (avg cells per mutation):")
avg_freq = mut_freq.groupby('mutation')['n_cells'].mean().sort_values(ascending=False)
for m, v in avg_freq.items():
    print(f"  {m}: {v:.1f} cells")

# ============================================================
# 3. Vaccine Composition Analysis
# ============================================================
print("\n=== VACCINE COMPOSITION ===")

print(f"Budget: 10 elements")
print(f"Objective: MinSum (adaptive)")
print(f"\nSelected vaccine elements (adaptive consensus):")
print(vaccine_comp.to_string(index=False))

# Per-replicate vaccine composition
rep_vaccines = {}
for rep in range(10):
    rep_elements = selected_elements[selected_elements['repetition'] == rep]['peptide'].tolist()
    rep_vaccines[rep] = set(rep_elements)
    
# Check if all replicates select the same elements
all_same = all(rep_vaccines[0] == rep_vaccines[i] for i in range(10))
print(f"\nAll replicates select identical elements: {all_same}")

# Element selection frequency
element_counts = selected_elements.groupby('peptide')['repetition'].count()
print(f"\nElement selection frequency across replicates:")
for m, c in element_counts.sort_values(ascending=False).items():
    print(f"  {m}: {c}/10 replicates")

# ============================================================
# 4. IoU Analysis of Vaccine Compositions
# ============================================================
print("\n=== IoU ANALYSIS ===")

iou_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        intersection = len(rep_vaccines[i] & rep_vaccines[j])
        union = len(rep_vaccines[i] | rep_vaccines[j])
        iou_matrix[i, j] = intersection / union if union > 0 else 0

print(f"IoU matrix (pairwise):")
print(f"  Mean IoU: {np.mean(iou_matrix[np.triu_indices(10, k=1)]):.4f}")
print(f"  Min IoU: {np.min(iou_matrix[np.triu_indices(10, k=1)]):.4f}")
print(f"  Max IoU: {np.max(iou_matrix[np.triu_indices(10, k=1)]):.4f}")

# Save IoU matrix
iou_df = pd.DataFrame(iou_matrix, 
                       index=[f"rep-{i}" for i in range(10)],
                       columns=[f"rep-{i}" for i in range(10)])
iou_df.to_csv(f"{OUT_DIR}/iou_matrix.csv")

# ============================================================
# 5. Per-Cell Response Probability Analysis
# ============================================================
print("\n=== PER-CELL RESPONSE PROBABILITY ===")

# From final-response-likelihoods (adaptive vaccine applied to each replicate population)
print(f"Overall response probability statistics (adaptive vaccine):")
print(f"  Mean: {final_resp['p_response'].mean():.4f}")
print(f"  Median: {final_resp['p_response'].median():.4f}")
print(f"  Std: {final_resp['p_response'].std():.4f}")
print(f"  Min: {final_resp['p_response'].min():.6f}")
print(f"  Max: {final_resp['p_response'].max():.6f}")

# Per-replicate statistics
final_resp['rep'] = final_resp['population'].apply(lambda x: int(x.split(', ')[1]))
per_rep_stats = final_resp.groupby('rep')['p_response'].agg(['mean', 'median', 'std', 'min', 'max'])
print(f"\nPer-replicate response probability:")
print(per_rep_stats.to_string())

# Sim-specific response likelihoods
sim_resp['rep'] = sim_resp['vaccine'].apply(lambda x: int(x.split('rep-')[1]))
sim_resp['pop_rep'] = sim_resp['population'].apply(lambda x: int(x.split(', ')[1]))

# ============================================================
# 6. Coverage Analysis
# ============================================================
print("\n=== COVERAGE ANALYSIS ===")

thresholds = np.arange(0, 1.01, 0.01)
coverage_by_threshold = []
for t in thresholds:
    covered = (final_resp['p_response'] >= t).mean()
    coverage_by_threshold.append({'threshold': t, 'coverage': covered})

coverage_df = pd.DataFrame(coverage_by_threshold)
coverage_df.to_csv(f"{OUT_DIR}/coverage_curve.csv", index=False)

# Key coverage metrics
for t in [0.5, 0.7, 0.8, 0.9, 0.95]:
    cov = (final_resp['p_response'] >= t).mean()
    print(f"  Coverage at p_response >= {t}: {cov:.4f} ({cov*100:.1f}%)")

# Per-replicate coverage
print(f"\nPer-replicate coverage at p_response >= 0.5:")
for rep in range(10):
    rep_data = final_resp[final_resp['rep'] == rep]
    cov = (rep_data['p_response'] >= 0.5).mean()
    print(f"  Rep-{rep}: {cov:.4f} ({cov*100:.1f}%)")

# ============================================================
# 7. Per-Element Contribution Analysis
# ============================================================
print("\n=== PER-ELEMENT CONTRIBUTION ===")

# Analyze how each vaccine element contributes to cell responses
element_contributions = {}
for rep in range(10):
    df = rep_scores[rep]
    for elem in df['vaccine_element'].unique():
        elem_data = df[df['vaccine_element'] == elem]
        if elem not in element_contributions:
            element_contributions[elem] = []
        element_contributions[elem].append({
            'rep': rep,
            'mean_p_response': elem_data['p_response'].mean(),
            'median_p_response': elem_data['p_response'].median(),
            'max_p_response': elem_data['p_response'].max(),
            'cells_with_response': (elem_data['p_response'] > 0.01).sum(),
            'total_cells': len(elem_data)
        })

# Aggregate
elem_summary = []
for elem, reps in element_contributions.items():
    reps_df = pd.DataFrame(reps)
    elem_summary.append({
        'element': elem,
        'mean_p_response': reps_df['mean_p_response'].mean(),
        'std_p_response': reps_df['mean_p_response'].std(),
        'mean_cells_with_response': reps_df['cells_with_response'].mean(),
        'in_vaccine': elem in vaccine_comp['peptide'].values
    })

elem_summary_df = pd.DataFrame(elem_summary).sort_values('mean_p_response', ascending=False)
elem_summary_df.to_csv(f"{OUT_DIR}/element_contributions.csv", index=False)
print(elem_summary_df.to_string(index=False))

# ============================================================
# 8. Compute cell-level aggregate response
# ============================================================
print("\n=== CELL-LEVEL AGGREGATE RESPONSE ===")

# For each replicate, compute the combined response probability for each cell
# P(response) = 1 - prod(P(no_response_j)) for all vaccine elements j
cell_aggregate = []
for rep in range(10):
    df = rep_scores[rep]
    # Only use vaccine elements (10 selected)
    vaccine_elements = vaccine_comp['peptide'].values
    df_vac = df[df['vaccine_element'].isin(vaccine_elements)]
    
    # Aggregate per cell: sum log_p_no_response, then convert
    cell_agg = df_vac.groupby('cell_id').agg(
        sum_log_p_no_response=('log_p_no_response', 'sum')
    ).reset_index()
    cell_agg['p_no_response'] = np.exp(cell_agg['sum_log_p_no_response'])
    cell_agg['p_response'] = 1 - cell_agg['p_no_response']
    cell_agg['rep'] = rep
    cell_aggregate.append(cell_agg)

cell_agg_df = pd.concat(cell_aggregate, ignore_index=True)
cell_agg_df.to_csv(f"{OUT_DIR}/cell_aggregate_response.csv", index=False)

print(f"Computed aggregate response for {len(cell_agg_df)} cell-replicate pairs")
print(f"Mean p_response: {cell_agg_df['p_response'].mean():.4f}")
print(f"Median p_response: {cell_agg_df['p_response'].median():.4f}")

# Verify consistency with final-response-likelihoods
print(f"\nVerification: final-response-likelihoods mean = {final_resp['p_response'].mean():.4f}")
print(f"Computed aggregate mean = {cell_agg_df['p_response'].mean():.4f}")

# ============================================================
# 9. Greedy vs Optimal comparison (simulate alternative selections)
# ============================================================
print("\n=== ALTERNATIVE VACCINE COMPARISON ===")

# Compare the adaptive MinSum vaccine with:
# 1. Top-N by frequency (most commonly presented mutations)
# 2. Top-N by mean response probability

# Top-N by presentation frequency
mut_presentation = cell_pop.groupby('mutation')['cell_ids'].nunique()
top_by_freq = mut_presentation.sort_values(ascending=False).head(10).index.tolist()

# Top-N by mean response probability
top_by_response = elem_summary_df.sort_values('mean_p_response', ascending=False).head(10)['element'].tolist()

# MinSum adaptive selection
minsum_selection = vaccine_comp['peptide'].tolist()

print(f"MinSum adaptive: {sorted(minsum_selection)}")
print(f"Top by frequency: {sorted(top_by_freq)}")
print(f"Top by response: {sorted(top_by_response)}")

# Compute coverage for each strategy
def compute_coverage_for_selection(selection, rep_scores_dict, n_reps=10):
    """Compute mean p_response across all cells and replicates for a given selection."""
    all_responses = []
    for rep in range(n_reps):
        df = rep_scores_dict[rep]
        df_sel = df[df['vaccine_element'].isin(selection)]
        cell_agg = df_sel.groupby('cell_id').agg(
            sum_log_p_no_response=('log_p_no_response', 'sum')
        ).reset_index()
        cell_agg['p_response'] = 1 - np.exp(cell_agg['sum_log_p_no_response'])
        all_responses.extend(cell_agg['p_response'].tolist())
    return np.array(all_responses)

minsum_responses = compute_coverage_for_selection(minsum_selection, rep_scores)
freq_responses = compute_coverage_for_selection(top_by_freq, rep_scores)
response_responses = compute_coverage_for_selection(top_by_response, rep_scores)

strategies = {
    'MinSum Adaptive': minsum_responses,
    'Top by Frequency': freq_responses,
    'Top by Response': response_responses
}

for name, resp in strategies.items():
    print(f"\n{name}:")
    print(f"  Mean p_response: {resp.mean():.4f}")
    print(f"  Coverage (>0.5): {(resp >= 0.5).mean():.4f}")
    print(f"  Coverage (>0.8): {(resp >= 0.8).mean():.4f}")
    print(f"  Coverage (>0.9): {(resp >= 0.9).mean():.4f}")

# ============================================================
# 10. Runtime Analysis
# ============================================================
print("\n=== RUNTIME ANALYSIS ===")
print(runtime_data.to_string(index=False))

# Summary statistics
for pop_size in sorted(runtime_data['PopulationSize'].unique()):
    subset = runtime_data[runtime_data['PopulationSize'] == pop_size]
    print(f"\nPopulation {pop_size}: mean={subset['RunTime'].mean():.3f}s, "
          f"std={subset['RunTime'].std():.3f}s, "
          f"min={subset['RunTime'].min():.3f}s, max={subset['RunTime'].max():.3f}s")

# ============================================================
# FIGURES
# ============================================================
print("\n=== GENERATING FIGURES ===")

# ----------------------------------------------------------
# Figure 1: Cell Population Overview
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1a: Mutations per cell across replicates
cells_muts = cell_pop.groupby(['repetition', 'cell_ids'])['mutation'].nunique().reset_index()
cells_muts.columns = ['repetition', 'cell_id', 'n_mutations']

sns.boxplot(data=cells_muts, x='repetition', y='n_mutations', ax=axes[0], 
            palette='viridis', fliersize=3)
axes[0].set_xlabel('Replicate')
axes[0].set_ylabel('Number of Mutations per Cell')
axes[0].set_title('(A) Mutation Load per Cell Across Replicates')

# 1b: Mutation presentation frequency heatmap
mut_cell_matrix = cell_pop.groupby(['repetition', 'mutation'])['cell_ids'].nunique().unstack(fill_value=0)
sns.heatmap(mut_cell_matrix.T, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            cbar_kws={'label': 'Number of Cells'})
axes[1].set_xlabel('Replicate')
axes[1].set_ylabel('Mutation')
axes[1].set_title('(B) Mutation Presentation Across Replicates')

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig1_cell_population_overview.png", bbox_inches='tight')
plt.close()
print("Saved fig1_cell_population_overview.png")

# ----------------------------------------------------------
# Figure 2: Response Probability Distributions
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2a: Overall distribution
axes[0].hist(final_resp['p_response'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(final_resp['p_response'].mean(), color='red', linestyle='--', 
                label=f'Mean = {final_resp["p_response"].mean():.3f}')
axes[0].axvline(final_resp['p_response'].median(), color='orange', linestyle='--',
                label=f'Median = {final_resp["p_response"].median():.3f}')
axes[0].set_xlabel('Per-Cell Response Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('(A) Distribution of Per-Cell Response Probabilities')
axes[0].legend()

# 2b: Per-replicate violin plot
final_resp_plot = final_resp.copy()
final_resp_plot['Replicate'] = final_resp_plot['rep'].apply(lambda x: f'Rep-{x}')
sns.violinplot(data=final_resp_plot, x='Replicate', y='p_response', ax=axes[1],
               palette='Set3', inner='box', cut=0)
axes[1].set_xlabel('Replicate')
axes[1].set_ylabel('Response Probability')
axes[1].set_title('(B) Response Probability by Replicate')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig2_response_distributions.png", bbox_inches='tight')
plt.close()
print("Saved fig2_response_distributions.png")

# ----------------------------------------------------------
# Figure 3: Coverage Curves
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3a: Overall coverage curve
axes[0].plot(coverage_df['threshold'], coverage_df['coverage'], 'b-', linewidth=2)
axes[0].fill_between(coverage_df['threshold'], coverage_df['coverage'], alpha=0.2)
axes[0].set_xlabel('Response Probability Threshold')
axes[0].set_ylabel('Fraction of Cells Covered')
axes[0].set_title('(A) Coverage Curve (Adaptive MinSum Vaccine)')
axes[0].axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% coverage')
axes[0].axvline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[0].legend()

# 3b: Per-replicate coverage curves
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for rep in range(10):
    rep_data = final_resp[final_resp['rep'] == rep]
    rep_cov = [(rep_data['p_response'] >= t).mean() for t in thresholds]
    axes[1].plot(thresholds, rep_cov, color=colors[rep], alpha=0.7, label=f'Rep-{rep}')

axes[1].set_xlabel('Response Probability Threshold')
axes[1].set_ylabel('Fraction of Cells Covered')
axes[1].set_title('(B) Per-Replicate Coverage Curves')
axes[1].legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig3_coverage_curves.png", bbox_inches='tight')
plt.close()
print("Saved fig3_coverage_curves.png")

# ----------------------------------------------------------
# Figure 4: Vaccine Element Heatmap (per-element, per-replicate scores)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Build matrix: element × replicate mean response
elem_rep_matrix = pd.DataFrame(index=sorted(vaccine_comp['peptide'].values), columns=range(10))
for rep in range(10):
    df = rep_scores[rep]
    for elem in elem_rep_matrix.index:
        elem_data = df[df['vaccine_element'] == elem]
        elem_rep_matrix.loc[elem, rep] = elem_data['p_response'].mean()

elem_rep_matrix = elem_rep_matrix.astype(float)
sns.heatmap(elem_rep_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
            cbar_kws={'label': 'Mean Response Probability'},
            xticklabels=[f'Rep-{i}' for i in range(10)])
ax.set_xlabel('Replicate')
ax.set_ylabel('Vaccine Element (Mutation)')
ax.set_title('Mean Per-Cell Response Probability by Vaccine Element and Replicate')

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig4_element_heatmap.png", bbox_inches='tight')
plt.close()
print("Saved fig4_element_heatmap.png")

# ----------------------------------------------------------
# Figure 5: IoU Matrix
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

mask = np.zeros_like(iou_matrix)
# Don't mask anything - show full matrix
sns.heatmap(iou_df, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            vmin=0, vmax=1, cbar_kws={'label': 'IoU Score'})
ax.set_title('IoU of Vaccine Compositions Across Replicates')

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig5_iou_matrix.png", bbox_inches='tight')
plt.close()
print("Saved fig5_iou_matrix.png")

# ----------------------------------------------------------
# Figure 6: Runtime vs Population Size
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 6a: Runtime by sample
for sample in sorted(runtime_data['SampleID'].unique()):
    subset = runtime_data[runtime_data['SampleID'] == sample]
    axes[0].plot(subset['PopulationSize'], subset['RunTime'], 'o-', label=f'Sample {sample}')

axes[0].set_xlabel('Population Size (number of cells)')
axes[0].set_ylabel('Runtime (seconds)')
axes[0].set_title('(A) Optimization Runtime vs Population Size')
axes[0].legend(fontsize=8)
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# 6b: Runtime scaling (log-log)
mean_runtime = runtime_data.groupby('PopulationSize')['RunTime'].agg(['mean', 'std']).reset_index()
axes[1].errorbar(mean_runtime['PopulationSize'], mean_runtime['mean'], 
                 yerr=mean_runtime['std'], fmt='o-', color='steelblue', capsize=5, linewidth=2)
axes[1].set_xlabel('Population Size (number of cells)')
axes[1].set_ylabel('Mean Runtime (seconds)')
axes[1].set_title('(B) Mean Runtime with Standard Deviation')
axes[1].set_xscale('log')
axes[1].set_yscale('log')

# Fit power law
from numpy.polynomial import polynomial as P
log_pop = np.log10(mean_runtime['PopulationSize'].values)
log_rt = np.log10(mean_runtime['mean'].values)
coeffs = np.polyfit(log_pop, log_rt, 1)
x_fit = np.linspace(log_pop.min(), log_pop.max(), 100)
y_fit = np.polyval(coeffs, x_fit)
axes[1].plot(10**x_fit, 10**y_fit, 'r--', alpha=0.7, 
             label=f'Power law fit: slope={coeffs[0]:.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig6_runtime_analysis.png", bbox_inches='tight')
plt.close()
print("Saved fig6_runtime_analysis.png")

# ----------------------------------------------------------
# Figure 7: Per-Element Contribution Analysis
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 7a: Bar chart of element contributions
elem_sorted = elem_summary_df.sort_values('mean_p_response', ascending=True)
colors_bar = ['steelblue' if v else 'lightcoral' for v in elem_sorted['in_vaccine']]
axes[0].barh(elem_sorted['element'], elem_sorted['mean_p_response'], 
             xerr=elem_sorted['std_p_response'], color=colors_bar, capsize=3)
axes[0].set_xlabel('Mean Response Probability')
axes[0].set_ylabel('Mutation')
axes[0].set_title('(A) Per-Element Mean Response Probability')
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label='In Vaccine'),
                   Patch(facecolor='lightcoral', label='Not in Vaccine')]
axes[0].legend(handles=legend_elements)

# 7b: Number of cells with meaningful response per element
elem_sorted2 = elem_summary_df.sort_values('mean_cells_with_response', ascending=True)
colors_bar2 = ['steelblue' if v else 'lightcoral' for v in elem_sorted2['in_vaccine']]
axes[1].barh(elem_sorted2['element'], elem_sorted2['mean_cells_with_response'],
             color=colors_bar2)
axes[1].set_xlabel('Mean Number of Cells with Response (p > 0.01)')
axes[1].set_ylabel('Mutation')
axes[1].set_title('(B) Cell Coverage per Vaccine Element')
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig7_element_contributions.png", bbox_inches='tight')
plt.close()
print("Saved fig7_element_contributions.png")

# ----------------------------------------------------------
# Figure 8: Strategy Comparison
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 8a: Distribution comparison
for name, resp in strategies.items():
    axes[0].hist(resp, bins=50, alpha=0.5, label=name, density=True)
axes[0].set_xlabel('Per-Cell Response Probability')
axes[0].set_ylabel('Density')
axes[0].set_title('(A) Response Probability Distributions by Strategy')
axes[0].legend()

# 8b: Coverage comparison
for name, resp in strategies.items():
    cov = [(resp >= t).mean() for t in thresholds]
    axes[1].plot(thresholds, cov, linewidth=2, label=name)
axes[1].set_xlabel('Response Probability Threshold')
axes[1].set_ylabel('Fraction of Cells Covered')
axes[1].set_title('(B) Coverage Curves by Strategy')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig8_strategy_comparison.png", bbox_inches='tight')
plt.close()
print("Saved fig8_strategy_comparison.png")

# ----------------------------------------------------------
# Figure 9: Cell Vulnerability Analysis
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Identify hardest-to-cover cells
cell_agg_mean = cell_agg_df.groupby('cell_id')['p_response'].mean().reset_index()
cell_agg_mean.columns = ['cell_id', 'mean_p_response']
cell_agg_mean = cell_agg_mean.sort_values('mean_p_response')

# 9a: Distribution of mean cell response
axes[0].bar(range(len(cell_agg_mean)), cell_agg_mean['mean_p_response'].values,
            color='steelblue', alpha=0.8)
axes[0].set_xlabel('Cell Index (sorted by response)')
axes[0].set_ylabel('Mean Response Probability')
axes[0].set_title('(A) Per-Cell Mean Response (Sorted)')
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='p=0.5 threshold')
axes[0].legend()

# 9b: Heatmap of cell × element for a representative replicate
rep0 = rep_scores[0]
rep0_vac = rep0[rep0['vaccine_element'].isin(vaccine_comp['peptide'].values)]
pivot = rep0_vac.pivot_table(index='cell_id', columns='vaccine_element', values='p_response')
# Show subset of cells (sorted by total response)
cell_order = pivot.sum(axis=1).sort_values(ascending=False).index[:30]
sns.heatmap(pivot.loc[cell_order], cmap='RdYlGn', ax=axes[1], 
            cbar_kws={'label': 'Response Probability'},
            yticklabels=True)
axes[1].set_xlabel('Vaccine Element')
axes[1].set_ylabel('Cell ID')
axes[1].set_title('(B) Cell × Element Response (Rep-0, Top 30 Cells)')

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig9_cell_vulnerability.png", bbox_inches='tight')
plt.close()
print("Saved fig9_cell_vulnerability.png")

# ============================================================
# Save summary statistics
# ============================================================
summary = {
    "vaccine_composition": {
        "objective": "MinSum",
        "budget": 10,
        "method": "adaptive",
        "selected_elements": sorted(minsum_selection),
        "selection_consistent_across_replicates": all_same
    },
    "response_probability": {
        "mean": float(final_resp['p_response'].mean()),
        "median": float(final_resp['p_response'].median()),
        "std": float(final_resp['p_response'].std()),
        "min": float(final_resp['p_response'].min()),
        "max": float(final_resp['p_response'].max())
    },
    "coverage": {
        "at_0.5": float((final_resp['p_response'] >= 0.5).mean()),
        "at_0.7": float((final_resp['p_response'] >= 0.7).mean()),
        "at_0.8": float((final_resp['p_response'] >= 0.8).mean()),
        "at_0.9": float((final_resp['p_response'] >= 0.9).mean()),
        "at_0.95": float((final_resp['p_response'] >= 0.95).mean())
    },
    "iou": {
        "mean_pairwise": float(np.mean(iou_matrix[np.triu_indices(10, k=1)])),
        "min_pairwise": float(np.min(iou_matrix[np.triu_indices(10, k=1)])),
        "max_pairwise": float(np.max(iou_matrix[np.triu_indices(10, k=1)]))
    },
    "runtime": {
        "power_law_exponent": float(coeffs[0]),
        "mean_runtime_100_cells": float(runtime_data[runtime_data['PopulationSize']==100]['RunTime'].mean()),
        "mean_runtime_10000_cells": float(runtime_data[runtime_data['PopulationSize']==10000]['RunTime'].mean())
    }
}

with open(f"{OUT_DIR}/summary_statistics.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== ALL FIGURES AND OUTPUTS GENERATED ===")
print(f"Summary: {json.dumps(summary, indent=2)}")
