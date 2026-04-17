#!/usr/bin/env python3
"""
Personalized Neoantigen Vaccine Optimization Analysis

This script processes patient-specific sequencing data to determine optimal
personalized neoantigen vaccine composition, compute efficacy metrics, and
document optimization performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob

# Set up paths
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")

# Ensure output directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_IMAGES_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("=" * 60)
print("Personalized Neoantigen Vaccine Optimization Analysis")
print("=" * 60)

# =============================================================================
# 1. Load All Data Files
# =============================================================================
print("\n[1] Loading data files...")

# Cell populations
cell_populations = pd.read_csv(DATA_DIR / "cell-populations.csv")
print(f"  - cell-populations.csv: {len(cell_populations)} rows")

# Final response likelihoods
final_likelihoods = pd.read_csv(DATA_DIR / "final-response-likelihoods.csv")
print(f"  - final-response-likelihoods.csv: {len(final_likelihoods)} rows")

# Sim-specific response likelihoods
sim_likelihoods = pd.read_csv(DATA_DIR / "sim-specific-response-likelihoods.csv")
print(f"  - sim-specific-response-likelihoods.csv: {len(sim_likelihoods)} rows")

# Selected vaccine elements
selected_elements = pd.read_csv(DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
print(f"  - selected-vaccine-elements.budget-10.minsum.adaptive.csv: {len(selected_elements)} rows")

# Vaccine composition
vaccine_composition = pd.read_csv(DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv")
print(f"  - vaccine.budget-10.minsum.adaptive.csv: {len(vaccine_composition)} rows")

# Optimization runtime data
runtime_data = pd.read_csv(DATA_DIR / "optimization_runtime_data.csv")
print(f"  - optimization_runtime_data.csv: {len(runtime_data)} rows")

# Load all replicate score files
replicate_files = sorted(glob.glob(str(DATA_DIR / "vaccine-elements.scores.100-cells.10x.rep-*.csv")))
replicate_scores = {}
for f in replicate_files:
    rep_name = Path(f).stem.split("-")[-1]  # e.g., "rep-0" -> "0"
    replicate_scores[rep_name] = pd.read_csv(f)
    print(f"  - vaccine-elements.scores.100-cells.10x.rep-{rep_name}.csv: {len(replicate_scores[rep_name])} rows")

print(f"\nTotal replicate files loaded: {len(replicate_scores)}")

# =============================================================================
# 2. Data Overview Statistics
# =============================================================================
print("\n[2] Computing data overview statistics...")

# Cell population statistics
unique_cells = cell_populations['cell_ids'].nunique()
unique_peptides = cell_populations['presented_peptides'].nunique()
unique_mutations = cell_populations['mutation'].nunique()
unique_hlas = cell_populations['presented_hlas'].nunique()

print(f"  Unique cells: {unique_cells}")
print(f"  Unique peptides: {unique_peptides}")
print(f"  Unique mutations: {unique_mutations}")
print(f"  Unique HLA alleles: {unique_hlas}")

# Mutation distribution in cell populations
mutation_counts = cell_populations['mutation'].value_counts()
print(f"\nTop mutations by presentation frequency:")
for mut, count in mutation_counts.head(10).items():
    print(f"    {mut}: {count} presentations")

# Vaccine composition summary
print(f"\nVaccine composition (budget=10, MinSum adaptive):")
print(vaccine_composition.to_string(index=False))

# =============================================================================
# 3. Compute Per-Cell Immune Response Probabilities
# =============================================================================
print("\n[3] Computing per-cell immune response probabilities...")

# Aggregate scores across replicates
all_replicate_data = []
for rep_name, df in replicate_scores.items():
    df_copy = df.copy()
    df_copy['replicate'] = int(rep_name)
    all_replicate_data.append(df_copy)

combined_scores = pd.concat(all_replicate_data, ignore_index=True)
print(f"Combined scores shape: {combined_scores.shape}")

# Compute mean response probability per cell across vaccine elements
cell_response_stats = combined_scores.groupby('cell_id')['p_response'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
cell_response_stats.columns = ['cell_id', 'mean_p_response', 'std_p_response', 'min_p_response', 'max_p_response', 'n_elements']

print(f"Cell response statistics computed for {len(cell_response_stats)} cells")
print(f"Mean response probability (across all cells/elements): {cell_response_stats['mean_p_response'].mean():.4f}")
print(f"Std of response probability: {cell_response_stats['mean_p_response'].std():.4f}")

# Save intermediate results
cell_response_stats.to_csv(OUTPUTS_DIR / "cell_response_statistics.csv", index=False)

# =============================================================================
# 4. Analyze Vaccine Element Selection
# =============================================================================
print("\n[4] Analyzing vaccine element selection...")

# Selection frequency across replicates
selection_freq = selected_elements.groupby('peptide').size().reset_index(name='selection_count')
selection_freq['selection_rate'] = selection_freq['selection_count'] / len(replicate_scores)
print("Vaccine element selection frequency across replicates:")
print(selection_freq.to_string(index=False))

# Save selection frequency
selection_freq.to_csv(OUTPUTS_DIR / "vaccine_element_selection_frequency.csv", index=False)

# Compute IoU between replicates
def compute_iou(set1, set2):
    """Compute Intersection over Union between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# Get selected elements per replicate
replicate_selections = {}
for rep_name, df in replicate_scores.items():
    # Get elements that were selected (those appearing in selected_elements for this rep)
    rep_selected = selected_elements[selected_elements['repetition'] == int(rep_name)]['peptide'].tolist()
    replicate_selections[int(rep_name)] = set(rep_selected)

# Compute pairwise IoU
iou_matrix = np.zeros((len(replicate_scores), len(replicate_scores)))
rep_names = sorted([int(r) for r in replicate_scores.keys()])
for i, rep_i in enumerate(rep_names):
    for j, rep_j in enumerate(rep_names):
        iou = compute_iou(replicate_selections[rep_i], replicate_selections[rep_j])
        iou_matrix[i, j] = iou

mean_iou = np.mean([iou_matrix[i, j] for i in range(len(rep_names)) for j in range(i+1, len(rep_names))])
print(f"\nMean pairwise IoU of vaccine compositions: {mean_iou:.4f}")

# Save IoU matrix
iou_df = pd.DataFrame(iou_matrix, index=rep_names, columns=rep_names)
iou_df.to_csv(OUTPUTS_DIR / "vaccine_composition_iou_matrix.csv")

# =============================================================================
# 5. Coverage Ratio Analysis
# =============================================================================
print("\n[5] Computing coverage ratio...")

# From final-response-likelihoods, compute coverage
# Coverage = proportion of cells with response probability above threshold
thresholds = [0.5, 0.7, 0.9]
coverage_results = []

for thresh in thresholds:
    covered = (final_likelihoods['p_response'] >= thresh).sum()
    total = len(final_likelihoods)
    coverage = covered / total
    coverage_results.append({
        'threshold': thresh,
        'covered_cells': covered,
        'total_cells': total,
        'coverage_ratio': coverage
    })

coverage_df = pd.DataFrame(coverage_results)
print("Coverage ratio by threshold:")
print(coverage_df.to_string(index=False))
coverage_df.to_csv(OUTPUTS_DIR / "coverage_ratio.csv", index=False)

# =============================================================================
# 6. Generate Figures
# =============================================================================
print("\n[6] Generating figures...")

# Figure 1: Data Overview - Mutation Distribution
fig1, ax1 = plt.subplots(figsize=(12, 6))
mutation_order = mutation_counts.sort_values(ascending=True).index
ax1.barh(mutation_order, mutation_counts.sort_values())
ax1.set_xlabel('Number of Presentations')
ax1.set_ylabel('Mutation')
ax1.set_title('Figure 1: Mutation Distribution in Cell Populations')
ax1.invert_yaxis()
plt.tight_layout()
fig1.savefig(REPORT_IMAGES_DIR / "fig1_mutation_distribution.png", dpi=150)
plt.close(fig1)
print("  - Saved: fig1_mutation_distribution.png")

# Figure 2: Response Probability Distribution
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(data=final_likelihoods, x='p_response', bins=30, kde=True, ax=ax2)
ax2.set_xlabel('Response Probability (p_response)')
ax2.set_ylabel('Count')
ax2.set_title('Figure 2: Distribution of Cell Response Probabilities')
ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold = 0.5')
ax2.axvline(x=0.9, color='g', linestyle='--', label='Threshold = 0.9')
ax2.legend()
plt.tight_layout()
fig2.savefig(REPORT_IMAGES_DIR / "fig2_response_probability_distribution.png", dpi=150)
plt.close(fig2)
print("  - Saved: fig2_response_probability_distribution.png")

# Figure 3: Vaccine Composition
fig3, ax3 = plt.subplots(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(vaccine_composition)))
bars = ax3.bar(vaccine_composition['peptide'], vaccine_composition['counts'], color=colors)
ax3.set_xlabel('Mutation (Neoantigen)')
ax3.set_ylabel('Selection Count (across replicates)')
ax3.set_title('Figure 3: Vaccine Composition (Budget=10, MinSum Adaptive)')
ax3.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, vaccine_composition['counts']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), 
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
fig3.savefig(REPORT_IMAGES_DIR / "fig3_vaccine_composition.png", dpi=150)
plt.close(fig3)
print("  - Saved: fig3_vaccine_composition.png")

# Figure 4: Coverage Curve
fig4, ax4 = plt.subplots(figsize=(10, 6))
sorted_responses = np.sort(final_likelihoods['p_response'].values)[::-1]
cumulative_coverage = np.arange(1, len(sorted_responses) + 1) / len(sorted_responses)
ax4.plot(sorted_responses, cumulative_coverage, linewidth=2)
ax4.set_xlabel('Response Probability Threshold')
ax4.set_ylabel('Cumulative Coverage Ratio')
ax4.set_title('Figure 4: Coverage Curve - Tumor Cell Coverage vs Response Threshold')
ax4.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(REPORT_IMAGES_DIR / "fig4_coverage_curve.png", dpi=150)
plt.close(fig4)
print("  - Saved: fig4_coverage_curve.png")

# Figure 5: Replicate Comparison - Response Probability by Replicate
fig5, axes5 = plt.subplots(2, 5, figsize=(20, 8))
axes5 = axes5.flatten()

for idx, rep_name in enumerate(sorted(replicate_scores.keys(), key=int)):
    rep_df = replicate_scores[rep_name]
    cell_means = rep_df.groupby('cell_id')['p_response'].mean()
    axes5[idx].hist(cell_means, bins=20, edgecolor='black', alpha=0.7)
    axes5[idx].set_xlabel('Mean p_response')
    axes5[idx].set_ylabel('Cell Count')
    axes5[idx].set_title(f'Replicate {rep_name}\nMean: {cell_means.mean():.3f}')
    axes5[idx].axvline(x=cell_means.mean(), color='r', linestyle='--')

fig5.suptitle('Figure 5: Response Probability Distribution Across Replicates', fontsize=14)
plt.tight_layout()
fig5.savefig(REPORT_IMAGES_DIR / "fig5_replicate_comparison.png", dpi=150)
plt.close(fig5)
print("  - Saved: fig5_replicate_comparison.png")

# Figure 6: Runtime vs Population Size
fig6, ax6 = plt.subplots(figsize=(10, 6))
samples = runtime_data['SampleID'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(samples)))

for idx, sample in enumerate(sorted(samples)):
    sample_data = runtime_data[runtime_data['SampleID'] == sample]
    ax6.plot(sample_data['PopulationSize'], sample_data['RunTime'], 
             marker='o', label=f'Sample {sample}', color=colors[idx], linewidth=2)

ax6.set_xlabel('Population Size')
ax6.set_ylabel('Runtime (seconds)')
ax6.set_title('Figure 6: Optimization Runtime vs Population Size')
ax6.legend(loc='upper left')
ax6.grid(True, alpha=0.3)
plt.tight_layout()
fig6.savefig(REPORT_IMAGES_DIR / "fig6_runtime_vs_population.png", dpi=150)
plt.close(fig6)
print("  - Saved: fig6_runtime_vs_population.png")

# Figure 7: IoU Heatmap
fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.heatmap(iou_df, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax7, 
            square=True, cbar_kws={'label': 'IoU'})
ax7.set_xlabel('Replicate')
ax7.set_ylabel('Replicate')
ax7.set_title('Figure 7: IoU Heatmap - Vaccine Composition Consistency')
plt.tight_layout()
fig7.savefig(REPORT_IMAGES_DIR / "fig7_iou_heatmap.png", dpi=150)
plt.close(fig7)
print("  - Saved: fig7_iou_heatmap.png")

# Figure 8: Cell Response Heatmap (subset)
fig8, ax8 = plt.subplots(figsize=(12, 8))
# Sample subset of cells for visualization
sample_cells = combined_scores[combined_scores['replicate'] == 0]['cell_id'].unique()[:50]
sample_data = combined_scores[(combined_scores['cell_id'].isin(sample_cells)) & (combined_scores['replicate'] == 0)]
pivot_data = sample_data.pivot(index='cell_id', columns='vaccine_element', values='p_response')
sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax8, cbar_kws={'label': 'p_response'})
ax8.set_xlabel('Vaccine Element (Mutation)')
ax8.set_ylabel('Cell ID')
ax8.set_title('Figure 8: Cell Response Heatmap (Replicate 0, 50 cells)')
plt.tight_layout()
fig8.savefig(REPORT_IMAGES_DIR / "fig8_cell_response_heatmap.png", dpi=150)
plt.close(fig8)
print("  - Saved: fig8_cell_response_heatmap.png")

# =============================================================================
# 7. Summary Statistics Output
# =============================================================================
print("\n[7] Saving summary statistics...")

summary_stats = {
    "data_overview": {
        "total_cell_presentations": len(cell_populations),
        "unique_cells": int(unique_cells),
        "unique_peptides": int(unique_peptides),
        "unique_mutations": int(unique_mutations),
        "unique_hla_alleles": int(unique_hlas)
    },
    "vaccine_composition": {
        "budget": 10,
        "objective": "MinSum.adaptive",
        "selected_mutations": vaccine_composition['peptide'].tolist(),
        "total_elements": len(vaccine_composition)
    },
    "efficacy_metrics": {
        "mean_response_probability": float(final_likelihoods['p_response'].mean()),
        "std_response_probability": float(final_likelihoods['p_response'].std()),
        "min_response_probability": float(final_likelihoods['p_response'].min()),
        "max_response_probability": float(final_likelihoods['p_response'].max()),
        "coverage_at_0.5": float(coverage_df[coverage_df['threshold'] == 0.5]['coverage_ratio'].values[0]),
        "coverage_at_0.7": float(coverage_df[coverage_df['threshold'] == 0.7]['coverage_ratio'].values[0]),
        "coverage_at_0.9": float(coverage_df[coverage_df['threshold'] == 0.9]['coverage_ratio'].values[0])
    },
    "iou_analysis": {
        "mean_pairwise_iou": float(mean_iou),
        "min_iou": float(np.min(iou_matrix[iou_matrix < 1.0])) if np.any(iou_matrix < 1.0) else 1.0,
        "max_iou": 1.0
    },
    "runtime_analysis": {
        "samples_analyzed": len(samples),
        "population_sizes_tested": runtime_data['PopulationSize'].unique().tolist(),
        "min_runtime_seconds": float(runtime_data['RunTime'].min()),
        "max_runtime_seconds": float(runtime_data['RunTime'].max()),
        "mean_runtime_seconds": float(runtime_data['RunTime'].mean())
    }
}

with open(OUTPUTS_DIR / "summary_statistics.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("Summary statistics saved to outputs/summary_statistics.json")

# Print key results
print("\n" + "=" * 60)
print("KEY RESULTS SUMMARY")
print("=" * 60)
print(f"\nVaccine Composition (Budget=10):")
for _, row in vaccine_composition.iterrows():
    print(f"  - {row['peptide']}: selected in {row['counts']} replicates")

print(f"\nEfficacy Metrics:")
print(f"  Mean per-cell response probability: {summary_stats['efficacy_metrics']['mean_response_probability']:.4f}")
print(f"  Coverage ratio (threshold=0.5): {summary_stats['efficacy_metrics']['coverage_at_0.5']:.4f}")
print(f"  Coverage ratio (threshold=0.7): {summary_stats['efficacy_metrics']['coverage_at_0.7']:.4f}")
print(f"  Coverage ratio (threshold=0.9): {summary_stats['efficacy_metrics']['coverage_at_0.9']:.4f}")

print(f"\nOptimization Consistency:")
print(f"  Mean pairwise IoU: {summary_stats['iou_analysis']['mean_pairwise_iou']:.4f}")

print(f"\nRuntime Performance:")
print(f"  Runtime range: {summary_stats['runtime_analysis']['min_runtime_seconds']:.3f} - {summary_stats['runtime_analysis']['max_runtime_seconds']:.1f} seconds")
print(f"  Population sizes: {summary_stats['runtime_analysis']['population_sizes_tested']}")

print("\n" + "=" * 60)
print("Analysis complete! All figures saved to report/images/")
print("=" * 60)
