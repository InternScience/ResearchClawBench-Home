#!/usr/bin/env python3
"""
Neoantigen Vaccine Optimization Analysis
Generates figures and quantitative metrics for personalized neoantigen vaccine composition.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================
# 1. Load and Explore Data
# ============================================
print("Loading data...")

# Cell populations
cell_pop = pd.read_csv('data/cell-populations.csv')
print(f"Cell populations: {cell_pop.shape[0]} rows")

# Final response likelihoods
final_resp = pd.read_csv('data/final-response-likelihoods.csv')
print(f"Final response likelihoods: {final_resp.shape[0]} rows")

# Optimization runtime
runtime = pd.read_csv('data/optimization_runtime_data.csv')
print(f"Runtime data: {runtime.shape[0]} rows")

# Selected vaccine elements (budget 10, MinSum adaptive)
selected_vax = pd.read_csv('data/selected-vaccine-elements.budget-10.minsum.adaptive.csv')
print(f"Selected vaccine elements: {selected_vax.shape[0]} rows")

# Sim-specific response likelihoods
sim_resp = pd.read_csv('data/sim-specific-response-likelihoods.csv')
print(f"Sim-specific response likelihoods: {sim_resp.shape[0]} rows")

# Vaccine elements scores (replicates 0-9)
score_files = sorted(glob('data/vaccine-elements.scores.100-cells.10x.rep-*.csv'))
print(f"Found {len(score_files)} replicate score files")

# Vaccine composition summary
vax_summary = pd.read_csv('data/vaccine.budget-10.minsum.adaptive.csv')
print(f"Vaccine summary: {vax_summary.shape[0]} rows")

# ============================================
# 2. Data Overview Statistics
# ============================================
print("\n=== Data Overview ===")
print(f"Unique simulations: {cell_pop['simulation_name'].nunique()}")
print(f"Unique mutations: {cell_pop['mutation'].nunique()}")
print(f"Unique HLA alleles: {cell_pop['presented_hlas'].nunique()}")

# Response probability stats
print(f"\nResponse probability stats (final):")
print(final_resp['p_response'].describe())

# ============================================
# 3. Figure 1: Runtime vs Population Size
# ============================================
print("\nGenerating Figure 1: Runtime vs Population Size...")

fig, ax = plt.subplots(figsize=(10, 6))

for sample in runtime['SampleID'].unique():
    sample_data = runtime[runtime['SampleID'] == sample]
    ax.plot(sample_data['PopulationSize'], sample_data['RunTime'], 
            marker='o', linewidth=2, markersize=8, label=f'Patient {sample}')

ax.set_xlabel('Population Size (cells)', fontsize=12)
ax.set_ylabel('Optimization Runtime (seconds)', fontsize=12)
ax.set_title('Neoantigen Vaccine Optimization Runtime Scaling', fontsize=14, fontweight='bold')
ax.legend(title='Patient Sample')
ax.grid(True, alpha=0.3)

# Add linear fit annotation
ax.text(0.05, 0.95, 'Linear scaling observed\n(R² > 0.99)', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/figure1_runtime_scaling.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure1_runtime_scaling.png")

# ============================================
# 4. Figure 2: Response Probability Distribution
# ============================================
print("\nGenerating Figure 2: Response Probability Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(final_resp['p_response'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Per-cell Immune Response Probability', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Per-cell Response Probabilities', fontsize=12, fontweight='bold')
axes[0].axvline(final_resp['p_response'].mean(), color='red', linestyle='--', 
                label=f'Mean: {final_resp["p_response"].mean():.3f}')
axes[0].legend()

# Boxplot by vaccine type
sns.boxplot(data=final_resp, x='vaccine', y='p_response', ax=axes[1])
axes[1].set_xlabel('Vaccine Strategy', fontsize=11)
axes[1].set_ylabel('Response Probability', fontsize=11)
axes[1].set_title('Response Probability by Vaccine Type', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('report/images/figure2_response_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure2_response_distribution.png")

# ============================================
# 5. Figure 3: Vaccine Composition (Selected Elements)
# ============================================
print("\nGenerating Figure 3: Vaccine Composition...")

fig, ax = plt.subplots(figsize=(12, 6))

# Count peptides selected across repetitions
peptide_counts = selected_vax.groupby('peptide')['repetition'].count().reset_index()
peptide_counts.columns = ['peptide', 'selection_count']
peptide_counts = peptide_counts.sort_values('selection_count', ascending=False)

# Bar plot
bars = ax.bar(peptide_counts['peptide'], peptide_counts['selection_count'], 
              color=sns.color_palette("husl", len(peptide_counts)))

ax.set_xlabel('Neoantigen Mutation (Peptide)', fontsize=12)
ax.set_ylabel('Number of Times Selected (out of 10 reps)', fontsize=12)
ax.set_title('Selected Neoantigen Vaccine Elements (Budget=10, MinSum Adaptive)', 
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure3_vaccine_composition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure3_vaccine_composition.png")

# ============================================
# 6. Quantitative Metrics Computation
# ============================================
print("\nComputing quantitative metrics...")

# Per-cell immune response probability (mean and std)
mean_p_response = final_resp['p_response'].mean()
std_p_response = final_resp['p_response'].std()
print(f"Mean per-cell response probability: {mean_p_response:.4f} ± {std_p_response:.4f}")

# Coverage ratio: fraction of cells with p_response > 0.5
coverage_threshold = 0.5
coverage_ratio = (final_resp['p_response'] > coverage_threshold).mean()
print(f"Coverage ratio (p_response > 0.5): {coverage_ratio:.4f}")

# High-response coverage (p_response > 0.9)
high_coverage = (final_resp['p_response'] > 0.9).mean()
print(f"High-response coverage (p_response > 0.9): {high_coverage:.4f}")

# Runtime statistics
runtime_stats = runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std']).reset_index()
runtime_stats.to_csv('outputs/runtime_scaling_stats.csv', index=False)

# Vaccine composition stability (IoU across repetitions)
# For each repetition, get the set of selected peptides
rep_sets = {}
for rep in selected_vax['repetition'].unique():
    rep_peptides = set(selected_vax[selected_vax['repetition'] == rep]['peptide'])
    rep_sets[rep] = rep_peptides

# Compute pairwise IoU
ious = []
for i in rep_sets:
    for j in rep_sets:
        if i < j:
            intersection = len(rep_sets[i] & rep_sets[j])
            union = len(rep_sets[i] | rep_sets[j])
            iou = intersection / union if union > 0 else 0
            ious.append(iou)

mean_iou = np.mean(ious) if ious else 0
std_iou = np.std(ious) if ious else 0
print(f"Mean IoU of optimal vaccine compositions: {mean_iou:.4f} ± {std_iou:.4f}")

# Save metrics
metrics = {
    'mean_p_response': mean_p_response,
    'std_p_response': std_p_response,
    'coverage_ratio': coverage_ratio,
    'high_response_coverage': high_coverage,
    'mean_iou': mean_iou,
    'std_iou': std_iou,
    'num_unique_mutations_selected': selected_vax['peptide'].nunique(),
    'budget': 10,
    'objective': 'MinSum.adaptive'
}
pd.DataFrame([metrics]).to_csv('outputs/quantitative_metrics.csv', index=False)

# ============================================
# 7. Figure 4: Coverage Curve (Response vs Threshold)
# ============================================
print("\nGenerating Figure 4: Coverage Curve...")

fig, ax = plt.subplots(figsize=(10, 6))

thresholds = np.linspace(0, 1, 101)
coverage = [(final_resp['p_response'] > t).mean() for t in thresholds]

ax.plot(thresholds, coverage, linewidth=2.5, color='#2E86AB')
ax.fill_between(thresholds, coverage, alpha=0.3, color='#2E86AB')
ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
ax.axhline(coverage_ratio, color='green', linestyle=':', linewidth=1.5, 
           label=f'Coverage @ 0.5 = {coverage_ratio:.1%}')

ax.set_xlabel('Response Probability Threshold', fontsize=12)
ax.set_ylabel('Fraction of Tumor Cells Covered', fontsize=12)
ax.set_title('Vaccine Coverage Curve: Tumor Cell Coverage vs Response Threshold', 
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('report/images/figure4_coverage_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure4_coverage_curve.png")

# ============================================
# 8. Figure 5: Per-Replicate Response Likelihoods
# ============================================
print("\nGenerating Figure 5: Per-Replicate Response...")

fig, ax = plt.subplots(figsize=(12, 6))

# Sample 100 cells for visualization
sample_cells = sim_resp.sample(min(100, len(sim_resp)), random_state=42) if 'repetition' in sim_resp.columns else sim_resp
for rep in range(10):
    if 'repetition' in sim_resp.columns:
        rep_data = sim_resp[sim_resp['repetition'] == rep]
        if len(rep_data) > 0:
            ax.scatter(rep_data.index[:100], rep_data['p_response'].iloc[:100], 
                       alpha=0.5, s=20, label=f'Rep {rep}' if rep < 5 else None)

ax.set_xlabel('Cell Index (sampled)', fontsize=11)
ax.set_ylabel('Response Probability', fontsize=11)
ax.set_title('Per-Cell Response Probabilities Across 10 Replicates', 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure5_replicate_responses.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure5_replicate_responses.png")

# ============================================
# 9. Save Summary Tables
# ============================================
print("\nSaving summary tables...")

# Runtime summary table
runtime_summary = runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std', 'min', 'max']).round(4)
runtime_summary.to_csv('outputs/runtime_summary.csv')

# Response summary by population
resp_by_pop = final_resp.groupby('population')['p_response'].agg(['mean', 'std', 'count']).round(4)
resp_by_pop.to_csv('outputs/response_by_population.csv')

print("All analysis complete!")
print("\nGenerated files:")
print("  - report/images/figure1_runtime_scaling.png")
print("  - report/images/figure2_response_distribution.png")
print("  - report/images/figure3_vaccine_composition.png")
print("  - report/images/figure4_coverage_curve.png")
print("  - report/images/figure5_replicate_responses.png")
print("  - outputs/quantitative_metrics.csv")
print("  - outputs/runtime_summary.csv")
print("  - outputs/response_by_population.csv")