#!/usr/bin/env python3
"""
Neoantigen Vaccine Composition Optimization Analysis
Analyzes patient-specific neoantigen vaccine data to understand:
1. Vaccine composition and selection patterns
2. Immune response probability distributions
3. Optimization runtime scaling
4. Coverage and efficacy metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from itertools import combinations

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directories
OUTPUT_DIR = Path("outputs")
REPORT_IMG_DIR = Path("report/images")
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Loading data files...")
print("=" * 60)

# Load all data files
cell_pop = pd.read_csv('data/cell-populations.csv')
final_resp = pd.read_csv('data/final-response-likelihoods.csv')
opt_runtime = pd.read_csv('data/optimization_runtime_data.csv')
selected_elements = pd.read_csv('data/selected-vaccine-elements.budget-10.minsum.adaptive.csv')
sim_resp = pd.read_csv('data/sim-specific-response-likelihoods.csv')
vaccine_budget = pd.read_csv('data/vaccine.budget-10.minsum.adaptive.csv')

# Load all replicate vaccine element scores
rep_scores = {}
for i in range(10):
    rep_scores[i] = pd.read_csv(f'data/vaccine-elements.scores.100-cells.10x.rep-{i}.csv')

print(f"Cell populations: {cell_pop.shape}")
print(f"Final response likelihoods: {final_resp.shape}")
print(f"Optimization runtime data: {opt_runtime.shape}")
print(f"Selected vaccine elements: {selected_elements.shape}")
print(f"Sim-specific response likelihoods: {sim_resp.shape}")
print(f"Vaccine budget: {vaccine_budget.shape}")

# ============================================================================
# Figure 1: Vaccine Composition Analysis
# ============================================================================
print("\n" + "=" * 60)
print("Figure 1: Vaccine Composition Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1a: Selected vaccine elements across repetitions
selected_by_rep = selected_elements.groupby('repetition')['peptide'].apply(list).reset_index()
rep_counts = selected_elements.groupby('repetition')['peptide'].count()

axes[0].bar(range(10), rep_counts.values, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Repetition', fontsize=12)
axes[0].set_ylabel('Number of Selected Elements', fontsize=12)
axes[0].set_title('Selected Elements per Repetition', fontsize=14)
axes[0].set_xticks(range(10))

# 1b: Frequency of each mutation across all repetitions
mutation_freq = selected_elements['peptide'].value_counts().sort_values(ascending=False)
axes[1].barh(range(len(mutation_freq)), mutation_freq.values, color='coral', edgecolor='black', alpha=0.8)
axes[1].set_yticks(range(len(mutation_freq)))
axes[1].set_yticklabels(mutation_freq.index, fontsize=10)
axes[1].set_xlabel('Frequency Across Repetitions', fontsize=12)
axes[1].set_title('Mutation Selection Frequency', fontsize=14)
axes[1].invert_yaxis()

# 1c: Vaccine composition (from budget file)
axes[2].barh(range(len(vaccine_budget)), vaccine_budget['counts'].values, 
             color='seagreen', edgecolor='black', alpha=0.8)
axes[2].set_yticks(range(len(vaccine_budget)))
axes[2].set_yticklabels(vaccine_budget['peptide'].values, fontsize=10)
axes[2].set_xlabel('Counts (Cells Covered)', fontsize=12)
axes[2].set_title('Optimal Vaccine Composition\n(Budget=10, MinSum)', fontsize=14)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_1_vaccine_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_1_vaccine_composition.png")

# ============================================================================
# Figure 2: Immune Response Probability Distributions
# ============================================================================
print("\n" + "=" * 60)
print("Figure 2: Immune Response Probability Distributions")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 2a: Distribution of p_response across all cells (combined)
axes[0, 0].hist(final_resp['p_response'], bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7, density=True)
axes[0, 0].axvline(final_resp['p_response'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {final_resp["p_response"].mean():.3f}')
axes[0, 0].axvline(final_resp['p_response'].median(), color='orange', 
                   linestyle='--', linewidth=2, label=f'Median: {final_resp["p_response"].median():.3f}')
axes[0, 0].set_xlabel('p_response', fontsize=12)
axes[0, 0].set_ylabel('Density', fontsize=12)
axes[0, 0].set_title('Distribution of Per-Cell Response Probability\n(All Replicates Combined)', fontsize=14)
axes[0, 0].legend(fontsize=10)

# 2b: Box plot of p_response by repetition
reps = final_resp['population'].str.extract(r',\s*(\d+)$')[0].astype(float).astype('Int64')
final_resp_copy = final_resp.copy()
final_resp_copy['repetition'] = reps
axes[0, 1].boxplot([final_resp_copy[final_resp_copy['repetition'] == r]['p_response'].values 
                    for r in range(10)], 
                   labels=range(10))
axes[0, 1].set_xlabel('Repetition', fontsize=12)
axes[0, 1].set_ylabel('p_response', fontsize=12)
axes[0, 1].set_title('Response Probability by Repetition', fontsize=14)

# 2c: Sim-specific response likelihoods (by vaccine)
# Parse the repetition from vaccine name
sim_resp_copy = sim_resp.copy()
sim_resp_copy['repetition'] = sim_resp_copy['vaccine'].str.extract(r'rep-(\d+)')[0].astype(int)
for rep in range(10):
    rep_data = sim_resp_copy[sim_resp_copy['repetition'] == rep]['p_response']
    axes[1, 0].plot(sorted(rep_data.values), 
                   np.linspace(0, 1, len(rep_data)), 
                   label=f'Rep {rep}', alpha=0.7)
axes[1, 0].set_xlabel('p_response', fontsize=12)
axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 0].set_title('CDF of Response Probability\nby Replicate', fontsize=14)
axes[1, 0].legend(fontsize=8, ncol=2)

# 2d: Log response probability distribution
axes[1, 1].hist(final_resp['log_p_response'], bins=50, color='mediumpurple', 
                edgecolor='black', alpha=0.7, density=True)
axes[1, 1].set_xlabel('log(p_response)', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('Distribution of Log Response Probability', fontsize=14)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_2_response_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_2_response_distributions.png")

# ============================================================================
# Figure 3: Coverage Analysis
# ============================================================================
print("\n" + "=" * 60)
print("Figure 3: Coverage Analysis")
print("=" * 60)

# Compute coverage at different thresholds
thresholds = np.arange(0.1, 1.0, 0.05)
coverage_data = []

for threshold in thresholds:
    # Combined coverage
    combined_coverage = (final_resp['p_response'] > threshold).mean()
    coverage_data.append({
        'threshold': threshold,
        'coverage_combined': combined_coverage
    })
    
coverage_df = pd.DataFrame(coverage_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 3a: Coverage curve (fraction of cells with p_response > threshold)
axes[0].plot(coverage_df['threshold'], coverage_df['coverage_combined'], 
             'o-', color='steelblue', linewidth=2, markersize=6)
axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold = 0.5')
axes[0].set_xlabel('Response Probability Threshold', fontsize=12)
axes[0].set_ylabel('Fraction of Cells Covered', fontsize=12)
axes[0].set_title('Vaccine Coverage Curve\n(Fraction of Cells with p > threshold)', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.05)

# 3b: Per-mutation coverage (from vaccine-elements scores rep-0)
# Each cell is covered by a vaccine element if p_response > 0 for that element
rep0 = rep_scores[0]
mutation_coverage = rep0.groupby('vaccine_element')['p_response'].apply(
    lambda x: (x > 0.01).mean()
).sort_values(ascending=False)

axes[1].barh(range(len(mutation_coverage)), mutation_coverage.values, 
             color='coral', edgecolor='black', alpha=0.8)
axes[1].set_yticks(range(len(mutation_coverage)))
axes[1].set_yticklabels(mutation_coverage.index, fontsize=10)
axes[1].set_xlabel('Coverage Ratio (p_response > 0.01)', fontsize=12)
axes[1].set_title('Per-Mutation Cell Coverage\n(Replicate 0)', fontsize=14)
axes[1].invert_yaxis()

# 3c: Number of presented peptides vs p_response
axes[2].scatter(final_resp['num_presented_peptides'], final_resp['p_response'], 
                alpha=0.3, s=20, color='steelblue')
# Add trend line
z = np.polyfit(final_resp['num_presented_peptides'], final_resp['p_response'], 1)
p = np.poly1d(z)
x_range = np.linspace(final_resp['num_presented_peptides'].min(), 
                      final_resp['num_presented_peptides'].max(), 100)
axes[2].plot(x_range, p(x_range), 'r--', linewidth=2, label=f'Trend: slope={z[0]:.4f}')
axes[2].set_xlabel('Number of Presented Peptides', fontsize=12)
axes[2].set_ylabel('p_response', fontsize=12)
axes[2].set_title('Presented Peptides vs Response', fontsize=14)
axes[2].legend(fontsize=10)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_3_coverage_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_3_coverage_analysis.png")

# ============================================================================
# Figure 4: Optimization Runtime Scaling
# ============================================================================
print("\n" + "=" * 60)
print("Figure 4: Optimization Runtime Scaling")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4a: Runtime vs Population Size (all samples)
sample_ids = opt_runtime['SampleID'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(sample_ids)))

for idx, sample_id in enumerate(sample_ids):
    sample_data = opt_runtime[opt_runtime['SampleID'] == sample_id]
    axes[0].plot(sample_data['PopulationSize'], sample_data['RunTime'], 
                'o-', color=colors[idx], linewidth=2, markersize=6, 
                label=f'Sample {sample_id}')

axes[0].set_xlabel('Population Size', fontsize=12)
axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
axes[0].set_title('Optimization Runtime vs Population Size', fontsize=14)
axes[0].legend(fontsize=8, ncol=2)
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# 4b: Average runtime scaling
avg_runtime = opt_runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std']).reset_index()
axes[1].errorbar(avg_runtime['PopulationSize'], avg_runtime['mean'], 
                yerr=avg_runtime['std'], fmt='o-', color='steelblue', 
                linewidth=2, markersize=8, capsize=5, capthick=2)
axes[1].set_xlabel('Population Size', fontsize=12)
axes[1].set_ylabel('Runtime (seconds)', fontsize=12)
axes[1].set_title('Average Runtime Scaling\n(Mean ± SD across patients)', fontsize=14)
axes[1].set_xscale('log')
axes[1].set_yscale('log')

# Fit power law
log_pop = np.log10(avg_runtime['PopulationSize'].values)
log_time = np.log10(avg_runtime['mean'].values)
coeffs = np.polyfit(log_pop, log_time, 1)
x_fit = np.linspace(log_pop.min(), log_pop.max(), 100)
y_fit = np.polyval(coeffs, x_fit)
axes[1].plot(10**x_fit, 10**y_fit, 'r--', linewidth=2, 
            label=f'Power law: t ∝ N^{coeffs[0]:.2f}')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_4_runtime_scaling.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_4_runtime_scaling.png")

# ============================================================================
# Figure 5: IoU Analysis of Vaccine Compositions
# ============================================================================
print("\n" + "=" * 60)
print("Figure 5: IoU Analysis of Vaccine Compositions")
print("=" * 60)

# Get vaccine elements per repetition
elements_per_rep = {}
for rep in range(10):
    rep_data = selected_elements[selected_elements['repetition'] == rep]
    elements_per_rep[rep] = set(rep_data['peptide'].values)

# Compute pairwise IoU
iou_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        intersection = len(elements_per_rep[i] & elements_per_rep[j])
        union = len(elements_per_rep[i] | elements_per_rep[j])
        iou_matrix[i, j] = intersection / union if union > 0 else 0

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5a: IoU heatmap
im = axes[0].imshow(iou_matrix, cmap='YlOrRd', vmin=0, vmax=1)
axes[0].set_xticks(range(10))
axes[0].set_yticks(range(10))
axes[0].set_xlabel('Repetition', fontsize=12)
axes[0].set_ylabel('Repetition', fontsize=10)
axes[0].set_title('IoU of Vaccine Compositions\nAcross Replicates', fontsize=14)
plt.colorbar(im, ax=axes[0], label='IoU')

# Add text annotations
for i in range(10):
    for j in range(10):
        axes[0].text(j, i, f'{iou_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8,
                    color='white' if iou_matrix[i, j] > 0.5 else 'black')

# 5b: IoU distribution (off-diagonal)
off_diag_iou = iou_matrix[np.triu_indices(10, k=1)]
axes[1].hist(off_diag_iou, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(off_diag_iou.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean IoU: {off_diag_iou.mean():.3f}')
axes[1].set_xlabel('IoU', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of Pairwise IoU\n(Off-diagonal Elements)', fontsize=14)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_5_iou_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_5_iou_analysis.png")

# ============================================================================
# Figure 6: Per-Mutation Response Analysis
# ============================================================================
print("\n" + "=" * 60)
print("Figure 6: Per-Mutation Response Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 6a: Mean p_response per vaccine element (aggregated across all replicates)
all_rep_responses = []
for rep in range(10):
    rep_data = rep_scores[rep].copy()
    rep_data['replicate'] = rep
    all_rep_responses.append(rep_data)
all_responses = pd.concat(all_rep_responses)

# Aggregate by vaccine element
element_stats = all_responses.groupby('vaccine_element')['p_response'].agg(['mean', 'std']).reset_index()
element_stats = element_stats.sort_values('mean', ascending=False)

axes[0].barh(range(len(element_stats)), element_stats['mean'].values, 
             xerr=element_stats['std'].values, 
             color='steelblue', edgecolor='black', alpha=0.8, capsize=3)
axes[0].set_yticks(range(len(element_stats)))
axes[0].set_yticklabels(element_stats['vaccine_element'].values, fontsize=10)
axes[0].set_xlabel('Mean p_response ± SD', fontsize=12)
axes[0].set_title('Mean Response Probability\nper Vaccine Element', fontsize=14)
axes[0].invert_yaxis()

# 6b: Box plot of p_response for top elements
top_elements = element_stats.head(6)['vaccine_element'].values
top_data = all_responses[all_responses['vaccine_element'].isin(top_elements)]
box_data = [top_data[top_data['vaccine_element'] == elem]['p_response'].values 
            for elem in top_elements]
bp = axes[1].boxplot(box_data, labels=top_elements, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_elements)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_xlabel('Vaccine Element', fontsize=12)
axes[1].set_ylabel('p_response', fontsize=12)
axes[1].set_title('Response Distribution\nfor Top 6 Elements', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(REPORT_IMG_DIR / 'figure_6_mutation_response.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure_6_mutation_response.png")

# ============================================================================
# Quantitative Metrics Calculation
# ============================================================================
print("\n" + "=" * 60)
print("Computing Quantitative Metrics")
print("=" * 60)

# 1. Per-cell immune response probability statistics
p_response_stats = {
    'mean': float(final_resp['p_response'].mean()),
    'std': float(final_resp['p_response'].std()),
    'median': float(final_resp['p_response'].median()),
    'min': float(final_resp['p_response'].min()),
    'max': float(final_resp['p_response'].max()),
    'q25': float(final_resp['p_response'].quantile(0.25)),
    'q75': float(final_resp['p_response'].quantile(0.75)),
}

print(f"Per-cell p_response statistics:")
print(f"  Mean: {p_response_stats['mean']:.4f}")
print(f"  Std: {p_response_stats['std']:.4f}")
print(f"  Median: {p_response_stats['median']:.4f}")
print(f"  Min: {p_response_stats['min']:.6f}")
print(f"  Max: {p_response_stats['max']:.6f}")

# 2. Coverage ratio at different thresholds
coverage_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
coverage_ratios = {}
for threshold in coverage_thresholds:
    coverage_ratios[f'coverage_p>{threshold}'] = float((final_resp['p_response'] > threshold).mean())
    
print(f"\nCoverage ratios:")
for k, v in coverage_ratios.items():
    print(f"  {k}: {v:.4f}")

# 3. IoU statistics
iou_stats = {
    'mean_iou': float(off_diag_iou.mean()),
    'std_iou': float(off_diag_iou.std()),
    'min_iou': float(off_diag_iou.min()),
    'max_iou': float(off_diag_iou.max()),
    'median_iou': float(np.median(off_diag_iou)),
}

print(f"\nIoU statistics:")
for k, v in iou_stats.items():
    print(f"  {k}: {v:.4f}")

# 4. Runtime scaling
runtime_stats = {
    'power_law_exponent': float(coeffs[0]),
    'runtime_100_cells': float(opt_runtime[opt_runtime['PopulationSize'] == 100]['RunTime'].mean()),
    'runtime_10000_cells': float(opt_runtime[opt_runtime['PopulationSize'] == 10000]['RunTime'].mean()),
}

print(f"\nRuntime scaling:")
print(f"  Power law exponent: {runtime_stats['power_law_exponent']:.2f}")
print(f"  Mean runtime at 100 cells: {runtime_stats['runtime_100_cells']:.3f}s")
print(f"  Mean runtime at 10000 cells: {runtime_stats['runtime_10000_cells']:.3f}s")

# 5. Vaccine composition statistics
vaccine_stats = {
    'num_selected_elements': int(vaccine_budget.shape[0]),
    'budget': 10,
    'objective': 'MinSum',
    'num_unique_mutations_in_pool': int(cell_pop['mutation'].nunique()),
    'selection_rate': float(vaccine_budget.shape[0] / cell_pop['mutation'].nunique()),
}

print(f"\nVaccine composition:")
print(f"  Number of selected elements: {vaccine_stats['num_selected_elements']}")
print(f"  Budget: {vaccine_stats['budget']}")
print(f"  Selection rate: {vaccine_stats['selection_rate']:.2%}")

# 6. Per-repetition consistency
rep_element_sets = [elements_per_rep[i] for i in range(10)]
common_elements = set.intersection(*rep_element_sets)
unique_per_rep = [len(elements_per_rep[i]) for i in range(10)]

consistency_stats = {
    'common_elements_across_all_reps': sorted(list(common_elements)),
    'num_common_elements': len(common_elements),
    'elements_per_rep': unique_per_rep,
    'mean_elements_per_rep': float(np.mean(unique_per_rep)),
}

print(f"\nConsistency across repetitions:")
print(f"  Common elements across all 10 reps: {sorted(list(common_elements))}")
print(f"  Number of common elements: {consistency_stats['num_common_elements']}")

# Combine all metrics
all_metrics = {
    'p_response_stats': p_response_stats,
    'coverage_ratios': coverage_ratios,
    'iou_stats': iou_stats,
    'runtime_stats': runtime_stats,
    'vaccine_stats': vaccine_stats,
    'consistency_stats': consistency_stats,
}

# Save metrics
with open(OUTPUT_DIR / 'vaccine_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)
print(f"\nSaved metrics to {OUTPUT_DIR / 'vaccine_metrics.json'}")

# ============================================================================
# Save coverage analysis data
# ============================================================================
coverage_df.to_csv(OUTPUT_DIR / 'coverage_analysis.csv', index=False)
print(f"Saved coverage analysis to {OUTPUT_DIR / 'coverage_analysis.csv'}")

# Save IoU data
iou_df = pd.DataFrame(iou_matrix, index=range(10), columns=range(10))
iou_df.to_csv(OUTPUT_DIR / 'iou_analysis.csv')
print(f"Saved IoU analysis to {OUTPUT_DIR / 'iou_analysis.csv'}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
