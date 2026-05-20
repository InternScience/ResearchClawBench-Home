"""
Personalized Neoantigen Vaccine Optimization Analysis
======================================================
This script analyzes simulated data from a personalized neoantigen vaccine
optimization pipeline using the MinSum objective with a budget of 10 elements.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

print("=" * 60)
print("Loading data files...")
print("=" * 60)

# Load all data files
cell_pop = pd.read_csv('data/cell-populations.csv')
final_resp = pd.read_csv('data/final-response-likelihoods.csv')
runtime = pd.read_csv('data/optimization_runtime_data.csv')
selected = pd.read_csv('data/selected-vaccine-elements.budget-10.minsum.adaptive.csv')
sim_resp = pd.read_csv('data/sim-specific-response-likelihoods.csv')
vaccine_simple = pd.read_csv('data/vaccine.budget-10.minsum.adaptive.csv')

# Load all 10 replicates of vaccine element scores
reps = []
for i in range(10):
    df = pd.read_csv(f'data/vaccine-elements.scores.100-cells.10x.rep-{i}.csv')
    df['repetition'] = i
    reps.append(df)
scores_all = pd.concat(reps, ignore_index=True)

print(f"cell_populations: {cell_pop.shape}")
print(f"final_response_likelihoods: {final_resp.shape}")
print(f"optimization_runtime: {runtime.shape}")
print(f"selected_vaccine_elements: {selected.shape}")
print(f"sim_specific_response: {sim_resp.shape}")
print(f"vaccine_scores_all: {scores_all.shape}")

# ============================================================
# 1. VACCINE COMPOSITION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("1. VACCINE COMPOSITION ANALYSIS")
print("=" * 60)

# Selected vaccine elements are identical across repetitions
selected_unique = selected['peptide'].unique()
print(f"Selected vaccine elements ({len(selected_unique)}): {selected_unique.tolist()}")

# Check consistency across repetitions
iou_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        set_i = set(selected[selected['repetition'] == i]['peptide'])
        set_j = set(selected[selected['repetition'] == j]['peptide'])
        iou = len(set_i & set_j) / len(set_i | set_j)
        iou_matrix[i, j] = iou

print(f"IoU matrix mean (off-diagonal): {np.mean(iou_matrix[~np.eye(10, dtype=bool)]):.4f}")
print(f"All IoU values are 1.0: {np.all(iou_matrix == 1.0)}")

# Save IoU matrix
pd.DataFrame(iou_matrix, index=[f'rep_{i}' for i in range(10)],
             columns=[f'rep_{i}' for i in range(10)]).to_csv('outputs/iou_matrix.csv')

# ============================================================
# 2. MUTATION COVERAGE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("2. MUTATION COVERAGE ANALYSIS")
print("=" * 60)

# What mutations are present in cell populations?
all_mutations = cell_pop['mutation'].unique()
print(f"All mutations in cell populations: {sorted(all_mutations.tolist())}")
print(f"Number of unique mutations: {len(all_mutations)}")

# Coverage: how many cells have at least one peptide from selected mutations
coverage_by_rep = {}
for rep in range(10):
    rep_cells = cell_pop[cell_pop['repetition'] == rep]
    selected_mutations = set(selected[selected['repetition'] == rep]['peptide'])
    cells_with_selected = rep_cells[rep_cells['mutation'].isin(selected_mutations)]['cell_ids'].nunique()
    total_cells = rep_cells['cell_ids'].nunique()
    coverage = cells_with_selected / total_cells
    coverage_by_rep[rep] = {
        'total_cells': total_cells,
        'covered_cells': cells_with_selected,
        'coverage_ratio': coverage
    }
    print(f"Rep {rep}: {cells_with_selected}/{total_cells} cells covered = {coverage:.4f}")

coverage_df = pd.DataFrame.from_dict(coverage_by_rep, orient='index')
coverage_df.to_csv('outputs/coverage_by_repetition.csv')
print(f"Mean coverage ratio: {coverage_df['coverage_ratio'].mean():.4f} ± {coverage_df['coverage_ratio'].std():.4f}")

# ============================================================
# 3. PER-CELL IMMUNE RESPONSE PROBABILITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. PER-CELL IMMUNE RESPONSE PROBABILITY")
print("=" * 60)

print(final_resp['p_response'].describe())

# Compute coverage at different thresholds
thresholds = np.arange(0.0, 1.01, 0.01)
coverage_curves = []
for rep in range(10):
    rep_final = final_resp[final_resp['population'].str.contains(f', {rep}')]
    for thresh in thresholds:
        covered = (rep_final['p_response'] >= thresh).sum()
        total = len(rep_final)
        coverage_curves.append({
            'repetition': rep,
            'threshold': thresh,
            'coverage': covered / total,
            'num_covered': covered,
            'num_total': total
        })

coverage_curve_df = pd.DataFrame(coverage_curves)
coverage_curve_df.to_csv('outputs/coverage_curves.csv', index=False)

# Also compute from sim-specific data
sim_coverage_curves = []
for rep in range(10):
    rep_sim = sim_resp[sim_resp['vaccine'].str.endswith(f'rep-{rep}')]
    for thresh in thresholds:
        covered = (rep_sim['p_response'] >= thresh).sum()
        total = len(rep_sim)
        sim_coverage_curves.append({
            'repetition': rep,
            'threshold': thresh,
            'coverage': covered / total,
            'num_covered': covered,
            'num_total': total
        })

sim_coverage_curve_df = pd.DataFrame(sim_coverage_curves)
sim_coverage_curve_df.to_csv('outputs/sim_coverage_curves.csv', index=False)

# ============================================================
# 4. VACCINE ELEMENT SCORES ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. VACCINE ELEMENT SCORES ANALYSIS")
print("=" * 60)

# Per-element response statistics
element_stats = scores_all.groupby('vaccine_element')['p_response'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
element_stats = element_stats.sort_values('mean', ascending=False)
print("Top vaccine elements by mean p_response:")
print(element_stats.head(10))
element_stats.to_csv('outputs/element_response_stats.csv', index=False)

# Selected vs non-selected elements
selected_elements = set(selected_unique)
scores_all['is_selected'] = scores_all['vaccine_element'].isin(selected_elements)
selected_stats = scores_all.groupby('is_selected')['p_response'].agg(['mean', 'std', 'median', 'count'])
print("\nSelected vs non-selected element scores:")
print(selected_stats)
selected_stats.to_csv('outputs/selected_vs_nonselected_stats.csv')

# ============================================================
# 5. OPTIMIZATION RUNTIME ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. OPTIMIZATION RUNTIME ANALYSIS")
print("=" * 60)

print(runtime.head())
runtime_summary = runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std', 'median', 'min', 'max'])
print("\nRuntime summary by population size:")
print(runtime_summary)
runtime_summary.to_csv('outputs/runtime_summary.csv')

# Fit scaling model
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * (x ** b)

def linear_law(x, a, b):
    return a * x + b

x = runtime['PopulationSize'].values
y = runtime['RunTime'].values

popt_power, _ = curve_fit(power_law, x, y, p0=[1e-4, 1.0])
popt_linear, _ = curve_fit(linear_law, x, y, p0=[1e-4, 0.0])

print(f"\nPower law fit: Runtime = {popt_power[0]:.2e} * N^{popt_power[1]:.3f}")
print(f"Linear fit: Runtime = {popt_linear[0]:.2e} * N + {popt_linear[1]:.3f}")

# R-squared for power law
y_pred_power = power_law(x, *popt_power)
ss_res_power = np.sum((y - y_pred_power) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_power = 1 - ss_res_power / ss_tot

y_pred_linear = linear_law(x, *popt_linear)
ss_res_linear = np.sum((y - y_pred_linear) ** 2)
r2_linear = 1 - ss_res_linear / ss_tot

print(f"R^2 (power law): {r2_power:.4f}")
print(f"R^2 (linear): {r2_linear:.4f}")

scaling_results = pd.DataFrame({
    'model': ['power_law', 'linear'],
    'param_a': [popt_power[0], popt_linear[0]],
    'param_b': [popt_power[1], popt_linear[1]],
    'r_squared': [r2_power, r2_linear]
})
scaling_results.to_csv('outputs/scaling_fit_results.csv', index=False)

# ============================================================
# 6. GENERATE FIGURES
# ============================================================
print("\n" + "=" * 60)
print("6. GENERATING FIGURES")
print("=" * 60)

# ---- Figure 1: Vaccine Composition Bar Chart ----
fig, ax = plt.subplots(figsize=(8, 5))
vaccine_counts = vaccine_simple.sort_values('counts', ascending=False)
bars = ax.bar(vaccine_counts['peptide'], vaccine_counts['counts'], color='steelblue', edgecolor='black')
ax.set_xlabel('Neoantigen (Mutation)', fontsize=12)
ax.set_ylabel('Selection Frequency (out of 10 reps)', fontsize=12)
ax.set_title('Optimal Personalized Neoantigen Vaccine Composition\n(MinSum Objective, Budget = 10)', fontsize=13)
ax.set_ylim(0, 12)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('report/images/figure1_vaccine_composition.png')
plt.close()
print("Saved figure1_vaccine_composition.png")

# ---- Figure 2: Per-Cell Response Probability Distribution ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Histogram
axes[0].hist(final_resp['p_response'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(final_resp['p_response'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {final_resp["p_response"].mean():.3f}')
axes[0].axvline(final_resp['p_response'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {final_resp["p_response"].median():.3f}')
axes[0].set_xlabel('Per-Cell Immune Response Probability', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Per-Cell Response Probabilities\n(Aggregated over 10 Repetitions)', fontsize=12)
axes[0].legend()

# Right: Box plot by repetition
rep_data = []
rep_labels = []
for rep in range(10):
    rep_vals = final_resp[final_resp['population'].str.contains(f', {rep}')]['p_response'].values
    rep_data.append(rep_vals)
    rep_labels.append(f'Rep {rep}')

bp = axes[1].boxplot(rep_data, labels=rep_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1].set_xlabel('Simulation Repetition', fontsize=12)
axes[1].set_ylabel('Per-Cell Immune Response Probability', fontsize=12)
axes[1].set_title('Per-Cell Response Probability by Repetition', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('report/images/figure2_response_probability_distribution.png')
plt.close()
print("Saved figure2_response_probability_distribution.png")

# ---- Figure 3: Coverage Ratio Curve ----
fig, ax = plt.subplots(figsize=(8, 6))
mean_coverage = coverage_curve_df.groupby('threshold')['coverage'].mean()
std_coverage = coverage_curve_df.groupby('threshold')['coverage'].std()

ax.plot(mean_coverage.index, mean_coverage.values, color='darkgreen', linewidth=2.5, label='Mean Coverage')
ax.fill_between(mean_coverage.index,
                mean_coverage.values - std_coverage.values,
                mean_coverage.values + std_coverage.values,
                color='green', alpha=0.2, label='±1 SD')
ax.set_xlabel('Response Probability Threshold', fontsize=12)
ax.set_ylabel('Tumor Cell Coverage Ratio', fontsize=12)
ax.set_title('Vaccine Coverage Ratio vs. Response Probability Threshold\n(Fraction of Tumor Cells with p_response ≥ threshold)', fontsize=12)
ax.legend(loc='upper right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Annotate key points
for thresh in [0.5, 0.9, 0.95]:
    idx = np.argmin(np.abs(mean_coverage.index - thresh))
    cov = mean_coverage.iloc[idx]
    ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'p={thresh:.2f}\ncov={cov:.3f}',
                xy=(thresh, cov), xytext=(thresh+0.05, cov-0.1),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('report/images/figure3_coverage_curve.png')
plt.close()
print("Saved figure3_coverage_curve.png")

# ---- Figure 4: IoU Heatmap ----
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(iou_matrix, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels([f'Rep {i}' for i in range(10)], rotation=45, ha='right')
ax.set_yticklabels([f'Rep {i}' for i in range(10)])
ax.set_title('IoU of Optimal Vaccine Compositions Across Repetitions\n(Identical Selections: IoU = 1.0)', fontsize=12)

# Add text annotations
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, f'{iou_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black" if iou_matrix[i, j] > 0.5 else "white",
                       fontsize=8)

fig.colorbar(im, ax=ax, label='Intersection over Union (IoU)')
plt.tight_layout()
plt.savefig('report/images/figure4_iou_heatmap.png')
plt.close()
print("Saved figure4_iou_heatmap.png")

# ---- Figure 5: Optimization Runtime Scaling ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Runtime by patient
for sample in runtime['SampleID'].unique():
    sample_data = runtime[runtime['SampleID'] == sample]
    axes[0].plot(sample_data['PopulationSize'], sample_data['RunTime'], marker='o', label=f'Patient {sample}')

axes[0].set_xlabel('Cell Population Size', fontsize=12)
axes[0].set_ylabel('Optimization Runtime (seconds)', fontsize=12)
axes[0].set_title('Optimization Runtime vs. Population Size\n(Per Patient)', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Right: Mean runtime with fit
mean_runtime = runtime.groupby('PopulationSize')['RunTime'].agg(['mean', 'std']).reset_index()
axes[1].errorbar(mean_runtime['PopulationSize'], mean_runtime['mean'], yerr=mean_runtime['std'],
                 fmt='o-', color='darkblue', capsize=5, linewidth=2, markersize=8, label='Observed Mean ± SD')

# Plot fits
x_fit = np.logspace(1.9, 4.1, 100)
axes[1].plot(x_fit, power_law(x_fit, *popt_power), '--', color='red', linewidth=2,
             label=f'Power Law Fit: O(N^{popt_power[1]:.2f}), R²={r2_power:.3f}')
axes[1].plot(x_fit, linear_law(x_fit, *popt_linear), ':', color='green', linewidth=2,
             label=f'Linear Fit, R²={r2_linear:.3f}')

axes[1].set_xlabel('Cell Population Size', fontsize=12)
axes[1].set_ylabel('Optimization Runtime (seconds)', fontsize=12)
axes[1].set_title('Mean Optimization Runtime with Scaling Fits', fontsize=12)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure5_runtime_scaling.png')
plt.close()
print("Saved figure5_runtime_scaling.png")

# ---- Figure 6: Mutation Coverage & Peptide Presentation ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Coverage ratio by repetition
reps = coverage_df.index
coverage_vals = coverage_df['coverage_ratio'].values
bars = axes[0].bar([f'Rep {r}' for r in reps], coverage_vals, color='seagreen', edgecolor='black')
axes[0].axhline(coverage_vals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {coverage_vals.mean():.4f}')
axes[0].set_xlabel('Simulation Repetition', fontsize=12)
axes[0].set_ylabel('Tumor Cell Coverage Ratio', fontsize=12)
axes[0].set_title('Fraction of Tumor Cells Covered by\nOptimal Vaccine (Budget = 10)', fontsize=12)
axes[0].set_ylim(0, 1.05)
axes[0].legend()
for bar in bars:
    height = bar.get_height()
    axes[0].annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

# Right: Peptides per cell distribution
peptides_per_cell = cell_pop.groupby(['repetition', 'cell_ids']).size().reset_index(name='num_peptides')
axes[1].hist(peptides_per_cell['num_peptides'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(peptides_per_cell['num_peptides'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {peptides_per_cell["num_peptides"].mean():.1f}')
axes[1].axvline(peptides_per_cell['num_peptides'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median = {peptides_per_cell["num_peptides"].median():.1f}')
axes[1].set_xlabel('Number of Presented Peptides per Cell', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Peptide Presentation\nper Tumor Cell', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('report/images/figure6_coverage_and_presentation.png')
plt.close()
print("Saved figure6_coverage_and_presentation.png")

# ---- Figure 7: Selected vs Non-Selected Element Response Comparison ----
fig, ax = plt.subplots(figsize=(8, 6))
selected_scores = scores_all[scores_all['is_selected']]['p_response']
nonselected_scores = scores_all[~scores_all['is_selected']]['p_response']

ax.hist(selected_scores, bins=50, alpha=0.6, color='green', label=f'Selected (n={len(selected_scores)})', density=True)
ax.hist(nonselected_scores, bins=50, alpha=0.6, color='gray', label=f'Non-Selected (n={len(nonselected_scores)})', density=True)
ax.axvline(selected_scores.mean(), color='darkgreen', linestyle='--', linewidth=2,
           label=f'Selected Mean = {selected_scores.mean():.4f}')
ax.axvline(nonselected_scores.mean(), color='black', linestyle='--', linewidth=2,
           label=f'Non-Selected Mean = {nonselected_scores.mean():.4f}')
ax.set_xlabel('Per-Cell Response Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Response Probability Distribution:\nSelected vs. Non-Selected Vaccine Elements', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('report/images/figure7_selected_vs_nonselected.png')
plt.close()
print("Saved figure7_selected_vs_nonselected.png")

# ---- Figure 8: Vaccine Element Efficacy Heatmap ----
fig, ax = plt.subplots(figsize=(10, 6))
pivot_scores = scores_all.groupby(['vaccine_element', 'cell_id'])['p_response'].mean().reset_index()
pivot_matrix = pivot_scores.pivot(index='vaccine_element', columns='cell_id', values='p_response')

# Sort by mean response
mean_resp = pivot_matrix.mean(axis=1).sort_values(ascending=False)
pivot_matrix = pivot_matrix.loc[mean_resp.index]

# Only show first 50 cells for readability
sns.heatmap(pivot_matrix.iloc[:, :50], cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'p_response'})
ax.set_title('Per-Cell Response Probabilities by Vaccine Element\n(First 50 Cells, Sorted by Mean Response)', fontsize=12)
ax.set_xlabel('Cell ID', fontsize=11)
ax.set_ylabel('Vaccine Element (Mutation)', fontsize=11)
plt.tight_layout()
plt.savefig('report/images/figure8_element_heatmap.png')
plt.close()
print("Saved figure8_element_heatmap.png")

# ============================================================
# 7. SUMMARY STATISTICS OUTPUT
# ============================================================
print("\n" + "=" * 60)
print("7. SUMMARY STATISTICS")
print("=" * 60)

summary = {
    'total_simulated_cells': int(cell_pop.groupby('repetition')['cell_ids'].nunique().sum()),
    'unique_mutations_in_population': int(len(all_mutations)),
    'selected_vaccine_elements': int(len(selected_unique)),
    'mean_per_cell_response_probability': float(final_resp['p_response'].mean()),
    'median_per_cell_response_probability': float(final_resp['p_response'].median()),
    'std_per_cell_response_probability': float(final_resp['p_response'].std()),
    'mean_coverage_ratio': float(coverage_df['coverage_ratio'].mean()),
    'std_coverage_ratio': float(coverage_df['coverage_ratio'].std()),
    'mean_peptides_per_cell': float(peptides_per_cell['num_peptides'].mean()),
    'median_peptides_per_cell': float(peptides_per_cell['num_peptides'].median()),
    'mean_mutations_per_cell': float(cell_pop.groupby(['repetition', 'cell_ids'])['mutation'].nunique().mean()),
    'min_runtime_seconds': float(runtime['RunTime'].min()),
    'max_runtime_seconds': float(runtime['RunTime'].max()),
    'mean_runtime_100_cells': float(runtime[runtime['PopulationSize']==100]['RunTime'].mean()),
    'mean_runtime_10000_cells': float(runtime[runtime['PopulationSize']==10000]['RunTime'].mean()),
    'runtime_scaling_exponent': float(popt_power[1]),
    'runtime_scaling_r2': float(r2_power),
    'iou_mean_across_repetitions': float(np.mean(iou_matrix[~np.eye(10, dtype=bool)])),
    'selected_element_mean_p_response': float(selected_scores.mean()),
    'nonselected_element_mean_p_response': float(nonselected_scores.mean()),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('outputs/summary_statistics.csv', index=False)
print("Summary statistics saved to outputs/summary_statistics.csv")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
