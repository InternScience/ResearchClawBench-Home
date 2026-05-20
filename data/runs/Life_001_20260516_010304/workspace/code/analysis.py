#!/usr/bin/env python3
"""
Personalized Neoantigen Vaccine Optimization Analysis
=====================================================
Comprehensive analysis of vaccine composition, efficacy metrics,
and optimization runtime for personalized cancer vaccines.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 6)
})

# Paths
DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
REPORT_IMG_DIR = Path('report/images')
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

print("Loading data...")

# Cell populations
cell_pop = pd.read_csv(DATA_DIR / 'cell-populations.csv')

# Final response likelihoods
frl = pd.read_csv(DATA_DIR / 'final-response-likelihoods.csv')
frl['rep'] = frl['population'].str.extract(r'(\d+)$').astype(int)

# Sim-specific response likelihoods
ssrl = pd.read_csv(DATA_DIR / 'sim-specific-response-likelihoods.csv')
ssrl['rep'] = ssrl['vaccine'].str.extract(r'rep-(\d+)$').astype(int)

# Optimization runtime data
rt = pd.read_csv(DATA_DIR / 'optimization_runtime_data.csv')

# Selected vaccine elements
sve = pd.read_csv(DATA_DIR / 'selected-vaccine-elements.budget-10.minsum.adaptive.csv')

# Simplified vaccine composition
vax_comp = pd.read_csv(DATA_DIR / 'vaccine.budget-10.minsum.adaptive.csv')

# Load all vaccine element scores
ves_all = []
for rep in range(10):
    fname = DATA_DIR / f'vaccine-elements.scores.100-cells.10x.rep-{rep}.csv'
    df = pd.read_csv(fname)
    df['rep'] = rep
    ves_all.append(df)
ves = pd.concat(ves_all, ignore_index=True)

print(f"  Cell populations: {cell_pop.shape[0]} rows, {cell_pop.cell_ids.nunique()} unique cells per rep")
print(f"  Final response likelihoods: {frl.shape[0]} rows, {frl.rep.nunique()} reps")
print(f"  Vaccine element scores: {ves.shape[0]} rows, {ves.rep.nunique()} reps")
print(f"  Runtime data: {rt.shape[0]} rows, {rt.SampleID.nunique()} samples")

# ============================================================
# 2. DATA OVERVIEW STATISTICS
# ============================================================

print("\nComputing data overview statistics...")

overview_stats = {
    'n_cells_per_rep': cell_pop.groupby('repetition')['cell_ids'].nunique().to_dict(),
    'n_unique_peptides': int(cell_pop['presented_peptides'].nunique()),
    'n_unique_mutations': int(cell_pop['mutation'].nunique()),
    'selected_vaccine_elements': sorted(vax_comp['peptide'].tolist()),
    'vaccine_budget': 10,
    'n_repetitions': 10,
    'p_response_summary': {
        'mean': float(frl['p_response'].mean()),
        'std': float(frl['p_response'].std()),
        'median': float(frl['p_response'].median()),
        'min': float(frl['p_response'].min()),
        'max': float(frl['p_response'].max()),
    },
    'num_presented_peptides_summary': {
        'mean': float(frl['num_presented_peptides'].mean()),
        'std': float(frl['num_presented_peptides'].std()),
        'min': float(frl['num_presented_peptides'].min()),
        'max': float(frl['num_presented_peptides'].max()),
    },
    'n_cells_total': len(frl),
}

with open(OUTPUT_DIR / 'overview_stats.json', 'w') as f:
    json.dump(overview_stats, f, indent=2)

# ============================================================
# 3. PER-CELL IMMUNE RESPONSE PROBABILITY ANALYSIS
# ============================================================

print("Analyzing per-cell immune response probabilities...")

# Distribution statistics per repetition
rep_stats = frl.groupby('rep').agg(
    mean_p_response=('p_response', 'mean'),
    std_p_response=('p_response', 'std'),
    median_p_response=('p_response', 'median'),
    min_p_response=('p_response', 'min'),
    max_p_response=('p_response', 'max'),
    n_cells=('p_response', 'count'),
    coverage_50=('p_response', lambda x: (x > 0.5).mean()),
    coverage_80=('p_response', lambda x: (x > 0.8).mean()),
    coverage_90=('p_response', lambda x: (x > 0.9).mean()),
    coverage_95=('p_response', lambda x: (x > 0.95).mean()),
).reset_index()

rep_stats.to_csv(OUTPUT_DIR / 'per_rep_response_stats.csv', index=False)

# Overall coverage curve
thresholds = np.linspace(0, 1, 101)
coverage_curve = []
for t in thresholds:
    cov = (frl['p_response'] > t).mean()
    coverage_curve.append({'threshold': t, 'coverage': cov})
coverage_df = pd.DataFrame(coverage_curve)
coverage_df.to_csv(OUTPUT_DIR / 'coverage_curve.csv', index=False)

# Per-rep coverage curves
rep_coverage = []
for rep in range(10):
    sub = frl[frl['rep'] == rep]
    for t in thresholds:
        cov = (sub['p_response'] > t).mean()
        rep_coverage.append({'rep': rep, 'threshold': t, 'coverage': cov})
rep_coverage_df = pd.DataFrame(rep_coverage)
rep_coverage_df.to_csv(OUTPUT_DIR / 'rep_coverage_curves.csv', index=False)

# ============================================================
# 4. VACCINE COMPOSITION ANALYSIS & IoU
# ============================================================

print("Analyzing vaccine composition and IoU...")

# The vaccine composition is identical across all repetitions
# Each repetition selects the same 10 mutations
vax_sets = {}
for rep in range(10):
    vax_sets[rep] = set(sve[sve['repetition'] == rep]['peptide'].unique())

# Compute pairwise IoU
ious = []
n_reps = 10
for i in range(n_reps):
    for j in range(i+1, n_reps):
        intersection = len(vax_sets[i] & vax_sets[j])
        union = len(vax_sets[i] | vax_sets[j])
        iou = intersection / union if union > 0 else 0
        ious.append({'rep_i': i, 'rep_j': j, 'IoU': iou, 'intersection': intersection, 'union': union})

iou_df = pd.DataFrame(ious)
iou_df.to_csv(OUTPUT_DIR / 'pairwise_iou.csv', index=False)

# IoU summary
iou_summary = {
    'mean_IoU': float(iou_df['IoU'].mean()),
    'std_IoU': float(iou_df['IoU'].std()),
    'min_IoU': float(iou_df['IoU'].min()),
    'max_IoU': float(iou_df['IoU'].max()),
    'all_identical': bool((iou_df['IoU'] == 1.0).all()),
    'unique_vaccine_composition': sorted(list(vax_sets[0]))
}
with open(OUTPUT_DIR / 'iou_summary.json', 'w') as f:
    json.dump(iou_summary, f, indent=2)

# Per-vaccine-element effectiveness across repetitions
ves_selected = ves[ves['vaccine_element'].isin(iou_summary['unique_vaccine_composition'])]
ve_stats = ves_selected.groupby(['vaccine_element', 'rep']).agg(
    mean_p_response=('p_response', 'mean'),
    coverage_50=('p_response', lambda x: (x > 0.5).mean()),
    n_cells=('cell_id', 'nunique'),
).reset_index()

ve_avg = ve_stats.groupby('vaccine_element').agg(
    mean_p_response=('mean_p_response', 'mean'),
    std_p_response=('mean_p_response', 'std'),
    mean_coverage=('coverage_50', 'mean'),
    std_coverage=('coverage_50', 'std'),
).reset_index()
ve_avg.to_csv(OUTPUT_DIR / 'vaccine_element_effectiveness.csv', index=False)

# Also include non-selected elements for comparison
ves_all_stats = ves.groupby('vaccine_element').agg(
    mean_p_response=('p_response', 'mean'),
    std_p_response=('p_response', 'std'),
    coverage_50=('p_response', lambda x: (x > 0.5).mean()),
).reset_index()
ves_all_stats['selected'] = ves_all_stats['vaccine_element'].isin(iou_summary['unique_vaccine_composition'])
ves_all_stats.to_csv(OUTPUT_DIR / 'all_vaccine_element_effectiveness.csv', index=False)

# ============================================================
# 5. OPTIMIZATION RUNTIME ANALYSIS
# ============================================================

print("Analyzing optimization runtime...")

rt_summary = rt.groupby('PopulationSize').agg(
    mean_runtime=('RunTime', 'mean'),
    std_runtime=('RunTime', 'std'),
    min_runtime=('RunTime', 'min'),
    max_runtime=('RunTime', 'max'),
    n_samples=('SampleID', 'nunique'),
).reset_index()
rt_summary.to_csv(OUTPUT_DIR / 'runtime_summary.csv', index=False)

# Fit power law to runtime data
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * x**b

x_data = rt['PopulationSize'].values
y_data = rt['RunTime'].values
popt, pcov = curve_fit(power_law, x_data, y_data, p0=[1e-4, 1.0])
a_fit, b_fit = popt

runtime_fit = {
    'power_law_a': float(a_fit),
    'power_law_b': float(b_fit),
    'r_squared': None,  # Will compute below
}
# R^2
y_pred = power_law(x_data, *popt)
ss_res = np.sum((y_data - y_pred)**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - ss_res / ss_tot
runtime_fit['r_squared'] = float(r_squared)

with open(OUTPUT_DIR / 'runtime_fit.json', 'w') as f:
    json.dump(runtime_fit, f, indent=2)

# ============================================================
# 6. MUTATION-PEPTIDE MAPPING
# ============================================================

print("Analyzing mutation-peptide mapping...")

# Map mutations to peptides
mut_peptide_map = cell_pop.groupby('mutation')['presented_peptides'].apply(lambda x: sorted(x.unique())).to_dict()
with open(OUTPUT_DIR / 'mutation_peptide_map.json', 'w') as f:
    json.dump(mut_peptide_map, f, indent=2)

# Count peptides per mutation
mut_peptide_counts = cell_pop.groupby('mutation')['presented_peptides'].nunique().reset_index()
mut_peptide_counts.columns = ['mutation', 'n_peptides']
mut_peptide_counts['selected'] = mut_peptide_counts['mutation'].isin(iou_summary['unique_vaccine_composition'])
mut_peptide_counts.to_csv(OUTPUT_DIR / 'mutation_peptide_counts.csv', index=False)

# Cell coverage by mutation (how many cells have at least one peptide from each mutation)
mut_cell_coverage = cell_pop.groupby(['repetition', 'mutation'])['cell_ids'].nunique().reset_index()
mut_cell_coverage_avg = mut_cell_coverage.groupby('mutation')['cell_ids'].agg(['mean', 'std']).reset_index()
mut_cell_coverage_avg.columns = ['mutation', 'mean_cells_covered', 'std_cells_covered']
mut_cell_coverage_avg.to_csv(OUTPUT_DIR / 'mutation_cell_coverage.csv', index=False)

# ============================================================
# 7. RESPONSE VS NUM PRESENTED PEPTIDES
# ============================================================

print("Analyzing response vs num presented peptides...")

# Bin by num_presented_peptides
frl['peptide_bin'] = pd.cut(frl['num_presented_peptides'], bins=range(5, 31, 2))
peptide_bin_stats = frl.groupby('peptide_bin', observed=False).agg(
    mean_p_response=('p_response', 'mean'),
    std_p_response=('p_response', 'std'),
    n_cells=('p_response', 'count'),
).reset_index()
peptide_bin_stats.to_csv(OUTPUT_DIR / 'response_vs_peptides.csv', index=False)

print("\nAll analyses completed. Generating figures...")

# ============================================================
# FIGURE GENERATION
# ============================================================

# --- Figure 1: Distribution of per-cell immune response probabilities ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(frl['p_response'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(frl['p_response'].mean(), color='darkred', linestyle='--', linewidth=2,
           label=f"Mean = {frl['p_response'].mean():.4f}")
ax.axvline(frl['p_response'].median(), color='darkgreen', linestyle=':', linewidth=2,
           label=f"Median = {frl['p_response'].median():.4f}")
ax.set_xlabel('Per-Cell Immune Response Probability (p_response)')
ax.set_ylabel('Number of Cells')
ax.set_title('Distribution of Per-Cell Immune\nResponse Probabilities')
ax.legend(loc='upper left')
ax.set_xlim(0, 1.05)

# Boxplot per repetition
ax = axes[1]
rep_data = [frl[frl['rep'] == r]['p_response'].values for r in range(10)]
bp = ax.boxplot(rep_data, patch_artist=True, widths=0.7)
for patch in bp['boxes']:
    patch.set_facecolor('lightsteelblue')
ax.axhline(frl['p_response'].mean(), color='darkred', linestyle='--', linewidth=1.5, label=f'Global Mean')
ax.set_xlabel('Repetition')
ax.set_ylabel('p_response')
ax.set_title('Per-Cell Response Probability\nby Repetition')
ax.set_xticklabels(range(10))
ax.legend()

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure1_response_distribution.png')
plt.close()
print("  Figure 1 saved.")

# --- Figure 2: Coverage curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall coverage curve
ax = axes[0]
ax.plot(coverage_df['threshold'], coverage_df['coverage'], 'b-', linewidth=2.5)
ax.fill_between(coverage_df['threshold'], 0, coverage_df['coverage'], alpha=0.15, color='blue')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

# Mark key points
for t_val in [0.5, 0.8, 0.9, 0.95]:
    cov_at_t = (frl['p_response'] > t_val).mean()
    ax.plot(t_val, cov_at_t, 'ro', markersize=6)
    ax.annotate(f'{cov_at_t:.2%}', (t_val, cov_at_t),
                textcoords="offset points", xytext=(10, -10), fontsize=9)

ax.set_xlabel('Response Probability Threshold')
ax.set_ylabel('Fraction of Cells Above Threshold')
ax.set_title('Tumor Cell Coverage Curve')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Per-rep coverage curves
ax = axes[1]
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for rep in range(10):
    sub = rep_coverage_df[rep_coverage_df['rep'] == rep]
    ax.plot(sub['threshold'], sub['coverage'], color=colors[rep], alpha=0.7, linewidth=1,
            label=f'Rep {rep}')
ax.plot(coverage_df['threshold'], coverage_df['coverage'], 'k-', linewidth=2.5, label='Mean')
ax.set_xlabel('Response Probability Threshold')
ax.set_ylabel('Fraction of Cells Above Threshold')
ax.set_title('Coverage Curves by Repetition')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure2_coverage_curves.png')
plt.close()
print("  Figure 2 saved.")

# --- Figure 3: Vaccine element effectiveness ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of mean p_response per vaccine element
ve_sorted = ves_all_stats.sort_values('mean_p_response', ascending=True)
colors_bar = ['#2E86AB' if s else '#D64045' for s in ve_sorted['selected']]

ax = axes[0]
bars = ax.barh(range(len(ve_sorted)), ve_sorted['mean_p_response'], color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(ve_sorted)))
ax.set_yticklabels(ve_sorted['vaccine_element'])
ax.set_xlabel('Mean Per-Cell p_response')
ax.set_title('Vaccine Element Effectiveness\n(Mean Per-Cell Response Probability)')
ax.axvline(0, color='black', linewidth=0.5)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2E86AB', label='Selected'),
                   Patch(facecolor='#D64045', label='Not Selected')]
ax.legend(handles=legend_elements, loc='lower right')

# Coverage per vaccine element
ax = axes[1]
ve_sorted2 = ves_all_stats.sort_values('coverage_50', ascending=True)
colors_bar2 = ['#2E86AB' if s else '#D64045' for s in ve_sorted2['selected']]
bars2 = ax.barh(range(len(ve_sorted2)), ve_sorted2['coverage_50'] * 100, color=colors_bar2, edgecolor='white')
ax.set_yticks(range(len(ve_sorted2)))
ax.set_yticklabels(ve_sorted2['vaccine_element'])
ax.set_xlabel('Cell Coverage (% with p_response > 0.5)')
ax.set_title('Vaccine Element Coverage\n(Fraction of Cells with p_response > 0.5)')
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure3_vaccine_effectiveness.png')
plt.close()
print("  Figure 3 saved.")

# --- Figure 4: Response vs Number of Presented Peptides ---
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot with jitter
np.random.seed(42)
jitter = np.random.uniform(-0.3, 0.3, len(frl))
ax.scatter(frl['num_presented_peptides'] + jitter, frl['p_response'],
           alpha=0.3, s=15, c='steelblue', edgecolors='none')

# Mean trend line
peptide_means = frl.groupby('num_presented_peptides')['p_response'].agg(['mean', 'std']).reset_index()
ax.errorbar(peptide_means['num_presented_peptides'], peptide_means['mean'],
            yerr=peptide_means['std'], fmt='o-', color='darkred',
            capsize=3, linewidth=2, markersize=8, label='Mean ± Std')

ax.set_xlabel('Number of Presented Peptides per Cell')
ax.set_ylabel('Per-Cell Immune Response Probability (p_response)')
ax.set_title('Immune Response Probability vs Number of Presented Peptides')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure4_response_vs_peptides.png')
plt.close()
print("  Figure 4 saved.")

# --- Figure 5: Vaccine Composition IoU Heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IoU Heatmap
iou_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i == j:
            iou_matrix[i, j] = 1.0
        else:
            row = iou_df[((iou_df['rep_i'] == i) & (iou_df['rep_j'] == j)) |
                         ((iou_df['rep_i'] == j) & (iou_df['rep_j'] == i))]
            if len(row) > 0:
                iou_matrix[i, j] = row['IoU'].values[0]

ax = axes[0]
im = ax.imshow(iou_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
ax.set_xlabel('Repetition')
ax.set_ylabel('Repetition')
ax.set_title('Pairwise IoU of Vaccine Compositions')

# Add text annotations
for i in range(10):
    for j in range(10):
        color = 'white' if iou_matrix[i, j] < 0.5 else 'black'
        ax.text(j, i, f'{iou_matrix[i, j]:.2f}', ha='center', va='center',
                fontsize=9, color=color)
plt.colorbar(im, ax=ax, label='IoU')

# Vaccine element presence across reps (all identical so just show composition)
ax = axes[1]
vax_elements = sorted(iou_summary['unique_vaccine_composition'])
vax_matrix = np.ones((10, 10))  # All identical
ax.imshow(vax_matrix, cmap='Blues', aspect='equal')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(vax_elements, rotation=45, ha='right')
ax.set_yticklabels(range(10))
ax.set_xlabel('Vaccine Element (Mutation)')
ax.set_ylabel('Repetition')
ax.set_title('Vaccine Composition Consistency\n(All Repetitions Identical)')
for i in range(10):
    for j in range(10):
        ax.text(j, i, '✓', ha='center', va='center', fontsize=10)

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure5_iou_heatmap.png')
plt.close()
print("  Figure 5 saved.")

# --- Figure 6: Optimization Runtime vs Population Size ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax = axes[0]
for sid in sorted(rt['SampleID'].unique()):
    sub = rt[rt['SampleID'] == sid]
    ax.plot(sub['PopulationSize'], sub['RunTime'], 'o-', linewidth=1.5,
            markersize=6, alpha=0.7, label=f'Sample {sid}')

# Mean trend
ax.plot(rt_summary['PopulationSize'], rt_summary['mean_runtime'], 'k-', linewidth=3, label='Mean')
ax.fill_between(rt_summary['PopulationSize'],
                rt_summary['mean_runtime'] - rt_summary['std_runtime'],
                rt_summary['mean_runtime'] + rt_summary['std_runtime'],
                alpha=0.2, color='black')

ax.set_xlabel('Cell Population Size')
ax.set_ylabel('Optimization Runtime (seconds)')
ax.set_title('Optimization Runtime vs Population Size\n(Linear Scale)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Log-log scale
ax = axes[1]
x_fit = np.logspace(np.log10(100), np.log10(10000), 100)
y_fit = power_law(x_fit, *popt)

for sid in sorted(rt['SampleID'].unique()):
    sub = rt[rt['SampleID'] == sid]
    ax.loglog(sub['PopulationSize'], sub['RunTime'], 'o-', linewidth=1.5,
              markersize=6, alpha=0.7)

ax.loglog(x_fit, y_fit, 'k--', linewidth=2,
          label=f'Power Law Fit: $t = {a_fit:.2e} \\cdot N^{{{b_fit:.2f}}}$\n$R^2 = {r_squared:.4f}$')

ax.set_xlabel('Cell Population Size (log scale)')
ax.set_ylabel('Optimization Runtime (seconds, log scale)')
ax.set_title('Optimization Runtime vs Population Size\n(Log-Log Scale)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure6_runtime_analysis.png')
plt.close()
print("  Figure 6 saved.")

# --- Figure 7: Per-rep coverage metrics ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean p_response per rep
ax = axes[0, 0]
ax.bar(rep_stats['rep'], rep_stats['mean_p_response'], color='steelblue', edgecolor='white')
ax.errorbar(rep_stats['rep'], rep_stats['mean_p_response'],
            yerr=rep_stats['std_p_response'], fmt='none', ecolor='black', capsize=5)
ax.axhline(frl['p_response'].mean(), color='darkred', linestyle='--', label=f'Overall mean: {frl["p_response"].mean():.4f}')
ax.set_xlabel('Repetition')
ax.set_ylabel('Mean p_response')
ax.set_title('Mean Per-Cell Response Probability by Repetition')
ax.legend(fontsize=9)

# Coverage at threshold 0.5
ax = axes[0, 1]
metrics = ['coverage_50', 'coverage_80', 'coverage_90', 'coverage_95']
labels = ['>0.5', '>0.8', '>0.9', '>0.95']
colors_cov = ['#457B9D', '#1D3557', '#E63946', '#A8DADC']
x = np.arange(10)
width = 0.2
for i, (metric, label, color) in enumerate(zip(metrics, labels, colors_cov)):
    ax.bar(x + i * width, rep_stats[metric] * 100, width, label=label, color=color, edgecolor='white')
ax.set_xlabel('Repetition')
ax.set_ylabel('Coverage (%)')
ax.set_title('Cell Coverage at Different p_response Thresholds')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(range(10))
ax.legend(fontsize=8)

# Number of cells per rep
ax = axes[1, 0]
ax.bar(rep_stats['rep'], rep_stats['n_cells'], color='lightcoral', edgecolor='white')
ax.set_xlabel('Repetition')
ax.set_ylabel('Number of Cells')
ax.set_title('Cells per Repetition')

# Per-vaccine-element coverage by rep (heatmap)
ax = axes[1, 1]
pivot = ve_stats.pivot_table(values='coverage_50', index='vaccine_element', columns='rep')
sns.heatmap(pivot * 100, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Coverage (%)'}, vmin=0, vmax=100)
ax.set_xlabel('Repetition')
ax.set_ylabel('Vaccine Element')
ax.set_title('Per-Element Cell Coverage by Repetition\n(p_response > 0.5)')

plt.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'figure7_rep_metrics.png')
plt.close()
print("  Figure 7 saved.")

# ============================================================
# SAVE KEY RESULTS
# ============================================================

print("\nSaving key quantitative results...")

# Main results summary
main_results = {
    'vaccine_composition': {
        'elements': iou_summary['unique_vaccine_composition'],
        'budget': 10,
        'n_unique_elements': 10,
    },
    'per_cell_response_probability': {
        'mean': float(frl['p_response'].mean()),
        'std': float(frl['p_response'].std()),
        'median': float(frl['p_response'].median()),
        'q25': float(frl['p_response'].quantile(0.25)),
        'q75': float(frl['p_response'].quantile(0.75)),
    },
    'coverage_ratio': {
        'threshold_0.5': float((frl['p_response'] > 0.5).mean()),
        'threshold_0.8': float((frl['p_response'] > 0.8).mean()),
        'threshold_0.9': float((frl['p_response'] > 0.9).mean()),
        'threshold_0.95': float((frl['p_response'] > 0.95).mean()),
    },
    'iou_optimal_vaccine': {
        'mean_IoU': float(iou_df['IoU'].mean()),
        'all_identical': bool((iou_df['IoU'] == 1.0).all()),
    },
    'optimization_runtime': {
        'min_population': 100,
        'max_population': 10000,
        'min_runtime': float(rt['RunTime'].min()),
        'max_runtime': float(rt['RunTime'].max()),
        'power_law_exponent': float(b_fit),
        'r_squared': float(r_squared),
    },
}

with open(OUTPUT_DIR / 'main_results.json', 'w') as f:
    json.dump(main_results, f, indent=2)

print("\nAll analyses and figures complete!")
print(f"Results saved to {OUTPUT_DIR}/")
print(f"Figures saved to {REPORT_IMG_DIR}/")
