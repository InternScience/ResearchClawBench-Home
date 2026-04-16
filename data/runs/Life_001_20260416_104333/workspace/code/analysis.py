import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Figure 1: Distribution of mutations per cell
df_cells = pd.read_csv('data/cell-populations.csv')
muts_per_cell = df_cells.groupby(['repetition', 'cell_ids'])['mutation'].nunique().reset_index()

plt.figure(figsize=(8, 6))
sns.histplot(muts_per_cell['mutation'], bins=10, kde=False)
plt.title('Distribution of Number of Mutations per Cell')
plt.xlabel('Number of Unique Mutations')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('report/images/fig1_mutations_per_cell.png')
plt.close()

# Figure 2: Vaccine composition weights
df_vaccine = pd.read_csv('data/vaccine.budget-10.minsum.adaptive.csv')
plt.figure(figsize=(8, 6))
sns.barplot(data=df_vaccine, x='peptide', y='counts', color='steelblue')
plt.title('Vaccine Composition (Budget=10)')
plt.xlabel('Selected Mutation')
plt.ylabel('Selection Count across Repetitions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/images/fig2_vaccine_composition.png')
plt.close()

# Figure 3: Vaccine efficacy (per-cell immune response probability)
df_resp = pd.read_csv('data/final-response-likelihoods.csv')
plt.figure(figsize=(8, 6))
sns.histplot(df_resp['p_response'], bins=20, kde=True)
plt.title('Distribution of Per-Cell Immune Response Probability')
plt.xlabel('Probability of Response')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('report/images/fig3_response_probability.png')
plt.close()

# Figure 4: Coverage ratio of tumor cells
thresholds = np.linspace(0, 1, 100)
coverage = [(df_resp['p_response'] >= t).mean() for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, coverage, lw=2)
plt.title('Coverage Ratio of Tumor Cells vs. Response Threshold')
plt.xlabel('Minimum Probability of Response (Threshold)')
plt.ylabel('Fraction of Cells Covered')
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/fig4_coverage_ratio.png')
plt.close()

# Figure 5: IoU of optimal vaccine compositions across repetitions
df_sel = pd.read_csv('data/selected-vaccine-elements.budget-10.minsum.adaptive.csv')
reps = df_sel['repetition'].unique()
n_reps = len(reps)
iou_matrix = np.zeros((n_reps, n_reps))

for i, rep1 in enumerate(reps):
    set1 = set(df_sel[df_sel['repetition'] == rep1]['peptide'])
    for j, rep2 in enumerate(reps):
        set2 = set(df_sel[df_sel['repetition'] == rep2]['peptide'])
        iou = len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
        iou_matrix[i, j] = iou

plt.figure(figsize=(8, 6))
sns.heatmap(iou_matrix, annot=True, cmap='Blues', xticklabels=reps, yticklabels=reps)
plt.title('IoU of Selected Vaccine Compositions Across Repetitions')
plt.xlabel('Repetition')
plt.ylabel('Repetition')
plt.tight_layout()
plt.savefig('report/images/fig5_iou_heatmap.png')
plt.close()

# Figure 6: Optimization runtime vs population size
df_runtime = pd.read_csv('data/optimization_runtime_data.csv')
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_runtime, x='PopulationSize', y='RunTime', hue='SampleID', marker='o')
plt.title('Optimization Runtime vs. Population Size')
plt.xlabel('Population Size')
plt.ylabel('Run Time (seconds)')
plt.legend(title='Sample ID')
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/fig6_runtime.png')
plt.close()
