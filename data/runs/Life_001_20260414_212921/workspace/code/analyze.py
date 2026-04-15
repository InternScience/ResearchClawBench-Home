import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

data_dir = Path('data')
outputs_dir = Path('outputs')
images_dir = Path('report/images')
outputs_dir.mkdir(exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

# 1. Vaccine compositions
sel_df = pd.read_csv(data_dir / 'selected-vaccine-elements.budget-10.minsum.adaptive.csv')
vaccine_comps = {}
for rep in sel_df.repetition.unique():
    peptides = set(sel_df[sel_df.repetition == rep].peptide.unique())
    vaccine_comps[rep] = list(peptides)
common_vaccine = set.intersection(*[set(p) for p in vaccine_comps.values()])
print('Common vaccine elements:', sorted(common_vaccine))
vaccine_df = pd.DataFrame({
    'repetition': list(vaccine_comps.keys()),
    'selected_peptides': [','.join(sorted(p)) for p in vaccine_comps.values()]
})
vaccine_df.to_csv(outputs_dir / 'vaccine_compositions.csv', index=False)

# IoU matrix
reps = sorted(sel_df.repetition.unique())
iou_matrix = np.zeros((len(reps), len(reps)))
for i, r1 in enumerate(reps):
    for j, r2 in enumerate(reps):
        s1 = set(sel_df[sel_df.repetition == r1].peptide)
        s2 = set(sel_df[sel_df.repetition == r2].peptide)
        inter = len(s1 & s2)
        union = len(s1 | s2)
        iou_matrix[i,j] = inter / union if union > 0 else 1.0
iou_df = pd.DataFrame(iou_matrix, index=reps, columns=reps)
iou_df.to_csv(outputs_dir / 'vaccine_iou.csv')
print('IoU mean:', iou_matrix.mean())

# Plot IoU heatmap
plt.figure(figsize=(8,6))
sns.heatmap(iou_matrix, annot=True, cmap='Blues', xticklabels=reps, yticklabels=reps)
plt.title('IoU of Selected Vaccine Elements Across Repetitions')
plt.xlabel('Repetition')
plt.ylabel('Repetition')
plt.tight_layout()
plt.savefig(images_dir / 'vaccine_iou_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Efficacy metrics from final-response
final_df = pd.read_csv(data_dir / 'final-response-likelihoods.csv')
metrics = []
for pop in final_df.population.unique():
    sub = final_df[final_df.population == pop]
    mean_pr = sub.p_response.mean()
    std_pr = sub.p_response.std()
    cov05 = (sub.p_response > 0.5).mean()
    cov09 = (sub.p_response > 0.9).mean()
    metrics.append({'population': pop, 'mean_p_response': mean_pr, 'std_p_response': std_pr,
                    'coverage_0.5': cov05, 'coverage_0.9': cov09})
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(outputs_dir / 'efficacy_metrics.csv', index=False)
print(metrics_df.describe())

# Response dist plot
plt.figure(figsize=(10,6))
sns.histplot(data=final_df, x='p_response', hue='population', stat='density', common_norm=True)
plt.title('Distribution of Per-Cell Immune Response Probabilities')
plt.xlabel('p_response')
plt.ylabel('Density')
plt.savefig(images_dir / 'response_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Coverage curve
thresh = np.linspace(0,1,101)
cov_curves = []
for pop in final_df.population.unique():
    sub = final_df[final_df.population == pop].p_response
    cov = [(sub > t).mean() for t in thresh]
    cov_curves.append(cov)
cov_df = pd.DataFrame({'threshold': thresh, **{f'pop_{i}': cov for i,cov in enumerate(cov_curves)}})
cov_df.to_csv(outputs_dir / 'coverage_curves.csv', index=False)
plt.figure(figsize=(8,6))
for i, pop in enumerate(final_df.population.unique()):
    plt.plot(thresh, cov_curves[i], label=pop.split(', ')[1])
plt.title('Tumor Cell Coverage vs Response Threshold')
plt.xlabel('Threshold')
plt.ylabel('Coverage Ratio')
plt.legend()
plt.savefig(images_dir / 'coverage_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Runtime
runtime_df = pd.read_csv(data_dir / 'optimization_runtime_data.csv')
plt.figure(figsize=(10,6))
sns.lineplot(data=runtime_df, x='PopulationSize', y='RunTime', hue='SampleID', marker='o')
plt.xscale('log')
plt.title('Optimization Runtime vs Population Size (Figure 6)')
plt.xlabel('Population Size (log scale)')
plt.ylabel('Runtime (s)')
plt.savefig(images_dir / 'runtime_vs_popsize.png', dpi=300, bbox_inches='tight')
plt.close()
agg_runtime = runtime_df.groupby('PopulationSize').RunTime.agg(['mean','std']).reset_index()
agg_runtime.to_csv(outputs_dir / 'runtime_summary.csv', index=False)

# 4. Data overview: cell populations
cellpops = pd.read_csv(data_dir / 'cell-populations.csv')
cellpops['rep'] = cellpops.repetition.astype(int)
num_pep_per_cell = cellpops.groupby(['rep', 'cell_ids']).size().reset_index(name='num_presented_peptides')
num_mut_per_cell = cellpops.groupby(['rep', 'cell_ids']).mutation.nunique().reset_index(name='num_mutations')
overview_df = pd.merge(num_pep_per_cell, num_mut_per_cell, on=['rep','cell_ids'])
overview_df.to_csv(outputs_dir / 'cell_overview.csv', index=False)

fig, axs = plt.subplots(2,2, figsize=(12,10))
sns.histplot(data=num_pep_per_cell, x='num_presented_peptides', hue='rep', ax=axs[0,0], discrete=True)
axs[0,0].set_title('Num Presented Peptides per Cell')
sns.histplot(data=num_mut_per_cell, x='num_mutations', hue='rep', ax=axs[0,1], discrete=True)
axs[0,1].set_title('Num Unique Mutations per Cell')
sns.boxplot(data=overview_df, x='rep', y='num_presented_peptides', ax=axs[1,0])
axs[1,0].set_title('Num Peptides per Rep')
sns.boxplot(data=overview_df, x='rep', y='num_mutations', ax=axs[1,1])
axs[1,1].set_title('Num Mutations per Rep')
plt.tight_layout()
plt.savefig(images_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print('All artifacts generated.')