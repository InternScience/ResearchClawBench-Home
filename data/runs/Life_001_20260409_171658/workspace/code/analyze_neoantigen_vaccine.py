import json
import math
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def parse_population_rep(pop):
    # format like '100-cells.10x, 0'
    parts = str(pop).split(',')
    sim = parts[0].strip()
    rep = int(parts[1].strip()) if len(parts) > 1 else None
    return sim, rep


def safe_iou(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ensure_png(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    final_df = pd.read_csv(DATA / 'final-response-likelihoods.csv')
    sim_df = pd.read_csv(DATA / 'sim-specific-response-likelihoods.csv')
    cell_df = pd.read_csv(DATA / 'cell-populations.csv')
    sel_df = pd.read_csv(DATA / 'selected-vaccine-elements.budget-10.minsum.adaptive.csv')
    runtime_df = pd.read_csv(DATA / 'optimization_runtime_data.csv')
    vaccine_df = pd.read_csv(DATA / 'vaccine.budget-10.minsum.adaptive.csv')

    score_frames = []
    for fp in sorted(DATA.glob('vaccine-elements.scores.100-cells.10x.rep-*.csv')):
        rep = int(fp.stem.split('rep-')[-1])
        df = pd.read_csv(fp)
        df['repetition'] = rep
        score_frames.append(df)
    scores_df = pd.concat(score_frames, ignore_index=True)

    final_df[['simulation_name', 'repetition']] = final_df['population'].apply(lambda x: pd.Series(parse_population_rep(x)))
    sim_df['repetition'] = sim_df['vaccine'].str.extract(r'rep-(\d+)').astype(int)
    sim_df['simulation_name'] = sim_df['population'].str.split(',').str[0].str.strip()

    selected_set = sorted(vaccine_df['peptide'].tolist())
    selected_counts = sel_df.groupby('peptide').size().rename('selection_count').reset_index()

    # composition stability and recall wrt available candidate elements in score files
    per_rep_selection = sel_df.groupby('repetition')['peptide'].apply(list)
    reps = sorted(per_rep_selection.index.tolist())
    iou_records = []
    for i in reps:
        for j in reps:
            iou_records.append({'rep_i': i, 'rep_j': j, 'iou': safe_iou(per_rep_selection[i], per_rep_selection[j])})
    iou_df = pd.DataFrame(iou_records)

    candidate_by_rep = scores_df.groupby('repetition')['vaccine_element'].unique().apply(list)
    coverage_records = []
    for rep, elems in candidate_by_rep.items():
        sel = set(per_rep_selection[rep])
        cand = set(elems)
        coverage_records.append({
            'repetition': rep,
            'selected_candidate_recall': len(sel & cand) / len(sel) if sel else np.nan,
            'candidate_coverage_fraction': len(sel & cand) / len(cand) if cand else np.nan,
            'num_candidates': len(cand),
            'num_selected': len(sel),
            'num_overlap': len(sel & cand),
        })
    coverage_df = pd.DataFrame(coverage_records)

    # empirical per-cell response from chosen vaccine elements using independence approximation
    selected_scores = scores_df[scores_df['vaccine_element'].isin(selected_set)].copy()
    agg = selected_scores.groupby(['repetition', 'cell_id']).agg(
        p_response_empirical=('p_response', lambda x: 1 - float(np.prod(1 - np.clip(np.array(x), 0, 1)))),
        mean_element_response=('p_response', 'mean'),
        max_element_response=('p_response', 'max'),
        selected_elements_present=('vaccine_element', 'nunique')
    ).reset_index()

    final_compare = final_df[['repetition', 'name', 'p_response', 'num_presented_peptides']].rename(columns={'name': 'cell_id', 'p_response': 'p_response_reported'})
    merged = agg.merge(final_compare, on=['repetition', 'cell_id'], how='left')
    merged['abs_error'] = (merged['p_response_empirical'] - merged['p_response_reported']).abs()

    thresholds = [0.5, 0.7, 0.9, 0.95]
    coverage_long = []
    for thr in thresholds:
        tmp = merged.groupby('repetition').apply(lambda g: pd.Series({
            'coverage_ratio': (g['p_response_empirical'] >= thr).mean(),
            'reported_coverage_ratio': (g['p_response_reported'] >= thr).mean()
        }), include_groups=False).reset_index()
        tmp['threshold'] = thr
        coverage_long.append(tmp)
    coverage_long = pd.concat(coverage_long, ignore_index=True)

    rep_summary = merged.groupby('repetition').agg(
        mean_empirical_p_response=('p_response_empirical', 'mean'),
        median_empirical_p_response=('p_response_empirical', 'median'),
        mean_reported_p_response=('p_response_reported', 'mean'),
        median_reported_p_response=('p_response_reported', 'median'),
        mae_vs_reported=('abs_error', 'mean'),
        max_abs_error=('abs_error', 'max'),
        n_cells=('cell_id', 'count')
    ).reset_index()
    rep_summary = rep_summary.merge(coverage_df, on='repetition', how='left')

    mut_cov = cell_df.groupby(['repetition', 'mutation'])['cell_ids'].nunique().reset_index(name='cells_with_mutation')
    total_cells = cell_df.groupby('repetition')['cell_ids'].nunique().rename('total_cells').reset_index()
    mut_cov = mut_cov.merge(total_cells, on='repetition', how='left')
    mut_cov['cell_fraction'] = mut_cov['cells_with_mutation'] / mut_cov['total_cells']
    selected_mut_cov = mut_cov[mut_cov['mutation'].isin(selected_set)].copy()
    mutation_rank = selected_mut_cov.groupby('mutation')['cell_fraction'].agg(['mean', 'std', 'min', 'max']).reset_index().sort_values('mean', ascending=False)

    # runtime analysis
    runtime_summary = runtime_df.groupby('PopulationSize').agg(mean_runtime=('RunTime', 'mean'), median_runtime=('RunTime', 'median')).reset_index()
    coef = np.polyfit(np.log10(runtime_df['PopulationSize']), np.log10(runtime_df['RunTime']), 1)
    scaling_exponent = float(coef[0])

    # outputs tables
    selected_counts.to_csv(OUT / 'selected_peptides_summary.csv', index=False)
    iou_df.to_csv(OUT / 'composition_iou_matrix_long.csv', index=False)
    coverage_df.to_csv(OUT / 'candidate_recall_by_repetition.csv', index=False)
    merged.to_csv(OUT / 'empirical_cell_response_vs_reported.csv', index=False)
    coverage_long.to_csv(OUT / 'coverage_by_threshold.csv', index=False)
    rep_summary.to_csv(OUT / 'repetition_summary_metrics.csv', index=False)
    mutation_rank.to_csv(OUT / 'selected_mutation_cell_fraction_summary.csv', index=False)
    runtime_summary.to_csv(OUT / 'runtime_summary.csv', index=False)

    summary = {
        'selected_vaccine_elements': selected_set,
        'num_selected_elements': len(selected_set),
        'mean_pairwise_iou': float(iou_df.query('rep_i < rep_j')['iou'].mean()),
        'min_pairwise_iou': float(iou_df.query('rep_i < rep_j')['iou'].min()),
        'max_pairwise_iou': float(iou_df.query('rep_i < rep_j')['iou'].max()),
        'mean_empirical_cell_response': float(merged['p_response_empirical'].mean()),
        'median_empirical_cell_response': float(merged['p_response_empirical'].median()),
        'mean_reported_cell_response': float(merged['p_response_reported'].mean()),
        'mean_abs_error_empirical_vs_reported': float(merged['abs_error'].mean()),
        'coverage_thresholds_empirical_mean': {str(t): float(coverage_long[coverage_long['threshold'] == t]['coverage_ratio'].mean()) for t in thresholds},
        'coverage_thresholds_reported_mean': {str(t): float(coverage_long[coverage_long['threshold'] == t]['reported_coverage_ratio'].mean()) for t in thresholds},
        'runtime_scaling_exponent_loglog': scaling_exponent,
    }
    with open(OUT / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # figures
    fig, ax = plt.subplots(figsize=(10, 6))
    order = mutation_rank['mutation'].tolist()
    sns.barplot(data=selected_mut_cov, x='mutation', y='cell_fraction', order=order, estimator=np.mean, errorbar='sd', ax=ax, color='#4C78A8')
    ax.set_xlabel('Selected vaccine element (mutation)')
    ax.set_ylabel('Fraction of cells presenting mutation')
    ax.set_title('Presentation prevalence of selected vaccine elements across repetitions')
    ax.set_ylim(0, 1.0)
    ensure_png(fig, IMG / 'figure1_selected_mutation_coverage.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    mat = iou_df.pivot(index='rep_i', columns='rep_j', values='iou')
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Repetition')
    ax.set_ylabel('Repetition')
    ax.set_title('IoU of optimal vaccine compositions across repetitions')
    ensure_png(fig, IMG / 'figure2_composition_iou_heatmap.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=merged, x='p_response_empirical', hue='repetition', common_norm=False, fill=False, linewidth=1.5, ax=ax)
    ax.set_xlabel('Per-cell immune response probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of empirical per-cell response probabilities')
    ensure_png(fig, IMG / 'figure3_empirical_response_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = coverage_long.groupby('threshold')[['coverage_ratio', 'reported_coverage_ratio']].mean().reset_index()
    ax.plot(plot_df['threshold'], plot_df['coverage_ratio'], marker='o', label='Empirical from element scores')
    ax.plot(plot_df['threshold'], plot_df['reported_coverage_ratio'], marker='s', label='Reported final response file')
    ax.set_xlabel('Response probability threshold')
    ax.set_ylabel('Coverage ratio of tumor cells')
    ax.set_title('Tumor-cell coverage under increasing response thresholds')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    ensure_png(fig, IMG / 'figure4_coverage_curve.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=merged, x='p_response_reported', y='p_response_empirical', hue='repetition', ax=ax, s=50)
    lim = [0, 1]
    ax.plot(lim, lim, '--', color='black', linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Reported final per-cell response probability')
    ax.set_ylabel('Recomputed empirical response probability')
    ax.set_title('Consistency between reported and recomputed cell response')
    ensure_png(fig, IMG / 'figure5_reported_vs_empirical_scatter.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=runtime_df, x='PopulationSize', y='RunTime', hue='SampleID', marker='o', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cell population size (log scale)')
    ax.set_ylabel('Optimization runtime in seconds (log scale)')
    ax.set_title('Runtime scaling of neoantigen vaccine optimization')
    ensure_png(fig, IMG / 'figure6_runtime_scaling.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=rep_summary, y='mae_vs_reported', color='#72B7B2', ax=ax)
    sns.stripplot(data=rep_summary, y='mae_vs_reported', color='black', size=6, ax=ax)
    ax.set_ylabel('Mean absolute error per repetition')
    ax.set_title('Recomputation error against reported final response values')
    ensure_png(fig, IMG / 'figure7_recomputation_error.png')


if __name__ == '__main__':
    main()
