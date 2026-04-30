#!/usr/bin/env python3
"""Reproducible analysis for personalized neoantigen vaccine simulation outputs."""
from __future__ import annotations
import json
import re
from pathlib import Path
from itertools import combinations

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

sns.set_theme(style='whitegrid', context='paper')


def parse_rep_from_population(x: str):
    m = re.search(r',\s*(\d+)\s*$', str(x))
    return int(m.group(1)) if m else np.nan


def load():
    cell = pd.read_csv(DATA / 'cell-populations.csv')
    final = pd.read_csv(DATA / 'final-response-likelihoods.csv')
    sim = pd.read_csv(DATA / 'sim-specific-response-likelihoods.csv')
    runtime = pd.read_csv(DATA / 'optimization_runtime_data.csv')
    selected = pd.read_csv(DATA / 'selected-vaccine-elements.budget-10.minsum.adaptive.csv')
    vacc = pd.read_csv(DATA / 'vaccine.budget-10.minsum.adaptive.csv')
    score_files = sorted(DATA.glob('vaccine-elements.scores.100-cells.10x.rep-*.csv'))
    scores = []
    for p in score_files:
        rep = int(re.search(r'rep-(\d+)\.csv$', p.name).group(1))
        df = pd.read_csv(p)
        df['repetition'] = rep
        scores.append(df)
    scores = pd.concat(scores, ignore_index=True)
    return cell, final, sim, runtime, selected, vacc, scores


def save_data_overview(cell, final, sim, runtime, selected, vacc, scores):
    overview = pd.DataFrame([
        {'dataset':'cell-populations','rows':len(cell),'columns':cell.shape[1],'unique_cells':cell[['repetition','cell_ids']].drop_duplicates().shape[0],'unique_elements':cell['mutation'].nunique()},
        {'dataset':'final-response-likelihoods','rows':len(final),'columns':final.shape[1],'unique_cells':final[['population','name']].drop_duplicates().shape[0],'unique_elements':np.nan},
        {'dataset':'sim-specific-response-likelihoods','rows':len(sim),'columns':sim.shape[1],'unique_cells':sim[['population','name']].drop_duplicates().shape[0],'unique_elements':np.nan},
        {'dataset':'vaccine-element scores','rows':len(scores),'columns':scores.shape[1],'unique_cells':scores[['repetition','cell_id']].drop_duplicates().shape[0],'unique_elements':scores['vaccine_element'].nunique()},
        {'dataset':'selected vaccine elements','rows':len(selected),'columns':selected.shape[1],'unique_cells':np.nan,'unique_elements':selected['peptide'].nunique()},
        {'dataset':'runtime','rows':len(runtime),'columns':runtime.shape[1],'unique_cells':np.nan,'unique_elements':np.nan},
    ])
    overview.to_csv(OUT / 'data_overview.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(data=overview, x='rows', y='dataset', ax=ax, color='#4C72B0')
    ax.set_xscale('log')
    ax.set_xlabel('Rows (log scale)')
    ax.set_ylabel('Dataset')
    ax.set_title('Input data overview')
    for i, row in overview.iterrows():
        ax.text(row['rows']*1.05, i, f"{int(row['rows']):,}", va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / 'data_overview.png', dpi=200)
    plt.close(fig)
    return overview


def composition(selected, vacc):
    selected = selected.copy()
    selected['element'] = selected['peptide']
    per_rep = selected.groupby(['simulation_name','repetition']).agg(
        n_elements=('element','nunique'), total_weight=('weight','sum'), run_time=('run_time','first')
    ).reset_index()
    per_rep.to_csv(OUT / 'selected_vaccine_per_repetition.csv', index=False)
    comp = selected.groupby('element').agg(
        selected_repetitions=('repetition','nunique'), total_weight=('weight','sum'), mean_weight=('weight','mean')
    ).reset_index().sort_values(['selected_repetitions','element'], ascending=[False, True])
    # merge counts from simplified file when available
    vacc2 = vacc.rename(columns={'peptide':'element','counts':'simplified_counts','weight':'simplified_weight'})
    comp = comp.merge(vacc2, on='element', how='outer')
    comp.to_csv(OUT / 'selected_vaccine_composition.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    plot = comp.sort_values('selected_repetitions', ascending=True)
    sns.barplot(data=plot, x='selected_repetitions', y='element', ax=ax, color='#55A868')
    ax.set_xlabel('Number of repetitions selecting element')
    ax.set_ylabel('Vaccine element / mutation')
    ax.set_title('Budget-10 MinSum adaptive selected composition')
    ax.set_xlim(0, max(10, plot['selected_repetitions'].max()))
    fig.tight_layout()
    fig.savefig(IMG / 'vaccine_composition.png', dpi=200)
    plt.close(fig)
    return per_rep, comp


def response_and_coverage(final, sim):
    final = final.copy(); sim = sim.copy()
    for df in (final, sim):
        df['repetition'] = df['population'].map(parse_rep_from_population)
        df['simulation_name'] = df['population'].astype(str).str.replace(r',\s*\d+\s*$', '', regex=True)
    response_summary = final.groupby(['simulation_name','vaccine']).agg(
        n_cells=('p_response','size'), mean_p_response=('p_response','mean'), sd_p_response=('p_response','std'),
        median_p_response=('p_response','median'), q05_p_response=('p_response', lambda x: x.quantile(.05)),
        q95_p_response=('p_response', lambda x: x.quantile(.95)), min_p_response=('p_response','min'), max_p_response=('p_response','max'),
        mean_presented_peptides=('num_presented_peptides','mean')
    ).reset_index()
    rep_summary = final.groupby(['simulation_name','repetition','vaccine']).agg(
        n_cells=('p_response','size'), mean_p_response=('p_response','mean'), sd_p_response=('p_response','std'),
        median_p_response=('p_response','median')
    ).reset_index()
    response_summary.to_csv(OUT / 'response_summary_by_vaccine.csv', index=False)
    rep_summary.to_csv(OUT / 'response_summary_by_repetition.csv', index=False)
    thresholds = np.round(np.linspace(0, 0.99, 100), 2)
    cov_rows = []
    for (simname, rep, vaccine), g in final.groupby(['simulation_name','repetition','vaccine']):
        for t in thresholds:
            cov_rows.append({'simulation_name':simname,'repetition':rep,'vaccine':vaccine,'threshold':t,
                             'coverage_ratio':float((g['p_response']>=t).mean()), 'n_cells':len(g)})
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(OUT / 'coverage_by_threshold.csv', index=False)
    coverage_summary = coverage.groupby(['simulation_name','vaccine','threshold']).agg(
        mean_coverage=('coverage_ratio','mean'), sd_coverage=('coverage_ratio','std'),
        min_coverage=('coverage_ratio','min'), max_coverage=('coverage_ratio','max')
    ).reset_index()
    coverage_summary.to_csv(OUT / 'coverage_summary_by_threshold.csv', index=False)

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(data=final, x='p_response', bins=30, ax=ax, color='#4C72B0')
    ax.axvline(final['p_response'].mean(), color='black', linestyle='--', label=f"mean={final['p_response'].mean():.3f}")
    ax.set_xlabel('Per-cell immune response probability')
    ax.set_ylabel('Cell count')
    ax.set_title('Distribution of vaccine-induced response probabilities')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'response_distributions.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(coverage_summary['threshold'], coverage_summary['mean_coverage'], color='#C44E52', lw=2)
    ax.fill_between(coverage_summary['threshold'].to_numpy(),
                    (coverage_summary['mean_coverage']-coverage_summary['sd_coverage']).clip(0,1).to_numpy(),
                    (coverage_summary['mean_coverage']+coverage_summary['sd_coverage']).clip(0,1).to_numpy(),
                    color='#C44E52', alpha=0.2, label='±1 SD across repetitions')
    ax.set_xlabel('Response probability threshold')
    ax.set_ylabel('Coverage ratio (fraction of cells ≥ threshold)')
    ax.set_ylim(0,1.02)
    ax.set_title('Tumor-cell coverage curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'coverage_curves.png', dpi=200)
    plt.close(fig)
    return final, sim, response_summary, rep_summary, coverage, coverage_summary


def composition_iou(selected):
    sets = selected.groupby('repetition')['peptide'].apply(lambda x: set(x)).to_dict()
    reps = sorted(sets)
    mat = pd.DataFrame(index=reps, columns=reps, dtype=float)
    rows=[]
    for i in reps:
        for j in reps:
            inter=len(sets[i] & sets[j]); union=len(sets[i] | sets[j])
            val=inter/union if union else np.nan
            mat.loc[i,j]=val
            rows.append({'rep_i':i,'rep_j':j,'intersection':inter,'union':union,'iou':val})
    mat.index.name='repetition'; mat.to_csv(OUT / 'composition_iou_matrix.csv')
    pd.DataFrame(rows).to_csv(OUT / 'composition_iou_pairs.csv', index=False)
    off=[r['iou'] for r in rows if r['rep_i']<r['rep_j']]
    summ=pd.DataFrame([{'n_repetitions':len(reps),'mean_pairwise_iou':np.mean(off), 'sd_pairwise_iou':np.std(off, ddof=1),
                        'min_pairwise_iou':np.min(off),'max_pairwise_iou':np.max(off)}])
    summ.to_csv(OUT / 'composition_iou_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(mat.astype(float), vmin=0, vmax=1, cmap='viridis', annot=True, fmt='.2f', square=True, cbar_kws={'label':'IoU'}, ax=ax)
    ax.set_title('IoU of selected vaccine compositions')
    ax.set_xlabel('Repetition'); ax.set_ylabel('Repetition')
    fig.tight_layout()
    fig.savefig(IMG / 'composition_iou_heatmap.png', dpi=200)
    plt.close(fig)
    return mat, summ


def runtime_analysis(runtime, selected):
    runtime = runtime.copy()
    summary = runtime.groupby('PopulationSize').agg(
        n_samples=('SampleID','nunique'), mean_runtime_sec=('RunTime','mean'), sd_runtime_sec=('RunTime','std'),
        median_runtime_sec=('RunTime','median'), min_runtime_sec=('RunTime','min'), max_runtime_sec=('RunTime','max')
    ).reset_index()
    # log-log scaling fit
    x=np.log10(runtime['PopulationSize'].to_numpy(dtype=float)); y=np.log10(runtime['RunTime'].to_numpy(dtype=float))
    slope, intercept=np.polyfit(x,y,1)
    pred=intercept+slope*x
    r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    fit=pd.DataFrame([{'log10_runtime_vs_log10_population_slope':slope,'intercept':intercept,'r2':r2}])
    summary.to_csv(OUT / 'runtime_summary.csv', index=False)
    fit.to_csv(OUT / 'runtime_scaling_fit.csv', index=False)
    selected.groupby('repetition')['run_time'].first().reset_index().to_csv(OUT / 'selected_optimizer_runtime_by_repetition.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.lineplot(data=runtime, x='PopulationSize', y='RunTime', hue='SampleID', marker='o', ax=ax, palette='tab10', legend='brief')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Population size (cells, log scale)')
    ax.set_ylabel('Optimization runtime (s, log scale)')
    ax.set_title(f'Runtime scaling across patient samples (slope={slope:.2f}, R²={r2:.2f})')
    ax.legend(title='SampleID', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig(IMG / 'runtime_scaling.png', dpi=200)
    plt.close(fig)
    return summary, fit


def element_level(scores, selected):
    # Mean element response across all cells/reps; mark selected frequency.
    selected_freq=selected.groupby('peptide')['repetition'].nunique().rename('selected_repetitions')
    el=scores.groupby('vaccine_element').agg(
        mean_cell_response=('p_response','mean'), sd_cell_response=('p_response','std'),
        median_cell_response=('p_response','median'), n_cell_element_pairs=('p_response','size')
    ).reset_index().merge(selected_freq, left_on='vaccine_element', right_index=True, how='left')
    el['selected_repetitions']=el['selected_repetitions'].fillna(0).astype(int)
    el.to_csv(OUT / 'element_response_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.scatterplot(data=el, x='mean_cell_response', y='selected_repetitions', size='n_cell_element_pairs', hue='selected_repetitions', palette='viridis', ax=ax, legend=False)
    ax.set_xlabel('Mean cell-level response probability for element')
    ax.set_ylabel('Selected repetitions')
    ax.set_title('Element response signal and optimizer selection frequency')
    fig.tight_layout()
    fig.savefig(IMG / 'element_response_vs_selection.png', dpi=200)
    plt.close(fig)
    return el


def validation_and_claims(overview, response_summary, coverage_summary, iou_summary, runtime_summary, runtime_fit, comp):
    # Important thresholds direct table
    thresholds=[0.5,0.75,0.9,0.95]
    cov=pd.read_csv(OUT/'coverage_summary_by_threshold.csv')
    direct=cov[cov['threshold'].isin(thresholds)].copy()
    direct.to_csv(OUT/'direct_coverage_thresholds.csv', index=False)
    claims=[]
    rs=response_summary.iloc[0]
    claims.append({'claim':'Mean per-cell response probability under budget-10 MinSum adaptive vaccine', 'value':f"{rs['mean_p_response']:.6f}", 'supporting_artifact':'outputs/response_summary_by_vaccine.csv'})
    claims.append({'claim':'Median per-cell response probability under budget-10 MinSum adaptive vaccine', 'value':f"{rs['median_p_response']:.6f}", 'supporting_artifact':'outputs/response_summary_by_vaccine.csv'})
    for t in thresholds:
        row=direct[direct['threshold']==t].iloc[0]
        claims.append({'claim':f'Mean coverage ratio at p_response >= {t}', 'value':f"{row['mean_coverage']:.6f}", 'supporting_artifact':'outputs/direct_coverage_thresholds.csv'})
    io=iou_summary.iloc[0]
    claims.append({'claim':'Mean pairwise IoU of selected vaccine compositions across repetitions', 'value':f"{io['mean_pairwise_iou']:.6f}", 'supporting_artifact':'outputs/composition_iou_summary.csv'})
    rf=runtime_fit.iloc[0]
    claims.append({'claim':'Runtime log-log scaling exponent across supplied patient samples', 'value':f"{rf['log10_runtime_vs_log10_population_slope']:.6f} (R2={rf['r2']:.6f})", 'supporting_artifact':'outputs/runtime_scaling_fit.csv'})
    claims.append({'claim':'All repetitions selected exactly budget-10 elements', 'value':str(bool((pd.read_csv(OUT/'selected_vaccine_per_repetition.csv')['n_elements']==10).all())), 'supporting_artifact':'outputs/selected_vaccine_per_repetition.csv'})
    pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv', index=False)
    return direct, pd.DataFrame(claims)


def update_inventory():
    path=OUT/'target_artifact_inventory.json'
    inv=json.load(open(path))
    for section, items in inv.items():
        if isinstance(items, list):
            for item in items:
                art=item.get('artifact')
                if art and (ROOT/art).exists():
                    item['status']='satisfied'
                elif art:
                    item['status']='unsatisfied'
                    item['reason']='File not produced by analysis script'
    json.dump(inv, open(path,'w'), indent=2)


def main():
    cell, final, sim, runtime, selected, vacc, scores = load()
    overview = save_data_overview(cell, final, sim, runtime, selected, vacc, scores)
    per_rep, comp = composition(selected, vacc)
    final2, sim2, response_summary, rep_summary, coverage, coverage_summary = response_and_coverage(final, sim)
    iou_mat, iou_summary = composition_iou(selected)
    runtime_summary, runtime_fit = runtime_analysis(runtime, selected)
    el = element_level(scores, selected)
    direct, claims = validation_and_claims(overview, response_summary, coverage_summary, iou_summary, runtime_summary, runtime_fit, comp)
    update_inventory()
    print('Analysis complete')
    print(claims.to_string(index=False))

if __name__ == '__main__':
    main()
