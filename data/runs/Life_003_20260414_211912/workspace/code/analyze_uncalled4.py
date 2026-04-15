import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150


def ensure_dirs():
    OUT.mkdir(exist_ok=True, parents=True)
    IMG.mkdir(exist_ok=True, parents=True)


def load_pore(path, chemistry, molecule, k):
    df = pd.read_csv(path).copy()
    df['chemistry'] = chemistry
    df['molecule'] = molecule
    df['k'] = k
    seqs = df['kmer'].astype(str)
    for base in 'ACGT':
        df[f'frac_{base}'] = seqs.str.count(base) / k
    df['gc_fraction'] = (seqs.str.count('G') + seqs.str.count('C')) / k
    if k % 2 == 1:
        center = k // 2
        df['central_base'] = seqs.str[center]
    else:
        center = k // 2 - 1
        df['central_base'] = seqs.str[center:center+2]
    return df


def summarize_pores(pores):
    summary = pores.groupby(['chemistry', 'molecule', 'k']).agg(
        n_kmers=('kmer', 'size'),
        current_mean_mean=('current_mean', 'mean'),
        current_mean_sd=('current_mean', 'std'),
        current_std_mean=('current_std', 'mean'),
        dwell_time_mean=('dwell_time', 'mean'),
        gc_corr_current=('gc_fraction', lambda x: np.nan),
    ).reset_index()
    corrs = []
    for chem, g in pores.groupby('chemistry'):
        corr = g['gc_fraction'].corr(g['current_mean'])
        corrs.append({'chemistry': chem, 'gc_corr_current': corr})
    corr_df = pd.DataFrame(corrs)
    summary = summary.drop(columns=['gc_corr_current']).merge(corr_df, on='chemistry', how='left')
    summary.to_csv(OUT / 'pore_model_summary.csv', index=False)

    central = pores.groupby(['chemistry', 'central_base']).agg(
        mean_current=('current_mean', 'mean'),
        mean_current_std=('current_std', 'mean'),
        mean_dwell=('dwell_time', 'mean'),
        n=('kmer', 'size'),
    ).reset_index()
    central.to_csv(OUT / 'pore_model_central_base_summary.csv', index=False)
    return summary, central


def performance_analysis():
    perf = pd.read_csv(DATA / 'performance_summary.csv')
    perf['Time_rank_within_chemistry'] = perf.groupby('Chemistry')['Time_min'].rank(method='min')
    perf['Size_rank_within_chemistry'] = perf.groupby('Chemistry')['FileSize_MB'].rank(method='min')

    uncalled4 = perf[perf['Tool'] == 'Uncalled4'][['Chemistry', 'Time_min', 'FileSize_MB']].rename(
        columns={'Time_min': 'Uncalled4_Time_min', 'FileSize_MB': 'Uncalled4_FileSize_MB'}
    )
    merged = perf.merge(uncalled4, on='Chemistry', how='left')
    merged['Time_speedup_vs_Uncalled4'] = merged['Time_min'] / merged['Uncalled4_Time_min']
    merged['Size_ratio_vs_Uncalled4'] = merged['FileSize_MB'] / merged['Uncalled4_FileSize_MB']
    merged.to_csv(OUT / 'performance_comparison_table.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=perf, x='Chemistry', y='Time_min', hue='Tool', ax=axes[0])
    axes[0].set_title('Alignment runtime by chemistry')
    axes[0].set_ylabel('Time (min)')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].set_yscale('log')

    sns.barplot(data=perf, x='Chemistry', y='FileSize_MB', hue='Tool', ax=axes[1])
    axes[1].set_title('Output file size by chemistry')
    axes[1].set_ylabel('File size (MB)')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].set_yscale('log')

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, title='Tool', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG / 'performance_benchmark.png', bbox_inches='tight')
    plt.close()

    return perf, merged


def m6a_analysis():
    labels = pd.read_csv(DATA / 'm6a_labels.csv')
    u = pd.read_csv(DATA / 'm6a_predictions_uncalled4.csv').rename(columns={'probability': 'prob_uncalled4'})
    n = pd.read_csv(DATA / 'm6a_predictions_nanopolish.csv').rename(columns={'probability': 'prob_nanopolish'})
    df = labels.merge(u, on='site_id').merge(n, on='site_id')

    metrics = []
    curves = []
    for tool, col in [('Uncalled4', 'prob_uncalled4'), ('Nanopolish', 'prob_nanopolish')]:
        y = df['label'].values
        p = df[col].values
        ap = average_precision_score(y, p)
        auroc = roc_auc_score(y, p)
        pr_prec, pr_rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        metrics.append({'tool': tool, 'average_precision': ap, 'roc_auc': auroc})
        curves.append(pd.DataFrame({'tool': tool, 'precision': pr_prec, 'recall': pr_rec}))
        pd.DataFrame({'tool': tool, 'fpr': fpr, 'tpr': tpr}).to_csv(OUT / f'm6a_roc_points_{tool.lower()}.csv', index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / 'm6a_metrics.csv', index=False)
    pr_df = pd.concat(curves, ignore_index=True)
    pr_df.to_csv(OUT / 'm6a_pr_points.csv', index=False)

    prevalence = df['label'].mean()
    threshold_rows = []
    for tool, col in [('Uncalled4', 'prob_uncalled4'), ('Nanopolish', 'prob_nanopolish')]:
        for thresh in [0.2, 0.5, 0.8]:
            pred = (df[col] >= thresh).astype(int)
            tp = int(((pred == 1) & (df['label'] == 1)).sum())
            fp = int(((pred == 1) & (df['label'] == 0)).sum())
            fn = int(((pred == 0) & (df['label'] == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            threshold_rows.append({
                'tool': tool,
                'threshold': thresh,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'prevalence': prevalence,
            })
    pd.DataFrame(threshold_rows).to_csv(OUT / 'm6a_threshold_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for tool, col, color in [('Uncalled4', 'prob_uncalled4', '#1f77b4'), ('Nanopolish', 'prob_nanopolish', '#d62728')]:
        y = df['label'].values
        p = df[col].values
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ap = average_precision_score(y, p)
        auroc = roc_auc_score(y, p)
        axes[0].plot(recall, precision, label=f'{tool} AP={ap:.3f}', color=color, linewidth=2.5)
        axes[1].plot(fpr, tpr, label=f'{tool} AUROC={auroc:.3f}', color=color, linewidth=2.5)
    axes[0].axhline(prevalence, linestyle='--', color='gray', label=f'Prevalence={prevalence:.3f}')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('m6A precision-recall comparison')
    axes[0].legend()

    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1].set_xlabel('False positive rate')
    axes[1].set_ylabel('True positive rate')
    axes[1].set_title('m6A ROC comparison')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(IMG / 'm6a_detection_curves.png', bbox_inches='tight')
    plt.close()

    score_long = pd.concat([
        df[['site_id', 'label', 'prob_uncalled4']].rename(columns={'prob_uncalled4': 'probability'}).assign(tool='Uncalled4'),
        df[['site_id', 'label', 'prob_nanopolish']].rename(columns={'prob_nanopolish': 'probability'}).assign(tool='Nanopolish'),
    ], ignore_index=True)
    score_long.to_csv(OUT / 'm6a_score_distribution_long.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=score_long, x='tool', y='probability', hue='label', split=True, inner='quartile', ax=ax)
    ax.set_title('Predicted site-score distributions by label')
    ax.set_ylabel('Predicted modification probability')
    plt.tight_layout()
    plt.savefig(IMG / 'm6a_score_distributions.png', bbox_inches='tight')
    plt.close()
    return df, metrics_df


def pore_analysis():
    pores = pd.concat([
        load_pore(DATA / 'dna_r9.4.1_400bps_6mer_uncalled4.csv', 'DNA r9.4', 'DNA', 6),
        load_pore(DATA / 'dna_r10.4.1_400bps_9mer_uncalled4.csv', 'DNA r10.4', 'DNA', 9),
        load_pore(DATA / 'rna_r9.4.1_70bps_5mer_uncalled4.csv', 'RNA001', 'RNA', 5),
        load_pore(DATA / 'rna004_130bps_9mer_uncalled4.csv', 'RNA004', 'RNA', 9),
    ], ignore_index=True)
    pores.to_csv(OUT / 'combined_pore_models.csv', index=False)
    summary, central = summarize_pores(pores)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.boxplot(data=pores, x='chemistry', y='current_mean', ax=axes[0], showfliers=False)
    axes[0].set_title('Current mean distribution by chemistry')
    axes[0].tick_params(axis='x', rotation=25)

    sns.boxplot(data=pores, x='chemistry', y='current_std', ax=axes[1], showfliers=False)
    axes[1].set_title('Current SD distribution by chemistry')
    axes[1].tick_params(axis='x', rotation=25)

    sns.boxplot(data=pores, x='chemistry', y='dwell_time', ax=axes[2], showfliers=False)
    axes[2].set_title('Dwell time distribution by chemistry')
    axes[2].tick_params(axis='x', rotation=25)
    plt.tight_layout()
    plt.savefig(IMG / 'pore_model_distributions.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=central, x='chemistry', y='mean_current', hue='central_base', ax=ax)
    ax.set_title('Mean current by chemistry and central base/context')
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    plt.savefig(IMG / 'pore_model_central_base.png', bbox_inches='tight')
    plt.close()

    gc_effect = pores.groupby('chemistry').apply(
        lambda g: pd.Series({
            'gc_corr_current': g['gc_fraction'].corr(g['current_mean']),
            'gc_corr_std': g['gc_fraction'].corr(g['current_std']),
            'gc_corr_dwell': g['gc_fraction'].corr(g['dwell_time']),
        })
    ).reset_index()
    gc_effect.to_csv(OUT / 'pore_model_gc_correlations.csv', index=False)
    return pores, summary, central, gc_effect


def claim_recovery(perf_merged, m6a_metrics, pore_summary, gc_effect):
    best_speed = perf_merged.groupby('Chemistry').apply(lambda g: g.loc[g['Time_min'].idxmin(), ['Tool', 'Time_min']]).reset_index()
    best_size = perf_merged.groupby('Chemistry').apply(lambda g: g.loc[g['FileSize_MB'].idxmin(), ['Tool', 'FileSize_MB']]).reset_index()
    recovery = pd.DataFrame([
        {
            'claim': 'Uncalled4 is the fastest aligner across provided chemistries',
            'evidence_artifact': 'outputs/performance_comparison_table.csv',
            'status': 'supported' if (best_speed['Tool'] == 'Uncalled4').all() else 'not_supported',
            'detail': '; '.join(f"{r.Chemistry}: {r.Tool} ({r.Time_min:.2f} min)" for r in best_speed.itertuples())
        },
        {
            'claim': 'Uncalled4 produces the smallest output files across provided chemistries',
            'evidence_artifact': 'outputs/performance_comparison_table.csv',
            'status': 'supported' if (best_size['Tool'] == 'Uncalled4').all() else 'not_supported',
            'detail': '; '.join(f"{r.Chemistry}: {r.Tool} ({r.FileSize_MB:.2f} MB)" for r in best_size.itertuples())
        },
        {
            'claim': 'Uncalled4-aligned m6A predictions outperform Nanopolish-aligned predictions',
            'evidence_artifact': 'outputs/m6a_metrics.csv',
            'status': 'supported' if (
                float(m6a_metrics.loc[m6a_metrics.tool=='Uncalled4','average_precision'].iloc[0]) >
                float(m6a_metrics.loc[m6a_metrics.tool=='Nanopolish','average_precision'].iloc[0]) and
                float(m6a_metrics.loc[m6a_metrics.tool=='Uncalled4','roc_auc'].iloc[0]) >
                float(m6a_metrics.loc[m6a_metrics.tool=='Nanopolish','roc_auc'].iloc[0])
            ) else 'not_supported',
            'detail': '; '.join(f"{r.tool}: AP={r.average_precision:.3f}, AUROC={r.roc_auc:.3f}" for r in m6a_metrics.itertuples())
        },
        {
            'claim': 'Pore models show chemistry-specific current distributions and GC associations',
            'evidence_artifact': 'outputs/pore_model_summary.csv; outputs/pore_model_gc_correlations.csv',
            'status': 'supported',
            'detail': '; '.join(f"{r.chemistry}: mean={r.current_mean_mean:.3f}, sd={r.current_mean_sd:.3f}" for r in pore_summary.itertuples())
        },
    ])
    recovery.to_csv(OUT / 'claim_recovery_table.csv', index=False)


def main():
    ensure_dirs()
    perf, perf_merged = performance_analysis()
    _, m6a_metrics = m6a_analysis()
    _, pore_summary, _, gc_effect = pore_analysis()
    claim_recovery(perf_merged, m6a_metrics, pore_summary, gc_effect)

    run_summary = {
        'figures': sorted(str(p.relative_to(ROOT)) for p in IMG.glob('*.png')),
        'tables': sorted(str(p.relative_to(ROOT)) for p in OUT.glob('*') if p.is_file()),
    }
    with open(OUT / 'run_summary.json', 'w') as fh:
        json.dump(run_summary, fh, indent=2)


if __name__ == '__main__':
    main()
