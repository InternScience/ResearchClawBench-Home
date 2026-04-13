
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_theme(style='whitegrid', context='talk')
ROOT = Path('.')
DATA = ROOT/'data'
OUT = ROOT/'outputs'
IMG = ROOT/'report'/'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

perf = pd.read_csv(DATA/'performance_summary.csv')
labels = pd.read_csv(DATA/'m6a_labels.csv')
unc = pd.read_csv(DATA/'m6a_predictions_uncalled4.csv')
nano = pd.read_csv(DATA/'m6a_predictions_nanopolish.csv')
merged = labels.merge(unc, on='site_id').merge(nano, on='site_id', suffixes=('_uncalled4', '_nanopolish'))
merged = merged.rename(columns={'probability_uncalled4':'uncalled4_probability','probability_nanopolish':'nanopolish_probability'})

perf_piv_time = perf.pivot(index='Chemistry', columns='Tool', values='Time_min')
rows=[]
for chem in perf['Chemistry'].unique():
    sub=perf[perf['Chemistry']==chem].copy()
    u=sub[sub.Tool=='Uncalled4'].iloc[0]
    for _,r in sub.iterrows():
        rows.append({'Chemistry': chem,'Tool': r.Tool,'time_speedup_vs_uncalled4': r.Time_min/u.Time_min,'size_ratio_vs_uncalled4': r.FileSize_MB/u.FileSize_MB})
perf_relative = pd.DataFrame(rows)

metrics=[]
for tool,col in [('Uncalled4','uncalled4_probability'),('Nanopolish','nanopolish_probability')]:
    y=merged['label'].values
    s=merged[col].values
    prec, rec, thr = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    ap = average_precision_score(y,s)
    roc_auc = auc(fpr,tpr)
    f1_arr = (2*prec*rec)/(prec+rec+1e-12)
    best_idx = int(np.nanargmax(f1_arr))
    best_thr = 1.0 if best_idx==0 else float(thr[best_idx-1])
    pred = (s >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y,pred).ravel()
    metrics.append({'tool': tool,'average_precision': ap,'roc_auc': roc_auc,'best_f1_threshold': best_thr,'precision_at_best_f1': precision_score(y,pred),'recall_at_best_f1': recall_score(y,pred),'f1_at_best_f1': f1_score(y,pred),'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
metrics_df = pd.DataFrame(metrics)

thresholds = np.linspace(0,1,101)
curve_rows=[]
for tool,col in [('Uncalled4','uncalled4_probability'),('Nanopolish','nanopolish_probability')]:
    y=merged['label'].values
    s=merged[col].values
    for t in thresholds:
        pred=(s>=t).astype(int)
        tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
        prec=tp/(tp+fp) if tp+fp else 1.0
        rec=tp/(tp+fn) if tp+fn else 0.0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
        curve_rows.append({'tool':tool,'threshold':t,'precision':prec,'recall':rec,'f1':f1,'positives_called':int(pred.sum())})
threshold_df = pd.DataFrame(curve_rows)

model_files = {'DNA R9.4.1 6mer': DATA/'dna_r9.4.1_400bps_6mer_uncalled4.csv','DNA R10.4.1 9mer': DATA/'dna_r10.4.1_400bps_9mer_uncalled4.csv','RNA R9.4.1 5mer': DATA/'rna_r9.4.1_70bps_5mer_uncalled4.csv','RNA004 9mer': DATA/'rna004_130bps_9mer_uncalled4.csv'}
summary_rows=[]
basepos_rows=[]
for name,path in model_files.items():
    df=pd.read_csv(path)
    k=len(df.iloc[0]['kmer'])
    for b in 'ACGT':
        frac=df['kmer'].str.count(b)/k
        for feature in ['current_mean','current_std','dwell_time']:
            corr=np.corrcoef(frac, df[feature])[0,1]
            summary_rows.append({'model':name,'feature':feature,'base':b,'correlation':corr})
    for pos in range(k):
        chars=df['kmer'].str[pos]
        for b in 'ACGT':
            mask=(chars==b)
            basepos_rows.append({'model':name,'position':pos+1,'base':b,'mean_current':df.loc[mask,'current_mean'].mean(),'mean_std':df.loc[mask,'current_std'].mean(),'mean_dwell':df.loc[mask,'dwell_time'].mean()})
basecorr_df = pd.DataFrame(summary_rows)
basepos_df = pd.DataFrame(basepos_rows)

perf_relative.to_csv(OUT/'performance_relative_metrics.csv', index=False)
metrics_df.to_csv(OUT/'m6a_classification_metrics.csv', index=False)
threshold_df.to_csv(OUT/'m6a_threshold_metrics.csv', index=False)
basecorr_df.to_csv(OUT/'pore_model_base_correlations.csv', index=False)
basepos_df.to_csv(OUT/'pore_model_position_effects.csv', index=False)
merged.to_csv(OUT/'m6a_merged_predictions.csv', index=False)

fig, axes = plt.subplots(1,2, figsize=(16,6))
sns.barplot(data=perf, x='Chemistry', y='Time_min', hue='Tool', ax=axes[0])
axes[0].set_title('Alignment runtime across chemistries'); axes[0].set_ylabel('Time (min)'); axes[0].tick_params(axis='x', rotation=25); axes[0].legend(frameon=False, ncol=2)
sns.barplot(data=perf, x='Chemistry', y='FileSize_MB', hue='Tool', ax=axes[1])
axes[1].set_title('Output file size across chemistries'); axes[1].set_ylabel('File size (MB)'); axes[1].tick_params(axis='x', rotation=25); axes[1].legend_.remove()
fig.tight_layout(); fig.savefig(IMG/'performance_benchmarks.png', dpi=220, bbox_inches='tight'); plt.close(fig)

rel = perf_relative[perf_relative['Tool']!='Uncalled4'].copy()
fig, axes = plt.subplots(1,2, figsize=(14,6))
sns.barplot(data=rel, x='Chemistry', y='time_speedup_vs_uncalled4', hue='Tool', ax=axes[0])
axes[0].set_title('Runtime ratio relative to Uncalled4'); axes[0].set_ylabel('Competitor / Uncalled4 runtime'); axes[0].axhline(1, color='black', lw=1); axes[0].tick_params(axis='x', rotation=25); axes[0].legend(frameon=False)
sns.barplot(data=rel, x='Chemistry', y='size_ratio_vs_uncalled4', hue='Tool', ax=axes[1])
axes[1].set_title('File size ratio relative to Uncalled4'); axes[1].set_ylabel('Competitor / Uncalled4 file size'); axes[1].axhline(1, color='black', lw=1); axes[1].tick_params(axis='x', rotation=25); axes[1].legend_.remove()
fig.tight_layout(); fig.savefig(IMG/'uncalled4_relative_efficiency.png', dpi=220, bbox_inches='tight'); plt.close(fig)

fig, axes = plt.subplots(1,2, figsize=(14,6))
for tool,col,color in [('Uncalled4','uncalled4_probability','#1f77b4'),('Nanopolish','nanopolish_probability','#d62728')]:
    y=merged['label'].values; s=merged[col].values
    prec, rec, _ = precision_recall_curve(y,s)
    fpr, tpr, _ = roc_curve(y,s)
    ap=average_precision_score(y,s); ra=auc(fpr,tpr)
    axes[0].plot(rec, prec, label=f'{tool} AP={ap:.3f}', color=color, lw=2.5)
    axes[1].plot(fpr, tpr, label=f'{tool} ROC AUC={ra:.3f}', color=color, lw=2.5)
axes[0].axhline(merged['label'].mean(), ls='--', color='gray', label=f"Baseline={merged['label'].mean():.3f}")
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision'); axes[0].set_title('m6A precision-recall')
axes[1].plot([0,1],[0,1], ls='--', color='gray'); axes[1].set_xlabel('False positive rate'); axes[1].set_ylabel('True positive rate'); axes[1].set_title('m6A ROC')
for ax in axes: ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(IMG/'m6a_pr_roc.png', dpi=220, bbox_inches='tight'); plt.close(fig)

fig, axes = plt.subplots(1,2, figsize=(14,6))
sns.lineplot(data=threshold_df, x='threshold', y='f1', hue='tool', ax=axes[0], lw=2.5)
axes[0].set_title('F1 versus decision threshold'); axes[0].set_ylabel('F1 score')
sns.lineplot(data=threshold_df, x='threshold', y='positives_called', hue='tool', ax=axes[1], lw=2.5)
axes[1].set_title('Sites called positive versus threshold'); axes[1].set_ylabel('Positive calls')
for ax in axes: ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(IMG/'m6a_threshold_tradeoff.png', dpi=220, bbox_inches='tight'); plt.close(fig)

models = list(model_files.keys())
fig, axes = plt.subplots(2,2, figsize=(16,12), sharey=False)
for ax, model in zip(axes.flat, models):
    sub=basepos_df[basepos_df['model']==model]
    sns.lineplot(data=sub, x='position', y='mean_current', hue='base', marker='o', ax=ax)
    ax.set_title(model); ax.set_ylabel('Mean current'); ax.set_xlabel('Position in k-mer'); ax.legend(frameon=False, ncol=2)
fig.suptitle('Base-position effects on pore-model current', y=1.02)
fig.tight_layout(); fig.savefig(IMG/'pore_model_position_effects.png', dpi=220, bbox_inches='tight'); plt.close(fig)

corr_heat = basecorr_df.copy(); corr_heat['label']=corr_heat['base']+' vs '+corr_heat['feature']
pivot = corr_heat.pivot(index='model', columns='label', values='correlation')
fig, ax = plt.subplots(figsize=(14,6))
sns.heatmap(pivot, cmap='coolwarm', center=0, annot=True, fmt='.2f', ax=ax)
ax.set_title('Correlation between base composition and pore-model features')
fig.tight_layout(); fig.savefig(IMG/'pore_model_base_correlation_heatmap.png', dpi=220, bbox_inches='tight'); plt.close(fig)

summary = {'n_sites': int(len(merged)),'positive_rate': float(merged['label'].mean()),'best_tool_by_ap': metrics_df.sort_values('average_precision', ascending=False).iloc[0]['tool'],'metrics': metrics_df.to_dict(orient='records'),'fastest_tool_per_chemistry': perf.loc[perf.groupby('Chemistry')['Time_min'].idxmin(), ['Chemistry','Tool','Time_min']].to_dict(orient='records')}
with open(OUT/'analysis_summary.json','w') as f: json.dump(summary, f, indent=2)
print('analysis complete')
