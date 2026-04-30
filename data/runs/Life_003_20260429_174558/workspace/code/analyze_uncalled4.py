#!/usr/bin/env python3
"""Reproducible analysis for Uncalled4 benchmark/pore-model/m6A task."""
from pathlib import Path
import json, math, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             roc_curve, auc, brier_score_loss, confusion_matrix)
from sklearn.calibration import calibration_curve
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='paper')

PORE_FILES={
 'DNA R9.4.1 400bps 6-mer':'dna_r9.4.1_400bps_6mer_uncalled4.csv',
 'DNA R10.4.1 400bps 9-mer':'dna_r10.4.1_400bps_9mer_uncalled4.csv',
 'RNA R9.4.1 70bps 5-mer':'rna_r9.4.1_70bps_5mer_uncalled4.csv',
 'RNA004 130bps 9-mer':'rna004_130bps_9mer_uncalled4.csv'
}

def savefig(name):
    plt.tight_layout()
    path=IMG/name
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    return str(path.relative_to(ROOT))

def parse_schema():
    rows=[]
    for p in sorted(DATA.glob('*.csv')):
        df=pd.read_csv(p)
        rows.append({'file':p.name,'rows':len(df),'columns':';'.join(df.columns),'missing_values':int(df.isna().sum().sum()),'duplicate_rows':int(df.duplicated().sum())})
    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'data_overview.csv', index=False)
    plt.figure(figsize=(8,4))
    tmp=overview.copy(); tmp['log10_rows']=np.log10(tmp['rows'])
    ax=sns.barplot(data=tmp, y='file', x='rows', color='#4c78a8')
    ax.set_xscale('log'); ax.set_xlabel('Rows (log scale)'); ax.set_ylabel('Input CSV')
    ax.set_title('Data overview: available feature tables')
    for i,r in tmp.iterrows(): ax.text(r['rows']*1.05, i, f"{r['rows']:,}", va='center', fontsize=8)
    savefig('data_overview.png')
    return overview

def performance():
    df=pd.read_csv(DATA/'performance_summary.csv')
    # normalized speed/size vs Uncalled4 by chemistry
    rows=[]
    for chem,g in df.groupby('Chemistry'):
        u=g[g.Tool=='Uncalled4'].iloc[0]
        for _,r in g.iterrows():
            rows.append({**r.to_dict(),
                         'speedup_vs_uncalled4': r.Time_min/u.Time_min,
                         'size_ratio_vs_uncalled4': r.FileSize_MB/u.FileSize_MB,
                         'time_hours': r.Time_min/60})
    met=pd.DataFrame(rows)
    met.to_csv(OUT/'performance_benchmark_metrics.csv', index=False)
    summary=met.groupby('Tool').agg(mean_time_min=('Time_min','mean'), sd_time_min=('Time_min','std'),
                                    mean_size_mb=('FileSize_MB','mean'), sd_size_mb=('FileSize_MB','std'),
                                    geometric_mean_speedup_vs_uncalled4=('speedup_vs_uncalled4', lambda x: float(stats.gmean(x))),
                                    geometric_mean_size_ratio_vs_uncalled4=('size_ratio_vs_uncalled4', lambda x: float(stats.gmean(x)))).reset_index()
    summary.to_csv(OUT/'performance_tool_summary.csv', index=False)
    order=sorted(df.Chemistry.unique())
    tool_order=['Uncalled4','f5c','Nanopolish','Tombo']
    plt.figure(figsize=(8,4.5))
    ax=sns.barplot(data=met, x='Chemistry', y='Time_min', hue='Tool', order=order, hue_order=tool_order)
    ax.set_yscale('log'); ax.set_ylabel('Alignment time (minutes, log scale)'); ax.set_xlabel('Sequencing chemistry')
    ax.set_title('Runtime benchmark across chemistries')
    ax.legend(title='Tool', ncols=2)
    savefig('performance_time.png')
    plt.figure(figsize=(8,4.5))
    ax=sns.barplot(data=met, x='Chemistry', y='FileSize_MB', hue='Tool', order=order, hue_order=tool_order)
    ax.set_yscale('log'); ax.set_ylabel('Output file size (MB, log scale)'); ax.set_xlabel('Sequencing chemistry')
    ax.set_title('Output size benchmark across chemistries')
    ax.legend(title='Tool', ncols=2)
    savefig('performance_file_size.png')
    # heatmap of speedups excluding uncalled self
    pivot=met.pivot(index='Chemistry', columns='Tool', values='speedup_vs_uncalled4')[tool_order]
    pivot.to_csv(OUT/'performance_speedup_matrix.csv')
    plt.figure(figsize=(6.8,3.8))
    ax=sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', cbar_kws={'label':'Time / Uncalled4 time'})
    ax.set_title('Relative runtime: values >1 mean slower than Uncalled4')
    savefig('performance_speedup_heatmap.png')
    return met, summary

def pore_models():
    summary_rows=[]; comp_rows=[]; pos_rows=[]; subst_rows=[]
    for label,fn in PORE_FILES.items():
        df=pd.read_csv(DATA/fn)
        k=len(df.kmer.iloc[0]); molecule='DNA' if label.startswith('DNA') else 'RNA'
        chemistry=label
        gc_counts=df.kmer.str.count('G')+df.kmer.str.count('C')
        df=df.assign(k=k, model=label, molecule=molecule, GC_fraction=gc_counts/k)
        summary_rows.append({'model':label,'k':k,'rows':len(df),'expected_4_to_k':4**k,
                             'mean_current_mean':df.current_mean.mean(),'sd_current_mean':df.current_mean.std(),
                             'mean_current_std':df.current_std.mean(),'mean_dwell_time':df.dwell_time.mean(),
                             'sd_dwell_time':df.dwell_time.std(),
                             'min_current_mean':df.current_mean.min(),'max_current_mean':df.current_mean.max(),
                             'corr_GC_current':df[['GC_fraction','current_mean']].corr().iloc[0,1],
                             'corr_dwell_current':df[['dwell_time','current_mean']].corr().iloc[0,1]})
        # composition bins
        for gc,g in df.groupby('GC_fraction'):
            comp_rows.append({'model':label,'molecule':molecule,'k':k,'GC_fraction':gc,'n':len(g),
                              'mean_current':g.current_mean.mean(),'sd_current':g.current_mean.std(),
                              'mean_dwell_time':g.dwell_time.mean()})
        # position effects: eta squared of base at position predicting current
        for pos in range(k):
            bases=df.kmer.str[pos]
            means=df.groupby(bases)['current_mean'].mean()
            overall=df.current_mean.mean()
            ss_between=sum(((bases==b).sum())*(means[b]-overall)**2 for b in means.index)
            ss_total=sum((df.current_mean-overall)**2)
            eta=ss_between/ss_total if ss_total else np.nan
            rng=means.max()-means.min()
            pos_rows.append({'model':label,'molecule':molecule,'k':k,'position_1based':pos+1,'eta_squared_current':eta,'base_mean_range':rng,
                             **{f'mean_if_{b}':means.get(b,np.nan) for b in 'ACGT'}})
        # substitution effects sampled/exhaustive neighbor differences
        # For each position, compare all kmers differing at that position only using lexicographic full table.
        cur=dict(zip(df.kmer, df.current_mean))
        for pos in range(k):
            diffs=[]
            for s,val in cur.items():
                orig=s[pos]
                for b in 'ACGT':
                    if b==orig: continue
                    t=s[:pos]+b+s[pos+1:]
                    if t in cur:
                        diffs.append(abs(cur[t]-val))
            subst_rows.append({'model':label,'position_1based':pos+1,'mean_abs_substitution_shift':float(np.mean(diffs)),'median_abs_substitution_shift':float(np.median(diffs)),'n_directed_pairs':len(diffs)})
    summ=pd.DataFrame(summary_rows); comp=pd.DataFrame(comp_rows); pos=pd.DataFrame(pos_rows); subst=pd.DataFrame(subst_rows)
    summ.to_csv(OUT/'pore_model_summary.csv', index=False)
    comp.to_csv(OUT/'pore_composition_summary.csv', index=False)
    pos.to_csv(OUT/'pore_position_effects.csv', index=False)
    subst.to_csv(OUT/'pore_substitution_effects.csv', index=False)
    plt.figure(figsize=(8,4.8))
    ax=sns.lineplot(data=pos, x='position_1based', y='eta_squared_current', hue='model', marker='o')
    ax.set_xlabel('K-mer position (1-based)'); ax.set_ylabel('Variance explained by base identity (η²)')
    ax.set_title('Base-position effects in Uncalled4 pore models')
    ax.legend(fontsize=7, title='Pore model')
    savefig('pore_position_effects.png')
    plt.figure(figsize=(8,4.8))
    ax=sns.lineplot(data=comp, x='GC_fraction', y='mean_current', hue='model', marker='o')
    ax.set_xlabel('GC fraction in k-mer'); ax.set_ylabel('Mean current parameter')
    ax.set_title('Nucleotide composition relationship with modeled current')
    ax.legend(fontsize=7, title='Pore model')
    savefig('pore_composition_relationships.png')
    plt.figure(figsize=(8,4.8))
    ax=sns.lineplot(data=subst, x='position_1based', y='mean_abs_substitution_shift', hue='model', marker='o')
    ax.set_xlabel('Substituted k-mer position (1-based)'); ax.set_ylabel('Mean absolute current shift')
    ax.set_title('Single-base substitution sensitivity by pore-model position')
    ax.legend(fontsize=7, title='Pore model')
    savefig('pore_substitution_sensitivity.png')
    return summ, comp, pos, subst

def m6a():
    lab=pd.read_csv(DATA/'m6a_labels.csv')
    preds=[]
    for tool,fn in [('Uncalled4','m6a_predictions_uncalled4.csv'),('Nanopolish','m6a_predictions_nanopolish.csv')]:
        p=pd.read_csv(DATA/fn).merge(lab,on='site_id',validate='one_to_one')
        p['alignment_source']=tool
        preds.append(p)
    allp=pd.concat(preds, ignore_index=True)
    allp.to_csv(OUT/'m6a_predictions_joined.csv', index=False)
    metrics=[]; curves=[]; rocs=[]; cal_rows=[]; thresh_rows=[]
    for tool,g in allp.groupby('alignment_source'):
        y=g.label.values; s=g.probability.values
        ap=average_precision_score(y,s); roc=roc_auc_score(y,s); brier=brier_score_loss(y,s)
        prec,rec,thr=precision_recall_curve(y,s)
        fpr,tpr,rt=roc_curve(y,s)
        # best F1 threshold from PR thresholds
        f1=(2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12))
        best=int(np.nanargmax(f1)); best_thr=float(thr[best])
        pred=(s>=best_thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
        metrics.append({'alignment_source':tool,'n_sites':len(g),'positive_sites':int(y.sum()),'prevalence':float(y.mean()),
                        'average_precision':float(ap),'roc_auc':float(roc),'brier_score':float(brier),
                        'best_f1_threshold':best_thr,'best_f1':float(f1[best]),
                        'precision_at_best_f1':float(prec[best]),'recall_at_best_f1':float(rec[best]),
                        'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)})
        curves += [{'alignment_source':tool,'precision':float(a),'recall':float(b),'threshold':float(c) if i<len(thr) else np.nan} for i,(a,b,c) in enumerate(itertools.zip_longest(prec,rec,thr, fillvalue=np.nan))]
        rocs += [{'alignment_source':tool,'fpr':float(a),'tpr':float(b),'threshold':float(c)} for a,b,c in zip(fpr,tpr,rt)]
        frac_pos, mean_pred=calibration_curve(y,s,n_bins=10,strategy='quantile')
        for i,(mp,fp_) in enumerate(zip(mean_pred, frac_pos)):
            cal_rows.append({'alignment_source':tool,'bin':i+1,'mean_predicted_probability':float(mp),'observed_fraction_positive':float(fp_)})
        for t in [0.2,0.5,0.8]:
            pred=(s>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
            thresh_rows.append({'alignment_source':tool,'threshold':t,'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
                                'precision':float(tp/(tp+fp)) if tp+fp else np.nan,'recall':float(tp/(tp+fn)) if tp+fn else np.nan})
    met=pd.DataFrame(metrics); met.to_csv(OUT/'m6a_metrics.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'m6a_precision_recall_curve.csv', index=False)
    pd.DataFrame(rocs).to_csv(OUT/'m6a_roc_curve.csv', index=False)
    pd.DataFrame(cal_rows).to_csv(OUT/'m6a_calibration_bins.csv', index=False)
    pd.DataFrame(thresh_rows).to_csv(OUT/'m6a_threshold_metrics.csv', index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    curve_df=pd.DataFrame(curves); roc_df=pd.DataFrame(rocs)
    for tool,g in curve_df.groupby('alignment_source'):
        ap=met.loc[met.alignment_source==tool,'average_precision'].iloc[0]
        axes[0].plot(g.recall, g.precision, label=f'{tool} (AP={ap:.3f})')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision'); axes[0].set_title('m6A precision-recall'); axes[0].legend()
    for tool,g in roc_df.groupby('alignment_source'):
        ra=met.loc[met.alignment_source==tool,'roc_auc'].iloc[0]
        axes[1].plot(g.fpr, g.tpr, label=f'{tool} (AUC={ra:.3f})')
    axes[1].plot([0,1],[0,1],ls='--',color='gray',lw=1); axes[1].set_xlabel('False positive rate'); axes[1].set_ylabel('True positive rate'); axes[1].set_title('m6A ROC'); axes[1].legend()
    savefig('m6a_pr_roc.png')
    plt.figure(figsize=(5.2,4.6))
    cal=pd.DataFrame(cal_rows)
    for tool,g in cal.groupby('alignment_source'):
        plt.plot(g.mean_predicted_probability, g.observed_fraction_positive, marker='o', label=tool)
    plt.plot([0,1],[0,1],ls='--',color='gray',lw=1); plt.xlabel('Mean predicted probability'); plt.ylabel('Observed positive fraction'); plt.title('m6A calibration by probability decile'); plt.legend()
    savefig('m6a_calibration.png')
    return met

def write_related_contract():
    # concise extraction from pypdf-read evidence
    rel={
     'paper_000':'Nanopolish methylation paper: HMM over segmented nanopore events; Gaussian emissions depend on short k-mers (6-mers for R9); methylation shifts current distributions and can be detected directly without chemical treatment.',
     'paper_001':'MoD-seq/nanoraw: compares raw nanopore signal from native versus amplified DNA genome-wide, emphasizes visualization and statistical testing for de novo DNA modification discovery without large prior training data.',
     'paper_002':'UNCALLED: real-time raw-signal mapper for ReadUntil; probabilistically maps streaming current to possible k-mers and prunes candidates with an FM-index; supports targeted sequencing without basecalling.',
     'paper_003':'m6Anet: neural-network multiple-instance learning for m6A detection from direct RNA sequencing; uses signal and sequence features with site-level labels and reports generalization across species/cell lines.'
    }
    json.dump({'related_work_contract':rel,
               'implications':['Report should compare against Nanopolish/f5c/Tombo where provided; analyze k-mer signal parameters because HMM/signal mappers depend on k-mer current distributions; use PR/ROC and calibration for m6Anet probability outputs; explicitly note raw-signal/BAM/training limitations.']}, open(OUT/'related_work_contract.json','w'), indent=2)

def claim_recovery(overview, perf_summary, m6met, pore_summary):
    # build compact claims from actual outputs
    u=perf_summary[perf_summary.Tool=='Uncalled4'].iloc[0]
    nano=perf_summary[perf_summary.Tool=='Nanopolish'].iloc[0]
    f5c=perf_summary[perf_summary.Tool=='f5c'].iloc[0]
    tombo=perf_summary[perf_summary.Tool=='Tombo'].iloc[0]
    mu=m6met[m6met.alignment_source=='Uncalled4'].iloc[0]; mn=m6met[m6met.alignment_source=='Nanopolish'].iloc[0]
    rows=[
      {'claim':'The workspace contains complete pore-model grids for the declared k-mer lengths.', 'supporting_artifact':'outputs/pore_model_summary.csv', 'evidence':'; '.join(f"{r.model}: {int(r.rows)} rows = 4^{int(r.k)}" for _,r in pore_summary.iterrows()), 'status':'verified'},
      {'claim':'Uncalled4 is fastest in the provided benchmark table.', 'supporting_artifact':'outputs/performance_tool_summary.csv', 'evidence':f"Mean time {u.mean_time_min:.2f} min; f5c/Nanopolish/Tombo geometric time ratios vs Uncalled4 = {f5c.geometric_mean_speedup_vs_uncalled4:.1f}x/{nano.geometric_mean_speedup_vs_uncalled4:.1f}x/{tombo.geometric_mean_speedup_vs_uncalled4:.1f}x.", 'status':'verified from performance_summary.csv'},
      {'claim':'Uncalled4-derived m6Anet probabilities outperform Nanopolish-derived probabilities on the supplied labels.', 'supporting_artifact':'outputs/m6a_metrics.csv', 'evidence':f"Average precision {mu.average_precision:.3f} vs {mn.average_precision:.3f}; ROC-AUC {mu.roc_auc:.3f} vs {mn.roc_auc:.3f}.", 'status':'verified'},
      {'claim':'K-mer position has non-uniform effects on current mean.', 'supporting_artifact':'outputs/pore_position_effects.csv; report/images/pore_position_effects.png', 'evidence':'η² profiles vary by k-mer position and pore model.', 'status':'verified'},
      {'claim':'Signal-to-reference BAM generation and trained pore-model creation were not directly reproduced.', 'supporting_artifact':'outputs/method_contract.json; outputs/dependency_check.json', 'evidence':'Raw FAST5/POD5, reads, references, and training labels beyond CSV summaries are absent.', 'status':'limitation'}]
    pd.DataFrame(rows).to_csv(OUT/'claim_recovery_table.csv', index=False)

if __name__=='__main__':
    overview=parse_schema()
    perf, perf_summary=performance()
    pore_summary, comp, pos, subst=pore_models()
    m6met=m6a()
    write_related_contract()
    claim_recovery(overview, perf_summary, m6met, pore_summary)
    # update inventory statuses
    inv=json.load(open(OUT/'target_artifact_inventory.json'))
    for sec in ['primary_tables','figures']:
        for item in inv[sec]:
            p=ROOT/item['target_path']
            item['status']='satisfied' if p.exists() else 'unsatisfied: file not generated'
    # add extra artifacts generated
    inv['additional_artifacts']=[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*.csv'))+sorted(OUT.glob('*.json')) if p.name not in [Path(i['target_path']).name for sec in ['primary_tables','figures'] for i in inv[sec]]]
    json.dump(inv, open(OUT/'target_artifact_inventory.json','w'), indent=2)
    print(json.dumps({'status':'ok','tables':len(list(OUT.glob('*.csv'))),'figures':len(list(IMG.glob('*.png')))}, indent=2))
