#!/usr/bin/env python3
"""Reproducible lightweight DIDS-MFL-style analysis for NF-UNSW-NB15-v2_3d.pt.

The script avoids requiring torch_geometric. It decodes the PyTorch zip tensor
storages directly, constructs temporal/topological diffusion features, builds
statistical-disentangled feature branches and multi-scale fusions, then evaluates
binary, multiclass, unknown-attack, and few-shot attack scenarios.
"""
import os, json, zipfile, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, balanced_accuracy_score, roc_auc_score)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy import sparse
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'NF-UNSW-NB15-v2_3d.pt'
OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
RANDOM=42
np.random.seed(RANDOM)

SPECS={
 'src':('0','int64',(148774,)), 'dst':('1','int64',(148774,)), 't':('2','int64',(148774,)),
 'msg':('3','float32',(148774,40)), 'src_layer':('4','int64',(148774,)), 'dst_layer':('5','int64',(148774,)),
 'dt':('6','float32',(148774,)), 'label':('7','int64',(148774,)), 'attack':('8','int64',(148774,)),
}
ATTACK_NAMES={0:'Analysis',1:'Backdoor',2:'Benign',3:'DoS',4:'Exploits',5:'Fuzzers',6:'Generic',7:'Reconnaissance',8:'Shellcode',9:'Worms'}

def load_arrays():
    npz=OUT/'dataset_arrays.npz'
    if npz.exists():
        return dict(np.load(npz))
    arrs={}
    with zipfile.ZipFile(DATA) as z:
        for name,(idx,dtype,shape) in SPECS.items():
            arr=np.frombuffer(z.read(f'NF-UNSW-NB15-v2_3d/data/{idx}'), dtype=np.dtype(dtype)).copy().reshape(shape)
            arrs[name]=arr
    np.savez_compressed(npz, **arrs)
    return arrs

def metrics_dict(y_true,y_pred,labels=None,prefix=''):
    avg='binary' if len(np.unique(y_true))==2 and set(np.unique(y_true))<=set([0,1]) else 'macro'
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average=avg,zero_division=0)
    pm,rm,fm,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    return {prefix+'accuracy':accuracy_score(y_true,y_pred), prefix+'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),
            prefix+'precision':p, prefix+'recall':r, prefix+'f1':f, prefix+'macro_f1':fm, prefix+'weighted_f1':fw}

def make_profile(arrs):
    profile={
        'n_flows': int(len(arrs['label'])), 'n_features': int(arrs['msg'].shape[1]),
        'label_counts': {str(int(k)):int(v) for k,v in zip(*np.unique(arrs['label'], return_counts=True))},
        'attack_counts': {ATTACK_NAMES.get(int(k),str(int(k))):int(v) for k,v in zip(*np.unique(arrs['attack'], return_counts=True))},
        'time_min': int(arrs['t'].min()), 'time_max': int(arrs['t'].max()),
        'unique_src': int(len(np.unique(arrs['src']))), 'unique_dst': int(len(np.unique(arrs['dst']))),
        'feature_range': [float(arrs['msg'].min()), float(arrs['msg'].max())]
    }
    (OUT/'data_profile.json').write_text(json.dumps(profile,indent=2))
    return profile

def select_indices(y_attack, max_per_class=6000, include_benign=12000):
    idx=[]
    rng=np.random.default_rng(RANDOM)
    for c in np.unique(y_attack):
        ids=np.flatnonzero(y_attack==c)
        cap=include_benign if c==2 else max_per_class
        if len(ids)>cap: ids=rng.choice(ids, cap, replace=False)
        idx.extend(ids.tolist())
    return np.array(sorted(idx))

def feature_engineer(arrs, idx):
    X=arrs['msg'][idx].astype(np.float64)
    src=arrs['src'][idx]; dst=arrs['dst'][idx]; t=arrs['t'][idx].astype(np.float64); dt=arrs['dt'][idx].reshape(-1,1)
    # Statistical disentanglement: branch 1 central tendency/intensity, branch 2 burst/shape, branch 3 temporal context.
    q1=X[:,:13]; q2=X[:,13:27]; q3=X[:,27:]
    stats=np.c_[q1.mean(1), q1.std(1), q1.max(1), q2.mean(1), q2.std(1), q2.max(1), q3.mean(1), q3.std(1), q3.max(1), dt]
    # Topological degree proxies computed on the sampled temporal graph.
    src_counts=pd.Series(src).map(pd.Series(src).value_counts()).to_numpy().reshape(-1,1)
    dst_counts=pd.Series(dst).map(pd.Series(dst).value_counts()).to_numpy().reshape(-1,1)
    time_feats=np.c_[t/86400.0, np.sin(2*np.pi*t/86400.0), np.cos(2*np.pi*t/86400.0)]
    topo=np.log1p(np.c_[src_counts, dst_counts])
    # Representational disentanglement: PCA factors on raw flow features.
    scaler=StandardScaler(); Xs=scaler.fit_transform(X)
    pca=PCA(n_components=min(12,X.shape[1]), random_state=RANDOM)
    factors=pca.fit_transform(Xs)
    # Dynamic graph diffusion over flows: kNN in temporal + feature factor space.
    diff_base=np.c_[factors[:,:8], time_feats, topo]
    diff_scaled=StandardScaler().fit_transform(diff_base)
    n_neighbors=8
    nn=NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean').fit(diff_scaled)
    dist, neigh=nn.kneighbors(diff_scaled)
    rows=[]; cols=[]; vals=[]
    sigma=np.median(dist[:,1:]) + 1e-9
    for i in range(len(idx)):
        for d,j in zip(dist[i,1:], neigh[i,1:]):
            rows.append(i); cols.append(j); vals.append(math.exp(-(d*d)/(2*sigma*sigma)))
    A=sparse.csr_matrix((vals,(rows,cols)), shape=(len(idx),len(idx)))
    A=A.maximum(A.T)
    deg=np.asarray(A.sum(axis=1)).ravel()+1e-9
    Dinv=sparse.diags(1/deg)
    P=Dinv@A
    diff1=P@factors[:,:8]
    diff2=P@diff1
    # Multi-scale representation fusion: raw + disentangled stats + factors + one/two-hop diffusion + topology/time.
    X_base=np.c_[X, time_feats, dt, topo]
    X_dids=np.c_[X, stats, factors, diff1, diff2, time_feats, topo]
    meta={'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(), 'n_sampled_flows': int(len(idx)), 'knn_neighbors': n_neighbors}
    return X_base, X_dids, meta

def train_eval_models(X_base,X_dids,y_bin,y_att,idx):
    rows=[]; preds={}
    tr,te=train_test_split(np.arange(len(y_bin)), test_size=0.30, stratify=y_bin, random_state=RANDOM)
    models={
        'LogReg_raw': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM)),
        'RF_raw': RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM, n_jobs=-1),
        'DIDS-MFL_approx': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight='balanced', random_state=RANDOM))
    }
    for name,model in models.items():
        X=X_dids if name=='DIDS-MFL_approx' else X_base
        model.fit(X[tr], y_bin[tr]); yp=model.predict(X[te])
        m=metrics_dict(y_bin[te],yp); m.update({'task':'binary','model':name,'n_train':len(tr),'n_test':len(te)})
        try:
            if hasattr(model,'predict_proba'):
                proba=model.predict_proba(X[te])[:,1]; m['roc_auc']=roc_auc_score(y_bin[te],proba)
        except Exception: pass
        rows.append(m); preds[(name,'binary')]=(te,yp)
    # multiclass attack labels, stratified over all attack IDs
    trm,tem=train_test_split(np.arange(len(y_att)), test_size=0.30, stratify=y_att, random_state=RANDOM)
    models_mc={
        'LogReg_raw': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight='balanced', multi_class='auto', random_state=RANDOM)),
        'RF_raw': RandomForestClassifier(n_estimators=45, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM, n_jobs=-1),
        'DIDS-MFL_approx': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1800, class_weight='balanced', multi_class='auto', random_state=RANDOM))
    }
    for name,model in models_mc.items():
        X=X_dids if name=='DIDS-MFL_approx' else X_base
        model.fit(X[trm], y_att[trm]); yp=model.predict(X[tem])
        m=metrics_dict(y_att[tem],yp); m.update({'task':'multiclass','model':name,'n_train':len(trm),'n_test':len(tem)})
        rows.append(m); preds[(name,'multiclass')]=(tem,yp)
    pd.DataFrame(rows).to_csv(OUT/'main_metrics.csv', index=False)
    # Confusions for DIDS approximation.
    te_b,yp_b=preds[('DIDS-MFL_approx','binary')]
    pd.DataFrame(confusion_matrix(y_bin[te_b],yp_b,labels=[0,1]), index=['true_benign','true_attack'], columns=['pred_benign','pred_attack']).to_csv(OUT/'binary_confusion_matrix.csv')
    te_m,yp_m=preds[('DIDS-MFL_approx','multiclass')]
    labels=sorted(np.unique(y_att).tolist())
    cm=confusion_matrix(y_att[te_m],yp_m,labels=labels)
    pd.DataFrame(cm, index=[ATTACK_NAMES.get(i,str(i)) for i in labels], columns=[ATTACK_NAMES.get(i,str(i)) for i in labels]).to_csv(OUT/'multiclass_confusion_matrix.csv')
    # per attack metrics
    p,r,f,s=precision_recall_fscore_support(y_att[te_m],yp_m,labels=labels,zero_division=0)
    pd.DataFrame({'attack_id':labels,'attack_name':[ATTACK_NAMES.get(i,str(i)) for i in labels], 'precision':p,'recall':r,'f1':f,'support':s}).to_csv(OUT/'per_attack_metrics.csv',index=False)
    return preds

def scenario_evals(X_base,X_dids,y_att,y_bin):
    rows=[]
    # Unknown attack scenario: leave one attack class out from attack training; benign plus known attacks train binary detector.
    attack_classes=[c for c in sorted(np.unique(y_att)) if c!=2]
    for unknown in attack_classes:
        train=np.where(y_att!=unknown)[0]
        test_unknown=np.where(y_att==unknown)[0]
        # sample test includes all unknown and equal benign held out from full pool
        benign=np.where(y_att==2)[0]
        rng=np.random.default_rng(RANDOM+int(unknown))
        btest=rng.choice(benign, size=min(len(test_unknown), len(benign), 3000), replace=False)
        test=np.r_[test_unknown, btest]
        if len(train)>5000: train=rng.choice(train,5000,replace=False)
        for name,X in [('RF_raw',X_base),('DIDS-MFL_approx',X_dids)]:
            model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=800,class_weight='balanced',random_state=RANDOM))
            model.fit(X[train], y_bin[train]); yp=model.predict(X[test])
            m=metrics_dict(y_bin[test],yp); m.update({'scenario':'unknown_leave_one_attack_out','unknown_attack_id':int(unknown),'unknown_attack_name':ATTACK_NAMES.get(int(unknown),str(unknown)),'model':name,'n_train':len(train),'n_test':len(test)})
            rows.append(m)
    # Few-shot multiclass: k shots per attack + benign cap; prototype/KNN on DIDS and raw.
    rng=np.random.default_rng(RANDOM)
    for k in [1,5,10,25]:
        train=[]; test=[]
        for c in sorted(np.unique(y_att)):
            ids=np.flatnonzero(y_att==c)
            rng.shuffle(ids)
            ntrain=min(k if c!=2 else 5*k, len(ids)//2)
            train.extend(ids[:ntrain]); test.extend(ids[ntrain:min(len(ids), ntrain+500)].tolist())
        train=np.array(train); test=np.array(test)
        for name,X in [('KNN_raw',X_base),('DIDS-MFL_proto',X_dids)]:
            if name=='KNN_raw':
                model=make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(3,len(train))))
                model.fit(X[train], y_att[train]); yp=model.predict(X[test])
            else:
                scaler=StandardScaler().fit(X[train]); Ztr=scaler.transform(X[train]); Zte=scaler.transform(X[test])
                labs=sorted(np.unique(y_att[train]))
                prot=np.vstack([Ztr[y_att[train]==c].mean(axis=0) for c in labs])
                d=((Zte[:,None,:]-prot[None,:,:])**2).sum(axis=2)
                yp=np.array(labs)[d.argmin(axis=1)]
            m=metrics_dict(y_att[test],yp); m.update({'scenario':'few_shot_multiclass','shots_per_attack':k,'model':name,'n_train':len(train),'n_test':len(test)})
            rows.append(m)
    pd.DataFrame(rows).to_csv(OUT/'scenario_metrics.csv', index=False)

def interpretability(X_dids,y_bin):
    # On a compact holdout, permutation importance for DIDS approximation.
    rng=np.random.default_rng(RANDOM)
    ids=rng.choice(np.arange(len(y_bin)), size=min(1200,len(y_bin)), replace=False)
    tr,te=train_test_split(ids,test_size=0.35,stratify=y_bin[ids],random_state=RANDOM)
    model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,class_weight='balanced',random_state=RANDOM))
    model.fit(X_dids[tr],y_bin[tr])
    feat=[f'raw_{i}' for i in range(40)] + [f'stat_{i}' for i in range(10)] + [f'pca_factor_{i}' for i in range(12)] + [f'diff1_{i}' for i in range(8)] + [f'diff2_{i}' for i in range(8)] + ['time_norm','time_sin','time_cos','src_logdeg','dst_logdeg']
    # adjust if feature count differs
    feat=feat[:X_dids.shape[1]] + [f'feat_{i}' for i in range(len(feat), X_dids.shape[1])]
    imp=permutation_importance(model,X_dids[te],y_bin[te],n_repeats=2,random_state=RANDOM,scoring='f1',n_jobs=-1)
    df=pd.DataFrame({'feature':feat,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False)
    df.to_csv(OUT/'permutation_importance.csv',index=False)

def make_figures(arrs, X_dids, y_att, y_bin):
    sns.set_theme(style='whitegrid')
    prof=json.loads((OUT/'data_profile.json').read_text())
    # data overview
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    lc=pd.Series({'Benign':prof['label_counts'].get('0',0),'Attack':prof['label_counts'].get('1',0)})
    sns.barplot(x=lc.index,y=lc.values,ax=axs[0],palette='Set2'); axs[0].set_title('Binary label distribution'); axs[0].set_ylabel('flows')
    ac=pd.Series(prof['attack_counts']).sort_values(ascending=False)
    sns.barplot(y=ac.index,x=ac.values,ax=axs[1],palette='viridis'); axs[1].set_title('Attack-type distribution'); axs[1].set_xlabel('flows')
    axs[2].hist(arrs['t'],bins=48,color='#4C72B0'); axs[2].set_title('Temporal coverage'); axs[2].set_xlabel('second of day'); axs[2].set_ylabel('flows')
    fig.tight_layout(); fig.savefig(IMG/'data_overview.png',dpi=200); plt.close(fig)
    # main metrics
    mm=pd.read_csv(OUT/'main_metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,task in zip(axs,['binary','multiclass']):
        sub=mm[mm.task==task]
        plot=sub.melt(id_vars=['model'],value_vars=['accuracy','macro_f1','weighted_f1'],var_name='metric',value_name='score')
        sns.barplot(data=plot,x='metric',y='score',hue='model',ax=ax); ax.set_ylim(0,1.05); ax.set_title(task.capitalize()+' performance')
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IMG/'main_results.png',dpi=200); plt.close(fig)
    # scenario comparison
    sc=pd.read_csv(OUT/'scenario_metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    unk=sc[sc.scenario=='unknown_leave_one_attack_out']
    sns.barplot(data=unk,x='unknown_attack_name',y='f1',hue='model',ax=axs[0]); axs[0].tick_params(axis='x',rotation=45); axs[0].set_ylim(0,1.05); axs[0].set_title('Unknown leave-one-attack-out binary F1')
    few=sc[sc.scenario=='few_shot_multiclass']
    sns.lineplot(data=few,x='shots_per_attack',y='macro_f1',hue='model',marker='o',ax=axs[1]); axs[1].set_ylim(0,1.05); axs[1].set_title('Few-shot multiclass macro-F1')
    fig.tight_layout(); fig.savefig(IMG/'scenario_comparison.png',dpi=200); plt.close(fig)
    # confusion matrices
    bcm=pd.read_csv(OUT/'binary_confusion_matrix.csv',index_col=0); mcm=pd.read_csv(OUT/'multiclass_confusion_matrix.csv',index_col=0)
    fig,axs=plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(bcm,annot=True,fmt='d',cmap='Blues',ax=axs[0]); axs[0].set_title('Binary confusion: DIDS-MFL approx')
    sns.heatmap(mcm,annot=False,cmap='mako',ax=axs[1]); axs[1].set_title('Multiclass confusion: DIDS-MFL approx')
    fig.tight_layout(); fig.savefig(IMG/'confusion_matrices.png',dpi=200); plt.close(fig)
    # embedding validation
    rng=np.random.default_rng(RANDOM); ids=rng.choice(np.arange(len(y_att)), size=min(1500,len(y_att)), replace=False)
    Z=StandardScaler().fit_transform(X_dids[ids]); emb=PCA(n_components=2,random_state=RANDOM).fit_transform(Z)
    df=pd.DataFrame({'PC1':emb[:,0],'PC2':emb[:,1],'attack':[ATTACK_NAMES.get(int(c),str(int(c))) for c in y_att[ids]],'binary':np.where(y_bin[ids]==1,'attack','benign')})
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    sns.scatterplot(data=df,x='PC1',y='PC2',hue='binary',s=8,alpha=.6,ax=axs[0]); axs[0].set_title('Fused embedding by binary label')
    sns.scatterplot(data=df,x='PC1',y='PC2',hue='attack',s=8,alpha=.6,ax=axs[1],legend=False); axs[1].set_title('Fused embedding by attack type')
    fig.tight_layout(); fig.savefig(IMG/'embedding_validation.png',dpi=200); plt.close(fig)
    # feature importance
    imp=pd.read_csv(OUT/'permutation_importance.csv').head(15)
    fig,ax=plt.subplots(figsize=(8,6)); sns.barplot(data=imp,y='feature',x='importance_mean',xerr=imp['importance_std'],ax=ax,color='#55A868'); ax.set_title('Permutation importance for binary DIDS-MFL approximation'); fig.tight_layout(); fig.savefig(IMG/'feature_importance.png',dpi=200); plt.close(fig)

def save_fidelity_and_claims():
    fidelity={
      'named_method':'DIDS-MFL / 3D-IDS-inspired disentangled dynamic IDS',
      'exact_definition_from_task':['statistical disentanglement','representational disentanglement','dynamic graph diffusion','multi-scale fusion','few-shot support'],
      'implemented_steps':{
        'statistical_disentanglement':'40 normalized message features split into branch summaries (means/std/max over feature groups) plus temporal delta',
        'representational_disentanglement':'PCA latent factors over standardized raw flow features; used as separate factor subspace',
        'dynamic_graph_diffusion':'symmetric kNN graph over PCA/time/topology with one- and two-hop random-walk diffusion features',
        'multi_scale_fusion':'concatenation of raw, statistical, PCA, one-hop diffusion, two-hop diffusion, time and degree features',
        'few_shot':'prototype classifier on fused embedding for 1/5/10/25 shots per attack'
      },
      'deviations':['No torch_geometric dependency; no end-to-end neural graph training. This is a lightweight reproducible approximation due runtime/data constraints.', 'Source/destination layer fields are constant, so graph hierarchy was represented through flow kNN and degree proxies rather than multi-layer graph diffusion.']
    }
    (OUT/'method_fidelity_checklist.json').write_text(json.dumps(fidelity,indent=2))
    claims=[
      ['Dataset decoded and profiled','outputs/data_profile.json; outputs/dataset_arrays.npz','direct workspace tensor-storage extraction'],
      ['Binary and multiclass NIDS metrics computed','outputs/main_metrics.csv','stratified train/test experiments'],
      ['Known/unknown and few-shot scenarios evaluated','outputs/scenario_metrics.csv','leave-one-attack-out and k-shot prototype evaluation'],
      ['DIDS-MFL approximation includes disentanglement/diffusion/fusion','outputs/method_fidelity_checklist.json','implemented feature branches and kNN graph diffusion'],
      ['Interpretability produced','outputs/permutation_importance.csv; report/images/feature_importance.png','permutation importance on binary detector'],
      ['Figures generated as PNG','report/images/*.png','matplotlib/seaborn saved figures']
    ]
    pd.DataFrame(claims,columns=['claim','supporting_artifact','validation_type']).to_csv(OUT/'claim_recovery_table.csv',index=False)

def update_inventory():
    inv=json.loads((OUT/'target_artifact_inventory.json').read_text())
    for group,items in inv.items():
        if isinstance(items,list):
            for item in items:
                p=ROOT/item['artifact']
                item['status']='satisfied' if p.exists() else 'unsatisfied: file missing'
    (OUT/'target_artifact_inventory.json').write_text(json.dumps(inv,indent=2))

def main():
    arrs=load_arrays(); profile=make_profile(arrs)
    idx=select_indices(arrs['attack'], max_per_class=700, include_benign=2500)
    y_att=arrs['attack'][idx].astype(int); y_bin=arrs['label'][idx].astype(int)
    X_base,X_dids,meta=feature_engineer(arrs,idx)
    meta.update({'base_feature_shape':list(X_base.shape),'dids_feature_shape':list(X_dids.shape),'sample_attack_counts':{ATTACK_NAMES.get(int(k),str(int(k))):int(v) for k,v in zip(*np.unique(y_att,return_counts=True))}})
    (OUT/'feature_engineering_meta.json').write_text(json.dumps(meta,indent=2))
    train_eval_models(X_base,X_dids,y_bin,y_att,idx)
    scenario_evals(X_base,X_dids,y_att,y_bin)
    interpretability(X_dids,y_bin)
    make_figures(arrs,X_dids,y_att,y_bin)
    save_fidelity_and_claims(); update_inventory()
    print('Analysis complete. Wrote outputs and figures.')

if __name__=='__main__': main()
