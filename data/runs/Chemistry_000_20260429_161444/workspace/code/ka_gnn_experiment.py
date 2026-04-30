#!/usr/bin/env python3
import os, json, time, math, random, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True,exist_ok=True)
SEED=13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cpu'
MAX_N=80
MAX_DATASET_N={'bace':1513,'bbbp':2039,'clintox':1477,'hiv':5000,'muv':6000}
EPOCHS={'bace':8,'bbbp':8,'clintox':8,'hiv':5,'muv':4}

ATOM_LIST=[1,5,6,7,8,9,15,16,17,35,53]
BOND_LIST=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]

def atom_features(a):
    z=a.GetAtomicNum(); hyb=a.GetHybridization()
    feat=[]
    feat += [1.0 if z==v else 0.0 for v in ATOM_LIST] + [1.0 if z not in ATOM_LIST else 0.0]
    feat += [a.GetTotalDegree()/5, a.GetFormalCharge()/3, a.GetTotalNumHs()/4, float(a.GetIsAromatic()), float(a.IsInRing())]
    feat += [1.0 if hyb==h else 0.0 for h in [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]]
    feat += [a.GetMass()/200.0]
    return np.array(feat,dtype=np.float32)

def bond_weight(b):
    w={Chem.BondType.SINGLE:1.0,Chem.BondType.DOUBLE:1.5,Chem.BondType.TRIPLE:2.0,Chem.BondType.AROMATIC:1.3}.get(b.GetBondType(),1.0)
    if b.GetIsConjugated(): w += 0.1
    if b.IsInRing(): w += 0.1
    return w

def mol_descriptors(mol):
    vals=[Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcFractionCSP3(mol), mol.GetNumAtoms(), mol.GetNumBonds()]
    return np.array(vals,dtype=np.float32)

def featurize_smiles(smiles, noncovalent=True):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    n=min(mol.GetNumAtoms(),MAX_N)
    x=np.zeros((MAX_N,21),dtype=np.float32)
    adj=np.zeros((MAX_N,MAX_N),dtype=np.float32)
    for i,a in enumerate(mol.GetAtoms()):
        if i>=MAX_N: break
        x[i]=atom_features(a)
    for b in mol.GetBonds():
        i=b.GetBeginAtomIdx(); j=b.GetEndAtomIdx()
        if i<MAX_N and j<MAX_N:
            w=bond_weight(b); adj[i,j]=adj[j,i]=w
    # non-covalent proxy: connect hetero atoms and aromatic atoms within graph-distance 3-5 with weak edges.
    if noncovalent:
        dmat=Chem.GetDistanceMatrix(mol)
        for i in range(n):
            ai=mol.GetAtomWithIdx(i); zi=ai.GetAtomicNum()
            for j in range(i+1,n):
                if adj[i,j]>0: continue
                dist=dmat[i,j]
                aj=mol.GetAtomWithIdx(j); zj=aj.GetAtomicNum()
                hetero=(zi in (7,8,9,15,16,17) and zj in (7,8,9,15,16,17))
                arom=(ai.GetIsAromatic() and aj.GetIsAromatic())
                if 2.0 <= dist <= 5.0 and (hetero or arom):
                    adj[i,j]=adj[j,i]=0.2/(dist)
    np.fill_diagonal(adj[:n,:n],1.0)
    mask=np.zeros(MAX_N,dtype=np.float32); mask[:n]=1
    desc=mol_descriptors(mol)
    fp=np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512),dtype=np.float32)
    return {'x':x,'adj':adj,'mask':mask,'desc':desc,'fp':fp,'n_atoms':n,'n_bonds':mol.GetNumBonds()}

def label_cols(name, df):
    if name=='clintox': return ['FDA_APPROVED','CT_TOX']
    if name=='muv': return [c for c in df.columns if c.startswith('MUV-')]
    return ['label']

class MolDataset(Dataset):
    def __init__(self, feats, ys, masks): self.feats=feats; self.ys=ys; self.masks=masks
    def __len__(self): return len(self.feats)
    def __getitem__(self,i):
        f=self.feats[i]
        return (torch.tensor(f['x']), torch.tensor(f['adj']), torch.tensor(f['mask']), torch.tensor(self.ys[i]).float(), torch.tensor(self.masks[i]).float())

def collate(batch):
    xs,adjs,masks,ys,ymasks=zip(*batch)
    return torch.stack(xs), torch.stack(adjs), torch.stack(masks), torch.stack(ys), torch.stack(ymasks)

class FourierKAN(nn.Module):
    def __init__(self, in_dim, out_dim, K=4):
        super().__init__(); self.K=K
        self.base=nn.Linear(in_dim,out_dim)
        self.sin=nn.Parameter(torch.randn(K,in_dim,out_dim)*0.03)
        self.cos=nn.Parameter(torch.randn(K,in_dim,out_dim)*0.03)
        self.bias=nn.Parameter(torch.zeros(out_dim))
        self.scale=nn.Parameter(torch.ones(in_dim))
    def forward(self,x):
        z=torch.tanh(x)*self.scale
        out=self.base(x)
        for k in range(1,self.K+1):
            out=out+torch.einsum('...i,io->...o', torch.sin(k*z), self.sin[k-1])
            out=out+torch.einsum('...i,io->...o', torch.cos(k*z), self.cos[k-1])
        return out+self.bias

class GNNLayer(nn.Module):
    def __init__(self,din,dout,kind='mlp'):
        super().__init__(); self.kind=kind
        if kind=='ka': self.phi=FourierKAN(din,dout,K=4)
        else: self.phi=nn.Sequential(nn.Linear(din,dout),nn.ReLU(),nn.Linear(dout,dout))
        self.bn=nn.LayerNorm(dout)
    def forward(self,h,adj,mask):
        deg=adj.sum(-1,keepdim=True).clamp(min=1)
        m=torch.bmm(adj,h)/deg
        out=F.relu(self.bn(self.phi(m)))
        return out*mask.unsqueeze(-1)

class GraphClassifier(nn.Module):
    def __init__(self,in_dim,n_tasks,kind='mlp',hidden=48,layers=2):
        super().__init__(); self.kind=kind
        self.embed=nn.Linear(in_dim,hidden)
        self.layers=nn.ModuleList([GNNLayer(hidden,hidden,kind) for _ in range(layers)])
        head_in=hidden*2+10
        if kind=='ka': self.head=nn.Sequential(FourierKAN(head_in,hidden,K=4),nn.ReLU(),nn.Linear(hidden,n_tasks))
        else: self.head=nn.Sequential(nn.Linear(head_in,hidden),nn.ReLU(),nn.Linear(hidden,n_tasks))
    def forward(self,x,adj,mask,desc=None):
        h=F.relu(self.embed(x))*mask.unsqueeze(-1)
        for layer in self.layers: h=layer(h,adj,mask)
        sumh=h.sum(1); mean=sumh/mask.sum(1,keepdim=True).clamp(min=1); maxh=h.masked_fill(mask.unsqueeze(-1)==0,-1e9).max(1).values
        if desc is None: desc=torch.zeros((x.shape[0],10),device=x.device)
        return self.head(torch.cat([mean,maxh,desc],1))

def split_indices(y, mask):
    idx=np.where(mask.sum(axis=1)>0)[0]
    primary=np.nan_to_num(y[idx,0],nan=0).astype(int)
    try:
        tr,temp=train_test_split(idx,test_size=0.3,random_state=SEED,stratify=primary)
        val,te=train_test_split(temp,test_size=0.5,random_state=SEED,stratify=np.nan_to_num(y[temp,0],nan=0).astype(int))
    except Exception:
        tr,temp=train_test_split(idx,test_size=0.3,random_state=SEED)
        val,te=train_test_split(temp,test_size=0.5,random_state=SEED)
    return tr,val,te

def eval_nn(model, ds, descs, batch=128):
    model.eval(); logits=[]; ys=[]; masks=[]; probs=[]
    loader=DataLoader(ds,batch_size=batch,shuffle=False,collate_fn=collate)
    off=0
    with torch.no_grad():
        for x,adj,mask,y,ym in loader:
            d=torch.tensor(descs[off:off+len(x)]).float(); off+=len(x)
            logit=model(x,adj,mask,d); logits.append(logit); ys.append(y); masks.append(ym); probs.append(torch.sigmoid(logit))
    y=torch.cat(ys).numpy(); m=torch.cat(masks).numpy(); p=torch.cat(probs).numpy()
    return metric_dict(y,p,m), y,p,m

def metric_dict(y,p,m):
    aucs=[]; aps=[]; accs=[]; briers=[]
    for t in range(y.shape[1]):
        ok=m[:,t]>0
        if ok.sum()<3 or len(np.unique(y[ok,t]))<2: continue
        aucs.append(roc_auc_score(y[ok,t],p[ok,t])); aps.append(average_precision_score(y[ok,t],p[ok,t])); accs.append(accuracy_score(y[ok,t],p[ok,t]>=0.5)); briers.append(brier_score_loss(y[ok,t],p[ok,t]))
    return {'roc_auc':float(np.mean(aucs)) if aucs else np.nan,'avg_precision':float(np.mean(aps)) if aps else np.nan,'accuracy':float(np.mean(accs)) if accs else np.nan,'brier':float(np.mean(briers)) if briers else np.nan}

def train_nn(name, kind, feats, y, ymask, tr, val, te, epochs):
    descs=np.stack([f['desc'] for f in feats]); scaler=StandardScaler().fit(descs[tr]); descs=scaler.transform(descs).astype(np.float32)
    dstr=MolDataset([feats[i] for i in tr],y[tr],ymask[tr]); dsval=MolDataset([feats[i] for i in val],y[val],ymask[val]); dste=MolDataset([feats[i] for i in te],y[te],ymask[te])
    model=GraphClassifier(21,y.shape[1],kind=kind).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=None; best_state=None; hist=[]
    t0=time.time()
    for ep in range(epochs):
        model.train(); losses=[]; loader=DataLoader(dstr,batch_size=64,shuffle=True,collate_fn=collate)
        for bi,(x,adj,mask,yy,ym) in enumerate(loader):
            d=torch.tensor(descs[tr][bi*64:bi*64+len(x)]).float() # approximate desc alignment only if shuffle false would fail; handle below by no shuffle for desc
        # redo with no shuffle to keep simple and deterministic
        model.train(); loader=DataLoader(dstr,batch_size=64,shuffle=False,collate_fn=collate)
        off=0
        for x,adj,mask,yy,ym in loader:
            d=torch.tensor(descs[tr][off:off+len(x)]).float(); off+=len(x)
            opt.zero_grad(); logit=model(x,adj,mask,d); loss=F.binary_cross_entropy_with_logits(logit,yy,reduction='none')
            loss=(loss*ym).sum()/ym.sum().clamp(min=1); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        mv,_,_,_=eval_nn(model,dsval,descs[val])
        hist.append({'epoch':ep+1,'loss':np.mean(losses),'val_auc':mv['roc_auc']})
        score=mv['roc_auc'] if not math.isnan(mv['roc_auc']) else -1
        if best is None or score>best: best=score; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    train_time=time.time()-t0
    if best_state: model.load_state_dict(best_state)
    mt,yt,pt,mtask=eval_nn(model,dste,descs[te])
    params=sum(p.numel() for p in model.parameters())
    # inference time
    s=time.time(); eval_nn(model,dste,descs[te]); infer=(time.time()-s)/max(1,len(te))
    return mt, {'params':params,'train_time_sec':train_time,'inference_sec_per_mol':infer}, hist, model, (yt,pt,mtask)

def baseline_fp(name, feats, y, ymask, tr, val, te):
    X=np.stack([np.concatenate([f['fp'],f['desc']]) for f in feats])
    scaler=StandardScaler(with_mean=False).fit(X[tr]); Xs=scaler.transform(X)
    allp=np.zeros_like(y,dtype=float)+0.5
    t0=time.time(); params=0
    for task in range(y.shape[1]):
        oktr=ymask[tr,task]>0
        if oktr.sum()<10 or len(np.unique(y[tr][oktr,task]))<2: continue
        clf=LogisticRegression(max_iter=400,class_weight='balanced',solver='liblinear',random_state=SEED)
        clf.fit(Xs[tr][oktr],y[tr][oktr,task]); allp[:,task]=clf.predict_proba(Xs)[:,1]; params+=Xs.shape[1]
    train_time=time.time()-t0
    mt=metric_dict(y[te],allp[te],ymask[te])
    return mt, {'params':params,'train_time_sec':train_time,'inference_sec_per_mol':0.0}, (y[te],allp[te],ymask[te])

def prepare_dataset(name):
    df=pd.read_csv(DATA/f'{name}.csv')
    tasks=label_cols(name,df)
    # For large datasets, keep all positives and random negatives/labeled rows for computational feasibility.
    if name in ('hiv','muv'):
        rng=np.random.default_rng(SEED)
        labmask=df[tasks].notna().any(axis=1)
        posmask=(df[tasks].fillna(0)==1).any(axis=1)
        keep=np.where(posmask | labmask)[0]
        if len(keep)>MAX_DATASET_N[name]:
            pos=np.where(posmask)[0]; rest=np.setdiff1d(np.where(labmask)[0],pos)
            rest_s=rng.choice(rest,size=max(0,MAX_DATASET_N[name]-len(pos)),replace=False)
            keep=np.concatenate([pos,rest_s])
        df=df.iloc[np.sort(keep)].reset_index(drop=True)
    feats=[]; rows=[]
    for i,smi in enumerate(df['smiles']):
        f=featurize_smiles(smi)
        if f is not None:
            feats.append(f); rows.append(i)
    df=df.iloc[rows].reset_index(drop=True)
    yy=df[tasks].astype(float).values
    ym=(~np.isnan(yy)).astype(np.float32); yy=np.nan_to_num(yy,nan=0.0).astype(np.float32)
    return df,tasks,feats,yy,ym

def main():
    all_results=[]; eff=[]; task_results=[]; histories={}; predictions={}; interprets={}
    for name in ['bace','bbbp','clintox','hiv','muv']:
        print('DATASET',name,flush=True)
        df,tasks,feats,y,ym=prepare_dataset(name)
        tr,val,te=split_indices(y,ym)
        split_info={'train':len(tr),'val':len(val),'test':len(te),'tasks':tasks,'n_featurized':len(feats)}
        with open(OUT/f'{name}_split.json','w') as f: json.dump(split_info,f,indent=2)
        for modelname,fn in [('Fingerprint LR',None),('GNN-MLP',None),('KA-GNN',None)]:
            print(' ',modelname,flush=True)
            if modelname=='Fingerprint LR': metrics,ef,pred=baseline_fp(name,feats,y,ym,tr,val,te); hist=[]; model=None
            else: metrics,ef,hist,model,pred=train_nn(name,'ka' if modelname=='KA-GNN' else 'mlp',feats,y,ym,tr,val,te,EPOCHS[name])
            row={'dataset':name.upper(),'model':modelname,**metrics,'n_train':len(tr),'n_test':len(te)}; all_results.append(row)
            eff.append({'dataset':name.upper(),'model':modelname,**ef})
            histories[f'{name}_{modelname}']=hist
            yt,pt,mtask=pred
            for ti,task in enumerate(tasks):
                ok=mtask[:,ti]>0
                if ok.sum()>2 and len(np.unique(yt[ok,ti]))>1:
                    task_results.append({'dataset':name.upper(),'task':task,'model':modelname,'roc_auc':roc_auc_score(yt[ok,ti],pt[ok,ti]),'avg_precision':average_precision_score(yt[ok,ti],pt[ok,ti]),'n_test_labeled':int(ok.sum()),'n_test_pos':int((yt[ok,ti]==1).sum())})
            if modelname=='KA-GNN':
                predictions[name]={'y':yt.tolist(),'p':pt.tolist(),'mask':mtask.tolist(),'tasks':tasks}
                # gradient saliency for first positive or first test molecule
                try:
                    idx=te[0]
                    for ii in te:
                        if y[ii].sum()>0: idx=ii; break
                    model.eval(); f=feats[idx]; desc=np.stack([f['desc']]); desc=StandardScaler().fit(np.stack([ff['desc'] for ff in feats])[tr]).transform(desc).astype(np.float32)
                    x=torch.tensor(f['x']).unsqueeze(0).requires_grad_(True); adj=torch.tensor(f['adj']).unsqueeze(0); mask=torch.tensor(f['mask']).unsqueeze(0); d=torch.tensor(desc).float()
                    logit=model(x,adj,mask,d)[0,0]; logit.backward(); sal=x.grad.abs().sum(-1).detach().numpy()[0]
                    interprets[name]={'smiles':df.iloc[idx]['smiles'],'tasks':tasks,'atom_saliency':sal[:int(f['n_atoms'])].tolist(),'n_atoms':int(f['n_atoms'])}
                except Exception as e: interprets[name]={'error':repr(e)}
    pd.DataFrame(all_results).to_csv(OUT/'main_results.csv',index=False)
    pd.DataFrame(eff).to_csv(OUT/'efficiency_results.csv',index=False)
    pd.DataFrame(task_results).to_csv(OUT/'task_level_results.csv',index=False)
    with open(OUT/'training_histories.json','w') as f: json.dump(histories,f,indent=2)
    with open(OUT/'predictions.json','w') as f: json.dump(predictions,f)
    with open(OUT/'interpretability.json','w') as f: json.dump(interprets,f,indent=2)
    make_figures()

def make_figures():
    sns.set_theme(style='whitegrid')
    overview=pd.read_csv(OUT/'dataset_overview.csv')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    base=overview.groupby('dataset').agg(n_rows=('n_rows','first'),positive_rate=('positive_rate','mean')).reset_index()
    sns.barplot(data=base,x='dataset',y='n_rows',ax=ax[0]); ax[0].set_yscale('log'); ax[0].set_title('Dataset sizes (log scale)')
    sns.barplot(data=overview,x='dataset',y='positive_rate',ax=ax[1]); ax[1].set_title('Task positive-rate distribution'); ax[1].tick_params(axis='x',rotation=45)
    fig.tight_layout(); fig.savefig(IMG/'data_overview.png',dpi=200); plt.close(fig)
    res=pd.read_csv(OUT/'main_results.csv')
    fig,ax=plt.subplots(figsize=(10,5)); sns.barplot(data=res,x='dataset',y='roc_auc',hue='model',ax=ax); ax.set_ylim(0,1); ax.set_title('Held-out ROC-AUC by dataset and model'); fig.tight_layout(); fig.savefig(IMG/'main_results.png',dpi=200); plt.close(fig)
    eff=pd.read_csv(OUT/'efficiency_results.csv')
    fig,ax=plt.subplots(1,2,figsize=(12,4)); sns.barplot(data=eff,x='dataset',y='params',hue='model',ax=ax[0]); ax[0].set_yscale('log'); ax[0].set_title('Parameter count'); sns.barplot(data=eff,x='dataset',y='train_time_sec',hue='model',ax=ax[1]); ax[1].set_title('Training time'); ax[1].tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig(IMG/'efficiency.png',dpi=200); plt.close(fig)
    # calibration/reliability: bin all KA-GNN predictions
    pred=json.load(open(OUT/'predictions.json'))
    rows=[]
    for ds,v in pred.items():
        y=np.array(v['y']); p=np.array(v['p']); m=np.array(v['mask'])
        ok=m>0
        rows += [{'dataset':ds.upper(),'p':float(pp),'y':float(yy)} for yy,pp in zip(y[ok],p[ok])]
    cal=pd.DataFrame(rows)
    if len(cal):
        cal['bin']=pd.cut(cal['p'],bins=np.linspace(0,1,11),include_lowest=True)
        c=cal.groupby(['dataset','bin'],observed=True).agg(pred=('p','mean'),obs=('y','mean'),n=('y','size')).reset_index()
        c.to_csv(OUT/'calibration_bins.csv',index=False)
        fig,ax=plt.subplots(figsize=(6,5));
        for ds,g in c.groupby('dataset'): ax.plot(g['pred'],g['obs'],marker='o',label=ds)
        ax.plot([0,1],[0,1],'k--',alpha=.5); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed positive rate'); ax.set_title('KA-GNN calibration by dataset'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(IMG/'validation_calibration.png',dpi=200); plt.close(fig)
    interp=json.load(open(OUT/'interpretability.json'))
    fig,axes=plt.subplots(len(interp),1,figsize=(10,2.2*len(interp)),sharex=False)
    if len(interp)==1: axes=[axes]
    for ax,(ds,v) in zip(axes,interp.items()):
        sal=np.array(v.get('atom_saliency',[]));
        if len(sal): ax.bar(range(len(sal)),sal); ax.set_ylabel(ds.upper()); ax.set_title('KA-GNN gradient atom saliency')
        else: ax.text(.1,.5,str(v),transform=ax.transAxes)
    axes[-1].set_xlabel('Atom index in selected molecule'); fig.tight_layout(); fig.savefig(IMG/'interpretability.png',dpi=200); plt.close(fig)

if __name__=='__main__': main()
