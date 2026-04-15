import json, math, os, random, time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'outputs')
IMG_DIR = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

ATOM_LIST = ['C','N','O','S','F','P','Cl','Br','I','B','Si','Se','other']
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot_with_unknown(x, xs):
    out = [0] * (len(xs) + 1)
    try:
        out[xs.index(x)] = 1
    except ValueError:
        out[-1] = 1
    return out


def atom_features(atom):
    symbol = atom.GetSymbol()
    feats = []
    feats += one_hot_with_unknown(symbol, ATOM_LIST[:-1])
    feats += one_hot_with_unknown(atom.GetHybridization(), HYB_LIST)
    feats += [atom.GetAtomicNum() / 100.0,
              atom.GetTotalDegree() / 6.0,
              atom.GetFormalCharge() / 4.0,
              atom.GetTotalNumHs() / 4.0,
              float(atom.GetIsAromatic()),
              atom.GetMass() / 250.0,
              float(atom.IsInRing())]
    return np.array(feats, dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    feats = [1.0 if bt == b else 0.0 for b in BOND_TYPES]
    feats += [float(bond.GetIsConjugated()), float(bond.IsInRing()), float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    return np.array(feats, dtype=np.float32)


def graph_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    adj = np.eye(n, dtype=np.float32)
    edge_attr = np.zeros((n, n, 7), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        adj[i, j] = adj[j, i] = 1.0
        edge_attr[i, j] = edge_attr[j, i] = bf
    # approximate non-covalent interaction edges from 3D distance if conformer can be embedded
    nc_edges = []
    try:
        m3d = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        status = AllChem.EmbedMolecule(m3d, params)
        if status == 0:
            AllChem.UFFOptimizeMolecule(m3d, maxIters=50)
            conf = m3d.GetConformer()
            coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n)])
            dmat = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i,j] == 0 and dmat[i,j] <= 4.5:
                        adj[i,j] = adj[j,i] = 1.0
                        edge_attr[i,j,-1] = edge_attr[j,i,-1] = 1.0
                        nc_edges.append((i,j,float(dmat[i,j])))
    except Exception:
        pass
    deg = adj.sum(1)
    d_inv_sqrt = np.diag(np.power(np.clip(deg, 1e-8, None), -0.5))
    norm_adj = d_inv_sqrt @ adj @ d_inv_sqrt
    return {
        'x': x,
        'adj': norm_adj.astype(np.float32),
        'edge_attr': edge_attr.astype(np.float32),
        'num_nodes': int(n),
        'num_nc_edges': int(len(nc_edges)),
    }


class FourierKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_freq=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_freq = num_freq
        freqs = torch.arange(1, num_freq + 1, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(in_dim * (1 + 2 * num_freq), out_dim)

    def forward(self, x):
        comps = [x]
        for f in self.freqs:
            comps.append(torch.sin(f * x))
            comps.append(torch.cos(f * x))
        z = torch.cat(comps, dim=-1)
        return self.proj(z)


class GraphBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, mode='mlp'):
        super().__init__()
        self.mode = mode
        self.edge_proj = nn.Linear(7, in_dim)
        if mode == 'mlp':
            self.update = nn.Sequential(nn.Linear(in_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        else:
            self.update = nn.Sequential(FourierKANLayer(in_dim * 2, hidden_dim), nn.ReLU(), FourierKANLayer(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, edge_attr):
        edge_msg = self.edge_proj(edge_attr).sum(dim=1)
        agg = adj @ x + edge_msg / max(edge_attr.shape[1], 1)
        h = self.update(torch.cat([x, agg], dim=-1))
        return self.norm(F.relu(h))


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode='mlp'):
        super().__init__()
        self.block1 = GraphBlock(in_dim, hidden_dim, mode=mode)
        self.block2 = GraphBlock(hidden_dim, hidden_dim, mode=mode)
        if mode == 'mlp':
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        else:
            self.head = nn.Sequential(FourierKANLayer(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x, adj, edge_attr):
        h = self.block1(x, adj, edge_attr)
        h = self.block2(h, adj, edge_attr)
        g = h.mean(dim=0)
        return self.head(g), h


@dataclass
class Sample:
    smiles: str
    y: np.ndarray
    mask: np.ndarray
    graph: dict


def build_dataset(name, limit=None):
    path = os.path.join(DATA_DIR, f'{name}.csv')
    df = pd.read_csv(path)
    if name == 'clintox':
        task_cols = ['FDA_APPROVED', 'CT_TOX']
        smiles_col = 'smiles'
    elif name == 'muv':
        task_cols = [c for c in df.columns if c.startswith('MUV-')]
        smiles_col = 'smiles'
    else:
        task_cols = ['label']
        smiles_col = 'smiles'
    if limit is not None:
        df = df.iloc[:limit].copy()
    samples = []
    skipped = 0
    for _, row in df.iterrows():
        smi = row[smiles_col]
        g = graph_from_smiles(smi)
        if g is None or g['num_nodes'] < 1:
            skipped += 1
            continue
        y = row[task_cols].to_numpy(dtype=np.float32)
        mask = ~pd.isna(y)
        y = np.nan_to_num(y, nan=0.0)
        samples.append(Sample(smiles=smi, y=y.astype(np.float32), mask=mask.astype(np.float32), graph=g))
    return samples, task_cols, skipped


def scaffold_split(samples, frac=(0.8,0.1,0.1)):
    groups = {}
    for i, s in enumerate(samples):
        mol = Chem.MolFromSmiles(s.smiles)
        scaf = Chem.MolToSmiles(Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)) if mol is not None else s.smiles[:10]
        groups.setdefault(scaf, []).append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    n = len(samples)
    train_cut, val_cut = frac[0]*n, (frac[0]+frac[1])*n
    train=[]; val=[]; test=[]
    for grp in ordered:
        if len(train)+len(grp) <= train_cut:
            train += grp
        elif len(val)+len(grp) <= val_cut-train_cut:
            val += grp
        else:
            test += grp
    if len(val)==0:
        val=test[:max(1,len(test)//2)]
        test=test[max(1,len(test)//2):]
    return train, val, test


def batch_loss(model, batch, out_dim, pos_weight=None):
    losses=[]
    logits_list=[]; ys=[]; masks=[]
    for s in batch:
        x=torch.tensor(s.graph['x'], device=DEVICE)
        adj=torch.tensor(s.graph['adj'], device=DEVICE)
        ea=torch.tensor(s.graph['edge_attr'], device=DEVICE)
        logits,_=model(x,adj,ea)
        y=torch.tensor(s.y, device=DEVICE)
        mask=torch.tensor(s.mask, device=DEVICE)
        loss_raw=F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_weight)
        loss=(loss_raw*mask).sum()/mask.sum().clamp(min=1.0)
        losses.append(loss)
        logits_list.append(logits.detach().cpu().numpy())
        ys.append(s.y)
        masks.append(s.mask)
    return torch.stack(losses).mean(), np.vstack(logits_list), np.vstack(ys), np.vstack(masks)


def compute_metrics(logits, y, mask):
    probs=1/(1+np.exp(-logits))
    metrics={}
    rocs=[]; prs=[]
    for t in range(y.shape[1]):
        idx=mask[:,t].astype(bool)
        if idx.sum()<2 or len(np.unique(y[idx,t]))<2:
            continue
        yt=y[idx,t]
        pt=probs[idx,t]
        rocs.append(roc_auc_score(yt, pt))
        prs.append(average_precision_score(yt, pt))
    metrics['roc_auc_mean']=float(np.mean(rocs)) if rocs else float('nan')
    metrics['pr_auc_mean']=float(np.mean(prs)) if prs else float('nan')
    metrics['tasks_evaluated']=int(len(rocs))
    return metrics


def evaluate(model, samples, out_dim):
    model.eval()
    with torch.no_grad():
        _, logits, y, mask = batch_loss(model, samples, out_dim)
    return compute_metrics(logits, y, mask), logits, y, mask


def train_model(train_samples, val_samples, in_dim, out_dim, mode='mlp', epochs=8, lr=1e-3, hidden=64):
    model=GraphClassifier(in_dim, hidden, out_dim, mode=mode).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    ys=np.vstack([s.y for s in train_samples])
    ms=np.vstack([s.mask for s in train_samples]).astype(bool)
    pos_weights=[]
    for t in range(out_dim):
        yt=ys[ms[:,t],t]
        pos=max(float((yt==1).sum()),1.0); neg=max(float((yt==0).sum()),1.0)
        pos_weights.append(neg/pos)
    pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=DEVICE)
    best_state=None; best_val=-1e9
    hist=[]
    start=time.time()
    for epoch in range(1, epochs+1):
        model.train()
        rng=np.random.default_rng(SEED+epoch)
        order=rng.permutation(len(train_samples))
        total=0.0
        for idx in order:
            opt.zero_grad()
            loss, _, _, _ = batch_loss(model, [train_samples[idx]], out_dim, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            total += float(loss.item())
        val_metrics, _, _, _ = evaluate(model, val_samples, out_dim)
        score = np.nan_to_num(val_metrics['pr_auc_mean'], nan=-1.0) + np.nan_to_num(val_metrics['roc_auc_mean'], nan=-1.0)
        hist.append({'epoch': epoch, 'train_loss': total/len(train_samples), **val_metrics})
        if score > best_val:
            best_val = score
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    train_time=time.time()-start
    model.load_state_dict(best_state)
    param_count=sum(p.numel() for p in model.parameters())
    return model, hist, train_time, param_count


def permutation_importance(model, samples, feature_names, out_dim, n_repeats=1):
    base_metrics, _, _, _ = evaluate(model, samples, out_dim)
    base = np.nan_to_num(base_metrics['pr_auc_mean'], nan=0.0)
    drops=[]
    for fi in range(len(feature_names)):
        vals=[]
        for _ in range(n_repeats):
            perturbed=[]
            for s in samples:
                g={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in s.graph.items()}
                col=g['x'][:,fi].copy()
                rng=np.random.default_rng(SEED+fi)
                rng.shuffle(col)
                g['x'][:,fi]=col
                perturbed.append(Sample(smiles=s.smiles,y=s.y,mask=s.mask,graph=g))
            m,_,_,_=evaluate(model, perturbed, out_dim)
            vals.append(base - np.nan_to_num(m['pr_auc_mean'], nan=0.0))
        drops.append(float(np.mean(vals)))
    return {'base_pr_auc': float(base), 'feature_importance_drop': drops, 'feature_names': feature_names}


def atom_saliency(model, sample):
    model.eval()
    x=torch.tensor(sample.graph['x'], device=DEVICE, requires_grad=True)
    adj=torch.tensor(sample.graph['adj'], device=DEVICE)
    ea=torch.tensor(sample.graph['edge_attr'], device=DEVICE)
    logits,h=model(x,adj,ea)
    target=logits.max()
    target.backward()
    sal=x.grad.abs().sum(dim=1).detach().cpu().numpy()
    return sal.tolist()


def plot_dataset_overview(summary_df):
    plt.figure(figsize=(12,5))
    sns.barplot(data=summary_df, x='task', y='positive_rate', hue='dataset')
    plt.xticks(rotation=60, ha='right')
    plt.ylabel('Positive class rate')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'dataset_imbalance_overview.png'), dpi=200)
    plt.close()


def plot_performance(results_df):
    plt.figure(figsize=(10,5))
    melted=results_df.melt(id_vars=['dataset','model'], value_vars=['roc_auc_mean','pr_auc_mean'], var_name='metric', value_name='value')
    sns.catplot(data=melted, x='dataset', y='value', hue='model', col='metric', kind='bar', height=4, aspect=1.3)
    plt.savefig(os.path.join(IMG_DIR,'model_performance_comparison.png'), dpi=200)
    plt.close('all')


def plot_efficiency(results_df):
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=results_df, x='train_time_sec', y='pr_auc_mean', hue='model', style='dataset', s=90)
    for _,r in results_df.iterrows():
        plt.text(r['train_time_sec'], r['pr_auc_mean'], r['dataset'], fontsize=8)
    plt.xlabel('Training time (s)')
    plt.ylabel('PR-AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'efficiency_tradeoff.png'), dpi=200)
    plt.close()


def plot_feature_importance(imp_df):
    plt.figure(figsize=(12,5))
    top=imp_df.sort_values('importance', ascending=False).head(15)
    sns.barplot(data=top, x='importance', y='feature')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'permutation_importance.png'), dpi=200)
    plt.close()


def plot_atom_saliency(saliency):
    plt.figure(figsize=(8,3))
    sns.barplot(x=list(range(len(saliency))), y=saliency, color='steelblue')
    plt.xlabel('Atom index')
    plt.ylabel('Gradient saliency')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'atom_saliency_example.png'), dpi=200)
    plt.close()


def main():
    dataset_limits={'bace':700,'bbbp':900,'clintox':900,'hiv':2000,'muv':2200}
    summary_rows=[]
    result_rows=[]
    histories={}
    interpretability_saved=False
    feat_names=[f'atom_{a}' for a in ATOM_LIST] + [f'hyb_{i}' for i in range(len(HYB_LIST)+1)] + ['atomic_num','degree','formal_charge','num_h','aromatic','mass','ring']
    all_metrics={}
    for name,limit in dataset_limits.items():
        samples, tasks, skipped = build_dataset(name, limit=limit)
        for ti,t in enumerate(tasks):
            yy=np.array([s.y[ti] for s in samples if s.mask[ti]==1])
            summary_rows.append({'dataset':name,'task':t,'n_labeled':int(len(yy)),'positive_rate':float(yy.mean()) if len(yy)>0 else np.nan,'skipped':skipped})
        train_idx,val_idx,test_idx=scaffold_split(samples)
        train=[samples[i] for i in train_idx]
        val=[samples[i] for i in val_idx]
        test=[samples[i] for i in test_idx]
        in_dim=train[0].graph['x'].shape[1]
        out_dim=len(tasks)
        for mode in ['mlp','kagnn']:
            model,hist,tt,params=train_model(train,val,in_dim,out_dim,mode='mlp' if mode=='mlp' else 'kan',epochs=6 if name!='muv' else 4,hidden=48)
            metrics, logits, y, mask = evaluate(model,test,out_dim)
            row={'dataset':name,'model':mode,'train_time_sec':tt,'params':params,**metrics,'n_train':len(train),'n_test':len(test),'tasks':out_dim}
            result_rows.append(row)
            histories[f'{name}_{mode}']=hist
            all_metrics[f'{name}_{mode}']=row
            if name=='bace' and mode=='kagnn' and not interpretability_saved:
                imp=permutation_importance(model, test[:min(len(test),80)], feat_names, out_dim, n_repeats=1)
                imp_df=pd.DataFrame({'feature':imp['feature_names'],'importance':imp['feature_importance_drop']})
                imp_df.to_csv(os.path.join(OUT_DIR,'permutation_importance_bace.csv'), index=False)
                plot_feature_importance(imp_df)
                sal=atom_saliency(model, test[0])
                with open(os.path.join(OUT_DIR,'atom_saliency_bace_example.json'),'w') as f:
                    json.dump({'smiles':test[0].smiles,'saliency':sal}, f, indent=2)
                plot_atom_saliency(sal)
                interpretability_saved=True
    summary_df=pd.DataFrame(summary_rows)
    results_df=pd.DataFrame(result_rows)
    summary_df.to_csv(os.path.join(OUT_DIR,'dataset_summary.csv'), index=False)
    results_df.to_csv(os.path.join(OUT_DIR,'benchmark_results.csv'), index=False)
    with open(os.path.join(OUT_DIR,'training_histories.json'),'w') as f:
        json.dump(histories,f,indent=2)
    with open(os.path.join(OUT_DIR,'metrics_summary.json'),'w') as f:
        json.dump(all_metrics,f,indent=2)
    plot_dataset_overview(summary_df)
    plot_performance(results_df)
    plot_efficiency(results_df)
    claim_rows=[]
    for ds in results_df['dataset'].unique():
        sub=results_df[results_df.dataset==ds].set_index('model')
        if {'mlp','kagnn'}.issubset(set(sub.index)):
            claim_rows.append({
                'dataset': ds,
                'claim': 'KA-GNN vs baseline PR-AUC difference',
                'baseline_pr_auc': float(sub.loc['mlp','pr_auc_mean']),
                'kagnn_pr_auc': float(sub.loc['kagnn','pr_auc_mean']),
                'difference': float(sub.loc['kagnn','pr_auc_mean'] - sub.loc['mlp','pr_auc_mean'])
            })
    pd.DataFrame(claim_rows).to_csv(os.path.join(OUT_DIR,'claim_recovery_table.csv'), index=False)

if __name__ == '__main__':
    from rdkit.Chem.Scaffolds import MurckoScaffold
    Chem.Scaffolds = type('Scaffolds', (), {'MurckoScaffold': MurckoScaffold})
    main()
