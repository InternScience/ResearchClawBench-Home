"""
KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction
Fast version with simplified KAN layer
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from rdkit import Chem
import warnings; warnings.filterwarnings('ignore')
import rdkit.RDLogger as rl; rl.DisableLog('rdApp.*')

# ---- Featurization ----
def one_hot(v, choices):
    e = [0]*(len(choices)+1); e[choices.index(v) if v in choices else len(choices)] = 1; return e

ATOM_NUMS = list(range(1,119))
DEGREES = [0,1,2,3,4,5]
CHARGES = [-2,-1,0,1,2]
NUM_HS = [0,1,2,3,4]
HYBS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
STEREOS = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
           Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]

def atom_features(a):
    f = one_hot(a.GetAtomicNum(),ATOM_NUMS)+one_hot(a.GetTotalDegree(),DEGREES)+one_hot(a.GetFormalCharge(),CHARGES)+one_hot(a.GetTotalNumHs(),NUM_HS)+one_hot(a.GetHybridization(),HYBS)+one_hot(int(a.GetIsAromatic()),[0,1])+one_hot(int(a.IsInRing()),[0,1])+[a.GetMass()/200.0]
    return f

def bond_features(b):
    return one_hot(b.GetBondType(),BOND_TYPES)+one_hot(int(b.GetIsConjugated()),[0,1])+one_hot(int(b.IsInRing()),[0,1])+one_hot(b.GetStereo(),STEREOS)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ef = [], []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        ei.extend([[i,j],[j,i]]); ef.extend([bf,bf])
    if not ei:
        ei = [[k,k] for k in range(mol.GetNumAtoms())]
        ef = [[0]*16 for _ in range(mol.GetNumAtoms())]
    return Data(x=x, edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ef,dtype=torch.float))

# ---- KAN Layer (Fourier-based, efficient) ----
class KANLinear(nn.Module):
    def __init__(self, in_f, out_f, num_basis=5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.base_w = nn.Parameter(torch.randn(out_f, in_f)*0.1)
        # Learnable Fourier coefficients: sin(k*pi*x) for k=1..num_basis
        self.fourier_w = nn.Parameter(torch.randn(out_f, in_f, num_basis)*0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x):
        sh = x.shape; x = x.view(-1, self.in_f)
        base = F.linear(F.silu(x), self.base_w)
        fourier = sum(F.linear(torch.sin((k+1)*np.pi*x), self.fourier_w[:,:,k]) for k in range(self.fourier_w.shape[-1]))
        return (base + fourier + self.bias).view(list(sh[:-1])+[self.out_f])

# ---- GNN Models ----
class KAGNNConv(nn.Module):
    def __init__(self, ic, oc, ed=None):
        super().__init__()
        self.nk = KANLinear(ic, oc)
        self.ek = KANLinear(ed, oc) if ed else None
        self.ck = KANLinear(oc, oc)
        self.norm = nn.LayerNorm(oc)
    def forward(self, x, ei, ea=None):
        r,c = ei; xt = self.nk(x); m = xt[c]
        if ea is not None and self.ek: m = m + self.ek(ea)
        agg = torch.zeros_like(xt); agg.index_add_(0,r,m)
        from torch_geometric.utils import degree
        d = degree(r,x.size(0),dtype=x.dtype).clamp(min=1)
        return self.norm(self.ck(agg/d.unsqueeze(-1) + xt))

class KAGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = KANLinear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([KAGNNConv(hidden_dim,hidden_dim,edge_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.o1 = KANLinear(hidden_dim, hidden_dim)
        self.o2 = KANLinear(hidden_dim, num_classes)
    def forward(self, d):
        x,ei,ea,b = d.x,d.edge_index,d.edge_attr,d.batch
        x = self.inp(x)
        for cv,nm in zip(self.convs,self.norms):
            x = nm(cv(x,ei,ea)+x); x = F.dropout(x,p=self.drop,training=self.training)
        x = global_mean_pool(x,b)
        return self.o2(F.dropout(F.silu(self.o1(x)),p=self.drop,training=self.training))

class GCNBaseline(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim,hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))
    def forward(self, d):
        x,ei,b = d.x,d.edge_index,d.batch
        x = F.relu(self.inp(x))
        for cv,nm in zip(self.convs,self.norms):
            x = nm(cv(x,ei)+x); x = F.dropout(x,p=self.drop,training=self.training)
        return self.out(global_mean_pool(x,b))

class GATBaseline(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, heads=4, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = nn.Linear(node_dim, hidden_dim)
        self.convs,self.norms = nn.ModuleList(),nn.ModuleList()
        for i in range(num_layers):
            h=1 if i==num_layers-1 else heads; oc=hidden_dim if h==1 else hidden_dim//heads
            self.convs.append(GATConv(hidden_dim,oc,heads=h)); self.norms.append(nn.LayerNorm(hidden_dim))
        self.out = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))
    def forward(self, d):
        x,ei,b = d.x,d.edge_index,d.batch
        x = F.relu(self.inp(x))
        for cv,nm in zip(self.convs,self.norms):
            x = nm(cv(x,ei)+x); x = F.dropout(x,p=self.drop,training=self.training)
        return self.out(global_mean_pool(x,b))

# ---- Data Loading ----
def load_dataset(path, name):
    df = pd.read_csv(path)
    cfg = {'bace':('smiles',['label']),'bbbp':('smiles',['label']),
           'clintox':('smiles',['FDA_APPROVED','CT_TOX']),
           'hiv':('smiles',['label']),
           'muv':('smiles',[c for c in df.columns if c.startswith('MUV')])}
    sc,lc = cfg[name]
    mx = {'hiv':4000,'muv':3000}
    if name in mx and len(df)>mx[name]:
        if len(lc)==1:
            pos=df[df[lc[0]]==1]; neg=df[df[lc[0]]==0]
            np2=min(len(pos),mx[name]//4); nn2=mx[name]-np2
            df=pd.concat([pos.sample(n=min(len(pos),np2),random_state=42),
                          neg.sample(n=min(len(neg),nn2),random_state=42)]).sample(frac=1,random_state=42).reset_index(drop=True)
        else:
            df=df.sample(n=mx[name],random_state=42).reset_index(drop=True)
    graphs=[]
    for _,row in df.iterrows():
        g=smiles_to_graph(row[sc])
        if g is None: continue
        labels=[float('nan') if pd.isna(row[l]) else float(row[l]) for l in lc]
        g.y=torch.tensor([labels],dtype=torch.float); graphs.append(g)
    print(f"  {name}: {len(graphs)} valid graphs"); return graphs,lc

# ---- Training ----
def train_epoch(model,loader,opt,crit,dev):
    model.train(); tot=0; n=0
    for b in loader:
        b=b.to(dev); opt.zero_grad()
        out=model(b); tgt=b.y.view_as(out); mask=~torch.isnan(tgt)
        if mask.sum()==0: continue
        loss=crit(out[mask],tgt[mask]); loss.backward(); opt.step()
        tot+=loss.item()*mask.sum().item(); n+=mask.sum().item()
    return tot/max(n,1)

@torch.no_grad()
def eval_model(model,loader,dev,nt=1):
    model.eval(); ps,ts=[],[]
    for b in loader:
        b=b.to(dev); out=model(b); tgt=b.y.view_as(out); mask=~torch.isnan(tgt)
        if mask.sum()==0: continue
        ps.append(out[mask].cpu()); ts.append(tgt[mask].cpu())
    if not ps: return 0.5
    ps=torch.cat(ps).numpy(); ts=torch.cat(ts).numpy()
    try:
        if nt==1: return roc_auc_score(ts,ps) if len(np.unique(ts))>=2 else 0.5
        aucs=[]
        for t in range(nt):
            yt=ts[:,t] if ts.ndim>1 else ts; yp=ps[:,t] if ps.ndim>1 else ps
            m=~np.isnan(yt)
            if m.sum()>0 and len(np.unique(yt[m]))>=2: aucs.append(roc_auc_score(yt[m],yp[m]))
        return np.mean(aucs) if aucs else 0.5
    except: return 0.5

def run_experiment(ds,path,mc,mkw,n_epochs=30,lr=1e-3,bs=64,n_folds=3,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  {ds}/{mc.__name__} dev={dev}")
    graphs,lc=load_dataset(path,ds); nt=len(lc)
    nd=graphs[0].x.shape[1]; ed=graphs[0].edge_attr.shape[1]
    labels=[g.y[0,0].item() for g in graphs]
    vm=~np.isnan(labels); vi=np.where(vm)[0]; vl=np.array(labels)[vm]
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)
    faucs=[]
    for fold,(tr,te) in enumerate(skf.split(vi,vl)):
        tl=DataLoader([graphs[vi[i]] for i in tr],batch_size=bs,shuffle=True)
        vl2=DataLoader([graphs[vi[i]] for i in te],batch_size=bs)
        model=mc(node_dim=nd,edge_dim=ed,num_classes=nt,**mkw).to(dev)
        opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
        crit=nn.BCEWithLogitsLoss()
        best=0; pc=0
        for ep in range(n_epochs):
            train_epoch(model,tl,opt,crit,dev)
            if (ep+1)%5==0:
                a=eval_model(model,vl2,dev,nt)
                if a>best: best=a; pc=0
                else: pc+=1
                if pc>=6: break
        fa=eval_model(model,vl2,dev,nt); faucs.append(fa)
    r={'dataset':ds,'model_name':mc.__name__,'fold_aucs':[float(a) for a in faucs],
       'mean_auc':float(np.mean(faucs)),'std_auc':float(np.std(faucs)),'num_tasks':nt,'num_samples':len(graphs)}
    print(f"    {mc.__name__}: {r['mean_auc']:.4f}+/-{r['std_auc']:.4f}"); return r

# ---- Main ----
def main():
    datasets={'bace':'data/bace.csv','bbbp':'data/bbbp.csv','clintox':'data/clintox.csv',
              'hiv':'data/hiv.csv','muv':'data/muv.csv'}
    models={'KA-GNN':(KAGNN,{'hidden_dim':64,'num_layers':3}),
            'GCN':(GCNBaseline,{'hidden_dim':64,'num_layers':3}),
            'GAT':(GATBaseline,{'hidden_dim':64,'num_layers':3})}
    all_results=[]
    for ds,dp in datasets.items():
        print(f"\n{'='*50}\nDataset: {ds}")
        for mn,(mc,mk) in models.items():
            try:
                r=run_experiment(ds,dp,mc,dict(mk),n_epochs=30,lr=1e-3,bs=64,n_folds=3)
                r['model_name']=mn; all_results.append(r)
            except Exception as e:
                print(f"ERROR {ds}/{mn}: {e}"); import traceback; traceback.print_exc()
    os.makedirs('outputs',exist_ok=True)
    with open('outputs/experiment_results.json','w') as f: json.dump(all_results,f,indent=2)
    print("\nResults saved."); return all_results

if __name__=='__main__': main()
