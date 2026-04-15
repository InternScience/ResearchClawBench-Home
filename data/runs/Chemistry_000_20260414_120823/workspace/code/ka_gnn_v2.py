"""
KA-GNN v2: Kolmogorov-Arnold Graph Neural Networks
Proper KAN implementation with learnable B-spline activations
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from rdkit import Chem
import rdkit.RDLogger as rl; rl.DisableLog('rdApp.*')
import warnings; warnings.filterwarnings('ignore')

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
    return one_hot(a.GetAtomicNum(),ATOM_NUMS)+one_hot(a.GetTotalDegree(),DEGREES)+one_hot(a.GetFormalCharge(),CHARGES)+one_hot(a.GetTotalNumHs(),NUM_HS)+one_hot(a.GetHybridization(),HYBS)+one_hot(int(a.GetIsAromatic()),[0,1])+one_hot(int(a.IsInRing()),[0,1])+[a.GetMass()/200.0]

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

# ---- KAN Layer (B-spline + Fourier) ----
class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network layer with learnable univariate activations.
    Each input-output pair has a learnable univariate function parameterized
    by B-spline basis functions plus Fourier components.
    """
    def __init__(self, in_f, out_f, num_splines=8, spline_order=3):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.num_splines = num_splines
        self.spline_order = spline_order
        
        # Grid points for B-spline (shared across all inputs)
        grid = torch.linspace(-1.5, 1.5, num_splines)
        self.register_buffer('grid', grid)
        
        # Learnable spline coefficients: (out_f, in_f, num_splines + spline_order)
        total_basis = num_splines + spline_order
        self.spline_coef = nn.Parameter(torch.randn(out_f, in_f, total_basis) * 0.02)
        
        # Base linear weight (residual connection for stability)
        self.base_weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        
        # Fourier enhancement
        self.fourier_coef = nn.Parameter(torch.randn(out_f, in_f, 3) * 0.02)
        
        self.bias = nn.Parameter(torch.zeros(out_f))
    
    def b_spline_eval(self, x):
        """Evaluate B-spline basis at points x using Cox-de Boor recursion."""
        # x: (batch, in_f)
        batch_size = x.shape[0]
        grid = self.grid  # (num_splines,)
        n = len(grid)
        
        # Degree-0 basis: indicator functions
        # Create knots by extending grid
        knots = torch.cat([grid[:1] - (grid[1]-grid[0])] * self.spline_order + 
                         [grid] + 
                         [grid[-1:] + (grid[-1]-grid[-2])] * self.spline_order)
        
        # Evaluate using Cox-de Boor
        x_exp = x.unsqueeze(-1)  # (batch, in_f, 1)
        
        # Initialize: degree 0 basis functions
        num_knots = len(knots)
        B = []
        for i in range(num_knots - 1):
            left, right = knots[i], knots[i+1]
            if i == num_knots - 2:
                bi = ((x_exp >= left) & (x_exp <= right)).float()
            else:
                bi = ((x_exp >= left) & (x_exp < right)).float()
            B.append(bi)
        
        # Recursive higher-order basis
        for d in range(1, self.spline_order + 1):
            B_new = []
            for i in range(num_knots - d - 1):
                left = knots[i]
                mid = knots[i + d]
                right = knots[i + d + 1]
                
                denom1 = mid - left
                denom2 = right - knots[i + 1]
                
                t1 = (x_exp - left) / (denom1 + 1e-8) * B[i] if denom1 > 1e-8 else torch.zeros_like(B[i])
                t2 = (right - x_exp) / (denom2 + 1e-8) * B[i+1] if denom2 > 1e-8 else torch.zeros_like(B[i+1])
                B_new.append(t1 + t2)
            B = B_new
        
        # Stack basis: (batch, in_f, num_basis)
        B = torch.cat(B, dim=-1)
        return B
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.in_f)  # (batch, in_f)
        
        # Base linear transformation with activation
        base = F.linear(F.silu(x), self.base_weight)  # (batch, out_f)
        
        # B-spline transformation
        B = self.b_spline_eval(x)  # (batch, in_f, num_basis)
        # Trim to match spline_coef dimensions
        nb = min(B.shape[-1], self.spline_coef.shape[-1])
        spline_out = torch.einsum('bik,oik->bo', B[:,:,:nb], self.spline_coef[:,:,:nb])
        
        # Fourier enhancement
        fourier = torch.zeros(x.shape[0], self.out_f, device=x.device)
        for k in range(self.fourier_coef.shape[-1]):
            freq = k + 1
            sin_feat = torch.sin(freq * np.pi * x)
            fourier = fourier + F.linear(sin_feat, self.fourier_coef[:,:,k])
        
        out = base + spline_out + 0.1 * fourier + self.bias
        return out.view(list(orig_shape[:-1]) + [self.out_f])

# ---- GNN Models ----
class KAGNNConv(nn.Module):
    def __init__(self, ic, oc, ed=None):
        super().__init__()
        self.node_kan = KANLinear(ic, oc)
        self.edge_kan = KANLinear(ed, oc) if ed else None
        self.combine_kan = KANLinear(oc * 2, oc)
        self.norm = nn.LayerNorm(oc)
    
    def forward(self, x, ei, ea=None):
        r, c = ei
        # Transform node features
        h = self.node_kan(x)  # (N, oc)
        # Messages from neighbors
        msgs = h[c]  # (E, oc)
        if ea is not None and self.edge_kan is not None:
            msgs = msgs + self.edge_kan(ea)
        # Aggregate
        agg = torch.zeros_like(h)
        agg.index_add_(0, r, msgs)
        # Normalize
        from torch_geometric.utils import degree
        d = degree(r, x.size(0), dtype=x.dtype).clamp(min=1)
        agg = agg / d.unsqueeze(-1)
        # Combine self + aggregated
        combined = torch.cat([h, agg], dim=-1)
        out = self.combine_kan(combined)
        return self.norm(out)

class KAGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([KAGNNConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, d):
        x, ei, ea, b = d.x, d.edge_index, d.edge_attr, d.batch
        x = F.relu(self.inp(x))
        for cv, nm in zip(self.convs, self.norms):
            x = nm(cv(x, ei, ea) + x)  # residual
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.out_layers(global_mean_pool(x, b))

class GCNBaseline(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    
    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = F.relu(self.inp(x))
        for cv, nm in zip(self.convs, self.norms):
            x = nm(cv(x, ei) + x)
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.out(global_mean_pool(x, b))

class GATBaseline(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1, dropout=0.2, heads=4, **kw):
        super().__init__()
        self.drop = dropout
        self.inp = nn.Linear(node_dim, hidden_dim)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            h = 1 if i == num_layers - 1 else heads
            oc = hidden_dim if h == 1 else hidden_dim // heads
            self.convs.append(GATConv(hidden_dim, oc, heads=h))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    
    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = F.relu(self.inp(x))
        for cv, nm in zip(self.convs, self.norms):
            x = nm(cv(x, ei) + x)
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.out(global_mean_pool(x, b))

# ---- Data Loading ----
def load_dataset(path, name):
    df = pd.read_csv(path)
    cfg = {'bace':('smiles',['label']), 'bbbp':('smiles',['label']),
           'clintox':('smiles',['FDA_APPROVED','CT_TOX']),
           'hiv':('smiles',['label']),
           'muv':('smiles',[c for c in df.columns if c.startswith('MUV')])}
    sc, lc = cfg[name]
    mx = {'hiv': 4000, 'muv': 3000}
    if name in mx and len(df) > mx[name]:
        if len(lc) == 1:
            pos = df[df[lc[0]] == 1]; neg = df[df[lc[0]] == 0]
            np2 = min(len(pos), mx[name] // 4); nn2 = mx[name] - np2
            df = pd.concat([pos.sample(n=min(len(pos), np2), random_state=42),
                            neg.sample(n=min(len(neg), nn2), random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(n=mx[name], random_state=42).reset_index(drop=True)
    graphs = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row[sc])
        if g is None: continue
        labels = [float('nan') if pd.isna(row[l]) else float(row[l]) for l in lc]
        g.y = torch.tensor([labels], dtype=torch.float)
        graphs.append(g)
    print(f"  {name}: {len(graphs)} valid graphs")
    return graphs, lc

# ---- Training ----
def train_epoch(model, loader, opt, crit, dev):
    model.train(); tot = 0; n = 0
    for b in loader:
        b = b.to(dev); opt.zero_grad()
        out = model(b); tgt = b.y.view_as(out); mask = ~torch.isnan(tgt)
        if mask.sum() == 0: continue
        loss = crit(out[mask], tgt[mask]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * mask.sum().item(); n += mask.sum().item()
    return tot / max(n, 1)

@torch.no_grad()
def eval_model(model, loader, dev, nt=1):
    model.eval(); ps, ts = [], []
    for b in loader:
        b = b.to(dev); out = model(b); tgt = b.y.view_as(out); mask = ~torch.isnan(tgt)
        if mask.sum() == 0: continue
        ps.append(out[mask].cpu()); ts.append(tgt[mask].cpu())
    if not ps: return 0.5
    ps = torch.cat(ps).numpy(); ts = torch.cat(ts).numpy()
    try:
        if nt == 1: return roc_auc_score(ts, ps) if len(np.unique(ts)) >= 2 else 0.5
        aucs = []
        for t in range(nt):
            yt = ts[:, t] if ts.ndim > 1 else ts; yp = ps[:, t] if ps.ndim > 1 else ps
            m = ~np.isnan(yt)
            if m.sum() > 0 and len(np.unique(yt[m])) >= 2: aucs.append(roc_auc_score(yt[m], yp[m]))
        return np.mean(aucs) if aucs else 0.5
    except: return 0.5

def run_experiment(ds, path, mc, mkw, n_epochs=40, lr=5e-4, bs=64, n_folds=3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, lc = load_dataset(path, ds); nt = len(lc)
    nd = graphs[0].x.shape[1]; ed = graphs[0].edge_attr.shape[1]
    labels = [g.y[0, 0].item() for g in graphs]
    vm = ~np.isnan(labels); vi = np.where(vm)[0]; vl = np.array(labels)[vm]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    faucs = []
    for fold, (tr, te) in enumerate(skf.split(vi, vl)):
        tl = DataLoader([graphs[vi[i]] for i in tr], batch_size=bs, shuffle=True)
        vl2 = DataLoader([graphs[vi[i]] for i in te], batch_size=bs)
        model = mc(node_dim=nd, edge_dim=ed, num_classes=nt, **mkw).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit = nn.BCEWithLogitsLoss()
        best = 0; pc = 0
        for ep in range(n_epochs):
            train_epoch(model, tl, opt, crit, dev); sch.step()
            if (ep + 1) % 5 == 0:
                a = eval_model(model, vl2, dev, nt)
                if a > best: best = a; pc = 0
                else: pc += 1
                if pc >= 8: break
        fa = eval_model(model, vl2, dev, nt); faucs.append(fa)
    r = {'dataset': ds, 'model_name': mc.__name__, 'fold_aucs': [float(a) for a in faucs],
         'mean_auc': float(np.mean(faucs)), 'std_auc': float(np.std(faucs)),
         'num_tasks': nt, 'num_samples': len(graphs)}
    print(f"    {mc.__name__}: {r['mean_auc']:.4f} +/- {r['std_auc']:.4f}")
    return r

# ---- Main ----
def main():
    datasets = {'bace': 'data/bace.csv', 'bbbp': 'data/bbbp.csv', 'clintox': 'data/clintox.csv',
                'hiv': 'data/hiv.csv', 'muv': 'data/muv.csv'}
    models = {'KA-GNN': (KAGNN, {'hidden_dim': 64, 'num_layers': 3}),
              'GCN': (GCNBaseline, {'hidden_dim': 64, 'num_layers': 3}),
              'GAT': (GATBaseline, {'hidden_dim': 64, 'num_layers': 3})}
    all_results = []
    for ds, dp in datasets.items():
        print(f"\n{'='*50}\nDataset: {ds}")
        for mn, (mc, mk) in models.items():
            try:
                r = run_experiment(ds, dp, mc, dict(mk), n_epochs=40, lr=5e-4, bs=64, n_folds=3)
                r['model_name'] = mn; all_results.append(r)
            except Exception as e:
                print(f"ERROR {ds}/{mn}: {e}"); import traceback; traceback.print_exc()
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/experiment_results.json', 'w') as f: json.dump(all_results, f, indent=2)
    print("\nResults saved.")
    return all_results

if __name__ == '__main__': main()
