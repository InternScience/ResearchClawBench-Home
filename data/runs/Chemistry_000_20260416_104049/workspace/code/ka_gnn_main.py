"""
KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction
Optimized: smaller hidden dim, fewer harmonics, subsample HIV/MUV for speed
"""

import os, json, warnings, random, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv as PyGGCNConv, GATConv as PyGGATConv
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

SEED = 42; NUM_RUNS = 3; EPOCHS = 30; BATCH_SIZE = 128; LR = 0.005
HIDDEN_DIM = 64; NUM_LAYERS = 2; KAN_HARMONICS = 4
DROPOUT = 0.2; PATIENCE = 8; DEVICE = torch.device('cpu')
MAX_SAMPLES = 5000  # subsample large datasets

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ROOT = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_000_20260416_104049'
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'outputs')
IMG_DIR = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(IMG_DIR, exist_ok=True)

ATOM_TYPES = ['C','N','O','S','F','P','Cl','Br','I','Si','other']
DEGREES = [0,1,2,3,4,5]; CHARGES = [-1,0,1]; NUM_HS = [0,1,2,3]
HYBRID = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
          Chem.rdchem.HybridizationType.SP3, 'other']

def one_hot(val, choices):
    enc = [0]*len(choices)
    try: enc[choices.index(val)] = 1
    except ValueError:
        if 'other' in choices: enc[choices.index('other')] = 1
    return enc

def atom_features(atom):
    f = one_hot(atom.GetSymbol(), ATOM_TYPES) + one_hot(atom.GetDegree(), DEGREES)
    f += one_hot(atom.GetFormalCharge(), CHARGES) + one_hot(atom.GetTotalNumHs(), NUM_HS)
    hyb = atom.GetHybridization()
    if hyb not in HYBRID: hyb = 'other'
    f += one_hot(hyb, HYBRID) + [int(atom.GetIsAromatic())] + [int(atom.IsInRing())]
    return f

ATOM_DIM = len(ATOM_TYPES)+len(DEGREES)+len(CHARGES)+len(NUM_HS)+len(HYBRID)+2

def smiles_to_data(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atoms = [atom_features(a) for a in mol.GetAtoms()]
    if not atoms: return None
    x = torch.tensor(atoms, dtype=torch.float)
    ei = []
    for i in range(len(atoms)): ei.append([i,i])
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei.append([i,j]); ei.append([j,i])
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([float(label)], dtype=torch.float))

def load_dataset(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv'))
    col_map = {'bace':('smiles','label'), 'bbbp':('smiles','label'),
               'clintox':('smiles','CT_TOX'), 'hiv':('smiles','label'),
               'muv':('smiles','MUV-466')}
    smi_col, lbl_col = col_map[name]
    graphs=[]; labels=[]
    for _, row in df.iterrows():
        lbl = row[lbl_col]
        if pd.isna(lbl): continue
        g = smiles_to_data(row[smi_col], int(lbl))
        if g is not None: graphs.append(g); labels.append(int(lbl))
    
    # Subsample large datasets maintaining class balance
    if len(graphs) > MAX_SAMPLES:
        pos_idx = [i for i,l in enumerate(labels) if l==1]
        neg_idx = [i for i,l in enumerate(labels) if l==0]
        # Keep all positives, sample negatives
        keep_neg = min(len(neg_idx), MAX_SAMPLES - len(pos_idx))
        if keep_neg < len(neg_idx):
            random.shuffle(neg_idx)
            neg_idx = neg_idx[:keep_neg]
        keep_idx = sorted(pos_idx + neg_idx)
        graphs = [graphs[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
    
    print(f"{name}: {len(graphs)} valid, labels={dict(zip(*np.unique(labels, return_counts=True)))}")
    return graphs, labels

def scaffold_split(graphs, labels):
    n = len(graphs); idx = list(range(n))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tv_idx, test_idx = list(skf.split(idx, labels))[0]
    tv_labels = [labels[i] for i in tv_idx]
    skf2 = StratifiedKFold(n_splits=8, shuffle=True, random_state=SEED)
    tr_rel, val_rel = list(skf2.split(tv_idx, tv_labels))[0]
    train_idx = [tv_idx[i] for i in tr_rel]
    val_idx = [tv_idx[i] for i in val_rel]
    return train_idx, val_idx, test_idx

class FourierKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_harmonics=4):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.num_harmonics = num_harmonics
        self.fourier_coeffs = nn.Parameter(
            torch.randn(in_dim, out_dim, num_harmonics, 2) * (1.0/np.sqrt(in_dim*num_harmonics)))
        self.residual_weight = nn.Parameter(torch.randn(in_dim, out_dim)*0.01)
        self.residual_bias = nn.Parameter(torch.zeros(out_dim))
        freqs = torch.arange(1, num_harmonics+1, dtype=torch.float) * 2.0
        self.base_freq = nn.Parameter(freqs, requires_grad=False)

    def forward(self, x):
        angles = x.unsqueeze(-1) * self.base_freq.unsqueeze(0).unsqueeze(0)
        cos_v = torch.cos(angles); sin_v = torch.sin(angles)
        basis = torch.stack([cos_v, sin_v], dim=-1)
        result = torch.einsum('nihz,iohz->no', basis, self.fourier_coeffs)
        return result + x @ self.residual_weight + self.residual_bias

class KA_GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_harmonics=4):
        super().__init__()
        self.kan = FourierKANLayer(in_dim, out_dim, num_harmonics)

    def forward(self, x, edge_index):
        h = self.kan(x)
        row, col = edge_index
        deg = torch.zeros(x.shape[0], device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.shape[0], device=x.device))
        deg_inv_sqrt = deg.pow(-0.5); deg_inv_sqrt[deg_inv_sqrt==float('inf')] = 0
        norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]
        msgs = h[col]*norm.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msgs), msgs)
        return out

class KA_GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, num_harmonics=4):
        super().__init__()
        self.heads = heads; self.out_per_head = out_dim // heads
        assert out_dim % heads == 0
        self.kan_src = FourierKANLayer(in_dim, out_dim, num_harmonics)
        self.kan_dst = FourierKANLayer(in_dim, out_dim, num_harmonics)
        self.attn_w = nn.Parameter(torch.randn(heads, 2*self.out_per_head)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        N = x.shape[0]; row, col = edge_index
        h_s = self.kan_src(x).view(N, self.heads, self.out_per_head)
        h_d = self.kan_dst(x).view(N, self.heads, self.out_per_head)
        attn_in = torch.cat([h_s[col], h_d[row]], dim=-1)
        scores = torch.einsum('ehd,hd->eh', attn_in, self.attn_w)
        scores = F.leaky_relu(scores, 0.2)
        max_s = torch.zeros(N, self.heads, device=x.device)
        max_s.scatter_reduce_(0, row.unsqueeze(-1).expand_as(scores), scores, reduce='amax', include_self=True)
        exp_s = torch.exp(scores - max_s[row])
        sum_s = torch.zeros(N, self.heads, device=x.device)
        sum_s.scatter_add_(0, row.unsqueeze(-1).expand_as(exp_s), exp_s)
        alpha = exp_s / (sum_s[row] + 1e-8)
        msg = h_s[col] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.heads, self.out_per_head, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        return out.reshape(N, self.heads*self.out_per_head) + self.bias

class GNNModel(nn.Module):
    def __init__(self, convs, hidden_dim, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            if isinstance(conv, (PyGGCNConv, PyGGATConv)):
                x = conv(x, ei)
            else:
                x = conv(x, ei)
            x = self.bns[i](x); x = F.relu(x); x = F.dropout(x, self.dropout, self.training)
        gx = global_mean_pool(x, batch)
        return self.head(gx).squeeze(-1)

def build_models(atom_dim, hidden_dim, num_layers, dropout):
    models = {}
    heads = 4; hd = 64
    # GCN
    convs = [PyGGCNConv(atom_dim, hidden_dim)]
    for _ in range(num_layers-1): convs.append(PyGGCNConv(hidden_dim, hidden_dim))
    models['GCN'] = GNNModel(convs, hidden_dim, num_layers, dropout)
    # GAT
    convs = [PyGGATConv(atom_dim, hd//heads, heads)]
    for _ in range(num_layers-1): convs.append(PyGGATConv(hd, hd//heads, heads))
    models['GAT'] = GNNModel(convs, hd, num_layers, dropout)
    # KA-GCN
    convs = [KA_GCNConv(atom_dim, hidden_dim, KAN_HARMONICS)]
    for _ in range(num_layers-1): convs.append(KA_GCNConv(hidden_dim, hidden_dim, KAN_HARMONICS))
    models['KA-GCN'] = GNNModel(convs, hidden_dim, num_layers, dropout)
    # KA-GAT
    convs = [KA_GATConv(atom_dim, hd, heads, KAN_HARMONICS)]
    for _ in range(num_layers-1): convs.append(KA_GATConv(hd, hd, heads, KAN_HARMONICS))
    models['KA-GAT'] = GNNModel(convs, hd, num_layers, dropout)
    return models

def train_eval(model, train_ds, val_ds, test_ds):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_auc=0; best_state=None; pc=0; losses=[]; val_aucs=[]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(EPOCHS):
        model.train(); tl=0; nb=0
        for batch in train_loader:
            batch = batch.to(DEVICE); opt.zero_grad()
            pred = model(batch)
            loss = F.binary_cross_entropy_with_logits(pred, batch.y)
            loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        sched.step(); losses.append(tl/nb)
        model.eval(); preds=[]; labs=[]
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                p = model(batch); preds.extend(p.cpu().numpy()); labs.extend(batch.y.cpu().numpy())
        auc = roc_auc_score(labs, preds) if len(set(labs))>=2 else 0.5
        val_aucs.append(auc)
        if auc > best_auc: best_auc=auc; best_state=copy.deepcopy(model.state_dict()); pc=0
        else: pc+=1
        if pc>=PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); preds=[]; labs=[]
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            p = model(batch); preds.extend(p.cpu().numpy()); labs.extend(batch.y.cpu().numpy())
    test_auc = roc_auc_score(labs, preds) if len(set(labs))>=2 else 0.5
    return test_auc, best_auc, losses, val_aucs

DATASETS = ['bace','bbbp','clintox','hiv','muv']
MODEL_NAMES = ['GCN','GAT','KA-GCN','KA-GAT']

def main():
    all_results = {}; all_curves = {}; param_counts = {}
    start_time = time.time()
    for ds_name in DATASETS:
        print(f"\n=== {ds_name} ===")
        graphs, labels = load_dataset(ds_name)
        tr_idx, val_idx, te_idx = scaffold_split(graphs, labels)
        tr_ds = [graphs[i] for i in tr_idx]
        val_ds = [graphs[i] for i in val_idx]
        te_ds = [graphs[i] for i in te_idx]
        print(f"  Train={len(tr_ds)} Val={len(val_ds)} Test={len(te_ds)}")

        builders = build_models(ATOM_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
        ds_results = {}; ds_curves = {}

        for mname in MODEL_NAMES:
            run_test = []; run_val = []; run_losses = []; run_va = []
            for run in range(NUM_RUNS):
                torch.manual_seed(SEED+run); random.seed(SEED+run); np.random.seed(SEED+run)
                model = copy.deepcopy(builders[mname]).to(DEVICE)
                nparams = sum(p.numel() for p in model.parameters())
                t0 = time.time()
                test_auc, val_auc, losses, va = train_eval(model, tr_ds, val_ds, te_ds)
                elapsed = time.time()-t0
                run_test.append(test_auc); run_val.append(val_auc)
                run_losses.append(losses); run_va.append(va)
                print(f"  {mname} run{run}: test={test_auc:.4f} val={val_auc:.4f} params={nparams} t={elapsed:.1f}s")
            param_counts[mname] = nparams
            ds_results[mname] = {
                'test_auc_mean': float(np.mean(run_test)),
                'test_auc_std': float(np.std(run_test)),
                'val_auc_mean': float(np.mean(run_val)),
                'val_auc_std': float(np.std(run_val)),
                'per_run_test': [float(v) for v in run_test],
                'per_run_val': [float(v) for v in run_val],
                'num_params': nparams
            }
            ds_curves[mname] = {'losses': run_losses, 'val_aucs': run_va}
            print(f"  {mname}: test={np.mean(run_test):.4f}±{np.std(run_test):.4f}")
        all_results[ds_name] = ds_results; all_curves[ds_name] = ds_curves

    total = time.time()-start_time
    print(f"\nTotal: {total:.1f}s")
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f: json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT_DIR, 'curves.json'), 'w') as f: json.dump(all_curves, f, indent=2)
    with open(os.path.join(OUT_DIR, 'param_counts.json'), 'w') as f: json.dump(param_counts, f, indent=2)
    print("Saved!")
    return all_results, all_curves

if __name__ == '__main__':
    main()