import os, json, math, random, argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GINEConv

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ATOM_LIST = list(range(1, 119))
HYBRID_LIST = [
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


def one_hot(x, values):
    out = [0] * (len(values) + 1)
    try:
        out[values.index(x)] = 1
    except ValueError:
        out[-1] = 1
    return out


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_LIST[:20] + [26, 35, 53])
    feats += one_hot(atom.GetTotalDegree(), [0,1,2,3,4,5])
    feats += one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2])
    feats += one_hot(int(atom.GetChiralTag()), [0,1,2,3])
    feats += one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])
    feats += one_hot(atom.GetHybridization(), HYBRID_LIST)
    feats += [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    feats += [atom.GetMass() * 0.01, atom.GetImplicitValence() / 6.0]
    return torch.tensor(feats, dtype=torch.float)


def bond_features_from_geom(dist, kind='noncovalent'):
    kind_map = {'covalent':0, 'noncovalent':1, 'self':2}
    feats = [0]*3
    feats[kind_map[kind]] = 1
    feats += [dist/5.0, math.exp(-dist), 1.0/(dist+1e-6)]
    feats += [0,0,0,0,0,0,0]
    return torch.tensor(feats, dtype=torch.float)


def bond_features(bond, dist=1.5, kind='covalent'):
    bt = bond.GetBondType() if bond is not None else None
    feats = [0]*3
    feats[0 if kind=='covalent' else 1 if kind=='noncovalent' else 2] = 1
    feats += [dist/5.0, math.exp(-dist), 1.0/(dist+1e-6)]
    feats += one_hot(bt, BOND_TYPES)
    feats += [int(bond.GetIsConjugated()) if bond is not None else 0, int(bond.IsInRing()) if bond is not None else 0]
    return torch.tensor(feats, dtype=torch.float)


def build_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=SEED)
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    except Exception:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
    return mol


def mol_to_graph(smiles, y, use_noncovalent=True, nc_cutoff=4.5):
    mol = build_mol(smiles)
    if mol is None:
        return None
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edges, eattr = [], []
    positions = []
    for i in range(n):
        p = conf.GetAtomPosition(i)
        positions.append(np.array([p.x,p.y,p.z], dtype=float))
    positions = np.vstack(positions)
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = float(np.linalg.norm(positions[i]-positions[j]))
        bf = bond_features(b, d, 'covalent')
        edges += [[i,j],[j,i]]
        eattr += [bf,bf]
    if use_noncovalent:
        bonded = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) in bonded:
                    continue
                d = float(np.linalg.norm(positions[i]-positions[j]))
                if d <= nc_cutoff:
                    bf = bond_features_from_geom(d, 'noncovalent')
                    edges += [[i,j],[j,i]]
                    eattr += [bf,bf]
    if len(edges) == 0:
        edges = [[0,0]]
        eattr = [bond_features_from_geom(0.0, 'self')]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(eattr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(y, dtype=torch.float).view(1,-1), smiles=smiles)
    return data


class FourierKAN(nn.Module):
    def __init__(self, dim, hidden_mult=2, n_freq=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_freq = n_freq
        self.freq = nn.Parameter(torch.arange(1, n_freq+1, dtype=torch.float).view(1,1,-1), requires_grad=False)
        self.lin1 = nn.Linear(dim * n_freq * 2, dim * hidden_mult)
        self.lin2 = nn.Linear(dim * hidden_mult, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        z = x.unsqueeze(-1) * self.freq.to(x.device) * math.pi
        phi = torch.cat([torch.sin(z), torch.cos(z)], dim=-1).reshape(x.size(0), -1)
        out = self.lin2(F.silu(self.lin1(phi)))
        out = self.dropout(out)
        return self.norm(x + out)


class EdgeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def forward(self, x):
        return self.net(x)


class BaseGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, layers=3, dropout=0.1, use_kan=False):
        super().__init__()
        self.node_in = nn.Linear(in_dim, hidden_dim)
        self.edge_enc = EdgeMLP(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.post = nn.ModuleList()
        self.use_kan = use_kan
        for _ in range(layers):
            nn_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(nn_mlp, edge_dim=hidden_dim))
            self.post.append(FourierKAN(hidden_dim, n_freq=8, dropout=dropout) if use_kan else nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout
    def forward(self, data):
        x = self.node_in(data.x)
        e = self.edge_enc(data.edge_attr)
        for conv, post in zip(self.convs, self.post):
            x = conv(x, data.edge_index, e)
            x = post(x)
        g = global_mean_pool(x, data.batch)
        return self.head(g)


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    n = len(dataset)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    n_test = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED))


def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(batch.y.cpu().numpy())
            ps.append(probs)
    y = np.vstack(ys)
    p = np.vstack(ps)
    metrics = {}
    aucs=[]; aps=[]
    for j in range(y.shape[1]):
        mask = ~np.isnan(y[:,j])
        if mask.sum() < 2 or len(np.unique(y[mask,j])) < 2:
            continue
        aucs.append(roc_auc_score(y[mask,j], p[mask,j]))
        aps.append(average_precision_score(y[mask,j], p[mask,j]))
    metrics['roc_auc'] = float(np.mean(aucs)) if aucs else float('nan')
    metrics['pr_auc'] = float(np.mean(aps)) if aps else float('nan')
    return metrics, y, p


def train_one(model, train_loader, val_loader, device, epochs=15, lr=1e-3, pos_weight=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = None
    hist = []
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, dtype=torch.float, device=device)
    else:
        pw = None
    for epoch in range(1, epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            y = batch.y
            mask = ~torch.isnan(y)
            if pw is None:
                loss_mat = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y, nan=0.0), reduction='none')
            else:
                loss_mat = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y, nan=0.0), reduction='none', pos_weight=pw)
            loss = (loss_mat * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            loss.backward(); opt.step(); losses.append(loss.item())
        val_metrics,_,_ = eval_model(model, val_loader, device)
        rec = {'epoch':epoch, 'train_loss':float(np.mean(losses)), **val_metrics}
        hist.append(rec)
        score = np.nan_to_num(val_metrics['roc_auc'], nan=-1)
        if best is None or score > best['score']:
            best = {'score':score, 'state':{k:v.cpu().clone() for k,v in model.state_dict().items()}, 'epoch':epoch}
    model.load_state_dict(best['state'])
    return hist, best


def load_dataset(name, max_samples=None, use_noncovalent=True):
    path = os.path.join('data', f'{name}.csv')
    df = pd.read_csv(path)
    if name == 'clintox':
        label_cols = ['FDA_APPROVED','CT_TOX']
    elif name == 'muv':
        label_cols = [c for c in df.columns if c.startswith('MUV-')]
    else:
        label_cols = ['label']
    keep = ['smiles'] + label_cols
    df = df[keep].copy()
    if max_samples is not None and len(df) > max_samples:
        if name == 'muv':
            df = df.sample(max_samples, random_state=SEED)
        else:
            df = df.sample(min(max_samples, len(df)), random_state=SEED)
    dataset=[]
    for _, row in df.iterrows():
        y = [float(row[c]) if pd.notna(row[c]) else float('nan') for c in label_cols]
        g = mol_to_graph(row['smiles'], y, use_noncovalent=use_noncovalent)
        if g is not None:
            dataset.append(g)
    return dataset, label_cols


def compute_pos_weight(dataset, out_dim):
    y = torch.cat([d.y for d in dataset], dim=0).numpy()
    ws=[]
    for j in range(out_dim):
        col = y[:,j]
        mask = ~np.isnan(col)
        pos = (col[mask]==1).sum(); neg = (col[mask]==0).sum()
        ws.append(float(neg/max(pos,1)))
    return ws


def experiment(dataset_name, max_samples, epochs, hidden_dim, layers, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, label_cols = load_dataset(dataset_name, max_samples=max_samples, use_noncovalent=True)
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    in_dim = dataset[0].x.shape[1]; edge_dim = dataset[0].edge_attr.shape[1]; out_dim = dataset[0].y.shape[1]
    pos_weight = compute_pos_weight(train_ds, out_dim)
    results=[]
    for model_name, use_kan in [('GINE-MLP', False), ('KA-GNN', True)]:
        model = BaseGNN(in_dim, edge_dim, hidden_dim, out_dim, layers=layers, use_kan=use_kan)
        hist, best = train_one(model, train_loader, val_loader, device, epochs=epochs, pos_weight=pos_weight)
        test_metrics, y, p = eval_model(model, test_loader, device)
        results.append({'dataset':dataset_name,'model':model_name,'label_cols':label_cols,'history':hist,'best_epoch':best['epoch'],'test_metrics':test_metrics,'n_graphs':len(dataset),'out_dim':out_dim})
        os.makedirs('outputs/models', exist_ok=True)
        torch.save(model.state_dict(), f'outputs/models/{dataset_name}_{model_name}.pt')
        pd.DataFrame(hist).to_csv(f'outputs/{dataset_name}_{model_name}_history.csv', index=False)
        np.savez(f'outputs/{dataset_name}_{model_name}_preds.npz', y=y, p=p)
    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/{dataset_name}_results.json','w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()
    experiment(args.dataset, args.max_samples, args.epochs, args.hidden_dim, args.layers, args.batch_size)
