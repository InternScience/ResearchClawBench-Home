"""
Training and evaluation script for KA-GNN vs baselines on MoleculeNet datasets.
CPU-optimized: small models, fewer molecules, fewer epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import time
import json
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kagnn_model import KAGNN, BaselineGCN, MultiFeatureEmbedding, FourierKANLayer

from rdkit import Chem
from rdkit.Chem import AllChem

# Feature definitions
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se', 'Te', 'As', 'Sn', 'other']
ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ATOM_CHIRALITIES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_atom_features(atom):
    sym = atom.GetSymbol()
    return [
        ATOM_TYPES.index(sym) if sym in ATOM_TYPES else len(ATOM_TYPES)-1,
        min(atom.GetDegree(), 6),
        max(-3, min(3, atom.GetFormalCharge())) + 3,
        ATOM_HYBRIDIZATIONS.index(atom.GetHybridization()) if atom.GetHybridization() in ATOM_HYBRIDIZATIONS else 0,
        ATOM_CHIRALITIES.index(atom.GetChiralTag()) if atom.GetChiralTag() in ATOM_CHIRALITIES else 0,
        min(atom.GetTotalNumHs(), 4),
        1 if atom.IsInRing() else 0,
        1 if atom.GetIsAromatic() else 0,
        min(int(atom.GetMass()), 200),
    ]

def get_bond_features(bond):
    bt = bond.GetBondType()
    st = bond.GetStereo()
    return [
        BOND_TYPES.index(bt) if bt in BOND_TYPES else 0,
        BOND_STEREOS.index(st) if st in BOND_STEREOS else 0,
        1 if bond.GetIsConjugated() else 0,
        1 if bond.IsInRing() else 0,
        1 if bond.GetIsAromatic() else 0,
    ]

def smiles_to_graph(smiles, add_nc=True, nc_cutoff=5.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_nc:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            add_nc = False
    
    atom_feats = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.long)
    
    ei_list, ea_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        ei_list.extend([[i,j],[j,i]])
        ea_list.extend([bf, bf])
    
    if add_nc and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        n = len(atom_feats)
        for i in range(n):
            for j in range(i+1, n):
                if mol.GetBondBetweenAtoms(i,j) is not None:
                    continue
                if np.linalg.norm(pos[i]-pos[j]) < nc_cutoff:
                    nc = [len(BOND_TYPES),0,0,0,0]
                    ei_list.extend([[i,j],[j,i]])
                    ea_list.extend([nc, nc])
    
    if not ei_list:
        ei_list, ea_list = [[0,0]], [[0,0,0,0,0]]
    
    return Data(
        x=x,
        edge_index=torch.tensor(ei_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(ea_list, dtype=torch.long),
    )

def load_dataset(csv_path, label_col=None, max_mols=None):
    df = pd.read_csv(csv_path)
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    
    if label_col and label_col in df.columns:
        labels = df[label_col].values.astype(np.float64)
        lt = 'single'
    elif 'FDA_APPROVED' in df.columns and 'CT_TOX' in df.columns:
        labels = df[['FDA_APPROVED','CT_TOX']].values.astype(np.float64)
        lt = 'multi'
    else:
        # Check for MUV format (mol_id + smiles + MUV-xxx columns)
        if 'mol_id' in df.columns and 'smiles' in df.columns:
            muv = [c for c in df.columns if c.startswith('MUV-')]
        else: muv = []
        if muv:
            labels = df[muv].replace('', np.nan).values.astype(np.float64)
            lt = 'multi'
        else:
            for c in df.columns:
                if c.lower() in ['label','activity']:
                    labels = df[c].values.astype(np.float64)
                    lt = 'single'
                    break
            else:
                raise ValueError(f"No labels in {csv_path}")
    
    # For MUV multi-task, use only the first task column with enough positive samples
    if lt == 'multi' and labels.shape[1] > 1:
        # Find the task with the most positive labels in the subset
        task_pos = []
        for t in range(labels.shape[1]):
            mask_t = ~np.isnan(labels[:, t])
            if mask_t.sum() > 0:
                task_pos.append((t, labels[mask_t, t].sum()))
        if task_pos:
            best_t = max(task_pos, key=lambda x: x[1])[0]
            labels = labels[:, best_t]
            lt = 'single'
        else:
            labels = labels[:, 0]; lt = 'single'
    
    mask = ~np.isnan(labels) if lt == 'single' else ~np.isnan(labels).any(axis=1)
    df, labels = df[mask].reset_index(drop=True), labels[mask]
    
    if max_mols and len(df) > max_mols:
        df, labels = df.iloc[:max_mols], labels[:max_mols]
    
    graphs, vi = [], []
    for i in range(len(df)):
        g = smiles_to_graph(str(df.iloc[i][smiles_col]))
        if g is not None:
            graphs.append(g); vi.append(i)
    labels = labels[vi]
    
    for i, g in enumerate(graphs):
        g.y = torch.tensor([labels[i]], dtype=torch.float32) if lt == 'single' else torch.tensor(labels[i], dtype=torch.float32)
    
    return graphs, labels, lt


class BaselineGAT(nn.Module):
    def __init__(self, atom_feature_sizes=(17,7,7,6,5,5,3,3,201), atom_embed_dim=4,
                 hidden_dim=64, num_layers=2, dropout=0.2, pool='mean', num_tasks=1, heads=4):
        super().__init__()
        self.atom_embed = MultiFeatureEmbedding(atom_feature_sizes, atom_embed_dim)
        at = atom_embed_dim * len(atom_feature_sizes)
        self.node_init = nn.Sequential(nn.Linear(at, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=dropout) for _ in range(num_layers)
        ])
        pred_in = hidden_dim * 2 if pool == 'meanmax' else hidden_dim
        self.pred_head = nn.Sequential(
            nn.Linear(pred_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_tasks))
        self.pool = pool
        self.num_tasks = num_tasks
    
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.atom_embed(x); x = self.node_init(x)
        for c in self.convs:
            x = c(x, ei); x = F.silu(x)
        if self.pool == 'meanmax':
            x = torch.cat([global_mean_pool(x,batch), global_max_pool(x,batch)], dim=-1)
        else:
            x = global_mean_pool(x, batch)
        out = self.pred_head(x)
        return out.squeeze(-1) if self.num_tasks == 1 else out


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        if out.dim() == 1:
            loss = criterion(out, batch.y.squeeze())
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            n = 0
            for t in range(out.size(1)):
                m = ~torch.isnan(batch.y[:, t])
                if m.sum() > 0:
                    loss = loss + criterion(out[m, t], batch.y[m, t])
                    n += 1
            loss = loss / max(n, 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        all_preds.append(torch.sigmoid(out).cpu())
        all_labels.append(batch.y.cpu())
    preds = torch.cat(all_preds, 0)
    labels = torch.cat(all_labels, 0)
    m = {}
    if preds.dim() == 1:
        try: m['auc'] = roc_auc_score(labels.squeeze().numpy(), preds.numpy())
        except: m['auc'] = 0.5
        pb = (preds > 0.5).float()
        m['acc'] = accuracy_score(labels.squeeze().numpy(), pb.numpy())
        m['f1'] = f1_score(labels.squeeze().numpy(), pb.numpy(), zero_division=0)
    else:
        aucs = []
        for t in range(preds.size(1)):
            mask = ~torch.isnan(labels[:, t])
            if mask.sum() > 1 and len(torch.unique(labels[mask, t])) > 1:
                try: aucs.append(roc_auc_score(labels[mask,t].numpy(), preds[mask,t].numpy()))
                except: pass
        m['auc'] = np.mean(aucs) if aucs else 0.5
        m['acc'] = 0; m['f1'] = 0
    return m


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=50, lr=0.001, wd=1e-5, patience=15):
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8, min_lr=1e-6)
    
    best_vauc, best_st, pc = 0, None, 0
    hist = {'train_loss':[], 'val_auc':[], 'test_auc':[]}
    
    for ep in range(epochs):
        tl = train_epoch(model, train_loader, opt, crit, device)
        vm = evaluate(model, val_loader if val_loader else train_loader, device)
        vauc = vm.get('auc', 0)
        tm = evaluate(model, test_loader, device)
        tauc = tm.get('auc', 0)
        hist['train_loss'].append(tl)
        hist['val_auc'].append(vauc)
        hist['test_auc'].append(tauc)
        
        if vauc > best_vauc:
            best_vauc = vauc; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}; pc = 0
        else:
            pc += 1
        if pc >= patience: break
        sch.step(vauc)
    
    if best_st: model.load_state_dict(best_st)
    return evaluate(model, test_loader, device), hist


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    datasets = [
        ('BACE', 'data/bace.csv', 'label', 800),
        ('BBBP', 'data/bbbp.csv', 'label', 800),
        ('ClinTox', 'data/clintox.csv', 'FDA_APPROVED', 800),
        ('HIV', 'data/hiv.csv', 'label', 800),
        ('MUV', 'data/muv.csv', None, 500),
    ]
    
    all_results = {}
    
    for name, path, lcol, max_mols in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        try:
            graphs, labels, lt = load_dataset(path, lcol, max_mols)
            print(f"Loaded {len(graphs)} molecules, type={lt}")
            if len(graphs) < 50: continue
            
            n = len(graphs)
            idx = np.random.RandomState(42).permutation(n)
            nt, nv = int(n*0.8), int(n*0.1)
            tl = DataLoader([graphs[i] for i in idx[:nt]], batch_size=32, shuffle=True)
            vl = DataLoader([graphs[i] for i in idx[nt:nt+nv]], batch_size=32, shuffle=False)
            sl = DataLoader([graphs[i] for i in idx[nt+nv:]], batch_size=32, shuffle=False)
            print(f"Train:{nt} Val:{nv} Test:{n-nt-nv}")
            
            ntasks = 1 if lt == 'single' else labels.shape[1]
            
            models = {
                'KA-GNN': KAGNN(hidden_dim=64, num_layers=2, kan_gridsize=4, dropout=0.2, pool='mean', num_tasks=ntasks),
                'Baseline GCN': BaselineGCN(hidden_dim=64, num_layers=2, dropout=0.2, pool='mean', num_tasks=ntasks),
                'Baseline GAT': BaselineGAT(hidden_dim=64, num_layers=2, dropout=0.2, pool='mean', num_tasks=ntasks),
            }
            
            dsr = {}
            for mn, model in models.items():
                print(f"\n  Training {mn}...")
                set_seed(42)
                model = model.to(device)
                np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Params: {np_:,}")
                t0 = time.time()
                met, hist = train_model(model, tl, vl, sl, device, epochs=50, lr=0.001, patience=15)
                tt = time.time()-t0
                dsr[mn] = {
                    'auc': float(met.get('auc',0)), 'acc': float(met.get('acc',0)),
                    'f1': float(met.get('f1',0)), 'time': float(tt), 'params': int(np_),
                    'history': {k:[float(x) for x in v] for k,v in hist.items()},
                }
                print(f"  AUC={dsr[mn]['auc']:.4f} F1={dsr[mn]['f1']:.4f} Time={tt:.1f}s")
            
            all_results[name] = dsr
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
    
    os.makedirs('outputs', exist_ok=True)
    ser = {}
    for dn, dr in all_results.items():
        ser[dn] = {}
        for mn, mr in dr.items():
            ser[dn][mn] = {k: mr[k] for k in ['auc','acc','f1','time','params','history']}
    
    with open('outputs/results.json', 'w') as f:
        json.dump(ser, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dn, dr in ser.items():
        print(f"\n{dn}:")
        for mn, mr in dr.items():
            print(f"  {mn}: AUC={mr['auc']:.4f} F1={mr['f1']:.4f} Time={mr['time']:.1f}s")
    
    return ser

if __name__ == '__main__':
    main()
