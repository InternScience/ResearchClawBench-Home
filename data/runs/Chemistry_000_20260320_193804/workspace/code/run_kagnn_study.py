import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'outputs'
REPORT_IMG_DIR = ROOT / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')


@dataclass
class GraphSample:
    x: torch.Tensor
    adj: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    num_nodes: int
    metadata: Dict


def set_plot_style():
    sns.set_theme(style='whitegrid', context='talk')


def atom_features(atom: Chem.Atom) -> List[float]:
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: [1, 0, 0, 0, 0],
        Chem.rdchem.HybridizationType.SP2: [0, 1, 0, 0, 0],
        Chem.rdchem.HybridizationType.SP3: [0, 0, 1, 0, 0],
        Chem.rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],
        Chem.rdchem.HybridizationType.SP3D2: [0, 0, 0, 0, 1],
    }.get(hyb, [0, 0, 0, 0, 0])
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 8.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs() / 8.0,
        float(atom.GetIsAromatic()),
        atom.GetMass() / 250.0,
        float(atom.IsInRing()),
        atom.GetImplicitValence() / 8.0,
        atom.GetExplicitValence() / 8.0,
        float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
    ] + hyb_map


def bond_weight(bond: Chem.Bond) -> float:
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1.0
    if bt == Chem.rdchem.BondType.DOUBLE:
        return 1.5
    if bt == Chem.rdchem.BondType.TRIPLE:
        return 2.0
    if bt == Chem.rdchem.BondType.AROMATIC:
        return 1.25
    return 1.0


def smiles_to_graph(smiles: str, y: np.ndarray, dist_threshold: int = 2) -> GraphSample:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    feats = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    adj = np.eye(n, dtype=np.float32)
    # Covalent edges
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        w = bond_weight(bond)
        adj[i, j] = max(adj[i, j], w)
        adj[j, i] = max(adj[j, i], w)
    # Simple non-covalent/proximity edges using graph distance in bond graph
    dmat = Chem.GetDistanceMatrix(mol)
    for i in range(n):
        for j in range(i + 1, n):
            d = dmat[i, j]
            if 1 < d <= dist_threshold and adj[i, j] == 0:
                ai = mol.GetAtomWithIdx(i)
                aj = mol.GetAtomWithIdx(j)
                hetero_bonus = 0.15 if (ai.GetAtomicNum() in [7, 8, 9, 15, 16, 17] or aj.GetAtomicNum() in [7, 8, 9, 15, 16, 17]) else 0.0
                adj[i, j] = adj[j, i] = 0.2 + hetero_bonus
    mask = np.ones(n, dtype=np.float32)
    metadata = {
        'smiles': smiles,
        'num_atoms': n,
        'mw': float(Descriptors.MolWt(mol)),
        'logp': float(Descriptors.MolLogP(mol)),
    }
    return GraphSample(
        x=torch.tensor(feats),
        adj=torch.tensor(adj),
        y=torch.tensor(y.astype(np.float32)),
        mask=torch.tensor(mask),
        num_nodes=n,
        metadata=metadata,
    )


class MoleculeDataset(Dataset):
    def __init__(self, samples: List[GraphSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_graphs(batch: List[GraphSample]):
    max_nodes = max(s.num_nodes for s in batch)
    feat_dim = batch[0].x.shape[1]
    task_dim = batch[0].y.shape[0]
    xs = torch.zeros(len(batch), max_nodes, feat_dim)
    adjs = torch.zeros(len(batch), max_nodes, max_nodes)
    masks = torch.zeros(len(batch), max_nodes)
    ys = torch.zeros(len(batch), task_dim)
    ymask = torch.zeros(len(batch), task_dim)
    metas = []
    for i, s in enumerate(batch):
        n = s.num_nodes
        xs[i, :n] = s.x
        adjs[i, :n, :n] = s.adj
        masks[i, :n] = s.mask
        ys[i] = torch.nan_to_num(s.y, nan=0.0)
        ymask[i] = (~torch.isnan(s.y)).float()
        metas.append(s.metadata)
    return xs.to(DEVICE), adjs.to(DEVICE), masks.to(DEVICE), ys.to(DEVICE), ymask.to(DEVICE), metas


class FourierKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_freq: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_freq = n_freq
        self.base = nn.Linear(in_dim, out_dim)
        self.cos_coeff = nn.Parameter(torch.randn(out_dim, in_dim, n_freq) * 0.05)
        self.sin_coeff = nn.Parameter(torch.randn(out_dim, in_dim, n_freq) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: [..., in_dim]
        out = self.base(x)
        for k in range(1, self.n_freq + 1):
            out = out + F.linear(torch.cos(k * x), self.cos_coeff[:, :, k - 1])
            out = out + F.linear(torch.sin(k * x), self.sin_coeff[:, :, k - 1])
        return out + self.bias


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int, transform: str = 'mlp', n_freq: int = 6):
        super().__init__()
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        if transform == 'kan':
            self.update = FourierKANLayer(hidden_dim * 2, hidden_dim, n_freq=n_freq)
        else:
            self.update = MLPBlock(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, mask):
        deg = adj.sum(-1, keepdim=True).clamp_min(1e-6)
        agg = torch.bmm(adj, self.msg_lin(x)) / deg
        h = torch.cat([self.self_lin(x), agg], dim=-1)
        h = self.update(h)
        h = F.relu(self.norm(h))
        return h * mask.unsqueeze(-1)


class GraphModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, transform: str = 'mlp', n_freq: int = 6):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConvLayer(hidden_dim, transform=transform, n_freq=n_freq) for _ in range(depth)])
        if transform == 'kan':
            self.readout = FourierKANLayer(hidden_dim * 2, hidden_dim, n_freq=n_freq)
        else:
            self.readout = MLPBlock(hidden_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj, mask):
        h = F.relu(self.embed(x)) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = h + layer(h, adj, mask)
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        mean_pool = h.sum(1) / denom
        max_pool = (h + (1 - mask).unsqueeze(-1) * -1e9).max(1).values
        g = torch.cat([mean_pool, max_pool], dim=-1)
        g = F.relu(self.readout(g))
        return self.head(g)


class DescriptorMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(name: str, max_samples: int = None) -> Tuple[List[GraphSample], List[str]]:
    path = DATA_DIR / f'{name}.csv'
    df = pd.read_csv(path)
    if name == 'clintox':
        df = df[['smiles', 'FDA_APPROVED', 'CT_TOX']].copy()
        label_cols = ['FDA_APPROVED', 'CT_TOX']
    elif name == 'muv':
        label_cols = [c for c in df.columns if c.startswith('MUV-')]
        keep = label_cols + ['smiles']
        df = df[keep].copy()
    else:
        label_cols = ['label']
        df = df[['smiles'] + label_cols].copy()
    df = df.dropna(subset=['smiles']).reset_index(drop=True)
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()
    samples = []
    bad = 0
    for _, row in df.iterrows():
        y = row[label_cols].astype(np.float32).values
        try:
            samples.append(smiles_to_graph(row['smiles'], y))
        except Exception:
            bad += 1
    return samples, label_cols


def stratify_labels(samples: List[GraphSample]) -> np.ndarray:
    y = np.array([0 if torch.isnan(s.y[0]) else int(s.y[0].item() > 0.5) for s in samples])
    if len(np.unique(y)) < 2:
        y = None
    return y


def split_samples(samples: List[GraphSample]):
    idx = np.arange(len(samples))
    strat = stratify_labels(samples)
    try:
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
    except Exception:
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=None)
    strat_tr = strat[tr_idx] if strat is not None and len(set(strat[tr_idx])) > 1 else None
    try:
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, random_state=SEED, stratify=strat_tr)
    except Exception:
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, random_state=SEED, stratify=None)
    return [samples[i] for i in tr_idx], [samples[i] for i in va_idx], [samples[i] for i in te_idx]


def bce_loss_logits(logits, targets, target_mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = loss * target_mask
    return loss.sum() / target_mask.sum().clamp_min(1.0)


def evaluate_graph_model(model, loader):
    model.eval()
    ys, ps, ms = [], [], []
    with torch.no_grad():
        for x, adj, mask, y, ymask, _ in loader:
            logits = model(x, adj, mask)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            ms.append(ymask.cpu().numpy())
    return compute_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ms))


def compute_metrics(y_true, y_prob, y_mask):
    roc_list = []
    ap_list = []
    for t in range(y_true.shape[1]):
        mask = y_mask[:, t] > 0.5
        if mask.sum() < 2:
            continue
        yt = y_true[mask, t]
        yp = y_prob[mask, t]
        if len(np.unique(yt)) < 2:
            continue
        roc_list.append(roc_auc_score(yt, yp))
        ap_list.append(average_precision_score(yt, yp))
    return {
        'roc_auc_macro': float(np.mean(roc_list)) if roc_list else float('nan'),
        'avg_precision_macro': float(np.mean(ap_list)) if ap_list else float('nan'),
        'num_valid_tasks': int(len(roc_list)),
    }


def train_graph_model(model, train_loader, val_loader, lr=1e-3, epochs=12):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = -1e9
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, adj, mask, y, ymask, _ in train_loader:
            opt.zero_grad()
            logits = model(x, adj, mask)
            loss = bce_loss_logits(logits, y, ymask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        val_metrics = evaluate_graph_model(model, val_loader)
        score = val_metrics['roc_auc_macro'] if not math.isnan(val_metrics['roc_auc_macro']) else val_metrics['avg_precision_macro']
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), **val_metrics})
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def compute_descriptors(samples: List[GraphSample]) -> np.ndarray:
    rows = []
    for s in samples:
        smi = s.metadata['smiles']
        mol = Chem.MolFromSmiles(smi)
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
        ])
    arr = np.array(rows, dtype=np.float32)
    mu = arr.mean(0, keepdims=True)
    sd = arr.std(0, keepdims=True) + 1e-6
    return (arr - mu) / sd


def train_descriptor_baseline(train_s, val_s, test_s):
    xtr = torch.tensor(compute_descriptors(train_s), dtype=torch.float32)
    xva = torch.tensor(compute_descriptors(val_s), dtype=torch.float32)
    xte = torch.tensor(compute_descriptors(test_s), dtype=torch.float32)
    ytr = torch.stack([torch.nan_to_num(s.y, nan=0.0) for s in train_s])
    yva = torch.stack([torch.nan_to_num(s.y, nan=0.0) for s in val_s])
    yte = torch.stack([torch.nan_to_num(s.y, nan=0.0) for s in test_s])
    mtr = torch.stack([(~torch.isnan(s.y)).float() for s in train_s])
    mva = torch.stack([(~torch.isnan(s.y)).float() for s in val_s])
    mte = torch.stack([(~torch.isnan(s.y)).float() for s in test_s])
    model = DescriptorMLP(xtr.shape[1], ytr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_val = -1e9
    for _ in range(15):
        model.train()
        opt.zero_grad()
        logits = model(xtr)
        loss = bce_loss_logits(logits, ytr, mtr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(xva)).numpy()
        metr = compute_metrics(yva.numpy(), vp, mva.numpy())
        score = metr['roc_auc_macro'] if not math.isnan(metr['roc_auc_macro']) else metr['avg_precision_macro']
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        te_prob = torch.sigmoid(model(xte)).numpy()
    metrics = compute_metrics(yte.numpy(), te_prob, mte.numpy())
    return metrics, te_prob, yte.numpy(), mte.numpy()


def dataset_overview(dataset_specs):
    rows = []
    for name, max_n in dataset_specs:
        samples, label_cols = load_dataset(name, max_samples=max_n)
        y = np.stack([s.y.numpy() for s in samples])
        valid = ~np.isnan(y)
        positives = np.nansum(y, axis=0)
        totals = valid.sum(axis=0)
        rows.append({
            'dataset': name,
            'molecules': len(samples),
            'tasks': len(label_cols),
            'avg_atoms': float(np.mean([s.num_nodes for s in samples])),
            'median_atoms': float(np.median([s.num_nodes for s in samples])),
            'positive_rate_mean': float(np.nanmean(positives / np.maximum(totals, 1))),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'dataset_overview.csv', index=False)
    return df


def plot_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(data=df, x='dataset', y='molecules', ax=axes[0], color='steelblue')
    axes[0].set_title('Dataset size')
    axes[0].tick_params(axis='x', rotation=30)
    sns.barplot(data=df, x='dataset', y='avg_atoms', ax=axes[1], color='darkorange')
    axes[1].set_title('Average atoms per molecule')
    axes[1].tick_params(axis='x', rotation=30)
    sns.barplot(data=df, x='dataset', y='positive_rate_mean', ax=axes[2], color='seagreen')
    axes[2].set_title('Mean positive rate')
    axes[2].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'dataset_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_results(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x='dataset', y='roc_auc_macro', hue='model', ax=ax)
    ax.set_title('Test ROC-AUC comparison across datasets')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'main_results_rocauc.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x='dataset', y='avg_precision_macro', hue='model', ax=ax)
    ax.set_title('Test Average Precision comparison across datasets')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'main_results_ap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history_map):
    rows = []
    for key, hist in history_map.items():
        dataset, model = key.split('::')
        for r in hist:
            rows.append({'dataset': dataset, 'model': model, 'epoch': r['epoch'], 'train_loss': r['train_loss'], 'val_roc_auc': r['roc_auc_macro']})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=df, x='epoch', y='train_loss', hue='model', style='dataset', ax=axes[0])
    axes[0].set_title('Training loss trajectories')
    sns.lineplot(data=df, x='epoch', y='val_roc_auc', hue='model', style='dataset', ax=axes[1])
    axes[1].set_title('Validation ROC-AUC trajectories')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_efficiency(eff_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=eff_df, x='train_seconds', y='roc_auc_macro', hue='model', style='dataset', s=120, ax=ax)
    ax.set_title('Accuracy-efficiency trade-off')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'efficiency_tradeoff.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_scatter(pred_records):
    df = pd.DataFrame(pred_records)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='true_label', y='pred_prob', hue='model', ax=ax)
    ax.set_title('Prediction separation on the BACE test set')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'prediction_separation_bace.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_study():
    set_plot_style()
    dataset_specs = [
        ('bace', None),
        ('bbbp', None),
        ('clintox', None),
        ('hiv', 3000),
        ('muv', 4000),
    ]
    overview = dataset_overview(dataset_specs)
    plot_dataset_overview(overview)

    all_results = []
    history_map = {}
    eff_rows = []
    pred_records = []

    for name, max_n in dataset_specs:
        print('Running dataset', name)
        samples, label_cols = load_dataset(name, max_samples=max_n)
        train_s, val_s, test_s = split_samples(samples)
        train_loader = DataLoader(MoleculeDataset(train_s), batch_size=48, shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(MoleculeDataset(val_s), batch_size=96, shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(MoleculeDataset(test_s), batch_size=96, shuffle=False, collate_fn=collate_graphs)
        in_dim = train_s[0].x.shape[1]
        out_dim = train_s[0].y.shape[0]

        for model_name, transform in [('GNN-MLP', 'mlp'), ('KA-GNN', 'kan')]:
            hidden_dim = 48 if name in ['hiv', 'muv'] else 64
            depth = 2 if name in ['hiv', 'muv'] else 3
            freqs = 4 if name in ['hiv', 'muv'] else 5
            model = GraphModel(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=depth, transform=transform, n_freq=freqs).to(DEVICE)
            start = pd.Timestamp.now()
            hist = train_graph_model(model, train_loader, val_loader, epochs=5 if name in ['hiv', 'muv'] else 8)
            elapsed = (pd.Timestamp.now() - start).total_seconds()
            metrics = evaluate_graph_model(model, test_loader)
            metrics.update({'dataset': name, 'model': model_name, 'train_seconds': elapsed})
            all_results.append(metrics)
            history_map[f'{name}::{model_name}'] = hist
            eff_rows.append({'dataset': name, 'model': model_name, 'train_seconds': elapsed, 'roc_auc_macro': metrics['roc_auc_macro']})

            if name == 'bace':
                model.eval()
                with torch.no_grad():
                    for x, adj, mask, y, ymask, _ in test_loader:
                        probs = torch.sigmoid(model(x, adj, mask)).cpu().numpy()[:, 0]
                        truths = y.cpu().numpy()[:, 0]
                        valid = ymask.cpu().numpy()[:, 0] > 0.5
                        for p, t, m in zip(probs, truths, valid):
                            if m:
                                pred_records.append({'model': model_name, 'pred_prob': float(p), 'true_label': int(t)})

        base_metrics, _, _, _ = train_descriptor_baseline(train_s, val_s, test_s)
        base_metrics.update({'dataset': name, 'model': 'Descriptor-MLP', 'train_seconds': 0.0})
        all_results.append(base_metrics)
        eff_rows.append({'dataset': name, 'model': 'Descriptor-MLP', 'train_seconds': 0.0, 'roc_auc_macro': base_metrics['roc_auc_macro']})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / 'results_summary.csv', index=False)
    with open(OUT_DIR / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history_map, f, indent=2)
    pd.DataFrame(eff_rows).to_csv(OUT_DIR / 'efficiency_summary.csv', index=False)
    pd.DataFrame(pred_records).to_csv(OUT_DIR / 'bace_prediction_records.csv', index=False)

    plot_results(results_df)
    plot_training_curves(history_map)
    plot_efficiency(pd.DataFrame(eff_rows))
    plot_prediction_scatter(pred_records)

    pivot = results_df.pivot(index='dataset', columns='model', values='roc_auc_macro')
    if 'KA-GNN' in pivot.columns and 'GNN-MLP' in pivot.columns:
        improv = (pivot['KA-GNN'] - pivot['GNN-MLP']).reset_index()
        improv.columns = ['dataset', 'roc_auc_gain']
        improv.to_csv(OUT_DIR / 'kagnn_improvement_vs_gnnmlp.csv', index=False)

    summary = {
        'seed': SEED,
        'datasets': [d for d, _ in dataset_specs],
        'device': str(DEVICE),
    }
    with open(OUT_DIR / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    run_study()
